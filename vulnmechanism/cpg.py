from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .process import run_process


class CPGError(RuntimeError):
    pass


class TargetMethodError(CPGError):
    pass


class CPGQualityError(CPGError):
    pass


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    label: str
    code: str


@dataclass(frozen=True)
class GraphEdge:
    kind: str
    source: str
    target: str


@dataclass(frozen=True)
class FunctionGraph:
    function: str
    nodes: dict[str, GraphNode]
    edges: tuple[GraphEdge, ...]
    quality: dict | None = None


_INVALID_METHOD_NAMES = {"if", "for", "while", "switch", "catch", "sizeof", "do"}


def _find_executable(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise CPGError(f"required Joern executable not found under {root}: {', '.join(names)}")


def _environment(java_home: str | Path) -> dict[str, str]:
    java_home = Path(java_home).expanduser()
    java = java_home / "bin" / "java"
    if not java.is_file() or not os.access(java, os.X_OK):
        raise CPGError(f"Java executable not found at {java}")
    env = os.environ.copy()
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = str(java.parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("JAVA_OPTS", "-Xmx4g -XX:ActiveProcessorCount=4")
    return env


def read_neo4jcsv(directory: Path):
    """Read Joern's streaming Neo4j CSV export."""
    csv.field_size_limit(sys.maxsize)
    nodes, edges = {}, []
    for header_path in sorted(directory.glob("nodes_*_header.csv")):
        with header_path.open(encoding="utf-8", newline="") as handle:
            columns = next(csv.reader(handle))
        if columns[:2] != [":ID", ":LABEL"]:
            raise CPGError(f"unexpected Joern node CSV header: {header_path.name}")
        data_path = header_path.with_name(header_path.name.replace("_header.csv", "_data.csv"))
        with data_path.open(encoding="utf-8", newline="") as handle:
            for values in csv.reader(handle):
                if len(values) != len(columns):
                    raise CPGError(f"malformed node row in {header_path.name}")
                node = {"kind": values[1]}
                for field, value in zip(columns[2:], values[2:]):
                    if not value:
                        continue
                    name, _, kind = field.partition(":")
                    if kind in {"int", "long"} or name in {
                        "LINE_NUMBER", "LINE_NUMBER_END", "COLUMN_NUMBER", "COLUMN_NUMBER_END",
                        "OFFSET", "OFFSET_END", "ORDER",
                    }:
                        node[name] = int(value)
                    elif kind == "boolean" or name == "IS_EXTERNAL":
                        if value not in {"true", "false"}:
                            raise CPGError(f"invalid boolean {value!r} in {header_path.name}")
                        node[name] = value == "true"
                    else:
                        node[name] = value.replace("\\\\", "\\")
                nodes[values[0]] = node

    for kind in ("AST", "CFG", "CDG", "REACHING_DEF"):
        header_path = directory / f"edges_{kind}_header.csv"
        if not header_path.exists():
            continue
        with header_path.open(encoding="utf-8", newline="") as handle:
            columns = next(csv.reader(handle))
        if columns[:3] != [":START_ID", ":END_ID", ":TYPE"]:
            raise CPGError(f"unexpected Joern edge CSV header: {header_path.name}")
        with directory.joinpath(f"edges_{kind}_data.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) != len(columns) or row[2] != kind:
                    raise CPGError(f"malformed edge row in {header_path.name}")
                edges.append((kind, row[0], row[1]))
    if not nodes:
        raise CPGError("Joern export contains no nodes")
    return nodes, edges


def _methods_for_file(nodes, filename: str, *, include_global: bool = False):
    result = []
    for key, node in nodes.items():
        if node.get("kind") != "METHOD" or node.get("IS_EXTERNAL"):
            continue
        name = str(node.get("NAME", ""))
        if name in _INVALID_METHOD_NAMES:
            continue
        if not include_global and name == "<global>":
            continue
        if include_global and name != "<global>":
            continue
        if Path(str(node.get("FILENAME", ""))).name != filename:
            continue
        result.append((key, node))
    return result


def _unqualified(name: str) -> str:
    return name.rsplit("::", 1)[-1].strip()


def resolve_target_graph(
    nodes,
    edges,
    *,
    filename: str,
    source: str,
    start_line: int = 1,
    function_hint: str = "",
) -> FunctionGraph:
    """Resolve a function-level CPG using the input file as the primary sample identity.

    For standalone function files, a unique source METHOD is preferred. If Joern
    keeps the parsed code under one or more synthetic <global> METHOD roots, the
    whole file graph is accepted when structural checks pass. This mirrors common
    function-level CPG pipelines while retaining explicit quality auditing.
    """
    from .syntax import source_tokens

    tokens = source_tokens(source)
    if not tokens or "{" not in tokens or "}" not in tokens:
        raise TargetMethodError("source has no complete function body")

    file_nodes = [
        node for node in nodes.values()
        if node.get("kind") == "FILE" and Path(str(node.get("NAME", ""))).name == filename
    ]
    contents = str(file_nodes[0].get("CONTENT", "")) if len(file_nodes) == 1 else ""
    standalone = start_line == 1 and contents.rstrip() == source.rstrip()
    end_line = start_line + len(source.rstrip().splitlines()) - 1
    hint = _unqualified(function_hint) if function_hint else ""

    methods = _methods_for_file(nodes, filename)
    available = [(n.get("NAME"), n.get("LINE_NUMBER"), n.get("LINE_NUMBER_END")) for _, n in methods]
    selected = methods
    resolution_mode = "standalone_file_unique_method" if standalone else "full_file_line_range"

    if not standalone:
        selected = [
            (key, node)
            for key, node in selected
            if isinstance(node.get("LINE_NUMBER"), int)
            and isinstance(node.get("LINE_NUMBER_END"), int)
            and node["LINE_NUMBER"] <= start_line <= node["LINE_NUMBER_END"]
            and node["LINE_NUMBER_END"] >= end_line
        ]

    if len(selected) != 1 and hint:
        by_hint = [
            (key, node) for key, node in selected
            if _unqualified(str(node.get("NAME", ""))) == hint
        ]
        if len(by_hint) == 1:
            selected = by_hint
            resolution_mode += "+name_hint"

    whole_file_fallback = False
    if len(selected) == 0 and standalone:
        globals_ = _methods_for_file(nodes, filename, include_global=True)
        if globals_:
            root_ids = [key for key, _ in globals_]
            method = globals_[0][1]
            whole_file_fallback = True
            resolution_mode = "standalone_global_file_graph"
        else:
            raise TargetMethodError(
                f"no source METHOD or <global> graph for {filename}; hint={function_hint!r}; methods={available[:30]}"
            )
    elif len(selected) == 1:
        root_ids = [selected[0][0]]
        method = selected[0][1]
    else:
        raise TargetMethodError(
            f"expected one source-defined target method in {filename}; found {len(selected)}; "
            f"hint={function_hint!r}; methods={available[:30]}"
        )

    ast = defaultdict(list)
    for kind, source_id, target_id in edges:
        if kind == "AST":
            ast[source_id].append(target_id)

    owned, pending = set(), list(root_ids)
    root_set = set(root_ids)
    while pending:
        key = pending.pop()
        if key in owned:
            continue
        node = nodes.get(key)
        if node is None:
            continue
        if not whole_file_fallback and key not in root_set and node.get("kind") == "METHOD":
            continue
        owned.add(key)
        pending.extend(ast.get(key, ()))

    encoded = contents.encode("utf-16-le")
    line_starts = [0]
    for line in contents.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line.encode("utf-16-le")) // 2)

    def physical_span(node):
        line, last = node.get("LINE_NUMBER"), node.get("LINE_NUMBER_END")
        column, end_column = node.get("COLUMN_NUMBER"), node.get("COLUMN_NUMBER_END")
        if all(isinstance(value, int) for value in (line, last, column, end_column)):
            if 1 <= line <= last < len(line_starts) and column >= 1 and end_column >= 1:
                begin = line_starts[line - 1] + column - 1
                finish = min(line_starts[last - 1] + end_column, line_starts[last])
                if 0 <= begin < finish <= len(encoded) // 2:
                    return begin, finish
        begin, finish = node.get("OFFSET"), node.get("OFFSET_END")
        if isinstance(begin, int) and isinstance(finish, int) and 0 <= begin < finish <= len(encoded) // 2:
            return begin, finish
        return None

    def physical_code(node):
        span = physical_span(node)
        if span is None:
            return None
        return encoded[2 * span[0] : 2 * span[1]].decode("utf-16-le")

    graph_nodes = {}
    for key in sorted(owned, key=lambda v: (0, int(v)) if str(v).isdigit() else (1, str(v))):
        node = nodes[key]
        label = node.get("NAME", "CALL") if node.get("kind") == "CALL" else {
            "METHOD_PARAMETER_IN": "PARAM"
        }.get(node.get("kind"), node.get("kind"))
        code = str(node.get("CODE", ""))
        if node.get("kind") not in {"METHOD", "BLOCK", "CONTROL_STRUCTURE"} and len(code) == 1000 and code.endswith("..."):
            restored = physical_code(node)
            if restored is None or not restored.startswith(code[:-3]):
                raise CPGQualityError(
                    f"truncated {node.get('kind')} node code without verified source coordinates"
                )
            code = restored
        if node.get("kind") == "METHOD":
            code = str(node.get("NAME", ""))
        graph_nodes[str(key)] = GraphNode(str(key), str(label), code)

    edge_kinds = {"AST": "AST", "CFG": "CFG", "CDG": "CDG", "REACHING_DEF": "DDG"}
    graph_edges = tuple(
        GraphEdge(edge_kinds[kind], str(source_id), str(target_id))
        for kind, source_id, target_id in edges
        if kind in edge_kinds and source_id in owned and target_id in owned
    )
    counts = Counter(edge.kind for edge in graph_edges)
    unknown = [key for key in owned if nodes[key].get("kind") == "UNKNOWN"]

    reasons = []
    warnings = []
    if not counts["AST"] or not counts["CFG"]:
        reasons.append("missing_ast_or_cfg")
    if unknown:
        reasons.append("unknown_ast_nodes")

    native_name = str(method.get("NAME", ""))
    if not whole_file_fallback:
        body_node = next(
            (nodes[key] for key in ast.get(root_ids[0], ()) if nodes[key].get("kind") == "BLOCK"),
            None,
        )
        body_tokens = source_tokens(str(body_node.get("CODE", ""))) if body_node else ()
        source_body_tokens = tokens[tokens.index("{") :] if "{" in tokens else ()
        if not body_tokens or body_tokens == ("<", "empty", ">"):
            warnings.append("missing_body")
        elif body_tokens == ("{", "}") and len(source_body_tokens) > 2:
            warnings.append("body_content_missing")

        header = source.split("{", 1)[0]
        if re.fullmatch(r"[A-Z][A-Z_0-9]*", native_name):
            sole_macro = re.match(r"\s*" + re.escape(native_name) + r"\s*\(", header)
            name_macro = re.search(r"\b" + re.escape(native_name) + r"\s*\([^()]*\)\s*\(", header)
            if sole_macro or name_macro:
                reasons.append("unresolved_macro_signature")
    else:
        warnings.append("no_source_method_global_file_graph")

    exact_occurrences = [m.start() for m in re.finditer(re.escape(source), contents)] if contents else []
    source_occurrence_exact = any(contents[:p].count("\n") + 1 == start_line for p in exact_occurrences)
    method_tokens = source_tokens(str(method.get("CODE", "")))
    source_token_match = bool(method_tokens) and method_tokens == tokens
    if not source_token_match:
        warnings.append("source_token_mismatch")
    if hint and not whole_file_fallback and hint != _unqualified(native_name):
        warnings.append("hint_mismatch")

    graph_name = hint or (native_name if native_name != "<global>" else "<global>")
    quality = {
        "target_method_id": str(root_ids[0]),
        "target_method_ids": [str(value) for value in root_ids],
        "target_name": graph_name,
        "target_full_name": method.get("FULL_NAME"),
        "filename": filename,
        "resolution_mode": resolution_mode,
        "source_occurrence_exact": source_occurrence_exact,
        "source_token_match": source_token_match,
        "unknown_nodes": len(unknown),
        "node_count": len(owned),
        "edge_count": len(graph_edges),
        "edge_counts": {kind: counts[kind] for kind in ("AST", "CFG", "CDG", "DDG")},
        "missing_relation_kinds": [kind for kind in ("AST", "CFG", "CDG", "DDG") if not counts[kind]],
        "warnings": sorted(set(warnings)),
        "status": "rejected" if reasons else "accepted",
        "reasons": reasons,
    }
    if reasons:
        raise CPGQualityError(json.dumps(quality, sort_keys=True))
    return FunctionGraph(graph_name, graph_nodes, graph_edges, quality)


def _isolate_batch(
    requests: list[dict],
    runner: Callable[[list[dict]], list[FunctionGraph | CPGError]],
) -> list[FunctionGraph | CPGError]:
    """Deterministically isolate batch-level Joern failures by recursive bisection."""
    try:
        return runner(requests)
    except (CPGError, subprocess.TimeoutExpired, OSError) as error:
        if len(requests) <= 1:
            return [error]
        middle = len(requests) // 2
        return _isolate_batch(requests[:middle], runner) + _isolate_batch(requests[middle:], runner)


def extract_function_cpg_batch(
    requests: list[dict],
    *,
    joern_dir: str | Path = "/home/phy/joern",
    java_home: str | Path = "/home/phy/jdk21",
    timeout: int = 300,
) -> list[FunctionGraph | CPGError]:
    """Build function CPGs in bounded batches, isolating batch-level failures."""
    if not requests:
        return []
    root = Path(os.environ.get("JOERN_HOME", str(joern_dir))).expanduser()
    parser = _find_executable(root, ("joern-parse", "joern-cli/joern-parse", "joern-cli/bin/joern-parse"))
    exporter = _find_executable(root, ("joern-export", "joern-cli/joern-export", "joern-cli/bin/joern-export"))
    env = _environment(java_home)

    def run_once(batch: list[dict]) -> list[FunctionGraph | CPGError]:
        with tempfile.TemporaryDirectory(prefix="vulnmechanism-cpg-") as directory:
            work = Path(directory)
            src = work / "src"
            src.mkdir()
            filenames = []
            for index, request in enumerate(batch):
                if request["language"] not in {"c", "cpp"}:
                    raise ValueError("language must be c or cpp")
                filename = f"sample_{index:04d}." + ("cpp" if request["language"] == "cpp" else "c")
                (src / filename).write_text(request.get("full_source", request["source"]), encoding="utf-8")
                filenames.append(filename)

            cpg = work / "cpg.bin"
            result = run_process(
                [str(parser), str(src), "--output", str(cpg), "--frontend-args", "--enable-file-content"],
                timeout=timeout,
                env=env,
            )
            if result.returncode or not cpg.is_file():
                raise CPGError("joern-parse failed: " + (result.stderr or result.stdout)[-4000:])

            output = work / "graph"
            result = run_process(
                [str(exporter), "--repr", "all", "--format", "neo4jcsv", "--out", str(output), str(cpg)],
                timeout=timeout,
                env=env,
            )
            if result.returncode:
                raise CPGError("joern-export failed: " + (result.stderr or result.stdout)[-4000:])

            nodes, edges = read_neo4jcsv(output)
            resolved = []
            for filename, request in zip(filenames, batch):
                try:
                    resolved.append(
                        resolve_target_graph(
                            nodes,
                            edges,
                            filename=filename,
                            source=request["source"],
                            start_line=request.get("start_line", 1),
                            function_hint=request.get("function", ""),
                        )
                    )
                except (TargetMethodError, CPGQualityError) as error:
                    resolved.append(error)
            return resolved

    return _isolate_batch(requests, run_once)


def extract_function_cpg(
    source: str,
    function: str,
    *,
    language: str = "c",
    joern_dir: str | Path = "/home/phy/joern",
    java_home: str | Path = "/home/phy/jdk21",
    timeout: int = 300,
    full_source: str | None = None,
    start_line: int = 1,
) -> FunctionGraph:
    request = {"source": source, "function": function, "language": language, "start_line": start_line}
    if full_source is not None:
        request["full_source"] = full_source
    result = extract_function_cpg_batch(
        [request], joern_dir=joern_dir, java_home=java_home, timeout=timeout
    )[0]
    if isinstance(result, CPGError):
        raise result
    return result
