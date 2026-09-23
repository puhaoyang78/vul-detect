"""Lossless function-graph sidecars and train-only abstract CFG attributes.

The four attribute families and assignment-centred extraction follow DeepDFA's
ideas. This is an adaptation to the current Joern export, not its exact artifact.
No labels, commit identifiers, graph IDs, or source positions are input features.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Iterable

from .cpg import JOERN_SOURCE_PREPROCESSING_VERSION, _prepare_joern_source

GRAPH_SCHEMA = 1
FEATURE_SCHEMA = 1
FAMILIES = ("api", "datatype", "literal", "operator")
ASSIGNMENTS = {
    "assignment", "assignmentDivision", "assignmentExponentiation",
    "assignmentPlus", "assignmentMinus", "assignmentModulo",
    "assignmentMultiplication", "preIncrement", "preDecrement",
    "postIncrement", "postDecrement", "assignmentOr", "assignmentAnd",
    "assignmentXor", "assignmentArithmeticShiftRight",
    "assignmentLogicalShiftRight", "assignmentShiftLeft",
}
EMPTY = "[]"
UNKNOWN = "<UNK>"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def identity(record: dict) -> dict:
    return {"sample_key": record["sample_key"], "dataset": record["dataset"],
            "split": record["split"], "label": record["label"],
            "source_sha256": source_hash(record["raw_source"])}


def cohort_hash(records: Iterable[dict]) -> str:
    return digest(sorted((identity(r) for r in records), key=lambda r: r["sample_key"]))


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                rows.append(row)
    return rows


def read_records(path: str | Path, source_dataset: str) -> list[dict]:
    if source_dataset not in {"primevul", "cleanvul", "sven"}:
        raise ValueError("source_dataset must be primevul, cleanvul, or sven")
    rows = [r for r in read_jsonl(path) if r.get("dataset") == source_dataset]
    if not rows:
        raise ValueError(f"no {source_dataset} records in {path}")
    seen = set()
    for r in rows:
        key = r.get("sample_key")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError(f"missing/duplicate sample_key: {key!r}")
        seen.add(key)
        if r.get("schema_version") != 9:
            raise ValueError(f"{key}: expected existing schema-9 function dataset")
        if type(r.get("label")) is not int or r["label"] not in (0, 1):
            raise ValueError(f"{key}: invalid label")
        if r.get("split") not in {"train", "valid", "test", "external_test"}:
            raise ValueError(f"{key}: invalid split")
        if source_dataset == "sven" and r["split"] != "external_test":
            raise ValueError("SVEN is external-test-only")
        if not isinstance(r.get("raw_source"), str) or not r["raw_source"].strip():
            raise ValueError(f"{key}: missing raw_source")
    return rows


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(partial, path)


@contextmanager
def output_lock(path: str | Path):
    """Advisory lock. Keep its inode so a second process cannot bypass the lock."""
    import fcntl
    lock = Path(str(path) + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process is using {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_graph(graph: dict) -> None:
    if not isinstance(graph, dict):
        raise ValueError("graph must be an object")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
        raise ValueError("graph requires nonempty nodes and an edge list")
    seen = set()
    for n in nodes:
        if not isinstance(n, dict):
            raise ValueError("invalid node")
        key = n.get("id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("missing/duplicate node ID")
        seen.add(key)
        if not isinstance(n.get("code"), str) or not isinstance(n.get("label"), str):
            raise ValueError("node code/label missing")
        if not isinstance(n.get("properties"), dict) or not n["properties"].get("kind"):
            raise ValueError("raw Joern properties missing; install the cpg.py update and rebuild")
    kinds = Counter()
    for e in edges:
        if (not isinstance(e, dict) or e.get("source") not in seen or
                e.get("target") not in seen or
                e.get("kind") not in {"AST", "CFG", "CDG", "DDG"}):
            raise ValueError("invalid/dangling graph edge")
        kinds[e["kind"]] += 1
    if not kinds["AST"] or not kinds["CFG"]:
        raise ValueError("AST and CFG are required; missing DDG/CDG is allowed")


def serialize_graph(graph) -> dict:
    result = {
        "nodes": [{"id": n.node_id, "label": n.label, "code": n.code,
                   "properties": dict(getattr(n, "properties", {}))}
                  for n in graph.nodes.values()],
        "edges": [{"kind": e.kind, "source": e.source, "target": e.target}
                  for e in graph.edges],
    }
    validate_graph(result)
    return result


def load_graphs(path: str | Path, records: list[dict], *, require_complete: bool = True) -> dict[str, dict]:
    expected = {r["sample_key"]: identity(r) for r in records}
    result = {}
    for row in read_jsonl(path):
        key = row.get("sample_key")
        if key not in expected or key in result:
            raise ValueError(f"unexpected/duplicate graph key {key!r}; use one sidecar per cohort")
        if row.get("graph_schema_version") != GRAPH_SCHEMA:
            raise ValueError(f"{key}: incompatible graph schema")
        if row.get("preprocessing_version") != JOERN_SOURCE_PREPROCESSING_VERSION:
            raise ValueError(f"{key}: stale Joern preprocessing version; rebuild the graph sidecar")
        if any(row.get(k) != v for k, v in expected[key].items()):
            raise ValueError(f"{key}: graph/source/label/split mismatch; do not mix datasets")
        original_hash = row.get("original_source_sha256")
        parsed_hash = row.get("parsed_source_sha256")
        applied = row.get("preprocessing_applied")
        if original_hash != expected[key]["source_sha256"]:
            raise ValueError(f"{key}: preprocessing audit/source mismatch")
        if (type(applied) is not bool or not isinstance(parsed_hash, str) or
                re.fullmatch(r"[0-9a-f]{64}", parsed_hash) is None or
                (parsed_hash != original_hash) != applied):
            raise ValueError(f"{key}: invalid preprocessing audit metadata")
        validate_graph(row.get("graph"))
        result[key] = row["graph"]
    missing = sorted(set(expected) - set(result))
    if missing and require_complete:
        raise ValueError(f"{len(missing)} graphs missing (e.g. {missing[:5]}). "
                         "Rerun build; no samples are silently removed from A/B/C.")
    return result


def repair_tail(path: Path) -> None:
    """Recover only a non-newline-terminated final write, never interior corruption."""
    if not path.exists():
        return
    with path.open("r+b") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                try:
                    json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    handle.truncate(offset)
                    print(f"recovered_incomplete_graph_tail={path}", flush=True)
                else:
                    handle.seek(0, os.SEEK_END)
                    handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                break
            if line.strip():
                json.loads(line)  # An invalid complete row is an error, not something to discard.


def build_graphs(dataset_path: str, output_path: str, *, source_dataset: str = "primevul",
                 joern_dir: str = "/home/phy/joern", java_home: str = "/home/phy/jdk21",
                 timeout: int = 300, batch_size: int = 8, extractor=None) -> dict:
    if not 1 <= batch_size <= 32 or timeout <= 0:
        raise ValueError("batch_size must be 1..32 and timeout must be positive")
    rows = read_records(dataset_path, source_dataset)
    output = Path(output_path)
    derived = [output, Path(str(output) + ".meta.json"), Path(str(output) + ".audit.json"),
               Path(str(output) + ".errors.jsonl"), Path(str(output) + ".lock")]
    if Path(dataset_path).resolve() in {p.resolve() for p in derived}:
        raise ValueError("graph output must not overwrite the original dataset")
    # Match the existing extractor's JOERN_HOME precedence in cache provenance.
    effective_joern_dir = os.environ.get("JOERN_HOME", str(joern_dir))
    expected_meta = dict(graph_schema_version=GRAPH_SCHEMA, cohort_sha256=cohort_hash(rows),
                         preprocessing_version=JOERN_SOURCE_PREPROCESSING_VERSION,
                         source_dataset=source_dataset, samples=len(rows),
                         joern_dir=str(Path(effective_joern_dir).expanduser().resolve()),
                         java_home=str(Path(java_home).expanduser().resolve()))
    if extractor is None:
        from .cpg import extract_function_cpg_batch
        extractor = extract_function_cpg_batch
    with output_lock(output):
        meta_path = derived[1]
        if meta_path.exists():
            if json.loads(meta_path.read_text()) != expected_meta:
                raise ValueError("graph cache metadata mismatch; remove the stale graph sidecar and rebuild it")
        elif output.exists() and output.stat().st_size:
            raise ValueError("graph sidecar exists without ownership metadata; choose a new path")
        else:
            atomic_json(meta_path, expected_meta)
        repair_tail(output)
        done = load_graphs(output, rows, require_complete=False) if output.exists() else {}
        pending = [r for r in rows if r["sample_key"] not in done]
        print(f"graph_total={len(rows)} resumed={len(done)} pending={len(pending)}", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        errors = []
        with output.open("a", encoding="utf-8") as out, derived[3].open("a", encoding="utf-8") as errout:
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                requests = []
                preprocessing_audits = []
                for r in batch:
                    language = r.get("resolved_language") or r.get("language")
                    if language not in {"c", "cpp"}:
                        from .syntax import resolve_language
                        language = resolve_language(r["raw_source"], r.get("language", "c_cpp"), "")
                    prepared_source = _prepare_joern_source(
                        r["raw_source"], language=language, standalone=True
                    )
                    preprocessing_audits.append({
                        "preprocessing_version": JOERN_SOURCE_PREPROCESSING_VERSION,
                        "preprocessing_applied": prepared_source != r["raw_source"],
                        "original_source_sha256": source_hash(r["raw_source"]),
                        "parsed_source_sha256": source_hash(prepared_source),
                    })
                    requests.append(dict(source=r["raw_source"], language=language,
                                         function=r.get("function_name") or r.get("syntax_hint") or ""))
                started = time.monotonic()
                try:
                    results = extractor(requests, joern_dir=joern_dir, java_home=java_home, timeout=timeout)
                except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                    results = [exc] * len(batch)
                if len(results) != len(batch):
                    raise RuntimeError("extractor returned the wrong batch size")
                for r, raw, preprocessing_audit in zip(batch, results, preprocessing_audits):
                    try:
                        if isinstance(raw, BaseException):
                            raise raw
                        graph = serialize_graph(raw)
                        abstract_cfg(graph)  # Check that the graph is usable before marking success.
                    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
                        error = dict(identity(r), error_type=type(exc).__name__, error=str(exc))
                        errors.append(error)
                        errout.write(json.dumps(error, ensure_ascii=False) + "\n")
                        errout.flush()
                        print(f"graph_failed={r['sample_key']} error={exc}", flush=True)
                        continue
                    row = dict(identity(r), graph_schema_version=GRAPH_SCHEMA,
                               **preprocessing_audit, graph=graph, cpg_quality=raw.quality,
                               seconds=(time.monotonic()-started)/len(batch))
                    out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    done[r["sample_key"]] = graph
                    print(f"graph_done={r['sample_key']} ok={len(done)}/{len(rows)} "
                          f"nodes={len(graph['nodes'])} edges={len(graph['edges'])}", flush=True)
        groups = defaultdict(Counter)
        for r in rows:
            group = groups[f"{r['split']}/label_{r['label']}"]
            group["total"] += 1
            if r["sample_key"] in done:
                group["success"] += 1
                view = abstract_cfg(done[r["sample_key"]])
                group["cfg_nodes"] += len(view["node_ids"])
                group["definition_nodes"] += view["definition_count"]
                group["without_definition"] += view["definition_count"] == 0
            else:
                group["failed"] += 1
        report = dict(expected_meta, success=len(done), failed=len(rows)-len(done),
                      groups={k: dict(v) for k, v in sorted(groups.items())},
                      failures_this_run=errors, complete=len(done) == len(rows))
        atomic_json(derived[2], report)
        print(json.dumps({k: v for k, v in report.items() if k != "failures_this_run"}, ensure_ascii=False), flush=True)
        return report


def abstract_cfg(graph: dict) -> dict:
    """Assignment-centred API/type/literal/operator multisets at CFG node IDs.

    Node identities are preserved even for repeated source text. Raw positions
    remain in the sidecar but are not embedded. No graph size truncation occurs.
    """
    validate_graph(graph)
    nodes = {n["id"]: n for n in graph["nodes"]}
    ast = defaultdict(list)
    cfg_edges = set()
    for edge in graph["edges"]:
        if edge["kind"] == "AST":
            ast[edge["source"]].append(edge["target"])
        elif edge["kind"] == "CFG":
            cfg_edges.add((edge["source"], edge["target"]))
    used = {i for pair in cfg_edges for i in pair}
    node_ids = sorted(used, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    index = {key: i for i, key in enumerate(node_ids)}

    def kind(key):
        return nodes[key]["properties"].get("kind", "")

    def name(key):
        return str(nodes[key]["properties"].get("NAME", ""))

    def argument(key, position):
        for child in ast.get(key, ()):
            props = nodes[child]["properties"]
            order = props.get("ARGUMENT_INDEX", props.get("ORDER"))
            if order == position:
                return child
        return None

    def datatype(key, seen=None):
        if key is None:
            return None
        seen = set() if seen is None else seen
        if key in seen:
            return None
        seen.add(key)
        t = nodes[key]["properties"].get("TYPE_FULL_NAME")
        if isinstance(t, str) and t not in {"", "ANY", "<unknown>", "UNKNOWN"}:
            # Keep the actual expression's type when Joern knows it (including qualifiers).
            return " ".join(t.split())
        if kind(key) == "CALL":
            return datatype(argument(key, 2 if name(key) == "<operator>.cast" else 1), seen)
        return None

    signatures, definitions = [], 0
    for root in node_ids:
        values = {f: [] for f in FAMILIES}
        op = name(root).removeprefix("<operator>.")
        if kind(root) == "CALL" and name(root).startswith("<operator>.") and op in ASSIGNMENTS:
            definitions += 1
            t = datatype(argument(root, 1))
            if t:
                values["datatype"].append(t)
            seen, stack = set(), list(ast.get(root, ()))
            while stack:
                key = stack.pop()
                if key in seen or kind(key) == "METHOD":
                    continue
                seen.add(key)
                if kind(key) == "LITERAL":
                    values["literal"].append(nodes[key]["code"])
                elif kind(key) == "CALL":
                    n = name(key)
                    if n.startswith("<operator>."):
                        operator = n.removeprefix("<operator>.")
                        if operator != "indirection":
                            values["operator"].append(operator)
                    elif n:
                        values["api"].append(n)
                stack.extend(ast.get(key, ()))
        signatures.append([json.dumps(sorted(values[f]), ensure_ascii=False, separators=(",", ":"))
                           for f in FAMILIES])
    position_fields = ("OFFSET", "OFFSET_END", "LINE_NUMBER", "COLUMN_NUMBER",
                       "LINE_NUMBER_END", "COLUMN_NUMBER_END")
    locations = [dict(code=nodes[key]["code"],
                      **{field: nodes[key]["properties"][field] for field in position_fields
                         if field in nodes[key]["properties"]}) for key in node_ids]
    return dict(node_ids=node_ids, signatures=signatures, locations=locations,
                edges=sorted((index[s], index[t]) for s, t in cfg_edges),
                definition_count=definitions)


class AttributeVocabulary:
    """A separate categorical multiset vocabulary for each family, fit on train only."""
    def __init__(self, values: dict[str, list[str]]):
        if set(values) != set(FAMILIES):
            raise ValueError("invalid attribute vocabulary families")
        for family in FAMILIES:
            items = values[family]
            if (not isinstance(items, list) or items[:2] != [EMPTY, UNKNOWN] or
                    not all(isinstance(v, str) for v in items) or len(items) != len(set(items))):
                raise ValueError("invalid attribute vocabulary")
        self.values = values
        self.indices = {f: {v: i for i, v in enumerate(values[f])} for f in FAMILIES}

    @classmethod
    def fit(cls, rows: list[dict], views: dict[str, dict], limit: int = 2048):
        if limit < 2 or not rows or any(r["split"] != "train" for r in rows):
            raise ValueError("vocabulary fitting requires train-only records and limit >= 2")
        counts = {f: Counter() for f in FAMILIES}
        for r in rows:
            for signatures in views[r["sample_key"]]["signatures"]:
                for f, signature in zip(FAMILIES, signatures):
                    if signature != EMPTY:
                        counts[f][signature] += 1
        return cls({f: [EMPTY, UNKNOWN] + [s for s, _ in sorted(counts[f].items(),
                     key=lambda item: (-item[1], item[0]))[:limit-2]] for f in FAMILIES})

    def encode(self, view: dict) -> list[list[int]]:
        return [[self.indices[f].get(s, 1) for f, s in zip(FAMILIES, row)]
                for row in view["signatures"]]

    def sizes(self) -> list[int]:
        return [len(self.values[f]) for f in FAMILIES]
