"""Batched, bounded Joern extraction; candidate relations are not bug labels."""

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from .artifacts import write_json
from .data import read_jsonl, write_jsonl
from .syntax import source_token_spans

JOERN_VERSION = "4.0.465"
EDGE_TYPES = ("AST", "CFG", "CDG", "REACHING_DEF", "REF", "POST_DOMINATE")
from .static_patterns import PATTERN_TYPES, extract_patterns

RELATIONS = EDGE_TYPES + PATTERN_TYPES
PATTERN_SCHEMA = 4

def source_key(source):
    return hashlib.sha256(source.encode()).hexdigest()


def source_extension(row):
    if row["sample_key"].startswith("devign:") or Path(row.get("file_name", "")).suffix == ".c":
        return ".c"
    return ".cpp"


def cache_key(row):
    return source_key(source_extension(row) + "\0" + row["raw_source"])


def align_graph(graph, source, *, language="c"):
    """Exact node-code spans when available; otherwise a marked location anchor.

    Joern C/C++ columns are one-based. Non-exact anchors use one lexical token,
    rather than guessing an entire expression/line as a precise node span.
    """
    starts = [0]
    for i, char in enumerate(source):
        if char == "\n":
            starts.append(i + 1)
    tokens = source_token_spans(source)
    mapped, nodes, exact = {}, [], 0
    for node in graph["nodes"]:
        line, col = node["line"], node["column"]
        if line < 1 or line > len(starts) or col < 1:
            continue
        start = starts[line - 1] + col - 1
        if not 0 <= start < len(source):
            continue
        code = node["code"]
        is_exact = bool(code) and source.startswith(code, start)
        if is_exact:
            end = start + len(code)
        else:
            matches = [(a, b) for _, a, b in tokens if a <= start < b]
            if not matches:
                continue
            start, end = matches[0]
        mapped[node["id"]] = len(nodes)
        nodes.append(dict(span=[start, end], kind=node["kind"],
                          alignment="exact_code" if is_exact else "location_token"))
        exact += is_exact
    edges, counts = [], Counter()
    for a, b, kind in graph["edges"]:
        if kind in EDGE_TYPES and a in mapped and b in mapped:
            edges.append([mapped[a], mapped[b], RELATIONS.index(kind)])
            counts[kind] += 1
    result = dict(
        status="ok" if nodes and counts["AST"] and counts["CFG"] else "incomplete_graph",
        nodes=nodes, edges=edges, counts=dict(counts), raw_nodes=len(graph["nodes"]),
        raw_edges=dict(Counter(e[2] for e in graph["edges"])), exact_spans=exact,
        location_only_spans=len(nodes) - exact,
        extraction="Joern intraprocedural AST, CFG, CDG, reaching definitions and symbol references",
    )
    return attach_patterns(result, source, language=language)


def attach_patterns(graph, source, *, language="c"):
    """Upgrade cached CPG relations with program-pattern candidates in place.

    A graph's CPG edges remain unchanged; the added typed edges connect a
    candidate operation to its operands/guard rather than simply marking API
    names. Cached Joern outputs can be upgraded without rerunning the JVM.
    """
    if graph.get("schema_version") == PATTERN_SCHEMA:
        return graph
    result = dict(graph)
    # A standalone function may fail Joern parsing because of missing project
    # headers/macros while still having syntactically identifiable operations.
    # Keep the original parser status and confidence distinction; do not invent
    # dataflow edges when the CPG is unavailable.
    nodes = [{k: v for k, v in n.items() if k != "candidates"} for n in graph["nodes"]]
    edges = [list(e) for e in graph["edges"] if e[2] < len(EDGE_TYPES)]
    counts = Counter({name: sum(e[2] == i for e in edges) for i, name in enumerate(EDGE_TYPES)})
    indexed = {tuple(n["span"]): i for i, n in enumerate(nodes)}

    def anchor(span):
        key = tuple(span)
        if key not in indexed:
            indexed[key] = len(nodes)
            nodes.append(dict(span=list(span), kind="PATTERN_POSITION", alignment="syntax_span"))
        return indexed[key]

    # Associate call/access sites with nearby CPG definitions and conditions.
    # Only genuine Joern REACHING_DEF/CDG/REF edges extend the syntax pairs.
    base_count = len(nodes)
    nearby = defaultdict(set)
    program_kinds = {EDGE_TYPES.index(x) for x in ("REACHING_DEF", "CDG", "REF")}
    for left, right, kind in edges:
        if kind in program_kinds:
            nearby[left].add(right)
            nearby[right].add(left)

    patterns = extract_patterns(source, language=language)
    for p in patterns:
        kind = RELATIONS.index(p["kind"])
        a = anchor(p["anchor"])
        linked = p["related"] or [p["anchor"]]
        for span in linked:
            b = anchor(span)
            edges.append([a, b, kind])
            counts[p["kind"]] += 1

        # Context is selected through actual program dependencies, not token
        # proximity or a fixed-source/sink vocabulary. The query remains the
        # matched operation, but its candidate keys include dependent definitions
        # and conditions from the intraprocedural CPG when available.
        start, end = p["anchor"]
        matching = [
            i for i in range(base_count)
            if nodes[i]["span"][0] < end and start < nodes[i]["span"][1]
            and nodes[i]["span"][1] - nodes[i]["span"][0] < 1024
        ]
        matching.sort(key=lambda i: (nodes[i]["span"][1] - nodes[i]["span"][0], i))
        destinations = set()
        for j in matching[:12]:
            destinations.update(nearby[j])
        for j in sorted(destinations)[:8]:
            if j != a:
                edges.append([a, j, kind])
                counts[p["kind"]] += 1
    result.update(nodes=nodes, edges=edges, counts=dict(counts),
                  patterns=patterns, pattern_counts=dict(Counter(p["kind"] for p in patterns)),
                  schema_version=PATTERN_SCHEMA,
                  candidates="typed syntax and CPG-context patterns, not vulnerability labels",
                  analysis_source="joern_plus_syntax" if graph["status"] == "ok" else "syntax_only")
    return result


def run_process(argv, log, timeout=300):
    env = dict(os.environ, JAVA_HOME="/home/phy/jdk21")
    env["PATH"] = env["JAVA_HOME"] + "/bin:" + env.get("PATH", "")
    env["JAVA_TOOL_OPTIONS"] = "-Xmx4g -XX:ActiveProcessorCount=4"
    started, peak = time.monotonic(), 0

    def tree_rss(pid):
        total, pending = 0, [pid]
        while pending:
            child = pending.pop()
            try:
                root = Path(f"/proc/{child}")
                for line in (root / "status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1]) * 1024
                children = set()
                for task in (root / "task").iterdir():
                    try:
                        children.update(map(int, (task / "children").read_text().split()))
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                pending.extend(children)
            except (FileNotFoundError, ProcessLookupError):
                pass
        return total
    with Path(log).open("w") as handle:
        p = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while p.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"Joern exceeded {timeout}s: {argv[0]}")
                peak = max(peak, tree_rss(p.pid))
                time.sleep(.25)
            if p.returncode:
                raise RuntimeError(f"Joern exit {p.returncode}; see {log}")
        except BaseException:
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL);p.wait()
            except ProcessLookupError:
                pass
            raise
    return dict(seconds=time.monotonic() - started, peak_process_tree_rss_bytes=peak)


def extract(rows, output_dir, *, joern="/home/phy/joern", batch_size=128, cache=True, cache_root=None):
    """Only supplied functions are parsed; successful and failed outcomes persist."""
    output_dir = Path(output_dir);output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("static_extract.sc")
    namespace = hashlib.sha256(JOERN_VERSION.encode() + script.read_bytes()).hexdigest()[:12]
    cache_dir = (Path(cache_root) if cache_root is not None else output_dir / "cache") / namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending, results, hits = {}, {}, 0
    started = time.monotonic()
    for row in rows:
        key = cache_key(row)
        path = cache_dir / (key + ".json")
        if cache and path.exists():
            cached = json.loads(path.read_text())
            graph = attach_patterns(cached, row["raw_source"],
                                    language="c" if source_extension(row) == ".c" else "cpp")
            results[key] = graph
            if graph.get("schema_version") != cached.get("schema_version"):
                write_json(path, graph)
            hits += 1
        else:
            pending[key] = row
    costs = []
    items = list(pending.items())
    for at in range(0, len(items), batch_size):
        chunk = items[at:at + batch_size]
        with tempfile.TemporaryDirectory(prefix="joern-", dir=output_dir) as temp:
            work = Path(temp);src = work / "source";src.mkdir()
            for key, row in chunk:
                (src / (key + source_extension(row))).write_text(row["raw_source"])
            log_base = output_dir / f"batch.{time.time_ns()}"
            parse = run_process([str(Path(joern) / "joern-parse"), "-J-Xmx4g", "-J-XX:ActiveProcessorCount=4", str(src), "--output", str(work / "cpg.bin")], str(log_base) + ".parse.log")
            export = run_process([str(Path(joern) / "joern"), "-J-Xmx4g", "--script", str(script.resolve()),
                                  "--param", f"inputPath={work / 'cpg.bin'}", "--param", f"outputPath={work / 'graphs.jsonl'}"], str(log_base) + ".export.log")
            by_file = defaultdict(list)
            for graph in read_jsonl(work / "graphs.jsonl"):
                by_file[Path(graph["file"]).stem].append(graph)
            for key, row in chunk:
                definitions = by_file.get(key, [])
                # One function per file. Multiple definitions are explicitly
                # marked, never silently replaced by a same-name call stub.
                if len(definitions) == 1:
                    result = align_graph(definitions[0], row["raw_source"], language="c" if source_extension(row) == ".c" else "cpp")
                else:
                    result = dict(status="no_definition" if not definitions else "multiple_definitions",
                                  definition_count=len(definitions), nodes=[], edges=[], counts={},
                                  schema_version=PATTERN_SCHEMA, patterns=[], pattern_counts={})
                results[key] = result
                write_json(cache_dir / (key + ".json"), result)
            costs.append(dict(functions=len(chunk), parse=parse, export=export))
            with (output_dir / "batches.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(completed_at=time.time(), **costs[-1])) + "\n")
            print(f"Joern {min(at + batch_size, len(items))}/{len(items)}: "
                  f"{parse['seconds'] + export['seconds']:.1f}s", flush=True)
    return {r["sample_key"]: results[cache_key(r)] for r in rows}, dict(
        samples=len(rows), unique_uncached=len(items), cache_hits=hits, batches=costs,
        seconds=time.monotonic() - started, joern=JOERN_VERSION, heap="4 GiB per JVM", cpus_per_jvm=4)


def prepare_relations(data_dir, output_dir, datasets=("bigvul", "devign", "reveal"), *, cache_root=None):
    all_costs = {}
    for dataset in datasets:
        paths = [Path(data_dir) / "pilot" / dataset / f"{s}.jsonl" for s in ("train", "valid")]
        rows = [r for path in paths for r in read_jsonl(path)]
        graphs, costs = extract(rows, Path(output_dir) / "joern", cache_root=cache_root)
        for path in paths:
            write_jsonl(path, [dict(r, program_relations=graphs[r["sample_key"]]) for r in read_jsonl(path)])
        report = dict(costs=costs, status=dict(Counter(g["status"] for g in graphs.values())),
                      relation_functions={k: sum(bool(g["counts"].get(k)) for g in graphs.values()) for k in RELATIONS},
                      exact_spans=sum(g.get("exact_spans", 0) for g in graphs.values()),
                      location_only_spans=sum(g.get("location_only_spans", 0) for g in graphs.values()),
                      by_split_label={f"{s}:{y}": dict(
                          samples=sum(r["split"] == s and r["label"] == y for r in rows),
                          status=dict(Counter(graphs[r["sample_key"]]["status"] for r in rows if r["split"] == s and r["label"] == y)),
                          relation_functions={k: sum(bool(graphs[r["sample_key"]]["counts"].get(k))
                                                     for r in rows if r["split"] == s and r["label"] == y) for k in RELATIONS})
                          for s in ("train", "valid") for y in (0, 1)},
                      test_parsed=False)
        write_json(Path(output_dir) / f"{dataset}.coverage.json", report)
        all_costs[dataset] = report
    return all_costs


def benchmark(raw_root, index_dir, output_dir):
    from .static_data import native_rows
    from .data import balanced_subset
    rows = [r for r in native_rows(raw_root, "devign", index_dir) if r["split"] == "train"]
    rows = balanced_subset(rows, 32, 42)
    root = Path(output_dir)
    singles = []
    for row in rows[:2]:
        _, cost = extract([row], root / "joern", cache=False)
        singles.append(cost)
    graphs, batch = extract(rows, root / "joern", batch_size=32, cache=False)
    _, cached = extract(rows, root / "joern")
    report = dict(protocol="Seed 42 balanced 32 upstream-Train Devign functions; 2 single-function cold runs, same functions included in 32-function batch, then identical cache reread",
                  members=[r["sample_key"] for r in rows], singles=singles, batch=batch, cached=cached,
                  status=dict(Counter(g["status"] for g in graphs.values())),
                  relations={k: sum(bool(g["counts"].get(k)) for g in graphs.values()) for k in RELATIONS})
    write_json(root / "benchmark.json", report)
    print(json.dumps(report), flush=True)
    return report
