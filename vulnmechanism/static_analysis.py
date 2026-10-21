"""Batched, bounded Joern extraction; candidate relations are not bug labels."""

from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
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
PATTERN_SCHEMA = 6

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
    original = [i for i, n in enumerate(graph["nodes"]) if n["kind"] != "PATTERN_POSITION"]
    remap = {old: new for new, old in enumerate(original)}
    nodes = [{k: v for k, v in graph["nodes"][i].items() if k != "candidates"} for i in original]
    edges = [[remap[a], remap[b], kind] for a, b, kind in graph["edges"]
             if kind < len(EDGE_TYPES) and a in remap and b in remap]
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
    # Only short overlapping CPG spans can supply context. Bound the lookup
    # by source positions instead of rescanning every CPG node per pattern.
    context_nodes = sorted((n["span"][0], i) for i, n in enumerate(nodes)
                           if n["span"][1] - n["span"][0] < 1024)
    context_starts = [start for start, _ in context_nodes]
    nearby = defaultdict(set)
    program_kinds = {EDGE_TYPES.index(x) for x in ("REACHING_DEF", "CDG", "REF")}
    for left, right, kind in edges:
        if kind in program_kinds:
            nearby[left].add(right)
            nearby[right].add(left)

    diagnostics = {}
    patterns = extract_patterns(source, language=language,
                                graph=dict(graph, nodes=nodes, edges=edges), diagnostics=diagnostics)
    diagnostics.pop("traces")  # Detailed per-site evidence is generated for audit cases.
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
        first = bisect_right(context_starts, start - 1024)
        last = bisect_left(context_starts, end)
        matching = [i for _, i in context_nodes[first:last] if start < nodes[i]["span"][1]]
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
    result["pattern_diagnostics"] = diagnostics
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
                                  definition_count=len(definitions), nodes=[], edges=[], counts={})
                    result = attach_patterns(result, row["raw_source"],
                                             language="c" if source_extension(row) == ".c" else "cpp")
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


def audit_patterns(data_dir, output_dir, datasets=("bigvul", "devign", "reveal"), *, model_path):
    """Compare cached patterns with this matcher on the identical Train/Valid members.

    This stage reads cached CPGs and never invokes Joern, a classifier, or Test.
    Run before extract to preserve the pre-change comparison. Detailed site
    decisions are included for a fixed stratified sample and first changed rules.
    """
    import csv
    import random
    from .model import AutoTokenizer, InputBuilder
    from .syntax import byte_character_map, parser_for, walk

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    builder = InputBuilder(tokenizer)
    budget = builder.source_max_length - len(builder.source_prefix) - 2
    rng = random.Random(42)
    specific = set(PATTERN_TYPES) - {
        "memory_access", "pointer_access", "guard", "sensitive_read", "sensitive_write", "allocation", "release"
    }
    tables, members, summaries = [], [], {}
    selected, strata_seen, changes_seen = {}, Counter(), set()
    started = time.monotonic()

    for dataset in datasets:
        before, after, before_instances, after_instances = Counter(), Counter(), Counter(), Counter()
        status, strata = Counter(), defaultdict(lambda: dict(samples=0, before=Counter(), after=Counter()))
        stages = {kind: Counter() for kind in PATTERN_TYPES}
        visible, truncated, alignment_missing = Counter(), Counter(), Counter()
        certainty, call_models, parse_errors = Counter(), Counter(), 0
        analysis_seconds, token_seconds = 0.0, 0.0
        count = 0
        for split in ("train", "valid"):
            path = Path(data_dir) / "pilot" / dataset / (split + ".jsonl")
            for row in read_jsonl(path):
                count += 1
                source, old = row["raw_source"], row["program_relations"]
                language = "c" if source_extension(row) == ".c" else "cpp"
                diag = {}
                tick = time.monotonic()
                patterns = extract_patterns(source, language=language, graph=old, diagnostics=diag)
                analysis_seconds += time.monotonic() - tick
                old_count = Counter(p["kind"] for p in old.get("patterns", []))
                new_count = Counter(p["kind"] for p in patterns)
                before.update(old_count.keys()); after.update(new_count.keys())
                before_instances.update(old_count); after_instances.update(new_count)
                status[old["status"]] += 1
                parse_errors += diag["tree_has_error"]
                certainty.update(p["evidence"].get("certainty", "syntactic_candidate") for p in patterns)
                call_models.update(diag["unmodeled_calls"])
                bucket = strata[f"{split}:{row['label']}"]
                bucket["samples"] += 1
                bucket["before"].update(old_count.keys()); bucket["after"].update(new_count.keys())
                for kind, values in diag["rule_stages"].items():
                    stages[kind].update(values)

                tick = time.monotonic()
                offsets = tokenizer(source, add_special_tokens=False, truncation=True, max_length=budget,
                                    return_offsets_mapping=True)["offset_mapping"]
                ends = [b for _, b in offsets]
                for p in patterns:
                    a, b = p["anchor"]
                    if not 0 <= a < b <= len(source):
                        raise ValueError(f"invalid syntax span: {row['sample_key']}")
                    for c, d in p["related"] + p["evidence"].get("definitions", []):
                        if not 0 <= c < d <= len(source):
                            raise ValueError(f"invalid relation operand span: {row['sample_key']}")
                    pos = bisect_right(ends, a)
                    if pos < len(offsets) and offsets[pos][0] < b:
                        visible[p["kind"]] += 1
                    elif offsets and a >= offsets[-1][1] and len(offsets) == budget:
                        truncated[p["kind"]] += 1
                    else:
                        alignment_missing[p["kind"]] += 1
                token_seconds += time.monotonic() - tick
                members.append(dict(dataset=dataset, split=split, label=row["label"], sample_key=row["sample_key"],
                                    joern_status=old["status"], tree_sitter_has_error=diag["tree_has_error"],
                                    before_patterns=";".join(sorted(old_count)), after_patterns=";".join(sorted(new_count)),
                                    added=";".join(sorted(new_count.keys() - old_count.keys())),
                                    removed=";".join(sorted(old_count.keys() - new_count.keys()))))

                # One fixed reservoir sample for each label/matched stratum.
                eligible = bool(diag["subscript_nodes"] or any(
                    values.get("sites", 0) for kind, values in diag["rule_stages"].items() if kind in specific))
                if eligible:
                    stratum = (dataset, row["label"], bool(specific & old_count.keys()))
                    strata_seen[stratum] += 1
                    if rng.randrange(strata_seen[stratum]) == 0:
                        selected[("stratum", *stratum)] = (row, patterns, diag)
                # First changed target family, capped at three extra functions/dataset.
                changed = (old_count.keys() ^ new_count.keys()) & (specific | {"allocation"})
                for kind in sorted(changed):
                    slot = (dataset, kind)
                    if slot not in changes_seen and sum(k[0] == "change" and k[1] == dataset for k in selected) < 3:
                        changes_seen.add(slot)
                        selected[("change", dataset, kind)] = (row, patterns, diag)
        if count != 1280:
            raise ValueError(f"expected unchanged 1280-function pilot: {dataset}: {count}")
        for kind in PATTERN_TYPES:
            tables.append(dict(dataset=dataset, pattern=kind, samples=count,
                               before_functions=before[kind], after_functions=after[kind],
                               before_instances=before_instances[kind], after_instances=after_instances[kind],
                               before_fraction=before[kind] / count, after_fraction=after[kind] / count,
                               sites=stages[kind]["sites"], matched_sites=stages[kind]["matched"],
                               filtered_sites=stages[kind]["filtered"], unknown_sites=stages[kind]["unknown"],
                               resolved_extents=stages[kind]["resolved_extents"],
                               visible_anchor_instances=visible[kind], truncated_anchor_instances=truncated[kind],
                               alignment_missing_instances=alignment_missing[kind]))
        summaries[dataset] = dict(
            samples=count, joern_status_before=dict(status), joern_status_after=dict(status),
            tree_sitter_error_functions=parse_errors, rule_stages={k: dict(v) for k, v in stages.items()},
            by_split_label={key: dict(samples=v["samples"], before=dict(v["before"]), after=dict(v["after"]))
                            for key, v in strata.items()},
            matches_by_certainty=dict(certainty), unmodeled_calls_top30=call_models.most_common(30),
            rule_seconds=analysis_seconds, token_audit_seconds=token_seconds,
            visible_anchors=dict(visible), truncated_anchors=dict(truncated), alignment_missing=dict(alignment_missing),
        )
        print(dataset, "old/new target coverage", {
            k: (before[k], after[k]) for k in ("buffer_size_mismatch", "off_by_one", "buffer_overread",
                                               "double_free", "use_after_free", "negative_index_read", "negative_index_write")},
              "rule seconds", round(analysis_seconds, 3), flush=True)

    cases = []
    unique = {row["sample_key"]: (row, p, d) for row, p, d in selected.values()}
    for key, (row, patterns, diag) in sorted(unique.items()):
        source, graph = row["raw_source"], row["program_relations"]
        language = "c" if source_extension(row) == ".c" else "cpp"
        data = source.encode()
        mapping = byte_character_map(source)
        syntax = list(walk(parser_for(language).parse(data).root_node))
        events, event_keys = [], set()
        # Keep matched/filtered/unknown examples of each target family.
        priority = {k: i for i, k in enumerate(("buffer_size_mismatch", "off_by_one", "buffer_overread",
                                               "double_free", "use_after_free", "source_size_copy", "destination_size_copy"))}
        for trace in sorted(diag["traces"], key=lambda v: priority.get(v["kind"], len(priority))):
            if trace["kind"] not in specific:
                continue
            signature = (trace["kind"], trace["outcome"], trace["reason"])
            if signature in event_keys or len(events) >= 12:
                continue
            event_keys.add(signature)
            a, b = trace["anchor"]
            ast_nodes = [n for n in syntax if mapping[n.start_byte] == a and mapping[n.end_byte] == b]
            cpg = [i for i, n in enumerate(graph["nodes"])
                   if n["kind"] != "PATTERN_POSITION" and n["span"][0] <= a < n["span"][1]]
            cpg.sort(key=lambda i: (graph["nodes"][i]["span"][1] - graph["nodes"][i]["span"][0], i))
            sites = set(cpg[:3])
            links = [[graph["nodes"][x]["span"], graph["nodes"][y]["span"], EDGE_TYPES[t]]
                     for x, y, t in graph["edges"] if t < len(EDGE_TYPES) and (x in sites or y in sites)]
            line = source.count("\n", 0, a) + 1
            lines = source.splitlines()
            parents = []
            parent = ast_nodes[0].parent if ast_nodes else None
            for _ in range(5):
                if parent is None:
                    break
                parents.append(parent.type); parent = parent.parent
            events.append(dict(
                **trace, code=source[a:b], excerpt="\n".join(
                    f"{i + 1}: {lines[i]}" for i in range(max(0, line - 3), min(len(lines), line + 3))),
                ast_types=[n.type for n in ast_nodes],
                ast_context=parents,
                cpg_sites=[dict(index=i, **graph["nodes"][i]) for i in cpg[:3]],
                cpg_incident_edges=links[:20],
            ))
        cases.append(dict(sample_key=key, dataset=row["dataset"], split=row["split"], label=row["label"],
                          project=row["project"], language=language, joern_status=graph["status"],
                          tree_sitter_has_error=diag["tree_has_error"],
                          old_patterns=dict(Counter(p["kind"] for p in graph.get("patterns", []))),
                          new_patterns=dict(Counter(p["kind"] for p in patterns)), events=events))
    for name, values in (("pattern_coverage.csv", tables), ("pattern_members.csv", members)):
        with (root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, list(values[0]))
            writer.writeheader(); writer.writerows(values)
    write_json(root / "pattern_cases.json", dict(
        selection="Seed42 one reservoir sample per dataset/label/old-specific-match stratum, plus up to three first changed families per dataset; not prevalence or vulnerability-precision estimation",
        cases=cases))
    result = dict(
        datasets=summaries, comparison=tables, elapsed_seconds=time.monotonic() - started,
        protocol=dict(seed=42, splits=["train", "valid"], samples_per_dataset=1280,
                      before="cached input program_relations, without replacing historical model results",
                      after="current extract_patterns with the same cached CPG",
                      models_used=False, test_used=False, joern_processes_started=0,
                      logical_input_budget=builder.source_max_length, source_token_budget=budget,
                      candidate_unit="rule-site or same-object event-pair attempt; accepted instances are deduplicated separately",
                      unknowns="not admitted as vulnerability-pattern edges; generic syntax relations remain available"),
        cases_file="pattern_cases.json",
        interpretation="Relations/candidates are not vulnerability truth. Missing extent, path, alias or parser context is unknown. Signedness alone does not prove a negative index. Token truncation is separate from missing alignment.",
    )
    write_json(root / "pattern_audit.json", result)
    return result


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
