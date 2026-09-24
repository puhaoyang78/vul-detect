"""Read-only coverage audit for source-aligned CFG nodes."""
from __future__ import annotations

from collections import Counter, defaultdict

from .cfg_alignment import _source_coordinates, align_nodes, coverage_summary
from .cfg_data import EMPTY, abstract_cfg, load_graphs, read_records

STATUSES = ("aligned_nodes", "missing_position", "position_mismatch",
            "outside_visible_source", "no_visible_token", "invalid_token_offsets")
MISMATCH_TYPES = ("coordinate_invalid", "coordinate_conflict", "text_mismatch")
POSITION_FIELDS = ("OFFSET", "OFFSET_END", "LINE_NUMBER", "COLUMN_NUMBER",
                   "LINE_NUMBER_END", "COLUMN_NUMBER_END")


def _summary(counts: Counter) -> dict:
    result = coverage_summary(counts)
    result["statuses"] = {status: counts[status] for status in STATUSES}
    result["mismatch_types"] = {kind: counts[kind] for kind in MISMATCH_TYPES}
    if result["total_nodes"] != sum(result["statuses"].values()):
        raise ValueError("node alignment statuses do not cover the CFG")
    if counts["position_mismatch"] != sum(result["mismatch_types"].values()):
        raise ValueError("position mismatch subtypes do not cover failures")
    return result


def _short(text: str | None, limit: int = 180) -> dict | None:
    if text is None:
        return None
    return {"text": text[:limit], "length": len(text), "truncated": len(text) > limit}


def _example(source: str, sample_key: str, split: str, node: dict,
             attributes_all_empty: bool, mismatch_type: str) -> dict:
    location = node["properties"]
    code = node["code"]
    units_to_chars, line_starts = _source_coordinates(source)
    begin, finish = location.get("OFFSET"), location.get("OFFSET_END")
    offset_span = (units_to_chars[begin], units_to_chars[finish]) if (
        type(begin) is int and type(finish) is int and begin in units_to_chars and
        finish in units_to_chars and begin < finish) else None
    line, column = location.get("LINE_NUMBER"), location.get("COLUMN_NUMBER")
    line_span = None
    if type(line) is int and type(column) is int and 1 <= line < len(line_starts) and column >= 1:
        line_begin = line_starts[line - 1] + column - 1
        line_finish = line_begin + len(code.encode("utf-16-le")) // 2
        if line_begin in units_to_chars and line_finish in units_to_chars:
            line_span = units_to_chars[line_begin], units_to_chars[line_finish]
    anchor = offset_span[0] if offset_span is not None else line_span[0] if line_span is not None else None
    source_lines = source.splitlines(keepends=True)
    source_line = source_lines[line - 1] if type(line) is int and 1 <= line <= len(source_lines) else None
    return {
        "sample_key": sample_key, "split": split, "node_id": node["id"],
        "node_kind": location["kind"], "operation_name": location.get("NAME") if location["kind"] == "CALL" else None,
        "attributes_all_empty": attributes_all_empty, "mismatch_type": mismatch_type,
        "positions": {field: location.get(field) for field in POSITION_FIELDS},
        "node_code": _short(code),
        "offset_slice": _short(source[offset_span[0]:offset_span[1]]) if offset_span is not None else None,
        "line_slice": _short(source[line_span[0]:line_span[1]]) if line_span is not None else None,
        "source_line": _short(source_line),
        "source_context": _short(source[max(0, anchor - 60):anchor + 120]) if anchor is not None else None,
    }


def audit_alignment(dataset_path: str, graphs_path: str, tokenizer, *,
                    source_dataset: str = "primevul", source_max_length: int = 2048,
                    example_limit: int = 8, progress=None) -> dict:
    """Reuse the saved graph cohort and source tokenizer; never load Qwen weights."""
    if source_max_length <= 0 or example_limit < 0:
        raise ValueError("source_max_length must be positive and example_limit nonnegative")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("alignment audit requires a fast tokenizer with offset_mapping")
    rows = read_records(dataset_path, source_dataset)
    graphs = load_graphs(graphs_path, rows)
    rows = [{"sample_key": row["sample_key"], "raw_source": row["raw_source"],
             "split": row["split"]} for row in rows]
    overall = Counter()
    by_split = defaultdict(Counter)
    by_kind = defaultdict(Counter)
    by_call_name = defaultdict(Counter)
    example_candidates = {reason: defaultdict(list) for reason in MISMATCH_TYPES}
    example_groups = {reason: set() for reason in MISMATCH_TYPES}
    for number, row in enumerate(rows, 1):
        key, source = row["sample_key"], row["raw_source"]
        graph = graphs.pop(key)
        view = abstract_cfg(graph)
        nodes = {node["id"]: node for node in graph["nodes"]}
        tokenized = tokenizer(source, add_special_tokens=False, truncation=True,
                              max_length=source_max_length, return_offsets_mapping=True)
        if len(tokenized["input_ids"]) != len(tokenized["offset_mapping"]):
            raise ValueError(f"{key}: tokenizer IDs and offsets have different lengths")
        statuses = []
        _, counts = align_nodes(source, view["locations"], tokenized["offset_mapping"],
                                0, statuses=statuses)
        if len(statuses) != len(view["node_ids"]):
            raise ValueError(f"{key}: alignment status count does not match CFG nodes")
        overall.update(counts)
        by_split[row["split"]].update(counts)
        for node_id, signatures, (status, mismatch_type) in zip(
                view["node_ids"], view["signatures"], statuses):
            node = nodes[node_id]
            kind = node["properties"]["kind"]
            all_empty = all(value == EMPTY for value in signatures)
            groups = [(by_kind, (kind, all_empty))]
            if kind == "CALL":
                groups.append((by_call_name,
                               (str(node["properties"].get("NAME") or "<missing>"), all_empty)))
            for table, group in groups:
                table[group]["total_nodes"] += 1
                table[group][status] += 1
                if mismatch_type is not None:
                    table[group][mismatch_type] += 1
            if mismatch_type is not None:
                overall[mismatch_type] += 1
                by_split[row["split"]][mismatch_type] += 1
                group = (kind, str(node["properties"].get("NAME")) if kind == "CALL" else None, all_empty)
                candidates = example_candidates[mismatch_type][kind]
                if len(candidates) < min(example_limit, 3) and group not in example_groups[mismatch_type]:
                    example_groups[mismatch_type].add(group)
                    candidates.append(_example(source, key, row["split"], node, all_empty, mismatch_type))
        if progress is not None and (number % 500 == 0 or number == len(rows)):
            progress(number, len(rows), counts)
    if graphs:
        raise ValueError("unconsumed graph records remain after the cohort audit")
    mismatch_examples = {}
    for mismatch_type in MISMATCH_TYPES:
        candidates = example_candidates[mismatch_type]
        kind_order = sorted(candidates, key=lambda kind: (
            -sum(by_kind.get((kind, empty), {}).get(mismatch_type, 0)
                 for empty in (False, True)), kind))
        chosen = []
        for rank in range(3):
            for kind in kind_order:
                if rank < len(candidates[kind]) and len(chosen) < example_limit:
                    chosen.append(candidates[kind][rank])
        mismatch_examples[mismatch_type] = chosen

    def table_rows(table, field):
        result = []
        for (name, all_empty), counts in table.items():
            result.append({field: name, "attributes_all_empty": all_empty, **_summary(counts)})
        return sorted(result, key=lambda row: (-row["total_nodes"], str(row[field]), row["attributes_all_empty"]))

    return {
        "dataset": str(dataset_path), "graphs": str(graphs_path),
        "source_dataset": source_dataset, "samples": len(rows),
        "source_max_length": source_max_length,
        "overall": _summary(overall),
        "by_split": {split: _summary(counts) for split, counts in sorted(by_split.items())},
        "by_node_kind": table_rows(by_kind, "node_kind"),
        "by_call_name": table_rows(by_call_name, "operation_name"),
        "mismatch_examples": mismatch_examples,
    }
