"""Map Joern's UTF-16 source positions to visible source-token offsets."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter


def _source_coordinates(source: str):
    units_to_chars = {0: 0}
    units = 0
    for index, char in enumerate(source, 1):
        units += 2 if ord(char) > 0xFFFF else 1
        units_to_chars[units] = index
    line_starts = [0]
    for line in source.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line.encode("utf-16-le")) // 2)
    return units_to_chars, line_starts


def _node_span(source: str, location: dict, units_to_chars: dict, line_starts: list[int]):
    code = location.get("code")
    if not isinstance(code, str) or not code:
        return None, "missing_position"
    begin, finish = location.get("OFFSET"), location.get("OFFSET_END")
    has_offsets = type(begin) is int and type(finish) is int
    line, column = location.get("LINE_NUMBER"), location.get("COLUMN_NUMBER")
    has_line = type(line) is int and type(column) is int
    if not has_offsets and not has_line:
        return None, "missing_position"
    if has_line and 1 <= line < len(line_starts) and column >= 1:
        line_begin = line_starts[line - 1] + column - 1
    else:
        line_begin = None
    if has_offsets:
        if begin not in units_to_chars or finish not in units_to_chars or begin >= finish:
            return None, "coordinate_invalid"
        if has_line and line_begin is None:
            return None, "coordinate_invalid"
        if has_line and line_begin != begin:
            return None, "coordinate_conflict"
    else:
        if line_begin is None:
            return None, "coordinate_invalid"
        begin = line_begin
        finish = begin + len(code.encode("utf-16-le")) // 2
        if begin not in units_to_chars or finish not in units_to_chars:
            return None, "coordinate_invalid"
    end_line, end_column = location.get("LINE_NUMBER_END"), location.get("COLUMN_NUMBER_END")
    if type(end_line) is int and type(end_column) is int:
        if not 1 <= end_line < len(line_starts) or end_column < 1:
            return None, "coordinate_invalid"
        if finish != line_starts[end_line - 1] + end_column:
            return None, "coordinate_conflict"
    char_begin, char_finish = units_to_chars[begin], units_to_chars[finish]
    if source[char_begin:char_finish] != code:
        return None, "text_mismatch"
    return (char_begin, char_finish), None


def align_nodes(source: str, locations: list[dict], offsets: list[tuple[int, int]],
                prefix_tokens: int, statuses: list[tuple[str, str | None]] | None = None
                ) -> tuple[list[tuple[int, int]], Counter]:
    """Return (node index, sequence token index) pairs and node coverage counts.

    Offsets are from tokenizing only the source with the original source budget.
    A node needs a verified whole-source span fully inside the visible tokens.
    """
    counts = Counter(total_nodes=len(locations))
    if type(prefix_tokens) is not int or prefix_tokens < 0:
        raise ValueError("invalid source-prefix token count")
    tokens = []
    previous_start, previous_end = -1, -1
    for index, offset in enumerate(offsets):
        if (not isinstance(offset, (list, tuple)) or len(offset) != 2 or
                any(type(value) is not int for value in offset)):
            counts["invalid_token_offsets"] = len(locations)
            if statuses is not None:
                statuses.extend([("invalid_token_offsets", None)] * len(locations))
            return [], counts
        start, end = offset
        if not 0 <= start <= end <= len(source) or start < previous_start or end < previous_end:
            counts["invalid_token_offsets"] = len(locations)
            if statuses is not None:
                statuses.extend([("invalid_token_offsets", None)] * len(locations))
            return [], counts
        previous_start, previous_end = start, end
        if start < end:
            tokens.append((start, end, prefix_tokens + index))
    if not tokens:
        counts["no_visible_token"] = len(locations)
        if statuses is not None:
            statuses.extend([("no_visible_token", None)] * len(locations))
        return [], counts
    units_to_chars, line_starts = _source_coordinates(source)
    token_ends = [end for _, end, _ in tokens]
    visible_end = tokens[-1][1]
    pairs = []
    for node_index, location in enumerate(locations):
        span, reason = _node_span(source, location, units_to_chars, line_starts)
        if reason:
            status = "position_mismatch" if reason in {
                "coordinate_invalid", "coordinate_conflict", "text_mismatch"} else reason
            counts[status] += 1
            if statuses is not None:
                statuses.append((status, reason if status == "position_mismatch" else None))
            continue
        begin, finish = span
        if finish > visible_end:
            counts["outside_visible_source"] += 1
            if statuses is not None:
                statuses.append(("outside_visible_source", None))
            continue
        position = bisect_right(token_ends, begin)
        matched = 0
        while position < len(tokens):
            start, end, token_index = tokens[position]
            if start >= finish:
                break
            if end > begin:
                pairs.append((node_index, token_index))
                matched += 1
            position += 1
        status = "aligned_nodes" if matched else "no_visible_token"
        counts[status] += 1
        if statuses is not None:
            statuses.append((status, None))
    return pairs, counts


def coverage_summary(counts: Counter) -> dict:
    total = counts["total_nodes"]
    return dict(total_nodes=total, aligned_nodes=counts["aligned_nodes"],
                coverage=counts["aligned_nodes"] / total if total else 0.0,
                **{reason: counts[reason] for reason in (
                    "missing_position", "position_mismatch", "outside_visible_source",
                    "no_visible_token", "invalid_token_offsets")})


def region_candidates(record, view, encoded, regions, builder):
    """Visible operation-end states and distributions over C's existing categories.

    Overlapping operations retain the outermost complete operation. Exact
    duplicates are counted once; conflicts/cross-region duplicates are excluded.
    No node is removed from H. EMPTY=0 and UNK=1 are never reconstruction answers.
    """
    from collections import defaultdict
    from .cfg_data import FAMILIES
    if record["split"] not in {"train", "valid"}:
        raise ValueError("region targets are train/valid only")
    if not getattr(builder.tokenizer, "is_fast", False):
        raise ValueError("region targets require reliable fast-tokenizer offsets")
    source = record["raw_source"]
    tokens = builder.tokenizer(source, add_special_tokens=False, truncation=True,
        max_length=builder.source_max_length, return_offsets_mapping=True)
    statuses = []
    pairs, counts = align_nodes(source, view["locations"], tokens["offset_mapping"],
                                len(builder.source_prefix), statuses=statuses)
    ends = {}
    for node, token in pairs:
        ends[node] = max(token, ends.get(node, token))
    stats = Counter({"alignment_"+k: v for k, v in counts.items()})
    units, lines = _source_coordinates(source)
    spans = defaultdict(list)
    for node, values in enumerate(encoded):
        for family, value in zip(FAMILIES, values):
            stats[f"{family}_empty_nodes" if value == 0 else
                  f"{family}_unk_nodes" if value == 1 else f"{family}_known_nodes"] += 1
        if not any(value >= 2 for value in values):
            stats["no_known_target_nodes"] += 1
            continue
        if statuses[node][0] != "aligned_nodes":
            stats["target_excluded_" + statuses[node][0]] += 1
            for family, value in zip(FAMILIES, values):
                if value >= 2:
                    stats[f"{family}_target_excluded_{statuses[node][0]}"] += 1
            continue
        span, reason = _node_span(source, view["locations"][node], units, lines)
        if reason is not None:
            raise ValueError("alignment span changed")
        spans[span].append(node)
    kept, covered_end, used_tokens = [], -1, set()
    for span, nodes in sorted(spans.items(), key=lambda item: (item[0][0], -item[0][1])):
        signatures = {(regions["node_to_region"][node], tuple(encoded[node])) for node in nodes}
        if len(signatures) != 1:
            stats["ambiguous_duplicate_operations"] += len(nodes)
            continue
        stats["duplicate_operations"] += len(nodes)-1
        node = nodes[0]
        if span[0] < covered_end or ends[node] in used_tokens:
            stats["overlapping_operations"] += 1
            continue
        covered_end = span[1]
        used_tokens.add(ends[node])
        kept.append((node, span, ends[node]))
    grouped = defaultdict(list)
    for node, span, token in kept:
        grouped[regions["node_to_region"][node]].append((node, span, token))
    candidates = []
    for region, operations in grouped.items():
        targets = []
        for i, family in enumerate(FAMILIES):
            counts = Counter(encoded[node][i] for node, _, _ in operations if encoded[node][i] >= 2)
            total = sum(counts.values())
            targets.append([[category, count/total] for category, count in sorted(counts.items())])
            stats[f"{family}_supervisable_regions"] += bool(total)
            stats[f"{family}_nonconstant_regions"] += len(counts) > 1
        candidates.append({"region": region, "tokens": [t for _, _, t in operations],
                           "nodes": [n for n, _, _ in operations],
                           "spans": [list(s) for _, s, _ in operations], "targets": targets})
    candidates.sort(key=lambda item: item["spans"])
    stats["supervisable_regions"] = len(candidates)
    stats["supervisable_functions"] = bool(candidates)
    stats["supervisable_multinode_regions"] = sum(len(regions["members"][r["region"]]) > 1 for r in candidates)
    return candidates, stats


def region_targets(record, view, encoded, regions, builder):
    """Original seed-42 sampling, shared with the read-only candidate audit."""
    import random
    from .cfg_data import FAMILIES
    candidates, stats = region_candidates(record, view, encoded, regions, builder)
    indices = sorted(random.Random(f"42:{record['sample_key']}").sample(
        range(len(candidates)), min(8, len(candidates))))
    selected = [candidates[i] for i in indices]
    stats["supervisable_nonconstant_regions"] = sum(any(len(t) > 1 for t in r["targets"]) for r in candidates)
    stats["selected_nonconstant_regions"] = sum(any(len(t) > 1 for t in r["targets"]) for r in selected)
    stats["selected_multinode_regions"] = sum(len(regions["members"][r["region"]]) > 1 for r in selected)
    stats["selected_regions"] = len(selected)
    stats["selected_operations"] = sum(len(r["tokens"]) for r in selected)
    for family_index, family in enumerate(FAMILIES):
        stats[f"{family}_effective_functions"] = any(r["targets"][family_index] for r in selected)
        stats[f"{family}_selected_regions"] = sum(bool(r["targets"][family_index]) for r in selected)
    return selected, stats
