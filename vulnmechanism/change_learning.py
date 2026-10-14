"""Original repair-pair source checks and preparation; historical learners are archived."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import platform
import random
import re

from .prepare_benchmark import (
    NearIndex, SPLITS, jsonl_rows, normalized_source, overlap_report,
    prime_provenance, unit, validate,
)
from .syntax import identifier, parser_for, resolve_language, semantic_preserving_view, walk


from .artifacts import write_json

def read_rows(path):
    return [r for _, r in jsonl_rows(Path(path))]


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


@lru_cache(maxsize=16384)
def function_identity(source, language, file_name=""):
    language = resolve_language(source, language, file_name)
    encoded = source.encode()
    root = parser_for(language).parse(encoded).root_node
    functions = [n for n in walk(root) if n.type == "function_definition"]
    if len(functions) != 1:
        return None
    function = functions[0]
    body = function.child_by_field_name("body")
    declarator = function.child_by_field_name("declarator")
    name = identifier(declarator, encoded)
    if not name or body is None or declarator is None or declarator.has_error:
        return None
    # Exact tokenized signatures conservatively exclude overloads, unrelated
    # functions, and signature-changing fixes. Body changes are not constrained.
    signature = encoded[function.start_byte:body.start_byte].decode()
    return name, normalized_source(signature)


def check_pair(before, after):
    if (before["label"], after["label"]) != (1, 0):
        return "invalid_labels"
    if before["dataset"] != after["dataset"] or before["split"] != after["split"]:
        return "cross_partition_pair"
    if normalized_source(before["raw_source"]) == normalized_source(after["raw_source"]):
        return "unchanged_source"
    left, right = [function_identity(r["raw_source"], r["language"], r.get("file_name", ""))
                   for r in (before, after)]
    if left is None or right is None:
        return "identity_unresolved"
    if left[0] != right[0]:
        return "function_name_mismatch"
    if left != right:
        return "function_signature_mismatch"
    return "accepted"


def build_views(records):
    transforms, counts = [], defaultdict(Counter)
    for row in records:
        language = resolve_language(row["raw_source"], row["language"], row.get("file_name", ""))
        transformed, kind = semantic_preserving_view(row["raw_source"], language)
        counts[f'{row["dataset"]}/{row["split"]}'][kind] += 1
        if transformed is not None:
            transforms.append(dict(sample_key=row["sample_key"], dataset=row["dataset"],
                split=row["split"], label=row["label"], raw_source=transformed, transformation=kind))
    return transforms, {k: dict(v) for k, v in sorted(counts.items())}


def verify_paired_sources(rows, raw_dir):
    files = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["dataset"] in {"cleanvul", "sven"}:
            files[r["source_file"]][r["source_row"]].append(r)
    counts = Counter()
    for relative, wanted in files.items():
        path = Path(raw_dir) / relative
        with path.open(encoding="utf-8", newline="") as handle:
            iterator = enumerate(csv.DictReader(handle), 1) if path.suffix == ".csv" else (
                (i, json.loads(s)) for i, s in enumerate(handle, 1))
            found = set()
            for line, native in iterator:
                if line not in wanted:
                    continue
                for r in wanted[line]:
                    if r["dataset"] == "cleanvul":
                        if int(native["vulnerability_score"]) != 4:
                            raise ValueError(f"{path}:{line}: CleanVul score mismatch")
                        field = "func_before" if r["label"] else "func_after"
                    else:
                        field = "func_src_before" if r["label"] else "func_src_after"
                    if native[field] != r.get("raw_source", r.get("function")):
                        raise ValueError(f'{r["sample_key"]}: source does not match {path}:{line}')
                    counts[r["dataset"]] += 1
                found.add(line)
            if found != set(wanted):
                raise ValueError(f"{path}: unresolved native source rows")
    return dict(counts)


def prepare(manifest, raw_dir, output):
    """Keep classification holdouts; recover official within-split repair ends.

    Missing TRAIN repair ends join the common BCE cohort in every ablation.
    Missing VALID/TEST repair ends are pair-evaluation-only. All recovered ends
    must exist verbatim in their official non-paired split and pass the existing
    benchmark's duplicate and external-commit boundaries.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = read_rows(manifest)
    source_checks = verify_paired_sources(rows, raw_dir)
    grouped = defaultdict(list)
    for row in rows:
        if row["dataset"] not in {"primevul", "cleanvul", "sven"}:
            raise ValueError("unexpected benchmark dataset")
        if row["dataset"] == "cleanvul" and row.get("score") != 4:
            raise ValueError("CleanVul must have score=4")
        grouped[row.get("pair_id") or row["sample_key"]].append(row)
    units = [unit(part) for part in grouped.values()]
    print("Checking the existing strict benchmark (including exact near-duplicate join)", flush=True)
    checks = validate(units)
    overlaps = overlap_report(units)
    sven_commits = {r["commit"] for r in rows if r["dataset"] == "sven" and r["commit"]}
    shared = [r["sample_key"] for r in rows if r["split"] in {"train", "valid"}
              and any(r["commit"] and (r["commit"].startswith(c) or c.startswith(r["commit"]))
                      for c in sven_commits)]
    if shared:
        raise ValueError(f"train/valid shares SVEN commits: {shared[:5]}")
    records = [dict(r, raw_source=r["function"]) for r in rows]
    for r in records:
        del r["function"]
        r["classification_member"] = True
    members = {r["sample_key"]: r for r in records}
    pair_counts = defaultdict(Counter)
    pairs, rejected, seen_endpoints = [], [], defaultdict(set)
    pair_members = []
    near = NearIndex()
    for item in units:
        near.add(item)
    source_index = defaultdict(list)
    for row in records:
        source_index[normalized_source(row["raw_source"])].append(row)

    def admit(pair_id, before, after, provenance):
        partition = f'{before["dataset"]}/{before["split"]}'
        reason = check_pair(before, after)
        endpoints = {before["sample_key"], after["sample_key"]}
        if reason == "accepted" and endpoints & seen_endpoints[partition]:
            reason = "reused_endpoint"
        pair_counts[partition][reason] += 1
        item = dict(pair_id=pair_id, dataset=before["dataset"], split=before["split"],
                    before_key=before["sample_key"], after_key=after["sample_key"],
                    provenance=provenance)
        if reason == "accepted":
            seen_endpoints[partition].update(endpoints)
            pairs.append(item)
        else:
            rejected.append(dict(item, reason=reason))

    raw_counts = {}
    for split in reversed(SPLITS):
        print(f"Checking official PrimeVul {split} sources and repair ends", flush=True)
        path = Path(raw_dir) / "PrimeVul_v0.1" / f"primevul_{split}_paired.jsonl"
        official_pairs = list(jsonl_rows(path))
        wanted = {str(r["idx"]) for _, r in official_pairs}
        wanted.update(str(r["idx"]) for r in records if r["dataset"] == "primevul" and r["split"] == split)
        official = {}
        raw_labels = Counter()
        function_path = path.with_name(f"primevul_{split}.jsonl")
        for source_line, raw in jsonl_rows(function_path):
            raw_labels[raw["target"]] += 1
            if str(raw["idx"]) in wanted:
                official[str(raw["idx"])] = (source_line, raw)
        raw_counts[split] = {str(k): v for k, v in raw_labels.items()}
        for row in records:
            if row["dataset"] != "primevul" or row["split"] != split:
                continue
            entry = official.get(str(row["idx"]))
            if entry is None or (entry[1]["func"], entry[1]["target"]) != (row["raw_source"], row["label"]):
                raise ValueError(f'{row["sample_key"]}: manifest/source mismatch')
        iterator = iter(official_pairs)
        for line, before in iterator:
            following = next(iterator, None)
            if following is None or before["target"] != 1 or following[1]["target"] != 0:
                raise ValueError(f"{path}:{line}: invalid official adjacent pair")
            after = following[1]
            counts = pair_counts[f"primevul/{split}"]
            counts["source_pairs"] += 1
            keys = [f'primevul:{r["idx"]}' for r in (before, after)]
            present = [key in members for key in keys]
            counts[f"members_present_{sum(present)}"] += 1
            counts["retained_positive_anchors"] += present[0]
            if not present[0]:
                counts["positive_anchor_not_retained"] += 1
                continue
            a = members[keys[0]]
            if a["split"] != split:
                raise ValueError(f"{path}:{line}: official repair pair crosses retained splits")
            if (a["raw_source"], a["label"]) != (before["func"], before["target"]):
                raise ValueError(f"{path}:{line}: paired identity/source/label mismatch")
            entry = official.get(str(after["idx"]))
            if entry is None:
                counts["fixed_end_not_in_official_function_split"] += 1
                continue
            source_line, native = entry
            if (native["func"], native["target"]) != (after["func"], 0):
                raise ValueError(f"{path}:{line}: fixed end does not match native split")
            if present[1]:
                b = members[keys[1]]
                if b["split"] != split or (b["raw_source"], b["label"]) != (after["func"], 0):
                    raise ValueError(f"{path}:{line}: retained fixed end mismatch")
            else:
                repo, commit, provenance = prime_provenance(native)
                b = dict(dataset="primevul", sample_key=keys[1], idx=native["idx"],
                         raw_source=native["func"], label=0, split=split, official_split=split,
                         language=a["language"], file_name=native.get("file_name") or "",
                         repo=repo, commit=commit, source_file=str(function_path), source_row=source_line,
                         counterpart_groups=a.get("counterpart_groups", []),
                         classification_member=split == "train", repair_supplement=True, **provenance)
                reason = check_pair(a, b)
                if reason != "accepted":
                    counts[reason] += 1
                    rejected.append(dict(pair_id=f"primevul:{split}:paired:{line}", before_key=keys[0],
                                         after_key=keys[1], dataset="primevul", split=split, reason=reason))
                    continue
                if split in {"train", "valid"} and (
                    not re.fullmatch(r"[0-9a-f]{7,40}", commit)
                    or any(commit.startswith(c) or c.startswith(commit) for c in sven_commits)
                ):
                    counts["fixed_end_sven_commit_or_unresolved_revision"] += 1
                    continue
                normalized = normalized_source(b["raw_source"])
                if normalized in source_index:
                    counts["fixed_end_exact_or_normalized_duplicate"] += 1
                    continue
                candidate = unit([dict(b, function=b["raw_source"])])
                if near.matches(candidate):
                    counts["fixed_end_near_duplicate_outside_counterpart"] += 1
                    continue
                members[b["sample_key"]] = b
                pair_members.append(b)
                near.add(candidate)
                source_index[normalized].append(b)
                counts["recovered_fixed_ends"] += 1
            counts["project_metadata_mismatch"] += before.get("project") != after.get("project")
            counts["commit_metadata_mismatch"] += before.get("commit_id") != after.get("commit_id")
            admit(f"primevul:{split}:paired:{line}", a, b,
                  dict(source_file=str(path), source_row=line, evidence="official_adjacent_vulnerable_fixed_pair"))
    for part in grouped.values():
        if part[0]["dataset"] == "primevul":
            continue
        before, after = sorted((members[r["sample_key"]] for r in part), key=lambda r: -r["label"])
        counts = pair_counts[f'{before["dataset"]}/{before["split"]}']
        counts["source_pairs"] += 1
        counts["members_present_2"] += 1
        admit(before["pair_id"], before, after,
              dict(source_file=before["source_file"], source_row=before["source_row"],
                   evidence="score4_fixing_commit_pair" if before["dataset"] == "cleanvul" else "sven_security_fix_pair"))
    # Every method sees the same recovered train labels via ordinary BCE. Holdout
    # function counts stay identical to the existing strict benchmark.
    records.extend(r for r in pair_members if r["split"] == "train")
    transforms, transform_counts = build_views(records)
    counts = Counter(f'{r["dataset"]}/{r["split"]}/label_{r["label"]}' for r in records)
    augmented = [unit([dict(r, function=r["raw_source"])])
                 for r in records + [p for p in pair_members if p["split"] != "train"]
                 if r["dataset"] == "primevul"]
    augmented += [u for u in units if u["dataset"] != "primevul"]
    augmented_overlaps = overlap_report(augmented)
    if any(not name.endswith(": repo_commit") for name in augmented_overlaps):
        raise ValueError("recovered repair ends introduce cross-partition source/counterpart leakage")
    report = dict(manifest=str(manifest), raw_dir=str(raw_dir), seed=42,
                  policy="Existing score=4 SVEN-commit-disjoint benchmark and unchanged classification holdouts. Official train fixed ends recovered for common BCE; Valid/Test fixed ends are pair-only. No resplit or CPG success filter.",
                  original_classification_samples=dict(Counter(f'{r["dataset"]}/{r["split"]}/label_{r["label"]}' for r in rows)),
                  official_primevul_function_counts=raw_counts,
                  native_paired_source_rows_verified=source_checks,
                  samples=dict(sorted(counts.items())), original_manifest_checks=checks,
                  shared_train_valid_sven_commits=len(shared), cross_partition_overlaps=overlaps,
                  augmented_cross_partition_overlaps=augmented_overlaps,
                  recovered_end_checks=dict(cross_partition_source_or_counterpart_leakage=0,
                      train_valid_sven_commit_overlap=0, duplicate_gate="existing NearIndex Jaccard >=0.90 and exact/normalized source; native within-split counterpart exceptions retained"),
                  repair_pairs={k: dict(v) for k, v in sorted(pair_counts.items())},
                  transformations=transform_counts,
                  pair_admission="Retained positive anchor; exact official same-split fixed end; native vulnerable/fixed labels; same function name and signature; changed source; existing duplicate/commit boundaries; no reused endpoint",
                  transformation="One verified scope-bound rename, or inline comment preserving preprocessing tokens and line count; parse errors excluded from robustness denominator, never from BCE")
    output.mkdir(parents=True)
    write_rows(output / "records.jsonl", records)
    write_rows(output / "pair_members.jsonl", pair_members)
    write_rows(output / "pairs.jsonl", pairs)
    write_rows(output / "rejected_pairs.jsonl", rejected)
    write_rows(output / "transforms.jsonl", transforms)
    write_json(output / "statistics.json", report)
    return report
