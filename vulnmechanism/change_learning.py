"""Source-only invariance and real-repair ranking on the existing benchmark."""
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
from .syntax import identifier, parser_for, resolve_language, semantic_preserving_view, walk, repair_candidate_lines


OBJECTIVES = ("bce", "bce_invariance", "bce_ranking", "full", "bce_random_ranking")
LOCAL_OBJECTIVES = ('A', 'B', 'C', 'D', 'C_random')
MODELS = {
    "codebert": "/home/PublicData/PHY-data/resource/codebert-base",
    "qwen": "/home/phy/models/Qwen2.5-Coder-7B-Instruct",
}


def read_rows(path):
    return [r for _, r in jsonl_rows(Path(path))]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


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


def random_pair_control(pairs, seed):
    """Derange the ACTUALLY sampled pairs, preserving endpoint multiplicities."""
    if len(pairs) < 2:
        raise ValueError("random ranking control requires at least two sampled pairs per epoch")
    groups = defaultdict(list)
    for i, pair in enumerate(pairs):
        groups[pair[1]["sample_key"]].append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    order = [i for key in keys for i in groups[key]]
    shift = max(map(len, groups.values()))
    if shift * 2 > len(order):
        raise ValueError("sampled endpoint multiplicities do not permit an exact derangement")
    shifted = order[shift:] + order[:shift]
    negatives = dict(zip(order, shifted))
    return [(pairs[i][0], pairs[negatives[i]][1]) for i in range(len(pairs))]


def validate_supervision(train_records, pairs, views):
    members = {r["sample_key"]: r for r in train_records}
    if len(members) != len(train_records) or not pairs or not views:
        raise ValueError("change learning requires unique training members, real pairs and nonempty safe views")
    endpoints = [[], []]
    for pair in pairs:
        if len(pair) != 2 or check_pair(*pair) != "accepted":
            raise ValueError("ranking requires verified vulnerable/fixed function pairs")
        for i, row in enumerate(pair):
            if row.get("split") != "train" or members.get(row["sample_key"]) != row:
                raise ValueError("repair supervision must be an exact training member")
            endpoints[i].append(row["sample_key"])
    if any(len(set(keys)) != len(keys) for keys in endpoints):
        raise ValueError("repair endpoints must be unique for a marginal-matched random control")
    for original, transformed in views:
        if members.get(original["sample_key"]) != original or original.get("split") != "train":
            raise ValueError("invariance supervision must be an exact training member")
        if any(original[k] != transformed[k] for k in ("sample_key", "dataset", "split", "label")):
            raise ValueError("invariance view identity/label mismatch")
        language = resolve_language(original["raw_source"], original["language"], original.get("file_name", ""))
        expected, _ = semantic_preserving_view(original["raw_source"], language)
        if expected is None or expected != transformed["raw_source"]:
            raise ValueError("invariance view is not the verified semantic-preserving transformation")


def invariance_loss(logits):
    """Jensen-Shannon divergence of two Bernoulli predictions, in nats."""
    if logits.shape != (2,):
        raise ValueError("invariance requires two logits")
    return bernoulli_js(logits[0], logits[1]).mean()


def bernoulli_js(left, right):
    import torch
    from torch.nn import functional as F
    if left.shape != right.shape:
        raise ValueError('Bernoulli JS shape mismatch')
    logits = torch.stack((left.float(), right.float()))
    log_p = torch.stack((F.logsigmoid(-logits), F.logsigmoid(logits)), dim=-1)
    log_m = torch.logsumexp(log_p, dim=0) - math.log(2)
    return (log_p.exp() * (log_p - log_m)).sum(dim=-1).mean(dim=0)


def line_invariance_loss(local):
    if len(local) != 2 or local[0][0] != local[1][0] or not local[0][0]:
        raise ValueError('local invariance requires matching nonempty original physical lines')
    return bernoulli_js(local[0][1], local[1][1]).mean()


def region_loss(scores, line_numbers, candidates):
    import torch
    if scores.shape != (len(line_numbers),) or not line_numbers:
        raise ValueError('region scores/line numbers mismatch or m=0')
    candidates = set(candidates)
    selected = [i for i, line in enumerate(line_numbers) if line in candidates]
    if not selected or len(selected) == len(line_numbers):
        return None
    return torch.logsumexp(scores.float(), 0) - torch.logsumexp(scores[selected].float(), 0)


def validate_local_supervision(train_records, regions, views):
    members = {r['sample_key']: r for r in train_records}
    if len(members) != len(train_records) or not isinstance(regions, dict) or views is None:
        raise ValueError('local supervision requires unique training members and explicit regions/views')
    for key, lines in regions.items():
        if key not in members or members[key]['label'] != 1 or members[key]['split'] != 'train':
            raise ValueError('regions must belong to vulnerable training functions')
        if not all(type(line) is int and line > 0 for line in lines) or len(lines) != len(set(lines)):
            raise ValueError('candidate lines must be unique original positive line numbers')
    for original, transformed in views:
        if members.get(original['sample_key']) != original or original['split'] != 'train':
            raise ValueError('local invariance must use original training members')
        if any(original[key] != transformed[key] for key in ('sample_key', 'dataset', 'split', 'label')):
            raise ValueError('local invariance identity mismatch')
        language = resolve_language(original['raw_source'], original['language'], original.get('file_name', ''))
        expected, _ = semantic_preserving_view(original['raw_source'], language)
        if expected != transformed['raw_source'] or expected is None:
            raise ValueError('local invariance requires the existing verified transformation')


def prepare_local(data_dir):
    """Normal training cache: full-source candidates from training repairs only."""
    path = Path(data_dir) / 'local_candidates.jsonl'
    cached = read_rows(path) if path.exists() else []
    if any(r.get('candidate_version') != 1 or r['split'] != 'train' for r in cached):
        raise ValueError('local candidate cache configuration mismatch')
    done = {(r['before_key'], r['after_key']) for r in cached}
    if len(done) != len(cached):
        raise ValueError('duplicate candidate cache repair identities')
    for dataset in ('primevul', 'cleanvul'):
        _, pairs, _ = supervision(data_dir, dataset)
        expected = {(a['sample_key'], b['sample_key']) for a, b in pairs}
        existing = {(r['before_key'], r['after_key']) for r in cached if r['dataset'] == dataset}
        if not existing <= expected:
            raise ValueError('candidate cache has unknown training repairs')
        print(f'Local candidates {dataset}: {len(existing)}/{len(pairs)} cached', flush=True)
        with path.open('a', encoding='utf-8') as handle:
            for index, (before, after) in enumerate(pairs):
                if (before['sample_key'], after['sample_key']) in done:
                    continue
                item = training_candidate(before, after)
                cached.append(item)
                handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
                handle.flush()
                if (index+1) % 250 == 0:
                    print(f'Local candidates {dataset}: {index+1}/{len(pairs)}', flush=True)
    counts = {}
    for dataset in ('primevul', 'cleanvul'):
        part = [r for r in cached if r['dataset'] == dataset]
        counts[dataset] = dict(pairs=len(part), nonempty_candidates=sum(bool(r['lines']) for r in part),
            rules=dict(Counter(event['rule'] for r in part for event in r['events'])),
            skip_reasons=dict(Counter(reason for r in part for reason in r['skip_reasons'])))
    return dict(cache=str(path), training_only=True, coverage=counts)


def training_candidate(before, after):
    if check_pair(before, after) != 'accepted' or before['split'] != 'train':
        raise ValueError('local candidate source is not a verified training repair')
    language = resolve_language(before['raw_source'], before['language'], before.get('file_name', ''))
    result = repair_candidate_lines(before['raw_source'], after['raw_source'], language)
    return dict(candidate_version=1, dataset=before['dataset'], split='train',
                before_key=before['sample_key'], after_key=after['sample_key'], **result)


def _local_data(args, model_path):
    from transformers import AutoTokenizer
    from .model import InputBuilder
    rows, pairs, views = supervision(args.data_dir, args.source_dataset)
    rows = [r for r in rows if r['split'] in {'train', 'valid'}]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    builder = InputBuilder(tokenizer, source_max_length=2048, context_max_length=384)
    if args.smoke_samples:
        if args.smoke_samples < 4:
            raise ValueError('local smoke needs at least four training functions')
        by_view = {a['sample_key']: b for a, b in views}
        eligible = None
        for a, b in pairs:
            candidate = training_candidate(a, b)
            original = builder.local_layout(a['raw_source'], args.encoder)
            fixed = builder.local_layout(b['raw_source'], args.encoder)
            changed = by_view.get(a['sample_key'])
            if changed is None:
                continue
            transformed = builder.local_layout(changed['raw_source'], args.encoder)
            selected = set(candidate['lines']) & set(original['full_lines'])
            if (original['ids'] != fixed['ids'] and 0 < len(selected) < len(original['lines'])
                    and original['complete'] and transformed['complete'] and original['ids'] != transformed['ids']):
                eligible = (a, b)
                break
        if eligible is None:
            raise ValueError('no real smoke sample with both region and local invariance supervision')
        keep = {r['sample_key'] for r in eligible}
        for r in rows:
            if r['split'] == 'train' and len(keep) < args.smoke_samples:
                keep.add(r['sample_key'])
        valid = {next(r['sample_key'] for r in rows if r['split']=='valid' and r['label']==y) for y in (0,1)}
        rows = [r for r in rows if r['sample_key'] in keep | valid]
        pairs = [p for p in pairs if all(r['sample_key'] in keep for r in p)]
        views = [v for v in views if v[0]['sample_key'] in keep]
        cache = [training_candidate(a,b) for a,b in pairs]
    else:
        prepare_local(args.data_dir)
        cache = [r for r in read_rows(Path(args.data_dir)/'local_candidates.jsonl') if r['dataset'] == args.source_dataset]
    layouts = {r['sample_key']: builder.local_layout(r['raw_source'], args.encoder) for r in rows}
    by_pair = {(r['before_key'], r['after_key']): r for r in cache}
    if set(by_pair) != {(a['sample_key'], b['sample_key']) for a, b in pairs}:
        raise ValueError('training repair/candidate cache membership mismatch')
    union, all_lines, events, identical = defaultdict(set), defaultdict(set), defaultdict(list), Counter()
    visible_pairs = []
    for a, b in pairs:
        candidate = by_pair[(a['sample_key'], b['sample_key'])]
        all_lines[a['sample_key']].update(candidate['lines'])
        events[a['sample_key']].append(candidate)
        if layouts[a['sample_key']]['ids'] != layouts[b['sample_key']]['ids']:
            visible_pairs.append((a, b))
            union[a['sample_key']].update(candidate['lines'])
        else:
            identical[a['sample_key']] += 1
    regions, region_cache = {}, []
    for key in all_lines:
        raw_lines = union[key]
        layout = layouts[key]
        selected = sorted(raw_lines & set(layout['full_lines']))
        reason = ('input_identical_repairs' if not raw_lines and identical[key] == len(events[key]) else
                  'empty_visible_M' if not selected else 'all_visible_instances' if len(selected) == len(layout['lines']) else None)
        if reason is None:
            regions[key] = selected
        region_cache.append(dict(sample_key=key, original_lines=sorted(raw_lines), visible_lines=selected,
            all_repair_candidate_lines=sorted(all_lines[key]), input_identical_repairs=identical[key],
            effective_instances=len(layout['lines']), skip_reason=reason, repairs=events[key]))
    valid_views, view_counts = [], Counter()
    for original, transformed in views:
        before = layouts[original['sample_key']]
        after = builder.local_layout(transformed['raw_source'], args.encoder)
        if not before['complete'] or not after['complete']:
            view_counts['truncated'] += 1
        elif before['ids'] == after['ids']:
            view_counts['input_identical'] += 1
        else:
            if before['lines'] != after['lines']:
                raise ValueError('verified transformation changed effective original line numbers')
            valid_views.append((original, transformed))
            view_counts['eligible'] += 1
    if args.smoke_samples:
        if args.smoke_samples < 4:
            raise ValueError('local smoke needs at least four training functions')
        view_keys = {a['sample_key'] for a, _ in valid_views}
        eligible = next((p for p in visible_pairs if p[0]['sample_key'] in regions and p[0]['sample_key'] in view_keys), None)
        if eligible is None:
            raise ValueError('no real smoke sample with both region and local invariance supervision')
        keep = {r['sample_key'] for r in eligible}
        for r in rows:
            if r['split'] == 'train' and len(keep) < args.smoke_samples:
                keep.add(r['sample_key'])
        valid = {next(r['sample_key'] for r in rows if r['split'] == 'valid' and r['label'] == y) for y in (0, 1)}
        rows = [r for r in rows if r['sample_key'] in keep | valid]
        regions = {key: value for key, value in regions.items() if key in keep}
        valid_views = [v for v in valid_views if v[0]['sample_key'] in keep]
        region_cache = [r for r in region_cache if r['sample_key'] in keep]
        visible_pairs = [p for p in visible_pairs if p[0]['sample_key'] in keep and p[1]['sample_key'] in keep]
    coverage = dict(training_functions=sum(r['split'] == 'train' for r in rows),
        raw_repair_pairs=len(pairs), input_visible_repair_pairs=len(visible_pairs),
        input_identical_repair_pairs=len(pairs)-len(visible_pairs),
        region_functions=len(regions), region_function_coverage=len(regions)/sum(r['split'] == 'train' for r in rows),
        vulnerable_function_coverage=len(regions)/sum(r['split']=='train' and r['label']==1 for r in rows),
        region_skip_reasons=dict(Counter(r['skip_reason'] for r in region_cache if r['skip_reason'])),
        raw_safe_views=len(views), local_invariance_views=len(valid_views), view_filter_counts=dict(view_counts),
        token_budget='CodeBERT logical total 2048; Qwen source 2048',
        source_token_budget=2048-len(builder.source_prefix)-2 if args.encoder == 'codebert' else 2048,
        full_source_candidates_before_visibility=True)
    return rows, regions, valid_views, region_cache, coverage


def train_local(args):
    import torch
    import transformers
    from .model import train_model
    if (args.seed, args.source_max_length, args.epochs, args.lora_r, args.lora_alpha, args.lora_dropout,
            args.weight_decay, args.batch_size*args.gradient_accumulation) != (42, 2048, 3, 16, 32, .05, .01, 8):
        raise ValueError('local risk requires the specified seed, 2048 budget, 3 epochs, LoRA and effective batch')
    model_path = args.model or MODELS[args.encoder]
    rows, regions, views, region_cache, coverage = _local_data(args, model_path)
    root = Path(args.output_dir) / args.encoder / args.source_dataset
    root.mkdir(parents=True, exist_ok=True)
    config = dict(encoder=args.encoder, trained_on=args.source_dataset, model_path=model_path, seed=42,
        source_max_length=2048, epochs=3, initial_bce_epochs=3, learning_rate=args.learning_rate,
        batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
        lora_r=16, lora_alpha=32, lora_dropout=.05, weight_decay=.01,
        head_learning_rate=1e-4, lora_learning_rate=2e-5, region_weight=.1, invariance_weight=.01,
        warmup_epochs=1, joint_epochs=2, region_ramp_epochs=1, hidden_dim=64,
        data_dir=str(args.data_dir), smoke_only=bool(args.smoke_samples),
        train_samples=sum(r['split']=='train' for r in rows), valid_samples=sum(r['split']=='valid' for r in rows),
        candidate_coverage=coverage, input_coverage=coverage,
        selection='corresponding Valid MCC, existing F1/accuracy/AUC tie-break and threshold selection',
        budget='all functions BCE; 1 frozen-encoder head epoch, 2 joint epochs; two zero/dropout-free auxiliary encodings per joint step in all four groups',
        environment=dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
            device=args.device, gpu=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None))
    if args.smoke_samples:
        config['smoke_members'] = [r['sample_key'] for r in rows]
    config_path = root/'training.metrics.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('existing local-risk configuration differs')
    write_json(config_path, config)
    write_rows(root/'candidates.jsonl', region_cache)
    initial_path = root/'initial_bce'/'best.pt'
    common = dict(records=rows, variant='codebert_source' if args.encoder=='codebert' else 'baseline',
        model_path=model_path, source_max_length=2048, batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation, epochs=3, seed=42,
        learning_rate=args.learning_rate, weight_decay=.01, lora_r=16, lora_alpha=32, lora_dropout=.05,
        device=args.device, log_every=args.log_every)
    if not initial_path.exists():
        train_model(None, initial_path, **common)
    members = {split: [(r['sample_key'], r['label']) for r in rows if r['split']==split] for split in ('train', 'valid')}
    results = {}
    for objective in ([args.objective] if args.objective else LOCAL_OBJECTIVES[:4]):
        selected_regions = regions
        if objective == 'C_random':
            from .model import InputBuilder
            from transformers import AutoTokenizer
            builder = InputBuilder(AutoTokenizer.from_pretrained(model_path), source_max_length=2048, context_max_length=384)
            generator = random.Random(42)
            selected_regions = {r['sample_key']: sorted(generator.sample(builder.local_layout(r['raw_source'], args.encoder)['full_lines'],
                len(regions[r['sample_key']]))) for r in rows if r['sample_key'] in regions}
        target = root/objective/'best.pt'
        if target.exists():
            checkpoint = torch.load(target, map_location='cpu', weights_only=False)
            if checkpoint.get('local_supervision', {}).get('objective') != objective or checkpoint.get('cohort_members') != members:
                raise ValueError('existing local checkpoint objective/cohort mismatch')
        else:
            checkpoint = train_model(None, target, **common, initial_checkpoint=initial_path,
                classifier_head='linear' if objective=='A' else 'local_risk', local_objective=objective,
                local_regions=selected_regions, invariant_views=views)
        results[objective] = dict(selected_epoch=checkpoint['selected_epoch'], threshold=checkpoint['decision_threshold'],
                                  validation=checkpoint['validation'])
        write_json(root/'comparison.valid.metrics.json', results)
    if args.smoke_samples and 'D' in results:
        verify_local_smoke(root, args, rows, regions, views)
    return results


def verify_local_smoke(root, args, rows, regions, views):
    """Real-model, real-source auxiliary-only updates plus strict save/reload."""
    import gc
    import tempfile
    import torch
    from transformers import AutoTokenizer
    from . import model as core
    checkpoint = torch.load(root/'D'/'best.pt', map_location='cpu', weights_only=False)
    device = core._resolve_device(args.device)
    original, transformed = next(v for v in views if v[0]['sample_key'] in regions)
    builder = core.InputBuilder(AutoTokenizer.from_pretrained(checkpoint['model_path']), source_max_length=2048, context_max_length=384)
    options = dict(variant=checkpoint['variant'], excluded_groups=(), device=device)

    def probe():
        model = core._load_model(checkpoint, device=device)
        if device.type == 'cuda':
            model.encoder.gradient_checkpointing_enable()
            model.encoder.enable_input_require_grads()
        measures = {}
        for kind, weight in (('region', .1), ('local_invariance', .01)):
            core._restore_adapter(model.encoder, checkpoint['adapter_state'])
            model.task_modules.load_state_dict(checkpoint['task_state'], strict=True)
            model.zero_grad(set_to_none=True)
            before = core._cpu_state(model.task_modules.state_dict())
            with core._auxiliary_mode(model):
                if kind == 'region':
                    _, local = core._forward_batch(model, [original], builder, return_local=True, **options)
                    loss = region_loss(local[0][1], local[0][0], regions[original['sample_key']])
                else:
                    _, local = core._forward_batch(model, [original, transformed], builder, return_local=True, **options)
                    loss = line_invariance_loss(local)
                if loss is None or not torch.isfinite(loss) or float(loss.detach()) <= 0:
                    raise ValueError(f'real smoke has no positive {kind} supervision')
                (weight*loss).backward()
            head_grad = sum(float(p.grad.detach().float().abs().sum()) for p in model.task_modules.parameters() if p.grad is not None)
            lora_grad = sum(float(p.grad.detach().float().abs().sum()) for p in model.encoder.parameters() if p.requires_grad and p.grad is not None)
            if head_grad <= 0 or lora_grad <= 0:
                raise ValueError(f'real smoke {kind} did not reach both risk head and LoRA')
            optimizer = torch.optim.AdamW([
                dict(params=[p for p in model.encoder.parameters() if p.requires_grad], lr=2e-5),
                dict(params=model.task_modules.parameters(), lr=1e-4)], weight_decay=0.)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()
            after = core._cpu_state(model.task_modules.state_dict())
            changed = sum(float((after[key]-before[key]).abs().sum()) for key in after)
            if changed <= 0:
                raise ValueError(f'real smoke {kind} caused no risk-head update')
            measures[kind] = dict(loss=float(loss.detach()), weighted_loss=float(weight*loss.detach()),
                                 head_gradient_l1=head_grad, lora_gradient_l1=lora_grad, head_update_l1=changed)
        model.eval()
        with torch.no_grad():
            expected = core._forward_batch(model, [original], builder, **options).float().cpu()
        updated = dict(checkpoint, adapter_state=core._cpu_state(core.get_peft_model_state_dict(model.encoder)),
                       task_state=core._cpu_state(model.task_modules.state_dict()))
        return updated, expected, measures

    updated, expected, measures = probe()
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)/'smoke.pt'
        torch.save(updated, target)
        restored = core.predict_checkpoint(target, [original], device=args.device, batch_size=1)
    torch.testing.assert_close(restored, expected, rtol=1e-5, atol=1e-5)
    value = dict(smoke_only=True, encoder=args.encoder, trained_on=args.source_dataset,
        sample_key=original['sample_key'], candidate_lines=regions[original['sample_key']],
        original_line_alignment=True, dropout=False, auxiliary_updates=measures, save_reload=True,
        input='single function only; the patch is not an encoder input')
    write_json(root/'smoke.metrics.json', value)
    return value


def ranking_loss(logits):
    from .cgvl import local_rank_loss
    if logits.shape != (2,):
        raise ValueError("ranking requires vulnerable then fixed logits")
    return local_rank_loss(logits[1:].float(), logits[:1].float())


def supervision(data_dir, dataset, split="train"):
    root = Path(data_dir)
    rows = [r for r in read_rows(root / "records.jsonl") if r["dataset"] == dataset]
    members = {r["sample_key"]: r for r in rows}
    members.update({r["sample_key"]: r for r in read_rows(root / "pair_members.jsonl") if r["dataset"] == dataset})
    pairs = [(members[p["before_key"]], members[p["after_key"]])
             for p in read_rows(root / "pairs.jsonl") if p["dataset"] == dataset and p["split"] == split]
    views = [(members[v["sample_key"]], dict(members[v["sample_key"]], raw_source=v["raw_source"],
                                            transformation=v["transformation"]))
             for v in read_rows(root / "transforms.jsonl") if v["dataset"] == dataset and v["split"] == split]
    return rows, pairs, views


def input_coverage(rows, pairs, views, encoder, model_path, source_max_length):
    from transformers import AutoTokenizer
    from .model import InputBuilder
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    builder = InputBuilder(tokenizer, source_max_length=source_max_length, context_max_length=384)
    budget = source_max_length - len(builder.source_prefix) - 2 if encoder == "codebert" else source_max_length
    ids, counts = {}, defaultdict(Counter)
    for r in rows:
        tokens = tokenizer(r["raw_source"], add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        ids[r["sample_key"]] = tokens[:budget]
        counts[r["split"]]["functions"] += 1
        counts[r["split"]]["source_truncated"] += len(tokens) > budget
    visible_pairs = [p for p in pairs if ids[p[0]["sample_key"]] != ids[p[1]["sample_key"]]]
    visible_views = [v for v in views if ids[v[0]["sample_key"]] != builder._encode(v[1]["raw_source"], budget)]
    return visible_pairs, visible_views, dict(source_token_budget=budget,
        total_encoder_budget=source_max_length if encoder == "codebert" else budget + len(builder.source_prefix) + 1,
        function_coverage={k: dict(v) for k, v in counts.items()},
        raw_repair_pairs=len(pairs), input_visible_repair_pairs=len(visible_pairs),
        token_identical_repair_pairs=len(pairs) - len(visible_pairs),
        raw_safe_views=len(views), input_visible_safe_views=len(visible_views),
        token_identical_safe_views=len(views) - len(visible_views))


def train(args):
    import torch
    import transformers
    from .model import train_model
    if args.seed != 42:
        raise ValueError("this experiment fixes seed=42")
    if args.encoder == "qwen" and args.smoke_samples == 0 and args.device == "cpu":
        raise ValueError("formal Qwen suite requires an explicitly available CUDA device")
    rows, pairs, views = supervision(args.data_dir, args.source_dataset)
    if args.smoke_samples:
        if args.smoke_samples < 4:
            raise ValueError("smoke needs at least four training functions")
        keep = {r["sample_key"] for p in pairs[:2] for r in p}
        # Cover real ranking and an actual safe transformation in the same smoke.
        keep.update(r["sample_key"] for r, _ in views[:1])
        for row in rows:
            if row["split"] == "train" and len(keep) < args.smoke_samples:
                keep.add(row["sample_key"])
        valid = {next(r["sample_key"] for r in rows if r["split"] == "valid" and r["label"] == y)
                 for y in (0, 1)}
        rows = [r for r in rows if r["sample_key"] in keep | valid]
        pairs = [p for p in pairs if all(r["sample_key"] in keep for r in p)]
        views = [v for v in views if v[0]["sample_key"] in keep]
    # Tests and SVEN cannot reach the training or selection call.
    rows = [r for r in rows if r["split"] in {"train", "valid"}]
    pairs, views, coverage = input_coverage(rows, pairs, views, args.encoder,
        args.model or MODELS[args.encoder], args.source_max_length)
    validate_supervision([r for r in rows if r["split"] == "train"], pairs, views)
    objectives = [args.objective] if args.objective else OBJECTIVES
    root = Path(args.output_dir) / args.encoder / args.source_dataset
    root.mkdir(parents=True, exist_ok=True)
    config = dict(encoder=args.encoder, trained_on=args.source_dataset,
                  model_path=args.model or MODELS[args.encoder], seed=42,
                  source_max_length=args.source_max_length, epochs=args.epochs,
                  batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
                  learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                  lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                  invariance_weight=args.invariance_weight, ranking_weight=args.ranking_weight,
                  selection="corresponding Valid MCC; F1/accuracy/AUC tie-break; threshold by existing Valid-only selector",
                  data_dir=str(args.data_dir), smoke_only=bool(args.smoke_samples),
                  train_samples=sum(r["split"] == "train" for r in rows),
                  valid_samples=sum(r["split"] == "valid" for r in rows),
                  repair_pairs=len(pairs), safe_views=len(views),
                  input_coverage=coverage,
                  budget="All objectives: identical BCE cohort and order; one deterministic original/transformed pair and one ranking pair per optimizer step, including zero-weight controls",
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   transformers=transformers.__version__, device=args.device,
                                   gpu=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None))
    if args.smoke_samples:
        config["smoke_members"] = [r["sample_key"] for r in rows]
    config_path = root / "training.metrics.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("existing suite configuration differs; use a separate experiment directory")
    write_json(config_path, config)
    comparison_path = root / "comparison.valid.metrics.json"
    result = json.loads(comparison_path.read_text()) if comparison_path.exists() else {}
    for objective in objectives:
        target = root / objective / "best.pt"
        if target.exists():
            checkpoint = torch.load(target, weights_only=False, map_location="cpu")
            if checkpoint.get("change_supervision", {}).get("objective") != objective:
                raise ValueError("existing checkpoint objective mismatch")
            if objective == "bce_random_ranking" and checkpoint["change_supervision"].get("random_sampling_protocol") != "epoch_marginal_matched":
                raise ValueError("existing random control lacks epoch-matched endpoints; replace this invalid control before evaluation")
        else:
            checkpoint = train_model(None, target, records=rows,
                variant="codebert_source" if args.encoder == "codebert" else "baseline",
                model_path=config["model_path"], seed=42, source_max_length=args.source_max_length,
                batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
                epochs=args.epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                device=args.device, log_every=args.log_every,
                change_objective=objective, repair_pairs=pairs, invariant_views=views,
                invariance_weight=args.invariance_weight, ranking_weight=args.ranking_weight)
        result[objective] = dict(selected_epoch=checkpoint["selected_epoch"],
                                 threshold=checkpoint["decision_threshold"],
                                 validation=checkpoint["validation"])
        write_json(root / "comparison.valid.metrics.json", result)
    return result


def score_metrics(rows, logits, threshold, pairs, views, view_logits, pair_rows=(), pair_logits=None,
                  *, input_identical_pairs=()):
    import numpy as np
    import torch
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve
    from .cfg_metrics import decision_boundary, metrics
    if len(rows) != len(logits) or len(views) != len(view_logits):
        raise ValueError("prediction alignment mismatch")
    if not torch.isfinite(logits).all() or not torch.isfinite(view_logits).all():
        raise ValueError("nonfinite evaluation logits")
    scores = logits.sigmoid().tolist()
    labels = [r["label"] for r in rows]
    result = metrics(labels, scores, threshold)
    if len(set(labels)) == 2:
        precision, recall, _ = precision_recall_curve(labels, scores)
        result["pr_auc"] = float(auc(recall, precision))
        result["average_precision"] = float(average_precision_score(labels, scores))
    else:
        result["pr_auc"] = result["average_precision"] = None
    result["bce"] = float(np.mean(np.logaddexp(0, logits.numpy().astype(float)) - np.array(labels) * logits.numpy()))
    boundary = decision_boundary(threshold)
    all_rows = rows + list(pair_rows)
    all_logits = logits if pair_logits is None else torch.cat((logits, pair_logits))
    if len(all_rows) != len(all_logits) or not torch.isfinite(all_logits).all():
        raise ValueError("pair prediction alignment mismatch")
    all_scores = all_logits.sigmoid().tolist()
    lookup = {r["sample_key"]: i for i, r in enumerate(all_rows)}
    if len(lookup) != len(all_rows):
        raise ValueError("duplicate evaluation identity")
    input_identical_pairs = set(input_identical_pairs)
    if not input_identical_pairs <= {(a["sample_key"], b["sample_key"]) for a, b in pairs}:
        raise ValueError("input-identical pair identity mismatch")
    correct, ordered, ties = 0, 0, 0
    for before, after in pairs:
        a, b = lookup[before["sample_key"]], lookup[after["sample_key"]]
        # Identical encoder inputs cannot discriminate a repair. Different batch
        # padding/kernels can still produce small numerical logit differences.
        if (before["sample_key"], after["sample_key"]) in input_identical_pairs:
            ties += 1
            continue
        correct += all_scores[a] >= boundary and all_scores[b] < boundary
        ordered += float(all_logits[a]) > float(all_logits[b])
        ties += float(all_logits[a]) == float(all_logits[b])
    result["repair"] = dict(pairs=len(pairs), pairwise_correct=correct / len(pairs) if pairs else None,
                            ranking_accuracy=ordered / len(pairs) if pairs else None, ties=ties,
                            input_identical_pairs=len(input_identical_pairs),
                            tie_definition="Identical encoder inputs or exactly equal logits; ties count as incorrect")
    consistent, robust, original_correct = 0, 0, 0
    by_kind = defaultdict(Counter)
    for (original, transformed), z in zip(views, view_logits):
        old = all_scores[lookup[original["sample_key"]]] >= boundary
        new = float(z.sigmoid()) >= boundary
        consistent += old == new
        original_correct += old == bool(original["label"])
        robust += old == new == bool(original["label"])
        group = by_kind[transformed.get("transformation", "rename")]
        group["eligible"] += 1
        group["consistent"] += old == new
        group["robust_correct"] += old == new == bool(original["label"])
    n = len(views)
    result["robustness"] = dict(eligible=n, total=len(rows), coverage=n / len(rows) if rows else 0,
        prediction_consistency=consistent / n if n else None,
        robust_accuracy=robust / n if n else None,
        original_accuracy_same_subset=original_correct / n if n else None,
        definition="Robust Accuracy: both original and transformed predictions correct; denominator is eligible transformations only",
        by_transformation={k: dict(eligible=v["eligible"], prediction_consistency=v["consistent"] / v["eligible"],
                                  robust_accuracy=v["robust_correct"] / v["eligible"]) for k, v in by_kind.items()})
    return result


def evaluate(args):
    import torch
    from .model import predict_checkpoint
    root = Path(args.output_dir) / args.encoder / args.source_dataset
    config = json.loads((root / "training.metrics.json").read_text())
    objectives = LOCAL_OBJECTIVES[:4] if getattr(args, 'local_risk', False) else OBJECTIVES
    if config["smoke_only"] and args.split != "valid":
        raise ValueError("smoke checkpoints cannot produce formal Test or SVEN results")
    if args.split != "valid" and any(not (root / name / "best.pt").exists() for name in objectives):
        raise ValueError("finish all Valid-only ablations before final evaluation")
    target_dataset = "sven" if args.split == "external_test" else args.source_dataset
    rows, pairs, views = supervision(args.data_dir, target_dataset, args.split)
    rows = [r for r in rows if r["split"] == args.split]
    if config["smoke_only"]:
        rows = [r for r in rows if r["sample_key"] in config["smoke_members"]]
        keys = {r["sample_key"] for r in rows}
        pairs = [p for p in pairs if p[0]["sample_key"] in keys]
        views = [v for v in views if v[0]["sample_key"] in keys]
    if not rows:
        raise ValueError("empty evaluation split")
    keys = {r["sample_key"] for r in rows}
    pair_rows = {r["sample_key"]: r for pair in pairs for r in pair if r["sample_key"] not in keys}
    pair_rows = list(pair_rows.values())
    visible_pairs, _, coverage = input_coverage(rows + pair_rows, pairs, views, args.encoder,
                                               config["model_path"], config["source_max_length"])
    visible_keys = {(a["sample_key"], b["sample_key"]) for a, b in visible_pairs}
    input_identical_pairs = {(a["sample_key"], b["sample_key"]) for a, b in pairs} - visible_keys
    results = {}
    for objective in ([args.objective] if args.objective else objectives):
        checkpoint_path = root / objective / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if getattr(args, 'local_risk', False) and checkpoint.get('local_supervision', {}).get('objective') != objective:
            raise ValueError('local risk evaluation checkpoint objective mismatch')
        if objective == "bce_random_ranking" and checkpoint.get("change_supervision", {}).get("random_sampling_protocol") != "epoch_marginal_matched":
            raise ValueError("random control must match actually sampled epoch endpoints")
        threshold = checkpoint["decision_threshold"]
        output = root / objective / f"{args.split}.metrics.json"
        predictions = root / objective / f"{args.split}.predictions.jsonl"
        if predictions.exists():
            saved = read_rows(predictions)
            expected = rows + pair_rows
            if [(r["sample_key"], r["label"]) for r in saved] != [(r["sample_key"], r["label"]) for r in expected] or any(
                    r["threshold"] != threshold for r in saved):
                raise ValueError("cached prediction members/threshold mismatch")
            zs = torch.tensor([r["logit"] for r in saved], dtype=torch.float32)
            lookup = {r["sample_key"]: r for r in saved}
            if any(lookup[r["sample_key"]]["transformed_logit"] is None for r, _ in views):
                raise ValueError("cached transformation prediction missing")
            view_logits = torch.tensor([lookup[r["sample_key"]]["transformed_logit"] for r, _ in views], dtype=torch.float32)
        else:
            # One model load for both original and transformed examples.
            zs = predict_checkpoint(checkpoint_path, rows + pair_rows + [v for _, v in views],
                                    batch_size=args.batch_size, device=args.device)
            view_logits = zs[len(rows) + len(pair_rows):]
        logits = zs[:len(rows)]
        pair_logits = zs[len(rows):len(rows) + len(pair_rows)]
        result = score_metrics(rows, logits, threshold, pairs, views, view_logits, pair_rows, pair_logits,
                               input_identical_pairs=input_identical_pairs)
        result.update(trained_on=args.source_dataset, evaluated_on=target_dataset,
                      split=args.split, objective=objective, selected_epoch=checkpoint["selected_epoch"],
                      smoke_only=config["smoke_only"], input_coverage=coverage)
        transformed = {a["sample_key"]: float(z) for (a, _), z in zip(views, view_logits)}
        if not predictions.exists():
            write_rows(predictions, [dict(sample_key=r["sample_key"], dataset=r["dataset"], split=r["split"],
                    label=r["label"], logit=float(z), score=float(z.sigmoid()), threshold=threshold,
                    classification_member=r["sample_key"] in keys,
                    transformed_logit=transformed.get(r["sample_key"]))
                    for r, z in zip(rows + pair_rows, torch.cat((logits, pair_logits)))])
        write_json(output, result)
        results[objective] = result
    write_json(root / f"comparison.{args.split}.metrics.json", results)
    return results


def report(output_dir, *, local_risk=False):
    """Export the complete matrix; never substitute incomplete or smoke runs."""
    import torch
    root = Path(output_dir)
    exported, effects, budgets = [], {}, {}
    objectives = LOCAL_OBJECTIVES[:4] if local_risk else OBJECTIVES
    for encoder in MODELS:
        for dataset in ("primevul", "cleanvul"):
            folder = root / encoder / dataset
            config = json.loads((folder / "training.metrics.json").read_text())
            if config["smoke_only"]:
                raise ValueError("formal report cannot contain smoke results")
            counts = set()
            parameters = {}
            for objective in objectives:
                checkpoint = torch.load(folder / objective / "best.pt", map_location="cpu", weights_only=False)
                if objective == "bce_random_ranking" and checkpoint["change_supervision"].get("random_sampling_protocol") != "epoch_marginal_matched":
                    raise ValueError("formal random control has unmatched sampled endpoints")
                history = [r for r in read_rows(folder / objective / "best.training.jsonl") if r["event"] == "epoch"]
                parameters[objective] = checkpoint['trainable_parameters']
                count = (len(history), sum(r["optimizer_steps"] for r in history), sum(r["auxiliary_function_forwards"] for r in history))
                if local_risk:
                    count += (sum(r['bce_encoder_sequences'] for r in history),
                              sum(r['auxiliary_encoder_sequences'] for r in history))
                counts.add(count if local_risk else (*count, checkpoint['trainable_parameters']))
                if local_risk and ([r['phase'] for r in history] != ['head_bce', 'joint', 'joint']
                                   or history[0]['auxiliary_function_forwards'] != 0):
                    raise ValueError('local risk phase schedule differs from the specified protocol')
            if len(counts) != 1:
                raise ValueError(f"{encoder}/{dataset}: ablation budgets differ")
            budgets[f"{encoder}/{dataset}"] = dict(zip(
                (("epochs", "optimizer_steps", "auxiliary_function_forwards", "bce_encoder_sequences", "auxiliary_encoder_sequences")
                 if local_risk else ("epochs", "optimizer_steps", "auxiliary_function_forwards", "trainable_parameters")), counts.pop()))
            if local_risk:
                if len({parameters[o] for o in ('B', 'C', 'D')}) != 1:
                    raise ValueError('local heads have different parameter budgets')
                budgets[f'{encoder}/{dataset}']['trainable_parameters'] = parameters
                budgets[f'{encoder}/{dataset}']['initial_bce_epochs'] = config['initial_bce_epochs']
            for split in ("valid", "test", "external_test"):
                part = {}
                for objective in objectives:
                    value = json.loads((folder / objective / f"{split}.metrics.json").read_text())
                    if value["smoke_only"] or value["trained_on"] != dataset or value["objective"] != objective:
                        raise ValueError("evaluation identity mismatch")
                    if value["repair"].get("input_identical_pairs") != value["input_coverage"]["token_identical_repair_pairs"]:
                        raise ValueError("input-identical repairs must count as ties")
                    flat = dict(encoder=encoder, trained_on=dataset, evaluated_on=value["evaluated_on"],
                        split=split, objective=objective, selected_epoch=value["selected_epoch"],
                        threshold=value["threshold"], samples=value["samples"])
                    flat.update({key: value[key] for key in ("accuracy", "precision", "recall", "f1", "mcc", "auc", "pr_auc", "average_precision")})
                    flat.update(value["repair"])
                    flat.update({key: value["robustness"][key] for key in (
                        "eligible", "coverage", "prediction_consistency", "robust_accuracy", "original_accuracy_same_subset")})
                    exported.append(flat)
                    part[objective] = flat
                def delta(left, right):
                    keys = ("f1", "mcc", "auc", "pr_auc", "pairwise_correct", "ranking_accuracy", "prediction_consistency", "robust_accuracy")
                    return {k: left[k] - right[k] if left[k] is not None and right[k] is not None else None for k in keys}
                effects[f"{encoder}/{dataset}/{split}"] = (dict(
                    B_vs_A=delta(part['B'], part['A']), C_vs_B=delta(part['C'], part['B']),
                    D_vs_C=delta(part['D'], part['C']), D_vs_A=delta(part['D'], part['A'])) if local_risk else dict(
                    versus_bce={o: delta(part[o], part["bce"]) for o in OBJECTIVES if o != "bce"},
                    real_vs_random_ranking=delta(part["bce_ranking"], part["bce_random_ranking"])))
    payload = dict(seed=42, budgets=budgets, rows=exported, effects=effects,
                   selection="Corresponding Valid only; fixed candidate configuration; no Test/SVEN tuning")
    write_json(root / "comparison.metrics.json", payload)
    with (root / "comparison.metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    return payload
