"""Leakage-controlled cohorts for the static-relation pilot (not old experiments)."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from .artifacts import write_json
from .data import SPLITS, balanced_subset, class_counts, split_projects, write_jsonl
from .syntax import source_token_spans

DATASETS = ("bigvul", "devign", "reveal")
SOURCES = {
    "bigvul": "https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset",
    "devign": "https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection",
    "reveal": "https://github.com/VulDetProject/ReVeal",
}


def native_rows(raw_root, dataset, index_dir):
    root = Path(raw_root)
    if dataset == "bigvul":
        csv.field_size_limit(2**31 - 1)
        with (root / "bigvul/MSR_data_cleaned.csv").open(newline="") as handle:
            for i, r in enumerate(csv.DictReader(handle)):
                yield dict(sample_key=f"bigvul:{r['']}", raw_source=r["func_before"],
                           label=int(r["vul"]), project=r["project"].strip().casefold(),
                           commit_id=r["commit_id"], file_name=r["file_name"], source_row=i + 1)
                if (i + 1) % 50000 == 0:
                    print(f"bigvul: read {i + 1} raw rows", flush=True)
    elif dataset == "devign":
        assignment = {}
        for split in SPLITS:
            for line in (Path(index_dir) / f"devign_{split}.txt").read_text().splitlines():
                idx = int(line)
                if idx in assignment:
                    raise ValueError("overlapping upstream Devign indices")
                assignment[idx] = split
        rows = json.loads((root / "devign/dataset.json").read_text())
        if set(assignment) != set(range(len(rows))):
            raise ValueError("Devign official indices do not cover the local array")
        for i, r in enumerate(rows):
            yield dict(sample_key=f"devign:{i}", raw_source=r["func"], label=r["target"],
                       project=r["project"].casefold(), commit_id=r["commit_id"],
                       split=assignment[i], source_row=i + 1)
    elif dataset == "reveal":
        for label, name in ((0, "non-vulnerables.json"), (1, "vulnerables.json")):
            for i, r in enumerate(json.loads((root / "reveal" / name).read_text())):
                yield dict(sample_key=f"reveal:{label}:{i}", raw_source=r["code"], label=label,
                           project=r["project"], upstream_group=str(r["hash"]), source_row=i + 1)
    else:
        raise ValueError(dataset)


def remove_duplicates(rows):
    """Remove source/token-equivalent label conflicts; keep one same-label member.

    Prefer Test over Valid over Train when an upstream split exists. Token
    spelling is preserved: normalization removes only whitespace and comments.
    """
    priority = {"test": 0, "valid": 1, "train": 2, None: 3}
    groups = defaultdict(list)
    for row in rows:
        key = "\0".join(t for t, _, _ in source_token_spans(row["raw_source"]))
        groups[key].append(row)
        if len(groups) % 1000 == 0:
            source_token_spans.cache_clear()
    kept, removed = [], []
    reasons = Counter()
    for group in groups.values():
        group.sort(key=lambda r: (priority[r.get("split")], r["sample_key"]))
        if len({r["label"] for r in group}) > 1:
            reason, discard = "conflicting_source_or_token_labels", group
        else:
            kept.append(group[0])
            reason, discard = "source_or_token_duplicate", group[1:]
        for row in discard:
            removed.append(dict(sample_key=row["sample_key"], reason=reason))
            reasons[reason] += 1
    source_token_spans.cache_clear()
    return kept, removed, dict(reasons)


def group_splits(rows, dataset, seed=42):
    if dataset == "bigvul":
        split_projects(rows, seed)
        return "Seed 42 project groups, greedy positive-count 80/10/10; no upstream split supplied"
    if dataset == "reveal":
        # The supplied project field identifies only Debian/Chrome collections.
        # Keep the supplied hash-associated families together, within collection.
        groups = defaultdict(list)
        for row in rows:
            groups[row["project"] + ":" + row["upstream_group"]].append(row)
        proxies = [dict(project=key, label=r["label"]) for key, rs in groups.items() for r in rs]
        split_projects(proxies, seed)
        assignment = {r["project"]: r["split"] for r in proxies}
        for key, rs in groups.items():
            for row in rs:
                row["split"] = assignment[key]
        return "Seed 42 collection+upstream-hash groups, greedy positive-count 80/10/10; project field retained"
    return "CodeXGLUE published indices, with commit families retained only in highest-priority Test/Valid/Train split"


def isolate_families(rows, dataset):
    if dataset != "devign":
        return rows, []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["project"], row["commit_id"])].append(row)
    priority = {"test": 0, "valid": 1, "train": 2}
    kept, removed = [], []
    for rs in groups.values():
        split = min((r["split"] for r in rs), key=priority.get)
        for row in rs:
            if row["split"] == split:
                kept.append(row)
            else:
                removed.append(dict(sample_key=row["sample_key"], reason="cross_split_commit_family"))
    return kept, removed


def balance_splits(rows, seed=42):
    """Without replacement; use the smaller class when cleaning removes positives."""
    selected = []
    for split in SPLITS:
        rs = sorted((r for r in rows if r["split"] == split), key=lambda r: r["sample_key"])
        n = min(sum(r["label"] == y for r in rs) for y in (0, 1))
        if not n:
            raise ValueError(f"{split}: an empty class after cleaning")
        selected.extend(balanced_subset(rs, 2 * n, seed))
    return selected


def near_overlaps(rows):
    """Complete prefix filter, verified full token-5-gram Jaccard >= .90.

    Only balanced candidate members are compared, before the pilot is drawn.
    Retain Test > Valid > Train. Every cross-split pair is enumerated before
    removal, so transitive groups cannot leave an unexamined overlapping pair.
    """
    vocab, arrays, freq = {}, [], Counter()
    for row in rows:
        tokens = [t for t, _, _ in source_token_spans(row["raw_source"])]
        grams = {tuple(tokens[i:i + 5]) for i in range(max(len(tokens) - 4, 0))}
        ids = np.array(sorted(vocab.setdefault(g, len(vocab)) for g in grams), dtype=np.int32)
        arrays.append(ids)
        freq.update(ids.tolist())
    source_token_spans.cache_clear()
    del vocab
    inverted = defaultdict(list)
    removed, pairs = set(), []
    priority = {"test": 0, "valid": 1, "train": 2}
    order = sorted(range(len(rows)), key=lambda i: (priority[rows[i]["split"]], rows[i]["sample_key"]))
    for i in order:
        a = arrays[i]
        if not len(a):
            continue
        prefix = sorted(a.tolist(), key=lambda g: (freq[g], g))[:len(a) - (9 * len(a) + 9) // 10 + 1]
        candidates = set(j for g in prefix for j in inverted[g])
        for j in sorted(candidates):
            if rows[i]["split"] == rows[j]["split"]:
                continue
            b = arrays[j]
            if 10 * min(len(a), len(b)) < 9 * max(len(a), len(b)):
                continue
            inter = len(np.intersect1d(a, b, assume_unique=True))
            union = len(a) + len(b) - inter
            if 10 * inter >= 9 * union:
                removed.add(i)
                pairs.append(dict(removed=rows[i]["sample_key"], higher_priority_reference=rows[j]["sample_key"],
                                  jaccard=inter / union))
        for g in prefix:
            inverted[g].append(i)
    return [r for i, r in enumerate(rows) if i not in removed], pairs


def check_isolation(rows, dataset):
    from itertools import combinations
    if len({r["sample_key"] for r in rows}) != len(rows):
        raise ValueError("duplicate sample IDs")
    keys = {}
    for split in SPLITS:
        rs = [r for r in rows if r["split"] == split]
        if not rs or sum(r["label"] == 0 for r in rs) != sum(r["label"] == 1 for r in rs):
            raise ValueError("unbalanced final split")
        keys[split] = set(r["project"] if dataset == "bigvul" else
                          (r["project"], r["commit_id"] if dataset == "devign" else r["upstream_group"])
                          for r in rs)
    if any(keys[a] & keys[b] for a, b in combinations(SPLITS, 2)):
        raise ValueError("associated families cross final splits")
    source_sets = {s: {r["raw_source"] for r in rows if r["split"] == s} for s in SPLITS}
    exact = sum(len(source_sets[a] & source_sets[b]) for a, b in combinations(SPLITS, 2))
    _, pairs = near_overlaps(rows)
    if exact or pairs:
        raise ValueError("final source leakage")
    return dict(exact_cross_split=exact, near_cross_split=len(pairs), family_cross_split=0,
                unique_sample_keys=len(rows), balanced_splits=True)


def transfer_overlap(data_dir, output_dir, datasets=DATASETS):
    """Audit actual pilot Train/Valid versus fixed Test; never change the splits."""
    from .data import read_jsonl
    root = Path(data_dir)
    references, tests, members = [], [], {}
    for dataset in datasets:
        for split in ("train", "valid", "test"):
            path = root / dataset / "test.jsonl" if split == "test" else root / "pilot" / dataset / f"{split}.jsonl"
            for row in read_jsonl(path):
                item = {k: v for k, v in row.items() if k != "program_relations"}
                members[item["sample_key"]] = item
                # near_overlaps compares different splits. Both seen splits
                # belong to the reference side; Test/Test pairs are irrelevant.
                (tests if split == "test" else references).append(dict(item, split="test" if split == "test" else "train"))
    started = time.monotonic()
    _, pairs = near_overlaps(tests + references)
    matches = []
    for pair in pairs:
        source, target = members[pair["removed"]], members[pair["higher_priority_reference"]]
        if source["split"] not in ("train", "valid") or target["split"] != "test":
            raise ValueError("near-overlap audit sides changed")
        matches.append(dict(source_dataset=source["dataset"], source_split=source["split"],
                            source_key=source["sample_key"], source_label=source["label"],
                            target_dataset=target["dataset"], target_key=target["sample_key"],
                            target_label=target["label"], jaccard=pair["jaccard"],
                            raw_exact=source["raw_source"] == target["raw_source"]))
    commit_matches = []
    commits = defaultdict(list)
    for row in references:
        if row.get("commit_id"):
            commits[(row["project"].casefold(), row["commit_id"].casefold())].append(row)
    for row in tests:
        if not row.get("commit_id"):
            continue
        for source in commits[(row["project"].casefold(), row["commit_id"].casefold())]:
            commit_matches.append(dict(source_dataset=source["dataset"], source_key=source["sample_key"],
                                       target_dataset=row["dataset"], target_key=row["sample_key"],
                                       project=row["project"], commit_id=row["commit_id"]))
    near_keys = {p["target_key"] for p in matches}
    commit_keys = {p["target_key"] for p in commit_matches}
    result = dict(
        protocol=dict(references="actual 1024 Train / 256 Valid per participating source model",
                      candidates="unchanged full Test; no Train/Valid fitting or threshold selection",
                      near_rule="comment-free spelling-preserving token 5-gram set Jaccard >= 0.90; complete prefix filter from near_overlaps",
                      timing="matching and metadata joins, excluding input reads",
                      use="diagnostic near-free sensitivity cohort shared by all source models; no resampling",
                      shared_records="project+commit only where available; ReVeal upstream hash is not treated as a commit/CVE; hidden shared vulnerability records remain unknown"),
        seconds=time.monotonic() - started,
        near_pairs=matches, shared_commit_pairs=commit_matches,
        targets={dataset: dict(
            original=class_counts([r for r in tests if r["dataset"] == dataset]),
            near_overlap=class_counts([r for r in tests if r["dataset"] == dataset and
                                      r["sample_key"] in near_keys]),
            shared_commit=class_counts([r for r in tests if r["dataset"] == dataset and
                                        r["sample_key"] in commit_keys]),
        ) for dataset in datasets},
    )
    write_json(Path(output_dir) / "overlap.json", result)
    return result


def prepare(raw_root, output_dir, index_dir, datasets=DATASETS):
    for dataset in datasets:
        started = time.monotonic()
        dest = Path(output_dir) / dataset
        if any((dest / f"{split}.jsonl").exists() for split in SPLITS):
            raise FileExistsError(f"cohort already prepared: {dest}")
        rows = list(native_rows(raw_root, dataset, index_dir))
        for row in rows:
            if type(row["label"]) is not int or row["label"] not in (0, 1) or not isinstance(row["raw_source"], str) or not row["raw_source"].strip():
                raise ValueError(f"invalid native row: {row['sample_key']}")
            row.update(dataset=dataset, language="c_cpp", train_overlap=False, valid_overlap=False)
        native = class_counts(rows)
        rows, removed, dedup = remove_duplicates(rows)
        rule = group_splits(rows, dataset)
        rows, family_removed = isolate_families(rows, dataset)
        removed.extend(family_removed)
        cleaned = {s: class_counts([r for r in rows if r["split"] == s]) for s in SPLITS}
        rows = balance_splits(rows)
        before_near = len(rows)
        rows, pairs = near_overlaps(rows)
        rows = balance_splits(rows)
        final = {s: class_counts([r for r in rows if r["split"] == s]) for s in SPLITS}
        checks = check_isolation(rows, dataset)
        dest.mkdir(parents=True, exist_ok=True)
        pilot_counts, pilot_members = {}, []
        for split in SPLITS:
            rs = [r for r in rows if r["split"] == split]
            write_jsonl(dest / f"{split}.jsonl", rs)
            if split != "test":
                pilot = balanced_subset(rs, 1024 if split == "train" else 256, 42)
                write_jsonl(Path(output_dir) / "pilot" / dataset / f"{split}.jsonl", pilot)
                pilot_counts[split] = class_counts(pilot)
                pilot_members.extend(pilot)
        write_jsonl(dest / "removed.jsonl", removed)
        write_jsonl(dest / "near_overlaps.jsonl", pairs)
        fields = ("sample_key", "split", "label", "project")
        for filename, members in (("members.csv", rows), ("pilot.members.csv", pilot_members)):
            with (dest / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fields);writer.writeheader()
                writer.writerows({k: r[k] for k in fields} for r in members)
        report = dict(dataset=dataset, seed=42, source=SOURCES[dataset], native=native, checks=checks,
                      dedup=dedup, family_removed=len(family_removed), split_rule=rule,
                      cleaned_before_balance=cleaned, balanced_before_near=before_near,
                      cross_split_near_pairs=len(pairs), cross_split_near_removed=len({p['removed'] for p in pairs}),
                      final=final, pilot=pilot_counts, seconds=time.monotonic() - started,
                      near_rule="comment-free spelling-preserving token 5-gram set Jaccard >= 0.90; Test > Valid > Train",
                      balance="equal smaller-class count per split, Seed 42, without replacement",
                      metadata_limitations=("ReVeal lacks commit/CVE/true repository IDs; associated upstream hashes kept together; hidden family leakage cannot be certified" if dataset == "reveal" else None))
        write_json(dest / "statistics.json", report)
        print(json.dumps(report), flush=True)
