"""Native labels, positive-preserving balance and exact train/Test overlap checks."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import random

from .artifacts import write_json

DATASETS = ("primevul", "megavul", "bigvul", "secvuleval")
SPLITS = ("train", "valid", "test")


def json_array(path):
    """Read the upstream MegaVul array without loading its graph/diff metadata."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as handle:
        buffer, position, eof = "", 0, False

        def space():
            nonlocal buffer, position, eof
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                block = handle.read(1024 * 1024)
                buffer, position, eof = block, 0, not block

        space()
        if buffer[position : position + 1] != "[":
            raise ValueError("MegaVul input must be a JSON array")
        position += 1
        space()
        first = True
        while buffer[position : position + 1] != "]":
            if not first:
                if buffer[position : position + 1] != ",":
                    raise ValueError("missing JSON array separator")
                position += 1
                space()
            while True:
                try:
                    value, position = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    block = handle.read(1024 * 1024)
                    buffer, position, eof = buffer[position:] + block, 0, not block
            if not isinstance(value, dict):
                raise ValueError("MegaVul rows must be objects")
            yield value
            first = False
            space()
        position += 1
        space()
        if buffer[position:] or handle.read(1):
            raise ValueError("unexpected content after JSON array")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def split_projects(rows, seed=42):
    """Greedily fill 80/10/10 positive targets; never split a project."""
    positives = Counter(r["project"] for r in rows if r["label"] == 1)
    names = sorted({r["project"] for r in rows})
    random.Random(seed).shuffle(names)
    names.sort(key=lambda name: -positives[name])
    total = sum(positives.values())
    targets = dict(
        zip(
            SPLITS,
            (int(total * 0.8), int(total * 0.1), total - int(total * 0.8) - int(total * 0.1)),
        )
    )
    assigned = Counter()
    projects = {s: [] for s in SPLITS}
    for name in names:
        split = max(SPLITS, key=lambda s: targets[s] - assigned[s])
        projects[split].append(name)
        assigned[split] += positives[name]
    assignment = {name: s for s, names in projects.items() for name in names}
    for row in rows:
        row["split"] = assignment[row["project"]]
    return {s: sorted(names) for s, names in projects.items()}


def load_native(raw_root, dataset, seed=42):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    raw_root = Path(raw_root)
    rows = []
    paths = {
        "primevul": [raw_root / "PrimeVul_v0.1" / f"primevul_{s}.jsonl" for s in SPLITS],
        "megavul": [raw_root / "megavul" / "megavul.json"],
        "bigvul": [raw_root / "bigvul" / "MSR_data_cleaned.csv"],
        "secvuleval": [raw_root / "SecVulEval" / "data" / "train-00000-of-00001.parquet"],
    }[dataset]
    for path in paths:
        if dataset == "primevul":
            with path.open(encoding="utf-8") as handle:
                _append_native(
                    rows, (json.loads(line) for line in handle), dataset, path, raw_root
                )
        elif dataset == "megavul":
            _append_native(rows, json_array(path), dataset, path, raw_root)
        elif dataset == "bigvul":
            csv.field_size_limit(2**31 - 1)
            with path.open(encoding="utf-8", newline="") as handle:
                _append_native(rows, csv.DictReader(handle), dataset, path, raw_root)
        else:
            import pyarrow.parquet as pq

            records = (
                row for batch in pq.ParquetFile(path).iter_batches()
                for row in batch.to_pylist()
            )
            _append_native(rows, records, dataset, path, raw_root)
    if len({r["sample_key"] for r in rows}) != len(rows):
        raise ValueError(f"{dataset}: duplicate native sample IDs")
    return rows, split_projects(rows, seed) if dataset != "primevul" else None


def _append_native(rows, iterator, dataset, path, raw_root):
    for number, raw in enumerate(iterator, 1):
        if dataset == "primevul":
            label, source, filename, project, key = (
                raw["target"], raw["func"], raw.get("file_name"), raw.get("project"), raw["idx"]
            )
        elif dataset == "megavul":
            label = raw["is_vul"]
            source, filename, project, key = (
                raw["func_before"] if label else raw["func"], raw.get("file_path"), raw["repo_name"], number
            )
        elif dataset == "bigvul":
            label, source, filename, project, key = (
                int(raw["vul"]), raw["func_before"], raw.get("file_name"), raw["project"], raw[""]
            )
        else:
            label, source, filename, project, key = (
                raw["is_vulnerable"], raw["func_body"], raw["filepath"],
                raw["project_url"].strip().rstrip("/"), raw["idx"],
            )
        expected_type = bool if dataset in ("megavul", "secvuleval") else int
        if type(label) is not expected_type or label not in (0, 1):
            raise ValueError(f"{path}:{number}: invalid native label")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{path}:{number}: missing native function")
        filename = filename or ""
        if filename == "None":
            filename = ""
        suffix = Path(filename).suffix
        project = project or ""
        if dataset != "primevul" and not project.strip():
            raise ValueError(f"{path}:{number}: missing project")
        rows.append(
            dict(
                sample_key=f"{dataset}:{key}",
                dataset=dataset,
                split=path.stem.removeprefix("primevul_") if dataset == "primevul" else None,
                label=int(label),
                raw_source=source,
                file_name=filename,
                language=(
                    "c"
                    if suffix == ".c"
                    else (
                        "cpp"
                        if suffix
                        in {".C", ".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}
                        else "c_cpp"
                    )
                ),
                project=project.strip().casefold(),
                source_file=str(path.relative_to(raw_root)),
                source_row=number,
            )
        )
        if number % 50000 == 0:
            print(f"{dataset} {path.name}: read {number}", flush=True)


def balanced(rows, seed=42):
    """Retain every positive; sample only the negatives, independently per Split."""
    selected = set()
    for split in SPLITS:
        positive = [i for i, r in enumerate(rows) if r["split"] == split and r["label"] == 1]
        negative = [i for i, r in enumerate(rows) if r["split"] == split and r["label"] == 0]
        if not positive or len(negative) < len(positive):
            raise ValueError(f"{split}: need positives and at least as many negatives")
        selected.update(positive)
        selected.update(random.Random(seed).sample(negative, len(positive)))
    return [row for i, row in enumerate(rows) if i in selected]


def class_counts(rows):
    positive = sum(r["label"] for r in rows)
    return dict(samples=len(rows), positive=positive, negative=len(rows) - positive)


def balanced_subset(rows, limit=None, seed=42):
    """Optional diagnostic subset; preserve the formal cohort when limit is None."""
    if limit is None:
        return rows
    if limit <= 0 or limit % 2:
        raise ValueError("diagnostic sample limit must be a positive even number")
    rng = random.Random(seed)
    selected = set()
    for label in (0, 1):
        candidates = [i for i, row in enumerate(rows) if row["label"] == label]
        if len(candidates) < limit // 2:
            raise ValueError("not enough members for the requested balanced diagnostic subset")
        selected.update(rng.sample(candidates, limit // 2))
    return [row for i, row in enumerate(rows) if i in selected]


def counts(rows):
    return {s: class_counts([r for r in rows if r["split"] == s]) for s in SPLITS}


def test_rows(data_dir):
    """Read the established Test cohort, excluding its recorded exact overlaps."""
    return [
        r for r in read_jsonl(Path(data_dir) / "test.jsonl")
        if not r["train_overlap"] and not r.get("valid_overlap", False)
    ]


def validation_rows(data_dir):
    return [
        r for r in read_jsonl(Path(data_dir) / "valid.jsonl")
        if not r.get("train_overlap", [])
    ]


def paired_groups(rows):
    """Official pair order, with vulnerable/patch roles checked rather than inferred."""
    groups = {}
    for row in rows:
        groups.setdefault(row["pair_key"], []).append(row)
    result = []
    for key, members in groups.items():
        if len(members) != 2 or {r["label"] for r in members} != {0, 1}:
            raise ValueError(f"{key}: a pair needs exactly one vulnerable and one patched function")
        result.append(tuple(sorted(members, key=lambda r: -r["label"])))
    return result


def paired_subset(rows, limit=None, seed=42):
    pairs = paired_groups(rows)
    if limit is not None:
        if limit <= 0 or limit > len(pairs):
            raise ValueError("pair limit exceeds available training pairs")
        selected = set(random.Random(seed).sample(range(len(pairs)), limit))
        pairs = [p for i, p in enumerate(pairs) if i in selected]
    return [r for pair in pairs for r in pair]


def official_paired_rows(raw_root, split):
    """Official adjacency defines a pair; upstream idx is not unique."""
    path = Path(raw_root) / "PrimeVul_v0.1" / f"primevul_{split}_paired.jsonl"
    native = read_jsonl(path)
    if len(native) % 2:
        raise ValueError(f"{path}: incomplete adjacent pair")
    rows, anomalies = [], []
    for at in range(0, len(native), 2):
        a, b = native[at:at + 2]
        if {a["target"], b["target"]} != {0, 1}:
            raise ValueError(f"{path}:{at + 1}: noncomplementary official labels")
        pair_key = f"primevul_paired:{split}:{at // 2}"
        mismatch = [k for k in ("project", "commit_id", "cve") if a.get(k) != b.get(k)]
        if mismatch:
            anomalies.append(dict(pair_key=pair_key, fields=mismatch,
                                  vulnerable_idx=a["idx"] if a["target"] else b["idx"],
                                  patched_idx=b["idx"] if a["target"] else a["idx"]))
        for offset, r in enumerate((a, b)):
            if type(r["target"]) is not int or not isinstance(r["func"], str) or not r["func"].strip():
                raise ValueError(f"{path}:{at + offset + 1}: invalid official function/label")
            rows.append(dict(sample_key=f"primevul_paired:{split}:{at + offset + 1}",
                             pair_key=pair_key, dataset="primevul_paired", split=split,
                             label=r["target"], raw_source=r["func"], project=r["project"],
                             commit_id=r["commit_id"], cve=r.get("cve"),
                             source_file=str(path.relative_to(raw_root)), source_row=at + offset + 1,
                             upstream_idx=r["idx"], language="c_cpp", train_overlap=False,
                             valid_overlap=False))
    return rows, anomalies


def prepare_paired(raw_root, output, model_path):
    """Keep official held-out members; remove only Train-to-held-out input leakage."""
    from itertools import combinations
    from .model import AutoTokenizer, InputBuilder

    dest = Path(output) / "primevul_paired"
    if any((dest / f"{s}.jsonl").exists() for s in SPLITS):
        raise FileExistsError(f"paired data already prepared: {dest}")
    builder = InputBuilder(AutoTokenizer.from_pretrained(model_path))
    budget = builder.source_max_length - len(builder.source_prefix) - 2
    splits, tokens, signatures, anomalies, duplicates = {}, {}, {}, [], {}
    for split in SPLITS:
        rows, mismatches = official_paired_rows(raw_root, split)
        anomalies.extend(mismatches)
        for a, b in zip(rows[::2], rows[1::2]):
            encoded = [tuple(builder._encode(r["raw_source"], budget)) for r in (a, b)]
            distinguishable = encoded[0] != encoded[1]
            for r, ids in zip((a, b), encoded):
                tokens[r["sample_key"]] = ids
                r["pair_distinguishable"] = distinguishable
        splits[split] = rows
        signatures[split] = {tokens[r["sample_key"]] for r in rows}
        duplicates[split] = dict(repeated_upstream_idx_rows=len(rows) - len({r["upstream_idx"] for r in rows}),
                                 repeated_source_rows=len(rows) - len({r["raw_source"] for r in rows}))
    overlaps = {}
    for a, b in combinations(SPLITS, 2):
        overlaps[f"{a}:{b}"] = dict(
            exact_sources=len({r["raw_source"] for r in splits[a]} & {r["raw_source"] for r in splits[b]}),
            encoded_inputs=len(signatures[a] & signatures[b]))
    heldout = signatures["valid"] | signatures["test"]
    removed = {r["pair_key"] for r in splits["train"] if tokens[r["sample_key"]] in heldout}
    original = splits["train"]
    splits["train"] = [r for r in original if r["pair_key"] not in removed]
    pilot = {r["sample_key"] for r in paired_subset(splits["train"], min(512, len(splits["train"]) // 2))}
    dest.mkdir(parents=True, exist_ok=True)
    members = []
    for split in SPLITS:
        write_jsonl(dest / f"{split}.jsonl", splits[split])
        for row in (original if split == "train" else splits[split]):
            members.append({k: row[k] for k in ("sample_key", "pair_key", "split", "label", "source_file", "source_row", "upstream_idx", "pair_distinguishable")})
            members[-1].update(excluded_train_overlap=row["pair_key"] in removed,
                               pilot_train=row["sample_key"] in pilot)
    with (dest / "members.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(members[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(members)
    write_json(dest / "statistics.json", dict(
        seed=42, model_path=str(model_path), source_budget=budget,
        native_pairs={s: (len(original) if s == "train" else len(splits[s])) // 2 for s in SPLITS},
        final={s: dict(**class_counts(rows), pairs=len(rows) // 2,
                       indistinguishable_pairs=sum(not r[0]["pair_distinguishable"] for r in paired_groups(rows)))
               for s, rows in splits.items()},
        cross_split_overlap=overlaps, removed_training_pairs=sorted(removed),
        within_split_duplicates=duplicates,
        metadata_anomalies=anomalies,
        protocol="official adjacent complementary labels; metadata anomalies retained; identical encoded Train pairs keep BCE but no pair constraint; exclude whole Train pairs with held-out identical inputs; Valid/Test untouched; no near/project/CVE filtering"))


def prepare(raw_root, output, seed=42, datasets=DATASETS):
    output = Path(output)
    for dataset in datasets:
        if any((output / dataset).glob("*.jsonl")):
            raise FileExistsError(f"{output / dataset}: prepared source lists already exist")
    native, selected, assignments = {}, {}, {}
    for dataset in datasets:
        native[dataset], assignments[dataset] = load_native(raw_root, dataset, seed)
        selected[dataset] = balanced(native[dataset], seed)
    training_sources = {
        dataset: {r["raw_source"] for r in rows if r["split"] == "train"}
        for dataset, rows in selected.items()
    }
    statistics = {}
    for dataset in datasets:
        root = output / dataset
        rows = selected[dataset]
        tests = [r for r in rows if r["split"] == "test"]
        valid = [r for r in rows if r["split"] == "valid"]
        validation_sources = {r["raw_source"] for r in valid}
        for row in valid + tests:
            row["train_overlap"] = [dataset] if row["raw_source"] in training_sources[dataset] else []
            if row["split"] == "test":
                row["valid_overlap"] = row["raw_source"] in validation_sources
        excluded = [r for r in tests if r["train_overlap"] or r["valid_overlap"]]
        evaluated = [r for r in tests if not r["train_overlap"] and not r["valid_overlap"]]
        for split in SPLITS:
            write_jsonl(root / f"{split}.jsonl", (r for r in rows if r["split"] == split))
        with (root / "members.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "sample_key",
                "split",
                "label",
                "project",
                "source_file",
                "source_row",
                "train_overlap",
                "valid_overlap",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **{k: row[k] for k in fields if k not in ("train_overlap", "valid_overlap")},
                        "train_overlap": ";".join(row.get("train_overlap", [])),
                        "valid_overlap": row.get("valid_overlap", False),
                    }
                )
        stats = dict(
            dataset=dataset,
            seed=seed,
            raw_root=str(raw_root),
            native=counts(native[dataset]),
            balanced=counts(rows),
            positives_removed=0,
            sampling="retain all native positives; random.Random(42) samples negatives; stable source order",
            test_exact_overlap=dict(
                by_training_dataset={
                    dataset: class_counts([r for r in tests if r["train_overlap"]]),
                },
                excluded=class_counts(excluded),
                evaluated=class_counts(evaluated),
                with_valid=class_counts([r for r in tests if r["valid_overlap"]]),
                policy="retain all positives in data lists; exclude own Train exact overlap from Valid, and own Train/Valid exact overlap from Test evaluation; no cross-dataset filtering",
            ),
            valid_exact_overlap=dict(
                excluded=class_counts([r for r in valid if r["train_overlap"]]),
                evaluated=class_counts([r for r in valid if not r["train_overlap"]]),
            ),
        )
        if assignments[dataset] is not None:
            write_json(root / "projects.json", assignments[dataset])
            stats["projects"] = {s: len(names) for s, names in assignments[dataset].items()}
        write_json(root / "statistics.json", stats)
        statistics[dataset] = stats
        print(json.dumps(stats, indent=2), flush=True)
    return statistics
