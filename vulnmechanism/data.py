"""Native labels, positive-preserving balance and exact train/Test overlap checks."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import random

from .artifacts import write_json

DATASETS = ("primevul", "megavul")
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
    paths = (
        [raw_root / "PrimeVul_v0.1" / f"primevul_{s}.jsonl" for s in SPLITS]
        if dataset == "primevul"
        else [raw_root / "megavul" / "megavul.json"]
    )
    for path in paths:
        if dataset == "primevul":
            with path.open(encoding="utf-8") as handle:
                _append_native(
                    rows, (json.loads(line) for line in handle), dataset, path, raw_root
                )
        else:
            _append_native(rows, json_array(path), dataset, path, raw_root)
    return rows, split_projects(rows, seed) if dataset == "megavul" else None


def _append_native(rows, iterator, dataset, path, raw_root):
    for number, raw in enumerate(iterator, 1):
        label = raw["target"] if dataset == "primevul" else raw["is_vul"]
        if (dataset == "primevul" and (type(label) is not int or label not in (0, 1))) or (
            dataset == "megavul" and type(label) is not bool
        ):
            raise ValueError(f"{path}:{number}: invalid native label")
        source = raw["func"] if dataset == "primevul" or not label else raw["func_before"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{path}:{number}: missing native function")
        filename = raw.get("file_name" if dataset == "primevul" else "file_path") or ""
        if filename == "None":
            filename = ""
        suffix = Path(filename).suffix
        key = raw["idx"] if dataset == "primevul" else number
        project = raw.get("project" if dataset == "primevul" else "repo_name") or ""
        if dataset == "megavul" and not project.strip():
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


def counts(rows):
    return {s: class_counts([r for r in rows if r["split"] == s]) for s in SPLITS}


def test_rows(data_dir):
    """The same independent Test cohort is used for in-domain and transfer tests."""
    return [r for r in read_jsonl(Path(data_dir) / "test.jsonl") if not r["train_overlap"]]


def prepare(raw_root, output, seed=42):
    output = Path(output)
    for dataset in DATASETS:
        if any((output / dataset).glob("*.jsonl")):
            raise FileExistsError(f"{output / dataset}: prepared source lists already exist")
    native, selected, assignments = {}, {}, {}
    for dataset in DATASETS:
        native[dataset], assignments[dataset] = load_native(raw_root, dataset, seed)
        selected[dataset] = balanced(native[dataset], seed)
    training_sources = {
        dataset: {r["raw_source"] for r in rows if r["split"] == "train"}
        for dataset, rows in selected.items()
    }
    statistics = {}
    for dataset in DATASETS:
        root = output / dataset
        rows = selected[dataset]
        tests = [r for r in rows if r["split"] == "test"]
        for row in tests:
            row["train_overlap"] = [
                source for source in DATASETS if row["raw_source"] in training_sources[source]
            ]
        excluded = [r for r in tests if r["train_overlap"]]
        evaluated = [r for r in tests if not r["train_overlap"]]
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
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **{k: row[k] for k in fields[:-1]},
                        "train_overlap": ";".join(row.get("train_overlap", [])),
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
                    source: class_counts([r for r in tests if source in r["train_overlap"]])
                    for source in DATASETS
                },
                excluded=class_counts(excluded),
                evaluated=class_counts(evaluated),
                policy="keep all positives in data lists; exclude exact raw-source overlap with either actual training set only from Test evaluation; shared in-domain/transfer cohort",
            ),
        )
        if assignments[dataset] is not None:
            write_json(root / "projects.json", assignments[dataset])
            stats["projects"] = {s: len(names) for s, names in assignments[dataset].items()}
        write_json(root / "statistics.json", stats)
        statistics[dataset] = stats
        print(json.dumps(stats, indent=2), flush=True)
    return statistics
