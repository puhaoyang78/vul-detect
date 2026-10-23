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
            (
                int(total * 0.8),
                int(total * 0.1),
                total - int(total * 0.8) - int(total * 0.1),
            ),
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
        "primevul": [
            raw_root / "PrimeVul_v0.1" / f"primevul_{s}.jsonl" for s in SPLITS
        ],
        "megavul": [raw_root / "megavul" / "megavul.json"],
    }[dataset]
    for path in paths:
        if dataset == "primevul":
            with path.open(encoding="utf-8") as handle:
                _append_native(
                    rows, (json.loads(line) for line in handle), dataset, path, raw_root
                )
        elif dataset == "megavul":
            _append_native(rows, json_array(path), dataset, path, raw_root)
    if len({r["sample_key"] for r in rows}) != len(rows):
        raise ValueError(f"{dataset}: duplicate native sample IDs")
    return rows, split_projects(rows, seed) if dataset != "primevul" else None


def _append_native(rows, iterator, dataset, path, raw_root):
    for number, raw in enumerate(iterator, 1):
        if dataset == "primevul":
            label, source, filename, project, key = (
                raw["target"],
                raw["func"],
                raw.get("file_name"),
                raw.get("project"),
                raw["idx"],
            )
        elif dataset == "megavul":
            label = raw["is_vul"]
            source, filename, project, key = (
                raw["func_before"] if label else raw["func"],
                raw.get("file_path"),
                raw["repo_name"],
                number,
            )
        expected_type = bool if dataset == "megavul" else int
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
                split=(
                    path.stem.removeprefix("primevul_")
                    if dataset == "primevul"
                    else None
                ),
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
        positive = [
            i for i, r in enumerate(rows) if r["split"] == split and r["label"] == 1
        ]
        negative = [
            i for i, r in enumerate(rows) if r["split"] == split and r["label"] == 0
        ]
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
            raise ValueError(
                "not enough members for the requested balanced diagnostic subset"
            )
        selected.update(rng.sample(candidates, limit // 2))
    return [row for i, row in enumerate(rows) if i in selected]


def counts(rows):
    return {s: class_counts([r for r in rows if r["split"] == s]) for s in SPLITS}


def test_rows(data_dir):
    """Read the established Test cohort, excluding its recorded exact overlaps."""
    return [
        r
        for r in read_jsonl(Path(data_dir) / "test.jsonl")
        if not r["train_overlap"] and not r.get("valid_overlap", False)
    ]


def validation_rows(data_dir):
    return [
        r
        for r in read_jsonl(Path(data_dir) / "valid.jsonl")
        if not r.get("train_overlap", [])
    ]


def prepare(raw_root, output, seed=42, datasets=DATASETS):
    output = Path(output)
    for dataset in datasets:
        if any((output / dataset).glob("*.jsonl")):
            raise FileExistsError(
                f"{output / dataset}: prepared source lists already exist"
            )
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
            row["train_overlap"] = (
                [dataset] if row["raw_source"] in training_sources[dataset] else []
            )
            if row["split"] == "test":
                row["valid_overlap"] = row["raw_source"] in validation_sources
        excluded = [r for r in tests if r["train_overlap"] or r["valid_overlap"]]
        evaluated = [
            r for r in tests if not r["train_overlap"] and not r["valid_overlap"]
        ]
        for split in SPLITS:
            write_jsonl(
                root / f"{split}.jsonl", (r for r in rows if r["split"] == split)
            )
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
                        **{
                            k: row[k]
                            for k in fields
                            if k not in ("train_overlap", "valid_overlap")
                        },
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
            stats["projects"] = {
                s: len(names) for s, names in assignments[dataset].items()
            }
        write_json(root / "statistics.json", stats)
        statistics[dataset] = stats
        print(json.dumps(stats, indent=2), flush=True)
    return statistics


def modified_lines(source, counterpart):
    """Exact LOSVER Differ rule: nonblank deleted lines, separately on each side."""
    import difflib

    result, position = [], 0
    for line in difflib.Differ().compare(
        source.splitlines(keepends=True), counterpart.splitlines(keepends=True)
    ):
        if line.startswith("- "):
            if line[2:].strip():
                result.append(position)
            position += 1
        elif line.startswith("  "):
            position += 1
    return result


def paired_encoding(code, tokenizer, input_mode="512", *, tokenization=None):
    """One original-source tokenization, shared by localization and detection.

    Physical line IDs are zero-based, including blank lines in their numbering.
    A token spanning several nonempty lines belongs to each of those lines.
    Whitespace-only tokens and special tokens have no aggregation membership.
    The 512 mode retains and marks the visible part of a truncated final line.
    L1 keeps every source token. Stored IDs/offsets use unique-source coordinates;
    (left, right, core_start, core_end) ranges are half-open source-token indices.
    paired_collate builds the actual native 512-token encoder windows.
    """
    from bisect import bisect_right

    if input_mode not in ("512", "l1"):
        raise ValueError("paired input must be 512 or l1")
    if not tokenizer.is_fast:
        raise ValueError(
            "paired encoding requires a fast tokenizer with source offsets"
        )
    encoded = (
        tokenization
        if tokenization is not None
        else tokenizer(
            code,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            verbose=False,
        )
    )
    parts = code.splitlines(keepends=True)
    starts, position = [], 0
    for part in parts:
        starts.append(position)
        position += len(part)
    memberships = [[] for _ in parts]
    special_ids = set(tokenizer.all_special_ids)
    for token, (token_id, (start, end)) in enumerate(
        zip(encoded["input_ids"], encoded["offset_mapping"])
    ):
        if token_id in special_ids or end <= start:
            continue
        line = max(0, bisect_right(starts, start) - 1)
        while line < len(parts) and starts[line] < end:
            a, b = max(start, starts[line]), min(end, starts[line] + len(parts[line]))
            if code[a:b].strip():
                memberships[line].append(token)
            line += 1
    source_count = len(encoded["input_ids"])
    visible_count = min(510, source_count) if input_mode == "512" else source_count
    ids = (
        [tokenizer.cls_token_id]
        + encoded["input_ids"][:visible_count]
        + [tokenizer.sep_token_id]
    )
    length = len(ids)
    ids += [tokenizer.pad_token_id] * max(0, 512 - length)
    if visible_count <= 510:
        windows = [(0, visible_count, 0, visible_count)]
    else:
        windows = [
            (
                max(0, a - 63),
                min(source_count, a + 384 + 63),
                a,
                min(source_count, a + 384),
            )
            for a in range(0, source_count, 384)
        ]
    line_numbers, line_tokens, complete = [], [], []
    for line, tokens in enumerate(memberships):
        visible = [token + 1 for token in tokens if token < visible_count]
        if visible:
            line_numbers.append(line)
            line_tokens.append(visible)
            complete.append(tokens[-1] < visible_count)
    code_mask = [False] * len(ids)
    for tokens in line_tokens:
        for token in tokens:
            code_mask[token] = True
    if not any(code_mask):
        raise ValueError("no non-whitespace source tokens in the input")
    return dict(
        input_mode=input_mode,
        windows=windows,
        pad_token_id=tokenizer.pad_token_id,
        input_ids=ids,
        attention_mask=[int(i < length) for i in range(len(ids))],
        code_mask=code_mask,
        offset_mapping=[(0, 0)]
        + encoded["offset_mapping"][:visible_count]
        + [(0, 0)] * (len(ids) - 1 - visible_count),
        line_numbers=line_numbers,
        line_token_indices=line_tokens,
        line_complete=complete,
        source_tokens=len(encoded["input_ids"]),
        source_lines=len(parts),
        nonempty_lines=sum(bool(part.strip()) for part in parts),
        unmapped_nonempty_lines=sum(
            bool(part.strip()) and not tokens
            for part, tokens in zip(parts, memberships)
        ),
    )


def paired_features(row, tokenizer, variant, *, encoding=None, input_mode="512"):
    """Attach labels or binary 5/1 weights to the same source-aligned input."""
    if variant not in ("localizer", "baseline", "predicted", "oracle"):
        raise ValueError(f"unknown paired variant: {variant}")
    f = dict(
        paired_encoding(row["func"], tokenizer, input_mode)
        if encoding is None
        else encoding
    )
    if variant == "localizer":
        gt = set(row.get("lines_ground_truth", []))
        f["labels"] = [int(n in gt) for n in f["line_numbers"]]
    else:
        f["labels"] = row["target"]
    if variant == "oracle":
        selected = set(row["lines_ground_truth"])
    elif variant == "predicted":
        selected = set(row["lines_predicted"])
        if row["line_numbers"] != f["line_numbers"]:
            raise ValueError("predicted line numbers do not match source offsets")
        if not selected.issubset(f["line_numbers"]):
            raise ValueError("predicted line is outside the visible source")
    else:
        selected = set()
    weights = [1.0] * len(f["input_ids"])
    for number, tokens in zip(f["line_numbers"], f["line_token_indices"]):
        if number in selected:
            for token in tokens:
                weights[token] = 5.0
    f["token_weights"] = weights
    return f


def paired_statistics(rows, features):
    """Summarize known Diff labels and actual source-token visibility."""
    counts = Counter()
    signatures = []
    for row, f in zip(rows, features):
        visible = set(f["line_numbers"])
        full = {
            n for n, complete in zip(f["line_numbers"], f["line_complete"]) if complete
        }
        gt = set(row["lines_ground_truth"])
        counts.update(
            functions=1,
            source_tokens=f["source_tokens"],
            physical_lines=f["source_lines"],
            code_tokens=sum(f["code_mask"]),
            truncated_functions=int(
                sum(d - c for a, b, c, d in f["windows"]) < f["source_tokens"]
            ),
            windows=len(f["windows"]),
            encoder_tokens=512 * len(f["windows"]),
            encoder_source_tokens=sum(b - a for a, b, c, d in f["windows"]),
            core_tokens=sum(d - c for a, b, c, d in f["windows"]),
            context_tokens=sum((b - a) - (d - c) for a, b, c, d in f["windows"]),
            nonempty_lines=f["nonempty_lines"],
            visible_nonempty_lines=len(visible),
            fully_visible_nonempty_lines=len(full),
            partial_lines=len(visible - full),
            modified_lines=len(gt),
            visible_modified_lines=len(gt & visible),
            fully_visible_modified_lines=len(gt & full),
            unmapped_nonempty_lines=f["unmapped_nonempty_lines"],
        )
        signatures.append(tuple(f["input_ids"]))
        counts["max_windows"] = max(counts["max_windows"], len(f["windows"]))
    counts["identical_pairs"] = sum(
        signatures[i] == signatures[i + 1] for i in range(0, len(signatures), 2)
    )
    return dict(counts)


def prepare_paired(raw_root, output_dir, model_path):
    """Official original PrimeVul members; Diff supervision is stored separately."""
    from itertools import combinations
    from transformers import RobertaTokenizerFast

    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False)
    root = Path(raw_root) / "PrimeVul_original"
    dest = Path(output_dir) / "primevul_paired"
    if any((dest / f"{s}.jsonl").exists() for s in SPLITS):
        raise FileExistsError(f"{dest}: paired data already prepared")
    sources, encoded = {}, {}
    stats = dict(
        raw_root=str(root),
        model_path=str(model_path),
        release="PrimeVul original",
        line_mapping="source_offsets",
        upstream_commit="a158905cb45b4400f481f18eeb1cb4473ccfccdc",
        splits={},
        overlap={},
        protocol="All official adjacent members retained; no resampling, no extra duplicate exclusions. Oracle Diff labels are separate from normal inputs.",
        identical_inputs_definition="Token-ID equality under the common source encoding. Oracle line weights can still differ; no pair is removed.",
    )
    members = []
    for split in SPLITS:
        raw = read_jsonl(root / f"primevul_{split}_paired.jsonl")
        if len(raw) % 2 or [r["target"] for r in raw] != [1, 0] * (len(raw) // 2):
            raise ValueError(f"{split}: official vulnerable/patched adjacency required")
        rows = []
        annotations = []
        anomalies = []
        for i in range(0, len(raw), 2):
            a, b = raw[i : i + 2]
            mismatch = [k for k in ("project", "commit_id") if a.get(k) != b.get(k)]
            if mismatch:
                anomalies.append(
                    dict(pair=i // 2, fields=mismatch, idx=[a["idx"], b["idx"]])
                )
            for offset, (r, other) in enumerate(((a, b), (b, a))):
                if not isinstance(r["func"], str) or not r["func"].strip():
                    raise ValueError(f"{split}:{i+offset+1}: empty source")
                row = dict(
                    func=r["func"],
                    target=r["target"],
                    idx=r["idx"],
                    sample_key=f"primevul_paired:{split}:{i+offset+1}",
                    pair_key=f"primevul_paired:{split}:{i//2}",
                    source_row=i + offset + 1,
                    project=r.get("project"),
                    commit_id=r.get("commit_id"),
                )
                rows.append(row)
                annotations.append(
                    dict(
                        row, lines_ground_truth=modified_lines(r["func"], other["func"])
                    )
                )
                members.append({k: v for k, v in row.items() if k != "func"})
        print(f"{split}: encoding {len(rows)} official functions", flush=True)
        features = [paired_encoding(r["func"], tokenizer) for r in rows]
        encoded[split] = {tuple(f["input_ids"]) for f in features}
        sources[split] = {r["func"] for r in rows}
        v01 = Path(raw_root) / "PrimeVul_v0.1" / f"primevul_{split}_paired.jsonl"
        old = read_jsonl(v01) if v01.exists() else []
        old_sources = {r["func"] for r in old}
        stats["splits"][split] = dict(
            pairs=len(rows) // 2,
            positive=len(rows) // 2,
            negative=len(rows) // 2,
            v01_pairs=len(old) // 2,
            old_source_members=sum(r["func"] in old_sources for r in rows),
            duplicate_idx_rows=len(rows) - len({r["idx"] for r in rows}),
            metadata_anomalies=anomalies,
            **paired_statistics(annotations, features),
        )
        write_jsonl(dest / f"{split}.jsonl", rows)
        write_jsonl(
            dest / (f"{split}.lines.jsonl" if split != "test" else "test.oracle.jsonl"),
            annotations,
        )
        print(split, stats["splits"][split], flush=True)
    for a, b in combinations(SPLITS, 2):
        stats["overlap"][f"{a}:{b}"] = dict(
            exact_sources=len(sources[a] & sources[b]),
            encoded_inputs=len(encoded[a] & encoded[b]),
        )
    write_json(dest / "statistics.json", stats)
    with (dest / "members.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(members[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(members)
    return stats
