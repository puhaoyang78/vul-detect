"""Native labels, positive-preserving balance and exact train/Test overlap checks."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import random

from .artifacts import write_json

DATASETS = ("primevul", "megavul")
RGCL_DATASETS = ("secvuleval", "megavul", "cleanvul")
SPLITS = ("train", "valid", "test")


def paired_folds(rows, seed=123456):
    """Two deterministic pair-level folds without advancing the training RNG."""
    groups = {}
    samples = set()
    for row in rows:
        if row["sample_key"] in samples:
            raise ValueError("duplicate OOF sample_key")
        samples.add(row["sample_key"])
        groups.setdefault(row["pair_key"], []).append(row["target"])
    if len(groups) < 2 or any(sorted(labels) != [0, 1] for labels in groups.values()):
        raise ValueError("OOF requires complete vulnerable/patched pairs")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    return {key: i % 2 for i, key in enumerate(keys)}


def prepare_oof_predictions(data_dir, output_dir, reference_dir):
    """Merge held-out predictions, checking provenance before detector training."""
    import math

    if reference_dir is None:
        raise ValueError("OOF preparation requires --reference-dir")
    root = Path(output_dir)
    reference = Path(reference_dir) / "primevul_paired/losver_localizer"
    full = json.loads((reference / "training.metrics.json").read_text())
    if (full["input_mode"] != "512" or full["supervision"] != "original_diff"
            or full["train_samples"] != 1024 or full["valid_samples"] != 256):
        raise ValueError("OOF reference must be the completed original 512 Pilot localizer")
    source = read_jsonl(Path(data_dir) / "primevul_paired/train.jsonl")
    keys = set(full["training_members"])
    rows = [r for r in source if r["sample_key"] in keys]
    if [r["sample_key"] for r in rows] != full["training_members"]:
        raise ValueError("full localizer and source Train members/order differ")
    folds = paired_folds(rows, full["training_config"]["seed"])
    expected_settings = {k: v for k, v in full["training_config"].items() if k != "epochs"}
    expected_scores = {r["sample_key"]: r for r in read_jsonl(reference / "train.line_scores.jsonl")}
    merged, provenance = {}, []
    for fold in (0, 1):
        run = root / f"fold_{fold}/primevul_paired/losver_localizer"
        config = json.loads((run / "training.metrics.json").read_text())
        held = [r["sample_key"] for r in rows if folds[r["pair_key"]] == fold]
        trained = [r["sample_key"] for r in rows if folds[r["pair_key"]] != fold]
        if (config["training_members"] != trained
                or config["oof"]["prediction_members"] != held
                or config["oof"]["held_out_fold"] != fold
                or config["validation_members"] != full["validation_members"]
                or config["optimizer_steps"] != full["optimizer_steps"]
                or config["supervision"] != "original_diff"
                or config["input_mode"] != "512"
                or config["model_path"] != full["model_path"]
                or config["parameters"] != full["parameters"]
                or config["selection"] != full["selection"]
                or {k:v for k,v in config["training_config"].items() if k != "epochs"} != expected_settings):
            raise ValueError(f"fold {fold}: unmatched data, optimization budget or OOF provenance")
        scores = read_jsonl(run / "train.line_scores.jsonl")
        if [r["sample_key"] for r in scores] != held:
            raise ValueError(f"fold {fold}: exported predictions are not exactly held-out members")
        for score in scores:
            key = score["sample_key"]
            ns, ps = score["line_numbers"], score["line_scores"]
            if (key in merged or ns != expected_scores[key]["line_numbers"]
                    or len(ns) != len(ps) or len(ns) != len(set(ns))
                    or any(not math.isfinite(p) or not 0 <= p <= 1 for p in ps)):
                raise ValueError("OOF score coverage, physical lines or probabilities differ")
            merged[key] = score
        provenance.append(dict(fold=fold, directory=str(run), epochs=config["epochs"],
            functions=len(trained), held_out_functions=len(held),
            optimizer_steps=config["optimizer_steps"], selected_epoch=config["selected_epochs"]["f1"],
            selected_optimizer_steps=config["selected_epochs"]["f1"] * math.ceil(len(trained) / config["training_config"]["batch_size"])))
    if set(merged) != keys:
        raise ValueError("merged OOF predictions do not cover every Pilot Train member once")
    destination = root / "oof_predictions"
    write_jsonl(destination / "train.line_scores.jsonl", [merged[r["sample_key"]] for r in rows])
    write_jsonl(destination / "train.predicted.jsonl", [dict(r,
        line_numbers=merged[r["sample_key"]]["line_numbers"],
        lines_predicted=[n for n,p in zip(merged[r["sample_key"]]["line_numbers"],
                                        merged[r["sample_key"]]["line_scores"]) if p > .5]) for r in rows])
    by_code = {}
    for r in rows:
        by_code.setdefault(r["func"], set()).add(folds[r["pair_key"]])
    overlap = {code for code, fs in by_code.items() if len(fs) > 1}
    config = dict(reference_dir=str(reference_dir), full_localizer_dir=str(reference),
        seed=full["training_config"]["seed"], train_functions=len(rows), valid_functions=full["valid_samples"],
        full_optimizer_steps=full["optimizer_steps"], folds=provenance, test_used=False,
        cross_fold_pair_overlap=0, cross_fold_exact_source_groups=len(overlap),
        cross_fold_exact_source_functions=sum(r["func"] in overlap for r in rows))
    write_json(root / "oof.json", config)
    with (root / "folds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_key", "pair_key", "target", "held_out_fold"])
        writer.writeheader()
        writer.writerows(dict(sample_key=r["sample_key"], pair_key=r["pair_key"], target=r["target"],
                              held_out_fold=folds[r["pair_key"]]) for r in rows)
    return config


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


def load_rgcl_native(raw_root, dataset):
    """Original binary members only; repair counterparts supply no new labels."""
    import math

    raw_root = Path(raw_root)
    rows, counterparts, issues = [], {}, Counter()
    if dataset == "secvuleval":
        import pyarrow.parquet as pq

        paths = sorted((raw_root / "SecVulEval/data").glob("*.parquet"))
        if [p.name for p in paths] != ["train-00000-of-00001.parquet"]:
            raise ValueError("inspect changed SecVulEval release/splits before preparation")
        native = pq.read_table(paths[0]).to_pylist()
        by_id = {r["idx"]: r for r in native}
        if len(by_id) != len(native):
            raise ValueError("SecVulEval idx must be unique")
        reliable, referenced = {}, set()
        for r in native:
            if type(r["is_vulnerable"]) is not bool:
                raise ValueError("SecVulEval requires native boolean labels")
            fixed = r["fixed_func_idx"]
            other = by_id.get(int(fixed)) if fixed is not None and math.isfinite(fixed) and fixed == int(fixed) else None
            if not r["is_vulnerable"]:
                continue
            if other is None:
                issues["missing_fixed_id"] += 1
            elif other["is_vulnerable"]:
                issues["fixed_label_not_zero"] += 1
            elif any(r[k] != other[k] for k in ("project_url", "filepath", "func_name", "commit_id")):
                issues["repair_identity_mismatch"] += 1
            elif r["func_body"] == other["func_body"]:
                issues["unchanged_repair_source"] += 1
            else:
                reliable[r["idx"]] = other["idx"]
                referenced.add(other["idx"])
        reverse = {}
        for old, new in reliable.items():
            reverse.setdefault(new, []).append(old)
        for r in native:
            idx = r["idx"]
            partner = reliable.get(idx)
            if partner is None and idx in reverse:
                # Multiple antecedents must imply identical positive targets.
                candidates = reverse[idx]
                if len(candidates) == 1:
                    partner = candidates[0]
                else:
                    masks = {tuple(_line_diff(r["func_body"], by_id[i]["func_body"])[0]) for i in candidates}
                    if len(masks) == 1:
                        partner = candidates[0]
                    else:
                        issues["ambiguous_repaired_diff"] += 1
            key = f"secvuleval:{idx}"
            row = dict(sample_key=key, pair_key=key, target=int(r["is_vulnerable"]),
                func=r["func_body"], project=r["project_url"], commit_id=r["commit_id"],
                cves=sorted(set(r["cve_list"] or [])), file_name=r["filepath"],
                func_name=r["func_name"], source_row=idx,
                normal_kind="vulnerable" if r["is_vulnerable"] else "repaired" if idx in referenced else "ordinary",
                repair_key=f"secvuleval:{partner}" if partner is not None else None)
            rows.append(row)
            if partner is not None:
                counterparts[key] = by_id[partner]["func_body"]
        source = str(paths[0])
    elif dataset == "megavul":
        source = str(raw_root / "megavul/megavul.json")
        repaired = {}
        for i, r in enumerate(json_array(source), 1):
            if type(r["is_vul"]) is not bool:
                raise ValueError("MegaVul requires native boolean labels")
            label = int(r["is_vul"])
            code = r["func_before"] if label else r["func"]
            key = f"megavul:{i}"
            identity = (r["repo_name"], r["commit_hash"], r["file_path"], r["func_name"])
            row = dict(sample_key=key, pair_key=key, target=label, func=code,
                project=r["repo_name"], commit_id=r["commit_hash"], cves=[r["cve_id"]] if r["cve_id"] else [],
                file_name=r["file_path"], func_name=r["func_name"], source_row=i,
                normal_kind="vulnerable" if label else "ordinary", repair_key=None)
            rows.append(row)
            if label:
                after = r["func"]
                if isinstance(after, str) and after.strip() and after != code:
                    counterparts[key] = after
                    repaired.setdefault(identity, set()).add(after)
                else:
                    issues["missing_or_unchanged_repair_source"] += 1
            if i % 100000 == 0:
                print(f"MegaVul: loaded {i} native members", flush=True)
        for r in rows:
            identity = (r["project"], r["commit_id"], r["file_name"], r["func_name"])
            if r["target"] == 0 and r["func"] in repaired.get(identity, set()):
                r["normal_kind"] = "repaired"
                issues["negative_matches_repaired_source"] += 1
        # func in an is_vul=True record is a counterpart, not a native target=0
        # member. In particular, never synthesize negatives from these records.
    else:
        raise ValueError(f"unknown RGCL dataset: {dataset}")
    for r in rows:
        if not isinstance(r["func"], str) or not r["func"].strip():
            raise ValueError(f"{r['sample_key']}: empty native source")
        if not r["file_name"].endswith((".c", ".h", ".C", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".c++")):
            raise ValueError(f"{r['sample_key']}: inspect unsupported language {r['file_name']}")
    return rows, counterparts, dict(source=source, official_evaluation_split=False,
        native_labels="is_vulnerable" if dataset == "secvuleval" else "is_vul",
        raw_functions=len(rows), native_class_counts=dict(Counter(str(r["target"]) for r in rows)),
        normal_kinds=dict(Counter(r["normal_kind"] for r in rows)), reliable_counterparts=len(counterparts),
        issues=dict(issues), labels_reconstructed=False, synthesized_fixed_negatives=0)


def split_rgcl_groups(rows, counterparts, seed=42):
    """80/10/10 class targets; connected CVE/commit/repair/duplicate groups."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    import hashlib
    index, keys, duplicate_codes = {}, {r["sample_key"]: i for i, r in enumerate(rows)}, Counter()
    for i, r in enumerate(rows):
        values = [("cve", c.upper()) for c in r["cves"] if c]
        if r["commit_id"]:
            values.append(("commit", r["commit_id"]))
        for code in (r["func"], counterparts.get(r["sample_key"])):
            if code is not None:
                digest = hashlib.sha256(" ".join(code.split()).encode()).digest()
                values.append(("code", digest))
        duplicate_codes[hashlib.sha256(" ".join(r["func"].split()).encode()).digest()] += 1
        for value in values:
            if value in index:
                union(i, index[value])
            else:
                index[value] = i
        if r["repair_key"] in keys:
            union(i, keys[r["repair_key"]])
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda k: -len(groups[k]))
    total = Counter(r["target"] for r in rows)
    targets = {s: {y: total[y] * fraction for y in (0, 1)} for s, fraction in zip(SPLITS, (.8, .1, .1))}
    counts = {s: Counter() for s in SPLITS}
    for key in order:
        labels = Counter(rows[i]["target"] for i in groups[key])
        cost = lambda s: sum(((counts[s][y] + labels[y] - targets[s][y]) ** 2 -
                              (counts[s][y] - targets[s][y]) ** 2) / max(1, targets[s][y]) for y in (0, 1))
        split = min(SPLITS, key=cost)
        counts[split].update(labels)
        for i in groups[key]:
            rows[i].update(split=split, group_key=f"group:{key}")
    if any(not counts[s][0] or not counts[s][1] for s in SPLITS):
        raise ValueError("connected groups cannot provide both labels in every split")
    # Recheck every grouping key; no CVE, commit, counterpart or duplicate may cross.
    for i, r in enumerate(rows):
        assert rows[find(i)]["split"] == r["split"]
        if r["repair_key"] in keys:
            assert r["split"] == rows[keys[r["repair_key"]]]["split"]
    return dict(seed=seed, target_ratios=[.8, .1, .1], groups=len(groups),
        largest_groups=sorted((len(g) for g in groups.values()), reverse=True)[:10],
        grouping="transitive union of CVE, fix commit, repair link and whitespace-normalized source (including counterparts)",
        cross_split_group_overlap=0, members_removed=0,
        duplicate_code_groups=sum(n > 1 for n in duplicate_codes.values()))


def prepare_rgcl(raw_root, output_dir, dataset, model_path):
    """Native function classification and separate reliable Diff annotations."""
    from transformers import RobertaTokenizerFast

    dest = Path(output_dir) / dataset
    if any((dest / f"{s}.jsonl").exists() for s in SPLITS):
        raise FileExistsError(f"{dest}: native members already prepared")
    rows, counterparts, statistics = load_rgcl_native(raw_root, dataset)
    statistics["split_protocol"] = split_rgcl_groups(rows, counterparts)
    if dataset == "megavul":
        # Group the complete native corpus before sampling, including negatives
        # that will not be used for classification.
        full_rows = rows
        rows = balanced(full_rows, seed=42, label_key="target")
        selected_keys = {r["sample_key"] for r in rows}
        statistics["negative_sampling"] = dict(seed=42,
            protocol="group full native C/C++ corpus first; retain all positives; independently sample equal native negatives within each split",
            selected_functions=len(rows), positive_members_removed=0,
            negative_members_removed=len(full_rows) - len(rows),
            splits={s: dict(
                before=dict(Counter(str(r["target"]) for r in full_rows if r["split"] == s)),
                selected=dict(Counter(str(r["target"]) for r in rows if r["split"] == s)))
                for s in SPLITS})
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / "grouped_members.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["sample_key", "target", "split", "group_key", "selected"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(dict(r, selected=int(r["sample_key"] in selected_keys)) for r in full_rows)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False)
    statistics.update(model_path=str(model_path), input_tokens=512, splits={})
    for split in SPLITS:
        selected = [r for r in rows if r["split"] == split]
        clean, annotated, coverage = [], [], Counter()
        for r in selected:
            other = counterparts.get(r["sample_key"])
            truth = _line_diff(r["func"], other)[0] if other is not None else []
            annotation = dict(r, lines_ground_truth=truth, diff_reliable=other is not None)
            f = paired_features(annotation, tokenizer, "baseline")
            mask = f["line_labels"]
            visible = set(f["line_numbers"])
            complete = {n for n, ok in zip(f["line_numbers"], f["line_complete"]) if ok}
            coverage.update(functions=1, reliable_counterpart=other is not None,
                with_diff_lines=bool(truth), visible_diff=bool(set(truth) & visible),
                partial_visible_diff=bool(set(truth) & (visible - complete)),
                effective_region=1 in mask and 0 in mask,
                no_region_function_bce_only=not (1 in mask and 0 in mask),
                truncated=f["source_tokens"] > 510)
            for y in (0, 1):
                if r["target"] == y:
                    coverage[f"target_{y}_functions"] += 1
                    coverage[f"target_{y}_effective_region"] += int(1 in mask and 0 in mask)
            clean.append(r)
            annotated.append(annotation)
            if len(clean) % 2000 == 0:
                print(f"{dataset} {split}: mapped {len(clean)}/{len(selected)} functions", flush=True)
        write_jsonl(dest / f"{split}.jsonl", clean)
        write_jsonl(dest / f"{split}.lines.jsonl", annotated)
        statistics["splits"][split] = dict(functions=len(selected),
            class_counts=dict(Counter(str(r["target"]) for r in selected)),
            normal_kinds=dict(Counter(r["normal_kind"] for r in selected)), coverage=dict(coverage),
            actual_fraction=len(selected) / len(rows))
        print(f"{dataset} {split}: {statistics['splits'][split]}", flush=True)
    train = statistics["splits"]["train"]["class_counts"]
    statistics["classification_positive_weight"] = train["0"] / train["1"]
    statistics["classification_weight_rule"] = "Train negatives / positives; same weighted BCE for all three methods"
    write_json(dest / "statistics.json", statistics)
    with (dest / "members.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["sample_key", "target", "split", "group_key", "normal_kind", "repair_key"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return statistics


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


def balanced(rows, seed=42, *, label_key="label"):
    """Retain every positive; sample only the negatives, independently per Split."""
    selected = set()
    for split in SPLITS:
        positive = [
            i for i, r in enumerate(rows) if r["split"] == split and r[label_key] == 1
        ]
        negative = [
            i for i, r in enumerate(rows) if r["split"] == split and r[label_key] == 0
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


def _line_diff(source, counterpart, *, return_alignment=False):
    """Original LOSVER Differ labels and edit coordinates from the same alignment."""
    import difflib

    result, edits, position, other, edit = [], [], 0, 0, None
    unchanged, blocks, block = [], [], None
    for line in difflib.Differ().compare(
        source.splitlines(keepends=True), counterpart.splitlines(keepends=True)
    ):
        if line.startswith(("- ", "+ ")) and edit is None:
            edit = dict(start=position, end=position, other_lines=[])
            block = [position, position, other, other]
        if line.startswith("- "):
            if line[2:].strip():
                result.append(position)
            position += 1
            edit["end"] = position
            block[1] = position
        elif line.startswith("+ "):
            if line[2:].strip():
                edit["other_lines"].append(other)
            other += 1
            block[3] = other
        elif line.startswith("  "):
            if edit is not None:
                edits.append(edit)
                blocks.append(block)
                edit = None
            if return_alignment:
                unchanged.append((position, other))
            position += 1
            other += 1
    if edit is not None:
        edits.append(edit)
        blocks.append(block)
    if return_alignment:
        return result, edits, dict(unchanged=unchanged, blocks=blocks)
    return result, edits


def rccl_annotations(rows, features, mode, seed=123456):
    """Unique exact Diff matches on fully visible lines; never change line labels.

    Edit neighbors are the first nonblank equal lines on either side of a block.
    Blank equal lines can be crossed, another edit cannot. An ambiguous or
    incomplete nearest code line is skipped rather than replaced by a distant one.
    Random controls draw the same count from all reliable equal lines in the pair.
    """
    from collections import Counter
    import random

    if mode not in ("baseline", "diff", "random") or len(rows) != len(features) or len(rows) % 2:
        raise ValueError("RCCL requires complete official pairs and baseline/diff/random")
    rng, records = random.Random(seed), []
    for i in range(0, len(rows), 2):
        old, new = rows[i:i + 2]
        a, b = features[i:i + 2]
        if old["target"] != 1 or new["target"] != 0 or old["pair_key"] != new["pair_key"]:
            raise ValueError("RCCL pair order must be vulnerable, repaired with the same pair key")
        deleted, edits, alignment = _line_diff(old["func"], new["func"], return_alignment=True)
        if deleted != old.get("lines_ground_truth", deleted):
            raise ValueError("RCCL source alignment differs from original Diff labels")
        sources = [r["func"].splitlines(keepends=True) for r in (old, new)]
        frequencies = [Counter(s) for s in sources]
        slots = [{n: k for k, n in enumerate(f["line_numbers"]) if f["line_complete"][k]
                  and f["line_token_indices"][k]} for f in (a, b)]
        equal = set(alignment["unchanged"])
        reliable, ambiguous, incomplete = {}, 0, 0
        for left, right in alignment["unchanged"]:
            text = sources[0][left]
            if not text.strip():
                continue
            if text != sources[1][right]:
                raise ValueError("Diff equal line contents differ")
            if frequencies[0][text] != 1 or frequencies[1][text] != 1:
                ambiguous += 1
            elif left not in slots[0] or right not in slots[1]:
                incomplete += 1
            else:
                reliable[(left, right)] = (slots[0][left], slots[1][right])
        neighbors = set()
        for start, end, other_start, other_end in alignment["blocks"]:
            for left, right, direction in ((start - 1, other_start - 1, -1), (end, other_end, 1)):
                while (left, right) in equal and not sources[0][left].strip():
                    left, right = left + direction, right + direction
                if (left, right) in reliable:
                    neighbors.add((left, right))
        different = a["input_ids"] != b["input_ids"]
        near = [reliable[p] for p in sorted(neighbors)] if different else []
        pool = [reliable[p] for p in sorted(reliable)]
        selected = sorted(rng.sample(pool, len(near))) if mode == "random" else near
        for f in (a, b):
            f["rccl_pair_key"] = old["pair_key"]
        a.update(rccl_near=near, rccl_selected=selected, rccl_alignment=pool)
        b.update(rccl_near=[], rccl_selected=[], rccl_alignment=[])
        records.append(dict(
            pair_key=old["pair_key"], sample_keys=[old["sample_key"], new["sample_key"]],
            pure_addition=not deleted and any(e["other_lines"] for e in edits),
            visible_inputs_differ=different, edit_blocks=len(edits),
            ambiguous_equal_lines=ambiguous, incomplete_equal_lines=incomplete,
            reliable_equal_lines=len(pool), eligible=bool(near),
            near_lines=[[a["line_numbers"][x], b["line_numbers"][y]] for x, y in near],
            selected_lines=[[a["line_numbers"][x], b["line_numbers"][y]] for x, y in selected],
        ))
    return records


def rccl_coverage(records):
    """Pair denominators include invisible, ambiguous and empty pairs."""
    result = {}
    for name, members in (("all", records), ("pure_addition", [r for r in records if r["pure_addition"]])):
        eligible = sum(r["eligible"] for r in members)
        result[name] = dict(pairs=len(members), eligible_pairs=eligible,
                            pair_coverage=eligible / len(members) if members else None,
                            visible_inputs_differ=sum(r["visible_inputs_differ"] for r in members),
                            selected_lines=sum(len(r["selected_lines"]) for r in members),
                            ambiguous_equal_lines=sum(r["ambiguous_equal_lines"] for r in members),
                            incomplete_equal_lines=sum(r["incomplete_equal_lines"] for r in members))
    return result


def modified_lines(source, counterpart):
    """Exact LOSVER Differ rule: nonblank deleted lines, separately on each side."""
    return _line_diff(source, counterpart)[0]


def insertion_annotations(rows):
    """Pure additions from the SAME Differ alignment, in old-side gap coordinates.

    Gap p is before old physical line p; 0/N are the function boundaries.
    A replacement of blank old lines is an interval [start,end], not a guessed
    neighboring code label. Only old nonblank deletions disqualify a pair.
    """
    result = []
    if len(rows) % 2:
        raise ValueError("insertion annotations require complete vulnerable/patched pairs")
    for i in range(0, len(rows), 2):
        old, new = rows[i:i+2]
        if [old["target"], new["target"]] != [1, 0] or old["pair_key"] != new["pair_key"]:
            raise ValueError("insertion annotations require unchanged official pair order")
        deleted, edits = _line_diff(old["func"], new["func"])
        if "lines_ground_truth" in old and deleted != old["lines_ground_truth"]:
            raise ValueError("original Diff and insertion alignment disagree")
        added = [e for e in edits if e["other_lines"]]
        pure = not deleted and bool(added)
        result.extend((dict(old, pure_addition_pair=pure, insertion_edits=added if pure else []),
                       dict(new, pure_addition_pair=pure, insertion_edits=[])))
    return result


def insertion_features(feature, edits):
    """Code-only gaps; collapse blank-run positions with identical two sides.

    Each range contains physical insertion coordinates, never code-line labels.
    Both real sides must be fully encoded. End-of-visible-prefix is NOT EOF.
    Zero vectors represent the real function boundaries, without parameters.
    """
    numbers, full = feature["line_numbers"], feature["line_complete"]
    left, right, ranges = [], [], []
    for gap in range(len(numbers) + 1):
        a, b = gap - 1, gap
        if not numbers or (a >= 0 and not full[a]) or (b < len(numbers) and not full[b]):
            continue
        if b == len(numbers) and feature["source_tokens"] > 510:
            continue
        left.append(a)
        right.append(b if b < len(numbers) else -1)
        ranges.append([numbers[a] + 1 if a >= 0 else 0,
                       numbers[b] if b < len(numbers) else feature["source_lines"]])
    positive = {i for i, (a, b) in enumerate(ranges)
                if any(a <= e["start"] <= e["end"] <= b for e in edits)}
    labels = [int(i in positive) for i in range(len(ranges))] if positive else [-1] * len(ranges)
    return dict(insertion_left=left, insertion_right=right, insertion_ranges=ranges,
                insertion_labels=labels, insertion_truth=list(labels))


def _neighbour_line_kinds(source):
    """Lexical boundaries only: comments/literals cannot create fake braces."""
    from bisect import bisect_right
    from .syntax import source_token_spans

    lines = source.splitlines(keepends=True)
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    tokens = [[] for _ in lines]
    for token, a, b in source_token_spans(source):
        n = max(0, bisect_right(starts, a) - 1)
        while n < len(lines) and starts[n] < b:
            if source[max(a, starts[n]) : min(b, starts[n] + len(lines[n]))].strip():
                tokens[n].append(token)
            n += 1
    kinds, preprocessor = [], False
    for line, ts in zip(lines, tokens):
        if preprocessor or line.lstrip().startswith("#"):
            kinds.append("preprocessor_boundary")
            preprocessor = line.rstrip("\r\n").endswith("\\")
        elif not ts:
            kinds.append("blank_or_comment")
        elif any(t in ("{", "}") for t in ts):
            # A brace-bearing line terminates this deliberately local search.
            kinds.append("block_boundary")
        elif all(t in ("(", ")", "[", "]", ";", ",", ":", "\\") for t in ts):
            kinds.append("delimiter")
        else:
            kinds.append("code")
    return kinds


def diff_neighbour_supervision(rows, encodings):
    """Binary Diff labels plus bounded edit-adjacent proxies on a missing side.

    The original positive set must be empty in the FULL source, not just after
    truncation. A whole opposite edit must be fully visible; a partially visible
    edit is conservatively ineligible. Each direction can cross one blank/comment
    physical line, but no brace/preprocessor/delimiter line. No semantic expansion.
    """
    if len(rows) != 2 or len(encodings) != 2:
        raise ValueError("diff_neighbour requires one aligned pair")
    v, p = rows
    if v["pair_key"] != p["pair_key"] or [v["target"], p["target"]] != [1, 0]:
        raise ValueError("unaligned official pair")
    if any(e["input_mode"] != "512" for e in encodings):
        raise ValueError("diff_neighbour requires 512 input")
    identical = encodings[0]["input_ids"] == encodings[1]["input_ids"]
    result, records = [], []
    for row, other, e, oe in (
        (v, p, encodings[0], encodings[1]),
        (p, v, encodings[1], encodings[0]),
    ):
        original, edits = _line_diff(row["func"], other["func"])
        if original != row["lines_ground_truth"]:
            raise ValueError("stored Diff labels differ from the original alignment")
        eligible = not original and bool(other["lines_ground_truth"])
        record = dict(
            pair_key=row["pair_key"],
            sample_key=row["sample_key"],
            target=row["target"],
            eligible=eligible,
            identical_input=identical,
            events=[],
        )
        added = set()
        if not eligible:
            record["reason"] = (
                "original_positive_present" if original else "no_opposite_positive"
            )
        elif identical:
            record["reason"] = "identical_input"
        else:
            kinds = _neighbour_line_kinds(row["func"])
            visible = {n for n, full in zip(e["line_numbers"], e["line_complete"]) if full}
            opposite_visible = {
                n for n, full in zip(oe["line_numbers"], oe["line_complete"]) if full
            }
            for edit in edits:
                if not edit["other_lines"]:
                    continue
                event = dict(edit, selected={}, rejected={})
                record["events"].append(event)
                if not set(edit["other_lines"]).issubset(opposite_visible):
                    event["reason"] = "opposite_edit_not_fully_visible"
                    continue
                for direction, at, step in (
                    ("before", edit["start"] - 1, -1),
                    ("after", edit["end"], 1),
                ):
                    skipped = 0
                    while 0 <= at < len(kinds) and kinds[at] == "blank_or_comment":
                        skipped += 1
                        if skipped > 1:
                            break
                        at += step
                    reason = None
                    if skipped > 1:
                        reason = "more_than_one_blank_or_comment"
                    elif not 0 <= at < len(kinds):
                        reason = "source_boundary"
                    elif kinds[at] != "code":
                        reason = kinds[at]
                    elif at not in visible:
                        reason = "neighbour_not_fully_visible"
                    if reason:
                        event["rejected"][direction] = reason
                    else:
                        event["selected"][direction] = at
                        added.add(at)
                event["reason"] = (
                    "selected" if event["selected"] else "no_qualified_neighbour"
                )
            record["reason"] = "supplemented" if added else "no_qualified_neighbour"
        record["added_lines"] = sorted(added)
        record["selection"] = (
            "both"
            if any(len(ev["selected"]) == 2 for ev in record["events"])
            else "single" if added else "none"
        )
        result.append(
            dict(
                row,
                lines_original_diff=original,
                lines_ground_truth=sorted(set(original) | added),
                neighbour_lines=sorted(added),
            )
        )
        records.append(record)
    return result, records


def prepare_diff_neighbour(data_dir, output_dir, model_path):
    """Generate only Train/Valid proxies; preserve all original rows and labels."""
    import time
    from transformers import RobertaTokenizerFast

    root = Path(output_dir) / "supervision" / "diff_neighbour"
    if root.exists():
        raise FileExistsError(f"{root}: supervision already exists")
    root.mkdir(parents=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False)
    statistics = dict(
        input_mode="512",
        model_path=str(model_path),
        splits={},
        test_supervision_generated=False,
        rule=(
            "same LOSVER Differ; one code line before/after each edit; "
            "at most one blank/comment per direction; braces/delimiters/preprocessor stop; "
            "complete visibility on both sides; no identical input"
        ),
        localizer_selection="original Valid Diff positive F1, not proxy F1",
    )
    started = time.monotonic()
    for split in ("train", "valid"):
        rows = read_jsonl(Path(data_dir) / "primevul_paired" / f"{split}.lines.jsonl")
        generated, records, counts = [], [], Counter()
        for i in range(0, len(rows), 2):
            pair = rows[i : i + 2]
            encodings = [paired_encoding(r["func"], tokenizer) for r in pair]
            supplemented, details = diff_neighbour_supervision(pair, encodings)
            generated.extend(supplemented)
            records.extend(details)
            for row, e, detail in zip(pair, encodings, details):
                visible = set(e["line_numbers"])
                counts.update(
                    functions=1,
                    visible_lines=len(visible),
                    original_visible_positives=len(
                        set(row["lines_ground_truth"]) & visible
                    ),
                    added_positives=len(detail["added_lines"]),
                )
                if detail["eligible"]:
                    prefix = "pure_addition" if row["target"] else "pure_deletion"
                    counts[prefix + "_pairs"] += 1
                    counts[prefix + "_covered"] += bool(detail["added_lines"])
                    counts[prefix + "_added_positives"] += len(detail["added_lines"])
                    counts[prefix + "_" + detail["selection"]] += 1
        counts["pairs"] = len(rows) // 2
        counts["augmented_visible_positives"] = (
            counts["original_visible_positives"] + counts["added_positives"]
        )
        statistics["splits"][split] = dict(
            counts,
            eligible_reasons=dict(Counter(r["reason"] for r in records if r["eligible"])),
            rejection_reasons=dict(
                Counter(
                    why for r in records for ev in r["events"]
                    for why in ev["rejected"].values()
                )
            ),
            edit_outcomes=dict(Counter(ev["reason"] for r in records for ev in r["events"])),
            added_lines_distribution=dict(
                Counter(len(r["added_lines"]) for r in records if r["eligible"])
            ),
            edit_selection=dict(Counter(len(ev["selected"]) for r in records for ev in r["events"])),
        )
        write_jsonl(root / f"{split}.jsonl", generated)
        write_jsonl(root / f"{split}.audit.jsonl", records)
        print(split, json.dumps(statistics["splits"][split]), flush=True)
    statistics["seconds"] = time.monotonic() - started
    write_json(root / "statistics.json", statistics)
    return statistics


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


def repair_region_supervision(
    vulnerable, patched, *, relation=True, anomalous=False, diffs=None
):
    """Conservative, source-aligned weak labels, never vulnerability-location truth.

    Direct=1/1 (target/weight); related=1/.6; heuristic=-1/0;
    structural background=0/.2; all other/failed/ambiguous lines=-1/0.
    No arbitrary unmodified executable line is forced to be negative.
    """
    import re
    from .syntax import (
        parse_source,
        statement_units,
        source_lines,
        edit_hunks,
        walk,
        text,
        source_token_spans,
        CONTROLS,
    )

    codes = (vulnerable, patched)
    parsed = [parse_source(c) for c in codes]
    diffs = (
        [
            set(modified_lines(vulnerable, patched)),
            set(modified_lines(patched, vulnerable)),
        ]
        if diffs is None
        else [set(d) for d in diffs]
    )
    result = []
    for code in codes:
        lines = code.splitlines(keepends=True)
        starts, offset = [], 0
        for line in lines:
            starts.append(offset)
            offset += len(line)
        result.append(
            dict(
                region_labels=[-1.0] * len(lines),
                region_weights=[0.0] * len(lines),
                region_levels=["unknown"] * len(lines),
                region_spans=[],
                region_reasons=[],
                region_loss_active=False,
                physical_offsets=[(a, a + len(l)) for a, l in zip(starts, lines)],
            )
        )
    if anomalous or (
        parsed[0]["name"]
        and parsed[1]["name"]
        and parsed[0]["name"] != parsed[1]["name"]
    ):
        for r in result:
            r["region_reasons"].append("pair_correspondence_unknown")
        return result

    for side, (code, p, r, diff) in enumerate(zip(codes, parsed, result, diffs)):
        code_lines = set()
        from bisect import bisect_right

        starts = [a for a, b in r["physical_offsets"]]
        for token, a, b in source_token_spans(code):
            code_lines.update(
                range(bisect_right(starts, a) - 1, bisect_right(starts, b - 1))
            )

        def assign(ns, level, q, weight):
            for n in sorted(ns):
                if (
                    n < 0
                    or n >= len(r["region_labels"])
                    or not code.splitlines()[n].strip()
                ):
                    continue
                if weight > r["region_weights"][n]:
                    r["region_labels"][n] = q
                    r["region_weights"][n] = weight
                    r["region_levels"][n] = level

        # Genuine unchanged structural background only; executable non-candidates
        # remain unlabeled. Modified comments are unknown rather than negatives.
        background = {
            n
            for n, line in enumerate(code.splitlines())
            if n not in diff
            and line.strip()
            and (n not in code_lines or re.fullmatch(r"[{};]+", line.strip()))
        }
        assign(background, "structural_background", 0.0, 0.2)
        direct = diff & code_lines
        assign(direct, "direct", 1.0, 1.0)
        if not p["strict"]:
            r["region_reasons"].append("parse_failure_direct_only")
            continue
        units, names = statement_units(p)
        # Expand direct changes to a bounded atomic expression, not an entire
        # initializer list, compound statement, or multi-variable declaration.
        for u in units:
            if not (u["lines"] & direct):
                continue
            nodes = list(walk(u["node"]))
            broad = any(
                n.type in {"initializer_list", "lambda_expression"} for n in nodes
            )
            multi = len(u["node"].children_by_field_name("declarator")) > 1
            if (
                not broad
                and not multi
                and len(u["lines"]) <= 8
                and len(text(u["node"], p["raw"])) <= 400
            ):
                assign(u["lines"] & code_lines, "direct", 1.0, 1.0)
        if not relation:
            continue
        other = parsed[1 - side]
        if not other["strict"]:
            r["region_reasons"].append("counterpart_parse_failure")
            continue
        other_units, other_names = statement_units(other)

        def accesses(unit, source, bindings):
            found = set()
            for n in walk(unit["node"]):
                if n.type not in {"identifier", "field_expression"}:
                    continue
                if n.type == "identifier" and n.parent.type == "field_expression":
                    continue
                if n.type == "field_expression" and n.parent.type == "field_expression":
                    continue
                name = text(n, source)
                root = re.split(r"->|\.", name)[0].strip()
                if len(bindings.get(root, [])) == 1:
                    found.add((root, re.sub(r"\s+", "", name)))
            return found

        own_access = [accesses(u, p["raw"], names) for u in units]
        anchors = []
        if side == 0:
            for h in edit_hunks(vulnerable, patched):
                if h["tag"] == "insert":
                    anchors.append((h["v_start"], set(range(h["p_start"], h["p_end"]))))
        else:
            for h in edit_hunks(patched, vulnerable):
                if h["tag"] == "insert":
                    anchors.append((h["v_start"], set(range(h["p_start"], h["p_end"]))))
        chosen = set(i for i, u in enumerate(units) if u["lines"] & direct)
        for anchor, changed in anchors:
            refs = (
                set().union(
                    *(
                        accesses(u, other["raw"], other_names)
                        for u in other_units
                        if u["lines"] & changed
                    )
                )
                if other_units
                else set()
            )
            for root, path in sorted(refs):
                if len(names.get(root, [])) != 1 or len(other_names.get(root, [])) != 1:
                    r["region_reasons"].append("ambiguous_binding:" + root)
                    continue
                candidates = [
                    i
                    for i, u in enumerate(units)
                    if (root, path) in own_access[i]
                    and abs(u["node"].start_point.row - anchor) <= 8
                ]
                if not candidates:
                    r["region_reasons"].append("no_bounded_bound_use:" + path)
                    continue
                i = min(
                    candidates,
                    key=lambda i: (abs(units[i]["node"].start_point.row - anchor), i),
                )
                u = units[i]
                ns = u["lines"] & code_lines
                if len(ns) > 8 or any(
                    n.type == "initializer_list" for n in walk(u["node"])
                ):
                    r["region_reasons"].append("broad_ast_unit_unknown")
                    continue
                operators = [
                    n
                    for n in walk(u["node"])
                    if n.type
                    in {
                        "binary_expression",
                        "subscript_expression",
                        "pointer_expression",
                        "assignment_expression",
                        "update_expression",
                    }
                ]
                reliable = any(
                    (root, path) in accesses(dict(node=n), p["raw"], names)
                    and not any(x.type == "call_expression" for x in walk(n))
                    for n in operators
                )
                if reliable:
                    assign(ns, "bound_relation", 1.0, 0.6)
                    chosen.add(i)
                else:
                    r["region_reasons"].append("unknown_heuristic_relation:" + path)
        # One bounded lexical def-use hop; branch feasibility is NOT proven.
        for i in sorted(chosen):
            unit = units[i]
            for binding in unit["refs"]:
                before = [
                    j
                    for j, u in enumerate(units)
                    if binding in u["writes"]
                    and u["node"].end_byte <= unit["node"].start_byte
                    and unit["node"].start_point.row - u["node"].start_point.row <= 8
                ]
                if before:
                    j = max(before, key=lambda j: units[j]["node"].end_byte)
                    ns = units[j]["lines"] & code_lines
                    if len(ns) <= 4 and not any(
                        n.type == "initializer_list" for n in walk(units[j]["node"])
                    ):
                        r["region_reasons"].append("unknown_lexical_definition")
            parent = unit["node"].parent
            while parent is not None and parent != p["function"]:
                if parent.type in CONTROLS:
                    cond = parent.child_by_field_name("condition")
                    if cond and len(source_lines(cond)) <= 2:
                        assign(
                            source_lines(cond) & code_lines,
                            "control_relation",
                            1.0,
                            0.6,
                        )
                parent = parent.parent
        # At most eight additional non-direct lines, deterministic by proximity.
        extras = [
            n
            for n, l in enumerate(r["region_levels"])
            if l not in {"direct", "unknown", "structural_background"}
        ]
        origin = direct or {a for a, _ in anchors}
        extras.sort(key=lambda n: (min((abs(n - x) for x in origin), default=0), n))
        for n in extras[8:]:
            r["region_labels"][n] = -1.0
            r["region_weights"][n] = 0.0
            r["region_levels"][n] = "unknown"
        if len(extras) > 8:
            r["region_reasons"].append("eight_extra_lines_limit")
    for r in result:
        r["region_loss_active"] = any(y == 1 for y in r["region_labels"])
        r["region_spans"] = [
            dict(line=n, start=a, end=b, level=l)
            for n, ((a, b), l) in enumerate(
                zip(r["physical_offsets"], r["region_levels"])
            )
            if l != "unknown"
        ]
    return result


def prepare_repair_regions(data_dir, output_dir):
    """Train/Valid supervision only. Test Diff is deliberately never opened."""
    import time

    root = Path(output_dir) / "supervision"
    if (root / "statistics.json").exists():
        raise FileExistsError(f"{root}: supervision already prepared")
    root.mkdir(parents=True, exist_ok=True)
    statistics = {}
    started = time.monotonic()
    for split in ("train", "valid"):
        rows = read_jsonl(Path(data_dir) / "primevul_paired" / f"{split}.lines.jsonl")
        methods = {"diff": [], "repair_region": [], "neighbour_region": []}
        for i in range(0, len(rows), 2):
            v, p = rows[i : i + 2]
            if v["pair_key"] != p["pair_key"] or [v["target"], p["target"]] != [1, 0]:
                raise ValueError("unaligned official pair")
            # Intrinsic metadata only; no manual Test judgments or member deletion.
            vi, pi = v.get("big_vul_idx"), p.get("big_vul_idx")
            anomalous = (
                v.get("project") != p.get("project")
                or v.get("commit_id") != p.get("commit_id")
                or (vi is not None and pi is not None and vi != pi)
            )
            related = repair_region_supervision(
                v["func"],
                p["func"],
                anomalous=anomalous,
                diffs=(v["lines_ground_truth"], p["lines_ground_truth"]),
            )
            import copy

            direct = copy.deepcopy(related)
            for r in direct:
                for n, l in enumerate(r["region_levels"]):
                    if l not in {"direct", "structural_background"}:
                        r["region_labels"][n] = -1.0
                        r["region_weights"][n] = 0.0
                        r["region_levels"][n] = "unknown"
                r["region_loss_active"] = any(y == 1 for y in r["region_labels"])
                r["region_spans"] = [
                    s
                    for s in r["region_spans"]
                    if r["region_levels"][s["line"]] != "unknown"
                ]
            for row, d, r in zip((v, p), direct, related):
                methods["diff"].append(dict(row, **d))
                methods["repair_region"].append(dict(row, **r))
                # Same extra-line count and confidence multiset; nearest available
                # executable lines are a control for simply enlarging the region.
                n = copy.deepcopy(d)
                extra = [
                    j
                    for j, l in enumerate(r["region_levels"])
                    if l not in {"direct", "unknown", "structural_background"}
                ]
                candidates = [
                    j
                    for j, l in enumerate(n["region_levels"])
                    if l == "unknown" and row["func"].splitlines()[j].strip()
                ]
                from .syntax import edit_hunks

                other = p["func"] if row is v else v["func"]
                anchors = set(row["lines_ground_truth"]) or {
                    h["v_start"] for h in edit_hunks(row["func"], other)
                }
                candidates.sort(
                    key=lambda j: (min((abs(j - a) for a in anchors), default=j), j)
                )
                for dst, src in zip(candidates, extra):
                    n["region_labels"][dst] = r["region_labels"][src]
                    n["region_weights"][dst] = r["region_weights"][src]
                    n["region_levels"][dst] = "neighbour_control"
                n["region_spans"] = [
                    dict(line=j, start=a, end=b, level=l)
                    for j, ((a, b), l) in enumerate(
                        zip(n["physical_offsets"], n["region_levels"])
                    )
                    if l != "unknown"
                ]
                n["region_loss_active"] = any(y == 1 for y in n["region_labels"])
                methods["neighbour_region"].append(dict(row, **n))
            if (i // 2 + 1) % 500 == 0:
                print(
                    f"{split}: {i//2+1} pairs, {time.monotonic()-started:.1f}s",
                    flush=True,
                )
        for name, rr in methods.items():
            write_jsonl(root / name / f"{split}.jsonl", rr)
            statistics[f"{name}:{split}"] = dict(
                functions=len(rr),
                levels=dict(Counter(l for r in rr for l in r["region_levels"])),
                labels=dict(
                    Counter(str(int(y)) for r in rr for y in r["region_labels"])
                ),
                no_positive_functions=sum(not r["region_loss_active"] for r in rr),
                reasons=dict(Counter(l for r in rr for l in r["region_reasons"])),
            )
    write_json(
        root / "statistics.json",
        dict(
            seconds=time.monotonic() - started,
            splits=statistics,
            weights=dict(
                direct=1.0, bound_relation=0.6, structural_background=0.2, unknown=0.0
            ),
            targets=dict(
                direct=1, bound_relation=1, structural_background=0, unknown=-1
            ),
            focal=dict(alpha=0.5, gamma=1.5, pos_weight=1.0),
            test_supervision_generated=False,
        ),
    )
    return statistics


def paired_features(row, tokenizer, variant, *, encoding=None, input_mode="512", insertion=False):
    """Attach labels or binary 5/1 weights to the same source-aligned input."""
    if variant not in ("localizer", "baseline", "predicted", "oracle"):
        raise ValueError(f"unknown paired variant: {variant}")
    f = dict(
        paired_encoding(row["func"], tokenizer, input_mode)
        if encoding is None
        else encoding
    )
    if insertion:
        if variant != "baseline" or input_mode != "512":
            raise ValueError("insertion localization requires the shared 512 baseline input")
        f.update(insertion_features(f, row.get("insertion_edits", [])))
    if variant == "localizer":
        gt = set(row.get("lines_ground_truth", []))
        f["labels"] = [int(n in gt) for n in f["line_numbers"]]
        if "region_labels" in row:
            f["labels"] = [row["region_labels"][n] for n in f["line_numbers"]]
            f["label_weights"] = [row["region_weights"][n] for n in f["line_numbers"]]
            if 1 not in f["labels"]:
                f["label_weights"] = [0.0] * len(f["labels"])
    else:
        f["labels"] = row["target"]
        if variant == "baseline" and "lines_ground_truth" in row:
            gt = set(row["lines_ground_truth"])
            f["line_labels"] = [int(n in gt) for n in f["line_numbers"]]
            if "diff_reliable" in row:
                complete = {n for n, ok in zip(f["line_numbers"], f["line_complete"]) if ok}
                visible = gt.intersection(f["line_numbers"])
                # A partial edited line cannot be relabeled as background.
                # Skip its function's auxiliary term; classification is retained.
                active = row["diff_reliable"] and bool(visible) and visible.issubset(complete)
                if not active:
                    f["line_labels"] = [0] * len(f["line_numbers"])
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
    line_scores = dict(zip(row.get("line_numbers", []), row.get("line_scores", [])))
    if "line_scores" in row and (
        list(line_scores) != f["line_numbers"]
        or any(not 0 <= p <= 1 for p in line_scores.values())
    ):
        raise ValueError("region scores must align to visible physical lines")
    scores = [0.0] * len(f["input_ids"]) if line_scores else None
    for number, tokens in zip(f["line_numbers"], f["line_token_indices"]):
        if number in selected:
            for token in tokens:
                weights[token] = 5.0
        if scores is not None:
            for token in tokens:
                scores[token] = max(scores[token], line_scores.get(number, 0.0))
    f["token_weights"] = weights
    if scores is not None:
        f["region_scores"] = scores
    return f


def paired_statistics(rows, features, *, paired=True):
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
    if paired:
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
