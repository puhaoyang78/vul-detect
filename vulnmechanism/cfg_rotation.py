"""PrimeVul train-only negative rotation for the unchanged CFG classifier."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import random

from .cfg_data import (atomic_json, build_graphs, cohort_hash, digest, file_sha256,
                       load_graphs, output_lock, read_jsonl, read_records,
                       source_hash, abstract_cfg)
from .prepare_benchmark import (NearIndex, clean_label_conflicts, link_counterparts,
                                load_sources, official_counterparts, unit)
from .syntax import resolve_language, target_hint

ROTATION_VARIANTS = ("cfg_rotation_fixed", "cfg_rotation_rotating")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(partial, path)


def screen_candidate_units(units, official_edges, protected_rows, base_rows, seed):
    """Reuse benchmark conflict, counterpart, exact, normalized, and near rules."""
    linked = link_counterparts(units, official_edges)
    consistent, conflicts = clean_label_conflicts(linked)
    protected_by_key = {r["sample_key"]: r for r in protected_rows}
    if len(protected_by_key) != len(protected_rows):
        raise ValueError("duplicate key in benchmark manifest")
    for row in base_rows:
        old = protected_by_key.get(row["sample_key"])
        if (old is None or old["function"] != row["raw_source"] or
                old["label"] != row["label"] or old["split"] != row["split"] or
                old["dataset"] != row["dataset"]):
            raise ValueError(f"original C member differs from benchmark manifest: {row['sample_key']}")
    seen = {}
    near = NearIndex()
    protected_units = [unit([r]) for r in protected_rows]
    # Protect every official train vulnerable function, including ones without a graph.
    protected_units.extend(u for u in consistent if u["dataset"] == "primevul" and
                           u["split"] == "train" and u["rows"][0]["label"] == 1 and
                           u["rows"][0]["sample_key"] not in protected_by_key)
    for item in protected_units:
        for key in item["keys"]:
            if key[0] != "repo_commit":
                seen.setdefault(key, item["id"])
        near.add(item)
    pool = sorted((u for u in consistent if u["dataset"] == "primevul" and
                   u["split"] == "train" and u["rows"][0]["label"] == 0),
                  key=lambda u: u["rows"][0]["sample_key"])
    random.Random(seed).shuffle(pool)
    counts = Counter(language_eligible_train_benign=sum(
                         u["dataset"] == "primevul" and u["split"] == "train" and
                         u["rows"][0]["label"] == 0 for u in linked),
                     train_benign_after_label_conflicts=len(pool),
                     label_conflict_units=len(conflicts),
                     label_conflict_train_benign=sum(
                         e["dataset"] == "primevul" and e["split"] == "train" and e["labels"] == [0]
                         for e in conflicts),
                     protected_functions=len(protected_units))
    chosen = []
    for item in pool:
        row = item["rows"][0]
        if (row.get("official_split") != "train" or
                row.get("source_file") != "PrimeVul_v0.1/primevul_train.jsonl"):
            raise ValueError(f"candidate is not from official PrimeVul train: {item['id']}")
        if row["sample_key"] in protected_by_key:
            counts["previously_used"] += 1
            continue
        matches = [k[0] for k in item["keys"] if k[0] != "repo_commit" and k in seen]
        if matches:
            counts["overlap_" + sorted(matches)[0]] += 1
            continue
        if near.matches(item):
            counts["overlap_near_duplicate"] += 1
            continue
        chosen.append(row)
        for key in item["keys"]:
            if key[0] != "repo_commit":
                seen.setdefault(key, item["id"])
        near.add(item)
    counts["eligible"] = len(chosen)
    return chosen, dict(sorted(counts.items()))


def _reference_config(c_run_dir):
    root = Path(c_run_dir)
    config = json.loads((root / "config.json").read_text())
    if (config.get("source_dataset") != "primevul" or config.get("seed") != 42 or
            config.get("epochs") != 3 or config.get("source_max_length") != 2048 or
            not (root / "cfg" / "complete.json").is_file()):
        raise ValueError("reference must be the completed original PrimeVul C run (3 epochs, seed 42, 2048 tokens)")
    return root, config


def prepare_rotation(raw_root, benchmark_manifest, c_run_dir, output_dir):
    reference, config = _reference_config(c_run_dir)
    raw_root = Path(raw_root).resolve()
    output = Path(output_dir).resolve()
    if output == raw_root or raw_root in output.parents:
        raise ValueError("rotation output must be outside the raw dataset")
    if (output / "preparation.json").exists() or (output / "candidates.jsonl").exists():
        raise FileExistsError("rotation preparation already exists; build from this directory")
    original = read_records(config["dataset"], "primevul")
    if cohort_hash(original) != config["cohort_sha256"]:
        raise ValueError("original C dataset changed")
    negatives = sum(r["split"] == "train" and r["label"] == 0 for r in original)
    if not negatives:
        raise ValueError("original C has no train negatives")
    protected = read_jsonl(benchmark_manifest)
    existing_all = read_jsonl(config["dataset"])
    units, _, source_exclusions = load_sources(raw_root)
    official_edges, _ = official_counterparts(raw_root)
    chosen, screening = screen_candidate_units(units, official_edges, protected, existing_all, config["seed"])
    needed = 2 * negatives
    if len(chosen) < needed:
        raise ValueError(f"only {len(chosen)} isolated PrimeVul train negatives; need {needed}")
    raw_train = raw_root / "PrimeVul_v0.1" / "primevul_train.jsonl"
    by_line = {r["source_row"]: r for r in chosen}
    if len(by_line) != len(chosen):
        raise ValueError("duplicate official source_row among eligible candidates")
    offsets = {}
    with raw_train.open("rb") as stream:
        line = 0
        while content := stream.readline():
            line += 1
            if line in by_line:
                offsets[line] = stream.tell() - len(content)
    if set(offsets) != set(by_line):
        raise ValueError("official PrimeVul train source rows changed")
    candidates = [dict(rank=i, sample_key=r["sample_key"], source_row=r["source_row"],
                       byte_offset=offsets[r["source_row"]], source_sha256=source_hash(r["function"]),
                       language=r["language"], file_name=r["file_name"],
                       counterpart_groups=r.get("counterpart_groups", []))
                  for i, r in enumerate(chosen)]
    report = dict(seed=config["seed"], reference_run_dir=str(reference.resolve()),
                  reference_config_sha256=digest(config), original_cohort_sha256=cohort_hash(original),
                  original_graph_sha256=config["graph_file_sha256"],
                  benchmark_manifest=str(Path(benchmark_manifest).resolve()),
                  benchmark_manifest_sha256=file_sha256(benchmark_manifest),
                  raw_train=str(raw_train), raw_train_sha256=file_sha256(raw_train),
                  negative_slots=negatives, required_new_negatives=needed,
                  candidate_count=len(candidates), screening=screening,
                  source_exclusion_counts=dict(sorted(Counter(
                      e["reason"] for e in source_exclusions).items())))
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output / "rotation"):
        if (output / "preparation.json").exists() or (output / "candidates.jsonl").exists():
            raise FileExistsError("rotation preparation already exists; keep it and build from this directory")
        write_jsonl(output / "candidates.jsonl", candidates)
        report["candidates_sha256"] = file_sha256(output / "candidates.jsonl")
        atomic_json(output / "preparation.json", report)
    return report


def _preparation(directory):
    directory = Path(directory)
    report = json.loads((directory / "preparation.json").read_text())
    if file_sha256(directory / "candidates.jsonl") != report["candidates_sha256"]:
        raise ValueError("rotation candidate index changed")
    reference, config = _reference_config(report["reference_run_dir"])
    if digest(config) != report["reference_config_sha256"]:
        raise ValueError("original C configuration changed")
    return report, config


def _candidate_record(candidate, raw_stream):
    raw_stream.seek(candidate["byte_offset"])
    original = json.loads(raw_stream.readline())
    source = original["func"]
    if (original["target"] != 0 or f"primevul:{original['idx']}" != candidate["sample_key"] or
            source_hash(source) != candidate["source_sha256"]):
        raise ValueError(f"official PrimeVul candidate changed: {candidate['sample_key']}")
    language = resolve_language(source, candidate["language"], candidate["file_name"])
    hint = target_hint(source, language).name
    return dict(schema_version=9, sample_key=candidate["sample_key"], dataset="primevul",
                split="train", label=0, raw_source=source, language=candidate["language"],
                resolved_language=language, function_name=hint, syntax_hint=hint,
                official_source_row=candidate["source_row"])


def build_rotation(directory, *, joern_dir="/home/phy/joern", java_home="/home/phy/jdk21",
                   timeout=300, batch_size=8, extractor=None):
    directory = Path(directory)
    report, _ = _preparation(directory)
    if file_sha256(report["raw_train"]) != report["raw_train_sha256"]:
        raise ValueError("official PrimeVul train file changed since candidate preparation")
    candidates = read_jsonl(directory / "candidates.jsonl")
    if len(candidates) != report["candidate_count"]:
        raise ValueError("candidate count changed")
    needed = report["required_new_negatives"]
    parts = directory / "parts"
    parts.mkdir(exist_ok=True)
    with output_lock(directory / "rotation"):
        if (directory / "selection.json").exists():
            return load_selected(directory, expected_count=needed)[2]
        selected_records, selected_graph_rows = [], []
        cursor, failed = 0, 0
        with Path(report["raw_train"]).open("rb") as raw_stream:
            while len(selected_records) < needed:
                remaining = needed - len(selected_records)
                batch_candidates = candidates[cursor:cursor + remaining]
                if not batch_candidates:
                    raise ValueError(f"only {len(selected_records)} usable new graphs; need {needed}; "
                                     f"{failed} candidates failed; see {parts}")
                prefix = parts / f"{cursor:06d}"
                dataset_path = Path(str(prefix) + ".dataset.jsonl")
                graphs_path = Path(str(prefix) + ".graphs.jsonl")
                done_path = Path(str(prefix) + ".done.json")
                if dataset_path.exists():
                    batch = read_records(dataset_path, "primevul")
                    if [r["sample_key"] for r in batch] != [c["sample_key"] for c in batch_candidates]:
                        raise ValueError("existing rotation graph batch differs from candidate order")
                else:
                    batch = [_candidate_record(c, raw_stream) for c in batch_candidates]
                    write_jsonl(dataset_path, batch)
                if not done_path.exists():
                    build_graphs(str(dataset_path), str(graphs_path), joern_dir=joern_dir,
                                 java_home=java_home, timeout=timeout, batch_size=batch_size,
                                 extractor=extractor)
                    usable = load_graphs(graphs_path, batch, require_complete=False)
                    missing = {r["sample_key"] for r in batch} - set(usable)
                    errors_path = Path(str(graphs_path) + ".errors.jsonl")
                    logged = {r["sample_key"] for r in read_jsonl(errors_path)} if errors_path.exists() else set()
                    if not missing <= logged:
                        raise RuntimeError("graph batch stopped before recording every failed candidate")
                    atomic_json(done_path, dict(dataset_sha256=file_sha256(dataset_path),
                                                graphs_sha256=file_sha256(graphs_path),
                                                attempted=len(batch), usable=len(usable), failed=len(missing)))
                done = json.loads(done_path.read_text())
                if (done["dataset_sha256"] != file_sha256(dataset_path) or
                        done["graphs_sha256"] != file_sha256(graphs_path)):
                    raise ValueError("completed rotation graph batch changed")
                usable = load_graphs(graphs_path, batch, require_complete=False)
                raw_graph_rows = {r["sample_key"]: r for r in read_jsonl(graphs_path)}
                for row in batch:
                    key = row["sample_key"]
                    if key in usable:
                        selected_records.append(row)
                        selected_graph_rows.append(raw_graph_rows[key])
                    else:
                        failed += 1
                cursor += len(batch)
        if len(selected_records) != needed:
            raise ValueError("rotation graph selection count mismatch")
        write_jsonl(directory / "selected.jsonl", selected_records)
        write_jsonl(directory / "selected.graphs.jsonl", selected_graph_rows)
        selection = dict(selected_count=needed, attempted=cursor, failed=failed,
                         selected_sample_keys=[r["sample_key"] for r in selected_records],
                         selected_sha256=file_sha256(directory / "selected.jsonl"),
                         selected_graphs_sha256=file_sha256(directory / "selected.graphs.jsonl"))
        atomic_json(directory / "selection.json", selection)
        return selection


def load_selected(directory, *, expected_count=None):
    directory = Path(directory)
    report, config = _preparation(directory)
    selected = json.loads((directory / "selection.json").read_text())
    if (expected_count is not None and selected["selected_count"] != expected_count or
            selected["selected_count"] != report["required_new_negatives"] or
            file_sha256(directory / "selected.jsonl") != selected["selected_sha256"] or
            file_sha256(directory / "selected.graphs.jsonl") != selected["selected_graphs_sha256"]):
        raise ValueError("rotation selection is incomplete or changed")
    rows = read_records(directory / "selected.jsonl", "primevul")
    candidates = {r["sample_key"]: r for r in read_jsonl(directory / "candidates.jsonl")}
    ranks = [candidates[r["sample_key"]]["rank"] for r in rows if r["sample_key"] in candidates]
    if (len(rows) != selected["selected_count"] or
            [r["sample_key"] for r in rows] != selected["selected_sample_keys"] or
            len(ranks) != len(rows) or ranks != sorted(ranks) or
            any(r["sample_key"] not in candidates or r["label"] != 0 or r["split"] != "train" or
                source_hash(r["raw_source"]) != candidates[r["sample_key"]]["source_sha256"]
                for r in rows)):
        raise ValueError("rotation selected records differ from screened candidates")
    graphs = load_graphs(directory / "selected.graphs.jsonl", rows)
    views = {key: abstract_cfg(graph) for key, graph in graphs.items()}
    return rows, views, selected


def epoch_train_rows(original_train, selected, variant, epochs):
    if variant not in ROTATION_VARIANTS or epochs != 3:
        raise ValueError("rotation requires fixed/rotating variants and three epochs")
    negatives = [r for r in original_train if r["label"] == 0]
    positives = [r for r in original_train if r["label"] == 1]
    if (len(selected) != 2 * len(negatives) or len({r["sample_key"] for r in selected}) != len(selected) or
            any(r["label"] != 0 or r["split"] != "train" for r in selected)):
        raise ValueError("rotation requires two disjoint complete new negative sets")
    schedule = []
    for epoch in range(epochs):
        replacement = (negatives if variant == "cfg_rotation_fixed" or epoch == 0 else
                       selected[(epoch - 1) * len(negatives):epoch * len(negatives)])
        iterator = iter(replacement)
        rows = [row if row["label"] == 1 else next(iterator) for row in original_train]
        if (len(rows) != len(original_train) or [r["label"] for r in rows] !=
                [r["label"] for r in original_train] or
                [r["sample_key"] for r in rows if r["label"] == 1] !=
                [r["sample_key"] for r in positives]):
            raise ValueError("rotation changed train slots or vulnerable member order")
        schedule.append(rows)
    return schedule


def schedule_report(schedule):
    seen_negatives = set()
    previous = []
    result = []
    for i, rows in enumerate(schedule, 1):
        keys = {r["sample_key"] for r in rows}
        negatives = {r["sample_key"] for r in rows if r["label"] == 0}
        result.append(dict(epoch=i, sample_keys=[r["sample_key"] for r in rows],
                           positive_count=sum(r["label"] == 1 for r in rows),
                           negative_count=len(negatives),
                           previous_negative_overlap=len(negatives & seen_negatives),
                           pairwise_total_overlap=[len(keys & earlier) for earlier in previous]))
        seen_negatives.update(negatives)
        previous.append(keys)
    return result
