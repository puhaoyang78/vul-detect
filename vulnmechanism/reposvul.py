"""Inspect native ReposVul labels and repair evidence before preparing training."""

from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path

from .artifacts import write_json
from .data import SPLITS, _line_diff, paired_features, split_rgcl_groups, write_jsonl


CC_LANGUAGES = {"c", "h", "cpp", "cc", "cxx", "c++", "hh", "hpp", "hxx"}


def _digest(source, normalize=False):
    # Used in memory for duplicate grouping, never exported as a new identifier.
    return hashlib.sha256((" ".join(source.split()) if normalize else source).encode()).digest()


def _identity(row):
    return row["project"], row["commit_id"], row["file_name"], _digest(row["func"])


def _quality(rows):
    result = {}
    for normalized in (False, True):
        groups = defaultdict(list)
        for r in rows:
            groups[_digest(r["func"], normalized)].append(r)
        conflicts = [g for g in groups.values() if len({r["target"] for r in g}) > 1]
        result["normalized" if normalized else "exact"] = dict(
            unique_sources=len(groups), duplicate_groups=sum(len(g) > 1 for g in groups.values()),
            conflicting_groups=len(conflicts), conflicting_functions=sum(len(g) for g in conflicts),
            conflicting_positive_functions=sum(r["target"] == 1 for g in conflicts for r in g),
            examples=[[dict(sample_key=r["sample_key"], target=r["target"], cves=r["cves"], project=r["project"],
                            commit_id=r["commit_id"], file_name=r["file_name"]) for r in g[:8]]
                      for g in conflicts[:12]])
    return result


def _raw_rows(path):
    rows, pairs, issues, languages = [], [], Counter(), Counter()
    counts, fields = Counter(), Counter()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            counts["records"] += 1
            fields.update(r.keys())
            for fi, d in enumerate(r["details"]):
                language = d["file_language"].lower()
                languages[language] += 1
                if language not in CC_LANGUAGES:
                    continue
                counts["cc_files"] += 1
                sides = {}
                for side in ("before", "after"):
                    functions = d.get(f"function_{side}")
                    if functions is None:
                        issues[f"missing_function_{side}_array"] += 1
                        functions = []
                    if not isinstance(functions, list):
                        raise ValueError(f"record {number}, file {fi}: invalid function array")
                    sides[side] = []
                    for j, f in enumerate(functions):
                        if type(f["target"]) is not int or f["target"] not in (0, 1):
                            raise ValueError(f"record {number}, file {fi}, {side}:{j}: invalid native 0/1 label")
                        if not isinstance(f["function"], str) or not f["function"].strip():
                            raise ValueError(f"record {number}, file {fi}, {side}:{j}: empty source")
                        key = f"reposvul:{number}:{fi}:{side}:{j}"
                        row = dict(sample_key=key, pair_key=key, target=f["target"], func=f["function"],
                            project=r["project"], commit_id=r["commit_id"], cves=[r["cve_id"]],
                            file_name=d["file_name"], file_language=language, source_record=number,
                            source_index=r["index"], side=side, repair_key=None,
                            normal_kind="vulnerable" if f["target"] else "ordinary")
                        rows.append(row)
                        sides[side].append(row)
                        counts[f"{side}_target_{row['target']}"] += 1
                        counts[f"language_{language}_functions"] += 1
                # Conservative lower bound: exact unique declaration in the same
                # file, plus source membership in both original full-file texts.
                # Never align by the two lists' ordinal positions or similarity.
                by_header = {}
                for side, functions in sides.items():
                    table = defaultdict(list)
                    for row in functions:
                        header = " ".join(row["func"].partition("{")[0].split())
                        table[header].append(row)
                    by_header[side] = table
                for header, old in by_header["before"].items():
                    for a in old:
                        if a["target"] != 1:
                            continue
                        new = by_header["after"].get(header, [])
                        if len(old) != 1 or len(new) != 1:
                            issues["positive_missing_or_ambiguous_declaration"] += 1
                        elif new[0]["target"] != 0:
                            issues["counterpart_label_not_zero"] += 1
                        elif (not isinstance(d.get("code_before"), str) or
                              not isinstance(d.get("code"), str) or
                              a["func"] not in d["code_before"] or new[0]["func"] not in d["code"]):
                            issues["function_not_in_corresponding_full_file"] += 1
                        else:
                            b = new[0]
                            a["repair_key"], b["repair_key"] = b["sample_key"], a["sample_key"]
                            b["normal_kind"] = "repaired"
                            pairs.append((a, b))
            if number % 1000 == 0:
                print(f"ReposVul raw: inspected {number} records", flush=True)
    return rows, pairs, dict(source=str(path), source_bytes=Path(path).stat().st_size,
        counts=dict(counts), file_languages=dict(languages), issues=dict(issues),
        native_labels="function_before/after[].target; file target never supplies a function label",
        official_split_in_local_file=any(k in fields for k in ("split", "train", "valid", "test")),
        association_rule="unique exact declaration within the same file and both sources present in corresponding full files; unmatched declarations remain unknown",
        labels_reconstructed=False, synthesized_classification_members=0)


def inspect(raw_root, output_dir, model_path, *, official_dir=None):
    """Export evidence only: no training members, label fixes or model fitting."""
    from transformers import RobertaTokenizerFast

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows, pairs, result = _raw_rows(Path(raw_root)/"ReposVul_c.jsonl")
    result.update(functions=len(rows), class_counts=dict(Counter(str(r["target"]) for r in rows)),
                  normal_kinds=dict(Counter(r["normal_kind"] for r in rows)), label_quality=_quality(rows))
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False)
    coverage, records, repair_index = Counter(), [], defaultdict(set)
    for a, b in pairs:
        same = a["func"] == b["func"]
        record = dict(before=a["sample_key"], after=b["sample_key"], cve=a["cves"][0],
            project=a["project"], commit_id=a["commit_id"], file_name=a["file_name"],
            identical_source=same, same_normalized_source=_digest(a["func"], True) == _digest(b["func"], True))
        coverage["matched_positive_pairs"] += 1
        coverage["identical_source_opposite_label_pairs"] += int(same)
        record["sides"] = {}
        signatures = []
        for row, other in ((a, b), (b, a)):
            truth = _line_diff(row["func"], other["func"])[0]
            feature = paired_features(dict(row, lines_ground_truth=truth, diff_reliable=not same), tokenizer, "baseline")
            mask = feature["line_labels"]
            effective = 1 in mask and 0 in mask
            values = dict(diff_lines=len(truth), visible_diff_lines=sum(x == 1 for x in mask),
                          effective_region=effective, source_tokens=feature["source_tokens"],
                          truncated=feature["source_tokens"] > 510)
            record["sides"][row["side"]] = values
            coverage[f"target_{row['target']}_effective_region"] += int(effective)
            coverage[f"target_{row['target']}_has_diff_lines"] += bool(truth)
            signatures.append((tuple(feature["input_ids"]), tuple(feature["code_mask"]),
                               tuple(tuple(t) for t in feature["line_token_indices"])))
        record["identical_512_model_input"] = signatures[0] == signatures[1]
        coverage["identical_512_model_input_pairs"] += int(record["identical_512_model_input"])
        if not same:
            repair_index[_identity(a)].add(b["func"])
        records.append(record)
    result.update(input_tokens=512, coverage=dict(coverage), model_path=str(model_path),
        coverage_denominator="all native C/C++ functions; only conservatively matched pairs are tokenized for region coverage")
    write_jsonl(root/"repair_pairs.jsonl", records)
    if official_dir is not None:
        official, splits, indices, field_names = [], {}, {}, set()
        raw_labels = defaultdict(set)
        for r in rows:
            raw_labels[_identity(r)].add(r["target"])
        for split in SPLITS:
            counts, fields, members = Counter(), Counter(), []
            lookup = {k:set() for k in ("cve", "commit", "exact_source", "normalized_source", "function_id")}
            with (Path(official_dir)/f"{split}.jsonl").open(encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    fields.update(r.keys())
                    if r["file_language"].lower() not in CC_LANGUAGES:
                        counts["non_cc_skipped"] += 1
                        continue
                    if type(r["target"]) is not int or r["target"] not in (0, 1):
                        raise ValueError(f"official {split}:{number}: invalid native label")
                    if not isinstance(r["function"], str) or not r["function"].strip():
                        raise ValueError(f"official {split}:{number}: empty source")
                    row = dict(sample_key=f"reposvul:official:{split}:{number}", pair_key=r["function_id"],
                        func=r["function"], target=r["target"], project=r["project"], commit_id=r["commit_id"],
                        cves=[r["cve_id"]], file_name=r["file_name"], repair_key=None, split=split)
                    official.append(row)
                    members.append(row)
                    counts[f"target_{row['target']}"] += 1
                    counts[f"language_{r['file_language'].lower()}"] += 1
                    labels = raw_labels.get(_identity(row), set())
                    counts["exact_local_source_match"] += bool(labels)
                    counts["native_label_agrees_with_local"] += row["target"] in labels
                    counts[f"target_{row['target']}_local_label_disagreement"] += bool(labels) and row["target"] not in labels
                    candidates = repair_index.get(_identity(row), set()) if row["target"] == 1 else set()
                    counts["unique_local_changed_counterpart"] += len(candidates) == 1
                    if len(candidates) == 1:
                        other = next(iter(candidates))
                        truth = _line_diff(row["func"], other)[0]
                        f = paired_features(dict(row, lines_ground_truth=truth, diff_reliable=True), tokenizer, "baseline")
                        counts["positive_effective_region"] += 1 in f["line_labels"] and 0 in f["line_labels"]
                    for key, values in (("cve", row["cves"]), ("commit", [row["commit_id"]]),
                        ("exact_source", [_digest(row["func"])]), ("normalized_source", [_digest(row["func"], True)]),
                        ("function_id", [r["function_id"]])):
                        lookup[key].update(v for v in values if v)
            splits[split] = dict(functions=len(members), counts=dict(counts), fields=dict(fields))
            field_names.update(fields)
            indices[split] = lookup
            print(f"ReposVul official {split}: {splits[split]}", flush=True)
        overlap = {f"{a}:{b}":{k:len(indices[a][k] & indices[b][k]) for k in indices[a]}
                   for a,b in combinations(SPLITS,2)}
        result["official"] = dict(directory=str(official_dir), splits=splits, overlap=overlap,
            label_quality=_quality(official),
            repair_versions_in_release=bool(field_names & {"function_before", "function_after", "func_before", "func_after", "fixed_func", "fixed_func_idx"}),
            local_repair_matching="same project, fix commit, file and exact original function; never by function_id alone",
            reliable_grouped_split=all(n == 0 for pair in overlap.values() for n in pair.values()))
    # Diagnose the required grouping without writing or selecting a balanced
    # training cohort. It does not repair the native label contradictions.
    try:
        grouped = dict(split_rgcl_groups(rows, {}, seed=42), feasible=True)
    except ValueError as error:
        if str(error) != "connected groups cannot provide both labels in every split":
            raise
        grouped = dict(feasible=False, error=str(error), seed=42)
    grouped["splits"] = {s:dict(functions=sum(r.get("split") == s for r in rows),
        class_counts=dict(Counter(str(r["target"]) for r in rows if r.get("split") == s))) for s in SPLITS}
    result.update(grouped_split_feasibility=grouped, training_started=False,
        classification_members_prepared=False, negative_downsampling_performed=False,
        test_predictions_used=False)
    write_json(root/"data_inspection.json", result)
    return result


def check(official_dir, output_dir, *, device="auto"):
    """Frozen historical models on official Train sources; no optimizer or fit."""
    import torch
    from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
    from . import model as core
    from .evaluation import load_paired
    from .rgcl_study import run_path
    from .training import RGCL_METHODS, paired_collate, paired_forward

    device = core.resolve_device(device)
    if device.type != "cuda":
        raise ValueError("ReposVul inference smoke requires a real GPU")
    rows = []
    with (Path(official_dir)/"train.jsonl").open() as handle:
        for number, line in enumerate(handle, 1):
            r = json.loads(line)
            if r["file_language"].lower() in CC_LANGUAGES:
                rows.append(dict(sample_key=f"reposvul:official:train:{number}", pair_key=r["function_id"],
                    func=r["function"], target=r["target"], lines_ground_truth=[], diff_reliable=False))
            if len(rows) == 8:
                break
    if len(rows) != 8:
        raise ValueError("inference smoke requires eight native official Train functions")
    result = dict(device=str(device), gpu=torch.cuda.get_device_name(device), functions=8,
        source="official Train only; existing PrimeVul checkpoints frozen", training_started=False,
        test_used=False, methods={})
    for method in RGCL_METHODS:
        path = run_path("results/rgcl_multiseed_512", "primevul_paired", 123456, method)/"best-p_c.pt"
        original = path.stat()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model, tokenizer = load_paired(checkpoint, device)
        model.eval()
        if method == "plm_baseline" and not isinstance(model.classifier, RobertaClassificationHead):
            raise ValueError("CLS baseline does not contain RobertaClassificationHead")
        if method == "rgcl":
            bounded = checkpoint["bounded_attention"]
            if bounded["k"] != 5 or bounded["region_weight"] != .1 or bounded.get("region_weighting", "token") != "token":
                raise ValueError("the original RGCL checkpoint protocol changed")
        batch = paired_collate([paired_features(r, tokenizer, "baseline") for r in rows])
        with torch.no_grad():
            probabilities = paired_forward(model, batch, "baseline", device, with_labels=False)[:, 0].cpu()
            altered = dict(batch, line_labels=torch.ones_like(batch["line_labels"]))
            other = paired_forward(model, altered, "baseline", device, with_labels=False)[:, 0].cpu()
        torch.testing.assert_close(probabilities, other, rtol=0, atol=0)
        single = core.predict_checkpoint(path, [dict(raw_source=rows[0]["func"])], device=device).sigmoid()
        torch.testing.assert_close(single.cpu(), probabilities[:1], rtol=1e-5, atol=1e-6)
        assert torch.isfinite(probabilities).all()
        assert original.st_size == path.stat().st_size and original.st_mtime_ns == path.stat().st_mtime_ns
        result["methods"][method] = dict(checkpoint=str(path), finite=True, diff_independent=True,
            single_function_max_delta=float((single.cpu()-probabilities[:1]).abs().max()),
            checkpoint_unchanged=True, native_train_members=[r["sample_key"] for r in rows])
        del model, checkpoint
        torch.cuda.empty_cache()
    result["passed"] = True
    write_json(Path(output_dir)/"gpu-smoke.json", result)
    return result
