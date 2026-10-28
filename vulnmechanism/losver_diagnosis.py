"""Read-only LOSVER diagnostics; inference interventions never update parameters."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .artifacts import write_json
from .data import paired_encoding, paired_features, read_jsonl
from .metrics import paired_metrics


MODES = ("512", "l1")
SPLITS = ("train", "valid", "test")
METHODS = ("unixcoder", "losver", "losver_oracle")
SEED = 42
POLICIES = (
    "equal_1", "predicted_2", "predicted_3", "predicted_5", "predicted_10",
    "random_42", "random_43", "random_44", "true_5", "predicted_tp_5",
    "predicted_fp_5",
)


def run_dir(mode, method):
    return Path("results") / f"losver_{mode}" / "primevul_paired" / method


def annotations(split):
    name = "test.oracle.jsonl" if split == "test" else f"{split}.lines.jsonl"
    return read_jsonl(Path("data/primevul_paired") / name)


def quantiles(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return dict(zip(("min", "q25", "median", "q75", "q90", "max"),
                    np.quantile(x, [0, .25, .5, .75, .9, 1]).tolist()))


def mean(values):
    x = [float(x) for x in values if x is not None and math.isfinite(x)]
    return float(np.mean(x)) if x else None


def binary_summary(labels, probabilities):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = p > .5
    tp = int(((y == 1) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    both = len(set(y)) == 2
    return dict(
        n=len(y), positive=int(y.sum()), tp=tp, tn=tn, fp=fp, fn=fn,
        precision=tp/(tp+fp) if tp+fp else 0.,
        recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        accuracy=(tp+tn)/len(y) if len(y) else None,
        mcc=(tp*tn-fp*fn)/denominator if denominator else 0.,
        auc=float(roc_auc_score(y, p)) if both else None,
        auprc=float(average_precision_score(y, p)) if both else None,
    )


def calibration(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    wrong = (p > .5) != y
    extreme = (p < .01) | (p > .99)
    buckets = np.minimum((p * 10).astype(int), 9)
    bins = []
    for index in range(10):
        at = buckets == index
        if at.any():
            bins.append(dict(bin=index, n=int(at.sum()), probability=float(p[at].mean()),
                             positive_rate=float(y[at].mean())))
    return dict(
        brier=float(brier_score_loss(y, p)),
        ece=sum(b["n"]*abs(b["probability"]-b["positive_rate"]) for b in bins)/len(y),
        extreme_fraction=float(extreme.mean()),
        saturated_fraction=float(((p == 0) | (p == 1)).mean()),
        wrong=int(wrong.sum()), confident_wrong=int((wrong & extreme).sum()),
        confident_fraction_of_wrong=float(extreme[wrong].mean()) if wrong.any() else None,
        probability=quantiles(p), bins=bins,
    )


def pair_summary(rows, probabilities):
    p = np.asarray(probabilities)
    result = paired_metrics(rows, p.tolist())
    gap = p[::2] - p[1::2]
    return dict(result, ranking_accuracy=float((gap > 0).mean()),
                ranking_ties=int((gap == 0).sum()),
                ranking_half_ties=float(((gap > 0) + .5*(gap == 0)).mean()),
                probability_gap=quantiles(gap), calibration=calibration(
                    [r["target"] for r in rows], p))


def pair_states(probabilities):
    p = np.asarray(probabilities) > .5
    return ["C" if a and not b else "V" if a and b else "B" if not a and not b else "R"
            for a, b in zip(p[::2], p[1::2])]


def transition(first, second):
    names = ("C", "V", "B", "R")
    counts = Counter(zip(pair_states(first), pair_states(second)))
    return dict(order=names, counts=[[counts[a, b] for b in names] for a in names],
                denominator=len(first)//2)


def selected_mass(selected, total, strength=5.):
    if total <= 0 or not 0 <= selected <= total:
        raise ValueError("invalid unique code-token counts")
    amplification = math.exp(strength-1)
    return selected*amplification / (total-selected + selected*amplification)


def line_tokens(feature, selected):
    return {token-1 for line, tokens in zip(feature["line_numbers"],
                                          feature["line_token_indices"])
            if line in selected for token in tokens}


def length_group(tokens):
    return "<=510" if tokens <= 510 else "511-2048" if tokens <= 2048 else "2049-8192" if tokens <= 8192 else ">8192"


def window_group(count):
    return "1" if count == 1 else "2-4" if count <= 4 else "5-16" if count <= 16 else "17+"


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("empty diagnostic table")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_functions(output):
    with (Path(output)/"functions.csv").open() as f:
        rows = list(csv.DictReader(f))
    text = {"mode", "split", "sample_key", "pair_key", "project", "length_group", "window_group"}
    for row in rows:
        for k, v in row.items():
            if k not in text:
                row[k] = float(v) if v else None
    return rows


def line_summary(rows):
    tp = sum(r["line_tp"] for r in rows)
    fp = sum(r["line_fp"] for r in rows)
    fn = sum(r["line_fn"] for r in rows)
    n = sum(r["visible_lines"] for r in rows)
    positive = tp+fn
    return dict(
        functions=len(rows), lines=n, modified_lines=positive,
        positive_rate=positive/n if n else None, tp=tp, fp=fp, fn=fn, tn=n-tp-fp-fn,
        precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        macro_f1_with_visible_gt=mean([r["line_f1"] for r in rows if r["gt_visible"]]),
        no_prediction=sum(r["predicted_lines"] == 0 for r in rows),
        no_visible_gt=sum(r["gt_visible"] == 0 for r in rows),
        missed_all_gt=sum(r["gt_visible"] > 0 and r["line_tp"] == 0 for r in rows),
        any_gt_hit=sum(r["line_tp"] > 0 for r in rows),
        predicted_lines=quantiles([r["predicted_lines"] for r in rows]),
        predicted_line_fraction=quantiles([r["predicted_lines"]/r["visible_lines"] for r in rows]),
        selected_token_fraction=quantiles([r["selected_token_fraction"] for r in rows]),
        selected_attention_mass=quantiles([r["selected_mass"] for r in rows]),
        false_attention_mass=quantiles([r["false_mass"] for r in rows]),
        true_attention_mass=quantiles([r["true_mass"] for r in rows]),
        false_mass_above_half=sum(r["false_mass"] > .5 for r in rows),
        false_mass_above_80pct=sum(r["false_mass"] > .8 for r in rows),
        effective_token_fraction=quantiles([r["effective_token_fraction"] for r in rows]),
    )


def collect(output):
    from transformers import RobertaTokenizerFast

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = {s: read_jsonl(Path("data/primevul_paired")/f"{s}.jsonl") for s in SPLITS}
    probe_pairs = sorted(random.Random(SEED).sample(range(len(data["train"])//2), 512))
    probe = [data["train"][2*i+j]["sample_key"] for i in probe_pairs for j in (0, 1)]
    write_json(output/"config.json", dict(
        source_commit="67fc7b4076115a88dd68e72fad53868956403e0f", seed=SEED,
        data="unchanged official PrimeVul 4354/562/564 pairs", threshold=.5,
        selection="source Valid P-C; localizer source Valid modified-line positive F1",
        modes=MODES, policies=POLICIES, train_inference_members=probe,
        inference_scope="512 Train pairs, all Valid and Test; identical members for all heads/policies",
        random_rule="sample the same count of visible physical lines without replacement; per-function RNG seed + 1000003*(split_index+1) + source_row",
        random_seeds=[42, 43, 44], inference_only=True, parameter_updates=0,
        length_bins=[510, 2048, 8192], window_bins=[1, 4, 16],
        test_usage="post-hoc diagnostics only; no model/threshold/policy selection",
    ))
    first = json.loads((run_dir("l1", "unixcoder")/"config.json").read_text())
    tokenizer = RobertaTokenizerFast.from_pretrained(first["model_path"], trim_offsets=False)
    scores = {m: {s: {r["sample_key"]: r for r in read_jsonl(
        run_dir(m, "losver_localizer")/f"{s}.line_scores.jsonl")} for s in SPLITS} for m in MODES}
    source_sets = {s: {r["func"] for r in rows} for s, rows in data.items()}
    signatures = {m: {s: defaultdict(list) for s in SPLITS} for m in MODES}
    audit = dict(members={}, overlap={}, metadata_anomalies=[], annotation_range_errors=0,
                 token_mapping_errors=0, unique_core_errors=0, prediction_alignment_errors=0)
    functions = []
    line_metrics = {}
    started = time.monotonic()
    for split in SPLITS:
        rows = data[split]
        ann = annotations(split)
        assert len(rows) == len(ann)
        assert [r["target"] for r in rows] == [1, 0]*(len(rows)//2)
        raw = read_jsonl(Path("/home/PublicData/PHY-data/vul_detect/data/PrimeVul_original")/
                         f"primevul_{split}_paired.jsonl")
        assert len(raw) == len(rows)
        for a, b in zip(raw, rows):
            assert all(a[k] == b[k] for k in ("func", "target", "idx", "project", "commit_id"))
        audit["members"][split] = dict(functions=len(rows), pairs=len(rows)//2,
                                      duplicate_idx=len(rows)-len({r["idx"] for r in rows}))
        for i in range(0, len(rows), 2):
            a, b = rows[i:i+2]
            assert a["pair_key"] == b["pair_key"]
            if a["project"] != b["project"] or a["commit_id"] != b["commit_id"]:
                audit["metadata_anomalies"].append(a["pair_key"])
        line_acc = {m: dict(y=[], p=[]) for m in MODES}
        for index, (row, label) in enumerate(zip(rows, ann)):
            assert row["sample_key"] == label["sample_key"]
            parts = row["func"].splitlines(keepends=True)
            gt = set(label["lines_ground_truth"])
            assert all(0 <= n < len(parts) and parts[n].strip() for n in gt)
            tokenization = tokenizer(row["func"], add_special_tokens=False,
                                     return_offsets_mapping=True, return_attention_mask=False,
                                     verbose=False)
            encodings = {m: paired_encoding(row["func"], tokenizer, m, tokenization=tokenization)
                         for m in MODES}
            for mode in MODES:
                f = encodings[mode]
                s = scores[mode][split][row["sample_key"]]
                assert s["line_numbers"] == f["line_numbers"]
                assert len(s["line_scores"]) == len(s["line_numbers"])
                predicted = {n for n, p in zip(s["line_numbers"], s["line_scores"]) if p > .5}
                visible = set(f["line_numbers"])
                y = [int(n in gt) for n in s["line_numbers"]]
                line_acc[mode]["y"].extend(y)
                line_acc[mode]["p"].extend(s["line_scores"])
                tp, fp, fn = len(gt & predicted), len(predicted-gt), len((gt & visible)-predicted)
                pred_tokens = line_tokens(f, predicted)
                gt_tokens = line_tokens(f, gt)
                total = sum(f["code_mask"])
                mass = selected_mass(len(pred_tokens), total)
                z = total + (math.exp(4)-1)*len(pred_tokens)
                false_mass = len(pred_tokens-gt_tokens)*math.exp(4)/z
                true_mass = (len(pred_tokens & gt_tokens)*math.exp(4)+len(gt_tokens-pred_tokens))/z
                effective = z*z/(total-len(pred_tokens)+math.exp(8)*len(pred_tokens))/total
                signature = tuple(f["input_ids"])
                signatures[mode][split][signature].append((row["sample_key"], row["target"]))
                windows = f["windows"]
                assert sum(d-c for a, b, c, d in windows) == min(510, f["source_tokens"]) if mode == "512" else sum(d-c for a, b, c, d in windows) == f["source_tokens"]
                assert all(d-c <= 384 or len(windows) == 1 for a, b, c, d in windows)
                functions.append(dict(
                    mode=mode, split=split, sample_key=row["sample_key"], pair_key=row["pair_key"],
                    project=row["project"], target=row["target"], source_row=row["source_row"],
                    physical_lines=len(parts), nonempty_lines=f["nonempty_lines"],
                    source_tokens=f["source_tokens"], windows=len(windows),
                    length_group=length_group(f["source_tokens"]), window_group=window_group(len(windows)),
                    truncated=int(mode == "512" and f["source_tokens"] > 510),
                    visible_lines=len(visible), gt_total=len(gt), gt_visible=len(gt & visible),
                    gt_complete=sum(n in gt and full for n, full in zip(f["line_numbers"], f["line_complete"])),
                    predicted_lines=len(predicted), line_tp=tp, line_fp=fp, line_fn=fn,
                    line_precision=tp/(tp+fp) if tp+fp else None,
                    line_recall=tp/(tp+fn) if tp+fn else None,
                    line_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                    code_tokens=total, predicted_tokens=len(pred_tokens), gt_tokens=len(gt_tokens),
                    selected_token_fraction=len(pred_tokens)/total, selected_mass=mass,
                    false_mass=false_mass, true_mass=true_mass,
                    effective_token_fraction=effective,
                    new_code_tokens=sum(p >= 510 and valid for p, valid in enumerate(f["code_mask"][1:])),
                    new_predicted_tokens=sum(p >= 510 for p in pred_tokens),
                    new_true_tokens=sum(p >= 510 for p in gt_tokens),
                    unmapped_lines=f["unmapped_nonempty_lines"],
                ))
            if (index+1) % 500 == 0:
                print(f"statistics {split}: {index+1}/{len(rows)}, elapsed {time.monotonic()-started:.1f}s", flush=True)
        for mode in MODES:
            own = [r for r in functions if r["mode"] == mode and r["split"] == split]
            extra = binary_summary(line_acc[mode]["y"], line_acc[mode]["p"])
            line_metrics[f"{mode}/{split}"] = dict(line_summary(own),
                line_auc=extra["auc"], line_auprc=extra["auprc"],
                calibration=calibration(line_acc[mode]["y"], line_acc[mode]["p"]),
                by_label={str(label): line_summary([r for r in own if r["target"] == label])
                          for label in (0, 1)},
                by_length={g: line_summary([r for r in own if r["length_group"] == g])
                           for g in ("<=510", "511-2048", "2049-8192", ">8192")},
            )
        print("completed split", split, flush=True)
    for a, b in combinations(SPLITS, 2):
        overlap = dict(exact_sources=len(source_sets[a] & source_sets[b]))
        for mode in MODES:
            keys = signatures[mode][a].keys() & signatures[mode][b].keys()
            overlap[mode] = dict(unique_encoded_inputs=len(keys),
                first_members=sum(len(signatures[mode][a][k]) for k in keys),
                second_members=sum(len(signatures[mode][b][k]) for k in keys),
                opposite_label_inputs=sum(any(x[1] != y[1] for x in signatures[mode][a][k]
                                             for y in signatures[mode][b][k]) for k in keys))
        audit["overlap"][f"{a}:{b}"] = overlap
    for mode in MODES:
        for split in SPLITS:
            reverse = {key: sig for sig, members in signatures[mode][split].items() for key, _ in members}
            same = {data[split][i]["pair_key"] for i in range(0, len(data[split]), 2)
                    if reverse[data[split][i]["sample_key"]] == reverse[data[split][i+1]["sample_key"]]}
            for r in functions:
                if r["mode"] == mode and r["split"] == split:
                    r["encoded_pair_identical"] = int(r["pair_key"] in same)
    old = {(r["split"], r["sample_key"]): r["encoded_pair_identical"]
           for r in functions if r["mode"] == "512"}
    for r in functions:
        r["old512_pair_identical"] = old[r["split"], r["sample_key"]]
    write_csv(output/"functions.csv", functions)
    write_json(output/"line_metrics.json", line_metrics)
    write_json(output/"data_audit.json", audit)
    common_line_analysis(output, functions)
    metadata_analysis(output)
    region_analysis(output)
    checkpoint_audit(output, functions)
    print("STATISTICS_COMPLETE", len(functions), "function-mode rows", flush=True)


def policy_weights(features, batch, rows, split):
    """Change only post-encoder pooling; normal predicted_5 never uses Diff."""
    import torch

    weights = torch.ones((len(POLICIES), *batch["input_ids"].shape))
    for owner, (f, row) in enumerate(zip(features, rows)):
        visible = f["line_numbers"]
        predicted = set(row["lines_predicted"])
        gt = set(row["lines_ground_truth"]) & set(visible)
        seeds = {}
        for seed in (42, 43, 44):
            rng = random.Random(seed + 1000003*(SPLITS.index(split)+1) + row["source_row"])
            seeds[seed] = set(rng.sample(visible, len(predicted)))
        groups = dict(
            equal_1=(set(), 1), predicted_2=(predicted, 2),
            predicted_3=(predicted, 3), predicted_5=(predicted, 5),
            predicted_10=(predicted, 10), random_42=(seeds[42], 5),
            random_43=(seeds[43], 5), random_44=(seeds[44], 5),
            true_5=(gt, 5), predicted_tp_5=(predicted & gt, 5),
            predicted_fp_5=(predicted-gt, 5),
        )
        at = batch["window_owners"] == owner
        positions = batch["source_positions"][at]
        width = sum(d-c for a, b, c, d in f["windows"])
        for i, name in enumerate(POLICIES):
            lines, strength = groups[name]
            source = torch.ones(width)
            tokens = sorted(line_tokens(f, lines))
            if tokens:
                source[tokens] = float(strength)
            values = source[positions.clamp_min(0)]
            values[positions < 0] = 1
            weights[i, at] = values
    return weights


def intervened_forward(model, batch, weights, device):
    """Reuse the exact detector pool and global window pool for every policy."""
    import torch
    from .model import _window_mean

    ids = batch["input_ids"].to(device)
    attention = batch["attention_mask"].to(device)
    code = batch["code_mask"].to(device)
    owners = batch["window_owners"].to(device)
    weights = weights.to(device)
    pieces = [[] for _ in range(len(weights))]
    for start in range(0, len(ids), model.window_micro_batch):
        end = start + model.window_micro_batch
        states = model.encoder(ids[start:end], attention_mask=attention[start:end])[0]
        valid = code[start:end]
        occupied = valid.any(-1)
        for i in range(len(weights)):
            pooled = states.new_zeros((len(states), states.shape[-1]))
            if occupied.any():
                pooled[occupied] = model.pool(states[occupied], valid[occupied],
                                              weights[i, start:end][occupied])
            pieces[i].append(pooled)
    result = []
    count = len(batch["line_counts"])
    for i in range(len(weights)):
        windows = torch.cat(pieces[i])
        if len(windows) == count and torch.equal(owners, torch.arange(count, device=device)):
            pooled = windows
        else:
            mass = (weights[i].masked_fill(~code, float("-inf"))-5).exp().sum(-1)
            pooled = _window_mean(windows[:, None], (mass > 0)[:, None], owners,
                                  mass[:, None], count)
        result.append(model.classifier(pooled).squeeze(-1))
    return torch.stack(result).T


def gradient_check(model, tokenizer, rows, variant, mode, device):
    import torch
    from .training import paired_collate, paired_forward

    # Two small source-Train functions only. No optimizer is constructed.
    chosen, features = [], []
    for row in rows:
        f = paired_features(row, tokenizer, variant, input_mode=mode)
        if f["source_tokens"] <= 2048 and (variant != "localizer" or any(f["labels"])):
            chosen.append(row["sample_key"])
            features.append(f)
        if len(features) == 2:
            break
    if len(features) != 2:
        raise ValueError("two eligible source Train functions required for gradient check")
    model.zero_grad(set_to_none=True)
    loss, _ = paired_forward(model, paired_collate(features), variant, device)
    loss.backward()
    groups = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            group = "encoder" if name.startswith("encoder.") else "attention" if name.startswith("line_attention.") else "head"
            groups[group].append(float(parameter.grad.double().square().sum()))
    norms = {k: math.sqrt(sum(v)) for k, v in groups.items()}
    assert norms["encoder"] > 0 and norms["head"] > 0
    if variant == "localizer":
        assert norms["attention"] > 0
    model.zero_grad(set_to_none=True)
    return dict(members=chosen, loss=float(loss.detach()), gradient_norms=norms,
                training_mode=False, optimizer_steps=0)


def infer(output, mode, device):
    import torch
    from .evaluation import load_paired
    from .training import paired_collate, paired_forward

    torch.set_num_threads(4)
    output = Path(output)
    config = json.loads((output/"config.json").read_text())
    probe = set(config["train_inference_members"])
    data, predictions = {}, {}
    for split in SPLITS:
        ann = annotations(split)
        localized = read_jsonl(run_dir(mode, "losver_localizer")/f"{split}.predicted.jsonl")
        assert [r["sample_key"] for r in localized] == [r["sample_key"] for r in ann]
        data[split] = [dict(r, lines_ground_truth=a["lines_ground_truth"])
                       for r, a in zip(localized, ann) if split != "train" or r["sample_key"] in probe]
        predictions[split] = {}
        if split != "train":
            for method in METHODS:
                saved = read_jsonl(run_dir(mode, method)/f"{split}.predictions.jsonl")
                assert [r["sample_key"] for r in saved] == [r["sample_key"] for r in data[split]]
                predictions[split][method] = {r["sample_key"]: r["probability"] for r in saved}
    fields = ["mode", "method", "split", "sample_key", "pair_key", "target"]
    fields += [f"{kind}_{p}" for p in POLICIES for kind in ("logit", "probability")]
    summary = dict(mode=mode, device=device, inference_only=True, optimizer_steps=0,
                   methods={}, localizer={}, policies=POLICIES)
    with (output/f"interventions.{mode}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method in METHODS:
            started = time.monotonic()
            checkpoint = torch.load(run_dir(mode, method)/"best-p_c.pt", map_location="cpu",
                                    weights_only=False, mmap=True)
            model, tokenizer = load_paired(checkpoint, torch.device(device))
            origin = {"unixcoder": "equal_1", "losver": "predicted_5", "losver_oracle": "true_5"}[method]
            origin_index = POLICIES.index(origin)
            summary["methods"][method] = dict(
                selected_epoch=checkpoint["selected_epoch"], variant=checkpoint["variant"],
                parameters=sum(p.numel() for p in model.parameters()),
                classifier_weight_norm=float(model.classifier.weight.detach().norm()),
                classifier_bias=float(model.classifier.bias.detach()[0]),
                encoder_dropout=model.encoder.config.hidden_dropout_prob,
                splits={}, canonical_checks=[],
            )
            torch.cuda.reset_peak_memory_stats(device)
            for split in SPLITS:
                repro, original, max_error, disagree = [], [], 0., 0
                checked_long = False
                features = [paired_features(row, tokenizer, "predicted", input_mode=mode)
                            for row in data[split]]
                for start in range(0, len(features), 8):
                    fs, rows = features[start:start+8], data[split][start:start+8]
                    batch = paired_collate(fs)
                    weights = policy_weights(fs, batch, rows, split)
                    with torch.no_grad():
                        logits = intervened_forward(model, batch, weights, torch.device(device))
                        p = logits.sigmoid()
                        is_long = len(batch["input_ids"]) > len(rows)
                        if start == 0 or (is_long and not checked_long):
                            normal_batch = dict(batch, token_weights=weights[origin_index])
                            normal = paired_forward(model, normal_batch, checkpoint["variant"],
                                                    device, with_labels=False)[:, 0]
                            error = float((normal-p[:, origin_index]).abs().max())
                            assert error < 2e-5, ("canonical pool mismatch", mode, method, error)
                            assert torch.equal(normal > .5, p[:, origin_index] > .5)
                            summary["methods"][method]["canonical_checks"].append(
                                dict(split=split, start=start, functions=len(rows),
                                     windows=len(batch["input_ids"]), max_probability_error=error))
                            checked_long |= is_long
                    z_values, probabilities = logits.cpu().tolist(), p.cpu().tolist()
                    for row, z, values in zip(rows, z_values, probabilities):
                        record = {k: row[k] for k in ("sample_key", "pair_key", "target")}
                        record.update(mode=mode, method=method, split=split)
                        for j, policy in enumerate(POLICIES):
                            record[f"logit_{policy}"] = z[j]
                            record[f"probability_{policy}"] = values[j]
                        writer.writerow(record)
                        original.append(values[origin_index])
                        if split != "train":
                            saved = predictions[split][method][row["sample_key"]]
                            repro.append(saved)
                            max_error = max(max_error, abs(saved-values[origin_index]))
                            disagree += int((saved > .5) != (values[origin_index] > .5))
                    f.flush()
                    if start % 256 == 0:
                        print(f"inference {mode} {method} {split}: {start+len(rows)}/{len(features)}", flush=True)
                assert max_error < 2e-5 and disagree == 0, ("saved prediction mismatch", mode, method, split, max_error, disagree)
                summary["methods"][method]["splits"][split] = dict(
                    functions=len(original), max_saved_probability_error=max_error,
                    decision_disagreements=disagree, metrics=pair_summary(data[split], original))
                del features
            summary["methods"][method]["gradient_check"] = gradient_check(
                model, tokenizer, data["train"], checkpoint["variant"], mode, device)
            assert all(torch.equal(value.detach().cpu(), checkpoint["state_dict"][name])
                       for name, value in model.state_dict().items())
            summary["methods"][method].update(
                parameters_unchanged=True, seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
            del checkpoint, model
            torch.cuda.empty_cache()
            print("completed checkpoint", mode, method, flush=True)
    cp = torch.load(run_dir(mode, "losver_localizer")/"best-f1.pt", map_location="cpu",
                    weights_only=False, mmap=True)
    localizer, tokenizer = load_paired(cp, torch.device(device))
    summary["localizer"] = dict(selected_epoch=cp["selected_epoch"],
        gradient_check=gradient_check(localizer, tokenizer, data["train"], "localizer", mode, device),
        focal_positive_negative_cost_ratio_at_p_half=20,
        encoder_dropout=localizer.encoder.config.hidden_dropout_prob)
    assert all(torch.equal(value.detach().cpu(), cp["state_dict"][name])
               for name, value in localizer.state_dict().items())
    summary["localizer"]["parameters_unchanged"] = True
    summary["complete"] = True
    write_json(output/f"inference.{mode}.json", summary)
    print("INFERENCE_COMPLETE", mode, flush=True)


def oracle_count_control(output, device):
    """Separate Oracle position information from Diff-conditioned mask density."""
    import torch
    from .evaluation import load_paired
    from .training import paired_collate

    torch.set_num_threads(4)
    output = Path(output)
    policies = ("equal_1", "true_5", "random_lines_42", "random_lines_43", "random_lines_44",
                "random_tokens_42", "random_tokens_43", "random_tokens_44")
    config_path = output/"config.json"
    config = json.loads(config_path.read_text())
    config["oracle_count_control"] = dict(
        reason="source Train Diff labels have class-dependent empty-mask and density distributions",
        checkpoint="source-Valid-selected Oracle; no parameter updates",
        scope="all Valid and Test, both modes", policies=policies,
        rule="random physical lines match GT visible-line count; random code tokens match GT unique-token count exactly; all weights remain 5/1; seeds 42,43,44")
    write_json(config_path, config)
    fields = ["mode", "split", "sample_key", "pair_key", "target"]
    fields += [f"{kind}_{p}" for p in policies for kind in ("logit", "probability")]
    summary = dict(policies=policies, inference_only=True, optimizer_steps=0, modes={})
    with (output/"oracle_count_control.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for mode in MODES:
            started = time.monotonic()
            cp = torch.load(run_dir(mode, "losver_oracle")/"best-p_c.pt", map_location="cpu",
                            weights_only=False, mmap=True)
            model, tokenizer = load_paired(cp, torch.device(device))
            maximum, disagreements = 0., 0
            for split in ("valid", "test"):
                rows = annotations(split)
                features = [paired_features(r, tokenizer, "oracle", input_mode=mode) for r in rows]
                saved = read_jsonl(run_dir(mode, "losver_oracle")/f"{split}.predictions.jsonl")
                for start in range(0, len(rows), 8):
                    own, fs = rows[start:start+8], features[start:start+8]
                    batch = paired_collate(fs)
                    weights = torch.ones((len(policies), *batch["input_ids"].shape))
                    for owner, (row, feature) in enumerate(zip(own, fs)):
                        width = sum(d-c for a, b, c, d in feature["windows"])
                        gt_lines = set(row["lines_ground_truth"]) & set(feature["line_numbers"])
                        gt_tokens = line_tokens(feature, gt_lines)
                        code = [i for i in range(width) if feature["code_mask"][i+1]]
                        selected = [set(), gt_tokens]
                        for seed in (42, 43, 44):
                            rng = random.Random(seed+1000003*(SPLITS.index(split)+1)+row["source_row"])
                            selected.append(line_tokens(feature, set(rng.sample(feature["line_numbers"], len(gt_lines)))))
                        for seed in (42, 43, 44):
                            rng = random.Random(seed+1000003*(SPLITS.index(split)+1)+row["source_row"])
                            selected.append(set(rng.sample(code, len(gt_tokens))))
                        at = batch["window_owners"] == owner
                        positions = batch["source_positions"][at]
                        for i, tokens in enumerate(selected):
                            source = torch.ones(width)
                            if tokens:
                                source[sorted(tokens)] = 5.
                            local = source[positions.clamp_min(0)]
                            local[positions < 0] = 1.
                            weights[i, at] = local
                    with torch.no_grad():
                        logits = intervened_forward(model, batch, weights, torch.device(device))
                        probabilities = logits.sigmoid().cpu().tolist()
                    for i, (row, z, p) in enumerate(zip(own, logits.cpu().tolist(), probabilities)):
                        expected = saved[start+i]
                        assert expected["sample_key"] == row["sample_key"]
                        maximum = max(maximum, abs(expected["probability"]-p[1]))
                        disagreements += int((expected["probability"] > .5) != (p[1] > .5))
                        record = dict(mode=mode, split=split,
                                      **{k: row[k] for k in ("sample_key", "pair_key", "target")})
                        for j, policy in enumerate(policies):
                            record[f"logit_{policy}"], record[f"probability_{policy}"] = z[j], p[j]
                        writer.writerow(record)
                    f.flush()
                    if start % 256 == 0:
                        print("oracle count control", mode, split, start+len(own), len(rows), flush=True)
            assert maximum < 2e-5 and disagreements == 0
            assert all(torch.equal(value.detach().cpu(), cp["state_dict"][name])
                       for name, value in model.state_dict().items())
            summary["modes"][mode] = dict(max_probability_error=maximum,
                decision_disagreements=disagreements, seconds=time.monotonic()-started,
                parameters_unchanged=True)
            del cp, model
            torch.cuda.empty_cache()
    summary["complete"] = True
    write_json(output/"oracle_count_control.json", summary)
    print("ORACLE_COUNT_CONTROL_COMPLETE", flush=True)


def replay_localizer(output, device):
    import torch
    from .evaluation import load_paired
    from .model import predict_paired
    from .training import paired_collate, paired_forward

    torch.set_num_threads(4)
    result = {}
    for mode in MODES:
        cp = torch.load(run_dir(mode, "losver_localizer")/"best-f1.pt", map_location="cpu",
                        weights_only=False, mmap=True)
        model, tokenizer = load_paired(cp, torch.device(device))
        own = {}
        for split in SPLITS:
            rows = read_jsonl(Path("data/primevul_paired")/f"{split}.jsonl")
            pairs = sorted(random.Random(SEED).sample(range(len(rows)//2), 8))
            chosen = [rows[2*i+j] for i in pairs for j in (0, 1)]
            cached = {r["sample_key"]: r for r in read_jsonl(
                run_dir(mode, "losver_localizer")/f"{split}.line_scores.jsonl")}
            maximum, flips, count = 0., 0, 0
            for start in range(0, len(chosen), 8):
                part = chosen[start:start+8]
                batch = paired_collate([paired_features(r, tokenizer, "localizer", input_mode=mode) for r in part])
                with torch.no_grad():
                    probabilities = paired_forward(model, batch, "localizer", device, with_labels=False)
                for i, (row, p) in enumerate(zip(part, probabilities)):
                    cached_row = cached[row["sample_key"]]
                    assert batch["line_numbers"][i][batch["line_numbers"][i] >= 0].tolist() == cached_row["line_numbers"]
                    a, b = p.cpu().numpy(), np.asarray(cached_row["line_scores"])
                    maximum = max(maximum, float(abs(a-b).max()))
                    flips += int(((a > .5) != (b > .5)).sum())
                    count += len(a)
            assert maximum < 2e-5 and flips == 0, (mode, split, maximum, flips)
            own[split] = dict(members=[r["sample_key"] for r in chosen], lines=count,
                max_probability_error=maximum, line_decision_flips=flips, source_only=True)
        assert all(torch.equal(value.detach().cpu(), cp["state_dict"][name])
                   for name, value in model.state_dict().items())
        del model, cp
        torch.cuda.empty_cache()
        cp = torch.load(run_dir(mode, "losver")/"best-p_c.pt", map_location="cpu",
                        weights_only=False, mmap=True)
        valid = read_jsonl(Path("data/primevul_paired/valid.jsonl"))[:2]
        saved = read_jsonl(run_dir(mode, "losver")/"valid.predictions.jsonl")[:2]
        probabilities = predict_paired(cp, [dict(raw_source=r["func"]) for r in valid],
                                       device=device).sigmoid().cpu().tolist()
        maximum = max(abs(p-r["probability"]) for p, r in zip(probabilities, saved))
        assert maximum < 2e-5
        assert all((p > .5) == bool(r["prediction"]) for p, r in zip(probabilities, saved))
        own["single_function"] = dict(members=[r["sample_key"] for r in valid],
            max_probability_error=maximum, decision_flips=0, source_only=True)
        result[mode] = own
        del cp
        torch.cuda.empty_cache()
    write_json(Path(output)/"localizer_replay.json", result)
    print("LOCALIZER_AND_SINGLE_FUNCTION_REPLAY_COMPLETE", flush=True)


def bootstrap_delta(rows, first, second, repetitions=1000):
    """Resample complete pairs; this is uncertainty, not Test-based selection."""
    n = len(rows)//2
    a, b = np.asarray(first).reshape(n, 2), np.asarray(second).reshape(n, 2)
    labels = np.tile([1, 0], n)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(repetitions):
        at = rng.integers(n, size=n)
        x, y = a[at].ravel(), b[at].ravel()
        xc = (a[at, 0] > .5) & (a[at, 1] <= .5)
        yc = (b[at, 0] > .5) & (b[at, 1] <= .5)
        values.append((float(yc.mean()-xc.mean()),
                       float(roc_auc_score(labels, y)-roc_auc_score(labels, x))))
    delta = np.asarray(values)
    ac = (a[:, 0] > .5) & (a[:, 1] <= .5)
    bc = (b[:, 0] > .5) & (b[:, 1] <= .5)
    return dict(repetitions=repetitions, seed=SEED, pairs=n,
                pc_95pct=np.quantile(delta[:, 0], [.025, .975]).tolist(),
                auc_95pct=np.quantile(delta[:, 1], [.025, .975]).tolist(),
                pair_gained=int((~ac & bc).sum()), pair_lost=int((ac & ~bc).sum()),
                limitation="pair bootstrap does not adjust for project or shared-commit clusters")


def correlation(x, y):
    from scipy.stats import spearmanr

    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return dict(n=len(pairs), spearman=None)
    a, b = np.asarray(pairs).T
    if np.std(a) == 0 or np.std(b) == 0:
        return dict(n=len(pairs), spearman=None)
    return dict(n=len(pairs), spearman=float(spearmanr(a, b).statistic))


def common_line_analysis(output, functions):
    comparisons, feature_auc = {}, {}
    for split in SPLITS:
        ann = {r["sample_key"]: set(r["lines_ground_truth"]) for r in annotations(split)}
        old = {r["sample_key"]: r for r in read_jsonl(
            run_dir("512", "losver_localizer")/f"{split}.line_scores.jsonl")}
        new = read_jsonl(run_dir("l1", "losver_localizer")/f"{split}.line_scores.jsonl")
        groups = {key: ([], []) for key in ("512_prefix", "l1_prefix", "l1_new")}
        for row in new:
            key = row["sample_key"]
            prefix = set(old[key]["line_numbers"])
            for n, p in zip(old[key]["line_numbers"], old[key]["line_scores"]):
                groups["512_prefix"][0].append(int(n in ann[key]))
                groups["512_prefix"][1].append(p)
            for n, p in zip(row["line_numbers"], row["line_scores"]):
                target = groups["l1_prefix" if n in prefix else "l1_new"]
                target[0].append(int(n in ann[key]))
                target[1].append(p)
        comparisons[split] = {k: binary_summary(y, p) for k, (y, p) in groups.items()}
    for mode in MODES:
        for feature in ("predicted_lines", "selected_token_fraction", "selected_mass", "no_prediction",
                        "gt_visible", "no_visible_gt"):
            groups = {s: [r for r in functions if r["mode"] == mode and r["split"] == s] for s in SPLITS}
            def value(r):
                if feature == "no_prediction":
                    return int(r["predicted_lines"] == 0)
                if feature == "no_visible_gt":
                    return int(r["gt_visible"] == 0)
                return r[feature]
            raw_train_auc = float(roc_auc_score([r["target"] for r in groups["train"]],
                                               [value(r) for r in groups["train"]]))
            direction = 1 if raw_train_auc >= .5 else -1
            feature_auc[f"{mode}/{feature}"] = dict(
                train_raw_auc=raw_train_auc, direction_selected_on_train=direction,
                uses_diff=feature in ("gt_visible", "no_visible_gt"),
                splits={s: dict(n=len(rs), auc=float(roc_auc_score(
                    [r["target"] for r in rs], [direction*value(r) for r in rs])))
                        for s, rs in groups.items()},
                scope="single-feature association only; no classifier was fitted; direction fixed using Train")
    write_json(Path(output)/"common_line_metrics.json", comparisons)
    write_json(Path(output)/"mask_feature_auc.json", feature_auc)


def metadata_analysis(output):
    from .syntax import parser_for, walk, text

    def function_name(source):
        raw = source.encode()
        tree = parser_for("cpp").parse(raw)
        if tree.root_node.has_error:
            return None
        nodes = [n for n in walk(tree.root_node) if n.type == "function_definition"]
        if len(nodes) != 1:
            return None
        node = nodes[0].child_by_field_name("declarator")
        while node is not None:
            if node.type in ("identifier", "field_identifier", "operator_name", "destructor_name"):
                return text(node, raw)
            node = node.child_by_field_name("name" if node.type in
                ("qualified_identifier", "scoped_identifier") else "declarator")
        return None

    records, summary = [], {}
    for split in SPLITS:
        raw = read_jsonl(Path("/home/PublicData/PHY-data/vul_detect/data/PrimeVul_original")/
                         f"primevul_{split}_paired.jsonl")
        ann = annotations(split)
        counts = Counter()
        for i in range(0, len(raw), 2):
            a, b = raw[i:i+2]
            an, bn = function_name(a["func"]), function_name(b["func"])
            mismatches = {k: a.get(k) != b.get(k) for k in ("project", "commit_id", "cwe", "big_vul_idx")}
            known = an is not None and bn is not None
            name_diff = known and an != bn
            modified = len(ann[i]["lines_ground_truth"])+len(ann[i+1]["lines_ground_truth"])
            counts.update(pairs=1, names_known=int(known), name_mismatch=int(name_diff),
                          unknown_name=int(not known), modified_lines=modified)
            for key, flag in mismatches.items():
                counts[key+"_mismatch"] += flag
            if mismatches["big_vul_idx"]:
                counts["modified_lines_big_vul_idx_mismatch"] += modified
            if name_diff:
                counts["modified_lines_name_mismatch"] += modified
            records.append(dict(
                split=split, pair_key=ann[i]["pair_key"], project=a["project"], commit_id=a["commit_id"],
                vulnerable_name=an or "unknown", benign_name=bn or "unknown",
                name_mismatch=int(name_diff), name_known=int(known),
                big_vul_idx_v=a.get("big_vul_idx"), big_vul_idx_p=b.get("big_vul_idx"),
                cwe_v=a.get("cwe"), cwe_p=b.get("cwe"),
                **{k+"_mismatch": int(v) for k, v in mismatches.items()},
                modified_lines_v=len(ann[i]["lines_ground_truth"]),
                modified_lines_p=len(ann[i+1]["lines_ground_truth"])))
        summary[split] = dict(counts)
    write_csv(Path(output)/"pair_metadata.csv", records)
    write_json(Path(output)/"metadata_diagnostics.json", dict(
        splits=summary, definition="metadata/name mismatch flags are candidates, not automatic bad-pair labels; parser failures remain unknown"))


def region_analysis(output):
    from transformers import RobertaTokenizerFast

    model_path = json.loads((run_dir("l1", "losver")/"config.json").read_text())["model_path"]
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False)
    records = []
    for split in ("valid", "test"):
        scores = {r["sample_key"]: r for r in read_jsonl(
            run_dir("l1", "losver_localizer")/f"{split}.line_scores.jsonl")}
        for row in annotations(split):
            feature = paired_encoding(row["func"], tokenizer, "l1")
            own = scores[row["sample_key"]]
            predicted = line_tokens(feature, {n for n, p in zip(own["line_numbers"], own["line_scores"]) if p > .5})
            gt = line_tokens(feature, set(row["lines_ground_truth"]))
            code = {i-1 for i, valid in enumerate(feature["code_mask"]) if valid}
            new = {i for i in code if i >= 510}
            normalizer = len(code)+(math.exp(4)-1)*len(predicted)
            def mass(tokens):
                return (len(tokens-predicted)+math.exp(4)*len(tokens & predicted))/normalizer
            records.append(dict(
                split=split, sample_key=row["sample_key"], pair_key=row["pair_key"], target=row["target"],
                code_tokens=len(code), new_code_tokens=len(new),
                new_mass=mass(new), new_false_mass=mass((new & predicted)-gt),
                new_true_mass=mass(new & gt), prefix_false_mass=mass((predicted-new)-gt),
                selected_new_tokens=len(new & predicted),
                selected_new_true_tokens=len(new & predicted & gt)))
        print("new region audit complete", split, flush=True)
    write_csv(Path(output)/"new_region_mass.csv", records)


def checkpoint_audit(output, functions):
    import torch

    result, concentration = {}, {}
    for mode in MODES:
        configs = {m: json.loads((run_dir(mode, m)/"config.json").read_text()) for m in METHODS}
        first = configs["unixcoder"]
        for config in configs.values():
            assert config["training_config"] == first["training_config"]
            assert config["model_path"] == first["model_path"]
            assert config["classifier_head"] == first["classifier_head"]
            assert config["parameters"] == first["parameters"]
            if mode == "l1":
                assert config["window_micro_batch"] == first["window_micro_batch"]
                assert config["function_micro_batch"] == first["function_micro_batch"]
        result[mode] = dict(
            model_path=first["model_path"], parameters=first["parameters"],
            classifier=first["classifier_head"], training_config=first["training_config"],
            recorded_micro_batch={k: first[k] for k in ("window_micro_batch", "function_micro_batch") if k in first},
            environment=first["environment"],
            final_optimizer={})
        for method in (*METHODS, "losver_localizer"):
            path = run_dir(mode, method)/"last.pt"
            if path.exists():
                cp = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                result[mode]["final_optimizer"][method] = dict(
                    global_step=cp["loop"]["global_step"], next_epoch=cp["loop"]["epoch"],
                    lr=[g["lr"] for g in cp["optimizer"]["param_groups"]],
                    scheduler_last_epoch=cp["scheduler"]["last_epoch"],
                    optimizer_parameter_states=len(cp["optimizer"]["state"]))
                del cp
            else:
                result[mode]["final_optimizer"][method] = dict(available=False,
                    evidence="completed historical run did not retain last.pt; inspect epoch and LR logs")
        for split in SPLITS:
            own = [r for r in functions if r["mode"] == mode and r["split"] == split]
            ranked = sorted(own, key=lambda r: r["visible_lines"], reverse=True)
            cells = {}
            for n in dict.fromkeys((1, 10, math.ceil(len(own)*.01))):
                top, rest = ranked[:n], ranked[n:]
                cells[str(n)] = dict(
                    share_visible_lines=sum(r["visible_lines"] for r in top)/sum(r["visible_lines"] for r in own),
                    share_gt_lines=sum(r["gt_visible"] for r in top)/sum(r["gt_visible"] for r in own),
                    share_predicted_lines=sum(r["predicted_lines"] for r in top)/sum(r["predicted_lines"] for r in own),
                    remaining=line_summary(rest))
            largest = max(own, key=lambda r: r["predicted_lines"])
            concentration[f"{mode}/{split}"] = dict(top_by_visible_lines=cells, max_predicted_lines_function=largest)
    write_json(Path(output)/"fairness.json", result)
    write_json(Path(output)/"length_concentration.json", concentration)


def analyze(output):
    output = Path(output)
    functions = load_functions(output)
    features = {(r["mode"], r["split"], r["sample_key"]): r for r in functions}
    data = {s: read_jsonl(Path("data/primevul_paired")/f"{s}.jsonl") for s in SPLITS}
    for name in ("common_line_metrics.json", "metadata_diagnostics.json", "new_region_mass.csv", "fairness.json"):
        if not (output/name).is_file():
            raise FileNotFoundError(f"statistics stage must produce {name}")
    saved, summaries, histories = {}, {}, {}
    for mode in MODES:
        for method in METHODS:
            run = run_dir(mode, method)
            config = json.loads((run/"config.json").read_text())
            training = json.loads((run/"training.metrics.json").read_text())
            # Completed 512 configs predate explicit member lists. Verify their
            # saved predictions and actual training rows, rather than invent metadata.
            if "training_members" in config:
                assert config["training_members"] == [r["sample_key"] for r in data["train"]]
                assert config["validation_members"] == [r["sample_key"] for r in data["valid"]]
            epochs = [r for r in read_jsonl(run/"best.training.jsonl") if r["event"] == "epoch"]
            assert len(epochs) == 16 and training["completed_epochs"] == 16
            selected = max(epochs, key=lambda r: round(r["validation"]["p_c"], 4))
            assert training["selected_epochs"]["p_c"] == selected["epoch"]
            for split in ("valid", "test"):
                rows = read_jsonl(run/f"{split}.predictions.jsonl")
                assert [(r["sample_key"], r["pair_key"], r["target"]) for r in rows] == [
                    (r["sample_key"], r["pair_key"], r["target"]) for r in data[split]]
                p = [r["probability"] for r in rows]
                expected = json.loads((run/f"{split}.metrics.json").read_text())
                checked = paired_metrics(rows, p)
                assert all(expected["metrics"][k] == v for k, v in checked.items())
                assert all(r["prediction"] == int(r["probability"] > .5) for r in rows)
                assert expected["selected_epoch"] == selected["epoch"]
                saved[mode, method, split] = p
                summaries[f"{mode}/{method}/{split}"] = dict(
                    epoch=selected["epoch"], **pair_summary(rows, p),
                    logged_loss=expected["metrics"]["loss"])
            histories[f"{mode}/{method}"] = dict(
                epochs=epochs, selected_epoch=selected["epoch"],
                auc_history_available=all("auc" in r["validation"] for r in epochs),
                best_auc_epoch=(max(epochs, key=lambda r: r["validation"]["auc"])["epoch"]
                                if all("auc" in r["validation"] for r in epochs) else None),
                best_loss_epoch=min(epochs, key=lambda r: r["validation"]["loss"])["epoch"],
                near_best_pc_epochs=[r["epoch"] for r in epochs if
                    selected["validation"]["p_c"]-r["validation"]["p_c"] <= 1/562+1e-10],
                settings=config["training_config"],
                learning_rate_steps=[r for r in read_jsonl(run/"best.training.jsonl") if r["event"] == "step"],
            )
        run = run_dir(mode, "losver_localizer")
        epochs = [r for r in read_jsonl(run/"best.training.jsonl") if r["event"] == "epoch"]
        assert len(epochs) == 40
        selected = max(epochs, key=lambda r: round(r["validation"]["f1"], 4))
        assert selected["epoch"] == json.loads((run/"best-f1.json").read_text())["epoch"]
        histories[f"{mode}/losver_localizer"] = dict(epochs=epochs, selected_epoch=selected["epoch"])
    transitions, error_groups, correlations, strata = {}, {}, {}, []
    for mode in MODES:
        for split in ("valid", "test"):
            rows = data[split]
            a, b, oracle = [saved[mode, m, split] for m in METHODS]
            transitions[f"{mode}/{split}/baseline_to_losver"] = transition(a, b)
            transitions[f"{mode}/{split}/oracle_to_losver"] = transition(oracle, b)
            truth = np.asarray([r["target"] for r in rows])
            ca, cb = (np.asarray(a) > .5) == truth, (np.asarray(b) > .5) == truth
            groups = {}
            for label in (0, 1):
                at = truth == label
                groups[str(label)] = dict(
                    n=int(at.sum()), baseline_correct=int(ca[at].sum()),
                    losver_correct=int(cb[at].sum()), lost=int((ca & ~cb & at).sum()),
                    gained=int((~ca & cb & at).sum()), shared_wrong=int((~ca & ~cb & at).sum()))
            error_groups[f"{mode}/{split}"] = groups
            fs = [features[mode, split, r["sample_key"]] for r in rows]
            for label in (0, 1):
                for bucket in ("all", "<=510", "511-2048", "2049-8192", ">8192"):
                    at = [i for i, f in enumerate(fs) if f["target"] == label and
                          (bucket == "all" or f["length_group"] == bucket)]
                    gain = [(2*truth[i]-1)*(b[i]-a[i]) for i in at]
                    correlations[f"{mode}/{split}/label{label}/{bucket}"] = dict(
                        line_f1_vs_correct_probability_gain=correlation([fs[i]["line_f1"] for i in at], gain),
                        false_mass_vs_correct_probability_gain=correlation([fs[i]["false_mass"] for i in at], gain))
            pairs = [fs[i:i+2] for i in range(0, len(fs), 2)]
            definitions = dict(
                length=[length_group(max(r["source_tokens"] for r in p)) for p in pairs],
                windows=[window_group(max(int(r["windows"]) for r in p)) for p in pairs],
                old512_identical=[str(int(p[0]["old512_pair_identical"])) for p in pairs],
                diff_type=[("v+" if p[0]["gt_total"] > 0 else "v0")+"/"+
                           ("p+" if p[1]["gt_total"] > 0 else "p0") for p in pairs],
                modified_visible=["none" if sum(r["gt_visible"] for r in p) == 0 else
                                  "partial" if any(r["gt_complete"] < r["gt_total"] for r in p) else "complete" for p in pairs],
                predicted_hit=["both" if all(r["line_tp"] > 0 for r in p) else
                               "one" if any(r["line_tp"] > 0 for r in p) else
                               "missed" if any(r["gt_visible"] > 0 for r in p) else "no_visible_gt" for p in pairs],
                no_prediction=[str(sum(r["predicted_lines"] == 0 for r in p)) for p in pairs],
                false_mass=[">.5" if mean([r["false_mass"] for r in p]) > .5 else "<=.5" for p in pairs],
                new_true_region=["yes" if any(r["new_true_tokens"] > 0 for r in p) else "no" for p in pairs],
            )
            for group, values in definitions.items():
                for value in dict.fromkeys(values):
                    at = [i for i, v in enumerate(values) if v == value]
                    indices = [2*i+j for i in at for j in (0, 1)]
                    selected_rows = [rows[i] for i in indices]
                    for method in METHODS:
                        result = pair_summary(selected_rows, [saved[mode, method, split][i] for i in indices])
                        strata.append(dict(mode=mode, split=split, group=group, value=value,
                            method=method, pairs=result["pairs"], p_c=result["p_c"], p_v=result["p_v"],
                            p_b=result["p_b"], p_r=result["p_r"], accuracy=result["accuracy"],
                            f1=result["f1"], mcc=result["mcc"], auc=result["auc"],
                            ranking_accuracy=result["ranking_accuracy"]))
    uncertainty = {f"{mode}/baseline_to_losver": bootstrap_delta(data["test"],
                    saved[mode, "unixcoder", "test"], saved[mode, "losver", "test"])
                   for mode in MODES}
    for method in METHODS:
        uncertainty[f"512_to_l1/{method}"] = bootstrap_delta(data["test"],
            saved["512", method, "test"], saved["l1", method, "test"])
    intervention_rows = []
    intervention_metrics = []
    for mode in MODES:
        completion = json.loads((output/f"inference.{mode}.json").read_text())
        assert completion["complete"] and completion["optimizer_steps"] == 0
        with (output/f"interventions.{mode}.csv").open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["target"] = int(row["target"])
            for key in row:
                if key.startswith(("logit_", "probability_")):
                    row[key] = float(row[key])
        intervention_rows.extend(rows)
        for method in METHODS:
            for split in SPLITS:
                own = [r for r in rows if r["method"] == method and r["split"] == split]
                expected = data[split]
                if split == "train":
                    members = set(json.loads((output/"config.json").read_text())["train_inference_members"])
                    expected = [r for r in expected if r["sample_key"] in members]
                assert [r["sample_key"] for r in own] == [r["sample_key"] for r in expected]
                for policy in POLICIES:
                    p = [r[f"probability_{policy}"] for r in own]
                    z = np.asarray([r[f"logit_{policy}"] for r in own])
                    y = np.asarray([r["target"] for r in own])
                    result = pair_summary(own, p)
                    raw_rank = z[::2] > z[1::2]
                    pc = (np.asarray(p)[::2] > .5) & (np.asarray(p)[1::2] <= .5)
                    intervention_metrics.append(dict(mode=mode, method=method, split=split, policy=policy,
                        pairs=result["pairs"], p_c=result["p_c"], accuracy=result["accuracy"],
                        f1=result["f1"], mcc=result["mcc"], auc=result["auc"], auprc=result["auprc"],
                        recall=result["recall"], precision=result["precision"],
                        ranking_accuracy=result["ranking_accuracy"],
                        bce=float((np.logaddexp(0, z)-y*z).mean()),
                        brier=result["calibration"]["brier"], ece=result["calibration"]["ece"],
                        extreme_fraction=result["calibration"]["extreme_fraction"],
                        logit_abs_median=float(np.median(abs(z))),
                        logit_gap_mean=float(np.mean(z[::2]-z[1::2])),
                        logit_ranking_accuracy=float(raw_rank.mean()),
                        logit_ranking_ties=int((z[::2] == z[1::2]).sum()),
                        logit_auc=float(roc_auc_score(y, z)),
                        ranked_correct_but_not_pc=int((raw_rank & ~pc).sum()),
                        pair_margin_median=float(np.median(np.minimum(z[::2], -z[1::2])))))
                if method == "losver" and split == "test":
                    for policy in ("equal_1", "true_5"):
                        uncertainty[f"{mode}/frozen_predicted5_to_{policy}"] = bootstrap_delta(
                            own, [r["probability_predicted_5"] for r in own],
                            [r[f"probability_{policy}"] for r in own])
    write_json(output/"detector_metrics.json", summaries)
    write_json(output/"training_analysis.json", histories)
    write_json(output/"pair_transitions.json", transitions)
    write_json(output/"error_changes.json", error_groups)
    write_json(output/"quality_correlations.json", correlations)
    write_json(output/"uncertainty.json", uncertainty)
    write_csv(output/"strata.csv", strata)
    write_csv(output/"intervention_metrics.csv", intervention_metrics)
    oracle_completion = json.loads((output/"oracle_count_control.json").read_text())
    assert oracle_completion["complete"] and oracle_completion["optimizer_steps"] == 0
    with (output/"oracle_count_control.csv").open() as f:
        oracle_rows = list(csv.DictReader(f))
    control_metrics = []
    for mode in MODES:
        for split in ("valid", "test"):
            own = [r for r in oracle_rows if r["mode"] == mode and r["split"] == split]
            assert [r["sample_key"] for r in own] == [r["sample_key"] for r in data[split]]
            for r in own:
                r["target"] = int(r["target"])
            for policy in oracle_completion["policies"]:
                m = pair_summary(own, [float(r[f"probability_{policy}"]) for r in own])
                control_metrics.append(dict(mode=mode, split=split, policy=policy,
                    **{k: m[k] for k in ("pairs", "p_c", "accuracy", "f1", "mcc", "auc", "auprc")}))
    write_csv(output/"oracle_count_control_metrics.csv", control_metrics)
    examples(output, data, features, saved, intervention_rows)
    figures(output, summaries, histories, transitions, strata, intervention_metrics)
    print("ANALYSIS_COMPLETE", len(strata), "strata", len(intervention_metrics), "interventions", flush=True)


def examples(output, data, features, saved, interventions):
    """Outcome-selected illustrations; not an unbiased estimate of case frequency."""
    rows = data["test"]
    annotations_by_key = {r["sample_key"]: r for r in annotations("test")}
    line_scores = {m: {r["sample_key"]: r for r in read_jsonl(
        run_dir(m, "losver_localizer")/"test.line_scores.jsonl")} for m in MODES}
    state = {m: {method: pair_states(saved[m, method, "test"]) for method in METHODS} for m in MODES}
    fixed = {r["sample_key"]: r for r in interventions if
             r["mode"] == "l1" and r["method"] == "losver" and r["split"] == "test"}
    pairs = len(rows)//2
    predicates = {
        "baseline_correct_losver_wrong": lambda i: state["l1"]["unixcoder"][i] == "C" and state["l1"]["losver"][i] != "C",
        "baseline_wrong_losver_correct": lambda i: state["l1"]["unixcoder"][i] != "C" and state["l1"]["losver"][i] == "C",
        "oracle_correct_losver_wrong": lambda i: state["l1"]["losver_oracle"][i] == "C" and state["l1"]["losver"][i] != "C",
        "newly_distinguishable_oracle_gain": lambda i: features["l1", "test", rows[2*i]["sample_key"]]["old512_pair_identical"] == 1 and state["512"]["losver_oracle"][i] != "C" and state["l1"]["losver_oracle"][i] == "C" and state["l1"]["losver"][i] != "C",
        "no_prediction_oracle_correct": lambda i: all(features["l1", "test", r["sample_key"]]["predicted_lines"] == 0 for r in rows[2*i:2*i+2]) and state["l1"]["losver_oracle"][i] == "C" and state["l1"]["losver"][i] != "C",
        "false_token_mass_dominates": lambda i: max(features["l1", "test", r["sample_key"]]["false_mass"] for r in rows[2*i:2*i+2]) > .8 and state["l1"]["losver_oracle"][i] == "C" and state["l1"]["losver"][i] != "C",
        "frozen_true_lines_hurt_correct_pair": lambda i: state["l1"]["losver"][i] == "C" and not (fixed[rows[2*i]["sample_key"]]["probability_true_5"] > .5 and fixed[rows[2*i+1]["sample_key"]]["probability_true_5"] <= .5),
    }
    used, selected = set(), []
    for category, predicate in predicates.items():
        candidates = [i for i in range(pairs) if predicate(i)]
        available = [i for i in candidates if i not in used]
        if not available:
            selected.append(dict(category=category, matching_pairs=len(candidates), example=None))
            continue
        index = min(available, key=lambda i: max(features["l1", "test", r["sample_key"]]["source_tokens"] for r in rows[2*i:2*i+2]))
        used.add(index)
        case = dict(category=category, matching_pairs=len(candidates), pair_key=rows[2*index]["pair_key"],
                    selection="shortest unused qualifying pair; outcome-selected, illustrative only", functions=[])
        for j in (0, 1):
            row = rows[2*index+j]
            key = row["sample_key"]
            gt = set(annotations_by_key[key]["lines_ground_truth"])
            scores = line_scores["l1"][key]
            score_map = dict(zip(scores["line_numbers"], scores["line_scores"]))
            predicted = {n for n, p in score_map.items() if p > .5}
            source = row["func"].splitlines()
            context = sorted({n+d for n in gt | predicted for d in (-1, 0, 1) if 0 <= n+d < len(source)})
            context = context[:28]
            excerpt = [dict(line=n+1, diff_modified=n in gt, predicted=n in predicted,
                            score=score_map.get(n), source=source[n]) for n in context]
            case["functions"].append(dict(
                sample_key=key, target=row["target"], project=row["project"], commit_id=row["commit_id"],
                source_header=source[0] if source else "", total_physical_lines=len(source),
                diff_modified_lines=[n+1 for n in sorted(gt)], predicted_lines=[n+1 for n in sorted(predicted)],
                features=features["l1", "test", key],
                probabilities={f"{m}/{method}": saved[m, method, "test"][2*index+j]
                               for m in MODES for method in METHODS},
                frozen_losver_probabilities={p: fixed[key][f"probability_{p}"] for p in POLICIES},
                excerpt=excerpt, excerpt_limit=28, line_numbers="one-based for human review"))
        selected.append(case)
    write_json(Path(output)/"cases.json", selected)


def figures(output, summaries, histories, transitions, strata, interventions):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vul-detect-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    output = Path(output)
    dest = output/"figures"
    dest.mkdir(exist_ok=True)
    colors = ("#0072B2", "#D55E00", "#009E73")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(dest/f"{name}.png", dpi=180)
        fig.savefig(dest/f"{name}.pdf")
        plt.close(fig)

    lines = json.loads((output/"line_metrics.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for column, mode in enumerate(MODES):
        ax = axes[0, column]
        for i, key in enumerate(("precision", "recall", "f1")):
            ax.bar(np.arange(3)+(i-1)*.24, [lines[f"{mode}/{s}"][key] for s in SPLITS],
                   width=.23, color=colors[i], label=key)
        ax.set(xticks=range(3), xticklabels=SPLITS, ylim=(0, 1), title=f"{mode}: visible-line metrics")
        ax.legend()
        ax = axes[1, column]
        for i, key in enumerate(("selected_attention_mass", "false_attention_mass")):
            ax.bar(np.arange(3)+(i-.5)*.32, [lines[f"{mode}/{s}"][key]["median"] for s in SPLITS],
                   width=.31, color=colors[i], label=key.replace("_", " "))
        ax.set(xticks=range(3), xticklabels=SPLITS, ylim=(0, 1), title="Median per-function aggregation mass")
        ax.legend(fontsize=8)
    save(fig, "localization_shift")

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for i, mode in enumerate(MODES):
        for j, method in enumerate((*METHODS, "losver_localizer")):
            h = histories[f"{mode}/{method}"]
            epochs = h["epochs"]
            ax = axes[i, j]
            ax.plot([r["epoch"] for r in epochs], [r["train_loss"] for r in epochs], label="Train")
            ax.plot([r["epoch"] for r in epochs], [r["validation"]["loss"] for r in epochs], label="Valid")
            ax.axvline(h["selected_epoch"], ls="--", color="black", alpha=.5, label="Selected")
            ax.set(title=f"{mode}: {method}", xlabel="Epoch", ylabel="Focal loss" if method == "losver_localizer" else "BCE")
            ax.legend(fontsize=7)
    save(fig, "learning_curves")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, mode in zip(axes, MODES):
        record = transitions[f"{mode}/test/baseline_to_losver"]
        matrix = np.asarray(record["counts"])
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max()/2 else "black")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=record["order"],
               yticklabels=record["order"], xlabel="LOSVER pair state", ylabel="Baseline pair state",
               title=f"{mode}: Test {record['denominator']} pairs")
    save(fig, "pair_transitions")

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for column, mode in enumerate(MODES):
        records = [next(r for r in interventions if r["mode"] == mode and r["method"] == "losver"
                        and r["split"] == "test" and r["policy"] == p) for p in POLICIES]
        for row, key in enumerate(("p_c", "auc")):
            ax = axes[row, column]
            ax.bar(range(len(POLICIES)), [r[key] for r in records], color=[
                colors[0] if p == "predicted_5" else colors[2] if p == "true_5" else "#999999" for p in POLICIES])
            ax.axhline(records[POLICIES.index("predicted_5")][key], color=colors[0], ls="--")
            ax.set(xticks=range(len(POLICIES)), xticklabels=POLICIES, ylabel=key,
                   title=f"{mode}: frozen LOSVER checkpoint, Test")
            ax.tick_params(axis="x", rotation=65)
    save(fig, "inference_interventions")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = ("<=510", "511-2048", "2049-8192", ">8192")
    for ax, mode in zip(axes, MODES):
        for j, method in enumerate(METHODS):
            rs = [next(r for r in strata if r["mode"] == mode and r["split"] == "test" and
                       r["group"] == "length" and r["value"] == b and r["method"] == method) for b in bins]
            ax.bar(np.arange(4)+(j-1)*.24, [r["p_c"] for r in rs], width=.23,
                   color=colors[j], label=method)
        ax.set(xticks=range(4), xticklabels=[f"{b}\nn={r['pairs']} pairs" for b, r in zip(bins, rs)],
               ylabel="Test P-C", title=f"{mode}: pair maximum source tokens")
        ax.legend(fontsize=7)
    save(fig, "length_strata")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("statistics", "inference", "oracle-control", "analysis", "replay"))
    parser.add_argument("--output-dir", default="results/losver_diagnosis")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "statistics":
        collect(args.output_dir)
    elif args.stage == "analysis":
        analyze(args.output_dir)
    elif args.stage == "oracle-control":
        oracle_count_control(args.output_dir, args.device)
    elif args.stage == "replay":
        replay_localizer(args.output_dir, args.device)
    elif args.mode is None:
        parser.error("inference requires --mode")
    else:
        infer(args.output_dir, args.mode, args.device)


if __name__ == "__main__":
    main()
