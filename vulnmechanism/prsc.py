"""Pair-guided masks and matched Random / original Top-Gradient mask controls."""

from __future__ import annotations
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import tempfile
import time

import torch
from torch.nn import functional as F

from .augmentation import surface_seed
from .model import auxiliary_mode


MASK_METHODS = ("none", "random", "rsc", "prsc")
PILOT = (("hard_aug_bce", "none", 0.05),) + tuple(
    (f"{method}_{round(fraction * 100)}", method, fraction)
    for method in ("random", "rsc", "prsc") for fraction in (0.05, 0.10)
)
RSC_REFERENCE = "https://arxiv.org/abs/2007.02454"
RSC_CODE = "https://github.com/DeLightCMU/RSC/blob/master/Domain_Generalization/models/resnet.py"


@torch.no_grad()
def feature_mask(method, head_weight, original, changed, labels, fraction, *, seeds=None):
    """Return detached keep masks; no inverted-dropout scaling or bias masking.

    RSC uses signed true-class logit gradients, NOT absolute gradients or
    gradient-times-activation (paper section 3.1, equations 1-3). The equivalent
    two-class logits [-z/2, z/2] give gradient (2*y-1)*w/2 for binary BCE.
    With a linear head this ranking is necessarily identical within each class.
    This is the original mask rule in the common additive two-view BCE protocol,
    not a reproduction of the paper's CNN spatial/batch curriculum extensions.
    """
    if method not in MASK_METHODS[1:] or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("invalid representation mask method or fraction")
    if (original.ndim != 2 or original.shape != changed.shape
            or head_weight.numel() != original.shape[1] or labels.shape != original.shape[:1]):
        raise ValueError("incompatible paired representations and binary linear head")
    if not all(torch.isfinite(v).all() for v in (head_weight, original, changed)):
        raise ValueError("nonfinite mask inputs")
    n, dimensions = original.shape
    count = math.ceil(dimensions * fraction)
    weight = head_weight.detach().flatten().float()
    if method == "prsc":
        scores = ((changed.float() - original.float()) * weight).abs()
    elif method == "rsc":
        scores = (2 * labels.float() - 1)[:, None] * weight[None, :] / 2
    else:
        if seeds is None or len(seeds) != n:
            raise ValueError("random masks require one independent seed per function")
        # Private CPU RNGs leave the main training dropout stream unchanged.
        scores = torch.stack([
            torch.rand(dimensions, generator=torch.Generator().manual_seed(seed))
            for seed in seeds
        ]).to(original.device)
    # Stable sort fixes ties by feature index; exactly ceil(k*d) dimensions drop.
    indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :count]
    return torch.ones_like(original, dtype=torch.float32).scatter_(1, indices, 0).detach()


def pair_eligibility(row, selected, builder):
    """Strict selected views must also change the actual truncated encoder input."""
    if selected is None:
        return "no_legal_view"
    if not selected["strict_semantic"]:
        return "input_only_stress"
    budget = builder.source_max_length - len(builder.source_prefix) - 2
    if builder._encode(row["raw_source"], budget) == builder._encode(selected["raw_source"], budget):
        return "identical_encoder_input"
    return "eligible"


@torch.no_grad()
def training_masks(model, records, selections, builder, *, device, method, fraction, seed, epoch,
                   references=None):
    """Reuse detached dropout-free representations from this batch's Hardest scoring."""
    reasons = Counter()
    paired = []
    for row, selection in zip(records, selections):
        reason = pair_eligibility(row, selection["selected"], builder)
        reasons[reason] += 1
        if reason == "eligible":
            paired.append((row, selection["selected"]))
    costs = Counter()
    result = {}
    head = model.task_modules["classifier"]
    for row, selected in paired:
        if method == "prsc":
            if references is None:
                raise ValueError("PRSC requires the current dropout-free Hardest representations")
            original = references[(row["sample_key"], row["raw_source"])][None].to(device=device, dtype=torch.float32)
            changed = references[(row["sample_key"], selected["raw_source"])][None].to(device=device, dtype=torch.float32)
            costs.update(reused_search_representations=2)
        else:
            # Random and linear RSC rankings need no encoder queries.
            original = head.weight.new_zeros((1, head.in_features))
            changed = original
        labels = original.new_tensor([row["label"]])
        result[row["sample_key"]] = feature_mask(
            method, head.weight, original, changed, labels, fraction,
            seeds=[surface_seed(seed, epoch, row["sample_key"], "representation-mask")],
        )[0]
    return result, dict(reasons), dict(costs)


def masked_bce(head, representations, labels, masks, weights):
    """Extra weighted BCE; mask detached, encoder and head gradients retained."""
    logits = head(representations * masks.detach()).squeeze(-1)
    terms = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (terms * weights).sum()


def _csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def probe(data_dir, output_dir, checkpoint_path, *, sample_limit=256, device="auto"):
    """Source-Train evidence, including an independently generated second view."""
    from . import model as core
    from .artifacts import write_json
    from .augmentation import search_surface_views
    from .data import balanced_subset, class_counts, read_jsonl, write_jsonl
    from .metrics import decision_boundary, metrics
    from .syntax import surface_materials
    from .training import TRAINING

    if sample_limit not in (128, 256):
        raise ValueError("use a fixed balanced 128/256-function source Train probe")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "protocol.json", dict(
        dataset="primevul", seed=42, train=1024, valid=256, test=256,
        training=TRAINING, pilot=[dict(name=n, mask=m, fraction=f, weight=.2) for n, m, f in PILOT],
        search="unchanged Hardest: width 2, depth 3, 12 scored candidates in Train; 48 in Adaptive Test",
        selection="own Clean Valid only; original threshold/epoch rules; Test once after all training",
        masks="ceil(fraction * hidden size); detached; same mask for both views; no rescaling; unchanged bias",
        eligibility="strict semantic selected view and actual truncated encoder-input change",
        reference="reuse detached dropout-free representations from current Hardest scoring; main and masked BCE reuse training-dropout representations",
        rsc="paper section 3.1 signed true-class Top-Gradient mask, in the same additive pair-BCE protocol; no spatial/batch curriculum",
        rsc_reference=RSC_REFERENCE, rsc_author_code=RSC_CODE,
        scope="Clean function classification; matched small pilot only; no cross-domain or full training",
    ))
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset = cp["dataset"]
    if dataset != "primevul":
        raise ValueError("this fixed pilot uses PrimeVul")
    train = balanced_subset(read_jsonl(Path(data_dir) / dataset / "train.jsonl"), 1024)
    rows = balanced_subset(train, sample_limit)
    materials = surface_materials(train)
    device = core.resolve_device(device)
    core.seed_everything(42)
    builder = core.InputBuilder(core.AutoTokenizer.from_pretrained(cp["model_path"]))
    model = core.load_model(cp, device)
    if device.type == "cuda":
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    budget = builder.source_max_length - len(builder.source_prefix) - 2
    features, search_costs, second_costs = {}, Counter(), Counter()

    def score(batch, costs=search_costs):
        logits = []
        with torch.no_grad(), auxiliary_mode(model):
            for at in range(0, len(batch), 4):
                part = batch[at:at + 4]
                inputs = builder.codebert_batch(part, device=device)
                z, h = model(*inputs, return_representation=True)
                for row, value, representation in zip(part, z.cpu().tolist(), h.cpu()):
                    features[(row["sample_key"], row["raw_source"])] = (value, representation.double())
                    logits.append(value)
                costs.update(function_forwards=len(part), encoder_sequences=len(inputs[0]),
                             encoder_token_slots=inputs[0].numel(), encoder_active_tokens=int(inputs[1].sum()))
        return logits

    started = time.monotonic()
    pairs, traces, coverage, selection_reasons = [], [], Counter(), Counter()
    for at in range(0, len(rows), 4):
        part = rows[at:at + 4]
        selections = search_surface_views(part, score, builder.tokenizer, budget, materials, epoch=1)
        for row, entry in zip(part, selections):
            selected = entry["selected"]
            reason = pair_eligibility(row, selected, builder)
            coverage[reason] += 1
            selection_reasons[entry["selection_reason"]] += 1
            second = None
            if reason == "eligible":
                # Independent RNG and generation from x, with no scoring/selection
                # by the model. Only source/input identity with x* is excluded.
                generated = search_surface_views(
                    [row], None, builder.tokenizer, budget, materials, epoch=1,
                    purpose="prsc-independent-view", random_selection=True,
                    kinds=["local_name", "comment_insert", "comment_delete", "comment_replace"],
                )[0]
                selected_ids = builder._encode(selected["raw_source"], budget)
                legal = [v for v in generated["visited"] if v["strict_semantic"]
                         and builder._encode(v["raw_source"], budget) != selected_ids]
                if legal:
                    rng = random.Random(surface_seed(42, 1, row["sample_key"], "prsc-independent-choice"))
                    second = rng.choice(legal)
                    score([dict(row, raw_source=second["raw_source"])], second_costs)
                    pairs.append((row, selected, second))
                    coverage["independent_second_view"] += 1
                else:
                    coverage["no_independent_second_view"] += 1
            traces.append(dict(
                sample_key=row["sample_key"], label=row["label"], eligibility=reason,
                selection_reason=entry["selection_reason"], candidate_queries=entry["candidate_queries"],
                original_queries=entry["original_queries"],
                hard_units=None if selected is None else selected["units"],
                independent_units=None if second is None else second["units"],
            ))
        print(f"PRSC source Train diagnostic {min(at + 4, len(rows))}/{len(rows)}", flush=True)
    if not pairs or {r["label"] for r, _, _ in pairs} != {0, 1}:
        raise ValueError("insufficient independent strict pairs for the diagnostic")
    if sum(t["candidate_queries"] + t["original_queries"] for t in traces) != search_costs["function_forwards"]:
        raise AssertionError("diagnostic query counters disagree")
    head = model.task_modules["classifier"]
    weight, bias = head.weight.detach().cpu().double(), head.bias.detach().cpu().double()
    labels = torch.tensor([r["label"] for r, _, _ in pairs])
    h = torch.stack([features[(r["sample_key"], r["raw_source"])][1] for r, _, _ in pairs])
    hard = torch.stack([features[(r["sample_key"], c["raw_source"])][1] for r, c, _ in pairs])
    second = torch.stack([features[(r["sample_key"], c["raw_source"])][1] for r, _, c in pairs])
    z = [F.linear(v, weight, bias).squeeze(-1) for v in (h, hard, second)]
    native = torch.tensor([
        [features[(row["sample_key"], view["raw_source"])][0] for view in (row, first, other)]
        for row, first, other in pairs
    ], dtype=torch.float64).T
    contributions = [(v - h) * weight for v in (hard, second)]
    identity_errors = [(native[i + 1] - native[0] - contributions[i].sum(1)).abs() for i in range(2)]
    error = max(float(e.max()) for e in identity_errors)
    if error > 1e-5:
        raise AssertionError(f"linear logit decomposition failed: {error}")
    threshold = cp["decision_threshold"]
    boundary = decision_boundary(threshold)

    def classification(logits):
        result = metrics(labels.tolist(), logits.sigmoid().tolist(), threshold)
        result["bce"] = float(F.binary_cross_entropy_with_logits(logits, labels.double()))
        return result

    baseline = {name: classification(value) for name, value in zip(("original", "hard", "independent"), z)}
    details, summaries = [], []
    for method in ("random", "rsc", "prsc"):
        for fraction in (.05, .10):
            mask = feature_mask(
                method, weight, h, hard, labels, fraction,
                seeds=[surface_seed(42, 1, r["sample_key"], "representation-mask") for r, _, _ in pairs],
            ).double()
            removed = 1 - mask
            after = [F.linear(v * mask, weight, bias).squeeze(-1) for v in (h, hard, second)]
            capture = [(c.abs() * removed).sum(1) / c.abs().sum(1).clamp_min(1e-30) for c in contributions]
            delta, predicted = z[2] - z[0], (contributions[1] * removed).sum(1)
            residual = after[2] - after[0]
            varied = float(delta.var(unbiased=False)) > 0 and float(predicted.var(unbiased=False)) > 0
            outcomes = {}
            for name, before, masked in zip(("original", "hard", "independent"), z, after):
                correct = before.sigmoid().ge(boundary).eq(labels.bool())
                still = masked.sigmoid().ge(boundary).eq(labels.bool())
                outcomes[name] = dict(
                    metrics=classification(masked), correct_before=int(correct.sum()),
                    correct_to_wrong=int((correct & ~still).sum()), wrong_to_correct=int((~correct & still).sum()),
                )
            summary = dict(
                method=method, fraction=fraction, samples=len(pairs), dropped_dimensions=int(removed[0].sum()),
                random_expected_capture=float(removed[0].mean()),
                hard_l1_capture_mean=float(capture[0].mean()), hard_l1_capture_median=float(capture[0].median()),
                independent_l1_capture_mean=float(capture[1].mean()), independent_l1_capture_median=float(capture[1].median()),
                independent_capture_above_random=int((capture[1] > removed[0].mean()).sum()),
                hard_abs_delta_mean=float((z[1] - z[0]).abs().mean()),
                hard_masked_abs_delta_mean=float((after[1] - after[0]).abs().mean()),
                independent_abs_delta_mean=float(delta.abs().mean()),
                independent_masked_abs_delta_mean=float(residual.abs().mean()),
                independent_abs_delta_increased=int((residual.abs() > delta.abs() + 1e-8).sum()),
                independent_explained_r2=1 - float(residual.square().sum() / (delta - delta.mean()).square().sum())
                if float(delta.var(unbiased=False)) > 0 else None,
                independent_predicted_delta_correlation=float(torch.corrcoef(torch.stack([delta, predicted]))[0, 1])
                if varied else None,
                classification=outcomes,
            )
            summaries.append(summary)
            for i, (row, _, _) in enumerate(pairs):
                details.append(dict(
                    sample_key=row["sample_key"], label=row["label"], method=method, fraction=fraction,
                    removed_dimensions=json.dumps(removed[i].nonzero().flatten().tolist()),
                    hard_l1_capture=float(capture[0][i]), independent_l1_capture=float(capture[1][i]),
                    original_logit=float(z[0][i]), hard_logit=float(z[1][i]), independent_logit=float(z[2][i]),
                    masked_original_logit=float(after[0][i]), masked_hard_logit=float(after[1][i]),
                    masked_independent_logit=float(after[2][i]),
                    independent_abs_delta_increased=bool(residual[i].abs() > delta[i].abs() + 1e-8),
                    original_correct_to_wrong=bool(z[0][i].sigmoid().ge(boundary).eq(labels[i].bool())
                                                  & ~after[0][i].sigmoid().ge(boundary).eq(labels[i].bool())),
                ))

    # Actual pretrained encoder forward/backward and strict save/reload, without
    # optimizer updates. The checkpoint on disk is never changed.
    core.seed_everything(42)
    model.train()
    row, variant, _ = pairs[0]
    input_rows = [row, dict(row, raw_source=variant["raw_source"])]
    masks, _, _ = training_masks(
        model, [row], [dict(selected=variant)], builder, device=device,
        method="prsc", fraction=.10, seed=42, epoch=1,
        references={key: value[1] for key, value in features.items()},
    )
    inputs = builder.codebert_batch(input_rows, device=device)
    logits, representation = model(*inputs, return_representation=True)
    target = logits.new_full((2,), row["label"])
    loss = F.binary_cross_entropy_with_logits(logits, target) + .2 * masked_bce(
        head, representation, target, masks[row["sample_key"]].expand(2, -1), logits.new_full((2,), .5),
    )
    loss.backward()
    gradients = dict(
        encoder_norm=float(model.encoder.embeddings.word_embeddings.weight.grad.norm()),
        head_norm=float(head.weight.grad.norm()),
        all_finite=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
    )
    if not gradients["all_finite"] or not gradients["encoder_norm"] > 0 or not gradients["head_norm"] > 0:
        raise AssertionError("PRSC failed finite encoder/head gradient checks")
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        expected = model(*inputs)
    with tempfile.TemporaryDirectory(prefix="prsc-reload-") as tmp:
        path = Path(tmp) / "model.pt"
        torch.save(dict(model_path=cp["model_path"], state_dict=model.state_dict()), path)
        reloaded = core.load_model(torch.load(path, map_location="cpu", weights_only=False), device)
        with torch.no_grad():
            actual = reloaded(*inputs)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    report = dict(
        checkpoint=str(checkpoint_path), checkpoint_method=cp["method"], checkpoint_epoch=cp["selected_epoch"],
        dataset=dataset, seed=42, source_train=class_counts(train), samples=class_counts(rows),
        analyzed_pairs=class_counts([r for r, _, _ in pairs]), threshold=threshold,
        sample_keys=[r["sample_key"] for r in rows], coverage=dict(coverage), selection_reasons=dict(selection_reasons),
        search_costs=dict(search_costs), independent_costs=dict(second_costs),
        decomposition_max_absolute_error=error, baseline=baseline, masks=summaries,
        real_forward_backward_reload=dict(loss=float(loss.detach()), gradients=gradients,
                                         inference_logits_exact=True, parameter_update=False),
        seconds=time.monotonic() - started, environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                 device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        interpretation="Same-pair logit suppression is descriptive, not evidence of learning. Independent-view effects and classification harm are reported on Train only.",
    )
    write_jsonl(root / "diagnostic.pairs.jsonl", traces)
    _csv(root / "diagnostic.features.csv", details)
    write_json(root / "diagnostic.json", report)
    return report


def comparison(data_dir, output_dir):
    """Report every preset; verify members, training budgets and shared materials."""
    import numpy as np
    from .artifacts import write_json
    from .data import balanced_subset, class_counts, read_jsonl, test_rows, validation_rows
    from .syntax import surface_materials
    from .training import TRAINING

    root = Path(output_dir)
    data = Path(data_dir) / "primevul"
    splits = dict(train=balanced_subset(read_jsonl(data / "train.jsonl"), 1024),
                  valid=balanced_subset(validation_rows(data), 256),
                  test=balanced_subset(test_rows(data), 256))
    members = [dict(split=split, sample_key=r["sample_key"], label=r["label"], project=r.get("project", ""))
               for split, rows in splits.items() for r in rows]
    sets = [{r["sample_key"] for r in rows} for rows in splits.values()]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise AssertionError("pilot Split membership overlaps")
    material_bank = surface_materials(splits["train"])
    clean_rows, robustness_rows, costs_rows, predictions, runs = [], [], [], [], {}
    initial_search, fixed_candidates = None, None
    initial_error = 0.
    for name, method, fraction in PILOT:
        source = root / name / "primevul"
        run = source / "hard_aug_bce"
        summary = json.loads((run / "training.metrics.json").read_text())
        if (summary["training_config"] != TRAINING or summary["optimizer_steps"] != 384
                or summary["completed_epochs"] != 3 or summary["train_samples"] != 1024
                or summary["valid_samples"] != 256 or summary["method"] != "hard_aug_bce"
                or summary["augmentation"]["selection"] != "hardest"):
            raise AssertionError(f"{name}: unmatched training protocol")
        configuration = summary["representation_mask"]
        if ((method == "none" and configuration is not None) or method != "none" and (
                configuration["method"] != method or configuration["fraction"] != fraction
                or configuration["weight"] != .2)):
            raise AssertionError(f"{name}: mask preset changed")
        if json.loads((source / "materials.json").read_text()) != material_bank:
            raise AssertionError(f"{name}: materials must come from the same 1024 Train functions")
        trace = read_jsonl(run / "training.candidates.jsonl")
        for epoch in (1, 2, 3):
            if Counter(t["sample_key"] for t in trace if t["epoch"] == epoch) != Counter(sets[0]):
                raise AssertionError(f"{name}: Train members or number of exposures differ")
        first = trace[:8]
        if initial_search is None:
            initial_search = first
        else:
            for left, right in zip(initial_search, first):
                if (left["sample_key"] != right["sample_key"] or
                    (None if left["selected"] is None else left["selected"]["units"]) !=
                    (None if right["selected"] is None else right["selected"]["units"])):
                    raise AssertionError("initial Hardest candidates/selection changed")
                if left["original_logit"] is not None:
                    initial_error = max(initial_error, abs(left["original_logit"] - right["original_logit"]))
        current_candidates = [read_jsonl(source / f"{bank}.candidates.jsonl") for bank in ("fixed", "heldout")]
        if fixed_candidates is None:
            fixed_candidates = current_candidates
        elif fixed_candidates != current_candidates:
            raise AssertionError("Fixed/Held-out evaluation candidates differ")
        for split in ("valid", "test"):
            predicted = read_jsonl(run / f"{split}.predictions.jsonl")
            if [(r["sample_key"], r["label"]) for r in predicted] != [
                    (r["sample_key"], r["label"]) for r in splits[split]]:
                raise AssertionError(f"{name}: {split} evaluation members differ")
            if split == "test":
                predictions.extend(dict(method=name, **p) for p in predicted)
        epochs = [r for r in read_jsonl(run / "best.training.jsonl") if r["event"] == "epoch"]
        if method != "none":
            for epoch in epochs:
                n = epoch["mask_reasons"].get("eligible", 0)
                costs = epoch["mask_costs"]
                if (costs.get("masked_head_views", 0) != 2 * n
                        or costs.get("function_forwards", 0) != 0
                        or method == "prsc" and costs.get("reused_search_representations", 0) != 2 * n
                        or not math.isclose(epoch["total_loss"], epoch["train_loss"] + .2 * epoch["masked_bce"],
                                            rel_tol=1e-9, abs_tol=1e-9)):
                    raise AssertionError(f"{name}: pair weighting or representation reuse differs")
        selected = max(epochs, key=lambda r: tuple(r["validation"][k] for k in ("mcc", "f1", "accuracy", "auc")))
        if selected["epoch"] != summary["selected_epoch"]:
            raise AssertionError("checkpoint was not selected by source Clean Valid")
        test = json.loads((run / "test.metrics.json").read_text())
        robust = json.loads((run / "robustness.metrics.json").read_text())
        if robust["adaptive"]["query_limit"] != 48:
            raise AssertionError("Adaptive budget differs")
        row = dict(method=name, mask=method, fraction=None if method == "none" else fraction,
                   epoch=summary["selected_epoch"], threshold=test["metrics"]["threshold"],
                   **{k: test["metrics"][k] for k in ("accuracy", "f1", "mcc", "auc", "auprc")})
        clean_rows.append(row)
        for bank in ("fixed", "heldout", "adaptive"):
            group = robust[bank if bank == "adaptive" else bank + "_overall"]
            for scope in ("strict_semantic", "input_only_stress"):
                result = group[scope]
                robustness_rows.append(dict(
                    method=name, bank=bank, scope=scope, eligible=result["eligible"], total=result["total"],
                    coverage=result["coverage"], candidates=result["candidates"],
                    **{k: result["worst_complete"][k] for k in ("accuracy", "f1", "mcc", "auc", "auprc")},
                    asr=result["asr"]["rate"], asr_count=result["asr"]["count"],
                    asr_denominator=result["asr"]["denominator"],
                ))
        counts = Counter()
        for epoch in epochs:
            for component in ("search_costs", "bce_costs", "mask_costs"):
                counts.update(epoch.get(component, {}))
        counts.setdefault("masked_head_views", 0)
        counts.setdefault("reused_search_representations", 0)
        eligible_entries = [t for t in trace if t["selected"] is not None and t["selected"]["strict_semantic"]]
        eligible = len(eligible_entries)
        eligible_keys = {t["sample_key"] for t in eligible_entries}
        train_labels = {r["sample_key"]: r["label"] for r in splits["train"]}
        if method != "none" and eligible != sum(e["mask_reasons"].get("eligible", 0) for e in epochs):
            raise AssertionError("strict selected views and effective regularization coverage differ")
        costs_rows.append(dict(
            method=name, optimizer_steps=summary["optimizer_steps"], train_function_exposures=len(trace),
            strict_pair_exposures=eligible, strict_pair_coverage=eligible / len(trace),
            strict_unique_functions=len(eligible_keys), strict_unique_coverage=len(eligible_keys) / len(train_labels),
            strict_positive_exposures=sum(train_labels[t["sample_key"]] for t in eligible_entries),
            strict_negative_exposures=sum(1 - train_labels[t["sample_key"]] for t in eligible_entries),
            regularized_exposures=eligible if method != "none" else 0,
            epoch_loop_seconds=sum(e["epoch_seconds"] for e in epochs),
            search_seconds=sum(e["search_seconds"] for e in epochs),
            mask_reference_seconds=sum(e.get("mask_seconds", 0) for e in epochs),
            scored_search_candidates=sum(e["candidate_queries"] for e in epochs),
            **dict(counts),
        ))
        runs[name] = dict(config=summary, epochs=epochs, clean=test, robustness=robust)
    if initial_error > 1e-6:
        raise AssertionError(f"initial search logits differ: {initial_error}")

    labels = np.array([r["label"] for r in splits["test"]])
    positive, negative = np.flatnonzero(labels), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(42)
    pc = rng.multinomial(len(positive), np.full(len(positive), 1 / len(positive)), size=2000)
    nc = rng.multinomial(len(negative), np.full(len(negative), 1 / len(negative)), size=2000)
    bootstrap = {}
    for name, _, _ in PILOT:
        values = np.array([p["score"] for p in predictions if p["method"] == name])
        concordance = (values[positive, None] > values[None, negative]).astype(float)
        concordance += .5 * (values[positive, None] == values[None, negative])
        bootstrap[name] = ((pc @ concordance) * nc).sum(1) / (len(positive) * len(negative))
    by_name = {r["method"]: r for r in clean_rows}
    delta_rows = []
    for name, method, fraction in PILOT[1:]:
        references = ["hard_aug_bce"]
        if method == "prsc":
            references += [f"random_{round(fraction * 100)}", f"rsc_{round(fraction * 100)}"]
        for reference in references:
            ci = np.quantile(bootstrap[name] - bootstrap[reference], [.025, .975])
            p = [r for r in predictions if r["method"] == name]
            b = [r for r in predictions if r["method"] == reference]
            correct, base_correct = np.array([r["prediction"] == r["label"] for r in p]), np.array([
                r["prediction"] == r["label"] for r in b])
            delta_rows.append(dict(
                method=name, reference=reference,
                **{f"delta_{k}": by_name[name][k] - by_name[reference][k] for k in ("auc", "mcc", "f1")},
                delta_auc_ci_low=float(ci[0]), delta_auc_ci_high=float(ci[1]),
                corrected=int((correct & ~base_correct).sum()), regressed=int((~correct & base_correct).sum()),
            ))
    robust_by = {(r["method"], r["bank"], r["scope"]): r for r in robustness_rows}
    robust_differences = []
    for pair in delta_rows:
        for bank in ("fixed", "heldout", "adaptive"):
            for scope in ("strict_semantic", "input_only_stress"):
                left = robust_by[(pair["method"], bank, scope)]
                right = robust_by[(pair["reference"], bank, scope)]
                robust_differences.append(dict(
                    method=pair["method"], reference=pair["reference"], bank=bank, scope=scope,
                    **{f"delta_{k}": left[k] - right[k] for k in ("accuracy", "f1", "mcc", "auc", "auprc")},
                    delta_asr=left["asr"] - right["asr"] if left["asr"] is not None and right["asr"] is not None else None,
                    asr_count=left["asr_count"], asr_denominator=left["asr_denominator"],
                    reference_asr_count=right["asr_count"], reference_asr_denominator=right["asr_denominator"],
                    eligible=left["eligible"], reference_eligible=right["eligible"],
                ))
    output = dict(
        protocol=json.loads((root / "protocol.json").read_text()),
        samples={k: class_counts(v) for k, v in splits.items()}, initial_search_logit_max_error=initial_error,
        clean=clean_rows, robustness=robustness_rows, costs=costs_rows, differences=delta_rows,
        robustness_differences=robust_differences,
        bootstrap="2000 paired stratified Test-function resamples, seed 42; exploratory 95% percentile CI, not training-seed uncertainty",
        runs=runs,
    )
    for filename, rows in (("members.csv", members), ("clean.csv", clean_rows),
                           ("robustness.csv", robustness_rows), ("costs.csv", costs_rows),
                           ("differences.csv", delta_rows), ("test.predictions.csv", predictions)):
        _csv(root / filename, rows)
    _csv(root / "robustness.differences.csv", robust_differences)
    write_json(root / "results.json", output)
    return output
