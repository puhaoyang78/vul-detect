"""Read-only CSD diagnostics on source Train; no optimizer or target data."""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import torch

from .artifacts import write_json
from .data import read_jsonl
from .metrics import decision_boundary, metrics
from . import model as core
from .progress import training_bar

DATASETS = ("primevul", "secvuleval")
SEED = 42
PROBE_SIZE = 8


def distribution(values):
    x = np.asarray(values, dtype=float).ravel()
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("nonempty finite diagnostic values required")
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), p01=float(np.quantile(x, .01)),
                p25=float(np.quantile(x, .25)), median=float(np.median(x)),
                p75=float(np.quantile(x, .75)), p95=float(np.quantile(x, .95)),
                p99=float(np.quantile(x, .99)), max=float(x.max()),
                l2=float(np.linalg.norm(x)))


def head_statistics(head):
    w = head.common.weight.detach().cpu().double()
    u = head.specific_weight.detach().cpu().double()
    a = head.coefficients.weight.detach().cpu().double()
    b = head.specific_bias.detach().cpu().double()
    weights = torch.cat((w[None], u))
    gram = torch.einsum("rch,sch->crs", weights, weights)
    error = (gram - torch.eye(len(weights))).square()
    diagonal = error.diagonal(dim1=1, dim2=2).sum()
    total = error.sum()
    residual = torch.einsum("dr,rch->dch", a, u)
    residual_bias = a @ b
    relative = residual.flatten(1).norm(dim=1) / w.norm()
    decision_relative = (residual[:, 1] - residual[:, 0]).norm(dim=1) / (w[1] - w[0]).norm()
    cosine = torch.einsum("ch,rch->cr", w, u) / (w.norm(dim=1)[:, None] * u.norm(dim=2).T)
    return dict(
        parameters={name: distribution(p.detach().cpu().numpy()) for name, p in head.named_parameters()},
        common_row_norms=w.norm(dim=1).tolist(), basis_row_norms=u.norm(dim=2).tolist(),
        common_weight_norm=float(w.norm()), common_decision_weight_norm=float((w[1] - w[0]).norm()),
        basis_weight_norm=float(u.norm()), coefficients_abs=distribution(a.abs().numpy()),
        residual_weight_norm=distribution(residual.flatten(1).norm(dim=1).numpy()),
        relative_residual_weight_norm=distribution(relative.numpy()),
        relative_decision_residual_weight_norm=distribution(decision_relative.numpy()),
        residual_bias_norm=distribution(residual_bias.norm(dim=1).numpy()),
        fraction_projects_residual_below_0p1_percent=float((relative < .001).double().mean()),
        fraction_projects_residual_below_1_percent=float((relative < .01).double().mean()),
        gram=gram.tolist(), common_basis_cosines=cosine.tolist(),
        orthonormality_sum=float(total), orthonormality_mean=float(error.mean()),
        sum_to_mean_factor=error.numel(), diagonal_norm_penalty=float(diagonal),
        off_diagonal_penalty=float(total - diagonal), diagonal_fraction=float(diagonal / total) if total else None,
    )


def paired_metrics(predictions, threshold):
    labels = [r["label"] for r in predictions]
    logits = torch.tensor([[r["common_logit"], r["specific_logit"]] for r in predictions], dtype=torch.float64)
    # Preserve the existing float32 score/threshold protocol; raw-logit AUC
    # additionally reveals effects hidden by sigmoid rounding or saturation.
    probabilities = logits.float().sigmoid().tolist()
    values = {}
    for column, name in enumerate(("common", "specific")):
        value = metrics(labels, [r[column] for r in probabilities], threshold)
        value["bce"] = float(torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, column], torch.tensor(labels, dtype=torch.float64)))
        value["auc_logits"] = float(roc_auc_score(labels, logits[:, column])) if len(set(labels)) == 2 else None
        values[name] = value
    decisions = np.asarray(probabilities) >= decision_boundary(threshold)
    correct = decisions == np.asarray(labels)[:, None]
    flips = decisions[:, 0] != decisions[:, 1]
    deltas = (logits[:, 1] - logits[:, 0]).numpy()
    common_norm = float(logits[:, 0].norm())
    return dict(
        **values, logit_delta=distribution(deltas), absolute_logit_delta=distribution(abs(deltas)),
        delta_to_common_logit_l2=float(np.linalg.norm(deltas) / common_norm) if common_norm else None,
        flips=int(flips.sum()), denominator=len(labels), flip_rate=float(flips.mean()),
        correct_to_wrong=int((correct[:, 0] & ~correct[:, 1]).sum()),
        wrong_to_correct=int((~correct[:, 0] & correct[:, 1]).sum()),
        bce_reduction=values["common"]["bce"] - values["specific"]["bce"],
        auc_gain=values["specific"]["auc"] - values["common"]["auc"] if len(set(labels)) == 2 else None,
        auc_logits_gain=values["specific"]["auc_logits"] - values["common"]["auc_logits"] if len(set(labels)) == 2 else None,
    )


def project_statistics(rows, predictions, head, projects, threshold):
    by_project = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        if row["sample_key"] != prediction["sample_key"] or row["label"] != prediction["label"]:
            raise ValueError("Train predictions differ from source members/order")
        by_project[row["project"]].append(prediction)
    w = head.common.weight.detach().cpu().double()
    u = head.specific_weight.detach().cpu().double()
    a = head.coefficients.weight.detach().cpu().double()
    residual = torch.einsum("dr,rch->dch", a, u)
    bias = a @ head.specific_bias.detach().cpu().double()
    records = []
    for i, project in enumerate(projects):
        members = by_project[project]
        value = paired_metrics(members, threshold)
        positive = sum(r["label"] for r in members)
        records.append(dict(
            project=project, domain_id=i, samples=len(members), positive=positive,
            negative=len(members) - positive, positive_fraction=positive / len(members),
            single_class=positive in (0, len(members)), coefficient=float(a[i, 0]),
            coefficient_norm=float(a[i].norm()), residual_weight_norm=float(residual[i].norm()),
            relative_residual_weight_norm=float(residual[i].norm() / w.norm()),
            residual_decision_weight_norm=float((residual[i, 1] - residual[i, 0]).norm()),
            relative_decision_residual_weight_norm=float((residual[i, 1] - residual[i, 0]).norm() / (w[1] - w[0]).norm()),
            residual_bias_norm=float(bias[i].norm()), residual_decision_bias=float(bias[i, 1] - bias[i, 0]),
            common_bce=value["common"]["bce"], specific_bce=value["specific"]["bce"],
            bce_reduction=value["bce_reduction"], common_auc=value["common"]["auc"],
            specific_auc=value["specific"]["auc"], auc_gain=value["auc_gain"],
            mean_delta_logit=value["logit_delta"]["mean"],
            mean_abs_delta_logit=value["absolute_logit_delta"]["mean"],
            flips=value["flips"], flip_rate=value["flip_rate"],
            correct_to_wrong=value["correct_to_wrong"], wrong_to_correct=value["wrong_to_correct"],
        ))
    sizes = [r["samples"] for r in records]
    single = [r for r in records if r["single_class"]]
    entropy = lambda p: -sum(t * math.log2(t) for t in (p, 1-p) if t > 0)
    positive_fraction = sum(r["label"] for r in rows) / len(rows)
    summary = dict(
        projects=len(records), samples=len(rows), size=distribution(sizes),
        single_class_projects=len(single), single_class_project_fraction=len(single) / len(records),
        positive_only_projects=sum(r["positive"] == r["samples"] for r in records),
        negative_only_projects=sum(r["positive"] == 0 for r in records),
        single_class_samples=sum(r["samples"] for r in single),
        single_class_sample_fraction=sum(r["samples"] for r in single) / len(rows),
        mixed_projects=len(records) - len(single),
        projects_at_most_four_samples=sum(n <= 4 for n in sizes),
        top1_sample_fraction=max(sizes) / len(rows),
        top5_sample_fraction=sum(sorted(sizes, reverse=True)[:5]) / len(rows),
        top10_sample_fraction=sum(sorted(sizes, reverse=True)[:10]) / len(rows),
        project_label_mutual_information_bits=entropy(positive_fraction) - sum(
            r["samples"] * entropy(r["positive_fraction"]) for r in records) / len(rows),
        train_project_majority_accuracy=sum(max(r["positive"], r["negative"]) for r in records) / len(rows),
    )
    correlations = {}
    for left, right in (
        ("samples", "coefficient_norm"), ("samples", "mean_abs_delta_logit"),
        ("samples", "bce_reduction"), ("positive_fraction", "coefficient"),
        ("positive_fraction", "mean_delta_logit"), ("class_imbalance", "coefficient_norm"),
    ):
        x = [abs(r["positive_fraction"] - .5) if left == "class_imbalance" else r[left] for r in records]
        y = [r[right] for r in records]
        correlations[f"{left}__{right}"] = float(spearmanr(x, y).statistic) if np.ptp(x) and np.ptp(y) else None
    groups = []
    for name, lower, upper in (("1", 1, 1), ("2-4", 2, 4), ("5-19", 5, 19),
                               ("20-99", 20, 99), ("100+", 100, math.inf)):
        members = [r for r in records if lower <= r["samples"] <= upper]
        if members:
            groups.append(dict(size_group=name, projects=len(members), samples=sum(r["samples"] for r in members),
                               median_abs_coefficient=float(np.median([r["coefficient_norm"] for r in members])),
                               median_relative_residual=float(np.median([r["relative_residual_weight_norm"] for r in members])),
                               flips=sum(r["flips"] for r in members),
                               sample_mean_abs_logit_delta=sum(r["samples"] * r["mean_abs_delta_logit"] for r in members) / sum(r["samples"] for r in members)))
    return records, summary, correlations, groups


def gradient_probe(model, rows, builder, projects, device, batch_size):
    """Measure gradients without any optimizer/parameter update; dropout is off."""
    domain_ids = {p: i for i, p in enumerate(projects)}
    model.eval()
    model.zero_grad(set_to_none=True)
    with core.auxiliary_mode(model):
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            common, specific = model(*builder.codebert_batch(batch, device=device),
                                     project_ids=torch.tensor([domain_ids[r["project"]] for r in batch], device=device))
            labels = common.new_tensor([r["label"] for r in batch])
            loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(z, labels, reduction="sum")
                       for z in (common, specific)) / len(rows)
            loss.backward()
    encoder_norm2 = sum(float(p.grad.norm()) ** 2 for p in model.encoder.parameters() if p.grad is not None)
    head = model.task_modules["classifier"]
    ce = {k: p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p, device="cpu")
          for k, p in head.named_parameters()}
    model.zero_grad(set_to_none=True)
    regularizer = head.orthonormality()
    regularizer.backward()
    reg = {k: p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p, device="cpu")
           for k, p in head.named_parameters()}
    factor = head.common.out_features * (len(head.specific_weight) + 1) ** 2
    norm = lambda parts: math.sqrt(sum(float(g.double().square().sum()) for g in parts))
    ce_head_norm, reg_norm = norm(ce.values()), norm(reg.values())
    ce_norm = math.sqrt(encoder_norm2 + ce_head_norm ** 2)
    total = math.sqrt(encoder_norm2 + norm(ce[k] + reg[k] for k in ce) ** 2)
    mean_total = math.sqrt(encoder_norm2 + norm(ce[k] + reg[k] / factor for k in ce) ** 2)
    model.zero_grad(set_to_none=True)
    return dict(
        samples=len(rows), sample_keys=[r["sample_key"] for r in rows], mode="dropout off; no optimizer or parameter updates",
        classification_encoder_grad_norm=math.sqrt(encoder_norm2), classification_head_grad_norm=ce_head_norm,
        classification_total_grad_norm=ce_norm, regularizer_sum_grad_norm=reg_norm,
        regularizer_mean_grad_norm=reg_norm / factor,
        regularizer_to_classification_grad_ratio=reg_norm / ce_norm if ce_norm else None,
        regularizer_to_classification_head_grad_ratio=reg_norm / ce_head_norm if ce_head_norm else None,
        head_classification_grad_norms={k: float(v.norm()) for k, v in ce.items()},
        head_regularizer_grad_norms={k: float(v.norm()) for k, v in reg.items()},
        combined_sum_grad_norm=total, combined_mean_grad_norm=mean_total,
        clip_scale_sum_at_norm_one=min(1., 1. / (total + 1e-6)),
        clip_scale_mean_at_norm_one=min(1., 1. / (mean_total + 1e-6)),
    )


@torch.no_grad()
def predict_train(model, rows, builder, projects, device, batch_size):
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    domain_ids = {p: i for i, p in enumerate(projects)}
    model.eval()
    predictions = [None] * len(rows)
    for start in training_bar(range(0, len(rows), batch_size), desc="CSD source Train"):
        indexes = order[start:start + batch_size]
        batch = [rows[i] for i in indexes]
        common, specific = model(*builder.codebert_batch(batch, device=device),
                                 project_ids=torch.tensor([domain_ids[r["project"]] for r in batch], device=device))
        if not torch.isfinite(common).all() or not torch.isfinite(specific).all():
            raise ValueError("nonfinite CSD Train diagnostic logits")
        for i, a, b in zip(indexes, common.cpu().tolist(), specific.cpu().tolist()):
            predictions[i] = dict(sample_key=rows[i]["sample_key"], project=rows[i]["project"],
                                  label=rows[i]["label"], common_logit=a, specific_logit=b,
                                  delta_logit=b-a)
        if start == 0 or (start // batch_size + 1) % 250 == 0:
            print(f"Source Train diagnostics: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return predictions


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def diagnose(data_dir, output_dir, dataset, *, device="auto"):
    if dataset not in DATASETS:
        raise ValueError("CSD diagnostics supports PrimeVul and SecVulEval")
    run = Path(output_dir) / dataset / "csd"
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["dataset"] != dataset or checkpoint["method"] != "csd" or checkpoint["csd"]["rank"] != 1:
        raise ValueError("source rank-one CSD checkpoint required")
    rows = read_jsonl(Path(data_dir) / dataset / "train.jsonl")
    projects = checkpoint["csd"]["projects"]
    if sorted({r["project"] for r in rows}) != projects or any(r["split"] != "train" or r["dataset"] != dataset for r in rows):
        raise ValueError("only the checkpoint's source Train projects/members may be used")
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    probe = [rows[i] for i in order[:PROBE_SIZE]]
    batch_size = checkpoint["training_config"]["batch_size"]
    device = core.resolve_device(device)
    builder = core.InputBuilder(core.AutoTokenizer.from_pretrained(checkpoint["model_path"]),
                                source_max_length=checkpoint["source_max_length"])
    core.seed_everything(checkpoint["training_config"]["seed"])
    initial_model = core.CodeBERT(checkpoint["model_path"], device, csd=checkpoint["csd"],
                                 gradient_checkpointing=device.type == "cuda")
    initial = head_statistics(initial_model.task_modules["classifier"])
    initial_gradients = gradient_probe(initial_model, probe, builder, projects, device, batch_size)
    del initial_model
    model = core.load_model(checkpoint, device)
    if device.type == "cuda":
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    final = head_statistics(model.task_modules["classifier"])
    gradients = gradient_probe(model, probe, builder, projects, device, batch_size)
    predictions = predict_train(model, rows, builder, projects, device, batch_size)
    threshold = checkpoint["decision_threshold"]
    records, composition, correlations, groups = project_statistics(
        rows, predictions, model.task_modules["classifier"], projects, threshold)
    epochs = []
    factor = final["sum_to_mean_factor"]
    for event in read_jsonl(run / "best.training.jsonl"):
        if event["event"] == "epoch":
            losses = event["csd_losses"]
            ce = losses["common_loss"] + losses["specific_loss"]
            reg = losses["orthonormality"]
            epochs.append(dict(dataset=dataset, epoch=event["epoch"], common_ce=losses["common_loss"],
                               specific_ce=losses["specific_loss"], orthonormality_sum=reg,
                               regularizer_over_combined_ce=reg / ce, regularizer_loss_fraction=reg / (reg + ce),
                               mean_at_same_parameters=reg / factor,
                               mean_regularizer_loss_fraction=(reg / factor) / (reg / factor + ce)))
    result = dict(
        dataset=dataset, selected_epoch=checkpoint["selected_epoch"], threshold=threshold,
        protocol=dict(seed=SEED, cohort="all source Train members, Seed-42 shuffled inference; CSV retains source order",
                      train_samples=len(rows), gradient_probe_samples=len(probe),
                      inference_batch_size=batch_size, labels="native saved Train labels",
                      target_data_read=False, optimizer_updates=0, inference="same encoder forward supplies both heads",
                      model_path=checkpoint["model_path"], training_config=checkpoint["training_config"],
                      csd={k: v for k, v in checkpoint["csd"].items() if k != "projects"},
                      checkpoint_environment=checkpoint["environment"], torch=torch.__version__,
                      initialization="reconstructed from the same pretrained model, constructor and seed; not a saved initial checkpoint",
                      gradient_probe="eval/dropout-off gradients at reconstructed initialization and saved checkpoint; not historical training-step gradients"),
        initial_head=initial, final_head=final, initial_gradient_probe=initial_gradients,
        final_gradient_probe=gradients, train=paired_metrics(predictions, threshold),
        train_at_threshold_half=paired_metrics(predictions, .5), epochs=epochs,
        project_composition=composition, project_spearman_correlations=correlations, project_size_groups=groups,
        references=dict(paper="https://proceedings.mlr.press/v119/piratla20a.html",
                        author_loss="https://github.com/vihari/CSD/blob/master/pacs/train_csd.py#L85-L94",
                        author_head="https://github.com/vihari/CSD/blob/master/pacs/models/resnet_csd.py#L20-L25"),
    )
    root = Path(output_dir) / "csd_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / f"{dataset}.train.csv", predictions)
    _write_csv(root / f"{dataset}.projects.csv", records)
    write_json(root / f"{dataset}.json", result)
    return dict(dataset=dataset, samples=len(rows), projects=len(projects), output=str(root))


def report(output_dir):
    root = Path(output_dir) / "csd_diagnostics"
    summaries, epochs = [], []
    for dataset in DATASETS:
        r = json.loads((root / f"{dataset}.json").read_text())
        train, head, projects = r["train"], r["final_head"], r["project_composition"]
        summaries.append(dict(
            dataset=dataset, train_samples=projects["samples"], projects=projects["projects"],
            median_project_size=projects["size"]["median"], single_class_projects=projects["single_class_projects"],
            single_class_samples=projects["single_class_samples"],
            median_abs_coefficient=head["coefficients_abs"]["median"],
            median_relative_residual=head["relative_residual_weight_norm"]["median"],
            max_relative_residual=head["relative_residual_weight_norm"]["max"],
            mean_abs_logit_delta=train["absolute_logit_delta"]["mean"], flips=train["flips"],
            flip_denominator=train["denominator"], flip_rate=train["flip_rate"],
            common_bce=train["common"]["bce"], specific_bce=train["specific"]["bce"],
            common_auc=train["common"]["auc"], specific_auc=train["specific"]["auc"],
            auc_gain=train["auc_gain"], bce_reduction=train["bce_reduction"],
        ))
        epochs.extend(r["epochs"])
    _write_csv(root / "summary.csv", summaries)
    _write_csv(root / "epochs.csv", epochs)
    return dict(datasets=len(summaries), epoch_rows=len(epochs), output=str(root))
