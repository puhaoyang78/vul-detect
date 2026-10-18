"""One CodeBERT training loop for BCE, LP-FT, SWA and source-view baselines."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.optim.swa_utils import AveragedModel
import transformers

from .artifacts import write_json
from .augmentation import search_surface_views, surface_seed
from .data import DATASETS, balanced_subset, read_jsonl, validation_rows
from .metrics import select_threshold
from .model import (
    AutoTokenizer,
    CodeBERT,
    InputBuilder,
    auxiliary_mode,
    predict_logits,
    resolve_device,
    seed_everything,
)
from .progress import TrainingProgress, training_bar
from .syntax import surface_materials
from .vat import VAT_CONFIG, VAT_DIRECTIONS, vat_loss

METHODS = ("bce", "vat", "random_aug_bce", "hard_aug_bce", "random_aug_vat", "hard_aug_vat")
GENERALIZATION_METHODS = ("bce", "lp_ft", "swa")
TRAINING = dict(
    seed=42,
    epochs=3,
    batch_size=4,
    gradient_accumulation=2,
    learning_rate=1e-5,
    weight_decay=0.01,
    source_max_length=2048,
    gradient_clip=1.0,
)


def function_view_batches(records, views, micro_size):
    """Every original function has total BCE weight one, with or without a view."""
    expanded, coefficients = [], []
    for row in records:
        sources = [row["raw_source"], *views.get(row["sample_key"], [])]
        expanded.extend(dict(row, raw_source=source) for source in sources)
        coefficients.extend([1.0 / len(sources)] * len(sources))
    for start in range(0, len(expanded), micro_size):
        yield expanded[start : start + micro_size], coefficients[start : start + micro_size]


def prepare_materials(data_dir, output_dir, dataset):
    path = Path(output_dir) / dataset / "materials.json"
    if path.exists():
        return json.loads(path.read_text())
    materials = surface_materials(read_jsonl(Path(data_dir) / dataset / "train.jsonl"))
    write_json(path, materials)
    return materials


def train(
    data_dir,
    output_dir,
    model_path,
    dataset,
    method,
    *,
    device="auto",
    materials=None,
    config=None,
    hard_selection="hardest",
    vat_direction="finite_difference",
    train_limit=None,
    valid_limit=None,
):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    if method not in (*METHODS, "lp_ft", "swa"):
        raise ValueError(f"unknown method: {method}")
    if hard_selection not in ("hardest", "boundary") or vat_direction not in VAT_DIRECTIONS:
        raise ValueError("unknown hard selection or VAT direction")
    if hard_selection != "hardest" and method not in ("hard_aug_bce", "hard_aug_vat"):
        raise ValueError("boundary selection requires a hard augmentation method")
    if vat_direction != "finite_difference" and method not in ("vat", "random_aug_vat", "hard_aug_vat"):
        raise ValueError("logit direction requires a VAT method")
    config = dict(TRAINING if config is None else config)
    if (
        config["batch_size"] <= 0
        or config["gradient_accumulation"] <= 0
        or config["epochs"] <= 0
        or config["learning_rate"] <= 0
    ):
        raise ValueError("invalid training budget")
    if method in ("lp_ft", "swa") and config["epochs"] < 3:
        raise ValueError("LP-FT and SWA require at least three total epochs")
    run = Path(output_dir) / dataset / method
    run.mkdir(parents=True, exist_ok=True)
    target = run / "best.pt"
    if target.exists() or target.with_suffix(".training.jsonl").exists():
        raise FileExistsError(
            f"{run}: training already started; refusing to reuse fitted weights"
        )
    train_rows = balanced_subset(
        read_jsonl(Path(data_dir) / dataset / "train.jsonl"), train_limit, config["seed"]
    )
    valid_rows = balanced_subset(validation_rows(Path(data_dir) / dataset), valid_limit, config["seed"])
    steps_per_epoch = math.ceil(len(train_rows) / (config["batch_size"] * config["gradient_accumulation"]))
    if method == "swa" and steps_per_epoch < 2:
        raise ValueError("SWA requires two distinct weight samples per averaging epoch")
    for split, rows in (("train", train_rows), ("valid", valid_rows)):
        if not rows or {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{split}: both classes required")
        if any(r["dataset"] != dataset or r["split"] != split for r in rows):
            raise ValueError(f"training and selection require {dataset} {split}")
    source_views = method in ("random_aug_bce", "hard_aug_bce", "random_aug_vat", "hard_aug_vat")
    use_vat = method in ("vat", "random_aug_vat", "hard_aug_vat")
    search = method in ("hard_aug_bce", "hard_aug_vat")
    if source_views and materials is None:
        if train_limit is None:
            materials = prepare_materials(data_dir, output_dir, dataset)
        else:
            materials = surface_materials(train_rows, seed=config["seed"])
            write_json(Path(output_dir) / dataset / "materials.json", materials)
    device = resolve_device(device)
    seed = config["seed"]
    seed_everything(seed)
    builder = InputBuilder(
        AutoTokenizer.from_pretrained(model_path),
        source_max_length=config["source_max_length"],
    )
    model = CodeBERT(model_path, device, gradient_checkpointing=device.type == "cuda")
    seed_everything(seed)
    averaged = AveragedModel(model) if method == "swa" else None
    averaged_steps = []
    history = TrainingProgress(target)
    metadata = dict(
        dataset=dataset,
        method=method,
        model_path=str(model_path),
        training_config=config,
        train_samples=len(train_rows),
        valid_samples=len(valid_rows),
        diagnostic_limits=dict(train=train_limit, valid=valid_limit),
        environment=dict(
            torch=torch.__version__,
            transformers=transformers.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        ),
        selection=f"{dataset} clean Valid: threshold MCC/F1/Accuracy/closest to 0.5; "
        "epoch MCC/F1/Accuracy/AUC",
        precision="FP32 parameters/optimizer; BF16 CUDA main forward; FP32 auxiliary VAT",
        input_protocol=dict(
            logical_budget=config["source_max_length"],
            prefix_tokens=len(builder.source_prefix),
            source_budget=config["source_max_length"] - len(builder.source_prefix) - 2,
            window=512,
            stride=(512 - len(builder.source_prefix) - 2) // 2,
            pooling="mean over unique source positions; prefix/BOS/EOS averaged across windows",
        ),
        vat=dict(VAT_CONFIG, direction=vat_direction) if use_vat else None,
        lp_ft=(dict(linear_probe_epochs=1, linear_probe_learning_rate=1e-3,
                    fine_tune_epochs=config["epochs"] - 1, encoder_dropout_during_probe=False,
                    reset_optimizer_at_fine_tune=True,
                    selection="fine-tuning epochs only",
                    reference="https://openreview.net/forum?id=UYneFzXSJWh")
               if method == "lp_ft" else None),
        swa=(dict(start_epoch=2, samples_per_epoch=2, averaging="equal parameter weights",
                  sample_steps_per_epoch=[math.ceil(steps_per_epoch / 2), steps_per_epoch],
                  learning_rate="unchanged constant BCE learning rate",
                  selection="averaged models only, at epoch ends with at least two weight samples",
                  reference="https://arxiv.org/abs/1803.05407") if method == "swa" else None),
        augmentation=(
            dict(
                search=search,
                selection=hard_selection if search else "random",
                beam_width=2,
                depth=3,
                candidate_limit=12,
                proposal_limit=24,
                comment_token_limit=64,
            )
            if source_views
            else None
        ),
    )
    write_json(run / "config.json", metadata)
    best_key = (float("-inf"),) * 4
    global_step = 0
    selected_epoch = None
    batch_size = config["batch_size"]
    accumulation = config["gradient_accumulation"]
    for epoch in range(1, config["epochs"] + 1):
        started = time.monotonic()
        linear_probe = method == "lp_ft" and epoch == 1
        if epoch == 1 or (method == "lp_ft" and epoch == 2):
            model.encoder.requires_grad_(not linear_probe)
            optimizer = torch.optim.AdamW(
                (p for p in model.parameters() if p.requires_grad),
                lr=1e-3 if linear_probe else config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
        order = list(range(len(train_rows)))
        random.Random(seed + epoch - 1).shuffle(order)
        model.train()
        if linear_probe:
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        running, vat_total, transformed, queries, original_queries = 0.0, 0.0, 0, 0, 0
        vat_eligible, vat_zero, vat_norm_min, vat_norm_max = 0, 0, None, None
        search_seconds = 0.0
        selection_reasons = Counter()
        search_costs = Counter()
        bce_costs = Counter()
        vat_costs = Counter()
        steps = 0
        num_batches = math.ceil(len(order) / batch_size)
        bar = training_bar(
            range(0, len(order), batch_size),
            desc=f'{dataset}/{method} epoch {epoch}/{config["epochs"]}',
        )
        for batch_index, start in enumerate(bar):
            batch = [train_rows[i] for i in order[start : start + batch_size]]
            group_start = batch_index // accumulation * accumulation
            group_end = min(group_start + accumulation, num_batches)
            group_samples = min(group_end * batch_size, len(order)) - group_start * batch_size
            selections, views = [], {}
            if source_views:
                search_started = time.monotonic()

                def score(rows):
                    with torch.no_grad(), auxiliary_mode(model):
                        values = []
                        for at in range(0, len(rows), batch_size):
                            inputs = builder.codebert_batch(
                                rows[at : at + batch_size], device=device
                            )
                            values.extend(model(*inputs).detach().cpu().tolist())
                            search_costs.update(
                                function_forwards=inputs[-1], encoder_sequences=len(inputs[0]),
                                encoder_token_slots=inputs[0].numel(),
                                encoder_active_tokens=int(inputs[1].sum()),
                            )
                        return values

                selections = search_surface_views(
                    batch,
                    score,
                    builder.tokenizer,
                    config["source_max_length"] - len(builder.source_prefix) - 2,
                    materials,
                    seed=seed,
                    epoch=epoch,
                    query_limit=12,
                    random_selection=not search,
                    selection=hard_selection,
                )
                search_seconds += time.monotonic() - search_started
                views = {
                    r["sample_key"]: [s["selected"]["raw_source"]] if s["selected"] else []
                    for r, s in zip(batch, selections)
                }
                transformed += sum(s["selected"] is not None for s in selections)
                queries += sum(s["candidate_queries"] for s in selections)
                original_queries += sum(s["original_queries"] for s in selections)
                selection_reasons.update(s["selection_reason"] for s in selections)
                with (run / "training.candidates.jsonl").open("a", encoding="utf-8") as handle:
                    for entry in selections:
                        chosen = entry["selected"]
                        handle.write(
                            json.dumps(
                                dict(
                                    sample_key=entry["sample_key"],
                                    epoch=epoch,
                                    selected=(
                                        None
                                        if chosen is None
                                        else {
                                            k: v
                                            for k, v in chosen.items()
                                            if k != "raw_source"
                                        }
                                    ),
                                    candidate_queries=entry["candidate_queries"],
                                    original_queries=entry["original_queries"],
                                    selection_reason=entry["selection_reason"],
                                    stop_depth=entry["stop_depth"],
                                    original_logit=entry["original_logit"],
                                    loss_delta=entry["loss_delta"],
                                    visibility_encodings=entry["visibility_encodings"],
                                    reasons=entry["reasons"],
                                )
                            )
                            + "\n"
                        )
            batch_loss = 0.0
            for forwarded, weights in function_view_batches(batch, views, batch_size):
                inputs = builder.codebert_batch(forwarded, device=device)
                logits = model(*inputs)
                bce_costs.update(
                    function_forwards=inputs[-1], encoder_sequences=len(inputs[0]),
                    encoder_token_slots=inputs[0].numel(),
                    encoder_active_tokens=int(inputs[1].sum()),
                )
                labels = logits.new_tensor([r["label"] for r in forwarded])
                terms = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                loss = (terms * terms.new_tensor(weights)).sum()
                if not torch.isfinite(loss):
                    raise ValueError(
                        f"nonfinite classification loss: {method} epoch {epoch}, batch {batch_index}"
                    )
                (loss / group_samples).backward()
                batch_loss += float(loss.detach())
            if use_vat:
                vat_rows = []
                for i, row in enumerate(batch):
                    selected = selections[i]["selected"] if selections else None
                    rng = random.Random(
                        surface_seed(seed, epoch, row["sample_key"], "vat-view")
                    )
                    vat_rows.append(
                        dict(row, raw_source=selected["raw_source"])
                        if selected is not None and rng.randrange(2)
                        else row
                    )
                with auxiliary_mode(model):
                    terms, info = vat_loss(
                        model,
                        vat_rows,
                        builder,
                        device=device,
                        noise_seed=seed + (epoch - 1) * len(order) + start,
                        direction_method=vat_direction,
                    )
                    if terms.requires_grad:
                        (terms.sum() / group_samples).backward()
                vat_total += float(terms.detach().sum())
                vat_eligible += info["eligible"]
                vat_zero += info["zero_direction"]
                vat_costs.update({k: info[k] for k in (
                    "function_forwards", "encoder_sequences", "encoder_token_slots",
                    "encoder_active_tokens", "zero_initial_projection",
                ) if info[k] is not None})
                if info["norm_min"] is not None:
                    vat_norm_min = (
                        min(vat_norm_min, info["norm_min"])
                        if vat_norm_min is not None
                        else info["norm_min"]
                    )
                    vat_norm_max = (
                        max(vat_norm_max, info["norm_max"])
                        if vat_norm_max is not None
                        else info["norm_max"]
                    )
            running += batch_loss
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == num_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                global_step += 1
                if (averaged is not None and epoch >= 2
                        and steps in metadata["swa"]["sample_steps_per_epoch"]):
                    averaged.update_parameters(model)
                    averaged_steps.append(global_step)
                    history.add(dict(event="swa_sample", epoch=epoch, step=global_step,
                                     averaged_models=len(averaged_steps)))
                if global_step == 1 or global_step % 25 == 0 or batch_index + 1 == num_batches:
                    history.add(
                        dict(
                            event="step",
                            epoch=epoch,
                            step=global_step,
                            loss=batch_loss / len(batch),
                            samples_seen=min(start + len(batch), len(order)),
                        )
                    )
                    print(
                        f"{dataset}/{method} epoch {epoch} step {steps}/{math.ceil(len(order) / (batch_size * accumulation))}",
                        flush=True,
                    )
        candidate = averaged.module if averaged_steps else model
        eligible = not linear_probe and (averaged is None or len(averaged_steps) >= 2)
        probabilities = predict_logits(
            candidate, valid_rows, builder, device=device, batch_size=batch_size
        ).sigmoid()
        threshold, validation = select_threshold(
            [r["label"] for r in valid_rows], probabilities.tolist()
        )
        history.add(
            dict(
                event="epoch",
                method=method,
                epoch=epoch,
                train_loss=running / len(train_rows),
                total_loss=(running + vat_total) / len(train_rows),
                optimizer_steps=steps,
                validation_threshold=threshold,
                validation=validation,
                augmented_functions=transformed,
                candidate_queries=queries,
                original_search_queries=original_queries,
                search_seconds=search_seconds,
                selection_reasons=dict(selection_reasons),
                search_costs=dict(search_costs),
                bce_costs=dict(bce_costs),
                vat_costs=dict(vat_costs),
                vat_loss=vat_total / len(train_rows),
                vat_eligible_functions=vat_eligible,
                vat_zero_direction_functions=vat_zero,
                vat_norm_min=vat_norm_min,
                vat_norm_max=vat_norm_max,
                epoch_seconds=time.monotonic() - started,
                stage="linear_probe" if linear_probe else "fine_tune",
                learning_rate=optimizer.param_groups[0]["lr"],
                encoder_trainable=not linear_probe,
                validation_model="swa" if averaged_steps else "online",
                selection_eligible=eligible,
                averaged_steps=list(averaged_steps),
            )
        )
        key = tuple(
            validation[k] if validation[k] is not None else float("-inf")
            for k in ("mcc", "f1", "accuracy", "auc")
        )
        if eligible and key > best_key:
            best_key = key
            selected_epoch = epoch
            checkpoint = dict(
                metadata,
                state_dict={
                    k: v.detach().cpu().clone() for k, v in candidate.state_dict().items()
                },
                source_max_length=config["source_max_length"],
                decision_threshold=threshold,
                selected_epoch=epoch,
                validation=validation,
                averaged_steps=list(averaged_steps),
            )
            temporary = target.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(target)
            del checkpoint
    summary = dict(
        metadata,
        selected_epoch=selected_epoch,
        completed_epochs=config["epochs"],
        optimizer_steps=global_step,
        averaged_steps=averaged_steps,
    )
    write_json(run / "training.metrics.json", summary)
    return summary
