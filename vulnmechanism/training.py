"""Shared training for CodeBERT augmentation/VAT and UniXcoder LOSVER."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import transformers

from .artifacts import write_json
from .augmentation import search_surface_views, surface_seed
from .data import DATASETS, balanced_subset, read_jsonl, validation_rows, write_jsonl
from .metrics import metrics, select_threshold
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

METHODS = (
    "bce",
    "vat",
    "random_aug_bce",
    "hard_aug_bce",
    "random_aug_vat",
    "hard_aug_vat",
)
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


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_state(run, model, optimizer, metadata, loop, *, scheduler=None):
    """Only called at optimizer boundaries; an interrupted write keeps last.pt intact."""
    run = Path(run)
    checkpoint = dict(
        config={
            k: v for k, v in metadata.items()
            if k not in ("environment", "checkpoint_interval")
        },
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict() if scheduler is not None else None,
        loop=loop,
        rng=rng_state(),
        log_positions={
            name: (run / name).stat().st_size
            for name in ("best.training.jsonl", "training.candidates.jsonl")
            if (run / name).exists()
        },
    )
    temporary = run / "last.pt.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(run / "last.pt")


def load_training_state(run, model, optimizer, metadata, *, scheduler=None):
    run = Path(run)
    cp = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    expected = {
        k: v for k, v in metadata.items()
        if k not in ("environment", "checkpoint_interval")
    }
    if cp["config"] != expected:
        raise ValueError(f"{run}: resume configuration or data members differ")
    model.load_state_dict(cp["model"])
    optimizer.load_state_dict(cp["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(cp["scheduler"])
    # Work after the last committed optimizer step will be replayed, including
    # validation and candidate selection. Discard its incomplete/duplicate logs.
    for name in ("best.training.jsonl", "training.candidates.jsonl"):
        path = run / name
        position = cp["log_positions"].get(name, 0)
        if position and (not path.exists() or path.stat().st_size < position):
            raise ValueError(f"{path}: training history is shorter than the checkpoint")
        if path.exists():
            with path.open("r+b") as handle:
                handle.truncate(position)
    return cp["loop"], cp["rng"]


def function_view_batches(records, views, micro_size):
    """Every original function has total BCE weight one, with or without a view."""
    expanded, coefficients = [], []
    for row in records:
        sources = [row["raw_source"], *views.get(row["sample_key"], [])]
        expanded.extend(dict(row, raw_source=source) for source in sources)
        coefficients.extend([1.0 / len(sources)] * len(sources))
    for start in range(0, len(expanded), micro_size):
        yield expanded[start : start + micro_size], coefficients[
            start : start + micro_size
        ]


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
    paired_input="512",
    window_micro_batch=None,
    function_micro_batch=None,
    resume=False,
    save_every=100,
    paired_epochs=None,
):
    if save_every < 1:
        raise ValueError("save_every must be positive")
    if method in LOSVER_METHODS:
        if (
            dataset != "primevul_paired"
            or config is not None
            or hard_selection != "hardest"
            or vat_direction != "finite_difference"
        ):
            raise ValueError(
                "paired training requires primevul_paired and the shared LOSVER protocol"
            )
        return train_losver(
            data_dir,
            output_dir,
            model_path,
            method,
            device=device,
            input_mode=paired_input,
            window_micro_batch=window_micro_batch,
            function_micro_batch=function_micro_batch,
            resume=resume,
            save_every=save_every,
            epochs_override=paired_epochs,
            train_limit=train_limit,
            valid_limit=valid_limit,
        )
    if (
        paired_input != "512"
        or window_micro_batch is not None
        or function_micro_batch is not None
        or paired_epochs is not None
    ):
        raise ValueError("paired training options apply only to paired methods")
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if (
        hard_selection not in ("hardest", "boundary")
        or vat_direction not in VAT_DIRECTIONS
    ):
        raise ValueError("unknown hard selection or VAT direction")
    if hard_selection != "hardest" and method not in ("hard_aug_bce", "hard_aug_vat"):
        raise ValueError("boundary selection requires a hard augmentation method")
    if vat_direction != "finite_difference" and method not in (
        "vat",
        "random_aug_vat",
        "hard_aug_vat",
    ):
        raise ValueError("logit direction requires a VAT method")
    config = dict(TRAINING if config is None else config)
    if (
        config["batch_size"] <= 0
        or config["gradient_accumulation"] <= 0
        or config["epochs"] <= 0
        or config["learning_rate"] <= 0
    ):
        raise ValueError("invalid training budget")
    run = Path(output_dir) / dataset / method
    run.mkdir(parents=True, exist_ok=True)
    target = run / "best.pt"
    if resume and (run / "training.metrics.json").exists():
        print(f"{run}: training complete; skipping", flush=True)
        return json.loads((run / "training.metrics.json").read_text())
    continuing = resume and (run / "last.pt").exists()
    if not continuing and (
        target.exists()
        or target.with_suffix(".training.jsonl").exists()
        or (run / "last.pt").exists()
    ):
        raise FileExistsError(
            f"{run}: training already started; use --resume with last.pt"
        )
    builder = InputBuilder(
        AutoTokenizer.from_pretrained(model_path),
        source_max_length=config["source_max_length"],
    )
    train_rows = balanced_subset(
        read_jsonl(Path(data_dir) / dataset / "train.jsonl"),
        train_limit,
        config["seed"],
    )
    valid_rows = balanced_subset(
        validation_rows(Path(data_dir) / dataset), valid_limit, config["seed"]
    )
    units = train_rows
    unit_batch_size = config["batch_size"]
    steps_per_epoch = math.ceil(
        len(train_rows) / (config["batch_size"] * config["gradient_accumulation"])
    )
    for split, rows in (("train", train_rows), ("valid", valid_rows)):
        if not rows or {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{split}: both classes required")
        expected_dataset = dataset
        if any(r["dataset"] != expected_dataset or r["split"] != split for r in rows):
            raise ValueError(f"training and selection require {dataset} {split}")
    source_views = method in (
        "random_aug_bce",
        "hard_aug_bce",
        "random_aug_vat",
        "hard_aug_vat",
    )
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
    model = CodeBERT(model_path, device, gradient_checkpointing=device.type == "cuda")
    seed_everything(seed)
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
            device=(
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        ),
        selection=f"{dataset} clean Valid: threshold MCC/F1/Accuracy/closest to 0.5; "
        "epoch MCC/F1/Accuracy/AUC",
        precision="FP32 parameters/optimizer; BF16 CUDA main forward; FP32 auxiliary VAT",
        parameter_count=sum(p.numel() for p in model.parameters()),
        checkpoint_interval=save_every,
        training_members=[r["sample_key"] for r in train_rows],
        validation_members=[r["sample_key"] for r in valid_rows],
        input_protocol=dict(
            logical_budget=config["source_max_length"],
            prefix_tokens=len(builder.source_prefix),
            source_budget=config["source_max_length"] - len(builder.source_prefix) - 2,
            window=512,
            stride=(512 - len(builder.source_prefix) - 2) // 2,
            pooling="mean over unique source positions; prefix/BOS/EOS averaged across windows",
        ),
        vat=dict(VAT_CONFIG, direction=vat_direction) if use_vat else None,
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    state = dict(
        epoch=1,
        next_batch=0,
        global_step=0,
        best_key=(float("-inf"),) * 4,
        selected_epoch=None,
        totals={},
        training_seconds=0.0,
    )
    if continuing:
        state, saved_rng = load_training_state(run, model, optimizer, metadata)
        restore_rng(saved_rng)
        print(
            f"{run}: resuming epoch {state['epoch']}, batch {state['next_batch']}, "
            f"step {state['global_step']}",
            flush=True,
        )
    if not continuing:
        save_training_state(run, model, optimizer, metadata, state)
    history = TrainingProgress(target, resume=continuing)
    write_json(run / "config.json", metadata)
    best_key = state["best_key"]
    global_step = state["global_step"]
    selected_epoch = state["selected_epoch"]
    batch_size = config["batch_size"]
    accumulation = config["gradient_accumulation"]
    for epoch in range(state["epoch"], config["epochs"] + 1):
        started = time.monotonic()
        previous_seconds = state["training_seconds"]
        totals = state["totals"]
        order = list(range(len(units)))
        random.Random(seed + epoch - 1).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = totals.get("running", 0.0)
        vat_total = totals.get("vat_total", 0.0)
        transformed = totals.get("transformed", 0)
        queries = totals.get("queries", 0)
        original_queries = totals.get("original_queries", 0)
        vat_eligible, vat_zero = totals.get("vat_eligible", 0), totals.get(
            "vat_zero", 0
        )
        vat_norm_min, vat_norm_max = totals.get("vat_norm_min"), totals.get(
            "vat_norm_max"
        )
        search_seconds = totals.get("search_seconds", 0.0)
        selection_reasons = Counter(totals.get("selection_reasons", {}))
        search_costs = Counter(totals.get("search_costs", {}))
        bce_costs = Counter(totals.get("bce_costs", {}))
        vat_costs = Counter(totals.get("vat_costs", {}))
        steps = totals.get("steps", 0)
        num_batches = math.ceil(len(order) / unit_batch_size)
        bar = training_bar(
            range(state["next_batch"] * unit_batch_size, len(order), unit_batch_size),
            desc=f'{dataset}/{method} epoch {epoch}/{config["epochs"]}',
        )
        for batch_index, start in enumerate(bar, start=state["next_batch"]):
            selected_units = [units[i] for i in order[start : start + unit_batch_size]]
            batch = selected_units
            group_start = batch_index // accumulation * accumulation
            group_end = min(group_start + accumulation, num_batches)
            group_samples = (
                min(group_end * unit_batch_size, len(order))
                - group_start * unit_batch_size
            )
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
                            logits = model(*inputs)
                            values.extend(logits.detach().cpu().tolist())
                            search_costs.update(
                                function_forwards=inputs[-1],
                                encoder_sequences=len(inputs[0]),
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
                    r["sample_key"]: (
                        [s["selected"]["raw_source"]] if s["selected"] else []
                    )
                    for r, s in zip(batch, selections)
                }
                transformed += sum(s["selected"] is not None for s in selections)
                queries += sum(s["candidate_queries"] for s in selections)
                original_queries += sum(s["original_queries"] for s in selections)
                selection_reasons.update(s["selection_reason"] for s in selections)
                with (run / "training.candidates.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
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
                    function_forwards=inputs[4],
                    encoder_sequences=len(inputs[0]),
                    encoder_token_slots=inputs[0].numel(),
                    encoder_active_tokens=int(inputs[1].sum()),
                )
                labels = logits.new_tensor([r["label"] for r in forwarded])
                terms = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                loss = (terms * terms.new_tensor(weights)).sum()
                total_loss = loss
                if not torch.isfinite(total_loss):
                    raise ValueError(
                        f"nonfinite classification loss: {method} epoch {epoch}, batch {batch_index}"
                    )
                (total_loss / group_samples).backward()
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
                vat_costs.update(
                    {
                        k: info[k]
                        for k in (
                            "function_forwards",
                            "encoder_sequences",
                            "encoder_token_slots",
                            "encoder_active_tokens",
                            "zero_initial_projection",
                        )
                        if info[k] is not None
                    }
                )
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
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                global_step += 1
                if (
                    global_step == 1
                    or global_step % 25 == 0
                    or batch_index + 1 == num_batches
                ):
                    history.add(
                        dict(
                            event="step",
                            epoch=epoch,
                            step=global_step,
                            loss=batch_loss / len(batch),
                            samples_seen=min(start + unit_batch_size, len(order)),
                        )
                    )
                    print(
                        f"{dataset}/{method} epoch {epoch} step {steps}/{steps_per_epoch}",
                        flush=True,
                    )
                if (
                    global_step == 1
                    or global_step % save_every == 0
                    or batch_index + 1 == num_batches
                ):
                    state.update(
                        next_batch=batch_index + 1,
                        global_step=global_step,
                        training_seconds=previous_seconds + time.monotonic() - started,
                        totals=dict(
                            running=running,
                            vat_total=vat_total,
                            transformed=transformed,
                            queries=queries,
                            original_queries=original_queries,
                            vat_eligible=vat_eligible,
                            vat_zero=vat_zero,
                            vat_norm_min=vat_norm_min,
                            vat_norm_max=vat_norm_max,
                            search_seconds=search_seconds,
                            selection_reasons=dict(selection_reasons),
                            search_costs=dict(search_costs),
                            bce_costs=dict(bce_costs),
                            vat_costs=dict(vat_costs),
                            steps=steps,
                        ),
                    )
                    save_training_state(run, model, optimizer, metadata, state)
        training_seconds = previous_seconds + time.monotonic() - started
        validation_started = time.monotonic()
        valid_logits = predict_logits(
            model, valid_rows, builder, device=device, batch_size=batch_size
        )
        probabilities = valid_logits.sigmoid()
        threshold, validation = select_threshold(
            [r["label"] for r in valid_rows], probabilities.tolist()
        )
        epoch_loss = (running + vat_total) / len(train_rows)
        history.add(
            dict(
                event="epoch",
                method=method,
                epoch=epoch,
                train_loss=running / len(train_rows),
                total_loss=epoch_loss,
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
                epoch_seconds=previous_seconds + time.monotonic() - started,
                training_seconds=training_seconds,
                validation_seconds=time.monotonic() - validation_started,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        )
        key = tuple(
            validation[k] if validation[k] is not None else float("-inf")
            for k in ("mcc", "f1", "accuracy", "auc")
        )
        if key > best_key:
            best_key = key
            selected_epoch = epoch
            checkpoint = dict(
                metadata,
                state_dict={
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                source_max_length=config["source_max_length"],
                decision_threshold=threshold,
                selected_epoch=epoch,
                validation=validation,
            )
            temporary = target.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(target)
            del checkpoint
        state.update(
            epoch=epoch + 1,
            next_batch=0,
            global_step=global_step,
            best_key=best_key,
            selected_epoch=selected_epoch,
            totals={},
            training_seconds=0.0,
        )
        save_training_state(run, model, optimizer, metadata, state)
    summary = dict(
        metadata,
        selected_epoch=selected_epoch,
        completed_epochs=config["epochs"],
        optimizer_steps=global_step,
    )
    write_json(run / "training.metrics.json", summary)
    return summary


LOSVER_METHODS = (
    "unixcoder",
    "losver_localizer",
    "losver",
    "losver_oracle",
    "diff_localizer",
    "repair_region_localizer",
    "neighbour_region_localizer",
    "diff_hard_weight",
    "repair_region_hard_weight",
    "diff_robust_fusion",
    "repair_region_robust_fusion",
    "neighbour_region_robust_fusion",
    "diff_neighbour_localizer",
    "diff_neighbour",
)
LOSVER_VARIANTS = {
    m: "localizer" if m.endswith("_localizer") else "predicted" for m in LOSVER_METHODS
}
LOSVER_VARIANTS.update(unixcoder="baseline", losver_oracle="oracle")
LOSVER_CONFIG = dict(
    seed=123456,
    batch_size=8,
    eval_batch_size=8,
    learning_rate=2e-5,
    adam_epsilon=1e-8,
    weight_decay=0.0,
    gradient_clip=1.0,
    warmup_fraction=0.1,
    block_size=512,
    epochs=dict(baseline=16, localizer=40, predicted=16, oracle=16),
    fp16=False,
    dropout_probability=0.0,
    class_weight=1.0,
    early_stopping=None,
)


class LOSVERAdamW(torch.optim.Optimizer):
    """The Transformers 4.19 AdamW rule, including epsilon before bias correction."""

    def __init__(self, params, lr=2e-5, eps=1e-8):
        super().__init__(params, dict(lr=lr, eps=eps, betas=(0.9, 0.999)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("sparse gradients are not supported")
                state = self.state[p]
                if not state:
                    state.update(
                        step=0,
                        exp_avg=torch.zeros_like(p),
                        exp_avg_sq=torch.zeros_like(p),
                    )
                state["step"] += 1
                b1, b2 = group["betas"]
                g = p.grad
                avg, sq = state["exp_avg"], state["exp_avg_sq"]
                avg.mul_(b1).add_(g, alpha=1 - b1)
                sq.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = sq.sqrt().add_(group["eps"])
                step_size = (
                    group["lr"]
                    * math.sqrt(1 - b2 ** state["step"])
                    / (1 - b1 ** state["step"])
                )
                p.addcdiv_(avg, denom, value=-step_size)


def paired_collate(features):
    """Native windows and unique cores; physical lines are shared across windows."""
    (
        ids,
        attention,
        masks,
        contexts,
        weights,
        owners,
        positions,
        window_lines,
        region_scores,
    ) = ([] for _ in range(9))
    line_offset = 0
    for owner, f in enumerate(features):
        lookup = {}
        source_scores = f.get("region_scores")
        visible_count = sum(d - c for a, b, c, d in f["windows"])
        for line, tokens in enumerate(f["line_token_indices"]):
            for token in tokens:
                lookup.setdefault(token - 1, []).append(line_offset + line)
        for left, right, core_start, core_end in f["windows"]:
            width = right - left
            ids.append(
                [
                    f["input_ids"][0],
                    *f["input_ids"][left + 1 : right + 1],
                    f["input_ids"][visible_count + 1],
                ]
                + [f["pad_token_id"]] * (510 - width)
            )
            attention.append([1] * (width + 2) + [0] * (510 - width))
            core = [False] * 512
            context = [False] * 512
            token_weights = [1.0] * 512
            token_scores = [0.0] * 512
            mapping = {}
            for slot, source in enumerate(range(left, right), 1):
                context[slot] = f["code_mask"][source + 1]
                if core_start <= source < core_end:
                    core[slot] = context[slot]
                    token_weights[slot] = f["token_weights"][source + 1]
                    token_scores[slot] = (
                        source_scores[source + 1] if source_scores is not None else 0.0
                    )
                    for line in lookup.get(source, []):
                        mapping.setdefault(line, []).append(slot)
            masks.append(core)
            contexts.append(context)
            weights.append(token_weights)
            region_scores.append(token_scores)
            owners.append(owner)
            positions.append([-1, *range(left, right), *([-1] * (511 - width))])
            window_lines.append(mapping)
        line_offset += len(f["line_numbers"])
    window_line_count = max(len(m) for m in window_lines)
    membership = torch.zeros(len(ids), window_line_count, 512, dtype=torch.bool)
    line_indices = torch.full((len(ids), window_line_count), -1, dtype=torch.long)
    for window, mapping in enumerate(window_lines):
        for slot, (line, tokens) in enumerate(mapping.items()):
            membership[window, slot, tokens] = True
            line_indices[window, slot] = line
    batch = dict(
        input_ids=torch.tensor(ids),
        attention_mask=torch.tensor(attention),
        code_mask=torch.tensor(masks, dtype=torch.bool),
        context_mask=torch.tensor(contexts, dtype=torch.bool),
        token_weights=torch.tensor(weights),
        region_scores=torch.tensor(region_scores),
        window_owners=torch.tensor(owners),
        source_positions=torch.tensor(positions),
        line_indices=line_indices,
        line_counts=torch.tensor([len(f["line_numbers"]) for f in features]),
    )
    count = max(len(f["line_numbers"]) for f in features)
    numbers = torch.full((len(features), count), -1, dtype=torch.long)
    local = isinstance(features[0]["labels"], list)
    labels = (
        torch.full((len(features), count), -1.0, dtype=torch.float)
        if local
        else torch.tensor([f["labels"] for f in features])
    )
    label_weights = torch.zeros(len(features), count)
    for i, f in enumerate(features):
        n = len(f["line_numbers"])
        numbers[i, :n] = torch.tensor(f["line_numbers"])
        if local:
            labels[i, :n] = torch.tensor(f["labels"])
            label_weights[i, :n] = torch.tensor(f.get("label_weights", [1.0] * n))
    return dict(
        batch,
        line_token_mask=membership,
        line_numbers=numbers,
        labels=labels,
        label_weights=label_weights,
    )


def paired_loader(
    rows, tokenizer, variant, *, train=False, batch_size=8, input_mode="512"
):
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    from .data import paired_features

    features = [
        paired_features(r, tokenizer, variant, input_mode=input_mode) for r in rows
    ]
    sampler = RandomSampler(features) if train else SequentialSampler(features)
    return DataLoader(
        features,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=paired_collate,
        num_workers=4,
        pin_memory=True,
    )


def paired_forward(model, batch, variant, device, *, with_labels=True):
    keys = ["input_ids", "attention_mask", "code_mask", "window_owners"]
    keys.extend(
        ["line_token_mask", "context_mask", "line_indices", "line_counts"]
        if variant == "localizer"
        else ["token_weights"]
    )
    if with_labels:
        keys.append("labels")
        if variant == "localizer" and getattr(model, "weak_supervision", False):
            keys.append("label_weights")
    if getattr(model, "pooling", None) == "robust_fusion":
        keys.append("region_scores")
    return model(**{key: batch[key].to(device) for key in keys})


def paired_rows(data_dir, output_dir, variant, split, *, method=None):
    """A predicted detector never opens the test oracle annotations."""
    root = Path(data_dir) / "primevul_paired"
    name = (
        (f"{split}.lines.jsonl" if split != "test" else "test.oracle.jsonl")
        if variant in ("localizer", "oracle")
        else f"{split}.jsonl"
    )
    if variant == "predicted":
        localizer = (
            (method.split("_hard_weight")[0].split("_robust_fusion")[0] + "_localizer")
            if method and method not in ("losver",)
            else "losver_localizer"
        )
        root = Path(output_dir) / "primevul_paired" / localizer
        name = f"{split}.predicted.jsonl"
    rows = read_jsonl(root / name)
    if variant == "predicted" and method and method != "losver":
        scores = {
            r["sample_key"]: r for r in read_jsonl(root / f"{split}.line_scores.jsonl")
        }
        for row in rows:
            score = scores[row["sample_key"]]
            ns, ps = score["line_numbers"], score["line_scores"]
            if (
                ns != row["line_numbers"]
                or len(ns) != len(ps)
                or [n for n, p in zip(ns, ps) if p > 0.5] != row["lines_predicted"]
            ):
                raise ValueError("predicted line scores and physical lines differ")
            row["line_scores"] = ps
    return rows


def train_losver(
    data_dir,
    output_dir,
    model_path,
    method,
    *,
    device="auto",
    config=None,
    input_mode="512",
    window_micro_batch=None,
    function_micro_batch=None,
    resume=False,
    save_every=100,
    epochs_override=None,
    train_limit=None,
    valid_limit=None,
):
    """Train corrected LOSVER on all unchanged official paired members."""
    from .model import losver_model
    from .evaluation import validate_paired, localize_paired
    import platform

    variant = LOSVER_VARIANTS[method]
    settings = dict(LOSVER_CONFIG if config is None else config)
    supervision, pooling = "diff", "hard_weight"
    neighbour = method in ("diff_neighbour_localizer", "diff_neighbour")
    weak = not neighbour and method not in (
        "unixcoder", "losver_localizer", "losver", "losver_oracle"
    )
    if neighbour:
        if input_mode != "512":
            raise ValueError("diff_neighbour currently requires 512 input")
        supervision = "diff_neighbour"
    if weak:
        if input_mode != "512":
            raise ValueError("repair-region methods currently require 512 input")
        supervision = (
            method.split("_localizer")[0]
            .split("_hard_weight")[0]
            .split("_robust_fusion")[0]
        )
        pooling = "robust_fusion" if method.endswith("robust_fusion") else "hard_weight"
    if epochs_override is not None:
        if epochs_override < 1:
            raise ValueError("epochs must be positive")
        settings = dict(
            settings, epochs=dict(settings["epochs"], **{variant: epochs_override})
        )
    if save_every < 1:
        raise ValueError("save_every must be positive")
    if input_mode not in ("512", "l1"):
        raise ValueError("paired input must be 512 or l1")
    window_micro_batch = (
        window_micro_batch
        if window_micro_batch is not None
        else (1 if input_mode == "l1" else 8)
    )
    function_micro_batch = (
        function_micro_batch
        if function_micro_batch is not None
        else (1 if input_mode == "l1" else settings["batch_size"])
    )
    if function_micro_batch < 1 or settings["batch_size"] % function_micro_batch:
        raise ValueError("function micro-batch must divide effective batch size")
    accumulation = settings["batch_size"] // function_micro_batch
    root = Path(output_dir) / "primevul_paired"
    run = root / method
    run.mkdir(parents=True, exist_ok=True)
    if resume and (run / "training.metrics.json").exists():
        print(f"{run}: training complete; skipping", flush=True)
        return json.loads((run / "training.metrics.json").read_text())
    continuing = resume and (run / "last.pt").exists()
    if not continuing and (
        (run / "best.training.jsonl").exists() or (run / "last.pt").exists()
    ):
        raise FileExistsError(
            f"{run}: training already started; use --resume with last.pt"
        )
    if neighbour and variant == "localizer":
        train_rows = read_jsonl(
            Path(output_dir) / "supervision" / "diff_neighbour" / "train.jsonl"
        )
        # Selection remains the ORIGINAL Diff F1. Proxy labels are diagnostics,
        # never a different selection target for this supervision-only ablation.
        valid_rows = paired_rows(data_dir, output_dir, variant, "valid")
    elif weak and variant == "localizer":
        train_rows = read_jsonl(
            Path(output_dir) / "supervision" / supervision / "train.jsonl"
        )
        valid_rows = read_jsonl(
            Path(output_dir) / "supervision" / supervision / "valid.jsonl"
        )
    else:
        train_rows = paired_rows(data_dir, output_dir, variant, "train", method=method)
        valid_rows = paired_rows(data_dir, output_dir, variant, "valid", method=method)

    def subset(rows, limit):
        if limit is None:
            return rows
        if limit < 2 or limit % 2:
            raise ValueError("paired limits are positive even function counts")
        keys = sorted({r["pair_key"] for r in rows})
        chosen = set(random.Random(42).sample(keys, min(limit // 2, len(keys))))
        return [r for r in rows if r["pair_key"] in chosen]

    train_rows = subset(train_rows, train_limit)
    valid_rows = subset(valid_rows, valid_limit)
    device = resolve_device(device)
    seed_everything(settings["seed"])
    np.random.seed(settings["seed"])
    torch.backends.cudnn.deterministic = True
    model, tokenizer = losver_model(
        model_path,
        variant,
        window_micro_batch=window_micro_batch,
        gradient_checkpointing=input_mode == "l1",
    )
    model.input_mode = input_mode
    model.weak_supervision = weak and variant == "localizer"
    if variant != "localizer":
        model.pooling = pooling
    if pooling == "robust_fusion" and input_mode != "512":
        raise ValueError("robust fusion requires 512 input")
    model.to(device)
    loader = paired_loader(
        train_rows,
        tokenizer,
        variant,
        train=True,
        batch_size=function_micro_batch,
        input_mode=input_mode,
    )
    valid_loader = paired_loader(
        valid_rows,
        tokenizer,
        variant,
        batch_size=settings["eval_batch_size"],
        input_mode=input_mode,
    )
    from .data import paired_statistics

    input_coverage = {}
    for split, data_loader, rows in (
        ("train", loader, train_rows),
        ("valid", valid_loader, valid_rows),
    ):
        annotations = read_jsonl(
            Path(data_dir) / "primevul_paired" / f"{split}.lines.jsonl"
        )
        selected = {r["sample_key"] for r in rows}
        annotations = [r for r in annotations if r["sample_key"] in selected]
        if [r["sample_key"] for r in annotations] != [r["sample_key"] for r in rows]:
            raise ValueError("input coverage annotations and training members differ")
        # Annotation statistics only; no Diff feature enters a normal detector.
        input_coverage[split] = paired_statistics(annotations, data_loader.dataset)
        if model.weak_supervision:
            fs = data_loader.dataset
            counts = Counter(int(y) for f in fs for y in f["labels"])
            input_coverage[split]["region_supervision"] = dict(
                visible_lines=sum(counts.values()),
                labels=dict(counts),
                unknown_fraction=counts[-1] / sum(counts.values()),
                positive_fraction=counts[1] / sum(counts.values()),
                background_fraction=counts[0] / sum(counts.values()),
                no_positive_functions=sum(1 not in f["labels"] for f in fs),
                no_effective_line_loss_functions=sum(
                    not any(f["label_weights"]) for f in fs
                ),
                effective_line_weight=sum(sum(f["label_weights"]) for f in fs),
            )
    epochs = settings["epochs"][variant]
    steps_per_epoch = math.ceil(len(train_rows) / settings["batch_size"])
    optimizer = LOSVERAdamW(
        model.parameters(), lr=settings["learning_rate"], eps=settings["adam_epsilon"]
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=epochs * steps_per_epoch * settings["warmup_fraction"],
        num_training_steps=epochs * steps_per_epoch,
    )
    metadata = dict(
        protocol="losver",
        line_mapping="source_offsets",
        input_mode=input_mode,
        window_micro_batch=window_micro_batch,
        function_micro_batch=function_micro_batch,
        gradient_accumulation=accumulation,
        gradient_checkpointing=input_mode == "l1",
        input_coverage=input_coverage,
        window_rule=(
            "single window up to 510 source tokens; otherwise 384 core + up to 63 context each side"
            if input_mode == "l1"
            else "first 510 source tokens"
        ),
        classifier_head="code-only pooling + shared Linear(hidden_size, 1)",
        dataset="primevul_paired",
        method=method,
        supervision=supervision if weak or neighbour else "original_diff",
        weak_supervision=model.weak_supervision,
        region_loss=(
            dict(
                labels="1=repair-related,0=weak background,-1=unknown",
                unknown="excluded from loss and denominator; retained in encoder and all detection BCE",
                alpha=0.5,
                gamma=1.5,
                pos_weight=1.0,
            )
            if weak
            else None
        ),
        pooling=pooling,
        fusion_beta=0.25 if pooling == "robust_fusion" else None,
        fusion_validation_grid=[0.0, 0.1, 0.25] if pooling == "robust_fusion" else None,
        variant=variant,
        model_path=str(model_path),
        training_config=settings,
        epochs=epochs,
        train_samples=len(train_rows),
        valid_samples=len(valid_rows),
        decision_threshold=0.5,
        comparison=">",
        upstream_commit="a158905cb45b4400f481f18eeb1cb4473ccfccdc",
        selection=(
            "Valid modified-line positive F1 rounded to 4 decimals"
            if variant == "localizer"
            else "Valid P-C rounded to 4 decimals"
        ),
        selection_ties="earlier epoch; strict improvement",
        precision=(
            "FP32 eager attention; window activation checkpointing; no AMP"
            if input_mode == "l1"
            else "FP32 eager attention; no AMP/checkpointing"
        ),
        localizer_attention="same 8-head attention within each window; core-token sums merged by physical line before classification; dropout=0",
        localizer_path=(
            str(
                root
                / ((supervision + "_localizer") if weak or neighbour else "losver_localizer")
                / "best-f1.pt"
            )
            if variant == "predicted"
            else None
        ),
        parameters=sum(p.numel() for p in model.parameters()),
        checkpoint_interval=save_every,
        training_members=[r["sample_key"] for r in train_rows],
        validation_members=[r["sample_key"] for r in valid_rows],
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=transformers.__version__,
            cuda=torch.version.cuda,
            device=(
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        ),
    )
    if pooling == "robust_fusion":
        metadata["validation_rng"] = (
            "beta grid preserves exactly one validation-loader RNG advance"
        )
    best = (
        {"f1": -1.0}
        if variant == "localizer"
        else {"accuracy": -1.0, "f1": -1.0, "p_c": -1.0}
    )
    chosen = {}
    global_step = 0
    state = dict(
        epoch=1,
        next_batch=0,
        global_step=0,
        best=best,
        chosen=chosen,
        running=0.0,
        training_seconds=0.0,
        epoch_rng=None,
    )
    if continuing:
        state, saved_rng = load_training_state(
            run,
            model,
            optimizer,
            metadata,
            scheduler=scheduler,
        )
        best, chosen, global_step = state["best"], state["chosen"], state["global_step"]
        restore_rng(saved_rng)
        print(
            f"{run}: resuming epoch {state['epoch']}, optimizer batch "
            f"{state['next_batch']}, step {global_step}",
            flush=True,
        )
    if not continuing:
        save_training_state(run, model, optimizer, metadata, state, scheduler=scheduler)
    history = TrainingProgress(run / "best.pt", resume=continuing)
    write_json(run / "config.json", metadata)
    model.zero_grad()
    for epoch in range(state["epoch"], epochs + 1):
        started = time.monotonic()
        previous_seconds = state["training_seconds"]
        running = state["running"]
        model.train()
        from itertools import islice

        if state["next_batch"]:
            # Recreate the original RandomSampler/worker seeds without querying
            # the model. Features and collation are deterministic. Restore the
            # saved model RNG after advancing to the next unprocessed function.
            restore_rng(state["epoch_rng"])
        else:
            state["epoch_rng"] = rng_state()
        iterator = iter(loader)
        if state["next_batch"]:
            for _ in range(min(state["next_batch"] * accumulation, len(loader))):
                next(iterator)
            restore_rng(saved_rng)
        for step in training_bar(
            range(state["next_batch"], steps_per_epoch),
            desc=f"{method} {epoch}/{epochs}",
        ):
            group = list(islice(iterator, accumulation))
            if not group:
                raise ValueError("training loader ended before the function budget")
            # Preserve the original line-average Focal objective and function-
            # average BCE, including the smaller final effective batch.
            sizes = [
                (
                    (
                        float((b["label_weights"] * (b["labels"] >= 0)).sum())
                        if model.weak_supervision
                        else int((b["labels"] >= 0).sum())
                    )
                    if variant == "localizer"
                    else len(b["labels"])
                )
                for b in group
            ]
            group_loss = 0.0
            for batch, size in zip(group, sizes):
                loss, _ = paired_forward(model, batch, variant, device)
                if not torch.isfinite(loss):
                    raise ValueError(f"{method}: nonfinite loss {epoch}:{step}")
                coefficient = size / sum(sizes) if sum(sizes) else 1 / len(sizes)
                (loss * coefficient).backward()
                group_loss += float(loss.detach()) * coefficient
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            running += group_loss
            if global_step == 1 or global_step % 100 == 0:
                item = dict(
                    event="step",
                    epoch=epoch,
                    step=global_step,
                    batch=step + 1,
                    loss=group_loss,
                    learning_rate=optimizer.param_groups[0]["lr"],
                )
                history.add(item)
                print(json.dumps(item), flush=True)
            if (
                global_step == 1
                or global_step % save_every == 0
                or step + 1 == steps_per_epoch
            ):
                state.update(
                    next_batch=step + 1,
                    global_step=global_step,
                    running=running,
                    training_seconds=previous_seconds + time.monotonic() - started,
                )
                save_training_state(
                    run, model, optimizer, metadata, state, scheduler=scheduler
                )
        train_seconds = previous_seconds + time.monotonic() - started
        grid = []
        before_validation = rng_state() if pooling == "robust_fusion" else None
        after_validation = None
        for beta in ([0.0, 0.1, 0.25] if pooling == "robust_fusion" else [None]):
            if grid and before_validation is not None:
                restore_rng(before_validation)
            if beta is not None:
                model.fusion_beta = beta
            value, pred = validate_paired(
                model, valid_rows, tokenizer, variant, device, loader=valid_loader
            )
            grid.append((beta, value, pred))
            if before_validation is not None and after_validation is None:
                after_validation = rng_state()
        if after_validation is not None:
            restore_rng(after_validation)
        selected_beta, validation, predictions = (
            max(grid, key=lambda t: (round(t[1]["p_c"], 4), -(t[0] or 0)))
            if pooling == "robust_fusion"
            else grid[0]
        )
        if pooling == "robust_fusion":
            model.fusion_beta = 0.25
        event = dict(
            event="epoch",
            variant=method,
            epoch=epoch,
            train_loss=running / steps_per_epoch,
            optimizer_steps=steps_per_epoch,
            global_step=global_step,
            validation_threshold=0.5,
            validation=validation,
            fusion_validation=(
                [dict(beta=b, metrics=v) for b, v, p in grid]
                if pooling == "robust_fusion"
                else None
            ),
            training_seconds=train_seconds,
            epoch_seconds=previous_seconds + time.monotonic() - started,
        )
        history.add(event)
        print(json.dumps(event), flush=True)
        for key in best:
            score = round(validation[key], 4)
            if score > best[key]:
                best[key] = score
                chosen[key] = epoch
                cp = dict(
                    metadata,
                    state_dict={
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    selection_metric=key,
                    selected_epoch=epoch,
                    validation=validation,
                    selected_fusion_beta=selected_beta,
                )
                temporary = run / f"best-{key}.pt.tmp"
                torch.save(cp, temporary)
                temporary.replace(run / f"best-{key}.pt")
                write_json(
                    run / f"best-{key}.json",
                    dict(
                        epoch=epoch,
                        validation=validation,
                        selected_fusion_beta=selected_beta,
                    ),
                )
                write_jsonl(run / f"valid.best-{key}.predictions.jsonl", predictions)
                del cp
        state.update(
            epoch=epoch + 1,
            next_batch=0,
            global_step=global_step,
            best=best,
            chosen=chosen,
            running=0.0,
            training_seconds=0.0,
            epoch_rng=None,
        )
        save_training_state(run, model, optimizer, metadata, state, scheduler=scheduler)
    required = "f1" if variant == "localizer" else "p_c"
    if required not in chosen:
        raise RuntimeError(f"{method}: no finite validation checkpoint was selected")
    summary = dict(
        metadata,
        completed_epochs=epochs,
        optimizer_steps=global_step,
        selected_epochs=chosen,
    )
    if variant == "localizer":
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pilot = train_limit is not None or valid_limit is not None
        for split in (
            ("valid", "train") if weak or neighbour or pilot else ("test", "valid", "train")
        ):
            localize_paired(
                data_dir,
                output_dir,
                split,
                device=device,
                method=method,
                sample_keys=(
                    {
                        r["sample_key"]
                        for r in (valid_rows if split == "valid" else train_rows)
                    }
                    if weak or neighbour or pilot
                    else None
                ),
            )
    write_json(run / "training.metrics.json", summary)
    return summary
