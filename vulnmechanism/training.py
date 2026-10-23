"""CodeBERT BCE, dynamic source augmentation and VAT training."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
import time

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
):
    if method in LOSVER_METHODS:
        if (
            dataset != "primevul_paired"
            or train_limit is not None
            or valid_limit is not None
            or config is not None
            or hard_selection != "hardest"
            or vat_direction != "finite_difference"
        ):
            raise ValueError(
                "LOSVER requires whole official paired data and the unchanged official protocol"
            )
        return train_losver(data_dir, output_dir, model_path, method, device=device)
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
    if target.exists() or target.with_suffix(".training.jsonl").exists():
        raise FileExistsError(
            f"{run}: training already started; refusing to reuse fitted weights"
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
            device=(
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        ),
        selection=f"{dataset} clean Valid: threshold MCC/F1/Accuracy/closest to 0.5; "
        "epoch MCC/F1/Accuracy/AUC",
        precision="FP32 parameters/optimizer; BF16 CUDA main forward; FP32 auxiliary VAT",
        parameter_count=sum(p.numel() for p in model.parameters()),
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
    write_json(run / "config.json", metadata)
    best_key = (float("-inf"),) * 4
    global_step = 0
    selected_epoch = None
    batch_size = config["batch_size"]
    accumulation = config["gradient_accumulation"]
    for epoch in range(1, config["epochs"] + 1):
        started = time.monotonic()
        if epoch == 1:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
        order = list(range(len(units)))
        random.Random(seed + epoch - 1).shuffle(order)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, vat_total, transformed, queries, original_queries = 0.0, 0.0, 0, 0, 0
        vat_eligible, vat_zero, vat_norm_min, vat_norm_max = 0, 0, None, None
        search_seconds = 0.0
        selection_reasons = Counter()
        search_costs = Counter()
        bce_costs = Counter()
        vat_costs = Counter()
        steps = 0
        num_batches = math.ceil(len(order) / unit_batch_size)
        bar = training_bar(
            range(0, len(order), unit_batch_size),
            desc=f'{dataset}/{method} epoch {epoch}/{config["epochs"]}',
        )
        for batch_index, start in enumerate(bar):
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
        training_seconds = time.monotonic() - started
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
                epoch_seconds=time.monotonic() - started,
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
    summary = dict(
        metadata,
        selected_epoch=selected_epoch,
        completed_epochs=config["epochs"],
        optimizer_steps=global_step,
    )
    write_json(run / "training.metrics.json", summary)
    return summary


LOSVER_METHODS = ("unixcoder", "losver_localizer", "losver", "losver_oracle")
LOSVER_VARIANTS = dict(
    zip(LOSVER_METHODS, ("baseline", "localizer", "predicted", "oracle"))
)
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


def paired_loader(rows, tokenizer, variant, *, train=False, batch_size=8):
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    from .data import paired_features

    features = [paired_features(r, tokenizer, variant) for r in rows]
    dataset = [
        (
            torch.tensor(f["input_ids"]),
            torch.tensor(f["labels"]),
            torch.tensor(f.get("attention_mask", [0] * 512)),
            torch.tensor((f.get("line_indices", [])[:512] + [-1] * 512)[:512]),
        )
        for f in features
    ]
    sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
    return DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, num_workers=4, pin_memory=True
    )


def paired_forward(model, batch, variant, device, *, with_labels=True):
    ids, y, mask, lines = batch
    kwargs = dict(input_ids=ids.to(device))
    if with_labels:
        kwargs["labels"] = y.to(device)
    if variant in ("predicted", "oracle"):
        kwargs.update(
            attention_mask=mask.to(device),
            line_indices=[[x for x in r if x >= 0] for r in lines.tolist()],
        )
    return model(**kwargs)


def paired_rows(data_dir, output_dir, variant, split):
    """A predicted detector never opens the test oracle annotations."""
    root = Path(data_dir) / "primevul_paired"
    name = (
        (f"{split}.lines.jsonl" if split != "test" else "test.oracle.jsonl")
        if variant in ("localizer", "oracle")
        else f"{split}.jsonl"
    )
    if variant == "predicted":
        root = Path(output_dir) / "primevul_paired/losver_localizer"
        name = f"{split}.predicted.jsonl"
    return read_jsonl(root / name)


def train_losver(
    data_dir, output_dir, model_path, method, *, device="auto", config=None
):
    """Run the original training and Valid selection on all official paired members."""
    from .model import losver_model
    from .evaluation import validate_paired, localize_paired
    import platform

    variant = LOSVER_VARIANTS[method]
    settings = dict(LOSVER_CONFIG if config is None else config)
    root = Path(output_dir) / "primevul_paired"
    run = root / method
    run.mkdir(parents=True, exist_ok=True)
    if (run / "best.training.jsonl").exists():
        raise FileExistsError(f"{run}: training already started")
    train_rows = paired_rows(data_dir, output_dir, variant, "train")
    valid_rows = paired_rows(data_dir, output_dir, variant, "valid")
    import numpy as np

    device = resolve_device(device)
    seed_everything(settings["seed"])
    np.random.seed(settings["seed"])
    torch.backends.cudnn.deterministic = True
    model, tokenizer = losver_model(model_path, variant)
    model.to(device)
    loader = paired_loader(
        train_rows, tokenizer, variant, train=True, batch_size=settings["batch_size"]
    )
    valid_loader = paired_loader(
        valid_rows, tokenizer, variant, batch_size=settings["eval_batch_size"]
    )
    epochs = settings["epochs"][variant]
    optimizer = LOSVERAdamW(
        model.parameters(), lr=settings["learning_rate"], eps=settings["adam_epsilon"]
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=epochs * len(loader) * settings["warmup_fraction"],
        num_training_steps=epochs * len(loader),
    )
    metadata = dict(
        protocol="losver",
        dataset="primevul_paired",
        method=method,
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
            "Valid weighted line F1 rounded to 4 decimals"
            if variant == "localizer"
            else "Valid P-C rounded to 4 decimals"
        ),
        selection_ties="earlier epoch; strict improvement",
        precision="FP32 eager attention; no AMP/checkpointing",
        localizer_attention="same 8-head cross-attention, queries batched per function; dropout=0",
        localizer_path=(
            str(root / "losver_localizer/best-f1.pt")
            if variant == "predicted"
            else None
        ),
        parameters=sum(p.numel() for p in model.parameters()),
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
    write_json(run / "config.json", metadata)
    history = TrainingProgress(run / "best.pt")
    best = (
        {"f1": 0.0}
        if variant == "localizer"
        else {"accuracy": 0.0, "f1": 0.0, "p_c": 0.0}
    )
    chosen = {}
    global_step = 0
    model.zero_grad()
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        running = 0.0
        model.train()
        for step, batch in enumerate(
            training_bar(loader, desc=f"{method} {epoch}/{epochs}")
        ):
            loss, _ = paired_forward(model, batch, variant, device)
            if not torch.isfinite(loss):
                raise ValueError(f"{method}: nonfinite loss {epoch}:{step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            running += float(loss.detach())
            if global_step == 1 or global_step % 100 == 0:
                item = dict(
                    event="step",
                    epoch=epoch,
                    step=global_step,
                    batch=step + 1,
                    loss=float(loss.detach()),
                    learning_rate=optimizer.param_groups[0]["lr"],
                )
                history.add(item)
                print(json.dumps(item), flush=True)
        train_seconds = time.monotonic() - started
        validation, predictions = validate_paired(
            model, valid_rows, tokenizer, variant, device, loader=valid_loader
        )
        event = dict(
            event="epoch",
            variant=method,
            epoch=epoch,
            train_loss=running / len(loader),
            optimizer_steps=len(loader),
            global_step=global_step,
            validation_threshold=0.5,
            validation=validation,
            training_seconds=train_seconds,
            epoch_seconds=time.monotonic() - started,
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
                )
                torch.save(cp, run / f"best-{key}.pt")
                write_json(
                    run / f"best-{key}.json", dict(epoch=epoch, validation=validation)
                )
                write_jsonl(run / f"valid.best-{key}.predictions.jsonl", predictions)
                del cp
    required = "f1" if variant == "localizer" else "p_c"
    if required not in chosen:
        raise RuntimeError(
            f"{method}: no checkpoint improved above the official zero initialization"
        )
    summary = dict(
        metadata,
        completed_epochs=epochs,
        optimizer_steps=global_step,
        selected_epochs=chosen,
    )
    write_json(run / "training.metrics.json", summary)
    if variant == "localizer":
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for split in ("test", "valid", "train"):
            localize_paired(data_dir, output_dir, split, device=device)
    return summary
