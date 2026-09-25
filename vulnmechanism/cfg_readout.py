"""Frozen-C representation cache and three residual readout screens."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from .cfg_data import (AttributeVocabulary, atomic_json, cohort_hash, digest, identity,
                       output_lock, read_jsonl, read_records)
from .cfg_metrics import select_threshold
from .cfg_experiment import (CHECKPOINT_SCHEMA, READOUT_VARIANTS, _base_module,
                             _graph_inputs, _prediction_outputs, _prepare, _save_torch,
                             _tokenizer, file_sha256)
from .cfg_network import build_model

READOUT_SCHEMA = 1
WIDTH = 32


class ResidualReadout(nn.Module):
    """Add one learned correction to a fixed C logit."""
    def __init__(self, mode: str, source_dim: int, graph_dim: int):
        super().__init__()
        if mode not in READOUT_VARIANTS or source_dim <= 0 or graph_dim <= 0:
            raise ValueError("invalid readout configuration")
        self.mode = mode
        if mode == "linear":
            self.output = nn.Linear(source_dim + graph_dim, 1)
        elif mode == "mlp":
            self.source_graph = nn.Linear(source_dim + graph_dim, WIDTH, bias=False)
            self.output = nn.Linear(WIDTH, 1, bias=False)
        else:
            self.source = nn.Linear(source_dim, WIDTH, bias=False)
            self.graph = nn.Linear(graph_dim, WIDTH, bias=False)
            self.output = nn.Linear(WIDTH, 1, bias=False)
        # Leave internal projections at their normal initialization.
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, source: torch.Tensor, graph: torch.Tensor,
                c_logit: torch.Tensor) -> torch.Tensor:
        if self.mode == "linear":
            features = torch.cat((source, graph), dim=-1)
        elif self.mode == "mlp":
            features = F.gelu(self.source_graph(torch.cat((source, graph), dim=-1)))
        else:
            features = self.source(source) * self.graph(graph)
        return c_logit + self.output(features).squeeze(-1)


def _c_run(c_run_dir: str | Path) -> tuple[Path, dict, Path, str]:
    root = Path(c_run_dir)
    config = json.loads((root / "config.json").read_text())
    if config["seed"] != 42 or config["source_max_length"] != 2048:
        raise ValueError("readout screening requires the original seed-42, 2048-token C run")
    checkpoint_path = root / "cfg" / "best.pt"
    completed = json.loads((root / "cfg" / "complete.json").read_text())
    checkpoint_hash = file_sha256(checkpoint_path)
    if (completed.get("config_sha256") != digest(dict(config, variant="cfg")) or
            completed.get("checkpoint_sha256") != checkpoint_hash or
            completed.get("validation_predictions_sha256") !=
            file_sha256(root / "cfg" / "valid.predictions.jsonl")):
        raise ValueError("saved C checkpoint or valid predictions differ from the completed run")
    return root, config, checkpoint_path, checkpoint_hash


def _reference_valid(root: Path, valid_samples: list[dict], threshold: float) -> list[dict]:
    reference = read_jsonl(root / "cfg" / "valid.predictions.jsonl")
    if (len(reference) != len(valid_samples) or
            any(any(prediction.get(key) != value for key, value in sample.items())
                for sample, prediction in zip(valid_samples, reference)) or
            any(row.get("threshold") != threshold for row in reference)):
        raise ValueError("saved C validation predictions do not match cache order or threshold")
    return reference


def extract_representations(c_run_dir: str | Path, output: str | Path, *,
                            device: str = "auto", base=None) -> dict:
    """One frozen C forward per train/valid batch; never forward test rows."""
    root, config, checkpoint_path, checkpoint_hash = _c_run(c_run_dir)
    output = Path(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("representation cache must be outside the original C run")
    if output.exists():
        raise FileExistsError(f"representation cache exists: {output}")
    rows, views = _prepare(config["dataset"], config["graphs"], config["source_dataset"])
    if (cohort_hash(rows) != config["cohort_sha256"] or
            file_sha256(config["graphs"]) != config["graph_file_sha256"]):
        raise ValueError("C dataset or graph sidecar differs from its recorded run")
    selected = [row for row in rows if row["split"] in {"train", "valid"}]
    if {row["split"] for row in selected} != {"train", "valid"}:
        raise ValueError("C run requires nonempty train and valid splits")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("cfg_ablation_version") != CHECKPOINT_SCHEMA or
            checkpoint.get("model_config") != dict(config, variant="cfg") or
            checkpoint.get("feature_schema") != config["feature_schema"]):
        raise ValueError("incompatible C checkpoint")
    vocabulary = AttributeVocabulary(checkpoint["vocabulary"])
    if digest(vocabulary.values) != config["vocabulary_sha256"]:
        raise ValueError("C checkpoint vocabulary differs from its run")
    base = _base_module() if base is None else base
    resolved_device = base._resolve_device(device)
    model = build_model(base, checkpoint["model_config"], vocabulary.sizes(),
                        resolved_device, training=False)
    base.set_peft_model_state_dict(model.encoder, checkpoint["adapter_state"])
    model.task_modules.load_state_dict(checkpoint["task_state"])
    model.requires_grad_(False)
    model.eval()
    builder = _tokenizer(base, config)
    encoded = {row["sample_key"]: vocabulary.encode(views[row["sample_key"]]) for row in selected}
    source_vectors, graph_vectors, c_logits = [], [], []
    batch_size = config["batch_size"]
    with torch.no_grad():
        for start in tqdm(range(0, len(selected), batch_size), desc="frozen C train/valid"):
            batch = selected[start:start + batch_size]
            inputs = _graph_inputs(model, batch, builder, views, encoded, resolved_device)
            source, graph, logit = model.representations(*inputs)
            if not all(torch.isfinite(value).all() for value in (source, graph, logit)):
                raise ValueError("non-finite frozen C representation")
            source_vectors.append(source.float().cpu())
            graph_vectors.append(graph.float().cpu())
            c_logits.append(logit.float().cpu())
    samples = [identity(row) for row in selected]
    source_tensor = torch.cat(source_vectors)
    graph_tensor = torch.cat(graph_vectors)
    logit_tensor = torch.cat(c_logits)
    valid_indices = [i for i, sample in enumerate(samples) if sample["split"] == "valid"]
    reference = _reference_valid(root, [samples[i] for i in valid_indices],
                                 float(checkpoint["decision_threshold"]))
    reference_scores = torch.tensor([row["score"] for row in reference], dtype=torch.float32)
    difference = (logit_tensor[valid_indices].sigmoid() - reference_scores).abs().max().item()
    if difference > 1e-4:
        raise ValueError(f"frozen C valid logits disagree with saved C predictions: max difference {difference:g}")
    cache = dict(schema_version=READOUT_SCHEMA, c_run_dir=str(root.resolve()),
                 c_checkpoint_sha256=checkpoint_hash, cohort_sha256=config["cohort_sha256"],
                 graph_file_sha256=config["graph_file_sha256"],
                 source_max_length=config["source_max_length"],
                 samples=samples, source=source_tensor, graph=graph_tensor,
                 c_logit=logit_tensor, valid_reference_max_score_difference=difference)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_torch(output, cache)
    return dict(output=str(output), counts=dict(Counter(sample["split"] for sample in samples)),
                source_dim=source_tensor.shape[1], graph_dim=graph_tensor.shape[1],
                valid_reference_max_score_difference=difference)


def _load_cache(cache_path: str | Path, c_run_dir: str | Path) -> tuple[dict, list[dict], list[dict], dict]:
    root, config, _, checkpoint_hash = _c_run(c_run_dir)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if (cache.get("schema_version") != READOUT_SCHEMA or
            cache.get("c_run_dir") != str(root.resolve()) or
            cache.get("c_checkpoint_sha256") != checkpoint_hash or
            cache.get("cohort_sha256") != config["cohort_sha256"] or
            cache.get("graph_file_sha256") != config["graph_file_sha256"] or
            cache.get("source_max_length") != config["source_max_length"]):
        raise ValueError("representation cache does not match the saved C run")
    rows = read_records(config["dataset"], config["source_dataset"])
    selected = [row for row in rows if row["split"] in {"train", "valid"}]
    samples = [identity(row) for row in selected]
    if cache.get("samples") != samples:
        raise ValueError("representation cache sample keys, labels, splits, or source order differ")
    n = len(samples)
    source, graph, logits = (cache.get(name) for name in ("source", "graph", "c_logit"))
    if (not all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and
                value.device.type == "cpu" and not value.requires_grad for value in (source, graph, logits)) or
            source.ndim != 2 or graph.ndim != 2 or logits.ndim != 1 or
            source.shape[0] != n or graph.shape[0] != n or logits.shape[0] != n or
            source.shape[1] <= 0 or graph.shape[1] <= 0 or
            not all(torch.isfinite(value).all() for value in (source, graph, logits))):
        raise ValueError("invalid or non-finite cached C representations")
    valid_indices = [i for i, sample in enumerate(samples) if sample["split"] == "valid"]
    completed = json.loads((root / "cfg" / "complete.json").read_text())
    reference = _reference_valid(root, [samples[i] for i in valid_indices],
                                 float(completed["validation"]["threshold"]))
    reference_scores = torch.tensor([row["score"] for row in reference], dtype=torch.float32)
    if (logits[valid_indices].sigmoid() - reference_scores).abs().max().item() > 1e-4:
        raise ValueError("cached C logits disagree with saved C valid predictions")
    return cache, selected, reference, config


def _valid_scores(model: ResidualReadout, cache: dict, indices: list[int],
                  device: torch.device, batch_size: int) -> list[float]:
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            logits = model(cache["source"][batch].to(device), cache["graph"][batch].to(device),
                           cache["c_logit"][batch].to(device))
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite readout prediction")
            scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return scores


def train_readouts(cache_path: str | Path, c_run_dir: str | Path, output_dir: str | Path, *,
                   modes: tuple[str, ...] = READOUT_VARIANTS, device: str = "cpu",
                   resume: bool = False) -> dict:
    if not modes or len(set(modes)) != len(modes) or any(mode not in READOUT_VARIANTS for mode in modes):
        raise ValueError("choose distinct linear, mlp, or interaction readouts")
    cache, rows, reference, c_config = _load_cache(cache_path, c_run_dir)
    c_root = Path(c_run_dir)
    root = Path(output_dir)
    if root.resolve().is_relative_to(c_root.resolve()):
        raise ValueError("readout output must be outside the original C run")
    train_indices = [i for i, sample in enumerate(cache["samples"]) if sample["split"] == "train"]
    valid_indices = [i for i, sample in enumerate(cache["samples"]) if sample["split"] == "valid"]
    valid_labels = [cache["samples"][i]["label"] for i in valid_indices]
    if any({cache["samples"][i]["label"] for i in group} != {0, 1}
           for group in (train_indices, valid_indices)):
        raise ValueError("readout train and valid splits must both contain two labels")
    settings = dict(schema_version=READOUT_SCHEMA, c_run_dir=str(c_root.resolve()),
                    cache=str(Path(cache_path).resolve()),
                    c_checkpoint_sha256=cache["c_checkpoint_sha256"],
                    seed=c_config["seed"], epochs=c_config["epochs"],
                    batch_size=c_config["batch_size"],
                    gradient_accumulation=c_config["gradient_accumulation"],
                    learning_rate=c_config["graph_learning_rate"],
                    weight_decay=c_config["weight_decay"],
                    hidden_size=WIDTH, interaction_rank=WIDTH,
                    selection="validation_mcc")
    source_dim, graph_dim = cache["source"].shape[1], cache["graph"].shape[1]
    settings["trainable_parameters"] = {
        mode: sum(p.numel() for p in ResidualReadout(mode, source_dim, graph_dim).parameters())
        for mode in READOUT_VARIANTS}
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    with output_lock(root / "readout_experiment"):
        config_path, settings_path = root / "config.json", root / "readout_config.json"
        if config_path.exists() or settings_path.exists():
            if (not resume or not config_path.exists() or not settings_path.exists() or
                    json.loads(config_path.read_text()) != c_config or
                    json.loads(settings_path.read_text()) != settings):
                raise ValueError("readout run already exists or uses different settings; choose a new directory")
        else:
            if any((root / mode).exists() for mode in READOUT_VARIANTS):
                raise FileExistsError("readout variant directory exists without run metadata")
            atomic_json(config_path, c_config)
            atomic_json(settings_path, settings)
        for mode in modes:
            folder = root / mode
            completed = folder / "complete.json"
            variant_config = dict(settings, mode=mode)
            if completed.exists() and resume:
                state = json.loads(completed.read_text())
                if (state.get("config_sha256") != digest(variant_config) or
                        state.get("checkpoint_sha256") != file_sha256(folder / "best.pt") or
                        state.get("validation_predictions_sha256") !=
                        file_sha256(folder / "valid.predictions.jsonl")):
                    raise ValueError(f"{mode}: completed readout changed")
                results[mode] = state["validation"]
                continue
            if folder.exists():
                raise FileExistsError(f"{folder} already exists; choose a new run directory")
            folder.mkdir()
            atomic_json(folder / "config.json", variant_config)
            random.seed(settings["seed"])
            torch.manual_seed(settings["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(settings["seed"])
            target_device = torch.device(device)
            model = ResidualReadout(mode, source_dim, graph_dim).to(target_device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                          weight_decay=settings["weight_decay"])
            labels = torch.tensor([sample["label"] for sample in cache["samples"]],
                                  dtype=torch.float32, device=target_device)
            source = cache["source"].to(target_device)
            graph = cache["graph"].to(target_device)
            c_logits = cache["c_logit"].to(target_device)
            best_key = (-float("inf"),) * 4
            best_scores = best_threshold = None
            with (folder / "history.jsonl").open("x", encoding="utf-8") as history:
                for epoch in range(settings["epochs"]):
                    model.train()
                    order = list(train_indices)
                    random.Random(settings["seed"] + epoch).shuffle(order)
                    optimizer.zero_grad(set_to_none=True)
                    total_loss = 0.0
                    steps = 0
                    batch_size = settings["batch_size"]
                    accumulation = settings["gradient_accumulation"]
                    batches = math.ceil(len(order) / batch_size)
                    for batch_index, start in enumerate(range(0, len(order), batch_size)):
                        indices = order[start:start + batch_size]
                        logits = model(source[indices], graph[indices], c_logits[indices])
                        loss = F.binary_cross_entropy_with_logits(logits.float(), labels[indices])
                        if not torch.isfinite(loss):
                            raise ValueError(f"{mode}: non-finite training loss")
                        group_start = (batch_index // accumulation) * accumulation
                        group_end = min(group_start + accumulation, batches)
                        group_samples = min(group_end * batch_size, len(order)) - group_start * batch_size
                        (loss * len(indices) / group_samples).backward()
                        total_loss += float(loss.detach().cpu()) * len(indices)
                        if ((batch_index + 1) % accumulation == 0 or batch_index + 1 == batches):
                            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            if not torch.isfinite(norm):
                                raise ValueError(f"{mode}: non-finite gradient norm")
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            steps += 1
                    scores = _valid_scores(model, cache, valid_indices, target_device, batch_size)
                    threshold, validation = select_threshold(valid_labels, scores)
                    event = dict(event="epoch", mode=mode, epoch=epoch + 1,
                                 train_loss=total_loss / len(train_indices), optimizer_steps=steps,
                                 validation_threshold=threshold, validation=validation)
                    history.write(json.dumps(event, allow_nan=False) + "\n")
                    history.flush()
                    print(json.dumps(event, allow_nan=False), flush=True)
                    key = (validation["mcc"], validation["f1"], validation["accuracy"],
                           validation["auc"] if validation["auc"] is not None else -float("inf"))
                    if key > best_key:
                        best_key, best_scores, best_threshold = key, list(scores), threshold
                        _save_torch(folder / "best.pt", dict(schema_version=READOUT_SCHEMA,
                            config=variant_config, state_dict={k: v.detach().cpu().clone()
                                                               for k, v in model.state_dict().items()},
                            selected_epoch=epoch + 1, decision_threshold=threshold,
                            validation=validation, trainable_parameters=settings["trainable_parameters"][mode]))
            valid_rows = [rows[i] for i in valid_indices]
            summary = _prediction_outputs(folder, "valid", valid_rows, best_scores, best_threshold,
                                          None, checkpoint_hash=file_sha256(folder / "best.pt"),
                                          source_reference=reference)
            atomic_json(completed, dict(config_sha256=digest(variant_config),
                                        checkpoint_sha256=file_sha256(folder / "best.pt"),
                                        validation_predictions_sha256=file_sha256(
                                            folder / "valid.predictions.jsonl"),
                                        validation=summary["selected"]))
            results[mode] = summary["selected"]
            del model, optimizer
    return dict(metrics=results, trainable_parameters=settings["trainable_parameters"])
