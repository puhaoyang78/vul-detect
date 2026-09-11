from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch import nn
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    auc: float | None

    def as_json(self) -> dict[str, float | None]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mcc": self.mcc,
            "auc": self.auc,
        }


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            key_value = record.get("sample_key")
            key = str(key_value) if key_value is not None else ""
            if not key:
                raise ValueError(f"{path}:{line_number}: sample_key is required")
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate sample_key {key}")
            label = record.get("label")
            if label not in {0, 1}:
                raise ValueError(f"{path}:{line_number}: label must be 0 or 1")
            if not isinstance(record.get("raw_source"), str):
                raise ValueError(f"{path}:{line_number}: raw_source is required")
            if not isinstance(record.get("semantic_facts"), str):
                raise ValueError(
                    f"{path}:{line_number}: semantic_facts is required; rebuild the dataset with the semantic extractor"
                )
            seen.add(key)
            records.append(record)
    return records


def _split_for_key(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "valid"
    return "test"


def _record_split(record: dict[str, object]) -> str:
    split = str(record.get("split") or "").lower()
    if split in {"train", "valid", "validation", "test"}:
        return "valid" if split == "validation" else split
    return _split_for_key(str(record["sample_key"]))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


class InputBuilder:
    """Use identical source tokens in both variants and a separate budget for semantic facts."""

    def __init__(self, tokenizer, *, source_max_length: int, semantic_max_length: int) -> None:
        if source_max_length <= 0 or semantic_max_length <= 0:
            raise ValueError("source_max_length and semantic_max_length must be positive")
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.semantic_max_length = semantic_max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.prefix = self._encode("TASK: classify whether this C/C++ function is vulnerable.\nFUNCTION:\n")
        self.semantic_prefix = self._encode("\nVULNERABILITY_SEMANTICS_FROM_CPG:\n")

    def _encode(self, text: str, max_length: int | None = None) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=max_length is not None,
            max_length=max_length,
        )
        return list(encoded["input_ids"])

    def record_ids(self, record: dict[str, object], *, include_semantics: bool) -> list[int]:
        source_ids = self._encode(str(record["raw_source"]), self.source_max_length)
        ids = [*self.prefix, *source_ids]
        if include_semantics:
            semantic_ids = self._encode(str(record["semantic_facts"]), self.semantic_max_length)
            ids.extend(self.semantic_prefix)
            ids.extend(semantic_ids)
        if self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        return ids

    def batch(
        self,
        records: list[dict[str, object]],
        *,
        include_semantics: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [self.record_ids(record, include_semantics=include_semantics) for record in records]
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :length] = 1
        return input_ids.to(device), attention_mask.to(device)


class LoRABinaryClassifier(nn.Module):
    def __init__(
        self,
        model_path: str,
        *,
        device: torch.device,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: tuple[str, ...],
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        base = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        base.config.use_cache = False
        if gradient_checkpointing:
            base.gradient_checkpointing_enable()
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
        config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules),
            bias="none",
        )
        self.encoder = get_peft_model(base, config)
        self.hidden_size = int(base.config.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, 1)
        self.to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(pooled.float()).squeeze(-1)


def _labels(records: list[dict[str, object]], device: torch.device) -> torch.Tensor:
    return torch.tensor([float(record["label"]) for record in records], dtype=torch.float32, device=device)


def _binary_auc(truth: torch.Tensor, scores: torch.Tensor) -> float | None:
    positives = int(truth.sum().item())
    negatives = int((~truth).sum().item())
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(
        ((float(score), bool(label)) for score, label in zip(scores.tolist(), truth.tolist())),
        key=lambda item: item[0],
    )
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (rank + (rank + end - index - 1)) / 2.0
        rank_sum += average_rank * sum(1 for _, label in pairs[index:end] if label)
        rank += end - index
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _classification_metrics(records: list[dict[str, object]], probabilities: torch.Tensor) -> Metrics:
    predictions = probabilities >= 0.5
    truth = torch.tensor([int(record["label"]) for record in records], dtype=torch.bool)
    tp = int((predictions & truth).sum())
    fp = int((predictions & ~truth).sum())
    tn = int((~predictions & ~truth).sum())
    fn = int((~predictions & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(records) if records else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    auc = _binary_auc(truth, probabilities)
    return Metrics(accuracy, precision, recall, f1, mcc, auc)


@torch.no_grad()
def _predict(
    model: LoRABinaryClassifier,
    records: list[dict[str, object]],
    input_builder: InputBuilder,
    *,
    include_semantics: bool,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    probabilities: list[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        input_ids, attention_mask = input_builder.batch(
            batch_records,
            include_semantics=include_semantics,
            device=device,
        )
        probabilities.append(torch.sigmoid(model(input_ids, attention_mask)).cpu())
    return torch.cat(probabilities) if probabilities else torch.empty(0)


def _cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _selection_score(metrics: Metrics) -> float:
    return metrics.auc if metrics.auc is not None else metrics.f1


def _train_variant(
    train_records: list[dict[str, object]],
    valid_records: list[dict[str, object]],
    input_builder: InputBuilder,
    *,
    model_path: str,
    include_semantics: bool,
    epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    weight_decay: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: tuple[str, ...],
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    _seed_everything(seed)
    model = LoRABinaryClassifier(
        model_path,
        device=device,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        gradient_checkpointing=device.type == "cuda",
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_score = float("-inf")
    best_adapter: dict[str, torch.Tensor] | None = None
    best_head: dict[str, torch.Tensor] | None = None
    best_validation: Metrics | None = None

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_records)))
        random.Random(seed + epoch).shuffle(order)
        running_loss = 0.0
        optimizer_steps = 0
        pending_steps = 0
        num_batches = math.ceil(len(order) / batch_size)

        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            indices = order[start : start + batch_size]
            batch_records = [train_records[index] for index in indices]
            input_ids, attention_mask = input_builder.batch(
                batch_records,
                include_semantics=include_semantics,
                device=device,
            )
            labels = _labels(batch_records, device)
            loss = loss_fn(model(input_ids, attention_mask), labels)
            group_start = (batch_index // gradient_accumulation) * gradient_accumulation
            group_size = min(gradient_accumulation, num_batches - group_start)
            (loss / group_size).backward()
            running_loss += float(loss.detach().cpu())
            pending_steps += 1

            is_last_batch = batch_index + 1 == num_batches
            if pending_steps == gradient_accumulation or is_last_batch:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending_steps = 0

        validation_metrics = None
        if valid_records:
            probabilities = _predict(
                model,
                valid_records,
                input_builder,
                include_semantics=include_semantics,
                batch_size=batch_size,
                device=device,
            )
            validation_metrics = _classification_metrics(valid_records, probabilities)
            score = _selection_score(validation_metrics)
        else:
            score = float(epoch)

        print(
            json.dumps(
                {
                    "variant": "semantic" if include_semantics else "baseline",
                    "epoch": epoch + 1,
                    "train_loss": running_loss / max(1, num_batches),
                    "optimizer_steps": optimizer_steps,
                    "validation": validation_metrics.as_json() if validation_metrics else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_adapter = _cpu_state(get_peft_model_state_dict(model.encoder))
            best_head = _cpu_state(model.classifier.state_dict())
            best_validation = validation_metrics

    if best_adapter is None or best_head is None:
        raise RuntimeError("training did not produce a checkpoint")

    result = {
        "adapter_state": best_adapter,
        "head_state": best_head,
        "validation": best_validation.as_json() if best_validation else {},
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _load_variant(checkpoint: dict[str, object], *, variant: str, device: torch.device) -> LoRABinaryClassifier:
    target_modules = tuple(str(value) for value in checkpoint["target_modules"])
    model = LoRABinaryClassifier(
        str(checkpoint["model_path"]),
        device=device,
        lora_r=int(checkpoint["lora_r"]),
        lora_alpha=int(checkpoint["lora_alpha"]),
        lora_dropout=float(checkpoint["lora_dropout"]),
        target_modules=target_modules,
        gradient_checkpointing=False,
    )
    set_peft_model_state_dict(model.encoder, checkpoint[f"{variant}_adapter_state"])
    model.classifier.load_state_dict(checkpoint[f"{variant}_head_state"])
    model.eval()
    return model


def train_models(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str,
    source_max_length: int = 1536,
    semantic_max_length: int = 384,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, object]:
    if batch_size <= 0 or gradient_accumulation <= 0 or epochs <= 0:
        raise ValueError("batch_size, gradient_accumulation, and epochs must be positive")
    if lora_r <= 0 or lora_alpha <= 0:
        raise ValueError("lora_r and lora_alpha must be positive")

    records = _read_jsonl(dataset_path)
    train_records = [record for record in records if _record_split(record) == "train"]
    valid_records = [record for record in records if _record_split(record) == "valid"]
    if not train_records:
        raise ValueError("training split is empty")
    if {int(record["label"]) for record in train_records} != {0, 1}:
        raise ValueError("training split must contain both labels 0 and 1")

    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    input_builder = InputBuilder(
        tokenizer,
        source_max_length=source_max_length,
        semantic_max_length=semantic_max_length,
    )
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

    baseline = _train_variant(
        train_records,
        valid_records,
        input_builder,
        model_path=model_path,
        include_semantics=False,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        seed=seed,
        device=resolved_device,
    )
    semantic = _train_variant(
        train_records,
        valid_records,
        input_builder,
        model_path=model_path,
        include_semantics=True,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        seed=seed,
        device=resolved_device,
    )

    checkpoint = {
        "checkpoint_version": 2,
        "model_path": model_path,
        "source_max_length": source_max_length,
        "semantic_max_length": semantic_max_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "baseline_adapter_state": baseline["adapter_state"],
        "baseline_head_state": baseline["head_state"],
        "semantic_adapter_state": semantic["adapter_state"],
        "semantic_head_state": semantic["head_state"],
        "validation": {"baseline": baseline["validation"], "semantic": semantic["validation"]},
        "trainable_parameters": {
            "baseline": baseline["trainable_parameters"],
            "semantic": semantic["trainable_parameters"],
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    print(
        json.dumps(
            {
                "checkpoint": str(target),
                "validation": checkpoint["validation"],
                "trainable_parameters": checkpoint["trainable_parameters"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return checkpoint


def evaluate_models(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    batch_size: int = 1,
    device: str = "auto",
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != 2:
        raise ValueError("checkpoint is incompatible with the current LoRA semantic model")
    records = [record for record in _read_jsonl(dataset_path) if _record_split(record) == split]
    if not records:
        raise ValueError(f"{split} split is empty")

    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint["model_path"]), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_builder = InputBuilder(
        tokenizer,
        source_max_length=int(checkpoint["source_max_length"]),
        semantic_max_length=int(checkpoint["semantic_max_length"]),
    )

    baseline = _load_variant(checkpoint, variant="baseline", device=resolved_device)
    baseline_probability = _predict(
        baseline,
        records,
        input_builder,
        include_semantics=False,
        batch_size=batch_size,
        device=resolved_device,
    )
    baseline_metrics = _classification_metrics(records, baseline_probability)
    del baseline
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()

    semantic = _load_variant(checkpoint, variant="semantic", device=resolved_device)
    semantic_probability = _predict(
        semantic,
        records,
        input_builder,
        include_semantics=True,
        batch_size=batch_size,
        device=resolved_device,
    )
    semantic_metrics = _classification_metrics(records, semantic_probability)
    del semantic
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "split": split,
        "samples": len(records),
        "positive": sum(int(record["label"]) == 1 for record in records),
        "negative": sum(int(record["label"]) == 0 for record in records),
        "baseline": baseline_metrics.as_json(),
        "semantic": semantic_metrics.as_json(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
