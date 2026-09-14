from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .dataset import DATASET_SCHEMA_VERSION
from .semantics import (
    FEATURE_TO_GROUP,
    SEMANTIC_GROUPS,
    VULNERABILITY_FEATURES,
    render_semantic_items,
    validate_semantic_groups,
)


MODEL_VARIANTS = (
    "baseline",
    "raw_cpg",
    "semantic_concat",
    "semantic_fusion",
    "full",
)
_SEQUENCE_VARIANTS = {"baseline", "raw_cpg", "semantic_concat"}
_FUSION_VARIANTS = {"semantic_fusion", "full"}
_CHECKPOINT_VERSION = 4


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


@dataclass(frozen=True)
class FeatureMetrics:
    precision: float
    recall: float
    f1: float

    def as_json(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            if record.get("schema_version") != DATASET_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{line_number}: dataset schema must be {DATASET_SCHEMA_VERSION}; rebuild the dataset"
                )
            key_value = record.get("sample_key")
            key = str(key_value) if key_value is not None else ""
            if not key:
                raise ValueError(f"{path}:{line_number}: sample_key is required")
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate sample_key {key}")
            if record.get("label") not in {0, 1}:
                raise ValueError(f"{path}:{line_number}: label must be 0 or 1")
            if not isinstance(record.get("raw_source"), str):
                raise ValueError(f"{path}:{line_number}: raw_source is required")
            if not isinstance(record.get("cpg_relations"), str):
                raise ValueError(f"{path}:{line_number}: cpg_relations is required")
            if not isinstance(record.get("vulnerability_semantics"), str):
                raise ValueError(f"{path}:{line_number}: vulnerability_semantics is required")
            semantic_items = record.get("semantic_items")
            if not isinstance(semantic_items, list) or not all(
                isinstance(item, dict)
                and isinstance(item.get("category"), str)
                and isinstance(item.get("kind"), str)
                and isinstance(item.get("detail"), str)
                for item in semantic_items
            ):
                raise ValueError(f"{path}:{line_number}: semantic_items is malformed")
            features = record.get("vulnerability_features")
            if not isinstance(features, list) or not all(
                isinstance(feature, str) and feature in VULNERABILITY_FEATURES
                for feature in features
            ):
                raise ValueError(f"{path}:{line_number}: vulnerability_features is malformed")
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


def _validate_variant(variant: str, excluded_groups: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {', '.join(MODEL_VARIANTS)}")
    excluded = validate_semantic_groups(excluded_groups)
    if excluded and variant in {"baseline", "raw_cpg"}:
        raise ValueError("semantic group ablation is only valid for semantic variants")
    return variant, excluded


class InputBuilder:
    """Keep the function token budget identical while varying only auxiliary context."""

    def __init__(self, tokenizer, *, source_max_length: int, context_max_length: int) -> None:
        if source_max_length <= 0 or context_max_length <= 0:
            raise ValueError("source_max_length and context_max_length must be positive")
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.context_max_length = context_max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.source_prefix = self._encode(
            "TASK: classify whether this C/C++ function is vulnerable.\nFUNCTION:\n"
        )
        self.raw_cpg_prefix = self._encode("\nCODE_PROPERTY_GRAPH_RELATIONS:\n")
        self.semantic_concat_prefix = self._encode(
            "\nVULNERABILITY_RELATED_PROGRAM_SEMANTICS:\n"
        )
        self.semantic_only_prefix = self._encode(
            "VULNERABILITY_RELATED_PROGRAM_SEMANTICS:\n"
        )

    def _encode(self, text: str, max_length: int | None = None) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=max_length is not None,
            max_length=max_length,
        )
        return list(encoded["input_ids"])

    def _with_eos(self, ids: list[int]) -> list[int]:
        if self.eos_token_id is None:
            return ids
        return [*ids, self.eos_token_id]

    def source_ids(self, record: dict[str, object]) -> list[int]:
        source = self._encode(str(record["raw_source"]), self.source_max_length)
        return self._with_eos([*self.source_prefix, *source])

    def semantic_text(
        self,
        record: dict[str, object],
        *,
        excluded_groups: tuple[str, ...],
    ) -> str:
        items = record["semantic_items"]
        assert isinstance(items, list)
        return render_semantic_items(items, excluded_groups=excluded_groups)

    def semantic_ids(
        self,
        record: dict[str, object],
        *,
        excluded_groups: tuple[str, ...],
    ) -> list[int]:
        context = self._encode(
            self.semantic_text(record, excluded_groups=excluded_groups),
            self.context_max_length,
        )
        return self._with_eos([*self.semantic_only_prefix, *context])

    def sequence_ids(
        self,
        record: dict[str, object],
        *,
        variant: str,
        excluded_groups: tuple[str, ...],
    ) -> list[int]:
        source = self._encode(str(record["raw_source"]), self.source_max_length)
        ids = [*self.source_prefix, *source]
        if variant == "raw_cpg":
            context = self._encode(str(record["cpg_relations"]), self.context_max_length)
            ids.extend(self.raw_cpg_prefix)
            ids.extend(context)
        elif variant == "semantic_concat":
            context = self._encode(
                self.semantic_text(record, excluded_groups=excluded_groups),
                self.context_max_length,
            )
            ids.extend(self.semantic_concat_prefix)
            ids.extend(context)
        elif variant != "baseline":
            raise ValueError(f"{variant} is not a sequence-input variant")
        return self._with_eos(ids)

    def _pad(
        self,
        sequences: list[list[int]],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("cannot create an empty batch")
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :length] = 1
        return input_ids.to(device), attention_mask.to(device)

    def sequence_batch(
        self,
        records: list[dict[str, object]],
        *,
        variant: str,
        excluded_groups: tuple[str, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pad(
            [
                self.sequence_ids(
                    record,
                    variant=variant,
                    excluded_groups=excluded_groups,
                )
                for record in records
            ],
            device=device,
        )

    def fusion_batch(
        self,
        records: list[dict[str, object]],
        *,
        excluded_groups: tuple[str, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_ids, source_mask = self._pad(
            [self.source_ids(record) for record in records],
            device=device,
        )
        semantic_ids, semantic_mask = self._pad(
            [
                self.semantic_ids(record, excluded_groups=excluded_groups)
                for record in records
            ],
            device=device,
        )
        return source_ids, source_mask, semantic_ids, semantic_mask


def _build_lora_encoder(
    model_path: str,
    *,
    device: torch.device,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: tuple[str, ...],
    gradient_checkpointing: bool,
):
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
    encoder = get_peft_model(base, config)
    return encoder, int(base.config.hidden_size)


def _masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _last_valid_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1) - 1
    if bool((lengths < 0).any()):
        raise ValueError("attention_mask contains an empty sequence")
    rows = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[rows, lengths]


class SequenceVulnerabilityClassifier(nn.Module):
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
        self.encoder, hidden_size = _build_lora_encoder(
            model_path,
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.task_modules = nn.ModuleDict({"classifier": nn.Linear(hidden_size, 1)})
        self.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        pooled = _last_valid_token(hidden, attention_mask)
        logits = self.task_modules["classifier"](pooled.float()).squeeze(-1)
        return logits, None


class SemanticFusionClassifier(nn.Module):
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
        fusion_dim: int,
        fusion_heads: int,
        feature_supervision: bool,
    ) -> None:
        super().__init__()
        if fusion_dim <= 0 or fusion_heads <= 0 or fusion_dim % fusion_heads != 0:
            raise ValueError("fusion_dim must be positive and divisible by fusion_heads")
        self.encoder, hidden_size = _build_lora_encoder(
            model_path,
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
        )
        modules: dict[str, nn.Module] = {
            "source_projection": nn.Linear(hidden_size, fusion_dim),
            "semantic_projection": nn.Linear(hidden_size, fusion_dim),
            "cross_attention": nn.MultiheadAttention(
                fusion_dim,
                fusion_heads,
                batch_first=True,
            ),
            "layer_norm": nn.LayerNorm(fusion_dim),
            "classifier": nn.Linear(fusion_dim, 1),
        }
        if feature_supervision:
            modules["feature_head"] = nn.Linear(hidden_size, len(VULNERABILITY_FEATURES))
        self.task_modules = nn.ModuleDict(modules)
        self.feature_supervision = feature_supervision
        self.to(device)

    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        semantic_ids: torch.Tensor,
        semantic_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        source_hidden = self.encoder(
            input_ids=source_ids,
            attention_mask=source_mask,
            use_cache=False,
        ).last_hidden_state
        semantic_hidden = self.encoder(
            input_ids=semantic_ids,
            attention_mask=semantic_mask,
            use_cache=False,
        ).last_hidden_state

        source_projected = self.task_modules["source_projection"](source_hidden.float())
        semantic_projected = self.task_modules["semantic_projection"](semantic_hidden.float())
        attended, _ = self.task_modules["cross_attention"](
            query=source_projected,
            key=semantic_projected,
            value=semantic_projected,
            key_padding_mask=~semantic_mask.bool(),
            need_weights=False,
        )
        fused = self.task_modules["layer_norm"](source_projected + attended)
        pooled = _masked_mean(fused, source_mask)
        logits = self.task_modules["classifier"](pooled).squeeze(-1)

        feature_logits = None
        if self.feature_supervision:
            source_pooled = _masked_mean(source_hidden, source_mask)
            feature_logits = self.task_modules["feature_head"](source_pooled.float())
        return logits, feature_logits


def _build_model(
    variant: str,
    model_path: str,
    *,
    device: torch.device,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: tuple[str, ...],
    gradient_checkpointing: bool,
    fusion_dim: int,
    fusion_heads: int,
):
    if variant in _SEQUENCE_VARIANTS:
        return SequenceVulnerabilityClassifier(
            model_path,
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
        )
    if variant in _FUSION_VARIANTS:
        return SemanticFusionClassifier(
            model_path,
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
            fusion_dim=fusion_dim,
            fusion_heads=fusion_heads,
            feature_supervision=variant == "full",
        )
    raise ValueError(f"unsupported model variant {variant}")


def _labels(records: list[dict[str, object]], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [float(record["label"]) for record in records],
        dtype=torch.float32,
        device=device,
    )


def _feature_targets(
    records: list[dict[str, object]],
    *,
    excluded_groups: tuple[str, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded = set(excluded_groups)
    active = torch.tensor(
        [FEATURE_TO_GROUP[feature] not in excluded for feature in VULNERABILITY_FEATURES],
        dtype=torch.float32,
        device=device,
    )
    if int(active.sum().item()) == 0:
        raise ValueError("feature supervision has no active vulnerability features")
    rows: list[list[float]] = []
    for record in records:
        present = set(record["vulnerability_features"])
        rows.append([1.0 if feature in present else 0.0 for feature in VULNERABILITY_FEATURES])
    targets = torch.tensor(rows, dtype=torch.float32, device=device)
    mask = active.unsqueeze(0).expand_as(targets)
    return targets, mask


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


def _classification_metrics(
    records: list[dict[str, object]],
    probabilities: torch.Tensor,
) -> Metrics:
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


def _feature_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> FeatureMetrics:
    predictions = probabilities >= 0.5
    truth = targets >= 0.5
    active = mask >= 0.5
    tp = int((predictions & truth & active).sum())
    fp = int((predictions & ~truth & active).sum())
    fn = int((~predictions & truth & active).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return FeatureMetrics(precision, recall, f1)


def _forward_batch(
    model,
    records: list[dict[str, object]],
    input_builder: InputBuilder,
    *,
    variant: str,
    excluded_groups: tuple[str, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if variant in _SEQUENCE_VARIANTS:
        input_ids, attention_mask = input_builder.sequence_batch(
            records,
            variant=variant,
            excluded_groups=excluded_groups,
            device=device,
        )
        return model(input_ids, attention_mask)
    source_ids, source_mask, semantic_ids, semantic_mask = input_builder.fusion_batch(
        records,
        excluded_groups=excluded_groups,
        device=device,
    )
    return model(source_ids, source_mask, semantic_ids, semantic_mask)


@torch.no_grad()
def _predict(
    model,
    records: list[dict[str, object]],
    input_builder: InputBuilder,
    *,
    variant: str,
    excluded_groups: tuple[str, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    model.eval()
    probabilities: list[torch.Tensor] = []
    feature_probabilities: list[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        logits, feature_logits = _forward_batch(
            model,
            batch_records,
            input_builder,
            variant=variant,
            excluded_groups=excluded_groups,
            device=device,
        )
        probabilities.append(torch.sigmoid(logits).cpu())
        if feature_logits is not None:
            feature_probabilities.append(torch.sigmoid(feature_logits).cpu())
    features = torch.cat(feature_probabilities) if feature_probabilities else None
    return torch.cat(probabilities) if probabilities else torch.empty(0), features


def _cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _selection_score(metrics: Metrics) -> float:
    return metrics.auc if metrics.auc is not None else metrics.f1


def _train_variant(
    train_records: list[dict[str, object]],
    valid_records: list[dict[str, object]],
    input_builder: InputBuilder,
    *,
    variant: str,
    excluded_groups: tuple[str, ...],
    model_path: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    weight_decay: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: tuple[str, ...],
    fusion_dim: int,
    fusion_heads: int,
    feature_loss_weight: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    _seed_everything(seed)
    model = _build_model(
        variant,
        model_path,
        device=device,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        gradient_checkpointing=device.type == "cuda",
        fusion_dim=fusion_dim,
        fusion_heads=fusion_heads,
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    classification_loss_fn = nn.BCEWithLogitsLoss()

    best_score = float("-inf")
    best_adapter: dict[str, torch.Tensor] | None = None
    best_task: dict[str, torch.Tensor] | None = None
    best_validation: Metrics | None = None
    best_feature_validation: FeatureMetrics | None = None

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_records)))
        random.Random(seed + epoch).shuffle(order)
        running_loss = 0.0
        running_classification_loss = 0.0
        running_feature_loss = 0.0
        optimizer_steps = 0
        pending_steps = 0
        num_batches = math.ceil(len(order) / batch_size)

        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            indices = order[start : start + batch_size]
            batch_records = [train_records[index] for index in indices]
            labels = _labels(batch_records, device)
            logits, feature_logits = _forward_batch(
                model,
                batch_records,
                input_builder,
                variant=variant,
                excluded_groups=excluded_groups,
                device=device,
            )
            classification_loss = classification_loss_fn(logits, labels)
            feature_loss = torch.zeros((), dtype=torch.float32, device=device)
            if variant == "full":
                if feature_logits is None:
                    raise RuntimeError("full variant did not produce vulnerability-feature logits")
                targets, feature_mask = _feature_targets(
                    batch_records,
                    excluded_groups=excluded_groups,
                    device=device,
                )
                per_feature = F.binary_cross_entropy_with_logits(
                    feature_logits,
                    targets,
                    reduction="none",
                )
                feature_loss = (per_feature * feature_mask).sum() / feature_mask.sum().clamp_min(1.0)
            loss = classification_loss + feature_loss_weight * feature_loss

            group_start = (batch_index // gradient_accumulation) * gradient_accumulation
            group_size = min(gradient_accumulation, num_batches - group_start)
            (loss / group_size).backward()
            running_loss += float(loss.detach().cpu())
            running_classification_loss += float(classification_loss.detach().cpu())
            running_feature_loss += float(feature_loss.detach().cpu())
            pending_steps += 1

            is_last_batch = batch_index + 1 == num_batches
            if pending_steps == gradient_accumulation or is_last_batch:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending_steps = 0

        validation_metrics = None
        feature_validation = None
        if valid_records:
            probabilities, feature_probabilities = _predict(
                model,
                valid_records,
                input_builder,
                variant=variant,
                excluded_groups=excluded_groups,
                batch_size=batch_size,
                device=device,
            )
            validation_metrics = _classification_metrics(valid_records, probabilities)
            if variant == "full":
                if feature_probabilities is None:
                    raise RuntimeError("full variant did not return vulnerability-feature predictions")
                targets, feature_mask = _feature_targets(
                    valid_records,
                    excluded_groups=excluded_groups,
                    device=torch.device("cpu"),
                )
                feature_validation = _feature_metrics(
                    feature_probabilities,
                    targets,
                    feature_mask,
                )
            score = _selection_score(validation_metrics)
        else:
            score = float(epoch)

        print(
            json.dumps(
                {
                    "variant": variant,
                    "excluded_groups": list(excluded_groups),
                    "epoch": epoch + 1,
                    "train_loss": running_loss / max(1, num_batches),
                    "classification_loss": running_classification_loss / max(1, num_batches),
                    "feature_loss": (
                        running_feature_loss / max(1, num_batches)
                        if variant == "full"
                        else None
                    ),
                    "optimizer_steps": optimizer_steps,
                    "validation": validation_metrics.as_json() if validation_metrics else None,
                    "feature_validation": (
                        feature_validation.as_json() if feature_validation else None
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_adapter = _cpu_state(get_peft_model_state_dict(model.encoder))
            best_task = _cpu_state(model.task_modules.state_dict())
            best_validation = validation_metrics
            best_feature_validation = feature_validation

    if best_adapter is None or best_task is None:
        raise RuntimeError("training did not produce a checkpoint")

    result = {
        "adapter_state": best_adapter,
        "task_state": best_task,
        "validation": best_validation.as_json() if best_validation else {},
        "feature_validation": (
            best_feature_validation.as_json() if best_feature_validation else None
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def train_model(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    variant: str,
    model_path: str,
    source_max_length: int = 1536,
    context_max_length: int = 384,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    fusion_dim: int = 256,
    fusion_heads: int = 8,
    feature_loss_weight: float = 0.2,
    excluded_groups: tuple[str, ...] = (),
    seed: int = 42,
    device: str = "auto",
) -> dict[str, object]:
    variant, excluded_groups = _validate_variant(variant, excluded_groups)
    if batch_size <= 0 or gradient_accumulation <= 0 or epochs <= 0:
        raise ValueError("batch_size, gradient_accumulation, and epochs must be positive")
    if lora_r <= 0 or lora_alpha <= 0:
        raise ValueError("lora_r and lora_alpha must be positive")
    if feature_loss_weight < 0:
        raise ValueError("feature_loss_weight must be non-negative")
    if fusion_dim <= 0 or fusion_heads <= 0 or fusion_dim % fusion_heads != 0:
        raise ValueError("fusion_dim must be positive and divisible by fusion_heads")

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
        context_max_length=context_max_length,
    )
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")

    trained = _train_variant(
        train_records,
        valid_records,
        input_builder,
        variant=variant,
        excluded_groups=excluded_groups,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        fusion_dim=fusion_dim,
        fusion_heads=fusion_heads,
        feature_loss_weight=feature_loss_weight,
        seed=seed,
        device=resolved_device,
    )

    checkpoint = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "variant": variant,
        "excluded_groups": excluded_groups,
        "model_path": model_path,
        "source_max_length": source_max_length,
        "context_max_length": context_max_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "fusion_dim": fusion_dim,
        "fusion_heads": fusion_heads,
        "feature_loss_weight": feature_loss_weight,
        "adapter_state": trained["adapter_state"],
        "task_state": trained["task_state"],
        "validation": trained["validation"],
        "feature_validation": trained["feature_validation"],
        "trainable_parameters": trained["trainable_parameters"],
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    print(
        json.dumps(
            {
                "checkpoint": str(target),
                "variant": variant,
                "excluded_groups": list(excluded_groups),
                "validation": checkpoint["validation"],
                "feature_validation": checkpoint["feature_validation"],
                "trainable_parameters": checkpoint["trainable_parameters"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return checkpoint


def _load_model(checkpoint: dict[str, object], *, device: torch.device):
    variant = str(checkpoint["variant"])
    target_modules = tuple(str(value) for value in checkpoint["target_modules"])
    model = _build_model(
        variant,
        str(checkpoint["model_path"]),
        device=device,
        lora_r=int(checkpoint["lora_r"]),
        lora_alpha=int(checkpoint["lora_alpha"]),
        lora_dropout=float(checkpoint["lora_dropout"]),
        target_modules=target_modules,
        gradient_checkpointing=False,
        fusion_dim=int(checkpoint["fusion_dim"]),
        fusion_heads=int(checkpoint["fusion_heads"]),
    )
    set_peft_model_state_dict(model.encoder, checkpoint["adapter_state"])
    model.task_modules.load_state_dict(checkpoint["task_state"])
    model.eval()
    return model


def evaluate_model(
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
    if checkpoint.get("checkpoint_version") != _CHECKPOINT_VERSION:
        raise ValueError("checkpoint is incompatible with the current semantic-learning model")
    variant = str(checkpoint["variant"])
    excluded_groups = validate_semantic_groups(
        tuple(str(value) for value in checkpoint.get("excluded_groups", ()))
    )
    _validate_variant(variant, excluded_groups)

    records = [record for record in _read_jsonl(dataset_path) if _record_split(record) == split]
    if not records:
        raise ValueError(f"{split} split is empty")

    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint["model_path"]), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    input_builder = InputBuilder(
        tokenizer,
        source_max_length=int(checkpoint["source_max_length"]),
        context_max_length=int(checkpoint["context_max_length"]),
    )

    model = _load_model(checkpoint, device=resolved_device)
    probabilities, feature_probabilities = _predict(
        model,
        records,
        input_builder,
        variant=variant,
        excluded_groups=excluded_groups,
        batch_size=batch_size,
        device=resolved_device,
    )
    metrics = _classification_metrics(records, probabilities)
    feature_metrics = None
    if variant == "full":
        if feature_probabilities is None:
            raise RuntimeError("full variant did not return vulnerability-feature predictions")
        targets, feature_mask = _feature_targets(
            records,
            excluded_groups=excluded_groups,
            device=torch.device("cpu"),
        )
        feature_metrics = _feature_metrics(
            feature_probabilities,
            targets,
            feature_mask,
        )

    result = {
        "split": split,
        "variant": variant,
        "excluded_groups": list(excluded_groups),
        "samples": len(records),
        "positive": sum(int(record["label"]) == 1 for record in records),
        "negative": sum(int(record["label"]) == 0 for record in records),
        "metrics": metrics.as_json(),
        "feature_metrics": feature_metrics.as_json() if feature_metrics else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del model
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return result