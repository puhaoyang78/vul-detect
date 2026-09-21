from __future__ import annotations

import json
import math
import random
import sys
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
from tqdm.auto import tqdm

from .benchmark_view import record_dataset, record_split
from .dataset import DATASET_SCHEMA_VERSION
from .progress import TrainingProgress
from .semantics import (
    MECHANISM_GROUPS,
    render_mechanism_items,
    validate_mechanism_groups,
)


MODEL_VARIANTS = (
    "baseline",
    "source_attention",
    "source_bidirectional",
    "source_bidirectional_attention",
    "raw_cpg",
    "mechanism_concat",
    "mechanism_fusion",
)
_SOURCE_VARIANTS = {"baseline", "source_attention", "source_bidirectional", "source_bidirectional_attention"}
_SEQUENCE_VARIANTS = _SOURCE_VARIANTS | {"raw_cpg", "mechanism_concat"}
_FUSION_VARIANTS = {"mechanism_fusion"}
_CHECKPOINT_VERSION = 8


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
            key = record.get("sample_key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path}:{line_number}: sample_key is required")
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate sample_key {key}")
            record_dataset(record)
            record_split(record)
            if type(record.get("label")) is not int or record.get("label") not in {0, 1}:
                raise ValueError(f"{path}:{line_number}: label must be integer 0 or 1")
            if not isinstance(record.get("raw_source"), str) or not record["raw_source"]:
                raise ValueError(f"{path}:{line_number}: raw_source is required")
            if not isinstance(record.get("cpg_relations"), str) or not record["cpg_relations"]:
                raise ValueError(f"{path}:{line_number}: cpg_relations is required")
            items = record.get("mechanism_items")
            if not isinstance(items, list) or not all(
                isinstance(item, dict)
                and isinstance(item.get("category"), str)
                and isinstance(item.get("kind"), str)
                and isinstance(item.get("detail"), str)
                for item in items
            ):
                raise ValueError(f"{path}:{line_number}: mechanism_items is malformed")
            if not isinstance(record.get("mechanism_context"), str):
                raise ValueError(f"{path}:{line_number}: mechanism_context is required")
            if not isinstance(record.get("cpg_quality"), dict):
                raise ValueError(f"{path}:{line_number}: cpg_quality is required")
            seen.add(key)
            records.append(record)
    if not records:
        raise ValueError("dataset is empty")
    return records


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


def _validate_variant(
    variant: str,
    excluded_groups: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {', '.join(MODEL_VARIANTS)}")
    excluded = validate_mechanism_groups(excluded_groups)
    if excluded and variant in _SOURCE_VARIANTS | {"raw_cpg"}:
        raise ValueError("mechanism group ablation is only valid for mechanism variants")
    return variant, excluded


class InputBuilder:
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
        self.mechanism_prefix = self._encode("\nCPG_DERIVED_VULNERABILITY_MECHANISMS:\n")
        self.mechanism_only_prefix = self._encode("CPG_DERIVED_VULNERABILITY_MECHANISMS:\n")

    def _encode(self, text: str, max_length: int | None = None) -> list[int]:
        result = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=max_length is not None,
            max_length=max_length,
        )
        return list(result["input_ids"])

    def _with_eos(self, ids: list[int]) -> list[int]:
        return ids if self.eos_token_id is None else [*ids, self.eos_token_id]

    def source_ids(self, record: dict[str, object]) -> list[int]:
        source = self._encode(str(record["raw_source"]), self.source_max_length)
        return self._with_eos([*self.source_prefix, *source])

    def mechanism_text(
        self,
        record: dict[str, object],
        *,
        excluded_groups: tuple[str, ...],
    ) -> str:
        items = record["mechanism_items"]
        assert isinstance(items, list)
        return render_mechanism_items(items, excluded_groups=excluded_groups)

    def mechanism_ids(
        self,
        record: dict[str, object],
        *,
        excluded_groups: tuple[str, ...],
    ) -> list[int]:
        context = self._encode(
            self.mechanism_text(record, excluded_groups=excluded_groups),
            self.context_max_length,
        )
        return self._with_eos([*self.mechanism_only_prefix, *context])

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
            ids.extend(self.raw_cpg_prefix)
            ids.extend(self._encode(str(record["cpg_relations"]), self.context_max_length))
        elif variant == "mechanism_concat":
            ids.extend(self.mechanism_prefix)
            ids.extend(self._encode(
                self.mechanism_text(record, excluded_groups=excluded_groups),
                self.context_max_length,
            ))
        elif variant not in _SOURCE_VARIANTS:
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
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(sequences), width), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :len(sequence)] = 1
        return input_ids.to(device), attention_mask.to(device)

    def sequence_batch(
        self,
        records: list[dict[str, object]],
        *,
        variant: str,
        excluded_groups: tuple[str, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._pad([
            self.sequence_ids(record, variant=variant, excluded_groups=excluded_groups)
            for record in records
        ], device=device)

    def fusion_batch(
        self,
        records: list[dict[str, object]],
        *,
        excluded_groups: tuple[str, ...],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_ids, source_mask = self._pad(
            [self.source_ids(record) for record in records], device=device
        )
        mechanism_ids, mechanism_mask = self._pad(
            [self.mechanism_ids(record, excluded_groups=excluded_groups) for record in records],
            device=device,
        )
        return source_ids, source_mask, mechanism_ids, mechanism_mask


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
    return get_peft_model(base, config), int(base.config.hidden_size)


def _masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


class BidirectionalSourceAdapter(nn.Module):
    """One non-causal layer over contextual token states; Qwen stays causal."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, 256)
        self.context = nn.TransformerEncoderLayer(
            256, 4, dim_feedforward=512, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.up = nn.Linear(256, hidden_size)
        # Start from the original representation, rather than perturbing it at step zero.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        local = self.down(self.norm(hidden))
        local = self.context(local, src_key_padding_mask=~mask.bool())
        return hidden + self.up(local)


class SourceAttentionPool(nn.Module):
    """Four learned queries, with no predefined vulnerability categories."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.keys = nn.Linear(hidden_size, 256)
        self.queries = nn.Parameter(torch.empty(4, 256))
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("btd,qd->bqt", self.keys(self.norm(hidden)), self.queries) / 16.0
        weights = scores.masked_fill(~mask.bool().unsqueeze(1), float("-inf")).softmax(dim=-1)
        # Preserve the baseline representation width and linear classifier size.
        return torch.einsum("bqt,bth->bqh", weights, hidden).mean(dim=1)


class SequenceVulnerabilityClassifier(nn.Module):
    def __init__(self, model_path: str, *, device: torch.device, lora_r: int, lora_alpha: int,
                 lora_dropout: float, target_modules: tuple[str, ...], gradient_checkpointing: bool,
                 variant: str = "baseline") -> None:
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
        # Extra CPU initialization must not alter the encoder/classifier initialization
        # or the subsequent training RNG stream of the baseline control.
        with torch.random.fork_rng(devices=[]):
            if variant in {"source_bidirectional", "source_bidirectional_attention"}:
                self.task_modules["source_adapter"] = BidirectionalSourceAdapter(hidden_size)
        with torch.random.fork_rng(devices=[]):
            if variant in {"source_attention", "source_bidirectional_attention"}:
                self.task_modules["source_pool"] = SourceAttentionPool(hidden_size)
        self.to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        if "source_adapter" in self.task_modules:
            hidden = self.task_modules["source_adapter"](hidden.float(), attention_mask)
        if "source_pool" in self.task_modules:
            pooled = self.task_modules["source_pool"](hidden.float(), attention_mask)
        else:
            pooled = _masked_mean(hidden, attention_mask).float()
        return self.task_modules["classifier"](
            pooled
        ).squeeze(-1)


class MechanismFusionClassifier(nn.Module):
    def __init__(self, model_path: str, *, device: torch.device, lora_r: int, lora_alpha: int,
                 lora_dropout: float, target_modules: tuple[str, ...], gradient_checkpointing: bool,
                 fusion_dim: int, fusion_heads: int) -> None:
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
        self.task_modules = nn.ModuleDict({
            "source_projection": nn.Linear(hidden_size, fusion_dim),
            "mechanism_projection": nn.Linear(hidden_size, fusion_dim),
            "cross_attention": nn.MultiheadAttention(fusion_dim, fusion_heads, batch_first=True),
            "layer_norm": nn.LayerNorm(fusion_dim),
            "classifier": nn.Linear(fusion_dim, 1),
        })
        self.to(device)

    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        mechanism_ids: torch.Tensor,
        mechanism_mask: torch.Tensor,
    ) -> torch.Tensor:
        source_hidden = self.encoder(
            input_ids=source_ids, attention_mask=source_mask, use_cache=False
        ).last_hidden_state
        mechanism_hidden = self.encoder(
            input_ids=mechanism_ids, attention_mask=mechanism_mask, use_cache=False
        ).last_hidden_state
        source = self.task_modules["source_projection"](source_hidden.float())
        mechanism = self.task_modules["mechanism_projection"](mechanism_hidden.float())
        attended, _ = self.task_modules["cross_attention"](
            query=source,
            key=mechanism,
            value=mechanism,
            key_padding_mask=~mechanism_mask.bool(),
            need_weights=False,
        )
        fused = self.task_modules["layer_norm"](source + attended)
        return self.task_modules["classifier"](
            _masked_mean(fused, source_mask)
        ).squeeze(-1)


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
    common = dict(
        device=device,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        gradient_checkpointing=gradient_checkpointing,
    )
    if variant in _SEQUENCE_VARIANTS:
        return SequenceVulnerabilityClassifier(model_path, variant=variant, **common)
    if variant == "mechanism_fusion":
        return MechanismFusionClassifier(
            model_path, fusion_dim=fusion_dim, fusion_heads=fusion_heads, **common
        )
    raise ValueError(f"unsupported model variant {variant}")


def _labels(records: list[dict[str, object]], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [float(record["label"]) for record in records], dtype=torch.float32, device=device
    )


def _forward_batch(
    model,
    records: list[dict[str, object]],
    input_builder: InputBuilder,
    *,
    variant: str,
    excluded_groups: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    if variant in _SEQUENCE_VARIANTS:
        input_ids, mask = input_builder.sequence_batch(
            records, variant=variant, excluded_groups=excluded_groups, device=device
        )
        return model(input_ids, mask)
    source_ids, source_mask, mechanism_ids, mechanism_mask = input_builder.fusion_batch(
        records, excluded_groups=excluded_groups, device=device
    )
    return model(source_ids, source_mask, mechanism_ids, mechanism_mask)


@torch.no_grad()
def _predict_logits(
    model,
    records: list[dict[str, object]],
    input_builder: InputBuilder,
    *,
    variant: str,
    excluded_groups: tuple[str, ...],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    predictions: list[torch.Tensor] = []
    for start in tqdm(range(0, len(records), batch_size), desc="Predict", unit="batch",
                      dynamic_ncols=True, mininterval=1, file=sys.stdout, colour="cyan"):
        logits = _forward_batch(
            model,
            records[start:start + batch_size],
            input_builder,
            variant=variant,
            excluded_groups=excluded_groups,
            device=device,
        )
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite prediction logits")
        predictions.append(logits.float().cpu())
    return torch.cat(predictions) if predictions else torch.empty(0)


def _predict(model, records, input_builder, **kwargs) -> torch.Tensor:
    return torch.sigmoid(_predict_logits(model, records, input_builder, **kwargs))


def _binary_auc(truth: torch.Tensor, scores: torch.Tensor) -> float | None:
    positives = int(truth.sum().item())
    negatives = int((~truth).sum().item())
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(
        ((float(score), bool(label)) for score, label in zip(scores.tolist(), truth.tolist())),
        key=lambda item: item[0],
    )
    rank_sum, rank, index = 0.0, 1, 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (rank + rank + end - index - 1) / 2.0
        rank_sum += average_rank * sum(1 for _, label in pairs[index:end] if label)
        rank += end - index
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _classification_metrics(
    records: list[dict[str, object]], probabilities: torch.Tensor, *, threshold: float
) -> Metrics:
    predictions = probabilities >= threshold
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
    return Metrics(accuracy, precision, recall, f1, mcc, _binary_auc(truth, probabilities))


def _mechanism_presence(record: dict[str, object]) -> tuple[bool, bool]:
    items = record.get("mechanism_items")
    if not isinstance(items, list):
        raise ValueError(f"{record.get('sample_key')}: mechanism_items is missing or malformed")
    has_relation = any(
        isinstance(item, dict) and item.get("category") == "MECHANISM_RELATION"
        for item in items
    )
    has_candidate = any(
        isinstance(item, dict) and item.get("category") == "MECHANISM_CANDIDATE"
        for item in items
    )
    return has_relation, has_candidate


def _stratified_metrics(
    records: list[dict[str, object]],
    probabilities: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, object]:
    masks: dict[str, list[bool]] = {
        "relation_present": [],
        "relation_absent": [],
        "candidate_present": [],
        "relation_only": [],
    }
    for record in records:
        has_relation, has_candidate = _mechanism_presence(record)
        masks["relation_present"].append(has_relation)
        masks["relation_absent"].append(not has_relation)
        masks["candidate_present"].append(has_candidate)
        masks["relation_only"].append(has_relation and not has_candidate)

    result: dict[str, object] = {}
    for name, mask_values in masks.items():
        indices = [index for index, keep in enumerate(mask_values) if keep]
        if not indices:
            continue
        subgroup_records = [records[index] for index in indices]
        subgroup_probabilities = probabilities[indices]
        result[name] = {
            "samples": len(subgroup_records),
            "positive": sum(int(record["label"]) == 1 for record in subgroup_records),
            "negative": sum(int(record["label"]) == 0 for record in subgroup_records),
            "metrics": _classification_metrics(
                subgroup_records,
                subgroup_probabilities,
                threshold=threshold,
            ).as_json(),
        }
    return result


def _select_validation_threshold(
    records: list[dict[str, object]], probabilities: torch.Tensor
) -> tuple[float, Metrics]:
    best_threshold = 0.5
    best = _classification_metrics(records, probabilities, threshold=best_threshold)
    best_key = (best.mcc, best.f1, best.accuracy, -abs(best_threshold - 0.5))
    for value in range(5, 96):
        threshold = value / 100.0
        metrics = _classification_metrics(records, probabilities, threshold=threshold)
        key = (metrics.mcc, metrics.f1, metrics.accuracy, -abs(threshold - 0.5))
        if key > best_key:
            best_threshold, best, best_key = threshold, metrics, key
    return best_threshold, best


def _cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def train_model(
    dataset_path: str | Path | None,
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
    excluded_groups: tuple[str, ...] = (),
    seed: int = 42,
    device: str = "auto",
    fixed_epochs: bool = False,
    records: list[dict[str, object]] | None = None,
    log_every: int = 10,
    initial_checkpoint: str | Path | None = None,
    sample_weights: dict[str, float] | None = None,
) -> dict[str, object]:
    variant, excluded_groups = _validate_variant(variant, excluded_groups)
    if batch_size <= 0 or gradient_accumulation <= 0 or epochs <= 0:
        raise ValueError("batch_size, gradient_accumulation, and epochs must be positive")
    resolved_device = _resolve_device(device)
    if log_every <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid log_every, learning_rate or weight_decay")
    if (dataset_path is None) == (records is None):
        raise ValueError("provide exactly one of dataset_path or records")
    records = _read_jsonl(dataset_path) if records is None else records
    sources = {record_dataset(record) for record in records}
    if len(sources) != 1:
        raise ValueError(f"training dataset view must contain exactly one source, got {sorted(sources)}")
    source_dataset = next(iter(sources))
    if source_dataset == "sven":
        raise ValueError("SVEN is external-test-only")
    train_records = [record for record in records if record_split(record) == "train"]
    valid_records = [record for record in records if record_split(record) == "valid"]
    if fixed_epochs and len(train_records) != len(records):
        raise ValueError("fixed-epoch fitting accepts training records only")
    if not train_records or (not fixed_epochs and not valid_records):
        raise ValueError("formal training requires non-empty train and valid splits")
    if {int(record["label"]) for record in train_records} != {0, 1}:
        raise ValueError("training split must contain both labels")
    if not fixed_epochs and {int(record["label"]) for record in valid_records} != {0, 1}:
        raise ValueError("validation split must contain both labels")

    if sample_weights is not None:
        keys = [r["sample_key"] for r in train_records]
        if len(set(keys)) != len(keys) or set(sample_weights) != set(keys):
            raise ValueError("sample_weights must cover unique training keys exactly")
        if any(not math.isfinite(w) or w <= 0 for w in sample_weights.values()):
            raise ValueError("sample weights must be finite and positive")
    initial = None
    if initial_checkpoint is not None:
        if not fixed_epochs:
            raise ValueError("checkpoint continuation requires fixed_epochs")
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        expected = dict(checkpoint_version=_CHECKPOINT_VERSION, dataset_schema_version=DATASET_SCHEMA_VERSION,
                        trained_on=source_dataset, variant=variant, model_path=model_path,
                        source_max_length=source_max_length, context_max_length=context_max_length,
                        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                        excluded_groups=excluded_groups,
                        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                        fusion_dim=fusion_dim, fusion_heads=fusion_heads)
        for name, value in expected.items():
            if initial.get(name) != value:
                raise ValueError(f"initial checkpoint mismatch: {name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    input_builder = InputBuilder(
        tokenizer, source_max_length=source_max_length, context_max_length=context_max_length
    )
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
    _seed_everything(seed)
    model = _build_model(
        variant,
        model_path,
        device=resolved_device,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        gradient_checkpointing=resolved_device.type == "cuda",
        fusion_dim=fusion_dim,
        fusion_heads=fusion_heads,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if initial is not None:
        set_peft_model_state_dict(model.encoder, initial["adapter_state"])
        model.task_modules.load_state_dict(initial["task_state"])
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_key = (float("-inf"),) * 4
    best_threshold = 0.5
    best_adapter = None
    best_task = None
    best_validation = None
    selected_epoch = None
    history = TrainingProgress(output_path)
    global_step = 0

    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_records)))
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        optimizer_steps = 0
        pending = 0
        num_batches = math.ceil(len(order) / batch_size)
        window_loss, window_samples = 0.0, 0
        bar = tqdm(range(0, len(order), batch_size), total=num_batches,
                   desc=f"{variant} · epoch {epoch+1}/{epochs}", unit="batch",
                   dynamic_ncols=True, mininterval=1, file=sys.stdout, colour="cyan")
        for batch_index, start in enumerate(bar):
            batch = [train_records[index] for index in order[start:start + batch_size]]
            logits = _forward_batch(
                model, batch, input_builder,
                variant=variant, excluded_groups=excluded_groups, device=resolved_device,
            )
            losses = loss_fn(logits, _labels(batch, resolved_device))
            if sample_weights is not None:
                weights = losses.new_tensor([sample_weights[r["sample_key"]] for r in batch])
                losses = losses * weights
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss in epoch {epoch+1}, batch {batch_index+1}")
            group_start = (batch_index // gradient_accumulation) * gradient_accumulation
            group_end = min(group_start + gradient_accumulation, num_batches)
            group_samples = min(group_end * batch_size, len(order)) - group_start * batch_size
            (loss * len(batch) / group_samples).backward()
            value = float(loss.detach().cpu())
            running_loss += value * len(batch)
            window_loss += value * len(batch)
            window_samples += len(batch)
            bar.set_postfix(loss=f"{window_loss/window_samples:.4f}", step=global_step,
                            lr=f"{learning_rate:.1e}", refresh=False)
            pending += 1
            if pending == gradient_accumulation or batch_index + 1 == num_batches:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_step += 1
                pending = 0
                bar.set_postfix(loss=f"{window_loss/window_samples:.4f}", step=global_step,
                                lr=f"{learning_rate:.1e}", refresh=False)
                if global_step == 1 or global_step % log_every == 0 or batch_index+1 == num_batches:
                    history.add(dict(event="step", epoch=epoch+1, step=global_step,
                                     loss=window_loss/window_samples, learning_rate=learning_rate,
                                     samples_seen=min(start+len(batch), len(order))))
                    window_loss, window_samples = 0.0, 0

        validation = None
        threshold = 0.5
        if not fixed_epochs:
            probabilities = _predict(
                model, valid_records, input_builder,
                variant=variant, excluded_groups=excluded_groups,
                batch_size=batch_size, device=resolved_device,
            )
            threshold, validation = _select_validation_threshold(valid_records, probabilities)
        epoch_row = {
            "event": "epoch",
            "variant": variant,
            "excluded_groups": list(excluded_groups),
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_records),
            "optimizer_steps": optimizer_steps,
            "validation_threshold": threshold,
            "validation": validation.as_json() if validation else None,
        }
        history.add(epoch_row)
        key = ((epoch+1,) if fixed_epochs else
               (validation.mcc, validation.f1, validation.accuracy,
                validation.auc if validation.auc is not None else float("-inf")))
        if key > best_key:
            best_key = key
            best_threshold = threshold
            best_adapter = _cpu_state(get_peft_model_state_dict(model.encoder))
            best_task = _cpu_state(model.task_modules.state_dict())
            best_validation = validation
            selected_epoch = epoch+1

    if best_adapter is None or best_task is None:
        raise RuntimeError("training did not produce a checkpoint")
    checkpoint = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "trained_on": source_dataset,
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
        "decision_threshold": best_threshold,
        "selection": "fixed_epochs" if fixed_epochs else "validation_mcc",
        "selected_epoch": selected_epoch,
        "seed": seed,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint is not None else None,
        "sample_weights": sample_weights,
        "training_config": dict(epochs=epochs, batch_size=batch_size,
                                gradient_accumulation=gradient_accumulation,
                                learning_rate=learning_rate, weight_decay=weight_decay),
        "adapter_state": best_adapter,
        "task_state": best_task,
        "validation": best_validation.as_json() if best_validation else None,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(target)
    print(json.dumps({
        "checkpoint": str(target),
        "trained_on": source_dataset,
        "variant": variant,
        "decision_threshold": best_threshold,
        "validation": best_validation.as_json() if best_validation else None,
        "trainable_parameters": checkpoint["trainable_parameters"],
    }, ensure_ascii=False), flush=True)
    return checkpoint


def predict_checkpoint(checkpoint_path: str | Path, records: list[dict], *,
                       batch_size: int = 1, device: str = "auto") -> torch.Tensor:
    """Raw logits for explicit records; fold ownership is enforced by the caller."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != _CHECKPOINT_VERSION or checkpoint.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("checkpoint is incompatible with current model/dataset")
    resolved = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint["model_path"]), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    builder = InputBuilder(tokenizer, source_max_length=int(checkpoint["source_max_length"]),
                           context_max_length=int(checkpoint["context_max_length"]))
    model = _load_model(checkpoint, device=resolved)
    return _predict_logits(model, records, builder, variant=str(checkpoint["variant"]),
                           excluded_groups=tuple(checkpoint["excluded_groups"]),
                           batch_size=batch_size, device=resolved)


def _load_model(checkpoint: dict[str, object], *, device: torch.device):
    model = _build_model(
        str(checkpoint["variant"]),
        str(checkpoint["model_path"]),
        device=device,
        lora_r=int(checkpoint["lora_r"]),
        lora_alpha=int(checkpoint["lora_alpha"]),
        lora_dropout=float(checkpoint["lora_dropout"]),
        target_modules=tuple(str(value) for value in checkpoint["target_modules"]),
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
    split: str,
    batch_size: int = 1,
    device: str = "auto",
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_version") != _CHECKPOINT_VERSION:
        raise ValueError("checkpoint is incompatible with the current model")
    if checkpoint.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("checkpoint was not trained on the current dataset schema")
    variant = str(checkpoint["variant"])
    excluded_groups = validate_mechanism_groups(
        tuple(str(value) for value in checkpoint.get("excluded_groups", ()))
    )
    _validate_variant(variant, excluded_groups)
    all_records = _read_jsonl(dataset_path)
    sources = {record_dataset(record) for record in all_records}
    if len(sources) != 1:
        raise ValueError(f"evaluation dataset view must contain exactly one source, got {sorted(sources)}")
    evaluated_on = next(iter(sources))
    records = [record for record in all_records if record_split(record) == split]
    if not records:
        raise ValueError(f"{split} split is empty")

    resolved_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint["model_path"]), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_builder = InputBuilder(
        tokenizer,
        source_max_length=int(checkpoint["source_max_length"]),
        context_max_length=int(checkpoint["context_max_length"]),
    )
    model = _load_model(checkpoint, device=resolved_device)
    probabilities = _predict(
        model, records, input_builder,
        variant=variant, excluded_groups=excluded_groups,
        batch_size=batch_size, device=resolved_device,
    )
    threshold = float(checkpoint["decision_threshold"])
    metrics = _classification_metrics(records, probabilities, threshold=threshold)
    result = {
        "trained_on": checkpoint["trained_on"],
        "evaluated_on": evaluated_on,
        "split": split,
        "variant": variant,
        "excluded_groups": list(excluded_groups),
        "decision_threshold": threshold,
        "samples": len(records),
        "positive": sum(int(record["label"]) == 1 for record in records),
        "negative": sum(int(record["label"]) == 0 for record in records),
        "metrics": metrics.as_json(),
        "mechanism_subgroups": _stratified_metrics(
            records,
            probabilities,
            threshold=threshold,
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
