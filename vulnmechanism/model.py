from __future__ import annotations

import json
import math
import random
import sys
from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
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
from .progress import training_bar as tqdm

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
    "codebert_source",
    "source_attention",
    "source_bidirectional",
    "source_bidirectional_attention",
    "raw_cpg",
    "mechanism_concat",
    "mechanism_fusion",
    "graph_attributes",
    "graph_cfg",
)
_SOURCE_VARIANTS = {"baseline", "codebert_source", "source_attention", "source_bidirectional", "source_bidirectional_attention"}
_SEQUENCE_VARIANTS = _SOURCE_VARIANTS | {"raw_cpg", "mechanism_concat"}
_FUSION_VARIANTS = {"mechanism_fusion"}
_GRAPH_VARIANTS = {"graph_attributes", "graph_cfg"}
_CHECKPOINT_VERSION = 8
_CLASSIFIER_INITIALIZATION = 'shared base RNG; LoRA initialization isolated; training RNG reset'


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


def _read_jsonl(path: str | Path, *, source_only: bool = False) -> list[dict[str, object]]:
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
            if not source_only and record.get("schema_version") != DATASET_SCHEMA_VERSION:
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
            if source_only and 'raw_source' not in record:
                record['raw_source'] = record.get('function')
            if not isinstance(record.get("raw_source"), str) or not record["raw_source"]:
                raise ValueError(f"{path}:{line_number}: raw_source is required")
            if source_only:
                seen.add(key)
                records.append(record)
                continue
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
    if excluded and variant in _SOURCE_VARIANTS | _GRAPH_VARIANTS | {"raw_cpg"}:
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

    @lru_cache(maxsize=32768)
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

    def codebert_batch(self, records, *, device, return_source_positions=False):
        """Native 512-token windows, half-window stride, one weighted function pool.

        source_max_length is the logical total budget, including one copy of
        the unchanged task prefix and the two RoBERTa boundary tokens. Each
        source position and each prefix/boundary slot has total weight one.
        """
        if self.source_max_length not in (512, 2048):
            raise ValueError('CodeBERT comparison requires logical length 512 or 2048')
        bos = getattr(self.tokenizer, 'bos_token_id', None)
        if bos is None or self.eos_token_id is None:
            raise ValueError('CodeBERT requires native BOS/EOS tokens')
        capacity = 512 - len(self.source_prefix) - 2
        budget = self.source_max_length - len(self.source_prefix) - 2
        if capacity <= 0:
            raise ValueError('task prefix leaves no CodeBERT source capacity')
        stride = capacity // 2
        sequences, weights, owners, positions = [], [], [], []
        for owner, record in enumerate(records):
            tokens = self._encode(str(record['raw_source']), budget)
            starts = [0]
            while starts[-1] + capacity < len(tokens):
                starts.append(starts[-1] + stride)
            coverage = [0] * len(tokens)
            for start in starts:
                for pos in range(start, min(start + capacity, len(tokens))):
                    coverage[pos] += 1
            for start in starts:
                end = min(start + capacity, len(tokens))
                sequences.append([bos, *self.source_prefix, *tokens[start:end], self.eos_token_id])
                weights.append([1. / len(starts)] * (1 + len(self.source_prefix)) +
                               [1. / coverage[pos] for pos in range(start, end)] + [1. / len(starts)])
                owners.append(owner)
                positions.append([-1] * (1 + len(self.source_prefix)) + list(range(start, end)) + [-1])
        ids, mask = self._pad(sequences, device=device)
        pool_weights = torch.zeros_like(mask, dtype=torch.float32)
        for i, values in enumerate(weights):
            pool_weights[i, :len(values)] = torch.tensor(values, device=device)
        result = (ids, mask, torch.tensor(owners, dtype=torch.long, device=device), pool_weights, len(records))
        if return_source_positions:
            source_positions = torch.full_like(ids, -1)
            for i, values in enumerate(positions):
                source_positions[i, :len(values)] = torch.tensor(values, device=device)
            return (*result, source_positions)
        return result

    @lru_cache(maxsize=16384)
    def local_layout(self, source: str, encoder: str) -> dict:
        """Source-only fast offsets and lexical character overlap; no supervision."""
        from .syntax import code_line_spans
        if not getattr(self.tokenizer, 'is_fast', False):
            raise ValueError('local risk pooling requires a fast tokenizer')
        result = self.tokenizer(source, add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True, verbose=False)
        tokens, offsets = list(result['input_ids']), list(result['offset_mapping'])
        if len(tokens) != len(offsets) or tokens != self._encode(source):
            raise ValueError('source tokenizer IDs/offsets mismatch')
        if any(not (0 <= a <= b <= len(source)) for a, b in offsets) or any(
                offsets[i][0] < offsets[i-1][0] for i in range(1, len(offsets))):
            raise ValueError('source offset mapping is outside Python character coordinates')
        budget = self.source_max_length - len(self.source_prefix) - 2 if encoder == 'codebert' else self.source_max_length
        special = set(self.tokenizer.all_special_ids)
        # A literal may contain e.g. "</s>". Its fast offset is valid source
        # evidence even though that special token must not enter the line pool.
        valid_offsets = [(a, b) for a, b in offsets if b > a]
        merged = []
        for a, b in valid_offsets:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
            else:
                merged.append((a, b))
        lexical = code_line_spans(source)
        interval = 0
        for a, b in sorted(span for spans in lexical.values() for span in spans):
            while interval < len(merged) and merged[interval][1] <= a:
                interval += 1
            if interval == len(merged) or merged[interval][0] > a or merged[interval][1] < b:
                raise ValueError('source offsets do not cover lexical code characters')
        parts = sorted((a, b, line) for line, spans in lexical.items() for a, b in spans)
        ends = [b for _, b, _ in parts]
        overlaps = {}
        for index, (token, (a, b)) in enumerate(zip(tokens[:budget], offsets[:budget])):
            if b <= a or token in special:
                continue
            counts = {}
            at = bisect_right(ends, a)
            while at < len(parts) and parts[at][0] < b:
                left, right, line = parts[at]
                counts[line] = counts.get(line, 0) + max(0, min(b, right) - max(a, left))
                at += 1
            total = sum(counts.values())
            for line, count in counts.items():
                if count:
                    overlaps.setdefault(line, []).append((index, count / total))
        lines = sorted(overlaps)
        if not lines:
            raise ValueError('function has no visible effective code lines (m=0)')
        visible_end = max((b for _, b in offsets[:budget]), default=0)
        return dict(ids=tokens[:budget], token_count=len(tokens), lines=lines, eos_token_id=self.eos_token_id,
                    full_lines=[line for line in lines if max(b for _, b in lexical[line]) <= visible_end],
                    complete=len(tokens) <= budget,
                    line_weights=[overlaps[line] for line in lines])

    def local_batch(self, records, *, encoder, device):
        layouts = [self.local_layout(str(r['raw_source']), encoder) for r in records]
        if encoder == 'codebert':
            return (*self.codebert_batch(records, device=device, return_source_positions=True), layouts)
        if self.eos_token_id is None:
            raise ValueError('Qwen local risk requires an effective EOS token')
        sequences = [self._with_eos([*self.source_prefix, *layout['ids']]) for layout in layouts]
        ids, mask = self._pad(sequences, device=device)
        positions = torch.full_like(ids, -1)
        for i, layout in enumerate(layouts):
            start = len(self.source_prefix)
            positions[i, start:start+len(layout['ids'])] = torch.arange(len(layout['ids']), device=device)
        return (ids, mask, torch.arange(len(records), device=device), None, len(records), positions, layouts)

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

    def source_alignment_batch(
        self,
        records: list[dict[str, object]],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[tuple[int, int]]]]:
        """Baseline source IDs plus source-only offsets for CFG node alignment."""
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("source alignment requires a fast tokenizer with offset_mapping")
        sequences, offsets = [], []
        for record in records:
            result = self.tokenizer(
                str(record["raw_source"]), add_special_tokens=False,
                truncation=True, max_length=self.source_max_length,
                return_offsets_mapping=True,
            )
            source_ids = list(result["input_ids"])
            source_offsets = list(result["offset_mapping"])
            if len(source_ids) != len(source_offsets):
                raise ValueError("tokenizer IDs and source offsets have different lengths")
            sequences.append(self._with_eos([*self.source_prefix, *source_ids]))
            offsets.append(source_offsets)
        ids, mask = self._pad(sequences, device=device)
        return ids, mask, offsets

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

    def source_function_batch(self, records, *, device, exclude_boundary_token=False):
        """Unchanged encoder input, independent complete-function readout mask.

        Any token overlapping the function participates, including a token that
        also contains boundary whitespace. Interior comments stay in the span.
        EOS and the task prefix are encoder context, not direct readout sites.
        """
        from .syntax import function_readout_span
        ids, attention, offsets = self.source_alignment_batch(records, device=device)
        pooling = torch.zeros_like(attention)
        for index, (record, positions) in enumerate(zip(records, offsets)):
            start, end = function_readout_span(
                str(record['raw_source']), str(record.get('resolved_language') or record['language']))
            for token, (left, right) in enumerate(positions, start=len(self.source_prefix)):
                if left < end and right > start and right > left:
                    pooling[index, token] = 1
                    # A pure terminal delimiter can merge with trailing whitespace
                    # without changing the program. Never discard a mixed token
                    # containing an operator, literal, name, or other code.
                    if (exclude_boundary_token and left <= end - 1 < right
                            and str(record['raw_source'])[max(left, start):min(right, end)].strip() == '}'):
                        pooling[index, token] = 0
        if not pooling.any(dim=1).all():
            raise ValueError('target function has no visible tokens in the unchanged source window')
        return ids, attention, pooling


def _build_lora_encoder(
    model_path: str,
    *,
    device: torch.device,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: tuple[str, ...],
    gradient_checkpointing: bool,
    finetuning_mode: str | None = None,
):
    # Explicit CodeBERT modes share FP32 weights, BF16 CUDA autocast and head RNG.
    # None retains the historical LoRA construction/precision for saved runs.
    if finetuning_mode not in (None, 'lora', 'full'):
        raise ValueError('unknown finetuning mode')
    dtype = torch.bfloat16 if device.type == "cuda" and finetuning_mode is None else torch.float32
    base = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        **({'add_pooling_layer': False} if finetuning_mode is not None else {}),
    )
    base.config.use_cache = False
    if gradient_checkpointing:
        base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
    if finetuning_mode == 'full':
        base.requires_grad_(True)
        return base, int(base.config.hidden_size)
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    if finetuning_mode == 'lora':
        with torch.random.fork_rng(devices=[]):
            encoder = get_peft_model(base, config)
    else:
        encoder = get_peft_model(base, config)
    return encoder, int(base.config.hidden_size)


def _masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _window_mean(hidden, mask, owners, weights, function_count):
    if (weights.shape != mask.shape or hidden.shape[:2] != mask.shape or
            owners.shape != (len(hidden),) or function_count <= 0 or
            torch.any(owners < 0) or torch.any(owners >= function_count) or
            torch.any(weights < 0) or torch.any(weights[~mask.bool()] != 0)):
        raise ValueError('invalid function window pooling inputs')
    totals = weights.sum(1)
    denominator = totals.new_zeros(function_count).index_add(0, owners, totals)
    if torch.any(denominator <= 0):
        raise ValueError('every function must have visible pooling tokens')
    sums = (hidden.float() * weights.unsqueeze(-1)).sum(1)
    return sums.new_zeros((function_count, hidden.shape[-1])).index_add(0, owners, sums) / denominator[:, None]


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


class MultiPrototypeHead(nn.Module):
    """K affine positive scores against a shared zero negative reference."""
    def __init__(self, hidden_size: int, k: int = 4):
        super().__init__()
        if k < 1:
            raise ValueError("k must be positive")
        self.prototypes = nn.Linear(hidden_size, k)

    def forward(self, features):
        scores = self.prototypes(features)
        return torch.logsumexp(scores, dim=-1, keepdim=True) - math.log(scores.shape[-1])


class LocalRiskHead(nn.Module):
    """Shared 64-dimensional local/global risk head; no global logit branch."""
    def __init__(self, hidden_size):
        super().__init__()
        self.local = nn.Linear(hidden_size, 64)
        self.global_context = nn.Linear(hidden_size, 64, bias=False)
        self.score = nn.Linear(64, 1)

    def forward(self, lines, global_context):
        scores = self.score(torch.nn.functional.gelu(
            self.local(lines.float()) + self.global_context(global_context.float()))).squeeze(-1)
        if scores.numel() == 0:
            raise ValueError('local risk head received m=0')
        return torch.logsumexp(scores, dim=0) - math.log(len(scores)), scores


def _pool_source_lines(hidden, owners, positions, layouts):
    """Average duplicate window states by global source token, then pool lines."""
    if positions.shape != hidden.shape[:2] or len(owners) != len(hidden):
        raise ValueError('window/source position shape mismatch')
    result = []
    for owner, layout in enumerate(layouts):
        selected = owners == owner
        indices = positions[selected].reshape(-1)
        states = hidden[selected].reshape(-1, hidden.shape[-1]).float()
        keep = indices >= 0
        indices, states = indices[keep], states[keep]
        n = len(layout['ids'])
        if len(indices) == 0 or torch.any(indices >= n):
            raise ValueError('source token indices missing or outside visible source')
        sums = states.new_zeros((n, hidden.shape[-1])).index_add(0, indices, states)
        counts = states.new_zeros(n).index_add(0, indices, states.new_ones(len(indices)))
        if torch.any(counts <= 0):
            raise ValueError('visible source token absent from its encoder windows')
        tokens = sums / counts[:, None]
        line_indices, token_indices, fractions = [], [], []
        for i, weights in enumerate(layout['line_weights']):
            for token, fraction in weights:
                line_indices.append(i)
                token_indices.append(token)
                fractions.append(fraction)
        line_indices = torch.tensor(line_indices, dtype=torch.long, device=hidden.device)
        token_indices = torch.tensor(token_indices, dtype=torch.long, device=hidden.device)
        weights = states.new_tensor(fractions)
        m = len(layout['lines'])
        if m == 0:
            raise ValueError('local risk has no visible code instances (m=0)')
        totals = states.new_zeros(m).index_add(0, line_indices, weights)
        if torch.any(totals <= 0):
            raise ValueError('local code line has zero tokenizer overlap')
        lines = states.new_zeros((m, states.shape[-1])).index_add(
            0, line_indices, tokens[token_indices] * weights[:, None]) / totals[:, None]
        result.append(lines)
    return result


class SequenceVulnerabilityClassifier(nn.Module):
    def __init__(self, model_path: str, *, device: torch.device, lora_r: int, lora_alpha: int,
                 lora_dropout: float, target_modules: tuple[str, ...], gradient_checkpointing: bool,
                 variant: str = "baseline", classifier_head: str = "linear",
                 finetuning_mode: str | None = None) -> None:
        super().__init__()
        self.encoder, hidden_size = _build_lora_encoder(
            model_path,
            device=device,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            gradient_checkpointing=gradient_checkpointing,
            **({'finetuning_mode': finetuning_mode} if finetuning_mode is not None else {}),
        )
        self.variant = variant
        self.finetuning_mode = finetuning_mode
        self.classifier_head = classifier_head
        self.task_modules = nn.ModuleDict({"classifier": nn.Linear(hidden_size, 1)})
        if classifier_head not in {"linear", "mlp4", "prototype4", "local_risk"}:
            raise ValueError("unknown classifier head")
        if classifier_head == 'local_risk' and variant not in {'baseline', 'codebert_source'}:
            raise ValueError('local risk requires a source-only encoder')
        if classifier_head not in {"linear", "local_risk"} and variant != "codebert_source":
            raise ValueError("head comparison is restricted to CodeBERT source")
        # Preserve the original encoder initialization and subsequent dropout RNG.
        with torch.random.fork_rng(devices=[]):
            if classifier_head == "mlp4":
                self.task_modules["classifier"] = nn.Sequential(
                    nn.Linear(hidden_size, 4), nn.Tanh(), nn.Linear(4, 1))
            elif classifier_head == "prototype4":
                original = self.task_modules["classifier"]
                head = MultiPrototypeHead(hidden_size, 4)
                with torch.no_grad():
                    head.prototypes.weight.sub_(head.prototypes.weight.mean(0, keepdim=True))
                    head.prototypes.weight.add_(original.weight)
                    head.prototypes.bias.fill_(float(original.bias.detach()))
                self.task_modules["classifier"] = head
            elif classifier_head == 'local_risk':
                self.task_modules['classifier'] = LocalRiskHead(hidden_size)
        # Extra CPU initialization must not alter the encoder/classifier initialization
        # or the subsequent training RNG stream of the baseline control.
        with torch.random.fork_rng(devices=[]):
            if variant in {"source_bidirectional", "source_bidirectional_attention"}:
                self.task_modules["source_adapter"] = BidirectionalSourceAdapter(hidden_size)
        with torch.random.fork_rng(devices=[]):
            if variant in {"source_attention", "source_bidirectional_attention"}:
                self.task_modules["source_pool"] = SourceAttentionPool(hidden_size)
        self.to(device)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                window_owners=None, pooling_weights=None, function_count=None) -> torch.Tensor:
        hidden = self._source_hidden(input_ids, attention_mask)
        if "source_adapter" in self.task_modules:
            hidden = self.task_modules["source_adapter"](hidden.float(), attention_mask)
        if window_owners is not None:
            pooled = _window_mean(hidden, attention_mask, window_owners, pooling_weights, function_count)
        elif "source_pool" in self.task_modules:
            pooled = self.task_modules["source_pool"](hidden.float(), attention_mask)
        else:
            pooled = _masked_mean(hidden, attention_mask).float()
        return pooled

    def _source_hidden(self, input_ids, attention_mask):
        encoder_grad = torch.is_grad_enabled() and any(p.requires_grad for p in self.encoder.parameters())
        with torch.set_grad_enabled(encoder_grad), torch.autocast(
                input_ids.device.type, dtype=torch.bfloat16,
                enabled=self.finetuning_mode is not None and input_ids.is_cuda):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                use_cache=False).last_hidden_state

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                window_owners=None, pooling_weights=None, function_count=None,
                source_positions=None, layouts=None, *, return_local=False) -> torch.Tensor:
        if self.classifier_head == 'local_risk':
            if source_positions is None or layouts is None or len(layouts) != function_count:
                raise ValueError('local risk requires source-only offsets and original line numbers')
            hidden = self._source_hidden(input_ids, attention_mask)
            if self.variant == 'codebert_source':
                global_context = _window_mean(hidden, attention_mask, window_owners, pooling_weights, function_count)
            else:
                ends = attention_mask.long().sum(1) - 1
                if torch.any(ends < 0) or any(int(input_ids[i, end]) != layout['eos_token_id']
                                            for i, (end, layout) in enumerate(zip(ends, layouts))):
                    raise ValueError('Qwen global context must use the last effective EOS')
                global_context = hidden[torch.arange(function_count, device=hidden.device), ends].float()
            line_features = _pool_source_lines(hidden, window_owners, source_positions, layouts)
            output, local = [], []
            for features, context, layout in zip(line_features, global_context, layouts):
                z, scores = self.task_modules['classifier'](features, context)
                output.append(z)
                local.append((layout['lines'], scores))
            logits = torch.stack(output)
            return (logits, local) if return_local else logits
        pooled = self.encode(input_ids, attention_mask, window_owners, pooling_weights, function_count)
        return self.task_modules["classifier"](pooled).squeeze(-1)


class GraphVulnerabilityClassifier(SequenceVulnerabilityClassifier):
    """Uncompressed source representation plus a DeepDFA-style graph vector.

    The two linear terms are exactly a linear classifier on concatenated
    representations. Keeping the source term separate preserves its original
    initialization. This is not an error router or a guarantee against regressions.
    """

    def __init__(self, model_path: str, *, graph_config: dict, variant: str, **kwargs) -> None:
        from .graph_nn import StaticGraphEncoder
        super().__init__(model_path, variant="baseline", **kwargs)
        with torch.random.fork_rng(devices=[]):
            graph_encoder = StaticGraphEncoder(graph_config, propagate=variant == "graph_cfg")
            graph_classifier = nn.Linear(graph_encoder.out_dim, 1, bias=False)
            # All three variants start with identical source logits. Joint
            # training may still change both the source and the graph branch.
            nn.init.zeros_(graph_classifier.weight)
        self.task_modules["graph_encoder"] = graph_encoder
        self.task_modules["graph_classifier"] = graph_classifier
        self.to(kwargs["device"])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                graphs: list[dict | None]) -> torch.Tensor:
        source_logits = super().forward(input_ids, attention_mask)
        graph_vector = self.task_modules["graph_encoder"](graphs)
        return source_logits + self.task_modules["graph_classifier"](graph_vector).squeeze(-1)


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
    graph_config: dict | None = None,
    classifier_head: str = "linear",
    finetuning_mode: str | None = None,
):
    if finetuning_mode is not None and (finetuning_mode not in {'lora', 'full'}
            or variant != 'codebert_source' or classifier_head != 'linear'):
        raise ValueError('explicit finetuning modes require CodeBERT mean pooling and linear head')
    common = dict(
        device=device,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        gradient_checkpointing=gradient_checkpointing,
    )
    if variant in _GRAPH_VARIANTS:
        if graph_config is None:
            raise ValueError("graph variants require a training-only feature vocabulary")
        return GraphVulnerabilityClassifier(
            model_path, variant=variant, graph_config=graph_config, **common
        )
    if variant in _SEQUENCE_VARIANTS:
        return SequenceVulnerabilityClassifier(model_path, variant=variant, classifier_head=classifier_head,
            **({'finetuning_mode': finetuning_mode} if finetuning_mode is not None else {}), **common)
    if variant == "mechanism_fusion":
        return MechanismFusionClassifier(
            model_path, fusion_dim=fusion_dim, fusion_heads=fusion_heads, **common
        )
    raise ValueError(f"unsupported model variant {variant}")


def _labels(records: list[dict[str, object]], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [float(record["label"]) for record in records], dtype=torch.float32, device=device
    )


@contextmanager
def _auxiliary_mode(model):
    """Disable dropout while retaining gradient checkpointing and gradients."""
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    for module, _ in flags:
        if getattr(module, 'gradient_checkpointing', False) and torch.is_grad_enabled():
            module.training = True
    try:
        yield
    finally:
        for module, flag in flags:
            module.training = flag


def _forward_batch(
    model,
    records: list[dict[str, object]],
    input_builder: InputBuilder,
    *,
    variant: str,
    excluded_groups: tuple[str, ...],
    device: torch.device,
    return_local: bool = False,
) -> torch.Tensor:
    if getattr(model, 'classifier_head', None) == 'local_risk':
        encoder = 'codebert' if variant == 'codebert_source' else 'qwen'
        return model(*input_builder.local_batch(records, encoder=encoder, device=device), return_local=return_local)
    if return_local:
        raise ValueError('line scores require the local risk classifier')
    if variant == 'codebert_source':
        return model(*input_builder.codebert_batch(records, device=device))
    if variant in _GRAPH_VARIANTS:
        input_ids, mask = input_builder.sequence_batch(
            records, variant="baseline", excluded_groups=(), device=device
        )
        if any("static_graph" not in record for record in records):
            raise ValueError("static_graph missing; run graph_experiment build first")
        return model(input_ids, mask, [record["static_graph"] for record in records])
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


def function_view_batches(records, views, micro_size):
    """Each original function has total BCE weight one, regardless of view count."""
    if views is None:
        yield records, None
        return
    expanded, coefficients = [], []
    for record in records:
        sources = [record['raw_source'], *views[record['sample_key']]]
        expanded.extend(dict(record, raw_source=source) for source in sources)
        coefficients.extend([1./len(sources)] * len(sources))
    for start in range(0, len(expanded), micro_size):
        yield expanded[start:start+micro_size], coefficients[start:start+micro_size]


def synthetic_supervision_loss(logits, objective, labels=None):
    """Absolute BCE on all members; optional ranking only for opposite labels."""
    if logits.shape != (2,):
        raise ValueError('one verified two-member group is required')
    labels=logits.new_tensor([0.,1.] if labels is None else labels)
    if labels.shape!=(2,) or not bool(((labels==0)|(labels==1)).all()):
        raise ValueError('binary local safety labels required')
    if objective not in ('bce','bce_rank'):
        raise ValueError('unknown synthetic objective')
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
    if objective=='bce_rank' and bool(labels[0]!=labels[1]):
        from .cgvl import local_rank_loss
        loss=loss+local_rank_loss(logits[labels==0],logits[labels==1])
    return loss


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
    initial_adapter_state: dict | None = None,
    sample_weights: dict[str, float] | None = None,
    graph_options: dict | None = None,
    synthetic_pairs: list | None = None,
    synthetic_objective: str | None = None,
    classifier_head: str = "linear",
    change_objective: str | None = None,
    repair_pairs: list | None = None,
    invariant_views: list | None = None,
    invariance_weight: float = 1.0,
    ranking_weight: float = 1.0,
    local_objective: str | None = None,
    local_regions: dict | None = None,
    finetuning_mode: str | None = None,
    function_views: dict[str, list[str]] | None = None,
    augmentation_config: dict | None = None,
) -> dict[str, object]:
    variant, excluded_groups = _validate_variant(variant, excluded_groups)
    if finetuning_mode is not None and (finetuning_mode not in {'lora', 'full'}
            or variant != 'codebert_source' or classifier_head != 'linear'
            or any(x is not None for x in (change_objective, local_objective, synthetic_pairs,
                                           initial_adapter_state))):
        raise ValueError('explicit finetuning modes require CodeBERT function BCE without auxiliary losses')
    if classifier_head not in {"linear", "local_risk"} and variant != "codebert_source":
        raise ValueError("alternative heads require codebert_source")
    if variant == 'codebert_source' and source_max_length not in (512, 2048):
        raise ValueError('CodeBERT comparison requires logical length 512 or 2048')
    if batch_size <= 0 or gradient_accumulation <= 0 or epochs <= 0:
        raise ValueError("batch_size, gradient_accumulation, and epochs must be positive")
    if synthetic_pairs is not None:
        if variant != 'baseline' or synthetic_objective not in ('bce', 'bce_rank') or not synthetic_pairs:
            raise ValueError('synthetic supervision requires baseline and nonempty matched pairs')
        if any(len(p)!=2 or any(r.get('label') not in (0,1) for r in p) or
               any(r.get('split')!='train' for r in p) for p in synthetic_pairs):
            raise ValueError('synthetic supervision accepts verified binary training pairs only')
    elif synthetic_objective is not None:
        raise ValueError('synthetic objective requires verified pairs')
    resolved_device = _resolve_device(device)
    if log_every <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid log_every, learning_rate or weight_decay")
    if (dataset_path is None) == (records is None):
        raise ValueError("provide exactly one of dataset_path or records")
    records = _read_jsonl(dataset_path, source_only=variant == 'codebert_source') if records is None else records
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
    if function_views is not None:
        if (finetuning_mode != 'full' or variant != 'codebert_source' or classifier_head != 'linear'
                or any(x is not None for x in (sample_weights, change_objective, local_objective,
                                               synthetic_pairs, initial_checkpoint, initial_adapter_state))
                or fixed_epochs or not isinstance(augmentation_config, dict)
                or augmentation_config.get('mode') not in ('random', 'high_low')):
            raise ValueError('function views require fresh full CodeBERT BCE without other objectives')
        if set(function_views) != {r['sample_key'] for r in train_records}:
            raise ValueError('function views must cover exactly the original Train functions')
        for record in train_records:
            views = function_views[record['sample_key']]
            if (len(views) not in (0, 2) or any(not isinstance(s, str) or not s for s in views)
                    or len(set([record['raw_source'], *views])) != 1 + len(views)):
                raise ValueError('function views require zero or two distinct transformed sources')
    elif augmentation_config is not None:
        raise ValueError('augmentation configuration requires function views')

    if change_objective is not None:
        from .change_learning import (OBJECTIVES, invariance_loss, random_pair_control,
                                      ranking_loss, validate_supervision)
        if (change_objective not in OBJECTIVES or variant not in {"baseline", "codebert_source"}
                or classifier_head != "linear" or synthetic_pairs is not None
                or sample_weights is not None or initial_checkpoint is not None
                or initial_adapter_state is not None or fixed_epochs):
            raise ValueError("change learning requires a fresh source-only linear classifier with Valid selection")
        if seed != 42 or source_dataset not in {"primevul", "cleanvul"}:
            raise ValueError("change learning requires seed=42 and independent PrimeVul/CleanVul training")
        if any(not math.isfinite(w) or w <= 0 for w in (invariance_weight, ranking_weight)):
            raise ValueError("change loss weights must be finite and positive")
        validate_supervision(train_records, repair_pairs, invariant_views)
        if len(repair_pairs) < 2:
            raise ValueError("at least two real pairs are required for the random ranking control")
    elif repair_pairs is not None or (invariant_views is not None and local_objective is None):
        raise ValueError("repair pairs and invariant views require a change objective")

    if local_objective is not None:
        from .change_learning import (LOCAL_OBJECTIVES, invariance_loss, line_invariance_loss,
                                      region_loss, validate_local_supervision)
        if (local_objective not in LOCAL_OBJECTIVES or variant not in {'baseline', 'codebert_source'}
                or classifier_head != ('linear' if local_objective == 'A' else 'local_risk')
                or initial_checkpoint is None or fixed_epochs or change_objective is not None
                or synthetic_pairs is not None or sample_weights is not None or initial_adapter_state is not None):
            raise ValueError('local risk requires a matching BCE initialization and source-only Valid selection')
        if (seed != 42 or source_max_length != 2048 or epochs != 3 or batch_size * gradient_accumulation != 8
                or (lora_r, lora_alpha, lora_dropout, weight_decay) != (16, 32, .05, .01)):
            raise ValueError('local risk fixes seed=42, length=2048, LoRA=16/32/.05, 3 epochs and effective batch=8')
        validate_local_supervision(train_records, local_regions, invariant_views)
    elif classifier_head == 'local_risk' or local_regions is not None:
        raise ValueError('local risk heads/regions require a local objective')

    graph_config = None
    if variant in _GRAPH_VARIANTS:
        from .graph_features import fit_graph_config, validate_record_graph
        if initial_checkpoint is not None:
            raise ValueError("graph checkpoint continuation is not supported; vocabulary must stay fixed")
        for record in train_records + valid_records:
            validate_record_graph(record)
        graph_config = fit_graph_config(train_records, **(graph_options or {}))
    elif graph_options:
        raise ValueError("graph_options are only valid for graph variants")

    if sample_weights is not None:
        keys = [r["sample_key"] for r in train_records]
        if len(set(keys)) != len(keys) or set(sample_weights) != set(keys):
            raise ValueError("sample_weights must cover unique training keys exactly")
        if any(not math.isfinite(w) or w <= 0 for w in sample_weights.values()):
            raise ValueError("sample weights must be finite and positive")
    initial = None
    if initial_adapter_state is not None and (initial_checkpoint is not None or variant != "baseline"):
        raise ValueError("adapter initialization requires a fresh baseline classifier")
    if initial_checkpoint is not None:
        if not fixed_epochs and local_objective is None:
            raise ValueError("checkpoint continuation requires fixed_epochs")
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        if initial.get('finetuning_mode') != finetuning_mode:
            raise ValueError('initial checkpoint mismatch: finetuning_mode')
        expected = dict(checkpoint_version=_CHECKPOINT_VERSION, dataset_schema_version=DATASET_SCHEMA_VERSION,
                        trained_on=source_dataset, variant=variant, model_path=model_path,
                        source_max_length=source_max_length, context_max_length=context_max_length,
                        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                        excluded_groups=excluded_groups,
                        target_modules=(("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense")
                                        if variant == 'codebert_source' else ("q_proj", "k_proj", "v_proj", "o_proj")),
                        fusion_dim=fusion_dim, fusion_heads=fusion_heads)
        for name, value in expected.items():
            if initial.get(name) != value:
                raise ValueError(f"initial checkpoint mismatch: {name}")
        if local_objective is not None:
            members = {split: [(r['sample_key'], r['label']) for r in records if r['split'] == split]
                       for split in ('train', 'valid')}
            if (initial.get('classifier_head', 'linear') != 'linear' or initial.get('seed') != 42
                    or initial.get('cohort_members') != members
                    or initial.get('training_config', {}).get('epochs') != 3):
                raise ValueError('initial BCE checkpoint does not match the cohort, head, seed or budget')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    input_builder = InputBuilder(
        tokenizer, source_max_length=source_max_length, context_max_length=context_max_length
    )
    if local_objective is not None:
        encoder_name = 'codebert' if variant == 'codebert_source' else 'qwen'
        normalized_regions, region_skips = {}, {}
        bce_encoder_sequences, bce_visible_tokens = 0, 0
        for record in train_records:
            layout = input_builder.local_layout(record['raw_source'], encoder_name)
            bce_visible_tokens += len(layout['ids'])
            capacity = 512-len(input_builder.source_prefix)-2
            bce_encoder_sequences += (1+math.ceil(max(0, len(layout['ids'])-capacity)/(capacity//2))
                                      if encoder_name == 'codebert' else 1)
            raw = local_regions.get(record['sample_key'], [])
            chosen = sorted(set(raw) & set(layout['full_lines']))
            if 0 < len(chosen) < len(layout['lines']):
                normalized_regions[record['sample_key']] = chosen
            elif raw:
                region_skips[record['sample_key']] = 'empty_visible_M' if not chosen else 'all_visible_instances'
        local_regions = normalized_regions
    target_modules = (("attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense")
                      if variant == 'codebert_source' else ("q_proj", "k_proj", "v_proj", "o_proj"))
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
        classifier_head=classifier_head,
        **({'finetuning_mode': finetuning_mode} if finetuning_mode is not None else {}),
        **({"graph_config": graph_config} if graph_config is not None else {}),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if initial_adapter_state is not None:
        set_peft_model_state_dict(model.encoder, initial_adapter_state)
    if initial is not None:
        if finetuning_mode == 'full':
            model.encoder.load_state_dict(initial['encoder_state'], strict=True)
        else:
            _restore_adapter(model.encoder, initial['adapter_state'])
        if local_objective in {None, 'A'}:
            model.task_modules.load_state_dict(initial["task_state"], strict=True)
    if finetuning_mode is not None:
        _seed_everything(seed)
    if local_objective is not None:
        lora_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
        head_parameters = list(model.task_modules.parameters())
        for parameter in lora_parameters:
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(head_parameters, lr=1e-4, weight_decay=.01)
    else:
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
        if local_objective is not None and epoch == 1:
            for parameter in lora_parameters:
                parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW([
                dict(params=lora_parameters, lr=2e-5), dict(params=head_parameters, lr=1e-4)], weight_decay=.01)
        model.train()
        order = list(range(len(train_records)))
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        synthetic_total = 0.0
        synthetic_order = list(range(len(synthetic_pairs))) if synthetic_pairs is not None else []
        random.Random(seed + epoch).shuffle(synthetic_order)
        if change_objective is not None:
            pair_order = list(range(len(repair_pairs)))
            view_order = list(range(len(invariant_views)))
            random.Random(seed + epoch).shuffle(pair_order)
            random.Random(seed + epoch).shuffle(view_order)
            epoch_steps = math.ceil(math.ceil(len(order) / batch_size) / gradient_accumulation)
            ranking_pairs = [repair_pairs[pair_order[i % len(pair_order)]] for i in range(epoch_steps)]
            if change_objective == "bce_random_ranking":
                ranking_pairs = random_pair_control(ranking_pairs, seed + epoch)
            invariance_total, ranking_total = 0.0, 0.0
        optimizer_steps = 0
        pending = 0
        num_batches = math.ceil(len(order) / batch_size)
        if local_objective is not None:
            local_total, local_count, inv_total, inv_count = 0., 0, 0., 0
            local_weighted, inv_weighted = 0., 0.
            empty_aux, auxiliary_forwards = 0, 0
            auxiliary_encoder_sequences = 0
            epoch_steps = math.ceil(num_batches / gradient_accumulation)
            view_order = list(range(len(invariant_views)))
            random.Random(seed + epoch).shuffle(view_order)
            phase = 'head_bce' if epoch == 0 else 'joint'
        window_loss, window_samples = 0.0, 0
        view_forwards, encoder_sequences, encoder_token_slots = 0, 0, 0
        bar = tqdm(range(0, len(order), batch_size), total=num_batches,
                   desc=f"{variant} · epoch {epoch+1}/{epochs}", unit="batch",
                   dynamic_ncols=True, mininterval=1, file=sys.stdout, colour="cyan")
        for batch_index, start in enumerate(bar):
            batch = [train_records[index] for index in order[start:start + batch_size]]
            group_start = (batch_index // gradient_accumulation) * gradient_accumulation
            group_end = min(group_start + gradient_accumulation, num_batches)
            group_samples = min(group_end * batch_size, len(order)) - group_start * batch_size
            value = 0.
            for forwarded, coefficients in function_view_batches(batch, function_views, batch_size):
                prediction = _forward_batch(
                    model, forwarded, input_builder,
                    variant=variant, excluded_groups=excluded_groups, device=resolved_device,
                    return_local=local_objective is not None and local_objective != 'A',
                )
                logits, local_scores = prediction if local_objective is not None and local_objective != 'A' else (prediction, None)
                losses = loss_fn(logits, _labels(forwarded, resolved_device))
                if sample_weights is not None:
                    weights = losses.new_tensor([sample_weights[r["sample_key"]] for r in forwarded])
                    losses = losses * weights
                loss = (losses.mean() if coefficients is None else
                        (losses * losses.new_tensor(coefficients)).sum()/len(batch))
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite loss in epoch {epoch+1}, batch {batch_index+1}")
                training_loss = loss * len(batch) / group_samples
                if local_objective is not None:
                    ramp = (optimizer_steps / max(1, epoch_steps-1)) if epoch == 1 else 1.
                    loc_weight = .1 * ramp if epoch > 0 and local_objective in {'C', 'D', 'C_random'} else 0.
                    inv_weight = .01 * ramp if epoch > 0 and local_objective == 'D' else 0.
                    if epoch > 0 and local_scores is not None:
                        selected = []
                        for record, (line_numbers, scores) in zip(batch, local_scores):
                            if record['label'] == 1:
                                term = region_loss(scores, line_numbers, local_regions.get(record['sample_key'], []))
                                if term is not None:
                                    selected.append(term)
                        if selected:
                            group_records = [train_records[i] for i in order[group_start*batch_size:min(group_end*batch_size, len(order))]]
                            group_valid = 0
                            encoder_name = 'codebert' if variant == 'codebert_source' else 'qwen'
                            for row in group_records:
                                if row['label'] == 1:
                                    layout = input_builder.local_layout(row['raw_source'], encoder_name)
                                    candidate = set(local_regions.get(row['sample_key'], [])) & set(layout['full_lines'])
                                    group_valid += 0 < len(candidate) < len(layout['lines'])
                            term = torch.stack(selected).mean()
                            training_loss = training_loss + loc_weight * term * len(selected) / group_valid
                            local_total += float(term.detach()) * len(selected)
                            local_weighted += loc_weight * float(term.detach()) * len(selected)
                            local_count += len(selected)
                training_loss.backward()
                value += float(loss.detach().cpu())
                if function_views is not None:
                    capacity = 512 - len(input_builder.source_prefix) - 2
                    budget = source_max_length - len(input_builder.source_prefix) - 2
                    lengths = [len(input_builder._encode(r['raw_source'], budget)) for r in forwarded]
                    sequences = sum(1+math.ceil(max(0, n-capacity)/(capacity//2)) for n in lengths)
                    view_forwards += len(forwarded)
                    encoder_sequences += sequences
                    encoder_token_slots += sequences * min(512, max(lengths)+len(input_builder.source_prefix)+2)
            running_loss += value * len(batch)
            window_loss += value * len(batch)
            window_samples += len(batch)
            bar.set_postfix(loss=f"{window_loss/window_samples:.4f}", step=global_step,
                            lr=f"{learning_rate:.1e}", refresh=False)
            pending += 1
            if pending == gradient_accumulation or batch_index + 1 == num_batches:
                if local_objective is not None and epoch > 0:
                    if invariant_views:
                        view = invariant_views[view_order[optimizer_steps % len(view_order)]]
                        with _auxiliary_mode(model):
                            aux = _forward_batch(model, list(view), input_builder, variant=variant,
                                excluded_groups=excluded_groups, device=resolved_device, return_local=local_objective != 'A')
                            term = invariance_loss(aux) if local_objective == 'A' else line_invariance_loss(aux[1])
                            if not torch.isfinite(term):
                                raise ValueError('nonfinite local invariance loss')
                            (inv_weight * term).backward()
                        inv_total += float(term.detach())
                        inv_weighted += inv_weight * float(term.detach())
                        inv_count += 1
                        auxiliary_forwards += 2
                        for record in view:
                            layout = input_builder.local_layout(record['raw_source'], encoder_name)
                            capacity = 512-len(input_builder.source_prefix)-2
                            auxiliary_encoder_sequences += (1+math.ceil(max(0,len(layout['ids'])-capacity)/(capacity//2))
                                if encoder_name == 'codebert' else 1)
                    else:
                        empty_aux += 1
                if change_objective is not None:
                    # Deterministic auxiliary predictions isolate code changes
                    # from dropout noise. Gradients still update the shared model.
                    # Zero-weight ablations execute the same forwards/backwards,
                    # preserving the number of examples and optimizer updates.
                    model.eval()
                    view = invariant_views[view_order[optimizer_steps % len(view_order)]]
                    view_logits = _forward_batch(model, list(view), input_builder, variant=variant,
                        excluded_groups=excluded_groups, device=resolved_device)
                    invariant_loss = invariance_loss(view_logits)
                    alpha = invariance_weight if change_objective in {"bce_invariance", "full"} else 0.0
                    if not torch.isfinite(invariant_loss):
                        raise ValueError("non-finite invariance loss")
                    (alpha * invariant_loss).backward()
                    invariance_total += float(invariant_loss.detach().cpu())
                    pair = ranking_pairs[optimizer_steps]
                    pair_logits = _forward_batch(model, list(pair), input_builder, variant=variant,
                        excluded_groups=excluded_groups, device=resolved_device)
                    rank_loss = ranking_loss(pair_logits)
                    beta = ranking_weight if change_objective in {"bce_ranking", "full", "bce_random_ranking"} else 0.0
                    if not torch.isfinite(rank_loss):
                        raise ValueError("non-finite repair ranking loss")
                    (beta * rank_loss).backward()
                    ranking_total += float(rank_loss.detach().cpu())
                    model.train()
                if synthetic_pairs is not None:
                    pair = synthetic_pairs[synthetic_order[optimizer_steps % len(synthetic_order)]]
                    pair_logits = _forward_batch(model, pair, input_builder, variant=variant,
                        excluded_groups=excluded_groups, device=resolved_device)
                    synthetic_loss = synthetic_supervision_loss(pair_logits, synthetic_objective, [r['label'] for r in pair])
                    if not torch.isfinite(synthetic_loss):
                        raise ValueError('non-finite synthetic supervision loss')
                    synthetic_loss.backward()
                    synthetic_total += float(synthetic_loss.detach().cpu())
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
                                     loss=window_loss/window_samples, learning_rate=1e-4 if local_objective is not None else learning_rate,
                                     samples_seen=min(start+len(batch), len(order)),
                                     **(dict(phase=phase, region_weight=loc_weight, invariance_weight=inv_weight,
                                             head_learning_rate=1e-4, lora_learning_rate=0. if epoch == 0 else 2e-5)
                                        if local_objective is not None else {})))
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
        if function_views is not None:
            epoch_row.update(augmentation_mode=augmentation_config['mode'],
                augmented_functions=sum(bool(v) for v in function_views.values()),
                bce_function_forwards=view_forwards, bce_encoder_sequences=encoder_sequences,
                encoder_token_slots=encoder_token_slots,
                function_loss='mean view BCE per function, then mean over original functions')
        if synthetic_pairs is not None:
            epoch_row['synthetic_loss'] = synthetic_total / optimizer_steps
            epoch_row['synthetic_objective'] = synthetic_objective
            epoch_row['synthetic_pairs_seen'] = optimizer_steps
        if change_objective is not None:
            epoch_row.update(change_objective=change_objective,
                invariance_loss=invariance_total / optimizer_steps,
                ranking_loss=ranking_total / optimizer_steps,
                auxiliary_function_forwards=4 * optimizer_steps,
                ranking_positive_unique=len({p[0]["sample_key"] for p in ranking_pairs}),
                ranking_negative_unique=len({p[1]["sample_key"] for p in ranking_pairs}),
                total_loss=running_loss / len(train_records)
                    + alpha * invariance_total / optimizer_steps + beta * ranking_total / optimizer_steps)
        if local_objective is not None:
            epoch_row.update(local_objective=local_objective, phase=phase,
                region_loss=local_total/local_count if local_count else None, region_functions=local_count,
                region_skipped=len(train_records)-local_count,
                invariance_loss=inv_total/inv_count if inv_count else None, invariance_functions=inv_count,
                weighted_region_loss=local_weighted/local_count if local_count else 0.,
                weighted_invariance_loss=inv_weighted/inv_count if inv_count else 0.,
                empty_invariance_steps=empty_aux, auxiliary_function_forwards=auxiliary_forwards,
                bce_function_forwards=len(train_records), bce_encoder_sequences=bce_encoder_sequences,
                bce_visible_source_tokens=bce_visible_tokens, auxiliary_encoder_sequences=auxiliary_encoder_sequences,
                head_learning_rate=1e-4, lora_learning_rate=0. if epoch == 0 else 2e-5,
                total_loss=running_loss/len(train_records) + (local_weighted/local_count if local_count else 0.)
                    + (inv_weighted/inv_count if inv_count else 0.))
        history.add(epoch_row)
        key = ((epoch+1,) if fixed_epochs else
               (validation.mcc, validation.f1, validation.accuracy,
                validation.auc if validation.auc is not None else float("-inf")))
        if key > best_key:
            best_key = key
            best_threshold = threshold
            best_adapter = _cpu_state(model.encoder.state_dict() if finetuning_mode == 'full'
                                      else get_peft_model_state_dict(model.encoder))
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
        "classifier_head": classifier_head,
        "cohort_members": {split: [(r['sample_key'], r['label']) for r in records if r['split'] == split]
                           for split in ('train', 'valid')},
        "training_config": dict(epochs=epochs, batch_size=batch_size,
                                gradient_accumulation=gradient_accumulation,
                                learning_rate=1e-4 if local_objective is not None else learning_rate, weight_decay=weight_decay),
        "encoder_state" if finetuning_mode == 'full' else "adapter_state": best_adapter,
        "task_state": best_task,
        "validation": best_validation.as_json() if best_validation else None,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }
    if finetuning_mode is not None:
        checkpoint['finetuning_mode'] = finetuning_mode
        checkpoint['encoder_precision'] = 'FP32 weights/optimizer; BF16 CUDA autocast; FP32 head/pooling/loss'
        checkpoint['classifier_initialization'] = _CLASSIFIER_INITIALIZATION
    if function_views is not None:
        checkpoint['augmentation_config'] = augmentation_config
    if synthetic_pairs is not None:
        checkpoint['synthetic_supervision'] = dict(objective=synthetic_objective, weight=1.0,
            pairs_per_optimizer_step=1, unique_pairs=len(synthetic_pairs))
    if change_objective is not None:
        checkpoint["change_supervision"] = dict(objective=change_objective,
            invariance_weight=alpha, ranking_weight=beta,
            unique_repair_pairs=len(repair_pairs), unique_safe_views=len(invariant_views),
            auxiliary_pairs_per_optimizer_step=2, auxiliary_dropout=False,
            ranking_control="deranged real-pair negatives" if change_objective == "bce_random_ranking" else "real repairs",
            bce_samples_per_epoch=len(train_records), optimizer_steps_per_epoch=optimizer_steps,
            auxiliary_function_forwards_per_epoch=4 * optimizer_steps)
        if change_objective == "bce_random_ranking":
            checkpoint["change_supervision"]["random_sampling_protocol"] = "epoch_marginal_matched"
    if local_objective is not None:
        checkpoint['local_supervision'] = dict(objective=local_objective,
            head_version=1 if local_objective != 'A' else None, hidden_dim=64 if local_objective != 'A' else None,
            region_weight=.1 if local_objective in {'C', 'D', 'C_random'} else 0.,
            invariance_weight=.01 if local_objective == 'D' else 0.,
            warmup_epochs=1, joint_epochs=2, optimizer_reset=True,
            lora_learning_rate=2e-5, head_learning_rate=1e-4, weight_decay=.01, gradient_clip=1.,
            warmup_loss='BCE only; encoder frozen', ramp='first joint epoch: linear 0 to configured weights',
            selected_phase='head_bce' if selected_epoch == 1 else 'joint',
            region_functions=len(local_regions), safe_views=len(invariant_views),
            region_skips=region_skips,
            auxiliary_dropout=False, auxiliary_functions_per_joint_step=2 if invariant_views else 0,
            input='single original function; no patch, candidate, or label dependent input construction',
            line_pooling='lexical character overlap; duplicate source-window token states averaged first',
            global_pooling=('existing function mean' if local_objective == 'A' else
                'existing CodeBERT unique-token function mean' if variant == 'codebert_source' else 'last effective EOS'))
    if variant == 'codebert_source':
        capacity = 512 - len(input_builder.source_prefix) - 2
        checkpoint['input_protocol'] = dict(encoder='roberta', native_window=512,
            logical_budget=source_max_length, source_budget=source_max_length-len(input_builder.source_prefix)-2,
            window_source_capacity=capacity, stride=capacity//2,
            pooling='mean over unique source positions; repeated prefix/BOS/EOS averaged across windows',
            function_loss=('mean view BCE per function, then mean over original functions'
                           if function_views is not None else 'one BCE per original function'))
    if graph_config is not None:
        checkpoint["graph_config"] = graph_config
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
    mode = checkpoint.get('finetuning_mode')
    if mode not in (None, 'lora', 'full'):
        raise ValueError('unknown checkpoint finetuning mode')
    if mode is not None and checkpoint.get('encoder_precision') != 'FP32 weights/optimizer; BF16 CUDA autocast; FP32 head/pooling/loss':
        raise ValueError('finetuning checkpoint precision configuration mismatch')
    state_key = 'encoder_state' if mode == 'full' else 'adapter_state'
    if state_key not in checkpoint or ('adapter_state' if mode == 'full' else 'encoder_state') in checkpoint:
        raise ValueError('checkpoint encoder state does not match finetuning mode')
    if checkpoint.get('classifier_head') == 'local_risk':
        config = checkpoint.get('local_supervision', {})
        if (config.get('head_version'), config.get('hidden_dim'), checkpoint.get('source_max_length')) != (1, 64, 2048):
            raise ValueError('local risk head/input checkpoint configuration is incompatible')
    model = _build_model(
        str(checkpoint["variant"]),
        str(checkpoint["model_path"]),
        device=device,
        lora_r=int(checkpoint["lora_r"]),
        lora_alpha=int(checkpoint["lora_alpha"]),
        lora_dropout=float(checkpoint["lora_dropout"]),
        target_modules=tuple(str(value) for value in checkpoint["target_modules"]),
        gradient_checkpointing=False,
        classifier_head=str(checkpoint.get("classifier_head", "linear")),
        **({'finetuning_mode': mode} if mode is not None else {}),
        fusion_dim=int(checkpoint["fusion_dim"]),
        fusion_heads=int(checkpoint["fusion_heads"]),
        **({"graph_config": checkpoint.get("graph_config")}
           if checkpoint["variant"] in _GRAPH_VARIANTS else {}),
    )
    if mode == 'full':
        model.encoder.load_state_dict(checkpoint['encoder_state'], strict=True)
    else:
        _restore_adapter(model.encoder, checkpoint['adapter_state'])
    model.task_modules.load_state_dict(checkpoint["task_state"], strict=True)
    model.eval()
    return model


def _restore_adapter(encoder, state):
    expected = get_peft_model_state_dict(encoder)
    if set(expected) != set(state) or any(expected[key].shape != state[key].shape for key in expected):
        raise ValueError('LoRA checkpoint keys/shapes do not exactly match the encoder')
    set_peft_model_state_dict(encoder, state)


def evaluate_model(
    dataset_path: str | Path | None,
    checkpoint_path: str | Path,
    *,
    split: str,
    batch_size: int = 1,
    device: str = "auto",
    prediction_path: str | Path | None = None,
    records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if prediction_path is not None and Path(prediction_path).exists():
        raise FileExistsError(prediction_path)
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
    if (dataset_path is None) == (records is None):
        raise ValueError('provide exactly one of dataset_path or records')
    all_records = _read_jsonl(dataset_path, source_only=variant == 'codebert_source') if records is None else records
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
    predict_options = dict(variant=variant, excluded_groups=excluded_groups,
                           batch_size=batch_size, device=resolved_device)
    if prediction_path is None:
        probabilities = _predict(model, records, input_builder, **predict_options)
    else:
        logits = _predict_logits(model, records, input_builder, **predict_options)
        probabilities = torch.sigmoid(logits)
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
        "mechanism_subgroups": {} if variant == 'codebert_source' and any(
            'mechanism_items' not in r for r in records) else _stratified_metrics(
            records,
            probabilities,
            threshold=threshold,
        ),
    }
    if prediction_path is not None:
        from .cfg_data import identity
        from .cfg_metrics import metrics as binary_metrics, decision_boundary
        result['bce'] = float(torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), torch.tensor([r['label'] for r in records], dtype=torch.float32)))
        counts = binary_metrics([r['label'] for r in records], probabilities.tolist(), threshold)
        result['metrics'].update({key: counts[key] for key in ('tp', 'tn', 'fp', 'fn')})
        result['fixed_0_5'] = binary_metrics([r['label'] for r in records], probabilities.tolist())
        target = Path(prediction_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as handle:
            for i, row in enumerate(records):
                count = len(input_builder._encode(str(row['raw_source'])))
                budget = int(checkpoint.get('input_protocol', {}).get('source_budget', checkpoint['source_max_length']))
                visible = min(count, budget)
                protocol = checkpoint.get('input_protocol')
                windows = (1 + math.ceil(max(0, visible-protocol['window_source_capacity'])/protocol['stride'])
                           if protocol else 1)
                handle.write(json.dumps({**identity(row), 'score':float(probabilities[i]),
                    'logit':float(logits[i]),'prediction':int(float(probabilities[i])>=decision_boundary(threshold)),
                    'threshold':threshold,'source_token_count':count,'source_truncated':count>budget,
                    'visible_source_tokens':visible,'windows':windows})+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
