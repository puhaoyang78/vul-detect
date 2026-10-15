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
from .artifacts import DATASET_SCHEMA_VERSION


from .progress import TrainingProgress


MODEL_VARIANTS = ('codebert_source',)

_SOURCE_VARIANTS = {'codebert_source'}

_SEQUENCE_VARIANTS = {'codebert_source'}

_FUSION_VARIANTS = set()

_GRAPH_VARIANTS = set()

_CHECKPOINT_VERSION = 8


_CLASSIFIER_INITIALIZATION = 'shared base RNG; LoRA initialization isolated; training RNG reset'


VAT_CONFIG=dict(version=3,epsilon=1.,xi=.001,power_iterations=1,
    precision='FP32 auxiliary encoder; float64 pooling, linear head, Bernoulli KL and norms',
    norm='per function L2 over expanded actual encoder tokens, including overlapping-window copies',
    direction='finite-difference KL gradient; divide by actual copies, multiply soft weights once, normalize')


ROBUSTNESS_INCREMENTAL_MODES=('random_aug_bce','hard_aug_bce','random_aug_vat','hard_aug_vat')


ROBUSTNESS_MODES = ('hard_aug_bce','hard_aug_vat','bce','vat','random_aug_bce','random_aug_vat')
ROBUSTNESS_HISTORY_MODES = ('ours','ours_uniform','ours_no_consistency','ours_no_vat','ours_no_search')

def robustness_config(mode,*,xi=.001):
    if mode not in ROBUSTNESS_MODES+ROBUSTNESS_HISTORY_MODES or mode=='bce':raise ValueError('BCE has no auxiliary robustness configuration')
    incremental=mode in ROBUSTNESS_INCREMENTAL_MODES
    return dict(version=1,mode=mode,alpha=0. if mode in ('vat','ours_uniform') or incremental else 3.,
        lambda_real=0. if mode in ('vat','ours_no_consistency') or incremental else 1.,
        lambda_vat=0. if mode in ('ours_no_vat','random_aug_bce','hard_aug_bce') else 1.,vat=dict(VAT_CONFIG,xi=xi),
        beam_width=2,max_depth=3,train_queries=12,attack_queries=48,
        source_views=mode!='vat',search=mode not in ('vat','ours_no_search','random_aug_bce','random_aug_vat'),
        view_sampling='independent seed/epoch/sample_key coin flip over original and selected view',
        comment_token_limit=64)


def _validate_robustness_config(config):
    if (not isinstance(config,dict) or config.get('mode') not in ROBUSTNESS_MODES+ROBUSTNESS_HISTORY_MODES or config.get('mode')=='bce'
            or not isinstance(config.get('vat'),dict) or config['vat'].get('xi') not in (.001,.01)
            or config!=robustness_config(config['mode'],xi=config['vat']['xi'])):
        raise ValueError('robustness configuration is incompatible')


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


def _validate_variant(variant, excluded_groups):
    if variant != 'codebert_source' or excluded_groups:
        raise ValueError('current mainline requires CodeBERT source and no mechanism exclusions')
    return variant, ()

class InputBuilder:

    def __init__(self, tokenizer, *, source_max_length: int, context_max_length: int) -> None:
        if source_max_length <= 0 or context_max_length <= 0:
            raise ValueError('source_max_length and context_max_length must be positive')
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.context_max_length = context_max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.source_prefix = self._encode('TASK: classify whether this C/C++ function is vulnerable.\nFUNCTION:\n')

    @lru_cache(maxsize=32768)
    def _encode(self, text: str, max_length: int | None=None) -> list[int]:
        result = self.tokenizer(text, add_special_tokens=False, truncation=max_length is not None, max_length=max_length)
        return list(result['input_ids'])

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
        sequences, weights, owners, positions = ([], [], [], [])
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
                weights.append([1.0 / len(starts)] * (1 + len(self.source_prefix)) + [1.0 / coverage[pos] for pos in range(start, end)] + [1.0 / len(starts)])
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

    def vat_batch(self, records, *, alpha, device):
        if alpha != 0.0:
            raise ValueError('mainline VAT uses ordinary global alpha=0')
        batch = self.codebert_batch(records, device=device, return_source_positions=True)
        ids, mask, owners, pooling, n, positions = batch
        budget = self.source_max_length - len(self.source_prefix) - 2
        special = set(getattr(self.tokenizer, 'all_special_ids', [self.pad_token_id, self.eos_token_id, getattr(self.tokenizer, 'bos_token_id', None)]))
        source_ids = [self._encode(r['raw_source'], budget) for r in records]
        valid = [[t not in special for t in tokens] for tokens in source_ids]
        constraints = [[1.0] * len(tokens) for tokens in source_ids]
        eligible = [any(row) for row in valid]
        group_rows = [[-1] * ids.shape[1] for _ in range(len(ids))]
        lookup = {}
        group_owners = []
        group_weights = []
        source_positions = positions.cpu().tolist()
        for window, owner in enumerate(owners.tolist()):
            if not eligible[owner]:
                continue
            for slot, pos in enumerate(source_positions[window]):
                if pos < 0 or not valid[owner][pos]:
                    continue
                key = (owner, pos)
                if key not in lookup:
                    lookup[key] = len(lookup)
                    group_owners.append(owner)
                    group_weights.append(constraints[owner][pos])
                group_rows[window][slot] = lookup[key]
        group_ids = torch.tensor(group_rows, device=device, dtype=torch.long)
        return ((ids, mask, owners, pooling, n), group_ids, torch.tensor(group_owners, device=device, dtype=torch.long), eligible, torch.tensor(group_weights, device=device, dtype=torch.float32))

    def _pad(self, sequences: list[list[int]], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError('cannot create an empty batch')
        width = max((len(sequence) for sequence in sequences))
        input_ids = torch.full((len(sequences), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :len(sequence)] = 1
        return (input_ids.to(device), attention_mask.to(device))

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


def _window_mean(hidden, mask, owners, weights, function_count,*,dtype=torch.float32):
    if (weights.shape != mask.shape or hidden.shape[:2] != mask.shape or
            owners.shape != (len(hidden),) or function_count <= 0 or
            torch.any(owners < 0) or torch.any(owners >= function_count) or
            torch.any(weights < 0) or torch.any(weights[~mask.bool()] != 0)):
        raise ValueError('invalid function window pooling inputs')
    weights=weights.to(dtype)
    totals = weights.sum(1)
    denominator = totals.new_zeros(function_count).index_add(0, owners, totals)
    if torch.any(denominator <= 0):
        raise ValueError('every function must have visible pooling tokens')
    sums = (hidden.to(dtype) * weights.unsqueeze(-1)).sum(1)
    return sums.new_zeros((function_count, hidden.shape[-1])).index_add(0, owners, sums) / denominator[:, None]


def region_mean(hidden, mask, owners, weights, function_count, source_positions, region_token_weights):
    """Read original-source positions with exactly the existing window coverage."""
    if source_positions.shape != mask.shape or len(region_token_weights) != function_count:
        raise ValueError('region/source position shape mismatch')
    selected = torch.zeros_like(weights, dtype=torch.float32)
    for owner, values in enumerate(region_token_weights):
        values = weights.new_tensor(values, dtype=torch.float32)
        if values.ndim != 1 or not torch.isfinite(values).all() or torch.any(values < 0) or torch.any(values > 1):
            raise ValueError('invalid region token weights')
        windows = owners == owner
        positions = source_positions[windows]
        valid = (positions >= 0) & mask[windows].bool()
        if not valid.any() or int(positions[valid].max()) + 1 != len(values):
            raise ValueError('region token weights differ from actual source positions')
        gathered = torch.zeros_like(positions, dtype=torch.float32)
        gathered[valid] = values[positions[valid]]
        selected[windows] = gathered
    return _window_mean(hidden, mask, owners, weights * selected, function_count)


class SequenceVulnerabilityClassifier(nn.Module):

    def __init__(self, model_path: str, *, device: torch.device, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: tuple[str, ...], gradient_checkpointing: bool, variant: str='codebert_source', classifier_head: str='linear', finetuning_mode: str | None=None, conditional_length_config: dict | None=None) -> None:
        super().__init__()
        self.encoder, hidden_size = _build_lora_encoder(model_path, device=device, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules, gradient_checkpointing=gradient_checkpointing, **{'finetuning_mode': finetuning_mode} if finetuning_mode is not None else {})
        self.variant = 'codebert_source'
        self.finetuning_mode = finetuning_mode
        self.classifier_head = 'linear'
        self.task_modules = nn.ModuleDict({'classifier': nn.Linear(hidden_size, 1)})
        self.to(device)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, window_owners=None, pooling_weights=None, function_count=None, *, embedding_delta=None, force_fp32=False) -> torch.Tensor:
        hidden = self._source_hidden(input_ids, attention_mask, embedding_delta=embedding_delta, force_fp32=force_fp32)
        if window_owners is not None:
            pooled = _window_mean(hidden, attention_mask, window_owners, pooling_weights, function_count, dtype=torch.float64 if force_fp32 else torch.float32)
        else:
            pooled = _masked_mean(hidden, attention_mask).float()
        return pooled

    def _source_hidden(self, input_ids, attention_mask, *, embedding_delta=None, force_fp32=False):
        encoder_grad = torch.is_grad_enabled() and any((p.requires_grad for p in self.encoder.parameters()))
        with torch.set_grad_enabled(encoder_grad), torch.autocast(input_ids.device.type, dtype=torch.bfloat16, enabled=self.finetuning_mode is not None and input_ids.is_cuda and (not force_fp32)):
            if embedding_delta is None:
                return self.encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
            embeds = self.encoder.get_input_embeddings()(input_ids)
            if embeds.shape != embedding_delta.shape or not torch.isfinite(embedding_delta).all():
                raise ValueError('VAT embedding perturbation has invalid shape or values')
            pad = self.encoder.config.pad_token_id
            valid = input_ids.ne(pad).long()
            position_ids = valid.cumsum(-1) * valid + pad
            return self.encoder(inputs_embeds=embeds + embedding_delta, attention_mask=attention_mask, position_ids=position_ids, use_cache=False).last_hidden_state

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, window_owners=None, pooling_weights=None, function_count=None, source_positions=None, layouts=None, *, return_local=False, embedding_delta=None, force_fp32=False) -> torch.Tensor:
        pooled = self.encode(input_ids, attention_mask, window_owners, pooling_weights, function_count, embedding_delta=embedding_delta, force_fp32=force_fp32)
        if force_fp32:
            head = self.task_modules['classifier']
            return torch.nn.functional.linear(pooled.double(), head.weight.double(), head.bias.double()).squeeze(-1)
        return self.task_modules['classifier'](pooled).squeeze(-1)

    def region_readout(self, input_ids, attention_mask, window_owners, pooling_weights,
                       function_count, source_positions, region_token_weights):
        """Training/diagnosis only; a complete single version enters the encoder."""
        hidden = self._source_hidden(input_ids, attention_mask)
        local = region_mean(hidden, attention_mask, window_owners, pooling_weights,
                            function_count, source_positions, region_token_weights)
        head = self.task_modules['classifier']
        global_pool = _window_mean(hidden, attention_mask, window_owners, pooling_weights, function_count)
        return head(local).squeeze(-1), local, head(global_pool).squeeze(-1)

def _build_model(variant: str, model_path: str, *, device: torch.device, lora_r: int, lora_alpha: int, lora_dropout: float, target_modules: tuple[str, ...], gradient_checkpointing: bool, fusion_dim: int, fusion_heads: int, graph_config: dict | None=None, classifier_head: str='linear', finetuning_mode: str | None=None, conditional_length_config: dict | None=None):
    if variant != 'codebert_source' or classifier_head != 'linear' or graph_config is not None or conditional_length_config is not None:
        raise ValueError('current factory supports CodeBERT mean pooling and linear head only')
    if finetuning_mode is not None and finetuning_mode not in {'lora', 'full'}:
        raise ValueError('explicit finetuning modes require CodeBERT mean pooling and linear head')
    common = dict(device=device, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules, gradient_checkpointing=gradient_checkpointing)
    return SequenceVulnerabilityClassifier(model_path, variant='codebert_source', classifier_head='linear', **{}, **{'finetuning_mode': finetuning_mode} if finetuning_mode is not None else {}, **common)

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


def expand_vat_groups(values,group_ids):
    if not len(values):return values.new_zeros((*group_ids.shape,values.shape[-1]))
    return values[group_ids.clamp_min(0)]*group_ids.ge(0).unsqueeze(-1)


def normalize_vat_groups(values,group_ids,group_owners,window_owners,function_count,epsilon):
    expanded=expand_vat_groups(values,group_ids)
    # VAT's finite-difference gradient may be tiny but nonzero. Squaring it
    # in float32 can underflow, misstate the radius or discard the direction.
    squared=expanded.new_zeros(function_count,dtype=torch.float64).index_add(
        0,window_owners,expanded.double().square().sum((1,2)))
    norms=squared.sqrt()
    scale=torch.where(norms>0,epsilon/norms.clamp_min(torch.finfo(norms.dtype).tiny),torch.zeros_like(norms))
    return (values.double()*scale[group_owners,None]).to(values.dtype),norms


def bernoulli_vat_kl(reference,perturbed):
    """KL(Bernoulli(p_detached) || Bernoulli(q)); no true-label input."""
    reference=reference.detach().double();perturbed=perturbed.double()
    p=reference.sigmoid()
    return (p*(torch.nn.functional.logsigmoid(reference)-torch.nn.functional.logsigmoid(perturbed))+
        (1-p)*(torch.nn.functional.logsigmoid(-reference)-torch.nn.functional.logsigmoid(-perturbed)))


def vat_loss(model,records,builder,config,*,alpha,device,noise_seed,return_delta=False,diagnose=False):
    if (config!=dict(VAT_CONFIG,xi=config.get('xi')) or config.get('xi') not in (.001,.01)
            or alpha != 0.):
        raise ValueError('global VAT configuration is incompatible')
    batch,group_ids,group_owners,eligible,weights=builder.vat_batch(records,alpha=alpha,device=device)
    ids,mask,owners,pooling,n=batch
    hidden=model.encoder.config.hidden_size
    if not len(group_owners):
        result=torch.zeros(n,device=device)
        info=dict(eligible=0,zero_direction=0,kl=0.,norm_min=None,norm_max=None,
                  function_forwards=0,encoder_sequences=0,encoder_token_slots=0,weighted_source_tokens=0)
        return (result,info,torch.zeros((*ids.shape,hidden),device=device)) if return_delta else (result,info)
    generator=torch.Generator(device=device).manual_seed(noise_seed)
    direction=torch.randn((len(group_owners),hidden),device=device,generator=generator,dtype=torch.float32)
    direction,_=normalize_vat_groups(direction,group_ids,group_owners,owners,n,1.)
    direction=direction.detach().requires_grad_(True)
    with _auxiliary_mode(model):
        with torch.no_grad():reference=model(*batch,force_fp32=True)
        probe=expand_vat_groups(direction,group_ids)*config['xi']
        near=model(*batch,embedding_delta=probe,force_fp32=True)
        divergence=bernoulli_vat_kl(reference,near)
        gradient=torch.autograd.grad(divergence.sum(),direction,only_inputs=True)[0].detach()
        if not torch.isfinite(gradient).all():raise ValueError('nonfinite VAT direction gradient')
        # The tied embedding subspace has metric A^T A = occurrence counts.
        # Autograd sums tied gradients; divide by counts to project the physical
        # token-space gradient, then normalize in the expanded token space.
        counts=torch.bincount(group_ids[group_ids>=0],minlength=len(group_owners)).float()
        gradient=gradient/counts[:,None]
        weighted=gradient*weights[:,None]
        direction,norms=normalize_vat_groups(weighted,group_ids,group_owners,owners,n,config['epsilon'])
        delta=expand_vat_groups(direction.detach(),group_ids)
        perturbed=model(*batch,embedding_delta=delta,force_fp32=True)
        losses=bernoulli_vat_kl(reference,perturbed).float()
        diagnostic={}
        if diagnose:
            baseline=torch.zeros_like(gradient,requires_grad=True)
            logits=model(*batch,embedding_delta=expand_vat_groups(baseline,group_ids),force_fp32=True)
            jacobian=torch.autograd.grad(logits.sum(),baseline,only_inputs=True)[0].detach()/counts[:,None]
            local,local_norm=normalize_vat_groups(jacobian,group_ids,group_owners,owners,n,1.)
            exact=expand_vat_groups(local,group_ids)
            products=(delta.double()*exact.double()).sum((1,2))
            cosine=products.new_zeros(n).index_add(0,owners,products).abs()/config['epsilon']
            random_direction=torch.randn((len(group_owners),hidden),device=device,generator=generator,dtype=torch.float32)
            random_direction,_=normalize_vat_groups(random_direction,group_ids,group_owners,owners,n,config['epsilon'])
            random_delta=expand_vat_groups(random_direction,group_ids)
            with torch.no_grad():random_logits=model(*batch,embedding_delta=random_delta,force_fp32=True)
            random_kl=bernoulli_vat_kl(reference,random_logits)
            diagnostic=dict(reference_logits=reference.detach().cpu().tolist(),probe_logits=near.detach().cpu().tolist(),
                analytical_score_gradient_norm=local_norm.cpu().tolist(),direction_cosine_abs=cosine.cpu().tolist(),
                signal_loss=sum(bool(e) and (float(norms[i])==0 or float(reference[i])==float(near[i].detach()))
                                and float(local_norm[i])>0 for i,e in enumerate(eligible)),
                true_zero=sum(bool(e) and float(local_norm[i])==0 for i,e in enumerate(eligible)),
                random_kl=random_kl.detach().cpu().tolist(),adversarial_kl=losses.detach().cpu().tolist())
    if torch.any(losses < -1e-7) or not torch.isfinite(losses).all():raise ValueError('invalid VAT KL')
    losses=losses.clamp_min(0)*torch.tensor(eligible,device=device)
    squared=delta.double().square().sum((1,2))
    actual=squared.new_zeros(n).index_add(0,owners,squared).sqrt()
    active=actual[actual>0]
    if not torch.allclose(active,active.new_full(active.shape,config['epsilon']),rtol=1e-5,atol=1e-7):
        raise ValueError('expanded VAT perturbations do not match the per-function radius')
    info=dict(eligible=sum(eligible),zero_direction=sum(e and float(norms[i])==0 for i,e in enumerate(eligible)),
        kl=float(losses.detach().sum()),norm_min=float(active.min()) if len(active) else None,
        norm_max=float(active.max()) if len(active) else None,
        function_forwards=(5 if diagnose else 3)*n,encoder_sequences=(5 if diagnose else 3)*len(ids),
        encoder_token_slots=(5 if diagnose else 3)*ids.numel(),weighted_source_tokens=int((weights>1).sum()),**diagnostic)
    return (losses,info,delta.detach()) if return_delta else (losses,info)


def _forward_batch(model, records, input_builder, *, variant, excluded_groups, device, return_local=False):
    _validate_variant(variant, excluded_groups)
    if return_local:raise ValueError('current mainline predicts functions, not local risk')
    return model(*input_builder.codebert_batch(records, device=device))

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
        expanded.extend(dict(record,**(source if isinstance(source,dict) else dict(raw_source=source))) for source in sources)
        coefficients.extend([1./len(sources)] * len(sources))
    for start in range(0, len(expanded), micro_size):
        yield expanded[start:start+micro_size], coefficients[start:start+micro_size]


def train_model(dataset_path: str | Path | None, output_path: str | Path, *,
    variant: str,
    model_path: str,
    source_max_length: int = 1536,
    context_max_length: int = 384,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
    epochs: int = 3,
    learning_rate: float = 0.0002,
    weight_decay: float = 0.01,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    fusion_dim: int = 256,
    fusion_heads: int = 8,
    excluded_groups: tuple[str, ...] = (),
    seed: int = 42,
    device: str = 'auto',
    records: list[dict[str, object]] | None = None,
    log_every: int = 10,
    classifier_head: str = 'linear',
    finetuning_mode: str | None = None,
    robustness_options: dict | None = None,
    surface_materials: dict | None = None,
    repair_supervision: dict | None = None,
    repair_pairs: list[dict] | None = None,
    repair_pairs_path: str | None = None,
) -> dict[str, object]:
    if variant != 'codebert_source' or classifier_head != 'linear' or excluded_groups:
        raise ValueError('current training supports CodeBERT mean pooling and linear BCE only')
    if robustness_options is not None and robustness_options['mode'] not in ROBUSTNESS_MODES:
        raise ValueError('historical robustness modes are inference/report only')
    variant, excluded_groups = _validate_variant('codebert_source', excluded_groups)
    if finetuning_mode is not None and (finetuning_mode not in {'lora', 'full'} or 'codebert_source' != 'codebert_source' or 'linear' != 'linear' or any((x is not None for x in (None, None, None, None)))):
        raise ValueError('explicit finetuning modes require CodeBERT function BCE without auxiliary losses')
    if 'codebert_source' == 'codebert_source' and source_max_length not in (512, 2048):
        raise ValueError('CodeBERT comparison requires logical length 512 or 2048')
    if batch_size <= 0 or gradient_accumulation <= 0 or epochs <= 0:
        raise ValueError('batch_size, gradient_accumulation, and epochs must be positive')
    resolved_device = _resolve_device(device)
    if log_every <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError('invalid log_every, learning_rate or weight_decay')
    if (dataset_path is None) == (records is None):
        raise ValueError('provide exactly one of dataset_path or records')
    records = _read_jsonl(dataset_path, source_only='codebert_source' == 'codebert_source') if records is None else records
    sources = {record_dataset(record) for record in records}
    if len(sources) != 1:
        raise ValueError(f'training dataset view must contain exactly one source, got {sorted(sources)}')
    source_dataset = next(iter(sources))
    if source_dataset == 'sven':
        raise ValueError('SVEN is external-test-only')
    train_records = [record for record in records if record_split(record) == 'train']
    valid_records = [record for record in records if record_split(record) == 'valid']
    if False and len(train_records) != len(records):
        raise ValueError('fixed-epoch fitting accepts training records only')
    if not train_records or (not False and (not valid_records)):
        raise ValueError('formal training requires non-empty train and valid splits')
    if {int(record['label']) for record in train_records} != {0, 1}:
        raise ValueError('training split must contain both labels')
    if not False and {int(record['label']) for record in valid_records} != {0, 1}:
        raise ValueError('validation split must contain both labels')
    if robustness_options is not None:
        _validate_robustness_config(robustness_options)
        if finetuning_mode != 'full' or 'codebert_source' != 'codebert_source' or 'linear' != 'linear' or any((x is not None for x in (None, None, None, None, None, None, None, None, None))) or False or (seed != 42) or (source_max_length != 2048) or (robustness_options['source_views'] and (not isinstance(surface_materials, dict))) or (not robustness_options['source_views'] and surface_materials is not None):
            raise ValueError('robustness requires fresh full CodeBERT; pure VAT has no source views or materials')
    if repair_supervision is not None:
        from .repair_supervision import validate_config, auxiliary_schedule, auxiliary_scale, local_losses, region_readouts
        validate_config(repair_supervision)
        if (finetuning_mode != 'full' or robustness_options is None
                or robustness_options['mode'] != 'hard_aug_bce' or not repair_pairs or not repair_pairs_path):
            raise ValueError('repair supervision requires hard_aug_bce, full CodeBERT and prepared audit pairs')
    elif repair_pairs is not None or repair_pairs_path is not None:
        raise ValueError('audit data require an explicit repair supervision mode')
    graph_config = None
    initial = None
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('tokenizer must define pad_token_id or eos_token_id')
    input_builder = InputBuilder(tokenizer, source_max_length=source_max_length, context_max_length=context_max_length)
    target_modules = ('attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense')
    _seed_everything(seed)
    model = _build_model('codebert_source', model_path, device=resolved_device, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules, gradient_checkpointing=resolved_device.type == 'cuda', fusion_dim=fusion_dim, fusion_heads=fusion_heads, classifier_head='linear', **{}, **{'finetuning_mode': finetuning_mode} if finetuning_mode is not None else {}, **{})
    if robustness_options is not None and resolved_device.type == 'cuda':
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if finetuning_mode is not None:
        _seed_everything(seed)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    best_key = (float('-inf'),) * 4
    best_threshold = 0.5
    best_adapter = None
    best_task = None
    best_validation = None
    selected_epoch = None
    history = TrainingProgress(output_path)
    global_step = 0
    for epoch in range(epochs):
        if None is not None and epoch == 1:
            for parameter in lora_parameters:
                parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW([dict(params=lora_parameters, lr=2e-05), dict(params=head_parameters, lr=0.0001)], weight_decay=0.01)
        model.train()
        order = list(range(len(train_records)))
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        length_total, length_window = (0.0, 0.0)
        vat_total = 0.0
        vat_eligible = 0
        vat_zero = 0
        vat_forwards = 0
        vat_sequences = 0
        vat_slots = 0
        vat_norm_min = None
        vat_norm_max = None
        real_total = 0.0
        real_functions = 0
        real_forwards = 0
        real_sequences = 0
        search_queries = 0
        original_queries = 0
        search_sequences = 0
        search_slots = 0
        search_reasons = {}
        transformed_functions = 0
        search_seconds = 0.0
        epoch_started = None
        if robustness_options is not None:
            import time
            from .augmentation import search_surface_views, surface_seed
            epoch_started = time.monotonic()
            candidate_path = Path(output_path).with_suffix('.candidates.jsonl')
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_total = 0.0
        synthetic_order = []
        random.Random(seed + epoch).shuffle(synthetic_order)
        optimizer_steps = 0
        pending = 0
        num_batches = math.ceil(len(order) / batch_size)
        if repair_supervision is not None:
            audit_steps = math.ceil(num_batches / gradient_accumulation)
            audit_schedule = auxiliary_schedule(repair_pairs, audit_steps, seed=seed, epoch=epoch + 1)
            audit_scale = auxiliary_scale(audit_steps, len(repair_pairs), repair_supervision['lambda_audit'])
            audit_state, audit_contrast, audit_count = 0., 0., 0
            audit_anchors, audit_no_positive, audit_forwards, audit_sequences, audit_slots = 0, 0, 0, 0, 0
            audit_seconds = 0.
        window_loss, window_samples = (0.0, 0)
        view_forwards, encoder_sequences, encoder_token_slots = (0, 0, 0)
        bar = tqdm(range(0, len(order), batch_size), total=num_batches, desc=f"{'codebert_source'} · epoch {epoch + 1}/{epochs}", unit='batch', dynamic_ncols=True, mininterval=1, file=sys.stdout, colour='cyan')
        for batch_index, start in enumerate(bar):
            batch = [train_records[index] for index in order[start:start + batch_size]]
            group_start = batch_index // gradient_accumulation * gradient_accumulation
            group_end = min(group_start + gradient_accumulation, num_batches)
            group_samples = min(group_end * batch_size, len(order)) - group_start * batch_size
            value = 0.0
            length_value = 0.0
            dynamic_views = None
            selections = []
            if robustness_options is not None and robustness_options['source_views']:
                started = time.monotonic()

                def score(rows):
                    nonlocal search_sequences, search_slots
                    with torch.no_grad(), _auxiliary_mode(model):
                        values = []
                        for at in range(0, len(rows), batch_size):
                            grouped = input_builder.codebert_batch(rows[at:at + batch_size], device=resolved_device)
                            search_sequences += len(grouped[0])
                            search_slots += grouped[0].numel()
                            values.extend(model(*grouped).detach().cpu().tolist())
                    return values
                selections = search_surface_views(batch, score, input_builder.tokenizer, source_max_length - len(input_builder.source_prefix) - 2, surface_materials, seed=seed, epoch=epoch + 1, query_limit=robustness_options['train_queries'], random_selection=not robustness_options['search'])
                search_seconds += time.monotonic() - started
                dynamic_views = {r['sample_key']: [dict(raw_source=s['selected']['raw_source'])] if s['selected'] else [] for r, s in zip(batch, selections)}
                transformed_functions += sum((s['selected'] is not None for s in selections))
                search_queries += sum((s['candidate_queries'] for s in selections))
                original_queries += sum((s['original_queries'] for s in selections))
                with candidate_path.open('a', encoding='utf-8') as handle:
                    for selection in selections:
                        for reason, count in selection['reasons'].items():
                            search_reasons[reason] = search_reasons.get(reason, 0) + count
                        chosen = selection['selected']
                        handle.write(json.dumps(dict(sample_key=selection['sample_key'], epoch=epoch + 1, selected=None if chosen is None else {k: v for k, v in chosen.items() if k != 'raw_source'}, visited=[{k: v for k, v in c.items() if k != 'raw_source'} for c in selection['visited']], candidate_queries=selection['candidate_queries'], original_queries=selection['original_queries'], loss_delta=selection['loss_delta'], reasons=selection['reasons'])) + '\n')
            for forwarded, coefficients in function_view_batches(batch, dynamic_views if dynamic_views is not None else None, batch_size):
                prediction = _forward_batch(model, forwarded, input_builder, variant='codebert_source', excluded_groups=excluded_groups, device=resolved_device, return_local=None is not None and None != 'A')
                logits, local_scores = (prediction, None)
                losses = loss_fn(logits, _labels(forwarded, resolved_device))
                loss = losses.mean() if coefficients is None else (losses * losses.new_tensor(coefficients)).sum() / len(batch)
                if not torch.isfinite(loss):
                    raise ValueError(f'non-finite loss in epoch {epoch + 1}, batch {batch_index + 1}')
                training_loss = loss * len(batch) / group_samples
                training_loss.backward()
                value += float(loss.detach().cpu())
                if None is not None or robustness_options is not None:
                    capacity = 512 - len(input_builder.source_prefix) - 2
                    budget = source_max_length - len(input_builder.source_prefix) - 2
                    lengths = [len(input_builder._encode(r['raw_source'], budget)) for r in forwarded]
                    sequences = sum((1 + math.ceil(max(0, n - capacity) / (capacity // 2)) for n in lengths))
                    view_forwards += len(forwarded)
                    encoder_sequences += sequences
                    encoder_token_slots += sequences * min(512, max(lengths) + len(input_builder.source_prefix) + 2)
            if robustness_options is not None and robustness_options['lambda_vat']:
                vat_rows = []
                for i, record in enumerate(batch):
                    selected = selections[i]['selected'] if selections else None
                    rng = random.Random(surface_seed(seed, epoch + 1, record['sample_key'], 'vat-view'))
                    vat_rows.append(dict(record, raw_source=selected['raw_source']) if selected is not None and rng.randrange(2) else record)
                with _auxiliary_mode(model):
                    terms, info = vat_loss(model, vat_rows, input_builder, robustness_options['vat'], alpha=robustness_options['alpha'], device=resolved_device, noise_seed=seed + epoch * len(order) + start)
                    if terms.requires_grad:
                        (robustness_options['lambda_vat'] * terms.sum() / group_samples).backward()
                vat_total += float(terms.detach().sum())
                vat_eligible += info['eligible']
                vat_zero += info['zero_direction']
                vat_forwards += info['function_forwards']
                vat_sequences += info['encoder_sequences']
                vat_slots += info['encoder_token_slots']
                if info['norm_min'] is not None:
                    vat_norm_min = info['norm_min'] if vat_norm_min is None else min(vat_norm_min, info['norm_min'])
                    vat_norm_max = info['norm_max'] if vat_norm_max is None else max(vat_norm_max, info['norm_max'])
            running_loss += value * len(batch)
            length_total += length_value * len(batch)
            length_window += length_value * len(batch)
            window_loss += value * len(batch)
            window_samples += len(batch)
            bar.set_postfix(loss=f'{window_loss / window_samples:.4f}', step=global_step, lr=f'{learning_rate:.1e}', refresh=False)
            pending += 1
            if pending == gradient_accumulation or batch_index + 1 == num_batches:
                if repair_supervision is not None:
                    started = time.monotonic()
                    # Backward remains INSIDE auxiliary mode: checkpointed
                    # recomputation must see the same disabled dropout flags.
                    with _auxiliary_mode(model):
                        for indices in audit_schedule.get(optimizer_steps, []):
                            pairs = [repair_pairs[index] for index in indices]
                            vectors, local_logits, _, cost = region_readouts(model, input_builder, pairs, device=resolved_device)
                            audit_forwards += cost['function_forwards']
                            audit_sequences += cost['encoder_sequences']
                            audit_slots += cost['encoder_token_slots']
                            state, contrast, info = local_losses(vectors, local_logits, pairs,
                                            beta=repair_supervision['beta'], tau=repair_supervision['tau'])
                            auxiliary = audit_scale * (state + repair_supervision['beta'] * contrast).sum()
                            if not torch.isfinite(auxiliary):
                                raise ValueError('non-finite audited local supervision loss')
                            auxiliary.backward()
                            audit_state += float(state.detach().sum())
                            audit_contrast += float(contrast.detach().sum())
                            audit_count += len(pairs)
                            audit_anchors += info['contrast_anchors']
                            audit_no_positive += info['no_positive_anchors']
                    audit_seconds += time.monotonic() - started
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                global_step += 1
                pending = 0
                bar.set_postfix(loss=f'{window_loss / window_samples:.4f}', step=global_step, lr=f'{learning_rate:.1e}', refresh=False)
                if global_step == 1 or global_step % log_every == 0 or batch_index + 1 == num_batches:
                    history.add(dict(event='step', epoch=epoch + 1, step=global_step, loss=window_loss / window_samples, learning_rate=learning_rate, samples_seen=min(start + len(batch), len(order)), **{}, **{}))
                    window_loss, window_samples = (0.0, 0)
                    length_window = 0.0
        validation = None
        threshold = 0.5
        probabilities = _predict(model, valid_records, input_builder, variant='codebert_source', excluded_groups=excluded_groups, batch_size=batch_size, device=resolved_device)
        threshold, validation = _select_validation_threshold(valid_records, probabilities)
        epoch_row = {'event': 'epoch', 'variant': 'codebert_source', 'excluded_groups': list(excluded_groups), 'epoch': epoch + 1, 'train_loss': running_loss / len(train_records), 'optimizer_steps': optimizer_steps, 'validation_threshold': threshold, 'validation': validation.as_json() if validation else None}
        if robustness_options is not None:
            epoch_row.update(augmented_functions=0, bce_function_forwards=view_forwards, bce_encoder_sequences=encoder_sequences, encoder_token_slots=encoder_token_slots, function_loss='one BCE per original function')
            weight = robustness_options['lambda_vat']
            epoch_row.update(robustness_mode=robustness_options['mode'], vat_loss=vat_total / len(train_records), vat_weight=weight, vat_eligible_functions=vat_eligible, vat_skipped_functions=len(train_records) - vat_eligible, vat_zero_direction_functions=vat_zero, vat_norm_min=vat_norm_min, vat_norm_max=vat_norm_max, vat_function_forwards=vat_forwards, vat_encoder_sequences=vat_sequences, vat_encoder_token_slots=vat_slots, real_js=real_total / len(train_records), real_js_functions=real_functions, real_function_forwards=real_forwards, real_encoder_sequences=real_sequences, augmented_functions=transformed_functions, function_loss='mean original/selected BCE per function, then mean over original functions' if robustness_options['source_views'] else 'one BCE per original function', candidate_queries=search_queries, original_search_queries=original_queries, search_encoder_sequences=search_sequences, search_encoder_token_slots=search_slots, search_seconds=search_seconds, epoch_seconds=time.monotonic() - epoch_started, search_reasons=search_reasons, total_loss=(running_loss + weight * vat_total + robustness_options['lambda_real'] * real_total) / len(train_records))
        if repair_supervision is not None:
            if audit_count != len(repair_pairs):
                raise RuntimeError('not every qualified audit pair was used exactly once this epoch')
            epoch_row.update(repair_supervision=repair_supervision['mode'], audit_pairs=audit_count,
                             audit_state_loss=audit_state / audit_count, audit_contrast_loss=audit_contrast / audit_count,
                             audit_contrast_anchors=audit_anchors, audit_no_positive_anchors=audit_no_positive,
                             audit_function_forwards=audit_forwards, audit_encoder_sequences=audit_sequences,
                             audit_encoder_token_slots=audit_slots, audit_seconds=audit_seconds,
                             audit_weight=repair_supervision['lambda_audit'], audit_beta=repair_supervision['beta'],
                             audit_scale_per_update=audit_scale)
            epoch_row['total_loss'] += repair_supervision['lambda_audit'] * (audit_state + repair_supervision['beta'] * audit_contrast) / audit_count
        history.add(epoch_row)
        key = (validation.mcc, validation.f1, validation.accuracy, validation.auc if validation.auc is not None else float('-inf'))
        if key > best_key:
            best_key = key
            best_threshold = threshold
            best_adapter = _cpu_state(model.encoder.state_dict() if finetuning_mode == 'full' else get_peft_model_state_dict(model.encoder))
            best_task = _cpu_state(model.task_modules.state_dict())
            best_validation = validation
            selected_epoch = epoch + 1
    if best_adapter is None or best_task is None:
        raise RuntimeError('training did not produce a checkpoint')
    checkpoint = {'checkpoint_version': _CHECKPOINT_VERSION, 'dataset_schema_version': DATASET_SCHEMA_VERSION, 'trained_on': source_dataset, 'variant': 'codebert_source', 'excluded_groups': excluded_groups, 'model_path': model_path, 'source_max_length': source_max_length, 'context_max_length': context_max_length, 'lora_r': lora_r, 'lora_alpha': lora_alpha, 'lora_dropout': lora_dropout, 'target_modules': target_modules, 'fusion_dim': fusion_dim, 'fusion_heads': fusion_heads, 'decision_threshold': best_threshold, 'selection': 'validation_mcc', 'selected_epoch': selected_epoch, 'seed': seed, 'initial_checkpoint': None, 'sample_weights': None, 'classifier_head': 'linear', 'cohort_members': {split: [(r['sample_key'], r['label']) for r in records if r['split'] == split] for split in ('train', 'valid')}, 'training_config': dict(epochs=epochs, batch_size=batch_size, gradient_accumulation=gradient_accumulation, learning_rate=learning_rate, weight_decay=weight_decay), 'encoder_state' if finetuning_mode == 'full' else 'adapter_state': best_adapter, 'task_state': best_task, 'validation': best_validation.as_json() if best_validation else None, 'trainable_parameters': sum((parameter.numel() for parameter in trainable))}
    if finetuning_mode is not None:
        checkpoint['finetuning_mode'] = finetuning_mode
        checkpoint['encoder_precision'] = 'FP32 weights/optimizer; BF16 CUDA autocast; FP32 head/pooling/loss'
        checkpoint['classifier_initialization'] = _CLASSIFIER_INITIALIZATION
    if robustness_options is not None:
        checkpoint['robustness_options'] = robustness_options
        if surface_materials is not None:
            checkpoint['surface_materials'] = surface_materials
    if repair_supervision is not None:
        checkpoint['repair_supervision'] = repair_supervision
        checkpoint['repair_pairs_path'] = str(repair_pairs_path)
        checkpoint['repair_training_members'] = [dict(sample_id=pair['sample_id'], dataset=pair['dataset'],
                                                      event_id=pair['event_id'], group=pair['group']) for pair in repair_pairs]
    capacity = 512 - len(input_builder.source_prefix) - 2
    checkpoint['input_protocol'] = dict(encoder='roberta', native_window=512, logical_budget=source_max_length, source_budget=source_max_length - len(input_builder.source_prefix) - 2, window_source_capacity=capacity, stride=capacity // 2, pooling='mean over unique source positions; repeated prefix/BOS/EOS averaged across windows', function_loss='mean view BCE per function, then mean over original functions' if None is not None or (robustness_options is not None and robustness_options['source_views']) else 'one BCE per original function')
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(target)
    print(json.dumps({'checkpoint': str(target), 'trained_on': source_dataset, 'variant': 'codebert_source', 'decision_threshold': best_threshold, 'validation': best_validation.as_json() if best_validation else None, 'trainable_parameters': checkpoint['trainable_parameters']}, ensure_ascii=False), flush=True)
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


def _load_model(checkpoint: dict[str, object], *, device: torch.device, include_length_head: bool=False):
    if checkpoint.get('variant') != 'codebert_source' or checkpoint.get('classifier_head', 'linear') != 'linear' or checkpoint.get('conditional_length_config') is not None:
        raise ValueError('historical model architecture requires the archived implementation')
    mode = checkpoint.get('finetuning_mode')
    if checkpoint.get('repair_supervision') is not None:
        from .repair_supervision import validate_config
        validate_config(checkpoint['repair_supervision'])
        members = checkpoint.get('repair_training_members')
        if (mode != 'full' or (checkpoint.get('robustness_options') or {}).get('mode') != 'hard_aug_bce'
                or not isinstance(members, list) or not members or not checkpoint.get('repair_pairs_path')
                or len({(row['dataset'], row['sample_id']) for row in members}) != len(members)):
            raise ValueError('invalid repair-supervised checkpoint metadata')
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
    length_config = checkpoint.get('conditional_length_config')
    if include_length_head:
        raise ValueError('checkpoint has no conditional length head')
    if checkpoint.get('robustness_options') is not None:
        _validate_robustness_config(checkpoint['robustness_options'])
        if mode != 'full' or checkpoint.get('variant') != 'codebert_source' or checkpoint.get('classifier_head') != 'linear' or (checkpoint.get('source_max_length') != 2048) or (checkpoint.get('augmentation_config') is not None) or (bool(checkpoint.get('surface_materials')) != checkpoint['robustness_options']['source_views']):
            raise ValueError('robustness checkpoint input/encoder configuration differs')
        builder = InputBuilder(AutoTokenizer.from_pretrained(checkpoint['model_path']), source_max_length=checkpoint['source_max_length'], context_max_length=checkpoint['context_max_length'])
        capacity = 512 - len(builder.source_prefix) - 2
        protocol = dict(encoder='roberta', native_window=512, logical_budget=2048, source_budget=2048 - len(builder.source_prefix) - 2, window_source_capacity=capacity, stride=capacity // 2, pooling='mean over unique source positions; repeated prefix/BOS/EOS averaged across windows', function_loss='mean view BCE per function, then mean over original functions' if checkpoint['robustness_options']['source_views'] else 'one BCE per original function')
        if checkpoint.get('input_protocol') != protocol:
            raise ValueError('robustness checkpoint input protocol differs from the public builder')
    model = _build_model(str(checkpoint['variant']), str(checkpoint['model_path']), device=device, lora_r=int(checkpoint['lora_r']), lora_alpha=int(checkpoint['lora_alpha']), lora_dropout=float(checkpoint['lora_dropout']), target_modules=tuple((str(value) for value in checkpoint['target_modules'])), gradient_checkpointing=False, classifier_head=str(checkpoint.get('classifier_head', 'linear')), **{}, **{'finetuning_mode': mode} if mode is not None else {}, fusion_dim=int(checkpoint['fusion_dim']), fusion_heads=int(checkpoint['fusion_heads']), **{'graph_config': checkpoint.get('graph_config')} if checkpoint['variant'] in _GRAPH_VARIANTS else {})
    if mode == 'full':
        model.encoder.load_state_dict(checkpoint['encoder_state'], strict=True)
    else:
        _restore_adapter(model.encoder, checkpoint['adapter_state'])
    model.task_modules.load_state_dict(checkpoint['task_state'], strict=True)
    if None is not None and (not include_length_head):
        del model.task_modules['length_adversary']
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
    excluded_groups = tuple(str(value) for value in checkpoint.get('excluded_groups', ()))
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
        from .artifacts import identity
        from .metrics import metrics as binary_metrics, decision_boundary
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
