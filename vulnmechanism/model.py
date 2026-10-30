"""CodeBERT and UniXcoder encoders with native windows and function pooling."""

from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
import random
import math
import torch
from torch import nn
from types import SimpleNamespace
from transformers import (
    AutoModel,
    AutoTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizerFast,
)
import torch.nn.functional as F
from .progress import training_bar


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


class InputBuilder:

    def __init__(self, tokenizer, *, source_max_length: int = 2048) -> None:
        if source_max_length <= 0:
            raise ValueError("source_max_length must be positive")
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.source_prefix = self._encode(
            "TASK: classify whether this C/C++ function is vulnerable.\nFUNCTION:\n"
        )

    @lru_cache(maxsize=8192)
    def _encode(self, text: str, max_length: int | None = None) -> list[int]:
        result = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=max_length is not None,
            max_length=max_length,
        )
        return list(result["input_ids"])

    def codebert_batch(self, records, *, device, return_source_positions=False):
        """Native 512-token windows, half-window stride, one weighted function pool.

        source_max_length is the logical total budget, including one copy of
        the unchanged task prefix and the two RoBERTa boundary tokens. Each
        source position and each prefix/boundary slot has total weight one.
        """
        if self.source_max_length != 2048:
            raise ValueError("the mainline requires logical length 2048")
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is None or self.eos_token_id is None:
            raise ValueError("CodeBERT requires native BOS/EOS tokens")
        capacity = 512 - len(self.source_prefix) - 2
        budget = self.source_max_length - len(self.source_prefix) - 2
        if capacity <= 0:
            raise ValueError("task prefix leaves no CodeBERT source capacity")
        stride = capacity // 2
        sequences, weights, owners, positions = ([], [], [], [])
        for owner, record in enumerate(records):
            # Dynamic search already validated these IDs together with offsets.
            tokens = record.get("_source_token_ids")
            if tokens is None:
                tokens = self._encode(str(record["raw_source"]), budget)
            elif len(tokens) > budget:
                raise ValueError("cached source tokens exceed the logical budget")
            starts = [0]
            while starts[-1] + capacity < len(tokens):
                starts.append(starts[-1] + stride)
            coverage = [0] * len(tokens)
            for start in starts:
                for pos in range(start, min(start + capacity, len(tokens))):
                    coverage[pos] += 1
            for start in starts:
                end = min(start + capacity, len(tokens))
                sequences.append(
                    [bos, *self.source_prefix, *tokens[start:end], self.eos_token_id]
                )
                weights.append(
                    [1.0 / len(starts)] * (1 + len(self.source_prefix))
                    + [1.0 / coverage[pos] for pos in range(start, end)]
                    + [1.0 / len(starts)]
                )
                owners.append(owner)
                positions.append(
                    [-1] * (1 + len(self.source_prefix))
                    + list(range(start, end))
                    + [-1]
                )
        ids, mask = self._pad(sequences, device=device)
        pool_weights = torch.zeros_like(mask, dtype=torch.float32)
        for i, values in enumerate(weights):
            pool_weights[i, : len(values)] = torch.tensor(values, device=device)
        result = (
            ids,
            mask,
            torch.tensor(owners, dtype=torch.long, device=device),
            pool_weights,
            len(records),
        )
        if return_source_positions:
            source_positions = torch.full_like(ids, -1)
            for i, values in enumerate(positions):
                source_positions[i, : len(values)] = torch.tensor(values, device=device)
            return (*result, source_positions)
        return result

    def vat_batch(self, records, *, device):
        batch = self.codebert_batch(
            records, device=device, return_source_positions=True
        )
        ids, mask, owners, pooling, n, positions = batch
        budget = self.source_max_length - len(self.source_prefix) - 2
        special = set(
            getattr(
                self.tokenizer,
                "all_special_ids",
                [
                    self.pad_token_id,
                    self.eos_token_id,
                    getattr(self.tokenizer, "bos_token_id", None),
                ],
            )
        )
        source_ids = [self._encode(r["raw_source"], budget) for r in records]
        valid = [[t not in special for t in tokens] for tokens in source_ids]
        eligible = [any(row) for row in valid]
        group_rows = [[-1] * ids.shape[1] for _ in range(len(ids))]
        lookup = {}
        group_owners = []
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
                group_rows[window][slot] = lookup[key]
        group_ids = torch.tensor(group_rows, device=device, dtype=torch.long)
        return (
            (ids, mask, owners, pooling, n),
            group_ids,
            torch.tensor(group_owners, device=device, dtype=torch.long),
            eligible,
        )

    def _pad(
        self, sequences: list[list[int]], *, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("cannot create an empty batch")
        width = max((len(sequence) for sequence in sequences))
        input_ids = torch.full(
            (len(sequences), width), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, : len(sequence)] = 1
        return (input_ids.to(device), attention_mask.to(device))


def _window_mean(hidden, mask, owners, weights, function_count, *, dtype=torch.float32):
    if (
        weights.shape != mask.shape
        or hidden.shape[:2] != mask.shape
        or owners.shape != (len(hidden),)
        or function_count <= 0
        or torch.any(owners < 0)
        or torch.any(owners >= function_count)
        or torch.any(weights < 0)
        or torch.any(weights[~mask.bool()] != 0)
    ):
        raise ValueError("invalid function window pooling inputs")
    weights = weights.to(dtype)
    totals = weights.sum(1)
    denominator = totals.new_zeros(function_count).index_add(0, owners, totals)
    if torch.any(denominator <= 0):
        raise ValueError("every function must have visible pooling tokens")
    sums = (hidden.to(dtype) * weights.unsqueeze(-1)).sum(1)
    return (
        sums.new_zeros((function_count, hidden.shape[-1])).index_add(0, owners, sums)
        / denominator[:, None]
    )


@contextmanager
def auxiliary_mode(model):
    """Disable dropout while retaining gradient checkpointing and gradients."""
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    for module, _ in flags:
        if getattr(module, "gradient_checkpointing", False) and torch.is_grad_enabled():
            module.training = True
    try:
        yield
    finally:
        for module, flag in flags:
            module.training = flag


class CodeBERT(nn.Module):
    """The shared pretrained encoder and linear risk head for function classification."""

    def __init__(self, model_path, device, *, gradient_checkpointing=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_path, dtype=torch.float32, add_pooling_layer=False
        )
        self.encoder.config.use_cache = False
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        hidden_size = self.encoder.config.hidden_size
        self.task_modules = nn.ModuleDict({"classifier": nn.Linear(hidden_size, 1)})
        self.to(device)

    def forward(
        self,
        input_ids,
        attention_mask,
        window_owners,
        pooling_weights,
        function_count,
        source_positions=None,
        *,
        embedding_delta=None,
        force_fp32=False,
        return_representation=False,
    ):
        with torch.autocast(
            input_ids.device.type,
            dtype=torch.bfloat16,
            enabled=input_ids.is_cuda and not force_fp32,
        ):
            if embedding_delta is None:
                hidden = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                ).last_hidden_state
            else:
                embeds = self.encoder.get_input_embeddings()(input_ids)
                if (
                    embeds.shape != embedding_delta.shape
                    or not torch.isfinite(embedding_delta).all()
                ):
                    raise ValueError("invalid VAT embedding perturbation")
                pad = self.encoder.config.pad_token_id
                valid = input_ids.ne(pad).long()
                position_ids = valid.cumsum(-1) * valid + pad
                hidden = self.encoder(
                    inputs_embeds=embeds + embedding_delta,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                ).last_hidden_state
        pooled = _window_mean(
            hidden,
            attention_mask,
            window_owners,
            pooling_weights,
            function_count,
            dtype=torch.float64 if force_fp32 else torch.float32,
        )
        head = self.task_modules["classifier"]
        logits = (
            torch.nn.functional.linear(pooled, head.weight.double(), head.bias.double())
            if force_fp32
            else head(pooled)
        ).squeeze(-1)
        return (logits, pooled) if return_representation else logits


@torch.no_grad()
def predict_logits(model, records, builder, *, device, batch_size=4):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    values = []
    for start in training_bar(range(0, len(records), batch_size), desc="Predict"):
        rows = records[start : start + batch_size]
        batch = builder.codebert_batch(rows, device=device)
        logits = model(*batch)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite prediction logits")
        values.append(logits.float().cpu())
    return torch.cat(values) if values else torch.empty(0)


def load_model(checkpoint, device, *, model_path=None):
    model = CodeBERT(model_path or checkpoint["model_path"], device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def predict_checkpoint(
    checkpoint_path, records, *, device="auto", batch_size=4, model_path=None
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") == "losver":
        return predict_paired(checkpoint, records, device=device, model_path=model_path)
    device = resolve_device(device)
    path = model_path or checkpoint["model_path"]
    builder = InputBuilder(
        AutoTokenizer.from_pretrained(path),
        source_max_length=checkpoint["source_max_length"],
    )
    model = load_model(checkpoint, device, model_path=path)
    return predict_logits(model, records, builder, device=device, batch_size=batch_size)


# Adapted from waroad/losver, MIT; see LICENSES/LOSVER.txt.


class LOSVERResidualBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        return x + self.layers(x)


class LOSVERLocalizer(nn.Module):
    def __init__(self, encoder, config, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.args = args
        self.drop_out = getattr(args, "dropout_probability", 0.0)
        if not 0 <= self.drop_out < 1:
            raise ValueError("localizer dropout must lie in [0, 1)")
        self.window_micro_batch = getattr(args, "window_micro_batch", 8)
        self.gradient_checkpointing = getattr(args, "gradient_checkpointing", False)
        self.input_mode = "512"
        self.weak_supervision = False

        # Single attention layer for global context
        self.line_attention = nn.MultiheadAttention(
            self.hidden_size, num_heads=8, batch_first=True, dropout=self.drop_out
        )

        self.line_classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.drop_out),
            LOSVERResidualBlock(  # Optional residual block
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.LayerNorm(self.hidden_size),
                    nn.GELU(),
                    nn.Dropout(self.drop_out),
                )
            ),
            nn.Linear(self.hidden_size, 1),
        )

    @staticmethod
    def focal_loss(logits, labels, *, alpha=0.8, gamma=1.5, pos_weight=5):
        pos_weight = torch.tensor(pos_weight, device=logits.device)

        # BCEWithLogitsLoss applies sigmoid internally
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="none"
        )
        p_t = torch.sigmoid(logits) * labels + (1 - torch.sigmoid(logits)) * (
            1 - labels
        )
        alpha_factor = labels * alpha + (1 - labels) * (1 - alpha)
        modulating_factor = (1.0 - p_t).pow(gamma)
        return alpha_factor * modulating_factor * ce_loss

    @staticmethod
    def symmetric_bernoulli_kl(first, second, valid):
        """Mean (KL12+KL21)/2 over valid independent binary line logits.

        The Bernoulli identity avoids log(0) and preserves BOTH gradients:
        (KL12+KL21)/2 = (sigmoid(z1)-sigmoid(z2)) * (z1-z2) / 2.
        Padding is indexed out before any arithmetic, not treated as a class.
        """
        if first.shape != second.shape or first.shape != valid.shape:
            raise ValueError("line logits and valid mask must have equal shapes")
        a, b = first[valid.bool()], second[valid.bool()]
        if not a.numel():
            return a.sum() + b.sum()
        return ((a.sigmoid() - b.sigmoid()) * (a - b) * 0.5).mean()

    def forward(
        self,
        input_ids,
        attention_mask,
        code_mask,
        line_token_mask,
        context_mask,
        line_indices,
        line_counts,
        window_owners,
        labels=None,
        label_weights=None,
        return_logits=False,
    ):
        from torch.utils.checkpoint import checkpoint

        def encode(ids, attention, context_valid, membership):
            states = self.encoder(ids, attention_mask=attention)[0]
            parts = []
            for hidden, valid, lines in zip(states, context_valid, membership):
                occupied = lines.any(-1)
                if not occupied.any():
                    continue
                context = hidden[valid]
                contextual, _ = self.line_attention(
                    context[None], context[None], context[None]
                )
                core = lines[occupied][:, valid].to(hidden.dtype)
                parts.append(torch.cat((core @ context, core @ contextual[0]), -1))
            return (
                torch.cat(parts)
                if parts
                else states.new_zeros((0, self.hidden_size * 2))
            )

        total_lines = int(line_counts.sum())
        sums = self.line_classifier[0].weight.new_zeros(
            (total_lines, self.hidden_size * 2)
        )
        counts = sums.new_zeros(total_lines)
        for start in range(0, len(input_ids), self.window_micro_batch):
            end = start + self.window_micro_batch
            args = (
                input_ids[start:end],
                attention_mask[start:end],
                context_mask[start:end],
                line_token_mask[start:end],
            )
            features = (
                checkpoint(encode, *args, use_reentrant=False)
                if self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
                else encode(*args)
            )
            valid_lines = line_indices[start:end] >= 0
            indices = line_indices[start:end][valid_lines]
            sums = sums.index_add(0, indices, features)
            counts = counts.index_add(
                0,
                indices,
                line_token_mask[start:end].sum(-1)[valid_lines].to(counts.dtype),
            )
        if not (counts > 0).all():
            raise ValueError("physical line has no core source-token membership")
        z = self.line_classifier(sums / counts[:, None]).squeeze(-1)
        outputs = list((z if return_logits else z.sigmoid()).split(line_counts.tolist()))
        if labels is not None:
            y = torch.cat([row[:n] for row, n in zip(labels, line_counts.tolist())])
            if label_weights is not None:
                weights = torch.cat(
                    [row[:n] for row, n in zip(label_weights, line_counts.tolist())]
                )
                valid = (weights > 0) & (y >= 0)
                if (
                    not ((y == 0) | (y == 1) | (y == -1)).all()
                    or not (weights >= 0).all()
                ):
                    raise ValueError(
                        "region labels must be 0/1/-1 with nonnegative weights"
                    )
                # The same symmetric focal definition for BOTH weak-supervision
                # localizers. Unknown lines have zero gradient, not negative labels.
                w = weights * valid
                per_line = self.focal_loss(z, y.clamp(0, 1), alpha=0.5, pos_weight=1)
                loss = (per_line * w).sum() / w.sum().clamp_min(1e-12)
                return loss, outputs
            if not ((y == 0) | (y == 1)).all():
                raise ValueError("visible physical lines require binary labels")
            return self.focal_loss(z, y.float()).mean(), outputs
        return outputs


class LOSVERDetector(nn.Module):
    """One code-only pooling and linear head for baseline, predicted and Oracle."""

    def __init__(self, encoder, config, args):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.class_weight = args.class_weight
        self.window_micro_batch = getattr(args, "window_micro_batch", 8)
        self.gradient_checkpointing = getattr(args, "gradient_checkpointing", False)
        self.input_mode = "512"
        self.pooling = "bounded_attention" if getattr(args, "bounded_attention", False) else "hard_weight"
        self.fusion_beta = 0.25
        self.attention_region = getattr(args, "attention_region", None)
        self.force_uniform_attention = False
        if self.pooling == "bounded_attention":
            # Preserve encoder/classifier initialization and all sampler RNG.
            with torch.random.fork_rng(devices=[]):
                self.line_attention = nn.Linear(config.hidden_size, 1)
            nn.init.zeros_(self.line_attention.weight)
            nn.init.zeros_(self.line_attention.bias)
        self.line_auxiliary_weight = getattr(args, "line_auxiliary_weight", None)
        if self.line_auxiliary_weight is not None:
            if not math.isfinite(self.line_auxiliary_weight) or self.line_auxiliary_weight < 0:
                raise ValueError("line auxiliary weight must be finite and nonnegative")
            # Keep encoder/head initialization AND the original sampler RNG.
            with torch.random.fork_rng(devices=[]):
                self.line_classifier = nn.Linear(config.hidden_size, 1)

    @staticmethod
    def pool(states, code_mask, token_weights):
        code_mask = code_mask.bool()
        if not code_mask.any(-1).all():
            raise ValueError("detector input has no valid source tokens")
        attention = token_weights.masked_fill(~code_mask, float("-inf")).softmax(-1)
        return (states * attention.unsqueeze(-1)).sum(1)

    @staticmethod
    def robust_pool(states, code_mask, region_scores, beta):
        if not 0 <= beta <= 0.25:
            raise ValueError("fusion beta must lie in [0,0.25]")
        valid = code_mask.to(states.dtype)
        count = valid.sum(-1, keepdim=True)
        if not (count > 0).all():
            raise ValueError("detector input has no valid source tokens")
        global_mean = (states * valid[..., None]).sum(1) / count
        # Scores are focal outputs, NOT calibrated probabilities. Fixed .5 gate
        # and excess-score attenuation prevent tiny positives getting full beta.
        s = ((region_scores - 0.5) / 0.5).clamp(0, 1) * valid
        mass = s.sum(-1, keepdim=True)
        local = (states * s[..., None]).sum(1) / mass.clamp_min(1e-12)
        strength = s.amax(-1, keepdim=True)
        result = global_mean + beta * strength * (local - global_mean)
        return torch.where(mass > 0, result, global_mean)

    def bounded_pool(self, states, code_mask, line_token_mask):
        """One shared encoding; token-count-weighted physical-line MIL, K=5."""
        membership = (line_token_mask & code_mask[:, None].bool()).to(states.dtype)
        copies = membership.sum(1)
        if not torch.equal(copies > 0, code_mask.bool()):
            raise ValueError("line mapping must cover every valid code token exactly in aggregate")
        # Existing offset mapping can share a cross-line token. Conserve its
        # single token mass rather than silently double-counting it. For normal
        # one-line tokens this is exactly the ordinary mean/count definition.
        membership = membership / copies.clamp_min(1)[:, None]
        counts = membership.sum(-1)
        lines = torch.bmm(membership, states) / counts.clamp_min(1e-12)[..., None]
        rates = 1 + 4 * self.line_attention(lines).squeeze(-1).sigmoid()
        if self.force_uniform_attention:
            rates = torch.ones_like(rates)
        rates = rates.masked_fill(counts == 0, 0)
        mass = counts * rates
        attention = mass / mass.sum(-1, keepdim=True)
        pooled = (lines * attention[..., None]).sum(1)
        token_attention = (membership * (attention / counts.clamp_min(1e-12))[..., None]).sum(1)
        return pooled, dict(line_rates=rates, line_counts=counts,
                            line_attention=attention, token_attention=token_attention)

    @staticmethod
    def attention_region_loss(rates, counts, labels):
        """Mean over eligible functions; zero/all-token regions keep all BCE."""
        selected = (labels == 1).to(rates.dtype)
        region = counts * selected
        background = counts * (1 - selected)
        nr, nb = region.sum(-1), background.sum(-1)
        valid = (nr > 0) & (nb > 0)
        mean_r = (region * rates).sum(-1) / nr.clamp_min(1e-12)
        mean_b = (background * rates).sum(-1) / nb.clamp_min(1e-12)
        margin = F.relu(math.log(2) - mean_r.clamp_min(1e-12).log() + mean_b.clamp_min(1e-12).log())
        loss = margin[valid].mean() if valid.any() else rates.sum() * 0
        return loss, valid.sum()

    def forward(
        self,
        input_ids,
        attention_mask,
        code_mask,
        token_weights,
        window_owners,
        labels=None,
        region_scores=None,
        line_token_mask=None,
        line_indices=None,
        line_counts=None,
        line_labels=None,
        return_auxiliary=False,
        return_attention=False,
    ):
        from torch.utils.checkpoint import checkpoint

        if return_auxiliary and (
            self.line_auxiliary_weight is None or self.input_mode != "512"
            or self.pooling != "hard_weight" or line_labels is None or labels is None
        ):
            raise ValueError("line auxiliary training requires labeled 512 baseline inputs")
        bounded = self.pooling == "bounded_attention"
        if bounded and (self.input_mode != "512" or line_token_mask is None):
            raise ValueError("bounded attention requires the shared 512 physical-line map")
        if return_attention and not bounded:
            raise ValueError("attention diagnostics require bounded attention")

        def encode(ids, attention, valid, weights):
            states = self.encoder(ids, attention_mask=attention)[0]
            if bounded:
                return states
            occupied = valid.any(-1)
            pooled = states.new_zeros((len(states), states.shape[-1]))
            if occupied.any():
                pooled = pooled.index_copy(
                    0,
                    occupied.nonzero().flatten(),
                    (
                        self.robust_pool(
                            states[occupied],
                            valid[occupied],
                            weights[occupied],
                            self.fusion_beta,
                        )
                        if self.pooling == "robust_fusion"
                        else self.pool(
                            states[occupied], valid[occupied], weights[occupied]
                        )
                    ),
                )
            return (pooled, states) if return_auxiliary else pooled

        parts = []
        token_states = []
        for start in range(0, len(input_ids), self.window_micro_batch):
            end = start + self.window_micro_batch
            args = (
                input_ids[start:end],
                attention_mask[start:end],
                code_mask[start:end],
                (region_scores if self.pooling == "robust_fusion" else token_weights)[
                    start:end
                ],
            )
            encoded = (
                checkpoint(encode, *args, use_reentrant=False)
                if self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
                else encode(*args)
            )
            if return_auxiliary:
                token_states.append(encoded[1])
                encoded = encoded[0]
            parts.append(encoded)
        windows = torch.cat(parts)
        function_count = int(window_owners.max()) + 1
        if bounded:
            if len(windows) != function_count or not torch.equal(
                window_owners, torch.arange(function_count, device=window_owners.device)
            ):
                raise ValueError("bounded attention uses one 512-token window per function")
            pooled, attention_details = self.bounded_pool(windows, code_mask, line_token_mask)
        elif len(windows) == function_count and torch.equal(
            window_owners, torch.arange(function_count, device=window_owners.device)
        ):
            pooled = windows
        else:
            if self.pooling == "robust_fusion":
                raise ValueError(
                    "robust fusion currently supports the 512 setting only"
                )
            # Reuse the native sliding-window function pool. Window mass makes
            # this exactly one global Softmax over all unique core code tokens.
            mass = (
                (token_weights.masked_fill(~code_mask, float("-inf")) - 5).exp().sum(-1)
            )
            pooled = _window_mean(
                windows[:, None],
                (mass > 0)[:, None],
                window_owners,
                mass[:, None],
                function_count,
            )
        logits = self.classifier(pooled)
        probability = logits.sigmoid()
        if labels is not None:
            y = labels.float()
            weights = y * self.class_weight + (1 - y)
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], y, weight=weights)
            if bounded:
                region_loss, effective = loss * 0, logits.new_zeros((), dtype=torch.long)
                if self.attention_region is not None:
                    if line_labels is None:
                        raise ValueError("region training requires original or fixed random line labels")
                    region_loss, effective = self.attention_region_loss(
                        attention_details["line_rates"], attention_details["line_counts"], line_labels
                    )
                attention_details.update(bce=loss, region_loss=region_loss, region_functions=effective,
                                         logits=logits[:, 0])
                total = loss + (0.1 * region_loss if self.attention_region is not None else 0)
                return (total, probability, attention_details) if return_attention else (total, probability)
            if return_auxiliary:
                states = torch.cat(token_states)
                membership = line_token_mask.to(states.dtype)
                valid = line_indices >= 0
                indices = line_indices[valid]
                total_lines = int(line_counts.sum())
                sums = states.new_zeros((total_lines, states.shape[-1])).index_add(
                    0, indices, torch.bmm(membership, states)[valid]
                )
                counts = states.new_zeros(total_lines).index_add(
                    0, indices, membership.sum(-1)[valid]
                )
                if not (counts > 0).all():
                    raise ValueError("auxiliary line has no source-token membership")
                z = self.line_classifier(sums / counts[:, None]).squeeze(-1)
                ns = line_counts.tolist()
                targets = torch.cat([row[:n] for row, n in zip(line_labels, ns)])
                if not ((targets == 0) | (targets == 1)).all():
                    raise ValueError("Diff auxiliary labels must be binary on visible lines")
                per_line = LOSVERLocalizer.focal_loss(z, targets.float())
                focal = torch.stack([x.mean() for x in per_line.split(ns)]).mean()
                return loss + self.line_auxiliary_weight * focal, probability, dict(
                    bce=loss, line_focal=focal,
                    line_probabilities=list(z.sigmoid().split(ns)),
                )
            return loss, probability
        if bounded and return_attention:
            return probability, dict(attention_details, logits=logits[:, 0])
        return probability


def losver_model(
    model_path, variant, *, window_micro_batch=8, gradient_checkpointing=False,
    dropout_probability=0.0,
    line_auxiliary_weight=None,
    bounded_attention=False,
    attention_region=None,
):
    """Shared UniXcoder encoder and source-offset tokenizer; no task prefix."""
    config = RobertaConfig.from_pretrained(model_path)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False)
    if window_micro_batch < 1:
        raise ValueError("window micro-batch must be positive")
    if bounded_attention and (variant != "baseline" or line_auxiliary_weight is not None):
        raise ValueError("bounded attention uses only the original baseline encoder/classifier")
    if attention_region is not None and (not bounded_attention or attention_region not in ("diff", "random")):
        raise ValueError("region supervision requires bounded attention and diff/random")
    args = SimpleNamespace(
        class_weight=1.0,
        dropout_probability=dropout_probability,
        window_micro_batch=window_micro_batch,
        gradient_checkpointing=gradient_checkpointing,
        line_auxiliary_weight=line_auxiliary_weight,
        bounded_attention=bounded_attention,
        attention_region=attention_region,
    )
    encoder = RobertaModel.from_pretrained(
        model_path, config=config, attn_implementation="eager", add_pooling_layer=False
    )
    model_cls = LOSVERLocalizer if variant == "localizer" else LOSVERDetector
    if variant not in ("localizer", "baseline", "predicted", "oracle"):
        raise ValueError(f"unknown LOSVER variant: {variant}")
    return model_cls(encoder, config, args), tokenizer


@torch.no_grad()
def predict_paired(checkpoint, records, *, device="auto", model_path=None):
    """Single-function localization/detection, sharing one tokenization and map."""
    from .data import paired_encoding, paired_features
    from .training import paired_collate, paired_forward
    from .evaluation import load_paired

    device = resolve_device(device)
    variant = checkpoint["variant"]
    if variant not in ("baseline", "predicted"):
        raise ValueError(
            "single-function detection uses baseline or predicted lines; Oracle requires Diff"
        )
    if model_path is not None:
        checkpoint = dict(checkpoint, model_path=model_path)
    model, tokenizer = load_paired(checkpoint, device)
    localizer = None
    if variant == "predicted":
        cp = torch.load(
            checkpoint["localizer_path"], map_location="cpu", weights_only=False
        )
        if model_path is not None:
            cp = dict(cp, model_path=model_path)
        localizer, _ = load_paired(cp, device)
        if cp["input_mode"] != checkpoint["input_mode"]:
            raise ValueError("localizer and detector input modes differ")
    result = []
    for r in records:
        row = dict(func=r["raw_source"], target=0)
        encoding = paired_encoding(row["func"], tokenizer, checkpoint["input_mode"])
        if localizer is not None:
            f = paired_features(row, tokenizer, "localizer", encoding=encoding)
            scores = paired_forward(
                localizer, paired_collate([f]), "localizer", device, with_labels=False
            )[0]
            row.update(
                line_numbers=encoding["line_numbers"],
                line_scores=scores.cpu().tolist(),
                lines_predicted=[
                    n for n, p in zip(encoding["line_numbers"], scores) if p > 0.5
                ],
            )
        f = paired_features(row, tokenizer, variant, encoding=encoding)
        probability = paired_forward(
            model, paired_collate([f]), variant, device, with_labels=False
        )[:, 0]
        result.append(torch.logit(probability).cpu())
    return torch.cat(result)
