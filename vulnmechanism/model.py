"""Shared CodeBERT with native sliding windows and mean pooling."""

from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
import random
import torch
from torch import nn
from types import SimpleNamespace
from transformers import (
    AutoModel,
    AutoTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,
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
    def __init__(self, encoder, config, tokenizer, args):
        super(LOSVERLocalizer, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.hidden_size = self.config.hidden_size
        self.args = args
        self.drop_out = 0.0

        self.newline_token_id = self.tokenizer.convert_tokens_to_ids("\n")

        # If tokenizer doesn't recognize '\n' directly, try alternate encodings
        if self.newline_token_id == self.tokenizer.unk_token_id:
            # Try with special token format
            self.newline_token_id = self.tokenizer.convert_tokens_to_ids(
                "Ċ"
            )  # Some tokenizers use this

            # If still not found, you can try to encode a string with a newline and check
            if self.newline_token_id == self.tokenizer.unk_token_id:
                newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
                if newline_tokens:  # If not empty
                    self.newline_token_id = newline_tokens[0]

        print(f"Newline token ID: {self.newline_token_id}")
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

    def focal_loss(self, logits, labels):
        pos_weight = torch.tensor(5, device=logits.device)

        # BCEWithLogitsLoss applies sigmoid internally
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="none"
        )
        p_t = torch.sigmoid(logits) * labels + (1 - torch.sigmoid(logits)) * (
            1 - labels
        )
        alpha = 0.8
        gamma = 1.5
        alpha_factor = labels * alpha + (1 - labels) * (1 - alpha)
        modulating_factor = (1.0 - p_t).pow(gamma)
        return alpha_factor * modulating_factor * ce_loss

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        line_logits = []
        for i in range(input_ids.size(0)):  # Batch loop
            line_indices = (input_ids[i] == self.newline_token_id).nonzero(
                as_tuple=True
            )[
                0
            ]  # 30790
            line_start_indices = torch.cat(
                [torch.tensor([0], device=input_ids.device), line_indices + 1]
            )
            code_end = line_start_indices[-1]  # Last line start is end of code

            sample_line_logits = []
            line_features, feature_slots = [], []
            global_context = outputs[i, :code_end]
            # The official MHA has dropout=0 and independent queries. Compute the
            # same context attention once per function, then pool each line's queries.
            # Newline queries are unused; keys/values and every line boundary match upstream.
            if code_end > 0:
                contextual, _ = self.line_attention(
                    global_context.unsqueeze(0),
                    global_context.unsqueeze(0),
                    global_context.unsqueeze(0),
                )
                contextual = contextual.squeeze(0)
            for j in range(len(line_start_indices) - 1):
                start = line_start_indices[j]
                end = line_start_indices[j + 1] - 1

                if not end > start:
                    sample_line_logits.append(
                        torch.tensor(-10, device=input_ids.device)
                    )
                    continue

                line_repr = outputs[i, start:end]
                line_final = torch.cat(
                    [
                        line_repr.mean(0),
                        contextual[start:end].mean(0),
                    ]
                )
                feature_slots.append(len(sample_line_logits))
                line_features.append(line_final)
                sample_line_logits.append(None)

            if line_features:
                # LayerNorm/GELU/zero-dropout classifier acts independently on each line.
                classified = self.line_classifier(torch.stack(line_features)).squeeze(
                    -1
                )
                for slot, logit in zip(feature_slots, classified):
                    sample_line_logits[slot] = logit
            if sample_line_logits:
                line_logits.append(torch.stack(sample_line_logits))
            else:
                line_logits.append(torch.zeros(1, device=input_ids.device))

        # Loss computation with focal loss
        if labels is not None:
            loss = 0.0
            total_valid = 0

            for i, logit in enumerate(line_logits):
                valid_length = min(logit.size(0), labels[i].size(0))
                valid_mask = (labels[i, :valid_length] != -1).float()
                valid_labels = labels[i, :valid_length].clamp(0, 1).float()
                focal_loss = self.focal_loss(
                    logit[:valid_length],
                    valid_labels,
                )
                loss += (focal_loss * valid_mask).sum()
                total_valid += valid_mask.sum()

            loss = loss / total_valid if total_valid > 0 else loss * 0.0
            line_logits = [torch.sigmoid(logits) for logits in line_logits]
            return loss, line_logits

        return [torch.sigmoid(logits) for logits in line_logits]


class LOSVERBaseline(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(LOSVERBaseline, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        outputs = self.dropout(outputs)

        logits = outputs
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            weight = labels * self.args.class_weight + (1 - labels) * 1.0
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log(
                (1 - prob)[:, 0] + 1e-10
            ) * (1 - labels)
            loss = loss * weight
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


class LOSVERDetector(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(LOSVERDetector, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        # Get hidden size from encoder's output

    def get_line_boundaries(self, input_ids, batch_idx):
        """Extract line boundary information from input tokens"""
        line_boundaries = []
        current_start = 0

        for pos, token_id in enumerate(input_ids[batch_idx].tolist()):
            token = self.tokenizer.convert_ids_to_tokens([token_id])[0]

            if token == self.tokenizer.sep_token or token == self.tokenizer.pad_token:
                if current_start < pos:
                    line_boundaries.append((current_start, pos))
                break

            if "\n" in token or token == "<EOL>" or token.startswith("Ċ"):
                line_boundaries.append((current_start, pos + 1))
                current_start = pos + 1

        return line_boundaries

    def forward(
        self, input_ids=None, labels=None, attention_mask=None, line_indices=None
    ):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)[
            0
        ]  # Use attention_mask instead of mask
        hidden_states = outputs
        batch_size, seq_len = input_ids.shape
        weight_value = 5

        # Initialize line weights
        line_weights = torch.ones((batch_size, seq_len), device=hidden_states.device)
        if line_indices is not None:
            for batch_idx in range(batch_size):
                line_boundaries = self.get_line_boundaries(input_ids, batch_idx)

                if line_indices[batch_idx]:
                    for line_idx, (start, end) in enumerate(line_boundaries):
                        if line_idx in line_indices[batch_idx]:
                            line_weights[batch_idx, start:end] = weight_value
            # Apply attention mask and normalize

        line_weights = line_weights.masked_fill(~attention_mask.bool(), 0)
        attn_scores = torch.softmax(line_weights, dim=1).unsqueeze(-1)
        # Apply weights and sum
        weighted_hidden = hidden_states * attn_scores
        logits = weighted_hidden.sum(dim=1) / (attn_scores.sum(dim=1) + 1e-9)

        prob = torch.sigmoid(logits)

        if labels is not None:
            labels = labels.float()
            weight = labels * self.args.class_weight + (1 - labels) * 1.0
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log(
                (1 - prob)[:, 0] + 1e-10
            ) * (1 - labels)
            loss = loss * weight
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


def losver_model(model_path, variant):
    """Official heads and eager FP32 encoder; no task prefix or CodeBERT head."""
    config = RobertaConfig.from_pretrained(model_path)
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    args = SimpleNamespace(class_weight=1.0, dropout_probability=0.0)
    cls = RobertaForSequenceClassification if variant == "baseline" else RobertaModel
    encoder = cls.from_pretrained(
        model_path, config=config, attn_implementation="eager"
    )
    model_cls = {
        "localizer": LOSVERLocalizer,
        "baseline": LOSVERBaseline,
        "predicted": LOSVERDetector,
        "oracle": LOSVERDetector,
    }[variant]
    return model_cls(encoder, config, tokenizer, args), tokenizer


@torch.no_grad()
def predict_paired(checkpoint, records, *, device="auto", model_path=None):
    """Same official detector/localizer encoding, with one raw function at inference."""
    from .data import paired_features

    device = resolve_device(device)
    variant = checkpoint["variant"]
    if variant in ("oracle", "localizer"):
        raise ValueError(
            "single-function detection uses the baseline or predicted-line LOSVER; Oracle requires Diff"
        )
    path = model_path or checkpoint["model_path"]
    model, tokenizer = losver_model(path, variant)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    localizer = None
    if variant == "predicted":
        cp = torch.load(
            checkpoint["localizer_path"], map_location="cpu", weights_only=False
        )
        localizer, _ = losver_model(path, "localizer")
        localizer.load_state_dict(cp["state_dict"], strict=True)
        localizer.to(device).eval()
    result = []
    for r in records:
        row = dict(func=r["raw_source"], target=0)
        if localizer is not None:
            f = paired_features(row, tokenizer, "localizer")
            scores = localizer(input_ids=torch.tensor([f["input_ids"]], device=device))[
                0
            ]
            row["lines_predicted"] = [i for i, p in enumerate(scores) if p > 0.5]
        f = paired_features(row, tokenizer, variant)
        ids = torch.tensor([f["input_ids"]], device=device)
        if variant == "baseline":
            prob = model(input_ids=ids)[:, 0]
        else:
            prob = model(
                input_ids=ids,
                attention_mask=torch.tensor([f["attention_mask"]], device=device),
                line_indices=[f["line_indices"]],
            )[:, 0]
        # Return exact effective logits for the common CLI sigmoid probability output.
        result.append(torch.logit(prob).cpu())
    return torch.cat(result)
