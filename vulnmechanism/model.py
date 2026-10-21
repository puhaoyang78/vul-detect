"""Full CodeBERT, native sliding windows and one mean-pooled function logit."""

from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
import random
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
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

    def __init__(self, tokenizer, *, source_max_length: int = 2048, relation_mode="none") -> None:
        if source_max_length <= 0:
            raise ValueError("source_max_length must be positive")
        self.tokenizer = tokenizer
        self.relation_mode = relation_mode
        self.relation_cache = {}
        self.source_max_length = source_max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.source_prefix = self._encode(
            "TASK: classify whether this C/C++ function is vulnerable.\nFUNCTION:\n"
        )

    @lru_cache(maxsize=8192)
    def _encode(self, text: str, max_length: int | None = None) -> list[int]:
        result = self.tokenizer(
            text, add_special_tokens=False, truncation=max_length is not None, max_length=max_length
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
                sequences.append([bos, *self.source_prefix, *tokens[start:end], self.eos_token_id])
                weights.append(
                    [1.0 / len(starts)] * (1 + len(self.source_prefix))
                    + [1.0 / coverage[pos] for pos in range(start, end)]
                    + [1.0 / len(starts)]
                )
                owners.append(owner)
                positions.append(
                    [-1] * (1 + len(self.source_prefix)) + list(range(start, end)) + [-1]
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
        if return_source_positions or self.relation_mode != "none":
            source_positions = torch.full_like(ids, -1)
            for i, values in enumerate(positions):
                source_positions[i, : len(values)] = torch.tensor(values, device=device)
            if self.relation_mode != "none":
                from .relations import relation_tokens
                packed = []
                for owner, record in enumerate(records):
                    key = record["sample_key"]
                    if key not in self.relation_cache:
                        self.relation_cache[key] = relation_tokens(record, self.tokenizer, budget, self.relation_mode)
                    packed.extend((owner, a, b, kind) for a, b, kind in self.relation_cache[key][0])
                edges = torch.tensor(packed, device=device, dtype=torch.long).reshape(-1, 4)
                return (*result, source_positions, edges)
            return (*result, source_positions)
        return result

    def vat_batch(self, records, *, device):
        batch = self.codebert_batch(records, device=device, return_source_positions=True)
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
        input_ids = torch.full((len(sequences), width), self.pad_token_id, dtype=torch.long)
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
    """The shared pretrained encoder and fresh linear classification head."""

    def __init__(self, model_path, device, *, gradient_checkpointing=False, relation_mode="none"):
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
        if relation_mode != "none":
            from .relations import RelationPool
            self.task_modules["relations"] = RelationPool(hidden_size)
        self.to(device)

    def forward(
        self,
        input_ids,
        attention_mask,
        window_owners,
        pooling_weights,
        function_count,
        source_positions=None,
        relation_edges=None,
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
        if "relations" in self.task_modules:
            if source_positions is None or relation_edges is None:
                raise ValueError("relation readout requires explicit source positions and edges")
            pooled = pooled + self.task_modules["relations"](
                hidden, source_positions, window_owners, pooling_weights, relation_edges, function_count
            )
        head = self.task_modules["classifier"]
        if force_fp32:
            logits = torch.nn.functional.linear(
                pooled, head.weight.double(), head.bias.double()
            ).squeeze(-1)
        else:
            logits = head(pooled).squeeze(-1)
        return (logits, pooled) if return_representation else logits


@torch.no_grad()
def predict_logits(model, records, builder, *, device, batch_size=4):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    values = []
    for start in training_bar(range(0, len(records), batch_size), desc="Predict"):
        batch = builder.codebert_batch(records[start : start + batch_size], device=device)
        logits = model(*batch)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite prediction logits")
        values.append(logits.float().cpu())
    return torch.cat(values) if values else torch.empty(0)


def load_model(checkpoint, device, *, model_path=None):
    model = CodeBERT(model_path or checkpoint["model_path"], device,
                     relation_mode=(checkpoint.get("static_relations") or {}).get("mode", "none"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def predict_checkpoint(checkpoint_path, records, *, device="auto", batch_size=4, model_path=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = resolve_device(device)
    path = model_path or checkpoint["model_path"]
    builder = InputBuilder(
        AutoTokenizer.from_pretrained(path), source_max_length=checkpoint["source_max_length"],
        relation_mode=(checkpoint.get("static_relations") or {}).get("mode", "none"),
    )
    model = load_model(checkpoint, device, model_path=path)
    return predict_logits(model, records, builder, device=device, batch_size=batch_size)
