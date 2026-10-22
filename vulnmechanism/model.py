"""Shared CodeBERT with native sliding windows and mean or line-attention readout."""

from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
import random
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as checkpoint_forward
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

    def __init__(self, tokenizer, *, source_max_length: int = 2048, relation_mode="none", full_source=False) -> None:
        if source_max_length <= 0:
            raise ValueError("source_max_length must be positive")
        self.tokenizer = tokenizer
        self.relation_mode = relation_mode
        self.relation_cache = {}
        self.source_max_length = source_max_length
        self.full_source = full_source
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
        full_source bypasses only the prefix budget: native windows and pooling
        remain identical, and no pair/diff information selects input windows.
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
        sequences, weights, owners, positions, spans = ([], [], [], [], [])
        for owner, record in enumerate(records):
            # Dynamic search already validated these IDs together with offsets.
            tokens = record.get("_source_token_ids")
            if tokens is None:
                tokens = self._encode(str(record["raw_source"]), None if self.full_source else budget)
            elif not self.full_source and len(tokens) > budget:
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
                spans.append((owner, start, end))
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
                prefix = len(self.source_prefix) + 1  # BOS and task prefix
                grouped_windows = {}
                for window_index, (owner, start, end) in enumerate(spans):
                    grouped_windows.setdefault(owner, []).append((window_index, start, end))
                for owner, record in enumerate(records):
                    key = record["sample_key"]
                    if key not in self.relation_cache:
                        self.relation_cache[key] = relation_tokens(
                            record, self.tokenizer, budget, self.relation_mode
                        )
                    pairs, info = self.relation_cache[key]
                    mapped = set()
                    copies = 0
                    for window_index, start, end in grouped_windows[owner]:
                        for a, b, kind in pairs:
                            if start <= a < end and start <= b < end:
                                packed.append((window_index, prefix + a - start,
                                               prefix + b - start, kind))
                                mapped.add((a, b, kind))
                                copies += 1
                    info["pairs_in_shared_windows"] = len(mapped)
                    info["pairs_outside_shared_windows"] = len(set(pairs) - mapped)
                    info["window_edge_instances"] = copies
                if self.relation_mode == "shuffled":
                    from .relations import shuffle_window_edges
                    packed, changed = shuffle_window_edges(packed, spans, records)
                    for owner, record in enumerate(records):
                        self.relation_cache[record["sample_key"]][1]["changed_endpoints"] = \
                            changed[owner]
                edges = torch.tensor(packed, device=device, dtype=torch.long).reshape(-1, 4)
                return (*result, source_positions, edges)
            return (*result, source_positions)
        return result

    def line_mapping(self, records, batch):
        """Diff-independent token-to-line indices for the very same windows."""
        from .pair_representation import attach_line_metadata
        ids, mask, owners, weights, count, positions = batch
        mapping = torch.full_like(ids, -1)
        line_owners, offset = [], 0
        budget = None if self.full_source else self.source_max_length-len(self.source_prefix)-2
        for owner, row in enumerate(records):
            if "_token_lines" not in row:
                row = dict(row)
                encoded = self.tokenizer(row["raw_source"], add_special_tokens=False,
                                         truncation=budget is not None, max_length=budget,
                                         return_offsets_mapping=True)
                attach_line_metadata(row, encoded["input_ids"], encoded["offset_mapping"],
                                     set(self.tokenizer.all_special_ids))
            numbers = row["_line_numbers"]
            if not numbers:
                raise ValueError("line MIL requires mapped nonempty source lines")
            windows = owners == owner
            local_positions = positions[windows]
            lookup = torch.tensor(row["_token_lines"], device=ids.device)
            local_lines = lookup[local_positions.clamp_min(0)]
            local_lines = torch.where((local_positions >= 0) & (local_lines >= 0),
                                      local_lines + offset, -1)
            mapping[windows] = local_lines
            line_owners.extend([owner] * len(numbers))
            offset += len(numbers)
        return dict(line_ids=mapping, line_owners=torch.tensor(line_owners, device=ids.device))

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
    """The shared pretrained encoder and linear risk head, with optional line MIL."""

    def __init__(self, model_path, device, *, gradient_checkpointing=False, relation_mode="none", readout="mean"):
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
        if readout not in ("mean", "line_mil"):
            raise ValueError("unknown function readout")
        self.readout = readout
        if readout == "line_mil":
            self.task_modules.update(dict(
                attention_v=nn.Linear(hidden_size, 128),
                attention_u=nn.Linear(hidden_size, 128),
                attention_score=nn.Linear(128, 1, bias=False),
            ))
        self._active_relation_edges = None
        self._relation_hooks = []
        if relation_mode != "none":
            from .relations import StaticRelationEncoder
            blocks = getattr(getattr(self.encoder, "encoder", None), "layer", None)
            if blocks is None or not len(blocks):
                raise ValueError("static relations require a RoBERTa/CodeBERT encoder")
            n = len(blocks)
            selected_layers = sorted({max(0, n // 3 - 1), max(0, (2 * n) // 3 - 1)})
            self.task_modules["relations"] = StaticRelationEncoder(hidden_size, selected_layers)

            def _static_layer_hook(_layer, _inputs, output, *, index):
                # The hooked hidden state feeds the NEXT transformer layer.
                # Keep the relation batch available for gradient checkpoint replay.
                if self._active_relation_edges is None:
                    return output
                first = output[0] if isinstance(output, (tuple, list)) else output
                changed = self.task_modules["relations"](
                    first, self._active_relation_edges, index
                )
                if isinstance(output, tuple):
                    return (changed, *output[1:])
                if isinstance(output, list):
                    return [changed, *output[1:]]
                return changed

            for index in selected_layers:
                self._relation_hooks.append(
                    blocks[index].register_forward_hook(
                        lambda module, inputs, result, layer_id=index:
                            _static_layer_hook(module, inputs, result, index=layer_id)
                    )
                )
        self.to(device)

    def _forward_lines(self, input_ids, mask, owners, weights, count, line_ids, line_owners,
                       *, window_batch_size=8, return_line_scores=False, return_representation=False):
        if line_ids is None or line_owners is None or line_ids.shape != input_ids.shape:
            raise ValueError("line MIL requires aligned source-only line indices")
        n = len(line_owners)
        if n == 0 or set(line_owners.tolist()) != set(range(count)):
            raise ValueError("every function must contain a source line")
        visible = line_ids >= 0
        if torch.any(visible & ~mask.bool()) or torch.any(line_ids[visible] >= n):
            raise ValueError("invalid visible source-line indices")
        def encode_pool(ids, attention, pool_weights, local_ids, number_lines):
            with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.is_cuda):
                hidden = self.encoder(input_ids=ids, attention_mask=attention,
                                      use_cache=False).last_hidden_state.float()
            valid = local_ids >= 0
            return hidden.new_zeros((number_lines, hidden.shape[-1])).index_add(
                0, local_ids[valid], (hidden * pool_weights.unsqueeze(-1))[valid])
        sums = self.task_modules["classifier"].weight.new_zeros((n, self.encoder.config.hidden_size))
        for at in range(0, len(input_ids), window_batch_size):
            end = at + window_batch_size
            global_ids = line_ids[at:end]
            unique = global_ids[global_ids >= 0].unique(sorted=True)
            local_ids = torch.where(global_ids >= 0, torch.searchsorted(unique, global_ids), -1)
            args = (input_ids[at:end], mask[at:end], weights[at:end], local_ids, len(unique))
            values = (checkpoint_forward(encode_pool, *args, use_reentrant=False)
                      if self.training and torch.is_grad_enabled() else encode_pool(*args))
            sums = sums.index_add(0, unique, values)
        denominator = weights.new_zeros(n).index_add(0, line_ids[visible], weights[visible])
        if torch.any(denominator <= 0):
            raise ValueError("source lines must have visible tokens")
        # Algebraically: average each source token over its occurrences FIRST,
        # then mean its line's unique tokens. Prefix/special/pad IDs are -1.
        lines = sums / denominator[:, None]
        risks = self.task_modules["classifier"](lines).squeeze(-1)
        scores = self.task_modules["attention_score"](
            torch.tanh(self.task_modules["attention_v"](lines)) *
            torch.sigmoid(self.task_modules["attention_u"](lines))).squeeze(-1)
        logits, attention, pooled = [], torch.zeros_like(scores), []
        for owner in range(count):
            selected = line_owners == owner
            a = scores[selected].softmax(0)
            attention = attention.masked_scatter(selected, a)
            logits.append((a * risks[selected]).sum())
            pooled.append((a[:, None] * lines[selected]).sum(0))
        logits = torch.stack(logits)
        if return_line_scores:
            return logits, risks, attention
        return (logits, torch.stack(pooled)) if return_representation else logits

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
        window_batch_size=None,
        line_ids=None,
        line_owners=None,
        return_line_scores=False,
    ):
        if self.readout == "line_mil":
            if embedding_delta is not None or force_fp32 or "relations" in self.task_modules:
                raise ValueError("line MIL uses the source encoder without extra perturbation/modules")
            if source_positions is None or line_ids is None or torch.any((line_ids >= 0) & (source_positions < 0)):
                raise ValueError("prefix/boundary/padding slots cannot belong to a source line")
            return self._forward_lines(input_ids, attention_mask, window_owners, pooling_weights,
                                       function_count, line_ids, line_owners,
                                       window_batch_size=window_batch_size or 8,
                                       return_line_scores=return_line_scores,
                                       return_representation=return_representation)
        if "relations" in self.task_modules:
            if source_positions is None or relation_edges is None:
                raise ValueError("static relation adapters require local window edges")
            self._active_relation_edges = relation_edges
        streamed = bool(window_batch_size) and embedding_delta is None
        with torch.autocast(
            input_ids.device.type,
            dtype=torch.bfloat16,
            enabled=input_ids.is_cuda and not force_fp32,
        ):
            if streamed:
                if "relations" in self.task_modules:
                    raise ValueError("window chunks are only supported for the source encoder")
                # Checkpoint the encoding AND pooling, so even very long functions
                # retain only window inputs and pooled outputs between chunks.
                def encode_pool(ids, mask, weights):
                    h = self.encoder(input_ids=ids, attention_mask=mask,
                                     use_cache=False).last_hidden_state
                    dtype = torch.float64 if force_fp32 else torch.float32
                    h = h.to(dtype)
                    return (h * weights.unsqueeze(-1)).sum(1)

                global_sums = []
                for i in range(0, len(input_ids), window_batch_size):
                    stop = i + window_batch_size
                    weights = pooling_weights[i:stop]
                    arguments = (input_ids[i:stop], attention_mask[i:stop], weights)
                    sums = (checkpoint_forward(encode_pool, *arguments, use_reentrant=False)
                            if self.training and torch.is_grad_enabled() else encode_pool(*arguments))
                    global_sums.append(sums)

                def merge(sums, weights):
                    values = torch.cat(sums)
                    denominator = weights.sum(1).new_zeros(function_count).index_add(
                        0, window_owners, weights.sum(1))
                    if torch.any(denominator <= 0):
                        raise ValueError("every function/region must have pooling tokens")
                    return values.new_zeros((function_count, values.shape[-1])).index_add(
                        0, window_owners, values) / denominator[:, None]

                pooled = merge(global_sums, pooling_weights)
            elif embedding_delta is None:
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
        if not streamed:
            pooled = _window_mean(
                hidden, attention_mask, window_owners, pooling_weights, function_count,
                dtype=torch.float64 if force_fp32 else torch.float32,
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
        rows = records[start : start + batch_size]
        mil = model.readout == "line_mil"
        batch = builder.codebert_batch(rows, device=device, return_source_positions=mil)
        line_info = builder.line_mapping(rows, batch) if mil else {}
        logits = model(*batch, window_batch_size=8 if builder.full_source else None, **line_info)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite prediction logits")
        values.append(logits.float().cpu())
    return torch.cat(values) if values else torch.empty(0)


def load_model(checkpoint, device, *, model_path=None):
    model = CodeBERT(model_path or checkpoint["model_path"], device,
                     relation_mode=(checkpoint.get("static_relations") or {}).get("mode", "none"),
                     readout=checkpoint.get("readout", "mean"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def predict_checkpoint(checkpoint_path, records, *, device="auto", batch_size=4, model_path=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = resolve_device(device)
    path = model_path or checkpoint["model_path"]
    builder = InputBuilder(
        AutoTokenizer.from_pretrained(path, trim_offsets=not checkpoint.get("input_protocol", {}).get("full_source", False)),
        source_max_length=checkpoint["source_max_length"],
        relation_mode=(checkpoint.get("static_relations") or {}).get("mode", "none"),
        full_source=checkpoint.get("input_protocol", {}).get("full_source", False),
    )
    model = load_model(checkpoint, device, model_path=path)
    return predict_logits(model, records, builder, device=device, batch_size=batch_size)
