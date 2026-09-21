"""Matched local-node/CFG encoders; no DGL or torch-geometric dependency."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class GraphBatch:
    attributes: torch.Tensor  # [nodes, 4]
    edges: torch.Tensor       # [2, edges], source -> target
    ptr: tuple[int, ...]      # graph boundaries in the node array

    def to(self, device) -> "GraphBatch":
        return GraphBatch(self.attributes.to(device), self.edges.to(device), self.ptr)


def collate_graphs(views: list[dict], encoded: list[list[list[int]]], device="cpu") -> GraphBatch:
    if not views or len(views) != len(encoded):
        raise ValueError("nonempty matching graph/attribute lists required")
    attributes, edges, ptr = [], [], [0]
    for view, values in zip(views, encoded):
        n = len(values)
        if not n or n != len(view["node_ids"]) or any(len(v) != 4 for v in values):
            raise ValueError("invalid node attributes")
        offset = ptr[-1]
        for s, t in view["edges"]:
            if not 0 <= s < n or not 0 <= t < n:
                raise ValueError("out-of-range CFG endpoint")
            edges.append((s + offset, t + offset))
        attributes.extend(values)
        ptr.append(offset + n)
    edge_tensor = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
    return GraphBatch(torch.tensor(attributes, dtype=torch.long), edge_tensor, tuple(ptr)).to(device)


class AttributeCFGEncoder(nn.Module):
    """B and C have identical parameters, initialization and update counts.

    B (attributes): node-local self messages only, without inter-node exchange.
    C (cfg): the same self messages plus directed CFG predecessor messages.
    The raw attribute embedding is concatenated with the final node state before
    attention pooling, as in the DeepDFA encoder design. These are learned
    summaries, not a sound static-analysis result.
    """
    def __init__(self, vocabulary_sizes: list[int], *, hidden_size: int = 128,
                 steps: int = 5, mode: str = "cfg"):
        super().__init__()
        if (len(vocabulary_sizes) != 4 or min(vocabulary_sizes) < 2 or
                hidden_size < 4 or hidden_size % 4 or steps <= 0 or mode not in {"attributes", "cfg"}):
            raise ValueError("invalid graph encoder configuration")
        self.mode, self.steps = mode, steps
        self.embedding = nn.ModuleList([nn.Embedding(size, hidden_size // 4) for size in vocabulary_sizes])
        self.message = nn.Linear(hidden_size, hidden_size)
        self.update = nn.GRUCell(hidden_size, hidden_size)
        self.pool_gate = nn.Linear(2 * hidden_size, 1)
        self.output_dim = 2 * hidden_size

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = torch.cat([layer(batch.attributes[:, i]) for i, layer in enumerate(self.embedding)], dim=-1)
        state = x
        src, dst = batch.edges
        # Remove duplicate self edges: both controls receive exactly one self message.
        mask = src != dst
        src, dst = src[mask], dst[mask]
        for _ in range(self.steps):
            transformed = self.message(state)
            incoming = transformed.clone()
            if self.mode == "cfg" and src.numel():
                incoming.index_add_(0, dst, transformed[src])
            state = self.update(incoming, state)
        combined = torch.cat((state, x), dim=-1)
        scores = self.pool_gate(combined).squeeze(-1)
        return torch.stack([(scores[a:b].softmax(dim=0).unsqueeze(-1) * combined[a:b]).sum(dim=0)
                            for a, b in zip(batch.ptr[:-1], batch.ptr[1:])])


class SourceGraphClassifier(nn.Module):
    """A full-width source head plus a graph head = linear classification of concat.

    The graph-head weights start at zero; initially the logits equal the source
    baseline exactly. Both source LoRA and graph modules are trained. This is NOT
    a guarantee of unchanged predictions after optimization.
    """
    def __init__(self, source_model, vocabulary_sizes: list[int], *, hidden_size: int,
                 steps: int, mode: str, device):
        super().__init__()
        self.encoder = source_model.encoder
        self.task_modules = source_model.task_modules
        with torch.random.fork_rng(devices=[]):
            graph = AttributeCFGEncoder(vocabulary_sizes, hidden_size=hidden_size, steps=steps, mode=mode)
            head = nn.Linear(graph.output_dim, 1, bias=False)
            nn.init.zeros_(head.weight)
            self.task_modules["cfg_encoder"] = graph
            self.task_modules["cfg_classifier"] = head
        self.to(device)

    def forward(self, input_ids, attention_mask, graph_batch: GraphBatch):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              use_cache=False).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float()
        source_logit = self.task_modules["classifier"](pooled).squeeze(-1)
        graph_vector = self.task_modules["cfg_encoder"](graph_batch)
        return source_logit + self.task_modules["cfg_classifier"](graph_vector).squeeze(-1)


def build_model(base, config: dict, vocabulary_sizes: list[int] | None, device, *, training: bool):
    source = base.SequenceVulnerabilityClassifier(
        config["model_path"], variant="baseline", device=device,
        lora_r=config["lora_r"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        gradient_checkpointing=training and device.type == "cuda")
    if config["variant"] == "baseline":
        return source
    if vocabulary_sizes is None:
        raise ValueError("graph vocabulary required")
    return SourceGraphClassifier(source, vocabulary_sizes,
                                 hidden_size=config["graph_hidden_size"], steps=config["graph_steps"],
                                 mode=config["variant"], device=device)
