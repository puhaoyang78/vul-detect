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
    token_pairs: torch.Tensor | None = None  # [pairs, 3]: node, batch row, source token
    ddg_edges: torch.Tensor | None = None  # [2, edges], reaching definition -> use

    def to(self, device) -> "GraphBatch":
        pairs = None if self.token_pairs is None else self.token_pairs.to(device)
        ddg = None if self.ddg_edges is None else self.ddg_edges.to(device)
        return GraphBatch(self.attributes.to(device), self.edges.to(device), self.ptr, pairs, ddg)


def collate_graphs(views: list[dict], encoded: list[list[list[int]]], device="cpu",
                   alignments: list[list[tuple[int, int]]] | None = None,
                   ddg_edges: list[list[tuple[int, int]]] | None = None) -> GraphBatch:
    if (not views or len(views) != len(encoded) or
            (alignments is not None and len(alignments) != len(views)) or
            (ddg_edges is not None and len(ddg_edges) != len(views))):
        raise ValueError("nonempty matching graph/attribute lists required")
    attributes, edges, ptr, token_pairs, ddg = [], [], [0], [], []
    for row, (view, values) in enumerate(zip(views, encoded)):
        n = len(values)
        if not n or n != len(view["node_ids"]) or any(len(v) != 4 for v in values):
            raise ValueError("invalid node attributes")
        offset = ptr[-1]
        for s, t in view["edges"]:
            if not 0 <= s < n or not 0 <= t < n:
                raise ValueError("out-of-range CFG endpoint")
            edges.append((s + offset, t + offset))
        if ddg_edges is not None:
            for s, t in ddg_edges[row]:
                if not 0 <= s < n or not 0 <= t < n:
                    raise ValueError("out-of-range DDG endpoint")
                ddg.append((s + offset, t + offset))
        if alignments is not None:
            for node, token in alignments[row]:
                if not 0 <= node < n or token < 0:
                    raise ValueError("out-of-range source alignment")
                token_pairs.append((offset + node, row, token))
        attributes.extend(values)
        ptr.append(offset + n)
    edge_tensor = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
    pairs = None if alignments is None else torch.tensor(token_pairs, dtype=torch.long).reshape(-1, 3)
    ddg_tensor = None if ddg_edges is None else torch.tensor(ddg, dtype=torch.long).reshape(-1, 2).T.contiguous()
    return GraphBatch(torch.tensor(attributes, dtype=torch.long), edge_tensor, tuple(ptr), pairs,
                      ddg_tensor).to(device)


class AttributeCFGEncoder(nn.Module):
    """B/C, D/E, and F/G share the existing attribute and CFG encoder.

    B and D use node-local self messages only; C and E also sum directed
    predecessor messages. Each pair shares parameters and update counts.
    D/E add the same projected, position-aligned source-token mean to the
    attribute embedding before the existing graph encoder. F/G additionally sum
    directed DDG messages through their own projection before the same GRU update.
    The raw attribute embedding is concatenated with the final node state before
    attention pooling, as in the DeepDFA encoder design. These are learned
    summaries, not a sound static-analysis result.
    """
    def __init__(self, vocabulary_sizes: list[int], *, hidden_size: int = 128,
                 steps: int = 5, mode: str = "cfg", source_hidden_size: int | None = None):
        super().__init__()
        if (len(vocabulary_sizes) != 4 or min(vocabulary_sizes) < 2 or
                hidden_size < 4 or hidden_size % 4 or steps <= 0 or
                mode not in {"attributes", "cfg", "aligned_attributes", "aligned_cfg",
                             "cfg_ddg", "cfg_ddg_shuffled"} or
                (mode.startswith("aligned_") and (source_hidden_size is None or source_hidden_size <= 0))):
            raise ValueError("invalid graph encoder configuration")
        self.mode, self.steps = mode, steps
        self.embedding = nn.ModuleList([nn.Embedding(size, hidden_size // 4) for size in vocabulary_sizes])
        if mode.startswith("aligned_"):
            self.source_projection = nn.Linear(source_hidden_size, hidden_size, bias=False)
        self.message = nn.Linear(hidden_size, hidden_size)
        self.update = nn.GRUCell(hidden_size, hidden_size)
        self.pool_gate = nn.Linear(2 * hidden_size, 1)
        # Allocate after every shared module so C/F/G share seeded initialization.
        if mode in {"cfg_ddg", "cfg_ddg_shuffled"}:
            self.ddg_message = nn.Linear(hidden_size, hidden_size)
        self.output_dim = 2 * hidden_size

    def forward(self, batch: GraphBatch, token_hidden: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([layer(batch.attributes[:, i]) for i, layer in enumerate(self.embedding)], dim=-1)
        if self.mode.startswith("aligned_"):
            if token_hidden is None or batch.token_pairs is None:
                raise ValueError("aligned graph encoder requires source-token positions and hidden states")
            pairs = batch.token_pairs
            node_source = token_hidden.new_zeros((x.shape[0], token_hidden.shape[-1]), dtype=torch.float32)
            if pairs.numel():
                selected = token_hidden[pairs[:, 1], pairs[:, 2]].float()
                node_source.index_add_(0, pairs[:, 0], selected)
                counts = torch.bincount(pairs[:, 0], minlength=x.shape[0]).clamp_min(1).unsqueeze(-1)
                node_source = node_source / counts
            x = x + self.source_projection(node_source)
        state = x
        src, dst = batch.edges
        # Remove duplicate CFG self edges: every mode already receives one self message.
        mask = src != dst
        src, dst = src[mask], dst[mask]
        use_ddg = self.mode in {"cfg_ddg", "cfg_ddg_shuffled"}
        if use_ddg:
            if batch.ddg_edges is None:
                raise ValueError("DDG graph encoder requires directed DDG edges")
            ddg_src, ddg_dst = batch.ddg_edges
        for _ in range(self.steps):
            transformed = self.message(state)
            incoming = transformed.clone()
            if self.mode in {"cfg", "aligned_cfg", "cfg_ddg", "cfg_ddg_shuffled"} and src.numel():
                incoming.index_add_(0, dst, transformed[src])
            if use_ddg and ddg_src.numel():
                ddg_transformed = self.ddg_message(state)
                incoming.index_add_(0, ddg_dst, ddg_transformed[ddg_src])
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
            graph = AttributeCFGEncoder(vocabulary_sizes, hidden_size=hidden_size, steps=steps, mode=mode,
                                        source_hidden_size=(self.task_modules["classifier"].in_features
                                                            if mode.startswith("aligned_") else None))
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
        graph_vector = self.task_modules["cfg_encoder"](graph_batch, hidden)
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
    mode = "cfg" if config["variant"] in {"cfg_double_ce", "cfg_rdrop"} else config["variant"]
    return SourceGraphClassifier(source, vocabulary_sizes,
                                 hidden_size=config["graph_hidden_size"], steps=config["graph_steps"],
                                 mode=mode, device=device)
