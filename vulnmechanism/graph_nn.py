"""PyTorch-only DeepDFA-style attribute embeddings, GGNN and attentive readout.

B and C have identical parameter counts and update depth. B supplies each node
only its own transformed state; C sums transformed states from CFG predecessors.
Neither variant adds graph-text tokens or consumes mechanism/pair/CVE labels.
"""
from __future__ import annotations

import torch
from torch import nn

from .graph_features import FEATURE_NAMES, GRAPH_SCHEMA_VERSION, feature_signature


class StaticGraphEncoder(nn.Module):
    def __init__(self, config: dict, *, propagate: bool) -> None:
        super().__init__()
        if config.get("schema_version") != GRAPH_SCHEMA_VERSION:
            raise ValueError("incompatible graph encoder schema")
        self.vocabulary = config["vocabulary"]
        self.steps = int(config["steps"])
        self.propagate = propagate
        width = int(config["embedding_dim"])
        if width <= 0 or self.steps <= 0:
            raise ValueError("invalid graph encoder dimensions")
        for name in FEATURE_NAMES:
            ids = list(self.vocabulary[name].values())
            if sorted(ids) != list(range(len(ids))) or self.vocabulary[name].get("[]") != 0 or self.vocabulary[name].get("<UNK>") != 1:
                raise ValueError(f"invalid {name} feature vocabulary")
        self.embeddings = nn.ModuleDict({name: nn.Embedding(len(self.vocabulary[name]), width)
                                        for name in FEATURE_NAMES})
        self.hidden_dim = width * len(FEATURE_NAMES)
        self.out_dim = 2 * self.hidden_dim
        self.message = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.update = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.pool_gate = nn.Linear(self.out_dim, 1)

    def forward(self, graphs: list[dict | None]) -> torch.Tensor:
        if not graphs:
            raise ValueError("cannot encode an empty graph batch")
        device = self.message.weight.device
        fields, senders, receivers, sizes = [], [], [], []
        for graph in graphs:
            if graph is None:
                sizes.append(0)
                continue
            start = len(fields)
            nodes = graph["cfg_nodes"]
            sizes.append(len(nodes))
            for node in nodes:
                fields.append([self.vocabulary[name].get(
                    feature_signature(node["features"][name]), 1
                ) for name in FEATURE_NAMES])
            if self.propagate:
                for u, v in graph["cfg_edges"]:
                    senders.append(start + u)
                    receivers.append(start + v)
        if not fields:
            return self.message.weight.new_zeros((len(graphs), self.out_dim))
        indices = torch.tensor(fields, dtype=torch.long, device=device)
        initial = torch.cat([self.embeddings[name](indices[:, i])
                             for i, name in enumerate(FEATURE_NAMES)], dim=-1)
        hidden = initial
        source = torch.tensor(senders, dtype=torch.long, device=device)
        target = torch.tensor(receivers, dtype=torch.long, device=device)
        for _ in range(self.steps):
            messages = self.message(hidden)
            if self.propagate:
                incoming = torch.zeros_like(hidden)
                incoming.index_add_(0, target, messages[source])
            else:
                incoming = messages  # local recurrent control: no cross-node information
            hidden = self.update(incoming, hidden)
        states = torch.cat((hidden, initial), dim=-1)
        gates = self.pool_gate(states).squeeze(-1)
        pooled, start = [], 0
        for size in sizes:
            if not size:
                pooled.append(states.new_zeros(self.out_dim))
                continue
            weights = gates[start:start + size].softmax(dim=0)
            pooled.append((weights.unsqueeze(-1) * states[start:start + size]).sum(dim=0))
            start += size
        return torch.stack(pooled)
