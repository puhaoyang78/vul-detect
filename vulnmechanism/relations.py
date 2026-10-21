"""Typed static-relation attention for function-level vulnerability classification.

The CPG and syntax-based vulnerability patterns restrict the candidate pairs;
their learned attention scores are conditioned on the *paired* program tokens.
The resulting representation is formed BEFORE the classifier, not appended
as a near-zero residual to an already-finalized logit.
"""

from bisect import bisect_right
from collections import defaultdict
import hashlib
import math
import random

import torch
from torch import nn

from .static_analysis import EDGE_TYPES, RELATIONS

MODES = ("none", "sequence", "shuffled", "real")
RANK = 64
PER_TYPE = 24


def relation_tokens(record, tokenizer, budget, mode):
    """Return source/destination subtoken positions and static relation IDs."""
    if mode not in MODES or mode == "none":
        raise ValueError("an active relation mode is required")
    encoded = tokenizer(record["raw_source"], add_special_tokens=False, truncation=True,
                        max_length=budget, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    n = len(offsets)
    if not n:
        return [], dict(visible_nodes=0, selected_edges=0)
    rng = random.Random(int.from_bytes(
        hashlib.sha256(record["raw_source"].encode()).digest()[:8], "big") ^ 42)
    if mode == "sequence":
        # Same parameterized attention, without program-analysis relations.
        edges = []
        positions = [min(n - 1, i * n // min(PER_TYPE, n)) for i in range(min(PER_TYPE, n))]
        for kind in range(len(RELATIONS)):
            edges.extend((p, min(n - 1, p + kind + 1), kind) for p in positions)
        return edges, dict(visible_nodes=n, selected_edges=len(edges),
                           type_edges={name: len(positions) for name in RELATIONS})
    graph = record["program_relations"]
    if graph["status"] != "ok":
        return [], dict(visible_nodes=0, selected_edges=0, graph_failure=graph["status"])
    ends = [b for _, b in offsets]
    node_positions = {}
    for i, node in enumerate(graph["nodes"]):
        start, end = node["span"]
        at = bisect_right(ends, start)
        if at < n and offsets[at][0] < end and ends[at] > start:
            node_positions[i] = at

    grouped = defaultdict(set)
    for a, b, kind in graph["edges"]:
        if a in node_positions and b in node_positions and 0 <= kind < len(RELATIONS):
            grouped[kind].add((node_positions[a], node_positions[b], kind))
    edges = []
    for kind in range(len(RELATIONS)):
        choices = sorted(grouped[kind])
        if len(choices) > PER_TYPE:
            # Prefer all supported pattern pairings; sample high-volume CPG edges.
            if kind >= len(EDGE_TYPES):
                choices = choices[:PER_TYPE]
            else:
                choices = sorted(rng.sample(choices, PER_TYPE))
        edges.extend(choices)
    info = dict(visible_nodes=len(node_positions), located_nodes=len(graph["nodes"]),
                selected_edges=len(edges), type_edges={RELATIONS[k]: sum(e[2] == k for e in edges)
                                                        for k in range(len(RELATIONS))})
    if mode == "shuffled":
        rewired = list(edges)
        changed = 0
        by_kind = defaultdict(list)
        for i, (_, _, kind) in enumerate(edges):
            by_kind[kind].append(i)
        for positions in by_kind.values():
            if len(positions) < 2:
                continue
            destinations = [edges[i][1] for i in positions]
            shuffled = list(destinations)
            rng.shuffle(shuffled)
            if shuffled == destinations and len(set(shuffled)) > 1:
                shuffled = shuffled[1:] + shuffled[:1]
            for index, dst in zip(positions, shuffled):
                src, old_dst, kind = edges[index]
                changed += dst != old_dst
                rewired[index] = (src, dst, kind)
        edges = rewired
        info["changed_endpoints"] = changed
    return edges, info


class StaticRelationAttention(nn.Module):
    """Pair-sensitive relation attention followed by gated function-level readout.

    The source/destination interaction is nonlinear, so rewiring a typed graph
    changes learned relations even when node marginals and edge counts are fixed.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, RANK, bias=False)
        self.key = nn.Linear(hidden_size, RANK, bias=False)
        self.value = nn.Linear(hidden_size, RANK, bias=False)
        self.pair = nn.Linear(hidden_size, RANK, bias=False)
        self.kind = nn.Embedding(len(RELATIONS), RANK)
        self.kind_bias = nn.Embedding(len(RELATIONS), 1)
        self.output = nn.Linear(RANK, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Linear(hidden_size * 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        nn.init.zeros_(self.kind_bias.weight)
        with torch.no_grad():
            # Mild, trainable priority for vulnerability-pattern relations.
            self.kind_bias.weight[len(EDGE_TYPES):, 0] = 0.25

    def forward(self, hidden, positions, owners, pooling, edges, base):
        count, hidden_size = base.shape
        if edges.numel() == 0:
            return base
        valid = positions >= 0
        length = int(positions.max().item()) + 1
        owner_slots = owners[:, None].expand_as(positions)
        indices = owner_slots[valid] * length + positions[valid]
        # Overlapping windows give each source position total weight one.
        values = hidden[valid].float() * pooling[valid].unsqueeze(-1)
        tokens = values.new_zeros((count * length, hidden_size)).index_add(0, indices, values)
        owner, a, b, kind = edges.unbind(1)
        left = tokens[owner * length + a]
        right = tokens[owner * length + b]
        q, k = self.query(left), self.key(right)
        scores = ((q * k).sum(-1) / math.sqrt(RANK) +
                  self.kind_bias(kind).squeeze(-1))
        # Normalize across *candidate pairings*, separately for every function.
        maxima = scores.new_full((count,), float("-inf"))
        maxima = maxima.scatter_reduce(0, owner, scores, reduce="amax", include_self=True)
        weights = torch.exp(scores - maxima[owner])
        denominator = scores.new_zeros(count).index_add(0, owner, weights)
        weights = weights / denominator[owner].clamp_min(1e-10)
        message = torch.tanh(self.value(right) + self.pair(left * right) + self.kind(kind))
        context = message.new_zeros((count, RANK)).index_add(
            0, owner, weights[:, None] * message)
        context = self.norm(self.output(context))
        alpha = torch.sigmoid(self.gate(torch.cat([base.float(), context], dim=-1)))
        present = denominator > 0
        alpha = alpha * present[:, None]
        return (1 - alpha) * base.float() + alpha * context
