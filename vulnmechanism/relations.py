"""Directed, type-conditioned attention over program dependencies before readout."""

from bisect import bisect_right
from collections import defaultdict
import hashlib
import math
import random

import torch
from torch import nn

from .static_analysis import RELATIONS

MODES = ("none", "sequence", "shuffled", "real")
RANK = 64
PER_TYPE = 32


def relation_tokens(record, tokenizer, budget, mode):
    """Map CPG edges and vulnerability-pattern relationships to source subwords.

    The randomized control preserves endpoints separately inside each type,
    rather than destroying type-specific endpoint frequencies. Missing CPGs do
    not alter the source-only readout. All selections are label-independent.
    """
    if mode not in MODES or mode == "none":
        raise ValueError("relation_tokens needs a relation mode")
    encoded = tokenizer(record["raw_source"], add_special_tokens=False,
                        truncation=True, max_length=budget, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    n = len(offsets)
    if n == 0:
        return [], dict(visible_nodes=0, selected_edges=0)
    digest = hashlib.sha256(record["raw_source"].encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big") ^ 42)
    if mode == "sequence":
        edges = []
        for kind in range(len(RELATIONS)):
            length = min(PER_TYPE, n)
            positions = [min(n - 1, i * n // length) for i in range(length)]
            edges.extend((p, min(n - 1, p + kind + 1), kind) for p in positions)
        return edges, dict(visible_nodes=n, selected_edges=len(edges), changed_endpoints=0)

    graph = record.get("program_relations", {})
    if graph.get("status") != "ok":
        return [], dict(visible_nodes=0, selected_edges=0,
                        graph_failure=graph.get("status", "missing"))
    ends = [end for _, end in offsets]
    positions = {}
    for i, node in enumerate(graph["nodes"]):
        start, end = node["span"]
        at = bisect_right(ends, start)
        if at < n and offsets[at][0] < end and ends[at] > start:
            positions[i] = at
    grouped = defaultdict(set)
    for a, b, kind in graph["edges"]:
        if a in positions and b in positions and 0 <= kind < len(RELATIONS):
            grouped[kind].add((positions[a], positions[b], kind))
    edges = []
    for kind in range(len(RELATIONS)):
        candidates = sorted(grouped[kind])
        if len(candidates) > PER_TYPE:
            candidates = sorted(rng.sample(candidates, PER_TYPE))
        edges.extend(candidates)
    info = dict(visible_nodes=len(positions), located_nodes=len(graph["nodes"]),
                selected_edges=len(edges),
                type_edges={RELATIONS[k]: sum(e[2] == k for e in edges)
                            for k in range(len(RELATIONS))})
    if mode == "shuffled":
        output = []
        changed = 0
        for kind in range(len(RELATIONS)):
            chunk = [(a, b, k) for a, b, k in edges if k == kind]
            dest = [b for _, b, _ in chunk]
            if len(dest) > 1:
                # Derangement whenever possible, while preserving destination
                # multiplicities and all sources inside this relation type.
                best = dest[:]
                for _ in range(8):
                    candidate = dest[:]
                    rng.shuffle(candidate)
                    if sum(a != b for a, b in zip(candidate, dest)) > sum(
                            a != b for a, b in zip(best, dest)):
                        best = candidate
                dest = best
            output.extend((a, b_new, k) for (a, b_old, k), b_new in zip(chunk, dest))
            changed += sum(b_old != b_new for (_, b_old, _), b_new in zip(chunk, dest))
        info["changed_endpoints"] = changed
        edges = output
    return edges, info


class StaticRelationAttention(nn.Module):
    """Use code-graph relationships to pool token states, not a weak readout residual.

    An ordered (source, target, edge type) creates each attention key; pairwise
    interactions and their direction make the output sensitive to rewiring.
    A learned gate combines the relation-focused and standard function pooling.
    The gate is not zero-initialized and updates both encoder and attention
    parameters from the actual binary vulnerability-classification loss.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.normalize = nn.LayerNorm(hidden_size)
        self.kind = nn.Embedding(len(RELATIONS), RANK)
        self.pair = nn.Linear(4 * hidden_size, RANK, bias=True)
        self.query = nn.Linear(hidden_size, RANK, bias=True)
        self.key = nn.Linear(RANK, RANK, bias=False)
        self.value = nn.Linear(RANK, hidden_size, bias=False)
        self.type_bias = nn.Embedding(len(RELATIONS), 1)
        self.gate = nn.Linear(2 * hidden_size, 1, bias=True)
        nn.init.zeros_(self.type_bias.weight)
        # An operative branch at initialization; not an inactive zero head.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -0.5)

    def forward(self, hidden, positions, owners, pooling, edges, function_count, base):
        if edges.numel() == 0:
            return base
        if positions.shape != hidden.shape[:2] or pooling.shape != positions.shape:
            raise ValueError("inconsistent token positions or window weights")
        if base.shape != (function_count, self.hidden_size):
            raise ValueError("invalid source representation")
        valid = positions >= 0
        if not valid.any():
            return base
        length = int(positions.max().item()) + 1
        owner_slots = owners[:, None].expand_as(positions)
        indices = owner_slots[valid] * length + positions[valid]
        values = hidden[valid].float() * pooling[valid].unsqueeze(-1)
        tokens = values.new_zeros((function_count * length, self.hidden_size)).index_add(0, indices, values)
        owner, src, dst, kind = edges.unbind(1)
        if (owner.min() < 0 or owner.max() >= function_count or src.min() < 0
                or src.max() >= length or dst.min() < 0 or dst.max() >= length
                or kind.min() < 0 or kind.max() >= len(RELATIONS)):
            raise ValueError("relation token indices outside available representations")
        head = self.normalize(tokens[owner * length + src])
        tail = self.normalize(tokens[owner * length + dst])
        pair = torch.tanh(self.pair(torch.cat((head, tail, head * tail, tail - head), -1))
                          + self.kind(kind))
        query = self.query(self.normalize(base))[owner]
        score = (query * self.key(pair)).sum(dim=-1) / math.sqrt(RANK)
        score = score + self.type_bias(kind).squeeze(-1)
        # Retain source/target features and their pair-conditioned interaction.
        edge_values = 0.5 * (tokens[owner * length + src] +
                             tokens[owner * length + dst]) + self.value(pair)
        result = []
        for i in range(function_count):
            member = owner == i
            if not member.any():
                result.append(base[i])
                continue
            weights = torch.softmax(score[member], dim=0)
            focused = (edge_values[member] * weights[:, None]).sum(dim=0)
            strength = torch.sigmoid(self.gate(torch.cat((base[i], focused), dim=0))).squeeze()
            result.append((1 - strength) * base[i] + strength * focused)
        return torch.stack(result)
