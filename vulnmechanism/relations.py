"""One low-rank relation readout and capacity/rewiring controls."""

from bisect import bisect_right
from collections import defaultdict
import hashlib
import random

import torch
from torch import nn

from .static_analysis import RELATIONS

MODES = ("none", "sequence", "shuffled", "real")
RANK = 32
PER_TYPE = 32


def relation_tokens(record, tokenizer, budget, mode):
    """Return (source token, target token, type), without labels/model outputs.

    Endpoints use the first overlapping subtoken of a located CPG node. The
    shuffled control preserves source endpoints, destination multiset, typed
    edge counts and candidate regions, while destroying endpoint pairing.
    """
    if mode not in MODES or mode == "none":
        raise ValueError("relation_tokens requires an active relation mode")
    encoded = tokenizer(record["raw_source"], add_special_tokens=False, truncation=True,
                        max_length=budget, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    n = len(offsets)
    if not n:
        return [], dict(visible_nodes=0, selected_edges=0)
    rng = random.Random(int.from_bytes(hashlib.sha256(record["raw_source"].encode()).digest()[:8], "big") ^ 42)
    if mode == "sequence":
        # A content-independent sequence neighbourhood, no CPG/count/coverage
        # information. All parameters are used, with the identical type budget.
        edges = []
        for kind in range(len(RELATIONS)):
            positions = [min(n - 1, i * n // min(PER_TYPE, n)) for i in range(min(PER_TYPE, n))]
            edges.extend((p, min(n - 1, p + kind + 1), kind) for p in positions)
        return edges, dict(visible_nodes=n, selected_edges=len(edges), changed_endpoints=0)
    graph = record["program_relations"]
    if graph["status"] != "ok":
        return [], dict(visible_nodes=0, selected_edges=0, graph_failure=graph["status"])
    ends = [b for _, b in offsets]
    positions = {}
    for i, node in enumerate(graph["nodes"]):
        start, end = node["span"]
        at = bisect_right(ends, start)
        if at < n and offsets[at][0] < end and ends[at] > start:
            positions[i] = at
    grouped = defaultdict(set)
    for a, b, kind in graph["edges"]:
        if a in positions and b in positions:
            grouped[kind].add((positions[a], positions[b], kind))
    edges = []
    for kind in range(len(RELATIONS)):
        candidates = sorted(grouped[kind])
        if len(candidates) > PER_TYPE:
            candidates = sorted(rng.sample(candidates, PER_TYPE))
        edges.extend(candidates)
    info = dict(visible_nodes=len(positions), located_nodes=len(graph["nodes"]), selected_edges=len(edges),
                type_edges={RELATIONS[k]: sum(e[2] == k for e in edges) for k in range(len(RELATIONS))})
    if mode == "shuffled":
        destinations = [e[1] for e in edges]
        rng.shuffle(destinations)
        info["changed_endpoints"] = sum(b != dest for (_, b, _), dest in zip(edges, destinations))
        edges = [(a, dest, k) for (a, _, k), dest in zip(edges, destinations)]
    return edges, info


class RelationPool(nn.Module):
    """Learn pair messages in rank 32, mean by type, project to source space.

    No feature concatenation, attention supervision or auxiliary labels. Zero
    output initialization preserves exactly the original classifier initially.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.source = nn.Linear(hidden_size, RANK, bias=False)
        self.target = nn.Linear(hidden_size, RANK, bias=False)
        self.kind = nn.Embedding(len(RELATIONS), RANK)
        self.output = nn.Linear(RANK, hidden_size, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, hidden, positions, owners, pooling, edges, function_count):
        if edges.numel() == 0:
            return hidden.new_zeros((function_count, hidden.shape[-1]), dtype=torch.float32)
        valid = positions >= 0
        length = int(positions.max()) + 1
        owner_slots = owners[:, None].expand_as(positions)
        indices = owner_slots[valid] * length + positions[valid]
        values = hidden[valid].float() * pooling[valid].unsqueeze(-1)
        # Overlapping-window appearances sum to one per source position.
        tokens = values.new_zeros((function_count * length, hidden.shape[-1])).index_add(0, indices, values)
        owner, a, b, kind = edges.unbind(1)
        messages = torch.tanh(self.source(tokens[owner * length + a])
                              + self.target(tokens[owner * length + b]) + self.kind(kind))
        groups = owner * len(RELATIONS) + kind
        counts = messages.new_zeros(function_count * len(RELATIONS)).index_add(0, groups, messages.new_ones(len(edges)))
        summed = messages.new_zeros((function_count * len(RELATIONS), RANK)).index_add(0, groups, messages)
        typed = summed / counts.clamp_min(1)[:, None]
        typed = typed.reshape(function_count, len(RELATIONS), RANK)
        active = (counts > 0).reshape(function_count, len(RELATIONS)).sum(1).clamp_min(1)
        return self.output(typed.sum(1) / active[:, None])
