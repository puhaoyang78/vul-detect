"""Sparse, operation-oriented program relations injected between CodeBERT layers.

Joern relationships determine candidate token pairs, while learned, typed
attention determines how each code position uses related definitions, conditions,
and vulnerability-pattern operands. The module does not infer bug labels from
static matches. There is only one static-relation training implementation.
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
MAX_PAIRS = 192
# All CPG edges are exported for auditing. Only these relations carry value
# definition, control dependence, or symbol-reference information into encoding.
CPG_CONTEXT = ("REACHING_DEF", "CDG", "REF")
CPG_IDS = {RELATIONS.index(name) for name in CPG_CONTEXT}
PATTERN_START = len(EDGE_TYPES)


def _balanced_sample(items, limit, rng):
    """Round-robin over distinct query sites rather than first lines of code."""
    groups = defaultdict(list)
    for q, k, typ in sorted(set(items)):
        groups[q].append((q, k, typ))
    anchors = list(groups)
    rng.shuffle(anchors)
    for values in groups.values():
        rng.shuffle(values)
    chosen = []
    while anchors and len(chosen) < limit:
        left = []
        for anchor in anchors:
            if len(chosen) >= limit:
                break
            chosen.append(groups[anchor].pop())
            if groups[anchor]:
                left.append(anchor)
        anchors = left
    return chosen


def relation_tokens(record, tokenizer, budget, mode):
    """Map candidate program relations to *directed* source-token positions.

    CPG REACHING_DEF and CDG: affected use/statement queries its definition or
    controlling condition. REF: identifier queries its resolved declaration.
    Vulnerability patterns: candidate operation queries its operands and local
    program dependencies. A false/missing pattern is never a classification label.
    """
    if mode not in MODES or mode == "none":
        raise ValueError("relation_tokens requires an active mode")
    encoded = tokenizer(record["raw_source"], add_special_tokens=False, truncation=True,
                        max_length=budget, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    n = len(offsets)
    if not n:
        return [], dict(visible_nodes=0, selected_edges=0)
    seed = int.from_bytes(
        hashlib.sha256(record["raw_source"].encode("utf-8")).digest()[:8], "big") ^ 42
    rng = random.Random(seed)
    if mode == "sequence":
        positions = [min(n - 1, i * n // 24) for i in range(min(24, n))]
        edges = []
        for kind in sorted(CPG_IDS) + list(range(PATTERN_START, len(RELATIONS))):
            edges.extend((p, min(n - 1, p + 1 + (kind % 3)), kind) for p in positions)
        cpg = _balanced_sample([e for e in edges if e[2] in CPG_IDS], 80, rng)
        pat = _balanced_sample([e for e in edges if e[2] >= PATTERN_START], 112, rng)
        return cpg + pat, dict(visible_nodes=n, selected_edges=len(cpg) + len(pat),
                               selected_cpg=len(cpg), selected_patterns=len(pat),
                               method="source-neighborhood-only")

    graph = record["program_relations"]
    if graph["status"] != "ok" and not graph.get("patterns"):
        return [], dict(visible_nodes=0, selected_edges=0, graph_failure=graph["status"])
    ends = [end for _, end in offsets]
    token_locations = {}
    for i, node in enumerate(graph["nodes"]):
        start, end = node["span"]
        pos = bisect_right(ends, start)
        if pos < n and offsets[pos][0] < end and ends[pos] > start:
            token_locations[i] = pos

    grouped = defaultdict(set)
    for a, b, kind in graph["edges"]:
        if a not in token_locations or b not in token_locations:
            continue
        if kind < PATTERN_START:
            if kind not in CPG_IDS:
                continue
            # Reaching definitions / control dependence point from the
            # provider to the use, so reverse them for query->context.
            q, k = (token_locations[a], token_locations[b]) if RELATIONS[kind] == "REF" \
                else (token_locations[b], token_locations[a])
        else:
            # Pattern spans are anchored on the memory operation / query site.
            q, k = token_locations[a], token_locations[b]
        grouped[kind].add((q, k, kind))

    per_kind = {kind: _balanced_sample(edges, PER_TYPE, rng)
                for kind, edges in sorted(grouped.items())}
    # Preserve a useful ratio between CPG context and vulnerability-pattern
    # operands. Avoid whole-function, source-line-order truncation.
    structural = _balanced_sample(
        [e for kind, rows in per_kind.items() if kind in CPG_IDS for e in rows],
        80, rng)
    patterns = _balanced_sample(
        [e for kind, rows in per_kind.items() if kind >= PATTERN_START for e in rows],
        MAX_PAIRS - 80, rng)
    edges = structural + patterns
    info = dict(
        visible_nodes=len(token_locations), located_nodes=len(graph["nodes"]),
        selected_edges=len(edges), selected_cpg=len(structural),
        selected_patterns=len(patterns), extracted_edges=len(graph["edges"]),
        type_edges={RELATIONS[k]: sum(e[2] == k for e in edges) for k in range(len(RELATIONS))},
        graph_status=graph["status"],
    )
    if mode == "shuffled":
        rewired = list(edges)
        by_kind = defaultdict(list)
        for i, (_, _, kind) in enumerate(edges):
            by_kind[kind].append(i)
        changed = 0
        for positions in by_kind.values():
            if len(positions) < 2:
                continue
            before = [edges[i][1] for i in positions]
            after = list(before)
            rng.shuffle(after)
            if after == before and len(set(after)) > 1:
                after = after[1:] + after[:1]
            for index, key in zip(positions, after):
                query, old, kind = edges[index]
                rewired[index] = query, key, kind
                changed += int(old != key)
        edges = rewired
        info["changed_endpoints"] = changed
    return edges, info


class _RelationLayer(nn.Module):
    """Separate, normalized neighborhood attention at each query token."""

    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, RANK, bias=False)
        self.key = nn.Linear(hidden_size, RANK, bias=False)
        self.value = nn.Linear(hidden_size, RANK, bias=False)
        self.interaction = nn.Linear(hidden_size, RANK, bias=False)
        self.relation = nn.Embedding(len(RELATIONS), RANK)
        self.relation_bias = nn.Embedding(len(RELATIONS), 1)
        self.output = nn.Linear(RANK, hidden_size, bias=False)
        self.normalization = nn.LayerNorm(hidden_size)
        self.gate_logit = nn.Parameter(torch.tensor(-1.8))
        nn.init.zeros_(self.relation_bias.weight)
        with torch.no_grad():
            self.relation_bias.weight[PATTERN_START:, 0] = 0.2

    def forward(self, hidden, edges):
        if edges.numel() == 0:
            return hidden
        windows, width, dimension = hidden.shape
        window, query, key, kind = edges.unbind(1)
        if (torch.any(window < 0) or torch.any(window >= windows)
                or torch.any(query < 0) or torch.any(query >= width)
                or torch.any(key < 0) or torch.any(key >= width)):
            raise ValueError("relation edge outside encoded 512-token window")
        with torch.autocast(hidden.device.type, enabled=False):
            base = hidden.float()
            left = base[window, query]
            right = base[window, key]
            q, k = self.query(left), self.key(right)
            logits = ((q * k).sum(-1) / math.sqrt(RANK) +
                      self.relation_bias(kind).squeeze(-1))
            groups = window * width + query
            num_groups = windows * width
            maxima = logits.new_full((num_groups,), float("-inf"))
            maxima.scatter_reduce_(0, groups, logits, reduce="amax", include_self=True)
            probabilities = torch.exp(logits - maxima[groups])
            denominators = logits.new_zeros(num_groups).index_add_(0, groups, probabilities)
            weights = probabilities / denominators[groups].clamp_min(1e-12)
            messages = torch.tanh(
                self.value(right) + self.interaction(left * right) + self.relation(kind)
            )
            context = messages.new_zeros((num_groups, RANK))
            context.index_add_(0, groups, weights[:, None] * messages)
            context = self.normalization(self.output(context))
            present = denominators > 0
            delta = (torch.sigmoid(self.gate_logit) * context *
                     present[:, None]).reshape(windows, width, dimension)
            return (base + delta).to(hidden.dtype)


class StaticRelationEncoder(nn.Module):
    """Two differentiable sparse program-relation adapters inside CodeBERT."""

    def __init__(self, hidden_size, layer_indices):
        super().__init__()
        if not layer_indices:
            raise ValueError("at least one CodeBERT layer must be selected")
        self.layer_indices = tuple(sorted(set(layer_indices)))
        self.adapters = nn.ModuleDict({
            str(index): _RelationLayer(hidden_size) for index in self.layer_indices
        })

    def forward(self, hidden, window_edges, layer_index):
        return self.adapters[str(layer_index)](hidden, window_edges)
