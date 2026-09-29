"""Matched local-node/CFG encoders; no DGL or torch-geometric dependency."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn


@dataclass
class GraphBatch:
    attributes: torch.Tensor  # [nodes, 4]
    edges: torch.Tensor       # [2, edges], source -> target
    ptr: tuple[int, ...]      # graph boundaries in the node array
    token_pairs: torch.Tensor | None = None  # [pairs, 3]: node, batch row, source token
    ddg_edges: torch.Tensor | None = None  # [2, edges], reaching definition -> use
    program: ProgramBatch | None = None  # sparse check/update/use paths

    def to(self, device) -> "GraphBatch":
        pairs = None if self.token_pairs is None else self.token_pairs.to(device)
        ddg = None if self.ddg_edges is None else self.ddg_edges.to(device)
        return GraphBatch(self.attributes.to(device), self.edges.to(device), self.ptr, pairs,
                          ddg, None if self.program is None else self.program.to(device))


@dataclass
class ProgramBatch:
    features: torch.Tensor  # kind, comparison, branch, side, type, literal-present
    literals: torch.Tensor  # normalized numeric literal or zero
    edges: torch.Tensor  # value-version and copy links
    plain_edges: torch.Tensor  # ablation: all earlier events into the use
    path_ends: tuple[int, ...]
    use_ptr: tuple[int, ...]  # path boundaries per use
    function_ptr: tuple[int, ...]  # use boundaries per function
    relation_groups: tuple  # per use: check/use key, path check/ref/update pointers and versions

    def to(self, device) -> "ProgramBatch":
        return ProgramBatch(self.features.to(device), self.literals.to(device),
                            self.edges.to(device), self.plain_edges.to(device),
                            self.path_ends, self.use_ptr, self.function_ptr,
                            self.relation_groups)


def collate_programs(programs: list[dict], device="cpu") -> ProgramBatch:
    from .cfg_program import KINDS, TYPES, PROGRAM_SCHEMA

    features, literals, edges, plain_edges, path_ends = [], [], [], [], []
    use_ptr, function_ptr, relation_groups = [0], [0], []
    for program in programs:
        if program.get("schema") != PROGRAM_SCHEMA:
            raise ValueError("program batch requires current prepared structure schema")
        for use in program["slices"]:
            per_path_relations = []
            for path in use["paths"]:
                start = len(features)
                events = path["events"]
                if not events or events[-1]["kind"] not in {"use_array", "use_scalar"}:
                    raise ValueError("program path requires a final use event")
                for event in events:
                    literal = event.get("literal")
                    features.append((KINDS[event["kind"]], event["op"], event["branch"],
                                     event["side"], TYPES[event["type"]], int(literal is not None)))
                    literals.append((0.0 if literal is None else math.log1p(literal) / math.log1p(32767),))
                for a, b in path["edges"]:
                    if not 0 <= a < len(events) or not 0 <= b < len(events):
                        raise ValueError("program value-version edge is out of range")
                    edges.append((start + a, start + b))
                for a, b in path["plain_edges"]:
                    if not 0 <= a < len(events) or not 0 <= b < len(events):
                        raise ValueError("program plain edge is out of range")
                    plain_edges.append((start + a, start + b))
                mapped = {}
                for relation in path.get("relations", []):
                    indices = (relation["check_event"], relation["reference_event"],
                               *relation["update_events"])
                    if any(type(index) is not int or not 0 <= index < len(events)
                           for index in indices):
                        raise ValueError("program relation event is out of range")
                    key = (tuple(relation["check_span"]), relation["side"])
                    if key in mapped:
                        raise ValueError("repeated relation in one program path")
                    mapped[key] = (start + relation["check_event"],
                                   start + relation["reference_event"],
                                   tuple(start + item for item in relation["update_events"]),
                                   bool(relation["incomplete"] or path["incomplete"]),
                                   relation["target_version"], relation["reference_version"],
                                   relation["current_version"])
                per_path_relations.append(mapped)
                path_ends.append(len(features) - 1)
            keys = sorted(set().union(*(item.keys() for item in per_path_relations)))
            relation_groups.append(tuple((key, tuple(item.get(key) for item in per_path_relations))
                                         for key in keys))
            use_ptr.append(len(path_ends))
        function_ptr.append(len(use_ptr) - 1)
    return ProgramBatch(torch.tensor(features, dtype=torch.long).reshape(-1, 6),
                        torch.tensor(literals, dtype=torch.float32).reshape(-1, 1),
                        torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
                        torch.tensor(plain_edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
                        tuple(path_ends), tuple(use_ptr), tuple(function_ptr),
                        tuple(relation_groups)).to(device)


def collate_graphs(views: list[dict], encoded: list[list[list[int]]], device="cpu",
                   alignments: list[list[tuple[int, int]]] | None = None,
                   ddg_edges: list[list[tuple[int, int]]] | None = None,
                   programs: list[dict] | None = None) -> GraphBatch:
    if (not views or len(views) != len(encoded) or
            (alignments is not None and len(alignments) != len(views)) or
            (ddg_edges is not None and len(ddg_edges) != len(views)) or
            (programs is not None and len(programs) != len(views))):
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
                      ddg_tensor, collate_programs(programs, device) if programs is not None else None).to(device)


class AttributeCFGEncoder(nn.Module):
    """B/C, D/E, F/G, and CFG round-readout variants share this encoder.

    B and D use node-local self messages only; C and E also sum directed
    predecessor messages. Each pair shares parameters and update counts.
    D/E add the same projected, position-aligned source-token mean to the
    attribute embedding before the existing graph encoder. F/G additionally sum
    directed DDG messages through their own projection before the same GRU update.
    The raw attribute embedding is concatenated with the final or aggregated
    node state before attention pooling, as in the DeepDFA encoder design.
    These are learned summaries, not a sound static-analysis result.
    """
    def __init__(self, vocabulary_sizes: list[int], *, hidden_size: int = 128,
                 steps: int = 5, mode: str = "cfg", source_hidden_size: int | None = None):
        super().__init__()
        if (len(vocabulary_sizes) != 4 or min(vocabulary_sizes) < 2 or
                hidden_size < 4 or hidden_size % 4 or steps <= 0 or
                mode not in {"attributes", "cfg", "aligned_attributes", "aligned_cfg",
                             "cfg_ddg", "cfg_ddg_shuffled", "cfg_jk_mean", "cfg_jk_max",
                             "program_plain", "program_state"} or
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
        step_states = [] if self.mode in {"cfg_jk_mean", "cfg_jk_max"} else None
        for _ in range(self.steps):
            transformed = self.message(state)
            incoming = transformed.clone()
            if self.mode in {"cfg", "aligned_cfg", "cfg_ddg", "cfg_ddg_shuffled",
                             "cfg_jk_mean", "cfg_jk_max", "program_plain", "program_state"} and src.numel():
                incoming.index_add_(0, dst, transformed[src])
            if use_ddg and ddg_src.numel():
                ddg_transformed = self.ddg_message(state)
                incoming.index_add_(0, ddg_dst, ddg_transformed[ddg_src])
            state = self.update(incoming, state)
            if step_states is not None:
                step_states.append(state)
        if step_states is not None:
            stacked = torch.stack(step_states, dim=0)
            state = stacked.mean(dim=0) if self.mode == "cfg_jk_mean" else stacked.amax(dim=0)
        combined = torch.cat((state, x), dim=-1)
        scores = self.pool_gate(combined).squeeze(-1)
        return torch.stack([(scores[a:b].softmax(dim=0).unsqueeze(-1) * combined[a:b]).sum(dim=0)
                            for a, b in zip(batch.ptr[:-1], batch.ptr[1:])])


class ProgramCFGEncoder(AttributeCFGEncoder):
    """One graph branch: C attributes plus sparse, use-centered program states."""

    def __init__(self, vocabulary_sizes, *, hidden_size=128, steps=5, mode="program_state"):
        super().__init__(vocabulary_sizes, hidden_size=hidden_size, steps=steps, mode=mode)
        # Both controls allocate identical modules after the original C weights.
        from .cfg_program import TYPES
        self.program_embedding = nn.ModuleList(nn.Embedding(size, hidden_size) for size in
                                               (8, 7, 3, 3, len(TYPES) + 1, 2))
        self.literal_projection = nn.Linear(1, hidden_size, bias=False)
        self.program_message = nn.Linear(hidden_size, hidden_size)
        self.program_update = nn.GRUCell(hidden_size, hidden_size)
        self.path_projection = nn.Linear(6 * hidden_size, 2 * hidden_size)
        self.path_merge = nn.Linear(4 * hidden_size + 3, 2 * hidden_size)
        self.use_gate = nn.Linear(2 * hidden_size, 1)
        self.use_projection = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

    def forward(self, batch: GraphBatch, token_hidden=None):
        base = super().forward(batch, token_hidden)
        program = batch.program
        if program is None:
            raise ValueError("program graph variant requires prepared structural facts")
        if not program.path_ends:
            return base
        x = sum(layer(program.features[:, index])
                for index, layer in enumerate(self.program_embedding))
        x = x + self.literal_projection(program.literals)
        state = x
        edges = program.plain_edges if self.mode == "program_plain" else program.edges
        src, dst = edges
        for _ in range(self.steps):
            transformed = self.program_message(state)
            incoming = transformed.clone()
            if src.numel():
                incoming.index_add_(0, dst, transformed[src])
            state = self.program_update(incoming, state)
        zero = state.new_zeros((state.shape[-1],))
        use_states = []
        for use_index, (a, b) in enumerate(zip(program.use_ptr[:-1], program.use_ptr[1:])):
            groups = program.relation_groups[use_index]
            if not groups:
                # An observed use without an analyzable relation is retained as structure.
                groups = ((None, tuple(None for _ in range(a, b))),)
            relation_states = []
            for _, members in groups:
                path_vectors = []
                missing = incomplete = 0
                for relative, path_index in enumerate(range(a, b)):
                    end = program.path_ends[path_index]
                    member = members[relative]
                    if member is None:
                        missing += 1
                        check = reference = updates = zero
                        incomplete += 1
                    else:
                        check_index, reference_index, update_indices, flagged = member[:4]
                        check, reference = state[check_index], state[reference_index]
                        updates = zero
                        for update_index in update_indices:
                            updates = self.program_update(state[update_index], updates)
                        incomplete += int(flagged)
                    path_vectors.append(self.path_projection(torch.cat((
                        check, state[end], reference, updates, x[end],
                        x[member[1]] if member is not None else zero))))
                path_matrix = torch.stack(path_vectors)
                masks = state.new_tensor((missing / (b-a), incomplete / (b-a), (b-a) / 8.0))
                relation_states.append(self.path_merge(torch.cat((
                    path_matrix.amin(dim=0), path_matrix.amax(dim=0), masks))))
            relation_matrix = torch.stack(relation_states)
            weights = self.use_gate(relation_matrix).squeeze(-1).softmax(dim=0)
            use_states.append((weights.unsqueeze(-1) * relation_matrix).sum(dim=0))
        use_states = torch.stack(use_states)
        additions = []
        for a, b in zip(program.function_ptr[:-1], program.function_ptr[1:]):
            if a == b:
                additions.append(base.new_zeros((self.output_dim,)))
            else:
                weights = self.use_gate(use_states[a:b]).squeeze(-1).softmax(dim=0)
                additions.append(self.use_projection((weights.unsqueeze(-1) *
                                                      use_states[a:b]).sum(dim=0)))
        return base + torch.stack(additions)


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
            graph_class = ProgramCFGEncoder if mode in {"program_plain", "program_state"} else AttributeCFGEncoder
            graph = graph_class(vocabulary_sizes, hidden_size=hidden_size, steps=steps, mode=mode,
                                **({} if mode in {"program_plain", "program_state"} else
                                   {"source_hidden_size": self.task_modules["classifier"].in_features
                                    if mode.startswith("aligned_") else None}))
            head = nn.Linear(graph.output_dim, 1, bias=False)
            nn.init.zeros_(head.weight)
            self.task_modules["cfg_encoder"] = graph
            self.task_modules["cfg_classifier"] = head
        self.to(device)

    def _branch_outputs(self, input_ids, attention_mask, graph_batch: GraphBatch):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              use_cache=False).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float()
        source_logit = self.task_modules["classifier"](pooled).squeeze(-1)
        graph_vector = self.task_modules["cfg_encoder"](graph_batch, hidden)
        graph_logit = self.task_modules["cfg_classifier"](graph_vector).squeeze(-1)
        return pooled, graph_vector, source_logit, graph_logit

    def branch_logits(self, input_ids, attention_mask, graph_batch: GraphBatch):
        """Return source and graph logits from one encoder forward."""
        _, _, source_logit, graph_logit = self._branch_outputs(input_ids, attention_mask, graph_batch)
        return source_logit, graph_logit

    def representations(self, input_ids, attention_mask, graph_batch: GraphBatch):
        """Return the existing source pool, graph pool, and combined C logit."""
        pooled, graph_vector, source_logit, graph_logit = self._branch_outputs(
            input_ids, attention_mask, graph_batch)
        return pooled, graph_vector, source_logit + graph_logit

    def forward(self, input_ids, attention_mask, graph_batch: GraphBatch):
        source_logit, graph_logit = self.branch_logits(input_ids, attention_mask, graph_batch)
        return source_logit + graph_logit


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
    mode = ("cfg" if config["variant"] in {"cfg_double_ce", "cfg_rdrop",
                                          "cfg_source_aux", "cfg_source_detach",
                                          "cfg_rotation_fixed", "cfg_rotation_rotating",
                                          "lm_pretrain_cfg", "dep_pretrain_cfg",
                                          "composition_pretrain_cfg"} else
            "program_plain" if config["variant"] == "dep_pretrain_program_plain" else
            "program_state" if config["variant"] in {"dep_pretrain_program_state",
                                                      "composition_pretrain_program_state"}
            else config["variant"])
    return SourceGraphClassifier(source, vocabulary_sizes,
                                 hidden_size=config["graph_hidden_size"], steps=config["graph_steps"],
                                 mode=mode, device=device)
