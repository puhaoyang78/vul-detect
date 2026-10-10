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
    regions: RegionBatch | None = None
    behavior: list | None = None

    def to(self, device) -> "GraphBatch":
        pairs = None if self.token_pairs is None else self.token_pairs.to(device)
        ddg = None if self.ddg_edges is None else self.ddg_edges.to(device)
        return GraphBatch(self.attributes.to(device), self.edges.to(device), self.ptr, pairs,
                          ddg, None if self.program is None else self.program.to(device),
                          None if self.regions is None else self.regions.to(device), self.behavior)


@dataclass
class RegionBatch:
    members: tuple[tuple[int, ...], ...]
    edges: torch.Tensor
    ptr: tuple[int, ...]

    def to(self, device):
        return RegionBatch(self.members, self.edges.to(device), self.ptr)


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
                   programs: list[dict] | None = None,
                   regions: list[dict] | None = None, behavior: list[dict] | None = None) -> GraphBatch:
    if (not views or len(views) != len(encoded) or
            (alignments is not None and len(alignments) != len(views)) or
            (ddg_edges is not None and len(ddg_edges) != len(views)) or
            (programs is not None and len(programs) != len(views)) or
            (regions is not None and len(regions) != len(views)) or
            (behavior is not None and len(behavior) != len(views))):
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
    region_batch = None
    if regions is not None:
        members, cross, region_ptr = [], [], [0]
        for row, partition in enumerate(regions):
            n = ptr[row+1] - ptr[row]
            if sorted(node for group in partition["members"] for node in group) != list(range(n)):
                raise ValueError("regions must cover each CFG node exactly once")
            count = len(partition["members"])
            for a, b in partition["edges"]:
                if not 0 <= a < count or not 0 <= b < count:
                    raise ValueError("region CFG endpoint out of range")
                cross.append((a+region_ptr[-1], b+region_ptr[-1]))
            members.extend(tuple(node+ptr[row] for node in group) for group in partition["members"])
            region_ptr.append(region_ptr[-1]+count)
        region_batch = RegionBatch(tuple(members),
            torch.tensor(cross, dtype=torch.long).reshape(-1, 2).T.contiguous(), tuple(region_ptr))
    return GraphBatch(torch.tensor(attributes, dtype=torch.long), edge_tensor, tuple(ptr), pairs,
                      ddg_tensor, collate_programs(programs, device) if programs is not None else None,
                      region_batch, behavior).to(device)


class BoundFactEncoder(nn.Module):
    """Bind each value to its ordered AST role path before multiset pooling."""
    def __init__(self, sizes, roles):
        super().__init__()
        self.values = nn.ModuleList(nn.Embedding(size,16) for size in sizes)
        self.roles = nn.Embedding(roles,8)
        self.path_gru = nn.GRU(8,8,batch_first=True)
        self.bind = nn.ModuleList(nn.Sequential(nn.Linear(24,16),nn.ReLU(),nn.Linear(16,16)) for _ in sizes)
    def forward(self, rows, disabled=()):
        device=self.roles.weight.device
        paths=list(dict.fromkeys(tuple(fact[1]) for row in rows for facts in row for fact in facts))
        path_index={p:i for i,p in enumerate(paths)}
        if not paths:
            return self.roles.weight.new_zeros((len(rows),17*len(self.values)))
        lengths=torch.tensor([len(p) for p in paths])
        padded=torch.zeros((len(paths),int(lengths.max())),dtype=torch.long,device=device)
        for i,path in enumerate(paths):padded[i,:len(path)]=torch.tensor(path,device=device)
        packed=nn.utils.rnn.pack_padded_sequence(self.roles(padded),lengths,batch_first=True,enforce_sorted=False)
        _,hidden=self.path_gru(packed);hidden=hidden.squeeze(0)
        columns=[]
        for family,(embedding,bind) in enumerate(zip(self.values,self.bind)):
            facts=[(i,f) for i,row in enumerate(rows) for f in row[family] if f[2] and family not in disabled]
            pooled=hidden.new_zeros((len(rows),16));counts=hidden.new_zeros((len(rows),1))
            if facts:
                indices=torch.tensor([i for i,_ in facts],device=device)
                values=torch.tensor([f[0] for _,f in facts],device=device)
                role_ids=torch.tensor([path_index[tuple(f[1])] for _,f in facts],device=device)
                bound=bind(torch.cat((embedding(values),hidden[role_ids]),-1))
                pooled.index_add_(0,indices,bound)
                counts.index_add_(0,indices,counts.new_ones((len(facts),1)))
            columns.append(torch.cat((pooled/counts.clamp_min(1),counts.log1p()),-1))
        return torch.cat(columns,-1)


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
                 steps: int = 5, mode: str = "cfg", source_hidden_size: int | None = None,
                 behavior_sizes=None, disabled_families=()):
        super().__init__()
        if (len(vocabulary_sizes) != 4 or min(vocabulary_sizes) < 2 or
                hidden_size < 4 or hidden_size % 4 or steps <= 0 or
                mode not in {"attributes", "cfg", "aligned_attributes", "aligned_cfg",
                             "cfg_ddg", "cfg_ddg_shuffled", "cfg_jk_mean", "cfg_jk_max",
                             "program_plain", "program_state", "cfg_hierarchical", "region_local", "region_context", "behavior_nodes", "behavior_edges", "behavior_joint", "behavior_masked", "joint_nodes", "joint_edges", "joint", "joint_shuffled", "joint_transport", "joint_transport_shuffled"} or
                (mode in {"cfg_hierarchical", "region_local", "region_context"} and steps != 5) or
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
        if mode in {"region_local", "region_context"}:
            # Linear's constructor consumes RNG even though W is then zeroed.
            # Isolate it so shared modules and subsequent training keep C's stream.
            with torch.random.fork_rng(devices=[]):
                self.region_projection = nn.Linear(hidden_size, hidden_size, bias=False)
                nn.init.zeros_(self.region_projection.weight)
        if mode.startswith("behavior_"):
            from .cfg_behavior import FAMILIES
            if steps != 5 or behavior_sizes is None or len(behavior_sizes) != 12 or set(disabled_families)-set(FAMILIES):
                raise ValueError("behavior models require complete versioned attributes and five steps")
            self.disabled_families = set(disabled_families)
            with torch.random.fork_rng(devices=[]):
                if mode != "behavior_edges":
                    # Node ablation must not shift the shared edge-network initialization.
                    with torch.random.fork_rng(devices=[]):
                        self.node_behavior = nn.ModuleList(nn.EmbeddingBag(size, 16, mode="mean") for size in behavior_sizes[:8])
                        self.node_projection = nn.Linear(128, hidden_size)
                if mode != "behavior_nodes":
                    self.edge_behavior = nn.ModuleList(nn.EmbeddingBag(size, 8, mode="mean") for size in behavior_sizes[8:])
                    self.edge_hidden = nn.Linear(2*hidden_size+32, 128)
                    self.edge_output = nn.Linear(128, hidden_size, bias=False)
        if mode in {"joint_nodes", "joint_edges", "joint", "joint_shuffled", "joint_transport", "joint_transport_shuffled"}:
            from .cfg_behavior import FAMILIES
            if steps != 5 or behavior_sizes is None or len(behavior_sizes) != 13 or set(disabled_families)-set(FAMILIES):
                raise ValueError("joint model requires role-bound schema 2 and five steps")
            self.disabled_families=set(disabled_families)
            with torch.random.fork_rng(devices=[]):
                if mode != "joint_edges":
                    with torch.random.fork_rng(devices=[]):
                        self.bound_nodes=BoundFactEncoder(behavior_sizes[:8],behavior_sizes[-1])
                        self.node_residual=nn.Linear(8*17,hidden_size,bias=False)
                        nn.init.zeros_(self.node_residual.weight)
                if mode != "joint_nodes":
                    self.bound_edges=BoundFactEncoder(behavior_sizes[8:12],behavior_sizes[-1])
                    self.edge_projection=nn.Linear(4*17,32,bias=False)
                    self.semantic_hidden=nn.Linear(2*hidden_size+32,128)
                    self.semantic_output=nn.Linear(128,hidden_size,bias=False)
                    nn.init.zeros_(self.semantic_output.weight)
        self.diagnostics = {}
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
        behavior_edges = None
        if self.mode.startswith("behavior_"):
            from .cfg_behavior import NODE_FAMILIES, EDGE_FAMILIES
            if batch.behavior is None:
                raise ValueError("behavior model cannot load a cache without behavior attributes")
            nodes = [v for graph in batch.behavior for v in graph["nodes"]]
            alternatives = [alts for graph in batch.behavior for alts in graph["edges"]]
            if len(nodes) != len(x) or len(alternatives) != batch.edges.shape[1] or any(not a for a in alternatives):
                raise ValueError("behavior attributes do not match original CFG endpoints")
            def embed(families, layers, rows, masked=False):
                columns=[]
                for i,(family,layer) in enumerate(zip(families,layers)):
                    values,offsets=[],[]
                    for row in rows:
                        offsets.append(len(values))
                        values.extend([0] if masked or family in self.disabled_families else row[i])
                    columns.append(layer(torch.tensor(values,dtype=torch.long,device=x.device),
                                         torch.tensor(offsets,dtype=torch.long,device=x.device)))
                return torch.cat(columns,-1)
            if self.mode != "behavior_edges":
                x = self.node_projection(embed(NODE_FAMILIES,self.node_behavior,nodes))
            if self.mode != "behavior_nodes":
                expanded=[v for alts in alternatives for v in alts]
                if expanded:
                    behavior_edges=embed(EDGE_FAMILIES,self.edge_behavior,expanded,self.mode=="behavior_masked")
                    alternative_edge=torch.tensor([i for i,alts in enumerate(alternatives) for _ in alts],device=x.device)
                    alternative_counts=torch.tensor([len(a) for a in alternatives],device=x.device).unsqueeze(-1)
        original_x = x
        semantic = None
        if self.mode in {"joint_nodes", "joint_edges", "joint", "joint_shuffled", "joint_transport", "joint_transport_shuffled"}:
            from .cfg_behavior import NODE_FAMILIES, EDGE_FAMILIES
            if batch.behavior is None or any(g.get("schema")!=2 for g in batch.behavior):
                raise ValueError("joint model refuses unbound/old behavior cache")
            nodes=[v for graph in batch.behavior for v in graph["nodes"]]
            alternatives=[v for graph in batch.behavior for v in graph["edges"]]
            if len(nodes)!=len(x) or len(alternatives)!=batch.edges.shape[1]:
                raise ValueError("joint attributes/CFG mismatch")
            if self.mode!="joint_edges":
                delta=self.node_residual(self.bound_nodes(nodes,{i for i,f in enumerate(NODE_FAMILIES) if f in self.disabled_families}))
                x=x+delta
                self.diagnostics["node_residual_rms"]=delta.detach().square().mean().sqrt()
            if self.mode!="joint_nodes" and alternatives:
                expanded=[a for alts in alternatives for a in alts]
                neutral=[graph["neutral"] for graph in batch.behavior for alts in graph["edges"] for _ in alts]
                disabled={i for i,f in enumerate(EDGE_FAMILIES) if f in self.disabled_families}
                # Mirror unavailable families in the neutral record; UNKNOWN is not ordinary.
                neutral=[[[[fact[0],fact[1],int(any(f[2] for f in row[i]))] for fact in family]
                          for i,family in enumerate(n)] for row,n in zip(expanded,neutral)]
                semantic=self.edge_projection(self.bound_edges(expanded,disabled))
                neutral_semantic=self.edge_projection(self.bound_edges(neutral,disabled))
                alternative_edge=torch.tensor([i for i,alts in enumerate(alternatives) for _ in alts],device=x.device)
                alternative_counts=torch.tensor([len(a) for a in alternatives],device=x.device).unsqueeze(-1)
                semantic_available=x.new_tensor([any(f[2] for i,fs in enumerate(row) if i not in disabled for f in fs) for row in expanded]).unsqueeze(-1)
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
        for _ in range(2 if self.mode == "cfg_hierarchical" else self.steps):
            transformed = self.message(state)
            incoming = transformed.clone()
            if self.mode in {"cfg", "aligned_cfg", "cfg_ddg", "cfg_ddg_shuffled",
                             "cfg_jk_mean", "cfg_jk_max", "program_plain", "program_state",
                             "cfg_hierarchical", "region_local", "region_context", "behavior_nodes", "behavior_edges", "behavior_joint", "behavior_masked", "joint_nodes", "joint_edges", "joint", "joint_shuffled", "joint_transport", "joint_transport_shuffled"} and src.numel():
                incoming.index_add_(0, dst, transformed[src])
            if behavior_edges is not None:
                all_src,all_dst=batch.edges
                terms=self.edge_output(torch.relu(self.edge_hidden(torch.cat((
                    state[all_src[alternative_edge]],state[all_dst[alternative_edge]],behavior_edges),-1))))
                residual=state.new_zeros((batch.edges.shape[1],state.shape[-1]))
                residual.index_add_(0,alternative_edge,terms)
                residual=residual/alternative_counts
                # Every explicit self-loop adds its attributes, not a second self message.
                incoming.index_add_(0,all_dst,residual)
            if semantic is not None:
                all_src,all_dst=batch.edges
                endpoints=torch.cat((state[all_src[alternative_edge]],state[all_dst[alternative_edge]]),-1)
                actual=self.semantic_output(torch.relu(self.semantic_hidden(torch.cat((endpoints,semantic),-1))))
                neutral=self.semantic_output(torch.relu(self.semantic_hidden(torch.cat((endpoints,neutral_semantic),-1))))
                modulation=torch.tanh(actual-neutral)*semantic_available
                residual=state.new_zeros((batch.edges.shape[1],state.shape[-1]))
                # Transport can introduce semantic components even when the base
                # message is zero; legacy joint only rescales that message.
                terms = (modulation if self.mode in {"joint_transport", "joint_transport_shuffled"}
                         else transformed[all_src[alternative_edge]]*modulation)
                residual.index_add_(0,alternative_edge,terms)
                incoming.index_add_(0,all_dst,residual/alternative_counts)
                self.diagnostics["edge_modulation_rms"]=modulation.detach().square().mean().sqrt()
            if use_ddg and ddg_src.numel():
                ddg_transformed = self.ddg_message(state)
                incoming.index_add_(0, ddg_dst, ddg_transformed[ddg_src])
            state = self.update(incoming, state)
            if step_states is not None:
                step_states.append(state)
        if step_states is not None:
            stacked = torch.stack(step_states, dim=0)
            state = stacked.mean(dim=0) if self.mode == "cfg_jk_mean" else stacked.amax(dim=0)
        if self.mode in {"joint_nodes", "joint_edges", "joint", "joint_shuffled", "joint_transport", "joint_transport_shuffled"}:
            x = original_x
        ptr = batch.ptr
        if self.mode in {"cfg_hierarchical", "region_local", "region_context"}:
            if batch.regions is None:
                raise ValueError("hierarchical CFG requires the shared region partition")
            combined = torch.cat((state, x), dim=-1)
            gates = self.pool_gate(combined).squeeze(-1)
            summaries = []
            for members in batch.regions.members:
                indices = torch.tensor(members, device=state.device)
                weights = gates[indices].softmax(dim=0).unsqueeze(-1)
                values = combined if self.mode == "cfg_hierarchical" else state
                summaries.append((weights * values[indices]).sum(dim=0))
            node_state = state
            pooled_regions = torch.stack(summaries)
            if self.mode == "cfg_hierarchical":
                region_initial, region_x = pooled_regions.chunk(2, dim=-1)
            else:
                region_initial = pooled_regions
            state = region_initial
            src, dst = batch.regions.edges
            mask = src != dst
            src, dst = src[mask], dst[mask]
            for _ in range(3):
                transformed = self.message(state)
                incoming = transformed.clone()
                if self.mode != "region_local" and src.numel():
                    incoming.index_add_(0, dst, transformed[src])
                state = self.update(incoming, state)
            if self.mode == "cfg_hierarchical":
                x, ptr = region_x, batch.regions.ptr
            else:
                node_to_region = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
                for region, members in enumerate(batch.regions.members):
                    node_to_region[list(members)] = region
                context = (state - region_initial)[node_to_region]
                state = node_state + self.region_projection(context)
        combined = torch.cat((state, x), dim=-1)
        scores = self.pool_gate(combined).squeeze(-1)
        return torch.stack([(scores[a:b].softmax(dim=0).unsqueeze(-1) * combined[a:b]).sum(dim=0)
                            for a, b in zip(ptr[:-1], ptr[1:])])


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


POOL_VARIANTS = ('joint_source_pool', 'joint_structure_pool')


class ConditionalSourcePool(nn.Module):
    """One additive attention readout; fixed conditioning is the source-only control."""
    def __init__(self, source_size, graph_size, *, structural, rank=32):
        super().__init__()
        self.structural = structural
        self.tokens = nn.Linear(source_size, rank, bias=False)
        self.condition = nn.Linear(graph_size, rank)
        self.score = nn.Linear(rank, 1, bias=False)
        nn.init.zeros_(self.score.weight)
        self.diagnostics = {}

    def weights(self, hidden, mask, graph):
        valid = mask.bool()
        if not valid.any(dim=1).all():
            raise ValueError('source pooling requires at least one visible token')
        condition = graph if self.structural else torch.ones_like(graph)
        scores = self.score(torch.tanh(self.tokens(hidden.float()) +
                           self.condition(condition.float()).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(~valid, -torch.inf)
        numerator = (scores-scores.max(dim=1,keepdim=True).values).exp()
        return numerator/numerator.sum(dim=1,keepdim=True)

    def forward(self, hidden, mask, graph, mean):
        weights = self.weights(hidden, mask, graph)
        with torch.no_grad():
            self.diagnostics = {
                'pool_entropy':-(weights*weights.clamp_min(1e-30).log()).sum(dim=1),
                'pool_max_weight':weights.max(dim=1).values}
        uniform = mask.float()/mask.float().sum(dim=1,keepdim=True)
        # Retain the original (including mixed-precision) mean exactly at startup.
        # In real arithmetic this equals the attention-weighted token sum.
        correction = torch.einsum('bt,bth->bh',weights-uniform,hidden.float())
        return mean + correction


class SourceGraphClassifier(nn.Module):
    """Full-width source and graph heads with optional structure-conditioned source readout.

    The graph-head weights start at zero; initially the logits equal the source
    baseline exactly, including the zero-score initialization of learned pooling.
    Both source LoRA and graph modules are trained. This is NOT
    a guarantee of unchanged predictions after optimization.
    """
    def __init__(self, source_model, vocabulary_sizes: list[int], *, hidden_size: int,
                 steps: int, mode: str, device, behavior_sizes=None, disabled_families=(),
                 source_pool='mean'):
        super().__init__()
        self.encoder = source_model.encoder
        self.task_modules = source_model.task_modules
        with torch.random.fork_rng(devices=[]):
            graph_class = ProgramCFGEncoder if mode in {"program_plain", "program_state"} else AttributeCFGEncoder
            graph = graph_class(vocabulary_sizes, hidden_size=hidden_size, steps=steps, mode=mode,
                                **({} if mode in {"program_plain", "program_state"} else
                                   {"source_hidden_size": self.task_modules["classifier"].in_features
                                    if mode.startswith("aligned_") else None,
                                    "behavior_sizes": behavior_sizes, "disabled_families": disabled_families}))
            head = nn.Linear(graph.output_dim, 1, bias=False)
            nn.init.zeros_(head.weight)
            self.task_modules["cfg_encoder"] = graph
            self.task_modules["cfg_classifier"] = head
        if source_pool != 'mean':
            if source_pool not in ('source','structure') or mode != 'joint':
                raise ValueError('learned source pooling requires the joint graph')
            with torch.random.fork_rng(devices=[]):
                self.task_modules['source_pool'] = ConditionalSourcePool(
                    self.task_modules['classifier'].in_features,graph.output_dim,
                    structural=source_pool=='structure')
        self.to(device)

    def _branch_outputs(self, input_ids, attention_mask, graph_batch: GraphBatch, pooling_mask=None):
        readout_mask = attention_mask if pooling_mask is None else pooling_mask
        if (readout_mask.shape != attention_mask.shape or
                not ((readout_mask == 0) | (readout_mask == 1)).all() or
                (readout_mask.bool() & ~attention_mask.bool()).any() or
                not readout_mask.any(dim=1).all()):
            raise ValueError('pooling mask must select nonempty visible token sets')
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              use_cache=False).last_hidden_state
        mask = readout_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).float()
        graph_vector = self.task_modules["cfg_encoder"](graph_batch, hidden)
        if 'source_pool' in self.task_modules:
            pooled = self.task_modules['source_pool'](hidden,readout_mask,graph_vector,pooled)
        source_logit = self.task_modules["classifier"](pooled).squeeze(-1)
        graph_logit = self.task_modules["cfg_classifier"](graph_vector).squeeze(-1)
        return pooled, graph_vector, source_logit, graph_logit

    def branch_logits(self, input_ids, attention_mask, graph_batch: GraphBatch, pooling_mask=None):
        """Return source and graph logits from one encoder forward."""
        _, _, source_logit, graph_logit = self._branch_outputs(input_ids, attention_mask, graph_batch, pooling_mask)
        return source_logit, graph_logit

    def representations(self, input_ids, attention_mask, graph_batch: GraphBatch):
        """Return the existing source pool, graph pool, and combined C logit."""
        pooled, graph_vector, source_logit, graph_logit = self._branch_outputs(
            input_ids, attention_mask, graph_batch)
        return pooled, graph_vector, source_logit + graph_logit

    def forward(self, input_ids, attention_mask, graph_batch: GraphBatch, pooling_mask=None):
        source_logit, graph_logit = self.branch_logits(input_ids, attention_mask, graph_batch, pooling_mask)
        return source_logit + graph_logit


def build_model(base, config: dict, vocabulary_sizes: list[int] | None, device, *, training: bool):
    source = base.SequenceVulnerabilityClassifier(
        config["model_path"], variant="baseline", device=device,
        lora_r=config["lora_r"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        gradient_checkpointing=training and device.type == "cuda")
    if config["variant"] in {"baseline", "lm_pretrain_source", "dep_pretrain_source"}:
        return source
    if vocabulary_sizes is None:
        raise ValueError("graph vocabulary required")
    mode = ("cfg" if config["variant"] in {"cfg_double_ce", "cfg_rdrop",
                                          "cfg_source_aux", "cfg_source_detach",
                                          "cfg_rotation_fixed", "cfg_rotation_rotating",
                                          "lm_pretrain_cfg", "dep_pretrain_cfg",
                                          "composition_pretrain_cfg", "region_pretrain_cfg", "control_cfg"} else
            "joint_transport" if config["variant"] == "control_transport" else
            "joint_transport_shuffled" if config["variant"] == "control_transport_shuffled" else
            "joint" if config["variant"] in ('control_joint', *POOL_VARIANTS) else
            "cfg_hierarchical" if config["variant"] in {"dep_pretrain_hierarchical",
                                                         "region_pretrain_hierarchical"} else
            "program_plain" if config["variant"] == "dep_pretrain_program_plain" else
            "program_state" if config["variant"] in {"dep_pretrain_program_state",
                                                      "composition_pretrain_program_state"}
            else "attributes" if config["variant"] == "dep_pretrain_attributes"
            else config["variant"])
    return SourceGraphClassifier(source, vocabulary_sizes,
                                 hidden_size=config["graph_hidden_size"], steps=config["graph_steps"],
                                 mode=mode, device=device, behavior_sizes=config.get("behavior_sizes"),
                                 disabled_families=config.get("disabled_behavior_families", ()),
                                 source_pool={'joint_source_pool':'source','joint_structure_pool':'structure'}.get(config['variant'],'mean'))
