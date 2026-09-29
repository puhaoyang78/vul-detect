"""Conservative, source-positioned scalar definition/use supervision from cached Joern graphs.

Joern's saved graph has DDG but no REF edges. A C/C++ syntax tree supplies lexical
declaration scopes; an identifier is admitted only when its exact source span,
scope and scalar declaration are unambiguous. Missing DDG edges are never labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import torch
from torch import nn

from .cfg_alignment import _node_span, _source_coordinates, align_nodes
from .cfg_data import (atomic_json, cohort_hash, file_sha256, load_graphs,
                       read_records, source_hash)
from .syntax import parser_for, walk

MAX_POSITIVE = 32
MAX_NEGATIVE = 32
SCOPES = {"function_definition", "compound_statement", "for_statement", "if_statement",
          "while_statement", "switch_statement", "catch_clause"}
SCALAR_WORDS = {"_Bool", "bool", "char", "short", "int", "long", "float", "double",
                "signed", "unsigned"}
INTEGER_TYPES = {"int", "unsigned", "unsigned int", "short", "unsigned short",
                 "long", "unsigned long", "long long", "unsigned long long",
                 "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
                 "int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t",
                 "int64_t", "uint64_t"}

def _declarator_name(node):
    if node is None:
        return None, False
    simple = True
    while node is not None and node.type != "identifier":
        if node.type in {"function_declarator", "array_declarator", "pointer_declarator",
                         "reference_declarator", "abstract_pointer_declarator"}:
            simple = False
        child = node.child_by_field_name("declarator")
        if child is None:
            return None, False
        node = child
    return node, simple and node is not None


def _scope(node):
    while node is not None and node.type not in SCOPES:
        node = node.parent
    return node


def _declaration_names(node):
    if node.type == "parameter_declaration":
        return [node.child_by_field_name("declarator")]
    if node.type != "declaration":
        return []
    # field_name_for_child also finds the second and later declarators in `int a,b`.
    return [child for i, child in enumerate(node.children)
            if node.field_name_for_child(i) == "declarator"]


def _scalar_declaration(node, declarator, simple):
    if not simple:
        return False
    type_node = node.child_by_field_name("type")
    if type_node is None:
        return False
    words = type_node.text.decode("utf-8", errors="replace").split()
    if type_node.type != "primitive_type" or not words or any(w not in SCALAR_WORDS for w in words):
        return False
    return True


def _identifier_role(node):
    current = node
    while current.parent is not None:
        parent = current.parent
        if parent.type.startswith("preproc_"):
            return "macro_or_preprocessor"
        if parent.type == "field_expression":
            return "field_access"
        if parent.type in {"subscript_expression", "pointer_expression"}:
            return "alias_or_address"
        if parent.type in {"update_expression", "qualified_identifier",
                           "template_argument_list", "sizeof_expression", "alignof_expression",
                           "lambda_expression", "asm_statement", "gnu_asm_expression"}:
            return "unsupported_write_or_context"
        if parent.type == "call_expression":
            # Function-like macros cannot be ruled out without compilation context.
            return "call_or_macro_ambiguous"
        if parent.type == "assignment_expression":
            left = parent.child_by_field_name("left")
            if left is not None and left.start_byte <= node.start_byte < left.end_byte:
                op = parent.child_by_field_name("operator")
                return "write" if op is not None and op.text == b"=" else "unsupported_write_or_context"
        if parent.type in {"init_declarator", "parameter_declaration", "declaration"}:
            declarator = parent.child_by_field_name("declarator")
            if declarator is not None and declarator.start_byte <= node.start_byte < declarator.end_byte:
                return "write"
            if parent.type == "init_declarator":
                if declarator is not None and any(
                        part.type == "reference_declarator" for part in walk(declarator)):
                    return "alias_or_address"
                return "read"
        if parent.type == "function_definition":
            break
        current = parent
    return "read"


def lexical_bindings(source: str, language: str, *, integer_types=INTEGER_TYPES) -> tuple[
        dict[tuple[int, int], dict], str | None]:
    """Map exact character spans to scoped declaration IDs and read/write roles."""
    encoded = source.encode("utf-8")
    root = parser_for(language).parse(encoded).root_node
    functions = [node for node in walk(root) if node.type == "function_definition"]
    if root.has_error or len(functions) != 1:
        return {}, "syntax_error_or_multiple_functions"
    function = functions[0]
    byte_to_char = {0: 0}
    offset = 0
    for index, char in enumerate(source, 1):
        offset += len(char.encode("utf-8"))
        byte_to_char[offset] = index
    declarations = defaultdict(list)
    declaration_nodes = {}
    main_declarator = function.child_by_field_name("declarator")
    parameter_list = next((node for node in walk(main_declarator)
                           if node.type == "parameter_list"), None) if main_declarator else None
    for node in walk(function):
        if node.type not in {"declaration", "parameter_declaration"}:
            continue
        if node.type == "parameter_declaration" and node.parent != parameter_list:
            continue
        scope = function if node.type == "parameter_declaration" else _scope(node.parent)
        if scope is None:
            continue
        for declarator in _declaration_names(node):
            name_node, simple = _declarator_name(declarator)
            if name_node is None:
                continue
            name = name_node.text.decode("utf-8", errors="replace")
            type_node = node.child_by_field_name("type")
            primitive = (" ".join(type_node.text.decode("utf-8", errors="replace").split())
                         if type_node is not None and type_node.type in {"primitive_type", "type_identifier"}
                         else None)
            qualifiers = {child.text.decode("utf-8", errors="replace") for child in node.children
                          if child.type == "type_qualifier"}
            safe_integer = (simple and primitive in integer_types and
                            not qualifiers.intersection({"volatile", "_Atomic"}))
            entry = {"binding": f"{scope.start_byte}:{name_node.start_byte}:{name}",
                     "name": name, "start": name_node.start_byte,
                     "declaration_span": (byte_to_char[name_node.start_byte],
                                          byte_to_char[name_node.end_byte]),
                     "scalar": _scalar_declaration(node, declarator, simple),
                     "integer_type": primitive if safe_integer else None,
                     "parameter": node.type == "parameter_declaration"}
            declarations[(scope.id, name)].append(entry)
            declaration_nodes[(name_node.start_byte, name_node.end_byte)] = entry
    result = {}
    for node in walk(function):
        if node.type != "identifier" or node.start_byte not in byte_to_char or node.end_byte not in byte_to_char:
            continue
        span = (byte_to_char[node.start_byte], byte_to_char[node.end_byte])
        name = node.text.decode("utf-8", errors="replace")
        direct = declaration_nodes.get((node.start_byte, node.end_byte))
        if direct is None:
            scope = _scope(node)
            chosen = None
            while scope is not None:
                available = [item for item in declarations[(scope.id, name)]
                             if item["start"] <= node.start_byte]
                if available:
                    chosen = max(available, key=lambda item: item["start"])
                    # A duplicate declaration in one scope is not an auditable binding.
                    if len(available) != 1:
                        chosen = None
                    break
                scope = _scope(scope.parent)
            direct = chosen
        role = _identifier_role(node)
        value = {"binding": direct["binding"] if direct else None,
                 "scalar": bool(direct and direct["scalar"]), "role": role,
                 "integer_type": direct["integer_type"] if direct else None,
                 "parameter": bool(direct and direct["parameter"]),
                 "declaration_span": direct["declaration_span"] if direct else None,
                 "name": name}
        if span in result and result[span] != value:
            result[span] = {"binding": None, "scalar": False, "role": "unsupported", "name": name}
        else:
            result[span] = value
    return result, None


def _cfg_root(node_id, parents, cfg_ids):
    seen = set()
    while node_id is not None and node_id not in seen:
        if node_id in cfg_ids:
            return node_id
        seen.add(node_id)
        node_id = parents.get(node_id)
    return None


def _reachable(start, end, successors, blocked=frozenset()):
    if start in blocked or end in blocked:
        return False
    pending, seen = [start], set()
    while pending:
        node = pending.pop()
        if node == end:
            return True
        if node in seen or node in blocked:
            continue
        seen.add(node)
        pending.extend(successors[node])
    return False


def function_relations(record: dict, graph: dict, tokenizer, *, source_max_length: int = 2048,
                       prefix_tokens: int = 0, for_valid: bool = False) -> tuple[list[dict], Counter]:
    """Positive DDG pairs and CFG-proved killed old definitions for one function."""
    expected_split = "valid" if for_valid else "train"
    if record["split"] != expected_split:
        raise ValueError(f"relation extraction requires the {expected_split} split")
    source = record["raw_source"]
    stats = Counter(functions=1)
    language = record.get("resolved_language") or record.get("language")
    if language not in {"c", "cpp"}:
        stats["unsupported_language"] += 1
        return [], stats
    bindings, error = lexical_bindings(source, language)
    if error:
        stats[error] += 1
        return [], stats
    tokenized = tokenizer(source, add_special_tokens=False, truncation=True,
                          max_length=source_max_length, return_offsets_mapping=True)
    if len(tokenized["input_ids"]) != len(tokenized["offset_mapping"]):
        raise ValueError("tokenizer IDs and offsets have different lengths")
    nodes = {node["id"]: node for node in graph["nodes"]}
    identifiers = [node for node in graph["nodes"] if node["properties"]["kind"] == "IDENTIFIER"]
    locations = [dict(code=node["code"], **{key: value for key, value in node["properties"].items()
                                            if key in {"OFFSET", "OFFSET_END", "LINE_NUMBER", "COLUMN_NUMBER",
                                                       "LINE_NUMBER_END", "COLUMN_NUMBER_END"}})
                 for node in identifiers]
    statuses = []
    token_pairs, coverage = align_nodes(source, locations, tokenized["offset_mapping"],
                                        prefix_tokens, statuses=statuses)
    stats.update({f"alignment_{key}": value for key, value in coverage.items()})
    token_by_index = defaultdict(list)
    for index, token in token_pairs:
        token_by_index[index].append(token)
    units_to_chars, line_starts = _source_coordinates(source)
    ast_parents = {}
    cfg_ids, successors, ddg = set(), defaultdict(set), set()
    for edge in graph["edges"]:
        if edge["kind"] == "AST":
            ast_parents[edge["target"]] = edge["source"]
        elif edge["kind"] == "CFG":
            cfg_ids.update((edge["source"], edge["target"]))
            successors[edge["source"]].add(edge["target"])
        elif edge["kind"] == "DDG":
            ddg.add((edge["source"], edge["target"]))
    endpoint = {}
    repeated_spans = Counter()
    for index, node in enumerate(identifiers):
        if statuses[index][0] != "aligned_nodes":
            continue
        span, reason = _node_span(source, locations[index], units_to_chars, line_starts)
        if reason is not None:
            continue
        repeated_spans[span] += 1
        binding = bindings.get(span)
        if binding is None or binding["binding"] is None:
            stats["missing_binding"] += 1
            continue
        if not binding["scalar"]:
            stats["non_scalar_or_unsupported_binding"] += 1
            continue
        if binding["role"] not in {"read", "write"} or binding["name"] != node["code"]:
            stats[f"skip_{binding['role']}"] += 1
            continue
        cfg = _cfg_root(node["id"], ast_parents, cfg_ids)
        if cfg is None:
            stats["missing_cfg_site"] += 1
            continue
        endpoint[node["id"]] = dict(binding=binding["binding"], role=binding["role"],
                                    span=list(span), token=max(token_by_index[index]), cfg=cfg)
    unsafe_bindings = {value["binding"] for value in bindings.values()
                       if value["binding"] and value["role"] in
                       {"alias_or_address", "unsupported_write_or_context",
                        "call_or_macro_ambiguous"}}
    for key in list(endpoint):
        if endpoint[key]["binding"] in unsafe_bindings:
            stats["skip_alias_macro_or_unmodelled_binding"] += 1
            del endpoint[key]
    for key in list(endpoint):
        if repeated_spans[tuple(endpoint[key]["span"])] > 1:
            stats["ambiguous_joern_span"] += 1
            del endpoint[key]
    definitions, uses = {}, {}
    for key, item in endpoint.items():
        parent = nodes.get(ast_parents.get(key))
        if (item["role"] == "write" and parent is not None and
                parent["properties"].get("NAME") == "<operator>.assignment" and
                nodes[key]["properties"].get("ARGUMENT_INDEX") == 1):
            definitions[key] = item
        elif item["role"] == "read":
            uses[key] = item
    stats["raw_ddg_edges"] += sum(edge["kind"] == "DDG" for edge in graph["edges"])
    stats["candidate_definitions"] += len(definitions)
    stats["candidate_uses"] += len(uses)
    positives = set()
    for source_id, target_id in ddg:
        if source_id not in definitions or target_id not in uses:
            stats["ddg_outside_verified_scalar_pairs"] += 1
            continue
        definition, use = definitions[source_id], uses[target_id]
        if definition["binding"] != use["binding"]:
            stats["ddg_binding_conflict"] += 1
            continue
        positives.add((source_id, target_id))
    stats["verified_positive_candidates"] += len(positives)
    by_binding = defaultdict(list)
    for key, item in definitions.items():
        by_binding[item["binding"]].append((key, item))
    negatives = set()
    for _, use_id in positives:
        use = uses[use_id]
        for old_id, old in by_binding[use["binding"]]:
            if ((old_id, use_id) in positives or old["cfg"] == use["cfg"] or
                    old["span"][0] >= use["span"][0] or
                    not _reachable(old["cfg"], use["cfg"], successors)):
                continue
            blockers = {new["cfg"] for key, new in by_binding[use["binding"]]
                        if key != old_id and old["span"][0] < new["span"][0] < use["span"][0]
                        and new["cfg"] not in {old["cfg"], use["cfg"]}}
            if blockers and not _reachable(old["cfg"], use["cfg"], successors, blockers):
                negatives.add((old_id, use_id))
    stats["proved_negative_candidates"] += len(negatives)
    candidate_pairs = {(definition_id, use_id)
                       for use_id in {target for _, target in positives}
                       for definition_id, _ in by_binding[uses[use_id]["binding"]]}
    stats["unproved_non_edges"] += len(candidate_pairs - positives - negatives)
    def sort_key(pair):
        left, right = pair
        return (uses[right]["span"][0], definitions[left]["span"][0], left, right)
    chosen = [(pair, 1, "joern_ddg_scope_verified") for pair in sorted(positives, key=sort_key)[:MAX_POSITIVE]]
    chosen += [(pair, 0, "cfg_killed_same_binding") for pair in sorted(negatives, key=sort_key)[:MAX_NEGATIVE]]
    relations = []
    for (definition_id, use_id), label, provenance in sorted(chosen, key=lambda x: sort_key(x[0])):
        definition, use = definitions[definition_id], uses[use_id]
        relations.append({"sample_key": record["sample_key"], "split": expected_split,
                          "source_sha256": source_hash(source), "definition_node_id": definition_id,
                          "use_node_id": use_id, "definition_span": definition["span"],
                          "use_span": use["span"],
                          "definition_coordinates": {key: value for key, value in
                              nodes[definition_id]["properties"].items() if key in
                              {"OFFSET", "OFFSET_END", "LINE_NUMBER", "COLUMN_NUMBER",
                               "LINE_NUMBER_END", "COLUMN_NUMBER_END"}},
                          "use_coordinates": {key: value for key, value in
                              nodes[use_id]["properties"].items() if key in
                              {"OFFSET", "OFFSET_END", "LINE_NUMBER", "COLUMN_NUMBER",
                               "LINE_NUMBER_END", "COLUMN_NUMBER_END"}},
                          "definition_token": definition["token"],
                          "use_token": use["token"], "binding": definition["binding"],
                          "label": label, "source": provenance})
    stats["selected_positive"] += sum(row["label"] == 1 for row in relations)
    stats["selected_negative"] += sum(row["label"] == 0 for row in relations)
    stats["functions_with_relations"] += bool(relations)
    return relations, stats


def prepare_supervision(dataset: str, graphs_path: str, reference_run_dir: str,
                        output_dir: str, tokenizer, *, source_max_length: int = 2048) -> dict:
    """Write train-only relation rows; never rewrite the graph cache or source data."""
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"supervision output exists: {root}")
    reference = Path(reference_run_dir).resolve()
    c_config = json.loads((reference / "config.json").read_text())
    if source_max_length != c_config["source_max_length"] or source_max_length != 2048:
        raise ValueError("relation supervision must keep original C's 2048 source tokens")
    rows = read_records(dataset, c_config["source_dataset"])
    if (cohort_hash(rows) != c_config["cohort_sha256"] or
            str(Path(dataset).resolve()) != c_config["dataset"] or
            file_sha256(graphs_path) != c_config["graph_file_sha256"]):
        raise ValueError("dataset or graphs differ from original C")
    graphs = load_graphs(graphs_path, rows)
    train = [row for row in rows if row["split"] == "train"]
    if cohort_hash(train) != c_config["train_cohort_sha256"]:
        raise ValueError("train cohort differs from original C")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("relation supervision requires a fast tokenizer with offset_mapping")
    from .model import InputBuilder
    builder = InputBuilder(tokenizer, source_max_length=source_max_length, context_max_length=384)
    stats = Counter()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "relations.jsonl"
    with path.open("x", encoding="utf-8") as output, (root / "function_coverage.jsonl").open(
            "x", encoding="utf-8") as function_output:
        for index, row in enumerate(train, 1):
            relations, counts = function_relations(row, graphs[row["sample_key"]], tokenizer,
                                                  source_max_length=source_max_length,
                                                  prefix_tokens=len(builder.source_prefix))
            stats.update(counts)
            function_output.write(json.dumps({"sample_key": row["sample_key"],
                                              "source_sha256": source_hash(row["raw_source"]),
                                              "counts": dict(counts)}, ensure_ascii=False) + "\n")
            for relation in relations:
                output.write(json.dumps(relation, ensure_ascii=False) + "\n")
            if index % 200 == 0 or index == len(train):
                print(f"relation_supervision={index}/{len(train)}", flush=True)
    report = {"reference_run_dir": str(reference), "dataset": str(Path(dataset).resolve()),
              "graphs": str(Path(graphs_path).resolve()), "source_max_length": source_max_length,
              "train_cohort_sha256": cohort_hash(train), "relations_sha256": file_sha256(path),
              "function_coverage_sha256": file_sha256(root / "function_coverage.jsonl"),
              "counts": dict(stats), "max_positive_per_function": MAX_POSITIVE,
              "max_negative_per_function": MAX_NEGATIVE,
              "binding_source": "tree_sitter_lexical_scope_exact_span; cached_joern_ddg_and_cfg",
              "joern_ref_edges_in_cache": False}
    atomic_json(root / "audit.json", report)
    return report


def clm_shifted_loss(hidden, input_ids, attention_mask, lm_weight, prefix_tokens: int,
                     *, chunk_size: int = 128):
    """Next-token CLM on visible source and EOS; prefix and padding targets are ignored."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    if (hidden.shape[:2] != input_ids.shape or attention_mask.shape != input_ids.shape or
            lm_weight.shape[1] != hidden.shape[-1] or prefix_tokens < 1 or chunk_size < 1):
        raise ValueError("invalid CLM hidden/input/head shapes")
    target = input_ids[:, 1:].clone()
    target_positions = torch.arange(1, input_ids.shape[1], device=input_ids.device)
    valid = attention_mask[:, 1:].bool() & (target_positions >= prefix_tokens)
    target[~valid] = -100
    count = int(valid.sum().item())
    if not count:
        raise ValueError("no visible source tokens for CLM")

    def chunk_loss(states, labels):
        logits = F.linear(states, lm_weight).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                               ignore_index=-100, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, target.shape[1], chunk_size):
        labels = target[:, start:start + chunk_size]
        if (labels != -100).any():
            states = hidden[:, start:min(start + chunk_size, target.shape[1])]
            total = total + (checkpoint(chunk_loss, states, labels, use_reentrant=False)
                             if states.requires_grad else chunk_loss(states, labels))
    return total / count, count


class DirectedRelationHead(nn.Module):
    """Small ordered pair head, with independent definition/use projections."""

    def __init__(self, hidden_size: int, rank: int = 32):
        super().__init__()
        self.definition = nn.Linear(hidden_size, rank, bias=False)
        self.use = nn.Linear(hidden_size, rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(()))
        self.rank = rank

    def forward(self, definition_hidden, use_hidden):
        return (self.definition(definition_hidden.float()) *
                self.use(use_hidden.float())).sum(dim=-1) / self.rank**0.5 + self.bias


COMPOSED_HEAD_VERSION = 2


class ComposedRelationHead(nn.Module):
    """Causal source positions plus an explicit comparison query; no analyzer verdict input."""

    def __init__(self, hidden_size: int, rank: int = 32):
        super().__init__()
        from .cfg_program import OPS
        self.check = nn.Linear(hidden_size, rank, bias=False)
        self.target = nn.Linear(hidden_size, rank, bias=False)
        self.reference = nn.Linear(hidden_size, rank, bias=False)
        self.use = nn.Linear(hidden_size, rank, bias=False)
        self.definition = nn.Linear(hidden_size, rank, bias=False)
        self.update = nn.GRUCell(hidden_size, rank)
        # operator(6), desired branch(2), target side(2), reference kind(2),
        # no update, missing/incomplete path, operation, path count = 17 descriptors.
        self.path_query = nn.Sequential(nn.Linear(5 * rank + 17, rank), nn.Tanh(),
                                        nn.Linear(rank, rank), nn.Tanh())
        self.output = nn.Linear(2 * rank, 1)
        self.rank = rank

    def forward(self, source_hidden, query):
        import torch.nn.functional as F
        from .cfg_program import OPS
        if query["comparison"] not in OPS or query["branch"] not in (1, 2) or (
                query["target_side"] not in (1, 2)):
            raise ValueError("invalid composed query description")
        states = source_hidden.float()
        check = self.check(states[query["check_token"]])
        target = self.target(states[query["target_token"]])
        reference = self.reference(states[query["reference_token"]])
        use = self.use(states[query["use_token"]])
        descriptor = torch.cat((
            F.one_hot(torch.tensor(OPS[query["comparison"]]-1, device=states.device), 6),
            F.one_hot(torch.tensor(query["branch"]-1, device=states.device), 2),
            F.one_hot(torch.tensor(query["target_side"]-1, device=states.device), 2),
            F.one_hot(torch.tensor(int(query["reference_kind"] == "binding"), device=states.device), 2),
        )).float()
        path_vectors = []
        missing = incomplete = 0
        for path in query["paths"]:
            if path["missing"]:
                missing += 1
                continue
            incomplete += int(path["incomplete"])
            target_def = self.definition(states[path["target_definition_token"]])
            current_def = self.definition(states[path["current_definition_token"]])
            reference_def = self.definition(states[path["reference_definition_token"]])
            update = states.new_zeros((self.rank,))
            for token in path["update_tokens"]:
                update = self.update(states[token], update)
            path_vectors.append(torch.cat((check * use, target * reference,
                                           target_def * reference_def, current_def * use,
                                           update * use)))
        if not path_vectors:
            raise ValueError("composed query has no visible path")
        stacked = torch.stack(path_vectors)
        count = len(query["paths"])
        description = torch.cat((descriptor, states.new_tensor((
            float(all(not p["update_tokens"] for p in query["paths"] if not p["missing"])),
            missing / count, incomplete / count,
            float(query["operation"] == "array_index"), count / 8.0))))
        conditioned = self.path_query(torch.cat((
            stacked, description.expand(stacked.shape[0], -1)), dim=-1))
        return self.output(torch.cat((conditioned.amin(dim=0),
                                      conditioned.amax(dim=0)))).squeeze(-1)


def relation_function_loss(hidden, relation_batch, head):
    """Mean pair BCE per eligible function, then mean across eligible functions."""
    import torch
    import torch.nn.functional as F

    if len(relation_batch) != hidden.shape[0]:
        raise ValueError("relation batch size mismatch")
    function_losses, labels, logits = [], [], []
    for index, relations in enumerate(relation_batch):
        if not relations:
            continue
        expected = torch.tensor([row["label"] for row in relations],
                                dtype=torch.float32, device=hidden.device)
        if isinstance(head, ComposedRelationHead):
            predicted = torch.stack([head(hidden[index], row) for row in relations])
        else:
            indices = [torch.tensor([row[f"{stem}_token"] for row in relations],
                                    dtype=torch.long, device=hidden.device)
                       for stem in ("definition", "use")]
            if any((item < 0).any() or (item >= hidden.shape[1]).any() for item in indices):
                raise ValueError("relation token index is outside the encoded sequence")
            predicted = head(*(hidden[index, item] for item in indices))
        function_losses.append(F.binary_cross_entropy_with_logits(predicted.float(), expected))
        labels.extend(int(value) for value in expected.detach().cpu().tolist())
        logits.extend(predicted.detach().float().cpu().tolist())
    loss = torch.stack(function_losses).mean() if function_losses else hidden.float().sum() * 0
    return loss, labels, logits


def accumulation_window_counts(order, train, supervision, batch_size, accumulation):
    """Precount all and relation-bearing functions in each optimizer window."""
    if batch_size < 1 or accumulation < 1:
        raise ValueError("batch size and gradient accumulation must be positive")
    counts = []
    width = batch_size * accumulation
    for start in range(0, len(order), width):
        indices = order[start:start + width]
        total = len(indices)
        effective = sum(bool(supervision.get(train[index]["sample_key"])) for index in indices)
        counts.extend([(total, effective)] * ((total + batch_size - 1) // batch_size))
    return counts


def accumulation_window_loss(lm_loss, relation_loss, m, total, k, effective):
    """One microbatch's contribution to a function-averaged optimizer window."""
    if not (0 < m <= total and 0 <= k <= m and 0 <= effective <= total and k <= effective):
        raise ValueError("invalid optimizer-window function counts")
    return lm_loss * (m / total) + (relation_loss * (k / effective) if effective else 0)


def _load_qwen_lm_weight(model_path: str, hidden_size: int, device):
    from safetensors import safe_open
    import torch

    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(f"Qwen LM head index missing: {index}")
    weight_map = json.loads(index.read_text())["weight_map"]
    shard = weight_map.get("lm_head.weight")
    if shard is None:
        raise ValueError("pretrained Qwen LM head is absent; CLM must use original pretrained logits")
    with safe_open(str(root / shard), framework="pt", device="cpu") as source:
        weight = source.get_tensor("lm_head.weight")
    if weight.ndim != 2 or weight.shape[1] != hidden_size:
        raise ValueError("Qwen LM head and encoder hidden size differ")
    return weight.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)


def _checked_relation_rows(supervision_dir: str, train: list[dict], builder) -> tuple[dict, dict]:
    from .cfg_data import read_jsonl

    if not getattr(builder.tokenizer, "is_fast", False):
        raise ValueError("pretraining requires the same fast source tokenizer")
    root = Path(supervision_dir)
    audit = json.loads((root / "audit.json").read_text())
    path = root / "relations.jsonl"
    if audit["relations_sha256"] != file_sha256(path):
        raise ValueError("relation supervision has changed")
    expected = {row["sample_key"]: row for row in train}
    result = defaultdict(list)
    seen = set()
    for relation in read_jsonl(path):
        key = relation.get("sample_key")
        if key not in expected or relation.get("split") != "train" or (
                relation.get("source_sha256") != source_hash(expected[key]["raw_source"])):
            raise ValueError("relation row is outside the original C train cohort")
        pair = (key, relation["definition_node_id"], relation["use_node_id"])
        if pair in seen or relation.get("label") not in (0, 1):
            raise ValueError("duplicate or invalid relation label")
        seen.add(pair)
        result[key].append(relation)
    for key, relations in result.items():
        source = expected[key]["raw_source"]
        tokenized = builder.tokenizer(source, add_special_tokens=False, truncation=True,
                                      max_length=builder.source_max_length,
                                      return_offsets_mapping=True)
        if len(tokenized["input_ids"]) != len(tokenized["offset_mapping"]):
            raise ValueError("pretrain tokenizer IDs and source offsets differ")
        units_to_chars, line_starts = _source_coordinates(source)
        for relation in relations:
            locations = []
            for stem in ("definition", "use"):
                span = relation[f"{stem}_span"]
                if (not isinstance(span, list) or len(span) != 2 or
                        any(type(value) is not int for value in span) or
                        not 0 <= span[0] < span[1] <= len(source)):
                    raise ValueError("relation source span is invalid")
                location = dict(relation[f"{stem}_coordinates"], code=source[span[0]:span[1]])
                verified, reason = _node_span(source, location, units_to_chars, line_starts)
                if reason is not None or verified != tuple(span):
                    raise ValueError("relation node coordinates no longer match the source")
                locations.append(location)
            pairs, coverage = align_nodes(source, locations, tokenized["offset_mapping"],
                                          len(builder.source_prefix))
            by_node = defaultdict(list)
            for index, token in pairs:
                by_node[index].append(token)
            if (coverage["aligned_nodes"] != 2 or
                    max(by_node[0]) != relation["definition_token"] or
                    max(by_node[1]) != relation["use_token"]):
                raise ValueError("relation token offsets changed or lie outside visible source")
    if len(result) != audit["counts"]["functions_with_relations"]:
        raise ValueError("relation supervision coverage differs from audit")
    return dict(result), audit


def _checked_program_queries(program_dir, records, builder, split="train"):
    """Validate full model inputs and identity before excluding unknown labels."""
    from .cfg_data import read_jsonl
    from .cfg_program import PROGRAM_SCHEMA, OPS

    if split not in {"train", "valid"}:
        raise ValueError("program relation queries are train/valid only")
    root = Path(program_dir)
    audit = json.loads((root / "audit.json").read_text())
    if audit.get("program_schema") != PROGRAM_SCHEMA:
        raise ValueError("old program query schema is incompatible; prepare a new directory")
    path = root / f"{split}.queries.jsonl"
    if file_sha256(path) != audit[f"{split}_queries_sha256"]:
        raise ValueError("program query audit/content mismatch")
    expected = {row["sample_key"]: row for row in records}
    if any(row["split"] != split for row in records):
        raise ValueError("program query records have the wrong split")
    result = defaultdict(list)
    seen = {}
    for query in read_jsonl(path):
        key = query["sample_key"]
        if (query.get("schema") != PROGRAM_SCHEMA or key not in expected or
                query["split"] != split or
                query["source_sha256"] != source_hash(expected[key]["raw_source"])):
            raise ValueError("program query is outside the requested original C split/schema")
        if query["comparison"] not in OPS or query["branch"] not in (1, 2) or (
                query["target_side"] not in (1, 2)) or query["label"] not in (0, 1, None):
            raise ValueError(f"malformed composed query: {key}")
        length = len(builder.source_ids(expected[key])) - 1
        tokens = [query.get(stem) for stem in ("check_token", "use_token", "target_token",
                                               "reference_token")]
        for item in query["paths"]:
            if item["missing"]:
                continue
            tokens.extend((item.get("target_definition_token"),
                           item.get("current_definition_token"),
                           item.get("reference_definition_token")))
            tokens.extend(item["update_tokens"])
        if query["label"] is not None and (not tokens or any(type(t) is not int or
                                               not 0 <= t < length for t in tokens) or
                                            not query["check_token"] <= query["use_token"] or
                                            any(t > query["use_token"] for t in tokens)):
            raise ValueError(f"program query token is outside causal visible source: {key}")
        model_input = (key, query["check_token"], query["use_token"], query["target_token"],
                       query["reference_token"], query["comparison"], query["branch"],
                       query["target_side"], query["reference_kind"], query["operation"],
                       tuple(sorted((item["missing"], item["incomplete"],
                                     -1 if item.get("target_definition_token") is None else
                                     item["target_definition_token"],
                                     -1 if item.get("current_definition_token") is None else
                                     item["current_definition_token"],
                                     -1 if item.get("reference_definition_token") is None else
                                     item["reference_definition_token"],
                                     tuple(-1 if token is None else token
                                           for token in item.get("update_tokens", ())))
                                    for item in query["paths"])))
        semantic = (query["target_binding"], query["reference_binding"],
                    query["reference_version"], tuple(query["check_span"]),
                    tuple(query["use_span"]),
                    tuple((path.get("target_version"), path.get("reference_version"),
                           path.get("current_version")) for path in query["paths"]))
        if model_input in seen:
            previous = seen[model_input]
            raise ValueError(f"composed query input collision/conflict at {key} "
                             f"check={query['check_span']} use={query['use_span']}: "
                             f"previous={previous} current={(semantic, query['label'])}")
        seen[model_input] = (semantic, query["label"])
        if query["label"] is not None:
            result[key].append(query)
    return dict(result), audit


def pretrain_causal_dependency(reference_run_dir: str, supervision_dir: str, output_dir: str,
                               *, modes=("lm_pretrain", "dep_pretrain"), device="auto",
                               resume=False, base=None, program_dir=None):
    """One train-only source pass per mode; the relationship task shares Qwen's forward."""
    import gc
    import math
    import random
    import sys
    import torch
    from tqdm.auto import tqdm

    if not modes or len(modes) != len(set(modes)) or any(
            mode not in {"lm_pretrain", "dep_pretrain", "composition_pretrain"} for mode in modes):
        raise ValueError("unsupported source pretraining mode")
    if "composition_pretrain" in modes and (not program_dir or len(modes) != 1):
        raise ValueError("composition pretraining requires --program-dir and its own mode run")
    if any(mode in {"lm_pretrain", "dep_pretrain"} for mode in modes) and not supervision_dir:
        raise ValueError("existing source pretraining requires --supervision-dir")
    from .cfg_experiment import _base_module, _save_torch, _tokenizer
    base = _base_module() if base is None else base
    reference = Path(reference_run_dir).resolve()
    c_config = json.loads((reference / "config.json").read_text())
    if c_config["source_max_length"] != 2048 or c_config["seed"] != 42:
        raise ValueError("pretraining must use original C's 2048-token, seed-42 setting")
    rows = read_records(c_config["dataset"], c_config["source_dataset"])
    train = [row for row in rows if row["split"] == "train"]
    if (cohort_hash(rows) != c_config["cohort_sha256"] or
            cohort_hash(train) != c_config["train_cohort_sha256"] or
            file_sha256(c_config["graphs"]) != c_config["graph_file_sha256"]):
        raise ValueError("pretraining data or graph identity differs from original C")
    builder = _tokenizer(base, c_config)
    old_supervision = old_audit = None
    if any(mode in {"lm_pretrain", "dep_pretrain"} for mode in modes):
        old_supervision, old_audit = _checked_relation_rows(supervision_dir, train, builder)
        if old_audit["reference_run_dir"] != str(reference) or (
                old_audit["train_cohort_sha256"] != c_config["train_cohort_sha256"]):
            raise ValueError("supervision was not prepared for this original C train cohort")
        if "dep_pretrain" in modes and (old_audit["counts"].get("selected_positive", 0) == 0 or
                                         old_audit["counts"].get("selected_negative", 0) == 0):
            raise ValueError("verified relation supervision needs both classes")
    program_supervision = program_audit = None
    if "composition_pretrain" in modes:
        program_supervision, program_audit = _checked_program_queries(program_dir, train, builder)
        if (program_audit["reference_run_dir"] != str(reference) or
                program_audit["cohort_sha256"] != c_config["cohort_sha256"]):
            raise ValueError("program queries differ from the original C cohort")
        valid = [row for row in rows if row["split"] == "valid"]
        valid_supervision, _ = _checked_program_queries(program_dir, valid, builder, split="valid")
        for name, groups in (("train", program_supervision), ("valid", valid_supervision)):
            if not program_audit["readiness"][name]["eligible"]:
                raise ValueError(f"composition {name} independent-function coverage is too small; "
                                 "see prepared audit before formal training")
            labels = {relation["label"] for group in groups.values() for relation in group}
            if labels != {0, 1}:
                raise ValueError(f"composition supervision needs both verified labels in {name}; "
                                 "see prepared audit before formal training")
    root = Path(output_dir)
    config = {"c_config": c_config, "reference_run_dir": str(reference),
              "supervision_dir": str(Path(supervision_dir).resolve()) if old_audit else None,
              "relations_sha256": (old_audit["relations_sha256"] if old_audit else
                                    program_audit["train_queries_sha256"]),
              "pretrain_epochs": 1, "relation_alpha": 1.0, "relation_rank": 32,
              "relation_accumulation_normalization": "window_effective_functions",
              "objective": ("source_clm_plus_optional_composed_relation"
                            if "composition_pretrain" in modes else
                            "source_clm_plus_optional_scoped_scalar_relation")}
    if program_audit is not None:
        config.update(program_dir=str(Path(program_dir).resolve()),
                      program_sha256=program_audit["program_sha256"],
                      program_schema=program_audit["program_schema"],
                      composed_head_version=COMPOSED_HEAD_VERSION)
    root.mkdir(parents=True, exist_ok=True)
    from .cfg_data import output_lock
    with output_lock(root / "experiment"):
        meta = root / "config.json"
        if meta.exists():
            if not resume or json.loads(meta.read_text()) != config:
                raise ValueError("pretrain output exists or its configuration differs")
        else:
            if any((root / mode).exists() for mode in ("lm_pretrain", "dep_pretrain",
                                                      "composition_pretrain")):
                raise FileExistsError("pretrain mode directory exists without config")
            atomic_json(meta, config)
        resolved_device = base._resolve_device(device)
        for mode in modes:
            folder = root / mode
            complete = folder / "complete.json"
            if complete.exists():
                state = json.loads(complete.read_text())
                if (state["checkpoint_sha256"] != file_sha256(folder / "last.pt") or
                        state["mode"] != mode):
                    raise ValueError("completed pretraining checkpoint changed")
                print(f"skip_completed_pretrain={mode}", flush=True)
                continue
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"interrupted pretraining at {folder}; move it aside")
            folder.mkdir(exist_ok=True)
            base._seed_everything(c_config["seed"])
            encoder, hidden_size = base._build_lora_encoder(
                c_config["model_path"], device=resolved_device, lora_r=c_config["lora_r"],
                lora_alpha=c_config["lora_alpha"], lora_dropout=c_config["lora_dropout"],
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                gradient_checkpointing=resolved_device.type == "cuda")
            encoder.to(resolved_device)
            encoder.train()
            lm_weight = _load_qwen_lm_weight(c_config["model_path"], hidden_size, resolved_device)
            with torch.random.fork_rng(devices=[]):
                head = (DirectedRelationHead(hidden_size, config["relation_rank"]).to(resolved_device)
                        if mode == "dep_pretrain" else
                        ComposedRelationHead(hidden_size, config["relation_rank"]).to(resolved_device)
                        if mode == "composition_pretrain" else None)
            supervision = (program_supervision if mode == "composition_pretrain" else
                           old_supervision if old_supervision is not None else {})
            lora_parameters = [p for p in encoder.parameters() if p.requires_grad]
            groups = [{"params": lora_parameters, "lr": c_config["learning_rate"]}]
            if head is not None:
                groups.append({"params": list(head.parameters()), "lr": c_config["graph_learning_rate"]})
            optimizer = torch.optim.AdamW(groups, weight_decay=c_config["weight_decay"])
            trainable = lora_parameters + (list(head.parameters()) if head is not None else [])
            order = list(range(len(train)))
            random.Random(c_config["seed"]).shuffle(order)
            batch_size = c_config["batch_size"]
            accumulation = c_config["gradient_accumulation"]
            num_batches = math.ceil(len(order) / batch_size)
            window_counts = accumulation_window_counts(
                order, train, supervision if head is not None else {}, batch_size, accumulation)
            optimizer.zero_grad(set_to_none=True)
            total_lm = total_dep = 0.0
            total_functions = effective_functions = 0
            positive_relations = negative_relations = 0
            relation_labels, relation_scores = [], []
            optimizer_steps = 0
            with (folder / "history.jsonl").open("x", encoding="utf-8") as history:
                bar = tqdm(range(0, len(order), batch_size), total=num_batches,
                           desc=f"{mode} epoch 1/1", file=sys.stdout, dynamic_ncols=True, mininterval=1)
                for batch_index, start in enumerate(bar):
                    batch = [train[i] for i in order[start:start + batch_size]]
                    ids, mask = builder.sequence_batch(batch, variant="baseline",
                                                       excluded_groups=(), device=resolved_device)
                    hidden = encoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                    lm, _ = clm_shifted_loss(hidden, ids, mask, lm_weight, len(builder.source_prefix))
                    relation_batch = ([supervision.get(row["sample_key"], []) for row in batch]
                                      if head is not None else [])
                    if head is not None:
                        dep, labels, logits = relation_function_loss(hidden, relation_batch, head)
                        relation_labels.extend(labels)
                        relation_scores.extend(torch.sigmoid(torch.tensor(logits)).tolist())
                    else:
                        dep, labels = lm.new_zeros(()), []
                    m = len(batch)
                    k = sum(bool(relations) for relations in relation_batch)
                    M, K = window_counts[batch_index]
                    loss = accumulation_window_loss(lm, dep, m, M, k, K)
                    if not torch.isfinite(loss):
                        raise ValueError(f"non-finite pretrain loss at batch {batch_index + 1}")
                    loss.backward()
                    total_lm += float(lm.detach()) * m
                    total_dep += float(dep.detach()) * k
                    total_functions += m
                    effective_functions += k
                    positive_relations += sum(labels)
                    negative_relations += len(labels) - sum(labels)
                    if (batch_index + 1) % accumulation == 0 or batch_index + 1 == num_batches:
                        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        if not torch.isfinite(norm):
                            raise ValueError("non-finite pretrain gradient norm")
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
                        if (optimizer_steps == 1 or optimizer_steps % c_config["log_every"] == 0 or
                                batch_index + 1 == num_batches):
                            event = {"event": "step", "mode": mode, "step": optimizer_steps,
                                     "lm_loss_mean_so_far": total_lm / total_functions,
                                     "relation_loss_mean_so_far": (total_dep / effective_functions
                                                                   if effective_functions else 0.0),
                                     "total_functions_so_far": total_functions,
                                     "effective_relation_functions_so_far": effective_functions,
                                     "relation_coverage_so_far": effective_functions / total_functions,
                                     "relation_positive_so_far": positive_relations,
                                     "relation_negative_so_far": negative_relations}
                            history.write(json.dumps(event) + "\n")
                            history.flush()
                            print(json.dumps(event), flush=True)
                    bar.set_postfix(loss=f"{float(loss.detach()):.4f}", step=optimizer_steps, refresh=False)
                    del hidden, lm, dep, loss, ids, mask
                from .cfg_metrics import metrics
                relation_metrics = (metrics(relation_labels, relation_scores, 0.5)
                                    if relation_labels and len(set(relation_labels)) == 2 else None)
                summary = {"mode": mode, "epoch": 1, "samples": len(train),
                           "optimizer_steps": optimizer_steps, "lm_loss": total_lm / total_functions,
                           "relation_loss": (total_dep / effective_functions
                                             if effective_functions else 0.0),
                           "total_functions": total_functions,
                           "effective_relation_functions": effective_functions,
                           "relation_coverage": effective_functions / total_functions,
                           "relation_pairs": len(relation_labels),
                           "relation_positive": positive_relations,
                           "relation_negative": negative_relations,
                           "relation_metrics_train_online": relation_metrics}
                history.write(json.dumps({"event": "epoch", **summary}) + "\n")
                history.flush()
            checkpoint = {"mode": mode, "pretrain_config": config, "summary": summary,
                          "adapter_state": base._cpu_state(base.get_peft_model_state_dict(encoder)),
                          **({"relation_head_state": base._cpu_state(head.state_dict())} if head else {})}
            _save_torch(folder / "last.pt", checkpoint)
            atomic_json(folder / "complete.json", {"mode": mode,
                                                     "checkpoint_sha256": file_sha256(folder / "last.pt"),
                                                     "summary": summary})
            del encoder, head, lm_weight, optimizer, checkpoint, lora_parameters, trainable, groups
            gc.collect()
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()
    return {mode: json.loads((root / mode / "complete.json").read_text())["summary"] for mode in modes}


def _fixed_relation_predictions(encoder, head, builder, records, supervision, batch_size, device):
    """Score only verified pairs from a fixed, single-forward eval model."""
    from .cfg_metrics import metrics

    if batch_size < 1:
        raise ValueError("relation evaluation batch size must be positive")
    selected = [row for row in records if supervision.get(row["sample_key"])]
    predictions = []
    encoder.eval()
    head.eval()
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            ids, mask = builder.sequence_batch(batch, variant="baseline",
                                               excluded_groups=(), device=device)
            hidden = encoder(input_ids=ids, attention_mask=mask,
                             use_cache=False).last_hidden_state
            relation_batch = [supervision[row["sample_key"]] for row in batch]
            _, labels, logits = relation_function_loss(hidden, relation_batch, head)
            pairs = [relation for relations in relation_batch for relation in relations]
            if len(pairs) != len(labels) or len(pairs) != len(logits):
                raise ValueError("relation prediction order/count changed")
            scores = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).tolist()
            predictions.extend({**relation, "score": score,
                                "prediction": int(score >= 0.5), "threshold": 0.5}
                               for relation, score in zip(pairs, scores))
            if (start // batch_size + 1) % 100 == 0 or start + batch_size >= len(selected):
                print(f"fixed_relation_eval={start + len(batch)}/{len(selected)}", flush=True)
    labels = [row["label"] for row in predictions]
    scores = [row["score"] for row in predictions]
    sources = {row["sample_key"]: row["raw_source"] for row in records}
    examples = {"correct": [], "incorrect": []}
    for prediction in predictions:
        bucket = "correct" if prediction["prediction"] == prediction["label"] else "incorrect"
        if len(examples[bucket]) >= 4:
            continue
        source = sources[prediction["sample_key"]]
        example = {key: prediction[key] for key in ("sample_key", "split", "label",
                    "prediction", "score", "source") if key in prediction}
        for stem in ("definition", "check", "update", "use"):
            span = prediction.get(f"{stem}_span")
            if span is not None:
                example[f"{stem}_text"] = source[slice(*span)]
        examples[bucket].append(example)
    summary = {"total_functions": len(records), "effective_relation_functions": len(selected),
               "relation_coverage": len(selected) / len(records) if records else 0.0,
               "relation_pairs": len(predictions), "relation_positive": sum(labels),
               "relation_negative": len(labels) - sum(labels),
               "fixed_model": metrics(labels, scores, 0.5) if labels else None,
               "all_positive_reference": metrics(labels, [1.0] * len(labels), 0.5)
                                         if labels else None,
               "examples": examples}
    return predictions, summary


def evaluate_fixed_relations(pretrain_dir: str, output_dir: str, *, batch_size: int = 1,
                             device="auto", base=None, mode="dep_pretrain") -> dict:
    """Evaluate the saved stage-1 relation model on train and cached-graph valid pairs."""
    import gc
    from .cfg_experiment import _base_module, _tokenizer

    if mode not in {"dep_pretrain", "composition_pretrain"} or batch_size < 1:
        raise ValueError("fixed relation evaluation needs a valid relation mode and batch size")
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"fixed relation evaluation output exists: {root}")
    stage1 = Path(pretrain_dir).resolve()
    config = json.loads((stage1 / "config.json").read_text())
    c_config = config["c_config"]
    reference = Path(config["reference_run_dir"])
    if (json.loads((reference / "config.json").read_text()) != c_config or
            c_config["source_max_length"] != 2048 or c_config["seed"] != 42):
        raise ValueError("fixed relation evaluation differs from original C")
    rows = read_records(c_config["dataset"], c_config["source_dataset"])
    train = [row for row in rows if row["split"] == "train"]
    valid = [row for row in rows if row["split"] == "valid"]
    if (cohort_hash(rows) != c_config["cohort_sha256"] or
            cohort_hash(train) != c_config["train_cohort_sha256"] or
            cohort_hash(valid) != c_config["valid_cohort_sha256"] or
            file_sha256(c_config["graphs"]) != c_config["graph_file_sha256"]):
        raise ValueError("fixed relation evaluation data/graph identity changed")
    base = _base_module() if base is None else base
    builder = _tokenizer(base, c_config)
    if mode == "dep_pretrain":
        supervision, audit = _checked_relation_rows(config["supervision_dir"], train, builder)
        if (audit["reference_run_dir"] != str(reference.resolve()) or
                audit["train_cohort_sha256"] != c_config["train_cohort_sha256"] or
                audit["relations_sha256"] != config["relations_sha256"]):
            raise ValueError("train relation supervision differs from stage-1 provenance")
        graphs = load_graphs(c_config["graphs"], rows)
        valid_supervision = {}
        valid_counts = Counter()
        for row in valid:
            relations, counts = function_relations(
                row, graphs[row["sample_key"]], builder.tokenizer,
                source_max_length=builder.source_max_length,
                prefix_tokens=len(builder.source_prefix), for_valid=True)
            valid_counts.update(counts)
            if relations:
                valid_supervision[row["sample_key"]] = relations
        del graphs
        gc.collect()
    else:
        supervision, audit = _checked_program_queries(config["program_dir"], train, builder)
        valid_supervision, _ = _checked_program_queries(config["program_dir"], valid, builder,
                                                        split="valid")
        if (audit["reference_run_dir"] != str(reference.resolve()) or
                audit["cohort_sha256"] != c_config["cohort_sha256"] or
                audit["train_queries_sha256"] != config["relations_sha256"] or
                audit["program_sha256"] != config["program_sha256"] or
                audit["program_schema"] != config.get("program_schema")):
            raise ValueError("composition query provenance differs from stage 1")
        valid_counts = Counter(audit["splits"]["valid"])
    if mode == "composition_pretrain" and config.get("composed_head_version") != COMPOSED_HEAD_VERSION:
        raise ValueError("incompatible composed query head checkpoint")
    checkpoint_path = stage1 / mode / "last.pt"
    complete = json.loads((checkpoint_path.parent / "complete.json").read_text())
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if (complete["mode"] != mode or complete["checkpoint_sha256"] != checkpoint_sha256):
        raise ValueError("fixed relation checkpoint identity changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("mode") != mode or checkpoint.get("pretrain_config") != config or
            "relation_head_state" not in checkpoint):
        raise ValueError("fixed relation checkpoint lacks matching LoRA/relation head")
    resolved_device = base._resolve_device(device)
    encoder, hidden_size = base._build_lora_encoder(
        c_config["model_path"], device=resolved_device, lora_r=c_config["lora_r"],
        lora_alpha=c_config["lora_alpha"], lora_dropout=c_config["lora_dropout"],
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        gradient_checkpointing=False)
    encoder.to(resolved_device)
    base.set_peft_model_state_dict(encoder, checkpoint["adapter_state"])
    head_type = DirectedRelationHead if mode == "dep_pretrain" else ComposedRelationHead
    head = head_type(hidden_size, config["relation_rank"]).to(resolved_device)
    head.load_state_dict(checkpoint["relation_head_state"])
    root.mkdir(parents=True, exist_ok=False)
    result = {"pretrain_dir": str(stage1), "mode": mode,
              "checkpoint_sha256": checkpoint_sha256, "threshold": 0.5,
              "valid_relation_source": "existing_cached_joern",
              "valid_relation_audit": dict(valid_counts), "splits": {}}
    for split, records, relations in (("train", train, supervision),
                                       ("valid", valid, valid_supervision)):
        predictions, summary = _fixed_relation_predictions(
            encoder, head, builder, records, relations, batch_size, resolved_device)
        with (root / f"{split}.predictions.jsonl").open("x", encoding="utf-8") as output:
            for prediction in predictions:
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        result["splits"][split] = summary
    atomic_json(root / "metrics.json", result)
    return result
