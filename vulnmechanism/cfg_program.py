"""Conservative check/update/use facts shared by graph classification and pretraining.

Only exact Joern locations and a small, explicit C/C++ statement subset are used.
The local query verdict is exported for pretraining audits, never as a graph feature.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from .cfg_alignment import _node_span, _source_coordinates, align_nodes
from .cfg_data import atomic_json, cohort_hash, file_sha256, load_graphs, read_records, source_hash
from .cfg_dependency import lexical_bindings, _cfg_root, _reachable
from .syntax import parser_for, walk

MAX_PATHS = 8
MAX_USES = 16
MAX_EVENTS = 32
MAX_QUERIES = 32
MAX_UPDATES = 8
PROGRAM_SCHEMA = 7
OPS = {"<": 1, "<=": 2, ">": 3, ">=": 4, "==": 5, "!=": 6}
INVERSE = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
SWAP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
JOERN_OPS = {"<": "lessThan", "<=": "lessEqualsThan", ">": "greaterThan",
             ">=": "greaterEqualsThan", "==": "equals", "!=": "notEquals"}
KINDS = {"check": 1, "read": 2, "write": 3, "copy": 4,
         "use_array": 5, "use_scalar": 6, "reference": 7}
SIGNED_TYPES = {"int", "signed", "signed int", "signed short", "signed long",
                "signed long long", "signed char", "short", "long", "long long",
                "ssize_t", "ptrdiff_t",
                "intptr_t", "int8_t", "int16_t", "int32_t", "int64_t"}
UNSIGNED_TYPES = {"unsigned", "unsigned int", "unsigned short", "unsigned char",
                  "u_int8_t", "u_int16_t", "u_int32_t", "u_int64_t", "unsigned long",
                  "unsigned long long", "size_t", "uintptr_t", "uint8_t", "uint16_t",
                  "uint32_t", "uint64_t"}
TYPES = {name: index + 1 for index, name in enumerate(sorted(SIGNED_TYPES | UNSIGNED_TYPES))}


def _literal(node):
    if node is None or node.type != "number_literal":
        return None
    value = node.text.decode("ascii", errors="ignore")
    if not value or value.startswith("0") and len(value) > 1 and not value.startswith(("0x", "0X")):
        return None
    try:
        number = int(value, 0)
    except ValueError:
        return None
    return number if 0 <= number <= 32767 else None


def _type_literal_limit(name):
    return (127 if name in {"int8_t", "signed char"} else
            255 if name in {"uint8_t", "u_int8_t", "unsigned char"} else 32767)


def _holds(left, operator, right):
    return {"<": left < right, "<=": left <= right, ">": left > right,
            ">=": left >= right, "==": left == right, "!=": left != right}[operator]


def _edges(events):
    """Connect exact value versions; plain control discards this association."""
    precise, plain = set(), set()
    latest = {}
    checks = defaultdict(list)
    use_index = len(events) - 1
    for index, event in enumerate(events):
        binding, version = event["binding"], event["version"]
        previous = latest.get((binding, version))
        if previous is not None:
            precise.add((previous, index))
        if event["kind"] == "copy":
            source = latest.get((event["source_binding"], event["source_version"]))
            if source is not None:
                precise.add((source, index))
        # A write starts a new value version. The old value has no message edge
        # to it; a copy receives only from the confirmed source version.
        if event["kind"] == "check":
            checks[event["check_span_key"]].append(index)
        latest[(binding, version)] = index
        if index < use_index:
            plain.add((index, use_index))
    for members in checks.values():
        if len(members) == 2:
            precise.add((members[0], members[1]))
            precise.add((members[1], members[0]))
    return sorted(precise), sorted(plain)


class _Builder:
    def __init__(self, record, graph, tokenizer, source_max_length, prefix_tokens):
        self.record, self.source, self.graph = record, record["raw_source"], graph
        self.tokenizer, self.budget, self.prefix = tokenizer, source_max_length, prefix_tokens
        self.stats = Counter(functions=1)
        self.bindings, self.binding_error = lexical_bindings(
            self.source, record.get("resolved_language") or record.get("language"),
            integer_types=set(TYPES))
        self.root = parser_for(record.get("resolved_language") or record.get("language")).parse(
            self.source.encode("utf-8")).root_node
        self.function = next((node for node in walk(self.root) if node.type == "function_definition"), None)
        self.byte_to_char = {0: 0}
        offset = 0
        for position, char in enumerate(self.source, 1):
            offset += len(char.encode("utf-8"))
            self.byte_to_char[offset] = position
        units, lines = _source_coordinates(self.source)
        self.cache = defaultdict(list)
        self.locations = {}
        relevant = {"IDENTIFIER", "CONTROL_STRUCTURE", "RETURN", "CALL"}
        for node in graph["nodes"]:
            if node["properties"].get("kind") not in relevant or (len(node["code"]) > 512 and
                    node["properties"].get("kind") != "CONTROL_STRUCTURE"):
                continue
            location = dict(code=node["code"], **{key: value for key, value in node["properties"].items()
                                                   if key in {"OFFSET", "OFFSET_END", "LINE_NUMBER",
                                                              "COLUMN_NUMBER", "LINE_NUMBER_END",
                                                              "COLUMN_NUMBER_END"}})
            span, reason = _node_span(self.source, location, units, lines)
            if reason is None:
                kind = (node["properties"].get("NAME") if node["properties"].get("kind") == "CALL"
                        else node["properties"].get("kind"))
                self.cache[(kind, span)].append(node["id"])
                self.locations[node["id"]] = location
        parents, cfg_ids, successors = {}, set(), defaultdict(set)
        for edge in graph["edges"]:
            if edge["kind"] == "AST":
                parents[edge["target"]] = edge["source"]
            elif edge["kind"] == "CFG":
                cfg_ids.update((edge["source"], edge["target"]))
                successors[edge["source"]].add(edge["target"])
        self.parents, self.cfg_ids, self.successors = parents, cfg_ids, successors
        self.nodes_by_id = {node["id"]: node for node in graph["nodes"]}
        self.records = defaultdict(list)
        self.use_count = 0
        self.structural_truncation = False
        self.offsets = None
        if tokenizer is not None:
            tokenized = tokenizer(self.source, add_special_tokens=False, truncation=True,
                                  max_length=source_max_length, return_offsets_mapping=True)
            if len(tokenized["input_ids"]) != len(tokenized["offset_mapping"]):
                raise ValueError("program tokenizer IDs and offsets differ")
            self.offsets = tokenized["offset_mapping"]

    def span(self, node):
        if node is None:
            return None
        a, b = self.byte_to_char.get(node.start_byte), self.byte_to_char.get(node.end_byte)
        return (a, b) if a is not None and b is not None else None

    def match(self, kind, node):
        span = self.span(node)
        matches = self.cache.get((kind, span), [])
        if len(matches) != 1:
            self.stats["ambiguous_cached_position" if matches else f"missing_cached_{kind}"] += 1
            return None
        return matches[0]

    def cfg(self, node_id):
        return _cfg_root(node_id, self.parents, self.cfg_ids) if node_id else None

    def info(self, identifier):
        if identifier is None or identifier.type != "identifier":
            self.stats["unsupported_operand"] += 1
            return None
        span = self.span(identifier)
        binding = self.bindings.get(span)
        node_id = self.match("IDENTIFIER", identifier)
        if node_id is None or not binding or not binding.get("binding"):
            self.stats["missing_verified_binding"] += 1
            return None
        if binding.get("integer_type") not in TYPES:
            self.stats["unsupported_integer_type"] += 1
            return None
        return {"binding": binding["binding"], "type": binding["integer_type"],
                "parameter": binding["parameter"], "span": span, "node_id": node_id,
                "declaration_span": binding.get("declaration_span")}

    def state(self, path, info):
        key = info["binding"]
        if key not in path["env"]:
            path["env"][key] = {"version": 0, "value_key": (key, 0),
                                "constant": None, "initialized": bool(info["parameter"]),
                                "last_update": None,
                                "definition_span": info.get("declaration_span")}
        return path["env"][key]

    def event(self, kind, info, state, *, op=0, branch=0, side=0, literal=None,
              span=None, **extra):
        return {"kind": kind, "binding": info["binding"], "version": state["version"],
                "type": info["type"], "span": list(span or info["span"]), "op": op,
                "branch": branch, "side": side, "literal": literal,
                "value_key": state["value_key"], **extra}

    def token(self, node_id):
        if self.offsets is None:
            return None
        pairs, coverage = align_nodes(self.source, [self.locations[node_id]], self.offsets,
                                      self.prefix)
        if coverage["aligned_nodes"] != 1:
            self.stats["query_outside_visible_or_unaligned"] += 1
            return None
        return max(token for _, token in pairs)

    def token_span(self, span):
        if self.offsets is None:
            return None
        start, end = span
        if (start is None or end is None or not 0 <= start < end <= len(self.source)):
            return None
        source_units = len(self.source[:start].encode("utf-16-le")) // 2
        end_units = len(self.source[:end].encode("utf-16-le")) // 2
        location = {"code": self.source[start:end], "OFFSET": source_units,
                    "OFFSET_END": end_units}
        pairs, coverage = align_nodes(self.source, [location], self.offsets, self.prefix)
        if coverage["aligned_nodes"] != 1:
            self.stats["query_outside_visible_or_unaligned"] += 1
            return None
        return max(token for _, token in pairs)

    def compare(self, path, node):
        condition = node.child_by_field_name("condition")
        while condition is not None and condition.type == "parenthesized_expression":
            condition = condition.named_children[0] if len(condition.named_children) == 1 else None
        if condition is None or condition.type != "binary_expression":
            self.stats["unsupported_check"] += 1
            return None
        operator_node = condition.child_by_field_name("operator")
        operator = operator_node.text.decode("ascii", errors="ignore") if operator_node else None
        if operator not in OPS:
            self.stats["unsupported_check"] += 1
            return None
        comparison_id = self.match("<operator>." + JOERN_OPS[operator], condition)
        comparison_cfg = self.cfg(comparison_id)
        # Joern may serialize a large IF's code/range differently from tree-sitter.
        # Its AST ancestry and exact UTF-16 start coordinate still identify the
        # controlling IF without searching source text or using CDG as a substitute.
        control_id = None
        ancestor = self.parents.get(comparison_id)
        while ancestor is not None:
            candidate = self.nodes_by_id[ancestor]
            if candidate["properties"].get("kind") == "CONTROL_STRUCTURE":
                start = self.span(node)[0]
                units = len(self.source[:start].encode("utf-16-le")) // 2
                if (candidate["properties"].get("CONTROL_STRUCTURE_TYPE") == "IF" and
                        candidate["properties"].get("OFFSET") == units):
                    control_id = ancestor
                    if ("CONTROL_STRUCTURE", self.span(node)) not in self.cache:
                        self.stats["control_ast_start_recovered"] += 1
                break
            ancestor = self.parents.get(ancestor)
        if control_id is None or comparison_id is None or comparison_cfg is None:
            self.stats["missing_check_cfg_site"] += 1
            return None
        operands = []
        for side in ("left", "right"):
            child = condition.child_by_field_name(side)
            if child is not None and child.type == "identifier":
                info = self.info(child)
                if info is None:
                    return None
                state = self.state(path, info)
                if not state["initialized"]:
                    self.stats["uninitialized_check_operand"] += 1
                    return None
                operands.append({"info": info, "value_key": state["value_key"],
                                 "constant": state["constant"], "version": state["version"],
                                 "definition_span": state["definition_span"]})
            else:
                literal = _literal(child)
                if literal is None:
                    self.stats["unsupported_check_operand"] += 1
                    return None
                operands.append({"literal": literal, "span": self.span(child)})
        if not any("info" in item for item in operands):
            self.stats["constant_only_check"] += 1
            return None
        if len({item["info"]["type"] for item in operands if "info" in item}) != 1:
            self.stats["implicit_conversion_unknown"] += 1
            return None
        operand_type = next(item["info"]["type"] for item in operands if "info" in item)
        limit = _type_literal_limit(operand_type)
        if any(item.get("literal", 0) is not None and item.get("literal", 0) > limit
               for item in operands if "literal" in item):
            self.stats["literal_conversion_unknown"] += 1
            return None
        return {"span": self.span(condition), "operator": operator, "operands": operands,
                "node_id": comparison_id, "cfg": comparison_cfg}

    def branch_possible(self, path, check, branch):
        """Exclude only contradictions in exact scalar/literal checks on the same value."""
        constraints = list(path["active"]) + [dict(check, branch=branch)]
        bounds = {}
        excluded = defaultdict(set)
        for item in constraints:
            left, right = item["operands"]
            if "info" in left and "info" in right and left["value_key"] == right["value_key"]:
                truth = _holds(0, item["operator"], 0)
                if truth != (item["branch"] == 1):
                    return False
                continue
            for side, operand in enumerate((left, right)):
                if "info" not in operand:
                    continue
                other = (right, left)[side]
                constant = other.get("literal", other.get("constant"))
                if constant is None:
                    continue
                operator = item["operator"] if side == 0 else SWAP[item["operator"]]
                if item["branch"] == 2:
                    operator = INVERSE[operator]
                key = operand["value_key"]
                low, high = bounds.get(key, (None, None))
                if operator == "==":
                    low = high = constant
                elif operator == "!=":
                    excluded[key].add(constant)
                elif operator == "<":
                    high = min(high, constant - 1) if high is not None else constant - 1
                elif operator == "<=":
                    high = min(high, constant) if high is not None else constant
                elif operator == ">":
                    low = max(low, constant + 1) if low is not None else constant + 1
                elif operator == ">=":
                    low = max(low, constant) if low is not None else constant
                previous = bounds.get(key)
                if previous is not None:
                    old_low, old_high = previous
                    if old_low is not None:
                        low = max(low, old_low) if low is not None else old_low
                    if old_high is not None:
                        high = min(high, old_high) if high is not None else old_high
                if (low is not None and high is not None and
                        (low > high or (low == high and low in excluded[key]))):
                    return False
                bounds[key] = (low, high)
        return True

    def assign(self, path, target, value, operation):
        target_info = self.info(target)
        if target_info is None:
            self.reset_unknown_effect(path, "unscoped_assignment_effect")
            return
        if operation.type == "assignment_expression":
            if self.match("<operator>.assignment", operation) is None:
                self.reset_unknown_effect(path, "unscoped_assignment_effect")
                return
        elif self.match("IDENTIFIER", target) is None:
            self.reset_unknown_effect(path, "unscoped_assignment_effect")
            return
        if value is None:
            self.state(path, target_info)
            return
        if any(item.type in {"call_expression", "update_expression", "assignment_expression"}
               for item in walk(value)):
            self.reset_unknown_effect(path, "unbounded_rhs_effect")
            self.kill_binding(path, target, "unbounded_rhs_target")
            return
        if value is not None and value.type == "identifier":
            source_info = self.info(value)
            if source_info is None or source_info["type"] != target_info["type"]:
                self.stats["implicit_conversion_unknown"] += 1
                self.kill_binding(path, target, "scoped_copy_conversion")
                return
            source_state = self.state(path, source_info)
            if not source_state["initialized"]:
                self.stats["uninitialized_copy"] += 1
                self.kill_binding(path, target, "scoped_uninitialized_copy")
                return
            path["trace"].append(self.event("read", source_info, source_state))
            new_value_key, constant = source_state["value_key"], source_state["constant"]
            kind = "copy"
        else:
            constant = _literal(value)
            if constant is None or constant > _type_literal_limit(target_info["type"]):
                self.stats["unsupported_update"] += 1
                self.kill_binding(path, target, "scoped_unsupported_update")
                return
            new_value_key = (target_info["binding"], self.span(operation))
            kind = "write"
        old = self.state(path, target_info)
        new = {"version": old["version"] + 1, "value_key": new_value_key,
               "constant": constant, "initialized": True,
               "last_update": self.span(operation),
               "definition_span": self.span(operation)}
        path["trace"].append(self.event(
            kind, target_info, new, span=self.span(operation), literal=constant,
            old_version=old["version"], **({"source_binding": source_info["binding"],
                                            "source_version": source_state["version"]}
                                           if kind == "copy" else {})))
        path["env"][target_info["binding"]] = new
        self.stats["copy_updates" if kind == "copy" else "constant_updates"] += 1

    def candidate(self, path, check, side, target_state, desired_branch):
        """Check-time reference version is fixed; only the target value evolves."""
        target = check["operands"][side]
        reference = check["operands"][1 - side]
        current = target_state
        if path["incomplete"] or current is None:
            return None, "incomplete_path"
        if current["value_key"] == target["value_key"]:
            return int(desired_branch == check["branch"]), "unchanged_check_time_value"
        reference_constant = reference.get("literal", reference.get("constant"))
        if current["constant"] is not None and reference_constant is not None:
            values = [None, None]
            values[side], values[1 - side] = current["constant"], reference_constant
            operator = check["operator"] if desired_branch == 1 else INVERSE[check["operator"]]
            return int(_holds(values[0], operator, values[1])), "confirmed_constant_comparison"
        return None, "unproved_current_value"

    def record_use(self, path, target_info, use_id, use_span, operation):
        state = self.state(path, target_info)
        use_cfg = self.cfg(use_id)
        if not state["initialized"] or use_cfg is None:
            self.stats["uninitialized_or_missing_use_cfg"] += 1
            return
        related = []
        for check in path["active"]:
            if not _reachable(check["cfg"], use_cfg, self.successors):
                self.stats["check_not_cfg_reachable"] += 1
                continue
            for side, item in enumerate(check["operands"]):
                if "info" in item and (item["info"]["binding"] == target_info["binding"] or
                                       item["value_key"] == state["value_key"]):
                    related.append((check, side))
        relevant = {target_info["binding"]}
        for check, _ in related:
            relevant.update(item["info"]["binding"] for item in check["operands"] if "info" in item)
        for event in reversed(path["trace"]):
            if event["kind"] == "copy" and event["binding"] in relevant:
                relevant.add(event["source_binding"])
        check_spans = {check["span"] for check, _ in related}
        events = [dict(event) for event in path["trace"]
                  if event["binding"] in relevant and
                  (event["kind"] != "check" or event["check_span_key"] in check_spans)]
        relations = []
        relation_updates = {}
        for check, side in related:
            target, reference = check["operands"][side], check["operands"][1-side]
            target_event = next((i for i, event in enumerate(events) if
                                 event["kind"] == "check" and event["check_span_key"] == check["span"] and
                                 event["side"] == side + 1 and event["binding"] == target["info"]["binding"]), None)
            if target_event is None:
                self.stats["missing_check_event"] += 1
                continue
            if "info" in reference:
                reference_event = next((i for i, event in enumerate(events) if
                                        event["kind"] == "check" and event["check_span_key"] == check["span"] and
                                        event["side"] == 2 - side and
                                        event["binding"] == reference["info"]["binding"]), None)
            else:
                reference_event = len(events)
                literal_info = dict(target["info"], binding=("literal", check["span"], side))
                events.append(self.event("reference", literal_info,
                                         {"version": 0, "value_key": ("literal", check["span"]),
                                          "constant": reference["literal"]},
                                         side=2-side, literal=reference["literal"], span=reference["span"]))
            if reference_event is None:
                self.stats["missing_reference_event"] += 1
                continue
            # The query fixes the reference at check time. Follow the target's
            # copy lineage and its subsequent writes; unrelated reference writes
            # do not become artificial query updates.
            lineage = {target_info["binding"], target["info"]["binding"]}
            for event in reversed(events):
                if event["kind"] == "copy" and event["binding"] in lineage:
                    lineage.add(event["source_binding"])
            update_indices = [i for i, event in enumerate(events) if i > target_event and
                              event["kind"] in {"write", "copy"} and
                              event["binding"] in lineage]
            relation_updates[(check["span"], side)] = update_indices
            if len(update_indices) > MAX_UPDATES:
                self.stats["too_many_related_updates"] += 1
            relations.append({"check_span": list(check["span"]), "operator": check["operator"],
                              "side": side + 1, "branch": check["branch"],
                              "check_event": target_event, "reference_event": reference_event,
                              "target_version": target["version"],
                              "reference_version": reference.get("version"),
                              "current_version": state["version"],
                              "update_events": update_indices[:MAX_UPDATES],
                              "incomplete": bool(path["incomplete_history"] or
                                                 len(update_indices) > MAX_UPDATES)})
        events.append(self.event("use_array" if operation == "array_index" else "use_scalar",
                                 target_info, state, span=use_span, literal=state["constant"]))
        if len(events) > MAX_EVENTS:
            self.stats["too_many_events_for_use"] += 1
            # Retain a flagged path, rather than silently proving a statement from the others.
            events = [events[-1]]
            relations = []
            path_incomplete = True
        else:
            path_incomplete = bool(path["incomplete_history"])
        edges, plain_edges = _edges(events)
        candidates = {}
        for check, side in related:
            target, reference = check["operands"][side], check["operands"][1-side]
            update_indices = relation_updates.get((check["span"], side), [])
            updates = [events[i]["span"] for i in update_indices]
            for desired in (1, 2):
                verdict, reason = self.candidate(path, check, side, state, desired)
                if path["incomplete"] or len(updates) > MAX_UPDATES:
                    verdict, reason = None, "incomplete_or_update_limit"
                candidates[(check["span"], side, desired)] = {
                    "label": verdict, "source": reason, "check": check,
                    "side": side, "target": target, "reference": reference,
                    "current": state, "updates": updates[:MAX_UPDATES]}
        self.records[(use_span, operation)].append({
            "slice": {"events": events, "edges": edges, "plain_edges": plain_edges,
                      "relations": relations, "incomplete": path_incomplete},
            "candidates": candidates, "use_node_id": use_id,
            "target_operand_span": target_info["span"], "incomplete": path_incomplete})
        self.use_count += 1
        self.stats[f"verified_{operation}_paths"] += 1

    def use(self, path, expression):
        if any(item.type in {"call_expression", "update_expression", "assignment_expression"}
               for item in walk(expression)):
            self.stats["use_context_effect_unknown"] += 1
            return
        if self.use_count >= MAX_USES:
            self.stats["use_limit"] += 1
            self.structural_truncation = True
            path["incomplete"] = True
            return
        subscripts = [item for item in walk(expression) if item.type == "subscript_expression"]
        for subscript in subscripts:
            argument, index = (subscript.child_by_field_name(name) for name in ("argument", "index"))
            if argument is None or index is None or index.type != "identifier" or any(
                    item.type == "call_expression" for item in walk(argument)):
                self.stats["unsupported_array_operand"] += 1
                continue
            access_id = self.match("<operator>.indirectIndexAccess", subscript)
            if access_id is None:
                continue
            target_info = self.info(index)
            if target_info is not None:
                self.record_use(path, target_info, access_id, self.span(subscript), "array_index")
        direct = expression
        if direct.type == "return_statement":
            direct = direct.named_children[0] if len(direct.named_children) == 1 else None
        while direct is not None and direct.type == "parenthesized_expression":
            direct = direct.named_children[0] if len(direct.named_children) == 1 else None
        if direct is None or direct.type != "identifier" or self.use_count >= MAX_USES or (
                self.stats["verified_scalar_read_paths"] >= 8):
            return
        binding = self.bindings.get(self.span(direct))
        if binding and binding["role"] == "read":
            target_info = self.info(direct)
            if target_info is not None:
                self.record_use(path, target_info, target_info["node_id"],
                                self.span(direct), "scalar_read")

    def kill_binding(self, path, identifier, reason):
        """A verified direct local write affects one binding, without inventing its value."""
        info = self.info(identifier)
        if info is None:
            return self.reset_unknown_effect(path, reason)
        old = self.state(path, info)
        version = old["version"] + 1
        path["env"][info["binding"]] = {
            "version": version, "value_key": (info["binding"], "unknown", version),
            "constant": None, "initialized": True, "last_update": self.span(identifier),
            "definition_span": self.span(identifier)}
        path["trace"].append(self.event("write", info, path["env"][info["binding"]],
                                        span=self.span(identifier), unknown_value=True))
        self.stats["scoped_unknown_local_write"] += 1
        return path

    def reset_unknown_effect(self, path, reason):
        """Unbounded effects invalidate all active facts and leave an incomplete path."""
        self.stats[reason] += 1
        path["incomplete"] = True
        path["incomplete_history"] = True
        path["permanent_incomplete"] = bool(path.get("permanent_incomplete") or
                                           reason.startswith("path_limit") or reason == "unsupported_control_flow")
        path["env"] = {key: {"version": value["version"] + 1,
                             "value_key": (key, "unknown", value["version"] + 1),
                             "constant": None, "initialized": value["initialized"],
                             "last_update": None, "definition_span": None}
                       for key, value in path["env"].items()}
        path["trace"] = []
        path["active"] = []
        return path

    def execute(self, path, node):
        kind = node.type
        if kind == "compound_statement":
            paths = [path]
            for child in node.named_children:
                paths = [result for current in paths for result in self.execute(current, child)]
                if len(paths) > MAX_PATHS:
                    self.stats["path_limit"] += 1
                    self.structural_truncation = True
                    return [self.reset_unknown_effect(path, "path_limit_incomplete")]
            return paths
        if kind == "if_statement":
            check = self.compare(path, node)
            if check is not None and path["active"]:
                condition = node.child_by_field_name("condition")
                while condition is not None and condition.type == "parenthesized_expression":
                    condition = condition.named_children[0] if len(condition.named_children) == 1 else None
                for side in ("left", "right"):
                    operand = condition.child_by_field_name(side)
                    if operand is not None and operand.type == "identifier":
                        self.use(path, operand)
            elif check is None:
                # Retain both arms while invalidating prior facts. Any later exact check
                # can establish a new relation, but the structural history stays flagged.
                self.stats["unsupported_branch_incomplete"] += 1
                self.reset_unknown_effect(path, "unknown_if_effect")
            result = []
            consequence = node.child_by_field_name("consequence")
            alternative = node.child_by_field_name("alternative")
            if alternative is not None and alternative.type == "else_clause":
                alternative = alternative.named_children[0] if len(alternative.named_children) == 1 else None
            for branch, body in ((1, consequence), (2, alternative)):
                if check is not None and not self.branch_possible(path, check, branch):
                    self.stats["infeasible_branch_skipped"] += 1
                    continue
                branch_path = {"env": dict(path["env"]), "trace": list(path["trace"]),
                               "active": list(path["active"]),
                               "incomplete": bool(path["permanent_incomplete"] or check is None),
                               "incomplete_history": path["incomplete_history"],
                               "permanent_incomplete": path["permanent_incomplete"]}
                if check is not None:
                    branch_path["active"].append(dict(check, branch=branch))
                    for side, operand in enumerate(check["operands"], 1):
                        if "info" not in operand:
                            continue
                        branch_path["trace"].append(self.event(
                            "check", operand["info"], branch_path["env"][operand["info"]["binding"]],
                            op=OPS[check["operator"]], branch=branch, side=side,
                            span=check["span"], literal=check["operands"][2-side].get("literal"),
                            check_span_key=check["span"]))
                result.extend(self.execute(branch_path, body) if body is not None else [branch_path])
            return result
        if kind == "declaration":
            for declarator in [child for child in node.named_children if child.type == "init_declarator"]:
                value = declarator.child_by_field_name("value")
                if value is not None and not any(item.type in {"assignment_expression", "update_expression"}
                                                 for item in walk(value)):
                    self.use(path, value)
                target = declarator.child_by_field_name("declarator")
                if target is not None and target.type == "identifier":
                    self.assign(path, target, value, declarator)
                else:
                    self.reset_unknown_effect(path, "unsupported_declaration")
            return [path]
        if kind == "expression_statement":
            expression = node.named_children[0] if len(node.named_children) == 1 else None
            if expression is not None and expression.type == "assignment_expression":
                left, right = (expression.child_by_field_name(name) for name in ("left", "right"))
                op = expression.child_by_field_name("operator")
                if op is not None and op.text == b"=":
                    if right is not None and not any(item.type in {"assignment_expression", "update_expression"}
                                                      for item in walk(right)):
                        self.use(path, right)
                    if (left is not None and left.type == "subscript_expression" and
                            right is not None and not any(item.type in {
                                "call_expression", "assignment_expression", "update_expression"}
                                for item in walk(right))):
                        self.use(path, left)
                    self.assign(path, left, right, expression)
                    return [path]
                if left is not None and left.type == "identifier" and right is not None and (
                        right.type in {"identifier", "number_literal"}):
                    self.kill_binding(path, left, "unsupported_compound_write")
                    return [path]
            if expression is not None and expression.type == "update_expression":
                identifiers = [item for item in walk(expression) if item.type == "identifier"]
                if len(identifiers) == 1:
                    self.kill_binding(path, identifiers[0], "unsupported_update_expression")
                    return [path]
            if expression is not None and not any(item.type in {"assignment_expression", "update_expression"}
                                                  for item in walk(expression)):
                self.use(path, expression)
                if expression.type == "subscript_expression":
                    return [path]
        if kind == "return_statement":
            if self.match("RETURN", node) is not None:
                self.use(path, node)
            return []
        if kind in {"for_statement", "while_statement", "do_statement", "switch_statement",
                    "goto_statement", "break_statement", "continue_statement"}:
            return [self.reset_unknown_effect(path, "unsupported_control_flow")]
        return [self.reset_unknown_effect(path, "unsupported_statement")]

    def build(self):
        if self.binding_error or self.root.has_error or self.function is None:
            self.stats[self.binding_error or "syntax_error"] += 1
            return {"schema": PROGRAM_SCHEMA, "slices": [], "queries": []}, self.stats
        body = self.function.child_by_field_name("body")
        if body is None:
            self.stats["missing_function_body"] += 1
            return {"schema": PROGRAM_SCHEMA, "slices": [], "queries": []}, self.stats
        self.execute({"env": {}, "trace": [], "active": [], "incomplete": False,
                      "incomplete_history": False, "permanent_incomplete": False}, body)
        slices, queries = [], []
        for (use_span, operation), paths in self.records.items():
            slices.append({"use_span": list(use_span), "operation": operation,
                           "paths": [item["slice"] for item in paths]})
            keys = set().union(*(item["candidates"] for item in paths))
            for key in sorted(keys):
                candidates = [item["candidates"].get(key) for item in paths]
                present = [item for item in candidates if item is not None]
                first = present[0]
                check, side = first["check"], first["side"]
                labels = [item["label"] if item is not None else None for item in candidates]
                # A verified reachable counterexample suffices; a positive requires every path.
                label = (None if self.structural_truncation else
                         0 if 0 in labels else 1 if labels and all(value == 1 for value in labels)
                         else None)
                row = {"schema": PROGRAM_SCHEMA, "sample_key": self.record["sample_key"],
                       "split": self.record["split"], "source_sha256": source_hash(self.source),
                       "check_span": list(check["span"]), "use_span": list(use_span),
                       "target_operand_span": list(paths[0]["target_operand_span"]),
                       "target_check_span": list(first["target"]["info"]["span"]),
                       "reference_span": list(first["reference"].get("span") or
                                              first["reference"]["info"]["span"]),
                       "check_node_id": check["node_id"], "use_node_id": paths[0]["use_node_id"],
                       "branch": key[2], "comparison": check["operator"], "target_side": side + 1,
                       "reference_kind": "literal" if "literal" in first["reference"] else "binding",
                       "target_binding": first["target"]["info"]["binding"],
                       "reference_binding": (first["reference"]["info"]["binding"]
                                             if "info" in first["reference"] else None),
                       "reference_version": first["reference"].get("version"),
                       "reference_definition_span": (list(first["reference"].get("definition_span"))
                                                     if first["reference"].get("definition_span") else None),
                       "operation": operation, "label": label,
                       "source": ("verified_counterexample" if label == 0 else
                                  "all_paths_support" if label == 1 else "unknown_incomplete_or_unproved"),
                       "paths": []}
                if self.offsets is not None:
                    positions = {"check_token": self.token(check["node_id"]),
                                 "use_token": self.token(paths[0]["use_node_id"]),
                                 "target_token": self.token(first["target"]["info"]["node_id"]),
                                 "reference_token": self.token_span(row["reference_span"])}
                    row.update(positions)
                    for item in candidates:
                        if item is None:
                            row["paths"].append({"missing": True, "incomplete": True})
                            continue
                        target_def = item["target"].get("definition_span")
                        reference_def = item["reference"].get("definition_span")
                        path_row = {"missing": False, "incomplete": bool(
                            item["source"] == "incomplete_path"),
                            "target_version": item["target"]["version"],
                            "reference_version": item["reference"].get("version"),
                            "target_definition_span": list(target_def) if target_def else None,
                            "current_definition_span": (list(item["current"]["definition_span"])
                                                        if item["current"]["definition_span"] else None),
                            "current_version": item["current"]["version"],
                            "reference_definition_span": list(reference_def) if reference_def else None,
                            "update_spans": [list(span) for span in item["updates"]]}
                        path_row["target_definition_token"] = (self.token_span(target_def)
                                                                if target_def else None)
                        path_row["current_definition_token"] = (
                            self.token_span(item["current"]["definition_span"])
                            if item["current"]["definition_span"] else None)
                        path_row["reference_definition_token"] = (self.token_span(reference_def)
                                                                   if reference_def else positions["reference_token"])
                        path_row["update_tokens"] = [self.token_span(span) for span in item["updates"]]
                        row["paths"].append(path_row)
                    visible = (all(value is not None for value in positions.values()) and
                               positions["check_token"] <= positions["use_token"] and
                               all(p["missing"] or
                                   (p["target_definition_token"] is not None and
                                    p["current_definition_token"] is not None and
                                    p["reference_definition_token"] is not None and
                                    all(t is not None and positions["check_token"] <= t <= positions["use_token"]
                                        for t in p["update_tokens"])) for p in row["paths"]))
                    if not visible:
                        label, row["label"], row["source"] = None, None, "unknown_truncated_or_position"
                if len(queries) >= MAX_QUERIES:
                    self.stats["query_limit"] += 1
                    continue
                queries.append(row)
                self.stats["query_positive" if label == 1 else "query_negative" if label == 0
                           else "query_unknown"] += 1
                self.stats[f"query_{operation}_{'positive' if label == 1 else 'negative' if label == 0 else 'unknown'}"] += 1
                self.stats[f"branch_{key[2]}_{'positive' if label == 1 else 'negative' if label == 0 else 'unknown'}"] += 1
        self.stats["functions_with_slices"] += bool(slices)
        self.stats["functions_with_queries"] += any(row["label"] is not None for row in queries)
        for name, value in (("positive", 1), ("negative", 0), ("unknown", None)):
            self.stats[f"functions_with_{name}_queries"] += any(
                row["label"] == value for row in queries)
        return {"schema": PROGRAM_SCHEMA, "slices": slices, "queries": queries}, self.stats


def build_program(record, graph, tokenizer=None, *, source_max_length=2048, prefix_tokens=0):
    """Return source/graph-verified structural slices and local diagnostic queries."""
    if record["split"] not in {"train", "valid", "test"}:
        raise ValueError("program representation needs the original train/valid/test cohort")
    if tokenizer is not None and not getattr(tokenizer, "is_fast", False):
        raise ValueError("program query positions require a fast tokenizer")
    return _Builder(record, graph, tokenizer, source_max_length, prefix_tokens).build()


def prepare_program(dataset, graphs_path, reference_run_dir, output_dir, tokenizer):
    """Reuse the saved graph and source; write structure for all splits, query labels only train/valid."""
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"program output already exists: {root}")
    reference = Path(reference_run_dir).resolve()
    config = json.loads((reference / "config.json").read_text())
    if config["source_max_length"] != 2048 or config["seed"] != 42:
        raise ValueError("program representation must use original C's source budget and seed")
    rows = read_records(dataset, config["source_dataset"])
    if (cohort_hash(rows) != config["cohort_sha256"] or
            str(Path(dataset).resolve()) != config["dataset"] or
            file_sha256(graphs_path) != config["graph_file_sha256"]):
        raise ValueError("program dataset or graph cache differs from original C")
    graphs = load_graphs(graphs_path, rows)
    from .model import InputBuilder
    builder = InputBuilder(tokenizer, source_max_length=2048, context_max_length=384)
    root.mkdir(parents=True, exist_ok=False)
    totals = {split: Counter() for split in ("train", "valid")}
    examples = {split: {"positive": [], "negative": [], "unknown": []}
                for split in ("train", "valid")}
    with (root / "program.jsonl").open("x", encoding="utf-8") as output, \
         (root / "train.queries.jsonl").open("x", encoding="utf-8") as train_out, \
         (root / "valid.queries.jsonl").open("x", encoding="utf-8") as valid_out:
        for index, record in enumerate(rows, 1):
            split = record["split"]
            program, counts = build_program(record, graphs[record["sample_key"]],
                                            tokenizer if split in {"train", "valid"} else None,
                                            prefix_tokens=len(builder.source_prefix))
            output.write(json.dumps({"sample_key": record["sample_key"], "split": split,
                                     "source_sha256": source_hash(record["raw_source"]),
                                     "program_schema": PROGRAM_SCHEMA, "slices": program["slices"]}, ensure_ascii=False) + "\n")
            if split in totals:
                totals[split].update(counts)
                stream = train_out if split == "train" else valid_out
                for query in program["queries"]:
                    stream.write(json.dumps(query, ensure_ascii=False) + "\n")
                    name = "unknown" if query["label"] is None else "positive" if query["label"] else "negative"
                    if (len(examples[split][name]) < 4 and not any(
                            item["sample_key"] == record["sample_key"]
                            for item in examples[split][name])):
                        examples[split][name].append({**query,
                            "check_text": record["raw_source"][slice(*query["check_span"])],
                            "use_text": record["raw_source"][slice(*query["use_span"])],
                            "reference_text": record["raw_source"][slice(*query["reference_span"])]})
            if index % 200 == 0 or index == len(rows):
                print(f"program_prepare={index}/{len(rows)}", flush=True)
    readiness = {}
    for split, counts in totals.items():
        # A few opposite-direction pairs from the same sources cannot support
        # independent train/valid diagnostics. Keep the low-coverage run auditable
        # and reject expensive formal pretraining until a broader sample exists.
        minimum = max(2, (counts["functions"] + 19) // 20)
        readiness[split] = {"effective_functions": counts["functions_with_queries"],
                            "coverage": counts["functions_with_queries"] / counts["functions"],
                            "required_coverage": 0.05,
                            "array_index_positive_queries": counts["query_array_index_positive"],
                            "array_index_negative_queries": counts["query_array_index_negative"],
                            "minimum_independent_functions": minimum,
                            "positive_functions": counts["functions_with_positive_queries"],
                            "negative_functions": counts["functions_with_negative_queries"],
                            "eligible": (counts["functions_with_queries"] >= minimum and
                                         counts["functions_with_positive_queries"] >= 2 and
                                         counts["functions_with_negative_queries"] >= 2)}
    audit = {"program_schema": PROGRAM_SCHEMA, "readiness": readiness, "reference_run_dir": str(reference), "dataset": str(Path(dataset).resolve()),
             "graphs": str(Path(graphs_path).resolve()), "source_max_length": 2048,
             "cohort_sha256": cohort_hash(rows), "program_sha256": file_sha256(root / "program.jsonl"),
             "train_queries_sha256": file_sha256(root / "train.queries.jsonl"),
             "valid_queries_sha256": file_sha256(root / "valid.queries.jsonl"),
             "splits": {split: dict(counts) for split, counts in totals.items()},
             "examples": examples, "test_queries_generated": False}
    atomic_json(root / "audit.json", audit)
    return audit


def load_programs(program_dir, rows, reference_run_dir):
    """Read prepared facts with original-cohort and content checks; query labels stay separate."""
    from .cfg_data import read_jsonl

    root = Path(program_dir)
    audit = json.loads((root / "audit.json").read_text())
    if (audit.get("program_schema") != PROGRAM_SCHEMA or
            audit["reference_run_dir"] != str(Path(reference_run_dir).resolve()) or
            audit["cohort_sha256"] != cohort_hash(rows) or
            audit["program_sha256"] != file_sha256(root / "program.jsonl")):
        raise ValueError("program facts differ from the original C cohort")
    expected = {row["sample_key"]: row for row in rows}
    result = {}
    for item in read_jsonl(root / "program.jsonl"):
        key = item["sample_key"]
        if (item.get("program_schema") != PROGRAM_SCHEMA or key not in expected or
                key in result or item["split"] != expected[key]["split"] or
                item["source_sha256"] != source_hash(expected[key]["raw_source"])):
            raise ValueError("program fact source/split identity mismatch")
        result[key] = {"schema": PROGRAM_SCHEMA, "slices": item["slices"]}
    if set(result) != set(expected):
        raise ValueError("program facts do not cover the entire original cohort")
    return result, audit


def program_cost(programs, rows, steps):
    """Deterministic sparse message counts, excluding unchanged source Qwen/CFG work."""
    report = {}
    for split in ("train", "valid", "test"):
        selected = [programs[row["sample_key"]] for row in rows if row["split"] == split]
        paths = [path for program in selected for use in program["slices"] for path in use["paths"]]
        report[split] = {"functions": len(selected), "uses": sum(len(p["slices"]) for p in selected),
                         "paths": len(paths), "relation_instances": sum(len(p["relations"]) for p in paths),
                         "incomplete_paths": sum(bool(p["incomplete"]) for p in paths),
                         "event_states": sum(len(p["events"]) for p in paths),
                         "plain_message_edges_per_step": sum(len(p["plain_edges"]) for p in paths),
                         "versioned_message_edges_per_step": sum(len(p["edges"]) for p in paths),
                         "different_topology_paths": sum(p["plain_edges"] != p["edges"] for p in paths),
                         "extra_program_steps": steps}
    return report
