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
OPS = {"<": 1, "<=": 2, ">": 3, ">=": 4, "==": 5, "!=": 6}
INVERSE = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
JOERN_OPS = {"<": "lessThan", "<=": "lessEqualsThan", ">": "greaterThan",
             ">=": "greaterEqualsThan", "==": "equals", "!=": "notEquals"}
KINDS = {"check": 1, "read": 2, "write": 3, "copy": 4,
         "use_array": 5, "use_scalar": 6}
SIGNED_TYPES = {"int", "short", "long", "long long", "ssize_t", "ptrdiff_t",
                "intptr_t", "int8_t", "int16_t", "int32_t", "int64_t"}
UNSIGNED_TYPES = {"unsigned", "unsigned int", "unsigned short", "unsigned long",
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
    return 127 if name == "int8_t" else 255 if name == "uint8_t" else 32767


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
            self.source, record.get("resolved_language") or record.get("language"))
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
            if node["properties"].get("kind") not in relevant or len(node["code"]) > 512:
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
        self.records = defaultdict(list)
        self.use_count = 0
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
                "parameter": binding["parameter"], "span": span, "node_id": node_id}

    def state(self, path, info):
        key = info["binding"]
        if key not in path["env"]:
            path["env"][key] = {"version": 0, "value_key": (key, 0),
                                "constant": None, "initialized": bool(info["parameter"]),
                                "last_update": None}
        return path["env"][key]

    def event(self, kind, info, state, *, op=0, branch=0, side=0, literal=None,
              span=None, **extra):
        return {"kind": kind, "binding": info["binding"], "version": state["version"],
                "type": info["type"], "span": list(span or info["span"]), "op": op,
                "branch": branch, "side": side, "literal": literal, **extra}

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
        control_id = self.match("CONTROL_STRUCTURE", node)
        comparison_id = self.match("<operator>." + JOERN_OPS[operator], condition)
        if control_id is None or comparison_id is None or self.cfg(control_id) is None:
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
                                 "constant": state["constant"], "version": state["version"]})
            else:
                literal = _literal(child)
                if literal is None:
                    self.stats["unsupported_check_operand"] += 1
                    return None
                operands.append({"literal": literal})
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
                "node_id": comparison_id, "cfg": self.cfg(control_id)}

    def assign(self, path, target, value, operation):
        target_info = self.info(target)
        if target_info is None:
            path["unknown"] = True
            return
        if operation.type == "assignment_expression":
            if self.match("<operator>.assignment", operation) is None:
                path["unknown"] = True
                return
        elif self.match("IDENTIFIER", target) is None:
            path["unknown"] = True
            return
        if value is not None and value.type == "identifier":
            source_info = self.info(value)
            if source_info is None or source_info["type"] != target_info["type"]:
                self.stats["implicit_conversion_unknown"] += 1
                path["unknown"] = True
                return
            source_state = self.state(path, source_info)
            if not source_state["initialized"]:
                self.stats["uninitialized_copy"] += 1
                path["unknown"] = True
                return
            path["trace"].append(self.event("read", source_info, source_state))
            new_value_key, constant = source_state["value_key"], source_state["constant"]
            kind = "copy"
        else:
            constant = _literal(value)
            if constant is None or constant > _type_literal_limit(target_info["type"]):
                self.stats["unsupported_update"] += 1
                path["unknown"] = True
                return
            new_value_key = (target_info["binding"], self.span(operation))
            kind = "write"
        old = self.state(path, target_info)
        new = {"version": old["version"] + 1, "value_key": new_value_key,
               "constant": constant, "initialized": True,
               "last_update": self.span(operation)}
        path["trace"].append(self.event(
            kind, target_info, new, span=self.span(operation), literal=constant,
            old_version=old["version"], **({"source_binding": source_info["binding"],
                                            "source_version": source_state["version"]}
                                           if kind == "copy" else {})))
        path["env"][target_info["binding"]] = new
        self.stats["copy_updates" if kind == "copy" else "constant_updates"] += 1

    def candidate(self, path, check, target_info, target_state, desired_branch):
        operands = check["operands"]
        sides = [index for index, item in enumerate(operands)
                 if "info" in item and (item["info"]["binding"] == target_info["binding"] or
                                        item["value_key"] == target_state["value_key"])]
        if not sides:
            return None
        # If both operands share the same binding, there is no distinct local bound.
        if len(sides) != 1:
            return None
        index = sides[0]
        other = operands[1 - index]
        other_constant = other["literal"] if "literal" in other else None
        other_stable = True
        if "info" in other:
            current = path["env"].get(other["info"]["binding"])
            other_stable = bool(current and current["value_key"] == other["value_key"])
            other_constant = current["constant"] if current and current["initialized"] else None
        target_stable = target_state["value_key"] == operands[index]["value_key"]
        if target_stable and other_stable:
            return (int(desired_branch == check["branch"]),
                    "same_value_versions" if desired_branch == check["branch"]
                    else "opposite_branch_counterexample")
        if target_state["constant"] is not None and other_constant is not None:
            values = [None, None]
            values[index], values[1 - index] = target_state["constant"], other_constant
            operator = check["operator"] if desired_branch == 1 else INVERSE[check["operator"]]
            holds = _holds(values[0], operator, values[1])
            return int(holds), "constant_proof_or_counterexample"
        return None, "unproved_after_update"

    def record_use(self, path, target_info, use_id, use_span, operation):
        state = self.state(path, target_info)
        use_cfg = self.cfg(use_id)
        if not state["initialized"] or use_cfg is None:
            self.stats["uninitialized_or_missing_use_cfg"] += 1
            return
        related_checks = [check for check in path["active"]
                          if any("info" in item and
                                 (item["info"]["binding"] == target_info["binding"] or
                                  item["value_key"] == state["value_key"])
                                 for item in check["operands"])]
        relevant = {target_info["binding"]}
        for check in related_checks:
            relevant.update(item["info"]["binding"] for item in check["operands"]
                            if "info" in item)
        for event in reversed(path["trace"]):
            if event["kind"] == "copy" and event["binding"] in relevant:
                relevant.add(event["source_binding"])
        check_spans = {check["span"] for check in related_checks}
        events = [dict(event) for event in path["trace"]
                  if event["binding"] in relevant and
                  (event["kind"] != "check" or event["check_span_key"] in check_spans)]
        events.append(self.event("use_array" if operation == "array_index" else "use_scalar",
                                 target_info, state, span=use_span, literal=state["constant"]))
        if len(events) > MAX_EVENTS:
            self.stats["too_many_events_for_use"] += 1
            return
        edges, plain_edges = _edges(events)
        candidates = {}
        for check in related_checks:
            if not _reachable(check["cfg"], use_cfg, self.successors):
                self.stats["check_not_cfg_reachable"] += 1
                continue
            latest_updates = [path["env"][item["info"]["binding"]]["last_update"]
                              for item in check["operands"] if "info" in item]
            updates = [span for span in latest_updates if span and span[0] > check["span"][0]]
            for desired_branch in ((check["branch"], 1) if check["branch"] == 2 else (1,)):
                if desired_branch == 2 and check.get("branch_anchor") is None:
                    continue
                status = self.candidate(path, check, target_info, state, desired_branch)
                if status is None:
                    continue
                verdict, reason = status
                key = (check["span"], desired_branch)
                anchor = (check["branch_anchor"] if desired_branch == 2 and not updates
                          else check["span"])
                update_span = max(updates, key=lambda span: span[0]) if updates else anchor
                candidates[key] = {"label": verdict, "source": reason,
                                   "check": dict(check, branch=desired_branch),
                                   "update_span": update_span,
                                   "target_binding": target_info["binding"]}
        self.records[(use_span, operation)].append({"slice": {"events": events, "edges": edges,
                                                              "plain_edges": plain_edges},
                                                    "candidates": candidates, "use_node_id": use_id})
        self.use_count += 1
        self.stats[f"verified_{operation}_paths"] += 1

    def use(self, path, expression):
        if path["unknown"] or self.use_count >= MAX_USES:
            self.stats["unknown_or_use_limit"] += 1
            return
        subscripts = [item for item in walk(expression) if item.type == "subscript_expression"]
        for subscript in subscripts:
            argument, index = (subscript.child_by_field_name(name) for name in ("argument", "index"))
            if argument is None or argument.type != "identifier" or index is None or index.type != "identifier":
                self.stats["unsupported_array_operand"] += 1
                continue
            access_id = self.match("<operator>.indirectIndexAccess", subscript)
            if access_id is None or self.match("IDENTIFIER", argument) is None:
                continue
            target_info = self.info(index)
            if target_info is not None:
                self.record_use(path, target_info, access_id, self.span(subscript), "array_index")
        # A scalar query uses the identifier itself, not a larger arithmetic/call
        # expression whose conversion, overflow or evaluation order is uncertain.
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

    def reset_unknown_effect(self, path, reason):
        """Discard prior value/condition facts; later independent checks can be analyzed."""
        self.stats[reason] += 1
        path["env"] = {key: {"version": value["version"] + 1,
                             "value_key": (key, "unknown", value["version"] + 1),
                             "constant": None, "initialized": value["initialized"],
                             "last_update": None}
                       for key, value in path["env"].items()}
        path["trace"] = []
        path["active"] = []
        path["unknown"] = False
        return path

    def execute(self, path, node):
        if path["unknown"]:
            self.reset_unknown_effect(path, "unknown_effect_reset")
        kind = node.type
        if kind == "compound_statement":
            paths = [path]
            for child in node.named_children:
                paths = [result for current in paths for result in self.execute(current, child)]
                if len(paths) > MAX_PATHS:
                    return [self.reset_unknown_effect(path, "path_limit")]
            return paths
        if kind == "if_statement":
            check = self.compare(path, node)
            if check is None:
                # The branch condition is unknown, but a later independent check may be safe.
                return [self.reset_unknown_effect(path, "unknown_if_reset")]
            if path["active"]:
                # A later simple check reads its direct scalar operands before establishing
                # its own branch constraint. Previous path facts can apply to that read.
                condition = node.child_by_field_name("condition")
                while condition is not None and condition.type == "parenthesized_expression":
                    condition = condition.named_children[0] if len(condition.named_children) == 1 else None
                for side in ("left", "right"):
                    operand = condition.child_by_field_name(side)
                    if operand is not None and operand.type == "identifier":
                        self.use(path, operand)
            result = []
            consequence = node.child_by_field_name("consequence")
            alternative = node.child_by_field_name("alternative")
            if alternative is not None and alternative.type == "else_clause":
                alternative = alternative.named_children[0] if len(alternative.named_children) == 1 else None
            false_anchor = None
            if alternative is not None:
                start = self.byte_to_char.get(node.child_by_field_name("alternative").start_byte)
                if start is not None:
                    false_anchor = (start, start + 4)  # exact tree-sitter else keyword
            else:
                returned = (consequence if consequence is not None and
                            consequence.type == "return_statement" else
                            consequence.named_children[0] if consequence is not None and
                            consequence.type == "compound_statement" and
                            len(consequence.named_children) == 1 and
                            consequence.named_children[0].type == "return_statement" else None)
                if returned is not None and self.match("RETURN", returned) is not None:
                    false_anchor = self.span(returned)
            for branch, body in ((1, consequence), (2, alternative)):
                branch_path = {"env": dict(path["env"]), "trace": list(path["trace"]),
                               "active": list(path["active"]), "unknown": False}
                effective = check["operator"] if branch == 1 else INVERSE[check["operator"]]
                active = dict(check, branch=branch, effective_operator=effective,
                              branch_anchor=check["span"] if branch == 1 else false_anchor)
                branch_path["active"].append(active)
                for side, operand in enumerate(check["operands"], 1):
                    if "info" not in operand:
                        continue
                    branch_path["trace"].append(self.event(
                        "check", operand["info"], branch_path["env"][operand["info"]["binding"]],
                        op=OPS[check["operator"]], branch=branch, side=side,
                        span=check["span"],
                        literal=check["operands"][2 - side].get("literal"),
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
                value = declarator.child_by_field_name("value")
                if target is not None and target.type == "identifier":
                    self.assign(path, target, value, declarator)
                else:
                    self.reset_unknown_effect(path, "unsupported_declaration")
            return [path]
        if kind == "expression_statement":
            expression = node.named_children[0] if len(node.named_children) == 1 else None
            if expression is not None and expression.type == "assignment_expression":
                operator = expression.child_by_field_name("operator")
                if operator is not None and operator.text == b"=":
                    left = expression.child_by_field_name("left")
                    right = expression.child_by_field_name("right")
                    if right is not None and not any(item.type in {"assignment_expression", "update_expression"}
                                                      for item in walk(right)):
                        self.use(path, right)
                    if left is not None and left.type == "subscript_expression":
                        self.use(path, left)
                    self.assign(path, left, right, expression)
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
        return [self.reset_unknown_effect(path, "unsupported_statement")]

    def build(self):
        if self.binding_error or self.root.has_error or self.function is None:
            self.stats[self.binding_error or "syntax_error"] += 1
            return {"slices": [], "queries": []}, self.stats
        body = self.function.child_by_field_name("body")
        if body is None:
            self.stats["missing_function_body"] += 1
            return {"slices": [], "queries": []}, self.stats
        self.execute({"env": {}, "trace": [], "active": [], "unknown": False}, body)
        slices, queries = [], []
        for (use_span, operation), paths in self.records.items():
            slices.append({"use_span": list(use_span), "operation": operation,
                           "paths": [item["slice"] for item in paths]})
            keys = set().union(*(item["candidates"] for item in paths))
            for key in sorted(keys):
                candidates = [item["candidates"].get(key) for item in paths]
                present = [item for item in candidates if item is not None]
                first = present[0]
                labels = {item["label"] for item in present}
                updates = {item["update_span"] for item in present}
                label = (next(iter(labels)) if len(present) == len(paths) and len(labels) == 1
                         and None not in labels and len(updates) == 1 else None)
                check = first["check"]
                row = {"sample_key": self.record["sample_key"], "split": self.record["split"],
                       "source_sha256": source_hash(self.source), "check_span": list(check["span"]),
                       "use_span": list(use_span), "update_span": list(first["update_span"]),
                       "check_node_id": check["node_id"], "use_node_id": paths[0]["use_node_id"],
                       "branch": check["branch"], "comparison": check["operator"],
                       "effective_comparison": (check["operator"] if check["branch"] == 1
                                                else INVERSE[check["operator"]]),
                       "comparison_sides": [{"role": "left" if i == 0 else "right",
                                             "type": item["info"]["type"] if "info" in item else "literal",
                                             "span": list(item["info"]["span"]) if "info" in item else None,
                                             "binding": item["info"]["binding"] if "info" in item else None,
                                             "value_version": item.get("version"),
                                             "literal": item.get("literal")}
                                            for i, item in enumerate(check["operands"])],
                       "target_operand_span": list(use_span),
                       "target_binding": first["target_binding"],
                       "target_versions": sorted({item["slice"]["events"][-1]["version"]
                                                  for item in paths}),
                       "operation": operation,
                       "label": label,
                       "source": first["source"] if label is not None else "unknown_multi_path_or_update"}
                if label is not None and self.offsets is not None:
                    check_token = self.token(check["node_id"])
                    use_token = self.token(paths[0]["use_node_id"])
                    update_id = None
                    for kind in ("<operator>.assignment", "IDENTIFIER"):
                        matches = self.cache.get((kind, first["update_span"]), [])
                        if len(matches) == 1:
                            update_id = matches[0]
                            break
                    update_token = (self.token(update_id) if update_id else
                                    self.token_span(first["update_span"]))
                    if (check_token is None or use_token is None or update_token is None or
                            not check_token <= update_token <= use_token):
                        label = None
                        row["label"] = None
                        row["source"] = "unknown_truncated_or_position"
                    else:
                        row.update(check_token=check_token, update_token=update_token,
                                   use_token=use_token)
                if len(queries) >= MAX_QUERIES:
                    self.stats["query_limit"] += 1
                    continue
                queries.append(row)
                self.stats["query_positive" if label == 1 else "query_negative" if label == 0
                           else "query_unknown"] += 1
                self.stats[f"query_{operation}_{'positive' if label == 1 else 'negative' if label == 0 else 'unknown'}"] += 1
        self.stats["functions_with_slices"] += bool(slices)
        self.stats["functions_with_queries"] += any(row["label"] is not None for row in queries)
        return {"slices": slices, "queries": queries}, self.stats


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
                                     "slices": program["slices"]}, ensure_ascii=False) + "\n")
            if split in totals:
                totals[split].update(counts)
                stream = train_out if split == "train" else valid_out
                for query in program["queries"]:
                    stream.write(json.dumps(query, ensure_ascii=False) + "\n")
                    name = "unknown" if query["label"] is None else "positive" if query["label"] else "negative"
                    if len(examples[split][name]) < 4:
                        examples[split][name].append({**query,
                            "check_text": record["raw_source"][slice(*query["check_span"])],
                            "use_text": record["raw_source"][slice(*query["use_span"])],
                            "update_text": record["raw_source"][slice(*query["update_span"])]})
            if index % 200 == 0 or index == len(rows):
                print(f"program_prepare={index}/{len(rows)}", flush=True)
    audit = {"reference_run_dir": str(reference), "dataset": str(Path(dataset).resolve()),
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
    if (audit["reference_run_dir"] != str(Path(reference_run_dir).resolve()) or
            audit["cohort_sha256"] != cohort_hash(rows) or
            audit["program_sha256"] != file_sha256(root / "program.jsonl")):
        raise ValueError("program facts differ from the original C cohort")
    expected = {row["sample_key"]: row for row in rows}
    result = {}
    for item in read_jsonl(root / "program.jsonl"):
        key = item["sample_key"]
        if (key not in expected or key in result or item["split"] != expected[key]["split"] or
                item["source_sha256"] != source_hash(expected[key]["raw_source"])):
            raise ValueError("program fact source/split identity mismatch")
        result[key] = {"slices": item["slices"]}
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
                         "paths": len(paths), "event_states": sum(len(p["events"]) for p in paths),
                         "plain_message_edges_per_step": sum(len(p["plain_edges"]) for p in paths),
                         "versioned_message_edges_per_step": sum(len(p["edges"]) for p in paths),
                         "different_topology_paths": sum(p["plain_edges"] != p["edges"] for p in paths),
                         "extra_program_steps": steps}
    return report
