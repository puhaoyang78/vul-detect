"""Definition attributes and directed CFGs, following DeepDFA's feature design.

Reference: ISU-PAAL/DeepDFA, commit 14414f34293cd994001faa69f4553f6fc559b2b4,
DDFA/sastvd/scripts/abstract_dataflow_full.py and models/flow_gnn/ggnn.py.
This is an adaptation, not an exact reproduction of its BigVul preprocessing.
No labels, patch metadata, mechanism candidates, or execution traces are used.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict

GRAPH_SCHEMA_VERSION = 1
FEATURE_NAMES = ("api", "datatype", "literal", "operator")
DEFINITION_OPERATORS = {
    "assignment", "assignmentAnd", "assignmentArithmeticShiftRight",
    "assignmentDivision", "assignmentExponentiation", "assignmentLogicalShiftRight",
    "assignmentMinus", "assignmentModulo", "assignmentMultiplication",
    "assignmentOr", "assignmentPlus", "assignmentShiftLeft", "assignmentXor",
    "postDecrement", "postIncrement", "preDecrement", "preIncrement",
}


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_identity(record: dict) -> dict:
    return {name: record[name] for name in
            ("sample_key", "dataset", "split", "label", "raw_source")}


def records_fingerprint(records: list[dict]) -> str:
    # Sorting makes a membership digest independent of storage order.
    rows = sorted((record_identity(r) for r in records), key=lambda r: r["sample_key"])
    return source_hash(canonical_json(rows))


def _operator(name: str) -> str | None:
    match = re.fullmatch(r"<operators?>\.(.+)", name)
    return match.group(1) if match else None


def _clean_type(value: object) -> str | None:
    if not isinstance(value, str) or value.strip() in {"", "ANY", "<unknown>", "UNKNOWN"}:
        return None
    value = re.sub(r"^const\s+", "", value.strip())
    return re.sub(r"\s+", " ", re.sub(r"\s*\[.*?\]", "[]", value)).strip()


def graph_to_record(graph, source: str) -> dict:
    """Keep every target-function node/edge; derive attributes on definitions only.

    Native ARGUMENT_INDEX (ORDER as fallback) on direct AST children replaces
    the upstream ARGUMENT table. Unknown types remain unknown, not guessed.
    CFG edges are kept directed. Equal node text never merges node identities.
    """
    raw = {}
    for node_id, node in graph.nodes.items():
        if node.properties is None:
            raise ValueError("native Joern attributes are missing; re-export the graph")
        raw[node_id] = dict(node.properties)
    children = defaultdict(list)
    for edge in graph.edges:
        if edge.source not in raw or edge.target not in raw:
            raise ValueError("graph contains a dangling edge")
        if edge.kind == "AST":
            children[edge.source].append(edge.target)
    cfg_edges = sorted({(e.source, e.target) for e in graph.edges if e.kind == "CFG"})
    cfg_ids = sorted({n for pair in cfg_edges for n in pair})
    if not cfg_ids:
        raise ValueError("target function has no CFG")
    indices = {node_id: i for i, node_id in enumerate(cfg_ids)}

    def argument(node_id: str, index: int) -> str | None:
        direct = children.get(node_id, ())
        matches = [n for n in direct if raw[n].get("ARGUMENT_INDEX") == index]
        if not matches:
            matches = [n for n in direct if "ARGUMENT_INDEX" not in raw[n]
                       and raw[n].get("ORDER") == index]
        return matches[0] if len(matches) == 1 else None

    def assigned_type(node_id: str) -> str | None:
        lhs = argument(node_id, 1)
        visited = set()
        while lhs is not None and lhs not in visited:
            visited.add(lhs)
            datatype = _clean_type(raw[lhs].get("TYPE_FULL_NAME"))
            if datatype:
                return datatype
            op = _operator(str(raw[lhs].get("NAME", "")))
            if op not in {"indirectIndexAccess", "indexAccess", "indirectFieldAccess",
                          "indirection", "fieldAccess", "postIncrement", "postDecrement",
                          "preIncrement", "preDecrement", "addressOf", "cast", "addition"}:
                return None
            lhs = argument(lhs, 2 if op == "cast" else 1)
        return None

    cfg_nodes = []
    definitions, missing_types = 0, 0
    for node_id in cfg_ids:
        fields = {name: set() for name in FEATURE_NAMES}
        info = raw[node_id]
        is_definition = (info.get("kind") == "CALL" and
                         _operator(str(info.get("NAME", ""))) in DEFINITION_OPERATORS)
        if is_definition:
            definitions += 1
            datatype = assigned_type(node_id)
            if datatype:
                fields["datatype"].add(datatype)
            else:
                missing_types += 1
                fields["datatype"].add("<TYPE_UNKNOWN>")
            # Exclude the root assignment, matching upstream's AST descendants.
            visited, pending = {node_id}, list(children.get(node_id, ()))
            while pending:
                descendant = pending.pop()
                if descendant in visited:
                    continue
                visited.add(descendant)
                child = raw[descendant]
                if child.get("kind") == "METHOD":
                    continue
                if child.get("kind") == "LITERAL":
                    fields["literal"].add(graph.nodes[descendant].code)
                if child.get("kind") == "CALL":
                    name = str(child.get("NAME", ""))
                    op = _operator(name)
                    if op is not None:
                        if op != "indirection":
                            fields["operator"].add(op)
                    elif name:
                        fields["api"].add(name)
                pending.extend(children.get(descendant, ()))
        cfg_nodes.append({"id": node_id, "is_definition": is_definition,
                          "features": {k: sorted(v) for k, v in fields.items()}})

    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "source_sha256": source_hash(source),
        "nodes": [{"id": node_id, "label": node.label, "code": node.code,
                   "properties": raw[node_id]} for node_id, node in graph.nodes.items()],
        "edges": [{"kind": e.kind, "source": e.source, "target": e.target}
                  for e in graph.edges],
        "cfg_nodes": cfg_nodes,
        "cfg_edges": [[indices[u], indices[v]] for u, v in cfg_edges],
        "quality": graph.quality,
        "statistics": {"nodes": len(raw), "edges": len(graph.edges),
                       "cfg_nodes": len(cfg_nodes), "cfg_edges": len(cfg_edges),
                       "definitions": definitions, "definitions_unknown_type": missing_types},
    }


def validate_record_graph(record: dict) -> None:
    key = record.get("sample_key", "<unknown>")
    if "static_graph" not in record:
        raise ValueError(f"{key}: static_graph missing; use graph_experiment build first")
    graph = record["static_graph"]
    if graph is None:
        if record.get("graph_status") != "unavailable":
            raise ValueError(f"{key}: missing graph must be explicitly marked unavailable")
        return
    if not isinstance(graph, dict) or graph.get("schema_version") != GRAPH_SCHEMA_VERSION:
        raise ValueError(f"{key}: unsupported static graph schema")
    if record.get("graph_status") != "available":
        raise ValueError(f"{key}: invalid graph status")
    if graph.get("source_sha256") != source_hash(record["raw_source"]):
        raise ValueError(f"{key}: graph/source mismatch")
    nodes, edges = graph.get("cfg_nodes"), graph.get("cfg_edges")
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list) or not edges:
        raise ValueError(f"{key}: malformed or empty CFG")
    ids = []
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise ValueError(f"{key}: CFG node identity missing")
        ids.append(node["id"])
        fields = node.get("features")
        if not isinstance(fields, dict) or set(fields) != set(FEATURE_NAMES):
            raise ValueError(f"{key}: wrong abstract feature channels")
        if any(not isinstance(v, list) or not all(isinstance(x, str) for x in v)
               for v in fields.values()):
            raise ValueError(f"{key}: malformed abstract attributes")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{key}: duplicate CFG node identities")
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or any(
            type(i) is not int or not 0 <= i < len(nodes) for i in edge
        ):
            raise ValueError(f"{key}: CFG edge has invalid endpoints")


def feature_signature(values: list[str]) -> str:
    return canonical_json(sorted(set(values)))


def fit_graph_config(train_records: list[dict], *, embedding_dim: int = 32,
                     steps: int = 5, vocab_size: int = 2048) -> dict:
    if min(embedding_dim, steps) <= 0 or vocab_size < 2:
        raise ValueError("graph dimensions/steps must be positive; vocabulary must be >= 2")
    if not train_records or any(r["split"] != "train" for r in train_records):
        raise ValueError("graph vocabulary must be fitted on training records only")
    counts = {name: Counter() for name in FEATURE_NAMES}
    available = 0
    for record in train_records:
        validate_record_graph(record)
        graph = record["static_graph"]
        if graph is None:
            continue
        available += 1
        for node in graph["cfg_nodes"]:
            for name in FEATURE_NAMES:
                signature = feature_signature(node["features"][name])
                if signature != "[]":
                    counts[name][signature] += 1
    if not available:
        raise ValueError("no usable training graphs; refusing a silently empty graph experiment")
    vocab = {}
    for name, counter in counts.items():
        ordered = sorted(counter, key=lambda text: (-counter[text], text))[:vocab_size - 2]
        vocab[name] = {"[]": 0, "<UNK>": 1,
                       **{text: index + 2 for index, text in enumerate(ordered)}}
    return {"schema_version": GRAPH_SCHEMA_VERSION, "embedding_dim": embedding_dim,
            "steps": steps, "vocab_size": vocab_size, "vocabulary": vocab,
            "training_fingerprint": records_fingerprint(train_records),
            "training_graphs": available,
            "feature_scope": "definition_ast_descendants", "edge_type": "directed_CFG"}
