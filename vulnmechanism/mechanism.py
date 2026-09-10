from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .cpg import FunctionGraph, GraphNode, extract_function_cpg
from .syntax import parse_function


_MEMORY_NAMES = {
    'memcpy', 'memmove', 'mempcpy', 'memset', 'memcmp', 'bcopy', 'bzero',
    'read', 'recv', 'recvfrom', 'fread', 'write', 'send', 'sendto', 'fwrite',
    'strcpy', 'strcat', 'strncpy', 'strncat', 'strlcpy', 'strlcat',
    'sprintf', 'vsprintf', 'snprintf', 'vsnprintf', 'free',
}
_ALLOC_NAMES = {'malloc', 'calloc', 'realloc', 'kmalloc', 'kzalloc', 'vmalloc', 'new'}
_COMPARE = re.compile(r'<=|>=|==|!=|<|>')
_ARITHMETIC = re.compile(r'(?<![+\-*/%&|^<>])[+\-*/%]|<<|>>')


@dataclass(frozen=True)
class GraphRelation:
    kind: str
    source: str
    target: str

    def as_text(self) -> str:
        return f'{self.kind}|{self.source}|{self.target}'


def _node_categories(node: GraphNode) -> set[str]:
    code = node.code
    lower = code.lower()
    categories: set[str] = set()
    if node.label == 'CONTROL_STRUCTURE' or lower.startswith(('if ', 'if(', 'while ', 'while(', 'for ', 'for(')):
        categories.add('control')
    if _COMPARE.search(code):
        categories.add('comparison')
    if any(re.search(rf'\b{re.escape(name)}\b', code) for name in _ALLOC_NAMES):
        categories.add('allocation')
    if any(re.search(rf'\b{re.escape(name)}\b', code) for name in _MEMORY_NAMES):
        categories.add('memory')
    if any(token in node.label for token in ('indirectIndexAccess', 'indirection', 'fieldAccess')) or '[' in code:
        categories.add('pointer_index')
    if _ARITHMETIC.search(code) or any(
        token in node.label
        for token in ('addition', 'subtraction', 'multiplication', 'division', 'shiftLeft', 'shiftRight')
    ):
        categories.add('arithmetic')
    return categories


def _node_text(node: GraphNode) -> str:
    code = re.sub(r'\s+', ' ', node.code).strip()
    return f'{node.label}:{code}'


def graph_relations(graph: FunctionGraph) -> tuple[GraphRelation, ...]:
    """Keep CPG relations touching security-relevant operations plus their one-hop context."""
    relevant = {
        node_id
        for node_id, node in graph.nodes.items()
        if _node_categories(node)
    }
    for edge in graph.edges:
        if edge.source in relevant or edge.target in relevant:
            relevant.update((edge.source, edge.target))

    relations: list[GraphRelation] = []
    seen: set[str] = set()
    for edge in graph.edges:
        if edge.source not in relevant and edge.target not in relevant:
            continue
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source is None or target is None:
            continue
        relation = GraphRelation(edge.kind, _node_text(source), _node_text(target))
        text = relation.as_text()
        if text not in seen:
            seen.add(text)
            relations.append(relation)
    return tuple(relations)


def render_graph(graph: FunctionGraph, max_relations: int = 160) -> str:
    relations = graph_relations(graph)
    return '\n'.join(item.as_text() for item in relations[:max_relations]) or 'NO_SECURITY_RELEVANT_GRAPH_RELATIONS'


def _required_string(record: dict[str, object], names: tuple[str, ...], label: str) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f'missing {label}; expected one of {", ".join(names)}')


def _normalize_label(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, float) and value in {0.0, 1.0}:
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'0', 'benign'}:
            return 0
        if normalized in {'1', 'vulnerable'}:
            return 1
    raise ValueError(f'label must be binary 0/1, got {value!r}')


def _sample_fields(record: dict[str, object]) -> tuple[str, str, int, str, str | None, str | None]:
    key_value = record.get('sample_key') or record.get('id') or record.get('idx')
    if key_value is None or str(key_value).strip() == '':
        raise ValueError('sample record requires sample_key, id, or idx')
    key = str(key_value)
    source = _required_string(record, ('function', 'func', 'source', 'code', 'func_before'), 'function source')

    if 'label' in record:
        label_value = record['label']
    elif 'target' in record:
        label_value = record['target']
    else:
        raise ValueError(f'{key}: sample record requires label or target')
    label = _normalize_label(label_value)

    language = str(record.get('language') or 'c').lower()
    language = 'cpp' if language in {'c++', 'cpp'} else language
    if language not in {'c', 'cpp'}:
        raise ValueError(f'{key}: language must be c or cpp')

    function_name = record.get('function_name')
    split_value = record.get('split')
    split = str(split_value).lower() if split_value is not None else None
    if split == 'validation':
        split = 'valid'
    if split is not None and split not in {'train', 'valid', 'test'}:
        raise ValueError(f'{key}: split must be train, valid, validation, or test')
    return key, source, label, language, str(function_name) if function_name else None, split


def build_function_dataset(
    samples_path: str | Path,
    output_path: str | Path,
    *,
    joern_dir: str | Path = '/home/phy/joern',
    java_home: str | Path = '/home/phy/jdk21',
    timeout: int = 300,
) -> list[dict[str, object]]:
    """Build one CPG-augmented record per function while preserving the dataset's original binary label."""
    records: list[dict[str, object]] = []
    with Path(samples_path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'invalid JSON at {samples_path}:{line_number}: {error}') from error
            if not isinstance(raw, dict):
                raise ValueError(f'{samples_path}:{line_number}: each JSONL row must be an object')

            key, source, label, language, function_name, split = _sample_fields(raw)
            parsed = parse_function(source, language, function_name)
            graph = extract_function_cpg(
                source,
                parsed.name,
                language=language,
                joern_dir=joern_dir,
                java_home=java_home,
                timeout=timeout,
            )
            record: dict[str, object] = {
                'sample_key': key,
                'label': label,
                'language': language,
                'function_name': parsed.name,
                'raw_source': source,
                'graph': render_graph(graph),
            }
            if split is not None:
                record['split'] = split
            records.append(record)
            print(
                f'function_sample_done={key} label={label} relations={len(graph_relations(graph))}',
                flush=True,
            )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    return records
