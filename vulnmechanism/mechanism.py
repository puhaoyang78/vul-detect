from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .cpg import FunctionGraph, GraphNode, extract_function_cpg
from .syntax import local_identifiers, parse_function


MECHANISM_COMPONENTS = (
    'guard',
    'bounds',
    'allocation',
    'memory_access',
    'size_arithmetic',
    'data_dependency',
    'control_dependency',
    'pointer_index',
    'other',
)

_C_KEYWORDS = {
    'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 'double',
    'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'inline', 'int', 'long',
    'register', 'restrict', 'return', 'short', 'signed', 'sizeof', 'static', 'struct',
    'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while', '_Bool',
    'class', 'namespace', 'template', 'typename', 'public', 'private', 'protected', 'new',
    'delete', 'nullptr', 'true', 'false', 'using', 'this', 'virtual', 'override', 'constexpr',
}
_STANDARD_NAMES = {
    'malloc', 'calloc', 'realloc', 'kmalloc', 'kzalloc', 'vmalloc',
    'memcpy', 'memmove', 'mempcpy', 'memset', 'memcmp', 'bcopy', 'bzero',
    'read', 'recv', 'recvfrom', 'fread', 'write', 'send', 'sendto', 'fwrite',
    'strcpy', 'strcat', 'strncpy', 'strncat', 'strlcpy', 'strlcat',
    'sprintf', 'vsprintf', 'snprintf', 'vsnprintf', 'free', 'strlen', 'strnlen',
    'sizeof', 'strcmp', 'strncmp', 'strchr', 'strrchr', 'strstr', 'memchr',
}
_IDENTIFIER = re.compile(r'\b[A-Za-z_]\w*\b')
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
class Canonicalizer:
    mapping: dict[str, str]

    def text(self, value: str) -> str:
        replaced = _IDENTIFIER.sub(lambda match: self.mapping.get(match.group(0), match.group(0)), value)
        return re.sub(r'\s+', ' ', replaced).strip()


@dataclass(frozen=True)
class GroundedRelation:
    kind: str
    source: str
    target: str

    def as_text(self) -> str:
        return f'{self.kind}|{self.source}|{self.target}'


@dataclass(frozen=True)
class MechanismRecord:
    security_effect: str
    critical_statements: tuple[str, ...]
    changed_relations: tuple[str, ...]
    added_relations: tuple[str, ...]
    removed_relations: tuple[str, ...]
    components: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return asdict(self)


def build_canonicalizer(source: str, language: str = 'c', function_name: str | None = None) -> Canonicalizer:
    function = parse_function(source, language, function_name)
    mapping: dict[str, str] = {function.name: 'FUNC'}
    for index, parameter in enumerate(function.parameters):
        mapping[parameter] = f'P{index}'
    preserved = _C_KEYWORDS | _STANDARD_NAMES
    next_index = 0
    for token in _IDENTIFIER.findall(source):
        if token in mapping or token in preserved:
            continue
        mapping[token] = f'V{next_index}'
        next_index += 1
    return Canonicalizer(mapping)


def canonicalize_source(source: str, language: str = 'c', function_name: str | None = None) -> str:
    canonicalizer = build_canonicalizer(source, language, function_name)
    return '\n'.join(canonicalizer.text(line) for line in source.splitlines() if line.strip())


def rename_local_identifiers(source: str, language: str = 'c', function_name: str | None = None) -> str:
    function = parse_function(source, language, function_name)
    names = local_identifiers(source, language, function.name)
    mapping: dict[str, str] = {function.name: 'renamed_func'}
    for index, parameter in enumerate(function.parameters):
        mapping[parameter] = f'renamed_p{index}'
    local_index = 0
    for name in names:
        if name in mapping:
            continue
        mapping[name] = f'renamed_v{local_index}'
        local_index += 1
    return _IDENTIFIER.sub(lambda match: mapping.get(match.group(0), match.group(0)), source)


def _node_category(node: GraphNode) -> set[str]:
    code = node.code
    lower = code.lower()
    categories: set[str] = set()
    if node.label == 'CONTROL_STRUCTURE' or lower.startswith(('if ', 'if(', 'while ', 'while(', 'for ', 'for(')):
        categories.add('guard')
    if _COMPARE.search(code) or any(token in node.label.lower() for token in ('greaterthan', 'lessthan', 'equals')):
        categories.update({'guard', 'bounds'})
    if any(re.search(rf'\b{re.escape(name)}\b', code) for name in _ALLOC_NAMES):
        categories.update({'allocation', 'size_arithmetic'})
    if any(re.search(rf'\b{re.escape(name)}\b', code) for name in _MEMORY_NAMES):
        categories.add('memory_access')
    if any(token in node.label for token in ('indirectIndexAccess', 'indirection', 'fieldAccess')) or '[' in code:
        categories.update({'memory_access', 'pointer_index'})
    if _ARITHMETIC.search(code) or any(
        token in node.label for token in ('addition', 'subtraction', 'multiplication', 'division', 'shiftLeft', 'shiftRight')
    ):
        categories.add('size_arithmetic')
    return categories


def _canonical_node(node: GraphNode, canonicalizer: Canonicalizer) -> str:
    return f'{node.label}:{canonicalizer.text(node.code)}'


def graph_relations(graph: FunctionGraph, canonicalizer: Canonicalizer) -> tuple[GroundedRelation, ...]:
    categories = {node_id: _node_category(node) for node_id, node in graph.nodes.items()}
    relevant = {node_id for node_id, kinds in categories.items() if kinds}
    for edge in graph.edges:
        if edge.source in relevant or edge.target in relevant:
            relevant.update((edge.source, edge.target))
    relations: list[GroundedRelation] = []
    seen: set[str] = set()
    for edge in graph.edges:
        if edge.source not in relevant and edge.target not in relevant:
            continue
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source is None or target is None:
            continue
        relation = GroundedRelation(edge.kind, _canonical_node(source, canonicalizer), _canonical_node(target, canonicalizer))
        text = relation.as_text()
        if text not in seen:
            seen.add(text)
            relations.append(relation)
    return tuple(relations)


def _critical_statements(graph: FunctionGraph, canonicalizer: Canonicalizer) -> tuple[str, ...]:
    statements: list[str] = []
    for node in graph.nodes.values():
        if not _node_category(node):
            continue
        value = _canonical_node(node, canonicalizer)
        if value not in statements:
            statements.append(value)
    return tuple(statements[:80])


def _changed_source_lines(before: str, after: str, before_c: Canonicalizer, after_c: Canonicalizer) -> tuple[list[str], list[str]]:
    left = [before_c.text(line) for line in before.splitlines() if line.strip()]
    right = [after_c.text(line) for line in after.splitlines() if line.strip()]
    removed: list[str] = []
    added: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=left, b=right, autojunk=False).get_opcodes():
        if tag in {'delete', 'replace'}:
            removed.extend(left[i1:i2])
        if tag in {'insert', 'replace'}:
            added.extend(right[j1:j2])
    return removed, added


def _effect_and_components(
    before_graph: FunctionGraph,
    after_graph: FunctionGraph,
    before_relations: set[str],
    after_relations: set[str],
    removed_lines: Iterable[str],
    added_lines: Iterable[str],
) -> tuple[str, tuple[str, ...]]:
    changed = (after_relations - before_relations) | (before_relations - after_relations)
    added_text = ' '.join(added_lines)
    removed_text = ' '.join(removed_lines)
    all_changed = f'{removed_text} {added_text}'
    components: set[str] = set()
    if any(text.startswith('CDG|') for text in changed):
        components.update({'guard', 'control_dependency'})
    if any(text.startswith('DDG|') for text in changed):
        components.add('data_dependency')
    if _COMPARE.search(all_changed):
        components.update({'guard', 'bounds'})
    if any(name in all_changed for name in _ALLOC_NAMES):
        components.add('allocation')
    if any(name in all_changed for name in _MEMORY_NAMES):
        components.add('memory_access')
    if _ARITHMETIC.search(all_changed):
        components.add('size_arithmetic')
    if '[' in all_changed or '*' in all_changed:
        components.add('pointer_index')
    before_categories = set().union(*(_node_category(node) for node in before_graph.nodes.values())) if before_graph.nodes else set()
    after_categories = set().union(*(_node_category(node) for node in after_graph.nodes.values())) if after_graph.nodes else set()
    components.update((before_categories ^ after_categories) & set(MECHANISM_COMPONENTS))

    before_guards = sum('guard' in _node_category(node) for node in before_graph.nodes.values())
    after_guards = sum('guard' in _node_category(node) for node in after_graph.nodes.values())
    if after_guards > before_guards:
        effect = 'guard_added'
    elif _COMPARE.search(removed_text) and _COMPARE.search(added_text):
        effect = 'bound_changed'
    elif 'allocation' in components or ('size_arithmetic' in components and _ARITHMETIC.search(all_changed)):
        effect = 'size_changed'
    elif 'data_dependency' in components:
        effect = 'dataflow_changed'
    elif 'memory_access' in components:
        effect = 'memory_operation_changed'
    else:
        effect = 'other'
        components.add('other')
    return effect, tuple(component for component in MECHANISM_COMPONENTS if component in components)


def derive_mechanism(
    vulnerable_source: str,
    fixed_source: str,
    vulnerable_graph: FunctionGraph,
    fixed_graph: FunctionGraph,
    *,
    language: str = 'c',
    function_name: str | None = None,
) -> MechanismRecord:
    before_c = build_canonicalizer(vulnerable_source, language, function_name)
    after_c = build_canonicalizer(fixed_source, language, function_name)
    before_relations = {item.as_text() for item in graph_relations(vulnerable_graph, before_c)}
    after_relations = {item.as_text() for item in graph_relations(fixed_graph, after_c)}
    removed_lines, added_lines = _changed_source_lines(vulnerable_source, fixed_source, before_c, after_c)
    effect, components = _effect_and_components(
        vulnerable_graph, fixed_graph, before_relations, after_relations, removed_lines, added_lines
    )
    added_relations = tuple(sorted(after_relations - before_relations))
    removed_relations = tuple(sorted(before_relations - after_relations))
    changed_relations = tuple(
        [f'ADD:{item}' for item in added_relations] + [f'REMOVE:{item}' for item in removed_relations]
    )
    critical = list(_critical_statements(vulnerable_graph, before_c))
    for item in _critical_statements(fixed_graph, after_c):
        if item not in critical:
            critical.append(item)
    return MechanismRecord(
        security_effect=effect,
        critical_statements=tuple(critical[:80]),
        changed_relations=changed_relations[:160],
        added_relations=added_relations[:80],
        removed_relations=removed_relations[:80],
        components=components,
    )


def render_graph(graph: FunctionGraph, canonicalizer: Canonicalizer, max_relations: int = 160) -> str:
    relations = graph_relations(graph, canonicalizer)
    return '\n'.join(item.as_text() for item in relations[:max_relations]) or 'NO_SECURITY_RELEVANT_GRAPH_RELATIONS'


def _required(record: dict[str, object], names: tuple[str, ...], label: str) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f'missing {label}; expected one of {", ".join(names)}')


def _pair_fields(record: dict[str, object]) -> tuple[str, str, str, str, str | None]:
    key = str(record.get('sample_key') or record.get('id') or record.get('commit_id') or '')
    if not key:
        raise ValueError('pair record requires sample_key, id, or commit_id')
    vulnerable = _required(record, ('vulnerable', 'func_before'), 'vulnerable function')
    fixed = _required(record, ('fixed', 'func_after', 'func'), 'fixed function')
    language = str(record.get('language') or 'c').lower()
    language = 'cpp' if language in {'c++', 'cpp'} else language
    if language not in {'c', 'cpp'}:
        raise ValueError(f'{key}: language must be c or cpp')
    function_name = record.get('function_name') or record.get('function')
    return key, vulnerable, fixed, language, str(function_name) if function_name else None


def _side_record(
    *, key: str, side: str, source: str, graph: FunctionGraph, mechanism: MechanismRecord,
    language: str, function_name: str, split: str | None = None,
) -> dict[str, object]:
    canonicalizer = build_canonicalizer(source, language, function_name)
    relevant = set(mechanism.components)
    targets = [1.0 if side == 'fixed' and component in relevant else 0.0 for component in MECHANISM_COMPONENTS]
    mask = [1.0 if component in relevant else 0.0 for component in MECHANISM_COMPONENTS]
    renamed = rename_local_identifiers(source, language, function_name)
    return {
        'sample_key': key,
        'side': side,
        'label': 0 if side == 'fixed' else 1,
        'language': language,
        'function_name': function_name,
        'raw_source': source,
        'renamed_source': renamed,
        'renamed_canonical_source': canonicalize_source(renamed, language, 'renamed_func'),
        'canonical_source': canonicalize_source(source, language, function_name),
        'graph': render_graph(graph, canonicalizer),
        'mechanism_effect': mechanism.security_effect,
        'mechanism_components': list(mechanism.components),
        'mechanism_targets': targets,
        'mechanism_mask': mask,
        'changed_relations': list(mechanism.changed_relations),
        'critical_statements': list(mechanism.critical_statements),
        **({'split': split} if split else {}),
    }


def build_mechanism_dataset(
    pairs_path: str | Path,
    output_path: str | Path,
    *,
    joern_dir: str | Path = '/home/phy/joern',
    java_home: str | Path = '/home/phy/jdk21',
    timeout: int = 300,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(pairs_path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'invalid JSON at {pairs_path}:{line_number}: {error}') from error
            key, vulnerable, fixed, language, function_name = _pair_fields(raw)
            vulnerable_function = parse_function(vulnerable, language, function_name)
            fixed_function = parse_function(fixed, language, function_name or vulnerable_function.name)
            if vulnerable_function.name != fixed_function.name:
                raise ValueError(f'{key}: vulnerable/fixed function names differ')
            function_name = vulnerable_function.name
            before_graph = extract_function_cpg(
                vulnerable, function_name, language=language,
                joern_dir=joern_dir, java_home=java_home, timeout=timeout,
            )
            after_graph = extract_function_cpg(
                fixed, function_name, language=language,
                joern_dir=joern_dir, java_home=java_home, timeout=timeout,
            )
            mechanism = derive_mechanism(
                vulnerable, fixed, before_graph, after_graph,
                language=language, function_name=function_name,
            )
            split = str(raw.get('split')) if raw.get('split') else None
            records.extend([
                _side_record(
                    key=key, side='vulnerable', source=vulnerable, graph=before_graph,
                    mechanism=mechanism, language=language, function_name=function_name, split=split,
                ),
                _side_record(
                    key=key, side='fixed', source=fixed, graph=after_graph,
                    mechanism=mechanism, language=language, function_name=function_name, split=split,
                ),
            ])
            print(
                f'mechanism_pair_done={key} effect={mechanism.security_effect} '
                f'components={",".join(mechanism.components)}', flush=True,
            )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
    return records
