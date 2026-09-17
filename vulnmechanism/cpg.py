from __future__ import annotations

import csv
import sys
import json
from collections import Counter, defaultdict
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .process import run_process


class CPGError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    label: str
    code: str


@dataclass(frozen=True)
class GraphEdge:
    kind: str
    source: str
    target: str


@dataclass(frozen=True)
class FunctionGraph:
    function: str
    nodes: dict[str, GraphNode]
    edges: tuple[GraphEdge, ...]
    quality: dict | None = None


def _find_executable(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise CPGError(f"required Joern executable not found under {root}: {', '.join(names)}")


def _environment(java_home: str | Path) -> dict[str, str]:
    java_home = Path(java_home).expanduser()
    java = java_home / 'bin' / 'java'
    if not java.is_file() or not os.access(java, os.X_OK):
        raise CPGError(f'Java executable not found at {java}')
    env = os.environ.copy()
    env['JAVA_HOME'] = str(java_home)
    env['PATH'] = str(java.parent) + os.pathsep + env.get('PATH', '')
    env.setdefault('JAVA_OPTS', '-Xmx4g -XX:ActiveProcessorCount=4')
    return env


class TargetMethodError(CPGError):
    pass


class CPGQualityError(CPGError):
    pass


def read_neo4jcsv(directory: Path):
    """Read Joern's streaming export, preserving typed source coordinates."""
    csv.field_size_limit(sys.maxsize)
    nodes, edges = {}, []
    for header_path in sorted(directory.glob('nodes_*_header.csv')):
        with header_path.open(encoding='utf-8', newline='') as handle:
            columns = next(csv.reader(handle))
        if columns[:2] != [':ID', ':LABEL']:
            raise CPGError(f'unexpected Joern node CSV header: {header_path.name}')
        with header_path.with_name(header_path.name.replace('_header.csv', '_data.csv')).open(encoding='utf-8', newline='') as handle:
            for values in csv.reader(handle):
                if len(values) != len(columns):
                    raise CPGError(f'malformed node row in {header_path.name}')
                node = dict(kind=values[1])
                for field, value in zip(columns[2:], values[2:]):
                    if not value:
                        continue
                    name, _, kind = field.partition(':')
                    if kind in {'int', 'long'} or name in {'LINE_NUMBER', 'LINE_NUMBER_END', 'COLUMN_NUMBER', 'COLUMN_NUMBER_END', 'OFFSET', 'OFFSET_END', 'ORDER'}:
                        node[name] = int(value)
                    elif kind == 'boolean' or name == 'IS_EXTERNAL':
                        if value not in {'true', 'false'}:
                            raise CPGError(f'invalid boolean {value!r} in {header_path.name}')
                        node[name] = value == 'true'
                    else:
                        # Flatgraph 0.1.27 escapeSpecialCharacters doubles every
                        # backslash before CSV quoting; csv.reader undoes only
                        # the quoting. Undo the exporter escape exactly once.
                        node[name] = value.replace('\\\\', '\\')
                nodes[values[0]] = node
    for kind in ('AST', 'CFG', 'CDG', 'REACHING_DEF'):
        header_path = directory / f'edges_{kind}_header.csv'
        if not header_path.exists():
            continue
        with header_path.open(encoding='utf-8', newline='') as handle:
            columns = next(csv.reader(handle))
        if columns[:3] != [':START_ID', ':END_ID', ':TYPE']:
            raise CPGError(f'unexpected Joern edge CSV header: {header_path.name}')
        with directory.joinpath(f'edges_{kind}_data.csv').open(encoding='utf-8', newline='') as handle:
            for row in csv.reader(handle):
                if len(row) != len(columns) or row[2] != kind:
                    raise CPGError(f'malformed edge row in {header_path.name}')
                edges.append((kind, row[0], row[1]))
    if not nodes:
        raise CPGError('Joern export contains no nodes')
    return nodes, edges


def resolve_target_graph(nodes, edges, *, filename: str, source: str,
                         start_line: int = 1, function_hint: str = '') -> FunctionGraph:
    from .syntax import source_tokens
    tokens = source_tokens(source)
    if not tokens or '{' not in tokens or '}' not in tokens:
        raise TargetMethodError('source has no complete function body')
    end_line = start_line + len(source.rstrip().splitlines()) - 1
    contents = next((str(n['CONTENT']) for n in nodes.values()
                     if n['kind'] == 'FILE' and n.get('NAME') == filename), '')
    encoded = contents.encode('utf-16-le')

    line_starts = [0]
    for line in contents.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line.encode('utf-16-le')) // 2)

    def physical_span(node):
        # CDT OFFSET_END uses preprocessed signature lengths for macro-rich
        # functions. Physical line/column endpoints remain source coordinates.
        line, last = node.get('LINE_NUMBER'), node.get('LINE_NUMBER_END')
        column, end_column = node.get('COLUMN_NUMBER'), node.get('COLUMN_NUMBER_END')
        if all(isinstance(v, int) for v in (line, last, column, end_column)):
            if 1 <= line <= last < len(line_starts) and column >= 1 and end_column >= 1:
                start = line_starts[line - 1] + column - 1
                # End columns can also be based on the preprocessed length.
                # When outside the reported physical line, use that line's end;
                # exact whole-definition token matching below still must pass.
                end = min(line_starts[last - 1] + end_column, line_starts[last])
                if 0 <= start < end <= len(encoded) // 2:
                    return start, end
        if node['kind'] == 'METHOD':
            return None
        start, end = node.get('OFFSET'), node.get('OFFSET_END')
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(encoded) // 2:
            return start, end
        return None

    def physical_code(node):
        span = physical_span(node)
        if span is None:
            return None
        return encoded[2 * span[0]:2 * span[1]].decode('utf-16-le')

    occurrences = [m.start() for m in re.finditer(re.escape(source), contents)]
    occurrences = [p for p in occurrences if contents[:p].count('\n') + 1 == start_line]
    if len(occurrences) != 1:
        raise TargetMethodError('target source occurrence is not unique at requested line')
    target_start = len(contents[:occurrences[0]].encode('utf-16-le')) // 2
    target_end = target_start + len(source.rstrip().encode('utf-16-le')) // 2
    candidates = []
    available = []
    for key, node in nodes.items():
        if node['kind'] != 'METHOD' or node.get('IS_EXTERNAL') or node.get('NAME') in {'<global>', 'if', 'for', 'while', 'switch', 'catch', 'sizeof', 'do'}:
            continue
        if Path(str(node.get('FILENAME', ''))).as_posix() != filename:
            continue
        available.append((node.get('NAME'), node.get('LINE_NUMBER'), node.get('LINE_NUMBER_END')))
        code = physical_code(node)
        first, last = node.get('LINE_NUMBER'), node.get('LINE_NUMBER_END')
        # A full-file definition may include a return type/modifier omitted by
        # the upstream function extractor. Admit only a declaration prefix, with
        # the exact requested body and endpoint; never a containing class/method.
        span = physical_span(node)
        if span is None or code is None:
            continue
        offset, stop = span
        if stop < target_end:
            continue
        if offset > target_start and source_tokens(encoded[2 * target_start:2 * offset].decode('utf-16-le')):
            continue
        prefix = encoded[2 * offset:2 * target_start].decode('utf-16-le') if offset <= target_start else ''
        suffix = encoded[2 * target_end:2 * stop].decode('utf-16-le') if stop >= target_end else ''
        if set(source_tokens(prefix)).intersection({'{', '}', ';', '=', '#'}) or source_tokens(suffix):
            continue
        if source_tokens(code) != source_tokens(prefix) + tokens:
            continue
        if isinstance(first, int) and isinstance(last, int) and first <= end_line and last <= end_line:
            candidates.append((key, node))
    if len(candidates) != 1:
        raise TargetMethodError(f'expected one source-aligned target method; found {len(candidates)}; '
                                f'hint={function_hint!r}; methods={available[:30]}')
    method_id, method = candidates[0]
    ast = defaultdict(list)
    for kind, a, b in edges:
        if kind == 'AST':
            ast[a].append(b)
    owned, pending = set(), [method_id]
    while pending:
        key = pending.pop()
        if key in owned:
            continue
        # Nested lambdas/methods belong to separate functions, not this target.
        if key != method_id and nodes[key]['kind'] == 'METHOD':
            continue
        owned.add(key)
        pending.extend(ast[key])
    graph_nodes = {}
    for key in sorted(owned, key=int):
        n = nodes[key]
        label = n.get('NAME', 'CALL') if n['kind'] == 'CALL' else {'METHOD_PARAMETER_IN': 'PARAM'}.get(n['kind'], n['kind'])
        code = str(n.get('CODE', ''))
        if n['kind'] not in {'METHOD', 'BLOCK', 'CONTROL_STRUCTURE'} and len(code) == 1000 and code.endswith('...'):
            restored = physical_code(n)
            if restored is None or not restored.startswith(code[:-3]):
                raise CPGQualityError(f"truncated {n['kind']} node code without verified source coordinates")
            code = restored
        if n['kind'] == 'METHOD':
            code = str(n['NAME'])
        graph_nodes[key] = GraphNode(key, str(label), code)
    kinds = {'AST': 'AST', 'CFG': 'CFG', 'CDG': 'CDG', 'REACHING_DEF': 'DDG'}
    graph_edges = tuple(GraphEdge(kinds[k], a, b) for k, a, b in edges
                        if k in kinds and a in owned and b in owned)
    counts = Counter(e.kind for e in graph_edges)
    unknown = [k for k in owned if nodes[k]['kind'] == 'UNKNOWN']
    executable = [k for k in owned if nodes[k]['kind'] in {'CALL', 'RETURN', 'CONTROL_STRUCTURE', 'JUMP_TARGET'}]
    body_node = next((nodes[k] for k in ast[method_id] if nodes[k]['kind'] == 'BLOCK'), {})
    body_tokens = source_tokens(str(body_node.get('CODE', '')))
    body_line, body_column = body_node.get('LINE_NUMBER'), body_node.get('COLUMN_NUMBER')
    expected_body = ()
    if isinstance(body_line, int) and isinstance(body_column, int) and 1 <= body_line < len(line_starts):
        body_start = line_starts[body_line - 1] + body_column - 1
        expected_body = source_tokens(encoded[2 * body_start:2 * physical_span(method)[1]].decode('utf-16-le'))
    reasons = []
    if not counts['AST'] or not counts['CFG']:
        reasons.append('missing_ast_or_cfg')
    if unknown:
        reasons.append('unknown_ast_nodes')
    header = source.split('{', 1)[0]
    native_name = str(method['NAME'])
    if re.fullmatch(r'[A-Z][A-Z_0-9]*', native_name):
        sole_macro = re.match(r'\s*' + re.escape(native_name) + r'\s*\(', header)
        name_macro = re.search(r'\b' + re.escape(native_name) + r'\s*\([^()]*\)\s*\(', header)
        if sole_macro or name_macro:
            reasons.append('unresolved_macro_signature')
    if body_tokens == ('{', '}') and len(expected_body) > 2:
        reasons.append('body_content_missing')
    if len(body_tokens) > 2 and not executable:
        # A body containing only local declarations can legitimately have no CALL.
        if not any(nodes[k]['kind'] == 'LOCAL' for k in owned):
            reasons.append('nonempty_body_without_executable_nodes')
    if not body_tokens or body_tokens in {('<', 'empty', '>'), ()}:
        reasons.append('missing_body')
    quality = dict(target_method_id=method_id, target_name=method['NAME'],
                   target_full_name=method.get('FULL_NAME'), filename=filename,
                   start_line=method.get('LINE_NUMBER'), end_line=method.get('LINE_NUMBER_END'),
                   source_alignment='file_content_physical_line_columns_exact_body_with_declaration_prefix',
                   method_offset_disagreement=physical_span(method) != (method.get('OFFSET'), method.get('OFFSET_END')),
                   hint_mismatch=bool(function_hint and function_hint.rsplit('::', 1)[-1] != method['NAME']),
                   unknown_nodes=len(unknown), executable_nodes=len(executable),
                   executable_nodes_without_cfg=len(set(executable) - {v for e in graph_edges if e.kind == 'CFG' for v in (e.source, e.target)}),
                   truncated_container_nodes=sum(nodes[k]['kind'] in {'BLOCK', 'CONTROL_STRUCTURE'} and len(str(nodes[k].get('CODE', ''))) == 1000 and str(nodes[k].get('CODE', '')).endswith('...') for k in owned),
                   node_count=len(owned), edge_count=len(graph_edges),
                   edge_counts={k: counts[k] for k in ('AST', 'CFG', 'CDG', 'DDG')},
                   missing_relation_kinds=[k for k in ('AST', 'CFG', 'CDG', 'DDG') if not counts[k]],
                   status='rejected' if reasons else 'accepted', reasons=reasons)
    if reasons:
        raise CPGQualityError(json.dumps(quality, sort_keys=True))
    return FunctionGraph(str(method['NAME']), graph_nodes, graph_edges, quality)


def extract_function_cpg_batch(requests: list[dict], *, joern_dir='/home/phy/joern',
                               java_home='/home/phy/jdk21', timeout=300) -> list[FunctionGraph | CPGError]:
    """One parser and one lossless graph export per bounded batch; no retries."""
    if not requests:
        return []
    root = Path(os.environ.get('JOERN_HOME', str(joern_dir))).expanduser()
    parser = _find_executable(root, ('joern-parse', 'joern-cli/joern-parse', 'joern-cli/bin/joern-parse'))
    exporter = _find_executable(root, ('joern-export', 'joern-cli/joern-export', 'joern-cli/bin/joern-export'))
    env = _environment(java_home)
    with tempfile.TemporaryDirectory(prefix='vulnmechanism-cpg-') as directory:
        work = Path(directory)
        src = work / 'src'
        src.mkdir()
        filenames = []
        for index, request in enumerate(requests):
            if request['language'] not in {'c', 'cpp'}:
                raise ValueError('language must be c or cpp')
            filename = f'sample_{index:04d}.' + ('cpp' if request['language'] == 'cpp' else 'c')
            contents = request.get('full_source', request['source'])
            start = request.get('start_line', 1)
            if 'full_source' in request:
                positions = [m.start() for m in re.finditer(re.escape(request['source']), contents)]
                if len(positions) != 1 or contents[:positions[0]].count('\n') + 1 != start:
                    raise ValueError('full-file target must have one exact source occurrence at start_line')
            (src / filename).write_text(contents, encoding='utf-8')
            filenames.append(filename)
        cpg = work / 'cpg.bin'
        result = run_process([str(parser), str(src), '--output', str(cpg), '--frontend-args', '--enable-file-content'], timeout=timeout, env=env)
        if result.returncode or not cpg.is_file():
            raise CPGError('joern-parse failed: ' + (result.stderr or result.stdout)[-4000:])
        output = work / 'graph'
        result = run_process([str(exporter), '--repr', 'all', '--format', 'neo4jcsv',
                              '--out', str(output), str(cpg)], timeout=timeout, env=env)
        if result.returncode:
            raise CPGError('joern-export failed: ' + (result.stderr or result.stdout)[-4000:])
        nodes, edges = read_neo4jcsv(output)
        results = []
        for filename, request in zip(filenames, requests):
            try:
                results.append(resolve_target_graph(nodes, edges, filename=filename,
                    source=request['source'], start_line=request.get('start_line', 1),
                    function_hint=request.get('function', '')))
            except (TargetMethodError, CPGQualityError) as error:
                results.append(error)
        return results


def extract_function_cpg(source: str, function: str, *, language='c',
                         joern_dir='/home/phy/joern', java_home='/home/phy/jdk21', timeout=300,
                         full_source: str | None = None, start_line: int = 1) -> FunctionGraph:
    request = dict(source=source, function=function, language=language, start_line=start_line)
    if full_source is not None:
        request['full_source'] = full_source
    result = extract_function_cpg_batch([request], joern_dir=joern_dir,
                                       java_home=java_home, timeout=timeout)[0]
    if isinstance(result, CPGError):
        raise result
    return result
