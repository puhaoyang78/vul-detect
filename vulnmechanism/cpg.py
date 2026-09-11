from __future__ import annotations

import os
import re
import tempfile
from html import unescape
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


_NODE = re.compile(r'^\s*"?(\d+)"?\s*\[label\s*=\s*(.+?)\]\s*;?\s*$')
_EDGE = re.compile(r'^\s*"?(\d+)"?\s*->\s*"?(\d+)"?')
_GRAPH = re.compile(r'^\s*digraph\s+"?([^"{]+)"?')


def _clean_dot_label(raw: str) -> str:
    value = raw.strip()
    if value.startswith('<') and value.endswith('>'):
        parts = re.split(r'<BR\s*/?>', value[1:-1], maxsplit=1, flags=re.I)
        kind = unescape(parts[0].split(',', 1)[0].strip())
        code = parts[1] if len(parts) == 2 else kind
        return '(' + kind + ',' + unescape(code) + ')'
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    value = value.replace('\\"', '"').replace('\\n', ' ')
    return ' '.join(value.split())


def _node_parts(label: str) -> tuple[str, str]:
    value = label[1:-1] if label.startswith('(') and label.endswith(')') else label
    parts = [part.strip() for part in value.split(',', 1)]
    kind = parts[0] if parts else ''
    code = parts[-1] if len(parts) >= 2 else kind
    return kind, code


def parse_dot_graph(text: str, edge_kind: str, function: str | None = None) -> FunctionGraph:
    graph_name = function or ''
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    for line in text.splitlines():
        if not graph_name:
            match = _GRAPH.match(line)
            if match:
                graph_name = unescape(match.group(1).strip())
        edge = _EDGE.match(line)
        if edge:
            edges.append(GraphEdge(edge_kind.upper(), edge.group(1), edge.group(2)))
            continue
        node = _NODE.match(line)
        if node:
            label = _clean_dot_label(node.group(2))
            kind, code = _node_parts(label)
            nodes[node.group(1)] = GraphNode(node.group(1), kind, code)
    return FunctionGraph(graph_name, nodes, tuple(edges))


def merge_graphs(function: str, graphs: list[FunctionGraph]) -> FunctionGraph:
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for graph in graphs:
        nodes.update(graph.nodes)
        for edge in graph.edges:
            key = (edge.kind, edge.source, edge.target)
            if key not in seen:
                seen.add(key)
                edges.append(edge)
    return FunctionGraph(function, nodes, tuple(edges))


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


def _matching_dot(directory: Path, function: str, method_index: str | None = None) -> Path:
    """Select a source-defined method, then keep its export index across layers."""
    # Joern omits "operator" for overloads and conversions ([], bool, ...).
    operator = re.fullmatch(r'operator\s+(.+)|operator\s*([+\-*/%<>=!&|^~\[\](),]+)', function)
    exported_name = (operator.group(1) or operator.group(2)) if operator else function
    matches: list[Path] = []
    available: list[str] = []
    for path in sorted(directory.rglob('*.dot')):
        try:
            contents = path.read_text(errors='replace')
            first = contents.splitlines()[0]
        except (OSError, IndexError):
            continue
        match = _GRAPH.match(first)
        if match:
            name = unescape(match.group(1).strip())
            available.append(name)
            if name == exported_name:
                if method_index is not None:
                    if path.name == f'{method_index}-{directory.name}.dot':
                        matches.append(path)
                else:
                    # External call stubs also have METHOD/BLOCK nodes, but
                    # only source definitions carry a METHOD source line.
                    for line in contents.splitlines():
                        node = _NODE.match(line)
                        if node and re.match(r'<METHOD,\s*\d+<BR\s*/?>', node.group(2), re.I):
                            matches.append(path)
                            break
    if len(matches) != 1:
        raise CPGError(f'expected one {directory.name} graph for {function}, '
                       f'found {len(matches)}; exported methods: {available}')
    return matches[0]


def extract_function_cpg(
    source: str,
    function: str,
    *,
    language: str = 'c',
    joern_dir: str | Path = '/home/phy/joern',
    java_home: str | Path = '/home/phy/jdk21',
    timeout: int = 300,
) -> FunctionGraph:
    """Build AST/CFG/CDG/DDG from one standalone C/C++ function snippet."""
    if language not in {'c', 'cpp'}:
        raise ValueError('language must be c or cpp')
    root = Path(os.environ.get('JOERN_HOME', str(joern_dir))).expanduser()
    parser = _find_executable(root, (
        'joern-parse', 'joern-cli/joern-parse', 'joern-cli/bin/joern-parse'
    ))
    exporter = _find_executable(root, (
        'joern-export', 'joern-cli/joern-export', 'joern-cli/bin/joern-export'
    ))
    env = _environment(java_home)

    with tempfile.TemporaryDirectory(prefix='vulnmechanism-cpg-') as directory:
        work = Path(directory)
        src = work / 'src'
        src.mkdir()
        suffix = '.cpp' if language == 'cpp' else '.c'
        (src / f'input{suffix}').write_text(source)
        cpg = work / 'cpg.bin'
        parsed = run_process([
            str(parser), str(src), '--output', str(cpg),
        ], timeout=timeout, env=env)
        if parsed.returncode != 0 or not cpg.is_file():
            raise CPGError('joern-parse failed: ' + (parsed.stderr.strip() or parsed.stdout.strip()))

        graphs: list[FunctionGraph] = []
        method_index: str | None = None
        for representation in ('ast', 'cfg', 'cdg', 'ddg'):
            output = work / representation
            exported = run_process([
                str(exporter), '--repr', representation, '--format', 'dot',
                '--out', str(output), str(cpg),
            ], timeout=timeout, env=env)
            if exported.returncode != 0:
                raise CPGError(
                    f'joern-export {representation} failed: '
                    + (exported.stderr.strip() or exported.stdout.strip())
                )
            path = _matching_dot(output, function, method_index)
            if representation == 'ast':
                index_match = re.fullmatch(r'(\d+)-ast\.dot', path.name)
                if index_match is None:
                    raise CPGError(f'unexpected Joern export filename: {path.name}')
                method_index = index_match.group(1)
            graph = parse_dot_graph(path.read_text(errors='replace'), representation, function)
            if representation in {'ast', 'cfg'} and (not graph.nodes or not graph.edges):
                raise CPGError(f'{function}: empty {representation} graph')
            graphs.append(graph)
        return merge_graphs(function, graphs)
