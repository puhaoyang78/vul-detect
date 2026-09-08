from __future__ import annotations

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


_NODE = re.compile(r'^\s*"?(\d+)"?\s*\[label\s*=\s*(.+?)\]\s*;?\s*$')
_EDGE = re.compile(r'^\s*"?(\d+)"?\s*->\s*"?(\d+)"?')
_GRAPH = re.compile(r'^\s*digraph\s+"?([^"{]+)"?')


def _clean_dot_label(raw: str) -> str:
    value = raw.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    value = value.replace('\\"', '"').replace('\\n', ' ')
    value = re.sub(r'<[^>]+>', '', value)
    return ' '.join(value.split())


def _node_parts(label: str) -> tuple[str, str]:
    value = label[1:-1] if label.startswith('(') and label.endswith(')') else label
    parts = [part.strip() for part in value.split(',', 2)]
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
                graph_name = match.group(1).strip()
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
    return env


def _matching_dot(directory: Path, function: str) -> Path:
    matches: list[Path] = []
    for path in sorted(directory.rglob('*.dot')):
        try:
            first = path.read_text(errors='replace').splitlines()[0]
        except (OSError, IndexError):
            continue
        match = _GRAPH.match(first)
        if match and match.group(1).strip().strip('"') == function:
            matches.append(path)
    if len(matches) != 1:
        raise CPGError(f'expected one graph for {function}, found {len(matches)}')
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
    c2cpg = _find_executable(root, (
        'c2cpg.sh', 'joern-cli/c2cpg.sh', 'joern-cli/frontends/c2cpg/c2cpg.sh'
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
            str(c2cpg), str(src), '--output', str(cpg),
            '--with-include-auto-discovery', '--log-problems',
        ], timeout=timeout, env=env)
        if parsed.returncode != 0 or not cpg.is_file():
            raise CPGError('standalone c2cpg failed: ' + (parsed.stderr.strip() or parsed.stdout.strip()))

        graphs: list[FunctionGraph] = []
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
            path = _matching_dot(output, function)
            graphs.append(parse_dot_graph(path.read_text(errors='replace'), representation, function))
        return merge_graphs(function, graphs)
