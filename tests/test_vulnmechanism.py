import unittest
import tempfile
from pathlib import Path

<<<<<<< Updated upstream
from vulnmechanism.cpg import FunctionGraph, GraphEdge, GraphNode, parse_dot_graph
from vulnmechanism.mechanism import _normalize_label, graph_relations
from vulnmechanism.semantics import extract_vulnerability_semantics
=======
from vulnmechanism.cpg import CPGError, FunctionGraph, GraphEdge, GraphNode, _matching_dot, parse_dot_graph
from vulnmechanism.mechanism import _normalize_label, graph_relations, render_graph
>>>>>>> Stashed changes
from vulnmechanism.syntax import parse_function


class SyntaxTests(unittest.TestCase):
    def test_cpp_special_names_and_macro_parameter_header(self):
        cases = [
            ('Foo::~Foo() {}', 'cpp', '~Foo'),
            ('int Foo::operator[](int index) { return index; }', 'cpp', 'operator[]'),
            ('bool Foo::operator==(const Foo& other) { return true; }', 'cpp', 'operator=='),
            ('Foo::operator bool() { return true; }', 'cpp', 'operator bool'),
            ('path_inter(PG_FUNCTION_ARGS) { return 0; }', 'c', 'path_inter'),
            ('compute(ARGUMENTS) { return 0; }', 'c', 'compute'),
        ]
        for source, language, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(parse_function(source, language).name, expected)

    def test_implicit_return_type_void_parameter(self):
        for name in ('cleanup', 'release_resources'):
            parsed = parse_function(f'{name}(void) {{ close(0); }}', 'c')
            self.assertEqual(parsed.name, name)
            self.assertEqual(parsed.parameters, ())
        self.assertEqual(parse_function('void (cleanup)(void) {}', 'c').name, 'cleanup')

    def test_parse_function(self):
        parsed = parse_function('int foo(char *buf, int len) { return buf[len]; }', 'c', 'foo')
        self.assertEqual(parsed.name, 'foo')
        self.assertEqual(parsed.parameters, ('buf', 'len'))


class CPGTests(unittest.TestCase):
    def test_joern_symbolic_operator_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '0-ast.dot'
            path.write_text('digraph "[]" {\n"1" [label = <METHOD, 1<BR/>[]> ]\n}\n')
            self.assertEqual(_matching_dot(Path(directory), 'operator[]'), path)
            self.assertEqual(_matching_dot(Path(directory), 'operator []'), path)
            path.write_text('digraph "bool" {\n"1" [label = <METHOD, 1<BR/>bool> ]\n}\n')
            self.assertEqual(_matching_dot(Path(directory), 'operator bool'), path)

    def test_relation_budget_preserves_kinds_and_redistributes_unused_slots(self):
        nodes = {str(i): GraphNode(str(i), 'IDENTIFIER', f'buf[{i}]') for i in range(164)}
        edges = tuple(GraphEdge('AST', '0', str(i)) for i in range(1, 161))
        edges += tuple(GraphEdge(kind, '0', str(i)) for i, kind in enumerate(('CFG', 'CDG', 'DDG'), 161))
        graph = FunctionGraph('f', nodes, edges)
        rows = render_graph(graph).splitlines()
        self.assertEqual(len(rows), 160)
        self.assertEqual({r.split('|')[0] for r in rows}, {'AST', 'CFG', 'CDG', 'DDG'})
        self.assertEqual(len(set(rows)), 160)
        self.assertEqual(len(render_graph(graph, 200).splitlines()), len(edges))
        self.assertEqual(render_graph(graph), render_graph(graph))
        with self.assertRaises(ValueError):
            render_graph(graph, 0)

    def test_joern_html_labels_preserve_operators_and_arguments(self):
        graph = parse_dot_graph('''digraph "foo" {
"1" [label = <METHOD, 1<BR/>foo> ]
"2" [label = <&lt;operator&gt;.lessThan, 2<BR/>n &lt; cap> ]
"3" [label = <memcpy, 3<BR/>memcpy(dst, src, n)> ]
"1" -> "2"
"2" -> "3"
}''', 'ast')
        self.assertEqual(graph.nodes['1'].label, 'METHOD')
        self.assertEqual(graph.nodes['2'].label, '<operator>.lessThan')
        self.assertEqual(graph.nodes['2'].code, 'n < cap')
        self.assertEqual(graph.nodes['3'].code, 'memcpy(dst, src, n)')
        self.assertEqual(len(graph.edges), 2)

    def test_matching_html_graph_name_and_missing_method(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '0-ast.dot'
            path.write_text('digraph "foo&lt;int&gt;" {\n"1" [label = <METHOD, 1<BR/>foo&lt;int&gt;> ]\n}\n')
            self.assertEqual(_matching_dot(Path(directory), 'foo<int>'), path)
            with self.assertRaisesRegex(CPGError, 'exported methods:'):
                _matching_dot(Path(directory), 'absent')

    def test_same_name_external_method_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ast = root / 'ast'
            ast.mkdir()
            # Give the external stub the lower index to rule out first-match selection.
            (ast / '0-ast.dot').write_text(
                'digraph "foo" {\n"1" [label = <METHOD<BR/>foo> ]\n'
                '"2" [label = <BLOCK<BR/>&lt;empty&gt;> ]\n"1" -> "2"\n}\n')
            definition = ast / '3-ast.dot'
            definition.write_text(
                'digraph "foo" {\n"3" [label = <METHOD, 1<BR/>foo> ]\n'
                '"4" [label = <BLOCK, 1<BR/>{}> ]\n"3" -> "4"\n}\n')
            self.assertEqual(_matching_dot(ast, 'foo'), definition)
            # An empty CDG has no METHOD node: retain the selected AST index.
            cdg = root / 'cdg'
            cdg.mkdir()
            for index in (0, 3):
                (cdg / f'{index}-cdg.dot').write_text('digraph "foo" {\n}\n')
            self.assertEqual(_matching_dot(cdg, 'foo', '3'), cdg / '3-cdg.dot')
            (ast / '4-ast.dot').write_text(definition.read_text())
            with self.assertRaisesRegex(CPGError, 'found 2'):
                _matching_dot(ast, 'foo')
            definition.unlink()
            (ast / '4-ast.dot').unlink()
            with self.assertRaisesRegex(CPGError, 'found 0'):
                _matching_dot(ast, 'foo')

    def test_parse_dot_graph(self):
        graph = parse_dot_graph(
            '''digraph "foo" {
"1" [label = "(METHOD,foo)" ]
"2" [label = "(CONTROL_STRUCTURE,if (n < cap),if (n < cap))" ]
"3" [label = "(memcpy,memcpy(dst, src, n))" ]
"2" -> "3"
}
''',
            'cdg',
        )
        self.assertEqual(graph.function, 'foo')
        self.assertEqual(graph.nodes['2'].label, 'CONTROL_STRUCTURE')
        self.assertEqual(graph.edges, (GraphEdge('CDG', '2', '3'),))

    def test_security_relevant_relations_keep_graph_kind(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'CONTROL_STRUCTURE', 'if (n < cap)'),
                '2': GraphNode('2', 'CALL', 'memcpy(dst, src, n)'),
                '3': GraphNode('3', 'IDENTIFIER', 'n'),
            },
            (GraphEdge('CDG', '1', '2'), GraphEdge('DDG', '3', '2')),
        )
        relations = graph_relations(graph)
        self.assertEqual({relation.kind for relation in relations}, {'CDG', 'DDG'})
        self.assertTrue(any('memcpy' in relation.as_text() for relation in relations))


class SemanticTests(unittest.TestCase):
    def test_guarded_write_extracts_memory_dependency_and_guard(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'METHOD_PARAMETER_IN', 'len'),
                '2': GraphNode('2', 'CONTROL_STRUCTURE', 'if (len <= 64)'),
                '3': GraphNode('3', 'CALL', 'memcpy(buf, src, len)'),
            },
            (
                GraphEdge('DDG', '1', '3'),
                GraphEdge('CDG', '2', '3'),
                GraphEdge('CFG', '2', '3'),
            ),
        )
        semantics = extract_vulnerability_semantics(graph)
        text = semantics.render()
        self.assertIn('WRITE', text)
        self.assertIn('PARAMETER_DEP', text)
        self.assertIn('GUARD_PROTECTS', text)
        self.assertNotIn('UNGUARDED_WRITE_EXTENT', text)

    def test_unguarded_write_is_risk_candidate(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'METHOD_PARAMETER_IN', 'len'),
                '2': GraphNode('2', 'CALL', 'memcpy(buf, src, len)'),
            },
            (GraphEdge('DDG', '1', '2'),),
        )
        text = extract_vulnerability_semantics(graph).render()
        self.assertIn('UNGUARDED_WRITE_EXTENT', text)

    def test_cfg_free_then_use_is_lifetime_candidate(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'CALL', 'free(p)'),
                '2': GraphNode('2', '<operator>.indirection', '*p'),
            },
            (GraphEdge('CFG', '1', '2'),),
        )
        text = extract_vulnerability_semantics(graph).render()
        self.assertIn('FREE', text)
        self.assertIn('USE_AFTER_FREE', text)


class LabelTests(unittest.TestCase):
    def test_binary_labels_are_preserved(self):
        self.assertEqual(_normalize_label(0), 0)
        self.assertEqual(_normalize_label(1), 1)
        self.assertEqual(_normalize_label('0'), 0)
        self.assertEqual(_normalize_label('1'), 1)

    def test_invalid_label_is_rejected(self):
        with self.assertRaises(ValueError):
            _normalize_label(2)


if __name__ == '__main__':
    unittest.main()
