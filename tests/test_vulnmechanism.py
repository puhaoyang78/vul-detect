import tempfile
from pathlib import Path
import unittest

from vulnmechanism.cpg import CPGError, FunctionGraph, GraphEdge, GraphNode, _matching_dot, parse_dot_graph
from vulnmechanism.dataset import _normalize_label, extract_cpg_relations, render_cpg_relations
from vulnmechanism.semantics import (
    VULNERABILITY_FEATURES,
    extract_vulnerability_semantics,
    render_semantic_items,
    validate_semantic_groups,
)
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
        nodes = {str(i): GraphNode(str(i), 'IDENTIFIER', f'value{i}') for i in range(164)}
        edges = tuple(GraphEdge('AST', '0', str(i)) for i in range(1, 161))
        edges += tuple(GraphEdge(kind, '0', str(i)) for i, kind in enumerate(('CFG', 'CDG', 'DDG'), 161))
        graph = FunctionGraph('f', nodes, edges)
        rows = render_cpg_relations(graph).splitlines()
        self.assertEqual(len(rows), 160)
        self.assertEqual({r.split('|')[0] for r in rows}, {'AST', 'CFG', 'CDG', 'DDG'})
        self.assertEqual(len(set(rows)), 160)
        self.assertEqual(len(render_cpg_relations(graph, 200).splitlines()), len(edges))
        with self.assertRaises(ValueError):
            render_cpg_relations(graph, 0)

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
            (ast / '0-ast.dot').write_text(
                'digraph "foo" {\n"1" [label = <METHOD<BR/>foo> ]\n'
                '"2" [label = <BLOCK<BR/>&lt;empty&gt;> ]\n"1" -> "2"\n}\n')
            definition = ast / '3-ast.dot'
            definition.write_text(
                'digraph "foo" {\n"3" [label = <METHOD, 1<BR/>foo> ]\n'
                '"4" [label = <BLOCK, 1<BR/>{}> ]\n"3" -> "4"\n}\n')
            self.assertEqual(_matching_dot(ast, 'foo'), definition)
            cdg = root / 'cdg'
            cdg.mkdir()
            for index in (0, 3):
                (cdg / f'{index}-cdg.dot').write_text('digraph "foo" {\n}\n')
            self.assertEqual(_matching_dot(cdg, 'foo', '3'), cdg / '3-cdg.dot')

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

    def test_raw_cpg_relations_keep_all_graph_kinds_without_semantic_filtering(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'METHOD', 'foo'),
                '2': GraphNode('2', 'IDENTIFIER', 'ordinary_value'),
                '3': GraphNode('3', 'CALL', 'helper(ordinary_value)'),
            },
            (
                GraphEdge('AST', '1', '2'),
                GraphEdge('CFG', '2', '3'),
                GraphEdge('DDG', '2', '3'),
            ),
        )
        relations = extract_cpg_relations(graph)
        self.assertEqual({relation.kind for relation in relations}, {'AST', 'CFG', 'DDG'})
        self.assertEqual(len(relations), 3)
        self.assertTrue(any('ordinary_value' in relation.as_text() for relation in relations))


class SemanticTests(unittest.TestCase):
    def test_upper_bound_form_condition_controls_write(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'METHOD_PARAMETER_IN', 'len'),
                '2': GraphNode('2', '<operator>.lessEqualsThan', 'len <= 64'),
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
        self.assertIn('MEMORY_WRITE', text)
        self.assertIn('PARAMETER_DEPENDENCE', text)
        self.assertIn('CONDITION_CONTROLS', text)
        self.assertIn('UPPER_BOUND_RELATED_CONDITION', text)
        self.assertNotIn('WRITE_EXTENT_WITHOUT_UPPER_BOUND_CONDITION', text)
        self.assertIn('memory_write', semantics.feature_names)
        self.assertIn('bounds_constraint', semantics.feature_names)
        self.assertTrue(set(semantics.feature_names) <= set(VULNERABILITY_FEATURES))

    def test_positive_length_test_is_not_upper_bound_condition(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', '<operator>.greaterThan', 'len > 0'),
                '2': GraphNode('2', 'CALL', 'memcpy(buf, src, len)'),
            },
            (GraphEdge('CDG', '1', '2'),),
        )
        text = extract_vulnerability_semantics(graph).render()
        self.assertIn('WRITE_EXTENT_WITHOUT_UPPER_BOUND_CONDITION', text)
        self.assertNotIn('UPPER_BOUND_RELATED_CONDITION', text)

    def test_condition_is_not_claimed_to_be_a_safe_guard(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', '<operator>.greaterThan', 'len > cap'),
                '2': GraphNode('2', 'CALL', 'memcpy(buf, src, len)'),
            },
            (GraphEdge('CDG', '1', '2'),),
        )
        text = extract_vulnerability_semantics(graph).render()
        self.assertIn('WRITE_EXTENT_WITHOUT_UPPER_BOUND_CONDITION', text)
        self.assertNotIn('UPPER_BOUND_RELATED_CONDITION', text)
        self.assertNotIn('GUARD_PROTECTS', text)
        self.assertNotIn('BOUNDS_CHECK', text)

    def test_unguarded_write_is_potential_pattern(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'METHOD_PARAMETER_IN', 'len'),
                '2': GraphNode('2', 'CALL', 'memcpy(buf, src, len)'),
            },
            (GraphEdge('DDG', '1', '2'),),
        )
        text = extract_vulnerability_semantics(graph).render()
        self.assertIn('WRITE_EXTENT_WITHOUT_UPPER_BOUND_CONDITION', text)
        self.assertIn('data_from=len', text)

    def test_null_branch_is_not_treated_as_nonnull_condition(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', '<operator>.equals', 'p == NULL'),
                '2': GraphNode('2', '<operator>.indirection', '*p'),
            },
            (GraphEdge('CDG', '1', '2'),),
        )
        text = extract_vulnerability_semantics(graph).render()
        self.assertIn('NULL_RELATED_CONDITION', text)
        self.assertIn('DEREFERENCE_WITHOUT_NONNULL_CONDITION', text)
        self.assertNotIn('NONNULL_RELATED_CONDITION', text)

    def test_nonnull_form_condition_is_recorded(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', '<operator>.notEquals', 'p != NULL'),
                '2': GraphNode('2', '<operator>.indirection', '*p'),
            },
            (GraphEdge('CDG', '1', '2'),),
        )
        text = extract_vulnerability_semantics(graph).render()
        self.assertIn('NONNULL_RELATED_CONDITION', text)
        self.assertNotIn('DEREFERENCE_WITHOUT_NONNULL_CONDITION', text)

    def test_dynamic_allocation_capacity_is_attached_to_write(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'CALL', 'buf = malloc(cap)'),
                '2': GraphNode('2', 'CALL', 'memcpy(buf, src, len)'),
            },
            (
                GraphEdge('CFG', '1', '2'),
                GraphEdge('DDG', '1', '2'),
            ),
        )
        text = extract_vulnerability_semantics(graph).render()
        self.assertIn('DYNAMIC_CAPACITY object=buf capacity=cap', text)
        self.assertIn('capacity=cap', text)

    def test_ambiguous_dynamic_capacities_are_not_collapsed(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'CALL', 'buf = malloc(a)'),
                '2': GraphNode('2', 'CALL', 'buf = realloc(buf, b)'),
                '3': GraphNode('3', 'CALL', 'memcpy(buf, src, len)'),
            },
            (
                GraphEdge('CFG', '1', '2'),
                GraphEdge('CFG', '2', '3'),
            ),
        )
        text = extract_vulnerability_semantics(graph).render()
        self.assertNotIn('DYNAMIC_CAPACITY object=buf', text)

    def test_cfg_free_then_use_is_lifetime_relation_and_pattern(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'CALL', 'free(p)'),
                '2': GraphNode('2', '<operator>.indirection', '*p'),
            },
            (GraphEdge('CFG', '1', '2'),),
        )
        semantics = extract_vulnerability_semantics(graph)
        text = semantics.render()
        self.assertIn('DEALLOCATION', text)
        self.assertIn('FREE_THEN_USE', text)
        self.assertIn('lifetime_relation', semantics.feature_names)
        pattern_only = semantics.render(excluded_groups=('memory', 'dependence', 'constraint', 'lifetime'))
        self.assertIn('FREE_THEN_USE', pattern_only)

    def test_pointer_syntax_is_not_size_arithmetic(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', '<operator>.indirection', '*p'),
                '2': GraphNode('2', 'CALL', 'memcpy(buf, p, n)'),
            },
            (GraphEdge('DDG', '1', '2'),),
        )
        semantics = extract_vulnerability_semantics(graph)
        self.assertNotIn('size_arithmetic', semantics.feature_names)
        self.assertNotIn('SIZE_ARITHMETIC', semantics.render())

    def test_cpp_new_delete_are_memory_and_lifetime_operations(self):
        graph = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', '<operator>.new', 'p = new char[n]'),
                '2': GraphNode('2', '<operator>.delete', 'delete[] p'),
            },
            (GraphEdge('CFG', '1', '2'),),
        )
        semantics = extract_vulnerability_semantics(graph)
        text = semantics.render()
        self.assertIn('ALLOCATION', text)
        self.assertIn('DEALLOCATION', text)
        self.assertIn('allocation', semantics.feature_names)
        self.assertIn('deallocation', semantics.feature_names)

    def test_patterns_render_before_lower_level_categories(self):
        items = [
            {'category': 'MEMORY_OPERATION', 'kind': 'MEMORY_WRITE', 'detail': 'object=buf'},
            {'category': 'POTENTIAL_PATTERN', 'kind': 'UNBOUNDED_WRITE', 'detail': 'object=buf'},
            {'category': 'DATA_DEPENDENCE', 'kind': 'PARAMETER_DEPENDENCE', 'detail': 'parameter=len'},
        ]
        text = render_semantic_items(items)
        self.assertLess(text.index('[POTENTIAL_PATTERN]'), text.index('[MEMORY_OPERATION]'))
        self.assertLess(text.index('[MEMORY_OPERATION]'), text.index('[DATA_DEPENDENCE]'))

    def test_semantic_group_ablation_filters_only_requested_group(self):
        items = [
            {'category': 'MEMORY_OPERATION', 'kind': 'MEMORY_WRITE', 'detail': 'object=buf'},
            {'category': 'DATA_DEPENDENCE', 'kind': 'PARAMETER_DEPENDENCE', 'detail': 'parameter=len'},
            {'category': 'CONTROL_CONSTRAINT', 'kind': 'UPPER_BOUND_RELATED_CONDITION', 'detail': 'expr=len<cap'},
        ]
        text = render_semantic_items(items, excluded_groups=('dependence',))
        self.assertIn('MEMORY_WRITE', text)
        self.assertNotIn('PARAMETER_DEPENDENCE', text)
        self.assertIn('UPPER_BOUND_RELATED_CONDITION', text)
        self.assertEqual(validate_semantic_groups(('Memory', 'memory', 'constraint')), ('memory', 'constraint'))
        with self.assertRaises(ValueError):
            validate_semantic_groups(('unknown',))


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