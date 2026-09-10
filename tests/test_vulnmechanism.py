import unittest

from vulnmechanism.cpg import FunctionGraph, GraphEdge, GraphNode, parse_dot_graph
from vulnmechanism.mechanism import _normalize_label, graph_relations
from vulnmechanism.syntax import parse_function


class SyntaxTests(unittest.TestCase):
    def test_parse_function(self):
        parsed = parse_function('int foo(char *buf, int len) { return buf[len]; }', 'c', 'foo')
        self.assertEqual(parsed.name, 'foo')
        self.assertEqual(parsed.parameters, ('buf', 'len'))


class CPGTests(unittest.TestCase):
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
                '2': GraphNode('2', 'memcpy', 'memcpy(dst, src, n)'),
                '3': GraphNode('3', 'IDENTIFIER', 'n'),
            },
            (
                GraphEdge('CDG', '1', '2'),
                GraphEdge('DDG', '3', '2'),
            ),
        )
        relations = graph_relations(graph)
        self.assertEqual({relation.kind for relation in relations}, {'CDG', 'DDG'})
        self.assertTrue(any('memcpy' in relation.as_text() for relation in relations))


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
