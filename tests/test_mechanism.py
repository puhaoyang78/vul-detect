import unittest

from semantic_demo.function_cpg import FunctionGraph, GraphEdge, GraphNode, parse_dot_graph
from semantic_demo.mechanism import (
    MECHANISM_COMPONENTS,
    canonicalize_source,
    derive_mechanism,
    rename_local_identifiers,
)


class FunctionCPGTests(unittest.TestCase):
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


class CanonicalizationTests(unittest.TestCase):
    def test_local_identifier_rename_preserves_canonical_source(self):
        source = 'int foo(char *buf, int len) { int i = len; return buf[i]; }'
        renamed = rename_local_identifiers(source, 'c', 'foo')
        self.assertNotIn('foo', renamed)
        self.assertNotIn('buf', renamed)
        self.assertEqual(
            canonicalize_source(source, 'c', 'foo'),
            canonicalize_source(renamed, 'c', 'renamed_func'),
        )


class MechanismTests(unittest.TestCase):
    def test_bound_change_is_grounded_as_mechanism(self):
        vulnerable = '''
size_t foo(const char *userp, const char *passwdp) {
  size_t ulen = strlen(userp);
  size_t plen = strlen(passwdp);
  if ((ulen > SIZE_T_MAX/2) || (plen > (SIZE_T_MAX/2 - 2))) return 0;
  size_t plainlen = 2 * ulen + plen + 2;
  char *plainauth = malloc(plainlen);
  memcpy(plainauth, userp, ulen);
  return plainlen;
}
'''
        fixed = vulnerable.replace('ulen > SIZE_T_MAX/2', 'ulen > SIZE_T_MAX/4')
        before = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'CONTROL_STRUCTURE', 'if ((ulen > SIZE_T_MAX/2) || (plen > (SIZE_T_MAX/2 - 2)))'),
                '2': GraphNode('2', '<operator>.multiplication', '2 * ulen + plen + 2'),
                '3': GraphNode('3', 'malloc', 'malloc(plainlen)'),
                '4': GraphNode('4', 'memcpy', 'memcpy(plainauth, userp, ulen)'),
            },
            (
                GraphEdge('CDG', '1', '2'),
                GraphEdge('DDG', '2', '3'),
                GraphEdge('DDG', '2', '4'),
            ),
        )
        after = FunctionGraph(
            'foo',
            {
                '1': GraphNode('1', 'CONTROL_STRUCTURE', 'if ((ulen > SIZE_T_MAX/4) || (plen > (SIZE_T_MAX/2 - 2)))'),
                '2': GraphNode('2', '<operator>.multiplication', '2 * ulen + plen + 2'),
                '3': GraphNode('3', 'malloc', 'malloc(plainlen)'),
                '4': GraphNode('4', 'memcpy', 'memcpy(plainauth, userp, ulen)'),
            },
            before.edges,
        )
        mechanism = derive_mechanism(
            vulnerable,
            fixed,
            before,
            after,
            language='c',
            function_name='foo',
        )
        self.assertEqual(mechanism.security_effect, 'bound_changed')
        self.assertIn('bounds', mechanism.components)
        self.assertIn('guard', mechanism.components)
        self.assertTrue(mechanism.changed_relations)
        self.assertTrue(all(component in MECHANISM_COMPONENTS for component in mechanism.components))


if __name__ == '__main__':
    unittest.main()
