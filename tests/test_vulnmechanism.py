import unittest

from vulnmechanism.cpg import CPGError, FunctionGraph, GraphEdge, GraphNode
from vulnmechanism.dataset import extract_cpg_relations, render_cpg_relations
from vulnmechanism.semantics import (
    VULNERABILITY_FEATURES,
    extract_vulnerability_semantics,
    render_semantic_items,
    validate_semantic_groups,
)
from vulnmechanism.syntax import parse_function, single_function_language


class SyntaxTests(unittest.TestCase):
    def test_parse_c_and_cpp_functions(self):
        self.assertEqual(
            parse_function("int foo(char *buf, int len) { return buf[len]; }", "c", "foo").parameters,
            ("buf", "len"),
        )
        cases = [
            ("Foo::~Foo() {}", "~Foo"),
            ("int Foo::operator[](int index) { return index; }", "operator[]"),
            ("Foo::operator bool() { return true; }", "operator bool"),
        ]
        for source, expected in cases:
            self.assertEqual(parse_function(source, "cpp").name, expected)

    def test_language_inference_is_conservative(self):
        self.assertEqual(single_function_language("int f(void) { return 0; }"), "c")
        self.assertEqual(single_function_language("Foo::Foo() : x(0) {}"), "cpp")
        self.assertEqual(single_function_language("int f() {}", "source.cpp"), "cpp")
        for source in ("", "int f();", "int f() {", "int f() {} int g() {}"):
            self.assertIsNone(single_function_language(source))


class CPGTests(unittest.TestCase):
    def test_relation_rendering_preserves_graph_kinds(self):
        nodes = {str(i): GraphNode(str(i), "IDENTIFIER", f"value{i}") for i in range(164)}
        edges = tuple(GraphEdge("AST", "0", str(i)) for i in range(1, 161))
        edges += tuple(GraphEdge(kind, "0", str(i)) for i, kind in enumerate(("CFG", "CDG", "DDG"), 161))
        graph = FunctionGraph("f", nodes, edges)
        rows = render_cpg_relations(graph).splitlines()
        self.assertEqual(len(rows), 160)
        self.assertEqual({row.split("|")[0] for row in rows}, {"AST", "CFG", "CDG", "DDG"})
        self.assertEqual(len(extract_cpg_relations(graph)), len(edges))



class SemanticTests(unittest.TestCase):
    def test_upper_bound_condition_controls_write(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "METHOD_PARAMETER_IN", "len"),
                "2": GraphNode("2", "<operator>.lessEqualsThan", "len <= 64"),
                "3": GraphNode("3", "CALL", "memcpy(buf, src, len)"),
            },
            (
                GraphEdge("DDG", "1", "3"),
                GraphEdge("CDG", "2", "3"),
                GraphEdge("CFG", "2", "3"),
            ),
        )
        semantics = extract_vulnerability_semantics(graph)
        rendered = semantics.render()
        self.assertIn("MEMORY_WRITE", rendered)
        self.assertIn("UPPER_BOUND_RELATED_CONDITION", rendered)
        self.assertNotIn("WRITE_EXTENT_WITHOUT_UPPER_BOUND_CONDITION", rendered)
        self.assertTrue(set(semantics.feature_names) <= set(VULNERABILITY_FEATURES))

    def test_positive_length_is_not_upper_bound(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "<operator>.greaterThan", "len > 0"),
                "2": GraphNode("2", "CALL", "memcpy(buf, src, len)"),
            },
            (GraphEdge("CDG", "1", "2"),),
        )
        rendered = extract_vulnerability_semantics(graph).render()
        self.assertIn("WRITE_EXTENT_WITHOUT_UPPER_BOUND_CONDITION", rendered)
        self.assertNotIn("UPPER_BOUND_RELATED_CONDITION", rendered)

    def test_null_branch_is_not_nonnull_protection(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "<operator>.equals", "p == NULL"),
                "2": GraphNode("2", "<operator>.indirection", "*p"),
            },
            (GraphEdge("CDG", "1", "2"),),
        )
        rendered = extract_vulnerability_semantics(graph).render()
        self.assertIn("NULL_RELATED_CONDITION", rendered)
        self.assertIn("DEREFERENCE_WITHOUT_NONNULL_CONDITION", rendered)

    def test_free_then_use_is_lifetime_pattern(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "CALL", "free(p)"),
                "2": GraphNode("2", "<operator>.indirection", "*p"),
            },
            (GraphEdge("CFG", "1", "2"),),
        )
        rendered = extract_vulnerability_semantics(graph).render()
        self.assertIn("FREE_THEN_USE", rendered)

    def test_patterns_render_first_and_ablation_is_explicit(self):
        items = [
            {"category": "MEMORY_OPERATION", "kind": "MEMORY_WRITE", "detail": "object=buf"},
            {"category": "POTENTIAL_PATTERN", "kind": "UNBOUNDED_WRITE", "detail": "object=buf"},
            {"category": "DATA_DEPENDENCE", "kind": "PARAMETER_DEPENDENCE", "detail": "parameter=len"},
        ]
        text = render_semantic_items(items)
        self.assertLess(text.index("[POTENTIAL_PATTERN]"), text.index("[MEMORY_OPERATION]"))
        filtered = render_semantic_items(items, excluded_groups=("dependence",))
        self.assertNotIn("PARAMETER_DEPENDENCE", filtered)
        self.assertEqual(validate_semantic_groups(("Memory", "memory", "constraint")), ("memory", "constraint"))
        with self.assertRaises(ValueError):
            validate_semantic_groups(("unknown",))


if __name__ == "__main__":
    unittest.main()
