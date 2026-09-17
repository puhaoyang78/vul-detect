import unittest

from vulnmechanism.cpg import FunctionGraph, GraphEdge, GraphNode
from vulnmechanism.dataset import extract_cpg_relations, render_cpg_relations
from vulnmechanism.semantics import (
    extract_mechanism_semantics,
    render_mechanism_items,
    validate_mechanism_groups,
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


class MechanismTests(unittest.TestCase):
    def test_bounds_flow_requires_relation_evidence(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "int len"),
                "2": GraphNode("2", "<operator>.lessEqualsThan", "len <= 64"),
                "3": GraphNode("3", "memcpy", "memcpy(buf, src, len)"),
            },
            (
                GraphEdge("DDG", "1", "3"),
                GraphEdge("CDG", "2", "3"),
                GraphEdge("CFG", "2", "3"),
            ),
        )
        semantics = extract_mechanism_semantics(graph)
        rendered = semantics.render()
        self.assertIn("BOUNDS_FLOW", rendered)
        self.assertIn("source=parameter:len", rendered)
        self.assertIn("bound_related_condition=present", rendered)
        self.assertEqual(semantics.candidate_count, 1)

    def test_generic_array_access_is_not_a_mechanism(self):
        graph = FunctionGraph(
            "foo",
            {"1": GraphNode("1", "<operator>.indexAccess", "a[i]")},
            (),
        )
        semantics = extract_mechanism_semantics(graph)
        self.assertEqual(semantics.candidate_count, 0)
        self.assertIn("ARRAY_ACCESS", semantics.render())

    def test_ast_descendant_inherits_related_control_condition(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "int i"),
                "2": GraphNode("2", "<operator>.lessThan", "i < 8"),
                "3": GraphNode("3", "CALL", "x = a[i]"),
                "4": GraphNode("4", "<operator>.indexAccess", "a[i]"),
                "5": GraphNode("5", "LOCAL", "int a[8]"),
            },
            (
                GraphEdge("DDG", "1", "4"),
                GraphEdge("CDG", "2", "3"),
                GraphEdge("AST", "3", "4"),
                GraphEdge("AST", "3", "5"),
                GraphEdge("CFG", "2", "3"),
            ),
        )
        rendered = extract_mechanism_semantics(graph).render()
        self.assertIn("BOUNDS_FLOW", rendered)
        self.assertIn("bound_related_condition=present", rendered)

    def test_null_flow_needs_data_source(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "char *p"),
                "2": GraphNode("2", "<operator>.notEquals", "p != NULL"),
                "3": GraphNode("3", "<operator>.indirection", "*p"),
            },
            (
                GraphEdge("DDG", "1", "3"),
                GraphEdge("CDG", "2", "3"),
                GraphEdge("CFG", "2", "3"),
            ),
        )
        rendered = extract_mechanism_semantics(graph).render()
        self.assertIn("NULL_DEREFERENCE_FLOW", rendered)
        self.assertIn("null_related_condition=present", rendered)

    def test_lifetime_path_stops_at_redefinition(self):
        unsafe = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "free", "free(p)"),
                "2": GraphNode("2", "<operator>.indirection", "*p"),
            },
            (GraphEdge("CFG", "1", "2"),),
        )
        self.assertIn("USE_AFTER_FREE_FLOW", extract_mechanism_semantics(unsafe).render())

        redefined = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "free", "free(p)"),
                "2": GraphNode("2", "<operator>.assignment", "p = q"),
                "3": GraphNode("3", "<operator>.indirection", "*p"),
            },
            (GraphEdge("CFG", "1", "2"), GraphEdge("CFG", "2", "3")),
        )
        self.assertNotIn("USE_AFTER_FREE_FLOW", extract_mechanism_semantics(redefined).render())

    def test_rendering_and_ablation_are_mechanism_specific(self):
        items = [
            {"category": "SECURITY_OPERATION", "kind": "MEMORY_WRITE", "detail": "object=buf"},
            {"category": "MECHANISM_RELATION", "kind": "BOUND_RELATION", "detail": "source=parameter:n"},
            {"category": "MECHANISM_CANDIDATE", "kind": "BOUNDS_FLOW", "detail": "source=parameter:n"},
        ]
        text = render_mechanism_items(items)
        self.assertLess(text.index("[MECHANISM_CANDIDATE]"), text.index("[SECURITY_OPERATION]"))
        filtered = render_mechanism_items(items, excluded_groups=("operation",))
        self.assertNotIn("MEMORY_WRITE", filtered)
        self.assertEqual(
            validate_mechanism_groups(("Mechanism", "mechanism", "relation")),
            ("mechanism", "relation"),
        )
        with self.assertRaises(ValueError):
            validate_mechanism_groups(("unknown",))


if __name__ == "__main__":
    unittest.main()
