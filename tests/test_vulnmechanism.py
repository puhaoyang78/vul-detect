import tempfile
from pathlib import Path
import unittest

from vulnmechanism.cpg import CPGError, FunctionGraph, GraphEdge, GraphNode, _matching_dot, parse_dot_graph
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
    def test_matching_source_method_and_export_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ast = root / "ast"
            ast.mkdir()
            (ast / "0-ast.dot").write_text(
                'digraph "foo" {\n"1" [label = <METHOD<BR/>foo> ]\n"2" [label = <BLOCK<BR/>&lt;empty&gt;> ]\n"1" -> "2"\n}\n'
            )
            definition = ast / "3-ast.dot"
            definition.write_text(
                'digraph "foo" {\n"3" [label = <METHOD, 1<BR/>foo> ]\n"4" [label = <BLOCK, 1<BR/>{}> ]\n"3" -> "4"\n}\n'
            )
            self.assertEqual(_matching_dot(ast, "foo"), definition)
            cdg = root / "cdg"
            cdg.mkdir()
            for index in (0, 3):
                (cdg / f"{index}-cdg.dot").write_text('digraph "foo" {\n}\n')
            self.assertEqual(_matching_dot(cdg, "foo", "3"), cdg / "3-cdg.dot")

    def test_symbolic_operator_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0-ast.dot"
            path.write_text('digraph "[]" {\n"1" [label = <METHOD, 1<BR/>[]> ]\n}\n')
            self.assertEqual(_matching_dot(Path(directory), "operator[]"), path)

    def test_dot_parser_preserves_operator_text(self):
        graph = parse_dot_graph(
            '''digraph "foo" {
"1" [label = <METHOD, 1<BR/>foo> ]
"2" [label = <&lt;operator&gt;.lessThan, 2<BR/>n &lt; cap> ]
"3" [label = <memcpy, 3<BR/>memcpy(dst, src, n)> ]
"1" -> "2"
"2" -> "3"
}''',
            "ast",
        )
        self.assertEqual(graph.nodes["2"].label, "<operator>.lessThan")
        self.assertEqual(graph.nodes["2"].code, "n < cap")
        self.assertEqual(graph.nodes["3"].code, "memcpy(dst, src, n)")

    def test_relation_rendering_preserves_graph_kinds(self):
        nodes = {str(i): GraphNode(str(i), "IDENTIFIER", f"value{i}") for i in range(164)}
        edges = tuple(GraphEdge("AST", "0", str(i)) for i in range(1, 161))
        edges += tuple(GraphEdge(kind, "0", str(i)) for i, kind in enumerate(("CFG", "CDG", "DDG"), 161))
        graph = FunctionGraph("f", nodes, edges)
        rows = render_cpg_relations(graph).splitlines()
        self.assertEqual(len(rows), 160)
        self.assertEqual({row.split("|")[0] for row in rows}, {"AST", "CFG", "CDG", "DDG"})
        self.assertEqual(len(extract_cpg_relations(graph)), len(edges))

    def test_missing_source_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0-ast.dot"
            path.write_text('digraph "other" {\n"1" [label = <METHOD, 1<BR/>other> ]\n}\n')
            with self.assertRaises(CPGError):
                _matching_dot(Path(directory), "foo")


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
