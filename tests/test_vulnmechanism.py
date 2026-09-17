import unittest

from vulnmechanism.cpg import FunctionGraph, GraphEdge, GraphNode
from vulnmechanism.dataset import extract_cpg_relations, render_cpg_relations
from vulnmechanism.semantics import (
    _is_arithmetic_node,
    _node_operations,
    _parameter_sources,
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
        edges += tuple(
            GraphEdge(kind, "0", str(i))
            for i, kind in enumerate(("CFG", "CDG", "DDG"), 161)
        )
        graph = FunctionGraph("f", nodes, edges)
        rows = render_cpg_relations(graph).splitlines()
        self.assertEqual(len(rows), 160)
        self.assertEqual({row.split("|")[0] for row in rows}, {"AST", "CFG", "CDG", "DDG"})
        self.assertEqual(len(extract_cpg_relations(graph)), len(edges))


class MechanismTests(unittest.TestCase):
    def _candidate_kinds(self, semantics):
        return {
            item.kind for item in semantics.items
            if item.category == "MECHANISM_CANDIDATE"
        }

    def _relations(self, semantics, kind=None):
        return [
            item for item in semantics.items
            if item.category == "MECHANISM_RELATION"
            and (kind is None or item.kind == kind)
        ]

    def test_protected_bounds_relation_is_not_vulnerability_candidate(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "int len"),
                "2": GraphNode("2", "<operator>.lessEqualsThan", "len <= 64"),
                "3": GraphNode("3", "memcpy", "memcpy(buf, src, len)"),
                "4": GraphNode("4", "IDENTIFIER", "len"),
            },
            (
                GraphEdge("AST", "3", "4"),
                GraphEdge("DDG", "1", "4"),
                GraphEdge("CDG", "2", "3"),
                GraphEdge("CFG", "2", "3"),
            ),
        )
        semantics = extract_mechanism_semantics(graph)
        relations = self._relations(semantics, "BOUND_RELATION")
        self.assertTrue(relations)
        self.assertIn("bound_related_condition=present", relations[0].detail)
        self.assertEqual(semantics.candidate_count, 0)
        self.assertIn("NO_CPG_DERIVED_MECHANISM_EVIDENCE", semantics.render())

    def test_unprotected_bounds_relation_becomes_candidate(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "int len"),
                "2": GraphNode("2", "<operator>.greaterThan", "flag > 0"),
                "3": GraphNode("3", "memcpy", "memcpy(buf, src, len)"),
                "4": GraphNode("4", "IDENTIFIER", "len"),
            },
            (
                GraphEdge("AST", "3", "4"),
                GraphEdge("DDG", "1", "4"),
                GraphEdge("CDG", "2", "3"),
            ),
        )
        semantics = extract_mechanism_semantics(graph)
        self.assertIn("BOUNDS_FLOW", self._candidate_kinds(semantics))
        self.assertIn("source=parameter:len", semantics.render())
        self.assertIn("bound_related_condition=not_observed", semantics.render())

    def test_generic_array_access_is_audit_only(self):
        graph = FunctionGraph(
            "foo",
            {"1": GraphNode("1", "<operator>.indexAccess", "a[i]")},
            (),
        )
        semantics = extract_mechanism_semantics(graph)
        self.assertEqual(semantics.candidate_count, 0)
        self.assertTrue(any(
            item.category == "SECURITY_OPERATION" and item.kind == "ARRAY_ACCESS"
            for item in semantics.items
        ))
        self.assertNotIn("ARRAY_ACCESS", semantics.render())

    def test_ast_descendant_inherits_protective_condition_without_candidate(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "int i"),
                "2": GraphNode("2", "<operator>.lessThan", "i < 8"),
                "3": GraphNode("3", "CALL", "x = a[i]"),
                "4": GraphNode("4", "<operator>.indexAccess", "a[i]"),
                "5": GraphNode("5", "IDENTIFIER", "i"),
                "6": GraphNode("6", "LOCAL", "int a[8]"),
            },
            (
                GraphEdge("AST", "3", "4"),
                GraphEdge("AST", "4", "5"),
                GraphEdge("AST", "3", "6"),
                GraphEdge("DDG", "1", "5"),
                GraphEdge("CDG", "2", "3"),
                GraphEdge("CFG", "2", "3"),
            ),
        )
        semantics = extract_mechanism_semantics(graph)
        relations = self._relations(semantics, "BOUND_RELATION")
        self.assertTrue(relations)
        self.assertTrue(any("bound_related_condition=present" in item.detail for item in relations))
        self.assertEqual(semantics.candidate_count, 0)

    def test_missing_cdg_is_unknown_and_not_candidate(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "int len"),
                "2": GraphNode("2", "memcpy", "memcpy(buf, src, len)"),
                "3": GraphNode("3", "IDENTIFIER", "len"),
            },
            (
                GraphEdge("AST", "2", "3"),
                GraphEdge("DDG", "1", "3"),
            ),
        )
        semantics = extract_mechanism_semantics(graph)
        relations = self._relations(semantics, "BOUND_RELATION")
        self.assertTrue(relations)
        self.assertTrue(any("unknown_no_cdg" in item.state for item in relations))
        self.assertEqual(semantics.candidate_count, 0)

    def test_pointer_parameter_alone_is_not_null_mechanism(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "char *p"),
                "2": GraphNode("2", "<operator>.indirection", "*p"),
                "3": GraphNode("3", "IDENTIFIER", "p"),
            },
            (
                GraphEdge("AST", "2", "3"),
                GraphEdge("DDG", "1", "3"),
            ),
        )
        semantics = extract_mechanism_semantics(graph)
        self.assertNotIn("NULL_DEREFERENCE_FLOW", self._candidate_kinds(semantics))

    def test_nullable_allocation_without_cdg_is_audit_relation_only(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "malloc", "p = malloc(n)"),
                "2": GraphNode("2", "<operator>.indirection", "*p"),
                "3": GraphNode("3", "IDENTIFIER", "p"),
            },
            (
                GraphEdge("AST", "2", "3"),
                GraphEdge("DDG", "1", "3"),
            ),
        )
        semantics = extract_mechanism_semantics(graph)
        relations = self._relations(semantics, "NULLABLE_SOURCE_TO_DEREFERENCE")
        self.assertTrue(relations)
        self.assertTrue(any("nullable_allocation:malloc" in item.detail for item in relations))
        self.assertTrue(any("unknown_no_cdg" in item.state for item in relations))
        self.assertEqual(semantics.candidate_count, 0)

    def test_nullable_allocation_without_observed_null_check_is_candidate(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "malloc", "p = malloc(n)"),
                "2": GraphNode("2", "<operator>.greaterThan", "flag > 0"),
                "3": GraphNode("3", "<operator>.indirection", "*p"),
                "4": GraphNode("4", "IDENTIFIER", "p"),
            },
            (
                GraphEdge("AST", "3", "4"),
                GraphEdge("DDG", "1", "4"),
                GraphEdge("CDG", "2", "3"),
            ),
        )
        semantics = extract_mechanism_semantics(graph)
        self.assertIn("NULL_DEREFERENCE_FLOW", self._candidate_kinds(semantics))
        self.assertIn("nullable_allocation:malloc", semantics.render())

    def test_explicit_null_origin_without_cdg_is_audit_relation_only(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "<operator>.assignment", "p = NULL"),
                "2": GraphNode("2", "<operator>.indirection", "*p"),
                "3": GraphNode("3", "IDENTIFIER", "p"),
            },
            (
                GraphEdge("AST", "2", "3"),
                GraphEdge("DDG", "1", "3"),
            ),
        )
        semantics = extract_mechanism_semantics(graph)
        relations = self._relations(semantics, "NULLABLE_SOURCE_TO_DEREFERENCE")
        self.assertTrue(relations)
        self.assertTrue(any("explicit_null_assignment" in item.detail for item in relations))
        self.assertEqual(semantics.candidate_count, 0)

    def test_chained_field_access_keeps_full_pointer_base(self):
        operations = _node_operations(
            GraphNode("1", "<operator>.fieldAccess", "tree->cdr->car->cdr")
        )
        pointers = {
            operation.object_name
            for operation in operations
            if operation.kind == "POINTER_DEREFERENCE"
        }
        self.assertEqual(pointers, {"tree->cdr->car"})

    def test_destination_bound_api_needs_independent_capacity(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "size_t n"),
                "2": GraphNode("2", "snprintf", "snprintf(buf, n, \"%s\", src)"),
                "3": GraphNode("3", "IDENTIFIER", "n"),
            },
            (
                GraphEdge("AST", "2", "3"),
                GraphEdge("DDG", "1", "3"),
            ),
        )
        self.assertNotIn("BOUNDS_FLOW", self._candidate_kinds(extract_mechanism_semantics(graph)))

    def test_sizeof_expression_does_not_inherit_unrelated_parameter_source(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "struct sockaddr_storage *ss"),
                "2": GraphNode("2", "memcpy", "memcpy(ss, src, sizeof(local))"),
                "3": GraphNode("3", "<operator>.sizeOf", "sizeof(local)"),
            },
            (
                GraphEdge("AST", "2", "3"),
                GraphEdge("DDG", "1", "2"),
            ),
        )
        self.assertEqual(_parameter_sources(graph, "2", "sizeof(local)"), ())
        semantics = extract_mechanism_semantics(graph)
        self.assertFalse(self._relations(semantics, "BOUND_RELATION"))

    def test_pointer_declaration_is_not_arithmetic_node(self):
        self.assertFalse(_is_arithmetic_node(GraphNode(
            "1", "<operator>.indirection", "char *buf"
        )))

    def test_expression_local_ddg_does_not_mix_call_arguments(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "char *dst"),
                "2": GraphNode("2", "PARAM", "char *src"),
                "3": GraphNode("3", "PARAM", "size_t n"),
                "4": GraphNode("4", "memcpy", "memcpy(dst, src, n)"),
                "5": GraphNode("5", "IDENTIFIER", "dst"),
                "6": GraphNode("6", "IDENTIFIER", "src"),
                "7": GraphNode("7", "IDENTIFIER", "n"),
            },
            (
                GraphEdge("AST", "4", "5"),
                GraphEdge("AST", "4", "6"),
                GraphEdge("AST", "4", "7"),
                GraphEdge("DDG", "1", "5"),
                GraphEdge("DDG", "2", "6"),
                GraphEdge("DDG", "3", "7"),
            ),
        )
        self.assertEqual(_parameter_sources(graph, "4", "n"), ("n",))

    def test_control_arithmetic_without_operand_guard_is_candidate(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "long data_size"),
                "2": GraphNode("2", "PARAM", "long header_size"),
                "3": GraphNode("3", "<operator>.lessThan", "i < data_size - header_size"),
                "4": GraphNode("4", "<operator>.indexAccess", "data[i]"),
                "5": GraphNode("5", "IDENTIFIER", "i"),
            },
            (
                GraphEdge("AST", "4", "5"),
                GraphEdge("CDG", "3", "4"),
            ),
        )
        semantics = extract_mechanism_semantics(graph)
        self.assertIn("SIZE_ARITHMETIC_FLOW", self._candidate_kinds(semantics))
        self.assertIn("data_size - header_size", semantics.render())
        self.assertIn("range_related_condition=not_observed", semantics.render())

    def test_control_arithmetic_with_operand_guard_is_not_candidate(self):
        graph = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "PARAM", "long data_size"),
                "2": GraphNode("2", "PARAM", "long header_size"),
                "3": GraphNode("3", "<operator>.lessThan", "data_size < header_size"),
                "4": GraphNode("4", "<operator>.lessThan", "i < data_size - header_size"),
                "5": GraphNode("5", "<operator>.indexAccess", "data[i]"),
                "6": GraphNode("6", "IDENTIFIER", "i"),
            },
            (
                GraphEdge("AST", "5", "6"),
                GraphEdge("CDG", "4", "5"),
            ),
        )
        semantics = extract_mechanism_semantics(graph)
        self.assertNotIn("SIZE_ARITHMETIC_FLOW", self._candidate_kinds(semantics))
        relations = self._relations(semantics, "ARITHMETIC_CONTROL_TO_MEMORY_SINK")
        self.assertTrue(relations)
        self.assertTrue(any("operand_range_condition=present" in item.detail for item in relations))

    def test_lifetime_path_stops_at_redefinition(self):
        unsafe = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "free", "free(p)"),
                "2": GraphNode("2", "<operator>.indirection", "*p"),
            },
            (GraphEdge("CFG", "1", "2"),),
        )
        self.assertIn("USE_AFTER_FREE_FLOW", self._candidate_kinds(extract_mechanism_semantics(unsafe)))

        redefined = FunctionGraph(
            "foo",
            {
                "1": GraphNode("1", "free", "free(p)"),
                "2": GraphNode("2", "<operator>.assignment", "p = q"),
                "3": GraphNode("3", "<operator>.indirection", "*p"),
            },
            (GraphEdge("CFG", "1", "2"), GraphEdge("CFG", "2", "3")),
        )
        self.assertNotIn("USE_AFTER_FREE_FLOW", self._candidate_kinds(extract_mechanism_semantics(redefined)))

    def test_renderer_excludes_audit_operations_and_unlinked_relations(self):
        items = [
            {"category": "SECURITY_OPERATION", "kind": "MEMORY_WRITE", "detail": "code=memcpy(buf,src,n)"},
            {"category": "MECHANISM_RELATION", "kind": "BOUND_RELATION", "detail": "source=parameter:n", "key": "bounds-1"},
            {"category": "MECHANISM_CANDIDATE", "kind": "BOUNDS_FLOW", "detail": "source=parameter:n sink=MEMORY_WRITE object=buf", "key": "bounds-1"},
            {"category": "MECHANISM_RELATION", "kind": "UNLINKED", "detail": "should_not_render", "key": "missing"},
        ]
        text = render_mechanism_items(items)
        self.assertIn("[MECHANISM_CANDIDATE]", text)
        self.assertIn("[MECHANISM_RELATION]", text)
        self.assertNotIn("SECURITY_OPERATION", text)
        self.assertNotIn("memcpy(buf,src,n)", text)
        self.assertNotIn("should_not_render", text)
        self.assertEqual(
            validate_mechanism_groups(("Mechanism", "mechanism", "relation")),
            ("mechanism", "relation"),
        )
        with self.assertRaises(ValueError):
            validate_mechanism_groups(("operation",))


if __name__ == "__main__":
    unittest.main()
