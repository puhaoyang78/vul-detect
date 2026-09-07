import unittest

from semantic_demo.analyzer import analyze
from semantic_demo.semantics import Validation
from semantic_demo.source import parse_functions


class DependencyLocalReasoningTests(unittest.TestCase):
    def test_unrelated_indirect_call_does_not_block_safe_access_fact(self):
        function = parse_functions(
            "local.c",
            """
            void f(void (*cb)(int), int x)
            {
                char buf[8];
                cb(x);
                memset(buf, 0, 4);
            }
            """,
        )[0]
        result = analyze(function)
        accesses = result.constraint_result["accesses"]
        self.assertTrue(accesses)
        self.assertEqual("SAFE", accesses[0]["status"])
        self.assertNotIn("unresolved call", accesses[0]["reason"])

    def test_related_indirect_call_blocks_only_dependent_access(self):
        function = parse_functions(
            "local.c",
            """
            void f(void (*cb)(char *))
            {
                char buf[8];
                cb(buf);
                memset(buf, 0, 4);
            }
            """,
        )[0]
        result = analyze(function)
        accesses = result.constraint_result["accesses"]
        self.assertTrue(accesses)
        self.assertEqual("UNKNOWN", accesses[0]["status"])
        self.assertIn("unresolved call", accesses[0]["reason"])

    def test_fixed_width_guarded_arithmetic_is_not_rejected_mechanically(self):
        function = parse_functions(
            "bounded.c",
            """
            void f(uint32_t n)
            {
                char buf[16];
                if (n < 16) {
                    memset(buf, 0, n + 1);
                }
            }
            """,
        )[0]
        result = analyze(function)
        accesses = result.constraint_result["accesses"]
        self.assertTrue(accesses)
        self.assertEqual("SAFE", accesses[0]["status"])
        self.assertNotIn("overflow semantics", accesses[0]["reason"])

    def test_c_unsigned_hex_literal_is_encoded(self):
        function = parse_functions(
            "literal.c",
            """
            void f(uint32_t n)
            {
                char buf[65536];
                if (n <= 0xFFFFU) {
                    memset(buf, 0, n);
                }
            }
            """,
        )[0]
        result = analyze(function)
        accesses = result.constraint_result["accesses"]
        self.assertTrue(accesses)
        self.assertEqual("SAFE", accesses[0]["status"])
        self.assertNotIn("unsupported path constraint", accesses[0]["reason"])

    def test_direct_allocator_establishes_capacity(self):
        function = parse_functions(
            "alloc.c",
            """
            void f(void)
            {
                char *p = malloc(8);
                memset(p, 0, 9);
            }
            """,
        )[0]
        result = analyze(function)
        self.assertEqual("VULNERABLE", result.verdict)
        accesses = result.constraint_result["accesses"]
        self.assertTrue(any(item["status"] == "POTENTIAL_VIOLATION" for item in accesses))

    def test_parameter_capacity_summary_is_applied_at_exact_callsite(self):
        entry = parse_functions(
            "entry.c",
            """
            struct buffer { char *data; };
            void f(struct buffer *buf)
            {
                init_buffer(buf, 8);
                memset(buf->data, 0, 9);
            }
            """,
        )[0]
        call = next(item for item in entry.calls() if item.name == "init_buffer")
        validation = Validation(
            sample_key="S01",
            function="init_buffer",
            source_path="buffer.c",
            source_line=1,
            summary={"kind": "ALLOC", "buffer": "arg0->data", "size": "arg1"},
            passed=True,
            reason="test fixture",
            call_lines=(call.line,),
            status="VERIFIED",
        )
        result = analyze(entry, [validation])
        self.assertEqual("VULNERABLE", result.verdict)

    def test_same_name_summary_is_not_applied_to_other_callsite(self):
        entry = parse_functions(
            "entry.c",
            """
            void f(char *a, char *b)
            {
                helper(a, 8);
                helper(b, 64);
                memset(b, 0, 16);
            }
            """,
        )[0]
        calls = [item for item in entry.calls() if item.name == "helper"]
        validation = Validation(
            sample_key="S01",
            function="helper",
            source_path="a.c",
            source_line=1,
            summary={"kind": "ALLOC", "buffer": "arg0", "size": "arg1"},
            passed=True,
            reason="test fixture",
            call_lines=(calls[0].line,),
            status="VERIFIED",
        )
        result = analyze(entry, [validation])
        op_lines = [
            item.line
            for item in result.operations
            if item.kind == "ALLOC" and item.callee == "helper"
        ]
        self.assertEqual([calls[0].line], op_lines)


if __name__ == "__main__":
    unittest.main()
