import unittest

from semantic_demo.analyzer import analyze
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


if __name__ == "__main__":
    unittest.main()
