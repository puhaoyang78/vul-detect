import unittest

from semantic_demo.candidate_graph import _target_can_produce_summary
from semantic_demo.symbol_resolution import ResolvedTarget, _macro_source, _source_method


class SourceMacroCandidateTests(unittest.TestCase):
    def _target(self, body: str):
        source = _macro_source(
            "debug.h",
            body,
            "MACRO",
            1,
            1,
            ("dst", "src", "n"),
            body,
            "c",
        )
        self.assertIsNotNone(source)
        return ResolvedTarget(
            _source_method(source, "source-macro"), source, "source-macro"
        )

    def test_inert_statement_macro_is_not_candidate(self):
        self.assertFalse(_target_can_produce_summary(self._target("do { } while (0)")))

    def test_standard_memory_effect_macro_remains_candidate(self):
        self.assertTrue(
            _target_can_produce_summary(self._target("memcpy(dst, src, n)"))
        )


if __name__ == "__main__":
    unittest.main()
