"""Source coordinates, source-token budget, and alignment fallbacks."""
from __future__ import annotations

import unittest

import torch

from vulnmechanism.cfg_alignment import align_nodes, coverage_summary
from vulnmechanism.model import InputBuilder


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    is_fast = True

    def __call__(self, text, *, add_special_tokens=False, truncation=False,
                 max_length=None, return_offsets_mapping=False):
        ids = [ord(char) % 30 + 2 for char in text]
        if truncation:
            ids = ids[:max_length]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(ids))]
        return result


def location(source, start, code):
    preceding = source[:start]
    line = preceding.count("\n") + 1
    line_start = preceding.rfind("\n") + 1
    return {"code": code, "OFFSET": len(preceding.encode("utf-16-le")) // 2,
            "OFFSET_END": len(source[:start + len(code)].encode("utf-16-le")) // 2,
            "LINE_NUMBER": line,
            "COLUMN_NUMBER": len(source[line_start:start].encode("utf-16-le")) // 2 + 1}


class AlignmentTests(unittest.TestCase):
    def test_utf16_to_character_offsets_with_crlf_and_repeated_text(self):
        source = "// 😀\r\nint café = 1;\r\nint café = 2;"
        first = source.index("café")
        second = source.index("café", first + 1)
        offsets = [(i, i + 1) for i in range(len(source))]
        pairs, counts = align_nodes(source, [location(source, first, "café"),
                                            location(source, second, "café")], offsets, 7)
        self.assertEqual(counts["aligned_nodes"], 2)
        self.assertEqual(pairs[:4], [(0, 7 + first + i) for i in range(4)])
        self.assertEqual(pairs[4:], [(1, 7 + second + i) for i in range(4)])
        self.assertEqual(coverage_summary(counts)["coverage"], 1.0)

    def test_repeated_code_is_not_searched_when_position_is_wrong(self):
        source = "foo bar foo"
        valid = location(source, 8, "foo")
        wrong = location(source, 4, "foo")
        pairs, counts = align_nodes(source, [valid, wrong],
                                    [(i, i + 1) for i in range(len(source))], 3)
        self.assertEqual(pairs, [(0, 11), (0, 12), (0, 13)])
        self.assertEqual(counts["position_mismatch"], 1)

    def test_line_column_fallback_and_conflicting_coordinates(self):
        source = "😀\nvalue"
        correct = location(source, 2, "value")
        line_only = {k: v for k, v in correct.items() if not k.startswith("OFFSET")}
        conflicting = dict(correct, OFFSET=0)
        pairs, counts = align_nodes(source, [line_only, conflicting],
                                    [(i, i + 1) for i in range(len(source))], 2)
        self.assertEqual(pairs, [(0, i) for i in range(4, 9)])
        self.assertEqual(counts["position_mismatch"], 1)

    def test_truncated_or_unreliable_nodes_keep_only_attributes(self):
        source = "abcdefgh"
        locations = [location(source, 1, "bc"), location(source, 3, "de"),
                     {"code": "fg"}]
        pairs, counts = align_nodes(source, locations, [(i, i + 1) for i in range(4)], 5)
        self.assertEqual(pairs, [(0, 6), (0, 7)])
        self.assertEqual(counts["outside_visible_source"], 1)
        self.assertEqual(counts["missing_position"], 1)
        self.assertEqual(counts["aligned_nodes"], 1)
        pairs, counts = align_nodes(source, locations, [(0, 1), (1, 99)], 5)
        self.assertEqual(pairs, [])
        self.assertEqual(counts["invalid_token_offsets"], 3)

    def test_formal_input_builder_matches_baseline_ids_and_excludes_prefix_eos_pad(self):
        builder = InputBuilder(CharacterTokenizer(), source_max_length=4, context_max_length=1)
        rows = [{"raw_source": "abc"}, {"raw_source": "abcdefgh"}]
        baseline_ids, baseline_mask = builder.sequence_batch(
            rows, variant="baseline", excluded_groups=(), device=torch.device("cpu"))
        ids, mask, offsets = builder.source_alignment_batch(rows, device=torch.device("cpu"))
        torch.testing.assert_close(ids, baseline_ids)
        torch.testing.assert_close(mask, baseline_mask)
        self.assertEqual(offsets, [[(0, 1), (1, 2), (2, 3)],
                                   [(0, 1), (1, 2), (2, 3), (3, 4)]])
        pairs, counts = align_nodes("abc", [location("abc", 1, "b")], offsets[0],
                                    len(builder.source_prefix))
        self.assertEqual(pairs, [(0, len(builder.source_prefix) + 1)])
        self.assertEqual(counts["aligned_nodes"], 1)
        self.assertLess(pairs[0][1], int(mask[0].sum()) - 1)  # EOS follows source.


if __name__ == "__main__":
    unittest.main()
