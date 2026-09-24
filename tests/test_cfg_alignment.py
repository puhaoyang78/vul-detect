"""Source coordinates, source-token budget, and alignment fallbacks."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from vulnmechanism import cfg_data as data
from vulnmechanism.cfg_alignment import align_nodes, coverage_summary
from vulnmechanism.cfg_alignment_audit import audit_alignment
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

    def test_position_mismatch_subtypes_preserve_alignment_decisions(self):
        source = "abcd"
        valid = location(source, 1, "bc")
        nodes = [dict(valid, OFFSET=-1), dict(valid, COLUMN_NUMBER=1),
                 dict(valid, code="zz"), valid]
        statuses = []
        pairs, counts = align_nodes(source, nodes, [(i, i + 1) for i in range(4)],
                                    2, statuses=statuses)
        self.assertEqual(counts["position_mismatch"], 3)
        self.assertEqual(counts["aligned_nodes"], 1)
        self.assertEqual([detail for _, detail in statuses],
                         ["coordinate_invalid", "coordinate_conflict", "text_mismatch", None])
        self.assertEqual(pairs, [(3, 3), (3, 4)])

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


class AuditTests(unittest.TestCase):
    def test_cached_graph_audit_reports_kind_name_attribute_state_and_examples(self):
        source = "x=1; if(x<2) sink();"
        def node(key, kind, code, start=None, name=None, **changes):
            props = {"kind": kind}
            if start is not None:
                props.update(location(source, start, code))
                props.pop("code")
            if name is not None:
                props["NAME"] = name
            props.update(changes)
            return {"id": key, "label": kind, "code": code, "properties": props}
        assignment = source.index("x=1")
        compare = source.index("x<2")
        sink = source.index("sink()")
        nodes = [node("0", "METHOD", "f"),
                 node("1", "CALL", "x=1", assignment, "<operator>.assignment"),
                 node("2", "IDENTIFIER", "x", assignment, TYPE_FULL_NAME="int", ARGUMENT_INDEX=1),
                 node("3", "LITERAL", "1", assignment + 2),
                 node("4", "CALL", "x<2", compare, "<operator>.lessThan"),
                 node("5", "CALL", "sink()", sink, "sink"),
                 node("6", "CALL", "fake()", sink, "synthetic"),
                 node("7", "CALL", "x<2", compare, "<operator>.lessThan", COLUMN_NUMBER=1),
                 node("8", "CALL", "x=1", assignment, "synthetic_invalid", OFFSET=-5)]
        edges = [{"kind": "AST", "source": "0", "target": str(i)} for i in range(1, 9)]
        edges += [{"kind": "AST", "source": "1", "target": str(i)} for i in (2, 3)]
        edges += [{"kind": "CFG", "source": str(a), "target": str(b)}
                  for a, b in ((0, 1), (1, 4), (4, 5), (5, 6), (6, 7), (7, 8))]
        record = {"schema_version": 9, "sample_key": "primevul:tiny", "dataset": "primevul",
                  "split": "train", "label": 0, "raw_source": source}
        graph = {"nodes": nodes, "edges": edges}
        exported = dict(data.identity(record), graph_schema_version=data.GRAPH_SCHEMA,
                        preprocessing_version=data.JOERN_SOURCE_PREPROCESSING_VERSION,
                        preprocessing_applied=False,
                        original_source_sha256=data.source_hash(source),
                        parsed_source_sha256=data.source_hash(source), graph=graph)
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path, graph_path = Path(tmp) / "source.jsonl", Path(tmp) / "graphs.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n")
            graph_path.write_text(json.dumps(exported) + "\n")
            source_before, graph_before = dataset_path.read_bytes(), graph_path.read_bytes()
            report = audit_alignment(str(dataset_path), str(graph_path), CharacterTokenizer(),
                                     source_max_length=128)
            self.assertEqual(report["overall"]["mismatch_types"],
                             {"coordinate_invalid": 1, "coordinate_conflict": 1, "text_mismatch": 1})
            self.assertEqual(report["overall"]["aligned_nodes"], 3)
            self.assertTrue(all(row["total_nodes"] > 0 for row in report["by_node_kind"]))
            self.assertEqual(sum(row["total_nodes"] for row in report["by_node_kind"]),
                             report["overall"]["total_nodes"])
            calls = {(row["operation_name"], row["attributes_all_empty"]): row
                     for row in report["by_call_name"]}
            self.assertEqual(calls[("<operator>.assignment", False)]["aligned_nodes"], 1)
            self.assertEqual(calls[("<operator>.lessThan", True)]["aligned_nodes"], 1)
            self.assertEqual(calls[("sink", True)]["aligned_nodes"], 1)
            examples = report["mismatch_examples"]
            self.assertEqual({kind: len(items) for kind, items in examples.items()},
                             {"coordinate_invalid": 1, "coordinate_conflict": 1, "text_mismatch": 1})
            mismatch = examples["text_mismatch"][0]
            self.assertEqual(mismatch["node_code"]["text"], "fake()")
            self.assertEqual(mismatch["offset_slice"]["text"], "sink()")
            self.assertEqual(dataset_path.read_bytes(), source_before)
            self.assertEqual(graph_path.read_bytes(), graph_before)


if __name__ == "__main__":
    unittest.main()
