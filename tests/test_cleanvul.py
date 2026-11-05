"""CleanVul release members, supervision and shared three-seed training routes."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_losver import Tokens
from vulnmechanism import cleanvul
from vulnmechanism.cli import build_parser, main
from vulnmechanism.data import read_jsonl


def entry(i, score=4, **updates):
    row = dict(func_before=f"int f{i}() {{\n int x = {i};\n return x;\n}}\n",
        func_after=f"int f{i}() {{\n int x = {i};\n return x + 1;\n}}\n",
        commit_msg="fix bounds", commit_url=f"https://github.com/p/r/commit/{i+1:040x}",
        cve_id="", cwe_id="", file_name="a.c", vulnerability_score=str(score),
        extension="c", is_test="False", date="")
    row.update(updates)
    return row


def raw_files(root, rows):
    for score in range(5):
        with (Path(root)/f"vulnerability_score_{score}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entry(0)))
            writer.writeheader()
            writer.writerows(r for r in rows if int(r["vulnerability_score"]) == score)


class CleanVulTests(unittest.TestCase):
    def test_threshold_retains_parser_name_and_lexical_failures_as_classification_members(self):
        with tempfile.TemporaryDirectory() as root:
            source = entry(0, 3, commit_msg="fix CVE-2024-12345")
            raw_files(root, [source, entry(1, 2), entry(2, func_after="int other() {return 0;}"),
                entry(3, func_after=entry(3)["func_before"]+"// explanation\n"),
                entry(4, func_after="int f4() {")])
            rows, counterparts, stats, audit = cleanvul.load_pairs(root)
        self.assertEqual(len(rows), 8)
        self.assertEqual(stats["high_confidence_cc_pairs"], 4)
        self.assertEqual(stats["diagnostic_reasons"], {"retained": 1, "function_name_mismatch": 1,
            "comments_or_formatting_only": 1, "parse_incomplete_or_not_single_function": 1})
        self.assertEqual(stats["classification_members_removed"], 0)
        self.assertTrue(all(r["included_for_classification"] for r in audit))
        self.assertEqual(rows[0]["func"], source["func_before"])
        self.assertEqual(rows[1]["func"], source["func_after"])
        self.assertEqual([r["target"] for r in rows], [1, 0]*4)
        self.assertEqual(rows[0]["cves"], ["CVE-2024-12345"])
        self.assertEqual(rows[0]["cve_source"], "commit_message")
        self.assertEqual(counterparts[rows[0]["sample_key"]], rows[1]["func"])
        self.assertEqual(rows[0]["repair_key"], rows[1]["sample_key"])

    def test_exact_repeated_pairs_and_ambiguous_edits_are_retained(self):
        with tempfile.TemporaryDirectory() as root:
            raw_files(root, [entry(0), entry(0), entry(1),
                            entry(1, func_after=entry(1)["func_after"].replace("+ 1", "+ 2"))])
            rows, _, stats, _ = cleanvul.load_pairs(root)
        self.assertEqual(len(rows), 8)
        self.assertEqual(stats["diagnostic_reasons"], {"retained": 2, "ambiguous_same_commit_file_function": 2})
        self.assertEqual(len({r["sample_key"] for r in rows}), 8)

    def test_empty_source_or_changed_release_identity_fails_without_silent_removal(self):
        for updates, message in ((dict(func_after=""), "empty released source"),
                                 (dict(commit_url="https://github.com/p/r/commit/not-a-sha"), "release metadata")):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as root:
                raw_files(root, [entry(0, **updates)])
                with self.assertRaisesRegex(ValueError, message):
                    cleanvul.load_pairs(root)

    def test_region_coverage_obeys_visibility_background_and_reliability(self):
        def record(code, truth, reliable=True):
            row = dict(func=code, target=1, sample_key="x", pair_key="p",
                       lines_ground_truth=truth, diff_reliable=reliable)
            return cleanvul.region_record(row, Tokens(), "cleanvul", "train")
        code = "int f() {\n int x = 1;\n return x;\n}"
        self.assertTrue(record(code, [2])["effective_region"])
        self.assertEqual(record(code, [])["reason"], "no_changed_source_lines_on_this_side")
        self.assertFalse(record(code, [0, 1, 2, 3])["effective_region"])
        self.assertEqual(record(code, [2], False)["reason"], "unreliable_or_missing_counterpart")
        long_line = "int f() {\n" + "x " * 600 + "\nreturn x;\n}"
        self.assertEqual(record(long_line, [1])["reason"], "partially_visible_changed_line")
        self.assertEqual(record(long_line, [2])["reason"], "changed_lines_outside_visible_input")

    def test_preparation_is_balanced_grouped_deterministic_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/"raw"
            raw.mkdir()
            raw_files(raw, [entry(i) for i in range(30)])
            original = {p.name: p.read_bytes() for p in raw.iterdir()}
            with patch("transformers.RobertaTokenizerFast.from_pretrained", return_value=Tokens()):
                a = cleanvul.prepare(raw, Path(tmp)/"one", "unused")
                b = cleanvul.prepare(raw, Path(tmp)/"two", "unused")
                with self.assertRaises(FileExistsError):
                    cleanvul.prepare(raw, Path(tmp)/"one", "unused")
            self.assertEqual(a, b)
            groups = {}
            for split in ("train", "valid", "test"):
                path = Path(tmp)/"one/rgcl/cleanvul"/f"{split}.lines.jsonl"
                rows = read_jsonl(path)
                self.assertTrue(rows)
                self.assertEqual(sum(r["target"] for r in rows), len(rows)//2)
                for row in rows:
                    groups.setdefault(row["pair_key"], set()).add(split)
            self.assertTrue(all(len(s) == 1 for s in groups.values()))
            self.assertFalse(any(a["cross_split_overlap_groups"].values()))
            self.assertEqual(original, {p.name: p.read_bytes() for p in raw.iterdir()})

    def test_cli_uses_shared_training_without_changing_historical_protocol(self):
        args = build_parser().parse_args(["rgcl-study", "inspect", "--dataset", "cleanvul",
                                         "--raw-root", "raw", "--seed", "42"])
        self.assertEqual(args.seed, 42)
        with patch("vulnmechanism.rgcl_study.run", return_value={}) as shared:
            with patch("sys.argv", ["vulnmechanism", "rgcl-study", "run", "--dataset", "cleanvul",
                                    "--seed", "42", "--methods", "rgcl"]):
                main()
            self.assertEqual(shared.call_args.kwargs["dataset"], "cleanvul")
            self.assertEqual(shared.call_args.kwargs["methods"], ["rgcl"])
        with patch("sys.argv", ["vulnmechanism", "rgcl-study", "run", "--dataset", "cleanvul",
                                "--seed", "42"]), patch("vulnmechanism.rgcl_study.run", return_value={}):
            main()

    def test_inspection_cannot_overwrite_historical_report(self):
        with tempfile.TemporaryDirectory() as root:
            report = Path(root)/"report.md"
            report.write_text("historical result")
            with self.assertRaises(FileExistsError):
                cleanvul.inspect("unused", "unused", root, "unused")
            self.assertEqual(report.read_text(), "historical result")

    def test_short_commit_references_are_retained_and_grouped_with_full_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/"raw"
            raw.mkdir()
            rows = [entry(i) for i in range(30)]
            rows[0]["commit_url"] = "https://github.com/p/r/commit/abc1234"
            rows[1]["commit_url"] = "https://github.com/p/r/commit/abc1234" + "0"*33
            raw_files(raw, rows)
            with patch("transformers.RobertaTokenizerFast.from_pretrained", return_value=Tokens()):
                stats = cleanvul.prepare(raw, Path(tmp)/"prepared", "unused")
            self.assertEqual(stats["retained_pairs"], 30)
            self.assertEqual(stats["commit_reference_kinds"]["abbreviated"], 1)
            members = [r for split in ("train", "valid", "test") for r in
                       read_jsonl(Path(tmp)/"prepared/rgcl/cleanvul"/f"{split}.jsonl")]
            related = [r for r in members if r["func_name"] in {"f0", "f1"}]
            self.assertEqual(len({r["split"] for r in related}), 1)
            self.assertEqual(len({r["commit_group"] for r in related}), 1)
            self.assertIn("abc1234", {r["commit_id"] for r in related})

    def test_counterpart_only_denominators_exclude_ordinary_and_unmatched_members(self):
        records = []
        for i, (target, kind, reliable) in enumerate(((1, "vulnerable", True), (0, "repaired", True),
                                                     (1, "vulnerable", False), (0, "ordinary", False))):
            row = dict(func=entry(0)["func_before"], target=target, normal_kind=kind,
                       sample_key=str(i), pair_key=str(i//2), lines_ground_truth=[2], diff_reliable=reliable)
            records.append(cleanvul.region_record(row, Tokens(), "secvuleval", "train"))
        c = cleanvul.coverage_table(records)["all"]
        self.assertEqual(c["all"]["functions"], 4)
        self.assertEqual(c["1"]["functions"], 2)
        self.assertEqual(c["0"]["functions"], 2)
        self.assertEqual(c["paired"]["functions"], 2)
        self.assertEqual(c["paired_before"]["functions"], 1)
        self.assertEqual(c["repaired"]["functions"], 1)
