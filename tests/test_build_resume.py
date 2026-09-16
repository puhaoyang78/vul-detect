import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from vulnmechanism.benchmark_view import record_split
from vulnmechanism.cpg import CPGError, FunctionGraph, GraphEdge, GraphNode
from vulnmechanism.dataset import build_function_dataset


class BuildResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.samples = Path(self.directory.name) / "samples.jsonl"
        self.output = Path(self.directory.name) / "graphs.jsonl"
        self.rows = [
            {
                "sample_key": f"primevul:{i}",
                "dataset": "primevul",
                "function": f"void f{i}(void) {{ free(0); }}",
                "label": i % 2,
                "language": "c",
                "split": "train",
            }
            for i in range(3)
        ]
        self._write_rows()
        self.graph = FunctionGraph(
            "f",
            {
                "1": GraphNode("1", "METHOD", "f"),
                "2": GraphNode("2", "free", "free(0)"),
                "3": GraphNode("3", "RETURN", "return"),
            },
            (
                GraphEdge("AST", "1", "2"),
                GraphEdge("CFG", "2", "3"),
            ),
        )

    def _write_rows(self):
        self.samples.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def test_failure_preserves_successes_and_resume_skips_them(self):
        with patch(
            "vulnmechanism.dataset.extract_function_cpg",
            side_effect=[self.graph, CPGError("failure"), self.graph],
        ):
            result = build_function_dataset(self.samples, self.output)
        self.assertEqual([row["sample_key"] for row in result], ["primevul:0", "primevul:2"])
        errors = self.output.with_suffix(".errors.jsonl")
        self.assertEqual(json.loads(errors.read_text())["sample_key"], "primevul:1")
        with patch("vulnmechanism.dataset.extract_function_cpg", return_value=self.graph) as extract:
            records = build_function_dataset(self.samples, self.output)
            self.assertEqual([call.args[1] for call in extract.call_args_list], ["f1"])
        self.assertEqual({row["sample_key"] for row in records}, {"primevul:0", "primevul:1", "primevul:2"})
        self.assertEqual(errors.read_text(), "")
        before = self.output.read_bytes()
        with patch("vulnmechanism.dataset.extract_function_cpg") as extract:
            self.assertEqual(build_function_dataset(self.samples, self.output), records)
            extract.assert_not_called()
        self.assertEqual(self.output.read_bytes(), before)

    def test_interruption_and_partial_last_line_resume(self):
        with patch(
            "vulnmechanism.dataset.extract_function_cpg",
            side_effect=[self.graph, KeyboardInterrupt],
        ):
            with self.assertRaises(KeyboardInterrupt):
                build_function_dataset(self.samples, self.output)
        with self.output.open("ab") as handle:
            handle.write(b'{"sample_key": "primevul:1", "raw_source": "\xe4')
        with patch("vulnmechanism.dataset.extract_function_cpg", return_value=self.graph) as extract:
            records = build_function_dataset(self.samples, self.output)
            self.assertEqual([call.args[1] for call in extract.call_args_list], ["f1", "f2"])
        self.assertEqual([row["sample_key"] for row in records], ["primevul:0", "primevul:1", "primevul:2"])

    def test_changed_input_and_corrupt_complete_line_are_rejected(self):
        with patch("vulnmechanism.dataset.extract_function_cpg", return_value=self.graph):
            build_function_dataset(self.samples, self.output)
        before = self.output.read_bytes()
        self.rows[0]["label"] = 1
        self._write_rows()
        with self.assertRaisesRegex(ValueError, "does not match input"):
            build_function_dataset(self.samples, self.output)
        self.assertEqual(self.output.read_bytes(), before)
        self.output.write_bytes(b"broken\n")
        with self.assertRaisesRegex(ValueError, "invalid saved JSON"):
            build_function_dataset(self.samples, self.output)

    def test_syntax_failure_and_timeout_are_logged(self):
        self.rows[0]["function"] = "int f(void);"
        self._write_rows()
        with patch(
            "vulnmechanism.dataset.extract_function_cpg",
            side_effect=[subprocess.TimeoutExpired("joern", 1), self.graph],
        ):
            records = build_function_dataset(self.samples, self.output)
        self.assertEqual([row["sample_key"] for row in records], ["primevul:2"])
        failures = [
            json.loads(line)
            for line in self.output.with_suffix(".errors.jsonl").read_text().splitlines()
        ]
        self.assertEqual([row["stage"] for row in failures], ["syntax", "joern"])
        audit = json.loads(self.output.with_suffix(".audit.json").read_text())
        self.assertEqual(audit["success"], 1)
        self.assertEqual(audit["failed"], 2)

    def test_programming_error_is_not_hidden(self):
        with patch("vulnmechanism.dataset.extract_function_cpg", side_effect=TypeError("bug")):
            with self.assertRaisesRegex(TypeError, "bug"):
                build_function_dataset(self.samples, self.output)

    def test_formal_schema_is_strict(self):
        bad = dict(self.rows[0])
        for field in ("sample_key", "dataset", "function", "label", "language", "split"):
            row = dict(bad)
            row.pop(field)
            self.samples.write_text(json.dumps(row) + "\n")
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    build_function_dataset(self.samples, self.output)

    def test_c_family_and_external_roundtrip(self):
        from vulnmechanism.dataset import _resolve_sample, _sample_fields

        for name, source, expected in [
            ("x.c", "void f(void) {}", "c"),
            ("x.cpp", "void f(void) {}", "cpp"),
            ("x.C", "void f(void) {}", "cpp"),
            ("", "void f(void) {}", "c"),
            ("", "void f() { auto x = []() { return 1; }; }", "cpp"),
        ]:
            row = {
                "sample_key": "sven:p:before",
                "dataset": "sven",
                "pair_id": "sven:p",
                "function": source,
                "label": 1,
                "language": "c_cpp",
                "file_name": name,
                "split": "external_test",
            }
            self.assertEqual(_resolve_sample(_sample_fields(row, 1))[0], expected)

        self.rows = [
            {
                "sample_key": "sven:p:before",
                "dataset": "sven",
                "pair_id": "sven:p",
                "function": "void f(void) {}",
                "label": 1,
                "language": "c_cpp",
                "file_name": "x.cpp",
                "split": "external_test",
            },
            {
                "sample_key": "sven:p:after",
                "dataset": "sven",
                "pair_id": "sven:p",
                "function": "not a function",
                "label": 0,
                "language": "c_cpp",
                "split": "external_test",
            },
        ]
        self._write_rows()
        with patch("vulnmechanism.dataset.extract_function_cpg", return_value=self.graph):
            records = build_function_dataset(self.samples, self.output)
        self.assertEqual(records[0]["dataset"], "sven")
        self.assertEqual(records[0]["pair_id"], "sven:p")
        self.assertEqual(records[0]["resolved_language"], "cpp")
        self.assertEqual(record_split(records[0]), "external_test")


if __name__ == "__main__":
    unittest.main()
