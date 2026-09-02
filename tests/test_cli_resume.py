import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from semantic_demo.cli import detect, normalize_command, read_jsonl, write_jsonl
from semantic_demo.semantics import Candidate
from semantic_demo.source import parse_functions


class CheckpointTests(unittest.TestCase):
    def test_llm_normalization_resumes_after_completed_candidate(self):
        entry = parse_functions("entry.c", "void entry(void) {}\n")[0]
        functions = [
            parse_functions(
                "wrappers.c", f"void wrapper_{index}(char *p) {{ free(p); }}\n"
            )[0]
            for index in (1, 2)
        ]
        candidates = [Candidate("S01", function, (1,)) for function in functions]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples_path = root / "samples.jsonl"
            output_path = root / "normalizer.jsonl"
            write_jsonl(
                samples_path,
                [
                    {
                        "sample_key": "S01",
                        "repository_git_dir": "/unused",
                        "vulnerable_commit": "a" * 40,
                        "entry_path": "entry.c",
                        "entry_function": "entry",
                        "scan_paths": [],
                    }
                ],
            )
            args = SimpleNamespace(
                samples=str(samples_path),
                output=str(output_path),
                llm_backend="api",
                llama_server="/unused",
                local_model="/unused",
                refresh=False,
            )

            def interrupted(candidate, **_options):
                if candidate.function.name == "wrapper_1":
                    return [{"kind": "VALUE", "expression": "arg0"}]
                raise RuntimeError("interrupted")

            with patch("semantic_demo.cli._load_entry", return_value=(None, entry)), patch(
                "semantic_demo.cli.discover_candidates", return_value=candidates
            ), patch("semantic_demo.cli.llm_normalize", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "normalization failed"):
                    normalize_command(args)

            self.assertEqual(
                ["wrapper_1"],
                [record["function"] for record in read_jsonl(output_path)],
            )

            with patch("semantic_demo.cli._load_entry", return_value=(None, entry)), patch(
                "semantic_demo.cli.discover_candidates", return_value=candidates
            ), patch(
                "semantic_demo.cli.llm_normalize",
                return_value=[{"kind": "VALUE", "expression": "arg0"}],
            ) as normalize:
                normalize_command(args)

            self.assertEqual(1, normalize.call_count)
            self.assertEqual("wrapper_2", normalize.call_args.args[0].function.name)
            self.assertEqual(
                ["wrapper_1", "wrapper_2"],
                [record["function"] for record in read_jsonl(output_path)],
            )

    def test_detection_resumes_after_completed_sample(self):
        entries = [
            parse_functions(
                f"entry_{index}.c", f"void entry_{index}(void) {{}}\n"
            )[0]
            for index in (1, 2)
        ]
        samples = [
            {
                "sample_key": f"S0{index}",
                "repository_git_dir": "/unused",
                "vulnerable_commit": str(index) * 40,
                "entry_path": entry.path,
                "entry_function": entry.name,
                "scan_paths": [],
            }
            for index, entry in enumerate(entries, 1)
        ]
        verdict = Mock()
        verdict.as_json.return_value = {"verdict": "NOT_DETECTED", "reason": "test"}
        baseline_model = Mock()
        baseline_model.signature = "test-linevul"
        baseline_model.predict.return_value = verdict

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples_path = root / "samples.jsonl"
            replay_path = root / "replay.jsonl"
            semantics_path = root / "semantics.jsonl"
            detections_path = root / "detections.jsonl"
            write_jsonl(samples_path, samples)
            write_jsonl(replay_path, [])

            joern = Mock()
            joern.timeout = 180
            joern.ensure_available.return_value = None

            with patch(
                "semantic_demo.cli._load_entry",
                side_effect=[(None, entries[0]), RuntimeError("interrupted")],
            ), patch(
                "semantic_demo.cli.discover_candidates", return_value=[]
            ), patch(
                "semantic_demo.cli.LineVulBaseline", return_value=baseline_model
            ), patch(
                "semantic_demo.cli.JoernValidator", return_value=joern
            ), patch("semantic_demo.cli.analyze", return_value=verdict):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    detect(
                        str(samples_path),
                        str(replay_path),
                        str(semantics_path),
                        str(detections_path),
                    )

            self.assertEqual(
                ["S01"],
                [record["sample_key"] for record in read_jsonl(detections_path)],
            )

            with patch(
                "semantic_demo.cli._load_entry",
                side_effect=[(None, entries[0]), (None, entries[1])],
            ), patch(
                "semantic_demo.cli.discover_candidates", return_value=[]
            ), patch(
                "semantic_demo.cli.LineVulBaseline", return_value=baseline_model
            ), patch(
                "semantic_demo.cli.JoernValidator", return_value=joern
            ), patch("semantic_demo.cli.analyze", return_value=verdict) as analyze:
                detect(
                    str(samples_path),
                    str(replay_path),
                    str(semantics_path),
                    str(detections_path),
                )

            self.assertEqual(1, analyze.call_count)
            self.assertEqual(
                ["S01", "S02"],
                [record["sample_key"] for record in read_jsonl(detections_path)],
            )


if __name__ == "__main__":
    unittest.main()
