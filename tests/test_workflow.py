import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from semantic_demo.candidate_graph import discover_relevant_candidates
from semantic_demo.joern import RepositoryCall, RepositoryMethod
from semantic_demo.semantics import Candidate, NORMALIZATION_SCHEMA_VERSION
from semantic_demo.source import parse_functions
from semantic_demo.workflow import normalize
from semantic_demo.cli import read_jsonl, write_jsonl


class _FakeRepository:
    def __init__(self, sources):
        self.sources = sources

    def read_blob(self, path):
        return self.sources[path]

    def function_source(
        self,
        *,
        path,
        name,
        start_line,
        end_line,
        parameters,
        parameter_types,
        language_hint=None,
    ):
        functions = [
            function
            for function in parse_functions(path, self.sources[path], language_hint=language_hint)
            if function.name == name and function.start_line == start_line
        ]
        if len(functions) != 1:
            raise AssertionError((path, name, start_line, len(functions)))
        return functions[0]


class _FakeIndex:
    def __init__(self, repository, methods):
        self.repository = repository
        self._methods = {method.full_name: method for method in methods}

    def methods(self):
        return self._methods

    def callee_methods(self, call):
        if call.dispatch_type != "STATIC_DISPATCH":
            return []
        method = self._methods.get(call.method_full_name)
        return [method] if method is not None else []


class CandidatePolicyTests(unittest.TestCase):
    def test_recursive_expansion_requires_summary_relevant_flow(self):
        entry_source = "void entry(char *p, int n) { helper(p, n); }\n"
        helper_source = """
int helper(char *p, int n)
{
    irrelevant(1);
    return relevant(n);
}
void irrelevant(int x) { (void)x; }
int relevant(int n) { return n; }
"""
        entry = parse_functions("entry.c", entry_source)[0]
        parsed = {f.name: f for f in parse_functions("helpers.c", helper_source)}

        relevant_call = next(call for call in parsed["helper"].calls() if call.name == "relevant")
        irrelevant_call = next(call for call in parsed["helper"].calls() if call.name == "irrelevant")
        entry_call = entry.calls()[0]

        entry_method = RepositoryMethod(
            "entry", "entry", "entry.c", entry.start_line, entry.end_line,
            "void", entry.parameters, entry.parameter_types,
            (RepositoryCall(entry_call.line, "helper", "helper", "STATIC_DISPATCH"),),
        )
        helper_method = RepositoryMethod(
            "helper", "helper", "helpers.c", parsed["helper"].start_line,
            parsed["helper"].end_line, "int", parsed["helper"].parameters,
            parsed["helper"].parameter_types,
            (
                RepositoryCall(irrelevant_call.line, "irrelevant", "irrelevant", "STATIC_DISPATCH"),
                RepositoryCall(relevant_call.line, "relevant", "relevant", "STATIC_DISPATCH"),
            ),
        )
        irrelevant_method = RepositoryMethod(
            "irrelevant", "irrelevant", "helpers.c", parsed["irrelevant"].start_line,
            parsed["irrelevant"].end_line, "void", parsed["irrelevant"].parameters,
            parsed["irrelevant"].parameter_types, (),
        )
        relevant_method = RepositoryMethod(
            "relevant", "relevant", "helpers.c", parsed["relevant"].start_line,
            parsed["relevant"].end_line, "int", parsed["relevant"].parameters,
            parsed["relevant"].parameter_types, (),
        )
        repository = _FakeRepository({"entry.c": entry_source, "helpers.c": helper_source})
        index = _FakeIndex(repository, [entry_method, helper_method, irrelevant_method, relevant_method])

        discovery = discover_relevant_candidates("S01", index, entry_method, entry)
        names = {candidate.function.name for candidate in discovery.candidates}
        self.assertEqual({"helper", "relevant"}, names)
        self.assertEqual(1, discovery.direct_candidates)
        self.assertEqual(1, discovery.recursive_candidates)


class NormalizationWorkflowTests(unittest.TestCase):
    def _args(self, root, samples_path, output_path):
        return SimpleNamespace(
            samples=str(samples_path),
            output=str(output_path),
            refresh=False,
            llm_backend="api",
            local_model="/unused/model.gguf",
            llama_server="/unused/llama-server",
            joern_dir="/unused",
            java_home="/unused",
            cpg_cache_dir=str(root / "cpg"),
        )

    def test_subset_normalize_preserves_unselected_records(self):
        function = parse_functions("helper.c", "int helper(int n) { return n; }\n")[0]
        candidate = Candidate("S01", function, (1,), method_full_name="helper")
        manifest_record = {
            "record_type": "candidate",
            "sample_key": "S01",
            "source_path": "helper.c",
            "function": "helper",
            "source_line": function.start_line,
            "source_fingerprint": "new-fingerprint",
            "skip_reason": None,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples_path = root / "samples.jsonl"
            output_path = root / "normalizer.jsonl"
            write_jsonl(samples_path, [{
                "sample_key": "S01",
                "repository_git_dir": "/unused",
                "vulnerable_commit": "a" * 40,
                "entry_path": "entry.c",
                "entry_function": "entry",
                "scan_paths": [],
            }])
            write_jsonl(output_path, [{
                "schema_version": NORMALIZATION_SCHEMA_VERSION,
                "sample_key": "S99",
                "source_path": "old.c",
                "function": "old",
                "source_line": 1,
                "source_fingerprint": "old",
                "normalizer": "static-skip",
                "skip_reason": "old",
                "summaries": [],
            }])
            args = self._args(root, samples_path, output_path)

            with patch(
                "semantic_demo.workflow._load_sample_manifest",
                return_value=(Mock(), ({"candidate_count": 1}, [manifest_record])),
            ), patch(
                "semantic_demo.workflow.load_manifest_candidate", return_value=candidate
            ), patch(
                "semantic_demo.workflow.candidate_source_fingerprint",
                return_value="new-fingerprint",
            ), patch(
                "semantic_demo.workflow.llm_normalize", return_value=[]
            ):
                normalize(args)

            records = read_jsonl(output_path)
            self.assertEqual({"S01", "S99"}, {record["sample_key"] for record in records})

    def test_complete_cache_does_not_start_local_llm(self):
        function = parse_functions("helper.c", "int helper(int n) { return n; }\n")[0]
        manifest_record = {
            "record_type": "candidate",
            "sample_key": "S01",
            "source_path": "helper.c",
            "function": "helper",
            "source_line": function.start_line,
            "source_fingerprint": "same",
            "skip_reason": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples_path = root / "samples.jsonl"
            output_path = root / "normalizer.jsonl"
            write_jsonl(samples_path, [{
                "sample_key": "S01",
                "repository_git_dir": "/unused",
                "vulnerable_commit": "a" * 40,
                "entry_path": "entry.c",
                "entry_function": "entry",
                "scan_paths": [],
            }])
            write_jsonl(output_path, [{
                "schema_version": NORMALIZATION_SCHEMA_VERSION,
                "sample_key": "S01",
                "source_path": "helper.c",
                "function": "helper",
                "source_line": function.start_line,
                "source_fingerprint": "same",
                "normalizer": "localized-hybrid",
                "llm_backend": "local",
                "llm_model": "model",
                "summaries": [],
            }])
            args = self._args(root, samples_path, output_path)
            args.llm_backend = "local"
            args.local_model = "/unused/model.gguf"

            with patch(
                "semantic_demo.workflow._load_sample_manifest",
                return_value=(Mock(), ({"candidate_count": 1}, [manifest_record])),
            ), patch("semantic_demo.workflow.cli.local_llm_server") as server:
                normalize(args)
            server.assert_not_called()


if __name__ == "__main__":
    unittest.main()
