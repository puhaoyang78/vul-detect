import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from semantic_demo.candidate_graph import discover_relevant_candidates
from semantic_demo.joern import RepositoryCall, RepositoryMethod
from semantic_demo.normalization_v2 import NORMALIZATION_IMPLEMENTATION_VERSION, _slice_source
from semantic_demo.semantics import Candidate, NORMALIZATION_SCHEMA_VERSION
from semantic_demo.source import parse_functions
from semantic_demo.standard_semantics import summaries_for_function
from semantic_demo.workflow import _upsert_by_sample, normalize
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
    def test_recursive_expansion_follows_memory_relevance_not_every_call(self):
        entry_source = "void entry(char *p, int n) { helper(p, n); memcpy(p, p, n); }\n"
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
        entry_call = next(call for call in entry.calls() if call.name == "helper")
        relevant_call = next(call for call in parsed["helper"].calls() if call.name == "relevant")
        irrelevant_call = next(call for call in parsed["helper"].calls() if call.name == "irrelevant")

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

    def test_opaque_typedef_parameter_is_not_pruned(self):
        entry_source = "void entry(buf_T b) { helper(b); }\n"
        helper_source = "void helper(buf_T b) { strcpy(b, \"x\"); }\n"
        entry = parse_functions("entry.c", entry_source)[0]
        helper = parse_functions("helper.c", helper_source)[0]
        call = entry.calls()[0]
        entry_method = RepositoryMethod(
            "entry", "entry", "entry.c", entry.start_line, entry.end_line,
            "void", entry.parameters, ("buf_T",),
            (RepositoryCall(call.line, "helper", "helper", "STATIC_DISPATCH"),),
        )
        helper_method = RepositoryMethod(
            "helper", "helper", "helper.c", helper.start_line, helper.end_line,
            "void", helper.parameters, ("buf_T",), (),
        )
        repository = _FakeRepository({"entry.c": entry_source, "helper.c": helper_source})
        index = _FakeIndex(repository, [entry_method, helper_method])
        discovery = discover_relevant_candidates("S02", index, entry_method, entry)
        self.assertEqual(["helper"], [item.function.name for item in discovery.candidates])


class StandardSemanticTests(unittest.TestCase):
    def test_strcpy_wrapper_gets_read_and_write_summaries(self):
        function = parse_functions(
            "copy.c", "void copy(char *d, const char *s) { strcpy(d, s); }\n"
        )[0]
        summaries = summaries_for_function(function)
        self.assertIn(
            {"kind": "WRITE", "buffer": "arg0", "length": "strlen(arg1) + 1"},
            summaries,
        )
        self.assertIn(
            {"kind": "READ", "buffer": "arg1", "length": "strlen(arg1) + 1"},
            summaries,
        )

    def test_snprintf_wrapper_uses_destination_capacity(self):
        function = parse_functions(
            "fmt.c",
            "void fmt(char *d, unsigned long n, const char *s) { snprintf(d, n, \"%s\", s); }\n",
        )[0]
        self.assertIn(
            {"kind": "WRITE", "buffer": "arg0", "length": "arg1"},
            summaries_for_function(function),
        )


class SliceTests(unittest.TestCase):
    def test_large_function_slice_keeps_reaching_definition(self):
        filler = "\n".join(f"int unused_{i} = {i};" for i in range(1200))
        source = (
            "void big(char *dst, int n) {\n"
            "int len = n + 1;\n"
            + filler
            + "\ncustom_copy(dst, len);\n}"
        )
        function = parse_functions("big.c", source)[0]
        call = next(call for call in function.calls() if call.name == "custom_copy")
        sliced = _slice_source(function, call.line, call.arguments)
        self.assertIn("int len = n + 1;", sliced)
        self.assertIn("custom_copy(dst, len);", sliced)
        self.assertLess(len(sliced), len(function.text))


class NormalizationWorkflowTests(unittest.TestCase):
    def _args(self, root, samples_path, output_path, refresh=False):
        return SimpleNamespace(
            samples=str(samples_path),
            output=str(output_path),
            refresh=refresh,
            llm_backend="api",
            local_model="/unused/model.gguf",
            llama_server="/unused/llama-server",
            joern_dir="/unused",
            java_home="/unused",
            cpg_cache_dir=str(root / "cpg"),
        )

    @staticmethod
    def _sample(key="S01"):
        return {
            "sample_key": key,
            "repository_git_dir": "/unused",
            "vulnerable_commit": "a" * 40,
            "entry_path": "entry.c",
            "entry_function": "entry",
            "scan_paths": [],
        }

    def test_subset_refresh_preserves_unselected_records_and_prunes_stale_selected(self):
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
            write_jsonl(samples_path, [self._sample()])
            write_jsonl(output_path, [
                {
                    "schema_version": NORMALIZATION_SCHEMA_VERSION,
                    "sample_key": "S99",
                    "source_path": "old.c",
                    "function": "old",
                    "source_line": 1,
                    "source_fingerprint": "old",
                    "normalizer": "static-skip",
                    "skip_reason": "old",
                    "summaries": [],
                },
                {
                    "schema_version": NORMALIZATION_SCHEMA_VERSION,
                    "sample_key": "S01",
                    "source_path": "stale.c",
                    "function": "stale",
                    "source_line": 9,
                    "source_fingerprint": "stale",
                    "normalizer": "static-skip",
                    "skip_reason": "stale",
                    "summaries": [],
                },
            ])
            args = self._args(root, samples_path, output_path, refresh=True)
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
            self.assertNotIn("stale", {record["function"] for record in records})

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
            write_jsonl(samples_path, [self._sample()])
            write_jsonl(output_path, [{
                "schema_version": NORMALIZATION_SCHEMA_VERSION,
                "normalization_implementation_version": NORMALIZATION_IMPLEMENTATION_VERSION,
                "sample_key": "S01",
                "source_path": "helper.c",
                "function": "helper",
                "source_line": function.start_line,
                "source_fingerprint": "same",
                "normalizer": "relevance-sliced-hybrid",
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


class RunMergeTests(unittest.TestCase):
    def test_partial_selected_results_preserve_unprocessed_old_samples(self):
        old = [
            {"sample_key": "S01", "value": "old1"},
            {"sample_key": "S02", "value": "old2"},
            {"sample_key": "S03", "value": "old3"},
        ]
        new = [{"sample_key": "S01", "value": "new1"}]
        merged = _upsert_by_sample(old, new, {"S01", "S02"})
        by_key = {record["sample_key"]: record["value"] for record in merged}
        self.assertEqual({"S01": "new1", "S02": "old2", "S03": "old3"}, by_key)


if __name__ == "__main__":
    unittest.main()
