import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from semantic_demo.analyzer import Operation, analyze
from semantic_demo.joern import (
    JoernCall,
    RepositoryCall,
    RepositoryMethod,
    JoernError,
    JoernFacts,
    JoernMethodNotFound,
    JoernTimeout,
    JoernValidator,
)
from semantic_demo.semantics import (
    Candidate,
    NORMALIZATION_RESPONSE_SCHEMA,
    Validation,
    candidate_validation_error,
    _extract_json_object,
    discover_candidates,
    llm_normalize,
    _response_content,
    _validate_by_composition,
    validate_summary,
)
from semantic_demo.source import FunctionSource, GitRepository, parse_functions
from semantic_demo.z3_reasoner import reason_memory_safety


class StaticFactsValidator:
    def __init__(self, facts):
        self._facts = facts

    def facts(self, _candidate):
        return self._facts


def copy_facts(function, call_name="memcpy"):
    call = JoernCall(
        line=function.start_line + 2,
        name=call_name,
        arguments={0: "dst", 1: "src", 2: "len"},
    )
    return JoernFacts(
        parameters={
            0: ("dst", "char *"),
            1: ("src", "const char *"),
            2: ("len", "unsigned long"),
        },
        calls={(call.line, call.name): call},
        flows={
            (0, call.line, call.name, 0),
            (1, call.line, call.name, 1),
            (2, call.line, call.name, 2),
        },
    )


class GitRepositoryTests(unittest.TestCase):
    def test_read_blob_resolves_repository_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(
                ["git", "init", "--bare", str(root / "repo.git")],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            work = root / "work"
            subprocess.run(
                ["git", "clone", str(root / "repo.git"), str(work)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(work), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(work), "config", "user.name", "Test"],
                check=True,
            )
            (work / "real").mkdir()
            (work / "real" / "list.h").write_text(
                "static inline void list_add_tail(void) {}\n"
            )
            (work / "include").mkdir()
            (work / "include" / "list.h").symlink_to("../real/list.h")
            subprocess.run(["git", "-C", str(work), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(work), "commit", "-m", "fixture"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(work), "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "-C", str(work), "push", "origin", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            repository = GitRepository(str(root / "repo.git"), revision)
            self.assertEqual(
                "static inline void list_add_tail(void) {}\n",
                repository.read_blob("include/list.h"),
            )
            materialized = root / "materialized"
            repository.materialize(materialized, ("include",))
            self.assertTrue((materialized / "real" / "list.h").is_file())
            self.assertEqual(
                "../real/list.h",
                (materialized / "include" / "list.h").readlink().as_posix(),
            )


class CandidateParseBoundaryTests(unittest.TestCase):
    def test_signature_annotation_error_does_not_skip_clean_body(self):
        function = FunctionSource(
            path="annotated.c",
            name="annotated",
            text=(
                "RZ_API int annotated(RZ_NONNULL char *p) "
                "{ if (!p) return -1; return p[0]; }"
            ),
            translation_unit="",
            language="c",
            parameters=("p",),
            parameter_types=("char *",),
            parameter_pointer_like=(True,),
            parameter_signatures=("char*$",),
            start_line=1,
            end_line=1,
            parse_has_error=True,
        )
        self.assertIsNone(candidate_validation_error(function))


class SemanticValidationTests(unittest.TestCase):
    def setUp(self):
        self.function = parse_functions(
            "wrapper.c",
            """
            void copy_wrap(char *dst, const char *src, unsigned long len)
            {
                memcpy(dst, src, len);
            }
            """,
        )[0]
        self.candidate = Candidate("sample", self.function, (10,))
        self.validator = StaticFactsValidator(copy_facts(self.function))

    def test_write_buffer_and_length_reach_specified_sink(self):
        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            joern=self.validator,
        )
        self.assertTrue(result.passed)

    def test_wrong_length_is_rejected(self):
        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg1"},
            joern=self.validator,
        )
        self.assertFalse(result.passed)

    def test_source_parameter_name_is_canonicalized(self):
        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "dst", "length": "len"},
            joern=self.validator,
        )
        self.assertTrue(result.passed)
        self.assertEqual("arg0", result.summary["buffer"])
        self.assertEqual("arg2", result.summary["length"])

    def test_read_buffer_and_length_reach_same_source(self):
        function = parse_functions(
            "reader.c",
            """
            int send_wrap(int fd, const char *src, unsigned long len)
            {
                return write(fd, src, len);
            }
            """,
        )[0]
        call = JoernCall(
            line=function.start_line + 2,
            name="write",
            arguments={0: "fd", 1: "src", 2: "len"},
        )
        facts = JoernFacts(
            parameters={
                0: ("fd", "int"),
                1: ("src", "const char *"),
                2: ("len", "unsigned long"),
            },
            calls={(call.line, call.name): call},
            flows={
                (0, call.line, call.name, 0),
                (1, call.line, call.name, 1),
                (2, call.line, call.name, 2),
            },
        )
        result = validate_summary(
            Candidate("sample", function, (30,)),
            {"kind": "READ", "buffer": "arg1", "length": "arg2"},
            joern=StaticFactsValidator(facts),
        )
        self.assertTrue(result.passed)

    def test_unknown_name_family_is_not_trusted_as_sink(self):
        function = parse_functions(
            "custom.c",
            """
            void wrap(char *dst, unsigned long len)
            {
                socket_recv(0, dst, len);
            }
            """,
        )[0]
        call = JoernCall(
            line=function.start_line + 2,
            name="socket_recv",
            arguments={0: "0", 1: "dst", 2: "len"},
        )
        facts = JoernFacts(
            parameters={0: ("dst", "char *"), 1: ("len", "unsigned long")},
            calls={(call.line, call.name): call},
            flows={
                (0, call.line, call.name, 1),
                (1, call.line, call.name, 2),
            },
        )
        result = validate_summary(
            Candidate("sample", function, (20,)),
            {"kind": "WRITE", "buffer": "arg0", "length": "arg1"},
            joern=StaticFactsValidator(facts),
        )
        self.assertFalse(result.passed)

    def test_value_return_relation_requires_exact_return(self):
        function = parse_functions(
            "value.c",
            """
            unsigned long identity(unsigned long len)
            {
                return len;
            }
            """,
        )[0]
        facts = JoernFacts(
            parameters={0: ("len", "unsigned long")},
            returns=["return len;"],
            return_flows={0},
        )
        result = validate_summary(
            Candidate("sample", function, (40,)),
            {"kind": "VALUE", "target": "return", "expression": "arg0"},
            joern=StaticFactsValidator(facts),
        )
        self.assertTrue(result.passed)

    def test_indirect_call_does_not_reject_unrelated_direct_summary(self):
        function = parse_functions(
            "mixed.c",
            """
            void mixed(char *dst, const char *src, unsigned long len, void (*cb)(char *))
            {
                memcpy(dst, src, len);
                cb(dst);
            }
            """,
        )[0]
        call = JoernCall(
            line=function.start_line + 2,
            name="memcpy",
            arguments={0: "dst", 1: "src", 2: "len"},
        )
        facts = JoernFacts(
            parameters={
                0: ("dst", "char *"),
                1: ("src", "const char *"),
                2: ("len", "unsigned long"),
                3: ("cb", "void (*)(char *)"),
            },
            calls={(call.line, call.name): call},
            flows={
                (0, call.line, call.name, 0),
                (1, call.line, call.name, 1),
                (2, call.line, call.name, 2),
            },
        )
        result = validate_summary(
            Candidate("sample", function, (1,)),
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            joern=StaticFactsValidator(facts),
        )
        self.assertTrue(result.passed)

    def test_guard_summary_is_not_part_of_sound_schema(self):
        result = validate_summary(
            self.candidate,
            {"kind": "GUARD", "relation": "arg2 <= 8"},
            joern=self.validator,
        )
        self.assertFalse(result.passed)
        self.assertIn("kind must be", result.reason)

    def test_scalar_parameter_cannot_be_memory_buffer(self):
        function = parse_functions(
            "bad.c",
            """
            void bad_wrap(int fd, unsigned long len)
            {
                write(fd, (const void *)len, 4);
            }
            """,
        )[0]
        result = validate_summary(
            Candidate("sample", function, (50,)),
            {"kind": "READ", "buffer": "arg1", "length": "4"},
            joern=StaticFactsValidator(JoernFacts()),
        )
        self.assertFalse(result.passed)
        self.assertIn("not pointer-like", result.reason)

    def test_local_parser_error_does_not_discard_independent_facts(self):
        function = parse_functions(
            "broken.c",
            "void broken(char *p) { int = ; p[0] = 1; }",
        )[0]
        self.assertTrue(function.parse_has_error)
        self.assertIsNone(candidate_validation_error(function))
        accesses = function.direct_memory_accesses()
        self.assertTrue(
            any(
                access.kind == "WRITE" and "p" in access.buffer
                for access in accesses
            )
        )

    def test_missing_joern_method_rejects_only_summary(self):
        class Missing:
            def facts(self, _candidate):
                raise JoernMethodNotFound("method_not_found:copy_wrap")

        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            joern=Missing(),
        )
        self.assertFalse(result.passed)
        self.assertIn("method_not_found", result.reason)

    def test_joern_infrastructure_error_propagates(self):
        class Broken:
            def facts(self, _candidate):
                raise JoernError("launcher failed")

        with self.assertRaisesRegex(JoernError, "launcher failed"):
            validate_summary(
                self.candidate,
                {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
                joern=Broken(),
            )

    def test_joern_timeout_rejects_only_summary(self):
        class Timeout:
            def facts(self, _candidate):
                raise JoernTimeout("Joern timed out")

        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            joern=Timeout(),
        )
        self.assertFalse(result.passed)
        self.assertIn("timed out", result.reason)

    def test_joern_timeout_is_cached_per_function(self):
        validator = JoernValidator(timeout=1)
        validator.ensure_available = lambda: None
        with patch(
            "semantic_demo.joern.subprocess.run",
            side_effect=subprocess.TimeoutExpired("joern", 1),
        ) as run:
            with self.assertRaises(JoernTimeout):
                validator.facts(self.candidate)
            with self.assertRaises(JoernTimeout):
                validator.facts(self.candidate)
        self.assertEqual(1, run.call_count)

    def test_joern_call_ids_keep_same_line_calls_distinct(self):
        facts = JoernValidator._parse(
            "\n".join(
                [
                    "PARAM\t0\tdst\tchar *",
                    "ARG\t11\t10\tmemcpy\t0\tmemcpy(a,b,n)\ta",
                    "ARG\t11\t10\tmemcpy\t2\tmemcpy(a,b,n)\tn",
                    "FLOW\t0\t11\t0",
                    "ARG\t12\t10\tmemcpy\t0\tmemcpy(c,d,m)\tc",
                    "ARG\t12\t10\tmemcpy\t2\tmemcpy(c,d,m)\tm",
                ]
            )
        )
        self.assertEqual({"11", "12"}, set(facts.calls))
        self.assertNotEqual(
            facts.calls["11"].arguments,
            facts.calls["12"].arguments,
        )

    def test_response_content_accepts_final_answer(self):
        result = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"summaries":[]}',
                        "reasoning_content": "internal",
                    },
                }
            ]
        }
        self.assertEqual('{"summaries":[]}', _response_content(result))

    def test_llm_json_requires_one_strict_object(self):
        self.assertEqual(
            {"summaries": []},
            _extract_json_object('{"summaries":[]}'),
        )
        with self.assertRaisesRegex(ValueError, "must be one JSON object"):
            _extract_json_object('[]')
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            _extract_json_object('prefix {"summaries":[]}')

    def test_response_content_rejects_truncated_output_even_with_content(self):
        result = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": '{"summaries":[{"kind":"VALUE"',
                        "reasoning_content": "",
                    },
                }
            ],
            "usage": {"completion_tokens": 512},
        }
        with self.assertRaisesRegex(ValueError, "truncated at max_tokens"):
            _response_content(result)

    def test_standard_memcpy_summary_needs_no_llm_call(self):
        candidate = Candidate(
            "sample",
            parse_functions(
                "copy.c",
                """
                void copy(char *dst, const char *src, unsigned long len)
                {
                    memcpy(dst, src, len);
                }
                """,
            )[0],
            (),
        )
        with patch("semantic_demo.semantics.urllib.request.urlopen") as open_url:
            summaries = llm_normalize(
                candidate,
                api_key="local",
                base_url="http://127.0.0.1:1/v1",
                model="test",
                response_schema=NORMALIZATION_RESPONSE_SCHEMA,
            )
        open_url.assert_not_called()
        self.assertIn(
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            summaries,
        )
        self.assertIn(
            {"kind": "READ", "buffer": "arg1", "length": "arg2"},
            summaries,
        )

    def test_local_length_standard_call_uses_one_localized_llm_endpoint(self):
        candidate = Candidate(
            "sample",
            parse_functions(
                "copy.c",
                """
                void copy(char *dst, const char *src, unsigned long len)
                {
                    unsigned long n = len;
                    memcpy(dst, src, n);
                }
                """,
            )[0],
            (),
        )
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "summaries": [
                                        {
                                            "kind": "WRITE",
                                            "buffer": "arg0",
                                            "length": "arg2",
                                        },
                                        {
                                            "kind": "READ",
                                            "buffer": "arg1",
                                            "length": "arg2",
                                        },
                                    ]
                                }
                            )
                        },
                    }
                ]
            }
        ).encode()

        with patch(
            "semantic_demo.semantics.urllib.request.urlopen",
            return_value=response,
        ) as open_url:
            summaries = llm_normalize(
                candidate,
                api_key="local",
                base_url="http://127.0.0.1:1/v1",
                model="test",
                response_schema=NORMALIZATION_RESPONSE_SCHEMA,
            )

        self.assertEqual(1, open_url.call_count)
        self.assertIn(
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            summaries,
        )
        self.assertIn(
            {"kind": "READ", "buffer": "arg1", "length": "arg2"},
            summaries,
        )

    def test_invalid_llm_summary_is_rejected_without_aborting_candidate(self):
        candidate = Candidate(
            "sample",
            parse_functions(
                "level.c",
                """
                int level(int lvl)
                {
                    if (lvl)
                        return 1;
                    return 0;
                }
                """,
            )[0],
            (),
        )
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "summaries": [
                                        {
                                            "kind": "VALUE",
                                            "target": "return",
                                            "expression": "arg1",
                                        }
                                    ]
                                }
                            )
                        },
                    }
                ]
            }
        ).encode()

        with patch(
            "semantic_demo.semantics.urllib.request.urlopen",
            return_value=response,
        ):
            summaries = llm_normalize(
                candidate,
                api_key="local",
                base_url="http://127.0.0.1:1/v1",
                model="test",
                response_schema=NORMALIZATION_RESPONSE_SCHEMA,
            )

        self.assertEqual([], summaries)

    def test_llm_normalize_passes_explicit_response_schema(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"summaries":[]}'},
                    }
                ]
            }
        ).encode()

        with patch(
            "semantic_demo.semantics.urllib.request.urlopen", return_value=response
        ) as open_url:
            candidate = Candidate(
                "sample",
                parse_functions("simple.c", "int simple(int value) { return value; }")[0],
                (),
            )
            self.assertEqual(
                [],
                llm_normalize(
                    candidate,
                    api_key="local",
                    base_url="http://127.0.0.1:1/v1",
                    model="test",
                    response_schema=NORMALIZATION_RESPONSE_SCHEMA,
                ),
            )

        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(
            NORMALIZATION_RESPONSE_SCHEMA,
            payload["response_format"]["schema"],
        )


class CompositionalValidationTests(unittest.TestCase):
    def test_parent_write_summary_composes_from_unique_validated_callee(self):
        function = parse_functions(
            "wrapper.c",
            """
            void copy_outer(char *dst, const char *src, unsigned long len)
            {
                copy_inner(dst, src, len);
            }
            """,
        )[0]
        call = JoernCall(
            line=function.start_line + 2,
            name="copy_inner",
            arguments={0: "dst", 1: "src", 2: "len"},
        )
        facts = JoernFacts(
            parameters={
                0: ("dst", "char *"),
                1: ("src", "const char *"),
                2: ("len", "unsigned long"),
            },
            calls={(call.line, call.name): call},
            flows={
                (0, call.line, call.name, 0),
                (1, call.line, call.name, 1),
                (2, call.line, call.name, 2),
            },
        )
        passed, reason = _validate_by_composition(
            Candidate("sample", function, (10,)),
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            StaticFactsValidator(facts),
            {
                ("inner.c", "copy_inner"): [
                    {"kind": "WRITE", "buffer": "arg0", "length": "arg2"}
                ]
            },
        )
        self.assertTrue(passed)
        self.assertIn("composition verified write", reason)

    def test_ambiguous_same_name_callee_does_not_compose(self):
        function = parse_functions(
            "wrapper.c",
            "void outer(char *p, unsigned long n) { inner(p, n); }",
        )[0]
        call = JoernCall(
            line=function.start_line,
            name="inner",
            arguments={0: "p", 1: "n"},
        )
        facts = JoernFacts(
            parameters={0: ("p", "char *"), 1: ("n", "unsigned long")},
            calls={(call.line, call.name): call},
            flows={
                (0, call.line, call.name, 0),
                (1, call.line, call.name, 1),
            },
        )
        passed, _ = _validate_by_composition(
            Candidate("sample", function, (1,)),
            {"kind": "WRITE", "buffer": "arg0", "length": "arg1"},
            StaticFactsValidator(facts),
            {
                ("a.c", "inner"): [{"kind": "WRITE", "buffer": "arg0", "length": "arg1"}],
                ("b.c", "inner"): [{"kind": "WRITE", "buffer": "arg0", "length": "arg1"}],
            },
        )
        self.assertFalse(passed)


class CandidateDiscoveryTests(unittest.TestCase):
    class StaticRepository:
        def __init__(self, functions):
            self.functions = {
                (function.path, function.name, function.start_line): function
                for function in functions
            }

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
            return self.functions[(path, name, start_line)]

        def read_blob(self, path):
            matches = [
                function.translation_unit
                for function in self.functions.values()
                if function.path == path
            ]
            return matches[0]

    class StaticIndex:
        def __init__(self, methods, functions):
            self.repository = CandidateDiscoveryTests.StaticRepository(functions)
            self._methods = {method.full_name: method for method in methods}

        def methods(self):
            return self._methods

        def callee_methods(self, call):
            if call.dispatch_type != "STATIC_DISPATCH":
                return []
            method = self._methods.get(call.method_full_name)
            return [] if method is None else [method]

    @staticmethod
    def method(full_name, function, calls=()):
        return RepositoryMethod(
            full_name=full_name,
            name=function.name,
            path=function.path,
            start_line=function.start_line,
            end_line=function.end_line,
            return_type=(
                "void"
                if function.text.lstrip().startswith("void ")
                else "int"
            ),
            parameters=function.parameters,
            parameter_types=function.parameter_types,
            calls=tuple(calls),
        )

    def test_same_signature_duplicates_without_preprocessor_are_not_variants(self):
        definitions = parse_functions(
            "defs.c",
            """
            int inner(char *buffer) { return buffer[0]; }
            int inner(char *buffer) { return buffer[1]; }
            """,
        )
        entry = parse_functions(
            "wrapper.c",
            "int outer(char *buffer) { return inner(buffer); }",
        )[0]
        inner_method = self.method("inner:int(char*)", definitions[0])
        entry_method = self.method(
            "outer:int(char*)",
            entry,
            (
                RepositoryCall(
                    line=entry.start_line,
                    name="inner",
                    method_full_name=inner_method.full_name,
                    dispatch_type="STATIC_DISPATCH",
                ),
            ),
        )
        index = self.StaticIndex(
            [entry_method, inner_method],
            [entry, *definitions],
        )
        candidates = discover_candidates("sample", index, entry_method)
        self.assertEqual(1, len(candidates))

    def test_same_file_c_definitions_are_explicit_variants(self):
        definitions = parse_functions(
            "defs.c",
            """
            #ifdef FEATURE
            int inner(char *buffer) { return buffer[0]; }
            #else
            int inner(char *buffer) { return buffer[1]; }
            #endif
            """,
        )
        entry = parse_functions(
            "wrapper.c",
            "int outer(char *buffer) { return inner(buffer); }",
        )[0]
        inner_method = self.method("inner:int(char*)", definitions[0])
        entry_method = self.method(
            "outer:int(char*)",
            entry,
            (
                RepositoryCall(
                    line=entry.start_line,
                    name="inner",
                    method_full_name=inner_method.full_name,
                    dispatch_type="STATIC_DISPATCH",
                ),
            ),
        )
        index = self.StaticIndex(
            [entry_method, inner_method],
            [entry, *definitions],
        )

        candidates = discover_candidates(
            "sample",
            index,
            entry_method,
        )
        self.assertEqual(2, len(candidates))
        self.assertEqual(
            2,
            len({candidate.function.start_line for candidate in candidates}),
        )

    def test_unknown_return_type_is_not_pruned(self):
        function = parse_functions(
            "unknown.c",
            "int unknown(int value) { return value; }",
        )[0]
        entry = parse_functions(
            "entry.c",
            "int entry(int value) { return unknown(value); }",
        )[0]
        unknown_method = RepositoryMethod(
            full_name="unknown:ANY(int)",
            name="unknown",
            path="unknown.c",
            start_line=function.start_line,
            end_line=function.end_line,
            return_type="ANY",
            parameters=function.parameters,
            parameter_types=function.parameter_types,
            calls=(),
        )
        entry_method = self.method(
            "entry:int(int)",
            entry,
            (
                RepositoryCall(
                    line=entry.start_line,
                    name="unknown",
                    method_full_name=unknown_method.full_name,
                    dispatch_type="STATIC_DISPATCH",
                ),
            ),
        )
        index = self.StaticIndex(
            [entry_method, unknown_method],
            [entry, function],
        )
        candidates = discover_candidates("sample", index, entry_method)
        self.assertEqual(["unknown"], [item.function.name for item in candidates])

    def test_schema_inexpressible_helper_prunes_its_subgraph(self):
        helper = parse_functions(
            "helper.c",
            """
            void helper(int value)
            {
                deep(value);
            }
            """,
        )[0]
        deep = parse_functions(
            "deep.c",
            "int deep(int value) { return value; }",
        )[0]
        entry = parse_functions(
            "entry.c",
            "void entry(void) { helper(1); }",
        )[0]

        deep_method = self.method("deep:int(int)", deep)
        helper_method = self.method(
            "helper:void(int)",
            helper,
            (
                RepositoryCall(
                    line=helper.start_line + 2,
                    name="deep",
                    method_full_name=deep_method.full_name,
                    dispatch_type="STATIC_DISPATCH",
                ),
            ),
        )
        entry_method = self.method(
            "entry:void()",
            entry,
            (
                RepositoryCall(
                    line=entry.start_line,
                    name="helper",
                    method_full_name=helper_method.full_name,
                    dispatch_type="STATIC_DISPATCH",
                ),
            ),
        )
        index = self.StaticIndex(
            [entry_method, helper_method, deep_method],
            [entry, helper, deep],
        )

        candidates = discover_candidates(
            "sample",
            index,
            entry_method,
        )
        self.assertEqual([], candidates)

    def test_candidate_discovery_has_no_fixed_128_function_cap(self):
        count = 129
        functions = {}
        for index in range(count):
            next_call = (
                f"return f{index + 1}(value);"
                if index + 1 < count
                else "return value;"
            )
            functions[f"f{index}"] = parse_functions(
                f"f{index}.c",
                f"int f{index}(int value) {{ {next_call} }}",
            )[0]
        entry = parse_functions(
            "entry.c",
            "int entry(int value) { return f0(value); }",
        )[0]

        methods = []
        for index in range(count):
            calls = ()
            if index + 1 < count:
                calls = (
                    RepositoryCall(
                        line=functions[f"f{index}"].start_line,
                        name=f"f{index + 1}",
                        method_full_name=f"f{index + 1}:int(int)",
                        dispatch_type="STATIC_DISPATCH",
                    ),
                )
            methods.append(
                self.method(
                    f"f{index}:int(int)",
                    functions[f"f{index}"],
                    calls,
                )
            )
        entry_method = self.method(
            "entry:int(int)",
            entry,
            (
                RepositoryCall(
                    line=entry.start_line,
                    name="f0",
                    method_full_name="f0:int(int)",
                    dispatch_type="STATIC_DISPATCH",
                ),
            ),
        )
        index = self.StaticIndex(
            [entry_method, *methods],
            [entry, *functions.values()],
        )

        candidates = discover_candidates(
            "sample",
            index,
            entry_method,
        )
        self.assertEqual(count, len(candidates))

    def test_dynamic_joern_call_remains_opaque(self):
        callee = parse_functions(
            "callee.c",
            "int callee(int value) { return value; }",
        )[0]
        entry = parse_functions(
            "entry.c",
            "int entry(int value) { return callee(value); }",
        )[0]
        callee_method = self.method("callee:int(int)", callee)
        entry_method = self.method(
            "entry:int(int)",
            entry,
            (
                RepositoryCall(
                    line=entry.start_line,
                    name="callee",
                    method_full_name=callee_method.full_name,
                    dispatch_type="DYNAMIC_DISPATCH",
                ),
            ),
        )
        index = self.StaticIndex(
            [entry_method, callee_method],
            [entry, callee],
        )
        self.assertEqual(
            [],
            discover_candidates("sample", index, entry_method),
        )

    def test_unresolved_joern_call_remains_opaque(self):
        entry = parse_functions(
            "entry.c",
            "int entry(int value) { return missing(value); }",
        )[0]
        entry_method = self.method(
            "entry:int(int)",
            entry,
            (
                RepositoryCall(
                    line=entry.start_line,
                    name="missing",
                    method_full_name="<unknown>.missing",
                    dispatch_type="STATIC_DISPATCH",
                ),
            ),
        )
        index = self.StaticIndex([entry_method], [entry])
        self.assertEqual(
            [],
            discover_candidates("sample", index, entry_method),
        )


class ParsingRegressionTests(unittest.TestCase):
    def test_header_language_uses_explicit_tu_hint(self):
        functions = parse_functions(
            "sample.h",
            "class Foo { public: int value() { return 1; } };",
            language_hint="cpp",
        )
        self.assertEqual("cpp", functions[0].language)

    def test_cpp_qualified_method_name_uses_terminal_identifier(self):
        functions = parse_functions(
            "sample.cpp",
            """
            class Foo { public: int bar(int x); };
            int Foo::bar(int x) { return x; }
            """,
        )
        self.assertEqual(["bar"], [function.name for function in functions])
        self.assertEqual("cpp", functions[0].language)

    def test_macro_embedded_kernel_function_is_not_indexed_as_host_c(self):
        functions = parse_functions(
            "accelerate-private.h",
            r"""
            #define STRINGIFY(...) #__VA_ARGS__
            void host(double x) { (void) x; }
            const char *kernels =
              STRINGIFY(
                inline float4 ConvertHSBToRGB(const float4 pixel)
                {
                    return pixel;
                }
              );
            """,
        )
        self.assertEqual(["host"], [function.name for function in functions])

    def test_preprocessor_branch_host_functions_remain_indexed(self):
        functions = parse_functions(
            "variants.c",
            """
            #ifdef FEATURE
            int inner(int x) { return x; }
            #else
            int inner(int x) { return x + 1; }
            #endif
            """,
        )
        self.assertEqual(["inner", "inner"], [function.name for function in functions])

    def test_function_pointer_parameter_is_marked_indirect(self):
        function = parse_functions(
            "sample.c",
            "int entry(int (*cb)(int), int x) { return cb(x); }",
        )[0]
        self.assertTrue(function.has_indirect_calls())


    def test_direct_call_assignment_records_result(self):
        function = parse_functions(
            "sample.c",
            """
            void *entry(unsigned long n)
            {
                void *tmp = malloc(n);
                return tmp;
            }
            """,
        )[0]
        call = next(call for call in function.calls() if call.name == "malloc")
        self.assertEqual("tmp", call.result)
        self.assertFalse(call.returned)

    def test_nested_call_expression_is_not_direct_assignment_result(self):
        function = parse_functions(
            "sample.c",
            """
            void *entry(unsigned long n)
            {
                void *tmp = malloc(n) + 1;
                return tmp;
            }
            """,
        )[0]
        call = next(call for call in function.calls() if call.name == "malloc")
        self.assertIsNone(call.result)

    def test_local_function_pointer_is_marked_indirect(self):
        function = parse_functions(
            "sample.c",
            """
            int target(int x) { return x; }
            int entry(int x)
            {
                int (*cb)(int) = target;
                return cb(x);
            }
            """,
        )[1]
        self.assertTrue(function.has_indirect_calls())


class Z3ReasonerTests(unittest.TestCase):
    def test_known_local_capacity_mismatch_is_feasible(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, unsigned long len)
            {
                char buf[8];
                memcpy(buf, src, len);
            }
            """,
        )[0]
        result = analyze(entry)
        self.assertEqual("VULNERABLE", result.verdict)

    def test_matching_guard_makes_access_safe_but_function_abstains(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, unsigned long len)
            {
                char buf[8];
                if (len > 8)
                    return;
                memcpy(buf, src, len);
            }
            """,
        )[0]
        result = analyze(entry)
        write = next(
            access
            for access in result.constraint_result["accesses"]
            if access["access_kind"] == "WRITE"
        )
        self.assertEqual("SAFE", write["status"])
        self.assertEqual("UNKNOWN", result.verdict)

    def test_nested_return_does_not_create_false_continuation_constraint(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, unsigned long len, int x, int y)
            {
                char buf[8];
                if (x > 0) {
                    if (y > 0)
                        return;
                }
                memcpy(buf, src, len);
            }
            """,
        )[0]
        line = entry.start_line + 7
        constraints = entry.continuation_constraints_before(line)
        self.assertNotIn("x<=0", constraints)

    def test_enclosing_branch_condition_is_a_path_constraint(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, unsigned long len)
            {
                char buf[8];
                if (len <= 8) {
                    memcpy(buf, src, len);
                }
            }
            """,
        )[0]
        line = entry.start_line + 4
        self.assertIn("len<=8", entry.continuation_constraints_before(line))

    def test_relevant_unknown_call_guard_causes_abstention(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, int len)
            {
                char buf[8];
                if (len < 0)
                    report_and_maybe_abort();
                memcpy(buf, src, len);
            }
            """,
        )[0]
        result = analyze(entry)
        self.assertEqual("UNKNOWN", result.verdict)

    def test_compound_early_exit_constraints_use_de_morgan(self):
        entry = parse_functions(
            "entry.c",
            """
            int entry(const char *src, unsigned long slen)
            {
                char buf[16];
                if (slen == 0 || slen >= sizeof(buf))
                    return 0;
                memcpy(buf, src, slen);
                return 1;
            }
            """,
        )[0]
        constraints = entry.continuation_constraints_before(entry.start_line + 5)
        self.assertIn("slen!=0", constraints)
        self.assertIn("slen<sizeof(buf)", constraints)

    def test_reaching_definitions_keep_only_latest_sequential_value(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src)
            {
                char buf[8];
                int n = 4;
                n = 6;
                memcpy(buf, src, n);
            }
            """,
        )[0]
        relations = dict(entry.value_relations_before(entry.start_line + 5))
        self.assertEqual("6", relations["n"])

    def test_branch_assignment_is_not_merged_into_reaching_definition(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, int flag)
            {
                char buf[8];
                int n = 4;
                if (flag > 0)
                    n = 100;
                memcpy(buf, src, n);
            }
            """,
        )[0]
        relations = dict(entry.value_relations_before(entry.start_line + 6))
        self.assertEqual("4", relations["n"])

    def test_signed_parameter_name_is_not_matched_by_constraint_substring(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, int n, int length)
            {
                char buf[8];
                if (length < 8) {
                    memcpy(buf, src, n);
                }
            }
            """,
        )[0]
        result = analyze(entry)
        self.assertEqual("UNKNOWN", result.verdict)
        self.assertIn("unconstrained parameter domain", result.reason)

    def test_signed_length_without_domain_is_unknown(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, int len)
            {
                char buf[8];
                memcpy(buf, src, len);
            }
            """,
        )[0]
        result = analyze(entry)
        self.assertEqual("UNKNOWN", result.verdict)
        self.assertIn("unconstrained parameter domain", result.reason)

    def test_unknown_object_capacity_is_unknown(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(struct ctx *ctx, const char *src, unsigned long len)
            {
                memcpy(ctx->data, src, len);
            }
            """,
        )[0]
        result = analyze(entry)
        self.assertEqual("UNKNOWN", result.verdict)
        self.assertIn("object capacity", result.reason)

    def test_unresolved_macro_capacity_is_unknown_not_counterexample(self):
        entry = parse_functions(
            "entry.c",
            """
            int entry(unsigned int i)
            {
                char buf[MAX_ITEMS];
                return buf[i];
            }
            """,
        )[0]
        result = analyze(entry)
        self.assertEqual("UNKNOWN", result.verdict)
        self.assertIn("unresolved compile-time symbol", result.reason)

    def test_min_is_encoded_semantically(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, unsigned long cur_size, unsigned long new_size)
            {
                char *tmp = malloc(new_size);
                memcpy(tmp, src, min(cur_size, new_size));
            }
            """,
        )[0]
        result = analyze(entry)
        writes = [
            access
            for access in result.constraint_result["accesses"]
            if access["access_kind"] == "WRITE"
        ]
        self.assertEqual("SAFE", writes[0]["status"])

    def test_nested_allocator_call_is_not_treated_as_direct_definition(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, unsigned long n)
            {
                char *tmp = malloc(n) + 1;
                memcpy(tmp, src, n);
            }
            """,
        )[0]
        result = analyze(entry)
        writes = [
            access
            for access in result.constraint_result["accesses"]
            if access["access_kind"] == "WRITE"
        ]
        self.assertEqual("UNKNOWN", writes[0]["status"])
        self.assertIn("object capacity", writes[0]["reason"])

    def test_non_allocation_pointer_redefinition_remains_unknown(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, unsigned long new_size)
            {
                char *tmp = malloc(new_size);
                tmp = choose_buffer(tmp);
                memcpy(tmp, src, new_size);
            }
            """,
        )[0]
        result = analyze(entry)
        writes = [
            access
            for access in result.constraint_result["accesses"]
            if access["access_kind"] == "WRITE"
        ]
        self.assertEqual("UNKNOWN", writes[0]["status"])
        self.assertIn("reaching value definition", writes[0]["reason"])

    def test_memcpy_exposes_read_and_write(self):
        entry = parse_functions(
            "entry.c",
            "void entry(char *dst, const char *src, unsigned long n) { memcpy(dst, src, n); }",
        )[0]
        effects = {
            (item["kind"], item["buffer"])
            for item in analyze(entry).as_json()["operations"]
        }
        self.assertIn(("WRITE", "dst"), effects)
        self.assertIn(("READ", "src"), effects)

    def test_memcmp_exposes_both_read_operands(self):
        entry = parse_functions(
            "entry.c",
            "int entry(const char *a, const char *b) { return memcmp(a, b, 4); }",
        )[0]
        reads = [
            item["buffer"]
            for item in analyze(entry).as_json()["operations"]
            if item["kind"] == "READ" and item["callee"] == "memcmp"
        ]
        self.assertEqual(["a", "b"], reads)

    def test_array_use_does_not_overwrite_declared_capacity(self):
        entry = parse_functions(
            "entry.c",
            """
            int entry(unsigned int index)
            {
                char buf[8];
                if (index >= 8)
                    return 0;
                buf[index] = 0;
                return buf[0];
            }
            """,
        )[0]
        result = analyze(entry)
        ast_accesses = [
            access
            for access in result.constraint_result["accesses"]
            if access["buffer"].startswith("buf+")
        ]
        self.assertTrue(ast_accesses)
        self.assertTrue(all(access["status"] == "SAFE" for access in ast_accesses))

    def test_local_array_byte_and_element_capacity_are_distinct(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const unsigned char *src)
            {
                uint16_t buf[8];
                memcpy(buf, src, 16);
                buf[7] = 0;
            }
            """,
        )[0]
        result = analyze(entry)
        accesses = result.constraint_result["accesses"]
        write_memcpy = next(
            access
            for access in accesses
            if access["access_kind"] == "WRITE" and access["extent"] == "16"
        )
        subscript = next(
            access
            for access in accesses
            if access["access_kind"] == "WRITE" and access["buffer"] == "buf+(7)"
        )
        self.assertEqual("SAFE", write_memcpy["status"])
        self.assertEqual("SAFE", subscript["status"])

    def test_direct_pointer_dereference_is_explicit_access(self):
        entry = parse_functions(
            "entry.c",
            "int entry(int *p) { return *p; }",
        )[0]
        operations = analyze(entry).as_json()["operations"]
        self.assertTrue(
            any(item["callee"] == "AST_DEREF" and item["buffer"] == "p" for item in operations)
        )

    def test_strcpy_literal_has_concrete_extent(self):
        entry = parse_functions(
            "entry.c",
            'void entry(void) { char buf[8]; strcpy(buf, "abc"); }',
        )[0]
        writes = [
            operation
            for operation in analyze(entry).as_json()["operations"]
            if operation["callee"] == "strcpy" and operation["kind"] == "WRITE"
        ]
        self.assertEqual("4", writes[0]["extent"])

    def test_variant_specific_summary_does_not_propagate(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, unsigned long n)
            {
                char buf[8];
                unsigned long len = identity(n);
                memcpy(buf, src, len);
            }
            """,
        )[0]
        summary = {"kind": "VALUE", "target": "return", "expression": "arg0"}
        validation = Validation(
            "sample",
            "identity",
            "identity.c",
            10,
            summary,
            True,
            "validated",
        )
        result = analyze(
            entry,
            [validation],
            variant_counts={("identity.c", "identity"): 2},
        )
        self.assertEqual("UNKNOWN", result.verdict)

    def test_validated_value_summary_enters_caller_constraints(self):
        entry = parse_functions(
            "entry.c",
            """
            void entry(const char *src, unsigned long n)
            {
                char buf[8];
                unsigned long len = identity(n);
                memcpy(buf, src, len);
            }
            """,
        )[0]
        validation = Validation(
            "sample",
            "identity",
            "identity.c",
            1,
            {"kind": "VALUE", "target": "return", "expression": "arg0"},
            True,
            "validated",
        )
        result = analyze(entry, [validation])
        self.assertEqual("VULNERABLE", result.verdict)


if __name__ == "__main__":
    unittest.main()
