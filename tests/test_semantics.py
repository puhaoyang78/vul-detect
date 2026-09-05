import unittest

from semantic_demo.joern import JoernCall, JoernFacts, JoernError, JoernMethodNotFound, JoernTimeout
from semantic_demo.semantics import (
    Candidate,
    _extract_json_object,
    _response_content,
    _validate_by_composition,
    candidate_validation_error,
    validate_summary,
)
from semantic_demo.source import FunctionSource, parse_functions


class StaticFactsValidator:
    def __init__(self, facts):
        self._facts = facts

    def facts(self, _candidate):
        return self._facts


def copy_facts(function):
    call = JoernCall(
        line=function.start_line + 2,
        name="memcpy",
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


class CandidateBoundaryTests(unittest.TestCase):
    def test_signature_annotation_error_does_not_skip_recoverable_body(self):
        function = FunctionSource(
            path="annotated.c",
            name="annotated",
            text="RZ_API int annotated(RZ_NONNULL char *p) { return p[0]; }",
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
        self.candidate = Candidate("sample", self.function, (1,))
        self.validator = StaticFactsValidator(copy_facts(self.function))

    def test_write_buffer_and_length_are_verified(self):
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

    def test_source_parameter_names_are_canonicalized(self):
        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "dst", "length": "len"},
            joern=self.validator,
        )
        self.assertTrue(result.passed)
        self.assertEqual("arg0", result.summary["buffer"])
        self.assertEqual("arg2", result.summary["length"])

    def test_non_pointer_buffer_summary_is_rejected(self):
        function = parse_functions(
            "bad.c", "void bad(int value, unsigned long n) { write(1, &value, n); }"
        )[0]
        result = validate_summary(
            Candidate("sample", function, (1,)),
            {"kind": "READ", "buffer": "arg0", "length": "arg1"},
            joern=StaticFactsValidator(JoernFacts(parameters={}, calls={}, flows=set())),
        )
        self.assertFalse(result.passed)
        self.assertIn("not pointer-like", result.reason)

    def test_method_not_found_rejects_summary(self):
        class Missing:
            def facts(self, _candidate):
                raise JoernMethodNotFound("method_not_found")

        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            joern=Missing(),
        )
        self.assertFalse(result.passed)
        self.assertIn("method_not_found", result.reason)

    def test_timeout_rejects_summary(self):
        class Timeout:
            def facts(self, _candidate):
                raise JoernTimeout("timed out")

        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            joern=Timeout(),
        )
        self.assertFalse(result.passed)
        self.assertIn("timed out", result.reason)

    def test_candidate_local_joern_error_rejects_summary(self):
        class Broken:
            def facts(self, _candidate):
                raise JoernError("TU parse failed")

        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            joern=Broken(),
        )
        self.assertFalse(result.passed)
        self.assertIn("candidate-local", result.reason)

    def test_exact_value_return_is_verified(self):
        function = parse_functions(
            "value.c", "unsigned long identity(unsigned long len) { return len; }"
        )[0]
        facts = JoernFacts(
            parameters={0: ("len", "unsigned long")},
            calls={},
            flows=set(),
            returns=["return len;"],
            return_flows={0},
        )
        result = validate_summary(
            Candidate("sample", function, (1,)),
            {"kind": "VALUE", "target": "return", "expression": "arg0"},
            joern=StaticFactsValidator(facts),
        )
        self.assertTrue(result.passed)


class CompositionTests(unittest.TestCase):
    def test_parent_write_composes_from_unique_callee(self):
        function = parse_functions(
            "wrapper.c",
            "void outer(char *dst, const char *src, unsigned long len) { inner(dst, src, len); }",
        )[0]
        call = JoernCall(
            line=function.start_line,
            name="inner",
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
            Candidate("sample", function, (1,)),
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            StaticFactsValidator(facts),
            {
                ("inner.c", "inner"): [
                    {"kind": "WRITE", "buffer": "arg0", "length": "arg2"}
                ]
            },
        )
        self.assertTrue(passed)
        self.assertIn("composition verified write", reason)

    def test_ambiguous_same_name_callee_does_not_compose(self):
        function = parse_functions(
            "wrapper.c", "void outer(char *p, unsigned long n) { inner(p, n); }"
        )[0]
        call = JoernCall(
            line=function.start_line,
            name="inner",
            arguments={0: "p", 1: "n"},
        )
        facts = JoernFacts(
            parameters={0: ("p", "char *"), 1: ("n", "unsigned long")},
            calls={(call.line, call.name): call},
            flows={(0, call.line, call.name, 0), (1, call.line, call.name, 1)},
        )
        passed, _ = _validate_by_composition(
            Candidate("sample", function, (1,)),
            {"kind": "WRITE", "buffer": "arg0", "length": "arg1"},
            StaticFactsValidator(facts),
            {
                ("a.c", "inner"): [
                    {"kind": "WRITE", "buffer": "arg0", "length": "arg1"}
                ],
                ("b.c", "inner"): [
                    {"kind": "WRITE", "buffer": "arg0", "length": "arg1"}
                ],
            },
        )
        self.assertFalse(passed)


class ResponseParsingTests(unittest.TestCase):
    def test_response_content_accepts_final_answer(self):
        result = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"summaries":[]}', "reasoning_content": "internal"},
            }]
        }
        self.assertEqual('{"summaries":[]}', _response_content(result))

    def test_response_content_rejects_truncation(self):
        result = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "partial", "reasoning_content": ""},
            }],
            "usage": {"completion_tokens": 512},
        }
        with self.assertRaisesRegex(ValueError, "truncated at max_tokens"):
            _response_content(result)

    def test_extract_json_requires_one_object(self):
        self.assertEqual({"summaries": []}, _extract_json_object('{"summaries":[]}'))
        with self.assertRaisesRegex(ValueError, "must be one JSON object"):
            _extract_json_object("[]")
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            _extract_json_object('prefix {"summaries":[]}')


if __name__ == "__main__":
    unittest.main()
