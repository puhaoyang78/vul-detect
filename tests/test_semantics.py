import unittest

from semantic_demo.analyzer import Operation, analyze
from semantic_demo.z3_reasoner import reason_memory_safety
from semantic_demo.joern import JoernCall, JoernError, JoernFacts, JoernMethodNotFound
from semantic_demo.semantics import (
    Candidate,
    Validation,
    _family_role_indices,
    _response_content,
    _validate_by_composition,
    validate_summary,
)
from semantic_demo.source import FunctionSource, parse_functions


class SemanticValidationTests(unittest.TestCase):
    def setUp(self):
        source = """
        void copy_wrap(char *dst, const char *src, unsigned long len)
        {
            memcpy(dst, src, len);
        }
        """
        self.function = parse_functions("wrapper.c", source)[0]
        self.candidate = Candidate("sample", self.function, (10,))

    def test_write_buffer_and_length_reach_same_sink(self):
        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
        )
        self.assertTrue(result.passed)

    def test_wrong_length_is_rejected(self):
        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg1"},
        )
        self.assertFalse(result.passed)

    def test_source_parameter_name_is_canonicalized(self):
        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "dst", "length": "len"},
        )
        self.assertTrue(result.passed)
        self.assertEqual("arg0", result.summary["buffer"])
        self.assertEqual("arg2", result.summary["length"])

    def test_read_buffer_and_length_reach_same_source(self):
        source = """
        int send_wrap(int fd, const char *src, unsigned long len)
        {
            return write(fd, src, len);
        }
        """
        function = parse_functions("reader.c", source)[0]
        candidate = Candidate("sample", function, (30,))
        result = validate_summary(
            candidate,
            {"kind": "READ", "buffer": "arg1", "length": "arg2"},
        )
        self.assertTrue(result.passed)

    def test_value_return_relation(self):
        source = """
        unsigned long identity(unsigned long len)
        {
            return len;
        }
        """
        function = parse_functions("value.c", source)[0]
        candidate = Candidate("sample", function, (40,))
        result = validate_summary(
            candidate,
            {"kind": "VALUE", "target": "return", "expression": "arg0"},
        )
        self.assertTrue(result.passed)

    def test_guard_must_exist_in_candidate(self):
        guard = parse_functions(
            "guard.c",
            "int valid(unsigned long capacity, unsigned long length)"
            "{ return length <= capacity; }",
        )[0]
        candidate = Candidate("sample", guard, (20,))
        self.assertTrue(
            validate_summary(
                candidate, {"kind": "GUARD", "relation": "arg1 <= arg0"}
            ).passed
        )
        self.assertFalse(
            validate_summary(
                candidate, {"kind": "GUARD", "relation": "arg1 < arg0"}
            ).passed
        )

    def test_deep_ast_does_not_exhaust_python_recursion(self):
        depth = 1100
        source = (
            "int deep(void) {"
            + "{" * depth
            + "return 0;"
            + "}" * depth
            + "}"
        )

        functions = parse_functions("deep.c", source)

        self.assertEqual(["deep"], [function.name for function in functions])


    def test_scalar_parameter_cannot_be_memory_buffer(self):
        source = """
        void bad_wrap(int fd, unsigned long len)
        {
            write(fd, (const void *)len, 4);
        }
        """
        function = parse_functions("bad.c", source)[0]
        candidate = Candidate("sample", function, (50,))
        result = validate_summary(
            candidate,
            {"kind": "READ", "buffer": "arg1", "length": "4"},
        )
        self.assertFalse(result.passed)
        self.assertIn("not pointer-like", result.reason)

    def test_io_family_roles_preserve_direction(self):
        self.assertEqual(((1,), (2,)), _family_role_indices("socket_recv", "WRITE", 3))
        self.assertEqual(((1,), (2,)), _family_role_indices("socket_send", "READ", 3))
        self.assertEqual(((), ()), _family_role_indices("socket_recv", "READ", 3))
        self.assertEqual(((), ()), _family_role_indices("socket_send", "WRITE", 3))

    def test_missing_joern_method_rejects_only_the_summary(self):
        class MissingMethodValidator:
            def facts(self, _candidate):
                raise JoernMethodNotFound("method_not_found:copy_wrap")

        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            joern=MissingMethodValidator(),
        )
        self.assertFalse(result.passed)
        self.assertIn("method_not_found:copy_wrap", result.reason)

    def test_joern_infrastructure_error_still_propagates(self):
        class BrokenValidator:
            def facts(self, _candidate):
                raise JoernError("launcher failed")

        with self.assertRaisesRegex(JoernError, "launcher failed"):
            validate_summary(
                self.candidate,
                {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
                joern=BrokenValidator(),
            )

    def test_response_content_accepts_final_answer(self):
        result = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"summaries":[]}',
                        "reasoning_content": "internal reasoning",
                    },
                }
            ]
        }
        self.assertEqual('{"summaries":[]}', _response_content(result))

    def test_response_content_rejects_reasoning_without_final_answer(self):
        result = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "", "reasoning_content": "unfinished"},
                }
            ],
            "usage": {"completion_tokens": 384},
        }
        with self.assertRaisesRegex(ValueError, "finish_reason='length'"):
            _response_content(result)


class CompositionalValidationTests(unittest.TestCase):
    def test_parent_write_summary_composes_from_validated_callee(self):
        function = parse_functions(
            "wrapper.c",
            """
            void copy_outer(char *dst, const char *src, unsigned long len)
            {
                copy_inner(dst, src, len);
            }
            """,
        )[0]
        candidate = Candidate("sample", function, (10,))
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

        class FakeValidator:
            def facts(self, _candidate):
                return facts

        passed, reason = _validate_by_composition(
            candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg2"},
            FakeValidator(),
            {
                "copy_inner": [
                    {"kind": "WRITE", "buffer": "arg0", "length": "arg2"}
                ]
            },
        )
        self.assertTrue(passed)
        self.assertIn("composition verified write", reason)


class Z3ReasonerTests(unittest.TestCase):
    def test_capacity_mismatch_is_feasible(self):
        entry = parse_functions(
            "entry.c",
            """
            int entry(char *buf, unsigned long buflen, struct result res)
            {
                unsigned long acl_len = res.len - 4;
                if (acl_len > buflen)
                    return -1;
                copy_wrap(buf, res.pages, 0, res.len);
                return 0;
            }
            """,
        )[0]
        result = reason_memory_safety(
            entry,
            [
                Operation(
                    "WRITE",
                    "copy_wrap",
                    "buf",
                    "res.len",
                    entry.start_line + 5,
                    True,
                )
            ],
        )
        self.assertEqual("POTENTIAL_VIOLATION", result.status)
        self.assertIn("res.len <= buflen", result.reason)

    def test_unrelated_guard_is_not_used_as_access_bound(self):
        entry = parse_functions(
            "entry.c",
            """
            int entry(char *buf, unsigned long buflen, struct result res)
            {
                unsigned long npages = (res.len + 4095) >> 12;
                unsigned long acl_len = res.len - res.offset;
                if (npages > 1)
                    return -1;
                if (acl_len > buflen)
                    return -1;
                copy_wrap(buf, src, res.len);
                return 0;
            }
            """,
        )[0]
        result = reason_memory_safety(
            entry,
            [
                Operation(
                    "WRITE",
                    "copy_wrap",
                    "buf",
                    "res.len",
                    entry.start_line + 7,
                    True,
                )
            ],
        )
        self.assertEqual("POTENTIAL_VIOLATION", result.status)
        self.assertIn("res.len<=buflen", result.reason.replace(" ", ""))
        self.assertNotIn("res.len<=1", result.reason.replace(" ", ""))

    def test_matching_guard_proves_bounds(self):
        entry = parse_functions(
            "entry.c",
            """
            int entry(char *buf, unsigned long buflen, unsigned long len)
            {
                if (len > buflen)
                    return -1;
                copy_wrap(buf, src, len);
                return 0;
            }
            """,
        )[0]
        result = reason_memory_safety(
            entry,
            [
                Operation(
                    "WRITE",
                    "copy_wrap",
                    "buf",
                    "len",
                    entry.start_line + 4,
                    True,
                )
            ],
        )
        self.assertEqual("SAFE", result.status)

    def test_signed_length_without_guard_is_feasible(self):
        entry = parse_functions(
            "entry.c",
            """
            int entry(char *buf, int len)
            {
                copy_wrap(buf, src, len);
                return 0;
            }
            """,
        )[0]
        result = reason_memory_safety(
            entry,
            [
                Operation(
                    "WRITE",
                    "copy_wrap",
                    "buf",
                    "len",
                    entry.start_line + 2,
                    True,
                )
            ],
        )
        self.assertEqual("POTENTIAL_VIOLATION", result.status)
        self.assertIn("len >= 0", result.reason)

    def test_unknown_capacity_is_reported(self):
        entry = parse_functions(
            "entry.c",
            """
            int entry(struct ctx *ctx, unsigned long len)
            {
                copy_wrap(ctx->data, src, len);
                return 0;
            }
            """,
        )[0]
        result = reason_memory_safety(
            entry,
            [
                Operation(
                    "WRITE",
                    "copy_wrap",
                    "ctx->data",
                    "len",
                    entry.start_line + 2,
                    True,
                )
            ],
        )
        self.assertEqual("UNKNOWN", result.status)
        self.assertIn("capacity/valid extent is unknown", result.reason)


class PropagationTests(unittest.TestCase):
    def test_custom_write_exposes_capacity_mismatch(self):
        entry = parse_functions(
            "entry.c",
            """
            int entry(char *buf, unsigned long buflen, struct result res)
            {
                unsigned long acl_len = res.len - 4;
                if (acl_len > buflen)
                    return -1;
                copy_wrap(buf, res.pages, 0, res.len);
                return 0;
            }
            """,
        )[0]
        validation = Validation(
            "sample",
            "copy_wrap",
            "wrapper.c",
            {"kind": "WRITE", "buffer": "arg0", "length": "arg3"},
            True,
            "validated",
        )
        verdict = analyze(entry, [validation], proposed=True)
        self.assertEqual("VULNERABLE", verdict.verdict)
        self.assertIn("res.len <= buflen", verdict.reason)


if __name__ == "__main__":
    unittest.main()
