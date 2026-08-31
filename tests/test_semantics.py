import unittest

from semantic_demo.analyzer import analyze
from semantic_demo.semantics import Candidate, Validation, validate_summary
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

    def test_source_parameter_name_is_rejected(self):
        result = validate_summary(
            self.candidate,
            {"kind": "WRITE", "buffer": "dst", "length": "arg2"},
        )
        self.assertFalse(result.passed)

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
        self.assertIn("different expression", verdict.reason)


if __name__ == "__main__":
    unittest.main()
