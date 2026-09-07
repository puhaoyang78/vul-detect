import unittest

from semantic_demo.joern import RepositoryMethod
from semantic_demo.semantics import Candidate, validate_summary
from semantic_demo.source import parse_functions
from semantic_demo.summary_validator import SummaryValidator


class _Index:
    preprocess_entry = False
    entry_path = ""

    def __init__(self, method):
        self._methods = {method.full_name: method}

    def methods(self):
        return self._methods

    @staticmethod
    def _normalize_repository_path(path):
        return path

    def ensure_available(self):
        return None


class LocalSummaryValidationTests(unittest.TestCase):
    def _validator(
        self,
        source,
        name,
        return_type="void",
        *,
        required_parameters=(),
        require_return=False,
    ):
        function = next(
            item for item in parse_functions("sample.c", source) if item.name == name
        )
        method = RepositoryMethod(
            full_name=name,
            name=name,
            path="sample.c",
            start_line=function.start_line,
            end_line=function.end_line,
            return_type=return_type,
            parameters=function.parameters,
            parameter_types=function.parameter_types,
            calls=(),
        )
        candidate = Candidate(
            "S01",
            function,
            (function.start_line,),
            method_full_name=name,
            required_parameters=tuple(required_parameters),
            require_return=require_return,
        )
        return candidate, SummaryValidator(_Index(method))

    def test_parameter_reaches_call_through_local_assignment(self):
        candidate, validator = self._validator(
            """
void wrapper(char *dst, unsigned long n)
{
    unsigned long len = n;
    custom_write(dst, len);
}
""",
            "wrapper",
            required_parameters=(0, 1),
        )
        facts = validator.facts(candidate)
        call = next(item for item in facts.call_list() if item.name == "custom_write")
        self.assertTrue(facts.parameter_reaches(0, call, 0))
        self.assertTrue(facts.parameter_reaches(1, call, 1))

    def test_value_summary_accepts_parameter_return_through_assignment(self):
        candidate, validator = self._validator(
            """
unsigned long identity(unsigned long n)
{
    unsigned long value = n;
    return value;
}
""",
            "identity",
            return_type="unsigned long",
            require_return=True,
        )
        result = validate_summary(
            candidate,
            {"kind": "VALUE", "target": "return", "expression": "arg0"},
            joern=validator,
        )
        self.assertTrue(result.passed)
        self.assertEqual("VERIFIED", result.status)

    def test_parameter_field_allocation_is_verified_from_source(self):
        candidate, validator = self._validator(
            """
struct buffer { char *data; };
void init_buffer(struct buffer *buf, unsigned long n)
{
    buf->data = malloc(n);
}
""",
            "init_buffer",
            required_parameters=(0, 1),
        )
        result = validate_summary(
            candidate,
            {"kind": "ALLOC", "buffer": "arg0->data", "size": "arg1"},
            joern=validator,
        )
        self.assertTrue(result.passed)
        self.assertEqual("VERIFIED", result.status)

    def test_source_resolved_candidate_does_not_require_joern_identity(self):
        function = parse_functions(
            "macro.h",
            "void init(char *dst, unsigned long n) { memset(dst, 0, n); }",
        )[0]
        candidate = Candidate(
            "S01",
            function,
            (12,),
            required_parameters=(0, 1),
            resolution="source-function",
        )
        validator = SummaryValidator(_Index(RepositoryMethod(
            "unrelated", "unrelated", "other.c", 1, 1, "void", (), (), ()
        )))
        result = validate_summary(
            candidate,
            {"kind": "WRITE", "buffer": "arg0", "length": "arg1"},
            joern=validator,
        )
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
