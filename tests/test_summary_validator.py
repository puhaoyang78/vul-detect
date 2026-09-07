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
    def _validator(self, source, name, return_type="void"):
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
            "S01", function, (function.start_line,), method_full_name=name
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
        )
        result = validate_summary(
            candidate,
            {"kind": "VALUE", "target": "return", "expression": "arg0"},
            joern=validator,
        )
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
