import unittest
from types import SimpleNamespace

from semantic_demo.candidate_graph import discover_relevant_candidates
from semantic_demo.joern import RepositoryCall, RepositoryMethod
from semantic_demo.source import parse_functions
from semantic_demo.symbol_resolution import SymbolResolver


class _SearchRepository:
    revision = "deadbeef"

    def __init__(self, sources):
        self.sources = sources

    def read_blob(self, path):
        return self.sources[path]

    def _git(self, *args, check=True):
        symbol = args[3] if len(args) > 3 else ""
        matches = [
            path for path, text in self.sources.items() if symbol in text
        ]
        stdout = "".join(f"{self.revision}:{path}\n" for path in matches)
        return SimpleNamespace(returncode=0 if matches else 1, stdout=stdout, stderr="")


class _ResolverIndex:
    scopes = ("src", "include")
    context_paths = ()

    def __init__(self, repository, methods=()):
        self.repository = repository
        self._methods = {method.full_name: method for method in methods}

    def methods(self):
        return self._methods

    def callee_methods(self, _call):
        return []


class _CandidateRepository:
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
        matches = [
            function
            for function in parse_functions(
                path, self.sources[path], language_hint=language_hint
            )
            if function.name == name and function.start_line == start_line
        ]
        if len(matches) != 1:
            raise AssertionError((path, name, start_line, len(matches)))
        return matches[0]


class _CandidateIndex:
    def __init__(self, repository, methods):
        self.repository = repository
        self._methods = {method.full_name: method for method in methods}

    def methods(self):
        return self._methods

    def callee_methods(self, call):
        target = self._methods.get(call.method_full_name)
        return [target] if target is not None else []


class LayeredResolutionTests(unittest.TestCase):
    def test_unique_source_function_resolves_when_index_binding_is_missing(self):
        caller = parse_functions(
            "src/entry.c",
            "void entry(char *p, unsigned long n) { helper(p, n); }",
        )[0]
        repository = _SearchRepository({
            "src/entry.c": caller.translation_unit,
            "src/helper.c": "void helper(char *p, unsigned long n) { memset(p, 0, n); }",
        })
        resolver = SymbolResolver(_ResolverIndex(repository))
        source_call = caller.calls()[0]
        targets = resolver.resolve(None, source_call, caller, "c")
        self.assertEqual(1, len(targets))
        self.assertEqual("source-function", targets[0].resolution)
        self.assertEqual("helper", targets[0].method.name)

    def test_function_like_macro_resolves_when_no_method_exists(self):
        caller = parse_functions(
            "src/entry.c",
            "void entry(char *p, unsigned long n) { init_buffer(p, n); }",
        )[0]
        repository = _SearchRepository({
            "src/entry.c": caller.translation_unit,
            "include/buffer.h": "#define init_buffer(buf, n) memset((buf), 0, (n))\n",
        })
        resolver = SymbolResolver(_ResolverIndex(repository))
        source_call = caller.calls()[0]
        targets = resolver.resolve(None, source_call, caller, "c")
        self.assertEqual(1, len(targets))
        self.assertEqual("source-macro", targets[0].resolution)
        self.assertEqual("init_buffer", targets[0].method.name)

    def test_ambiguous_source_fallback_is_not_guessed(self):
        caller = parse_functions(
            "src/entry.c", "void entry(char *p) { helper(p); }"
        )[0]
        repository = _SearchRepository({
            "src/entry.c": caller.translation_unit,
            "lib/a.c": "void helper(char *p) { p[0] = 0; }",
            "other/b.c": "void helper(char *p) { p[0] = 1; }",
        })
        resolver = SymbolResolver(_ResolverIndex(repository))
        targets = resolver.resolve(None, caller.calls()[0], caller, "c")
        self.assertEqual((), targets)


class FirstHopRecallTests(unittest.TestCase):
    def test_direct_pointer_helper_is_retained_without_existing_memory_seed(self):
        entry_source = "void entry(char *p, unsigned long n) { init_buffer(p, n); }\n"
        helper_source = "void init_buffer(char *p, unsigned long n) { memset(p, 0, n); }\n"
        entry = parse_functions("entry.c", entry_source)[0]
        helper = parse_functions("helper.c", helper_source)[0]
        call = entry.calls()[0]
        entry_method = RepositoryMethod(
            "entry", "entry", "entry.c", entry.start_line, entry.end_line,
            "void", entry.parameters, entry.parameter_types,
            (RepositoryCall(call.line, "init_buffer", "init_buffer", "STATIC_DISPATCH"),),
        )
        helper_method = RepositoryMethod(
            "init_buffer", "init_buffer", "helper.c",
            helper.start_line, helper.end_line, "void",
            helper.parameters, helper.parameter_types, (),
        )
        repository = _CandidateRepository({
            "entry.c": entry_source,
            "helper.c": helper_source,
        })
        discovery = discover_relevant_candidates(
            "S01",
            _CandidateIndex(repository, [entry_method, helper_method]),
            entry_method,
            entry,
        )
        self.assertEqual(["init_buffer"], [item.function.name for item in discovery.candidates])
        candidate = discovery.candidates[0]
        self.assertIn(0, candidate.required_parameters)
        self.assertEqual(1, discovery.direct_candidates)


if __name__ == "__main__":
    unittest.main()
