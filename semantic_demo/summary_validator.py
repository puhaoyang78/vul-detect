from __future__ import annotations

import re
from dataclasses import dataclass

from .joern import JoernCall, JoernError, JoernFacts, JoernMethodNotFound
from .source import FunctionSource, normalize_expression


_IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
_RETURN = re.compile(r"\breturn\s+([^;]+);", re.MULTILINE)


def _identifiers(expression: str) -> set[str]:
    return set(_IDENTIFIER.findall(normalize_expression(expression)))


def _relation_map(function: FunctionSource, line: int) -> dict[str, str]:
    relations: dict[str, str] = {}
    for left, right in function.value_relations_before(line):
        left_normalized = normalize_expression(left)
        if re.fullmatch(r"[A-Za-z_]\w*", left_normalized):
            relations[left_normalized] = normalize_expression(right)
    return relations


def _dependency_closure(
    expression: str,
    relations: dict[str, str],
) -> set[str]:
    pending = list(_identifiers(expression))
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        replacement = relations.get(name)
        if replacement:
            pending.extend(_identifiers(replacement) - seen)
    return seen


def _resolve_simple_value(expression: str, relations: dict[str, str]) -> str:
    """Resolve only pure identifier-to-identifier assignment chains."""
    current = normalize_expression(expression)
    seen: set[str] = set()
    while re.fullmatch(r"[A-Za-z_]\w*", current) and current not in seen:
        seen.add(current)
        replacement = relations.get(current)
        if replacement is None:
            break
        replacement = normalize_expression(replacement)
        if not re.fullmatch(r"[A-Za-z_]\w*", replacement):
            break
        current = replacement
    return current


def _return_expressions(function: FunctionSource) -> list[str]:
    # FunctionSource has already been structurally parsed. The expression-only
    # extraction here deliberately stays local and does not infer across calls.
    return [
        normalize_expression(match.group(1))
        for match in _RETURN.finditer(function.text)
    ]


@dataclass(frozen=True)
class _Identity:
    path: str
    name: str
    start_line: int
    end_line: int


class SummaryValidator:
    """Validate LLM summaries using Joern identity plus local source dataflow.

    Joern remains responsible for repository/revision binding, exact method
    identity, and resolved static-call discovery during preflight. Summary
    validation itself is intentionally intraprocedural: parameter-to-argument
    and parameter-to-return dependencies are reconstructed from the same
    Tree-sitter reaching definitions used by candidate slicing. This avoids
    invoking Joern OSS dataflow for every real-world translation unit.
    """

    backend = "joern-index+local-flow"
    timeout = None

    def __init__(self, repository_index) -> None:
        self.repository_index = repository_index
        self._cache: dict[str, JoernFacts] = {}

    def ensure_available(self) -> None:
        # A valid preflight/index is the static-analysis prerequisite. This
        # performs only availability checks; it does not launch OSS dataflow.
        self.repository_index.ensure_available()

    @staticmethod
    def _key(function: FunctionSource) -> str:
        return (
            f"{function.path}:{function.name}:"
            f"{function.start_line}:{function.end_line}"
        )

    def _identity(self, candidate) -> _Identity:
        if not candidate.method_full_name:
            raise JoernMethodNotFound(
                "candidate has no Joern method identity: "
                f"{candidate.function.path}:{candidate.function.name}"
            )
        method = self.repository_index.methods().get(candidate.method_full_name)
        if method is None:
            raise JoernMethodNotFound(
                f"method_not_found:{candidate.method_full_name}"
            )
        function = candidate.function
        same_path = (
            self.repository_index._normalize_repository_path(method.path)
            == self.repository_index._normalize_repository_path(function.path)
        )
        if not same_path or method.name != function.name:
            raise JoernError(
                f"candidate/Joern identity mismatch: "
                f"{function.path}:{function.name} != {method.path}:{method.name}"
            )
        # Preprocessed entry methods can have recovered original source ranges;
        # the manifest fingerprint already binds the exact recovered source.
        if (
            not self.repository_index.preprocess_entry
            or method.path != self.repository_index.entry_path
        ):
            if (
                method.start_line != function.start_line
                or method.end_line != function.end_line
            ):
                raise JoernError(
                    f"candidate/Joern range mismatch: "
                    f"{function.path}:{function.name}@"
                    f"{function.start_line}-{function.end_line} != "
                    f"{method.start_line}-{method.end_line}"
                )
        return _Identity(
            method.path,
            method.name,
            method.start_line,
            method.end_line,
        )

    def facts(self, candidate) -> JoernFacts:
        function = candidate.function
        key = self._key(function)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        self._identity(candidate)
        facts = JoernFacts(
            parameters={
                index: (name, function.parameter_types[index])
                for index, name in enumerate(function.parameters)
            }
        )

        direct_calls = [
            call for call in function.calls() if not call.indirect
        ]
        occurrence: dict[tuple[int, str], int] = {}
        for call in direct_calls:
            occurrence_key = (call.line, call.name)
            ordinal = occurrence.get(occurrence_key, 0)
            occurrence[occurrence_key] = ordinal + 1
            call_id = f"source:{call.line}:{call.name}:{ordinal}"
            arguments = {
                index: normalize_expression(argument)
                for index, argument in enumerate(call.arguments)
            }
            joern_call = JoernCall(
                line=call.line,
                name=call.name,
                arguments=arguments,
                call_id=call_id,
                code=call.code,
            )
            facts.calls[call_id] = joern_call

            relations = _relation_map(function, call.line)
            for argument_index, argument in arguments.items():
                dependencies = _dependency_closure(argument, relations)
                for parameter_index, parameter in enumerate(
                    function.parameters
                ):
                    if parameter in dependencies:
                        facts.flows.add(
                            (parameter_index, call_id, argument_index)
                        )

        returns = _return_expressions(function)
        full_relations = _relation_map(function, function.end_line + 1)
        for expression in returns:
            facts.returns.append(expression)
            resolved = _resolve_simple_value(expression, full_relations)
            if resolved != expression:
                facts.returns.append(resolved)
            dependencies = _dependency_closure(expression, full_relations)
            for parameter_index, parameter in enumerate(function.parameters):
                if parameter in dependencies:
                    facts.return_flows.add(parameter_index)

        self._cache[key] = facts
        return facts
