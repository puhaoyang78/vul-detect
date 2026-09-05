from __future__ import annotations

from . import semantics
from .joern import JoernError
from .standard_semantics import summaries_for_function


def _static_standard_validation(candidate, clean):
    function = candidate.function
    if clean not in summaries_for_function(function):
        return None
    error = semantics._schema_error(clean, len(function.parameters))
    if error is not None:
        return None
    if clean.get("kind") in {"READ", "WRITE"}:
        root = semantics._buffer_root_index(clean.get("buffer", ""))
        if (
            root is None
            or root >= len(function.parameter_types)
            or not semantics._type_definitely_pointer(function.parameter_types[root])
        ):
            return None
    return semantics.Validation(
        candidate.sample_key,
        function.name,
        function.path,
        function.start_line,
        clean,
        True,
        "standard API role verified from the statically resolved wrapper call",
        variant_group=candidate.variant_group,
        variant_count=candidate.variant_count,
    )


def validate_summary(candidate, summary, joern, callee_summaries=None):
    function = candidate.function
    clean = semantics.canonicalize_summary(function, summary)
    static = _static_standard_validation(candidate, clean)
    if static is not None:
        return static
    try:
        return semantics.validate_summary(
            candidate,
            clean,
            joern=joern,
            callee_summaries=callee_summaries,
        )
    except JoernError as error:
        return semantics.Validation(
            candidate.sample_key,
            function.name,
            function.path,
            function.start_line,
            clean,
            False,
            f"candidate-local Joern validation unavailable: {error}",
            variant_group=candidate.variant_group,
            variant_count=candidate.variant_count,
        )
