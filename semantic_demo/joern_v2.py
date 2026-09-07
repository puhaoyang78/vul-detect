from __future__ import annotations

from .summary_validator import SummaryValidator


class JoernValidatorV2(SummaryValidator):
    """Compatibility name for the current lightweight summary validator.

    Repository-level Joern remains authoritative for revision binding, method
    identity, and resolved static-call discovery during preflight.  The run
    stage intentionally performs only intraprocedural local reaching-definition
    validation and therefore launches no Joern/Java OSS-dataflow process.
    """

    def __init__(
        self,
        joern_dir=None,
        *,
        java_home=None,
        timeout=None,
        repository_index=None,
    ) -> None:
        if repository_index is None:
            raise ValueError("summary validation requires a completed Joern repository index")
        super().__init__(repository_index)
