from __future__ import annotations

"""Public command entrypoint.

The original implementation is retained in semantic_demo.legacy_cli as an internal
execution engine. User-facing preflight/normalize/run commands are routed through
the staged workflow so candidate discovery, checkpoints, and subset updates have a
single set of semantics.
"""

from . import legacy_cli as _legacy


# Re-export internal helpers used by workflow/tests, including private helpers.
for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


def _sync_runtime_hooks() -> None:
    """Propagate workflow monkeypatches to the retained detect implementation."""
    for name in ("discover_candidates", "validate_summary", "JoernValidator", "analyze"):
        if name in globals():
            setattr(_legacy, name, globals()[name])


def detect(*args, **kwargs):
    _sync_runtime_hooks()
    return _legacy.detect(*args, **kwargs)


def build_parser():
    from .workflow import build_parser as staged_build_parser

    return staged_build_parser()


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
