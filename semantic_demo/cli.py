from __future__ import annotations

"""Public command entrypoint for the staged vulnerability workflow."""

import hashlib
from pathlib import Path

from . import legacy_cli as _legacy


for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


_original_analysis_fingerprint = _legacy._analysis_fingerprint


def _analysis_implementation_digest() -> str:
    names = (
        "analyzer.py",
        "z3_reasoner_v2.py",
        "validation_v2.py",
        "joern_v2.py",
        "standard_semantics.py",
    )
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in names:
        path = root / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _analysis_fingerprint(*args, **kwargs):
    base = _original_analysis_fingerprint(*args, **kwargs)
    return hashlib.sha256(
        (base + "\0" + _analysis_implementation_digest()).encode()
    ).hexdigest()


_legacy._analysis_fingerprint = _analysis_fingerprint


_RUNTIME_NAMES = (
    "read_jsonl",
    "write_jsonl",
    "_preflight_samples",
    "_load_entry",
    "_load_repository_index",
    "_entry_from_index",
    "discover_candidates",
    "llm_normalize",
    "load_replay",
    "validate_summary",
    "local_llm_server",
    "LineVulBaseline",
    "JoernValidator",
    "analyze",
)


def _sync_runtime_hooks() -> None:
    for name in _RUNTIME_NAMES:
        if name in globals():
            setattr(_legacy, name, globals()[name])


def detect(*args, **kwargs):
    _sync_runtime_hooks()
    return _legacy.detect(*args, **kwargs)


def normalize_command(*args, **kwargs):
    _sync_runtime_hooks()
    return _legacy.normalize_command(*args, **kwargs)


def preflight_command(*args, **kwargs):
    _sync_runtime_hooks()
    return _legacy.preflight_command(*args, **kwargs)


def run_command(*args, **kwargs):
    _sync_runtime_hooks()
    return _legacy.run_command(*args, **kwargs)


def build_parser():
    from .workflow import build_parser as staged_build_parser

    return staged_build_parser()


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
