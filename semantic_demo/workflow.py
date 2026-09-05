from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import tempfile
import urllib.error
from pathlib import Path

from . import cli
from .candidate_graph import (
    CANDIDATE_MANIFEST_VERSION,
    DISCOVERY_POLICY_VERSION,
    candidate_manifest_path,
    candidate_source_fingerprint,
    discover_relevant_candidates,
    load_manifest_candidate,
    read_candidate_manifest,
    write_candidate_manifest,
)
from .normalization_v2 import NORMALIZATION_IMPLEMENTATION_VERSION, llm_normalize
from .semantics import NORMALIZATION_SCHEMA_VERSION, candidate_validation_error


WORKFLOW_PREFLIGHT_VERSION = 2


def _read(path: str | Path) -> list[dict[str, object]]:
    return cli.read_jsonl(path)


def _write(path: str | Path, records) -> None:
    ordered = sorted(
        list(records),
        key=lambda record: (
            str(record.get("sample_key", "")),
            str(record.get("source_path", "")),
            str(record.get("function", "")),
            int(record.get("source_line", 0) or 0),
        ),
    )
    cli.write_jsonl(path, ordered)


def _upsert_by_sample(old_records, new_records, selected_keys: set[str]):
    current_by_sample: dict[str, list[dict[str, object]]] = {}
    for record in new_records:
        current_by_sample.setdefault(str(record.get("sample_key", "")), []).append(record)

    result = [
        record for record in old_records
        if str(record.get("sample_key", "")) not in selected_keys
    ]
    old_selected: dict[str, list[dict[str, object]]] = {}
    for record in old_records:
        key = str(record.get("sample_key", ""))
        if key in selected_keys:
            old_selected.setdefault(key, []).append(record)
    for key in selected_keys:
        result.extend(current_by_sample.get(key, old_selected.get(key, [])))
    return result


def _preflight_checkpoint_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "workflow_preflight.jsonl"


def _preflight_fingerprint(sample: dict[str, object], index) -> str:
    payload = {
        "version": WORKFLOW_PREFLIGHT_VERSION,
        "candidate_manifest_version": CANDIDATE_MANIFEST_VERSION,
        "discovery_policy_version": DISCOVERY_POLICY_VERSION,
        "sample": sample,
        "index_fingerprint": index.fingerprint,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _checkpoint_cache(cache_dir: str | Path) -> dict[str, dict[str, object]]:
    path = _preflight_checkpoint_path(cache_dir)
    if not path.is_file():
        return {}
    return {
        str(record.get("sample_key", "")): record
        for record in _read(path)
        if record.get("version") == WORKFLOW_PREFLIGHT_VERSION
    }


def _load_index(sample: dict[str, object], args):
    return cli._load_repository_index(
        sample,
        joern_dir=args.joern_dir,
        java_home=args.java_home,
        cpg_cache_dir=args.cpg_cache_dir,
    )


def _valid_completed_preflight(sample, index, cached) -> bool:
    if cached is None or cached.get("fingerprint") != _preflight_fingerprint(sample, index):
        return False
    if not index.cpg_path.is_file() or not index.index_path.is_file():
        return False
    if not candidate_manifest_path(index).is_file():
        return False
    try:
        header, candidates = read_candidate_manifest(index)
    except RuntimeError:
        return False
    return len(candidates) == int(header.get("candidate_count", -1))


def preflight(args: argparse.Namespace) -> None:
    samples = _read(args.samples)
    cli.validate_detection_manifest(samples)
    checkpoint_path = _preflight_checkpoint_path(args.cpg_cache_dir)
    # Refresh only invalidates the selected samples. Other sample checkpoints survive.
    checkpoints = _checkpoint_cache(args.cpg_cache_dir)
    failures: list[str] = []

    for position, sample in enumerate(samples, 1):
        key = str(sample["sample_key"])
        print(f"preflight_start={position}/{len(samples)} sample={key}", flush=True)
        try:
            repository, index = _load_index(sample, args)
            cached = checkpoints.get(key)
            if not args.refresh and _valid_completed_preflight(sample, index, cached):
                header, _records = read_candidate_manifest(index)
                print(
                    f"preflight_cached={key} candidates={header['candidate_count']} "
                    f"direct={header['direct_candidates']} recursive={header['recursive_candidates']} "
                    f"unresolved_relevant={header['unresolved_relevant_calls']}",
                    flush=True,
                )
                continue

            method_count = len(index.methods())
            print(f"preflight_index_ready={key} methods={method_count}", flush=True)
            entry_method, entry = cli._entry_from_index(sample, repository, index)
            print(
                f"preflight_entry_ready={key} entry={entry.name}@{entry.start_line}",
                flush=True,
            )
            discovery = discover_relevant_candidates(key, index, entry_method, entry)
            manifest = write_candidate_manifest(index, discovery)
            unrecoverable = sum(
                candidate_validation_error(candidate.function) is not None
                for candidate in discovery.candidates
            )
            checkpoints[key] = {
                "version": WORKFLOW_PREFLIGHT_VERSION,
                "sample_key": key,
                "fingerprint": _preflight_fingerprint(sample, index),
                "candidate_count": len(discovery.candidates),
                "direct_candidates": discovery.direct_candidates,
                "recursive_candidates": discovery.recursive_candidates,
                "expanded_methods": discovery.expanded_methods,
                "unrecoverable_candidates": unrecoverable,
                "unresolved_relevant_calls": discovery.unresolved_relevant_calls,
                "manifest": str(manifest),
            }
            _write(checkpoint_path, checkpoints.values())
            print(
                f"preflight_done={key} candidates={len(discovery.candidates)} "
                f"direct={discovery.direct_candidates} recursive={discovery.recursive_candidates} "
                f"expanded={discovery.expanded_methods} unrecoverable={unrecoverable} "
                f"unresolved_relevant={discovery.unresolved_relevant_calls} "
                f"manifest={manifest}",
                flush=True,
            )
        except Exception as error:
            failures.append(f"{key}: {error}")
            print(f"preflight_failed={key} error={error}", flush=True)

    if failures:
        raise RuntimeError(
            f"preflight failed for {len(failures)} sample(s):\n" + "\n".join(failures)
        )
    print(f"preflight_complete=samples:{len(samples)}", flush=True)


def _normalization_key(record: dict[str, object]):
    return (
        str(record.get("sample_key", "")),
        str(record.get("source_path", "")),
        str(record.get("function", "")),
        int(record.get("source_line", 0)),
    )


def _expected_model(args) -> str:
    if args.llm_backend == "local":
        return Path(args.local_model).stem
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def _record_matches_source(record, manifest_record) -> bool:
    return (
        record.get("schema_version") == NORMALIZATION_SCHEMA_VERSION
        and record.get("normalization_implementation_version")
        == NORMALIZATION_IMPLEMENTATION_VERSION
        and record.get("source_fingerprint") == manifest_record.get("source_fingerprint")
    )


def _record_reusable(record, manifest_record, args, expected_model) -> bool:
    if not _record_matches_source(record, manifest_record):
        return False
    if manifest_record.get("skip_reason") is not None:
        return (
            record.get("normalizer") == "static-skip"
            and record.get("skip_reason") == manifest_record.get("skip_reason")
        )
    return (
        record.get("normalizer") == "relevance-sliced-hybrid"
        and record.get("llm_backend") == args.llm_backend
        and record.get("llm_model") == expected_model
    )


def _record_usable_for_run(record, manifest_record) -> bool:
    if not _record_matches_source(record, manifest_record):
        return False
    if manifest_record.get("skip_reason") is not None:
        return record.get("normalizer") == "static-skip"
    return record.get("normalizer") == "relevance-sliced-hybrid"


def _load_sample_manifest(sample, args):
    _repository, index = _load_index(sample, args)
    checkpoint = _checkpoint_cache(args.cpg_cache_dir).get(str(sample["sample_key"]))
    if not _valid_completed_preflight(sample, index, checkpoint):
        raise RuntimeError(
            f"{sample['sample_key']}: preflight manifest missing or stale; run preflight first"
        )
    return index, read_candidate_manifest(index)


def normalize(args: argparse.Namespace) -> None:
    samples = _read(args.samples)
    cli.validate_detection_manifest(samples)
    selected_keys = {str(sample["sample_key"]) for sample in samples}
    old_records = _read(args.output) if Path(args.output).is_file() else []
    preserved = [
        record for record in old_records
        if str(record.get("sample_key", "")) not in selected_keys
    ]
    old_selected = {
        _normalization_key(record): record
        for record in old_records
        if str(record.get("sample_key", "")) in selected_keys
    }
    expected_model = _expected_model(args)
    existing: dict[tuple[str, str, str, int], dict[str, object]] = {}
    sample_work = []
    pending_total = 0
    candidate_total = 0

    for sample in samples:
        key = str(sample["sample_key"])
        index, (_header, manifest_records) = _load_sample_manifest(sample, args)
        parse_cache = {}
        pending = []
        for manifest_record in manifest_records:
            logical_key = (
                key,
                str(manifest_record["source_path"]),
                str(manifest_record["function"]),
                int(manifest_record["source_line"]),
            )
            cached = old_selected.get(logical_key)
            if (
                not args.refresh
                and cached is not None
                and _record_reusable(cached, manifest_record, args, expected_model)
            ):
                existing[logical_key] = cached
                continue
            candidate = load_manifest_candidate(index, manifest_record, parse_cache)
            pending.append((candidate, manifest_record))
        candidate_total += len(manifest_records)
        pending_total += len(pending)
        sample_work.append((sample, pending))
        print(
            f"normalize_sample_ready={key} candidates={len(manifest_records)} "
            f"reused={len(manifest_records)-len(pending)} pending={len(pending)}",
            flush=True,
        )

    def checkpoint() -> None:
        _write(args.output, [*preserved, *existing.values()])

    if pending_total == 0:
        checkpoint()
        print(
            f"normalize_complete=candidates:{candidate_total} reused:{candidate_total} "
            f"generated:0 output={args.output}",
            flush=True,
        )
        return

    llm_context = (
        cli.local_llm_server(args.llama_server, args.local_model)
        if args.llm_backend == "local"
        else contextlib.nullcontext({
            "max_tokens": 512,
            "model": expected_model,
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        })
    )
    generated = 0
    with llm_context as llm_options:
        for sample, pending in sample_work:
            key = str(sample["sample_key"])
            if not pending:
                print(f"normalize_sample_cached={key}", flush=True)
                continue
            for position, (candidate, manifest_record) in enumerate(pending, 1):
                logical_key = (
                    key,
                    candidate.function.path,
                    candidate.function.name,
                    candidate.function.start_line,
                )
                skip_reason = manifest_record.get("skip_reason")
                if skip_reason is None:
                    try:
                        summaries = llm_normalize(candidate, **llm_options)
                    except (RuntimeError, ValueError, urllib.error.URLError, TimeoutError) as error:
                        checkpoint()
                        raise RuntimeError(
                            f"{key}:{candidate.function.name}: normalization failed"
                        ) from error
                    normalizer = "relevance-sliced-hybrid"
                    generated += 1
                else:
                    summaries = []
                    normalizer = "static-skip"
                record = {
                    "schema_version": NORMALIZATION_SCHEMA_VERSION,
                    "normalization_implementation_version": NORMALIZATION_IMPLEMENTATION_VERSION,
                    "sample_key": key,
                    "source_path": candidate.function.path,
                    "function": candidate.function.name,
                    "source_line": candidate.function.start_line,
                    "parameters": list(candidate.function.parameters),
                    "source_fingerprint": candidate_source_fingerprint(candidate),
                    "normalizer": normalizer,
                    "summaries": summaries,
                }
                if skip_reason is None:
                    record["llm_backend"] = args.llm_backend
                    record["llm_model"] = llm_options.get("model")
                else:
                    record["skip_reason"] = skip_reason
                existing[logical_key] = record
                checkpoint()
                print(
                    f"normalize_candidate_done={key}:{position}/{len(pending)} "
                    f"{candidate.function.name}@{candidate.function.start_line}",
                    flush=True,
                )
            print(f"normalize_sample_done={key}", flush=True)

    checkpoint()
    print(
        f"normalize_complete=candidates:{candidate_total} "
        f"reused:{candidate_total-generated} generated:{generated} output={args.output}",
        flush=True,
    )


def _manifest_candidates_for_detect(sample_key, index, _entry_method, _entry_language=None):
    _header, records = read_candidate_manifest(index)
    parse_cache = {}
    return [load_manifest_candidate(index, record, parse_cache) for record in records]


def _selected_replay(samples, args) -> list[dict[str, object]]:
    records = _read(args.replay) if Path(args.replay).is_file() else []
    by_key = {_normalization_key(record): record for record in records}
    selected: list[dict[str, object]] = []
    missing: list[str] = []
    for sample in samples:
        key = str(sample["sample_key"])
        _index, (_header, manifest_records) = _load_sample_manifest(sample, args)
        for manifest_record in manifest_records:
            logical_key = (
                key,
                str(manifest_record["source_path"]),
                str(manifest_record["function"]),
                int(manifest_record["source_line"]),
            )
            record = by_key.get(logical_key)
            if record is None or not _record_usable_for_run(record, manifest_record):
                missing.append(
                    f"{key}:{manifest_record['function']}@{manifest_record['source_line']}"
                )
            else:
                selected.append(record)
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise RuntimeError(
            f"normalization incomplete for {len(missing)} candidate(s): {preview}{suffix}; "
            "run normalize first"
        )
    return selected


def run(args: argparse.Namespace) -> None:
    samples = _read(args.samples)
    cli.validate_detection_manifest(samples)
    selected_keys = {str(sample["sample_key"]) for sample in samples}
    replay_records = _selected_replay(samples, args)
    old_detections = _read(args.detections) if Path(args.detections).is_file() else []
    old_semantics = _read(args.semantics) if Path(args.semantics).is_file() else []
    old_selected_detections = [
        record for record in old_detections
        if str(record.get("sample_key", "")) in selected_keys
    ]
    old_selected_semantics = [
        record for record in old_semantics
        if str(record.get("sample_key", "")) in selected_keys
    ]

    with tempfile.TemporaryDirectory(prefix="vul-workflow-run-") as directory:
        root = Path(directory)
        replay_path = root / "replay.jsonl"
        detections_path = root / "detections.jsonl"
        semantics_path = root / "semantics.jsonl"
        _write(replay_path, replay_records)
        if not args.refresh:
            _write(detections_path, old_selected_detections)
            _write(semantics_path, old_selected_semantics)

        original_discover = cli.discover_candidates
        cli.discover_candidates = _manifest_candidates_for_detect
        error: BaseException | None = None
        try:
            cli.detect(
                args.samples,
                str(replay_path),
                str(semantics_path),
                str(detections_path),
                joern_dir=args.joern_dir,
                java_home=args.java_home,
                cpg_cache_dir=args.cpg_cache_dir,
                resume=not args.refresh,
                linevul_codebert_path=args.linevul_codebert_path,
                linevul_checkpoint=args.linevul_checkpoint,
                linevul_threshold=args.linevul_threshold,
                linevul_device=args.linevul_device,
            )
        except BaseException as caught:
            error = caught
        finally:
            cli.discover_candidates = original_discover
            new_detections = _read(detections_path) if detections_path.is_file() else []
            new_semantics = _read(semantics_path) if semantics_path.is_file() else []
            _write(
                args.detections,
                _upsert_by_sample(old_detections, new_detections, selected_keys),
            )
            _write(
                args.semantics,
                _upsert_by_sample(old_semantics, new_semantics, selected_keys),
            )
        if error is not None:
            raise error

    oracle_keys = {str(record["sample_key"]) for record in _read(args.oracle)}
    detection_keys = {str(record["sample_key"]) for record in _read(args.detections)}
    if detection_keys == oracle_keys:
        cli.evaluate(args.detections, args.oracle, args.table, args.summary)
    else:
        print(
            f"run_complete=selected:{len(samples)} evaluation=skipped "
            f"detections:{len(detection_keys)}/{len(oracle_keys)}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Staged vulnerability workflow: preflight -> normalize -> run"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command):
        command.add_argument("--samples", default="data/detection_samples.jsonl")
        command.add_argument("--joern-dir", default="/home/phy/joern")
        command.add_argument("--java-home", default=os.environ.get("JAVA_HOME", "/home/phy/jdk21"))
        command.add_argument("--cpg-cache-dir", default="data/joern_cpg")
        command.add_argument("--refresh", action="store_true")

    p = sub.add_parser("preflight")
    common(p)
    p.set_defaults(func=preflight)

    n = sub.add_parser("normalize")
    common(n)
    n.add_argument("--llm-backend", choices=("local", "api"), default="local")
    n.add_argument("--llama-server", default="/home/phy/llama.cpp/build/bin/llama-server")
    n.add_argument(
        "--local-model",
        default="/home/phy/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-MXFP4_MOE.gguf",
    )
    n.add_argument("--output", default="data/normalizer_outputs.jsonl")
    n.set_defaults(func=normalize)

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--replay", default="data/normalizer_outputs.jsonl")
    r.add_argument("--oracle", default="data/oracle.jsonl")
    r.add_argument("--semantics", default="results/validated_semantics.jsonl")
    r.add_argument("--detections", default="results/detections.jsonl")
    r.add_argument("--table", default="results/results.csv")
    r.add_argument("--summary", default="results/summary.md")
    r.add_argument("--linevul-codebert-path", default="/home/PublicData/PHY-data/resource/codebert-base")
    r.add_argument("--linevul-checkpoint", default="/home/PublicData/PHY-data/resource/linevul/12heads_linevul_model.bin")
    r.add_argument("--linevul-threshold", type=float, default=0.5)
    r.add_argument("--linevul-device", default="auto")
    r.set_defaults(func=run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
