from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from .analyzer import analyze
from .linevul_baseline import LineVulBaseline
from .joern import JoernError, JoernValidator
from .semantics import (
    NORMALIZATION_SCHEMA_VERSION,
    Validation,
    discover_candidates,
    llm_normalize,
    load_replay,
    validate_summary,
)
from .source import GitRepository


FORBIDDEN_DETECTION_FIELDS = {
    "cve",
    "cve_id",
    "fix_commit",
    "patch",
    "description",
    "mechanism",
    "ground_truth",
}
ANALYSIS_CHECKPOINT_VERSION = 7


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=target.parent,
            prefix=f".{target.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def validate_detection_manifest(samples: list[dict[str, object]]) -> None:
    keys: set[str] = set()
    for sample in samples:
        overlap = FORBIDDEN_DETECTION_FIELDS & set(sample)
        if overlap:
            raise ValueError(
                f"detection manifest contains forbidden oracle fields: {sorted(overlap)}"
            )
        key = str(sample["sample_key"])
        if key in keys:
            raise ValueError(f"duplicate sample_key: {key}")
        keys.add(key)
        if len(str(sample["vulnerable_commit"])) != 40:
            raise ValueError(f"{key}: vulnerable_commit must be a full Git object id")


def _load_entry(sample: dict[str, object]):
    repository = GitRepository(
        str(sample["repository_git_dir"]), str(sample["vulnerable_commit"])
    )
    if not repository.has_revision():
        raise ValueError(f"{sample['sample_key']}: vulnerable revision is unavailable")
    entry = repository.find_function(
        str(sample["entry_function"]),
        preferred_path=str(sample["entry_path"]),
        scopes=tuple(str(item) for item in sample.get("scan_paths", [])),
    )
    if entry is None:
        raise ValueError(
            f"{sample['sample_key']}: entry function {sample['entry_function']} not found"
        )
    return repository, entry


@contextlib.contextmanager
def local_llm_server(
    executable: str, model_path: str
) -> Iterator[dict[str, object]]:
    server = Path(executable)
    model = Path(model_path)
    if not server.is_file():
        raise FileNotFoundError(f"llama-server not found: {server}")
    if not model.is_file():
        raise FileNotFoundError(f"local Qwen model not found: {model}")

    with socket.socket() as port_reservation:
        port_reservation.bind(("127.0.0.1", 0))
        port = port_reservation.getsockname()[1]

    command = [
        str(server),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        "16384",
        "-np",
        "1",
        "-ngl",
        "99",
        "-t",
        "24",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]
    print(f"starting_local_llm={model}", flush=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.monotonic() + 180
            health_url = f"http://127.0.0.1:{port}/health"
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.flush()
                    log.seek(0)
                    raise RuntimeError(
                        "local llama-server exited during startup:\n" + log.read()[-4000:]
                    )
                try:
                    with opener.open(health_url, timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise RuntimeError("local llama-server was not ready within 180 seconds")

            print(f"local_llm_ready=http://127.0.0.1:{port}", flush=True)
            yield {
                "api_key": "local",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "model": model.stem,
                "max_tokens": 512,
                "disable_proxy": True,
            }
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)


def _candidate_fingerprint(candidate) -> str:
    payload = (
        candidate.function.path
        + "\0"
        + candidate.function.name
        + "\0"
        + candidate.function.text
        + "\0"
        + hashlib.sha256(candidate.function.translation_unit.encode()).hexdigest()
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalization_cache(path: str | Path) -> dict[tuple[str, str, str, str], dict[str, object]]:
    target = Path(path)
    if not target.is_file():
        return {}
    cache: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for record in read_jsonl(target):
        fingerprint = str(record.get("source_fingerprint", ""))
        key = (
            str(record.get("sample_key", "")),
            str(record.get("source_path", "")),
            str(record.get("function", "")),
            fingerprint,
        )
        cache[key] = record
    return cache


def _analysis_fingerprint(
    sample: dict[str, object],
    entry,
    candidates,
    replay: dict[tuple[str, str, str], list[dict[str, str]]],
    backend: str,
    joern_timeout: int | None,
    baseline_signature: str,
) -> str:
    candidate_inputs = []
    sample_key = str(sample["sample_key"])
    for candidate in candidates:
        replay_key = (
            sample_key,
            candidate.function.path,
            candidate.function.name,
        )
        candidate_inputs.append(
            {
                "source_path": candidate.function.path,
                "function": candidate.function.name,
                "source_fingerprint": _candidate_fingerprint(candidate),
                "summaries": replay.get(replay_key, []),
            }
        )
    payload = {
        "checkpoint_version": ANALYSIS_CHECKPOINT_VERSION,
        "sample": sample,
        "entry_source": hashlib.sha256(entry.text.encode()).hexdigest(),
        "candidates": candidate_inputs,
        "backend": backend,
        "joern_timeout": joern_timeout,
        "baseline_signature": baseline_signature,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalize_command(args: argparse.Namespace) -> None:
    samples = read_jsonl(args.samples)
    validate_detection_manifest(samples)
    records: list[dict[str, object]] = []
    cache = {} if args.refresh else _normalization_cache(args.output)
    if args.refresh:
        write_jsonl(args.output, [])
    reused = 0
    generated = 0
    if args.llm_backend == "local":
        llm_context = local_llm_server(args.llama_server, args.local_model)
    else:
        llm_context = contextlib.nullcontext({"max_tokens": 8192})

    with llm_context as llm_options:
        for sample in samples:
            repository, entry = _load_entry(sample)
            candidates = discover_candidates(
                str(sample["sample_key"]),
                repository,
                entry,
                tuple(str(item) for item in sample.get("scan_paths", [])),
            )
            for candidate in candidates:
                fingerprint = _candidate_fingerprint(candidate)
                cache_key = (
                    candidate.sample_key,
                    candidate.function.path,
                    candidate.function.name,
                    fingerprint,
                )
                cached = cache.get(cache_key)
                if (
                    cached is not None
                    and cached.get("schema_version") == NORMALIZATION_SCHEMA_VERSION
                    and cached.get("normalizer") == "llm"
                    and cached.get("llm_backend") == args.llm_backend
                ):
                    records.append(cached)
                    reused += 1
                    continue

                try:
                    summaries = llm_normalize(candidate, **llm_options)
                except (
                    RuntimeError,
                    ValueError,
                    urllib.error.URLError,
                    TimeoutError,
                ) as error:
                    raise RuntimeError(
                        f"{candidate.sample_key}:{candidate.function.name}: "
                        "normalization failed"
                    ) from error
                generated += 1
                record = {
                    "schema_version": NORMALIZATION_SCHEMA_VERSION,
                    "sample_key": candidate.sample_key,
                    "source_path": candidate.function.path,
                    "function": candidate.function.name,
                    "parameters": list(candidate.function.parameters),
                    "source_fingerprint": fingerprint,
                    "normalizer": "llm",
                    "llm_backend": args.llm_backend,
                    "llm_model": llm_options.get("model"),
                    "summaries": summaries,
                }
                records.append(record)
                cache[cache_key] = record
                write_jsonl(args.output, cache.values())
                print(
                    f"normalize_candidate_done={candidate.sample_key}:"
                    f"{candidate.function.name} checkpoint={args.output}",
                    flush=True,
                )
    write_jsonl(args.output, records)
    print(
        f"normalized_candidates={len(records)} reused={reused} generated={generated} "
        f"output={args.output}"
    )


def detect(
    samples_path: str,
    replay_path: str,
    semantics_path: str,
    detections_path: str,
    joern_dir: str = "/home/phy/joern",
    java_home: str = "/home/phy/jdk21",
    resume: bool = True,
    linevul_codebert_path: str = "/home/PublicData/PHY-data/resource/codebert-base",
    linevul_checkpoint: str = "/home/PublicData/PHY-data/resource/linevul/12heads_linevul_model.bin",
    linevul_threshold: float = 0.5,
    linevul_device: str = "auto",
) -> None:
    samples = read_jsonl(samples_path)
    validate_detection_manifest(samples)
    replay = load_replay(replay_path)
    baseline_model = LineVulBaseline(
        linevul_codebert_path,
        linevul_checkpoint,
        threshold=linevul_threshold,
        device=linevul_device,
    )
    joern = JoernValidator(joern_dir, java_home=java_home)
    joern.ensure_available()
    semantic_records: list[dict[str, object]] = []
    detection_records: list[dict[str, object]] = []
    detection_cache = (
        {
            str(record["sample_key"]): record
            for record in read_jsonl(detections_path)
        }
        if resume and Path(detections_path).is_file()
        else {}
    )

    backend = "joern"
    print(f"validation_start samples={len(samples)} backend={backend}", flush=True)
    for sample_index, sample in enumerate(samples, 1):
        sample_key = str(sample["sample_key"])
        repository, entry = _load_entry(sample)
        candidates = discover_candidates(
            sample_key,
            repository,
            entry,
            tuple(str(item) for item in sample.get("scan_paths", [])),
        )
        summary_entries: list[tuple[object, dict[str, str]]] = []
        for candidate in candidates:
            key = (sample_key, candidate.function.path, candidate.function.name)
            for summary in replay.get(key, []):
                summary_entries.append((candidate, summary))

        analysis_fingerprint = _analysis_fingerprint(
            sample,
            entry,
            candidates,
            replay,
            backend,
            joern.timeout,
            baseline_model.signature,
        )

        summary_count = len(summary_entries)
        print(
            f"validate_sample={sample_index}/{len(samples)} sample={sample_key} "
            f"candidates={len(candidates)} summaries={summary_count}",
            flush=True,
        )

        cached_detection = detection_cache.get(sample_key)
        if (
            cached_detection is not None
            and cached_detection.get("analysis_fingerprint")
            == analysis_fingerprint
        ):
            detection_records.append(cached_detection)
            semantic_records.extend(
                cached_detection.get("semantic_validations", [])
            )
            write_jsonl(detections_path, detection_records)
            write_jsonl(semantics_path, semantic_records)
            print(f"validate_sample_resumed={sample_key}", flush=True)
            continue

        # Fixed-point validation: primitive-backed summaries seed the process;
        # wrapper summaries may then be validated from already accepted callees.
        accepted: dict[tuple[str, str], list[dict[str, str]]] = {}
        pending = list(range(len(summary_entries)))
        final: dict[int, Validation] = {}

        while pending:
            progress = False
            next_pending: list[int] = []
            for index in pending:
                candidate, summary = summary_entries[index]
                try:
                    validation = validate_summary(
                        candidate,
                        summary,
                        joern=joern,
                        callee_summaries=accepted,
                    )
                except JoernError as error:
                    raise JoernError(
                        f"{sample_key}:{candidate.function.name}: {error}"
                    ) from error
                final[index] = validation
                if validation.passed:
                    bucket = accepted.setdefault(
                        (validation.source_path, validation.function), []
                    )
                    if validation.summary not in bucket:
                        bucket.append(validation.summary)
                    progress = True
                else:
                    next_pending.append(index)
            if not progress:
                break
            pending = next_pending

        # Re-evaluate unresolved summaries once with the complete accepted set so
        # rejection reasons reflect the final fixed point.
        for index in pending:
            candidate, summary = summary_entries[index]
            final[index] = validate_summary(
                candidate,
                summary,
                joern=joern,
                callee_summaries=accepted,
            )

        validations = [final[index] for index in range(len(summary_entries))]
        semantic_records.extend(item.as_json() for item in validations)

        baseline = baseline_model.predict(entry.text)
        proposed = analyze(entry, validations=validations)
        detection_records.append(
            {
                "sample_key": sample_key,
                "analysis_fingerprint": analysis_fingerprint,
                "repository_git_dir": sample["repository_git_dir"],
                "vulnerable_commit": sample["vulnerable_commit"],
                "entry_path": entry.path,
                "entry_function": entry.name,
                "candidate_functions": [candidate.function.name for candidate in candidates],
                "baseline": baseline.as_json(),
                "proposed": proposed.as_json(),
                "validated_semantic_count": sum(item.passed for item in validations),
                "rejected_semantic_count": sum(not item.passed for item in validations),
                "semantic_validations": [item.as_json() for item in validations],
            }
        )
        print(
            f"validate_sample_done={sample_key} "
            f"passed={sum(item.passed for item in validations)} "
            f"rejected={sum(not item.passed for item in validations)}",
            flush=True,
        )
        write_jsonl(detections_path, detection_records)
        write_jsonl(semantics_path, semantic_records)

    write_jsonl(semantics_path, semantic_records)
    write_jsonl(detections_path, detection_records)


def evaluate(
    detections_path: str,
    oracle_path: str,
    table_path: str,
    summary_path: str,
) -> None:
    detections = {record["sample_key"]: record for record in read_jsonl(detections_path)}
    oracle = read_jsonl(oracle_path)
    if set(detections) != {record["sample_key"] for record in oracle}:
        raise ValueError("detection/oracle sample keys differ")

    rows: list[dict[str, object]] = []
    for truth in oracle:
        result = detections[truth["sample_key"]]
        fix_repository = GitRepository(
            str(result["repository_git_dir"]), str(truth["fix_commit"])
        )
        if not fix_repository.has_revision():
            raise ValueError(f"{truth['sample_key']}: fixing commit is unavailable")
        baseline = result["baseline"]
        proposed = result["proposed"]
        validations = result["semantic_validations"]
        automatic_semantics = " | ".join(
            f"{item['function']}={json.dumps(item['summary'], ensure_ascii=False, sort_keys=True)}"
            for item in validations
        )
        rejected_details = " | ".join(
            f"{item['function']}: {item['reason']}"
            for item in validations
            if not item["passed"]
        )
        corrected = (
            baseline["verdict"] != "VULNERABLE"
            and proposed["verdict"] == "VULNERABLE"
            and truth["ground_truth"] == "VULNERABLE"
        )
        constraint_result = proposed.get("constraint_result") or {}
        access_checks = constraint_result.get("accesses") or []
        condition_text = " | ".join(
            str(condition.get("condition", ""))
            for access in access_checks
            for condition in (access.get("conditions") or [])
            if condition.get("condition")
        )
        z3_models = [
            {
                "line": access.get("line"),
                "buffer": access.get("buffer"),
                "status": access.get("status"),
                "model": access.get("model") or {},
            }
            for access in access_checks
            if access.get("model")
        ]
        rows.append(
            {
                "CVE": truth["cve"],
                "repository": truth["repository"],
                "ground_truth": truth["ground_truth"],
                "vulnerable_commit": result["vulnerable_commit"],
                "custom_functions": ", ".join(truth["custom_functions"]),
                "automatic_semantics": automatic_semantics,
                "validation": (
                    f"PASS={result['validated_semantic_count']}; "
                    f"REJECT={result['rejected_semantic_count']}"
                ),
                "validated_semantics": result["validated_semantic_count"],
                "rejected_semantics": result["rejected_semantic_count"],
                "rejected_details": rejected_details,
                "baseline_verdict": baseline["verdict"],
                "baseline_reason": baseline["reason"],
                "proposed_verdict": proposed["verdict"],
                "proposed_reason": proposed["reason"],
                "z3_status": constraint_result.get("status", ""),
                "verification_conditions": condition_text,
                "z3_model": json.dumps(
                    z3_models,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "fix_commit": truth["fix_commit"],
                "fixing_patch": truth["fix_url"],
                "human_verified_mechanism": truth["mechanism"],
                "proposed_corrected_baseline": "YES" if corrected else "NO",
            }
        )

    target = Path(table_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def _classification_metrics(prefix: str) -> dict[str, float | int]:
        tp = fp = tn = fn = unknown_vulnerable = unknown_benign = 0
        for row in rows:
            truth_label = row["ground_truth"]
            verdict = row[f"{prefix}_verdict"]
            if verdict == "UNKNOWN":
                if truth_label == "VULNERABLE":
                    unknown_vulnerable += 1
                else:
                    unknown_benign += 1
                continue
            predicted = "VULNERABLE" if verdict == "VULNERABLE" else "BENIGN"
            if truth_label == "VULNERABLE" and predicted == "VULNERABLE":
                tp += 1
            elif truth_label == "BENIGN" and predicted == "VULNERABLE":
                fp += 1
            elif truth_label == "BENIGN":
                tn += 1
            else:
                fn += 1
        decided = tp + fp + tn + fn
        unknown = unknown_vulnerable + unknown_benign
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = (
            tp / (tp + fn + unknown_vulnerable)
            if tp + fn + unknown_vulnerable
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy_decided = (tp + tn) / decided if decided else 0.0
        coverage = decided / len(rows) if rows else 0.0
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "unknown": unknown,
            "unknown_vulnerable": unknown_vulnerable,
            "unknown_benign": unknown_benign,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy_decided, "coverage": coverage,
        }

    baseline_metrics = _classification_metrics("baseline")
    proposed_metrics = _classification_metrics("proposed")
    rejected = sum(int(row["rejected_semantics"]) for row in rows)
    z3_statuses = Counter(row["z3_status"] for row in rows if row.get("z3_status"))
    failures = Counter(
        row["proposed_reason"] for row in rows if row["proposed_verdict"] != "VULNERABLE"
    )
    vulnerable_count = sum(row["ground_truth"] == "VULNERABLE" for row in rows)
    benign_count = len(rows) - vulnerable_count
    lines = [
        "# Demo 结果摘要",
        "",
        f"- 函数级样本：{len(rows)}（VULNERABLE={vulnerable_count}, BENIGN={benign_count}）",
        (
            "- LineVul Baseline："
            f"TP={baseline_metrics['tp']}, FP={baseline_metrics['fp']}, "
            f"TN={baseline_metrics['tn']}, FN={baseline_metrics['fn']}, "
            f"UNKNOWN={baseline_metrics['unknown']} "
            f"(V={baseline_metrics['unknown_vulnerable']}, B={baseline_metrics['unknown_benign']}), "
            f"Precision={baseline_metrics['precision']:.4f}, "
            f"Recall={baseline_metrics['recall']:.4f}, F1={baseline_metrics['f1']:.4f}, "
            f"Accuracy={baseline_metrics['accuracy']:.4f}, "
            f"Coverage={baseline_metrics['coverage']:.4f}"
        ),
        (
            "- Proposed："
            f"TP={proposed_metrics['tp']}, FP={proposed_metrics['fp']}, "
            f"TN={proposed_metrics['tn']}, FN={proposed_metrics['fn']}, "
            f"UNKNOWN={proposed_metrics['unknown']} "
            f"(V={proposed_metrics['unknown_vulnerable']}, B={proposed_metrics['unknown_benign']}), "
            f"Precision={proposed_metrics['precision']:.4f}, "
            f"Recall={proposed_metrics['recall']:.4f}, F1={proposed_metrics['f1']:.4f}, "
            f"Accuracy={proposed_metrics['accuracy']:.4f}, "
            f"Coverage={proposed_metrics['coverage']:.4f}"
        ),
        f"- 静态验证拒绝的语义摘要：{rejected} 条",
        (
            "- Z3 状态：" + ", ".join(
                f"{status}={count}" for status, count in sorted(z3_statuses.items())
            ) if z3_statuses else "- Z3 状态：无"
        ),
        "",
        "## 主要未解析原因",
        "",
    ]
    lines.extend(f"- {count} 个：{reason}" for reason, count in failures.most_common())
    Path(summary_path).write_text("\n".join(lines) + "\n")
    print(
        f"samples={len(rows)} baseline_f1={baseline_metrics['f1']:.4f} "
        f"proposed_f1={proposed_metrics['f1']:.4f} table={table_path}"
    )


def run_command(args: argparse.Namespace) -> None:
    detect(
        args.samples,
        args.replay,
        args.semantics,
        args.detections,
        joern_dir=args.joern_dir,
        java_home=args.java_home,
        resume=not args.refresh,
        linevul_codebert_path=args.linevul_codebert_path,
        linevul_checkpoint=args.linevul_checkpoint,
        linevul_threshold=args.linevul_threshold,
        linevul_device=args.linevul_device,
    )
    evaluate(args.detections, args.oracle, args.table, args.summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize", help="normalize candidate functions")
    normalize.add_argument("--samples", default="data/detection_samples.jsonl")
    normalize.add_argument(
        "--llm-backend",
        choices=("local", "api"),
        default="local",
        help="use local Qwen by default; api reads DEEPSEEK_* environment variables",
    )
    normalize.add_argument(
        "--llama-server",
        default="/home/phy/llama.cpp/build/bin/llama-server",
    )
    normalize.add_argument(
        "--local-model",
        default=(
            "/home/phy/models/Qwen3.6-35B-A3B-MTP-GGUF/"
            "Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
        ),
    )
    normalize.add_argument("--output", default="data/normalizer_outputs.jsonl")
    normalize.add_argument(
        "--refresh",
        action="store_true",
        help="ignore cached normalization records and regenerate all candidate summaries",
    )
    normalize.set_defaults(func=normalize_command)

    run = subparsers.add_parser("run", help="run recovery, detection, and isolated evaluation")
    run.add_argument("--samples", default="data/detection_samples.jsonl")
    run.add_argument("--replay", default="data/normalizer_outputs.jsonl")
    run.add_argument("--oracle", default="data/oracle.jsonl")
    run.add_argument("--semantics", default="results/validated_semantics.jsonl")
    run.add_argument("--detections", default="results/detections.jsonl")
    run.add_argument("--table", default="results/results.csv")
    run.add_argument("--summary", default="results/summary.md")
    run.add_argument(
        "--linevul-codebert-path",
        default="/home/PublicData/PHY-data/resource/codebert-base",
        help="local CodeBERT base model used by LineVul",
    )
    run.add_argument(
        "--linevul-checkpoint",
        default="/home/PublicData/PHY-data/resource/linevul/12heads_linevul_model.bin",
        help="official LineVul RQ1 checkpoint",
    )
    run.add_argument("--linevul-threshold", type=float, default=0.5)
    run.add_argument(
        "--linevul-device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc.",
    )
    run.add_argument("--joern-dir", default="/home/phy/joern")
    run.add_argument(
        "--java-home",
        default=os.environ.get("JAVA_HOME", "/home/phy/jdk21"),
    )
    run.add_argument(
        "--refresh",
        action="store_true",
        help="ignore completed sample checkpoints and rerun all validation",
    )
    run.set_defaults(func=run_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
