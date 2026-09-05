from __future__ import annotations

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
from .candidate_graph import load_manifest_candidate, read_candidate_manifest
from .joern import JoernRepositoryIndex
from .joern_v2 import JoernValidatorV2
from .linevul_baseline import LineVulBaseline
from .semantics import NORMALIZATION_RESPONSE_SCHEMA, Validation
from .source import GitRepository, source_language
from .validation_v2 import validate_summary


FORBIDDEN_DETECTION_FIELDS = {
    "cve",
    "cve_id",
    "fix_commit",
    "patch",
    "description",
    "mechanism",
    "ground_truth",
}
ANALYSIS_CHECKPOINT_VERSION = 13


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
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
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
        if "language" in sample and sample["language"] not in {"c", "cpp"}:
            raise ValueError(f"{key}: language must be c or cpp")
        for field in ("defines", "include_paths"):
            value = sample.get(field, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError(f"{key}: {field} must be a list of strings")


def load_repository_index(
    sample: dict[str, object],
    *,
    joern_dir: str = "/home/phy/joern",
    java_home: str = "/home/phy/jdk21",
    cpg_cache_dir: str = "data/joern_cpg",
):
    repository = GitRepository(
        str(sample["repository_git_dir"]), str(sample["vulnerable_commit"])
    )
    if not repository.has_revision():
        raise ValueError(f"{sample['sample_key']}: vulnerable revision is unavailable")
    index = JoernRepositoryIndex(
        repository,
        str(sample["sample_key"]),
        tuple(str(item) for item in sample.get("scan_paths", [])),
        str(sample["entry_path"]),
        defines=tuple(str(item) for item in sample.get("defines", [])),
        include_paths=tuple(str(item) for item in sample.get("include_paths", [])),
        joern_dir=joern_dir,
        java_home=java_home,
        cache_dir=cpg_cache_dir,
    )
    return repository, index


def entry_from_index(sample: dict[str, object], repository, index):
    entry_method = index.find_entry(
        str(sample["entry_function"]), str(sample["entry_path"])
    )
    if entry_method is None:
        raise ValueError(
            f"{sample['sample_key']}: Joern entry method {sample['entry_function']} "
            f"not found in {sample['entry_path']}; diagnostics={index.diagnostics_path}"
        )
    entry = repository.function_source(
        path=entry_method.path,
        name=entry_method.name,
        start_line=entry_method.start_line,
        end_line=entry_method.end_line,
        parameters=entry_method.parameters,
        parameter_types=entry_method.parameter_types,
        language_hint=source_language(
            entry_method.path, str(sample.get("language", "")) or None
        ),
    )
    return entry_method, entry


def load_entry(
    sample: dict[str, object],
    *,
    joern_dir: str = "/home/phy/joern",
    java_home: str = "/home/phy/jdk21",
    cpg_cache_dir: str = "data/joern_cpg",
):
    repository, index = load_repository_index(
        sample,
        joern_dir=joern_dir,
        java_home=java_home,
        cpg_cache_dir=cpg_cache_dir,
    )
    entry_method, entry = entry_from_index(sample, repository, index)
    return repository, index, entry_method, entry


@contextlib.contextmanager
def local_llm_server(executable: str, model_path: str) -> Iterator[dict[str, object]]:
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
        str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-c", "16384", "-np", "1", "-ngl", "99", "-t", "24",
        "--reasoning", "off", "--reasoning-budget", "0",
    ]
    print(f"starting_local_llm={model}", flush=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, text=True
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
                "response_schema": NORMALIZATION_RESPONSE_SCHEMA,
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
        candidate.function.path + "\0" + candidate.function.name + "\0"
        + candidate.function.text + "\0"
        + hashlib.sha256(candidate.function.translation_unit.encode()).hexdigest()
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _implementation_digest() -> str:
    root = Path(__file__).parent
    names = (
        "runtime.py",
        "candidate_graph.py",
        "normalization_v2.py",
        "validation_v2.py",
        "analyzer.py",
        "z3_reasoner_v2.py",
        "joern_v2.py",
        "standard_semantics.py",
    )
    digest = hashlib.sha256()
    for name in names:
        path = root / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _analysis_fingerprint(
    sample: dict[str, object],
    entry,
    candidates,
    replay: dict[tuple[str, str, str, int], list[dict[str, str]]],
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
            candidate.function.start_line,
        )
        candidate_inputs.append({
            "source_path": candidate.function.path,
            "function": candidate.function.name,
            "source_line": candidate.function.start_line,
            "source_fingerprint": _candidate_fingerprint(candidate),
            "summaries": replay.get(replay_key, []),
        })
    payload = {
        "checkpoint_version": ANALYSIS_CHECKPOINT_VERSION,
        "implementation": _implementation_digest(),
        "sample": sample,
        "entry_source": hashlib.sha256(entry.text.encode()).hexdigest(),
        "candidates": candidate_inputs,
        "backend": backend,
        "joern_timeout": joern_timeout,
        "baseline_signature": baseline_signature,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _load_replay(path: str | Path):
    replay: dict[tuple[str, str, str, int], list[dict[str, str]]] = {}
    for record in read_jsonl(path):
        summaries = record.get("summaries")
        if not isinstance(summaries, list) or not all(isinstance(item, dict) for item in summaries):
            raise ValueError(f"{path}: invalid summaries field")
        key = (
            str(record["sample_key"]),
            str(record["source_path"]),
            str(record["function"]),
            int(record["source_line"]),
        )
        replay[key] = summaries
    return replay


def _manifest_candidates(index):
    _header, records = read_candidate_manifest(index)
    parse_cache = {}
    return [load_manifest_candidate(index, record, parse_cache) for record in records]


def detect(
    samples_path: str,
    replay_path: str,
    semantics_path: str,
    detections_path: str,
    joern_dir: str = "/home/phy/joern",
    java_home: str = "/home/phy/jdk21",
    cpg_cache_dir: str = "data/joern_cpg",
    resume: bool = True,
    linevul_codebert_path: str = "/home/PublicData/PHY-data/resource/codebert-base",
    linevul_checkpoint: str = "/home/PublicData/PHY-data/resource/linevul/12heads_linevul_model.bin",
    linevul_threshold: float = 0.5,
    linevul_device: str = "auto",
) -> None:
    samples = read_jsonl(samples_path)
    validate_detection_manifest(samples)
    replay = _load_replay(replay_path)
    baseline_model = LineVulBaseline(
        linevul_codebert_path,
        linevul_checkpoint,
        threshold=linevul_threshold,
        device=linevul_device,
    )
    semantic_records: list[dict[str, object]] = []
    detection_records: list[dict[str, object]] = []
    detection_cache = (
        {str(record["sample_key"]): record for record in read_jsonl(detections_path)}
        if resume and Path(detections_path).is_file()
        else {}
    )

    backend = "joern"
    print(f"validation_start samples={len(samples)} backend={backend}", flush=True)
    for sample_index, sample in enumerate(samples, 1):
        sample_key = str(sample["sample_key"])
        _repository, index, _entry_method, entry = load_entry(
            sample,
            joern_dir=joern_dir,
            java_home=java_home,
            cpg_cache_dir=cpg_cache_dir,
        )
        candidates = _manifest_candidates(index)
        joern = JoernValidatorV2(
            joern_dir,
            java_home=java_home,
            repository_index=index,
        )
        joern.ensure_available()
        summary_entries: list[tuple[object, dict[str, str]]] = []
        for candidate in candidates:
            key = (
                sample_key,
                candidate.function.path,
                candidate.function.name,
                candidate.function.start_line,
            )
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
        print(
            f"validate_sample={sample_index}/{len(samples)} sample={sample_key} "
            f"candidates={len(candidates)} summaries={len(summary_entries)}",
            flush=True,
        )

        cached_detection = detection_cache.get(sample_key)
        if cached_detection is not None and cached_detection.get("analysis_fingerprint") == analysis_fingerprint:
            detection_records.append(cached_detection)
            semantic_records.extend(cached_detection.get("semantic_validations", []))
            write_jsonl(detections_path, detection_records)
            write_jsonl(semantics_path, semantic_records)
            print(f"validate_sample_resumed={sample_key}", flush=True)
            continue

        accepted_by_member: dict[tuple[str, str, str, int], list[dict[str, str]]] = {}
        expected_by_group: dict[tuple[str, str, str], int] = {}
        accepted: dict[tuple[str, str], list[dict[str, str]]] = {}
        pending = list(range(len(summary_entries)))
        final: dict[int, Validation] = {}

        def publish_common_summaries() -> None:
            accepted.clear()
            complete_groups: dict[tuple[str, str], list[list[dict[str, str]]]] = {}
            grouped: dict[
                tuple[str, str, str], dict[int, list[dict[str, str]]]
            ] = {}
            for (path, name, group_id, source_line), summaries in accepted_by_member.items():
                grouped.setdefault((path, name, group_id), {})[source_line] = summaries
            for (path, name, group_id), by_line in grouped.items():
                expected = expected_by_group[(path, name, group_id)]
                if len(by_line) != expected:
                    continue
                members = list(by_line.values())
                common = [
                    summary for summary in members[0]
                    if all(summary in member for member in members[1:])
                ]
                if common:
                    complete_groups.setdefault((path, name), []).append(common)
            for key, groups in complete_groups.items():
                if len(groups) == 1:
                    accepted[key] = groups[0]

        while pending:
            progress = False
            next_pending: list[int] = []
            for item_index in pending:
                candidate, summary = summary_entries[item_index]
                validation = validate_summary(
                    candidate,
                    summary,
                    joern=joern,
                    callee_summaries=accepted,
                )
                final[item_index] = validation
                if validation.passed:
                    group_id = validation.variant_group or f"single:{validation.source_line}"
                    group_key = (
                        validation.source_path,
                        validation.function,
                        group_id,
                    )
                    expected_by_group[group_key] = validation.variant_count
                    bucket = accepted_by_member.setdefault(
                        (
                            validation.source_path,
                            validation.function,
                            group_id,
                            validation.source_line,
                        ),
                        [],
                    )
                    if validation.summary not in bucket:
                        bucket.append(validation.summary)
                        progress = True
                else:
                    next_pending.append(item_index)
            publish_common_summaries()
            if not progress:
                break
            pending = next_pending

        for item_index in pending:
            candidate, summary = summary_entries[item_index]
            final[item_index] = validate_summary(
                candidate,
                summary,
                joern=joern,
                callee_summaries=accepted,
            )

        validations = [final[index] for index in range(len(summary_entries))]
        semantic_records.extend(item.as_json() for item in validations)
        baseline = baseline_model.predict(entry.text)
        proposed = analyze(entry, validations=validations)
        detection_records.append({
            "sample_key": sample_key,
            "analysis_fingerprint": analysis_fingerprint,
            "repository_git_dir": sample["repository_git_dir"],
            "vulnerable_commit": sample["vulnerable_commit"],
            "entry_path": entry.path,
            "entry_function": entry.name,
            "candidate_functions": [
                f"{candidate.function.name}@{candidate.function.start_line}"
                for candidate in candidates
            ],
            "baseline": baseline.as_json(),
            "proposed": proposed.as_json(),
            "validated_semantic_count": sum(item.passed for item in validations),
            "rejected_semantic_count": sum(not item.passed for item in validations),
            "semantic_validations": [item.as_json() for item in validations],
        })
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
            for item in validations if not item["passed"]
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
            for access in access_checks if access.get("model")
        ]
        rows.append({
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
            "z3_model": json.dumps(z3_models, ensure_ascii=False, sort_keys=True),
            "fix_commit": truth["fix_commit"],
            "fixing_patch": truth["fix_url"],
            "human_verified_mechanism": truth["mechanism"],
            "proposed_corrected_baseline": "YES" if corrected else "NO",
        })

    target = Path(table_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def classification_metrics(prefix: str) -> dict[str, float | int]:
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
        recall = tp / (tp + fn + unknown_vulnerable) if tp + fn + unknown_vulnerable else 0.0
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

    baseline_metrics = classification_metrics("baseline")
    proposed_metrics = classification_metrics("proposed")
    rejected = sum(int(row["rejected_semantics"]) for row in rows)
    z3_statuses = Counter(row["z3_status"] for row in rows if row.get("z3_status"))
    failures = Counter(
        row["proposed_reason"] for row in rows if row["proposed_verdict"] != "VULNERABLE"
    )
    vulnerable_count = sum(row["ground_truth"] == "VULNERABLE" for row in rows)
    benign_count = len(rows) - vulnerable_count
    lines = [
        "# Demo 结果摘要", "",
        f"- 函数级样本：{len(rows)}（VULNERABLE={vulnerable_count}, BENIGN={benign_count}）",
        (
            "- LineVul Baseline："
            f"TP={baseline_metrics['tp']}, FP={baseline_metrics['fp']}, "
            f"TN={baseline_metrics['tn']}, FN={baseline_metrics['fn']}, "
            f"UNKNOWN={baseline_metrics['unknown']} "
            f"(V={baseline_metrics['unknown_vulnerable']}, B={baseline_metrics['unknown_benign']}), "
            f"Precision={baseline_metrics['precision']:.4f}, "
            f"Recall={baseline_metrics['recall']:.4f}, F1={baseline_metrics['f1']:.4f}, "
            f"Accuracy={baseline_metrics['accuracy']:.4f}, Coverage={baseline_metrics['coverage']:.4f}"
        ),
        (
            "- Proposed："
            f"TP={proposed_metrics['tp']}, FP={proposed_metrics['fp']}, "
            f"TN={proposed_metrics['tn']}, FN={proposed_metrics['fn']}, "
            f"UNKNOWN={proposed_metrics['unknown']} "
            f"(V={proposed_metrics['unknown_vulnerable']}, B={proposed_metrics['unknown_benign']}), "
            f"Precision={proposed_metrics['precision']:.4f}, "
            f"Recall={proposed_metrics['recall']:.4f}, F1={proposed_metrics['f1']:.4f}, "
            f"Accuracy={proposed_metrics['accuracy']:.4f}, Coverage={proposed_metrics['coverage']:.4f}"
        ),
        f"- 静态验证拒绝的语义摘要：{rejected} 条",
        (
            "- Z3 状态：" + ", ".join(
                f"{status}={count}" for status, count in sorted(z3_statuses.items())
            ) if z3_statuses else "- Z3 状态：无"
        ),
        "", "## 主要未解析原因", "",
    ]
    lines.extend(f"- {count} 个：{reason}" for reason, count in failures.most_common())
    Path(summary_path).write_text("\n".join(lines) + "\n")
    print(
        f"samples={len(rows)} baseline_f1={baseline_metrics['f1']:.4f} "
        f"proposed_f1={proposed_metrics['f1']:.4f} table={table_path}"
    )
