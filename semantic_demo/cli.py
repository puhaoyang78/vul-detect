from __future__ import annotations

import argparse
import contextlib
import csv
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
from .joern import JoernError, JoernValidator
from .semantics import (
    Validation,
    discover_candidates,
    llm_normalize,
    load_replay,
    rule_normalize,
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
    with target.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


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


def normalize_command(args: argparse.Namespace) -> None:
    samples = read_jsonl(args.samples)
    validate_detection_manifest(samples)
    records: list[dict[str, object]] = []
    if args.normalizer == "llm" and args.llm_backend == "local":
        llm_context = local_llm_server(args.llama_server, args.local_model)
    elif args.normalizer == "llm":
        llm_context = contextlib.nullcontext({"max_tokens": 8192})
    else:
        llm_context = contextlib.nullcontext({})

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
                try:
                    summaries = (
                        llm_normalize(candidate, **llm_options)
                        if args.normalizer == "llm"
                        else rule_normalize(candidate)
                    )
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
                records.append(
                    {
                        "sample_key": candidate.sample_key,
                        "source_path": candidate.function.path,
                        "function": candidate.function.name,
                        "parameters": list(candidate.function.parameters),
                        "normalizer": args.normalizer,
                        "llm_backend": (
                            args.llm_backend if args.normalizer == "llm" else None
                        ),
                        "llm_model": (
                            llm_options.get("model")
                            if args.normalizer == "llm"
                            else None
                        ),
                        "summaries": summaries,
                    }
                )
    write_jsonl(args.output, records)
    print(f"normalized_candidates={len(records)} output={args.output}")


def detect(
    samples_path: str,
    replay_path: str,
    semantics_path: str,
    detections_path: str,
    joern_dir: str = "/home/phy/joern",
    java_home: str = "/home/phy/jdk21",
    use_joern: bool = True,
) -> None:
    samples = read_jsonl(samples_path)
    validate_detection_manifest(samples)
    replay = load_replay(replay_path)
    joern = JoernValidator(joern_dir, java_home=java_home) if use_joern else None
    if joern is not None:
        joern.ensure_available()
    semantic_records: list[dict[str, object]] = []
    detection_records: list[dict[str, object]] = []

    backend = "joern" if use_joern else "lightweight"
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
        validations: list[Validation] = []
        summary_count = sum(
            len(
                replay.get(
                    (sample_key, candidate.function.path, candidate.function.name), []
                )
            )
            for candidate in candidates
        )
        print(
            f"validate_sample={sample_index}/{len(samples)} sample={sample_key} "
            f"candidates={len(candidates)} summaries={summary_count}",
            flush=True,
        )
        for candidate in candidates:
            key = (sample_key, candidate.function.path, candidate.function.name)
            summaries = replay.get(key, [])
            if summaries:
                print(
                    f"validate_candidate={candidate.function.name} "
                    f"summaries={len(summaries)}",
                    flush=True,
                )
            for summary in summaries:
                try:
                    validation = validate_summary(candidate, summary, joern=joern)
                except JoernError as error:
                    raise JoernError(
                        f"{sample_key}:{candidate.function.name}: {error}"
                    ) from error
                validations.append(validation)
                semantic_records.append(validation.as_json())

        baseline = analyze(entry, proposed=False)
        proposed = analyze(entry, validations=validations, proposed=True)
        detection_records.append(
            {
                "sample_key": sample_key,
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
        rows.append(
            {
                "CVE": truth["cve"],
                "repository": truth["repository"],
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

    baseline_hits = sum(row["baseline_verdict"] == "VULNERABLE" for row in rows)
    proposed_hits = sum(row["proposed_verdict"] == "VULNERABLE" for row in rows)
    corrections = sum(row["proposed_corrected_baseline"] == "YES" for row in rows)
    rejected = sum(int(row["rejected_semantics"]) for row in rows)
    failures = Counter(
        row["proposed_reason"]
        for row in rows
        if row["proposed_verdict"] != "VULNERABLE"
    )
    lines = [
        "# Demo 结果摘要",
        "",
        f"- 真实 C/C++ 内存安全 CVE：{len(rows)} 个",
        f"- Baseline 检出：{baseline_hits}/{len(rows)}",
        f"- Proposed 检出：{proposed_hits}/{len(rows)}",
        f"- Proposed 纠正 Baseline 漏检：{corrections} 个",
        f"- 静态验证拒绝的语义摘要：{rejected} 条",
        "",
        (
            "结论：出现明确但有限的正向信号。自动恢复并验证的项目语义使检出数从 "
            f"{baseline_hits} 增加到 {proposed_hits}，纠正了 {corrections} 个漏检。"
        ),
        (
            "该样本集全部为漏洞样本，并且优先选择了自定义内存函数，因此这里只能说明"
            "召回方向值得继续，不能据此判断误报率或泛化效果。"
        ),
        "",
        "## 主要失败原因",
        "",
    ]
    lines.extend(f"- {count} 个：{reason}" for reason, count in failures.most_common())
    lines.extend(
        [
            "- BSON 样本的模型摘要未严格归一化或未通过同一写入点的数据流验证。",
            "- ImageMagick 样本需要关联分配大小与后续循环读取范围，新增 READ/VALUE 后需重新评估。",
            "- Sofia SIP 样本依赖剩余输入长度与越界读取关系，新增 READ 语义后需重新评估。",
            "- FreeType 样本依赖宏、回调和整数偏移，直接调用候选筛选未恢复到写入语义。",
        ]
    )
    Path(summary_path).write_text("\n".join(lines) + "\n")
    print(
        f"samples={len(rows)} baseline={baseline_hits} proposed={proposed_hits} "
        f"corrections={corrections} table={table_path}"
    )


def run_command(args: argparse.Namespace) -> None:
    detect(
        args.samples,
        args.replay,
        args.semantics,
        args.detections,
        joern_dir=args.joern_dir,
        java_home=args.java_home,
        use_joern=not args.no_joern,
    )
    evaluate(args.detections, args.oracle, args.table, args.summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize", help="normalize candidate functions")
    normalize.add_argument("--samples", default="data/detection_samples.jsonl")
    normalize.add_argument("--normalizer", choices=("rules", "llm"), default="rules")
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
    normalize.set_defaults(func=normalize_command)

    run = subparsers.add_parser("run", help="run recovery, detection, and isolated evaluation")
    run.add_argument("--samples", default="data/detection_samples.jsonl")
    run.add_argument("--replay", default="data/normalizer_outputs.jsonl")
    run.add_argument("--oracle", default="data/oracle.jsonl")
    run.add_argument("--semantics", default="results/validated_semantics.jsonl")
    run.add_argument("--detections", default="results/detections.jsonl")
    run.add_argument("--table", default="results/results.csv")
    run.add_argument("--summary", default="results/summary.md")
    run.add_argument("--joern-dir", default="/home/phy/joern")
    run.add_argument(
        "--java-home",
        default=os.environ.get("JAVA_HOME", "/home/phy/jdk21"),
    )
    run.add_argument(
        "--no-joern",
        action="store_true",
        help="use the lightweight fallback validator instead of Joern",
    )
    run.set_defaults(func=run_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
