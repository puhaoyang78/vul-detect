from __future__ import annotations

import json
import os
import re
import subprocess
<<<<<<< Updated upstream
=======
from itertools import zip_longest
>>>>>>> Stashed changes
from dataclasses import dataclass
from pathlib import Path

from .cpg import CPGError, FunctionGraph, GraphNode, extract_function_cpg
<<<<<<< Updated upstream
from .semantics import extract_vulnerability_semantics
=======
>>>>>>> Stashed changes
from .syntax import parse_function


DATASET_SCHEMA_VERSION = 2

_MEMORY_NAMES = {
    "memcpy", "memmove", "mempcpy", "memset", "memcmp", "bcopy", "bzero",
    "read", "recv", "recvfrom", "fread", "write", "send", "sendto", "fwrite",
    "strcpy", "strcat", "strncpy", "strncat", "strlcpy", "strlcat",
    "sprintf", "vsprintf", "snprintf", "vsnprintf", "free",
}
_ALLOC_NAMES = {"malloc", "calloc", "realloc", "kmalloc", "kzalloc", "vmalloc", "new"}
_COMPARE = re.compile(r"<=|>=|==|!=|<|>")
_ARITHMETIC = re.compile(r"(?<![+\-*/%&|^<>])[+\-*/%]|<<|>>")


@dataclass(frozen=True)
class GraphRelation:
    kind: str
    source: str
    target: str

    def as_text(self) -> str:
        return f"{self.kind}|{self.source}|{self.target}"


@dataclass(frozen=True)
class FunctionSample:
    sample_key: str
    source: str
    label: int
    language: str
    function_name: str | None
    split: str | None


def _node_categories(node: GraphNode) -> set[str]:
    code = node.code
    lower = code.lower()
    categories: set[str] = set()
    if node.label == "CONTROL_STRUCTURE" or lower.startswith(("if ", "if(", "while ", "while(", "for ", "for(")):
        categories.add("control")
    if _COMPARE.search(code):
        categories.add("comparison")
    if any(re.search(rf"\b{re.escape(name)}\b", code) for name in _ALLOC_NAMES):
        categories.add("allocation")
    if any(re.search(rf"\b{re.escape(name)}\b", code) for name in _MEMORY_NAMES):
        categories.add("memory")
    if any(token in node.label for token in ("indirectIndexAccess", "indirection", "fieldAccess")) or "[" in code:
        categories.add("pointer_index")
    if _ARITHMETIC.search(code) or any(
        token in node.label
        for token in ("addition", "subtraction", "multiplication", "division", "shiftLeft", "shiftRight")
    ):
        categories.add("arithmetic")
    return categories


def _node_text(node: GraphNode) -> str:
    code = re.sub(r"\s+", " ", node.code).strip()
    return f"{node.label}:{code}"


def graph_relations(graph: FunctionGraph) -> tuple[GraphRelation, ...]:
    relevant = {node_id for node_id, node in graph.nodes.items() if _node_categories(node)}
    for edge in graph.edges:
        if edge.source in relevant or edge.target in relevant:
            relevant.update((edge.source, edge.target))

    relations: list[GraphRelation] = []
    seen: set[str] = set()
    for edge in graph.edges:
        if edge.source not in relevant and edge.target not in relevant:
            continue
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source is None or target is None:
            continue
        relation = GraphRelation(edge.kind, _node_text(source), _node_text(target))
        text = relation.as_text()
        if text not in seen:
            seen.add(text)
            relations.append(relation)
    return tuple(relations)


def render_graph(graph: FunctionGraph, max_relations: int = 160) -> str:
    if max_relations <= 0:
        raise ValueError('max_relations must be positive')
    relations = graph_relations(graph)
<<<<<<< Updated upstream
    return "\n".join(item.as_text() for item in relations[:max_relations]) or "NO_SECURITY_RELEVANT_GRAPH_RELATIONS"
=======
    groups: dict[str, list[GraphRelation]] = {}
    for relation in relations:
        groups.setdefault(relation.kind, []).append(relation)
    # Round-robin shares the budget across available kinds; unused slots
    # automatically go to kinds with more edges.
    selected = [item for row in zip_longest(*groups.values()) for item in row if item is not None]
    return '\n'.join(item.as_text() for item in selected[:max_relations]) or 'NO_SECURITY_RELEVANT_GRAPH_RELATIONS'
>>>>>>> Stashed changes


def _required_string(record: dict[str, object], names: tuple[str, ...], label: str) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"missing {label}; expected one of {', '.join(names)}")


def _normalize_label(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, float) and value in {0.0, 1.0}:
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "benign"}:
            return 0
        if normalized in {"1", "vulnerable"}:
            return 1
    raise ValueError(f"label must be binary 0/1, got {value!r}")


def _sample_fields(record: dict[str, object]) -> FunctionSample:
    key_value = next(
        (record[name] for name in ("sample_key", "id", "idx") if name in record and record[name] is not None),
        None,
    )
    if key_value is None or str(key_value).strip() == "":
        raise ValueError("sample record requires sample_key, id, or idx")
    key = str(key_value)
    source = _required_string(record, ("function", "func", "source", "code", "func_before"), "function source")

    if "label" in record:
        label_value = record["label"]
    elif "target" in record:
        label_value = record["target"]
    else:
        raise ValueError(f"{key}: sample record requires label or target")
    label = _normalize_label(label_value)

    language = str(record.get("language") or "c").lower()
    language = "cpp" if language in {"c++", "cpp"} else language
    if language not in {"c", "cpp"}:
        raise ValueError(f"{key}: language must be c or cpp")

    function_name = record.get("function_name")
    split_value = record.get("split")
    split = str(split_value).lower() if split_value is not None else None
    if split == "validation":
        split = "valid"
    if split is not None and split not in {"train", "valid", "test"}:
        raise ValueError(f"{key}: split must be train, valid, validation, or test")
    return FunctionSample(key, source, label, language, str(function_name) if function_name else None, split)


<<<<<<< Updated upstream
def _read_samples(samples_path: str | Path) -> list[FunctionSample]:
    samples: list[FunctionSample] = []
    seen: set[str] = set()
=======
def build_function_dataset(
    samples_path: str | Path,
    output_path: str | Path,
    *,
    joern_dir: str | Path = '/home/phy/joern',
    java_home: str | Path = '/home/phy/jdk21',
    timeout: int = 300,
) -> list[dict[str, object]]:
    """Build one CPG-augmented record per function while preserving the dataset's original binary label."""
    target = Path(output_path)
    if target.resolve() == Path(samples_path).resolve():
        raise ValueError('samples and output must be different files')
    errors_path = target.with_suffix('.errors.jsonl')
    if errors_path.resolve() in {target.resolve(), Path(samples_path).resolve()}:
        raise ValueError('error log must be different from samples and output')
    samples = {}
>>>>>>> Stashed changes
    with Path(samples_path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {samples_path}:{line_number}: {error}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"{samples_path}:{line_number}: each JSONL row must be an object")
            sample = _sample_fields(raw)
            if sample.sample_key in seen:
                raise ValueError(f"{samples_path}:{line_number}: duplicate sample_key {sample.sample_key}")
            seen.add(sample.sample_key)
            samples.append(sample)
    return samples

<<<<<<< Updated upstream

def _load_existing(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, object]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("schema_version") != DATASET_SCHEMA_VERSION:
                continue
            key = record.get("sample_key")
            if isinstance(key, str) and key:
                records[key] = record
    return records


def _record_matches_sample(record: dict[str, object], sample: FunctionSample) -> bool:
    return (
        record.get("raw_source") == sample.source
        and record.get("label") == sample.label
        and record.get("language") == sample.language
        and record.get("split") == sample.split
        and isinstance(record.get("semantic_facts"), str)
    )


def _write_jsonl_line(handle, value: dict[str, object]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def build_function_dataset(
    samples_path: str | Path,
    output_path: str | Path,
    *,
    joern_dir: str | Path = "/home/phy/joern",
    java_home: str | Path = "/home/phy/jdk21",
    timeout: int = 300,
) -> list[dict[str, object]]:
    samples = _read_samples(samples_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    errors_path = target.with_suffix(".errors.jsonl")

    existing = _load_existing(target)
    reusable = {
        sample.sample_key: existing[sample.sample_key]
        for sample in samples
        if sample.sample_key in existing and _record_matches_sample(existing[sample.sample_key], sample)
    }

    records: list[dict[str, object]] = []
    with target.open("w") as output:
        for sample in samples:
            record = reusable.get(sample.sample_key)
            if record is not None:
                records.append(record)
                _write_jsonl_line(output, record)

    resumed = len(records)
    print(f"function_samples_total={len(samples)} resumed={resumed}", flush=True)

    failures = 0
    with target.open("a") as output, errors_path.open("w") as errors:
        completed = {str(record["sample_key"]) for record in records}
        for sample in samples:
            if sample.sample_key in completed:
                continue
            try:
                parsed = parse_function(sample.source, sample.language, sample.function_name)
            except ValueError as error:
                failures += 1
                _write_jsonl_line(errors, {"sample_key": sample.sample_key, "stage": "parse", "error": str(error)})
                print(f"function_sample_failed={sample.sample_key} stage=parse error={error}", flush=True)
                continue

            try:
                graph = extract_function_cpg(
                    sample.source,
                    parsed.name,
                    language=sample.language,
                    joern_dir=joern_dir,
                    java_home=java_home,
                    timeout=timeout,
                )
            except (CPGError, subprocess.TimeoutExpired, OSError) as error:
                failures += 1
                _write_jsonl_line(errors, {"sample_key": sample.sample_key, "stage": "joern", "error": str(error)})
                print(f"function_sample_failed={sample.sample_key} stage=joern error={error}", flush=True)
                continue

            semantics = extract_vulnerability_semantics(graph)
            relations = graph_relations(graph)
=======
            key, source, label, language, function_name, split = _sample_fields(raw)
            if key in samples:
                raise ValueError(f'{samples_path}:{line_number}: duplicate sample_key={key}')
            samples[key] = (line_number, source, label, language, function_name, split)

    records: list[dict[str, object]] = []
    completed = set()
    truncate_at = None
    needs_newline = False
    if target.exists():
        with target.open('rb') as handle:
            size = target.stat().st_size
            while line := handle.readline():
                start = handle.tell() - len(line)
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    if handle.tell() == size and not line.endswith(b'\n'):
                        truncate_at = start
                        break
                    raise ValueError(f'{target}: invalid saved JSON at byte {start}') from error
                if not isinstance(record, dict):
                    raise ValueError(f'{target}: saved record must be an object at byte {start}')
                key = record.get('sample_key')
                if not isinstance(key, str) or key not in samples or key in completed:
                    raise ValueError(f'{target}: unknown or duplicate saved sample_key={key!r}')
                _, source, label, language, function_name, split = samples[key]
                parsed = parse_function(source, language, function_name)
                expected = {'raw_source': source, 'label': label, 'language': language,
                            'function_name': parsed.name, 'split': split}
                if (any(record.get(field) != value for field, value in expected.items())
                        or type(record.get('label')) is not int
                        or not isinstance(record.get('graph'), str) or not record['graph'].strip()):
                    raise ValueError(f'{target}: saved record does not match input or lacks graph: {key}')
                records.append(record)
                completed.add(key)
                needs_newline = not line.endswith(b'\n')

    target.parent.mkdir(parents=True, exist_ok=True)
    if truncate_at is not None:
        with target.open('r+b') as handle:
            handle.truncate(truncate_at)
        print(f'discard_incomplete_output_tail={target} byte={truncate_at}', flush=True)
    print(f'function_samples_total={len(samples)} resumed={len(completed)}', flush=True)
    failures = 0
    with target.open('a', encoding='utf-8') as handle, errors_path.open('w', encoding='utf-8') as errors:
        if needs_newline:
            handle.write('\n')
        for key, (line_number, source, label, language, function_name, split) in samples.items():
            if key in completed:
                continue
            stage = 'syntax'
            try:
                parsed = parse_function(source, language, function_name)
            except ValueError as error:
                failure = error
            else:
                failure = None
            if failure is None:
                stage = 'joern'
                try:
                    graph = extract_function_cpg(
                        source,
                        parsed.name,
                        language=language,
                        joern_dir=joern_dir,
                        java_home=java_home,
                        timeout=timeout,
                    )
                except (CPGError, subprocess.TimeoutExpired) as error:
                    failure = error
            if failure is not None:
                failures += 1
                errors.write(json.dumps({
                    'sample_key': key, 'line': line_number, 'label': label,
                    'split': split, 'language': language, 'stage': stage,
                    'error_type': type(failure).__name__, 'error': str(failure),
                }, ensure_ascii=False) + '\n')
                errors.flush()
                os.fsync(errors.fileno())
                print(f'function_sample_failed={key} stage={stage} error={failure}', flush=True)
                continue
>>>>>>> Stashed changes
            record: dict[str, object] = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "sample_key": sample.sample_key,
                "label": sample.label,
                "language": sample.language,
                "function_name": parsed.name,
                "raw_source": sample.source,
                "graph": render_graph(graph),
                "semantic_facts": semantics.render(),
                "semantic_tags": list(semantics.tags),
                "relation_count": len(relations),
                "semantic_fact_count": len(semantics.facts),
                "split": sample.split,
            }
<<<<<<< Updated upstream
=======
            if split is not None:
                record['split'] = split
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
>>>>>>> Stashed changes
            records.append(record)
            completed.add(sample.sample_key)
            _write_jsonl_line(output, record)
            print(
                f"function_sample_done={sample.sample_key} label={sample.label} "
                f"relations={len(relations)} semantic_facts={len(semantics.facts)}",
                flush=True,
            )

<<<<<<< Updated upstream
    total = len(samples)
    success = len(records)
    rate = success / total if total else 0.0
    print(
        f"function_build_summary total={total} success={success} failed={failures} "
        f"success_rate={rate:.2%} errors={errors_path}",
        flush=True,
    )
=======
    rate = len(records) / len(samples) if samples else 0.0
    print(f'function_build_summary total={len(samples)} success={len(records)} '
          f'failed={failures} success_rate={rate:.2%} errors={errors_path}', flush=True)
>>>>>>> Stashed changes
    return records
