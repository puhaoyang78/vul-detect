"""Source identities, JSON artifacts and the unchanged checkpoint schema."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

DATASET_SCHEMA_VERSION = 9

def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()

def identity(record: dict) -> dict:
    return {"sample_key": record["sample_key"], "dataset": record["dataset"],
            "split": record["split"], "label": record["label"],
            "source_sha256": source_hash(record["raw_source"])}

def iter_jsonl(path: str | Path, *, description=None, expected_sha256=None):
    """Parse, optionally checksum and report progress in one bounded-memory pass."""
    from .progress import training_bar
    import sys
    path = Path(path)
    checksum = hashlib.sha256() if expected_sha256 is not None else None
    started = time.monotonic()
    if description:
        print(f"{description} · {path.name}", flush=True)
    with path.open('rb') as handle, training_bar(None, total=path.stat().st_size,
            desc=description or 'Read', unit='B', unit_scale=True,
            bar_format='{desc}  {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} '
                       '[{elapsed}<{remaining}, {rate_fmt}]',
            disable=not description or not sys.stdout.isatty()) as progress:
        for number, line in enumerate(handle, 1):
            if checksum is not None:checksum.update(line)
            if line.strip():
                try:
                    row = json.loads(line.decode('utf-8'))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                yield row
            progress.update(len(line))
    if checksum is not None and checksum.hexdigest() != expected_sha256:
        raise ValueError(f'{path}: cache identity/schema mismatch (SHA256)')
    if description:
        print(f"{description} · complete ({time.monotonic()-started:.1f}s)", flush=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
