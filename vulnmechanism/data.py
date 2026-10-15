"""Independent native-label preparation for PrimeVul and MegaVul."""

from __future__ import annotations

from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re

from .artifacts import write_json

SPLITS = ("train", "valid", "test")
PRIORITY = {"test": 0, "valid": 1, "train": 2}
TOKEN = re.compile(
    r"(?P<space>\s+)|(?P<comment>//(?:\\\r?\n|[^\n])*|/\*[\s\S]*?\*/)"
    r'|(?:u8|u|U|L)?R"(?P<delimiter>[^\s()\\]{0,16})\([\s\S]*?\)(?P=delimiter)"'
    r'|(?:u8|u|U|L)?"(?:\\[\s\S]|[^"\\])*"'
    r"|(?:u8|u|U|L)?'(?:\\[\s\S]|[^'\\])*'"
    r"|[a-zA-Z_$\u0080-\uffff][\w$]*"
    r"|(?:\d|\.\d)(?:[eEpP][+-]|[\w.'])*"
    r"|>>=|<<=|<=>|->\*|\.\.\.|##|::|\.\*|->|\+\+|--|<<|>>|<=|>="
    r"|==|!=|&&|\|\||\*=|/=|%=|\+=|-=|&=|\^=|\|=|[^\s]"
)


def tokens(source):
    """Remove comments/spacing while preserving literals and token boundaries."""
    return tuple(
        m.group() for m in TOKEN.finditer(source) if m.lastgroup not in {"space", "comment"}
    )


def shingles(code_tokens):
    width = min(5, len(code_tokens))
    return {code_tokens[i : i + width] for i in range(max(1, len(code_tokens) - width + 1))}


def json_array(path):
    """Read the upstream MegaVul array without loading its graph/diff metadata."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as handle:
        buffer, position, eof = "", 0, False

        def space():
            nonlocal buffer, position, eof
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                block = handle.read(1024 * 1024)
                buffer, position, eof = block, 0, not block

        space()
        if buffer[position : position + 1] != "[":
            raise ValueError("MegaVul input must be a JSON array")
        position += 1
        space()
        first = True
        while buffer[position : position + 1] != "]":
            if not first:
                if buffer[position : position + 1] != ",":
                    raise ValueError("missing JSON array separator")
                position += 1
                space()
            while True:
                try:
                    value, position = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    block = handle.read(1024 * 1024)
                    buffer, position, eof = buffer[position:] + block, 0, not block
            if not isinstance(value, dict):
                raise ValueError("MegaVul rows must be objects")
            yield value
            first = False
            space()
        position += 1
        space()
        if buffer[position:] or handle.read(1):
            raise ValueError("unexpected content after JSON array")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_native(raw_root, dataset, seed=42):
    raw_root = Path(raw_root)
    rows = []
    paths = (
        [raw_root / "PrimeVul_v0.1" / f"primevul_{s}.jsonl" for s in SPLITS]
        if dataset == "primevul"
        else [raw_root / "megavul" / "megavul.json"]
    )
    for path in paths:
        if dataset == "primevul":
            with path.open(encoding="utf-8") as handle:
                iterator = (json.loads(line) for line in handle)
                _append_native(rows, iterator, dataset, path, raw_root)
        else:
            _append_native(rows, json_array(path), dataset, path, raw_root)
    projects = None
    if dataset == "megavul":
        names = sorted({r["project"] for r in rows})
        random.Random(seed).shuffle(names)
        n_train, n_valid = int(0.8 * len(names)), int(0.1 * len(names))
        projects = {
            s: sorted(v)
            for s, v in zip(
                SPLITS,
                (names[:n_train], names[n_train : n_train + n_valid], names[n_train + n_valid :]),
            )
        }
        assignment = {name: s for s, names in projects.items() for name in names}
        for row in rows:
            row["split"] = assignment[row["project"]]
    return rows, projects


def _append_native(rows, iterator, dataset, path, raw_root):
    for number, raw in enumerate(iterator, 1):
        label = raw["target"] if dataset == "primevul" else raw["is_vul"]
        if (dataset == "primevul" and (type(label) is not int or label not in (0, 1))) or (
            dataset == "megavul" and type(label) is not bool
        ):
            raise ValueError(f"{path}:{number}: invalid native label")
        source = raw["func"] if dataset == "primevul" or not label else raw["func_before"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{path}:{number}: missing native function")
        filename = raw.get("file_name" if dataset == "primevul" else "file_path") or ""
        if filename == "None":
            filename = ""
        suffix = Path(filename).suffix
        key = raw["idx"] if dataset == "primevul" else number
        project = raw.get("project" if dataset == "primevul" else "repo_name") or ""
        if dataset == "megavul" and not project.strip():
            raise ValueError(f"{path}:{number}: missing project")
        rows.append(
            dict(
                sample_key=f"{dataset}:{key}",
                dataset=dataset,
                split=path.stem.removeprefix("primevul_") if dataset == "primevul" else None,
                label=int(label),
                raw_source=source,
                file_name=filename,
                language=(
                    "c"
                    if suffix == ".c"
                    else (
                        "cpp"
                        if suffix in {".C", ".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx"}
                        else "c_cpp"
                    )
                ),
                project=project.strip().casefold(),
                source_file=str(path.relative_to(raw_root)),
                source_row=number,
            )
        )
        if number % 50000 == 0:
            print(f"{dataset} {path.name}: read {number}", flush=True)


def deduplicate(rows):
    """Exact Jaccard join, connected duplicate groups, then conflict removal.

    Compact shingle IDs are internal indexing keys, never artifact checksums.
    Every proposed near match is verified against the actual token tuples.
    All normalized groups enter the index, including eventual discarded groups,
    so transitive conflicting labels cannot survive behind a removed duplicate.
    """
    groups, by_normalized, exact = [], {}, {}
    stats = Counter()
    for i, row in enumerate(rows):
        source = row["raw_source"]
        if source in exact:
            group = exact[source]
            stats["exact_duplicate_rows"] += 1
        else:
            normalized = json.dumps(tokens(source), ensure_ascii=False)
            if normalized in by_normalized:
                group = by_normalized[normalized]
                stats["normalized_duplicate_rows"] += 1
            else:
                group = len(groups)
                by_normalized[normalized] = group
                groups.append([])
            exact[source] = group
        groups[group].append(i)
    del by_normalized, exact
    parents = list(range(len(groups)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    signatures = []
    frequency = Counter()
    for progress, indices in enumerate(groups, 1):
        signature = array(
            "Q",
            sorted(
                {
                    int.from_bytes(
                        hashlib.blake2b(
                            json.dumps(gram, ensure_ascii=False).encode(), digest_size=8
                        ).digest(),
                        "big",
                    )
                    for gram in shingles(tokens(rows[indices[0]]["raw_source"]))
                }
            ),
        )
        signatures.append(signature)
        frequency.update(signature)
        if progress % 10000 == 0:
            print(f"build shingle index: {progress}/{len(groups)}", flush=True)
    lengths = [len(s) for s in signatures]
    postings = defaultdict(list)
    for progress, i in enumerate(sorted(range(len(groups)), key=lambda j: (lengths[j], j)), 1):
        signature = signatures[i]
        # Rare-first global ordering sharply reduces frequent boilerplate hits.
        prefix = sorted(signature, key=lambda t: (frequency[t], t))[
            : len(signature) - (9 * len(signature) + 9) // 10 + 1
        ]
        minimum = (9 * len(signature) + 9) // 10
        candidates = set()
        for token in prefix:
            posting = postings[token]
            start = bisect_left(posting, minimum, key=lambda j: lengths[j])
            candidates.update(posting[start:])
        current = set(signature)
        actual = None
        for j in sorted(candidates):
            if root(i) == root(j):
                continue
            other = signatures[j]
            common = len(current.intersection(other))
            if common * 10 < (len(signature) + len(other) - common) * 9:
                continue
            if actual is None:
                actual = shingles(tokens(rows[groups[i][0]]["raw_source"]))
            previous = shingles(tokens(rows[groups[j][0]]["raw_source"]))
            overlap = len(actual.intersection(previous))
            if overlap * 10 >= len(actual.union(previous)) * 9:
                parents[root(i)] = root(j)
        for token in prefix:
            postings[token].append(i)
        if progress % 10000 == 0:
            print(f"near-duplicate join: {progress}/{len(groups)}", flush=True)
    components = defaultdict(list)
    for i, indices in enumerate(groups):
        components[root(i)].extend(indices)
    stats["near_duplicate_groups_merged"] = len(groups) - len(components)
    kept = []
    for indices in components.values():
        if len({rows[i]["label"] for i in indices}) > 1:
            stats["conflict_groups"] += 1
            stats["conflict_rows"] += len(indices)
        else:
            selected = min(indices, key=lambda i: (PRIORITY[rows[i]["split"]], i))
            kept.append(selected)
            stats["same_label_duplicate_rows_removed"] += len(indices) - 1
    return [rows[i] for i in sorted(kept)], dict(stats)


def balanced(rows, seed=42):
    selected = set()
    for split in SPLITS:
        classes = [
            [i for i, r in enumerate(rows) if r["split"] == split and r["label"] == y]
            for y in (0, 1)
        ]
        count = min(map(len, classes))
        if not count:
            raise ValueError(f"{split}: cannot balance an empty class")
        rng = random.Random(seed)
        for members in classes:
            selected.update(rng.sample(members, count))
    return [row for i, row in enumerate(rows) if i in selected]


def counts(rows):
    return {
        s: dict(
            samples=sum(r["split"] == s for r in rows),
            positive=sum(r["split"] == s and r["label"] == 1 for r in rows),
            negative=sum(r["split"] == s and r["label"] == 0 for r in rows),
        )
        for s in SPLITS
    }


def prepare(raw_root, output, dataset, seed=42):
    root = Path(output) / dataset
    if any(root.glob("*.jsonl")):
        raise FileExistsError(
            f"{root}: source lists already exist; use an output directory without prepared JSONL files"
        )
    rows, projects = load_native(raw_root, dataset, seed)
    before = counts(rows)
    print(f"{dataset} native: {before}", flush=True)
    clean, removed = deduplicate(rows)
    subset = balanced(clean, seed)
    for split in SPLITS:
        write_jsonl(root / f"{split}.jsonl", (r for r in subset if r["split"] == split))
        write_jsonl(root / f"{split}.natural.jsonl", (r for r in clean if r["split"] == split))
    keys = {r["sample_key"] for r in subset}
    with (root / "members.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "sample_key",
            "split",
            "label",
            "project",
            "source_file",
            "source_row",
            "balanced",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in clean:
            writer.writerow(
                {**{k: row[k] for k in fields[:-1]}, "balanced": int(row["sample_key"] in keys)}
            )
    stats = dict(
        dataset=dataset,
        seed=seed,
        raw_root=str(raw_root),
        native=before,
        removed=removed,
        natural=counts(clean),
        balanced=counts(subset),
        deduplication="within dataset; exact, lexical normalization, token 5-gram Jaccard >= 0.90; "
        "connected components; conflicting labels removed; test > valid > train",
        sampling="random.Random(42), stable source order, sample each class to the minority count",
    )
    if projects is not None:
        write_json(root / "projects.json", projects)
        stats["projects"] = {s: len(v) for s, v in projects.items()}
    write_json(root / "statistics.json", stats)
    print(json.dumps(stats, indent=2), flush=True)
    return stats
