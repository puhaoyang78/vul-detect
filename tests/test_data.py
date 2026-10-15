import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest

from vulnmechanism.data import (
    balanced,
    deduplicate,
    json_array,
    load_native,
    prepare,
    shingles,
    tokens,
)


def row(key, source, label=0, split="train"):
    return dict(sample_key=key, raw_source=source, label=label, split=split)


class DataTests(unittest.TestCase):
    def test_prepare_rebuilds_published_metadata_but_refuses_existing_source_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw/PrimeVul_v0.1"
            raw.mkdir(parents=True)
            for split_index, split in enumerate(("train", "valid", "test")):
                rows = [
                    dict(
                        idx=split_index * 2 + y,
                        target=y,
                        func=f"int f{split_index}_{y}(){{return {split_index * 2 + y};}}",
                    )
                    for y in (0, 1)
                ]
                (raw / f"primevul_{split}.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows)
                )
            output = root / "data/primevul"
            output.mkdir(parents=True)
            (output / "members.csv").write_text("published reference metadata\n")
            (output / "statistics.json").write_text("{}")
            stats = prepare(root / "raw", root / "data", "primevul")
            self.assertTrue(
                all(stats["balanced"][s]["samples"] == 2 for s in ("train", "valid", "test"))
            )
            self.assertTrue((output / "members.csv").read_text().startswith("sample_key,"))
            with self.assertRaises(FileExistsError):
                prepare(root / "raw", root / "data", "primevul")

    def test_lexical_normalization_preserves_literals_and_boundaries(self):
        self.assertEqual(tokens("int f(){/* note */return 1;}"), tokens("int  f() { return 1; }"))
        self.assertNotEqual(tokens("a + +b"), tokens("a++b"))
        self.assertEqual(tokens('"// not a comment"'), ('"// not a comment"',))
        self.assertEqual(tokens('R"tag(/* literal */)tag"'), ('R"tag(/* literal */)tag"',))
        self.assertEqual(tokens("x // comment\\\n continued\n y"), ("x", "y"))

    def test_native_sources_and_official_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "PrimeVul_v0.1").mkdir()
            for i, split in enumerate(("train", "valid", "test")):
                (root / "PrimeVul_v0.1" / f"primevul_{split}.jsonl").write_text(
                    json.dumps(dict(idx=i, target=i % 2, func="unparseable fragment", project="P"))
                    + "\n"
                )
            rows, projects = load_native(root, "primevul")
            self.assertIsNone(projects)
            self.assertEqual([r["split"] for r in rows], ["train", "valid", "test"])
            self.assertEqual([r["label"] for r in rows], [0, 1, 0])
            self.assertTrue(all(r["raw_source"] == "unparseable fragment" for r in rows))
            (root / "megavul").mkdir()
            raw = [
                dict(
                    is_vul=bool(i % 2),
                    repo_name=f"P{i // 2}",
                    func=f"negative {i}",
                    func_before=f"positive {i}",
                    file_path="f.c",
                )
                for i in range(40)
            ]
            (root / "megavul/megavul.json").write_text(json.dumps(raw))
            rows, projects = load_native(root, "megavul")
            self.assertEqual(
                {s: len(v) for s, v in projects.items()}, dict(train=16, valid=2, test=2)
            )
            self.assertEqual(projects, load_native(root, "megavul")[1])
            for a, b in itertools.combinations(projects.values(), 2):
                self.assertFalse(set(a) & set(b))
            for i, r in enumerate(rows):
                self.assertEqual(r["raw_source"], f"positive {i}" if i % 2 else f"negative {i}")
                self.assertEqual(r["label"], i % 2)

    def test_conflicts_transitivity_and_heldout_priority_against_brute_force(self):
        base = [f"t{i}" for i in range(200)]
        variants = []
        for i in range(15):
            changed = base.copy()
            changed[i : i + 2] = [f"x{i}", f"y{i}"]
            variants.append(" ".join(changed))
        rows = [
            row(str(i), code, int(i == 14), ["train", "valid", "test"][i % 3])
            for i, code in enumerate(variants)
        ]
        rows += [
            row("same_train", 'int unique(){return "//literal";}', split="train"),
            row("same_test", 'int unique() { /* comment */ return "//literal"; }', split="test"),
            row("independent", "void entirely_different(double p){consume(p);}", 1, "valid"),
        ]
        # Reference connected components use the real token tuples, no index.
        components = [{i} for i in range(len(rows))]
        sets = [shingles(tokens(r["raw_source"])) for r in rows]
        for i, j in itertools.combinations(range(len(rows)), 2):
            if 10 * len(sets[i] & sets[j]) >= 9 * len(sets[i] | sets[j]):
                a = next(c for c in components if i in c)
                b = next(c for c in components if j in c)
                if a is not b:
                    a.update(b)
                    components.remove(b)
        priority = dict(test=0, valid=1, train=2)
        expected = sorted(
            min(c, key=lambda i: (priority[rows[i]["split"]], i))
            for c in components
            if len({rows[i]["label"] for i in c}) == 1
        )
        kept, stats = deduplicate(rows)
        self.assertEqual([r["sample_key"] for r in kept], [rows[i]["sample_key"] for i in expected])
        self.assertEqual([r["sample_key"] for r in kept], ["same_test", "independent"])
        self.assertEqual(stats["conflict_rows"], 15)

    def test_index_matches_brute_force_on_different_set_lengths(self):
        rng = random.Random(42)
        sources = []
        for i in range(30):
            base = [f"t{j}" for j in range(80 + i % 4 * 7)]
            for _ in range(i % 5):
                base[rng.randrange(len(base))] = f"r{rng.randrange(100)}"
            sources.append(" ".join(base))
        for i, j in itertools.combinations(range(len(sources)), 2):
            a, b = (shingles(tokens(sources[k])) for k in (i, j))
            near = 10 * len(a & b) >= 9 * len(a | b)
            kept, _ = deduplicate([row("a", sources[i]), row("b", sources[j])])
            self.assertEqual(len(kept), 1 if near else 2)

    def test_balancing_is_per_split_deterministic_and_preserves_labels(self):
        rows = [
            row(f"{s}-{i}", str(i), int(i < 3), s)
            for s in ("train", "valid", "test")
            for i in range(10)
        ]
        selected = balanced(rows)
        self.assertEqual(selected, balanced(rows))
        for s in ("train", "valid", "test"):
            self.assertEqual(
                [sum(r["split"] == s and r["label"] == y for r in selected) for y in (0, 1)], [3, 3]
            )
        self.assertTrue(all(r in rows for r in selected))

    def test_streaming_array_handles_chunk_boundaries_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "array.json"
            rows = [dict(value="x" * (1024 * 1024 + 5)), dict(value="最后")]
            path.write_text(json.dumps(rows))
            self.assertEqual(list(json_array(path)), rows)
            path.write_text('[{"value": 1}')
            with self.assertRaises(ValueError):
                list(json_array(path))


if __name__ == "__main__":
    unittest.main()
