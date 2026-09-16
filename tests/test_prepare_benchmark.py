import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from vulnmechanism.prepare_benchmark import (
    NearIndex,
    canonicalize_commits,
    clean_label_conflicts,
    commit_disjoint,
    commit_link,
    digest,
    link_counterparts,
    normalized_source,
    overlap_report,
    prepare,
    prime_provenance,
    repository,
    select,
    shingles,
    source_language,
    temporal_split,
    unit,
    validate,
)


def make(dataset, key, split, code, label=1, commit="", date="2000-01-01T00:00:00+00:00"):
    row = dict(
        dataset=dataset,
        sample_key=key,
        split=split,
        function=code,
        label=label,
        language="c",
        repo="github.com/o/r",
        commit=commit,
        official_split=split,
        commit_time=date,
    )
    if dataset == "primevul":
        return unit([row])
    row["pair_id"] = key
    fixed = dict(row, sample_key=key + ":after", label=0, function=code.replace("return 1;", "return 0;"))
    return unit([row, fixed])


class BenchmarkTests(unittest.TestCase):
    def test_lexical_normalization(self):
        self.assertEqual(
            normalized_source("int f(){ /* x */ return 1; }"),
            normalized_source("int\r\nf ( ) {return 1 ;}// tail"),
        )
        for first, second in [
            ('"a b"', '"ab"'),
            ("a + +b", "a++b"),
            ("foo bar", "foobar"),
            ("return 1;", "return 2;"),
        ]:
            self.assertNotEqual(normalized_source(first), normalized_source(second))

    def test_repository_and_commit_identity(self):
        self.assertEqual(repository("https://GitHub.com/Owner/Repo.git/"), "github.com/owner/repo")
        self.assertEqual(
            commit_link("github.com/Owner/Repo/commit/" + "a" * 40),
            ("github.com/owner/repo", "a" * 40),
        )
        with self.assertRaises(ValueError):
            commit_link("github.com/o/r/commit/short")

    def test_short_commit_resolution(self):
        full = "abcdef1" + "0" * 33
        first = make("primevul", "a", "train", "int a(){return 1;}", commit=full)
        second = make("cleanvul", "b", "test", "int b(){return 1;}", commit="abcdef1")
        report = canonicalize_commits([first, second])
        self.assertEqual(report["expanded_function_rows"], 2)
        self.assertEqual(first["rows"][0]["commit"], second["rows"][0]["commit"])

    def test_gitiles_provenance(self):
        sha = "a" * 40
        row = dict(
            project_url="None",
            commit_url="https://android.googlesource.com/platform/x/+/" + sha,
            commit_id=sha,
        )
        repo, commit, evidence = prime_provenance(row)
        self.assertEqual(repo, "android.googlesource.com/platform/x")
        self.assertEqual(commit, sha)
        self.assertEqual(evidence["repo_evidence"], "gitiles_commit_url")

    def test_language_evidence(self):
        self.assertEqual(source_language("EXPORT int f() {}", "x.c"), ("c", "extension"))
        self.assertEqual(source_language("int f() {}", "x.C"), ("cpp", "extension"))
        self.assertIsNone(source_language("int f() {}", "x.java")[0])

    def test_temporal_split_keeps_commit_together(self):
        rows = [
            make(
                "cleanvul",
                f"p{i}",
                "unassigned",
                f"int f{i}() {{return 1;}}",
                commit=str(i // 2),
                date=f"2000-01-{i // 2 + 1:02}T00:00:00+00:00",
            )
            for i in range(20)
        ]
        assigned, report = temporal_split(rows)
        memberships = {}
        for item in assigned:
            memberships.setdefault(item["rows"][0]["commit"], set()).add(item["split"])
        self.assertTrue(all(len(splits) == 1 for splits in memberships.values()))
        self.assertEqual(
            [report["before_leakage_filter"][split]["units"] for split in ("train", "valid", "test")],
            [14, 2, 4],
        )

    def test_test_priority_pair_atomicity_and_balance(self):
        external = make("sven", "external", "external_test", "int ext(){return 1;}", commit="ext")
        pair = make("cleanvul", "pair", "train", "int ext(){return 1;}")
        rows = [
            external,
            pair,
            make("primevul", "v", "train", "int v(){return 1;}", commit="v"),
            make("primevul", "b1", "train", "int b1(){return 0;}", label=0),
        ]
        chosen, removed = select(rows)
        self.assertIn("external", {item["id"] for item in chosen})
        self.assertIn("pair", {entry["unit_id"] for entry in removed})
        validate(chosen)

    def test_official_split_duplicate_is_removed_from_lower_priority_side(self):
        rows = [
            make("primevul", "vtrain", "train", "int b(){return 1;}", commit="same"),
            make("primevul", "vtest", "test", "int b(){return 1;}", commit="same"),
            make("primevul", "btest", "test", "int c(){return 0;}", label=0),
        ]
        chosen, removed = select(rows)
        self.assertEqual({item["id"] for item in chosen}, {"vtest", "btest"})
        self.assertEqual(removed[0]["unit_id"], "vtrain")

    def test_same_commit_different_function_is_reported_not_removed(self):
        rows = [
            make("sven", "external", "external_test", "int ext(){return 1;}", commit="same"),
            make("primevul", "v", "train", "int other(){return 1;}", commit="same"),
            make("primevul", "b", "train", "int benign(){return 0;}", label=0, commit="other"),
        ]
        chosen, _ = select(rows)
        self.assertEqual(len(chosen), 3)
        self.assertTrue(all(name.endswith(": repo_commit") for name in overlap_report(chosen)))

    def test_strict_commit_disjoint_removes_training_overlap(self):
        sha = "a" * 40
        external = make("sven", "ext", "external_test", "int ext(){return 1;}", commit=sha)
        vulnerable = make("primevul", "v", "train", "int other(){return 1;}", commit=sha)
        benign = make("primevul", "b", "train", "int benign(){return 0;}", label=0, commit="b" * 40)
        candidates = [external, vulnerable, benign]
        main, _ = select(candidates)
        strict, excluded = commit_disjoint(main, candidates)
        self.assertNotIn("v", {item["id"] for item in strict})
        self.assertTrue(any(entry["unit_id"] == "v" for entry in excluded))

    def test_counterpart_label_conflict_quarantine(self):
        before = "int a(){return 1;}"
        after = "int a(){return 0;}"
        edges = [(digest(normalized_source(before)), digest(normalized_source(after)))]
        rows = [
            make("primevul", "v", "train", before),
            make("primevul", "b", "train", after, label=0),
            make("primevul", "contradiction", "test", before, label=0),
        ]
        remaining, removed = clean_label_conflicts(link_counterparts(rows, edges))
        self.assertEqual(remaining, [])
        self.assertEqual({entry["reason"] for entry in removed}, {"label_conflict", "label_conflict_counterpart"})

    def test_near_duplicate_index_matches_bruteforce(self):
        base = "int f(int x){" + "".join(f"x += {i};" for i in range(80)) + "return x;}"
        index = NearIndex()
        previous = []
        for i in range(10):
            code = base.replace("x += 40;", f"x += {100 + i};") if i else base
            item = make("primevul", str(i), "train", code)
            signature = shingles(code)
            brute = any(len(signature & other) * 10 >= len(signature | other) * 9 for other in previous)
            self.assertEqual(bool(index.matches(item)), brute)
            index.add(item)
            previous.append(signature)

    def test_real_schema_repeatability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            prime = root / "PrimeVul_v0.1"
            prime.mkdir(parents=True)
            (prime / "file_info.json").write_text("{}")
            for split_index, split in enumerate(("train", "valid", "test")):
                rows = [
                    dict(
                        idx=split_index * 10 + i,
                        func=f"int f{split_index}_{i}() {{return {i % 2};}}",
                        target=i % 2,
                        file_name="x.c",
                        func_hash=i,
                        project_url="https://github.com/o/prime",
                        commit_id=(str(split_index + 1) * 40)[:40],
                    )
                    for i in range(4)
                ]
                (prime / f"primevul_{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                (prime / f"primevul_{split}_paired.jsonl").write_text("")

            clean = root / "cleanvul"
            clean.mkdir()
            columns = ["func_before", "func_after", "vulnerability_score", "extension", "date", "commit_url", "file_name"]
            for score in (4, 3):
                with (clean / f"vulnerability_score_{score}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    for i in range(10):
                        writer.writerow(
                            dict(
                                func_before=f"int c{score}_{i}() {{return 1;}}",
                                func_after=f"int c{score}_{i}() {{return 0;}}",
                                vulnerability_score=score,
                                extension="c",
                                date=f"200{i}-01-01T00:00:00Z",
                                commit_url=f"https://github.com/o/clean/commit/{score}{i:039x}",
                                file_name="x.c",
                            )
                        )

            sven = root / "sven/data_train_val/train"
            sven.mkdir(parents=True)
            row = dict(
                func_src_before="int ext(){return 1;}",
                func_src_after="int ext(){return 0;}",
                file_name="x.cpp",
                func_name="ext",
                vul_type="cwe-125",
                commit_link="github.com/o/sven/commit/" + "a" * 40,
            )
            (sven / "cwe-125.jsonl").write_text(json.dumps(row) + "\n")
            output = Path(tmp) / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                first = prepare(root, output, 4)
                before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
                second = prepare(root, output, 4)
            self.assertEqual(first, second)
            self.assertEqual(before, {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main()
