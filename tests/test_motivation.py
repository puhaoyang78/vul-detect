from collections import Counter
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from vulnmechanism.artifacts import write_json
from vulnmechanism.data import write_jsonl
from vulnmechanism.metrics import metrics
from vulnmechanism.motivation import (
    SCORE_BINS, code_tokens, match_lengths, match_tfidf, shortcut_features,
    tfidf_experiment, tfidf_report, _input_conflicts, _project_auc_interval,
    _transfer_native_metadata, transfer_diagnostics, write_table,
)


class Tokenizer:
    def __call__(self, source, **kwargs):
        return {"input_ids": source.split()}


class MotivationTests(unittest.TestCase):
    def test_transfer_diagnosis_joins_fixed_members_and_preserves_baselines(self):
        class DiagnosticTokenizer:
            pad_token_id, eos_token_id = 0, 2

            def __call__(self, source, max_length=None, **kwargs):
                encode = lambda s: [ord(c) for c in s][:max_length]
                return dict(input_ids=encode(source) if isinstance(source, str) else [encode(s) for s in source])

        class LexicalModel:
            def predict_proba(self, sources):
                return np.array([[.2, .8] if "positive" in s else [.8, .2] for s in sources])

        def metadata(raw_root, selected):
            pairs = {}
            for dataset, values in selected.items():
                for r in values.values():
                    r.update(project_url=f"https://github.com/{dataset}/repo", known_fixed_negative=r["label"] == 0)
                pairs[dataset] = [(f"{dataset}:test-positive-{i}", f"{dataset}:test-negative-{i}") for i in (0, 1)]
            return pairs, {}, []

        from vulnmechanism.model import InputBuilder
        tokenizer = DiagnosticTokenizer()
        budget = 2048 - len(InputBuilder(tokenizer).source_prefix) - 2
        with tempfile.TemporaryDirectory() as temporary:
            data, output = Path(temporary) / "data", Path(temporary) / "results"
            root = output / "transfer_primevul_secvuleval"
            root.mkdir(parents=True)
            results, all_tests = [], {}
            for d in ("primevul", "secvuleval"):
                features = []
                for split in ("train", "valid", "test"):
                    rows = [dict(
                        sample_key=f"{d}:{split}-{name}-{i}", label=y, dataset=d, split=split,
                        raw_source=f"int {d}_{split}_{name}_{i}(){{\n" + "line\n" * i + "}",
                        project=d, train_overlap=[], valid_overlap=False,
                    ) for name, y in (("negative", 0), ("positive", 1)) for i in (0, 1)]
                    write_jsonl(data / d / f"{split}.jsonl", rows)
                    features.extend(dict(sample_key=r["sample_key"], label=r["label"],
                                         total_lines=len(r["raw_source"].splitlines()), codebert_tokens=len(r["raw_source"])) for r in rows)
                    if split == "test":
                        all_tests[d] = rows
                        write_table(root / f"{d}.members.csv", [dict(sample_key=r["sample_key"], label=r["label"], independent=True) for r in rows])
                write_json(data / d / "statistics.json", dict(raw_root="unused"))
                write_table(output / "motivation" / f"{d}.features.csv", features)
                write_json(output / "motivation" / f"{d}.tfidf.metrics.json", dict(members=dict(train=4, valid=4), selected_C=1., valid=dict(threshold=.5)))
            for source in all_tests:
                for method in ("bce", "lp_ft", "swa"):
                    run = output / source / method
                    write_json(run / "config.json", dict(input_protocol=dict(source_budget=budget)))
                    for target, rows in all_tests.items():
                        name = "test" if source == target else f"{target}.test"
                        saved = [dict(sample_key=r["sample_key"], label=r["label"], score=.8 if r["label"] else .2,
                                      logit=1. if r["label"] else -1., threshold=.5) for r in rows]
                        write_jsonl(run / f"{name}.predictions.jsonl", saved)
                        results.append(dict(view="independent", training_dataset=source, test_dataset=target, method=method,
                                            **metrics([r["label"] for r in saved], [r["score"] for r in saved], .5)))
            write_json(root / "results.json", dict(results=results))
            original = (root / "results.json").read_bytes()
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                 patch("vulnmechanism.motivation.joblib.load", return_value=LexicalModel()), \
                 patch("vulnmechanism.motivation._transfer_native_metadata", side_effect=metadata), \
                 patch("vulnmechanism.motivation._project_auc_interval", return_value=dict(low=1., high=1.)):
                transfer_diagnostics(data, output, "unused")
            summary = json.loads((root / "diagnosis.json").read_text())
            formal = [r for r in summary["performance"] if r["subset"] == "independent"]
            self.assertEqual(len(formal), 12)
            self.assertTrue(all(r["samples"] == 4 and r["positive"] == r["negative"] == 2 and r["auc"] == 1. for r in formal))
            self.assertEqual((root / "results.json").read_bytes(), original)
            self.assertNotIn("project_url", json.loads((data / "primevul/train.jsonl").read_text().splitlines()[0]))

    def test_identical_model_inputs_bound_accuracy_without_filtering(self):
        rows = [dict(sample_key=str(i), label=y) for i, y in enumerate((1, 0, 0, 1))]
        encoded = dict(zip((r["sample_key"] for r in rows), (b"prefix", b"prefix", b"prefix", b"other")))
        result = _input_conflicts(rows, encoded)
        self.assertEqual(result, dict(groups=1, samples=3, positive=1, minimum_errors=1, maximum_accuracy=.75))
        self.assertEqual(len(rows), 4)

    def test_project_bootstrap_keeps_pairs_and_known_ranking(self):
        rows = [dict(project_url=f"project-{i}", project="", label=y) for i in range(4) for y in (0, 1)]
        interval = _project_auc_interval(rows, [r["label"] for r in rows], repetitions=20)
        self.assertEqual((interval["low"], interval["high"]), (1., 1.))
        self.assertEqual(interval["usable_repetitions"], 20)
        reverse = _project_auc_interval(rows, [1-r["label"] for r in rows], repetitions=20)
        self.assertEqual((reverse["low"], reverse["high"]), (0., 0.))

    def test_native_audit_preserves_repair_pairs_and_flags_label_disagreement(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "SecVulEval/data").mkdir(parents=True)
            (root / "PrimeVul_v0.1").mkdir()
            shared, fixed = "int f(){return 0;}", "int f(){return 1;}"
            native = [dict(
                idx=i, is_vulnerable=y, func_body=code, fixed_func_idx=1.,
                project_url="https://github.com/Owner/Repo.git/", cwe_list=["CWE-1"],
                context={"Globals": []},
            ) for i, y, code in ((0, True, shared), (1, False, fixed))]
            pq.write_table(pa.Table.from_pylist(native), root / "SecVulEval/data/train-00000-of-00001.parquet")
            selected = {"primevul": {}, "secvuleval": {
                f"secvuleval:{r['idx']}": dict(label=int(r["is_vulnerable"]), raw_source=r["func_body"])
                for r in native
            }}
            for split in ("train", "valid", "test"):
                prime = [dict(idx=f"{split}-{i}", target=y, func=code,
                              project_url="https://github.com/owner/repo", cwe=[])
                         for i, y, code in ((0, 1, "another function"), (1, 0, shared))]
                for suffix in ("", "_paired"):
                    write_jsonl(root / f"PrimeVul_v0.1/primevul_{split}{suffix}.jsonl", prime)
                for r in prime:
                    selected["primevul"][f"primevul:{r['idx']}"] = dict(label=r["target"], raw_source=r["func"])
            pairs, statistics, conflicts = _transfer_native_metadata(root, selected)
            self.assertEqual(pairs["secvuleval"], [("secvuleval:0", "secvuleval:1")])
            self.assertTrue(selected["secvuleval"]["secvuleval:1"]["known_fixed_negative"])
            self.assertEqual(selected["secvuleval"]["secvuleval:0"]["project_url"], "https://github.com/owner/repo")
            self.assertEqual(statistics["selected_label_sets"], {"(0,):(1,)": 1})
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(len(selected["primevul"]), 6)
            selected["secvuleval"]["secvuleval:0"]["label"] = 0
            with self.assertRaisesRegex(ValueError, "prepared code/label"):
                _transfer_native_metadata(root, selected)

    def test_exact_matching_ignores_scores_and_excludes_unsupported_lengths(self):
        rows = [dict(sample_key=f"{length}-{label}-{i}", label=label,
            raw_source="\n".join(["code"] * length), score=i)
            for length, positive, negative in ((2, 8, 3), (5, 2, 7), (9, 4, 0))
            for label, n in ((0, negative), (1, positive)) for i in range(n)]
        selected = match_lengths(rows)
        self.assertEqual(len(selected), 10)
        self.assertEqual(selected, match_lengths(rows))
        self.assertEqual(Counter(len(r["raw_source"].splitlines()) for r in selected if r["label"] == 0),
            Counter(len(r["raw_source"].splitlines()) for r in selected if r["label"] == 1))
        changed = [dict(r, score=10000-r["score"]) for r in rows]
        self.assertEqual([r["sample_key"] for r in selected], [r["sample_key"] for r in match_lengths(changed)])

    def test_lexical_counts_respect_literals_comments_and_control_nesting(self):
        source = 'int f(int n) {\n /* if while */\n if(n) { while(n--) {} }\n const char *s="// literal";\n return 1;\n}'
        result = shortcut_features(dict(raw_source=source, language="c", file_name="f.c"), Tokenizer())
        self.assertEqual(result["total_lines"], 6)
        self.assertEqual(result["code_lines"], 5)
        self.assertEqual(result["comments"], 1)
        self.assertEqual(result["comment_lines"], 1)
        self.assertEqual(result["branches"], 1)
        self.assertEqual(result["loops"], 1)
        self.assertEqual(result["max_control_depth"], 2)
        self.assertEqual(result["string_literals"], 1)
        self.assertEqual(result["numeric_literals"], 1)
        self.assertEqual(result["parse_errors"], 0)

    def test_tfidf_tokens_preserve_cpp_lexemes_and_omit_comments(self):
        tokens = code_tokens('Obj::Next(x != 0, "// literal"); /* ignored */ // ignored\n')
        self.assertEqual(tokens, [
            "Obj", "::", "Next", "(", "x", "!=", "0", ",", '"// literal"', ")", ";",
        ])

    def test_score_matching_balances_fixed_bins_without_replacement(self):
        rows, scores = [], []
        for label, values in ((0, (0., .019, .02, 1.)), (1, (.01, .015, .018, .039, .999))):
            for i, score in enumerate(values):
                rows.append(dict(sample_key=f"{label}-{i}", label=label, raw_source="line", score=1-score))
                scores.append(score)
        selected = match_tfidf(rows, scores)
        self.assertEqual(len(selected), 8)
        self.assertEqual(len({r["sample_key"] for r in selected}), len(selected))
        self.assertEqual(selected, match_tfidf(rows, scores))
        by_key = dict(zip((r["sample_key"] for r in rows), scores))
        histograms = [Counter(min(int(by_key[r["sample_key"]] * SCORE_BINS), SCORE_BINS-1)
            for r in selected if r["label"] == y) for y in (0, 1)]
        self.assertEqual(histograms[0], histograms[1])
        changed = [dict(r, score=100-r["score"]) for r in rows]
        self.assertEqual([r["sample_key"] for r in selected],
            [r["sample_key"] for r in match_tfidf(changed, scores)])
        self.assertEqual(match_tfidf(rows, [0.] * 4 + [1.] * 5), [])

    def test_joint_matching_controls_exact_lines_and_score_bins(self):
        rows = [dict(sample_key=str(i), label=y, raw_source="\n".join(["line"] * length))
            for i, (y, length) in enumerate(((0, 1), (0, 2), (1, 1), (1, 3)))]
        selected = match_tfidf(rows, [.51] * 4, joint=True)
        self.assertEqual([r["sample_key"] for r in selected], ["0", "2"])

    def test_tfidf_vocabulary_and_idf_use_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root / "data", root / "results"
            splits = {}
            for split, n in (("train", 8), ("valid", 4), ("test", 4)):
                splits[split] = [dict(
                    sample_key=f"primevul:{split}:{i}", label=i % 2, split=split,
                    raw_source=f"int {split}Only() {{ return {'danger' if i % 2 else 'safe'}(x); }}",
                    project="fixture", source_file="fixture.jsonl", source_row=i+1, train_overlap=[],
                ) for i in range(n)]
                write_jsonl(data / "primevul" / f"{split}.jsonl", splits[split])
            bce = [dict(sample_key=r["sample_key"], label=r["label"], score=.8 if r["label"] else .2)
                for r in splits["test"]]
            write_jsonl(output / "primevul/bce/test.predictions.jsonl", bce)
            write_json(output / "primevul/bce/test.metrics.json", dict(metrics=metrics(
                [r["label"] for r in bce], [r["score"] for r in bce], .5,
            )))
            result = tfidf_experiment(data, output, "primevul")
            pipeline = joblib.load(output / "motivation/primevul.tfidf.joblib")
            vectorizer = pipeline[0]
            self.assertNotIn("validOnly", vectorizer.vocabulary_)
            self.assertNotIn("testOnly", vectorizer.vocabulary_)
            self.assertAlmostEqual(vectorizer.idf_[vectorizer.vocabulary_["danger"]], math.log(9/5)+1)
            self.assertIn("return danger", vectorizer.vocabulary_)
            self.assertEqual(result["comparisons"][1]["length_auc"], .5)
            self.assertEqual(result["comparisons"][3]["status"], "skipped_insufficient_samples")
            summary = tfidf_report(output, ["primevul"])
            self.assertEqual(summary, dict(datasets=1, classifiers=1, subsets=4))
            saved = json.loads((output / "motivation/primevul.tfidf.metrics.json").read_text())
            self.assertEqual(saved["test"], result["test"])


if __name__ == "__main__":
    unittest.main()
