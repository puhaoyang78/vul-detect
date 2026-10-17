from collections import Counter
import json
import math
from pathlib import Path
import tempfile
import unittest

import joblib

from vulnmechanism.artifacts import write_json
from vulnmechanism.data import write_jsonl
from vulnmechanism.metrics import metrics
from vulnmechanism.motivation import (
    SCORE_BINS, code_tokens, match_lengths, match_tfidf, shortcut_features,
    tfidf_experiment, tfidf_report,
)


class Tokenizer:
    def __call__(self, source, **kwargs):
        return {"input_ids": source.split()}


class MotivationTests(unittest.TestCase):
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
