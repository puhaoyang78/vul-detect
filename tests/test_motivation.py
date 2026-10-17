from collections import Counter
import unittest

from vulnmechanism.motivation import match_lengths, shortcut_features


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


if __name__ == "__main__":
    unittest.main()
