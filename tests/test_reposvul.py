"""Native labels, conservative repairs and official split leakage inspection."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_losver import Tokens
from vulnmechanism.data import read_jsonl, write_jsonl
from vulnmechanism.reposvul import _raw_rows, inspect


def native(i, before=None, after=None):
    before = before or f"int f{i}() {{\n return {i + 1};\n}}"
    after = after or f"int f{i}() {{\n return {i + 2};\n}}"
    ordinary = f"int g{i}() {{\n return {i};\n}}"
    return dict(index=i, project="p", cve_id=f"CVE-{i}", commit_id=f"fix-{i}", details=[dict(
        file_language="c", file_name=f"{i}.c", target=0,
        code_before=before+"\n"+ordinary, code=after+"\n"+ordinary,
        function_before=[dict(function=before,target=1),dict(function=ordinary,target=0)],
        function_after=[dict(function=after,target=0),dict(function=ordinary,target=0)])])


class ReposVulTests(unittest.TestCase):
    def test_native_labels_not_file_labels_and_no_ordinal_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"ReposVul_c.jsonl"
            data=native(0)
            data["details"][0]["function_after"].reverse()
            write_jsonl(p,[data])
            rows,pairs,stats=_raw_rows(p)
            self.assertEqual([r["target"] for r in rows],[1,0,0,0])
            self.assertEqual(len(pairs),1)
            self.assertTrue(pairs[0][1]["sample_key"].endswith("after:1"))
            self.assertEqual(stats["synthesized_classification_members"],0)
            data["details"][0]["function_after"][1]["function"]="int different() {return 0;}"
            write_jsonl(p,[data])
            rows,pairs,stats=_raw_rows(p)
            self.assertEqual(len(rows),4)
            self.assertEqual(pairs,[])
            self.assertEqual(stats["issues"]["positive_missing_or_ambiguous_declaration"],1)

    def test_invalid_native_label_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"ReposVul_c.jsonl"; data=native(0)
            data["details"][0]["function_before"][0]["target"]="1"
            write_jsonl(p,[data])
            with self.assertRaisesRegex(ValueError,"invalid native 0/1 label"):
                _raw_rows(p)

    def test_unchanged_pure_addition_truncation_and_no_training_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=[native(i) for i in range(10)]
            same="int f0() {\n return 0;\n}"
            data[0]=native(0,same,same)
            data[1]=native(1,"int f1() {\n return 1;\n}","int f1() {\n if (1) return 0;\n return 1;\n}")
            before="int f2() {\n"+"\n".join(f" int v{i} = {i};" for i in range(600))+"\n return 2;\n}"
            data[2]=native(2,before,before.replace("return 2;","return 3;"))
            write_jsonl(root/"ReposVul_c.jsonl",data)
            with patch("transformers.RobertaTokenizerFast.from_pretrained",return_value=Tokens()):
                result=inspect(root,root/"out","tiny")
            self.assertEqual(result["class_counts"],{"1":10,"0":30})
            self.assertEqual(result["coverage"]["identical_source_opposite_label_pairs"],1)
            self.assertGreaterEqual(result["label_quality"]["exact"]["conflicting_groups"],1)
            records=read_jsonl(root/"out/repair_pairs.jsonl")
            self.assertTrue(records[0]["identical_512_model_input"])
            self.assertFalse(records[0]["sides"]["before"]["effective_region"])
            self.assertEqual(records[1]["sides"]["before"]["diff_lines"],0)
            self.assertGreater(records[1]["sides"]["after"]["diff_lines"],0)
            self.assertTrue(records[2]["sides"]["before"]["truncated"])
            self.assertFalse(records[2]["sides"]["before"]["effective_region"])
            self.assertTrue(result["grouped_split_feasibility"]["feasible"])
            self.assertFalse(result["training_started"])
            self.assertEqual(list((root/"out").glob("train*.jsonl")),[])

    def test_official_overlap_and_unavailable_repairs_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=[native(i) for i in range(10)]
            write_jsonl(root/"ReposVul_c.jsonl",data)
            row=dict(function_id="same-id",function=data[0]["details"][0]["function_before"][0]["function"],
                target=1,project="p",commit_id="fix-0",cve_id="CVE-0",file_name="0.c",file_language="c")
            for split in ("train","valid","test"):
                write_jsonl(root/f"official/{split}.jsonl",[row])
            with patch("transformers.RobertaTokenizerFast.from_pretrained",return_value=Tokens()):
                result=inspect(root,root/"out","tiny",official_dir=root/"official")
            official=result["official"]
            self.assertFalse(official["reliable_grouped_split"])
            self.assertEqual(official["overlap"]["train:test"]["exact_source"],1)
            self.assertEqual(official["overlap"]["train:valid"]["cve"],1)
            self.assertEqual(official["splits"]["train"]["counts"]["unique_local_changed_counterpart"],1)


if __name__ == "__main__":
    unittest.main()
