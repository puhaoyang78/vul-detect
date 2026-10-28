import unittest

from vulnmechanism.data import modified_lines
from vulnmechanism.losver_patch_audit import (
    apply_commit_patch, candidate_region, edit_hunks, edit_shape,
    function_catalog, lexical_code_lines, semantic_edit_family,
)


class PatchAuditTests(unittest.TestCase):
    def test_addition_has_no_vulnerable_diff_label(self):
        old = "int f(char *p, int n) {\n  return p[n];\n}\n"
        new = "int f(char *p, int n) {\n  if (n < 0) return -1;\n  return p[n];\n}\n"
        dv, dp = modified_lines(old,new), modified_lines(new,old)
        self.assertEqual(dv,[])
        self.assertEqual(dp,[1])
        self.assertEqual(edit_shape(dv,dp,edit_hunks(old,new)),"addition_only")
        region = candidate_region(old,new,dv,dp)
        self.assertEqual(region["status"],"ok")
        self.assertIn(1,region["region_lines"])
        self.assertGreater(region["insert_variable_matches"],0)
        self.assertTrue(any(r["certainty"]=="heuristic_candidate" for r in region["relations"]))

    def test_binding_scope_does_not_link_shadowed_definition(self):
        old = "int f(int n) {\n  int x=n;\n  {\n    int x=9;\n    use(x);\n  }\n  return x;\n}\n"
        new = old.replace("return x;","return x+1;")
        region = candidate_region(old,new,modified_lines(old,new),modified_lines(new,old))
        self.assertEqual(region["status"],"ok")
        self.assertIn(1,region["region_lines"])
        self.assertNotIn(3,region["region_lines"])
        self.assertNotIn(4,region["region_lines"])

    def test_enclosing_condition_is_header_not_entire_branch(self):
        old = "int f(char *p,int n) {\n  if (n>0) {\n    log(n);\n    return p[n];\n  }\n  return 0;\n}\n"
        new = old.replace("p[n]","p[n-1]")
        region = candidate_region(old,new,modified_lines(old,new),modified_lines(new,old))
        self.assertIn(1,region["region_lines"])
        self.assertNotIn(2,region["region_lines"])
        self.assertNotIn(5,region["region_lines"])

    def test_unknown_parsing_and_correspondence_are_not_success(self):
        r = candidate_region("int f(","int f(int n) { return n; }",[],[0])
        self.assertEqual(r["status"],"parse_or_function_boundary_failure")
        self.assertFalse(r["region_lines"])
        a,b = "int f(){return 0;}","int g(){return 1;}"
        r = candidate_region(a,b,[0],[0])
        self.assertEqual(r["status"],"function_correspondence_unknown")
        self.assertFalse(r["region_lines"])

    def test_semantic_categories_separate_check_cleanup_and_state(self):
        old = "int f(char *p,int n) {\n  return p[n];\n}\n"
        variants = [
            ("  if (n<0) return -1;\n", "addition_check_candidate"),
            ("  free(p);\n", "addition_other"),
            ("  int flag=1;\n  if (n<0) return -1;\n", "addition_other"),
            ("  OP_REQUIRES(ctx,n>0,error());\n", "addition_check_candidate"),
        ]
        for insertion,expected in variants:
            with self.subTest(insertion=insertion):
                new=old.replace("  return",insertion+"  return")
                record=dict(shape="addition_only",diff_p=modified_lines(new,old),diff_v=[])
                self.assertEqual(semantic_edit_family(record,{"func":old},{"func":new}),expected)

    def test_whole_function_patch_replay_and_rename(self):
        old = "int f(int x) {\n  return x;\n}\n"
        new = "int g(int x) {\n  return x+1;\n}\n"
        patch="@@ -1,3 +1,3 @@\n-int f(int x) {\n-  return x;\n+int g(int x) {\n+  return x+1;\n }"
        after,forward=apply_commit_patch(old,patch)
        before,reverse=apply_commit_patch(new,patch,reverse=True)
        self.assertEqual(after,"\n".join(s.strip() for s in new.splitlines()))
        self.assertEqual(before,"\n".join(s.strip() for s in old.splitlines()))
        self.assertTrue(forward and reverse)

    def test_ambiguous_hunk_is_not_applied(self):
        source="int f(){\n  use(x);\n  use(x);\n}\n"
        result,witness=apply_commit_patch(source,"@@ -1 +1 @@\n-  use(x);\n+  use(y);")
        self.assertIsNone(result)
        self.assertEqual(witness,[])

    def test_qualified_class_and_preprocessor_identity(self):
        source="namespace T {\nclass Forward {public:int Compute(){return 1;}};\nclass Gradient {public:int Compute(){return 2;}};\n}\n#ifdef WINDOWS\nint f(){return 1;}\n#else\nint f(){return 2;}\n#endif\n"
        catalog=function_catalog(source)
        compute=[r for r in catalog if r["name"]=="Compute"]
        self.assertEqual(len(compute),2)
        self.assertNotEqual(compute[0]["context"],compute[1]["context"])
        f=[r for r in catalog if r["name"]=="f"]
        self.assertEqual(len(f),2)
        self.assertNotEqual(f[0]["context"],f[1]["context"])

    def test_comment_lines_and_crlf_are_not_executable_changes(self):
        code='int f(){\r\n/* comment\r\n still comment */\r\n return "// literal";\r\n}\r\n'
        self.assertEqual(lexical_code_lines(code),{0,3,4})


if __name__ == "__main__":
    unittest.main()
