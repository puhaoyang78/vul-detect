import unittest

from vulnmechanism.syntax import (static_relations, parse_source, statement_units, text, walk,
    statement_dependence_graph, bounded_dependence_slice, bidirectional_pair_slice,
    align_statement_units, SLICE_RULES, targeted_pair_slice)

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


class StaticPatternTests(unittest.TestCase):
    def patterns(self,code):
        return static_relations(code)["relations"]

    def test_null_branch_and_early_exit_are_not_safety_verdicts(self):
        good="int f(int *p){ if(!p) return 0; return *p; }"
        bad="int f(int *p){ if(!p) return *p; return 0; }"
        a=[r for r in self.patterns(good) if r["pattern"]=="null_check_relation"]
        b=[r for r in self.patterns(bad) if r["pattern"]=="null_check_relation"]
        self.assertEqual(a[0]["zero_case"],"excluded")
        self.assertEqual(b[0]["zero_case"],"allowed")
        self.assertEqual(a[0]["safety"],"unknown")

    def test_bound_index_requires_same_binding(self):
        good="int f(int *p,int i){ if(i<0) return 0; return p[i]; }"
        wrong="int f(int *p,int i,int j){ if(j<0) return 0; return p[i]; }"
        shadow="int f(int *p,int i){ if(i>0) { int i=9; return p[i]; } return 0; }"
        self.assertTrue(any(r["pattern"]=="boundary_check_relation" for r in self.patterns(good)))
        self.assertFalse(any(r["pattern"]=="boundary_check_relation" for r in self.patterns(wrong)))
        self.assertFalse(any(r["pattern"]=="boundary_check_relation" for r in self.patterns(shadow)))

    def test_mutation_and_address_escape_invalidate_checks(self):
        for change in ("i=100;","i++;","mutate(&i);"):
            code=f"int f(int *p,int i){{ if(i<0) return 0; {change} return p[i]; }}"
            self.assertFalse(any(r["pattern"]=="boundary_check_relation" for r in self.patterns(code)))

    def test_field_paths_are_not_interchangeable(self):
        code="int f(struct S *s){ if(s->length>0) return use(s->other); return 0; }"
        self.assertFalse(any(r["pattern"]=="control_call_relation" for r in self.patterns(code)))

    def test_memory_api_alone_does_not_emit_extent(self):
        a="int f(char *d,char *s,int n){memcpy(d,s,n);return 0;}"
        b="int f(char *s,int n){char d[16];memcpy(d,s,n);return 0;}"
        self.assertFalse(any(r["pattern"]=="memory_extent_relation" for r in self.patterns(a)))
        matched=[r for r in self.patterns(b) if r["pattern"]=="memory_extent_relation"]
        self.assertEqual(matched[0]["capacity"],"16")
        self.assertEqual(matched[0]["extent_comparison"],"unknown")

    def test_computed_argument_definition_rejects_conditional_and_stale(self):
        good="int f(int n){int z=n*2;return use(z);}"
        bad="int f(int n){int z=n*2;z=1;return use(z);}"
        conditional="int f(int n){int z; if(n) {z=n*2;} return use(z);}"
        self.assertTrue(any(r["pattern"]=="call_argument_relation" for r in self.patterns(good)))
        self.assertFalse(any(r["pattern"]=="call_argument_relation" for r in self.patterns(bad)))
        self.assertFalse(any(r["pattern"]=="call_argument_relation" for r in self.patterns(conditional)))

    def test_resource_relations_need_object_and_path(self):
        a="int f(){char *p=malloc(8);free(p);return 0;}"
        self.assertTrue(any(r["role"]=="allocation_to_release" for r in self.patterns(a)))
        good="int f(int *p){free(p);return *p;}"
        self.assertTrue(any(r["role"]=="release_to_access_candidate" for r in self.patterns(good)))
        for code in ("int f(int *p){free(p);p=0;return *p;}",
                     "int f(int *p,int c){if(c)free(p);return *p;}"):
            self.assertFalse(any(r["role"]=="release_to_access_candidate" for r in self.patterns(code)))

    def test_integer_pointer_and_float_are_not_integer_range(self):
        for kind in ("float","int *"):
            code=f"int f({kind} n){{if(n>0) return use(n+1);return 0;}}"
            self.assertFalse(any(r["pattern"]=="integer_range_relation" for r in self.patterns(code)))

    def test_crlf_unicode_and_byte_ranges(self):
        code="int f(int *p){\r\n// 中文\r\nif(p) return *p;\r\nreturn 0;\r\n}"
        r=next(r for r in self.patterns(code) if r["pattern"]=="null_check_relation")
        self.assertEqual(r["operation"]["lines"],[2])
        span=r["operation"]
        self.assertEqual(code.encode()[span["start_byte"]:span["end_byte"]].decode(),"*p")

    def test_multidimensional_capacity_keeps_all_dimensions(self):
        code="int f(){unsigned char ColorMap[3][256];memset(ColorMap,0,3*256);return 0;}"
        r=next(r for r in self.patterns(code) if r["pattern"]=="memory_extent_relation")
        self.assertEqual(r["capacity_dimensions"],["3","256"])
        self.assertEqual(r["capacity"],"3 * 256")
        self.assertEqual(r["capacity_bytes"],"sizeof(ColorMap)")
        self.assertEqual(r["capacity_unit"],"elements")

    def test_form_feed_uses_existing_physical_splitlines_coordinates(self):
        code="int f(int *p){\f if(p) return *p;\nreturn 0;}"
        r=next(r for r in self.patterns(code) if r["pattern"]=="null_check_relation")
        self.assertEqual(r["operation"]["lines"],[1])
        self.assertIn("*p",code.splitlines()[1])

    def test_jump_over_definition_is_unknown(self):
        code="int f(int n){int z=0;goto done;z=n*2;done:return use(z);}"
        self.assertFalse(any(r["pattern"]=="call_argument_relation" for r in self.patterns(code)))

    def test_added_bound_check_has_pair_pattern_without_old_diff(self):
        from vulnmechanism.losver_patch_audit import ast_pattern_changes
        a="int f(int *p,int i){return p[i];}"
        b="int f(int *p,int i){if(i<0)return 0;return p[i];}"
        change=ast_pattern_changes(static_relations(a),static_relations(b),a,b)
        self.assertIn("boundary_relation_change",change["patterns"])
        comment=b.replace("return p[i]","/* same */ return p[i]")
        self.assertFalse(ast_pattern_changes(static_relations(b),static_relations(comment),b,comment)["patterns"])

    def test_parse_failure_and_source_only_signature(self):
        self.assertEqual(static_relations("int f(")["status"],"parse_unknown")
        self.assertFalse(static_relations("int f(")["relations"])
        with self.assertRaises(TypeError):
            static_relations("int f(){return 0;}",diff=[0])


class BidirectionalSliceTests(unittest.TestCase):
    def graph(self,source,links):
        parsed=parse_source(source)
        units,_=statement_units(parsed)
        nodes={}
        for i,u in enumerate(units):
            n=u["node"]
            line=n.start_point.row+1
            prefix=parsed["raw"].split(b"\n")[line-1][:n.start_point.column].decode()
            nodes[str(i)]=dict(kind="CALL",NAME="<operator>.test",CODE=text(n,parsed["raw"]),
                LINE_NUMBER=line,COLUMN_NUMBER=len(prefix.encode("utf-16-le"))//2+1)
        edges=[dict(source=str(a),target=str(b),kind=k,variable=var) for a,b,k,var in links]
        return statement_dependence_graph(source,dict(status="ok",nodes=nodes,edges=edges,counts=dict(CFG=1)))

    def test_pure_addition_maps_controlled_access_and_backward_length(self):
        old="int f(int *p,int n){\n int len=n*2;\n return p[len];\n}\n"
        new="int f(int *p,int n){\n int len=n*2;\n if(len<0)return 0;\n return p[len];\n}\n"
        a=self.graph(old,[(0,1,"REACHING_DEF","len")])
        b=self.graph(new,[(1,3,"CDG",""),(0,3,"REACHING_DEF","len")])
        r=bidirectional_pair_slice(old,new,a,b)
        self.assertEqual(r["v"]["seed_units"],[])
        self.assertIn(2,r["v"]["region_lines"])
        self.assertIn(1,r["v"]["region_lines"])
        self.assertTrue(any(e["kind"]=="VERSION_MAP" for u in r["v"]["selected_units"] for e in u["path"]))

    def test_no_dependence_does_not_fill_neighbour(self):
        old="int f(int *p){\n log(1);\n return *p;\n}\n"
        new=old.replace(" log(1);"," if(p==0)return 0;\n log(1);")
        r=bidirectional_pair_slice(old,new,self.graph(old,[]),self.graph(new,[]))
        self.assertEqual(r["v"]["region_lines"],[])

    def test_coordinates_crlf_unicode_and_mismatch(self):
        source='int f(int n){\r\n log("中文"); return n;\r\n}\n'
        g=self.graph(source,[(0,1,"REACHING_DEF","n")])
        self.assertEqual(len(g["edges"]),1)
        self.assertEqual(g["units"][1]["lines"],[1])
        bad=dict(status="ok",nodes={"0":dict(kind="CALL",CODE="wrong",NAME="wrong",LINE_NUMBER=2,COLUMN_NUMBER=2)},edges=[],counts={})
        self.assertEqual(statement_dependence_graph(source,bad)["mapped_nodes"],0)

    def test_alias_dependency_is_explicitly_unknown(self):
        s="int f(int *p){\n p[0]=2;\n return p[0];\n}"
        g=self.graph(s,[(0,1,"REACHING_DEF","p[0]")])
        self.assertEqual(g["edges"][0]["certainty"],"unknown")
        self.assertIn("alias_or_memory_state_unknown",g["edges"][0]["unknown"])

    def test_call_arguments_do_not_hide_unknown_call_summary(self):
        source="int f(int n){\n int x=read(n);\n return x;\n}"
        nodes={
            "call":dict(kind="CALL",NAME="read",CODE="read(n)",LINE_NUMBER=2,COLUMN_NUMBER=8),
            "arg":dict(kind="IDENTIFIER",NAME="n",CODE="n",LINE_NUMBER=2,COLUMN_NUMBER=13),
            "ret":dict(kind="IDENTIFIER",NAME="x",CODE="x",LINE_NUMBER=3,COLUMN_NUMBER=9),
        }
        g=statement_dependence_graph(source,dict(status="ok",nodes=nodes,
            edges=[dict(source="arg",target="ret",kind="REACHING_DEF",variable="n")],counts=dict(CFG=1,REACHING_DEF=1)))
        self.assertIn("external_call_summary_unknown",g["edges"][0]["unknown"])

    def test_backward_does_not_start_from_forward_descendants(self):
        g=dict(edges=[dict(source=0,target=1,kind="CDG"),dict(source=2,target=1,kind="REACHING_DEF")])
        r=bounded_dependence_slice(g,[0])
        self.assertIn(1,r["selected"])
        self.assertNotIn(2,r["selected"])

    def test_retained_paths_are_not_replaced_or_reexpanded(self):
        g=dict(edges=[dict(source=i,target=i+1,kind="CDG") for i in range(5)])
        first=bounded_dependence_slice(g,[0])
        self.assertEqual(set(first["selected"]),{0,1,2})
        final=bounded_dependence_slice(g,[0],retained=first["selected"],roots={99:dict(path=[dict(kind="VERSION_MAP")],direction="cross_version")})
        self.assertTrue(set(first["selected"]).issubset(final["selected"]))
        self.assertNotIn(3,final["selected"])
        self.assertEqual(first["selected"][2],final["selected"][2])

    def test_bounds_and_cfg_adjacency_are_not_dependency(self):
        g=dict(edges=[dict(source=0,target=i,kind="CDG") for i in range(1,50)])
        r=bounded_dependence_slice(g,[0])
        self.assertEqual(len(r["selected"]),1+SLICE_RULES["max_extra_statements"])
        self.assertTrue(r["limit_reached"])
        g=self.graph("int f(){\n a();\n b();\n}",[(0,1,"CFG","")])
        self.assertEqual(g["edges"],[])

    def test_duplicate_statements_do_not_map_arbitrarily(self):
        a="int f(){\n use(1);\n use(1);\n}"
        b="int f(){\n if(x)use(1);\n}"
        ga,gb=self.graph(a,[]),self.graph(b,[])
        self.assertEqual(align_statement_units(a,b,ga,gb),[])

    def test_known_bad_pair_cannot_transfer_supervision(self):
        a="int f(int *p){\n return p[0];\n}"
        b=a.replace(" return"," if(!p)return 0;\n return")
        ga,gb=self.graph(a,[]),self.graph(b,[(0,2,"CDG","")])
        r=bidirectional_pair_slice(a,b,ga,gb,allow_mapping=False)
        self.assertFalse(r["mapping_allowed"])
        self.assertEqual(r["v"]["region_lines"],[])

    def test_unparsed_source_keeps_unknown_not_fallback(self):
        r=statement_dependence_graph("int f(",dict(status="ok",counts={}))
        self.assertEqual(r["status"],"parse_or_method_unknown")
        self.assertEqual(r["units"],[])


class TargetedSliceTests(unittest.TestCase):
    def native(self,source,links):
        """Small explicit native graph fixture, with original byte coordinates."""
        parsed=parse_source(source);units,_=statement_units(parsed);nodes={}
        for i,u in enumerate(units):
            n=u["node"]
            # Model native expression endpoints, including operands in returns.
            children=[c for c in walk(n) if c.type in {"identifier","subscript_expression","binary_expression","pointer_expression"}]
            n=next((c for c in children if c.type in {"subscript_expression","binary_expression","pointer_expression"}),n)
            prefix=parsed["raw"].split(b"\n")[n.start_point.row][:n.start_point.column].decode()
            nodes[str(i)]=dict(kind="CALL",NAME="<operator>.fixture",CODE=text(n,parsed["raw"]),
                LINE_NUMBER=n.start_point.row+1,COLUMN_NUMBER=len(prefix.encode("utf-16-le"))//2+1)
        nodes["entry"]=dict(kind="METHOD",CODE=source)
        edges=[dict(source=str(a),target=str(b),kind=k,variable=var) for a,b,k,var in links]
        edges += [dict(source="entry",target="0",kind="CFG",variable="")]
        edges += [dict(source=str(i),target=str(i+1),kind="CFG",variable="") for i in range(len(units)-1)]
        cpg=dict(status="ok",nodes=nodes,edges=edges,counts=dict(CFG=len(units),REACHING_DEF=sum(e["kind"]=="REACHING_DEF" for e in edges),CDG=sum(e["kind"]=="CDG" for e in edges)))
        return statement_dependence_graph(source,cpg),cpg

    def pair(self,old,new,links=(),old_links=(),bypass=False):
        gv,cv=self.native(old,old_links);gp,cp=self.native(new,links)
        if bypass:
            cp["edges"].append(dict(source="entry",target=str(len(gp["units"])-1),kind="CFG",variable=""))
        return targeted_pair_slice(old,new,gv,gp,cv,cp)

    def test_for_header_endpoints_get_exact_units(self):
        source="int f(int n){\n int i;\n for(i=0;i<n;i++){n+=i;}\n return n;\n}"
        g,c=self.native(source,[])
        kinds=[u["kind"] for u in g["units"]]
        self.assertIn("for_initializer",kinds)
        self.assertIn("for_update",kinds)
        update=next(u for u in g["units"] if u["kind"]=="for_update")
        self.assertEqual(source.encode()[update["start_byte"]:update["end_byte"]],b"i++")

    def test_parameter_output_loss_is_a_boundary_not_bad_code_mapping(self):
        source="int f(int n){\n return n;\n}"
        _,cpg=self.native(source,[])
        cpg["nodes"]["out"]=dict(kind="METHOD_PARAMETER_OUT",CODE="int n",LINE_NUMBER=1,COLUMN_NUMBER=7)
        cpg["edges"].append(dict(source="0",target="out",kind="REACHING_DEF",variable="n"))
        g=statement_dependence_graph(source,cpg)
        self.assertEqual(g["mapping_stats"]["REACHING_DEF:boundary_unmapped"],1)
        self.assertEqual(g["mapping_stats"].get("REACHING_DEF:internal_unmapped",0),0)

    def test_target_retains_access_not_unrelated_cdg_sibling(self):
        a="int f(int*p,int i,int j){\n log(j);\n return p[i];\n}"
        b=a.replace(" log(j);"," if(i<0)return 0;\n log(j);")
        r=self.pair(a,b,[(0,2,"CDG",""),(0,3,"CDG","")])
        self.assertEqual(r["status"],"ok")
        self.assertIn(2,r["v"]["region_lines"])
        self.assertNotIn(1,r["v"]["region_lines"])
        self.assertTrue(r["targets"][0]["cfg"]["dominates"])

    def test_control_without_matching_operand_is_unknown(self):
        a="int f(int*p,int i,int j){\n return p[i];\n}"
        b=a.replace(" return"," if(j<0)return 0;\n return")
        r=self.pair(a,b,[(0,2,"CDG","")])
        self.assertEqual(r["v"]["region_lines"],[])
        self.assertGreater(r["reasons"]["no_bound_operand_or_value_path"],0)

    def test_same_array_base_is_not_an_index_check(self):
        a="int f(int*p,int i){\n return p[i];\n}"
        b=a.replace(" return"," if(p[0]<0)return 0;\n return")
        r=self.pair(a,b,[(0,2,"CDG","")])
        self.assertEqual(r["v"]["region_lines"],[])

    def test_checked_element_does_not_check_shared_address_index(self):
        # Train 2721 counterexample: counts[i] and offsets[i] are different
        # values. A native control edge and the spelling i do not link them.
        a="int f(struct S*s,int i){\n return s->offsets[i];\n}"
        b=a.replace(" return", " if(s->counts[i]>100)return 0;\n return")
        r=self.pair(a,b,[(0,2,"CDG","")])
        self.assertEqual(r["v"]["region_lines"],[])
        self.assertEqual(r["reasons"]["added_checks_without_resolved_tested_value"],1)

    def test_opaque_check_call_does_not_validate_its_argument(self):
        a="int f(int*p,int i){\n return p[i];\n}"
        b=a.replace(" return", " if(check(i))return 0;\n return")
        self.assertEqual(self.pair(a,b,[(0,2,"CDG","")])["v"]["region_lines"],[])

    def test_integer_work_without_memory_consumer_is_not_a_target(self):
        a="int f(int n){\n int x=n*2;\n return x;\n}"
        b=a.replace(" int x"," if(n<0)return 0;\n int x")
        r=self.pair(a,b,[(0,2,"CDG","")])
        self.assertEqual(r["v"]["region_lines"],[])

    def test_shadowed_target_binding_does_not_transfer(self):
        a="int f(int*p,int n){\n {\n return p[n];\n }\n}"
        b="int f(int*p,int n){\n {\n int n=4;\n if(n<0)return 0;\n return p[n];\n }\n}"
        r=self.pair(a,b,[(1,3,"CDG","")])
        self.assertEqual(r["v"]["region_lines"],[])

    def test_cfg_bypass_rejects_apparent_cdg(self):
        a="int f(int*p,int i){\n return p[i];\n}"
        b=a.replace(" return"," if(i<0)return 0;\n return")
        r=self.pair(a,b,[(0,2,"CDG","")],bypass=True)
        self.assertEqual(r["v"]["region_lines"],[])
        self.assertIn("cfg_dominance_not_established",r["reasons"])

    def test_changed_operand_and_field_paths_are_rejected(self):
        a="int f(int*p,int i){\n i=9;\n return p[i];\n}"
        b=a.replace(" i=9;"," if(i<0)return 0;\n i=9;")
        self.assertEqual(self.pair(a,b,[(0,3,"CDG","")])["v"]["region_lines"],[])
        a="int f(struct S*s,int*p){\n return p[s->other];\n}"
        b=a.replace(" return"," if(s->length<0)return 0;\n return")
        self.assertEqual(self.pair(a,b,[(0,2,"CDG","")])["v"]["region_lines"],[])

    def test_backward_sources_need_actual_matching_definition(self):
        a="int f(int*p,int n){\n int i=n*2;\n return p[i];\n}"
        b=a.replace(" int i", " if(n<0)return 0;\n int i")
        r=self.pair(a,b,[(0,3,"CDG",""),(2,3,"REACHING_DEF","i")],[(0,1,"REACHING_DEF","i")])
        self.assertEqual(r["status"],"ok")
        self.assertEqual(set(r["v"]["region_lines"]),{1,2})
        self.assertTrue(any(p["kind"]=="support" and any(e["kind"]=="REACHING_DEF" for e in p["edges"])
                            for u in r["v"]["selected_units"] for p in u["paths"]))

    def test_duplicate_or_different_context_has_no_correspondence(self):
        a="int f(int*p,int n,int x){\n if(x){return p[n];}\n return 0;\n}"
        b="int f(int*p,int n,int x){\n if(n<0)return 0;\n if(!x){return p[n];}\n return 0;\n}"
        r=self.pair(a,b,[(0,3,"CDG","")])
        self.assertEqual(r["v"]["region_lines"],[])

    def test_missing_graph_or_known_bad_pair_never_adds_labels(self):
        a="int f(){\nreturn 0;\n}";b=a.replace("return","if(1)return 1;\nreturn")
        gv,cv=self.native(a,[]);gp,cp=self.native(b,[])
        r=targeted_pair_slice(a,b,gv,gp,cv,cp,allow_mapping=False)
        self.assertEqual(r["status"],"unknown")
        self.assertEqual(r["v"]["region_lines"],[])

    def test_budget_keeps_complete_targets_without_cdg_siblings(self):
        a="int f(int*p,int i){\n"+"\n".join(f" int x{j}=p[i];" for j in range(18))+"\nreturn 0;\n}"
        b=a.replace("\n int x0", "\n if(i<0)return 0;\n int x0")
        r=self.pair(a,b,[(0,j+2,"CDG","") for j in range(18)])
        self.assertEqual(len(r["v"]["selected_units"]),12)
        self.assertTrue(r["v"]["limit_reached"])
        selected={u["id"] for u in r["p"]["selected_units"]}
        for target in r["targets"]:
            self.assertTrue({e[k] for e in target["control_path"] for k in ("source","target")}<=selected)


if __name__ == "__main__":
    unittest.main()
