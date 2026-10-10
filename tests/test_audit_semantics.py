import json
from pathlib import Path
import tempfile
import unittest

from vulnmechanism.audit_semantics import audit_mechanism_fidelity


def row(pair_id, suffix, label, candidates, split="test", source="int f(void){}"):
    return {
        "sample_key": f"{pair_id}:{suffix}",
        "dataset": "cleanvul",
        "pair_id": pair_id,
        "split": split,
        "label": label,
        "raw_source": source,
        "mechanism_context": "\n".join(
            ["[MECHANISM_CANDIDATE]"]
            + [f"{kind} {detail}" for key, kind, state, detail in candidates]
        ) if candidates else "[VULNERABILITY_MECHANISM_CONTEXT]\nNO_CPG_DERIVED_MECHANISM_EVIDENCE",
        "mechanism_items": [
            {
                "category": "MECHANISM_CANDIDATE",
                "kind": kind,
                "detail": detail,
                "key": key,
                "state": state,
            }
            for key, kind, state, detail in candidates
        ],
    }


class MechanismAuditTests(unittest.TestCase):
    def test_pair_candidate_state_and_detail_changes_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            rows = [
                row("p1", "before", 1, [
                    (
                        "bounds|buf|n",
                        "BOUNDS_FLOW",
                        "bound_condition=not_observed|static_violation=no",
                        "source=parameter:n sink=MEMORY_WRITE object=buf expression=n capacity=64 occurrences=2",
                    ),
                    (
                        "null|p",
                        "NULL_DEREFERENCE_FLOW",
                        "null_condition=not_observed",
                        "source=nullable_allocation:malloc sink=POINTER_DEREFERENCE object=p occurrences=1",
                    ),
                ], source="int f(){ return 1; }") ,
                row("p1", "after", 0, [
                    (
                        "bounds|buf|n",
                        "BOUNDS_FLOW",
                        "bound_condition=present|static_violation=no",
                        "source=parameter:n sink=MEMORY_WRITE object=buf expression=n capacity=128 occurrences=9",
                    ),
                ], source="int f(){ return 0; }") ,
                row("p2", "before", 1, [
                    (
                        "uaf|p",
                        "USE_AFTER_FREE_FLOW",
                        "path_without_redefinition=present",
                        "source=deallocation sink=POINTER_DEREFERENCE object=p occurrences=1",
                    ),
                ]),
                row("p2", "after", 0, []),
            ]
            path.write_text("".join(json.dumps(value) + "\n" for value in rows))
            report = audit_mechanism_fidelity(path, dataset="cleanvul")

        self.assertEqual(report["complete_build_pairs"], 2)
        self.assertEqual(report["pairs_with_candidate_removed"], 2)
        self.assertEqual(report["pairs_with_candidate_state_changed"], 1)
        self.assertEqual(report["pairs_with_candidate_detail_changed"], 1)
        self.assertEqual(report["pairs_with_model_context_changed"], 2)
        self.assertEqual(report["pairs_with_source_change"], 1)
        self.assertEqual(report["candidate_removed_by_kind"]["NULL_DEREFERENCE_FLOW"], 1)
        self.assertEqual(report["candidate_removed_by_kind"]["USE_AFTER_FREE_FLOW"], 1)
        self.assertEqual(report["candidate_detail_changed_by_kind"]["BOUNDS_FLOW"], 1)
        self.assertIn(
            "BOUNDS_FLOW:bound_condition=not_observed|static_violation=no->bound_condition=present|static_violation=no",
            report["candidate_state_changes"],
        )
        # occurrence count alone must not define a mechanism-detail change
        before = report["candidate_detail_changes"][0]["before"]
        after = report["candidate_detail_changes"][0]["after"]
        self.assertNotIn("occurrences=", before)
        self.assertNotIn("occurrences=", after)

    def test_occurrence_only_change_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            rows = [
                row("p1", "before", 1, [
                    (
                        "bounds|buf|n",
                        "BOUNDS_FLOW",
                        "bound_condition=not_observed|static_violation=no",
                        "source=parameter:n sink=MEMORY_WRITE object=buf expression=n occurrences=1",
                    ),
                ]),
                row("p1", "after", 0, [
                    (
                        "bounds|buf|n",
                        "BOUNDS_FLOW",
                        "bound_condition=not_observed|static_violation=no",
                        "source=parameter:n sink=MEMORY_WRITE object=buf expression=n occurrences=8",
                    ),
                ]),
            ]
            path.write_text("".join(json.dumps(value) + "\n" for value in rows))
            report = audit_mechanism_fidelity(path, dataset="cleanvul")
        self.assertEqual(report["pairs_with_candidate_detail_changed"], 0)
        self.assertEqual(report["pairs_with_model_context_changed"], 0)

    def test_incomplete_build_pair_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.jsonl"
            path.write_text(json.dumps(row(
                "p1", "before", 1,
                [(
                    "uaf|p",
                    "USE_AFTER_FREE_FLOW",
                    "path_without_redefinition=present",
                    "source=deallocation sink=POINTER_DEREFERENCE object=p occurrences=1",
                )],
            )) + "\n")
            report = audit_mechanism_fidelity(path, dataset="cleanvul")
        self.assertEqual(report["complete_build_pairs"], 0)
        self.assertEqual(report["incomplete_build_pairs"], 1)


class RepairAvailabilityTests(unittest.TestCase):
    def test_native_identity_heldout_exclusion_and_no_auto_truth(self):
        from unittest.mock import patch
        from vulnmechanism.audit_semantics import audit_repair_availability
        def pair(index,project='p',commit='c'):
            return [dict(idx=index,target=1,project=project,commit_id=commit,func='int f(){return 1;}'),
                    dict(idx=index+1,target=0,project=project,commit_id=commit,func='int f(){return 0;}')]
        good=pair(0);bad=pair(2);bad[1]['commit_id']='different'
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);(base/'megavul').mkdir();(base/'megavul/megavul.json').write_text('[]')
            (base/'cleanvul').mkdir()
            def records(path):
                name=Path(path).name
                if name=='cohort':return []
                if name=='primevul_train_paired.jsonl':return good+bad
                if name=='primevul_valid_paired.jsonl' or name=='pairs_all.jsonl':return []
                raise AssertionError('unexpected file, including test: '+name)
            with patch('vulnmechanism.audit_semantics._records',side_effect=records):
                r=audit_repair_availability(base,'cohort',base/'out')
            self.assertEqual(r['datasets']['PrimeVul:train']['native_pair_metadata_mismatch'],1)
            case=json.loads((base/'out/cases.jsonl').read_text())
            self.assertNotIn('safety_effect',case)
            self.assertEqual(case['split'],'train')
            def heldout(path):
                if Path(path).name=='cohort':return [dict(raw_source=good[0]['func'],dataset='primevul',split='test',sample_key='heldout')]
                return records(path)
            with patch('vulnmechanism.audit_semantics._records',side_effect=heldout):
                r=audit_repair_availability(base,'cohort',base/'excluded')
            self.assertEqual(r['excluded']['PrimeVul:heldout_source_overlap'],1)
            self.assertEqual((base/'excluded/cases.jsonl').read_text(),'')


class OperationAssociationTests(unittest.TestCase):
    def test_guard_intervention_requires_semantic_difference_and_persistence(self):
        from vulnmechanism.audit_semantics import operation_associations,checked_guard_interventions
        def check(source):
            ops=operation_associations(source,'c')
            for o in ops:o['visible']=True
            return checked_guard_interventions(dict(raw_source=source,operations=ops),'c')
        source='int f(int *p) { if(p) return *p; if(!p) return *p; return 0; }'
        eligible,pairs,_=check(source)
        self.assertEqual(len(eligible),2);self.assertEqual(len(pairs),1)
        self.assertNotEqual(*pairs[0]['predicate_values'])
        self.assertFalse(check('int f(int *p) { if(p) return *p; if(p!=0) return *p; return 0; }')[1])
        self.assertFalse(check('int f(int *p,int flag) { if(p && flag) return *p; if(p && !flag) return *p; return 0; }')[1])
        self.assertFalse(check('int f(int *p) { if(p) { p=0; return *p; } return 0; }')[0])
        self.assertFalse(check('int f(int *p) { if(p) { while(1) { return *p; } } return 0; }')[0])

    def test_bounds_references_and_no_lifetime_truth_from_null_guard(self):
        from vulnmechanism.audit_semantics import operation_associations,checked_guard_interventions
        source='int f(int *a,int i,int n) { if(i<n) return a[i]; if(i<=n) return a[i]; return 0; }'
        ops=operation_associations(source,'c')
        for o in ops:o['visible']=True
        eligible,pairs,_=checked_guard_interventions(dict(raw_source=source,operations=ops),'c')
        self.assertEqual(len(pairs),1)
        self.assertIn('reference0',pairs[0]['witness'])
        source='void f(int *p) { if(p) free(p); }'
        ops=operation_associations(source,'c')
        for o in ops:o['visible']=True
        self.assertFalse(checked_guard_interventions(dict(raw_source=source,operations=ops),'c')[0])

    def test_cpp_condition_wrapper_and_plain_store_are_checked(self):
        from vulnmechanism.audit_semantics import operation_associations,checked_guard_interventions
        for language in ('c','cpp'):
            source='int f(int *a,int i,int n) { if(i<n) { a[i]=0; } if(i<=n) { a[i]=1; } return 0; }'
            ops=operation_associations(source,language)
            for o in ops:o['visible']=True
            eligible,pairs,_=checked_guard_interventions(dict(raw_source=source,operations=ops),language)
            self.assertEqual(len(eligible),2)
            self.assertEqual(len(pairs),1)
            source='int f(int *p) { if(p) { *p=g(); } return 0; }'
            ops=operation_associations(source,language)
            for o in ops:o['visible']=True
            self.assertFalse(checked_guard_interventions(dict(raw_source=source,operations=ops),language)[0])

    def test_scoped_objects_and_unicode_positions(self):
        from vulnmechanism.audit_semantics import operation_associations
        source='int f(int *p) { /* 汉字 */ if(p) { *p=1; } { int *p=0; *p=2; } return 0; }'
        ops=operation_associations(source,'c')
        self.assertEqual(len(ops),2)
        self.assertNotEqual(ops[0]['binding'],ops[1]['binding'])
        self.assertTrue(any(c['role']=='if_statement:body' for c in ops[0]['context']))
        self.assertFalse(any(c['role'].startswith('if_statement') for c in ops[1]['context']))
        for op in ops:
            self.assertEqual(source[slice(*op['span'])],op['code'])
            self.assertEqual(op['status'],'ConfirmedAssociation')
            self.assertNotIn('label',op)
            self.assertNotIn('safe',op)

    def test_unresolved_objects_remain_unknown_and_api_is_not_duplicated(self):
        from vulnmechanism.audit_semantics import operation_associations
        source='int f(char *p, char *q, int n) { memcpy(p,q,n); return global->x; }'
        ops=operation_associations(source,'c')
        self.assertEqual([o['kind'] for o in ops],['memcpy:write','memcpy:read','field_indirection'])
        self.assertEqual(ops[-1]['status'],'Unknown')
        self.assertEqual(ops[-1]['reason'],'unresolved_member_global_or_declaration')

    def test_exchange_preserves_kind_origin_and_rejects_identity(self):
        from vulnmechanism.audit_semantics import operation_exchange
        def op(binding,text,kind='subscript'):
            return dict(status='ConfirmedAssociation',visible=True,kind=kind,origin='local',binding=binding,
                        context=[dict(role='for_statement:body')],context_text=text,local_text=text+'local')
        self.assertEqual(operation_exchange([op('a','x'),op('b','y')]),[0,1])
        self.assertEqual(operation_exchange([op('a','x'),op('a','y')]),[0,1])
        for second in (op('b','x'),op('b','y','free')):
            self.assertEqual(operation_exchange([op('a','x'),second]),[])

    def test_predicate_does_not_control_its_own_dereference(self):
        from vulnmechanism.audit_semantics import operation_associations
        source='int f(int *p) { if (*p) return *p; return sizeof(*p); }'
        ops=operation_associations(source,'c')
        self.assertFalse(any(c['role'].startswith('if_statement') for c in ops[0]['context']))
        self.assertTrue(any(c['role']=='if_statement:body' for c in ops[1]['context']))
        self.assertEqual(ops[0]['local_span'],(source.index('(*p)'),source.index('(*p)')+4))
        self.assertEqual(ops[2]['reason'],'unevaluated_or_address_only')

    def test_source_operand_guard_and_loop_update_are_not_value_proofs(self):
        from vulnmechanism.audit_semantics import operation_associations
        source='int f(int *p, int n) { if(n>0) { p=p+n; *p=1; } for(;p;p++) { *p=2; } return 0; }'
        ops=operation_associations(source,'c')
        self.assertTrue(any(c['code']=='(n>0)' for c in ops[0]['context']))
        update=next(c for c in ops[1]['context'] if c['code']=='p++')
        self.assertEqual(update['role'],'lexically_earlier_write_not_reaching_value')


if __name__ == "__main__":
    unittest.main()
