import unittest
from vulnmechanism.cpg import resolve_target_graph, TargetMethodError, CPGQualityError, GraphNode
from vulnmechanism.syntax import parse_function, target_hint, source_tokens
from vulnmechanism.semantics import _node_operations


def fixture(source='int C::f() { return 1; }'):
    end = len(source.encode('utf-16-le')) // 2
    body_start = source.index('{')
    nodes = {
        '0': dict(kind='FILE', NAME='input.cpp', CONTENT=source),
        '1': dict(kind='METHOD', NAME='f', FILENAME='input.cpp', FULL_NAME='C.f:int()',
                  LINE_NUMBER=1, LINE_NUMBER_END=len(source.splitlines()), COLUMN_NUMBER=1,
                  COLUMN_NUMBER_END=len(source.splitlines()[-1].encode('utf-16-le'))//2, OFFSET=0,
                  OFFSET_END=end, IS_EXTERNAL=False, CODE=source[:997]+'...' if len(source)>1000 else source),
        '2': dict(kind='BLOCK', CODE=source[body_start:], OFFSET=body_start, OFFSET_END=end),
        '3': dict(kind='RETURN', CODE='return 1;'),
        '4': dict(kind='METHOD_RETURN', CODE='int'),
    }
    edges = [('AST','1','2'),('AST','2','3'),('AST','1','4'),('CFG','1','3'),('CFG','3','4')]
    return nodes, edges


class TargetResolutionTests(unittest.TestCase):
    def test_scoped_and_mismatched_names_are_advisory(self):
        self.assertEqual(parse_function('int C::f() { return 1; }','cpp','C::f').name,'f')
        self.assertEqual(target_hint('int C::f() { return 1; }','cpp','wrong').name,'f')
        n,e=fixture()
        g=resolve_target_graph(n,e,filename='input.cpp',source=n['0']['CONTENT'],function_hint='wrong')
        self.assertEqual(g.function,'f')
        self.assertTrue(g.quality['hint_mismatch'])

    def test_nested_function_or_wrong_file_is_not_a_target(self):
        n,e=fixture()
        with self.assertRaises(TargetMethodError):
            resolve_target_graph(n,e,filename='other.cpp',source=n['0']['CONTENT'])
        n['1']['OFFSET']=n['0']['CONTENT'].index('{')
        n['1']['COLUMN_NUMBER']=n['1']['OFFSET']+1
        with self.assertRaises(TargetMethodError):
            resolve_target_graph(n,e,filename='input.cpp',source=n['0']['CONTENT'])

    def test_external_global_and_ambiguous_methods_are_rejected(self):
        for field,value in [('IS_EXTERNAL',True),('NAME','<global>'),('NAME','if')]:
            n,e=fixture();n['1'][field]=value
            with self.assertRaises(TargetMethodError):
                resolve_target_graph(n,e,filename='input.cpp',source=n['0']['CONTENT'])
        n,e=fixture();n['5']=dict(n['1'])
        with self.assertRaises(TargetMethodError):
            resolve_target_graph(n,e,filename='input.cpp',source=n['0']['CONTENT'])

    def test_truncated_method_uses_verified_utf16_offsets(self):
        source='int f() { /* 😀 '+ 'x'*1200 +' */ return 1; }'
        n,e=fixture(source)
        g=resolve_target_graph(n,e,filename='input.cpp',source=source)
        self.assertEqual(g.function,'f')

    def test_unknown_or_missing_cfg_fails_quality(self):
        n,e=fixture();n['3']['kind']='UNKNOWN'
        with self.assertRaises(CPGQualityError):
            resolve_target_graph(n,e,filename='input.cpp',source=n['0']['CONTENT'])
        n,e=fixture()
        with self.assertRaises(CPGQualityError):
            resolve_target_graph(n,[x for x in e if x[0]=='AST'],filename='input.cpp',source=n['0']['CONTENT'])

    def test_container_and_literal_do_not_invent_memory_operations(self):
        for label in ('BLOCK','METHOD','LOCAL','PARAM','LITERAL'):
            self.assertEqual(_node_operations(GraphNode('1',label,'memcpy(dst,src,n); a[i]')),())
        self.assertTrue(_node_operations(GraphNode('1','memcpy','memcpy(dst,src,n)')))

    def test_tokens_keep_literals_and_ignore_comments(self):
        self.assertNotEqual(source_tokens('"a b"'),source_tokens('"ab"'))
        self.assertEqual(source_tokens('x /* c */ + y'),source_tokens('x+y'))

    def test_full_file_allows_real_return_type_prefix_only(self):
        source='static\nint C::f() { return 1; }'
        n,e=fixture(source)
        snippet=source.split('\n',1)[1]
        g=resolve_target_graph(n,e,filename='input.cpp',source=snippet,start_line=2)
        self.assertEqual(g.function,'f')
        n['0']['CONTENT']='void outer() {\n'+snippet
        n['1']['OFFSET_END']=len(n['0']['CONTENT'])
        with self.assertRaises(TargetMethodError):
            resolve_target_graph(n,e,filename='input.cpp',source=snippet,start_line=2)

    def test_source_indentation_and_trailing_newline(self):
        source='  int C::f() { return 1; }\n'
        n,e=fixture(source)
        n['1']['OFFSET']=2
        n['1']['COLUMN_NUMBER']=3
        n['1']['OFFSET_END']=len(source.rstrip())
        self.assertEqual(resolve_target_graph(n,e,filename='input.cpp',source=source).function,'f')

    def test_preprocessor_offset_length_does_not_truncate_target(self):
        source='int f() {\n#define LIMIT 10\nreturn LIMIT;\n}'
        n,e=fixture(source)
        n['1']['OFFSET_END']=12
        n['1']['COLUMN_NUMBER_END']=20
        graph=resolve_target_graph(n,e,filename='input.cpp',source=source)
        self.assertTrue(graph.quality['method_offset_disagreement'])

    def test_unexpanded_function_macro_is_not_a_resolved_definition(self):
        source='HANDLER(handler_name) { return 1; }'
        n,e=fixture(source);n['1']['NAME']='HANDLER'
        with self.assertRaisesRegex(CPGQualityError, 'unresolved_macro_signature'):
            resolve_target_graph(n,e,filename='input.cpp',source=source)
        # A real expansion gives the native method its defined name, while the
        # physical source still points at the original macro invocation.
        n['1']['NAME']='handler_name'
        self.assertEqual(resolve_target_graph(n,e,filename='input.cpp',source=source).function,'handler_name')

    def test_model_semantics_ignore_joern_ids_and_input_order(self):
        from vulnmechanism.semantics import render_semantic_items
        items=[dict(category='CONTROL_CONSTRAINT',kind='CONTROL_CONDITION',detail='node=12 expr=n < 4'),
               dict(category='CONTROL_CONSTRAINT',kind='CONTROL_CONDITION',detail='node=13 expr=p != 0')]
        changed=[dict(r,detail=r['detail'].replace('12','900').replace('13','901')) for r in reversed(items)]
        self.assertEqual(render_semantic_items(items), render_semantic_items(changed))
        self.assertNotIn('node=',render_semantic_items(items))

    def test_streaming_csv_restores_source_literals_and_typed_coordinates(self):
        import csv
        import tempfile
        from pathlib import Path
        from vulnmechanism.cpg import read_neo4jcsv
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source='int f() { const char *s = "\\n\\\\";\nreturn 1; }'
            specs={
                'nodes_FILE': ([':ID',':LABEL','NAME:string','CONTENT:string'],
                               [['0','FILE','input.c',source.replace('\\','\\\\')]]),
                'nodes_METHOD': ([':ID',':LABEL','NAME:string','OFFSET','LINE_NUMBER:int','IS_EXTERNAL:boolean'],
                                 [['1','METHOD','f','0','1','false']]),
                'edges_AST': ([':START_ID',':END_ID',':TYPE'],[['1','2','AST']]),
            }
            for name,(header,rows) in specs.items():
                with (root/(name+'_header.csv')).open('w',newline='') as handle:
                    csv.writer(handle).writerow(header)
                with (root/(name+'_data.csv')).open('w',newline='') as handle:
                    csv.writer(handle).writerows(rows)
            nodes,edges=read_neo4jcsv(root)
            self.assertEqual(nodes['0']['CONTENT'],source)
            self.assertEqual(nodes['1']['OFFSET'],0)
            self.assertIs(nodes['1']['IS_EXTERNAL'],False)
            self.assertEqual(edges,[('AST','1','2')])
