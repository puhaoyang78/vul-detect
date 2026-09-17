import unittest

from vulnmechanism.cpg import (
    CPGError,
    CPGQualityError,
    GraphNode,
    TargetMethodError,
    _isolate_batch,
    resolve_target_graph,
)
from vulnmechanism.semantics import _node_operations, render_mechanism_items
from vulnmechanism.syntax import parse_function, source_tokens, target_hint


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
        self.assertIn('hint_mismatch', g.quality['warnings'])
        self.assertEqual(g.quality['resolution_mode'], 'standalone_file_unique_method')

    def test_wrong_file_and_ambiguous_nested_method_require_hint(self):
        n,e=fixture()
        with self.assertRaises(TargetMethodError):
            resolve_target_graph(n,e,filename='other.cpp',source=n['0']['CONTENT'])
        n,e=fixture()
        n['5'] = dict(kind='METHOD', NAME='inner', FILENAME='input.cpp', FULL_NAME='f.inner:void()',
                      LINE_NUMBER=1, LINE_NUMBER_END=1, IS_EXTERNAL=False, CODE='inner')
        n['6'] = dict(kind='BLOCK', CODE='{ return; }')
        e += [('AST','1','5'),('AST','5','6')]
        with self.assertRaises(TargetMethodError):
            resolve_target_graph(n,e,filename='input.cpp',source=n['0']['CONTENT'])
        g=resolve_target_graph(n,e,filename='input.cpp',source=n['0']['CONTENT'],function_hint='f')
        self.assertEqual(g.function,'f')
        self.assertIn('name_hint', g.quality['resolution_mode'])

    def test_external_global_and_ambiguous_top_level_methods_are_rejected(self):
        for field,value in [('IS_EXTERNAL',True),('NAME','<global>'),('NAME','if')]:
            n,e=fixture(); n['1'][field]=value
            with self.assertRaises(TargetMethodError):
                resolve_target_graph(n,e,filename='input.cpp',source=n['0']['CONTENT'])
        n,e=fixture(); n['5']=dict(n['1']); n['5']['NAME']='g'
        with self.assertRaises(TargetMethodError):
            resolve_target_graph(n,e,filename='input.cpp',source=n['0']['CONTENT'])

    def test_filename_is_primary_for_standalone_function(self):
        source='int f() { return 1; }'
        n,e=fixture(source)
        n['1']['CODE']='int f(){return 1;}'
        g=resolve_target_graph(n,e,filename='input.cpp',source=source,function_hint='wrong')
        self.assertEqual(g.function,'f')
        self.assertTrue(g.quality['source_occurrence_exact'])
        self.assertIn('hint_mismatch', g.quality['warnings'])

    def test_full_file_uses_line_range_then_hint(self):
        snippet='int f() { return 1; }'
        full='static int helper() { return 0; }\n'+snippet
        n,e=fixture(full)
        n['1'].update(NAME='helper', FULL_NAME='helper:int()', LINE_NUMBER=1, LINE_NUMBER_END=1,
                      CODE='static int helper() { return 0; }')
        n['2']['CODE']='{ return 0; }'
        n['5']=dict(kind='METHOD', NAME='f', FILENAME='input.cpp', FULL_NAME='f:int()',
                    LINE_NUMBER=2, LINE_NUMBER_END=2, IS_EXTERNAL=False, CODE=snippet)
        n['6']=dict(kind='BLOCK', CODE='{ return 1; }')
        n['7']=dict(kind='RETURN', CODE='return 1;')
        n['8']=dict(kind='METHOD_RETURN', CODE='int')
        e += [('AST','5','6'),('AST','6','7'),('AST','5','8'),('CFG','5','7'),('CFG','7','8')]
        g=resolve_target_graph(n,e,filename='input.cpp',source=snippet,start_line=2,function_hint='f')
        self.assertEqual(g.function,'f')
        self.assertTrue(g.quality['resolution_mode'].startswith('full_file_line_range'))

    def test_unknown_or_missing_cfg_fails_quality(self):
        n,e=fixture(); n['3']['kind']='UNKNOWN'
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

    def test_source_indentation_and_trailing_newline(self):
        source='  int C::f() { return 1; }\n'
        n,e=fixture(source)
        self.assertEqual(resolve_target_graph(n,e,filename='input.cpp',source=source).function,'f')

    def test_unexpanded_function_macro_is_not_a_resolved_definition(self):
        source='HANDLER(handler_name) { return 1; }'
        n,e=fixture(source); n['1']['NAME']='HANDLER'
        with self.assertRaisesRegex(CPGQualityError, 'unresolved_macro_signature'):
            resolve_target_graph(n,e,filename='input.cpp',source=source)
        n['1']['NAME']='handler_name'
        self.assertEqual(resolve_target_graph(n,e,filename='input.cpp',source=source).function,'handler_name')

    def test_batch_level_failure_is_isolated(self):
        calls=[]
        def runner(requests):
            calls.append([r['id'] for r in requests])
            if any(r['id']==2 for r in requests):
                if len(requests)==1:
                    raise CPGError('bad sample')
                raise CPGError('batch failed')
            return [f'ok-{r["id"]}' for r in requests]
        result=_isolate_batch([{'id':1},{'id':2},{'id':3},{'id':4}], runner)
        self.assertEqual(result[0], 'ok-1')
        self.assertIsInstance(result[1], CPGError)
        self.assertEqual(result[2:], ['ok-3','ok-4'])
        self.assertIn([2], calls)

    def test_mechanism_renderer_is_deterministic(self):
        items=[
            dict(category='MECHANISM_CANDIDATE',kind='BOUNDS_FLOW',detail='source=n'),
            dict(category='SECURITY_OPERATION',kind='ARRAY_ACCESS',detail='object=a'),
        ]
        changed=list(reversed(items))
        self.assertEqual(render_mechanism_items(items), render_mechanism_items(changed))

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


if __name__ == '__main__':
    unittest.main()
