"""CGVL admission and operation-bound readout. Verdicts are audit-only.

The first local proof subset deliberately does not infer function vulnerability.
A verdict describes the named property if the operation is executed, under C
primitive-object and standard malloc/free contracts; unknown contexts stay unknown.
"""
from __future__ import annotations

from collections import Counter
from bisect import bisect_left, bisect_right
from pathlib import Path
import json
import re

import torch
from torch import nn

from .syntax import parser_for, walk, identifier
from .semantics import _static_integer
from .cfg_data import read_records, atomic_json, cohort_hash

FAMILIES = ('Bounds', 'Pointer', 'Lifetime')
STATES = ('Satisfied', 'Violated', 'Unknown')


def _span(node):
    return [node.start_byte, node.end_byte]


def local_operations(source, language):
    """Source-only candidates plus separate conservative audit verdicts.

    Bounds: one-dimensional primitive local arrays and literal indices.
    Pointer: direct *p store after an explicit local address/null definition.
    Lifetime: direct free of a locally allocated standard-malloc object; duplicate
    free is violated only with established non-nullness (free(NULL) is allowed).
    Unsupported statements invalidate pointer state, never establish safety.
    """
    raw = source.encode('utf-8'); root = parser_for(language).parse(raw).root_node
    nodes = list(walk(root)); functions = [n for n in nodes if n.type == 'function_definition']
    operations = []
    by_span = {}
    def add(n, family, obj):
        row = dict(family=family, operation=_span(n), object=obj,
                   context=[], state='Unknown', reason='unsupported_local_context', edits=[])
        operations.append(row); by_span[tuple(_span(n))] = row
    for n in nodes:
        if n.type == 'subscript_expression':
            arg=n.child_by_field_name('argument')
            add(n,'Bounds',arg.text.decode() if arg is not None and arg.type=='identifier' else None)
        elif n.type == 'pointer_expression' and n.text.lstrip().startswith(b'*'):
            arg=n.child_by_field_name('argument')
            add(n,'Pointer',arg.text.decode() if arg is not None and arg.type=='identifier' else None)
        elif n.type == 'field_expression' and b'->' in n.text:
            arg=n.child_by_field_name('argument')
            add(n,'Pointer',arg.text.decode() if arg is not None and arg.type=='identifier' else None)
        elif n.type == 'call_expression':
            fn=n.child_by_field_name('function'); args=n.child_by_field_name('arguments')
            if fn is not None and fn.text==b'free':
                children=args.named_children if args else []
                add(n,'Lifetime',children[0].text.decode() if len(children)==1 and children[0].type=='identifier' else None)
    if root.has_error or len(functions)!=1 or any(n.type.startswith('preproc_') for n in nodes):
        for row in operations: row['reason']='parse_or_preprocessor_unknown'
        return operations
    # Names with more than one declaration are excluded rather than guessed across scopes.
    declarations={}; duplicates=set()
    for n in nodes:
        if n.type not in ('declaration','parameter_declaration'): continue
        for d in n.children_by_field_name('declarator'):
            name=identifier(d,raw)
            if name in declarations: duplicates.add(name)
            declarations[name]=(n,d)
    # Candidate context is bound without consulting a verdict. Unknown operations
    # keep the same source-only feature path; no proof status enters the model.
    node_lookup={tuple(_span(n)):n for n in nodes}
    for row in operations:
        name=row['object']
        if name in declarations and name not in duplicates:
            decl,_=declarations[name]
            scope=decl.parent
            if decl.end_byte<=row['operation'][0] and (decl.type=='parameter_declaration' or
                    scope.type=='compound_statement' and row['operation'][1]<=scope.end_byte):
                row['context'].append(_span(decl))
        parent=node_lookup[tuple(row['operation'])].parent
        while parent is not None:
            if parent.type=='if_statement':
                cond=parent.child_by_field_name('condition')
                if cond is not None and cond.end_byte<=row['operation'][0]:row['context'].append(_span(cond))
            parent=parent.parent
    def publish(node,state,reason,context,edits=()):
        row=by_span.get(tuple(_span(node)))
        if row is not None:
            row.update(state=state,reason=reason,context=context,edits=list(edits))
    # Bounds is an object extent property, not a claim about initialized contents.
    for n in nodes:
        row=by_span.get(tuple(_span(n)))
        if not row or row['family']!='Bounds' or row['object'] not in declarations: continue
        name=row['object']; decl,d=declarations[name]
        if name in duplicates or decl.type!='declaration' or decl.start_byte>=n.start_byte: continue
        if d.type=='init_declarator': d=d.child_by_field_name('declarator')
        typ=decl.child_by_field_name('type')
        if d.type!='array_declarator' or d.child_by_field_name('declarator').type!='identifier' or typ.type!='primitive_type': continue
        scope=decl.parent
        if scope.type!='compound_statement' or not scope.start_byte<=n.start_byte<n.end_byte<=scope.end_byte: continue
        idx=n.child_by_field_name('index'); size=d.child_by_field_name('size')
        capacity=_static_integer(size.text.decode()) if size is not None else None
        index=_static_integer(idx.text.decode()) if idx is not None else None
        if capacity is None or index is None or not 1<=capacity<=32767: continue
        # Only assignment stores with literal RHS: excludes sizeof, &a[N], macro,
        # overloaded/compound expressions and unsequenced modifications.
        parent=n.parent
        if parent.type!='assignment_expression' or parent.child_by_field_name('left')!=n or parent.child_by_field_name('right').type not in ('number_literal','char_literal'): continue
        if parent.parent.type!='expression_statement': continue
        state='Satisfied' if index<capacity else 'Violated'
        edits=[dict(kind='unsafe',span=_span(idx),text=str(capacity)),
               dict(kind='safe_change',span=_span(idx),text=str((index+1)%capacity))] if state=='Satisfied' else []
        publish(n,state,'literal_index_vs_local_extent',[_span(decl)],edits)
    # State is restricted to an uninterrupted structured local path. Calls and
    # unsupported statements invalidate all pointers; no alias assumptions.
    def block(body,env):
        env={k:dict(v) for k,v in env.items()}
        for stmt in body.named_children:
            if stmt.type=='comment': continue
            if stmt.type=='declaration':
                ds=stmt.children_by_field_name('declarator');typ=stmt.child_by_field_name('type')
                if len(ds)!=1 or typ is None or typ.type!='primitive_type': env.clear();continue
                d=ds[0];name=identifier(d,raw)
                if name in duplicates: env.clear();continue
                value=d.child_by_field_name('value') if d.type=='init_declarator' else None
                actual=d.child_by_field_name('declarator') if value is not None else d
                if actual.type=='identifier':
                    env[name]=dict(kind='scalar',context=[_span(stmt)])
                elif actual.type=='pointer_declarator' and actual.child_by_field_name('declarator').type=='identifier':
                    state=None
                    if value is not None and value.type=='pointer_expression' and value.text.startswith(b'&'):
                        arg=value.child_by_field_name('argument'); obj=env.get(arg.text.decode(),{})
                        if obj.get('kind')=='scalar': state=dict(kind='address',context=obj['context']+[_span(stmt)])
                    if value is not None and value.text==b'0':state=dict(kind='null',context=[_span(stmt)])
                    if value is not None and value.type=='call_expression' and 'malloc' not in declarations and 'free' not in declarations:
                        fn=value.child_by_field_name('function');args=value.child_by_field_name('arguments').named_children
                        if fn.text==b'malloc' and len(args)==1 and (_static_integer(args[0].text.decode()) or 0)>0:
                            state=dict(kind='heap',nonnull=False,freed=False,context=[_span(stmt)])
                    env.pop(name,None)
                    if state:env[name]=dict(state,initializer=_span(value))
                else: env.clear()
                continue
            if stmt.type=='if_statement':
                cond=stmt.child_by_field_name('condition'); cs=cond.named_children
                var=cs[0].text.decode() if len(cs)==1 and cs[0].type=='identifier' else None
                then=stmt.child_by_field_name('consequence')
                if var in env and env[var]['kind']=='heap' and then.type=='compound_statement':
                    branch={k:dict(v) for k,v in env.items()};branch[var].update(nonnull=True,context=env[var]['context']+[_span(cond)])
                    block(then,branch)
                env.clear();continue
            if stmt.type=='expression_statement' and len(stmt.named_children)==1:
                expr=stmt.named_children[0]
                if expr.type=='assignment_expression':
                    lhs=expr.child_by_field_name('left');rhs=expr.child_by_field_name('right')
                    row=by_span.get(tuple(_span(lhs)))
                    if row and row['family']=='Pointer' and rhs.type=='number_literal' and b'=' in expr.text:
                        obj=env.get(row['object'],{})
                        if obj.get('kind') in ('address','null'):
                            state='Satisfied' if obj['kind']=='address' else 'Violated'
                            edits=[dict(kind='unsafe',span=obj['initializer'],text='0')] if state=='Satisfied' else []
                            publish(lhs,state,'local_address_or_null_store',obj['context'],edits)
                    env.clear();continue
                row=by_span.get(tuple(_span(expr)))
                if row and row['family']=='Lifetime':
                    name=row['object'];obj=env.get(name,{})
                    if obj.get('kind')=='heap':
                        state=('Violated' if obj['nonnull'] else 'Unknown') if obj['freed'] else 'Satisfied'
                        edits=[dict(kind='unsafe',span=[stmt.start_byte,stmt.start_byte],text=f'free({name}); ')] if state=='Satisfied' and obj['nonnull'] else []
                        publish(expr,state,'standard_heap_free_nonnull' if obj['nonnull'] else 'nullable_heap_free',obj['context'],edits)
                        env[name]=dict(obj,freed=True,context=obj['context']+[_span(stmt)])
                        continue
            env.clear()
    block(functions[0].child_by_field_name('body'),{})
    # Native pointer parameters under an immediate, side-effect-free null guard.
    # This proves only nullness at dereference, not pointee extent or lifetime.
    for row in operations:
        if row['family']!='Pointer' or row['state']!='Unknown':continue
        name=row['object']
        if name not in declarations or name in duplicates:continue
        decl,d=declarations[name]
        if decl.type!='parameter_declaration' or d.type!='pointer_declarator':continue
        if any(n.type=='type_qualifier' and n.text==b'volatile' for n in walk(decl)):continue
        node=node_lookup[tuple(row['operation'])];statement=node
        while statement.parent is not None and statement.type not in ('return_statement','expression_statement'):
            statement=statement.parent
        # The proof currently admits a directly evaluated return operand only;
        # unevaluated language extensions are not inferred from token spelling.
        if statement.type!='return_statement' or statement.named_children!=[node]:continue
        body=statement.parent
        if body.type!='compound_statement' or body.parent is None or body.parent.type!='if_statement':continue
        if [n for n in body.named_children if n.type!='comment'][0]!=statement:continue
        parent=body.parent
        if parent.child_by_field_name('consequence')!=body:continue
        # No calls, writes, increments or alternative expression evaluation can
        # change the tested parameter before this dereference.
        if any(n.type in ('call_expression','assignment_expression','update_expression',
                          'conditional_expression','sizeof_expression') for n in walk(statement)):continue
        if any(n.type=='pointer_expression' and n.text.lstrip().startswith(b'&') or
               n.type=='binary_expression' and any(c.type in ('&&','||') for c in n.children)
               for n in walk(statement)):continue
        cond=parent.child_by_field_name('condition')
        if len(cond.named_children)!=1:continue
        expr=cond.named_children[0]; text=expr.text.decode().replace(' ','')
        nonnull = text in (name, name+'!=0', '0!='+name)
        null = text in ('!'+name, name+'==0', '0=='+name)
        if not (nonnull or null):continue
        context=list(row['context'])
        edits=[dict(kind='unsafe',span=_span(expr),text='!'+name),
               dict(kind='safe_change',span=_span(expr),text=name+' != 0')] if nonnull else []
        publish(node,'Satisfied' if nonnull else 'Violated','guarded_parameter_nullness',context,edits)
    return operations


def verified_pairs(source, language, operations):
    """Reparse and re-prove every edit; no mutated function label is emitted."""
    raw=source.encode(); pairs=[]
    for i,op in enumerate(operations):
        if op['state']!='Satisfied':continue
        for edit in op['edits']:
            a,b=edit['span']; replacement=edit['text'].encode()
            if raw[a:b]==replacement:continue
            changed=(raw[:a]+replacement+raw[b:]).decode()
            delta=len(replacement)-(b-a)
            # An edit inside the operation changes its end; an insertion before
            # it shifts both endpoints, including an inserted earlier free.
            start,end=op['operation'];start+=delta if a<=start else 0;end+=delta if a<end else 0
            match=[x for x in local_operations(changed,language) if x['family']==op['family'] and x['operation']==[start,end]]
            expected='Violated' if edit['kind']=='unsafe' else 'Satisfied'
            if len(match)!=1 or match[0]['state']!=expected:continue
            pairs.append(dict(operation_index=i,family=op['family'],kind=edit['kind'],edit=edit,
                              original_state='Satisfied',changed_state=expected,changed_source=changed,
                              changed_operation=match[0],function_label=None))
    root=parser_for(language).parse(raw).root_node
    nodes=list(walk(root))
    for i,op in enumerate(operations):
        if op['state']!='Satisfied':continue
        controls=[('unrelated_comment', source+' /* cgvl neutral comment */')]
        name=op['object'];renamed='cgvl_local_object'
        if name and not re.search(r'\b'+renamed+r'\b',source):
            changes=[n for n in nodes if n.type=='identifier' and n.text.decode()==name]
            # Unexpanded calls may be macros which stringify or capture names.
            in_call=False
            for n in changes:
                ancestor=n.parent
                while ancestor is not None:
                    if ancestor.type=='call_expression' or 'asm' in ancestor.type:in_call=True
                    ancestor=ancestor.parent
            if in_call:changes=[]
            mutated=raw
            for n in sorted(changes,key=lambda x:x.start_byte,reverse=True):
                mutated=mutated[:n.start_byte]+renamed.encode()+mutated[n.end_byte:]
            if changes:controls.append(('rename',mutated.decode()))
        for kind,changed in controls:
            checked=local_operations(changed,language)
            if len(checked)!=len(operations) or checked[i]['family']!=op['family'] or checked[i]['state']!='Satisfied':continue
            pairs.append(dict(operation_index=i,family=op['family'],kind=kind,
                              original_state='Satisfied',changed_state='Satisfied',changed_source=changed,
                              changed_operation=checked[i],function_label=None))
    return pairs


def model_spans(source,operations,tokenizer,prefix_tokens=0,max_length=2048):
    """Whitelist source positions; never expose verdict/reason/edit/repair labels."""
    offsets=tokenizer(source,add_special_tokens=False,truncation=True,max_length=max_length,
                      return_offsets_mapping=True)['offset_mapping']
    raw=source.encode()
    starts=[x for x,y in offsets];ends=[y for x,y in offsets]
    def tokens(span):
        a,b=span;a=len(raw[:a].decode());b=len(raw[:b].decode())
        ids=list(range(bisect_right(ends,a)+prefix_tokens,bisect_left(starts,b)+prefix_tokens))
        return ids if ids and offsets[ids[-1]-prefix_tokens][1]>=b else []
    result=[]
    for op in operations:
        ids=tokens(op['operation']); context=[tokens(s) for s in op['context']]
        if ids:result.append(dict(family=FAMILIES.index(op['family']),operation=ids,
                                 context=[x for x in context if x]))
    return result


class OperationConstraintAlignment(nn.Module):
    """Shared operation/context interaction; risk-conditioned token readout.

    The returned risk is local-ranking only. Function logits must be computed
    by the ordinary classifier on the returned source vector, not added scores.
    """
    def __init__(self,hidden_size,rank=32):
        super().__init__()
        self.operation=nn.Linear(hidden_size,rank)
        self.context=nn.Linear(hidden_size,rank)
        self.interaction=nn.Sequential(nn.Linear(3*rank,rank),nn.Tanh())
        self.risk=nn.Linear(rank,1)
        self.tokens=nn.Linear(hidden_size,rank,bias=False)

    def forward(self,hidden,mask,items,*,shuffled=False):
        if hidden.ndim!=2 or mask.shape!=hidden.shape[:1] or not mask.bool().any():
            raise ValueError('one nonempty function required')
        vectors=[]
        contexts=[item['context'] for item in items]
        if shuffled and len(items)>1:contexts=contexts[1:]+contexts[:1]
        for item,spans in zip(items,contexts):
            indices=item['operation']+[i for s in spans for i in s]
            if not item['operation'] or any(type(i)!=int or not 0<=i<len(hidden) or not bool(mask[i]) for i in indices):
                raise ValueError('operation/context must reference visible source tokens')
            op=hidden[item['operation']].mean(0)
            ctx=hidden[[i for s in spans for i in s]].mean(0) if spans else hidden.new_zeros(hidden.shape[-1])
            a=self.operation(op);b=self.context(ctx)
            vectors.append(self.interaction(torch.cat((a,b,a*b))))
        if not vectors:return hidden[mask.bool()].mean(0),hidden.new_empty(0)
        local=torch.stack(vectors);risk=self.risk(local).squeeze(-1)
        condition=(risk.softmax(0)[:,None]*local).sum(0)
        weights=(self.tokens(hidden)@condition/(condition.numel()**.5)).masked_fill(~mask.bool(),-torch.inf).softmax(0)
        return (hidden*weights[:,None]).sum(0),risk


def local_rank_loss(safe,unsafe):
    if safe.shape!=unsafe.shape or not safe.numel():raise ValueError('matched nonempty local pairs required')
    return torch.nn.functional.softplus(safe-unsafe).mean()


def prepare_cgvl(dataset,reference_run_dir,output_dir):
    from .cfg_experiment import _base_module,_tokenizer
    root=Path(output_dir)
    if root.exists():raise FileExistsError(root)
    config=json.loads((Path(reference_run_dir)/'config.json').read_text())
    builder=_tokenizer(_base_module(),config)
    rows=read_records(dataset,config['source_dataset'])
    if cohort_hash(rows)!=config['cohort_sha256']:
        raise ValueError('CGVL must use the unchanged classification cohort')
    rows=[r for r in rows if r['split'] in ('train','valid')]
    root.mkdir(parents=True)
    counts={s:{f:Counter() for f in FAMILIES} for s in ('train','valid')}
    reasons={s:{f:Counter() for f in FAMILIES} for s in ('train','valid')}
    functions={s:{f:set() for f in FAMILIES} for s in ('train','valid')}
    with (root/'operations.jsonl').open('x') as out,(root/'pairs.jsonl').open('x') as pout:
        for row in rows:
            source=row['raw_source'];lang=row.get('resolved_language') or row.get('language')
            ops=local_operations(source,lang);pairs=verified_pairs(source,lang,ops)
            for op in ops:
                counts[row['split']][op['family']][op['state']]+=1
                if op['state']=='Unknown':reasons[row['split']][op['family']][op['reason']]+=1
            features=model_spans(source,ops,builder.tokenizer,len(builder.source_prefix))
            accepted=[]
            for pair in pairs:
                original=model_spans(source,[ops[pair['operation_index']]],builder.tokenizer,len(builder.source_prefix))
                changed=model_spans(pair['changed_source'],[pair['changed_operation']],builder.tokenizer,len(builder.source_prefix))
                if not original or not changed or not original[0]['context'] or not changed[0]['context']:
                    counts[row['split']][pair['family']]['pair_outside_window']+=1;continue
                pair['visible_input_changed']=(builder.source_ids(row)!=
                    builder.source_ids(dict(row,raw_source=pair['changed_source'])))
                counts[row['split']][pair['family']][pair['kind']+'_pairs']+=1
                if pair['visible_input_changed']:
                    counts[row['split']][pair['family']]['visible_'+pair['kind']+'_pairs']+=1
                if pair['kind']=='unsafe':functions[row['split']][pair['family']].add(row['sample_key'])
                accepted.append(pair)
            out.write(json.dumps(dict(sample_key=row['sample_key'],split=row['split'],operations=ops,model_spans=features))+'\n')
            for pair in accepted:pout.write(json.dumps(dict(sample_key=row['sample_key'],split=row['split'],**pair))+'\n')
    for split in counts:
        for f in FAMILIES:
            for state in STATES:counts[split][f].setdefault(state,0)
            counts[split][f]['independent_rank_functions']=len(functions[split][f])
    # Minimum feasibility, not a statistical-power claim. A three-family result
    # cannot be evaluated with almost no held-out examples in one family.
    ready=all(len(functions['train'][f])>=20 and len(functions['valid'][f])>=5 for f in FAMILIES)
    report=dict(dataset=str(Path(dataset).resolve()),reference_run_dir=str(Path(reference_run_dir).resolve()),
                seed=config['seed'],model_path=config['model_path'],source_max_length=config['source_max_length'],
                counts=counts,unknown_reasons=reasons,functions=Counter(r['split'] for r in rows),test_used=False,
                ready=ready,minimum_independent_functions=dict(train_per_family=20,valid_per_family=5),
                note='local property conditional on execution; never a function safety label')
    atomic_json(root/'coverage.json',report)
    return report


def construction_sites(source, language):
    """Small entry-reachable proof subset for real-source transformations.

    Pointer proves null-dereference exclusion only, with other pointer validity
    properties explicitly outside scope. Bounds proves local primitive-array
    writes in an entry loop. No callee summary or patch label supplies truth.
    """
    raw = source.encode()
    root = parser_for(language).parse(raw).root_node
    nodes = list(walk(root))
    fs = [n for n in nodes if n.type == 'function_definition']
    if language != 'c' or root.has_error or len(fs) != 1 or any(
            n.type.startswith('preproc_') for n in nodes):
        return []
    f = fs[0]
    body = f.child_by_field_name('body')
    statements = [n for n in body.named_children if n.type != 'comment']
    declarations = {}
    for n in nodes:
        if n.type in ('declaration', 'parameter_declaration'):
            for d in n.children_by_field_name('declarator'):
                name = identifier(d, raw)
                declarations.setdefault(name, []).append((n, d))
    sites = []
    def first_statement(node):
        if node.type != 'compound_statement':
            return node
        children = [n for n in node.named_children if n.type != 'comment']
        return children[0] if children else None
    def null_test(node):
        text = re.sub(r'\s+', '', node.text.decode()).strip('()')
        for name, defs in declarations.items():
            if not name or len(defs) != 1:
                continue
            decl, d = defs[0]
            if decl.type != 'parameter_declaration' or d.type != 'pointer_declarator':
                continue
            if any(n.type == 'type_qualifier' for n in walk(decl)):
                continue
            if text in (name, '!!'+name, name+'!=0', '0!='+name, name+'!=NULL', 'NULL!='+name):
                return name, True, decl
            if text in ('!'+name, name+'==0', '0=='+name, name+'==NULL', 'NULL=='+name):
                return name, False, decl
        return None
    if statements and statements[0].type == 'if_statement':
        guard = statements[0]
        cond = guard.child_by_field_name('condition')
        test = null_test(cond)
        if test and guard.child_by_field_name('alternative') is None:
            name, positive, decl = test
            branch = first_statement(guard.child_by_field_name('consequence'))
            target_statement = branch
            taken = True
            # An unconditional literal return makes the following statement
            # reachable on exactly the complementary guard condition.
            if branch is not None and branch.type == 'return_statement' and all(
                    n.type in ('number_literal', 'null') or n.text == b'NULL'
                    for n in branch.named_children):
                target_statement = statements[1] if len(statements) > 1 else None
                taken = False
            if target_statement is not None:
                expr = None
                if target_statement.type == 'return_statement' and len(target_statement.named_children) == 1:
                    expr = target_statement.named_children[0]
                if target_statement.type == 'expression_statement' and len(target_statement.named_children) == 1:
                    assignment = target_statement.named_children[0]
                    if assignment.type == 'assignment_expression':
                        right = assignment.child_by_field_name('right')
                        operator = assignment.child_by_field_name('operator')
                        if right.type == 'number_literal' and operator.text == b'=':
                            expr = assignment.child_by_field_name('left')
                arg = expr.child_by_field_name('argument') if expr is not None else None
                direct = expr is not None and (expr.type == 'field_expression' and b'->' in expr.text
                         or expr.type == 'pointer_expression' and expr.text.startswith(b'*'))
                if direct and arg is not None and arg.type == 'identifier' and arg.text.decode() == name:
                    nonnull = positive == taken
                    sites.append(dict(family='Pointer', operation=_span(expr), object=name,
                        context=[_span(decl), _span(cond)], edit_span=_span(cond),
                        statement_start=target_statement.start_byte,
                        state='Safe' if nonnull else 'Unsafe', template='entry_guard' if taken else 'entry_early_return',
                        proof='Guard fixes nullness at the immediately evaluated dereference; no preceding statements.',
                        witness={name:'NULL'} if not nonnull else {name:'address of a live, initialized object of the declared pointee type'},
                        scope='Null-dereference property only; bounds, dangling pointers and object invariants are not inferred.',
                        unsafe_text='('+name+(' == 0)' if taken else ' != 0)'),
                        safe_text='('+name+(' != 0)' if taken else ' == 0)')))
    # Entry prefix may contain uninitialized primitive declarations, never calls,
    # pointer aliases, initialized expressions, branches or other executed effects.
    arrays = {}
    integers = set()
    for stmt in statements:
        if stmt.type == 'declaration':
            typ = stmt.child_by_field_name('type')
            if typ is None or typ.type != 'primitive_type' or any(n.type == 'type_qualifier' for n in walk(stmt)):
                break
            valid = True
            for d in stmt.children_by_field_name('declarator'):
                name = identifier(d, raw)
                if len(declarations.get(name, [])) != 1:
                    valid = False; break
                if d.type == 'identifier' and typ.text == b'int':
                    integers.add(name)
                elif d.type == 'array_declarator' and d.child_by_field_name('declarator').type == 'identifier' and typ.text != b'void':
                    size = d.child_by_field_name('size')
                    cap = _static_integer(size.text.decode()) if size is not None else None
                    if cap is None or not 1 <= cap <= 32760:
                        valid = False; break
                    arrays[name] = (cap, size, stmt)
                else:
                    valid = False; break
            if not valid:
                break
            continue
        if stmt.type != 'for_statement':
            break
        init = stmt.child_by_field_name('initializer')
        cond = stmt.child_by_field_name('condition')
        update = stmt.child_by_field_name('update')
        if init is None or cond is None or update is None:
            break
        if init.type == 'declaration':
            ds = init.children_by_field_name('declarator')
            typ = init.child_by_field_name('type')
            if typ is None or typ.text != b'int' or len(ds) != 1 or ds[0].type != 'init_declarator':
                break
            value = ds[0].child_by_field_name('value')
            if (value is None or value.text != b'0' or
                    ds[0].child_by_field_name('declarator').type != 'identifier' or
                    any(n.type == 'type_qualifier' for n in walk(init))):
                break
            index = identifier(ds[0], raw)
            if len(declarations.get(index, [])) != 1:
                break
        elif init.type == 'assignment_expression':
            left = init.child_by_field_name('left')
            if left.type != 'identifier' or init.child_by_field_name('operator').text != b'=' or init.child_by_field_name('right').text != b'0':
                break
            index = left.text.decode()
            if index not in integers:
                break
        else:
            break
        if cond.type != 'binary_expression' or cond.child_by_field_name('left').text.decode() != index or cond.child_by_field_name('operator').text != b'<':
            break
        bound_node = cond.child_by_field_name('right')
        bound = _static_integer(bound_node.text.decode())
        if bound is None or not 1 <= bound <= 32760 or update.text.decode().replace(' ', '') not in (index+'++', '++'+index):
            break
        loop_body = stmt.child_by_field_name('body')
        body_items = [n for n in loop_body.named_children if n.type != 'comment'] if loop_body.type == 'compound_statement' else [loop_body]
        if len(body_items) != 1 or body_items[0].type != 'expression_statement':
            break
        expr = body_items[0].named_children[0]
        if expr.type != 'assignment_expression' or expr.child_by_field_name('operator').text != b'=' or expr.child_by_field_name('right').text not in (b'0', b'1'):
            break
        lhs = expr.child_by_field_name('left')
        if lhs.type != 'subscript_expression' or lhs.child_by_field_name('index').text.decode() != index:
            break
        obj = lhs.child_by_field_name('argument').text.decode()
        if obj not in arrays:
            break
        capacity, size, decl = arrays[obj]
        sites.append(dict(family='Bounds', operation=_span(lhs), object=obj, context=[_span(decl),_span(stmt)],
            edit_span=_span(bound_node), capacity_span=_span(size), capacity=capacity, bound=bound,
            statement_start=body_items[0].start_byte, state='Safe' if bound <= capacity else 'Unsafe',
            template='entry_counted_array_loop', proof='int induction range [0, bound-1], no body effects except the literal store; fixed local extent.',
            witness=dict(iteration=capacity) if bound > capacity else dict(index_range=[0,bound-1]),
            scope='Array extent of this store; not function vulnerability or initialization of other objects.'))
        break
    return sites


def constructed_variants(source, language):
    """Return re-proved variants, preserving every byte outside explicit edits."""
    result = []
    for site in construction_sites(source, language):
        edits = []
        if site['family'] == 'Pointer':
            edits = [('safe', [(site['edit_span'], site['safe_text'])]),
                     ('unsafe', [(site['edit_span'], site['unsafe_text'])])]
        else:
            n = site['capacity']
            # Cross object extent with loop bound: neither literal alone is truth.
            edits = [('safe', [(site['edit_span'], str(n))]),
                     ('unsafe', [(site['edit_span'], str(n+1))]),
                     ('safe_change', [(site['capacity_span'], str(n+1)), (site['edit_span'],str(n+1))]),
                     ('unsafe_cross', [(site['capacity_span'], str(n+1)), (site['edit_span'],str(n+2))])]
        safe_source = None
        for kind, changes in edits:
            raw = source.encode()
            for (a,b), text in sorted(changes, reverse=True):
                raw = raw[:a]+text.encode()+raw[b:]
            changed = raw.decode()
            checked = [s for s in construction_sites(changed, language) if s['family']==site['family'] and s['object']==site['object']]
            expected = 'Unsafe' if kind.startswith('unsafe') else 'Safe'
            if len(checked) != 1 or checked[0]['state'] != expected:
                raise ValueError('construction failed re-analysis')
            result.append(dict(kind=kind, original_operation=site['operation'], source=changed,
                edits=[dict(span=span,text=text) for span,text in changes], verified=checked[0]))
            if kind == 'safe':
                safe_source = changed
        # A visible comment at a statement boundary is semantics preserving; no
        # trailing comment outside the model window is counted as a control.
        checked = [s for s in construction_sites(safe_source,language) if s['family']==site['family'] and s['object']==site['object']][0]
        position = checked['statement_start']
        raw = safe_source.encode()
        changed = (raw[:position]+b'/* local control */ '+raw[position:]).decode()
        proof = [s for s in construction_sites(changed,language) if s['family']==site['family'] and s['object']==site['object']]
        if len(proof) != 1 or proof[0]['state'] != 'Safe':
            raise ValueError('semantic control failed re-analysis')
        result.append(dict(kind='semantic_control', original_operation=site['operation'],source=changed,
            edits=[dict(span=[position,position],text='/* local control */ ')],edit_base='safe',verified=proof[0]))
    return result


def prepare_constructed(dataset, reference_run_dir, output_dir, primevul_dir):
    """CPU-only inventory, entry-path proofs, and immutable derived examples."""
    from .cfg_experiment import _base_module, _tokenizer
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(root)
    config = json.loads((Path(reference_run_dir)/'config.json').read_text())
    if config['source_dataset'] != 'primevul' or config['source_max_length'] != 2048:
        raise ValueError('construction requires the unchanged PrimeVul 2048-token protocol')
    rows = read_records(dataset, 'primevul')
    if cohort_hash(rows) != config['cohort_sha256']:
        raise ValueError('classification cohort changed')
    rows = [r for r in rows if r['split'] in ('train','valid')]
    metadata = {}
    wanted = {int(r['sample_key'].split(':')[-1]) for r in rows}
    # Explicit train/valid filenames; no glob capable of opening test sources.
    for split in ('train','valid'):
        with (Path(primevul_dir)/('primevul_'+split+'.jsonl')).open() as stream:
            for line in stream:
                r = json.loads(line)
                if r['idx'] in wanted:
                    metadata[r['idx']] = dict(project=r['project'],commit=r['commit_id'],
                        function_source=r['func'],native_split=split,commit_url=r.get('commit_url'),
                        file_name=r.get('file_name'),cve=r.get('cve'))
    group_splits = {}
    source_splits = {}
    for r in rows:
        m = metadata[int(r['sample_key'].split(':')[-1])]
        if m['function_source'].strip() != r['raw_source'].strip() or m['native_split'] != r['split']:
            raise ValueError('native source or split mismatch')
        group = (m['project'],m['commit'],m['file_name'],r['function_name'])
        m['group'] = group
        group_splits.setdefault(group,set()).add(r['split'])
        source_splits.setdefault(r['raw_source'].strip(),set()).add(r['split'])
    builder = _tokenizer(_base_module(),config)
    counts = {s:{f:Counter() for f in FAMILIES} for s in ('train','valid')}
    accepted = {s:{f:set() for f in FAMILIES} for s in ('train','valid')}
    projects = {s:set() for s in ('train','valid')}
    templates = {s:Counter() for s in ('train','valid')}
    seen = {s:set() for s in ('train','valid')}
    records = []
    for r in rows:
        source = r['raw_source'];split = r['split'];language=r.get('resolved_language') or r['language']
        operations = local_operations(source,language)
        families = {op['family'] for op in operations}
        for family in families:
            counts[split][family]['candidate_functions'] += 1
        for op in operations:
            counts[split][op['family']]['candidate_operations'] += 1
        m = metadata[int(r['sample_key'].split(':')[-1])]
        reason = None
        if len(group_splits[m['group']]) != 1 or len(source_splits[source.strip()]) != 1:
            reason = 'source_or_repair_group_cross_split'
        elif source.strip() in seen[split]:
            reason = 'duplicate_source_in_split'
        seen[split].add(source.strip())
        variants = [] if reason else constructed_variants(source,language)
        native_sites = construction_sites(source,language) if variants else []
        for family in families:
            c = counts[split][family]
            selected = [v for v in variants if v['verified']['family']==family]
            if not selected:
                parse_failed = any(op['reason']=='parse_or_preprocessor_unknown' for op in operations if op['family']==family)
                c['Unknown_functions'] += 1
                c['reason:'+ (reason or ('parse_or_preprocessor_unknown' if parse_failed else
                    'cpp_semantics_not_certified' if language!='c' else
                    'no_entry_reachable_owned_heap_proof' if family=='Lifetime' else
                    'outside_entry_guard_or_counted_loop_subset'))] += 1
                continue
            proofs = [x for x in native_sites if x['family']==family]
            visible = True
            for text,sites in [(source,proofs)]+[(v['source'],[v['verified']]) for v in selected]:
                offsets = builder.tokenizer(text,add_special_tokens=False,truncation=True,max_length=2048,
                                            return_offsets_mapping=True)['offset_mapping']
                end = len(text[:offsets[-1][1]].encode()) if offsets else 0
                if any(span[1]>end for site in sites for span in [site['operation'],*site['context']]):
                    visible = False
            c['proved_before_visibility'] += 1
            if not visible:
                c['Unknown_functions'] += 1;c['reason:outside_2048_window'] += 1
                continue
            accepted[split][family].add(r['sample_key']);projects[split].add(m['project'])
            templates[split][proofs[0]['template']] += 1
            for variant in selected:
                c[variant['kind']+'_variants'] += 1
                c[variant['verified']['state']+'_variants'] += 1
            c['safety_difference_pairs'] += sum(v['kind'].startswith('unsafe') for v in selected)
            c['safe_property_preserving_pairs'] += sum(v['kind']=='safe_change' for v in selected)
            c['semantic_preserving_pairs'] += sum(v['kind']=='semantic_control' for v in selected)
            records.append(dict(sample_key=r['sample_key'],split=split,original_function_label=r['label'],
                family=family,source=source,language=language,project=m['project'],repair_group=list(m['group']),
                commit_url=m['commit_url'],cve=m['cve'],original_proof=proofs,
                variants=selected,derived_function_label=None))
    root.mkdir(parents=True)
    with (root/'samples.jsonl').open('x') as out:
        for record in records:
            out.write(json.dumps(record)+'\n')
    for split in counts:
        for family in FAMILIES:
            counts[split][family]['independent_source_functions'] = len(accepted[split][family])
    prior = {s:set() for s in ('train','valid')}
    prior_path = Path('data/cgvl_admission_seed42/pairs.jsonl')
    if prior_path.exists():
        for line in prior_path.open():
            r=json.loads(line)
            if r['kind']=='unsafe':prior[r['split']].add(r['sample_key'])
    report = dict(protocol='CPU entry-reachable local property construction; no function-label mutation',
        dataset=str(dataset),reference_run_dir=str(reference_run_dir),seed=42,source_max_length=2048,
        model_path=config['model_path'],counts=counts,templates=templates,
        projects={s:sorted(p) for s,p in projects.items()},project_overlap=sorted(projects['train']&projects['valid']),
        source_functions={s:{f:sorted(keys) for f,keys in fam.items()} for s,fam in accepted.items()},
        prior_comparison={s:dict(prior_functions=len(prior[s]),current_functions=len(set().union(*accepted[s].values())),
            added_sources=sorted(set().union(*accepted[s].values())-prior[s]),
            overlap=len(set().union(*accepted[s].values())&prior[s])) for s in prior},
        readiness=False,note='No readiness inferred from pair multiplicity. Pointer means nullness only; independent validation and structural diversity must be reviewed.',
        test_used=False,gpu_used=False)
    atomic_json(root/'coverage.json',report)
    return report
