"""Local node/edge behavior sidecars on the unchanged original C CFG."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

from .cfg_data import (abstract_cfg, atomic_json, cohort_hash, digest, file_sha256,
                       read_records, source_hash, read_jsonl, iter_jsonl, load_graphs)

SCHEMA = 1
NODE_FAMILIES = ('node_kind', 'api', 'datatype', 'literal', 'operator', 'access_form', 'access_mode', 'operand_role')
EDGE_FAMILIES = ('branch', 'guard', 'transfer', 'loop_role')
FAMILIES = NODE_FAMILIES + EDGE_FAMILIES
SPECIAL = ('NOT_APPLICABLE', 'NONE', 'UNKNOWN')
VARIANTS = ('behavior_nodes', 'behavior_edges', 'behavior_joint', 'behavior_masked')
ASSIGN = {'assignment', 'assignmentPlus', 'assignmentMinus', 'assignmentMultiplication', 'assignmentDivision',
          'assignmentModulo', 'assignmentAnd', 'assignmentOr', 'assignmentXor', 'assignmentShiftLeft',
          'assignmentArithmeticShiftRight', 'assignmentLogicalShiftRight', 'assignmentExponentiation'}
INC = {'preIncrement', 'postIncrement', 'preDecrement', 'postDecrement'}
ACCESS = {'indirection': 'dereference', 'indexAccess': 'array_subscript',
          'fieldAccess': 'direct_field', 'indirectFieldAccess': 'indirect_field', 'addressOf': 'address_of'}


def literal(code):
    """Lexical normalization without a target ABI, suffix erasure or clipping."""
    s = code.strip()
    if s == 'nullptr':
        return ['kind=nullptr', 'value=nullptr']
    if s in ('true', 'false'):
        return ['kind=boolean', 'value='+s]
    match = re.fullmatch(r"([+-]?)(0[xX][0-9a-fA-F']+|0[bB][01']+|[0-9][0-9']*)([uUlLzZ]*)", s)
    if match:
        sign, digits, suffix = match.groups()
        digits = digits.replace("'", '')
        base = 16 if digits.lower().startswith('0x') else 2 if digits.lower().startswith('0b') else 8 if digits.startswith('0') else 10
        try:
            value = int(digits, base) * (-1 if sign == '-' else 1)
            return ['kind=integer', 'base='+str(base), 'suffix='+suffix.lower(), 'sign='+('minus' if sign=='-' else 'plus' if sign=='+' else 'none'), 'value='+str(value)]
        except ValueError:
            return ['kind=integer', 'UNKNOWN', 'lexeme='+s]
    if re.fullmatch(r"(?:[+-]?(?:\d[\d']*\.?[\d']*|\.[\d']+)(?:[eE][+-]?\d+)?[fFlL]?|[+-]?0[xX][\da-fA-F']*\.?[\da-fA-F']+[pP][+-]?\d+[fFlL]?)", s):
        clean=s.replace("'", '')
        suffix=clean[-1] if clean[-1] in 'fFlL' else ''
        number=clean[:-1] if suffix else clean
        result=['kind=floating', 'lexeme='+clean, 'suffix='+suffix.lower()]
        if '0x' not in number.lower():
            from decimal import Decimal, InvalidOperation
            try:
                parts=Decimal(number).as_tuple();digits=''.join(map(str,parts.digits));exponent=parts.exponent
                while len(digits)>1 and digits.endswith('0'):digits=digits[:-1];exponent+=1
                result.append(f'value=decimal:{parts.sign}:{digits}:{exponent}')
            except InvalidOperation:result.append('UNKNOWN')
        else:
            result.append('value='+number.lower())
        return result
    chunks=[];position=0
    token=re.compile(r'(u8|u|U|L)?("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')',re.S)
    raw=re.compile(r'(u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\((.*?)\)\2"',re.S)
    while position<len(s):
        if s[position].isspace():position+=1;continue
        match=raw.match(s,position)
        is_raw=match is not None
        if not match:match=token.match(s,position)
        if not match:return ['UNKNOWN','lexeme='+s]
        if is_raw:
            prefix,_,body=match.groups();quote='"';units=[(ord(c),False) for c in body]
        else:
            prefix,quoted=match.groups();quote=quoted[0];units=[]
            pieces=re.findall(r'\\(?:x[0-9a-fA-F]+|[0-7]{1,3}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|.)|[^\\]',quoted[1:-1],re.S)
            escapes={'a':7,'b':8,'f':12,'n':10,'r':13,'t':9,'v':11,'\\':92,'"':34,"'":39,'?':63}
            for piece in pieces:
                if not piece.startswith('\\'):units.append((ord(piece),False))
                elif piece[1:] in escapes:units.append((escapes[piece[1:]],True))
                elif piece.startswith('\\x'):units.append((int(piece[2:],16),True))
                elif piece.startswith(('\\u','\\U')):units.append((int(piece[2:],16),False))
                elif piece[1:].isdigit():units.append((int(piece[1:],8),True))
                else:return ['kind='+('string' if quote=='"' else 'character'),'lexeme='+s,'UNKNOWN']
        chunks.append((prefix or 'ordinary',quote,units));position=match.end()
    prefixes={p for p,_,_ in chunks if p!='ordinary'}
    if not chunks or len(prefixes)>1 or (len(chunks)>1 and any(q!="\"" for _,q,_ in chunks)):
        return ['UNKNOWN','lexeme='+s]
    prefix=next(iter(prefixes),'ordinary');units=[u for _,_,us in chunks for u in us]
    result=['kind='+('string' if chunks[0][1]=='"' else 'character'),'prefix='+prefix,'lexeme='+s,
            'value_units='+','.join(str(v) for v,_ in units)]
    length=0
    for value,numeric in units:
        if numeric:length+=1
        elif value<128:length+=1
        elif prefix=='u8' and value<=0x10ffff and not 0xd800<=value<=0xdfff:length+=len(chr(value).encode('utf-8'))
        elif prefix=='u' and value<=0x10ffff and not 0xd800<=value<=0xdfff:length+=1+(value>0xffff)
        elif prefix=='U' and value<=0x10ffff and not 0xd800<=value<=0xdfff:length+=1
        else:length=None;break
    result.append('length='+str(length) if length is not None else 'length=UNKNOWN')
    return result


def type_features(value, role):
    if not isinstance(value, str) or value in ('', 'ANY', '<unknown>', 'UNKNOWN'):
        return [role+':UNKNOWN']
    value = ' '.join(value.split())
    value = re.sub(r'\[([^\]]*)\]', lambda m: '['+m[1].strip()+']' if re.fullmatch(r'\s*[0-9]+\s*',m[1]) else '[UNKNOWN]', value)
    value = re.sub(r'(anonymous|<lambda>)(?:[_.$:]?[0-9]+)+', r'\1', value)
    result = [role+':type='+value, role+':pointer_depth='+str(value.count('*')),
              role+':array_depth='+str(value.count('[')), role+':width=UNKNOWN']
    result += [role+':qualifier='+q for q in ('const','volatile','restrict','_Atomic') if re.search(r'\b'+q+r'\b',value)]
    result += [role+':signedness='+('unsigned' if re.search(r'\bunsigned\b',value) else 'signed' if re.search(r'\b(signed|short|int|long)\b',value) else 'UNKNOWN')]
    return result


class LocalGraph:
    def __init__(self, graph):
        self.nodes = {n['id']:n for n in graph['nodes']}
        self.ast, self.parents, self.conditions = defaultdict(list), defaultdict(list), defaultdict(list)
        self.receivers = set()
        for e in graph['edges']:
            if e['kind']=='AST':
                self.ast[e['source']].append(e['target']);self.parents[e['target']].append(e['source'])
            elif e['kind']=='CONDITION':self.conditions[e['source']].append(e['target'])
            elif e['kind']=='RECEIVER':self.receivers.add(e['target'])
        self._desc = {}
        self._syntax_parts = {}
    def prop(self, node):return self.nodes[node]['properties']
    def kind(self, node):return self.prop(node).get('kind', 'UNKNOWN')
    def op(self, node):
        op=str(self.prop(node).get('NAME','')).removeprefix('<operator>.')
        return 'indexAccess' if op in ('indirectIndexAccess','indexAccess') else op
    def children(self, node):return self.ast.get(node, [])
    def argument(self, node, index):
        result=[c for c in self.children(node) if self.prop(c).get('ARGUMENT_INDEX')==index]
        return result[0] if len(result)==1 else None
    def ordered(self,node,index):
        result=[c for c in self.children(node) if self.prop(c).get('ORDER')==index]
        return result[0] if len(result)==1 else None
    def syntax_part(self,node,field):
        # A declaration in the initializer shifts Joern ORDER values. Recover
        # the named syntactic field, requiring a unique direct AST correspondence.
        if node not in self._syntax_parts:
            from .syntax import parser_for, walk, source_tokens
            code=self.nodes[node]['code'].strip();prefix='void __cfg(){';encoded=(prefix+code+' {}}').encode()
            parsed=parser_for('cpp').parse(encoded).root_node
            kind=self.prop(node).get('CONTROL_STRUCTURE_TYPE')
            wanted={'IF':'if_statement','FOR':'for_statement','WHILE':'while_statement','DO':'do_statement','SWITCH':'switch_statement'}.get(kind)
            statements=[n for n in walk(parsed) if n.type==wanted and n.start_byte==len(prefix)]
            parts={}
            if len(statements)==1:
                for name in ('condition','update'):
                    part=statements[0].child_by_field_name(name)
                    while part is not None and part.type in ('parenthesized_expression','condition_clause'):
                        value=part.child_by_field_name('value')
                        part=value if value is not None else part.named_children[0] if len(part.named_children)==1 else None
                    if part is not None and not part.has_error:
                        tokens=source_tokens(encoded[part.start_byte:part.end_byte].decode())
                        matches=[c for c in self.children(node) if self.kind(c) not in ('LOCAL','BLOCK')
                                 and source_tokens(self.nodes[c]['code'])==tokens]
                        if len(matches)==1:parts[name]=matches[0]
            self._syntax_parts[node]=parts
        return self._syntax_parts[node].get(field)
    def body(self,node):
        control=self.prop(node).get('CONTROL_STRUCTURE_TYPE')
        # Joern AST role slots, verified against braced/unbraced and declaration
        # initializer exports; these are not CFG successor or source-line guesses.
        if control=='FOR':
            children=[c for c in self.children(node) if self.kind(c)!='LOCAL']
            candidate=max(children,key=lambda c:self.prop(c).get('ORDER',-1)) if children else None
            return candidate if candidate not in (self.condition(node),self.syntax_part(node,'update')) else None
        if control in ('IF','WHILE','SWITCH'):return self.ordered(node,2)
        if control=='DO':return self.ordered(node,1)
        blocks=[c for c in self.children(node) if self.kind(c)=='BLOCK']
        return blocks[0] if len(blocks)==1 else None
    def condition(self,node):
        known=self.conditions.get(node, [])
        if len(known)==1:return known[0]
        kind=self.prop(node).get('CONTROL_STRUCTURE_TYPE')
        return self.syntax_part(node,'condition') if kind in ('IF','FOR','WHILE','DO','SWITCH') else None
    def descendants(self,node):
        if node is None:return set()
        if node in self._desc:return self._desc[node]
        seen=set();todo=[node]
        while todo:
            n=todo.pop()
            if n in seen:continue
            seen.add(n)
            if self.kind(n)!='METHOD' or n==node:todo.extend(self.children(n))
        self._desc[node]=seen
        return seen
    def local(self,node):
        todo=[node];seen=set()
        while todo:
            n=todo.pop()
            if n in seen:continue
            seen.add(n);yield n
            if self.kind(n) in ('METHOD','METHOD_REF','BLOCK','TYPE_DECL','LOCAL'):continue
            if self.kind(n)=='CONTROL_STRUCTURE':
                c=self.condition(n)
                if c is not None:todo.append(c)
            else:todo.extend(self.children(n))
    def role(self,node):
        parents=self.parents[node]
        if len(parents)!=1:return 'UNKNOWN' if parents else 'NOT_APPLICABLE'
        parent=parents[0];op=self.op(parent);p=self.prop(node);i=p.get('ARGUMENT_INDEX')
        if node in self.receivers:return 'call_receiver'
        if self.kind(parent)=='RETURN':return 'return_value'
        if self.kind(parent)=='CONTROL_STRUCTURE':return 'condition' if node==self.condition(parent) else 'NOT_APPLICABLE'
        if self.kind(parent)!='CALL':return 'NOT_APPLICABLE'
        if op=='pointerCall' or not str(self.prop(parent).get('NAME','')).startswith('<operator>.'):
            return 'call_receiver' if i==0 or (op=='pointerCall' and i==-1) else 'argument:'+str(i) if isinstance(i,int) and i>0 else 'UNKNOWN'
        if op in ASSIGN:return {1:'assignment_left',2:'assignment_right'}.get(i,'UNKNOWN')
        if op=='indexAccess':return {1:'array_base',2:'array_index'}.get(i,'UNKNOWN')
        if op in ('fieldAccess','indirectFieldAccess'):return {1:'field_base',2:'field_name'}.get(i,'UNKNOWN')
        if op=='cast':return 'cast_operand' if i==2 else 'cast_type'
        if op=='conditional':return {1:'condition',2:'true_operand',3:'false_operand'}.get(i,'UNKNOWN')
        return ('unary_operand' if len(self.children(parent))==1 else {1:'binary_left',2:'binary_right'}.get(i,'UNKNOWN'))
    def access_mode(self,node):
        if self.kind(node)=='FIELD_IDENTIFIER':return 'NOT_APPLICABLE'
        if self.kind(node) not in ('IDENTIFIER','CALL'):return 'NOT_APPLICABLE'
        if self.kind(node)=='CALL' and self.op(node) not in ACCESS:
            return 'UNKNOWN' if self.op(node)=='pointerCall' or not str(self.prop(node).get('NAME','')).startswith('<operator>.') else 'NOT_APPLICABLE'
        current=node;seen=set();address=False;address_operand=False
        while len(self.parents[current])==1:
            parent=self.parents[current][0]
            if parent in seen:break
            seen.add(parent);op=self.op(parent)
            if op in ('sizeOf','alignOf'):
                typ=self.prop(current).get('TYPE_FULL_NAME','ANY')
                if typ in ('ANY','<unknown>','') or re.search(r'\[[^\]0-9]*[A-Za-z_]',typ):return 'UNKNOWN'
                return 'UNEVALUATED'
            if op=='addressOf':address=True
            index=self.prop(current).get('ARGUMENT_INDEX')
            if (op in ('indirection','indirectFieldAccess') and index==1) or (op=='indexAccess' and index in (1,2)):
                address_operand=True
            if self.kind(parent) in ('BLOCK','METHOD','CONTROL_STRUCTURE'):break
            current=parent
        parents=self.parents[node]
        if len(parents)>1:return 'UNKNOWN'
        if parents:
            parent=parents[0];op=self.op(parent);i=self.prop(node).get('ARGUMENT_INDEX')
            if op in ASSIGN and i==1:return 'WRITE' if op=='assignment' else 'READ_WRITE'
            if op in INC:return 'READ_WRITE'
            if op in ('indirection','indirectFieldAccess','indexAccess') and i==1:
                return 'ADDRESS_ONLY' if '[' in str(self.prop(node).get('TYPE_FULL_NAME','')) else 'READ_ADDRESS_OPERAND'
            if op=='indexAccess' and i==2:return 'READ_ADDRESS_OPERAND'
            if op=='fieldAccess' and i==1:return 'ADDRESS_ONLY'
        if address_operand:return 'READ_ADDRESS_OPERAND'
        if address:return 'ADDRESS_ONLY'
        return 'READ'


def node_features(g,root,local=True):
    p=g.prop(root);kind=g.kind(root)
    features={f:[] for f in NODE_FAMILIES}
    features['node_kind']=[kind]
    if kind=='CALL':features['node_kind'].append('expression' if str(p.get('NAME','')).startswith('<operator>.') and g.op(root)!='pointerCall' else 'call')
    elif kind in ('IDENTIFIER','LITERAL','FIELD_IDENTIFIER'):features['node_kind'].append('expression')
    if kind=='CONTROL_STRUCTURE':features['node_kind'].append(p.get('CONTROL_STRUCTURE_TYPE','UNKNOWN'))
    if kind=='METHOD':features['node_kind'].append('entry')
    if kind=='METHOD_RETURN':features['node_kind'].append('exit')
    if any(root==g.condition(parent) for parent in g.parents[root]):features['node_kind'].append('condition')
    for n in (g.local(root) if local else [root]):
        q=g.prop(n);k=g.kind(n);op=g.op(n);role='result' if n==root else g.role(n)
        if k in ('IDENTIFIER','CALL','LITERAL','METHOD_RETURN','METHOD_PARAMETER_IN'):
            features['datatype'].extend(type_features(q.get('TYPE_FULL_NAME'),role))
        if k=='LITERAL':features['literal'].extend(literal(g.nodes[n]['code']))
        if k=='CALL':
            name=str(q.get('NAME',''))
            if name.startswith('<operator>.'):
                if op=='pointerCall':features['api'].extend(['dispatch=DYNAMIC_DISPATCH','callee=UNKNOWN'])
                features['operator'].append(('root:' if n==root else 'internal:')+op)
                if op in ACCESS:features['access_form'].append(ACCESS[op])
            else:
                dispatch=q.get('DISPATCH_TYPE','UNKNOWN')
                features['api'].append('dispatch='+dispatch)
                # Dynamic dispatch cannot turn a variable/function-pointer name into an API category.
                features['api'].append('name='+name if dispatch=='STATIC_DISPATCH' and name else 'UNKNOWN')
        if k=='IDENTIFIER':features['access_form'].append('scalar')
        if k in ('IDENTIFIER','CALL'):
            mode=g.access_mode(n)
            features['access_mode'].append(('object:' if op in ACCESS else 'operand:')+mode)
        features['operand_role'].append(g.role(n))
    for f,values in features.items():
        features[f]=values or ['NOT_APPLICABLE' if kind in ('METHOD','METHOD_RETURN','BLOCK') else 'NONE']
    return [features[f] for f in NODE_FAMILIES]


def loop_roles(view,kinds):
    """Natural loops from entry dominators; SCC membership is not a back-edge test."""
    n=len(view['node_ids']);succ=[set() for _ in range(n)];pred=[set() for _ in range(n)]
    for a,b in view['edges']:succ[a].add(b);pred[b].add(a)
    roots=[i for i,k in enumerate(kinds) if k=='METHOD']
    if len(roots)!=1:return [['UNKNOWN'] for _ in view['edges']]
    entry=roots[0];reachable={entry};todo=[entry]
    while todo:
        for v in succ[todo.pop()]:
            if v not in reachable:reachable.add(v);todo.append(v)
    mask=sum(1<<v for v in reachable);dom={v:mask for v in reachable};dom[entry]=1<<entry
    changed=True
    while changed:
        changed=False
        for v in reachable-{entry}:
            incoming=[p for p in pred[v] if p in reachable]
            value=mask
            for p in incoming:value &= dom[p]
            value |= 1<<v
            if value!=dom[v]:dom[v]=value;changed=True
    natural={}
    back=set()
    for u,v in view['edges']:
        if u in dom and dom[u] & (1<<v):
            back.add((u,v));members=natural.setdefault(v,{v});todo=[u] if u!=v else []
            while todo:
                w=todo.pop()
                if w in members:continue
                members.add(w);todo.extend(p for p in pred[w] if p in reachable)
    # Iterative SCCs, including unreachable cycles; multiple external entries
    # identify irreducibility independently of natural-loop flags.
    seen=set();order=[]
    for start in range(n):
        if start in seen:continue
        seen.add(start);stack=[(start,iter(succ[start]))]
        while stack:
            u,it=stack[-1];v=next(it,None)
            if v is None:order.append(u);stack.pop()
            elif v not in seen:seen.add(v);stack.append((v,iter(succ[v])))
    seen=set();irreducible=set()
    for start in reversed(order):
        if start in seen:continue
        component=set();todo=[start];seen.add(start)
        while todo:
            u=todo.pop();component.add(u)
            for v in pred[u]:
                if v not in seen:seen.add(v);todo.append(v)
        entries={v for v in component if any(p not in component for p in pred[v])}
        if len(component)>1 and len(entries)>1:irreducible.update(component)
    result=[]
    for u,v in view['edges']:
        flags=[]
        if u not in reachable or v not in reachable:flags.append('UNKNOWN')
        for header,members in natural.items():
            if u in members and v in members:flags.append('inside')
            elif u not in members and v in members:flags.append('entry')
            elif u in members and v not in members:flags.append('exit')
        if (u,v) in back:flags.append('back')
        if u in irreducible or v in irreducible:flags.append('irreducible')
        result.append(sorted(flags) or ['NONE'])
    return result


def extract_behavior(graph):
    view=abstract_cfg(graph);g=LocalGraph(graph);ids=view['node_ids'];used=set(ids)
    attributes=[node_features(g,n) for n in ids]
    condition_rules=defaultdict(list);switch_targets={};catch_nodes=set()
    for root in g.nodes:
        p=g.prop(root);control=p.get('CONTROL_STRUCTURE_TYPE');op=g.op(root)
        if control=='CATCH':catch_nodes.update(g.descendants(root))
        cond=g.condition(root)
        if control in ('IF','WHILE','FOR','DO') and cond:
            body=g.body(root)
            if body is None:
                condition_rules[cond].append((None,None,True));continue
            other=[c for c in g.children(root) if g.prop(c).get('CONTROL_STRUCTURE_TYPE')=='ELSE']
            alternative=other[0] if control=='IF' and len(other)==1 else None
            positive=g.descendants(body)&used
            if control!='IF' and not positive:
                positive=g.descendants(g.syntax_part(root,'update') if control=='FOR' else None)&used
                if not positive:positive=g.descendants(cond)&used
            condition_rules[cond].append((positive,g.descendants(alternative)&used,alternative is not None))
        elif op=='conditional':
            cond=g.argument(root,1)
            if cond:condition_rules[cond].append((g.descendants(g.argument(root,2))&used,g.descendants(g.argument(root,3))&used,True))
        elif op in ('logicalAnd','logicalOr'):
            left=g.argument(root,1);right=g.descendants(g.argument(root,2))&used
            if left:
                condition_rules[left].append((right,{root},True) if op=='logicalAnd' else ({root},right,True))
        elif control=='SWITCH' and cond:
            targets={}
            for n in g.descendants(root):
                if g.kind(n)!='JUMP_TARGET':continue
                ancestors=list(g.parents[n]);nearest=None
                while len(ancestors)==1:
                    parent=ancestors[0]
                    if g.prop(parent).get('CONTROL_STRUCTURE_TYPE')=='SWITCH':nearest=parent;break
                    ancestors=g.parents[parent]
                if nearest!=root:continue
                code=g.nodes[n]['code'].strip()
                if re.match(r'^default\s*:',code):targets[n]=('default',[])
                elif re.match(r'^case\b',code):targets[n]=('case',literal(re.sub(r'^case\s*|:\s*$','',code)))
            switch_targets[cond]=targets
    successors=defaultdict(set)
    for a,b in view['edges']:successors[ids[a]].add(ids[b])
    loops=loop_roles(view,[g.kind(n) for n in ids]);result=[]
    for (a,b),loop in zip(view['edges'],loops):
        u,v=ids[a],ids[b];branches=[]
        for positive,negative,has_else in condition_rules.get(u,()):
            if positive is None:
                branches.append(('UNKNOWN',[]));continue
            if v in positive:branches.append(('true',[]))
            if v in negative:branches.append(('false',[]))
            if v not in positive and v not in negative:
                if not has_else:branches.append(('false',[]))
                if not positive:branches.append(('true',[]))
                if has_else and not negative:branches.append(('false',[]))
        if u in switch_targets:
            if v in switch_targets[u]:branches.append(switch_targets[u][v])
            elif not any(t[0]=='default' for t in switch_targets[u].values()):branches.append(('default',[]))
        if not branches:branches=[('UNKNOWN' if len(successors[u])>1 or u in condition_rules or u in switch_targets else 'unconditional',[])]
        transfer=[];control=g.prop(u).get('CONTROL_STRUCTURE_TYPE')
        if g.kind(u)=='METHOD':transfer.append('function_enter')
        if g.kind(v)=='METHOD_RETURN':transfer.append('function_exit')
        if g.kind(u)=='RETURN':transfer.append('return')
        if control in ('BREAK','CONTINUE','GOTO'):transfer.append(control.lower())
        if control=='THROW' or (v in catch_nodes and u not in catch_nodes):transfer.append('explicit_exception')
        if g.kind(v)=='JUMP_TARGET' and any(v in t for t in switch_targets.values()) and u not in switch_targets:transfer.append('switch_fallthrough')
        guard=[]
        if u in condition_rules or u in switch_targets:
            local=node_features(g,u)
            guard=[f+':'+token for f,tokens in zip(NODE_FAMILIES,local) if f in ('operator','datatype','literal','operand_role') for token in tokens]
        alternatives=[]
        for branch,case in branches:
            alternative=[[branch],guard+['case:'+x for x in case] or ['NONE' if branch=='unconditional' else 'UNKNOWN'],transfer or ['ordinary'],loop]
            if alternative not in alternatives:alternatives.append(alternative)
        result.append(alternatives)
    return {'schema':SCHEMA,'node_ids':ids,'cfg_edges':[list(e) for e in view['edges']],
            'nodes':attributes,'edges':result}


class BehaviorVocabulary:
    def __init__(self,values):
        if set(values)!=set(FAMILIES) or any(v[:3]!=list(SPECIAL) or len(v)!=len(set(v)) for v in values.values()):
            raise ValueError('invalid behavior vocabulary')
        self.values=values;self.indices={f:{v:i for i,v in enumerate(values[f])} for f in FAMILIES}
    @classmethod
    def fit(cls,records,features,limit=2048):
        if any(r['split']!='train' for r in records):raise ValueError('behavior vocabulary is train-only')
        counts={f:Counter() for f in FAMILIES}
        for row in records:
            item=features[row['sample_key']]
            for values in item['nodes']:
                for f,tokens in zip(NODE_FAMILIES,values):counts[f].update(tokens)
            for alternatives in item['edges']:
                for values in alternatives:
                    for f,tokens in zip(EDGE_FAMILIES,values):counts[f].update(tokens)
        return cls({f:list(SPECIAL)+[t for t,_ in sorted(c.items(),key=lambda x:(-x[1],x[0])) if t not in SPECIAL][:limit-3] for f,c in counts.items()})
    def encode(self,item):
        def encode(families,values):return [[self.indices[f].get(t,2) for t in ts] for f,ts in zip(families,values)]
        return {'nodes':[encode(NODE_FAMILIES,v) for v in item['nodes']],
                'edges':[[encode(EDGE_FAMILIES,v) for v in alternatives] for alternatives in item['edges']]}
    def sizes(self):return [len(self.values[f]) for f in FAMILIES]


def prepare_behavior(reference_run_dir,output_dir,supplement_path=None, *, bound=False):
    root=Path(output_dir);reference=Path(reference_run_dir).resolve()
    if root.exists():raise FileExistsError(f'behavior output exists: {root}')
    config=json.loads((reference/'config.json').read_text())
    records=read_records(config['dataset'],config['source_dataset'])
    if cohort_hash(records)!=config['cohort_sha256'] or file_sha256(config['graphs'])!=config['graph_file_sha256']:
        raise ValueError('behavior preparation differs from original C')
    graphs=load_graphs(config['graphs'],records)
    supplemented=set()
    if supplement_path:
        by_key={r['sample_key']:r for r in records}
        for item in read_jsonl(supplement_path):
            key=item['sample_key']
            if key not in graphs or key in supplemented or item['original_graph_sha256']!=config['graph_file_sha256'] or item['source_sha256']!=source_hash(by_key[key]['raw_source']):raise ValueError('supplement mismatch')
            original=abstract_cfg(graphs[key]);replacement=abstract_cfg(item['graph'])
            if original['node_ids']!=replacement['node_ids'] or original['edges']!=replacement['edges']:raise ValueError('supplement changed original CFG')
            graphs[key]=item['graph'];supplemented.add(key)
    features={};counts={s:Counter() for s in ('train','valid')};examples={s:[] for s in counts}
    for i,row in enumerate(records):
        key=row['sample_key'];item=(extract_bound_behavior if bound else extract_behavior)(graphs.pop(key));features[key]=item
        if row['split'] in counts:
            c=counts[row['split']];c.update(functions=1,nodes=len(item['nodes']),edges=len(item['edges']),supplemented_functions=int(key in supplemented))
            for values in item['nodes']:
                c['non_assignment_nodes']+=not any(t.startswith('root:assignment') for t in values[4])
                c['non_assignment_with_local_attributes']+=not any(t.startswith('root:assignment') for t in values[4]) and any(t not in SPECIAL and 'UNKNOWN' not in t and 'NOT_APPLICABLE' not in t for ts in values[1:] for t in ts)
                for family,tokens in zip(NODE_FAMILIES,values):
                    c[family+'_present']+=any(t not in SPECIAL and 'UNKNOWN' not in t and 'NOT_APPLICABLE' not in t for t in tokens)
                    c[family+'_unknown']+=any('UNKNOWN' in t for t in tokens)
            for alternatives in item['edges']:
                for j,family in enumerate(EDGE_FAMILIES):
                    c[family+'_present_edges']+=any(t not in SPECIAL and 'UNKNOWN' not in t and 'NOT_APPLICABLE' not in t for a in alternatives for t in a[j])
                    c[family+'_unknown_edges']+=any('UNKNOWN' in t for a in alternatives for t in a[j])
                c['multiple_transfer_edges']+=len(alternatives)>1
                c['branch_edges']+=any(a[0][0] in ('true','false','case','default') for a in alternatives)
            if len(examples[row['split']])<2:examples[row['split']].append({'sample_key':key,'source':row['raw_source'],'attributes':item})
        if (i+1)%200==0:print(f'behavior_prepare={i+1}/{len(records)}',flush=True)
    vocab=(BoundVocabulary if bound else BehaviorVocabulary).fit([r for r in records if r['split']=='train'],features)
    for row in records:
        if row['split'] not in counts:continue
        item=features[row['sample_key']];c=counts[row['split']]
        if bound:
            node_groups=[[[f['value'] for f in facts] for facts in node] for node in item['node_facts']]
            edge_groups=[[[f['value'] for f in facts] for facts in alt] for alts in item['edge_facts'] for alt in alts]
        else:
            node_groups=item['nodes'];edge_groups=[v for alts in item['edges'] for v in alts]
        for families,groups in ((NODE_FAMILIES,node_groups), (EDGE_FAMILIES,edge_groups)):
            for values in groups:
                for f,tokens in zip(families,values):
                    c[f+'_tokens']+=len(tokens)
                    c[f+'_oov_tokens']+=sum(t not in vocab.indices[f] for t in tokens)
    root.mkdir(parents=True)
    with (root/'attributes.jsonl').open('x') as out:
        for row in records:
            out.write(json.dumps({'sample_key':row['sample_key'],'split':row['split'],'source_sha256':source_hash(row['raw_source']),**features[row['sample_key']]})+'\n')
    atomic_json(root/'vocabulary.json',vocab.values)
    report={'schema':BOUND_SCHEMA if bound else SCHEMA,'reference_run_dir':str(reference),'c_config':config,
        'attributes_sha256':file_sha256(root/'attributes.jsonl'),'vocabulary_sha256':digest(vocab.values),
        'node_families':NODE_FAMILIES,'edge_families':EDGE_FAMILIES,'splits':counts,'examples':examples}
    if bound:report['bound_coverage']=bound_coverage(root)
    atomic_json(root/'audit.json',report);return report


def _attribute_rows(path):
    """Keep the multi-gigabyte attribute sidecar off the Python heap."""
    yield from iter_jsonl(path)


def load_behavior(directory,records,views,reference_run_dir):
    root=Path(directory);audit=json.loads((root/'audit.json').read_text());config=json.loads((Path(reference_run_dir)/'config.json').read_text())
    values=json.loads((root/'vocabulary.json').read_text());vocab=(BoundVocabulary if audit['schema']==BOUND_SCHEMA else BehaviorVocabulary)(values)
    if (audit['schema'] not in (SCHEMA, BOUND_SCHEMA) or audit['c_config']!=config or cohort_hash(records)!=config['cohort_sha256'] or
        audit['vocabulary_sha256']!=digest(values)):
        raise ValueError('behavior cache identity/schema mismatch')
    expected={r['sample_key']:r for r in records};seen=set()
    for item in iter_jsonl(root/'attributes.jsonl', description='Load attributes',
                          expected_sha256=audit['attributes_sha256']):
        key=item['sample_key'];row=expected.get(key)
        if row is None or key in seen or item['split']!=row['split'] or item['source_sha256']!=source_hash(row['raw_source']):raise ValueError('behavior source/split mismatch')
        view=views[key]
        if item['schema']!=audit['schema'] or item['node_ids']!=view['node_ids'] or item['cfg_edges']!=[list(e) for e in view['edges']] or len(item['nodes'])!=len(view['node_ids']) or len(item['edges'])!=len(view['edges']):raise ValueError('behavior CFG alignment mismatch')
        if any(len(v)!=8 or any(not ts for ts in v) for v in item['nodes']) or any(not alts or any(len(v)!=4 or any(not ts for ts in v) for v in alts) for alts in item['edges']):raise ValueError('incomplete behavior attributes')
        seen.add(key);view['behavior']=vocab.encode(item)
    if seen!=set(expected):raise ValueError('missing behavior functions')
    return vocab,audit


def supplement_behavior(reference_run_dir, output_path, joern_dir='/home/phy/joern', java_home='/home/phy/jdk21'):
    """Incremental fresh attribute export, accepted only if original AST/CFG match.

    New Joern IDs are mapped by unique rooted AST roles, never by equal code or
    node numbering. The original sidecar is never written. Failed matches stop
    the run, preserving completed records for the next invocation.
    """
    from .cpg import extract_function_cpg_batch
    from .cfg_data import serialize_graph, output_lock, repair_tail
    reference=Path(reference_run_dir).resolve();config=json.loads((reference/'config.json').read_text())
    records=read_records(config['dataset'],config['source_dataset'])
    if cohort_hash(records)!=config['cohort_sha256'] or file_sha256(config['graphs'])!=config['graph_file_sha256']:
        raise ValueError('supplement inputs differ from C')
    path=Path(output_path)
    if path.resolve()==Path(config['graphs']).resolve():raise ValueError('cannot overwrite original CFG')
    def paths(graph):
        g=LocalGraph(graph);roots=[n for n in g.nodes if g.kind(n)=='METHOD' and not g.parents[n]]
        if len(roots)!=1:raise ValueError('supplement requires one rooted method')
        result={};todo=[(roots[0],())]
        while todo:
            n,key=todo.pop()
            if key in result:raise ValueError('ambiguous AST role path; supplement rejected')
            result[key]=n
            for c in g.children(n):
                p=g.prop(c);todo.append((c,key+((p.get('ORDER'),p.get('ARGUMENT_INDEX'),p['kind']),)))
        if len(result)!=len(g.nodes):raise ValueError('unmapped original AST nodes')
        return result,g
    with output_lock(path):
        repair_tail(path);completed={}
        if path.exists():
            for row in read_jsonl(path):
                if row.get('original_graph_sha256')!=config['graph_file_sha256'] or row['sample_key'] in completed:raise ValueError('supplement provenance mismatch')
                completed[row['sample_key']]=row['source_sha256']
        graphs=load_graphs(config['graphs'],records);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as out:
            for row in records:
                key=row['sample_key']
                if key in completed:
                    if completed[key]!=source_hash(row['raw_source']):raise ValueError('supplement source changed')
                    continue
                from .syntax import resolve_language
                language=resolve_language(row['raw_source'],row.get('language','c_cpp'),'')
                result=extract_function_cpg_batch([{'source':row['raw_source'],'language':language,'function':row.get('function_name','')}],joern_dir=joern_dir,java_home=java_home)[0]
                if isinstance(result,Exception):raise result
                fresh=serialize_graph(result);old_paths,old=paths(graphs[key]);new_paths,new=paths(fresh)
                if set(old_paths)!=set(new_paths):raise ValueError(f'{key}: AST changed; original CFG retained')
                mapping={new_paths[p]:old_paths[p] for p in old_paths}
                for p,old_id in old_paths.items():
                    if old.nodes[old_id]['code']!=new.nodes[new_paths[p]]['code']:raise ValueError(f'{key}: node text changed at AST role')
                old_edges={(e['source'],e['target']) for e in graphs[key]['edges'] if e['kind']=='CFG'}
                new_edges={(mapping[e['source']],mapping[e['target']]) for e in fresh['edges'] if e['kind']=='CFG'}
                if old_edges!=new_edges:raise ValueError(f'{key}: frontend CFG changed; refusing replacement')
                for node in fresh['nodes']:node['id']=mapping[node['id']]
                for edge in fresh['edges']:edge['source']=mapping[edge['source']];edge['target']=mapping[edge['target']]
                out.write(json.dumps({'sample_key':key,'source_sha256':source_hash(row['raw_source']),
                    'original_graph_sha256':config['graph_file_sha256'],'attribute_export_version':2,'graph':fresh})+'\n')
                out.flush()
                import os
                os.fsync(out.fileno())
                print(f'behavior_supplement={key}',flush=True)

# Role-bound residual representation; schema 1 remains a reproducible ablation.
BOUND_SCHEMA = 2
JOINT_VARIANTS = ('joint_nodes', 'joint_edges', 'joint', 'joint_shuffled')


def bound_node_facts(g, root):
    """Identity-deduplicated local facts. Object IDs are provenance, never features."""
    allowed = set(g.local(root))
    paths = {root: ['SELF']}
    todo = [root]
    while todo:
        parent = todo.pop()
        for child in g.children(parent):
            if child not in allowed or child in paths:
                continue
            owner = g.op(parent) if str(g.prop(parent).get('NAME', '')).startswith('<operator>.') else g.kind(parent)
            paths[child] = paths[parent] + [owner + '/' + g.role(child)]
            todo.append(child)
    result = [[] for _ in NODE_FAMILIES]
    for node in g.local(root):
        values = node_features(g, node, local=False)
        for family, tokens in enumerate(values):
            for token in sorted(set(tokens)):
                if family == 4 and node != root:
                    token = token.replace('root:', 'internal:', 1)
                result[family].append({'object': node, 'path': paths.get(node, ['UNKNOWN']), 'value': token})
    return result


def extract_bound_behavior(graph):
    item = extract_behavior(graph)
    g = LocalGraph(graph)
    item['schema'] = BOUND_SCHEMA
    item['node_facts'] = [bound_node_facts(g, n) for n in item['node_ids']]
    edge_facts = []
    for (source, _), alternatives in zip(item['cfg_edges'], item['edges']):
        bound = []
        for values in alternatives:
            families = [[{'object': None, 'path': ['SELF'], 'value': v} for v in ts] for ts in values]
            if values[1] not in (['NONE'], ['UNKNOWN']):
                families[1] = [dict(fact, value=NODE_FAMILIES[f]+':'+fact['value'])
                    for f in (2,3,4,7) for fact in item['node_facts'][source][f]]
                families[1] += [{'object': None, 'path': ['case'], 'value': t}
                                for t in values[1] if t.startswith('case:')]
            bound.append(families)
        edge_facts.append(bound)
    item['edge_facts'] = edge_facts
    return item


class BoundVocabulary:
    def __init__(self, values):
        if set(values) != set(FAMILIES) | {'roles'}:
            raise ValueError('role-bound vocabulary required')
        self.values = values
        if any(v[:3] != list(SPECIAL) or len(v) != len(set(v)) for v in values.values()):
            raise ValueError('invalid role-bound vocabulary')
        self.indices = {f: {t:i for i,t in enumerate(ts)} for f,ts in values.items()}
    @classmethod
    def fit(cls, records, features, limit=2048):
        if any(r['split'] != 'train' for r in records):
            raise ValueError('bound vocabulary is train-only')
        counts = {f: Counter() for f in (*FAMILIES, 'roles')}
        for row in records:
            item = features[row['sample_key']]
            for families, groups in ((NODE_FAMILIES, item['node_facts']),
                    (EDGE_FAMILIES, (a for alternatives in item['edge_facts'] for a in alternatives))):
                for group in groups:
                    for family, facts in zip(families, group):
                        for fact in facts:
                            counts[family][fact['value']] += 1
                            counts['roles'].update(fact['path'])
        return cls({f:list(SPECIAL)+['OUT_OF_VOCABULARY']+[t for t,_ in sorted(c.items(),key=lambda x:(-x[1],x[0]))
                                    if t not in SPECIAL][:limit-4] for f,c in counts.items()})
    def sizes(self):return [len(self.values[f]) for f in (*FAMILIES, 'roles')]
    def encode(self, item):
        # The same operand facts recur at enclosing operations and guards.
        # Memoization is function-local: no cross-function growth or vocabulary changes.
        paths={};encoded_facts={};role_indices=self.indices['roles']
        def group(families, values):
            encoded=[]
            for family,facts in zip(families,values):
                unknown_literals={fact.get('object') for fact in facts if
                    (family=='literal' and fact['value']=='UNKNOWN') or
                    (family=='guard' and fact['value'] in ('literal:UNKNOWN','case:UNKNOWN'))} if family in ('literal','guard') else set()
                column=[]
                for fact in facts:
                    # An unparsed frontend literal (e.g. <global>) is not a
                    # usable lexical category. Preserve provenance, mask input.
                    unparsed_lexeme=('lexeme=' in fact['value'] and fact.get('object') in unknown_literals)
                    path=tuple(fact['path']);key=(family,fact['value'],path,unparsed_lexeme)
                    feature=encoded_facts.get(key)
                    if feature is None:
                        if path not in paths:
                            paths[path]=([role_indices.get(role,3) for role in path],
                                         not any(role=='UNKNOWN' or role.endswith('/UNKNOWN') for role in path))
                        role_ids,path_known=paths[path]
                        available='UNKNOWN' not in fact['value'] and path_known and not unparsed_lexeme
                        feature=[self.indices[family].get(fact['value'],3),role_ids,int(available)]
                        encoded_facts[key]=feature
                    column.append(feature)
                encoded.append(column)
            return encoded
        neutral = [[{'path':['SELF'],'value':v}] for v in ('unconditional','NONE','ordinary','NONE')]
        return {'schema':BOUND_SCHEMA, 'nodes':[group(NODE_FAMILIES,v) for v in item['node_facts']],
                'edges':[[group(EDGE_FAMILIES,v) for v in alts] for alts in item['edge_facts']],
                'neutral':group(EDGE_FAMILIES,neutral)}


def bound_coverage(directory):
    """Audit usable facts with the same masks as the encoder, on train/valid only."""
    root=Path(directory)
    vocab=BoundVocabulary(json.loads((root/'vocabulary.json').read_text()))
    counters={s:Counter() for s in ('train','valid')}
    for item in _attribute_rows(root/'attributes.jsonl'):
        if item['split'] not in counters:continue
        encoded=vocab.encode(item);c=counters[item['split']]
        for families,raw,groups in ((NODE_FAMILIES,item['node_facts'],encoded['nodes']),
                (EDGE_FAMILIES,[a for alts in item['edge_facts'] for a in alts],
                 [a for alts in encoded['edges'] for a in alts])):
            for values,ids in zip(raw,groups):
                for family,facts,features in zip(families,values,ids):
                    known=[bool(f[2]) and r['value'] not in SPECIAL and
                           not r['value'].endswith(':NONE') and 'NOT_APPLICABLE' not in r['value']
                           for r,f in zip(facts,features)]
                    c[family+'_records']+=1
                    c[family+'_usable_records']+=any(known)
                    c[family+'_unavailable_records']+=any(not f[2] for f in features)
                    c[family+'_facts']+=len(features)
                    c[family+'_unavailable_facts']+=sum(not f[2] for f in features)
                    c[family+'_oov_facts']+=sum(f[0]==3 for f in features)
    return {s:dict(c) for s,c in counters.items()}
