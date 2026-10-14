"""The existing visibility checks and dynamic width-two, depth-three source search."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
import numpy as np

def visible_source_statistics(source, tokenizer, source_budget):
    """Character prefix covered by the encoder's actual source-token budget."""
    if source_budget <= 0 or not tokenizer.is_fast:
        raise ValueError('visible source statistics require a positive budget and fast tokenizer')
    enc = tokenizer(source, add_special_tokens=False, truncation=True,
                    max_length=source_budget, return_offsets_mapping=True)
    offsets = enc['offset_mapping']
    if not offsets or any(not 0 <= a <= b <= len(source) for a, b in offsets):
        raise ValueError('empty or invalid original-source tokenizer offsets')
    end = max(b for a, b in offsets)
    visible = source[:end]
    return visible, dict(visible_tokens=len(enc['input_ids']), visible_chars=end,
                         visible_lines=visible.count('\n')+1)


def visible_semantic_candidates(source, language, tokenizer, source_budget, *, unseen=False, file_name='',
                                proposals=None,reference_source=None):
    """Reject a view if it displaces any part of the original visible prefix."""
    from .syntax import semantic_preserving_candidates, apply_source_edits, resolve_language,source_token_spans
    language = resolve_language(source, language, file_name)
    if proposals is not None:
        if not getattr(tokenizer,'is_fast',False):raise ValueError('dynamic visibility requires fast original source offsets')
        reference=source if reference_source is None else reference_source
        def visibility(value):
            encoded=tokenizer(value,add_special_tokens=False,return_offsets_mapping=True,truncation=False,verbose=False)
            offsets=encoded['offset_mapping'][:source_budget]
            if len(encoded['input_ids'])!=len(encoded['offset_mapping']):raise ValueError('source IDs/offset length mismatch')
            if any(not 0<=a<=b<=len(value) for a,b in offsets):raise ValueError('source offsets outside Python character coordinates')
            # Source offsets are monotone; merge UTF-8/BPE duplicate character spans.
            end=max((b for a,b in offsets if b>a),default=0)
            tokens=source_token_spans(value)
            states=[(0 if a>=end else 2 if b<=end else 1,min(max(end-a,0),b-a)) for _,a,b in tokens]
            return encoded['input_ids'][:source_budget],states,tokens,len(encoded['input_ids'])>source_budget,end
        original_ids,before,original_tokens,truncated,boundary=visibility(reference)
        result=[];reasons=Counter()
        for candidate in proposals:
            transformed=apply_source_edits(source,candidate['edits'])
            ids,after,tokens,_,end=visibility(transformed)
            if len(tokens)!=len(original_tokens):raise ValueError('dynamic surface edit changed lexical token cardinality')
            lost=any(old[0]>new[0] or old[0]==new[0]==1 and old[1]>new[1]
                     for old,new in zip(before,after))
            gained=truncated and any(old[0]<new[0] or old[0]==new[0]==1 and old[1]<new[1]
                                    for old,new in zip(before,after))
            partial_rename=any(old[0]==1 and a[0]!=b[0] for old,a,b in zip(before,original_tokens,tokens))
            if lost:reasons['visible_execution_removed']+=1
            elif gained:reasons['extra_execution_exposed']+=1
            elif partial_rename:reasons['partial_visible_identifier_changed']+=1
            elif ids==original_ids:reasons['identical_encoder_input']+=1
            else:
                result.append(dict(candidate,raw_source=transformed,visible_tokens=len(ids),visible_chars=end,
                    original_visible_chars=boundary,original_truncated=truncated,
                    visible_execution_tokens=sum(s[0]==2 for s in after),visibility_verified=True))
        return result,dict(reasons)
    original, before = visible_source_statistics(source, tokenizer, source_budget)
    original_ids = tokenizer(source, add_special_tokens=False, truncation=True,
                             max_length=source_budget)['input_ids']
    proposed, reasons = semantic_preserving_candidates(source, language, unseen=unseen)
    reasons = Counter(reasons)
    kept = []
    for candidate in proposed:
        boundary, mapped = before['visible_chars'], before['visible_chars']
        if any(a < boundary < b for a, b, _ in candidate['edits']):
            reasons['edit_crosses_visible_boundary'] += 1
            continue
        for a, b, value in candidate['edits']:
            if a < boundary:
                mapped += len(value) - (b-a)
        transformed = apply_source_edits(source, candidate['edits'])
        visible, stats = visible_source_statistics(transformed, tokenizer, source_budget)
        ids = tokenizer(transformed, add_special_tokens=False, truncation=True,
                        max_length=source_budget)['input_ids']
        if stats['visible_chars'] < mapped:
            reasons['would_displace_original_visible_source'] += 1
        elif ids == original_ids:
            reasons['identical_encoder_input'] += 1
        else:
            kept.append(dict(candidate, **stats, original_visible_chars=boundary, preserved_visible_chars=mapped))
    return kept, dict(reasons)


def surface_seed(seed,epoch,sample_key,purpose):
    """Stable method RNG; independent of Python hash, dropout and batch ordering."""
    import hashlib
    return int.from_bytes(hashlib.sha256(json.dumps([seed,epoch,sample_key,purpose]).encode()).digest()[:8],'big')


def replay_surface_units(source,units):
    from .syntax import apply_source_edits
    for unit in units:source=apply_source_edits(source,unit['edits'])
    return source


def search_surface_views(records,score,tokenizer,source_budget,materials,*,seed=42,epoch=0,
                         query_limit=12,bank='train',purpose='train',random_selection=False,
                         kinds=None,required_combination=None,original_logits=None):
    """Width-two depth-three search; batch score queries, bounded proposals.

    score accepts function rows and returns logits without changing parameters.
    It is never called by random-selection ablation or candidate-only preparation.
    All visited candidates remain available for worst-case and multi-view metrics.
    """
    import random
    from .syntax import surface_preserving_candidates,resolve_language,SURFACE_RULES
    if query_limit not in (12,48) or bank not in ('train','heldout'):raise ValueError('unsupported surface search budget')
    states=[]
    for row in records:
        rng=random.Random(surface_seed(seed,epoch,row['sample_key'],purpose))
        source=row['raw_source'];ids=tuple(tokenizer(source,add_special_tokens=False,truncation=True,max_length=source_budget)['input_ids'])
        states.append(dict(row=row,rng=rng,beams=[dict(raw_source=source,units=[],added_comment_tokens=0,used_bindings=[])],
            seen_sources={source},seen_inputs={ids},visited=[],reasons=Counter(),original_logit=None,original_queries=0))
    per_depth=query_limit//3
    for depth in range(3):
        pending=[]
        for index,state in enumerate(states):
            row=state['row'];draft=[]
            for parent in state['beams']:
                allowed=([required_combination[depth]] if required_combination and depth<len(required_combination)
                         else [] if required_combination else kinds)
                if allowed==[]:continue
                proposed,reasons=surface_preserving_candidates(parent['raw_source'],
                    resolve_language(parent['raw_source'],row.get('language','c'),row.get('file_name','')),
                    rng=state['rng'],materials=materials,tokenizer=tokenizer,bank=bank,kinds=allowed,
                    used_bindings=parent['used_bindings'],comment_tokens=parent['added_comment_tokens'],
                    limit=per_depth//len(state['beams']),proposal_limit=24)
                state['reasons'].update(reasons)
                legal,reasons=visible_semantic_candidates(parent['raw_source'],row.get('language','c'),tokenizer,source_budget,
                    file_name=row.get('file_name',''),proposals=proposed,reference_source=row['raw_source'])
                state['reasons'].update(reasons)
                for unit in legal:
                    path=parent['units']+[{k:v for k,v in unit.items() if k not in ('raw_source',)}]
                    types=Counter(u['kind'] for u in path)
                    if bank=='train' and required_combination is None and any(not (Counter(combo)-types) for combo in SURFACE_RULES['heldout_combinations']):
                        state['reasons']['heldout_operator_combination']+=1;continue
                    changed=unit['raw_source'];ids=tuple(tokenizer(changed,add_special_tokens=False,truncation=True,max_length=source_budget)['input_ids'])
                    if changed in state['seen_sources'] or ids in state['seen_inputs']:
                        state['reasons']['duplicate_source_or_input']+=1;continue
                    state['seen_sources'].add(changed);state['seen_inputs'].add(ids)
                    candidate=dict(raw_source=changed,units=path,added_comment_tokens=parent['added_comment_tokens']+unit['added_comment_tokens'],
                        used_bindings=parent['used_bindings']+([unit['binding_token_index']] if unit['binding_token_index'] is not None else []),
                        strict_semantic=all(u['strict_semantic'] for u in path))
                    draft.append(candidate)
            for candidate in draft[:per_depth]:pending.append((index,candidate))
        if not pending:break
        if score is not None and not random_selection:
            missing=[i for i,_ in pending if states[i]['original_logit'] is None]
            missing=list(dict.fromkeys(missing))
            if original_logits is not None:
                for i in missing:states[i]['original_logit']=float(original_logits[states[i]['row']['sample_key']])
            elif missing:
                values=score([states[i]['row'] for i in missing])
                for i,z in zip(missing,values):states[i]['original_logit']=float(z);states[i]['original_queries']=1
            values=score([dict(states[i]['row'],raw_source=c['raw_source']) for i,c in pending])
            if len(values)!=len(pending):raise ValueError('candidate score cardinality mismatch')
        else:values=[None]*len(pending)
        next_beams=defaultdict(list)
        for (i,candidate),z in zip(pending,values):
            state=states[i]
            if z is not None:
                if not np.isfinite(z):raise ValueError('nonfinite candidate score')
                candidate.update(logit=float(z),loss=float(np.logaddexp(0,float(z))-state['row']['label']*float(z)))
            state['visited'].append(candidate);next_beams[i].append(candidate)
        for i,state in enumerate(states):
            children=next_beams[i]
            if random_selection or score is None:
                state['rng'].shuffle(children);state['beams']=children[:2]
            else:state['beams']=sorted(children,key=lambda c:c['loss'],reverse=True)[:2]
    result=[]
    for state in states:
        visited=state['visited']
        if required_combination:visited=[c for c in visited if len(c['units'])==len(required_combination)]
        selected=(state['rng'].choice(visited) if random_selection or score is None else max(visited,key=lambda c:c['loss'])) if visited else None
        base=state['original_logit'];y=state['row']['label'] if base is not None else None
        result.append(dict(sample_key=state['row']['sample_key'],selected=selected,visited=state['visited'],
            candidate_queries=len(state['visited']) if score is not None and not random_selection else 0,
            original_queries=state['original_queries'],original_logit=base,
            loss_delta=selected['loss']-float(np.logaddexp(0,base)-y*base) if selected is not None and base is not None else None,
            reasons=dict(state['reasons'])))
        if len(state['visited'])>query_limit:raise ValueError('surface candidate query budget exceeded')
    return result
