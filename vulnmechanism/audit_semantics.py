from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


PAIRED_DATASETS = {"cleanvul", "sven"}
_OCCURRENCES = re.compile(r"\s+occurrences=\d+")


def _records(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _mechanism_detail(detail: str) -> str:
    """Normalize a candidate for semantic comparison.

    Occurrence count is intentionally excluded: repeated uses of the same
    mechanism are useful audit metadata but do not constitute a different
    source/relation/sink mechanism.
    """
    return " ".join(_OCCURRENCES.sub("", detail).split())


def _model_context(record: dict[str, object]) -> str:
    context = record.get("mechanism_context")
    if not isinstance(context, str):
        raise ValueError(f"{record.get('sample_key')}: mechanism_context is missing or malformed")
    return "\n".join(
        _OCCURRENCES.sub("", line).strip()
        for line in context.splitlines()
        if line.strip()
    )


def _candidates(record: dict[str, object]) -> dict[str, tuple[str, str, str]]:
    items = record.get("mechanism_items")
    if not isinstance(items, list):
        raise ValueError(f"{record.get('sample_key')}: mechanism_items is missing or malformed")
    result: dict[str, tuple[str, str, str]] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("category") != "MECHANISM_CANDIDATE":
            continue
        kind = item.get("kind")
        key = item.get("key")
        state = item.get("state", "")
        detail = item.get("detail", "")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{record.get('sample_key')}: candidate kind is malformed")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{record.get('sample_key')}: candidate key is malformed")
        if not isinstance(state, str):
            raise ValueError(f"{record.get('sample_key')}: candidate state is malformed")
        if not isinstance(detail, str):
            raise ValueError(f"{record.get('sample_key')}: candidate detail is malformed")
        result[key] = (kind, state, _mechanism_detail(detail))
    return result


def audit_mechanism_fidelity(
    dataset_path: str | Path,
    *,
    dataset: str,
) -> dict[str, object]:
    if dataset not in PAIRED_DATASETS:
        raise ValueError(f"dataset must be one of {sorted(PAIRED_DATASETS)}")
    rows = [row for row in _records(Path(dataset_path)) if row.get("dataset") == dataset]
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{row.get('sample_key')}: paired record requires pair_id")
        groups[pair_id].append(row)

    complete = 0
    incomplete = 0
    before_with_candidate = 0
    after_with_candidate = 0
    any_removed = 0
    any_added = 0
    any_state_changed = 0
    any_detail_changed = 0
    any_model_context_changed = 0
    any_source_changed = 0
    unchanged = 0
    removed_kinds = Counter()
    added_kinds = Counter()
    state_changes = Counter()
    detail_change_kinds = Counter()
    persisted_kinds = Counter()
    candidate_count_before = Counter()
    candidate_count_after = Counter()
    by_split = Counter()
    detail_changes: list[dict[str, str]] = []

    for pair_id, pair_rows in sorted(groups.items()):
        if len(pair_rows) != 2 or {row.get("label") for row in pair_rows} != {0, 1}:
            incomplete += 1
            continue
        vulnerable = next(row for row in pair_rows if row["label"] == 1)
        fixed = next(row for row in pair_rows if row["label"] == 0)
        if vulnerable.get("split") != fixed.get("split"):
            raise ValueError(f"{pair_id}: pair split mismatch")

        complete += 1
        by_split[str(vulnerable.get("split"))] += 1
        before = _candidates(vulnerable)
        after = _candidates(fixed)
        before_with_candidate += bool(before)
        after_with_candidate += bool(after)
        candidate_count_before[str(len(before))] += 1
        candidate_count_after[str(len(after))] += 1

        before_keys = set(before)
        after_keys = set(after)
        removed = before_keys - after_keys
        added = after_keys - before_keys
        persisted = before_keys & after_keys
        state_changed = {
            key for key in persisted
            if before[key][1] != after[key][1]
        }
        detail_changed = {
            key for key in persisted
            if before[key][2] != after[key][2]
        }

        source_changed = vulnerable.get("raw_source") != fixed.get("raw_source")
        model_context_changed = _model_context(vulnerable) != _model_context(fixed)

        any_removed += bool(removed)
        any_added += bool(added)
        any_state_changed += bool(state_changed)
        any_detail_changed += bool(detail_changed)
        any_source_changed += bool(source_changed)
        any_model_context_changed += bool(model_context_changed)
        unchanged += not removed and not added and not state_changed and not detail_changed

        for key in removed:
            removed_kinds[before[key][0]] += 1
        for key in added:
            added_kinds[after[key][0]] += 1
        for key in persisted:
            persisted_kinds[before[key][0]] += 1
        for key in state_changed:
            kind = before[key][0]
            state_changes[f"{kind}:{before[key][1]}->{after[key][1]}"] += 1
        for key in detail_changed:
            kind = before[key][0]
            detail_change_kinds[kind] += 1
            if len(detail_changes) < 100:
                detail_changes.append({
                    "pair_id": pair_id,
                    "key": key,
                    "kind": kind,
                    "before": before[key][2],
                    "after": after[key][2],
                })

    def rate(value: int) -> float | None:
        return value / complete if complete else None

    return {
        "dataset": dataset,
        "complete_build_pairs": complete,
        "incomplete_build_pairs": incomplete,
        "pairs_by_split": dict(sorted(by_split.items())),
        "pairs_with_source_change": any_source_changed,
        "source_change_rate": rate(any_source_changed),
        "before_with_mechanism_candidate": before_with_candidate,
        "after_with_mechanism_candidate": after_with_candidate,
        "before_candidate_coverage": rate(before_with_candidate),
        "after_candidate_coverage": rate(after_with_candidate),
        "pairs_with_candidate_removed": any_removed,
        "candidate_removal_rate": rate(any_removed),
        "pairs_with_candidate_added": any_added,
        "candidate_addition_rate": rate(any_added),
        "pairs_with_candidate_state_changed": any_state_changed,
        "candidate_state_change_rate": rate(any_state_changed),
        "pairs_with_candidate_detail_changed": any_detail_changed,
        "candidate_detail_change_rate": rate(any_detail_changed),
        "pairs_with_model_context_changed": any_model_context_changed,
        "model_context_change_rate": rate(any_model_context_changed),
        "pairs_with_unchanged_mechanism_candidates": unchanged,
        "unchanged_mechanism_rate": rate(unchanged),
        "candidate_removed_by_kind": dict(sorted(removed_kinds.items())),
        "candidate_added_by_kind": dict(sorted(added_kinds.items())),
        "candidate_persisted_by_kind": dict(sorted(persisted_kinds.items())),
        "candidate_state_changes": dict(sorted(state_changes.items())),
        "candidate_detail_changed_by_kind": dict(sorted(detail_change_kinds.items())),
        "candidate_detail_changes": detail_changes,
        "candidate_count_distribution_before": dict(
            sorted(candidate_count_before.items(), key=lambda x: int(x[0]))
        ),
        "candidate_count_distribution_after": dict(
            sorted(candidate_count_after.items(), key=lambda x: int(x[0]))
        ),
        "interpretation": (
            "This diagnostic separates candidate addition/removal, constraint-state changes, and concrete "
            "mechanism-detail changes. The model-context metric compares the actual mechanism text given to "
            "the Code LLM while ignoring occurrence counts. These are fidelity diagnostics, not causal ground truth."
        ),
    }


def audit_semantic_fidelity(dataset_path: str | Path, *, dataset: str) -> dict[str, object]:
    return audit_mechanism_fidelity(dataset_path, dataset=dataset)


def audit_repair_availability(data_root, cohort_path, output_dir):
    """CPU inventory and deterministic train/valid review sample, never auto-label effects."""
    import csv
    import difflib
    import hashlib
    import random
    from urllib.parse import urlparse
    from .syntax import parse_function
    from .cfg_data import atomic_json

    output=Path(output_dir)
    if output.exists():raise FileExistsError(output)
    base=Path(data_root)
    key=lambda source:hashlib.sha256(source.strip().encode()).digest()
    membership=defaultdict(set); cohort=Counter()
    for row in _records(Path(cohort_path)):
        membership[key(row['raw_source'])].add((row['dataset'],row['split'],row['sample_key']))
        cohort[row['dataset']+':'+row['split']]+=1
    report=dict(cohort=dict(cohort),datasets={},excluded=Counter(),sampling='seed42; up to 4 distinct projects per dataset/split; not prevalence-weighted',test_semantics_used=False)
    pairs=[]
    def admit(row):
        before,after=row['before'],row['after']
        if not before.strip() or not after.strip() or before.strip()==after.strip():
            report['excluded'][row['dataset']+':empty_or_unchanged']+=1;return
        members=membership[key(before)]|membership[key(after)]
        row['cohort_members']=[dict(dataset=d,split=s,sample_key=k) for d,s,k in sorted(members)]
        if any(s not in ('train','valid') for d,s,k in members):
            report['excluded'][row['dataset']+':heldout_source_overlap']+=1;return
        if any(s!=row['split'] for d,s,k in members):
            report['excluded'][row['dataset']+':cohort_split_conflict']+=1;return
        pairs.append(row)
    # Official paired files are native alternating vulnerable/fixed records.
    for split in ('train','valid'):
        path=base/'PrimeVul_v0.1'/f'primevul_{split}_paired.jsonl'
        rows=_records(path); counts=Counter(rows=len(rows)); projects=set()
        if len(rows)%2:raise ValueError('native PrimeVul pair file has odd row count')
        for a,b in zip(rows[::2],rows[1::2]):
            if (a['target'],b['target'])!=(1,0) or any(a[k]!=b[k] for k in ('project','commit_id')):
                counts['native_pair_metadata_mismatch']+=1
                continue
            names=[]
            for r in (a,b):
                name=None
                for lang in ('c','cpp'):
                    for prefix in ('','int '):
                        try:name=parse_function(prefix+r['func'],lang).name;break
                        except ValueError:pass
                    if name:break
                names.append(name)
            if not names[0] or names[0]!=names[1]:counts['unresolved_function_identity']+=1;continue
            counts['recoverable_pairs']+=1;projects.add(a['project'])
            admit(dict(dataset='PrimeVul',sample_key=f"primevul_pair:{a['idx']}:{b['idx']}",split=split,
                       project=a['project'],function_name=names[0],before=a['func'],after=b['func'],
                       commit=a['commit_id'],commit_url=a.get('commit_url'),cve=a.get('cve'),
                       commit_message=a.get('commit_message'),cve_description=a.get('cve_desc'),
                       file_name=b.get('file_name'),provenance=str(path)))
        report['datasets']['PrimeVul:'+split]=dict(counts,projects=len(projects))
    # Keep the existing MegaVul project split; do not regenerate it.
    mega=_records(Path('data/pairs_all.jsonl'))
    for split in ('train','valid'):
        selected=[r for r in mega if r['split']==split]
        report['datasets']['MegaVul:'+split]=dict(recoverable_pairs=len(selected),projects=len({r['project'] for r in selected}))
        for r in selected:admit(dict(dataset='MegaVul',sample_key=r['sample_key'],split=split,project=r['project'],
            function_name=r['function_name'],before=r['func_before'],after=r['func_after'],cwe=r['cwe'],
            provenance='data/pairs_all.jsonl',raw_row_index=int(r['sample_key'].rsplit('_',1)[1])))
    del mega
    # CSV is_test is a test-code flag, not a train/valid/test split. Only exact
    # source membership in the already split cohort supplies CleanVul admission.
    csv.field_size_limit(100000000)
    clean_counts=Counter();clean_projects=set();seen=set()
    for path in sorted((base/'cleanvul').glob('vulnerability_score_*.csv')):
        with path.open() as stream:
            for i,r in enumerate(csv.DictReader(stream)):
                clean_counts['rows']+=1
                if r['extension'].lower().lstrip('.') not in ('c','cpp','cc','cxx','h','hpp'):continue
                clean_counts['c_cpp_rows']+=1
                before,after=r['func_before'],r['func_after'];pair=(key(before),key(after))
                if pair in seen:continue
                seen.add(pair)
                if not before.strip() or not after.strip() or before.strip()==after.strip():continue
                clean_counts['recoverable_changed_pairs']+=1
                project=urlparse(r['commit_url']).path.split('/commit/')[0].strip('/')
                clean_projects.add(project)
                members=membership[key(before)]|membership[key(after)]
                splits={s for d,s,k in members if d=='cleanvul'}
                if len(splits)!=1 or next(iter(splits)) not in ('train','valid'):
                    clean_counts['no_unique_existing_train_valid_split']+=1;continue
                split=next(iter(splits))
                admit(dict(dataset='CleanVul',sample_key=f'cleanvul:{path.stem}:{i}',split=split,project=project,
                           before=before,after=after,commit_url=r['commit_url'],commit_message=r['commit_msg'],
                           cve=r['cve_id'],cwe=r['cwe_id'],file_name=r['file_name'],score=r['vulnerability_score'],
                           provenance=str(path),raw_row_index=i))
    report['datasets']['CleanVul:inventory']=dict(clean_counts,projects=len(clean_projects))
    # Source-identical cross-dataset pairs remain one supervision example.
    grouped=defaultdict(list)
    for r in pairs:grouped[key(r['before']),key(r['after'])].append(r)
    usable=[]
    for group in grouped.values():
        if len({r['split'] for r in group})>1:
            report['excluded']['cross_dataset_pair_split_conflict']+=len(group);continue
        chosen=group[0];chosen['duplicate_sources']=[dict(dataset=r['dataset'],sample_key=r['sample_key']) for r in group[1:]]
        usable.append(chosen)
    report['eligible_before_cross_dataset_dedup']=dict(Counter(r['dataset']+':'+r['split'] for r in pairs))
    report['eligible_unique_pairs']=dict(Counter(r['dataset']+':'+r['split'] for r in usable))
    report['eligible_projects']=len({r['project'] for r in usable})
    report['current_primevul_functions_with_pair']=dict(Counter(s for s,k in {(m['split'],m['sample_key']) for r in usable for m in r['cohort_members'] if m['dataset']=='primevul'}))
    selected=[];rng=random.Random(42)
    for dataset in ('PrimeVul','MegaVul','CleanVul'):
        for split in ('train','valid'):
            projects=defaultdict(list)
            for r in usable:
                if r['dataset']==dataset and r['split']==split:projects[r['project']].append(r)
            for project in rng.sample(sorted(projects),min(4,len(projects))):
                selected.append(rng.choice(sorted(projects[project],key=lambda r:r['sample_key'])))
    # Recover metadata for selected MegaVul rows from the existing raw source.
    import ijson
    wanted={r['raw_row_index']:r for r in selected if r['dataset']=='MegaVul'}
    with (base/'megavul'/'megavul.json').open('rb') as stream:
        for i,r in enumerate(ijson.items(stream,'item',use_float=True)):
            if i not in wanted:continue
            row=wanted.pop(i)
            if r.get('func_before')!=row['before'] or r.get('func')!=row['after']:
                # The raw MegaVul schema uses func_after in some exports.
                if r.get('func_before')!=row['before'] or r.get('func_after')!=row['after']:
                    raise ValueError('MegaVul source row mismatch')
            row['raw_metadata']={k:v for k,v in r.items() if k not in ('func','func_before','func_after') and not isinstance(v,(dict,list))}
            if not wanted:break
    if wanted:raise ValueError('missing MegaVul provenance rows')
    output.mkdir(parents=True)
    atomic_json(output/'inventory.json',report)
    with (output/'eligible.jsonl').open('x') as stream:
        for r in usable:stream.write(json.dumps({k:v for k,v in r.items() if k not in ('before','after')})+'\n')
    with (output/'cases.jsonl').open('x') as stream:
        for i,r in enumerate(selected):
            r['review_index']=i
            r['diff']=''.join(difflib.unified_diff(r['before'].splitlines(True),r['after'].splitlines(True),fromfile='before',tofile='after',n=5))
            stream.write(json.dumps(r)+'\n')
    return report


def operation_associations(source, language, graph=None):
    """Exact AST/scoped-name associations, NOT memory safety or alias proofs.

    Native DDG/CDG are separately reported, never promoted to safe/unsafe labels.
    Context updates are lexically earlier writes, not inferred reaching values.
    """
    from .syntax import parser_for, walk
    from .cfg_dependency import lexical_bindings
    from .cfg_alignment import _source_coordinates, _node_span
    from .semantics import _WRITE_APIS, _READ_APIS, _FREE_APIS
    raw = source.encode(); root = parser_for(language).parse(raw).root_node
    nodes = list(walk(root)); bindings, error = lexical_bindings(source, language)
    chars = {0: 0}; offset = 0
    for i, c in enumerate(source, 1):
        offset += len(c.encode()); chars[offset] = i
    def span(n): return (chars[n.start_byte], chars[n.end_byte])
    binding_cache = {}
    def bs(n):
        if n.id not in binding_cache:
            binding_cache[n.id] = {bindings[span(x)]['binding'] for x in walk(n)
                if x.type == 'identifier' and span(x) in bindings and bindings[span(x)]['binding']}
        return binding_cache[n.id]
    declarations = {}
    for n in nodes:
        if n.type in ('declaration', 'parameter_declaration'):
            for x in walk(n):
                b = bindings.get(span(x), {})
                if x.type == 'identifier' and b.get('binding') and tuple(b['declaration_span']) == span(x):
                    declarations[b['binding']] = n
    # Native endpoints must exactly agree with cached source coordinates.
    native = {}; at_span = defaultdict(list); incoming = defaultdict(list)
    if graph:
        coordinates = _source_coordinates(source)
        for n in graph['nodes']:
            s, reason = _node_span(source, dict(n['properties'], code=n['code']), *coordinates)
            if s and not reason: native[n['id']] = s; at_span[s].append(n['id'])
        for e in graph['edges']:
            if e['kind'] in ('DDG', 'CDG'): incoming[e['target']].append(e)
    from bisect import bisect_left
    ordered_spans = sorted(at_span)
    ordered_bindings = sorted(bindings)
    native_bindings = {}
    def bindings_within(s):
        if s not in native_bindings:
            begin=bisect_left(ordered_bindings,(s[0],-1));end=bisect_left(ordered_bindings,(s[1],-1))
            native_bindings[s]={bindings[k]['binding'] for k in ordered_bindings[begin:end]
                                if k[1]<=s[1] and bindings[k]['binding']}
        return native_bindings[s]
    writes=defaultdict(list)
    for x in nodes:
        lhs=(x.child_by_field_name('left') if x.type=='assignment_expression' else
             x.child_by_field_name('argument') if x.type=='update_expression' else None)
        if lhs is not None and lhs.type=='identifier':
            for binding in bs(lhs): writes[binding].append(x)
    candidates = []
    for n in nodes:
        arg = n.child_by_field_name('argument')
        if n.type == 'subscript_expression':
            candidates.append((n, arg, 'Bounds', 'subscript'))
        elif n.type == 'pointer_expression' and n.text.lstrip().startswith(b'*'):
            candidates.append((n, arg, 'Pointer', 'indirection'))
        elif n.type == 'field_expression' and b'->' in n.text:
            candidates.append((n, arg, 'Pointer', 'field_indirection'))
        elif n.type == 'call_expression':
            fn = n.child_by_field_name('function'); args = n.child_by_field_name('arguments')
            if fn is None or fn.type != 'identifier' or args is None: continue
            name = fn.text.decode(); args = args.named_children
            for direction, apis in (('write', _WRITE_APIS), ('read', _READ_APIS)):
                if name in apis and len(args) > apis[name][0]:
                    candidates.append((n, args[apis[name][0]], 'Bounds', name+':'+direction))
            if name in _FREE_APIS and len(args) == 1:
                candidates.append((n, args[0], 'Lifetime', name))
    out = []
    for n, arg, family, kind in candidates:
        item = dict(family=family, kind=kind, span=span(n), code=n.text.decode(),
                    object=arg.text.decode() if arg else None, binding=None, context=[],
                    status='Unknown', external_context='Unknown', native_relations=[])
        out.append(item)
        if error: item['reason'] = error; continue
        if any(x.type.startswith('preproc_') for x in nodes):
            item['reason'] = 'unresolved_preprocessor'; continue
        if arg is None or arg.type != 'identifier':
            item['reason'] = 'compound_object_or_alias'; continue
        parent=n.parent
        while parent is not None and parent.type not in ('expression_statement','return_statement','declaration'):
            if parent.type in ('sizeof_expression','alignof_expression') or (parent.type=='pointer_expression' and parent.text.lstrip().startswith(b'&')):
                item['reason']='unevaluated_or_address_only'; break
            parent=parent.parent
        if 'reason' in item:continue
        b = bindings.get(span(arg), {})
        if not b.get('binding') or b['binding'] not in declarations:
            item['reason'] = 'unresolved_member_global_or_declaration'; continue
        item.update(status='ConfirmedAssociation', binding=b['binding'],
                    object_span=span(arg), origin='parameter' if b['parameter'] else 'local',
                    external_context='pointee_extent_lifetime_and_callee_contract_not_established')
        relevant = set(bs(n)); decl = declarations[b['binding']]
        # One-hop syntactic data sources are associations, not value propagation.
        updates={x.id:x for binding in relevant for x in writes[binding] if x.end_byte<=n.start_byte}
        for x in updates.values(): relevant.update(bs(x))
        relevant.update(bs(decl))
        contexts = [(decl, 'object_declaration')]
        for binding in sorted(relevant - {b['binding']}):
            if binding in declarations: contexts.append((declarations[binding], 'operand_declaration'))
        # Enclosing branch records polarity; a source predicate is not a proven guard.
        child = n; parent = n.parent
        while parent is not None:
            if parent.type in ('if_statement', 'while_statement', 'for_statement', 'switch_statement'):
                cond = parent.child_by_field_name('condition')
                if cond is not None and child != cond and bs(cond) & relevant:
                    branch = 'else' if parent.child_by_field_name('alternative') == child else 'body'
                    contexts.append((cond, parent.type+':'+branch))
            child, parent = parent, parent.parent
        for x in updates.values():
            contexts.append((x, 'lexically_earlier_write_not_reaching_value'))
            child=x;parent=x.parent
            while parent is not None:
                if parent.type in ('if_statement','for_statement','while_statement'):
                    cond=parent.child_by_field_name('condition')
                    if cond is not None and child!=cond and child!=parent.child_by_field_name('initializer'):
                        branch='else' if parent.child_by_field_name('alternative')==child else 'body_or_update'
                        contexts.append((cond,'write_path:'+parent.type+':'+branch))
                child,parent=parent,parent.parent
        unique = {}
        for x, role in contexts:
            unique[(span(x), role)] = dict(span=span(x), role=role, code=x.text.decode())
        item['context'] = sorted(unique.values(), key=lambda x: (x['span'], x['role']))
        stmt = n
        while stmt.parent is not None and stmt.type not in ('expression_statement', 'return_statement', 'declaration'):
            if stmt.parent.type in ('compound_statement', 'function_definition','if_statement','for_statement','while_statement','switch_statement'): break
            stmt = stmt.parent
        item['local_span'] = span(stmt)
        begin=bisect_left(ordered_spans,(span(n)[0],-1));end=bisect_left(ordered_spans,(span(n)[1],-1))
        for s in ordered_spans[begin:end]:
            if s[1] > span(n)[1]: continue
            for ident in at_span[s]:
                for e in incoming[ident]:
                    upstream = native.get(e['source'])
                    if upstream is None: continue
                    left = bindings_within(upstream); right = bindings_within(s)
                    item['native_relations'].append(dict(kind=e['kind'], source_span=upstream,
                        target_span=s, same_scoped_binding=bool(left & right), properties=e.get('properties', {})))
    return out


def operation_exchange(operations):
    """First nontrivial within-function same-kind/origin exchange; labels unused."""
    for i, a in enumerate(operations):
        if a['status'] != 'ConfirmedAssociation' or not a.get('visible'): continue
        if not any(x['role'] not in ('object_declaration', 'operand_declaration') for x in a['context']): continue
        for j in range(i+1, len(operations)):
            b = operations[j]
            if b['status'] != 'ConfirmedAssociation' or not b.get('visible'): continue
            if (a['kind'],a['origin']) != (b['kind'],b['origin']): continue
            if not any(x['role'] not in ('object_declaration', 'operand_declaration') for x in b['context']): continue
            if a['context_text'] == b['context_text']: continue
            if a['local_text'] == b['local_text']: continue
            return [i,j]
    return []


def audit_operation_context(dataset, graphs, native_dir, output_dir, model_path):
    """Train/valid inventory and prespecified matched-cohort admission, CPU only."""
    import math
    from transformers import AutoTokenizer
    from .cfg_data import read_records, iter_jsonl, identity, atomic_json
    from .syntax import parser_for, walk
    output = Path(output_dir)
    if output.exists(): raise FileExistsError(output)
    rows = read_records(dataset, 'primevul', splits=('train','valid'))
    by_key = {r['sample_key']:r for r in rows}; metadata = {}
    for split in ('train','valid'):
        for r in iter_jsonl(Path(native_dir)/f'primevul_{split}.jsonl'):
            key = 'primevul:'+str(r['idx'])
            if key not in by_key: continue
            original = by_key[key]
            if r['func'] != original['raw_source'] or r['target'] != original['label'] or split != original['split']:
                raise ValueError('native provenance mismatch: '+key)
            metadata[key] = {k:r.get(k) for k in ('project','commit_id','file_name')}
    if set(metadata) != set(by_key): raise ValueError('native provenance missing')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    output.mkdir(parents=True)
    report = dict(test_used=False, seed=42, dataset=str(dataset), graphs=str(graphs),
                  native_dir=str(native_dir), model_path=str(model_path), source_max_length=2048,
                  functions=Counter(), operations=Counter(),
                  reasons=Counter(), family_functions={}, native=Counter(), matching={},
                  protocol='exact project/kind/origin; log2 token length/count bins; first eligible within-function exchange; no safety labels',
                  probe_minimum_pairs={'train':100,'valid':30})
    summaries=[]; candidate_groups=defaultdict(lambda:defaultdict(list)); seen=set(); normalized=defaultdict(list)
    with (output/'operations.jsonl').open('x') as out:
        for cached in iter_jsonl(graphs):
            key=cached['sample_key']
            if key not in by_key: continue
            row=by_key[key]; source=row['raw_source']; split=row['split']; label=row['label']
            if key in seen or any(cached[k]!=v for k,v in identity(row).items()): raise ValueError('graph identity mismatch')
            seen.add(key); ops=operation_associations(source,row['resolved_language'],cached['graph'])
            tok=tokenizer(source,add_special_tokens=False,return_offsets_mapping=True)
            offsets=tok['offset_mapping'][:2048]; visible_end=offsets[-1][1] if offsets else 0
            parsed=parser_for(row['resolved_language']).parse(source.encode()).root_node
            from .semantics import _WRITE_APIS, _READ_APIS, _FREE_APIS, _ALLOC_APIS
            names={n.text.decode() for n in walk(parsed) if n.type in ('identifier','field_identifier','type_identifier')}
            names-=set(_WRITE_APIS)|set(_READ_APIS)|_FREE_APIS|_ALLOC_APIS
            # Per-operation canonical names remove project/variable spelling. The
            # target is object in both correct and donor contexts (no name mismatch cue).
            def normalized_text(text, target):
                ids={target:'object'}; count=0
                def replace(m):
                    nonlocal count
                    name=m.group()
                    if name not in names:return name
                    if name not in ids:ids[name]='v'+str(count);count+=1
                    return ids[name]
                return re.sub(r'\b[A-Za-z_]\w*\b',replace,text)
            for op in ops:
                op['visible']=bool(op.get('local_span') and max([op['span'][1],op['local_span'][1]]+[x['span'][1] for x in op['context']])<=visible_end)
                if op['status']=='ConfirmedAssociation':
                    op['local_text']=normalized_text(source[slice(*op['local_span'])],op['object'])
                    op['context_text']=normalized_text('\n'.join(x['role']+': '+x['code'] for x in op['context']),op['object'])
                prefix=f'{split}:{label}:{op["family"]}'
                report['operations'][prefix+':'+op['status']]+=1
                report['operations'][prefix+':visible']+=int(op['visible'])
                report['reasons'][op.get('reason','binding_confirmed_safety_unknown')]+=1
                for edge in op['native_relations']:
                    report['native'][edge['kind']+':'+str(edge['same_scoped_binding'])]+=1
            exchange=operation_exchange(ops)
            record=dict(sample_key=key,split=split,label=label,**metadata[key],tokens=len(tok['input_ids']),
                        operations=ops,exchange=exchange,raw_source=source)
            out.write(json.dumps(record,ensure_ascii=False)+'\n')
            report['functions'][f'{split}:{label}:all']+=1
            for family in ('Bounds','Pointer','Lifetime'):
                family_ops=[o for o in ops if o['family']==family]
                if family_ops:
                    group=(split,metadata[key]['project'],family,int(math.log2(max(1,len(tok['input_ids'])))),int(math.log2(len(family_ops))))
                    candidate_groups[group][label].append(key)
                for status,yes in [('candidate',bool(family_ops)),('bound',any(o['status']=='ConfirmedAssociation' for o in family_ops))]:
                    name=f'{split}:{label}:{family}:{status}'
                    report['family_functions'][name]=report['family_functions'].get(name,0)+int(yes)
            clean=re.sub(r'/\*.*?\*/|//[^\n]*','',source,flags=re.S)
            norm=' '.join(re.findall(r'[A-Za-z_]\w*|\d+|[^\s]',normalized_text(clean,'__no_target__')))
            normalized[norm].append(key)
            if exchange:
                first=ops[exchange[0]]
                summaries.append(dict(sample_key=key,split=split,label=label,**metadata[key],tokens=record['tokens'],
                    count=len(ops),family=first['family'],kind=first['kind'],origin=first['origin']))
                report['functions'][f'{split}:{label}:exchange']+=1
    if seen!=set(by_key):raise ValueError('missing graph members')
    leaked=set(); overlaps=[]
    for keys in normalized.values():
        if len({by_key[k]['split'] for k in keys})>1: leaked.update(keys);overlaps.append(keys)
    # Audit source/commit overlap without moving or relabeling any original member.
    provenance=defaultdict(list)
    for key,m in metadata.items():provenance[(m['project'],m['commit_id'])].append(key)
    commit_overlap=[keys for keys in provenance.values() if len({by_key[k]['split'] for k in keys})>1]
    report['leakage']=dict(alpha_normalized_cross_split_groups=overlaps,shared_commit_groups=commit_overlap,
        projects={s:sorted({m['project'] for k,m in metadata.items() if by_key[k]['split']==s}) for s in ('train','valid')})
    # Train-fitted lexical near-clone screen; report candidates, never infer safety.
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts={key:text for text,keys in normalized.items() for key in keys}
    tr=sorted(k for k in by_key if by_key[k]['split']=='train');va=sorted(k for k in by_key if by_key[k]['split']=='valid')
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(5,5),dtype=__import__('numpy').float32)
    xt=vectorizer.fit_transform([texts[k] for k in tr]);xv=vectorizer.transform([texts[k] for k in va])
    similarity=(xv@xt.T).tocsr();near=[]
    for i,k in enumerate(va):
        row=similarity.getrow(i)
        for j,s in zip(row.indices,row.data):
            if s>=.95:near.append(dict(train=tr[j],valid=k,cosine=float(s)))
    report['leakage']['near_clone_candidates']=near
    report['candidate_matches']={s:{f:sum(min(len(labels[0]),len(labels[1])) for g,labels in candidate_groups.items() if g[0]==s and g[2]==f)
                                   for f in ('Bounds','Pointer','Lifetime')} for s in ('train','valid')}
    # Keep the original cohort intact. No result-dependent matching search or
    # removal of hard examples: shared sources in the matched subset block admission.
    groups=defaultdict(lambda:defaultdict(list))
    for r in summaries:
        key=(r['split'],r['project'],r['kind'],r['origin'],int(math.log2(max(1,r['tokens']))),int(math.log2(max(1,r['count']))))
        groups[key][r['label']].append(r)
    matches=[]
    for group, labels in sorted(groups.items()):
        for a,b in zip(sorted(labels[0],key=lambda r:r['sample_key']),sorted(labels[1],key=lambda r:r['sample_key'])):
            matches.append(dict(split=group[0],project=group[1],kind=group[2],keys=[a['sample_key'],b['sample_key']],
                                tokens=[a['tokens'],b['tokens']],counts=[a['count'],b['count']],family=a['family']))
    report['matching']=dict(pairs=dict(Counter(m['split'] for m in matches)),
        by_family=dict(Counter(m['split']+':'+m['family'] for m in matches)),
        projects={s:sorted({m['project'] for m in matches if m['split']==s}) for s in ('train','valid')},
        normalized_overlap_matched_members=sorted({k for m in matches for k in m['keys']} & leaked))
    selected={k for m in matches for k in m['keys']}
    report['matching']['near_clone_candidates']=[r for r in near if r['train'] in selected and r['valid'] in selected]
    report['matching']['shared_commit_groups']=[sorted(set(keys)&selected) for keys in commit_overlap
        if len({by_key[k]['split'] for k in keys if k in selected})>1]
    report['ready']=all(report['matching']['pairs'].get(s,0)>=n for s,n in report['probe_minimum_pairs'].items()) and not any(
        report['matching'][k] for k in ('normalized_overlap_matched_members','near_clone_candidates','shared_commit_groups'))
    atomic_json(output/'coverage.json',report);atomic_json(output/'matches.json',matches)
    return report


def checked_guard_interventions(record, language):
    """Recheck cached associations, not a new operation detector or safety oracle.

    Admit side-effect-free if predicates on a primitive pointer or integer
    operand, unchanged between check and operation in a sequential execution.
    Find an explicit valuation distinguishing predicates after operand-role
    normalization. No witness is interpreted as a feasible vulnerable execution.
    """
    import itertools
    from .syntax import parser_for, walk
    from .cfg_dependency import lexical_bindings
    source=record['raw_source'];raw=source.encode()
    bindings,error=lexical_bindings(source,language)
    if error:return [], [], Counter(parse_unknown=len(record['operations']))
    root=parser_for(language).parse(raw).root_node;nodes=list(walk(root))
    chars={0:0};offset=0
    for i,c in enumerate(source,1):offset+=len(c.encode());chars[offset]=i
    span=lambda n:(chars[n.start_byte],chars[n.end_byte])
    lookup={span(n):n for n in nodes if n.is_named}
    counts=Counter();eligible=[]
    def evaluate(expr,values):
        op=expr[0]
        if op=='var':return values[expr[1]]
        if op=='int':return expr[1]
        if op=='not':return not evaluate(expr[1],values)
        a,b=evaluate(expr[1],values),evaluate(expr[2],values)
        return {'<':lambda:a<b,'<=':lambda:a<=b,'>':lambda:a>b,'>=':lambda:a>=b,
                '==':lambda:a==b,'!=':lambda:a!=b,'&&':lambda:bool(a) and bool(b),
                '||':lambda:bool(a) or bool(b)}[op]()
    for index,operation in enumerate(record['operations']):
        if operation['status']!='ConfirmedAssociation' or not operation['visible']:
            counts['unbound_or_outside_window']+=1;continue
        if operation['family']=='Lifetime':
            counts['lifetime_ownership_state_unknown']+=1;continue
        node=lookup.get(tuple(operation['span']))
        if node is None:counts['operation_ast_unknown']+=1;continue
        target=operation['binding'];roles={};types={};names={}
        for s,b in bindings.items():
            if operation['span'][0]<=s[0] and s[1]<=operation['span'][1] and b['binding']:
                if operation['family']=='Pointer' and b['binding']==target:
                    declaration=next(x['code'] for x in operation['context'] if x['role']=='object_declaration')
                    declaration_node=next((x for x in nodes if x.type in ('parameter_declaration','declaration') and
                        span(x)[0]<=b['declaration_span'][0] and b['declaration_span'][1]<=span(x)[1]),None)
                    if declaration_node is not None and any(x.type=='pointer_declarator' and span(x)[0]<=b['declaration_span'][0] and b['declaration_span'][1]<=span(x)[1] for x in walk(declaration_node)) and not re.search(r'\bvolatile\b',declaration):
                        roles[target]='object';types['object']='pointer';names[target]=b['name']
                elif operation['family']=='Bounds' and b.get('integer_type'):
                    roles.setdefault(b['binding'],'operand'+str(len(roles)))
                    types[roles[b['binding']]]=b['integer_type'];names[b['binding']]=b['name']
        if not roles:counts['target_operand_type_unknown']+=1;continue
        guards=[]
        target_roles=set(roles)
        for context in operation['context']:
            if not context['role'].startswith('if_statement:'):continue
            condition=lookup.get(tuple(context['span']))
            if condition is None:continue
            used=set();constants=set();unsupported=False
            def expression(n):
                nonlocal unsupported
                if n.type in ('parenthesized_expression','condition_clause') and len(n.named_children)==1:return expression(n.named_children[0])
                if n.type=='identifier':
                    b=bindings.get(span(n),{});key=b.get('binding')
                    if key and key not in roles and b.get('integer_type'):
                        roles[key]='reference'+str(sum(v.startswith('reference') for v in roles.values()))
                        types[roles[key]]=b['integer_type']
                    if key not in roles:unsupported=True;return None
                    used.add(key);return ('var',roles[key])
                if n.type in ('number_literal','null') and n.text in (b'0',b'NULL',b'nullptr'):
                    constants.add(0);return ('int',0)
                if n.type=='number_literal' and re.fullmatch(rb'[0-9]+',n.text) and int(n.text)<=32767:
                    constants.add(int(n.text));return ('int',int(n.text))
                if n.type=='unary_expression' and n.child_by_field_name('operator').text==b'!':
                    return ('not',expression(n.child_by_field_name('argument')))
                if n.type=='binary_expression':
                    operator=n.child_by_field_name('operator').text.decode()
                    if operator in ('<','<=','>','>=','==','!=','&&','||'):
                        return (operator,expression(n.child_by_field_name('left')),expression(n.child_by_field_name('right')))
                unsupported=True;return None
            expr=expression(condition)
            if unsupported or not used.intersection(target_roles):counts['guard_expression_or_operand_unknown']+=1;continue
            if operation['family']=='Pointer':
                if constants-{0} or any(x.type=='binary_expression' and x.child_by_field_name('operator').text not in (b'==',b'!=',b'&&',b'||') for x in walk(condition)):
                    counts['pointer_non_nullness_predicate']+=1;continue
            if context['role'].endswith(':else'):expr=('not',expr)
            # No optimistic inference across writes, calls, aliasing writes,
            # increments or loops. This verifies persistence, not global safety.
            between=[n for n in nodes if context['span'][1]<=span(n)[0] and span(n)[1]<=operation['span'][0]]
            if any(n.type in ('call_expression','assignment_expression','update_expression','for_statement','while_statement','do_statement','goto_statement','asm_statement') for n in between):
                counts['intervening_effects_state_unknown']+=1;continue
            ancestor=node.parent;loop_between=False
            while ancestor is not None and span(ancestor)!=span(condition.parent):
                if ancestor.type in ('for_statement','while_statement','do_statement'):loop_between=True
                ancestor=ancestor.parent
            if loop_between:counts['intervening_loop_state_unknown']+=1;continue
            # A plain store through the target does not update its pointer/index
            # before evaluating that target. RHS calls/updates can, and are unknown.
            statement=node
            while statement.parent is not None and statement.type not in ('expression_statement','return_statement','declaration','compound_statement'):
                if statement.parent==condition.parent:break
                statement=statement.parent
            uncertain=False
            for part in walk(statement):
                if part.type=='update_expression':uncertain=True
                if part.type=='call_expression' and not (span(node)[0]<=span(part)[0] and span(part)[1]<=span(node)[1]):uncertain=True
                if part.type=='assignment_expression':
                    left=part.child_by_field_name('left');operator=part.child_by_field_name('operator')
                    target_store=left is not None and operator.text==b'=' and span(left)[0]<=span(node)[0] and span(node)[1]<=span(left)[1] and left.type in ('subscript_expression','pointer_expression','field_expression')
                    if not target_store:uncertain=True
            if uncertain:counts['target_expression_state_unknown']+=1;continue
            guards.append(dict(span=context['span'],code=context['code'],branch=context['role'],expression=expr,
                               types={roles[k]:types[roles[k]] for k in used},constants=sorted(constants),
                               meaning='predicate at operation under sequential primitive semantics; not complete safety'))
        if guards:
            eligible.append(dict(operation_index=index,family=operation['family'],kind=operation['kind'],origin=operation['origin'],guards=guards))
            counts['reliable_guard_operations']+=1
    interventions=[];taken=set()
    for a in eligible:
        if a['operation_index'] in taken:continue
        for b in eligible:
            if b['operation_index']<=a['operation_index'] or b['operation_index'] in taken:continue
            if (a['kind'],a['origin'])!=(b['kind'],b['origin']):continue
            # Swap one guard endpoint, preserving the complete feature multiset.
            found=None
            for ga in a['guards']:
                for gb in b['guards']:
                    if ga['types']!=gb['types'] or ga['expression']==gb['expression']:continue
                    if a['family']=='Pointer':
                        # p && flag versus p && !flag changes a predicate but
                        # preserves the same non-null requirement. It is not a
                        # non-equivalent nullness intervention.
                        if ga['types']!={'object':'pointer'}:continue
                        if any(len({bool(evaluate(g['expression'],{'object':v})) for v in (0,1)})!=2 for g in (ga,gb)):continue
                    keys=sorted(ga['types'])
                    if len(keys)>2:continue
                    values=sorted({0,1,2}|{v+d for v in ga['constants']+gb['constants'] for d in (-1,0,1) if 0<=v+d<=127})
                    domains=[(0,1) if ga['types'][k] in ('pointer','bool','_Bool') else values for k in keys]
                    for vals in itertools.product(*domains):
                        witness=dict(zip(keys,vals));left=bool(evaluate(ga['expression'],witness));right=bool(evaluate(gb['expression'],witness))
                        if left!=right:found=dict(operations=[a['operation_index'],b['operation_index']],guards=[ga,gb],witness=witness,predicate_values=[left,right]);break
                    if found:break
                if found:break
            if found:
                interventions.append(found);taken.update(found['operations']);break
    return eligible,interventions,counts


def audit_binding_interventions(cache_dir,dataset,output_dir):
    """Full original membership; no cross-function positive/negative matching."""
    from .cfg_data import read_records,iter_jsonl,atomic_json
    root=Path(output_dir)
    if root.exists():raise FileExistsError(root)
    originals={r['sample_key']:r for r in read_records(dataset,'primevul',splits=('train','valid'))}
    root.mkdir(parents=True);counts={s:Counter() for s in ('train','valid')};reasons={s:Counter() for s in counts};seen=set()
    with (root/'interventions.jsonl').open('x') as out:
        for r in iter_jsonl(Path(cache_dir)/'operations.jsonl'):
            key=r['sample_key'];original=originals[key]
            if key in seen or any(r[k]!=original[k] for k in ('raw_source','split','label')):raise ValueError('original membership mismatch')
            seen.add(key);s=r['split'];eligible,pairs,why=checked_guard_interventions(r,original['resolved_language'])
            counts[s]['functions']+=1;reasons[s].update(why)
            for op in r['operations']:
                counts[s]['cached_declaration_bound_operations']+=op['status']=='ConfirmedAssociation'
                counts[s]['cached_same_binding_ddg_operations']+=any(e['kind']=='DDG' and e['same_scoped_binding'] for e in op['native_relations'])
                counts[s]['cached_control_associated_operations']+=any(c['role'].startswith(('if_statement:','for_statement:','while_statement:','switch_statement:')) for c in op['context'])
                counts[s]['cached_lexical_update_operations']+=any(c['role']=='lexically_earlier_write_not_reaching_value' for c in op['context'])
            counts[s]['reliable_guard_functions']+=bool(eligible);counts[s]['reliable_guard_operations']+=len(eligible)
            counts[s]['intervention_functions']+=bool(pairs);counts[s]['intervention_pairs']+=len(pairs)
            if pairs:counts[s]['intervention_label_'+str(r['label'])]+=1
            for f in ('Bounds','Pointer','Lifetime'):
                ids={x['operation_index'] for x in eligible if x['family']==f}
                counts[s][f+':guard_functions']+=bool(ids);counts[s][f+':guard_operations']+=len(ids)
                counts[s][f+':intervention_functions']+=any(p['operations'][0] in ids for p in pairs)
            out.write(json.dumps(dict(sample_key=key,split=s,label=r['label'],project=r['project'],eligible=eligible,interventions=pairs))+'\n')
    if seen!=set(originals):raise ValueError('missing original members')
    report=dict(counts=counts,reasons=reasons,test_used=False,dataset=str(dataset),operation_cache=str(cache_dir),
        classification_members='all original 5886 train / 741 valid',
        intervention='within-function same-kind/origin typed guard-edge swaps, concrete predicate non-equivalence witness; no text mutation',
        minimum_independent_intervention_functions=dict(train=50,valid=20,valid_each_label=5),
        ready=counts['train']['intervention_functions']>=50 and counts['valid']['intervention_functions']>=20 and all(counts['valid']['intervention_label_'+str(y)]>=5 for y in (0,1)),
        note='No safety labels. A witness separates primitive predicates, not feasible full-function executions or vulnerability labels. Unknown relations and all original functions are retained.')
    atomic_json(root/'coverage.json',report);return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit CPG-derived vulnerability-mechanism changes across vulnerable/fixed pairs"
    )
    parser.add_argument("--dataset", default="data/function_dataset.jsonl")
    parser.add_argument("--source-dataset", choices=sorted(PAIRED_DATASETS))
    parser.add_argument("--output")
    parser.add_argument("--repair-data-root")
    parser.add_argument("--repair-output-dir")
    parser.add_argument("--operation-output-dir")
    parser.add_argument("--binding-output-dir")
    parser.add_argument("--operation-cache", default="results/operation_context_checked_seed42")
    parser.add_argument("--graphs", default="data/graphs/primevul_cfg.jsonl")
    parser.add_argument("--native-dir", default="/home/PublicData/PHY-data/vul_detect/data/PrimeVul_v0.1")
    parser.add_argument("--model-path", default="/home/phy/models/Qwen2.5-Coder-7B-Instruct")
    args = parser.parse_args()
    if args.binding_output_dir:
        print(json.dumps(audit_binding_interventions(args.operation_cache,args.dataset,args.binding_output_dir),indent=2));return
    if args.operation_output_dir:
        report=audit_operation_context(args.dataset,args.graphs,args.native_dir,args.operation_output_dir,args.model_path)
        print(json.dumps({k:v for k,v in report.items() if k!='leakage'},indent=2));return
    if args.repair_data_root:
        if not args.repair_output_dir:parser.error("--repair-output-dir is required")
        print(json.dumps(audit_repair_availability(args.repair_data_root,args.dataset,args.repair_output_dir),indent=2))
        return
    if not args.source_dataset:parser.error("--source-dataset is required for mechanism fidelity")
    report = audit_mechanism_fidelity(args.dataset, dataset=args.source_dataset)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
