"""Source/patch audit only: no model loading, member changes, or training."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import difflib
import json
import os
from pathlib import Path
import random
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time

from sklearn.metrics import roc_auc_score

from .artifacts import write_json
from .data import modified_lines, read_jsonl, write_jsonl
from .losver_diagnosis import annotations, quantiles, selected_mass, write_csv
from .syntax import (CONTROLS, STATEMENTS, source_lines, parse_source, edit_hunks,
                     statement_units, parser_for, source_token_spans, text, walk,
                     static_relations, STATIC_RELATION_RULES, SLICE_RULES,
                     statement_dependence_graph, bidirectional_pair_slice)


RAW = Path("/home/PublicData/PHY-data/vul_detect/data/PrimeVul_original")
FIRST = Path("results/losver_diagnosis")
OUT = Path("results/losver_patch_audit")
SPLITS = ("train", "valid", "test")
SEED = 42
SHAPES = ("addition_only", "replacement_only", "deletion_only", "mixed")


def rows_by_pair():
    result = {}
    for split in SPLITS:
        raw, ann = read_jsonl(RAW / f"primevul_{split}_paired.jsonl"), annotations(split)
        assert len(raw) == len(ann)
        for i in range(0, len(raw), 2):
            a, b = ann[i:i+2]
            assert a["pair_key"] == b["pair_key"]
            assert [a["target"], b["target"]] == [1, 0]
            assert all(raw[i+j]["func"] == ann[i+j]["func"] for j in (0, 1))
            result[a["pair_key"]] = (split, raw[i], raw[i+1], a, b)
    return result








def edit_shape(dv, dp, hunks):
    if not dv and dp:
        return "addition_only"
    if dv and not dp:
        return "deletion_only"
    if not dv and not dp:
        return "unchanged"
    return "replacement_only" if all(h["tag"] == "replace" for h in hunks) else "mixed"


def changed_roles(parsed, changed):
    roles = defaultdict(set)
    for n in parsed["nodes"]:
        overlap = source_lines(n) & set(changed)
        if not overlap:
            continue
        if n.type == "comment":
            roles["comment"].update(overlap)
        if n.type in {"declaration", "assignment_expression", "init_declarator"}:
            roles["definition_or_assignment"].update(overlap)
        if n.type in {"subscript_expression", "pointer_expression"}:
            roles["memory_or_pointer_operation"].update(overlap)
        if n.type == "call_expression":
            roles["call"].update(overlap)
            f = n.child_by_field_name("function")
            name = text(f, parsed["raw"]) if f else ""
            if re.search(r"(?:^|_)(?:free|alloc|realloc|lock|unlock|release|close|destroy)(?:$|_)", name):
                roles["resource_call_candidate"].update(overlap)
            if name in {"assert", "ASSERT", "BUG_ON", "WARN_ON", "CHECK", "DCHECK"}:
                roles["assertion_candidate"].update(overlap)
        if n.type in CONTROLS:
            condition = n.child_by_field_name("condition") or n.child_by_field_name("right")
            if condition:
                roles["condition"].update(source_lines(condition) & set(changed))
    return {k: sorted(v) for k, v in roles.items()}


def collect(output=OUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    old = {r["pair_key"]: r for r in csv.DictReader((FIRST/"pair_metadata.csv").open())}
    first = {r["sample_key"]: r for r in csv.DictReader((FIRST/"functions.csv").open())
             if r["mode"] == "l1"}
    records, timing = [], []
    started = time.perf_counter()
    for index, (key, (split, a, b, av, bp)) in enumerate(rows_by_pair().items()):
        t = time.perf_counter()
        va, pa = parse_source(a["func"]), parse_source(b["func"])
        dv, dp = av["lines_ground_truth"], bp["lines_ground_truth"]
        h = edit_hunks(a["func"], b["func"])
        shape = edit_shape(dv, dp, h)
        flags = [f for f in ("big_vul_idx", "cwe", "name") if old[key][f+"_mismatch"] == "1"]
        known = bool(va["name"] and pa["name"])
        roles_v, roles_p = changed_roles(va, dv), changed_roles(pa, dp)
        record = dict(
            pair_key=key, split=split, project=a["project"], commit_id=a["commit_id"],
            sample_key_v=av["sample_key"], sample_key_p=bp["sample_key"],
            idx_v=a["idx"], idx_p=b["idx"],
            big_vul_idx_v=a.get("big_vul_idx"), big_vul_idx_p=b.get("big_vul_idx"),
            cwe_v=a.get("cwe"), cwe_p=b.get("cwe"),
            first_round_flags=flags, first_round_candidate=bool(flags),
            name_v=va["name"], name_p=pa["name"],
            name_mismatch=known and va["name"] != pa["name"], name_known=known,
            parse_strict_v=va["strict"], parse_strict_p=pa["strict"],
            parse_errors_v=va["errors"], parse_errors_p=pa["errors"],
            language_v=va["language"], language_p=pa["language"],
            shape=shape, hunks=h, deleted_lines=len(dv), added_lines=len(dp),
            diff_v=dv, diff_p=dp,
            roles_v=roles_v, roles_p=roles_p,
            added_condition_candidate=not dv and bool(roles_p.get("condition") or roles_p.get("assertion_candidate")),
            removed_operation_candidate=not dp and bool(roles_v.get("call") or roles_v.get("memory_or_pointer_operation")),
            source_lines_v=len(a["func"].splitlines()), source_lines_p=len(b["func"].splitlines()),
            source_tokens_v=int(float(first[av["sample_key"]]["source_tokens"])),
            source_tokens_p=int(float(first[bp["sample_key"]]["source_tokens"])),
            gt_tokens_v=int(float(first[av["sample_key"]]["gt_tokens"])),
            gt_tokens_p=int(float(first[bp["sample_key"]]["gt_tokens"])),
            diff_position_v=sum((n+.5)/max(1,len(a["func"].splitlines())) for n in dv)/len(dv) if dv else None,
            diff_position_p=sum((n+.5)/max(1,len(b["func"].splitlines())) for n in dp)/len(dp) if dp else None,
        )
        records.append(record)
        timing.append(time.perf_counter()-t)
        if (index+1) % 500 == 0:
            print(f"audited {index+1} pairs in {time.perf_counter()-started:.1f}s", flush=True)
    cohorts = []
    for s in SPLITS:
        for shape in SHAPES:
            candidates = sorted([r["pair_key"] for r in records if r["split"] == s and r["shape"] == shape])
            chosen = random.Random(SEED).sample(candidates, min(8, len(candidates)))
            for key in chosen:
                cohorts.append(dict(pair_key=key, split=s, stratum=shape, reason="stratified_random",
                                    stratum_population=len(candidates), stratum_sample=len(chosen)))
    train_probe = []
    for shape in SHAPES:
        candidates = [c["pair_key"] for c in cohorts if c["split"] == "train" and c["stratum"] == shape]
        train_probe.extend(candidates)
    json_records = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v
                     for k,v in r.items()} for r in records]
    write_csv(output/"pairs.csv", json_records)
    write_csv(output/"review_sample.csv", cohorts)
    write_json(output/"config.json", dict(
        source_commit="032b05e7b78134f023a74a4aeb1156849374d82b", seed=SEED,
        official_pairs=dict(train=4354, valid=562, test=564), raw_root=str(RAW),
        original_diff="data.modified_lines: difflib.Differ, separately in each direction; nonblank deletions",
        audit_alignment="SequenceMatcher(autojunk=False), structural alignment only; original labels unchanged",
        sample="8 pairs per split x structural edit type, without replacement, same Seed42, independent of flags/model scores",
        review_strata=SHAPES, train_region_evaluation=train_probe,
        region_rule_scope="Train only; no Valid/Test region extraction or rule selection",
        no_training=True, no_model_loading=True, no_member_changes=True,
    ))
    write_json(output/"collection_cost.json", dict(pairs=len(records), seconds=time.perf_counter()-started,
                                                pair_seconds=quantiles(timing)))


def load_pairs(output=OUT):
    rows = list(csv.DictReader((Path(output)/"pairs.csv").open()))
    structured = {"first_round_flags", "hunks", "diff_v", "diff_p", "roles_v", "roles_p", "semantic_tags", "diff_noncode_v", "diff_noncode_p"}
    booleans = {"first_round_candidate", "name_mismatch", "name_known", "parse_strict_v", "parse_strict_p",
                "added_condition_candidate", "removed_operation_candidate", "index_one_missing", "index_both_present_different",
                "cwe_nonempty_conflict", "code_tokens_identical", "source_witness_available"}
    integers = {"deleted_lines", "added_lines", "source_lines_v", "source_lines_p", "source_tokens_v", "source_tokens_p",
                "gt_tokens_v", "gt_tokens_p", "parse_errors_v", "parse_errors_p"}
    for r in rows:
        for k in structured:
            if k in r:
                r[k] = json.loads(r[k])
        for k in booleans:
            if k in r:
                r[k] = r[k] == "True"
        for k in integers:
            r[k] = int(r[k])
        for k in ("diff_position_v", "diff_position_p"):
            r[k] = float(r[k]) if r[k] else None
    return rows


def patch_chunks(patch):
    chunks, current = [], None
    for line in (patch or "").splitlines():
        if line.startswith("@@"):
            current = dict(header=line, rows=[])
            chunks.append(current)
        elif current is not None and line[:1] in {" ", "+", "-"}:
            current["rows"].append((line[0], line[1:]))
    return chunks


def apply_commit_patch(source, patch, reverse=False):
    """Unique contiguous hunk witnesses; trim context only, never changed lines.

    Indentation is ignored for source-extraction differences. A match documents
    source correspondence, not vulnerability truth or whole-file provenance.
    """
    original = [line.strip() for line in source.splitlines()]
    edits, witnesses = [], []
    for chunk in patch_chunks(patch):
        rows = [("-" if t == "+" else "+" if t == "-" else t, s)
                for t,s in chunk["rows"]] if reverse else chunk["rows"]
        changed = [i for i,(tag,_) in enumerate(rows) if tag != " "]
        if not changed:
            continue
        leading, trailing = changed[0], len(rows)-1-changed[-1]
        matches = []
        # Most context first; short ambiguous fragments do not qualify.
        for trim in range(leading+trailing+1):
            for left in range(leading+1):
                right = trim-left
                if not 0 <= right <= trailing:
                    continue
                part = rows[left:len(rows)-right if right else len(rows)]
                before = [s.strip() for t,s in part if t != "+"]
                after = [s.strip() for t,s in part if t != "-"]
                if not before:
                    continue
                positions = [i for i in range(len(original)-len(before)+1)
                             if original[i:i+len(before)] == before]
                if len(positions) == 1:
                    matches.append((positions[0],len(before),after))
            if matches:
                break
        if len({(i,n,tuple(a)) for i,n,a in matches}) == 1:
            at,n,after = matches[0]
            edits.append((at,at+n,after))
            witnesses.append(dict(header=chunk["header"], source_start=at,
                                  source_end=at+n, trimmed_context=trim))
    edits.sort()
    if not edits or any(a[1]>b[0] for a,b in zip(edits,edits[1:])):
        return None, []
    updated = original[:]
    for start,end,after in reversed(edits):
        updated[start:end] = after
    return "\n".join(updated), witnesses


def patch_correspondence(output=OUT):
    output = Path(output)
    commits = {c["sha"]:c for c in read_jsonl(output/"commit_evidence.jsonl")}
    records, pairs = [], rows_by_pair()
    for r in load_pairs(output):
        split,a,b,av,bp = pairs[r["pair_key"]]
        commit = commits.get(r["commit_id"].split("?")[0])
        conflicting_index = a.get("big_vul_idx") is not None and b.get("big_vul_idx") is not None and a["big_vul_idx"] != b["big_vul_idx"]
        confident_name_change = r["name_mismatch"] and r["parse_strict_v"] and r["parse_strict_p"]
        result = dict(pair_key=r["pair_key"], split=split,
            status="suspicious" if conflicting_index or confident_name_change else "uncertain",
            evidence_level="no_retrieved_commit", evidence_url=commit["html_url"] if commit else "",
            mechanism="", witnesses=[])
        if commit:
            result["evidence_level"] = "commit_retrieved_no_complete_witness"
            exact, lexical, separate = [], [], []
            for file in commit["files"]:
                patch = file.get("patch")
                if not patch:
                    continue
                after, vw = apply_commit_patch(a["func"], patch)
                before, pw = apply_commit_patch(b["func"], patch, reverse=True)
                canonical = "\n".join(line.strip() for line in b["func"].splitlines())
                if after is not None and after == canonical:
                    exact.append(dict(file=file["filename"], vulnerable_patch=vw))
                elif (after is not None and [t[0] for t in source_token_spans(after)] ==
                      [t[0] for t in source_token_spans(b["func"])]) or (
                        before is not None and [t[0] for t in source_token_spans(before)] ==
                        [t[0] for t in source_token_spans(a["func"])]):
                    lexical.append(dict(file=file["filename"], vulnerable_patch=vw, patched_reverse_patch=pw))
                if after is not None and before is not None and r["name_mismatch"]:
                    an, bn = parse_source(after)["name"], parse_source(before)["name"]
                    if an == r["name_v"] and bn == r["name_p"]:
                        separate.append(dict(file=file["filename"], vulnerable_patch=vw, patched_reverse_patch=pw))
            if exact:
                result.update(status="legal", evidence_level="complete_function_patch_reconstruction",
                    mechanism="rename" if r["name_mismatch"] else "same_function_patch",
                    witnesses=exact)
            elif lexical:
                result.update(status="legal", evidence_level="patch_reconstruction_equal_code_tokens",
                    mechanism="rename_with_formatting" if r["name_mismatch"] else "same_function_patch_with_formatting",
                    witnesses=lexical)
            elif separate:
                result.update(status="verified_error", evidence_level="two_independent_function_patch_witnesses",
                    mechanism="different_functions_repaired_separately", witnesses=separate)
        records.append(result)
    with (output/"patch_correspondence.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(dict(Counter(r["status"] for r in records)),flush=True)


def function_catalog(source):
    raw = source.encode()
    root = parser_for("cpp").parse(raw).root_node
    result = []
    for n in walk(root):
        if n.type != "function_definition":
            continue
        content = text(n, raw)
        name = parse_source(content)["name"]
        context, parent = [], n.parent
        while parent is not None:
            if parent.type in {"class_specifier", "struct_specifier", "namespace_definition"}:
                identifier = parent.child_by_field_name("name")
                context.append(parent.type+":"+(text(identifier,raw) if identifier else "anonymous"))
            elif parent.type in {"preproc_if", "preproc_ifdef", "preproc_else", "preproc_elif"}:
                context.append(parent.type+":"+text(parent,raw).splitlines()[0].strip())
            parent = parent.parent
        result.append(dict(name=name, context=list(reversed(context)), start_line=n.start_point.row+1,
                           end_line=n.end_point.row+1, parse_error=n.has_error,
                           tokens=tuple(t[0] for t in source_token_spans(content))))
    return result


def source_witnesses(files_path, output=OUT):
    """Persist source locations/matches, not copies of complete upstream files."""
    files = json.loads(Path(files_path).read_text())
    catalogs = defaultdict(list)
    start = time.perf_counter()
    for file in files:
        for record in function_catalog(file["content"]):
            catalogs[file["commit"]].append(dict(record, path=file["path"], ref=file["ref"],
                repo=file["repo"], is_child=file["ref"]==file["commit"]))
    rows = []
    for key,(split,a,b,av,bp) in rows_by_pair().items():
        catalog = catalogs.get(a["commit_id"].split("?")[0])
        if not catalog:
            continue
        record = dict(pair_key=key, split=split, status="uncertain", mechanism="incomplete_source_match", matches={})
        for side,local in (("v",a),("p",b)):
            tokens = tuple(t[0] for t in source_token_spans(local["func"]))
            found = []
            for item in catalog:
                # Original extraction sometimes omits a standalone return type.
                extra = len(item["tokens"])-len(tokens)
                if 0 <= extra <= 8 and item["tokens"][-len(tokens):] == tokens:
                    match = {k:v for k,v in item.items() if k!="tokens"}
                    match.update(omitted_prefix_tokens=extra,
                        url=f"https://github.com/{item['repo']}/blob/{item['ref']}/{item['path']}#L{item['start_line']}")
                    found.append(match)
            record["matches"][side] = found
        old = [x for x in record["matches"]["v"] if not x["is_child"]]
        new = [x for x in record["matches"]["p"] if x["is_child"]]
        p_old = [x for x in record["matches"]["p"] if not x["is_child"]]
        if len(old)==1 and len(new)==1:
            v, p = old[0], new[0]
            different_preexisting = [x for x in p_old if x["path"]!=v["path"] or
                                    (x["start_line"],x["end_line"])!=(v["start_line"],v["end_line"])]
            if different_preexisting:
                record.update(status="verified_error", mechanism="patched_member_is_a_different_preexisting_function")
            elif (v["path"],v["name"],v["context"]) == (p["path"],p["name"],p["context"]):
                record.update(status="legal", mechanism="original_parent_child_same_function")
        rows.append(record)
    with (Path(output)/"source_witnesses.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    write_json(Path(output)/"source_witness_cost.json", dict(files=len(files), pairs=len(rows),
        seconds=time.perf_counter()-start, statuses=dict(Counter(r["status"] for r in rows)),
        matching="whole function code tokens, omitting comments/whitespace; at most 8 omitted leading return-type/qualifier tokens; all matched source locations retained"))
    print(dict(Counter(r["status"] for r in rows)),flush=True)


CHECK_MACROS = {"assert", "ASSERT", "BUG_ON", "WARN_ON", "CHECK", "DCHECK", "OP_REQUIRES", "OP_REQUIRES_OK"}


def lexical_code_lines(source):
    starts, position = [], 0
    for line in source.splitlines(keepends=True):
        starts.append(position)
        position += len(line)
    result = set()
    for _,a,b in source_token_spans(source):
        result.update(range(bisect_right(starts,a)-1,bisect_right(starts,b-1)))
    return result


def semantic_edit_family(record, a, b):
    """Train-defined syntactic categories, not vulnerability or repair truth."""
    shape = record["shape"]
    if shape == "replacement_only":
        return "replacement"
    if shape == "mixed":
        return "mixed"
    if shape == "addition_only":
        parsed, changed = parse_source(b["func"]), set(record["diff_p"])
        if not parsed["strict"]:
            return "addition_unknown_parse"
        guards = []
        for n in parsed["nodes"]:
            if n.type == "if_statement":
                condition = n.child_by_field_name("condition")
                if condition and source_lines(condition) <= changed:
                    guards.append(n)
        new_units = [n for n in parsed["nodes"] if n.type in STATEMENTS and source_lines(n)&changed]
        outside = []
        assertions = 0
        for n in new_units:
            protected = any(g.start_byte <= n.start_byte and n.end_byte <= g.end_byte for g in guards)
            calls = [x for x in walk(n) if x.type=="call_expression"]
            assertion = any(text(x.child_by_field_name("function"), parsed["raw"]) in CHECK_MACROS
                            for x in calls if x.child_by_field_name("function") is not None)
            assertions += int(assertion)
            if not protected and not assertion:
                outside.append(n)
        return "addition_check_candidate" if (guards or assertions) and not outside else "addition_other"
    if shape == "deletion_only":
        removed = "\n".join(a["func"].splitlines()[i] for i in record["diff_v"])
        if any(re.search(r"\b"+re.escape(name)+r"\s*\(",removed) for name in CHECK_MACROS):
            return "deletion_assertion_candidate"
        if record["roles_v"].get("call") or record["roles_v"].get("memory_or_pointer_operation"):
            return "deletion_operation_candidate"
        return "deletion_other"
    return "unchanged"


def summarize(output=OUT):
    output = Path(output)
    records, pairs = load_pairs(output), rows_by_pair()
    correspondence = {r["pair_key"]:r for r in read_jsonl(output/"patch_correspondence.jsonl")}
    source = {r["pair_key"]:r for r in read_jsonl(output/"source_witnesses.jsonl")}
    reviews = {r["pair_key"]:r for r in read_jsonl(output/"manual_reviews.jsonl")}
    sample = list(csv.DictReader((output/"review_sample.csv").open()))
    assert set(r["pair_key"] for r in sample) == {k for k,r in reviews.items() if r["scope"]=="stratified_random"}
    metadata = Counter()
    feature_rows, type_rows, augmented = [], [], []
    functions = {r["sample_key"]:r for r in csv.DictReader((FIRST/"functions.csv").open()) if r["mode"]=="l1"}
    for r in records:
        key = r["pair_key"]
        split,a,b,av,bp = pairs[key]
        audit = correspondence[key]
        if source.get(key,{}).get("status") in {"legal","verified_error"}:
            audit = dict(audit,**{k:v for k,v in source[key].items() if k not in {"pair_key","split","matches"}},
                         evidence_level="original_parent_child_function_match")
        review = reviews.get(key,{})
        if review.get("status_override"):
            audit = dict(audit,status=review["status_override"],evidence_level="original_patch_source_review",mechanism=review["note"])
        assert audit["status"] in {"verified_error","suspicious","legal","uncertain"}
        index_mismatch = a.get("big_vul_idx") != b.get("big_vul_idx")
        index_missing = index_mismatch and (a.get("big_vul_idx") is None or b.get("big_vul_idx") is None)
        cwe_conflict = bool(a.get("cwe") and b.get("cwe") and a["cwe"]!=b["cwe"])
        metadata[split,"index_both_present_different"] += int(index_mismatch and not index_missing)
        metadata[split,"index_one_missing"] += int(index_missing)
        metadata[split,"cwe_raw_mismatch"] += int(a.get("cwe")!=b.get("cwe"))
        metadata[split,"cwe_nonempty_conflict"] += int(cwe_conflict)
        family = semantic_edit_family(r,a,b)
        va = tuple(x[0] for x in source_token_spans(a["func"]))
        pb = tuple(x[0] for x in source_token_spans(b["func"]))
        r.update(quality_status=audit["status"],evidence_level=audit["evidence_level"],
            evidence_url=audit["evidence_url"],quality_mechanism=audit["mechanism"],
            index_one_missing=index_missing,index_both_present_different=index_mismatch and not index_missing,
            cwe_nonempty_conflict=cwe_conflict,semantic_family=family,
            code_tokens_identical=va==pb,manual_scope=review.get("scope",""),
            semantic_tags=review.get("semantic_tags",[]),review_note=review.get("note",""),
            source_witness_available=key in source,
            diff_noncode_v=sorted(set(r["diff_v"])-lexical_code_lines(a["func"])),
            diff_noncode_p=sorted(set(r["diff_p"])-lexical_code_lines(b["func"])))
        augmented.append(r)
        for side,local in (("v",av),("p",bp)):
            f = functions[local["sample_key"]]
            count = r["deleted_lines" if side=="v" else "added_lines"]
            total = r["source_lines_"+side]
            gt_tokens,code_tokens = int(float(f["gt_tokens"])),int(float(f["code_tokens"]))
            feature_rows.append(dict(pair_key=key,sample_key=local["sample_key"],split=split,
                label=local["target"],side=side,shape=r["shape"],quality_status=audit["status"],
                diff_lines=count,diff_line_fraction=count/total if total else 0,
                diff_tokens=gt_tokens,diff_token_fraction=gt_tokens/code_tokens if code_tokens else 0,
                empty_mask=int(count==0),
                mean_relative_position=r["diff_position_"+side] if r["diff_position_"+side] is not None else -1.,
                selected_mass=selected_mass(gt_tokens,code_tokens) if code_tokens else 0.))
    write_csv(output/"pairs.csv", [{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in augmented])
    write_csv(output/"oracle_features.csv",feature_rows)
    features = ("diff_lines","diff_line_fraction","diff_tokens","diff_token_fraction","empty_mask","mean_relative_position","selected_mass")
    auc_rows = []
    for scope in ("all", "both_nonempty", "equal_nonzero_line_count"):
        eligible = {r["pair_key"] for r in augmented if scope=="all" or
                    (r["deleted_lines"]>0 and r["added_lines"]>0 and
                     (scope=="both_nonempty" or r["deleted_lines"]==r["added_lines"]))}
        for name in features:
            train = [r for r in feature_rows if r["split"]=="train" and r["pair_key"] in eligible]
            train_auc = roc_auc_score([r["label"] for r in train],[r[name] for r in train])
            sign = 1 if train_auc>=.5 else -1
            for split in SPLITS:
                rr=[r for r in feature_rows if r["split"]==split and r["pair_key"] in eligible]
                auc_rows.append(dict(feature=name,scope=scope,split=split,direction_from_train=sign,
                    pairs=len(rr)//2,n=len(rr),positive=sum(r["label"] for r in rr),
                    auc=float(roc_auc_score([r["label"] for r in rr],[sign*r[name] for r in rr])),
                    uses_paired_diff=True,deployable_from_single_function=False))
    write_csv(output/"oracle_feature_auc.csv",auc_rows)
    summary = dict(splits={}, metadata={}, random_sample={}, train_region={}, manual_semantics={},
        warnings=["No population error-rate estimate from targeted candidates or disproportionate stratified samples.",
                  "Legal means supported function lineage/source correspondence, not verified vulnerable/secure labels.",
                  "Uncertain means not adjudicated, absent/incomplete provenance, or ambiguous function evolution.",
                  "Semantic/manual region labels are this assistant's source review, not independent dual-human gold labels."])
    for s in SPLITS:
        rr=[r for r in augmented if r["split"]==s]
        flagged=[r for r in rr if r["first_round_candidate"]]
        sample_keys={r["pair_key"] for r in sample if r["split"]==s}
        sampled=[r for r in rr if r["pair_key"] in sample_keys]
        summary["splits"][s]=dict(pairs=len(rr),
            quality=dict(Counter(r["quality_status"] for r in rr)),
            first_round_candidates=len(flagged),candidate_quality=dict(Counter(r["quality_status"] for r in flagged)),
            index_mismatch_quality=dict(Counter(r["quality_status"] for r in rr if r["index_one_missing"] or r["index_both_present_different"])),
            index_missing_quality=dict(Counter(r["quality_status"] for r in rr if r["index_one_missing"])),
            index_values_conflict_quality=dict(Counter(r["quality_status"] for r in rr if r["index_both_present_different"])),
            shapes=dict(Counter(r["shape"] for r in rr)),semantic_families=dict(Counter(r["semantic_family"] for r in rr)),
            zero_v_diff=sum(r["deleted_lines"]==0 for r in rr),zero_p_diff=sum(r["added_lines"]==0 for r in rr),
            code_token_identical_pairs=sum(r["code_tokens_identical"] for r in rr),
            strict_both_parse=sum(r["parse_strict_v"] and r["parse_strict_p"] for r in rr),
            deleted_lines=quantiles([r["deleted_lines"] for r in rr]),added_lines=quantiles([r["added_lines"] for r in rr]),
            known_error_diff_v=sum(r["deleted_lines"] for r in rr if r["quality_status"]=="verified_error"),
            known_error_diff_p=sum(r["added_lines"] for r in rr if r["quality_status"]=="verified_error"),
            noncode_diff_v=sum(len(r["diff_noncode_v"]) for r in rr),
            noncode_diff_p=sum(len(r["diff_noncode_p"]) for r in rr),
            total_diff_v=sum(r["deleted_lines"] for r in rr),total_diff_p=sum(r["added_lines"] for r in rr))
        summary["metadata"][s]={k:v for (split,k),v in metadata.items() if split==s}
        summary["random_sample"][s]=dict(pairs=len(sampled),quality=dict(Counter(r["quality_status"] for r in sampled)),
            errors=[r["pair_key"] for r in sampled if r["quality_status"]=="verified_error"],
            uncertain_security=[r["pair_key"] for r in sampled if "security_relevance_uncertain" in r["semantic_tags"]])
        summary["manual_semantics"][s]=dict(pairs=len(sampled),nonexclusive_tags=dict(Counter(t for r in sampled for t in r["semantic_tags"])),
            addition_sample=[dict(pair_key=r["pair_key"],tags=r["semantic_tags"]) for r in sampled if r["shape"]=="addition_only"])
        for shape in SHAPES:
            group=[r for r in rr if r["shape"]==shape]
            type_rows.append(dict(split=s,shape=shape,pairs=len(group),fraction=len(group)/len(rr),
                deleted_median=quantiles([r["deleted_lines"] for r in group])["median"],
                added_median=quantiles([r["added_lines"] for r in group])["median"],
                deleted_total=sum(r["deleted_lines"] for r in group),added_total=sum(r["added_lines"] for r in group)))
    write_csv(output/"repair_types.csv",type_rows)
    regions = read_jsonl(output/"regions.train.jsonl")
    train_meta = {r["pair_key"]:r for r in augmented if r["split"]=="train"}
    samples = {r["pair_key"] for r in sample if r["split"]=="train"}
    jud = json.loads((output/"region_review_labels.json").read_text())
    line_reviews = []
    for r in regions:
        if r["pair_key"] not in samples:
            continue
        labels = jud.get(r["pair_key"].split(":")[-1],{})
        labeled = [n for tag in ("related","unrelated","unknown") for n in labels.get(tag,[])]
        assert len(labeled)==len(set(labeled)) and set(labeled)=={n+1 for n in r["added_lines"]}
        for tag in ("related","unrelated","unknown"):
            for n in labels.get(tag,[]):
                code = pairs[r["pair_key"]][1]["func"].splitlines()[n-1]
                line_reviews.append(dict(pair_key=r["pair_key"],sample_key=r["sample_key"],line=n,
                    judgment=tag,source=code,note=labels.get("note","")))
    write_csv(output/"region_line_review.csv",line_reviews)
    for group in ("all_train","addition_only","replacement_only","deletion_only","mixed"):
        rr = [r for r in regions if group=="all_train" or train_meta[r["pair_key"]]["shape"]==group]
        summary["train_region"][group] = dict(pairs=len(rr),status=dict(Counter(r["status"] for r in rr)),
            nonempty_regions=sum(bool(r["region_lines"]) for r in rr),
            with_insert_variable_match=sum(r["insert_variable_matches"]>0 for r in rr),
            neighbour_without_variable_match=sum(r["status"]=="ok" and train_meta[r["pair_key"]]["deleted_lines"]==0 and r["insert_variable_matches"]==0 for r in rr),
            region_lines=quantiles([len(r["region_lines"]) for r in rr if r["status"]=="ok"]),
            added_lines=quantiles([len(r["added_lines"]) for r in rr if r["status"]=="ok"]),
            limit_reached=sum(r["limit_reached"] for r in rr),
            unknown_insert_symbols=sum(bool(r["unresolved_insert_identifiers"]) for r in rr))
    summary["region_review"]=dict(pairs=len(samples),status=dict(Counter(r["status"] for r in regions if r["pair_key"] in samples)),
        pairs_with_added_lines=len({r["pair_key"] for r in line_reviews}),lines=len(line_reviews),
        judgments=dict(Counter(r["judgment"] for r in line_reviews)),
        pairs_with_unrelated=len({r["pair_key"] for r in line_reviews if r["judgment"]=="unrelated"}),
        definition="Related/unrelated/unknown concern the repair relation visible in reviewed source, not a vulnerability-location gold standard.")
    duplicate_rows = []
    summary["within_split_duplicates"] = {}
    for split in SPLITS:
        by_source = defaultdict(list)
        for s,a,b,av,bp in pairs.values():
            if s==split:
                by_source[av["func"]].append(av)
                by_source[bp["func"]].append(bp)
        groups = [rr for rr in by_source.values() if len(rr)>1]
        summary["within_split_duplicates"][split] = dict(groups=len(groups),extra_members=sum(len(rr)-1 for rr in groups),
            opposite_labels=sum(len({r["target"] for r in rr})>1 for rr in groups))
        for i,group in enumerate(groups):
            for row in group:
                duplicate_rows.append(dict(split=split,group=i,sample_key=row["sample_key"],pair_key=row["pair_key"],target=row["target"]))
    write_csv(output/"duplicate_members.csv",duplicate_rows)
    write_json(output/"summary.json",summary)
    config=json.loads((output/"config.json").read_text())
    config.update(region_rules=dict(language="existing Tree-sitter C/C++; no source rewriting", steps=[
        "nonempty deleted lines -> complete atomic statement/control header",
        "pure insertion -> next statement + nearest use of uniquely named local bindings across versions",
        "one lexical definition/use hop and enclosing control headers",
        "at most 12 additional dependency units; Diff/insertion seed units are retained"],
        max_added_dependency_units=12,dependency_certainty="AST containment and scope binding are syntactic; def/use links are heuristic candidates, not proven reaching definitions",
        rules_developed_from="Train examples; Valid/Test source review only after Train rules fixed; region extraction Train only"),
        semantic_check_macros=sorted(CHECK_MACROS),original_source_requests="source_requests.json",
        manual_review="91 fixed stratified source/Diff reviews plus targeted original-commit reviews; not independent human annotations")
    write_json(output/"config.json",config)
    print(json.dumps(dict(splits={s:summary["splits"][s]["quality"] for s in SPLITS},region_review=summary["region_review"]),ensure_ascii=False),flush=True)


def inspect(keys):
    pairs = rows_by_pair()
    for key in keys:
        s, a, b, av, bp = pairs[key]
        print(f"\n{key} project={a['project']} commit={a['commit_id']}")
        for side, r in (("v", av), ("p", bp)):
            print(side, "GT", [x+1 for x in r["lines_ground_truth"]])
            for n,line in enumerate(r["func"].splitlines(),1):
                print(f"{n:4d} {'D' if n-1 in r['lines_ground_truth'] else ' '} {line}")
        print("\n".join(difflib.unified_diff(a["func"].splitlines(), b["func"].splitlines(), n=3, lineterm="")))


def upstream_records(output=OUT):
    """Stream only candidate provenance from the existing upstream CSV, read-only."""
    output = Path(output)
    candidates = [r for r in csv.DictReader((FIRST/"pair_metadata.csv").open())
                  if r["big_vul_idx_mismatch"] == "1"]
    selected = {int(r[k]) for r in candidates for k in ("big_vul_idx_v", "big_vul_idx_p") if r[k]}
    pairs = rows_by_pair()
    by_index = defaultdict(list)
    for s, a, b, av, bp in pairs.values():
        for raw, ann in ((a,av),(b,bp)):
            if raw.get("big_vul_idx") in selected:
                by_index[raw["big_vul_idx"]].append(ann)
    csv.field_size_limit(2**31-1)
    found, start, seen = [], time.perf_counter(), 0
    path = Path("/home/PublicData/PHY-data/vul_detect/data/bigvul/MSR_data_cleaned.csv")
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            seen += 1
            index = int(row[""])
            if index in selected:
                local = by_index[index]
                record = {k:row[k] for k in ("codeLink", "commit_id", "parentID", "file_name", "project", "vul", "CVE ID", "CWE ID")}
                record.update(big_vul_idx=index,
                    local_members=[dict(sample_key=r["sample_key"], pair_key=r["pair_key"], target=r["target"],
                                        exact_before=r["func"]==row["func_before"], exact_after=r["func"]==row["func_after"],
                                        stripped_before=r["func"].strip()==row["func_before"].strip(),
                                        stripped_after=r["func"].strip()==row["func_after"].strip()) for r in local])
                found.append(record)
            if index >= max(selected):
                break
    missing = selected-{r["big_vul_idx"] for r in found}
    if missing:
        raise ValueError(f"upstream candidate rows missing: {sorted(missing)}")
    write_json(output/"bigvul_provenance.json", dict(source=str(path), inspected_rows=seen,
        seconds=time.perf_counter()-start, records=found,
        scope="only existing metadata and exact source comparisons; no raw source copied or modified"))
    print(f"upstream provenance: {len(found)} candidate records, {seen} streamed rows, {time.perf_counter()-start:.1f}s", flush=True)




def candidate_region(vulnerable, patched, dv, dp):
    """Bounded Train-only region probe. Dependencies are explicitly candidates.

    Fixed before examining held-out patch semantics: atomic AST units; insertion
    neighbour + uniquely named local binding uses; one lexical def/use hop and
    enclosing control headers; at most 12 added units. No alias/CFG claim.
    """
    start = time.perf_counter()
    v, p = parse_source(vulnerable), parse_source(patched)
    result = dict(status="ok", parse_strict_v=v["strict"], parse_strict_p=p["strict"],
                  original_diff_v=sorted(dv), original_diff_p=sorted(dp),
                  syntax_lines=[], region_lines=[], added_lines=[], relations=[],
                  seed_units=0, added_units=0, insert_variable_matches=0,
                  unresolved_insert_identifiers=[], limit_reached=False)
    if not v["strict"] or not p["strict"]:
        result["status"] = "parse_or_function_boundary_failure"
    elif not v["name"] or not p["name"] or v["name"] != p["name"]:
        result["status"] = "function_correspondence_unknown"
    else:
        units, names = statement_units(v)
        pu, pnames = statement_units(p)
        seed = {i for i,u in enumerate(units) if u["lines"] & set(dv)}
        relations = []
        for i in sorted(seed):
            relations.append(dict(kind="diff_ast_unit", certainty="syntactic", unit=i,
                                  lines=sorted(units[i]["lines"]), symbol=None))
        insert_anchors = []
        unknown = set()
        for h in edit_hunks(vulnerable, patched):
            if h["tag"] != "insert":
                continue
            anchor = h["v_start"]
            insert_anchors.append(anchor)
            changed = set(range(h["p_start"], h["p_end"]))
            refs = set().union(*(u["refs"] for u in pu if u["lines"] & changed)) if pu else set()
            # Unique local names are required on both versions; no guessed rename.
            for name, _ in sorted(refs):
                if len(names.get(name, [])) != 1 or len(pnames.get(name, [])) != 1:
                    unknown.add(name)
                    continue
                binding = names[name][0]["key"]
                uses = [i for i,u in enumerate(units) if binding in u["refs"] and binding not in u["writes"]]
                if not uses:
                    unknown.add(name)
                    continue
                after = [i for i in uses if units[i]["node"].start_point.row >= anchor]
                chosen = min(after or uses, key=lambda i: (abs(units[i]["node"].start_point.row-anchor), i))
                seed.add(chosen)
                result["insert_variable_matches"] += 1
                relations.append(dict(kind="inserted_binding_use", certainty="heuristic_candidate", unit=chosen,
                                      lines=sorted(units[chosen]["lines"]), symbol=name))
            nearby = [i for i,u in enumerate(units) if u["node"].start_point.row >= anchor]
            if units:
                chosen = min(nearby or range(len(units)), key=lambda i: (abs(units[i]["node"].start_point.row-anchor), i))
                seed.add(chosen)
                relations.append(dict(kind="insertion_neighbour", certainty="heuristic_candidate", unit=chosen,
                                      lines=sorted(units[chosen]["lines"]), symbol=None))
        result["unresolved_insert_identifiers"] = sorted(unknown)
        syntax = set().union(*(units[i]["lines"] for i in seed)) if seed else set()
        proposals = []
        for i in sorted(seed):
            unit = units[i]
            for key in sorted(unit["refs"]):
                before = [j for j,u in enumerate(units) if key in u["writes"]
                          and u["node"].end_byte <= unit["node"].start_byte]
                if before:
                    j = max(before, key=lambda j: units[j]["node"].end_byte)
                    proposals.append((j,"lexical_definition_candidate",key[0]))
                if key in unit["writes"]:
                    after = [j for j,u in enumerate(units) if key in u["refs"]
                             and u["node"].start_byte >= unit["node"].end_byte]
                    if after:
                        j = min(after, key=lambda j: units[j]["node"].start_byte)
                        proposals.append((j,"next_bound_use_candidate",key[0]))
            parent = unit["node"].parent
            while parent is not None and parent != v["function"]:
                if parent.type in CONTROLS:
                    cond = parent.child_by_field_name("condition") or parent.child_by_field_name("right")
                    if cond:
                        found = [j for j,u in enumerate(units) if u["node"] == cond]
                        if found:
                            proposals.append((found[0],"enclosing_control_header",None))
                parent = parent.parent
        added = []
        for j,kind,symbol in proposals:
            if j not in seed and j not in added:
                if len(added) == 12:
                    result["limit_reached"] = True
                    continue
                added.append(j)
                relations.append(dict(kind=kind,
                    certainty="syntactic" if kind=="enclosing_control_header" else "heuristic_candidate",
                    unit=j, lines=sorted(units[j]["lines"]), symbol=symbol))
        lines = set().union(*(units[i]["lines"] for i in sorted(seed|set(added)))) if seed or added else set()
        nonempty = {i for i,l in enumerate(vulnerable.splitlines()) if l.strip()}
        lines &= nonempty
        result.update(syntax_lines=sorted(syntax & nonempty), region_lines=sorted(lines),
                      added_lines=sorted(lines-set(dv)), relations=relations,
                      seed_units=len(seed), added_units=len(added))
        if not lines:
            result["status"] = "no_resolved_statement"
    result["seconds"] = time.perf_counter()-start
    return result


def regions(output=OUT):
    output = Path(output)
    start, rows = time.perf_counter(), []
    with (output/"regions.train.jsonl").open("w") as f:
        for i,(key,(split,a,b,av,bp)) in enumerate(rows_by_pair().items()):
            if split != "train":
                continue
            region = candidate_region(a["func"], b["func"], av["lines_ground_truth"], bp["lines_ground_truth"])
            region.update(pair_key=key, sample_key=av["sample_key"], split="train")
            f.write(json.dumps(region,ensure_ascii=False)+"\n")
            rows.append(region)
            if (i+1)%500 == 0:
                print(f"Train region probes {i+1}, elapsed {time.perf_counter()-start:.1f}s",flush=True)
    write_json(output/"region_cost.json",dict(pairs=len(rows),seconds=time.perf_counter()-start,
        per_pair_seconds=quantiles([r["seconds"] for r in rows]),
        status=dict(Counter(r["status"] for r in rows)),
        definition="Tree-sitter syntactic/binding candidates; nearest lexical definitions are NOT sound reaching definitions or dependence proofs",
        unavailable="no alias analysis, interprocedural analysis, CFG dominance/postdominance, or missing caller context"))


PATTERN_EDITS = ("boundary_relation_change", "integer_relation_change", "integer_arithmetic_change", "null_relation_change",
                 "memory_operation_change", "resource_relation_change", "release_call_change",
                 "call_argument_change", "control_flow_change")


def relation_signature(relation, source):
    raw = source.encode()
    def tokens(span):
        return tuple(t[0] for t in source_token_spans(raw[span["start_byte"]:span["end_byte"]].decode()))
    return (relation["pattern"],tokens(relation["operation"]),tokens(relation["evidence"]),
            relation["role"],relation.get("argument"),relation.get("branch_truth"),relation.get("zero_case"))


def ast_pattern_changes(old, new, source_old, source_new):
    """Conservative multiset AST/token changes, not a vulnerability taxonomy."""
    if old["status"]!="ok" or new["status"]!="ok":
        return dict(status="parse_unknown", patterns=[], counts={}, call_changes=[])
    before=Counter(relation_signature(r,source_old) for r in old["relations"])
    after=Counter(relation_signature(r,source_new) for r in new["relations"])
    delta=(before-after)+(after-before)
    count=Counter()
    families=dict(boundary_check_relation="boundary_relation_change", integer_range_relation="integer_relation_change",
                  null_check_relation="null_relation_change",memory_extent_relation="memory_operation_change",
                  resource_lifetime_relation="resource_relation_change")
    for key,n in delta.items():
        if key[0] in families:
            count[families[key[0]]]+=n
    ops=[Counter((r["kind"],r["signature"]) for r in f["ast_operations"]) for f in (old,new)]
    for (kind,signature),n in ((ops[0]-ops[1])+(ops[1]-ops[0])).items():
        if kind in {"memory_span","indexed_memory","pointer_access"}:
            count["memory_operation_change"]+=n
        if kind=="integer_arithmetic":
            count["integer_arithmetic_change"]+=n
    calls=[Counter((c["name"],tuple(c["arguments"])) for c in f["ast_calls"]) for f in (old,new)]
    unmatched=[calls[0]-calls[1],calls[1]-calls[0]]
    changes=[]
    for name in sorted({k[0] for c in unmatched for k in c}):
        a=[args for (callee,args),n in unmatched[0].items() if callee==name for _ in range(n)]
        b=[args for (callee,args),n in unmatched[1].items() if callee==name for _ in range(n)]
        if len(a)==len(b)==1 and a[0]!=b[0]:
            changes.append(dict(callee=name, before=list(a[0]), after=list(b[0])))
    count["call_argument_change"]=len(changes)
    count["release_call_change"]=sum(n for c in unmatched for (name,args),n in c.items() if name in {"free","fclose","close"})
    units=[Counter((u["kind"],u["signature"]) for u in f["ast_units"] if u["kind"] in {
        "control_condition","return_statement","goto_statement","break_statement","continue_statement"}) for f in (old,new)]
    count["control_flow_change"]=sum(((units[0]-units[1])+(units[1]-units[0])).values())
    return dict(status="ok",patterns=[k for k in PATTERN_EDITS if count[k]],
                counts={k:count[k] for k in PATTERN_EDITS},call_changes=changes)


def pattern_review_sample(functions, split):
    """Sample functions first, one relation per function; never select by LOSVER."""
    rng=random.Random(SEED)
    sample=[]
    per_side=4 if split=="train" else 2
    for pattern in STATIC_RELATION_RULES["patterns"]:
        for target in (1,0):
            pool=[r for r in functions if r["target"]==target and any(x["pattern"]==pattern for x in r["relations"])]
            pool.sort(key=lambda r:r["sample_key"])
            for row in rng.sample(pool,min(per_side,len(pool))):
                relation=rng.choice([r for r in row["relations"] if r["pattern"]==pattern])
                rid=f"{row['sample_key']}:{pattern}:{relation['operation']['start_byte']}:{relation['evidence']['start_byte']}:{relation['role']}"
                sample.append(dict(review_id=rid,sample_key=row["sample_key"],pair_key=row["pair_key"],
                    target=target,pattern=pattern,relation=relation))
    for status in ("ok","parse_unknown"):
        for target in (1,0):
            pool=[r for r in functions if r["target"]==target and r["status"]==status and not r["relations"]]
            pool.sort(key=lambda r:r["sample_key"])
            for row in rng.sample(pool,min(2 if split=="train" else 1,len(pool))):
                sample.append(dict(review_id=f"{row['sample_key']}:{status}:no_candidate",sample_key=row["sample_key"],
                    pair_key=row["pair_key"],target=target,pattern="no_candidate",status=status))
    return sample


def pattern_cases(data, sample, output):
    by_key={r["sample_key"]:r for r in data}
    pairs={r["pair_key"]:[] for r in data}
    for r in data:
        pairs[r["pair_key"]].append(r)
    lines=["# Fixed source review samples", "", "1-based display lines; stored positions are 0-based. Review is by this assistant, not independent gold annotation.", ""]
    for case in sample:
        row=by_key[case["sample_key"]]
        lines += [f"## {case['review_id']}",f"target={row['target']}, pair={row['pair_key']}"]
        relation=case.get("relation")
        if relation:
            lines.append(json.dumps(relation,ensure_ascii=False))
            ns=set(relation["operation"]["lines"]+relation["evidence"]["lines"])
            chosen={j for n in ns for j in range(max(0,n-3),n+4)}
        else:
            chosen=set(range(min(35,len(row["func"].splitlines()))))
        lines += ["```c"]
        code=row["func"].splitlines()
        for n in sorted(chosen)[:70]:
            if n<len(code):
                lines.append(f"{n+1:5}: {code[n][:500]}")
        lines += ["```", "Pair edit (structural context only; never extractor input):", "```diff"]
        v,p=pairs[row["pair_key"]]
        edit=list(difflib.unified_diff(v["func"].splitlines(),p["func"].splitlines(),n=2,lineterm=""))
        lines += [s[:500] for s in edit[:70]]
        if len(edit)>70:
            lines.append("... display truncated; source retained unchanged ...")
        lines += ["```", ""]
    Path(output).write_text("\n".join(lines),encoding="utf-8")


def mine_patterns(data_dir, output, split):
    """All Train first; the identical source-only rules are then evaluated on Valid."""
    output=Path(output)
    if output==OUT:
        raise ValueError("patterns requires a separate output directory; historical patch audit is read-only")
    output.mkdir(parents=True,exist_ok=True)
    if split=="valid":
        config=json.loads((output/"rules.json").read_text())
        if config["rules"]!=STATIC_RELATION_RULES:
            raise ValueError("rules differ from completed Train audit; Valid must not select new rules")
        review=read_jsonl(output/"train.review.jsonl")
        expected=read_jsonl(output/"train.review_sample.jsonl")
        if ({r["review_id"] for r in review}!={r["review_id"] for r in expected}
                or any(r.get("relation_judgment") not in {"confirmed_relation","false_relation","unknown","not_applicable"} for r in review)):
            raise ValueError("complete Train source review before running Valid")
    data=read_jsonl(Path(data_dir)/"primevul_paired"/f"{split}.lines.jsonl")
    raw=read_jsonl(RAW/f"primevul_{split}_paired.jsonl")
    expected_pairs={"train":4354,"valid":562}[split]
    if len(data)!=expected_pairs*2 or len(raw)!=len(data):
        raise ValueError("unchanged official Original pair members required")
    for r,upstream in zip(data,raw):
        if r["func"]!=upstream["func"] or r["target"]!=upstream["target"]:
            raise ValueError("derived source or labels differ from original")
    with (OUT/"pairs.csv").open() as handle:
        previous={r["pair_key"]:r for r in csv.DictReader(handle) if r["split"]==split}
    scores=read_jsonl(Path("results/losver_512/primevul_paired/losver_localizer")/f"{split}.line_scores.jsonl")
    if [r["sample_key"] for r in scores]!=[r["sample_key"] for r in data]:
        raise ValueError("full 512 LOSVER scores and official members differ")
    started=time.perf_counter()
    functions,pairs,cost=[],[],[]
    with (output/f"functions.{split}.jsonl").open("w") as handle:
        for i in range(0,len(data),2):
            v,p=data[i:i+2]
            if [v["target"],p["target"]]!=[1,0] or v["pair_key"]!=p["pair_key"]:
                raise ValueError("unaligned official pair")
            local=[]
            for row in (v,p):
                start=time.perf_counter()
                result=static_relations(row["func"])
                cost.append(time.perf_counter()-start)
                local.append(result)
                public={k:value for k,value in result.items() if not k.startswith("ast_")}
                public.update(sample_key=row["sample_key"],pair_key=row["pair_key"],target=row["target"],
                              split=split,quality_status=previous[row["pair_key"]]["quality_status"])
                handle.write(json.dumps(public,ensure_ascii=False)+"\n")
                functions.append(public)
            delta=ast_pattern_changes(*local,v["func"],p["func"])
            delta.update(pair_key=v["pair_key"], split=split, shape=previous[v["pair_key"]]["shape"],
                         quality_status=previous[v["pair_key"]]["quality_status"])
            pairs.append(delta)
            if (i//2+1)%250==0:
                print(f"{split}: {i//2+1}/{expected_pairs} AST pairs, {time.perf_counter()-started:.1f}s",flush=True)
    write_jsonl(output/f"pairs.{split}.jsonl",pairs)
    sample=pattern_review_sample(functions,split)
    write_jsonl(output/f"{split}.review_sample.jsonl",sample)
    pattern_cases(data,sample,output/f"{split}.cases.md")
    write_json(output/f"{split}.cost.json",dict(functions=len(data),pairs=expected_pairs,
        seconds=time.perf_counter()-started,extraction_seconds=sum(cost),function_seconds=quantiles(cost)))
    if split=="train":
        write_json(output/"rules.json",dict(rules=STATIC_RELATION_RULES, seed=SEED,
            source_members="unchanged official Original 4354 Train pairs", original_diff_unchanged=True,
            stage_order="Train mining and source review before fixed-rule Valid; no Test read",
            localizer_reference="existing full 512 LOSVER, positive threshold >0.5; no inference rerun",
            ast_difference="multiset of atomic AST units and non-comment tokens; unique remaining callee matches for argument edits; not optimal tree-edit or semantic equivalence"))
    return dict(split=split,functions=len(functions),pairs=len(pairs),review_samples=len(sample))


def pattern_report(data_dir, output):
    """Existing LOSVER probabilities and Diff proxies only; no model execution."""
    output=Path(output)
    splits=[s for s in ("train","valid") if (output/f"{s}.cost.json").exists()]
    with (FIRST/"functions.csv").open() as handle:
        geometry={(r["mode"],r["sample_key"]):r for r in csv.DictReader(handle) if r["split"] in splits}
    frequency,edits,proxy,transitions,failures,reviews,stages=[],[],[],[],[],[],[]
    summary={}
    for split in splits:
        functions=read_jsonl(output/f"functions.{split}.jsonl")
        pairs=read_jsonl(output/f"pairs.{split}.jsonl")
        source=read_jsonl(Path(data_dir)/"primevul_paired"/f"{split}.lines.jsonl")
        if [r["sample_key"] for r in functions]!=[r["sample_key"] for r in source]:
            raise ValueError("pattern audit / labels / existing localizer members differ")
        counts=Counter()
        for r in functions:
            counts.update(r["failures"])
        counts["parse_unknown"]=sum(r["status"]!="ok" for r in functions)
        failures += [dict(split=split,reason=k,events=n,
                         functions=sum((r["status"]!="ok") if k=="parse_unknown" else k in r["failures"] for r in functions))
                     for k,n in sorted(counts.items())]
        eligible=sum(r["status"]=="ok" for r in pairs)
        summary[split]=dict(functions=len(functions),pairs=len(pairs),parsed_functions=sum(r["status"]=="ok" for r in functions),
            parsed_pairs=eligible,matched_functions=sum(bool(r["relations"]) for r in functions),
            matched_pairs=sum(bool(functions[i]["relations"] or functions[i+1]["relations"]) for i in range(0,len(functions),2)),
            parse_failures_with_errors=sum(r["status"]!="ok" and r["errors"]>0 for r in functions),
            parse_failures_without_unique_function=sum(r["status"]!="ok" and r["function_count"]!=1 for r in functions),
            costs=json.loads((output/f"{split}.cost.json").read_text()),
            known_wrong_pairs=sum(r["quality_status"]=="verified_error" for r in pairs),
            unclassified_parsed_pairs=sum(r["status"]=="ok" and not r["patterns"] for r in pairs))
        for pattern in PATTERN_EDITS:
            selected=[r for r in pairs if pattern in r["patterns"]]
            edits.append(dict(split=split,pattern=pattern,pairs=len(selected),all_pairs=len(pairs),
                coverage=len(selected)/len(pairs),parsed_pairs=eligible,
                coverage_among_parsed=len(selected)/eligible if eligible else 0,
                known_wrong_pairs=sum(r["quality_status"]=="verified_error" for r in selected)))
        for pattern in STATIC_RELATION_RULES["patterns"]:
            for target in (1,0):
                fs=[r for r in functions if r["target"]==target]
                selected=[r for r in fs if any(x["pattern"]==pattern for x in r["relations"])]
                frequency.append(dict(split=split,pattern=pattern,target=target,functions=len(selected),
                    total=len(fs),coverage=len(selected)/len(fs),
                    relations=sum(x["pattern"]==pattern for r in selected for x in r["relations"])))
            for i in range(0,len(functions),2):
                a,b=functions[i:i+2]
                na=sum(x["pattern"]==pattern for x in a["relations"])
                nb=sum(x["pattern"]==pattern for x in b["relations"])
                transitions.append(dict(split=split,pattern=pattern,pair_key=a["pair_key"],
                    vulnerable=na,patched=nb,delta=nb-na,
                    parsing_known=a["status"]==b["status"]=="ok"))
        operation_counts=Counter()
        for f in functions:
            operation_counts.update(f["operations"])
        trigger=dict(boundary_check_relation=("indexed_memory","memory_span"),null_check_relation=("indexed_memory","pointer_access"),
                     integer_range_relation=("integer_arithmetic",),memory_extent_relation=("memory_span",),
                     call_argument_relation=("call",),control_call_relation=("call",))
        for pattern in STATIC_RELATION_RULES["patterns"]:
            stages.append(dict(split=split,pattern=pattern,
                trigger_operations=sum(operation_counts[k] for k in trigger[pattern]) if pattern in trigger else None,
                matched_operations=len({(f["sample_key"],r["operation"]["start_byte"]) for f in functions for r in f["relations"] if r["pattern"]==pattern}),
                relation_instances=sum(r["pattern"]==pattern for f in functions for r in f["relations"])))
        for mode in ("512","l1"):
            loc=read_jsonl(Path(f"results/losver_{mode}/primevul_paired/losver_localizer")/f"{split}.line_scores.jsonl")
            if [r["sample_key"] for r in loc]!=[r["sample_key"] for r in source]:
                raise ValueError("existing localizer members differ")
            masks=defaultdict(Counter)
            for row,fn,lp in zip(source,functions,loc):
                ns=lp["line_numbers"]
                if len(ns)!=len(lp["line_scores"]) or len(ns)!=len(set(ns)):
                    raise ValueError("invalid existing localizer line map")
                g=geometry[mode,row["sample_key"]]
                code=row["func"].splitlines()
                if int(g["physical_lines"])!=len(code):
                    raise ValueError("historical source geometry differs")
                common=set(ns)
                complete=set(ns[:-1] if mode=="512" and g["truncated"]=="1" else ns)
                nonempty={i for i,line in enumerate(code) if line.strip()}
                full,prefix=set(),set()
                for r in fn["relations"]:
                    region=set(r["operation"]["lines"]+r["evidence"]["lines"])&nonempty
                    full.update(region)
                    if region.issubset(complete):
                        prefix.update(region)
                gt=set(row["lines_ground_truth"])&common
                selected=dict(static_full_source_projected=full,
                    losver={n for n,p in zip(ns,lp["line_scores"]) if p>.5})
                if mode=="512":
                    selected["static_prefix_evidence_only"]=prefix
                for name,mask in selected.items():
                    selected_lines=mask&common
                    for side in ("all","vulnerable" if row["target"] else "patched"):
                        masks[name,side].update(functions=1,nonempty=bool(selected_lines),lines=len(common),
                            positive_lines=len(gt),selected_lines=len(selected_lines),tp=len(gt&selected_lines))
            for (name,side),acc in masks.items():
                tp,npred,positive=acc["tp"],acc["selected_lines"],acc["positive_lines"]
                proxy.append(dict(split=split,mode=mode,method=name,side=side,functions=acc["functions"],
                    covered_functions=acc["nonempty"],lines=acc["lines"],positive_lines=positive,selected_lines=npred,
                    tp=tp,fp=npred-tp,fn=positive-tp,precision=tp/npred if npred else 0.,
                    recall=tp/positive if positive else 0.,f1=2*tp/(npred+positive) if npred+positive else 0.,
                    selected_fraction=npred/acc["lines"],meaning="original Diff proxy, NOT vulnerability-location accuracy"))
        rp=output/f"{split}.review.jsonl"
        if rp.exists():
            manual=read_jsonl(rp)
            expected=read_jsonl(output/f"{split}.review_sample.jsonl")
            if {r["review_id"] for r in manual}!={r["review_id"] for r in expected} or len(manual)!=len(expected):
                raise ValueError("source review sample keys/denominator differ")
            by_id={r["review_id"]:r for r in expected}
            for pattern in STATIC_RELATION_RULES["patterns"]+["no_candidate"]:
                selected=[r for r in manual if by_id[r["review_id"]]["pattern"]==pattern]
                c=Counter(r["relation_judgment"] for r in selected)
                rel=Counter(r["repair_relevance"] for r in selected)
                reviews.append(dict(split=split,pattern=pattern,reviewed=len(selected),
                    confirmed=c["confirmed_relation"],false_relation=c["false_relation"],unknown=c["unknown"],
                    not_applicable=c["not_applicable"],repair_related=rel["related"],repair_unrelated=rel["unrelated"],repair_unknown=rel["unknown"]))
    changes=[]
    for split in splits:
        for pattern in STATIC_RELATION_RULES["patterns"]:
            selected=[r for r in transitions if r["split"]==split and r["pattern"]==pattern and r["parsing_known"]]
            changes.append(dict(split=split,pattern=pattern,parsed_pairs=len(selected),
                more_in_patched=sum(r["delta"]>0 for r in selected),fewer_in_patched=sum(r["delta"]<0 for r in selected),
                same_nonempty=sum(r["delta"]==0 and r["vulnerable"]>0 for r in selected),
                absent_both=sum(r["vulnerable"]==r["patched"]==0 for r in selected),
                patched_only=sum(r["vulnerable"]==0 and r["patched"]>0 for r in selected),
                vulnerable_only=sum(r["vulnerable"]>0 and r["patched"]==0 for r in selected)))
    for name,rows in (("pattern_frequency.csv",frequency),("ast_repair_changes.csv",edits),
                      ("diff_proxy_comparison.csv",proxy),("pair_pattern_changes.csv",transitions),("failures.csv",failures)):
        write_csv(output/name,rows)
    write_csv(output/"pair_pattern_change_summary.csv",changes)
    write_csv(output/"extraction_stages.csv",stages)
    if reviews:
        write_csv(output/"source_review_summary.csv",reviews)
    write_json(output/"summary.json",dict(splits=summary,source_reviews=reviews,
        no_training=True,no_model_execution=True,test_used=False,rules=STATIC_RELATION_RULES))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def read_slice_export(directory):
    """Reuse the historical Joern neo4jcsv contract, retaining native evidence."""
    csv.field_size_limit(sys.maxsize)
    nodes,edges={},[]
    for header in sorted(Path(directory).glob("nodes_*_header.csv")):
        columns=next(csv.reader(header.open()))
        if columns[:2]!=[":ID",":LABEL"]:
            raise ValueError(f"unexpected Joern node header: {header}")
        with header.with_name(header.name.replace("_header","_data")).open(newline="") as handle:
            for values in csv.reader(handle):
                if len(values)!=len(columns):
                    raise ValueError(f"malformed Joern node: {header}")
                row=dict(kind=values[1])
                for name,value in zip(columns[2:],values[2:]):
                    name=name.partition(":")[0]
                    if name in {"LINE_NUMBER","LINE_NUMBER_END","COLUMN_NUMBER","COLUMN_NUMBER_END","OFFSET","OFFSET_END"}:
                        row[name]=int(value) if value else None
                    elif name in {"CODE","NAME","FILENAME"}:
                        row[name]=value.replace("\\\\","\\")
                    elif name=="IS_EXTERNAL":
                        row[name]=value=="true"
                nodes[values[0]]=row
    for kind in ("AST","CFG","CDG","REACHING_DEF"):
        header=Path(directory)/f"edges_{kind}_header.csv"
        if not header.exists():
            # neo4jcsv omits zero-edge labels (e.g. no recognized method).
            # Per-method CFG/edge counts below retain this as unknown/zero.
            continue
        columns=next(csv.reader(header.open()))
        with header.with_name(header.name.replace("_header","_data")).open(newline="") as handle:
            for row in csv.reader(handle):
                if len(row)!=len(columns) or row[2]!=kind:
                    raise ValueError(f"malformed native {kind} edge")
                edges.append(dict(source=row[0],target=row[1],kind=kind,
                    variable=row[3].replace("\\\\","\\") if kind=="REACHING_DEF" else ""))
    if not nodes:
        raise ValueError("empty Joern export")
    return nodes,edges


def split_slice_export(directory, records, filenames):
    nodes,edges=read_slice_export(directory)
    ast=defaultdict(list); outgoing=defaultdict(list)
    methods=defaultdict(list)
    for key,n in nodes.items():
        if n["kind"]=="METHOD" and not n.get("IS_EXTERNAL") and n.get("NAME")!="<global>" and n.get("LINE_NUMBER"):
            methods[Path(n.get("FILENAME","")).name].append(key)
    for edge in edges:
        outgoing[edge["source"]].append(edge)
        if edge["kind"]=="AST":
            ast[edge["source"]].append(edge["target"])
    result=[]
    for row,filename in zip(records,filenames):
        ids=methods[filename]
        r=dict(sample_key=row["sample_key"],pair_key=row["pair_key"],status="ok",nodes={},edges=[],counts={})
        if len(ids)!=1:
            r.update(status="joern_method_unknown",methods=len(ids));result.append(r);continue
        pending=[ids[0]];owned=set()
        while pending:
            key=pending.pop()
            if key in owned or (key!=ids[0] and nodes[key]["kind"]=="METHOD"):
                continue
            owned.add(key);pending.extend(ast[key])
        r["nodes"]={k:nodes[k] for k in sorted(owned)}
        r["edges"]=[e for key in sorted(owned) for e in outgoing[key]
                    if e["target"] in owned and e["kind"]!="AST"]
        r["counts"]=dict(Counter(e["kind"] for e in r["edges"]))
        if not r["counts"].get("CFG"):
            r["status"]="joern_cfg_missing"
        result.append(r)
    return result


def run_slice_process(command, log, *, timeout=600):
    """One bounded JVM task; terminate all descendants on timeout/interruption."""
    env=dict(os.environ,JAVA_OPTS="-Xmx8g -Xms256m -XX:ActiveProcessorCount=4")
    started=time.perf_counter()
    with Path(log).open("w") as handle:
        proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            code=proc.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # A wrapper can exit before its Java descendants; clean the group.
            try:
                os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
    if code:
        raise RuntimeError(f"Joern exited {code}; inspect {log}")
    return dict(seconds=time.perf_counter()-started,
                max_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


def slice_graphs(data_dir, output, split, keys=None, batch_size=128):
    """Batch unchanged function files; persist per-file native graph caches."""
    output=Path(output)
    rows=read_jsonl(Path(data_dir)/"primevul_paired"/f"{split}.lines.jsonl")
    languages={r["sample_key"]:r["language"] for r in read_jsonl(
        Path("results/primevul_static_patterns")/f"functions.{split}.jsonl")}
    if keys:
        requested=set(keys);rows=[r for r in rows if r["pair_key"] in requested]
        if {r["pair_key"] for r in rows}!=requested:
            raise ValueError("unknown pair keys")
    if batch_size<2 or batch_size%2:
        raise ValueError("batch size must be a positive even number")
    root=output/"graphs"/split;root.mkdir(parents=True,exist_ok=True)
    batches=[];batch=[];size=0
    for i in range(0,len(rows),2):
        pair=rows[i:i+2];n=sum(len(r["func"].encode()) for r in pair)
        if batch and (len(batch)+2>batch_size or size+n>1_000_000):
            batches.append(batch);batch=[];size=0
        batch.extend(pair);size+=n
    if batch:
        batches.append(batch)
    costs=[]
    for index,batch in enumerate(batches):
        work=root/f"batch_{index:04d}";src=work/"src";src.mkdir(parents=True,exist_ok=True)
        filenames=[]
        for row in batch:
            filename=row["sample_key"].replace(":","_")+(".c" if languages[row["sample_key"]]=="c" else ".cpp")
            path=src/filename
            if path.exists() and path.read_bytes()!=row["func"].encode():
                raise ValueError(f"changed source in existing Joern cache: {path}")
            if not path.exists():
                path.write_bytes(row["func"].encode())
            filenames.append(filename)
        cache=work/"functions.jsonl"
        if cache.exists():
            existing=read_jsonl(cache)
            if [r["sample_key"] for r in existing]!=[r["sample_key"] for r in batch]:
                raise ValueError("cached Joern members differ; do not reuse a different batch size/scope")
            costs.append(json.loads((work/"cost.json").read_text()));continue
        started=time.perf_counter();cpg=work/"cpg.bin"
        parse_stats=work/"parse.cost.json"
        if not parse_stats.exists():
            stats=run_slice_process(["/home/phy/joern/joern-parse",str(src.resolve()),"--output",str(cpg.resolve())],work/"parse.log")
            if not cpg.is_file():
                raise RuntimeError("Joern did not write CPG")
            write_json(parse_stats,stats)
        else:
            stats=json.loads(parse_stats.read_text())
        export_stats=work/"export.cost.json"
        if not export_stats.exists():
            # Thousands of small CSV writes on the shared filesystem dominated
            # the measured pilot. Keep the CPG and reduced native graph, not CSV.
            with tempfile.TemporaryDirectory(prefix="primevul-slice-export-") as tmp:
                graph=Path(tmp)/"graph"
                exported=run_slice_process(["/home/phy/joern/joern-export","--repr","all","--format","neo4jcsv",
                    "--out",str(graph),str(cpg.resolve())],work/"export.log")
                result=split_slice_export(graph,batch,filenames)
                write_jsonl(cache,result)
            exported["temporary_export"]="local /tmp; CPG and reduced native evidence retained"
            write_json(export_stats,exported)
        else:
            exported=json.loads(export_stats.read_text())
            result=split_slice_export(work/"graph",batch,filenames)
            write_jsonl(cache,result)
        cost=dict(batch=index,functions=len(batch),source_bytes=sum(len(r["func"].encode()) for r in batch),
                  seconds=time.perf_counter()-started,parse=stats,export=exported,
                  statuses=dict(Counter(r["status"] for r in result)))
        write_json(work/"cost.json",cost);costs.append(cost)
        print(f"{split} Joern batch {index+1}/{len(batches)}: {len(batch)} functions, {cost['seconds']:.1f}s, {cost['statuses']}",flush=True)
    write_json(output/f"{split}.graph_cost.json",dict(functions=len(rows),batches=len(batches),costs=costs,
        java_opts="-Xmx8g -Xms256m -XX:ActiveProcessorCount=4",unchanged_source=True))


def extract_slices(data_dir, output, split):
    output=Path(output)
    if split=="valid" and json.loads((output/"rules.json").read_text())["rules"]!=SLICE_RULES:
        raise ValueError("complete Train development before fixed-rule Valid")
    if split=="valid":
        expected=read_jsonl(output/"train.review_sample.jsonl")
        reviewed=read_jsonl(output/"train.review.jsonl")
        if (len(reviewed)!=len(expected) or {r["review_id"] for r in reviewed}!={r["review_id"] for r in expected}
                or any(r["judgment"] not in {"related","unrelated","unknown"} for r in reviewed)):
            raise ValueError("complete the fixed Train source review before Valid")
    rows=read_jsonl(Path(data_dir)/"primevul_paired"/f"{split}.lines.jsonl")
    old={r["pair_key"]:r for r in csv.DictReader((OUT/"pairs.csv").open()) if r["split"]==split}
    raw=read_jsonl(RAW/f"primevul_{split}_paired.jsonl")
    if len(rows)!=2*{"train":4354,"valid":562}[split] or len(raw)!=len(rows) or any(
            r["func"]!=s["func"] or r["target"]!=s["target"] for r,s in zip(rows,raw)):
        raise ValueError("slicing requires all unchanged official Original functions and labels")
    graphs={}
    for file in sorted((output/"graphs"/split).glob("batch_*/functions.jsonl")):
        for r in read_jsonl(file):
            if r["sample_key"] in graphs:
                raise ValueError("duplicate cached graph member")
            graphs[r["sample_key"]]=r
    if set(graphs)!={r["sample_key"] for r in rows}:
        raise ValueError("complete native graphs required for every unchanged member")
    started=time.perf_counter();cost=[];result=[]
    for i in range(0,len(rows),2):
        v,p=rows[i:i+2];t=time.perf_counter()
        if [v["target"],p["target"]]!=[1,0] or v["pair_key"]!=p["pair_key"]:
            raise ValueError("official pair order differs")
        gv,gp=[statement_dependence_graph(r["func"],graphs[r["sample_key"]]) for r in (v,p)]
        meta=old[v["pair_key"]]
        pair=bidirectional_pair_slice(v["func"],p["func"],gv,gp,
            allow_mapping=meta["quality_status"]!="verified_error")
        pair.update(pair_key=v["pair_key"],split=split,shape=meta["shape"],quality_status=meta["quality_status"])
        for side,row,g in (("v",v,gv),("p",p,gp)):
            pair[side].update(sample_key=row["sample_key"],total_units=len(g["units"]),
                all_code_lines=sorted({n for u in g["units"] for n in u["lines"]}),
                edge_counts=dict(Counter(e["kind"] for e in g["edges"])))
        result.append(pair);cost.append(time.perf_counter()-t)
        if (i//2+1)%250==0:
            print(f"{split}: sliced {i//2+1}/{len(rows)//2} pairs in {time.perf_counter()-started:.1f}s",flush=True)
    write_jsonl(output/f"pairs.{split}.jsonl",result)
    write_json(output/f"{split}.slice_cost.json",dict(pairs=len(result),seconds=time.perf_counter()-started,pair_seconds=quantiles(cost)))
    if split=="train":
        write_json(output/"rules.json",dict(rules=SLICE_RULES,seed=SEED,test_used=False,
            raw_members="unchanged Original 4354/562 pairs",diff_usage="paired supervision generation only; not deployable single-function localization"))


def slicing_masks(row, result, static, prediction):
    return dict(original_diff=set(row["lines_ground_truth"]),
        static_candidates={n for r in static["relations"] for field in ("operation","evidence") for n in r[field]["lines"]},
        bidirectional_slice=set(result["region_lines"]),
        losver={n for n,p in zip(prediction["line_numbers"],prediction["line_scores"]) if p>.5})


def slicing_review_sample(source, pairs, masks, split):
    """The same fixed historical cohort for early source review and final tables."""
    rng=random.Random(SEED)
    cohort=list(csv.DictReader((OUT/"review_sample.csv").open()))
    chosen=[]
    for shape in SHAPES:
        pool=sorted(r["pair_key"] for r in cohort if r["split"]==split and r["stratum"]==shape)
        chosen+=rng.sample(pool,min(2,len(pool)))
    if not set(chosen).issubset(pairs):
        raise ValueError("all fixed review pairs must have completed extraction")
    sample=[]
    for row in source:
        if row["pair_key"] not in chosen:
            continue
        key=row["sample_key"];code=row["func"].splitlines();pair=pairs[row["pair_key"]]
        executable=lexical_code_lines(row["func"])-{i for i,line in enumerate(code) if re.fullmatch(r"[\s{}();,]+",line)}
        selected=set()
        for name,mask in masks[key].items():
            pool=sorted(mask&executable)
            if pool:
                selected.add(rng.choice(pool))
        for line in sorted(selected):
            sample.append(dict(review_id=f"{key}:line:{line}",sample_key=key,pair_key=row["pair_key"],line=line,
                target=row["target"],shape=pair["shape"],
                slice_status=pair["v" if row["target"] else "p"]["status"],quality_status=pair["quality_status"],
                methods=[name for name,mask in masks[key].items() if line in mask],code=code[line]))
    return sample,chosen


def slice_report(data_dir, output, split):
    """Fixed old review cohorts and original Diff proxies; no model or Test read."""
    output=Path(output)
    source=read_jsonl(Path(data_dir)/"primevul_paired"/f"{split}.lines.jsonl")
    rows={r["sample_key"]:r for r in source}
    pairs={r["pair_key"]:r for r in read_jsonl(output/f"pairs.{split}.jsonl")}
    static={r["sample_key"]:r for r in read_jsonl(Path("results/primevul_static_patterns")/f"functions.{split}.jsonl")}
    localizers={mode:{r["sample_key"]:r for r in read_jsonl(Path(f"results/losver_{mode}/primevul_paired/losver_localizer")/f"{split}.line_scores.jsonl")}
                for mode in ("512","l1")}
    if set(pairs)!={r["pair_key"] for r in source} or set(static)!=set(rows) or any(set(p)!=set(rows) for p in localizers.values()):
        raise ValueError("slice / old patterns / localizer members differ")
    masks={};stats=[];proxy=defaultdict(Counter);failures=Counter();path_unknown=Counter()
    for row in source:
        key=row["sample_key"];pair=pairs[row["pair_key"]];side="v" if row["target"] else "p";result=pair[side]
        if key!=result["sample_key"]:
            raise ValueError("slice side / labels differ")
        masks[key]=slicing_masks(row,result,static[key],localizers["l1"][key])
        selected=result["selected_units"]
        failures.update(result["unknown"])
        for u in selected:
            for e in u["path"]:
                path_unknown.update(e.get("unknown",[]))
        stats.append(dict(sample_key=key,pair_key=row["pair_key"],target=row["target"],shape=pair["shape"],
            quality_status=pair["quality_status"],status=result["status"],source_lines=len(row["func"].splitlines()),
            joern_status=result["parse_details"]["joern_status"],
            header_type_absent=result["parse_details"]["header_type_absent"],
            tree_sitter_errors=result["parse_details"]["tree_sitter_errors"],
            total_units=result["total_units"],selected_units=len(selected),seed_units=len(result["seed_units"]),
            identical_token_seed_units=len(result["identical_token_seed_units"]),
            diff_lines=len(row["lines_ground_truth"]),region_lines=len(result["region_lines"]),
            added_non_diff_lines=len(set(result["region_lines"])-set(row["lines_ground_truth"])),
            selected_fraction=len(selected)/result["total_units"] if result["total_units"] else 0,
            mapped_roots=result["mapped_roots"],limit_reached=result["limit_reached"],
            unknown_path_units=sum(any(e.get("certainty")=="unknown" for e in u["path"]) for u in selected),
            native_cfg=result["native"].get("CFG",0),native_ddg=result["native"].get("REACHING_DEF",0),native_cdg=result["native"].get("CDG",0),
            statement_ddg=result["edge_counts"].get("REACHING_DEF",0),statement_cdg=result["edge_counts"].get("CDG",0)))
        for mode in ("512","l1"):
            lp=localizers[mode][key];common=set(lp["line_numbers"])
            if len(common)!=len(lp["line_numbers"]) or len(common)!=len(lp["line_scores"]):
                raise ValueError("invalid original localizer coordinates")
            gt=set(row["lines_ground_truth"])&common
            selected_masks=slicing_masks(row,result,static[key],lp)
            for method,mask in selected_masks.items():
                predicted=mask&common
                for group in ("all",side,"addition_only" if pair["shape"]=="addition_only" else "other"):
                    proxy[mode,method,group].update(functions=1,covered=bool(predicted),lines=len(common),positive_lines=len(gt),
                        selected_lines=len(predicted),tp=len(predicted & gt))
    proxy_rows=[]
    for (mode,method,group),a in proxy.items():
        tp,n,p=a["tp"],a["selected_lines"],a["positive_lines"]
        proxy_rows.append(dict(split=split,mode=mode,method=method,group=group,**a,
            precision=tp/n if n else 0.,recall=tp/p if p else 0.,f1=2*tp/(n+p) if n+p else 0.))
    # Reuse all 56/24 previously inspected relation anchors, including Unknown.
    old_samples=read_jsonl(Path("results/primevul_static_patterns")/f"{split}.review_sample.jsonl")
    old_judgments={r["review_id"]:r for r in read_jsonl(Path("results/primevul_static_patterns")/f"{split}.review.jsonl")}
    statuses={r["sample_key"]:r["status"] for r in stats}
    anchors=[]
    for s in old_samples:
        if "relation" not in s:
            continue
        key=s["sample_key"];line=s["relation"]["operation"]["lines"][0]
        judgment=old_judgments[s["review_id"]]
        anchors.append(dict(review_id=s["review_id"],sample_key=key,line=line+1,
            relation_judgment=judgment["relation_judgment"],repair_relevance=judgment["repair_relevance"],
            slice_status=statuses[key],
            **{name:line in mask for name,mask in masks[key].items()}))
    old_lines=[]
    if split=="train":
        for r in csv.DictReader((OUT/"region_line_review.csv").open()):
            line=int(r["line"])-1
            old_lines.append(dict(sample_key=r["sample_key"],line=line+1,judgment=r["judgment"],
                slice_status=statuses[r["sample_key"]],
                **{name:line in mask for name,mask in masks[r["sample_key"]].items()}))
    sample,chosen=slicing_review_sample(source,pairs,masks,split)
    expected=output/f"{split}.review_sample.jsonl"
    if expected.exists() and read_jsonl(expected)!=sample:
        raise ValueError("fixed source review changed; inspect coordinate/member differences")
    write_jsonl(expected,sample)
    display=["# Fixed slicing source review", "", "0-based stored / 1-based displayed physical rows. Assistant review, not independent vulnerability gold."]
    for key in chosen:
        members=[r for r in source if r["pair_key"]==key]
        display += ["",f"## {key} / {pairs[key]['shape']} / {pairs[key]['quality_status']}","```diff"]
        diff=list(difflib.unified_diff(members[0]["func"].splitlines(),members[1]["func"].splitlines(),n=2,lineterm=""))
        display += [s[:350] for s in diff[:100]]+["```"]
        if len(diff)>100:
            display.append("Diff display truncated; source unchanged and available in original data.")
        for r in members:
            display += [f"### {r['sample_key']}","```c"]
            cases=[s for s in sample if s["sample_key"]==r["sample_key"]]
            ns={i for s in cases for i in range(max(0,s["line"]-3),s["line"]+4)}
            code=r["func"].splitlines()
            display += [f"{i+1:5}: {code[i][:450]}" for i in sorted(ns) if i<len(code)]+["```"]
            display += [json.dumps(s,ensure_ascii=False) for s in cases]
    (output/f"{split}.cases.md").write_text("\n".join(display),encoding="utf-8")
    reviewed=[]
    review_file=output/f"{split}.review.jsonl"
    if review_file.exists():
        judgments=read_jsonl(review_file)
        if (len(judgments)!=len(sample) or {r["review_id"] for r in judgments}!={r["review_id"] for r in sample}
                or any(r["judgment"] not in {"related","unrelated","unknown"} for r in judgments)):
            raise ValueError("incomplete fixed slicing source review")
        by_id={r["review_id"]:r for r in judgments}
        for method in masks[source[0]["sample_key"]]:
            for scope in ("all_selected","outside_original_diff"):
                selected=[by_id[r["review_id"]]["judgment"] for r in sample if method in r["methods"]
                          and (scope=="all_selected" or "original_diff" not in r["methods"])]
                reviewed.append(dict(method=method,scope=scope,reviewed=len(selected),**dict(Counter(selected))))
    pair_stats=[]
    for p in pairs.values():
        pair_stats.append(dict(pair_key=p["pair_key"],shape=p["shape"],quality_status=p["quality_status"],
            both_parse=p["v"]["status"]==p["p"]["status"]=="ok",mapping_allowed=p["mapping_allowed"],
            alignment_count=p["alignment_count"],v_lines=len(p["v"]["region_lines"]),p_lines=len(p["p"]["region_lines"]),
            v_seed_units=len(p["v"]["seed_units"]),p_seed_units=len(p["p"]["seed_units"]),
            recovered_v=bool(p["v"]["mapped_roots"] and p["v"]["region_lines"]),
            recovered_p=bool(p["p"]["mapped_roots"] and p["p"]["region_lines"])))
    addition=[r for r in pair_stats if r["shape"]=="addition_only"]
    summary=dict(split=split,pairs=len(pairs),functions=len(source),both_parsed=sum(r["both_parse"] for r in pair_stats),
        nonempty_v=sum(r["v_lines"]>0 for r in pair_stats),nonempty_p=sum(r["p_lines"]>0 for r in pair_stats),
        addition_pairs=len(addition),addition_recovered_v=sum(r["recovered_v"] for r in addition),
        addition_both_parsed=sum(r["both_parse"] for r in addition),
        addition_without_v_seed=sum(r["both_parse"] and r["v_seed_units"]==0 and r["p_seed_units"]>0 for r in addition),
        addition_without_v_seed_recovered=sum(r["both_parse"] and r["v_seed_units"]==0 and r["p_seed_units"]>0 and r["recovered_v"] for r in addition),
        function_status=dict(Counter(r["status"] for r in stats)),
        header_type_absent=sum(r["header_type_absent"] is True for r in stats),
        header_type_absent_and_joern_unknown=sum(r["header_type_absent"] is True and r["joern_status"]!="ok" for r in stats),
        region_lines=quantiles([r["region_lines"] for r in stats]),
        selected_units=quantiles([r["selected_units"] for r in stats]),
        limit_reached=sum(r["limit_reached"] for r in stats),
        identical_token_seed_units=sum(r["identical_token_seed_units"] for r in stats),
        parsed_functions_without_native_ddg=sum(r["status"]=="ok" and not r["native_ddg"] for r in stats),
        parsed_functions_without_mapped_dependencies=sum(r["status"]=="ok" and not (r["statement_ddg"] or r["statement_cdg"]) for r in stats),
        unknown_path_units=sum(r["unknown_path_units"] for r in stats),
        failures=dict(failures),unknown_path_reasons=dict(path_unknown),
        new_review_pairs=len(chosen),new_review_lines=len(sample),source_review=reviewed,
        test_used=False,training=False,diff_f1_is_vulnerability_accuracy=False)
    for name,table in (("functions",stats),("pairs",pair_stats),("diff_proxy",proxy_rows),("old_relation_review",anchors),("old_line_review",old_lines)):
        if table:
            write_csv(output/f"{split}.{name}.csv",table)
    write_json(output/f"{split}.summary.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"failures","unknown_path_reasons"}},ensure_ascii=False,indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("collect", "inspect", "upstream", "regions", "correspondence", "source-witnesses", "summarize", "patterns", "pattern-report", "slice-graphs", "slices", "slice-report"))
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--keys", nargs="+")
    p.add_argument("--files", type=Path)
    p.add_argument("--data-dir",type=Path,default=Path("data"))
    p.add_argument("--split",choices=("train","valid"),default="train")
    p.add_argument("--batch-size",type=int,default=128)
    args = p.parse_args()
    if args.stage in {"slice-graphs","slices","slice-report"}:
        if args.output_dir in {OUT,Path("results/primevul_static_patterns")}:
            p.error("slicing requires its own results directory; existing audits remain unchanged")
        if args.stage=="slice-graphs":
            slice_graphs(args.data_dir,args.output_dir,args.split,args.keys,args.batch_size)
        elif args.stage=="slices":
            extract_slices(args.data_dir,args.output_dir,args.split)
        else:
            slice_report(args.data_dir,args.output_dir,args.split)
    elif args.stage == "patterns":
        print(json.dumps(mine_patterns(args.data_dir,args.output_dir,args.split)),flush=True)
    elif args.stage == "pattern-report":
        pattern_report(args.data_dir,args.output_dir)
    elif args.stage == "collect":
        collect(args.output_dir)
    elif args.stage == "inspect":
        inspect(args.keys or [])
    elif args.stage == "upstream":
        upstream_records(args.output_dir)
    elif args.stage == "regions":
        regions(args.output_dir)
    elif args.stage == "correspondence":
        patch_correspondence(args.output_dir)
    elif args.stage == "source-witnesses":
        if args.files is None:
            p.error("source-witnesses requires --files: fetched original source records")
        source_witnesses(args.files,args.output_dir)
    elif args.stage == "summarize":
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
