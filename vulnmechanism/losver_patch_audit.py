"""Source/patch audit only: no model loading, member changes, or training."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import difflib
import json
from pathlib import Path
import random
import re
import time

from sklearn.metrics import roc_auc_score

from .artifacts import write_json
from .data import modified_lines, read_jsonl
from .losver_diagnosis import annotations, quantiles, selected_mass, write_csv
from .syntax import (CONTROLS, STATEMENTS, source_lines, parse_source, edit_hunks,
                     statement_units, parser_for, source_token_spans, text, walk)


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("collect", "inspect", "upstream", "regions", "correspondence", "source-witnesses", "summarize"))
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--keys", nargs="+")
    p.add_argument("--files", type=Path)
    args = p.parse_args()
    if args.stage == "collect":
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
