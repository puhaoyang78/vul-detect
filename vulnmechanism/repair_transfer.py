"""Frozen cross-sample repair-region retrieval; Train references, Train/Valid audit."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path
import random
import statistics
import time

import torch
import torch.nn.functional as F

from .artifacts import write_json
from .data import read_jsonl, write_jsonl, paired_features
from .syntax import source_token_spans


SEED = 123456
METHODS = ("positive_nn", "contrastive_nn", "context_diff_attention", "random")
CHECKPOINT = Path("results/attention_context_512/context_diff/primevul_paired/unixcoder/best-p_c.pt")


def source_identity(source):
    """Exact non-comment token identity, also excluding formatting-only copies."""
    return tuple(t[0] for t in source_token_spans(source))


def load_inputs(data_dir, audit_dir):
    """Only the two explicit official input files; never dispatch to Test readers."""
    data_dir, audit_dir = Path(data_dir), Path(audit_dir)
    rows, cohorts = {}, {}
    for split, size in (("train", 8708), ("valid", 1124)):
        current = read_jsonl(data_dir / "primevul_paired" / f"{split}.lines.jsonl")
        if len(current) != size or any(not r["sample_key"].startswith(f"primevul_paired:{split}:") for r in current):
            raise ValueError("require unchanged complete official Train/Valid members")
        for i in range(0, len(current), 2):
            v, p = current[i:i+2]
            if [v["target"], p["target"]] != [1, 0] or v["pair_key"] != p["pair_key"]:
                raise ValueError("official vulnerable/patched pair order differs")
        rows[split] = current
        by_pair = {r["pair_key"]: r for r in current if r["target"] == 1}
        patched = {r["pair_key"]: r for r in current if r["target"] == 0}
        cohort = read_jsonl(audit_dir / f"pairs.{split}.jsonl")
        if len(cohort) != 24 or len({r["pair_key"] for r in cohort}) != 24:
            raise ValueError("reuse the existing fixed 24-pair audit per split")
        for r in cohort:
            if (r["pair_key"] not in by_pair or by_pair[r["pair_key"]]["lines_ground_truth"]
                    or not patched[r["pair_key"]]["lines_ground_truth"]):
                raise ValueError("queries must be original vulnerable-side pure additions")
        cohorts[split] = cohort
    return rows, cohorts


@torch.inference_mode()
def frozen_line_batch(model, batch, device):
    """Observe the real bounded_pool output at W's input; no duplicate pooling."""
    from .training import paired_forward

    if (model.training or any(p.requires_grad for p in model.parameters())
            or model.pooling != "bounded_attention" or not model.attention_context
            or model.attention_local or model.attention_context_scale):
        raise ValueError("require frozen eval original Context + Diff")
    captured = []

    def observe(_module, args):
        captured.append(args[0].detach())

    handle = model.line_attention.register_forward_pre_hook(observe)
    try:
        probability, details = paired_forward(model, batch, "baseline", device,
            with_labels=False, return_attention=True)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape[:2] != details["line_counts"].shape:
        raise ValueError("expected one real physical-line aggregation per batch")
    return probability, captured[0], details


def reference_groups(references):
    """Global Train-only text deduplication; ambiguous positive/background text is dropped."""
    groups = defaultdict(list)
    for i, row in enumerate(references):
        if row["split"] != "train" or row["target"] != 1:
            raise ValueError("reference library must contain Train vulnerable functions only")
        key = row["code"].strip()
        if not key or row["label"] not in (0, 1):
            raise ValueError("invalid visible reference line")
        groups[key].append(i)
    conflicts = sum(len({references[i]["label"] for i in ids}) != 1 for ids in groups.values())
    accepted = [ids for ids in groups.values() if len({references[i]["label"] for i in ids}) == 1]
    return accepted, dict(raw_lines=len(references), unique_texts=len(groups),
        conflicting_texts_removed=conflicts, usable_unique_texts=len(accepted))


def allowed_references(references, groups, query):
    selected = {0: [], 1: []}
    excluded_pair = excluded_source = 0
    for ids in groups:
        for i in ids:
            r = references[i]
            if r["pair_key"] == query["pair_key"]:
                excluded_pair += 1
                continue
            if query["source_group"] is not None and r["source_group"] == query["source_group"]:
                excluded_source += 1
                continue
            selected[r["label"]].append(i)
            break
    if not selected[0] or not selected[1]:
        raise ValueError("no eligible positive or background library after isolation")
    selected[0].sort(); selected[1].sort()
    return selected, dict(excluded_pair_occurrences=excluded_pair,
        excluded_source_occurrences=excluded_source,
        positive_references=len(selected[1]), background_references=len(selected[0]))


def cosine_nearest(query, bank, indices, device):
    if not len(indices) or query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]:
        raise ValueError("nonempty compatible cosine matrices required")
    q = F.normalize(query.to(device), dim=-1)
    # Bounded memory; strict > retains the first reference on exact ties.
    best = torch.full((len(q),), -float("inf"), device=device)
    chosen = torch.full((len(q),), -1, dtype=torch.long, device=device)
    for start in range(0, len(indices), 8192):
        ids = indices[start:start+8192]
        vectors = F.normalize(bank[ids].to(device), dim=-1)
        value, offset = (q @ vectors.T).max(-1)
        take = value > best
        best = torch.where(take, value, best)
        mapped = torch.tensor(ids, device=device)[offset]
        chosen = torch.where(take, mapped, chosen)
    if not torch.isfinite(best).all() or (chosen < 0).any():
        raise ValueError("invalid nearest-neighbor result")
    return best.cpu().tolist(), chosen.cpu().tolist()


def top_lines(numbers, scores, k=3):
    if len(numbers) != len(scores) or len(set(numbers)) != len(numbers):
        raise ValueError("one score per unique physical line required")
    return sorted(range(len(numbers)), key=lambda i: (-scores[i], numbers[i]))[:k]


def rank_query(query, vectors, bank, references, groups, device, ordinal):
    allowed, isolation = allowed_references(references, groups, query)
    positive, pos_ids = cosine_nearest(vectors, bank, allowed[1], device)
    background, bg_ids = cosine_nearest(vectors, bank, allowed[0], device)
    numbers = query["line_numbers"]
    scores = dict(positive_nn=positive,
        contrastive_nn=[a-b for a, b in zip(positive, background)],
        context_diff_attention=query["line_attention"])
    orders = {name: top_lines(numbers, value) for name, value in scores.items()}
    orders["random"] = random.Random(SEED + ordinal).sample(range(len(numbers)), min(3, len(numbers)))
    methods = {}
    for method, order in orders.items():
        selected = []
        for rank, i in enumerate(order, 1):
            row = dict(rank=rank, line=numbers[i], token_count=query["line_token_counts"][i])
            if method != "random":
                row["score"] = scores[method][i]
            if method in ("positive_nn", "contrastive_nn"):
                for label, ids, values in (("positive", pos_ids, positive), ("background", bg_ids, background)):
                    ref = dict(references[ids[i]], cosine=values[i])
                    if ref["pair_key"] == query["pair_key"] or (
                            query["source_group"] is not None and ref["source_group"] == query["source_group"]):
                        raise ValueError("query/source leakage into retrieved reference")
                    row[label + "_reference"] = ref
            selected.append(row)
        methods[method] = selected
    return dict(query, methods=methods, isolation=isolation)


def metric_rows(rankings, labels, *, subset="all"):
    """Unknown stays in the denominator; missing review is an error."""
    result = []
    for split in ("train", "valid"):
        queries = [q for q in rankings if q["split"] == split]
        if not queries:
            continue
        for method in METHODS:
            for k in (1, 3):
                counts = Counter(); hits = 0
                for q in queries:
                    judged = []
                    for row in q["methods"][method][:k]:
                        key = f"{q['sample_key']}:line:{row['line']}"
                        value = labels[key]
                        if value not in {"related", "unrelated", "unknown"}:
                            raise ValueError("complete actual source review required")
                        judged.append(value)
                    counts.update(judged)
                    hits += "related" in judged
                total = sum(counts.values())
                result.append(dict(split=split, subset=subset, method=method, k=k,
                    functions=len(queries), selected_lines=total, related=counts["related"],
                    unrelated=counts["unrelated"], unknown=counts["unknown"],
                    related_fraction=counts["related"]/total, unknown_fraction=counts["unknown"]/total,
                    hit_functions=hits, hit_fraction=hits/len(queries)))
    return result


def write_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)


def experiment_protocol(checkpoint, cohorts):
    return dict(checkpoint=str(checkpoint), seed=SEED, input_mode="512", hidden_size=768,
        query_pairs={s: [r["pair_key"] for r in rows] for s, rows in cohorts.items()},
        reference_split="train", reference_target=1,
        reference_functions="at least one visible original vulnerable-side Diff line",
        positive_nn="max cosine to directly modified reference lines",
        contrastive_nn="max positive cosine - max background cosine",
        attention="actual bounded MIL line mass: token_count * rate / sum(token_count * rate)",
        line_duplicates="trimmed exact text; discard positive/background conflicts; first eligible source occurrence",
        source_duplicates="exact non-comment token sequence, including formatting-only source copies",
        ties="earlier reference occurrence, then smaller query physical line",
        random="Random(123456 + canonical query ordinal), prefix of one three-line sample",
        selection="existing 24 Train and 24 Valid source-review cohorts; Valid never builds the library",
        test_used=False, training=False, parameter_search=False)


def extract(data_dir, audit_dir, checkpoint, output_dir, device):
    from torch.utils.data import DataLoader
    from .evaluation import load_paired
    from .training import paired_collate
    from .model import resolve_device

    root, checkpoint = Path(output_dir), Path(checkpoint)
    if (root / "embeddings.pt").exists():
        raise ValueError("completed embeddings already exist; use retrieve/report, do not re-extract")
    rows, cohorts = load_inputs(data_dir, audit_dir)
    root.mkdir(parents=True, exist_ok=True)
    protocol = experiment_protocol(checkpoint, cohorts)
    write_json(root / "protocol.json", protocol)
    original_stat = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
    import json
    selected = json.loads(checkpoint.with_suffix(".json").read_text())
    cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = cp["bounded_attention"]
    if (cp["input_mode"] != "512" or not cfg.get("context") or cfg.get("local")
            or cfg.get("context_scale") or cfg["region_supervision"] != "diff"):
        raise ValueError("require original selected Context + real Diff checkpoint")
    device = resolve_device(device)
    torch.set_float32_matmul_precision("highest")
    model, tokenizer = load_paired(cp, device)
    model.requires_grad_(False); model.eval()
    if model.encoder.config.hidden_size != 768:
        raise ValueError("require actual 768-dimensional UniXcoder")
    del cp
    versions = [p._version for p in model.parameters()]
    query_info = {r["pair_key"]: dict(cohort=r["cohort"],
        static_recovered=bool(r["targeted"]["v"]["region_lines"]))
        for group in cohorts.values() for r in group}
    source_groups = {}; prepared = []; functions = []; reference_functions = 0
    started = time.perf_counter()
    for split in ("train", "valid"):
        for r in rows[split]:
            if r["target"] != 1:
                continue
            signature = source_identity(r["func"])
            if split == "train":
                group = source_groups.setdefault(signature, len(source_groups))
            else:
                group = source_groups.get(signature)
            is_query = r["pair_key"] in query_info
            if split == "valid" and not is_query:
                continue
            feature = paired_features(r, tokenizer, "baseline", input_mode="512")
            gt = set(r["lines_ground_truth"])
            is_reference = split == "train" and bool(gt & set(feature["line_numbers"]))
            if not (is_query or is_reference):
                continue
            prepared.append(feature)
            functions.append(dict(row=r, split=split, source_group=group,
                reference=is_reference, query=is_query))
            reference_functions += is_reference
        print(f"prepared {split}: {len(prepared)} selected functions", flush=True)
    loader = DataLoader(prepared, batch_size=8, shuffle=False, collate_fn=paired_collate, num_workers=0)
    vectors, references, queries, metadata = [], [], [], []
    at = offset = 0; repeat_delta = 0.; saved_probability_delta = saved_rate_delta = 0.
    reference_run = checkpoint.parent
    saved = {}
    for split, filename in (("train", "train.selected.predictions.jsonl"), ("valid", "valid.best-p_c.predictions.jsonl")):
        saved.update({r["sample_key"]: r for r in read_jsonl(reference_run / filename)})
    for batch_no, batch in enumerate(loader):
        probability, lines, details = frozen_line_batch(model, batch, device)
        if batch_no == 0:
            other, other_lines, other_details = frozen_line_batch(model, batch, device)
            repeat_delta = float((lines - other_lines).abs().max())
            if not torch.equal(lines, other_lines) or not torch.equal(probability, other):
                raise ValueError("frozen eval embedding extraction is not repeatable")
        for i in range(len(batch["line_numbers"])):
            item, feature = functions[at], prepared[at]; row = item["row"]
            numbers = feature["line_numbers"]; n = len(numbers)
            counts = details["line_counts"][i, :n].cpu().tolist()
            if numbers != batch["line_numbers"][i, :n].tolist() or not (details["line_counts"][i, :n] > 0).all():
                raise ValueError("invalid physical-line/Token mapping")
            if abs(sum(counts) - int(batch["code_mask"][i].sum())) > 1e-4:
                raise ValueError("cross-line source Token mass was not conserved")
            previous = saved[row["sample_key"]]
            if numbers != previous["line_numbers"] or counts != previous["line_token_counts"]:
                raise ValueError("line universe differs from the existing 512 Context inference")
            rates = details["line_rates"][i, :n].cpu().tolist()
            pd = abs(float(probability[i, 0]) - previous["probability"])
            rd = max(abs(a-b) for a,b in zip(rates, previous["line_rates"]))
            saved_probability_delta = max(saved_probability_delta, pd)
            saved_rate_delta = max(saved_rate_delta, rd)
            if pd > 1e-5 or rd > 1e-4:
                raise ValueError("frozen model differs from saved Context + Diff predictions")
            current = lines[i, :n].cpu().clone()
            if not torch.isfinite(current).all() or (current.norm(dim=-1) == 0).any():
                raise ValueError("nonfinite or empty real line embeddings")
            vectors.append(current)
            info = dict(sample_key=row["sample_key"], pair_key=row["pair_key"], split=item["split"],
                project=row["project"], commit_id=row["commit_id"], target=1, source_group=item["source_group"],
                source_tokens=feature["source_tokens"], source_lines=len(row["func"].splitlines()),
                truncated=feature["source_tokens"] > 510, line_numbers=numbers,
                line_token_counts=counts, line_complete=feature["line_complete"],
                line_attention=details["line_attention"][i, :n].cpu().tolist(),
                line_rates=rates, vector_start=offset, vector_end=offset+n)
            metadata.append(info)
            if item["query"]:
                queries.append(dict(info, **query_info[row["pair_key"]]))
            if item["reference"]:
                code = row["func"].splitlines(); gt = set(row["lines_ground_truth"])
                for j, number in enumerate(numbers):
                    references.append(dict(sample_key=row["sample_key"], pair_key=row["pair_key"],
                        split="train", target=1, project=row["project"], commit_id=row["commit_id"],
                        source_group=item["source_group"], line=number, token_count=counts[j],
                        line_complete=feature["line_complete"][j], label=int(number in gt),
                        code=code[number], source_tokens=feature["source_tokens"],
                        source_lines=len(code), vector_index=offset+j))
            offset += n; at += 1
        if (batch_no + 1) % 50 == 0:
            print(f"encoded {at}/{len(functions)} functions, {offset} physical lines", flush=True)
    if len(queries) != 48 or at != len(functions):
        raise ValueError("fixed cohort or embedding coverage incomplete")
    if versions != [p._version for p in model.parameters()] or any(p.grad is not None for p in model.parameters()):
        raise ValueError("frozen encoder parameters changed")
    if original_stat != (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns):
        raise ValueError("original checkpoint file changed")
    embeddings = torch.cat(vectors)
    torch.save(dict(protocol=protocol, vectors=embeddings, references=references, queries=queries), root / "embeddings.pt")
    write_jsonl(root / "functions.jsonl", metadata)
    write_jsonl(root / "queries.jsonl", queries)
    groups, counts = reference_groups(references)
    library = []
    for number, ids in enumerate(groups):
        for i in ids:
            r = references[i]
            library.append({k: r[k] for k in ("sample_key", "pair_key", "split", "project", "source_group", "source_tokens", "source_lines", "line", "token_count", "label", "vector_index")} | dict(text_group=number))
    write_csv(root / "reference_origins.csv", library)
    write_json(root / "extraction.json", dict(reference_functions=reference_functions,
        extracted_functions=len(functions), extracted_lines=len(embeddings), query_functions=len(queries),
        dimensions=embeddings.shape[1], reference_library=counts, first_batch_repeat_max_delta=repeat_delta,
        saved_probability_max_delta=saved_probability_delta, saved_rate_max_delta=saved_rate_delta,
        selected_epoch=selected["epoch"], selected_valid_p_c=selected["validation"]["p_c"],
        checkpoint_unchanged=True, model_frozen=True, eval_mode=True,
        seconds=time.perf_counter()-started, torch=torch.__version__, cuda=torch.version.cuda,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        test_used=False, training=False))
    print(f"completed extraction: {reference_functions} donor functions, {len(references)} reference lines, 48 queries", flush=True)


def retrieve(data_dir, audit_dir, checkpoint, output_dir, device):
    from .model import resolve_device

    root = Path(output_dir)
    if (root / "review.jsonl").exists():
        raise ValueError("review already exists; preserve completed judgments")
    rows, cohorts = load_inputs(data_dir, audit_dir)
    cache = torch.load(root / "embeddings.pt", map_location="cpu", weights_only=False)
    if cache["protocol"] != experiment_protocol(checkpoint, cohorts):
        raise ValueError("embedding protocol differs from the requested fixed experiment")
    device = resolve_device(device); torch.set_float32_matmul_precision("highest")
    vectors, references = cache["vectors"], cache["references"]
    bank = vectors[[r["vector_index"] for r in references]]
    groups, counts = reference_groups(references)
    rankings = []
    for ordinal, query in enumerate(cache["queries"]):
        q = vectors[query["vector_start"]:query["vector_end"]]
        rankings.append(rank_query(query, q, bank, references, groups, device, ordinal))
        if (ordinal + 1) % 8 == 0:
            print(f"retrieved {ordinal+1}/48 queries", flush=True)
    # Repeat all queries from the persisted CPU embeddings, including every isolation mask.
    for ordinal, query in enumerate(cache["queries"]):
        q = vectors[query["vector_start"]:query["vector_end"]]
        if rankings[ordinal] != rank_query(query, q, bank, references, groups, device, ordinal):
            raise ValueError("retrieval/provenance is not reproducible")
    source = {r["sample_key"]:r for split in rows.values() for r in split}
    previous = {}; previous_code = {}
    for split in ("train", "valid"):
        previous.update({r["review_id"]:r for r in read_jsonl(Path(audit_dir) / f"{split}.review.jsonl")})
        previous_code.update({r["review_id"]:r for r in read_jsonl(Path(audit_dir) / f"{split}.review_sample.jsonl")})
    sample = []
    for query in rankings:
        selected = defaultdict(list)
        for method, ranked in query["methods"].items():
            for r in ranked:
                selected[r["line"]].append(dict(method=method, rank=r["rank"]))
        for line, methods in sorted(selected.items()):
            key = f"{query['sample_key']}:line:{line}"
            code = source[query["sample_key"]]["func"].splitlines()[line]
            old = previous.get(key)
            if old and previous_code[key]["code"] != code:
                raise ValueError("historical line review source changed")
            sample.append(dict(review_id=key, sample_key=query["sample_key"], pair_key=query["pair_key"],
                split=query["split"], cohort=query["cohort"], line=line, code=code, methods=methods,
                judgment=old["judgment"] if old else "pending", note=old["note"] if old else "",
                review_origin="existing_source_review" if old else "pending"))
    write_jsonl(root / "rankings.jsonl", rankings)
    write_jsonl(root / "review_sample.jsonl", sample)
    write_jsonl(root / "review.jsonl", sample)
    write_json(root / "retrieval.json", dict(query_functions=len(rankings), review_lines=len(sample),
        reused_reviews=sum(r["judgment"]!="pending" for r in sample),
        initial_pending_reviews=sum(r["judgment"]=="pending" for r in sample),
        reference_library=counts, repeated_all_rankings_equal=True, reference_split="train", test_used=False))
    print(f"candidate union: {len(sample)} lines; {sum(r['judgment']=='pending' for r in sample)} require source review", flush=True)


def report(output_dir):
    root = Path(output_dir)
    rankings = read_jsonl(root / "rankings.jsonl")
    sample = read_jsonl(root / "review_sample.jsonl")
    review = read_jsonl(root / "review.jsonl")
    if (len(review) != len(sample) or {r["review_id"] for r in review} != {r["review_id"] for r in sample}
            or any(r["judgment"] not in {"related", "unrelated", "unknown"} or not r["note"] for r in review)):
        raise ValueError("complete every union candidate review; unreviewed lines are never negatives")
    by_id = {r["review_id"]:r for r in sample}
    for r in review:
        if any(r[k] != by_id[r["review_id"]][k] for k in ("sample_key", "pair_key", "split", "line", "code")):
            raise ValueError("review identity/source differs from the fixed candidate union")
    labels = {r["review_id"]:r["judgment"] for r in review}
    metrics = metric_rows(rankings, labels)
    subsets = metric_rows([r for r in rankings if not r["static_recovered"]], labels, subset="static_unrecovered")
    for truncated in (False, True):
        subsets += metric_rows([r for r in rankings if r["truncated"]==truncated], labels,
            subset="truncated" if truncated else "untruncated")
    projects = []
    for split in ("train", "valid"):
        for method in ("positive_nn", "contrastive_nn"):
            for k in (1, 3):
                for reference_kind in ("positive_reference", "background_reference"):
                    groups = defaultdict(list)
                    for q in rankings:
                        if q["split"] != split:
                            continue
                        for r in q["methods"][method][:k]:
                            same = r[reference_kind]["project"] == q["project"]
                            groups[same].append(labels[f"{q['sample_key']}:line:{r['line']}"])
                    total = sum(map(len, groups.values()))
                    for same, judged in sorted(groups.items()):
                        projects.append(dict(split=split, method=method, k=k, reference_kind=reference_kind,
                            same_project=same, selected_lines=len(judged), match_fraction=len(judged)/total,
                            related=judged.count("related"), unknown=judged.count("unknown"),
                            related_fraction=judged.count("related")/len(judged), unknown_fraction=judged.count("unknown")/len(judged)))
    functions = []
    for q in rankings:
        row = {k:q[k] for k in ("split", "sample_key", "pair_key", "cohort", "project", "source_tokens", "source_lines", "truncated", "static_recovered")}
        row["visible_lines"] = len(q["line_numbers"])
        for method in METHODS:
            for k in (1, 3):
                judged = [labels[f"{q['sample_key']}:line:{r['line']}"] for r in q["methods"][method][:k]]
                row[f"{method}_top{k}_related"] = judged.count("related")
                row[f"{method}_top{k}_unknown"] = judged.count("unknown")
        functions.append(row)
    lengths = []
    for split in ("train", "valid"):
        for method in METHODS:
            for k in (1, 3):
                for hit in (False, True):
                    selected = [q for q in functions if q["split"]==split and bool(q[f"{method}_top{k}_related"])==hit]
                    if selected:
                        lengths.append(dict(split=split, method=method, k=k, hit=hit, functions=len(selected),
                            source_tokens_median=statistics.median(q["source_tokens"] for q in selected),
                            source_lines_median=statistics.median(q["source_lines"] for q in selected),
                            visible_lines_median=statistics.median(q["visible_lines"] for q in selected),
                            truncated_fraction=sum(q["truncated"] for q in selected)/len(selected)))
    paired = []
    for split in ("train", "valid"):
        for subset in ("all", "static_unrecovered"):
            selected = [q for q in rankings if q["split"] == split
                        and (subset == "all" or not q["static_recovered"])]
            if not selected:
                continue
            for method in ("positive_nn", "contrastive_nn"):
                for baseline in ("context_diff_attention", "random"):
                    gained, lost = [], []
                    for q in selected:
                        a, b = [any(labels[f"{q['sample_key']}:line:{r['line']}"] == "related"
                                    for r in q["methods"][m][:3]) for m in (method, baseline)]
                        if a and not b:
                            gained.append(q["pair_key"])
                        if b and not a:
                            lost.append(q["pair_key"])
                    paired.append(dict(split=split, subset=subset, method=method, baseline=baseline,
                        k=3, functions=len(selected), gained=len(gained), lost=len(lost),
                        gained_pairs=";".join(gained), lost_pairs=";".join(lost)))
    write_csv(root / "metrics.csv", metrics)
    write_csv(root / "subsets.csv", subsets)
    write_csv(root / "project_bias.csv", projects)
    write_csv(root / "functions.quality.csv", functions)
    write_csv(root / "length_bias.csv", lengths)
    write_csv(root / "paired_hits.csv", paired)
    write_json(root / "comparison.json", dict(metrics=metrics, subsets=subsets, project_bias=projects, length_bias=lengths,
        paired_hits=paired, reviewed_lines=len(review), review_judgments=dict(Counter(labels.values())),
        source_review="assistant source/patch inspection; necessary repair context included; not independent vulnerability gold",
        unknown_in_denominator=True, valid_used_for_parameters=False, test_used=False, training=False))
    lines = ["# 跨样本修复区域迁移：冻结模型离线检索", "",
        "固定Train24/Valid24纯新增修复审计组，参考库仅来自Train漏洞侧具有可见直接Diff的函数。使用原Context+Diff所选checkpoint，eval/冻结参数，正式512输入与bounded_pool行表示；没有新训练、参数搜索或Test使用。", "",
        "直接检索=max正参考余弦；对比检索=max正参考余弦−max背景余弦。注意力对照使用实际MIL行质量，随机对照Seed123456。完全重复代码、同配对和重复参考文本按预先固定协议隔离；正/背景冲突文本排除。", "",
        "所有方法共用同一可见行集合，逐行复核Top3并集。Unknown保留在相关比例和样本命中率的分母；未复核行不会当负例。Top1是同一排序的前缀。", "",
        "related包含补丁明确约束的旧操作，以及源码可确认的数据定义和控制上下文；上下文命中不等于恢复实际危险操作。只有注释、括号、通用函数签名、日志或不同字段的检查不算操作关联。调用/类型契约不足者保留Unknown。判断来自既有与本次助手源码/补丁复核，并非独立人工漏洞定位金标。", "",
        "|Split|方法|K|相关行/所选行|相关比例|至少一行相关|Unknown比例|",
        "|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"|{r['split']}|{r['method']}|{r['k']}|{r['related']}/{r['selected_lines']}|{r['related_fraction']:.4f}|{r['hit_functions']}/{r['functions']}|{r['unknown_fraction']:.4f}|" for r in metrics]
    lines += ["", "配对Top3样本命中变化（同一查询中新命中/丢失，完整成员见[paired_hits.csv](paired_hits.csv)）：", "",
        "|Split|检索方法|对照|新增命中|丢失命中|", "|---|---|---|---:|---:|"]
    lines += [f"|{r['split']}|{r['method']}|{r['baseline']}|{r['gained']}|{r['lost']}|"
              for r in paired if r["subset"] == "all"]
    lines += ["", "静态未恢复组及完整/截断源码分层见[subsets.csv](subsets.csv)；同项目与跨项目匹配质量见[project_bias.csv](project_bias.csv)。这些是条件统计，不能视为项目或长度的因果效应。逐函数结果见[functions.quality.csv](functions.quality.csv)，命中/未命中的源码Token数、物理行数和可见行数中位数见[length_bias.csv](length_bias.csv)。", "",
        "以下列出静态未恢复且检索新增命中、注意力Top3未命中的全部候选行。仅用于事后解释，不参与检索或参数选择。表内物理行号为1起始，JSON/CSV原始行号为0起始。", "",
        "|查询配对|方法/排名|查询行|正参考来源/行|源码关系复核|", "|---|---|---|---|---|"]
    for q in rankings:
        if q["static_recovered"] or any(labels[f"{q['sample_key']}:line:{r['line']}"] == "related"
                                         for r in q["methods"]["context_diff_attention"]):
            continue
        for method in ("positive_nn", "contrastive_nn"):
            for r in q["methods"][method]:
                key = f"{q['sample_key']}:line:{r['line']}"
                if labels[key] != "related":
                    continue
                reviewed, ref = next(v for v in review if v["review_id"] == key), r["positive_reference"]
                code = reviewed["code"].strip().replace("|", "&#124;")
                note = reviewed["note"].replace("|", "&#124;")
                lines.append(f"|{q['pair_key']}|{method}/{r['rank']}|L{r['line']+1}: `{code}`|{ref['sample_key']}/L{ref['line']+1}|{note}|")
    lines += ["",
        "匹配来源与正/背景相似度：[rankings.jsonl](rankings.jsonl)。候选并集与完整源码判断：[review.jsonl](review.jsonl)。行表示和完整参考元数据缓存留本地；原始数据和最佳checkpoint不改写。", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics
