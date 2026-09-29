"""Read-only H/P cache audit and fixed stage-1 source-to-region evaluation."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import torch

from .cfg_data import (FAMILIES, REGION_SCHEMA, AttributeVocabulary, abstract_cfg,
    atomic_json, cohort_hash, digest, file_sha256, read_records, source_hash, _cyclic_nodes)
from .cfg_alignment import region_candidates
from .cfg_dependency import RegionReconstructionHead, _checked_region_rows

SMOOTHING = 1e-8  # Additive probability smoothing; fixed before inspecting valid.


def _stream(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_inputs(pretrain_dir, base=None):
    """Validate frozen identities, never rebuild a partition or inspect test labels."""
    from .cfg_experiment import _base_module, _tokenizer
    stage = Path(pretrain_dir).resolve()
    config = json.loads((stage / "config.json").read_text())
    c = config["c_config"]
    reference = Path(config["reference_run_dir"])
    root = Path(config["region_dir"])
    audit = json.loads((root / "audit.json").read_text())
    vocab = AttributeVocabulary(json.loads((reference / "vocabulary.json").read_text()))
    if (json.loads((reference / "config.json").read_text()) != c or audit["c_config"] != c or
            audit["reference_run_dir"] != str(reference.resolve()) or
            config.get("objective") != "source_clm_plus_dependency_plus_region" or
            config.get("region_schema") != REGION_SCHEMA or audit["region_schema"] != REGION_SCHEMA or
            c["source_max_length"] != 2048 or c["seed"] != 42 or
            digest(vocab.values) != c["vocabulary_sha256"] or
            vocab.sizes() != audit["vocabulary_sizes"] or
            config["regions_sha256"] != audit["regions_sha256"] or
            config["region_targets_sha256"] != audit["train_targets_sha256"] or
            file_sha256(root / "regions.jsonl") != audit["regions_sha256"] or
            file_sha256(c["graphs"]) != c["graph_file_sha256"]):
        raise ValueError("region diagnostics cache/config/vocabulary identity mismatch")
    rows = read_records(c["dataset"], c["source_dataset"], splits={"train", "valid"})
    records = {split: [r for r in rows if r["split"] == split] for split in ("train", "valid")}
    for split, subset in records.items():
        if cohort_hash(subset) != c[f"{split}_cohort_sha256"]:
            raise ValueError("region diagnostics train/valid cohort changed")
    expected = {r["sample_key"]: r for r in rows}
    structures = {}
    for item in _stream(root / "regions.jsonl"):
        if item["split"] not in records:
            continue
        key = item["sample_key"]
        if (key not in expected or key in structures or item["split"] != expected[key]["split"] or
                item["source_sha256"] != source_hash(expected[key]["raw_source"])):
            raise ValueError("region diagnostics structure source/split mismatch")
        structures[key] = item
    if set(structures) != set(expected):
        raise ValueError("region diagnostics missing train/valid structures")
    # The persisted whole-file identity above authenticates the partition. Do not
    # call load_regions/cfg_regions here: those would construct partitions again.
    base = _base_module() if base is None else base
    builder = _tokenizer(base, c)
    partitions = {k: v["region"] for k, v in structures.items()}
    supervision = {split: _checked_region_rows(root, subset, builder, partitions, split)[0]
                   for split, subset in records.items()}
    return config, audit, vocab, records, structures, supervision, builder, base


def target_metrics(target, log_prediction):
    """Natural-log CE, entropy and KL, retaining the sparse target support."""
    if not target:
        return None
    indices = torch.tensor([k-2 for k, _ in target], device=log_prediction.device)
    weights = torch.tensor([p for _, p in target], dtype=torch.float64,
                           device=log_prediction.device)
    values = log_prediction.detach().double()[indices]
    ce = -(weights * values).sum().item()
    entropy = -(weights * weights.log()).sum().item()
    return {"ce": ce, "entropy": entropy, "kl": ce-entropy}


def constant_reference(records, supervision, sizes):
    """Train-only family means weighted by 1/(regions * valid families).

    The common effective-function denominator cancels on family normalization.
    This is the constant minimizer of the original hierarchical objective before
    the fixed additive smoothing, not an operation-weighted category histogram.
    """
    if any(r["split"] != "train" for r in records):
        raise ValueError("constant region reference is train-only")
    if set(supervision) - {r["sample_key"] for r in records}:
        raise ValueError("constant reference contains non-train functions")
    totals = [torch.zeros(size-2, dtype=torch.float64) for size in sizes]
    masses = [0.] * 4
    for row in records:
        regions = supervision.get(row["sample_key"], [])
        for region in regions:
            families = sum(bool(t) for t in region["targets"])
            if not families:
                raise ValueError("empty supervised region")
            weight = 1 / (len(regions) * families)
            for i, target in enumerate(region["targets"]):
                if not target:
                    continue
                masses[i] += weight
                for category, probability in target:
                    totals[i][category-2] += weight * probability
    return [((t / mass + SMOOTHING) / (1 + SMOOTHING * len(t))) if mass else None
            for t, mass in zip(totals, masses)]


def summarize_predictions(rows):
    """Order/batch independent family -> region -> effective-function means.

    Group summaries restrict regions/families first and renormalize within each
    eligible function. Per-family summaries use the same conditional hierarchy.
    """
    groups = {"all": lambda r, f: True,
              "single_operation": lambda r, f: r["operations"] == 1,
              "multiple_operations": lambda r, f: r["operations"] > 1,
              "single_category": lambda r, f: len(f["target"]) == 1,
              "multiple_categories": lambda r, f: len(f["target"]) > 1}
    groups.update({name: (lambda r, f, name=name: f["family"] == name) for name in FAMILIES})
    result = {}
    function_regions = Counter(r["sample_key"] for r in rows)
    for group, keep in groups.items():
        by_function = defaultdict(list)
        family_count = 0
        for row in rows:
            families = [f for f in row["families"] if keep(row, f)]
            if not families:
                continue
            family_count += len(families)
            region = {model: {metric: math.fsum(f[model][metric] for f in families)/len(families)
                             for metric in ("ce", "entropy", "kl")}
                      for model in ("fixed_model", "constant")}
            by_function[row["sample_key"]].append(region)
        summary = {"functions": len(by_function), "regions": sum(map(len, by_function.values())),
                   "family_targets": family_count}
        for model in ("fixed_model", "constant"):
            summary[model] = {metric: math.fsum(
                math.fsum(r[model][metric] for r in regions)/len(regions)
                for regions in by_function.values())/len(by_function) if by_function else None
                for metric in ("ce", "entropy", "kl")}
        if group in FAMILIES and function_regions:
            # Preserve original objective weights as an additional decomposable
            # family contribution (sum of four contributions equals overall).
            summary["objective_contribution"] = {model: {metric: math.fsum(
                f[model][metric] / (len(row["families"]) * function_regions[row["sample_key"]])
                for row in rows for f in row["families"] if f["family"] == group
            ) / len(function_regions) for metric in ("ce", "entropy", "kl")}
                for model in ("fixed_model", "constant")}
        result[group] = summary
    return result


def fixed_predictions(encoder, head, builder, records, supervision, reference,
                      batch_size, device, output=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected = [r for r in records if supervision.get(r["sample_key"])]
    encoder.eval()
    head.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start+batch_size]
            ids, mask = builder.sequence_batch(batch, variant="baseline", excluded_groups=(), device=device)
            hidden = encoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            for record, states in zip(batch, hidden):
                for region in supervision[record["sample_key"]]:
                    # Only source states and cached token positions enter the head.
                    logits = head(states, region["tokens"])
                    families = []
                    for i, (prediction, target) in enumerate(zip(logits, region["targets"])):
                        if not target:
                            continue
                        if prediction is None or reference[i] is None:
                            raise ValueError("target family lacks a model head or train reference")
                        logp = prediction.double().log_softmax(-1)
                        const_logp = reference[i].to(device=logp.device).log()
                        top = logp.topk(min(5, logp.numel()))
                        const_top = const_logp.topk(min(5, const_logp.numel()))
                        families.append({"family": FAMILIES[i], "target": target,
                            "fixed_model": target_metrics(target, logp),
                            "constant": target_metrics(target, const_logp),
                            "target_log_probabilities": [[k, logp[k-2].item()] for k, _ in target],
                            "prediction_top": [[k+2, p] for k, p in zip(top.indices.tolist(), top.values.exp().tolist())],
                            "constant_top": [[k+2, p] for k, p in zip(const_top.indices.tolist(), const_top.values.exp().tolist())]})
                    item = {"sample_key": record["sample_key"], "split": record["split"],
                            "region": region["region"], "operations": len(region["tokens"]),
                            "nodes": region["nodes"], "tokens": region["tokens"], "spans": region["spans"],
                            "source_operations": [record["raw_source"][a:b] for a, b in region["spans"]],
                            "families": families}
                    predictions.append(item)
                    if output is not None:
                        output.write(json.dumps(item, ensure_ascii=False) + "\n")
            if output is not None:
                output.flush()
            if (start//batch_size+1) % 100 == 0 or start+batch_size >= len(selected):
                print(f"fixed_region_eval={start+len(batch)}/{len(selected)}", flush=True)
    return predictions


def _scope(regions, total_regions=None, total_functions=None):
    operations = Counter(len(r["tokens"]) for r in regions)
    if total_regions is not None:
        operations[0] += total_regions-len(regions)
    families = {}
    count = total_regions if total_regions is not None else len(regions)
    for i, family in enumerate(FAMILIES):
        targets = [r["targets"][i] for r in regions if r["targets"][i]]
        entropies = [-sum(p*math.log(p) for _, p in t) for t in targets]
        single = sum(len(t) == 1 for t in targets)
        families[family] = {"valid_targets": len(targets),
            "effective_functions": len({r.get("sample_key") for r in regions if r["targets"][i]}),
            "empty_target_regions": count-len(targets),
            "target_coverage": len(targets)/count if count else None,
            "single_category": single,
            "single_category_fraction": single/len(targets) if targets else None,
            "multiple_categories": len(targets)-single,
            "multiple_category_fraction": (len(targets)-single)/len(targets) if targets else None,
            "distinct_distributions": len({tuple(map(tuple, t)) for t in targets}),
            "mean_target_entropy": math.fsum(entropies)/len(entropies) if entropies else None,
            "max_target_entropy": max(entropies, default=None)}
    return {"regions": count,
            "functions": total_functions if total_functions is not None else len({r.get("sample_key") for r in regions}),
            "operations": sum(k*v for k, v in operations.items()),
            "operation_histogram": dict(sorted(operations.items())),
            "zero_operation": operations[0], "single_operation": operations[1],
            "multiple_operations": sum(v for k, v in operations.items() if k > 1),
            "families": families}


def audit_regions(pretrain_dir, output_dir, base=None):
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"region diagnostic output exists: {root}")
    config, original, vocab, records, structures, supervision, builder, _ = load_inputs(pretrain_dir, base)
    by_key = {r["sample_key"]: r for subset in records.values() for r in subset}
    candidates = {s: [] for s in records}
    sampled = {s: [] for s in records}
    stats = {s: Counter() for s in records}
    size_counts = {s: Counter() for s in records}
    examples = {s: {"zero_operation": [], "single_operation": [], "multiple_operations": []} for s in records}
    seen = set()
    for cached in _stream(config["c_config"]["graphs"]):
        key = cached["sample_key"]
        if key not in by_key:
            continue
        if key in seen:
            raise ValueError("duplicate cached graph")
        seen.add(key)
        row, item = by_key[key], structures[key]
        split = row["split"]
        view = abstract_cfg(cached["graph"])
        if view["node_ids"] != item["node_ids"] or [list(e) for e in view["edges"]] != item["cfg_edges"]:
            raise ValueError("persisted CFG does not match existing graph")
        partition = item["region"]
        encoded = vocab.encode(view)
        available, counts = region_candidates(row, view, encoded, partition, builder)
        mapping = {r["region"]: r for r in available}
        selected = supervision[split].get(key, [])
        if any(r != mapping.get(r["region"]) for r in selected):
            raise ValueError("saved selected targets differ from read-only candidates")
        candidates[split].extend(dict(r, sample_key=key) for r in available)
        sampled[split].extend(dict(r, sample_key=key) for r in selected)
        c = stats[split]
        c.update(counts)
        for scope, targets in (("eligible", available), ("sampled", selected)):
            for target in targets:
                for node in target["nodes"]:
                    for family, value in zip(FAMILIES, encoded[node]):
                        kind = "empty" if value == 0 else "unk" if value == 1 else "known"
                        c[f"{scope}_{family}_{kind}_operations"] += 1
        n, nr = len(view["node_ids"]), len(partition["members"])
        c.update(functions=1, nodes=n, regions=nr, selected_regions=len(selected),
                 selected_operations=sum(len(r["tokens"]) for r in selected))
        successors, predecessors = [set() for _ in range(n)], [set() for _ in range(n)]
        for a, b in view["edges"]:
            successors[a].add(b)
            predecessors[b].add(a)
        cyclic = _cyclic_nodes(successors, predecessors)
        c["cyclic_nodes"] += len(cyclic)
        c["cyclic_nodes_in_singletons"] += sum(len(partition["members"][partition["node_to_region"][node]]) == 1 for node in cyclic)
        size_counts[split].update(map(len, partition["members"]))
        for region_index, members in enumerate(partition["members"]):
            target = mapping.get(region_index)
            bucket = "zero_operation" if target is None else "single_operation" if len(target["tokens"]) == 1 else "multiple_operations"
            if len(examples[split][bucket]) < 2:
                examples[split][bucket].append({"sample_key": key, "region": region_index,
                    "members": members, "node_ids": [view["node_ids"][j] for j in members],
                    "member_source": [view["locations"][j]["code"] for j in members],
                    "raw_source": row["raw_source"], "selected": any(r["region"] == region_index for r in selected),
                    "target": target, "target_categories": None if target is None else {
                        f: [[vocab.values[f][category], p] for category, p in t]
                        for f, t in zip(FAMILIES, target["targets"])}})
        if len(seen) % 200 == 0:
            print(f"region_diagnostic_audit={len(seen)}/{len(by_key)}", flush=True)
    if seen != set(by_key):
        raise ValueError("missing cached train/valid graphs")
    report = {"pretrain_dir": str(Path(pretrain_dir).resolve()), "region_dir": config["region_dir"],
              "scope": "train/valid only; fixed cached partitions and selected targets",
              "effective_operation": "complete visible, known nonempty attribute, deduplicated operation used by P",
              "all_region_targets": "visible eligible targets only; no invisible graph attributes are answers",
              "original_audit": str(Path(config["region_dir"]) / "audit.json"), "splits": {}}
    for split, c in stats.items():
        old = original["splits"][split]
        for name in ("nodes", "regions", "supervisable_functions", "supervisable_regions", "selected_regions", "selected_operations"):
            if c[name] != old[name]:
                raise ValueError(f"read-only audit differs from original {split} {name}")
        sizes = size_counts[split]
        family_nodes = {f: {"known": c[f"{f}_known_nodes"], "empty": c[f"{f}_empty_nodes"],
            "unk": c[f"{f}_unk_nodes"], "unk_fraction_all_nodes": c[f"{f}_unk_nodes"]/c["nodes"],
            "unk_fraction_nonempty": c[f"{f}_unk_nodes"]/(c[f"{f}_known_nodes"]+c[f"{f}_unk_nodes"])
                if c[f"{f}_known_nodes"]+c[f"{f}_unk_nodes"] else None}
            for f in FAMILIES}
        report["splits"][split] = {"counts": dict(c), "family_node_attributes": family_nodes, "size_histogram": dict(sorted(sizes.items())),
            "singleton_regions": sizes[1], "multinode_regions": sum(v for k, v in sizes.items() if k > 1),
            "regions_per_node": c["regions"]/c["nodes"], "node_reduction": 1-c["regions"]/c["nodes"],
            "cyclic_singleton_fraction": c["cyclic_nodes_in_singletons"]/c["cyclic_nodes"] if c["cyclic_nodes"] else None,
            "all": _scope(candidates[split], c["regions"], c["functions"]), "eligible": _scope(candidates[split]),
            "sampled": _scope(sampled[split]), "examples": examples[split]}
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "audit.json", report)
    return report


def evaluate_regions(pretrain_dir, output_dir, batch_size=1, device="auto", base=None):
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"fixed region output exists: {root}")
    config, audit, vocab, records, _, supervision, builder, base = load_inputs(pretrain_dir, base)
    checkpoint_path = Path(pretrain_dir) / "region_pretrain" / "last.pt"
    complete = json.loads((checkpoint_path.parent / "complete.json").read_text())
    checksum = file_sha256(checkpoint_path)
    if complete["mode"] != "region_pretrain" or complete["checkpoint_sha256"] != checksum:
        raise ValueError("fixed P1 checkpoint identity changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("mode") != "region_pretrain" or checkpoint.get("pretrain_config") != config or
            "region_head_state" not in checkpoint or "adapter_state" not in checkpoint):
        raise ValueError("fixed P1 checkpoint lacks matching stage-1 LoRA/region head")
    reference = constant_reference(records["train"], supervision["train"], vocab.sizes())
    resolved = base._resolve_device(device)
    c = config["c_config"]
    encoder, hidden_size = base._build_lora_encoder(c["model_path"], device=resolved,
        lora_r=c["lora_r"], lora_alpha=c["lora_alpha"], lora_dropout=c["lora_dropout"],
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"), gradient_checkpointing=False)
    encoder.to(resolved)
    base.set_peft_model_state_dict(encoder, checkpoint["adapter_state"])
    head = RegionReconstructionHead(hidden_size, vocab.sizes()).to(resolved)
    head.load_state_dict(checkpoint["region_head_state"])
    encoder.requires_grad_(False)
    head.requires_grad_(False)
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "constant_reference.json", {"fit_split": "train", "smoothing": SMOOTHING,
        "formula": "(weighted_mean + epsilon)/(1 + epsilon * known_categories)",
        "weights": "1 / (selected_regions_in_function * valid_families_in_region)",
        "probabilities": {f: p.tolist() if p is not None else None for f, p in zip(FAMILIES, reference)}})
    # Save the original category names so top items in JSONL are interpretable.
    atomic_json(root / "vocabulary.json", vocab.values)
    result = {"pretrain_dir": str(Path(pretrain_dir).resolve()), "region_dir": config["region_dir"],
        "checkpoint_sha256": checksum, "batch_size": batch_size, "device": str(resolved),
        "source_max_length": c["source_max_length"], "regions_sha256": audit["regions_sha256"],
        "target_cache_sha256": {s: audit[f"{s}_targets_sha256"] for s in ("train", "valid")},
        "evaluation": "fixed stage-1 eval/no_grad; natural-log units",
        "aggregation": "valid families -> selected regions -> effective functions",
        "family_and_group_aggregation": "restrict group then renormalize within each eligible function",
        "online_training_summary": checkpoint.get("summary"), "splits": {}}
    for split in ("train", "valid"):
        with (root / f"{split}.regions.jsonl").open("x", encoding="utf-8") as output:
            predictions = fixed_predictions(encoder, head, builder, records[split], supervision[split],
                                            reference, batch_size, resolved, output)
        result["splits"][split] = summarize_predictions(predictions)
        atomic_json(root / "metrics.json", {**result, "complete": False})
    result["complete"] = True
    atomic_json(root / "metrics.json", result)
    return result
