"""CFG ablations, source supervision, train-negative rotation, and source pretraining in one flow.

Examples are in docs/CFG_ABLATION.md. Training never evaluates the test split.
The baseline behavior, datasets, and old checkpoints are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import importlib
import json
import math
from pathlib import Path
import random
import sys

from .cfg_alignment import align_nodes, coverage_summary
from .cfg_data import (FEATURE_SCHEMA, AttributeVocabulary, abstract_cfg, atomic_json,
                       build_graphs, cohort_hash, digest, file_sha256, identity, load_graphs,
                       output_lock, read_jsonl, read_records)
from .cfg_metrics import decision_boundary, metrics, paired_changes, select_threshold
from .cfg_rotation import ROTATION_VARIANTS, epoch_train_rows, load_selected, schedule_report

JK_VARIANTS = ("cfg_jk_mean", "cfg_jk_max")
SOURCE_SUPERVISION_VARIANTS = ("cfg_source_aux", "cfg_source_detach")
PROGRAM_VARIANTS = ("dep_pretrain_program_plain", "dep_pretrain_program_state",
                    "composition_pretrain_program_state")
REGION_CONTEXT_VARIANTS = ("region_local", "region_context")
REGION_VARIANTS = (*REGION_CONTEXT_VARIANTS, "dep_pretrain_hierarchical", "region_pretrain_cfg", "region_pretrain_hierarchical")
PRETRAIN_CFG_VARIANTS = (*REGION_VARIANTS, "lm_pretrain_cfg", "dep_pretrain_cfg", *PROGRAM_VARIANTS,
                         "composition_pretrain_cfg")
PRETRAIN_MODE = {"region_local": "dep_pretrain", "region_context": "dep_pretrain", "dep_pretrain_hierarchical": "dep_pretrain",
                 "region_pretrain_cfg": "region_pretrain",
                 "region_pretrain_hierarchical": "region_pretrain", "lm_pretrain_cfg": "lm_pretrain", "dep_pretrain_cfg": "dep_pretrain",
                 "dep_pretrain_program_plain": "dep_pretrain",
                 "dep_pretrain_program_state": "dep_pretrain",
                 "composition_pretrain_cfg": "composition_pretrain",
                 "composition_pretrain_program_state": "composition_pretrain"}
VARIANTS = ("baseline", "attributes", "cfg", "aligned_attributes", "aligned_cfg",
            "cfg_ddg", "cfg_ddg_shuffled", "cfg_double_ce", "cfg_rdrop",
            *JK_VARIANTS, *SOURCE_SUPERVISION_VARIANTS, *ROTATION_VARIANTS,
            *PRETRAIN_CFG_VARIANTS)
DDG_VARIANTS = ("cfg_ddg", "cfg_ddg_shuffled")
DUAL_FORWARD_ALPHA = {"cfg_double_ce": 0.0, "cfg_rdrop": 1.0}
READOUT_VARIANTS = ("linear", "mlp", "interaction")
COMPARISON_VARIANTS = (*VARIANTS, *READOUT_VARIANTS)
DEFAULT_VARIANTS = VARIANTS[:3]
CHECKPOINT_SCHEMA = 1


def _base_module():
    try:
        return importlib.import_module("vulnmechanism.model")
    except ModuleNotFoundError as exc:
        raise RuntimeError("training needs the repository's existing requirements.txt "
                           "(torch, transformers, peft, tree-sitter, tqdm); no DGL is needed") from exc


def _tokenizer(base, config):
    tok = base.AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        raise ValueError("tokenizer has neither pad token nor EOS token")
    return base.InputBuilder(tok, source_max_length=config["source_max_length"],
                             context_max_length=384)


def _graph_inputs(model, batch, builder, views, encoded, device, coverage=None):
    from .cfg_network import collate_graphs
    aligned = model.task_modules["cfg_encoder"].mode.startswith("aligned_")
    if aligned:
        ids, mask, offsets = builder.source_alignment_batch(batch, device=device)
        alignments = []
        for record, source_offsets in zip(batch, offsets):
            view = views[record["sample_key"]]
            pairs, counts = align_nodes(record["raw_source"], view["locations"], source_offsets,
                                        len(builder.source_prefix))
            alignments.append(pairs)
            if coverage is not None:
                coverage.update(counts)
    else:
        ids, mask = builder.sequence_batch(batch, variant="baseline", excluded_groups=(), device=device)
        alignments = None
    keys = [r["sample_key"] for r in batch]
    mode = model.task_modules["cfg_encoder"].mode
    ddg_edges = ([views[k]["ddg_shuffled_edges" if mode == "cfg_ddg_shuffled" else "ddg_edges"]
                  for k in keys] if mode in DDG_VARIANTS else None)
    programs = [views[k]["program"] for k in keys] if mode in {"program_plain", "program_state"} else None
    graph_batch = collate_graphs([views[k] for k in keys], [encoded[k] for k in keys],
                                 device=device, alignments=alignments, ddg_edges=ddg_edges,
                                 programs=programs,
                                 regions=[views[k]["regions"] for k in keys] if mode in {"cfg_hierarchical", *REGION_CONTEXT_VARIANTS} else None)
    return ids, mask, graph_batch


def source_supervision_loss(source_logits, graph_logits, labels, *, detach_source: bool):
    """Unit-weight source BCE plus fused BCE; detach only the latter's source path."""
    import torch
    source_bce = torch.nn.functional.binary_cross_entropy_with_logits(source_logits.float(), labels.float())
    fused_logits = (source_logits.detach() if detach_source else source_logits) + graph_logits
    fusion_bce = torch.nn.functional.binary_cross_entropy_with_logits(fused_logits.float(), labels.float())
    return source_bce + fusion_bce, source_bce, fusion_bce


def _graph_scores(model, rows, builder, views, encoded, *, batch_size, device, coverage=None,
                  include_source=False):
    import torch
    model.eval()
    values, source_values = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            inputs = _graph_inputs(model, rows[start:start+batch_size], builder, views, encoded, device,
                                   coverage=coverage)
            if include_source:
                source_logits, graph_logits = model.branch_logits(*inputs)
                logits = source_logits + graph_logits
                if not torch.isfinite(source_logits).all():
                    raise ValueError("non-finite source prediction")
                source_values.extend(torch.sigmoid(source_logits.float()).cpu().tolist())
            else:
                logits = model(*inputs)
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite graph prediction")
            values.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return (values, source_values) if include_source else values


def _save_torch(path, checkpoint):
    import torch
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    torch.save(checkpoint, partial)
    partial.replace(path)


def _train_graph(base, config, rows, views, vocab, folder, device, *, epoch_rows=None,
                 initial_adapter_state=None):
    import torch
    from .cfg_network import build_model
    from tqdm.auto import tqdm

    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "valid"]
    train_epochs = epoch_rows if epoch_rows is not None else [train] * config["epochs"]
    if len(train_epochs) != config["epochs"] or any(len(group) != len(train) for group in train_epochs):
        raise ValueError("each training epoch must retain the original C sample count")
    train_views = {r["sample_key"]: views[r["sample_key"]]
                   for group in [*train_epochs, valid] for r in group}
    encoded = {k: vocab.encode(v) for k, v in train_views.items()}
    builder = _tokenizer(base, config)
    base._seed_everything(config["seed"])
    model = build_model(base, config, vocab.sizes(), device, training=True)
    if initial_adapter_state is not None:
        base.set_peft_model_state_dict(model.encoder, initial_adapter_state)
    graph_parameters = list(model.task_modules["cfg_encoder"].parameters()) + list(model.task_modules["cfg_classifier"].parameters())
    graph_ids = {id(p) for p in graph_parameters}
    source_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in graph_ids]
    trainable = source_parameters + graph_parameters
    optimizer = torch.optim.AdamW([
        {"params": source_parameters, "lr": config["learning_rate"]},
        {"params": graph_parameters, "lr": config["graph_learning_rate"]},
    ], weight_decay=config["weight_decay"])
    best_key = (-float("inf"),) * 4
    best_scores, best_threshold = None, None
    best_source_scores, best_source_threshold = None, None
    aligned = config["variant"].startswith("aligned_")
    source_supervision = config["variant"] in SOURCE_SUPERVISION_VARIANTS
    dual_alpha = DUAL_FORWARD_ALPHA.get(config["variant"])
    if dual_alpha is not None:
        from .rdrop import binary_rdrop_loss
    coverage = {"train": Counter(), "valid": Counter()} if aligned else None
    step = 0
    batch_size, accumulation = config["batch_size"], config["gradient_accumulation"]
    with (folder / "history.jsonl").open("x", encoding="utf-8") as history:
        def log(row):
            history.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            history.flush()
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

        if config["variant"] in REGION_VARIANTS:
            log({"event": "parameters", "graph_encoder": sum(p.numel() for p in model.task_modules["cfg_encoder"].parameters()),
                 "graph_classifier": sum(p.numel() for p in model.task_modules["cfg_classifier"].parameters()),
                 "trainable_total": sum(p.numel() for p in trainable)})
        if config["variant"] in REGION_CONTEXT_VARIANTS:
            work = {}
            for split, records in (("train", train), ("valid", valid)):
                graphs = [views[r["sample_key"]] for r in records]
                n = sum(len(g["node_ids"]) for g in graphs)
                r = sum(len(g["regions"]["members"]) for g in graphs)
                e = sum(sum(a != b for a, b in g["edges"]) for g in graphs)
                cross = sum(sum(a != b for a, b in g["regions"]["edges"]) for g in graphs)
                work[split] = {"nodes": n, "regions": r, "node_updates": 5*n,
                    "region_updates": 3*r, "node_edge_messages": 5*e,
                    "region_edge_messages": 3*cross if config["variant"] == "region_context" else 0,
                    "self_messages": 5*n+3*r, "pool_gate_rows": 2*n,
                    "projection_rows": n}
            log({"event": "graph_compute", "node_steps": 5, "region_steps": 3,
                 "additional_parameters": config["graph_hidden_size"]**2, "splits": work})
        for epoch in range(config["epochs"]):
            model.train()
            epoch_train = train_epochs[epoch]
            order = list(range(len(epoch_train)))
            random.Random(config["seed"] + epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            num_batches = math.ceil(len(order)/batch_size)
            total_loss, window_loss, window_samples = 0.0, 0.0, 0
            total_bce = total_kl = window_bce = window_kl = 0.0
            total_source_bce = total_fusion_bce = window_source_bce = window_fusion_bce = 0.0
            optimizer_steps = 0
            bar = tqdm(range(0, len(order), batch_size), total=num_batches,
                       desc=f"{config['variant']} epoch {epoch+1}/{config['epochs']}",
                       file=sys.stdout, dynamic_ncols=True, mininterval=1)
            for batch_index, start in enumerate(bar):
                batch = [epoch_train[i] for i in order[start:start+batch_size]]
                inputs = _graph_inputs(model, batch, builder, views, encoded, device,
                                       coverage=coverage["train"] if aligned and epoch == 0 else None)
                labels = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=device)
                if source_supervision:
                    source_logits, graph_logits = model.branch_logits(*inputs)
                    loss, source_bce, fusion_bce = source_supervision_loss(
                        source_logits, graph_logits, labels,
                        detach_source=config["variant"] == "cfg_source_detach")
                elif dual_alpha is None:
                    logits = model(*inputs)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                else:
                    # Reuse the exact same source/CFG tensors; dropout consumes fresh RNG state.
                    logits = model(*inputs)
                    second_logits = model(*inputs)
                    loss, bce, kl = binary_rdrop_loss(logits, second_logits, labels, alpha=dual_alpha)
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite loss in epoch {epoch+1}, batch {batch_index+1}")
                # Identical sample-weighted partial accumulation windows to model.py.
                group_start = (batch_index // accumulation) * accumulation
                group_end = min(group_start + accumulation, num_batches)
                group_samples = min(group_end*batch_size, len(order)) - group_start*batch_size
                (loss * len(batch)/group_samples).backward()
                value = float(loss.detach().cpu())
                total_loss += value*len(batch)
                window_loss += value*len(batch)
                window_samples += len(batch)
                if dual_alpha is not None:
                    bce_value, kl_value = float(bce.detach().cpu()), float(kl.detach().cpu())
                    total_bce += bce_value*len(batch)
                    total_kl += kl_value*len(batch)
                    window_bce += bce_value*len(batch)
                    window_kl += kl_value*len(batch)
                if source_supervision:
                    source_value = float(source_bce.detach().cpu())
                    fusion_value = float(fusion_bce.detach().cpu())
                    total_source_bce += source_value*len(batch)
                    total_fusion_bce += fusion_value*len(batch)
                    window_source_bce += source_value*len(batch)
                    window_fusion_bce += fusion_value*len(batch)
                if (batch_index+1) % accumulation == 0 or batch_index+1 == num_batches:
                    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    if not torch.isfinite(norm):
                        raise ValueError("non-finite gradient norm")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    step += 1
                    if step == 1 or step % config["log_every"] == 0 or batch_index+1 == num_batches:
                        log(dict(event="step", epoch=epoch+1, step=step, loss=window_loss/window_samples,
                                 source_lr=config["learning_rate"], graph_lr=config["graph_learning_rate"],
                                 **({"bce": window_bce/window_samples, "kl": window_kl/window_samples,
                                     "alpha": dual_alpha} if dual_alpha is not None else {}),
                                 **({"source_bce": window_source_bce/window_samples,
                                     "fusion_bce": window_fusion_bce/window_samples}
                                    if source_supervision else {})))
                        window_loss, window_samples = 0.0, 0
                        window_bce = window_kl = 0.0
                        window_source_bce = window_fusion_bce = 0.0
                bar.set_postfix(loss=f"{value:.4f}", step=step, refresh=False)
            validation_scores = _graph_scores(
                model, valid, builder, views, encoded, batch_size=batch_size, device=device,
                coverage=coverage["valid"] if aligned and epoch == 0 else None,
                include_source=source_supervision)
            scores, source_scores = validation_scores if source_supervision else (validation_scores, None)
            threshold, validation = select_threshold([r["label"] for r in valid], scores)
            source_threshold, source_validation = (select_threshold([r["label"] for r in valid], source_scores)
                                                   if source_supervision else (None, None))
            if dual_alpha is not None:
                training_loss = {"bce": total_bce/len(train), "kl": total_kl/len(train),
                                 "alpha": dual_alpha}
            elif source_supervision:
                training_loss = {"source_bce": total_source_bce/len(train),
                                 "fusion_bce": total_fusion_bce/len(train)}
            else:
                training_loss = None
            log(dict(event="epoch", variant=config["variant"], epoch=epoch+1,
                     train_loss=total_loss/len(train), optimizer_steps=optimizer_steps,
                     validation_threshold=threshold, validation=validation,
                     **({"source_validation_threshold": source_threshold,
                         "source_validation": source_validation} if source_supervision else {}),
                     **({"training_loss": training_loss} if training_loss is not None else {})))
            key = (validation["mcc"], validation["f1"], validation["accuracy"],
                   validation["auc"] if validation["auc"] is not None else -float("inf"))
            if key > best_key:
                best_key, best_scores, best_threshold = key, list(scores), threshold
                if source_supervision:
                    best_source_scores, best_source_threshold = list(source_scores), source_threshold
                checkpoint = dict(
                    cfg_ablation_version=CHECKPOINT_SCHEMA, feature_schema=FEATURE_SCHEMA,
                    model_config=config, vocabulary=vocab.values, selected_epoch=epoch+1,
                    decision_threshold=threshold, selection="validation_mcc", validation=validation,
                    trainable_parameters=sum(p.numel() for p in trainable),
                    **({"training_loss": training_loss} if training_loss is not None else {}),
                    **({"source_validation": source_validation,
                        "source_decision_threshold": source_threshold} if source_supervision else {}),
                    adapter_state=base._cpu_state(base.get_peft_model_state_dict(model.encoder)),
                    task_state=base._cpu_state(model.task_modules.state_dict()))
                _save_torch(folder / "best.pt", checkpoint)
                del checkpoint
    if best_scores is None:
        raise RuntimeError("no checkpoint was selected")
    del model, optimizer, graph_parameters, source_parameters, trainable
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (best_scores, best_threshold,
            {split: coverage_summary(counts) for split, counts in coverage.items()} if aligned else None,
            (best_source_scores, best_source_threshold) if source_supervision else None)


def _prediction_outputs(folder, split, rows, scores, threshold, builder, *, checkpoint_hash,
                        alignment_coverage=None, source_reference=None, prediction_suffix=""):
    if len(rows) != len(scores) or (source_reference is not None and len(source_reference) != len(rows)):
        raise ValueError("prediction count mismatch")
    labels = [r["label"] for r in rows]
    selected = metrics(labels, scores, threshold)
    predictions = []
    for i, (r, score) in enumerate(zip(rows, scores)):
        if source_reference is None:
            length = len(builder._encode(r["raw_source"]))
            truncated = length > builder.source_max_length
        else:
            reference = source_reference[i]
            if any(reference.get(key) != value for key, value in identity(r).items()):
                raise ValueError("source prediction metadata does not match sample order")
            length, truncated = reference["source_token_count"], reference["source_truncated"]
        predictions.append(dict(identity(r), score=float(score), prediction=int(score >= decision_boundary(threshold)),
                                threshold=float(threshold), source_token_count=length,
                                source_truncated=truncated))
    summary = dict(selected=selected, fixed_0_5=metrics(labels, scores, 0.5),
                   cohort_sha256=cohort_hash(rows), checkpoint_sha256=checkpoint_hash,
                   source_truncated=sum(p["source_truncated"] for p in predictions), subgroups={})
    if alignment_coverage is not None:
        summary["alignment_coverage"] = alignment_coverage
    for name, flag in (("source_not_truncated", False), ("source_truncated", True)):
        indices = [i for i, p in enumerate(predictions) if p["source_truncated"] == flag]
        if indices:
            summary["subgroups"][name] = metrics([labels[i] for i in indices], [scores[i] for i in indices], threshold)
    path = folder / f"{split}{prediction_suffix}.predictions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for p in predictions:
            handle.write(json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n")
    atomic_json(folder / f"{split}{prediction_suffix}.metrics.json", summary)
    return summary


def compare_run(root: str | Path, split: str = "valid",
                reference_root: str | Path | None = None, reference_variant: str = "cfg") -> dict:
    root = Path(root)
    if (root / "readout_config.json").exists() and split != "valid":
        raise ValueError("frozen C readout screening is valid-only; test was not extracted")
    reference = Path(reference_root) if reference_root is not None else None
    if reference is not None:
        candidate_config = json.loads((root / "config.json").read_text())
        reference_config = json.loads((reference / "config.json").read_text())
        if reference_variant in {"dep_pretrain_cfg", "dep_pretrain_hierarchical"}:
            declared = {"initialization_policy", "pretrain_run_dir", "pretrain_relations_sha256",
                        "reference_run_dir", "region_dir", "regions_sha256", "region_schema", "comparison_run_dir"}
            if ({k: v for k, v in candidate_config.items() if k not in declared} !=
                    {k: v for k, v in reference_config.items() if k not in declared} or
                    candidate_config.get("reference_run_dir") != reference_config.get("reference_run_dir") or
                    candidate_config.get("pretrain_relations_sha256") != reference_config.get("pretrain_relations_sha256")):
                raise ValueError("comparison with A differs beyond declared H/P metadata")
        elif candidate_config != reference_config:
            rotation_policy = candidate_config.get("training_member_policy")
            pretrain_policy = candidate_config.get("initialization_policy")
            if (rotation_policy != "primevul_negative_rotation" and
                    pretrain_policy != "source_dependency_pretraining"):
                raise ValueError("reference run uses a different dataset, graph, or training configuration")
            declared = ({"training_member_policy", "rotation_dir", "rotation_selected_sha256",
                         "rotation_graphs_sha256", "reference_run_dir"} if rotation_policy == "primevul_negative_rotation"
                        else {"initialization_policy", "pretrain_run_dir",
                              "pretrain_relations_sha256", "reference_run_dir",
                              "program_dir", "program_sha256", "program_schema"})
            shared = {k: v for k, v in candidate_config.items() if k not in declared}
            if (shared != reference_config or
                    candidate_config.get("reference_run_dir") != str(reference.resolve())):
                raise ValueError("comparison differs beyond declared rotation/pretraining metadata")
    rows, result = {}, {"split": split, "metrics": {}, "changes_vs_baseline": {}}
    if reference is not None:
        result["reference_run_dir"] = str(reference.resolve())
        for policy_name in ("training_member_policy", "initialization_policy"):
            if candidate_config.get(policy_name):
                result[policy_name] = candidate_config[policy_name]
    for variant in COMPARISON_VARIANTS:
        folder = reference if reference is not None and (
            variant == reference_variant or reference_variant == "cfg" and variant in DEFAULT_VARIANTS) else root
        path = folder / variant / f"{split}.predictions.jsonl"
        if not path.is_file():
            continue
        rows[variant] = read_jsonl(path)
        if not rows[variant]:
            raise ValueError(f"empty predictions: {path}")
        thresholds = {r["threshold"] for r in rows[variant]}
        if len(thresholds) != 1:
            raise ValueError("one checkpoint threshold is required per prediction file")
        result["metrics"][variant] = metrics([r["label"] for r in rows[variant]],
                                             [r["score"] for r in rows[variant]], thresholds.pop())
    if not rows:
        raise ValueError(f"no {split} predictions found in {root}")
    if reference is not None and reference_variant not in rows:
        raise FileNotFoundError(f"comparison requires saved {reference_variant} predictions in {reference}")
    if "baseline" in rows:
        base_threshold = rows["baseline"][0]["threshold"]
        for variant in COMPARISON_VARIANTS[1:]:
            if variant not in rows:
                continue
            result["changes_vs_baseline"][variant] = {
                "validation_selected_thresholds": paired_changes(rows["baseline"], rows[variant]),
                "both_fixed_at_0_5": paired_changes(rows["baseline"], rows[variant], base_threshold=0.5, candidate_threshold=0.5),
                "both_at_baseline_threshold": paired_changes(rows["baseline"], rows[variant],
                                                             base_threshold=base_threshold, candidate_threshold=base_threshold),
            }
    if "aligned_attributes" in rows and "aligned_cfg" in rows:
        local = rows["aligned_attributes"]
        directed = rows["aligned_cfg"]
        local_threshold = local[0]["threshold"]
        result["changes_aligned_cfg_vs_aligned_attributes"] = {
            "validation_selected_thresholds": paired_changes(local, directed),
            "both_fixed_at_0_5": paired_changes(local, directed, base_threshold=0.5, candidate_threshold=0.5),
            "both_at_aligned_attributes_threshold": paired_changes(
                local, directed, base_threshold=local_threshold, candidate_threshold=local_threshold),
        }
    if "cfg" in rows:
        cfg_threshold = rows["cfg"][0]["threshold"]
        result["changes_vs_cfg"] = {}
        for variant in (*DDG_VARIANTS, *DUAL_FORWARD_ALPHA, *JK_VARIANTS,
                        *SOURCE_SUPERVISION_VARIANTS, *ROTATION_VARIANTS,
                        *PRETRAIN_CFG_VARIANTS, *READOUT_VARIANTS):
            if variant not in rows:
                continue
            result["changes_vs_cfg"][variant] = {
                "validation_selected_thresholds": paired_changes(rows["cfg"], rows[variant]),
                "both_fixed_at_0_5": paired_changes(rows["cfg"], rows[variant],
                                                     base_threshold=0.5, candidate_threshold=0.5),
                "both_at_cfg_threshold": paired_changes(rows["cfg"], rows[variant],
                                                         base_threshold=cfg_threshold,
                                                         candidate_threshold=cfg_threshold),
            }
    if reference_variant == "dep_pretrain_cfg" and reference_variant in rows:
        result["changes_vs_A"] = {variant: {
            "validation_selected_thresholds": paired_changes(rows[reference_variant], predictions),
            "both_fixed_at_0_5": paired_changes(rows[reference_variant], predictions,
                                                 base_threshold=0.5, candidate_threshold=0.5)}
            for variant, predictions in rows.items() if variant != reference_variant}
    if "region_local" in rows and "region_context" in rows:
        result["changes_region_context_vs_local"] = {
            "validation_selected_thresholds": paired_changes(rows["region_local"], rows["region_context"]),
            "both_fixed_at_0_5": paired_changes(rows["region_local"], rows["region_context"],
                                                 base_threshold=0.5, candidate_threshold=0.5)}
    if reference_variant == "dep_pretrain_hierarchical" and reference_variant in rows:
        result["changes_vs_old_H"] = {variant: paired_changes(rows[reference_variant], predictions)
            for variant, predictions in rows.items() if variant != reference_variant}
    comparison_name = f"comparison.{split}" + (".vs_old_H" if reference_variant == "dep_pretrain_hierarchical" else "")
    atomic_json(root / f"{comparison_name}.json", result)
    fields = ["variant", "accuracy", "precision", "recall", "f1", "mcc", "auc", "tp", "fp", "tn", "fn", "threshold"]
    with (root / f"{comparison_name}.csv").open("w", newline="") as handle:
        out = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        out.writeheader()
        for variant, values in result["metrics"].items():
            out.writerow(dict(variant=variant, **values))
    print(json.dumps({k: v for k, v in result.items() if k != "changes_vs_baseline"}, ensure_ascii=False, indent=2))
    return result


def shuffle_ddg_edges(view: dict, seed: int, sample_key: str) -> list[tuple[int, int]]:
    """Relabel only this function's DDG endpoints; preserve its degree multisets."""
    n = len(view["node_ids"])
    permutation = list(range(n))
    random.Random(f"{seed}:{sample_key}").shuffle(permutation)
    if n > 1 and all(i == p for i, p in enumerate(permutation)):
        permutation = permutation[1:] + permutation[:1]
    return sorted((permutation[s], permutation[t]) for s, t in view["ddg_edges"])


def _ddg_degree_multiset(edges: list[tuple[int, int]], n: int) -> Counter:
    incoming, outgoing = [0] * n, [0] * n
    for source, target in edges:
        outgoing[source] += 1
        incoming[target] += 1
    return Counter(zip(incoming, outgoing))


def ddg_audit_report(rows: list[dict], views: dict[str, dict], seed: int) -> dict:
    """Summarize the cached DDG and the fixed, within-function G permutation."""
    splits = {}
    for row in rows:
        view = views[row["sample_key"]]
        split = row["split"]
        counts = splits.setdefault(split, Counter(functions=0, functions_with_ddg=0,
                                                  functions_with_usable_ddg=0, cfg_edges=0,
                                                  shuffled_edges=0, same_edge_count_functions=0,
                                                  same_degree_multiset_functions=0,
                                                  changed_edge_set_functions=0))
        counts["functions"] += 1
        counts["cfg_edges"] += len(view["edges"])
        for key, value in view["ddg_audit"].items():
            counts[key] += value
        counts["functions_with_ddg"] += bool(view["ddg_audit"]["raw_edges"])
        counts["functions_with_usable_ddg"] += bool(view["ddg_edges"])
        shuffled = view.get("ddg_shuffled_edges")
        if shuffled is None:
            shuffled = shuffle_ddg_edges(view, seed, row["sample_key"])
        counts["shuffled_edges"] += len(shuffled)
        counts["same_edge_count_functions"] += len(shuffled) == len(view["ddg_edges"])
        counts["same_degree_multiset_functions"] += (
            _ddg_degree_multiset(shuffled, len(view["node_ids"])) ==
            _ddg_degree_multiset(view["ddg_edges"], len(view["node_ids"])))
        counts["changed_edge_set_functions"] += set(shuffled) != set(view["ddg_edges"])
    report = {"seed": seed, "shuffle": "fixed per-function DDG node permutation; CFG unchanged",
              "splits": {}}
    for split, counts in sorted(splits.items()):
        counts = dict(counts)
        counts["usable_rate"] = (counts["usable_edges"] / counts["raw_edges"]
                                 if counts["raw_edges"] else None)
        report["splits"][split] = counts
    return report


def _prepare(dataset, graphs_path, source_dataset, *, shuffle_seed=None):
    rows = read_records(dataset, source_dataset)
    graphs = load_graphs(graphs_path, rows)
    views = {k: abstract_cfg(g) for k, g in graphs.items()}
    del graphs
    if shuffle_seed is not None:
        for row in rows:
            key = row["sample_key"]
            views[key]["ddg_shuffled_edges"] = shuffle_ddg_edges(views[key], shuffle_seed, key)
    return rows, views


def run_experiment(args, base=None):
    import torch
    if (any(v.startswith("aligned_") or v in DDG_VARIANTS or v in DUAL_FORWARD_ALPHA or
            v in JK_VARIANTS or v in SOURCE_SUPERVISION_VARIANTS or v in ROTATION_VARIANTS or
            v in PRETRAIN_CFG_VARIANTS
            for v in args.variants) and
            args.source_max_length != 2048):
        raise ValueError("CFG ablations require the original 2048-token source budget")
    rows, views = _prepare(args.dataset, args.graphs, args.source_dataset,
                           shuffle_seed=args.seed if "cfg_ddg_shuffled" in args.variants else None)
    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "valid"]
    if args.source_dataset == "sven" or any({r["label"] for r in group} != {0, 1} for group in (train, valid)):
        raise ValueError("training requires train and valid splits, each containing both labels; never train SVEN")
    rotating = any(v in ROTATION_VARIANTS for v in args.variants)
    if rotating and (any(v not in ROTATION_VARIANTS for v in args.variants) or
                     not args.rotation_dir or not args.reference_run_dir):
        raise ValueError("rotation run needs only fixed/rotating variants, --rotation-dir, and --reference-run-dir")
    pretraining = any(v in PRETRAIN_CFG_VARIANTS for v in args.variants)
    if pretraining and (any(v not in PRETRAIN_CFG_VARIANTS for v in args.variants) or
                        not args.pretrain_dir or not args.reference_run_dir or rotating):
        raise ValueError("pretrained C run needs only pretraining variants and their phase-1 run")
    if any(v in PROGRAM_VARIANTS for v in args.variants) and not args.program_dir:
        raise ValueError("program graph variants require --program-dir")
    selected_rows = None
    pretrain_root = Path(args.pretrain_dir).resolve() if pretraining else None
    if rotating or pretraining:
        reference = Path(args.reference_run_dir).resolve()
        reference_config = json.loads((reference / "config.json").read_text())
        vocab = AttributeVocabulary(json.loads((reference / "vocabulary.json").read_text()))
        if digest(vocab.values) != reference_config["vocabulary_sha256"]:
            raise ValueError("original C attribute vocabulary changed")
        if rotating:
            from .cfg_rotation import _preparation
            preparation, _ = _preparation(args.rotation_dir)
            if preparation["reference_run_dir"] != str(reference):
                raise ValueError("rotation candidates were prepared against another C run")
            selected_rows, selected_views, selection = load_selected(args.rotation_dir,
                                                                     expected_count=2 * sum(r["label"] == 0 for r in train))
            if set(selected_views) & set(views):
                raise ValueError("rotation candidates overlap original C records")
            views.update(selected_views)
        else:
            pretrain_config = json.loads((pretrain_root / "config.json").read_text())
            if (pretrain_config["c_config"] != reference_config or
                    pretrain_config["reference_run_dir"] != str(reference) or
                    pretrain_config["pretrain_epochs"] != 1 or
                    pretrain_config["relation_alpha"] != 1.0):
                raise ValueError("pretraining must match original C and the specified one-epoch objective")
            if any(PRETRAIN_MODE[v] == "composition_pretrain" for v in args.variants):
                from .cfg_program import PROGRAM_SCHEMA
                from .cfg_dependency import COMPOSED_HEAD_VERSION
                if (pretrain_config.get("composed_head_version") != COMPOSED_HEAD_VERSION or
                        not args.program_dir or pretrain_config.get("program_dir") !=
                        str(Path(args.program_dir).resolve()) or
                        pretrain_config.get("program_schema") != PROGRAM_SCHEMA):
                    raise ValueError("composition stage 2 requires matching current-schema phase-1 facts")
    else:
        vocab = AttributeVocabulary.fit(train, views, args.vocab_limit)
    root = Path(args.output_dir)
    config = {name: getattr(args, name) for name in (
        "model_path", "source_dataset", "source_max_length", "epochs", "batch_size",
        "gradient_accumulation", "learning_rate", "graph_learning_rate", "weight_decay",
        "lora_r", "lora_alpha", "lora_dropout", "seed", "graph_hidden_size", "graph_steps",
        "vocab_limit", "log_every")}
    config.update(dataset=str(Path(args.dataset).resolve()), graphs=str(Path(args.graphs).resolve()),
                  feature_schema=FEATURE_SCHEMA, cohort_sha256=cohort_hash(rows),
                  train_cohort_sha256=cohort_hash(train), valid_cohort_sha256=cohort_hash(valid),
                  graph_file_sha256=file_sha256(args.graphs), vocabulary_sha256=digest(vocab.values))
    if rotating or pretraining:
        if config != reference_config:
            raise ValueError("pretrained/rotated C must keep the original data, graph, model and training settings")
        if rotating:
            config.update(training_member_policy="primevul_negative_rotation",
                          rotation_dir=str(Path(args.rotation_dir).resolve()),
                          rotation_selected_sha256=selection["selected_sha256"],
                          rotation_graphs_sha256=selection["selected_graphs_sha256"],
                          reference_run_dir=str(reference))
        else:
            config.update(initialization_policy="source_dependency_pretraining",
                          pretrain_run_dir=str(pretrain_root),
                          pretrain_relations_sha256=pretrain_config["relations_sha256"],
                          reference_run_dir=str(reference))
    if any(v in REGION_VARIANTS for v in args.variants):
        from .cfg_data import load_regions
        if not args.region_dir or not args.comparison_run_dir or args.graph_steps != 5:
            raise ValueError("H/P require --region-dir, --comparison-run-dir A and five graph steps")
        partitions, region_audit = load_regions(args.region_dir, rows, reference, views)
        if (pretrain_config.get("relation_accumulation_normalization") != "window_effective_functions" or
                pretrain_config["relations_sha256"] != json.loads((Path(args.comparison_run_dir) /
                    "config.json").read_text())["pretrain_relations_sha256"]):
            raise ValueError("H/P require corrected P0 dependency supervision")
        if any(PRETRAIN_MODE[v] == "region_pretrain" for v in args.variants) and (
                pretrain_config.get("region_alpha") != 1.0 or
                pretrain_config.get("region_projection") != 128 or
                pretrain_config.get("region_schema") != region_audit["region_schema"] or
                pretrain_config.get("regions_sha256") != region_audit["regions_sha256"] or
                pretrain_config.get("region_targets_sha256") != region_audit["train_targets_sha256"]):
            raise ValueError("P1 region supervision differs from stage 1")
        for key, partition in partitions.items():
            views[key]["regions"] = partition
        config.update(region_dir=str(Path(args.region_dir).resolve()),
                      regions_sha256=region_audit["regions_sha256"],
                      region_schema=region_audit["region_schema"],
                      comparison_run_dir=str(Path(args.comparison_run_dir).resolve()))
        # Check A's shared settings before any model or output directory is created.
        a_config = json.loads((Path(args.comparison_run_dir) / "config.json").read_text())
        if (a_config.get("initialization_policy") != "source_dependency_pretraining" or
                a_config.get("reference_run_dir") != str(reference) or
                {k: a_config.get(k) for k in reference_config} != reference_config):
            raise ValueError("A differs from H/P's original C settings")
        a_pretrain = str(Path(a_config["pretrain_run_dir"]).resolve())
        if (any(PRETRAIN_MODE[v] == "dep_pretrain" for v in args.variants) and str(pretrain_root) != a_pretrain or
                any(PRETRAIN_MODE[v] == "region_pretrain" for v in args.variants) and
                pretrain_config.get("reference_pretrain_dir") != a_pretrain):
            raise ValueError("B must reuse A's P0; C/D must use the matching fresh P1")
    program_audit = None
    if any(variant in PROGRAM_VARIANTS for variant in args.variants):
        from .cfg_program import load_programs
        programs, program_audit = load_programs(args.program_dir, rows, reference)
        if (pretrain_config.get("program_sha256") is not None and
                (pretrain_config["program_sha256"] != program_audit["program_sha256"] or
                 pretrain_config.get("program_schema") != program_audit["program_schema"])):
            raise ValueError("classification program facts differ from composition pretraining")
        for key, program in programs.items():
            views[key]["program"] = program
        config.update(program_dir=str(Path(args.program_dir).resolve()),
                      program_sha256=program_audit["program_sha256"],
                      program_schema=program_audit["program_schema"])
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root / "experiment"):
        meta = root / "config.json"
        if meta.exists():
            if not args.resume:
                raise FileExistsError(f"{root} already contains a run; use --resume for completed variants or choose a new directory")
            if json.loads(meta.read_text()) != config:
                raise ValueError("existing run configuration/cohort differs; choose a new output directory")
        else:
            if any((root / variant).exists() for variant in VARIANTS):
                raise FileExistsError("variant directory exists without run metadata")
            atomic_json(meta, config)
            atomic_json(root / "vocabulary.json", vocab.values)
        if program_audit is not None:
            from .cfg_program import program_cost
            cost_path = root / "program_cost.json"
            cost = program_cost(programs, rows, args.graph_steps)
            if cost_path.exists():
                if json.loads(cost_path.read_text()) != cost:
                    raise ValueError("prepared program operation counts changed")
            else:
                atomic_json(cost_path, cost)
        if any(variant in DDG_VARIANTS for variant in args.variants):
            audit_path = root / "ddg_audit.json"
            audit = ddg_audit_report(rows, views, args.seed)
            if audit_path.exists():
                if json.loads(audit_path.read_text()) != audit:
                    raise ValueError("existing DDG audit differs from this graph cohort")
            else:
                atomic_json(audit_path, audit)
        base = _base_module() if base is None else base
        device = base._resolve_device(args.device)
        builder = _tokenizer(base, config)
        print(json.dumps(dict(event="cohort", splits=dict(Counter(r["split"] for r in rows)),
                              vocabulary_sizes=vocab.sizes(), graph_hidden_size=args.graph_hidden_size,
                              graph_steps=args.graph_steps, graph_scope="full_function_cfg")), flush=True)
        for variant in args.variants:
            folder = root / variant
            complete = folder / "complete.json"
            variant_config = dict(config, variant=variant)
            stage1_path = None
            if variant in PRETRAIN_CFG_VARIANTS:
                stage1_path = pretrain_root / PRETRAIN_MODE[variant] / "last.pt"
                stage1_complete = json.loads((stage1_path.parent / "complete.json").read_text())
                stage1_sha256 = file_sha256(stage1_path)
                if (stage1_complete["checkpoint_sha256"] != stage1_sha256 or
                        stage1_complete["mode"] != PRETRAIN_MODE[variant]):
                    raise ValueError(f"{variant}: phase-1 checkpoint identity differs")
                variant_config["pretrain_checkpoint_sha256"] = stage1_sha256
            if complete.exists():
                state = json.loads(complete.read_text())
                if state.get("config_sha256") != digest(variant_config) or state.get("checkpoint_sha256") != file_sha256(folder/"best.pt"):
                    raise ValueError(f"{variant}: completed-run identity mismatch")
                if (not (folder/"valid.predictions.jsonl").exists() or
                        state.get("validation_predictions_sha256") != file_sha256(folder/"valid.predictions.jsonl")):
                    raise ValueError(f"{variant}: completed run is missing predictions")
                if variant in SOURCE_SUPERVISION_VARIANTS:
                    source_path = folder/"valid.source.predictions.jsonl"
                    if (not source_path.exists() or
                            state.get("source_validation_predictions_sha256") != file_sha256(source_path)):
                        raise ValueError(f"{variant}: completed run is missing source predictions")
                print(f"skip_completed_variant={variant}", flush=True)
                continue
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"{folder} has an interrupted run. Existing files are preserved; "
                                      "move this variant directory aside or choose a new run directory.")
            folder.mkdir(exist_ok=True)
            atomic_json(folder / "config.json", variant_config)
            source_output = None
            if variant == "baseline":
                # Delegate A to the unchanged original trainer and model, not a reimplementation.
                checkpoint = base.train_model(
                    None, folder/"best.pt", variant="baseline", model_path=args.model_path,
                    records=rows, source_max_length=args.source_max_length, context_max_length=384,
                    batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
                    epochs=args.epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                    lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                    seed=args.seed, device=str(device), log_every=args.log_every)
                threshold = float(checkpoint["decision_threshold"])
                del checkpoint
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                scores = torch.sigmoid(base.predict_checkpoint(folder/"best.pt", valid,
                                                                batch_size=args.batch_size, device=str(device))).tolist()
                alignment_coverage = None
            else:
                schedule = (epoch_train_rows(train, selected_rows, variant, args.epochs)
                            if variant in ROTATION_VARIANTS else None)
                if schedule is not None:
                    atomic_json(folder / "train_epochs.json", schedule_report(schedule))
                initial_adapter_state = None
                if stage1_path is not None:
                    stage1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
                    if (stage1["mode"] != PRETRAIN_MODE[variant] or
                            stage1["pretrain_config"] != pretrain_config):
                        raise ValueError(f"{variant}: wrong phase-1 LoRA checkpoint")
                    initial_adapter_state = stage1["adapter_state"]
                    del stage1
                scores, threshold, alignment_coverage, source_output = _train_graph(
                    base, variant_config, rows, views, vocab, folder, device, epoch_rows=schedule,
                    initial_adapter_state=initial_adapter_state)
                del initial_adapter_state
                if alignment_coverage is not None:
                    atomic_json(folder / "alignment_coverage.json", alignment_coverage)
            checkpoint_hash = file_sha256(folder/"best.pt")
            summary = _prediction_outputs(folder, "valid", valid, scores, threshold, builder,
                                           checkpoint_hash=checkpoint_hash,
                                           alignment_coverage=(alignment_coverage["valid"]
                                                               if alignment_coverage else None))
            completed = dict(config_sha256=digest(variant_config), checkpoint_sha256=checkpoint_hash,
                             validation_predictions_sha256=file_sha256(folder/"valid.predictions.jsonl"),
                             validation=summary["selected"])
            if source_output is not None:
                source_scores, source_threshold = source_output
                source_summary = _prediction_outputs(
                    folder, "valid", valid, source_scores, source_threshold, builder,
                    checkpoint_hash=checkpoint_hash, prediction_suffix=".source")
                completed.update(source_validation_predictions_sha256=file_sha256(
                    folder/"valid.source.predictions.jsonl"), source_validation=source_summary["selected"])
            atomic_json(complete, completed)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return compare_run(root, "valid", reference_root=config.get("comparison_run_dir") or
                           (reference if rotating or pretraining else None),
                           reference_variant="dep_pretrain_cfg" if config.get("comparison_run_dir") else "cfg")


def evaluate_run(args, base=None):
    import torch
    from .cfg_network import build_model
    root = Path(args.run_dir)
    config = json.loads((root/"config.json").read_text())
    rows, views = _prepare(config["dataset"], config["graphs"], config["source_dataset"],
                           shuffle_seed=config["seed"] if "cfg_ddg_shuffled" in args.variants else None)
    if cohort_hash(rows) != config["cohort_sha256"] or file_sha256(config["graphs"]) != config["graph_file_sha256"]:
        raise ValueError("evaluation data/graphs differ from the recorded run")
    if config.get("region_dir"):
        from .cfg_data import load_regions
        partitions, audit = load_regions(config["region_dir"], rows, config["reference_run_dir"], views)
        if audit["regions_sha256"] != config["regions_sha256"] or audit["region_schema"] != config["region_schema"]:
            raise ValueError("evaluation region facts changed")
        for key, partition in partitions.items():
            views[key]["regions"] = partition
    if config.get("program_dir"):
        from .cfg_program import load_programs
        programs, audit = load_programs(config["program_dir"], rows, config["reference_run_dir"])
        if (audit["program_sha256"] != config["program_sha256"] or
                audit["program_schema"] != config.get("program_schema")):
            raise ValueError("evaluation program facts changed")
        for key, program in programs.items():
            views[key]["program"] = program
    if config.get("training_member_policy") == "primevul_negative_rotation":
        _, _, selection = load_selected(config["rotation_dir"])
        if (selection["selected_sha256"] != config["rotation_selected_sha256"] or
                selection["selected_graphs_sha256"] != config["rotation_graphs_sha256"]):
            raise ValueError("rotation training members or graphs changed")
    selected = [r for r in rows if r["split"] == args.split]
    if not selected:
        raise ValueError(f"empty split {args.split}")
    base = _base_module() if base is None else base
    device = base._resolve_device(args.device)
    builder = _tokenizer(base, config)
    with output_lock(root / "experiment"):
        for variant in args.variants:
            folder = root/variant
            if not (folder/"complete.json").exists():
                raise ValueError(f"{variant} did not finish training")
            completed = json.loads((folder/"complete.json").read_text())
            checkpoint = torch.load(folder/"best.pt", map_location="cpu", weights_only=False)
            variant_config = dict(config, variant=variant)
            if variant in PRETRAIN_CFG_VARIANTS:
                phase_hash = checkpoint.get("model_config", {}).get("pretrain_checkpoint_sha256")
                if not isinstance(phase_hash, str) or len(phase_hash) != 64 or any(
                        char not in "0123456789abcdef" for char in phase_hash):
                    raise ValueError(f"{variant}: missing phase-1 checkpoint provenance")
                variant_config["pretrain_checkpoint_sha256"] = phase_hash
            if (completed.get("config_sha256") != digest(variant_config) or
                    completed.get("checkpoint_sha256") != file_sha256(folder/"best.pt")):
                raise ValueError(f"{variant}: checkpoint/config has changed since training")
            path = folder/f"{args.split}.predictions.jsonl"
            source_path = folder/f"{args.split}.source.predictions.jsonl" if variant in SOURCE_SUPERVISION_VARIANTS else None
            for existing in (path, source_path):
                if existing is not None and existing.exists() and not args.replace_predictions:
                    raise FileExistsError(f"{existing} exists; use --replace-predictions to recompute this evaluation only")
            source_scores = None
            if variant == "baseline":
                scores = torch.sigmoid(base.predict_checkpoint(folder/"best.pt", selected,
                                        batch_size=args.batch_size, device=str(device))).tolist()
            else:
                if (checkpoint.get("cfg_ablation_version") != CHECKPOINT_SCHEMA or
                        checkpoint.get("feature_schema") != FEATURE_SCHEMA or
                        checkpoint.get("model_config") != variant_config):
                    raise ValueError("incompatible graph checkpoint")
                vocab = AttributeVocabulary(checkpoint["vocabulary"])
                if digest(vocab.values) != config["vocabulary_sha256"]:
                    raise ValueError("checkpoint vocabulary mismatch")
                model = build_model(base, checkpoint["model_config"], vocab.sizes(), device, training=False)
                base.set_peft_model_state_dict(model.encoder, checkpoint["adapter_state"])
                model.task_modules.load_state_dict(checkpoint["task_state"])
                encoded = {r["sample_key"]: vocab.encode(views[r["sample_key"]]) for r in selected}
                coverage = Counter() if variant.startswith("aligned_") else None
                predictions = _graph_scores(model, selected, builder, views, encoded, batch_size=args.batch_size,
                                            device=device, coverage=coverage,
                                            include_source=variant in SOURCE_SUPERVISION_VARIANTS)
                scores, source_scores = (predictions if variant in SOURCE_SUPERVISION_VARIANTS
                                         else (predictions, None))
                del model
            threshold = float(checkpoint["decision_threshold"])
            summary = _prediction_outputs(folder, args.split, selected, scores, threshold, builder,
                                          checkpoint_hash=file_sha256(folder/"best.pt"),
                                          alignment_coverage=coverage_summary(coverage) if variant != "baseline" and coverage is not None else None)
            if source_scores is not None:
                source_summary = _prediction_outputs(
                    folder, args.split, selected, source_scores,
                    float(checkpoint["source_decision_threshold"]), builder,
                    checkpoint_hash=file_sha256(folder/"best.pt"), prediction_suffix=".source")
            if args.split == "valid":
                completed["validation_predictions_sha256"] = file_sha256(path)
                completed["validation"] = summary["selected"]
                if source_scores is not None:
                    completed["source_validation_predictions_sha256"] = file_sha256(source_path)
                    completed["source_validation"] = source_summary["selected"]
                atomic_json(folder/"complete.json", completed)
            del checkpoint
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return compare_run(root, args.split,
                           reference_root=config.get("comparison_run_dir") or config.get("reference_run_dir"),
                           reference_variant="dep_pretrain_cfg" if config.get("comparison_run_dir") else "cfg")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="export a resumable graph sidecar; preserve old data")
    build.add_argument("--dataset", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--source-dataset", choices=("primevul", "cleanvul", "sven"), default="primevul")
    build.add_argument("--joern-dir", default="/home/phy/joern")
    build.add_argument("--java-home", default="/home/phy/jdk21")
    build.add_argument("--timeout", type=int, default=300)
    build.add_argument("--batch-size", type=int, default=8)
    run = sub.add_parser("run", help="train selected CFG ablations; validation only")
    run.add_argument("--dataset", required=True)
    run.add_argument("--graphs", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--rotation-dir", help="prepared and graph-verified new PrimeVul train negatives")
    run.add_argument("--pretrain-dir", help="completed train-only source pretraining run")
    run.add_argument("--region-dir", help="shared H/P region cache")
    run.add_argument("--comparison-run-dir", help="A: corrected dependency-pretrained C result")
    run.add_argument("--program-dir", help="prepared check/update/use facts for program graph variants")
    run.add_argument("--reference-run-dir", help="completed original C run; reuse its exact vocabulary and settings")
    run.add_argument("--source-dataset", choices=("primevul", "cleanvul"), default="primevul")
    run.add_argument("--model-path", default="/home/phy/models/Qwen2.5-Coder-7B-Instruct")
    run.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    for name, default in (("source-max-length", 2048), ("epochs", 3), ("batch-size", 1),
                          ("gradient-accumulation", 8), ("lora-r", 16), ("lora-alpha", 32),
                          ("graph-hidden-size", 128), ("graph-steps", 5), ("vocab-limit", 2048),
                          ("log-every", 10), ("seed", 42)):
        run.add_argument("--"+name, type=int, default=default)
    for name, default in (("learning-rate", 2e-4), ("graph-learning-rate", 1e-3),
                          ("weight-decay", 0.01), ("lora-dropout", 0.05)):
        run.add_argument("--"+name, type=float, default=default)
    run.add_argument("--device", default="auto")
    run.add_argument("--resume", action="store_true", help="skip completed variants after identity checks")
    evaluate = sub.add_parser("eval", help="evaluate fixed selected checkpoints; never tune on test")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--split", choices=("valid", "test"), default="test")
    evaluate.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--replace-predictions", action="store_true")
    extract = sub.add_parser("extract-readout", help="cache frozen C train/valid representations")
    extract.add_argument("--c-run-dir", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--device", default="auto")
    readout = sub.add_parser("train-readout", help="train residual heads on frozen C representations")
    readout.add_argument("--cache", required=True)
    readout.add_argument("--c-run-dir", required=True)
    readout.add_argument("--output-dir", required=True)
    readout.add_argument("--modes", nargs="+", choices=READOUT_VARIANTS,
                         default=list(READOUT_VARIANTS))
    readout.add_argument("--device", default="cpu")
    readout.add_argument("--resume", action="store_true")
    compare = sub.add_parser("compare", help="recompute tables/error changes from saved predictions")
    compare.add_argument("--run-dir", required=True)
    compare.add_argument("--split", choices=("valid", "test"), default="valid")
    compare.add_argument("--reference-variant", choices=("cfg", "dep_pretrain_cfg", "dep_pretrain_hierarchical"), default="cfg")
    compare.add_argument("--reference-run-dir", help="read saved A/B/C predictions from a matching prior run")
    audit = sub.add_parser("audit-alignment", help="audit cached CFG node/source alignment without Qwen")
    audit.add_argument("--dataset", required=True)
    audit.add_argument("--graphs", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--source-dataset", choices=("primevul", "cleanvul", "sven"), default="primevul")
    audit.add_argument("--model-path", default="/home/phy/models/Qwen2.5-Coder-7B-Instruct")
    audit.add_argument("--example-limit", type=int, default=8)
    ddg_audit = sub.add_parser("audit-ddg", help="audit cached DDG coverage and G shuffling without Qwen")
    ddg_audit.add_argument("--dataset", required=True)
    ddg_audit.add_argument("--graphs", required=True)
    ddg_audit.add_argument("--output", required=True)
    ddg_audit.add_argument("--source-dataset", choices=("primevul", "cleanvul", "sven"), default="primevul")
    ddg_audit.add_argument("--seed", type=int, default=42)
    prepare_rotation = sub.add_parser("prepare-rotation", help="screen official train negatives without Joern")
    prepare_rotation.add_argument("--raw-root", default="/home/PublicData/PHY-data/vul_detect/data")
    prepare_rotation.add_argument("--benchmark-manifest", default="data/benchmark/manifest.jsonl")
    prepare_rotation.add_argument("--reference-run-dir", required=True)
    prepare_rotation.add_argument("--output-dir", required=True)
    build_rotation = sub.add_parser("build-rotation", help="graph only needed new negatives; refill failures")
    build_rotation.add_argument("--rotation-dir", required=True)
    build_rotation.add_argument("--joern-dir", default="/home/phy/joern")
    build_rotation.add_argument("--java-home", default="/home/phy/jdk21")
    build_rotation.add_argument("--timeout", type=int, default=300)
    build_rotation.add_argument("--batch-size", type=int, default=8)
    region_prepare = sub.add_parser("prepare-regions", help="prepare shared H/P regions using cached CFG and C vocabulary")
    region_prepare.add_argument("--reference-run-dir", required=True)
    region_prepare.add_argument("--output-dir", required=True)
    prepare_program_cmd = sub.add_parser("prepare-program", help="prepare cached check/update/use facts and train/valid queries")
    prepare_program_cmd.add_argument("--dataset", required=True)
    prepare_program_cmd.add_argument("--graphs", required=True)
    prepare_program_cmd.add_argument("--reference-run-dir", required=True)
    prepare_program_cmd.add_argument("--output-dir", required=True)
    prepare_program_cmd.add_argument("--comparison-program-dir", help="prior train/valid query audit for coverage comparison")
    prepare_dep = sub.add_parser("prepare-dep", help="prepare scoped scalar relations from the saved Joern graph")
    prepare_dep.add_argument("--dataset", required=True)
    prepare_dep.add_argument("--graphs", required=True)
    prepare_dep.add_argument("--reference-run-dir", required=True)
    prepare_dep.add_argument("--output-dir", required=True)
    pretrain_dep = sub.add_parser("pretrain-dep", help="one-epoch train-only CLM and relation pretraining")
    pretrain_dep.add_argument("--reference-run-dir", required=True)
    pretrain_dep.add_argument("--supervision-dir", help="existing scoped relation supervision; required for old modes")
    pretrain_dep.add_argument("--region-dir", help="prepared H/P source-to-region targets")
    pretrain_dep.add_argument("--reference-pretrain-dir", help="corrected P0 run for equal-budget P1 provenance")
    pretrain_dep.add_argument("--program-dir", help="prepared composition queries and structural facts")
    pretrain_dep.add_argument("--output-dir", required=True)
    pretrain_dep.add_argument("--modes", nargs="+", choices=("lm_pretrain", "dep_pretrain",
                                                       "composition_pretrain", "region_pretrain"),
                              default=["lm_pretrain", "dep_pretrain"])
    pretrain_dep.add_argument("--device", default="auto")
    pretrain_dep.add_argument("--resume", action="store_true")
    eval_dep = sub.add_parser("eval-dep-relations", help="score saved stage-1 relation head on train/valid only")
    eval_dep.add_argument("--pretrain-dir", required=True)
    eval_dep.add_argument("--mode", choices=("dep_pretrain", "composition_pretrain"),
                          default="dep_pretrain")
    eval_dep.add_argument("--output-dir", required=True)
    eval_dep.add_argument("--batch-size", type=int, default=1)
    eval_dep.add_argument("--device", default="auto")
    for command in ("audit-regions", "eval-regions"):
        diagnostic = sub.add_parser(command, help="read-only H/P train/valid diagnostics")
        diagnostic.add_argument("--pretrain-dir", required=True)
        diagnostic.add_argument("--output-dir", required=True)
        if command == "eval-regions":
            diagnostic.add_argument("--batch-size", type=int, default=1)
            diagnostic.add_argument("--device", default="auto")
    return p


def main():
    args = parser().parse_args()
    try:
        if args.command == "build":
            report = build_graphs(args.dataset, args.output, source_dataset=args.source_dataset,
                                  joern_dir=args.joern_dir, java_home=args.java_home,
                                  timeout=args.timeout, batch_size=args.batch_size)
            if not report["complete"]:
                return 2
        elif args.command == "prepare-rotation":
            from .cfg_rotation import prepare_rotation
            print(json.dumps(prepare_rotation(args.raw_root, args.benchmark_manifest,
                                              args.reference_run_dir, args.output_dir),
                             ensure_ascii=False), flush=True)
        elif args.command == "build-rotation":
            from .cfg_rotation import build_rotation
            selection = build_rotation(args.rotation_dir, joern_dir=args.joern_dir,
                                       java_home=args.java_home, timeout=args.timeout,
                                       batch_size=args.batch_size)
            print(json.dumps({k: v for k, v in selection.items() if k != "selected_sample_keys"},
                             ensure_ascii=False), flush=True)
        elif args.command == "prepare-regions":
            from .cfg_data import prepare_regions
            from transformers import AutoTokenizer
            config = json.loads((Path(args.reference_run_dir) / "config.json").read_text())
            tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
            print(json.dumps(prepare_regions(args.reference_run_dir, args.output_dir, tokenizer)), flush=True)
        elif args.command == "prepare-program":
            from transformers import AutoTokenizer
            from .cfg_program import prepare_program
            reference_config = json.loads((Path(args.reference_run_dir) / "config.json").read_text())
            tokenizer = AutoTokenizer.from_pretrained(reference_config["model_path"], trust_remote_code=True)
            print(json.dumps(prepare_program(args.dataset, args.graphs, args.reference_run_dir,
                                             args.output_dir, tokenizer,
                                             comparison_program_dir=args.comparison_program_dir),
                             ensure_ascii=False), flush=True)
        elif args.command == "prepare-dep":
            from transformers import AutoTokenizer
            from .cfg_dependency import prepare_supervision
            reference_config = json.loads((Path(args.reference_run_dir) / "config.json").read_text())
            tokenizer = AutoTokenizer.from_pretrained(reference_config["model_path"], trust_remote_code=True)
            report = prepare_supervision(args.dataset, args.graphs, args.reference_run_dir,
                                         args.output_dir, tokenizer)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        elif args.command == "pretrain-dep":
            from .cfg_dependency import pretrain_causal_dependency
            print(json.dumps(pretrain_causal_dependency(
                args.reference_run_dir, args.supervision_dir, args.output_dir,
                modes=tuple(args.modes), device=args.device, resume=args.resume,
                program_dir=args.program_dir, region_dir=args.region_dir,
                reference_pretrain_dir=args.reference_pretrain_dir),
                ensure_ascii=False), flush=True)
        elif args.command == "audit-regions":
            from .cfg_region_diagnostics import audit_regions
            report = audit_regions(args.pretrain_dir, args.output_dir)
            print(json.dumps({"output": args.output_dir, "splits": {
                s: v["counts"] for s, v in report["splits"].items()}}), flush=True)
        elif args.command == "eval-regions":
            from .cfg_region_diagnostics import evaluate_regions
            print(json.dumps(evaluate_regions(args.pretrain_dir, args.output_dir,
                batch_size=args.batch_size, device=args.device)), flush=True)
        elif args.command == "eval-dep-relations":
            from .cfg_dependency import evaluate_fixed_relations
            print(json.dumps(evaluate_fixed_relations(
                args.pretrain_dir, args.output_dir, batch_size=args.batch_size,
                device=args.device, mode=args.mode), ensure_ascii=False), flush=True)
        elif args.command == "run":
            for name in ("source_max_length", "epochs", "batch_size", "gradient_accumulation",
                         "lora_r", "lora_alpha", "graph_steps", "log_every"):
                if getattr(args, name) <= 0:
                    raise ValueError(f"{name} must be positive")
            if (args.graph_hidden_size < 4 or args.graph_hidden_size % 4 or args.vocab_limit < 2 or
                    not 0 <= args.lora_dropout < 1 or args.learning_rate <= 0 or
                    args.graph_learning_rate <= 0 or args.weight_decay < 0 or
                    not all(math.isfinite(v) for v in (args.learning_rate, args.graph_learning_rate,
                                                       args.weight_decay, args.lora_dropout)) or
                    len(args.variants) != len(set(args.variants))):
                raise ValueError("invalid graph/training hyperparameters")
            run_experiment(args)
        elif args.command == "eval":
            if args.batch_size <= 0:
                raise ValueError("batch_size must be positive")
            evaluate_run(args)
        elif args.command == "extract-readout":
            from .cfg_readout import extract_representations
            print(json.dumps(extract_representations(args.c_run_dir, args.output,
                                                     device=args.device), ensure_ascii=False), flush=True)
        elif args.command == "train-readout":
            from .cfg_readout import train_readouts
            print(json.dumps(train_readouts(args.cache, args.c_run_dir, args.output_dir,
                                            modes=tuple(args.modes), device=args.device,
                                            resume=args.resume), ensure_ascii=False), flush=True)
        elif args.command == "compare":
            compare_run(args.run_dir, args.split, reference_root=args.reference_run_dir,
                        reference_variant=args.reference_variant)
        elif args.command == "audit-ddg":
            output = Path(args.output)
            if output.exists():
                raise FileExistsError(f"DDG audit output already exists: {output}")
            rows, views = _prepare(args.dataset, args.graphs, args.source_dataset)
            report = ddg_audit_report(rows, views, args.seed)
            atomic_json(output, report)
            print(json.dumps({"output": str(output), "splits": report["splits"]},
                             ensure_ascii=False), flush=True)
        else:
            from transformers import AutoTokenizer
            from .cfg_alignment_audit import audit_alignment
            output = Path(args.output)
            if output.exists():
                raise FileExistsError(f"alignment audit output already exists: {output}")
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            report = audit_alignment(
                args.dataset, args.graphs, tokenizer, source_dataset=args.source_dataset,
                source_max_length=2048, example_limit=args.example_limit,
                progress=lambda done, total, _: print(f"alignment_audit={done}/{total}", flush=True))
            report["tokenizer_path"] = str(Path(args.model_path).resolve())
            atomic_json(output, report)
            print(json.dumps({"output": str(output), "samples": report["samples"],
                              "overall": report["overall"]}, ensure_ascii=False), flush=True)
    except (ValueError, RuntimeError, FileExistsError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
