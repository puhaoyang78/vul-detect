"""A/B/C, aligned D/E, DDG F/G, and C dual-forward runner in one training flow.

Examples are in docs/CFG_ABLATION.md. Training never evaluates the test split.
The baseline behavior, datasets, and old checkpoints are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys

from .cfg_alignment import align_nodes, coverage_summary
from .cfg_data import (FEATURE_SCHEMA, AttributeVocabulary, abstract_cfg, atomic_json,
                       build_graphs, cohort_hash, digest, identity, load_graphs,
                       output_lock, read_jsonl, read_records)
from .cfg_metrics import decision_boundary, metrics, paired_changes, select_threshold

VARIANTS = ("baseline", "attributes", "cfg", "aligned_attributes", "aligned_cfg",
            "cfg_ddg", "cfg_ddg_shuffled", "cfg_double_ce", "cfg_rdrop")
DDG_VARIANTS = ("cfg_ddg", "cfg_ddg_shuffled")
DUAL_FORWARD_ALPHA = {"cfg_double_ce": 0.0, "cfg_rdrop": 1.0}
DEFAULT_VARIANTS = VARIANTS[:3]
CHECKPOINT_SCHEMA = 1


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


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
    graph_batch = collate_graphs([views[k] for k in keys], [encoded[k] for k in keys],
                                 device=device, alignments=alignments, ddg_edges=ddg_edges)
    return ids, mask, graph_batch


def _graph_forward(model, batch, builder, views, encoded, device, coverage=None):
    return model(*_graph_inputs(model, batch, builder, views, encoded, device, coverage))


def _graph_scores(model, rows, builder, views, encoded, *, batch_size, device, coverage=None):
    import torch
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            logits = _graph_forward(model, rows[start:start+batch_size], builder, views, encoded, device,
                                    coverage=coverage)
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite graph prediction")
            values.extend(torch.sigmoid(logits.float()).cpu().tolist())
    return values


def _save_torch(path, checkpoint):
    import torch
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    torch.save(checkpoint, partial)
    partial.replace(path)


def _train_graph(base, config, rows, views, vocab, folder, device):
    import torch
    from .cfg_network import build_model
    from tqdm.auto import tqdm

    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "valid"]
    train_views = {r["sample_key"]: views[r["sample_key"]] for r in train+valid}
    encoded = {k: vocab.encode(v) for k, v in train_views.items()}
    builder = _tokenizer(base, config)
    base._seed_everything(config["seed"])
    model = build_model(base, config, vocab.sizes(), device, training=True)
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
    aligned = config["variant"].startswith("aligned_")
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

        for epoch in range(config["epochs"]):
            model.train()
            order = list(range(len(train)))
            random.Random(config["seed"] + epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            num_batches = math.ceil(len(order)/batch_size)
            total_loss, window_loss, window_samples = 0.0, 0.0, 0
            total_bce = total_kl = window_bce = window_kl = 0.0
            optimizer_steps = 0
            bar = tqdm(range(0, len(order), batch_size), total=num_batches,
                       desc=f"{config['variant']} epoch {epoch+1}/{config['epochs']}",
                       file=sys.stdout, dynamic_ncols=True, mininterval=1)
            for batch_index, start in enumerate(bar):
                batch = [train[i] for i in order[start:start+batch_size]]
                inputs = _graph_inputs(model, batch, builder, views, encoded, device,
                                       coverage=coverage["train"] if aligned and epoch == 0 else None)
                logits = model(*inputs)
                labels = torch.tensor([r["label"] for r in batch], dtype=torch.float32, device=device)
                if dual_alpha is None:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                else:
                    # Reuse the exact same source/CFG tensors; dropout consumes fresh RNG state.
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
                                     "alpha": dual_alpha} if dual_alpha is not None else {})))
                        window_loss, window_samples = 0.0, 0
                        window_bce = window_kl = 0.0
                bar.set_postfix(loss=f"{value:.4f}", step=step, refresh=False)
            scores = _graph_scores(model, valid, builder, views, encoded, batch_size=batch_size, device=device,
                                   coverage=coverage["valid"] if aligned and epoch == 0 else None)
            threshold, validation = select_threshold([r["label"] for r in valid], scores)
            training_loss = ({"bce": total_bce/len(train), "kl": total_kl/len(train),
                              "alpha": dual_alpha} if dual_alpha is not None else None)
            log(dict(event="epoch", variant=config["variant"], epoch=epoch+1,
                     train_loss=total_loss/len(train), optimizer_steps=optimizer_steps,
                     validation_threshold=threshold, validation=validation,
                     **({"training_loss": training_loss} if training_loss is not None else {})))
            key = (validation["mcc"], validation["f1"], validation["accuracy"],
                   validation["auc"] if validation["auc"] is not None else -float("inf"))
            if key > best_key:
                best_key, best_scores, best_threshold = key, list(scores), threshold
                checkpoint = dict(
                    cfg_ablation_version=CHECKPOINT_SCHEMA, feature_schema=FEATURE_SCHEMA,
                    model_config=config, vocabulary=vocab.values, selected_epoch=epoch+1,
                    decision_threshold=threshold, selection="validation_mcc", validation=validation,
                    trainable_parameters=sum(p.numel() for p in trainable),
                    **({"training_loss": training_loss} if training_loss is not None else {}),
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
    return best_scores, best_threshold, ({split: coverage_summary(counts)
                                         for split, counts in coverage.items()} if aligned else None)


def _prediction_outputs(folder, split, rows, scores, threshold, builder, *, checkpoint_hash,
                        alignment_coverage=None):
    if len(rows) != len(scores):
        raise ValueError("prediction count mismatch")
    labels = [r["label"] for r in rows]
    selected = metrics(labels, scores, threshold)
    predictions = []
    for r, score in zip(rows, scores):
        length = len(builder._encode(r["raw_source"]))
        predictions.append(dict(identity(r), score=float(score), prediction=int(score >= decision_boundary(threshold)),
                                threshold=float(threshold), source_token_count=length,
                                source_truncated=length > builder.source_max_length))
    summary = dict(selected=selected, fixed_0_5=metrics(labels, scores, 0.5),
                   cohort_sha256=cohort_hash(rows), checkpoint_sha256=checkpoint_hash,
                   source_truncated=sum(p["source_truncated"] for p in predictions), subgroups={})
    if alignment_coverage is not None:
        summary["alignment_coverage"] = alignment_coverage
    for name, flag in (("source_not_truncated", False), ("source_truncated", True)):
        indices = [i for i, p in enumerate(predictions) if p["source_truncated"] == flag]
        if indices:
            summary["subgroups"][name] = metrics([labels[i] for i in indices], [scores[i] for i in indices], threshold)
    path = folder / f"{split}.predictions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for p in predictions:
            handle.write(json.dumps(p, ensure_ascii=False, allow_nan=False) + "\n")
    atomic_json(folder / f"{split}.metrics.json", summary)
    return summary


def compare_run(root: str | Path, split: str = "valid",
                reference_root: str | Path | None = None) -> dict:
    root = Path(root)
    reference = Path(reference_root) if reference_root is not None else None
    if reference is not None:
        if json.loads((root / "config.json").read_text()) != json.loads((reference / "config.json").read_text()):
            raise ValueError("reference run uses a different dataset, graph, or training configuration")
    rows, result = {}, {"split": split, "metrics": {}, "changes_vs_baseline": {}}
    if reference is not None:
        result["reference_run_dir"] = str(reference.resolve())
    for variant in VARIANTS:
        folder = reference if reference is not None and variant in DEFAULT_VARIANTS else root
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
    if reference is not None and "cfg" not in rows:
        raise FileNotFoundError(f"comparison requires saved C predictions in {reference / 'cfg'}")
    if "baseline" in rows:
        base_threshold = rows["baseline"][0]["threshold"]
        for variant in VARIANTS[1:]:
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
        for variant in (*DDG_VARIANTS, *DUAL_FORWARD_ALPHA):
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
    atomic_json(root / f"comparison.{split}.json", result)
    fields = ["variant", "accuracy", "precision", "recall", "f1", "mcc", "auc", "tp", "fp", "tn", "fn", "threshold"]
    with (root / f"comparison.{split}.csv").open("w", newline="") as handle:
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
    if (any(v.startswith("aligned_") or v in DDG_VARIANTS or v in DUAL_FORWARD_ALPHA
            for v in args.variants) and args.source_max_length != 2048):
        raise ValueError("D/E/F/G and C dual-forward runs require the original 2048-token source budget")
    rows, views = _prepare(args.dataset, args.graphs, args.source_dataset,
                           shuffle_seed=args.seed if "cfg_ddg_shuffled" in args.variants else None)
    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "valid"]
    if args.source_dataset == "sven" or any({r["label"] for r in group} != {0, 1} for group in (train, valid)):
        raise ValueError("training requires train and valid splits, each containing both labels; never train SVEN")
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
            if complete.exists():
                state = json.loads(complete.read_text())
                if state.get("config_sha256") != digest(variant_config) or state.get("checkpoint_sha256") != file_sha256(folder/"best.pt"):
                    raise ValueError(f"{variant}: completed-run identity mismatch")
                if (not (folder/"valid.predictions.jsonl").exists() or
                        state.get("validation_predictions_sha256") != file_sha256(folder/"valid.predictions.jsonl")):
                    raise ValueError(f"{variant}: completed run is missing predictions")
                print(f"skip_completed_variant={variant}", flush=True)
                continue
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError(f"{folder} has an interrupted run. Existing files are preserved; "
                                      "move this variant directory aside or choose a new run directory.")
            folder.mkdir(exist_ok=True)
            atomic_json(folder / "config.json", variant_config)
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
                scores, threshold, alignment_coverage = _train_graph(
                    base, variant_config, rows, views, vocab, folder, device)
                if alignment_coverage is not None:
                    atomic_json(folder / "alignment_coverage.json", alignment_coverage)
            checkpoint_hash = file_sha256(folder/"best.pt")
            summary = _prediction_outputs(folder, "valid", valid, scores, threshold, builder,
                                           checkpoint_hash=checkpoint_hash,
                                           alignment_coverage=(alignment_coverage["valid"]
                                                               if alignment_coverage else None))
            atomic_json(complete, dict(config_sha256=digest(variant_config), checkpoint_sha256=checkpoint_hash,
                                       validation_predictions_sha256=file_sha256(folder/"valid.predictions.jsonl"),
                                       validation=summary["selected"]))
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return compare_run(root, "valid")


def evaluate_run(args, base=None):
    import torch
    from .cfg_network import build_model
    root = Path(args.run_dir)
    config = json.loads((root/"config.json").read_text())
    rows, views = _prepare(config["dataset"], config["graphs"], config["source_dataset"],
                           shuffle_seed=config["seed"] if "cfg_ddg_shuffled" in args.variants else None)
    if cohort_hash(rows) != config["cohort_sha256"] or file_sha256(config["graphs"]) != config["graph_file_sha256"]:
        raise ValueError("evaluation data/graphs differ from the recorded run")
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
            if (completed.get("config_sha256") != digest(dict(config, variant=variant)) or
                    completed.get("checkpoint_sha256") != file_sha256(folder/"best.pt")):
                raise ValueError(f"{variant}: checkpoint/config has changed since training")
            path = folder/f"{args.split}.predictions.jsonl"
            if path.exists() and not args.replace_predictions:
                raise FileExistsError(f"{path} exists; use --replace-predictions to recompute this evaluation only")
            checkpoint = torch.load(folder/"best.pt", map_location="cpu", weights_only=False)
            if variant == "baseline":
                scores = torch.sigmoid(base.predict_checkpoint(folder/"best.pt", selected,
                                        batch_size=args.batch_size, device=str(device))).tolist()
            else:
                if (checkpoint.get("cfg_ablation_version") != CHECKPOINT_SCHEMA or
                        checkpoint.get("feature_schema") != FEATURE_SCHEMA or
                        checkpoint.get("model_config") != dict(config, variant=variant)):
                    raise ValueError("incompatible graph checkpoint")
                vocab = AttributeVocabulary(checkpoint["vocabulary"])
                if digest(vocab.values) != config["vocabulary_sha256"]:
                    raise ValueError("checkpoint vocabulary mismatch")
                model = build_model(base, checkpoint["model_config"], vocab.sizes(), device, training=False)
                base.set_peft_model_state_dict(model.encoder, checkpoint["adapter_state"])
                model.task_modules.load_state_dict(checkpoint["task_state"])
                encoded = {r["sample_key"]: vocab.encode(views[r["sample_key"]]) for r in selected}
                coverage = Counter() if variant.startswith("aligned_") else None
                scores = _graph_scores(model, selected, builder, views, encoded, batch_size=args.batch_size,
                                       device=device, coverage=coverage)
                del model
            threshold = float(checkpoint["decision_threshold"])
            summary = _prediction_outputs(folder, args.split, selected, scores, threshold, builder,
                                          checkpoint_hash=file_sha256(folder/"best.pt"),
                                          alignment_coverage=coverage_summary(coverage) if variant != "baseline" and coverage is not None else None)
            if args.split == "valid":
                completed["validation_predictions_sha256"] = file_sha256(path)
                completed["validation"] = summary["selected"]
                atomic_json(folder/"complete.json", completed)
            del checkpoint
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return compare_run(root, args.split)


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
    run = sub.add_parser("run", help="train selected A/B/C/D/E/F/G and C dual-forward variants; validation only")
    run.add_argument("--dataset", required=True)
    run.add_argument("--graphs", required=True)
    run.add_argument("--output-dir", required=True)
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
    compare = sub.add_parser("compare", help="recompute tables/error changes from saved predictions")
    compare.add_argument("--run-dir", required=True)
    compare.add_argument("--split", choices=("valid", "test"), default="valid")
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
        elif args.command == "compare":
            compare_run(args.run_dir, args.split, reference_root=args.reference_run_dir)
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
