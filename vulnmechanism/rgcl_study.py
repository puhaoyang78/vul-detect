"""Fixed three-seed RGCL comparisons through the existing trainer/evaluators."""

import gc
import json
import math
import statistics
import time
from pathlib import Path

import torch

from .artifacts import write_json
from .data import read_jsonl
from .training import RGCL_METHODS, LOSVER_CONFIG, train_losver

SEEDS = (42, 1234, 123456)
DATASETS = ("primevul_paired", "secvuleval")


def data_root(data_dir, dataset):
    return Path(data_dir) if dataset == "primevul_paired" else Path(data_dir)/"rgcl"


def run_path(output_dir, dataset, seed, method):
    if seed == 123456:
        if dataset == "primevul_paired" and method != "plm_baseline":
            old = "context_bce" if method == "context_only" else "context_diff"
            return Path("results/attention_context_512")/old/dataset/"unixcoder"
        if dataset == "secvuleval":
            return Path("results/rgcl_512/seed123456")/dataset/method
    return Path(output_dir)/f"seed{seed}"/dataset/method


def inspect_run(path, data_dir, dataset, seed, method, model_path, *, checkpoint=True):
    """Require the original members, all epochs and the stipulated Valid choice."""
    path = Path(path)
    config = json.loads((path/"training.metrics.json").read_text())
    directory = data_root(data_dir, dataset)/dataset
    train = read_jsonl(directory/"train.lines.jsonl")
    valid = read_jsonl(directory/("valid.lines.jsonl" if dataset == "primevul_paired" else "valid.jsonl"))
    positive_weight = (1. if dataset == "primevul_paired" else
                      json.loads((directory/"statistics.json").read_text())["classification_positive_weight"])
    expected = dict(LOSVER_CONFIG, seed=seed, class_weight=positive_weight)
    if (config["training_config"] != expected or config["input_mode"] != "512"
        or config["model_path"] != str(model_path) or config["completed_epochs"] != 16
        or config["optimizer_steps"] != 16*math.ceil(len(train)/8)
        or config["function_micro_batch"] != 8 or config["gradient_accumulation"] != 1
        or config["decision_threshold"] != .5 or config["comparison"] != ">"
        or config["training_members"] != [r["sample_key"] for r in train]
        or config["validation_members"] != [r["sample_key"] for r in valid]):
        raise ValueError(f"{path}: mismatched training protocol or official members")
    for key in ("rccl", "local_mil", "insertion_localization", "attention_regularization", "line_auxiliary",
                "input_marginalization", "paired_batches", "patch_attention", "patch_context"):
        if config.get(key):
            raise ValueError(f"{path}: excluded module/protocol {key}")
    bounded = config.get("bounded_attention")
    if method == "plm_baseline":
        if not config.get("cls_baseline") or bounded:
            raise ValueError(f"{path}: CLS baseline must use RobertaClassificationHead")
    elif (not bounded or not bounded.get("context") or bounded.get("k") != 5
          or bounded.get("region_weighting", "token") != "token"
          or bounded.get("region_supervision") != ("diff" if method == "rgcl" else None)
          or bounded.get("region_weight") != (.1 if method == "rgcl" else 0.)
          or bounded.get("local") or bounded.get("context_scale")):
        raise ValueError(f"{path}: not the fixed original Context/RGCL architecture")
    key = "p_c" if dataset == "primevul_paired" else "mcc"
    chosen = json.loads((path/f"best-{key}.json").read_text())
    epochs = [r for r in read_jsonl(path/"best.training.jsonl") if r["event"] == "epoch"]
    if [r["epoch"] for r in epochs] != list(range(1, 17)):
        raise ValueError(f"{path}: incomplete canonical epoch history")
    selected = max(epochs, key=lambda r: round(r["validation"][key], 4) if key == "p_c" else r["validation"][key])
    if (selected["epoch"] != chosen["epoch"] or selected["validation"] != chosen["validation"]
        or config["selected_epochs"][key] != chosen["epoch"]):
        raise ValueError(f"{path}: checkpoint was not selected by complete Valid {key}")
    if checkpoint:
        cp = torch.load(path/f"best-{key}.pt", map_location="cpu", weights_only=False)
        if cp["selected_epoch"] != chosen["epoch"] or cp["validation"] != chosen["validation"]:
            raise ValueError(f"{path}: checkpoint and selection record disagree")
        state = cp["state_dict"]
        if method == "plm_baseline":
            if tuple(state["classifier.dense.weight"].shape) != (768, 768) or tuple(state["classifier.out_proj.weight"].shape) != (1, 768):
                raise ValueError("CLS checkpoint does not contain the standard classification head")
        elif any(tuple(state[name].shape) != (32, 768) for name in ("context_u.weight", "context_v.weight")):
            raise ValueError("RGCL checkpoint context projections differ")
        del cp, state
        gc.collect()
    return dict(run=str(path), complete=True, selected_epoch=chosen["epoch"], selection=key,
        validation=chosen["validation"], epochs=16, optimizer_steps=config["optimizer_steps"],
        train_functions=len(train), valid_functions=len(valid),
        training_seconds=sum(r["training_seconds"] for r in epochs),
        epoch_seconds=sum(r["epoch_seconds"] for r in epochs), checkpoint_checked=checkpoint)


def inventory(data_dir, output_dir, model_path):
    entries = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for method in RGCL_METHODS:
                path = run_path(output_dir, dataset, seed, method)
                reused = not path.is_relative_to(Path(output_dir))
                if (path/"training.metrics.json").exists():
                    info = inspect_run(path, data_dir, dataset, seed, method, model_path)
                elif reused:
                    raise FileNotFoundError(f"required historical result is missing: {path}")
                else:
                    info = dict(run=str(path), complete=False, resumable=(path/"last.pt").exists())
                entries.append(dict(dataset=dataset, seed=seed, method=method, reused=reused, **info))
    result = dict(seeds=list(SEEDS), excluded_historical_seed=123457, model_path=str(model_path),
        input_tokens=512, batch_size=8, epochs=16, learning_rate=2e-5, k=5, region_weight=.1,
        region_weighting="token", threshold=.5, comparison=">", sampler="original function RandomSampler",
        patch="stopped; results/weights preserved separately; not part of RGCL",
        test_used_for_selection=False, experiments=entries,
        reused=sum(e["reused"] for e in entries), missing=sum(not e["complete"] for e in entries))
    write_json(Path(output_dir)/"experiments.json", result)
    return result


def check(data_dir, output_dir, model_path, *, device="auto"):
    """Exercise both datasets plus the original RGCL checkpoint, without Test."""
    from . import evaluation, model as core
    from .training import paired_loader, paired_collate, paired_forward

    results = {}
    for dataset in DATASETS:
        results[dataset] = evaluation.check_rgcl(data_root(data_dir, dataset), output_dir, model_path, dataset, device=device)
    original = run_path(output_dir, "primevul_paired", 123456, "rgcl")
    cp = torch.load(original/"best-p_c.pt", map_location="cpu", weights_only=False)
    model, tokenizer = evaluation.load_paired(cp, core.resolve_device(device))
    rows = read_jsonl(Path(data_dir)/"primevul_paired/valid.jsonl")[:8]
    loader = paired_loader(rows, tokenizer, "baseline")
    batch = paired_collate(loader.dataset)
    expected = read_jsonl(original/"valid.best-p_c.predictions.jsonl")[:8]
    if [r["sample_key"] for r in expected] != [r["sample_key"] for r in rows]:
        raise ValueError("original RGCL validation members changed")
    with torch.no_grad():
        probability = paired_forward(model, batch, "baseline", device, with_labels=False)[:, 0].cpu()
    torch.testing.assert_close(probability, torch.tensor([r["probability"] for r in expected]), rtol=1e-5, atol=1e-6)
    single = core.predict_checkpoint(original/"best-p_c.pt", [dict(raw_source=rows[0]["func"])], device=device).sigmoid()
    torch.testing.assert_close(single, probability[:1], rtol=1e-5, atol=1e-6)
    results["original_rgcl_checkpoint"] = dict(path=str(original), passed=True,
        source_only_predictions_match_historical_valid=True, single_function=True)
    write_json(Path(output_dir)/"gpu-smoke.json", dict(passed=True, results=results, test_used=False))
    return results


def run(data_dir, output_dir, model_path, dataset, seed, methods, *, device="auto", save_every=500):
    from .evaluation import evaluate

    if dataset not in DATASETS or seed not in SEEDS or any(m not in RGCL_METHODS for m in methods):
        raise ValueError("only the fixed two-dataset, three-seed RGCL comparison is allowed")
    if not json.loads((Path(output_dir)/"gpu-smoke.json").read_text())["passed"]:
        raise ValueError("complete real GPU verification before full training")
    for method in methods:
        path = run_path(output_dir, dataset, seed, method)
        reused = not path.is_relative_to(Path(output_dir))
        if reused:
            inspect_run(path, data_dir, dataset, seed, method, model_path)
            if not (path/"test.metrics.json").exists() or not (path/"test.predictions.jsonl").exists():
                raise FileNotFoundError(f"historical final evaluation is missing: {path}")
            continue
        if not (path/"training.metrics.json").exists():
            started = time.monotonic()
            train_losver(data_root(data_dir, dataset), Path(output_dir)/f"seed{seed}", model_path, method,
                dataset=dataset, seed=seed, epochs_override=16, device=device, resume=True, save_every=save_every)
            write_json(path/"commands.json", dict(dataset=dataset, seed=seed, method=method, device=device,
                model_path=str(model_path), epochs=16, save_every=save_every, wall_seconds=time.monotonic()-started,
                entry="shared train_losver; original RandomSampler, optimizer and losses"))
        info = inspect_run(path, data_dir, dataset, seed, method, model_path)
        write_json(path/"valid_selection.json", dict(selection=info["selection"], selected_epoch=info["selected_epoch"],
            validation=info["validation"], source="full Valid; fixed before Test evaluation"))
        missing = tuple(s for s in ("valid", "test") if not (path/f"{s}.metrics.json").exists())
        if missing:
            evaluate(data_root(data_dir, dataset), path.parent.parent, dataset, [path.name],
                     device=device, robust=False, splits=missing)
    return dict(dataset=dataset, seed=seed, methods=list(methods), complete=True)


def report(data_dir, output_dir, model_path):
    """Recompute probabilities at the fixed threshold and summarize three seeds."""
    from .metrics import metrics, paired_metrics

    root = Path(output_dir)
    result = inventory(data_dir, output_dir, model_path)
    if result["missing"]:
        raise ValueError("finish all missing combinations before reporting three-seed results")
    records = []
    for entry in result["experiments"]:
        dataset, path = entry["dataset"], Path(entry["run"])
        for split in ("valid", "test"):
            filename = "valid.best-p_c.predictions.jsonl" if dataset == "primevul_paired" and split == "valid" else f"{split}.predictions.jsonl"
            pred = read_jsonl(path/filename)
            rows = read_jsonl(data_root(data_dir, dataset)/dataset/f"{split}.jsonl")
            if ([r["sample_key"] for r in rows] != [r["sample_key"] for r in pred]
                or any(r["target"] != p["target"] for r,p in zip(rows,pred))):
                raise ValueError(f"{path}: {split} prediction members/labels differ")
            scores = [p["probability"] for p in pred]
            value = paired_metrics(rows, scores) if dataset == "primevul_paired" else metrics([r["target"] for r in rows], scores, strict=True)
            if split == "valid":
                for key in ("p_c", "mcc", "f1", "auc", "auprc", "recall"):
                    if key in entry["validation"] and value[key] != entry["validation"][key]:
                        raise ValueError(f"{path}: saved full Valid prediction metrics differ")
            else:
                saved = json.loads((path/"test.metrics.json").read_text())
                if saved["selected_epoch"] != entry["selected_epoch"] or any(saved["metrics"][k] != value[k] for k in value):
                    raise ValueError(f"{path}: fixed-checkpoint Test metrics do not reproduce")
            fpr = value["patched_fpr"] if dataset == "primevul_paired" else value["fp"]/(value["fp"]+value["tn"])
            records.append(dict(dataset=dataset, seed=entry["seed"], method=entry["method"], split=split,
                selected_epoch=entry["selected_epoch"], reused=entry["reused"], run=str(path),
                metrics={**{k:value[k] for k in ("mcc","f1","auc","auprc","recall")},
                         "fpr":fpr, **({"p_c":value["p_c"]} if dataset == "primevul_paired" else {})}))
    aggregate = []
    for dataset in DATASETS:
        for split in ("valid", "test"):
            for method in RGCL_METHODS:
                members = [r for r in records if (r["dataset"],r["split"],r["method"]) == (dataset,split,method)]
                if sorted(r["seed"] for r in members) != sorted(SEEDS):
                    raise ValueError("the mean must include exactly seeds 42, 1234 and 123456")
                aggregate.append(dict(dataset=dataset,split=split,method=method,
                    metrics={k:dict(mean=statistics.mean(r["metrics"][k] for r in members),
                                    std=statistics.stdev(r["metrics"][k] for r in members)) for k in members[0]["metrics"]}))
    result.update(records=records, aggregate=aggregate, std_definition="sample standard deviation; ddof=1; n=3")
    lines = ["# RGCL fixed three-seed comparison", "", "RGCL is the original Context + Diff: K=5, lambda=0.1, token-weighted region loss. "
        "CLS reuses the SecVulEval RobertaClassificationHead. Context-only differs from RGCL only by removing Diff loss. "
        "UniXcoder, 512 tokens, Batch8, 16 epochs, AdamW lr=2e-5, original function sampling; threshold strictly >0.5. "
        "PrimeVul selects full Valid P-C; SecVulEval selects full Valid MCC. Test never selects a checkpoint or configuration.", "",
        "Seeds are 42, 1234 and 123456. Five completed original runs are reused; thirteen missing combinations are trained. "
        "Seed123457, historical Global Mean and the stopped Patch study are preserved and excluded from the three-seed mean. "
        "Historical result paths and completed optimization budgets are recorded in experiments.json. "
        "Standard deviations below use ddof=1. New resumable checkpoints are saved every 500 steps and at epoch boundaries; "
        "this changes disk I/O only, with the original optimization budget and RNG behavior. "
        "PrimeVul FPR uses the complete repaired side of official pairs; SecVulEval FPR uses all original normal functions, "
        "including ordinary and repaired functions. PR-AUC retains the existing average-precision (AP) definition.", "",
        "| Dataset | Split | Seed | Method | Epoch | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        m=r["metrics"];pc=f"{m['p_c']:.2%}" if "p_c" in m else "—"
        lines.append(f"| {r['dataset']} | {r['split']} | {r['seed']} | {r['method']} | {r['selected_epoch']} | {pc} | "+
                     " | ".join(f"{m[k]:.6f}" for k in ("mcc","f1","auc","auprc","recall","fpr"))+" |")
    lines += ["", "| Dataset | Split | Method | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        values=[]
        for key in ("p_c","mcc","f1","auc","auprc","recall","fpr"):
            v=r["metrics"].get(key)
            values.append((f"{100*v['mean']:.2f}% ± {100*v['std']:.2f}pp" if key=="p_c" else
                           f"{v['mean']:.6f} ± {v['std']:.6f}") if v else "—")
        lines.append(f"| {r['dataset']} | {r['split']} | {r['method']} | "+" | ".join(values)+" |")
    deltas=[]
    for dataset in DATASETS:
        for split in ("valid","test"):
            values={r["method"]:r["metrics"] for r in aggregate if r["dataset"]==dataset and r["split"]==split}
            for before,after in (("plm_baseline","context_only"),("context_only","rgcl"),("plm_baseline","rgcl")):
                delta={k:values[after][k]["mean"]-values[before][k]["mean"] for k in values[after]}
                deltas.append(dict(dataset=dataset,split=split,before=before,after=after,metrics=delta))
                primary="p_c" if dataset=="primevul_paired" else "mcc"
                wins=sum(next(r for r in records if (r['dataset'],r['split'],r['seed'],r['method'])==(dataset,split,seed,after))['metrics'][primary] >
                         next(r for r in records if (r['dataset'],r['split'],r['seed'],r['method'])==(dataset,split,seed,before))['metrics'][primary] for seed in SEEDS)
                lines.extend(["",f"{dataset} {split}, {before} → {after}: mean {primary} {delta[primary]:+.6f}; improved in {wins}/3 seeds. "
                              f"AUC {delta['auc']:+.6f}; MCC {delta['mcc']:+.6f}; FPR {delta['fpr']:+.6f}."])
    result["deltas"]=deltas
    write_json(root/"comparison.json",result)
    (root/"report.md").write_text("\n".join(lines)+"\n")
    return result
