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
from .training import RGCL_METHODS, RGCL_ALL_METHODS, RGCL_REGION_TARGETS, LOSVER_CONFIG, train_losver

SEEDS = (42, 1234, 123456)
DATASETS = ("primevul_paired", "secvuleval")
SUPPORTED_DATASETS = (*DATASETS, "cleanvul")
BACKBONE_MODELS = {
    "unixcoder": "/home/PublicData/PHY-data/resource/unixcoder-base-nine",
    "codebert": "/home/PublicData/PHY-data/resource/codebert-base",
    "codet5": "/home/PublicData/PHY-data/resource/codet5-base",
}


def study_methods(backbone, methods=None):
    if backbone not in BACKBONE_MODELS:
        raise ValueError("choose a local UniXcoder, CodeBERT or CodeT5 backbone")
    selected = RGCL_METHODS if methods is None else tuple(methods)
    if not selected or len(set(selected)) != len(selected) or any(m not in RGCL_ALL_METHODS for m in selected):
        raise ValueError("choose distinct existing RGCL methods")
    if any(m in selected for m in ("random_region", *RGCL_REGION_TARGETS)) and backbone != "unixcoder":
        raise ValueError("region ablations are restricted to UniXcoder")
    return selected


def data_root(data_dir, dataset):
    return Path(data_dir) if dataset == "primevul_paired" else Path(data_dir)/"rgcl"


def run_path(output_dir, dataset, seed, method, *, backbone="unixcoder", reference_dir=None):
    if reference_dir is not None and method in RGCL_METHODS:
        return run_path(reference_dir, dataset, seed, method, backbone=backbone)
    if seed == 123456 and backbone == "unixcoder":
        if dataset == "primevul_paired" and method in ("context_only", "rgcl"):
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
          or bounded.get("region_supervision") != ("random" if method == "random_region" else "diff" if method in ("rgcl", *RGCL_REGION_TARGETS) else None)
          or bounded.get("region_weight") != (.1 if method in ("rgcl", "random_region", *RGCL_REGION_TARGETS) else 0.)
          or bounded.get("region_target") != RGCL_REGION_TARGETS.get(method)
          or bounded.get("local") or bounded.get("context_scale")):
        raise ValueError(f"{path}: not the fixed original Context/RGCL architecture")
    if method == "random_region":
        randomization = bounded.get("randomization", {})
        if (randomization.get("seed") != seed or not randomization.get("train_only") or
            randomization.get("exact_token_mass_matches") != len(train) or
            not randomization.get("fixed_across_epochs")):
            raise ValueError(f"{path}: Random-region must preserve each Train function's token mass")
    if method in RGCL_REGION_TARGETS:
        stats = bounded["side_supervision"]
        if (stats["target"] != RGCL_REGION_TARGETS[method] or stats["functions"] != len(train)
            or stats["by_target"][str(1 - stats["target"])]["supervised_functions"] != 0
            or config["region_function_visits"] != 16 * stats["supervised_functions"]):
            raise ValueError(f"{path}: incorrect single-side supervision coverage")
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
        if method in RGCL_REGION_TARGETS and cp["bounded_attention"] != bounded:
            raise ValueError(f"{path}: checkpoint auxiliary supervision side differs")
        state = cp["state_dict"]
        if method == "plm_baseline":
            if cp.get("classification_architecture") == "codet5_defect":
                if tuple(state["classifier.weight"].shape) != (2, 768):
                    raise ValueError("CodeT5 checkpoint does not contain the defect classification head")
            elif tuple(state["classifier.dense.weight"].shape) != (768, 768) or tuple(state["classifier.out_proj.weight"].shape) != (1, 768):
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


def inventory(data_dir, output_dir, model_path, *, datasets=DATASETS, backbone="unixcoder", methods=None, reference_dir=None):
    entries = []
    methods = study_methods(backbone, methods)
    if any(m in methods for m in ("random_region", *RGCL_REGION_TARGETS)) and tuple(datasets) != ("primevul_paired",):
        raise ValueError("region ablations use only PrimeVul")
    if not datasets or any(d not in SUPPORTED_DATASETS for d in datasets):
        raise ValueError("choose a prepared RGCL study dataset")
    for dataset in datasets:
        for seed in SEEDS:
            for method in methods:
                path = run_path(output_dir, dataset, seed, method, backbone=backbone, reference_dir=reference_dir)
                reused = not path.is_relative_to(Path(output_dir))
                if (path/"training.metrics.json").exists():
                    info = inspect_run(path, data_dir, dataset, seed, method, model_path)
                elif reused:
                    raise FileNotFoundError(f"required historical result is missing: {path}")
                else:
                    info = dict(run=str(path), complete=False, resumable=(path/"last.pt").exists())
                entries.append(dict(dataset=dataset, seed=seed, method=method, reused=reused, **info))
    result = dict(seeds=list(SEEDS), excluded_historical_seed=123457, model_path=str(model_path), backbone=backbone,
        input_tokens=512, batch_size=8, epochs=16, learning_rate=2e-5,
        k=5, region_weight=.1, region_weighting="token",
        threshold=.5, comparison=">", sampler="original function RandomSampler",
        patch="stopped; results/weights preserved separately; not part of RGCL",
        test_used_for_selection=False, experiments=entries,
        reused=sum(e["reused"] for e in entries), missing=sum(not e["complete"] for e in entries))
    write_json(Path(output_dir)/"experiments.json", result)
    return result


def check(data_dir, output_dir, model_path, *, device="auto", datasets=DATASETS, backbone="unixcoder", methods=None, reference_dir=None):
    """Exercise selected datasets and the original RGCL checkpoint, without Test."""
    from . import evaluation, model as core
    from .training import paired_loader, paired_collate, paired_forward

    results = {}
    methods = study_methods(backbone, methods)
    if any(m in methods for m in ("random_region", *RGCL_REGION_TARGETS)) and tuple(datasets) != ("primevul_paired",):
        raise ValueError("region ablations use only PrimeVul")
    for dataset in datasets:
        results[dataset] = evaluation.check_rgcl(data_root(data_dir, dataset), output_dir, model_path, dataset,
            device=device, methods=methods)
    if backbone != "unixcoder":
        write_json(Path(output_dir)/"gpu-smoke.json", dict(passed=True, results=results,
            model_path=str(model_path), backbone=backbone, test_used=False))
        return results
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


def run(data_dir, output_dir, model_path, dataset, seed, methods=None, *, device="auto", save_every=500, backbone="unixcoder", reference_dir=None):
    from .evaluation import evaluate

    methods = study_methods(backbone, methods)
    if dataset not in SUPPORTED_DATASETS or seed not in SEEDS or (any(m in methods for m in ("random_region", *RGCL_REGION_TARGETS)) and dataset != "primevul_paired"):
        raise ValueError("only the fixed three-seed RGCL comparison is allowed")
    smoke = json.loads((Path(output_dir)/"gpu-smoke.json").read_text())
    if not smoke["passed"] or (backbone != "unixcoder" and
        (smoke.get("backbone") != backbone or smoke.get("model_path") != str(model_path) or dataset not in smoke["results"])):
        raise ValueError("complete real GPU verification before full training")
    for method in methods:
        if method in ("random_region", *RGCL_REGION_TARGETS) and method not in smoke.get("results", {}).get("primevul_paired", {}).get("methods", {}):
            raise ValueError(f"{method} requires its own real GPU smoke before training")
    for method in methods:
        path = run_path(output_dir, dataset, seed, method, backbone=backbone, reference_dir=reference_dir)
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


def report(data_dir, output_dir, model_path, *, datasets=DATASETS, backbone="unixcoder", methods=None, reference_dir=None):
    """Recompute probabilities at the fixed threshold and summarize three seeds."""
    from .metrics import metrics, paired_metrics

    root = Path(output_dir)
    methods = study_methods(backbone, methods)
    result = inventory(data_dir, output_dir, model_path, datasets=datasets, backbone=backbone,
                       methods=methods, reference_dir=reference_dir)
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
                metrics={**{k:value[k] for k in ("accuracy","precision","mcc","f1","auc","auprc","recall")},
                         "fpr":fpr, **({"p_c":value["p_c"]} if dataset == "primevul_paired" else {})}))
    aggregate = []
    for dataset in datasets:
        for split in ("valid", "test"):
            for method in methods:
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
        "PrimeVul selects full Valid P-C; other datasets select full Valid MCC. Test never selects a checkpoint or configuration.", "",
        f"Seeds are 42, 1234 and 123456. {result.get('reused', 0)} original runs are reused; completed new runs use the identical shared trainer. "
        "Seed123457, historical Global Mean and the stopped Patch study are preserved and excluded from the three-seed mean. "
        "Historical result paths and completed optimization budgets are recorded in experiments.json. "
        "Standard deviations below use ddof=1. New resumable checkpoints are saved every 500 steps and at epoch boundaries; "
        "this changes disk I/O only, with the original optimization budget and RNG behavior. "
        "PrimeVul FPR uses the complete repaired side of official pairs; other datasets use all retained normal functions, "
        "including ordinary and repaired functions. PR-AUC retains the existing average-precision (AP) definition.", "",
        "| Dataset | Split | Seed | Method | Epoch | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if "cleanvul" in datasets:
        data = json.loads((data_root(data_dir, "cleanvul")/"cleanvul/statistics.json").read_text())
        result["cleanvul_data"] = data
        checks = data["diagnostic_reasons"]
        flagged = sum(n for name,n in checks.items() if name != "retained")
        details = ["", "## CleanVul 成员及监督口径", "",
            f"采用评分≥3的全部 C/C++ 发布记录：{data['retained_pairs']} 对，{sum(data['class_counts'].values())} 个函数。"
            "发布文件没有独立0/1字段；沿用修复前=1、修复后=0的约定。源码逐字保留，未进行新的质量筛选。"
            f"评分定义见 [CleanVul 原论文]({data['score_definition']['source']})。", "",
            f"此前{flagged}条排除来自可执行检查：解析不完整/非单函数{checks.get('parse_incomplete_or_not_single_function',0)}、"
            f"名称不同{checks.get('function_name_mismatch',0)}、名称未知{checks.get('unknown_function_name',0)}、"
            f"词法序列不变{checks.get('comments_or_formatting_only',0)}。"
            "这些是语法与对应关系诊断，不能判定分类标签错误；本轮全部恢复，检查代码仅记录原因，不过滤分类成员。"
            "对应来源及诊断见 pairs.csv，最终成员见 data/rgcl/cleanvul/members.csv。"
            "区域监督直接沿用同一发布行的前后版本及原 _line_diff / paired_features；不额外要求解析成功或同名。"
            "512可见区域须同时含修改行和背景行，保留原有部分可见修改行门控；无有效区域仍参与BCE。", "",
            "Seed42按CVE、仓库限定提交、修复关系和空白归一化重复代码传递分组，目标8:1:1。"
            f"由于恢复了原先排除的成员，重新对全部成员分组；旧{checks.get('retained',0)}对口径未用于任何训练。"
            "跨划分重叠组为0；不依据Valid/Test结果修改成员。", "",
            "| Split | Pairs | Vulnerable | Repaired | Effective before | Effective after | Effective / functions |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for split in ("train", "valid", "test"):
            s=data["splits"][split]; c=data["coverage"][split]
            details.append(f"| {split} | {s['pairs']} | {s['class_counts']['1']} | {s['class_counts']['0']} | "
                f"{c['1']['effective_region']} | {c['0']['effective_region']} | {c['all']['effective_region']}/{s['functions']} ({c['all']['effective_percent']:.2f}%) |")
        details.extend(["", "三种方法沿用同一函数随机采样、训练预算与预训练权重；Context-only与RGCL仅相差0.1倍真实Diff区域损失。"
            "不增加Patch、RCCL或局部头。历史PrimeVul/SecVulEval成员、权重与结果均保留。", ""])
        lines[4:4] = details
    if backbone != "unixcoder":
        lines = [f"# {backbone} fixed three-seed CLS / Context-only / RGCL comparison", "",
            "Same frozen PrimeVul/SecVulEval/CleanVul members as the UniXcoder study; seeds 42, 1234, 123456; "
            "512 tokens, Batch8, 16 epochs, original AdamW lr=2e-5 and function sampling. "
            "PrimeVul selects full Valid P-C rounded to four decimals; other datasets select unrounded full Valid MCC. "
            "Threshold strictly >0.5. Test is evaluated only after Valid selection. PR-AUC is average precision; std uses ddof=1.", "",
            ("CodeBERT uses CLS + the existing RobertaClassificationHead and weighted function BCE."
             if backbone == "codebert" else
             "CodeT5 uses the official defect readout: same-source shifted decoder input, last source-EOS decoder state, "
             "Linear(hidden,2). Weighted BCE on class1-class0 is equivalent to weighted two-class cross-entropy. "
             "Context-only/RGCL reuse token-aligned decoder states from the same forward; no extra encoding or model head. "
             "No language-model loss or generation is used. "
             "[Official model](https://github.com/salesforce/CodeT5/blob/main/CodeT5/models.py)."), "",
            "Context-only and RGCL share the original rank-32 Context Attention and bounded line MIL, K=5, "
            "function BCE. RGCL adds only 0.1 times the original token-weighted Diff region loss. "
            "Inference reads one function only. No historical data or weights are overwritten.", "",
            "| Dataset | Split | Seed | Method | Epoch | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if "random_region" in methods:
        lines[0] = "# RGCL Random-region 三种子消融"
        lines[2] = ("只在 PrimeVul + UniXcoder 补充 Random-region；Context-only 与 RGCL 复用历史完整结果。"
                    "512 Token、Batch8、16 Epoch、AdamW lr=2e-5、K=5、区域损失权重0.1；按完整 Valid P-C 选模，阈值严格>0.5。"
                    "随机区域为有效代码行的随机子集，逐函数严格保持原 Diff 的有效 Token 质量及辅助损失可用性。"
                    "跨行 Token 按原实现分摊；局部独立 RNG 随训练种子固定，整个训练期间不重采样。"
                    "保持原区域损失、编码和函数 BCE；推理仅输入单函数，Test 不参与选模。")
    if any(m in methods for m in RGCL_REGION_TARGETS):
        lines[0] = "# RGCL 双侧修复监督消融"
        lines[2] = ("PrimeVul + UniXcoder，Context-only 与 RGCL 复用历史完整结果；仅新增 Vulnerable-only、Repaired-only 各三种子。"
                    "全部函数保留原标签参与 BCE，训练顺序不变；仅在训练特征中禁用另一侧的区域标签。"
                    "区域损失仍按有效 Token 加权、在有效启用函数间平均，系数0.1，不补偿监督数量差异。"
                    "K=5、512 Token、Batch8、16 Epoch、AdamW lr=2e-5；完整 Valid P-C（保留四位小数，先到先选）选模，阈值严格>0.5。"
                    "Test 只评价既定 checkpoint，不用于修改方法、训练轮次或参数。")
    for r in records:
        m=r["metrics"];pc=f"{m['p_c']:.2%}" if "p_c" in m else "—"
        lines.append(f"| {r['dataset']} | {r['split']} | {r['seed']} | {r['method']} | {r['selected_epoch']} | {pc} | "+
                     " | ".join(f"{m[k]:.6f}" for k in ("mcc","f1","auc","auprc","recall","fpr","accuracy","precision"))+" |")
    lines += ["", "| Dataset | Split | Method | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        values=[]
        for key in ("p_c","mcc","f1","auc","auprc","recall","fpr","accuracy","precision"):
            v=r["metrics"].get(key)
            values.append((f"{100*v['mean']:.2f}% ± {100*v['std']:.2f}pp" if key=="p_c" else
                           f"{v['mean']:.6f} ± {v['std']:.6f}") if v else "—")
        lines.append(f"| {r['dataset']} | {r['split']} | {r['method']} | "+" | ".join(values)+" |")
    deltas=[]
    for dataset in datasets:
        for split in ("valid","test"):
            values={r["method"]:r["metrics"] for r in aggregate if r["dataset"]==dataset and r["split"]==split}
            comparisons = ((("context_only","vulnerable_only"),("context_only","repaired_only"),
                            ("context_only","rgcl"),("vulnerable_only","rgcl"),("repaired_only","rgcl"))
                           if any(m in methods for m in RGCL_REGION_TARGETS) else
                           (("context_only","random_region"),("random_region","rgcl"),("context_only","rgcl"))
                           if "random_region" in methods else
                           (("plm_baseline","context_only"),("context_only","rgcl"),("plm_baseline","rgcl")))
            for before,after in comparisons:
                if before not in values or after not in values:
                    continue
                delta={k:values[after][k]["mean"]-values[before][k]["mean"] for k in values[after]}
                deltas.append(dict(dataset=dataset,split=split,before=before,after=after,metrics=delta))
                primary="p_c" if dataset=="primevul_paired" else "mcc"
                wins=sum(next(r for r in records if (r['dataset'],r['split'],r['seed'],r['method'])==(dataset,split,seed,after))['metrics'][primary] >
                         next(r for r in records if (r['dataset'],r['split'],r['seed'],r['method'])==(dataset,split,seed,before))['metrics'][primary] for seed in SEEDS)
                lines.extend(["",f"{dataset} {split}, {before} → {after}: mean {primary} {delta[primary]:+.6f}; improved in {wins}/3 seeds. "
                              f"AUC {delta['auc']:+.6f}; MCC {delta['mcc']:+.6f}; FPR {delta['fpr']:+.6f}."])
    result["deltas"]=deltas
    if any(m in methods for m in RGCL_REGION_TARGETS):
        from .model import RobertaTokenizerFast
        from .training import paired_loader, region_supervision_statistics

        rows = read_jsonl(data_root(data_dir, "primevul_paired")/"primevul_paired/train.lines.jsonl")
        tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False)
        features = paired_loader(rows, tokenizer, "baseline").dataset
        coverage = {target: region_supervision_statistics(features, target) for target in (None, 0, 1)}
        supervision = []
        lines.extend(["", "## 真实区域覆盖与实际训练监督", "",
            "有效区域要求512输入中同时存在修改 Token 和背景 Token。跨物理行 Token 按原 bounded_pool 规则分摊，"
            "表中 Token 质量是分摊后的数量，不是代码语义质量评分。空区域、全区域和 Padding 沿用原处理。", "",
            "| Train side | Functions | Eligible Diff functions | Side coverage | Foreground Token mass | Background Token mass |",
            "|---|---:|---:|---:|---:|---:|"])
        for label, side in coverage[None]["by_target"].items():
            lines.append(f"| {'Vulnerable' if label == '1' else 'Repaired'} | {side['functions']} | "
                f"{side['eligible_functions']} | {side['eligible_functions']/side['functions']:.2%} | "
                f"{side['supervised_region_tokens']:g} | {side['supervised_background_tokens']:g} |")
        lines.extend(["", "| Seed | Method | Supervised functions / all Train | Foreground Token mass / epoch | Background Token mass / epoch | Actual function visits (16 epochs) |",
                      "|---|---|---:|---:|---:|---:|"])
        for entry in result["experiments"]:
            method = entry["method"]
            config = json.loads((Path(entry["run"])/"training.metrics.json").read_text())
            stats = coverage[RGCL_REGION_TARGETS.get(method)]
            active = method != "context_only"
            visits = config.get("region_function_visits")
            if method in RGCL_REGION_TARGETS:
                if config["bounded_attention"]["side_supervision"] != stats:
                    raise ValueError("saved single-side token mass does not reproduce from original Train")
                masks = read_jsonl(Path(entry["run"])/"train.attention_regions.jsonl")
                if len(masks) != len(rows):
                    raise ValueError("single-side training omitted functions")
                for row, feature, mask in zip(rows, features, masks):
                    expected = feature["line_labels"] if row["target"] == RGCL_REGION_TARGETS[method] else [0]*len(feature["line_labels"])
                    if mask != dict(sample_key=row["sample_key"], line_numbers=feature["line_numbers"], labels=expected):
                        raise ValueError("actual region masks differ from the stipulated true Diff side")
                if visits != 16 * stats["supervised_functions"]:
                    raise ValueError("actual auxiliary function visits do not match all 16 epochs")
            record = dict(seed=entry["seed"], method=method, functions=len(rows),
                supervised_functions=stats["supervised_functions"] if active else 0,
                foreground_tokens=stats["supervised_region_tokens"] if active else 0,
                background_tokens=stats["supervised_background_tokens"] if active else 0,
                actual_function_visits=visits, visits_recorded=method in RGCL_REGION_TARGETS)
            supervision.append(record)
            lines.append(f"| {record['seed']} | {method} | {record['supervised_functions']}/{len(rows)} | "
                f"{record['foreground_tokens']:g} | {record['background_tokens']:g} | "
                f"{visits if visits is not None else '— (historical)'} |")
        result.update(side_supervision=supervision, original_train_coverage=coverage[None])
        lines.extend(["", "新增六组的实际函数访问次数来自可恢复训练状态，并核对每个样本保存的真实区域掩码。"
            "历史对照不改写训练日志；其每轮监督数量由同一原始 Train 编码和损失有效性条件重算。"
            "漏洞侧与修复侧可监督函数数及 Token 质量不同，且原损失在每批有效函数间平均。"
            "因此本消融同时改变监督来源与覆盖，不能把性能差异直接归因于两侧监督的语义质量，也不作数量补偿。", "",
            "结果回答固定协议下各侧监督的实际效果；不能据此认定统计显著性。"
            "三种子分别评价，均值与标准差以种子为单位，不把重复训练的同一测试样本当作独立样本。"])
        lines.extend(["", "## 固定协议下的结论", ""])
        test_values = {r["method"]: r["metrics"] for r in aggregate if r["split"] == "test"}
        for before, after in (("context_only", "vulnerable_only"), ("context_only", "repaired_only"),
                              ("vulnerable_only", "rgcl"), ("repaired_only", "rgcl")):
            if before not in test_values or after not in test_values:
                continue
            delta = {k: test_values[after][k]["mean"] - test_values[before][k]["mean"] for k in test_values[after]}
            differences = [next(r for r in records if (r["split"], r["seed"], r["method"]) == ("test", seed, after))["metrics"]["p_c"]
                         - next(r for r in records if (r["split"], r["seed"], r["method"]) == ("test", seed, before))["metrics"]["p_c"]
                           for seed in SEEDS]
            lines.append(f"{before} → {after}：Test 平均 P-C {100*delta['p_c']:+.2f} 个百分点，"
                f"MCC {delta['mcc']:+.6f}，Recall {100*delta['recall']:+.2f} 个百分点，"
                f"修复侧 FPR {100*delta['fpr']:+.2f} 个百分点；"
                f"P-C 提升/持平/下降的种子数为 {sum(d>0 for d in differences)}/{sum(d==0 for d in differences)}/{sum(d<0 for d in differences)}。")
        lines.extend(["", "验证记录：tests.log（区域损失、训练、恢复回归）、report-tests.log（四方法指标与覆盖分母），"
            "gpu-smoke.json / primevul_paired/gpu-smoke.json（真实 GPU 梯度、原 RGCL 权重恢复及单函数推理）。"
            "新增参数为0，初始化及采样 RNG 不变；推理不读取区域标签。"])
        if (root/"tests.sandbox.log").exists():
            lines.append("首次沙箱测试因 DataLoader 多进程套接字权限失败，原日志保留于 tests.sandbox.log；"
                         "相同测试已在宿主环境重跑通过，没有修改断言或跳过失败。")
    if "random_region" in methods:
        supervision = []
        lines.extend(["", "## 随机监督核对", "",
            "随机子集仅以真实区域的 Token 总量为约束，不利用真实修改位置筛选候选。"
            "Token 质量按原物理行聚合分摊，逐函数精确保持；空区域和全区域继续不计算区域损失。"
            "不强制排除与真实区域自然重合的抽样，精确 Token 总量可能限制可选子集。", "",
            "| Seed | Eligible functions | Exact token mass | Unchanged eligible | Token overlap / true positive tokens |",
            "|---|---:|---:|---:|---:|"])
        for entry in result["experiments"]:
            if entry["method"] != "random_region":
                continue
            path = Path(entry["run"])
            config = json.loads((path/"training.metrics.json").read_text())
            stats = config["bounded_attention"]["randomization"]
            actual = read_jsonl(path/"train.random_regions.jsonl")
            if ([r["sample_key"] for r in actual] != config["training_members"] or
                any(r["diff_region_tokens"] != r["random_region_tokens"] for r in actual) or
                sum(r["eligible"] for r in actual) != stats["eligible_functions"]):
                raise ValueError(f"{path}: random supervision records and training configuration differ")
            row = dict(stats, seed=entry["seed"])
            row["token_overlap_fraction"] = stats["overlap_tokens"] / stats["positive_tokens"] if stats["positive_tokens"] else 0.
            supervision.append(row)
            lines.append(f"| {entry['seed']} | {stats['eligible_functions']}/{stats['functions']} | "
                         f"{stats['exact_token_mass_matches']}/{stats['functions']} | "
                         f"{stats['unchanged_eligible_functions']}/{stats['eligible_functions']} | {row['token_overlap_fraction']:.2%} |")
        result["random_supervision"] = supervision
        lines.extend(["", "全部新增训练均完成16轮，三种子均完整保留；历史 Context-only/RGCL 未重复训练或改写。"
                      "初始化、采样、恢复与单函数推理检查见 tests.log、resume-tests.log 及 gpu-smoke.json。"
                      "Test 在完整 Valid P-C 选模后一次评价；没有依据 Test 改动方法、参数或阈值。"])
    write_json(root/"comparison.json",result)
    (root/"report.md").write_text("\n".join(lines)+"\n")
    return result
