"""Measured search/VAT checks and the four-switch pilot result table."""

from __future__ import annotations
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import time

import torch

from . import augmentation, model as core
from .artifacts import write_json
from .data import balanced_subset, class_counts, read_jsonl, test_rows, validation_rows
from .syntax import surface_materials
from .vat import VAT_CONFIG, VAT_DIRECTIONS, vat_loss


COMBINATIONS = tuple((s, d) for s in ("hardest", "boundary") for d in VAT_DIRECTIONS)


def _csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def probe(data_dir, output_dir, checkpoint_path, *, sample_limit=32, repeats=3, device="auto"):
    """Train-only balanced probe; never update a saved model or tune on Test."""
    if repeats < 1:
        raise ValueError("positive timing repeats required")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dataset = cp["dataset"]
    train = read_jsonl(Path(data_dir) / dataset / "train.jsonl")
    rows = balanced_subset(train, sample_limit)
    materials = surface_materials(train)
    device = core.resolve_device(device)
    builder = core.InputBuilder(core.AutoTokenizer.from_pretrained(cp["model_path"]))
    model = core.load_model(cp, device)
    if device.type == "cuda":
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    budget = builder.source_max_length - len(builder.source_prefix) - 2
    batch_size = 4
    report = dict(
        checkpoint=str(checkpoint_path), dataset=dataset, seed=42,
        selected_epoch=cp["selected_epoch"], samples=class_counts(rows),
        sample_keys=[r["sample_key"] for r in rows], repeats=repeats,
        device=str(device), hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        torch=torch.__version__, cuda=torch.version.cuda,
        protocol="balanced source Train; fixed per-function RNG; no optimizer updates; timing includes final KL backward",
        search={}, vat={},
    )
    # Warm both forward and backward kernels; do not count this in measured costs.
    warm = builder.codebert_batch(rows[:batch_size], device=device)
    with torch.no_grad(), core.auxiliary_mode(model):
        model(*warm)
    for method in VAT_DIRECTIONS:
        model.zero_grad(set_to_none=True)
        loss, _ = vat_loss(model, rows[:batch_size], builder, device=device,
                           noise_seed=42, direction_method=method)
        loss.sum().backward()
    model.zero_grad(set_to_none=True)
    _sync(device)

    traces = []
    sources = {r["raw_source"] for r in rows}
    selected_sources = {}
    search_runs = {s: [] for s in ("hardest", "boundary")}
    for repeat in range(repeats):
        # Alternate execution order to limit warm-cache/order effects.
        for selection in (("hardest", "boundary") if repeat % 2 == 0 else ("boundary", "hardest")):
            costs, reasons, entries = Counter(), Counter(), []

            def score(batch):
                values = []
                with torch.no_grad(), core.auxiliary_mode(model):
                    for at in range(0, len(batch), batch_size):
                        inputs = builder.codebert_batch(batch[at:at + batch_size], device=device)
                        values.extend(model(*inputs).float().cpu().tolist())
                        costs.update(calls=1, function_forwards=inputs[-1], encoder_sequences=len(inputs[0]),
                                     encoder_token_slots=inputs[0].numel(), encoder_active_tokens=int(inputs[1].sum()))
                return values

            _sync(device)
            started = time.monotonic()
            for at in range(0, len(rows), batch_size):
                entries.extend(augmentation.search_surface_views(
                    rows[at:at + batch_size], score, builder.tokenizer, budget, materials,
                    epoch=1, selection=selection,
                ))
            _sync(device)
            seconds = time.monotonic() - started
            queries = sum(e["candidate_queries"] + e["original_queries"] for e in entries)
            if queries != costs["function_forwards"]:
                raise AssertionError("reported search queries differ from actual scored functions")
            reasons.update(e["selection_reason"] for e in entries)
            search_runs[selection].append(dict(seconds=seconds, **costs))
            if repeat == 0:
                selected_sources[selection] = {
                    e["sample_key"]: None if e["selected"] is None else e["selected"]["raw_source"]
                    for e in entries
                }
                report["search"][selection] = dict(
                    candidate_queries=sum(e["candidate_queries"] for e in entries),
                    original_queries=sum(e["original_queries"] for e in entries),
                    generated=sum(len(e["visited"]) for e in entries),
                    visibility_encodings=sum(e["visibility_encodings"] for e in entries),
                    selection_reasons=dict(reasons),
                )
                for e in entries:
                    sources.update(c["raw_source"] for c in e["visited"])
                    if e["selection_reason"] == "original_wrong_random" and (
                        e["candidate_queries"] or any("logit" in c or "loss" in c for c in e["visited"])
                    ):
                        raise AssertionError("wrong-original variants must remain unscored")
                    traces.append(dict(
                        selection=selection, **{k: v for k, v in e.items() if k not in ("selected", "visited")},
                        selected=None if e["selected"] is None else {k: v for k, v in e["selected"].items() if k != "raw_source"},
                        visited=[{k: v for k, v in c.items() if k != "raw_source"} for c in e["visited"]],
                    ))
        print(f"checkpoint search probe repeat {repeat + 1}/{repeats}", flush=True)
    for selection, runs in search_runs.items():
        report["search"][selection].update(runs=runs, median_seconds=statistics.median(r["seconds"] for r in runs))
    first_layers = {
        (e["selection"], e["sample_key"]): [c["units"] for c in e["visited"] if len(c["units"]) == 1]
        for e in traces
    }
    if any(first_layers[("hardest", r["sample_key"])] != first_layers[("boundary", r["sample_key"])] for r in rows):
        raise AssertionError("selection strategy changed the initial candidate generator")
    report["search"]["same_first_layer_functions"] = len(rows)
    report["search"]["same_selected_source"] = sum(
        selected_sources["hardest"][r["sample_key"]] == selected_sources["boundary"][r["sample_key"]]
        for r in rows
    )
    # Check the cached-offset path against the original InputBuilder outside timing.
    for source in sources:
        ids = augmentation._source_visibility(source, builder.tokenizer, budget, {})[0]
        if ids != builder._encode(source, budget):
            raise AssertionError("offset/source tokenization differs from original input encoding")
        plain = builder.codebert_batch([dict(raw_source=source)], device="cpu")
        cached = builder.codebert_batch([dict(raw_source=source, _source_token_ids=ids)], device="cpu")
        if plain[-1] != cached[-1] or any(not torch.equal(a, b) for a, b in zip(plain[:-1], cached[:-1])):
            raise AssertionError("cached source IDs changed model inputs")
    report["input_cache_check"] = dict(unique_sources=len(sources), tokens_equal=True, batch_tensors_equal=True)

    vat_runs = {d: [] for d in VAT_DIRECTIONS}
    directions = {}
    details = {}
    for repeat in range(repeats):
        for method in (VAT_DIRECTIONS if repeat % 2 == 0 else VAT_DIRECTIONS[::-1]):
            costs = Counter()
            all_finite, gradient_nonzero = True, True
            _sync(device)
            started = time.monotonic()
            for at in range(0, len(rows), batch_size):
                batch_rows = rows[at:at + batch_size]
                model.zero_grad(set_to_none=True)
                losses, info, delta = vat_loss(
                    model, batch_rows, builder, device=device, noise_seed=42 + at,
                    direction_method=method, return_delta=True,
                )
                losses.mean().backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                all_finite &= bool(torch.stack([g.isfinite().all() for g in grads]).all())
                gradient_nonzero &= bool(torch.stack([g.abs().max() for g in grads]).max() > 0)
                costs.update({k: info[k] for k in ("function_forwards", "encoder_sequences", "encoder_token_slots",
                                                   "encoder_active_tokens", "eligible", "zero_direction", "zero_initial_projection")
                              if info[k] is not None})
                if repeat == 0:
                    _, _, owners, _, _ = builder.codebert_batch(batch_rows, device=device)
                    for i, row in enumerate(batch_rows):
                        key = row["sample_key"]
                        directions[(method, key)] = delta[owners == i].detach().cpu().flatten().double()
                        details.setdefault(key, dict(sample_key=key, label=row["label"], windows=int((owners == i).sum())))
                        details[key][method + "_kl"] = float(losses[i].detach())
                        details[key][method + "_radius"] = float(directions[(method, key)].norm())
                        details[key][method + "_reference_logit"] = info["reference_logits"][i]
                        if method == "finite_difference":
                            details[key]["finite_difference_probe_logit_change"] = info["probe_logit_change"][i]
                        else:
                            details[key]["logit_first_order_probe_change"] = VAT_CONFIG["xi"] * info["raw_initial_projection"][i]
            _sync(device)
            vat_runs[method].append(dict(seconds=time.monotonic() - started, **costs,
                                          all_parameter_gradients_finite=all_finite,
                                          every_batch_has_nonzero_parameter_gradient=gradient_nonzero))
            model.zero_grad(set_to_none=True)
        print(f"checkpoint VAT probe repeat {repeat + 1}/{repeats}", flush=True)
    for row in rows:
        key = row["sample_key"]
        a, b = (directions[(method, key)] for method in VAT_DIRECTIONS)
        norm = a.norm() * b.norm()
        details[key]["signed_cosine"] = float(a.dot(b) / norm) if norm > 0 else None
    cosines = [v["signed_cosine"] for v in details.values() if v["signed_cosine"] is not None]
    report["vat"]["signed_cosine"] = dict(
        count=len(cosines), minimum=min(cosines) if cosines else None,
        median=statistics.median(cosines) if cosines else None,
        mean=statistics.mean(cosines) if cosines else None,
        negative=sum(c < 0 for c in cosines), below_0_9=sum(c < 0.9 for c in cosines),
    )
    for method, runs in vat_runs.items():
        kl_values = [v[method + "_kl"] for v in details.values()]
        report["vat"][method] = dict(
            runs=runs, median_seconds=statistics.median(r["seconds"] for r in runs),
            mean_kl=statistics.mean(kl_values), median_kl=statistics.median(kl_values),
            min_kl=min(kl_values), max_kl=max(kl_values),
        )
    report["vat"]["logit_smaller_final_kl"] = sum(
        v["logit_kl"] < v["finite_difference_kl"] for v in details.values()
    )
    _csv(root / "probe.vat.csv", list(details.values()))
    with (root / "probe.search.jsonl").open("w") as handle:
        for entry in traces:
            handle.write(json.dumps(entry) + "\n")
    write_json(root / "probe.json", report)
    return report


def comparison(data_dir, output_dir, dataset, *, train_limit=1024, valid_limit=256, test_limit=256):
    """Read completed canonical train/evaluate outputs, and verify their budgets."""
    root = Path(output_dir)
    members = []
    populations, selected_sources = {}, {}
    for split, rows, limit in (
        ("train", read_jsonl(Path(data_dir) / dataset / "train.jsonl"), train_limit),
        ("valid", validation_rows(Path(data_dir) / dataset), valid_limit),
        ("test", test_rows(Path(data_dir) / dataset), test_limit),
    ):
        selected = balanced_subset(rows, limit)
        populations[split] = class_counts(rows)
        selected_sources[split] = {r["raw_source"] for r in selected}
        members.extend(dict(split=split, sample_key=r["sample_key"], label=r["label"]) for r in selected)
    _csv(root / "members.csv", members)
    train_members = Counter(r["sample_key"] for r in members if r["split"] == "train")
    valid_members = [(r["sample_key"], r["label"]) for r in members if r["split"] == "valid"]
    table, configs, materials, test_keys, saved_runs = [], [], [], [], {}
    fixed_candidates = []
    for selection, direction in COMBINATIONS:
        name = selection + "_" + direction
        run = root / name / dataset / "hard_aug_vat"
        config = json.loads((run / "training.metrics.json").read_text())
        if config["augmentation"]["selection"] != selection or config["vat"]["direction"] != direction:
            raise ValueError("output label differs from the trained switch configuration")
        configs.append(config)
        materials.append(json.loads((run.parent / "materials.json").read_text()))
        test_keys.append([(r["sample_key"], r["label"]) for r in read_jsonl(run / "test.predictions.jsonl")])
        clean = json.loads((run / "test.metrics.json").read_text())["metrics"]
        robust = json.loads((run / "robustness.metrics.json").read_text())
        if robust["adaptive"]["query_limit"] != 48:
            raise ValueError("adaptive evaluation budget differs from the original full search")
        epochs = [r for r in read_jsonl(run / "best.training.jsonl") if r["event"] == "epoch"]
        trace = read_jsonl(run / "training.candidates.jsonl")
        if [(r["sample_key"], r["label"]) for r in read_jsonl(run / "valid.predictions.jsonl")] != valid_members:
            raise ValueError("observed validation members differ across combinations")
        for epoch in epochs:
            entries = [e for e in trace if e["epoch"] == epoch["epoch"]]
            if Counter(e["sample_key"] for e in entries) != train_members:
                raise ValueError("observed training members differ across combinations")
            actual_queries = sum(e["candidate_queries"] + e["original_queries"] for e in entries)
            if actual_queries != epoch["search_costs"]["function_forwards"]:
                raise ValueError("logged queries differ from actual training search forwards")
            if any(e["candidate_queries"] or "logit" in e["selected"] or "loss" in e["selected"]
                   for e in entries if e["selection_reason"] == "original_wrong_random"):
                raise ValueError("wrong-original training variants were scored")
        saved_runs[name] = dict(config=config, epochs=epochs, clean=clean, robustness=robust)
        fixed_candidates.append([read_jsonl(run.parent / f"{name}.candidates.jsonl")
                                 for name in ("fixed", "heldout")])
        base = dict(selection=selection, vat_direction=direction, selected_epoch=config["selected_epoch"],
                    optimizer_steps=config["optimizer_steps"],
                    selected_optimizer_step=sum(e["optimizer_steps"] for e in epochs if e["epoch"] <= config["selected_epoch"]),
                    threshold=clean["threshold"],
                    training_seconds=sum(e["epoch_seconds"] for e in epochs),
                    candidate_queries=sum(e["candidate_queries"] for e in epochs),
                    original_queries=sum(e["original_search_queries"] for e in epochs),
                    search_windows=sum(e["search_costs"].get("encoder_sequences", 0) for e in epochs),
                    vat_windows=sum(e["vat_costs"].get("encoder_sequences", 0) for e in epochs),
                    search_token_slots=sum(e["search_costs"].get("encoder_token_slots", 0) for e in epochs),
                    vat_token_slots=sum(e["vat_costs"].get("encoder_token_slots", 0) for e in epochs),
                    clean_accuracy=clean["accuracy"], clean_f1=clean["f1"], clean_mcc=clean["mcc"],
                    clean_auc=clean["auc"], clean_auprc=clean["auprc"])
        for attack in ("fixed", "heldout", "adaptive"):
            costs = (robust["adaptive"]["costs"] if attack == "adaptive"
                     else robust["evaluation_costs"][attack])
            for scope in ("strict_semantic", "input_only_stress"):
                entry = robust[attack + "_overall" if attack != "adaptive" else attack][scope]
                worst = entry["worst_complete"]
                table.append(dict(base, attack=attack, scope=scope,
                                  attack_total_queries=costs["function_forwards"],
                                  attack_total_windows=costs["encoder_sequences"],
                                  attack_total_token_slots=costs["encoder_token_slots"],
                                  attack_total_seconds=costs["total_seconds"],
                                  worst_accuracy=worst["accuracy"], worst_mcc=worst["mcc"],
                                  worst_auc=worst["auc"], asr=entry["asr"]["rate"],
                                  asr_count=entry["asr"]["count"], asr_denominator=entry["asr"]["denominator"],
                                  eligible=entry["eligible"], total=entry["total"]))
    if any(c["training_config"] != configs[0]["training_config"] or
           c["optimizer_steps"] != configs[0]["optimizer_steps"] or
           c["diagnostic_limits"] != dict(train=train_limit, valid=valid_limit) or
           c["model_path"] != configs[0]["model_path"] for c in configs):
        raise ValueError("four-combination initial model, configuration or training budgets differ")
    expected = [(r["sample_key"], r["label"]) for r in members if r["split"] == "test"]
    expected_materials = surface_materials(balanced_subset(
        read_jsonl(Path(data_dir) / dataset / "train.jsonl"), train_limit
    ))
    if (any(k != expected for k in test_keys) or any(m != expected_materials for m in materials)
            or any(c != fixed_candidates[0] for c in fixed_candidates)):
        raise ValueError("four-combination Test members or Train-only materials differ")
    _csv(root / "comparison.csv", table)
    result = dict(dataset=dataset, seed=42, diagnostic=True,
                  limits=dict(train=train_limit, valid=valid_limit, test=test_limit),
                  populations=populations,
                  sampling="one random.Random(42), label 0 then 1, without replacement; retain source order",
                  exact_source_overlap={a + "_" + b: len(selected_sources[a] & selected_sources[b])
                                        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))},
                  observed_train_valid_test_members_verified=True,
                  training_config=configs[0]["training_config"],
                  evaluation="unchanged fixed/held-out generators and full depth-three adaptive search, 48 candidates per function",
                  runs=saved_runs, results=table)
    write_json(root / "results.json", result)
    _report(root, result)
    return result


def _report(root, result):
    measured = json.loads((root / "probe.json").read_text())
    rows = result["results"]
    clean_rows = [r for r in rows if r["attack"] == "adaptive" and r["scope"] == "strict_semantic"]
    limits = result["limits"]
    config = result["training_config"]
    lines = [
        "# Hardest / Boundary 与 VAT 方向消融",
        "",
        f"Seed 42；{result['dataset']} 固定平衡子集 Train/Valid/Test = "
        f"{limits['train']}/{limits['valid']}/{limits['test']}，各 Split 正负各半。"
        f"各组从同一预训练 CodeBERT 初始化，{config['epochs']} Epoch、有效 batch "
        f"{config['batch_size'] * config['gradient_accumulation']}、{clean_rows[0]['optimizer_steps']} 个优化器步；"
        "仅源 Valid 选择 Epoch 和阈值，素材仅取实际 Train。旧权重和正式 Test 不变。",
        "",
        "## 同一真实 checkpoint 的计算比较",
        "",
        f"checkpoint: `{measured['checkpoint']}`；平衡源 Train {measured['samples']['samples']} 条，"
        f"{measured['repeats']} 次交替顺序测量的耗时中位数。窗口/token 计显式前向，"
        "不计梯度检查点反向重计算；VAT 耗时包含方向反向和最终 KL 反向。",
        "",
        "| 搜索 | 变体查询 | 原函数查询 | 窗口 | Token slots（含 padding） | 秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("hardest", "boundary"):
        e = measured["search"][name]
        costs = e["runs"][0]
        lines.append(f"| {name} | {e['candidate_queries']} | {e['original_queries']} | "
                     f"{costs['encoder_sequences']} | {costs['encoder_token_slots']} | {e['median_seconds']:.3f} |")
    lines += ["", "| VAT | 函数前向 | 窗口 | Token slots | 秒 | 平均 KL | 中位 KL |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in VAT_DIRECTIONS:
        e = measured["vat"][name]
        costs = e["runs"][0]
        lines.append(f"| {name} | {costs['function_forwards']} | {costs['encoder_sequences']} | "
                     f"{costs['encoder_token_slots']} | {e['median_seconds']:.3f} | "
                     f"{e['mean_kl']:.6g} | {e['median_kl']:.6g} |")
    cosine = measured["vat"]["signed_cosine"]
    cosine_text = (
        f"Signed cosine：n={cosine['count']}，中位数 {cosine['median']:.8f}，"
        f"均值 {cosine['mean']:.8f}，最小值 {cosine['minimum']:.8f}，负值 {cosine['negative']} 条。"
        if cosine["count"] else "无非零方向对，signed cosine 不适用。"
    )
    lines += ["", cosine_text +
              "按每函数展开后的扰动计算。"
              "逐函数半径、KL、微小探针 logit 变化见 `probe.vat.csv`；参数梯度有限性和零方向计数见 `probe.json`。",
              "", "反例（保持符号，不取绝对 cosine）：", "",
              "| 样本 | Signed cosine | FD KL | Logit KL | FD 探针 Δlogit | xi·gᵀd |",
              "|---|---:|---:|---:|---:|---:|"]
    with (root / "probe.vat.csv").open() as handle:
        diagnostics = list(csv.DictReader(handle))
    for e in diagnostics:
        if e["signed_cosine"] and float(e["signed_cosine"]) < 0.9:
            lines.append(f"| {e['sample_key']} | {float(e['signed_cosine']):.8f} | "
                         f"{float(e['finite_difference_kl']):.6g} | {float(e['logit_kl']):.6g} | "
                         f"{float(e['finite_difference_probe_logit_change']):.3g} | "
                         f"{float(e['logit_first_order_probe_change']):.3g} |")
    reference_difference = max(abs(float(e["finite_difference_reference_logit"]) -
                                   float(e["logit_reference_logit"])) for e in diagnostics)
    lines += ["", f"两分支中心 logit 的最大差为 {reference_difference:.3g}。"
              "上述反例的微小探针增量与中心一阶预测反号，说明当前 xi 下有限差分方向"
              "与中心 Hessian 方向存在差异；这些测量尚未分离浮点误差与局部非线性的贡献。"
              "epsilon=1 的最终 KL 也不保证由中心二阶近似排序。"]
    lines += ["", "边界策略选样原因：`" + json.dumps(measured["search"]["boundary"]["selection_reasons"], ensure_ascii=False) + "`。"
              "无合法候选时只训练原函数；原函数已错的随机变体不附加 logit/loss。"
              "零梯度/零投影不使用替代方向；单元测试覆盖零方向退化。有限差分未计算原始 gᵀd，其零投影计数不适用。",
              "", "## 相同训练预算的学习效果", "",
              "选中模型对应更新步数依次为 " + "、".join(str(e["selected_optimizer_step"]) for e in clean_rows) +
              "；均沿用源 Valid 选模。", "",
              "| 策略 | VAT | Epoch | 阈值 | Clean Acc | F1 | MCC | AUC | AUPRC |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e in clean_rows:
        lines.append(f"| {e['selection']} | {e['vat_direction']} | {e['selected_epoch']} | {e['threshold']:.2f} | "
                     + " | ".join(f"{e['clean_' + m]:.4f}" for m in ("accuracy", "f1", "mcc", "auc", "auprc")) + " |")
    for scope, label in (("strict_semantic", "严格语义保持"), ("input_only_stress", "函数名等输入压力测试")):
        lines += ["", "### " + label, "",
                  "| 策略 | VAT | 扰动 | Worst Acc | Worst MCC | ASR（分子/分母） | 可扰动/总数 |",
                  "|---|---|---|---:|---:|---:|---:|"]
        for e in rows:
            if e["scope"] == scope:
                asr = "NA" if e["asr"] is None else f"{100 * e['asr']:.2f}%"
                lines.append(f"| {e['selection']} | {e['vat_direction']} | {e['attack']} | "
                             f"{e['worst_accuracy']:.4f} | {e['worst_mcc']:.4f} | "
                             f"{asr} ({e['asr_count']}/{e['asr_denominator']}) | {e['eligible']}/{e['total']} |")
    lines += ["", "ASR 分母为该扰动范围内可扰动且 Clean 正确的函数；Worst-case 包含原函数和全部 Test 成员。"
              "四组合固定/留出候选相同，Adaptive 保留完整 48 候选预算，不启用 Boundary。",
              "", "计算收益依据固定 checkpoint 探针；学习效果依据四次独立训练。训练 epoch 耗时记录于结果，"
              "可能受到并行运行的资源竞争影响，不作为隔离速度基准。该单 Seed 小样本消融不能代替完整数据实验。",
              ""]
    (root / "report.md").write_text("\n".join(lines))
