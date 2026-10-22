"""Source-line alignment and train-only repair-region ranking for attention MIL."""

from bisect import bisect_right
from collections import Counter
from difflib import SequenceMatcher
import csv
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .artifacts import write_json
from .data import official_paired_rows, paired_groups, paired_subset, read_jsonl, write_jsonl
from .metrics import metrics

METHODS = ("bce", "mil_bce", "mil_patch_rank", "mil_random_rank")
RANK_WEIGHT = 0.1


def patch_regions(before, after, *, context=2):
    """Half-open character spans; a zero-width edit anchors to adjacent lines.

    These are repair contexts, not per-line vulnerability annotations.
    """
    if context < 0:
        raise ValueError("patch context must be nonnegative")
    lines = [before.splitlines(keepends=True), after.splitlines(keepends=True)]
    boundaries = [[0], [0]]
    for side in (0, 1):
        for line in lines[side]:
            boundaries[side].append(boundaries[side][-1] + len(line))
    regions, hunks = [[], []], []
    for tag, i, j, a, b in SequenceMatcher(None, *lines, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        hunks.append(dict(kind=tag, vulnerable_lines=[i, j], patched_lines=[a, b]))
        for side, (start, end) in enumerate(((i, j), (a, b))):
            n = len(lines[side])
            if start == end:
                start, end = max(0, start - 1), min(n, end + 1)
            start, end = max(0, start-context), min(n, end+context)
            regions[side].append([boundaries[side][start], boundaries[side][end]])
    return regions, hunks


def attach_line_metadata(row, ids, offsets, special_ids):
    """Map actual source tokens to physical nonempty lines, without any diff.

    Use untrimmed byte-level offsets, including indentation. A token belongs
    to the physical line containing its first source character. Newlines on
    nonempty lines remain tokens; blank lines have no instance representation.
    """
    source = row["raw_source"]
    if len(ids) != len(offsets) or any(a < 0 or a > b or b > len(source) for a,b in offsets):
        raise ValueError("source token IDs and character offsets are not aligned")
    lines = source.splitlines(keepends=True)
    bounds = [0]
    for line in lines:
        bounds.append(bounds[-1] + len(line))
    physical, reasons = [], Counter()
    for token, (start, end) in zip(ids, offsets):
        line = -1
        if token in special_ids:
            reasons["special_tokens"] += 1
        else:
            if start >= end:
                reasons["zero_width_tokens"] += 1
            else:
                at = bisect_right(bounds, start) - 1
                if 0 <= at < len(lines) and lines[at].strip():
                    line = at
                else:
                    reasons["blank_line_tokens"] += 1
        physical.append(line)
    numbers = sorted(set(physical) - {-1})
    dense = {line: i for i, line in enumerate(numbers)}
    row["_source_token_ids"] = tuple(ids)
    row["_token_lines"] = tuple(dense.get(line, -1) for line in physical)
    row["_line_numbers"] = numbers
    row["_line_bounds"] = bounds
    row["_alignment"] = dict(reasons, mapped_tokens=sum(x >= 0 for x in physical),
                              nonempty_lines=sum(bool(line.strip()) for line in lines),
                              mapped_lines=len(numbers))


def matched_random_lines(numbers, candidates, *, seed):
    """Translate the union mask: exactly preserve count, physical span and shape."""
    if not candidates:
        return [], True
    selected = [numbers[i] for i in candidates]
    origin = min(selected)
    offsets = [line - origin for line in selected]
    available = set(numbers)
    starts = [line for line in numbers if all(line + d in available for d in offsets)]
    alternatives = [line for line in starts if line != origin]
    chosen = random.Random(seed).choice(alternatives or starts)
    dense = {line:i for i,line in enumerate(numbers)}
    return [dense[chosen+d] for d in offsets], not alternatives


def region_pair_ranking(risks, line_owners, records, *, field="candidate_lines"):
    """Differentiable top-k shared-risk means; one softplus per effective pair."""
    if len(records) % 2:
        raise ValueError("ranking requires complete vulnerable/patched pairs")
    values = []
    for i in range(0, len(records), 2):
        v,p = records[i:i+2]
        if (v["label"],p["label"]) != (1,0) or v["pair_key"] != p["pair_key"]:
            raise ValueError("ranking roles must be vulnerable then patched")
        if not v["patch_eligible"]:
            continue
        dv,dp = v[field],p[field]
        k=min(3,len(dv),len(dp))
        if not k:
            raise ValueError("eligible ranking pair has an empty region")
        uv=risks[line_owners==i][dv].topk(k).values.mean()
        up=risks[line_owners==i+1][dp].topk(k).values.mean()
        values.append(F.softplus(up-uv))
    return torch.stack(values) if values else risks[:0]


def representation_data(raw_root, builder, *, limit, seed=42, context=2, overfit=False):
    """Read official Train and both Valid files; never open Test."""
    train,anomalies=official_paired_rows(raw_root,"train")
    train=paired_subset(train,limit,seed)
    if overfit:
        auxiliary=[dict(r,split="valid") for r in train]
        valid=[dict(r,dataset="primevul",split="valid") for r in train]
        for r in valid: del r["pair_key"]
        valid_anomalies=[]
    else:
        auxiliary,valid_anomalies=official_paired_rows(raw_root,"valid")
        native=read_jsonl(Path(raw_root)/"PrimeVul_v0.1/primevul_valid.jsonl")
        valid=[dict(sample_key=f"primevul:official:valid:{i+1}",dataset="primevul",split="valid",
                    label=r["target"],raw_source=r["func"],upstream_idx=r["idx"])
               for i,r in enumerate(native)]
    special=set(builder.tokenizer.all_special_ids)
    for rows in (train,valid,auxiliary):
        for at in range(0,len(rows),128):
            group=rows[at:at+128]
            encoded=builder.tokenizer([r["raw_source"] for r in group],add_special_tokens=False,
                                      truncation=False,return_offsets_mapping=True)
            for r,ids,offsets in zip(group,encoded["input_ids"],encoded["offset_mapping"]):
                attach_line_metadata(r,ids,offsets,special)
                if not r["_line_numbers"]:
                    raise ValueError(f"{r['sample_key']}: no mapped nonempty source lines")
    selected={r["_source_token_ids"] for r in train}
    collisions={r["_source_token_ids"] for r in valid+auxiliary if r["_source_token_ids"] in selected}
    removed=({r["pair_key"] for r in train if r["_source_token_ids"] in collisions} if not overfit else set())
    native_count=len(train)//2
    train=[r for r in train if r["pair_key"] not in removed]
    anomaly_keys={r["pair_key"] for r in anomalies}
    coverage,reasons,members=Counter(),Counter(),[]
    prefix_budget=builder.source_max_length-len(builder.source_prefix)-2
    for v,p in paired_groups(train):
        spans,hunks=patch_regions(v["raw_source"],p["raw_source"],context=context)
        distinct=v["_source_token_ids"]!=p["_source_token_ids"]
        prefix_visible=v["_source_token_ids"][:prefix_budget]!=p["_source_token_ids"][:prefix_budget]
        changed_visible=False
        for side,row in enumerate((v,p)):
            bounds=row["_line_bounds"]
            candidates=[i for i,n in enumerate(row["_line_numbers"])
                        if any(bounds[n]<b and bounds[n+1]>a for a,b in spans[side])]
            row["candidate_lines"]=candidates
            row["random_lines"],row["random_forced_same"]=matched_random_lines(
                row["_line_numbers"],candidates,seed=f"{seed}:{row['sample_key']}:random-lines")
            changed={n for h in hunks for n in range(*h["vulnerable_lines" if side==0 else "patched_lines"])}
            changed_visible |= bool(changed.intersection(row["_line_numbers"]))
            row["pair_distinguishable"]=distinct
        flags=[]
        if not distinct: flags.append("identical_encoder_input")
        if not changed_visible: flags.append("change_not_visible")
        if not all(r["candidate_lines"] for r in (v,p)): flags.append("region_mapping_failed")
        if v["pair_key"] in anomaly_keys: flags.append("metadata_anomaly")
        eligible=not flags
        reasons.update(flags or ["eligible"])
        coverage.update(pairs=1,prefix_difference_visible=int(prefix_visible),
                        full_difference_visible=int(distinct),diff_line_visible=int(changed_visible),
                        eligible_pairs=int(eligible),
                        random_forced_same_pairs=int(all(r["random_forced_same"] for r in (v,p))))
        for row in (v,p):
            row["patch_eligible"]=eligible
            row["pair_exclusion_reasons"]=flags
            dense=row["_line_numbers"]
            members.append(dict(sample_key=row["sample_key"],pair_key=row["pair_key"],label=row["label"],
                                source_row=row["source_row"],upstream_idx=row["upstream_idx"],
                                tokens=len(row["_source_token_ids"]),nonempty_lines=len(dense),
                                candidate_lines=[dense[i] for i in row["candidate_lines"]],
                                random_lines=[dense[i] for i in row["random_lines"]],
                                random_forced_same=row["random_forced_same"],
                                eligible=eligible,exclusion_reasons=flags,
                                prefix_difference_visible=prefix_visible,diff_line_visible=changed_visible))
    for a,b in paired_groups(auxiliary):
        a["pair_distinguishable"]=b["pair_distinguishable"]=a["_source_token_ids"]!=b["_source_token_ids"]
    actual={r["_source_token_ids"] for r in train}
    sources={r["raw_source"] for r in train}
    audit=dict(seed=seed,overfit_train_replay=overfit,requested_pairs=limit,
               selected_pairs_before_overlap=native_count,train_pairs=len(train)//2,
               removed_train_pairs_for_valid_overlap=sorted(removed),
               ordinary_valid_functions=len(valid),ordinary_valid_positive=sum(r["label"] for r in valid),
               paired_valid_pairs=len(auxiliary)//2,
               actual_train_valid_exact_overlap=sum(r["raw_source"] in sources for r in valid),
               actual_train_valid_encoded_overlap=sum(r["_source_token_ids"] in actual for r in valid),
               actual_train_paired_valid_encoded_overlap=sum(r["_source_token_ids"] in actual for r in auxiliary),
               coverage=dict(coverage),exclusion_reasons=dict(reasons),metadata_anomalies=anomalies,
               paired_valid_metadata_anomalies=valid_anomalies,
               paired_valid_indistinguishable=sum(not a["pair_distinguishable"] for a,b in paired_groups(auxiliary)),
               alignment={split:dict(sum((Counter(r["_alignment"]) for r in rows),Counter()))
                          for split,rows in (("train",train),("ordinary_valid",valid),("paired_valid",auxiliary))},
               context_lines=context,prefix_source_budget=prefix_budget,
               source_rule="all fixed-stride native windows; inputs independent of diff; average overlapping token states, then nonempty lines",
               random_rule="uniform translation of union of candidate physical lines; exact size/span/shape; exclude original if alternatives exist",
               test_opened=False)
    return train,valid,auxiliary,members,audit


def save_members(run,members,audit):
    write_jsonl(Path(run)/"train.regions.jsonl",members)
    write_json(Path(run)/"input.coverage.json",audit)


def report(output_dir):
    """Aligned saved-prediction comparison; official ordinary Valid selects all models."""
    root=Path(output_dir)
    rows,costs,length_rows,changes=[],[],[],{}
    reference_members=reference_coverage=baseline=native=training_config=None
    for method in METHODS:
        run=root/"primevul_paired"/method
        config=json.loads((run/"training.metrics.json").read_text())
        if training_config is None:
            training_config=config["training_config"]
        elif config["training_config"]!=training_config:
            raise ValueError("ablation training hyperparameters/seeds differ")
        members=read_jsonl(run/"train.regions.jsonl")
        coverage=json.loads((run/"input.coverage.json").read_text())
        if reference_members is None:
            reference_members,reference_coverage=members,coverage
        if members!=reference_members or coverage!=reference_coverage:
            raise ValueError("MIL ablations differ in Train members/regions/coverage")
        if any(coverage[k] for k in ("actual_train_valid_exact_overlap","actual_train_valid_encoded_overlap",
                                    "actual_train_paired_valid_encoded_overlap")):
            raise ValueError("training/validation overlap")
        predictions=read_jsonl(run/"valid.predictions.jsonl")
        if baseline is None: baseline=predictions
        if [(r["sample_key"],r["label"]) for r in predictions]!=[(r["sample_key"],r["label"]) for r in baseline]:
            raise ValueError("ordinary Valid cohorts differ")
        if any(r["threshold"]!=.5 for r in predictions): raise ValueError("threshold must be 0.5")
        clean=json.loads((run/"valid.metrics.json").read_text())
        paired=json.loads((run/"paired.valid.metrics.json").read_text())["metrics"]
        rows.append(dict(method=method,selected_epoch=clean["selected_epoch"],**clean["metrics"],
                         paired_strict_accuracy=paired["strict_pairwise_accuracy"],
                         paired_strict_correct=paired["strict_correct_pairs"],
                         paired_pairs=paired["pairs"],paired_roc_auc=paired["auc"],
                         extra_parameters=config["line_mil"]["extra_parameters"]))
        epochs=[r for r in read_jsonl(run/"best.training.jsonl") if r["event"]=="epoch"]
        selected=max(epochs,key=lambda r:(r["validation"]["auprc"],r["validation"]["mcc"]))
        if clean["selected_epoch"]!=selected["epoch"]: raise ValueError("PR-AUC selection differs")
        costs.append(dict(method=method,optimizer_steps=config["optimizer_steps"],
                          training_seconds=sum(r["training_seconds"] for r in epochs),
                          validation_seconds=sum(r["validation_seconds"] for r in epochs),
                          encoder_windows=sum(r["bce_costs"]["encoder_sequences"] for r in epochs),
                          last_train_bce=epochs[-1]["train_loss"],
                          last_rank_loss=epochs[-1]["pair_regularizer"]))
        changes[method]=dict(corrected=sum(a["prediction"]!=a["label"] and b["prediction"]==b["label"] for a,b in zip(baseline,predictions)),
                             regressed=sum(a["prediction"]==a["label"] and b["prediction"]!=b["label"] for a,b in zip(baseline,predictions)))
        if native is None:
            native=read_jsonl(Path(config["pair_raw_root"])/"PrimeVul_v0.1/primevul_valid.jsonl")
        if len(native)!=len(predictions) or any(r["label"]!=n["target"] or r["sample_key"]!=f"primevul:official:valid:{i+1}"
                                                for i,(r,n) in enumerate(zip(predictions,native))):
            raise ValueError("saved predictions differ from official labels/order")
        for name,lo,hi in (("1-20",1,20),("21-60",21,60),("61-150",61,150),("151+",151,10**12)):
            subset=[r for r,n in zip(predictions,native) if lo<=len(n["func"].splitlines())<=hi]
            if subset:
                length_rows.append(dict(method=method,lines=name,positive=sum(r["label"] for r in subset),
                                        **metrics([r["label"] for r in subset],[r["score"] for r in subset],.5)))
    by_method={r["method"]:r for r in rows}
    if len({(r["optimizer_steps"],r["encoder_windows"]) for r in costs}) != 1:
        raise ValueError("ablation optimizer/encoder budgets differ")
    if len({r["extra_parameters"] for r in rows if r["method"]!="bce"}) != 1:
        raise ValueError("MIL ablations have different parameter budgets")
    true=by_method["mil_patch_rank"];mil=by_method["mil_bce"]
    differences={method:{key:true[key]-by_method[method][key]
                         for key in ("auprc","auc","mcc","f1","paired_strict_accuracy")}
                 for method in ("bce","mil_bce","mil_random_rank")}
    passed=(true["auprc"]>=max(by_method[m]["auprc"] for m in ("bce","mil_bce","mil_random_rank"))+.002
            and true["mcc"]>=mil["mcc"] and true["f1"]>=mil["f1"])
    gate=dict(passed=passed,rule="true-region Valid PR-AUC at least 0.002 above all three controls, with MCC/F1 not below MIL+BCE; then repeat all methods at seeds 43/44; otherwise stop")
    prevalence=reference_coverage["ordinary_valid_positive"]/reference_coverage["ordinary_valid_functions"]
    result=dict(seed=training_config["seed"],sampling_seed=42,
                metrics=rows,costs=costs,length_strata=length_rows,coverage=reference_coverage,
                true_region_differences=differences,constant_score_average_precision=prevalence,
                clean_prediction_changes=changes,seed_gate=gate,test_evaluated=False)
    for name,table in (("comparison.csv",rows),("costs.csv",costs),("length.csv",length_rows)):
        with (root/name).open("w",newline="") as h:
            writer=csv.DictWriter(h,fieldnames=list(table[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(table)
    write_json(root/"results.json",result)
    lines=["# 上下文化行级 attention-MIL 与修复区域排序 pilot", "",
           f"Seed {training_config['seed']}；{reference_coverage['train_pairs']} Train pairs；{training_config['epochs']} Epoch；"
           f"有效Batch {training_config['batch_size']*training_config['gradient_accumulation']}；{costs[0]['optimizer_steps']}次更新。四组同一预训练CodeBERT、输入和Train成员。"
           f"完整源码512-token滑窗；普通官方Valid全部{reference_coverage['ordinary_valid_functions']}函数"
           f"（{reference_coverage['ordinary_valid_positive']}正例），paired Valid {reference_coverage['paired_valid_pairs']}对。"
           "只按普通Valid PR-AUC选模，MCC作平局次级指标；阈值0.5；不读取Test。", "",
           "| 方法 | Epoch | PR-AUC | ROC-AUC | MCC | F1@0.5 | Paired两侧正确 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['selected_epoch']} | "+" | ".join(f"{r[k]:.4f}" for k in
                     ("auprc","auc","mcc","f1","paired_strict_accuracy"))+" |")
    lines += ["",f"普通Valid的恒定分数AP参考值为正例比例 {prevalence:.5f}。", "",
              "| 真实区域组−对照 | ΔPR-AUC | ΔROC-AUC | ΔMCC | ΔF1 |",
              "|---|---:|---:|---:|---:|"]
    for method,d in differences.items():
        lines.append(f"| {method} | "+" | ".join(f"{d[k]:+.5f}" for k in ("auprc","auc","mcc","f1"))+" |")
    c=reference_coverage["coverage"]
    lines += ["",f"差异可见：前2025 token {c['prefix_difference_visible']}/{c['pairs']}；完整输入 {c['full_difference_visible']}/{c['pairs']}；"
              f"映射到修改行 {c['diff_line_visible']}/{c['pairs']}；有效区域监督 {c['eligible_pairs']}/{c['pairs']}。"
              f"排除原因：{reference_coverage['exclusion_reasons']}。所有样本保留函数BCE。",
              f"随机区域两侧都没有替代位置而与真实集合相同：{c['random_forced_same_pairs']}对。"
              "随机区域严格匹配物理行数、跨度和hunk联合形状；这部分对照不能区分区域来源。", "",
              "MIL先按重叠次数平均同一源码token状态，再对非空源码行均值池化；prefix/特殊token/padding不参与。"
              "gated attention和风险线性头共享；函数Logit为行风险的attention加权和。"
              "辅助项只对两侧diff±2行合并集合的top-k均值排序，k=min(3,|Dv|,|Dp|)，权重0.1，按有效pair平均。"
              "没有局部伪标签、cosine、投影头或其他辅助损失。单函数推理不读diff。", "",
              "训练与两个Valid的源码/完整编码重叠为0。使用trim_offsets=False保留缩进Token坐标，未改变源码Token ID。"
              "真实区域正向差值仅是本轮观察，不能自动解释成修复语义的因果收益。", "",
              "| 方法 | 函数总行数 | N | 正例 | PR-AUC | ROC-AUC | MCC | F1@0.5 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in length_rows:
        lines.append(f"| {r['method']} | {r['lines']} | {r['samples']} | {r['positive']} | "+
                     " | ".join("—" if r[k] is None else f"{r[k]:.4f}" for k in ("auprc","auc","mcc","f1"))+" |")
    lines += ["",
              "预设多Seed门槛："+gate["rule"], "",
              "门槛通过，按预设协议复核Seed 43/44。" if passed else "门槛未通过；没有足够证据支持进入多Seed复核，停止本轮pilot。",
              "成本见costs.csv，为Epoch内部的训练和Valid时间，不含初始分词、模型加载及权重写盘。"
              "窗口计数不含梯度检查点重算；本次双卡并发耗时不作独立速度基准。", ""]
    (root/"report.md").write_text("\n".join(lines))
    return dict(output=str(root),metrics=rows,seed_gate=gate)
