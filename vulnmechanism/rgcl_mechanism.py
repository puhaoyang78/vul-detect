"""RQ3: repair-guided pairwise classification; frozen official Test predictions."""

import csv
import difflib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .data import modified_lines, read_jsonl, write_jsonl
from .metrics import paired_metrics
from .rgcl_study import BACKBONE_MODELS, SEEDS, inspect_run, run_path

METHODS = ("context_only", "rgcl")
STATES = ("p_c", "p_v", "p_b", "p_r")
VISIBILITY = ("full", "partial", "invisible", "no_direct_diff")


def line_counts(feature):
    """Match bounded_pool: a cross-line Token contributes 1/copies per line."""
    members = [set(t for t in ts if feature["code_mask"][t])
               for ts in feature["line_token_indices"]]
    copies = Counter(t for ts in members for t in ts)
    if set(copies) != {t for t, valid in enumerate(feature["code_mask"]) if valid}:
        raise ValueError("physical lines do not cover the valid code Tokens")
    return [math.fsum(1 / copies[t] for t in ts) for ts in members]


def coverage_record(row, feature):
    gt = set(row["lines_ground_truth"])
    numbers = feature["line_numbers"]
    complete = {n for n, ok in zip(numbers, feature["line_complete"]) if ok}
    visible = gt.intersection(numbers)
    category = ("no_direct_diff" if not gt else "full" if gt <= complete
                else "partial" if visible else "invisible")
    counts = line_counts(feature)
    foreground = math.fsum(c for n, c in zip(numbers, counts) if n in gt)
    total = math.fsum(counts)
    actual = math.fsum(c for c, y in zip(counts, feature["line_labels"]) if y == 1)
    if not math.isclose(actual, foreground, abs_tol=1e-9):
        raise ValueError("PrimeVul effective labels differ from the historical Diff rule")
    eligible = foreground > 0 and total - foreground > 0
    selected_tokens = {t for n, ts in zip(numbers, feature["line_token_indices"])
                       if n in gt for t in ts if feature["code_mask"][t]}
    return dict(sample_key=row["sample_key"], pair_key=row["pair_key"], target=row["target"],
        visibility=category, source_tokens=feature["source_tokens"], code_tokens=total,
        truncated=feature["source_tokens"] > 510, diff_lines=sorted(gt),
        visible_diff_lines=sorted(visible), fully_visible_diff_lines=sorted(gt & complete),
        partial_diff_lines=sorted(visible - complete), line_numbers=numbers,
        line_complete=feature["line_complete"], line_token_counts=counts,
        visible_region_token_mass=foreground, visible_region_distinct_tokens=len(selected_tokens),
        background_token_mass=total - foreground, eligible=eligible,
        effective_region_token_mass=foreground if eligible else 0.,
        effective_background_token_mass=total - foreground if eligible else 0.)


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return dict(n=0, mean=None, std=None, quantiles={})
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std(ddof=0)),
                quantiles={str(q): float(np.quantile(a, q)) for q in (0, .05, .25, .5, .75, .95, 1)})


def summarize_coverage(records):
    result = {}
    for side, target in (("all", None), ("vulnerable", 1), ("repaired", 0)):
        rs = [r for r in records if target is None or r["target"] == target]
        categories = {k: sum(r["visibility"] == k for r in rs) for k in VISIBILITY}
        labeled = len(rs) - categories["no_direct_diff"]
        result[side] = dict(functions=len(rs), direct_diff_functions=labeled,
            category_counts=categories,
            category_fraction_all={k: n / len(rs) if rs else None for k, n in categories.items()},
            category_fraction_nonempty={k: categories[k] / labeled if labeled else None
                                       for k in VISIBILITY[:3]},
            eligible_functions=sum(r["eligible"] for r in rs),
            eligible_with_partial_diff_line=sum(r["eligible"] and bool(r["partial_diff_lines"]) for r in rs),
            all_visible_code_is_region=sum(r["visible_region_token_mass"] > 0 and not r["eligible"] for r in rs),
            truncated_functions=sum(r["truncated"] for r in rs),
            diff_lines=sum(len(r["diff_lines"]) for r in rs),
            visible_diff_lines=sum(len(r["visible_diff_lines"]) for r in rs),
            fully_visible_diff_lines=sum(len(r["fully_visible_diff_lines"]) for r in rs),
            partial_diff_lines=sum(len(r["partial_diff_lines"]) for r in rs),
            token_sums={k: math.fsum(r[k] for r in rs) for k in (
                "code_tokens", "visible_region_token_mass", "visible_region_distinct_tokens",
                "effective_region_token_mass", "effective_background_token_mass")},
            eligible_region_tokens=distribution([r["effective_region_token_mass"] for r in rs if r["eligible"]]))
    return result


def align_predictions(rows, predictions, coverage):
    """Join unique sample IDs and check full pair keys, labels and source mapping."""
    if len(rows) != len(coverage) or any(r["sample_key"] != c["sample_key"] for r, c in zip(rows, coverage)):
        raise ValueError("coverage records do not match the source cohort")
    expected = {r["sample_key"] for r in rows}
    by_key = {r["sample_key"]: r for r in predictions}
    if len(expected) != len(rows) or len(by_key) != len(predictions) or set(by_key) != expected:
        raise ValueError("prediction IDs are duplicated, missing or outside the official cohort")
    result = []
    for row, cov in zip(rows, coverage):
        p = by_key[row["sample_key"]]
        if p["pair_key"] != row["pair_key"] or p["target"] != row["target"]:
            raise ValueError("prediction full pair_key/label differs")
        value = p["probability"]
        if not math.isfinite(value) or not 0 <= value <= 1 or p["prediction"] != int(value > .5):
            raise ValueError("invalid probability or changed fixed decision rule")
        if (p["line_numbers"] != cov["line_numbers"] or
                len(p["line_token_counts"]) != len(cov["line_token_counts"]) or
                not np.allclose(p["line_token_counts"], cov["line_token_counts"], rtol=1e-6, atol=1e-5)):
            raise ValueError("prediction physical lines/Token counts differ from source encoding")
        result.append(p)
    paired_metrics(rows, [p["probability"] for p in result])
    return result


def pair_state(a, b):
    return "p_c" if a > .5 and b <= .5 else "p_v" if a > .5 else "p_r" if b > .5 else "p_b"


def classification_records(rows, predictions, seed):
    """One official repair pair per row; no ranking or gap-based decisions."""
    for method in METHODS:
        if len(predictions[method]) != len(rows):
            raise ValueError("complete prediction cohort required")
        paired_metrics(rows, [p["probability"] for p in predictions[method]])
    records, keys = [], set()
    for i in range(0, len(rows), 2):
        a, b = rows[i:i+2]
        if a["pair_key"] in keys:
            raise ValueError("duplicate full pair_key")
        keys.add(a["pair_key"])
        record = dict(seed=seed, pair_key=a["pair_key"], sample_keys=[a["sample_key"], b["sample_key"]])
        for method in METHODS:
            ps = predictions[method][i:i+2]
            if any(p["sample_key"] != r["sample_key"] or p["pair_key"] != r["pair_key"] or
                   p["target"] != r["target"] or p["prediction"] != int(p["probability"] > .5)
                   for p, r in zip(ps, (a, b))):
                raise ValueError("prediction member, pair, label or fixed threshold differs")
            pv, pr = [p["probability"] for p in ps]
            record[method] = dict(vulnerable_probability=pv, repaired_probability=pr,
                vulnerable_prediction=int(pv > .5), repaired_prediction=int(pr > .5), state=pair_state(pv, pr))
        record["transition"] = record["context_only"]["state"] + "->" + record["rgcl"]["state"]
        records.append(record)
    return records


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_classification(records):
    """Count pair states and function error corrections on ONE seed/cohort."""
    if not records or len({r["seed"] for r in records}) != 1:
        raise ValueError("summarize exactly one nonempty seed, never pooled seed repetitions")
    n = len(records)
    if len({r["pair_key"] for r in records}) != n:
        raise ValueError("duplicate repair pairs")
    matrix = [[0] * 4 for _ in STATES]
    events = Counter(corrected_fp=0, new_fp=0, corrected_fn=0, new_fn=0)
    for r in records:
        c, g = [r[m] for m in METHODS]
        for p in (c, g):
            if (p["state"] != pair_state(p["vulnerable_probability"], p["repaired_probability"]) or
                p["vulnerable_prediction"] != int(p["vulnerable_probability"] > .5) or
                p["repaired_prediction"] != int(p["repaired_probability"] > .5)):
                raise ValueError("pair state differs from strict function classifications")
        matrix[STATES.index(c["state"])][STATES.index(g["state"])] += 1
        events.update(corrected_fp=int(c["repaired_prediction"] == 1 and g["repaired_prediction"] == 0),
            new_fp=int(c["repaired_prediction"] == 0 and g["repaired_prediction"] == 1),
            corrected_fn=int(c["vulnerable_prediction"] == 0 and g["vulnerable_prediction"] == 1),
            new_fn=int(c["vulnerable_prediction"] == 1 and g["vulnerable_prediction"] == 0))
    row_totals = list(map(sum, matrix))
    column_totals = [sum(row[j] for row in matrix) for j in range(4)]
    methods = {}
    for method, counts in zip(METHODS, (row_totals, column_totals)):
        methods[method] = dict(pair_counts=dict(zip(STATES, counts)), p_c=counts[0] / n,
            recall=(counts[0]+counts[1]) / n, patched_fpr=(counts[1]+counts[3]) / n,
            false_positives=counts[1]+counts[3], false_negatives=counts[2]+counts[3])
    gains = {state: matrix[i][0] for i, state in enumerate(STATES) if i}
    losses = {state: matrix[0][j] for j, state in enumerate(STATES) if j}
    gained, lost = sum(gains.values()), sum(losses.values())
    before, after = [methods[m] for m in METHODS]
    if (sum(row_totals) != n or sum(column_totals) != n or
        gained-lost != column_totals[0]-row_totals[0] or
        events["new_fp"]-events["corrected_fp"] != after["false_positives"]-before["false_positives"] or
        events["new_fn"]-events["corrected_fn"] != after["false_negatives"]-before["false_negatives"]):
        raise ValueError("transition/error accounting is inconsistent")
    opportunities = dict(corrected_fp=before["false_positives"], new_fp=n-before["false_positives"],
                         corrected_fn=before["false_negatives"], new_fn=n-before["false_negatives"])
    return dict(seed=records[0]["seed"], pairs=n, matrix=matrix, row_totals=row_totals,
        column_totals=column_totals, methods=methods,
        fraction_all=[[value/n for value in row] for row in matrix],
        fraction_source=[[fraction(value, row_totals[i]) for value in row] for i, row in enumerate(matrix)],
        pc=dict(gained=gained, lost=lost, net=gained-lost, retained=matrix[0][0],
            gained_fraction=gained/n, lost_fraction=lost/n, net_fraction=(gained-lost)/n,
            loss_fraction_of_original_pc=fraction(lost, row_totals[0]),
            gain_sources=gains, gain_source_shares={k:fraction(v, gained) for k,v in gains.items()},
            loss_destinations=losses),
        errors={k:dict(count=events[k], fraction_all=events[k]/n, opportunities=opportunities[k],
                       fraction_opportunity=fraction(events[k], opportunities[k])) for k in events},
        net_fp=events["new_fp"]-events["corrected_fp"], net_fn=events["new_fn"]-events["corrected_fn"])


def mean_sd(values):
    # An undefined conditional rate is not imputed or silently removed.
    return dict(mean=statistics.mean(values), std=statistics.stdev(values)) if all(
        v is not None for v in values) else dict(mean=None, std=None)


def aggregate_classification(summaries, records):
    if sorted(r["seed"] for r in summaries) != list(SEEDS):
        raise ValueError("aggregate exactly seeds 42, 1234 and 123456")
    cohorts = [{(r["pair_key"], tuple(r["sample_keys"])) for r in records if r["seed"] == seed} for seed in SEEDS]
    if not cohorts[0] or any(c != cohorts[0] for c in cohorts[1:]) or any(r["pairs"] != len(cohorts[0]) for r in summaries):
        raise ValueError("the three seeds must repeat exactly the same cohort")
    return dict(seeds=list(SEEDS), pairs_per_seed=len(cohorts[0]), std_definition="sample SD, ddof=1; n=3 seeds on the same cohort",
        matrix=[[mean_sd([r["matrix"][i][j] for r in summaries]) for j in range(4)] for i in range(4)],
        fraction_all=[[mean_sd([r["fraction_all"][i][j] for r in summaries]) for j in range(4)] for i in range(4)],
        fraction_source=[[mean_sd([r["fraction_source"][i][j] for r in summaries]) for j in range(4)] for i in range(4)],
        methods={m:{k:mean_sd([r["methods"][m][k] for r in summaries])
                    for k in ("p_c", "recall", "patched_fpr", "false_positives", "false_negatives")} for m in METHODS},
        pc={k:mean_sd([r["pc"][k] for r in summaries]) for k in ("gained", "lost", "net", "retained",
            "gained_fraction", "lost_fraction", "net_fraction", "loss_fraction_of_original_pc")},
        gain_sources={s:mean_sd([r["pc"]["gain_sources"][s] for r in summaries]) for s in STATES[1:]},
        gain_source_shares={s:mean_sd([r["pc"]["gain_source_shares"][s] for r in summaries]) for s in STATES[1:]},
        errors={k:{field:mean_sd([r["errors"][k][field] for r in summaries])
                   for field in ("count", "fraction_all", "fraction_opportunity")} for k in summaries[0]["errors"]},
        net_fp=mean_sd([r["net_fp"] for r in summaries]), net_fn=mean_sd([r["net_fn"] for r in summaries]))


def plot_transitions(summaries, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    maximum = max(max(row) for r in summaries for row in r["matrix"])
    labels = [s.upper().replace("_", "-") for s in STATES]
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
        for ax, r in zip(axes, summaries):
            im = ax.imshow(r["matrix"], cmap="Blues", vmin=0, vmax=maximum)
            ax.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
                   xlabel="RGCL state", ylabel="Context-only state", title=f"Seed {r['seed']} | n={r['pairs']} pairs")
            for i in range(4):
                for j in range(4):
                    value = r["matrix"][i][j]
                    percent = r["fraction_source"][i][j]
                    annotation = f"{value}\n({100*percent:.1f}%)" if percent is not None else f"{value}\n(N/A)"
                    ax.text(j, i, annotation, ha="center", va="center", fontsize=10,
                            color="white" if value > maximum*.65 else "black")
                    if i and not j:
                        ax.add_patch(Rectangle((j-.47, i-.47), .94, .94, fill=False, linewidth=1.7,
                                               edgecolor="#009E73"))
                    elif j and not i:
                        ax.add_patch(Rectangle((j-.47, i-.47), .94, .94, fill=False, linewidth=1.7,
                                               edgecolor="#D55E00", linestyle="--"))
        fig.colorbar(im, ax=axes, shrink=.78, label="Repair pairs (shared count scale)")
        for suffix in ("png", "svg"):
            fig.savefig(root / f"classification_transitions.{suffix}", dpi=200)
        plt.close(fig)
    path = root / "classification_transitions.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def write_tables(root, result):
    with (root/"classification_transitions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["seed", "from_state", "to_state", "count", "all_pairs", "fraction_all", "source_pairs", "fraction_source"])
        for r in result["summaries"]:
            for i, before in enumerate(STATES):
                for j, after in enumerate(STATES):
                    writer.writerow([r["seed"], before, after, r["matrix"][i][j], r["pairs"], r["fraction_all"][i][j],
                                     r["row_totals"][i], r["fraction_source"][i][j]])
    with (root/"classification_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        keys = ("gained", "lost", "net", "gained_fraction", "lost_fraction", "net_fraction")
        writer.writerow(["seed", "pairs", *[f"pc_{k}" for k in keys],
            *[f"{m}_{k}" for m in METHODS for k in ("p_c", "recall", "patched_fpr")],
            *result["summaries"][0]["errors"], "net_fp", "net_fn"])
        for r in result["summaries"]:
            writer.writerow([r["seed"], r["pairs"], *[r["pc"][k] for k in keys],
                *[r["methods"][m][k] for m in METHODS for k in ("p_c", "recall", "patched_fpr")],
                *[v["count"] for v in r["errors"].values()], r["net_fp"], r["net_fn"]])


def write_case_candidates(root, records):
    groups = {}
    for before, after in (("p_v", "p_c"), ("p_b", "p_c"), ("p_r", "p_c"), ("p_c", "p_v"), ("p_c", "p_b")):
        matches = {}
        for r in records:
            if (r["context_only"]["state"], r["rgcl"]["state"]) == (before, after):
                matches.setdefault(r["pair_key"], []).append(r["seed"])
        groups[before+"->"+after] = [dict(pair_key=k, seeds=matches[k]) for k in sorted(
            matches, key=lambda k:(-len(matches[k]), k))]
    write_json(root/"classification_candidates.json", dict(
        selection="transition type, then number of matching seeds, then full pair_key; no length/visibility/edit-size selection",
        groups=groups))


def analyze(data_dir, reference_dir, output_dir):
    """RQ3 uses only already selected Test predictions; no encoding/inference."""
    root, data = Path(output_dir), Path(data_dir)/"primevul_paired"
    rows = read_jsonl(data/"test.jsonl")
    coverage = json.loads((root/"coverage.json").read_text())
    source_mapping = read_jsonl(root/"test.coverage.jsonl")
    if len(rows) != 1128 or len({r["pair_key"] for r in rows}) != 564:
        raise ValueError("RQ3 requires all 564 official PrimeVul Test pairs")
    for side, stats in summarize_coverage(source_mapping).items():
        if stats != coverage["test"][side]:
            raise ValueError("preserved coverage statistics do not match the existing records")
    records, summaries, experiments = [], [], []
    for seed in SEEDS:
        predictions, metrics = {}, {}
        for method in METHODS:
            path = run_path(reference_dir, "primevul_paired", seed, method)
            info = inspect_run(path, data_dir, "primevul_paired", seed, method, BACKBONE_MODELS["unixcoder"], checkpoint=False)
            if not (path/"best-p_c.pt").is_file():
                raise FileNotFoundError(path/"best-p_c.pt")
            saved = json.loads((path/"test.metrics.json").read_text())
            if saved["selected_epoch"] != info["selected_epoch"]:
                raise ValueError("Test predictions use a different preselected checkpoint")
            pred = align_predictions(rows, read_jsonl(path/"test.predictions.jsonl"), source_mapping)
            value = paired_metrics(rows, [p["probability"] for p in pred])
            if any(value[k] != saved["metrics"][k] for k in value):
                raise ValueError("saved official Test metrics do not reproduce")
            predictions[method], metrics[method] = pred, value
            experiments.append(dict(seed=seed, method=method, run=str(path), selected_epoch=info["selected_epoch"],
                checkpoint=str(path/"best-p_c.pt"), predictions=str(path/"test.predictions.jsonl"),
                full_training_and_valid_selection_checked=True, weights_loaded=False,
                metrics={k:v for k,v in value.items() if not k.startswith("pair_ranking")}))
        members = classification_records(rows, predictions, seed)
        summary = summarize_classification(members)
        for method in METHODS:
            if summary["methods"][method]["pair_counts"] != metrics[method]["pair_counts"] or any(
                summary["methods"][method][k] != metrics[method][k] for k in ("p_c", "recall", "patched_fpr")):
                raise ValueError("matrix margins disagree with official paired metrics")
        records.extend(members)
        summaries.append(summary)
        print(f"seed {seed}: matrix total=564, gained P-C={summary['pc']['gained']}, lost={summary['pc']['lost']}, net={summary['pc']['net']}", flush=True)
    result = dict(title="RQ3: Repair-Guided Pairwise Classification Analysis", states=list(STATES),
        protocol=dict(dataset="PrimeVul official paired Test", backbone="UniXcoder", seeds=list(SEEDS),
            pairs_per_seed=564, functions_per_seed=1128, threshold=.5, comparison=">", input_tokens=512,
            k=5, region_loss_weight=.1, selection="existing full Valid P-C checkpoint; unchanged",
            trained=False, inference_run=False, test_used_for_selection=False,
            test_diff_use="only post-hoc source cases", strata="none; no function-length/visibility/edit-size analysis",
            denominator="matrix/all-pair rates:564; row rates:source state; gain shares:new P-C; error opportunities:old errors/correct functions",
            seed_statistics="mean and sample SD, ddof=1; same 564 pairs across seeds, never 1692 independent pairs"),
        coverage_background=coverage, experiments=experiments, summaries=summaries,
        aggregate=aggregate_classification(summaries, records))
    write_jsonl(root/"paired_classifications.jsonl", records)
    write_json(root/"rq3.json", result)
    write_tables(root, result)
    plot_transitions(summaries, root)
    write_case_candidates(root, records)
    case_count = write_cases(root, data, rows, records)
    write_report(root, result)
    write_json(root/"classification_checks.json", dict(passed=True, seeds=list(SEEDS),
        full_pair_keys=True, complete_same_cohort=True, pairs_per_seed=564,
        matrix_totals=[sum(map(sum,r["matrix"])) for r in summaries],
        matrix_row_column_totals_match_original_metrics=True, error_balance_identities=True,
        original_test_predictions=6, function_predictions_checked=6768,
        classification_pair_records=len(records), cases=case_count,
        source_diff_alignment_checked=True, trained=False, inference_run=False,
        test_used_for_selection=False, coverage_reused_without_changes=True))
    return result


CASE_STUDIES = (
    ("primevul_paired:test:444", "P-V→P-C：修复版本误报得到纠正",
     "原函数直接读取p[0]到p[3]；补丁新增 `if (end - p < 4) return 0;`，在剩余数据不足四字节时提前返回。"
     "Seed42/1234中，漏洞函数保持判为漏洞，修复函数由漏洞改判正常，形成P-V→P-C；Seed123456保持P-C。"
     "这是一项明确的修复侧分类改善，不能仅凭输出变化推断模型关注了检查行。"),
    ("primevul_paired:test:319", "P-B→P-C：原本漏检的漏洞函数被识别",
     "Vim的eval7处理嵌套表达式，并调用递归的eval1/eval_func路径。补丁新增静态递归计数器，在达到1000层时返回错误，"
     "进入受保护部分后递增，并在函数末尾递减，限制递归深度。三个种子都由P-B变为P-C："
     "漏洞函数由正常改判漏洞，修复函数仍判正常。Seed42中修复概率也升高至0.489171，但没有越过固定阈值；"
     "因此该例的新增P-C来自纠正漏洞漏检，并非要求修复概率必须下降。"),
    ("primevul_paired:test:318", "P-C→P-V：引入修复版本误报",
     "RTCP BYE的reason.slen原本只受目标缓冲区大小和报文声明长度约束，随后从pkt+9复制。"
     "补丁新增 `reason.slen = PJ_MIN(reason.slen, size-9);`，进一步限制实际报文剩余长度。"
     "Seed1234/123456由P-C退化为P-V：漏洞侧仍判漏洞，修复侧由正常误判为漏洞；Seed42保持P-C。"
     "这是新增误报，不能把补丁具有清晰安全含义或预测置信度提高当作修复识别成功。"),
    ("primevul_paired:test:504", "P-C→P-B：引入漏洞函数漏检",
     "TIPC密钥接收函数先按size分配内存，再按报文提供的keylen执行memcpy，原有大小一致性检查发生在复制之后。"
     "补丁将最小报文长度、keylen上界及size一致性验证移到分配和复制之前，并调整解锁出口。"
     "Seed42/1234由P-C变为P-B：漏洞函数由漏洞漏判为正常，修复函数仍为正常；Seed123456保持P-B。"
     "这说明所选RGCL模型也会丢失原模型已经正确的漏洞识别；不能从概率变化确定它如何处理了检查与复制的顺序。"),
)


def write_cases(root, data, rows, records):
    oracle = read_jsonl(data/"test.oracle.jsonl")
    fields = ("sample_key", "pair_key", "target", "func")
    if [{k:r[k] for k in fields} for r in oracle] != [{k:r[k] for k in fields} for r in rows]:
        raise ValueError("post-hoc case sources differ from the original Test input")
    by_key = {r["pair_key"]:oracle[i:i+2] for i,r in enumerate(oracle) if i % 2 == 0}
    lines = ["# RQ3 典型配对分类案例", "",
        "复用既定Test预测与原始双版本源码。四例分别说明P-V→P-C、P-B→P-C、P-C→P-V和P-C→P-B；"
        "优先复用上一轮已有案例，并选择同类转移在多个种子重复出现、代码修改语义可核实的实例。"
        "候选按转移类型、出现该转移的种子数和完整pair_key排序，未按函数长度、可见性或修改规模筛选。", "",
        "所有案例都展示三个种子；它们用于说明分类行为，不能用案例频率推断总体规律。"
        "表中概率显示9位有效数字，判定始终使用原始概率严格>0.5。源码Diff来自原文，保留上下文，"
        "展示用unified Diff；修改行标签另以正式双向_line_diff核对。Test Diff仅作事后解释。", ""]
    for key, title, note in CASE_STUDIES:
        a, b = by_key[key]
        for row, other in ((a,b), (b,a)):
            if row["lines_ground_truth"] != modified_lines(row["func"], other["func"]):
                raise ValueError("case Diff labels do not match the formal source alignment")
        lines += [f"## {key}：{title}", "", f"项目 `{a['project']}`；修复提交 `{a['commit_id']}`。"
            f"原始成员 `{a['sample_key']}` / `{b['sample_key']}`，标签1/0。", "", note, "",
            "| 种子 | Context漏洞概率 | Context修复概率 | RGCL漏洞概率 | RGCL修复概率 | 分类转移 |",
            "|---:|---:|---:|---:|---:|---|"]
        for seed in SEEDS:
            r = next(r for r in records if r["pair_key"] == key and r["seed"] == seed)
            c, g = [r[m] for m in METHODS]
            lines.append(f"| {seed} | {c['vulnerable_probability']:.9g} | {c['repaired_probability']:.9g} | "
                f"{g['vulnerable_probability']:.9g} | {g['repaired_probability']:.9g} | "
                f"{state_label(c['state'])}→{state_label(g['state'])} |")
        diff = "".join(difflib.unified_diff(a["func"].splitlines(True), b["func"].splitlines(True),
                                           fromfile="vulnerable", tofile="repaired"))
        lines += ["", "```diff", diff.rstrip("\n"), "```", ""]
    (root/"cases.md").write_text("\n".join(lines), encoding="utf-8")
    return len(CASE_STUDIES)


def state_label(state):
    return state.upper().replace("_", "-")


def write_report(root, result):
    percent = lambda x: f"{100*x:.2f}%" if x is not None else "N/A"
    ms = lambda value, scale=1: f"{scale*value['mean']:.2f}±{scale*value['std']:.2f}" if value["mean"] is not None else "N/A"
    aggregate, summaries = result["aggregate"], result["summaries"]
    pc_before, pc_after = [aggregate["methods"][m]["p_c"] for m in METHODS]
    lines = ["# RQ3: Repair-Guided Pairwise Classification Analysis", "",
        "**研究问题：修复引导如何改变漏洞函数及其修复版本的实际二分类结果？**", "",
        f"在固定Test协议下，RGCL的P-C从{ms(pc_before,100)}%提高到{ms(pc_after,100)}%，"
        f"每个种子平均新增{ms(aggregate['pc']['gained'])}对P-C、丢失{ms(aggregate['pc']['lost'])}对，"
        f"净增加{ms(aggregate['pc']['net'])}对，即{ms(aggregate['pc']['net_fraction'],100)}个百分点。"
        "新增正确配对主要来自纠正修复版本误报；完整结果也显示新的误报和漏检，不能将P-C收益解读为全面改善。", "",
        "## 协议与定义", "",
        "使用PrimeVul官方完整Test的564对/1128个函数，UniXcoder，Seed42、1234、123456，"
        "Context-only与RGCL。复用原完整Valid P-C选出的六个checkpoint对应的保存预测；"
        "不训练、不编码、不重新推理，不选择模型、Epoch或阈值。函数判定始终为严格 `prob > 0.5`，"
        "等于0.5判正常。RGCL保留512 Token、K=5、λ=0.1及Token加权Diff区域损失。", "",
        "P-C=(漏洞,正常)，两侧均正确；P-V=(漏洞,漏洞)，修复侧误报；P-B=(正常,正常)，漏洞侧漏检；"
        "P-R=(正常,漏洞)，两侧均错误。每个转移矩阵的行是Context-only状态，列是RGCL状态。", "",
        "统计以完整sample_key和pair_key对齐，核对标签、预测、保存的物理行/Token映射及所选Epoch。"
        "每个种子矩阵总数必须为564，行合计与Context-only原指标一致，列合计与RGCL原指标一致。"
        "三种子均值±样本标准差（ddof=1）的统计单位是种子；三个模型重复评价同一批564对，"
        "不将其计为1692个独立配对。", "",
        "## 1. 配对分类结果", "",
        "| 种子 | 方法 | P-C对数 | P-V | P-B | P-R | P-C | 漏洞Recall | 修复侧FPR |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        for m in METHODS:
            v = r["methods"][m]
            counts = " | ".join(str(v["pair_counts"][s]) for s in STATES)
            lines.append(f"| {r['seed']} | {m} | {counts} | {percent(v['p_c'])} | {percent(v['recall'])} | {percent(v['patched_fpr'])} |")
    for m in METHODS:
        v = aggregate["methods"][m]
        lines.append(f"\n{m}三种子：P-C {ms(v['p_c'],100)}%，Recall {ms(v['recall'],100)}%，修复侧FPR {ms(v['patched_fpr'],100)}%。")
    lines += ["", "## 2. 新增正确配对及分类退化", "",
        "前三列括号为占该种子新增P-C的比例；新增、丢失和净增加括号为占完整564对的比例。"
        "均值行的比例先在每个种子内计算，再取均值和标准差，不采用跨种子次数合并后的比例。", "",
        "| 种子 | P-V→P-C | P-B→P-C | P-R→P-C | 新增P-C | 丢失P-C | 净增加P-C |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        p = r["pc"]
        gains = [f"{p['gain_sources'][s]} ({percent(p['gain_source_shares'][s])})" for s in STATES[1:]]
        cells = [f"{p[k]} ({percent(p[k+'_fraction'])})" for k in ("gained", "lost", "net")]
        lines.append(f"| {r['seed']} | " + " | ".join(gains+cells) + " |")
    gains = [f"{ms(aggregate['gain_sources'][s])} ({ms(aggregate['gain_source_shares'][s],100)}%)" for s in STATES[1:]]
    cells = [f"{ms(aggregate['pc'][k])} ({ms(aggregate['pc'][k+'_fraction'],100)}%)" for k in ("gained", "lost", "net")]
    lines += ["| 均值±SD | " + " | ".join(gains+cells) + " |", "",
        f"新增P-C中，P-V→P-C平均占{ms(aggregate['gain_source_shares']['p_v'],100)}%，"
        f"P-B→P-C占{ms(aggregate['gain_source_shares']['p_b'],100)}%，"
        f"P-R→P-C占{ms(aggregate['gain_source_shares']['p_r'],100)}%。"
        "三次实验中，最大的新增来源均为P-V→P-C，说明修复引导确实帮助一部分修复函数从误报转为正常，"
        "同时保留对应漏洞函数的正确检出。P-B→P-C则说明收益也包含漏洞漏检的纠正。", "",
        "新增P-C的构成比例不等于原错误状态的转化率；前者还受到Context-only原有错误数量影响。"
        "以下以对应来源状态为分母，避免将较多的纠错数量解释为更高的条件纠错率。", "",
        "| 种子 | P-V→P-C / 原P-V | P-B→P-C / 原P-B | P-R→P-C / 原P-R |",
        "|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['seed']} | " + " | ".join(percent(r["fraction_source"][i][0]) for i in (1,2,3)) + " |")
    lines += ["| 均值±SD | " + " | ".join(ms(aggregate["fraction_source"][i][0],100)+"%" for i in (1,2,3)) + " |", "",
        "| 种子 | P-C→P-V：新误报 | P-C→P-B：新漏检 | P-C→P-R：两侧新错 | 保持P-C | 原P-C退化率 |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        p = r["pc"]
        cells = [str(p["loss_destinations"][s]) for s in STATES[1:]]
        lines.append(f"| {r['seed']} | " + " | ".join(cells) + f" | {p['retained']} | {percent(p['loss_fraction_of_original_pc'])} |")
    loss_cells = [ms(aggregate["matrix"][0][j]) for j in (1,2,3)]
    lines += ["| 均值±SD | " + " | ".join(loss_cells) + f" | {ms(aggregate['pc']['retained'])} | {ms(aggregate['pc']['loss_fraction_of_original_pc'],100)}% |", "",
        "上述退化均保留在完整矩阵和统计中；净收益是新增P-C减去丢失P-C，而不是只报告成功迁移。", "",
        "## 3. 全部函数上的纠错与新增错误", "",
        "下面统计全部564个漏洞函数或修复函数上的预测翻转，括号以各侧564个函数为分母。"
        "这是函数级事件，可能在同一配对中同时纠正一侧、损害另一侧，不能将事件数相加当作新增P-C。"
        "按原模型错误/正确成员数归一化的事件比例见rq3.json；各转移的总体与行内比例见classification_transitions.csv。", "",
        "| 种子 | 消除修复误报 | 新增修复误报 | 误报净变化 | 纠正漏洞漏检 | 新增漏洞漏检 | 漏检净变化 |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        cell = lambda k: f"{r['errors'][k]['count']} ({percent(r['errors'][k]['fraction_all'])})"
        lines.append(f"| {r['seed']} | {cell('corrected_fp')} | {cell('new_fp')} | {r['net_fp']:+d} | "
            f"{cell('corrected_fn')} | {cell('new_fn')} | {r['net_fn']:+d} |")
    cell = lambda k: f"{ms(aggregate['errors'][k]['count'])} ({ms(aggregate['errors'][k]['fraction_all'],100)}%)"
    lines += [f"| 均值±SD | {cell('corrected_fp')} | {cell('new_fp')} | {ms(aggregate['net_fp'])} | "
        f"{cell('corrected_fn')} | {cell('new_fn')} | {ms(aggregate['net_fn'])} |", "",
        "公式核对：修复侧FPR变化=(新增误报−消除误报)/564；漏洞Recall变化=(纠正漏检−新增漏检)/564。"
        "Seed42和123456的Recall提高，但修复侧FPR也升高；Seed1234的修复侧FPR下降，同时Recall下降。"
        "因而三种子的主要新增P-C来源相似，整体误报/漏检权衡的方向并不一致。", "",
        f"三种子修复侧FPR为{ms(aggregate['methods']['context_only']['patched_fpr'],100)}%→"
        f"{ms(aggregate['methods']['rgcl']['patched_fpr'],100)}%，Recall为{ms(aggregate['methods']['context_only']['recall'],100)}%→"
        f"{ms(aggregate['methods']['rgcl']['recall'],100)}%。"
        "可以确认局部分类纠错及净P-C收益，不能据此声称RGCL在全部修复函数上降低误报。", "",
        "## 4. 完整状态转移矩阵", "",
        "![三个种子的完整4×4配对分类转移矩阵：数字为配对数，括号为该来源状态内的比例](classification_transitions.png)", "",
        "[可编辑矢量SVG](classification_transitions.svg)。所有面板使用相同计数色阶；"
        "绿色实线框标出新增P-C，橙色虚线框标出丢失P-C，其余转移与对角线均完整展示。"
        "P-R来源数量较少，行内百分比必须结合原始计数理解。", ""]
    labels = [state_label(s) for s in STATES]
    for r in summaries:
        lines += [f"### Seed{r['seed']}（总计{r['pairs']}对）", "",
            "| Context→RGCL | " + " | ".join(labels) + " | 行合计 |", "|---|---:|---:|---:|---:|---:|"]
        for i, s in enumerate(labels):
            lines.append(f"| {s} | " + " | ".join(map(str,r["matrix"][i])) + f" | {r['row_totals'][i]} |")
        lines += ["| 列合计 | " + " | ".join(map(str,r["column_totals"])) + f" | {r['pairs']} |", ""]
    lines += ["### 三种子单元格均值±SD（每个种子同为564对）", "",
        "| Context→RGCL | " + " | ".join(labels) + " |", "|---|---:|---:|---:|---:|"]
    for i, s in enumerate(labels):
        lines.append(f"| {s} | " + " | ".join(ms(v) for v in aggregate["matrix"][i]) + " |")
    lines += ["", "## 5. 典型代码语义与分类变化", ""]
    for key, title, note in CASE_STUDIES:
        lines += [f"**{key}，{title}。** {note}", ""]
    lines += ["[完整源码Diff、原始成员/提交及逐种子预测概率](cases.md)。这些例子是事后解释，"
        "不能将个别成功案例当作普遍规律，也不能根据概率变化推断注意力落在哪些行。", "",
        "## 6. 修复监督覆盖率背景", "",
        "沿用既有统计，不重新构造标签或开展可见性、长度、修改规模的分类分层。"
        "完整/部分/不可见均指原始直接Diff区域在512 Token预算下的可见性；无直接Diff单列。"
        "Token质量按跨行成员数守恒，区域损失要求前景和背景都非空。", "",
        "| 划分 | 函数 | 完整 | 部分 | 不可见 | 无直接Diff | 可用区域监督 | 有效区域Token质量 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, values in result["coverage_background"].items():
        r = values["all"]
        cells = [f"{r['category_counts'][s]} ({percent(r['category_fraction_all'][s])})" for s in VISIBILITY]
        lines.append(f"| {split} | {r['functions']} | " + " | ".join(cells) +
            f" | {r['eligible_functions']} ({percent(r['eligible_functions']/r['functions'])}) | {r['token_sums']['effective_region_token_mass']:g} |")
    lines += ["", "Train为实际监督可用量，Valid/Test为事后可用性。"
        "正式编码保持原函数整体BPE、前510个源码Token加CLS/SEP、补齐至512；空白消耗预算但不参与聚合，注释保留。"
        "历史PrimeVul规则允许截断末行的可见Token参与区域损失。完整统计仍存于coverage.json及三个coverage.jsonl文件。", "",
        "## 中文结论", "",
        f"三个种子的净P-C变化分别为{'、'.join(str(r['pc']['net']) for r in summaries)}对，"
        f"{sum(r['pc']['net']>0 for r in summaries)}次增加、{sum(r['pc']['net']==0 for r in summaries)}次持平、"
        f"{sum(r['pc']['net']<0 for r in summaries)}次下降。"
        "RGCL在既定Test配对分类中获得平均净P-C收益，新增正确配对的首要来源是修复版本误报的纠正，"
        "其次是漏洞函数漏检的纠正。它也引入了新的误报、漏检和原P-C退化，整体修复侧FPR没有稳定降低。"
        "这些结果支持修复监督改善部分漏洞—修复配对的实际分类，不能扩展为全部函数或全部种子上全面改善。", "",
        "## 复现与检查", "",
        "`python -m vulnmechanism.cli rgcl-mechanism`。复用六组固定Test预测，未训练或执行前向。"
        "完整逐对状态见paired_classifications.jsonl，矩阵与比例见classification_transitions.csv，"
        "分类汇总见classification_summary.csv，三种子原始及汇总统计见rq3.json。"
        "classification_checks.json记录每个矩阵564对、行列合计、原始指标及错误收支恒等式核对；"
        "本轮最小测试日志为classification_tests.log。历史概率/间隔/ECDF结果保留，不参与论文版RQ3正文结论。", ""]
    (root/"report.md").write_text("\n".join(lines), encoding="utf-8")
