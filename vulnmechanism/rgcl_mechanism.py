"""Post-hoc RGCL coverage and paired-probability analysis; never train or select."""

import csv
import difflib
import gc
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .data import modified_lines, paired_features, read_jsonl, write_jsonl
from .metrics import paired_metrics
from .rgcl_study import BACKBONE_MODELS, SEEDS, inspect_run, run_path

METHODS = ("context_only", "rgcl")
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


def paired_records(rows, predictions, coverage, identical, seed, split):
    result = []
    keys = set()
    for i in range(0, len(rows), 2):
        a, b = rows[i:i + 2]
        if a["target"] != 1 or b["target"] != 0 or a["pair_key"] != b["pair_key"] or a["pair_key"] in keys:
            raise ValueError("official vulnerable/repaired pair is incomplete or duplicated")
        keys.add(a["pair_key"])
        r = dict(seed=seed, split=split, pair_key=a["pair_key"],
            sample_keys=[a["sample_key"], b["sample_key"]], project=a.get("project"),
            identical_visible_input=identical[i // 2],
            pure_addition=not a["lines_ground_truth"] and bool(b["lines_ground_truth"]),
            region_visibility=[coverage[j]["visibility"] for j in (i, i + 1)],
            region_eligible=[coverage[j]["eligible"] for j in (i, i + 1)],
            region_tokens=[coverage[j]["visible_region_token_mass"] for j in (i, i + 1)])
        for method in METHODS:
            pv, pr = (predictions[method][j]["probability"] for j in (i, i + 1))
            r[method] = dict(vulnerable=pv, repaired=pr, gap=pv-pr, ordered=pv > pr,
                             tie=pv == pr, state=pair_state(pv, pr))
        c, d = (r[m] for m in METHODS)
        r.update(gap_change=d["gap"]-c["gap"], vulnerable_change=d["vulnerable"]-c["vulnerable"],
                 repaired_change=d["repaired"]-c["repaired"],
                 gained_p_c=c["state"] != "p_c" and d["state"] == "p_c",
                 lost_p_c=c["state"] == "p_c" and d["state"] != "p_c")
        result.append(r)
    return result


def summarize_gaps(records):
    """Same denominator for strict ordering, ties, and both-correct classification."""
    n = len(records)
    result = dict(pairs=n, methods={})
    if not n:
        return result
    for m in METHODS:
        states = Counter(r[m]["state"] for r in records)
        result["methods"][m] = dict(
            gap=distribution([r[m]["gap"] for r in records]),
            vulnerable_probability=distribution([r[m]["vulnerable"] for r in records]),
            repaired_probability=distribution([r[m]["repaired"] for r in records]),
            ordered=sum(r[m]["ordered"] for r in records),
            ordering_rate=sum(r[m]["ordered"] for r in records) / n,
            ties=sum(r[m]["tie"] for r in records),
            reversed=sum(r[m]["gap"] < 0 for r in records),
            pair_counts={s: states[s] for s in ("p_c", "p_v", "p_b", "p_r")}, p_c=states["p_c"] / n,
            ordered_but_not_p_c=sum(r[m]["ordered"] and r[m]["state"] != "p_c" for r in records))
    result.update(
        gap_change=distribution([r["gap_change"] for r in records]),
        vulnerable_change=distribution([r["vulnerable_change"] for r in records]),
        repaired_change=distribution([r["repaired_change"] for r in records]),
        gap_increased=sum(r["gap_change"] > 0 for r in records),
        gap_decreased=sum(r["gap_change"] < 0 for r in records),
        gained_p_c=sum(r["gained_p_c"] for r in records), lost_p_c=sum(r["lost_p_c"] for r in records),
        gained_ordering=sum(not r["context_only"]["ordered"] and r["rgcl"]["ordered"] for r in records),
        lost_ordering=sum(r["context_only"]["ordered"] and not r["rgcl"]["ordered"] for r in records),
        transitions=dict(sorted(Counter(r["context_only"]["state"]+"->"+r["rgcl"]["state"] for r in records).items())))
    return result


def subgroup_records(records):
    return dict(all=records,
        visible_difference=[r for r in records if not r["identical_visible_input"]],
        identical_input=[r for r in records if r["identical_visible_input"]],
        pure_addition=[r for r in records if r["pure_addition"]],
        both_regions_full=[r for r in records if r["region_visibility"] == ["full", "full"]],
        any_invisible_region=[r for r in records if "invisible" in r["region_visibility"]])


def infer_train(run, destination, rows, selected_epoch, device, loader=None):
    """Use the formal single-source evaluator; no Diff annotations enter inference."""
    import torch
    from .evaluation import load_paired, validate_paired
    from .model import resolve_device
    from .training import paired_loader

    cache = destination.with_suffix(".metrics.json")
    if destination.exists():
        saved = json.loads(cache.read_text())
        if saved["run"] != str(run) or saved["selected_epoch"] != selected_epoch:
            raise ValueError("cached frozen Train predictions refer to another checkpoint")
        return read_jsonl(destination), loader
    cp = torch.load(run / "best-p_c.pt", map_location="cpu", weights_only=False)
    if cp["selected_epoch"] != selected_epoch:
        raise ValueError("Train inference checkpoint differs from the preselected Valid epoch")
    device = resolve_device(device)
    model, tokenizer = load_paired(cp, device)
    if loader is None:
        clean = [{k: r[k] for k in ("func", "sample_key", "pair_key", "target")} for r in rows]
        loader = paired_loader(clean, tokenizer, "baseline", batch_size=8)
        if any("line_labels" in f for f in loader.dataset):
            raise ValueError("Diff must never enter frozen inference")
    with torch.no_grad():
        value, predictions = validate_paired(model, rows, tokenizer, "baseline", device, loader=loader)
    write_jsonl(destination, predictions)
    write_json(cache, dict(run=str(run), selected_epoch=selected_epoch, metrics=value,
                           source_only=True, trained=False))
    del model, cp
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, loader


def plot_gaps(records, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 10, "svg.fonttype": "none", "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout="constrained", sharex=True, sharey=True)
        for row, split in enumerate(("train", "valid", "test")):
            for col, seed in enumerate(SEEDS):
                ax = axes[row, col]
                subset = [r for r in records if r["split"] == split and r["seed"] == seed]
                for method, label, color, style in (("context_only", "Context-only", "#0072B2", "-"),
                                                   ("rgcl", "RGCL", "#D55E00", "--")):
                    x = np.sort([r[method]["gap"] for r in subset])
                    ax.step(np.r_[-1., x, 1.], np.r_[0., np.arange(1, len(x)+1)/len(x), 1.],
                            where="post", label=label, color=color, linestyle=style)
                ax.axvline(0, color=".55", linewidth=.8)
                ax.set(xlim=(-1, 1), ylim=(0, 1.01), title=f"{split.title()} | seed {seed} | n={len(subset)}")
                if col == 0:
                    ax.set_ylabel("Cumulative pair fraction")
                if row == 2:
                    ax.set_xlabel("Probability gap: vulnerable - repaired")
        axes[0, 0].legend(loc="upper left")
        for extension in ("svg", "png"):
            fig.savefig(output_dir / f"probability_gap_ecdf.{extension}", dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained", sharex=True, sharey=True)
        for row, split in enumerate(("valid", "test")):
            for col, seed in enumerate(SEEDS):
                ax = axes[row, col]
                rs = [r for r in records if r["split"] == split and r["seed"] == seed]
                for same, label, color, marker in ((False, "Visible difference", "#0072B2", "."),
                                                   (True, "Identical visible input", "#D55E00", "x")):
                    subset = [r for r in rs if r["identical_visible_input"] == same]
                    ax.scatter([r["context_only"]["gap"] for r in subset],
                               [r["rgcl"]["gap"] for r in subset], s=13, marker=marker,
                               color=color, alpha=.55, label=label)
                ax.plot([-1, 1], [-1, 1], linestyle=":", color=".5")
                ax.axhline(0, color=".7", linewidth=.7)
                ax.axvline(0, color=".7", linewidth=.7)
                ax.set(xlim=(-1, 1), ylim=(-1, 1), title=f"{split.title()} | seed {seed}")
                if col == 0:
                    ax.set_ylabel("RGCL probability gap")
                if row == 1:
                    ax.set_xlabel("Context-only probability gap")
        axes[0, 0].legend(fontsize=8, loc="upper left")
        for extension in ("svg", "png"):
            fig.savefig(output_dir / f"paired_gap_changes.{extension}", dpi=180)
        plt.close(fig)
    for name in ("probability_gap_ecdf.svg", "paired_gap_changes.svg"):
        path = output_dir / name
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def analyze(data_dir, reference_dir, output_dir, *, device="cpu"):
    """Fixed three-seed checkpoints; Test is used only for retrospective analysis."""
    from transformers import RobertaTokenizerFast

    root, data = Path(output_dir), Path(data_dir) / "primevul_paired"
    root.mkdir(parents=True, exist_ok=True)
    model_path = BACKBONE_MODELS["unixcoder"]
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path, trim_offsets=False, local_files_only=True)
    all_rows, coverages, identical, coverage_summary = {}, {}, {}, {}
    for split in ("train", "valid", "test"):
        rows = read_jsonl(data / (f"{split}.lines.jsonl" if split != "test" else "test.oracle.jsonl"))
        clean = read_jsonl(data / f"{split}.jsonl")
        paired_metrics(rows, [.5] * len(rows))
        if len({r["pair_key"] for r in rows}) != len(rows) // 2:
            raise ValueError("official pair_key is duplicated across pairs")
        if [{k: r[k] for k in ("sample_key", "pair_key", "target", "func")} for r in rows] != [
                {k: r[k] for k in ("sample_key", "pair_key", "target", "func")} for r in clean]:
            raise ValueError("post-hoc Diff records differ from normal single-source inputs")
        records, signatures = [], []
        for i, row in enumerate(rows):
            if row["lines_ground_truth"] != modified_lines(row["func"], rows[i ^ 1]["func"]):
                raise ValueError("persisted Diff lines disagree with the formal Differ alignment")
            feature = paired_features(row, tokenizer, "baseline")
            records.append(coverage_record(row, feature))
            signatures.append(tuple(feature["input_ids"]))
        all_rows[split], coverages[split] = rows, records
        identical[split] = [signatures[i] == signatures[i+1] for i in range(0, len(rows), 2)]
        coverage_summary[split] = summarize_coverage(records)
        coverage_summary[split].update(pairs=len(rows)//2, identical_visible_pairs=sum(identical[split]),
            both_eligible_pairs=sum(records[i]["eligible"] and records[i+1]["eligible"] for i in range(0, len(rows), 2)),
            either_eligible_pairs=sum(records[i]["eligible"] or records[i+1]["eligible"] for i in range(0, len(rows), 2)))
        write_jsonl(root / f"{split}.coverage.jsonl", records)
        print(f"coverage {split}: {len(rows)} functions; {sum(identical[split])} identical pairs", flush=True)
    write_json(root / "coverage.json", coverage_summary)
    experiment_records, pair_records, summaries, loader = [], [], [], None
    for seed in SEEDS:
        predictions = {s: {} for s in all_rows}
        for method in METHODS:
            run = run_path(reference_dir, "primevul_paired", seed, method)
            info = inspect_run(run, data_dir, "primevul_paired", seed, method, model_path)
            train_mask = run / "train.attention_regions.jsonl"
            if method == "rgcl":
                masks = read_jsonl(train_mask)
                if len(masks) != len(all_rows["train"]) or any(
                    m["sample_key"] != r["sample_key"] or m["line_numbers"] != c["line_numbers"] or
                    m["labels"] != [int(n in r["lines_ground_truth"]) for n in c["line_numbers"]]
                    for m, r, c in zip(masks, all_rows["train"], coverages["train"])):
                    raise ValueError("saved Train region masks differ across the seed/cohort")
            info.update(seed=seed, method=method, predictions={})
            for split, rows in all_rows.items():
                filename = ("train.selected.predictions.jsonl" if split == "train" else
                            "valid.best-p_c.predictions.jsonl" if split == "valid" else "test.predictions.jsonl")
                path = run / filename
                if split == "train" and not path.exists():
                    path = root / "cache" / f"seed{seed}.{method}.train.predictions.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"frozen Train inference: seed {seed} {method}", flush=True)
                    pred, loader = infer_train(run, path, rows, info["selected_epoch"], device, loader)
                else:
                    pred = read_jsonl(path)
                    if split == "train":
                        metadata = json.loads((run / "train.selected.metrics.json").read_text())
                        if metadata["selected_epoch"] != info["selected_epoch"]:
                            raise ValueError("historical Train cache uses a different epoch")
                pred = align_predictions(rows, pred, coverages[split])
                predictions[split][method] = pred
                value = paired_metrics(rows, [p["probability"] for p in pred])
                saved = (info["validation"] if split == "valid" else
                         json.loads((run / "test.metrics.json").read_text())["metrics"] if split == "test" else None)
                if saved is not None and any(k in saved and value[k] != saved[k] for k in value):
                    raise ValueError("recomputed fixed-threshold metrics differ from the saved evaluation")
                info["predictions"][split] = dict(path=str(path), metrics=value)
            experiment_records.append(info)
        for split, rows in all_rows.items():
            pairs = paired_records(rows, predictions[split], coverages[split], identical[split], seed, split)
            pair_records.extend(pairs)
            for group, members in subgroup_records(pairs).items():
                summaries.append(dict(seed=seed, split=split, group=group, **summarize_gaps(members)))
        print(f"finished probability analysis: seed {seed}", flush=True)
    write_jsonl(root / "paired_gaps.jsonl", pair_records)
    aggregate = []
    for split in all_rows:
        for group in subgroup_records([]):
            rs = [r for r in summaries if r["split"] == split and r["group"] == group]
            if not rs[0]["pairs"]:
                continue
            entries = {}
            for method in METHODS:
                for name, fn in (("gap", lambda r: r["gap"]["mean"]),
                                 ("ordering_rate", lambda r: r["ordering_rate"]), ("p_c", lambda r: r["p_c"]),
                                 ("vulnerable_probability", lambda r: r["vulnerable_probability"]["mean"]),
                                 ("repaired_probability", lambda r: r["repaired_probability"]["mean"])):
                    vals = [fn(r["methods"][method]) for r in rs]
                    entries[f"{method}.{name}"] = dict(mean=statistics.mean(vals), std=statistics.stdev(vals))
            for name, fn in (("gap_change", lambda r: r["gap_change"]["mean"]),
                             ("p_c_change", lambda r: (r["gained_p_c"]-r["lost_p_c"])/r["pairs"]),
                             ("ordering_change", lambda r: (r["gained_ordering"]-r["lost_ordering"])/r["pairs"])):
                vals = [fn(r) for r in rs]
                entries[name] = dict(mean=statistics.mean(vals), std=statistics.stdev(vals))
            aggregate.append(dict(split=split, group=group, pairs_per_seed=rs[0]["pairs"], metrics=entries))
    result = dict(experiments=experiment_records, coverage=coverage_summary, summaries=summaries, aggregate=aggregate,
        protocol=dict(seeds=list(SEEDS), threshold=.5, comparison=">", trained=False,
            model_selection="unchanged existing full-Valid P-C checkpoints", test_use="post-hoc diagnostics only",
            std="between-seed sample SD, ddof=1; same cohort is not three independent sample sets",
            distribution_std="within-seed pair population SD, ddof=0",
            visibility="full: every direct modified line complete; partial: some Diff Token visible but not all lines complete; invisible: nonempty original Diff with no visible Diff Tokens; no_direct_diff: empty original Diff",
            encoding="formal RobertaTokenizerFast(trim_offsets=False), no prefix/additional special tokens; first510 source BPE Tokens plus CLS/SEP, pad to512; whitespace consumes budget but has no pooling mass; comments remain source; physical line IDs zero-based",
            token_mass="bounded_pool reciprocal cross-line membership; only foreground>0 AND background>0 contributes region loss; partial final lines retained by historical PrimeVul rule",
            supervision="Train actual; Valid/Test counterfactual availability only, never supplied to inference"))
    write_json(root / "analysis.json", result)
    plot_gaps(pair_records, root)
    write_tables(root, result)
    write_case_candidates(root, all_rows["test"], pair_records)
    write_report(root, result)
    return result


def write_tables(root, result):
    with (root / "gap_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["split", "seed", "group", "pairs", "method", "gap_mean", "gap_median", "gap_std",
                         "ordered", "ties", "reversed", "p_c", "p_v", "p_b", "p_r", "ordered_not_p_c"])
        for r in result["summaries"]:
            for m, v in r["methods"].items():
                writer.writerow([r["split"], r["seed"], r["group"], r["pairs"], m, v["gap"]["mean"],
                    v["gap"]["quantiles"]["0.5"], v["gap"]["std"], v["ordered"], v["ties"], v["reversed"],
                    *[v["pair_counts"][k] for k in ("p_c", "p_v", "p_b", "p_r")], v["ordered_but_not_p_c"]])


def write_case_candidates(root, rows, records):
    """Deterministic short-function shortlist for subsequent source-level review."""
    source = {r["pair_key"]: rows[i:i+2] for i, r in enumerate(rows) if i % 2 == 0}
    by_key = {}
    for r in records:
        if r["split"] == "test":
            by_key.setdefault(r["pair_key"], []).append(r)
    groups = dict(
        improved=lambda rs: sum(r["gained_p_c"] for r in rs) >= 2 and not any(r["lost_p_c"] for r in rs),
        introduced_error=lambda rs: sum(r["lost_p_c"] for r in rs) >= 2 and not any(r["gained_p_c"] for r in rs),
        ordered_but_both_vulnerable=lambda rs: all(r["rgcl"]["ordered"] and r["rgcl"]["state"] == "p_v" for r in rs),
        no_improvement=lambda rs: all(r["context_only"]["state"] != "p_c" and r["rgcl"]["state"] != "p_c" for r in rs)
                                        and not rs[0]["identical_visible_input"],
        invisible_change=lambda rs: rs[0]["identical_visible_input"])
    shortlist = {}
    for name, select in groups.items():
        keys = [k for k, rs in by_key.items() if select(rs)]
        keys.sort(key=lambda k: (max(len(r["func"].splitlines()) for r in source[k]), k))
        shortlist[name] = keys[:6]
    write_json(root / "case_candidates.json", shortlist)
    # Exact source text stays in this local review file until selected examples are reviewed.
    with (root / "case_candidates.local.md").open("w", encoding="utf-8") as handle:
        for group, keys in shortlist.items():
            handle.write(f"# {group}\n\n")
            for key in keys:
                a, b = source[key]
                handle.write(f"## {key} | {a.get('project')} | {a.get('commit_id')}\n\n")
                for r in sorted(by_key[key], key=lambda x: x["seed"]):
                    handle.write(json.dumps(r, ensure_ascii=False)+"\n\n")
                handle.write("```diff\n"+"".join(difflib.unified_diff(a["func"].splitlines(True), b["func"].splitlines(True),
                    fromfile="vulnerable", tofile="repaired"))+"\n```\n\n")
                for label, r in (("vulnerable", a), ("repaired", b)):
                    handle.write(f"### {label} | Diff lines (zero-based): {r['lines_ground_truth']}\n\n```c\n{r['func']}\n```\n\n")


def write_report(root, result):
    """Tables use complete cohorts; semantic interpretations live in the reviewed casebook."""
    percent = lambda x: f"{100*x:.2f}%"
    mean_sd = lambda x, scale=1: f"{scale*x['mean']:.4f} ± {scale*x['std']:.4f}"
    lines = ["# RGCL 修复引导机制分析", "",
        "仅使用 PrimeVul + UniXcoder 已完成的 Seed42、1234、123456 Context-only/RGCL。"
        "K=5、λ=0.1、Token 加权区域损失；不重新训练、不调整阈值或选模。"
        "六个 checkpoint 均核对完整16轮训练记录及原完整 Valid P-C 选择。Test Diff 仅用于事后统计。", "",
        "## 分词、可见性和实际监督", "",
        "复用正式 `RobertaTokenizerFast(trim_offsets=False)` 与 `paired_features`："
        "原函数整体分词，不单独逐行分词或删除注释；不添加 UniXcoder 模式前缀。"
        "先保留前510个源码 BPE Token，再加 CLS/SEP，不足512补 Padding。"
        "空白 Token 消耗输入预算，但空白、特殊 Token 和 Padding 没有聚合质量；"
        "注释仍属于原始输入。物理行编号包含空行，底层从0计数。", "",
        "Diff 沿用 `_line_diff`/`difflib.Differ` 的双向非空删除行规则，已逐函数核对持久化标签。"
        "完整可见指所有原始直接修改行的映射 Token 均在前510内；部分可见指至少一个修改 Token 可见，"
        "但并非所有修改行都完整；不可见指原本有直接修改行而一个修改 Token 也不可见。"
        "原侧无直接 Diff 单列，纯新增修复的漏洞侧属于这一类，不能当成截断。"
        "下表百分比以该侧全部函数为分母；只以非空 Diff 为分母的比例另存 coverage.json。", "",
        "| 划分 | 侧 | 函数 | 完整 | 部分 | 不可见 | 无直接 Diff | 可施加区域损失 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split, c in result["coverage"].items():
        for side in ("all", "vulnerable", "repaired"):
            v = c[side]
            cells = [f"{v['category_counts'][k]} ({percent(v['category_fraction_all'][k])})" for k in VISIBILITY]
            lines.append(f"| {split} | {side} | {v['functions']} | " + " | ".join(cells) +
                         f" | {v['eligible_functions']} ({percent(v['eligible_functions']/v['functions'])}) |")
    lines += ["", "实际区域损失要求可见前景和背景均非空；全区域及空区域不生效，函数 BCE 保留。"
        "PrimeVul 沿用历史规则，截断末行的可见部分仍可参与区域监督。跨行 Token 按1/归属行数分配，"
        "质量与 bounded_pool 一致，可能为分数 Token。三个种子的全部 Train 区域掩码均与同一统计核对，"
        "因此覆盖率不随种子变化。Train 为实际可用监督；Valid/Test 仅表示事后可用性，未输入推理。", "",
        "| 划分 | 可见前景 Token质量 | 实际可用前景质量 | 对应背景质量 | 生效函数内前景均值/中位数/P95 | 含部分修改行的生效函数 | 全可见代码均为区域 |",
        "|---|---:|---:|---:|---|---:|---:|"]
    for split, c in result["coverage"].items():
        v, t = c["all"], c["all"]["token_sums"]
        d = v["eligible_region_tokens"]
        lines.append(f"| {split} | {t['visible_region_token_mass']:.3f} | {t['effective_region_token_mass']:.3f} | "
            f"{t['effective_background_token_mass']:.3f} | {d['mean']:.2f} / {d['quantiles']['0.5']:.2f} / "
            f"{d['quantiles']['0.95']:.2f} | {v['eligible_with_partial_diff_line']} | {v['all_visible_code_is_region']} |")
    lines += ["", "区域损失为 `max(0, log(2) - log(mean_region_rate) + log(mean_background_rate))`，"
        "对生效函数平均。它在漏洞侧和修复侧都提高修改区域相对背景的平均倍率，"
        "没有直接要求这些行具有正漏洞 Logit，也没有直接最大化配对概率间隔。", "",
        "## 预测间隔、排序与两侧均正确", "",
        "Δp=P(漏洞函数)-P(修复函数)。严格排序要求 Δp>0；相等概率单列，不能计为正确排序。"
        "P-C 要求漏洞概率>0.5且修复概率≤0.5。P-V=两侧均判漏洞，P-B=两侧均判正常，"
        "P-R=两侧均判反。排序正确的 P-V/P-B 仍不是 P-C。"
        "所有预测以唯一 sample_key 加完整 pair_key、标签、物理行和 Token 质量核对，"
        "Valid/Test 指标与原评价文件完全一致。Train 复用两份历史预测，另四份只补做冻结单函数推理。", "",
        "| 划分 | 种子 | 方法 | 对数 | Δp均值 | Δp中位数 | 正确排序 | 概率相等 | P-C | 排序正确但非P-C |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["summaries"]:
        if r["group"] != "all":
            continue
        for m, v in r["methods"].items():
            lines.append(f"| {r['split']} | {r['seed']} | {m} | {r['pairs']} | {v['gap']['mean']:.6f} | "
                f"{v['gap']['quantiles']['0.5']:.6f} | {v['ordered']} ({percent(v['ordering_rate'])}) | "
                f"{v['ties']} | {v['pair_counts']['p_c']} ({percent(v['p_c'])}) | {v['ordered_but_not_p_c']} |")
    lines += ["", "| 划分 | 种子 | 平均Δp变化 | 平均漏洞概率变化 | 平均修复概率变化 | 新增/丢失正确排序 | 新增/丢失P-C |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in result["summaries"]:
        if r["group"] == "all":
            lines.append(f"| {r['split']} | {r['seed']} | {r['gap_change']['mean']:+.6f} | "
                f"{r['vulnerable_change']['mean']:+.6f} | {r['repaired_change']['mean']:+.6f} | "
                f"{r['gained_ordering']}/{r['lost_ordering']} | {r['gained_p_c']}/{r['lost_p_c']} |")
    lines += ["", "三种子汇总为均值±样本标准差（ddof=1，n=3）。每个种子评估同一批函数，"
              "不能把重复的成员当作三倍独立样本。分布内部标准差为ddof=0。", "",
              "| 划分 | 子集 | 每种子对数 | Context Δp | RGCL Δp | Context排序% | RGCL排序% | Context P-C% | RGCL P-C% |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["aggregate"]:
        if r["group"] not in ("all", "visible_difference", "identical_input"):
            continue
        v = r["metrics"]
        vals = [mean_sd(v[f"{m}.{k}"], 1 if k == "gap" else 100)
                for k in ("gap", "ordering_rate", "p_c") for m in METHODS]
        lines.append(f"| {r['split']} | {r['group']} | {r['pairs_per_seed']} | " + " | ".join(vals) + " |")
    lines += ["", "相同可见输入按正式512 Token ID序列相等定义；所有官方成员保留。"
        "额外的纯新增修复、两侧区域均完整、任一侧区域不可见子集及所有 P-C/P-V/P-B/P-R 迁移见 analysis.json，"
        "逐对结果见 paired_gaps.jsonl。完整分母包含不可见配对和概率相等配对。", "",
        "## 实际发现", ""]
    aggregate = {(r["split"], r["group"]): r["metrics"] for r in result["aggregate"]}
    for split in ("train", "valid", "test"):
        v = aggregate[split, "all"]
        lines.append(f"{split.title()}：平均Δp {v['context_only.gap']['mean']:.6f} → {v['rgcl.gap']['mean']:.6f}；"
            f"P-C {percent(v['context_only.p_c']['mean'])} → {percent(v['rgcl.p_c']['mean'])}；"
            f"严格正确排序 {percent(v['context_only.ordering_rate']['mean'])} → {percent(v['rgcl.ordering_rate']['mean'])}。\n")
    test = aggregate["test", "all"]
    v = aggregate["test", "visible_difference"]
    test_summaries = sorted((r for r in result["summaries"] if r["split"] == "test" and r["group"] == "all"),
                            key=lambda r: r["seed"])
    changes = "、".join(f"{r['gained_p_c']}/{r['lost_p_c']}" for r in test_summaries)
    test_metrics = {m: {k: statistics.mean(e["predictions"]["test"]["metrics"][k]
                    for e in result["experiments"] if e["method"] == m)
                    for k in ("patched_fpr", "mcc")} for m in METHODS}
    test_n = result["coverage"]["test"]["pairs"]
    test_same = result["coverage"]["test"]["identical_visible_pairs"]
    train_v = aggregate["train", "visible_difference"]
    epochs = {m: "/".join(str(next(e["selected_epoch"] for e in result["experiments"]
                              if e["method"] == m and e["seed"] == s)) for s in SEEDS) for m in METHODS}
    lines += ["RGCL 的平均间隔在三个种子的 Valid/Test 均增加，但 Test 严格排序只在Seed42增加，"
        "Seed1234/123456下降；Valid 为两个增加、一个下降。平均间隔是幅度统计，正确排序是符号统计，"
        f"不能相互替代。Test P-C 两个种子提高、一个持平，同时新增/丢失P-C分别为{changes}。"
        "各模型的严格排序正确配对中，仍有大量 P-V/P-B；没有依据宣称模型普遍更容易排对前后版本。", "",
        f"仅看Test中有可见差异的{test_n-test_same}对，P-C {percent(v['context_only.p_c']['mean'])} → "
        f"{percent(v['rgcl.p_c']['mean'])}，排序 {percent(v['context_only.ordering_rate']['mean'])} → "
        f"{percent(v['rgcl.ordering_rate']['mean'])}，结论仍相同。相同可见输入的{test_same}对在六个模型中"
        "概率均相等，Δp和P-C均为0；这部分没有通过修改评价分母被剔除。", "",
        f"Test 的平均漏洞概率变化为 {test['rgcl.vulnerable_probability']['mean']-test['context_only.vulnerable_probability']['mean']:+.6f}，"
        f"修复概率变化为 {test['rgcl.repaired_probability']['mean']-test['context_only.repaired_probability']['mean']:+.6f}。"
        "因此平均间隔扩大并不等于总体修复侧概率下降。Seed42/123456 的两侧平均概率都升高，"
        f"只有Seed1234修复侧均值下降。三种子Test的修复侧FPR从{percent(test_metrics['context_only']['patched_fpr'])}"
        f"升至{percent(test_metrics['rgcl']['patched_fpr'])}，MCC从{test_metrics['context_only']['mcc']:.6f}"
        f"略降至{test_metrics['rgcl']['mcc']:.6f}；P-C收益不代表整体误报下降。", "",
        f"Train中有可见差异配对的RGCL正确排序为{percent(train_v['rgcl.ordering_rate']['mean'])}、"
        f"P-C为{percent(train_v['rgcl.p_c']['mean'])}，Test对应为{percent(v['rgcl.ordering_rate']['mean'])}、"
        f"{percent(v['rgcl.p_c']['mean'])}。训练集的分离能力没有同幅度迁移到未见样本。"
        f"已有最佳轮次分别为Context-only {epochs['context_only']}、RGCL {epochs['rgcl']}"
        "（种子42/1234/123456）；Seed1234的Context-only概率尤其集中在0.5附近。"
        "不能把所选模型的全部差异解释成同一个训练时刻增加Diff损失的直接作用。", "",
        "## 分布图", "",
        "![逐种子Train/Valid/Test概率间隔经验CDF；不分箱、不平滑，横轴统一为负1至1](probability_gap_ecdf.png)", "",
        "[可编辑SVG](probability_gap_ecdf.svg)。每条曲线包含该划分全部配对；0处跳跃包含概率相等的配对。", "",
        "![逐对Context-only与RGCL间隔；对角线上方表示间隔增加，橙色叉为相同可见输入](paired_gap_changes.png)", "",
        "[可编辑SVG](paired_gap_changes.svg)。每个点对应一对函数，无抖动或样本删除；重叠点不会展示为独立密度。", "",
        "## 案例与解释", "",
        "原始源码、真实Diff、逐种子概率及代码语义核对见 [案例分析](cases.md)。"
        "候选按跨种子新增P-C、丢失P-C、排序正确但两侧判漏洞、持续失败、相同可见输入分组，"
        "各组优先短函数，完整规则见 write_case_candidates。案例用于解释已有结果，不用于选模型或阈值。", "",
        "概率变化只能说明函数输出变化，不能由此推断模型关注了哪些行。"
        "独立训练并按Valid选择的模型可能选中不同Epoch；这些事后比较描述与Diff监督相关的现象，"
        "不能独立证明具体代码行的注意力或因果中介作用。", "",
        "复现统计和图：`python -m vulnmechanism.cli rgcl-mechanism --device cuda:0`。"
        "缺失Train预测保存在新分析目录的本地cache，不改写任何历史目录、checkpoint或数据。"
        "26项相关最小测试记录见 tests.log；首次检查发现并修复了空侧分母的统计异常，未放宽断言。"
        "checks.json 另记录18组预测、65,760条函数记录、各子集统计恒等式、所选Epoch及12段案例原文的一致性核对。", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
