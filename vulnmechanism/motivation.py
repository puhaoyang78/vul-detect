"""Four-dataset diagnostics using ordinary BCE and shallow Shortcut Features."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import random

import joblib
import numpy as np
import sklearn
from scipy.stats import ks_2samp, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .artifacts import write_json
from .data import read_jsonl, test_rows, validation_rows, write_jsonl
from .metrics import metrics, select_threshold
from .syntax import parser_for, resolve_language, source_token_spans, walk, text

LR_C_VALUES = (.01, .1, 1., 10., 100.)
TFIDF_CONFIG = dict(
    ngram_range=(1, 2), min_df=2, max_features=100_000,
    lowercase=False, token_pattern=None, sublinear_tf=True,
    norm="l2", smooth_idf=True,
)
SCORE_BINS = 50
JOINT_MIN_SAMPLES = 200

FEATURES = (
    "total_lines", "code_lines", "codebert_tokens", "characters",
    "identifiers", "unique_identifiers", "mean_identifier_length",
    "literals", "string_literals", "numeric_literals", "comments",
    "comment_lines", "comment_fraction", "branches", "loops",
    "max_control_depth", "parse_errors",
)
CONTROL = {
    "if_statement", "switch_statement", "for_statement", "for_range_loop",
    "while_statement", "do_statement",
}


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"{path}: empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def shortcut_features(row, tokenizer):
    source = row["raw_source"]
    raw = source.encode()
    language = resolve_language(source, row["language"], row["file_name"])
    root = parser_for(language).parse(raw).root_node
    nodes = list(walk(root))
    counts = Counter(n.type for n in nodes)
    identifier_types = {
        "identifier", "field_identifier", "type_identifier", "namespace_identifier",
        "statement_identifier",
    }
    names = [text(n, raw) for n in nodes if n.type in identifier_types]
    comments = [n for n in nodes if n.type == "comment"]
    code = bytearray(raw)
    comment_lines = set()
    for n in comments:
        for position in range(n.start_byte, n.end_byte):
            if code[position] not in (10, 13):
                code[position] = 32
        end = n.end_point.row - (n.end_point.column == 0)
        comment_lines.update(range(n.start_point.row, end + 1))
    depth = 0
    stack = [(root, 0)]
    while stack:
        node, current = stack.pop()
        current += node.type in CONTROL
        depth = max(depth, current)
        stack.extend((child, current) for child in node.children)
    return dict(
        total_lines=len(source.splitlines()),
        code_lines=sum(bool(line.strip()) for line in code.splitlines()),
        codebert_tokens=len(tokenizer(
            source, add_special_tokens=False, truncation=False, verbose=False
        )["input_ids"]),
        characters=len(source), identifiers=len(names), unique_identifiers=len(set(names)),
        mean_identifier_length=sum(map(len, names)) / len(names) if names else 0.0,
        literals=sum(counts[t] for t in (
            "number_literal", "string_literal", "char_literal", "true", "false", "null", "nullptr"
        )),
        string_literals=counts["string_literal"], numeric_literals=counts["number_literal"],
        comments=len(comments), comment_lines=len(comment_lines),
        comment_fraction=sum(n.end_byte - n.start_byte for n in comments) / len(raw),
        branches=sum(counts[t] for t in (
            "if_statement", "switch_statement", "case_statement"
        )),
        loops=sum(counts[t] for t in (
            "for_statement", "for_range_loop", "while_statement", "do_statement"
        )),
        max_control_depth=depth,
        parse_errors=sum(n.is_error or n.is_missing for n in nodes),
    )


def match_lengths(rows, seed=42):
    """The established protocol: exact line counts, then labels 0/1, source order."""
    groups = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        groups[len(row["raw_source"].splitlines())][row["label"]].append(row)
    rng = random.Random(seed)
    selected = set()
    for length in sorted(groups):
        n = min(len(groups[length][0]), len(groups[length][1]))
        for label in (0, 1):
            selected.update(r["sample_key"] for r in rng.sample(groups[length][label], n))
    return [r for r in rows if r["sample_key"] in selected]


def length_members(rows, path):
    matched = match_lengths(rows)
    if not matched:
        raise ValueError("no exact line-count matching support")
    path = Path(path)
    if path.exists():
        with path.open() as handle:
            saved = list(csv.DictReader(handle))
        expected = [(r["sample_key"], r["label"]) for r in matched]
        if [(r["sample_key"], int(r["label"])) for r in saved] != expected:
            raise ValueError(f"{path}: established length-matched members changed")
    else:
        write_table(path, [dict(
            sample_key=r["sample_key"], label=r["label"], project=r["project"],
            function_lines=len(r["raw_source"].splitlines()),
            source_file=r["source_file"], source_row=r["source_row"],
        ) for r in matched])
    return matched


def feature_experiment(data_dir, output_dir, dataset, model_path):
    from transformers import AutoTokenizer

    root = Path(output_dir) / "motivation"
    root.mkdir(parents=True, exist_ok=True)
    data = Path(data_dir) / dataset
    length_members(test_rows(data), Path(output_dir) / "length_matched" / f"{dataset}.members.csv")
    rows = read_jsonl(data / "train.jsonl") + validation_rows(data) + test_rows(data)
    path = root / f"{dataset}.features.csv"
    if path.exists():
        with path.open() as handle:
            table = list(csv.DictReader(handle))
        expected = [(r["sample_key"], r["split"], r["label"]) for r in rows]
        if [(r["sample_key"], r["split"], int(r["label"])) for r in table] != expected:
            raise ValueError(f"{path}: cached feature members differ")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        table = []
        for index, row in enumerate(rows, 1):
            table.append(dict(
                sample_key=row["sample_key"], split=row["split"], label=row["label"],
                project=row["project"], **shortcut_features(row, tokenizer),
            ))
            if index % 500 == 0:
                print(f"{dataset}: Shortcut Features {index}/{len(rows)}", flush=True)
        write_table(path, table)
    subsets = {s: [r for r in table if r["split"] == s] for s in ("train", "valid", "test")}
    arrays = {
        s: np.array([[float(r[k]) for k in FEATURES] for r in values])
        for s, values in subsets.items()
    }
    labels = {s: [int(r["label"]) for r in values] for s, values in subsets.items()}
    statistics = []
    for index, feature in enumerate(FEATURES):
        auc = float(roc_auc_score(labels["test"], arrays["test"][:, index]))
        for label in (0, 1):
            values = arrays["test"][np.array(labels["test"]) == label, index]
            statistics.append(dict(
                dataset=dataset, feature=feature, label=label, samples=len(values),
                minimum=float(values.min()), q25=float(np.quantile(values, .25)),
                median=float(np.median(values)), q75=float(np.quantile(values, .75)),
                maximum=float(values.max()), mean=float(values.mean()),
                zero_fraction=float(np.mean(values == 0)), single_feature_auc=auc,
            ))
    write_table(root / f"{dataset}.feature_statistics.csv", statistics)
    candidates, best, best_key = [], None, None
    for c in LR_C_VALUES:
        pipeline = make_pipeline(
            FunctionTransformer(np.log1p), StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, random_state=42),
        )
        pipeline.fit(arrays["train"], labels["train"])
        probabilities = pipeline.predict_proba(arrays["valid"])[:, 1].tolist()
        auc = float(roc_auc_score(labels["valid"], probabilities))
        candidates.append(dict(
            C=c, valid_auc=auc, iterations=int(pipeline[-1].n_iter_.max())
        ))
        if best_key is None or (auc, -c) > best_key:
            best_key, best = (auc, -c), (c, pipeline, probabilities)
    c, pipeline, probabilities = best
    threshold, valid_metrics = select_threshold(labels["valid"], probabilities)
    test_scores = pipeline.predict_proba(arrays["test"])[:, 1].tolist()
    result = dict(
        dataset=dataset, features=list(FEATURES), seed=42, selected_C=c,
        preprocessing="log1p, then Train-fitted StandardScaler; all counts nonnegative",
        selection=(
            "Train-only fit; Valid AUC selects C, smaller C breaks ties; "
            "Valid MCC/F1/Accuracy selects threshold"
        ),
        candidates=candidates, valid=valid_metrics,
        test=metrics(labels["test"], test_scores, threshold),
        members={s: len(r) for s, r in subsets.items()},
        feature_definition=(
            "Tree-sitter C/C++ observed counts; errors retained and counted, "
            "not treated as compiler-verified structure. Code lines exclude "
            "blank/comment-only lines. Token counts use full untruncated CodeBERT tokenizer."
        ),
        standardized_log1p_coefficients=dict(zip(FEATURES, pipeline[-1].coef_[0].tolist())),
    )
    write_json(root / f"{dataset}.logistic.metrics.json", result)
    write_jsonl(
        root / f"{dataset}.logistic.test.predictions.jsonl",
        (dict(sample_key=r["sample_key"], label=int(r["label"]), score=p, threshold=threshold)
         for r, p in zip(subsets["test"], test_scores)),
    )
    joblib.dump(pipeline, root / f"{dataset}.logistic.joblib")
    print(
        f"{dataset}: Logistic Regression Test AUC={result['test']['auc']:.6f}, C={c}",
        flush=True,
    )
    return result


def code_tokens(source):
    """Reuse C/C++ lexical tokens; preserve spelling, literals and operators."""
    return [token for token, _, _ in source_token_spans(source)]


def match_tfidf(rows, scores, *, joint=False, seed=42):
    """Balance labels within fixed score bins, optionally crossed with exact lines."""
    if len(rows) != len(scores) or any(
        not np.isfinite(s) or not 0 <= s <= 1 for s in scores
    ):
        raise ValueError("one finite TF-IDF probability per sample required")
    groups = defaultdict(lambda: {0: [], 1: []})
    for row, score in zip(rows, scores):
        length = len(row["raw_source"].splitlines()) if joint else 0
        score_bin = min(int(score * SCORE_BINS), SCORE_BINS - 1)
        groups[length, score_bin][row["label"]].append(row["sample_key"])
    rng = random.Random(seed)
    selected = set()
    for key in sorted(groups):
        n = min(len(groups[key][0]), len(groups[key][1]))
        for label in (0, 1):
            selected.update(rng.sample(groups[key][label], n))
    return [r for r in rows if r["sample_key"] in selected]


def tfidf_experiment(data_dir, output_dir, dataset):
    """Fit the lexical baseline and compare saved BCE on prespecified Test subsets."""
    root = Path(output_dir) / "motivation"
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        seed=42, tokenizer="syntax.source_token_spans; comments/whitespace omitted; spelling retained",
        vectorizer=TFIDF_CONFIG,
        classifier=dict(solver="liblinear", C_values=LR_C_VALUES, max_iter=2000, tol=1e-4),
        selection=(
            "Train-only vocabulary, IDF and LR; Valid AUC selects C (smaller C ties); "
            "Valid selects threshold"
        ),
        matching=dict(
            bins=SCORE_BINS, edges=[i / SCORE_BINS for i in range(SCORE_BINS + 1)],
            intervals="[left, right), except final bin includes 1; index=min(int(score*50),49)",
            sampling=(
                "sorted strata, labels 0 then 1; random.Random(42), min(class counts), "
                "without replacement; original source order"
            ),
            joint="exact integer total lines crossed with score bin",
            joint_min_samples=JOINT_MIN_SAMPLES,
            selection_inputs="labels, total lines and TF-IDF probabilities only; no BCE predictions",
        ),
        interpretation=(
            "Control observed Test shortcut association; does not identify internal "
            "model reliance or causal contributions"
        ),
        environment=dict(sklearn=sklearn.__version__, numpy=np.__version__),
    )
    write_json(root / "tfidf.protocol.json", protocol)
    data = Path(data_dir) / dataset
    rows = {
        "train": read_jsonl(data / "train.jsonl"),
        "valid": validation_rows(data), "test": test_rows(data),
    }
    labels = {s: [r["label"] for r in values] for s, values in rows.items()}
    vectorizer = TfidfVectorizer(tokenizer=code_tokens, **TFIDF_CONFIG)
    print(f"{dataset}: fit Train-only TF-IDF on {len(rows['train'])} functions", flush=True)
    train_features = vectorizer.fit_transform(r["raw_source"] for r in rows["train"])
    valid_features = vectorizer.transform(r["raw_source"] for r in rows["valid"])
    source_token_spans.cache_clear()
    candidates, best, best_key = [], None, None
    for c in LR_C_VALUES:
        classifier = LogisticRegression(
            C=c, solver="liblinear", max_iter=2000, tol=1e-4, random_state=42,
        )
        classifier.fit(train_features, labels["train"])
        probabilities = classifier.predict_proba(valid_features)[:, 1].tolist()
        auc = float(roc_auc_score(labels["valid"], probabilities))
        candidates.append(dict(C=c, valid_auc=auc, iterations=int(classifier.n_iter_.max())))
        print(f"{dataset}: TF-IDF C={c:g}, Valid AUC={auc:.6f}", flush=True)
        if best_key is None or (auc, -c) > best_key:
            best_key, best = (auc, -c), (c, classifier, probabilities)
    c, classifier, probabilities = best
    threshold, valid_metrics = select_threshold(labels["valid"], probabilities)
    test_features = vectorizer.transform(r["raw_source"] for r in rows["test"])
    source_token_spans.cache_clear()
    scores = classifier.predict_proba(test_features)[:, 1].tolist()
    for split, values in (("valid", probabilities), ("test", scores)):
        write_jsonl(
            root / f"{dataset}.tfidf.{split}.predictions.jsonl",
            (dict(sample_key=r["sample_key"], label=r["label"], score=s, threshold=threshold)
             for r, s in zip(rows[split], values)),
        )
    joblib.dump(make_pipeline(vectorizer, classifier), root / f"{dataset}.tfidf.joblib")
    test = rows["test"]
    score_by_key = dict(zip((r["sample_key"] for r in test), scores))
    subsets = dict(
        original=test,
        length_matched=length_members(
            test, Path(output_dir) / "length_matched" / f"{dataset}.members.csv"
        ),
        tfidf_matched=match_tfidf(test, scores),
        joint_matched=match_tfidf(test, scores, joint=True),
    )
    # Save the selected members before reading any CodeBERT predictions.
    for subset in ("tfidf_matched", "joint_matched"):
        members = subsets[subset]
        if not members or (subset == "joint_matched" and len(members) < JOINT_MIN_SAMPLES):
            continue
        write_table(Path(output_dir) / subset / f"{dataset}.members.csv", [dict(
            sample_key=r["sample_key"], label=r["label"], project=r["project"],
            function_lines=len(r["raw_source"].splitlines()),
            tfidf_score=score_by_key[r["sample_key"]],
            score_bin=min(int(score_by_key[r["sample_key"]] * SCORE_BINS), SCORE_BINS - 1),
            source_file=r["source_file"], source_row=r["source_row"],
        ) for r in members])
    run = Path(output_dir) / dataset / "bce"
    predictions = read_jsonl(run / "test.predictions.jsonl")
    if [(r["sample_key"], r["label"]) for r in predictions] != [
        (r["sample_key"], r["label"]) for r in test
    ]:
        raise ValueError(f"{dataset}: BCE prediction members differ")
    original = json.loads((run / "test.metrics.json").read_text())["metrics"]
    recomputed = metrics(
        labels["test"], [r["score"] for r in predictions], original["threshold"]
    )
    if recomputed != original:
        raise ValueError(f"{dataset}: saved BCE metrics differ")
    bce = {r["sample_key"]: r["score"] for r in predictions}
    comparisons = []
    for subset, members in subsets.items():
        y = [r["label"] for r in members]
        skipped = not members or (
            subset == "joint_matched" and len(members) < JOINT_MIN_SAMPLES
        )
        value = dict(
            dataset=dataset, subset=subset,
            status="skipped_insufficient_samples" if skipped else "evaluated",
            samples=len(members), positive=sum(y), negative=len(y) - sum(y),
            positive_fraction=sum(y) / len(y) if y else None,
            length_auc=None, tfidf_auc=None, bce_auc=None, tfidf_score_ks=None,
        )
        if not skipped:
            lexical_scores = [score_by_key[r["sample_key"]] for r in members]
            value.update(
                length_auc=float(roc_auc_score(
                    y, [len(r["raw_source"].splitlines()) for r in members]
                )),
                tfidf_auc=float(roc_auc_score(y, lexical_scores)),
                bce_auc=float(roc_auc_score(y, [bce[r["sample_key"]] for r in members])),
                tfidf_score_ks=float(ks_2samp(
                    [s for s, label in zip(lexical_scores, y) if label == 0],
                    [s for s, label in zip(lexical_scores, y) if label == 1], method="asymp",
                ).statistic),
            )
        comparisons.append(value)
    result = dict(
        dataset=dataset, seed=42, protocol=protocol,
        members={s: len(values) for s, values in rows.items()},
        vocabulary=len(vectorizer.vocabulary_), selected_C=c, candidates=candidates,
        valid=valid_metrics, test=metrics(labels["test"], scores, threshold),
        comparisons=comparisons,
    )
    write_json(root / f"{dataset}.tfidf.metrics.json", result)
    print(f"{dataset}: TF-IDF Test AUC={result['test']['auc']:.6f}, C={c:g}", flush=True)
    return result


def tfidf_report(output_dir, datasets):
    root = Path(output_dir) / "motivation"
    experiments = {
        d: json.loads((root / f"{d}.tfidf.metrics.json").read_text()) for d in datasets
    }
    classifiers, comparisons = [], []
    for dataset, result in experiments.items():
        original = result["comparisons"][0]
        classifiers.append(dict(
            dataset=dataset, selected_C=result["selected_C"], vocabulary=result["vocabulary"],
            **{k: result["test"][k] for k in (
                "samples", "threshold", "accuracy", "f1", "mcc", "auc"
            )},
            length_auc=original["length_auc"], bce_auc=original["bce_auc"],
        ))
        comparisons.extend(result["comparisons"])
    write_table(root / "tfidf.csv", classifiers)
    write_table(root / "shortcut_control.csv", comparisons)
    write_json(root / "tfidf.results.json", dict(experiments=experiments))
    return dict(datasets=len(experiments), classifiers=len(classifiers), subsets=len(comparisons))


def motivation_report(data_dir, output_dir, datasets):
    root = Path(output_dir) / "motivation"
    clean, robustness, length_results, logistic, statistics = [], [], [], [], []
    for dataset in datasets:
        run = Path(output_dir) / dataset / "bce"
        rows = test_rows(Path(data_dir) / dataset)
        matched = length_members(
            rows, Path(output_dir) / "length_matched" / f"{dataset}.members.csv"
        )
        predictions = read_jsonl(run / "test.predictions.jsonl")
        if [(r["sample_key"], r["label"]) for r in predictions] != [
            (r["sample_key"], r["label"]) for r in rows
        ]:
            raise ValueError(f"{dataset}: Clean prediction members differ")
        original = json.loads((run / "test.metrics.json").read_text())["metrics"]
        recomputed = metrics(
            [r["label"] for r in predictions], [r["score"] for r in predictions],
            original["threshold"],
        )
        if recomputed != original:
            raise ValueError(f"{dataset}: Clean prediction metrics differ")
        clean.append(dict(dataset=dataset, **{k: original[k] for k in (
            "samples", "threshold", "accuracy", "f1", "mcc", "auc", "auprc",
            "tp", "tn", "fp", "fn",
        )}))
        report = json.loads((run / "robustness.metrics.json").read_text())
        if report["clean"]["metrics"] != original:
            raise ValueError(f"{dataset}: robustness Clean cohort differs")
        for setting in ("fixed", "heldout", "adaptive"):
            result = report[setting if setting == "adaptive" else setting + "_overall"]
            for kind in ("strict_semantic", "input_only_stress"):
                value = result[kind]
                worst = value["worst_complete"]
                assert worst["samples"] == len(rows)
                assert worst["tp"] + worst["tn"] == (
                    original["tp"] + original["tn"] - value["asr"]["count"]
                )
                robustness.append(dict(
                    dataset=dataset, setting=setting, perturbation=kind, samples=len(rows),
                    clean_accuracy=original["accuracy"], clean_mcc=original["mcc"],
                    worst_accuracy=worst["accuracy"], worst_mcc=worst["mcc"],
                    asr=value["asr"]["rate"], asr_count=value["asr"]["count"],
                    asr_denominator=value["asr"]["denominator"],
                    eligible=value["eligible"], coverage=value["coverage"],
                ))
        by_key = {r["sample_key"]: r for r in predictions}
        labels = [r["label"] for r in matched]
        histograms = [
            Counter(len(r["raw_source"].splitlines()) for r in matched if r["label"] == y)
            for y in (0, 1)
        ]
        assert histograms[0] == histograms[1]
        for model in ("function_lines", "bce"):
            if model == "function_lines":
                full_scores = [len(r["raw_source"].splitlines()) for r in rows]
                scores = [len(r["raw_source"].splitlines()) for r in matched]
            else:
                full_scores = [r["score"] for r in predictions]
                scores = [by_key[r["sample_key"]]["score"] for r in matched]
            value = float(roc_auc_score(labels, scores))
            if model == "function_lines":
                assert abs(value - .5) < 1e-15
            length_results.append(dict(
                dataset=dataset, model=model, original_samples=len(rows),
                matched_samples=len(matched),
                matched_positive=sum(labels), matched_negative=len(labels) - sum(labels),
                original_auc=float(roc_auc_score([r["label"] for r in rows], full_scores)),
                length_matched_auc=value,
            ))
        shallow = json.loads((root / f"{dataset}.logistic.metrics.json").read_text())
        logistic.append(dict(
            dataset=dataset, selected_C=shallow["selected_C"],
            **{k: shallow["test"][k] for k in (
                "samples", "threshold", "accuracy", "f1", "mcc", "auc", "auprc"
            )},
        ))
        with (root / f"{dataset}.feature_statistics.csv").open() as handle:
            statistics.extend(csv.DictReader(handle))
    motivation2 = []
    for value in clean:
        dataset = value["dataset"]
        feature_auc = {
            r["feature"]: float(r["single_feature_auc"])
            for r in statistics if r["dataset"] == dataset
        }
        matched = next(
            r for r in length_results if r["dataset"] == dataset and r["model"] == "bce"
        )
        shallow = next(r for r in logistic if r["dataset"] == dataset)
        motivation2.append(dict(
            dataset=dataset, samples=value["samples"], bce_auc=value["auc"],
            total_lines_auc=feature_auc["total_lines"],
            code_lines_auc=feature_auc["code_lines"],
            codebert_tokens_auc=feature_auc["codebert_tokens"], logistic_auc=shallow["auc"],
            matched_samples=matched["matched_samples"],
            matched_positive=matched["matched_positive"],
            matched_negative=matched["matched_negative"], matched_line_auc=.5,
            matched_bce_auc=matched["length_matched_auc"],
        ))
    for name, values in (
        ("clean", clean), ("motivation1", robustness), ("motivation2", motivation2),
        ("length_matched", length_results), ("logistic", logistic),
        ("shortcut_features", statistics),
    ):
        write_table(root / (name + ".csv"), values)
    summary = dict(
        datasets={
            d: json.loads((Path(data_dir) / d / "statistics.json").read_text())
            for d in datasets
        },
        training={
            d: json.loads((Path(output_dir) / d / "bce" / "training.metrics.json").read_text())
            for d in datasets
        },
        clean=clean, motivation1=robustness, motivation2=motivation2,
        length_matched=length_results, logistic=logistic,
        protocol=dict(
            seed=42, models="Ordinary CodeBERT BCE only; reuse existing PrimeVul/MegaVul outputs",
            length_matching=(
                "Exact integer total-line strata; min(class counts), random.Random(42); "
                "selection ignores model scores"
            ),
            asr_denominator=(
                "Clean-correct samples with at least one eligible perturbation "
                "in the indicated category"
            ),
            interpretation=(
                "Shortcut Features measure statistical association, not causal "
                "irrelevance or model causal dependence; matching changes the "
                "evaluation population."
            ),
        ),
    )
    write_json(root / "results.json", summary)
    return dict(
        datasets=len(datasets), clean_rows=len(clean), robustness_rows=len(robustness),
        length_rows=len(length_results), output=str(root),
    )


def _input_conflicts(rows, encoded):
    """Count indistinguishable opposite-label inputs without removing any rows."""
    groups = defaultdict(list)
    for row in rows:
        groups[encoded[row["sample_key"]]].append(row)
    mixed = [g for g in groups.values() if {r["label"] for r in g} == {0, 1}]
    unavoidable_errors = sum(min(Counter(r["label"] for r in g).values()) for g in mixed)
    return dict(
        groups=len(mixed), samples=sum(map(len, mixed)),
        positive=sum(r["label"] for g in mixed for r in g),
        minimum_errors=unavoidable_errors,
        maximum_accuracy=1 - unavoidable_errors / len(rows),
    )


def transfer_train_probe(data_dir, output_dir, model_path, device="auto"):
    """Infer a fixed 512-per-class source-Train sample using the saved BCE only."""
    import torch
    from .model import InputBuilder, AutoTokenizer, load_model, predict_logits, resolve_device, seed_everything

    seed_everything(42)
    device = resolve_device(device)
    builder = InputBuilder(AutoTokenizer.from_pretrained(model_path))
    output_dir = Path(output_dir)
    root = output_dir / "transfer_primevul_secvuleval"
    summary, predictions = [], []
    for dataset in ("primevul", "secvuleval"):
        rows = read_jsonl(Path(data_dir) / dataset / "train.jsonl")
        rng, keys = random.Random(42), set()
        for label in (0, 1):
            keys.update(r["sample_key"] for r in rng.sample([r for r in rows if r["label"] == label], 512))
        rows = [r for r in rows if r["sample_key"] in keys]
        checkpoint_path = output_dir / dataset / "bce/best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["dataset"] != dataset or checkpoint["method"] != "bce":
            raise ValueError("Train diagnostic checkpoint differs")
        model = load_model(checkpoint, device, model_path=model_path)
        logits = predict_logits(model, rows, builder, device=device, batch_size=4)
        scores = logits.sigmoid().tolist()
        threshold = checkpoint["decision_threshold"]
        summary.append(dict(
            dataset=dataset, checkpoint=str(checkpoint_path), selected_epoch=checkpoint["selected_epoch"],
            **metrics([r["label"] for r in rows], scores, threshold),
            bce=float(torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([r["label"] for r in rows], dtype=logits.dtype))),
        ))
        predictions.extend(dict(
            dataset=dataset, sample_key=r["sample_key"], label=r["label"], project=r["project"],
            source_file=r["source_file"], source_row=r["source_row"],
            logit=float(logit), score=score, threshold=threshold,
        ) for r, logit, score in zip(rows, logits.tolist(), scores))
        del model, checkpoint
        print(f"{dataset}: Train probe AUC={summary[-1]['auc']:.6f}, BCE={summary[-1]['bce']:.6f}", flush=True)
    write_jsonl(root / "diagnosis.train.predictions.jsonl", predictions)
    write_json(root / "diagnosis.train.metrics.json", dict(
        protocol="Source Train only; seed 42; sample 512 negatives then 512 positives without replacement; retain source order; frozen existing BCE, source Valid epoch/threshold unchanged. This is an in-sample learning diagnostic, not generalization performance.",
        results=summary,
    ))
    return summary


def _project_auc_interval(rows, scores, repetitions=1000):
    """Project-cluster bootstrap, keeping paired functions in the same draw."""
    projects = sorted({r["project_url"] or r["project"] for r in rows})
    index = {p: i for i, p in enumerate(projects)}
    owners = np.array([index[r["project_url"] or r["project"]] for r in rows])
    labels = np.array([r["label"] for r in rows])
    rng, values = np.random.default_rng(42), []
    for _ in range(repetitions):
        weights = np.bincount(rng.integers(len(projects), size=len(projects)), minlength=len(projects))[owners]
        if not all(weights[labels == y].sum() for y in (0, 1)):
            continue
        values.append(roc_auc_score(labels, scores, sample_weight=weights))
    if not values:
        raise ValueError("project bootstrap has no two-class replicates")
    return dict(
        low=float(np.percentile(values, 2.5)), high=float(np.percentile(values, 97.5)),
        repetitions=repetitions, usable_repetitions=len(values), projects=len(projects),
    )


def _transfer_native_metadata(raw_root, selected):
    """Audit native labels/code and recover only metadata needed for diagnosis."""
    import pyarrow.parquet as pq

    def project_url(value):
        return (value or "").strip().rstrip("/").removesuffix(".git").casefold()

    datasets = ("primevul", "secvuleval")
    pairs = {d: [] for d in datasets}
    sec = pq.read_table(raw_root / "SecVulEval/data/train-00000-of-00001.parquet").to_pylist()
    sec_by_id = {r["idx"]: r for r in sec}
    sec_by_source = defaultdict(list)
    references = set()
    for r in sec:
        sec_by_source[r["func_body"]].append(r)
        fixed = r["fixed_func_idx"]
        if r["is_vulnerable"] and fixed is not None and math.isfinite(fixed):
            other = sec_by_id[int(fixed)]
            if other["is_vulnerable"] or project_url(r["project_url"]) != project_url(other["project_url"]):
                raise ValueError("invalid SecVulEval native repair link")
            references.add(other["idx"])
            pairs["secvuleval"].append((f"secvuleval:{r['idx']}", f"secvuleval:{other['idx']}"))
    for r in sec:
        key = f"secvuleval:{r['idx']}"
        if key in selected["secvuleval"]:
            row = selected["secvuleval"][key]
            if row["label"] != int(r["is_vulnerable"]) or row["raw_source"] != r["func_body"]:
                raise ValueError(f"{key}: prepared code/label differs from native data")
            row.update(
                project_url=project_url(r["project_url"]),
                known_fixed_negative=r["idx"] in references,
            )

    prime_fixed, overlaps = set(), defaultdict(list)
    for split in ("train", "valid", "test"):
        paired = read_jsonl(raw_root / f"PrimeVul_v0.1/primevul_{split}_paired.jsonl")
        if len(paired) % 2:
            raise ValueError("odd number of PrimeVul paired records")
        for a, b in zip(paired[::2], paired[1::2]):
            if (a["target"], b["target"]) != (1, 0):
                raise ValueError("PrimeVul paired order is not vulnerable/fixed")
            pairs["primevul"].append((f"primevul:{a['idx']}", f"primevul:{b['idx']}"))
            prime_fixed.add(b["idx"])
        path = raw_root / f"PrimeVul_v0.1/primevul_{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                key = f"primevul:{r['idx']}"
                if r["func"] in sec_by_source:
                    overlaps[r["func"]].append((key, int(r["target"])))
                if key in selected["primevul"]:
                    row = selected["primevul"][key]
                    if row["label"] != r["target"] or row["raw_source"] != r["func"]:
                        raise ValueError(f"{key}: prepared code/label differs from native data")
                    row.update(
                        project_url=project_url(r.get("project_url")),
                        known_fixed_negative=r["idx"] in prime_fixed,
                    )
    for rows in selected.values():
        if any("project_url" not in r for r in rows.values()):
            raise ValueError("native metadata did not cover all prepared members")

    conflicts, matrix, selected_matrix = [], Counter(), Counter()
    for source, prime in overlaps.items():
        sec_rows = sec_by_source[source]
        prime_labels = tuple(sorted({y for _, y in prime}))
        sec_labels = tuple(sorted({int(r["is_vulnerable"]) for r in sec_rows}))
        matrix[str(prime_labels) + ":" + str(sec_labels)] += 1
        pp = [(k, y) for k, y in prime if k in selected["primevul"]]
        ss = [(f"secvuleval:{r['idx']}", int(r["is_vulnerable"])) for r in sec_rows
              if f"secvuleval:{r['idx']}" in selected["secvuleval"]]
        if pp and ss:
            selected_matrix[str(tuple(sorted({y for _, y in pp}))) + ":" +
                            str(tuple(sorted({y for _, y in ss})))] += 1
        if prime_labels != sec_labels:
            conflicts.append(dict(
                primevul_keys=";".join(k for k, _ in prime), primevul_labels=str(prime_labels),
                secvuleval_keys=";".join(f"secvuleval:{r['idx']}" for r in sec_rows),
                secvuleval_labels=str(sec_labels), in_both_selected=bool(pp and ss),
            ))
    return pairs, dict(
        native_overlapping_sources=len(overlaps), native_label_sets=dict(matrix),
        selected_overlapping_sources=sum(selected_matrix.values()), selected_label_sets=dict(selected_matrix),
        native_disagreeing_sources=len(conflicts),
        interpretation="Different native labels for identical code are annotation/context disagreements, not automatically verified label errors.",
    ), conflicts


def transfer_diagnostics(data_dir, output_dir, model_path):
    """Diagnose the established PrimeVul/SecVulEval transfer; never train or reselect."""
    from transformers import AutoTokenizer
    from .data import class_counts
    from .model import InputBuilder

    datasets = ("primevul", "secvuleval")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    root = output_dir / "transfer_primevul_secvuleval"
    formal = json.loads((root / "results.json").read_text())
    rows = {d: dict(
        train=read_jsonl(data_dir / d / "train.jsonl"), valid=validation_rows(data_dir / d),
        test=test_rows(data_dir / d),
    ) for d in datasets}
    selected = {d: {r["sample_key"]: r for values in rows[d].values() for r in values} for d in datasets}
    raw_roots = {json.loads((data_dir / d / "statistics.json").read_text())["raw_root"] for d in datasets}
    if len(raw_roots) != 1:
        raise ValueError("native data roots differ")
    print("Transfer diagnosis: verify native code, labels and repair links", flush=True)
    pairs, labels, conflicts = _transfer_native_metadata(Path(raw_roots.pop()), selected)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    builder = InputBuilder(tokenizer)
    budget = builder.source_max_length - len(builder.source_prefix) - 2
    encoded, features = {}, {}
    for d in datasets:
        with (output_dir / "motivation" / f"{d}.features.csv").open() as handle:
            cached = {r["sample_key"]: r for r in csv.DictReader(handle)}
        values = list(selected[d].values())
        for start in range(0, len(values), 64):
            batch = values[start:start + 64]
            tokens = tokenizer(
                [r["raw_source"] for r in batch], add_special_tokens=False,
                truncation=True, max_length=budget, verbose=False,
            )["input_ids"]
            for r, ids in zip(batch, tokens):
                key = r["sample_key"]
                encoded[key] = np.asarray(ids, dtype=np.uint32).tobytes()
                n_lines = len(r["raw_source"].splitlines())
                if key in cached:
                    old = cached[key]
                    if int(old["label"]) != r["label"] or int(old["total_lines"]) != n_lines:
                        raise ValueError(f"{key}: cached feature label/length differs")
                    n_tokens = int(old["codebert_tokens"])
                else:
                    n_tokens = len(tokenizer(r["raw_source"], add_special_tokens=False, verbose=False)["input_ids"])
                features[key] = dict(total_lines=n_lines, codebert_tokens=n_tokens)
        print(f"{d}: tokenized {len(values)} prepared members; budget {budget}", flush=True)

    training_sources = [
        {r["raw_source"] for r in rows[d][s]} for d in datasets for s in ("train", "valid")
    ]
    cohorts, matched = {}, {}
    for d in datasets:
        with (root / f"{d}.members.csv").open() as handle:
            members = list(csv.DictReader(handle))
        if [(r["sample_key"], int(r["label"])) for r in members] != [(r["sample_key"], r["label"]) for r in rows[d]["test"]]:
            raise ValueError("formal membership order/labels differ")
        keys = {r["sample_key"] for r in members if r["independent"] == "True"}
        expected = {r["sample_key"] for r in rows[d]["test"] if not any(r["raw_source"] in s for s in training_sources)}
        if keys != expected:
            raise ValueError("formal exact-overlap exclusions differ")
        cohorts[d] = [r for r in rows[d]["test"] if r["sample_key"] in keys]
        matched[d] = match_lengths(cohorts[d], seed=42)
    labels["independent_test_disagreements"] = {}
    for d in datasets:
        disagreeing_keys = {
            key for conflict in conflicts for key in conflict[f"{d}_keys"].split(";")
        }
        labels["independent_test_disagreements"][d] = class_counts(
            [r for r in cohorts[d] if r["sample_key"] in disagreeing_keys]
        )

    data_statistics = []
    for d in datasets:
        for split, values in {**rows[d], "independent_test": cohorts[d], "length_matched_test": matched[d]}.items():
            y = [r["label"] for r in values]
            keys = {r["sample_key"] for r in values}
            negative = [r for r in values if r["label"] == 0]
            entry = dict(
                dataset=d, split=split, **class_counts(values),
                projects=len({r["project"] for r in values}),
                known_fixed_negative=sum(r["known_fixed_negative"] for r in negative),
                complete_repair_pairs=sum(a in keys and b in keys for a, b in pairs[d]),
                length_auc=float(roc_auc_score(y, [features[r["sample_key"]]["total_lines"] for r in values])),
                token_auc=float(roc_auc_score(y, [features[r["sample_key"]]["codebert_tokens"] for r in values])),
                input_conflicts=_input_conflicts(values, encoded),
            )
            for label in (0, 1):
                group = [r for r in values if r["label"] == label]
                entry[f"class_{label}"] = dict(
                    median_lines=float(np.median([features[r["sample_key"]]["total_lines"] for r in group])),
                    median_tokens=float(np.median([features[r["sample_key"]]["codebert_tokens"] for r in group])),
                    truncated=sum(features[r["sample_key"]]["codebert_tokens"] > budget for r in group),
                )
            data_statistics.append(entry)

    predictions, lexical, lexical_metrics = {}, {}, []
    for source in datasets:
        lexical_model = joblib.load(output_dir / "motivation" / f"{source}.tfidf.joblib")
        lexical_selection = json.loads((output_dir / "motivation" / f"{source}.tfidf.metrics.json").read_text())
        if any(lexical_selection["members"][s] != len(rows[source][s]) for s in ("train", "valid")):
            raise ValueError("saved lexical classifier Train/Valid sizes differ")
        for target in datasets:
            lexical[source, target] = dict(zip(
                (r["sample_key"] for r in rows[target]["test"]),
                lexical_model.predict_proba([r["raw_source"] for r in rows[target]["test"]])[:, 1].tolist(),
            ))
            source_token_spans.cache_clear()
            values = cohorts[target]
            lexical_metrics.append(dict(
                source=source, target=target, selected_C=lexical_selection["selected_C"],
                **metrics([r["label"] for r in values],
                          [lexical[source, target][r["sample_key"]] for r in values],
                          lexical_selection["valid"]["threshold"]),
            ))
        for method in ("bce", "lp_ft", "swa"):
            run = output_dir / source / method
            config = json.loads((run / "config.json").read_text())
            if config["input_protocol"]["source_budget"] != budget:
                raise ValueError("model input protocol differs")
            for target in datasets:
                name = "test" if source == target else f"{target}.test"
                saved = read_jsonl(run / f"{name}.predictions.jsonl")
                if [(r["sample_key"], r["label"]) for r in saved] != [(r["sample_key"], r["label"]) for r in rows[target]["test"]]:
                    raise ValueError("saved BCE/LP-FT/SWA prediction members differ")
                if len({r["threshold"] for r in saved}) != 1:
                    raise ValueError("saved predictions contain different thresholds")
                predictions[source, method, target] = {r["sample_key"]: r for r in saved}

    performance, pair_members = [], []
    for target in datasets:
        keys = {r["sample_key"] for r in cohorts[target]}
        for positive, negative in pairs[target]:
            if positive not in keys or negative not in keys:
                continue
            a, b = selected[target][positive], selected[target][negative]
            pair_members.append(dict(
                dataset=target, positive_key=positive, negative_key=negative,
                positive_lines=features[positive]["total_lines"], negative_lines=features[negative]["total_lines"],
                same_visible_input=encoded[positive] == encoded[negative],
                same_full_source=a["raw_source"] == b["raw_source"],
                line_similarity=SequenceMatcher(None, a["raw_source"].splitlines(), b["raw_source"].splitlines(), autojunk=False).ratio(),
                **{f"{source}_{method}_logit_difference":
                   predictions[source, method, target][positive]["logit"] - predictions[source, method, target][negative]["logit"]
                   for source in datasets for method in ("bce", "lp_ft", "swa")},
            ))
        for source in datasets:
            seen = {r["project_url"] for r in rows[source]["train"] if r["project_url"]}
            subsets = dict(
                independent=cohorts[target], length_matched=matched[target],
                not_truncated=[r for r in cohorts[target] if features[r["sample_key"]]["codebert_tokens"] <= budget],
                project_seen=[r for r in cohorts[target] if r["project_url"] in seen],
                project_unseen=[r for r in cohorts[target] if r["project_url"] and r["project_url"] not in seen],
            )
            for method in ("bce", "lp_ft", "swa"):
                pred = predictions[source, method, target]
                threshold = next(iter(pred.values()))["threshold"]
                for subset, values in subsets.items():
                    if not values:
                        continue
                    scores = [pred[r["sample_key"]]["score"] for r in values]
                    result = dict(
                        source=source, method=method, target=target, subset=subset,
                        positive=sum(r["label"] for r in values),
                        negative=sum(r["label"] == 0 for r in values),
                        **metrics([r["label"] for r in values], scores, threshold),
                        rho_lines=float(spearmanr(scores, [features[r["sample_key"]]["total_lines"] for r in values]).statistic),
                        rho_source_tfidf=float(spearmanr(scores, [lexical[source, target][r["sample_key"]] for r in values]).statistic),
                        negative_score_median=float(np.median([pred[r["sample_key"]]["score"] for r in values if r["label"] == 0])),
                        positive_score_median=float(np.median([pred[r["sample_key"]]["score"] for r in values if r["label"] == 1])),
                    )
                    if subset == "independent":
                        reference = next(r for r in formal["results"] if r["view"] == "independent" and r["training_dataset"] == source and r["test_dataset"] == target and r["method"] == method)
                        if any(result[k] != reference[k] for k in ("auc", "accuracy", "f1", "mcc", "auprc", "tp", "tn", "fp", "fn")):
                            raise ValueError("diagnosis does not reproduce formal metrics")
                    if method == "bce" and subset in ("independent", "length_matched"):
                        result["project_bootstrap_auc_95ci"] = _project_auc_interval(values, scores)
                    performance.append(result)
        print(f"{target}: audited fixed Test {len(cohorts[target])}, line matched {len(matched[target])}", flush=True)

    pair_statistics = []
    for target in datasets:
        values = [r for r in pair_members if r["dataset"] == target]
        for source in datasets:
            for method in ("bce", "lp_ft", "swa"):
                delta = [r[f"{source}_{method}_logit_difference"] for r in values]
                distinct = [r[f"{source}_{method}_logit_difference"] for r in values if not r["same_visible_input"]]
                pair_statistics.append(dict(
                    source=source, method=method, target=target, pairs=len(values),
                    wins=sum(v > 0 for v in delta), ties=sum(v == 0 for v in delta),
                    pair_ranking_accuracy=(sum(v > 0 for v in delta) + .5 * sum(v == 0 for v in delta)) / len(delta),
                    median_logit_difference=float(np.median(delta)),
                    median_line_similarity=float(np.median([r["line_similarity"] for r in values])),
                    identical_visible_inputs=sum(r["same_visible_input"] for r in values),
                    distinct_input_pairs=len(distinct),
                    distinct_input_wins=sum(v > 0 for v in distinct),
                    distinct_input_ties=sum(v == 0 for v in distinct),
                    distinct_input_ranking_accuracy=(sum(v > 0 for v in distinct) + .5 * sum(v == 0 for v in distinct)) / len(distinct) if distinct else None,
                ))
    members = []
    for d in datasets:
        keys = {r["sample_key"] for r in cohorts[d]}
        match_keys = {r["sample_key"] for r in matched[d]}
        for r in rows[d]["test"]:
            members.append(dict(
                dataset=d, sample_key=r["sample_key"], label=r["label"],
                independent=r["sample_key"] in keys, length_matched=r["sample_key"] in match_keys,
                project_url=r["project_url"], known_fixed_negative=r["known_fixed_negative"],
                **features[r["sample_key"]],
            ))
    summary = dict(
        protocol=dict(
            seed=42, datasets=list(datasets), source_budget=budget,
            inputs="Existing native/prepared data, saved CodeBERT predictions, cached length features and source-fitted TF-IDF models only.",
            training="No training, checkpoint selection, threshold adjustment, CodeBERT inference or changes to data/evaluation members.",
            matching="Existing match_lengths: exact total lines, seed 42, labels 0/1, no replacement. New diagnostic subsets of the fixed independent cohorts; historical Motivation subsets unchanged.",
            project_identity="native project_url, strip trailing slash/.git, casefold; no inferred aliases; unknown URLs excluded only from project seen/unseen breakdowns",
            uncertainty="1000 project-cluster bootstrap draws, percentile 95% intervals, seed 42; conditional on the fixed checkpoint/cohort",
            interpretation="Matching changes the evaluated population; associations and score correlations do not identify internal causal reliance. Native label disagreements do not establish which label is correct.",
        ),
        native_labels=labels, data=data_statistics, lexical=lexical_metrics,
        performance=performance, pairs=pair_statistics,
    )
    for name, values in (("members", members), ("performance", performance), ("pairs", pair_members), ("label_disagreements", conflicts)):
        if values:
            fields = list(dict.fromkeys(k for r in values for k in r))
            write_table(root / f"diagnosis.{name}.csv", [
                {k: json.dumps(r.get(k)) if isinstance(r.get(k), (dict, list)) else r.get(k) for k in fields}
                for r in values
            ])
    write_json(root / "diagnosis.json", summary)
    return dict(output=str(root / "diagnosis.json"), rows=len(performance), repair_pairs=len(pair_members))
