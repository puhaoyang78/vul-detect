"""Four-dataset diagnostics using ordinary BCE and shallow Shortcut Features."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random

import joblib
import numpy as np
import sklearn
from scipy.stats import ks_2samp
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
