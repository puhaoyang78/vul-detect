"""Nested, train-only incremental-information diagnosis; no residual network."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .benchmark_view import select_source_records
from .prepare_benchmark import normalized_source
from .progress import print_table

PROTOCOL_VERSION = 1


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_cohort(dataset: str, manifest: str) -> tuple[list[dict], dict[str, str]]:
    from .model import _read_jsonl
    records, _ = select_source_records(_read_jsonl(dataset), "primevul")
    records = [r for r in records if r["split"] == "train"]
    metadata = {}
    for row in read_rows(Path(manifest)):
        if row.get("dataset") != "primevul" or row.get("split") != "train":
            continue
        key = row["sample_key"]
        if key in metadata:
            raise ValueError(f"duplicate manifest key: {key}")
        if not row.get("commit") or (not row.get("repo") and not re.fullmatch(r"[0-9a-fA-F]{40}", row["commit"])):
            raise ValueError(f"missing repository/commit provenance: {key}")
        metadata[key] = row
    for row in records:
        meta = metadata.get(row["sample_key"])
        if meta is None or meta["label"] != row["label"] or meta["function"] != row["raw_source"]:
            raise ValueError(f"dataset/manifest mismatch: {row['sample_key']}")
    # Connected components include unselected train rows so links remain transitive.
    parent = {key: key for key in metadata}

    def root(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners = {}
    for key, row in sorted(metadata.items()):
        # Full commit IDs link across forks and also cover a known missing-repo row.
        commit_key = (("commit", row["commit"].lower()) if re.fullmatch(r"[0-9a-fA-F]{40}", row["commit"])
                      else ("revision", row["repo"], row["commit"]))
        links = [commit_key,
                 ("source", normalized_source(row["function"]))]
        links.extend(("counterpart", value) for value in row.get("counterpart_groups", []))
        if row.get("pair_id"):
            links.append(("pair", row["pair_id"]))
        for link in links:
            if link in owners:
                a, b = root(key), root(owners[link])
                parent[max(a, b)] = min(a, b)
            else:
                owners[link] = key
    return records, {r["sample_key"]: root(r["sample_key"]) for r in records}


def group_folds(keys: list[str], rows: dict[str, dict], groups: dict[str, str],
                count: int, seed: int) -> list[tuple[list[str], list[str]]]:
    if count < 2 or len({groups[k] for k in keys}) < count:
        raise ValueError("not enough independent groups for requested folds")
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    result = []
    for train, test in splitter.split(np.zeros(len(keys)), [rows[k]["label"] for k in keys],
                                      [groups[k] for k in keys]):
        a, b = [keys[i] for i in train], [keys[i] for i in test]
        if {groups[k] for k in a} & {groups[k] for k in b}:
            raise AssertionError("group leakage")
        if any({rows[k]["label"] for k in part} != {0, 1} for part in (a, b)):
            raise ValueError("a grouped fold lacks one class; reduce folds or change seed explicitly")
        result.append((a, b))
    return result


def make_folds(records: list[dict], groups: dict[str, str], *, outer_folds: int,
               inner_folds: int, calibration_folds: int, seed: int) -> list[dict]:
    rows = {r["sample_key"]: r for r in records}
    if len(rows) != len(records) or any(r["split"] != "train" for r in records):
        raise ValueError("diagnosis requires unique official-training records")
    result = []
    for outer, (available, test) in enumerate(group_folds(sorted(rows), rows, groups, outer_folds, seed)):
        develop, calibration = group_folds(available, rows, groups, calibration_folds, seed+100+outer)[0]
        inner = group_folds(develop, rows, groups, inner_folds, seed+200+outer)
        result.append(dict(outer=outer, develop=develop, calibration=calibration, evaluation=test,
                           inner=[dict(fit=a, predict=b) for a, b in inner]))
    return result


def relation_features(row: dict, group: str) -> dict:
    items = [i for i in row["mechanism_items"] if i["category"] == "MECHANISM_RELATION"]
    counts = Counter(i["kind"] for i in items)
    # No candidate, label, provenance or audit-quality field enters R.
    text = "\n".join(sorted(f"{i['kind']} {i['detail']} {i.get('state', '')}" for i in items))
    return dict(sample_key=row["sample_key"], label=row["label"], group=group,
                counts=dict(sorted(counts.items())), content=text, length=len(text),
                observed_relation=bool(items))


def file_identity(path: str) -> dict:
    value = Path(path).resolve()
    stat = value.stat()
    return dict(path=str(value), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def prepare(args) -> tuple[list[dict], list[dict], list[dict]]:
    if args.shuffle_repeats < 1 or args.length_bin < 1 or args.max_features < 1 or args.probe_c <= 0:
        raise ValueError("shuffle_repeats, length_bin, max_features and probe_c must be positive")
    if min(args.outer_folds, args.inner_folds, args.calibration_folds) < 2:
        raise ValueError("all fold counts must be at least two")
    if min(args.epochs, args.batch_size, args.gradient_accumulation, args.source_max_length,
           args.log_every, args.lora_r, args.lora_alpha) <= 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid baseline training parameters")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("lora_dropout must be in [0, 1)")
    config = {k: getattr(args, k) for k in (
        "model", "outer_folds", "inner_folds", "calibration_folds", "seed", "epochs",
        "source_max_length", "batch_size", "gradient_accumulation", "learning_rate",
        "weight_decay", "lora_r", "lora_alpha", "lora_dropout", "shuffle_repeats",
        "length_bin", "probe_c", "max_features")}
    config.update(protocol_version=PROTOCOL_VERSION, dataset=file_identity(args.dataset),
                  manifest=file_identity(args.manifest),
                  packages={p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "scikit-learn", "numpy")})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("run configuration or input file metadata changed; use a different output directory")
    if not config_path.exists() and any(output.iterdir()):
        raise ValueError("output directory is nonempty without run.json")
    records, groups = load_cohort(args.dataset, args.manifest)
    folds = make_folds(records, groups, outer_folds=args.outer_folds, inner_folds=args.inner_folds,
                       calibration_folds=args.calibration_folds, seed=args.seed)
    features = [relation_features(r, groups[r["sample_key"]]) for r in records]
    if (output / "folds.json").exists() and json.loads((output / "folds.json").read_text()) != folds:
        raise ValueError("fold assignments changed")
    if (output / "features.jsonl").exists() and read_rows(output / "features.jsonl") != features:
        raise ValueError("diagnostic features changed")
    write_json(config_path, config)
    write_json(output / "folds.json", folds)
    write_rows(output / "features.jsonl", features)
    summary = dict(samples=len(records), groups=len(set(groups.values())),
                   labels=dict(Counter(r["label"] for r in records)),
                   observed_relations=sum(r["observed_relation"] for r in features),
                   baseline_fits=len(folds)*(args.inner_folds+1), fixed_epochs=args.epochs,
                   official_valid_test_used=False,
                   folds=[dict(outer=f["outer"], develop=len(f["develop"]), calibration=len(f["calibration"]),
                               evaluation=len(f["evaluation"])) for f in folds])
    print(json.dumps(summary, indent=2), flush=True)
    print_table("Nested OOF · official PrimeVul TRAIN only",
                ["Outer fold", "Develop", "Calibration", "Evaluation"],
                [[f["outer"]+1, len(f["develop"]), len(f["calibration"]), len(f["evaluation"])] for f in folds])
    return records, folds, features


def fold_jobs(args, fold: dict):
    base = Path(args.output) / f"outer_{fold['outer']:02d}"
    return [(base / f"inner_{i:02d}", inner["fit"], inner["predict"],
             args.seed+1000+fold["outer"]*100+i) for i, inner in enumerate(fold["inner"])] + [
        (base / "outer_model", fold["develop"], fold["calibration"]+fold["evaluation"], args.seed+2000+fold["outer"])]


def check_membership(directory: Path, fit: list[str], predict: list[str], seed: int) -> dict:
    specification = dict(fit=fit, predict=predict, seed=seed)
    path = directory / "membership.json"
    if not path.exists() or json.loads(path.read_text()) != specification:
        raise ValueError(f"missing or changed prediction provenance: {directory}")
    return specification


def validate_scores(path: Path, keys: list[str], rows: dict[str, dict]) -> dict[str, float]:
    values = read_rows(path)
    if len(values) != len(keys) or {r["sample_key"] for r in values} != set(keys):
        raise ValueError(f"prediction coverage mismatch: {path}")
    result = {}
    for row in values:
        key = row["sample_key"]
        if key in result or row["label"] != rows[key]["label"] or not math.isfinite(row["logit"]):
            raise ValueError(f"invalid prediction: {path}:{key}")
        result[key] = float(row["logit"])
    return result


def run_oof(args, records: list[dict], folds: list[dict]) -> None:
    import torch
    from .model import predict_checkpoint, train_model
    rows = {r["sample_key"]: r for r in records}
    jobs = [job for fold in folds for job in fold_jobs(args, fold)]
    for directory, fit, predict, seed in tqdm(jobs, desc="Nested OOF baselines", unit="model", file=sys.stdout):
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint, scores = directory / "baseline.pt", directory / "predictions.jsonl"
        # Persist exact membership/config before expensive work, never infer OOF from old checkpoints.
        specification = dict(fit=fit, predict=predict, seed=seed)
        spec_path = directory / "membership.json"
        if spec_path.exists() and json.loads(spec_path.read_text()) != specification:
            raise ValueError(f"membership changed: {directory}")
        if not spec_path.exists() and (checkpoint.exists() or scores.exists()):
            raise ValueError(f"unowned artifacts: {directory}")
        write_json(spec_path, specification)
        if scores.exists():
            validate_scores(scores, predict, rows)
            continue
        if set(fit) & set(predict):
            raise AssertionError("fit/predict overlap")
        if not checkpoint.exists():
            print(f"\nFit {directory}: train={len(fit)}, predict={len(predict)} (fixed {args.epochs} epochs)", flush=True)
            train_model(None, checkpoint, records=[rows[k] for k in fit], fixed_epochs=True,
                        variant="baseline", model_path=args.model, source_max_length=args.source_max_length,
                        batch_size=args.batch_size, gradient_accumulation=args.gradient_accumulation,
                        epochs=args.epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        seed=seed, device=args.device, log_every=args.log_every)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if any(saved.get(k) != v for k, v in dict(selection="fixed_epochs", variant="baseline",
                   seed=seed, selected_epoch=args.epochs, model_path=args.model).items()):
                raise ValueError(f"checkpoint was not fitted for this OOF job: {checkpoint}")
            del saved
        logits = predict_checkpoint(checkpoint, [rows[k] for k in predict], batch_size=args.batch_size,
                                    device=args.device).tolist()
        write_rows(scores, [dict(sample_key=k, label=rows[k]["label"], logit=z) for k, z in zip(predict, logits)])
        validate_scores(scores, predict, rows)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def shuffled_content(rows: list[dict], *, seed: int, length_bin: int) -> tuple[list[dict], dict]:
    """Permutation within exact type/count and char-length bins, between independent groups.

    Buckets with a majority provenance group cannot be deranged without related donors;
    leave the whole bucket untouched and explicitly exclude it from matched evaluation.
    """
    if length_bin < 1:
        raise ValueError("length_bin must be positive")
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        if row["observed_relation"]:
            buckets[(tuple(sorted(row["counts"].items())), row["length"] // length_bin)].append(i)
    rng = random.Random(seed)
    output = [dict(r) for r in rows]
    donors, eligible = {}, []
    for indices in buckets.values():
        by_group = defaultdict(list)
        for i in indices:
            by_group[rows[i]["group"]].append(i)
        largest = max(map(len, by_group.values()))
        if largest * 2 > len(indices):
            continue
        group_list = list(by_group.values())
        rng.shuffle(group_list)
        for values in group_list:
            rng.shuffle(values)
        ordered = [i for values in group_list for i in values]
        shift = rng.randint(largest, len(ordered)-largest)
        for j, receiver in enumerate(ordered):
            donor = ordered[(j+shift) % len(ordered)]
            if rows[receiver]["group"] == rows[donor]["group"]:
                raise AssertionError("related shuffle donor")
            output[receiver]["content"] = rows[donor]["content"]
            donors[rows[receiver]["sample_key"]] = rows[donor]["sample_key"]
            eligible.append(rows[receiver]["sample_key"])
    changed = sum(a["content"] != b["content"] for a, b in zip(rows, output))
    return output, dict(eligible=sorted(eligible), donors=donors, eligible_count=len(eligible),
                        observed_count=sum(r["observed_relation"] for r in rows),
                        changed_count=changed, samples=len(rows))


def metadata_features(row: dict) -> dict:
    return {"exists": float(row["observed_relation"]), "count": math.log1p(sum(row["counts"].values())),
            **{f"count:{k}": math.log1p(v) for k, v in row["counts"].items()}}


class Probe:
    """Same fixed regularized logistic model for all three feature sets."""
    def __init__(self, variant: str, *, c: float, max_features: int):
        if variant not in {"S", "S+U", "S+U+R"}:
            raise ValueError(variant)
        self.variant, self.c, self.max_features = variant, c, max_features
        self.score_scaler = StandardScaler()
        self.vectorizer = DictVectorizer()
        self.metadata_scaler = StandardScaler(with_mean=False)
        self.text = None
        self.classifier = LogisticRegression(C=c, solver="liblinear", max_iter=3000, random_state=0)

    def features(self, rows: list[dict], *, fit: bool):
        scores = np.array([r["logit"] for r in rows]).reshape(-1, 1)
        blocks = [sparse.csr_matrix(self.score_scaler.fit_transform(scores) if fit else self.score_scaler.transform(scores))]
        if self.variant != "S":
            metadata = [metadata_features(r) for r in rows]
            matrix = self.vectorizer.fit_transform(metadata) if fit else self.vectorizer.transform(metadata)
            blocks.append(self.metadata_scaler.fit_transform(matrix) if fit else self.metadata_scaler.transform(matrix))
        if self.variant == "S+U+R":
            if fit and any(r["content"].strip() for r in rows):
                self.text = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), max_features=self.max_features,
                                          token_pattern=r"(?u)[A-Za-z_]\w*|\d+|->|<=|>=|==|!=|<<|>>|[^\w\s]",
                                          sublinear_tf=True)
                matrix = self.text.fit_transform([r["content"] for r in rows])
                blocks.append(matrix)
            elif self.text is not None:
                blocks.append(self.text.transform([r["content"] for r in rows]))
        return sparse.hstack(blocks, format="csr")

    def fit(self, rows: list[dict]):
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError("probe fitting requires both classes")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.classifier.fit(self.features(rows, fit=True), [r["label"] for r in rows])
        return self

    def predict(self, rows: list[dict]) -> np.ndarray:
        return self.classifier.predict_proba(self.features(rows, fit=False))[:, 1]


def metrics(labels, probabilities, threshold: float) -> dict:
    y, p = np.asarray(labels), np.asarray(probabilities)
    if len(y) == 0:
        return dict(samples=0, positive=0, negative=0, auc=None, mcc=None, accuracy=None,
                    precision=None, recall=None, f1=None, log_loss=None, tp=0, fp=0, tn=0, fn=0)
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel().tolist()
    return dict(samples=len(y), positive=int(y.sum()), negative=int(len(y)-y.sum()),
                auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                mcc=float(matthews_corrcoef(y, predicted)), accuracy=float(accuracy_score(y, predicted)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)),
                f1=float(f1_score(y, predicted, zero_division=0)),
                log_loss=float(log_loss(y, p, labels=[0, 1])), tp=tp, fp=fp, tn=tn, fn=fn)


def choose_threshold(rows: list[dict], probabilities) -> float:
    y = np.array([r["label"] for r in rows], dtype=bool)
    if set(y) != {0, 1}:
        raise ValueError("calibration requires both classes")
    p = np.asarray(probabilities)
    def key(t):
        predicted = p >= t
        tp, fp = int((predicted & y).sum()), int((predicted & ~y).sum())
        fn, tn = int((~predicted & y).sum()), int((~predicted & ~y).sum())
        denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = (tp*tn-fp*fn)/denominator if denominator else 0.0
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        return (mcc, f1, (tp+tn)/len(y), -abs(t-0.5))
    return max([i/100 for i in range(5, 96)], key=key)


def prediction_report(rows: list[dict], probabilities, threshold: float,
                      baseline_probabilities, baseline_threshold: float, eligible: set[str]) -> dict:
    def section(indices):
        y = np.array([rows[i]["label"] for i in indices])
        p = np.asarray(probabilities)[indices]
        b = np.asarray(baseline_probabilities)[indices] >= baseline_threshold
        a = p >= threshold
        result = metrics(y, p, threshold)
        result["transitions_vs_S"] = {f"label_{label}": dict(
            corrected=int(((y == label) & (b != y) & (a == y)).sum()),
            damaged=int(((y == label) & (b == y) & (a != y)).sum())) for label in (0, 1)}
        return result
    return dict(threshold=threshold, all=section(list(range(len(rows)))),
                matched=section([i for i, r in enumerate(rows) if r["sample_key"] in eligible]),
                relation_present=section([i for i, r in enumerate(rows) if r["observed_relation"]]))


def run_probes(args, folds: list[dict], features: list[dict]) -> dict:
    by_key = {r["sample_key"]: r for r in features}
    reports, predictions = [], []
    for fold in tqdm(folds, desc="Incremental probes", unit="outer fold", file=sys.stdout):
        directory = Path(args.output) / f"outer_{fold['outer']:02d}"
        for job_directory, fit, predict, seed in fold_jobs(args, fold):
            check_membership(job_directory, fit, predict, seed)
        train_scores = {}
        for i, inner in enumerate(fold["inner"]):
            scores = validate_scores(directory / f"inner_{i:02d}" / "predictions.jsonl", inner["predict"], by_key)
            if train_scores.keys() & scores.keys():
                raise ValueError("duplicate inner OOF predictions")
            train_scores.update(scores)
        if set(train_scores) != set(fold["develop"]):
            raise ValueError("incomplete develop OOF predictions")
        held_scores = validate_scores(directory / "outer_model" / "predictions.jsonl",
                                      fold["calibration"]+fold["evaluation"], by_key)
        partitions = [[dict(by_key[k], logit=scores[k]) for k in fold[name]] for name, scores in
                      (("develop", train_scores), ("calibration", held_scores), ("evaluation", held_scores))]
        train, calibration, evaluation = partitions
        shuffles = []
        for repeat in range(args.shuffle_repeats):
            shuffles.append([shuffled_content(part, seed=args.seed+10000+fold["outer"]*1000+repeat*3+j,
                                               length_bin=args.length_bin) for j, part in enumerate(partitions)])
        eligible = set(shuffles[0][2][1]["eligible"])
        fits = {}
        for variant in tqdm(("S", "S+U", "S+U+R"), desc=f"Fold {fold['outer']+1} · content probes", file=sys.stdout):
            probe = Probe(variant, c=args.probe_c, max_features=args.max_features).fit(train)
            threshold = choose_threshold(calibration, probe.predict(calibration))
            fits[variant] = (probe, threshold, probe.predict(evaluation))
        base_probability, base_threshold = fits["S"][2], fits["S"][1]
        results = {}

        def add(name, p, threshold):
            results[name] = prediction_report(evaluation, p, threshold, base_probability, base_threshold, eligible)
            for row, probability in zip(evaluation, p):
                predictions.append(dict(sample_key=row["sample_key"], outer=fold["outer"], label=row["label"],
                                        variant=name, probability=float(probability), threshold=threshold,
                                        matched=row["sample_key"] in eligible))
        for variant, (_, threshold, probability) in fits.items():
            add(variant, probability, threshold)
        shuffle_reports = []
        for repeat, shuffled in enumerate(tqdm(shuffles, desc="Controlled shuffle", unit="repeat", file=sys.stdout)):
            shuffled_train, shuffled_cal, shuffled_eval = [entry[0] for entry in shuffled]
            probe = Probe("S+U+R", c=args.probe_c, max_features=args.max_features).fit(shuffled_train)
            threshold = choose_threshold(shuffled_cal, probe.predict(shuffled_cal))
            add(f"shuffle_{repeat}", probe.predict(shuffled_eval), threshold)
            # Evaluation-only intervention: same fitted model and original threshold.
            add(f"intervention_{repeat}", fits["S+U+R"][0].predict(shuffled_eval), fits["S+U+R"][1])
            shuffle_reports.append({name: entry[1] for name, entry in zip(("develop", "calibration", "evaluation"), shuffled)})
        report = dict(outer=fold["outer"], results=results, shuffle=shuffle_reports)
        reports.append(report)
        write_json(directory / "probe_results.json", report)
        write_json(Path(args.output) / "probe_results.json", dict(folds=reports, complete=False))
        write_rows(Path(args.output) / "probe_predictions.jsonl", predictions)
        print_probe_fold(report)
    summary = dict(complete=True, folds=reports, interpretation=(
        "Outer-fold evaluation uses official train only. Thresholds use a separate calibration partition. "
        "R is current relation text/state, not verified path conditions. No automatic significance or pass claim. "
        "Compare matched-subset full vs shuffled content; unchanged text exchanges are counted separately."))
    write_rows(Path(args.output) / "probe_predictions.jsonl", predictions)
    summary["fold_summary"] = summarize_folds(reports)
    write_json(Path(args.output) / "probe_results.json", summary)
    print_table("Paired outer-fold deltas · S+U+R minus S+U (no pooled AUC)",
                ["Metric", "Mean delta", "Fold SD", "Improved folds"],
                [[k, f"{v['mean_delta']:+.5f}", f"{v['fold_sd']:.5f}",
                  f"{v['improved_folds']}/{v['folds']}"] for k, v in summary["fold_summary"].items()])
    return summary


def summarize_folds(reports: list[dict]) -> dict:
    result = {}
    for key in ("auc", "mcc", "log_loss", "tp", "fp"):
        deltas = [r["results"]["S+U+R"]["all"][key]-r["results"]["S+U"]["all"][key] for r in reports]
        result[key] = dict(mean_delta=float(np.mean(deltas)),
                           fold_sd=float(np.std(deltas, ddof=1)) if len(deltas)>1 else 0.0,
                           improved_folds=sum(d < 0 if key in ("log_loss", "fp") else d > 0 for d in deltas),
                           folds=len(deltas))
    return result


def print_probe_fold(report: dict) -> None:
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"
    for subset in ("all", "matched"):
        print_table(f"Outer fold {report['outer']+1} · {subset}",
                    ["Probe", "N", "AUC", "MCC", "TP", "FP", "Log loss"],
                    [[name, r[subset]["samples"], fmt(r[subset]["auc"]), fmt(r[subset]["mcc"]),
                      r[subset]["tp"], r[subset]["fp"], fmt(r[subset]["log_loss"])]
                     for name, r in report["results"].items()])
    coverage = report["shuffle"][0]["evaluation"]
    print(f"Shuffle coverage: {coverage['eligible_count']}/{coverage['observed_count']} relation samples; "
          f"actually changed text (repeat 0): {coverage['changed_count']}. "
          "Matched comparisons use identical evaluation keys.\n", flush=True)


def run_diagnostic(args):
    records, folds, features = prepare(args)
    if args.stage in ("oof", "all"):
        run_oof(args, records, folds)
    if args.stage in ("probe", "all"):
        return run_probes(args, folds, features)
