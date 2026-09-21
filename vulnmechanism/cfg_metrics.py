"""Same binary metrics/threshold grid as the baseline, plus paired error changes."""
from __future__ import annotations
import math
import struct


def decision_boundary(threshold: float) -> float:
    """model.py compares a float32 probability tensor with a float32 threshold."""
    return struct.unpack("f", struct.pack("f", threshold))[0]


def metrics(labels: list[int], scores: list[float], threshold: float = 0.5) -> dict:
    if not labels or len(labels) != len(scores):
        raise ValueError("nonempty equal-length labels/scores required")
    if (any(type(y) is not int or y not in (0, 1) for y in labels) or
            any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores) or
            not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("invalid labels, probabilities or threshold")
    pred = [s >= decision_boundary(threshold) for s in scores]
    tp = sum(y == 1 and p for y, p in zip(labels, pred))
    tn = sum(y == 0 and not p for y, p in zip(labels, pred))
    fp = sum(y == 0 and p for y, p in zip(labels, pred))
    fn = sum(y == 1 and not p for y, p in zip(labels, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    den = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    positives, negatives = sum(labels), len(labels)-sum(labels)
    auc = None
    if positives and negatives:
        pairs = sorted(zip(scores, labels))
        i, rank_sum = 0, 0.0
        while i < len(pairs):
            j = i + 1
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                j += 1
            rank_sum += (i+1+j)/2 * sum(y for _, y in pairs[i:j])
            i = j
        auc = (rank_sum - positives*(positives+1)/2)/(positives*negatives)
    return dict(accuracy=(tp+tn)/len(labels), precision=precision, recall=recall, f1=f1,
                mcc=(tp*tn-fp*fn)/den if den else 0.0, auc=auc, tp=tp, tn=tn, fp=fp, fn=fn,
                samples=len(labels), threshold=threshold)


def select_threshold(labels, scores):
    threshold = 0.5
    best = metrics(labels, scores, threshold)
    key = lambda m: (m["mcc"], m["f1"], m["accuracy"], -abs(m["threshold"]-0.5))
    for i in range(5, 96):
        candidate = metrics(labels, scores, i/100)
        if key(candidate) > key(best):
            best = candidate
    return best["threshold"], best


def paired_changes(baseline: list[dict], candidate: list[dict], *, base_threshold=None,
                   candidate_threshold=None) -> dict:
    b = {r["sample_key"]: r for r in baseline}
    c = {r["sample_key"]: r for r in candidate}
    if len(b) != len(baseline) or len(c) != len(candidate) or set(b) != set(c):
        raise ValueError("prediction cohorts must match exactly, without duplicate keys")
    changes = {name: [] for name in ("fn_to_tp", "fp_to_tn", "tp_to_fn", "tn_to_fp")}
    for key in b:
        left, right = b[key], c[key]
        if any(left.get(f) != right.get(f) for f in ("dataset", "label", "split", "source_sha256")):
            raise ValueError(f"prediction identity mismatch: {key}")
        old = int(left["score"] >= decision_boundary(base_threshold)) if base_threshold is not None else left["prediction"]
        new = int(right["score"] >= decision_boundary(candidate_threshold)) if candidate_threshold is not None else right["prediction"]
        if old == new:
            continue
        y = left["label"]
        name = ("fn_to_tp" if old == 0 else "tp_to_fn") if y == 1 else ("fp_to_tn" if old == 1 else "tn_to_fp")
        changes[name].append(key)
    counts = {name: len(keys) for name, keys in changes.items()}
    return dict(counts=counts, sample_keys=changes,
                net_corrected=counts["fn_to_tp"]+counts["fp_to_tn"]-counts["tp_to_fn"]-counts["tn_to_fp"],
                net_fewer_fn=counts["fn_to_tp"]-counts["tp_to_fn"],
                net_fewer_fp=counts["fp_to_tn"]-counts["tn_to_fp"])
