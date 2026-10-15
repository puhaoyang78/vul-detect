"""Shared binary metrics and the established clean-Valid threshold grid."""

from __future__ import annotations
import math
import struct
from sklearn.metrics import average_precision_score, roc_auc_score


def decision_boundary(threshold):
    """Use the same float32 boundary in training, inference and robustness."""
    return struct.unpack("f", struct.pack("f", threshold))[0]


def metrics(labels, scores, threshold=0.5):
    if not labels or len(labels) != len(scores):
        raise ValueError("nonempty equal-length labels/scores required")
    if (
        any(type(y) is not int or y not in (0, 1) for y in labels)
        or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("invalid labels, probabilities or threshold")
    predictions = [s >= decision_boundary(threshold) for s in scores]
    tp = sum(y == 1 and p for y, p in zip(labels, predictions))
    tn = sum(y == 0 and not p for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p for y, p in zip(labels, predictions))
    fn = sum(y == 1 and not p for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    both = 0 < sum(labels) < len(labels)
    return dict(
        accuracy=(tp + tn) / len(labels),
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=(tp * tn - fp * fn) / denominator if denominator else 0.0,
        auc=float(roc_auc_score(labels, scores)) if both else None,
        auprc=float(average_precision_score(labels, scores)) if both else None,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        confusion_matrix=[[tn, fp], [fn, tp]],
        samples=len(labels),
        threshold=threshold,
    )


def select_threshold(labels, scores):
    best = metrics(labels, scores, 0.5)
    key = lambda m: (m["mcc"], m["f1"], m["accuracy"], -abs(m["threshold"] - 0.5))
    for value in range(5, 96):
        candidate = metrics(labels, scores, value / 100)
        if key(candidate) > key(best):
            best = candidate
    return best["threshold"], best
