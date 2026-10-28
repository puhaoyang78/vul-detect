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


def paired_metrics(rows, scores):
    """Official LOSVER pairing and strict probability > .5 classification."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        matthews_corrcoef,
    )

    if not rows or len(rows) % 2 or len(rows) != len(scores):
        raise ValueError("complete, aligned official pairs required")
    labels = [r["target"] for r in rows]
    if labels != [1, 0] * (len(rows) // 2) or any(
        not math.isfinite(float(p)) or not 0 <= p <= 1 for p in scores
    ):
        raise ValueError("official labels and finite probabilities required")
    p = np.asarray(scores) > 0.5
    a, b = p[::2], p[1::2]
    if any(
        rows[i].get("pair_key") != rows[i + 1].get("pair_key")
        for i in range(0, len(rows), 2)
    ):
        raise ValueError("pair_key alignment differs")
    n = len(a)
    count = dict(
        p_c=int((a & ~b).sum()),
        p_v=int((a & b).sum()),
        p_b=int((~a & ~b).sum()),
        p_r=int((~a & b).sum()),
    )
    return dict(
        samples=len(rows),
        pairs=n,
        threshold=0.5,
        comparison=">",
        pair_counts=count,
        **{k: v / n for k, v in count.items()},
        accuracy=accuracy_score(labels, p),
        precision=precision_score(labels, p, zero_division=0),
        recall=recall_score(labels, p, zero_division=0),
        f1=f1_score(labels, p, zero_division=0),
        mcc=float(matthews_corrcoef(labels, p)),
        auc=float(roc_auc_score(labels, scores)),
        auprc=float(average_precision_score(labels, scores)),
        pair_ranking_accuracy=float(
            (np.asarray(scores)[::2] > np.asarray(scores)[1::2]).mean()
        ),
        pair_ranking_ties=int(
            (np.asarray(scores)[::2] == np.asarray(scores)[1::2]).sum()
        ),
        patched_fpr=float(b.mean()),
    )
