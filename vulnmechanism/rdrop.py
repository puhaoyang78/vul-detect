"""Binary R-Drop loss shared by graph and source-only classifiers."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def binary_rdrop_loss(logits1: torch.Tensor, logits2: torch.Tensor,
                      labels: torch.Tensor, *, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total loss, mean two-pass BCE, and mean symmetric Bernoulli KL.

    Each scalar logit defines P = [sigmoid(-z), sigmoid(z)]. Both branches
    retain gradients, including when alpha is zero for the double-CE control.
    """
    if logits1.shape != logits2.shape or logits1.shape != labels.shape or logits1.ndim != 1:
        raise ValueError("R-Drop requires matching one-dimensional logits and labels")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("R-Drop alpha must be finite and nonnegative")
    z1, z2, y = logits1.float(), logits2.float(), labels.float()
    bce = 0.5 * (F.binary_cross_entropy_with_logits(z1, y) +
                 F.binary_cross_entropy_with_logits(z2, y))
    log_pos1, log_neg1 = F.logsigmoid(z1), F.logsigmoid(-z1)
    log_pos2, log_neg2 = F.logsigmoid(z2), F.logsigmoid(-z2)
    pos1, neg1 = log_pos1.exp(), log_neg1.exp()
    pos2, neg2 = log_pos2.exp(), log_neg2.exp()
    kl12 = pos1 * (log_pos1 - log_pos2) + neg1 * (log_neg1 - log_neg2)
    kl21 = pos2 * (log_pos2 - log_pos1) + neg2 * (log_neg2 - log_neg1)
    kl = 0.5 * (kl12 + kl21).mean()
    return bce + alpha * kl, bce, kl
