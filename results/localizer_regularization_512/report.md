# Original LOSVER localizer: regularization and line R-Drop

512 token Pilot, 1024 Train/256 Valid functions, 4 epochs and 512 optimizer steps per method. Member seed42; training seed123456. A reuses `results/repair_region_512/pilot`. B/C start from the same pretrained UniXcoder with identical initialization, members, Focal, LR/schedule and two-forward budget; C alone adds lambda=.1.

Original binary Diff labels and visible physical lines are identical. Focal alpha=.8, gamma=1.5, pos_weight=5. Best checkpoint remains original Valid positive F1 rounded to4 decimals, earlier tie; threshold remains >.5. AUPRC is reported at that checkpoint, never used to select a different epoch. Test and detectors are not evaluated or trained.

New attention/head dropout: A=0, B=C=.1. Encoder hidden/attention dropout stays .1. Weight decay: A=0, B=C=.01, decoupled AFTER the unchanged adaptive AdamW update; all bias (including in_proj_bias) and LayerNorm parameters are excluded. The two Focal losses are averaged in B/C; independent Bernoulli symmetric KL is averaged over real lines only, with gradients on both logits. Validation/inference uses one dropout-disabled forward.

## Selected checkpoint metrics

Metrics pool valid visible lines; vulnerable/patched refer to the original function target, not predicted classes. Positive-line fraction uses >.5; AUPRC uses continuous probabilities. All count denominators are saved in comparison.csv.

| Method | Split | Side | Epoch | Lines | AUPRC | F1 | Precision | Recall | Predicted positive fraction |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_losver | train | all | 4 | 37815 | 0.8881 | 0.5901 | 0.4194 | 0.9952 | 0.1685 |
| original_losver | train | vulnerable | 4 | 18754 | 0.8568 | 0.5134 | 0.3458 | 0.9959 | 0.1506 |
| original_losver | train | patched | 4 | 19061 | 0.9076 | 0.6457 | 0.4780 | 0.9947 | 0.1860 |
| original_losver | valid | all | 4 | 8701 | 0.1506 | 0.2283 | 0.1603 | 0.3966 | 0.2008 |
| original_losver | valid | vulnerable | 4 | 4291 | 0.1145 | 0.1738 | 0.1131 | 0.3750 | 0.1979 |
| original_losver | valid | patched | 4 | 4410 | 0.1886 | 0.2730 | 0.2049 | 0.4089 | 0.2036 |
| dropout_weight_decay | train | all | 4 | 37815 | 0.8523 | 0.5530 | 0.3830 | 0.9940 | 0.1843 |
| dropout_weight_decay | train | vulnerable | 4 | 18754 | 0.8218 | 0.4754 | 0.3120 | 0.9980 | 0.1673 |
| dropout_weight_decay | train | patched | 4 | 19061 | 0.8731 | 0.6108 | 0.4413 | 0.9918 | 0.2009 |
| dropout_weight_decay | valid | all | 4 | 8701 | 0.1587 | 0.2318 | 0.1623 | 0.4051 | 0.2025 |
| dropout_weight_decay | valid | vulnerable | 4 | 4291 | 0.1262 | 0.1808 | 0.1173 | 0.3945 | 0.2007 |
| dropout_weight_decay | valid | patched | 4 | 4410 | 0.1936 | 0.2739 | 0.2053 | 0.4111 | 0.2043 |
| line_rdrop | train | all | 4 | 37815 | 0.8613 | 0.5624 | 0.3922 | 0.9940 | 0.1800 |
| line_rdrop | train | vulnerable | 4 | 18754 | 0.8281 | 0.4854 | 0.3207 | 0.9980 | 0.1628 |
| line_rdrop | train | patched | 4 | 19061 | 0.8828 | 0.6194 | 0.4503 | 0.9918 | 0.1969 |
| line_rdrop | valid | all | 4 | 8701 | 0.1588 | 0.2288 | 0.1603 | 0.3994 | 0.2022 |
| line_rdrop | valid | vulnerable | 4 | 4291 | 0.1260 | 0.1749 | 0.1137 | 0.3789 | 0.1988 |
| line_rdrop | valid | patched | 4 | 4410 | 0.1945 | 0.2729 | 0.2042 | 0.4111 | 0.2054 |

## Train minus Valid gaps

| Method | Side | AUPRC gap | F1 gap |
| --- | --- | ---: | ---: |
| original_losver | all | 0.7375 | 0.3618 |
| original_losver | vulnerable | 0.7422 | 0.3396 |
| original_losver | patched | 0.7190 | 0.3727 |
| dropout_weight_decay | all | 0.6936 | 0.3212 |
| dropout_weight_decay | vulnerable | 0.6956 | 0.2945 |
| dropout_weight_decay | patched | 0.6795 | 0.3369 |
| line_rdrop | all | 0.7025 | 0.3336 |
| line_rdrop | vulnerable | 0.7021 | 0.3104 |
| line_rdrop | patched | 0.6883 | 0.3465 |

## Valid differences at the selected checkpoints

Differences below are absolute metric differences, not relative percentages.

| Comparison | Side | AUPRC difference | F1 difference |
| --- | --- | ---: | ---: |
| dropout_weight_decay minus original_losver | all | +0.008069 | +0.003475 |
| dropout_weight_decay minus original_losver | vulnerable | +0.011622 | +0.007086 |
| dropout_weight_decay minus original_losver | patched | +0.004968 | +0.000874 |
| line_rdrop minus dropout_weight_decay | all | +0.000075 | -0.002963 |
| line_rdrop minus dropout_weight_decay | vulnerable | -0.000179 | -0.005909 |
| line_rdrop minus dropout_weight_decay | patched | +0.000891 | -0.001010 |

A smaller gap can arise from worse Train ranking; read it together with Valid absolute AUPRC. Compare C against B to isolate KL from the base regularization and the two supervised forwards. B/C double the training forwards, not optimizer steps. These are one-seed, four-epoch pilot results, not full-data or Test claims.

## Checks and reproduction

[GPU smoke](gpu-smoke.json) records actual Train forward/backward, new-layer stochastic dropout, zero-gradient decoupled weight decay, padding-invariant KL, both logit gradients and exact save/reload. [Protocol checks](protocol-checks.json) inspect the persisted optimizer/scheduler steps and RNG states. The [full test log](tests.log) and [resume test log](resume-tests.log) retain the executed correctness checks.

The exact B/C train and report commands are in the [README](../../README.md). Configs, epoch/component losses and source-only line probabilities are retained under `dropout_weight_decay/` and `line_rdrop/`. Checkpoints stay locally in those same run directories; historical A remains at its original path.
