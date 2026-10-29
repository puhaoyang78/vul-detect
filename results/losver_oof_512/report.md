# LOSVER two-fold OOF 512 Pilot

Train 1,024 functions / 512 pairs; Valid 256 / 128. No Test was read. Member sampling seed 42, fold/training seed 123456. Original Diff, Focal and binary 5/1 Softmax unchanged. Both detectors use the same completed full Pilot localizer on Valid and for single-function inference.

## Localization

| Prediction source | Lines | Precision | Recall | F1 | AP | Selected lines | No prediction functions |
|---|---:|---:|---:|---:|---:|---:|---:|
| in_sample | 37815 | 0.4194 | 0.9952 | 0.5901 | 0.8881 | 16.85% | 261/1024 |
| oof | 37815 | 0.1398 | 0.3073 | 0.1922 | 0.1288 | 15.60% | 192/1024 |
| full_valid | 8701 | 0.1603 | 0.3966 | 0.2283 | 0.1506 | 20.08% | 30/256 |
| fold_0_valid | 8701 | 0.1739 | 0.3513 | 0.2326 | 0.1496 | 16.39% | 42/256 |
| fold_1_valid | 8701 | 0.1518 | 0.4164 | 0.2225 | 0.1644 | 22.26% | 35/256 |

Line metrics use original visible Diff labels. Vulnerable/patched breakdowns are in localization.csv.

| Prediction source | Selected-token precision | Selected-token recall | Diff mass | Background mass | Selected background mass |
|---|---:|---:|---:|---:|---:|
| in_sample | 0.4076 | 0.9948 | 0.2592 | 0.7408 | 0.4022 |
| oof | 0.1527 | 0.3200 | 0.1242 | 0.8758 | 0.5961 |
| full_valid | 0.1836 | 0.4206 | 0.1955 | 0.8045 | 0.6151 |
| fold_0_valid | 0.2133 | 0.3542 | 0.2080 | 0.7920 | 0.5409 |
| fold_1_valid | 0.1923 | 0.4516 | 0.1843 | 0.8157 | 0.6186 |

Token P/R are micro counts over valid code tokens; mass is averaged per function using the actual exp(5)/exp(1) weights. Functions with no visible Diff have Diff mass zero; visible-positive-only means and equal-pooling reference are also saved. Diff agreement is a supervision diagnostic, not proof of vulnerability-semantic correctness.

## Detector Valid

| Train predictions | Epoch | P-C | AUC | MCC | F1 | Vulnerable recall | Patched FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| in_sample | 4 | 13/128 (10.16%) | 0.5424 | 0.0825 | 0.3704 | 0.2734 | 0.2031 |
| oof | 4 | 9/128 (7.03%) | 0.5348 | 0.0262 | 0.3719 | 0.2891 | 0.2656 |

OOF minus in-sample: P-C -0.0312, AUC -0.0076, MCC -0.0563, F1 +0.0015.

Strict-pair gains: 4; losses: 8. All 128 pairs and their probability changes are saved in valid.pair_changes.csv; the full state transition table is in comparison.json.

Mean probability change: vulnerable +0.0120, patched +0.0141.

| In-sample → OOF | P-C | P-V | P-B | P-R |
|---|---:|---:|---:|---:|
| p_c | 5 | 4 | 4 | 0 |
| p_v | 0 | 14 | 8 | 0 |
| p_b | 4 | 9 | 72 | 4 |
| p_r | 0 | 1 | 1 | 2 |

## Observations

OOF vs in-sample Train AP: 0.1288 vs 0.8881; the common Valid AP is 0.1506. Absolute pooled selected-line-rate gap to Valid is 0.0323 in-sample and 0.0447 OOF. Ranking/label agreement and selection rate therefore need separate interpretation.

Function-level Wasserstein distances to the same full-Valid predictions:

| Quantity | In-sample → Valid | OOF → Valid |
|---|---:|---:|
| selected_line_fraction | 0.0529 | 0.0491 |
| true_diff_mass | 0.0833 | 0.0714 |
| selected_background_mass | 0.2129 | 0.0217 |

The distributions are closer on these three per-function statistics, especially selected-background mass; the pooled selected-line-rate gap still grows. Reduced distribution distances do not establish correct localization or classification benefit.

OOF does not improve either selected Valid P-C or AUC in this Pilot. Any higher recall/F1 must be interpreted alongside patched-side FPR; there is no classification benefit supporting a full run.

## Budget and interpretation

The full localizer trained 4 epochs on 1,024 functions; each fold localizer trains 8 epochs on 512 functions. Each runs 512 optimizer steps with the same LR/warmup schedule, but the fold models see half as many unique functions twice as often. Fold checkpoint selection uses only the same 256-function source Valid, never held-out Train. Eight versus four validation opportunities and potentially different selected steps remain comparison limitations.

- Fold 0: selected epoch 4, step 256/512; held-out functions 512.
- Fold 1: selected epoch 3, step 192/512; held-out functions 512.

Cross-fold pair overlap: 0. Exact-source overlap across different pairs: 0 groups / 0 functions; no member was removed or resampled.

Fold-specific Valid rows above diagnose half-data model quality only; neither detector uses those Valid predictions. Both use full_valid. Function-level distribution distances are in distribution_shift.csv; closer values on one statistic do not establish that every aspect of the shift improved.

On this common Valid, the fold models are not consistently worse than the full model in F1/AP. The much lower OOF Train quality than in-sample quality alone cannot distinguish missing in-sample exposure from half-data training effects.

Training/validation loop times are in costs.csv. Jobs ran concurrently on two RTX 4090 GPUs; these times exclude initial loading/feature extraction and final checkpoint/export I/O, and are not a controlled speed comparison. Real GPU cache and source-only single-function checks are saved in gpu_checks.json.

OOF changes both training membership exposure and the effective quality of localization. A weaker OOF localizer cannot isolate the effect of removing in-sample exposure alone. This is one fixed-seed Pilot, not evidence of stable population-level improvement. No extra method, full-data training or Test evaluation was launched.
