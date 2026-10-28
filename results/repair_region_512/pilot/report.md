# Repair-region supervision and robust global-local fusion (512)

Status: fixed Train/Valid pilot; Test not used. Pending training: []; pending formal Test: not part of pilot.

All functions retain binary labels 0=safe / 1=vulnerable. Line labels are 1=repair-related, 0=weak structural background, -1=Unknown. Unknown has zero focal weight and stays in encoding/BCE. Functions with no visible positive region have no line loss, but remain in detector training and inference.

Both weak localizers use the same masked focal loss (alpha=.5, gamma=1.5, pos_weight=1). Direct statement weight=1, bounded bound/control relation=.6, background=.2, Unknown=0. Heuristic/nearest definitions and unproven neighbours are Unknown. Original LOSVER uses its historical focal definition and is an additional reference, not the matched weak-loss control.

Robust fusion: s=clamp((p-.5)/.5,0,1) on valid code tokens; h_region=sum(s*h)/sum(s); h=h_global+beta*max(s)*(h_region-h_global). With no positive mass, h=h_global exactly. Focal sigmoid outputs are not assumed calibrated. Train beta=.25; each epoch selects inference beta from {0,.1,.25} on source Valid P-C, ties prefer smaller beta, then earlier epoch. No Test choice.

Same UniXcoder pretrained initialization, 512 token offsets/code mask, Linear(768,1), BCE and optimizer budget across detector controls. No GT regions are used in detector training. No OOF, pair loss, extra encoder or adaptive network.

The localizer stage optimizes only its line objective; the detector stage retains all original functions and optimizes function BCE. The reported syntactic relations are repair-related candidates, not proven vulnerability locations or a complete data-flow analysis. The existing corrected 512 global/LOSVER results are full-data references; the pilot trains its references on the same subset and budget.

| Model | Split | Epoch | Beta | P-C | AUC | MCC | F1 | Recall | Patched FPR | Pair ranking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_mean | valid | 2 | None | 0.0625 | 0.5409 | 0.1222 | 0.6721 | 0.9609 | 0.8984 | 0.5078 |
| original_losver | valid | 4 | None | 0.1016 | 0.5424 | 0.0825 | 0.3704 | 0.2734 | 0.2031 | 0.4922 |
| diff_hard_weight | valid | 4 | None | 0.0312 | 0.5390 | 0.0370 | 0.2038 | 0.1250 | 0.1016 | 0.5156 |
| repair_region_hard_weight | valid | 4 | None | 0.0312 | 0.5375 | 0.0370 | 0.2038 | 0.1250 | 0.1016 | 0.5156 |
| diff_robust_fusion | valid | 2 | 0.0 | 0.0469 | 0.5419 | 0.0359 | 0.6459 | 0.8906 | 0.8672 | 0.5078 |
| repair_region_robust_fusion | valid | 2 | 0.0 | 0.0469 | 0.5420 | 0.0359 | 0.6459 | 0.8906 | 0.8672 | 0.5078 |
| neighbour_region_robust_fusion | valid | 2 | 0.0 | 0.0469 | 0.5415 | 0.0382 | 0.6499 | 0.9062 | 0.8828 | 0.5078 |

## Classification evidence

- valid: full method versus global_mean: delta P-C=-0.0156, AUC=+0.0012, MCC=-0.0863, F1=-0.0262.
- valid: full method versus original_losver: delta P-C=-0.0547, AUC=-0.0004, MCC=-0.0466, F1=+0.2755.
- valid: full method versus diff_robust_fusion: delta P-C=+0.0000, AUC=+0.0001, MCC=+0.0000, F1=+0.0000.
- valid: full method versus repair_region_hard_weight: delta P-C=+0.0156, AUC=+0.0045, MCC=-0.0011, F1=+0.4421.
- valid: the table shows metric-specific differences from one seed; it does not establish stable or causal classification gains.

Beta=0 disables the local branch at inference; this is the global path of a model trained with beta=.25, not the independently trained global baseline. [fusion_usage.json](fusion_usage.json) records score-gate fallback and effective beta. No module is added in response to these results.

The neighbour-region pilot control matches added label counts and confidence weights per full-source function. Truncation can change its visible positive count and eligible line-loss denominator, so it is not an exact visible-area control and small differences cannot be attributed solely to program relations.

## Supervision and train–Valid mismatch

| Localizer | Split | Functions | Unknown | Positive | Background | No positive function | Evaluated lines | Predicted lines | Weak-label F1 | Diff reference F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diff | train | 1024 | 75.81% | 6.23% | 17.96% | 40.33% | 5874 | 80.61% | 0.9933 | 0.1441 |
| diff | valid | 256 | 73.98% | 7.05% | 18.97% | 35.55% | 1668 | 80.04% | 0.9223 | 0.1551 |
| repair_region | train | 1024 | 74.43% | 7.61% | 17.96% | 31.74% | 7122 | 80.62% | 0.9943 | 0.1443 |
| repair_region | valid | 256 | 72.73% | 8.30% | 18.97% | 29.69% | 1886 | 79.74% | 0.9369 | 0.1554 |
| neighbour_region | train | 1024 | 74.59% | 7.45% | 17.96% | 37.01% | 6571 | 81.72% | 0.9485 | 0.1389 |
| neighbour_region | valid | 256 | 72.77% | 8.25% | 18.97% | 33.98% | 1808 | 80.71% | 0.9127 | 0.1545 |

Train caches are in-sample predictions, Valid uses unseen functions. This mismatch remains; the experiment does not establish a causal effect from any association between localization and detection. Own weak-label metrics use different positive regions across supervision types and must not be compared as localization ground truth.

The negative labels here are weak structural background, not a representative set of unrelated executable statements. High supervised F1 can therefore reflect easy code-versus-structure separation; it is insufficient evidence of precise repair localization. A high score on an Unknown line cannot by itself be called a false positive. The broad predicted-region coverage below is evidence of weak selectivity, with semantic accuracy still unmeasured on most executable lines.

Repair-type metrics and denominators: [repair_types.json](repair_types.json). Aligned detector predictions, selected beta, epoch histories, optimizer steps and input/Unknown coverage are in each method directory. Historical checkpoints/results are referenced without modification.

- diff: 97.79% of Unknown-supervision visible lines receive score >.5. Unknown is a training-label mask, not an uncertain model output; this fraction does not measure an error rate without independent labels.
- repair_region: 97.69% of Unknown-supervision visible lines receive score >.5. Unknown is a training-label mask, not an uncertain model output; this fraction does not measure an error rate without independent labels.
- neighbour_region: 98.34% of Unknown-supervision visible lines receive score >.5. Unknown is a training-label mask, not an uncertain model output; this fraction does not measure an error rate without independent labels.

## Supervision extraction limits

- Full-source train, 8708 functions: direct positive lines=44828, bound-operation positives=1174, enclosing-control positives=6775, no positive function=1033; own parse-failure/direct-only functions=2102, correspondence-Unknown functions=106. These counts precede 512 truncation and are not pilot-subset counts. All functions remain in function BCE.
- Full-source valid, 1124 functions: direct positive lines=4956, bound-operation positives=149, enclosing-control positives=915, no positive function=155; own parse-failure/direct-only functions=236, correspondence-Unknown functions=28. These counts precede 512 truncation and are not pilot-subset counts. All functions remain in function BCE.

Most added positives are enclosing-control lines; the current conservative mapping does not establish full def-use or resource-state semantics. Unproved lexical definition/use paths stay Unknown. Strict parse failures and the 512 visibility boundary limit useful supervision.

The pre-existing Train development review in ../train_region_review.csv (or train_region_review.csv in the full root) contains 86 additional-line judgments: 13/55 reviewed related lines become positive, 40/55 stay Unknown and 2/55 weak background; 0/21 reviewed unrelated and 0/10 uncertain lines become positive. This reused development review is not independent precision/recall validation or vulnerability-location ground truth.

| 512 input | Functions | Truncated functions | Visible original Diff lines / all | Identical encoded pairs |
| --- | --- | --- | --- | --- |
| train | 1024 | 642 | 2685/7126 | 134 |
| valid | 256 | 148 | 706/1204 | 30 |

Input coverage is common to all compared methods; identical encoded pairs and invisible repair changes remain in function BCE and evaluation. Region labels cannot recover code outside the fixed 512 input. No member or input rule was changed to improve these counts.

## Addition-only repair pairs

| Model | Split | Pairs | P-C | AUC | Recall | Patched FPR |
| --- | --- | --- | --- | --- | --- | --- |
| global_mean | valid | 24 | 0.0833 | 0.5530 | 1.0000 | 0.9167 |
| original_losver | valid | 24 | 0.2083 | 0.5877 | 0.3333 | 0.2083 |
| diff_hard_weight | valid | 24 | 0.0417 | 0.5582 | 0.1250 | 0.0833 |
| repair_region_hard_weight | valid | 24 | 0.0417 | 0.5512 | 0.1250 | 0.0833 |
| diff_robust_fusion | valid | 24 | 0.0417 | 0.5530 | 0.9167 | 0.8750 |
| repair_region_robust_fusion | valid | 24 | 0.0417 | 0.5512 | 0.9167 | 0.8750 |
| neighbour_region_robust_fusion | valid | 24 | 0.0417 | 0.5530 | 0.9167 | 0.8750 |

Pure addition is defined from the original nonempty Diff deletion labels, before any new supervision. It includes checks, initialization, state/cleanup and comments; these are not all new checks. The group is descriptive and does not select hyperparameters.
