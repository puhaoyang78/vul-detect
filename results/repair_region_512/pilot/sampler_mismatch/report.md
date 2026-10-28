# Repair-region supervision and robust global-local fusion (512)

Status: fixed Train/Valid pilot; Test not used. Pending training: []; pending formal Test: not part of pilot.

All functions retain binary labels 0=safe / 1=vulnerable. Line labels are 1=repair-related, 0=weak structural background, -1=Unknown. Unknown has zero focal weight and stays in encoding/BCE. Functions with no visible positive region have no line loss, but remain in detector training and inference.

Both weak localizers use the same masked focal loss (alpha=.5, gamma=1.5, pos_weight=1). Direct statement weight=1, bounded bound/control relation=.6, background=.2, Unknown=0. Heuristic/nearest definitions and unproven neighbours are Unknown. Original LOSVER uses its historical focal definition and is an additional reference, not the matched weak-loss control.

Robust fusion: s=clamp((p-.5)/.5,0,1) on valid code tokens; h_region=sum(s*h)/sum(s); h=h_global+beta*max(s)*(h_region-h_global). With no positive mass, h=h_global exactly. Focal sigmoid outputs are not assumed calibrated. Train beta=.25; each epoch selects inference beta from {0,.1,.25} on source Valid P-C, ties prefer smaller beta, then earlier epoch. No Test choice.

Same UniXcoder pretrained initialization, 512 token offsets/code mask, Linear(768,1), BCE and optimizer budget across detector controls. No GT regions are used in detector training. No OOF, pair loss, extra encoder or adaptive network.

| Model | Split | Epoch | Beta | P-C | AUC | MCC | F1 | Recall | Patched FPR | Pair ranking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_mean | valid | 2 | None | 0.0625 | 0.5409 | 0.1222 | 0.6721 | 0.9609 | 0.8984 | 0.5078 |
| original_losver | valid | 4 | None | 0.1016 | 0.5424 | 0.0825 | 0.3704 | 0.2734 | 0.2031 | 0.4922 |
| diff_hard_weight | valid | 4 | None | 0.0312 | 0.5390 | 0.0370 | 0.2038 | 0.1250 | 0.1016 | 0.5156 |
| repair_region_hard_weight | valid | 4 | None | 0.0312 | 0.5375 | 0.0370 | 0.2038 | 0.1250 | 0.1016 | 0.5156 |
| diff_robust_fusion | valid | 4 | 0.0 | 0.0156 | 0.5407 | 0.0142 | 0.1477 | 0.0859 | 0.0781 | 0.4922 |
| repair_region_robust_fusion | valid | 4 | 0.0 | 0.0156 | 0.5403 | 0.0142 | 0.1477 | 0.0859 | 0.0781 | 0.4922 |
| neighbour_region_robust_fusion | valid | 4 | 0.0 | 0.0156 | 0.5398 | 0.0142 | 0.1477 | 0.0859 | 0.0781 | 0.4844 |

## Classification evidence

- valid: full method versus global_mean: delta P-C=-0.0469, AUC=-0.0006, MCC=-0.1080, F1=-0.5245.
- valid: full method versus original_losver: delta P-C=-0.0859, AUC=-0.0021, MCC=-0.0683, F1=-0.2227.
- valid: full method versus diff_robust_fusion: delta P-C=+0.0000, AUC=-0.0004, MCC=+0.0000, F1=+0.0000.
- valid: full method versus repair_region_hard_weight: delta P-C=-0.0156, AUC=+0.0027, MCC=-0.0227, F1=-0.0562.
- valid: no improvement over the global baseline on either P-C or AUC.

Beta=0 disables the local branch at inference; this is the global path of a model trained with beta=.25, not the independently trained global baseline. [fusion_usage.json](fusion_usage.json) records score-gate fallback and effective beta. No module is added in response to these results.

## Supervision and train–Valid mismatch

| Localizer | Split | Functions | Unknown | Positive | Background | No positive function | Predicted lines | Weak-label F1 | Diff reference F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diff | train | 1024 | 75.81% | 6.23% | 17.96% | 40.33% | 80.61% | 0.9933 | 0.1441 |
| diff | valid | 256 | 73.98% | 7.05% | 18.97% | 35.55% | 80.04% | 0.9223 | 0.1551 |
| repair_region | train | 1024 | 74.43% | 7.61% | 17.96% | 31.74% | 80.62% | 0.9943 | 0.1443 |
| repair_region | valid | 256 | 72.73% | 8.30% | 18.97% | 29.69% | 79.74% | 0.9369 | 0.1554 |
| neighbour_region | train | 1024 | 74.59% | 7.45% | 17.96% | 37.01% | 81.72% | 0.9485 | 0.1389 |
| neighbour_region | valid | 256 | 72.77% | 8.25% | 18.97% | 33.98% | 80.71% | 0.9127 | 0.1545 |

Train caches are in-sample predictions, Valid uses unseen functions. This mismatch remains; the experiment does not establish a causal effect from any association between localization and detection. Own weak-label metrics use different positive regions across supervision types and must not be compared as localization ground truth.

Repair-type metrics and denominators: [repair_types.json](repair_types.json). Aligned detector predictions, selected beta, epoch histories, optimizer steps and input/Unknown coverage are in each method directory. Historical checkpoints/results are referenced without modification.

## Addition-only repair pairs

| Model | Split | Pairs | P-C | AUC | Recall | Patched FPR |
| --- | --- | --- | --- | --- | --- | --- |
| global_mean | valid | 24 | 0.0833 | 0.5530 | 1.0000 | 0.9167 |
| original_losver | valid | 24 | 0.2083 | 0.5877 | 0.3333 | 0.2083 |
| diff_hard_weight | valid | 24 | 0.0417 | 0.5582 | 0.1250 | 0.0833 |
| repair_region_hard_weight | valid | 24 | 0.0417 | 0.5512 | 0.1250 | 0.0833 |
| diff_robust_fusion | valid | 24 | 0.0000 | 0.5564 | 0.0417 | 0.0417 |
| repair_region_robust_fusion | valid | 24 | 0.0000 | 0.5564 | 0.0417 | 0.0417 |
| neighbour_region_robust_fusion | valid | 24 | 0.0000 | 0.5530 | 0.0417 | 0.0417 |

Pure addition is defined from the original nonempty Diff deletion labels, before any new supervision. It includes checks, initialization, state/cleanup and comments; these are not all new checks. The group is descriptive and does not select hyperparameters.
