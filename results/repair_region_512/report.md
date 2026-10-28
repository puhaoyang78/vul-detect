# Repair-region supervision and robust global-local fusion (512)

Status: full-data runs; formal Test is reported only where completed. Pending training: []; pending formal Test: [].

All functions retain binary labels 0=safe / 1=vulnerable. Line labels are 1=repair-related, 0=weak structural background, -1=Unknown. Unknown has zero focal weight and stays in encoding/BCE. Functions with no visible positive region have no line loss, but remain in detector training and inference.

Both weak localizers use the same masked focal loss (alpha=.5, gamma=1.5, pos_weight=1). Direct statement weight=1, bounded bound/control relation=.6, background=.2, Unknown=0. Heuristic/nearest definitions and unproven neighbours are Unknown. Original LOSVER uses its historical focal definition and is an additional reference, not the matched weak-loss control.

Robust fusion: s=clamp((p-.5)/.5,0,1) on valid code tokens; h_region=sum(s*h)/sum(s); h=h_global+beta*max(s)*(h_region-h_global). With no positive mass, h=h_global exactly. Focal sigmoid outputs are not assumed calibrated. Train beta=.25; each epoch selects inference beta from {0,.1,.25} on source Valid P-C, ties prefer smaller beta, then earlier epoch. No Test choice.

Same UniXcoder pretrained initialization, 512 token offsets/code mask, Linear(768,1), BCE and optimizer budget across detector controls. No GT regions are used in detector training. No OOF, pair loss, extra encoder or adaptive network.

The localizer stage optimizes only its line objective; the detector stage retains all original functions and optimizes function BCE. The reported syntactic relations are repair-related candidates, not proven vulnerability locations or a complete data-flow analysis. The existing corrected 512 global/LOSVER results are full-data references; the pilot trains its references on the same subset and budget.

Sampling/RNG checks are in [protocol-checks.json](protocol-checks.json). The matched pilot compares saved RNG states against its freshly trained global baseline. Historical full-data references retain selected weights and recorded configuration, but no last optimizer/RNG checkpoint; an exact historical RNG-state comparison is unavailable. Full new detectors are compared against one another without retraining the historical baseline.

| Model | Split | Epoch | Beta | P-C | AUC | MCC | F1 | Recall | Patched FPR | Pair ranking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_mean | valid | 14 | None | 0.1619 | 0.5862 | 0.1256 | 0.5893 | 0.6281 | 0.5036 | 0.4537 |
| global_mean | test | 14 | None | 0.1454 | 0.5729 | 0.1125 | 0.5815 | 0.6170 | 0.5053 | 0.4645 |
| original_losver | valid | 14 | None | 0.1423 | 0.5637 | 0.0874 | 0.5272 | 0.5089 | 0.4217 | 0.4110 |
| original_losver | test | 14 | None | 0.1401 | 0.5679 | 0.0889 | 0.5267 | 0.5071 | 0.4184 | 0.4184 |
| diff_hard_weight | valid | 11 | None | 0.1459 | 0.5770 | 0.1246 | 0.4849 | 0.4146 | 0.2954 | 0.4359 |
| diff_hard_weight | test | 11 | None | 0.1099 | 0.5666 | 0.0826 | 0.4480 | 0.3741 | 0.2961 | 0.4468 |
| repair_region_hard_weight | valid | 15 | None | 0.1637 | 0.5742 | 0.1050 | 0.5481 | 0.5427 | 0.4377 | 0.4466 |
| repair_region_hard_weight | test | 15 | None | 0.1188 | 0.5527 | 0.0656 | 0.5357 | 0.5390 | 0.4734 | 0.4326 |
| diff_robust_fusion | valid | 12 | 0.1 | 0.1584 | 0.5791 | 0.1122 | 0.5443 | 0.5302 | 0.4181 | 0.4448 |
| diff_robust_fusion | test | 12 | 0.1 | 0.1330 | 0.5645 | 0.0922 | 0.5477 | 0.5496 | 0.4574 | 0.4681 |
| repair_region_robust_fusion | valid | 11 | 0.25 | 0.1512 | 0.5814 | 0.1050 | 0.5473 | 0.5409 | 0.4359 | 0.4448 |
| repair_region_robust_fusion | test | 11 | 0.25 | 0.1259 | 0.5672 | 0.0946 | 0.5184 | 0.4876 | 0.3936 | 0.4645 |

## Classification evidence

- valid: full method versus global_mean: delta P-C=-0.0107, AUC=-0.0048, MCC=-0.0206, F1=-0.0421.
- valid: full method versus original_losver: delta P-C=+0.0089, AUC=+0.0177, MCC=+0.0176, F1=+0.0201.
- valid: full method versus diff_robust_fusion: delta P-C=-0.0071, AUC=+0.0023, MCC=-0.0072, F1=+0.0030.
- valid: full method versus repair_region_hard_weight: delta P-C=-0.0125, AUC=+0.0071, MCC=+0.0000, F1=-0.0008.
- valid: no improvement over the global baseline on either P-C or AUC.
- valid: strict correct pairs=85/562 versus global 91/562; delta vulnerable Recall=-0.0872, patched FPR=-0.0676. Pair ranking correct=250, ties=164; global ranking correct=255, ties=164. These are observed decision/ranking changes, not causal attribution.
- valid: source Valid selected beta=0.25; the full-data result is not explained by beta=0 disabling the local branch. A positive coefficient does not establish a large or useful representation residual.
- test: full method versus global_mean: delta P-C=-0.0195, AUC=-0.0057, MCC=-0.0179, F1=-0.0631.
- test: full method versus original_losver: delta P-C=-0.0142, AUC=-0.0006, MCC=+0.0057, F1=-0.0083.
- test: full method versus diff_robust_fusion: delta P-C=-0.0071, AUC=+0.0027, MCC=+0.0024, F1=-0.0293.
- test: full method versus repair_region_hard_weight: delta P-C=+0.0071, AUC=+0.0146, MCC=+0.0290, F1=-0.0173.
- test: no improvement over the global baseline on either P-C or AUC.
- test: strict correct pairs=71/564 versus global 82/564; delta vulnerable Recall=-0.1294, patched FPR=-0.1117. Pair ranking correct=262, ties=163; global ranking correct=262, ties=163. These are observed decision/ranking changes, not causal attribution.
- test: source Valid selected beta=0.25; the full-data result is not explained by beta=0 disabling the local branch. A positive coefficient does not establish a large or useful representation residual.

Beta=0 disables the local branch at inference; this is the global path of a model trained with beta=.25, not the independently trained global baseline. [fusion_usage.json](fusion_usage.json) records score-gate fallback and effective beta. No module is added in response to these results.

The neighbour-region pilot control matches added label counts and confidence weights per full-source function. Truncation can change its visible positive count and eligible line-loss denominator, so it is not an exact visible-area control and small differences cannot be attributed solely to program relations.

## Supervision and train–Valid mismatch

| Localizer | Split | Functions | Unknown | Positive | Background | No positive function | Evaluated lines | Predicted lines | Weak-label F1 | Diff reference F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diff | train | 8708 | 75.91% | 6.33% | 17.76% | 42.20% | 49770 | 80.48% | 0.9831 | 0.1413 |
| diff | valid | 1124 | 75.03% | 6.12% | 18.85% | 44.40% | 6254 | 79.09% | 0.9631 | 0.1382 |
| repair_region | train | 8708 | 74.53% | 7.71% | 17.76% | 33.14% | 59879 | 80.29% | 0.9857 | 0.1415 |
| repair_region | valid | 1124 | 73.62% | 7.53% | 18.85% | 36.48% | 7425 | 79.11% | 0.9724 | 0.1382 |

Train caches are in-sample predictions, Valid uses unseen functions. This mismatch remains; the experiment does not establish a causal effect from any association between localization and detection. Own weak-label metrics use different positive regions across supervision types and must not be compared as localization ground truth.

The negative labels here are weak structural background, not a representative set of unrelated executable statements. High supervised F1 can therefore reflect easy code-versus-structure separation; it is insufficient evidence of precise repair localization. A high score on an Unknown line cannot by itself be called a false positive. The broad predicted-region coverage below is evidence of weak selectivity, with semantic accuracy still unmeasured on most executable lines.

This supports a supervision-selectivity explanation, but the experiment does not isolate its causal effect. The small Train/Valid gap on these weak labels also does not establish that in-sample localization is the primary cause of failure. No OOF or additional loss/background variant was run.

Repair-type metrics and denominators: [repair_types.json](repair_types.json). Aligned detector predictions, selected beta, epoch histories, optimizer steps and input/Unknown coverage are in each method directory. Historical checkpoints/results are referenced without modification.

- diff: 97.70% of Unknown-supervision visible lines receive score >.5. Unknown is a training-label mask, not an uncertain model output; this fraction does not measure an error rate without independent labels.
- repair_region: 97.74% of Unknown-supervision visible lines receive score >.5. Unknown is a training-label mask, not an uncertain model output; this fraction does not measure an error rate without independent labels.

## Supervision extraction limits

- Full-source train, 8708 functions: direct positive lines=44828, bound-operation positives=1174, enclosing-control positives=6775, no positive function=1033; own parse-failure/direct-only functions=2102, correspondence-Unknown functions=106. These counts precede 512 truncation and are not pilot-subset counts. All functions remain in function BCE.
- Full-source valid, 1124 functions: direct positive lines=4956, bound-operation positives=149, enclosing-control positives=915, no positive function=155; own parse-failure/direct-only functions=236, correspondence-Unknown functions=28. These counts precede 512 truncation and are not pilot-subset counts. All functions remain in function BCE.

Most added positives are enclosing-control lines; the current conservative mapping does not establish full def-use or resource-state semantics. Unproved lexical definition/use paths stay Unknown. Strict parse failures and the 512 visibility boundary limit useful supervision.

The pre-existing Train development review in ../train_region_review.csv (or train_region_review.csv in the full root) contains 86 additional-line judgments: 13/55 reviewed related lines become positive, 40/55 stay Unknown and 2/55 weak background; 0/21 reviewed unrelated and 0/10 uncertain lines become positive. This reused development review is not independent precision/recall validation or vulnerability-location ground truth.

| 512 input | Functions | Truncated functions | Visible original Diff lines / all | Identical encoded pairs |
| --- | --- | --- | --- | --- |
| train | 8708 | 5541 | 22627/49771 | 1254 |
| valid | 1124 | 671 | 2698/5401 | 164 |

Input coverage is common to all compared methods; identical encoded pairs and invisible repair changes remain in function BCE and evaluation. Region labels cannot recover code outside the fixed 512 input. This is a shared limitation and cannot alone explain differences between the methods. No member or input rule was changed to improve these counts.

## Addition-only repair pairs

| Model | Split | Pairs | P-C | AUC | Recall | Patched FPR |
| --- | --- | --- | --- | --- | --- | --- |
| global_mean | valid | 163 | 0.2025 | 0.6048 | 0.6196 | 0.4724 |
| global_mean | test | 189 | 0.2011 | 0.6128 | 0.6402 | 0.4603 |
| original_losver | valid | 163 | 0.1472 | 0.5848 | 0.4908 | 0.3988 |
| original_losver | test | 189 | 0.1905 | 0.5953 | 0.4921 | 0.3598 |
| diff_hard_weight | valid | 163 | 0.2209 | 0.6202 | 0.4724 | 0.2761 |
| diff_hard_weight | test | 189 | 0.1270 | 0.5847 | 0.4021 | 0.3069 |
| repair_region_hard_weight | valid | 163 | 0.2270 | 0.6055 | 0.5706 | 0.3988 |
| repair_region_hard_weight | test | 189 | 0.1587 | 0.5870 | 0.5556 | 0.4392 |
| diff_robust_fusion | valid | 163 | 0.2025 | 0.6080 | 0.5337 | 0.3620 |
| diff_robust_fusion | test | 189 | 0.1534 | 0.5974 | 0.5556 | 0.4286 |
| repair_region_robust_fusion | valid | 163 | 0.1902 | 0.6077 | 0.5460 | 0.4172 |
| repair_region_robust_fusion | test | 189 | 0.1534 | 0.5812 | 0.5132 | 0.3757 |

Pure addition is defined from the original nonempty Diff deletion labels, before any new supervision. It includes checks, initialization, state/cleanup and comments; these are not all new checks. The group is descriptive and does not select hyperparameters.

## Logged computational cost

| Stage | Completed epochs | Optimizer steps | Train seconds | Logged epoch seconds |
| --- | --- | --- | --- | --- |
| global_mean | 16 | 17424 | 2295.0 | 2395.3 |
| original_losver | 16 | 17424 | 2254.2 | 2352.4 |
| diff_hard_weight | 16 | 17424 | 2568.5 | 2676.9 |
| repair_region_hard_weight | 16 | 17424 | 2747.4 | 2854.8 |
| diff_robust_fusion | 16 | 17424 | 2592.7 | 2922.5 |
| repair_region_robust_fusion | 16 | 17424 | 2557.4 | 2883.9 |
| diff_localizer | 40 | 43560 | 7922.7 | 8298.6 |
| repair_region_localizer | 40 | 43560 | 8020.8 | 8405.0 |

Times are logged wall times, including loader/CPU/IO effects and concurrent GPU jobs, not isolated kernel benchmarks. Logged epoch time includes validation before end-of-epoch checkpoint serialization; startup, localization export and final evaluation are excluded. Robust methods use three Valid beta passes per epoch versus one in other detectors; training optimizer steps remain equal. Localizer cost is a separate stage shared by detectors using its scores.
