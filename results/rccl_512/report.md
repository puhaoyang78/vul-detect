# RCCL: matched Context + Diff experiments

Official paired Train 8,708 functions / 4,354 pairs; Valid 1,124 / 562. Seed123456, Batch8, 512 tokens, 16 epochs / 17,424 optimizer steps. Fresh pretrained UniXcoder initialization for all three runs. Historical Context + Diff checkpoint retained.

BCE + 0.1 Diff + 0.05 RCCL; temperature 0.1. Projection: bias-free 768→32 linear (24,576 parameters); two independent p=0.1 representation Dropout views per version. Four symmetric anchors per aligned line, with same-version positive and two opposite-version negatives. Mean lines within pair, then eligible pairs. Other lines are never contrastive negatives. A private CPU generator preserves encoder and sampler RNG.

Exact Diff equal lines must be unique in both complete functions and fully visible in both versions. Select the nearest nonblank equal line on each side of every edit block, skipping ambiguous/partial neighbors. Pairs with identical 512-token inputs are excluded from RCCL. Random selects the same number of reliable equal lines in the same pair without replacement. All functions retain BCE and original Diff labels.

Actual pair order is recorded for all 16 epochs and identical across the three runs; optimizer budget, preprocessing and Diff targets match. Full Valid P-C (rounded to four decimals; earlier epoch on ties) selects checkpoints; decisions use probability >0.5. Test, when present below, was evaluated after selection and never used for tuning.

| Method | Split | Cohort | Epoch | Pairs | P-C | AUC | MCC | F1 | Recall | Patched FPR | Pair ranking |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matched_context_diff | valid | all | 13 | 562 | 0.1708 | 0.5963 | 0.1320 | 0.5800 | 0.5996 | 0.4680 | 0.4733 |
| matched_context_diff | valid | pure_addition | 13 | 163 | 0.2209 | 0.6323 | 0.1841 | 0.5933 | 0.5951 | 0.4110 | 0.5460 |
| rccl_diff | valid | all | 12 | 562 | 0.1673 | 0.5800 | 0.1157 | 0.5931 | 0.6459 | 0.5320 | 0.4680 |
| rccl_diff | valid | pure_addition | 12 | 163 | 0.2270 | 0.6326 | 0.1845 | 0.6053 | 0.6258 | 0.4417 | 0.5215 |
| rccl_random | valid | all | 8 | 562 | 0.1744 | 0.5841 | 0.1216 | 0.5806 | 0.6085 | 0.4875 | 0.4609 |
| rccl_random | valid | pure_addition | 8 | 163 | 0.2209 | 0.6240 | 0.1472 | 0.5723 | 0.5706 | 0.4233 | 0.5276 |
| matched_context_diff | test | all | 13 | 564 | 0.1401 | 0.5713 | 0.0943 | 0.5644 | 0.5869 | 0.4929 | 0.4450 |
| matched_context_diff | test | pure_addition | 13 | 189 | 0.1640 | 0.6104 | 0.1274 | 0.5802 | 0.6032 | 0.4762 | 0.4550 |
| rccl_diff | test | all | 12 | 564 | 0.1436 | 0.5690 | 0.0958 | 0.5876 | 0.6454 | 0.5514 | 0.4326 |
| rccl_diff | test | pure_addition | 12 | 189 | 0.1905 | 0.5935 | 0.1438 | 0.5950 | 0.6296 | 0.4868 | 0.4656 |
| rccl_random | test | all | 8 | 564 | 0.1294 | 0.5696 | 0.0924 | 0.5594 | 0.5762 | 0.4840 | 0.4539 |
| rccl_random | test | pure_addition | 8 | 189 | 0.1587 | 0.5914 | 0.1111 | 0.5602 | 0.5661 | 0.4550 | 0.4286 |

## Coverage

- train all: 3017/4354 pairs (69.29%), 8341 aligned neighbor lines.
- train pure_addition: 823/1232 pairs (66.80%), 1716 aligned neighbor lines.
- valid all: 381/562 pairs (67.79%), 1018 aligned neighbor lines.
- valid pure_addition: 119/163 pairs (73.01%), 248 aligned neighbor lines.

## Training and representation diagnostics

Cosines below are raw shared-encoder line states from each Valid-selected checkpoint in eval mode on the same eligible true-neighbor cohort. They are pair-averaged; line-weighted values, standard deviations, fixed-random/other-line cohorts and projected cosines are in comparison.json. Projected cosines measure the training head only and do not affect inference.

| Method | Train min | RCCL loss epoch 1 → 16 | Train raw cosine | Valid raw cosine | Valid projected cosine |
|---|---:|---|---:|---:|---:|
| pretrained initialization | — | — | 0.9599 | 0.9617 | 0.9629 |
| matched_context_diff | 65.76 | — | 0.3016 | 0.8529 | — |
| rccl_diff | 47.13 | 0.3789 → 0.0132 | 0.3247 | 0.7638 | 0.6184 |
| rccl_random | 70.06 | 0.3809 → 0.0103 | 0.5566 | 0.7805 | 0.6306 |

RCCL losses are epoch means of the per-effective-batch eligible-pair mean (zero for an empty batch). Training durations include independent-process GPU sharing: GPU0 baseline/random, GPU1 true RCCL. The compute/update budget is identical.
Random control train: 1780/8341 selected lines also belong to the true neighbor set (21.34%); 80/3017 pairs select the identical set. Overlap is allowed by uniform sampling, never filtered using outcomes.
Random control valid: 234/1018 selected lines also belong to the true neighbor set (22.99%); 12/381 pairs select the identical set. Overlap is allowed by uniform sampling, never filtered using outcomes.

Validation: 85 distinct related unit tests passed. Actual NVIDIA GeForce RTX 4090 smoke passed initialization/RNG equality, single encoding, gradients and parameter update, padding, checkpoint restore and source-only inference. Inference omits the contrastive head; single/batched maximum probability difference 1.34e-07. Check scripts' corrected API/metadata errors are recorded in tests.json.

Historical Context + Diff (independent function shuffle, not the matched paired-batch control): Valid P-C 0.1779, selected epoch 14. Its checkpoint size and modification time are unchanged.

True RCCL fails to exceed both random RCCL and the matched Context + Diff baseline on complete Valid P-C. The fixed experiment is negative; no parameter search or additional variant was run.

The true-RCCL projection's near-line cosine is -0.1668 on Train versus 0.6184 on Valid. This large gap accompanies a low training contrastive loss; fitting the contrastive objective does not establish classification benefit.

Valid raw near-line cosine is 0.7638 with true RCCL versus 0.8529 for the matched baseline. Valid AUC changes from 0.5963 to 0.5800, MCC from 0.1320 to 0.1157, and patched FPR from 0.4680 to 0.5320. Increased context separation has to be assessed alongside these detection tradeoffs.

Post-selection Test P-C: baseline 79/564, true RCCL 81/564, random RCCL 73/564. True RCCL's AUC is 0.5690 versus baseline 0.5713, patched FPR 0.5514 versus 0.4929, and pair ranking 0.4326 versus 0.4450. Report these mixed Test results without changing the Valid-based conclusion or selecting a different checkpoint.

The pure-addition subgroup has a small positive P-C signal: Valid true/baseline/random 37/36/36 correct pairs out of 163; Test 36/31/30 out of 189. Pure-addition Test AUC is 0.5935 for true RCCL versus 0.6104 for the baseline. Preserve these subgroup gains while the primary full-Valid criterion remains unmet.
