# 512 diff_neighbour pilot

Only line supervision changes. Original LOSVER localizer/Focal loss (alpha=.8, gamma=1.5, pos_weight=5), 5/1 code-token Softmax, UniXcoder, linear detector, source input and >.5 threshold are unchanged. All members remain in training and evaluation. No Unknown/weak-region loss or robust fusion is used.

Reference: `results/repair_region_512/pilot`. The existing completed original LOSVER pilot is reused without retraining or overwriting it. Both methods use 1024 Train/256 Valid functions, localizer/detector epochs=4/4, optimizer steps=512/512, training seed 123456 and member-sampling seed 42. The localizer selects by ORIGINAL Valid Diff positive F1; detector selects by Valid P-C. Test is not opened or evaluated.

A proxy is added only to a side whose FULL-source original Diff positives are empty. One code line is selected before/after each edit in the same Differ alignment. Each direction crosses at most one blank/comment line; brace-bearing, delimiter-only and preprocessor lines stop selection. The entire opposite nonblank edit and selected neighbours must be completely visible inside 510 source tokens. Identical encoded pairs are unchanged. These nearby lines are proxies, not verified vulnerability locations.

## Supervision coverage

| Scope | Split | Pure-add pairs | Supplemented | Coverage | Added positives | Single / both / none |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Full official | train | 1232 | 773 | 62.74% | 1377 | 303 / 470 / 459 |
| Full official | valid | 163 | 110 | 67.48% | 198 | 47 / 63 / 53 |
| Fixed pilot | train | 146 | 97 | 66.44% | 185 | 32 / 65 / 49 |
| Fixed pilot | valid | 24 | 17 | 70.83% | 32 | 5 / 12 / 7 |

Single/both denotes whether a supplemented function has at least one edit selecting both directions; per-edit counts and explicit rejection reasons are in `supervision/diff_neighbour/statistics.json` and the source-aligned audit tables. Multiple edits are deduplicated.

| Full official 512 input | Visible lines | Original positives | Added positives (both repair types) | New positives |
| --- | ---: | ---: | ---: | ---: |
| train | 318058 | 22627 | 1463 | 24090 |
| valid | 38745 | 2698 | 201 | 2899 |

## Valid function classification

| Model | Subset | Pairs | Epoch | P-C | AUC | MCC | F1 | Recall | Patched FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| losver | all | 128 | 4 | 0.1016 | 0.5424 | 0.0825 | 0.3704 | 0.2734 | 0.2031 |
| losver | pure_addition | 24 | 4 | 0.2083 | 0.5877 | 0.1406 | 0.4324 | 0.3333 | 0.2083 |
| diff_neighbour | all | 128 | 4 | 0.0938 | 0.5579 | 0.0808 | 0.3834 | 0.2891 | 0.2188 |
| diff_neighbour | pure_addition | 24 | 4 | 0.1667 | 0.6068 | 0.2236 | 0.3750 | 0.2500 | 0.0833 |

all, new minus original: P-C -0.0078, AUC +0.0155, MCC -0.0017, F1 +0.0130; newly both-correct pairs=8, regressed pairs=9, denominator=128.

pure_addition, new minus original: P-C -0.0417, AUC +0.0191, MCC +0.0830, F1 -0.0574; newly both-correct pairs=2, regressed pairs=3, denominator=24.

Pure addition is defined by the original vulnerable-side Diff labels being empty and patched-side positives being nonempty, before augmentation or model results.

## Line localization: original versus proxy labels

| Model | Split | Labels | Lines | Positive | Predicted | Precision | Recall | F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| losver | train | original_diff | 37815 | 2685 | 6371 | 0.4194 | 0.9952 | 0.5901 |
| losver | train | proxy_diff_neighbour | 37815 | 2888 | 6371 | 0.4260 | 0.9398 | 0.5862 |
| losver | valid | original_diff | 8701 | 706 | 1747 | 0.1603 | 0.3966 | 0.2283 |
| losver | valid | proxy_diff_neighbour | 8701 | 738 | 1747 | 0.1654 | 0.3916 | 0.2326 |
| diff_neighbour | train | original_diff | 37815 | 2685 | 7164 | 0.3728 | 0.9948 | 0.5424 |
| diff_neighbour | train | proxy_diff_neighbour | 37815 | 2888 | 7164 | 0.4003 | 0.9931 | 0.5706 |
| diff_neighbour | valid | original_diff | 8701 | 706 | 1985 | 0.1582 | 0.4448 | 0.2334 |
| diff_neighbour | valid | proxy_diff_neighbour | 8701 | 738 | 1985 | 0.1673 | 0.4499 | 0.2438 |

Original-Diff and proxy metrics use the SAME visible lines and predictions, with different label sets. Better proxy F1 alone cannot establish better localization or function classification. Full per-function code and labels stay in the local supervision cache; source-free audit records retain zero-based physical line IDs.

| Model | Split | Added proxy lines | Predicted hits | Proxy-only recall |
| --- | --- | ---: | ---: | ---: |
| losver | train | 203 | 42 | 0.2069 |
| losver | valid | 32 | 9 | 0.2812 |
| diff_neighbour | train | 203 | 197 | 0.9704 |
| diff_neighbour | valid | 32 | 18 | 0.5625 |

This measures recovery of the added adjacency proxies, not semantic vulnerability localization. Train is in-sample for the localizer; Valid is unseen. The original/proxy supervision gap and the classifier changes are observational, not a causal attribution.

## Runtime

| Stage | Train seconds | Logged epoch seconds | Steps |
| --- | ---: | ---: | ---: |
| losver localizer | 251.7 | 261.6 | 512 |
| losver detector | 242.7 | 251.3 | 512 |
| diff_neighbour localizer | 168.5 | 179.2 | 512 |
| diff_neighbour detector | 145.4 | 154.5 | 512 |

Times include loader/IO and validation effects; they are not isolated GPU benchmarks. These are one-seed pilot results; Test and full-data training are not performed.

## Implementation and verification

- `data.py`: preserve original Differ labels; supplement only fully visible edit-adjacent proxies and record each rejection. `training.py`/CLI: two new method choices in the existing stages; original Focal and selection retained. `evaluation.py`: aligned comparison, original/proxy line metrics and the same fixed source Valid cohort. `model.py` is unchanged.
- [Data checks](data-checks.json) verify original source/targets/members/order, all original positives and all non-target function labels. [Protocol checks](protocol-checks.json) compare actual saved optimizer/scheduler steps, parameter shapes, input coverage, members and final Python/NumPy/CPU/CUDA RNG states against the reused pilot.
- [Train review](train_review.md) / [records](train_review.jsonl): Seed42 stratified inspection of 40 Train pair sides, covering supplemented, boundary/visibility rejection, identical-input and pure-deletion cases. These syntactic checks do not certify semantic relevance. `train:5179` contains an added test invocation; `train:1831` selects a signature line; `train:2698` has a signature-only side. Such cases remain, as required by the unchanged data protocol.
- [GPU smoke](smoke.json): real Train forward/backward, encoder/attention/head gradients, exact save/reload and source-only single-function inference. [Full tests](tests.log), [targeted tests](neighbour-tests.log) and [resume tests](resume-tests.log) retain the actual execution logs. The smoke log preserves an initial diagnostic API-call error and its corrected successful check.

## Interpretation

AUC and P-C measure different behavior. Ranking can improve while two sides remain on the same side of the fixed .5 boundary. Read the pure-add Recall and patched FPR together: fewer false alarms can accompany fewer detected vulnerabilities. Proxy coverage or proxy F1 cannot substitute for both-correct classification.

The Train review confirms that bounded neighbours need not identify the vulnerability operation. Original-Diff localization precision remains low, and hard weighting is intentionally unchanged. These observations are possible limitations, not a demonstrated cause of a classifier difference. The reported metrics have mixed directions and do not establish a stable overall classification gain; no further module, full-data training or Test tuning is launched.
