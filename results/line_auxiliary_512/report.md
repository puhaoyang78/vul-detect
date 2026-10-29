# Function classification with original-Diff line auxiliary supervision

Full official PrimeVul Original Train=4,354 pairs/8,708 functions and Valid=562 pairs/1,124 functions. All five runs use 512 tokens, seed123456, batch8, 16 epochs/17,424 optimizer steps, LR2e-5, 10% warmup and unchanged LOSVERAdamW (weight decay0). Threshold is strictly >.5. Test is never read.

All runs share the original UniXcoder and function linear-head initialization. A single additional Linear(hidden_size,1), initialized without advancing the sampler RNG, is present in all five controls. Each function's visible code-line states are means from the SAME encoder output. Original Focal (alpha=.8, gamma=1.5, pos_weight=5) is averaged over visible lines WITHIN each function, then over functions. Empty Diff is all-zero line supervision and never drops function BCE.

Independent `random.Random(123456)` permutes each Train function's visible line labels once, before training. Counts are preserved and the same permutations are used at both weights. Valid line AP pools all visible lines against real Diff, whereas Focal is function-balanced. Only function-level Valid P-C selects the epoch and lambda; epoch ties keep the earlier checkpoint, lambda ties prefer .1. Line AP and AUC do not select anything.

Inference runs only the original global code-token mean and original function linear head. Diff and line logits do not affect pooling, token weights or the function prediction. The lambda0 line head is untrained: its AP is a random-projection diagnostic, not a learned localizer baseline.

## Valid results

| Configuration | Selected epoch | Correct pairs /562 | P-C | AUC | MCC | F1 | True-Diff line AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 14 | 87/562 | 0.1548 | 0.5791 | 0.1152 | 0.5212 | 0.0526 |
| diff_0.1 | 16 | 81/562 | 0.1441 | 0.5698 | 0.1042 | 0.5807 | 0.1991 |
| diff_0.3 | 10 | 80/562 | 0.1423 | 0.5824 | 0.1014 | 0.5471 | 0.2294 |
| random_0.1 | 12 | 90/562 | 0.1601 | 0.5783 | 0.1072 | 0.5335 | 0.1261 |
| random_0.3 | 15 | 75/562 | 0.1335 | 0.5805 | 0.1107 | 0.5726 | 0.1267 |

## Selection and interpretation

diff: Valid P-C selects lambda=0.1; P-C difference from baseline=-0.010676, AUC difference=-0.009270, MCC difference=-0.010961.

random: Valid P-C selects lambda=0.1; P-C difference from baseline=+0.005338, AUC difference=-0.000749, MCC difference=-0.008020.

At matched lambda=0.1, real Diff minus random: P-C=-0.016014, AUC=-0.008522, MCC=-0.002942, F1=+0.047242.

At matched lambda=0.3, real Diff minus random: P-C=+0.008897, AUC=+0.001824, MCC=-0.009255, F1=-0.025564.

The predefined real-Diff auxiliary grid does not improve the primary Valid P-C over baseline. The other metrics must be read separately; no further module or Test experiment is added.

diff_0.1: F1=0.580699 versus baseline 0.521154; vulnerable Recall=0.620996 versus 0.482206; patched-side FPR=0.517794 versus 0.368327. A higher F1 does not by itself imply better pair discrimination or a lower false-positive rate.

diff_0.3: F1=0.547085 versus baseline 0.521154; vulnerable Recall=0.542705 versus 0.482206; patched-side FPR=0.441281 versus 0.368327. A higher F1 does not by itself imply better pair discrimination or a lower false-positive rate.

Random permutations changed 5084/8708 functions' visible labels. Unchanged all-zero/all-one or coincidentally identical permutations are retained. Line denominators are identical across the five runs and are saved in comparison.csv. Increasing line AP alone does not establish a classification gain; compare real Diff against the count-matched random control at BOTH preset weights. This is one training seed and one predefined lambda grid, with no Test selection and no additional method.

[Training losses](training.csv) contain both unweighted BCE and function-balanced Focal, total objective and Valid metrics for all 80 epochs. Each run retains configuration, best-P-C predictions (function probabilities plus physical-line scores), logs and resumable local checkpoints. [GPU smoke](gpu-smoke.json), [tests](tests.log) and [resume checks](resume-tests.log) record the actual correctness checks; [protocol checks](protocol-checks.json) inspect the completed runs. Reproduction commands are in the [README](../../README.md).
