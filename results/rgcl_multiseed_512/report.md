# RGCL fixed three-seed comparison

RGCL is the original Context + Diff: K=5, lambda=0.1, token-weighted region loss. CLS reuses the SecVulEval RobertaClassificationHead. Context-only differs from RGCL only by removing Diff loss. UniXcoder, 512 tokens, Batch8, 16 epochs, AdamW lr=2e-5, original function sampling; threshold strictly >0.5. PrimeVul selects full Valid P-C; SecVulEval selects full Valid MCC. Test never selects a checkpoint or configuration.

Seeds are 42, 1234 and 123456. Five completed original runs are reused; thirteen missing combinations are trained. Seed123457, historical Global Mean and the stopped Patch study are preserved and excluded from the three-seed mean. Historical result paths and completed optimization budgets are recorded in experiments.json. Standard deviations below use ddof=1. New resumable checkpoints are saved every 500 steps and at epoch boundaries; this changes disk I/O only, with the original optimization budget and RNG behavior. PrimeVul FPR uses the complete repaired side of official pairs; SecVulEval FPR uses all original normal functions, including ordinary and repaired functions. PR-AUC retains the existing average-precision (AP) definition.

| Dataset | Split | Seed | Method | Epoch | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| primevul_paired | valid | 42 | plm_baseline | 16 | 17.44% | 0.118068 | 0.580372 | 0.593237 | 0.585165 | 0.610320 | 0.492883 |
| primevul_paired | test | 42 | plm_baseline | 16 | 12.41% | 0.074812 | 0.558376 | 0.555100 | 0.553819 | 0.585106 | 0.510638 |
| primevul_paired | valid | 42 | context_only | 14 | 15.66% | 0.114408 | 0.577250 | 0.588396 | 0.573534 | 0.604982 | 0.491103 |
| primevul_paired | test | 42 | context_only | 14 | 13.65% | 0.101621 | 0.572873 | 0.568045 | 0.556916 | 0.602837 | 0.501773 |
| primevul_paired | valid | 42 | rgcl | 16 | 18.33% | 0.119044 | 0.592105 | 0.581556 | 0.573625 | 0.640569 | 0.523132 |
| primevul_paired | test | 42 | rgcl | 16 | 13.65% | 0.088549 | 0.583670 | 0.568611 | 0.565063 | 0.640071 | 0.553191 |
| primevul_paired | valid | 1234 | plm_baseline | 15 | 16.90% | 0.125012 | 0.580205 | 0.584433 | 0.579824 | 0.604982 | 0.480427 |
| primevul_paired | test | 1234 | plm_baseline | 15 | 12.94% | 0.075117 | 0.565724 | 0.560076 | 0.550027 | 0.602837 | 0.528369 |
| primevul_paired | valid | 1234 | context_only | 5 | 8.54% | 0.072352 | 0.604230 | 0.558787 | 0.547584 | 0.711744 | 0.644128 |
| primevul_paired | test | 1234 | context_only | 5 | 9.93% | 0.095091 | 0.614114 | 0.558596 | 0.554128 | 0.725177 | 0.636525 |
| primevul_paired | valid | 1234 | rgcl | 13 | 16.37% | 0.106080 | 0.598086 | 0.573302 | 0.571040 | 0.667260 | 0.564057 |
| primevul_paired | test | 1234 | rgcl | 13 | 14.72% | 0.109051 | 0.597444 | 0.571367 | 0.565368 | 0.663121 | 0.556738 |
| primevul_paired | valid | 123456 | plm_baseline | 15 | 14.41% | 0.106981 | 0.567241 | 0.579680 | 0.572675 | 0.585409 | 0.478648 |
| primevul_paired | test | 123456 | plm_baseline | 15 | 11.17% | 0.065620 | 0.527354 | 0.550070 | 0.552559 | 0.521277 | 0.455674 |
| primevul_paired | valid | 123456 | context_only | 16 | 15.48% | 0.098152 | 0.530990 | 0.575924 | 0.574062 | 0.510676 | 0.412811 |
| primevul_paired | test | 123456 | context_only | 16 | 12.23% | 0.092480 | 0.527675 | 0.569080 | 0.568154 | 0.507092 | 0.414894 |
| primevul_paired | valid | 123456 | rgcl | 14 | 17.79% | 0.143089 | 0.606998 | 0.589555 | 0.582306 | 0.663701 | 0.523132 |
| primevul_paired | test | 123456 | rgcl | 14 | 13.65% | 0.088486 | 0.582996 | 0.571188 | 0.564393 | 0.638298 | 0.551418 |
| secvuleval | valid | 42 | plm_baseline | 4 | — | 0.110898 | 0.567587 | 0.585923 | 0.507187 | 0.710000 | 0.603186 |
| secvuleval | test | 42 | plm_baseline | 4 | — | 0.136555 | 0.575635 | 0.602300 | 0.525491 | 0.710909 | 0.578255 |
| secvuleval | valid | 42 | context_only | 4 | — | 0.111417 | 0.565886 | 0.583368 | 0.506179 | 0.702727 | 0.594875 |
| secvuleval | test | 42 | context_only | 4 | — | 0.156026 | 0.595163 | 0.600513 | 0.511374 | 0.771818 | 0.626039 |
| secvuleval | valid | 42 | rgcl | 4 | — | 0.116963 | 0.592201 | 0.588595 | 0.512133 | 0.814545 | 0.713296 |
| secvuleval | test | 42 | rgcl | 4 | — | 0.158375 | 0.607637 | 0.600862 | 0.504238 | 0.839091 | 0.702909 |
| secvuleval | valid | 1234 | plm_baseline | 5 | — | 0.114022 | 0.353158 | 0.572133 | 0.502254 | 0.261818 | 0.168283 |
| secvuleval | test | 1234 | plm_baseline | 5 | — | 0.120903 | 0.382629 | 0.603469 | 0.518064 | 0.296364 | 0.192521 |
| secvuleval | valid | 1234 | context_only | 13 | — | 0.105140 | 0.473834 | 0.577561 | 0.493986 | 0.452727 | 0.349030 |
| secvuleval | test | 1234 | context_only | 13 | — | 0.127372 | 0.500689 | 0.593350 | 0.492086 | 0.495455 | 0.368421 |
| secvuleval | valid | 1234 | rgcl | 9 | — | 0.118560 | 0.492593 | 0.582440 | 0.501635 | 0.483636 | 0.365651 |
| secvuleval | test | 1234 | rgcl | 9 | — | 0.154480 | 0.534671 | 0.612189 | 0.518208 | 0.557273 | 0.401662 |
| secvuleval | valid | 123456 | plm_baseline | 10 | — | 0.114188 | 0.432946 | 0.570605 | 0.492045 | 0.372727 | 0.265928 |
| secvuleval | test | 123456 | plm_baseline | 10 | — | 0.087508 | 0.430548 | 0.584213 | 0.489970 | 0.381818 | 0.298476 |
| secvuleval | valid | 123456 | context_only | 4 | — | 0.101022 | 0.602432 | 0.570500 | 0.486454 | 0.900909 | 0.830332 |
| secvuleval | test | 123456 | context_only | 4 | — | 0.136319 | 0.610565 | 0.593291 | 0.502622 | 0.903636 | 0.804709 |
| secvuleval | valid | 123456 | rgcl | 4 | — | 0.126356 | 0.605853 | 0.574538 | 0.490590 | 0.884545 | 0.788781 |
| secvuleval | test | 123456 | rgcl | 4 | — | 0.167070 | 0.618340 | 0.593998 | 0.492920 | 0.907273 | 0.782548 |

| Dataset | Split | Method | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| primevul_paired | valid | plm_baseline | 16.25% ± 1.61pp | 0.116687 ± 0.009094 | 0.575939 ± 0.007533 | 0.585783 ± 0.006879 | 0.579221 ± 0.006267 | 0.600237 ± 0.013116 | 0.483986 ± 0.007756 |
| primevul_paired | valid | context_only | 13.23% ± 4.06pp | 0.094971 ± 0.021208 | 0.570823 ± 0.037040 | 0.574369 ± 0.014866 | 0.565060 ± 0.015137 | 0.609134 ± 0.100598 | 0.516014 ± 0.117653 |
| primevul_paired | valid | rgcl | 17.50% ± 1.01pp | 0.122737 ± 0.018779 | 0.599063 ± 0.007494 | 0.581471 ± 0.008127 | 0.575657 ± 0.005902 | 0.657177 ± 0.014492 | 0.536773 ± 0.023628 |
| primevul_paired | test | plm_baseline | 12.17% ± 0.91pp | 0.071850 ± 0.005397 | 0.550485 ± 0.020366 | 0.555082 ± 0.005003 | 0.552135 ± 0.001931 | 0.569740 ± 0.042897 | 0.498227 ± 0.037903 |
| primevul_paired | test | context_only | 11.94% ± 1.88pp | 0.096398 ± 0.004709 | 0.571554 ± 0.043235 | 0.565240 ± 0.005778 | 0.559733 ± 0.007425 | 0.611702 ± 0.109313 | 0.517730 ± 0.111674 |
| primevul_paired | test | rgcl | 14.01% ± 0.61pp | 0.095362 ± 0.011855 | 0.588037 ± 0.008154 | 0.570389 ± 0.001542 | 0.564941 ± 0.000499 | 0.647163 ± 0.013848 | 0.553783 ± 0.002708 |
| secvuleval | valid | plm_baseline | — | 0.113036 ± 0.001853 | 0.451230 ± 0.108378 | 0.576220 ± 0.008437 | 0.500495 ± 0.007723 | 0.448182 ± 0.233424 | 0.345799 ± 0.228188 |
| secvuleval | valid | context_only | — | 0.105860 ± 0.005234 | 0.547384 ± 0.066265 | 0.577143 ± 0.006444 | 0.495539 ± 0.009954 | 0.685455 ± 0.224590 | 0.591413 ± 0.240670 |
| secvuleval | valid | rgcl | — | 0.120626 ± 0.005026 | 0.563549 ± 0.061828 | 0.581858 ± 0.007047 | 0.501453 ± 0.010773 | 0.727576 ± 0.214137 | 0.622576 ± 0.225682 |
| secvuleval | test | plm_baseline | — | 0.114989 ± 0.025052 | 0.462937 ± 0.100497 | 0.596661 ± 0.010796 | 0.511175 ± 0.018736 | 0.463030 ± 0.218880 | 0.356417 ± 0.199288 |
| secvuleval | test | context_only | — | 0.139906 ± 0.014660 | 0.568806 ± 0.059491 | 0.595718 ± 0.004153 | 0.502027 ± 0.009658 | 0.723636 ± 0.208313 | 0.599723 ± 0.219331 |
| secvuleval | test | rgcl | — | 0.159975 ± 0.006446 | 0.586882 ± 0.045532 | 0.602350 ± 0.009186 | 0.505122 ± 0.012667 | 0.767879 ± 0.185549 | 0.629040 ± 0.200901 |

primevul_paired valid, plm_baseline → context_only: mean p_c -0.030249; improved in 1/3 seeds. AUC -0.011414; MCC -0.021716; FPR +0.032028.

primevul_paired valid, context_only → rgcl: mean p_c +0.042705; improved in 3/3 seeds. AUC +0.007102; MCC +0.027766; FPR +0.020759.

primevul_paired valid, plm_baseline → rgcl: mean p_c +0.012456; improved in 2/3 seeds. AUC -0.004312; MCC +0.006050; FPR +0.052788.

primevul_paired test, plm_baseline → context_only: mean p_c -0.002364; improved in 2/3 seeds. AUC +0.010158; MCC +0.024548; FPR +0.019504.

primevul_paired test, context_only → rgcl: mean p_c +0.020686; improved in 2/3 seeds. AUC +0.005148; MCC -0.001035; FPR +0.036052.

primevul_paired test, plm_baseline → rgcl: mean p_c +0.018322; improved in 3/3 seeds. AUC +0.015307; MCC +0.023512; FPR +0.055556.

secvuleval valid, plm_baseline → context_only: mean mcc -0.007176; improved in 1/3 seeds. AUC +0.000923; MCC -0.007176; FPR +0.245614.

secvuleval valid, context_only → rgcl: mean mcc +0.014766; improved in 3/3 seeds. AUC +0.004715; MCC +0.014766; FPR +0.031163.

secvuleval valid, plm_baseline → rgcl: mean mcc +0.007591; improved in 3/3 seeds. AUC +0.005637; MCC +0.007591; FPR +0.276777.

secvuleval test, plm_baseline → context_only: mean mcc +0.024917; improved in 3/3 seeds. AUC -0.000943; MCC +0.024917; FPR +0.243306.

secvuleval test, context_only → rgcl: mean mcc +0.020069; improved in 3/3 seeds. AUC +0.006632; MCC +0.020069; FPR +0.029317.

secvuleval test, plm_baseline → rgcl: mean mcc +0.044986; improved in 3/3 seeds. AUC +0.005689; MCC +0.044986; FPR +0.272622.


三种子结果支持保留原始 RGCL，并不支持“CLS → Context-only → RGCL”在所有指标上单调提升。PrimeVul Valid P-C 为 16.25±1.61%、13.23±4.06%、17.50±1.01%；Test 为 12.17±0.91%、11.94±1.88%、14.01±0.61%。RGCL 相对 Context-only 的 Test P-C 平均提高 2.07 个百分点（两个种子提高、一个持平），但 MCC 从 0.096398 略降至 0.095362，修复侧 FPR 从 51.77% 升至 55.38%。RGCL 的 Valid AUC 和 PR-AUC 也没有超过 CLS。

SecVulEval Test MCC 为 0.114989±0.025052、0.139906±0.014660、0.159975±0.006446。RGCL 相对 Context-only 在三个种子的 Valid/Test MCC 均提高，Test 平均提高 0.020069；同时 Recall 从 72.36% 升至 76.79%，FPR 从 59.97% 升至 62.90%，PR-AUC 仍低于 CLS。Recall 和 FPR 的种子间波动较大，不能将 MCC 收益解读为全面稳定改善。

本轮仅完成固定 RGCL 的正式三组对照，没有追加结构、阈值或参数调整；也没有重新开展 LOSVER 三种子比较，因此这些结果不能证明已达到相对 LOSVER 提高 2–3 个百分点的目标。67 项相关单元测试及两数据集的真实 RTX4090 smoke 均通过；检查包含 CLS 头、原 RGCL checkpoint、梯度与更新、RNG、恢复、单函数推理及 Diff 不参与推理。原始模型与数据处理实现保持不变；Patch 进程已停止，历史结果和权重保留。
