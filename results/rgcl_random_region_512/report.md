# RGCL Random-region 三种子消融

只在 PrimeVul + UniXcoder 补充 Random-region；Context-only 与 RGCL 复用历史完整结果。512 Token、Batch8、16 Epoch、AdamW lr=2e-5、K=5、区域损失权重0.1；按完整 Valid P-C 选模，阈值严格>0.5。随机区域为有效代码行的随机子集，逐函数严格保持原 Diff 的有效 Token 质量及辅助损失可用性。跨行 Token 按原实现分摊；局部独立 RNG 随训练种子固定，整个训练期间不重采样。保持原区域损失、编码和函数 BCE；推理仅输入单函数，Test 不参与选模。

Seeds are 42, 1234 and 123456. 6 original runs are reused; completed new runs use the identical shared trainer. Seed123457, historical Global Mean and the stopped Patch study are preserved and excluded from the three-seed mean. Historical result paths and completed optimization budgets are recorded in experiments.json. Standard deviations below use ddof=1. New resumable checkpoints are saved every 500 steps and at epoch boundaries; this changes disk I/O only, with the original optimization budget and RNG behavior. PrimeVul FPR uses the complete repaired side of official pairs; other datasets use all retained normal functions, including ordinary and repaired functions. PR-AUC retains the existing average-precision (AP) definition.

| Dataset | Split | Seed | Method | Epoch | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primevul_paired | valid | 42 | context_only | 14 | 15.66% | 0.114408 | 0.577250 | 0.588396 | 0.573534 | 0.604982 | 0.491103 | 0.556940 | 0.551948 |
| primevul_paired | test | 42 | context_only | 14 | 13.65% | 0.101621 | 0.572873 | 0.568045 | 0.556916 | 0.602837 | 0.501773 | 0.550532 | 0.545746 |
| primevul_paired | valid | 42 | random_region | 14 | 15.66% | 0.107033 | 0.568729 | 0.575564 | 0.562892 | 0.588968 | 0.482206 | 0.553381 | 0.549834 |
| primevul_paired | test | 42 | random_region | 14 | 13.12% | 0.081937 | 0.561760 | 0.566272 | 0.561791 | 0.588652 | 0.507092 | 0.540780 | 0.537217 |
| primevul_paired | valid | 42 | rgcl | 16 | 18.33% | 0.119044 | 0.592105 | 0.581556 | 0.573625 | 0.640569 | 0.523132 | 0.558719 | 0.550459 |
| primevul_paired | test | 42 | rgcl | 16 | 13.65% | 0.088549 | 0.583670 | 0.568611 | 0.565063 | 0.640071 | 0.553191 | 0.543440 | 0.536404 |
| primevul_paired | valid | 1234 | context_only | 5 | 8.54% | 0.072352 | 0.604230 | 0.558787 | 0.547584 | 0.711744 | 0.644128 | 0.533808 | 0.524934 |
| primevul_paired | test | 1234 | context_only | 5 | 9.93% | 0.095091 | 0.614114 | 0.558596 | 0.554128 | 0.725177 | 0.636525 | 0.544326 | 0.532552 |
| primevul_paired | valid | 1234 | random_region | 16 | 13.52% | 0.069835 | 0.559393 | 0.554999 | 0.549926 | 0.590747 | 0.521352 | 0.534698 | 0.531200 |
| primevul_paired | test | 1234 | random_region | 16 | 11.35% | 0.058980 | 0.557131 | 0.554374 | 0.555525 | 0.592199 | 0.533688 | 0.529255 | 0.525984 |
| primevul_paired | valid | 1234 | rgcl | 13 | 16.37% | 0.106080 | 0.598086 | 0.573302 | 0.571040 | 0.667260 | 0.564057 | 0.551601 | 0.541908 |
| primevul_paired | test | 1234 | rgcl | 13 | 14.72% | 0.109051 | 0.597444 | 0.571367 | 0.565368 | 0.663121 | 0.556738 | 0.553191 | 0.543605 |
| primevul_paired | valid | 123456 | context_only | 16 | 15.48% | 0.098152 | 0.530990 | 0.575924 | 0.574062 | 0.510676 | 0.412811 | 0.548932 | 0.552987 |
| primevul_paired | test | 123456 | context_only | 16 | 12.23% | 0.092480 | 0.527675 | 0.569080 | 0.568154 | 0.507092 | 0.414894 | 0.546099 | 0.550000 |
| primevul_paired | valid | 123456 | random_region | 11 | 14.06% | 0.106720 | 0.508309 | 0.575181 | 0.570359 | 0.462633 | 0.357651 | 0.552491 | 0.563991 |
| primevul_paired | test | 123456 | random_region | 11 | 11.35% | 0.084607 | 0.470348 | 0.562915 | 0.566106 | 0.407801 | 0.326241 | 0.540780 | 0.555556 |
| primevul_paired | valid | 123456 | rgcl | 14 | 17.79% | 0.143089 | 0.606998 | 0.589555 | 0.582306 | 0.663701 | 0.523132 | 0.570285 | 0.559220 |
| primevul_paired | test | 123456 | rgcl | 14 | 13.65% | 0.088486 | 0.582996 | 0.571188 | 0.564393 | 0.638298 | 0.551418 | 0.543440 | 0.536513 |

| Dataset | Split | Method | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primevul_paired | valid | context_only | 13.23% ± 4.06pp | 0.094971 ± 0.021208 | 0.570823 ± 0.037040 | 0.574369 ± 0.014866 | 0.565060 ± 0.015137 | 0.609134 ± 0.100598 | 0.516014 ± 0.117653 | 0.546560 ± 0.011747 | 0.543290 ± 0.015905 |
| primevul_paired | valid | random_region | 14.41% ± 1.11pp | 0.094529 ± 0.021386 | 0.545477 ± 0.032525 | 0.568581 ± 0.011764 | 0.561059 ± 0.010339 | 0.547450 ± 0.073458 | 0.453737 ± 0.085483 | 0.546856 ± 0.010539 | 0.548342 ± 0.016447 |
| primevul_paired | valid | rgcl | 17.50% ± 1.01pp | 0.122737 ± 0.018779 | 0.599063 ± 0.007494 | 0.581471 ± 0.008127 | 0.575657 ± 0.005902 | 0.657177 ± 0.014492 | 0.536773 ± 0.023628 | 0.560202 ± 0.009429 | 0.550529 ± 0.008657 |
| primevul_paired | test | context_only | 11.94% ± 1.88pp | 0.096398 ± 0.004709 | 0.571554 ± 0.043235 | 0.565240 ± 0.005778 | 0.559733 ± 0.007425 | 0.611702 ± 0.109313 | 0.517730 ± 0.111674 | 0.546986 ± 0.003196 | 0.542766 ± 0.009098 |
| primevul_paired | test | random_region | 11.94% ± 1.02pp | 0.075175 ± 0.014089 | 0.529746 ± 0.051493 | 0.561187 ± 0.006135 | 0.561141 ± 0.005320 | 0.529551 ± 0.105453 | 0.455674 ± 0.112878 | 0.536939 ± 0.006654 | 0.539586 ± 0.014927 |
| primevul_paired | test | rgcl | 14.01% ± 0.61pp | 0.095362 ± 0.011855 | 0.588037 ± 0.008154 | 0.570389 ± 0.001542 | 0.564941 ± 0.000499 | 0.647163 ± 0.013848 | 0.553783 ± 0.002708 | 0.546690 ± 0.005630 | 0.538840 ± 0.004126 |

primevul_paired valid, context_only → random_region: mean p_c +0.011862; improved in 1/3 seeds. AUC -0.005788; MCC -0.000442; FPR -0.062278.

primevul_paired valid, random_region → rgcl: mean p_c +0.030842; improved in 3/3 seeds. AUC +0.012890; MCC +0.028208; FPR +0.083037.

primevul_paired valid, context_only → rgcl: mean p_c +0.042705; improved in 3/3 seeds. AUC +0.007102; MCC +0.027766; FPR +0.020759.

primevul_paired test, context_only → random_region: mean p_c -0.000000; improved in 1/3 seeds. AUC -0.004053; MCC -0.021223; FPR -0.062057.

primevul_paired test, random_region → rgcl: mean p_c +0.020686; improved in 3/3 seeds. AUC +0.009202; MCC +0.020188; FPR +0.098109.

primevul_paired test, context_only → rgcl: mean p_c +0.020686; improved in 2/3 seeds. AUC +0.005148; MCC -0.001035; FPR +0.036052.

## 随机监督核对

随机子集仅以真实区域的 Token 总量为约束，不利用真实修改位置筛选候选。Token 质量按原物理行聚合分摊，逐函数精确保持；空区域和全区域继续不计算区域损失。不强制排除与真实区域自然重合的抽样，精确 Token 总量可能限制可选子集。

| Seed | Eligible functions | Exact token mass | Unchanged eligible | Token overlap / true positive tokens |
|---|---:|---:|---:|---:|
| 42 | 5180/8708 | 8708/8708 | 275/5180 | 34.92% |
| 1234 | 5180/8708 | 8708/8708 | 279/5180 | 34.60% |
| 123456 | 5180/8708 | 8708/8708 | 283/5180 | 34.73% |

全部新增训练均完成16轮，三种子均完整保留；历史 Context-only/RGCL 未重复训练或改写。初始化、采样、恢复与单函数推理检查见 tests.log、resume-tests.log 及 gpu-smoke.json。Test 在完整 Valid P-C 选模后一次评价；没有依据 Test 改动方法、参数或阈值。
