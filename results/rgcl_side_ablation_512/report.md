# RGCL 双侧修复监督消融

PrimeVul + UniXcoder，Context-only 与 RGCL 复用历史完整结果；仅新增 Vulnerable-only、Repaired-only 各三种子。全部函数保留原标签参与 BCE，训练顺序不变；仅在训练特征中禁用另一侧的区域标签。区域损失仍按有效 Token 加权、在有效启用函数间平均，系数0.1，不补偿监督数量差异。K=5、512 Token、Batch8、16 Epoch、AdamW lr=2e-5；完整 Valid P-C（保留四位小数，先到先选）选模，阈值严格>0.5。Test 只评价既定 checkpoint，不用于修改方法、训练轮次或参数。

Seeds are 42, 1234 and 123456. 6 original runs are reused; completed new runs use the identical shared trainer. Seed123457, historical Global Mean and the stopped Patch study are preserved and excluded from the three-seed mean. Historical result paths and completed optimization budgets are recorded in experiments.json. Standard deviations below use ddof=1. New resumable checkpoints are saved every 500 steps and at epoch boundaries; this changes disk I/O only, with the original optimization budget and RNG behavior. PrimeVul FPR uses the complete repaired side of official pairs; other datasets use all retained normal functions, including ordinary and repaired functions. PR-AUC retains the existing average-precision (AP) definition.

| Dataset | Split | Seed | Method | Epoch | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primevul_paired | valid | 42 | context_only | 14 | 15.66% | 0.114408 | 0.577250 | 0.588396 | 0.573534 | 0.604982 | 0.491103 | 0.556940 | 0.551948 |
| primevul_paired | test | 42 | context_only | 14 | 13.65% | 0.101621 | 0.572873 | 0.568045 | 0.556916 | 0.602837 | 0.501773 | 0.550532 | 0.545746 |
| primevul_paired | valid | 42 | vulnerable_only | 16 | 15.66% | 0.108541 | 0.553874 | 0.572738 | 0.556772 | 0.553381 | 0.444840 | 0.554270 | 0.554367 |
| primevul_paired | test | 42 | vulnerable_only | 16 | 10.82% | 0.063830 | 0.531083 | 0.554221 | 0.548915 | 0.530142 | 0.466312 | 0.531915 | 0.532028 |
| primevul_paired | valid | 42 | repaired_only | 10 | 15.12% | 0.108617 | 0.562445 | 0.572351 | 0.572411 | 0.572954 | 0.464413 | 0.554270 | 0.552316 |
| primevul_paired | test | 42 | repaired_only | 10 | 13.12% | 0.103071 | 0.566038 | 0.576747 | 0.569529 | 0.585106 | 0.482270 | 0.551418 | 0.548173 |
| primevul_paired | valid | 42 | rgcl | 16 | 18.33% | 0.119044 | 0.592105 | 0.581556 | 0.573625 | 0.640569 | 0.523132 | 0.558719 | 0.550459 |
| primevul_paired | test | 42 | rgcl | 16 | 13.65% | 0.088549 | 0.583670 | 0.568611 | 0.565063 | 0.640071 | 0.553191 | 0.543440 | 0.536404 |
| primevul_paired | valid | 1234 | context_only | 5 | 8.54% | 0.072352 | 0.604230 | 0.558787 | 0.547584 | 0.711744 | 0.644128 | 0.533808 | 0.524934 |
| primevul_paired | test | 1234 | context_only | 5 | 9.93% | 0.095091 | 0.614114 | 0.558596 | 0.554128 | 0.725177 | 0.636525 | 0.544326 | 0.532552 |
| primevul_paired | valid | 1234 | vulnerable_only | 16 | 14.06% | 0.075542 | 0.500960 | 0.559129 | 0.555929 | 0.464413 | 0.389680 | 0.537367 | 0.543750 |
| primevul_paired | test | 1234 | vulnerable_only | 16 | 13.65% | 0.085770 | 0.512287 | 0.560089 | 0.552736 | 0.480496 | 0.395390 | 0.542553 | 0.548583 |
| primevul_paired | valid | 1234 | repaired_only | 10 | 17.62% | 0.137081 | 0.600329 | 0.590819 | 0.591387 | 0.649466 | 0.514235 | 0.567616 | 0.558104 |
| primevul_paired | test | 1234 | repaired_only | 10 | 13.12% | 0.095219 | 0.580804 | 0.577312 | 0.568393 | 0.627660 | 0.533688 | 0.546986 | 0.540458 |
| primevul_paired | valid | 1234 | rgcl | 13 | 16.37% | 0.106080 | 0.598086 | 0.573302 | 0.571040 | 0.667260 | 0.564057 | 0.551601 | 0.541908 |
| primevul_paired | test | 1234 | rgcl | 13 | 14.72% | 0.109051 | 0.597444 | 0.571367 | 0.565368 | 0.663121 | 0.556738 | 0.553191 | 0.543605 |
| primevul_paired | valid | 123456 | context_only | 16 | 15.48% | 0.098152 | 0.530990 | 0.575924 | 0.574062 | 0.510676 | 0.412811 | 0.548932 | 0.552987 |
| primevul_paired | test | 123456 | context_only | 16 | 12.23% | 0.092480 | 0.527675 | 0.569080 | 0.568154 | 0.507092 | 0.414894 | 0.546099 | 0.550000 |
| primevul_paired | valid | 123456 | vulnerable_only | 15 | 16.55% | 0.119217 | 0.559216 | 0.582069 | 0.578049 | 0.558719 | 0.439502 | 0.559609 | 0.559715 |
| primevul_paired | test | 123456 | vulnerable_only | 15 | 12.41% | 0.079802 | 0.544337 | 0.562825 | 0.556920 | 0.549645 | 0.469858 | 0.539894 | 0.539130 |
| primevul_paired | valid | 123456 | repaired_only | 9 | 16.19% | 0.138117 | 0.539411 | 0.595136 | 0.590071 | 0.505338 | 0.368327 | 0.568505 | 0.578411 |
| primevul_paired | test | 123456 | repaired_only | 9 | 11.52% | 0.083890 | 0.479920 | 0.572085 | 0.569332 | 0.423759 | 0.342199 | 0.540780 | 0.553241 |
| primevul_paired | valid | 123456 | rgcl | 14 | 17.79% | 0.143089 | 0.606998 | 0.589555 | 0.582306 | 0.663701 | 0.523132 | 0.570285 | 0.559220 |
| primevul_paired | test | 123456 | rgcl | 14 | 13.65% | 0.088486 | 0.582996 | 0.571188 | 0.564393 | 0.638298 | 0.551418 | 0.543440 | 0.536513 |

| Dataset | Split | Method | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primevul_paired | valid | context_only | 13.23% ± 4.06pp | 0.094971 ± 0.021208 | 0.570823 ± 0.037040 | 0.574369 ± 0.014866 | 0.565060 ± 0.015137 | 0.609134 ± 0.100598 | 0.516014 ± 0.117653 | 0.546560 ± 0.011747 | 0.543290 ± 0.015905 |
| primevul_paired | valid | vulnerable_only | 15.42% ± 1.26pp | 0.101100 ± 0.022769 | 0.538017 ± 0.032203 | 0.571312 ± 0.011536 | 0.563583 ± 0.012535 | 0.525504 ± 0.052974 | 0.424674 ± 0.030423 | 0.550415 ± 0.011611 | 0.552611 ± 0.008126 |
| primevul_paired | valid | repaired_only | 16.31% ± 1.25pp | 0.127938 ± 0.016741 | 0.567395 ± 0.030759 | 0.586102 ± 0.012103 | 0.584623 ± 0.010596 | 0.575919 ± 0.072110 | 0.448992 ± 0.074166 | 0.563464 ± 0.007974 | 0.562944 ± 0.013705 |
| primevul_paired | valid | rgcl | 17.50% ± 1.01pp | 0.122737 ± 0.018779 | 0.599063 ± 0.007494 | 0.581471 ± 0.008127 | 0.575657 ± 0.005902 | 0.657177 ± 0.014492 | 0.536773 ± 0.023628 | 0.560202 ± 0.009429 | 0.550529 ± 0.008657 |
| primevul_paired | test | context_only | 11.94% ± 1.88pp | 0.096398 ± 0.004709 | 0.571554 ± 0.043235 | 0.565240 ± 0.005778 | 0.559733 ± 0.007425 | 0.611702 ± 0.109313 | 0.517730 ± 0.111674 | 0.546986 ± 0.003196 | 0.542766 ± 0.009098 |
| primevul_paired | test | vulnerable_only | 12.29% ± 1.42pp | 0.076467 ± 0.011344 | 0.529236 ± 0.016105 | 0.559045 ± 0.004396 | 0.552857 ± 0.004004 | 0.520095 ± 0.035653 | 0.443853 ± 0.042008 | 0.538121 ± 0.005536 | 0.539914 ± 0.008305 |
| primevul_paired | test | repaired_only | 12.59% ± 0.92pp | 0.094060 ± 0.009643 | 0.542254 ± 0.054485 | 0.575381 ± 0.002868 | 0.569085 ± 0.000607 | 0.545508 ± 0.107563 | 0.452719 ± 0.099106 | 0.546395 ± 0.005344 | 0.547291 ± 0.006437 |
| primevul_paired | test | rgcl | 14.01% ± 0.61pp | 0.095362 ± 0.011855 | 0.588037 ± 0.008154 | 0.570389 ± 0.001542 | 0.564941 ± 0.000499 | 0.647163 ± 0.013848 | 0.553783 ± 0.002708 | 0.546690 ± 0.005630 | 0.538840 ± 0.004126 |

primevul_paired valid, context_only → vulnerable_only: mean p_c +0.021945; improved in 2/3 seeds. AUC -0.003057; MCC +0.006129; FPR -0.091340.

primevul_paired valid, context_only → repaired_only: mean p_c +0.030842; improved in 2/3 seeds. AUC +0.011733; MCC +0.032967; FPR -0.067023.

primevul_paired valid, context_only → rgcl: mean p_c +0.042705; improved in 3/3 seeds. AUC +0.007102; MCC +0.027766; FPR +0.020759.

primevul_paired valid, vulnerable_only → rgcl: mean p_c +0.020759; improved in 3/3 seeds. AUC +0.010159; MCC +0.021638; FPR +0.112100.

primevul_paired valid, repaired_only → rgcl: mean p_c +0.011862; improved in 2/3 seeds. AUC -0.004630; MCC -0.005201; FPR +0.087782.

primevul_paired test, context_only → vulnerable_only: mean p_c +0.003546; improved in 2/3 seeds. AUC -0.006195; MCC -0.019930; FPR -0.073877.

primevul_paired test, context_only → repaired_only: mean p_c +0.006501; improved in 1/3 seeds. AUC +0.010141; MCC -0.002337; FPR -0.065012.

primevul_paired test, context_only → rgcl: mean p_c +0.020686; improved in 2/3 seeds. AUC +0.005148; MCC -0.001035; FPR +0.036052.

primevul_paired test, vulnerable_only → rgcl: mean p_c +0.017139; improved in 3/3 seeds. AUC +0.011344; MCC +0.018895; FPR +0.109929.

primevul_paired test, repaired_only → rgcl: mean p_c +0.014184; improved in 3/3 seeds. AUC -0.004993; MCC +0.001302; FPR +0.101064.

## 真实区域覆盖与实际训练监督

有效区域要求512输入中同时存在修改 Token 和背景 Token。跨物理行 Token 按原 bounded_pool 规则分摊，表中 Token 质量是分摊后的数量，不是代码语义质量评分。空区域、全区域和 Padding 沿用原处理。

| Train side | Functions | Eligible Diff functions | Side coverage | Foreground Token mass | Background Token mass |
|---|---:|---:|---:|---:|---:|
| Repaired | 4354 | 3005 | 69.02% | 154024 | 809335 |
| Vulnerable | 4354 | 2175 | 49.95% | 91472 | 586231 |

| Seed | Method | Supervised functions / all Train | Foreground Token mass / epoch | Background Token mass / epoch | Actual function visits (16 epochs) |
|---|---|---:|---:|---:|---:|
| 42 | context_only | 0/8708 | 0 | 0 | — (historical) |
| 42 | vulnerable_only | 2175/8708 | 91472 | 586231 | 34800 |
| 42 | repaired_only | 3005/8708 | 154024 | 809335 | 48080 |
| 42 | rgcl | 5180/8708 | 245496 | 1395566 | — (historical) |
| 1234 | context_only | 0/8708 | 0 | 0 | — (historical) |
| 1234 | vulnerable_only | 2175/8708 | 91472 | 586231 | 34800 |
| 1234 | repaired_only | 3005/8708 | 154024 | 809335 | 48080 |
| 1234 | rgcl | 5180/8708 | 245496 | 1395566 | — (historical) |
| 123456 | context_only | 0/8708 | 0 | 0 | — (historical) |
| 123456 | vulnerable_only | 2175/8708 | 91472 | 586231 | 34800 |
| 123456 | repaired_only | 3005/8708 | 154024 | 809335 | 48080 |
| 123456 | rgcl | 5180/8708 | 245496 | 1395566 | — (historical) |

新增六组的实际函数访问次数来自可恢复训练状态，并核对每个样本保存的真实区域掩码。历史对照不改写训练日志；其每轮监督数量由同一原始 Train 编码和损失有效性条件重算。漏洞侧与修复侧可监督函数数及 Token 质量不同，且原损失在每批有效函数间平均。因此本消融同时改变监督来源与覆盖，不能把性能差异直接归因于两侧监督的语义质量，也不作数量补偿。

结果回答固定协议下各侧监督的实际效果；不能据此认定统计显著性。三种子分别评价，均值与标准差以种子为单位，不把重复训练的同一测试样本当作独立样本。

## 固定协议下的结论

context_only → vulnerable_only：Test 平均 P-C +0.35 个百分点，MCC -0.019930，Recall -9.16 个百分点，修复侧 FPR -7.39 个百分点；P-C 提升/持平/下降的种子数为 2/0/1。
context_only → repaired_only：Test 平均 P-C +0.65 个百分点，MCC -0.002337，Recall -6.62 个百分点，修复侧 FPR -6.50 个百分点；P-C 提升/持平/下降的种子数为 1/0/2。
vulnerable_only → rgcl：Test 平均 P-C +1.71 个百分点，MCC +0.018895，Recall +12.71 个百分点，修复侧 FPR +10.99 个百分点；P-C 提升/持平/下降的种子数为 3/0/0。
repaired_only → rgcl：Test 平均 P-C +1.42 个百分点，MCC +0.001302，Recall +10.17 个百分点，修复侧 FPR +10.11 个百分点；P-C 提升/持平/下降的种子数为 3/0/0。

两种单侧监督仅获得较小、跨种子不一致的 Test P-C 收益：Vulnerable-only 平均提高0.35个百分点（2/3种子提高），Repaired-only 提高0.65个百分点（1/3提高）。两者的平均 MCC、F1、Recall 均低于 Context-only。单侧监督均降低了修复侧 FPR，但同时减少了漏洞检出，不能把误报下降单独视为检测改进。

双侧 RGCL 的 Test P-C 在3/3种子上都高于每种单侧方法，平均分别提高1.71和1.42个百分点；F1、Recall 也更高。这支持固定协议下双侧联合监督对配对分类的作用，但不能解释为全面优势：Repaired-only 的平均 AUC/PR-AUC 为0.575381/0.569085，高于双侧的0.570389/0.564941；双侧相对它的 MCC 仅提高0.001302，修复侧 FPR 则提高10.11个百分点。相对 Context-only，双侧平均 MCC 也未改善。区域覆盖和 Token 质量不等，不能将侧别间差异直接归因于修复语义质量。

验证记录：tests.log（64项区域损失、训练、恢复回归）、report-tests.log（1项四方法指标与覆盖分母测试），gpu-smoke.json / primevul_paired/gpu-smoke.json（真实 GPU 梯度、原 RGCL 权重恢复及单函数推理）。新增参数为0，初始化及采样 RNG 不变；推理不读取区域标签。完成训练后的 Seed42 四方法 CPU RNG、实际使用的 CUDA RNG 均完全一致，核对结果保留在 gpu-smoke.json 的 completed_seed42_rng_check 中。
首次沙箱测试因 DataLoader 多进程套接字权限失败，原日志保留于 tests.sandbox.log；相同测试已在宿主环境重跑通过，没有修改断言或跳过失败。

## 复现命令

在仓库根目录使用本地 vul-detect 环境。两种方法的三个种子分别在两张卡的独立 tmux 队列中依次运行；以下两条循环可在各自终端执行。已完成组合会由正式入口直接复用。

```bash
/home/phy/miniconda3/envs/vul-detect/bin/python -m unittest tests.test_rgcl tests.test_bounded_attention tests.test_resume
/home/phy/miniconda3/envs/vul-detect/bin/python -m vulnmechanism.cli rgcl-study check --dataset primevul_paired --methods context_only vulnerable_only repaired_only rgcl --output-dir results/rgcl_side_ablation_512 --reference-dir results/rgcl_multiseed_512 --device cuda:0

for seed in 42 1234 123456; do
  /home/phy/miniconda3/envs/vul-detect/bin/python -m vulnmechanism.cli rgcl-study run --dataset primevul_paired --methods vulnerable_only --seed "$seed" --output-dir results/rgcl_side_ablation_512 --reference-dir results/rgcl_multiseed_512 --device cuda:0 --save-every 500
done

for seed in 42 1234 123456; do
  /home/phy/miniconda3/envs/vul-detect/bin/python -m vulnmechanism.cli rgcl-study run --dataset primevul_paired --methods repaired_only --seed "$seed" --output-dir results/rgcl_side_ablation_512 --reference-dir results/rgcl_multiseed_512 --device cuda:1 --save-every 500
done

/home/phy/miniconda3/envs/vul-detect/bin/python -m vulnmechanism.cli rgcl-study report --dataset primevul_paired --methods context_only vulnerable_only repaired_only rgcl --output-dir results/rgcl_side_ablation_512 --reference-dir results/rgcl_multiseed_512
```
