# RGCL 显式函数上下文交互消融

PrimeVul + UniXcoder，比较 RGCL 与 RGCL w/o explicit context interaction。仅将 s_i=W(h_i)+dot(U(h_i),V(g))/sqrt(32) 改为 s_i=W(h_i)；h_i 本身仍是 UniXcoder 的上下文代码行表示，不能把后者称为无上下文编码器。两组均保留原函数分类头、1–5倍有界Token加权MIL、函数BCE及0.1倍双侧真实Diff区域损失。512 Token、Batch8、16 Epoch、AdamW lr=2e-5，种子42/1234/123456；按完整Valid P-C四位小数严格改善选模，同分取较早轮，阈值严格>0.5。Test仅评价固定checkpoint。原RGCL三组及同协议非Context的Seed123456直接复用，仅训练缺少的Seed42/1234，不覆盖历史权重。

Seeds are 42, 1234 and 123456. 4 original runs are reused; completed new runs use the identical shared trainer. Seed123457, historical Global Mean and the stopped Patch study are preserved and excluded from the three-seed mean. Historical result paths and completed optimization budgets are recorded in experiments.json. Standard deviations below use ddof=1. New resumable checkpoints are saved every 500 steps and at epoch boundaries; this changes disk I/O only, with the original optimization budget and RNG behavior. PrimeVul FPR uses the complete repaired side of official pairs; other datasets use all retained normal functions, including ordinary and repaired functions. PR-AUC retains the existing average-precision (AP) definition.

| Dataset | Split | Seed | Method | Epoch | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primevul_paired | valid | 42 | rgcl_no_context | 10 | 15.30% | 0.096755 | 0.573109 | 0.577872 | 0.570018 | 0.606762 | 0.510676 | 0.548043 | 0.542994 |
| primevul_paired | test | 42 | rgcl_no_context | 10 | 14.18% | 0.108534 | 0.571915 | 0.579904 | 0.571454 | 0.595745 | 0.487589 | 0.554078 | 0.549918 |
| primevul_paired | valid | 42 | rgcl | 16 | 18.33% | 0.119044 | 0.592105 | 0.581556 | 0.573625 | 0.640569 | 0.523132 | 0.558719 | 0.550459 |
| primevul_paired | test | 42 | rgcl | 16 | 13.65% | 0.088549 | 0.583670 | 0.568611 | 0.565063 | 0.640071 | 0.553191 | 0.543440 | 0.536404 |
| primevul_paired | valid | 1234 | rgcl_no_context | 15 | 15.30% | 0.104117 | 0.596323 | 0.568596 | 0.566512 | 0.663701 | 0.562278 | 0.550712 | 0.541364 |
| primevul_paired | test | 1234 | rgcl_no_context | 15 | 12.94% | 0.067167 | 0.578062 | 0.549469 | 0.545864 | 0.640071 | 0.574468 | 0.532801 | 0.527007 |
| primevul_paired | valid | 1234 | rgcl | 13 | 16.37% | 0.106080 | 0.598086 | 0.573302 | 0.571040 | 0.667260 | 0.564057 | 0.551601 | 0.541908 |
| primevul_paired | test | 1234 | rgcl | 13 | 14.72% | 0.109051 | 0.597444 | 0.571367 | 0.565368 | 0.663121 | 0.556738 | 0.553191 | 0.543605 |
| primevul_paired | valid | 123456 | rgcl_no_context | 12 | 16.01% | 0.128266 | 0.632593 | 0.587298 | 0.589426 | 0.759786 | 0.642349 | 0.558719 | 0.541878 |
| primevul_paired | test | 123456 | rgcl_no_context | 12 | 10.46% | 0.087927 | 0.619780 | 0.561887 | 0.560123 | 0.750000 | 0.670213 | 0.539894 | 0.528090 |
| primevul_paired | valid | 123456 | rgcl | 14 | 17.79% | 0.143089 | 0.606998 | 0.589555 | 0.582306 | 0.663701 | 0.523132 | 0.570285 | 0.559220 |
| primevul_paired | test | 123456 | rgcl | 14 | 13.65% | 0.088486 | 0.582996 | 0.571188 | 0.564393 | 0.638298 | 0.551418 | 0.543440 | 0.536513 |

| Dataset | Split | Method | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primevul_paired | valid | rgcl_no_context | 15.54% ± 0.41pp | 0.109713 ± 0.016484 | 0.600675 ± 0.029980 | 0.577922 ± 0.009351 | 0.575319 ± 0.012343 | 0.676750 ± 0.077342 | 0.571767 ± 0.066347 | 0.552491 ± 0.005556 | 0.542079 ± 0.000833 |
| primevul_paired | valid | rgcl | 17.50% ± 1.01pp | 0.122737 ± 0.018779 | 0.599063 ± 0.007494 | 0.581471 ± 0.008127 | 0.575657 ± 0.005902 | 0.657177 ± 0.014492 | 0.536773 ± 0.023628 | 0.560202 ± 0.009429 | 0.550529 ± 0.008657 |
| primevul_paired | test | rgcl_no_context | 12.53% ± 1.90pp | 0.087876 ± 0.020683 | 0.589919 ± 0.026042 | 0.563753 ± 0.015303 | 0.559147 ± 0.012823 | 0.661939 ± 0.079419 | 0.577423 ± 0.091348 | 0.542258 ± 0.010834 | 0.535005 ± 0.012926 |
| primevul_paired | test | rgcl | 14.01% ± 0.61pp | 0.095362 ± 0.011855 | 0.588037 ± 0.008154 | 0.570389 ± 0.001542 | 0.564941 ± 0.000499 | 0.647163 ± 0.013848 | 0.553783 ± 0.002708 | 0.546690 ± 0.005630 | 0.538840 ± 0.004126 |

primevul_paired valid, rgcl_no_context → rgcl: mean p_c +0.019573; improved in 3/3 seeds. AUC +0.003549; MCC +0.013025; FPR -0.034994.

primevul_paired test, rgcl_no_context → rgcl: mean p_c +0.014775; improved in 2/3 seeds. AUC +0.006635; MCC +0.007486; FPR -0.023641.

## 相同协议与实际分类得失

移除两个无偏置32维投影，参数减少49,152；编码器、行表示、原注意力头、分类头和损失不变。原上下文参数在独立RNG环境中初始化，因此共享参数初值和采样RNG相同。同种子核对完整配置、成员、输入覆盖、Train区域掩码及末尾采样RNG。本比较衡量该交互组件及其参数的整体贡献。

| Seed | Split | 新增P-C | 丢失P-C | 净增加P-C | P-C变化/pp |
|---|---|---:|---:|---:|---:|
| 42 | valid | 48 | 31 | +17 | +3.02 |
| 42 | test | 31 | 34 | -3 | -0.53 |
| 1234 | valid | 42 | 36 | +6 | +1.07 |
| 1234 | test | 44 | 34 | +10 | +1.77 |
| 123456 | valid | 46 | 36 | +10 | +1.78 |
| 123456 | test | 40 | 22 | +18 | +3.19 |

同一配对跨种子的记录分别统计，不把重复的Test样本当作新的独立样本。这项训练消融衡量显式函数级交互在真实Diff监督下的贡献，包含交互项与修复监督共同优化的作用；它不等同于证明全部上下文信息来自该项，也不能仅凭注意力变化解释分类提升。

## 固定协议下的结论

valid：加入显式上下文交互后，平均P-C +1.96个百分点，AUC +0.003549、MCC +0.013025、F1 -0.001612、Recall -1.96个百分点、修复侧FPR -3.50个百分点；P-C提升/持平/下降的种子数为3/0/0。
test：加入显式上下文交互后，平均P-C +1.48个百分点，AUC +0.006635、MCC +0.007486、F1 -0.001882、Recall -1.48个百分点、修复侧FPR -2.36个百分点；P-C提升/持平/下降的种子数为2/0/1。

三种子仅支持本固定协议下的稳定性描述，不据此宣称统计显著性。Test未用于选择结构、参数、阈值或轮次；没有增加新模块或进行超参数搜索。模型、训练、恢复及单函数推理共用正式实现；相关测试与GPU检查见tests.log、resume-tests.log、gpu-smoke.json。
