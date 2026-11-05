# RGCL fixed three-seed comparison

RGCL is the original Context + Diff: K=5, lambda=0.1, token-weighted region loss. CLS reuses the SecVulEval RobertaClassificationHead. Context-only differs from RGCL only by removing Diff loss. UniXcoder, 512 tokens, Batch8, 16 epochs, AdamW lr=2e-5, original function sampling; threshold strictly >0.5. PrimeVul selects full Valid P-C; other datasets select full Valid MCC. Test never selects a checkpoint or configuration.


## CleanVul 成员及监督口径

采用评分≥3的全部 C/C++ 发布记录：3136 对，6272 个函数。发布文件没有独立0/1字段；沿用修复前=1、修复后=0的约定。源码逐字保留，未进行新的质量筛选。评分定义见 [CleanVul 原论文](https://arxiv.org/html/2411.17274v3#S3)。

此前753条排除来自可执行检查：解析不完整/非单函数697、名称不同50、名称未知1、词法序列不变5。这些是语法与对应关系诊断，不能判定分类标签错误；本轮全部恢复，检查代码仅记录原因，不过滤分类成员。对应来源及诊断见 pairs.csv，最终成员见 data/rgcl/cleanvul/members.csv。区域监督直接沿用同一发布行的前后版本及原 _line_diff / paired_features；不额外要求解析成功或同名。512可见区域须同时含修改行和背景行，保留原有部分可见修改行门控；无有效区域仍参与BCE。

Seed42按CVE、仓库限定提交、修复关系和空白归一化重复代码传递分组，目标8:1:1。由于恢复了原先排除的成员，重新对全部成员分组；旧2383对口径未用于任何训练。跨划分重叠组为0；不依据Valid/Test结果修改成员。

| Split | Pairs | Vulnerable | Repaired | Effective before | Effective after | Effective / functions |
|---|---:|---:|---:|---:|---:|---:|
| train | 2508 | 2508 | 2508 | 1440 | 1852 | 3292/5016 (65.63%) |
| valid | 314 | 314 | 314 | 177 | 251 | 428/628 (68.15%) |
| test | 314 | 314 | 314 | 176 | 241 | 417/628 (66.40%) |

三种方法沿用同一函数随机采样、训练预算与预训练权重；Context-only与RGCL仅相差0.1倍真实Diff区域损失。不增加Patch、RCCL或局部头。历史PrimeVul/SecVulEval成员、权重与结果均保留。

Seeds are 42, 1234 and 123456. 0 original runs are reused; completed new runs use the identical shared trainer. Seed123457, historical Global Mean and the stopped Patch study are preserved and excluded from the three-seed mean. Historical result paths and completed optimization budgets are recorded in experiments.json. Standard deviations below use ddof=1. New resumable checkpoints are saved every 500 steps and at epoch boundaries; this changes disk I/O only, with the original optimization budget and RNG behavior. PrimeVul FPR uses the complete repaired side of official pairs; other datasets use all retained normal functions, including ordinary and repaired functions. PR-AUC retains the existing average-precision (AP) definition.

| Dataset | Split | Seed | Method | Epoch | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cleanvul | valid | 42 | plm_baseline | 9 | — | 0.119063 | 0.524871 | 0.578903 | 0.576875 | 0.487261 | 0.369427 | 0.558917 | 0.568773 |
| cleanvul | test | 42 | plm_baseline | 9 | — | 0.092436 | 0.555382 | 0.578467 | 0.569078 | 0.566879 | 0.474522 | 0.546178 | 0.544343 |
| cleanvul | valid | 42 | context_only | 10 | — | 0.140267 | 0.579439 | 0.578528 | 0.573842 | 0.592357 | 0.452229 | 0.570064 | 0.567073 |
| cleanvul | test | 42 | context_only | 10 | — | 0.099041 | 0.566616 | 0.563882 | 0.556839 | 0.589172 | 0.490446 | 0.549363 | 0.545723 |
| cleanvul | valid | 42 | rgcl | 7 | — | 0.112943 | 0.621803 | 0.558973 | 0.557983 | 0.735669 | 0.630573 | 0.552548 | 0.538462 |
| cleanvul | test | 42 | rgcl | 7 | — | 0.109086 | 0.618919 | 0.572949 | 0.559543 | 0.729299 | 0.627389 | 0.550955 | 0.537559 |
| cleanvul | valid | 1234 | plm_baseline | 12 | — | 0.110176 | 0.591837 | 0.561833 | 0.549875 | 0.646497 | 0.538217 | 0.554140 | 0.545699 |
| cleanvul | test | 1234 | plm_baseline | 12 | — | 0.088273 | 0.589413 | 0.567868 | 0.566967 | 0.656051 | 0.570064 | 0.542994 | 0.535065 |
| cleanvul | valid | 1234 | context_only | 16 | — | 0.112012 | 0.576631 | 0.567873 | 0.557708 | 0.605096 | 0.493631 | 0.555732 | 0.550725 |
| cleanvul | test | 1234 | context_only | 16 | — | 0.079871 | 0.557427 | 0.559658 | 0.565918 | 0.579618 | 0.500000 | 0.539809 | 0.536873 |
| cleanvul | valid | 1234 | rgcl | 7 | — | 0.127430 | 0.558065 | 0.567482 | 0.559767 | 0.550955 | 0.423567 | 0.563694 | 0.565359 |
| cleanvul | test | 1234 | rgcl | 7 | — | 0.079960 | 0.560122 | 0.566458 | 0.556890 | 0.585987 | 0.506369 | 0.539809 | 0.536443 |
| cleanvul | valid | 123456 | plm_baseline | 12 | — | 0.152995 | 0.614719 | 0.596348 | 0.579215 | 0.678344 | 0.528662 | 0.574841 | 0.562005 |
| cleanvul | test | 123456 | plm_baseline | 12 | — | 0.049444 | 0.578279 | 0.535595 | 0.545391 | 0.652866 | 0.605096 | 0.523885 | 0.518987 |
| cleanvul | valid | 123456 | context_only | 12 | — | 0.168931 | 0.620290 | 0.600633 | 0.584023 | 0.681529 | 0.515924 | 0.582803 | 0.569149 |
| cleanvul | test | 123456 | context_only | 12 | — | 0.075555 | 0.587234 | 0.560195 | 0.556064 | 0.659236 | 0.585987 | 0.536624 | 0.529412 |
| cleanvul | valid | 123456 | rgcl | 11 | — | 0.128555 | 0.612130 | 0.579237 | 0.572905 | 0.691083 | 0.566879 | 0.562102 | 0.549367 |
| cleanvul | test | 123456 | rgcl | 11 | — | 0.094733 | 0.610354 | 0.579278 | 0.575100 | 0.713376 | 0.624204 | 0.544586 | 0.533333 |

| Dataset | Split | Method | P-C | MCC | F1 | AUC | PR-AUC (AP) | Recall | FPR | Accuracy | Precision |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cleanvul | valid | plm_baseline | — | 0.127412 ± 0.022597 | 0.577142 ± 0.046691 | 0.579028 ± 0.017258 | 0.568655 ± 0.016306 | 0.604034 ± 0.102374 | 0.478769 ± 0.094813 | 0.562633 ± 0.010839 | 0.558826 ± 0.011861 |
| cleanvul | valid | context_only | — | 0.140403 ± 0.028460 | 0.592120 ± 0.024436 | 0.582344 ± 0.016710 | 0.571858 ± 0.013269 | 0.626327 ± 0.048229 | 0.487261 ± 0.032321 | 0.569533 ± 0.013543 | 0.562316 ± 0.010092 |
| cleanvul | valid | rgcl | — | 0.122976 ± 0.008707 | 0.597333 ± 0.034349 | 0.568564 ± 0.010175 | 0.563552 ± 0.008150 | 0.659236 ± 0.096387 | 0.540340 ± 0.106024 | 0.559448 ± 0.006029 | 0.551063 ± 0.013529 |
| cleanvul | test | plm_baseline | — | 0.076718 ± 0.023711 | 0.574358 ± 0.017351 | 0.560643 ± 0.022330 | 0.560479 ± 0.013109 | 0.625265 ± 0.050589 | 0.549894 ± 0.067583 | 0.537686 ± 0.012057 | 0.532798 ± 0.012829 |
| cleanvul | test | context_only | — | 0.084822 ± 0.012501 | 0.570426 ± 0.015264 | 0.561245 ± 0.002299 | 0.559607 ± 0.005479 | 0.609342 ± 0.043473 | 0.525478 ± 0.052620 | 0.541932 ± 0.006630 | 0.537336 ± 0.008165 |
| cleanvul | test | rgcl | — | 0.094593 ± 0.014564 | 0.596465 ± 0.031764 | 0.572895 ± 0.006410 | 0.563844 ± 0.009837 | 0.676221 ± 0.078549 | 0.585987 ± 0.068969 | 0.545117 ± 0.005592 | 0.535778 ± 0.002190 |

cleanvul valid, plm_baseline → context_only: mean mcc +0.012992; improved in 3/3 seeds. AUC +0.003317; MCC +0.012992; FPR +0.008493.

cleanvul valid, context_only → rgcl: mean mcc -0.017427; improved in 1/3 seeds. AUC -0.013780; MCC -0.017427; FPR +0.053079.

cleanvul valid, plm_baseline → rgcl: mean mcc -0.004436; improved in 1/3 seeds. AUC -0.010464; MCC -0.004436; FPR +0.061571.

cleanvul test, plm_baseline → context_only: mean mcc +0.008105; improved in 2/3 seeds. AUC +0.000602; MCC +0.008105; FPR -0.024416.

cleanvul test, context_only → rgcl: mean mcc +0.009771; improved in 3/3 seeds. AUC +0.011650; MCC +0.009771; FPR +0.060510.

cleanvul test, plm_baseline → rgcl: mean mcc +0.017875; improved in 2/3 seeds. AUC +0.012252; MCC +0.017875; FPR +0.036093.

## 结论与验证

这轮使用全部高分C/C++发布成员，没有用解析成功、同名或词法变化作为额外分类筛选条件。修复前后标签沿用既有约定；区域监督只采用原始Diff和既有512可见性条件。没有调整结构、K、λ、分类阈值或选模轮次规则，未根据Test结果追加实验。

Context-only相对CLS的Valid MCC在3/3种子提高，均值增加0.012992。RGCL相对Context-only的Valid MCC均值下降0.017427，只有1/3种子提高；Valid AUC均值下降0.013780。因而Valid结果不支持Diff监督带来稳定增益。

Test上RGCL相对Context-only的MCC在3/3种子略有提高，均值增加0.009771；其中Seed1234仅增加0.000089。平均AUC增加0.011650，F1增加0.026039，Recall增加6.69个百分点，同时修复侧FPR增加6.05个百分点（52.55%→58.60%）。AUC的提高表明存在小幅排序收益，但召回提高也伴随更多修复侧误报。RGCL相对CLS的Test MCC只在2/3种子提高。

综合Valid与Test，本轮只能报告有限的Test收益，不能声称RGCL在CleanVul上获得整体稳定优势；整体Test区分能力仍弱（RGCL平均AUC 0.572895、MCC 0.094593）。不把这些结果作为重新选择方法、参数或checkpoint的依据。未进行额外过滤消融，不能将差异直接归因于恢复的753条记录或某一语法类别。

已完成9组真实GPU训练，每组16 Epoch、10,032个优化步。相关单元测试80项通过，针对性22项回归复验通过；真实RTX4090 smoke验证了标准CLS头、Context-only/RGCL初始输出及CPU RNG一致性、梯度与参数更新、checkpoint恢复、单函数推理和Diff无关推理。原PrimeVul RGCL checkpoint在8个Valid smoke函数上的源码推理与历史预测一致。全部6,272个导出源码与CSV原文逐字一致；完整Valid/Test成员、标签、选模记录及逐样本概率已由共享report入口复核。

主要实现文件：vulnmechanism/cleanvul.py、data.py、cli.py、rgcl_study.py、evaluation.py及tests/test_cleanvul.py。复用既有模型、训练器和评价器，未修改model.py、training.py及RGCL主体。配置与命令见commands.txt、experiments.json及各组config.json；结果见comparison.json和各组Valid/Test预测文件。原始源码与模型权重保留在本地。
