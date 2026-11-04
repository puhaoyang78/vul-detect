# megavul: RGCL 原生函数二分类

仅 C/C++；先对全量原始成员分组划分，再在各划分内独立用 Seed42 下采样正常函数，保留全部漏洞函数，形成固定 1:1 子集；不生成修复后负例；同 CVE、提交、修复关联和空白归一化重复代码合并后，Seed42 按 8:1:1 目标分组划分。本地发布文件未附独立官方评价成员。

| Split | 函数 | 漏洞 | 正常 | 普通正常 | 修复后正常 | 有可靠对侧 | 有效可见区域 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 28760 | 14380 | 14380 | 14380 | 0 | 14380 | 7843 (27.3%) |
| valid | 3596 | 1798 | 1798 | 1798 | 0 | 1798 | 969 (26.9%) |
| test | 3594 | 1797 | 1797 | 1797 | 0 | 1797 | 967 (26.9%) |

区域覆盖率只作数据统计；Diff 辅助损失只用于 Train。PR-AUC 沿用现有 average_precision_score（AP）口径。

| Split | 抽样前漏洞 | 抽样前正常 | 保留漏洞 | 抽取正常 |
|---|---:|---:|---:|---:|
| train | 14380 | 268702 | 14380 | 14380 |
| valid | 1798 | 33596 | 1798 | 1798 |
| test | 1797 | 33600 | 1797 | 1797 |

实际子集成员见 data/rgcl/megavul/members.csv；全量分组及入选标记见 grouped_members.csv。先划分后抽样，不按 Test 结果修改成员。

预训练 UniXcoder；512 Token；Seed123456；Batch8；16 Epoch；LOSVERAdamW lr=2e-05、epsilon=1e-08，warmup=0.1，FP32。三组相同 Train 负/正分类权重 1.000000；完整 Valid MCC 选模（不舍入，同分取早）；固定 probability >0.5。

PLM Baseline 使用 CLS 和标准 RobertaClassificationHead（Dense→Tanh→1，标准 head Dropout）。参照 [LOSVER run_base.py](https://github.com/waroad/losver/blob/a158905cb45b4400f481f18eeb1cb4473ccfccdc/detection/run_base.py)。Context-only 和 RGCL 的编码器、参数及 K=5 Token 加权有界上下文 MIL 完全一致；只后者加 0.1 原 Diff 区域损失。无可靠区域、截断修改行和无正/背景区域的函数仍保留 BCE。MegaVul 的修复对侧不自动转成新负例。

| Split | Method | Epoch | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC | 正常 FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | plm_baseline | 2 | 0.7394 | 0.7510 | 0.7164 | 0.7333 | 0.4794 | 0.8213 | 0.8283 | 0.2375 |
| valid | context_only | 2 | 0.7528 | 0.7342 | 0.7925 | 0.7622 | 0.5072 | 0.8328 | 0.8366 | 0.2870 |
| valid | rgcl | 2 | 0.7486 | 0.7180 | 0.8187 | 0.7651 | 0.5022 | 0.8317 | 0.8343 | 0.3215 |
| test | plm_baseline | 2 | 0.7204 | 0.7497 | 0.6617 | 0.7029 | 0.4438 | 0.7973 | 0.8049 | 0.2209 |
| test | context_only | 2 | 0.7240 | 0.7179 | 0.7379 | 0.7278 | 0.4481 | 0.7992 | 0.8036 | 0.2899 |
| test | rgcl | 2 | 0.7181 | 0.7010 | 0.7607 | 0.7297 | 0.4379 | 0.7955 | 0.8004 | 0.3244 |

| Split | Method | 正常类型 | 函数 | 误报数 | FPR |
|---|---|---|---:|---:|---:|
| valid | plm_baseline | ordinary | 1798 | 427 | 0.2375 |
| valid | plm_baseline | repaired | 0 | 0 | — |
| valid | context_only | ordinary | 1798 | 516 | 0.2870 |
| valid | context_only | repaired | 0 | 0 | — |
| valid | rgcl | ordinary | 1798 | 578 | 0.3215 |
| valid | rgcl | repaired | 0 | 0 | — |
| test | plm_baseline | ordinary | 1797 | 397 | 0.2209 |
| test | plm_baseline | repaired | 0 | 0 | — |
| test | context_only | ordinary | 1797 | 521 | 0.2899 |
| test | context_only | repaired | 0 | 0 | — |
| test | rgcl | ordinary | 1797 | 583 | 0.3244 |
| test | rgcl | repaired | 0 | 0 | — |

| Method | 参数量 | 完整 Epoch | 优化步数 | 总耗时（分钟） |
|---|---:|---:|---:|---:|
| plm_baseline | 125930497 | 16 | 57520 | 146.61 |
| context_only | 125389826 | 16 | 57520 | 148.40 |
| rgcl | 125389826 | 16 | 57520 | 165.10 |

valid RGCL vs plm_baseline: MCC +0.0228，Recall +10.23pp，正常 FPR +8.40pp，AUC +0.0104。
valid RGCL vs context_only: MCC -0.0050，Recall +2.61pp，正常 FPR +3.45pp，AUC -0.0011。

test RGCL vs plm_baseline: MCC -0.0059，Recall +9.91pp，正常 FPR +10.35pp，AUC -0.0018。
test RGCL vs context_only: MCC -0.0103，Recall +2.28pp，正常 FPR +3.45pp，AUC -0.0037。

Valid/Test 推理只读取单函数源码，不读取 Diff、修复版本或外部上下文。Test 在完整训练和 Valid 选模后评价，不参与调参。

第二种子触发规则在结果之前固定：RGCL 的 Valid MCC 相比两个对照均提高至少 0.01，再完整复现三组；否则报告单种子结果，不声称稳定收益。
