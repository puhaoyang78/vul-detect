# secvuleval: RGCL 原生函数二分类

原始标签与全部成员保留；同 CVE、提交、修复关联和空白归一化重复代码合并后，Seed42 按 8:1:1 目标分组划分。本地发布文件未附独立官方评价成员。

| Split | 函数 | 漏洞 | 正常 | 普通正常 | 修复后正常 | 有可靠对侧 | 有效可见区域 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 20352 | 8798 | 11554 | 3611 | 7943 | 15886 | 12489 (61.4%) |
| valid | 2544 | 1100 | 1444 | 406 | 1038 | 2076 | 1636 (64.3%) |
| test | 2544 | 1100 | 1444 | 401 | 1043 | 2086 | 1604 (63.1%) |

区域覆盖率只作数据统计；Diff 辅助损失只用于 Train。PR-AUC 沿用现有 average_precision_score（AP）口径。

预训练 UniXcoder；512 Token；Seed123457；Batch8；16 Epoch；LOSVERAdamW lr=2e-05、epsilon=1e-08，warmup=0.1，FP32。三组相同 Train 负/正分类权重 1.313253；完整 Valid MCC 选模（不舍入，同分取早）；固定 probability >0.5。

PLM Baseline 使用 CLS 和标准 RobertaClassificationHead（Dense→Tanh→1，标准 head Dropout）。参照 [LOSVER run_base.py](https://github.com/waroad/losver/blob/a158905cb45b4400f481f18eeb1cb4473ccfccdc/detection/run_base.py)。Context-only 和 RGCL 的编码器、参数及 K=5 Token 加权有界上下文 MIL 完全一致；只后者加 0.1 原 Diff 区域损失。无可靠区域、截断修改行和无正/背景区域的函数仍保留 BCE。MegaVul 的修复对侧不自动转成新负例。

| Split | Method | Epoch | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC | 正常 FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | plm_baseline | 3 | 0.5252 | 0.4671 | 0.6973 | 0.5594 | 0.0946 | 0.5735 | 0.5004 | 0.6060 |
| valid | context_only | 6 | 0.5487 | 0.4818 | 0.5791 | 0.5260 | 0.1038 | 0.5827 | 0.5039 | 0.4744 |
| valid | rgcl | 6 | 0.5487 | 0.4829 | 0.6173 | 0.5419 | 0.1134 | 0.5837 | 0.5037 | 0.5035 |
| test | plm_baseline | 3 | 0.5346 | 0.4723 | 0.6518 | 0.5477 | 0.0981 | 0.5775 | 0.5030 | 0.5547 |
| test | context_only | 6 | 0.5590 | 0.4920 | 0.6182 | 0.5479 | 0.1313 | 0.5929 | 0.4967 | 0.4861 |
| test | rgcl | 6 | 0.5535 | 0.4883 | 0.6836 | 0.5697 | 0.1398 | 0.5992 | 0.5016 | 0.5457 |

| Split | Method | 正常类型 | 函数 | 误报数 | FPR |
|---|---|---|---:|---:|---:|
| valid | plm_baseline | ordinary | 406 | 194 | 0.4778 |
| valid | plm_baseline | repaired | 1038 | 681 | 0.6561 |
| valid | context_only | ordinary | 406 | 150 | 0.3695 |
| valid | context_only | repaired | 1038 | 535 | 0.5154 |
| valid | rgcl | ordinary | 406 | 163 | 0.4015 |
| valid | rgcl | repaired | 1038 | 564 | 0.5434 |
| test | plm_baseline | ordinary | 401 | 188 | 0.4688 |
| test | plm_baseline | repaired | 1043 | 613 | 0.5877 |
| test | context_only | ordinary | 401 | 150 | 0.3741 |
| test | context_only | repaired | 1043 | 552 | 0.5292 |
| test | rgcl | ordinary | 401 | 172 | 0.4289 |
| test | rgcl | repaired | 1043 | 616 | 0.5906 |

| Method | 参数量 | 完整 Epoch | 优化步数 | 总耗时（分钟） |
|---|---:|---:|---:|---:|
| plm_baseline | 125930497 | 16 | 40704 | 119.59 |
| context_only | 125389826 | 16 | 40704 | 122.53 |
| rgcl | 125389826 | 16 | 40704 | 121.69 |

valid RGCL vs plm_baseline: MCC +0.0188，Recall -8.00pp，正常 FPR -10.25pp，AUC +0.0102。
valid RGCL vs context_only: MCC +0.0096，Recall +3.82pp，正常 FPR +2.91pp，AUC +0.0010。

test RGCL vs plm_baseline: MCC +0.0417，Recall +3.18pp，正常 FPR -0.90pp，AUC +0.0217。
test RGCL vs context_only: MCC +0.0085，Recall +6.55pp，正常 FPR +5.96pp，AUC +0.0063。

Valid/Test 推理只读取单函数源码，不读取 Diff、修复版本或外部上下文。Test 在完整训练和 Valid 选模后评价，不参与调参。

第二种子触发规则在结果之前固定：RGCL 的 Valid MCC 相比两个对照均提高至少 0.01，再完整复现三组；否则报告单种子结果，不声称稳定收益。
