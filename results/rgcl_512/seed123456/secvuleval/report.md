# secvuleval: RGCL 原生函数二分类

原始标签与全部成员保留；同 CVE、提交、修复关联和空白归一化重复代码合并后，Seed42 按 8:1:1 目标分组划分。本地发布文件未附独立官方评价成员。

| Split | 函数 | 漏洞 | 正常 | 普通正常 | 修复后正常 | 有可靠对侧 | 有效可见区域 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 20352 | 8798 | 11554 | 3611 | 7943 | 15886 | 12489 (61.4%) |
| valid | 2544 | 1100 | 1444 | 406 | 1038 | 2076 | 1636 (64.3%) |
| test | 2544 | 1100 | 1444 | 401 | 1043 | 2086 | 1604 (63.1%) |

区域覆盖率只作数据统计；Diff 辅助损失只用于 Train。PR-AUC 沿用现有 average_precision_score（AP）口径。

预训练 UniXcoder；512 Token；Seed123456；Batch8；16 Epoch；LOSVERAdamW lr=2e-05、epsilon=1e-08，warmup=0.1，FP32。三组相同 Train 负/正分类权重 1.313253；完整 Valid MCC 选模（不舍入，同分取早）；固定 probability >0.5。

PLM Baseline 使用 CLS 和标准 RobertaClassificationHead（Dense→Tanh→1，标准 head Dropout）。参照 [LOSVER run_base.py](https://github.com/waroad/losver/blob/a158905cb45b4400f481f18eeb1cb4473ccfccdc/detection/run_base.py)。Context-only 和 RGCL 的编码器、参数及 K=5 Token 加权有界上下文 MIL 完全一致；只后者加 0.1 原 Diff 区域损失。无可靠区域、截断修改行和无正/背景区域的函数仍保留 BCE。MegaVul 的修复对侧不自动转成新负例。

| Split | Method | Epoch | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC | 正常 FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | plm_baseline | 10 | 0.5778 | 0.5164 | 0.3727 | 0.4329 | 0.1142 | 0.5706 | 0.4920 | 0.2659 |
| valid | context_only | 4 | 0.4858 | 0.4525 | 0.9009 | 0.6024 | 0.1010 | 0.5705 | 0.4865 | 0.8303 |
| valid | rgcl | 4 | 0.5024 | 0.4607 | 0.8845 | 0.6059 | 0.1264 | 0.5745 | 0.4906 | 0.7888 |
| test | plm_baseline | 10 | 0.5633 | 0.4935 | 0.3818 | 0.4305 | 0.0875 | 0.5842 | 0.4900 | 0.2985 |
| test | context_only | 4 | 0.5016 | 0.4610 | 0.9036 | 0.6106 | 0.1363 | 0.5933 | 0.5026 | 0.8047 |
| test | rgcl | 4 | 0.5157 | 0.4690 | 0.9073 | 0.6183 | 0.1671 | 0.5940 | 0.4929 | 0.7825 |

| Split | Method | 正常类型 | 函数 | 误报数 | FPR |
|---|---|---|---:|---:|---:|
| valid | plm_baseline | ordinary | 406 | 81 | 0.1995 |
| valid | plm_baseline | repaired | 1038 | 303 | 0.2919 |
| valid | context_only | ordinary | 406 | 305 | 0.7512 |
| valid | context_only | repaired | 1038 | 894 | 0.8613 |
| valid | rgcl | ordinary | 406 | 281 | 0.6921 |
| valid | rgcl | repaired | 1038 | 858 | 0.8266 |
| test | plm_baseline | ordinary | 401 | 81 | 0.2020 |
| test | plm_baseline | repaired | 1043 | 350 | 0.3356 |
| test | context_only | ordinary | 401 | 287 | 0.7157 |
| test | context_only | repaired | 1043 | 875 | 0.8389 |
| test | rgcl | ordinary | 401 | 267 | 0.6658 |
| test | rgcl | repaired | 1043 | 863 | 0.8274 |

| Method | 参数量 | 完整 Epoch | 优化步数 | 总耗时（分钟） |
|---|---:|---:|---:|---:|
| plm_baseline | 125930497 | 16 | 40704 | 118.84 |
| context_only | 125389826 | 16 | 40704 | 121.52 |
| rgcl | 125389826 | 16 | 40704 | 111.04 |

valid RGCL vs plm_baseline: MCC +0.0122，Recall +51.18pp，正常 FPR +52.29pp，AUC +0.0039。
valid RGCL vs context_only: MCC +0.0253，Recall -1.64pp，正常 FPR -4.16pp，AUC +0.0040。

test RGCL vs plm_baseline: MCC +0.0796，Recall +52.55pp，正常 FPR +48.41pp，AUC +0.0098。
test RGCL vs context_only: MCC +0.0308，Recall +0.36pp，正常 FPR -2.22pp，AUC +0.0007。

Valid/Test 推理只读取单函数源码，不读取 Diff、修复版本或外部上下文。Test 在完整训练和 Valid 选模后评价，不参与调参。

第二种子触发规则在结果之前固定：RGCL 的 Valid MCC 相比两个对照均提高至少 0.01，再完整复现三组；否则报告单种子结果，不声称稳定收益。
