# RGCL 跨数据集函数二分类：真实实验结果

SecVulEval 保留全量原始成员；MegaVul 仅 C/C++，先对全量原始成员分组划分，再在各集合中用 Seed42 抽取等量原始正常函数，保留全部有效漏洞函数。没有从修复版本新建分类负例，也没有做成对评价。

三组共享正式数据、训练和评价入口，均从同一本地 UniXcoder 预训练权重重新初始化。512 Token、Batch8、16 Epoch、训练 Seed123456；LOSVERAdamW lr=2e-5、epsilon=1e-8、weight_decay=0、warmup=.1，FP32。完整 Valid MCC 选模，不舍入，同分取早；固定 probability >0.5。
PLM Baseline 使用 CLS + 标准 RobertaClassificationHead（Dense→Tanh→1，.1 Dropout），参照 [LOSVER run_base.py](https://github.com/waroad/losver/blob/a158905cb45b4400f481f18eeb1cb4473ccfccdc/detection/run_base.py)。Context-only 与 RGCL 参数、初始化、上下文有界 MIL 完全相同，只有 RGCL 加 .1 原 Token 加权 Diff 区域损失，K=5。
第二种子规则在模型结果之前固定：Seed123456 的 RGCL Valid MCC 比两个对照均高至少 .01，才完整复现三组 Seed123457；最多两个种子。此决定先于 Test 评价。SecVulEval 触发，MegaVul 未触发。PR-AUC 沿用现有 AP 口径。

| 数据集 | Split | 漏洞 | 正常 | 普通正常 | 原生修复后正常 | 可靠对侧 | 有效可见区域 |
|---|---|---:|---:|---:|---:|---:|---:|
| secvuleval | train | 8798 | 11554 | 3611 | 7943 | 15886 | 12489 / 20352 (61.36%) |
| secvuleval | valid | 1100 | 1444 | 406 | 1038 | 2076 | 1636 / 2544 (64.31%) |
| secvuleval | test | 1100 | 1444 | 401 | 1043 | 2086 | 1604 / 2544 (63.05%) |
| megavul | train | 14380 | 14380 | 14380 | 0 | 14380 | 7843 / 28760 (27.27%) |
| megavul | valid | 1798 | 1798 | 1798 | 0 | 1798 | 969 / 3596 (26.95%) |
| megavul | test | 1797 | 1797 | 1797 | 0 | 1797 | 967 / 3594 (26.91%) |

区域覆盖率是数据统计，只有 Train 使用辅助监督。无可靠区域、截断修改行、空区域或全区域的函数保留原分类 BCE。Valid/Test 评价只读单函数源码，不读 Diff、修复版本或参考库。
SecVulEval 原始 25,440 个函数（10,998 漏洞 / 14,442 正常）全部保留，974 个漏洞函数缺可靠修复关联仍参与分类。MegaVul 原始 353,873 个函数（17,975 漏洞 / 335,898 正常），选中 35,950 个，移除的 317,923 个全是原始正常函数；原生修复后正常函数为 0。
分组使用 CVE、修复提交、原生修复关联和空白归一化重复源码（含对侧）。SecVulEval 5,605 组，MegaVul 6,895 组。独立复查全部 MegaVul 原始记录后，跨集合 CVE、提交、重复代码关系重叠为 0；原始标签和源码未变，逐集合 Seed42 抽样可复现。
三组分类正例权重均取 Train 负/正：SecVulEval 1.313253012；MegaVul 1.0。实际成员见 [SecVulEval members](../../data/rgcl/secvuleval/members.csv)、[MegaVul members](../../data/rgcl/megavul/members.csv)；全量分组及入选标记见 [grouped_members.csv](../../data/rgcl/megavul/grouped_members.csv)。

| 数据集 | Seed | Split | Method | Epoch | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC (AP) | 正常 FPR |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| secvuleval | 123456 | valid | plm_baseline | 10 | 0.5778 | 0.5164 | 0.3727 | 0.4329 | 0.1142 | 0.5706 | 0.4920 | 0.2659 |
| secvuleval | 123456 | valid | context_only | 4 | 0.4858 | 0.4525 | 0.9009 | 0.6024 | 0.1010 | 0.5705 | 0.4865 | 0.8303 |
| secvuleval | 123456 | valid | rgcl | 4 | 0.5024 | 0.4607 | 0.8845 | 0.6059 | 0.1264 | 0.5745 | 0.4906 | 0.7888 |
| secvuleval | 123456 | test | plm_baseline | 10 | 0.5633 | 0.4935 | 0.3818 | 0.4305 | 0.0875 | 0.5842 | 0.4900 | 0.2985 |
| secvuleval | 123456 | test | context_only | 4 | 0.5016 | 0.4610 | 0.9036 | 0.6106 | 0.1363 | 0.5933 | 0.5026 | 0.8047 |
| secvuleval | 123456 | test | rgcl | 4 | 0.5157 | 0.4690 | 0.9073 | 0.6183 | 0.1671 | 0.5940 | 0.4929 | 0.7825 |
| secvuleval | 123457 | valid | plm_baseline | 3 | 0.5252 | 0.4671 | 0.6973 | 0.5594 | 0.0946 | 0.5735 | 0.5004 | 0.6060 |
| secvuleval | 123457 | valid | context_only | 6 | 0.5487 | 0.4818 | 0.5791 | 0.5260 | 0.1038 | 0.5827 | 0.5039 | 0.4744 |
| secvuleval | 123457 | valid | rgcl | 6 | 0.5487 | 0.4829 | 0.6173 | 0.5419 | 0.1134 | 0.5837 | 0.5037 | 0.5035 |
| secvuleval | 123457 | test | plm_baseline | 3 | 0.5346 | 0.4723 | 0.6518 | 0.5477 | 0.0981 | 0.5775 | 0.5030 | 0.5547 |
| secvuleval | 123457 | test | context_only | 6 | 0.5590 | 0.4920 | 0.6182 | 0.5479 | 0.1313 | 0.5929 | 0.4967 | 0.4861 |
| secvuleval | 123457 | test | rgcl | 6 | 0.5535 | 0.4883 | 0.6836 | 0.5697 | 0.1398 | 0.5992 | 0.5016 | 0.5457 |
| megavul | 123456 | valid | plm_baseline | 2 | 0.7394 | 0.7510 | 0.7164 | 0.7333 | 0.4794 | 0.8213 | 0.8283 | 0.2375 |
| megavul | 123456 | valid | context_only | 2 | 0.7528 | 0.7342 | 0.7925 | 0.7622 | 0.5072 | 0.8328 | 0.8366 | 0.2870 |
| megavul | 123456 | valid | rgcl | 2 | 0.7486 | 0.7180 | 0.8187 | 0.7651 | 0.5022 | 0.8317 | 0.8343 | 0.3215 |
| megavul | 123456 | test | plm_baseline | 2 | 0.7204 | 0.7497 | 0.6617 | 0.7029 | 0.4438 | 0.7973 | 0.8049 | 0.2209 |
| megavul | 123456 | test | context_only | 2 | 0.7240 | 0.7179 | 0.7379 | 0.7278 | 0.4481 | 0.7992 | 0.8036 | 0.2899 |
| megavul | 123456 | test | rgcl | 2 | 0.7181 | 0.7010 | 0.7607 | 0.7297 | 0.4379 | 0.7955 | 0.8004 | 0.3244 |

| 数据集 | Split | Method | MCC 均值 ± 样本标准差 | F1 均值 | ROC-AUC 均值 | PR-AUC 均值 | Recall 均值 | 正常 FPR 均值 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| secvuleval | valid | plm_baseline | 0.1044 ± 0.0139 | 0.4962 | 0.5721 | 0.4962 | 0.5350 | 0.4359 |
| secvuleval | valid | context_only | 0.1024 ± 0.0020 | 0.5642 | 0.5766 | 0.4952 | 0.7400 | 0.6524 |
| secvuleval | valid | rgcl | 0.1199 ± 0.0092 | 0.5739 | 0.5791 | 0.4972 | 0.7509 | 0.6461 |
| secvuleval | test | plm_baseline | 0.0928 ± 0.0075 | 0.4891 | 0.5808 | 0.4965 | 0.5168 | 0.4266 |
| secvuleval | test | context_only | 0.1338 ± 0.0035 | 0.5793 | 0.5931 | 0.4996 | 0.7609 | 0.6454 |
| secvuleval | test | rgcl | 0.1534 ± 0.0193 | 0.5940 | 0.5966 | 0.4973 | 0.7955 | 0.6641 |
| megavul | valid | plm_baseline | 0.4794（单种子） | 0.7333 | 0.8213 | 0.8283 | 0.7164 | 0.2375 |
| megavul | valid | context_only | 0.5072（单种子） | 0.7622 | 0.8328 | 0.8366 | 0.7925 | 0.2870 |
| megavul | valid | rgcl | 0.5022（单种子） | 0.7651 | 0.8317 | 0.8343 | 0.8187 | 0.3215 |
| megavul | test | plm_baseline | 0.4438（单种子） | 0.7029 | 0.7973 | 0.8049 | 0.6617 | 0.2209 |
| megavul | test | context_only | 0.4481（单种子） | 0.7278 | 0.7992 | 0.8036 | 0.7379 | 0.2899 |
| megavul | test | rgcl | 0.4379（单种子） | 0.7297 | 0.7955 | 0.8004 | 0.7607 | 0.3244 |

SecVulEval：两个种子中 RGCL 的 Valid/Test MCC 都高于两个对照。相对 Context-only，平均 Valid MCC +0.01745、Test MCC +0.01962；平均 Test AUC 只增加 0.00350，AP 反而降低 0.00236。Test Recall 平均提高 3.45pp，正常 FPR 也提高 1.87pp。MCC 收益存在，但绝对 AUC 约 .60、误报仍高，不能称为全面改善。
MegaVul：RGCL Valid MCC .5022 低于 Context-only .5072；Test MCC .4379 也低于 Context-only .4481 和 CLS .4438。相对 Context-only，Test Recall +2.28pp、FPR +3.45pp，F1 仅 +0.19pp，MCC −0.01027、AUC −0.00366、AP −0.00320。没有证明修复监督的额外检测收益；未追加第二种子、参数搜索或新模块。
区域损失确实参与学习：MegaVul RGCL 训练 BCE 从第一 Epoch .56734 降至第十六 Epoch .00245，区域损失从 .52878 降至 .00366；所选 checkpoint 为 Epoch2。降低训练区域损失没有自动转化为更好的函数分类泛化。
结论：本轮只支持 SecVulEval 上有限的 MCC 收益，未建立“修复引导上下文学习在多个数据集上稳定优于 CLS 与 Context-only”的结论。MegaVul 是明确负结果。

相关 71 项单元检查最终通过，覆盖原始标签、分组、抽样、Diff 可见性、标准 CLS 头、初始化/RNG、梯度、恢复和原入口训练；初始 native bounded-attention 方法名校验拒绝已在共享入口修复。两套真实 RTX4090 GPU smoke 均通过。九组完整训练均正常退出，全部完成 16 Epoch；SecVulEval 每组 40,704 步、MegaVul 每组 57,520 步。
完成后核对三组 CPU 采样/Python/NumPy RNG 一致，Context-only/RGCL CUDA RNG 一致；Context-only/RGCL 同种子首步 BCE 完全相同。首个 MegaVul 批次无有效区域，初始区域损失为 0，后续真实区域损失及梯度正常。选定 RGCL checkpoint 的真实 GPU 单函数 CLI 与直接推理概率完全一致，改变诊断 Diff 标签不改变输出。
真实运行问题：SecVulEval 的训练、Test 和报告均完成后，编排进度临时 JSON 在重命名时不存在，编排中断。改用唯一临时文件后从 MegaVul 准备阶段续跑，未重跑或更改任何已完成模型。额外推理检查脚本曾错误假定纯源码批次含 line_labels；修正检查脚本后通过，正式模型无需修改。完整记录见 [tests.log](tests.log)、[pipeline.log](pipeline.log) 和 [commands.json](commands.json)。

逐种子完整报告：[SecVulEval 123456](seed123456/secvuleval/report.md)、[SecVulEval 123457](seed123457/secvuleval/report.md)、[MegaVul 123456](seed123456/megavul/report.md)。各方法保留完整配置、16 Epoch 日志、所选轮次和 Valid/Test 逐函数概率及行注意力；权重和原始源码仅保留本地。
