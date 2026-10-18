# Hardest / Boundary 与 VAT 方向消融

Seed 42；primevul 固定平衡子集 Train/Valid/Test = 1024/256/256，各 Split 正负各半。各组从同一预训练 CodeBERT 初始化，3 Epoch、有效 batch 8、384 个优化器步；仅源 Valid 选择 Epoch 和阈值，素材仅取实际 Train。旧权重和正式 Test 不变。

## 同一真实 checkpoint 的计算比较

checkpoint: `results/primevul/bce/best.pt`；平衡源 Train 32 条，3 次交替顺序测量的耗时中位数。窗口/token 计显式前向，不计梯度检查点反向重计算；VAT 耗时包含方向反向和最终 KL 反向。

| 搜索 | 变体查询 | 原函数查询 | 窗口 | Token slots（含 padding） | 秒 |
|---|---:|---:|---:|---:|---:|
| hardest | 234 | 21 | 469 | 203971 | 3.444 |
| boundary | 178 | 21 | 406 | 180818 | 2.973 |

| VAT | 函数前向 | 窗口 | Token slots | 秒 | 平均 KL | 中位 KL |
|---|---:|---:|---:|---:|---:|---:|
| finite_difference | 96 | 213 | 106320 | 2.608 | 0.332089 | 0.0792629 |
| logit | 64 | 142 | 70880 | 2.333 | 0.331985 | 0.0792631 |

Signed cosine：n=32，中位数 1.00000000，均值 0.81250000，最小值 -1.00000000，负值 3 条。按每函数展开后的扰动计算。逐函数半径、KL、微小探针 logit 变化见 `probe.vat.csv`；参数梯度有限性和零方向计数见 `probe.json`。

反例（保持符号，不取绝对 cosine）：

| 样本 | Signed cosine | FD KL | Logit KL | FD 探针 Δlogit | xi·gᵀd |
|---|---:|---:|---:|---:|---:|
| primevul:514 | -1.00000000 | 0.00469095 | 0.00493974 | 3.77e-07 | -1.7e-08 |
| primevul:4366 | -1.00000000 | 0.0034532 | 1.70282e-05 | -2.06e-07 | 4.17e-07 |
| primevul:171044 | -1.00000000 | 2.67302e-05 | 0.000218209 | -2.25e-07 | 1.46e-07 |

两分支中心 logit 的最大差为 0。上述反例的微小探针增量与中心一阶预测反号，说明当前 xi 下有限差分方向与中心 Hessian 方向存在差异；这些测量尚未分离浮点误差与局部非线性的贡献。epsilon=1 的最终 KL 也不保证由中心二阶近似排序。

边界策略选样原因：`{"no_boundary_max_bce": 15, "no_legal_candidate": 11, "original_wrong_random": 3, "first_boundary_min_bce": 3}`。无合法候选时只训练原函数；原函数已错的随机变体不附加 logit/loss。零梯度/零投影不使用替代方向；单元测试覆盖零方向退化。有限差分未计算原始 gᵀd，其零投影计数不适用。

## 相同训练预算的学习效果

选中模型对应更新步数依次为 256、256、256、384；均沿用源 Valid 选模。

| 策略 | VAT | Epoch | 阈值 | Clean Acc | F1 | MCC | AUC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hardest | finite_difference | 2 | 0.60 | 0.7539 | 0.7623 | 0.5091 | 0.7997 | 0.7581 |
| hardest | logit | 2 | 0.64 | 0.7539 | 0.7640 | 0.5097 | 0.7983 | 0.7562 |
| boundary | finite_difference | 2 | 0.59 | 0.7578 | 0.7704 | 0.5187 | 0.7999 | 0.7564 |
| boundary | logit | 3 | 0.51 | 0.7539 | 0.7640 | 0.5097 | 0.7983 | 0.7453 |

### 严格语义保持

| 策略 | VAT | 扰动 | Worst Acc | Worst MCC | ASR（分子/分母） | 可扰动/总数 |
|---|---|---|---:|---:|---:|---:|
| hardest | finite_difference | fixed | 0.7383 | 0.4777 | 3.42% (4/117) | 160/256 |
| hardest | finite_difference | heldout | 0.7344 | 0.4697 | 4.27% (5/117) | 159/256 |
| hardest | finite_difference | adaptive | 0.7266 | 0.4540 | 5.93% (7/118) | 161/256 |
| hardest | logit | fixed | 0.7344 | 0.4708 | 4.27% (5/117) | 160/256 |
| hardest | logit | heldout | 0.7344 | 0.4702 | 4.27% (5/117) | 159/256 |
| hardest | logit | adaptive | 0.7188 | 0.4388 | 7.63% (9/118) | 161/256 |
| boundary | finite_difference | fixed | 0.7383 | 0.4799 | 4.20% (5/119) | 160/256 |
| boundary | finite_difference | heldout | 0.7383 | 0.4799 | 4.20% (5/119) | 159/256 |
| boundary | finite_difference | adaptive | 0.7148 | 0.4327 | 9.17% (11/120) | 161/256 |
| boundary | logit | fixed | 0.7383 | 0.4790 | 3.42% (4/117) | 160/256 |
| boundary | logit | heldout | 0.7383 | 0.4790 | 3.42% (4/117) | 159/256 |
| boundary | logit | adaptive | 0.7305 | 0.4633 | 5.08% (6/118) | 161/256 |

### 函数名等输入压力测试

| 策略 | VAT | 扰动 | Worst Acc | Worst MCC | ASR（分子/分母） | 可扰动/总数 |
|---|---|---|---:|---:|---:|---:|
| hardest | finite_difference | fixed | 0.7500 | 0.5010 | 1.69% (1/59) | 83/256 |
| hardest | finite_difference | heldout | 0.7539 | 0.5091 | 0.00% (0/59) | 83/256 |
| hardest | finite_difference | adaptive | 0.7422 | 0.4853 | 5.08% (3/59) | 83/256 |
| hardest | logit | fixed | 0.7539 | 0.5097 | 0.00% (0/60) | 83/256 |
| hardest | logit | heldout | 0.7539 | 0.5097 | 0.00% (0/60) | 83/256 |
| hardest | logit | adaptive | 0.7344 | 0.4702 | 8.33% (5/60) | 83/256 |
| boundary | finite_difference | fixed | 0.7578 | 0.5187 | 0.00% (0/61) | 83/256 |
| boundary | finite_difference | heldout | 0.7578 | 0.5187 | 0.00% (0/61) | 83/256 |
| boundary | finite_difference | adaptive | 0.7383 | 0.4790 | 8.20% (5/61) | 83/256 |
| boundary | logit | fixed | 0.7539 | 0.5097 | 0.00% (0/58) | 83/256 |
| boundary | logit | heldout | 0.7500 | 0.5022 | 1.72% (1/58) | 83/256 |
| boundary | logit | adaptive | 0.7422 | 0.4873 | 5.17% (3/58) | 83/256 |

ASR 分母为该扰动范围内可扰动且 Clean 正确的函数；Worst-case 包含原函数和全部 Test 成员。四组合固定/留出候选相同，Adaptive 保留完整 48 候选预算，不启用 Boundary。

计算收益依据固定 checkpoint 探针；学习效果依据四次独立训练。训练 epoch 耗时记录于结果，可能受到并行运行的资源竞争影响，不作为隔离速度基准。该单 Seed 小样本消融不能代替完整数据实验。
