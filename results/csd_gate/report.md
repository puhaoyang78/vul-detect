# CSD 共享系数门控的最后一次验证

两个门控 CSD 已从相同预训练 CodeBERT 分别完成 3 Epoch。项目残差的幅度和 BCE 作用有所增加，但未形成稳定的额外排名能力：扣除纯门控缩放后，PrimeVul / SecVulEval 的源 Train 原始 Logit AUC 分别下降 0.000283 / 0.008095。跨域 AUC 相对无门控 mean 分别提高 0.003992 / 0.050931，但绝对值仅为 0.471923 / 0.504050，两个跨域 MCC 仍为负。不能据此认定有效的项目分解已带来实质性跨域泛化收益。本轮不继续叠加 CSD 变体。

## 唯一方法改动与固定协议

参考[作者分类头实现](https://github.com/vihari/CSD/blob/master/pacs/models/resnet_csd.py#L65-L76)，在现有 mean CSD 的 Specific 中加入一个全局可学习标量：

```text
g_init = 0.1
alpha = tanh(g)
Common(h)   = W_c h + b_c
Residual_p  = a_p (U h + b_U)
Specific_p  = alpha * Common(h) + Residual_p
```

共享 weight 和 bias 一起缩放；Common 输出本身不变。Valid、Test 和推理只使用 Common。按本次控制变量要求，保留现有未变换的项目系数 a_p，不同时引入作者代码对项目系数的 tanh，也不改变初始化分布。g 使用指定常数 0.1，不添加随机噪声；其创建不消耗随机数，新旧原有参数的重建初始化统计逐项相同。原有 mean Gram 正则、rank=1、λ=κ=1、偏置分解不变；g 不进入 Gram 正则，仍由原有 AdamW 优化。

Seed 42、3 Epoch、micro batch 4、梯度累积 2、lr=1e-5、weight decay=0.01、gradient clip=1、2048 逻辑输入、原有窗口/池化、FP32 参数和 BF16 CUDA 编码器均不变，无增强或 VAT。使用 `/home/PublicData/PHY-data/resource/codebert-base`，PyTorch 2.9.1+cu128、Transformers 4.57.3、RTX 4090。两次训练均从预训练模型初始化，未加载旧微调权重。

PrimeVul / SecVulEval 的 Train 为 9,724 / 17,596 条，实际选模 Valid 为 1,186 / 2,198 条；优化器更新 3,648 / 6,600 次。仅自身 Clean Valid 选择 Epoch 和阈值：阈值按 MCC/F1/Accuracy/距 0.5，Epoch 按 MCC/F1/Accuracy/AUC 排序。两个模型均选中第 3 Epoch，阈值分别为 0.38 / 0.54。未使用目标域数据训练、调参或校准。

正式比较沿用既有独立 Test：PrimeVul 911（369 正 / 542 负）、SecVulEval 2,070（986 正 / 1,084 负）。只排除与两库实际 Train/选模 Valid 完全相同的源码，排除数仍为 187 / 132。原完整 Test 1,098 / 2,202 条的参考结果也保存。未重新准备、采样或改变成员；两份成员 CSV 与原结果逐字节相同。BCE、sum、无门控 mean 的权重及结果未覆盖，本次直接读取其已保存预测。

## 相同独立 Test 上的完整结果

下表均为共享头、各自源域 Valid 阈值。四组对照 × 四个方向 × 完整/独立队列共 32 行见 [comparison.csv](comparison.csv)，包含 Epoch、阈值、类别数、混淆计数、域内外差值及相对 BCE/sum/mean 的差值。门控实验的原生配置及结果见 [results.json](transfer_primevul_secvuleval/results.json)。

| 训练 → Test | 对照 | Accuracy | F1 | MCC | AUC | AUPRC |
|---|---|---:|---:|---:|---:|---:|
| PrimeVul → PrimeVul | BCE | 0.770582 | 0.728218 | 0.531587 | 0.844958 | 0.760188 |
| PrimeVul → PrimeVul | CSD sum | 0.742042 | 0.721234 | 0.501325 | 0.829703 | 0.734148 |
| PrimeVul → PrimeVul | CSD mean | 0.767289 | 0.707182 | 0.514489 | 0.839538 | 0.752076 |
| PrimeVul → PrimeVul | CSD gate | 0.774973 | 0.737516 | 0.544153 | 0.843853 | 0.746736 |
| PrimeVul → SecVulEval | BCE | 0.471498 | 0.517637 | -0.047119 | 0.467211 | 0.443664 |
| PrimeVul → SecVulEval | CSD sum | 0.464251 | 0.545305 | -0.057319 | 0.465997 | 0.445130 |
| PrimeVul → SecVulEval | CSD mean | 0.485024 | 0.492381 | -0.026445 | 0.467931 | 0.447597 |
| PrimeVul → SecVulEval | CSD gate | 0.480193 | 0.509572 | -0.032189 | 0.471923 | 0.449083 |
| SecVulEval → SecVulEval | BCE | 0.532850 | 0.623589 | 0.107115 | 0.580023 | 0.539889 |
| SecVulEval → SecVulEval | CSD sum | 0.553623 | 0.546614 | 0.108160 | 0.572149 | 0.543970 |
| SecVulEval → SecVulEval | CSD mean | 0.535749 | 0.358906 | 0.055210 | 0.561804 | 0.530020 |
| SecVulEval → SecVulEval | CSD gate | 0.543961 | 0.243590 | 0.079135 | 0.569664 | 0.532640 |
| SecVulEval → PrimeVul | BCE | 0.407245 | 0.520426 | -0.081274 | 0.337628 | 0.309570 |
| SecVulEval → PrimeVul | CSD sum | 0.446762 | 0.419355 | -0.090391 | 0.410529 | 0.338618 |
| SecVulEval → PrimeVul | CSD mean | 0.520307 | 0.204007 | -0.094957 | 0.453120 | 0.363299 |
| SecVulEval → PrimeVul | CSD gate | 0.553238 | 0.153846 | -0.056966 | 0.504050 | 0.390664 |

门控相对 mean 的源域 Clean AUC 变化为 +0.004315 / +0.007861；相对 BCE 则为 −0.001105 / −0.010359。跨域相对 mean 为 +0.003992 / +0.050931，相对 BCE 为 +0.004712 / +0.166422。确实存在双向 AUC 数值改善，反向改善更大，不能报告成“完全没有改善”。但门控的跨域 AUC 相对自身 Test 仍下降 0.371931 / 0.065614；PrimeVul → SecVulEval 低于 0.5，反向仅略高于 0.5。跨域 F1 为 0.509572 / 0.153846、MCC 为 −0.032189 / −0.056966；SecVulEval 域内 F1 也从 mean 的 0.358906 降至 0.243590。改善没有在各项指标上一致体现，不能据此认定获得了可靠的跨域判别能力。

## 门控轨迹及 Valid 选模

初始化两库均为 g=0.1、tanh(g)≈0.099668。以下是每 Epoch 结束的实际参数和当时 Common Valid 指标，第 3 Epoch 为选中权重。

| 源库 | Epoch | g | tanh(g) | Valid MCC | Valid AUC | 阈值 |
|---|---:|---:|---:|---:|---:|---:|
| PrimeVul | 1 | 0.112589903 | 0.112116560 | 0.522042137 | 0.826127758 | 0.34 |
| PrimeVul | 2 | 0.125431448 | 0.124777764 | 0.537048028 | 0.838529329 | 0.71 |
| PrimeVul | 3 | 0.135744512 | 0.134916857 | 0.551065766 | 0.840608106 | 0.38 |
| SecVulEval | 1 | 0.103153437 | 0.102789119 | 0.042698408 | 0.514866687 | 0.38 |
| SecVulEval | 2 | 0.117676899 | 0.117136709 | 0.076661277 | 0.538092782 | 0.20 |
| SecVulEval | 3 | 0.132324785 | 0.131557837 | 0.080139363 | 0.550803899 | 0.54 |

g 确实参与优化，并未停留在初始化或塌缩为零。逐 Epoch 门控和实际损失保存在 [epochs.csv](csd_diagnostics/epochs.csv)。

## 区分 Logit 缩放与项目判别信息

诊断复用既有实现，仅对全部源域 Train 推理，Seed 42、相同成员和顺序、同一次编码同时得到 Common/Specific，不创建优化器。额外保存 `Gated Common = alpha * Common`；Common → Gated Common 测纯缩放，Gated Common → Specific 隔离项目残差。两次训练的 alpha 均为正，因此纯缩放的原始 Logit AUC 增益严格为 0。标准 AUC 仍沿用 float32 Sigmoid 分数，同时另报原始 Logit AUC，避免将舍入/饱和造成的变化误读为排名收益。

| 源库 | Train 分类器 | BCE | AUC（Sigmoid） | AUC（原始 Logit） |
|---|---|---:|---:|---:|
| PrimeVul | Common | 0.381156833 | 0.955077900 | 0.955174816 |
| PrimeVul | Gated Common | 0.369001222 | 0.955174774 | 0.955174816 |
| PrimeVul | Specific | 0.366448000 | 0.954891471 | 0.954891535 |
| SecVulEval | Common | 0.645056745 | 0.688275704 | 0.688276189 |
| SecVulEval | Gated Common | 0.643298171 | 0.688276157 | 0.688276189 |
| SecVulEval | Specific | 0.642414021 | 0.680180936 | 0.680180916 |

纯缩放使 BCE 降低 0.012155611 / 0.001758574，但不增加原始排名能力。扣除此项后的项目残差 BCE 收益为 0.002553222 / 0.000884150，原始 Logit AUC 增益为 −0.000283281 / −0.008095273。故存在小幅 BCE 改善，但没有额外的全局 Train 排名收益。直接将 Common/Specific 的 BCE 差归于项目残差会高估其作用。

以下比较保留旧 sum/mean 的诊断结果；无门控时其对照就是 Common，门控时为 Gated Common。范数以项目等权统计，Logit 和指标覆盖全部源 Train。

| 源库 | 对照 | 残差权重范数中位数 | 残差/未缩放共享范数 | 平均绝对残差 Logit | 残差 BCE 降低量 | 残差原始 Logit AUC 增益 |
|---|---|---:|---:|---:|---:|---:|
| PrimeVul | sum | 0.002113413 | 0.072402% | 0.002555979 | 0.000130991 | +0.000047358 |
| PrimeVul | mean | 0.002015088 | 0.065845% | 0.002513732 | 0.000111727 | +0.000033948 |
| PrimeVul | gate | 0.002415363 | 0.079704% | 0.055210239 | 0.002553222 | −0.000283281 |
| SecVulEval | sum | 0.001626896 | 0.074868% | 0.005671872 | 0.000138483 | +0.000123526 |
| SecVulEval | mean | 0.001586509 | 0.065428% | 0.003413535 | 0.000087385 | +0.000232092 |
| SecVulEval | gate | 0.001716037 | 0.072808% | 0.017546558 | 0.000884150 | −0.008095273 |

门控相对 mean 的残差/原始共享范数仅增加到约 1.21 / 1.11 倍。若改用缩放后的共享权重作分母，中位数变为 0.590765% / 0.553429%；该额外放大主要来自分母乘以约 0.135 / 0.132，不能当作残差学习增强的证据。Common → Specific 的平均绝对总 Logit 差为 9.046345 / 1.875014，而真正的残差仅为 0.055210 / 0.017547；总差主要是缩放。

在各自源 Valid 阈值下，加入残差（Gated Common → Specific）分别翻转 76/9,724、60/17,596 条，其中纠正/退化为 61/15、53/7。固定 0.5 时分别翻转 80、2,313 条，纠正/退化为 29/51、1,114/1,199。变化依赖阈值和边界附近分布，不能只报告有利阈值的翻转数。

## 项目与优化证据

Train 项目划分不变：PrimeVul 602 个项目，其中 218 个单类别项目（391 条样本）；SecVulEval 303 个项目，其中 75 个单类别项目（126 条样本）。项目正例率与平均有符号残差 Logit 的 Spearman 相关为 0.807121 / 0.885202；项目规模与平均绝对残差 Logit 的相关为 0.489579 / 0.246165。残差非零并反映项目类别构成，但这种关联本身不能证明捕获了项目特有的漏洞判别机制。

在可计算 AUC 的双类别项目内，PrimeVul 384 个项目的残差原始 Logit AUC 变化为 5 个提高、373 个不变、6 个下降；SecVulEval 228 个项目为 19/194/15。按每项目正负样本对数 `n_positive * n_negative` 加权的项目内 AUC 增益仅为 −0.000080189 / +0.000038202。结合全 Train 的负 AUC 增益，不支持“项目特定判别信息有效增强”。

| 源库 | 对照 | 第 3 Epoch Common CE | Specific CE | 实际正则项 | 正则占总损失 | 最终探针 U 正则/分类梯度 |
|---|---|---:|---:|---:|---:|---:|
| PrimeVul | sum | 0.354527 | 0.354399 | 61.564131 | 98.8616% | 8503.20 |
| PrimeVul | mean | 0.356734 | 0.356631 | 8.840567 | 92.5333% | 1335.73 |
| PrimeVul | gate | 0.488813 | 0.427893 | 8.575178 | 90.3422% | 244.44 |
| SecVulEval | sum | 0.653772 | 0.653646 | 15.238649 | 92.0983% | 4683.94 |
| SecVulEval | mean | 0.660600 | 0.660519 | 2.703080 | 67.1706% | 2854.55 |
| SecVulEval | gate | 0.681597 | 0.659293 | 2.433414 | 64.4732% | 1052.09 |

Epoch 损失是训练过程中含 dropout 的平均损失，不等于最终固定 checkpoint 的 Train BCE。梯度探针沿用相同的 8 条 Seed-42 源 Train 样本，关闭 dropout、无参数更新；它只描述这组样本和参数位置。JSON 明确保留 sum 参考梯度，上表对实际 mean 实验除以 8 后计算比值。门控降低了 U 上正则梯度相对分类梯度的比值，但仍为数百至上千倍；不能将单参数块的这一比值等同于整个模型的梯度主导关系。g 的分类梯度非零、Gram 正则梯度为零，与其实际轨迹一致。没有据此修改任何损失或参数。

## 检查、数值记录及复现

6 项直接相关单元测试通过：门控公式及未变参数初始化、CE/正则、两库共享训练与 Common-only 推理、缩放/残差诊断分离、项目统计和只读 Train 诊断。模型/训练改动后运行的命令为：

```bash
python -m unittest tests.test_codebert_source.CodeBERTTests.test_csd_shared_gate_changes_only_specific_common_term tests.test_codebert_source.CodeBERTTests.test_csd_decomposition_cross_entropy_and_orthonormality tests.test_codebert_source.CodeBERTTests.test_csd_trains_both_sources_and_infers_common_head_only -v
python -m unittest tests.test_csd_diagnostics -v
```

两库新旧已有参数初始化统计、训练配置及全部源 Train 成员/标签/项目/顺序一致；新增参数只有 g。27,320 条 Train CSV 可精确复算 Common/Specific 及门控分解指标，905 个项目覆盖全部样本；32 行统一比较的五项指标、混淆计数、样本数和差值均从保存预测精确复算。成员 CSV 逐字节一致，未放宽断言或修改任何评价成员。

PrimeVul Valid 重推的全部指标与选模记录一致。SecVulEval Valid 重推 AUC 为 0.5507989312808982，相比训练记录 0.5508038989866708 差 −4.96770577e−06；AUPRC 差 −4.52011321e−06。Epoch、阈值、Accuracy/F1/MCC、类别数和混淆计数完全一致。保留该微小浮点差异，未重新选模或替换预测；本次没有额外重跑一致性探针。此前无门控 mean 的固定 checkpoint 重推已记录微小 CUDA 归约波动，见 [原记录](../csd_mean/validation_repeatability.json)。本轮 Test 指标的精确复算均以各自实际保存预测为准。

训练、评价和诊断复用同一 CLI，完整命令见 [README](../../README.md#csd-共享门控验证)。本地新 checkpoint 位于 `results/csd_gate/<dataset>/csd/best.pt`，未覆盖任何旧权重。统一结果来自原 `results/csd_mean/comparison.csv` 的 24 行加本次原生评价的 8 行，并按相同 view/source/target 计算相对 BCE/sum/mean 差值。必要配置保存在原生 results.json，参数分布、梯度探针、Train Logit 和逐项目统计见 [诊断目录](csd_diagnostics/summary.csv) 及各库 JSON/CSV。

最终结论：门控确实学动了，残差幅度及小幅 BCE 作用也增强了；但没有获得有效增强的项目特定排名能力。共享头的双向跨域 AUC 有数值提高，反向从 mean 的 0.453120 提高到 0.504050，但绝对判别能力和阈值指标仍不足，不能把这一提高归因于有效项目分解，也不足以支持实质性的跨域泛化收益。到此结束本轮 CSD 验证，不再添加其他变体。
