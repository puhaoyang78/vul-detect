# CSD 正交正则 sum → mean 固定对照

本次只改变正交正则归约。两个 CSD 均从原预训练 CodeBERT 重新训练 3 Epoch，各自 Valid 均选中第 3 Epoch；PrimeVul / SecVulEval 阈值为 0.44 / 0.50，优化器更新数为 3,648 / 6,600。BCE 与 sum checkpoint、数据及旧结果保持原样。

跨域 AUC 相对 sum 在两个方向均提高，但 PrimeVul → SecVulEval 只增加 0.001933；SecVulEval → PrimeVul 增加 0.042590，绝对 AUC 仍为 0.453120。与此同时，SecVulEval 域内 AUC 下降 0.010345。项目残差未增强，不能将这些排名变化直接解释为更有效的项目分解。

## 固定协议

正则从 `sum((W_y W_y^T - I)^2)` 改为 [作者代码的 mean 归约](https://github.com/vihari/CSD/blob/master/pacs/train_csd.py#L80-L94)。二分类、rank=1 时，Gram 误差共 2×2×2=8 个元素，因此同参数处的正则值和梯度均为原来的 1/8。λ=κ=1、共享分类器与低秩项目残差、bias 分解及初始化均保持原实现，没有同时引入作者代码的门控等其他变化。

Seed 42、3 Epoch、micro batch 4、梯度累积 2、lr=1e-5、weight decay=0.01、gradient clip=1、2048 逻辑输入、窗口与池化、FP32 参数和 BF16 CUDA 编码器均不变。两库的新旧配置、项目映射及重建初始化统计逐项一致。仅自身 Clean Valid 按 MCC/F1/Accuracy/AUC 选 Epoch；阈值沿用自身 Valid 的规则，不在目标域重选。

Train / 选模 Valid 数为 PrimeVul 9,724 / 1,186、SecVulEval 17,596 / 2,198。正式 Test 队列沿用原精确重叠排除：PrimeVul 911（369 正 / 542 负），SecVulEval 2,070（986 正 / 1,084 负）。新旧两份成员 CSV 逐字节相同；未重新准备或采样数据。完整 Test 的 1,098 / 2,202 条参考结果也保留在 comparison.csv。

## 相同独立 Test 队列上的完整结果

所有测试仅使用共享分类器。下表为 `independent`；完整/独立两类队列 × 三个对照 × 四个方向共 24 行见 [comparison.csv](comparison.csv)，含 Epoch、阈值、类别数量、混淆计数及相对 BCE/sum 的差值。mean 的原生评价结果见 [transfer results](transfer_primevul_secvuleval/results.json)。

| 训练 → Test | 对照 | Accuracy | F1 | MCC | AUC | AUPRC |
|---|---|---:|---:|---:|---:|---:|
| PrimeVul → PrimeVul | bce | 0.770582 | 0.728218 | 0.531587 | 0.844958 | 0.760188 |
| PrimeVul → PrimeVul | csd_sum | 0.742042 | 0.721234 | 0.501325 | 0.829703 | 0.734148 |
| PrimeVul → PrimeVul | csd_mean | 0.767289 | 0.707182 | 0.514489 | 0.839538 | 0.752076 |
| PrimeVul → SecVulEval | bce | 0.471498 | 0.517637 | -0.047119 | 0.467211 | 0.443664 |
| PrimeVul → SecVulEval | csd_sum | 0.464251 | 0.545305 | -0.057319 | 0.465997 | 0.445130 |
| PrimeVul → SecVulEval | csd_mean | 0.485024 | 0.492381 | -0.026445 | 0.467931 | 0.447597 |
| SecVulEval → SecVulEval | bce | 0.532850 | 0.623589 | 0.107115 | 0.580023 | 0.539889 |
| SecVulEval → SecVulEval | csd_sum | 0.553623 | 0.546614 | 0.108160 | 0.572149 | 0.543970 |
| SecVulEval → SecVulEval | csd_mean | 0.535749 | 0.358906 | 0.055210 | 0.561804 | 0.530020 |
| SecVulEval → PrimeVul | bce | 0.407245 | 0.520426 | -0.081274 | 0.337628 | 0.309570 |
| SecVulEval → PrimeVul | csd_sum | 0.446762 | 0.419355 | -0.090391 | 0.410529 | 0.338618 |
| SecVulEval → PrimeVul | csd_mean | 0.520307 | 0.204007 | -0.094957 | 0.453120 | 0.363299 |

相对 sum，mean 的域内 AUC 变化为 PrimeVul +0.009835、SecVulEval −0.010345；跨域变化分别为 +0.001933 / +0.042590。相对 BCE，跨域 AUC 变化分别为 +0.000719 / +0.115491。mean 的跨域 AUC 相对自身 Test 下降 0.371608 / 0.108684，两方向仍低于 0.5。MCC、F1 未随 AUC 一致改善，尤其 SecVulEval → PrimeVul 的 F1/MCC 为 0.204007/−0.094957。

## 源域 Train 项目残差

仍使用全部 27,320 条源域 Train、同一 Seed-42 推理顺序、同一次编码比较 Common/Specific；新旧样本 ID、标签、项目及顺序完全相同。残差范数比例对项目等权统计；Logit 与分类指标覆盖全部 Train 样本。翻转使用各自源 Valid 阈值，不同阈值的翻转数不直接视作残差强弱比较。固定 0.5 的统计另存于各库 JSON。

| 源库 | 归约 | 阈值 | 残差/共享范数中位数 | 平均绝对 Logit 差 | Common AUC | Specific AUC | BCE 降低量 | 翻转（纠正/退化） |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PrimeVul | sum | 0.26 | 0.072402% | 0.002556 | 0.953496 | 0.953543 | 0.000130991 | 2/9724（1/1） |
| PrimeVul | mean | 0.44 | 0.065845% | 0.002514 | 0.951201 | 0.951235 | 0.000111727 | 4/9724（4/0） |
| SecVulEval | sum | 0.50 | 0.074868% | 0.005672 | 0.693217 | 0.693341 | 0.000138483 | 188/17596（88/100） |
| SecVulEval | mean | 0.50 | 0.065428% | 0.003414 | 0.675850 | 0.676082 | 0.000087385 | 87/17596（47/40） |

mean 下项目头的 Train AUC 增益为 PrimeVul 0.000033990、SecVulEval 0.000232092；BCE 收益均低于 0.00012。相对共享权重的残差范数中位数均下降到约 0.065%。这仍是非零但很弱的项目特定作用，没有形成实质性的额外分类能力。

原 sum 的完整统计保存在 [旧诊断目录](../csd_diagnostics/report.md)；本次参数范数、分布、项目构成、系数及完整 Train Logit 见 [mean 诊断汇总](csd_diagnostics/summary.csv) 和两库 JSON/CSV。

## 优化尺度

| 源库 | 归约 | 第 3 Epoch Common CE | Specific CE | 实际正则项 | 正则占总损失 | 最终探针 U 正则梯度/分类梯度 |
|---|---|---:|---:|---:|---:|---:|
| PrimeVul | sum | 0.354527 | 0.354399 | 61.564131 | 98.8616% | 8503.20 |
| PrimeVul | mean | 0.356734 | 0.356631 | 8.840567 | 92.5333% | 1335.73 |
| SecVulEval | sum | 0.653772 | 0.653646 | 15.238649 | 92.0983% | 4683.94 |
| SecVulEval | mean | 0.660600 | 0.660519 | 2.703080 | 67.1706% | 2854.55 |

Epoch 表统计实际训练损失；梯度探针使用与上轮相同的 8 条 Seed-42 源 Train 样本，在选中 checkpoint 上关闭 dropout，不创建优化器，也不是历史训练步梯度。JSON 的 `head_regularizer_grad_norms` 明确标为 sum 参考尺度，上表对 mean 实验除以 8 后才与其分类梯度比较。

mean 确实降低了正则压力，但 U 的实际正则梯度仍为分类梯度的千倍量级；残差的范数、Logit 和 BCE 收益均未增大。本次结果支持“归约修正仍不足以解决项目残差过弱”，尚无项目分解增强的证据。

## Valid 选模与数值复验

| 源库 | Epoch | Valid MCC | Valid AUC | 阈值 |
|---|---:|---:|---:|---:|
| PrimeVul | 1 | 0.523678534 | 0.834127212 | 0.44 |
| PrimeVul | 2 | 0.515897505 | 0.838281923 | 0.69 |
| PrimeVul | 3 | 0.546411542 | 0.846110752 | 0.44 |
| SecVulEval | 1 | 0.035279864 | 0.510136604 | 0.58 |
| SecVulEval | 2 | 0.050920651 | 0.524649756 | 0.48 |
| SecVulEval | 3 | 0.071488145 | 0.542075640 | 0.50 |

4 项相关单元测试通过，诊断单位标记更新后又复验了 2 项诊断测试。两库初始化及全部 Train 成员一致；全部 27,320 条 Train CSV 可精确复算配对指标；905 个项目覆盖全部样本。两份 Test 成员 CSV、类别数和排除数量与旧实验一致，24 行比较结果均从保存预测精确复算。

一项额外的 Valid 逐位一致断言没有通过：SecVulEval 训练选模 AUC 为 0.542075639943997，保存的重推 AUC 为 0.542073984042073，差值 -1.65590192e-06；AUPRC 差值为 -1.08368411e-06。Epoch、阈值、Accuracy/F1/MCC 和所有混淆计数完全一致。保留该不一致，未覆盖预测或重选参数。

预先固定的两次同 GPU、同 checkpoint Valid 重推中，524 条 Logit 不逐位相同，最大差值 4.76837158e-07；两个重推 AUC 分别为 0.542076467894960 / 0.542074398017554，分类结果均一致。窗口池化使用 CUDA `index_add`；[PyTorch 2.9 文档](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.index_add_.html)明确说明 CUDA 累加可能非确定性。本次记录与微小浮点归约波动一致。完整观察见 [validation_repeatability.json](validation_repeatability.json)。

## 复现与文件

训练、Clean/迁移评价及 Train 诊断均使用现有 CLI，完整命令见 [README 的 mean 对照](../../README.md#csd-mean-正则对照)。checkpoint 保存于 `results/csd_mean/<dataset>/csd/best.pt`；没有删除或重写旧模型。新诊断的 Epoch CSV 记录实际 `orthonormality`、归约方式，以及同参数的 sum/mean 换算。BCE/sum 数值直接读取旧固定队列结果，本次只训练两个 mean CSD。
