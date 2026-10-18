# 单源跨数据集泛化：BCE、LP-FT 与 SWA

四个新模型均已实际完成 3 Epoch，BCE 复用原 checkpoint 和预测。相对 BCE，LP-FT 的双向跨域 AUC 增加 0.000828 / 0.153942，SWA 增加 0.000865 / 0.025865；但四项跨域 AUC 仍全部低于 0.5，MCC 均为负。两种方法的源域 Clean AUC 也均下降，尤其 LP-FT。当前固定预算下观察到了数值改善，尚未获得有效的跨域漏洞判别能力。

## 方法与预算

复用同一个 `CodeBERT`、单层线性分类头、BCE、训练循环、阈值选择和评价实现。新增方法各自从同一预训练 CodeBERT 初始化，不加载旧微调权重；两项方法独立运行，不组合使用。无源码增强、VAT 或其他附加损失。

- **LP-FT**：采用 [Kumar et al., ICLR 2022](https://openreview.net/forum?id=UYneFzXSJWh) 的先线性探测再全参数微调流程。本次总预算固定为 1+2 Epoch：第 1 轮冻结整个编码器并关闭其 dropout，仅以 AdamW lr=1e-3 训练原有线性头；随后保留已训练的头，解冻编码器、重建 AdamW，并以原 lr=1e-5 全参数微调两轮。只从 FT 两轮的源域 Clean Valid 候选中选模。
- **SWA**：采用 [Izmailov et al., UAI 2018](https://arxiv.org/abs/1803.05407) 的后期轨迹等权平均，使用 PyTorch [AveragedModel](https://docs.pytorch.org/docs/2.9/generated/torch.optim.swa_utils.AveragedModel.html)。保留原 BCE 的 AdamW、恒定 lr=1e-5 和 3 Epoch，在第 2、3 轮各自中点和终点采样，共 4 份权重，编码器和分类头一起在线平均。第 2、3 轮末分别评价包含 2/4 份权重的平均模型，仅从平均候选选模。这里检验原 BCE 优化设置下的参数平均，不额外引入学习率调度。CodeBERT 使用 LayerNorm，无 BatchNorm 统计需要重估。

统一 Seed 42、micro batch 4、梯度累积 2、weight decay=0.01、gradient clip=1；参数/优化器为 FP32，CUDA 编码器为 BF16。保持原 2048 逻辑输入、512 窗口、stride=244 及唯一源码位置加权池化。预训练位置为 `/home/PublicData/PHY-data/resource/codebert-base`，环境为 PyTorch 2.9.1+cu128、Transformers 4.57.3、RTX 4090。完整配置见 [results.json](results.json) 的 `training`。

| 源库 | Train | 选模 Valid | 每轮优化器步数 | 每个方法总步数 | LP-FT 编码器更新步数 |
|---|---:|---:|---:|---:|---:|
| PrimeVul | 9,724 | 1,186 | 1,216 | 3,648 | 2,432 |
| SecVulEval | 17,596 | 2,198 | 2,200 | 6,600 | 4,400 |

LP-FT 的总样本遍历数和优化器步数与 BCE 相同，但编码器只更新后两轮。SWA 全程更新全部参数，平均成员的全局优化器步骤预先固定为 PrimeVul `[1824, 2432, 3040, 3648]`、SecVulEval `[3300, 4400, 5500, 6600]`。没有根据 Test 改变 LP/FT 分配、学习率、平均起点或采样频率。四次新增训练合计完成 20,496 次优化器更新。

## 固定数据及选模协议

未重新准备或采样数据，保留原 Train/Valid/Test。正式评价只排除 Test 与两库实际 Train/选模 Valid 完全相同的源码，沿用原有成员及顺序，无其他过滤。

| 目标 Test | 原完整队列 | 精确重叠排除 | 正式共用队列 | 正例 | 负例 |
|---|---:|---:|---:|---:|---:|
| PrimeVul | 1,098 | 187 | 911 | 369 | 542 |
| SecVulEval | 2,202 | 132 | 2,070 | 986 | 1,084 |

同一目标供两个来源、三种方法使用完全相同的队列。两份成员 CSV 与本轮起点逐字节相同，包含上游位置和排除原因。BCE 的 8 行完整/独立队列结果及训练元数据与原记录完全一致；实际缓存检查确认四份 BCE Test 预测均直接读取，无重新推理或写入。原始数据、预训练 CodeBERT、已有六方法及其他非本轮结果保持原样。

仅通过各自 Clean Valid 选择：阈值按 MCC/F1/Accuracy/距 0.5，Epoch 按 MCC/F1/Accuracy/AUC。目标域没有参与训练、选模或阈值校准。六个最终模型均为第 3 Epoch；BCE 未重新训练。

| 源库 | 方法 | 选中 Epoch | 源 Valid MCC | 源 Valid AUC | 固定阈值 |
|---|---|---:|---:|---:|---:|
| PrimeVul | BCE | 3 | 0.564589 | 0.852179 | 0.40 |
| PrimeVul | LP-FT | 3 | 0.541623 | 0.840654 | 0.28 |
| PrimeVul | SWA（4 份） | 3 | 0.554580 | 0.845681 | 0.56 |
| SecVulEval | BCE | 3 | 0.093329 | 0.548257 | 0.46 |
| SecVulEval | LP-FT | 3 | 0.068124 | 0.532595 | 0.39 |
| SecVulEval | SWA（4 份） | 3 | 0.087600 | 0.539422 | 0.36 |

上述为训练选模时的 Valid 记录。两个 LP 阶段的 Valid AUC 为 0.788760 / 0.502879，均未进入最终候选。各方法 3 轮共 18 条 Epoch 记录见 [training.csv](training.csv)，包含阶段、学习率、编码器是否更新、候选资格、平均成员和 Valid 指标。SWA 的 `train_bce` 是在线训练模型的损失，`valid_*` 在后两轮对应平均模型；不把二者混为同一个模型的损失。

## 相同独立 Test 上的完整结果

Accuracy、F1、MCC 使用各自源域 Valid 阈值；AUC/AUPRC 使用原始保存的概率分数。AUPRC 沿用项目的 Average Precision 定义。下表为正式 `independent`，完整 Test 参考也保存在 [comparison.csv](comparison.csv)。两个队列 × 六个模型 × 两个目标共 24 行，包含 Epoch、阈值、样本/类别数、混淆计数及各类差值。

| 训练 → Test | 方法 | Accuracy | F1 | MCC | AUC | AUPRC |
|---|---|---:|---:|---:|---:|---:|
| PrimeVul → PrimeVul | BCE | 0.770582 | 0.728218 | 0.531587 | 0.844958 | 0.760188 |
| PrimeVul → PrimeVul | LP-FT | 0.748628 | 0.716934 | 0.500180 | 0.828618 | 0.728824 |
| PrimeVul → PrimeVul | SWA | 0.776070 | 0.733681 | 0.542022 | 0.841153 | 0.752467 |
| PrimeVul → SecVulEval | BCE | 0.471498 | 0.517637 | -0.047119 | 0.467211 | 0.443664 |
| PrimeVul → SecVulEval | LP-FT | 0.478261 | 0.541985 | -0.029714 | 0.468039 | 0.448310 |
| PrimeVul → SecVulEval | SWA | 0.474396 | 0.516444 | -0.041815 | 0.468076 | 0.445184 |
| SecVulEval → SecVulEval | BCE | 0.532850 | 0.623589 | 0.107115 | 0.580023 | 0.539889 |
| SecVulEval → SecVulEval | LP-FT | 0.499034 | 0.632920 | 0.055394 | 0.533884 | 0.492592 |
| SecVulEval → SecVulEval | SWA | 0.510628 | 0.647407 | 0.106198 | 0.577194 | 0.541734 |
| SecVulEval → PrimeVul | BCE | 0.407245 | 0.520426 | -0.081274 | 0.337628 | 0.309570 |
| SecVulEval → PrimeVul | LP-FT | 0.422613 | 0.559464 | -0.001268 | 0.491570 | 0.411543 |
| SecVulEval → PrimeVul | SWA | 0.412733 | 0.559671 | -0.023210 | 0.363494 | 0.322846 |

## 是否改善跨数据集泛化

相对同来源 BCE、相同目标成员的 AUC 差值如下：

| 训练 → Test | LP-FT − BCE | SWA − BCE |
|---|---:|---:|
| PrimeVul → PrimeVul | −0.016340 | −0.003805 |
| PrimeVul → SecVulEval | +0.000828 | +0.000865 |
| SecVulEval → SecVulEval | −0.046140 | −0.002829 |
| SecVulEval → PrimeVul | +0.153942 | +0.025865 |

**LP-FT：反向跨域有较大的数值改善，但没有解决泛化。** SecVulEval → PrimeVul 从 0.337628 提高到 0.491570，缓解了低于随机的排序表现；PrimeVul → SecVulEval 只提高 0.000828。与此同时，两库域内 AUC 分别下降 0.016340 / 0.046140。两个跨域 AUC 仍低于 0.5，MCC 为 −0.029714 / −0.001268，不能把反向增益直接视为已获得有效的跨域漏洞判别能力。

**SWA：跨域 AUC 有小幅数值改善，绝对能力仍弱。** 两个方向分别提高 0.000865 / 0.025865，AUC 只有 0.468076 / 0.363494，MCC 为 −0.041815 / −0.023210。源域 AUC 均略低于 BCE。PrimeVul 域内 Accuracy/F1/MCC 则小幅提高，说明不同指标并非全部同向变化；完整结果保留了这些收益和下降。

这些结论限定于本次预先固定的 Seed 42、1+2 Epoch LP-FT 和原 BCE 优化设置下的 SWA，不外推为两类方法在所有设置下都无效。当前实证不支持它们已经解决 PrimeVul ↔ SecVulEval 的单源跨库泛化；本轮到此结束，不叠加其他训练方法。

## 实际检查与复现

10 项直接相关单元测试通过，覆盖原六方法、两库 LP-FT/SWA 微型训练、仅源 Valid 选模、冻结/解冻、严格权重加载、固定评价队列和缓存。LP 阶段的编码器不变而分类头更新；FT 编码器确实更新。微型训练捕获实际在线权重，核对第 2/3 轮平均候选等于对应 2/4 份权重的算术平均，并核对保存的 checkpoint 就是所选候选。缓存保护修改后，6 项评价测试再次通过。

```bash
python -m unittest tests.test_codebert_source tests.test_evaluation -v
python -m unittest tests.test_evaluation -v
```

实际实验核对了全部 24 行的五项指标、类别数、混淆计数及差值，均可从保存预测精确复算。两份成员文件逐字节一致，BCE 的全部旧记录保留。四个新 checkpoint 的配置、完成轮数/步数、源 Valid 最优候选、阈值、单层分类头和平均成员记录均已检查；18 行 Epoch 统计来源于实际训练日志。

PrimeVul 三模型的 Valid 重推与选模记录完全一致。SecVulEval 的分类指标、阈值和所有混淆计数也一致，但存在微小排名数值差异（重推减训练记录）：

| SecVulEval 方法 | Valid AUC 差值 | Valid AUPRC 差值 |
|---|---:|---:|
| BCE（已有缓存） | −1.65590192e−06 | −1.48212492e−06 |
| LP-FT | +1.24192644e−06 | +1.15938139e−06 |
| SWA | 0 | +6.95493280e−07 |

原样记录这些差异，未覆盖旧 BCE 预测、重复挑选推理结果或重新选模。正式 Test 结果均按实际保存预测精确复算。

完整训练、Clean 和双向评价命令见 [README](../../README.md#lp-ft-与-swa)。新权重、配置、日志及预测位于 `results/<dataset>/{lp_ft,swa}/`，checkpoint 为 `best.pt`；原 BCE 仍在各自 `bce/`。所有方法复用同一训练和评价流程，无并行实现或新增泛化模块。
