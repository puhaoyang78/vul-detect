# PrimeVul ↔ SecVulEval 迁移诊断

当前最强证据指向**正负样本构成及长度、词法关联不一致**。PrimeVul 的普通平衡分类有很强的长度信号；SecVulEval 包含大量相似的漏洞/修复函数，长度关联较弱且略反向。两个 BCE 的分数与长度也呈相反关联。SecVulEval 模型自身的判别能力有限，跨到 PrimeVul 后又遇到相反的标签关联，因此出现低于随机的排序。

TF-IDF + Logistic Regression 复现了类似的双向失败，说明现象不限于 CodeBERT。项目陌生、局部上游标签分歧及输入截断确实需要考虑，但现有证据不支持其中任何一项单独解释全部下降。以下是固定模型上的诊断证据，不是对唯一因果机制的证明。

## 固定协议与实现核查

复用已有 BCE、LP-FT、SWA 和源域拟合的 TF-IDF 模型，不训练、不重新选 Epoch/阈值。BCE 均使用原第 3 Epoch，PrimeVul / SecVulEval 阈值仍为 0.40 / 0.46。全部预处理、模型结构和训练超参数保持不变。

| Test | 完整队列（正/负） | 正式独立队列（正/负） |
|---|---:|---:|
| PrimeVul | 1098（549/549） | 911（369/542） |
| SecVulEval | 2202（1101/1101） | 2070（986/1084） |

正式队列仍仅排除与两库实际 Train/选模 Valid 完全相同的源码。同一目标供所有模型使用相同成员。逐条核对全部实际 Train/Valid/Test 的源码和原始 `target` / `is_vulnerable`，均与上游一致；核对缓存 ID、标签、输入预算和阈值，并从保存分数精确复现 12 行正式五项指标及混淆计数。未发现适配器翻转标签、预测错位或本次队列不一致。

原完整 Test 的双向跨域 BCE AUC 已是 0.490601 / 0.314830，独立队列为 0.467211 / 0.337628。排除规则会改变样本构成，但并未制造迁移失败这一现象。AUC 使用连续分数，不依赖分类阈值，单纯重新校准阈值无法纠正排序。

## 长度关联改变方向

| 数据与队列 | 正例行数中位数 | 负例行数中位数 | 原行数 AUC |
|---|---:|---:|---:|
| PrimeVul Train | 74 | 18 | 0.804661 |
| SecVulEval Train | 39 | 43 | 0.476638 |
| PrimeVul 独立 Test | 77 | 18 | 0.808433 |
| SecVulEval 独立 Test | 47 | 52 | 0.469581 |

在同一个 PrimeVul Test 上，PrimeVul BCE 分数与行数的 Spearman ρ 为 **+0.857286**，SecVulEval BCE 为 **−0.568823**。在同一个 SecVulEval Test 上，分别为 **+0.692450 / −0.310334**。因此，SecVulEval 模型跨到 PrimeVul 时偏向较短函数，而 PrimeVul 的漏洞正例明显较长；PrimeVul 模型跨到 SecVulEval 时，原先有效的“较长更可能为正例”关联也不成立。

沿用 `match_lengths` 的 Seed 42 逐行数精确匹配，不读取模型分数选样。只对当前独立队列构造新的诊断子集，不改正式 Test，不覆盖历史 Motivation 清单。

| 目标 | 匹配样本（正/负） | 匹配后行数 AUC | 匹配后完整 Token 数 AUC |
|---|---:|---:|---:|
| PrimeVul | 344（172/172） | 0.500000 | 0.514298 |
| SecVulEval | 1498（749/749） | 0.500000 | 0.496693 |

| BCE 训练 → Test | 原始独立 Test AUC | 长度匹配 AUC | 匹配 AUC 的项目 bootstrap 95% 区间 |
|---|---:|---:|---:|
| PrimeVul → PrimeVul | 0.844958 | 0.633112 | [0.564997, 0.707028] |
| PrimeVul → SecVulEval | 0.467211 | 0.494307 | [0.466196, 0.523665] |
| SecVulEval → SecVulEval | 0.580023 | 0.562494 | [0.532156, 0.590880] |
| SecVulEval → PrimeVul | 0.337628 | 0.531335 | [0.450597, 0.616631] |

原始反向 AUC 的区间为 [0.294888, 0.383439]；匹配后回到接近随机，区间包含 0.5。PrimeVul 模型域内也明显下降，但匹配子集仍保留高于随机的判别能力。结果支持长度关联是理解异常排序的重要线索，不能把 AUC 变化量直接解释为长度的因果贡献：匹配同时改变了评价人群，也没有控制所有词法特征。

区间使用 Seed 42、1000 次项目聚类 bootstrap，按标准化上游项目 URL 整组有放回抽样，保留同项目内的配对相关性；它仅描述固定模型和既定队列下的不确定性，不包含重新训练的方差。

## 词法基线也不能迁移

复用已有 C/C++ token unigram/bigram TF-IDF + LR，读取完整函数，不受 CodeBERT 的 2025 Token 截断约束。两库原 Train/Valid 与当前相同；词表、IDF、LR 仅由源 Train 拟合，C 和阈值由源 Valid 选择。这里仅对当前 Test 重新计算已有 LR 的分数，没有重新拟合，未使用目标标签调参。

| 训练 → 独立 Test | TF-IDF LR AUC | CodeBERT BCE AUC |
|---|---:|---:|
| PrimeVul → PrimeVul | 0.839073 | 0.844958 |
| PrimeVul → SecVulEval | 0.462854 | 0.467211 |
| SecVulEval → SecVulEval | 0.563636 | 0.580023 |
| SecVulEval → PrimeVul | 0.366774 | 0.337628 |

普通词法模型和 CodeBERT 呈现相同的失败方向。PrimeVul BCE 分数与自身 TF-IDF 分数在 PrimeVul Test 上的 ρ 为 0.868155，在 SecVulEval Test 上仍为 0.723540，但目标标签已无法由这些分数有效排序。不能把 PrimeVul 上的高 AUC 直接当成获得了可迁移的漏洞语义；也不能据此断言所有长度、词法特征都与真实漏洞无关。

## 负样本与实际判别任务不同

[PrimeVul 原论文 III-B](https://arxiv.org/html/2403.18624v2#S3.SS2) 将确认漏洞的修复后版本及同提交其他未变函数标为负例；其普通负例池远大于漏洞集合。[SecVulEval 原论文 3.2](https://arxiv.org/html/2505.19828v1#S3.SS2) 主要收集安全修复前后的函数，[上游数据卡](https://huggingface.co/datasets/arag0rn/SecVulEval/blob/main/README.md) 用 `fixed_func_idx` 指向对应修复版本。两库均按现有规则保留正例、抽取等量负例，仍会形成不同的正负类对比。

| 实际 Train | 负例数 | 可由上游链接确认的修复负例 | 比例 | 同时保留的完整修复对 |
|---|---:|---:|---:|---:|
| PrimeVul | 4862 | 132 | 2.7% | 133 |
| SecVulEval | 8798 | 6242 | 71.0% | 6242 |

PrimeVul 数量来自其官方 paired 列表，该列表只覆盖官方定义的相似修复对，因此不能将所有未链接负例一概称为非修复函数。133 个关系对应 132 个修复负例，关系数与唯一负例数并不必然相等。

独立 Test 中，PrimeVul 只有 6 个完整上游修复对，SecVulEval 有 639 对。后者完整函数的逐行序列相似度中位数为 0.928934，正负例经常只差局部修改。在其中 **594 对模型有效输入不同**的样本上：

| BCE 来源 | 漏洞函数分数高于修复函数 | 同分 | 配对排序正确率 |
|---|---:|---:|---:|
| PrimeVul | 263/594 | 0 | 0.442761 |
| SecVulEval | 367/594 | 0 | 0.617845 |

这是同一修复对内的排序正确率，不是把所有正负函数混在一起计算的 ROC-AUC。PrimeVul 的高普通分类分数没有转化为该配对任务的能力。SecVulEval 模型有一定局部判别信息，但强度有限。PrimeVul 自身只有 6 对，不据此作稳定的配对性能判断。全部原始 639 对的分数与排名仍保存，45 对输入相同的样本单独标记。

另从各自 Train 按 Seed 42、类别 0 后类别 1 各抽取 512 条，以最终 BCE checkpoint 做一次推理：

| 源库 | Train 诊断样本（正/负） | 最终 Train 抽样 AUC | 最终 Train 抽样 BCE | 源 Valid AUC | 独立 Test AUC |
|---|---:|---:|---:|---:|---:|
| PrimeVul | 1024（512/512） | 0.937931 | 0.340476 | 0.852179 | 0.844958 |
| SecVulEval | 1024（512/512） | 0.687670 | 0.624949 | 0.548257 | 0.580023 |

Valid 列取原选模日志。Train 损失是最终权重在固定抽样上的 BCE，不是训练过程中逐轮在线损失的平均。这些是训练成员上的学习诊断，不能作为泛化成绩。SecVulEval 模型并非完全没有学习，但在当前预算下，连源 Train 的判别能力都较有限。仅凭这次检查不能进一步区分训练不足、困难修复样本、缺失函数外上下文等因素各占多少，也不据此调整训练预算。

## 项目、标签及截断的作用

**项目划分更严格，但项目陌生不能解释全部迁移失败。** PrimeVul 保留官方时间划分，独立 Test 有 761/911 条项目 URL 已在自身 Train 出现；SecVulEval 按项目划分，自身 Test 项目均未见。在 PrimeVul Train 已见项目的 967 条 SecVulEval Test 上，PrimeVul BCE AUC 仍只有 0.460102；在 SecVulEval Train 已见项目的 545 条 PrimeVul Test 上，SecVulEval BCE 也只有 0.313575。这里仅按上游 URL 做直接标准化匹配，不推测项目别名，不把项目当成正式过滤条件。

**标签适配正确，但上游确有局部分歧。** 对两库全部原始函数做完全相同源码比较，共 4269 段重叠，其中 357 段标签不同：150 段 PrimeVul=1/SecVulEval=0，207 段相反；其余 3912 段一致。当前实际选择的两库数据共有 1369 段重叠，其中 119 段分歧。正式独立 Test 分别涉及 10 条 PrimeVul 正例、16 条 SecVulEval 正例。两库总体标签方向一致，局部分歧可能涉及标注准则、时间或函数外上下文，不能自动判定哪一方标错；本轮不改标签或追加排除。

**截断造成一部分不可区分输入，但完整函数子集仍难迁移。** CodeBERT 保留原 2025 源码 Token 预算。SecVulEval 独立 Test 中有 45 组、93 条相反标签样本得到完全相同的可见 Token 序列，确定性函数输入分类器至少会错 45/2070 条；其中包含 45 个上游修复对。实际 `InputBuilder` 对这 45 对产生的 ID、mask、owner、pooling 张量均逐对验证相等。PrimeVul 独立 Test 没有这种相反标签组。

仅作诊断地查看完整函数未被截断的成员，仍使用原分数与阈值：PrimeVul → SecVulEval 在 1597 条上的 AUC 为 **0.473669**，SecVulEval → PrimeVul 在 772 条上为 **0.359998**。同一未截断目标子集上的域内 AUC 分别为 PrimeVul **0.827321**、SecVulEval **0.590787**。截断不是唯一解释，正式 Test 未被这些子集替换。

相同可见输入的缓存 Logit 有小幅差异。选择缓存差异最大的 `secvuleval:10125/10126` 作数值检查，两个 BCE 在同批次 BF16 及现有 `force_fp32` 路径重放时，该对 Logit 都完全相同；后者编码器使用 FP32，池化及分类计算使用 FP64。改变批次形状时，PrimeVul BCE 同输入 Logit 从 1.951601 变为 1.957205，SecVulEval BCE 从 0.270321 变为 0.270587。模型始终为 eval/no_grad。这证实存在批次形状相关的数值变化，因此不把相同有效输入上的缓存微小排名当作语义判别证据，也没有用重放覆盖正式预测。

## 对已有 LP-FT / SWA 结果的解释

| SecVulEval 来源模型 → PrimeVul | 原 AUC | 分数与行数 ρ | 长度匹配 AUC |
|---|---:|---:|---:|
| BCE | 0.337628 | −0.568823 | 0.531335 |
| LP-FT | 0.491570 | −0.078472 | 0.528799 |
| SWA | 0.363494 | −0.446227 | 0.519166 |

LP-FT 的原队列反向增益，同时伴随长度负关联明显减弱；在同一长度匹配子集上，并未优于 BCE。这与“缓解反向统计关联，但尚未学出更强可迁移判别信息”的解释一致，不能将原 AUC 增益直接当成漏洞语义泛化已经改善。没有据此新增方法或调参。

## 结果与检查

- [diagnosis.json](diagnosis.json)：数据构成、输入碰撞、标签分歧、词法模型、57 行诊断指标、配对统计和项目 bootstrap 区间。
- [diagnosis.members.csv](diagnosis.members.csv)：全部原 Test 成员及当前独立/长度匹配标记，包含行数、Token 数和项目 URL；原正式清单未改。
- [diagnosis.performance.csv](diagnosis.performance.csv)、[diagnosis.pairs.csv](diagnosis.pairs.csv)：子集指标及全部 645 个上游配对的原 Logit 差。
- [diagnosis.label_disagreements.csv](diagnosis.label_disagreements.csv)：357 组上游相同源码标签分歧的原始 ID。
- [diagnosis.train.metrics.json](diagnosis.train.metrics.json)、[diagnosis.train.predictions.jsonl](diagnosis.train.predictions.jsonl)：两库固定 Train 抽样、来源位置、原预测及指标。
- [diagnosis.input_check.json](diagnosis.input_check.json)：相同输入的同批次/异批次、BF16/FP32 检查。

复现命令见 [README](../../README.md#双向迁移诊断)，复用 `data.py`、`metrics.py`、`InputBuilder`、`predict_logits` 和既有匹配/TF-IDF 实现。16 项 Motivation/评价单元测试通过，包含原始字段和修复链接核对、相同输入边界、项目 bootstrap、完整诊断流程及旧结果保护；新增派生配对统计后再次运行相关 Motivation 测试。实际结果独立复算了两库所有方法在原独立队列、长度匹配及未截断子集上的指标，并核对抽样均只来自源 Train、每类 512 条。原始数据、数据划分、既有模型、正式结果及历史 Motivation 结果未修改。
