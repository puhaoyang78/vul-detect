# Dynamic Hard Augmentation for Vulnerability Detection

当前实验比较普通 CodeBERT BCE 与 CSD（sum / mean / 共享门控对照）在 PrimeVul / SecVulEval 上的域内及双向迁移性能。保留已有四库 Motivation 和六方法实验结果；这些历史诊断使用其原有评价成员，当前 PrimeVul Test 协议及结果见下文双向迁移实验。

## 四数据集 Motivation

新增上游为 [BigVul](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset) 和 [SecVulEval](https://huggingface.co/datasets/arag0rn/SecVulEval)。实际读取 `bigvul/MSR_data_cleaned.csv` 的 `func_before` / `vul` / `project`，以及 `SecVulEval/data/train-00000-of-00001.parquet` 的 `func_body` / `is_vulnerable` / `project_url`。SecVulEval 原始项目名存在同一仓库 URL 多名的情况，分组使用原生仓库 URL 去首尾空白、尾部斜杠并 casefold，避免 `tpm2-tools` / `tpm2.0-tools` 等跨 Split。漏洞和修复函数均按原生标签进入样本池，不做近重复聚类或标签冲突组删除。

新增两库沿用 MegaVul 的 Seed 42 项目分组算法，尽量满足正例 80/10/10；每个 Split 保留所有正例，随机取等量负例。项目名去空白并 casefold，不跨 Split。仅在库内检查原始源码精确相等：Valid 选模排除 Train 重叠，Test 评价排除 Train/Valid 重叠；完整平衡清单仍保留原始正例，排除数量见 `statistics.json`。数据准备不执行跨库筛选。以下记录已完成 Motivation 的评价规模；当前 PrimeVul 已恢复 1,098 条完整平衡 Test，BCE checkpoint 保持不变。

| 数据集 | Train | Valid | Test 平衡清单 | 已保存 Motivation Test | 原始正例总数 |
|---|---:|---:|---:|---:|---:|
| PrimeVul | 9,724 | 1,186 | 1,098 | 926 | 6,004 |
| MegaVul | 28,760 | 3,594 | 3,596 | 3,472 | 17,975 |
| BigVul | 17,440 | 2,180 | 2,180 | 2,180 | 10,900 |
| SecVulEval | 17,596 | 2,198 | 2,202 | 2,202 | 10,998 |

新增两库均保留全部原始正例，实际检查未发现 Train/Valid/Test 之间的原始源码精确重叠。项目分配、平衡后成员和上游记录位置保存在各库 `projects.json`、`members.csv` 和 `statistics.json`。

四库使用相同全参数 CodeBERT、3 Epoch、Seed 42、有效 batch 8 和下文的统一超参数。新增两库仅训练 `bce`，各自 Clean Valid 选 Epoch 和阈值。Fixed、Held-out、Adaptive 复用既有实现；严格语义保持作为主结果，函数名等输入压力测试单列。Worst-case 的分母包含完整 Test；ASR 分母是该类别中可扰动且 Clean 正确的样本，保存攻击成功数、分母及覆盖率。

当前 Motivation 2 聚焦函数总行数和 TF-IDF 词法特征，协议及命令见下节。此前 17 项统计特征诊断保留原有结果，不再扩展人工统计特征：总行数、去除空白和注释后的代码行数、完整 CodeBERT Token 数、字符数、标识符次数/种类/平均长度、字面量/字符串/数值次数、注释数/行数/字节占比、分支数、循环数、最大控制嵌套深度及解析错误数。结构统计为 Tree-sitter 观察到的节点：分支计 `if/switch/case`，循环计 `for/range-for/while/do`。解析错误保留并计数，不将不完整树视为编译器验证的程序结构。逐样本特征 CSV、正负类分位数/中位数和原值单特征 AUC 均保存；低于 0.5 的单特征 AUC 表示反向关联。

此前统计特征的 Logistic Regression 使用 `log1p` 和仅在 Train 拟合的标准化。候选 `C={0.01,0.1,1,10,100}`，Valid AUC 选参，平局选较小 C；阈值沿用 Valid MCC/F1/Accuracy 网格。只在 Train 拟合，不合并 Train/Valid 重训。

长度匹配按各 Test 内的总行数逐值精确匹配，包含空行和注释。按行数升序、标签 0 再 1，每类以 `random.Random(42)` 抽取两类数量较小值，保留源码清单原顺序。先固定成员，再使用原 BCE 预测；既有 PrimeVul/MegaVul 匹配清单会检查并直接复用。匹配后行数基线 AUC 为 0.5。这是诊断子集，不能替代正式 Test；统计关联和匹配后的性能变化不直接证明因果依赖，也不预设所有 Shortcut Features 与漏洞无关。

在已有 PrimeVul/MegaVul 数据和结果的工作目录执行：

```bash
RAW=/path/to/raw
MODEL=/path/to/codebert-base
python -m vulnmechanism.cli prepare --raw-root "$RAW" --datasets bigvul secvuleval
python -m vulnmechanism.cli motivation features --dataset all --model "$MODEL"
python -m vulnmechanism.cli train --dataset bigvul --methods bce --model "$MODEL" --device cuda:0
python -m vulnmechanism.cli train --dataset secvuleval --methods bce --model "$MODEL" --device cuda:1
python -m vulnmechanism.cli evaluate --dataset bigvul --methods bce --device cuda:0
python -m vulnmechanism.cli evaluate --dataset secvuleval --methods bce --device cuda:1
python -m vulnmechanism.cli motivation report --dataset all
python -m unittest discover -s tests -v
```

输出在 `results/motivation/`：`clean.csv`、`motivation1.csv`、`motivation2.csv`、`shortcut_features.csv`、`logistic.csv`、`length_matched.csv`、`results.json`。各库另存特征表、LR 参数/模型/预测。`results/length_matched/` 保存固定匹配成员。上述四库 Motivation 评价均为库内实验；汇总只读取现有模型预测及鲁棒性结果，不重新推理已完成的 PrimeVul/MegaVul 实验。额外的 PrimeVul/SecVulEval 跨库基线见下节。本轮不绘图。

### 实际结果（Seed 42）

BigVul、SecVulEval 的 BCE 均从同一预训练 CodeBERT 实际完成 3 Epoch，分别完成 6,540、6,600 次优化器更新，两者均由各自 Valid 选中第 3 Epoch，阈值分别为 0.10、0.46。PrimeVul、MegaVul 直接复用原有 BCE。四库历史 Clean Test 结果见 [clean.csv](results/motivation/clean.csv)：其中 PrimeVul 使用原 926 条成员，后续扰动、长度及 TF-IDF 诊断也保留该固定队列；下文当前 BCE/CSD 结果使用修正后的队列，两者分开解释。

| 数据集 | Accuracy | F1 | MCC | AUC | AUPRC |
|---|---:|---:|---:|---:|---:|
| PrimeVul | 0.7948 | 0.7716 | 0.5913 | 0.8663 | 0.7936 |
| MegaVul | 0.7128 | 0.6730 | 0.4292 | 0.7991 | 0.7970 |
| BigVul | 0.8844 | 0.8821 | 0.7694 | 0.9213 | 0.9410 |
| SecVulEval | 0.5454 | 0.6413 | 0.1075 | 0.5763 | 0.5585 |

Motivation 1 的主结果仅统计严格语义保持扰动。Worst-case 包含原函数，沿用上表完整 Test 分母；ASR 括号内为成功攻击数 / 可扰动且 Clean 正确数。

| 数据集 | 设置 | Worst-case Accuracy | Worst-case MCC | ASR（分子/分母） |
|---|---|---:|---:|---:|
| PrimeVul | Fixed | 0.7343 | 0.4813 | 13.43%（56/417） |
| PrimeVul | Held-out | 0.7149 | 0.4493 | 17.54%（74/422） |
| PrimeVul | Adaptive | 0.6868 | 0.3974 | 23.47%（100/426） |
| MegaVul | Fixed | 0.6751 | 0.3515 | 6.28%（131/2085） |
| MegaVul | Held-out | 0.6803 | 0.3624 | 5.46%（113/2068） |
| MegaVul | Adaptive | 0.6426 | 0.2844 | 11.70%（244/2085） |
| BigVul | Fixed | 0.8248 | 0.6513 | 12.07%（130/1077） |
| BigVul | Held-out | 0.8147 | 0.6320 | 14.23%（152/1068） |
| BigVul | Adaptive | 0.7913 | 0.5878 | 18.94%（203/1072） |
| SecVulEval | Fixed | 0.5163 | 0.0393 | 10.60%（64/604） |
| SecVulEval | Held-out | 0.5223 | 0.0537 | 8.61%（51/592） |
| SecVulEval | Adaptive | 0.4991 | -0.0022 | 16.67%（102/612） |

函数名等输入压力测试单列于 [motivation1.csv](results/motivation/motivation1.csv) 的 `input_only_stress` 行；该表同时保存三种设置的 Clean/Worst-case Accuracy、MCC、ASR 分子/分母及可扰动覆盖率。四库均观察到严格扰动引发原本正确的预测出错；覆盖率和 Clean 性能不同，不将 ASR 直接解释为四库难度排序。

Motivation 2 的 AUC 对比如下。匹配成员由行数和标签预先确定，匹配后正负类各占一半，四库行数基线 AUC 均为 **0.5000**。

| 数据集 | 原始行数 AUC | 原始 BCE AUC | 浅层特征 LR AUC | 长度匹配样本数（正/负） | 匹配 BCE AUC |
|---|---:|---:|---:|---:|---:|
| PrimeVul | 0.8348 | 0.8663 | 0.8386 | 318（159/159） | 0.6556 |
| MegaVul | 0.7740 | 0.7991 | 0.7808 | 1988（994/994） | 0.6007 |
| BigVul | 0.7270 | 0.9213 | 0.7367 | 1304（652/652） | 0.8895 |
| SecVulEval | 0.4788 | 0.5763 | 0.5395 | 1612（806/806） | 0.5589 |

完整比较见 [motivation2.csv](results/motivation/motivation2.csv)；17 个特征的正负类数量、分位数、中位数及单特征 AUC 见 [shortcut_features.csv](results/motivation/shortcut_features.csv)，LR 的完整 Test 指标见 [logistic.csv](results/motivation/logistic.csv)。

PrimeVul/MegaVul 的长度及浅层特征与标签有较强统计关联，长度匹配后 BCE AUC 明显下降。BigVul 匹配后仍有 0.8895 AUC，说明总行数不能解释其全部区分能力；SecVulEval 的行数接近随机排序，浅层 LR 和 BCE 整体区分能力也较弱。因此，当前证据支持四库都存在扰动失稳，但不支持“四库都有同样强的长度相关性”。匹配改变了评价样本组成，不能据此计算模型依赖长度的因果比例。

### Motivation 2：函数长度与 TF-IDF 词法特征

四库复用当前 Train/Valid/Test 成员。TF-IDF 直接读取完整原函数，复用 `syntax.source_token_spans` 的 C/C++ 词法切分；保留大小写、标识符、字面量、运算符和标点，忽略空白和注释，不加入项目、CVE、标签或其他元数据。使用 unigram + bigram，`min_df=2`、最多 100,000 个特征、sublinear TF、平滑 IDF 和 L2 归一化。词表、文档频率和 IDF 均仅在 Train 拟合。

分类器为 L2 Logistic Regression（`liblinear`，Seed 42，`max_iter=2000`）。Valid AUC 从 `C={0.01,0.1,1,10,100}` 中选择，平局取较小 C；阈值使用相同 Valid MCC/F1/Accuracy 网格。选参后保留 Train 拟合的模型，不合并 Train/Valid 重训。Test 报告 Accuracy、F1、MCC、AUC，并与总行数单特征和已有 CodeBERT BCE AUC 比较。

匹配规则在实验前固定，四库完全相同：

- 既有长度匹配清单只读复用，不改成员或历史指标。
- TF-IDF 匹配固定为 50 个宽度 0.02 的预测概率区间，左闭右开，最后一区间包含 1。每个区间按标签 0、1 的顺序，用 `random.Random(42)` 无放回抽取两类数量的较小值，输出保留原 Test 顺序。
- 联合匹配以“精确总行数 × TF-IDF 区间”为分层键，抽样规则相同；不足 200 条（每类 100 条）时跳过联合子集评价，记录候选样本量，不扩大区间或放宽行数条件。
- 匹配清单在读取 BCE 预测前保存；选择仅使用标签、行数和 TF-IDF 分数。四种子集统一报告样本量、正负数量/比例、长度 AUC、TF-IDF AUC 和保存的 BCE AUC；另存 TF-IDF 两类分数的经验 KS 距离。

区间内两类数量相等只保证分箱后的分布一致，连续分数仍可能保留区分能力。因此报告实际 TF-IDF AUC 和 KS，不将匹配自动视为成功。该诊断控制的是评价子集上可观察的捷径关联，不能证明模型内部不再依赖捷径，也不能将 AUC 变化解释为捷径的因果贡献。

在已有四库数据和 BCE 预测上执行，整个新增实验仅训练 TF-IDF 分类器：

```bash
python -m vulnmechanism.cli motivation tfidf --dataset all
# 仅从已保存 TF-IDF 结果重新生成总表，无需拟合或 CodeBERT 推理
python -m vulnmechanism.cli motivation tfidf-report --dataset all
```

输出为 `results/motivation/tfidf.protocol.json`、各库 `*.tfidf.metrics.json`、四库统一分类表 `tfidf.csv`、子集比较表 `shortcut_control.csv` 及 `tfidf.results.json`。`results/tfidf_matched/`、`results/joint_matched/` 保存已评价子集的成员、原始记录位置、行数、TF-IDF 分数和区间编号；模型与 Valid/Test 预测保存在本地。

四库 TF-IDF 分类器均已实际完成拟合与评价，结果见 [tfidf.csv](results/motivation/tfidf.csv)。所有词表均达到预设的 100,000 个特征上限；Valid 选出的 C 分别为 1、1、0.1、1，阈值分别为 0.54、0.59、0.50、0.28。

| 数据集 | TF-IDF Accuracy | TF-IDF F1 | TF-IDF MCC | TF-IDF AUC | 总行数 AUC | BCE AUC |
|---|---:|---:|---:|---:|---:|---:|
| PrimeVul | 0.7840 | 0.7549 | 0.5648 | 0.8651 | 0.8348 | 0.8663 |
| MegaVul | 0.6953 | 0.6369 | 0.3991 | 0.7745 | 0.7740 | 0.7991 |
| BigVul | 0.6381 | 0.6934 | 0.2961 | 0.7210 | 0.7270 | 0.9213 |
| SecVulEval | 0.5014 | 0.6663 | 0.0178 | 0.5614 | 0.4788 | 0.5763 |

各子集的完整比较见 [shortcut_control.csv](results/motivation/shortcut_control.csv)。所有已评价匹配子集均正负各半；原始 Test 的类别数量沿用上文队列。

| 数据集 | Test 子集 | 样本数（正/负） | 行数 AUC | TF-IDF AUC | BCE AUC |
|---|---|---:|---:|---:|---:|
| PrimeVul | 原始 | 926（388/538） | 0.8348 | 0.8651 | 0.8663 |
| PrimeVul | 长度匹配 | 318（159/159） | 0.5000 | 0.6434 | 0.6556 |
| PrimeVul | TF-IDF 匹配 | 366（183/183） | 0.5425 | 0.4994 | 0.5874 |
| PrimeVul | 联合匹配 | 48（24/24） | 跳过 | 跳过 | 跳过 |
| MegaVul | 原始 | 3472（1683/1789） | 0.7740 | 0.7745 | 0.7991 |
| MegaVul | 长度匹配 | 1988（994/994） | 0.5000 | 0.5756 | 0.6007 |
| MegaVul | TF-IDF 匹配 | 2068（1034/1034） | 0.5693 | 0.5001 | 0.6023 |
| MegaVul | 联合匹配 | 930（465/465） | 0.5000 | 0.5000 | 0.5737 |
| BigVul | 原始 | 2180（1090/1090） | 0.7270 | 0.7210 | 0.9213 |
| BigVul | 长度匹配 | 1304（652/652） | 0.5000 | 0.5351 | 0.8895 |
| BigVul | TF-IDF 匹配 | 1502（751/751） | 0.5488 | 0.4996 | 0.8940 |
| BigVul | 联合匹配 | 712（356/356） | 0.5000 | 0.4981 | 0.8701 |
| SecVulEval | 原始 | 2202（1101/1101） | 0.4788 | 0.5614 | 0.5763 |
| SecVulEval | 长度匹配 | 1612（806/806） | 0.5000 | 0.5537 | 0.5589 |
| SecVulEval | TF-IDF 匹配 | 1976（988/988） | 0.4899 | 0.5012 | 0.5621 |
| SecVulEval | 联合匹配 | 798（399/399） | 0.5000 | 0.5017 | 0.5318 |

TF-IDF 匹配后四库的 TF-IDF AUC 均接近 0.5，分数 KS 距离分别为 0.0273、0.0106、0.0173、0.0142，但行数关联并不都完全消失。联合匹配同时得到相同的正负类行数分布，三库 TF-IDF AUC 为 0.4981–0.5017。PrimeVul 联合匹配仅支持 48 条，按预设门槛跳过，未调整区间补样。

PrimeVul 的 TF-IDF 分类性能接近 BCE；MegaVul、BigVul 的词法基线也高于随机排序，SecVulEval 则整体较弱。控制后的 BCE 结果存在数据集差异：BigVul 联合匹配仍为 0.8701，而 MegaVul、SecVulEval 分别为 0.5737、0.5318。这些结果描述的是不同固定评价子集上的判别能力，不能证明模型内部已消除捷径依赖或量化捷径的因果贡献。

## PrimeVul / SecVulEval：BCE 与原始 CSD

PrimeVul 已取消因 MegaVul Train 重叠产生的 172 条排除（161 正 / 11 负），恢复原有 1,098 条平衡 Test（549 / 549）。只修改重叠标记，样本 ID、顺序、标签和上游位置均未重采样。MegaVul 不参与本实验的任何排除。

双向迁移的正式结果使用同一目标队列：从完整 Test 中，只排除原始函数字符串与 PrimeVul 或 SecVulEval 实际 Train / 选模 Valid 完全相同的成员。无需项目、CVE、近重复或额外平衡筛选。每个目标队列供两个来源的 BCE、CSD 共四个模型使用。

| 目标 Test | 完整平衡成员（正/负） | 精确重叠排除（正/负） | 正式共用队列（正/负） | 原双向评价队列 |
|---|---:|---:|---:|---:|
| PrimeVul | 1098（549/549） | 187（180/7） | 911（369/542） | 803 |
| SecVulEval | 2202（1101/1101） | 132（115/17） | 2070（986/1084） | 2070 |

PrimeVul Test 与 SecVulEval Train / Valid 分别精确重叠 163 / 24 条；SecVulEval Test 与 PrimeVul Train / Valid 分别重叠 96 / 36 条。两库自己的 Test 与自己的 Train / Valid 均无精确重叠。排除按源码直接相等判定，保留未完全相同的漏洞与修复函数。

### 原始 CSD（sum 对照）

实现依据 [Piratla et al., ICML 2020 的 Algorithm 1](https://proceedings.mlr.press/v119/piratla20a.html)，并核对了[作者提供的代码](https://github.com/vihari/CSD)。只替换最终分类头，保留现有 CodeBERT 编码器、2048 逻辑输入、512 窗口及池化。每个标签的项目分类器满足 `w_p = w_c + U a_p`，bias 使用相同分解。训练项目 ID 仅来自源库实际 Train，PrimeVul / SecVulEval 分别为 602 / 303 个领域。

以下已保存的对照固定 rank=1，采用论文公式的求和正则，λ=κ=1；结果对应提交 [`479818a`](https://github.com/puhaoyang78/vul-detect/tree/479818a8560ad39f7b872474d9ee93e7dbc11d0d)。当前正式训练已将正则改为作者代码的 `mean`，其独立对照命令见下文。

```text
L = mean(CE(project classifier, y) + λ CE(common classifier, y))
    + κ sum_y ||W_y W_y^T - I||_F^2
W_y = stack(w_c[y], U[0,y])
```

CSD 使用二类分类权重，传递 `logit[1] - logit[0]` 给现有二分类评价和 BCE 损失代码；其 BCE 与二类交叉熵数学等价。正交归一约束仅作用于权重，bias 不进入该项。分类权重及 bias 以 N(0, 0.1²) 初始化，项目系数以 N(0, 10⁻⁸) 初始化。编码器、共享头、低秩基与项目系数共同优化。Valid、Test 和 `infer` 始终只调用共享分类器，不读取评价项目 ID。

两库 CSD 分别从同一预训练 CodeBERT 初始化：Seed 42、3 Epoch、micro batch 4、梯度累积 2、AdamW lr=1e-5、weight decay=0.01、gradient clip=1，与 BCE 一致。不使用源码增强或 VAT。仅自身 Clean Valid 按原有 MCC/F1/Accuracy/AUC 规则选 Epoch 和阈值。BCE 直接复用原有第 3 Epoch checkpoint（PrimeVul 阈值 0.40，SecVulEval 0.46），已有预测保留，仅补推两个 BCE 对恢复的 172 条 PrimeVul 成员的预测。

```bash
# 复用已保存的 sum checkpoint 与 BCE 预测
python -m vulnmechanism.cli transfer --datasets primevul secvuleval --methods bce csd --device cuda:0
python -m unittest discover -s tests -v
```

沿用 `results/transfer_primevul_secvuleval/`，保存 `comparison.csv`、`results.json` 和两库成员及精确排除原因。`independent` 为正式共用队列；`established` 记录完整 Test 排除跨库重叠前的参考指标。两种队列 × 两个方法 × 两个训练库 × 两个目标库共 16 行。指标包括 Accuracy、F1、MCC、AUC、AUPRC 和混淆计数；源库 Valid 阈值在目标上保持不变。

`delta_source_*` 是目标 Test 相对同模型自身 Test 的差值；`delta_target_*` 是相对同方法目标库模型在相同 Test 上的差值；`delta_bce_*` 是同来源、同目标队列上该方法相对 BCE 的变化。源库与目标库 Test 的构成不同，跨库差值同时受其难度影响。

两个 CSD 均已从预训练 CodeBERT 实际完成 3 Epoch，分别更新 3,648 / 6,600 次；各自 Valid 均选中第 3 Epoch，阈值为 PrimeVul 0.26、SecVulEval 0.50。新权重保存在 `results/<dataset>/csd/best.pt`，BCE 权重保持原样。

正式共用队列的完整结果如下；完整 Test 参考结果一并保存在 [comparison.csv](results/transfer_primevul_secvuleval/comparison.csv) 与 [results.json](results/transfer_primevul_secvuleval/results.json)。

| 训练库 → Test | 方法 | Accuracy | F1 | MCC | AUC | AUPRC |
|---|---|---:|---:|---:|---:|---:|
| PrimeVul → PrimeVul | BCE | 0.7706 | 0.7282 | 0.5316 | 0.8450 | 0.7602 |
| PrimeVul → PrimeVul | CSD | 0.7420 | 0.7212 | 0.5013 | 0.8297 | 0.7341 |
| PrimeVul → SecVulEval | BCE | 0.4715 | 0.5176 | -0.0471 | 0.4672 | 0.4437 |
| PrimeVul → SecVulEval | CSD | 0.4643 | 0.5453 | -0.0573 | 0.4660 | 0.4451 |
| SecVulEval → SecVulEval | BCE | 0.5329 | 0.6236 | 0.1071 | 0.5800 | 0.5399 |
| SecVulEval → SecVulEval | CSD | 0.5536 | 0.5466 | 0.1082 | 0.5721 | 0.5440 |
| SecVulEval → PrimeVul | BCE | 0.4072 | 0.5204 | -0.0813 | 0.3376 | 0.3096 |
| SecVulEval → PrimeVul | CSD | 0.4468 | 0.4194 | -0.0904 | 0.4105 | 0.3386 |

CSD 相对 BCE 的跨域 AUC 变化为 PrimeVul → SecVulEval **−0.0012（−0.12 个百分点）**、SecVulEval → PrimeVul **+0.0729（+7.29 个百分点）**。对应源域 Clean AUC 分别下降 **0.0153 / 0.0079**。因此，本次固定协议下未观察到双向一致改善；SecVulEval → PrimeVul 虽有 AUC 改善，绝对 AUC 仍为 0.4105。MCC、F1 的变化与 AUC 不完全一致，完整指标均按各自源库 Valid 阈值报告。

### CSD 源域诊断

固定现有 CSD checkpoint，使用两库全部源域 Train，以 Seed 42 决定推理顺序；同一次编码比较共享头与项目头。仅做参数、损失日志及无更新梯度探针统计，不读取 Valid/Test 文件，不创建优化器。诊断不修改原方法、权重或已有迁移结果。

```bash
python -m vulnmechanism.cli csd-diagnostics run --dataset primevul --device cuda:0
python -m vulnmechanism.cli csd-diagnostics run --dataset secvuleval --device cuda:1
python -m vulnmechanism.cli csd-diagnostics report
python -m unittest tests.test_csd_diagnostics -v
```

完整证据见 [诊断报告](results/csd_diagnostics/report.md)，统一统计见 [summary.csv](results/csd_diagnostics/summary.csv) 和 [epochs.csv](results/csd_diagnostics/epochs.csv)。项目残差相对共享权重的范数中位数仅为 0.0724% / 0.0749%；Specific 相对 Common 的 Train AUC 仅提高 0.0000473 / 0.0001234。残差非零且与项目类别构成相关，但没有形成有实质分类增益的分解。该 `sum` 对照在二分类、rank=1 下相对作者 `mean` 实现具有 8 倍有效正则权重，且原作者的共享系数门控也与当前论文公式版不同。该次诊断没有改动或重训，原始统计保留；随后仅验证 `sum` → `mean`，见下节。

### CSD mean 正则对照

正则为 `mean((W_y W_y^T - I)^2)`，即对全部类别及 Gram 元素取均值。二分类、rank=1 时为原 `sum` 的 1/8；κ 仍为 1。该次无门控对照的代码见提交 [`3cc62ff`](https://github.com/puhaoyang78/vul-detect/tree/3cc62ffba8adf3baff29109fbcf51f53544dd1aa)。当前正式训练在此基础上只增加下节的共享门控；已有 mean 权重和结果保留，仍通过同一模型与评价实现读取。

```bash
# 复用已有无门控 mean checkpoint；重新训练该对照请使用上述提交
python -m vulnmechanism.cli evaluate --dataset primevul --methods csd --clean-only --output-dir results/csd_mean --device cuda:0
python -m vulnmechanism.cli evaluate --dataset secvuleval --methods csd --clean-only --output-dir results/csd_mean --device cuda:1
python -m vulnmechanism.cli transfer --datasets primevul secvuleval --methods csd --output-dir results/csd_mean --device cuda:0
```

诊断继续显式报告同参数下的 sum/mean 两种梯度尺度，Epoch 表的 `orthonormality` 与 `orthonormality_reduction` 标识实际优化项，避免将 mean 日志误记成 sum。

两个 mean CSD 均已从预训练模型完成 3 Epoch，共更新 3,648 / 6,600 次；各自 Valid 均选中第 3 Epoch，阈值为 0.44 / 0.50。在与 BCE/sum 完全相同的 911 / 2,070 条独立 Test 队列上：

| 训练 → Test | BCE AUC | CSD sum AUC | CSD mean AUC |
|---|---:|---:|---:|
| PrimeVul → PrimeVul | 0.8450 | 0.8297 | 0.8395 |
| PrimeVul → SecVulEval | 0.4672 | 0.4660 | 0.4679 |
| SecVulEval → SecVulEval | 0.5800 | 0.5721 | 0.5618 |
| SecVulEval → PrimeVul | 0.3376 | 0.4105 | 0.4531 |

mean 相对 sum 的双向跨域 AUC 增加 0.001933 / 0.042590，但仍均低于 0.5；SecVulEval 域内 AUC 下降 0.010345。项目残差范数比例中位数反而从 0.0724% / 0.0749% 降至 0.0658% / 0.0654%，项目头 Train AUC 增益仅为 0.0000340 / 0.0002321。归约改变降低了正则压力，但没有解决项目残差过弱的问题。

完整五项指标、混淆计数、配置及源 Train 诊断见 [mean 对照报告](results/csd_mean/report.md) 和 [24 行统一结果](results/csd_mean/comparison.csv)。SecVulEval Valid 额外逐位一致性检查发现约 10⁻⁶ 级 AUC 差异；固定 checkpoint 重推确认存在微小 Logit 波动，分类结果一致，未替换预测或重新选模，数值原样保存在报告及 `validation_repeatability.json`。

### CSD 共享门控验证

只在项目特定分类器中加入一个全局可学习标量 `g`，初始化为 0.1：`z_specific = tanh(g) * z_common + project_residual`。共享项的 weight 和 bias 一起缩放；Common 自身不缩放，Valid/Test 和推理仍仅使用 Common。参考[作者实现](https://github.com/vihari/CSD/blob/master/pacs/models/resnet_csd.py#L65-L76)，本次仅引入共享门控，保留原有未变换的项目系数、mean 正则、rank=1、初始化及全部训练参数；没有同时对项目系数施加 tanh，也未改变正则或优化器。

```bash
MODEL=/path/to/codebert-base
python -m vulnmechanism.cli train --dataset primevul --methods csd --model "$MODEL" --output-dir results/csd_gate --device cuda:0
python -m vulnmechanism.cli train --dataset secvuleval --methods csd --model "$MODEL" --output-dir results/csd_gate --device cuda:1
python -m vulnmechanism.cli evaluate --dataset primevul --methods csd --clean-only --output-dir results/csd_gate --device cuda:0
python -m vulnmechanism.cli evaluate --dataset secvuleval --methods csd --clean-only --output-dir results/csd_gate --device cuda:1
python -m vulnmechanism.cli transfer --datasets primevul secvuleval --methods csd --output-dir results/csd_gate --device cuda:0
python -m vulnmechanism.cli csd-diagnostics run --output-dir results/csd_gate --device cuda:0
python -m vulnmechanism.cli csd-diagnostics report --output-dir results/csd_gate
```

Train 诊断使用同一次编码的 Common / Specific，并另存 `gated_common_logit = tanh(g) * common_logit`。比较 Common 与 gated Common 测量纯缩放，比较 gated Common 与 Specific 测量项目残差增益；同时报告原始 Logit AUC，避免将 Sigmoid 舍入或缩放造成的 BCE 变化视作项目分解增强。记录门控初始/选中值、每 Epoch 轨迹、实际残差 Logit，以及相对原始/门控共享权重的残差范数。

两个门控模型已完成 3 Epoch，共更新 3,648 / 6,600 次，各自 Valid 均选中第 3 Epoch，阈值为 0.38 / 0.54。最终 g 为 0.135745 / 0.132325，tanh(g) 为 0.134917 / 0.131558。所有旧权重和结果保留；下表沿用同样的 911 / 2,070 条独立 Test：

| 训练 → Test | BCE AUC | CSD sum AUC | CSD mean AUC | CSD gate AUC |
|---|---:|---:|---:|---:|
| PrimeVul → PrimeVul | 0.844958 | 0.829703 | 0.839538 | 0.843853 |
| PrimeVul → SecVulEval | 0.467211 | 0.465997 | 0.467931 | 0.471923 |
| SecVulEval → SecVulEval | 0.580023 | 0.572149 | 0.561804 | 0.569664 |
| SecVulEval → PrimeVul | 0.337628 | 0.410529 | 0.453120 | 0.504050 |

相对 mean，跨域 AUC 提高 0.003992 / 0.050931，但绝对值仍接近或低于随机水平，跨域 MCC 为 −0.032189 / −0.056966。残差/原始共享权重范数中位数为 0.079704% / 0.072808%；扣除纯缩放后的残差 Train BCE 收益为 0.002553 / 0.000884，原始 Logit AUC 却下降 0.000283 / 0.008095。门控与残差确实发挥了作用，但没有形成有效增强的项目特定排名能力，不能将跨域 AUC 提高归因于有效项目分解。本轮不再添加 CSD 变体。

完整五项指标、门控轨迹、缩放对照、项目分析及检查见 [门控报告](results/csd_gate/report.md) 和 [32 行统一结果](results/csd_gate/comparison.csv)。SecVulEval Valid 重推 AUC 与训练记录相差约 −4.97×10⁻⁶，分类指标及选模不变，数值差异原样记录。

## 历史 PrimeVul/MegaVul 六方法实验

以下记录已有两库六方法实验当时的数据及结果，历史准备实现见提交 [`a5a44f8`](https://github.com/puhaoyang78/vul-detect/tree/a5a44f834590801269247bb3b2bd46eee91a2458)。该实验曾因 MegaVul Train 重叠排除 PrimeVul 的 172 条 Test；本轮已取消这一排除。历史六方法及 Motivation 结果仍明确对应当时的 926 条 PrimeVul 队列，不作为当前 BCE/CSD 实验结果。当前 `prepare` 仅执行库内检查。

上游为 [PrimeVul v0.1](https://github.com/DLVulDet/PrimeVul)、[MegaVul](https://github.com/Icyrockton/MegaVul)；预训练模型为 [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)。请保留各上游的许可与引用要求。

```text
RAW/
├── PrimeVul_v0.1/
│   ├── primevul_train.jsonl
│   ├── primevul_valid.jsonl
│   └── primevul_test.jsonl
└── megavul/
    └── megavul.json
```

- PrimeVul 直接读取官方 `func`、`target`，保留官方 Train/Valid/Test 归属。
- MegaVul 使用原生 `is_vul`：正例取 `func_before`，负例取 `func`。项目名去首尾空格并 casefold，项目不可跨 Split。项目先排序并用 Seed 42 打乱，再按正例数量降序，依次分配给正例目标缺额最大的 Split；目标为正例数量的 80/10/10。
- 各库各 Split 保留全部原始正例，仅用 `random.Random(42)` 抽取等量负例，最终按原始顺序输出。负例不足时直接报错，不通过减少正例凑平衡。
- 不进行精确/词法/近重复聚类，不删除标签冲突组，不按语法解析结果筛选，不合并漏洞与修复样本。
- 当时只检查原始函数字符串完全相等的训练—Test 重叠，使用实际平衡训练成员。Test 清单保留全部正例，历史评价排除与两库任一 Train 精确重叠的成员；以下表格记录当时的分母。当前 PrimeVul 标记及 PrimeVul/SecVulEval 评价规则见前节。

已保存的 Seed 42 平衡清单如下，正负各占一半：

| 数据集 | Train | Valid | Test | 保留原始正例总数 |
|---|---:|---:|---:|---:|
| PrimeVul | 9,724 | 1,186 | 1,098 | 6,004 |
| MegaVul | 28,760 | 3,594 | 3,596 | 17,975 |

MegaVul 正例 Train/Valid/Test 为 **14,380/1,797/1,798**，项目数为 394/333/334。两库均未从清单中删除正例。

当时精确重叠影响的 Test 评价分母：

| Test 数据集 | 与 PrimeVul Train 重叠 | 与 MegaVul Train 重叠 | 排除并集（正/负） | 实际评价（正/负） |
|---|---:|---:|---:|---:|
| PrimeVul | 0 | 172 | 172（161/11） | 926（388/538） |
| MegaVul | 91 | 33 | 124（115/9） | 3,472（1,683/1,789） |

历史评价队列排除后未再次平衡。当前统计及成员见各库 `statistics.json`、`members.csv`；其中 PrimeVul 已更新为本轮恢复后的标记。项目分配另存 `data/megavul/projects.json`。数据准备只写派生目录，保留上游原始文件。

## 方法与训练

| 方法 | 源码增强 | 当前模型搜索 | 普通 VAT |
|---|---|---|---|
| `bce` | 无 | 无 | 无 |
| `vat` | 无 | 无 | 有 |
| `random_aug_bce` | 随机合法变体 | 无 | 无 |
| `hard_aug_bce` | 困难合法变体 | 有 | 无 |
| `random_aug_vat` | 随机合法变体 | 无 | 有 |
| `hard_aug_vat` | 困难合法变体 | 有 | 有 |

两个数据集各训练六组，共 12 个 checkpoint。全部从相同预训练 CodeBERT 初始化，复用一个训练循环、模型和超参数，不复用微调权重。Seed 42，3 Epoch，micro batch 4、累积 2，有效 batch 8；AdamW 学习率 `1e-5`、weight decay `0.01`、梯度裁剪 `1.0`。参数及优化器使用 FP32，CUDA 主前向使用 BF16，VAT 辅助前向使用 FP32。

逻辑输入预算 2048，包含固定前缀与 BOS/EOS；原生窗口不超过 512，源码按半窗步长滑动。mean pooling 对重叠的源码 token 位置去重，前缀及边界 token 跨窗口平均，再接线性分类头。

增强沿用局部绑定重命名、函数名重命名、普通注释插入/删除/替换。检查解析、绑定、执行 token、物理行数和截断可见性；材料只来自当前训练数据集的 Train。随机组不查询模型；困难组使用当前模型的 BCE 进行束宽 2、深度 3 的搜索，每函数训练候选最多 12 个，每节点最多提议 24 次，新增注释不超过 64 token。原函数和最多一个选中变体的 BCE 按函数平均；不能增强的函数仍参与训练。

VAT 覆盖有效源码 token，排除前缀、特殊 token 和 padding。`epsilon=1`、`xi=0.001`、一次方向估计、系数 1；重叠窗口中相同源码位置共享扰动，按实际展开后的每函数 L2 范数归一化。辅助前向关闭 dropout 并保留梯度。

模型仅使用自身数据集的 Clean Valid 选择 Epoch 和阈值。阈值网格为 0.05–0.95、步长 0.01，按 MCC/F1/Accuracy/接近 0.5 决胜；Epoch 按 MCC/F1/Accuracy/AUC 决胜，完全相同保留较早轮次。两库训练代码没有专用分支。

## 训练与评价入口

Python 3.11，依赖版本见 `requirements.txt`。四库 Motivation 的完整命令见上文。对已有源码清单运行统一入口：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

MODEL=microsoft/codebert-base  # 也可使用本地 CodeBERT 目录
python -m vulnmechanism.cli train --dataset primevul --model "$MODEL" --device cuda:0
python -m vulnmechanism.cli train --dataset megavul --model "$MODEL" --device cuda:1
python -m vulnmechanism.cli evaluate --dataset primevul --device cuda:0
python -m vulnmechanism.cli evaluate --dataset megavul --device cuda:1
```

`train`、`candidates`、`evaluate` 均支持四库的 `--dataset` 参数，默认 `all`；方法默认六组，可用 `--methods bce` 仅运行普通基线。本轮实际只新增 BigVul、SecVulEval 的 BCE，其余方法和历史两库实验均不重复训练。已开始的训练拒绝重复启动；已有源码 JSONL 清单时，`prepare` 拒绝覆盖。原始数据目录与派生数据目录分开保存。

## 评价与输出

当前 `evaluate` 仅评价自身 Clean Valid、自身 Test 的 Clean/Fixed/Held-out/Adaptive。评价沿用 checkpoint 的训练 micro batch，避免更换 BF16 批大小导致阈值边缘决策变化。Test 不参与训练、选模或阈值选择。下方双向迁移表和 `comparison.*` 是已保存结果，本轮保留并不重跑。

CUDA 并行池化仍可能产生极小浮点波动：本次 MegaVul `hard_aug_vat` 和 `random_aug_bce` 重载后的 Valid AUPRC 与训练记录分别相差约 `9.05e-8` 和 `4.23e-8`，阈值、分类指标和 Epoch 选择一致。

Fixed 和 Held-out 候选在同一数据集的六模型间共享。留出命名、注释形式和组合见 `syntax.SURFACE_RULES`。Adaptive 各模型独立执行相同预算的动态搜索，每函数最多 48 个候选。严格语义保持与函数名等输入压力测试分别报告，并给出全部扰动的合并结果。

Worst-case 包含原函数和不能扰动的函数，分母是该库完整的独立 Test 队列。ASR 分母为可扰动且原本分类正确的函数，保存成功攻击数及分母。混淆矩阵按真实标签行、预测标签列排列为 `[[TN, FP], [FN, TP]]`。AUPRC 使用 average precision。

```text
results/
├── primevul/
│   ├── materials.json
│   ├── fixed.candidates.jsonl
│   ├── heldout.candidates.jsonl
│   └── <method>/
│       ├── best.pt
│       ├── best.training.jsonl
│       ├── config.json
│       ├── training.metrics.json
│       ├── valid.metrics.json
│       ├── test.metrics.json
│       ├── robustness.metrics.json
│       └── megavul.test.metrics.json
├── megavul/                 # 同样布局；迁移结果为 primevul.test.metrics.json
├── comparison.json
└── comparison.csv
```

各方法目录另保存逐样本预测和搜索进度。总表含 12 个训练配置、数据统计、144 行评价结果，统一报告 Accuracy、Precision、Recall、F1、MCC、AUC、AUPRC、ASR、混淆计数及 `delta_*`。跨库 `delta_*` 定义为目标 Test 指标减去该模型自身 Test 指标。

Git 发布代码、必要测试、成员表、统计、项目分配和结果总表。本轮权重、候选、日志、含源码 JSONL 和逐样本预测保存在本地，避免新克隆中的运行缓存影响复现。

```bash
python -m vulnmechanism.cli infer \
  --checkpoint results/megavul/hard_aug_vat/best.pt \
  --source-file function.c --device cuda:0
```

推理只需函数源码，输出 logit、概率、源阈值和预测；模型移动后可用 `--model` 指定预训练基座位置。

## 实际结果（Seed 42）

12 组均实际完成 3 Epoch，共 86,598 次优化器更新。以下 Test 评价队列为 PrimeVul 926 条、MegaVul 3,472 条。完整的 144 条结果、混淆矩阵、配置和数据统计见 [CSV](results/comparison.csv) 与 [JSON](results/comparison.json)。

### Clean Test

| 数据集 | 方法 | 选中 Epoch | 阈值 | Accuracy | F1 | MCC | AUC | AUPRC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PrimeVul | `bce` | 3 | 0.40 | 0.7948 | 0.7716 | 0.5913 | 0.8663 | 0.7936 |
| PrimeVul | `vat` | 2 | 0.43 | 0.7873 | 0.7635 | 0.5763 | 0.8545 | 0.7731 |
| PrimeVul | `random_aug_bce` | 3 | 0.27 | 0.7862 | 0.7648 | 0.5769 | 0.8651 | 0.7911 |
| PrimeVul | `hard_aug_bce` | 3 | 0.31 | 0.7905 | 0.7718 | 0.5884 | 0.8623 | 0.7951 |
| PrimeVul | `random_aug_vat` | 2 | 0.46 | 0.7808 | 0.7557 | 0.5625 | 0.8546 | 0.7795 |
| PrimeVul | `hard_aug_vat` | 3 | 0.18 | 0.7732 | 0.7597 | 0.5626 | 0.8596 | 0.7890 |
| MegaVul | `bce` | 1 | 0.51 | 0.7128 | 0.6730 | 0.4292 | 0.7991 | 0.7970 |
| MegaVul | `vat` | 1 | 0.42 | 0.7146 | 0.6896 | 0.4290 | 0.8006 | 0.7981 |
| MegaVul | `random_aug_bce` | 1 | 0.40 | 0.7120 | 0.6986 | 0.4231 | 0.7996 | 0.7970 |
| MegaVul | `hard_aug_bce` | 1 | 0.38 | 0.7074 | 0.6876 | 0.4139 | 0.7935 | 0.7908 |
| MegaVul | `random_aug_vat` | 1 | 0.40 | 0.7134 | 0.6993 | 0.4260 | 0.7987 | 0.7971 |
| MegaVul | `hard_aug_vat` | 1 | 0.43 | 0.7177 | 0.6972 | 0.4349 | 0.7948 | 0.7937 |

### 双向直接迁移

| 方向 | 方法 | Accuracy | F1 | MCC | AUC | AUPRC | ΔF1 | ΔMCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PrimeVul → MegaVul | `bce` | 0.7120 | 0.6691 | 0.4285 | 0.7806 | 0.7676 | -0.1025 | -0.1628 |
| PrimeVul → MegaVul | `vat` | 0.7088 | 0.6469 | 0.4307 | 0.7910 | 0.7860 | -0.1166 | -0.1455 |
| PrimeVul → MegaVul | `random_aug_bce` | 0.7149 | 0.6720 | 0.4346 | 0.7836 | 0.7718 | -0.0929 | -0.1423 |
| PrimeVul → MegaVul | `hard_aug_bce` | 0.7031 | 0.6645 | 0.4084 | 0.7830 | 0.7761 | -0.1073 | -0.1800 |
| PrimeVul → MegaVul | `random_aug_vat` | 0.7100 | 0.6497 | 0.4323 | 0.7860 | 0.7754 | -0.1060 | -0.1302 |
| PrimeVul → MegaVul | `hard_aug_vat` | 0.7143 | 0.6816 | 0.4300 | 0.7866 | 0.7892 | -0.0781 | -0.1326 |
| MegaVul → PrimeVul | `bce` | 0.7721 | 0.7616 | 0.5651 | 0.8593 | 0.7968 | +0.0886 | +0.1359 |
| MegaVul → PrimeVul | `vat` | 0.7667 | 0.7584 | 0.5583 | 0.8591 | 0.7934 | +0.0688 | +0.1293 |
| MegaVul → PrimeVul | `random_aug_bce` | 0.7581 | 0.7597 | 0.5605 | 0.8604 | 0.7978 | +0.0610 | +0.1374 |
| MegaVul → PrimeVul | `hard_aug_bce` | 0.7613 | 0.7616 | 0.5643 | 0.8629 | 0.8012 | +0.0740 | +0.1504 |
| MegaVul → PrimeVul | `random_aug_vat` | 0.7559 | 0.7580 | 0.5572 | 0.8604 | 0.7940 | +0.0587 | +0.1312 |
| MegaVul → PrimeVul | `hard_aug_vat` | 0.7570 | 0.7552 | 0.5510 | 0.8602 | 0.7978 | +0.0580 | +0.1161 |

### Test 鲁棒性

Acc 为包含原函数的 Worst-case Accuracy；ASR 为成功率（成功数/可扰动且原本正确数）。下表分别列出合并扰动、严格语义保持及输入压力测试。各设置的 F1、MCC、AUC、AUPRC 和混淆计数均保存在总表中。

**全部扰动**

| 数据集 | 方法 | Fixed Acc | ASR | Held-out Acc | ASR | Adaptive Acc | ASR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PrimeVul | `bce` | 0.7268 | 15.07% (63/418) | 0.7095 | 18.63% (79/424) | 0.6425 | 32.94% (141/428) |
| PrimeVul | `vat` | 0.7603 | 6.00% (25/417) | 0.7646 | 4.96% (21/423) | 0.7300 | 12.41% (53/427) |
| PrimeVul | `random_aug_bce` | 0.7603 | 5.78% (24/415) | 0.7635 | 4.99% (21/421) | 0.7214 | 14.12% (60/425) |
| PrimeVul | `hard_aug_bce` | 0.7786 | 2.63% (11/419) | 0.7786 | 2.59% (11/425) | 0.7592 | 6.76% (29/429) |
| PrimeVul | `random_aug_vat` | 0.7657 | 3.37% (14/415) | 0.7678 | 2.85% (12/421) | 0.7408 | 8.71% (37/425) |
| PrimeVul | `hard_aug_vat` | 0.7613 | 2.70% (11/407) | 0.7592 | 3.15% (13/413) | 0.7505 | 5.04% (21/417) |
| MegaVul | `bce` | 0.6613 | 8.56% (179/2092) | 0.6685 | 7.41% (154/2079) | 0.6092 | 17.20% (360/2093) |
| MegaVul | `vat` | 0.6682 | 7.67% (161/2098) | 0.6743 | 6.71% (140/2085) | 0.6109 | 17.15% (360/2099) |
| MegaVul | `random_aug_bce` | 0.6668 | 7.52% (157/2089) | 0.6694 | 7.13% (148/2076) | 0.6118 | 16.65% (348/2090) |
| MegaVul | `hard_aug_bce` | 0.6892 | 3.04% (63/2070) | 0.6901 | 2.92% (60/2057) | 0.6668 | 6.81% (141/2071) |
| MegaVul | `random_aug_vat` | 0.6861 | 4.53% (95/2096) | 0.6892 | 4.03% (84/2083) | 0.6526 | 10.06% (211/2097) |
| MegaVul | `hard_aug_vat` | 0.7031 | 2.42% (51/2108) | 0.7016 | 2.67% (56/2095) | 0.6889 | 4.74% (100/2109) |

**严格语义保持**

| 数据集 | 方法 | Fixed Acc | ASR | Held-out Acc | ASR | Adaptive Acc | ASR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PrimeVul | `bce` | 0.7343 | 13.43% (56/417) | 0.7149 | 17.54% (74/422) | 0.6868 | 23.47% (100/426) |
| PrimeVul | `vat` | 0.7613 | 5.77% (24/416) | 0.7657 | 4.75% (20/421) | 0.7343 | 11.53% (49/425) |
| PrimeVul | `random_aug_bce` | 0.7603 | 5.80% (24/414) | 0.7635 | 5.01% (21/419) | 0.7300 | 12.29% (52/423) |
| PrimeVul | `hard_aug_bce` | 0.7786 | 2.63% (11/418) | 0.7797 | 2.36% (10/423) | 0.7646 | 5.62% (24/427) |
| PrimeVul | `random_aug_vat` | 0.7667 | 3.14% (13/414) | 0.7678 | 2.86% (12/419) | 0.7419 | 8.51% (36/423) |
| PrimeVul | `hard_aug_vat` | 0.7613 | 2.71% (11/406) | 0.7592 | 3.16% (13/411) | 0.7505 | 5.06% (21/415) |
| MegaVul | `bce` | 0.6751 | 6.28% (131/2085) | 0.6803 | 5.46% (113/2068) | 0.6426 | 11.70% (244/2085) |
| MegaVul | `vat` | 0.6711 | 7.22% (151/2091) | 0.6780 | 6.12% (127/2074) | 0.6290 | 14.20% (297/2091) |
| MegaVul | `random_aug_bce` | 0.6789 | 5.52% (115/2082) | 0.6838 | 4.75% (98/2065) | 0.6524 | 9.94% (207/2082) |
| MegaVul | `hard_aug_bce` | 0.6912 | 2.71% (56/2063) | 0.6921 | 2.59% (53/2046) | 0.6728 | 5.82% (120/2063) |
| MegaVul | `random_aug_vat` | 0.6915 | 3.64% (76/2089) | 0.6921 | 3.57% (74/2072) | 0.6694 | 7.32% (153/2089) |
| MegaVul | `hard_aug_vat` | 0.7031 | 2.43% (51/2101) | 0.7016 | 2.69% (56/2084) | 0.6918 | 4.28% (90/2101) |

**输入压力测试**

| 数据集 | 方法 | Fixed Acc | ASR | Held-out Acc | ASR | Adaptive Acc | ASR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PrimeVul | `bce` | 0.7840 | 4.72% (10/212) | 0.7808 | 6.07% (13/214) | 0.7192 | 32.56% (70/215) |
| PrimeVul | `vat` | 0.7840 | 1.42% (3/212) | 0.7851 | 0.93% (2/214) | 0.7678 | 8.37% (18/215) |
| PrimeVul | `random_aug_bce` | 0.7851 | 0.48% (1/210) | 0.7829 | 1.42% (3/212) | 0.7603 | 11.27% (24/213) |
| PrimeVul | `hard_aug_bce` | 0.7905 | 0.00% (0/211) | 0.7883 | 0.94% (2/213) | 0.7786 | 5.14% (11/214) |
| PrimeVul | `random_aug_vat` | 0.7786 | 0.96% (2/209) | 0.7808 | 0.00% (0/211) | 0.7678 | 5.66% (12/212) |
| PrimeVul | `hard_aug_vat` | 0.7711 | 0.96% (2/208) | 0.7700 | 1.43% (3/210) | 0.7646 | 3.79% (8/211) |
| MegaVul | `bce` | 0.6889 | 6.13% (83/1353) | 0.6918 | 5.38% (73/1356) | 0.6483 | 16.51% (224/1357) |
| MegaVul | `vat` | 0.6999 | 3.74% (51/1364) | 0.6999 | 3.73% (51/1367) | 0.6544 | 15.28% (209/1368) |
| MegaVul | `random_aug_bce` | 0.6895 | 5.78% (78/1349) | 0.6889 | 5.92% (80/1352) | 0.6446 | 17.29% (234/1353) |
| MegaVul | `hard_aug_bce` | 0.7028 | 1.19% (16/1347) | 0.7022 | 1.33% (18/1350) | 0.6855 | 5.63% (76/1351) |
| MegaVul | `random_aug_vat` | 0.7036 | 2.50% (34/1361) | 0.7028 | 2.71% (37/1364) | 0.6763 | 9.45% (129/1365) |
| MegaVul | `hard_aug_vat` | 0.7163 | 0.37% (5/1369) | 0.7163 | 0.36% (5/1372) | 0.7022 | 3.93% (54/1373) |

与 `bce` 相比，`hard_aug_vat` 的严格 Adaptive ASR 从 23.47% 降至 5.06%（PrimeVul），从 11.70% 降至 4.28%（MegaVul）。跨库 F1 在 PrimeVul → MegaVul 上变化 +0.0125，在 MegaVul → PrimeVul 上变化 -0.0064。

## 代码

`data.py` 负责原生读取、项目划分、保留正例的平衡及精确重叠标记；`model.py`、`syntax.py`、`augmentation.py`、`vat.py` 分别实现模型、绑定安全变换、动态搜索和普通 VAT；`training.py`、`evaluation.py`、`cli.py` 提供两个数据集共享的训练、评价和命令入口。

测试使用本地小型 RoBERTa，无需下载模型；覆盖全部正例保留、项目正例比例、精确泄漏排除、六方法及 CSD 训练路径、仅自身 Valid 选模、CSD 分解与正交损失、共享头推理、增量预测缓存、双向 Test 固定队列与阈值、窗口池化、VAT 梯度/范数、严格权重加载、攻击分母和结果汇总。
