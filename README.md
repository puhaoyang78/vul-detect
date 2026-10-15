# Dynamic Hard Augmentation for Vulnerability Detection

对单个 C/C++ 函数预测漏洞概率和 0/1 标签。项目使用全参数 CodeBERT、原生滑窗、去重 mean pooling 和线性分类头，对比动态困难样本增强及五组基线/对照。

## 数据

原始数据分别来自 [PrimeVul v0.1](https://github.com/DLVulDet/PrimeVul) 和 [MegaVul](https://github.com/Icyrockton/MegaVul)。预训练编码器为 [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)。请从上游获取数据和预训练模型，保留各自的许可与引用要求。

将原始文件放在一个目录下：

```text
RAW/
├── PrimeVul_v0.1/
│   ├── primevul_train.jsonl
│   ├── primevul_valid.jsonl
│   └── primevul_test.jsonl
└── megavul/
    └── megavul.json
```

- PrimeVul 直接读取官方 `func` 和 `target`，保留官方 Train/Valid/Test 归属。
- MegaVul 直接使用 `is_vul`：正例读取 `func_before`，负例读取 `func`，每条上游记录只产生一个函数。对去首尾空格并 casefold 的项目名排序，再用 `random.Random(42)` 打乱，按项目数的约 80/10/10 划分。
- 两库独立清理。依次合并精确重复、去注释与空白后的词法重复，以及 token 5-gram 集合 Jaccard ≥ 0.90 的近重复。字符串、数字和操作符边界保留。近重复使用精确候选连接与实际 token 集合复核，不使用随机 LSH。
- 重复关系按连通组处理。组内标签冲突时移除全部成员；其余组优先保留 Test，其次 Valid、Train，同 Split 保留原始文件顺序较早者。样本不会迁移 Split。
- 清理后，各 Split 分别用 Seed 42 从每类抽取与少数类相同数量的样本，输出仍按原始顺序排列。自然比例成员同时保留。
- 不使用其他数据集筛选，不强制跨库项目、CVE、提交或近重复隔离，不按语法解析是否成功筛除函数。

每库输出 `train/valid/test.jsonl`、对应的 `*.natural.jsonl`、`members.csv` 和 `statistics.json`。成员表记录上游行号、原生标签、划分和是否进入平衡子集；MegaVul 另保存项目分配 `projects.json`。Git 发布成员表、统计及 `results/comparison.json/csv` 总表，训练配置一并写入 JSON 总表。含源码的清单、逐模型运行日志、权重和预测保存在本地，使新克隆仓库可以直接复现完整流程。

本次 Seed 42 全量重建结果如下，平衡清单的每类样本数均为表中数字的一半：

| 数据集 | Train | Valid | Test |
|---|---:|---:|---:|
| PrimeVul | 3,950 | 538 | 406 |
| MegaVul | 21,168 | 8,632 | 2,548 |

MegaVul 自然比例 Valid 共 100,196 个函数，其中正例 4,316、负例 95,880。划分时项目数 Train/Valid/Test 为 848/106/107。标签冲突组共移除 PrimeVul 7,329 条、MegaVul 703 条记录；这些成员不再进入训练或评价。详细统计见 [PrimeVul](data/primevul/statistics.json) 和 [MegaVul](data/megavul/statistics.json)，最终成员见各目录的 `members.csv`。

## 方法与训练协议

| 方法 | 源码增强 | 当前模型搜索 | 普通 VAT |
|---|---|---|---|
| `bce` | 无 | 无 | 无 |
| `vat` | 无 | 无 | 有 |
| `random_aug_bce` | 随机合法变体 | 无 | 无 |
| `hard_aug_bce` | 困难合法变体 | 有 | 无 |
| `random_aug_vat` | 随机合法变体 | 无 | 有 |
| `hard_aug_vat` | 困难合法变体 | 有 | 有 |

六组共享相同的新 PrimeVul 成员和预训练初始化。Seed 42，训练 3 Epoch，micro batch 4、梯度累积 2，有效 batch 8；AdamW 学习率 `1e-5`、weight decay `0.01`、梯度裁剪 `1.0`。使用 FP32 参数与优化器，CUDA 主前向使用 BF16，VAT 辅助前向使用 FP32。

逻辑输入预算 2048，包含固定任务前缀和 BOS/EOS。编码器窗口不超过原生 512，源码窗口以半窗步长滑动；重复源码 token 的池化贡献去重，前缀与边界 token 在窗口间平均，每个函数产生一个 logit。

增强沿用局部绑定重命名、函数名重命名、普通注释插入/删除/替换。变换检查解析结果、绑定对应、执行 token、物理行数和截断可见性。生成材料只来自新 PrimeVul Train。随机组不查询模型；困难组以当前模型 BCE 搜索，束宽 2、深度 3，每函数最多 12 个训练候选，每节点最多 24 次提议，新增注释不超过 64 token。原版和最多一个选中变体的 BCE 先按函数平均；没有合法变体的函数仍参与训练。

VAT 使用全部有效源码 token，排除前缀、特殊 token 与 padding；`epsilon=1`、`xi=0.001`、一次方向估计、系数 1。相同源码位置在重叠窗口中共享扰动，按实际展开后的每函数 L2 范数归一化。辅助前向关闭 dropout，保留梯度。

只用 Clean PrimeVul Valid 选 Epoch 和阈值。阈值候选为 0.05–0.95、间隔 0.01，按 MCC、F1、Accuracy、接近 0.5 的顺序决胜；Epoch 按 MCC、F1、Accuracy、AUC 决胜，完全相同则保留较早 Epoch。训练入口始终从预训练基座创建模型，已有运行目录会拒绝重复训练。

## 复现

Python 3.11，依赖版本见 `requirements.txt`。在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# 修改 RAW 为上述原始数据目录；MODEL 也可改为本地 CodeBERT 目录。
RAW=/path/to/raw
MODEL=microsoft/codebert-base
python -m vulnmechanism.cli prepare --raw-root "$RAW"
python -m vulnmechanism.cli train --model "$MODEL" --device cuda:0
python -m vulnmechanism.cli candidates --model "$MODEL"
python -m vulnmechanism.cli evaluate --device cuda:0
python -m vulnmechanism.cli report
```

`train` 和 `evaluate` 默认运行六组，可用 `--methods hard_aug_bce hard_aug_vat` 指定成员。数据准备只写入派生数据目录，不修改原始文件。新克隆仓库中的成员表与统计可直接重建；已有源码 JSONL 清单时会拒绝重复准备，需改用尚无源码清单的输出目录。已经正常运行的训练不应重复启动。

## 评价与推理

Clean Valid/Test、固定扰动、留出扰动和自适应扰动统一使用已保存的 PrimeVul Valid 权重及阈值。评价自动沿用 checkpoint 的训练 micro batch（本次为 4），与 Clean Valid 选模保持一致。固定/留出候选由所有模型共享；自适应攻击各模型独立进行相同预算的搜索，每函数最多 48 个候选。留出命名、注释形式及组合见 `syntax.SURFACE_RULES`。

最坏情况评价包含原函数和无法扰动的函数，分母是完整 Split；ASR 分母是可扰动且原本分类正确的函数，并保存分子与分母。函数名变换属于单函数输入压力测试，严格语义保持结果单独报告。

MegaVul 只推理自然比例 Valid，再从同一份预测提取平衡 Valid。权重与阈值来自 PrimeVul，不进行目标库微调、选模或校准。MegaVul Test 仅保存数据清单，评价入口不读取它。

每组在 `results/<method>/` 保存配置、训练进度、最佳权重、逐样本预测和指标。`results/comparison.json/csv` 汇总 Accuracy、Precision、Recall、F1、MCC、AUC、AUPRC、ASR、混淆计数及跨库变化；AUPRC 使用 average precision。JSON 中混淆矩阵按真实标签行、预测标签列排列：`[[TN, FP], [FN, TP]]`。跨库变化同时以 PrimeVul Clean Valid/Test 为参照。

```bash
python -m vulnmechanism.cli infer \
  --checkpoint results/hard_aug_bce/best.pt \
  --source-file function.c --device cuda:0
```

推理仅需函数源码，输出 logit、概率、保存的阈值和预测。移动模型文件后，可用 `--model` 指定新的 CodeBERT 位置。

## 代码结构

```text
vulnmechanism/
├── data.py          # 原生数据、库内去重、平衡与成员清单
├── model.py         # CodeBERT、滑窗、池化与加载
├── syntax.py        # 绑定安全的源码变换
├── augmentation.py  # 可见性检查与动态搜索
├── vat.py           # 普通全局 VAT
├── training.py      # 六组共享训练
├── evaluation.py    # Clean、鲁棒性、MegaVul 与汇总
└── cli.py           # 统一命令入口
```

测试使用本地构造的小型 RoBERTa，无需下载模型；覆盖原生标签、项目划分、精确近重复连接、冲突清理、平衡、窗口池化、六条训练路径、VAT 梯度与范数、严格权重加载、攻击分母、固定阈值和 MegaVul Test 隔离。

## Seed 42 正式结果

六组均实际完成 3 Epoch，每轮 494 次、合计 1,482 次优化步；全部由 Clean Valid 选择第 3 轮。以下 Test 共 406 个函数，正负各 203。

| 方法 | 源阈值 | Accuracy | F1 | MCC | AUC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| `bce` | 0.22 | 0.6946 | 0.7169 | 0.3941 | 0.7422 | 0.6983 |
| `vat` | 0.30 | 0.6970 | 0.7146 | 0.3971 | 0.7651 | 0.7350 |
| `random_aug_bce` | 0.24 | 0.6847 | 0.7009 | 0.3716 | 0.7533 | 0.7348 |
| `hard_aug_bce` | 0.23 | 0.6970 | 0.7285 | 0.4051 | 0.7592 | 0.7352 |
| `random_aug_vat` | 0.22 | 0.6872 | 0.7171 | 0.3831 | 0.7568 | 0.7239 |
| `hard_aug_vat` | 0.39 | 0.6897 | 0.6912 | 0.3793 | 0.7596 | 0.7204 |

下面为 Test 的全部扰动族结果，包含函数名输入压力测试；严格语义保持结果单独保存在总表。最坏性能的完整分母仍为 406。

| 方法 | 固定 ASR | 留出 ASR | 自适应 ASR | 自适应最坏 Accuracy | 自适应最坏 MCC |
|---|---:|---:|---:|---:|---:|
| `bce` | 11.05% | 11.60% | 32.58% | 0.5517 | 0.1102 |
| `vat` | 9.09% | 9.09% | 24.86% | 0.5911 | 0.1852 |
| `random_aug_bce` | 9.04% | 9.60% | 30.46% | 0.5542 | 0.1107 |
| `hard_aug_bce` | 8.15% | 8.70% | 19.34% | 0.6108 | 0.2320 |
| `random_aug_vat` | 5.71% | 5.71% | 16.86% | 0.6158 | 0.2402 |
| `hard_aug_vat` | 4.60% | 4.60% | 9.94% | 0.6478 | 0.2960 |

MegaVul 平衡 Valid 共 8,632 个函数，仍使用上表中的源阈值。ΔMCC 为目标库减去 PrimeVul Clean Test。

| 方法 | Accuracy | F1 | MCC | AUC | AUPRC | ΔMCC |
|---|---:|---:|---:|---:|---:|---:|
| `bce` | 0.6361 | 0.6901 | 0.2905 | 0.6933 | 0.6593 | -0.1036 |
| `vat` | 0.6660 | 0.6895 | 0.3359 | 0.7225 | 0.6938 | -0.0612 |
| `random_aug_bce` | 0.6435 | 0.6769 | 0.2934 | 0.6919 | 0.6619 | -0.0783 |
| `hard_aug_bce` | 0.6336 | 0.6947 | 0.2916 | 0.6922 | 0.6559 | -0.1135 |
| `random_aug_vat` | 0.6649 | 0.6974 | 0.3376 | 0.7261 | 0.6989 | -0.0455 |
| `hard_aug_vat` | 0.6677 | 0.6717 | 0.3356 | 0.7190 | 0.6889 | -0.0437 |

MegaVul 自然比例 Valid 共 100,196 个函数，正例率为 4.31%。

| 方法 | Accuracy | F1 | MCC | AUC | AUPRC |
|---|---:|---:|---:|---:|---:|
| `bce` | 0.4757 | 0.1175 | 0.1107 | 0.6922 | 0.0840 |
| `vat` | 0.5969 | 0.1368 | 0.1364 | 0.7215 | 0.0984 |
| `random_aug_bce` | 0.5465 | 0.1242 | 0.1155 | 0.6909 | 0.0840 |
| `hard_aug_bce` | 0.4559 | 0.1166 | 0.1119 | 0.6917 | 0.0823 |
| `random_aug_vat` | 0.5686 | 0.1336 | 0.1352 | 0.7250 | 0.1009 |
| `hard_aug_vat` | 0.6540 | 0.1448 | 0.1406 | 0.7190 | 0.0962 |

本轮 `hard_aug_bce` 的 Clean Test MCC 最高，`hard_aug_vat` 的自适应 ASR 最低，`random_aug_vat` 的 MegaVul 平衡 MCC/AUC 最高。困难增强在鲁棒性上的收益没有表现为所有 Clean 与跨库指标的一致优势。

完整 Valid/Test 指标、混淆矩阵、ASR 分子与分母以及训练配置见 [JSON 总表](results/comparison.json)；便于筛选的长表见 [CSV](results/comparison.csv)。
