# Dynamic Hard Augmentation for Vulnerability Detection

使用相同的全参数 CodeBERT，对比动态困难源码增强及五组基线。两个数据集分别训练，研究自身 Test 上的鲁棒性，以及固定源模型、源阈值的双向跨库泛化。

## 数据

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
- 只检查原始函数字符串完全相等的训练—Test 重叠。训练集合指实际平衡训练成员，不包含未抽中的负例。Test 清单保留全部正例，`train_overlap` 记录与哪一库训练集重叠；评价时排除与任一训练集精确重叠的 Test 成员。同一库的域内和跨域评价使用完全相同的剩余 Test 队列，不再次抽样。

Seed 42 重建后的平衡清单如下，正负各占一半：

| 数据集 | Train | Valid | Test | 保留原始正例总数 |
|---|---:|---:|---:|---:|
| PrimeVul | 9,724 | 1,186 | 1,098 | 6,004 |
| MegaVul | 28,760 | 3,594 | 3,596 | 17,975 |

MegaVul 正例 Train/Valid/Test 为 **14,380/1,797/1,798**，项目数为 394/333/334。两库均未从清单中删除正例。

精确重叠仅影响 Test 评价分母：

| Test 数据集 | 与 PrimeVul Train 重叠 | 与 MegaVul Train 重叠 | 排除并集（正/负） | 实际评价（正/负） |
|---|---:|---:|---:|---:|
| PrimeVul | 0 | 172 | 172（161/11） | 926（388/538） |
| MegaVul | 91 | 33 | 124（115/9） | 3,472（1,683/1,789） |

评价队列经过必要的精确泄漏排除后不再强求 1:1。完整数量见各库 `statistics.json`；最终平衡成员、上游文件、记录位置（PrimeVul 为 JSONL 行号）及重叠标记见 `members.csv`。项目分配另存 `data/megavul/projects.json`。数据准备只写派生目录，保留上游原始文件。

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

## 复现

Python 3.11，依赖版本见 `requirements.txt`。在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

RAW=/path/to/raw
MODEL=microsoft/codebert-base  # 也可使用本地 CodeBERT 目录
python -m vulnmechanism.cli prepare --raw-root "$RAW"
python -m vulnmechanism.cli train --dataset all --model "$MODEL" --device cuda:0
python -m vulnmechanism.cli candidates --dataset all --model "$MODEL"
python -m vulnmechanism.cli evaluate --dataset all --device cuda:0
python -m vulnmechanism.cli report
```

`train`、`candidates`、`evaluate` 均用 `--dataset primevul` 或 `--dataset megavul` 指定数据集，默认 `all`。训练和评价可用 `--methods bce vat` 指定方法。已开始的训练拒绝重复启动；新克隆中的成员表和统计可以重建，已有源码 JSONL 清单时拒绝覆盖。原始数据目录与派生数据目录分开保存。

## 评价与输出

每个模型评价自身 Clean Valid、自身 Test 的 Clean/Fixed/Held-out/Adaptive，以及另一库 Test 的直接迁移。评价沿用 checkpoint 的训练 micro batch，避免更换 BF16 批大小导致阈值边缘决策变化。Test 不参与训练、选模或阈值选择；跨库只使用源权重、源阈值，不读取目标 Valid。

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

测试使用本地小型 RoBERTa，无需下载模型；覆盖全部正例保留、项目正例比例、精确泄漏排除、12 条训练路径、仅自身 Valid 选模、双向 Test 固定阈值、窗口池化、VAT 梯度/范数、严格权重加载、攻击分母和结果汇总。
