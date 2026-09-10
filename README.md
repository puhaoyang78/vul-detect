# Function-Level Vulnerability Classification

本仓库当前只保留一个直接的函数级 C/C++ 漏洞二分类实验：

> **给定单个函数及其函数内 CPG，预测数据集原始二分类标签 0 / 1。**

不再判断 vulnerable/fixed pair，不使用 patch、fix commit、CVE 描述、caller/callee 或仓库上下文完成分类。

## Pipeline

```text
single C/C++ function
        ↓
standalone Joern AST / CFG / CDG / DDG
        ↓
┌──────────────────────────────────────┐
│ Baseline: raw source                 │
│ CPG:      raw source + compact CPG   │
└──────────────────────────────────────┘
        ↓
frozen Qwen2.5-Coder-7B representation
        ↓
linear binary classifier
        ↓
0 / 1
```

第一阶段实验只回答一个问题：

> 函数内 CPG 信息能否在相同 Qwen 编码器和相同二分类设置下，相比纯源码输入提升函数级漏洞检测效果？

## Repository layout

```text
vulnmechanism/
  syntax.py       standalone C/C++ function parsing
  process.py      external-process execution
  cpg.py          standalone Joern AST/CFG/CDG/DDG extraction
  mechanism.py    compact CPG relation extraction and dataset building
  model.py        frozen-Qwen baseline and CPG classifier
  cli.py          build / train / eval commands

tests/
  test_vulnmechanism.py
```

## Input format

`build` 接收 JSONL，每行是一条独立函数样本。至少需要样本 ID、函数源码和原始二分类标签：

```json
{
  "sample_key": "example-1",
  "function_name": "foo",
  "language": "c",
  "function": "int foo(char *buf, int len) { return buf[len]; }",
  "label": 1,
  "split": "train"
}
```

支持的源码字段：

```text
function / func / source / code / func_before
```

支持的标签字段：

```text
label / target
```

标签必须是二分类 `0/1`。`BENIGN/VULNERABLE` 字符串也会分别映射到 `0/1`。

如果输入已经包含 `train / valid / test`，程序直接保留该划分。如果没有 `split`，训练和评估阶段按 `sample_key` 的稳定哈希产生 70/15/15 划分。

## Build dataset

```bash
python -m vulnmechanism.cli build \
  --samples data/functions.jsonl \
  --output data/function_dataset.jsonl
```

每个函数片段单独运行 Joern，并导出：

```text
AST
CFG
CDG
DDG
```

随后只保留与以下安全相关操作相连的紧凑关系及其一跳上下文：

```text
control structures
comparisons
allocation
memory APIs
pointer/index access
arithmetic
```

输出的每条记录仍然对应一个函数，并保留其原始 0/1 标签。

## Train

默认使用本地模型：

```text
/home/phy/models/Qwen2.5-Coder-7B-Instruct
```

运行：

```bash
python -m vulnmechanism.cli train \
  --dataset data/function_dataset.jsonl \
  --model /home/phy/models/Qwen2.5-Coder-7B-Instruct \
  --output results/function_classifier.pt
```

当前 Qwen 参数冻结，只训练两个独立线性分类头：

```text
Baseline:
raw source
→ frozen Qwen
→ Linear
→ 0 / 1

CPG:
raw source + compact intra-function CPG
→ frozen Qwen
→ Linear
→ 0 / 1
```

两个模型使用相同训练数据、Qwen 编码器、epoch、学习率和分类头结构。

## Evaluate

```bash
python -m vulnmechanism.cli eval \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/function_classifier.pt \
  --split test
```

报告：

```text
samples
positive / negative
Accuracy
Precision
Recall
F1
MCC
AUC
```

正式比较时重点观察同一 test split 下的 `baseline` 与 `cpg`。

## Environment

```text
Joern: /home/phy/joern
JDK: /home/phy/jdk21
Qwen: /home/phy/models/Qwen2.5-Coder-7B-Instruct
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

`data/` 和 `results/` 为运行时目录，不进入版本控制。
