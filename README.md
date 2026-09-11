# Function-Level Vulnerability Classification

当前主线是：

> **CPG-Grounded Vulnerability Semantic Learning for Function-Level C/C++ Classification**

最终任务仍然是单函数原始二分类标签 `0 / 1`。不使用 vulnerable/fixed pair、patch、CVE 描述、caller/callee 或仓库上下文作为测试输入。

## Pipeline

```text
single C/C++ function
        ↓
standalone Joern AST / CFG / CDG / DDG
        ↓
CPG-grounded vulnerability semantic extraction
        ↓
MEMORY / OBJECT / DEPENDENCY / GUARD / LIFETIME / RISK_CANDIDATE facts
        ↓
┌─────────────────────────────────────────────────┐
│ Baseline: source                                │
│ Proposed: source + CPG-derived semantic facts   │
└─────────────────────────────────────────────────┘
        ↓
Qwen2.5-Coder-7B + LoRA fine-tuning
        ↓
masked mean pooling + binary classification head
        ↓
0 / 1
```

Baseline 与 Proposed 使用相同数据、源码 token budget、LoRA 配置、pooling、分类头和训练设置。唯一实验变量是 Proposed 额外获得 CPG 派生的漏洞语义事实。

## CPG-derived semantics

当前从函数内 CPG 提取：

```text
MEMORY
  WRITE / READ / INDEX / DEREF

OBJECT
  ALLOC / STATIC_CAPACITY

DEPENDENCY
  DATA_DEP / PARAMETER_DEP / SIZE_ARITHMETIC

GUARD
  CONDITION / GUARD_PROTECTS

LIFETIME
  FREE

RISK_CANDIDATE
  UNGUARDED_WRITE_EXTENT
  WRITE_EXCEEDS_STATIC_CAPACITY
  UNCHECKED_INDEX
  UNGUARDED_DEREFERENCE
  UNGUARDED_SIZE_ARITHMETIC
  UNBOUNDED_WRITE_API
  DOUBLE_FREE
  USE_AFTER_FREE
```

`RISK_CANDIDATE` 只是由程序关系得到的保守候选事实，不作为人工标签，也不会覆盖数据集原始 `0/1` 标签。最终分类仍完全由训练数据监督。

## Build

输入 `data/functions.jsonl` 每行至少包含：

```json
{
  "sample_key": "example-1",
  "language": "c",
  "function": "int foo(char *buf, int len) { return buf[len]; }",
  "label": 1,
  "split": "train"
}
```

构建：

```bash
python -m vulnmechanism.cli build \
  --samples data/functions.jsonl \
  --output data/function_dataset.jsonl
```

输出保留原始源码和标签，并新增：

```text
graph
semantic_facts
semantic_tags
relation_count
semantic_fact_count
```

构建支持按 `sample_key` 复用同 schema 的已完成记录。失败样本写入与输出同名的 `.errors.jsonl`。

语义 schema 变化时旧记录不会被静默复用。

## Train

安装依赖：

```bash
python -m pip install -r requirements.txt
```

训练：

```bash
python -m vulnmechanism.cli train \
  --dataset data/function_dataset.jsonl \
  --model /home/phy/models/Qwen2.5-Coder-7B-Instruct \
  --output results/function_classifier.pt
```

默认设置：

```text
source token budget:   1536
semantic token budget: 384
LoRA targets: q_proj, k_proj, v_proj, o_proj
LoRA r: 16
LoRA alpha: 32
LoRA dropout: 0.05
batch size: 1
gradient accumulation: 8
epochs: 3
learning rate: 2e-4
```

源码部分在 Baseline 与 Proposed 中使用完全相同的截断结果；Proposed 的语义事实使用独立 token budget，因此不会通过加入语义信息进一步截短源码。

模型使用 masked mean pooling，不再取最后一个 token 作为函数表示。

验证集按 AUC 选择每个 variant 的最佳 epoch，并分别保存 Baseline 和 Proposed 的 LoRA adapter 与分类头。

## Evaluate

```bash
python -m vulnmechanism.cli eval \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/function_classifier.pt \
  --split test
```

报告：

```text
Accuracy
Precision
Recall
F1
MCC
AUC
```

## Environment

```text
Joern: /home/phy/joern
JDK: /home/phy/jdk21
Qwen: /home/phy/models/Qwen2.5-Coder-7B-Instruct
```

轻量单元测试：

```bash
python -m unittest discover -s tests -v
```

`data/` 和 `results/` 为运行时目录，不进入版本控制。
