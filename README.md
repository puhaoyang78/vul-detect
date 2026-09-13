# CPG-Guided Vulnerability-Semantic Learning

本仓库研究函数级 C/C++ 漏洞二分类：

> **利用静态程序分析提取 vulnerability-related program semantics，并用这些语义引导 Code LLM 学习函数的原始 0/1 漏洞标签。**

测试阶段的预测单位始终是单个函数。不使用 vulnerable/fixed pair、fix commit、CVE 描述、caller/callee 或仓库上下文作为分类输入。

## Method

```text
single C/C++ function
        │
        ├────────────── source code ───────────────────────────┐
        │                                                      │
        └→ Joern CPG                                           │
             ↓                                                 │
   CPG-based Vulnerability Semantic Extraction                 │
             ↓                                                 │
   vulnerability-related program semantics                     │
             │                                                 │
             └──── Cross-Attention-based Semantic Fusion ──────┤
                                                               ↓
                                                       LoRA Code LLM
                                                               ↓
                                                vulnerability classification
                                                               │
                                  optional Vulnerability-Feature Supervision
```

### CPG-based Vulnerability Semantic Extraction

Joern 为每个函数独立导出 AST / CFG / CDG / DDG。程序分析首先定位 security-relevant program points，并进一步提取：

- memory operations: write / read / array access / pointer dereference
- memory objects: allocation / static capacity
- data dependence: data dependence / parameter dependence / size arithmetic
- control constraints: control condition / guard / bounds check / null check
- lifetime semantics: deallocation and free-to-use/free relations
- potential vulnerability patterns: unchecked write/index/dereference, unbounded write, double free, use after free, etc.

`POTENTIAL_PATTERN` 是静态分析得到的候选模式，不作为 ground-truth vulnerability label。最终监督始终来自数据集原始 `0/1` 标签。

### Cross-Attention-based Semantic Fusion

源码与 structured vulnerability semantics 分别经过同一个 LoRA-Qwen 编码器。低维 cross-attention 使用源码表示作为 query、漏洞语义表示作为 key/value，在源码 token 上融合程序分析语义，再完成函数级分类。

### Vulnerability-Feature Supervision

完整模型在分类损失之外增加 auxiliary multi-label supervision。辅助目标来自 CPG 提取出的 vulnerability-related features，例如 memory write、array access、parameter dependence、bounds check 和 lifetime relation。

该辅助头只读取 **source representation before semantic fusion**，避免直接从已经输入的 semantic text 中复制答案；目的是要求源码表示本身编码这些 vulnerability-related features。

## Model variants

所有 variant 使用相同的 Qwen2.5-Coder-7B、LoRA 配置、源码 token budget、数据划分和二分类指标。

| Variant | Input / module | Purpose |
| --- | --- | --- |
| `baseline` | source only | Code LLM baseline |
| `raw_cpg` | source + compact CPG relations | test whether raw structural context helps |
| `semantic_concat` | source + vulnerability semantics by concatenation | semantic extraction only |
| `semantic_fusion` | separate source/semantics + cross-attention | add semantic fusion |
| `full` | semantic fusion + vulnerability-feature supervision | complete method |

这组 variant 对应主消融链：

```text
baseline
  → raw_cpg
  → semantic_concat
  → semantic_fusion
  → full
```

其中 `raw_cpg` 与 `semantic_concat` 使用相同 context token budget，因此 raw CPG 与 vulnerability semantics 的比较不会由额外输入长度造成。

## Semantic-group ablation

Structured vulnerability semantics 按五组组织：

```text
memory
  MEMORY_OPERATION + MEMORY_OBJECT

dependence
  DATA_DEPENDENCE

constraint
  CONTROL_CONSTRAINT

lifetime
  LIFETIME

pattern
  POTENTIAL_PATTERN
```

使用 `--exclude-groups` 删除指定组。例如：

```bash
python -m vulnmechanism.cli train \
  --variant full \
  --exclude-groups memory \
  --dataset data/function_dataset.jsonl \
  --output results/full_wo_memory.pt
```

对 `full` 做消融时，被删除语义组对应的 auxiliary feature targets 也会同时 mask，避免通过 feature supervision 重新引入被删除的信息。

## Repository layout

```text
vulnmechanism/
  syntax.py       standalone C/C++ function parsing
  process.py      external-process execution
  cpg.py          standalone Joern AST/CFG/CDG/DDG extraction
  semantics.py    vulnerability-related program semantic extraction
  dataset.py      CPG/semantic dataset construction and resume
  model.py        LoRA baselines, semantic fusion and feature supervision
  cli.py          build / train / eval commands

tests/
  test_vulnmechanism.py
```

## Input format

`build` 接收 JSONL，每行一个函数样本：

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

源码字段支持：

```text
function / func / source / code / func_before
```

标签字段支持：

```text
label / target
```

原始标签必须是 `0/1`；`BENIGN/VULNERABLE` 字符串分别映射为 `0/1`。已有 `train / valid / test` 会被直接保留；没有 `split` 时，训练阶段按 `sample_key` 稳定哈希得到 70/15/15 划分。

## Build dataset

新的 dataset schema 保存源码、compact CPG relations、structured vulnerability semantics 和 fixed-vocabulary vulnerability features：

```bash
python -m vulnmechanism.cli build \
  --samples data/functions.jsonl \
  --output data/function_dataset.jsonl
```

构建支持 resume。单个 Joern/语法失败会记录到：

```text
data/function_dataset.errors.jsonl
```

而不会丢失已经完成的样本。

## Train

默认模型：

```text
/home/phy/models/Qwen2.5-Coder-7B-Instruct
```

各 variant 单独训练并保存 checkpoint，便于逐项比较：

```bash
python -m vulnmechanism.cli train \
  --variant baseline \
  --dataset data/function_dataset.jsonl \
  --output results/baseline.pt

python -m vulnmechanism.cli train \
  --variant raw_cpg \
  --dataset data/function_dataset.jsonl \
  --output results/raw_cpg.pt

python -m vulnmechanism.cli train \
  --variant semantic_concat \
  --dataset data/function_dataset.jsonl \
  --output results/semantic_concat.pt

python -m vulnmechanism.cli train \
  --variant semantic_fusion \
  --dataset data/function_dataset.jsonl \
  --output results/semantic_fusion.pt

python -m vulnmechanism.cli train \
  --variant full \
  --dataset data/function_dataset.jsonl \
  --output results/full.pt
```

主要默认参数：

```text
source_max_length = 1536
context_max_length = 384
LoRA r = 16
LoRA alpha = 32
LoRA dropout = 0.05
fusion_dim = 256
fusion_heads = 8
feature_loss_weight = 0.2
```

`baseline`、`raw_cpg`、`semantic_concat` 使用 masked mean pooling。`semantic_fusion` 和 `full` 使用 source-to-semantics cross-attention 后在源码 token 上做 masked mean pooling。

## Evaluate

```bash
python -m vulnmechanism.cli eval \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/full.pt \
  --split test
```

分类指标：

```text
Accuracy
Precision
Recall
F1
MCC
AUC
```

`full` 额外报告 vulnerability-feature prediction 的 micro Precision / Recall / F1，用于检查 auxiliary supervision 是否实际学到对应程序特征。

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

`data/`、`results/` 和 `*.pt` 不进入版本控制。
