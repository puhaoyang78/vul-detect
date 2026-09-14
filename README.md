# CPG-Guided Vulnerability-Semantic Learning

本仓库研究函数级 C/C++ 漏洞二分类：

> **利用静态程序分析从 CPG 中提取与高风险内存操作直接相关的程序语义，并用这些语义引导 Code LLM 学习函数的原始 0/1 漏洞标签。**

测试阶段的预测单位始终是单个函数。不使用 vulnerable/fixed pair、fix commit、CVE 描述、caller/callee 或仓库上下文作为分类输入。

## Method

```text
single C/C++ function
        │
        ├────────────── source code ──────────────────────────┐
        │                                                      │
        └→ Joern AST / CFG / CDG / DDG                         │
             ↓                                                 │
   vulnerability-semantic extraction                          │
             ↓                                                 │
   high-risk operation + directly related semantics            │
             │                                                 │
             └──── optional cross-attention fusion ────────────┤
                                                               ↓
                                                       LoRA Code LLM
                                                               ↓
                                                vulnerability classification
```

## Vulnerability-semantic extraction

Joern 为每个函数独立导出 AST / CFG / CDG / DDG。实现首先识别 memory write、array access、pointer dereference、allocation/deallocation 等安全相关操作，再从 CPG 中提取与这些操作直接相关的信息。

每个 memory write、array access 和 pointer dereference 都保存在 `MEMORY_OPERATION` 中。若静态关系进一步表现出高风险特征，再额外生成 `POTENTIAL_PATTERN`。当前高风险模式包括：

- unbounded memory write；
- variable write extent without an upper-bound-form controlling condition；
- array index without an upper-bound-form controlling condition；
- pointer dereference without a non-null-form controlling condition；
- write/index exceeding a known capacity；
- unchecked size arithmetic reaching a memory operation；
- a free operation that can reach a later use or later free of the same object。

每条与高风险操作相关的语义尽量附带：

```text
operation
object / pointer
extent / index
known capacity
upper-bound-form or non-null-form controlling condition
data sources reachable through DDG (up to depth 3)
```

例如：

```text
WRITE_EXTENT_WITHOUT_UPPER_BOUND_CONDITION
operation=memcpy(dst, src, len)
object=dst
extent=len
capacity=64
upper_bound_condition=none
data_from=len
```

这里的 `UPPER_BOUND_RELATED_CONDITION` 和 `NONNULL_RELATED_CONDITION` 只表示 **控制该操作的条件文本具有相应形式**。当前 Joern DOT 导出在本项目的数据结构中没有保留 true/false 分支方向，因此实现不会把这些条件表述为“已证明安全的 guard”。所有 `POTENTIAL_PATTERN` 都只是静态分析得到的候选模式，不是 ground-truth vulnerability label。

当前上界条件识别是保守的：`len < cap`、`len <= cap`、`cap > len`、`cap >= len` 这类形式可被识别为与 `len` 上界相关；`len > 0`、`len != 0`、`len > cap` 不会被误认为上界条件。非空条件同样区分 `p != NULL` 与 `p == NULL`/`!p`。

### Capacity information

已知容量来自两类信息：

- local fixed-size arrays，例如 `char buf[64]`；
- 只有一个明确分配大小的 `malloc/calloc/realloc/new[]` 对象。

如果同一个对象存在多个不同的动态分配大小，实现不会选择其中一个作为容量，避免使用过期或不确定的 allocation extent。

### Compact semantic rendering

`POTENTIAL_PATTERN` 在 semantic text 中最先输出，随后是 `MEMORY_OPERATION`。两类信息内部都按 kind 轮转选择，避免同一种操作或模式占满 384-token context。较低层的 memory object、dependence、constraint 和 lifetime facts 排在它们之后。

因此 `semantic_concat` 的固定 context budget 优先保留高风险模式及其直接相关的信息，而不是先被大量普通 memory-operation facts 占满。

## Model variants

所有 variant 使用相同的 Qwen2.5-Coder-7B、LoRA 配置、源码 token budget、数据划分和二分类指标。

| Variant | Input / module |
| --- | --- |
| `baseline` | source only |
| `raw_cpg` | source + compact AST/CFG/CDG/DDG relations |
| `semantic_concat` | source + vulnerability semantics |
| `semantic_fusion` | separate source/semantics + cross-attention |
| `full` | semantic fusion + vulnerability-feature supervision |

`raw_cpg` 与 `semantic_concat` 使用相同的 context token budget。

### Sequence representation

`baseline`、`raw_cpg` 和 `semantic_concat` 都在输入末尾追加 EOS，并使用 **最后一个有效 token 的 hidden state** 完成二分类。对于 causal Code LLM，这个位置可以访问前面的完整 source/context；不再对大量 source tokens 与少量 semantic tokens 做全序列平均。

`semantic_fusion` 和 `full` 保持独立的 source/semantic encoding 与 source-to-semantics cross-attention。

## Semantic groups

结构化语义仍保存为五组，便于诊断：

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

使用 `--exclude-groups` 可以删除指定输出组。需要注意：`POTENTIAL_PATTERN` 本身由 CPG 的数据依赖、控制依赖、容量和生命周期关系推导，因此删除 lower-level group 不会反向删除已经形成的 pattern；如果要删除 pattern，必须显式排除 `pattern`。

## Build dataset

输入 `data/functions.jsonl` 每行是一个函数样本：

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

构建：

```bash
python -m vulnmechanism.cli build \
  --samples data/functions.jsonl \
  --output data/function_dataset.jsonl
```

当前 dataset schema 为 version 6。旧 semantic records 不会被复用，因此更新代码后重新执行 `build` 会重新生成语义数据。

构建支持 resume。单个 Joern/语法失败会写入：

```text
data/function_dataset.errors.jsonl
```

## Train

默认模型：

```text
/home/phy/models/Qwen2.5-Coder-7B-Instruct
```

示例：

```bash
python -m vulnmechanism.cli train \
  --variant baseline \
  --dataset data/function_dataset.jsonl \
  --output results/baseline.pt

python -m vulnmechanism.cli train \
  --variant semantic_concat \
  --dataset data/function_dataset.jsonl \
  --output results/semantic_concat.pt
```

默认参数：

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

模型表示方式已经变化，checkpoint version 为 4；旧 checkpoint 不能直接按新实现评估，需要重新训练。

## Evaluate

```bash
python -m vulnmechanism.cli eval \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/semantic_concat.pt \
  --split test
```

分类指标：Accuracy / Precision / Recall / F1 / MCC / AUC。

## Repository layout

```text
vulnmechanism/
  syntax.py       standalone C/C++ function parsing
  process.py      external-process execution
  cpg.py          standalone Joern AST/CFG/CDG/DDG extraction
  semantics.py    vulnerability-semantic extraction
  dataset.py      dataset construction and resume
  model.py        LoRA classifiers, semantic fusion and feature supervision
  cli.py          build / train / eval commands

tests/
  test_vulnmechanism.py
```

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
