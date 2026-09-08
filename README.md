# Function-Level Vulnerability Mechanism Learning

本仓库只保留当前研究主线：

> **Patch-Grounded, CPG-Constrained Vulnerability Mechanism Learning**

目标是研究：仅使用函数级源码和函数内 CPG，能否让小型 Code LLM 学到真正决定漏洞的程序关系，而不是函数名、API、变量名等 shortcut。

测试阶段只输入单个函数及其函数内 CPG，不依赖完整仓库、caller/callee、fix commit、CVE 描述或 ground truth。

## Pipeline

```text
vulnerable / fixed function pairs
            ↓
standalone AST / CFG / CDG / DDG
            ↓
aligned security-relevant graph delta
            ↓
vulnerability mechanism supervision
            ↓
Qwen mechanism bottleneck
            ↓
VUL / BENIGN
```

训练阶段允许使用 vulnerable/fixed 函数对生成 supervision；测试阶段只使用目标函数本身。

## Repository layout

```text
vulnmechanism/
  syntax.py       standalone C/C++ function parsing and identifier handling
  process.py      safe external-process execution
  cpg.py          standalone Joern AST/CFG/CDG/DDG extraction
  mechanism.py    graph-grounded mechanism construction and dataset builder
  model.py        Qwen baseline and mechanism-bottleneck classifier
  cli.py          build / train / eval commands

tests/
  test_vulnmechanism.py
```

旧的 repository-context、custom-function semantic normalization、Z3 verifier、LineVul experiment、旧结果与缓存均已移除。

## Pair input

JSONL 每条记录至少包含 vulnerable/fixed 函数：

```json
{
  "sample_key": "example-1",
  "function_name": "foo",
  "language": "c",
  "func_before": "... vulnerable function ...",
  "func_after": "... fixed function ...",
  "split": "train"
}
```

也支持 `vulnerable` / `fixed` 字段，以及 MegaVul 风格的 `func_before` / `func`。

## Build mechanism dataset

```bash
python -m vulnmechanism.cli build \
  --pairs data/pairs.jsonl \
  --output data/mechanism_dataset.jsonl
```

每个函数片段单独运行 Joern，导出 AST、CFG、CDG、DDG。不会克隆或恢复完整仓库。

输出包含：

- raw source
- canonicalized source
- identifier-renamed source
- compact security-relevant CPG relations
- vulnerable/fixed graph-relation delta
- mechanism components
- VUL/BENIGN label

当前 mechanism components：

```text
guard
bounds
allocation
memory_access
size_arithmetic
data_dependency
control_dependency
pointer_index
other
```

## Train

默认使用本地 Qwen2.5-Coder-7B-Instruct：

```bash
python -m vulnmechanism.cli train \
  --dataset data/mechanism_dataset.jsonl \
  --model /home/phy/models/Qwen2.5-Coder-7B-Instruct \
  --output results/mechanism_model.pt
```

Baseline：

```text
raw function -> frozen Qwen representation -> VUL/BENIGN
```

Proposed：

```text
canonical function + compact CPG relations
                ↓
          frozen Qwen representation
                ↓
       mechanism bottleneck
                ↓
            VUL/BENIGN
```

最终分类头只消费 mechanism bottleneck，不直接读取 Qwen hidden state。

## Evaluate

```bash
python -m vulnmechanism.cli eval \
  --dataset data/mechanism_dataset.jsonl \
  --checkpoint results/mechanism_model.pt \
  --split test
```

报告：

- Accuracy
- Precision / Recall / F1
- vulnerable/fixed Pair Accuracy
- mechanism component accuracy
- identifier-renaming prediction agreement
- identifier-renaming probability shift

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

`data/` 和 `results/` 是运行时生成目录，不进入版本控制。
