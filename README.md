# Function-Level Vulnerability Mechanism Learning

本仓库当前主实验方向是：

> **Patch-Grounded, CPG-Constrained Vulnerability Mechanism Learning**

目标不是继续扩展仓库级 Z3 verifier，也不是用 LLM 补全自定义函数语义，而是研究：

> 仅使用函数级源码和函数内 CPG，能否让小型 Code LLM 学到真正决定漏洞的程序关系，而不是函数名、API、变量名等 shortcut。

测试阶段只输入单个函数及其函数内 CPG，不依赖 repository clone、caller/callee、fix commit、CVE 描述或 ground truth。

## New experiment

训练阶段使用 vulnerable/fixed 函数对：

```text
vulnerable function ──┐
                      ├─> AST / CFG / CDG / DDG
fixed function ───────┘
                              ↓
                    aligned graph relation delta
                              ↓
                    vulnerability mechanism
                              ↓
          mechanism-supervised Qwen bottleneck
                              ↓
                       VUL / BENIGN
```

当前 mechanism 只保留可以回溯到函数源码或 CPG 的结构化信息：

- guard / bounds
- allocation
- memory access
- size arithmetic
- data dependency
- control dependency
- pointer/index relation

模型不允许直接从 patch 或 CVE 信息完成测试时判断。

### Pair input

`build` 接受 JSONL。每条记录至少包含一个 ID、vulnerable function 和 fixed function：

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

也支持 `vulnerable` / `fixed` 字段；MegaVul 风格的 `func_before` / `func` 也可直接映射。

### Build mechanism dataset

```bash
python -m semantic_demo.mechanism_cli build \
  --pairs data/pairs.jsonl \
  --output data/mechanism_dataset.jsonl
```

该阶段对每个函数片段单独运行 Joern，导出 AST、CFG、CDG、DDG，并比较 vulnerable/fixed 两侧的安全相关关系。不会克隆或恢复完整仓库。

输出同时保存：

- raw source
- canonicalized source
- identifier-renamed source
- compact security-relevant graph relations
- changed CPG relations
- mechanism components
- vulnerable/fixed label

### Train

默认使用本地 Qwen2.5-Coder-7B-Instruct：

```bash
python -m semantic_demo.mechanism_cli train \
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

分类头只能消费 mechanism bottleneck，不能直接绕过它读取 Qwen hidden state。

### Evaluate

```bash
python -m semantic_demo.mechanism_cli eval \
  --dataset data/mechanism_dataset.jsonl \
  --checkpoint results/mechanism_model.pt \
  --split test
```

当前报告：

- Accuracy
- Precision / Recall / F1
- vulnerable/fixed Pair Accuracy
- mechanism component accuracy
- identifier-renaming prediction agreement
- identifier-renaming probability shift

## Existing repository-context verifier

旧的 repository-context + custom semantics + Z3 selective verifier 暂时保留为历史基线，没有继续扩展。

入口仍然是：

```bash
python -m semantic_demo.cli preflight
python -m semantic_demo.cli normalize
python -m semantic_demo.cli run
```

其原有数据、结果和测试未删除。

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
