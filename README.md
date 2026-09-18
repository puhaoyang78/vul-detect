# CPG-Guided Vulnerability-Mechanism Learning

本仓库研究 **C/C++ 函数级漏洞二分类**。核心目标不是把整张 CPG 直接作为“漏洞知识”交给模型，而是从 Joern CPG 中恢复与漏洞形成过程相关的程序关系，再与源码表示融合，使 Code LLM 更关注漏洞机理而不是词法、命名或单纯的安全敏感操作等虚假相关特征。

正式实验使用：

- **PrimeVul**：官方 chronological train/valid/test，主 benchmark；
- **CleanVul score=4**：fixing-commit grouped temporal train/valid/test；
- **SVEN C/C++**：external test only。

fix commit、CVE 描述、仓库上下文和 pair 身份只用于 benchmark 构建与审计，不作为模型输入。

## 1. Formal benchmark

```bash
python -m vulnmechanism.prepare_benchmark \
  --source-dir /home/PublicData/PHY-data/vul_detect/data \
  --output-dir data/benchmark \
  --final-score 4
```

正式清单：

```text
data/benchmark/manifest.jsonl
data/benchmark/manifest_sven_commit_disjoint.jsonl
```

主 benchmark 规则：PrimeVul 保留官方 split；CleanVul/SVEN 保持 pair 原子性；exact/normalized/counterpart/高置信 near duplicate 按 test-first 优先级清理；SVEN 永远只作 external test。

## 2. CPG build

```bash
python -m vulnmechanism.cli build \
  --samples data/benchmark/manifest.jsonl \
  --output data/function_dataset.jsonl \
  --joern-dir /home/phy/joern \
  --java-home /home/phy/jdk21 \
  --timeout 300 \
  --batch-size 8
```

当前数据集 schema 为 **9**。旧 schema 不会被 resume。改变机理提取规则后的诊断实验应删除旧输出并完整重建，不能复用旧记录。

输出：

```text
data/function_dataset.jsonl
  build 成功样本；包含源码、CPG关系、机理上下文和质量信息

data/function_dataset.errors.jsonl
  当前失败样本及失败阶段

data/function_dataset.audit.json
  构图覆盖、失败类型、CPG质量和机理候选分布
```

### CPG quality boundary

函数以独立 `.c/.cpp` 文件送入 Joern。目标 METHOD 以文件归属为主、名称提示只用于确定性消歧。`<global>` 不作为真实函数回退。

硬失败条件：

- 无法唯一确定目标 METHOD；
- AST 或 CFG 缺失；
- 目标 AST 中出现 `UNKNOWN`；
- 明显未展开的宏函数签名；
- 非容器节点发生不可核验的代码截断。

`missing_body` / `body_content_missing` / source-token mismatch 仅记录为 warning，不再因为 Joern 某个 CODE 属性不完整而删除整张结构上可用的图。CDG/DDG 可以为空，但会影响后续机理证据强度。

当前 Joern 版本的 `joern-parse` 默认 CPG 已包含 `dataflowOss` overlay；导出的 `REACHING_DEF` 在仓库中映射为 DDG，因此不重复执行第二次 OSS dataflow pass。

批次级 Joern/export 失败使用确定性二分隔离，避免一个坏样本连带删除同批其他样本。

## 3. 从 CPG 到漏洞机理

`semantics.py` 内部保存三层信息：

```text
SECURITY_OPERATION
MECHANISM_RELATION
MECHANISM_CANDIDATE
```

模型上下文渲染：

```text
MECHANISM_CANDIDATE
+ MECHANISM_RELATION
```

`SECURITY_OPERATION` 仅用于审计，不进入 `mechanism_concat` 或 `mechanism_fusion`，避免模型把“pointer/array/memory 操作多”当成漏洞捷径。

### 3.1 Security operation

只记录与内存安全相关的原始操作事实，例如：

- memory read/write；
- array access；
- pointer dereference；
- allocation/deallocation。

这些事实 **不是漏洞标签，也不是模型输入本身**。

### 3.2 Mechanism relation

CPG 用来验证与漏洞形成有关的高层程序关系，例如：

- 参数或其派生值经 expression-local DDG 到达 array index 或 write extent；
- 内存对象存在静态/动态 capacity；
- arithmetic expression 经 DDG 到达 allocation/memory sink；
- arithmetic expression 位于控制条件中，并经 CDG 控制后续 memory sink；
- `malloc/calloc/realloc/kmalloc/kzalloc/vmalloc` 等可返回空的分配结果经 DDG 到达后续解引用；
- 显式 `p = NULL/nullptr/0` 经 DDG 到达后续解引用；
- free 经 CFG 到达后续 use/free，且路径上未观察到同名指针重定义；
- 控制条件通过直接 CDG 或 AST 祖先继承与具体操作关联；
- 简单局部/参数类型可作为关系属性，例如 `int`、`size_t`。

DDG 从具体 index / extent / allocation-size 表达式开始追踪，而不是从整个 call/sink 节点向上追，避免不同实参的数据流串线。`sizeof(...)` 不作为动态运行时来源，类型声明也不作为 arithmetic expression。

**普通 pointer parameter 经 DDG 到达 dereference 只表示数据关系，不足以构成 NULL_DEREFERENCE_FLOW。** 参数契约可能保证 non-null，因此必须有更明确的 nullable provenance。

控制条件不编码 true/false branch polarity，因此只能表述为 `present / not_observed / unknown_no_cdg`。这些状态是关系/候选的注解，**不能单独作为“安全”或“漏洞”的判据**。

### 3.3 Mechanism candidate

当前只构造四类核心候选：

```text
BOUNDS_FLOW
NULL_DEREFERENCE_FLOW
USE_AFTER_FREE_FLOW / DOUBLE_FREE_FLOW
SIZE_ARITHMETIC_FLOW
```

这里的 `MECHANISM_CANDIDATE` 表示 **CPG 支撑的漏洞形成机理候选**，不是当前函数已被静态证明存在漏洞，更不是 ground-truth root cause。

候选必须由结构关系本身成立，不能因为“没有观察到检查”就自动生成。具体规则包括：

- `BOUNDS_FLOW`：要求对象与独立 capacity 的关系；只有参数/索引流到 memory sink、但没有对象 capacity 时，仅保留 `INDEX_FLOW_TO_MEMORY_SINK` 或 `EXTENT_FLOW_TO_MEMORY_SINK` relation；
- `NULL_DEREFERENCE_FLOW`：要求明确 nullable provenance 到达 dereference；`present / not_observed / unknown_no_cdg` 仅描述相关控制条件状态；
- `SIZE_ARITHMETIC_FLOW`：要求 arithmetic expression 与 allocation/memory/control sink 的结构关系；约束状态只作为注解；
- `USE_AFTER_FREE_FLOW / DOUBLE_FREE_FLOW`：要求 CFG 上真实存在 free→use/free 路径，并且路径上未观察到同名对象重定义。

因此，**absence of a check alone never constitutes a vulnerability mechanism**。

NULL 候选目前要求明确的 nullable provenance，例如可返回空的分配结果或显式空值赋值；普通函数参数不作为 nullable 证据。链式 `tree->cdr->car->x` 使用完整 base expression，不把 `cdr`、`car` 这类字段名误当成独立指针变量。

对于 `snprintf/vsnprintf/strlcpy/strlcat`，size 参数本身是目标写入边界；只有存在独立 destination capacity 时才形成 bounds candidate，避免把安全 API 的 size 参数误当成过量写入证据。

同一对象/来源/机理的重复访问会聚合为一条候选并记录 `occurrences`，不再逐 CPG 节点生成数百条“缺少检查”pattern。

模型文本不复制原始 `examples=...` 源码片段，避免 semantic branch 通过重复源码而非程序关系获得增益。

## 4. Pair-level mechanism audit

CleanVul/SVEN 用真实 vulnerable/fixed pair 诊断机理 fidelity：

```bash
python -m vulnmechanism.audit_semantics \
  --dataset data/function_dataset.jsonl \
  --source-dataset cleanvul \
  --output results/cleanvul_mechanism_audit.json
```

以及：

```bash
python -m vulnmechanism.audit_semantics \
  --dataset data/function_dataset.jsonl \
  --source-dataset sven \
  --output results/sven_mechanism_audit.json
```

报告区分 candidate addition/removal、state change、detail change，以及真正送入 Code LLM 的 `mechanism_context` 是否变化。修复后发生变化可以支持机理相关性，但这些指标只用于诊断，不能当作因果 ground truth，也不能反向作为规则调参目标。

## 5. Fair comparison cohort

正式 baseline 和所有 CPG/mechanism 方法必须使用相同 build-success cohort：

- PrimeVul：每个官方 split 在 build-success 样本中取两类数量的最小值，并对超出的类别做确定性抽样，从而保持 split 内 1:1 平衡；
- CleanVul/SVEN：任一侧 build 失败则整对排除。

可以额外报告 full-source baseline 作为覆盖率参考，但不能与 CPG 方法直接计算主结果增益。

## 6. Model variants

所有 variant 使用相同的 Qwen2.5-Coder-7B-Instruct、LoRA 配置和正式 cohort。

| Variant | Input / module |
| --- | --- |
| `baseline` | source only |
| `raw_cpg` | source + compact AST/CFG/CDG/DDG relations |
| `mechanism_concat` | source + CPG-derived mechanism relation/candidate context |
| `mechanism_fusion` | source/mechanism 分开编码 + cross-attention |

旧的 generic `feature supervision` 已删除。原因是当前没有独立的 ground-truth mechanism labels，不能用规则自身产生的普通程序特征再反向监督 Code LLM。

默认训练参数：

```text
source_max_length = 1536
context_max_length = 384
LoRA r = 16
LoRA alpha = 32
LoRA dropout = 0.05
learning_rate = 2e-4
epochs = 3
batch_size = 1
gradient_accumulation = 8
```

## 7. Train / evaluate

PrimeVul baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
  --variant baseline \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --output results/primevul_baseline.pt \
  --device cuda
```

机理融合：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
  --variant mechanism_fusion \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --output results/primevul_mechanism_fusion.pt \
  --device cuda
```

测试：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli eval \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/primevul_mechanism_fusion.pt \
  --split test \
  --device cuda
```

SVEN只能用于 `external_test`，不能用于训练、epoch selection 或 threshold selection。

## 8. Repository layout

```text
vulnmechanism/
  prepare_benchmark.py  benchmark construction and leakage audit
  benchmark_view.py     build-success cohort selection
  syntax.py             C/C++ parser/language resolution
  cpg.py                Joern AST/CFG/CDG/DDG extraction
  semantics.py          CPG-derived vulnerability-mechanism extraction
  audit_semantics.py    vulnerable/fixed mechanism fidelity audit
  dataset.py            schema 9 build/resume/quality audit
  model.py              LoRA classifiers and mechanism fusion
  cli.py                build / train / eval
```

## 9. Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions运行同一套轻量测试。正式 Joern 构图仍需在本地服务器验证。
