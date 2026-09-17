# CPG-Guided Vulnerability-Semantic Learning

本仓库研究 **C/C++ 函数级漏洞二分类**。正式实验只使用三套数据：

- **PrimeVul**：官方 chronological train/valid/test，主 benchmark；
- **CleanVul score=4**：fixing-commit grouped temporal train/valid/test；
- **SVEN C/C++**：external test only。

模型输入始终是单个函数。fix commit、CVE 描述、仓库上下文和 pair 身份只用于 benchmark 构建与审计，不作为模型输入。

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

主 benchmark 的规则：

- PrimeVul 保留官方 split，不重新随机划分；每个 split 保留全部清理后的 vulnerable，并确定性选择等量 benign；
- CleanVul 只使用 score=4 的 C/C++ pair，同一 fixing commit 不跨 split，按 commit 时间形成 train/valid/test；
- SVEN 只保留 C/C++ pair，全部为 `external_test`；
- exact source、lexically normalized source、已知 counterpart、以及 token 5-gram Jaccard >= 0.90 的 near duplicate 按 test-first 优先级清理；
- 单纯 same repo+commit 但不同函数在主 benchmark 中只统计，不自动删除；
- CleanVul/SVEN 始终以完整 vulnerable/fixed pair 为原子单位。

准备过程会输出 `statistics.json` 和 `deletion_statistics.json`，用于回查每类删除原因和最终规模。

## 2. Build CPG dataset

正式 build 只接受 benchmark manifest 的严格 schema。每条记录必须显式包含：

```text
sample_key
dataset
function
label
language
split
```

CleanVul/SVEN 还必须包含 `pair_id`。不再支持旧字段别名、隐式 label、默认 language 或 hash 自动分 split。

```bash
python -m vulnmechanism.cli build \
  --samples data/benchmark/manifest.jsonl \
  --output data/function_dataset.jsonl \
  --joern-dir /home/phy/joern \
  --java-home /home/phy/jdk21 \
  --timeout 300 \
  --batch-size 8
```

输出：

```text
data/function_dataset.jsonl
  仅包含 build 成功样本；dataset schema = 8

data/function_dataset.errors.jsonl
  当前仍失败的样本及失败阶段；逐条 flush/fsync，中断后仍可审计

data/function_dataset.audit.json
  build 成功率、按 dataset/split/label 的覆盖率、失败类型、CPG 结构质量、pattern 覆盖
```

### C/C++ resolution

PrimeVul 中无法从上游元数据确定 C 或 C++ 的函数保留 `language=c_cpp`。build 优先使用明确扩展名；否则分别用 C/C++ parser 解析，并把确定性的解析选择记录为 `resolved_language`。这只是构建 CPG 所需的 parser 选择，不被宣称为数据集的真实语言标签。Tree-sitter 的函数名只作为诊断提示，解析失败或名称不一致不会阻止 Joern。

## 3. Joern 成功与实验公平性

**正式 baseline 和所有 CPG/semantic 方法使用完全相同的 build-success cohort。**

原因是 CPG 方法无法处理 Joern build 失败样本。如果 source-only baseline 保留这些样本，而 proposed method 删除它们，两者的 test cohort 不一致，指标不能直接比较。

build 后的数据 view 按以下规则形成：

- PrimeVul：每个官方 split 保留全部 build-success vulnerable，再按固定 hash 选择等量 build-success benign；
- CleanVul：任一侧 build 失败则整对排除；
- SVEN：任一侧 build 失败则整对排除。

CLI 在每次 train/eval 前输出 `benchmark_dataset_view=...`，该行就是本次实验实际使用的 cohort，应与实验结果一起保存。

可以额外报告一个 **full-source baseline** 作为覆盖率参考，但不能用它与 CPG 方法计算主结果增益。

## 4. CPG quality boundary

Joern build 成功只意味着：

1. 能定位到目标函数；
2. 通过 Joern 文件内容、物理行列范围和原始函数体核验目标 METHOD，排除 external stub、`<global>`、嵌套函数和歧义匹配；
3. 从该 METHOD 的 AST 子树提取 AST/CFG/CDG/REACHING_DEF，关系端点全部限制在目标内；
4. AST/CFG 非空、函数体存在、没有 UNKNOWN AST 节点，非容器节点的截断代码必须能核验恢复。

语义 JSON 保留节点 ID 供审计，但模型文本会移除运行时 ID，并按固定顺序组织同类条目；改变批次不应改变模型的语义输入。

每批只启动一次 parse 和一次 Neo4j CSV export（Joern 内置流式格式，不需要 Neo4j 服务），默认 8 条，最多 32 条；批次不跨 dataset/split。没有超时后换解析器、伪造声明或模糊名称兜底。批失败明确记录，不自动重试。容器 BLOCK/CONTROL_STRUCTURE 的 1000 字符截断单独计数，不拿容器全文重复提取内存操作。CDG/DDG 可合法为空，短函数和零语义项不直接判失败。schema 8 不复用旧 schema 7 的成功记录。

100 条有原始完整文件的固定样本可通过 `python -m vulnmechanism.audit_cpg --full-file` 重跑上下文对照；它要求原始函数在指定文件/行处精确出现，且只导出目标子图。对照输入与输出位于 `results/cpg_debug/`，`python -m vulnmechanism.audit_cpg` 汇总既有旧/新结果。此实验没有把完整文件模式自动应用到正式 manifest。

这 **不等于** 静态规则恢复出的 `POTENTIAL_PATTERN` 就是真实漏洞根因。

每条成功记录保存 `cpg_quality`。`function_dataset.audit.json` 会报告：

- AST / CFG / CDG / DDG edge coverage；
- build failure stage/type；
- 没有 CDG/DDG 的成功样本数量；
- 没有任何 `POTENTIAL_PATTERN` 的成功样本数量；
- pattern 数量分布。

当前模型使用的关系表示不编码控制分支 true/false polarity，因此 `UPPER_BOUND_RELATED_CONDITION`、`NONNULL_RELATED_CONDITION` 只表示“存在相应形式的控制条件”，不能表述为已证明安全的 guard。

因此，CPG-derived semantics 是 **静态候选机制**，而不是 ground-truth vulnerability mechanism。正式方法实验前应基于 CleanVul/SVEN pair 和人工抽样进一步验证 semantic fidelity。

## 5. Model variants

所有 variant 使用相同的 Qwen2.5-Coder-7B-Instruct、LoRA 配置、source token budget 和 build-success cohort。

| Variant | Input / module |
| --- | --- |
| `baseline` | source only |
| `raw_cpg` | source + compact AST/CFG/CDG/DDG relations |
| `semantic_concat` | source + CPG-derived vulnerability semantics |
| `semantic_fusion` | separate source/semantics + cross-attention |
| `full` | semantic fusion + feature supervision |

Sequence variants 使用 masked mean pooling。当前模型默认参数：

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

## 6. Train

PrimeVul baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
  --variant baseline \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --output results/primevul_baseline.pt \
  --device cuda
```

CleanVul baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
  --variant baseline \
  --source-dataset cleanvul \
  --dataset data/function_dataset.jsonl \
  --output results/cleanvul_baseline.pt \
  --device cuda
```

训练只读取 `train` 和 `valid`。每个 epoch 在 validation 上从 0.05 到 0.95 选择阈值，优先最大化 MCC，其次 F1、Accuracy，再优先接近 0.5。最佳 epoch 的阈值写入 checkpoint；test labels 永不用于选阈值。

## 7. Evaluate

PrimeVul test：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli eval \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/primevul_baseline.pt \
  --split test \
  --device cuda
```

SVEN external test：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli eval \
  --source-dataset sven \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/primevul_baseline.pt \
  --split external_test \
  --device cuda
```

SVEN不能用于训练、epoch selection 或 threshold selection。

分类指标：Accuracy / Precision / Recall / F1 / MCC / AUC。

## 8. Repository layout

```text
vulnmechanism/
  prepare_benchmark.py  fixed benchmark construction and leakage audit
  benchmark_view.py     build-success cohort selection
  syntax.py             C/C++ parsing and ambiguous-language resolution
  cpg.py                Joern AST/CFG/CDG/DDG extraction
  semantics.py          CPG-derived candidate vulnerability semantics
  dataset.py            strict schema, CPG build/resume, quality audit
  model.py              LoRA classifiers and evaluation
  cli.py                build / train / eval
```

## 9. Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same suite on every push to `main`.
