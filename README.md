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

`baseline`、`raw_cpg` 和 `semantic_concat` 对所有有效 token 的 hidden states 使用 **masked mean pooling** 后完成二分类。该实现与此前表现稳定的正式 baseline 保持一致；last-token pooling 已通过对照实验排除，因为它会显著降低当前函数级分类性能。

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

## Formal benchmark preparation

正式 benchmark 使用 `vulnmechanism.prepare_benchmark`。原始数据保持只读，不执行训练、Joern 或模型结构修改。现有 `prepare_primevul` 是早期小规模 smoke 数据入口，不用于正式 benchmark。

```bash
conda run --no-capture-output -n vul-detect \
  python -m vulnmechanism.prepare_benchmark \
  --source-dir /home/PublicData/PHY-data/vul_detect/data \
  --output-dir data/benchmark
```

首次运行同时打印并保存 CleanVul `score=4` 和 `score>=3` 的原始统计、清理损失及最终候选数量。审阅后用 `--final-score 4` 或 `--final-score 3` 生成正式清单；不通过随机种子重新抽样。

本次正式门槛选择 **score=4**：清理后保留 2,022 对；score>=3 保留 2,080 对，仅增加 58 对。复现正式清单时，在上面的命令追加 `--final-score 4`。

- PrimeVul：只读取官方非 paired train/valid/test 作为分类样本，保持每条记录的官方 split。paired 文件仅提供 counterpart 关系，不添加样本。扩展名明确的 C/C++ 片段不会因宏、上下文缺失或解析错误被删除；缺失文件名时先查 `file_info.json`，仍缺失才用现有语法识别。若仍无法细分，依据 [PrimeVul 官方 C/C++ 数据定义](https://github.com/DLVulDet/PrimeVul#-overview) 保留为 `language=c_cpp`，并记录 `upstream_c_cpp_unspecified`，不因解析失败丢弃漏洞函数；此值不是已确认的 C 或 C++ 子语言标签。`.h` 的 C/C++ 细分只是解析提示，`language_evidence` 保留其歧义。清理后保留所有剩余 vulnerable，在同一 split 内按 `SHA256("benchmark-benign-v1:" + sample_key)` 排序选择等量 benign。
- CleanVul：只用显式 C/C++ extension。以 fixing commit 为组，按 UTC commit 时间划分；在完整时间戳组之间选择最接近 70% 和 85% 累积 pair 数的边界，相同时间不跨 split。缺失日期的 pair 排除，不补造日期。先确定时间边界，再清理重叠，清理后比例可能偏离 70/15/15，不重新切分。
- SVEN：使用官方仓库 `data_train_val/{train,val}/*.jsonl` 中的 C/C++ 函数对，全部归入 `external_test`。不使用 `data_eval` 的代码生成 prompt，也不参与训练、验证或阈值选择。前后 normalized source 不变的 pair 不用于二分类。
- 所有来源：保留原始代码。exact source、去注释/格式差异但保留字面量及 token 边界的 normalized source、已知 pair counterpart，以及 token 5-gram 集合 Jaccard ≥ 0.90 的 near duplicate 均参与清理。近重复检索使用确定性的精确 prefix join，不使用随机 LSH；标识符不重命名。已知修复对内部允许近似且标签不同。exact/normalized 同代码的标签冲突及其 counterpart 整组隔离。
- 去重优先级依次为 SVEN external test、PrimeVul test、CleanVul test、PrimeVul valid、CleanVul valid、PrimeVul train、CleanVul train。CleanVul/SVEN 每次保留或删除完整 pair。同 repo+commit 的不同函数在主 benchmark 中仅统计，不因 commit 相同删除。
- 部分 CleanVul URL 使用缩写 commit SHA。仅使用本地三个来源中的唯一前缀对应关系展开，保留原始值；未能展开的前缀逐项列在统计报告中，不伪造完整 SHA。
- PrimeVul 缺失的仓库 URL 若能从 Gitiles commit URL 明确恢复，则记录恢复依据；仍缺失时保持为空，不把所有未知仓库视为同一仓库。原始 Git revision 表达式保留大小写并单列报告；无法解析为 SHA 的训练/验证记录不进入 commit-disjoint 实验。
- 严格泛化实验另存 `manifest_sven_commit_disjoint.jsonl`：固定主 benchmark 的三个测试集，从训练/验证候选中删除与保留的 SVEN 测试集共享 commit SHA（包括可匹配缩写）的记录及 counterpart；该严格规则跨仓库别名和分叉生效，再使用相同清理和排序规则选择样本、补齐 benign。它可能选择与主 benchmark 不同的 benign，但不会为了平衡而下采样剩余 vulnerable。

输出位于 `data/benchmark/`：

- `score_4/`、`score_3/`：两种门槛各自的主/严格实验 JSONL manifest 和逐条排除理由。
- `statistics.json`：原始语言/标签数量、时间边界、清理前后数量、重叠统计、运行环境和 SVEN revision。
- 指定 `--final-score` 后，根目录另写所选的 `manifest.jsonl` 和 `manifest_sven_commit_disjoint.jsonl`。manifest 每行含原始函数、标签、split、pair/counterpart 关系和可回查原始文件的记录位置；这些来源信息用于准备和审计，不作为模型输入。

完整 manifest 可直接传给 `build --samples data/benchmark/manifest.jsonl`。build 保留输入 `language`，新增 `resolved_language` 供 Joern 使用：`c_cpp` 优先依据明确的源文件扩展名（`.C` 为 C++，`.h` 保持歧义），否则双解析，选择能识别目标函数且语法错误/缺失节点更少的语言，同分固定选择 C；这只是确定性的解析提示，并非语言真值判定。双解析都不能识别函数时写入 errors JSONL。`external_test` 在成功记录、失败记录、恢复构建和模型读取中保留；train 仅使用 train/valid，eval 支持 `--split external_test` 并沿用 checkpoint 阈值，不在外部测试集选阈值。

删除统计写入 `data/benchmark/deletion_statistics.json`，按数据集、split、标签/完整 pair 分组，列出输入、保留、删除数量、互斥删除原因、互斥匹配证据组合及匹配的保留分区。既有产物无需重新划分即可审计：

```bash
python -m vulnmechanism.prepare_benchmark --output-dir data/benchmark --audit-deletions
```

score=4 对账：PrimeVul train vulnerable `4862 - 106（标签冲突）- 65（冲突 counterpart）- 771（重复/counterpart）= 3920`。CleanVul 先从 3067 对排除 28 对缺日期、8 对规范化后无变化，得到 3031 对；再删除 132 对标签冲突及 877 对重复，保留 2022 对。后一步按 train/valid/test 分别删除 888/93/28 对。重复证据可同时包含 exact、normalized 和 counterpart，不能把各证据命中数相加；near 检查只在前述查重未命中时执行。主 benchmark 不因单纯同 repo+commit 删除不同函数。

复验准备逻辑：

```bash
conda run --no-capture-output -n vul-detect \
  python -m unittest discover -s tests -p 'test_prepare*.py' -v
```

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

当前 dataset schema 为 version 6。旧 semantic records 不会被复用，因此更新语义提取代码后重新执行 `build` 会重新生成语义数据。

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

### Validation threshold selection

每个 epoch 都只使用 validation split 选择二分类阈值：

```text
threshold = 0.05, 0.06, ..., 0.95
```

首先最大化 validation MCC；MCC 相同时依次比较 F1、Accuracy，并优先选择更接近 0.5 的阈值。最佳 epoch 同样首先按 validation MCC 选择。AUC 继续报告，但不再用于选择 epoch。

最佳 epoch 的 `decision_threshold` 会保存到 checkpoint。测试时 `eval` 直接读取该阈值，不使用 test labels 调整阈值。

当前 checkpoint version 为 6；version 5 及以前的 checkpoint 不会被当前实现加载，需要重新训练。dataset schema 仍为 version 6，因此已经按新版语义构建好的 `data/function_dataset.jsonl` 不需要再次 build。

## Evaluate

```bash
python -m vulnmechanism.cli eval \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/semantic_concat.pt \
  --split test
```

输出会包含保存于 checkpoint 的 `decision_threshold`。分类指标为 Accuracy / Precision / Recall / F1 / MCC / AUC。

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
