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

## 10. Train-only incremental-information diagnosis

先验证 relation 内容的增量信息，不训练 residual 网络。入口：

```bash
# CPU preparation only; preserves the formal 5,670-row training cohort.
python -m vulnmechanism.cli diagnose --stage prepare

# User-run full experiment: 3 outer folds × (3 inner fits + 1 outer fit) = 12 baselines.
CUDA_VISIBLE_DEVICES=0 python -u -m vulnmechanism.cli diagnose --stage all --device cuda
```

默认每个 baseline 从本地 Qwen 基座重新训练 LoRA 和 classifier，固定 **3 epochs**。
不加载正式 baseline checkpoint，不用任何留出数据选 epoch。`--stage oof` 仅训练并保存
OOF 分数；`--stage probe` 仅运行廉价 probe；`--stage all` 顺序执行二者。
已完整保存的预测任务会校验并跳过；有 checkpoint、尚无预测时只补预测。中断在训练中的
单个模型会重新训练，**不宣称支持模型内优化器续训**。运行配置和输入文件元数据变化时
拒绝复用该输出目录。改变参数请显式指定另一个 `--output`，各阶段使用相同参数。

划分与信息边界：

- 只在正式 build-success、split 内平衡后的 PrimeVul **train** 中划分；官方 valid/test
  不参与模型拟合、词表、缩放、阈值选择或评价。加载正式 cohort 时仅复用既有选择规则。
- 从现有 `--manifest` 回连 sample_key，核对源码、标签与 split。按完整 commit、
  counterpart/pair、规范化源码的连通分量分组。完整 commit SHA 可跨 fork 关联，也允许
  已知 SHA 但缺少 repo 的记录；无法确定分组来源时直接报错。
- 每个外层训练分区再按组留出约 1/5 为 calibration；其余 develop 内做 inner OOF。
  probe 在 develop 的 OOF logits 上拟合；另一个仅在 develop 上训练的 baseline 同时
  预测 calibration 和外层 evaluation。calibration 只选阈值，evaluation 只评价。
  不同内外层 baseline 的训练规模不同，比较的三个 probe 共享完全相同的分数来源。
- `S` 是原始 logit；`U` 是 relation 存在性、总数量、各类型数量（数量 log1p）；
  `R` 是当前所有 `MECHANISM_RELATION` 的 kind/detail/state，使用大小写敏感、保留
  运算符的 unigram/bigram TF-IDF，最多 2,000 个训练词表特征。不加入 candidate。
  不沿用 LLM 的 384-token 截断；这里诊断完整持久化关系内容，不等同于旧 concat 输入。
- 三层均使用固定 C=1 的正则化 logistic regression；词表和 scaler 仅拟合 develop。
  阈值仅在 calibration 的 0.05–0.95 网格上按 MCC/F1/Accuracy 选择。
  不在 evaluation 上调 C、词表大小或阈值。
- `R` 是现有抽取信息，**不代表已验证 branch polarity/dominance 的路径条件**。
  内容收益也可能包含命名等信号，不能直接当成真实漏洞机理学习。

Controlled shuffle 在 develop/calibration/evaluation **各自内部**，按相同类型及每类
精确数量、相同 64 字符长度区间交换 R，不改变接收方的 S/U/标签。交换排除同一分组，
不使用标签匹配；无法整体置换的桶保持不变并排除出 matched 评价。每折默认 5 次置换，
报告可交换数、实际文本变化数以及 donor 对应。固定长度区间不保证所有“长度相近”的
样本都能交换，这是明确的覆盖边界。

同时运行两种对照：`shuffle_N` 重新拟合 shuffled train、在 shuffled calibration 选阈值；
`intervention_N` 固定正常训练模型及其阈值，仅替换 evaluation 内容。主比较需看相同
matched 子集上的 `S+U+R` 与 shuffle，并结合全体样本指标；不能将不同子集直接相减。

**终端持续显示**训练 batch 进度条、loss、optimizer step、学习率、耗时和 ETA，
每个 epoch 打印结果表；预测、外层任务、probe 和 shuffle 也显示进度。
`--log-every 10` 控制持久打印训练摘要及写入 JSONL 的 optimizer-step 间隔；
终端进度条在 batch 间持续刷新，不必等待 epoch。**不生成 HTML 可视化文件。**
原有 `train` 命令也使用相同终端显示。

输出位于 `results/primevul_incremental/`：

- `run.json`、`folds.json`、`features.jsonl`：配置、准确样本分配和 probe 特征；
- `outer_XX/inner_XX/` 与 `outer_XX/outer_model/`：任务成员、checkpoint、原始 logits；
- `*.training.jsonl`：逐步 loss 与 epoch 指标，供复现，不替代终端显示；
- `probe_predictions.jsonl`：外层逐样本概率、阈值、matched 标记；
- `probe_results.json`：逐折全体/matched/relation-present 指标、相对 S 的正负类纠错与
  破坏数量、shuffle 覆盖与 donor、配对折间增量均值和标准差。

报告 AUC、MCC、Accuracy、Precision、Recall、F1、log loss、TP/FP/TN/FN。
终端展示每折结果和相对 `S+U` 的增量；不混合不同 baseline 的跨折分数计算“总 AUC”，
也不自动宣布显著性或通过。继续做 residual 的依据是内容在元信息之上有稳定样本外收益，
并优于受控置换，不能只看 F1 或个别折。

### Source-only 难例迁移小实验

```bash
python -m vulnmechanism.cli source-pilot --stage prepare
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli source-pilot --stage run --device cuda
```

读取 `source_error_review.jsonl` 中已有局部证据的训练样本。从 outer_01 的 source-only
checkpoint 出发，以同一批 128 条训练样本、同一随机种子、1 epoch、学习率 2e-5，
比较普通继续训练与难例权重 3 倍的继续训练；权重在正负类内分别归一化。
初始模型、普通训练、加权训练均使用初始模型校准集选定的同一个阈值。
终端持续显示 batch/step，并打印 check、valid 和训练难例的指标。

检查集为 128 条原始 train 函数，排除初始模型训练成员、提交/归一化源码/配对关联组，
与继续训练集隔离项目，并排除相对初始/继续训练源码字符 TF-IDF cosine >= 0.90 的样本。
初始模型仍可能见过检查项目中的其他函数，因此不是完整的项目外泛化实验。
检查集标签沿用 PrimeVul，并非独立核实的同机制标签；这是探索性函数分类迁移实验。
正式 valid 仅报告、不用于调整本实验参数；test 不参与。比较对象是同一个 OOF
checkpoint，不能将结果直接写作超过已发布的完整训练 baseline。

`results/primevul_source_pilot/experiment.json` 保存成员和参数，逐组保存 checkpoint、
预测与指标，可用相同 run 命令继续尚未完成的组；最终指标见 `results.json`。
人工难例有效只支持可学习性，自动选择的有效性仍需独立对照。

### Source-only 分类结构对照

四组均只输入相同源码，使用同一正式 cohort、LoRA、BCE、优化器和 valid MCC
选 epoch/阈值规则；不使用 CPG 文本、人工难例权重或辅助标签：

| `--variant` | Qwen 输出之后的处理 |
| --- | --- |
| `baseline` | 原始 masked mean + linear |
| `source_attention` | 4 个学习 query 的注意力汇聚 + 原尺寸 linear |
| `source_bidirectional` | 双向残差层 + masked mean + 原尺寸 linear |
| `source_bidirectional_attention` | 双向残差层 + 注意力汇聚 + 原尺寸 linear |

双向残差层为 LayerNorm → 256 维投影 → 单层 Transformer（4 heads、FFN 512、
dropout 0）→ 投影回 Qwen 维度，与原 token 表示相加。最后的投影初始化为零，
起始表示不变；它的内部参数从后续优化步开始获得梯度。Qwen 自身仍是因果注意力。
直接使用 Qwen 的上下文化 token 表示，不另加位置编码。注意力汇聚用 256 维 key，
4 个 query 分别做 masked softmax，再对其汇聚结果求平均；分类器维度不变。
query 不对应人工漏洞类别，注意力权重也不是经过验证的漏洞解释。
新模块保存在共享 checkpoint 的 `task_state` 中，旧 baseline checkpoint 仍可读取。

第一轮保持原固定学习率，先隔离结构贡献。以下命令**由用户执行全量训练**，
重新训练同设置 baseline，不把历史不同长度的结果当作对照。显式统一为 2048 source
tokens（源码之外的前缀/EOS 与现有 baseline 相同），每个优化步输出 loss，持续显示 batch
进度；每个 epoch 显示 valid 指标。已有输出会跳过，避免重复训练：

```bash
conda activate vul-detect
for variant in baseline source_attention source_bidirectional source_bidirectional_attention; do
  output="results/source_architecture/seed42/${variant}.pt"
  if [ -e "$output" ]; then
    echo "Skip existing checkpoint: $output"
    continue
  fi
  CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
    --source-dataset primevul --dataset data/function_dataset.jsonl \
    --variant "$variant" --output "$output" \
    --source-max-length 2048 --epochs 3 --learning-rate 2e-4 \
    --batch-size 1 --gradient-accumulation 8 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --weight-decay 0.01 --seed 42 --device cuda --log-every 1 || break
done
```

先比较 valid 的 AUC/MCC/Accuracy/F1 与逐 epoch 曲线；该命令不评估 test。
单 seed 结果仅用于筛选，后续再对 baseline 和候选用相同多 seed 复验。
小规模 smoke 只验证执行、梯度、保存和重载，不代表分类性能提升。
