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

## 节点—边联合程序行为抽象（P0 分类消融）

正式入口仍为 `python -m vulnmechanism.cfg_experiment`。原C及旧结果保留；新变体
`behavior_nodes`、`behavior_edges`、`behavior_joint`、`behavior_masked` 分别开启节点、
边、两者，以及两者但屏蔽全部边属性的同容量对照。均加载 P0 的阶段1 LoRA，图和
分类头重新初始化，使用单次源码前向和融合 BCE；不启用区域、DDG 或额外分类分支。

节点保持原C的集合，使用八个独立家族：`node_kind`（Joern种类及控制类型）、`api`
（局部调用名称及分派方式）、`datatype`（带角色的类型、指针/数组层数及限定符）、
`literal`（种类、词法信息、值及可确定的字符串长度）、`operator`（根/内部运算符）、
`access_form`（标量/解引用/下标/字段/取地址）、`access_mode`（对象和地址计算操作数
的读取、写入、读写、仅地址计算、不求值或未知）、`operand_role`（赋值、二元、一元、
条件、接收者、实参、返回、下标、字段、转换的操作数角色）。局部遍历按节点身份去重，
不穿过函数、控制体、语句块及嵌套函数引用。类型中的符号数组长度不作为名字特征，
未知宽度不依据宿主平台填补。未表达的调用副作用、VLA求值或字符编码信息保留未知。

边保持原C的端点和方向，四个家族为 `branch`、`guard`、`transfer`、`loop_role`。
真假分支关联条件及分支AST，短路关联实际子条件；switch保留case/default。
旧缓存缺少CONDITION关系时，使用具名语法字段与唯一直接AST子节点对应；
无法唯一对应时保留未知，不固定取某个子节点序号。此恢复已用真实Joern夹具验证。
循环回边由函数入口支配关系确认，SCC仅用于不可归约标志。
条件摘要只描述当前决策边，不作为到达后续节点的已证明约束。
同端点可选转移独立编码并平均，不合并为条件合取。
显式异常只标注前端实际导出的连接，不补造throw到catch的边。

所有家族分别保留 `NOT_APPLICABLE`、`NONE`、`UNKNOWN`。未知字段不会删除节点或函数。
各家族词表仅由train拟合；语义UNKNOWN与词表外OOV在报告中分别计数。
节点每家族16维均值嵌入后投影到128维；边每家族8维，合计32维。
消息为 `original_message(h_u) + W2(ReLU(W1([h_u,h_v,e])))`，中间128维。
保留5轮GRU、求和聚合和 `[h_final;x]` 读出。显式自环只增加属性残差，不重复自消息。
`behavior_masked` 将各边家族统一映射到固定类别，连guard也屏蔽；网络和端点状态不变。
`--disable-behavior-family` 可以屏蔽指定家族；原P0+C对照仍为 `dep_pretrain_cfg`。

`cpg.py` 保留函数内所有导出关系和边属性。补导按唯一的根AST角色路径对应节点，
再检查原CFG完全一致；不按相同文本或ID配对。补导只追加到新文件，失败立即停止，
再次执行跳过已持久化项；不会修改原图。前端未提供的类型或宏语义不会因补导而自动恢复。

从仓库目录用项目的 Python 环境直接执行，无需预设环境变量。
`python scripts/cfg.py` 使用当前 Python，只转发到正式入口，不复制训练逻辑。
启动时显示目标设备，随后以动态进度条显示CFG读取和属性编码的已处理字节、比例及预计剩余时间。
这些准备工作在CPU上完成，之后才加载GPU模型；属性校验与读取合并为一次扫描，
原图逐条转换为CFG，重复事实编码仅在函数内复用，数据校验和训练输入保持不变。
训练参数逐项继承原C保存的 `config.json`；已有结果中的属性缓存路径继续复用。
完成的变体经过原身份检查后跳过，中断的训练不会被静默覆盖或自动续训。

```bash
# 四组训练；也可在 train 后指定 nodes / edges / joint / masked 单独训练。
python scripts/cfg.py behavior train
python scripts/cfg.py behavior valid
python scripts/cfg.py behavior test

# 单家族消融和对应评价，例如关闭 guard。
python scripts/cfg.py behavior ablate guard
python scripts/cfg.py behavior valid guard
python scripts/cfg.py behavior test guard

# 全部家族消融，出错即停止循环。
for family in node_kind api datatype literal operator access_form access_mode operand_role branch guard transfer loop_role; do
  python scripts/cfg.py behavior ablate "$family" || break
done

# 原 ABC 仍可使用原脚本，也可使用统一入口。
python scripts/cfg.py abc train
python scripts/cfg.py abc valid
python scripts/cfg.py abc test

# 仅查看展开后的正式命令，不加载模型、不训练。
python scripts/cfg.py behavior train --show-command

# 已有缓存无需再准备。prepare 的输出目录必须不存在。
python scripts/cfg.py behavior prepare
python scripts/cfg.py behavior supplement
```

每轮训练显示一条青色动态进度条，验证结束显示指标表；逐步JSON仍完整写入
原 `history.jsonl`，不再逐行刷屏。输出重定向到文件时自动关闭动态条。
`valid` 比较保存的预测，`test` 沿用保存的valid阈值并输出比较/纠错文件。
旧长参数入口及原ABC脚本保留，仍可显式指定独立路径和参数。
新结果目录可用 `--output-dir`，不同属性缓存可用 `--behavior-dir`。

```bash
python scripts/cfg.py behavior train --output-dir results/my_behavior --behavior-dir data/my_behavior
python scripts/cfg.py behavior valid --output-dir results/my_behavior
python scripts/cfg.py behavior test --output-dir results/my_behavior
```


属性准备输出 `attributes.jsonl`、train词表及 `audit.json`，后者记录train/valid的
各家族至少一个已知分量的覆盖、包含未知字段的节点/边数、OOV token计数、
非赋值节点覆盖和真实源码实例。含未知分量与含已知分量可同时成立，UNKNOWN比例不是
整个家族缺失的比例；例如类型宽度未知不抹去已知的指针层数。
结构边数以去重后的原C CFG为准；边家族覆盖按边计数，可选转移不增加拓扑边数。
训练日志记录参数量；预测和纠错统计沿用原流程。未经过正式训练，不能由属性覆盖率
或代码完整性推断分类性能改善。

## 保留原C的联合抽象与数据/控制关系预训练

正式短入口为 `python scripts/cfg.py joint ...` 和 `python scripts/cfg.py control ...`，仍调用同一个训练器。
旧 `behavior_*`、原C、P0和已有结果保留作历史对照。新缓存使用角色绑定schema 2；
新模型拒绝schema 1，不会把旧节点替换方案的权重当成残差模型加载。

研究问题是函数标签对局部程序行为的监督不足。图侧保留原C定义抽象并补充局部操作、
操作数角色及控制转移；源码侧在CLM和可靠定义—使用监督上增加明确的控制关系。
两者都不直接判定操作安全，也不新增第三个分类分支。

依据：[DeepDFA论文](https://arxiv.org/abs/2212.08108)及其
[官方四类特征加载实现](https://github.com/ISU-PAAL/DeepDFA/blob/master/DDFA/sastvd/linevd/graphmogrifier.py)
支持保留定义抽象，而不是用新增属性替换它。
[PDBERT论文](https://arxiv.org/abs/2402.00657)及
[官方双线性解码器](https://github.com/ZJU-CTAG/PDBERT/blob/main/pretrain/comp/nn/struct_decoder/directed/simple_separated_struct_decoder.py)
采用语句级控制和token级数据依赖任务；本项目不是对其双向CodeBERT/MLM的等同复现。
本实现维持Qwen因果注意力、CLM及操作结束token取点。
[Joern规范](https://cpg.joern.io/#dominators)明确了支配、后支配与CDG方向。
[2025年的PLM漏洞检测研究](https://arxiv.org/abs/2507.16887)也讨论了语义预训练和复杂依赖的局限；
因此不以“首次使用控制流”作为创新，也不由辅助任务拟合直接推断漏洞检测效果。

图侧只有一个新实现：

- 保留原C四类组合词表和 `x_C`。八类新增事实按所属对象身份去重，保存局部操作、
  有序相对AST角色路径、家族和值；对象ID不输入模型。角色路径GRU与家族值嵌入经
  两层非线性映射绑定，再汇聚均值和 `log(1+事实数)`。不同对象上的相同事实保留次数。
- `x = x_C + P_node(a)`，末层无偏置、零初始化；最终读出保留 `[h_final;x_C]`。
- 边消息采用 `base * (1 + tanh(F(h_u,h_v,e)-F(h_u,h_v,e_neutral)))`。
  `F`隐藏层128维，末层无偏置、零初始化；四家族投影成32维。
  neutral为普通无条件转移、无guard、无特殊循环角色。未知分量掩蔽，部分未知不抹去
  其他已知事实；全不可用或显式关闭时调节为零。完整可选转移分别计算后平均。
- 自消息只有一次；显式自环仍接受属性调节。五轮GRU与两路logit相加保持不变。
  五轮不代表等计算：每个可选转移每轮计算实际/中性两次MLP，还增加事实路径编码。
- 初始化输出与原C相同。残差末层先学习，上游随后才有梯度；原有零初始化图分类头
  还会推迟图编码器的首次学习。浮点梯度比较使用数值精度容差，不能把求和顺序造成的
  一个double ULP差异解释为不同数学梯度。

控制标签采用**面向可终止路径的边控制依赖**：对于原CFG边 `(c,s)`，当操作 `u`
后支配 `s`，但不严格后支配 `c` 时，`u`控制依赖该边。要求唯一METHOD入口与
METHOD_RETURN出口、所有入口可达节点都能到达出口、无未知CFG节点；不能满足时
仅退出控制监督。不可达节点不删除出分类图。循环的非终止运行不是这一定义的目标。
只有恢复出完整真假分支集合或case/default集合的决策才参与监督。
同一分支描述若对应多条转移，要求各转移对查询答案一致，否则unknown。
这不是将缺少CDG边当负例；负例来自完成的后支配计算。

预测头只接受条件、操作的完整可见结束token及所询问分支描述；case通过可靠对齐的
case标签结束表示描述，不能对齐则跳过。目标操作真实所属分支、节点ID、标签、
推导原因均不进入头。头采用非线性交互，避免查询仅成为固定偏置。
同输入冲突被排除，2048-token外或位置不可靠的点跳过。每函数最多64条，先按
token位置规范排序，再固定seed采样；优先选择同条件/分支下邻近操作作正负对照，
不重复填充。采样不依赖Joern编号或漏洞标签，train/valid准备后不再重新采样。

新阶段1为 `CLM + dependency + control`，系数均为1，训练1 epoch。
两类关系各自按函数内均值、累积窗口有效函数均值归一化；CLM保留原口径。
新阶段1从原始Qwen开始，与P0共享源码、顺序、初始化、token预算和更新次数。
新增头隔离初始化随机数；阶段2只加载LoRA。F/G复用一次新预训练。

必要对照：A复用P0+原C；`joint_nodes / joint_edges / joint / joint_shuffled` 对应B/C/D/E；
`control_cfg / control_joint` 对应F/G。E按固定seed在函数内打乱**完整边属性记录**，
不拆散branch和guard，不改变拓扑、节点输入、属性分布或模型容量。

每轮分类保存同一固定最多64个train样本和完整valid的eval模式BCE，以及同一次前向
产生的source/graph/fusion分数。文件为 `epochN.{train,valid}.branches.jsonl`，BCE写入
原history。train诊断成员在训练前固定，不按模型错误选择。源码分支是联合训练后的
分支，不是独立源码baseline。裁剪前LoRA/图梯度范数与残差/调节幅度按原日志间隔记录。
checkpoint及阈值仍只由融合valid MCC选择。
固定关系评价保存逐查询logit/score、函数平均BCE、AUC/MCC、类别和独立函数覆盖；
常量参照只由train的函数平均正例率估计，平滑固定为1e-6，不在valid调参。
固定评价与在线训练均值分开记录，不跨不同目标比较总loss。

```bash
# 已准备好的目录直接复用；prepare不覆盖已有缓存。
python scripts/cfg.py joint prepare
python scripts/cfg.py control prepare

# 必要时补导原生Joern属性；不改变原CFG，支持增量重入。
python scripts/cfg.py joint supplement
# 使用补导文件仍走同一个正式准备入口。
python scripts/cfg.py prepare-behavior --reference-run-dir results/cfg_abc_seed42 --bound \
  --supplement-path data/graphs/primevul_behavior_supplement.jsonl \
  --output-dir data/cfg_joint_supplemented_seed42

# 以下正式训练本轮不自动执行。
python scripts/cfg.py control pretrain        # 一次新阶段1；从原始Qwen开始
python scripts/cfg.py joint train             # B/C/D/E，复用有效P0
python scripts/cfg.py control train           # F/G，复用同一个新LoRA

python scripts/cfg.py joint valid
python scripts/cfg.py control valid
python scripts/cfg.py control relations       # 固定阶段1的train/valid关系评价和train-only常量参照

# 方法冻结后：按已保存的valid阈值评价test并生成相对A纠错统计。
python scripts/cfg.py joint test
python scripts/cfg.py control test

# 必要的单家族消融；不自动展开组合搜索。
python scripts/cfg.py joint ablate guard
python scripts/cfg.py joint valid guard
python scripts/cfg.py joint test guard

# 查看实际继承的参数，不启动模型。
python scripts/cfg.py joint train --show-command
python scripts/cfg.py control pretrain --show-command

python scripts/cfg.py --help
```

## 结构引导的判别性源码聚合

本轮检验“结构在源码形成函数表示之前参与计算是否更有效”，不把晚期分数融合
视为已证实的瓶颈。复用P0、原节点—边联合图和原两路分类头；不增加对齐、预训练、
分类分支或损失。原平均池化 `joint` 及原P0+C参照不改写。

唯一新读出为32维加性注意力：`s_t = vᵀ tanh(W_h h_t + W_g g + b)`，
对原attention_mask内的token归一化，并加权形成原源码分类头的输入。
`g`是联合图现有256维读出，图到权重到分类BCE的路径不detach。
范围完全沿用原平均池化，包括原有前缀、特殊token，padding排除；不新增token标注。
不构造token×节点矩阵。最终仍是source logit与graph logit相加。

三组比较：

- `joint`：原平均池化，直接复用 `results/cfg_joint_seed42/joint` 的已完成结果。
- `joint_source_pool`：相同注意力网络，以固定全1向量替代g，只依赖源码学习权重。
- `joint_structure_pool`：使用当前函数的真实图摘要g。

固定条件对照和结构条件模型具有完全相同的模块、初始化、参数量及投影运算量。
固定条件只表示可学习的全局偏置，因此二者的有效输入维度不同，不宣称有效表达能力相同。
采用带偏置的条件投影：小模型反例显示，无偏置tanh对正负对称输入会退化，不能
学会所需的结构条件化选择；加入偏置后同一四样本拟合通过。

评分向量v零初始化，初始均匀权重；用“原平均表示＋相对均匀权重的增量”计算，
保持原混合精度平均读出的数值初始化。实数算术下等价于注意力加权和；混合精度下
保留原均值的舍入误差。第一步上游投影梯度为零，v更新后上游开始学习。
初始化隔离随机数，保留共有LoRA、图和分类头的初始化及训练随机流。
新增聚合器跟随原源码参数组学习率；图组、裁剪、累积、BCE、3 epochs和valid选模不变。

Qwen隐藏维度3584时，两组各增加122,944个参数；2048 token时各增加234,954,752次
投影乘加，另有tanh、归一化及加权汇聚。投影预算相同，但不等于原平均池化的计算量；
真实GPU耗时和峰值显存尚未测量。

source/graph/fusion逐样本诊断、未缩放BCE及原分类指标保留；注意力熵和最大权重写入
分支诊断，裁剪前聚合器梯度范数写入原history。结构条件模型的source分数已含结构信息，
不能将它解释成独立源码baseline。注意力权重也不作为漏洞位置真值。

新结果写入 `results/cfg_pool_seed42`。启动时严格核对复用的平均模型配置、P0、checkpoint
及valid预测；配置不匹配则拒绝复用，不复制或覆盖历史模型。
比较文件同时含原P0+C、复用的平均池化、两种可学习池化，并输出相对原C、平均池化及
两种可学习池化之间的纠错/新增错误。test仅按已保存的融合valid阈值评价。

```bash
# 平均池化已经完成：自动校验并复用，无需重复训练。
python scripts/cfg.py pool train             # 训练两种可学习池化，形成三组对照
# 也可以单独执行：
python scripts/cfg.py pool train source
python scripts/cfg.py pool train structure

python scripts/cfg.py pool valid             # 三组与原P0+C比较
python scripts/cfg.py pool test              # 方法冻结后，使用保存的valid阈值
python scripts/cfg.py pool train --show-command
```

本轮只运行回归和CPU小模型训练/保存/重载/评价，不启动正式Qwen训练。
已有valid汇总仅提示需验证的假设，拟合成功不能代替真实分类增益。

## 固定模型的辅助任务迁移诊断

正式入口 `diagnose-transfer` 只做固定checkpoint推理，不训练、改阈值或生成新漏洞标签。
使用现有源码baseline、C、P0+C、control+C和结构池化模型；不读取test预测来设计诊断。
先固定valid样本并保存 `protocol.json` 和 `cases.jsonl`，再查看响应：

- 注释检查：按标签各随机选8个可完整解析且不截断的函数，seed=42；追加中性、风险措辞、
  安全措辞三种等token长度注释，验证可执行AST和原源码位置不变。因此复用不变的原图，
  用正式提取/编码函数同步生成模型输入。注释不是指令，不赋予任何新漏洞标签。
- 自然近邻：valid内同名、恰好两条且标签相反、字符相似度至少0.70的全部函数组。
  不是经过证明的修复对；单独记录可见范围、原标签判别、源码diff和缺失上下文。
- 控制任务位置参照：只从train统计分支、相对位置符号及固定距离分箱，Laplace平滑1；
  valid阈值固定0.5，不将该参照当成编码器能力证明。
- 读出干预：同一结构池化checkpoint，在原始/追加中性注释的输入上，只汇聚共同token前缀。
  两侧同时排除变化的边界和EOS，不更改编码器attention_mask或任何权重。
  这是机制诊断，不是新的正式分类方法；保留长度相关数值差异，不声称隐状态总是逐位相同。

```bash
python scripts/cfg.py diagnose-transfer --prepare-only  # 先固定样本
python scripts/cfg.py diagnose-transfer                 # 固定模型推理，默认cuda:0
python scripts/cfg.py diagnose-transfer --readout-check # 固定读出干预
python scripts/cfg.py diagnose-transfer --summarize-only # 仅重算持久化响应的统计
```

默认输出 `results/cfg_transfer_diagnostic_seed42`；已有响应不自动覆盖。
本轮已完成390次固定分类前向和32次读出干预，无参数更新。
原始分数复现最大误差约1.5e-8，注释变体的图logit不变。16个函数上，中性注释导致
baseline/C/P0+C/control+C/结构池化分别2/1/2/0/3次标签翻转。
结构池化平均绝对logit变化5.2403；共同前缀读出后降至0.00364。
原始前缀中有5个样本仍出现隐状态数值差异，尚未进一步定位底层长度相关计算路径。
风险与安全措辞之间没有标签翻转，不能把现象简单归因于“漏洞关键词诱导”。
结果说明当前读出存在可重复的无关尾部文本敏感性；不证明去掉注释就能提高完整valid性能。
详细响应、独立函数统计、人工源码审查及未知状态保存在上述目录。

### 单输入函数范围与末尾边界确认（2026-10-04）

分类器现在接受独立的 `pooling_mask`；不传时原读出不变，编码器仍使用原
`attention_mask`。`InputBuilder.source_function_batch` 使用Tree-sitter函数范围，
将字节坐标转换为字符坐标后映射到原token；不改变任务前缀、源码token、2048源码预算或EOS。
签名、模板声明、函数体和内部注释仍在范围内。缺失返回类型的片段只在解析时补合成类型，
不改变模型输入，也不推断真实类型。不完整或歧义边界明确报错，不删除成员、不返回零向量或回退旧池化。
错误恢复可能把 `else if` 当成函数，因此控制语句不能充当函数范围。

本轮考察两个读出候选：完整函数范围；以及在该范围上排除纯末尾右花括号token。
第二项仅排除在函数范围内只含 `}` 和空白的末尾token；含运算符、常量等内容的混合token保留。
这是边界位置的直接汇聚干预，不是删除注释，也没有添加新的注意力模块或损失。
原结构条件化注意力、图传播和P0权重不变；排除直接汇聚不表示消除对其他隐藏表示的影响。

两批各32个train/valid函数，互不重叠，也排除此前16个开发函数；各split按标签各选8个，
在读取响应前保存成员和全部扰动。每批含原始、换行、短/长块注释、行注释、内部注释共192个输入。
第二批的 `primevul:208675` 曾进入旧近邻源码审查，不能算全新函数；已保留该样本及全部结果，没有在看到响应后替换。
两个固定模型共完成768次Qwen前向，eval/no_grad，无参数更新。所有阈值保持原checkpoint值。

第二批valid确认集（16函数）上的平均绝对logit差：

| 固定模型/读出 | 仅换行 | 短尾注释 | 长尾注释 | 内部注释 |
| --- | ---: | ---: | ---: | ---: |
| P0+C / 原读出 | 1.22149 | 1.03900 | 0.60940 | 0.24364 |
| P0+C / 函数范围且排除纯末尾token | 0 | 0.00286 | 0.01051 | 0.33613 |
| 结构池化 / 原读出 | 4.42799 | 3.86561 | 3.45658 | 0.36442 |
| 结构池化 / 仅函数范围 | 3.62962 | 3.63978 | 3.63502 | 0.42733 |
| 结构池化 / 函数范围且排除纯末尾token | 0 | 0.01056 | 0.01790 | 0.43725 |

该批结构池化在原始valid确认输入上的正确数为12→13/16，但train确认集为15→13/16；
内部注释在valid确认集仍引起2次翻转。这些是固定模型干预，不能当成重训后的完整valid成绩。
尾部扰动后的共同token隐状态并非总是逐位一致，长度相关数值差异尚未定位；图logit最大差约1.2e-7。
新规则不是恒定输出：该valid确认集原始logit标准差为3.86→4.17。
已有完整741条valid成绩仍为A：MCC 0.49346/AUC 0.82083，结构池化：MCC 0.52027/AUC 0.81594；
本轮没有生成新的完整valid分类成绩。

最终边界核查为train 5862/5886、valid 734/741，31个输入未可靠恢复。
例如valid `primevul:379917` 混有上一函数尾部，而目标函数 `iscsi_destroy_flashnode_conn` 只剩签名及左花括号。
这不是模型token截断，不能在保留原输入的前提下恢复缺失函数体。
没有用子集训练、样本特判或旧读出回退绕过这些成员，因此正式训练0次、test终验0次。
关键代码token保留及小模型拟合已验证，但现有自然近邻缺少安全配对依据，真实关键差异的正确响应仍未验收。
目前只保留读出诊断能力，不将候选登记为正式训练方法或宣称分类净收益。

```bash
python scripts/cfg.py diagnose-transfer --scope-check --prepare-only
python scripts/cfg.py diagnose-transfer --scope-check
python scripts/cfg.py diagnose-transfer --boundary-check --prepare-only
python scripts/cfg.py diagnose-transfer --boundary-check
python scripts/cfg.py diagnose-transfer --boundary-check --summarize-only
python -m unittest tests.test_cfg_function_readout tests.test_cfg_source_pool tests.test_cfg_alignment tests.test_cfg_transfer_diagnostics -q
```

结果位于 `results/cfg_transfer_diagnostic_seed42/function_readout/`。
`protocol[.boundary].json` 保存源码、成员、范围缺项和输入规则；`responses[.boundary].jsonl`
保存原阈值下逐项logit及共同前缀隐状态差；`summary[.boundary].json` 保存分类指标、未缩放BCE、
纠错/新增错误和扰动统计。已有推理响应拒绝覆盖，`--summarize-only` 不重新运行Qwen。
