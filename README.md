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

## 2026-10-04—10-08：程序消息传输与保留原任务的语义学习

本轮固定原P0+C为A（`results/cfg_dep_cfg_windowfix_seed42`），只用train/valid开发。
原阶段1为1轮，分类3轮，seed=42，源码2048 token；不修改原始分类数据或标签。
新阶段1最多2次、分类最多8次。已有同协议结果复用，test仅在最终方案确定后终验。

研究依据与第一轮假设：

- [DeepDFA](https://arxiv.org/pdf/2212.08108)将定义抽象与CFG上的聚合/更新对应，
  [官方特征加载](https://github.com/ISU-PAAL/DeepDFA/blob/master/DDFA/sastvd/linevd/graphmogrifier.py)
  使用API、类型、字面量、运算符。这里保留原C的定义路径，同时检验局部行为对消息的增量。
  `joint_transport`复用现有schema-2角色绑定节点/边及全部参数，将旧联合模型的
  `m * tanh(F(actual)-F(neutral))`残差改为` tanh(F(actual)-F(neutral))`。
  因而在原消息某维为零时也能引入行为分量。保持原C自消息、五轮GRU、原属性读出、
  零初始化和未知掩蔽。不是GEN/KILL静态分析器，也不是仅增加容量。
  `joint_transport_shuffled`只打乱函数内完整边属性记录，参数、边数和计算形式相同。
  旧`joint`保持原实现，是同容量的乘法消息对照；短入口旧默认运行集合保持不变。
- [PDBERT](https://arxiv.org/html/2402.00657v1)从源码预测数据/控制依赖，并分别评价辅助任务
  和下游任务；双向CodeBERT不能直接等同于本项目的因果Qwen。现有控制预训练在valid上的
  数据依赖AUC为0.53117（P0为0.72180），控制AUC为0.90649，位置参照为0.77311。
  这些证据支持检验任务干扰，但不证明干扰是唯一原因。
  本轮借鉴[梯度投影](https://arxiv.org/abs/2001.06782)，保持同一源码前向、原CLM/定义—使用
  及控制任务、原采样和有效函数归一化。在累积窗口内分别累计LoRA的原任务梯度g与控制梯度a，
  若内积为负，使用`g + a - (a·g)/(g·g)*g`，否则使用`g+a`。
  原任务g为CLM与定义—使用之和，不能保证其中每项能力不下降；Adam有限步更新也不受一阶
  正交性质保证。头使用各自原梯度，最后沿用原梯度裁剪及AdamW。
  `--control-gradient-policy project`启用；默认`sum`完全保留旧路径。
  更新次数相同，但两次反向增加计算，不能称为等FLOPs比较；不是新提出的通用优化算法。

原定义—使用监督train覆盖1000/5886函数，其中248个函数同时有正负关系；valid覆盖114/741，
其中26个同时有两类。控制监督覆盖4737/5886和619/741。关系头只读可见token表示及查询分支，
不读图答案；两端表示的联合预测可见性以较晚端点为界，不要求较早位置看到后文。
角色、字面量、类型绑定与原输入路径不变。是否产生分类净收益必须由本轮完整分类检验。

固定读出对照使用`diagnose-transfer --original-valid`：仅比较原始valid中734个范围可靠成员，
原读出与唯一候选（函数范围并排除纯终止符token）共享权重、成员、阈值和一次编码。
其余7条保留为缺失覆盖；不补分数，不作为741条完整valid结果，也不阻塞M1/M2。

```bash
conda run -n vul-detect python -m unittest tests.test_cfg_joint_control tests.test_cfg_dependency tests.test_cfg_function_readout tests.test_cfg_transfer_diagnostics tests.test_cfg_source_pool tests.test_cfg_cli -q
conda run -n vul-detect python scripts/cfg.py diagnose-transfer --original-valid --device cuda:1
conda run -n vul-detect python scripts/cfg.py pretrain-dep \
  --reference-run-dir results/cfg_abc_seed42 --supervision-dir data/cfg_dep_seed42 \
  --control-dir data/cfg_control_seed42 --reference-pretrain-dir results/cfg_dep_pretrain_windowfix_seed42 \
  --control-gradient-policy project --modes control_pretrain \
  --output-dir results/cfg_project_pretrain_seed42 --device cuda:0
conda run -n vul-detect python scripts/cfg.py run \
  --dataset data/function_dataset.jsonl --graphs data/graphs/primevul_cfg.jsonl \
  --reference-run-dir results/cfg_abc_seed42 --comparison-run-dir results/cfg_dep_cfg_windowfix_seed42 \
  --pretrain-dir results/cfg_dep_pretrain_windowfix_seed42 --behavior-dir data/cfg_joint_seed42 \
  --variants joint_transport joint_transport_shuffled --output-dir results/cfg_transport_seed42 --device cuda:1
```

固定读出已完成（同权重、同阈值，734/741 valid，7条无候选预测）：

| 固定模型/读出 | BCE | AUC | MCC | Accuracy | Precision | Recall | F1 | 纠正/新增错误 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P0+C/原读出 | .766767 | .819932 | .491190 | .745232 | .759312 | .720109 | .739191 | — |
| P0+C/候选 | .755264 | .817964 | .476850 | .738420 | .740437 | .736413 | .738420 | 6/11 |
| 结构池化/原读出 | .704482 | .815589 | .518482 | .758856 | .745501 | .788043 | .766182 | — |
| 结构池化/候选 | .755424 | .807318 | .465799 | .728883 | .693364 | .823370 | .752795 | 13/35 |

候选在两种固定模型上均降低MCC/AUC，没有分类净收益，暂停M3，不消耗正式分类训练预算。
此结果保存在原诊断目录的`function_readout/{protocol,responses,summary}.valid.*`；不是完整valid结果。

阶段1实际完成：`cfg_project_pretrain_seed42/control_pretrain/last.pt`，5886函数、736次更新，
412/736个窗口触发投影，定义—使用4658条（1000函数），控制4737函数；编码器逻辑前向5886次、
反向11772次（另含梯度检查点重算）。在线CLM/定义—使用/控制均值为1.082193/2.805210/0.509394。
在线指标不是固定模型能力评价，不由这些loss判断下游收益。

阶段1固定train/valid评价已完成（`results/cfg_project_relations_seed42/metrics.json`）：

| 阶段1/任务 | train函数均值BCE | train AUC | valid函数均值BCE | valid AUC | valid MCC（0.5） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧控制预训练/定义—使用 | 1.997748 | .537149 | 2.611069 | .531169 | -.003376 |
| 投影版/定义—使用 | .443504 | .886028 | .419751 | .888525 | .500553 |
| 旧控制预训练/控制 | .331774 | .914984 | .346517 | .906488 | .650546 |
| 投影版/控制 | .355946 | .903670 | .362637 | .899598 | .607418 |

P0原定义—使用valid AUC为.721801。投影版恢复了当前关系头与编码器组合的任务表现，
但不能仅由此断言编码器产生了下游迁移增益。train/valid的成员、查询及阈值未改。
另查原随机顺序下184/736个累积窗口没有定义—使用监督，故投影不是对所有历史语义能力的保持约束。

```bash
conda run -n vul-detect python scripts/cfg.py eval-control \
  --pretrain-dir results/cfg_project_pretrain_seed42 --output-dir results/cfg_project_relations_seed42 --device cuda:0
conda run -n vul-detect python scripts/cfg.py run \
  --dataset data/function_dataset.jsonl --graphs data/graphs/primevul_cfg.jsonl \
  --reference-run-dir results/cfg_abc_seed42 --comparison-run-dir results/cfg_dep_cfg_windowfix_seed42 \
  --pretrain-dir results/cfg_project_pretrain_seed42 --variants control_cfg \
  --output-dir results/cfg_project_cfg_seed42 --device cuda:0
```

捷径边界：对同一use同时有已证实正负定义的227条valid查询（26个函数），P0、旧控制预训练、
投影版AUC分别为.776794/.693620/.841228；仅选择最后出现的定义达到.847368。
因此定义—使用头变好仍不能证明超越位置捷径，不能将.888525的全查询AUC直接解释为上下文推理。

与旧负结果的区别：`behavior_edges`已经尝试过加性边MLP，本轮不宣称首次加性消息。
该旧版本采用schema-1未绑定的边家族嵌入；`joint_transport`采用schema-2角色绑定事实、
实际/中性消息差、未知掩蔽及保留原C的节点残差，并与原`joint`严格同参数量。
所以本轮检验的是已有绑定表示下乘法限制的影响，不把更名或加性公式本身作为论文创新。

本轮实际完成1次阶段1、6个全量分类运行，全部正常结束；A与原P0+joint复用已有同协议结果。
六个新分类均为5886个原train成员、3轮、每轮736次优化更新。测试集761个成员均完成评价，
没有修改数据、标签、成员、输入预算或优化设置。新增候选各自保留完整3轮history、固定分支BCE、
最佳checkpoint和逐条预测；B打乱选中epoch 3，其余五个新运行选中epoch 2。

完整valid（741个原成员；E为test前选定候选；各阈值仅由valid决定）：

| 配置 | 阈值 | BCE | AUC | MCC | Accuracy | Precision | Recall | F1 | 纠正/新增（相对A） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A：P0+原C | 0.30 | 0.763567 | 0.820834 | 0.493458 | 0.746289 | 0.762040 | 0.721180 | 0.741047 | 0/0 |
| 原P0+joint | 0.26 | 0.646659 | 0.813724 | 0.495732 | 0.747638 | 0.737245 | 0.774799 | 0.755556 | 37/36 |
| B：P0+传输 | 0.39 | 0.816024 | 0.808974 | 0.482011 | 0.738192 | 0.780564 | 0.667560 | 0.719653 | 21/27 |
| B打乱 | 0.30 | 0.729469 | 0.813491 | 0.512641 | 0.755735 | 0.774286 | 0.726542 | 0.749654 | 51/44 |
| C：M2+原C | 0.09 | 0.818745 | 0.807655 | 0.496712 | 0.747638 | 0.730198 | 0.790885 | 0.759331 | 29/28 |
| D：M2+传输 | 0.11 | 0.862815 | 0.817782 | 0.514915 | 0.757085 | 0.771831 | 0.734584 | 0.752747 | 26/18 |
| D打乱 | 0.18 | 0.696296 | 0.819064 | 0.514138 | 0.757085 | 0.757333 | 0.761394 | 0.759358 | 28/20 |
| E：M2+原joint | 0.31 | 0.735600 | 0.825431 | 0.517230 | 0.755735 | 0.800000 | 0.686327 | 0.738817 | 31/24 |

完整test（761个原成员；E为test前选定候选；各阈值仅由valid决定）：

| 配置 | 阈值 | BCE | AUC | MCC | Accuracy | Precision | Recall | F1 | 纠正/新增（相对A） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A：P0+原C | 0.30 | 0.658639 | 0.839496 | 0.527317 | 0.763469 | 0.751938 | 0.776000 | 0.763780 | 0/0 |
| 原P0+joint | 0.26 | 0.567564 | 0.838701 | 0.492449 | 0.743758 | 0.712264 | 0.805333 | 0.755945 | 27/42 |
| B：P0+传输 | 0.39 | 0.687877 | 0.832642 | 0.480274 | 0.739816 | 0.755043 | 0.698667 | 0.725762 | 21/39 |
| B打乱 | 0.30 | 0.712049 | 0.818142 | 0.492761 | 0.746386 | 0.752778 | 0.722667 | 0.737415 | 44/57 |
| C：M2+原C | 0.09 | 0.699759 | 0.831959 | 0.506685 | 0.750329 | 0.715618 | 0.818667 | 0.763682 | 20/30 |
| D：M2+传输 | 0.11 | 0.735650 | 0.839102 | 0.526022 | 0.762155 | 0.741294 | 0.794667 | 0.767053 | 18/19 |
| D打乱 | 0.18 | 0.615468 | 0.839295 | 0.542323 | 0.770039 | 0.746305 | 0.808000 | 0.775928 | 25/20 |
| E：M2+原joint | 0.31 | 0.675074 | 0.837651 | 0.505890 | 0.752957 | 0.759003 | 0.730667 | 0.744565 | 19/27 |

统一固定阈值0.5的MCC（同一选中模型，无再训练、无再选点）：

| 配置 | valid | test |
| --- | ---: | ---: |
| A：P0+原C | 0.480797 | 0.499180 |
| 原P0+joint | 0.454756 | 0.507206 |
| B：P0+传输 | 0.459138 | 0.473752 |
| B打乱 | 0.473276 | 0.488083 |
| C：M2+原C | 0.447055 | 0.462549 |
| D：M2+传输 | 0.456147 | 0.475155 |
| D打乱 | 0.468412 | 0.483953 |
| E：M2+原joint | 0.482831 | 0.490853 |

BCE均未缩放；新运行valid来自选中epoch的固定logit评价，test来自`test.bce.json`。
A的BCE由保存的概率重算，已核查无0/1端点；这与直接保存logit的数值来源有区别。
`comparison.{valid,test}.json`保留逐条纠错身份及统一0.5阈值对照，不只保留最好的一轮。

本轮判断：

- **M1传输公式不采用。** B相对A的valid/test净正确数为−6/−18，打乱版本反而更好。
  D相对A的valid净增8条，但同M2打乱版本也净增8条，两者MCC仅差.000777；
  原乘法joint的valid MCC还更高。正确关系传输的增益没有被同容量对照支持。
  原C与原joint继续作为既有实验对照，不能把新增公式写成已验证的程序抽象贡献。
- **M2保留实现及研究结果，不认定已获得相对A的稳定分类净收益。** 定义—使用辅助能力恢复，
  C在valid只净增1条（少FN 26、多FP 25），test净少10条。E的valid MCC相对A增加.023771、
  test下降.021427，test纠正19条、新增27条。E相对同表示的P0+joint，valid/test MCC增加
  .021498/.013441，净正确数增加6/7条；但test少FP 35同时多FN 28，F1下降，仍未超过A。
  这是依赖基础表示和操作点的改善，不能写成全面增强漏洞语义。
- **不按test改选。** D的test净少1条；D打乱净增5条、MCC增加.015006，但它是预定负控制。
  不将打乱版本事后改选为最终方法，也不将其收益归因于正确程序关系。
- **M3暂停。** 原始valid覆盖范围内的固定读出对照没有分类净收益，未启动额外分类训练。

C第3轮train BCE从.365662升至.583511，valid AUC降至.762006；日志无非有限值或执行异常。
保留退化结果，不用改学习率、加轮数或换loss追分。辅助关系准确、可迁移能力、分类使用和主任务收益
仍是四个不同命题；本轮只充分支持辅助任务恢复，以及上述有限的分类操作点变化。
没有固定CLM能力复验，不能声称全部原P0能力都被保持；更不能把梯度投影本身当成新的通用优化贡献。

参数与计算：原C图编码器377,985参数，joint/传输及各自打乱版本均596,993，增加219,008；
含分类头的总可训练参数由10,474,370增至10,693,378。传输与原joint同容量、同连接和同MLP调用数。
每个train图传播pass仍为6,568,640节点更新、13,547,435条基础消息；额外绑定事实/MLP计算不记为免费。
M2不增加最终编码器参数，沿用相同数据、采样、epoch及736次阶段1更新，但每函数两次逻辑反向，
不是等FLOPs实验。没有消费剩余1次阶段1和2个分类预算去做无依据搜索。

**本轮test前固定决策（2026-10-08）**：停止新增训练，实际完成1次阶段1、6次分类。
按既有valid融合MCC规则，非打乱候选中选定“投影M2+原joint”（`cfg_project_joint_seed42/control_joint`），
而不保留新增M1传输作为论文模块；M3暂停。该候选仅有单seed、依赖读出阈值的valid信号，
尚不宣称普遍改善或完整原能力保持。阶段1权重、各分类best.pt与阈值全部固定。
终验范围预先确定为A、B、B打乱、C、D、D打乱、M2+原joint，并复用原P0+joint作为同表示对照。
各自固定阈值依次为.30/.39/.30/.09/.11/.18/.31；原P0+joint沿用其已保存valid阈值。
所有候选均保留，test之后不重新挑选、改方法或调阈值。test曾被历史多轮查看，
本次只能称固定终验，不能称从未使用的独立确认集。

终验已按上述固定决策完成；未依据test修改任何方法、checkpoint或阈值。

进一步核查的核心问题是：**关系任务是否给出了漏洞标签需要的判别信息，并实际迁移到分类器？**
这不是“关系头答得对”或“图结构更完整”可以替代的命题。使用原train/valid保存产物，未新增训练：

- 对完全一致的600条定义—使用valid查询（114函数），比较P0与投影阶段1的函数均值Brier分数。
  102个函数的辅助Brier改善，但A→C仅纠正其中3条分类错误，又引入4条；辅助与分类Brier改善
  的函数级Pearson相关为.076292。80个新关系查询全部答对的函数中，仍有23个分类错误。
  此处分类器保持原C，用于隔离阶段1变化；Brier避免旧概率恰为0/1时伪造有限BCE。
- 同一个E固定模型，源码分支valid AUC为.829016，融合后为.825431；沿用融合阈值.31时，
  融合相对源码分支纠正16条、引入11条，净增5条。图可以改变判别操作点，但这个结果不能
  证明提升了排序，也不能由移除读出分支反推训练期间图完全无用。
- 原诊断预先定义的7对同名、近似源码、相反标签成员全部保留，源码差异均在2048范围内。
  E正确排序4/7、两边同时分类正确0/7；A为3/7、1/7。它们没有核验过的修复对来源，
  部分依赖调用者或库契约，不能据此赋予局部安全标签，也不足以证明所有程序语义都不可学。

这些观测支持“辅助目标与下游判别之间尚有未验证的迁移环节”，不能单独断言是任务无关、
编码器未学会、分类器未使用或标签上下文不足中的哪一个。更深的问题因而应明确为：
在不扩大输入和数据的约束下，新增程序学习到底要改善哪一种可观察的漏洞判别能力，
以及现有标签和上下文是否足够检验它。此轮不再用另一个高辅助分数或偶然的分类高点替代回答。
逐函数关联及7对成员记录已追加到原诊断`results/cfg_transfer_diagnostic_seed42/saved_analysis.json`
的`project_iteration`字段，保留原分析；没有改写原分类数据或生成新安全标签。

剩余分类命令（其余预训练、B/C及辅助评价命令见上文）：

```bash
conda run -n vul-detect python scripts/cfg.py run \
  --dataset data/function_dataset.jsonl --graphs data/graphs/primevul_cfg.jsonl \
  --reference-run-dir results/cfg_abc_seed42 --comparison-run-dir results/cfg_dep_cfg_windowfix_seed42 \
  --pretrain-dir results/cfg_project_pretrain_seed42 --behavior-dir data/cfg_joint_seed42 \
  --variants control_transport --output-dir results/cfg_project_transport_seed42 --device cuda:0
conda run -n vul-detect python scripts/cfg.py run \
  --dataset data/function_dataset.jsonl --graphs data/graphs/primevul_cfg.jsonl \
  --reference-run-dir results/cfg_abc_seed42 --comparison-run-dir results/cfg_dep_cfg_windowfix_seed42 \
  --pretrain-dir results/cfg_project_pretrain_seed42 --behavior-dir data/cfg_joint_seed42 \
  --variants control_transport_shuffled --output-dir results/cfg_project_transport_shuffled_seed42 --device cuda:0
conda run -n vul-detect python scripts/cfg.py run \
  --dataset data/function_dataset.jsonl --graphs data/graphs/primevul_cfg.jsonl \
  --reference-run-dir results/cfg_abc_seed42 --comparison-run-dir results/cfg_dep_cfg_windowfix_seed42 \
  --pretrain-dir results/cfg_project_pretrain_seed42 --behavior-dir data/cfg_joint_seed42 \
  --variants control_joint --output-dir results/cfg_project_joint_seed42 --device cuda:1
```

固定终验命令如下，全部已经实际执行。已有预测时入口拒绝覆盖；复现新训练应选新的结果目录，
不要用`--replace-predictions`覆盖本轮结果。

```bash
conda run -n vul-detect python scripts/cfg.py eval --run-dir results/cfg_transport_seed42 --variants joint_transport joint_transport_shuffled --split test --device cuda:0
conda run -n vul-detect python scripts/cfg.py eval --run-dir results/cfg_project_cfg_seed42 --variants control_cfg --split test --device cuda:0
conda run -n vul-detect python scripts/cfg.py eval --run-dir results/cfg_project_transport_seed42 --variants control_transport --split test --device cuda:1
conda run -n vul-detect python scripts/cfg.py eval --run-dir results/cfg_project_transport_shuffled_seed42 --variants control_transport_shuffled --split test --device cuda:0
conda run -n vul-detect python scripts/cfg.py eval --run-dir results/cfg_project_joint_seed42 --variants control_joint --split test --device cuda:1
```

验证：相关6个测试模块共77项通过，包含原C/P0路径、共享初始化、角色绑定/未知、单次逻辑编码、
LoRA梯度、窗口归一化、梯度投影、小模型前后向/保存/加载/评价、读出覆盖及旧CLI默认集合。
初次扩展variant导致旧短CLI默认集合回归，已在正式训练前修复并复验；无遗留失败。
新增M2打乱别名后再次77项通过（30.235s）。最终逐条重算8个配置的valid/test指标，
核对741/761唯一成员、身份、浮点阈值决策、valid/test隔离和共用训练设置；六个新history均完整3轮。
交互中断发生在首批终验数据加载阶段，已确认进程不存在且无预测后重跑未完成评价；没有重复训练。

### P0 收益来源诊断（2026-10-08）

本轮只补贡献拆解、固定 CFG 扰动和冻结编码器探测，不设计新模型，不用 test 筛选。
复用 `cfg_abc_seed42/baseline` 与 `cfg_dep_cfg_windowfix_seed42/dep_pretrain_cfg`；
缺失的 `dep_pretrain_source` 委托原源码训练器，`dep_pretrain_attributes` 委托原图训练器，
两者加载同一个 P0 `last.pt`，分类头重新初始化，保持 seed=42、2048、3 epochs 和原优化设置。
无传播属性模型仍执行相同节点 GRU 更新，仅移除邻居消息，与原 attributes 对照一致。

`diagnose-origin --phase topology` 对原 valid 固定 P0+C 权重及阈值，每个函数只编码源码一次；
以种子 42/43/44 的有向双边交换保留每个节点入度、出度及自环，另报告无边和关闭图分支。
原连接重放必须与保存概率相差不超过 2e-5；无法交换的图留在全体分母，报告实际改变覆盖。

`diagnose-origin --phase probe` 冻结原始 Qwen、P0 阶段1、P0+C 阶段2编码器，复用原定义—使用
监督及原 train/valid；缓存同一源码输入的均值表示和定义/使用位置表示。统一使用训练集均值与标准差
归一化、seed=42、AdamW lr=1e-3、weight_decay=.01、每批32个函数、20 epochs。
关系头复用 rank=32 DirectedRelationHead，函数内平均损失后按函数平均，固定取第20轮；
源码线性探测按原 valid MCC/F1/Accuracy/AUC 规则选 checkpoint 和阈值。
这是冻结表示诊断，不能冒充 Qwen 全量分类训练或原辅助头成绩。

探测的首次提取因新增检查误将 `definition_token <= use_token` 作为必要条件而停止，未训练探测头。
原 P0 关系头在两个端点均已可见后判定，等价于在较晚端点之后读出；使用位置自身仍是因果表示。
因此保留原监督的反向位置正例（train 155、valid 13），检查两个位置均不越界，不改变关系标签。
失败产物保留在 `cfg_origin_probe_seed42`；修正检查后的完整探测使用 `cfg_origin_pair_probe_seed42`。

本轮已完成：新增2次全量分类（各3轮、2208次优化更新），未新增Qwen预训练；
完成741个valid成员的固定拓扑评价，以及3个冻结编码器、各2种探测头。
分类使用相同5886/741/761成员与原协议。所有分类checkpoint与阈值由valid确定。
三个诊断完成后，仅对预先指定的两个新增分类对照补一次test终验；Source-only、P0+C的历史test直接复用，
未根据test修改方法或参数。历史test不是未使用过的独立确认集。

| 模型 | 划分 | BCE | AUC | MCC | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Source-only | valid | .7788 | .8047 | .5116 | .7517 | .8058 | .6676 | .7302 |
| P0+Source-only | valid | .7057 | .8213 | .5126 | .7557 | .7743 | .7265 | .7497 |
| P0+无传播属性 | valid | .6685 | .8192 | .5108 | .7530 | .7230 | .8257 | .7710 |
| P0+C | valid | .7636 | .8208 | .4935 | .7463 | .7620 | .7212 | .7410 |
| Source-only | test | .7527 | .8142 | .4802 | .7385 | .7716 | .6667 | .7153 |
| P0+Source-only | test | .6252 | .8369 | .4957 | .7477 | .7377 | .7573 | .7474 |
| P0+无传播属性 | test | .5888 | .8368 | .5178 | .7516 | .7022 | .8613 | .7737 |
| P0+C | test | .6586 | .8395 | .5273 | .7635 | .7519 | .7760 | .7638 |

这里BCE由保存概率计算，没有缩放或裁剪，所有概率均非0/1；固定模型原生logit BCE在拓扑报告中保留。
四项阈值依次为 .31/.23/.16/.30，选中epoch依次为3/2/2/2。
源码两项的可训练参数均为10,096,129；无传播属性和C均为10,474,370。
P0阶段1权重相同；无传播属性与C有相同参数量和5次节点更新，C额外进行邻居聚合。

| 相邻对照（后者相对前者） | valid纠正/新增错误 | test纠正/新增错误 |
|---|---:|---:|
| Source-only → P0+Source-only | 47 / 44 | 58 / 51 |
| P0+Source-only → P0+无传播属性 | 41 / 43 | 47 / 44 |
| P0+无传播属性 → P0+C | 40 / 45 | 44 / 35 |
| P0+Source-only → P0+C | 16 / 23 | 32 / 20 |

这些是沿固定对照路径的条件差异，不能假设模块无交互后解释为独立可加的机制贡献。
P0已经复现主要AUC增益；属性模型BCE更低，但高F1伴随更多误报；C的test净收益确实存在，
却未在valid原选阈值指标上重复。统一0.5阈值时，四项valid MCC为
.4539/.4628/.4648/.4808，test为.4333/.4738/.4919/.4992；
这说明分类结论对操作点敏感，不能只选对某个模型有利的一组阈值结果。
完整指标、固定0.5结果、纠错成员和配置见 `results/cfg_origin_source_seed42/contribution.json`。

**固定拓扑评价只使用valid，所有干预都沿用P0+C阈值.30。**

| 干预 | 实际改变函数数 | 平均原边替换比例 | 平均概率绝对变化 | MCC | AUC | BCE |
|---|---:|---:|---:|---:|---:|---:|
| 原连接 | 0 | 0 | 0 | .493458 | .820834 | .763569 |
| 保度重连42 | 734 | .954745 | .000246 | .490677 | .820783 | .763651 |
| 保度重连43 | 734 | .953729 | .000196 | .493458 | .820827 | .763558 |
| 保度重连44 | 736 | .959620 | .000180 | .493458 | .820791 | .763810 |
| 移除全部边 | 741 | 1 | .006369 | .479725 | .821235 | .756347 |
| 关闭图分数 | — | — | .025450 | .477094 | .820645 | .717593 |

三次重连分别仅翻转1/0/0个分类决定。原连接重放通过全部741条保存概率一致性检查。
图分支logit并非常数（均值-.202904、标准差.528493），但保度重连几乎不影响输出。
因此valid证据支持对具体连接弱依赖，不能将C的收益直接归为真实CFG拓扑推理。
度数/节点统计和联合优化尚未被完全拆开；这里没有执行test拓扑扰动，不能把valid干预当成test结果。
关闭图分支相对融合净损失6个正确预测，但BCE改善、AUC几乎不变；这也不能代替独立训练的P0+Source-only。
逐成员结果见 `results/cfg_origin_topology_seed42/predictions.jsonl` 和 `summary.json`。

**统一冻结探测结果（valid）。** 三个编码器均处理全部6627个train/valid函数；
依赖任务共train 4658查询/1000函数、valid 600查询/114函数。每个编码器用同样的20轮探测预算。

| 冻结编码器 | 依赖AUC | 依赖MCC（.5） | 依赖函数均值BCE | 同使用点混合候选AUC | 线性分类AUC | 线性分类MCC |
|---|---:|---:|---:|---:|---:|---:|
| 原始Qwen | .7657 | .3910 | .9603 | .7153 | .7885 | .4628 |
| P0阶段1 | .9056 | .6598 | .4261 | .8326 | .7729 | .4407 |
| P0+C阶段2 | .8731 | .5191 | .9507 | .7774 | .7921 | .4696 |

P0提高了新关系头可读出的能力，因此不能再把全部辅助收益归为旧头拟合；阶段2后部分能力仍保留，
也不能称为完全遗忘。不过，关系概率校准差：三个函数均值BCE都高于train函数先验常数预测的.283317。
仅26个valid函数的227查询具有同使用点正负候选；不读代码语义、只选最后一个候选定义的AUC为.847368，
高于三个编码器的混合候选探测结果。所有train/valid完整指标和BCE均保留，未按辅助AUC单独宣称任务成功。

P0相对原始Qwen的辅助函数Brier在39/114个函数改善；这些函数中，正式P0+Source-only相对Source-only
纠正3条分类错误、引入2条，辅助/分类Brier改善相关系数为.151160。该关联弱且只覆盖有限函数，不能证明因果。
冻结线性探测没有显示P0带来直接分类增益，但这不是“表示中没有分类信息”的证明：
把原P0+C源码分类头作用于缓存均值表示，可重现原源码分支预测，最大概率误差1.79e-7；
其AUC为.820645，高于重新拟合探测头的.792072，说明读出拟合/正则化仍影响探测结论。
原始Qwen/P0/P0+C关系探测train AUC为.9641/.9813/.9711，分类探测train AUC为.9343/.9311/.9934，
同时存在明显train/valid差距。不能只用训练拟合或单个高辅助AUC说明获得漏洞判别能力。
完整缓存、六个探测头、20轮历史及关联分析见 `results/cfg_origin_pair_probe_seed42/`。

本轮最有依据的归因是：P0带来源码排序和概率质量改善，C另外改变判别分数及优化过程，
其test分类收益尚不能归因于具体CFG连接；新增可探测依赖能力也未直接解释分类收益。
下一步最值得研究的是：**相同源码适配预算下，依赖监督相对CLM-only究竟增加了哪些可泛化的漏洞判别信息？**
现有旧CLM+C不能替代同设置的CLM+Source-only控制；此轮不继续扩展训练或增加模型模块。

实际检查：80项相关回归通过；修正端点检查后，10项诊断测试通过；
两项全量分类、两项固定test评价、完整拓扑评价及三编码器探测均退出0。
首次小模型评价发现新源码checkpoint缺少阶段1来源记录，已在正式训练前修复；
首次冻结提取的错误检查及失败目录如上保留，没有删监督成员或修改原分类数据。

本轮实际命令（已有输出不会被覆盖；再次复现须使用空输出目录）：

```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
$PY scripts/cfg.py run --dataset data/function_dataset.jsonl --graphs data/graphs/primevul_cfg.jsonl --reference-run-dir results/cfg_abc_seed42 --pretrain-dir results/cfg_dep_pretrain_windowfix_seed42 --comparison-run-dir results/cfg_dep_cfg_windowfix_seed42 --variants dep_pretrain_source --output-dir results/cfg_origin_source_seed42 --device cuda:0
$PY scripts/cfg.py run --dataset data/function_dataset.jsonl --graphs data/graphs/primevul_cfg.jsonl --reference-run-dir results/cfg_abc_seed42 --pretrain-dir results/cfg_dep_pretrain_windowfix_seed42 --comparison-run-dir results/cfg_dep_cfg_windowfix_seed42 --variants dep_pretrain_attributes --output-dir results/cfg_origin_attributes_seed42 --device cuda:1
$PY scripts/cfg.py diagnose-origin --phase topology --run-dir results/cfg_dep_cfg_windowfix_seed42 --output-dir results/cfg_origin_topology_seed42 --device cuda:0
$PY scripts/cfg.py diagnose-origin --phase probe --run-dir results/cfg_dep_cfg_windowfix_seed42 --output-dir results/cfg_origin_pair_probe_seed42 --device cuda:0
$PY scripts/cfg.py eval --run-dir results/cfg_origin_source_seed42 --variants dep_pretrain_source --split test --device cuda:0
$PY scripts/cfg.py eval --run-dir results/cfg_origin_attributes_seed42 --variants dep_pretrain_attributes --split test --device cuda:1
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -m unittest tests.test_cfg_dependency tests.test_cfg_ablation tests.test_cfg_transfer_diagnostics tests.test_graph_training.GraphTrainingTests.test_source_pretrained_adapter_fresh_head_roundtrip -q
```

### CLM 与依赖监督的独立贡献（2026-10-08）

固定比较原 `cfg_abc_seed42/baseline`、新增 `cfg_clm_source_seed42/lm_pretrain_source`、
已有 `cfg_origin_source_seed42/dep_pretrain_source`。新增分支仅把 CLM LoRA 加载到原源码分类训练器，
不引入图分支，分类头、优化器重新初始化；阶段2同为 seed=42、2048 tokens、3轮，按原 valid 规则选权重和阈值。
CLM 复用 `cfg_dep_pretrain_seed42/lm_pretrain/last.pt`，P0 复用
`cfg_dep_pretrain_windowfix_seed42/dep_pretrain/last.pt`：阶段1都为5886个train函数、1轮、736次更新，
其配置除依赖损失累积归一化标记外一致。核查历史实现：CLM 的微批损失始终按 `m/M` 加权，
没有依赖头时该修复不改变CLM目标；关系头初始化使用独立CPU RNG作用域，不改变源码LoRA初始化。
因此无需重跑预训练。依赖头带来额外计算，等训练步数不等于严格等FLOPs。

补充CLM冻结探测，复用上述同一关系查询、标准化、头和训练协议；不重跑已完成的Qwen/P0探测。
所有设计在新增分类结果返回前确定，不用test选择方法或探测参数；分类选定后只作固定test评价。

本轮实际新增1次完整分类训练（3轮、2208次更新），未新增预训练；补齐CLM冻结编码器的两个探测头。
两个新命令及固定test评价均退出0。CLM按valid选中第2轮、阈值0.45；P0为第2轮、0.23，原始Source为第3轮、0.31。
以下BCE为未缩放的逐函数分类BCE，由保存概率计算（本轮全部严格介于0和1，无裁剪）。
固定0.5阈值只作预定诊断，不替换原valid阈值。

| 方法 | 划分 | AUC | MCC | BCE↓ | Accuracy | Precision | Recall | F1 | MCC@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Source | valid | 0.8047 | 0.5116 | 0.7788 | 0.7517 | 0.8058 | 0.6676 | 0.7302 | 0.4539 |
| Source | test | 0.8142 | 0.4802 | 0.7527 | 0.7385 | 0.7716 | 0.6667 | 0.7153 | 0.4333 |
| CLM_Source | valid | 0.8168 | 0.5052 | 0.6791 | 0.7476 | 0.8079 | 0.6542 | 0.7230 | 0.4746 |
| CLM_Source | test | 0.8340 | 0.5026 | 0.6052 | 0.7503 | 0.7761 | 0.6933 | 0.7324 | 0.5047 |
| P0_Source | valid | 0.8213 | 0.5126 | 0.7057 | 0.7557 | 0.7743 | 0.7265 | 0.7497 | 0.4628 |
| P0_Source | test | 0.8369 | 0.4957 | 0.6252 | 0.7477 | 0.7377 | 0.7573 | 0.7474 | 0.4738 |

| 对照方向 | 划分 | 纠正旧错误 | 引入新错误 | 净纠正 | FN→TP / FP→TN / TP→FN / TN→FP |
|---|---|---:|---:|---:|---|
| Source -> CLM_Source | valid | 39 | 42 | -3 | 22 / 17 / 27 / 15 |
| Source -> CLM_Source | test | 53 | 44 | +9 | 36 / 17 / 26 / 18 |
| Source -> P0_Source | valid | 47 | 44 | +3 | 40 / 7 / 18 / 26 |
| Source -> P0_Source | test | 58 | 51 | +7 | 50 / 8 / 16 / 35 |
| CLM_Source -> P0_Source | valid | 32 | 26 | +6 | 30 / 2 / 3 / 23 |
| CLM_Source -> P0_Source | test | 29 | 31 | -2 | 27 / 2 / 3 / 28 |

归因结论：当前单seed、匹配协议的Source-only路径中，**主要排序收益由CLM适配提供，依赖监督没有稳定分类净收益**。
CLM相对Source的valid/test AUC分别增加0.0120/0.0198；P0相对CLM仅再增加0.0046/0.0029。
依赖监督的valid MCC增加0.0074，但test MCC下降0.0069，valid/test BCE分别变差0.0266/0.0201。
P0相对CLM在test少漏报24例、同时多误报26例，不能将Recall/F1增加写成全面提升。
固定0.5下P0的valid/test MCC也低于CLM，因此这一结论不只来自各自valid阈值。
这些是固定本轮模型的观察，不是多seed显著性结论，也不能外推为CLM贡献了固定比例的全部P0+C收益。
CLM对照分离了依赖监督的增量，但未分离CLM内容适配与多一轮源码训练带来的优化/初始化收益。

冻结探测严格复用原train/valid查询（逐行相同）：train 4658条/1000函数，valid 600条/114函数。
原始Qwen、CLM、P0的依赖AUC分别为0.7657、0.8072、0.9056；MCC为0.3910、0.4034、0.6598。
函数平均依赖BCE分别为0.9603、1.0476、0.4261；关系查询平均BCE分别为2.3061、1.8480、1.2179。
P0相对CLM的依赖能力增量明确存在于冻结编码器的新探测头，不只是原预训练头学会了任务。
但冻结源码分类AUC为0.7885、0.7736、0.7729，MCC为0.4628、0.4645、0.4407；
这种更强关系可读性没有形成更好的统一线性分类结果。探测依赖自身的拟合和读出，不能据此断言所有漏洞信息减少。

因此，现有结果支持“定义—使用关系可以被编码器学会”，**尚不支持把原依赖目标作为有效漏洞分类贡献继续堆叠**。
是否继续程序依赖研究，应围绕一个尚未回答的问题：这些可预测关系是否包含CLM之外、与函数漏洞标签相关且被下游使用的信息？
当前无法将瓶颈确定为分类器不会利用、依赖目标与漏洞不匹配，或标签/可见上下文不足；本轮不新增机制。

实现仅扩展现有`lm_pretrain_source`分支及冻结探测的`--lm-pretrain-dir`入口；复用原训练器、模型和评价函数。
新增路径已通过81项相关测试，包含小模型前向/反向、预训练LoRA加载、保存/加载、评价及test不调阈值；无失败。
完整指标、固定阈值对照及所有纠错sample_key保存在`results/cfg_clm_source_seed42/contribution.json`，
CLM训练的全部3轮记录保存在`lm_pretrain_source/best.training.jsonl`；第3轮退化也原样保留，未追加训练。
冻结探测结果在`results/cfg_clm_probe_seed42/lm.json`，原始Qwen/P0复用`cfg_origin_pair_probe_seed42`。

复现命令（现有结果不可覆盖；重跑需另选空输出目录）：
```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
$PY -B scripts/cfg.py run --dataset data/function_dataset.jsonl --graphs data/graphs/primevul_cfg.jsonl \
  --reference-run-dir results/cfg_abc_seed42 --pretrain-dir results/cfg_dep_pretrain_seed42 \
  --variants lm_pretrain_source --output-dir results/cfg_clm_source_seed42 --device cuda:0
$PY -B scripts/cfg.py diagnose-origin --phase probe --run-dir results/cfg_dep_cfg_windowfix_seed42 \
  --lm-pretrain-dir results/cfg_dep_pretrain_seed42 --output-dir results/cfg_clm_probe_seed42 --device cuda:1
$PY -B scripts/cfg.py eval --run-dir results/cfg_clm_source_seed42 \
  --variants lm_pretrain_source --split test --device cuda:0
```

### 图模块存在时的依赖监督增量（2026-10-09）

复用完整的 `cfg_dep_cfg_seed42/lm_pretrain_cfg` 与 `cfg_dep_cfg_windowfix_seed42/dep_pretrain_cfg`。
核对结果：阶段2根配置只有预训练目录不同，checkpoint及其阶段1来源与完成记录一致；
数据、划分、词表、Qwen、2048 tokens、seed=42、初始化策略、优化器及3轮预算相同。
阶段1均为5886个train函数、1轮、736次更新；原CLM归一化与修正后的CLM项等价（见上一节）。
两组均完成3轮、2208次阶段2更新，可训练参数均为10,474,370。CLM+C按valid选第1轮、阈值0.4，
P0+C选第2轮、阈值0.3。历史源码核查未发现这两个原C分支的训练目标/初始化被后续扩展改变。
本轮新增训练0次；融合valid/test结果直接复用，只补固定分支评价，未据test选择权重、阈值或方法。

分支评价调用既有`branch_logits`，每个函数单次源码编码，比较`source+graph`、`source`、`graph`。
各自沿用融合模型的原valid阈值，另报告预定0.5阈值；不针对分支重选阈值。
BCE由原logit计算，未缩放。四次完整评价均退出0，融合概率重放最大误差5.96e-8。

| 模型 | 划分 | AUC | MCC | BCE↓ | F1 | Accuracy | Precision | Recall | MCC@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLM+C | valid | 0.7868 | 0.4462 | 0.6123 | 0.7494 | 0.7166 | 0.6753 | 0.8418 | 0.4431 |
| CLM+C | test | 0.8210 | 0.4810 | 0.5594 | 0.7597 | 0.7306 | 0.6778 | 0.8640 | 0.5076 |
| P0+C | valid | 0.8208 | 0.4935 | 0.7636 | 0.7410 | 0.7463 | 0.7620 | 0.7212 | 0.4808 |
| P0+C | test | 0.8395 | 0.5273 | 0.6586 | 0.7638 | 0.7635 | 0.7519 | 0.7760 | 0.4992 |

P0+C相对CLM+C：valid/test AUC增加0.0341/0.0185，原阈值MCC增加0.0472/0.0463；
但BCE分别变差0.1513/0.0992，F1分别变化−0.0084/+0.0041。
valid纠正80例、引入58例（净+22）；test纠正75例、引入50例（净+25）。
valid少误报67例、多漏报45例；test少误报58例、多漏报33例。
因此有排序与原评价协议下的分类增量，但不是全面提升。尤其固定0.5时test MCC为
CLM+C 0.5076、P0+C 0.4992，净纠正为−1；MCC优势依赖既定valid阈值的工作点，AUC优势不依赖阈值。

| 模型 | 划分 | 分支 | AUC | MCC（融合阈值） | BCE↓ | F1 |
|---|---|---|---:|---:|---:|---:|
| CLM+C | valid | source_branch | 0.7875 | 0.4409 | 0.6111 | 0.7455 |
| CLM+C | valid | graph_branch | 0.4345 | 0.0000 | 0.7018 | 0.6697 |
| CLM+C | test | source_branch | 0.8215 | 0.4951 | 0.5570 | 0.7651 |
| CLM+C | test | graph_branch | 0.4212 | 0.0358 | 0.7038 | 0.6608 |
| P0+C | valid | source_branch | 0.8206 | 0.4771 | 0.7176 | 0.7335 |
| P0+C | valid | graph_branch | 0.6777 | 0.2958 | 0.6443 | 0.7099 |
| P0+C | test | source_branch | 0.8371 | 0.5121 | 0.6319 | 0.7584 |
| P0+C | test | graph_branch | 0.7135 | 0.3047 | 0.6280 | 0.7050 |

在固定checkpoint内加回图分支，CLM+C在valid纠正7例/引入6例，test纠正2例/引入8例；
P0+C在valid纠正11例/引入5例，test纠正11例/引入5例。
P0图分支的额外AUC只有valid +0.0002、test +0.0024；两组源码分支之间已经有
valid +0.0331、test +0.0156的AUC差距。图分支在P0中有可用分类信号，但不能据此把总增量归因于图连接。
此前固定P0+C valid的保度重连改变约95%的边，AUC几乎不变、判定最多改变1例；
这仍不支持“真实CFG连接是收益关键”。本轮没有新增test拓扑扰动。

源码/图分支是联合优化得到的残差组成，去掉一支不撤销另一支在训练中受到的影响；
图单支接近随机不代表图训练从未影响源码，图单支有AUC也不证明该信号来自正确程序依赖。
融合阈值未经图单支重新校准，因此其MCC/Recall只表示固定干预结果，AUC更适合观察排序信号。

结合上一轮Source-only：CLM的valid/test AUC为0.8168/0.8340，P0为0.8213/0.8369；
在独立训练对照中加C后，CLM变为0.7868/0.8210，P0变为0.8208/0.8395。
较大的“有C时依赖增量”部分体现为CLM+C训练结果低于CLM+Source，而不是P0+C大幅超越P0+Source。
这提示预训练与联合优化存在交互，但没有验证具体因果机制。CLM+C的第2/3轮均未超过第1轮，
全部历史轮次保留于汇总，不因此额外重训、改学习率或改选择规则。

结论：**本轮匹配的CLM+C没有达到P0+C的AUC及原阈值MCC，不能把图场景的收益全部归给CLM。**
这是定义—使用监督这一训练干预的条件性增量证据；不能推断任何CLM+C都无法实现同等成绩。
它也没有证明下游因正确使用定义—使用关系而获益：缺少等预算但破坏关系语义的监督对照，
仍无法区分语义学习与辅助优化/正则化效应。单seed、历史test已使用，不能宣称稳定泛化或独立确认。
现有结果支持保留“依赖监督与图联合训练的交互”作为待验证研究问题，
**不支持把已证实的程序依赖利用当作创新前提，更不足以据此继续堆叠复杂机制**。本轮止于诊断，不尝试其他优化。

实现仅将现有固定诊断扩展到CLM/P0+C的分支模式，修改`cfg_transfer_diagnostics.py`、
`cfg_experiment.py`及直接相关测试；模型和训练代码未在本轮修改。61项既有相关测试及1项新增参数边界测试通过，
四次真实checkpoint全量重放通过，`git diff --check`通过。
完整指标、每轮历史、全部纠错ID、分支指标与重放误差保存在
`results/cfg_clm_c_diagnostic_seed42/comparison.json`；逐函数源码/图logit和干预概率在其四个子目录。

复现分支诊断（使用尚不存在的输出目录；不必重训）：
```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for split in valid test; do
  $PY -B scripts/cfg.py diagnose-origin --phase branches --run-dir results/cfg_dep_cfg_seed42 \
    --variant lm_pretrain_cfg --split "$split" --output-dir "results/cfg_clm_c_diagnostic_seed42/clm_$split" --device cuda:0
  $PY -B scripts/cfg.py diagnose-origin --phase branches --run-dir results/cfg_dep_cfg_windowfix_seed42 \
    --variant dep_pretrain_cfg --split "$split" --output-dir "results/cfg_clm_c_diagnostic_seed42/p0_$split" --device cuda:1
done
```

### CGVL / OCA 首轮准入与最小模块检查（2026-10-09）

本轮停在监督准入不足，**没有得到CGVL分类结果**。没有新增Qwen预训练、局部判别训练或B/C/D正式分类训练，
也没有进行test方法评价。A可复用`cfg_clm_source_seed42/lm_pretrain_source`；后续基础编码器应复用
`cfg_dep_pretrain_seed42/lm_pretrain/last.pt`，本轮未加载该Qwen权重开展训练，不叠加P0。

历史核查：当前`semantics.py`输出的是CPG支持的关系/候选，不能直接充当安全判断真值。
历史`f95f032`包含语义融合及特征监督，`c5d12cc`转向机制文本输入。
后续修复对审计补充核实：历史`698c524`确实实现过`MechanismBottleneckHead`，
`1cef6f8`实现过patch-grounded机制提取，不能因当前入口已删除就认定历史上未实现。
`cfg_program.py` schema 8及`ComposedRelationHead`监督关系查询，既有独立valid覆盖不足。
OCA的预期区别是操作绑定的局部安全差异直接约束参与函数读出的表示，而不是概念分类瓶颈、
关系预测头或整函数嵌入的普通对比目标；**该区别目前仅落实于最小模块，尚未由分类实验支持**。

新增`vulnmechanism/cgvl.py`，复用Tree-sitter解析、`semantics._static_integer`、原数据读取及原tokenizer。
没有重新运行Joern或要求SMT/完整编译。现有Joern候选与schema 8关系不能作为安全真值，
因此此次准入只使用有明确证据的局部语法子集：

- Bounds：同一词法作用域的原生固定数组，常量下标、常量/字符写入；比较下标与对象容量。
- Pointer：局部明确地址/空指针的直接写入，或原生指针参数紧邻空值检查后的直接返回解引用。
  只证明该次解引用的空值属性，不声称对象范围、初始化和生命周期均安全。
- Lifetime：标准malloc/free契约下、无未知中间操作的局部状态；malloc可能返回空，
  未证实非空时不能将第二次free直接标为Violated。别名、未知调用和不支持的路径不作结论。

这些命题均以“执行到该操作”为条件，不宣称整函数安全、路径必然可达或函数标签发生变化。
所有变换记录`function_label=null`；原分类数据、标签、成员和划分未改变。Unknown仍保留操作及原函数，
不产生局部排序真值。模型输入白名单只有家族及源码token位置；不传state、reason、edit、修复信息或标签。
操作和上下文来自相同2048-token源码窗口，源码编码器仍保持因果可见性，函数读出发生在编码之后。

`OperationConstraintAlignment`共享操作/上下文投影和乘性交互，局部表示调制源码token汇聚；
函数分类器应读取汇聚后的源码向量，不把局部风险分数相加。`local_rank_loss`是
`mean(softplus(r_safe-r_unsafe))`。共享参数的对应关系轮换用于最小D干预检查，单操作函数无法通过该轮换破坏对应，
故未来正式D还必须报告实际破坏覆盖；本轮没有将该smoke当作完整消融。
**正式分类训练器尚未接入OCA和成对损失，B/C/D训练入口未完成；数据门槛失败后暂缓该部分，而非提供空壳训练命令。**

最终全量准入结果位于`data/cgvl_admission_seed42/coverage.json`；只分析5886个train和741个valid函数。
表中状态是原始代码上的局部判断，Violated=0不包括随后生成且重新核验的变换：

| 划分 | 属性 | Satisfied | Violated | Unknown | 有效风险差异对 | 独立函数 |
|---|---|---:|---:|---:|---:|---:|
| train | Bounds | 23 | 0 | 21446 | 23 | 14 |
| train | Pointer | 0 | 0 | 111702 | 0 | 0 |
| train | Lifetime | 0 | 0 | 733 | 0 | 0 |
| valid | Bounds | 5 | 0 | 2181 | 5 | 2 |
| valid | Pointer | 0 | 0 | 13624 | 0 | 0 |
| valid | Lifetime | 0 | 0 | 75 | 0 | 0 |

另有train/valid各23/5组保持合法范围的下标变化，以及23/5组函数后注释对照；
注释对照在输入窗口中可见的仅14/5组，剩余9组有`visible_input_changed=false`，不能作为扰动稳定性证据。
实际重命名对照为0：这些真实对象涉及调用，未展开宏时不能保证名称替换不影响宏语义，故未准入；
重命名只通过了明确的小模型/语法用例。84组保存的变换均逐条重新解析并重核原操作与变换后操作的局部命题，
且核对无继承的函数标签。真实rank对全部是Bounds，不能用它们宣称已覆盖三种安全属性。

本次预先设置的最低可行性条件为每类至少20个train、5个valid独立函数（并非充分统计功效保证）。
即使不采用该数字，Pointer/Lifetime为0、Bounds valid只有2个独立函数，也无法验证三类共享方法。
首个纯数值写入子集只有train 9函数/16对、valid 0；补充字符写入后成为14/23及2/5，
没有改变标签标准或使用test扩展规则。早期扫描保留在`data/cgvl_seed42`和`data/cgvl_local_seed42`，不用于最终覆盖表。

瓶颈属于**当前准入实现覆盖不足**，不是已证明数据中不存在相关知识：
train中9825个Bounds、54662个Pointer、363个Lifetime候选处于解析/预处理未知上下文；
其余未知主要超出当前常量范围、直接参数空值检查及无别名线性生命周期子集。
不能把大量指针/释放候选当成独立可靠监督。尤其不能用缺失nonnull检查、缺少free记录或CFG可达替代安全命题。
当前Bounds变换也可能被简单的常量/位置规律解决，尚未排除普通源码/普通成对学习的捷径，未声称OCA获益。

检查：30项相关测试通过，覆盖三类正反用例、未知与别名/重定义、影子变量、unevaluated表达式、
真实源码窗口、模型输入不含真值、BCE到OCA及输入表示的梯度、Pairwise Logistic Loss、对应关系改变、保存/加载。
新增`__alignof__`反例最初错误预期存在运行时操作，Tree-sitter实际不产生该候选；修正为断言无候选后复验通过。
小模型维度8、rank 4，不是Qwen实验。全量数据命令退出0，`git diff --check`通过。
没有局部判别AUC/MCC或新函数分类指标可报告；因此不能回答OCA是否超过普通源码模型和普通成对学习。

复现（输出目录须不存在）：

```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B scripts/cfg.py prepare-cgvl \
  --dataset data/function_dataset.jsonl --reference-run-dir results/cfg_abc_seed42 \
  --output-dir data/cgvl_admission_seed42
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B -m unittest tests.test_cgvl tests.test_cfg_program -q
```


### 真实修复对安全监督可用性诊断（2026-10-09，CPU-only）

结论：真实补丁可以支持比原有局部 AST 准入更丰富的**条件性局部安全变化**，但本次最小审计
尚未建立规模足够的已核验监督集。没有启动 GPU、OCA、局部判别或函数分类训练。
此前 CGVL 准入不是完整 SMT 验证：它只确认了 Bounds 的 14 个 train / 2 个 valid 独立函数，
Pointer、Lifetime 均为 0；不能将本次可恢复补丁数量直接作为该覆盖率的增量。

实现复用 `vulnmechanism/audit_semantics.py`、Tree-sitter 函数名恢复、既有 MegaVul 项目划分、
原 `function_dataset.jsonl` 成员和本地原始修复元信息。新入口仅生成清单和供核查的源码对，
不产生安全标签。原数据、标签和划分未修改。CleanVul CSV 的 `is_test` 是测试代码标记，
不是 train/valid 划分；只接纳与现有 CleanVul train/valid 成员精确匹配的对。

结果目录：`results/repair_supervision_checked_seed42/`。
`inventory.json` / `eligible.jsonl` 保存可用池；`cases.jsonl` 保存 24 对源码、diff 和原始出处；
`reviews.jsonl` 保存逐例判断、原文行号、假设与未解决条件；`review_summary.json` 保存统计。
这些是助手逐例源码审阅结果，尚无独立专家复核，不是自动生成的训练真值。
此前失败目录 `results/repair_supervision_audit_seed42` 是不完整产物，不参与以下统计。

| 来源 | 可用池 train | 可用池 valid |
|---|---:|---:|
| PrimeVul | 3,588 | 414 |
| MegaVul | 9,303 | 633 |
| CleanVul | 519 | 302 |
| 合计 | 13,410 | 1,349 |

上述为源码去重、原划分冲突及当前 heldout 源码重叠排除后的 **14,759 对候选**，
不是已证实的漏洞修复根因对。对应 1,200 个项目标识；跨数据集项目别名尚未统一，
不能将其宣称为 1,200 个严格独立项目。
PrimeVul 原配对文件 train 的 3,789 对中，3,665 对通过元信息和函数名恢复，
107 对函数身份未恢复、17 对相邻记录的项目或提交不匹配；valid 的 480 对中恢复 462 对。
MegaVul 既有 train/valid 池分别为 10,648/721 对；CleanVul 原 CSV 有 5,583 对去重的
C/C++ 非空变化对，其中 4,282 对没有唯一的现有 train/valid 成员归属。
这些库存检查不能替代函数语义对齐，抽样中确实发现了一对 CleanVul 不同函数的错误配对。
可用池覆盖当前 PrimeVul 2,369 个 train / 302 个 valid 函数成员；这是**有补丁可查**的覆盖，
不是可靠安全监督覆盖，也不授权将其他项目样本加入原分类训练。

每个来源、每个 split 随机抽取 4 个不同项目，再各抽取 1 对，seed=42，共 24 对、24 个项目标识。
这是项目分层的最小可行性审阅，**不能将比例外推到全部 14,759 对**。

| 核查项 | 对数 / 24 | 比例 |
|---|---:|---:|
| 定位具体敏感操作（包含可识别 API 操作） | 17 | 70.8% |
| 识别源码中的相关约束变化 | 20 | 83.3% |
| 确认有明确假设的局部安全变化 | 5 | 20.8% |
| 有程序关联但安全影响未确认 | 16 | 66.7% |
| 回归测试 / 格式修改 / 配对错误 | 3 | 12.5% |

5 个确认案例为 5 个不同函数、3 train / 2 valid：Bounds 2、Lifetime 1、Type 1、Transport 1。
原 OCA 三类范围内只有 Bounds 的 1 train / 1 valid、Lifetime 的 1 train / 0 valid，Pointer 为 0。
候选还包括 Bounds 4、Pointer 4、Lifetime 1、Type 2、Filesystem 2、Injection 1、Arithmetic 1、Access 1。
未确认案例不能作为安全负例。确认也只针对指定操作、条件和执行路径，不赋予整函数
Satisfied/Violated 标签，更不假定修复后的整个函数安全。

可直接复核的例子（编号对应 `review_index`）：

- **1 / OpenSC**：`kinfo[8]` 改成 `[9]`。循环覆盖 7..15，共 9 次，成功路径逐次递增
  `num_keyinfo`；第九次写索引 8 超出旧容量、新容量可容纳。目标写、对象容量与索引路径直接对应。
- **10 / Linux tty**：底层分配失败时删除 `kfree(p1)`。`unicode=128` 时外层索引为 0，
  被复用的 `n` 为 2；旧代码释放槽 0 指向的对象却清空槽 2，留下悬空引用。
  新代码避免该悬空引用的产生；这里没有宣称已证明后续 UAF 执行。
- **5 / bzip2**：新增 `nSelectors <= BZ_MAX_SELECTORS`，绑定随后 `selectorMtf[i]` 的写入。
  数组容量来自修复上下文明示的声明，判断以该声明和错误退出宏契约为前提。
- **0 / ext-http**：对当前 `ptr` 加入数组类型检查，再使用其数组成员；依赖 PHP zval 宏契约。
- **12 / Shotcut**：具体升级检查请求由 HTTP 改成 HTTPS，仅确认传输条件变化，
  不证明证书策略或完整升级认证，也不计入首轮三类 OCA 覆盖。
- **17 / miniupnp（未确认）**：新 guard 检查 `ext_port`，局部却声明并使用 `rem_port`；
  不能因为出现判空就认定修复有效。没有擅自修改这一原始样本。
- **6 / unicorn、8 / KernelSU、23 / NumPy（未知）**：分别是回归测试调用、纯格式修改、
  before/after 不同函数；都不能按 vulnerable/fixed 两侧直接生成安全标签。

16 个候选中，12 个主要缺少被调函数契约或实际使用点，3 个需要跨函数状态/版本语义对齐，
1 个存在 guard 参数对应及标识符有效性问题。CPG 可以帮助定位关联，不能自行补足这些安全事实。
`visibility.json` 另保存本地 Qwen tokenizer 的 2048-source-token 可见性检查；不扩大输入、不切换切片。
5 个确认案例中 4 个的所需源码片段可见，bzip2 的关键写入超出范围。
因此首轮三类属性中，同时具备局部证据和原输入可见性的样本只有 2 个 train / 0 个 valid。
当前确认样本只有 OpenSC 映射到原 PrimeVul train 成员，没有确认样本映射到其 valid。
因此本次没有证明原 14/2 个准入函数的净增量。

历史方法核查：

- Patch/Slice：`1cef6f8` 的 patch-grounded 提取从改动的 CDG/DDG、操作符及调用推导机制候选；
  `fe491de` 的 `_slice_source` 按端点和变量赋值关系筛选上下文。它们提供候选/输入选择，
  不提供独立的局部安全影响核验。不能仅把相同补丁切片换名为新监督。
- Mechanism Bottleneck：`698c524` 的 `MechanismBottleneckHead` 将机制概率作为漏洞头输入；
  `1cef6f8` 的 `_side_record` 直接按 fixed 侧生成机制 target。既有
  `data/mechanism_smoke_quality.json` 中，50/50 无变化对照仍有组件，100/100 行可从 target
  恢复侧标签。这不能证明机制语义被学会。
- Composition：现有 `cfg_composition_pretrain_seed42/composition_pretrain/complete.json`
  记录 5,886 个训练函数中只有 6 个有效关系函数、9 个查询；这是关系组合监督，不是安全影响真值。

本次与历史方法的实质区别仅在于**核验证据标准**：明确操作、改动约束、路径/对象对应、
局部影响及其假设，拒绝由补丁侧或 CPG 差分直接推导标签。若后续仍把那 16 个候选当成真值，
就没有突破历史问题。当前证据支持“存在值得核验的真实局部安全变化”，不支持“已得到足够训练数据”，
也没有任何漏洞检测增益结论。停止于可用性诊断，不开发新的训练流程。

边界披露：早期临时 schema 查看误打开并显示了一个 PrimeVul test 记录的元信息和源码前缀；
它未用于监督规则、抽样或案例判断。正式入口明确只打开 train/valid paired 文件；
现有 heldout 的源码身份只用于重叠排除。本次没有 test 指标、训练或基于 test 的方案选择。

复现库存与相同抽样（输出目录必须不存在；逐例判断保存在 reviews 中，命令不自动重造判断）：

```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
OMP_NUM_THREADS=2 $PY -B -m vulnmechanism.audit_semantics \
  --repair-data-root /home/PublicData/PHY-data/vul_detect/data \
  --repair-output-dir results/repair_supervision_checked_seed42
$PY -B -m unittest tests.test_audit_semantics -q
```

实际检查：库存命令退出 0；相关 4 项测试通过，覆盖原机制审计、配对身份异常、
heldout 排除、原生 test 文件不被打开以及不自动生成安全真值。两项数值例证检查通过
（9 次迭代/容量、unicode 的两级索引）；它们是局部算术检查，不是原项目编译或漏洞复现。

### 真实源码安全差异构造实验（2026-10-09，未训练）

**结论：当前最小可核验子集不足以启动 EGCL。** 原 PrimeVul train 5,886 / valid 741 个函数中，
只得到 1 个 train 独立来源、0 个 valid 来源；并未证明真实源码衍生样本有足够规模或跨结构多样性。
这是否定当前构造子集的可用规模，不是证明其余真实函数不可能被可靠构造。
没有 GPU、局部判别或分类训练，没有修改原函数标签、数据划分或评价口径。

核查了 `cgvl.py` 的旧准入、`cfg_program.py` 的绑定/版本/路径约束、`semantics.py` 的常量及
关系候选处理，并复核上一节真实补丁审计。旧 CGVL 仅有 Bounds 的 14 train / 2 valid 个来源：
字面量索引、直接常量写入的限定非常窄；调用、复杂语句、预处理及别名又会使指针状态失效。
此外，旧结果主要是“若执行到目标操作”的条件性证据，不能自动提供从函数入口可达的 Unsafe 见证。
CPG 关系可以定位程序关联，但不能填补被调函数契约、对象所有权或路径可行性的空缺。

先做可构造性扫描：1,957 个函数存在解析/预处理边界；可解析部分有 718 个函数包含顶层 for，
492 个以 if 开始，其中 332 个同时有较短条件与指针操作候选。这些只是候选结构，不能当作监督。
正式实现仍在 `cgvl.py`，通过已有 `prepare-cgvl --construct-real-source` 入口运行，保留原准入对照。
本轮只实现两个可重新解析和核验的子集，没有增加验证器框架：

- **Bounds**：入口处简单原生类型声明之后的局部定长数组循环，`int i=0; i<N; i++`，
  循环体仅包含目标数组的 0/1 写入，排除浮点到整数的额外转换风险。推导索引范围并给出首次越界迭代；交叉改变容量与上界，
  使同一循环上界在不同容量下分别安全/不安全。真实数据没有满足完整条件的样本。
- **Pointer/nullness**：入口判空后直接返回/赋值解引用，或判空提前返回后的立即解引用。
  翻转条件，并重新证明空指针输入是否到达该操作。Safe 仅指排除空指针解引用；
  非空输入仍须满足原指针所指对象有效且已初始化的契约，不证明对象边界、生命周期或整函数安全。
- **Lifetime**：没有找到当前分析能够确认入口路径与堆对象所有权的可用证明，不把 `free(p)`
  或释放函数名当作所有权依据，也没有注入新的 malloc/free 小程序来冒充真实上下文。

结果：`data/real_source_construction_seed42/coverage.json`、`samples.jsonl`、`review.json`。
样本保存原始源码、修改源码、UTF-8 字节编辑范围、目标操作、证明范围、Unsafe 触发条件、
项目、原提交与函数分组。所有变换继承同一个来源记录的 split；跨 train/valid 的相同源码或
项目/提交/文件/函数组合被排除，而不是重新分配原样本。只打开原生 train/valid 元信息文件。

| 属性 | train 候选函数 / 操作 | valid 候选函数 / 操作 | 成功来源 train / valid |
|---|---:|---:|---:|
| Bounds | 2,059 / 21,469 | 258 / 2,186 | 0 / 0 |
| Pointer | 4,620 / 111,702 | 589 / 13,624 | 1 / 0 |
| Lifetime | 242 / 733 | 27 / 75 | 0 / 0 |

这里的候选操作只按语法定位，同一函数可能属于多个属性，不能相加为独立函数数。
未纳入监督的 Unknown 来源分别为 Bounds 2,059/258、Pointer 4,619/589、Lifetime 242/27。
零产出的类别不代表所有操作不安全，也不代表其安全性已被否定。

主要失败项（train / valid，函数数；每类内部互斥）：

| 属性 | 解析/预处理 | 未认证 C++ 语义 | 不满足所选路径/状态子集 |
|---|---:|---:|---:|
| Bounds | 787 / 98 | 267 / 55 | 1,005 / 105 |
| Pointer | 1,445 / 184 | 771 / 105 | 2,403 / 300 |
| Lifetime | 111 / 12 | 12 / 0 | 119 / 15 |

原源码与全部修改版本的目标操作和证明上下文均检查了原 tokenizer 的 2048-source-token 范围。
唯一通过证明的来源也通过可见性检查，无样本因窗口被剔除；本轮主要瓶颈是核验子集而非截断。

实际唯一来源为 train `primevul:257106`，项目 GPAC，函数 `gf_bs_get_cookie`：

```c
u64 gf_bs_get_cookie(GF_BitStream *bs)
{
    if (!bs) return 0;
    return bs->cookie;
}
```

Safe 版本将检查等价写为 `bs == 0`；Unsafe 改为 `bs != 0`，输入 `bs=NULL` 时从入口经过检查
到达同一个 `bs->cookie`，发生空指针成员访问。另在 Safe 的目标语句前插入可见注释作为语义保持对照。
源码其余字节不变，3 个版本都已按保存的编辑记录重放，并核对目标操作对应。
最终只有 **1 个 Safe/Unsafe 对、1 个语义保持对**；版本数为 Safe 2、Unsafe 1（包含对照版本）。
没有改变程序行为而保持安全属性的合格对，也没有真实交叉条件样本。不能把同源变体当成多个独立函数。

`runtime_check/` 保存实际函数体的最小类型契约夹具和 `result.json`：3 个版本均编译成功，
共运行 6 个 live/null 输入检查。Unsafe + NULL 被 UBSan 报告为 `member access within null pointer`，
退出 1，符合预期；其他 5 次退出 0。夹具中的 `GF_BitStream` 是仅含所需字段的声明，
**不是 GPAC 完整工程定义或构建**。运行证实这个见证；Safe 依据入口控制流证明，不能由未崩溃倒推。
可复现夹具检查示例：

```bash
cc -std=c11 -O0 -fsanitize=undefined -fno-sanitize-recover=all \
  data/real_source_construction_seed42/runtime_check/unsafe.c -o /tmp/real_source_unsafe
/tmp/real_source_unsafe null  # 预期 UBSan 报错、退出 1
```

新来源与旧 CGVL 14/2 来源不重合，但两套证明范围不同：本轮要求入口见证、Pointer 仅核验 nullness，
因此不能宣称把旧数据扩成同标准的 15/2 个训练来源。当前实际验收数是 **1/0**，没有覆盖率改善。
只有 1 个项目、1 种提前返回模板、1 个条件，valid 没有独立案例；
与完全合成代码相比，本轮只能确认来源真实，**不能证明结构或语义多样性更好**。
模型完全可能利用 guard 极性或改动痕迹取巧；不能据此验证 EGCL 的跨结构安全知识学习。
不继续堆叠构造规则，不接入分类器。

复现数据构造（输出目录必须不存在）：

```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B scripts/cfg.py prepare-cgvl \
  --construct-real-source --dataset data/function_dataset.jsonl \
  --reference-run-dir results/cfg_abc_seed42 \
  --output-dir data/real_source_construction_seed42
OMP_NUM_THREADS=2 $PY -B -m unittest tests.test_cgvl tests.test_cfg_program -q
```

实际检查：全量构造命令退出 0；相关 34 项单元测试通过，覆盖原 CGVL/Composition 回归、
入口可达性、空指针见证、循环容量交叉、拒绝别名/副作用/未求值上下文及 UTF-8 编辑对应。
实际样本编辑重放与目标对应检查通过；UBSan 的预期失败已单独记录，不计作测试失败。

最终对 6,627 个 train/valid 函数重新扫描，确认最终实现产物与已保存的 1 个来源、3 个版本完全一致
（`recheck.json`）；最后的相关 10 项测试复验通过，`git diff --check` 通过。

### EGCL 合成数据与冻结 CLM 对照（2026-10-09）

本轮从明确的有限语义生成 C 程序，不把这些标签当作真实函数标签。`egcl.py` 组合
内存状态建立、危险状态更新、恢复赋值、目标操作与控制形式；复用 `cgvl.local_rank_loss`、
原 Qwen 均值读出/线性分类头、原 tokenizer、指标及 valid 阈值选择。
每个程序枚举输入 x=0..3，所有内存控制取决于低两位。Bounds 覆盖局部数组单元素读/写、
容量与索引及路径约束，尚未覆盖可变长度拷贝；Pointer 覆盖本地有效地址、置空、恢复和解引用；
Lifetime 覆盖成功 malloc、free、指向本地对象的重新赋值及后续使用。分配失败直接退出。
这是受限的可组合状态系统，不是通用 C 验证器，也没有完整工程上下文。

生成前固定结构家族：train 为 bit 条件下的 if / early-return；valid 为 switch、
新增条件组合及 xor 条件组合。原样、重命名、无关语句版本继承同一家族划分，源码跨划分交集为 0。
另有保持 Safe 属性但改变条件的对照。未见容量为 5/9；其中数字 5 曾作为训练中的索引出现，
因此这个场景不能全部称为“未见数值 token”，容量 9 才包含新的数值 token。

| 属性 | train 基础差异对 | valid 基础差异对 |
|---|---:|---:|
| Bounds | 64 | 128 |
| Pointer | 16 | 24 |
| Lifetime | 16 | 24 |

合计 96/176 个基础配对，含语义保持版本的唯一源码为 train 576、valid 1,056。
不是 1,632 个独立真实函数；只有有限的控制结构家族。标签逐类平衡，Unknown=0 是生成子集闭合的结果，
不表示解决了真实代码中的 Unknown。C11 编译、ASan/UBSan 对全部源码各执行四个输入，
6,528 次均与穷举一致，816 次预期违规全部触发，失败 0。Safe 依据有限状态分析，而非仅依据未报错。
原始 token 计数 LogisticRegression 的 valid AUC=0.5、MCC=0；主要常量和 malloc/free 计数按标签平衡。
这只排除一个简单词法捷径，不能证明不存在更复杂的生成器线索。

数据及证据位于 `data/egcl_synthetic_seed42/`：`samples.jsonl`、`pairs.jsonl`、`controls.jsonl`、
`coverage.json`、`execution.c`、`execution.jsonl`、`execution_summary.json`、`lexical_diagnostic.json`。
源码中不含标签、状态真值或配对说明，模型只读取函数源码。

冻结实验实际使用 `cfg_dep_pretrain_seed42/lm_pretrain/last.pt`，未叠加 P0。
同一个原始线性分类头初始化，训练集合均为 576 个源码成员；同样的 288 对（含语义保持版本）、
每批 16 对、20 epochs、AdamW lr=0.001、weight decay=0.01、seed=42，沿用已有冻结 probe 的
train-only mean/std 标准化。A 用 BCE，B 用 Pairwise Logistic Ranking Loss；只有分类头更新。
原源码预算 2048，实际最长 205 tokens；原始均值池化包括 prefix/EOS，未修改读出。
checkpoint/阈值均按 valid MCC/F1/Accuracy/AUC 选取，不使用 test。

| 合成 valid（1,056 个版本） | BCE | EGCL |
|---|---:|---:|
| AUC | 0.712207 | 0.612028 |
| MCC | 0.374147 | 0.182024 |
| Accuracy | 0.685606 | 0.589015 |
| Precision | 0.712121 | 0.573668 |
| Recall | 0.623106 | 0.693182 |
| F1 | 0.664646 | 0.627787 |
| 未缩放 BCE | 0.669870 | 1.931300 |
| 基础成对排序准确率（176 对） | 0.948864 | 1.000000 |
| 选定 epoch / 阈值 | 20 / 0.45 | 13 / 0.05 |

两者 train 基础配对排序均为 100%，train 分类 Accuracy 为 94.97% / 59.55%。
valid 的 switch、未见容量和新增条件组合，BCE/EGCL 排序均为 100%；xor 组合从 BCE 的 81.25%
提高至 EGCL 的 100%。按 Bounds/Pointer/Lifetime 分组，排序分别从 96.09%/91.67%/91.67% 到 100%。
这支持这个有限系统中的**相对排序**能力，不支持稳定的绝对安全判定。
EGCL 相对 BCE 纠正 182 个版本的旧错误，却引入 284 个新错误；只统计原样 352 个源码时为 79/114。

重命名造成的分类翻转率 BCE/EGCL 为 45.17%/43.18%，无关语句为 19.60%/41.76%。
EGCL 在重命名和无关语句后的配对排序仍为 100%，但对应配对平均 logit 中心的绝对移动为
6.65/5.81，大于其平均组内 margin 2.23/2.22。排序损失对一对样本共同增加同一 logit 偏移不敏感；
这些观测与这种欠约束一致，但不证明它是唯一原因。不能用排序 100% 掩盖分类和稳定性退步。
完整历史、分属性/场景指标、逐例预测和 head 权重在 `results/egcl_frozen_seed42/`。

在读取结果前设定的排序迁移门槛（相对 BCE 至少 +5 个百分点，所有属性/场景至少 60%）通过，
但这个门槛后来被证明不足：更强的源码结构捷径诊断推翻了数据的机制准入，见下文。
原 CLM baseline 固定 valid 复算与旧预测最大差为 0：AUC=0.816755、MCC=0.505246、
F1=0.722963、BCE=0.679093。产物为 `results/egcl_transfer_baseline_seed42/`。

迁移复用 `model.train_model`：原函数仍用 BCE，在每个原 optimizer update 加一个合成 train 对，
B 用合成 BCE，C 用排序 loss，权重均固定为 1。不使用合成 valid 更新参数，不继承冻结实验的 head。
同一 CLM 初始化、原 5,886/741 成员、3 epochs、batch=1、accumulation=8、lr=2e-4、seed=42，
按原 PrimeVul valid 规则选模。B/C 有相同额外合成计算量；相对原 baseline 增加的数据与计算必须单独说明，
不能将任何提升全部归因于“学到了执行语义”。这两个作业已主动中断，退出码均为 130，
各自 `stopped.json` 保存原因及最后进度；没有 complete.json，也没有可报告的新 PrimeVul 分类指标。

```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
OMP_NUM_THREADS=2 $PY -B scripts/cfg.py egcl --phase prepare --output-dir data/egcl_synthetic_seed42
OMP_NUM_THREADS=2 $PY -B scripts/cfg.py egcl --phase frozen \
  --data-dir data/egcl_synthetic_seed42 --output-dir results/egcl_frozen_seed42 --device cuda:0
OMP_NUM_THREADS=2 $PY -B -m unittest tests.test_egcl tests.test_graph_training tests.test_cgvl -q
```

输出目录必须不存在；本环境 GPU 访问需要在沙箱外执行。23 项相关测试通过，包含结构隔离、
状态恢复/交叉条件、标签和控制对应、排序梯度、原训练器反向/保存/加载/评价以及拒绝 synthetic valid。
真实 Qwen 冻结实验已完成；小模型测试只检验实现，不作为性能结果。


**最终准入结论：当前生成器不合格，不支持“EGCL 学到了内存安全组合语义”的结论。**
`structural_shortcut.json` 给出一个不读取模型、不使用生成器隐藏状态的源码规则：找到比较低位
表达式的两个条件，预测“两个常量相等即 Unsafe”。它忽略数组容量、内存操作、指针/所有权和
恢复条件，却在 train 576/576、valid 1,056/1,056 上全部正确。
原因是生成器只保留翻转 target 条件能够得到一安全一不安全的配对，丢弃了恢复条件导致两侧均安全的
组合。新增 conjunction/xor 语法之后，保留样本的标签仍等价于同一个条件相等关系。
因此 unigram 的随机水平、源码结构不重叠、执行标签正确和排序 100% 都不足以排除这个捷径。
这没有证明 Qwen 必然使用了该规则，但已经证明本数据不能区分该规则与真正的目标机制。

这个检查应在迁移启动前完成。本轮在排序门槛通过后已经启动两个迁移作业，发现反例后只中断了
输出目录严格匹配的两个自有进程，没有触及其他作业；保留全部历史，不报告未完成的分类效果。
当前正式入口要求机制准入为 True 才允许 transfer，当前数据明确为 False，会在模型加载前拒绝。
没有重划数据、删除反例、追加新模型或调整超参数挽救结果，也没有使用 test。
本轮完成的是可靠的有限执行标签、真实 Qwen 冻结头对照以及发现构造偏差的负结果；
真实 PrimeVul 迁移价值**尚未建立**，不能写成有效或无效的分类结论。

最终还实际验证了 transfer 对当前数据在模型加载前拒绝，且不创建输出目录；`git diff --check` 通过。

### EGCL 生成与目标修正：强制反捷径准入（2026-10-09，未进行 EGCL 性能训练）

旧生成器的确切问题是 `sorted(states) != [0, 1]` 时丢弃整个组合：先筛选标签相反的配对，
再把它们当作分类数据。这使保留下来的标签等价于两个条件常量是否相等。旧的执行核验正确，
但不能排除这个 100% 的非内存语义规则。旧结果和 checkpoint 保留；本节没有复跑纯 Ranking。

正式 `egcl.py` 改成先枚举程序状态、生成全部成员，再依据核验标签建立配对。初始化、两次
受条件控制的对象赋值、赋值顺序及读/写分别组合；Bounds 的候选索引为 `n-1` 和 `n+offset`，
Pointer 为本地对象/空指针，Lifetime 为本地对象/已释放堆对象。Lifetime 只释放一次，先建立别名，
释放后只有目标操作可能使用失效对象。分配失败提前返回；其余初始化和清理不额外引入漏洞。
枚举低两位的四种输入组合，Unsafe 保存具体输入，Safe 依赖这个受限语义的完整枚举。
这仍是很窄的对象状态系统，不是通用 C 安全验证。

| 属性 | train 源码版本 | valid 源码版本 | train 相反标签对 | valid 相反标签对 |
|---|---:|---:|---:|---:|
| Bounds | 2,880 | 8,640 | 576 | 1,728 |
| Pointer | 768 | 1,536 | 144 | 288 |
| Lifetime | 768 | 1,536 | 144 | 288 |
| 合计 | 4,416 | 11,712 | 864 | 2,304 |

上述计数包含原样、重命名和无关语句三个版本；基础源码实例为 1,472 / 3,904，基础相反标签对
为 288 / 768，不能当作独立真实函数数量。另保留 train/valid Safe–Safe 192/1,632 对、
Unsafe–Unsafe 1,152/1,920 对，以及全部语义保持对照，没有按标签删除组合。每个训练成员
恰好属于一个训练组，A/B 使用完全相同的成员及批次。真实 PrimeVul 数据、标签和划分没有改动。

**数据仍未通过准入，没有启动 GPU 性能训练或 PrimeVul 迁移。**
`data/egcl_crossed_seed42/` 保存全部源码、spec、标签、见证、配对及核验记录。反捷径结果：

- 旧条件相等规则准确率：train 43.48%，valid 50.00%，原来的确定性捷径已消除。
- 词法 unigram/bigram LogisticRegression（固定 C=1，不搜索）valid AUC=0.792654；
  Bounds=0.785273，Pointer=0.812500，Lifetime=0.812500。
- 固定浅层词法随机森林 valid AUC=0.765302；Pointer=0.801310。三级结构统计树整体 AUC=0.687500。
  检查在任一属性达到 0.80 都阻止准入，不能用整体平均掩盖属性内的高分。
- Alpha/数值规范化后的源码交集为 0，0.95 相似度筛查没有发现近重复；但两侧只有各一个
  控制/谓词/组合规则家族（if+bit、switch+xor+conjunction）。8 种“初始—第一次—第二次赋值”
  状态组合全部在 train 出现，没有未见对象状态组合。
- 忽略控制形式、谓词形式和新增 conjunction 后，valid 的 8,832/11,712 个版本与 train 共享
  核心操作、对象选择、赋值顺序和条件字面量参数。这里的参数共享不意味着两个程序语义完全相同，
  但不能作为严格独立程序实例的证据。这个保守的生成来源检查与源码近重复检查分开报告。

聚合指标的第一层筛查曾通过；加入分属性分类器和生成来源检查后否决。
`structural_shortcut.json` 保留聚合筛查记录并给出最终不准入结论，`extended_shortcut.json`
保存独立复查结果。0.80 是本次预先固定的筛查阈值，不是语义学习与词法学习之间的理论边界。
词法高分本身也不是“不可能学到语义”的证明；它说明目前不能把模型分数归因于正确的操作—约束对应。
未通过删除困难成员、挑选子集或放宽阈值使数据过关。

`review.json` 保存三类真实生成案例：完整词袋相同，只调换两次条件赋值顺序，标签便发生变化。
Bounds 的样本 4440/4446、Pointer 的 10200/10206、Lifetime 的 11736/11742，在输入 x=0 时
分别从范围内/有效本地对象变成越界/空指针/已释放对象。这说明数据中存在关系敏感的实例，
但不能抵消整体准入失败。`visibility.json` 确认全部 16,128 个版本无截断，源码最长 191 tokens，
沿用原前缀和 EOS 后最长 206 tokens。

训练实现只保留 `bce` 和 `bce_rank`：后者为 BCE + 权重 1 的 Pairwise Logistic Ranking，
只对相反标签组追加排序项。同标签成员仍参与 BCE 和稳定性评价，不被删除、不施加排序约束。
`model.py` 的公共损失与训练器同步修正，`cfg_experiment.py` 使用同一正式入口；没有新增模型或辅助头。
冻结入口在设备解析、编码器加载之前检查准入。旧的仅看排序增量的迁移门槛已移除，必须检查绝对
分类和语义保持稳定性；本轮因数据门槛未过，没有进入这些性能比较，因此没有新的 BCE/BCE+Ranking
AUC、MCC、F1、BCE，也没有 EGCL 的真实漏洞检测收益可报告。

复现数据构造与核验（输出目录必须不存在），以及相关回归：

```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B scripts/cfg.py egcl --phase prepare \
  --output-dir data/egcl_crossed_reproduction_seed42
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B -m unittest \
  tests.test_egcl tests.test_graph_training tests.test_cgvl -q
```

下面命令用于验证拒绝路径，应在加载模型前退出，不创建实验结果目录：

```bash
OMP_NUM_THREADS=2 $PY -B scripts/cfg.py egcl --phase frozen \
  --data-dir data/egcl_crossed_seed42 --output-dir results/egcl_crossed_frozen_seed42 --device cuda:0
```

实际核验：C11 + ASan/UBSan 完成 64,512 次执行，全部与枚举结果一致；10,752 次预期违规
全部触发，失败 0（`execution_summary.json`）。28 项相关单元测试全部通过，覆盖完整训练成员、
同词袋顺序反例、BCE 不被替代、同标签不排序、分属性准入反例、模型加载前拒绝，以及小模型/
冻结分类头的反向、保存、加载和评价。当前生成器重放的 samples/pairs/controls 与保存结果逐项相等；
正式冻结入口实测退出码 1，原因为数据准入失败，没有创建模型结果目录。`git diff --check` 通过。

### 真实操作—上下文判别可行性（2026-10-09，未启动冻结探测）

本轮完成原 PrimeVul train 5,886 / valid 741 个函数的 CPU 全量审计。没有改动原始数据、
函数标签或划分，没有生成局部 Safe/Unsafe 标签，也没有使用 test 设计规则或评价。
**现有证据不能回答正确绑定是否优于普通源码表示；本轮受控验证成员不足，且交换未必破坏关键约束。**
因此没有 A/B/C/D 的 AUC、MCC、F1、BCE 或纠错成绩，不能用已有 CLM 分类结果替代。
这是当前对照设计的可行性负结果，不是“程序关系无用”的性能结论。

复用 `semantics.py` 的敏感 API 范围、`syntax.py` 的 AST、`cfg_dependency.lexical_bindings`
的声明身份以及 `cfg_alignment` 的精确 CPG 坐标检查；正式入口扩展在 `audit_semantics.py`。
源码节点直接定位数组访问、解引用、字段访问和相关 API 的操作数，避免把父赋值和子调用重复
识别为同一个 API 操作。声明、操作数来源声明、先前源码位置的直接赋值/更新、相关包围分支及
更新所在分支分别保存。它们是语法/作用域关联，不是别名解析、当前到达值、路径可行性或安全证明。
早退 guard 不在当前 AST 包围上下文序列中，原生 CDG/DDG 单独保留，没有伪称完整控制分析。

`operations.jsonl` 保存每个原函数、原标签、项目/提交/文件来源、所有候选、准确字符位置、
对象声明身份、关联片段及 Unknown 原因。只用原源码前 2048 tokens 判断操作和上下文可见性。
C++ 重载、成员/全局对象、宏、被调函数契约、容量与所有权等边界不能由这个绑定结果消除。
原生边另标记是否有相同作用域声明端点；操作关联范围内保存的原生边没有额外边属性，不能
假定其中存在可直接读取的依赖变量或对象状态。以下计数不把关系实例当作独立函数。

| 属性 | 有候选的函数 train / valid | 有已绑定对象的函数 train / valid | 属性家族层面正负匹配对 train / valid | 严格交换对照匹配对 train / valid |
|---|---:|---:|---:|---:|
| Bounds | 2,445 / 309 | 845 / 111 | 347 / 18 | 7 / 1 |
| Pointer | 4,620 / 589 | 2,794 / 367 | 818 / 58 | 119 / 5 |
| Lifetime | 393 / 36 | 205 / 16 | 50 / 2 | 0 / 0 |

跨属性函数重叠，不能直接相加。总体有候选的独立函数为 train 4,941 / valid 633；
至少有一个对象绑定的独立函数为 2,993 / 400。valid 候选函数的正/负分布为 Bounds 209/100、
Pointer 325/264、Lifetime 19/17。操作本身显然同时存在于两类函数中，但覆盖率不是判别增量。

共提取 154,568 个语法候选，其中 51,338 个可以确认局部声明关联（33.2%，不等于安全关联完整率）。
其余候选原因：解析/多函数边界 48,899，预处理未知 25,470，复合对象/别名 20,487，
不求值或仅取地址上下文 4,934，成员/全局或声明未解析 3,440。以上是操作数，不是函数数。
原输入最长有 218,030 tokens；tokenizer 的长度警告发生在长度统计时，没有将它送入模型。
所有后续可见性筛选仍严格使用原 2048-source-token 范围。

对照预先固定为：每个函数选第一组同操作类型、同 parameter/local 来源、上下文及局部语句
均非相同的操作；保留同一函数的操作与上下文集合，只交换对应。允许同一对象的不同使用位置，
不会把重复的相同片段当作有效交换。正负函数再按项目、操作类型、来源和 log2 长度/候选数量分组
匹配，不改变原始成员，只产生显式诊断子集，不根据 valid 分数调整规则。

可形成交换的函数为 train 正/负 756/370、valid 正/负 99/46，说明抽取/筛选成功率本身也与
标签相关，不能把这个选择效应记作程序语义收益。最终匹配 train 126 对（252 函数、18 项目）、
valid 6 对（12 函数、4 项目），valid 只有 Pointer 5 对、Bounds 1 对。未达到模型运行前固定的
最小 train 100 / valid 30 对要求；该要求只是本轮可行性下限，不是统计功效保证，也不证明其他
受控设计同样缺乏数据。没有放宽匹配或搜索有利操作类型来启动模型。

全 train/valid 的 alpha 规范化源码存在 2 个跨划分重叠组、1 个共享提交组；train 拟合的字符
5-gram TF-IDF cosine>=0.95 筛查得到 7 个近重复候选。严格匹配子集没有命中这些检查。
这是明确范围内的筛查，不是不存在任何同源关系的保证。按项目内匹配控制项目边际分布，
也不构成未见项目评价。原始数据和被标记的成员均保留。

`review.json` 保存 8 个完整真实函数及人工检查说明（展示性选择，不外推提取精度）：

- `primevul:212376`（qpdf，正例）：`buf[len-1]` 确实关联 `buf/len` 参数和 `if(len)`；
  仍不能由正例标签认定这个具体操作有漏洞，也没有证明外部缓冲区容量。
- `primevul:505390`（cgit，负例）：`write(..., txt, strlen(txt))` 能绑定 `txt` 参数，
  可读长度及 NUL 终止仍依赖外部契约。负例标签不等于这些契约已证明。
- `primevul:210536`（Linux，正例）：`kfree(vc)`、`vc_deallocate` 赋值及其外层 else、
  null/index 条件对应清楚，但被调函数与所有权状态仍未知。
- `primevul:413449`（OpenEXR，负例）：交换 `codeCount[i]` 与 `base[i]` 的上下文，
  虽改变声明/类型，却保留相同容量表达式与循环约束。文本不同不能证明有效破坏 Bounds 条件。
- `primevul:481108`（Linux，负例）：两个 `buf->tail` 位于不同条件下，关联可核查；
  交换不自动意味着安全性质或函数标签改变。其他例子包含 Pointer 和参数对象释放的外部边界。

初次人工抽查发现“条件内部的目标被条件自身控制”和“for 更新的源码顺序被理解成执行前状态”
两处问题。只中止了本次 CPU 审计（退出 130，`results/operation_context_seed42/stopped.json`），
修复正式提取和说明后重跑全量，退出 0。条件内部目标不再得到自身 guard；循环更新明确记作
`lexically_earlier_write_not_reaching_value`，不推断当前值。28 项相关测试通过，包含作用域遮蔽、
Unicode 位置、未绑定对象、API 去重、交换身份、目标自控制和循环更新回归；`git diff --check` 通过。

历史核查与本轮边界：Mechanism 的 fixed-side target 与函数侧标签耦合；Composition schema-8
只有 62/5 个有效 train/valid 函数；CGVL 的 14/2 是严格局部差异准入，不能与本轮关联覆盖直接
比较；EGCL 曾出现条件常量相等的确定性捷径，修正数据仍未通过准入。本轮不复用这些标签，
不训练辅助头，不把静态关联包装成安全真值或新模型贡献。

已有 CLM-only checkpoint `results/cfg_dep_pretrain_seed42/lm_pretrain/last.pt` 和
`results/cfg_clm_probe_seed42/lm.features.pt` 可用；后者是完整源码池化表示，不包含本轮独立的
操作/上下文视图，不能冒充 B/C/D。GPU 只做了只读资源检查，没有启动前向或训练。
停止原因是受控数据规模和干预有效性，而不是硬件不可用。当前不足以据此开展新的操作条件化
模型开发；应先建立可独立评价、且确实改变操作—约束对应的对照，不能继续靠增加规则或模块追分。

结果目录：`results/operation_context_checked_seed42/`，含 `coverage.json`、`matches.json`、
`operations.jsonl` 和 `review.json`。复现全量自动审计（输出目录须不存在；人工说明单独保留）：

```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B -m vulnmechanism.audit_semantics \
  --operation-output-dir results/operation_context_reproduction_seed42
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B -m unittest \
  tests.test_audit_semantics tests.test_cfg_dependency -q
```

### 全成员操作—约束增量实验准入（2026-10-09）

本轮取消跨函数正负匹配，保留全部原 PrimeVul train 5,886 / valid 741 成员、源码和标签。
复用上一轮 `operations.jsonl`，没有增加操作类别或重跑 CPG 提取。**没有得到可核验的非等价
C/D 干预，因此未启动 B/C/D 冻结探测，不能回答其分类增量，更不能据此判定关系无用。**
既有 A 的冻结源码分类头已在 CPU 上实际重放；没有 Qwen 前向、LoRA 微调或新分类训练。

`audit_semantics.py` 新增的是已有关系的检查和干预准入，不是新的操作提取器。区分四层：

| 证据层次（操作数） | train | valid |
|---|---:|---:|
| 缓存中的声明绑定 | 45,621 | 5,717 |
| 有同声明端点的原生 DDG 候选 | 42,429 | 5,637 |
| 缓存中的控制关联 | 18,009 | 2,204 |
| 源码位置较早的更新 | 13,465 | 1,433 |
| 本轮核验的简单 guard 在目标处持续有效 | 125 | 31 |

前四行可重叠，DDG 端点关联不是完整数据来源/别名证明，较早更新不是到达值。
最后一行仅在顺序、非 volatile 的原生指针/整数语义下检查：现有 if 关联、具体操作数声明身份、
受限纯比较/布尔表达式，以及检查到操作间没有无法解释的调用、写入或循环。
普通目标存储不被误当成先修改指针/索引；RHS 调用或更新仍为 Unknown。
Bounds 条件约束了操作数，不证明引用的上界就是对象容量；Pointer 只涉及 nullness；
`free` 的判空不证明所有权或存活，因此不冒充 Lifetime 状态约束。

| 属性 | 通过 guard 核验的操作 train / valid | 独立函数 train / valid | 非等价干预函数 train / valid |
|---|---:|---:|---:|
| Bounds | 19 / 2 | 17 / 1 | 0 / 0 |
| Pointer | 106 / 29 | 74 / 4 | 0 / 0 |
| Lifetime | 0 / 0 | 0 / 0 | 0 / 0 |

去重后为 90 / 5 个函数（train 正/负 57/33，valid 正/负 4/1）。没有删除其余函数；
每个原成员都保存在本轮 `interventions.jsonl`，Unknown 或没有干预的记录保留空列表。
这些覆盖率描述当前复用的关联和核验范围，不是所有真实函数可分析性的上界。
既有 `cfg_program` schema-8 的状态查询虽覆盖 62/5 个函数，但数组索引正负查询均为 0，
不能将一般标量关系任务的覆盖直接当作敏感操作状态覆盖。

D 在同一函数、同操作类型/来源内交换已核验 guard 端点，保留操作及条件集合；不修改源码，
不从其他函数引入项目风格或标签线索。操作数角色规范化后，保存能使两个纯谓词取值不同的
具体赋值；这是谓词非等价见证，不是完整程序的可行执行见证，也不是 Safe/Unsafe 标签。
例如 `i<n` 与 `i<=n` 可由 `i=n` 区分；`p` 与 `p!=0` 不计为有效干预。
`p&&flag` 与 `p&&!flag` 虽谓词不同，却保持同一非空要求，也不能冒充 nullness 干预。
本轮运行前设定最小独立干预函数 train 50、valid 20 且 valid 各标签至少 5；实际为 0/0，
结论不取决于这个数值门槛。没有用 valid 分数选择类别、放宽标准或搜索其他对照。

全部 5 个 valid guard 函数均有可核查记录：

- `primevul:198469`（gpac，正例）贡献 24 个操作，大量是可选输出参数的 `if(p) *p=...`；
  变量名变化后仍是同一非空要求，不能当作 24 个独立样本或非等价条件。
- `primevul:197433`（gpac，正例）和 `primevul:201806`（ast，正例）各有两处相同非空检查。
- `primevul:197665`（tensorflow，正例）的 `if(node_index) *node_index=...` 关联明确，
  但没有可交换的不同条件，也不表明该操作是函数漏洞根因。
- `primevul:318160`（rpm，负例）的两个 Bounds 条目是同一次 `memmove` 的读/写角色，
  使用同一个最新 `ne>0` 检查。外层检查之后重赋值 `ne`，不能混同为同一变量状态。

第一次检查发现 C++ `condition_clause` 未解包及普通存储被过度排除；补回归测试后修复并
完整重跑。旧输出保留在 `results/binding_increment_seed42/`，`correction.json` 明确其不用于
结论。最终判据又重放全部 95 个 guard 函数，结果与保存记录一致。31 项相关单元测试通过，
包括 C/C++ 条件、指针/数组存储、调用/更新失效、等价 nullness、无关 flag 变化以及 Lifetime
不由判空推断；`git diff --check` 通过。

复用上一轮对完整 train/valid 的来源审计：2 个 alpha 规范化跨划分重叠组、1 个共享提交组、
7 个字符近重复候选，原始成员全部保留。5 个 valid guard 函数没有命中这些筛查。
这不是未见项目评价；57/33 和 4/1 的标签分布也说明提取成功子集有选择偏差。
没有新 C/B/D 结果可用于归因或纠错比较，不报告“纠错 0”来代替未运行。

A 复用 `results/cfg_clm_probe_seed42/lm.features.pt` 与 `lm.classification.pt`，确认原 train/valid
成员、标签及既有 cohort 标识一致，固定原 epoch 3、阈值 0.33。CPU 重放最大概率差为
5.96e-8。原训练协议是冻结 CLM、线性头、train-only 标准化、seed=42、20 epochs、
AdamW lr=0.001/weight_decay=0.01、batch=32，按既有 valid MCC 规则选模；本轮没有重新选模。

| A 的固定评价范围 | 函数数 | AUC | MCC | F1 | 未缩放 BCE |
|---|---:|---:|---:|---:|---:|
| 完整 valid | 741 | 0.773582 | 0.464480 | 0.752768 | 0.650126 |
| Bounds 候选函数 | 309 | 0.704211 | 0.419159 | 0.843683 | 0.617544 |
| Pointer 候选函数 | 589 | 0.762028 | 0.437549 | 0.768794 | 0.660431 |
| Lifetime 候选函数 | 36 | 0.851393 | 0.431254 | 0.765957 | 0.471619 |
| 通过 guard 核验的函数 | 5 | 1.000000 | 0.000000 | 0.888889 | 0.459765 |

完整 valid Accuracy=0.728745、Precision=0.695455、Recall=0.820375，TP/TN/FP/FN=306/234/134/67。
5 个 guard 函数全部被 A 预测为正，只有 1 个负例；其 AUC=1 不能解释为学会安全机制。
有效干预子集为空，指标为不可计算。以上均为 **A 的参考表现**，不是操作—约束增量或新的
A/B/C/D 性能实验；没有 B/C/D 的 AUC、MCC、BCE、F1 或逐样本纠错可报告。

结果：`results/binding_increment_checked_seed42/{coverage.json,interventions.jsonl,review.json,
source_replay.json,source_valid.predictions.jsonl}`。保存全部 valid guard 案例、全部原始成员、
Unknown 原因及 A 的逐例预测。当前不足以支持开发操作条件化新模型：统计分类收益、
正确绑定的利用、漏洞触发机制理解这三层结论均不能由本轮 guard 覆盖推导。
没有继续增加提取规则、模块或训练来追求结果。

复现自动准入和测试（新输出目录，不覆盖现有结果）：

```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B -m vulnmechanism.audit_semantics \
  --binding-output-dir results/binding_increment_reproduction_seed42
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY -B -m unittest \
  tests.test_audit_semantics tests.test_cfg_dependency -q
```

A 的固定 CPU 重放，不训练、不搜索阈值：

```python
import torch
from pathlib import Path
from vulnmechanism.cfg_metrics import metrics
root = Path('results/cfg_clm_probe_seed42')
c = torch.load(root/'lm.features.pt', map_location='cpu', weights_only=False)
r = torch.load(root/'lm.classification.pt', map_location='cpu', weights_only=False)
v = torch.tensor(r['valid_indices'])
x = (c['pool'].float()[v] - r['mean']) / r['std']
s = r['selected']['head_state']
z = (x @ s['weight'].T + s['bias']).flatten()
y = c['labels'][v].float()
print(metrics(y.int().tolist(), z.sigmoid().tolist(), r['selected']['threshold']))
print(torch.nn.functional.binary_cross_entropy_with_logits(z, y).item())
```

### 函数级数据统计与可学习表征审计（2026-10-09）

本轮只研究原函数标签的可学习性，不构造操作级安全标签。新增 `vulnmechanism.audit_statistics`，
复用 `operation_context_checked_seed42` 的已核对来源与 token 计数、原 CPG sidecar、
`cfg_clm_probe_seed42/lm.features.pt` 以及历史 valid 预测。未执行 Joern、Qwen 前向、LoRA 或完整分类训练。
只对 train/valid 做统计和 CPU 线性探测；没有打开 test 数据或预测产物作分析、选特征或调参。
混合划分的正式 JSONL 在读取身份后过滤，非 train/valid 内容不进入特征或评价。

结果集中在 `results/function_statistics_seed42/`：`features.jsonl` 保存逐函数特征和提取质量，
`visible_sources.jsonl` 保存相同 2048-token 前缀；`distributions.csv`、`projects.csv`、
`data_quality.json` 保存分布、来源与重复检查。原始源码和标签不变。
标识符计数不是准确变量绑定数；CFG cycle rank 是保存边的图论统计，不是可行路径数。
CPG 的 `accepted` 表示通过原构图准入，不表示依赖关系完整或语义正确。

| 划分/标签 | 函数数 | Qwen token 中位数 | 超过 2048 tokens | Tree-sitter 根节点 has_error |
|---|---:|---:|---:|---:|
| train / 0 | 3051 | 125 | 43 | 643 |
| train / 1 | 2835 | 481 | 340 | 715 |
| valid / 0 | 368 | 129.5 | 6 | 78 |
| valid / 1 | 373 | 494 | 38 | 95 |

构图前正式清单 train 正负各3920，valid正负各507。以上是通过构图后保留的成员，
不是未经选择的原始 PrimeVul 分布。train有503个project字段取值，valid有172个；
103个valid函数来自train未出现的project。按project分组不等同于严格的仓库fork隔离。
完整源码无精确重复；2048-token可见源码文本有1组同train、异标签重复（2个函数）。
此前保留的字符5-gram近重复筛查有7个跨划分候选对，涉及4个valid函数；另有1组共享commit。
这些筛查不构成“已经排除所有克隆”的证明，完整valid始终保留。

全体当前成员都有已接受的CPG，但准确位置过滤后，train有143、valid有16个函数没有可见CFG边。
节点代码不匹配、坐标冲突及窗口外节点分别记录；观察到0条边不等同于真实程序没有关系。
原C读取完整函数图、不做图规模截断，而源码分支只读前2048 tokens，历史图收益因此不能全部
归为同一输入范围内的结构建模。本次新探测只计准确定位且两个端点都可见的关系；这些边仍来自
原完整函数的静态分析，不能声称其生成过程只依赖前缀。

train中，单个长度的rank AUC为0.7900；完整CFG/DDG/CDG边数分别为0.7874/0.7791/0.7780。
控制log可见长度、字符数、行数及project后的线性残差相关分别约0.050/0.023/0.062。
调用数、赋值数、分支数也有很强的单变量关联，但相应残差相关仅0.036/0.055/0.027。
这是有限线性控制下的关联减弱，不是因果消除，也不能证明更细的结构组织没有信息。
Bounds/Pointer/Lifetime候选的train正/负出现比例分别为56.2%/27.9%、87.4%/70.2%、9.1%/4.4%；
大量两类共有的操作不是足够的分类依据。相应计数控制长度与project后的残差相关仅0.035/0.009/0.023。

固定CPU协议：L2 LogisticRegression，C=1、liblinear、max_iter=2000、seed=42，不搜索参数。
数值标准化、词表、TF-IDF都仅在对应训练折拟合。词法模型用前缀内的词/符号1–2gram，min_df=3、
最多30000维。分别做3折分层train OOF、按project字段隔离的3折train OOF，再用完整train拟合。
分类阈值只由分层train OOF选择，valid只检查固定方案。图统计在相同语法/长度输入上增量加入。
已有冻结CLM池化向量只训练CPU线性头，没有重算编码器；该探测与历史20轮AdamW探测不是同一协议。
新探测和历史模型的阈值选择来源不同，不能将二者的MCC当成完全相同选模预算的竞赛结果。

固定对照实际结果如下；全部模型、阈值、纠错ID、分组指标均保留，不按valid筛掉负结果。

| CPU表示/特征 | train OOF AUC | project隔离OOF AUC | valid AUC | MCC | F1 | 概率BCE↓ |
|---|---:|---:|---:|---:|---:|---:|
| size | 0.7909 | 0.7895 | 0.7712 | 0.3964 | 0.7113 | 0.5740 |
| syntax | 0.8006 | 0.7963 | 0.7860 | 0.4315 | 0.7277 | 0.5601 |
| structure | 0.8018 | 0.7967 | 0.7856 | 0.4070 | 0.7158 | 0.5595 |
| lexical | 0.8083 | 0.7859 | 0.7988 | 0.4550 | 0.7349 | 0.5493 |
| lexical_structure | 0.8167 | 0.8013 | 0.7991 | 0.4684 | 0.7404 | 0.5462 |
| project_size | 0.8008 | 0.7879 | 0.7783 | 0.4305 | 0.7205 | 0.5714 |
| project_structure | 0.8081 | 0.7958 | 0.7829 | 0.4414 | 0.7266 | 0.5648 |
| frozen_clm | 0.8186 | 0.7492 | 0.6884 | 0.2929 | 0.6571 | 2.3442 |
| frozen_clm_structure | 0.8226 | 0.7550 | 0.6930 | 0.2775 | 0.6118 | 2.2864 |

`structure`包含size、syntax和可见图统计；组合词法/冻结表示的structure同样包含这些基础统计。
BCE从保存概率复算，边界裁剪为[1e-12, 1−1e-12]；两组冻结L2头在valid分别有17/13个概率被裁剪，其他组为0。
因此冻结头的数值不是直接保存原logit的BCE，不能隐藏极端置信度或据此比较校准上限。
冻结CLM阶段1已见过全部train源码；project隔离只针对本轮有监督读出，不能称为编码器预训练也未见这些项目。
项目折评估成员数为1963/1962/1961，正例各945。当前固定C=1探测明显弱于历史20轮AdamW冻结头
（valid AUC 0.7736），不同优化/正则化及选模协议不能混为同一个实验；此负结果不证明CLM表示没有可学习信息。

| 已有完整分类模型（同一741 valid） | AUC | MCC | F1 | 概率BCE↓ |
|---|---:|---:|---:|---:|
| Source | 0.8047 | 0.5116 | 0.7302 | 0.7788 |
| CLM_Source | 0.8168 | 0.5052 | 0.7230 | 0.6791 |
| P0_Source | 0.8213 | 0.5126 | 0.7497 | 0.7057 |
| C | 0.8107 | 0.4972 | 0.7307 | 0.7963 |
| P0_C | 0.8208 | 0.4935 | 0.7410 | 0.7636 |
| Attributes | 0.8134 | 0.5304 | 0.7661 | 0.7288 |
| CLM_C | 0.7868 | 0.4462 | 0.7494 | 0.6123 |

`C`表示原Qwen源码分支与原C图模块联合训练，非图单支；`Attributes`没有CFG传播。
历史模型保留原valid选中的checkpoint及阈值；另保存固定0.5结果。本轮未重选它们。

语法统计相对长度纠正35例、引入22例；在语法之上加图统计纠正10例、引入19例。
词法加统计纠正35例、引入30例，但AUC仅+0.00030，配对样本bootstrap区间[-0.01146,0.01306]。
冻结CLM加统计纠正35例、引入42例，AUC+0.00457，但MCC和F1下降。
这些结果不支持本次图规模/连接统计具有一致的额外分类价值；未检验所有可能的细粒度结构表示。

历史Source→CLM纠正39例、引入42例；Source→P0+Source为47/44；Source→P0+C为47/51；
CLM→P0+C为29/30。CLM在最长长度四分位净纠正4例，在第二四分位净损失7例；
P0+C相对CLM在第二四分位净纠正5例，其他三组分别−2/−1/−3。
这些均沿用各自固定阈值，不等于同一工作点下的因果贡献。
Source→P0+C的AUC差0.01611，其配对iid bootstrap区间[-0.00091,0.03327]。
bootstrap只条件于现有已选模型，未校正历史valid反复使用、多重比较或项目相关性，不能充当独立确认。

长度分组仅由train的94/244.5/631.75-token分位点确定。valid四组正/负成员为30/143、75/108、
107/75、161/42。P0+C各组FN/FP为24/4、41/19、31/31、8/30：短正例与长负例是明确的持续困难群体。
其四组AUC为0.728/0.715/0.706/0.725；Source为0.682/0.683/0.695/0.728。
模型分数在同标签内部仍与长度强相关：P0+C的负/正类Spearman分别0.689/0.679。
这说明分数与规模相关，不证明模型仅通过长度或某条特定因果路径决策。

| 表示/模型 | 总体valid AUC | 仅同长度分箱内正负对AUC | 同project且同长度分箱内AUC |
|---|---:|---:|---:|
| 长度统计 | 0.7712 | 0.5863 | 0.6447 |
| 词法 | 0.7988 | 0.6683 | 0.6514 |
| Source | 0.8047 | 0.6975 | 0.7105 |
| CLM+Source | 0.8168 | 0.7059 | 0.7284 |
| P0+C | 0.8208 | 0.7168 | 0.7310 |

条件AUC按可比较正负对数加权，不是各组AUC的简单平均，也不是新的选模标准。
同长度条件覆盖741函数/27177对；联合project和长度仅覆盖407函数/1565对。
它并未精确匹配长度或控制全部混杂因素，不能把与总体AUC之差解释为“某因素贡献百分比”。
但Qwen系列仍有超过长度/词法的组内区分能力，不能说模型只学会统计捷径。

结构分组给出相似而非独立的规模现象：P0+C在最低DDG边数四分位漏报28/30正例，
最高四分位误报27/37负例。CFG cycle rank四组AUC为0.776/0.783/0.742/0.735，
CDG边数四组为0.765/0.770/0.701/0.712。这些结构量与长度高度相关，不能直接归因为结构推理不足。
Tree-sitter无/有error两组AUC为0.8213/0.8171，没有观察到足以把整体错误解释为解析失败的差距。
Bounds/Pointer/Lifetime候选子集分别309/589/36例，P0+C AUC为0.7777/0.8128/0.8359；
Lifetime只有19正/17负，不据此声称某种安全知识已经学会。

五个主要模型共同误判119例（79正、40负），其中仅4例被截断。44个截断valid成员含38正、6负；
Source在此全部判正，Accuracy仍为0.8636，但MCC为0。P0+C的AUC为0.5702，样本尤其负例太少，
不能把这个子集的高Accuracy当成有效长函数表征，也不能把多数错误归因于截断。
排除5个已知跨划分近重复/共享commit成员的**补充诊断**为736例，Source/CLM/P0+C AUC为
0.8070/0.8209/0.8244，未改变主要观察；正式valid仍为741例，没有删样本重报主成绩。

复核历史固定P0+C拓扑干预：保度重连改变约95%边，AUC从0.820834仅变到0.820783–0.820827，
判定至多变1例；固定源码分支自身AUC为0.820645。依据分别在
`cfg_origin_topology_seed42/summary.json` 与 `cfg_clm_c_diagnostic_seed42/p0_valid/summary.json`。
结合本次统计，当前C的表现不能作为已经充分使用正确CFG拓扑的证据。
同样，历史依赖探测成功只说明关系信息可读，不能替代函数级分类的增量证据。

`examples.json`保存8个按预定错误类型、最接近train长度中位数选出的真实完整函数及分数：
例如Linux `primevul:204147`（`input_set_keycode`，原标签1，244 tokens）五模型均漏报；
Graphviz `primevul:505496`（`graphml_to_gv`，原标签0，239 tokens）五模型均误报。
二者都未截断，不从函数内容反推漏洞机理或更改标签。
另已实际核对OpenSSL `primevul:184276`/`primevul:9119`：原标签0/1、完整7237/7233 tokens，
前2048个源码token ID完全相同，均属train。它是有限窗口不可区分的真实案例，但只有1对。
`diagnostics.json`保存核对结果，`selection_coverage.json`保存全部2227个未进入当前train/valid成员的失败阶段计数；
它们都能对应原error记录。未保留函数在两类中都更长，不能把现有正负规模差全部归因于构图过滤。

研究判断：此前的研究路线确实过早把“缺少显式漏洞机理监督”当成主要前提。
当前可直接观察到的是强规模/词法关联、有限但真实的同规模源码区分能力，以及来源/读出协议敏感性。
现有数据不要求先恢复安全真值才可研究二分类，也尚未证明更丰富的关系目标是缺失的信息。
下一步优先验证两个**表征假设**，本轮不实施新模型：

1. 在已有表示中，易学的规模/来源成分与难学的函数内部差异混合，限制了短正例、长负例的判别。
   应先在train内区分这些成分与剩余表示，固定同容量读出，在项目隔离和长度条件下检查剩余增量。
   如剩余信息本身不可读，不能先把问题命名为融合不足。
2. 源码中超过词袋和图规模的组织信息可能有价值，但现有全局表示/训练未稳定保留它。
   应先用保留与破坏组织关系、同时控制可见内容与统计量的固定表示对照检验；只有出现条件分类增量，
   才讨论相应表示学习方法。当前组内AUC优势不能单独证明模型利用了顺序或依赖关系。

图表为 `train_distributions.png`、`historical_valid.png`、`length_errors.png`；
逐例分数/特征在 `valid_sample_analysis.csv`，全部分组指标在 `subgroups.csv`，纠错和区间在
`comparisons.json`，条件比较在 `conditional_metrics.csv`。没有新test结果或新大型模型结果。

实际运行5项直接相关测试全部通过：可见图诱导子图与cycle rank、UTF-8截断语法计数、指标与标签隔离、
全部9条探测路径不读取valid标签训练、条件AUC不计入跨project先验。
第一次CPU入口因SciPy对纯dense块的hstack处理报错，在任何探测结果产生前修正为显式CSR转换；
失败栈仍保留于 `analysis.log`。随后9组探测和描述统计入口均退出0，`git diff --check`通过。
未改动原数据、训练器、checkpoint、划分或旧结果。

复现命令（prepare/analyze拒绝覆盖已有结果；使用新的空输出目录）：

```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/vul-statistics-mpl
$PY -m unittest tests.test_audit_statistics -v
$PY -m vulnmechanism.audit_statistics --phase prepare --output-dir results/function_statistics_reproduction
$PY -m vulnmechanism.audit_statistics --phase analyze --output-dir results/function_statistics_reproduction
$PY -m vulnmechanism.audit_statistics --phase describe --output-dir results/function_statistics_reproduction
```

### CodeBERT 源码编码器替换：512 与 2048 滑窗（2026-10-10）

本轮以原 `cfg_abc_seed42/baseline` 的 Qwen Source-only 为参照，使用本地
`/home/PublicData/PHY-data/resource/codebert-base` 替换编码器。不增加 CLM/P0 预训练、图模块或辅助损失。
保持原 PrimeVul train 5886 / valid 741 成员、顺序和标签，seed=42、3 epochs、batch=1、
梯度累积8、AdamW lr=2e-4、weight decay=0.01、梯度裁剪1.0；每轮736次更新。
仍使用 LoRA r=16、alpha=32、dropout=0.05，映射到 CodeBERT 的 query/key/value/attention-output
投影；不改成全参数微调。模型原生的双向注意力、分词器、隐藏维度及预训练权重随编码器改变，
所以不能声称是等参数量、等FLOPs或完全相同源码覆盖的比较。

正式实现复用 `model.py` 的 `SequenceVulnerabilityClassifier`、训练器和保存/加载逻辑，
只增加 `codebert_source` 输入与滑窗聚合路径。分类头仍为平均源码表示上的单层线性头。
两组都从同一原始 CodeBERT 初始化，保留原任务提示；没有独立窗口分类头或窗口级标签。

- `native512`：CodeBERT 原生最大512个位置。
- `window2048`：逻辑预算2048，物理窗口仍最多512；源码窗口容量489、stride244。
- 两种逻辑预算均计入一份21-token任务提示及BOS/EOS，实际源码预算分别489和2025。
- 每个窗口单独做原生CodeBERT编码。重叠源码位置按覆盖次数归一化；重复的提示和边界位置
  在窗口间取平均，最后对唯一源码位置及一份提示/边界做均值聚合。padding权重为0。
  短函数不复制源码凑长度；每个原始函数仍只有一个logit和一次BCE。
- 没有跨窗口自注意力，2048配置不等同于具有原生2048位置的编码器。

原通用CLI会对构图成功的PrimeVul再次做类别平衡；本次通过显式 `--preserve-members` 保留原成员及顺序。
默认旧策略未修改。所有验证checkpoint、阈值仍按原valid MCC/F1/Accuracy/AUC规则选取，不用test调参。
训练/评价来自同一正式CLI，没有另建训练脚本。

CodeBERT分词通常比Qwen更长：valid负/正例token中位数为205/788。
512配置完整覆盖414/741函数，2048滑窗完整覆盖636/741；222例从截断变为完整，105例两组都截断。
其中双截断组为93正/12负，小组Accuracy不能直接作为有效判别证据。
原Qwen的2048源码token窗口仅截断44例，两种分词器的token预算不代表相同源码范围。
逐函数覆盖保存在 `results/codebert_source_seed42/token_coverage.jsonl`。

实现检查：16项相关测试通过，包括原Qwen/图路径回归、原始成员和顺序保持、滑窗覆盖权重、
padding屏蔽、短函数等价、函数级梯度，以及真实小型RoBERTa的训练/保存/加载/评价。
本地完整CodeBERT另已通过GPU smoke：短函数512/2048输出完全相同；真实长函数滑窗反向有非零LoRA梯度；
保存加载后的logit最大误差0。可训练参数1,180,417，总参数125,826,049（含LoRA与分类头）。
smoke独立于正式训练，结果在 `gpu_smoke.json`，不作为性能成绩。

两组正式训练及固定checkpoint评价均退出0；各完成3轮、每轮736次更新，共2208次。
512选中epoch 2，2048选中epoch 3。以下为完整valid 741例（373正/368负），
采用各自按原协议选定的阈值；逐样本成员、标签、源码身份及顺序与原Qwen预测完全一致。
未执行test评价。BCE未缩放，新模型从原始logit计算，Qwen复用已有概率计算且不做截断。

| Source-only模型 | 阈值 | AUC | MCC | BCE↓ | Accuracy | Precision | Recall | F1 | FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen原基线（复用） | .31 | .804727 | .511603 | .778766 | .751687 | .805825 | .667560 | .730205 | 60/124 |
| CodeBERT 512 | .15 | .782638 | .463924 | .725652 | .730094 | .704492 | .798928 | .748744 | 125/75 |
| CodeBERT 2048滑窗 | .54 | .790408 | .468791 | .592887 | .734143 | .745810 | .715818 | .730506 | 91/106 |

2048相对512：AUC +.007770、MCC +.004867、BCE −.132765，纠正49例、引入46例，净增加3例正确。
减少34个误报，同时增加31个漏报，F1下降.018238。512相对Qwen纠正70/引入86，
2048相对Qwen纠正51/引入64；两组CodeBERT均未超过Qwen的AUC/MCC。
512的F1较高主要伴随Recall提升和误报从60增至125，不能称为全面提升。
2048的低BCE说明本次概率损失更低，不等价于更强的排序或阈值分类能力。

固定阈值0.5的MCC/F1分别为Qwen .453921/.670846、512 .309558/.495413、
2048 .460164/.731903。512的表现明显依赖valid选取的低阈值；这与模型分数尺度有关，
不能只比较固定0.5或只比较最优F1来替代原选模协议。所有epoch均保留：

| 配置 | epoch | train BCE | valid AUC | valid MCC | valid阈值 |
|---|---:|---:|---:|---:|---:|
| 512 | 1 | .528347 | .774435 | .441844 | .49 |
| 512 | 2（选中） | .459831 | .782638 | .463924 | .15 |
| 512 | 3 | .431773 | .789945 | .461078 | .16 |
| 2048 | 1 | .520056 | .778044 | .462899 | .29 |
| 2048 | 2 | .440885 | .788681 | .464601 | .33 |
| 2048 | 3（选中） | .408099 | .790408 | .468791 | .54 |

按预先定义的CodeBERT源码覆盖分组，使用上述同一checkpoint和全局阈值，不在组内调参：

| 分组 | 样本（正/负） | Qwen AUC | 512 AUC/MCC | 2048 AUC/MCC | 2048相对512净纠错 |
|---|---:|---:|---:|---:|---:|
| ≤489，512已完整 | 414（136/278） | .736537 | .723921/.348789 | .730189/.323033 | 0 |
| 490–2025，滑窗新增完整覆盖 | 222（144/78） | .701834 | .631811/.161455 | .653045/.241708 | +4 |
| >2025，两组均截断 | 105（93/12） | .641577 | .532706/−.035223 | .547939/−.050055 | −1 |

新增完整覆盖组有有限改善，但仍低于Qwen；最长组两种CodeBERT均把12个负例全部误报。
当前结果仅支持“此配置的滑窗略改善CodeBERT排序和概率损失”，不支持“长上下文已解决长函数判别”，
也不是滑窗的纯推理因果效应：两组训练上下文和最终权重不同。单seed、valid选模，不能据此宣称稳定优势。
未因结果追加轮数、学习率搜索或其他模块。

产物：`results/codebert_source_seed42/comparison.valid.json`保存完整指标、逐样本纠错ID、
分组比较及各轮结果；同目录CSV为主表。两组子目录分别保存`best.pt`、`best.training.jsonl`、
`valid.predictions.jsonl`、`valid.metrics.json`和运行日志。
修改文件为`model.py`、`cli.py`、`benchmark_view.py`、`tests/test_codebert_source.py`及本说明。

复现命令（两组分别执行，输出目录须为未使用目录；已存在的评价文件拒绝覆盖）：

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
PYTHON=/home/phy/miniconda3/envs/vul-detect/bin/python
# native512: LENGTH=512, NAME=native512, DEVICE=cuda:0
# window2048: LENGTH=2048, NAME=window2048, DEVICE=cuda:1
LENGTH=512
NAME=native512
DEVICE=cuda:0
OUT=results/codebert_source_reproduce/$NAME
mkdir -p "$OUT"
"$PYTHON" -u -m vulnmechanism.cli train \
  --dataset data/function_dataset.jsonl --source-dataset primevul --preserve-members \
  --variant codebert_source --model /home/PublicData/PHY-data/resource/codebert-base \
  --source-max-length "$LENGTH" --batch-size 1 --gradient-accumulation 8 \
  --epochs 3 --learning-rate 0.0002 --weight-decay 0.01 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --seed 42 --log-every 100 \
  --device "$DEVICE" --output "$OUT/best.pt" > "$OUT/train.log" 2>&1
"$PYTHON" -u -m vulnmechanism.cli eval \
  --dataset data/function_dataset.jsonl --source-dataset primevul --preserve-members \
  --checkpoint "$OUT/best.pt" --split valid --batch-size 1 --device "$DEVICE" \
  --predictions "$OUT/valid.predictions.jsonl" --output "$OUT/valid.metrics.json" \
  > "$OUT/eval.log" 2>&1
"$PYTHON" -m unittest tests.test_codebert_source tests.test_benchmark_view tests.test_graph_training -q
```
