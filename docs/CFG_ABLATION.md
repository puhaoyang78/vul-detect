# CFG 属性与结构传播对照（DeepDFA-inspired adaptation）

本实现用于当前 Qwen2.5-Coder-7B-Instruct + LoRA 的 C/C++ 函数级分类实验。
它借鉴 DeepDFA 的赋值节点抽象属性、门控图传播和注意力汇聚，但**不是官方 DeepDFA 的逐项复现**。
没有执行轨迹、修复对排序、漏洞类型辅助监督或新增源码 Transformer。

## 1. 三组实验

| 标识 | 实现 |
|---|---|
| `baseline`（A） | 直接调用原 `model.train_model` 与 `predict_checkpoint`，保留源码编码、mean pooling 和分类头。 |
| `attributes`（B） | 同一源码分支 + 节点属性编码。每个节点只做自身状态更新，不与其他节点交换信息。 |
| `cfg`（C） | 与 B 相同的初始化、模块、参数量、更新次数、汇聚和分类头，额外沿有向 CFG 的真实前驱边聚合。 |

所有源码分支均从相同预训练模型与 seed 初始化，并正常微调 LoRA；不是冻结 Qwen 只训练图或分类头。
B/C 额外模块的 CPU 初始化隔离随机数状态；不改变源码分支初始权重或后续随机数流。
分类为完整宽度的源码线性头 + 图线性头，数学上等价于拼接表示后线性分类。
图线性头零初始化，所以初始输出等于源码 baseline；**训练以后不保证原正确样本不变**。

## 2. 与官方实现的对应关系及区别

参照的官方版本：`ISU-PAAL/DeepDFA@14414f34293cd994001faa69f4553f6fc559b2b4`。

- 原始属性提取：`DDFA/sastvd/scripts/abstract_dataflow_full.py`。
- 图编码结构：`DDFA/code_gnn/models/flow_gnn/ggnn.py`。
- 官方论文：Dataflow Analysis-Inspired Deep Learning for Efficient Vulnerability Detection，ICSE 2024。
- 官方代码：https://github.com/ISU-PAAL/DeepDFA

本实现沿用赋值/复合赋值/自增自减节点作为属性中心，从其 AST 后代提取 API、字面量、运算符，
从左值表达式获取类型；非赋值节点仍保留在 CFG，但四类属性使用空值。没有赋值节点的函数不删除。
属性不会读取 label、CVE、fix commit、pair_id、函数名或文件名作为独立特征。
API 名、类型名和原始字面量是明确保留的程序属性，并不声称完全消除了词法相关性。

与原 artifact 的具体区别：

1. 使用当前仓库的 Joern 函数图导出；按 AST 子节点的 `ARGUMENT_INDEX`/`ORDER` 寻找实参，
   不依赖旧版 Joern 的独立 ARGUMENT 边。优先使用表达式自身已解析类型，保留类型限定符。
2. 每类属性的有序规范化多重集拥有独立词表；只在训练集统计，默认每类最多 2048 项，包含空值和未知值。
   没有完全移植原 artifact 的多阶段词表过滤流程。
3. 使用纯 PyTorch 的 directed-sum + GRUCell，无 DGL 依赖。
   B/C 每轮都有一次自身消息，C 另加非自身 CFG 前驱消息；共 5 轮。
4. 四类嵌入拼接宽度默认 128。最终节点状态与初始属性向量拼接，再用线性门控 softmax 汇聚成 256 维图向量。
5. 使用本仓库的 Qwen、BCE、三轮训练及验证集选模，不复用 DeepDFA 的 BigVul 划分、训练轮数或权重。

因此，正负结果都只对应这个适配及其数据设置，不能冒充原论文复现结论。

## 3. 数据不覆盖、不静默删样本

只修改现有 `cpg.py`：给 `GraphNode` 增加可选 `properties` 字段，保留已有比较/hash语义及三参数构造方式。
`model.py`、`dataset.py`、`semantics.py` 及原 checkpoint 格式均不改动。

新图文件按原 schema-9 `function_dataset.jsonl` 的已成功样本逐一导出。
保留当前函数图视图的完整节点及 AST/CFG/CDG/DDG 边，不把 `cpg_relations` 文本反推为图，不按节点文本合并。
这里的“完整”是当前四类边视图，**不是 Joern 的全部边类型**；CFG 模型只沿 CFG 边传播，AST 用于属性提取。

每条图记录绑定 `sample_key`、原源码 SHA256、dataset、split、label；存在缺失/重复/身份不匹配会报错。
同一 PrimeVul cohort 必须全部导出成功才能训练；不会让 B/C 悄悄少一些难样本。
既有 schema-9 文件、原始源码和历史结果均不覆盖。

图旁文件：

- `*.meta.json`：cohort、工具路径与 Joern 输入预处理版本；不允许混用不同数据或旧预处理缓存。
- `*.errors.jsonl`：失败历史，追加记录；旧失败可能已在后续重试成功。
- `*.audit.json`：当前完整性、按 split/label 的失败和无定义节点情况。
- `*.lock`：Linux advisory lock 标识。锁文件存在不代表正在运行；不要为“解锁”直接删除它。

构图可直接重跑同一命令：只重试未完成的样本，逐样本落盘；中断的最后一条不完整写入会恢复。
缓存同时记录 Joern 输入预处理版本，以及每条样本的预处理是否触发、原始源码哈希和实际解析源码哈希。
抽取或预处理逻辑改变后，删除旧 sidecar 后仍使用固定的 `primevul_cfg.jsonl` 重新全量构建，不再通过 `v1/v2` 文件名区分版本。

## 4. 命令

在原 `(vul-detect)` 环境运行，不升级依赖，也不要求安装 DGL/torch-geometric。

```bash
cd /home/PublicData/PHY-data/vul_detect/work/vul-detect

# CPU 单元测试，不加载 Qwen 或 Joern。
python -m unittest discover -s tests -p 'test_cfg*.py' -v

# 为原成功 cohort 导出新图；先只做 PrimeVul。
python -m vulnmechanism.cfg_experiment build \
  --dataset data/function_dataset.jsonl \
  --source-dataset primevul \
  --output data/graphs/primevul_cfg.jsonl \
  --joern-dir /home/phy/joern \
  --java-home /home/phy/jdk21 \
  --batch-size 8 --timeout 300

# 按 A -> B -> C 顺序运行，单 seed，训练期间只评价 valid。
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cfg_experiment run \
  --dataset data/function_dataset.jsonl \
  --graphs data/graphs/primevul_cfg.jsonl \
  --source-dataset primevul \
  --model-path /home/phy/models/Qwen2.5-Coder-7B-Instruct \
  --output-dir results/cfg_abc_seed42 \
  --source-max-length 2048 \
  --batch-size 1 --gradient-accumulation 8 --epochs 3 \
  --learning-rate 2e-4 --graph-learning-rate 1e-3 \
  --seed 42 --device cuda --resume
```

显卡编号按当前空闲卡调整。`CUDA_VISIBLE_DEVICES=1` 时程序内部的 `cuda:0` 对应物理卡 1。
源码侧 lr=2e-4，图侧 lr=1e-3；B/C 完全相同；weight_decay=.01，LoRA r16/alpha32/dropout.05。
默认全量 cohort：不会再抽一百多条样本，也不会自动跑多 seed。

也可以使用包装脚本（保持上述默认设置）：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_cfg_abc.sh all
```

脚本 `all` 会先构图；失败则立即停止，不启动训练。支持 `build`、`train`、`test`、`compare`。
路径可以通过 `DATASET`、`GRAPHS`、`RUN_DIR`、`MODEL_PATH`、`JOERN_DIR`、`JAVA_HOME` 环境变量覆盖。
直接命令和包装脚本二选一，不要同时运行。

## 5. 输出与评价口径

```text
results/cfg_abc_seed42/
  config.json
  vocabulary.json
  baseline/   best.pt, config.json, complete.json, valid.predictions.jsonl, valid.metrics.json
  attributes/ best.pt, config.json, complete.json, history.jsonl, valid.predictions.jsonl, valid.metrics.json
  cfg/        best.pt, config.json, complete.json, history.jsonl, valid.predictions.jsonl, valid.metrics.json
  comparison.valid.csv
  comparison.valid.json
```

A 的训练曲线文件仍由原 `TrainingProgress` 输出，不改原格式。

阈值和 epoch 选择规则与原 baseline 一致：只用 valid，阈值 .05~.95 步长 .01，MCC 优先，
同分时按 F1、Accuracy 和阈值距离 .5 决定；epoch 按 MCC/F1/Accuracy/AUC 决定。
概率阈值比较遵循原 float32 Tensor 的边界行为。

每个样本保存源码 hash、标签、概率、判定阈值、预测、源码 token 数及是否超过源码预算。
图模型读取完整函数 CFG，不裁切图；因此同时报告未截断/已截断源码子组，
避免把额外可见代码带来的收益全部解释为结构收益。
比较报告保存：`fn_to_tp`、`fp_to_tn`、`tp_to_fn`、`tn_to_fp` 的数量和具体 sample_key，
以及净纠错、净减少漏报、净减少误报。分别比较各自 valid 阈值、共同 .5 阈值、共同 baseline 阈值。

训练过程中不调用 test 推理。确定方案后显式运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cfg_experiment eval \
  --run-dir results/cfg_abc_seed42 --split test --device cuda
```

test 固定使用所存 valid 阈值，绝不重新调阈值；已有预测默认拒绝覆盖。
计算汇总无需模型/GPU：

```bash
python -m vulnmechanism.cfg_experiment compare --run-dir results/cfg_abc_seed42 --split valid
```

## 6. 续跑与安全边界

`build` 支持样本级续建。`run --resume` 支持校验身份后跳过已经完整训练的变体；
**不是 optimizer/epoch 中途续训**。某一变体中断时保留它的目录，移到另一个名字后再重跑该变体，
或使用新的 run 目录。不要直接删除历史 checkpoint。

run 的数据、图文件、词表、参数、checkpoint 和完成预测均有校验；配置变化需使用新结果目录。
旧 baseline checkpoint 仍可通过原 CLI 使用，但这次 A/B/C 默认统一重跑，以保证完整可比的记录。

本补丁交付时已执行 CPU 测试：图提取规则、cache 续建、真实 Torch 图网络前反向、参数对照、
checkpoint 恢复、A/B/C 编排与指标。测试使用小型源码编码器和模拟 Joern 输出，
**没有在交付环境加载 7B Qwen、运行真实 Joern 或完成 GPU 实验**。
单 seed 的小幅差异只是开发信号；已有反复使用的验证/测试数据不能被描述为新的独立确认。
