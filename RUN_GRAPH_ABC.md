# Source / abstract attributes / directed CFG

这是一组固定口径的静态图实验。**保留现有源码 baseline、schema-9 数据和旧 checkpoint；不引入执行轨迹、修复对监督、机制辅助标签或新的源码 pooling。**

| 组 | `variant` | 结构 |
|---|---|---|
| A | `baseline` | 原有 Qwen + LoRA + masked mean + 线性分类 |
| B | `graph_attributes` | 原源码支路 + 抽象属性；节点独立更新，无跨节点传播 |
| C | `graph_cfg` | 与 B 相同的属性、参数量、迭代次数；沿真实有向 CFG 汇总前驱信息 |

## 方法边界

依据 DeepDFA 官方实现（`ISU-PAAL/DeepDFA`，commit `14414f34293cd994001faa69f4553f6fc559b2b4`）：

- `DDFA/sastvd/scripts/abstract_dataflow_full.py`：在赋值、复合赋值、增减等定义节点提取 API、左值类型、常量、运算符；其他 CFG 节点的属性为空。
- `DDFA/code_gnn/models/flow_gnn/ggnn.py`：四路属性 embedding、GGNN、初始/最终节点表示拼接、全局注意力汇聚。
- 官方仓库：https://github.com/ISU-PAAL/DeepDFA
- 论文：https://arxiv.org/abs/2212.08108

这里是适配到当前 Joern/Qwen 训练流程的 **DeepDFA-style 实现，不是原论文结果的严格复现**。差异包括：用原生 `ARGUMENT_INDEX`/`ORDER` 与 AST 定位实参；未知类型显式标记；每个属性通道将值集合编码为训练集词表中的 signature；有向 GGNN 使用纯 PyTorch，不依赖 DGL；与当前单函数 Qwen 分类器联合训练。

B 使用相同线性变换和 GRU，只在每个节点内部更新；C 使用 CFG 前驱消息之和。两者最终均保留完整源码向量，图向量不压缩源码表示。源码线性项加图线性项等价于拼接后线性分类；图线性项零初始化，让 A/B/C 初始源码 logits 一致。联合训练仍会更新 LoRA，**这不保证训练后没有新增错误**。

## 1. 本地环境与检查

在已有 `vul-detect` conda 环境、仓库根目录执行。不需要重新安装 CUDA/PyTorch，也不需要安装 DGL/PyG。新实现没有增加 requirements。

```bash
git pull --ff-only origin main
python -m unittest tests.test_graph_features tests.test_graph_training -v
```

测试使用微型模拟编码器和人工图，覆盖数据/结构/梯度/初始化/训练保存重载/预测比较，不会下载 7B 模型或启动 Joern。

如现有环境尚未装齐仓库依赖，再执行 `python -m pip install -r requirements.txt`。不要在正在运行其他训练的环境中盲目升级 PyTorch。

## 2. 单独导出图数据

```bash
python -u -m vulnmechanism.graph_experiment build \
  --dataset data/function_dataset.jsonl \
  --output data/graph_abc/primevul.jsonl \
  --source-dataset primevul \
  --joern-dir /home/phy/joern \
  --java-home /home/phy/jdk21 \
  --batch-size 8 \
  --timeout 300
```

从当前成功构建的 PrimeVul 记录出发，保留其所有 sample_key、源码、标签与 split。**不重新随机划分，不抽小样本，不读取 test 标签来选特征。** 图需重新导出：旧关系文本不含足够的节点身份/属性，不能恢复原始图。构图不使用 GPU。

导出保存目标函数所有原生节点属性以及现有提取器的 AST/CFG/CDG/DDG 四类边。CFG 不按 token 数截断；文字相同的节点不合并。图特征只使用四类程序属性，不使用标注、CVE、修复信息、候选风险标签或文件名。

新增导出失败的函数也保留，`static_graph=null`、`graph_status=unavailable`，模型使用精确零图向量。这样三组成员一致，不会偷偷只评估容易构图的样本。所有图均失败时拒绝训练。

输出：

```text
data/graph_abc/primevul.jsonl
data/graph_abc/primevul.meta.json
data/graph_abc/primevul.audit.json
data/graph_abc/primevul.errors.jsonl
```

每个样本完成后即时写入，可看到 `graph_done=... status=... cfg_nodes=...`。中断后重复同一命令恢复已完成记录；仅修复失败样本时添加 `--retry-failed`。数据或图 schema 不一致会报错，不能混用缓存。开始训练后不要修改/重建这份图数据；修改后应换新的实验输出目录。

## 3. 顺序跑 A/B/C（仅验证集）

```bash
mkdir -p results/graph_abc
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python -u -m vulnmechanism.graph_experiment run \
  --dataset data/graph_abc/primevul.jsonl \
  --output-dir results/graph_abc \
  --model /home/phy/models/Qwen2.5-Coder-7B-Instruct \
  --source-max-length 2048 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --epochs 3 \
  --learning-rate 2e-4 \
  --seed 42 \
  --device cuda \
  --resume \
  2>&1 | tee -a results/graph_abc/run.log
```

需要用物理 GPU 1 时，把 `CUDA_VISIBLE_DEVICES=0` 改为 `1`。可见设备内部仍使用 `cuda`。

三组都复用 `model.py::train_model`，相同源码输入、LoRA 设置、AdamW、分类损失、累积步数、验证 MCC 选模与验证阈值选择。默认图 embedding 每通道 32 维、5 步、每通道最大词表 2048；词表仅由 train 构造，写入 checkpoint。先保持这些默认值，避免同时更改太多变量。

每组训练与评估分进程运行，上一组退出后释放 GPU。完成后自动输出验证对照表和四种纠错/损坏数量，**不自动评估 test**。

`--resume` 跳过已经完整保存且数据/配置完全一致的 checkpoint。未完成的训练从该组起点重跑，并不是 optimizer 级续训；已保存但在写入实验协议前中断的 checkpoint 会拒绝复用，需另选输出目录，不会擅自覆盖。不要把旧实验 checkpoint 直接放入这个目录冒充 A。

原 baseline 代码和旧 checkpoint 仍可使用；A 重新跑一次，是为了让这次三组数据、训练协议和逐样本预测能够核对。

## 4. 结果位置

```text
results/graph_abc/
  baseline.pt
  graph_attributes.pt
  graph_cfg.pt
  <variant>.training.jsonl
  <variant>.run.json
  <variant>.valid.predictions.jsonl
  <variant>.valid.metrics.json
  valid_comparison.json
  run.log
```

`valid_comparison.json` 包含 Accuracy/Precision/Recall/F1/MCC/AUC、TP/FP/TN/FN，以及相对 A 的：

- `fn_to_tp` / `fp_to_tn`：纠正漏报 / 误报；
- `tp_to_fn` / `tn_to_fp`：新增漏报 / 误报；
- `net_corrected`、`net_fn_reduction`、`net_fp_reduction`。

同时保存各自验证阈值与统一 A 阈值下的翻转统计。共同阈值仅作诊断，不能消除模型分数标度的差异。AUC 另行比较，不以 F1/Recall 上升代替整体进步。

图读取完整函数，源码支路仍限 2048 个源码 tokens（与原 InputBuilder 一致，另含任务前缀和 EOS）。预测文件记录 `source_tokens`、`source_truncated`，指标文件分别报告未截断/截断及图可用/不可用子组。必须区分结构收益与“图看到了源码截断以后的内容”。

只重做汇总：

```bash
python -m vulnmechanism.graph_experiment compare \
  --output-dir results/graph_abc --split valid
```

## 5. 固定方案之后再评估 test

以下命令不重新训练、不选择 test 阈值，使用 checkpoint 中的验证阈值：

```bash
for variant in baseline graph_attributes graph_cfg; do
  CUDA_VISIBLE_DEVICES=0 python -u -m vulnmechanism.graph_experiment eval \
    --dataset data/graph_abc/primevul.jsonl \
    --checkpoint "results/graph_abc/${variant}.pt" \
    --split test --batch-size 1 --device cuda || break
done
python -m vulnmechanism.graph_experiment compare \
  --output-dir results/graph_abc --split test
```

当前流程不支持用 `eval` 直接混入另一份外部图数据；单独外部测试应显式扩展协议，而非绕过成员和文件哈希检查。

单 seed 是开发筛查，不是稳定性证明。没有在真实 Joern + 7B 模型上运行完之前，不能宣称新结构涨点。
