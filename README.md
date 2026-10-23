# vul-detect

函数级漏洞二分类。当前代码只保留 CodeBERT 动态困难样本增强及其六组对照，和 ASE 2025 LOSVER 的官方复现。共享数据读写、训练记录、指标与 CLI；不包含静态关系注入、Motivation、跨域或历史成对表示学习分支。

## 环境

```bash
conda activate vul-detect
pip install -r requirements.txt
python -m unittest discover -s tests
```

本机预训练模型：`/home/PublicData/PHY-data/resource/codebert-base`、`/home/PublicData/PHY-data/resource/unixcoder-base-nine`。上游原始数据放在 `/home/PublicData/PHY-data/vul_detect/data`，程序不修改原文件。权重留在本地，不纳入 Git。

## 动态困难样本增强

六组方法：`bce`、`vat`、`random_aug_bce`、`hard_aug_bce`、`random_aug_vat`、`hard_aug_vat`。共享 CodeBERT、mean pooling 和线性分类头，Seed 42、3 Epoch、有效 Batch 8、学习率 1e-5。源函数预算 2048，使用原有原生 512-token 重叠滑窗。仅自身 Clean Valid 选择 Epoch 和阈值。

PrimeVul 使用 v0.1 官方 Train/Valid/Test；MegaVul 使用原生标签、按项目分组划分。保留正例并随机抽取等量负例，记录精确源码泄漏。LOSVER 的原始配对数据使用下面单独说明的官方协议，不与此平衡数据混用。

增强素材只取 Train。保留注释、绑定安全的变量重命名、候选可见性、素材隔离与去重。Hardest 使用原 beam 搜索；可选 Boundary 训练策略，评估始终保留完整最坏情况搜索。每个原函数的双视图 BCE 总权重为 1。VAT 保留 `finite_difference` 默认方向及 `logit` 方向开关。

```bash
python -m vulnmechanism.cli prepare \
  --raw-root /home/PublicData/PHY-data/vul_detect/data
python -m vulnmechanism.cli train --dataset primevul \
  --methods bce vat random_aug_bce hard_aug_bce random_aug_vat hard_aug_vat \
  --device cuda:0
python -m vulnmechanism.cli train --dataset megavul \
  --methods bce vat random_aug_bce hard_aug_bce random_aug_vat hard_aug_vat \
  --device cuda:1
python -m vulnmechanism.cli evaluate --dataset all --device cuda:0
python -m vulnmechanism.cli report
```

训练可指定 `--hard-selection boundary`、`--vat-direction logit`；对应方法分别需要 Hard Aug、VAT。`--train-limit/--valid-limit/--test-limit` 用于固定平衡小样本。Fixed、Held-out、Adaptive 评价区分严格语义保持扰动与函数名等输入压力测试，保存实际查询、完整分母、Clean/Worst-case 指标及 ASR。已有搜索/VAT诊断通过 `ablation probe/report` 使用。

## LOSVER：PrimeVul 官方原始配对数据

来源：[论文 DOI](https://doi.org/10.1109/ASE63991.2025.00092)、[作者代码](https://github.com/waroad/losver)、[PrimeVul 原始发布](https://github.com/DLVulDet/PrimeVul)。对照官方代码提交 `a158905cb45b4400f481f18eeb1cb4473ccfccdc`，许可见 [LOSVER.txt](LICENSES/LOSVER.txt)。

使用 `PrimeVul_original/primevul_{train,valid,test}_paired.jsonl`，预期分别为 **4354/562/564 对**。v0.1 是 **3789/480/435 对**，不重划分或混用两个版本。保留官方相邻 vulnerable/patched 顺序和所有成员，记录跨 Split 源码/编码重复、元数据异常及截断后不可区分数量。Train/Valid 的 Diff 行标签和 Test Oracle 标签与普通输入文件分开；正常定位和检测不打开 Test Oracle 文件。

四个训练方法：

- `unixcoder`：官方 UniXcoder baseline，使用 `RobertaForSequenceClassification` 单输出头。
- `losver_localizer`：官方修改行定位器，8 头行 Token→全函数上下文注意力、拼接原行/上下文均值、原残差分类器；Focal Loss 的 pos_weight=5、alpha=0.8、gamma=1.5。
- `losver`：以定位器预测概率 >0.5 的修改行为依据，对相应 Token 赋 5、其他赋 1，经官方 Softmax 聚合及函数级 BCE。编码器重新从预训练初始化，不加载定位器编码器。
- `losver_oracle`：独立 Oracle 对照，训练和评价使用真实修改行；不作为正常单函数检测结果。

严格沿用官方 `RobertaTokenizer` 与三种 512-token 输入细节：定位器逐行加换行并只保留完整可容纳的行；baseline 保留 510 Token 后添加边界；加权检测器先加边界再截到 512。没有 task prefix、滑窗或额外 UniXcoder mode Token。保留官方加权检测器以聚合向量第 0 维计算 Sigmoid 的函数分类方式，以及 Softmax 前 padding 权重置 0 的行为，不替换分类头或掩码公式。

Seed **123456**、Batch **8**、学习率 **2e-5**、weight_decay **0**、梯度裁剪 **1**、线性 warmup **10%**；全部 FP32、无 AMP。定位器训练 **40 Epoch**，按 Valid 的加权行 F1 选模；baseline/预测修改行/Oracle 各 **16 Epoch**，按 Valid P-C 选模。选模分数按官方保留四位，严格改善才更新。函数概率固定 **>0.5**，不搜阈值。保留 Acc/F1/P-C 三种最佳检测器权重，正式结果使用最佳 P-C。

论文正文的通用设置写检测器 8 Epoch、按 Valid Accuracy 选模；作者发布的 PrimeVul 专用脚本实际默认 16 Epoch，Test 加载最佳 Valid P-C 权重。本仓库采用该专用脚本协议；这项论文与源码的差异保留在复现记录中，不依据 Test 更换选模口径。

```bash
python -m vulnmechanism.cli prepare --paired --datasets primevul \
  --raw-root /home/PublicData/PHY-data/vul_detect/data
python -m vulnmechanism.cli train --dataset primevul_paired \
  --methods losver_localizer --device cuda:0
python -m vulnmechanism.cli train --dataset primevul_paired \
  --methods unixcoder --device cuda:1
# 定位器完成后，使用其预测文件训练正常 LOSVER。
python -m vulnmechanism.cli train --dataset primevul_paired \
  --methods losver --device cuda:0
python -m vulnmechanism.cli train --dataset primevul_paired \
  --methods losver_oracle --device cuda:1
python -m vulnmechanism.cli evaluate --dataset primevul_paired \
  --methods unixcoder losver losver_oracle --clean-only --device cuda:0
python -m vulnmechanism.cli report --dataset primevul_paired
python -m vulnmechanism.cli infer \
  --checkpoint results/primevul_paired/losver/best-p_c.pt \
  --source-file example.c --device cuda:0
```

单函数 LOSVER 推理同时需要选定检测器和 `localizer_path` 指向的定位器，读取单个函数，先定位再检测；Oracle 不提供无 Diff 的普通推理。

以上命令记录本轮使用的默认输出位置。仓库包含已完成的训练记录；复跑时为 train/evaluate/report 统一指定一个空的 `--output-dir`，推理也指向该目录中的权重。训练入口拒绝已有训练记录，避免重复启动或覆盖。

`data/primevul_paired/statistics.json` 保存版本与输入检查。`results/primevul_paired/` 保存统一表、报告和运行检查；各方法目录保存配置、Epoch/step 记录、Valid/Test 预测和本地权重。P-C/P-V/P-B/P-R 分别为两侧都判对、两侧均判漏洞、两侧均判安全、两侧反向判错，四者总和为 1。论文 Table 3 的 baseline/LOSVER/Oracle P-C 参考值为 10.11%/14.01%/33.16%。

运行环境与作者环境不同：当前 Torch 2.9.1、Transformers 4.57.3，作者使用 Torch 2.4.1、Transformers 4.19.0。保留 eager attention 和旧 Transformers AdamW 的更新公式。定位器仅批量计算同一函数的独立 Attention 查询，参数、注意力方程、行边界及 dropout=0 不变；真实前向/损失/梯度差异和耗时检查写入 `checks.json`。真实指标由运行后生成的报告给出，不使用论文数值替代复现结果。

本轮正式 Test 为全部 564 对、1128 个函数；三组均使用各自最佳 Valid P-C 权重。

| 模型 | P-C | Accuracy | Precision | Recall | F1 | 论文 P-C |
|---|---:|---:|---:|---:|---:|---:|
| UniXcoder baseline | 10.64% | 53.10% | 53.33% | 49.65% | 51.42% | 10.11% |
| LOSVER：预测修改行 | 14.36% | 55.23% | 54.66% | 61.35% | 57.81% | 14.01% |
| Oracle：真实修改行 | 31.91% | 64.80% | 60.94% | 82.45% | 70.08% | 33.16% |

正常 LOSVER 相对 baseline 的 P-C 提高 3.72 个百分点、Accuracy 提高 2.13 个百分点。与论文的 P-C 差值分别为 +0.53、+0.35、−1.25 个百分点，复现了改善方向，但数值未完全一致。完整 P-C/P-V/P-B/P-R、Valid 选模、数据与实现差异见[实验报告](results/primevul_paired/report.md)；[逐组指标](results/primevul_paired/comparison.csv)和[真实运行检查](results/primevul_paired/checks.json)一并保存。34 项单元测试通过，三组预测的成员、分类指标及成对计数已独立复核。
