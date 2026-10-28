# vul-detect

函数级漏洞二分类。当前代码保留 CodeBERT 动态困难样本增强及其六组对照，以及 ASE 2025 LOSVER 的复现与实现修正。共享数据读写、训练记录、指标与 CLI；不包含静态关系注入、Motivation、跨域或历史成对表示学习分支。

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

## LOSVER：统一512输入与L1全覆盖窗口

参考 [LOSVER论文](https://doi.org/10.1109/ASE63991.2025.00092)及[作者源码](https://github.com/waroad/losver)，在现有 data/model/training/evaluation 和CLI中实现。历史源码忠实复现、修正版512和L1的结果分别保留；输入和分类头修正后的结果不称为论文数值的一比一复现。

数据保持 `PrimeVul_original/primevul_{train,valid,test}_paired.jsonl` 的 **4354/562/564对**，不重划分、不重采样。`difflib.Differ` 非空删除行规则不变。训练/Valid行标注及Test Oracle标注分开存储。正常定位和单函数检测不读取Test Diff；Oracle只通过明确的Oracle评价路径使用真实修改行。

所有方法使用 `RobertaTokenizerFast(trim_offsets=False)` 对原始源码分词一次，用 `offset_mapping` 共享Token→物理行映射。行号从0开始，保留空行编号间隔，支持CRLF、末行及跨行Token。BOS/EOS、Padding及纯空白Token不参加最终聚合。

通过 `--paired-input` 区分两种输入规则：

- `512`：BOS + 前510个源码Token + EOS，右侧Padding到512。保留截断末行的可见部分，部分/完整修改行覆盖分别统计。
- `l1`：短函数（不超过510个源码Token）保持完全相同的单窗口。长函数按384个核心Token划分，左右各附最多63个上下文Token，每窗口含BOS/EOS不超过512；没有函数总Token截断上限。所有源码Token恰属一个核心区域，上下文副本不重复参与聚合。

Localizer保留原8头MHA、MLP和Focal Loss（pos_weight=5、alpha=0.8、gamma=1.5）。MHA仅在窗口内计算；跨窗口代码行按核心代码Token数量合并表示，再对每个原始物理行分类一次。Baseline、LOSVER和Oracle共用 `RobertaModel(add_pooling_layer=False)`、768维函数表示和 `Linear(768,1)`；仅修改行二值5/1权重不同。检测器对全部核心代码Token执行一次函数级Softmax加权聚合，复用现有原生窗口函数聚合。Baseline等价于这些Token的均值。

定位器按源Valid **修改行正类F1** 选模并记录Precision/Recall；检测器按源Valid **P-C** 选模，均固定概率 **>0.5**。Seed **123456**、有效函数Batch **8**、学习率 **2e-5**、weight_decay **0**、裁剪 **1**、warmup **10%**、原Transformers4.19 AdamW、FP32不变；Localizer40 Epoch，三个检测器各16 Epoch。

L1默认函数Micro-batch1、窗口Micro-batch1、累积至8个函数再更新。Localizer保留原行平均Focal目标，检测器保留函数平均BCE；末组不足8个函数按实际成员归一化。窗口内编码和上下文计算使用保留Dropout RNG的梯度重计算。没有超长函数/窗口丢弃或自动缩短输入；资源错误直接报告。可用 `--window-micro-batch`、`--function-micro-batch` 调整显存，两者不改变函数级预算。

所有正式训练（CodeBERT六方法、LOSVER的512/L1四组）都会原子保存 `last.pt`：包括模型、优化器、学习率调度器（适用时）、Python/NumPy/PyTorch/CUDA随机状态、当前Epoch和批次、累计统计及Valid选模记录。初始化、首个优化器步、之后每100步、每轮训练结束和Valid选模结束保存；可用 `--save-every N` 调整间隔。只在梯度累积完成后保存，恢复到最近成功保存的更新，尚未保存的步骤会重算。

在原训练命令追加 `--resume`，保持数据、模型、训练参数、运行环境和GPU编号一致即可续训。已完成的方法直接跳过，尚未开始的方法正常从预训练初始化，因此同一串行命令可以重复运行。恢复时截去断点之后的候选/指标日志，重建同一Epoch的采样顺序，再恢复模型随机状态；不会从选模用的 `best*.pt` 猜测优化器状态。旧实验缺少 `last.pt` 时不能精确续训。定位器完成训练及源码预测导出后才标记完成；`best*.pt` 继续只用于选模、评价和推理。

续训测试见 `tests/test_resume.py`，真实预训练模型GPU检查见 [resume.smoke.json](results/losver_l1_checks/resume.smoke.json)。GPU检查使用确定性算子验证恢复状态；正式训练保留原CUDA设置。默认CUDA算子在相同权重和随机状态下重复前后向也存在数值差异，保存随机状态不代表默认GPU计算可逐位重现。

```bash
# 从原始文件准备成员及Diff标注；已有data/primevul_paired时无需重建。
python -m vulnmechanism.cli prepare --paired --datasets primevul \
  --raw-root /home/PublicData/PHY-data/vul_detect/data

python -m unittest discover -s tests -v
python -m vulnmechanism.cli check-paired --paired-input 512 \
  --output-dir results/losver_512_checks --device cuda:0
python -m vulnmechanism.cli check-paired --paired-input l1 \
  --output-dir results/losver_l1_checks --device cuda:0

# 同一正式入口。先完成定位器及源码预测导出，再训练三个检测器。
python -m vulnmechanism.cli train --dataset primevul_paired --paired-input 512 \
  --methods losver_localizer unixcoder losver losver_oracle \
  --output-dir results/losver_512 --device cuda:0
python -m vulnmechanism.cli train --dataset primevul_paired --paired-input l1 \
  --window-micro-batch 1 --function-micro-batch 1 \
  --methods losver_localizer unixcoder losver losver_oracle \
  --output-dir results/losver_l1 --device cuda:0 --resume

# 对两种输入各自运行；只使用各自源Valid选定的checkpoint。
python -m vulnmechanism.cli evaluate --dataset primevul_paired \
  --methods unixcoder losver losver_oracle --output-dir results/losver_l1 --device cuda:0
python -m vulnmechanism.cli report --dataset primevul_paired --output-dir results/losver_l1
```

[512检查](results/losver_512_checks/report.md)与[L1检查](results/losver_l1_checks/report.md)分别保存覆盖率、相同输入配对数、GPU显存/时间、梯度、重载与单函数推理检查。两个检查使用完全相同的Train/Valid样本与初始化；额外调用正式训练循环完成18个Train函数、8个Valid函数的一轮Pilot，共3个优化步，覆盖末组2函数情况。这些小样本数值只验证流程，不能证明分类提升。

本轮正式训练使用 `results/losver_512/` 和 `results/losver_l1/`，历史权重不覆盖。运行日志保存在各目录的 `train.gpu0.log`/`train.gpu1.log`，逐Epoch指标在 `primevul_paired/<method>/best.training.jsonl`。完整训练尚未结束时，不把中间Epoch或上述Smoke当作最终Valid/Test结果。前阶段输入修正检查保留在 [results/losver_checks](results/losver_checks/report.md)。

历史源码复现实验位于 [results/primevul_paired](results/primevul_paired/report.md)，对应提交 `12c4c337916623c12b49b934fa7149fb5a5ff6df`：

| 修正前模型 | Test P-C | Accuracy | F1 |
|---|---:|---:|---:|
| UniXcoder baseline | 10.64% | 53.10% | 51.42% |
| LOSVER：预测修改行 | 14.36% | 55.23% | 57.81% |
| Oracle：真实修改行 | 31.91% | 64.80% | 70.08% |

上述指标及历史权重原样保留，属于旧映射、旧分类头和旧选模协议。当前实现拒绝将旧 checkpoint 或缺少物理行号的预测文件用于修正后的实验；没有保留旧模型实现或自动兼容分支。

修正版512和L1的四组训练及检测器Valid/Test评价均已完成，完整结果见 [512正式结果](results/losver_512/primevul_paired/report.md)与[L1正式结果](results/losver_l1/primevul_paired/report.md)。下表使用相同的564对Test，各模型仅按自身Valid P-C选模，固定概率>0.5。

| 输入 | 模型 | Test P-C | Accuracy | F1 | AUC |
|---|---|---:|---:|---:|---:|
| 512 | UniXcoder Baseline | 14.54% | 55.59% | 58.15% | 0.5729 |
| L1 | UniXcoder Baseline | 15.96% | 54.43% | 50.00% | 0.5627 |
| 512 | LOSVER：预测修改行 | 14.01% | 54.43% | 52.67% | 0.5679 |
| L1 | LOSVER：预测修改行 | 14.01% | 54.79% | 58.13% | 0.5701 |
| 512 | Oracle：真实修改行 | 29.96% | 63.12% | 66.07% | 0.7028 |
| L1 | Oracle：真实修改行 | 42.02% | 69.15% | 72.12% | 0.7423 |

L1消除了函数截断并提高Oracle的P-C，但预测修改行LOSVER的Test P-C未超过L1 Baseline，且与512 LOSVER相同，不能认定全覆盖带来稳定的实际检测收益。L1定位器选中第8轮，Valid修改行正类F1为21.96%；512为28.05%，两者可见行范围不同。结果目录保存配置、逐Epoch日志、选模记录、指标及逐样本预测，权重和含原始源码的定位缓存保留在本地。
