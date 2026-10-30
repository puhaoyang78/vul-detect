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

LOSVER 512/L1的完整[失败原因诊断](results/losver_diagnosis/report.md)包含原始配对核验、定位器泛化、冻结checkpoint加权干预、逐样本案例与证据等级。复现统计和推理使用 `python -m vulnmechanism.losver_diagnosis`，具体命令、固定成员和结果表见报告。

[第二轮修复对与Diff监督审计](results/losver_patch_audit/report.md)区分字段缺失、真实错配与合法重构，保存分层随机复核、原始提交证据、Oracle后验关联及Train候选区域探针。使用 `python -m vulnmechanism.losver_patch_audit` 复核；该入口不加载模型、不训练、不修改原数据或既有结果。

修复关系区域监督与稳健融合使用同一套UniXcoder训练入口，结果独立写入 `results/repair_region_512/`。行标签为 `1=修复相关、0=弱背景、-1=Unknown`，Unknown从Focal分母排除，仍参与编码和函数BCE；没有可见区域正监督的函数只跳过行损失。直接修改语句权重1、可绑定的有限操作/控制关系0.6、未修改结构背景0.2；启发式依赖、邻近候选和未确认代码均为Unknown。Diff-only匹配对照共用有界直接语句映射和相同的masked Focal（alpha=0.5、gamma=1.5、pos_weight=1）；原LOSVER另列参照，保留其原始Diff与Focal。

`robust_fusion`共用检测器一次编码的代码Token状态，以连续行分数聚合局部表示；分数超过0.5的部分归一化，最大超额分数衰减残差，beta上限0.25，没有有效质量时严格使用全局均值。训练beta固定0.25，每轮仅按源Valid P-C从0/0.1/0.25选择推理beta，平分偏向较小beta和较早Epoch；这些分数不被视为校准概率。分类头仍为Linear(768,1)，检测器只使用定位器对单函数的真实预测，不读GT区域。没有OOF、成对损失或额外网络。

```bash
python -m vulnmechanism.cli prepare --repair-regions --raw-root data --output-dir results/repair_region_512
python -m vulnmechanism.cli check-paired --repair-regions --output-dir results/repair_region_512 --device cuda:0
# 在tmux内运行；先定位器、再检测器。默认全量40/16 Epoch、Seed123456、Batch8。
python -m vulnmechanism.cli train --dataset primevul_paired --output-dir results/repair_region_512 --methods diff_localizer repair_region_localizer --device cuda:0 --resume --save-every 2000
python -m vulnmechanism.cli train --dataset primevul_paired --output-dir results/repair_region_512 --methods diff_hard_weight repair_region_hard_weight diff_robust_fusion repair_region_robust_fusion --device cuda:0 --resume --save-every 2000
python -m vulnmechanism.cli evaluate --dataset primevul_paired --output-dir results/repair_region_512 --methods diff_localizer repair_region_localizer diff_hard_weight repair_region_hard_weight diff_robust_fusion repair_region_robust_fusion --split valid --clean-only --device cuda:0
# 方案由源Valid确定后，才执行Test；定位器Test步骤只读取单函数源码。
python -m vulnmechanism.cli evaluate --dataset primevul_paired --output-dir results/repair_region_512 --methods diff_localizer repair_region_localizer diff_hard_weight repair_region_hard_weight diff_robust_fusion repair_region_robust_fusion --split test --clean-only --device cuda:0
python -m vulnmechanism.cli report --dataset primevul_paired --output-dir results/repair_region_512
```

Pilot统一增加 `--train-limit 1024 --valid-limit 256 --paired-epochs 4`（单位为函数，即512/128对，Seed42固定抽样），输出到 `results/repair_region_512/pilot/`；其`supervision`链接到上一级生成结果，不复制原数据。`neighbour_region_localizer/neighbour_region_robust_fusion`使用逐函数相同新增行数量及置信权重的邻近区域监督对照；匹配发生在完整源码，512截断后的可见区域数量可能不同，不能据微小指标差异认定程序关系有效。Pilot不评价Test。正式数据成员、历史checkpoint和两轮审计均不变；权重、源码监督缓存和完整预测源码留在本地。

定位器阶段只训练行级Focal，检测器阶段保留全部函数并训练BCE。区域的“可靠关联”指有界语法绑定及控制关系候选，不是漏洞位置真值或完整数据流证明。弱背景主要为结构分隔符，自己的弱标签F1不能替代真实定位质量。β=0表示融合训练模型在推理时使用全局路径，不等同于独立训练的Baseline。

多β验证会重放相同RNG状态，并仅保留一次Valid遍历的RNG推进，保证下一轮训练采样与单遍验证对照一致。[Pilot协议检查](results/repair_region_512/pilot/protocol-checks.json)记录相同步数、成员、参数量和RNG核验。早期受验证采样影响的三个Pilot检测器已移至`pilot/sampler_mismatch/`保留记录，排除在正式对比之外；当前表使用修正后重跑结果。

完整512实验已完成：两组Localizer各40轮、43,560步，四组新Detector各16轮、17,424步；保留4354/562/564对原始划分，阈值固定0.5。下表均使用564对Test，各模型只通过源Valid选模；完整[报告](results/repair_region_512/report.md)、[Valid/Test对比表](results/repair_region_512/comparison.csv)及[协议核对](results/repair_region_512/protocol-checks.json)已保存。

| 方法 | Test P-C | AUC | MCC | F1 |
|---|---:|---:|---:|---:|
| Global Mean + Linear | 14.54% | 0.5729 | 0.1125 | 0.5815 |
| 原LOSVER参照 | 14.01% | 0.5679 | 0.0889 | 0.5267 |
| Diff-only + Hard Weight（匹配弱损失） | 10.99% | 0.5666 | 0.0826 | 0.4480 |
| Repair-Relation + Hard Weight | 11.88% | 0.5527 | 0.0656 | 0.5357 |
| Diff-only + Robust Fusion | 13.30% | 0.5645 | 0.0922 | 0.5477 |
| Repair-Relation + Robust Fusion | 12.59% | 0.5672 | 0.0946 | 0.5184 |

完整方法未超过Global Mean；其Valid/Test P-C分别为15.12%/12.59%，Valid选择β=0.25。Repair-Relation Localizer的可见Unknown比例为Train 74.53%、Valid 73.62%；33.14%/36.48%的函数没有可见区域正监督，但全部保留函数BCE。自身弱标签F1为0.9724，同时预测了79.11%的Valid行，不能据高F1认定修复定位准确。当前没有证据支持稳定分类收益，未继续增加变体、全覆盖输入或额外模块。权重与含源码的缓存留在本地，不含源码的行概率及检测预测随结果发布。

### Diff 邻接补标（`diff_neighbour`，512 Pilot）

保留原LOSVER二值Diff标签及Focal Loss，仅对完整源码中原始正标签为空的一侧补标。从同次`Differ`的编辑位置向前/后各选至多一条有效代码行，每方向最多跨一行空白或注释；花括号、预处理行和独立分隔符停止搜索。只使用两侧512输入内完整可见的编辑及邻接行，编码相同则不补标。多个位置去重，不作语义或变量扩展；所有函数及分类BCE保留。这些标签是邻接代理，不是漏洞位置真值。

定位器沿用原Focal（α=0.8、γ=1.5、pos_weight=5），按**原始Valid Diff正类F1**选模；补标后的代理指标单列。检测器仍使用预测行、5/1 Softmax和Valid P-C选模，阈值固定0.5。复用已完成的原LOSVER Pilot，与新增实验保持Train 1024/Valid 256函数、每阶段4 Epoch/512步一致；成员抽样Seed42，训练Seed123456。不读取Test。

```bash
python -m vulnmechanism.cli prepare --diff-neighbour --raw-root data --output-dir results/diff_neighbour_512
python -m vulnmechanism.cli train --data-dir data --output-dir results/diff_neighbour_512 --dataset primevul_paired --methods diff_neighbour_localizer diff_neighbour --device cuda:0 --paired-input 512 --train-limit 1024 --valid-limit 256 --paired-epochs 4 --resume --save-every 1000
python -m vulnmechanism.cli evaluate --data-dir data --output-dir results/diff_neighbour_512 --dataset primevul_paired --methods diff_neighbour --device cuda:0 --split valid --clean-only
python -m vulnmechanism.cli report --data-dir data --output-dir results/diff_neighbour_512 --dataset primevul_paired --reference-dir results/repair_region_512/pilot
```

训练应在tmux中执行。统计、40个Train抽查案例、真实GPU检查、两种行标签指标和分类对比保存到[实验目录](results/diff_neighbour_512/)。原始数据、历史结果及checkpoint保持不变；含完整源码的监督缓存和模型权重留在本地。

### 定位器基础正则化与逐行 R-Drop（512 Pilot）

仍使用 `losver_localizer`、原始Diff、原Focal和Valid正类F1选模。默认新增层dropout=0、weight decay=0、单前向保持不变；新增层dropout不修改UniXcoder编码器已有的0.1。`--localizer-rdrop`省略时单前向，显式`0`时两次Focal取平均，`0.1`再加入逐行Bernoulli对称KL；两次logit均保留梯度，padding不参与。`LOSVERAdamW`在原自适应更新后执行decoupled decay，排除全部bias和LayerNorm参数。

A复用原Pilot；B/C使用完全相同的dropout=0.1、weight decay=0.01、两次前向、Train1024/Valid256函数、4 Epoch/512步，仅λ不同。训练Seed123456、成员Seed42、LR2e-5、10% warmup、Focal α=.8/γ=1.5/pos_weight=5、阈值>.5和选模规则不变。分别在tmux会话运行以下B/C命令；未启动检测器或Test。

```bash
# B：基础正则化，双前向，λ=0
python -m vulnmechanism.cli train --data-dir data --dataset primevul_paired --methods losver_localizer --output-dir results/localizer_regularization_512/dropout_weight_decay --device cuda:0 --paired-input 512 --train-limit 1024 --valid-limit 256 --paired-epochs 4 --localizer-dropout .1 --localizer-weight-decay .01 --localizer-rdrop 0 --resume --save-every 1000
# C：与B相同，仅λ=.1
python -m vulnmechanism.cli train --data-dir data --dataset primevul_paired --methods losver_localizer --output-dir results/localizer_regularization_512/line_rdrop --device cuda:1 --paired-input 512 --train-limit 1024 --valid-limit 256 --paired-epochs 4 --localizer-dropout .1 --localizer-weight-decay .01 --localizer-rdrop .1 --resume --save-every 1000
python -m vulnmechanism.cli report --dataset primevul_paired --data-dir data --output-dir results/localizer_regularization_512 --reference-dir results/repair_region_512/pilot --localizer-regularization
```

所选checkpoint的Train/Valid AUPRC、F1、Precision、Recall及预测阳性比例，按全部/漏洞侧/修复侧汇总到[结果目录](results/localizer_regularization_512/)。AUPRC用于诊断排序能力，不替换原F1选模规则；较小的Train–Valid差距应结合Valid绝对性能解读。

### 修改行辅助的联合训练（完整512设置）

仍使用 `unixcoder`：全部有效代码Token等权均值＋原函数Linear分类头。配置 `--line-auxiliary` 后，在同一次encoder输出上对每行代码Token取均值，额外用一个Linear预测原始Diff；目标为函数BCE＋λ×Focal。Focal沿用α=.8/γ=1.5/pos_weight=5，先在函数内按有效可见行平均，再按batch平均。空Diff保留全零行标签和函数BCE。新增头不改变编码器、函数头初始化或采样RNG；默认入口不启用辅助头。

五组均使用完整Train8708/Valid1124函数、512-token原输入、16轮/17424步、Seed123456和原优化器。随机对照用独立的 `random.Random(123456)`，训练前在每个函数的可见行中固定打乱标签并保持正行数量，两个λ共享同一随机标签。Valid始终使用真实Diff。只按Valid P-C选checkpoint和λ，轮次并列取较早轮，λ并列取较小值；固定概率阈值>.5。本轮不读取Test。λ=0组也保留同容量但不学习的辅助头，其行AP仅作诊断。

在tmux内执行以下正式入口命令，可将不同配置分配到不同GPU和会话；每组目录独立，支持断点续训：

```bash
for setting in 'baseline diff 0' 'diff_0.1 diff .1' 'diff_0.3 diff .3' 'random_0.1 random .1' 'random_0.3 random .3'; do
  read -r name supervision weight <<< "$setting"
  python -m vulnmechanism.cli train --data-dir data --dataset primevul_paired --methods unixcoder --output-dir "results/line_auxiliary_512/$name" --device cuda:0 --paired-input 512 --line-auxiliary "$supervision" --line-auxiliary-weight "$weight" --resume --save-every 1000
done
python -m vulnmechanism.cli report --data-dir data --dataset primevul_paired --output-dir results/line_auxiliary_512
```

推理仍直接使用 `infer --checkpoint .../best-p_c.pt --source-file ...`，不读取Diff，不调用辅助行头。五组Valid分类指标、真实Diff行AP、逐样本预测和两项训练损失汇总到[实验目录](results/line_auxiliary_512/)，权重保留本地。

五组已完成全量训练。真实Diff λ=.1/.3的Valid P-C为14.41%/14.23%，低于Baseline的15.48%；修改行AP为.1991/.2294。λ=.1虽提高F1和漏洞Recall，但修复侧误报率从36.83%升至51.78%，MCC和AUC下降。随机λ=.1/.3的P-C为16.01%/13.35%。本轮没有一致的Clean分类收益，未继续叠加模块或读取Test；完整结果见[报告](results/line_auxiliary_512/report.md)。

### 两折 OOF 定位预测（512 Pilot）

复用 `results/repair_region_512/pilot/primevul_paired/losver_localizer/` 的完整 Pilot 定位器及其 Train 1024/Valid 256 函数。Train 按 `pair_key` 用独立 Seed123456 分为两折，修复前后始终同折。`--oof-fold` 指定**不参与该模型训练、只用于预测**的 Train 折；每个定位器只训练另512条函数，跑8轮/512步，匹配完整 Pilot 定位器4轮/512步的优化预算。折模型仍仅按同一源Valid的原始Diff F1选模。数据减半、重复次数及验证次数增加的影响须与OOF作用区分。

在 tmux 中分别运行两折，完成后合并；均支持原断点续训：

```bash
for fold in 0 1; do
  python -m vulnmechanism.cli train --data-dir data --dataset primevul_paired --methods losver_localizer --output-dir "results/losver_oof_512/fold_$fold" --device "cuda:$fold" --paired-input 512 --train-limit 1024 --valid-limit 256 --paired-epochs 8 --oof-fold "$fold" --resume --save-every 1000
done
python -m vulnmechanism.cli prepare --oof-predictions --raw-root data --output-dir results/losver_oof_512 --reference-dir results/repair_region_512/pilot
# 训练内预测对照
python -m vulnmechanism.cli train --data-dir data --dataset primevul_paired --methods losver --output-dir results/losver_oof_512/in_sample --device cuda:0 --paired-input 512 --train-limit 1024 --valid-limit 256 --paired-epochs 4 --localizer-dir results/repair_region_512/pilot/primevul_paired/losver_localizer --resume --save-every 1000
# 只替换Train行预测；Valid和正常单函数推理仍使用同一个完整定位器
python -m vulnmechanism.cli train --data-dir data --dataset primevul_paired --methods losver --output-dir results/losver_oof_512/oof --device cuda:1 --paired-input 512 --train-limit 1024 --valid-limit 256 --paired-epochs 4 --localizer-dir results/repair_region_512/pilot/primevul_paired/losver_localizer --train-localizer-dir results/losver_oof_512/oof_predictions --resume --save-every 1000
python -m vulnmechanism.cli report --data-dir data --dataset primevul_paired --output-dir results/losver_oof_512
```

两个Detector均从同一预训练UniXcoder和Seed123456初始化，维持原Focal、5/1 Softmax、Linear分类头、样本顺序、4轮/512步和固定阈值>.5，仅按Valid P-C选模。本轮不读取Test、不改标签、不增加正则或新模块。成员、折预测来源、原始Diff下行级质量、Token聚合质量及成对分类变化保存到[实验目录](results/losver_oof_512/)，含源码缓存和checkpoint保留本地。

四组Pilot训练已完成，两个Detector均选择第4轮：

| Detector Train行预测 | Valid P-C | AUC | MCC | F1 | 修复侧FPR |
|---|---:|---:|---:|---:|---:|
| 训练内 | 13/128（10.16%） | .5424 | .0825 | .3704 | 20.31% |
| OOF | 9/128（7.03%） | .5348 | .0262 | .3719 | 26.56% |

训练内/OOF/Valid定位AP为.8881/.1288/.1506。OOF的部分预测及聚合分布更接近Valid，但分类P-C、AUC、MCC下降，不能据此认定泛化改善。折模型独立训练样本减半，并在第256/192步选中checkpoint；完整定位器选在第512步。在共同Valid上的折模型F1/AP并未一致明显变差，仍不能单独隔离数据规模与训练内暴露的影响。未继续全量实验；见[完整报告](results/losver_oof_512/report.md)。

### PrimeVul 静态修复模式审计

复用 `losver_patch_audit.py` 与 `syntax.py`，审计原始4354对Train，完成源码复核后再用固定规则检查562对Valid，不读取Test、不改变成员、不训练。配对AST变化只在审计中使用；`syntax.static_relations(source)`的输入只有单个函数源码，输出带UTF-8 byte offset和物理行号的操作—控制条件/词法定义关系。绑定、值写入、字段路径或解析无法确定时拒绝补关系；不做邻近行或同名对象无约束扩展，所有安全性结论保持unknown。

```bash
python -m vulnmechanism.losver_patch_audit patterns --split train --output-dir results/primevul_static_patterns
# 核对train.cases.md并完成train.review.jsonl；固定规则后执行Valid。
python -m vulnmechanism.losver_patch_audit patterns --split valid --output-dir results/primevul_static_patterns
python -m vulnmechanism.losver_patch_audit pattern-report --output-dir results/primevul_static_patterns
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_losver_patch_audit tests.test_surface_vat tests.test_losver -q
```

规则、模式频次、逐对变化、固定抽样源码及复核意见、失败原因和已有512/L1定位预测的比较保存在[审计目录](results/primevul_static_patterns/)。全函数提取缓存留在本地，可由原始数据重建。静态关系成立、与修复相关、命中Diff、真实漏洞定位是不同结论；Diff重叠仅作为弱标签代理，不能报告为漏洞定位准确率。本轮没有接入新的模型或训练监督。

4354对Train和562对固定规则Valid已完成。候选配对覆盖为66.90%/68.33%；最终抽样中与当前修复可见相关的候选为9/56、6/24，不是总体漏洞定位precision。Valid有C++流插入被当整数移位的反例，资源生命周期规则零覆盖。L1原Diff代理F1为静态候选.1246、现有定位器.2196。当前不建议把这些候选直接当作强漏洞行监督；具体分母、Unknown及失败原因见[报告](results/primevul_static_patterns/report.md)。

### Diff引导双版本切片审计

同一审计入口可批量构建Joern CFG/REACHING_DEF/CDG，再映射到Tree-sitter原子语句。修改语句分别做两跳前向/后向切片，每侧最多新增12条语句；通过相同源码行/唯一相同Token语句做一次版本映射，不按邻近行补区域。解析、别名、调用摘要和宏语义无法确认时保留Unknown。

```bash
python -m vulnmechanism.losver_patch_audit slice-graphs --split train --batch-size 256 --output-dir results/primevul_bidirectional_slicing
python -m vulnmechanism.losver_patch_audit slices --split train --output-dir results/primevul_bidirectional_slicing
python -m vulnmechanism.losver_patch_audit slice-report --split train --output-dir results/primevul_bidirectional_slicing
# 完成Train源码复核，固定规则后，对valid依次运行以上三个阶段。
```

Joern路径为`/home/phy/joern`；单任务JVM配置8GB/4核，临时CSV导出使用本机`/tmp`，CPG和原生依赖缓存保留在结果目录。大缓存不纳入Git。此流程需要配对Diff，只用于离线监督审计，不能作为单函数定位器的推理输入；与已有LOSVER的Diff重叠比较不代表真实漏洞定位准确率。

全量4354对Train和562对固定规则Valid已完成。原Diff漏洞侧为空的纯新增组中，候选恢复为657/1232、84/163；两侧图/AST可用配对占54.87%/53.56%。补充穷尽复核旧Valid纯新增组中4个有输出配对的47行候选，26行与可见修复相关、16行无关、5行Unknown，不能外推为全库定位准确率。切片能补回部分旧操作，但新增区域质量尚不足以直接充当强行监督；没有接入模型或训练。完整分母、路径、源码案例、成本和检查见[报告](results/primevul_bidirectional_slicing/report.md)。

同一入口增加 `--strategy targeted`：新增检查必须通过精确绑定/字段路径或原生值依赖关联目标操作，并通过CDG和CFG支配验证；仅回溯目标的数据来源与必要条件。两侧仍最多增加12条语句。保留 `bidirectional` 作为同预算对照，不重建CPG。

```bash
python -m vulnmechanism.losver_patch_audit slices --strategy targeted --split train --output-dir results/primevul_targeted_slicing
# 完成固定train.review_sample.jsonl的源码复核，保存train.review.jsonl。
python -m vulnmechanism.losver_patch_audit slice-report --strategy targeted --split train --output-dir results/primevul_targeted_slicing
# 规则固定后，对valid执行以上两阶段；已讨论样本单列，Test不参与。
```

最小真实验证各处理24对（旧8对加Seed42新16对）。新Train/Valid组目标切片各保留5条相关行，但联合复核池的相关行保留率仅6.3%/7.7%，纯新增恢复为1/16、2/16；无法证明整体质量改善，未继续扩大实验或增加规则。原始成员和模型未改，见[完整报告与逐行复核](results/primevul_targeted_slicing/report.md)。
