# vul-detect

函数级漏洞二分类。当前代码保留 CodeBERT 动态困难样本增强及其六组对照，以及 ASE 2025 LOSVER 的复现与实现修正。共享数据读写、训练记录、指标与 CLI；不包含静态关系注入、Motivation、跨域或历史成对表示学习分支。

正式方法为 **RGCL（Repair-Guided Contextual Learning，即 Context + Diff）**：UniXcoder、512 Token、原始 Context 有界 MIL、K=5、0.1 倍 Token 加权 Diff 区域损失和函数 BCE。正式三组对照为 `plm_baseline`（CLS + `RobertaClassificationHead`）、`context_only`、`rgcl`；历史 Global Mean 结果保留。PrimeVul 按完整 Valid P-C、SecVulEval 按完整 Valid MCC 选模，阈值严格大于 0.5，统一 Seed42、1234、123456、Batch8、16 Epoch 和 AdamW lr=2e-5。

`python -m vulnmechanism.cli rgcl-study inventory` 核对并复用完整历史记录；`rgcl-study check --device cuda:0` 执行真实 GPU 检查；`rgcl-study run --dataset primevul_paired --seed 42 --device cuda:0` 补齐指定组合（SecVulEval 使用 `--dataset secvuleval`）；`rgcl-study report` 汇总三种子 Valid/Test。配置、复用来源和完整结果见[正式对照报告](results/rgcl_multiseed_512/report.md)。Seed123457、Patch 和其他历史方法不计入本轮均值。

ReposVul 使用 `rgcl-study inspect --dataset reposvul --raw-root /home/PublicData/PHY-data/vul_detect/data/reposvul --official-dir data/rgcl/reposvul_official --output-dir results/rgcl_reposvul_512` 检查原始标签、修复对应、512 可见区域和官方划分重叠；命令前缀为 `python -m vulnmechanism.cli`。官方归档链接和本地目录见[数据检查报告](results/rgcl_reposvul_512/report.md)。当前版本存在源码相同但标签相反及官方划分泄漏问题，本轮停在数据质量检查阶段。`rgcl-study check --dataset reposvul --official-dir data/rgcl/reposvul_official --output-dir results/rgcl_reposvul_512 --device cuda:0` 仅用官方 Train 源码验证冻结历史模型推理。

CleanVul 使用评分≥3的全部 C/C++ 发布记录，沿用修复前=1、修复后=0；解析、函数名及词法变化检查仅作诊断，不删除分类成员。`rgcl-study prepare --dataset cleanvul --seed 42 --raw-root /home/PublicData/PHY-data/vul_detect/data/cleanvul` 构建固定分组划分。`rgcl-study check/run/report --dataset cleanvul --output-dir results/rgcl_cleanvul_512` 复用同一三种子训练、Valid MCC 选模和单函数评价入口；`run` 需分别指定 `--seed 42/1234/123456`。数据、监督覆盖率及完整结果见[CleanVul报告](results/rgcl_cleanvul_512/report.md)。

CodeBERT 与 CodeT5 的补充三组对照复用相同数据成员和训练器：`rgcl-study check --backbone codebert --dataset all --output-dir results/plm_baselines_512/codebert --device cuda:0` 检查三个数据集；`rgcl-study run --backbone codebert --dataset primevul_paired --seed 42 --output-dir results/plm_baselines_512/codebert --device cuda:0` 运行指定种子、数据集的 CLS/Context-only/RGCL 三组；`rgcl-study report --backbone codebert --dataset all --output-dir results/plm_baselines_512/codebert` 汇总三种子。CodeT5 将 backbone 和结果目录改为 `codet5`。CodeBERT 沿用 CLS + RoBERTa 分类头；CodeT5 使用[官方缺陷检测读出](https://github.com/salesforce/CodeT5/blob/main/CodeT5/models.py)的同函数右移解码器与末尾 EOS 表示、双类别线性头，使用函数 BCE，不计算生成损失。CodeT5 Context-only/RGCL 直接使用同一次前向的解码器 Token 表示进行原有物理行聚合；不额外编码。Context-only 与 RGCL 仅相差原始 0.1 倍 Token 加权 Diff 区域损失，K=5。两个骨干 × 三个数据集 × 三种子 × 三种方法，共54组。三个数据集均固定 512 Token、Batch8、16 Epoch、AdamW lr=2e-5、Seed42/1234/123456、严格 >0.5 阈值，选模规则与正式 UniXcoder 对照一致；不修改 RGCL 或历史实验。

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

### 补丁引导的有界注意力MIL

直接扩展现有 `unixcoder`，默认Global Mean行为不变。`--bounded-attention`在同一次编码结果上按物理行求均值，增加零初始化 `Linear(768,1)`：`r_i=1+4*sigmoid(s_i)`，`a_i=n_i*r_i/sum(n_j*r_j)`；直接求加权和，不对倍率做Softmax。函数分类仍使用原线性头。等倍率等价于有效代码Token均值，每Token权重比不超过5。

`--attention-region diff`增加固定 `.1*relu(log(2)-log(mean_R)+log(mean_B))`，区域内外均按Token数求平均，按有效函数归一。只取原始双侧Diff的可见行；R为空或覆盖全部Token只跳过区域项，不跳过函数BCE，不补邻接或切片标签。`random`在Train内部使用独立RNG固定打乱行标签，保持正行数量和监督可用性，允许与真实集合偶然相同；Token覆盖率单独报告，不声称严格匹配。

四组均使用完整Train8708/Valid1124函数，Seed123456、512输入、16轮/17424步、Batch8及原学习率/优化器。从同一预训练模型初始化，仅按完整Valid P-C选epoch，平局取较早轮；固定概率>.5。预测与输入完全不读取Diff或配对版本，Valid编辑区域只在预测后用于诊断。没有独立Localizer、Top-k、第二函数分类头或其他辅助任务。

```bash
python -m unittest tests.test_bounded_attention tests.test_line_auxiliary tests.test_losver tests.test_resume
python -m vulnmechanism.cli check-paired --bounded-attention --output-dir results/bounded_attention_512 --device cuda:0
# 在独立tmux会话执行，每个目录只启动一次；--resume复用自身断点。
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --output-dir results/bounded_attention_512/global_mean --device cuda:0 --paired-input 512 --paired-epochs 16 --resume --save-every 1000
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --output-dir results/bounded_attention_512/attention_bce --device cuda:0 --paired-input 512 --paired-epochs 16 --bounded-attention --resume --save-every 1000
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --output-dir results/bounded_attention_512/diff_region --device cuda:1 --paired-input 512 --paired-epochs 16 --bounded-attention --attention-region diff --resume --save-every 1000
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --output-dir results/bounded_attention_512/random_region --device cuda:1 --paired-input 512 --paired-epochs 16 --bounded-attention --attention-region random --resume --save-every 1000
# 四组完整结束后，执行所选checkpoint的Train诊断及C的统一倍率推理干预。
python -m vulnmechanism.cli report --bounded-attention --dataset primevul_paired --output-dir results/bounded_attention_512 --device cuda:1
```

四组均已实际完成16轮；以下为各自Valid P-C最佳checkpoint在同一562对上的结果，阈值固定为概率>.5：

|配置|Epoch|P-C|AUC|MCC|F1|
|---|---:|---:|---:|---:|---:|
|A Global Mean|13|16.37%|0.5726|0.1141|56.99%|
|B 有界注意力+BCE|14|15.66%|0.5856|0.1304|45.95%|
|C 真实Diff区域约束|12|16.01%|0.5873|0.1283|63.26%|
|D 随机区域约束|13|15.30%|0.5775|0.1013|58.66%|

C未超过A的P-C，未满足预设净分类收益标准。C漏洞Recall为75.98%、修复侧FPR为64.23%，A分别为58.72%、47.33%；不能将F1提高直接解释为成对判别改善。

同一C最佳checkpoint将倍率全部固定为1后，Valid P-C为11.03%、AUC为0.5835：注意力确实影响分类，但完整方案仍未超过独立训练的A。C的编辑区域/背景平均倍率为Train 2.97、Valid 1.76；关注量提高与分类收益分开报告。

实际启动命令及GPU分配保存在 `results/bounded_attention_512/commands.json`。整体/纯新增Valid分类、Train/Valid关注量、随机区域重合及C同checkpoint的 `r_i=1` 干预见[实验报告](results/bounded_attention_512/report.md)；不读取Test或根据结果扩大搜索。零初始化使第一步纯区域梯度对encoder为0，关注头更新后正常进入encoder；函数BCE从第一步训练。模型权重留本地，不纳入Git。

### 有界注意力的独立正则化对照

复用 `bounded_attention + diff_region` 的512输入、K=5、原始双侧Diff和 `.1*L_region`。`--attention-suppression random|high` 在一次编码后计算正常/抑制两次聚合，两项函数BCE取平均；每函数只将 `floor(.1*N)` 条有效行的倍率置1，保留全部代码和原区域损失。high按detach后的倍率排序，平局按物理行；random使用独立可恢复的Python RNG。没有额外Softmax、表示置零或放大。

`--attention-ema` 独立复制encoder和attention head作为eval/no_grad参照，添加 `.1*KL(a_ema || a_online)`，有效行求和后按函数平均。EMA在每个实际optimizer.step后以.99更新一次，随 `last.pt` 保存恢复；不参与推理，也不复制函数分类头。抑制与EMA分别实验，不组合。

四组都添加 `--attention-diagnostics`：固定Train/Valid各128对，每轮eval记录编辑区域/背景倍率、相邻epoch注意力JSD、相对Token均值分布的JSD及倍率CV。诊断保留训练RNG，不参与选模；均匀参照是 `n_i/Σn_j`，不是每行等权。完整官方Train/Valid、Seed123456、16轮、17424步、Batch8、原优化器和Valid P-C选模保持一致，阈值为概率>.5。本轮不读取Test。

```bash
python -m unittest tests.test_bounded_attention tests.test_line_auxiliary tests.test_losver tests.test_resume
python -m vulnmechanism.cli check-paired --bounded-attention --attention-regularization --output-dir results/attention_regularization_512 --device cuda:0
# 每组在独立tmux会话执行，训练输出目录互不覆盖。
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --paired-input 512 --paired-epochs 16 --bounded-attention --attention-region diff --attention-diagnostics --output-dir results/attention_regularization_512/diff_region --device cuda:0 --resume --save-every 1000
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --paired-input 512 --paired-epochs 16 --bounded-attention --attention-region diff --attention-diagnostics --attention-suppression random --output-dir results/attention_regularization_512/random_suppression --device cuda:0 --resume --save-every 1000
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --paired-input 512 --paired-epochs 16 --bounded-attention --attention-region diff --attention-diagnostics --attention-suppression high --output-dir results/attention_regularization_512/high_suppression --device cuda:1 --resume --save-every 1000
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --paired-input 512 --paired-epochs 16 --bounded-attention --attention-region diff --attention-diagnostics --attention-ema --output-dir results/attention_regularization_512/ema_attention --device cuda:1 --resume --save-every 1000
python -m vulnmechanism.cli report --dataset primevul_paired --attention-regularization --output-dir results/attention_regularization_512 --device cuda:1
```

实际命令、检查、逐轮诊断、最佳checkpoint完整Train/Valid关注量和分类结果保存在 `results/attention_regularization_512/`。旧对照环境及预算核验后，因仅保留最佳/最后权重、不能补齐相邻epoch诊断，四组统一重跑。关注量或稳定性变化不能单独证明分类改善；最终以[完整实验报告](results/attention_regularization_512/report.md)的真实分类指标为准。

四组均已实际完成16轮/17424步；同一完整Valid562对上的结果如下：

|配置|Epoch|P-C|AUC|MCC|F1|漏洞Recall|修复侧FPR|
|---|---:|---:|---:|---:|---:|---:|---:|
|当前diff_region|12|16.01%|0.5873|0.1283|63.26%|75.98%|64.23%|
|random抑制|12|15.12%|0.5834|0.1144|59.60%|65.48%|54.27%|
|high抑制|14|17.26%|0.5853|0.1107|53.53%|51.25%|40.21%|
|仅EMA|16|16.90%|0.5845|0.1220|58.70%|62.46%|50.36%|

high、EMA的P-C分别多判对7对、5对，但三种新正则的AUC、MCC均下降，召回与误报也同时下降。本轮没有观察到一致的分类泛化收益，不追加组合、模块或参数搜索。新对照16轮的六项分类指标与旧实验逐轮完全一致；相关检查为52项测试加1项诊断隔离测试，恢复配置校验另行复验通过，且全部实验未读取Test。

固定Valid成员、同轮对齐的第2–16轮平均相邻注意力JSD：对照0.008533、random 0.009114、high 0.010566、EMA 0.006407。EMA降低约24.9%，但分类AUC/MCC未改善；不能以JSD下降单独宣布成功。完整最佳checkpoint的Train/Valid编辑区域倍率及纯新增子集结果见报告。

### 上下文条件化有界注意力

在现有 `--bounded-attention` 上增加 `--attention-context`。同一次编码中，`g` 为有效代码Token的全局均值，`h_i` 为原物理行表示；分数为 `s_i=W(h_i)+dot(U(h_i),V(g))/sqrt(32)`。U/V均为无偏置的32维投影。W保持零初始化，U随机、V零初始化，新增初始化不推进CPU/CUDA采样RNG。初始输出与旧有界注意力一致；V首步有梯度，随后U及编码器可学习。保持1～5倍倍率、原MIL聚合、函数BCE及固定 `.1*L_region`，不叠加抑制或EMA。

关闭开关时不新增参数或checkpoint字段，旧权重仍可严格加载。新checkpoint自动恢复上下文分支，`infer`仍只接收单函数源码。Padding不参与g，跨行Token在行聚合中守恒且在g中只计一次。

三组使用PrimeVul original官方完整Train8708/Valid1124函数、512输入、Seed123456、16轮/17424步和原优化器。只按完整Valid P-C选模，四位小数严格改善、平局取较早轮，概率阈值固定>.5。真实/随机Diff标签分别复用旧实验的相同定义与独立随机种子，不读取Test、不搜索超参数。

```bash
python -m unittest tests.test_bounded_attention tests.test_line_auxiliary tests.test_losver tests.test_resume
python -m vulnmechanism.cli check-paired --bounded-attention --attention-context --output-dir results/attention_context_512 --device cuda:0
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --output-dir results/attention_context_512/context_bce --device cuda:0 --paired-input 512 --paired-epochs 16 --bounded-attention --attention-context --resume --save-every 1000
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --output-dir results/attention_context_512/context_diff --device cuda:1 --paired-input 512 --paired-epochs 16 --bounded-attention --attention-context --attention-region diff --resume --save-every 1000
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --output-dir results/attention_context_512/context_random --device cuda:0 --paired-input 512 --paired-epochs 16 --bounded-attention --attention-context --attention-region random --resume --save-every 1000
python -m vulnmechanism.cli report --dataset primevul_paired --bounded-attention --attention-context --output-dir results/attention_context_512 --reference-dir results/bounded_attention_512 --device cuda:0
```

58项相关单元测试、非等长跨行Token均值复验、真实GPU梯度/交互/Padding/保存恢复/源码单函数检查及正式CLI最小训练通过。真实旧checkpoint单函数概率与原预测最大差为1.2e-7；新分支初始输出差为0。实际命令与检查见[实验目录](results/attention_context_512/)，权重保留本地。

三组均实际完成16轮，按完整Valid P-C选出的checkpoint如下。倍率为原始Diff区域/背景的Token加权平均倍率之比，再对有效函数平均；Train/Valid有效函数分别为5180/8708、658/1124。

|配置|Epoch|P-C|AUC|MCC|F1|Recall|修复侧FPR|Train/Valid倍率|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|原非Context真实Diff|12|16.01%|0.5873|0.1283|63.26%|75.98%|64.23%|2.97/1.76|
|Context+BCE|16|15.48%|0.5759|0.0982|53.10%|51.07%|41.28%|1.53/1.34|
|Context+真实Diff|14|17.79%|0.5896|0.1431|60.70%|66.37%|52.31%|3.15/1.78|
|Context+随机Diff|15|16.37%|0.5754|0.1014|54.95%|54.80%|44.66%|1.27/1.16|

在该固定Seed的Valid上，Context+真实Diff较Context+随机Diff多判对8对，较原真实Diff多10对，P-C分别提高1.42、1.78个百分点；也高于原Global Mean的16.37%。相对原真实Diff，AUC/MCC上升，F1/Recall下降、修复侧FPR下降。Context+BCE的P-C、AUC、MCC均低于原attention_bce，条件化本身未显示普遍收益。三组与旧对照的完整成员、环境、输入覆盖、最终CPU采样及CUDA dropout RNG、真实/随机Diff掩码均核对一致；没有读取Test或追加搜索。全部七组对照、逐轮指标及完整关注统计见[报告](results/attention_context_512/report.md)。

### 局部—全局双尺度有界注意力

在 `--bounded-attention --attention-context` 上增加 `--attention-local`。复用当前物理行表示 `h_i` 和32维U/V，候选邻居为物理行号距离1至3且有有效代码Token的行。邻居权重为 `softmax(dot(U(h_i),U(h_j))/sqrt(32))`，局部表示为邻居行表示的加权和 `l_i`；无邻居时为0。分数扩展为 `W(h_i)+dot(U(h_i),V(g)+T(l_i))/sqrt(32)`，仅增加无偏置 `T(768,32)` 的24576个参数。T零初始化，保持原参数、采样RNG及初始输出；关闭开关时不增加参数或checkpoint字段，原Context checkpoint可严格加载。推理从checkpoint恢复开关，仅输入单函数源码。

沿用512输入、完整官方Train8708/Valid1124函数、Seed123456、16轮、Batch8、原优化器、函数BCE及固定 `.1*L_region`，只按完整Valid P-C选模、固定概率>.5。不按Diff筛选邻居。真实Diff与固定随机Diff分别训练，已有Context三组作为对照。

```bash
python -m unittest tests.test_bounded_attention tests.test_line_auxiliary tests.test_losver tests.test_resume
python -m vulnmechanism.cli check-paired --bounded-attention --attention-context --attention-local --output-dir results/attention_local_512 --device cuda:0
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --paired-input 512 --paired-epochs 16 --bounded-attention --attention-context --attention-local --attention-region diff --output-dir results/attention_local_512/dual_diff --device cuda:0 --resume --save-every 1000
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --paired-input 512 --paired-epochs 16 --bounded-attention --attention-context --attention-local --attention-region random --output-dir results/attention_local_512/dual_random --device cuda:1 --resume --save-every 1000
python -m vulnmechanism.cli report --dataset primevul_paired --bounded-attention --attention-context --attention-local --output-dir results/attention_local_512 --reference-dir results/attention_context_512 --device cuda:0
```

实际命令、相关测试、GPU smoke、完整分类指标、所选checkpoint的Train/Valid编辑区域倍率和局部交互统计保存在[双尺度实验报告](results/attention_local_512/report.md)。局部贡献通过T参数范数、局部分数及同一行表示上移除局部项后的倍率变化记录；这些诊断不参与选模。Test不读取，权重留本地。

两组均已完成16轮/17424步。双尺度真实Diff的最佳轮为13，Valid P-C为16.37%、AUC为0.5827、MCC为0.1238；随机Diff的最佳轮为14，P-C为14.95%。真实Diff超过随机Diff，但相对原 `context_diff`（P-C 17.79%）少判对8对，P-C下降1.42个百分点，AUC/MCC也下降，修复侧FPR由52.31%升至57.47%。本轮未验证双尺度的净分类收益。

### 函数自适应上下文调节（context_scale）

在原 `--bounded-attention --attention-context` 上增加 `--attention-context-scale`，与局部双尺度互斥。`g`仍为有效代码Token均值，`alpha(g)=2*sigmoid(Linear(LayerNorm(g)))`，LayerNorm无可训练参数，Linear输出1维，权重与偏置均零初始化。仅新增769个参数，alpha初始严格为1，保留原参数、初始化输出及采样RNG。分数为 `W(h_i)+alpha(g)*dot(U(h_i),V(g))/sqrt(32)`，有界MIL、函数BCE及固定 `.1*L_region` 不变。

关闭开关不新增checkpoint字段或参数，已有Context checkpoint仍可严格加载；单函数推理从checkpoint恢复配置，Diff只用于训练损失及预测后的区域统计。两组采用完整官方Train8708/Valid1124函数、512输入、Seed123456、16轮/17424步、原优化器和固定概率>.5；仅按完整Valid P-C选模，不读取Test、不搜索参数。

```bash
python -m unittest tests.test_bounded_attention tests.test_line_auxiliary tests.test_losver tests.test_resume
python -m vulnmechanism.cli check-paired --bounded-attention --attention-context --attention-context-scale --output-dir results/context_scale_512 --device cuda:0
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --paired-input 512 --paired-epochs 16 --bounded-attention --attention-context --attention-context-scale --attention-region diff --output-dir results/context_scale_512/context_scale_diff --device cuda:0 --resume --save-every 1000
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --paired-input 512 --paired-epochs 16 --bounded-attention --attention-context --attention-context-scale --attention-region random --output-dir results/context_scale_512/context_scale_random --device cuda:1 --resume --save-every 1000
python -m vulnmechanism.cli report --dataset primevul_paired --bounded-attention --attention-context --attention-context-scale --output-dir results/context_scale_512 --reference-dir results/attention_context_512 --device cuda:0
```

报告复用已有三组Context对照，统计完整Train/Valid的alpha均值、总体标准差、分位数、固定直方图及修改区域倍率。对每个所选checkpoint将alpha固定为1做Train/Valid推理，不重训练、不重新选epoch。`learned_minus_one`度量同权重调节作用，`one_minus_control`保留新旧训练参数和选模的差异；两者相加为总差值。命令、检查及数据见[context_scale实验报告](results/context_scale_512/report.md)，权重留本地。

两组均已完成16轮/17424步。`context_scale_diff`所选epoch为15，Valid P-C为15.84%、AUC为0.5824、MCC为0.1201；`context_scale_random`所选epoch为14，P-C为15.30%。真实Diff比随机Diff多判对3对，但比原 `context_diff` 少11对，P-C下降1.96个百分点，AUC/MCC也下降；修复侧FPR由52.31%降至50.18%，漏洞Recall同时由66.37%降至62.10%。本轮没有净分类提升。

真实Diff的Train/Valid alpha为1.7845±0.2072、1.8337±0.1803，随机Diff为1.8131±0.1540、1.8202±0.1391；完整分位数和直方图已保存。真实Diff区域倍率为3.2495/1.8224（原Context为3.1495/1.7831）。同一checkpoint固定alpha=1后，真实Diff P-C仍为15.84%，随机Diff反而升至16.01%，未观察到调节项的P-C收益；不能将独立训练所得差异归因于推理时alpha。66项相关测试、真实GPU smoke和[训练后单函数源码推理](results/context_scale_512/trained-inference.json)均通过；不追加变体。

### 跨样本修复区域迁移（离线）

`repair-transfer`冻结既有 `context_diff/best-p_c.pt`，复用正式 `paired_features/paired_collate/paired_forward`。只观察原线性注意力头输入，直接取得 `bounded_pool` 已计算的768维物理行表示；不改变模型、训练入口或checkpoint。

参考库只含官方Train漏洞函数中可见直接Diff行非空者：修改行为正参考，其余可见有效Token行为背景。固定查询组为已有静态审计的Train24/Valid24对纯新增修复。检索前排除同配对、完全相同非注释Token源码及重复参考文本；正/背景冲突文本排除，其余文本选首个合格来源。来源样本、物理行、Token数和匹配余弦完整保留。

直接检索使用最大正参考余弦，对比检索使用最大正参考余弦减最大背景余弦。两者输出同一排序的Top1/Top3，与实际MIL行注意力质量 `n_i*r_i/sum(n_j*r_j)` 和固定Seed123456随机排序比较。所有方法使用正式512输入的同一可见行集合，不额外按语法或API筛行。源码长度和同项目匹配仅作分层诊断，不用于调参。

```bash
python -m unittest tests.test_repair_transfer tests.test_bounded_attention tests.test_losver
python -m vulnmechanism.cli repair-transfer extract --device cuda:0
python -m vulnmechanism.cli repair-transfer retrieve --device cuda:0
# 逐行完成results/repair_transfer_512/review.jsonl中的pending判断，保留精确候选成员和源码。
python -m vulnmechanism.cli repair-transfer report
```

复核四种方法Top3候选的完整并集；复用已存在且源码一致的判断，其余逐行核实为related/unrelated/unknown。未复核成员阻止报告，Unknown保留在分母。相关比例、样本命中率、Unknown、静态未恢复组、长度和项目分层见[离线检索报告](results/repair_transfer_512/report.md)。行向量缓存留本地；不读取Test、不训练、不搜索参数。

本次真实GPU提取2181个参考函数，71777条原始参考行去重并排除标签冲突后，得到5952条正参考及40988条背景参考。固定Train24/Valid24查询的462条候选并集已全部完成源码复核（复用125条、新复核337条）；related包含能确认的数据/控制上下文，不能全部当作危险操作定位金标。64项前置相关测试、16项收尾离线测试通过；RTX4090上的嵌入重复提取与48组完整检索重复结果一致，原checkpoint未改写。

Train的Top3相关比例为直接检索19.44%、对比检索27.78%、注意力34.72%、随机15.28%；Valid分别为18.06%、19.44%、38.89%、27.78%，样本命中率为37.50%、41.67%、58.33%、58.33%。Valid中直接/对比检索相对注意力各新增2/1个命中，同时丢失7/5个；新增案例包含OVS循环后的encap字段写入和JasPer宽度赋值，后者仅为数据上下文。检索未超过现有注意力和随机对照，本轮不支持将其作为弱监督接入Context+Diff，不追加变体。

### 修复插入位置定位（Repair Insertion Localization）

在原 `--bounded-attention --attention-context --attention-region diff` 上增加 `--insertion-supervision diff|random`。复用 `_line_diff` 的同一双版本对齐；旧侧没有非空删除、且新侧有非空新增的配对提供插入监督。目标是旧侧行间位置 `p`（0/N为真实函数首尾），不是邻近代码行。空白区间内共享同一对有效行的物理位置合并为一个间隙区间，保留原始Diff坐标。两侧真实行都必须完整编码，真实边界缺失侧为0；截断处不构造伪EOF，部分可见行不作为位置两侧。

同一次UniXcoder编码得到行均值及有效代码Token全局均值g。插入头为 `Linear(2304,32) -> tanh -> Linear(32,1,bias=False)`，拼接左右两侧及g，共73792个参数；在独立CPU `fork_rng` 中初始化，保持原模型参数、函数输出和采样RNG。多个正位置共同参与 `mean softplus(s_negative-s_positive)`：先在每个函数内对正负位置对平均，再对有效函数平均。总损失固定为 `BCE + .1*L_diff + .1*L_insertion`；没有有效正负位置的函数仍保留原BCE和Diff损失。

Train有1232对纯新增修复，其中840对可监督，977/1590处Diff插入能映射为975个不重复可见间隙；Valid有163对，其中122对可评价，147/223处插入可见。随机对照只随机化Train的插入位置，逐函数保留相同正位置数量；原Diff行标签和全部函数成员不变。Valid始终使用真实插入坐标评价。

```bash
python -m unittest tests.test_insertion_localization tests.test_bounded_attention tests.test_line_auxiliary tests.test_losver tests.test_resume
python -m vulnmechanism.cli check-paired --insertion-localization --output-dir results/insertion_localization_512 --device cuda:0
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --paired-input 512 --paired-epochs 16 --bounded-attention --attention-context --attention-region diff --insertion-supervision diff --output-dir results/insertion_localization_512/insertion_diff --device cuda:0 --resume --save-every 1000
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --paired-input 512 --paired-epochs 16 --bounded-attention --attention-context --attention-region diff --insertion-supervision random --output-dir results/insertion_localization_512/insertion_random --device cuda:1 --resume --save-every 1000
python -m vulnmechanism.cli report --dataset primevul_paired --insertion-localization --output-dir results/insertion_localization_512 --reference-dir results/attention_context_512 --device cuda:0
```

两组沿用官方完整配对Train8708/Valid1124函数、512输入、Seed123456、16轮、原优化器及固定阈值>.5，从与原Context+Diff相同的预训练初始化训练，只按完整Valid P-C选模。推理仍复用原单函数源码入口，不需要Diff、修复版本或参考库，不增加每个前向的编码次数。覆盖率、真实GPU检查、位置排序、整体/纯新增子集分类、区域倍率和实际耗时保存在[实验目录](results/insertion_localization_512)；不读取Test、不搜索参数、不叠加其他模块。

两组均已完成16轮/17424步，所选epoch均为14。下表除纯新增P-C外均为完整Valid指标，纯新增子集含163对/326函数。

|配置|P-C|纯新增P-C|AUC|MCC|F1|Recall|修复侧FPR|
|---|---:|---:|---:|---:|---:|---:|---:|
|原Context+Diff|17.79%|22.70%|0.5896|0.1431|60.70%|66.37%|52.31%|
|真实插入位置监督|15.66%|16.56%|0.5812|0.1055|52.86%|50.18%|39.68%|
|随机插入位置监督|16.73%|24.54%|0.5811|0.1068|55.96%|56.76%|46.09%|

在122个可映射Valid漏洞函数上，真实/随机监督的插入位置Top1命中为10/4个（8.20%/3.28%），Top3为47/19个（38.52%/15.57%）。将不可见位置保留在全部163个纯新增函数分母后，真实监督Top3为28.83%。定位有改善，但真实监督的整体P-C比基线少12对，纯新增P-C少10对，也低于随机位置监督；FPR降低伴随Recall、AUC和MCC下降。本轮没有漏洞检测净收益，不追加模块或变体。

真实监督的Train Top1/Top3为90.48%/99.29%，与Valid有明显泛化落差；训练位置拟合不能视为稳定的精确插入定位或漏洞检测提升。

80项前置相关测试及1项新增评价测试、真实RTX4090 smoke、三组所选checkpoint的[单函数源码推理](results/insertion_localization_512/trained-inference.json)均通过；单函数与批量概率最大差为2.9e-8。真实/随机两组16轮训练耗时分别为54.04/48.92分钟，含初始化、验证及导出的进程耗时为63.81/57.05分钟；同机并发的实际耗时不用于单独归因模型计算开销。完整Train/Valid分类、定位、区域倍率、逐轮损失和协议检查见[插入定位报告](results/insertion_localization_512/report.md)，权重保留本地。

### Context + Diff 的固定 checkpoint Test 评估

2026-11-02按用户后续要求，使用原Context+真实Diff在完整Valid P-C选出的第14轮checkpoint、512输入及固定概率>.5，评估完整官方配对Test（1128函数/564对）。Test P-C为13.65%（77/564对）、AUC为0.5712、MCC为0.0885、F1为58.30%、Recall为63.83%、修复侧FPR为55.14%；评估进程耗时41.06秒。Test在原训练及选模完成后使用，未改变原checkpoint或选择规则。指标已从导出预测复算，命令、数据及完整Valid/Test比较见[Context+Diff报告](results/attention_context_512/report.md)。

### 已完成512实验的统一 Test 对照

按用户后续要求，对七系列22组完整模型统一补充Test：行辅助损失、有界注意力、抑制/EMA、Context、双尺度、context_scale及插入定位。新跑21组，复用已验证的Context+Diff结果；均为原完整Train/Valid训练、512输入、Seed123456、16轮，固定原Valid P-C所选checkpoint和概率>.5，Test为1128函数/564对。全部预测已逐一复算，成员、标签、配对计数、512可见行及checkpoint未修改检查通过。

Test P-C：Global Mean 13.83%、Context+Diff 13.65%、EMA 14.18%、真实插入监督 14.36%。真实插入比Context+Diff多4对，但AUC为0.5650（Context+Diff 0.5712）、Recall为47.34%（63.83%），且原Valid P-C更低，不能按稳定的全面提升解释。插入Test定位真实/随机Top1为16.80%/5.60%，Top3为40.00%/16.00%（125个可映射函数）。未按Test改选模型或追加变体；全部22组六项指标、日志与命令见[统一Test报告](results/attempts_test_512/report.md)。此前各报告未读Test的说明对应原训练及选模阶段。


## RCCL 成对批次实验

RCCL 直接复用 `data.py` 的 Diff/512 Token 物理行映射、`model.py` 的有界 Context MIL 行表示及 `training.py` 的正式训练循环。`--rccl baseline|diff|random` 都按修复对打乱批次：baseline 保留 Context + Diff，diff 选取编辑块左右最近的可靠未修改行，random 在同对中抽取等量可靠对齐行。仅使用两侧完整可见、全函数中无重复歧义且可见输入不同的对齐行；这些行不改变 Diff 标签。

投影头仅用于训练：无偏置 768→32，两个独立 p=0.1 表示层 Dropout 视图，对称四视图 InfoNCE（温度 0.1）。损失固定 `BCE + 0.1*Diff + 0.05*RCCL`。新增初始化和 Dropout 保留原模型/采样 RNG；每个 Epoch 记录实际成对顺序。恢复与原训练共用 `--resume`。单函数推理仍使用原 `infer` 入口，加载时移除对比投影头。

```bash
python -m vulnmechanism.cli check-paired --rccl --output-dir results/rccl_512 --device cuda:0
python -m vulnmechanism.cli train --dataset primevul_paired --methods unixcoder --bounded-attention --attention-context --attention-region diff --rccl diff --paired-epochs 16 --output-dir results/rccl_512/rccl_diff --device cuda:0 --resume
python -m vulnmechanism.cli report --dataset primevul_paired --rccl --output-dir results/rccl_512 --device cuda:0
```

三组完整匹配实验、覆盖率、完整 Valid/纯新增子集指标、选模后 Test 及表示相似度诊断见 [results/rccl_512/report.md](results/rccl_512/report.md)。逐条命令和固定配置保存在 `results/rccl_512/commands.json`；历史 Context + Diff checkpoint 保留。

## RGCL Random-region 消融

仅在 PrimeVul + UniXcoder 补充种子 42、1234、123456 的 `random_region`。复用正式模型与训练器，将训练区域改为有效代码行的随机子集；随机子集逐函数严格保持真实 Diff 的有效 Token 总量（跨行 Token 沿用原分摊方式），保留辅助损失可用性。使用独立 RNG 固定整个训练期间的区域，不改变模型初始化或采样 RNG。旧版按正标签行数打乱的随机对照保持原行为。

```bash
python -m vulnmechanism.cli rgcl-study check --dataset primevul_paired --methods context_only random_region rgcl --output-dir results/rgcl_random_region_512 --device cuda:0
python -m vulnmechanism.cli rgcl-study run --dataset primevul_paired --methods random_region --seed 42 --output-dir results/rgcl_random_region_512 --device cuda:0
python -m vulnmechanism.cli rgcl-study report --dataset primevul_paired --methods context_only random_region rgcl --reference-dir results/rgcl_multiseed_512 --output-dir results/rgcl_random_region_512
```

训练命令分别使用三个指定种子；512 Token、Batch8、16 Epoch、AdamW lr=2e-5、K=5、区域损失权重0.1不变。完整 Valid P-C 选模、固定概率严格大于0.5，选模完成后评价 Test。Context-only 和 RGCL 只读取历史完整结果，不重复训练或改写。监督量、实际重合率及逐种子/均值±样本标准差见 [消融报告](results/rgcl_random_region_512/report.md)。
