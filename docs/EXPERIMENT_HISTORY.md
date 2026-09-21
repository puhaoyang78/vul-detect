# 从 lora_v0 到源码分类结构对照：实验总结

更新日期：2026-09-21。

## 1. 当前结论与记录范围

项目目标始终是 C/C++ 单函数漏洞二分类，而不是仅提高静态特征预测、局部定位或修复语义分类的准确率。截至本次记录，尚没有新增方案在可比实验中稳定、全面超过 source-only baseline。

- 静态分析关系可以与真实修复一致，但尚未稳定转化为函数分类收益。
- 正式 mechanism concat 主要提高 Recall/F1，Accuracy/MCC 未提高。
- relation-only 与增量信息 probe 表明，问题不只是 candidate 的正类先验。
- 少量已核查错误样本的加权继续训练没有产生净迁移收益。
- 本轮源码结构改进中，attention pooling 的 AUC 略升，但其他核心指标未升；双向层及组合也未超过 baseline。

本总结区分两种证据：**已核验记录**来自现有 Git 历史、训练日志、checkpoint 与结果 JSON；**历史回顾**来自项目讨论，未找到对应原始数值的部分不补写指标。旧 checkpoint 未保存的 seed、training_config 等字段在 JSON 中保留为 null，不按当前默认值反推。完整精度的可公开数值摘录、逐 epoch 结果、配置和本地来源路径见 [experiment_results.json](experiment_results.json)。来源路径标识原始本地文件，原始日志、数据和模型权重不包含在本次上传中。

下文 valid、test、train 内 OOF 和小样本 pilot 分开报告。不同数据版本、split、源码长度、训练成员或阈值选择方式之间，不计算“累计提升”。本次仅整理已有记录，没有重新训练或重新评估 test。

## 2. 方法演进与早期尝试

| 阶段 | 尝试与目的 | 实际结果和证据边界 |
| --- | --- | --- |
| `lora_v0` | Qwen2.5-Coder-7B + LoRA，函数级 C/C++ 0/1 分类；探索 CPG 语义输入辅助学习 | 历史回顾：早期看起来有提升，但数据和 split 与正式 cohort 不同，不能直接比较；本次未找到可核验的早期完整指标表 |
| 静态事实增强 | `baseline / raw_cpg / semantic_concat`，向源码补充 AST/CFG/CDG/DDG 和静态语义文本 | 历史回顾：Raw CPG 未形成稳定收益；正式版本后来再次验证了这一点 |
| 语义融合 | `semantic_fusion`，源码与语义表示 cross-attention | 历史回顾：未明显超过 baseline；不能把后续相同思路改名当作新贡献 |
| 通用特征辅助监督 | `full`，memory/dependence/constraint/lifetime/pattern 等提炼特征约束训练 | 历史回顾：auxiliary feature supervision 反而下降；这类标签不等同于真实漏洞形成条件 |
| 局部 slice / patch supervision | 学习局部定位与修复语义，再迁移到函数分类 | 历史回顾：局部任务较好，但函数 0/1 迁移弱；缺少原始数值时不声称具体增益 |
| sequence pooling 调整 | 尝试 last-token pooling，随后恢复 masked mean | Git 历史确认尝试及恢复；不把“改最后一个 token”再次作为未经尝试的新方案 |
| mechanism 关系抽取 | 从堆通用特征转为具体操作、对象、表达式、来源和条件状态 | 抽取语义更合理，但正式检测收益仍受限，详见下文 |
| source-only 结构改进 | 不添加外部特征，改进 Qwen 输出的上下文处理与汇聚 | 本轮完成四组正式 train/valid 训练；未获得全面提升 |

可核对的历史入口：

- [早期语义 LoRA 归档](https://github.com/puhaoyang78/vul-detect/tree/889f6c9bdad228c647e71b565964387f7606021f)（远端分支 `archive/semantic-lora-v0`）。这是找到的早期代码归档，不据此推定某个旧数据版本的成绩。
- [semantic fusion 与 feature supervision 实现](https://github.com/puhaoyang78/vul-detect/commit/f95f032)及[对应说明](https://github.com/puhaoyang78/vul-detect/commit/6822ee0)。
- [恢复 mean pooling](https://github.com/puhaoyang78/vul-detect/commit/ba45202)。
- [早期机制学习实验](https://github.com/puhaoyang78/vul-detect/commit/b38b1b6)。历史局部任务成绩不能用这条代码提交替代实测证据。

## 3. CPG 抽取的改进与正式数据口径

逐步明确了三条边界：`security operation ≠ vulnerability`；`missing check ≠ vulnerability`；`candidate ≠ ground-truth mechanism`。

抽取层围绕 `ARRAY_ACCESS / MEMORY_WRITE / ALLOCATION` 等具体操作绑定 object、expression、source，并修正 whole-sink DDG 串线、`sizeof` 错误传播、指针声明误当 arithmetic、nullable provenance、bounds capacity relation、size arithmetic flow、lifetime relation，进一步引入 expression-local DDG 和 active-variable provenance。

例如项目已观察到 `data_size, header_size → data_size-header_size → ARRAY_ACCESS(data)`，且修复前后 constraint state 从 `not_observed` 变为 `present`。这是关系与局部修复一致的证据，不是函数安全性的证明。特别是 `not_observed` 不能直接视为缺少有效检查，函数中出现比较也不等于该条件在到达操作的路径上成立。branch polarity、dominance/reachability、redefinition 等路径语义仍不能当作已完整解决。

抽取层操作绑定已经存在；此前模型层主要是 relation 文本与整段源码全局 concat/cross-attention，并没有显式保留 operation→source span 的对应。不能把二者混为一谈。[表达式局部抽取](https://github.com/puhaoyang78/vul-detect/commit/027ec19)、[active-variable provenance](https://github.com/puhaoyang78/vul-detect/commit/c26c5e0)可在历史中核对。

正式数据构建共 14,654 条，成功 11,266 条，失败 3,388 条。PrimeVul 在 build-success 后按 split 内确定性 1:1 平衡，最终 train 5,670、valid 736、test 750；正负类各为 2,835 / 368 / 375。构建总数包含其他数据源，不应写成 PrimeVul 最终训练集数量。

历史汇报中 PrimeVul relation/candidate 覆盖约为 17.2% / 11.3%；本总结将其保留为历史近似值，不假设其分母一定与平衡后 7,156 条完全一致。覆盖率也不是有效证据率或分类贡献率。

## 4. 正式 mechanism 实验

### 4.1 已保存的正式 test 结果

同一正式 test，750 条，正负各 375。各模型使用各自训练阶段选定的 valid 阈值。

| 方法 | Accuracy | Precision | Recall | F1 | MCC | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.7440 | 0.7886 | 0.6667 | 0.7225 | 0.4939 | 0.8265 |
| Raw CPG @384 | 0.7373 | 0.7445 | 0.7227 | 0.7334 | 0.4749 | 0.8152 |
| Mechanism Concat | 0.7413 | 0.7191 | 0.7920 | 0.7538 | 0.4852 | 0.8285 |

Mechanism Concat 相对 baseline：Recall 从 0.6667 升至 0.7920，F1 增加约 3.13 个百分点；Accuracy 下降约 0.27 个百分点，MCC 下降约 0.88 个百分点，AUC 仅增加约 0.20 个百分点。不能概括为整体判别能力显著提高。

历史 subgroup 分析发现 candidate-present 子集正类比例高，concat 在该子集 Recall 上升、MCC/AUC 下降，提示 candidate 可能被当成类别先验。本次没有找到独立保存的完整 subgroup 指标表，因此只记录定性发现，不制造具体数值。

### 4.2 各组所选 checkpoint 的 valid 结果

| 方法 | Accuracy | Precision | Recall | F1 | MCC | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.7459 | 0.8089 | 0.6440 | 0.7171 | 0.5024 | 0.8164 |
| Raw CPG @384 | 0.7459 | 0.7608 | 0.7174 | 0.7385 | 0.4927 | 0.8073 |
| Mechanism Concat | 0.7459 | 0.7363 | 0.7663 | 0.7510 | 0.4923 | 0.8126 |
| Mechanism Fusion | 0.7364 | 0.7153 | 0.7853 | 0.7487 | 0.4751 | 0.8096 |
| Relation-only Concat | 0.7432 | 0.7737 | 0.6875 | 0.7281 | 0.4895 | 0.8099 |
| Relation-only Fusion | 0.7432 | 0.7507 | 0.7283 | 0.7393 | 0.4866 | 0.8132 |

Relation-only 排除 `MECHANISM_CANDIDATE`，保留 relation；两种模型仍未超过 baseline，说明 candidate prior 不是唯一问题。Fusion 与 relation-only 未找到对应正式 test 结果，不以 valid 代替 test。

## 5. OOF 增量信息诊断

问题改为：relation 内容是否能在 baseline 分数之外解释剩余错误？仅用正式 train，完成 3 折外层评估（3 个 outer folds，各 1,890 条；内部另有训练/校准划分）。Qwen 源码长度为 1,536，不与 2,048 长度正式 baseline 或正式 test 数值直接比较。

- `S`：baseline 分数经 probe 处理。
- `U`：relation 存在性、类型、数量等元信息。
- `R`：当前 relation 文本/状态，并非已验证的完整路径条件。
- `shuffle`：在受控匹配条件下交换内容并重新拟合；`intervention`：对既有内容模型做置换干预。保留原始 probe 定义，不将两者混称。

以下为逐折指标的等权均值，不拼接不同模型的分数计算 pooled AUC；shuffle/intervention 先在每折内平均 5 次，再平均折。

| 输入/对照 | 全体 AUC | 全体 MCC | matched AUC | matched MCC |
| --- | --- | --- | --- | --- |
| S | 0.898812 | 0.641501 | 0.824286 | 0.470653 |
| S+U | 0.897677 | 0.633959 | 0.816716 | 0.423316 |
| S+U+R | 0.897008 | 0.637734 | 0.811564 | 0.462512 |
| shuffle | 0.897007 | 0.635839 | 0.811017 | 0.448889 |
| intervention | 0.896976 | 0.635762 | 0.809315 | 0.446037 |

加入 R 相对 S+U：全体 AUC 平均变化 -0.000669（仅 1/3 折上升），MCC +0.003775（2/3 折上升）；仍低于 S 的平均 AUC/MCC。全体上完整 R 与 shuffle 的 AUC 几乎相同。matched 子集也未形成同时改善 AUC/MCC 且超越 S 的证据。

结论是当前内容表示和 probe 没有证明稳定的增量判别价值，不是“任何静态语义都无用”。原先设想的 frozen-baseline signed residual、matched ranking 等不应记作已经跑出有效结果的模块；在此诊断后没有获得继续投入它们的充分实验证据。

## 6. 从 baseline 错误出发的 source-only 小实验

先核查训练错误，区分局部可判断、依赖函数外上下文、标签/修复归因未解和输入截断。这里的核查是项目内助手辅助审阅，**不是独立人工裁决**。最终用于小实验的 10 条是有局部证据的样本（3 正、7 负），负例的证据也不意味着整个函数不存在其他漏洞。

小实验从 outer_01 的 OOF checkpoint 出发（不是正式 full-train baseline）：同一批 128 条样本、64 正/64 负、1 epoch、学习率 2e-5。比较普通继续训练和难例 3 倍权重继续训练，权重在每个类别内归一化，保持两类总权重不变。四条难例此前已在这个初始模型的训练成员中，不能把 10 条都写成该模型未见错误。

检查集 128 条排除初始/继续训练成员和关联 provenance groups，与继续训练集隔离项目，并进行近重复筛查。初始模型仍可能见过检查项目的其他函数，因此不是严格项目外泛化；检查标签沿用 PrimeVul，并非独立核实的同机制标签。三组统一使用初始校准得到的阈值 0.76，valid 736 条只报告，test 不参与。

| 模型 | check AUC | check MCC | valid AUC | valid MCC | valid TP | valid FP |
| --- | --- | --- | --- | --- | --- | --- |
| initial | 0.9111 | 0.6681 | 0.7934 | 0.4372 | 196 | 45 |
| uniform | 0.9124 | 0.6455 | 0.7908 | 0.4068 | 172 | 37 |
| weighted | 0.9119 | 0.6455 | 0.7904 | 0.4068 | 172 | 37 |

训练难例额外纠正 3 条，但 valid 纠正 8 条、同时损坏 24 条原本正确判断；两种继续训练的最终分类相同，分数并非完全相同。没有验证“提高这些错误的训练权重即可迁移涨点”。这只是 10 条核查样本、单 seed 的小实验，不能否定全部难例学习方法，也不支持将人工筛选难例确立为主方法。

## 7. 本轮：source-only 分类结构对照

### 7.1 设计与控制

目标从添加人工静态特征转为检验源码表示的利用方式：baseline 为 Qwen 因果注意力输出 → masked mean → linear。新增模块不改变 Qwen 自身的 causal mask，也不输入 CPG 或人工特征标签。

| 变体 | Qwen 输出之后的结构 |
| --- | --- |
| `baseline` | 原始 masked mean + linear |
| `source_attention` | 4 个 learned queries 的注意力汇聚，分类器维度不变 |
| `source_bidirectional` | 256 维单层双向 Transformer 残差 adapter + masked mean |
| `source_bidirectional_attention` | 双向 adapter + 注意力汇聚 |

Adapter 为 LayerNorm、降维、4-head Transformer（FFN 512、dropout 0）和升维残差；升维初始化为零。query 不对应手工漏洞类型。源码、LoRA 和对应模块的初始化对齐，新增参数和分类头共同接受函数级 BCE。

本轮四组均为 Qwen2.5-Coder-7B-Instruct、源码 2,048 tokens、LoRA r=16/alpha=32/dropout=0.05、batch=1、gradient accumulation=8、学习率固定 2e-4、weight decay=0.01、seed=42、3 epochs。每 epoch 709 个优化步，四组全部完成。按 valid MCC 选 epoch 和阈值，不能分别挑每列最大值拼成一行。

以下实现说明对应本地已测试代码。本次上传范围为实验总结和数值摘录，不包含尚未提交的训练代码修改；不能仅凭本次文档提交在远端直接复现新增变体。代码验证记录为 12 项相关测试通过，以及四组真实 Qwen 小规模 smoke（各 4 条 train、2 条 valid，覆盖 2,048-token 输入、反向传播、保存/重载）。这些 smoke 不计入性能证据。

### 7.2 所选 checkpoint 的 valid 结果

| 变体 | epoch | 阈值 | Accuracy | Precision | Recall | F1 | MCC | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | 0.46 | 0.7541 | 0.7416 | 0.7799 | 0.7603 | 0.5088 | 0.8175 |
| source_attention | 3 | 0.66 | 0.7527 | 0.7409 | 0.7772 | 0.7586 | 0.5060 | 0.8207 |
| source_bidirectional | 3 | 0.48 | 0.7486 | 0.7173 | 0.8207 | 0.7655 | 0.5025 | 0.8119 |
| source_bidirectional_attention | 3 | 0.43 | 0.7459 | 0.7071 | 0.8397 | 0.7677 | 0.5007 | 0.8158 |

### 7.3 每个 epoch 的学习过程

| 变体 | epoch | Train loss | valid MCC | valid AUC |
| --- | --- | --- | --- | --- |
| baseline | 1 | 0.457204 | 0.508831 | 0.817492 |
| baseline | 2 | 0.376886 | 0.482042 | 0.806260 |
| baseline | 3 | 0.311448 | 0.476179 | 0.807538 |
| source_attention | 1 | 0.424825 | 0.499294 | 0.812153 |
| source_attention | 2 | 0.355014 | 0.477588 | 0.806408 |
| source_attention | 3 | 0.315188 | 0.506040 | 0.820667 |
| source_bidirectional | 1 | 0.490324 | 0.420972 | 0.750473 |
| source_bidirectional | 2 | 0.412318 | 0.463586 | 0.800427 |
| source_bidirectional | 3 | 0.340146 | 0.502522 | 0.811894 |
| source_bidirectional_attention | 1 | 0.459799 | 0.449248 | 0.790510 |
| source_bidirectional_attention | 2 | 0.425892 | 0.472889 | 0.805529 |
| source_bidirectional_attention | 3 | 0.335407 | 0.500728 | 0.815808 |

### 7.4 结果解释

- Baseline：TP=287、FP=100。注意力汇聚：TP=286、FP=100；AUC +0.003175，但 Accuracy/F1/MCC 略降，单 seed 只能视为基本持平。
- 双向层：TP=302、FP=119，相比 baseline 多检出 15 个正例，也多误报 19 个负例；AUC/MCC 下降。
- 双向层 + attention：TP=309、FP=128，多检出 22 个正例、多误报 28 个负例；F1 上升，但 Accuracy/MCC/AUC 未升。这些是总数差异，不是未经逐样本对齐就声称的纠错数量。
- Baseline 在 epoch 1 最好，之后 train loss 下降但 valid 排序/分类指标下降；新增模块到 epoch 3 最好，说明学习动态不同。不能据此直接断言多训练几个 epoch 就会超过 baseline。

“平均池化与缺少双向上下文是主要瓶颈”的假设，本轮没有得到足以支撑它的证据。双向层并非直接改造 Qwen 为双向预训练编码器，本结果也不能推广为所有双向模型无效。

本轮没有正式 test 结果；新 baseline 的 valid MCC=0.5088，与上一阶段正式 baseline 的 0.5024 是不同训练运行，不应混用。尚未完成多 seed 稳定性或显著性检验。

## 8. 截至本次的研究状态

目前不能把任何一条路线写成“已经稳定全面提升漏洞判别能力”：静态关系的抽取合理性、局部修复任务表现、训练难例拟合和模型容量增加，都不能替代最终函数分类证据。

已获得的较稳妥认识是：候选/静态操作不能直接视为漏洞；完整内容相对元信息的价值需要样本外检验；Recall/F1 的提高必须结合误报、MCC 和 AUC 解读；新模块必须和同配置 source-only baseline 比较。保留负结果，避免将已经试过的 fusion、feature supervision、slice/patch 迁移或 pooling 改名重复。

原始完整精度、各阶段的分开评价口径和逐折/逐 epoch 数值均在 [数值摘录](experiment_results.json) 中。当前缺失的早期数值、subgroup 细表、局部任务完整成绩，不用推测补齐。
