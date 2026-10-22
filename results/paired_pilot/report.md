# PrimeVul 官方漏洞—修复对判别学习

CodeBERT、2048逻辑输入、原生滑动窗口及共享线性分类头不变。单函数推理；固定Sigmoid阈值0.5，不搜索阈值。

BCE为两函数BCE的平均；Ranking=logsumexp(0, Δ−z_v+z_p)，Boundary=logsumexp(0,m−z_v,m+z_p)，Joint=logsumexp(0,m−z_v,m+z_p,Δ−z_v+z_p)。三组约束统一λ=0.2、m=0.2、Δ=0.6。按配对平均。

官方配对按相邻互补标签读取，不用项目/提交元数据异常删对。重复的上游idx用原始行位置区分并保留；源idx另存。Train与任意Valid/Test编码输入相同的整对不进入训练。对内不可区分的Train配对保留相同BCE成员，但不施加任何成对约束。Valid/Test原始成员全部保留。

选模：仅Clean Valid，Strict Pairwise Accuracy/MCC/AUC/F1顺序，完全相同保留早轮。严格配对正确要求漏洞预测为1、修复预测为0；Ranking要求z_v>z_p，平局不计正确；已核实编码相同的配对视为排序平局，保留在分母中，不把浮点归约噪声当成功。训练日志的原始浮点Ranking值原样保留；最终指标从已保存预测统一重算，此修正不影响不使用Ranking的选模。边界违反率表示z_v<m或z_p>−m，不等同于分类错误。

| Split | 原始配对 | 使用配对 | 对内不可区分 |
|---|---:|---:|---:|
| train | 3789 | 3781 | 386 |
| valid | 480 | 480 | 52 |
| test | 435 | 435 | 44 |

Train泄漏排除8对；元数据异常19对。原始跨Split重复及上游idx重复详见results.json。Test这里只作数据核验，是否模型评价由后面的结果行明确区分。

| 方法 | Split | Epoch | N对 | Strict Pair Acc | Ranking Acc | AUC | MCC | F1 | Logit Gap | 平均绝对Logit | 边界违反率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bce | valid | 2 | 480 | 0.0667 | 0.5771 | 0.5409 | 0.0466 | 0.4226 | 0.0112 | 0.0647 | 1.0000 |
| pair_ranking | valid | 2 | 480 | 0.0667 | 0.5667 | 0.5402 | 0.0674 | 0.2467 | 0.0156 | 0.1459 | 1.0000 |
| pair_boundary | valid | 3 | 480 | 0.0667 | 0.5958 | 0.5362 | 0.0485 | 0.4303 | 0.0189 | 0.1190 | 1.0000 |
| pair_joint | valid | 3 | 480 | 0.0667 | 0.5917 | 0.5386 | 0.0409 | 0.4518 | 0.0200 | 0.1193 | 1.0000 |

完整指标、逐样本Logit、混淆计数和成对可区分子集指标保存在各方法目录。AUC和MCC/F1同时用于检查分类收益，不能仅依据Logit Gap或绝对Logit扩大判定有效。

## 预设Valid门槛

Joint minus BCE: Strict Pairwise Accuracy >= +0.02, AUC >= 0, MCC >= 0; fixed before pilot training; Test never used

Joint−BCE：Strict +0.0000，AUC -0.0023，MCC -0.0057，F1 +0.0292。

门槛未通过，停止全量训练；不根据Test调整模型或损失。

单Seed、固定512对试验用于检验最小假设，不证明跨种子稳定性。若目录中没有test指标，说明没有进行Test推理。

| 方法 | BCE错误→正确对 | BCE正确→错误对 | 共同错误对 |
|---|---:|---:|---:|
| bce | 0 | 0 | 448 |
| pair_ranking | 25 | 25 | 423 |
| pair_boundary | 22 | 22 | 426 |
| pair_joint | 25 | 25 | 423 |

Joint的Logit Gap相对BCE改变+0.008858，平均绝对Logit改变+0.054642。但严格配对正确数为32/480，BCE为32/480。分数差或尺度扩大不能替代固定阈值下的分类改善；方法间也可能同时修正和损伤不同配对，不能将其直接解释成单一标量缩放。

训练记录与真实重载的AUC最大差为0.00001953；具体差异保存在training_validation_comparison。最终表使用已保存的重载预测。编码相同的函数仍可能因浮点归约产生微小分数差，排序指标按已核实的输入等价关系处理平局。
