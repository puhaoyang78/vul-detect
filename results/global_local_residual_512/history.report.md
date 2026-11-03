# 历史 Test 错误诊断

仅复用既有预测，不训练，不改变历史 checkpoint、阈值或轮次；完整 pair_key 与 target 连接，并核对 sample_key。
本节不能作为本轮结构或参数选择依据。本轮零局部头/直接加法设计由先前 Train/Valid 融合近零和首步梯度阻断支持。
完整 Test 为 564 对，Context + Diff 正确 77 对（13.65%）；相对 LOSVER 新增/丢失正确对 44/46，历史 Valid 的优势未保持。
其中 401 对具有可见差异：P-C/P-V/P-B/P-R 分别为 77/181/115/28；输入不可辨识不能解释其余这些错误。
P-C：漏洞侧判正、修复侧判负；P-V：两侧均正；P-B：两侧均负；P-R：方向完全相反。
行号从零开始。可见性按正式 512 Token ID 比较，另核对物理行 Token 成员；不删除不可辨识配对。
可见差异只表示编码输入不同，不保证关键修复语义完整可见；截断与源码抽取异常在案例中另行核实。
Test 缺少原始 Logit 的模型由冻结原函数头补齐；已保存的双分支分量按 FP32 相加恢复。Train/Valid 未保存 Logit 的非饱和项用概率逆变换，饱和项冻结补查。原预测和指标保持不变，不裁剪概率。
成对排序仍按保存概率计算；补查的 Logit 差距只解释饱和，不改变历史评价口径。
不同模型 Logit 尺度不同，平均差距不直接作为性能排序。
历史模型训练顺序与辅助任务不同，跨模型得失仅描述错误重合；本轮新的三组 Train/Valid 才是匹配训练对照。

| Split | 可见组 | 方法 | 对数 | P-C | P-V | P-B | P-R | 排序正确率 | 平均 Logit 差 | 差距 Unknown |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | all | context_diff | 4354 | 63.41% | 26.60% | 9.97% | 0.02% | 69.98% | 13.5226 | 0 |
| train | all | rccl | 4354 | 65.30% | 17.68% | 17.02% | 0.00% | 70.85% | 11.9946 | 0 |
| train | all | insertion | 4354 | 61.37% | 10.98% | 27.63% | 0.02% | 70.21% | 12.3466 | 0 |
| train | all | dual_baseline | 4354 | 63.50% | 21.04% | 15.41% | 0.05% | 70.33% | 13.5055 | 0 |
| train | all | dual_mean | 4354 | 30.36% | 18.17% | 50.71% | 0.76% | 62.24% | 1.3943 | 0 |
| train | all | dual_lse | 4354 | 60.01% | 34.31% | 5.63% | 0.05% | 69.57% | 10.1536 | 0 |
| train | visible_difference | context_diff | 3100 | 89.06% | 6.19% | 4.71% | 0.03% | 98.29% | 18.9928 | 0 |
| train | visible_difference | rccl | 3100 | 91.71% | 3.61% | 4.68% | 0.00% | 99.52% | 16.8466 | 0 |
| train | visible_difference | insertion | 3100 | 86.19% | 3.52% | 10.26% | 0.03% | 98.61% | 17.3410 | 0 |
| train | visible_difference | dual_baseline | 3100 | 89.19% | 5.77% | 4.97% | 0.06% | 98.77% | 18.9687 | 0 |
| train | visible_difference | dual_mean | 3100 | 42.65% | 18.84% | 37.45% | 1.06% | 87.42% | 1.9584 | 0 |
| train | visible_difference | dual_lse | 3100 | 84.29% | 11.52% | 4.13% | 0.06% | 97.71% | 14.2609 | 0 |
| train | identical_input | context_diff | 1254 | 0.00% | 77.03% | 22.97% | 0.00% | 0.00% | 0.0000 | 0 |
| train | identical_input | rccl | 1254 | 0.00% | 52.47% | 47.53% | 0.00% | 0.00% | 0.0000 | 0 |
| train | identical_input | insertion | 1254 | 0.00% | 29.43% | 70.57% | 0.00% | 0.00% | 0.0000 | 0 |
| train | identical_input | dual_baseline | 1254 | 0.00% | 58.77% | 41.23% | 0.00% | 0.00% | 0.0000 | 0 |
| train | identical_input | dual_mean | 1254 | 0.00% | 16.51% | 83.49% | 0.00% | 0.00% | 0.0000 | 0 |
| train | identical_input | dual_lse | 1254 | 0.00% | 90.67% | 9.33% | 0.00% | 0.00% | 0.0000 | 0 |
| valid | all | unixcoder | 562 | 16.19% | 46.62% | 33.45% | 3.74% | 45.37% | 0.7997 | 0 |
| valid | all | losver | 562 | 14.23% | 36.65% | 43.59% | 5.52% | 41.10% | 2.3096 | 0 |
| valid | all | context_diff | 562 | 17.79% | 48.58% | 29.89% | 3.74% | 46.62% | 1.7668 | 0 |
| valid | all | rccl | 562 | 16.73% | 47.86% | 30.07% | 5.34% | 46.80% | 1.6201 | 0 |
| valid | all | insertion | 562 | 15.66% | 34.52% | 44.66% | 5.16% | 46.98% | 1.6010 | 0 |
| valid | all | dual_baseline | 562 | 17.44% | 48.40% | 30.07% | 4.09% | 43.95% | 1.7482 | 0 |
| valid | all | dual_mean | 562 | 14.06% | 20.82% | 63.70% | 1.42% | 49.82% | 0.1885 | 0 |
| valid | all | dual_lse | 562 | 15.12% | 57.30% | 22.24% | 5.34% | 45.91% | 1.1607 | 0 |
| valid | visible_difference | unixcoder | 398 | 22.86% | 42.21% | 29.65% | 5.28% | 64.07% | 1.1293 | 0 |
| valid | visible_difference | losver | 398 | 20.10% | 31.91% | 40.20% | 7.79% | 58.04% | 3.2612 | 0 |
| valid | visible_difference | context_diff | 398 | 25.13% | 41.46% | 28.14% | 5.28% | 65.83% | 2.4949 | 0 |
| valid | visible_difference | rccl | 398 | 23.62% | 43.22% | 25.63% | 7.54% | 66.08% | 2.2876 | 0 |
| valid | visible_difference | insertion | 398 | 22.11% | 32.66% | 37.94% | 7.29% | 66.33% | 2.2607 | 0 |
| valid | visible_difference | dual_baseline | 398 | 24.62% | 44.47% | 25.13% | 5.78% | 62.06% | 2.4686 | 0 |
| valid | visible_difference | dual_mean | 398 | 19.85% | 22.61% | 55.53% | 2.01% | 70.35% | 0.2662 | 0 |
| valid | visible_difference | dual_lse | 398 | 21.36% | 46.98% | 24.12% | 7.54% | 64.82% | 1.6390 | 0 |
| valid | identical_input | unixcoder | 164 | 0.00% | 57.32% | 42.68% | 0.00% | 0.00% | 0.0000 | 0 |
| valid | identical_input | losver | 164 | 0.00% | 48.17% | 51.83% | 0.00% | 0.00% | 0.0000 | 0 |
| valid | identical_input | context_diff | 164 | 0.00% | 65.85% | 34.15% | 0.00% | 0.00% | 0.0000 | 0 |
| valid | identical_input | rccl | 164 | 0.00% | 59.15% | 40.85% | 0.00% | 0.00% | 0.0000 | 0 |
| valid | identical_input | insertion | 164 | 0.00% | 39.02% | 60.98% | 0.00% | 0.00% | 0.0000 | 0 |
| valid | identical_input | dual_baseline | 164 | 0.00% | 57.93% | 42.07% | 0.00% | 0.00% | 0.0000 | 0 |
| valid | identical_input | dual_mean | 164 | 0.00% | 16.46% | 83.54% | 0.00% | 0.00% | 0.0000 | 0 |
| valid | identical_input | dual_lse | 164 | 0.00% | 82.32% | 17.68% | 0.00% | 0.00% | 0.0000 | 0 |
| test | all | unixcoder | 564 | 14.54% | 47.16% | 34.93% | 3.37% | 46.45% | 0.5595 | 0 |
| test | all | losver | 564 | 14.01% | 36.70% | 44.15% | 5.14% | 41.84% | 2.1791 | 0 |
| test | all | context_diff | 564 | 13.65% | 50.18% | 31.21% | 4.96% | 42.91% | 1.4982 | 0 |
| test | all | rccl | 564 | 14.36% | 50.18% | 30.50% | 4.96% | 43.26% | 1.4200 | 0 |
| test | all | insertion | 564 | 14.36% | 32.98% | 48.05% | 4.61% | 45.04% | 1.0698 | 0 |
| test | all | dual_baseline | 564 | 12.41% | 49.47% | 33.33% | 4.79% | 44.15% | 1.4464 | 0 |
| test | all | dual_mean | 564 | 9.57% | 19.50% | 69.68% | 1.24% | 47.87% | 0.1381 | 0 |
| test | all | dual_lse | 564 | 10.11% | 64.36% | 22.52% | 3.01% | 43.09% | 1.0203 | 0 |
| test | visible_difference | unixcoder | 401 | 20.45% | 43.14% | 31.67% | 4.74% | 65.34% | 0.7870 | 0 |
| test | visible_difference | losver | 401 | 19.70% | 35.16% | 37.91% | 7.23% | 58.85% | 3.0649 | 0 |
| test | visible_difference | context_diff | 401 | 19.20% | 45.14% | 28.68% | 6.98% | 60.35% | 2.1072 | 0 |
| test | visible_difference | rccl | 401 | 20.20% | 47.88% | 24.94% | 6.98% | 60.85% | 1.9972 | 0 |
| test | visible_difference | insertion | 401 | 20.20% | 33.42% | 39.90% | 6.48% | 63.34% | 1.5046 | 0 |
| test | visible_difference | dual_baseline | 401 | 17.46% | 51.12% | 24.69% | 6.73% | 62.09% | 2.0343 | 0 |
| test | visible_difference | dual_mean | 401 | 13.47% | 22.94% | 61.85% | 1.75% | 67.33% | 0.1942 | 0 |
| test | visible_difference | dual_lse | 401 | 14.21% | 59.60% | 21.95% | 4.24% | 60.60% | 1.4350 | 0 |
| test | identical_input | unixcoder | 163 | 0.00% | 57.06% | 42.94% | 0.00% | 0.00% | 0.0000 | 0 |
| test | identical_input | losver | 163 | 0.00% | 40.49% | 59.51% | 0.00% | 0.00% | 0.0000 | 0 |
| test | identical_input | context_diff | 163 | 0.00% | 62.58% | 37.42% | 0.00% | 0.00% | 0.0000 | 0 |
| test | identical_input | rccl | 163 | 0.00% | 55.83% | 44.17% | 0.00% | 0.00% | 0.0000 | 0 |
| test | identical_input | insertion | 163 | 0.00% | 31.90% | 68.10% | 0.00% | 0.00% | 0.0000 | 0 |
| test | identical_input | dual_baseline | 163 | 0.00% | 45.40% | 54.60% | 0.00% | 0.00% | 0.0000 | 0 |
| test | identical_input | dual_mean | 163 | 0.00% | 11.04% | 88.96% | 0.00% | 0.00% | 0.0000 | 0 |
| test | identical_input | dual_lse | 163 | 0.00% | 76.07% | 23.93% | 0.00% | 0.00% | 0.0000 | 0 |

## Test 正确配对的相互得失

每格为行方法相对列方法的新增/丢失正确对；保留完整 564 对分母。

| 方法 | unixcoder | losver | context_diff | rccl | insertion | dual_baseline | dual_mean | dual_lse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unixcoder | — | 54/51 | 50/45 | 51/50 | 54/53 | 44/32 | 55/27 | 51/26 |
| losver | 51/54 | — | 46/44 | 46/48 | 51/53 | 53/44 | 54/29 | 53/31 |
| context_diff | 45/50 | 44/46 | — | 43/47 | 39/43 | 25/18 | 51/28 | 41/21 |
| rccl | 50/51 | 48/46 | 47/43 | — | 53/53 | 50/39 | 58/31 | 51/27 |
| insertion | 53/54 | 53/51 | 43/39 | 53/53 | — | 47/36 | 55/28 | 49/25 |
| dual_baseline | 32/44 | 44/53 | 18/25 | 39/50 | 36/47 | — | 45/29 | 30/17 |
| dual_mean | 27/55 | 29/54 | 28/51 | 31/58 | 28/55 | 29/45 | — | 34/37 |
| dual_lse | 26/51 | 31/53 | 21/41 | 27/51 | 25/49 | 17/30 | 37/34 | — |

Train 中无保存预测的 UniXcoder/LOSVER 不重训、不补造；Train 机制核对使用已有 Context/RCCL/插入/双分支结果。
完整分组、错误类型内排序方向、Logit 分布及配对转移见 history.summary.json、history.pairs.jsonl 和 history.comparisons.json。
局部行高分和 Diff 重叠仅为诊断证据，不等价于行漏洞真值；编码信息缺失与当前线性读出失效必须区分。

## 冻结行表示与当前线性分类器的容量

逐函数允许标签知情的最优权重，只用于区分冻结读出限制和权重选择问题。它不是可部署模型；相同输入的配对仍不能分别选择不同权重。
| Split | 可见组 | 函数错误 | 1–5倍也无法纠正 | 任意凸聚合也无法纠正 | 1–5倍存在可纠正权重 |
|---|---|---:|---:|---:|---:|
| train | all | 1594 | 975 | 84 | 619 |
| train | visible_difference | 340 | 170 | 22 | 170 |
| train | identical_input | 1254 | 805 | 62 | 449 |
| valid | all | 483 | 309 | 101 | 174 |
| valid | visible_difference | 319 | 217 | 84 | 102 |
| valid | identical_input | 164 | 92 | 17 | 72 |
| test | all | 515 | 324 | 85 | 191 |
| test | visible_difference | 352 | 223 | 70 | 129 |
| test | identical_input | 163 | 101 | 15 | 62 |

任意凸聚合也无法纠正，只能说明该冻结线性读出缺少正确方向，不能证明编码器完全没有信息；有可行权重也不证明正常推理能学会选择。

## 源码与真实 Diff 复核

- primevul_paired:test:1: 修复侧开头连续出现原函数签名与新增 const 签名；疑似源码抽取异常。该样本 P-C 正确不能直接解释为识别了安全机理。  机理判断：Unknown。
- primevul_paired:test:4: 物理行5删除 if 条件后的分号，使集合更新受 pageIsProcessSuppressible 条件约束；两侧几乎满置信判漏洞，修改行本已获得高注意力。 修复侧修改行投影 +7.665、倍率约5；全部行投影最小值 +4.182，因此任意凸聚合也不能判安全。 机理判断：条件约束行为可由源码确认；完整安全影响 Unknown。
- primevul_paired:test:5: 新增 BindingSecurity 访问检查并在 toV8 中改用 sanitizedCreationContext；上下文注意力已覆盖旧调用及新检查，但漏洞侧仍漏报。LOSVER/UniXcoder 正确。 漏洞侧真实修改行19投影为 -26.105，为该函数最负值，注意力占比26.8%、倍率4.992；高关注反而推动漏报。标签知情的1–5倍上界 +1.150，可行不等于学会选择修复相关线。 机理判断：上下文访问检查与调用替换可确认。
- primevul_paired:test:6: 删除末尾无条件 views::Widget::OnGestureEvent 调用；多个模型排序反向。  机理判断：Unknown。
- primevul_paired:test:7: Diff 在719行之后，可见输入最大物理行61；两个版本的512输入与行Token成员完全相同。  机理判断：不可见，不能从当前输入恢复。
- primevul_paired:test:8: SetPSDPixel 实参增加 ssize_t 中间转换，修改在可见末端65至67行；旧侧修改行已在注意力占比前三，Context+Diff 仍两侧判安全且排序反向。  机理判断：类型转换变化可确认；完整安全影响 Unknown。
- primevul_paired:test:9: 物理行50增加 create_kernel 失败路径的 delete_kernel，512可见范围只到43行；所有模型两侧概率分别完全相同。  机理判断：修复增加失败路径清理；当前模型输入不可辨识。
- primevul_paired:test:12: 物理行3增加 !hmhd 保护，短路避免空指针访问 hmhd->type；Context+Diff 判漏洞侧0.206、修复侧0.684，注意力占比前三均为背景行7/8/6。 保护行3投影从 +5.610降至 +3.318，方向有所改善；但未修改行7从 +4.901升至 +7.209，最终函数排序反向。两侧均存在1–5倍可纠正权重，不能归因于全部行投影同号。 机理判断：空指针保护可直接确认。
- primevul_paired:test:16: 物理行26将早返回 nfs_open 改为 openflags--；Context+Diff 与多个模型排序反向。注意力主要集中在其后背景行。  机理判断：Unknown。

每个预选样本的全局行投影范围、1–5倍极值、修改行分数、注意力前三行和历史局部头两类前三行见 history.cases.json。
这些样本的逐行资料是历史诊断，不作为新局部头的标签、特征或特判。
