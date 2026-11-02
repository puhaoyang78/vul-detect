# 已完成512实验的固定checkpoint Test对照

本轮按用户后续要求，覆盖七个既有实验系列的22组完整分类模型：21组新评估，复用此前已验证的Context+Diff Test结果。固定官方配对Test为1128函数/564对；所有模型原训练均为Seed123456、512输入、Train8708/Valid1124函数、16轮/17424步、相同原优化器配置。每组保持原完整Valid P-C选出的checkpoint和概率严格>.5，Test不用于更换epoch、阈值或参数。此前各系列报告中的未读取Test对应原训练及选模阶段。

Pilot仅训练子集；静态审计和冻结跨样本检索没有新的函数分类checkpoint；本轮只评价上述完整512分类实验。复用正式evaluate CLI、单函数源码推理及既有物理行映射，权重未修改；抑制/EMA仅使用原在线模型的正常推理。

|实验/配置|Epoch|Valid P-C|Test P-C|Test正确对数|AUC|MCC|F1|Recall|修复侧FPR|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|[line_auxiliary_512/baseline](../line_auxiliary_512/baseline/primevul_paired/unixcoder/test.metrics.json)|14|15.48%|10.64%|60/564|0.5544|0.0771|50.05%|46.28%|38.65%|
|[line_auxiliary_512/diff_0.1](../line_auxiliary_512/diff_0.1/primevul_paired/unixcoder/test.metrics.json)|16|14.41%|12.23%|69/564|0.5716|0.0928|57.12%|60.46%|51.24%|
|[line_auxiliary_512/diff_0.3](../line_auxiliary_512/diff_0.3/primevul_paired/unixcoder/test.metrics.json)|10|14.23%|13.12%|74/564|0.5672|0.0959|53.47%|51.95%|42.38%|
|[line_auxiliary_512/random_0.1](../line_auxiliary_512/random_0.1/primevul_paired/unixcoder/test.metrics.json)|12|16.01%|10.82%|61/564|0.5564|0.0585|52.29%|51.60%|45.74%|
|[line_auxiliary_512/random_0.3](../line_auxiliary_512/random_0.3/primevul_paired/unixcoder/test.metrics.json)|15|13.35%|10.82%|61/564|0.5706|0.0997|57.02%|59.75%|49.82%|
|[bounded_attention_512/attention_bce](../bounded_attention_512/attention_bce/primevul_paired/unixcoder/test.metrics.json)|14|15.66%|12.06%|68/564|0.5618|0.0654|41.76%|33.69%|27.66%|
|[bounded_attention_512/diff_region](../bounded_attention_512/diff_region/primevul_paired/unixcoder/test.metrics.json)|12|16.01%|10.46%|59/564|0.5619|0.0879|61.98%|75.00%|67.02%|
|[bounded_attention_512/global_mean](../bounded_attention_512/global_mean/primevul_paired/unixcoder/test.metrics.json)|13|16.37%|13.83%|78/564|0.5669|0.1047|55.97%|56.91%|46.45%|
|[bounded_attention_512/random_region](../bounded_attention_512/random_region/primevul_paired/unixcoder/test.metrics.json)|13|15.30%|9.93%|56/564|0.5611|0.0662|56.19%|59.93%|53.37%|
|[attention_regularization_512/diff_region](../attention_regularization_512/diff_region/primevul_paired/unixcoder/test.metrics.json)|12|16.01%|10.46%|59/564|0.5619|0.0879|61.98%|75.00%|67.02%|
|[attention_regularization_512/ema_attention](../attention_regularization_512/ema_attention/primevul_paired/unixcoder/test.metrics.json)|16|16.90%|14.18%|80/564|0.5712|0.0959|56.03%|57.62%|48.05%|
|[attention_regularization_512/high_suppression](../attention_regularization_512/high_suppression/primevul_paired/unixcoder/test.metrics.json)|14|17.26%|13.48%|76/564|0.5740|0.0943|52.73%|50.53%|41.13%|
|[attention_regularization_512/random_suppression](../attention_regularization_512/random_suppression/primevul_paired/unixcoder/test.metrics.json)|12|15.12%|13.30%|75/564|0.5688|0.0992|60.03%|67.91%|58.33%|
|[attention_context_512/context_bce](../attention_context_512/context_bce/primevul_paired/unixcoder/test.metrics.json)|16|15.48%|12.23%|69/564|0.5691|0.0925|52.77%|50.71%|41.49%|
|[attention_context_512/context_diff](../attention_context_512/context_diff/primevul_paired/unixcoder/test.metrics.json)|14|17.79%|13.65%|77/564|0.5712|0.0885|58.30%|63.83%|55.14%|
|[attention_context_512/context_random](../attention_context_512/context_random/primevul_paired/unixcoder/test.metrics.json)|15|16.37%|11.17%|63/564|0.5579|0.0656|52.48%|51.60%|45.04%|
|[attention_local_512/dual_diff](../attention_local_512/dual_diff/primevul_paired/unixcoder/test.metrics.json)|13|16.37%|12.59%|71/564|0.5686|0.0941|60.08%|68.44%|59.40%|
|[attention_local_512/dual_random](../attention_local_512/dual_random/primevul_paired/unixcoder/test.metrics.json)|14|14.95%|9.93%|56/564|0.5543|0.0552|45.40%|39.36%|34.04%|
|[context_scale_512/context_scale_diff](../context_scale_512/context_scale_diff/primevul_paired/unixcoder/test.metrics.json)|15|15.84%|13.48%|76/564|0.5713|0.0982|57.33%|60.64%|50.89%|
|[context_scale_512/context_scale_random](../context_scale_512/context_scale_random/primevul_paired/unixcoder/test.metrics.json)|14|15.30%|10.46%|59/564|0.5559|0.0604|51.82%|50.53%|44.50%|
|[insertion_localization_512/insertion_diff](../insertion_localization_512/insertion_diff/primevul_paired/unixcoder/test.metrics.json)|14|15.66%|14.36%|81/564|0.5650|0.0986|51.20%|47.34%|37.59%|
|[insertion_localization_512/insertion_random](../insertion_localization_512/insertion_random/primevul_paired/unixcoder/test.metrics.json)|14|16.73%|13.48%|76/564|0.5742|0.1011|55.17%|55.32%|45.21%|

表格按既有实验系列及配置列出，不按Test重新选模。完整数值、其他分类指标及各次评估实际耗时见[classification.csv](classification.csv)。

插入定位额外复用正式间隙评价函数，Diff只用于定位真值评价；函数预测仍只读取各自源码。Hit@K的条件分母为有可靠512映射的纯新增漏洞函数；all-pure保留不可见函数分母。原无插入头模型不构造间隙预测。

|插入监督|可映射/全部纯新增|Top1|Top3|全纯新增Top1|全纯新增Top3|正间隙Recall@1|正间隙Recall@3|
|---|---:|---:|---:|---:|---:|---:|---:|
|insertion_localization_512/insertion_diff|125/189|16.80%|40.00%|11.11%|26.46%|13.91%|33.77%|
|insertion_localization_512/insertion_random|125/189|5.60%|16.00%|3.70%|10.58%|4.64%|13.25%|

所有导出成员、顺序和标签均核对为同一完整Test；分类指标由保存的概率复算，并独立按配对及固定阈值核对P-C计数。有界模型还核对同一512可见物理行集合、Token计数及1～5倍率。检查结果见[comparison.json](comparison.json)。

全部现有CLI命令、预先固定的选模轮次、GPU分配及配置见[commands.json](commands.json)；实际开始/结束时间、耗时和返回码见[GPU0执行记录](execution.gpu0.json)、[GPU1执行记录](execution.gpu1.json)，逐模型日志位于本目录。评估包含加载、Token映射、推理及导出，两张GPU同机运行；这些时间不单独代表模型计算开销。

本轮Test结果解读：

各系列保留其原对照及对应权重，不同历史轮次的baseline作为独立行记录。表中的原完整Valid P-C选模结果全部保留，Test只追加评价。

- 原Context+Diff为13.65%（77/564），低于本轮Global Mean对照13.83%（78/564）一对。
- 真实插入监督在这22组固定模型中的Test P-C最高，为14.36%（81/564），比Context+Diff多4对、比随机插入多5对；但AUC为0.5650，低于Context+Diff的0.5712和随机插入的0.5742。其Recall/FPR为47.34%/37.59%，Context+Diff为63.83%/55.14%。原Valid P-C仍为15.66%，低于Context+Diff的17.79%；不能据此宣称稳定、全面的检测提升或改选模型。
- EMA的Test P-C为14.18%，较其非Context真实Diff对照10.46%多21对，AUC和MCC也高于该对照；相对Global Mean，P-C只多2对，MCC为0.0959，低于Global Mean的0.1047。
- 七组既有真实/随机监督对照中，真实监督的Test P-C都更高；AUC/MCC并非全部同向。例如真实插入的AUC/MCC低于随机插入，0.3倍行辅助真实Diff的AUC/MCC也低于相应随机对照。因此位置学习、P-C变化与全面分类收益仍需分开报告。

插入定位Test包含189个纯新增漏洞函数，其中125个有可靠512映射。真实/随机Top1为16.80%/5.60%，Top3为40.00%/16.00%；含不可见位置的全纯新增Top3为26.46%/10.58%。这证明本轮定位信号优于随机对照，但未消除训练集与未见函数间的差距，也不替代上面的检测结果。

本批21次新评估在两张RTX4090上完成，北京时间2026-11-02 17:37:52至17:46:35，总墙钟耗时522.91秒；此前Context+Diff结果单独复用。未追加训练、组合模块、参数搜索或其他Test推理变体。
