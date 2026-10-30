# 补丁引导的有界注意力MIL：完整512实验

官方Train4354对/8708函数，Valid562对/1124函数；四组均从相同预训练UniXcoder初始化，Seed123456、16 Epoch、每组17424步。Test未读取。
A=global_mean，B=attention_bce，C=diff_region，D=random_region。
保持原512输入、有效代码Token掩码、Linear(768,1)函数头和优化器。B/C/D仅增加零初始化关注头；K=5。C/D使用固定0.1区域损失，不搜索超参数。
各组仅按完整Valid P-C选epoch，四位小数严格改善；平局保留较早epoch。固定概率>.5。

|模型|组|Epoch|Pairs|P-C|AUC|MCC|F1|漏洞Recall|修复侧FPR|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|global_mean|all|13|562|0.1637|0.5726|0.1141|0.5699|0.5872|0.4733|
|global_mean|addition_only|13|163|0.2025|0.5933|0.1479|0.5924|0.6196|0.4724|
|attention_bce|all|14|562|0.1566|0.5856|0.1304|0.4595|0.3737|0.2527|
|attention_bce|addition_only|14|163|0.1963|0.6189|0.1703|0.4907|0.4049|0.2454|
|diff_region|all|12|562|0.1601|0.5873|0.1283|0.6326|0.7598|0.6423|
|diff_region|addition_only|12|163|0.2147|0.6281|0.1727|0.6344|0.7239|0.5583|
|random_region|all|13|562|0.1530|0.5775|0.1013|0.5866|0.6388|0.5391|
|random_region|addition_only|13|163|0.2270|0.6175|0.1672|0.6092|0.6503|0.4847|
|diff_region_uniform|all|12|562|0.1103|0.5835|0.1164|0.6601|0.8897|0.8060|
|diff_region_uniform|addition_only|12|163|0.1472|0.6292|0.1631|0.6667|0.8712|0.7423|

`addition_only`按完整原始双侧Diff定义：旧侧无删除行、新侧有新增行，不把截断导致的不可见误作纯新增。没有丢弃这些函数的BCE。

## 关注量诊断（所选checkpoint、eval模式）

B/C/D的Train/Valid统计来自所选checkpoint的单函数eval预测，再对照Diff计算；Valid Diff从不传入模型。A的倍率恒为1，关注量按定义计算，没有伪造A的Train分类预测。比例只表示编辑区域关注，不代表漏洞定位准确率。
|模型|Split|区域|有效函数|Token覆盖micro|区域/背景平均倍率|关注富集|
|---|---|---|---:|---:|---:|---:|
|attention_bce|train|original_diff|5180|0.0826|1.0749|1.0609|
|attention_bce|valid|original_diff|658|0.0835|1.0711|1.0546|
|diff_region|train|original_diff|5180|0.0826|2.9746|2.3339|
|diff_region|valid|original_diff|658|0.0835|1.7558|1.4673|
|global_mean|train|original_diff|5180|0.0826|1.0000|1.0000|
|global_mean|valid|original_diff|658|0.0835|1.0000|1.0000|
|random_region|train|fixed_random|5180|0.0726|1.9716|1.8091|
|random_region|train|original_diff|5180|0.0826|1.2879|1.2009|
|random_region|valid|original_diff|658|0.0835|1.1383|1.0979|

随机对照保持可见正行数量及监督可用性；3624/8708个函数集合完全相同，其中有效区域中重合96/5180。没有强制重抽消除巧合，也没有声称Token数严格匹配。

train：5180/8708个函数具有非平凡的可见编辑区域；空区域3516，全Token区域12。
valid：658/1124个函数具有非平凡的可见编辑区域；空区域464，全Token区域2。
Train的全部函数保留BCE，只有非平凡区域参与区域损失；Valid仅在预测后统计区域，不用于训练。

## 分类收益与解释

C减global_mean：P-C -0.003559，AUC +0.014732，MCC +0.014178，F1 +0.062644。
C减attention_bce：P-C +0.003559，AUC +0.001653，MCC -0.002180，F1 +0.173074。
C减random_region：P-C +0.007117，AUC +0.009783，MCC +0.027006，F1 +0.045991。
C减diff_region_uniform：P-C +0.049822，AUC +0.003761，MCC +0.011839，F1 -0.027473。

C没有同时超过A和B的Valid P-C；按预定标准，本轮未验证净分类收益，按无效报告，不叠加模块或搜索参数。

C相对A：漏洞Recall 0.5872→0.7598，修复侧FPR 0.4733→0.6423。F1提升不能代替成对判别收益。C减A的AUC为+0.0147、P-C正确对数变化为-2；C减B的MCC为-0.0022。
纯新增组共163对：C P-C为0.2147，A为0.2025，D为0.2270。真实位置相对随机位置的P-C差为-0.0123。
区域/背景平均倍率：Train 2.9746、Valid 1.7558；达到2倍的有效函数比例分别为87.28%与40.73%。这些统计用于检查区域监督的泛化差距，不能直接作为分类失败的因果解释。

`diff_region_uniform`是同一个C最佳checkpoint的仅推理干预：倍率全部置1，编码器与分类头不变；不重新训练、不重新选epoch。它不是独立训练的Baseline。关注量提高本身不能代替分类收益。
保留注意力较等权干预多判对28对，P-C 0.1103→0.1601，AUC 0.5835→0.5873。这说明注意力在C自身checkpoint中实际参与分类，不代表C训练方案优于独立训练的A。

## 正确性与复现

[映射检查](mapping-check.json)、[真实GPU检查](gpu-smoke.json)、[相关测试](tests.log)、[新增恢复测试](resume-tests.log)及[命令](commands.json)保留实际记录。全量Train/Valid没有有效Token重复行成员；实现对跨行Token保留总质量，等倍率等价于原Token均值。
注意力头零初始化时区域损失对encoder的梯度为0，这是链式法则的结果；头更新后真实GPU区域梯度进入encoder，见gpu-smoke.json。函数BCE从第一步正常训练。
[训练损失](training.csv)、[逐函数关注统计](region_statistics.csv)、[随机重合](random_overlap.csv)与各组逐样本预测支持复核。R为空或覆盖全部有效Token只跳过区域项，全部函数仍参与BCE。
四个完整任务在两张4090上并发；训练时间是并发墙钟记录，不用它推断方法单独运行的加速。权重留本地，不上传；没有读取Test、追加参数搜索或模块。
