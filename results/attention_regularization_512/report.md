# 有界注意力泛化：独立正则化对照

Train4354对/8708函数，Valid562对/1124函数，512 Token，Seed123456；每组完整16轮/17424步。Test未读取。
四组均从同一预训练UniXcoder初始化，K=5、原始双侧Diff、区域损失权重.1、函数分类头与优化器保持一致。
只按完整Valid P-C选模（四位小数严格改善，平局保留较早epoch），概率>.5。没有搜索参数或组合正则。

|方法|子集|Epoch|Pairs|P-C|AUC|MCC|F1|漏洞Recall|修复侧FPR|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|diff_region|all|12|562|0.1601|0.5873|0.1283|0.6326|0.7598|0.6423|
|diff_region|addition_only|12|163|0.2147|0.6281|0.1727|0.6344|0.7239|0.5583|
|random_suppression|all|12|562|0.1512|0.5834|0.1144|0.5960|0.6548|0.5427|
|random_suppression|addition_only|12|163|0.2086|0.6312|0.1806|0.6215|0.6748|0.4969|
|high_suppression|all|14|562|0.1726|0.5853|0.1107|0.5353|0.5125|0.4021|
|high_suppression|addition_only|14|163|0.2331|0.6294|0.1806|0.5503|0.5031|0.3252|
|ema_attention|all|16|562|0.1690|0.5845|0.1220|0.5870|0.6246|0.5036|
|ema_attention|addition_only|16|163|0.1963|0.6214|0.1657|0.5854|0.5890|0.4233|

纯新增按完整原始Diff旧侧无删除、新侧有新增定义，保留旧侧无区域监督；不补邻接或切片标签。

## 实现与公平性

random/high只把floor(.1*N)条有效行倍率改为1，同一次编码的正常与抑制聚合BCE取平均；原区域损失始终取正常倍率。high按detach的r_i排序，平局按物理行；random使用独立、可恢复的Python RNG。不删除Token，不缩放其余行。
EMA复制encoder+attention head，eval/no_grad；每个实际optimizer.step后无条件更新.99一次。KL(reference||online)按有效行求和、函数平均，.1*KL明确加入实际累积objective。best checkpoint及正常推理仅包含在线模型。
四组都重跑；旧对照只存最佳/最后权重，无法恢复逐epoch诊断。输入、环境、初始化规则、训练预算和诊断成员均已核对。EMA更新17424次，独立随机状态与EMA随last.pt保存，结束时四组CPU采样RNG一致。

## 最佳checkpoint完整Train/Valid关注量

|方法|Split|有效区域函数|区域/背景倍率|关注富集|编辑Token比例|
|---|---|---:|---:|---:|---:|
|diff_region|train|5180/8708|2.9746|2.3339|0.0826|
|diff_region|valid|658/1124|1.7558|1.4673|0.0835|
|random_suppression|train|5180/8708|2.9918|2.3445|0.0826|
|random_suppression|valid|658/1124|1.7337|1.4565|0.0835|
|high_suppression|train|5180/8708|2.8966|2.2528|0.0826|
|high_suppression|valid|658/1124|1.7868|1.4820|0.0835|
|ema_attention|train|5180/8708|2.9906|2.3559|0.0826|
|ema_attention|valid|658/1124|1.7171|1.4494|0.0835|

这些统计来自所选在线checkpoint的eval单函数预测，再对照Diff计算；完整Train/Valid关注量不代替分类结果。
按输入结构，Train中8397/8708函数满足N≥10、可执行非空抑制。Valid对应1068/1124，仅用于长度诊断，推理不启用抑制。

## 固定成员的逐epoch诊断

每组相同Train256/Valid256函数（各128对），只用于eval诊断，不用于选择epoch。成员见各config.json；逐函数概率、行倍率、坐标、JSD保存在attention.diagnostics.jsonl，汇总见[attention_epochs.csv](attention_epochs.csv)。
JSD使用自然对数，第一轮相邻epoch JSD为null。均匀参照为n_i/Σn_j，表示每Token等权；同时报告行注意力熵和倍率CV，不能把行等权误当Token均值。诊断保存/恢复Python、NumPy、Torch与CUDA随机状态，恢复保留前一epoch分布及日志边界。

以下汇总全部15个相邻过渡（第2–16轮），固定成员和学习率阶段对齐。最佳checkpoint的epoch不同，其单轮JSD不能直接归因于正则差异。

|方法|Split|平均相邻JSD|平均到Token均值JSD|平均倍率CV|
|---|---|---:|---:|---:|
|diff_region|train|0.009548|0.047076|0.682864|
|diff_region|valid|0.008533|0.047943|0.684684|
|random_suppression|train|0.009772|0.046401|0.677264|
|random_suppression|valid|0.009114|0.047913|0.685026|
|high_suppression|train|0.011757|0.049091|0.697341|
|high_suppression|valid|0.010566|0.048264|0.689171|
|ema_attention|train|0.006850|0.041718|0.638416|
|ema_attention|valid|0.006407|0.042351|0.638219|

同轮对齐的Valid平均相邻JSD相对对照变化：random +6.81%，high +23.83%，EMA -24.91%。不能以较晚最佳epoch的较小JSD直接声称抑制提高稳定性。
EMA的平均到Token均值JSD为0.042351、倍率CV为0.638219；这些值描述注意力与等权的距离，不能代替分类指标。

下表列出各自最佳epoch及最终epoch的固定成员统计（相同epoch只列一次）：

|方法|Epoch|Split|区域/背景倍率|相邻epoch JSD|到Token均值JSD|倍率CV|
|---|---:|---|---:|---:|---:|---:|
|diff_region|12|train|3.085307|0.004408|0.052053|0.727861|
|diff_region|12|valid|1.709237|0.005283|0.053206|0.732599|
|diff_region|16|train|3.153218|0.000382|0.054287|0.744418|
|diff_region|16|valid|1.690033|0.000489|0.056311|0.753422|
|random_suppression|12|train|3.038056|0.005908|0.050338|0.714503|
|random_suppression|12|valid|1.738730|0.005661|0.052368|0.732808|
|random_suppression|16|train|3.120848|0.000474|0.052014|0.728384|
|random_suppression|16|valid|1.704352|0.000565|0.055652|0.754002|
|high_suppression|14|train|2.925968|0.002485|0.055214|0.731274|
|high_suppression|14|valid|1.730842|0.003304|0.055995|0.737178|
|high_suppression|16|train|2.949392|0.000463|0.055384|0.740760|
|high_suppression|16|valid|1.730919|0.000725|0.056206|0.751374|
|ema_attention|16|train|2.971127|0.000181|0.047634|0.681688|
|ema_attention|16|valid|1.694405|0.000304|0.049757|0.701490|

## 分类差值与结论

random_suppression减当前方法：P-C -0.008897，AUC -0.003869，MCC -0.013914，Recall -0.104982，修复侧FPR -0.099644。
high_suppression减当前方法：P-C +0.012456，AUC -0.002042，MCC -0.017541，Recall -0.247331，修复侧FPR -0.240214。
ema_attention减当前方法：P-C +0.008897，AUC -0.002799，MCC -0.006264，Recall -0.135231，修复侧FPR -0.138790。

纯新增163对的P-C正确对数：对照35，random 34，high 38，EMA 32。该子集不能由整体P-C改善代替解释。

三种新正则的AUC、MCC均低于对照。部分P-C上升伴随漏洞Recall与修复侧FPR下降，不能将其等同于一致的分类泛化收益。本轮停止，不追加模块或参数搜索。

Train倍率下降、Train/Valid差距缩小、相邻epoch JSD下降或趋于均匀，均不能单独证明分类泛化改善。新增正则只改变训练；推理不读取Diff、不依赖EMA或抑制。

[相关测试](tests.log)、[诊断隔离](diagnostic-isolation-test.log)、[恢复配置复验](resume-config-test.log)、[真实GPU检查](gpu-smoke.json)、[正式入口smoke](training-smoke.json)、[旧对照核验](reference-check.json)及[四组命令](commands.json)保留实际记录。smoke明确不是完整实验结果。权重保留本地，不上传。训练时间包含两卡并发影响，不能据此直接宣称单独运行加速。
