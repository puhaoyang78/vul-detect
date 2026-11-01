# 函数自适应上下文调节context_scale：完整512实验

官方Train 8708函数/4354对、Valid 1124函数/562对。两组context_scale训练均完整16轮/17424步，Seed123456，Batch8，原优化器与固定概率>.5。复用同协议Context对照。Test未读取。
仅按完整Valid P-C选epoch，四位小数严格改善、并列取较早轮；不搜索超参数。
alpha(g)=2*sigmoid(Linear(LayerNorm(g)))，LayerNorm无可训练参数，Linear(768,1)权重/偏置均零初始化，alpha初始严格为1。仅新增769参数，不推进原模型或采样RNG。分数为W(h_i)+alpha(g)*dot(U(h_i),V(g))/sqrt(32)，保留K=5、有界MIL、函数BCE和固定.1区域损失；不组合局部注意力等其他模块。

|配置|Epoch|P-C|AUC|MCC|F1|漏洞Recall|修复侧FPR|
|---|---:|---:|---:|---:|---:|---:|---:|
|context_bce|16|0.1548|0.5759|0.0982|0.5310|0.5107|0.4128|
|context_diff|14|0.1779|0.5896|0.1431|0.6070|0.6637|0.5231|
|context_random|15|0.1637|0.5754|0.1014|0.5495|0.5480|0.4466|
|context_scale_diff|15|0.1584|0.5824|0.1201|0.5851|0.6210|0.5018|
|context_scale_random|14|0.1530|0.5759|0.0926|0.5355|0.5231|0.4306|

所选checkpoint在完整Train/Valid上的编辑区域/背景平均倍率；先单函数预测，再对照原始Diff统计，仅对非平凡可见区域取函数平均。

|配置|Split|有效函数|空区域|全区域|修改区域倍率|
|---|---|---:|---:|---:|---:|
|context_bce|train|5180/8708|3516|12|1.5310|
|context_bce|valid|658/1124|464|2|1.3374|
|context_diff|train|5180/8708|3516|12|3.1495|
|context_diff|valid|658/1124|464|2|1.7831|
|context_random|train|5180/8708|3516|12|1.2677|
|context_random|valid|658/1124|464|2|1.1592|
|context_scale_diff|train|5180/8708|3516|12|3.2495|
|context_scale_diff|valid|658/1124|464|2|1.8224|
|context_scale_random|train|5180/8708|3516|12|1.2504|
|context_scale_random|valid|658/1124|464|2|1.1147|
|context_scale_diff_context_scale_one|train|5180/8708|3516|12|3.0427|
|context_scale_diff_context_scale_one|valid|658/1124|464|2|1.7654|
|context_scale_random_context_scale_one|train|5180/8708|3516|12|1.2366|
|context_scale_random_context_scale_one|valid|658/1124|464|2|1.1016|

context_scale+真实Diff相对其他配置的差值：

|对照|P-C|AUC|MCC|F1|Recall|修复侧FPR|
|---|---:|---:|---:|---:|---:|---:|
|context_bce|+0.003559|+0.006505|+0.021973|+0.054090|+0.110320|+0.088968|
|context_diff|-0.019573|-0.007127|-0.022963|-0.021918|-0.042705|-0.021352|
|context_random|-0.005338|+0.007008|+0.018701|+0.035570|+0.072954|+0.055160|
|context_scale_random|+0.005338|+0.006576|+0.027500|+0.049561|+0.097865|+0.071174|

每函数alpha的完整Train/Valid分布；std为总体标准差。分位数和8个固定等宽区间[0,2]的频数另见[alpha汇总](context_scale_alpha.csv)、[alpha直方图](context_scale_alpha_histogram.csv)。alpha只由单函数代码计算。

|配置|Split|函数数|alpha均值|std|最小值|P05|P50|P95|最大值|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|context_scale_diff|train|8708|1.784480|0.207186|0.797159|1.364327|1.847121|1.972827|1.979096|
|context_scale_diff|valid|1124|1.833721|0.180335|0.995609|1.433856|1.922162|1.973087|1.979317|
|context_scale_random|train|8708|1.813095|0.154032|1.066038|1.471644|1.882445|1.943605|1.964098|
|context_scale_random|valid|1124|1.820224|0.139064|1.169722|1.521961|1.880676|1.942456|1.955937|

同一所选checkpoint将alpha固定为1，保留其余参数、输入和阈值，不重训练、不重新选epoch。以下为完整Valid推理消融；Train/Valid固定1分类见[完整消融数据](context_scale_one.csv)，关注倍率见上表。

|配置|Epoch|P-C|AUC|MCC|F1|Recall|修复侧FPR|
|---|---:|---:|---:|---:|---:|---:|---:|
|context_scale_diff alpha=1|15|0.158363|0.581857|0.120236|0.586466|0.624555|0.505338|
|context_scale_random alpha=1|14|0.160142|0.576907|0.106795|0.547748|0.540925|0.434164|

差值分解：learned_minus_one是在同一权重上启用调节的即时作用；one_minus_control比较固定1后的新权重与相应原Context对照，保留训练所得参数和选模轮次的差异。两者相加为learned_minus_control，不能将独立训练差异全部归因于推理时alpha。

|配置|差值|P-C|AUC|MCC|F1|Recall|修复侧FPR|
|---|---|---:|---:|---:|---:|---:|---:|
|context_scale_diff|learned_minus_one|+0.000000|+0.000571|-0.000110|-0.001387|-0.003559|-0.003559|
|context_scale_diff|one_minus_control|-0.019573|-0.007698|-0.022853|-0.020531|-0.039146|-0.017794|
|context_scale_diff|learned_minus_control|-0.019573|-0.007127|-0.022963|-0.021918|-0.042705|-0.021352|
|context_scale_random|learned_minus_one|-0.007117|-0.001054|-0.014169|-0.012229|-0.017794|-0.003559|
|context_scale_random|one_minus_control|-0.003559|+0.001486|+0.005370|-0.001762|-0.007117|-0.012456|
|context_scale_random|learned_minus_control|-0.010676|+0.000432|-0.008799|-0.013990|-0.024911|-0.016014|

按完整Valid P-C，context_scale+真实Diff超过context_scale+随机Diff，未超过原context_diff。
本轮未超过现有context_diff的P-C，按无净提升报告，不追加实验变体。alpha或区域倍率发生变化不能代替分类收益。
已核对数据、配置、输入覆盖、CPU采样和CUDA dropout RNG；真实/随机区域掩码分别与原Context对照一致。检查见[tests.log](tests.log)、[gpu-smoke.json](gpu-smoke.json)，命令见[commands.json](commands.json)。权重保留本地。
