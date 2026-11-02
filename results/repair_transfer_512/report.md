# 跨样本修复区域迁移：冻结模型离线检索

固定Train24/Valid24纯新增修复审计组，参考库仅来自Train漏洞侧具有可见直接Diff的函数。使用原Context+Diff所选checkpoint，eval/冻结参数，正式512输入与bounded_pool行表示；没有新训练、参数搜索或Test使用。

直接检索=max正参考余弦；对比检索=max正参考余弦−max背景余弦。注意力对照使用实际MIL行质量，随机对照Seed123456。完全重复代码、同配对和重复参考文本按预先固定协议隔离；正/背景冲突文本排除。

所有方法共用同一可见行集合，逐行复核Top3并集。Unknown保留在相关比例和样本命中率的分母；未复核行不会当负例。Top1是同一排序的前缀。

related包含补丁明确约束的旧操作，以及源码可确认的数据定义和控制上下文；上下文命中不等于恢复实际危险操作。只有注释、括号、通用函数签名、日志或不同字段的检查不算操作关联。调用/类型契约不足者保留Unknown。判断来自既有与本次助手源码/补丁复核，并非独立人工漏洞定位金标。

|Split|方法|K|相关行/所选行|相关比例|至少一行相关|Unknown比例|
|---|---|---:|---:|---:|---:|---:|
|train|positive_nn|1|4/24|0.1667|4/24|0.0833|
|train|positive_nn|3|14/72|0.1944|6/24|0.0417|
|train|contrastive_nn|1|10/24|0.4167|10/24|0.0417|
|train|contrastive_nn|3|20/72|0.2778|14/24|0.0556|
|train|context_diff_attention|1|11/24|0.4583|11/24|0.0417|
|train|context_diff_attention|3|25/72|0.3472|14/24|0.0556|
|train|random|1|4/24|0.1667|4/24|0.0417|
|train|random|3|11/72|0.1528|8/24|0.0556|
|valid|positive_nn|1|5/24|0.2083|5/24|0.2500|
|valid|positive_nn|3|13/72|0.1806|9/24|0.1528|
|valid|contrastive_nn|1|6/24|0.2500|6/24|0.1250|
|valid|contrastive_nn|3|14/72|0.1944|10/24|0.1667|
|valid|context_diff_attention|1|11/24|0.4583|11/24|0.0833|
|valid|context_diff_attention|3|28/72|0.3889|14/24|0.1111|
|valid|random|1|6/24|0.2500|6/24|0.1250|
|valid|random|3|20/72|0.2778|14/24|0.1111|

配对Top3样本命中变化（同一查询中新命中/丢失，完整成员见[paired_hits.csv](paired_hits.csv)）：

|Split|检索方法|对照|新增命中|丢失命中|
|---|---|---|---:|---:|
|train|positive_nn|context_diff_attention|2|10|
|train|positive_nn|random|3|5|
|train|contrastive_nn|context_diff_attention|2|2|
|train|contrastive_nn|random|8|2|
|valid|positive_nn|context_diff_attention|2|7|
|valid|positive_nn|random|3|8|
|valid|contrastive_nn|context_diff_attention|1|5|
|valid|contrastive_nn|random|3|7|

静态未恢复组及完整/截断源码分层见[subsets.csv](subsets.csv)；同项目与跨项目匹配质量见[project_bias.csv](project_bias.csv)。这些是条件统计，不能视为项目或长度的因果效应。逐函数结果见[functions.quality.csv](functions.quality.csv)，命中/未命中的源码Token数、物理行数和可见行数中位数见[length_bias.csv](length_bias.csv)。

以下列出静态未恢复且检索新增命中、注意力Top3未命中的全部候选行。仅用于事后解释，不参与检索或参数选择。表内物理行号为1起始，JSON/CSV原始行号为0起始。

|查询配对|方法/排名|查询行|正参考来源/行|源码关系复核|
|---|---|---|---|---|
|primevul_paired:train:2396|contrastive_nn/2|L53: `std::vector<Box_iref::Reference> references = iref_box->get_references_from(image->get_id());`|primevul_paired:train:5019/L44|references容器经ref.to_item_ID产生alpha分支的refs[0]，是可确认的数据来源上下文；不是master_iter解引用操作本身。 当前512只编码该行前缀，alpha分支的find及解引用都不可见。|
|primevul_paired:train:2532|positive_nn/1|L22: `header.deltaset = header_buffer[1];`|primevul_paired:train:4169/L22|header_size循环、s->buf[i+1]及header_buffer后续解析与新增包长检查可见相关；旧错误返回为上下文。|
|primevul_paired:train:2532|positive_nn/2|L28: `header.header_type = header_buffer[10];`|primevul_paired:train:4169/L22|header_size循环、s->buf[i+1]及header_buffer后续解析与新增包长检查可见相关；旧错误返回为上下文。|
|primevul_paired:train:2532|positive_nn/3|L27: `header.version = header_buffer[9];`|primevul_paired:train:4169/L22|header_size循环、s->buf[i+1]及header_buffer后续解析与新增包长检查可见相关；旧错误返回为上下文。|
|primevul_paired:train:2532|contrastive_nn/2|L19: `header_buffer[i - 1] = s->buf[i] ^ s->buf[i + 1];`|primevul_paired:train:4451/L29|header_size循环、s->buf[i+1]及header_buffer后续解析与新增包长检查可见相关；旧错误返回为上下文。|
|primevul_paired:train:2532|contrastive_nn/3|L3: `int i, ret;`|primevul_paired:train:7283/L3|声明header_size循环所用的i，是受新增包长检查保护的s->buf[i+1]索引上下文。|
|primevul_paired:train:2548|positive_nn/3|L11: `*write_info;`|primevul_paired:train:4251/L10|声明新增DestroyImageInfo释放的同一write_info资源指针，是资源生命周期定义上下文；实际失败路径在可见范围外。|
|primevul_paired:valid:175|positive_nn/2|L34: `encap->n_props = n_props;`|primevul_paired:train:4199/L31|新增encap重新取址紧邻此行之前；这是直接被修复的循环后encap指针字段写入。|
|primevul_paired:valid:340|positive_nn/3|L24: `dec->xend = siz->width;`|primevul_paired:train:4169/L22|width/height关联总像素数上限；tilewidth/tileheight及numcomps是不同字段，分量数组分配不能一起扩展，offset关系未知。|
|primevul_paired:valid:340|contrastive_nn/1|L24: `dec->xend = siz->width;`|primevul_paired:train:4169/L22|width/height关联总像素数上限；tilewidth/tileheight及numcomps是不同字段，分量数组分配不能一起扩展，offset关系未知。|

匹配来源与正/背景相似度：[rankings.jsonl](rankings.jsonl)。候选并集与完整源码判断：[review.jsonl](review.jsonl)。行表示和完整参考元数据缓存留本地；原始数据和最佳checkpoint不改写。

本次实际执行使用官方完整配对Train 8,708函数（4,354对），从中得到2,181个具有可见直接Diff正行的漏洞侧参考函数；Valid只有既定24个漏洞侧查询进入编码，没有进入参考库。71,777条原始参考行共47,626种去空白首尾后的文本，排除686种正/背景标签冲突文本，剩46,940种（正5,952、背景40,988）。每个查询每种文本只使用首个合格来源；[reference_origins.csv](reference_origins.csv)保留54,277条合格来源记录及text_group，用于追溯可替换来源，不代表实际重复使用这些参考行。48组查询的有效参考库大小均为5,952/40,988；同配对与源码克隆排除路径已测试，本组实际排除计数均为0。

462条候选并集最终为105条related、318条unrelated、39条Unknown。复用125条既有且逐字源码一致的判断，另337条由助手逐条结合旧函数、实际补丁和控制/数据关系复核。全部有具体理由，没有pending或以未标注充当负例。注释也按正式Token掩码进入共同候选集合，没有另行过滤。部分可见的物理行仍保留，完整标记见queries.jsonl；例如Train2396的L53只有前缀被编码，因此其命中仅说明共同引用来源上下文，不能声称模型看到了后面的alpha查找及解引用。Valid两种检索Top3中的相关行均为完整可见行。

静态分析未恢复组的结果如下（Train22/Valid22；分母为对应查询的66条Top3行）：

|方法|Train相关行|Train命中样本|Valid相关行|Valid命中样本|
|---|---:|---:|---:|---:|
|直接检索|11/66|5/22|8/66|7/22|
|对比检索|18/66|12/22|10/66|8/22|
|Context+Diff注意力|20/66|12/22|24/66|12/22|
|固定随机|8/66|6/22|17/66|12/22|

针对上一轮专门检查的12个Train未恢复保护检查样本（1190、1517、1651、1815、2349、2396、2426、2455、2721、3735、664、937），直接检索相关6/36行、命中2/12、Unknown2/36；对比检索相关8/36、命中5/12、Unknown4/36；注意力相关9/36、命中6/12、Unknown2/36；随机相关7/36、命中5/12、Unknown1/36。直接检索没有新增注意力未命中的样本；对比检索唯一新增的2396只找到上述部分可见的共享references定义，随机对照也命中同函数的控制上下文。没有证据表明它解决了这12例的静态恢复缺口。

真实相关案例（以下行号均从1开始，完整匹配代码及余弦见rankings.jsonl）：

- Train2455，Bento4 `AP4_HvccAtom`：新增`payload_size < 22`拒绝过短头部。直接检索的L18/L22/L21分别读取`payload[15]`、`payload[17]`、`payload[16]`，3/3确实受最小长度保护。最近正参考为Train样本4169/L33的QEMU字段读取，余弦约0.9912/0.9907/0.9895。这是相关区域命中；该查询的注意力和随机Top3也均为3/3，不能记为新增样本收益。
- Train2532，FFmpeg `truemotion1_decode_header`：新增`header.header_size + 1 > s->size`拒绝短包。对比检索Top2命中L19的`s->buf[i] ^ s->buf[i + 1]`读取，正参考来自Train4451/L29的libtiff解码输入调用；直接检索找到由同一header_buffer解析出的字段。注意力和随机Top3均未命中，这是Train中真实的新增操作关联。
- Valid175，OVS `decode_NXAST_RAW_ENCAP`：补丁在属性解码循环之后重新从out取得encap，紧接旧L34的`encap->n_props = n_props`。直接检索Top2命中该字段写入；正参考为Train4199/L31、QEMU的`ssp.data_bits = data_bits`，余弦0.83047。注意力Top3未命中；随机Top1已经命中使用encap的收尾调用。因此存在真实区域命中，但不能声称它超过两种对照。
- Valid135，TensorFlow `MakeEdge`：对比检索Top1/2命中`dst->input_type(input_index)`及`src->output_type(output_index)`，确实被新增索引上界检查保护；注意力主要找到后续类型兼容性上下文。这是操作级候选质量的局部改善，整组结果仍较差。
- Valid340，JasPer：两种检索找到L24的`dec->xend = siz->width`，注意力和随机均未命中。该行是新增全图宽高乘积上限的值使用上下文；最近正参考却是QEMU的`s->function = qemu_get_byte(f)`，不能据此认定检索恢复了相同漏洞机制。直接检索另两行x/y偏移仍为Unknown。

典型失败及已确认的原因：

- 余弦相似并不保证修复关系。Valid315的GIF补丁检查width/height非零，旧L25/L26尺寸读取就在512范围内，但直接检索Top1命中L20的版本警告，余弦高达0.99144；对比检索则选择版本处理注释和警告，全部不相关。Train937的SASL参数内容过滤也被服务器选择和主机名格式化所替代。
- 减背景的差值可能偏向弱匹配。Valid175真正的L34字段写入正/背景余弦为0.83047/0.85618，差为负；对比Top1却选择“Add supported encap header types”注释，正/背景余弦只有0.54841/0.51799，差0.03042。背景扣除没有可靠地把修复关系排在前面。
- 有些真实操作不在512范围内。Train1190 `ReadCINImage`新增尺寸检查位于旧侧约L351附近，而模型只看到L1–L74；检索集中在签名、断言和魔数校验，四种方法均未命中。Train2396的alpha `master_iter`查找/解引用在L133–L139，同样不可见。变量或共同控制状态被命中不等于这些操作已恢复。
- 有些修复针对旧侧尚不存在的状态。Valid377仅有5行，新增清空`highest_sack`，旧函数只操作另一字段`sk_send_head`。检索、注意力和随机只找到旧字段或括号，均没有可确认的相关行；不能因共享skb变量而将其算作正确修复位置。

源码长度的影响：Train/Valid各24查询的源码Token中位数为1,232/516，物理行数中位数111.5/51.5，截断函数为17/12个；共同可见行总数961/711。Valid直接检索命中/未命中函数的源码Token中位数为279/1,255，对比检索为512.5/853.5。按自然512截断边界分层如下，没有选择其他长度阈值：

|Valid组|方法|Top3相关比例|样本命中|
|---|---|---:|---:|
|完整可见12函数|直接检索|9/36=25.00%|6/12|
|完整可见12函数|对比检索|9/36=25.00%|5/12|
|完整可见12函数|注意力|17/36=47.22%|8/12|
|完整可见12函数|随机|14/36=38.89%|8/12|
|截断12函数|直接检索|4/36=11.11%|3/12|
|截断12函数|对比检索|5/36=13.89%|5/12|
|截断12函数|注意力|11/36=30.56%|6/12|
|截断12函数|随机|6/36=16.67%|6/12|

检索在完整可见组仍落后，不能只用截断解释失败。项目方面，Valid直接/对比Top3正参考的同项目匹配仅5/72（6.94%）和6/72（8.33%）；同项目相关比例分别2/5、2/6，跨项目为11/67、12/66。背景参考的同项目比例为8/72、5/72。少量同项目匹配质量较高，但数量很少；大部分跨项目候选仍不相关。项目及长度都是观察到的分层差异，没有进行项目排除重跑或参数选择。

执行验证：GPU前64项相关单元测试通过，覆盖正式物理行聚合、跨行Token质量守恒、Padding、冻结eval、Diff标签不参与编码、Train来源隔离、重复源码/参考行、排序与Unknown分母；增加报告及防覆盖保护后，16项离线模块测试复验通过。真实RTX4090提取768维行表示，同批重复的嵌入与概率逐位相同，所有2,229个已编码函数的行号及Token数与已有Context推理一致，概率/倍率最大差分别8.79e-7/1.43e-5；48组完整GPU检索及来源隔离重复一致。原参数无梯度或版本变化，checkpoint大小和mtime未变。环境Torch2.9.1+cu128、CUDA12.8，所用Context+Diff checkpoint为Valid P-C选出的epoch14（P-C 100/562）；命令、Seed123456及源码基线提交见[commands.json](commands.json)，详细检查见[extraction.json](extraction.json)、[retrieval.json](retrieval.json)、[tests.log](tests.log)、[report-tests.log](report-tests.log)。

结论：本次跨样本相似性检索实验失败。两种方法在Valid的Top1/Top3相关比例及Top3样本命中率均未超过注意力；Top3还低于固定随机对照，Unknown比例更高。少量新增命中不足以抵消丢失的正确样本，其中部分仅是定义或控制上下文。当前结果不支持把这套检索作为弱监督接入Context+Diff。没有训练新模型、读取Test、修改最佳checkpoint或调整划分，也没有追加参数搜索或实验变体。
