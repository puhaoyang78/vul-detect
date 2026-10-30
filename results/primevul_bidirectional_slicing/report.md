# PrimeVul Original：Diff引导双版本双向切片审计

完成日期：2026-10-30。复用 `syntax.py`、`losver_patch_audit.py`、历史静态模式审计及已有LOSVER行预测。全部4354对Train、562对Valid均保留并尝试处理。没有修改原始数据、检测模型或权重，没有训练、重新执行模型推理或读取官方Test。

## 结论

**纯新增修复的候选区域恢复已得到真实证据，但尚未证明能自动产生普遍更准确的漏洞区域监督。** Valid旧关系复核中保留5/6条修复相关锚点、3/14条无关锚点，显示聚焦作用；其中一部分过滤来自解析不可用。对旧纯新增组中4个有输出配对的47条漏洞侧候选行穷尽复核，26条与可见修复相关、16条无关、5条Unknown。包含真实操作，也有明显的控制范围扩张、同对象不同字段混连和语句对应歧义。不能把这些候选整体作为高可信正行标签。本轮到此停止增加规则。

## 1. 实现和固定协议

- 单函数先独立生成Joern原生CFG、REACHING_DEF和CDG；Tree-sitter复用已有严格单函数解析、作用域工具和原子语句/控制条件。源码不补类型、不改写、不截断。原始文件没有语言/文件路径字段，扩展名沿用现有Tree-sitter语法选择；没有穷尽其它语言/预处理环境。Unknown表示本协议未获得可靠映射，不是证明源码永久不可解析。
- 真实编辑hunk覆盖的原子语句/控制条件为种子；相同源码行且Token相同的未变语句不当种子。每侧分别做前向、后向最多两跳的原生依赖遍历。**不沿CFG邻接顺序补行**，不按API名字或同名变量填区域。
- 两版本优先按相同源码行与相同非注释Token对应；否则只允许唯一的相同Token语句。一对一对应后仅迁移一次，再在目标侧做有界依赖遍历。语法相同不代表绑定、路径或对象语义等价。
- 每侧最多增加12条非种子语句；直接Diff语句保留。遍历按方向和源码语句顺序确定，不按模型概率或标签质量选行。前后向不互相重启；首次局部切片保留，防止跨版本阶段丢失路径中间节点。最大路径为两段局部依赖加一次版本映射，不是无限可达闭包。
- 路径保留原生边类型、变量、Joern节点ID、方向及版本映射来源；未知调用摘要、内存/别名、全局或宏绑定单列Unknown。采用语句级may-dependence，不是路径可行性/别名证明，也不是`reachableByFlows`的上下文/污点状态求解。
- 已核实错配的Train29/Valid6对仍保留，禁止其跨版本迁移。其直接编辑与局部候选不等于可靠修复监督；其它未裁定配对也不自动等于合法。
- Diff和修复版本仅用于离线Train监督候选与Valid复核，**不进入正常单函数推理**。没有将候选接入Localizer/Detector训练，也没有创建第三类函数标签。

原生边含义依据[Joern CPG规范](https://cpg.joern.io/)；批量解析和导出遵循[官方导出接口](https://docs.joern.io/export/)。这些来源说明图关系，不提供漏洞位置真值。

## 2. 全量覆盖与解析限制

| 项目 | Train | Valid |
|---|---|---|
| 原始配对/函数 | 4354 / 8708 | 562 / 1124 |
| Joern唯一源码方法 | 6082/8708 | 746/1124 |
| Tree-sitter与Joern同时可用函数 | 4876/8708 (55.99%) | 612/1124 (54.45%) |
| 两侧同时可用配对 | 2389/4354 (54.87%) | 301/562 (53.56%) |
| 漏洞侧非空区域 | 2367/4354 | 297/562 |
| 修复侧非空区域 | 2375/4354 | 303/562 |
| Tree-sitter严格解析未知函数 | 2122 | 238 |
| Joern方法无法可靠匹配函数 | 2626 | 378 |
| 无可用映射依赖的可解析函数 | 122 | 22 |

两类解析失败会重叠，不能相加；单侧可解析时仍保留该侧结果，但不生成无法确认的跨版本对应。所有未知成员保留在分母与样本表中。Train/Valid AST返回类型字段缺失2167/286个函数，其中2004/264个也没有可匹配Joern方法。这是关联，包含合法C++无显式返回类型情况，不能把它们全部认定为错误数据或失败的唯一原因。

| 结构编辑类型 | Train配对 | Train两侧可用 | Valid配对 | Valid两侧可用 |
|---|---|---|---|---|
| addition_only | 1232 | 706 | 163 | 88 |
| replacement_only | 1671 | 950 | 231 | 121 |
| deletion_only | 74 | 37 | 3 | 1 |
| mixed | 1377 | 696 | 165 | 91 |

## 3. 纯新增修复恢复

| 分母/结果 | Train | Valid |
|---|---|---|
| 原Diff漏洞侧为空、修复侧非空的配对 | 1232 | 163 |
| 两侧可用 | 706 | 88 |
| 漏洞侧无种子、修复侧有有效种子 | 694 | 87 |
| 迁移后漏洞侧非空候选区域 | 657 | 84 |
| 占全部纯新增配对 | 657/1232 (53.33%) | 84/163 (51.53%) |
| 占两侧可用且有修复侧种子的配对 | 657/694 (94.67%) | 84/87 (96.55%) |

这里的“恢复”只表示找到了可追踪的旧版本候选语句，**不保证包含真实漏洞操作**。原Diff漏洞侧没有正标签，不能用该侧Diff F1证明恢复准确率；37/3个具备种子条件的配对仍未恢复旧侧区域。预处理边界、宏状态操作、缺少依赖或无法唯一对应等情况不做邻近行兜底。

## 4. 区域规模和不确定性

| 统计 | Train | Valid |
|---|---|---|
| 非空区域/全部函数 | 4742/8708 | 600/1124 |
| 非空区域物理行数中位数 | 14 | 13 |
| 全部函数区域行数中位数（含零） | 4 | 3 |
| 最大区域物理行数 | 808 | 101 |
| 选中语句总数 | 62263 | 7281 |
| 扩展达到上限的函数 | 2626 | 298 |
| 路径涉及已标记Unknown的语句 | 28347/62263 (45.53%) | 3319/7281 (45.58%) |
| 种子在另一版本存在相同Token语句 | 3094 | 248 |

区域行数是原子语句的物理跨度，可能包含多行声明；12条额外语句不等于12行，也不限制直接修改语句。相同Token的种子可能来自移动、格式或注释变化，不自动代表新增语义。边上的Unknown只是已识别的局限；未标Unknown的may-dependence也不等于修复相关或安全性已证实。完整边、两侧位置及路径见本地`pairs.train.jsonl`/`pairs.valid.jsonl`。

## 5. 复用旧源码判断：聚焦作用与代价

| 原固定关系锚点池 | 总池相关/无关/未知 | 旧静态保留 | 新切片保留 | LOSVER L1保留 |
|---|---|---|---|---|
| Train 56条 | 9 / 33 / 14 | 9 / 33 / 14 | 4 / 8 / 3 | 4 / 3 / 2 |
| Valid 24条 | 6 / 14 / 4 | 6 / 14 / 4 | 5 / 3 / 1 | 3 / 2 / 1 |
| Train仅图/AST可用的43条 | 5 / 26 / 12 | 5 / 26 / 12 | 4 / 8 / 3 | 1 / 2 / 2 |
| Valid仅图/AST可用的14条 | 5 / 7 / 2 | 5 / 7 / 2 | 5 / 3 / 1 | 3 / 1 / 1 |

每格依次为修复相关/无关/Unknown的操作首行锚点数量，不是全区域precision。Valid能保留大部分已知相关锚点，同时减少无关锚点；但必须单列解析不可用造成的过滤。原静态池由上一轮规则定义，选择偏差不能外推到所有新增行。

| 旧Train扩展行复核池（86行） | 相关 | 无关 | Unknown |
|---|---|---|---|
| 原池总数 | 55 | 21 | 10 |
| 旧源码静态候选选中 | 23 | 3 | 0 |
| 新切片选中 | 31 | 7 | 3 |
| 已有LOSVER L1选中 | 11 | 1 | 0 |

此池是旧审计产生的额外行，不能当作随机全库金标。新切片覆盖更多相关行，也多选了无关行，体现召回与误扩展的代价。Train LOSVER为训练内预测，不能据此判断泛化优劣。

## 6. 新的固定逐行复核

复用旧32对分层随机组，每种编辑类型固定Seed42抽2对：Train8对35行、Valid8对30行。对各方法随机抽一行后取联合行池；模型和规则不据这些Valid结果调整。各方法覆盖的行可能重叠，表中比例不是无偏总体精度。Related表示与可见修复相关（含明确注明的弱关联），不是确认CVE根因/危险行。

| Split/方法 | 全选中：相关/无关/未知 | 仅Diff外：相关/无关/未知 |
|---|---|---|
| train/original_diff | 14 / 0 / 3 | 0 / 0 / 0 |
| train/static_candidates | 3 / 6 / 3 | 1 / 6 / 3 |
| train/bidirectional_slice | 8 / 1 / 7 | 1 / 1 / 5 |
| train/losver | 17 / 2 / 4 | 3 / 2 / 1 |
| valid/original_diff | 12 / 2 / 1 | 0 / 0 / 0 |
| valid/static_candidates | 2 / 3 / 1 | 1 / 2 / 0 |
| valid/bidirectional_slice | 1 / 1 / 1 | 0 / 1 / 1 |
| valid/losver | 1 / 10 / 4 | 0 / 9 / 3 |

Valid固定8对中仅1对有两侧可用图；切片只覆盖3条抽查行，不能用这组小分母推算全库质量。为检查纯新增输出本身，再复核旧八对Valid纯新增组中**全部4对有恢复输出者**：31、54、203、298。不挑选其中的好案例，穷尽其漏洞侧47条候选行；这是可用性条件下的复核，不与主样本合并。

| Valid纯新增pair_key末号 | 候选行 | 修复相关 | 无关 | Unknown |
|---|---|---|---|---|
| 203 | 12 | 2 | 9 | 1 |
| 298 | 12 | 6 | 5 | 1 |
| 31 | 8 | 7 | 0 | 1 |
| 54 | 15 | 11 | 2 | 2 |
| 合计 | 47 | 26 | 16 | 5 |

这47行中无关行16/47=34.04%，Unknown为5/47=10.64%。26条相关行也包含定义和控制上下文，不能称为26条漏洞真值行。四对均能恢复有用关系，但不同样本差异明显；没有足够依据把这种条件复核结果外推为84对或全库准确率。

## 7. 典型真实收益与失败

**Train pair641，sample1283/1284：新增溢出检查恢复旧操作。** 旧侧Diff为空，能找到旧第7行长度乘法和第14行分配调用。路径是新检查→原生CDG→乘法→REACHING_DEF→分配→相同语句版本映射；调用/内存语义仍按Unknown记录。候选同时包含后续状态写入，不能把整个区域认定为漏洞位置。
```c
/* patched: 新增检查 */
if (params->buffer.fragment_size == 0 ||
    params->buffer.fragments > SIZE_MAX / params->buffer.fragment_size)
    return -EINVAL;
/* 两版本原有操作 */
buffer_size = params->buffer.fragment_size * params->buffer.fragments;
buffer = kmalloc(buffer_size, GFP_KERNEL);
```

**Valid pair203：找到endpoint[0]，也带入9条无关行。** 新增`alt->desc.bNumEndpoints < 1`检查后，恢复旧第17行`alt->endpoint[0].desc.wMaxPacketSize`及packet_size使用。但CDG同时覆盖固定UV寄存器配置和图像几何计算。旧第14行短`return;`被对齐到不同guard的返回位置，显示相同Token/编辑行对应不是语义等价证明；该行保留Unknown。

**Valid pair298：真实宽高定义与不同字段混在一起。** 新增`outputWidth > AVIF_MAX_IMAGE_SIZE / outputHeight`检查，恢复outputWidth/outputHeight赋值及读入；也经同一grid对象的粗粒度关系选中了rows/columns读取和递增。它们是不同字段，不能因对象相同就都标为修复相关。

**Valid pair54：生命周期上下文部分恢复，仍有关键漏项。** 找到`tmp_out_buf = mp_buf`、decoder缓冲区替换、后续`flb_free(mp_buf)`等；但旧第85行第一次`flb_free(mp_buf)`未进入候选。新增NULL状态的标量定义—使用并不完整表达之前的释放事件；调用摘要、所有权、别名和有界遍历仍有限制。人工复核为11相关、2无关、2未知，不是完整生命周期安全证明。

**Train pair1595 / Valid pair107：预处理与宏不能冒充普通运行时检查。** `#if !defined(OS_CHROMEOS)`改变编译平台范围，当前原子运行时语句种子无法表达；OP_REQUIRES等需要真实宏/调用契约。没有按名称或邻近行补成“可靠依赖”。

## 8. 与LOSVER的原Diff弱标签对照

使用相同sample_key、原始可见行集合和>.5阈值重算已有预测，不重新推理。512分母38745行/2698正行，L1为137992行/5401正行。静态审计使用全函数，再投影到相同可见行；不声称拥有相同输入信息。

| Valid范围 | 方法 | Diff Precision | Diff Recall | Diff F1 |
|---|---|---|---|---|
| 512 | static_candidates | 0.1337 | 0.2524 | 0.1748 |
| 512 | bidirectional_slice | 0.2512 | 0.4711 | 0.3277 |
| 512 | losver | 0.2496 | 0.3202 | 0.2805 |
| l1 | static_candidates | 0.0855 | 0.2300 | 0.1246 |
| l1 | bidirectional_slice | 0.2585 | 0.4081 | 0.3165 |
| l1 | losver | 0.1607 | 0.3464 | 0.2196 |

新切片读取真实Diff，LOSVER正常预测只读取单函数；因此这里不是公平的定位器推理性能赛。较高Diff F1主要说明与编辑标签更接近，**不能证明真实漏洞定位更准**。原Diff与自身比较为1，未把这个恒等结果当作性能提升。原始LOSVER Valid P/R/F1均复现既有best-f1记录，未改评价队列。

## 9. 处理成本与具体限制

| 阶段 | Train | Valid |
|---|---|---|
| 成功Joern解析/默认overlay累计 | 2136.92s | 168.11s |
| 成功原生导出累计 | 546.88s | 48.88s |
| 切片循环 | 590.22s | 29.11s |
| 最大子进程RSS | 2712.27 MiB | 1451.88 MiB |
| 超过4000 definitions的方法级警告 | 62 | 4 |

JVM固定8GB/4核、单Joern任务，批次最多256函数且目标源码不超过1MB；单个超大配对仍完整处理。以上不是端到端总时间：不含源码落盘、缓存加载、开发期被中断的导出及人工复核；子进程RSS也不含Python持有原生图缓存。前期共享盘导出首批耗时76.85s，后续临时CSV改到本机/tmp；输入不同，没有据此宣称严格加速比。成功缓存复用，没有重解析未改变的已完成批次。日志中的一次KeyboardInterrupt是切换临时导出位置的人工停止，最终Train50批/Valid5批全部完成。

Train sample6573/6574为23939/24047行的生成式scan函数。该批14函数耗时577.16s；线程栈位于`ReachingDefFlowGraph`初始化的`List.contains/filterNot`。两侧最终均触发4000-definition上限，原生DDG为0，CFG/CDG仍存在。没有把零DDG解释为无数据依赖，也没有跳过原函数。其它零边情况只能记为零或未知，不能都归因于此上限。

已观察的限制：
1. **语义不够聚焦：** 原生CDG会连接早退检查后的许多操作；对象/调用may-dependence和语句提升又会合并不同字段与参数。与实际修复无关的路径确实存在，来源不能全部归咎于Joern本身；本实现也没有追踪完整变量/别名/路径状态。[官方语义说明](https://docs.joern.io/dataflow-semantics/)明确描述未知外部调用可能产生额外路径，不能把这些边当作精确副作用证明。
2. **对应关系不等于同一操作：** 短return、重复语句、移动和格式变化需要上下文才能裁定；仅文本相同存在已观察歧义。
3. **覆盖有限：** 单函数缺少类型、宏、调用实现、资源状态与跨过程信息；两侧同时可用约54%。Unknown成员及边没有强行赋正负标签。
4. **范围与成本：** 约一半可解析函数达到额外语句上限；这会遗漏关系，放大范围又可能扩张无关代码。本轮没有按Valid改变跳数/预算。超大生成代码的DFG初始化成本很高。
5. **监督证据边界：** 原Diff含格式/控制结构/非漏洞修改，历史已知错配仍存在；源码判断是本次助手复核，不是独立双人漏洞位置标注。

## 10. 交付、复现与检查

改动仅在`syntax.py`、`losver_patch_audit.py`、同一测试文件、README和结果忽略规则；没有新增模型、训练入口或并行训练框架。已有动态困难增强与LOSVER测试保持通过。
```bash
# 依赖现有Original配对标注、两轮旧审计及静态模式缓存；缺少后者先按README上一节重建。
python -m vulnmechanism.losver_patch_audit slice-graphs --split train --batch-size 256 --output-dir results/primevul_bidirectional_slicing
python -m vulnmechanism.losver_patch_audit slices --split train --output-dir results/primevul_bidirectional_slicing
python -m vulnmechanism.losver_patch_audit slice-report --split train --output-dir results/primevul_bidirectional_slicing
# 已保存train.review.jsonl及固定成员；重新人工判断时需保持证据范围。
python -m vulnmechanism.losver_patch_audit slice-graphs --split valid --batch-size 256 --output-dir results/primevul_bidirectional_slicing
python -m vulnmechanism.losver_patch_audit slices --split valid --output-dir results/primevul_bidirectional_slicing
python -m vulnmechanism.losver_patch_audit slice-report --split valid --output-dir results/primevul_bidirectional_slicing
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_losver_patch_audit tests.test_surface_vat tests.test_losver -q
```
逐split统计由正式入口重建；合并CSV是这些文件的整理，报告解释与补充纯新增复核来自已保存的源码判断。条件纯新增组取旧八对Valid纯新增组中全部有恢复输出者，并穷尽其漏洞侧region_lines；47条成员/判断均保存，不能当作新的总体随机评测。

65项相关测试通过；另逐一检查69544条选中语句的源码byte/token/物理行、77085个路径步骤的方向/端点连续性和跨版本来源、所有成员/标签及每侧12条额外语句上限。Train35行、Valid30行固定复核ID完整；补充47行逐行保存。既有512/L1定位器指标与历史记录一致。没有官方Test评价。

关键文件：`rules.json`、`environment.json`、`coverage.csv`、`repair_shapes.csv`、`source_review.csv`、`old_anchor_retention.csv`、`costs.csv`、`train/valid.summary.json`、`train/valid.functions.csv`、`train/valid.pairs.csv`、`train/valid.diff_proxy.csv`、`*.review*.jsonl`、`*.cases.md`、`valid.addition.review_summary.csv`、`verification.log`、`unit-tests.log`。完整两侧区域/路径在本地`pairs.*.jsonl`，原生节点/边和CPG在本地`graphs/`；大缓存不纳入Git。

可行性判断：适合保留为带来源和Unknown的离线人工审计候选；已证明能补回部分纯新增修复的旧操作，但没有证明能稳定转成更准确的强区域监督。当前结果不支持直接修改检测模型或启动训练。
