# PrimeVul Original 静态修复模式审计

完成日期：2026-10-30。复用既有 `losver_patch_audit.py`、`syntax.py`、两轮审计、原始配对与已保存的512/L1定位概率。**未训练或修改研究模型，未重新执行checkpoint推理，未读取Test作开发或评价。**

## 主要结果

- 全部 **4354对Train / 562对Valid** 均处理，未清理、重划分或删除任何成员。可严格解析的双侧配对为 **3248/440**；其余保留unknown。
- 单函数至少有一种候选关系的配对为 **2913/4354（66.90%）**、**384/562（68.33%）**。这不是漏洞识别覆盖率。
- 最终固定抽样中，Train 56条候选有48条可确认局部关系、8条unknown；其中仅9条与当前修复可见相关。Valid 24条候选为19条关系确认、1条类型归类错误、4条unknown；6条与当前修复可见相关。另有Train8/Valid4条无候选或解析失败样本，不混入候选精度分母。
- Valid暴露C++流插入被当作整数移位的反例，保留错误、未回调规则。资源生命周期规则Valid零覆盖，不能解释为数据不存在资源操作。
- 静态候选与原Diff的重叠低于现有定位器；但Diff不是漏洞位置真值，不能把该差距或源码复核比例称为真实漏洞定位准确率。**当前规则不适合直接替代LOSVER，也不适合直接生成高可信漏洞行标签。**

## 1. 协议与证据边界

原始文件为 `/home/PublicData/PHY-data/vul_detect/data/PrimeVul_original/primevul_{train,valid}_paired.jsonl`，与现有 `data/primevul_paired/{train,valid}.lines.jsonl` 的全部源码和0/1标签逐条核对。按sample_key/pair_key对齐，保持1/0相邻配对。没有重新采样训练成员，也没有使用CWE作规则真值。
复用历史 `results/losver_patch_audit/pairs.csv` 的形状与配对质量，未重复提交溯源、旧区域扩展或Oracle实验。Train29对、Valid6对历史已核实错配仍保留；其频次单列，不能视为可信修复谱系。其余大量未裁定配对也不等于已确认合法。
先在Train试跑、全量挖掘和源码复核，再固定逻辑运行Valid。Train复核修正了多维数组仅记录首维、跳转可能绕过定义及集合遍历顺序问题。最终坐标检查发现Train同一对含换页符（sample_key train:15/16），按通用UTF-8字节→项目splitlines物理行映射修复；候选身份、规则和Valid提取结果均未变化。没有用Valid反例修改规则。
`rules.json`是普通规则说明和协议记录，不是哈希/冻结管理机制。固定抽样Seed42；按规则和函数标签抽函数，再每函数抽一条关系，不按Diff或LOSVER评分抽样。Valid沿用同一提取器，仅复核和汇总。

三个证据层级必须区分：

1. **AST结构变化候选**：由两版本的原子语句/控制头非注释Token多集合，以及取消相同调用后唯一剩余同callee的参数对应产生。不是完整树编辑算法，也不能可靠表达纯执行顺序变化。
2. **单函数关系候选**：只输入源码，用作用域绑定、AST包含/分支、局部词法定义与使用提取。不是完整CFG/DDG/CDG或路径证明。推理不读取Diff、修复版本、函数标签、修改行数量或模型概率。
3. **源码复核**：本次助手读取源码/编辑上下文，判断关系是否成立、是否与可见修复有关。不是独立双人标注，也不确认CVE根因或真实漏洞位置。

## 2. 全量解析与提取成本

| Split | 函数 | 严格解析函数 | 双侧可解析对 | 有候选函数 | 有候选配对 | 已解析但未归类的配对 |
|---|---|---|---|---|---|---|
| train | 8708 | 6586 (75.63%) | 3248 | 5661 | 2913 | 517 |
| valid | 1124 | 886 (78.83%) | 440 | 747 | 384 | 87 |

“未归类”不等于代码未变化或修复无效；宏检查、callee替换、字段类型未知、纯重排等都可能未落入预设类别。解析失败不补写源码、不猜测标签。
Train有2075个函数含解析错误、300个函数边界非唯一；Valid为232/26。两类可重叠，不相加。具体error/missing节点的byte与行位置在functions.*.jsonl中。

最终单进程CPU提取与保存耗时：Train 292.37s，Valid 33.72s；逐函数中位数 13.04/12.56ms，最大 5.95/2.25s。开发期试跑/修正重算不计入该最终提取计时。最长24,047行源码也正常处理，没有函数总Token截断。未运行Joern进程。

## 3. 成对AST变化频次

以下模式非互斥，分母始终为全部4354/562对；CSV另外给出可解析配对中的比例。**它们不是经逐例核实的漏洞根因分布**。尤其字段访问变化不必是内存漏洞，整数候选不必是溢出，释放调用出现不必是不安全释放。

| 候选模式 | Train | Valid | Train其中已知错配 |
|---|---|---|---|
| 索引/长度—条件关系变化 | 254 (5.83%) | 44 (7.83%) | 5 |
| 整数候选—条件关系变化 | 188 (4.32%) | 22 (3.91%) | 2 |
| 局部整数算术变化 | 256 (5.88%) | 32 (5.69%) | 4 |
| 空性条件—访问关系变化 | 194 (4.46%) | 24 (4.27%) | 3 |
| 下标/解引用/内存调用变化 | 1431 (32.87%) | 185 (32.92%) | 11 |
| 分配—释放/释放后访问关系变化 | 15 (0.34%) | 0 (0.00%) | 0 |
| 释放调用变化（仅结构） | 61 (1.40%) | 6 (1.07%) | 1 |
| 唯一可对应调用的参数变化 | 487 (11.19%) | 66 (11.74%) | 10 |
| 控制头/return/goto/break/continue变化 | 2189 (50.28%) | 285 (50.71%) | 17 |

完整计数见 `ast_repair_changes.csv`、`pairs.train.jsonl`、`pairs.valid.jsonl`。后两者在本地保留，用于逐对复核；不把分类未知的配对当成负例。

## 4. 单函数规则、覆盖与成对变化

正式函数为 `syntax.static_relations(source)`。直接复用现有parse_source、function_bindings、resolve_local_binding和原子AST单元。保留原始UTF-8 byte offset及0基物理行号；源码案例显示为1基。没有邻接扩展、同名变量无约束扩展或整个if块标记。
- 条件关系要求相同局部/参数绑定或精确字段路径，并位于实际分支/循环中，或同块先前的无条件return/throw退出条件后。中间重写、地址逃逸、未建模跳转/字段可能经调用改变时拒绝补关系。
- 容量规则要求目标数组声明或直接分配定义与memcpy/memmove/memset/strncpy的目标、显式长度对应。保存全部数组维度，区分元素数和字节数；不把仅出现API名称当作命中。符号容量/长度无法比较时为unknown。
- 调用参数规则关联计算定义和同一绑定的实参；资源规则要求对象定义/释放/访问关联。它们仍是词法候选，不承诺别名、所有权、类型宽度或跨过程语义。
- 所有`safety`均为unknown。无候选不表示安全；有检查不表示检查充分。整数候选的局部声明判断不是编译器表达式类型推断，Valid已经证实其不足。

| 规则 | Train漏洞侧 | Train修复侧 | Valid漏洞侧 | Valid修复侧 |
|---|---|---|---|---|
| 索引/长度—条件 | 1025/4354 (23.54%) | 1044/4354 (23.98%) | 139/562 (24.73%) | 143/562 (25.44%) |
| 空性条件—访问 | 811/4354 (18.63%) | 848/4354 (19.48%) | 98/562 (17.44%) | 97/562 (17.26%) |
| 整数候选—条件 | 627/4354 (14.40%) | 656/4354 (15.07%) | 90/562 (16.01%) | 91/562 (16.19%) |
| 目标容量—长度 | 105/4354 (2.41%) | 99/4354 (2.27%) | 11/562 (1.96%) | 11/562 (1.96%) |
| 资源生命周期候选 | 49/4354 (1.13%) | 46/4354 (1.06%) | 0/562 (0.00%) | 0/562 (0.00%) |
| 计算定义—调用参数 | 2129/4354 (48.90%) | 2133/4354 (48.99%) | 271/562 (48.22%) | 276/562 (49.11%) |
| 条件—调用 | 2316/4354 (53.19%) | 2352/4354 (54.02%) | 297/562 (52.85%) | 307/562 (54.63%) |

仅统计两侧都可解析的440对Valid，其候选数量变化如下。增加不等于修复有效，减少也不等于退化。

| 规则 | 修复侧更多 | 修复侧更少 | 同数量且非空 | 两侧皆无 |
|---|---|---|---|---|
| 索引/长度—条件 | 24 | 7 | 114 | 295 |
| 空性条件—访问 | 14 | 8 | 79 | 339 |
| 整数候选—条件 | 10 | 6 | 78 | 346 |
| 目标容量—长度 | 1 | 1 | 10 | 428 |
| 资源生命周期候选 | 0 | 0 | 0 | 440 |
| 计算定义—调用参数 | 60 | 21 | 195 | 164 |
| 条件—调用 | 87 | 24 | 199 | 130 |

逐对记录在 `pair_pattern_changes.csv`，抽取阶段触发/匹配次数在 `extraction_stages.csv`。正负两侧覆盖非常接近，单纯“有这种关系”并不能提供可靠漏洞判别依据。

## 5. 固定源码复核

最终Train复核64条：56条候选和8条无候选/解析未知。Valid复核28条：24条候选和4条无候选/解析未知。资源规则在Valid无输出，所以没有该组复核候选。另在Train开发期间查看过的案例只用于修正实现，不并入最终固定样本的质量分母。
**关系确认**只表示当前源码支持该局部关系；**修复相关**表示候选操作或证据与当前可见编辑有关，含明确记录的弱相关，不代表漏洞根因确认。无法确认的别名/复杂路径/谱系保留unknown。分层样本不能外推总体precision。

| Split/规则 | 候选数 | 关系确认 | 关系错误 | Unknown | 修复相关 | 修复无关 | 修复未知 |
|---|---|---|---|---|---|---|---|
| train/索引/长度—条件 | 8 | 8 | 0 | 0 | 1 | 4 | 3 |
| train/空性条件—访问 | 8 | 8 | 0 | 0 | 1 | 7 | 0 |
| train/整数候选—条件 | 8 | 8 | 0 | 0 | 3 | 4 | 1 |
| train/目标容量—长度 | 8 | 6 | 0 | 2 | 0 | 6 | 2 |
| train/资源生命周期候选 | 8 | 5 | 0 | 3 | 2 | 3 | 3 |
| train/计算定义—调用参数 | 8 | 6 | 0 | 2 | 0 | 6 | 2 |
| train/条件—调用 | 8 | 7 | 0 | 1 | 2 | 3 | 3 |
| valid/索引/长度—条件 | 4 | 4 | 0 | 0 | 1 | 3 | 0 |
| valid/空性条件—访问 | 4 | 3 | 0 | 1 | 0 | 3 | 1 |
| valid/整数候选—条件 | 4 | 3 | 1 | 0 | 1 | 3 | 0 |
| valid/目标容量—长度 | 4 | 3 | 0 | 1 | 2 | 1 | 1 |
| valid/资源生命周期候选 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| valid/计算定义—调用参数 | 4 | 3 | 0 | 1 | 1 | 2 | 1 |
| valid/条件—调用 | 4 | 3 | 0 | 1 | 1 | 2 | 1 |

Train候选关系确认48/56，unknown8/56；修复相关9/56、无关33/56、未知14/56。Valid为关系确认19/24、错误1/24、未知4/24；修复相关6/24、无关14/24、未知4/24。这些分母保留unknown，不把可裁定子集的比例包装成全库准确率。
`train.review.jsonl`、`valid.review.jsonl`逐例保存判断与理由；`*.review_sample.jsonl`保存固定抽样成员和原候选；`train.cases.md`、`valid.cases.md`含代码/行号及配对上下文。

## 6. 与现有LOSVER定位器比较

只重算已有预测的统计，不运行模型。512沿用原始可见行，L1沿用全覆盖可见行，全部成员及分母不变。静态full_source_projected使用全函数再投影到相同可见行；512的prefix_evidence_only只采用操作及证据均落在完整可见前缀的关系，保守排除可能不完整的末行。它仍用完整函数解析结构，不能声称与编码器信息完全相等。阈值固定使用既有概率>.5；静态规则不按GT行数或模型分数选Top-k。
主表为Valid，完整Train和两侧分组在CSV中。Train定位器是训练内预测，不能用其高F1当成泛化证据。

| 输入 | 方法 | 有候选函数 | 行数 | Diff P | Diff R | Diff F1 | 选中行比例 |
|---|---|---|---|---|---|---|---|
| 512 | static_full_source_projected | 700/1124 | 38745 | 0.1337 | 0.2524 | 0.1748 | 13.15% |
| 512 | losver | 758/1124 | 38745 | 0.2496 | 0.3202 | 0.2805 | 8.94% |
| 512 | static_prefix_evidence_only | 672/1124 | 38745 | 0.1304 | 0.2257 | 0.1653 | 12.05% |
| l1 | static_full_source_projected | 747/1124 | 137992 | 0.0855 | 0.2300 | 0.1246 | 10.53% |
| l1 | losver | 1020/1124 | 137992 | 0.1607 | 0.3464 | 0.2196 | 8.44% |

**这些是原Diff弱标签代理，不是漏洞位置precision/recall/F1。** 静态候选更少命中Diff，不能自动解释为真实漏洞区域更差；纯新增修复的漏洞侧没有Diff正标签，而它可以存在真实相关的旧操作。也不能因静态关系在源码上成立就宣称优于定位器。

## 7. 可复核案例与反例

以下短编号明确指sample_key；配对另列，避免与历史报告的pair_key短编号混淆。

| sample_key / pair_key | 源码事实 | 判断 |
|---|---|---|
| train:1283/1284 / train:641 | 字段乘积buffer_size用于kmalloc；修复新增fragment_size==0或fragments>SIZE_MAX/fragment_size早退 | 真实可见溢出检查，却因字段类型/分配API未建模而没有integer_range_relation；不能用188当整数漏洞总数 |
| train:6758 / train:3378 | ColorMap[3][MAXCOLORMAPSIZE]与memset(...,3*MAXCOLORMAPSIZE) | Train开发发现首维记录不全，已通用修复并测试；不推断宏值，也不称该清零为漏洞 |
| valid:329 / valid:164 | output_height=input_height/block_size；新增TF_LITE_ENSURE(context,block_size>0) | 旧侧Diff为空，计算定义—参数候选仍含除法。是修复相关性而非Diff命中为真的例子；宏语义未被AST变化分类恢复 |
| valid:851 / valid:425 | suffix[Name::SIZE]及清零大小改为Name::SIZE+1 | 容量/长度模式与本次编辑直接相关，仍不将其当完整字符串安全证明 |
| valid:200 / valid:99 | if(!label)报错后，define_label改为else路径才调用 | 精确控制关系能表达修复后限制；API契约仍需上下文 |
| valid:263 / valid:131 | LOG(INFO) << "ndims=" << ndims | 已确认类型归类错误：C++流插入不是整数移位。Valid后未修规则，计入错误 |
| valid:929/930 / valid:464 | last_char读取从getNext()->write之后移到之前 | AST多集合未归类此纯重排；缺少有序CFG/调用生命周期，不能说没有修复 |
| valid:517 / valid:258 | optbuf/source/type分配后多处goto err集中清理 | 当前规则拒绝跨跳转的词法定义关联，不强行认定可证明生命周期 |

典型旧侧源码（valid:329，原Diff为空）：
```cpp
int output_height = input_height / block_size;
TF_LITE_ENSURE_EQ(context, input_height, output_height * block_size);
```
修复仅在前方新增：
```cpp
TF_LITE_ENSURE(context, block_size > 0);
```
定义—参数候选可以覆盖除法，但并未证明该函数其他路径或所有输入都安全。不能由这个成功案例推广全库质量。

## 8. 失败原因与资源零覆盖核查

| 原因 | Train涉及函数/事件 | Valid涉及函数/事件 |
|---|---|---|
| parse_unknown | 2122 / 2122 | 238 / 238 |
| operation_without_resolved_operand | 4815 / 73080 | 621 / 8454 |
| definition_path_or_reassignment_unknown | 3641 / 52426 | 506 / 4983 |
| definition_unstructured_control_unknown | 1886 / 28581 | 212 / 2753 |
| arithmetic_type_or_binding_unknown | 3799 / 38302 | 490 / 4886 |
| guard_value_written_before_operation | 1549 / 17828 | 184 / 2108 |
| intervening_call_field_state_unknown | 1280 / 5739 | 154 / 527 |

事件可重复、原因可重叠，不能相加当作失败样本数量。
- 解析：PHP_FUNCTION、__user等宏/类型注解，以及OS条件编译中拆开的else、container_of的类型实参等可能无法被无预处理上下文的单函数语法恢复。保留实际error/missing位置，不声称全部错误都由宏导致。
- 字段/类型：Train:641的乘法依赖结构字段，缺少完整类型上下文；Valid:263的流操作则显示仅看局部整数叶子仍可误判表达式类型。当前“整数”规则既会遗漏真相关操作，也会错误归类。
- 路径/别名：源码中出现分配和释放，不能证明它们针对同一对象、同一版本、可达同一路径或拥有所有权。goto集中清理、realloc返回临时指针、free(*out)等均超出当前受限规则。
- 语义相关性：普通遍历、调试输出和正常清理也能产生大量正确关系。精确绑定可以避免部分同名误连，但不能据此找到本次漏洞的关键约束。

资源专项只读核查：对含标准分配/释放字样的96个Valid函数进一步检查AST，其中73个可严格解析，**24个同时有malloc/calloc/realloc与free调用**，仍无资源关系输出。见 `resource_presence.valid.jsonl`。这不是“Valid没有资源操作”，而是规则的对象与路径覆盖不足。
- valid:55/56实际是`*out = calloc(sz,1)`、`free(*out)`，间接目标不在直接局部变量/字段定义规则内。
- valid:119/120使用`bigger_escaped = realloc(escaped,...)`、失败时free(escaped)、成功时escaped=bigger_escaped，需要重分配与别名状态。
- valid:517/518有多处goto err和集中free，当前拒绝跨未建模跳转。
没有为提高覆盖追加API别名或忽略路径约束。此资源规则在Valid不可用，应如实记为负结果。

## 9. 可行性结论

1. **可行的部分：** 在单函数、无Diff输入下，轻量Tree-sitter规则能提供带位置和依据的局部程序关系，且大类覆盖在Valid上延续。
2. **未成立的部分：** “语法关系存在”并没有变成“当前修复相关区域可靠可定位”。固定源码复核中修复无关候选占比较高；整数类别有明确错误，资源类别无Valid覆盖，纯重排与宏检查存在漏项。
3. **当前不建议：** 把联合候选当作强漏洞行标签、直接替代LOSVER或据Diff F1推导分类收益。若以后需要准确类型、完整可达定义或控制依赖，需另行验证类型/CFG/别名能力；本轮没有实现或运行Joern CFG/DDG/CDG，不把AST/词法候选冒充这些分析。没有新增DL模块、训练或Test实验。

## 10. 复现与检查

```bash
python -m vulnmechanism.losver_patch_audit patterns --split train --output-dir results/primevul_static_patterns
# 按train.cases.md完成train.review.jsonl；固定规则后再运行Valid。
python -m vulnmechanism.losver_patch_audit patterns --split valid --output-dir results/primevul_static_patterns
python -m vulnmechanism.losver_patch_audit pattern-report --output-dir results/primevul_static_patterns
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_losver_patch_audit tests.test_surface_vat tests.test_losver -q
```

汇总命令重建CSV和summary.json；本报告的案例解释来自已保存源码复核。抽取缓存functions.*.jsonl和pairs.*.jsonl保留本地，可从原始数据重建。单函数部署接口只调用 `static_relations(source)`。
实际检查：**54项相关测试通过**；核对136,222条关系的byte范围、源码片段、物理行及成员/标签；两组复核ID/分母完整；复现两个PYTHONHASHSEED下的输出确定性；已有512/L1 Valid定位F1与原best-f1记录一致。坐标检查在Train发现的两个换页符函数已通用修正并复验；Valid没有这类分隔符。原始数据、模型文件、既有审计和预测均未覆盖。

关键文件：`rules.json`、`summary.json`、`ast_repair_changes.csv`、`pattern_frequency.csv`、`pair_pattern_change_summary.csv`、`diff_proxy_comparison.csv`、`source_review_summary.csv`、`extraction_stages.csv`、`failures.csv`、`*.review.jsonl`、`*.cases.md`、`unit-tests.log`。
