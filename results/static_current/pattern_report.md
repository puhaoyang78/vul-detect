# BigVul / Devign / ReVeal 静态模式覆盖率核查

日期：2026-10-21。每库沿用原 1024 Train / 256 Valid（每个 Split 正负各半），共 3840 个函数。没有读取 Test、重新采样、修改原始上游源码、训练模型或重新生成分类预测。

## 范围与判定口径

修改前来自 ff670b39cae4d4ac686b572cc591ce0fe5e96a4f 对应的原 schema-4 缓存；首次升级前实际记录全部候选及成员。最终统计使用正式提取器重新运行，再按 sample_key 对齐这次记录的原始统计，不把中间调试版本当作原基线。
源码/标签/划分和原生 CPG 边逐函数核对；缓存只更新候选模式与诊断。已有 12 组模型、配置、日志、预测和分类汇总不变，旧模型结果不代表使用修复后关系的表现。

模式边只是程序关系。已确定的常量或符号容量比较、语法属性和依赖上下文，与启发式 CFG 候选分开记录；无法确定对象、长度、别名或路径时为 unknown。未匹配不等于无漏洞，函数标签也不是某个局部模式的位置真值。

## 调查确认的根因与修复

- BigVul：旧长度计算只覆盖简单字节数组和字面量，不能跟随别名、使用点赋值、带偏移的目的地址或类型符号大小；C++ new/delete 没有作为 AST 事件提取。修复绑定/作用域、局部 Def-Use、sizeof/strlen 来源和字节偏移。真实样本中的可证明操作多数恰好落在容量内，不能为提高覆盖率改判溢出。
- Devign：大量对象是函数形参、结构体字段或库返回值，缺少调用方/类型定义，容量确实 unknown。free+continue、return 及下一次循环赋值使旧源码顺序 UAF/DF 候选不成立。新增 CFG 同对象和重定义截断，保留未知路径；`&array` 按整个数组处理，不能错误地当成一个字节。
- ReVeal：现有 C++ Tree-sitter 使用 `indices/subscript_argument_list`，旧规则仅按 C 的 `index` 取值，漏掉实际负常量下标。修复两种 AST，同时把 signed 变量的未知范围移出负下标模式。`histp` 改指向后，同名 `histp->key_data` 不能当成同一对象；未展开 `CHECK_BETTER` 宏会使旧常量传播失效，现记录 intervening_parse_error_or_macro。
- 共性：数组形参不提供调用方容量；sizeof(pointer) 不等于指向对象容量；`strncpy` 遇 NUL 后填充而非继续读源，`strcat/strncat` 包括已有目的长度和 NUL；calloc、realloc 参数角色、字面量转义和嵌入 NUL 按实际语义处理。未知 ABI 大小保留为符号，不假设 int/结构体固定字节数。
- 原规则把字符串中的比较符当 guard、把成员方法名当 C 库调用，也可能误认同名变量。现在使用 AST 比较、调用角色和声明绑定；guarded_operation 仍仅是控制关系，检查充分性为 unknown。
- Joern 解析失败时仍保留有依据的语法属性，解析状态不会伪装成 ok；无 CFG 时不产生 UAF/DF 路径候选。恢复 AST 中零宽 new 操作数不作为有效关系端点。

## 逐类覆盖率

单位为包含该类至少一个候选的真实函数数；每列分母都是 1280。实例数、比例和逐标签/Split 结果见 pattern_coverage.csv / pattern_audit.json。

| 模式 | BigVul 原→新 | Devign 原→新 | ReVeal 原→新 |
|---|---:|---:|---:|
| memory_access | 316 → 316 | 635 → 635 | 557 → 557 |
| pointer_access | 926 → 926 | 1076 → 1076 | 955 → 955 |
| allocation | 30 → 87 | 6 → 6 | 39 → 53 |
| release | 55 → 67 | 8 → 7 | 56 → 57 |
| guard | 737 → 666 | 923 → 912 | 790 → 819 |
| sensitive_read | 10 → 5 | 11 → 10 | 6 → 6 |
| sensitive_write | 126 → 119 | 218 → 217 | 174 → 174 |
| buffer_size_mismatch | 0 → 0 | 0 → 0 | 0 → 0 |
| source_size_copy | 7 → 4 | 15 → 4 | 14 → 6 |
| destination_size_copy | 6 → 2 | 23 → 2 | 19 → 8 |
| off_by_one | 0 → 0 | 0 → 0 | 0 → 0 |
| buffer_overread | 0 → 0 | 0 → 0 | 0 → 0 |
| double_free | 6 → 0 | 4 → 0 | 15 → 0 |
| use_after_free | 6 → 0 | 2 → 0 | 6 → 0 |
| negative_index_read | 67 → 3 | 285 → 4 | 0 → 16 |
| negative_index_write | 41 → 1 | 171 → 2 | 0 → 2 |
| guarded_operation | 20 → 36 | 24 → 84 | 22 → 68 |

三类缓冲区模式仍为零；在本小样本和局部信息范围内没有获得可证明越界的真实正例。不能据此断言这些数据集没有相关漏洞。旧 UAF/DF 候选是被过滤或转为 unknown，不能把所有删除项都宣称为已证明的误报。负下标尤其常见于解析器栈、图像邻域和重新定位的指针，不能当作漏洞标签。

## 候选阶段与过滤原因

以下单位为规则尝试的操作位点或同对象事件对，可能在一个函数中多次出现，不能与函数数相加。容量候选的分母只包括当前建模 API 角色，不是所有可能的内存访问。每次尝试恰有 matched / filtered / unknown 一种结局。

| 数据集 | 目的容量候选 | 可比较且未超过该容量上界 | unknown | 源读取候选 | 可比较且未超过该容量上界 | unknown |
|---|---:|---:|---:|---:|---:|---:|
| bigvul | 252 | 40 | 212 | 177 | 24 | 153 |
| devign | 365 | 31 | 334 | 229 | 27 | 202 |
| reveal | 334 | 34 | 300 | 236 | 8 | 228 |

**bigvul**：

- buffer_size_mismatch：filtered:not_exceeding_extent=40；unknown:buffer_expression_unknown=39；unknown:caller_extent_unknown=34；unknown:object_identity_unknown=34；unknown:callee_extent_unknown=31；unknown:field_extent_unknown=21；unknown:pointer_redefinition_unknown=17；unknown:field_base_object_unknown=9；unknown:unresolved_buffer_binding=8；unknown:loop_variant_definition=6；unknown:array_extent_unknown=5；unknown:noncomparable_symbolic_extents=5；unknown:conditional_definition=2；unknown:allocation_extent_unknown=1。
- buffer_overread：unknown:object_identity_unknown=36；unknown:caller_extent_unknown=32；filtered:not_exceeding_extent=24；unknown:buffer_expression_unknown=23；unknown:field_extent_unknown=15；unknown:callee_extent_unknown=9；unknown:unresolved_buffer_binding=9；unknown:field_base_object_unknown=8；unknown:pointer_redefinition_unknown=7；unknown:allocation_extent_unknown=2；unknown:conditional_definition=2；unknown:field_alias_write_unknown=2；unknown:loop_variant_definition=2；unknown:noncomparable_symbolic_extents=2；unknown:array_extent_unknown=1；unknown:pointer_slot_size_unknown=1；unknown:string_length_unknown=1；unknown:unsupported_expression=1。
- double_free：filtered:unconditional_return_before_use=41；filtered:mutually_exclusive_branches=12；filtered:no_redefinition_free_cfg_path=5；unknown:released_object_identity_unknown=4；unknown:use_object_identity_unknown=1。
- use_after_free：filtered:unconditional_return_before_use=16；filtered:no_redefinition_free_cfg_path=11；unknown:released_object_identity_unknown=7；unknown:use_object_identity_unknown=1。

**devign**：

- buffer_size_mismatch：unknown:buffer_expression_unknown=116；unknown:field_extent_unknown=64；filtered:not_exceeding_extent=31；unknown:object_identity_unknown=26；unknown:callee_extent_unknown=20；unknown:field_base_object_unknown=18；unknown:unresolved_buffer_binding=18；unknown:pointer_redefinition_unknown=17；unknown:caller_extent_unknown=13；unknown:loop_variant_definition=12；unknown:conditional_definition=10；unknown:array_extent_unknown=7；unknown:access_length_unknown=4；unknown:intervening_parse_error_or_macro=3；unknown:field_alias_write_unknown=2；unknown:string_length_unknown=2；unknown:allocation_extent_unknown=1；unknown:field_redefinition_unknown=1。
- buffer_overread：unknown:buffer_expression_unknown=57；filtered:not_exceeding_extent=27；unknown:unresolved_buffer_binding=20；unknown:conditional_definition=16；unknown:field_base_object_unknown=15；unknown:object_identity_unknown=15；unknown:callee_extent_unknown=14；unknown:caller_extent_unknown=14；unknown:field_extent_unknown=11；unknown:loop_variant_definition=10；unknown:pointer_redefinition_unknown=9；unknown:field_alias_write_unknown=6；unknown:unsupported_expression=5；unknown:array_extent_unknown=4；unknown:external_constant_or_variable=2；unknown:string_length_unknown=2；unknown:intervening_parse_error_or_macro=1；unknown:unstructured_control_flow=1。
- double_free：filtered:unconditional_return_before_use=3；unknown:use_object_identity_unknown=2；filtered:no_redefinition_free_cfg_path=1。
- use_after_free：filtered:no_redefinition_free_cfg_path=5；filtered:unconditional_return_before_use=3。

**reveal**：

- buffer_size_mismatch：unknown:buffer_expression_unknown=87；filtered:not_exceeding_extent=34；unknown:field_base_object_unknown=32；unknown:field_extent_unknown=28；unknown:object_identity_unknown=26；unknown:loop_variant_definition=21；unknown:array_extent_unknown=18；unknown:caller_extent_unknown=18；unknown:pointer_redefinition_unknown=12；unknown:call_expression_parse_error=10；unknown:unresolved_buffer_binding=10；unknown:callee_extent_unknown=9；unknown:conditional_definition=9；unknown:field_alias_write_unknown=6；unknown:allocation_extent_unknown=4；unknown:noncomparable_symbolic_extents=4；unknown:external_constant_or_variable=2；unknown:array_parameter_decay=1；unknown:field_redefinition_unknown=1；unknown:unstructured_control_flow=1；unknown:unsupported_expression=1。
- buffer_overread：unknown:buffer_expression_unknown=53；unknown:caller_extent_unknown=42；unknown:field_extent_unknown=26；unknown:field_base_object_unknown=18；unknown:object_identity_unknown=18；unknown:loop_variant_definition=16；unknown:unresolved_buffer_binding=11；unknown:pointer_redefinition_unknown=10；filtered:not_exceeding_extent=8；unknown:call_expression_parse_error=8；unknown:callee_extent_unknown=7；unknown:conditional_definition=6；unknown:field_alias_write_unknown=6；unknown:array_extent_unknown=3；unknown:field_redefinition_unknown=1；unknown:intervening_parse_error_or_macro=1；unknown:noncomparable_symbolic_extents=1；unknown:unsupported_expression=1。
- double_free：unknown:released_object_identity_unknown=12；filtered:unconditional_return_before_use=10；unknown:use_object_identity_unknown=9；filtered:no_redefinition_free_cfg_path=3。
- use_after_free：filtered:unconditional_return_before_use=76；unknown:use_object_identity_unknown=14；unknown:released_object_identity_unknown=3；filtered:no_redefinition_free_cfg_path=2。

unknown 的主要来源是外部/字段对象、调用方容量、条件或循环定义、符号不可比较及缺失语法；这不是 Token 对齐失败。完整每类原因保存在 pattern_audit.json，API 模型未覆盖的调用也单列，不默认安全。

## 可人工核查的真实样本

完整原文及字段来源见 tests/fixtures/static_patterns.json；pattern_cases.json 保存固定分层抽样的操作位置、AST 类型/父结点、CPG 节点与实际关联边。pattern_review.json 保存 18 个逐函数人工观察。以下均为原 Train/Valid，派生对照单列。

| sample_key / 标签 / Split | 真实代码和具体位置 | 核查结论 |
|---|---|---|
| bigvul:178941 / 1 / Train | 第12行 `memcpy(&hash[5], keyptr->secret, sizeof(__u32) * 7)`；`hash[12]` | 剩余恰好7个元素；可证明目的容量匹配，源字段容量 unknown |
| bigvul:180876 / 1 / Train | 第45/47行 `memcpy(xbuf,"perm",5)` / `"expd"`；`xbuf[16]` | 字面量含 NUL 共5字节，未越界 |
| bigvul:177924 / 1 / Train | 第13行 free(cd)，随后 return NULL；稍后 cd->nentry | 失败分支已返回，源码顺序不构成 UAF；原 Joern no_definition 保留 |
| devign:14075 / 0 / Train | `char buf_in[1024]`；`read(fd_in,&buf_in,8)`、`read(fd_in,&buf_in,1024)` | 整个数组地址，容量1024；首轮错误的1字节推导已纠正 |
| devign:18180 / 1 / Train | 循环中的 free(physmap)、continue 和重新赋值 | CPG 不支持跨对象版本的旧生命周期候选 |
| reveal:0:10800 / 0 / Train | `env -> fptags [ i ]`，`i<8` 循环 | C++ indices AST 现可识别；循环变量范围不据 signed 推成负数 |
| reveal:0:4495 / 0 / Valid | 第48/62行 histp 先指向 old_keys[j]，后指向 old_keys[knext] | 同名字段不能证明同一物理对象，旧释放配对失效 |
| reveal:1:1668 / 1 / Train | 第99行 best_site=-1；第111/122行 CHECK_BETTER；第125行数组访问 | 宏未展开且 AST 缺分号；不能沿用 -1。新结果 unknown，负下标误匹配已删除 |
| bigvul:181546 / 1 / Valid | 生成解析器的 `yyvsp[-2]` 等 | 真实负下标属性成立，不能据此认定越界或漏洞 |

人工审查拦下并修正了三处首轮候选：把 `&buf_in` 当作1字节、把重新赋值的 histp 字段当同一对象，以及穿过未知宏传播 best_site=-1。它们不计入最终有效覆盖率。
最终含任一负下标属性的函数中，BigVul 正/负标签为3/1、Devign为0/4、ReVeal为14/3。这直接说明该属性不能当成漏洞真值。选定4个函数的显式负下标属性与源码一致为4/4，另外5个已知局部容量样本，各取一个代表操作，其上界比较与源码一致为5/5；它们是正确过滤的样本。两项均不是随机抽样精度或漏洞检测精度。最终三类缓冲区及 UAF/DF 没有真实 matched 正例，其正例精度为 N/A；其余类别没有局部真值，不报告伪造的总体精度。

## 原论文与工具实现核对

- [VIPer 论文](https://arxiv.org/html/2501.13291v1)和[作者开源归档](https://zenodo.org/records/14902701)：核对容量、访问长度、DD 和释放路径定义。论文表II的 off-by-one 为目的 LENd+1，而归档 overflow_vulnerability_rules.py:73 的作者代码取 arg_src/LENs+1，存在不一致；本实现明确沿用论文的目的容量定义。source_size_copy 是长度来源候选，未额外断言 VIPer BSB 所需的目的溢出条件。
- [DeepDFA 论文](https://arxiv.org/abs/2212.08108)及[作者 Def-Use/GEN-KILL 实现](https://github.com/ISU-PAAL/DeepDFA/blob/master/DDFA/sastvd/scripts/abstract_dataflow_full.py)：借鉴使用点定义与重定义边界，不将其抽象数据流当作精确容量求解器。
- [PVDetector CSG 实现](https://github.com/jaylanli/PVDetector/blob/main/parser/CSG_c.py)：区分指针、下标、循环/API 启发式与实际边界结论，不照搬数值阈值作为漏洞证明。
- [Joern 数据流语义](https://docs.joern.io/dataflow-semantics/)和[控制流步骤](https://docs.joern.io/cpgql/control-flow-steps/)：CFG 可达与 REACHING_DEF/REF 是程序关系，外部函数默认传播可能过近似，不能证明条件路径可行或对象容量。
- [CodeQL 缓冲区示例](https://codeql.github.com/docs/codeql-language-guides/detecting-a-potential-buffer-overflow/)及[BufferAccess.qll](https://github.com/github/codeql/blob/main/cpp/ql/lib/semmle/code/cpp/security/BufferAccess.qll)：核对分配、存储长度、NUL 和类型字节/元素角色。
- [Clang CStringChecker](https://clang.llvm.org/doxygen/CStringChecker_8cpp_source.html)：核对 dynamic extent、偏移、strncpy 填充及 strcat 已有目的长度；不把未知状态判为越界。
- [SVF 抽象外部 API 实现](https://github.com/SVF-tools/SVF/blob/master/svf/lib/AE/Svfexe/AbsExtAPI.cpp)：核对 points-to/extent 和 strlen/内存函数模型，unknown/不可比较区间保持未知。

本轮实际执行的是 Joern 与 Tree-sitter；CodeQL、Clang Static Analyzer、SVF 仅作为官方方法/源码参考，没有声称运行这些工具或复现它们的完整精度。当前实现仍是函数局部、无宏展开/项目类型布局/跨过程 points-to 的最小关系提取器。容量比较是记录前提下的访问上界关系，不证明指针有效、整数运算无回绕或整个函数内存安全。

## 解析、对齐与成本

Joern 状态保持不变；Tree-sitter 的 error 表示该函数存在恢复语法，不能直接当作整个函数解析失败。

| 数据集 | Joern ok / 1280 | no_definition | multiple_definitions | Tree-sitter 含恢复错误 | 接受候选对齐缺失 | 截断候选实例 |
|---|---:|---:|---:|---:|---:|---:|
| bigvul | 1163 (90.86%) | 111 | 6 | 294 | 0 | 9239 |
| devign | 1209 (94.45%) | 68 | 3 | 273 | 0 | 14493 |
| reveal | 1082 (84.53%) | 184 | 14 | 319 | 0 | 10017 |

截断审查使用与现有 InputBuilder 相同的逻辑2048总 Token预算；只表示候选锚点可见性，不承诺所有关系端点同窗口可见。输入未改动，原图端点的原始 exact/location_only 标志保留。

| 数据集 | 新规则1280函数总秒数 / ms每函数 | 同一128 Train：旧→新中位ms | 旧→新P95 ms | 旧→新总秒数 | 缓存升级总秒数 |
|---|---:|---:|---:|---:|---:|
| bigvul | 15.223 / 11.893 | 1.479 → 2.636 | 13.696 → 25.806 | 0.787 → 1.407 | 63.296 |
| devign | 20.488 / 16.006 | 4.082 → 6.988 | 25.650 → 34.856 | 2.314 → 1.567 | 92.310 |
| reveal | 22.359 / 17.468 | 3.315 → 5.439 | 18.144 → 29.074 | 51.301 → 1.477 | 91.063 |

规则计时仅包括 AST+匹配；缓存升级还包括 CPG 读取、模式/上下文边附加及写盘，不等同于 Joern 解析时间。CPG 上下文查找使用字符位置索引，保留原来1024字符跨度/重叠条件，避免每个候选全扫描所有节点。
新规则的中位耗时增加，不能声称所有函数都更快。固定 Seed42 每库128个平衡 Train 函数交替计时；旧实现直接从所记录 Git commit 只读载入内存，未复制到正式代码。逐函数时间和成员在 pattern_timing.csv。旧 ReVeal 的 reveal:1:1666 存在明显长尾，详见该表；总体耗时差受它影响。

本轮原始3840函数的缓存命中 3840/3840，未缓存项0，Joern进程0；没有重解析未改变函数。另为真实源码的两种派生生命周期正反例单独批量解析2函数，33.650秒；相同2函数缓存复用0.058秒。首次开销主要含JVM启动，不能将此二函数比值外推为全量加速。
派生对照只补上原函数返回类型，分别保留/移除 free 后的 return，两个真实 Joern 图分别得到0/5个 UAF候选；原始源码和数据覆盖率不含这些对照。

## 验证与复现

```bash
python -m vulnmechanism.cli static audit --output-dir results/static_current
python -m vulnmechanism.cli static extract --cache-root results/static_guided/joern/cache --output-dir results/static_current
python -m unittest tests.test_static
python -m unittest discover -s tests
```

历史“修改前”必须使用该原始 commit/其原缓存；缓存升级后再次 audit 的输入基线就是升级后的模式，不能把新/新比较当作历史差值。发布文件保存本轮实际观测的原始对照统计和成员，不增加历史兼容实现。

核验：3840个函数的所有非图字段逐值相等，CPG边以端点原始跨度/类型和边类型的多重集合核对一致；解析状态、模式坐标和覆盖率一致。71个已有模型/配置/日志/预测/汇总文件的大小和修改时间未变。
必要检查：70项单元测试通过（静态模块29项）；其中含真实 CPG 路径正反例、C++下标、绑定/别名/重定义、字符串语义、恢复语法/UTF-8，以及原有真实 tiny-Transformer forward/backward/save-reload 检查。没有重新训练现有 CodeBERT。

环境：Python 3.11.14；torch 2.9.1，transformers 4.57.3，tree-sitter 0.25.2，tree-sitter-c 0.24.1，tree-sitter-cpp 0.23.4；Joern 4.0.465 / JDK21，单JVM 4GiB堆、4CPU，顺序批处理。

剩余限制：没有项目头文件、宏展开、类型布局或跨过程对象容量时继续 unknown；局部可达并非可满足路径。当前样本没有足够证据把零覆盖提升为真实漏洞匹配，应保留此结果，而不是扩充宽松 API/阈值凑覆盖率。
