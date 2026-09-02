# C/C++ 跨过程内存安全语义 Demo

该 Demo 面向函数级 C/C++ 内存安全检测，目标不是给大模型更多原始代码，而是恢复并验证
目标函数中缺失的跨过程安全语义，再将这些语义转化为可求解的程序约束。

当前安全语义包括 ALLOC、READ、WRITE、GUARD 和 VALUE。LLM 只负责把项目自定义函数
归一化成固定语义；它不直接判断漏洞。Joern 用于验证参数角色和数据流；Z3 用于求解
Verification Condition（VC）、Path Constraint 和 Bounds Constraint。

检测输入为 data/detection_samples.jsonl，其中不含 CVE、fixing commit、补丁、漏洞描述
或人工结论。data/oracle.jsonl 只在检测结果落盘之后用于评估。

## 当前分析流程

1. **Selective semantic frontier**
   - 目标函数中的项目自定义直接调用构成初始语义前沿。
   - 若候选函数已经直接接触标准内存 primitive，则在该处停止继续展开。
   - 只有当函数尚未暴露可验证内存行为时，才沿“参数继续传递给子调用”或“子调用结果继续返回”的调用链扩展。
   - 该策略不依赖固定 hop，也不构建整个可达调用图，从而减少无关函数和 LLM 调用。

2. **Semantic normalization**
   - LLM 每次只读取一个候选函数。
   - 输出固定 ALLOC / READ / WRITE / GUARD / VALUE JSON。
   - READ/WRITE 明确区分 source、destination 和 length 参数角色。

3. **Compositional static validation**
   - 直接接触 malloc/memcpy/read/recv/write/send 等标准 primitive 的摘要由 Joern 验证。
   - 已验证摘要作为下一层函数的语义模型。
   - 父函数摘要可由“已验证子摘要 + caller-to-callee 参数映射”组合验证。
   - 采用 fixed-point 迭代直到没有新的摘要能够通过验证，因此支持 wrapper -> wrapper -> primitive。

4. **Program-constraint extraction**
   - Tree-sitter AST 提取赋值/初始化得到 Value Constraint。
   - AST 提取 early-exit 分支在后续访问点成立的 Path Constraint。
   - 分配返回值和局部数组提供显式 buffer capacity。
   - 若缺少显式 capacity，但支配访问的边界条件约束了与访问长度存在数据依赖的值，则推导
     guard-derived access bound；该过程基于数据依赖，不依赖 buflen/size/capacity 等变量名。

5. **Per-access bounds verification**
   - 每一个 READ/WRITE 单独生成 Verification Condition。
   - 典型条件包括 extent >= 0、extent <= capacity、offset + extent <= capacity。
   - Z3 为每个 memory access 单独返回 SAFE / POTENTIAL_VIOLATION / UNKNOWN。
   - POTENTIAL_VIOLATION 表示当前已知 Path/Value Constraints 下存在违反 VC 的满足解；
     UNKNOWN 表示缺少 capacity、valid extent 或无法可靠编码的关系，不强行判漏洞。
   - 函数级结果最后再由各 memory-access 结果聚合，因此无关访问的 UNKNOWN 不会覆盖真正的违规访问。

## 环境

安装 Python 依赖：

    python -m pip install -r requirements.txt

默认使用：

    Joern: /home/phy/joern
    JDK: /home/phy/jdk21
    llama.cpp: /home/phy/llama.cpp/build/bin/llama-server
    Qwen: /home/phy/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-MXFP4_MOE.gguf

## 推荐运行顺序

先运行单元测试：

    python -m unittest discover -s tests -v

生成 semantic normalization。默认会复用同一 sample/path/function 的已有摘要，
只对 selective frontier 中新增的函数调用 LLM。每完成一个候选函数就原子保存，
中断后重新执行同一命令会从未完成的候选继续：

    python -m semantic_demo.cli normalize --normalizer llm

如需忽略缓存、强制重新生成全部摘要：

    python -m semantic_demo.cli normalize --normalizer llm --refresh

然后重新执行 Joern fixed-point validation 和 Z3 bounds verification：

    python -m semantic_demo.cli run --joern-dir /home/phy/joern

Joern 验证每完成一个样本就保存检查点。重新执行同一命令时，输入指纹一致的样本会
直接复用。需要忽略现有检查点并从 S01 重新运行时使用：

    python -m semantic_demo.cli run --joern-dir /home/phy/joern --refresh

只有需要外部 OpenAI-compatible API 时才显式指定：

    python -m semantic_demo.cli normalize --normalizer llm --llm-backend api

## 实现

- semantic_demo/source.py
  Tree-sitter 函数/调用解析、调用返回值绑定、AST Path/Value Constraint 提取。
- semantic_demo/semantics.py
  memory-semantic call-graph closure、LLM normalization、Joern role-sensitive validation、
  compositional fixed-point summary validation。
- semantic_demo/joern.py / joern_extract.sc
  CPG、参数到调用角色的数据流、比较和返回关系。
- semantic_demo/analyzer.py
  标准内存操作与验证后的跨过程语义统一传播到目标函数。
- semantic_demo/z3_reasoner.py
  per-memory-access Verification Condition 生成与 Z3 求解。
- semantic_demo/cli.py
  完整运行流程、结果落盘以及与 oracle 的隔离评估。

## 输出

- results/validated_semantics.jsonl：每条摘要的验证结果。
- results/detections.jsonl：逐样本操作、逐访问 Z3 结果和最终 verdict。
- results/results.csv：包含 Z3 status、Verification Conditions 和 counterexample model。
- results/summary.md：整体结果摘要。

results/ 目录中的现有数字来自本次方法重构之前的运行，仅作为历史对照。由于候选发现、
组合验证和 Z3 聚合方式均已改变，必须重新运行 normalization 和 run 后再评价新结果。


Z3 中 Path Constraint 与 buffer capacity 已严格分离。Guard 只作为路径约束；
当缺少显式 capacity 时，只有 guarded value 与实际 access extent 之间存在可编码的
def-use 数值关系，才会生成 guard-coverage VC。Guard 的右值不会再被解释成对象容量。
