# C/C++ 跨过程内存安全语义 Demo

该 Demo 用结构化、可验证的安全语义补充目标函数中缺失的跨过程信息。
当前语义包括 ALLOC、READ、WRITE、GUARD 和 VALUE。

LLM 只读取单个候选函数并输出固定 JSON，不直接判断漏洞。Joern 负责验证参数到
分配、读写操作的数据流、真实比较条件和返回值关系。验证后的语义再传播回入口函数。
对于跨过程内存访问，系统生成标准 Verification Condition（VC）和 Path Constraint，
并使用 Z3 判断 bounds constraint 是否可被证明、可能违反或因信息不足而未知。

检测输入为 data/detection_samples.jsonl。其中不含 CVE、fixing commit、补丁、
漏洞描述或人工结论。data/oracle.jsonl 只在检测结果落盘后读取。

## 环境

安装 Python 依赖：

    python -m pip install -r requirements.txt

默认使用以下本地资源：

    Joern: /home/phy/joern
    JDK: /home/phy/jdk21
    llama.cpp: /home/phy/llama.cpp/build/bin/llama-server
    Qwen: /home/phy/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-MXFP4_MOE.gguf

## 推荐运行顺序

先运行单元测试：

    python -m unittest discover -s tests -v

重新生成 LLM normalization。该命令默认自动启动和停止本地 Qwen，不读取
DEEPSEEK 环境变量，源码不会发送到外部：

    python -m semantic_demo.cli normalize --normalizer llm

然后运行 Joern 验证和检测：

    python -m semantic_demo.cli run --joern-dir /home/phy/joern

只有明确需要外部 OpenAI-compatible API 时，才配置 DEEPSEEK_API_KEY、
DEEPSEEK_BASE_URL 和 DEEPSEEK_MODEL，并显式指定 api：

    python -m semantic_demo.cli normalize --normalizer llm --llm-backend api

如需指定其他本地模型或 llama-server：

    python -m semantic_demo.cli normalize --normalizer llm \
      --local-model /path/to/model.gguf \
      --llama-server /path/to/llama-server

临时绕过 Joern 的调试命令：

    python -m semantic_demo.cli run --no-joern

## 实现

- semantic_demo/source.py 读取指定 Git revision，并用 tree-sitter 提取函数、参数和调用。
- semantic_demo/semantics.py 定义语义、筛选候选、调用 LLM，并执行结构化静态校验。
- semantic_demo/joern.py 调用本地 Joern，并解析 CPG 和数据流事实。
- semantic_demo/joern_extract.sc 提取参数、调用参数、数据流、比较和返回值事实。
- semantic_demo/analyzer.py 把验证后的跨过程语义传播回入口函数，并调用约束推理。
- semantic_demo/z3_reasoner.py 生成 Verification Condition、Path Constraint 和 bounds constraint，并用 Z3 求解。
- semantic_demo/cli.py 管理本地 Qwen、执行 normalization，并隔离检测与 oracle。

## 输出

- results/validated_semantics.jsonl 保存每条摘要的验证状态和拒绝原因。
- results/detections.jsonl 保存不读取 oracle 得到的 baseline 和 proposed 结果。
- results/results.csv 保存与人工核验机制合并后的完整结果表。
- results/summary.md 保存汇总结果。

results/ 目录中的现有结果均早于 Z3 Verification Condition 推理版本，仅作为历史对照。
当前 Proposed 已改为：Verified Semantic IR -> Verification Condition / Path Constraint
-> Z3 bounds reasoning。重新安装 requirements 后，必须重新运行 normalization 与 run
才能评价这一版结果。

Z3 输出三种状态：
- SAFE：当前约束可以证明生成的 bounds condition。
- POTENTIAL_VIOLATION：Z3 找到可违反 bounds condition 的满足解。
- UNKNOWN：缺少 capacity、valid extent 或无法可靠编码的关系，不强行判漏洞。
