# C/C++ 项目自定义内存语义 Demo

该 Demo 在 10 个本地真实漏洞仓库上恢复并验证 ALLOC、WRITE 和 GUARD。
候选函数来自 vulnerable revision 中入口函数的直接调用，并通过函数名和函数体中的
内存操作做静态筛选。LLM 只读取单个候选函数并输出固定 JSON。保存的归一化结果由本机
Qwen3.6-35B-A3B-MTP-GGUF 生成。

检测输入为 data/detection_samples.jsonl。其中不含 CVE、fixing commit、补丁、漏洞描述
或人工结论。data/oracle.jsonl 只在 baseline 和 proposed 结果落盘后读取，用于合并
fixing patch 和完整仓库人工核验结果。

## 运行

    python -m pip install -r requirements.txt
    python -m semantic_demo.cli run

默认复放 data/normalizer_outputs.jsonl，重新从 vulnerable revision 验证每条语义，
然后分别运行 baseline 和 proposed。当前环境需要保留 /SSD/phy/bln-cache/repositories
下的本地 bare repositories。

如需使用 OpenAI-compatible 接口重新归一化，可配置 DEEPSEEK_API_KEY、
DEEPSEEK_BASE_URL 和 DEEPSEEK_MODEL：

    python -m semantic_demo.cli normalize --normalizer llm
    python -m semantic_demo.cli run

## 实现

- semantic_demo/source.py 读取指定 Git revision，并用 tree-sitter 提取函数、参数和调用。
- semantic_demo/semantics.py 筛选候选、调用 LLM，并验证固定 JSON 与真实调用数据流。
- semantic_demo/analyzer.py 把通过验证的语义传播到调用点，再执行与 baseline 相同的规则。
- semantic_demo/cli.py 先完成检测落盘，再读取独立 oracle 合并人工核验结果。

主要输出为：

- results/validated_semantics.jsonl：每条模型摘要、通过状态和拒绝原因。
- results/detections.jsonl：不读取 oracle 得到的 baseline 和 proposed 结果。
- results/results.csv：10 个 CVE 的完整结果表、fixing patch 和人工核验机制。
- results/summary.md：检出统计、结论和主要失败原因。

当前结果为 baseline 检出 3/10，proposed 检出 6/10，纠正 3 个 baseline 漏检。
该结果是明确但有限的正向信号。样本全部为漏洞样本，并优先选择了自定义内存函数，
因此不能由此估计误报率。
