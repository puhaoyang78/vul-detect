# C/C++ 跨过程内存安全语义 Demo

该 Demo 用结构化、可验证的安全语义补充目标函数中缺失的跨过程信息。
当前语义包括：

- `ALLOC(buffer,size)`
- `READ(buffer,length)`
- `WRITE(buffer,length)`
- `GUARD(relation)`
- `VALUE(return,expression)`

LLM 只读取单个候选函数，并输出固定 JSON。它不直接判断漏洞。
Joern 负责验证参数到分配/读写操作的数据流、真实比较条件和返回值关系。
验证后的语义再传播回入口函数，由现有轻量规则检查明显的不安全关系。

检测输入为 `data/detection_samples.jsonl`。其中不含 CVE、fixing commit、补丁、
漏洞描述或人工结论。`data/oracle.jsonl` 只在检测结果落盘后读取。

## 环境

Python 依赖：

    python -m pip install -r requirements.txt

默认 Joern 路径：

    /home/phy/joern

当前开发环境对应 Joern 4.0.465。也可以通过 `--joern-dir` 或
`JOERN_HOME` 指定其他安装目录。

## 推荐运行顺序

先运行不依赖 Joern 的单元测试：

    python -m unittest discover -s tests -v

然后用当前保存的 normalization 输出做 Joern 验证和检测：

    python -m semantic_demo.cli run --joern-dir /home/phy/joern

要真正使用新增的 READ/VALUE 语义，需要重新调用 LLM normalization。
配置一个 OpenAI-compatible 接口：

    export DEEPSEEK_API_KEY=...
    export DEEPSEEK_BASE_URL=...
    export DEEPSEEK_MODEL=...

然后运行：

    python -m semantic_demo.cli normalize --normalizer llm
    python -m semantic_demo.cli run --joern-dir /home/phy/joern

如需临时绕过 Joern，仅用于调试：

    python -m semantic_demo.cli run --no-joern

## 实现

- `semantic_demo/source.py`
  读取指定 Git revision，并用 tree-sitter 提取函数、参数和调用。
- `semantic_demo/semantics.py`
  筛选候选、调用 LLM、定义 ALLOC/READ/WRITE/GUARD/VALUE，并执行语义校验。
- `semantic_demo/joern.py`
  调用本地 Joern，并解析 CPG/data-flow facts。
- `semantic_demo/joern_extract.sc`
  Joern 4.x 脚本，提取参数、调用参数、data flow、guard/operator 和 return facts。
- `semantic_demo/analyzer.py`
  把通过验证的跨过程安全语义传播回入口函数。
- `semantic_demo/cli.py`
  保证检测阶段与 fixing patch/oracle 隔离。

## 输出

- `results/validated_semantics.jsonl`
  每条摘要的 Joern 验证状态和拒绝原因。
- `results/detections.jsonl`
  不读取 oracle 得到的 baseline/proposed 检测结果。
- `results/results.csv`
  检测完成后与人工核验机制合并的结果表。
- `results/summary.md`
  汇总结果。

仓库中现有的 3/10 -> 6/10 结果来自 Joern 接入前的 Demo 0，只能视为初步信号。
合并 Joern 和 READ/VALUE 后应在本地重新生成结果，不应直接沿用旧数字作为新方法结果。
