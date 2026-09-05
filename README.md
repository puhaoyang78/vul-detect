# Function-Level C/C++ Memory-Safety Verification with Repository Context

本项目的最终预测单位是**指定目标函数**，但使用固定 Git revision 的仓库源码作为跨过程程序上下文：

> **function-level vulnerability prediction with repository-level program context**

它不是“输入任意仓库后自动发现其中所有漏洞”的全仓扫描器。

## Pipeline

```text
fixed repository revision + target function
        ↓
preflight
  source/context materialization
  Joern CPG + exact entry resolution
  memory-relevance candidate slicing
  versioned candidate manifest
        ↓
normalize
  static standard-API semantics
  relevance-sliced local context
  LLM normalization of unresolved custom endpoints
  ALLOC / READ / WRITE / VALUE
        ↓
run
  Joern summary validation
  validated wrapper composition
  target standard/custom memory effects
  Tree-sitter local path/value facts
  dependency-local opaque-call barriers
  Z3 per-access verification
        ↓
VULNERABLE / UNKNOWN
```

LLM 不直接判断漏洞，只把静态分析已经定位的 custom helper endpoint 归一为结构化语义摘要。

## Commands

只使用这一套入口：

```bash
python -m semantic_demo.cli preflight
python -m semantic_demo.cli normalize
python -m semantic_demo.cli run
```

### 1. Preflight

`preflight` 是唯一执行 candidate discovery 的阶段。它负责：

1. 验证 Git revision、scan paths、entry path；
2. 物化源码、include/config context 与 symlink target；
3. 构建或复用 Joern CPG/index；
4. 精确定位目标函数；
5. 从内存相关值出发做 candidate relevance slicing；
6. 写入 versioned candidate manifest；
7. sample 完成后立即 checkpoint。

Candidate 不再等于整个静态可达调用图。Manifest 为每个 candidate 保存：

```text
source_path
function
source_line
method_full_name
call_lines
depth
caller
selection_reason
source_fingerprint
variant metadata
```

因此 candidate 数量异常时可以直接追踪选择原因。Opaque typedef 不会按明确 scalar 静默裁掉。

重复执行：

```bash
python -m semantic_demo.cli preflight
```

输入和 candidate policy 未变化时直接复用 checkpoint。显式刷新：

```bash
python -m semantic_demo.cli preflight --refresh
```

`--refresh` 只作用于当前 `--samples` 选中的样本，不删除其他样本状态。

### 2. Normalize

`normalize` 只消费 candidate manifest，不重新执行 preflight 或 candidate discovery。

标准 API 统一由 `semantic_demo/standard_semantics.py` 定义，candidate selection、normalization、validation 和 target analyzer 共用同一 registry，包括常见 allocator、memcpy/memmove/memset、read/write/recv/send、strcpy/strcat、strncpy/strncat、strlcpy/strlcat、sprintf/snprintf 等。

小函数可直接使用完整函数；大函数使用程序生成的 relevance slice：

```text
endpoint
  ↓
endpoint arguments / return value
  ↓
reaching definitions
  ↓
相关控制条件
  ↓
函数签名
```

LLM 不自己选择仓库上下文。

每完成一个 candidate 就写入 `data/normalizer_outputs.jsonl`。再次运行时，schema、实现指纹、source fingerprint、backend 和 model 均匹配的结果直接复用。完整样本不会重新调用模型；没有 LLM pending 时不会启动 llama-server。

子集运行和 `--refresh` 只替换选中样本，不覆盖其他样本，也会清理选中样本中已经不属于当前 manifest 的陈旧 candidate 记录。

### 3. Run

`run` 要求选中样本的 candidate manifest 和 normalization 都完整，不执行 preflight，不重新发现 candidate，也不启动 LLM。

```text
candidate summaries
   ↓
Joern validation
   ↓
fixed-point wrapper composition
   ↓
entry memory operations
   ↓
Tree-sitter path/value facts
   ↓
per-access constraints
   ↓
Z3
```

单个 candidate/TU 的 Joern 解析或 dataflow 失败会把相关 summary 标为 REJECT/UNKNOWN，不会销毁已经完成的样本；Joern/JDK 整体不可用仍属于 fatal infrastructure error。

Run 使用 sample-level upsert。中途失败时，新完成样本写回，尚未处理样本的历史结果保留。

## Candidate relevance

递归 helper 只在能够影响 caller-visible memory semantics 时继续扩展，主要包括：

- callee result 进入 caller return；
- callee result 到达 memory-relevant value；
- callee argument 依赖 memory-relevant value；
- caller pointer/value 流入 summary-capable callee。

预处理导致 Joern CALL 坐标与原始 C 行号不一致时，如果 Joern 已明确解析 callee，不会因为行号无法精确对齐而静默漏掉该 callee；manifest 会记录相应 selection reason。

## Verification boundary

当前 Proposed 是 selective verifier：

```text
可信 counterexample -> VULNERABLE
其余             -> UNKNOWN
```

尚未证明完整函数级访问覆盖，因此不输出 `SAFE_FUNCTION`。

UNKNOWN 是 dependency-local 的：无关 parser error、无关 indirect/custom call 不会直接让整函数退出；只有 unresolved call 与当前访问的 buffer/extent/capacity 共享相关值时，该 access 才进入 UNKNOWN。

对 `n + 1`、`count * size` 等符号算术，不再机械地全部拒绝。当前只在明确 fixed-width 类型和路径约束足以证明不发生 wraparound 时继续使用整数模型，否则保持 UNKNOWN。

## Preprocessed translation units

对于 manifest 明确需要预处理的 C entry translation unit，repository-level Joern 和 run 阶段 TU validation 使用一致的 preprocessing 决策。普通 candidate 文件不会错误继承 `--with-preprocessed-files`。

## Cache safety

缓存通过输入指纹和实现指纹失效。修改 candidate policy、normalization、standard semantics、validation、analyzer、solver 或 Joern validation 实现后，不会依赖手工记忆版本号静默复用不兼容结果。

## Evaluation

Detection manifest 不包含 CVE、fix commit、patch、mechanism 或 ground truth。Oracle 只用于结果落盘后的评估。

Proposed 报告：

- TP / FP / TN / FN
- UNKNOWN vulnerable / benign
- Precision
- global Recall
- F1
- decided-sample Accuracy
- Coverage / abstention

由于 Proposed 是 selective verifier，F1 必须和 Coverage、UNKNOWN 一起解释，不能当作普通二分类器 F1 单独比较。

## Default environment

```text
Joern: /home/phy/joern
JDK: /home/phy/jdk21
llama-server: /home/phy/llama.cpp/build/bin/llama-server
Qwen: /home/phy/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-MXFP4_MOE.gguf
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

正式实验建议依次执行：

```bash
python -m semantic_demo.cli preflight --refresh
python -m semantic_demo.cli normalize --refresh
python -m semantic_demo.cli run --refresh
```

之后输入和实现未变化时去掉 `--refresh`，直接使用 checkpoint/resume。
