# Function-Level C/C++ Memory-Safety Verification Demo

该项目面向函数级 C/C++ 源码漏洞检测。Repository 仅作为跨过程上下文来源，最终预测单元始终是目标函数。

Baseline 使用官方 LineVul 函数级模型。Proposed 不使用漏洞启发式规则；正式主流程固定为：

    target function + repository revision
        -> repository-resolved call graph
        -> static semantic endpoints / standard effects
        -> localized LLM normalization for unresolved custom endpoints
        -> Joern data-flow validation
        -> validated ALLOC / READ / WRITE / VALUE effects
        -> Tree-sitter path / reaching-definition facts
        -> per-access Verification Conditions
        -> Z3
        -> VULNERABLE / UNKNOWN

LLM 不直接判断漏洞。Joern 负责验证自定义函数摘要是否由真实数据流和明确标准 API 支撑；
Z3 只对已经结构化并可可靠编码的程序事实求解。无法可靠恢复的事实统一进入 UNKNOWN，
不会通过函数名、变量名、正则漏洞模式或自由符号反例补全。

## Baseline

LineVul 只读取目标函数源码：

    CodeBERT base:
      /home/PublicData/PHY-data/resource/codebert-base

    Official LineVul checkpoint:
      /home/PublicData/PHY-data/resource/linevul/12heads_linevul_model.bin

默认 block size 为 512，函数级阈值为 0.5。

## Proposed 主流程

### 1. Repository call-graph discovery

- Tree-sitter 从目标函数开始提取项目调用。
- 标准 API 是明确 leaf，不依赖函数名相似度判断。
- 自定义调用通过 repository revision 中的真实函数定义解析。
- 同一 C 文件、同参数签名的条件编译实现作为显式 variants；跨文件、C++ 或不同签名歧义仍不猜测。
- 不再使用固定 hop、函数名 hints、read/copy/alloc family rule。
- 仅展开可能产生当前 ALLOC/READ/WRITE/VALUE caller-visible summary 的函数；无值返回且无 pointer-like 形参的 callee 及其后继不进入候选。
- 调用图按真实函数身份去重，不设置固定 hop 或固定函数数截断。

### 2. Semantic normalization

Normalization 采用静态定位优先的混合流程。Tree-sitter 先确定标准 API effect、return
以及自定义直接调用 endpoint；memcpy/read/write/malloc 等明确标准语义直接结构化，不再交给 LLM。
LLM 只处理单个 return 或自定义 direct-call endpoint，每次最多输出 4 条 summary，且仍只允许：

    ALLOC(return, size)
    READ(buffer, length)
    WRITE(buffer, length)
    VALUE(return, expression)

间接调用作为 opaque edge：不会据此推导 summary，但也不会使函数中与其无关的直接语义整体失效。
GUARD 仍不进入跨过程 schema。Normalization 输出带 schema version；旧 schema、旧 prompt 或旧缓存
不会被新主流程静默复用。函数过长超过显式 LLM source budget 时直接报错，不做静默字符截断。

### 3. Joern validation

- Joern 是正式流程必需组件，没有 lightweight fallback。
- 验证基于明确标准 API 的参数角色或已经验证的 callee summary composition。
- 不再根据 custom API 名称中是否包含 read、recv、send、copy、alloc、parse 等词猜角色。
- GUARD/VALUE 不再通过 substring 匹配。
- VALUE 接受精确 return expression；wrapper VALUE 仅在已验证 callee summary 可组合时传播。
- Joern 使用候选函数所在的完整 translation unit，而不是只把函数体写成孤立 candidate.c。
- 同名 C variants 使用源码范围定位 Joern method；无法按候选源码范围唯一解析时拒绝该验证结果。

### 4. Source parsing and access recovery

Tree-sitter 根据文件语言选择 C 或 C++ parser：

    C:   .c / C-style .h
    C++: .cc / .cpp / .cxx / .hh / .hpp / .hxx

.h 文件会比较 C/C++ 解析错误数量后选择更合适的 parser。

直接 memory access 当前包括：

- array subscript: a[i]
- pointer dereference: *p
- 明确标准 memory API

局部数组容量只来自 AST declaration -> array_declarator。
普通 a[i] 访问不会反向污染对象容量。多维数组在当前轻量 shape model 无法可靠建模时不发布
错误的 1-D capacity。

### 5. Object capacity and units

- AST subscript / dereference 使用 element capacity。
- memcpy/read/write 等 API 使用 byte capacity。
- 二者不混用。
- 只有类型拼写本身能确定字节宽度时才计算 byte capacity，例如 char、uint8_t、uint16_t、
  uint32_t、uint64_t。
- ABI 相关的 int、short、long、double、struct 等不会硬编码大小。
- heap allocation 不再通过字符串正则猜 element count。
- 无法确定的 sizeof、宏常量或 object bound 直接 UNKNOWN。

### 6. Path and value facts

Path Constraint 只加入结构上必然在访问点成立的条件：

- access 所在 if/else 分支条件；
- enclosing for/while/do 条件；
- 只有分支所有路径都 return 时，才对后续访问加入该条件的反条件。

break、goto、块中“存在一个 return”都不会被错误解释成函数路径终止。

Value Constraint 使用保守 reaching-definition：

- 只保留访问点可达顺序块中的简单支配定义；
- 同一变量保留最后一个顺序定义；
- sibling branch / loop 中的赋值不与主路径定义混合；
- 不再把函数中所有历史 assignment 一起塞入 Z3。

Validated VALUE summary 会在 caller 中形成真实 result = expression 等式。

### 7. Solver boundary

Z3 采用 per-access Verification Condition：

    extent >= 0
    extent <= capacity
    offset + extent <= capacity

但只有当前模型能可靠表示相关事实时才求解。

以下情况直接 UNKNOWN：

- parser error；
- unknown object capacity / valid extent；
- unresolved compile-time macro or sizeof；
- unsupported expression；
- 未建模的 C integer overflow / wraparound arithmetic；
- signed parameter domain 缺失；
- reaching definition 无法可靠编码；
- path constraint 无法可靠编码；
- access coverage 尚不完整。

未知事实不会作为任意 free integer 用于制造 SAT counterexample。

函数级 Proposed 当前是 selective verifier：

    任一 access 有可信 POTENTIAL_VIOLATION -> VULNERABLE
    否则 -> UNKNOWN

在尚未证明完整 function-level memory-access coverage 前，不输出 SAFE_FUNCTION。

## Standard effects

当前只对参数角色和访问范围足够明确的标准 API 建模，包括 malloc/calloc/realloc、
memcpy/memmove/memset、read/recv/recvfrom/fread、write/send/sendto/fwrite、memcmp 等。

memcpy/memmove 同时生成 destination WRITE 与 source READ。
memcmp 同时生成两个 source READ。

strcpy/strcat 只在能够恢复长度表达式时生成结构化 effect；动态 strlen 无法编码时保持 UNKNOWN。
sprintf/vsprintf 不再使用“出现即漏洞”的规则，无法可靠得到输出长度时保持 UNKNOWN。

## Environment

安装依赖：

    python -m pip install -r requirements.txt

默认路径：

    Joern: /home/phy/joern
    JDK: /home/phy/jdk21
    llama.cpp: /home/phy/llama.cpp/build/bin/llama-server
    Qwen: /home/phy/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-MXFP4_MOE.gguf

## Running

先运行测试：

    python -m unittest discover -s tests -v

由于 semantic schema 已更新，第一次必须重新生成 normalization：

    python -m semantic_demo.cli normalize --refresh

默认使用本地 Qwen。外部 OpenAI-compatible API：

    python -m semantic_demo.cli normalize --llm-backend api --refresh

然后运行 LineVul、Joern validation 和 Proposed verifier：

    python -m semantic_demo.cli run --joern-dir /home/phy/joern --refresh

后续输入 fingerprint 未变化时可以去掉 --refresh 复用检查点。

## Evaluation

detection manifest 不包含 CVE、fix commit、patch、mechanism 或 ground truth。
oracle 只在 detection 结果已经落盘后用于评估。

LineVul 仍按普通 binary classifier 统计。

Proposed 单独报告：

- TP / FP / TN / FN
- UNKNOWN vulnerable
- UNKNOWN benign
- Precision
- global Recall
- F1
- decided-sample Accuracy
- Coverage / abstention

UNKNOWN 不再自动折算成 benign。

## Files

- semantic_demo/source.py
  - C/C++ Tree-sitter parsing
  - repository function resolution
  - calls / direct accesses
  - local arrays
  - structural path facts
  - reaching definitions
- semantic_demo/semantics.py
  - LLM normalization
  - schema validation
  - Joern-backed semantic validation
  - unique callee composition
- semantic_demo/joern.py / joern_extract.sc
  - full-translation-unit CPG/data-flow facts
- semantic_demo/analyzer.py
  - standard + validated custom effects
  - no vulnerability heuristics
- semantic_demo/z3_reasoner.py
  - conservative object-bound and VC reasoning
- semantic_demo/linevul_baseline.py
  - independent LineVul function-level baseline
- semantic_demo/cli.py
  - normalization, validation, detection, checkpointing and isolated evaluation
