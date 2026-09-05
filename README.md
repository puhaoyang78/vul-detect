# Function-Level C/C++ Memory-Safety Verification with Repository Context

本项目的**最终预测单位仍然是指定目标函数**，但分析时使用固定 Git revision 的仓库源码作为跨过程上下文。因此更准确的任务定义是：

> **function-level vulnerability prediction with repository-level program context**

它不是“给任意仓库自动找出所有漏洞”的全仓扫描器。

## Current pipeline

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
  target-function standard/custom memory effects
  Tree-sitter local path/value facts
  dependency-local opaque-call barriers
  Z3 per-access verification
        ↓
VULNERABLE / UNKNOWN
```

LLM 不直接判断漏洞。它只把已经由静态分析选定的 custom helper endpoint 归一为结构化函数摘要。

## One public CLI

用户只使用这一套命令：

```bash
python -m semantic_demo.cli preflight
python -m semantic_demo.cli normalize
python -m semantic_demo.cli run
```

`semantic_demo/legacy_cli.py` 只是内部执行引擎，不是第二套用户流程。

---

## 1. Preflight

`preflight` 是**唯一允许发现 candidate helper 的阶段**。

它负责：

1. 验证固定 Git revision、scan paths、entry path；
2. 物化源码、include/config context、symlink target；
3. 构建或复用 Joern CPG/index；
4. 精确解析目标函数，不使用 fuzzy entry matching；
5. 基于 memory-relevance slice 发现需要理解的 custom helper；
6. 保存 versioned candidate manifest；
7. 每个 sample 完成后立即 checkpoint。

Candidate 不再等价于“整个静态可达调用图”。

选择逻辑从目标函数中的内存相关值出发，包括 buffer、length、capacity、allocation result、return-derived value 等，再沿局部 reaching definitions 和参数/返回值流决定 custom call 是否值得继续展开。

Manifest 为每个 candidate 保存：

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

因此 candidate 数量异常时可以直接追查“为什么被选中”。

Opaque typedef 不会被当作明确标量静默裁掉；只有已知 scalar type 才允许据此裁剪。

### Resume / refresh

正常重复执行：

```bash
python -m semantic_demo.cli preflight
```

sample/index/discovery policy/manifest 都未变化时直接跳过。

显式刷新：

```bash
python -m semantic_demo.cli preflight --refresh
```

`--refresh` **只刷新 `--samples` 中选中的 sample**，不会删除其他 sample 的 checkpoint。

---

## 2. Normalize

`normalize` **只消费 preflight candidate manifest**，不重新执行 preflight，也不重新遍历调用图。

### Standard semantics

标准 API 统一由 `semantic_demo/standard_semantics.py` 定义，candidate discovery、normalization、validation、target analyzer 共用同一 registry。

当前包括：

```text
malloc/calloc/realloc/kmalloc/kzalloc/vmalloc
memcpy/memmove/memset/memcmp
read/recv/recvfrom/fread/ReadFile
write/send/sendto/fwrite
strcpy/strcat/strncpy/strncat/strlcpy/strlcat
sprintf/vsprintf/snprintf/vsnprintf
```

`strcpy` 等 wrapper 不会再出现“被当作 standard leaf，但 normalize/validator 没有语义”的断层。

### LLM relevance slice

对于 custom endpoint，大函数不再发送完整函数，也不再使用固定 ±40 行窗口作为唯一上下文。

程序先构造静态 slice：

```text
endpoint arguments / return
        ↓
reaching assignments
        ↓
相关变量定义
        ↓
相关控制条件
        ↓
函数签名
```

LLM 只解释这个已经选好的 slice，不能自己到仓库里选择上下文。

允许的跨过程 schema 仍只有：

```text
ALLOC(return, size)
READ(buffer, length)
WRITE(buffer, length)
VALUE(return, expression)
```

### Checkpoints

每个 candidate 完成后立即写 `data/normalizer_outputs.jsonl`。

缓存匹配同时检查：

```text
schema version
normalization implementation version
source fingerprint
backend
model
```

如果所有 candidate 都已完成，local llama-server 不会启动。

子集运行和 `--refresh` 都只更新选中的 samples；未选样本不会被覆盖。旧 manifest 中已经删除的 candidate 记录会从选中 sample 的 normalization 集合中清理。

---

## 3. Run

`run` 要求：

```text
valid preflight manifest
+
manifest 中所有 candidate 均有当前 normalization
```

缺失时直接在开始前报错，不会运行到中途才偷偷补做 discovery。

Run 只对 selected samples 创建临时 replay/detection/semantic 文件，再按 sample upsert 回正式结果。因此：

- 已完成 sample 可以直接 resume；
- 后段 sample 报错不会删除前面结果；
- 子集运行不会覆盖未选择 sample；
- run 不要求用户再次声明 normalize 使用的模型。

Analysis fingerprint 自动包含 analyzer、Z3-v2、validation-v2、Joern-v2 和 standard semantics 实现哈希。修改核心分析代码会自动使旧 detection cache 失效。

---

## Joern validation

Repository-level CPG 用于 exact METHOD/CALL binding。

Run 阶段按 candidate translation unit 构建/复用 contextual facts。对于像 PJSIP S53 这种 entry 需要真实预处理的样本：

- entry TU validation 会复用同一 preprocessing decision；
- 只有该 entry TU 使用 `--with-preprocessed-files`；
- 其他 candidate 文件不会错误继承这个 flag；
- `.i` 坐标无法直接对应原 C 文件时，只在 name + parameter count 唯一时恢复 method identity，否则拒绝该 candidate。

Candidate/TU 局部 Joern 失败记为 summary rejection，不再让整个 60-sample run 直接退出；Joern/JDK 整体不可用仍属于 fatal infrastructure error。

---

## Target analysis and solver boundary

Target analyzer 使用：

1. source-level standard memory effects；
2. direct AST accesses；
3. validated custom summaries；
4. unresolved custom/indirect calls 作为 OPAQUE barriers。

Opaque call 不再让整个函数直接 UNKNOWN。只有当它出现在某个 access 之前，并与该 access 的 buffer/extent/capacity 共享相关值时，那个 access 才 UNKNOWN。

同样，目标函数存在无关 parser ERROR 时不再全函数一票否决；已有可靠 fact 仍可继续验证，但整体没有完整 coverage 时仍不会输出 SAFE_FUNCTION。

### Arithmetic

Z3 仍然坚持保守边界。

对于普通无法证明 C wraparound 安全的符号算术，仍然 UNKNOWN；但对于 `uint32_t/int32_t/uint64_t/int64_t` 等明确固定宽度类型，如果已有 path constraints 能证明相关 `+/-/*` 表达式不会越过类型范围，则允许继续使用整数模型求解。

因此不再是简单的“看到 `len + 1` 就一律 UNKNOWN”，但也不会把可能溢出的 C 表达式当无限精度整数。

最终函数级 Proposed 仍是 selective verifier：

```text
任一可信 access counterexample → VULNERABLE
否则 → UNKNOWN
```

在未建立完整 memory-access coverage 前不输出 SAFE_FUNCTION。

---

## Baseline

LineVul 仍只读取目标函数源码，默认：

```text
CodeBERT: /home/PublicData/PHY-data/resource/codebert-base
Checkpoint: /home/PublicData/PHY-data/resource/linevul/12heads_linevul_model.bin
block_size: 512
threshold: 0.5
```

---

## Recommended execution

当前 candidate policy、normalization implementation 和 workflow checkpoint version 已更新，因此本轮应重新生成阶段产物，但已有 Joern CPG/index 指纹未变化时仍会复用底层缓存。

```bash
# 1. 重新生成 candidate manifests
python -m semantic_demo.cli preflight --refresh

# 2. 根据新 manifest 干净生成 normalization
python -m semantic_demo.cli normalize --refresh

# 3. validation + target analysis + evaluation
python -m semantic_demo.cli run --refresh
```

之后正常断点续跑直接去掉 `--refresh`：

```bash
python -m semantic_demo.cli preflight
python -m semantic_demo.cli normalize
python -m semantic_demo.cli run
```

先运行测试：

```bash
python -m unittest discover -s tests -v
```

## Evaluation

Proposed 是 selective verifier，因此报告：

```text
Precision
Global Recall
F1
Decided-sample Accuracy
Coverage
UNKNOWN vulnerable
UNKNOWN benign
```

不能只把 Proposed F1 当作普通全覆盖 binary classifier F1 解读。
