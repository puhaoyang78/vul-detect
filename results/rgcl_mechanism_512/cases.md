# Test 代表性案例：原始源码、Diff 与预测变化

仅使用既定 Valid P-C checkpoint 的已有 Test 预测。以下六例为事后解释，不用于模型、阈值或 Epoch 选择。
先按新增/丢失 P-C、持续失败、排序正确但两侧判漏洞、相同可见输入分组，再优先检查短函数。选择444/215的多种子收益、318的多种子损失、148的清晰持续误报、275的排序与分类区别，以及189的不可见边界。它们不是随机抽样，不能用六例的频数估计总体收益。
案例行号从1开始；统计 JSON 中物理行号从0开始。源码正文直接来自本地官方 Test 文件，未重新格式化。Diff 行标签为正式双向 `_line_diff` 结果；unified Diff 仅用于展示前后变化。概率按原始保存值判定，表中显示9位有效数字。
最后的行表复用 Seed123456 的保存倍率与 Token 数：倍率 r 在1–5之间，归一化行注意力质量为 n*r/sum(n*r)。这是实际导出值，不是根据概率推测“关注了哪行”；也不能证明注意力变化造成概率变化。

## primevul_paired:test:444 — 改善：纯新增的剩余长度检查

项目 `php-src`；修复提交 `b6fe458ef9ac1372b60c3d3810b0358e2e20840d`。成员 `primevul_paired:test:889` / `primevul_paired:test:890`；原始标签1/0。

源码Token数 62/75；可见输入相同：False。两侧可见性 no_direct_diff / full；可见Diff Token质量 0.0/11.0。

实际Diff行（1起）：漏洞侧 []；修复侧 [3]。

原函数无条件读取 p[0] 到 p[3]。新增 `if (end - p < 4) return 0;` 在剩余字节不足4时提前返回，直接约束随后四字节读取。两版本均完整可见；漏洞侧没有直接 Diff 正标签，不能把邻近 return 行补作监督。Seed42/1234 从 P-V 变成 P-C，Seed123456 保留 P-C。这里的改善主要是修复版本概率大幅下降，而非仅把漏洞侧概率升高。

| 种子 | Context漏洞概率 | Context修复概率 | Context Δp / 状态 | RGCL漏洞概率 | RGCL修复概率 | RGCL Δp / 状态 | Δp变化 |
|---:|---:|---:|---|---:|---:|---|---:|
| 42 | 0.999740064 | 0.999940991 | -0.000200927258 / P-V | 0.99982965 | 2.01063335e-06 | +0.999827639 / P-C | +1.00002857 |
| 1234 | 0.549814105 | 0.506472468 | +0.0433416367 / P-V | 0.999916911 | 5.97527219e-07 | +0.999916314 / P-C | +0.956574677 |
| 123456 | 0.999992132 | 0.000766857527 | +0.999225275 / P-C | 0.999926567 | 4.03800833e-08 | +0.999926527 / P-C | +0.000701252038 |

### 实际Diff

```diff
--- vulnerable
+++ repaired
@@ -1,4 +1,5 @@
 utf32be_mbc_to_code(const UChar* p, const UChar* end ARG_UNUSED)
 {
+  if (end - p < 4) return 0;
   return (OnigCodePoint )(((p[0] * 256 + p[1]) * 256 + p[2]) * 256 + p[3]);
 }
```

### 漏洞函数（原文）

```cpp
utf32be_mbc_to_code(const UChar* p, const UChar* end ARG_UNUSED)
{
  return (OnigCodePoint )(((p[0] * 256 + p[1]) * 256 + p[2]) * 256 + p[3]);
}
```

### 修复函数（原文）

```cpp
utf32be_mbc_to_code(const UChar* p, const UChar* end ARG_UNUSED)
{
  if (end - p < 4) return 0;
  return (OnigCodePoint )(((p[0] * 256 + p[1]) * 256 + p[2]) * 256 + p[3]);
}
```

### 实际修改行的导出倍率（Seed123456）

| 侧 | 物理行 | Token质量 | Context倍率 | RGCL倍率 | Context归一化质量 | RGCL归一化质量 |
|---|---:|---:|---:|---:|---:|---:|
| 修复 | 3 | 11 | 5.0000000 | 4.9994717 | 48.6725% | 47.6657% |

## primevul_paired:test:215 — 改善但不跨全部种子成立：空指针检查

项目 `tensorflow`；修复提交 `5b048e87e4e55990dae6b547add4dae59f4e1c76`。成员 `primevul_paired:test:431` / `primevul_paired:test:432`；原始标签1/0。

源码Token数 62/73；可见输入相同：False。两侧可见性 no_direct_diff / full；可见Diff Token质量 0.0/9.0。

实际Diff行（1起）：漏洞侧 []；修复侧 [4]。

如果 GetMutableInput 返回 nullptr，原函数会在 tensor->is_variable 处解引用。新增提前返回检查避免这一条空指针路径。Seed42/1234 从 P-V 变成 P-C；Seed123456 虽降低修复版本的概率，仍大于0.5，保持 P-V。补丁可见且含义明确也不保证每个种子都正确分类。

| 种子 | Context漏洞概率 | Context修复概率 | Context Δp / 状态 | RGCL漏洞概率 | RGCL修复概率 | RGCL Δp / 状态 | Δp变化 |
|---:|---:|---:|---|---:|---:|---|---:|
| 42 | 0.99992919 | 0.810075402 | +0.189853787 / P-V | 0.909337759 | 0.00043303892 | +0.90890472 / P-C | +0.719050933 |
| 1234 | 0.543682277 | 0.529475689 | +0.0142065883 / P-V | 0.597619951 | 0.0129336817 | +0.584686269 / P-C | +0.570479681 |
| 123456 | 0.999992728 | 0.999551594 | +0.000441133976 / P-V | 0.996015489 | 0.815639317 | +0.180376172 / P-V | +0.179935038 |

### 实际Diff

```diff
--- vulnerable
+++ repaired
@@ -1,5 +1,6 @@
 TfLiteTensor* GetVariableInput(TfLiteContext* context, const TfLiteNode* node,
                                int index) {
   TfLiteTensor* tensor = GetMutableInput(context, node, index);
+  if (tensor == nullptr) return nullptr;
   return tensor->is_variable ? tensor : nullptr;
 }
```

### 漏洞函数（原文）

```cpp
TfLiteTensor* GetVariableInput(TfLiteContext* context, const TfLiteNode* node,
                               int index) {
  TfLiteTensor* tensor = GetMutableInput(context, node, index);
  return tensor->is_variable ? tensor : nullptr;
}
```

### 修复函数（原文）

```cpp
TfLiteTensor* GetVariableInput(TfLiteContext* context, const TfLiteNode* node,
                               int index) {
  TfLiteTensor* tensor = GetMutableInput(context, node, index);
  if (tensor == nullptr) return nullptr;
  return tensor->is_variable ? tensor : nullptr;
}
```

### 实际修改行的导出倍率（Seed123456）

| 侧 | 物理行 | Token质量 | Context倍率 | RGCL倍率 | Context归一化质量 | RGCL归一化质量 |
|---|---:|---:|---:|---:|---:|---:|
| 修复 | 4 | 9 | 1.0110968 | 2.0416737 | 14.1964% | 15.0220% |

## primevul_paired:test:318 — 引入错误：实际报文长度约束可见，修复侧仍被误报

项目 `pjproject`；修复提交 `8b621f192cae14456ee0b0ade52ce6c6f258af1e`。成员 `primevul_paired:test:637` / `primevul_paired:test:638`；原始标签1/0。

源码Token数 212/275；可见输入相同：False。两侧可见性 no_direct_diff / full；可见Diff Token质量 0.0/49.0。

实际Diff行（1起）：漏洞侧 []；修复侧 [9, 10, 11, 12, 13, 16]。

原 reason.slen 只取目标缓冲区容量和报文声明长度的较小值；从 pkt+9 复制时，还需要受实际剩余 size-9 限制。新增 PJ_MIN(reason.slen, size-9) 提供这一约束。两版本都未截断。Seed1234/123456 从 P-C 退化为 P-V，Seed42 保留 P-C；Seed123456 还出现修复侧概率高于漏洞侧。新增 Diff 六行中五行为注释，按正式 Token 映射占33个 Token，实际长度检查占16个 Token。这里的 Test Diff 仅为事后描述，未参与推理或训练。Seed123456 对新增长度检查行的实际 RGCL 倍率约4.985，但修复侧概率仍约0.997，说明提高该行倍率本身不足以保证安全版本判别。

| 种子 | Context漏洞概率 | Context修复概率 | Context Δp / 状态 | RGCL漏洞概率 | RGCL修复概率 | RGCL Δp / 状态 | Δp变化 |
|---:|---:|---:|---|---:|---:|---|---:|
| 42 | 0.988277256 | 0.00134633598 | +0.986930921 / P-C | 0.999221087 | 8.92218741e-05 | +0.999131865 / P-C | +0.0122009441 |
| 1234 | 0.512158573 | 0.486103654 | +0.0260549188 / P-C | 0.999100566 | 0.985971868 | +0.0131286979 / P-V | -0.0129262209 |
| 123456 | 0.999941349 | 0.005650518 | +0.994290831 / P-C | 0.993143082 | 0.997135758 | -0.00399267673 / P-V | -0.998283508 |

### 实际Diff

```diff
--- vulnerable
+++ repaired
@@ -6,8 +6,15 @@
 
     /* Check and get BYE reason */
     if (size > 8) {
+    	/* Make sure the BYE reason does not exceed:
+    	 * - the size of the available buffer
+    	 * - the declared reason's length
+    	 * - the actual packet size
+    	 */
 	reason.slen = PJ_MIN(sizeof(sess->stat.peer_sdes_buf_),
                              *((pj_uint8_t*)pkt+8));
+        reason.slen = PJ_MIN(reason.slen, size-9);
+
 	pj_memcpy(sess->stat.peer_sdes_buf_, ((pj_uint8_t*)pkt+9),
 		  reason.slen);
 	reason.ptr = sess->stat.peer_sdes_buf_;
```

### 漏洞函数（原文）

```cpp
static void parse_rtcp_bye(pjmedia_rtcp_session *sess,
			   const void *pkt,
			   pj_size_t size)
{
    pj_str_t reason = {"-", 1};

    /* Check and get BYE reason */
    if (size > 8) {
	reason.slen = PJ_MIN(sizeof(sess->stat.peer_sdes_buf_),
                             *((pj_uint8_t*)pkt+8));
	pj_memcpy(sess->stat.peer_sdes_buf_, ((pj_uint8_t*)pkt+9),
		  reason.slen);
	reason.ptr = sess->stat.peer_sdes_buf_;
    }

    /* Just print RTCP BYE log */
    PJ_LOG(5, (sess->name, "Received RTCP BYE, reason: %.*s",
	       reason.slen, reason.ptr));
}
```

### 修复函数（原文）

```cpp
static void parse_rtcp_bye(pjmedia_rtcp_session *sess,
			   const void *pkt,
			   pj_size_t size)
{
    pj_str_t reason = {"-", 1};

    /* Check and get BYE reason */
    if (size > 8) {
    	/* Make sure the BYE reason does not exceed:
    	 * - the size of the available buffer
    	 * - the declared reason's length
    	 * - the actual packet size
    	 */
	reason.slen = PJ_MIN(sizeof(sess->stat.peer_sdes_buf_),
                             *((pj_uint8_t*)pkt+8));
        reason.slen = PJ_MIN(reason.slen, size-9);

	pj_memcpy(sess->stat.peer_sdes_buf_, ((pj_uint8_t*)pkt+9),
		  reason.slen);
	reason.ptr = sess->stat.peer_sdes_buf_;
    }

    /* Just print RTCP BYE log */
    PJ_LOG(5, (sess->name, "Received RTCP BYE, reason: %.*s",
	       reason.slen, reason.ptr));
}
```

### 实际修改行的导出倍率（Seed123456）

| 侧 | 物理行 | Token质量 | Context倍率 | RGCL倍率 | Context归一化质量 | RGCL归一化质量 |
|---|---:|---:|---:|---:|---:|---:|
| 修复 | 9 | 11 | 2.1313026 | 1.1063122 | 6.8538% | 2.4297% |
| 修复 | 10 | 8 | 4.9999986 | 3.8998344 | 11.6938% | 6.2290% |
| 修复 | 11 | 7 | 1.6595914 | 1.3541055 | 3.3962% | 1.8925% |
| 修复 | 12 | 6 | 1.0000001 | 1.0041482 | 1.7541% | 1.2029% |
| 修复 | 13 | 1 | 1.0000000 | 2.0170789 | 0.2923% | 0.4027% |
| 修复 | 16 | 16 | 5.0000000 | 4.9851565 | 23.3876% | 15.9251% |

## primevul_paired:test:148 — 没有改善整对分类：格式串修复被高置信误报

项目 `v4l2loopback`；修复提交 `64a216af4c09c9ba9326057d7e78994271827eff`。成员 `primevul_paired:test:297` / `primevul_paired:test:298`；原始标签1/0。

源码Token数 81/84；可见输入相同：False。两侧可见性 full / full；可见Diff Token质量 12.0/15.0。

实际Diff行（1起）：漏洞侧 [4]；修复侧 [4]。

原 snprintf 将 card_label[nr] 作为格式串；修复使用固定 "%s"，将标签作为普通字符串参数，避免把标签内容中的格式说明符当作格式控制。两版本完全可见。两个模型的三个种子均为 P-V，RGCL 没有获得 P-C；Seed1234 甚至排序方向错误。Seed123456 的漏洞侧和修复侧修改调用行实际 RGCL 倍率均为5。高倍率没有区分调用的两种语义。

| 种子 | Context漏洞概率 | Context修复概率 | Context Δp / 状态 | RGCL漏洞概率 | RGCL修复概率 | RGCL Δp / 状态 | Δp变化 |
|---:|---:|---:|---|---:|---:|---|---:|
| 42 | 0.995931208 | 0.998850822 | -0.00291961432 / P-V | 0.999904275 | 0.999631166 | +0.000273108482 / P-V | +0.0031927228 |
| 1234 | 0.534256279 | 0.534185767 | +7.05122948e-05 / P-V | 0.999373853 | 0.999826849 | -0.0004529953 / P-V | -0.000523507595 |
| 123456 | 0.99798429 | 0.997598469 | +0.000385820866 / P-V | 0.997598827 | 0.978712082 | +0.018886745 / P-V | +0.0185009241 |

### 实际Diff

```diff
--- vulnerable
+++ repaired
@@ -1,7 +1,7 @@
 static void vidioc_fill_name(char *buf, int len, int nr)
 {
 	if (card_label[nr] != NULL) {
-		snprintf(buf, len, card_label[nr]);
+		snprintf(buf, len, "%s", card_label[nr]);
 	} else {
 		snprintf(buf, len, "Dummy video device (0x%04X)", nr);
 	}
```

### 漏洞函数（原文）

```cpp
static void vidioc_fill_name(char *buf, int len, int nr)
{
	if (card_label[nr] != NULL) {
		snprintf(buf, len, card_label[nr]);
	} else {
		snprintf(buf, len, "Dummy video device (0x%04X)", nr);
	}
}
```

### 修复函数（原文）

```cpp
static void vidioc_fill_name(char *buf, int len, int nr)
{
	if (card_label[nr] != NULL) {
		snprintf(buf, len, "%s", card_label[nr]);
	} else {
		snprintf(buf, len, "Dummy video device (0x%04X)", nr);
	}
}
```

### 实际修改行的导出倍率（Seed123456）

| 侧 | 物理行 | Token质量 | Context倍率 | RGCL倍率 | Context归一化质量 | RGCL归一化质量 |
|---|---:|---:|---:|---:|---:|---:|
| 漏洞 | 4 | 12 | 1.0000000 | 5.0000000 | 17.9104% | 32.2371% |
| 修复 | 4 | 15 | 1.0000000 | 5.0000000 | 21.4286% | 45.7753% |

## primevul_paired:test:275 — 排序正确仍不等于两侧判对：权限上下文的细粒度替换

项目 `linux`；修复提交 `a3727a8bac0a9e77c70820655fd8715523ba3db7`。成员 `primevul_paired:test:551` / `primevul_paired:test:552`；原始标签1/0。

源码Token数 68/68；可见输入相同：False。两侧可见性 full / full；可见Diff Token质量 16.0/16.0。

实际Diff行（1起）：漏洞侧 [4]；修复侧 [4]。

补丁把 avc_has_perm 的第一个 SID 实参从 task_sid_subj(parent) 改成 task_sid_obj(parent)。可直接确认的是权限查询所用的 SID 来源改变；仅靠该函数片段无法核实两个辅助函数的完整策略语义和漏洞触发条件，因此不进一步断言根因。三个种子中，两种模型均为 P-V；RGCL 三个种子都满足漏洞概率高于修复概率，但没有一个 P-C。该例只支持“相对排序可能正确而安全版本仍误报”，不能用作完整漏洞机制证明。

| 种子 | Context漏洞概率 | Context修复概率 | Context Δp / 状态 | RGCL漏洞概率 | RGCL修复概率 | RGCL Δp / 状态 | Δp变化 |
|---:|---:|---:|---|---:|---:|---|---:|
| 42 | 0.996830165 | 0.992942333 | +0.00388783216 / P-V | 0.99998343 | 0.999974132 | +9.29832458e-06 / P-V | -0.00387853384 |
| 1234 | 0.537759006 | 0.53742826 | +0.000330746174 / P-V | 0.997705758 | 0.988748312 | +0.00895744562 / P-V | +0.00862669945 |
| 123456 | 0.998424768 | 0.996914744 | +0.00151002407 / P-V | 0.99995625 | 0.999926686 | +2.95639038e-05 / P-V | -0.00148046017 |

### 实际Diff

```diff
--- vulnerable
+++ repaired
@@ -1,6 +1,6 @@
 static int selinux_ptrace_traceme(struct task_struct *parent)
 {
 	return avc_has_perm(&selinux_state,
-			    task_sid_subj(parent), task_sid_obj(current),
+			    task_sid_obj(parent), task_sid_obj(current),
 			    SECCLASS_PROCESS, PROCESS__PTRACE, NULL);
 }
```

### 漏洞函数（原文）

```cpp
static int selinux_ptrace_traceme(struct task_struct *parent)
{
	return avc_has_perm(&selinux_state,
			    task_sid_subj(parent), task_sid_obj(current),
			    SECCLASS_PROCESS, PROCESS__PTRACE, NULL);
}
```

### 修复函数（原文）

```cpp
static int selinux_ptrace_traceme(struct task_struct *parent)
{
	return avc_has_perm(&selinux_state,
			    task_sid_obj(parent), task_sid_obj(current),
			    SECCLASS_PROCESS, PROCESS__PTRACE, NULL);
}
```

### 实际修改行的导出倍率（Seed123456）

| 侧 | 物理行 | Token质量 | Context倍率 | RGCL倍率 | Context归一化质量 | RGCL归一化质量 |
|---|---:|---:|---:|---:|---:|---:|
| 漏洞 | 4 | 16 | 1.0000000 | 3.9011836 | 26.6667% | 39.2620% |
| 修复 | 4 | 16 | 1.0000000 | 4.9368272 | 26.6667% | 44.8465% |

## primevul_paired:test:189 — 没有可见输入差异：修改发生在512 Token预算之外

项目 `tensorflow`；修复提交 `c79ba87153ee343401dbe9d1954d7f79e521eb14`。成员 `primevul_paired:test:379` / `primevul_paired:test:380`；原始标签1/0。

源码Token数 727/734；可见输入相同：True。两侧可见性 invisible / invisible；可见Diff Token质量 0.0/0.0。

实际Diff行（1起）：漏洞侧 [51]；修复侧 [51]。

补丁在调用 c->Dim(input, in_idx) 前增加负向索引边界检查 `in_idx <= -rank`。实际修改位于第51个物理行，而两侧前510个源码 Token 只到第42行；两个完整源码不同，但正式输入 Token ID 完全相同。所有种子、两种模型的成对概率均严格相等，间隔为0。即使 RGCL 将两侧共同概率升高或降低，也不能据此获得 P-C。此例的分类限制来自当前可见输入，不应归因于模型没有关注第51行。

| 种子 | Context漏洞概率 | Context修复概率 | Context Δp / 状态 | RGCL漏洞概率 | RGCL修复概率 | RGCL Δp / 状态 | Δp变化 |
|---:|---:|---:|---|---:|---:|---|---:|
| 42 | 0.801099479 | 0.801099479 | +0 / P-V | 0.952820182 | 0.952820182 | +0 / P-V | +0 |
| 1234 | 0.484397948 | 0.484397948 | +0 / P-B | 0.360023409 | 0.360023409 | +0 / P-B | +0 |
| 123456 | 0.0420556329 | 0.0420556329 | +0 / P-B | 0.815747678 | 0.815747678 | +0 / P-V | +0 |

### 实际Diff

```diff
--- vulnerable
+++ repaired
@@ -48,7 +48,7 @@
 
     for (int32_t i = 0; i < rank; ++i) {
       int64_t in_idx = data[i];
-      if (in_idx >= rank) {
+      if (in_idx >= rank || in_idx <= -rank) {
         return errors::InvalidArgument("perm dim ", in_idx,
                                        " is out of range of input rank ", rank);
       }
```

### 漏洞函数（原文）

```cpp
Status TransposeShapeFn(InferenceContext* c) {
  ShapeHandle input = c->input(0);
  ShapeHandle perm_shape = c->input(1);
  const Tensor* perm = c->input_tensor(1);
  DimensionHandle perm_elems = c->NumElements(perm_shape);
  // If we don't have rank information on the input or value information on
  // perm we can't return any shape information, otherwise we have enough
  // information to at least find the rank of the output.
  if (!c->RankKnown(input) && !c->ValueKnown(perm_elems) && perm == nullptr) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }

  // Find our value of the rank.
  int64_t rank;
  if (c->RankKnown(input)) {
    rank = c->Rank(input);
  } else if (c->ValueKnown(perm_elems)) {
    rank = c->Value(perm_elems);
  } else {
    rank = perm->NumElements();
  }
  if (!c->RankKnown(input) && rank < 2) {
    // A permutation array containing a single element is ambiguous. It could
    // indicate either a scalar or a 1-dimensional array, both of which the
    // transpose op returns unchanged.
    c->set_output(0, input);
    return Status::OK();
  }

  std::vector<DimensionHandle> dims;
  dims.resize(rank);
  TF_RETURN_IF_ERROR(c->WithRank(input, rank, &input));
  // Ensure that perm is a vector and has rank elements.
  TF_RETURN_IF_ERROR(c->WithRank(perm_shape, 1, &perm_shape));
  TF_RETURN_IF_ERROR(c->WithValue(perm_elems, rank, &perm_elems));

  // If we know the rank of the input and the value of perm, we can return
  // all shape information, otherwise we can only return rank information,
  // but no information for the dimensions.
  if (perm != nullptr) {
    std::vector<int64_t> data;
    if (perm->dtype() == DT_INT32) {
      data = AsInt64<int32>(perm, rank);
    } else {
      data = AsInt64<int64_t>(perm, rank);
    }

    for (int32_t i = 0; i < rank; ++i) {
      int64_t in_idx = data[i];
      if (in_idx >= rank) {
        return errors::InvalidArgument("perm dim ", in_idx,
                                       " is out of range of input rank ", rank);
      }
      dims[i] = c->Dim(input, in_idx);
    }
  } else {
    for (int i = 0; i < rank; ++i) {
      dims[i] = c->UnknownDim();
    }
  }

  c->set_output(0, c->MakeShape(dims));
  return Status::OK();
}
```

### 修复函数（原文）

```cpp
Status TransposeShapeFn(InferenceContext* c) {
  ShapeHandle input = c->input(0);
  ShapeHandle perm_shape = c->input(1);
  const Tensor* perm = c->input_tensor(1);
  DimensionHandle perm_elems = c->NumElements(perm_shape);
  // If we don't have rank information on the input or value information on
  // perm we can't return any shape information, otherwise we have enough
  // information to at least find the rank of the output.
  if (!c->RankKnown(input) && !c->ValueKnown(perm_elems) && perm == nullptr) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }

  // Find our value of the rank.
  int64_t rank;
  if (c->RankKnown(input)) {
    rank = c->Rank(input);
  } else if (c->ValueKnown(perm_elems)) {
    rank = c->Value(perm_elems);
  } else {
    rank = perm->NumElements();
  }
  if (!c->RankKnown(input) && rank < 2) {
    // A permutation array containing a single element is ambiguous. It could
    // indicate either a scalar or a 1-dimensional array, both of which the
    // transpose op returns unchanged.
    c->set_output(0, input);
    return Status::OK();
  }

  std::vector<DimensionHandle> dims;
  dims.resize(rank);
  TF_RETURN_IF_ERROR(c->WithRank(input, rank, &input));
  // Ensure that perm is a vector and has rank elements.
  TF_RETURN_IF_ERROR(c->WithRank(perm_shape, 1, &perm_shape));
  TF_RETURN_IF_ERROR(c->WithValue(perm_elems, rank, &perm_elems));

  // If we know the rank of the input and the value of perm, we can return
  // all shape information, otherwise we can only return rank information,
  // but no information for the dimensions.
  if (perm != nullptr) {
    std::vector<int64_t> data;
    if (perm->dtype() == DT_INT32) {
      data = AsInt64<int32>(perm, rank);
    } else {
      data = AsInt64<int64_t>(perm, rank);
    }

    for (int32_t i = 0; i < rank; ++i) {
      int64_t in_idx = data[i];
      if (in_idx >= rank || in_idx <= -rank) {
        return errors::InvalidArgument("perm dim ", in_idx,
                                       " is out of range of input rank ", rank);
      }
      dims[i] = c->Dim(input, in_idx);
    }
  } else {
    for (int i = 0; i < rank; ++i) {
      dims[i] = c->UnknownDim();
    }
  }

  c->set_output(0, c->MakeShape(dims));
  return Status::OK();
}
```

### 实际修改行的导出倍率（Seed123456）

| 侧 | 物理行 | Token质量 | Context倍率 | RGCL倍率 | Context归一化质量 | RGCL归一化质量 |
|---|---:|---:|---:|---:|---:|---:|
| 两侧 | — | — | — | — | — | — |

修改行均不可见，没有该行的注意力导出值。
