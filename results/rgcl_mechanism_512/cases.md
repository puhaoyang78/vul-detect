# RQ3 典型配对分类案例

复用既定Test预测与原始双版本源码。四例分别说明P-V→P-C、P-B→P-C、P-C→P-V和P-C→P-B；优先复用上一轮已有案例，并选择同类转移在多个种子重复出现、代码修改语义可核实的实例。候选按转移类型、出现该转移的种子数和完整pair_key排序，未按函数长度、可见性或修改规模筛选。

所有案例都展示三个种子；它们用于说明分类行为，不能用案例频率推断总体规律。表中概率显示9位有效数字，判定始终使用原始概率严格>0.5。源码Diff来自原文，保留上下文，展示用unified Diff；修改行标签另以正式双向_line_diff核对。Test Diff仅作事后解释。

## primevul_paired:test:444：P-V→P-C：修复版本误报得到纠正

项目 `php-src`；修复提交 `b6fe458ef9ac1372b60c3d3810b0358e2e20840d`。原始成员 `primevul_paired:test:889` / `primevul_paired:test:890`，标签1/0。

原函数直接读取p[0]到p[3]；补丁新增 `if (end - p < 4) return 0;`，在剩余数据不足四字节时提前返回。Seed42/1234中，漏洞函数保持判为漏洞，修复函数由漏洞改判正常，形成P-V→P-C；Seed123456保持P-C。这是一项明确的修复侧分类改善，不能仅凭输出变化推断模型关注了检查行。

| 种子 | Context漏洞概率 | Context修复概率 | RGCL漏洞概率 | RGCL修复概率 | 分类转移 |
|---:|---:|---:|---:|---:|---|
| 42 | 0.999740064 | 0.999940991 | 0.99982965 | 2.01063335e-06 | P-V→P-C |
| 1234 | 0.549814105 | 0.506472468 | 0.999916911 | 5.97527219e-07 | P-V→P-C |
| 123456 | 0.999992132 | 0.000766857527 | 0.999926567 | 4.03800833e-08 | P-C→P-C |

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

## primevul_paired:test:319：P-B→P-C：原本漏检的漏洞函数被识别

项目 `vim`；修复提交 `fe6fb267e6ee5c5da2f41889e4e0e0ac5bf4b89d`。原始成员 `primevul_paired:test:639` / `primevul_paired:test:640`，标签1/0。

Vim的eval7处理嵌套表达式，并调用递归的eval1/eval_func路径。补丁新增静态递归计数器，在达到1000层时返回错误，进入受保护部分后递增，并在函数末尾递减，限制递归深度。三个种子都由P-B变为P-C：漏洞函数由正常改判漏洞，修复函数仍判正常。Seed42中修复概率也升高至0.489171，但没有越过固定阈值；因此该例的新增P-C来自纠正漏洞漏检，并非要求修复概率必须下降。

| 种子 | Context漏洞概率 | Context修复概率 | RGCL漏洞概率 | RGCL修复概率 | 分类转移 |
|---:|---:|---:|---:|---:|---|
| 42 | 0.313798308 | 0.000622959516 | 0.556807935 | 0.489170581 | P-B→P-C |
| 1234 | 0.49412477 | 0.482980609 | 0.503023088 | 0.056137789 | P-B→P-C |
| 123456 | 0.484041423 | 0.000390241155 | 0.579386473 | 0.00829960499 | P-B→P-C |

```diff
--- vulnerable
+++ repaired
@@ -12,6 +12,7 @@
     char_u	*start_leader, *end_leader;
     int		ret = OK;
     char_u	*alias;
+    static	int recurse = 0;
 
     /*
      * Initialise variable so that clear_tv() can't mistake this for a
@@ -37,6 +38,15 @@
 	++*arg;
 	return FAIL;
     }
+
+    // Limit recursion to 1000 levels.  At least at 10000 we run out of stack
+    // and crash.
+    if (recurse == 1000)
+    {
+	semsg(_(e_expression_too_recursive_str), *arg);
+	return FAIL;
+    }
+    ++recurse;
 
     switch (**arg)
     {
@@ -267,5 +277,7 @@
      */
     if (ret == OK && evaluate && end_leader > start_leader)
 	ret = eval7_leader(rettv, FALSE, start_leader, &end_leader);
+
+    --recurse;
     return ret;
 }
```

## primevul_paired:test:318：P-C→P-V：引入修复版本误报

项目 `pjproject`；修复提交 `8b621f192cae14456ee0b0ade52ce6c6f258af1e`。原始成员 `primevul_paired:test:637` / `primevul_paired:test:638`，标签1/0。

RTCP BYE的reason.slen原本只受目标缓冲区大小和报文声明长度约束，随后从pkt+9复制。补丁新增 `reason.slen = PJ_MIN(reason.slen, size-9);`，进一步限制实际报文剩余长度。Seed1234/123456由P-C退化为P-V：漏洞侧仍判漏洞，修复侧由正常误判为漏洞；Seed42保持P-C。这是新增误报，不能把补丁具有清晰安全含义或预测置信度提高当作修复识别成功。

| 种子 | Context漏洞概率 | Context修复概率 | RGCL漏洞概率 | RGCL修复概率 | 分类转移 |
|---:|---:|---:|---:|---:|---|
| 42 | 0.988277256 | 0.00134633598 | 0.999221087 | 8.92218741e-05 | P-C→P-C |
| 1234 | 0.512158573 | 0.486103654 | 0.999100566 | 0.985971868 | P-C→P-V |
| 123456 | 0.999941349 | 0.005650518 | 0.993143082 | 0.997135758 | P-C→P-V |

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

## primevul_paired:test:504：P-C→P-B：引入漏洞函数漏检

项目 `linux`；修复提交 `fa40d9734a57bcbfa79a280189799f76c88f7bb0`。原始成员 `primevul_paired:test:1009` / `primevul_paired:test:1010`，标签1/0。

TIPC密钥接收函数先按size分配内存，再按报文提供的keylen执行memcpy，原有大小一致性检查发生在复制之后。补丁将最小报文长度、keylen上界及size一致性验证移到分配和复制之前，并调整解锁出口。Seed42/1234由P-C变为P-B：漏洞函数由漏洞漏判为正常，修复函数仍为正常；Seed123456保持P-B。这说明所选RGCL模型也会丢失原模型已经正确的漏洞识别；不能从概率变化确定它如何处理了检查与复制的顺序。

| 种子 | Context漏洞概率 | Context修复概率 | RGCL漏洞概率 | RGCL修复概率 | 分类转移 |
|---:|---:|---:|---:|---:|---|
| 42 | 0.54737711 | 0.428200036 | 0.0838745013 | 0.0117012169 | P-C→P-B |
| 1234 | 0.50627321 | 0.473601937 | 0.254814088 | 0.309584588 | P-C→P-B |
| 123456 | 0.480903238 | 0.000831316283 | 0.226897463 | 0.1671478 | P-B→P-B |

```diff
--- vulnerable
+++ repaired
@@ -5,33 +5,42 @@
 	u16 key_gen = msg_key_gen(hdr);
 	u16 size = msg_data_sz(hdr);
 	u8 *data = msg_data(hdr);
+	unsigned int keylen;
+
+	/* Verify whether the size can exist in the packet */
+	if (unlikely(size < sizeof(struct tipc_aead_key) + TIPC_AEAD_KEYLEN_MIN)) {
+		pr_debug("%s: message data size is too small\n", rx->name);
+		goto exit;
+	}
+
+	keylen = ntohl(*((__be32 *)(data + TIPC_AEAD_ALG_NAME)));
+
+	/* Verify the supplied size values */
+	if (unlikely(size != keylen + sizeof(struct tipc_aead_key) ||
+		     keylen > TIPC_AEAD_KEY_SIZE_MAX)) {
+		pr_debug("%s: invalid MSG_CRYPTO key size\n", rx->name);
+		goto exit;
+	}
 
 	spin_lock(&rx->lock);
 	if (unlikely(rx->skey || (key_gen == rx->key_gen && rx->key.keys))) {
 		pr_err("%s: key existed <%p>, gen %d vs %d\n", rx->name,
 		       rx->skey, key_gen, rx->key_gen);
-		goto exit;
+		goto exit_unlock;
 	}
 
 	/* Allocate memory for the key */
 	skey = kmalloc(size, GFP_ATOMIC);
 	if (unlikely(!skey)) {
 		pr_err("%s: unable to allocate memory for skey\n", rx->name);
-		goto exit;
+		goto exit_unlock;
 	}
 
 	/* Copy key from msg data */
-	skey->keylen = ntohl(*((__be32 *)(data + TIPC_AEAD_ALG_NAME)));
+	skey->keylen = keylen;
 	memcpy(skey->alg_name, data, TIPC_AEAD_ALG_NAME);
 	memcpy(skey->key, data + TIPC_AEAD_ALG_NAME + sizeof(__be32),
 	       skey->keylen);
-
-	/* Sanity check */
-	if (unlikely(size != tipc_aead_key_size(skey))) {
-		kfree(skey);
-		skey = NULL;
-		goto exit;
-	}
 
 	rx->key_gen = key_gen;
 	rx->skey_mode = msg_key_mode(hdr);
@@ -39,9 +48,10 @@
 	rx->nokey = 0;
 	mb(); /* for nokey flag */
 
-exit:
+exit_unlock:
 	spin_unlock(&rx->lock);
 
+exit:
 	/* Schedule the key attaching on this crypto */
 	if (likely(skey && queue_delayed_work(tx->wq, &rx->work, 0)))
 		return true;
```
