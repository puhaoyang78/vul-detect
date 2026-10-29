# Train 邻接代理标签抽查（40 个配对侧）

Seed 42，分层随机抽样：纯新增双侧16、单侧10、无候选6、编码相同4；纯删除补标3、无候选1。抽样先于模型结果。这里核验 Diff 对齐、物理行、可见性和邻接限制，未将代理行当作漏洞位置真值；不据此修改既定规则。

## 1. `primevul_paired:train:5759` · addition_both

补充物理行（1 起）：[19, 21]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  19     TypedData_Get_Struct(self, ffi_cif, &function_data_type, cif);
  20
  21     values = xcalloc((size_t)argc + 1, (size_t)sizeof(void *));
```

对侧同次 Diff 新增/删除片段：

```c
  21     if (rb_safe_level() >= 1) {
  22 	for (i = 0; i < argc; i++) {
  23 	    VALUE src = argv[i];
  24 	    if (OBJ_TAINTED(src)) {
  25 		rb_raise(rb_eSecurityError, "tainted parameter not allowed");
  26 	    }
  27 	}
  28     }
```

## 2. `primevul_paired:train:715` · addition_both

补充物理行（1 起）：[8, 9]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
   8  		ka->sa.sa_flags = 0;
   9  		sigemptyset(&ka->sa.sa_mask);
  10  		ka++;
```

对侧同次 Diff 新增/删除片段：

```c
   9 #ifdef SA_RESTORER
  10 		ka->sa.sa_restorer = NULL;
  11 #endif
```

## 3. `primevul_paired:train:145` · addition_both

补充物理行（1 起）：[7, 9]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
   7         int ret= -1,i,inl;
   8
   9         EVP_MD_CTX_init(&ctx);
  10         i=OBJ_obj2nid(a->algorithm);
```

对侧同次 Diff 新增/删除片段：

```c
   9        if (!pkey)
  10                {
  11                ASN1err(ASN1_F_ASN1_ITEM_VERIFY, ERR_R_PASSED_NULL_PARAMETER);
  12                return -1;
  13                }
```

## 4. `primevul_paired:train:7057` · addition_both

补充物理行（1 起）：[10, 12]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  10     sigpipe_ignore(conn->data, &pipe_st);
  11     /* This will remove the connection from the cache */
  12     (void)Curl_disconnect(conn, FALSE);
```

对侧同次 Diff 新增/删除片段：

```c
  11     conn->data->easy_conn = NULL; /* clear the easy handle's connection
  12                                      pointer */
```

## 5. `primevul_paired:train:2373` · addition_both

补充物理行（1 起）：[36, 37]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  36  		down_write(&mm->mmap_sem);
  37  		mutex_lock(&ufile->umap_lock);
  38  		list_for_each_entry_safe (priv, next_priv, &ufile->umaps,
  50  		mutex_unlock(&ufile->umap_lock);
  51  		up_write(&mm->mmap_sem);
  52  		mmput(mm);
```

对侧同次 Diff 新增/删除片段：

```c
  37 		if (!mmget_still_valid(mm))
  38 			goto skip_mm;
  53 	skip_mm:
```

## 6. `primevul_paired:train:1979` · addition_both

补充物理行（1 起）：[16, 17, 18]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  16          t1_getline();
  17          strcat(t1_buf_array, t1_line_array);
  18          strcpy(t1_line_array, t1_buf_array);
  19          t1_line_ptr = eol(t1_line_array);
```

对侧同次 Diff 新增/删除片段：

```c
  17         alloc_array(t1_buf, strlen(t1_line_array) + strlen(t1_buf_array) + 1, T1_BUF_SIZE);
  19         alloc_array(t1_line, strlen(t1_buf_array) + 1, T1_BUF_SIZE);
```

## 7. `primevul_paired:train:1611` · addition_both

补充物理行（1 起）：[19, 21]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  19  		sock_reset_flag(newsk, SOCK_RCU_FREE);
  20
  21  		newsk->sk_mark = inet_rsk(req)->ir_mark;
  22  		atomic64_set(&newsk->sk_cookie,
```

对侧同次 Diff 新增/删除片段：

```c
  21 		inet_sk(newsk)->mc_list = NULL;
```

## 8. `primevul_paired:train:863` · addition_both

补充物理行（1 起）：[39, 40]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  39  	i = port->index;
  40  	sync.clock_rate = FST_RDL(card, portConfig[i].lineSpeed);
  41  	/* Lucky card and linux use same encoding here */
```

对侧同次 Diff 新增/删除片段：

```c
  40 	memset(&sync, 0, sizeof(sync));
```

## 9. `primevul_paired:train:6951` · addition_both

补充物理行（1 起）：[10, 11]；原因：`supplemented`。

Differ 将重复 return 行作为新增块的一部分；编辑对齐符合原规则，语义对应仍不确定。

```c
  10 	if (copy_from_user(&trace, optval, sizeof(trace)))
  11 		return -EFAULT;
  12
```

对侧同次 Diff 新增/删除片段：

```c
  11 		return -EFAULT;
  13 	if (trace.rx_traces > RDS_MSG_RX_DGRAM_TRACE_MAX)
```

## 10. `primevul_paired:train:689` · addition_both

补充物理行（1 起）：[23, 24]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  23  	mnt->mnt.mnt_flags = old->mnt.mnt_flags & ~MNT_WRITE_HOLD;
  24  	atomic_inc(&sb->s_active);
  25  	mnt->mnt.mnt_sb = sb;
```

对侧同次 Diff 新增/删除片段：

```c
  24 	/* Don't allow unprivileged users to change mount flags */
  25 	if ((flag & CL_UNPRIVILEGED) && (mnt->mnt.mnt_flags & MNT_READONLY))
  26 		mnt->mnt.mnt_flags |= MNT_LOCK_READONLY;
```

## 11. `primevul_paired:train:6141` · addition_both

补充物理行（1 起）：[25, 26]；原因：`supplemented`。

Differ 将重复 goto/括号作为新增块的一部分；保留原标签，不以规则替代语义核验。

```c
  25         icmp6_send_error(m, ICMP6_TOOBIG, 0);
  26         goto bad;
  27     }
```

对侧同次 Diff 新增/删除片段：

```c
  26         goto bad;
  27     }
  29     // Check if the message size is big enough to hold what's
  30     // set in the payload length header. If not this is an invalid
  31     // packet
  32     if (m->m_len < ntohs(ip6->ip_pl) + sizeof(struct ip6)) {
```

## 12. `primevul_paired:train:8385` · addition_both

补充物理行（1 起）：[3, 10, 12]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
   2 {
   3         g_return_val_if_fail (a_this && PRIVATE (a_this)
   4                               && a_byte, CR_BAD_PARAM_ERROR);
  10                 return CR_END_OF_INPUT_ERROR;
  11
  12         *a_byte = PRIVATE (a_this)->in_buf[PRIVATE (a_this)->next_byte_index];
```

对侧同次 Diff 新增/删除片段：

```c
   3         gulong nb_bytes_left = 0;
  14         nb_bytes_left = cr_input_get_nb_bytes_left (a_this);
  16         if (nb_bytes_left < 1) {
  17                 return CR_END_OF_INPUT_ERROR;
  18         }
```

## 13. `primevul_paired:train:4779` · addition_both

补充物理行（1 起）：[15, 16]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  15     const int64_t i_read = stream_Size( p_stream );
  16     psz_update_data = malloc( i_read + 1 ); /* terminating '\0' */
  17     if( !psz_update_data )
```

对侧同次 Diff 新增/删除片段：

```c
  17     if( i_read < 0 || i_read >= UINT16_MAX)
  18     {
  19         msg_Err(p_update->p_libvlc, "Status file too large");
  20         goto error;
  21     }
```

## 14. `primevul_paired:train:611` · addition_both

补充物理行（1 起）：[4, 6]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
   4  	struct kvm_vcpu *vcpu, *v;
   5
   6  	vcpu = kvm_arch_vcpu_create(kvm, id);
   7  	if (IS_ERR(vcpu))
```

对侧同次 Diff 新增/删除片段：

```c
   6 	if (id >= KVM_MAX_VCPUS)
   7 		return -EINVAL;
```

## 15. `primevul_paired:train:5179` · addition_both

补充物理行（1 起）：[16, 18]；原因：`supplemented`。

新增测试调用；邻接测试注册/销毁行符合规则，但不是漏洞操作真值。

```c
  16    test_heap_overflow_vrend_renderer_transfer_write_iov();
  17
  18    virgl_renderer_context_destroy(ctx_id);
```

对侧同次 Diff 新增/删除片段：

```c
  17    test_heap_overflow_vrend_renderer_transfer_write_iov_compressed_tex();
```

## 16. `primevul_paired:train:3337` · addition_both

补充物理行（1 起）：[49, 50]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  49                              reply->writeInplace(sizeof(QueueBufferOutput)));
  50              status_t result = queueBuffer(buf, input, output);
  51              reply->writeInt32(result);
```

对侧同次 Diff 新增/删除片段：

```c
  50             memset(output, 0, sizeof(QueueBufferOutput));
```

## 17. `primevul_paired:train:583` · addition_single

补充物理行（1 起）：[21]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  21      ncomponents       = bytestream2_get_be16u(&s->g); // CSiz
  22
  23      if (ncomponents <= 0) {
  24          av_log(s->avctx, AV_LOG_ERROR, "Invalid number of components: %d\n",
```

对侧同次 Diff 新增/删除片段：

```c
  23     if (s->image_offset_x || s->image_offset_y) {
  24         avpriv_request_sample(s->avctx, "Support for image offsets");
  25         return AVERROR_PATCHWELCOME;
  26     }
```

## 18. `primevul_paired:train:559` · addition_single

补充物理行（1 起）：[30]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  28  		}
  29
  30  		err = bnep_add_connection(&ca, nsock);
```

对侧同次 Diff 新增/删除片段：

```c
  29 		ca.device[sizeof(ca.device)-1] = 0;
```

## 19. `primevul_paired:train:1831` · addition_single

补充物理行（1 起）：[1]；原因：`supplemented`。

邻接行是函数签名，新增部分也呈现签名形式；仅能确认文本邻接。

```c
   1 jas_matrix_t *jas_seq2d_create(int xstart, int ystart, int xend, int yend)
   2  {
   3  	jas_matrix_t *matrix;
```

对侧同次 Diff 新增/删除片段：

```c
   2 jas_matrix_t *jas_seq2d_create(jas_matind_t xstart, jas_matind_t ystart,
   3   jas_matind_t xend, jas_matind_t yend)
```

## 20. `primevul_paired:train:3327` · addition_single

补充物理行（1 起）：[10]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  10                  "from pid=%d, uid=%d\n", pid, uid);
  11      } else {
  12          mCore->dump(result, prefix);
```

对侧同次 Diff 新增/删除片段：

```c
  11         android_errorWriteWithInfoLog(0x534e4554, "27046057", uid, NULL, 0);
```

## 21. `primevul_paired:train:3499` · addition_single

补充物理行（1 起）：[8]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
   8          s->cert->shared_sigalgs = NULL;
   9      }
  10      /* Clear certificate digests and validity flags */
```

对侧同次 Diff 新增/删除片段：

```c
   9         s->cert->shared_sigalgslen = 0;
```

## 22. `primevul_paired:train:7241` · addition_single

补充物理行（1 起）：[19]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  19 	transport_check_allowed("git");
  20
  21 	/*
```

对侧同次 Diff 新增/删除片段：

```c
  20 	if (strchr(target_host, '\n') || strchr(path, '\n'))
  21 		die(_("newline is forbidden in git:// hosts and repo paths"));
```

## 23. `primevul_paired:train:549` · addition_single

补充物理行（1 起）：[6]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
   6  	int ret = 0;
   7
   8  #if XTENSA_HAVE_COPROCESSORS
   9  	/* Flush all coprocessors before we overwrite them. */
```

对侧同次 Diff 新增/删除片段：

```c
   8 	if (!access_ok(VERIFY_READ, uregs, sizeof(elf_xtregs_t)))
   9 		return -EFAULT;
```

## 24. `primevul_paired:train:8177` · addition_single

补充物理行（1 起）：[3]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
   3 	struct seq_oss_synth *rec;
   4
   5 	if (dp->synths[dev].is_midi) {
```

对侧同次 Diff 新增/删除片段：

```c
   5 	if (dev < 0 || dev >= dp->max_synthdev)
   6 		return -ENXIO;
```

## 25. `primevul_paired:train:3117` · addition_single

补充物理行（1 起）：[32]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  32      DCHECK(content);
  33    }
  34
```

对侧同次 Diff 新增/删除片段：

```c
  33     // The conversion of |obj| could have triggered arbitrary JavaScript code,
  34     // so check that the frame is still valid to avoid dereferencing a stale
  35     // pointer.
  36     if (frame != blink::WebLocalFrame::FrameForCurrentContext()) {
  37       NOTREACHED();
  38       return;
  39     }
```

## 26. `primevul_paired:train:7883` · addition_single

补充物理行（1 起）：[24]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  24 	CHECKCP(dlen);
  25 	if (type_to_fetch != T_ANY && type != type_to_fetch) {
  26 		cp += dlen;
 114 			cp += n;
 115 			n = dlen - n - 2;
 116 			CHECKCP(n);
```

对侧同次 Diff 新增/删除片段：

```c
  25 	if (dlen == 0) {
  26 		/* No data in the response - nothing to do */
  27 		return NULL;
  28 	}
 119 			if ( (size_t) dlen < ((size_t)n) + 2 ) {
 120 				return NULL;
 121 			}
```

## 27. `primevul_paired:train:6233` · addition_no_candidate

补充物理行（1 起）：[]；原因：`no_qualified_neighbour`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  45         }
  46         /* enforce key type */
  47         plist_get_data(key)->type = PLIST_KEY;
```

对侧同次 Diff 新增/删除片段：

```c
  47         if (plist_get_data(key)->type != PLIST_STRING) {
  48             fprintf(stderr, "ERROR: Malformed binary plist dict, invalid node type for key!\n");
  49             plist_free(node);
  50             return NULL;
  51         }
```

## 28. `primevul_paired:train:3137` · addition_no_candidate

补充物理行（1 起）：[]；原因：`no_qualified_neighbour`。

初始化项中的花括号触发保守边界，未跨过这些行补标。

```c
   7        {"googlePaymentsCached", IDS_SETTINGS_GOOGLE_PAYMENTS_CACHED},
   8        {"addresses", IDS_SETTINGS_AUTOFILL_ADDRESSES_HEADING},
   9        {"addAddressTitle", IDS_SETTINGS_AUTOFILL_ADDRESSES_ADD_TITLE},
  28        {"passwords", IDS_SETTINGS_PASSWORDS},
  29        {"passwordsAutosigninLabel",
  30         IDS_SETTINGS_PASSWORDS_AUTOSIGNIN_CHECKBOX_LABEL},
```

对侧同次 Diff 新增/删除片段：

```c
   8       {"autofillFormsLabel", IDS_SETTINGS_AUTOFILL_TOGGLE_LABEL},
  30       {"passwordsSavePasswordsLabel",
  31        IDS_SETTINGS_PASSWORDS_SAVE_PASSWORDS_TOGGLE_LABEL},
```

## 29. `primevul_paired:train:6887` · addition_no_candidate

补充物理行（1 起）：[]；原因：`no_qualified_neighbour`。

编辑两侧均含块边界；即使存在潜在相关语句，也不跨块扩展。

```c
   2 {
   3 	switch (var->bits_per_pixel) {
   4 	case 8:
```

对侧同次 Diff 新增/删除片段：

```c
   3 	if (!var->pixclock)
   4 		return -EINVAL;
```

## 30. `primevul_paired:train:4443` · addition_no_candidate

补充物理行（1 起）：[]；原因：`no_qualified_neighbour`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  43 		read_old_length(inp, ctb, &pktlen, &pktsize);
  44
  45 	pkt->pkttype = pkttype;
```

对侧同次 Diff 新增/删除片段：

```c
  45 	/* enforce limits to ensure that the following calculations
  46 	 * do not overflow */
  47 	if (pktlen >= MAX_PACKET_LEN || pktsize >= MAX_PACKET_LEN) {
  48 		_cdk_log_info("cdk_pkt_read: too long packet\n");
  49 		return gnutls_assert_val(CDK_Inv_Packet);
  50 	}
```

## 31. `primevul_paired:train:101` · addition_no_candidate

补充物理行（1 起）：[]；原因：`no_qualified_neighbour`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  47      }
  48      BN_free(a);
  49      BN_free(b);
```

对侧同次 Diff 新增/删除片段：

```c
  49     /* Regression test for carry propagation bug in sqr8x_reduction */
  50     BN_hex2bn(&a, "050505050505");
  51     BN_hex2bn(&b, "02");
  52     BN_hex2bn(&c,
  53         "4141414141414141414141274141414141414141414141414141414141414141"
  54         "4141414141414141414141414141414141414141414141414141414141414141"
  55         "4141414141414141414141800000000000000000000000000000000000000000"
  56         "0000000000000000000000000000000000000000000000000000000000000000"
  57         "0000000000000000000000000000000000000000000000000000000000000000"
  58         "0000000000000000000000000000000000000000000000000000000001");
  59     BN_mod_exp(d, a, b, c, ctx);
  60     BN_mul(e, a, a, ctx);
  61     if (BN_cmp(d, e)) {
  62         fprintf(stderr, "BN_mod_exp and BN_mul produce different results!\n");
  63         return 0;
  64     }
```

## 32. `primevul_paired:train:2523` · addition_no_candidate

补充物理行（1 起）：[]；原因：`no_qualified_neighbour`。

编辑两侧被块边界限制，未放宽规则以增加覆盖。

```c
  23
  24  	if (table->total == table->max) {
  25  		/* No free vlan entries */
```

对侧同次 Diff 新增/删除片段：

```c
  24 	if (free < 0) {
  25 		err = -ENOMEM;
  26 		goto out;
  27 	}
```

## 33. `primevul_paired:train:8003` · addition_identical

补充物理行（1 起）：[]；原因：`identical_input`。

编码相同，原样保留且不补标。

源码编辑位置：[242]；两侧当前 Token ID 完全相同。

## 34. `primevul_paired:train:5111` · addition_identical

补充物理行（1 起）：[]；原因：`identical_input`。

编码相同，原样保留且不补标。

源码编辑位置：[123, 132]；两侧当前 Token ID 完全相同。

## 35. `primevul_paired:train:3883` · addition_identical

补充物理行（1 起）：[]；原因：`identical_input`。

编码相同，原样保留且不补标。

源码编辑位置：[47, 81, 111]；两侧当前 Token ID 完全相同。

## 36. `primevul_paired:train:3027` · addition_identical

补充物理行（1 起）：[]；原因：`identical_input`。

编码相同，原样保留且不补标。

源码编辑位置：[205]；两侧当前 Token ID 完全相同。

## 37. `primevul_paired:train:1178` · deletion_supplemented

补充物理行（1 起）：[23]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  22  		if (index < 0 || index >= intern->llist->count) {
  23  			zend_throw_exception(spl_ce_OutOfRangeException, "Offset invalid or out of range", 0);
  24  			return;
```

对侧同次 Diff 新增/删除片段：

```c
  23 			zval_ptr_dtor(value);
```

## 38. `primevul_paired:train:1824` · deletion_supplemented

补充物理行（1 起）：[16, 25]；原因：`supplemented`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
  16      buf4 = (const z_crc_t FAR *)(const void FAR *)buf;
  17      while (len >= 32) {
  18          DOBIG32;
  24      }
  25      buf = (const unsigned char FAR *)buf4;
  26
```

对侧同次 Diff 新增/删除片段：

```c
  17     buf4--;
  26     buf4++;
```

## 39. `primevul_paired:train:2698` · deletion_supplemented

补充物理行（1 起）：[1]；原因：`supplemented`。

缺失侧仅剩一行函数签名，另一侧包含函数体；保留官方成员，只记录结构异常，不据此断言错配。

```c
   1 void DocumentWriter::setDecoder(TextResourceDecoder* decoder)
```

对侧同次 Diff 新增/删除片段：

```c
   2 {
   3     m_decoder = decoder;
   4 }
```

## 40. `primevul_paired:train:3144` · deletion_no_candidate

补充物理行（1 起）：[]；原因：`no_qualified_neighbour`。

源码、Diff 编辑位置与边界/可见性限制一致；语义相关性未作为正标签真值核实。

```c
   1 bool VaapiJpegDecoder::Initialize(const base::RepeatingClosure& error_uma_cb) {
```

对侧同次 Diff 新增/删除片段：

```c
   2   vaapi_wrapper_ = VaapiWrapper::Create(VaapiWrapper::kDecode,
   3                                         VAProfileJPEGBaseline, error_uma_cb);
   4   if (!vaapi_wrapper_) {
   5     VLOGF(1) << "Failed initializing VAAPI";
   6     return false;
   7   }
   8   return true;
   9 }
```
