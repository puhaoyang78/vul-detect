# Fixed source review samples

1-based display lines; stored positions are 0-based. Review is by this assistant, not independent gold annotation.

## primevul_paired:valid:179:boundary_check_relation:2594:2463:enclosing_branch
target=1, pair=primevul_paired:valid:89
{"pattern": "boundary_check_relation", "operation": {"start_byte": 2594, "end_byte": 2607, "lines": [114], "code": "region[count]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 2463, "end_byte": 2484, "lines": [110], "code": "(count < max_regions)", "ast_type": "parenthesized_expression"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
  108: 		return -FDT_ERR_BADLAYOUT;
  109: 
  110: 	/* Add a region for the END tag and the string table */
  111: 	if (count < max_regions) {
  112: 		region[count].offset = base + start;
  113: 		region[count].size = nextoffset - start;
  114: 		if (add_string_tab)
  115: 			region[count].size += fdt_size_dt_strings(fdt);
  116: 	}
  117: 	count++;
  118: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -13,4 +13,5 @@
 	int want = 0;
 	int base = fdt_off_dt_struct(fdt);
+	bool expect_end = false;
 
 	end = path;
@@ -28,4 +29,8 @@
 		tag = fdt_next_tag(fdt, offset, &nextoffset);
 		stop_at = nextoffset;
+
+		/* If we see two root nodes, something is wrong */
+		if (expect_end && tag != FDT_END)
+			return -FDT_ERR_BADLAYOUT;
 
 		switch (tag) {
@@ -51,4 +56,8 @@
 				return -FDT_ERR_BADSTRUCTURE;
 			name = fdt_get_name(fdt, offset, &len);
+
+			/* The root node must have an empty name */
+			if (!depth && *name)
+				return -FDT_ERR_BADLAYOUT;
 			if (end - path + 2 + len >= path_len)
 				return -FDT_ERR_NOSPACE;
@@ -78,4 +87,6 @@
 				;
 			*end = '\0';
+			if (depth == -1)
+				expect_end = true;
 			break;
 
```

## primevul_paired:valid:1053:boundary_check_relation:534:244:enclosing_branch
target=1, pair=primevul_paired:valid:526
{"pattern": "boundary_check_relation", "operation": {"start_byte": 534, "end_byte": 547, "lines": [19], "code": "rp[2*m->size]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 244, "end_byte": 264, "lines": [7], "code": "(nlimbs > 2*m->size)", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
    5: 
    6:   mpn_set_base256_le (rp, nlimbs, digest, digest_size);
    7: 
    8:   if (nlimbs > 2*m->size)
    9:     {
   10:       /* Special case for Ed448: reduce rp to 2*m->size limbs.
   11: 	 After decoding rp from a hash of size 2*rn:
   17: 	 rp = r1' || r0
   18: 
   19: 	 where r1' has m->size limbs.  */
   20:       mp_limb_t hi = rp[2*m->size];
   21:       assert (nlimbs == 2*m->size + 1);
   22: 
   23:       hi = mpn_addmul_1 (rp + m->size, m->B, m->size, hi);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -3,4 +3,5 @@
 {
   mp_size_t nlimbs = (8*digest_size + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;
+  mp_limb_t cy;
 
   mpn_set_base256_le (rp, nlimbs, digest, digest_size);
@@ -26,4 +27,7 @@
       assert (hi == 0);
     }
-  m->mod (m, rp, rp);
+  m->mod (m, rp + m->size , rp);
+  /* Ensure canonical reduction. */
+  cy = mpn_sub_n (rp, rp + m->size, m->m, m->size);
+  cnd_copy (cy, rp, rp + m->size, m->size);
 }
```

## primevul_paired:valid:404:boundary_check_relation:373:347:loop_condition
target=0, pair=primevul_paired:valid:201
{"pattern": "boundary_check_relation", "operation": {"start_byte": 373, "end_byte": 392, "lines": [18], "code": "image.colorTable[i]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 347, "end_byte": 357, "lines": [17], "code": "i < number", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   15: 		return 0;
   16: 	}
   17: 	number = image.ncolors;
   18: 	for(i = 0; i < number; i++) {
   19: 		if (!image.colorTable[i].c_color) {
   20: 			goto done;
   21: 		}
   22: 	}
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -15,4 +15,10 @@
 		return 0;
 	}
+	number = image.ncolors;
+	for(i = 0; i < number; i++) {
+		if (!image.colorTable[i].c_color) {
+			goto done;
+		}
+	}
 
 	if (!(im = gdImageCreate(image.width, image.height))) {
@@ -20,5 +26,4 @@
 	}
 
-	number = image.ncolors;
 	colors = (int *) safe_emalloc(number, sizeof(int), 0);
 	for (i = 0; i < number; i++) {
```

## primevul_paired:valid:350:boundary_check_relation:1486:1464:loop_condition
target=0, pair=primevul_paired:valid:174
{"pattern": "boundary_check_relation", "operation": {"start_byte": 1486, "end_byte": 1490, "lines": [57], "code": "p[j]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 1464, "end_byte": 1469, "lines": [56], "code": "j < i", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   54: 	if (unlikely(slab_want_init_on_alloc(flags, s))) {
   55: 		int j;
   56: 
   57: 		for (j = 0; j < i; j++)
   58: 			memset(p[j], 0, s->object_size);
   59: 	}
   60: 
   61: 	/* memcg and kmem_cache debug support */
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -21,4 +21,13 @@
 
 		if (unlikely(!object)) {
+			/*
+			 * We may have removed an object from c->freelist using
+			 * the fastpath in the previous iteration; in that case,
+			 * c->tid has not been bumped yet.
+			 * Since ___slab_alloc() may reenable interrupts while
+			 * allocating memory, we should bump c->tid now.
+			 */
+			c->tid = next_tid(c->tid);
+
 			/*
 			 * Invoking slow path likely have side-effect
```

## primevul_paired:valid:163:null_check_relation:5349:5286:enclosing_branch
target=1, pair=primevul_paired:valid:81
{"pattern": "null_check_relation", "operation": {"start_byte": 5349, "end_byte": 5372, "lines": [116], "code": "input_backprop->flat<T>", "ast_type": "field_expression"}, "evidence": {"start_byte": 5286, "end_byte": 5313, "lines": [115], "code": "(input_backprop != nullptr)", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true, "zero_case": "excluded"}
```c
  113:       }
  114:     }
  115: 
  116:     if (input_backprop != nullptr) {
  117:       auto input_backprop_flat = input_backprop->flat<T>();
  118:       auto out_arg_max_flat = output_arg_max->flat<int64>();
  119:       auto out_backprop_flat = out_backprop.flat<T>();
  120: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -139,5 +139,7 @@
         // in_end)
         FastBoundsCheck(input_backprop_index - in_start, in_end - in_start);
-        input_backprop_flat(input_backprop_index) += out_backprop_flat(index);
+        if (index < out_backprop.NumElements()) {
+          input_backprop_flat(input_backprop_index) += out_backprop_flat(index);
+        }
       }
     }
```

## primevul_paired:valid:847:null_check_relation:1295:387:early_return
target=1, pair=primevul_paired:valid:423
{"pattern": "null_check_relation", "operation": {"start_byte": 1295, "end_byte": 1306, "lines": [37], "code": "fcstats->fc", "ast_type": "field_expression"}, "evidence": {"start_byte": 387, "end_byte": 404, "lines": [12], "code": "(fcstats == NULL)", "ast_type": "condition_clause"}, "role": "early_return", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false, "zero_case": "excluded"}
```c
   10: 	unsigned long   flags;
   11: 
   12: 	fcstats = kzalloc(sizeof(union bfa_port_stats_u), GFP_KERNEL);
   13: 	if (fcstats == NULL)
   14: 		return NULL;
   15: 
   16: 	hstats = &bfad->link_stats;
   35: 	hstats->nos_count = fcstats->fc.nos_count;
   36: 	hstats->error_frames = fcstats->fc.error_frames;
   37: 	hstats->dumped_frames = fcstats->fc.dropped_frames;
   38: 	hstats->link_failure_count = fcstats->fc.link_failures;
   39: 	hstats->loss_of_sync_count = fcstats->fc.loss_of_syncs;
   40: 	hstats->loss_of_signal_count = fcstats->fc.loss_of_signals;
   41: 	hstats->prim_seq_protocol_err_count = fcstats->fc.primseq_errs;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -21,6 +21,8 @@
 				fcstats, bfad_hcb_comp, &fcomp);
 	spin_unlock_irqrestore(&bfad->bfad_lock, flags);
-	if (rc != BFA_STATUS_OK)
+	if (rc != BFA_STATUS_OK) {
+		kfree(fcstats);
 		return NULL;
+	}
 
 	wait_for_completion(&fcomp.comp);
```

## primevul_paired:valid:56:null_check_relation:2840:2686:enclosing_branch
target=0, pair=primevul_paired:valid:27
{"pattern": "null_check_relation", "operation": {"start_byte": 2840, "end_byte": 2856, "lines": [107], "code": "pin_obj->content", "ast_type": "field_expression"}, "evidence": {"start_byte": 2686, "end_byte": 2723, "lines": [103], "code": "(!pin_obj || !pin_obj->content.value)", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false, "zero_case": "excluded"}
```c
  101: 			}
  102: 		}
  103: 
  104: 		if (!pin_obj || !pin_obj->content.value)    {
  105: 			rv = SC_ERROR_SECURITY_STATUS_NOT_SATISFIED;
  106: 		}
  107: 		else    {
  108: 			rv = sc_pkcs15_verify_pin(p15card, pin_obj, pin_obj->content.value, pin_obj->content.len);
  109: 			if (!rv)
  110: 				rv = sc_oberthur_read_file(p15card, in_path, out, out_len, 0);
  111: 		}
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -74,5 +74,5 @@
 		struct sc_pkcs15_object *objs[0x10], *pin_obj = NULL;
 		const struct sc_acl_entry *acl = sc_file_get_acl_entry(file, SC_AC_OP_READ);
-		int ii;
+		int ii, nobjs;
 
 		if (acl == NULL) {
@@ -83,16 +83,17 @@
 		}
 
-		rv = sc_pkcs15_get_objects(p15card, SC_PKCS15_TYPE_AUTH_PIN, objs, 0x10);
-		if (rv != SC_SUCCESS) {
+		nobjs = sc_pkcs15_get_objects(p15card, SC_PKCS15_TYPE_AUTH_PIN, objs, 0x10);
+		if (nobjs < 1) {
 			sc_file_free(file);
 			free(*out);
 			*out = NULL;
-			LOG_TEST_RET(ctx, rv, "Cannot read oberthur file: get AUTH objects error");
+			LOG_TEST_RET(ctx, SC_ERROR_DATA_OBJECT_NOT_FOUND,
+				"Cannot read oberthur file: get AUTH objects error");
 		}
 
-		for (ii=0; ii<rv; ii++)   {
+		for (ii = 0; ii < nobjs; ii++) {
 			struct sc_pkcs15_auth_info *auth_info = (struct sc_pkcs15_auth_info *) objs[ii]->data;
 			sc_log(ctx, "compare PIN/ACL refs:%i/%i, method:%i/%i",
-					auth_info->attrs.pin.reference, acl->key_ref, auth_info->auth_method, acl->method);
+				auth_info->attrs.pin.reference, acl->key_ref, auth_info->auth_method, acl->method);
 			if (auth_info->attrs.pin.reference == (int)acl->key_ref && auth_info->auth_method == (unsigned)acl->method)   {
 				pin_obj = objs[ii];
```

## primevul_paired:valid:1046:null_check_relation:241:99:early_return
target=0, pair=primevul_paired:valid:522
{"pattern": "null_check_relation", "operation": {"start_byte": 241, "end_byte": 257, "lines": [9], "code": "session->version", "ast_type": "field_expression"}, "evidence": {"start_byte": 99, "end_byte": 116, "lines": [3], "code": "(session == NULL)", "ast_type": "parenthesized_expression"}, "role": "early_return", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false, "zero_case": "excluded"}
```c
    1: int ssh_bind_accept_fd(ssh_bind sshbind, ssh_session session, socket_t fd){
    2:     int i, rc;
    3: 
    4:     if (session == NULL){
    5:         ssh_set_error(sshbind, SSH_FATAL,"session is null");
    6:         return SSH_ERROR;
    7:     }
    8: 
    9:     session->server = 1;
   10:     session->version = 2;
   11: 
   12:     /* copy options */
   13:     for (i = 0; i < 10; ++i) {
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -76,4 +76,6 @@
         }
     }
+    /* force PRNG to change state in case we fork after ssh_bind_accept */
+    ssh_reseed();
     return SSH_OK;
 }
```

## primevul_paired:valid:263:integer_range_relation:3863:2025:enclosing_branch
target=1, pair=primevul_paired:valid:131
{"pattern": "integer_range_relation", "operation": {"start_byte": 3863, "end_byte": 3893, "lines": [89], "code": "LOG(INFO) << \"ndims=\" << ndims", "ast_type": "binary_expression"}, "evidence": {"start_byte": 2025, "end_byte": 2037, "lines": [42], "code": "(ndims <= 1)", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false}
```c
   40:     const float max_z_value = output_range;
   41: 
   42:     const int ndims = bcast.x_reshape().size();
   43:     if (ndims <= 1) {
   44:       if (x.NumElements() == 1) {
   45:         ScalarAddition<T, Toutput>(context, y_data, min_y, max_y,
   46:                                    y.NumElements(), x_data[0], min_x, max_x,
   87:           tensor_min, tensor_max, tensor_num_elements, min_z_value, max_z_value,
   88:           z_data);
   89:     } else {
   90:       LOG(INFO) << "ndims=" << ndims;
   91:       LOG(INFO) << "bcast.x_reshape()="
   92:                 << TensorShape(bcast.x_reshape()).DebugString();
   93:       LOG(INFO) << "bcast.y_reshape()="
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -83,4 +83,6 @@
         tensor_max = max_x;
       }
+      OP_REQUIRES(context, vector_num_elements > 0,
+                  errors::InvalidArgument("Must have some elements to add"));
       VectorTensorAddition<T, Toutput>(
           vector_data, vector_min, vector_max, vector_num_elements, tensor_data,
```

## primevul_paired:valid:267:integer_range_relation:2396:2255:loop_condition
target=1, pair=primevul_paired:valid:133
{"pattern": "integer_range_relation", "operation": {"start_byte": 2396, "end_byte": 2420, "lines": [46], "code": "slice_index + num_split_", "ast_type": "binary_expression"}, "evidence": {"start_byte": 2255, "end_byte": 2279, "lines": [44], "code": "slice_index < num_split_", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   42:     OP_REQUIRES_OK(context, sparse::SparseTensor::Split<T>(
   43:                                 sparse_tensor, axis, num_split_, &outputs));
   44: 
   45:     for (int slice_index = 0; slice_index < num_split_; ++slice_index) {
   46:       context->set_output(slice_index, outputs[slice_index].indices());
   47:       context->set_output(slice_index + num_split_,
   48:                           outputs[slice_index].values());
   49:       Tensor* shape = nullptr;
   50:       OP_REQUIRES_OK(context, context->allocate_output(
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -33,9 +33,16 @@
                                         "), got ", num_split_));
 
+    // Prevent overflow by constructing the dense shape separately
+    TensorShape dense_shape;
+    const auto input_shape_flat = input_shape.flat<int64>();
+    for (int i = 0; i < input_shape.NumElements(); i++) {
+      OP_REQUIRES_OK(context,
+                     dense_shape.AddDimWithStatus(input_shape_flat(i)));
+    }
+
     sparse::SparseTensor sparse_tensor;
     OP_REQUIRES_OK(context,
-                   sparse::SparseTensor::Create(
-                       input_indices, input_values,
-                       TensorShape(input_shape.vec<int64>()), &sparse_tensor));
+                   sparse::SparseTensor::Create(input_indices, input_values,
+                                                dense_shape, &sparse_tensor));
 
     std::vector<sparse::SparseTensor> outputs;
```

## primevul_paired:valid:930:integer_range_relation:137:57:enclosing_branch
target=0, pair=primevul_paired:valid:464
{"pattern": "integer_range_relation", "operation": {"start_byte": 137, "end_byte": 144, "lines": [5], "code": "len - 1", "ast_type": "binary_expression"}, "evidence": {"start_byte": 57, "end_byte": 62, "lines": [2], "code": "(len)", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
    1: Pl_Count::write(unsigned char* buf, size_t len)
    2: {
    3:     if (len)
    4:     {
    5: 	this->m->count += QIntC::to_offset(len);
    6: 	this->m->last_char = buf[len - 1];
    7: 	getNext()->write(buf, len);
    8:     }
    9: }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -4,6 +4,6 @@
     {
 	this->m->count += QIntC::to_offset(len);
+	this->m->last_char = buf[len - 1];
 	getNext()->write(buf, len);
-	this->m->last_char = buf[len - 1];
     }
 }
```

## primevul_paired:valid:992:integer_range_relation:1332:1293:early_return
target=0, pair=primevul_paired:valid:495
{"pattern": "integer_range_relation", "operation": {"start_byte": 1332, "end_byte": 1340, "lines": [53], "code": "tmp << 4", "ast_type": "binary_expression"}, "evidence": {"start_byte": 1293, "end_byte": 1302, "lines": [51], "code": "(tmp < 0)", "ast_type": "parenthesized_expression"}, "role": "early_return", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false}
```c
   49: 	if (!(cf.can_id & CAN_RTR_FLAG)) {
   50: 		for (i = 0; i < cf.can_dlc; i++) {
   51: 			tmp = hex_to_bin(*cmd++);
   52: 			if (tmp < 0)
   53: 				return;
   54: 			cf.data[i] = (tmp << 4);
   55: 			tmp = hex_to_bin(*cmd++);
   56: 			if (tmp < 0)
   57: 				return;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -7,5 +7,5 @@
 	char *cmd = sl->rbuff;
 
-	cf.can_id = 0;
+	memset(&cf, 0, sizeof(cf));
 
 	switch (*cmd) {
@@ -46,6 +46,4 @@
 		return;
 
-	*(u64 *) (&cf.data) = 0; /* clear payload */
-
 	/* RTR frames may have a dlc > 0 but they never have any data bytes */
 	if (!(cf.can_id & CAN_RTR_FLAG)) {
```

## primevul_paired:valid:851:memory_extent_relation:274:242:destination_capacity_definition
target=1, pair=primevul_paired:valid:425
{"pattern": "memory_extent_relation", "operation": {"start_byte": 274, "end_byte": 304, "lines": [8], "code": "memset (suffix, 0, Name::SIZE)", "ast_type": "call_expression"}, "evidence": {"start_byte": 242, "end_byte": 260, "lines": [7], "code": "suffix[Name::SIZE]", "ast_type": "array_declarator"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "Name :: SIZE", "capacity_dimensions": ["Name :: SIZE"], "capacity_unit": "elements", "capacity_bytes": "sizeof(suffix)", "access_length": "Name :: SIZE", "extent_comparison": "unknown"}
```c
    5:                                 " (truncated rule).");
    6:             
    7:         {
    8:             char suffix[Name::SIZE];
    9:             memset (suffix, 0, Name::SIZE);
   10:             Xdr::read<CharPtrIO> (ptr, std::min(size, Name::SIZE-1), suffix);
   11:             _suffix = std::string(suffix);
   12:         }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -6,6 +6,7 @@
             
         {
-            char suffix[Name::SIZE];
-            memset (suffix, 0, Name::SIZE);
+            // maximum length of string plus one byte for terminating NULL
+            char suffix[Name::SIZE+1];
+            memset (suffix, 0, Name::SIZE+1);
             Xdr::read<CharPtrIO> (ptr, std::min(size, Name::SIZE-1), suffix);
             _suffix = std::string(suffix);
```

## primevul_paired:valid:963:memory_extent_relation:1587:1183:destination_capacity_definition
target=1, pair=primevul_paired:valid:481
{"pattern": "memory_extent_relation", "operation": {"start_byte": 1587, "end_byte": 1647, "lines": [39], "code": "memcpy(nonce, params->read.IV.data, AEAD_IMPLICIT_DATA_SIZE)", "ast_type": "call_expression"}, "evidence": {"start_byte": 1183, "end_byte": 1199, "lines": [30], "code": "nonce[blocksize]", "ast_type": "array_declarator"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "blocksize", "capacity_dimensions": ["blocksize"], "capacity_unit": "elements", "capacity_bytes": "sizeof(nonce)", "access_length": "AEAD_IMPLICIT_DATA_SIZE", "extent_comparison": "unknown"}
```c
   28:        */
   29:       if (explicit_iv && _gnutls_auth_cipher_is_aead(&params->read.cipher_state))
   30:         {
   31:           uint8_t nonce[blocksize];
   32:           /* Values in AEAD are pretty fixed in TLS 1.2 for 128-bit block
   33:            */
   34:           if (params->read.IV.data == NULL || params->read.IV.size != 4)
   37:           if (ciphertext->size < tag_size+AEAD_EXPLICIT_DATA_SIZE)
   38:             return gnutls_assert_val(GNUTLS_E_UNEXPECTED_PACKET_LENGTH);
   39: 
   40:           memcpy(nonce, params->read.IV.data, AEAD_IMPLICIT_DATA_SIZE);
   41:           memcpy(&nonce[AEAD_IMPLICIT_DATA_SIZE], ciphertext->data, AEAD_EXPLICIT_DATA_SIZE);
   42:           
   43:           _gnutls_auth_cipher_setiv(&params->read.cipher_state, nonce, AEAD_EXPLICIT_DATA_SIZE+AEAD_IMPLICIT_DATA_SIZE);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -77,5 +77,5 @@
       break;
     case CIPHER_BLOCK:
-      if (ciphertext->size < MAX(blocksize, tag_size) || (ciphertext->size % blocksize != 0))
+      if (ciphertext->size < blocksize || (ciphertext->size % blocksize != 0))
         return gnutls_assert_val(GNUTLS_E_UNEXPECTED_PACKET_LENGTH);
 
@@ -89,11 +89,8 @@
           ciphertext->size -= blocksize;
           ciphertext->data += blocksize;
+        }
 
-          if (ciphertext->size == 0)
-            {
-              gnutls_assert ();
-              return GNUTLS_E_DECRYPTION_FAILED;
-            }
-        }
+      if (ciphertext->size < tag_size)
+        return gnutls_assert_val(GNUTLS_E_DECRYPTION_FAILED);
 
       /* we don't use the auth_cipher interface here, since
@@ -108,4 +105,5 @@
 
       pad = ciphertext->data[ciphertext->size - 1] + 1;   /* pad */
+
 
       if ((int) pad > (int) ciphertext->size - tag_size)
```

## primevul_paired:valid:288:memory_extent_relation:1028:391:destination_capacity_definition
target=0, pair=primevul_paired:valid:143
{"pattern": "memory_extent_relation", "operation": {"start_byte": 1028, "end_byte": 1076, "lines": [21], "code": "memcpy(salt, saltPrefix, sizeof(saltPrefix) - 1)", "ast_type": "call_expression"}, "evidence": {"start_byte": 391, "end_byte": 419, "lines": [7], "code": "salt[OT_PBKDF2_SALT_MAX_LEN]", "ast_type": "array_declarator"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "OT_PBKDF2_SALT_MAX_LEN", "capacity_dimensions": ["OT_PBKDF2_SALT_MAX_LEN"], "capacity_unit": "elements", "capacity_bytes": "sizeof(salt)", "access_length": "sizeof ( saltPrefix ) - 1", "extent_comparison": "unknown"}
```c
    5: {
    6:     otError    error        = OT_ERROR_NONE;
    7:     const char saltPrefix[] = "Thread";
    8:     uint8_t    salt[OT_PBKDF2_SALT_MAX_LEN];
    9:     uint16_t   saltLen = 0;
   10:     uint16_t   passphraseLen;
   11:     uint8_t    networkNameLen;
   19:                  error = OT_ERROR_INVALID_ARGS);
   20: 
   21:     memset(salt, 0, sizeof(salt));
   22:     memcpy(salt, saltPrefix, sizeof(saltPrefix) - 1);
   23:     saltLen += static_cast<uint16_t>(sizeof(saltPrefix) - 1);
   24: 
   25:     memcpy(salt + saltLen, aExtPanId.m8, sizeof(aExtPanId));
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -4,26 +4,31 @@
                                    Pskc &                    aPskc)
 {
-    otError     error      = OT_ERROR_NONE;
-    const char *saltPrefix = "Thread";
-    uint8_t     salt[OT_PBKDF2_SALT_MAX_LEN];
-    uint16_t    saltLen = 0;
+    otError    error        = OT_ERROR_NONE;
+    const char saltPrefix[] = "Thread";
+    uint8_t    salt[OT_PBKDF2_SALT_MAX_LEN];
+    uint16_t   saltLen = 0;
+    uint16_t   passphraseLen;
+    uint8_t    networkNameLen;
 
-    VerifyOrExit((strlen(aPassPhrase) >= OT_COMMISSIONING_PASSPHRASE_MIN_SIZE) &&
-                     (strlen(aPassPhrase) <= OT_COMMISSIONING_PASSPHRASE_MAX_SIZE) &&
-                     (strlen(aNetworkName) <= OT_NETWORK_NAME_MAX_SIZE),
+    passphraseLen  = static_cast<uint16_t>(strnlen(aPassPhrase, OT_COMMISSIONING_PASSPHRASE_MAX_SIZE + 1));
+    networkNameLen = static_cast<uint8_t>(strnlen(aNetworkName, OT_NETWORK_NAME_MAX_SIZE + 1));
+
+    VerifyOrExit((passphraseLen >= OT_COMMISSIONING_PASSPHRASE_MIN_SIZE) &&
+                     (passphraseLen <= OT_COMMISSIONING_PASSPHRASE_MAX_SIZE) &&
+                     (networkNameLen <= OT_NETWORK_NAME_MAX_SIZE),
                  error = OT_ERROR_INVALID_ARGS);
 
     memset(salt, 0, sizeof(salt));
-    memcpy(salt, saltPrefix, strlen(saltPrefix));
-    saltLen += static_cast<uint16_t>(strlen(saltPrefix));
+    memcpy(salt, saltPrefix, sizeof(saltPrefix) - 1);
+    saltLen += static_cast<uint16_t>(sizeof(saltPrefix) - 1);
 
     memcpy(salt + saltLen, aExtPanId.m8, sizeof(aExtPanId));
     saltLen += OT_EXT_PAN_ID_SIZE;
 
-    memcpy(salt + saltLen, aNetworkName, strlen(aNetworkName));
-    saltLen += static_cast<uint16_t>(strlen(aNetworkName));
+    memcpy(salt + saltLen, aNetworkName, networkNameLen);
+    saltLen += networkNameLen;
 
-    otPbkdf2Cmac(reinterpret_cast<const uint8_t *>(aPassPhrase), static_cast<uint16_t>(strlen(aPassPhrase)),
-                 reinterpret_cast<const uint8_t *>(salt), saltLen, 16384, OT_PSKC_MAX_SIZE, aPskc.m8);
+    otPbkdf2Cmac(reinterpret_cast<const uint8_t *>(aPassPhrase), passphraseLen, reinterpret_cast<const uint8_t *>(salt),
+                 saltLen, 16384, OT_PSKC_MAX_SIZE, aPskc.m8);
 
 exit:
```

## primevul_paired:valid:824:memory_extent_relation:4204:4104:destination_capacity_definition
target=0, pair=primevul_paired:valid:411
{"pattern": "memory_extent_relation", "operation": {"start_byte": 4204, "end_byte": 4242, "lines": [106], "code": "memcpy(cs_name, str->str, str->length)", "ast_type": "call_expression"}, "evidence": {"start_byte": 4104, "end_byte": 4129, "lines": [104], "code": "cs_name[CHARSET_NAME_LEN]", "ast_type": "array_declarator"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "CHARSET_NAME_LEN", "capacity_dimensions": ["CHARSET_NAME_LEN"], "capacity_unit": "elements", "capacity_bytes": "sizeof(cs_name)", "access_length": "str -> length", "extent_comparison": "unknown"}
```c
  102:                 if (set_charset && str->length < CHARSET_NAME_LEN &&
  103:                     strncmp(mysql->charset->csname, str->str, str->length) != 0)
  104:                 {
  105:                   char cs_name[CHARSET_NAME_LEN];
  106:                   const MARIADB_CHARSET_INFO *cs_info;
  107:                   memcpy(cs_name, str->str, str->length);
  108:                   cs_name[str->length]= 0;
  109:                   if ((cs_info = mysql_find_charset_name(cs_name)))
  110:                     mysql->charset= cs_info;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -1,4 +1,5 @@
 int ma_read_ok_packet(MYSQL *mysql, uchar *pos, ulong length)
 {
+  uchar *end= mysql->net.read_pos+length;
   size_t item_len;
   mysql->affected_rows= net_field_length_ll(&pos);
@@ -8,8 +9,12 @@
   mysql->warning_count=uint2korr(pos);
   pos+=2;
-  if (pos < mysql->net.read_pos+length)
+  if (pos > end)
+    goto corrupted;
+  if (pos < end)
   {
     if ((item_len= net_field_length(&pos)))
       mysql->info=(char*) pos;
+    if (pos + item_len > end)
+      goto corrupted;
 
     /* check if server supports session tracking */
@@ -22,5 +27,5 @@
       {
         int i;
-        if (pos < mysql->net.read_pos + length)
+        if (pos < end)
         {
           LIST *session_item;
@@ -28,5 +33,9 @@
           enum enum_session_state_type si_type;
           uchar *old_pos= pos;
-          size_t item_len= net_field_length(&pos);  /* length for all items */
+
+          item_len= net_field_length(&pos);  /* length for all items */
+          if (pos + item_len > end)
+            goto corrupted;
+          end= pos + item_len;
 
           /* length was already set, so make sure that info will be zero terminated */
@@ -34,9 +43,8 @@
             *old_pos= 0;
 
-          while (item_len > 0)
+          while (pos < end)
           {
             size_t plen;
             char *data;
-            old_pos= pos;
             si_type= (enum enum_session_state_type)net_field_length(&pos);
             switch(si_type) {
@@ -48,4 +56,6 @@
                 net_field_length(&pos); /* ignore total length, item length will follow next */
               plen= net_field_length(&pos);
+              if (pos + plen > end)
+                goto corrupted;
               if (!(session_item= ma_multi_malloc(0,
                                   &session_item, sizeof(LIST),
@@ -53,9 +63,5 @@
                                   &data, plen,
                                   NULL)))
-              {
-                ma_clear_session_state(mysql);
-                SET_CLIENT_ERROR(mysql, CR_OUT_OF_MEMORY, SQLSTATE_UNKNOWN, 0);
-                return -1;
-              }
+                  goto oom;
               str->length= plen;
               str->str= data;
@@ -80,4 +86,6 @@
... display truncated; source retained unchanged ...
```

## primevul_paired:valid:231:call_argument_relation:1371:611:computed_definition_to_argument
target=1, pair=primevul_paired:valid:115
{"pattern": "call_argument_relation", "operation": {"start_byte": 1371, "end_byte": 1524, "lines": [25, 26, 27], "code": "errors::InvalidArgument(\n            \"sparse matrix must have the same rank as permutation; got: \", rank,\n            \" != \", perm_shape.dims(), \" + 1.\")", "ast_type": "call_expression"}, "evidence": {"start_byte": 611, "end_byte": 641, "lines": [11], "code": "rank = dense_shape.dim_size(0)", "ast_type": "init_declarator"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 1}
```c
    9:                     " but saw dtype: ", DataTypeString(sparse_matrix.dtype())));
   10: 
   11:     const Tensor& dense_shape = sparse_matrix.dense_shape();
   12:     const int rank = dense_shape.dim_size(0);
   13:     OP_REQUIRES(ctx, rank == 2 || rank == 3,
   14:                 errors::InvalidArgument("sparse matrix must have rank 2 or 3; ",
   15:                                         "but dense_shape has size ", rank));
   23:     const TensorShape& perm_shape = permutation_indices.shape();
   24:     OP_REQUIRES(
   25:         ctx, perm_shape.dims() + 1 == rank,
   26:         errors::InvalidArgument(
   27:             "sparse matrix must have the same rank as permutation; got: ", rank,
   28:             " != ", perm_shape.dims(), " + 1."));
   29:     OP_REQUIRES(
   30:         ctx, perm_shape.dim_size(rank - 2) == *num_rows,
   31:         errors::InvalidArgument(
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -1,44 +1,43 @@
-  void ValidateInputs(OpKernelContext* ctx,
-                      const CSRSparseMatrix& sparse_matrix,
-                      const Tensor& permutation_indices, int* batch_size,
-                      int64* num_rows) {
-    OP_REQUIRES(ctx, sparse_matrix.dtype() == DataTypeToEnum<T>::value,
-                errors::InvalidArgument(
-                    "Asked for a CSRSparseMatrix of type ",
-                    DataTypeString(DataTypeToEnum<T>::value),
-                    " but saw dtype: ", DataTypeString(sparse_matrix.dtype())));
+  Status ValidateInputs(const CSRSparseMatrix& sparse_matrix,
+                        const Tensor& permutation_indices, int* batch_size,
+                        int64* num_rows) {
+    if (sparse_matrix.dtype() != DataTypeToEnum<T>::value)
+      return errors::InvalidArgument(
+          "Asked for a CSRSparseMatrix of type ",
+          DataTypeString(DataTypeToEnum<T>::value),
+          " but saw dtype: ", DataTypeString(sparse_matrix.dtype()));
 
     const Tensor& dense_shape = sparse_matrix.dense_shape();
     const int rank = dense_shape.dim_size(0);
-    OP_REQUIRES(ctx, rank == 2 || rank == 3,
-                errors::InvalidArgument("sparse matrix must have rank 2 or 3; ",
-                                        "but dense_shape has size ", rank));
+    if (rank < 2 || rank > 3)
+      return errors::InvalidArgument("sparse matrix must have rank 2 or 3; ",
+                                     "but dense_shape has size ", rank);
     const int row_dim = (rank == 2) ? 0 : 1;
     auto dense_shape_vec = dense_shape.vec<int64>();
     *num_rows = dense_shape_vec(row_dim);
     const int64 num_cols = dense_shape_vec(row_dim + 1);
-    OP_REQUIRES(ctx, *num_rows == num_cols,
-                errors::InvalidArgument("sparse matrix must be square; got: ",
-                                        *num_rows, " != ", num_cols));
+    if (*num_rows != num_cols)
+      return errors::InvalidArgument(
+          "sparse matrix must be square; got: ", *num_rows, " != ", num_cols);
     const TensorShape& perm_shape = permutation_indices.shape();
-    OP_REQUIRES(
-        ctx, perm_shape.dims() + 1 == rank,
-        errors::InvalidArgument(
-            "sparse matrix must have the same rank as permutation; got: ", rank,
-            " != ", perm_shape.dims(), " + 1."));
-    OP_REQUIRES(
-        ctx, perm_shape.dim_size(rank - 2) == *num_rows,
-        errors::InvalidArgument(
-            "permutation must have the same number of elements in each batch "
-            "as the number of rows in sparse matrix; got: ",
-            perm_shape.dim_size(rank - 2), " != ", *num_rows));
+    if (perm_shape.dims() + 1 != rank)
+      return errors::InvalidArgument(
+          "sparse matrix must have the same rank as permutation; got: ", rank,
+          " != ", perm_shape.dims(), " + 1.");
+    if (perm_shape.dim_size(rank - 2) != *num_rows)
+      return errors::InvalidArgument(
+          "permutation must have the same number of elements in each batch "
+          "as the number of rows in sparse matrix; got: ",
+          perm_shape.dim_size(rank - 2), " != ", *num_rows);
 
     *batch_size = sparse_matrix.batch_size();
     if (*batch_size > 1) {
-      OP_REQUIRES(
-          ctx, perm_shape.dim_size(0) == *batch_size,
-          errors::InvalidArgument("permutation must have the same batch size "
-                                  "as sparse matrix; got: ",
-                                  perm_shape.dim_size(0), " != ", *batch_size));
+      if (perm_shape.dim_size(0) != *batch_size)
+        return errors::InvalidArgument(
... display truncated; source retained unchanged ...
```

## primevul_paired:valid:329:call_argument_relation:1119:1026:computed_definition_to_argument
target=1, pair=primevul_paired:valid:164
{"pattern": "call_argument_relation", "operation": {"start_byte": 1119, "end_byte": 1187, "lines": [28], "code": "TF_LITE_ENSURE_EQ(context, input_height, output_height * block_size)", "ast_type": "call_expression"}, "evidence": {"start_byte": 1026, "end_byte": 1067, "lines": [25], "code": "output_height = input_height / block_size", "ast_type": "init_declarator"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 2}
```c
   23:   const int block_size = params->block_size;
   24:   const int input_height = input->dims->data[1];
   25:   const int input_width = input->dims->data[2];
   26:   int output_height = input_height / block_size;
   27:   int output_width = input_width / block_size;
   28: 
   29:   TF_LITE_ENSURE_EQ(context, input_height, output_height * block_size);
   30:   TF_LITE_ENSURE_EQ(context, input_width, output_width * block_size);
   31: 
   32:   TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -22,4 +22,5 @@
 
   const int block_size = params->block_size;
+  TF_LITE_ENSURE(context, block_size > 0);
   const int input_height = input->dims->data[1];
   const int input_width = input->dims->data[2];
```

## primevul_paired:valid:134:call_argument_relation:892:798:computed_definition_to_argument
target=0, pair=primevul_paired:valid:66
{"pattern": "call_argument_relation", "operation": {"start_byte": 892, "end_byte": 966, "lines": [37], "code": "av_log(s, AV_LOG_ERROR, \"number of video tracks %d is not 1\\n\", num_video)", "ast_type": "call_expression"}, "evidence": {"start_byte": 798, "end_byte": 821, "lines": [33], "code": "num_video = avio_r8(pb)", "ast_type": "assignment_expression"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 3}
```c
   31:     off += ffio_read_varlen(pb); // val_5
   32: 
   33:     avio_r8(pb); // '2'
   34:     num_video = avio_r8(pb);
   35: 
   36:     avio_seek(pb, off, SEEK_SET);
   37:     if (num_video != 1) {
   38:         av_log(s, AV_LOG_ERROR, "number of video tracks %d is not 1\n", num_video);
   39:         return AVERROR_PATCHWELCOME;
   40:     }
   41: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -102,5 +102,5 @@
         if (avio_tell(pb) < off) {
             int num_data;
-            int xd_size = 0;
+            int xd_size = 1;
             int data_len[256];
             int offset = 1;
@@ -116,8 +116,8 @@
                 }
                 data_len[j] = len;
-                xd_size += len;
+                xd_size += len + 1 + len/255;
             }
 
-            ret = ff_alloc_extradata(st->codecpar, 64 + xd_size + xd_size / 255);
+            ret = ff_alloc_extradata(st->codecpar, xd_size);
             if (ret < 0)
                 return ret;
@@ -128,7 +128,5 @@
             for (j = 0; j < num_data - 1; j++) {
                 unsigned delta = av_xiphlacing(&p[offset], data_len[j]);
-                if (delta > data_len[j]) {
-                    return AVERROR_INVALIDDATA;
-                }
+                av_assert0(delta <= xd_size - offset);
                 offset += delta;
             }
@@ -141,4 +139,5 @@
                     break;
                 }
+                av_assert0(data_len[j] <= xd_size - offset);
                 offset += data_len[j];
             }
```

## primevul_paired:valid:72:call_argument_relation:952:790:computed_definition_to_argument
target=0, pair=primevul_paired:valid:35
{"pattern": "call_argument_relation", "operation": {"start_byte": 952, "end_byte": 999, "lines": [21], "code": "untrusted_cache->Malloc(sgx_params->input_size)", "ast_type": "call_expression"}, "evidence": {"start_byte": 790, "end_byte": 835, "lines": [18], "code": "sgx_params->input_size = input->MessageSize()", "ast_type": "assignment_expression"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 0}
```c
   16:   sgx_params->input_size = 0;
   17:   sgx_params->input = nullptr;
   18:   if (input) {
   19:     sgx_params->input_size = input->MessageSize();
   20:     if (sgx_params->input_size > 0) {
   21:       // Allocate and copy data to |input_buffer|.
   22:       sgx_params->input = untrusted_cache->Malloc(sgx_params->input_size);
   23:       const void *input_pointer = sgx_params->input;
   24:       uint64_t input_size = sgx_params->input_size;
   25:       if (!TrustedPrimitives::IsOutsideEnclave(input_pointer, input_size)) {
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -37,6 +37,7 @@
     untrusted_cache->Free(const_cast<void *>(sgx_params->input));
   }
-  if (!TrustedPrimitives::IsOutsideEnclave(sgx_params->output,
-                                           sgx_params->output_size)) {
+  const void *output_pointer = sgx_params->output;
+  uint64_t output_size = sgx_params->output_size;
+  if (!TrustedPrimitives::IsOutsideEnclave(output_pointer, output_size)) {
     TrustedPrimitives::BestEffortAbort(
         "UntrustedCall: sgx_param output should be in untrusted memory");
@@ -45,5 +46,5 @@
     // For the results obtained in |output_buffer|, copy them to |output|
     // before freeing the buffer.
-    output->Deserialize(sgx_params->output, sgx_params->output_size);
+    output->Deserialize(output_pointer, output_size);
     TrustedPrimitives::UntrustedLocalFree(sgx_params->output);
   }
```

## primevul_paired:valid:57:control_call_relation:1048:1011:loop_condition
target=1, pair=primevul_paired:valid:28
{"pattern": "control_call_relation", "operation": {"start_byte": 1048, "end_byte": 1072, "lines": [25], "code": "float_output.dim_size(i)", "ast_type": "call_expression"}, "evidence": {"start_byte": 1011, "end_byte": 1020, "lines": [24], "code": "i < axis_", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   22:                                         "Dequantize with axis != -1."));
   23: 
   24:       int64 pre_dim = 1, post_dim = 1;
   25:       for (int i = 0; i < axis_; ++i) {
   26:         pre_dim *= float_output.dim_size(i);
   27:       }
   28:       for (int i = axis_ + 1; i < float_output.dims(); ++i) {
   29:         post_dim *= float_output.dim_size(i);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -8,4 +8,16 @@
       num_slices = input.dim_size(axis_);
     }
+    OP_REQUIRES(ctx, input_min_tensor.NumElements() == num_slices,
+                errors::InvalidArgument(
+                    "input_min_tensor must have as many elements as input on "
+                    "the dequantization axis (",
+                    axis_, "), got ", input_min_tensor.NumElements(),
+                    ", expected ", num_slices));
+    OP_REQUIRES(ctx, input_max_tensor.NumElements() == num_slices,
+                errors::InvalidArgument(
+                    "input_max_tensor must have as many elements as input on "
+                    "the dequantization axis (",
+                    axis_, "), got ", input_max_tensor.NumElements(),
+                    ", expected ", num_slices));
 
     Tensor* output = nullptr;
```

## primevul_paired:valid:435:control_call_relation:612:427:early_return
target=1, pair=primevul_paired:valid:217
{"pattern": "control_call_relation", "operation": {"start_byte": 612, "end_byte": 643, "lines": [24], "code": "nfc_sock_link(&raw_sk_list, sk)", "ast_type": "call_expression"}, "evidence": {"start_byte": 427, "end_byte": 432, "lines": [16], "code": "(!sk)", "ast_type": "parenthesized_expression"}, "role": "early_return", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false}
```c
   14: 		sock->ops = &rawsock_ops;
   15: 
   16: 	sk = sk_alloc(net, PF_NFC, GFP_ATOMIC, nfc_proto->proto, kern);
   17: 	if (!sk)
   18: 		return -ENOMEM;
   19: 
   20: 	sock_init_data(sock, sk);
   22: 	sk->sk_destruct = rawsock_destruct;
   23: 	sock->state = SS_UNCONNECTED;
   24: 	if (sock->type == SOCK_RAW)
   25: 		nfc_sock_link(&raw_sk_list, sk);
   26: 	else {
   27: 		INIT_WORK(&nfc_rawsock(sk)->tx_work, rawsock_tx_work);
   28: 		nfc_rawsock(sk)->tx_work_scheduled = false;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -9,8 +9,11 @@
 		return -ESOCKTNOSUPPORT;
 
-	if (sock->type == SOCK_RAW)
+	if (sock->type == SOCK_RAW) {
+		if (!capable(CAP_NET_RAW))
+			return -EPERM;
 		sock->ops = &rawsock_raw_ops;
-	else
+	} else {
 		sock->ops = &rawsock_ops;
+	}
 
 	sk = sk_alloc(net, PF_NFC, GFP_ATOMIC, nfc_proto->proto, kern);
```

## primevul_paired:valid:902:control_call_relation:5763:5630:early_return
target=0, pair=primevul_paired:valid:450
{"pattern": "control_call_relation", "operation": {"start_byte": 5763, "end_byte": 5781, "lines": [226], "code": "jpc_ms_destroy(ms)", "ast_type": "call_expression"}, "evidence": {"start_byte": 5630, "end_byte": 5664, "lines": [221], "code": "(jpc_ms_gettype(ms) != JPC_MS_EPH)", "ast_type": "parenthesized_expression"}, "role": "early_return", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false}
```c
  219: 				jas_eprintf("cannot get (EPH) marker segment\n");
  220: 				return -1;
  221: 			}
  222: 			if (jpc_ms_gettype(ms) != JPC_MS_EPH) {
  223: 				jpc_ms_destroy(ms);
  224: 				jas_eprintf("missing EPH marker segment\n");
  225: 				return -1;
  226: 			}
  227: 			jpc_ms_destroy(ms);
  228: 		}
  229: 	}
  230: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -138,4 +138,18 @@
 				mycounter = 0;
 				if (numnewpasses > 0) {
+					if (cblk->firstpassno > 10000) {
+						/* workaround for
+						   CVE-2016-9398: this
+						   large value would
+						   make
+						   JPC_SEGPASSCNT()
+						   return a negative
+						   value, causing an
+						   assertion failure
+						   in
+						   jpc_floorlog2() */
+						jpc_bitstream_close(inb);
+						return -1;
+					}
 					if ((m = jpc_getcommacode(inb)) < 0) {
 						jpc_bitstream_close(inb);
```

## primevul_paired:valid:200:control_call_relation:4065:3762:enclosing_branch
target=0, pair=primevul_paired:valid:99
{"pattern": "control_call_relation", "operation": {"start_byte": 4065, "end_byte": 4223, "lines": [106, 107, 108], "code": "define_label(output_ins.label,\n                                 output_ins.oprs[0].segment,\n                                 output_ins.oprs[0].offset, false)", "ast_type": "call_expression"}, "evidence": {"start_byte": 3762, "end_byte": 3781, "lines": [101], "code": "(!output_ins.label)", "ast_type": "parenthesized_expression"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false}
```c
   99: 
  100:             /*  forw_ref */
  101:             if (output_ins.opcode == I_EQU) {
  102:                 if (!output_ins.label) {
  103:                     nasm_error(ERR_NONFATAL, "EQU not preceded by label");
  104:                 } else if (output_ins.operands == 1 &&
  105:                            (output_ins.oprs[0].type & IMMEDIATE) &&
  106:                            output_ins.oprs[0].wrt == NO_SEG) {
  107:                     define_label(output_ins.label,
  108:                                  output_ins.oprs[0].segment,
  109:                                  output_ins.oprs[0].offset, false);
  110:                 } else if (output_ins.operands == 2
  111:                            && (output_ins.oprs[0].type & IMMEDIATE)
  112:                            && (output_ins.oprs[0].type & COLON)
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -100,11 +100,9 @@
             /*  forw_ref */
             if (output_ins.opcode == I_EQU) {
-                if (!output_ins.label)
-                    nasm_error(ERR_NONFATAL,
-                               "EQU not preceded by label");
-
-                if (output_ins.operands == 1 &&
-                    (output_ins.oprs[0].type & IMMEDIATE) &&
-                    output_ins.oprs[0].wrt == NO_SEG) {
+                if (!output_ins.label) {
+                    nasm_error(ERR_NONFATAL, "EQU not preceded by label");
+                } else if (output_ins.operands == 1 &&
+                           (output_ins.oprs[0].type & IMMEDIATE) &&
+                           output_ins.oprs[0].wrt == NO_SEG) {
                     define_label(output_ins.label,
                                  output_ins.oprs[0].segment,
```

## primevul_paired:valid:11:ok:no_candidate
target=1, pair=primevul_paired:valid:5
```c
    1:  void ServiceWorkerScriptContext::OnInstallEvent(
    2:      int active_version_embedded_worker_id) {
    3:   DidHandleInstallEvent(current_request_id_);
    4:  }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -1,4 +1,4 @@
  void ServiceWorkerScriptContext::OnInstallEvent(
      int active_version_embedded_worker_id) {
-  DidHandleInstallEvent(current_request_id_);
+  proxy_->dispatchInstallEvent(current_request_id_);
  }
```

## primevul_paired:valid:466:ok:no_candidate
target=0, pair=primevul_paired:valid:232
```c
    1: static void set_pixel_format(VncState *vs,
    2:                              int bits_per_pixel, int depth,
    3:                              int big_endian_flag, int true_color_flag,
    4:                              int red_max, int green_max, int blue_max,
    5:                              int red_shift, int green_shift, int blue_shift)
    6: {
    7:     if (!true_color_flag) {
    8:         vnc_client_error(vs);
    9:         return;
   10:     }
   11: 
   12:     switch (bits_per_pixel) {
   13:     case 8:
   14:     case 16:
   15:     case 32:
   16:         break;
   17:     default:
   18:         vnc_client_error(vs);
   19:         return;
   20:     }
   21: 
   22:     vs->client_pf.rmax = red_max ? red_max : 0xFF;
   23:     vs->client_pf.rbits = hweight_long(red_max);
   24:     vs->client_pf.rshift = red_shift;
   25:     vs->client_pf.rmask = red_max << red_shift;
   26:     vs->client_pf.gmax = green_max ? green_max : 0xFF;
   27:     vs->client_pf.gbits = hweight_long(green_max);
   28:     vs->client_pf.gshift = green_shift;
   29:     vs->client_pf.gmask = green_max << green_shift;
   30:     vs->client_pf.bmax = blue_max ? blue_max : 0xFF;
   31:     vs->client_pf.bbits = hweight_long(blue_max);
   32:     vs->client_pf.bshift = blue_shift;
   33:     vs->client_pf.bmask = blue_max << blue_shift;
   34:     vs->client_pf.bits_per_pixel = bits_per_pixel;
   35:     vs->client_pf.bytes_per_pixel = bits_per_pixel / 8;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -20,13 +20,13 @@
     }
 
-    vs->client_pf.rmax = red_max;
+    vs->client_pf.rmax = red_max ? red_max : 0xFF;
     vs->client_pf.rbits = hweight_long(red_max);
     vs->client_pf.rshift = red_shift;
     vs->client_pf.rmask = red_max << red_shift;
-    vs->client_pf.gmax = green_max;
+    vs->client_pf.gmax = green_max ? green_max : 0xFF;
     vs->client_pf.gbits = hweight_long(green_max);
     vs->client_pf.gshift = green_shift;
     vs->client_pf.gmask = green_max << green_shift;
-    vs->client_pf.bmax = blue_max;
+    vs->client_pf.bmax = blue_max ? blue_max : 0xFF;
     vs->client_pf.bbits = hweight_long(blue_max);
     vs->client_pf.bshift = blue_shift;
```

## primevul_paired:valid:9:parse_unknown:no_candidate
target=1, pair=primevul_paired:valid:4
```c
    1: AudioOutputStream* AudioManagerBase::MakeAudioOutputStreamProxy(
    2:     const AudioParameters& params) {
    3:   DCHECK(GetMessageLoop()->BelongsToCurrentThread());
    4: 
    5:   scoped_refptr<AudioOutputDispatcher>& dispatcher =
    6:       output_dispatchers_[params];
    7:    if (!dispatcher) {
    8:      base::TimeDelta close_delay =
    9:          base::TimeDelta::FromSeconds(kStreamCloseDelaySeconds);
   10: #if defined(OS_WIN) || defined(OS_MACOSX)
   11:      const CommandLine* cmd_line = CommandLine::ForCurrentProcess();
   12:     if (!cmd_line->HasSwitch(switches::kDisableAudioMixer)) {
   13:        dispatcher = new AudioOutputMixer(this, params, close_delay);
   14:     } else
   15: #endif
   16:     {
   17:        dispatcher = new AudioOutputDispatcherImpl(this, params, close_delay);
   18:      }
   19:    }
   20:   return new AudioOutputProxy(dispatcher);
   21: }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -8,11 +8,11 @@
      base::TimeDelta close_delay =
          base::TimeDelta::FromSeconds(kStreamCloseDelaySeconds);
-#if defined(OS_WIN) || defined(OS_MACOSX)
      const CommandLine* cmd_line = CommandLine::ForCurrentProcess();
-    if (!cmd_line->HasSwitch(switches::kDisableAudioMixer)) {
+    // TODO(dalecurtis): Browser side mixing has a couple issues that must be
+    // fixed before it can be turned on by default: http://crbug.com/138098 and
+    // http://crbug.com/140247
+    if (cmd_line->HasSwitch(switches::kEnableAudioMixer)) {
        dispatcher = new AudioOutputMixer(this, params, close_delay);
-    } else
-#endif
-    {
+    } else {
        dispatcher = new AudioOutputDispatcherImpl(this, params, close_delay);
      }
```

## primevul_paired:valid:670:parse_unknown:no_candidate
target=0, pair=primevul_paired:valid:334
```c
    1: static void gprinter_free(struct usb_function *f)
    2: {
    3: 	struct printer_dev *dev = func_to_printer(f);
    4: 	struct f_printer_opts *opts;
    5: 
    6: 	opts = container_of(f->fi, struct f_printer_opts, func_inst);
    7: 
    8: 	kref_put(&dev->kref, printer_dev_free);
    9: 	mutex_lock(&opts->lock);
   10: 	--opts->refcnt;
   11: 	mutex_unlock(&opts->lock);
   12: }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -5,5 +5,6 @@
 
 	opts = container_of(f->fi, struct f_printer_opts, func_inst);
-	kfree(dev);
+
+	kref_put(&dev->kref, printer_dev_free);
 	mutex_lock(&opts->lock);
 	--opts->refcnt;
```
