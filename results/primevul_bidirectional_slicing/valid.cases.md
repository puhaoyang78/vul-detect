# Fixed slicing source review

0-based stored / 1-based displayed physical rows. Assistant review, not independent vulnerability gold.

## primevul_paired:valid:175 / addition_only / legal
```diff
--- 
+++ 
@@ -5,4 +5,5 @@
     struct ofpact_encap *encap;
     const struct ofp_ed_prop_header *ofp_prop;
+    const size_t encap_ofs = out->size;
     size_t props_len;
     uint16_t n_props = 0;
@@ -32,4 +33,5 @@
         n_props++;
     }
+    encap = ofpbuf_at_assert(out, encap_ofs, sizeof *encap);
     encap->n_props = n_props;
     out->header = &encap->ofpact;
```
### primevul_paired:valid:351
```c
   22:     encap->hdr_size = ntohs(nae->hdr_size);
   23: 
   24:     ofp_prop = nae->props;
   25:     props_len = ntohs(nae->len) - offsetof(struct nx_action_encap, props);
   26:     n_props = 0;
   27:     while (props_len > 0) {
   28:         err = decode_ed_prop(&ofp_prop, out, &props_len);
```
{"review_id": "primevul_paired:valid:351:line:24", "sample_key": "primevul_paired:valid:351", "pair_key": "primevul_paired:valid:175", "line": 24, "target": 1, "shape": "addition_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["losver"], "code": "    props_len = ntohs(nae->len) - offsetof(struct nx_action_encap, props);"}
### primevul_paired:valid:352
```c
    4: {
    5:     struct ofpact_encap *encap;
    6:     const struct ofp_ed_prop_header *ofp_prop;
    7:     const size_t encap_ofs = out->size;
    8:     size_t props_len;
    9:     uint16_t n_props = 0;
   10:     int err;
   25:     ofp_prop = nae->props;
   26:     props_len = ntohs(nae->len) - offsetof(struct nx_action_encap, props);
   27:     n_props = 0;
   28:     while (props_len > 0) {
   29:         err = decode_ed_prop(&ofp_prop, out, &props_len);
   30:         if (err) {
   31:             return err;
```
{"review_id": "primevul_paired:valid:352:line:6", "sample_key": "primevul_paired:valid:352", "pair_key": "primevul_paired:valid:175", "line": 6, "target": 0, "shape": "addition_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff"], "code": "    const size_t encap_ofs = out->size;"}
{"review_id": "primevul_paired:valid:352:line:27", "sample_key": "primevul_paired:valid:352", "pair_key": "primevul_paired:valid:175", "line": 27, "target": 0, "shape": "addition_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["losver"], "code": "    while (props_len > 0) {"}

## primevul_paired:valid:107 / addition_only / legal
```diff
--- 
+++ 
@@ -55,4 +55,9 @@
     const Tensor* ckpt_path_t;
     OP_REQUIRES_OK(context, context->input("ckpt_path", &ckpt_path_t));
+    OP_REQUIRES(
+        context, ckpt_path_t->NumElements() == 1,
+        errors::InvalidArgument("The `ckpt_path` tensor must have exactly one "
+                                "element, got tensor of shape ",
+                                ckpt_path_t->shape().DebugString()));
     const string& ckpt_path = ckpt_path_t->scalar<tstring>()();
     const Tensor* old_tensor_name_t;
```
### primevul_paired:valid:215
```c
  203:         context, context->input("initializing_values", &initializing_values_t));
  204:     const auto initializing_values = initializing_values_t->flat<float>();
  205:     int64 initializing_values_index = 0;
  206:     for (int i = 0; i < num_rows_; ++i) {
  207:       for (int j = 0; j < num_cols_; ++j) {
  208:         if (row_id_present[i] && col_id_present[j]) continue;
  209:         OP_REQUIRES(
  210:             context, initializing_values_index < initializing_values.size(),
  211:             errors::InvalidArgument(
  212:                 "initializing_values contained ", initializing_values.size(),
  213:                 " elements, but more missing values remain."));
  214:         output_matrix(i, j) = initializing_values(initializing_values_index);
  215:         ++initializing_values_index;
```
{"review_id": "primevul_paired:valid:215:line:205", "sample_key": "primevul_paired:valid:215", "pair_key": "primevul_paired:valid:107", "line": 205, "target": 1, "shape": "addition_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["static_candidates"], "code": "    for (int i = 0; i < num_rows_; ++i) {"}
{"review_id": "primevul_paired:valid:215:line:211", "sample_key": "primevul_paired:valid:215", "pair_key": "primevul_paired:valid:107", "line": 211, "target": 1, "shape": "addition_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["losver"], "code": "                \"initializing_values contained \", initializing_values.size(),"}
### primevul_paired:valid:216
```c
   54:     // Processes the checkpoint source and the provided Tensor name.
   55:     const Tensor* ckpt_path_t;
   56:     OP_REQUIRES_OK(context, context->input("ckpt_path", &ckpt_path_t));
   57:     OP_REQUIRES(
   58:         context, ckpt_path_t->NumElements() == 1,
   59:         errors::InvalidArgument("The `ckpt_path` tensor must have exactly one "
   60:                                 "element, got tensor of shape ",
  143:       // Potentially re-allocates the tensor buffer since the last slice may
  144:       // have fewer rows than the other slices.
  145:       if (loaded_tensor_t.shape() != slice_shape) {
  146:         loaded_tensor_t = Tensor(DT_FLOAT, slice_shape);
  147:       }
  148:       OP_REQUIRES_OK(context, reader.LookupSlice(old_tensor_name, tensor_slice,
  149:                                                  &loaded_tensor_t));
  188:                       errors::Internal(strings::StrCat(
  189:                           "new_row=", new_row, " and new_col=", new_col,
  190:                           " should have been less than num_rows_=", num_rows_,
  191:                           " and num_cols_=", num_cols_,
  192:                           " and non-negative. This should never have happened "
  193:                           "if the code were correct. Please file a bug.")));
  194:           output_matrix(new_row, new_col) = loaded_tensor(row, old_col);
```
{"review_id": "primevul_paired:valid:216:line:56", "sample_key": "primevul_paired:valid:216", "pair_key": "primevul_paired:valid:107", "line": 56, "target": 0, "shape": "addition_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff", "losver"], "code": "    OP_REQUIRES("}
{"review_id": "primevul_paired:valid:216:line:145", "sample_key": "primevul_paired:valid:216", "pair_key": "primevul_paired:valid:107", "line": 145, "target": 0, "shape": "addition_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["static_candidates"], "code": "        loaded_tensor_t = Tensor(DT_FLOAT, slice_shape);"}
{"review_id": "primevul_paired:valid:216:line:190", "sample_key": "primevul_paired:valid:216", "pair_key": "primevul_paired:valid:107", "line": 190, "target": 0, "shape": "addition_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["losver"], "code": "                          \" and num_cols_=\", num_cols_,"}

## primevul_paired:valid:243 / replacement_only / legal
```diff
--- 
+++ 
@@ -416,9 +416,11 @@
       size_t time = 0;
       while ( disposals[i] == DelDispose ) {
-        time += curr->delay*1000/curr->ticks_per_second;
+        time +=(size_t) (curr->delay*1000*
+          PerceptibleReciprocal((double) curr->ticks_per_second));
         curr=GetNextImageInList(curr);
         i++;
       }
-      time += curr->delay*1000/curr->ticks_per_second;
+      time += (size_t)(curr->delay*1000*
+        PerceptibleReciprocal((double) curr->ticks_per_second));
       prev_image->ticks_per_second = 100L;
       prev_image->delay = time*prev_image->ticks_per_second/1000;
```
### primevul_paired:valid:487
```c
  419:         curr=GetNextImageInList(curr);
  420:         i++;
  421:       }
  422:       time += curr->delay*1000/curr->ticks_per_second;
  423:       prev_image->ticks_per_second = 100L;
  424:       prev_image->delay = time*prev_image->ticks_per_second/1000;
  425:     }
```
{"review_id": "primevul_paired:valid:487:line:421", "sample_key": "primevul_paired:valid:487", "pair_key": "primevul_paired:valid:243", "line": 421, "target": 1, "shape": "replacement_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff"], "code": "      time += curr->delay*1000/curr->ticks_per_second;"}
### primevul_paired:valid:488
```c
  420:         curr=GetNextImageInList(curr);
  421:         i++;
  422:       }
  423:       time += (size_t)(curr->delay*1000*
  424:         PerceptibleReciprocal((double) curr->ticks_per_second));
  425:       prev_image->ticks_per_second = 100L;
  426:       prev_image->delay = time*prev_image->ticks_per_second/1000;
```
{"review_id": "primevul_paired:valid:488:line:422", "sample_key": "primevul_paired:valid:488", "pair_key": "primevul_paired:valid:243", "line": 422, "target": 0, "shape": "replacement_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff"], "code": "      time += (size_t)(curr->delay*1000*"}

## primevul_paired:valid:167 / replacement_only / legal
```diff
--- 
+++ 
@@ -11,19 +11,23 @@
     if (len < devlen)
         return_error(gs_error_rangecheck);     /* not even room for device len */
-    memcpy((char *)pscratch->value.bytes, iodev->dname, devlen);
-    code = iodev->procs.enumerate_next(pfen, (char *)pscratch->value.bytes + devlen,
-                len - devlen);
-    if (code == ~(uint) 0) {    /* all done */
-        esp -= 5;               /* pop proc, pfen, devlen, iodev , mark */
-        return o_pop_estack;
-    } else if (code > len)      /* overran string */
-        return_error(gs_error_rangecheck);
-    else {
-        push(1);
-        ref_assign(op, pscratch);
-        r_set_size(op, code + devlen);
-        push_op_estack(file_continue);  /* come again */
-        *++esp = pscratch[2];   /* proc */
-        return o_push_estack;
-    }
+
+    do {
+        memcpy((char *)pscratch->value.bytes, iodev->dname, devlen);
+        code = iodev->procs.enumerate_next(pfen, (char *)pscratch->value.bytes + devlen,
+                    len - devlen);
+        if (code == ~(uint) 0) {    /* all done */
+            esp -= 5;               /* pop proc, pfen, devlen, iodev , mark */
+            return o_pop_estack;
+        } else if (code > len)      /* overran string */
+            return_error(gs_error_rangecheck);
+        else if (iodev != iodev_default(imemory)
+              || (check_file_permissions_reduced(i_ctx_p, (char *)pscratch->value.bytes, code + devlen, "PermitFileReading")) == 0) {
+            push(1);
+            ref_assign(op, pscratch);
+            r_set_size(op, code + devlen);
+            push_op_estack(file_continue);  /* come again */
+            *++esp = pscratch[2];   /* proc */
+            return o_push_estack;
+        }
+    } while(1);
 }
```
### primevul_paired:valid:335
```c
    1: file_continue(i_ctx_t *i_ctx_p)
    2: {
    3:     os_ptr op = osp;
    4:     es_ptr pscratch = esp - 2;
    5:     file_enum *pfen = r_ptr(esp - 1, file_enum);
    6:     int devlen = esp[-3].value.intval;
    7:     gx_io_device *iodev = r_ptr(esp - 4, gx_io_device);
   12:         return_error(gs_error_rangecheck);     /* not even room for device len */
   13:     memcpy((char *)pscratch->value.bytes, iodev->dname, devlen);
   14:     code = iodev->procs.enumerate_next(pfen, (char *)pscratch->value.bytes + devlen,
   15:                 len - devlen);
   16:     if (code == ~(uint) 0) {    /* all done */
   17:         esp -= 5;               /* pop proc, pfen, devlen, iodev , mark */
   18:         return o_pop_estack;
   19:     } else if (code > len)      /* overran string */
   20:         return_error(gs_error_rangecheck);
   21:     else {
   22:         push(1);
   23:         ref_assign(op, pscratch);
   24:         r_set_size(op, code + devlen);
```
{"review_id": "primevul_paired:valid:335:line:3", "sample_key": "primevul_paired:valid:335", "pair_key": "primevul_paired:valid:167", "line": 3, "target": 1, "shape": "replacement_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["static_candidates"], "code": "    es_ptr pscratch = esp - 2;"}
{"review_id": "primevul_paired:valid:335:line:14", "sample_key": "primevul_paired:valid:335", "pair_key": "primevul_paired:valid:167", "line": 14, "target": 1, "shape": "replacement_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff", "static_candidates", "losver"], "code": "                len - devlen);"}
{"review_id": "primevul_paired:valid:335:line:20", "sample_key": "primevul_paired:valid:335", "pair_key": "primevul_paired:valid:167", "line": 20, "target": 1, "shape": "replacement_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff"], "code": "    else {"}
### primevul_paired:valid:336
```c
   17:                     len - devlen);
   18:         if (code == ~(uint) 0) {    /* all done */
   19:             esp -= 5;               /* pop proc, pfen, devlen, iodev , mark */
   20:             return o_pop_estack;
   21:         } else if (code > len)      /* overran string */
   22:             return_error(gs_error_rangecheck);
   23:         else if (iodev != iodev_default(imemory)
   24:               || (check_file_permissions_reduced(i_ctx_p, (char *)pscratch->value.bytes, code + devlen, "PermitFileReading")) == 0) {
   25:             push(1);
   26:             ref_assign(op, pscratch);
   27:             r_set_size(op, code + devlen);
   28:             push_op_estack(file_continue);  /* come again */
   29:             *++esp = pscratch[2];   /* proc */
   30:             return o_push_estack;
```
{"review_id": "primevul_paired:valid:336:line:19", "sample_key": "primevul_paired:valid:336", "pair_key": "primevul_paired:valid:167", "line": 19, "target": 0, "shape": "replacement_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff"], "code": "            return o_pop_estack;"}
{"review_id": "primevul_paired:valid:336:line:20", "sample_key": "primevul_paired:valid:336", "pair_key": "primevul_paired:valid:167", "line": 20, "target": 0, "shape": "replacement_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff", "static_candidates", "losver"], "code": "        } else if (code > len)      /* overran string */"}
{"review_id": "primevul_paired:valid:336:line:26", "sample_key": "primevul_paired:valid:336", "pair_key": "primevul_paired:valid:167", "line": 26, "target": 0, "shape": "replacement_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff", "static_candidates"], "code": "            r_set_size(op, code + devlen);"}

## primevul_paired:valid:142 / deletion_only / legal
```diff
--- 
+++ 
@@ -8,10 +8,4 @@
   }
 
-  // Give user a chance to cancel navigation.
-  if (Emit("will-navigate", params.url))
-    return nullptr;
-
-  // Don't load the URL if the web contents was marked as destroyed from a
-  // will-navigate event listener
   if (IsDestroyed())
     return nullptr;
```
### primevul_paired:valid:285
```c
    1: content::WebContents* WebContents::OpenURLFromTab(
    2:     content::WebContents* source,
    3:     const content::OpenURLParams& params) {
    4:   if (params.disposition != WindowOpenDisposition::CURRENT_TAB) {
    5:     Emit("-new-window", params.url, "", params.disposition, "", params.referrer,
    6:          params.post_data);
    7:     return nullptr;
    8:   }
    9: 
   10:   // Give user a chance to cancel navigation.
   11:   if (Emit("will-navigate", params.url))
   12:     return nullptr;
   13: 
   14:   // Don't load the URL if the web contents was marked as destroyed from a
```
{"review_id": "primevul_paired:valid:285:line:3", "sample_key": "primevul_paired:valid:285", "pair_key": "primevul_paired:valid:142", "line": 3, "target": 1, "shape": "deletion_only", "slice_status": "ok", "quality_status": "legal", "methods": ["bidirectional_slice", "losver"], "code": "  if (params.disposition != WindowOpenDisposition::CURRENT_TAB) {"}
{"review_id": "primevul_paired:valid:285:line:10", "sample_key": "primevul_paired:valid:285", "pair_key": "primevul_paired:valid:142", "line": 10, "target": 1, "shape": "deletion_only", "slice_status": "ok", "quality_status": "legal", "methods": ["original_diff", "bidirectional_slice"], "code": "  if (Emit(\"will-navigate\", params.url))"}
### primevul_paired:valid:286
```c
    2:     content::WebContents* source,
    3:     const content::OpenURLParams& params) {
    4:   if (params.disposition != WindowOpenDisposition::CURRENT_TAB) {
    5:     Emit("-new-window", params.url, "", params.disposition, "", params.referrer,
    6:          params.post_data);
    7:     return nullptr;
    8:   }
```
{"review_id": "primevul_paired:valid:286:line:4", "sample_key": "primevul_paired:valid:286", "pair_key": "primevul_paired:valid:142", "line": 4, "target": 0, "shape": "deletion_only", "slice_status": "ok", "quality_status": "legal", "methods": ["bidirectional_slice", "losver"], "code": "    Emit(\"-new-window\", params.url, \"\", params.disposition, \"\", params.referrer,"}

## primevul_paired:valid:396 / deletion_only / legal
```diff
--- 
+++ 
@@ -19,9 +19,4 @@
 
 		goto nextline;
-
-	case IO_DISCONNECTED:
-		io_free(fs->io);
-		fs->io = NULL;
-		break;
 	}
 }
```
### primevul_paired:valid:793
```c
    1: filter_session_io(struct io *io, int evt, void *arg)
    2: {
    3: 	struct filter_session *fs = arg;
    4: 	char *line = NULL;
    5: 	ssize_t len;
    6: 
    7: 	log_trace(TRACE_IO, "filter session: %p: %s %s", fs, io_strevent(evt),
   19: 
   20: 		goto nextline;
   21: 
   22: 	case IO_DISCONNECTED:
   23: 		io_free(fs->io);
   24: 		fs->io = NULL;
   25: 		break;
```
{"review_id": "primevul_paired:valid:793:line:3", "sample_key": "primevul_paired:valid:793", "pair_key": "primevul_paired:valid:396", "line": 3, "target": 1, "shape": "deletion_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["losver"], "code": "\tchar *line = NULL;"}
{"review_id": "primevul_paired:valid:793:line:21", "sample_key": "primevul_paired:valid:793", "pair_key": "primevul_paired:valid:396", "line": 21, "target": 1, "shape": "deletion_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff"], "code": "\tcase IO_DISCONNECTED:"}
### primevul_paired:valid:794
```c
   12: 	nextline:
   13: 		line = io_getline(fs->io, &len);
   14: 		/* No complete line received */
   15: 		if (line == NULL)
   16: 			return;
   17: 
   18: 		filter_data(fs->id, line);
```
{"review_id": "primevul_paired:valid:794:line:14", "sample_key": "primevul_paired:valid:794", "pair_key": "primevul_paired:valid:396", "line": 14, "target": 0, "shape": "deletion_only", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["losver"], "code": "\t\tif (line == NULL)"}

## primevul_paired:valid:185 / mixed / legal
```diff
--- 
+++ 
@@ -327,12 +327,7 @@
 	p->pid = pid_nr(pid);
 	if (clone_flags & CLONE_THREAD) {
-		p->exit_signal = -1;
 		p->group_leader = current->group_leader;
 		p->tgid = current->tgid;
 	} else {
-		if (clone_flags & CLONE_PARENT)
-			p->exit_signal = current->group_leader->exit_signal;
-		else
-			p->exit_signal = args->exit_signal;
 		p->group_leader = p;
 		p->tgid = p->pid;
@@ -378,7 +373,12 @@
 		p->real_parent = current->real_parent;
 		p->parent_exec_id = current->parent_exec_id;
+		if (clone_flags & CLONE_THREAD)
+			p->exit_signal = -1;
+		else
+			p->exit_signal = current->group_leader->exit_signal;
 	} else {
 		p->real_parent = current;
 		p->parent_exec_id = current->self_exec_id;
+		p->exit_signal = args->exit_signal;
 	}
 
```
### primevul_paired:valid:371
```c
  123: 			task_rlimit(p, RLIMIT_NPROC)) {
  124: 		if (p->real_cred->user != INIT_USER &&
  125: 		    !capable(CAP_SYS_RESOURCE) && !capable(CAP_SYS_ADMIN))
  126: 			goto bad_fork_free;
  127: 	}
  128: 	current->flags &= ~PF_NPROC_EXCEEDED;
  129: 
  333: 		if (clone_flags & CLONE_PARENT)
  334: 			p->exit_signal = current->group_leader->exit_signal;
  335: 		else
  336: 			p->exit_signal = args->exit_signal;
  337: 		p->group_leader = p;
  338: 		p->tgid = p->pid;
  339: 	}
```
{"review_id": "primevul_paired:valid:371:line:125", "sample_key": "primevul_paired:valid:371", "pair_key": "primevul_paired:valid:185", "line": 125, "target": 1, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["losver"], "code": "\t\t\tgoto bad_fork_free;"}
{"review_id": "primevul_paired:valid:371:line:335", "sample_key": "primevul_paired:valid:371", "pair_key": "primevul_paired:valid:185", "line": 335, "target": 1, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff"], "code": "\t\t\tp->exit_signal = args->exit_signal;"}
### primevul_paired:valid:372
```c
  120: #endif
  121: 	retval = -EAGAIN;
  122: 	if (atomic_read(&p->real_cred->user->processes) >=
  123: 			task_rlimit(p, RLIMIT_NPROC)) {
  124: 		if (p->real_cred->user != INIT_USER &&
  125: 		    !capable(CAP_SYS_RESOURCE) && !capable(CAP_SYS_ADMIN))
  126: 			goto bad_fork_free;
  372: 	if (clone_flags & (CLONE_PARENT|CLONE_THREAD)) {
  373: 		p->real_parent = current->real_parent;
  374: 		p->parent_exec_id = current->parent_exec_id;
  375: 		if (clone_flags & CLONE_THREAD)
  376: 			p->exit_signal = -1;
  377: 		else
  378: 			p->exit_signal = current->group_leader->exit_signal;
```
{"review_id": "primevul_paired:valid:372:line:122", "sample_key": "primevul_paired:valid:372", "pair_key": "primevul_paired:valid:185", "line": 122, "target": 0, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["losver"], "code": "\t\t\ttask_rlimit(p, RLIMIT_NPROC)) {"}
{"review_id": "primevul_paired:valid:372:line:374", "sample_key": "primevul_paired:valid:372", "pair_key": "primevul_paired:valid:185", "line": 374, "target": 0, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff"], "code": "\t\tif (clone_flags & CLONE_THREAD)"}

## primevul_paired:valid:274 / mixed / legal
```diff
--- 
+++ 
@@ -1,3 +1,4 @@
-lyxml_parse_elem(struct ly_ctx *ctx, const char *data, unsigned int *len, struct lyxml_elem *parent, int options)
+lyxml_parse_elem(struct ly_ctx *ctx, const char *data, unsigned int *len, struct lyxml_elem *parent, int options,
+                 int bt_count)
 {
     const char *c = data, *start, *e;
@@ -14,4 +15,9 @@
     *len = 0;
 
+    if (bt_count > LY_RECURSION_LIMIT) {
+        LOGVAL(ctx, LYE_XML_INVAL, LY_VLOG_NONE, NULL, "Recursion limit %d reached", LY_RECURSION_LIMIT);
+        return NULL;
+    }
+
     if (*c != '<') {
         return NULL;
@@ -197,5 +203,5 @@
                     elem->flags |= LYXML_ELEM_MIXED;
                 }
-                child = lyxml_parse_elem(ctx, c, &size, elem, options);
+                child = lyxml_parse_elem(ctx, c, &size, elem, options, bt_count + 1);
                 if (!child) {
                     goto error;
```
### primevul_paired:valid:549
```c
   28:     }
   29:     e += size;
   30:     uc = lyxml_getutf8(ctx, e, &size);
   31:     while (is_xmlnamechar(uc)) {
   32:         if (*e == ':') {
   33:             if (prefix_len) {
   34:                 LOGVAL(ctx, LYE_XML_INVAL, LY_VLOG_NONE, NULL, "element name, multiple colons found");
  196:                     lyxml_add_child(ctx, elem, child);
  197:                     elem->flags |= LYXML_ELEM_MIXED;
  198:                 }
  199:                 child = lyxml_parse_elem(ctx, c, &size, elem, options);
  200:                 if (!child) {
  201:                     goto error;
  202:                 }
```
{"review_id": "primevul_paired:valid:549:line:30", "sample_key": "primevul_paired:valid:549", "pair_key": "primevul_paired:valid:274", "line": 30, "target": 1, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["losver"], "code": "    while (is_xmlnamechar(uc)) {"}
{"review_id": "primevul_paired:valid:549:line:198", "sample_key": "primevul_paired:valid:549", "pair_key": "primevul_paired:valid:274", "line": 198, "target": 1, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff"], "code": "                child = lyxml_parse_elem(ctx, c, &size, elem, options);"}
### primevul_paired:valid:550
```c
    1: lyxml_parse_elem(struct ly_ctx *ctx, const char *data, unsigned int *len, struct lyxml_elem *parent, int options,
    2:                  int bt_count)
    3: {
    4:     const char *c = data, *start, *e;
    5:     const char *lws;    /* leading white space for handling mixed content */
  128: 
  129:                 /* check that it corresponds to opening tag */
  130:                 size = e - c;
  131:                 str = malloc((size + 1) * sizeof *str);
  132:                 LY_CHECK_ERR_GOTO(!str, LOGMEM(ctx), error);
  133:                 memcpy(str, c, e - c);
  134:                 str[e - c] = '\0';
```
{"review_id": "primevul_paired:valid:550:line:1", "sample_key": "primevul_paired:valid:550", "pair_key": "primevul_paired:valid:274", "line": 1, "target": 0, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff"], "code": "                 int bt_count)"}
{"review_id": "primevul_paired:valid:550:line:130", "sample_key": "primevul_paired:valid:550", "pair_key": "primevul_paired:valid:274", "line": 130, "target": 0, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["losver"], "code": "                str = malloc((size + 1) * sizeof *str);"}