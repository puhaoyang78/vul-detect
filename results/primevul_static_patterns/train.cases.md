# Fixed source review samples

1-based display lines; stored positions are 0-based. Review is by this assistant, not independent gold annotation.

## primevul_paired:train:2579:boundary_check_relation:6192:6137:loop_condition
target=1, pair=primevul_paired:train:1289
{"pattern": "boundary_check_relation", "operation": {"start_byte": 6192, "end_byte": 6211, "lines": [183], "code": "l2info->cookie[idx]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 6137, "end_byte": 6161, "lines": [182], "code": "idx < l2info->cookie_len", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
  180:                 ND_TCHECK2(p[0], l2info->cookie_len);
  181:                 if (ndo->ndo_eflag)
  182:                     ND_PRINT((ndo, ", cookie 0x"));
  183:                 for (idx = 0; idx < l2info->cookie_len; idx++) {
  184:                     l2info->cookie[idx] = p[idx]; /* copy cookie data */
  185:                     if (ndo->ndo_eflag) ND_PRINT((ndo, "%02x", p[idx]));
  186:                 }
  187:             }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -190,4 +190,5 @@
  
  
+            ND_TCHECK_16BITS(p+l2info->cookie_len);
              l2info->proto = EXTRACT_16BITS(p+l2info->cookie_len);
              break;
```

## primevul_paired:train:1395:boundary_check_relation:570:520:loop_condition
target=1, pair=primevul_paired:train:697
{"pattern": "boundary_check_relation", "operation": {"start_byte": 570, "end_byte": 588, "lines": [20], "code": "v->main_streams[i]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 520, "end_byte": 541, "lines": [19], "code": "i < v->n_main_streams", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   17:          * segment. */
   18:         if (v->ctx && v->ctx->nb_streams) {
   19:             v->needed = 0;
   20:             for (i = 0; i < v->n_main_streams; i++) {
   21:                 if (v->main_streams[i]->discard < AVDISCARD_ALL) {
   22:                     v->needed = 1;
   23:                     break;
   24:                 }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -5,4 +5,5 @@
      int ret, i;
      int just_opened = 0;
+    int reload_count = 0;
  
  restart:
@@ -36,4 +37,7 @@
  
  reload:
+        reload_count++;
+        if (reload_count > c->max_reload)
+            return AVERROR_EOF;
          if (!v->finished &&
              av_gettime_relative() - v->last_load_time >= reload_interval) {
```

## primevul_paired:train:5035:boundary_check_relation:247:189:enclosing_branch
target=1, pair=primevul_paired:train:2517
{"pattern": "boundary_check_relation", "operation": {"start_byte": 247, "end_byte": 257, "lines": [8], "code": "history[j]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 189, "end_byte": 212, "lines": [7], "code": "(history[j][0] != '\\0')", "ast_type": "parenthesized_expression"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
    5:     }
    6: 
    7:     for (int j = 0; j < historyLen; ++j) {
    8:         if (history[j][0] != '\0') {
    9:             fprintf(fp, "%s\n", history[j]);
   10:         }
   11:     }
   12:     fclose(fp);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -1,4 +1,14 @@
 int linenoiseHistorySave(const char* filename) {
-    FILE* fp = fopen(filename, "wt");
+    FILE* fp;
+#if _POSIX_C_SOURCE >= 1 || _XOPEN_SOURCE || _POSIX_SOURCE
+    int fd = open(filename, O_CREAT, S_IRUSR | S_IWUSR);
+    if (fd == -1) {
+        // report errno somehow?
+        return -1;
+    }
+    fp = fdopen(fd, "wt");
+#else
+    fp = fopen(filename, "wt");
+#endif  // _POSIX_C_SOURCE >= 1 || _XOPEN_SOURCE || _POSIX_SOURCE
     if (fp == NULL) {
         return -1;
@@ -10,5 +20,5 @@
         }
     }
-    fclose(fp);
+    fclose(fp);  // Also causes fd to be closed.
     return 0;
 }
```

## primevul_paired:train:471:boundary_check_relation:2015:1985:loop_condition
target=1, pair=primevul_paired:train:235
{"pattern": "boundary_check_relation", "operation": {"start_byte": 2015, "end_byte": 2019, "lines": [53], "code": "c[j]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 1985, "end_byte": 1990, "lines": [52], "code": "j < i", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   50: 	*m = (*m < j) ? j : ((*m > i) ? i : *m);
   51: 
   52: 	/* Adjust last length count to fill out codes, if needed */
   53: 	for (y = 1 << j; j < i; j++, y <<= 1) {
   54: 		y -= c[j];
   55: 		if (y < 0)
   56: 			return 2; /* bad input: more codes than bits */
   57: 	}
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -12,9 +12,10 @@
  	unsigned j;             /* counter */
  	int k;                  /* number of bits in current code */
-	unsigned *p;            /* pointer into c[], b[], or v[] */
+	const unsigned *p;      /* pointer into c[], b[], or v[] */
  	huft_t *q;              /* points to current table */
  	huft_t r;               /* table entry for structure assignment */
  	huft_t *u[BMAX];        /* table stack */
  	unsigned v[N_MAX];      /* values in order of bit length */
+	unsigned v_end;
  	int ws[BMAX + 1];       /* bits decoded stack */
  	int w;                  /* bits decoded */
@@ -31,5 +32,5 @@
  	/* Generate counts for each bit length */
  	memset(c, 0, sizeof(c));
-	p = (unsigned *) b; /* cast allows us to reuse p for pointing to b */
+	p = b;
  	i = n;
  	do {
```

## primevul_paired:train:2292:boundary_check_relation:7542:7129:loop_condition
target=0, pair=primevul_paired:train:1145
{"pattern": "boundary_check_relation", "operation": {"start_byte": 7542, "end_byte": 7551, "lines": [258], "code": "colors[i]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 7129, "end_byte": 7157, "lines": [248], "code": "i < (ssize_t) header.ncolors", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
  246:            ximage=(XImage *) RelinquishMagickMemory(ximage);
  247:           ThrowReaderException(ResourceLimitError,"MemoryAllocationFailed");
  248:         }
  249:       for (i=0; i < (ssize_t) header.ncolors; i++)
  250:       {
  251:         count=ReadBlob(image,sz_XWDColor,(unsigned char *) &color);
  252:         if (count != sz_XWDColor)
  256:             ThrowReaderException(CorruptImageError,"UnexpectedEndOfFile");
  257:           }
  258:         colors[i].pixel=color.pixel;
  259:         colors[i].red=color.red;
  260:         colors[i].green=color.green;
  261:         colors[i].blue=color.blue;
  262:         colors[i].flags=(char) color.flags;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -89,4 +89,6 @@
    if (header.header_size < sz_XWDheader)
      ThrowReaderException(CorruptImageError,"ImproperImageHeader");
+  if ((MagickSizeType) header.xoffset >= GetBlobSize(image))
+    ThrowReaderException(CorruptImageError,"ImproperImageHeader");
    switch (header.visual_class)
    {
@@ -102,5 +104,5 @@
      {
        if ((header.bits_per_pixel < 1) || (header.bits_per_pixel > 15) ||
-          (header.ncolors == 0))
+          (header.colormap_entries == 0))
          ThrowReaderException(CorruptImageError,"ImproperImageHeader");
        break;
@@ -169,6 +171,4 @@
        ThrowReaderException(CorruptImageError,"ImproperImageHeader");
    }
-  if (header.ncolors > 65535)
-    ThrowReaderException(CorruptImageError,"ImproperImageHeader");
    if (((header.bitmap_pad % 8) != 0) || (header.bitmap_pad > 32))
      ThrowReaderException(CorruptImageError,"ImproperImageHeader");
@@ -238,6 +238,8 @@
          color;
  
-      colors=(XColor *) AcquireQuantumMemory((size_t) header.ncolors,
-        sizeof(*colors));
+      length=(size_t) header.ncolors;
+      if (length > ((~0UL)/sizeof(*colors)))
+        ThrowReaderException(CorruptImageError,"ImproperImageHeader");
+      colors=(XColor *) AcquireQuantumMemory(length,sizeof(*colors));
        if (colors == (XColor *) NULL)
          {
```

## primevul_paired:train:7466:boundary_check_relation:599:512:loop_condition
target=0, pair=primevul_paired:train:3732
{"pattern": "boundary_check_relation", "operation": {"start_byte": 599, "end_byte": 608, "lines": [18], "code": "buffer[i]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 512, "end_byte": 524, "lines": [16], "code": "i < numElems", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   14:         case 1:
   15:         {
   16:             unsigned char *temp2 = (unsigned char *)temp;
   17:             for (int i = 0; i < numElems; i ++)
   18:             {
   19:                 temp2[i] = (unsigned char)saturate(buffer[i] * 128.0f + 128.0f, 0.0f, 255.0f);
   20:             }
   21:             break;
   22:         }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -8,6 +8,5 @@
     bytesPerSample = header.format.bits_per_sample / 8;
     numBytes = numElems * bytesPerSample;
-    int confBufBytes = (numBytes + 3) & -4; // round up to nearest multiple of four to avoid overflow with 24bit-value assignment
-    void *temp = getConvBuffer(confBufBytes);
+    void *temp = getConvBuffer(numBytes + 7);   // round bit up to avoid buffer overrun with 24bit-value assignment
 
     switch (bytesPerSample)
```

## primevul_paired:train:1502:boundary_check_relation:4372:4241:loop_condition
target=0, pair=primevul_paired:train:750
{"pattern": "boundary_check_relation", "operation": {"start_byte": 4372, "end_byte": 4390, "lines": [144], "code": "image->colormap[i]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 4241, "end_byte": 4268, "lines": [141], "code": "i < (ssize_t) image->colors", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
  139:     (void) memset(pcx_colormap,0,3*256*sizeof(*pcx_colormap));
  140:     q=pcx_colormap;
  141:     if ((image->storage_class == PseudoClass) && (image->colors <= 256))
  142:       for (i=0; i < (ssize_t) image->colors; i++)
  143:       {
  144:         *q++=ScaleQuantumToChar(image->colormap[i].red);
  145:         *q++=ScaleQuantumToChar(image->colormap[i].green);
  146:         *q++=ScaleQuantumToChar(image->colormap[i].blue);
  147:       }
  148:     (void) WriteBlob(image,3*16,(const unsigned char *) pcx_colormap);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -156,5 +156,8 @@
      pixel_info=AcquireVirtualMemory(length,pcx_info.planes*sizeof(*pixels));
      if (pixel_info == (MemoryInfo *) NULL)
-      ThrowWriterException(ResourceLimitError,"MemoryAllocationFailed");
+      {
+        pcx_colormap=(unsigned char *) RelinquishMagickMemory(pcx_colormap);
+        ThrowWriterException(ResourceLimitError,"MemoryAllocationFailed");
+      }
      pixels=(unsigned char *) GetVirtualMemoryBlob(pixel_info);
      q=pixels;
```

## primevul_paired:train:1456:boundary_check_relation:494:455:loop_condition
target=0, pair=primevul_paired:train:727
{"pattern": "boundary_check_relation", "operation": {"start_byte": 494, "end_byte": 501, "lines": [18], "code": "cp[len]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 455, "end_byte": 467, "lines": [17], "code": "len < length", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   15:  	if (ptr < 3 || ((ptr + 1) & 3) || ptr > length + 1)
   16:  		ND_PRINT((ndo, " [bad ptr %u]", cp[2]));
   17:  
   18:  	for (len = 3; len < length; len += 4) {
   19: 		ND_TCHECK2(cp[len], 4);
   20:  		ND_PRINT((ndo, " %s", ipaddr_string(ndo, &cp[len])));
   21:  		if (ptr > len)
   22:  			ND_PRINT((ndo, ","));
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -7,8 +7,9 @@
  	if (length < 3) {
  		ND_PRINT((ndo, " [bad length %u]", length));
-		return;
+		return (0);
  	}
  	if ((length + 1) & 3)
  		ND_PRINT((ndo, " [bad length %u]", length));
+	ND_TCHECK(cp[2]);
  	ptr = cp[2] - 1;
  	if (ptr < 3 || ((ptr + 1) & 3) || ptr > length + 1)
@@ -16,7 +17,12 @@
  
  	for (len = 3; len < length; len += 4) {
+		ND_TCHECK2(cp[len], 4);
  		ND_PRINT((ndo, " %s", ipaddr_string(ndo, &cp[len])));
  		if (ptr > len)
  			ND_PRINT((ndo, ","));
  	}
+	return (0);
+
+trunc:
+	return (-1);
  }
```

## primevul_paired:train:6807:null_check_relation:456:119:early_return
target=1, pair=primevul_paired:train:3403
{"pattern": "null_check_relation", "operation": {"start_byte": 456, "end_byte": 467, "lines": [16], "code": "new_path[0]", "ast_type": "subscript_expression"}, "evidence": {"start_byte": 119, "end_byte": 130, "lines": [4], "code": "(!new_path)", "ast_type": "parenthesized_expression"}, "role": "early_return", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false, "zero_case": "excluded"}
```c
    2: {
    3:   size_t len;
    4:   char *new_path = strdup(cookie_path);
    5:   if(!new_path)
    6:     return NULL;
    7: 
    8:   /* some stupid site sends path attribute with '"'. */
   14:   }
   15: 
   16:   /* RFC6265 5.2.4 The Path Attribute */
   17:   if(new_path[0] != '/') {
   18:     /* Let cookie-path be the default-path. */
   19:     free(new_path);
   20:     new_path = strdup("/");
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -7,9 +7,12 @@
 
   /* some stupid site sends path attribute with '"'. */
+  len = strlen(new_path);
   if(new_path[0] == '\"') {
-    memmove((void *)new_path, (const void *)(new_path + 1), strlen(new_path));
+    memmove((void *)new_path, (const void *)(new_path + 1), len);
+    len--;
   }
-  if(new_path[strlen(new_path) - 1] == '\"') {
-    new_path[strlen(new_path) - 1] = 0x0;
+  if(len && (new_path[len - 1] == '\"')) {
+    new_path[len - 1] = 0x0;
+    len--;
   }
 
@@ -23,6 +26,5 @@
 
   /* convert /hoge/ to /hoge */
-  len = strlen(new_path);
-  if(1 < len && new_path[len - 1] == '/') {
+  if(len && new_path[len - 1] == '/') {
     new_path[len - 1] = 0x0;
   }
```

## primevul_paired:train:3059:null_check_relation:3367:2708:early_return
target=1, pair=primevul_paired:train:1529
{"pattern": "null_check_relation", "operation": {"start_byte": 3367, "end_byte": 3401, "lines": [84], "code": "dest_render_frame_host->GetProcess", "ast_type": "field_expression"}, "evidence": {"start_byte": 2708, "end_byte": 2733, "lines": [66], "code": "(!dest_render_frame_host)", "ast_type": "condition_clause"}, "role": "early_return", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false, "zero_case": "excluded"}
```c
   64:     RenderFrameHostImpl* dest_render_frame_host =
   65:         frame_tree_node->render_manager()->Navigate(
   66:             dest_url, frame_entry, entry, reload_type != ReloadType::NONE);
   67:     if (!dest_render_frame_host)
   68:       return false;  // Unable to create the desired RenderFrameHost.
   69: 
   70:     if (is_pending_entry)
   82:     bool is_transfer_to_same =
   83:         is_transfer &&
   84:         entry.transferred_global_request_id().child_id ==
   85:             dest_render_frame_host->GetProcess()->GetID();
   86:     if (!is_transfer_to_same) {
   87:       navigation_data_.reset(new NavigationMetricsData(
   88:           navigation_start, dest_url, entry.restore_type()));
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -18,4 +18,14 @@
    }
  
+  // Don't attempt to navigate if the virtual URL is non-empty and invalid.
+  if (frame_tree_node->IsMainFrame()) {
+    const GURL& virtual_url = entry.GetVirtualURL();
+    if (!virtual_url.is_valid() && !virtual_url.is_empty()) {
+      LOG(WARNING) << "Refusing to load for invalid virtual URL: "
+                   << virtual_url.possibly_invalid_spec();
+      return false;
+    }
+  }
+
    if (!dest_url.is_valid() && !dest_url.is_empty()) {
      LOG(WARNING) << "Refusing to load invalid URL: "
```

## primevul_paired:train:8247:null_check_relation:1907:1836:enclosing_branch
target=1, pair=primevul_paired:train:4123
{"pattern": "null_check_relation", "operation": {"start_byte": 1907, "end_byte": 1921, "lines": [91], "code": "ipc.opt->faddr", "ast_type": "field_expression"}, "evidence": {"start_byte": 1836, "end_byte": 1861, "lines": [88], "code": "(ipc.opt && ipc.opt->srr)", "ast_type": "parenthesized_expression"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true, "zero_case": "excluded"}
```c
   86: 	saddr = ipc.addr;
   87: 	ipc.addr = faddr = daddr;
   88: 
   89: 	if (ipc.opt && ipc.opt->srr) {
   90: 		if (!daddr)
   91: 			return -EINVAL;
   92: 		faddr = ipc.opt->faddr;
   93: 		connected = 0;
   94: 	}
   95: 	tos = RT_TOS(inet->tos);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -171,4 +171,6 @@
 	else if (!corkreq)
 		err = udp_push_pending_frames(sk, up);
+	else if (unlikely(skb_queue_empty(&sk->sk_write_queue)))
+		up->pending = 0;
 	release_sock(sk);
 
```

## primevul_paired:train:7691:null_check_relation:3104:2962:enclosing_branch
target=1, pair=primevul_paired:train:3845
{"pattern": "null_check_relation", "operation": {"start_byte": 3104, "end_byte": 3115, "lines": [83], "code": "err->domain", "ast_type": "field_expression"}, "evidence": {"start_byte": 2962, "end_byte": 2967, "lines": [81], "code": "(err)", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true, "zero_case": "excluded"}
```c
   79: 
   80:       s = g_convert ((gchar *) node->data + offset, len - offset, "utf8",
   81:           charset, NULL, NULL, &err);
   82:       if (err) {
   83:         GST_DEBUG_OBJECT (qtdemux, "Failed to convert string from charset %s:"
   84:             " %s(%d): %s", charset, g_quark_to_string (err->domain), err->code,
   85:             err->message);
   86:         g_error_free (err);
   87:       }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -29,5 +29,5 @@
     len = QT_UINT32 (node->data);
     type = QT_UINT32 ((guint8 *) node->data + 4);
-    if ((type >> 24) == 0xa9) {
+    if ((type >> 24) == 0xa9 && len > 8 + 4) {
       gint str_len;
       gint lang_code;
@@ -48,5 +48,5 @@
 
       offset = 12;
-      len = str_len + 8 + 4;    /* remove trailing strings that we don't use */
+      len = MIN (len, str_len + 8 + 4); /* remove trailing strings that we don't use */
       GST_DEBUG_OBJECT (qtdemux, "found international text tag");
 
```

## primevul_paired:train:6692:null_check_relation:3892:2231:enclosing_branch
target=0, pair=primevul_paired:train:3345
{"pattern": "null_check_relation", "operation": {"start_byte": 3892, "end_byte": 3904, "lines": [138], "code": "ent->content", "ast_type": "field_expression"}, "evidence": {"start_byte": 2231, "end_byte": 2302, "lines": [84, 85], "code": "((ent != NULL) &&\n\t\t    (ent->etype == XML_INTERNAL_PREDEFINED_ENTITY))", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false, "zero_case": "allowed"}
```c
   82: 		ctxt->nbentities++;
   83: 		if (ent != NULL)
   84: 		    ctxt->nbentities += ent->owner;
   85: 		if ((ent != NULL) &&
   86: 		    (ent->etype == XML_INTERNAL_PREDEFINED_ENTITY)) {
   87: 		    if (len + 10 > buf_size) {
   88: 			growBuffer(buf, 10);
   89: 		    }
  136: 		     */
  137: 		    if ((ent->etype != XML_INTERNAL_PREDEFINED_ENTITY) &&
  138: 			(ent->content != NULL)) {
  139: 			rep = xmlStringDecodeEntities(ctxt, ent->content,
  140: 						  XML_SUBSTITUTE_REF, 0, 0, 0);
  141: 			if (rep != NULL) {
  142: 			    xmlFree(rep);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -184,5 +184,5 @@
 
     if ((in_space) && (normalize)) {
-        while (buf[len - 1] == 0x20) len--;
+        while ((len > 0) && (buf[len - 1] == 0x20)) len--;
     }
     buf[len] = 0;
```

## primevul_paired:train:3634:null_check_relation:15730:15685:enclosing_branch
target=0, pair=primevul_paired:train:1816
{"pattern": "null_check_relation", "operation": {"start_byte": 15730, "end_byte": 15748, "lines": [502], "code": "*read_info->scenes", "ast_type": "pointer_expression"}, "evidence": {"start_byte": 15685, "end_byte": 15721, "lines": [501], "code": "(read_info->scenes != (char *) NULL)", "ast_type": "parenthesized_expression"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true, "zero_case": "excluded"}
```c
  499:         (read_info->scene+read_info->number_scenes));
  500:       (void) ConcatenateMagickString(options,pages,MaxTextExtent);
  501:       read_info->number_scenes=0;
  502:       if (read_info->scenes != (char *) NULL)
  503:         *read_info->scenes='\0';
  504:     }
  505:   if (*image_info->magick == 'E')
  506:     {
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -265,4 +265,6 @@
            continue;
          length=extent;
+        if (length > GetBlobSize(image))
+          ThrowReaderException(CorruptImageError,"InsufficientImageDataInFile");
          profile=BlobToStringInfo((const void *) NULL,length);
          if (profile != (StringInfo *) NULL)
```

## primevul_paired:train:8658:null_check_relation:7537:6444:early_return
target=0, pair=primevul_paired:train:4328
{"pattern": "null_check_relation", "operation": {"start_byte": 7537, "end_byte": 7547, "lines": [178], "code": "data->body", "ast_type": "field_expression"}, "evidence": {"start_byte": 6444, "end_byte": 6475, "lines": [147], "code": "(!data || !ci_req_hasbody(req))", "ast_type": "parenthesized_expression"}, "role": "early_return", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false, "zero_case": "excluded"}
```c
  145:      }
  146: 
  147:      /* No data, so nothing to scan */
  148:      if (!data || !ci_req_hasbody(req)) {
  149: 	 ci_debug_printf(1, "DEBUG squidclamav_check_preview_handler: No body data, allow 204\n");
  150:           return CI_MOD_ALLOW204;
  151:      }
  176: 	ci_req_unlock_data(req);
  177: 	ci_simple_file_lock_all(data->body);
  178:      }
  179:      if (!data->body)
  180: 	return CI_ERROR;
  181: 
  182:      if (preview_data_len) {
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -86,8 +86,12 @@
      /* Check URL header against squidGuard */
      if (usepipe == 1) {
+	char *rbuff = NULL;
 	ci_debug_printf(2, "DEBUG squidclamav_check_preview_handler: Sending request to chained program: %s\n", squidguard);
 	ci_debug_printf(2, "DEBUG squidclamav_check_preview_handler: Request: %s %s %s %s\n", httpinf.url,clientip,username,httpinf.method);
-	fprintf(sgfpw,"%s %s %s %s\n",httpinf.url,clientip,username,httpinf.method);
+	/* escaping escaped character to prevent unescaping by squidguard */
+	rbuff = replace(httpinf.url, "%", "%25");
+	fprintf(sgfpw,"%s %s %s %s\n",rbuff,clientip,username,httpinf.method);
 	fflush(sgfpw);
+	xfree(rbuff);
 	/* the chained redirector must return empty line if ok or the redirection url */
 	chain_ret = (char *)malloc(sizeof(char)*MAX_URL_SIZE);
```

## primevul_paired:train:1070:null_check_relation:1780:1667:early_return
target=0, pair=primevul_paired:train:534
{"pattern": "null_check_relation", "operation": {"start_byte": 1780, "end_byte": 1792, "lines": [46], "code": "ssid_ie->len", "ast_type": "field_expression"}, "evidence": {"start_byte": 1667, "end_byte": 1718, "lines": [43], "code": "(!ssid_ie || ssid_ie->len > IEEE80211_MAX_SSID_LEN)", "ast_type": "condition_clause"}, "role": "early_return", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false, "zero_case": "excluded"}
```c
   41:  				(u8 *)&settings->beacon.head[ie_offset],
   42:  				settings->beacon.head_len - ie_offset,
   43:  				WLAN_EID_SSID);
   44: 		if (!ssid_ie || ssid_ie->len > IEEE80211_MAX_SSID_LEN)
   45:  			return -EINVAL;
   46:  
   47:  		memcpy(ssid_le.SSID, ssid_ie->data, ssid_ie->len);
   48: 		ssid_le.SSID_len = cpu_to_le32(ssid_ie->len);
   49: 		brcmf_dbg(TRACE, "SSID is (%s) in Head\n", ssid_le.SSID);
   50: 	} else {
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -42,5 +42,5 @@
  				settings->beacon.head_len - ie_offset,
  				WLAN_EID_SSID);
-		if (!ssid_ie)
+		if (!ssid_ie || ssid_ie->len > IEEE80211_MAX_SSID_LEN)
  			return -EINVAL;
  
```

## primevul_paired:train:4557:integer_range_relation:325:251:enclosing_branch
target=1, pair=primevul_paired:train:2278
{"pattern": "integer_range_relation", "operation": {"start_byte": 325, "end_byte": 335, "lines": [12], "code": "4 - bufPos", "ast_type": "binary_expression"}, "evidence": {"start_byte": 251, "end_byte": 263, "lines": [10], "code": "(bufPos < 4)", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
    8: 
    9:     int readBytes;
   10: 
   11:     if (bufPos < 4) {
   12: 
   13:         readBytes = readFromSocket(self, buffer + bufPos, 4 - bufPos);
   14: 
   15:         if (readBytes < 0)
   16:             goto exit_closed;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -44,4 +44,7 @@
     }
 
+    if (self->packetSize <= bufPos)
+        goto exit_error;
+
     readBytes = readFromSocket(self, buffer + bufPos, self->packetSize - bufPos);
 
```

## primevul_paired:train:3341:integer_range_relation:3515:3410:loop_condition
target=1, pair=primevul_paired:train:1670
{"pattern": "integer_range_relation", "operation": {"start_byte": 3515, "end_byte": 3518, "lines": [88], "code": "i+1", "ast_type": "binary_expression"}, "evidence": {"start_byte": 3410, "end_byte": 3419, "lines": [85], "code": "i + 1 < n", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   83:         camera3::CameraTraces::dump(fd, args);
   84: 
   85:  int n = args.size();
   86:  for (int i = 0; i + 1 < n; i++) {
   87:  String16 verboseOption("-v");
   88:  if (args[i] == verboseOption) {
   89:  String8 levelStr(args[i+1]);
   90:  int level = atoi(levelStr.string());
   91:                 result = String8::format("\nSetting log level to %d.\n", level);
   92:                 setLogLevel(level);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -71,5 +71,5 @@
              result = String8::format("  Device is open. Client instance dump:\n");
              write(fd, result.string(), result.size());
-            client->dump(fd, args);
+            client->dumpClient(fd, args);
          }
          if (!hasClient) {
```

## primevul_paired:train:3865:integer_range_relation:884:559:enclosing_branch
target=1, pair=primevul_paired:train:1932
{"pattern": "integer_range_relation", "operation": {"start_byte": 884, "end_byte": 903, "lines": [30], "code": "numInts*sizeof(int)", "ast_type": "binary_expression"}, "evidence": {"start_byte": 559, "end_byte": 578, "lines": [21], "code": "(numFds || numInts)", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   19:         free_handle();
   20:  }
   21: 
   22:  if (numFds || numInts) {
   23:         width  = buf[1];
   24:         height = buf[2];
   25:         stride = buf[3];
   28:          usage  = buf[5];
   29:          native_handle* h = native_handle_create(numFds, numInts);
   30:          memcpy(h->data,          fds,     numFds*sizeof(int));
   31:          memcpy(h->data + numFds, &buf[10], numInts*sizeof(int));
   32:          handle = h;
   33:  } else {
   34:         width = height = stride = format = usage = 0;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -10,8 +10,17 @@
      const size_t numInts = buf[9];
  
+    const size_t maxNumber = UINT_MAX / sizeof(int);
+    if (numFds >= maxNumber || numInts >= (maxNumber - 10)) {
+        width = height = stride = format = usage = 0;
+        handle = NULL;
+        ALOGE("unflatten: numFds or numInts is too large: %d, %d",
+                numFds, numInts);
+        return BAD_VALUE;
+    }
+
      const size_t sizeNeeded = (10 + numInts) * sizeof(int);
      if (size < sizeNeeded) return NO_MEMORY;
  
-    size_t fdCountNeeded = 0;
+    size_t fdCountNeeded = numFds;
      if (count < fdCountNeeded) return NO_MEMORY;
  
@@ -28,4 +37,10 @@
          usage  = buf[5];
          native_handle* h = native_handle_create(numFds, numInts);
+        if (!h) {
+            width = height = stride = format = usage = 0;
+            handle = NULL;
+            ALOGE("unflatten: native_handle_create failed");
+            return NO_MEMORY;
+        }
          memcpy(h->data,          fds,     numFds*sizeof(int));
          memcpy(h->data + numFds, &buf[10], numInts*sizeof(int));
```

## primevul_paired:train:5065:integer_range_relation:629:578:loop_condition
target=1, pair=primevul_paired:train:2532
{"pattern": "integer_range_relation", "operation": {"start_byte": 629, "end_byte": 634, "lines": [18], "code": "i - 1", "ast_type": "binary_expression"}, "evidence": {"start_byte": 578, "end_byte": 600, "lines": [17], "code": "i < header.header_size", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   15:     }
   16: 
   17:     /* unscramble the header bytes with a XOR operation */
   18:     for (i = 1; i < header.header_size; i++)
   19:         header_buffer[i - 1] = s->buf[i] ^ s->buf[i + 1];
   20: 
   21:     header.compression = header_buffer[0];
   22:     header.deltaset = header_buffer[1];
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -12,4 +12,9 @@
     {
         av_log(s->avctx, AV_LOG_ERROR, "invalid header size (%d)\n", s->buf[0]);
+        return AVERROR_INVALIDDATA;
+    }
+
+    if (header.header_size + 1 > s->size) {
+        av_log(s->avctx, AV_LOG_ERROR, "Input packet too small.\n");
         return AVERROR_INVALIDDATA;
     }
```

## primevul_paired:train:5172:integer_range_relation:3411:3191:enclosing_branch
target=0, pair=primevul_paired:train:2585
{"pattern": "integer_range_relation", "operation": {"start_byte": 3411, "end_byte": 3431, "lines": [154], "code": "nerr - nerr_loglimit", "ast_type": "binary_expression"}, "evidence": {"start_byte": 3191, "end_byte": 3213, "lines": [146], "code": "(nerr > nerr_maxlimit)", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false}
```c
  144: 		}
  145: 	}
  146: 	fclose(fp);
  147: 	if (nerr > nerr_maxlimit) {
  148: 		msyslog(LOG_ERR,
  149: 			"authreadkeys: emergeny break after %u errors",
  150: 			nerr);
  152: 	} else if (nerr > nerr_loglimit) {
  153: 		msyslog(LOG_ERR,
  154: 			"authreadkeys: found %u more error(s)",
  155: 			nerr - nerr_loglimit);
  156: 	}
  157: 	return (1);
  158: }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -12,5 +12,5 @@
 	size_t	len;
 	size_t	j;
-
+	size_t  nerr;
 	/*
 	 * Open file.  Complain and return if it can't be opened.
@@ -32,5 +32,8 @@
 	 * Now read lines from the file, looking for key entries
 	 */
+	nerr = 0;
 	while ((line = fgets(buf, sizeof buf, fp)) != NULL) {
+		if (nerr > nerr_maxlimit)
+			break;
 		token = nexttok(&line);
 		if (token == NULL)
@@ -42,13 +45,14 @@
 		keyno = atoi(token);
 		if (keyno == 0) {
-			msyslog(LOG_ERR,
-			    "authreadkeys: cannot change key %s", token);
+			log_maybe(&nerr,
+				  "authreadkeys: cannot change key %s",
+				  token);
 			continue;
 		}
 
 		if (keyno > NTP_MAXKEY) {
-			msyslog(LOG_ERR,
-			    "authreadkeys: key %s > %d reserved for Autokey",
-			    token, NTP_MAXKEY);
+			log_maybe(&nerr,
+				  "authreadkeys: key %s > %d reserved for Autokey",
+				  token, NTP_MAXKEY);
 			continue;
 		}
@@ -59,6 +63,7 @@
 		token = nexttok(&line);
 		if (token == NULL) {
-			msyslog(LOG_ERR,
-			    "authreadkeys: no key type for key %d", keyno);
+			log_maybe(&nerr,
+				  "authreadkeys: no key type for key %d",
+				  keyno);
 			continue;
 		}
@@ -72,11 +77,13 @@
 		keytype = keytype_from_text(token, NULL);
 		if (keytype == 0) {
-			msyslog(LOG_ERR,
-			    "authreadkeys: invalid type for key %d", keyno);
+			log_maybe(&nerr,
+				  "authreadkeys: invalid type for key %d",
+				  keyno);
 			continue;
 		}
 		if (EVP_get_digestbynid(keytype) == NULL) {
-			msyslog(LOG_ERR,
-			    "authreadkeys: no algorithm for key %d", keyno);
+			log_maybe(&nerr,
+				  "authreadkeys: no algorithm for key %d",
+				  keyno);
 			continue;
 		}
@@ -88,6 +95,7 @@
 		 */
 		if (!(*token == 'M' || *token == 'm')) {
-			msyslog(LOG_ERR,
... display truncated; source retained unchanged ...
```

## primevul_paired:train:5066:integer_range_relation:813:734:loop_condition
target=0, pair=primevul_paired:train:2532
{"pattern": "integer_range_relation", "operation": {"start_byte": 813, "end_byte": 818, "lines": [23], "code": "i + 1", "ast_type": "binary_expression"}, "evidence": {"start_byte": 734, "end_byte": 756, "lines": [22], "code": "i < header.header_size", "ast_type": "binary_expression"}, "role": "loop_condition", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   20:     }
   21: 
   22:     /* unscramble the header bytes with a XOR operation */
   23:     for (i = 1; i < header.header_size; i++)
   24:         header_buffer[i - 1] = s->buf[i] ^ s->buf[i + 1];
   25: 
   26:     header.compression = header_buffer[0];
   27:     header.deltaset = header_buffer[1];
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -12,4 +12,9 @@
     {
         av_log(s->avctx, AV_LOG_ERROR, "invalid header size (%d)\n", s->buf[0]);
+        return AVERROR_INVALIDDATA;
+    }
+
+    if (header.header_size + 1 > s->size) {
+        av_log(s->avctx, AV_LOG_ERROR, "Input packet too small.\n");
         return AVERROR_INVALIDDATA;
     }
```

## primevul_paired:train:8100:integer_range_relation:3720:3661:enclosing_branch
target=0, pair=primevul_paired:train:4049
{"pattern": "integer_range_relation", "operation": {"start_byte": 3720, "end_byte": 3735, "lines": [138], "code": "byte << (8-bit)", "ast_type": "binary_expression"}, "evidence": {"start_byte": 3661, "end_byte": 3671, "lines": [136], "code": "(bit != 0)", "ast_type": "parenthesized_expression"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
  134:             }
  135:            p++;
  136:          }
  137:          if (bit != 0)
  138:            {
  139:              *q++=(unsigned char) (byte << (8-bit));
  140:              x++;
  141:            }
  142:         for (x=(ssize_t) (image->columns+7)/8; x < (ssize_t) bytes_per_line; x++)
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -97,6 +97,6 @@
     Convert MIFF to DIB raster pixels.
   */
-  pixels=(unsigned char *) AcquireQuantumMemory(dib_info.image_size,
-    sizeof(*pixels));
+  pixels=(unsigned char *) AcquireQuantumMemory(image->rows,MagickMax(
+    bytes_per_line,image->columns+256UL)*sizeof(*pixels));
   if (pixels == (unsigned char *) NULL)
     ThrowWriterException(ResourceLimitError,"MemoryAllocationFailed");
```

## primevul_paired:train:4344:integer_range_relation:4311:4192:enclosing_branch
target=0, pair=primevul_paired:train:2171
{"pattern": "integer_range_relation", "operation": {"start_byte": 4311, "end_byte": 4328, "lines": [111], "code": "size + flags_size", "ast_type": "binary_expression"}, "evidence": {"start_byte": 4192, "end_byte": 4220, "lines": [109], "code": "(size + flags_size >= 1<<24)", "ast_type": "parenthesized_expression"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
  107:     if (sc->last_ts < ts)
  108:         sc->last_ts = ts;
  109: 
  110:     if (size + flags_size >= 1<<24) {
  111:         av_log(s, AV_LOG_ERROR, "Too large packet with size %u >= %u\n",
  112:                size + flags_size, 1<<24);
  113:         return AVERROR(EINVAL);
  114:     }
  115: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -8,10 +8,15 @@
     int size = pkt->size;
     uint8_t *data = NULL;
-    int flags = -1, flags_size, ret;
-    int64_t cur_offset = avio_tell(pb);
-
-    if (par->codec_id == AV_CODEC_ID_VP6F || par->codec_id == AV_CODEC_ID_VP6A ||
-        par->codec_id == AV_CODEC_ID_VP6  || par->codec_id == AV_CODEC_ID_AAC)
-        flags_size = 2;
+     int flags = -1, flags_size, ret;
+     int64_t cur_offset = avio_tell(pb);
+ 
+    if (par->codec_type == AVMEDIA_TYPE_AUDIO && !pkt->size) {
+        av_log(s, AV_LOG_WARNING, "Empty audio Packet\n");
+        return AVERROR(EINVAL);
+    }
+
+     if (par->codec_id == AV_CODEC_ID_VP6F || par->codec_id == AV_CODEC_ID_VP6A ||
+         par->codec_id == AV_CODEC_ID_VP6  || par->codec_id == AV_CODEC_ID_AAC)
+         flags_size = 2;
     else if (par->codec_id == AV_CODEC_ID_H264 || par->codec_id == AV_CODEC_ID_MPEG4)
         flags_size = 5;
@@ -118,5 +123,4 @@
         int64_t metadata_size_pos = avio_tell(pb);
         if (par->codec_id == AV_CODEC_ID_TEXT) {
-            // legacy FFmpeg magic?
             avio_w8(pb, AMF_DATA_TYPE_STRING);
             put_amf_string(pb, "onTextData");
@@ -132,5 +136,4 @@
             avio_w8(pb, AMF_END_OF_OBJECT);
         } else {
-            // just pass the metadata through
             avio_write(pb, data ? data : pkt->data, size);
         }
```

## primevul_paired:train:1847:memory_extent_relation:4472:4322:destination_capacity_definition
target=1, pair=primevul_paired:train:923
{"pattern": "memory_extent_relation", "operation": {"start_byte": 4472, "end_byte": 4518, "lines": [206], "code": "memset(h->attrBools, 1, (size_t)zip->numFiles)", "ast_type": "call_expression"}, "evidence": {"start_byte": 4322, "end_byte": 4397, "lines": [201, 202], "code": "h->attrBools = calloc((size_t)zip->numFiles,\n \t\t\t    sizeof(*h->attrBools))", "ast_type": "assignment_expression"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "calloc ( ( size_t ) zip -> numFiles , sizeof ( * h -> attrBools ) )", "capacity_dimensions": [], "capacity_unit": "allocation bytes", "capacity_bytes": "unknown", "access_length": "( size_t ) zip -> numFiles", "extent_comparison": "unknown"}
```c
  199:  			if ((p = header_bytes(a, 2)) == NULL)
  200:  				return (-1);
  201:  			allAreDefined = *p;
  202:  			h->attrBools = calloc((size_t)zip->numFiles,
  203:  			    sizeof(*h->attrBools));
  204:  			if (h->attrBools == NULL)
  205: 				return (-1);
  206: 			if (allAreDefined)
  207: 				memset(h->attrBools, 1, (size_t)zip->numFiles);
  208: 			else {
  209: 				if (read_Bools(a, h->attrBools,
  210: 				      (size_t)zip->numFiles) < 0)
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -88,4 +88,6 @@
  		switch (type) {
  		case kEmptyStream:
+			if (h->emptyStreamBools != NULL)
+				return (-1);
  			h->emptyStreamBools = calloc((size_t)zip->numFiles,
  			    sizeof(*h->emptyStreamBools));
@@ -108,4 +110,6 @@
  				break;
  			}
+			if (h->emptyFileBools != NULL)
+				return (-1);
  			h->emptyFileBools = calloc(empty_streams,
  			    sizeof(*h->emptyFileBools));
@@ -122,4 +126,6 @@
  				break;
  			}
+			if (h->antiBools != NULL)
+				return (-1);
  			h->antiBools = calloc(empty_streams,
  			    sizeof(*h->antiBools));
@@ -148,4 +154,6 @@
  				return (-1);
  
+			if (zip->entry_names != NULL)
+				return (-1);
  			zip->entry_names = malloc(ll);
  			if (zip->entry_names == NULL)
@@ -200,4 +208,6 @@
  				return (-1);
  			allAreDefined = *p;
+			if (h->attrBools != NULL)
+				return (-1);
  			h->attrBools = calloc((size_t)zip->numFiles,
  			    sizeof(*h->attrBools));
```

## primevul_paired:train:6659:memory_extent_relation:5951:5789:destination_capacity_definition
target=1, pair=primevul_paired:train:3329
{"pattern": "memory_extent_relation", "operation": {"start_byte": 5951, "end_byte": 5997, "lines": [140], "code": "memcpy(rawExifData, (char*)&exifLongHeader, 4)", "ast_type": "call_expression"}, "evidence": {"start_byte": 5789, "end_byte": 5822, "lines": [136], "code": "rawExifData = (byte*)malloc(size)", "ast_type": "assignment_expression"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "( byte * ) malloc ( size )", "capacity_dimensions": [], "capacity_unit": "allocation bytes", "capacity_bytes": "unknown", "access_length": "4", "extent_comparison": "unknown"}
```c
  134:                 }
  135: 
  136:                 const long size = payload.size_ + offset;
  137:                 rawExifData = (byte*)malloc(size);
  138: 
  139:                 if (s_header) {
  140:                     us2Data(size_buff, (uint16_t) (size - 6), bigEndian);
  141:                     memcpy(rawExifData, (char*)&exifLongHeader, 4);
  142:                     memcpy(rawExifData + 4, (char*)&size_buff, 2);
  143:                 }
  144: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -1,3 +1,3 @@
-    void WebPImage::decodeChunks(uint64_t filesize)
+    void WebPImage::decodeChunks(uint32_t filesize)
     {
         DataBuf   chunkId(5);
@@ -11,7 +11,8 @@
         chunkId.pData_[4] = '\0' ;
         while ( !io_->eof() && (uint64_t) io_->tell() < filesize) {
-            io_->read(chunkId.pData_, WEBP_TAG_SIZE);
-            io_->read(size_buff, WEBP_TAG_SIZE);
+            readOrThrow(*io_, chunkId.pData_, WEBP_TAG_SIZE, Exiv2::kerCorruptedMetadata);
+            readOrThrow(*io_, size_buff, WEBP_TAG_SIZE, Exiv2::kerCorruptedMetadata);
             const uint32_t size = Exiv2::getULong(size_buff, littleEndian);
+            enforce(io_->tell() <= filesize, Exiv2::kerCorruptedMetadata);
             enforce(size <= (filesize - io_->tell()), Exiv2::kerCorruptedMetadata);
 
@@ -24,5 +25,5 @@
                 byte size_buf[WEBP_TAG_SIZE];
 
-                io_->read(payload.pData_, payload.size_);
+                readOrThrow(*io_, payload.pData_, payload.size_, Exiv2::kerCorruptedMetadata);
 
                 // Fetch width
@@ -39,5 +40,5 @@
 
                 has_canvas_data = true;
-                io_->read(payload.pData_, payload.size_);
+                readOrThrow(*io_, payload.pData_, payload.size_, Exiv2::kerCorruptedMetadata);
                 byte size_buf[WEBP_TAG_SIZE];
 
@@ -60,5 +61,5 @@
                 byte size_buf_h[3];
 
-                io_->read(payload.pData_, payload.size_);
+                readOrThrow(*io_, payload.pData_, payload.size_, Exiv2::kerCorruptedMetadata);
 
                 // Fetch width
@@ -78,5 +79,5 @@
                 byte size_buf[WEBP_TAG_SIZE];
 
-                io_->read(payload.pData_, payload.size_);
+                readOrThrow(*io_, payload.pData_, payload.size_, Exiv2::kerCorruptedMetadata);
 
                 // Fetch width
@@ -90,8 +91,8 @@
                 pixelHeight_ = Exiv2::getULong(size_buf, littleEndian) + 1;
             } else if (equalsWebPTag(chunkId, WEBP_CHUNK_HEADER_ICCP)) {
-                io_->read(payload.pData_, payload.size_);
+                readOrThrow(*io_, payload.pData_, payload.size_, Exiv2::kerCorruptedMetadata);
                 this->setIccProfile(payload);
             } else if (equalsWebPTag(chunkId, WEBP_CHUNK_HEADER_EXIF)) {
-                io_->read(payload.pData_, payload.size_);
+                readOrThrow(*io_, payload.pData_, payload.size_, Exiv2::kerCorruptedMetadata);
 
                 byte  size_buff[2];
@@ -174,5 +175,5 @@
                 if (rawExifData) free(rawExifData);
             } else if (equalsWebPTag(chunkId, WEBP_CHUNK_HEADER_XMP)) {
-                io_->read(payload.pData_, payload.size_);
+                readOrThrow(*io_, payload.pData_, payload.size_, Exiv2::kerCorruptedMetadata);
                 xmpPacket_.assign(reinterpret_cast<char*>(payload.pData_), payload.size_);
                 if (xmpPacket_.size() > 0 && XmpParser::decode(xmpData_, xmpPacket_)) {
```

## primevul_paired:train:4655:memory_extent_relation:1956:1828:destination_capacity_definition
target=1, pair=primevul_paired:train:2327
{"pattern": "memory_extent_relation", "operation": {"start_byte": 1956, "end_byte": 1981, "lines": [39], "code": "memset(dst,0,sizeof(dst))", "ast_type": "call_expression"}, "evidence": {"start_byte": 1828, "end_byte": 1835, "lines": [35], "code": "dst[64]", "ast_type": "array_declarator"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "64", "capacity_dimensions": ["64"], "capacity_unit": "elements", "capacity_bytes": "sizeof(dst)", "access_length": "sizeof ( dst )", "extent_comparison": "unknown"}
```c
   33:         //
   34:         for(by = blocks.ra_MinY;by <= blocks.ra_MaxY;by++) {
   35:           for(bx = blocks.ra_MinX;bx <= blocks.ra_MaxX;bx++) {
   36:             LONG dst[64];
   37:             if (*m_pppImage[i]) {
   38:               FetchRegion(bx,*m_pppImage[i],dst);
   39:             } else {
   40:               memset(dst,0,sizeof(dst));
   41:             }
   42:             up->DefineRegion(bx,by,dst);
   43:           }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -123,5 +123,9 @@
           if (i >= rr->rr_usFirstComponent && i <= rr->rr_usLastComponent) {
             ExtractBitmap(m_ppTempIBM[i],r,i);
-            FetchRegion(x,*m_pppImage[i],dst);
+            if (*m_pppImage[i]) {
+              FetchRegion(x,*m_pppImage[i],dst);
+            } else {
+              memset(dst,0,sizeof(LONG) * 64);
+            }
           } else {
             memset(dst,0,sizeof(LONG) * 64);
```

## primevul_paired:train:7157:memory_extent_relation:2968:418:destination_capacity_definition
target=1, pair=primevul_paired:train:3578
{"pattern": "memory_extent_relation", "operation": {"start_byte": 2968, "end_byte": 3004, "lines": [101], "code": "memcpy(hash, hash_ptr, sizeof(hash))", "ast_type": "call_expression"}, "evidence": {"start_byte": 418, "end_byte": 440, "lines": [14], "code": "hash[sizeof(pss.hash)]", "ast_type": "array_declarator"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "sizeof ( pss . hash )", "capacity_dimensions": ["sizeof ( pss . hash )"], "capacity_unit": "elements", "capacity_bytes": "sizeof(hash)", "access_length": "sizeof ( hash )", "extent_comparison": "unknown"}
```c
   12: 
   13:   part_signature_sha256_t pss;
   14:   uint16_t pss_head_length;
   15:   char hash[sizeof(pss.hash)];
   16: 
   17:   gcry_md_hd_t hd;
   18:   gcry_error_t err;
   99:     sfree(pss.username);
  100:     return (-1);
  101:   }
  102:   memcpy(hash, hash_ptr, sizeof(hash));
  103: 
  104:   /* Clean up */
  105:   gcry_md_close(hd);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -23,12 +23,4 @@
   buffer_offset = 0;
 
-  if (se->data.server.userdb == NULL) {
-    c_complain(
-        LOG_NOTICE, &complain_no_users,
-        "network plugin: Received signed network packet but can't verify it "
-        "because no user DB has been configured. Will accept it.");
-    return (0);
-  }
-
   /* Check if the buffer has enough data for this structure. */
   if (buffer_len <= PART_SIGNATURE_SHA256_SIZE)
@@ -45,4 +37,16 @@
     ERROR("network plugin: HMAC-SHA-256 with invalid length received.");
     return (-1);
+  }
+
+  if (se->data.server.userdb == NULL) {
+    c_complain(
+        LOG_NOTICE, &complain_no_users,
+        "network plugin: Received signed network packet but can't verify it "
+        "because no user DB has been configured. Will accept it.");
+
+    *ret_buffer = buffer + pss_head_length;
+    *ret_buffer_len -= pss_head_length;
+
+    return (0);
   }
 
```

## primevul_paired:train:7720:memory_extent_relation:2372:230:destination_capacity_definition
target=0, pair=primevul_paired:train:3859
{"pattern": "memory_extent_relation", "operation": {"start_byte": 2372, "end_byte": 2405, "lines": [104], "code": "memset(command,0,sizeof(command))", "ast_type": "call_expression"}, "evidence": {"start_byte": 230, "end_byte": 252, "lines": [8], "code": "command[MaxTextExtent]", "ast_type": "array_declarator"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "MaxTextExtent", "capacity_dimensions": ["MaxTextExtent"], "capacity_unit": "elements", "capacity_bytes": "sizeof(command)", "access_length": "sizeof ( command )", "extent_comparison": "unknown"}
```c
    6: #define RenderPCLText  "  Rendering PCL...  "
    7: 
    8:   char
    9:     command[MaxTextExtent],
   10:     *density,
   11:     filename[MaxTextExtent],
   12:     geometry[MaxTextExtent],
  102:   (void) memset(&bounding_box,0,sizeof(bounding_box));
  103:   (void) memset(&bounds,0,sizeof(bounds));
  104:   (void) memset(&page,0,sizeof(page));
  105:   (void) memset(command,0,sizeof(command));
  106:   p=command;
  107:   for (c=ReadBlobByte(image); c != EOF; c=ReadBlobByte(image))
  108:   {
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -175,5 +175,8 @@
        delegate_info=GetDelegateInfo("pcl:color",(char *) NULL,exception);
   if (delegate_info == (const DelegateInfo *) NULL)
-    return((Image *) NULL);
+    {
+      image=DestroyImage(image);
+      return((Image *) NULL);
+    }
   if ((page.width == 0) || (page.height == 0))
     (void) ParseAbsoluteGeometry(PSPageGeometry,&page);
```

## primevul_paired:train:4308:memory_extent_relation:2914:2882:destination_capacity_definition
target=0, pair=primevul_paired:train:2153
{"pattern": "memory_extent_relation", "operation": {"start_byte": 2914, "end_byte": 2963, "lines": [69], "code": "memcpy (formatstr, caf_audio_format.mFormatID, 4)", "ast_type": "call_expression"}, "evidence": {"start_byte": 2882, "end_byte": 2895, "lines": [67], "code": "formatstr [5]", "ast_type": "array_declarator"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "5", "capacity_dimensions": ["5"], "capacity_unit": "elements", "capacity_bytes": "sizeof(formatstr)", "access_length": "4", "extent_comparison": "unknown"}
```c
   65:             desc_chunk = 1;
   66:  
   67:              if (debug_logging_mode) {
   68:                  char formatstr [5];
   69: 
   70:                 memcpy (formatstr, caf_audio_format.mFormatID, 4);
   71:                 formatstr [4] = 0;
   72:                 error_line ("format = %s, flags = %x, sampling rate = %g",
   73:                     formatstr, caf_audio_format.mFormatFlags, caf_audio_format.mSampleRate);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -1,8 +1,8 @@
-int ParseCaffHeaderConfig (FILE *infile, char *infilename, char *fourcc, WavpackContext *wpc, WavpackConfig *config)
-{
-    uint32_t chan_chunk = 0, channel_layout = 0, bcount;
-    unsigned char *channel_identities = NULL;
-    unsigned char *channel_reorder = NULL;
-    int64_t total_samples = 0, infilesize;
+ int ParseCaffHeaderConfig (FILE *infile, char *infilename, char *fourcc, WavpackContext *wpc, WavpackConfig *config)
+ {
+    uint32_t chan_chunk = 0, desc_chunk = 0, channel_layout = 0, bcount;
+     unsigned char *channel_identities = NULL;
+     unsigned char *channel_reorder = NULL;
+     int64_t total_samples = 0, infilesize;
     CAFFileHeader caf_file_header;
     CAFChunkHeader caf_chunk_header;
@@ -31,6 +31,4 @@
     }
 
-    // loop through all elements of the RIFF wav header
-    // (until the data chuck) and copy them to the output file
 
     while (1) {
@@ -48,6 +46,4 @@
         WavpackBigEndianToNative (&caf_chunk_header, CAFChunkHeaderFormat);
 
-        // if it's the format chunk, we want to get some info out of there and
-        // make sure it's a .caf file we can handle
 
         if (!strncmp (caf_chunk_header.mChunkType, "desc", 4)) {
@@ -64,10 +60,11 @@
                     error_line ("%s", WavpackGetErrorMessage (wpc));
                     return WAVPACK_SOFT_ERROR;
-            }
-
-            WavpackBigEndianToNative (&caf_audio_format, CAFAudioFormatFormat);
-
-            if (debug_logging_mode) {
-                char formatstr [5];
+             }
+ 
+             WavpackBigEndianToNative (&caf_audio_format, CAFAudioFormatFormat);
+            desc_chunk = 1;
+ 
+             if (debug_logging_mode) {
+                 char formatstr [5];
 
                 memcpy (formatstr, caf_audio_format.mFormatID, 4);
@@ -177,7 +174,4 @@
                         }
 
-                        // we allocate (and initialize to invalid values) a channel reorder array
-                        // (even though we might not end up doing any reordering) and a string for
-                        // any non-Microsoft channels we encounter
 
                         channel_reorder = malloc (num_descriptions);
@@ -185,5 +179,4 @@
                         channel_identities = malloc (num_descriptions+1);
 
-                        // convert the descriptions array to our native endian so it's easy to access
 
                         for (i = 0; i < num_descriptions; ++i) {
@@ -194,5 +187,4 @@
                         }
 
-                        // first, we go though and find any MS channels present, and move those to the beginning
 
                         for (label = 1; label <= 18; ++label)
@@ -204,5 +196,4 @@
... display truncated; source retained unchanged ...
```

## primevul_paired:train:8694:memory_extent_relation:1007:132:destination_capacity_definition
target=0, pair=primevul_paired:train:4346
{"pattern": "memory_extent_relation", "operation": {"start_byte": 1007, "end_byte": 1035, "lines": [36], "code": "strncpy(uHex, uStr + j*4, 4)", "ast_type": "call_expression"}, "evidence": {"start_byte": 132, "end_byte": 139, "lines": [4], "code": "uHex[5]", "ast_type": "array_declarator"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "5", "capacity_dimensions": ["5"], "capacity_unit": "elements", "capacity_bytes": "sizeof(uHex)", "access_length": "4", "extent_comparison": "unknown"}
```c
    2: 				   int offset) {
    3:   CharCode oldLen, i;
    4:   Unicode u;
    5:   char uHex[5];
    6:   int j;
    7: 
    8:   if (code >= mapLen) {
   34:     sMap[sMapLen].c = code;
   35:     sMap[sMapLen].len = n / 4;
   36:     for (j = 0; j < sMap[sMapLen].len && j < maxUnicodeString; ++j) {
   37:       strncpy(uHex, uStr + j*4, 4);
   38:       uHex[4] = '\0';
   39:       if (sscanf(uHex, "%x", &sMap[sMapLen].u[j]) != 1) {
   40: 	error(-1, "Illegal entry in ToUnicode CMap");
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -9,7 +9,12 @@
     oldLen = mapLen;
     mapLen = (code + 256) & ~255;
-    map = (Unicode *)greallocn(map, mapLen, sizeof(Unicode));
-    for (i = oldLen; i < mapLen; ++i) {
-      map[i] = 0;
+    if (unlikely(code >= mapLen)) {
+      error(-1, "Illegal code value in CharCodeToUnicode::addMapping");
+      return;
+    } else {
+      map = (Unicode *)greallocn(map, mapLen, sizeof(Unicode));
+      for (i = oldLen; i < mapLen; ++i) {
+        map[i] = 0;
+      }
     }
   }
```

## primevul_paired:train:4738:memory_extent_relation:5437:5335:destination_capacity_definition
target=0, pair=primevul_paired:train:2368
{"pattern": "memory_extent_relation", "operation": {"start_byte": 5437, "end_byte": 5465, "lines": [175], "code": "memset(dir, 0, sizeof(*dir))", "ast_type": "call_expression"}, "evidence": {"start_byte": 5335, "end_byte": 5367, "lines": [170], "code": "dir = malloc(sizeof(struct DIR))", "ast_type": "assignment_expression"}, "role": "destination_capacity_definition", "certainty": "syntactic_relation", "safety": "unknown", "capacity": "malloc ( sizeof ( struct DIR ) )", "capacity_dimensions": [], "capacity_unit": "allocation bytes", "capacity_bytes": "unknown", "access_length": "sizeof ( * dir )", "extent_comparison": "unknown"}
```c
  168: 
  169: 			log("\nfractal head type 1 length %4" PRIX64 " name %s address %" PRIX64 "\n", length, name, heap_header_address);
  170: 
  171: 			dir = malloc(sizeof(struct DIR));
  172: 			if (!dir) {
  173: 				free(name);
  174: 				return MYSOFA_NO_MEMORY;
  175: 			}
  176: 			memset(dir, 0, sizeof(*dir));
  177: 
  178: 			dir->next = dataobject->directory;
  179: 			dataobject->directory = dir;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -15,4 +15,9 @@
 	UNUSED(block_offset);
 
+	if(reader->recursive_counter >= 10)
+		return MYSOFA_INVALID_FORMAT;
+	else
+		reader->recursive_counter++;
+
 	/* read signature */
 	if (fread(buf, 1, 4, reader->fhd) != 4 || strncmp(buf, "FHDB", 4)) {
@@ -20,5 +25,5 @@
 		return MYSOFA_INVALID_FORMAT;
 	}
-	log("%08" PRIX64 " %.4s\n", (uint64_t )ftell(reader->fhd) - 4, buf);
+	log("%08" PRIX64 " %.4s stack %d\n", (uint64_t )ftell(reader->fhd) - 4, buf, reader->recursive_counter);
 
 	if (fgetc(reader->fhd) != 0) {
@@ -200,4 +205,5 @@
 	} while (typeandversion != 0);
 
+	reader->recursive_counter--;
 	return MYSOFA_OK;
 }
```

## primevul_paired:train:4207:resource_lifetime_relation:7236:3261:allocation_to_release
target=1, pair=primevul_paired:train:2103
{"pattern": "resource_lifetime_relation", "operation": {"start_byte": 7236, "end_byte": 7253, "lines": [144], "code": "free (prop_chunk)", "ast_type": "call_expression"}, "evidence": {"start_byte": 3261, "end_byte": 3320, "lines": [76], "code": "*prop_chunk = malloc ((size_t) dff_chunk_header.ckDataSize)", "ast_type": "init_declarator"}, "role": "allocation_to_release", "certainty": "syntactic_relation", "safety": "unknown"}
```c
   74:                 error_line ("dsdiff file version = 0x%08x", version);
   75:         }
   76:         else if (!strncmp (dff_chunk_header.ckID, "PROP", 4)) {
   77:             char *prop_chunk = malloc ((size_t) dff_chunk_header.ckDataSize);
   78: 
   79:             if (!DoReadFile (infile, prop_chunk, (uint32_t) dff_chunk_header.ckDataSize, &bcount) ||
   80:                 bcount != dff_chunk_header.ckDataSize) {
  142:                             if (strncmp (cptr, "DSD ", 4)) {
  143:                                 error_line ("DSDIFF files must be uncompressed, not \"%c%c%c%c\"!",
  144:                                     cptr [0], cptr [1], cptr [2], cptr [3]);
  145:                                 free (prop_chunk);
  146:                                 return WAVPACK_SOFT_ERROR;
  147:                             }
  148: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -34,6 +34,4 @@
 #endif
 
-    // loop through all elements of the DSDIFF header
-    // (until the data chuck) and copy them to the output file
 
     while (1) {
@@ -72,11 +70,21 @@
 
             if (debug_logging_mode)
-                error_line ("dsdiff file version = 0x%08x", version);
-        }
-        else if (!strncmp (dff_chunk_header.ckID, "PROP", 4)) {
-            char *prop_chunk = malloc ((size_t) dff_chunk_header.ckDataSize);
-
-            if (!DoReadFile (infile, prop_chunk, (uint32_t) dff_chunk_header.ckDataSize, &bcount) ||
-                bcount != dff_chunk_header.ckDataSize) {
+                 error_line ("dsdiff file version = 0x%08x", version);
+         }
+         else if (!strncmp (dff_chunk_header.ckID, "PROP", 4)) {
+            char *prop_chunk;
+
+            if (dff_chunk_header.ckDataSize < 4 || dff_chunk_header.ckDataSize > 1024) {
+                error_line ("%s is not a valid .DFF file!", infilename);
+                return WAVPACK_SOFT_ERROR;
+            }
+
+            if (debug_logging_mode)
+                error_line ("got PROP chunk of %d bytes total", (int) dff_chunk_header.ckDataSize);
+
+            prop_chunk = malloc ((size_t) dff_chunk_header.ckDataSize);
+ 
+             if (!DoReadFile (infile, prop_chunk, (uint32_t) dff_chunk_header.ckDataSize, &bcount) ||
+                 bcount != dff_chunk_header.ckDataSize) {
                     error_line ("%s is not a valid .DFF file!", infilename);
                     free (prop_chunk);
```

## primevul_paired:train:5555:resource_lifetime_relation:646:274:allocation_to_release
target=1, pair=primevul_paired:train:2777
{"pattern": "resource_lifetime_relation", "operation": {"start_byte": 646, "end_byte": 664, "lines": [22], "code": "free(matvar->data)", "ast_type": "call_expression"}, "evidence": {"start_byte": 274, "end_byte": 311, "lines": [11], "code": "matvar->data = malloc(matvar->nbytes)", "ast_type": "assignment_expression"}, "role": "allocation_to_release", "certainty": "syntactic_relation", "safety": "unknown"}
```c
    9:     if ( err )
   10:         return 0;
   11: 
   12:     matvar->data = malloc(matvar->nbytes);
   13:     if ( matvar->data != NULL ) {
   14:         size_t i;
   15:         matvar_t **functions = (matvar_t **)matvar->data;
   20:                 break;
   21:         }
   22:         if ( err ) {
   23:             free(matvar->data);
   24:             matvar->data = NULL;
   25:             matvar->data_size = 0;
   26:             matvar->nbytes = 0;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -21,4 +21,8 @@
         }
         if ( err ) {
+            size_t j;
+            for ( j = 0; j < i; j++ ) {
+                free(functions[j]);
+            }
             free(matvar->data);
             matvar->data = NULL;
```

## primevul_paired:train:6761:resource_lifetime_relation:486:251:allocation_to_release
target=1, pair=primevul_paired:train:3380
{"pattern": "resource_lifetime_relation", "operation": {"start_byte": 486, "end_byte": 502, "lines": [22], "code": "free (input_u32)", "ast_type": "call_expression"}, "evidence": {"start_byte": 251, "end_byte": 315, "lines": [14], "code": "input_u32 = (uint32_t *) malloc ((inlen + 1) * sizeof(uint32_t))", "ast_type": "assignment_expression"}, "role": "allocation_to_release", "certainty": "syntactic_relation", "safety": "unknown"}
```c
   12:       return IDN2_OK;
   13:     }
   14: 
   15:   input_u32 = (uint32_t *) malloc ((inlen + 1) * sizeof(uint32_t));
   16:   if (!input_u32)
   17:     return IDN2_MALLOC;
   18: 
   20:   input_u32[inlen] = 0;
   21: 
   22:   input_u8 = u32_to_u8 (input_u32, inlen + 1, NULL, &length);
   23:   free (input_u32);
   24:   if (!input_u8)
   25:     {
   26:       if (errno == ENOMEM)
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -38,8 +38,16 @@
        * least 63 characters plus the terminating zero.
        */
+      size_t len = strlen ((char *) output_u8);
+
+      if (len > 63)
+        {
+	  free (output_u8);
+	  return IDN2_TOO_BIG_DOMAIN;
+        }
+
       if (output)
-	strcpy (output, (const char *) output_u8);
+	strcpy (output, (char *) output_u8);
 
-      free(output_u8);
+      free (output_u8);
     }
 
```

## primevul_paired:train:4385:resource_lifetime_relation:11970:11341:allocation_to_release
target=1, pair=primevul_paired:train:2192
{"pattern": "resource_lifetime_relation", "operation": {"start_byte": 11970, "end_byte": 11981, "lines": [254], "code": "free (buff)", "ast_type": "call_expression"}, "evidence": {"start_byte": 11341, "end_byte": 11370, "lines": [242], "code": "buff = malloc (bytes_to_copy)", "ast_type": "assignment_expression"}, "role": "allocation_to_release", "certainty": "syntactic_relation", "safety": "unknown"}
```c
  240:                 return WAVPACK_SOFT_ERROR;
  241:             }
  242: 
  243:             buff = malloc (bytes_to_copy);
  244: 
  245:             if (debug_logging_mode)
  246:                 error_line ("extra unknown chunk \"%c%c%c%c\" of %d bytes",
  252:                 (!(config->qmode & QMODE_NO_STORE_WRAPPER) &&
  253:                 !WavpackAddWrapper (wpc, buff, bytes_to_copy))) {
  254:                     error_line ("%s", WavpackGetErrorMessage (wpc));
  255:                     free (buff);
  256:                     return WAVPACK_SOFT_ERROR;
  257:             }
  258: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -1,8 +1,8 @@
-int ParseRiffHeaderConfig (FILE *infile, char *infilename, char *fourcc, WavpackContext *wpc, WavpackConfig *config)
-{
-    int is_rf64 = !strncmp (fourcc, "RF64", 4), got_ds64 = 0;
-    int64_t total_samples = 0, infilesize;
-    RiffChunkHeader riff_chunk_header;
-    ChunkHeader chunk_header;
+ int ParseRiffHeaderConfig (FILE *infile, char *infilename, char *fourcc, WavpackContext *wpc, WavpackConfig *config)
+ {
+    int is_rf64 = !strncmp (fourcc, "RF64", 4), got_ds64 = 0, format_chunk = 0;
+     int64_t total_samples = 0, infilesize;
+     RiffChunkHeader riff_chunk_header;
+     ChunkHeader chunk_header;
     WaveHeader WaveHeader;
     DS64Chunk ds64_chunk;
@@ -31,6 +31,4 @@
     }
 
-    // loop through all elements of the RIFF wav header
-    // (until the data chuck) and copy them to the output file
 
     while (1) {
@@ -85,10 +83,15 @@
             }
         }
-        else if (!strncmp (chunk_header.ckID, "fmt ", 4)) {     // if it's the format chunk, we want to get some info out of there and
-            int supported = TRUE, format;                        // make sure it's a .wav file we can handle
-
-            if (chunk_header.ckSize < 16 || chunk_header.ckSize > sizeof (WaveHeader) ||
-                !DoReadFile (infile, &WaveHeader, chunk_header.ckSize, &bcount) ||
-                bcount != chunk_header.ckSize) {
+         else if (!strncmp (chunk_header.ckID, "fmt ", 4)) {     // if it's the format chunk, we want to get some info out of there and
+             int supported = TRUE, format;                        // make sure it's a .wav file we can handle
+ 
+            if (format_chunk++) {
+                error_line ("%s is not a valid .WAV file!", infilename);
+                return WAVPACK_SOFT_ERROR;
+            }
+
+             if (chunk_header.ckSize < 16 || chunk_header.ckSize > sizeof (WaveHeader) ||
+                 !DoReadFile (infile, &WaveHeader, chunk_header.ckSize, &bcount) ||
+                 bcount != chunk_header.ckSize) {
                     error_line ("%s is not a valid .WAV file!", infilename);
                     return WAVPACK_SOFT_ERROR;
```

## primevul_paired:train:7648:resource_lifetime_relation:1077:653:allocation_to_release
target=0, pair=primevul_paired:train:3823
{"pattern": "resource_lifetime_relation", "operation": {"start_byte": 1077, "end_byte": 1089, "lines": [40], "code": "free(reason)", "ast_type": "call_expression"}, "evidence": {"start_byte": 653, "end_byte": 713, "lines": [26], "code": "reason = (char*) calloc(msg.fdc.reasonLen + 1, sizeof(char))", "ast_type": "assignment_expression"}, "role": "allocation_to_release", "certainty": "syntactic_relation", "safety": "unknown"}
```c
   24: 		return;
   25: 	}
   26: 	
   27: 	reason = (char*) calloc(msg.fdc.reasonLen + 1, sizeof(char));
   28: 	if(reason == NULL) {
   29: 		rfbLog("File [%s]: Method [%s]: Fatal Error: Memory alloc failed\n", 
   30: 				__FILE__, __FUNCTION__);
   38: 					"FileDownloadCancelMsg\n", __FILE__, __FUNCTION__);
   39: 		
   40: 	    rfbCloseClient(cl);
   41: 	    free(reason);
   42: 	    return;
   43: 	}
   44: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -39,4 +39,6 @@
 		
 	    rfbCloseClient(cl);
+	    free(reason);
+	    return;
 	}
 
```

## primevul_paired:train:4334:resource_lifetime_relation:3515:3386:allocation_to_release
target=0, pair=primevul_paired:train:2166
{"pattern": "resource_lifetime_relation", "operation": {"start_byte": 3515, "end_byte": 3525, "lines": [138], "code": "free (tmp)", "ast_type": "call_expression"}, "evidence": {"start_byte": 3386, "end_byte": 3423, "lines": [134], "code": "tmp = malloc (strlen (str + pos) + 1)", "ast_type": "assignment_expression"}, "role": "allocation_to_release", "certainty": "syntactic_relation", "safety": "unknown"}
```c
  132: 					op->offset_sign = -1;
  133: 				}
  134: 				char *tmp;
  135: 				tmp = malloc (strlen (str + pos) + 1);
  136: 				strcpy (tmp, str + pos);
  137: 				strtok (tmp, "+-");
  138: 				st64 read = getnum (a, tmp);
  139: 				free (tmp);
  140: 				temp *= read;
  141: 			}
  142: 		}
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -5,12 +5,9 @@
 	bool explicit_size = false;
 	int reg_index = 0;
-	// Reset type
 	op->type = 0;
-	// Consume tokens denoting the operand size
 	while (size_token) {
 		pos = nextpos;
 		last_type = getToken (str, &pos, &nextpos);
 
-		// Token may indicate size: then skip
 		if (!r_str_ncasecmp (str + pos, "ptr", 3)) {
 			continue;
@@ -44,13 +41,8 @@
 	}
 
-	// Next token: register, immediate, or '['
 	if (str[pos] == '[') {
-		// Don't care about size, if none is given.
 		if (!op->type) {
 			op->type = OT_MEMORY;
 		}
-		// At the moment, we only accept plain linear combinations:
-		// part := address | [factor *] register
-		// address := part {+ part}*
 		op->offset = op->scale[0] = op->scale[1] = 0;
 
@@ -60,5 +52,4 @@
 		while (str[pos] != ']') {
 			if (pos > nextpos) {
-			//	eprintf ("Error parsing instruction\n");
 				break;
 			}
@@ -69,22 +60,22 @@
 			last_type = getToken (str, &pos, &nextpos);
 
-			if (last_type == TT_SPECIAL) {
-				if (str[pos] == '+' || str[pos] == '-' || str[pos] == ']') {
-					if (reg != X86R_UNDEFINED) {
-						op->regs[reg_index] = reg;
-						op->scale[reg_index] = temp;
-						++reg_index;
-					} else {
-						op->offset += temp;
-						op->regs[reg_index] = X86R_UNDEFINED;
-					}
+ 			if (last_type == TT_SPECIAL) {
+ 				if (str[pos] == '+' || str[pos] == '-' || str[pos] == ']') {
+ 					if (reg != X86R_UNDEFINED) {
+						if (reg_index < 2) {
+							op->regs[reg_index] = reg;
+							op->scale[reg_index] = temp;
+						}
+ 						++reg_index;
+ 					} else {
+ 						op->offset += temp;
+						if (reg_index < 2) {
+							op->regs[reg_index] = X86R_UNDEFINED;
+						}
+ 					}
+ 					temp = 1;
+ 					reg = X86R_UNDEFINED;
+ 				} else if (str[pos] == '*') {
 
-					temp = 1;
-					reg = X86R_UNDEFINED;
-				} else if (str[pos] == '*') {
-					// go to ], + or - to get scale
... display truncated; source retained unchanged ...
```

## primevul_paired:train:790:resource_lifetime_relation:4371:3137:allocation_to_release
target=0, pair=primevul_paired:train:394
{"pattern": "resource_lifetime_relation", "operation": {"start_byte": 4371, "end_byte": 4380, "lines": [140], "code": "free(cb1)", "ast_type": "call_expression"}, "evidence": {"start_byte": 3137, "end_byte": 3178, "lines": [107], "code": "cb1 = (unsigned char *)calloc(cb1_len, 1)", "ast_type": "assignment_expression"}, "role": "allocation_to_release", "certainty": "syntactic_relation", "safety": "unknown"}
```c
  105: 		return false;
  106: 	}
  107:  
  108:  	cb1 = (unsigned char *)calloc(cb1_len, 1);
  109:  	if (unlikely(!cb1))
  110:  		quithere(1, "Failed to calloc cb1 in parse_notify");
  111:  	hex2bin(cb1, coinbase1, cb1_len);
  138: 	}
  139: 	free(coinbase1);
  140: 	free(coinbase2);
  141: 	free(cb1);
  142: 	free(cb2);
  143: 
  144: 	/* A notify message is the closest stratum gets to a getwork */
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -99,6 +99,10 @@
  		"00000000", /* nonce */
  		workpadding);
-	if (unlikely(!hex2bin(pool->header_bin, header, 128)))
-		quit(1, "Failed to convert header to header_bin in parse_notify");
+	if (unlikely(!hex2bin(pool->header_bin, header, 128))) {
+		applog(LOG_WARNING, "%s: Failed to convert header to header_bin, got %s", __func__, header);
+		pool_failed(pool);
+		// TODO: memory leaks? goto out, clean up there?
+		return false;
+	}
  
  	cb1 = (unsigned char *)calloc(cb1_len, 1);
@@ -106,8 +110,10 @@
  		quithere(1, "Failed to calloc cb1 in parse_notify");
  	hex2bin(cb1, coinbase1, cb1_len);
+
  	cb2 = (unsigned char *)calloc(cb2_len, 1);
  	if (unlikely(!cb2))
  		quithere(1, "Failed to calloc cb2 in parse_notify");
  	hex2bin(cb2, coinbase2, cb2_len);
+
  	free(pool->coinbase);
  	align_len(&alloc_len);
@@ -117,4 +123,5 @@
  	memcpy(pool->coinbase, cb1, cb1_len);
  	memcpy(pool->coinbase + cb1_len, pool->nonce1bin, pool->n1_len);
+	// NOTE: gap for nonce2, filled at work generation time
  	memcpy(pool->coinbase + cb1_len + pool->n1_len + pool->n2size, cb2, cb2_len);
  	cg_wunlock(&pool->data_lock);
```

## primevul_paired:train:7678:resource_lifetime_relation:1769:1506:allocation_to_release
target=0, pair=primevul_paired:train:3838
{"pattern": "resource_lifetime_relation", "operation": {"start_byte": 1769, "end_byte": 1790, "lines": [53], "code": "free(compressed_data)", "ast_type": "call_expression"}, "evidence": {"start_byte": 1506, "end_byte": 1558, "lines": [45], "code": "*compressed_data = (unsigned char *)malloc(page_len)", "ast_type": "init_declarator"}, "role": "allocation_to_release", "certainty": "syntactic_relation", "safety": "unknown"}
```c
   43:     }
   44:     else {
   45: 	/* Allocate memory to hold the compressed segment */
   46: 	unsigned char *compressed_data = (unsigned char *)malloc(page_len);
   47: 	size_t compressed_data_len = page_len;
   48: 	if(compressed_data==0){
   49: 	    return -2;			// memory error
   51: 
   52: 	/* Get the data */
   53: 	if(af_get_page_raw(af,pagenum,&pageflag,compressed_data,&compressed_data_len)){
   54: 	    free(compressed_data);
   55: 	    return -3;			// read error
   56: 	}
   57: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -55,4 +55,9 @@
 	    return -3;			// read error
 	}
+
+	/* Sanity check to avoid undefined behaviour when calling malloc below with pagesize from a corrupt AFF image. */
+	if(af->image_pagesize <= 0 || af->image_pagesize > 16*1024*1024)
+	    return -1;
+
 
 	/* Now uncompress directly into the buffer provided by the caller, unless the caller didn't
```

## primevul_paired:train:7939:call_argument_relation:2237:2207:computed_definition_to_argument
target=1, pair=primevul_paired:train:3969
{"pattern": "call_argument_relation", "operation": {"start_byte": 2237, "end_byte": 2245, "lines": [59], "code": "TYPE(ch)", "ast_type": "call_expression"}, "evidence": {"start_byte": 2207, "end_byte": 2223, "lines": [58], "code": "ch = CHILD(n, i)", "ast_type": "assignment_expression"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 0}
```c
   56:     /* count the number of keyword only args &
   57:        defaults for keyword only args */
   58:     for ( ; i < NCH(n); ++i) {
   59:         ch = CHILD(n, i);
   60:         if (TYPE(ch) == DOUBLESTAR) break;
   61:         if (TYPE(ch) == tfpdef || TYPE(ch) == vfpdef) nkwonlyargs++;
   62:     }
   63:     posargs = (nposargs ? _Py_asdl_seq_new(nposargs, c->c_arena) : NULL);
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -173,5 +173,5 @@
                     return NULL;
                 i += 2; /* the double star and the name */
-                if (TYPE(CHILD(n, i)) == COMMA)
+                if (i < NCH(n) && TYPE(CHILD(n, i)) == COMMA)
                     i += 1; /* the comma, if present */
                 break;
```

## primevul_paired:train:6663:call_argument_relation:657:546:computed_definition_to_argument
target=1, pair=primevul_paired:train:3331
{"pattern": "call_argument_relation", "operation": {"start_byte": 657, "end_byte": 709, "lines": [26], "code": "talloc_asprintf_append(tmp, \" INTERSECT %s \", child)", "ast_type": "call_expression"}, "evidence": {"start_byte": 546, "end_byte": 610, "lines": [23], "code": "child = parsetree_to_sql(module, mem_ctx, t->u.list.elements[i])", "ast_type": "assignment_expression"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 2}
```c
   21: 
   22: 		for (i = 1; i < t->u.list.num_elements; i++) {
   23: 
   24: 			child = parsetree_to_sql(module, mem_ctx, t->u.list.elements[i]);
   25: 			if (child == NULL) return NULL;
   26: 
   27: 			tmp = talloc_asprintf_append(tmp, " INTERSECT %s ", child);
   28: 			if (tmp == NULL) return NULL;
   29: 		}
   30: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -78,4 +78,7 @@
 						ldb_dn_new(mem_ctx, ldb,
 							      (const char *)value.data));
+			if (cdn == NULL) {
+				return NULL;
+			}
 
 			return lsqlite3_tprintf(mem_ctx,
```

## primevul_paired:train:4983:call_argument_relation:3361:2446:computed_definition_to_argument
target=1, pair=primevul_paired:train:2491
{"pattern": "call_argument_relation", "operation": {"start_byte": 3361, "end_byte": 3481, "lines": [81, 82, 83], "code": "snprintf(msg, sizeof(msg),\n                \"bad DT_HASH nbucket=%#x  len=%#x\",\n                nbucket, (v_sym - v_hsh))", "ast_type": "call_expression"}, "evidence": {"start_byte": 2446, "end_byte": 2494, "lines": [61], "code": "v_hsh = elf_unsigned_dynamic(Elf32_Dyn::DT_HASH)", "ast_type": "init_declarator"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 4}
```c
   59:         }
   60:     }
   61:     // DT_HASH often ends at DT_SYMTAB
   62:     unsigned const v_hsh = elf_unsigned_dynamic(Elf32_Dyn::DT_HASH);
   63:     if (v_hsh && file_image) {
   64:         hashtab = (unsigned const *)elf_find_dynamic(Elf32_Dyn::DT_HASH);
   65:         if (!hashtab) {
   79:                 + sizeof(*chains) *nbucket  // chains
   80:            ))
   81:         ) {
   82:             char msg[90]; snprintf(msg, sizeof(msg),
   83:                 "bad DT_HASH nbucket=%#x  len=%#x",
   84:                 nbucket, (v_sym - v_hsh));
   85:             throwCantPack(msg);
   86:         }
   87:     }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -72,6 +72,6 @@
         unsigned const *const chains = &buckets[nbucket]; (void)chains;
 
-        unsigned const v_sym = get_te32(&dynp0[-1+ x_sym].d_val);
-        if (!nbucket
+        unsigned const v_sym = !x_sym ? 0 : get_te32(&dynp0[-1+ x_sym].d_val);
+        if (!nbucket || !v_sym
         || (nbucket>>31) || (file_size/sizeof(unsigned)) <= (2*nbucket)  // FIXME: weak
         || ((v_hsh < v_sym) && (v_sym - v_hsh) < (sizeof(unsigned)*2  // headers
@@ -103,6 +103,6 @@
       //unsigned     const *const gashend = &hasharr[n_bucket];  // minimum
 
-        unsigned const v_sym = get_te32(&dynp0[-1+ x_sym].d_val);
-        if (!n_bucket || !n_bitmask
+        unsigned const v_sym = !x_sym ? 0 : get_te32(&dynp0[-1+ x_sym].d_val);
+        if (!n_bucket || !n_bitmask || !v_sym
         || (-1+ n_bitmask) & n_bitmask  // not a power of 2
         || 8*sizeof(unsigned) <= gnu_shift  // shifted result always == 0
```

## primevul_paired:train:4359:call_argument_relation:1048:924:computed_definition_to_argument
target=1, pair=primevul_paired:train:2179
{"pattern": "call_argument_relation", "operation": {"start_byte": 1048, "end_byte": 1070, "lines": [40], "code": "readUByte(&refSegs[i])", "ast_type": "call_expression"}, "evidence": {"start_byte": 924, "end_byte": 976, "lines": [37], "code": "refSegs = (Guint *)gmallocn(nRefSegs, sizeof(Guint))", "ast_type": "assignment_expression"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 0}
```c
   35:     }
   36: 
   37:     // referred-to segment numbers
   38:     refSegs = (Guint *)gmallocn(nRefSegs, sizeof(Guint));
   39:     if (segNum <= 256) {
   40:       for (i = 0; i < nRefSegs; ++i) {
   41: 	if (!readUByte(&refSegs[i])) {
   42: 	  goto eofError2;
   43: 	}
   44:       }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -9,5 +9,4 @@
   while (readULong(&segNum)) {
 
-    // segment header flags
     if (!readUByte(&segFlags)) {
       goto eofError1;
@@ -15,5 +14,4 @@
     segType = segFlags & 0x3f;
 
-    // referred-to segment count and retention flags
     if (!readUByte(&refFlags)) {
       goto eofError1;
@@ -35,5 +33,4 @@
     }
 
-    // referred-to segment numbers
     refSegs = (Guint *)gmallocn(nRefSegs, sizeof(Guint));
     if (segNum <= 256) {
@@ -57,5 +54,4 @@
     }
 
-    // segment page association
     if (segFlags & 0x40) {
       if (!readULong(&page)) {
@@ -68,13 +64,10 @@
     }
 
-    // segment data length
     if (!readULong(&segLength)) {
       goto eofError2;
     }
 
-    // keep track of the start of the segment data 
     segDataPos = curStr->getPos();
 
-    // check for missing page information segment
     if (!pageBitmap && ((segType >= 4 && segType <= 7) ||
 			(segType >= 20 && segType <= 43))) {
@@ -83,5 +76,4 @@
     }
 
-    // read the segment data
     switch (segType) {
     case 0:
@@ -160,7 +152,4 @@
     }
 
-    // Make sure the segment handler read all of the bytes in the 
-    // segment data, unless this segment is marked as having an
-    // unknown length (section 7.2.7 of the JBIG2 Final Committee Draft)
 
     if (segLength != 0xffffffff) {
@@ -169,17 +158,9 @@
       if (segExtraBytes > 0) {
 
-	// If we didn't read all of the bytes in the segment data,
-	// indicate an error, and throw away the rest of the data.
 	
-	// v.3.1.01.13 of the LuraTech PDF Compressor Server will
-	// sometimes generate an extraneous NULL byte at the end of
-	// arithmetic-coded symbol dictionary segments when numNewSyms
-	// == 0.  Segments like this often occur for blank pages.
-	
-	error(errSyntaxError, curStr->getPos(), "{0:d} extraneous byte{1:s} after segment",
-	      segExtraBytes, (segExtraBytes > 1) ? "s" : "");
-	
-	// Burn through the remaining bytes -- inefficient, but
-	// hopefully we're not doing this much
... display truncated; source retained unchanged ...
```

## primevul_paired:train:5682:call_argument_relation:926:386:computed_definition_to_argument
target=0, pair=primevul_paired:train:2840
{"pattern": "call_argument_relation", "operation": {"start_byte": 926, "end_byte": 950, "lines": [27], "code": "snmp_coexist_V2toV1(PDU)", "ast_type": "call_expression"}, "evidence": {"start_byte": 386, "end_byte": 410, "lines": [14], "code": "PDU = snmp_pdu_create(0)", "ast_type": "assignment_expression"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 0}
```c
   12:     }
   13: 
   14:     debugs(49, 5, HERE << "Called.");
   15:     PDU = snmp_pdu_create(0);
   16:     /* Allways answer on SNMPv1 */
   17:     rq->session.Version = SNMP_VERSION_1;
   18:     Community = snmp_parse(&rq->session, PDU, buf, len);
   25:         checklist.snmp_community = (char *) Community;
   26:         allow = checklist.fastCheck();
   27: 
   28:         if (allow == ACCESS_ALLOWED && (snmp_coexist_V2toV1(PDU))) {
   29:             rq->community = Community;
   30:             rq->PDU = PDU;
   31:             debugs(49, 5, "snmpAgentParse: reqid=[" << PDU->reqid << "]");
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -33,4 +33,5 @@
         } else {
             debugs(49, DBG_IMPORTANT, "WARNING: SNMP agent query DENIED from : " << rq->from);
+            snmp_free_pdu(PDU);
         }
         xfree(Community);
```

## primevul_paired:train:702:call_argument_relation:406:178:computed_definition_to_argument
target=0, pair=primevul_paired:train:350
{"pattern": "call_argument_relation", "operation": {"start_byte": 406, "end_byte": 420, "lines": [17], "code": "kfree_skb(skb)", "ast_type": "call_expression"}, "evidence": {"start_byte": 178, "end_byte": 225, "lines": [7], "code": "skb = nlmsg_new(NLMSG_DEFAULT_SIZE, GFP_KERNEL)", "ast_type": "assignment_expression"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 0}
```c
    5: 	struct xfrm_dump_info info;
    6: 	struct sk_buff *skb;
    7: 
    8: 	skb = nlmsg_new(NLMSG_DEFAULT_SIZE, GFP_KERNEL);
    9: 	if (!skb)
   10: 		return ERR_PTR(-ENOMEM);
   11: 
   15: 	info.nlmsg_flags = 0;
   16: 
   17: 	if (dump_one_policy(xp, dir, 0, &info) < 0) {
   18: 		kfree_skb(skb);
   19: 		return NULL;
   20: 	}
   21: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -1,22 +1,23 @@
-static struct sk_buff *xfrm_state_netlink(struct sk_buff *in_skb,
-					  struct xfrm_state *x, u32 seq)
- {
- 	struct xfrm_dump_info info;
- 	struct sk_buff *skb;
- 
- 	skb = nlmsg_new(NLMSG_DEFAULT_SIZE, GFP_ATOMIC);
- 	if (!skb)
+static struct sk_buff *xfrm_policy_netlink(struct sk_buff *in_skb,
+					  struct xfrm_policy *xp,
+					  int dir, u32 seq)
+{
+	struct xfrm_dump_info info;
+	struct sk_buff *skb;
+
+	skb = nlmsg_new(NLMSG_DEFAULT_SIZE, GFP_KERNEL);
+	if (!skb)
 		return ERR_PTR(-ENOMEM);
 
 	info.in_skb = in_skb;
 	info.out_skb = skb;
- 	info.nlmsg_seq = seq;
- 	info.nlmsg_flags = 0;
- 
-	if (dump_one_state(x, 0, &info)) {
- 		kfree_skb(skb);
+	info.nlmsg_seq = seq;
+	info.nlmsg_flags = 0;
+
+	if (dump_one_policy(xp, dir, 0, &info) < 0) {
+		kfree_skb(skb);
 		return NULL;
- 	}
- 
- 	return skb;
+	}
+
+	return skb;
 }
```

## primevul_paired:train:4992:call_argument_relation:469:427:computed_definition_to_argument
target=0, pair=primevul_paired:train:2495
{"pattern": "call_argument_relation", "operation": {"start_byte": 469, "end_byte": 501, "lines": [14], "code": "unlikely(control->ramsize == -1)", "ast_type": "call_expression"}, "evidence": {"start_byte": 427, "end_byte": 462, "lines": [13], "code": "control->ramsize = get_ram(control)", "ast_type": "assignment_expression"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 0}
```c
   11: 	control->flags = FLAG_SHOW_PROGRESS | FLAG_KEEP_FILES | FLAG_THRESHOLD;
   12: 	control->suffix = strdup(".lrz");
   13: 	control->compression_level = 7;
   14: 	control->ramsize = get_ram(control);
   15: 	if (unlikely(control->ramsize == -1))
   16: 		return false;
   17: 	/* for testing single CPU */
   18: 	control->threads = PROCESSORS;	/* get CPUs for LZMA */
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -10,5 +10,5 @@
 	register_outputfile(control, control->msgout);
 	control->flags = FLAG_SHOW_PROGRESS | FLAG_KEEP_FILES | FLAG_THRESHOLD;
-	control->suffix = ".lrz";
+	control->suffix = strdup(".lrz");
 	control->compression_level = 7;
 	control->ramsize = get_ram(control);
```

## primevul_paired:train:1934:call_argument_relation:1433:1252:computed_definition_to_argument
target=0, pair=primevul_paired:train:966
{"pattern": "call_argument_relation", "operation": {"start_byte": 1433, "end_byte": 1577, "lines": [36, 37, 38], "code": "av_log(NULL, AV_LOG_ERROR,\n                           \"Corrupt stream (too large pkt_len %d)\\n\",\n                           mms->asf_packet_len)", "ast_type": "call_expression"}, "evidence": {"start_byte": 1252, "end_byte": 1315, "lines": [34], "code": "mms->asf_packet_len = AV_RL32(p + sizeof(ff_asf_guid) * 2 + 64)", "ast_type": "assignment_expression"}, "role": "computed_definition_to_argument", "certainty": "syntactic_relation", "safety": "unknown", "argument": 3}
```c
   32:         if (!memcmp(p, ff_asf_file_header, sizeof(ff_asf_guid))) {
   33:             /* read packet size */
   34:             if (end - p > sizeof(ff_asf_guid) * 2 + 68) {
   35:                 mms->asf_packet_len = AV_RL32(p + sizeof(ff_asf_guid) * 2 + 64);
   36:                 if (mms->asf_packet_len <= 0 || mms->asf_packet_len > sizeof(mms->in_buffer)) {
   37:                     av_log(NULL, AV_LOG_ERROR,
   38:                            "Corrupt stream (too large pkt_len %d)\n",
   39:                            mms->asf_packet_len);
   40:                     return AVERROR_INVALIDDATA;
   41:                  }
   42:              }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -42,19 +42,24 @@
              }
          } else if (!memcmp(p, ff_asf_stream_header, sizeof(ff_asf_guid))) {
-            flags     = AV_RL16(p + sizeof(ff_asf_guid)*3 + 24);
-            stream_id = flags & 0x7F;
-            if (mms->stream_num < MMS_MAX_STREAMS &&
-                    46 + mms->stream_num * 6 < sizeof(mms->out_buffer)) {
-                mms->streams = av_fast_realloc(mms->streams,
-                                   &mms->nb_streams_allocated,
-                                   (mms->stream_num + 1) * sizeof(MMSStream));
-                if (!mms->streams)
-                    return AVERROR(ENOMEM);
-                mms->streams[mms->stream_num].id = stream_id;
-                mms->stream_num++;
-            } else {
-                av_log(NULL, AV_LOG_ERROR,
-                       "Corrupt stream (too many A/V streams)\n");
-                return AVERROR_INVALIDDATA;
+            if (end - p >= (sizeof(ff_asf_guid) * 3 + 26)) {
+                flags     = AV_RL16(p + sizeof(ff_asf_guid)*3 + 24);
+                stream_id = flags & 0x7F;
+                //The second condition is for checking CS_PKT_STREAM_ID_REQUEST packet size,
+                //we can calculate the packet size by stream_num.
+                //Please see function send_stream_selection_request().
+                if (mms->stream_num < MMS_MAX_STREAMS &&
+                        46 + mms->stream_num * 6 < sizeof(mms->out_buffer)) {
+                    mms->streams = av_fast_realloc(mms->streams,
+                                       &mms->nb_streams_allocated,
+                                       (mms->stream_num + 1) * sizeof(MMSStream));
+                    if (!mms->streams)
+                        return AVERROR(ENOMEM);
+                    mms->streams[mms->stream_num].id = stream_id;
+                    mms->stream_num++;
+                } else {
+                    av_log(NULL, AV_LOG_ERROR,
+                           "Corrupt stream (too many A/V streams)\n");
+                    return AVERROR_INVALIDDATA;
+                }
              }
          } else if (!memcmp(p, ff_asf_ext_stream_header, sizeof(ff_asf_guid))) {
@@ -88,4 +93,10 @@
          } else if (!memcmp(p, ff_asf_head1_guid, sizeof(ff_asf_guid))) {
              chunksize = 46; // see references [2] section 3.4. This should be set 46.
+            if (chunksize > end - p) {
+                av_log(NULL, AV_LOG_ERROR,
+                    "Corrupt stream (header chunksize %"PRId64" is invalid)\n",
+                    chunksize);
+                return AVERROR_INVALIDDATA;
+            }
          }
          p += chunksize;
```

## primevul_paired:train:6525:control_call_relation:2656:2309:enclosing_branch
target=1, pair=primevul_paired:train:3262
{"pattern": "control_call_relation", "operation": {"start_byte": 2656, "end_byte": 2680, "lines": [69], "code": "IOVEC_MAKE((char*) p, l)", "ast_type": "call_expression"}, "evidence": {"start_byte": 2309, "end_byte": 2347, "lines": [62], "code": "(journal_field_valid(p, q - p, false))", "ast_type": "parenthesized_expression"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   60: 
   61:                 q = memchr(p, '=', e - p);
   62:                 if (q) {
   63:                         if (journal_field_valid(p, q - p, false)) {
   64:                                 size_t l;
   65: 
   66:                                 l = e - p;
   67: 
   68:                                 /* If the field name starts with an underscore, skip the variable, since that indicates
   69:                                  * a trusted field */
   70:                                 iovec[n++] = IOVEC_MAKE((char*) p, l);
   71:                                 entry_size += l;
   72: 
   73:                                 server_process_entry_meta(p, l, ucred,
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -49,4 +49,9 @@
 
                 /* A property follows */
+                if (n > ENTRY_FIELD_COUNT_MAX) {
+                        log_debug("Received an entry that has more than " STRINGIFY(ENTRY_FIELD_COUNT_MAX) " fields, ignoring entry.");
+                        r = 1;
+                        goto finish;
+                }
 
                 /* n existing properties, 1 new, +1 for _TRANSPORT */
```

## primevul_paired:train:7399:control_call_relation:2212:2061:enclosing_branch
target=1, pair=primevul_paired:train:3699
{"pattern": "control_call_relation", "operation": {"start_byte": 2212, "end_byte": 2230, "lines": [67], "code": "unlock_page(hpage)", "ast_type": "call_expression"}, "evidence": {"start_byte": 2061, "end_byte": 2165, "lines": [64, 65], "code": "((hwpoison_filter(p) && TestClearPageHWPoison(p))\n\t\t\t\t    || (p != hpage && TestSetPageHWPoison(hpage)))", "ast_type": "parenthesized_expression"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   62: 			 */
   63: 			lock_page(hpage);
   64: 			if (PageHWPoison(hpage)) {
   65: 				if ((hwpoison_filter(p) && TestClearPageHWPoison(p))
   66: 				    || (p != hpage && TestSetPageHWPoison(hpage))) {
   67: 					num_poisoned_pages_sub(nr_pages);
   68: 					unlock_page(hpage);
   69: 					return 0;
   70: 				}
   71: 			}
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -83,8 +83,8 @@
 
 	if (!PageHuge(p) && PageTransHuge(hpage)) {
-		lock_page(hpage);
-		if (!PageAnon(hpage) || unlikely(split_huge_page(hpage))) {
-			unlock_page(hpage);
-			if (!PageAnon(hpage))
+		lock_page(p);
+		if (!PageAnon(p) || unlikely(split_huge_page(p))) {
+			unlock_page(p);
+			if (!PageAnon(p))
 				pr_err("Memory failure: %#lx: non anonymous thp\n",
 					pfn);
@@ -97,7 +97,5 @@
 			return -EBUSY;
 		}
-		unlock_page(hpage);
-		get_hwpoison_page(p);
-		put_hwpoison_page(hpage);
+		unlock_page(p);
 		VM_BUG_ON_PAGE(!page_count(p), p);
 		hpage = compound_head(p);
```

## primevul_paired:train:3095:control_call_relation:560:493:enclosing_branch
target=1, pair=primevul_paired:train:1547
{"pattern": "control_call_relation", "operation": {"start_byte": 560, "end_byte": 581, "lines": [14], "code": "old_caches_.erase(it)", "ast_type": "call_expression"}, "evidence": {"start_byte": 493, "end_byte": 518, "lines": [12], "code": "(it != old_caches_.end())", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   10: 
   11:     Caches::iterator it =
   12:         std::find(old_caches_.begin(), old_caches_.end(), cache);
   13:     if (it != old_caches_.end()) {
   14:       AppCache* tmp_cache = *it;
   15:       old_caches_.erase(it);
   16:       tmp_cache->set_owning_group(nullptr);  // may cause group to be released
   17:     }
   18: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -2,7 +2,7 @@
    DCHECK(cache->associated_hosts().empty());
    if (cache == newest_complete_cache_) {
-    CancelUpdate();
      AppCache* tmp_cache = newest_complete_cache_;
      newest_complete_cache_ = nullptr;
+    CancelUpdate();
      tmp_cache->set_owning_group(nullptr);  // may cause this group to be deleted
    } else {
```

## primevul_paired:train:4789:control_call_relation:1132:1104:enclosing_branch
target=1, pair=primevul_paired:train:2394
{"pattern": "control_call_relation", "operation": {"start_byte": 1132, "end_byte": 1161, "lines": [41], "code": "LOG_FUNC_RETURN(card->ctx, r)", "ast_type": "call_expression"}, "evidence": {"start_byte": 1104, "end_byte": 1111, "lines": [39], "code": "(r < 1)", "ast_type": "parenthesized_expression"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   37: 	}
   38: 
   39: 	r = iso_ops->read_binary(card, 2, (unsigned char *)*tname, buf[1], 0);
   40: 	if (r < 1) {
   41: 		free(*tname);
   42: 		LOG_FUNC_RETURN(card->ctx, r);
   43: 	}
   44: 
   45: 	if ((*tname)[r-1] != '\0') {
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -5,4 +5,5 @@
 	sc_file_t *file = NULL;
 	u8 buf[2];
+	char *name;
 	int r;
 
@@ -32,18 +33,20 @@
 	sc_file_free(file);
 
-	*tname = malloc(buf[1]);
-	if (*tname == NULL) {
+	name = malloc(buf[1]);
+	if (name == NULL) {
 		LOG_FUNC_RETURN(card->ctx, SC_ERROR_OUT_OF_MEMORY);
 	}
 
-	r = iso_ops->read_binary(card, 2, (unsigned char *)*tname, buf[1], 0);
+	r = iso_ops->read_binary(card, 2, (unsigned char *)name, buf[1], 0);
 	if (r < 1) {
-		free(*tname);
+		free(name);
 		LOG_FUNC_RETURN(card->ctx, r);
 	}
 
-	if ((*tname)[r-1] != '\0') {
-		(*tname)[r-1] = '\0';
+	if (name[r-1] != '\0') {
+		name[r-1] = '\0';
 	}
+	*tname = name;
+
 	LOG_FUNC_RETURN(card->ctx, SC_SUCCESS);
 }
```

## primevul_paired:train:6502:control_call_relation:1039:956:early_return
target=0, pair=primevul_paired:train:3250
{"pattern": "control_call_relation", "operation": {"start_byte": 1039, "end_byte": 1061, "lines": [37], "code": "node_attach(node, val)", "ast_type": "call_expression"}, "evidence": {"start_byte": 956, "end_byte": 962, "lines": [32], "code": "(!val)", "ast_type": "parenthesized_expression"}, "role": "early_return", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false}
```c
   30: 
   31:         /* process value node */
   32:         plist_t val = parse_bin_node_at_index(bplist, index1);
   33:         if (!val) {
   34:             plist_free(node);
   35:             return NULL;
   36:         }
   37: 
   38:         node_attach(node, val);
   39:     }
   40: 
   41:     return node;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -2,8 +2,9 @@
 {
     uint64_t j;
-    uint32_t str_j = 0;
-    uint32_t index1;
-
+    uint64_t str_j = 0;
+    uint64_t index1;
     plist_data_t data = plist_new_plist_data();
+    const char *const end_data = bplist->data + bplist->size;
+    const char *index1_ptr = NULL;
 
     data->type = PLIST_ARRAY;
@@ -14,5 +15,12 @@
     for (j = 0; j < data->length; j++) {
         str_j = j * bplist->ref_size;
-        index1 = UINT_TO_HOST((*bnode) + str_j, bplist->ref_size);
+        index1_ptr = (*bnode) + str_j;
+
+        if (index1_ptr < bplist->data || index1_ptr + bplist->ref_size >= end_data) {
+            plist_free(node);
+            return NULL;
+        }
+
+        index1 = UINT_TO_HOST(index1_ptr, bplist->ref_size);
 
         if (index1 >= bplist->num_objects) {
```

## primevul_paired:train:5978:control_call_relation:4535:4525:enclosing_branch
target=0, pair=primevul_paired:train:2988
{"pattern": "control_call_relation", "operation": {"start_byte": 4535, "end_byte": 4558, "lines": [144], "code": "TORRENT_FAIL_BDECODE(e)", "ast_type": "call_expression"}, "evidence": {"start_byte": 4525, "end_byte": 4528, "lines": [143], "code": "(e)", "ast_type": "condition_clause"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
  141: 					if (start >= end) TORRENT_FAIL_BDECODE(bdecode_errors::unexpected_eof);
  142: 					bdecode_errors::error_code_enum e = bdecode_errors::no_error;
  143: 					start = parse_int(start, end, ':', len, e);
  144: 					if (e)
  145: 						TORRENT_FAIL_BDECODE(e);
  146: 
  147: 					// remaining buffer size excluding ':'
  148: 					const ptrdiff_t buff_size = end - start - 1;
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -139,4 +139,5 @@
 					char const* str_start = start;
 					++start;
+					if (start >= end) TORRENT_FAIL_BDECODE(bdecode_errors::unexpected_eof);
 					bdecode_errors::error_code_enum e = bdecode_errors::no_error;
 					start = parse_int(start, end, ':', len, e);
```

## primevul_paired:train:4208:control_call_relation:1916:1752:enclosing_branch
target=0, pair=primevul_paired:train:2103
{"pattern": "control_call_relation", "operation": {"start_byte": 1916, "end_byte": 1944, "lines": [44], "code": "WavpackGetErrorMessage (wpc)", "ast_type": "call_expression"}, "evidence": {"start_byte": 1752, "end_byte": 1879, "lines": [42, 43], "code": "(!(config->qmode & QMODE_NO_STORE_WRAPPER) &&\n            !WavpackAddWrapper (wpc, &dff_chunk_header, sizeof (DFFChunkHeader)))", "ast_type": "parenthesized_expression"}, "role": "enclosing_branch", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": true}
```c
   40:                 error_line ("%s is not a valid .DFF file!", infilename);
   41:                 return WAVPACK_SOFT_ERROR;
   42:         }
   43:         else if (!(config->qmode & QMODE_NO_STORE_WRAPPER) &&
   44:             !WavpackAddWrapper (wpc, &dff_chunk_header, sizeof (DFFChunkHeader))) {
   45:                 error_line ("%s", WavpackGetErrorMessage (wpc));
   46:                 return WAVPACK_SOFT_ERROR;
   47:         }
   48: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -34,6 +34,4 @@
 #endif
 
-    // loop through all elements of the DSDIFF header
-    // (until the data chuck) and copy them to the output file
 
     while (1) {
@@ -72,11 +70,21 @@
 
             if (debug_logging_mode)
-                error_line ("dsdiff file version = 0x%08x", version);
-        }
-        else if (!strncmp (dff_chunk_header.ckID, "PROP", 4)) {
-            char *prop_chunk = malloc ((size_t) dff_chunk_header.ckDataSize);
-
-            if (!DoReadFile (infile, prop_chunk, (uint32_t) dff_chunk_header.ckDataSize, &bcount) ||
-                bcount != dff_chunk_header.ckDataSize) {
+                 error_line ("dsdiff file version = 0x%08x", version);
+         }
+         else if (!strncmp (dff_chunk_header.ckID, "PROP", 4)) {
+            char *prop_chunk;
+
+            if (dff_chunk_header.ckDataSize < 4 || dff_chunk_header.ckDataSize > 1024) {
+                error_line ("%s is not a valid .DFF file!", infilename);
+                return WAVPACK_SOFT_ERROR;
+            }
+
+            if (debug_logging_mode)
+                error_line ("got PROP chunk of %d bytes total", (int) dff_chunk_header.ckDataSize);
+
+            prop_chunk = malloc ((size_t) dff_chunk_header.ckDataSize);
+ 
+             if (!DoReadFile (infile, prop_chunk, (uint32_t) dff_chunk_header.ckDataSize, &bcount) ||
+                 bcount != dff_chunk_header.ckDataSize) {
                     error_line ("%s is not a valid .DFF file!", infilename);
                     free (prop_chunk);
```

## primevul_paired:train:3004:control_call_relation:749:333:early_return
target=0, pair=primevul_paired:train:1501
{"pattern": "control_call_relation", "operation": {"start_byte": 749, "end_byte": 781, "lines": [19], "code": "ExtensionFrameHelper::Get(frame)", "ast_type": "call_expression"}, "evidence": {"start_byte": 333, "end_byte": 341, "lines": [9], "code": "(!frame)", "ast_type": "condition_clause"}, "role": "early_return", "certainty": "syntactic_relation", "safety": "unknown", "branch_truth": false}
```c
    7:   int frame_id = args[0]->Int32Value();
    8: 
    9:   content::RenderFrame* frame = content::RenderFrame::FromRoutingID(frame_id);
   10:   if (!frame) {
   11:     LOG(WARNING) << "No render frame found to register LoadWatcher.";
   12:      return;
   13:    }
   17:   base::Callback<void(bool)> callback(
   18:       base::Bind(&RenderFrameObserverNatives::InvokeCallback,
   19:                  weak_ptr_factory_.GetWeakPtr(), base::Passed(&v8_callback)));
   20:   if (ExtensionFrameHelper::Get(frame)->did_create_current_document_element()) {
   21:     // If the document element is already created, then we can call the callback
   22:     // immediately (though use PostTask to ensure that the callback is called
   23:     // asynchronously).
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -13,5 +13,18 @@
    }
  
-  new LoadWatcher(context(), frame, args[1].As<v8::Function>());
+  v8::Global<v8::Function> v8_callback(context()->isolate(),
+                                       args[1].As<v8::Function>());
+  base::Callback<void(bool)> callback(
+      base::Bind(&RenderFrameObserverNatives::InvokeCallback,
+                 weak_ptr_factory_.GetWeakPtr(), base::Passed(&v8_callback)));
+  if (ExtensionFrameHelper::Get(frame)->did_create_current_document_element()) {
+    // If the document element is already created, then we can call the callback
+    // immediately (though use PostTask to ensure that the callback is called
+    // asynchronously).
+    base::MessageLoop::current()->PostTask(FROM_HERE,
+                                           base::Bind(callback, true));
+  } else {
+    new LoadWatcher(frame, callback);
+  }
  
    args.GetReturnValue().Set(true);
```

## primevul_paired:train:8061:ok:no_candidate
target=1, pair=primevul_paired:train:4030
```c
    1: set_text_distance(gs_point *pdist, double dx, double dy, const gs_matrix *pmat)
    2: {
    3:     int code = gs_distance_transform_inverse(dx, dy, pmat, pdist);
    4:     double rounded;
    5: 
    6:     if (code == gs_error_undefinedresult) {
    7:         /* The CTM is degenerate.
    8:            Can't know the distance in user space.
    9:            Set zero because we believe it is not important for rendering.
   10:            We want to copy the text to PDF to make it searchable.
   11:            Bug 689006.
   12:          */
   13:         pdist->x = pdist->y = 0;
   14:     } else if (code < 0)
   15:         return code;
   16:     /* If the distance is very close to integers, round it. */
   17:     if (fabs(pdist->x - (rounded = floor(pdist->x + 0.5))) < 0.0005)
   18:         pdist->x = rounded;
   19:     if (fabs(pdist->y - (rounded = floor(pdist->y + 0.5))) < 0.0005)
   20:         pdist->y = rounded;
   21:     return 0;
   22: }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -1,6 +1,11 @@
 set_text_distance(gs_point *pdist, double dx, double dy, const gs_matrix *pmat)
 {
-    int code = gs_distance_transform_inverse(dx, dy, pmat, pdist);
+    int code;
     double rounded;
+
+    if (dx > 1e38 || dy > 1e38)
+        code = gs_error_undefinedresult;
+    else
+        code = gs_distance_transform_inverse(dx, dy, pmat, pdist);
 
     if (code == gs_error_undefinedresult) {
```

## primevul_paired:train:2093:ok:no_candidate
target=1, pair=primevul_paired:train:1046
```c
    1:  static int sock_close(struct inode *inode, struct file *filp)
    2:  {
    3: 	sock_release(SOCKET_I(inode));
    4:  	return 0;
    5:  }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -1,5 +1,5 @@
  static int sock_close(struct inode *inode, struct file *filp)
  {
-	sock_release(SOCKET_I(inode));
+	__sock_release(SOCKET_I(inode), inode);
  	return 0;
  }
```

## primevul_paired:train:2696:ok:no_candidate
target=0, pair=primevul_paired:train:1347
```c
    1:  NativeBrowserFrame* NativeBrowserFrame::CreateNativeBrowserFrame(
    2:      BrowserFrame* browser_frame,
    3:      BrowserView* browser_view) {
    4:   if (views::Widget::IsPureViews() &&
    5:       views::ViewsDelegate::views_delegate->GetDefaultParentView())
    6:      return new BrowserFrameViews(browser_frame, browser_view);
    7:    return new BrowserFrameGtk(browser_frame, browser_view);
    8:  }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -2,5 +2,6 @@
      BrowserFrame* browser_frame,
      BrowserView* browser_view) {
-  if (views::Widget::IsPureViews())
+  if (views::Widget::IsPureViews() &&
+      views::ViewsDelegate::views_delegate->GetDefaultParentView())
      return new BrowserFrameViews(browser_frame, browser_view);
    return new BrowserFrameGtk(browser_frame, browser_view);
```

## primevul_paired:train:6454:ok:no_candidate
target=0, pair=primevul_paired:train:3226
```c
    1: void json_object_seed(size_t seed) {
    2:     uint32_t new_seed = (uint32_t)seed;
    3: 
    4:     if (hashtable_seed == 0) {
    5:         if (__atomic_test_and_set(&seed_initialized, __ATOMIC_RELAXED) == 0) {
    6:             /* Do the seeding ourselves */
    7:             if (new_seed == 0)
    8:                 new_seed = generate_seed();
    9: 
   10:             __atomic_store_n(&hashtable_seed, new_seed, __ATOMIC_RELEASE);
   11:         } else {
   12:             /* Wait for another thread to do the seeding */
   13:             do {
   14: #ifdef HAVE_SCHED_YIELD
   15:                 sched_yield();
   16: #endif
   17:             } while(__atomic_load_n(&hashtable_seed, __ATOMIC_ACQUIRE) == 0);
   18:         }
   19:     }
   20: }
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -8,5 +8,5 @@
                 new_seed = generate_seed();
 
-            __atomic_store_n(&hashtable_seed, new_seed, __ATOMIC_ACQ_REL);
+            __atomic_store_n(&hashtable_seed, new_seed, __ATOMIC_RELEASE);
         } else {
             /* Wait for another thread to do the seeding */
```

## primevul_paired:train:3923:parse_unknown:no_candidate
target=1, pair=primevul_paired:train:1961
```c
    1: PHP_FUNCTION(openssl_seal)
    2: {
    3: 	zval *pubkeys, *pubkey, *sealdata, *ekeys, *iv = NULL;
    4: 	HashTable *pubkeysht;
    5: 	EVP_PKEY **pkeys;
    6: 	zend_resource ** key_resources;	/* so we know what to cleanup */
    7: 	int i, len1, len2, *eksl, nkeys, iv_len;
    8: 	unsigned char iv_buf[EVP_MAX_IV_LENGTH + 1], *buf = NULL, **eks;
    9: 	char * data;
   10: 	size_t data_len;
   11: 	char *method =NULL;
   12: 	size_t method_len = 0;
   13: 	const EVP_CIPHER *cipher;
   14: 	EVP_CIPHER_CTX *ctx;
   15: 
   16: 	if (zend_parse_parameters(ZEND_NUM_ARGS(), "sz/z/a/|sz/", &data, &data_len,
   17: 				&sealdata, &ekeys, &pubkeys, &method, &method_len, &iv) == FAILURE) {
   18: 		return;
   19: 	}
   20: 	pubkeysht = Z_ARRVAL_P(pubkeys);
   21: 	nkeys = pubkeysht ? zend_hash_num_elements(pubkeysht) : 0;
   22: 	if (!nkeys) {
   23: 		php_error_docref(NULL, E_WARNING, "Fourth argument to openssl_seal() must be a non-empty array");
   24: 		RETURN_FALSE;
   25: 	}
   26: 
   27: 	PHP_OPENSSL_CHECK_SIZE_T_TO_INT(data_len, data);
   28: 
   29: 	if (method) {
   30: 		cipher = EVP_get_cipherbyname(method);
   31: 		if (!cipher) {
   32: 			php_error_docref(NULL, E_WARNING, "Unknown signature algorithm.");
   33: 			RETURN_FALSE;
   34: 		}
   35: 	} else {
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -77,5 +77,5 @@
         EVP_CIPHER_CTX_cleanup(ctx);
  
-       if (!EVP_SealInit(ctx, cipher, eks, eksl, &iv_buf[0], pkeys, nkeys) ||
+       if (EVP_SealInit(ctx, cipher, eks, eksl, &iv_buf[0], pkeys, nkeys) <= 0 ||
                         !EVP_SealUpdate(ctx, buf, &len1, (unsigned char *)data, (int)data_len) ||
                         !EVP_SealFinal(ctx, buf + len1, &len2)) {
```

## primevul_paired:train:7583:parse_unknown:no_candidate
target=1, pair=primevul_paired:train:3791
```c
    1: int LibRaw::unpack(void)
    2: {
    3:     CHECK_ORDER_HIGH(LIBRAW_PROGRESS_LOAD_RAW);
    4:     CHECK_ORDER_LOW(LIBRAW_PROGRESS_IDENTIFY);
    5:     try {
    6: 
    7:         RUN_CALLBACK(LIBRAW_PROGRESS_LOAD_RAW,0,2);
    8:         if (O.shot_select >= P1.raw_count)
    9:             return LIBRAW_REQUEST_FOR_NONEXISTENT_IMAGE;
   10:         
   11:         if(!load_raw)
   12:             return LIBRAW_UNSPECIFIED_ERROR;
   13:         
   14:         if (O.use_camera_matrix && C.cmatrix[0][0] > 0.25) 
   15:             {
   16:                 memcpy (C.rgb_cam, C.cmatrix, sizeof (C.cmatrix));
   17:                 IO.raw_color = 0;
   18:             }
   19:         // already allocated ?
   20:         if(imgdata.image)
   21:             {
   22:                 free(imgdata.image);
   23:                 imgdata.image = 0;
   24:             }
   25: 
   26:         if (libraw_internal_data.unpacker_data.meta_length) 
   27:             {
   28:                 libraw_internal_data.internal_data.meta_data = 
   29:                     (char *) malloc (libraw_internal_data.unpacker_data.meta_length);
   30:                 merror (libraw_internal_data.internal_data.meta_data, "LibRaw::unpack()");
   31:             }
   32:         ID.input->seek(libraw_internal_data.unpacker_data.data_offset, SEEK_SET);
   33:         int save_document_mode = O.document_mode;
   34:         O.document_mode = 0;
   35: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -70,6 +70,6 @@
                 IO.shrink = 0;
                 // allocate image as temporary buffer, size 
-                imgdata.rawdata.raw_alloc = calloc(S.iwidth*S.iheight,sizeof(*imgdata.image));
-                imgdata.image = (ushort (*)[4]) imgdata.rawdata.raw_alloc;
+                imgdata.rawdata.raw_alloc = 0;
+                imgdata.image = (ushort (*)[4]) calloc(S.iwidth*S.iheight,sizeof(*imgdata.image));
             }
 
@@ -81,6 +81,6 @@
         if( decoder_info.decoder_flags & LIBRAW_DECODER_LEGACY)
             {
-                imgdata.image = 0; 
-                imgdata.rawdata.color_image = (ushort (*)[4]) imgdata.rawdata.raw_alloc;
+              imgdata.rawdata.raw_alloc = imgdata.rawdata.color_image = imgdata.image;
+              imgdata.image = 0; 
             }
 
```

## primevul_paired:train:1976:parse_unknown:no_candidate
target=0, pair=primevul_paired:train:987
```c
    1: static int chmd_read_headers(struct mspack_system *sys, struct mspack_file *fh,
    2: 			     struct mschmd_header *chm, int entire)
    3: {
    4:   unsigned int section, name_len, x, errors, num_chunks;
    5:   unsigned char buf[0x54], *chunk = NULL, *name, *p, *end;
    6:   struct mschmd_file *fi, *link = NULL;
    7:   off_t offset, length;
    8:   int num_entries;
    9: 
   10:   /* initialise pointers */
   11:   chm->files         = NULL;
   12:   chm->sysfiles      = NULL;
   13:   chm->chunk_cache   = NULL;
   14:   chm->sec0.base.chm = chm;
   15:   chm->sec0.base.id  = 0;
   16:   chm->sec1.base.chm = chm;
   17:   chm->sec1.base.id  = 1;
   18:   chm->sec1.content  = NULL;
   19:   chm->sec1.control  = NULL;
   20:   chm->sec1.spaninfo = NULL;
   21:   chm->sec1.rtable   = NULL;
   22: 
   23:   /* read the first header */
   24:   if (sys->read(fh, &buf[0], chmhead_SIZEOF) != chmhead_SIZEOF) {
   25:     return MSPACK_ERR_READ;
   26:   }
   27: 
   28:   /* check ITSF signature */
   29:   if (EndGetI32(&buf[chmhead_Signature]) != 0x46535449) {
   30:     return MSPACK_ERR_SIGNATURE;
   31:   }
   32: 
   33:   /* check both header GUIDs */
   34:   if (mspack_memcmp(&buf[chmhead_GUID1], &guids[0], 32L) != 0) {
   35:     D(("incorrect GUIDs"))
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -187,6 +187,4 @@
        READ_ENCINT(name_len);
        if (name_len > (unsigned int) (end - p)) goto chunk_end;
-      /* consider blank filenames to be an error */
-      if (name_len == 0) goto chunk_end;
        name = p; p += name_len;
        READ_ENCINT(section);
@@ -194,4 +192,7 @@
        READ_ENCINT(length);
  
+      /* ignore blank or one-char (e.g. "/") filenames we'd return as blank */
+      if (name_len < 2 || !name[0] || !name[1]) continue;
+
        /* empty files and directory names are stored as a file entry at
         * offset 0 with length 0. We want to keep empty files, but not
```

## primevul_paired:train:6750:parse_unknown:no_candidate
target=0, pair=primevul_paired:train:3374
```c
    1: InitializeCpuMpWorker (
    2:   IN CONST EFI_PEI_SERVICES     **PeiServices
    3:   )
    4: {
    5:   EFI_STATUS                      Status;
    6:   EFI_VECTOR_HANDOFF_INFO         *VectorInfo;
    7:   EFI_PEI_VECTOR_HANDOFF_INFO_PPI *VectorHandoffInfoPpi;
    8: 
    9:   //
   10:   // Get Vector Hand-off Info PPI
   11:   //
   12:   VectorInfo = NULL;
   13:   Status = PeiServicesLocatePpi (
   14:              &gEfiVectorHandoffInfoPpiGuid,
   15:              0,
   16:              NULL,
   17:              (VOID **)&VectorHandoffInfoPpi
   18:              );
   19:   if (Status == EFI_SUCCESS) {
   20:     VectorInfo = VectorHandoffInfoPpi->Info;
   21:   }
   22: 
   23:   //
   24:   // Initialize default handlers
   25:   //
   26:   Status = InitializeCpuExceptionHandlers (VectorInfo);
   27:   if (EFI_ERROR (Status)) {
   28:     return Status;
   29:   }
   30: 
   31:   Status = MpInitLibInitialize ();
   32:   if (EFI_ERROR (Status)) {
   33:     return Status;
   34:   }
   35: 
```
Pair edit (structural context only; never extractor input):
```diff
--- 
+++ 
@@ -1,8 +1,7 @@
-CpuMpPeimInit (
-  IN       EFI_PEI_FILE_HANDLE  FileHandle,
+InitializeCpuMpWorker (
   IN CONST EFI_PEI_SERVICES     **PeiServices
   )
 {
-  EFI_STATUS           Status;
+  EFI_STATUS                      Status;
   EFI_VECTOR_HANDOFF_INFO         *VectorInfo;
   EFI_PEI_VECTOR_HANDOFF_INFO_PPI *VectorHandoffInfoPpi;
@@ -21,12 +20,22 @@
     VectorInfo = VectorHandoffInfoPpi->Info;
   }
-  Status = InitializeCpuExceptionHandlers (VectorInfo);
-  ASSERT_EFI_ERROR (Status);
 
   //
-  // Wakeup APs to do initialization
+  // Initialize default handlers
   //
+  Status = InitializeCpuExceptionHandlers (VectorInfo);
+  if (EFI_ERROR (Status)) {
+    return Status;
+  }
+
   Status = MpInitLibInitialize ();
-  ASSERT_EFI_ERROR (Status);
+  if (EFI_ERROR (Status)) {
+    return Status;
+  }
+
+  //
+  // Special initialization for the sake of Stack Guard
+  //
+  InitializeMpExceptionStackSwitchHandlers ();
 
   //
```
