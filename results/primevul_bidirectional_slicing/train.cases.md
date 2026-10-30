# Fixed slicing source review

0-based stored / 1-based displayed physical rows. Assistant review, not independent vulnerability gold.

## primevul_paired:train:1595 / addition_only / legal
```diff
--- 
+++ 
@@ -6,4 +6,5 @@
 //// On Chrome OS, there is no component that can validate these extended
 //// attributes so there is no need to set them.
+#if !defined(OS_CHROMEOS)
    base::ScopedBlockingCall scoped_blocking_call(base::BlockingType::MAY_BLOCK);
    int result = setxattr(path, name, value, value_size, flags);
@@ -13,4 +14,5 @@
      return false;
    }
+#endif  // !defined(OS_CHROMEOS)
    return true;
  }
```
### primevul_paired:train:3191
```c
    6: //// On Chrome OS, there is no component that can validate these extended
    7: //// attributes so there is no need to set them.
    8:    base::ScopedBlockingCall scoped_blocking_call(base::BlockingType::MAY_BLOCK);
    9:    int result = setxattr(path, name, value, value_size, flags);
   10:    if (result) {
   11:      DPLOG(ERROR) << "Could not set extended attribute " << name << " on file "
   12:                   << path;
```
{"review_id": "primevul_paired:train:3191:line:8", "sample_key": "primevul_paired:train:3191", "pair_key": "primevul_paired:train:1595", "line": 8, "target": 1, "shape": "addition_only", "slice_status": "ok", "quality_status": "legal", "methods": ["losver"], "code": "   int result = setxattr(path, name, value, value_size, flags);"}
### primevul_paired:train:3192
```c
    5:                                int flags) {
    6: //// On Chrome OS, there is no component that can validate these extended
    7: //// attributes so there is no need to set them.
    8: #if !defined(OS_CHROMEOS)
    9:    base::ScopedBlockingCall scoped_blocking_call(base::BlockingType::MAY_BLOCK);
   10:    int result = setxattr(path, name, value, value_size, flags);
   11:    if (result) {
   13:                   << path;
   14:      return false;
   15:    }
   16: #endif  // !defined(OS_CHROMEOS)
   17:    return true;
   18:  }
```
{"review_id": "primevul_paired:train:3192:line:7", "sample_key": "primevul_paired:train:3192", "pair_key": "primevul_paired:train:1595", "line": 7, "target": 0, "shape": "addition_only", "slice_status": "ok", "quality_status": "legal", "methods": ["original_diff", "losver"], "code": "#if !defined(OS_CHROMEOS)"}
{"review_id": "primevul_paired:train:3192:line:15", "sample_key": "primevul_paired:train:3192", "pair_key": "primevul_paired:train:1595", "line": 15, "target": 0, "shape": "addition_only", "slice_status": "ok", "quality_status": "legal", "methods": ["original_diff", "losver"], "code": "#endif  // !defined(OS_CHROMEOS)"}

## primevul_paired:train:1190 / addition_only / legal
```diff
--- 
+++ 
@@ -349,4 +349,6 @@
        return(image);
      }
+  if (((MagickSizeType) image->columns*image->rows) > GetBlobSize(image))
+    ThrowReaderException(CorruptImageError,"InsufficientImageDataInFile");
    for ( ; offset < (MagickOffsetType) cin.file.image_offset; offset++)
    {
```
### primevul_paired:train:2381
```c
  329:       /*
  330:         User defined data.
  331:       */
  332:       if (cin.file.user_length > GetBlobSize(image))
  333:         ThrowReaderException(CorruptImageError,"InsufficientImageDataInFile");
  334:       profile=BlobToStringInfo((const unsigned char *) NULL,
  335:         cin.file.user_length);
  336:       if (profile == (StringInfo *) NULL)
  337:         ThrowReaderException(ResourceLimitError,"MemoryAllocationFailed");
  338:       offset+=ReadBlob(image,GetStringInfoLength(profile),
  366:   /*
  367:     Convert CIN raster image to pixel packets.
  368:   */
  369:   quantum_info=AcquireQuantumInfo(image_info,image);
  370:   if (quantum_info == (QuantumInfo *) NULL)
  371:     ThrowReaderException(ResourceLimitError,"MemoryAllocationFailed");
  372:   quantum_info->quantum=32;
```
{"review_id": "primevul_paired:train:2381:line:331", "sample_key": "primevul_paired:train:2381", "pair_key": "primevul_paired:train:1190", "line": 331, "target": 1, "shape": "addition_only", "slice_status": "ok", "quality_status": "legal", "methods": ["losver"], "code": "      if (cin.file.user_length > GetBlobSize(image))"}
{"review_id": "primevul_paired:train:2381:line:334", "sample_key": "primevul_paired:train:2381", "pair_key": "primevul_paired:train:1190", "line": 334, "target": 1, "shape": "addition_only", "slice_status": "ok", "quality_status": "legal", "methods": ["static_candidates"], "code": "        cin.file.user_length);"}
{"review_id": "primevul_paired:train:2381:line:368", "sample_key": "primevul_paired:train:2381", "pair_key": "primevul_paired:train:1190", "line": 368, "target": 1, "shape": "addition_only", "slice_status": "ok", "quality_status": "legal", "methods": ["static_candidates", "bidirectional_slice"], "code": "  quantum_info=AcquireQuantumInfo(image_info,image);"}
### primevul_paired:train:2382
```c
   70:       ((LocaleNCompare((char *) magick,"\200\052\137\327",4) != 0)))
   71:     ThrowReaderException(CorruptImageError,"ImproperImageHeader");
   72:   memset(&cin,0,sizeof(cin));
   73:   image->endian=(magick[0] == 0x80) && (magick[1] == 0x2a) &&
   74:     (magick[2] == 0x5f) && (magick[3] == 0xd7) ? MSBEndian : LSBEndian;
   75:   cin.file.image_offset=ReadBlobLong(image);
   76:   offset+=4;
  105:   /*
  106:     Image information.
  107:   */
  108:   cin.image.orientation=(unsigned char) ReadBlobByte(image);
  109:   offset++;
  110:   if (cin.image.orientation != (unsigned char) (~0))
  111:     (void) FormatImageProperty(image,"dpx:image.orientation","%d",
  329:       /*
  330:         User defined data.
  331:       */
  332:       if (cin.file.user_length > GetBlobSize(image))
  333:         ThrowReaderException(CorruptImageError,"InsufficientImageDataInFile");
  334:       profile=BlobToStringInfo((const unsigned char *) NULL,
  335:         cin.file.user_length);
  349:        return(image);
  350:      }
  351:   if (((MagickSizeType) image->columns*image->rows) > GetBlobSize(image))
  352:     ThrowReaderException(CorruptImageError,"InsufficientImageDataInFile");
  353:    for ( ; offset < (MagickOffsetType) cin.file.image_offset; offset++)
  354:    {
  355:      int
```
{"review_id": "primevul_paired:train:2382:line:72", "sample_key": "primevul_paired:train:2382", "pair_key": "primevul_paired:train:1190", "line": 72, "target": 0, "shape": "addition_only", "slice_status": "ok", "quality_status": "legal", "methods": ["bidirectional_slice"], "code": "  image->endian=(magick[0] == 0x80) && (magick[1] == 0x2a) &&"}
{"review_id": "primevul_paired:train:2382:line:107", "sample_key": "primevul_paired:train:2382", "pair_key": "primevul_paired:train:1190", "line": 107, "target": 0, "shape": "addition_only", "slice_status": "ok", "quality_status": "legal", "methods": ["static_candidates"], "code": "  cin.image.orientation=(unsigned char) ReadBlobByte(image);"}
{"review_id": "primevul_paired:train:2382:line:331", "sample_key": "primevul_paired:train:2382", "pair_key": "primevul_paired:train:1190", "line": 331, "target": 0, "shape": "addition_only", "slice_status": "ok", "quality_status": "legal", "methods": ["losver"], "code": "      if (cin.file.user_length > GetBlobSize(image))"}
{"review_id": "primevul_paired:train:2382:line:351", "sample_key": "primevul_paired:train:2382", "pair_key": "primevul_paired:train:1190", "line": 351, "target": 0, "shape": "addition_only", "slice_status": "ok", "quality_status": "legal", "methods": ["original_diff", "bidirectional_slice", "losver"], "code": "    ThrowReaderException(CorruptImageError,\"InsufficientImageDataInFile\");"}

## primevul_paired:train:2194 / replacement_only / legal
```diff
--- 
+++ 
@@ -27,8 +27,6 @@
 		gf_list_del(l2);
 		if (state->bs) {
-			if (gf_bs_get_position(state->bs)) {
-				u32 size;
-				gf_bs_get_content_no_truncate(state->bs, &state->frame_obus, &size, &state->frame_obus_alloc);
-			}
+			u32 size;
+			gf_bs_get_content_no_truncate(state->bs, &state->frame_obus, &size, &state->frame_obus_alloc);
 			gf_bs_del(state->bs);
 		}
```
### primevul_paired:train:4389
```c
    3: 	GF_List *l1, *l2;
    4: 
    5: 	if (state->frame_state.header_obus) {
    6: 		while (gf_list_count(state->frame_state.header_obus)) {
    7: 			GF_AV1_OBUArrayEntry *a = (GF_AV1_OBUArrayEntry*)gf_list_pop_back(state->frame_state.header_obus);
    8: 			if (a->obu) gf_free(a->obu);
    9: 			gf_free(a);
   25: 	if (is_destroy) {
   26: 		gf_list_del(l1);
   27: 		gf_list_del(l2);
   28: 		if (state->bs) {
   29: 			if (gf_bs_get_position(state->bs)) {
   30: 				u32 size;
   31: 				gf_bs_get_content_no_truncate(state->bs, &state->frame_obus, &size, &state->frame_obus_alloc);
   32: 			}
   33: 			gf_bs_del(state->bs);
```
{"review_id": "primevul_paired:train:4389:line:5", "sample_key": "primevul_paired:train:4389", "pair_key": "primevul_paired:train:2194", "line": 5, "target": 1, "shape": "replacement_only", "slice_status": "ok", "quality_status": "legal", "methods": ["static_candidates"], "code": "\t\twhile (gf_list_count(state->frame_state.header_obus)) {"}
{"review_id": "primevul_paired:train:4389:line:27", "sample_key": "primevul_paired:train:4389", "pair_key": "primevul_paired:train:2194", "line": 27, "target": 1, "shape": "replacement_only", "slice_status": "ok", "quality_status": "legal", "methods": ["static_candidates", "bidirectional_slice", "losver"], "code": "\t\tif (state->bs) {"}
{"review_id": "primevul_paired:train:4389:line:28", "sample_key": "primevul_paired:train:4389", "pair_key": "primevul_paired:train:2194", "line": 28, "target": 1, "shape": "replacement_only", "slice_status": "ok", "quality_status": "legal", "methods": ["original_diff", "static_candidates", "bidirectional_slice", "losver"], "code": "\t\t\tif (gf_bs_get_position(state->bs)) {"}
{"review_id": "primevul_paired:train:4389:line:29", "sample_key": "primevul_paired:train:4389", "pair_key": "primevul_paired:train:2194", "line": 29, "target": 1, "shape": "replacement_only", "slice_status": "ok", "quality_status": "legal", "methods": ["original_diff", "bidirectional_slice", "losver"], "code": "\t\t\t\tu32 size;"}
### primevul_paired:train:4390
```c
   26: 		gf_list_del(l1);
   27: 		gf_list_del(l2);
   28: 		if (state->bs) {
   29: 			u32 size;
   30: 			gf_bs_get_content_no_truncate(state->bs, &state->frame_obus, &size, &state->frame_obus_alloc);
   31: 			gf_bs_del(state->bs);
   32: 		}
   33: 		state->bs = NULL;
   34: 	}
   35: 	else {
   36: 		state->frame_state.frame_obus = l1;
   37: 		state->frame_state.header_obus = l2;
   38: 		if (state->bs)
   39: 			gf_bs_seek(state->bs, 0);
   40: 	}
   41: }
```
{"review_id": "primevul_paired:train:4390:line:28", "sample_key": "primevul_paired:train:4390", "pair_key": "primevul_paired:train:2194", "line": 28, "target": 0, "shape": "replacement_only", "slice_status": "ok", "quality_status": "legal", "methods": ["original_diff", "bidirectional_slice", "losver"], "code": "\t\t\tu32 size;"}
{"review_id": "primevul_paired:train:4390:line:29", "sample_key": "primevul_paired:train:4390", "pair_key": "primevul_paired:train:2194", "line": 29, "target": 0, "shape": "replacement_only", "slice_status": "ok", "quality_status": "legal", "methods": ["original_diff", "static_candidates", "bidirectional_slice", "losver"], "code": "\t\t\tgf_bs_get_content_no_truncate(state->bs, &state->frame_obus, &size, &state->frame_obus_alloc);"}
{"review_id": "primevul_paired:train:4390:line:32", "sample_key": "primevul_paired:train:4390", "pair_key": "primevul_paired:train:2194", "line": 32, "target": 0, "shape": "replacement_only", "slice_status": "ok", "quality_status": "legal", "methods": ["bidirectional_slice", "losver"], "code": "\t\tstate->bs = NULL;"}
{"review_id": "primevul_paired:train:4390:line:38", "sample_key": "primevul_paired:train:4390", "pair_key": "primevul_paired:train:2194", "line": 38, "target": 0, "shape": "replacement_only", "slice_status": "ok", "quality_status": "legal", "methods": ["static_candidates"], "code": "\t\t\tgf_bs_seek(state->bs, 0);"}

## primevul_paired:train:1522 / replacement_only / verified_error
```diff
--- 
+++ 
@@ -1,4 +1,4 @@
-void InputEngine::ProcessText(const std::string& message,
-                              ProcessTextCallback callback) {
-  NOTIMPLEMENTED();  // Text message not used in the rulebased engine.
+ void InputEngine::ProcessMessage(const std::vector<uint8_t>& message,
+                                  ProcessMessageCallback callback) {
+   NOTIMPLEMENTED();  // Protobuf message is not used in the rulebased engine.
 }
```
### primevul_paired:train:3045
```c
    1: void InputEngine::ProcessText(const std::string& message,
    2:                               ProcessTextCallback callback) {
    3:   NOTIMPLEMENTED();  // Text message not used in the rulebased engine.
    4: }
```
{"review_id": "primevul_paired:train:3045:line:0", "sample_key": "primevul_paired:train:3045", "pair_key": "primevul_paired:train:1522", "line": 0, "target": 1, "shape": "replacement_only", "slice_status": "ok", "quality_status": "verified_error", "methods": ["original_diff", "losver"], "code": "void InputEngine::ProcessText(const std::string& message,"}
{"review_id": "primevul_paired:train:3045:line:2", "sample_key": "primevul_paired:train:3045", "pair_key": "primevul_paired:train:1522", "line": 2, "target": 1, "shape": "replacement_only", "slice_status": "ok", "quality_status": "verified_error", "methods": ["original_diff", "bidirectional_slice", "losver"], "code": "  NOTIMPLEMENTED();  // Text message not used in the rulebased engine."}
### primevul_paired:train:3046
```c
    1:  void InputEngine::ProcessMessage(const std::vector<uint8_t>& message,
    2:                                   ProcessMessageCallback callback) {
    3:    NOTIMPLEMENTED();  // Protobuf message is not used in the rulebased engine.
    4: }
```
{"review_id": "primevul_paired:train:3046:line:2", "sample_key": "primevul_paired:train:3046", "pair_key": "primevul_paired:train:1522", "line": 2, "target": 0, "shape": "replacement_only", "slice_status": "ok", "quality_status": "verified_error", "methods": ["original_diff", "bidirectional_slice", "losver"], "code": "   NOTIMPLEMENTED();  // Protobuf message is not used in the rulebased engine."}

## primevul_paired:train:1693 / deletion_only / uncertain
```diff
--- 
+++ 
@@ -53,5 +53,4 @@
      ps_cur_slice = ps_dec->ps_cur_slice;
      ps_pic_params = ps_dec->ps_cur_pps;
-    ps_cur_slice->u1_field_pic_flag = 0;
  
      i4_frame_gaps = 0;
```
### primevul_paired:train:3387
```c
   52:  
   53:      ps_cur_slice = ps_dec->ps_cur_slice;
   54:      ps_pic_params = ps_dec->ps_cur_pps;
   55:     ps_cur_slice->u1_field_pic_flag = 0;
   56:  
   57:      i4_frame_gaps = 0;
   58:      ps_dpb_mgr = ps_dec->ps_dpb_mgr;
   85:             ret = ih264d_decode_pic_order_cnt(0, u4_next_frm_num,
   86:  &ps_dec->s_prev_pic_poc,
   87:  &s_tmp_poc, ps_cur_slice,
   88:                                               ps_pic_params, 1, 0, 0,
   89:  &i4_poc);
   90:  if(ret != OK)
   91:  return ret;
   92: 
   93:  /* Display seq no calculations */
   94:  if(i4_poc >= ps_dec->i4_max_poc)
   95:                 ps_dec->i4_max_poc = i4_poc;
   96:  /* IDR Picture or POC wrap around */
   97:  if(i4_poc == 0)
```
{"review_id": "primevul_paired:train:3387:line:54", "sample_key": "primevul_paired:train:3387", "pair_key": "primevul_paired:train:1693", "line": 54, "target": 1, "shape": "deletion_only", "slice_status": "ok", "quality_status": "uncertain", "methods": ["original_diff", "bidirectional_slice", "losver"], "code": "    ps_cur_slice->u1_field_pic_flag = 0;"}
{"review_id": "primevul_paired:train:3387:line:87", "sample_key": "primevul_paired:train:3387", "pair_key": "primevul_paired:train:1693", "line": 87, "target": 1, "shape": "deletion_only", "slice_status": "ok", "quality_status": "uncertain", "methods": ["static_candidates", "bidirectional_slice"], "code": "                                              ps_pic_params, 1, 0, 0,"}
{"review_id": "primevul_paired:train:3387:line:93", "sample_key": "primevul_paired:train:3387", "pair_key": "primevul_paired:train:1693", "line": 93, "target": 1, "shape": "deletion_only", "slice_status": "ok", "quality_status": "uncertain", "methods": ["bidirectional_slice"], "code": " if(i4_poc >= ps_dec->i4_max_poc)"}
### primevul_paired:train:3388
```c
   75:     ps_dpb_mgr->ai4_gaps_end_frm_num[i1_gap_idx] = u2_frame_num - 1;
   76:     pi1_gaps_per_seq = ps_dpb_mgr->ai1_gaps_per_seq;
   77:     pi1_gaps_per_seq[i1_gap_idx] = 0;
   78:  while(u4_next_frm_num != u2_frame_num)
   79:  {
   80:         ih264d_delete_nonref_nondisplay_pics(ps_dpb_mgr);
   81:  if(ps_pic_params->ps_sps->u1_pic_order_cnt_type)
   95:  /* IDR Picture or POC wrap around */
   96:  if(i4_poc == 0)
   97:  {
   98:                 ps_dec->i4_prev_max_display_seq =
   99:                                 ps_dec->i4_prev_max_display_seq
  100:  + ps_dec->i4_max_poc
  101:  + ps_dec->u1_max_dec_frame_buffering
```
{"review_id": "primevul_paired:train:3388:line:77", "sample_key": "primevul_paired:train:3388", "pair_key": "primevul_paired:train:1693", "line": 77, "target": 0, "shape": "deletion_only", "slice_status": "ok", "quality_status": "uncertain", "methods": ["static_candidates"], "code": " while(u4_next_frm_num != u2_frame_num)"}
{"review_id": "primevul_paired:train:3388:line:97", "sample_key": "primevul_paired:train:3388", "pair_key": "primevul_paired:train:1693", "line": 97, "target": 0, "shape": "deletion_only", "slice_status": "ok", "quality_status": "uncertain", "methods": ["bidirectional_slice"], "code": "                ps_dec->i4_prev_max_display_seq ="}

## primevul_paired:train:1583 / deletion_only / legal
```diff
--- 
+++ 
@@ -1,5 +1,4 @@
  void ServiceWorkerPaymentInstrument::OnPaymentAppInvoked(
      mojom::PaymentHandlerResponsePtr response) {
-  DCHECK(delegate_);
    if (delegate_ != nullptr) {
      delegate_->OnInstrumentDetailsReady(response->method_name,
```
### primevul_paired:train:3167
```c
    1:  void ServiceWorkerPaymentInstrument::OnPaymentAppInvoked(
    2:      mojom::PaymentHandlerResponsePtr response) {
    3:   DCHECK(delegate_);
    4:    if (delegate_ != nullptr) {
    5:      delegate_->OnInstrumentDetailsReady(response->method_name,
    6:                                          response->stringified_details);
```
{"review_id": "primevul_paired:train:3167:line:2", "sample_key": "primevul_paired:train:3167", "pair_key": "primevul_paired:train:1583", "line": 2, "target": 1, "shape": "deletion_only", "slice_status": "ok", "quality_status": "legal", "methods": ["original_diff", "bidirectional_slice", "losver"], "code": "  DCHECK(delegate_);"}
### primevul_paired:train:3168
```c
```

## primevul_paired:train:1682 / mixed / uncertain
```diff
--- 
+++ 
@@ -7,5 +7,4 @@
      unsigned int qxdm_data[6];
      const char *deactData[1] = {"1"};
-    char *actData[1];
      RIL_Dial dialData;
      int hangupData[1] = {1};
@@ -13,4 +12,5 @@
      char **args;
      RIL_SOCKET_ID socket_id = RIL_SOCKET_1;
+    int MAX_DIAL_ADDRESS = 128;
      int sim_id = 0;
  
@@ -157,10 +157,4 @@
              issueLocalRequest(RIL_REQUEST_SET_NETWORK_SELECTION_AUTOMATIC, NULL, 0, socket_id);
              break;
-        case 6:
-            RLOGI("Debug port: Setup Data Call, Apn :%s\n", args[1]);
-            actData[0] = args[1];
-            issueLocalRequest(RIL_REQUEST_SETUP_DATA_CALL, &actData,
-                              sizeof(actData), socket_id);
-            break;
          case 7:
              RLOGI("Debug port: Deactivate Data Call");
@@ -172,4 +166,10 @@
              RLOGI("Debug port: Dial Call");
              dialData.clir = 0;
+            if (strlen(args[1]) > MAX_DIAL_ADDRESS) {
+                RLOGE("Debug port: Error calling Dial");
+                freeDebugCallbackArgs(number, args);
+                close(acceptFD);
+                return;
+            }
              dialData.address = args[1];
              issueLocalRequest(RIL_REQUEST_DIAL, &dialData, sizeof(dialData), socket_id);
```
### primevul_paired:train:3365
```c
  161:             actData[0] = args[1];
  162:             issueLocalRequest(RIL_REQUEST_SETUP_DATA_CALL, &actData,
  163:                               sizeof(actData), socket_id);
  164:             break;
  165:          case 7:
  166:              RLOGI("Debug port: Deactivate Data Call");
  167:              issueLocalRequest(RIL_REQUEST_DEACTIVATE_DATA_CALL, &deactData,
```
{"review_id": "primevul_paired:train:3365:line:163", "sample_key": "primevul_paired:train:3365", "pair_key": "primevul_paired:train:1682", "line": 163, "target": 1, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "uncertain", "methods": ["original_diff", "losver"], "code": "            break;"}
### primevul_paired:train:3366
```c
  165:          case 8:
  166:              RLOGI("Debug port: Dial Call");
  167:              dialData.clir = 0;
  168:             if (strlen(args[1]) > MAX_DIAL_ADDRESS) {
  169:                 RLOGE("Debug port: Error calling Dial");
  170:                 freeDebugCallbackArgs(number, args);
  171:                 close(acceptFD);
  172:                 return;
  173:             }
  174:              dialData.address = args[1];
  175:              issueLocalRequest(RIL_REQUEST_DIAL, &dialData, sizeof(dialData), socket_id);
```
{"review_id": "primevul_paired:train:3366:line:167", "sample_key": "primevul_paired:train:3366", "pair_key": "primevul_paired:train:1682", "line": 167, "target": 0, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "uncertain", "methods": ["original_diff", "losver"], "code": "            if (strlen(args[1]) > MAX_DIAL_ADDRESS) {"}
{"review_id": "primevul_paired:train:3366:line:171", "sample_key": "primevul_paired:train:3366", "pair_key": "primevul_paired:train:1682", "line": 171, "target": 0, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "uncertain", "methods": ["original_diff", "losver"], "code": "                return;"}

## primevul_paired:train:2459 / mixed / legal
```diff
--- 
+++ 
@@ -33,7 +33,11 @@
     } else {
         if (ncolors > SIXEL_PALETTE_MAX) {
+            status = SIXEL_BAD_INPUT;
             ncolors = 256;
-        } else if (ncolors < 2) {
-            ncolors = 2;
+        } else if (ncolors < 1) {
+            status = SIXEL_BAD_INPUT;
+            sixel_helper_set_additional_message(
+                "sixel_dither_new: palette colors must be more than 0");
+            goto end;
         }
         quality_mode = SIXEL_QUALITY_LOW;
```
### primevul_paired:train:4919
```c
   17:         goto end;
   18:     }
   19: 
   20:     if (allocator == NULL) {
   21:         status = sixel_allocator_new(&allocator, NULL, NULL, NULL, NULL);
   22:         if (SIXEL_FAILED(status)) {
   23:             *ppdither = NULL;
   34:         if (ncolors > SIXEL_PALETTE_MAX) {
   35:             ncolors = 256;
   36:         } else if (ncolors < 2) {
   37:             ncolors = 2;
   38:         }
   39:         quality_mode = SIXEL_QUALITY_LOW;
   40:     }
```
{"review_id": "primevul_paired:train:4919:line:19", "sample_key": "primevul_paired:train:4919", "pair_key": "primevul_paired:train:2459", "line": 19, "target": 1, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["static_candidates"], "code": "    if (allocator == NULL) {"}
{"review_id": "primevul_paired:train:4919:line:36", "sample_key": "primevul_paired:train:4919", "pair_key": "primevul_paired:train:2459", "line": 36, "target": 1, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff", "losver"], "code": "            ncolors = 2;"}
### primevul_paired:train:4920
```c
   17:         goto end;
   18:     }
   19: 
   20:     if (allocator == NULL) {
   21:         status = sixel_allocator_new(&allocator, NULL, NULL, NULL, NULL);
   22:         if (SIXEL_FAILED(status)) {
   23:             *ppdither = NULL;
   33:     } else {
   34:         if (ncolors > SIXEL_PALETTE_MAX) {
   35:             status = SIXEL_BAD_INPUT;
   36:             ncolors = 256;
   37:         } else if (ncolors < 1) {
   38:             status = SIXEL_BAD_INPUT;
   39:             sixel_helper_set_additional_message(
   40:                 "sixel_dither_new: palette colors must be more than 0");
   41:             goto end;
   42:         }
   43:         quality_mode = SIXEL_QUALITY_LOW;
```
{"review_id": "primevul_paired:train:4920:line:19", "sample_key": "primevul_paired:train:4920", "pair_key": "primevul_paired:train:2459", "line": 19, "target": 0, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["static_candidates"], "code": "    if (allocator == NULL) {"}
{"review_id": "primevul_paired:train:4920:line:35", "sample_key": "primevul_paired:train:4920", "pair_key": "primevul_paired:train:2459", "line": 35, "target": 0, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["losver"], "code": "            ncolors = 256;"}
{"review_id": "primevul_paired:train:4920:line:39", "sample_key": "primevul_paired:train:4920", "pair_key": "primevul_paired:train:2459", "line": 39, "target": 0, "shape": "mixed", "slice_status": "parse_or_method_unknown", "quality_status": "legal", "methods": ["original_diff", "losver"], "code": "                \"sixel_dither_new: palette colors must be more than 0\");"}