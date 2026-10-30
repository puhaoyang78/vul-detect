# Conditional pure-addition review

All recovered pairs in the previously fixed eight-pair Valid addition cohort; conditional source review, not population precision. No extraction rules changed.

## primevul_paired:valid:203
补齐端点数量检查与错误退出。
```diff
--- 
+++ 
@@ -11,6 +11,11 @@
 	if (!alt) {
 		gspca_err(gspca_dev, "Couldn't get altsetting\n");
 		sd->gspca_dev.usb_err = -EIO;
+		return;
+	}
+
+	if (alt->desc.bNumEndpoints < 1) {
+		sd->gspca_dev.usb_err = -ENODEV;
 		return;
 	}
 
```
Vulnerable source context:
```c
   12: 		gspca_err(gspca_dev, "Couldn't get altsetting\n");
   13: 		sd->gspca_dev.usb_err = -EIO;
   14: 		return;
   15: 	}
   16: 
   17: 	packet_size = le16_to_cpu(alt->endpoint[0].desc.wMaxPacketSize);
   18: 	reg_w(sd, R51x_FIFO_PSIZE, packet_size >> 5);
   19: 
   20: 	reg_w(sd, R511_CAM_UV_EN, 0x01);
   21: 	reg_w(sd, R511_SNAP_UV_EN, 0x01);
   22: 	reg_w(sd, R511_SNAP_OPTS, 0x03);
   23: 
   24: 	/* Here I'm assuming that snapshot size == image size.
   25: 	 * I hope that's always true. --claudio
   26: 	 */
   27: 	hsegs = (sd->gspca_dev.pixfmt.width >> 3) - 1;
   28: 	vsegs = (sd->gspca_dev.pixfmt.height >> 3) - 1;
   29: 
   30: 	reg_w(sd, R511_CAM_PXCNT, hsegs);
   31: 	reg_w(sd, R511_CAM_LNCNT, vsegs);
   32: 	reg_w(sd, R511_CAM_PXDIV, 0x00);
   33: 	reg_w(sd, R511_CAM_LNDIV, 0x00);
   34: 
   35: 	/* YUV420, low pass filter on */
```
{"review_id": "primevul_paired:valid:407:addition:13", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 13, "code": "\t\treturn;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:407:addition:16", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 16, "code": "\tpacket_size = le16_to_cpu(alt->endpoint[0].desc.wMaxPacketSize);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:407:addition:17", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 17, "code": "\treg_w(sd, R51x_FIFO_PSIZE, packet_size >> 5);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:407:addition:19", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 19, "code": "\treg_w(sd, R511_CAM_UV_EN, 0x01);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:407:addition:20", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 20, "code": "\treg_w(sd, R511_SNAP_UV_EN, 0x01);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:407:addition:21", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 21, "code": "\treg_w(sd, R511_SNAP_OPTS, 0x03);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:407:addition:26", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 26, "code": "\thsegs = (sd->gspca_dev.pixfmt.width >> 3) - 1;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:407:addition:27", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 27, "code": "\tvsegs = (sd->gspca_dev.pixfmt.height >> 3) - 1;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:407:addition:29", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 29, "code": "\treg_w(sd, R511_CAM_PXCNT, hsegs);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:407:addition:30", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 30, "code": "\treg_w(sd, R511_CAM_LNCNT, vsegs);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:407:addition:31", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 31, "code": "\treg_w(sd, R511_CAM_PXDIV, 0x00);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:407:addition:32", "sample_key": "primevul_paired:valid:407", "pair_key": "primevul_paired:valid:203", "line": 32, "code": "\treg_w(sd, R511_CAM_LNDIV, 0x00);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
Dependency paths:
{"lines": [14], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [17], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [18], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [20], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [21], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [22], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [27], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [28], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [30], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [31], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [32], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [33], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}

## primevul_paired:valid:298
新增最大图像面积检查。
```diff
--- 
+++ 
@@ -28,5 +28,8 @@
         CHECK(avifROStreamReadU32(&s, &grid->outputWidth));  // unsigned int(FieldLength) output_width;
         CHECK(avifROStreamReadU32(&s, &grid->outputHeight)); // unsigned int(FieldLength) output_height;
     }
+    if (grid->outputWidth > AVIF_MAX_IMAGE_SIZE / grid->outputHeight) {
+        return AVIF_FALSE;
+    }
     return AVIF_TRUE;
 }
```
Vulnerable source context:
```c
    5:     uint8_t version, flags;
    6:     CHECK(avifROStreamRead(&s, &version, 1)); // unsigned int(8) version = 0;
    7:     if (version != 0) {
    8:         return AVIF_FALSE;
    9:     }
   10:     CHECK(avifROStreamRead(&s, &flags, 1));         // unsigned int(8) flags;
   11:     CHECK(avifROStreamRead(&s, &grid->rows, 1));    // unsigned int(8) rows_minus_one;
   12:     CHECK(avifROStreamRead(&s, &grid->columns, 1)); // unsigned int(8) columns_minus_one;
   13:     ++grid->rows;
   14:     ++grid->columns;
   15: 
   16:     uint32_t fieldLength = ((flags & 1) + 1) * 16;
   17:     if (fieldLength == 16) {
   18:         uint16_t outputWidth16, outputHeight16;
   19:         CHECK(avifROStreamReadU16(&s, &outputWidth16));  // unsigned int(FieldLength) output_width;
   20:         CHECK(avifROStreamReadU16(&s, &outputHeight16)); // unsigned int(FieldLength) output_height;
   21:         grid->outputWidth = outputWidth16;
   22:         grid->outputHeight = outputHeight16;
   23:     } else {
   24:         if (fieldLength != 32) {
   25:             // This should be impossible
   26:             return AVIF_FALSE;
   27:         }
   28:         CHECK(avifROStreamReadU32(&s, &grid->outputWidth));  // unsigned int(FieldLength) output_width;
   29:         CHECK(avifROStreamReadU32(&s, &grid->outputHeight)); // unsigned int(FieldLength) output_height;
   30:     }
   31:     return AVIF_TRUE;
   32: }
```
{"review_id": "primevul_paired:valid:597:addition:6", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 6, "code": "    if (version != 0) {", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:597:addition:10", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 10, "code": "    CHECK(avifROStreamRead(&s, &grid->rows, 1));    // unsigned int(8) rows_minus_one;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:597:addition:11", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 11, "code": "    CHECK(avifROStreamRead(&s, &grid->columns, 1)); // unsigned int(8) columns_minus_one;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:597:addition:12", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 12, "code": "    ++grid->rows;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:597:addition:13", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 13, "code": "    ++grid->columns;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:597:addition:16", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 16, "code": "    if (fieldLength == 16) {", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:597:addition:20", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 20, "code": "        grid->outputWidth = outputWidth16;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:597:addition:21", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 21, "code": "        grid->outputHeight = outputHeight16;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:597:addition:23", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 23, "code": "        if (fieldLength != 32) {", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:597:addition:27", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 27, "code": "        CHECK(avifROStreamReadU32(&s, &grid->outputWidth));  // unsigned int(FieldLength) output_width;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:597:addition:28", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 28, "code": "        CHECK(avifROStreamReadU32(&s, &grid->outputHeight)); // unsigned int(FieldLength) output_height;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:597:addition:30", "sample_key": "primevul_paired:valid:597", "pair_key": "primevul_paired:valid:298", "line": 30, "code": "    return AVIF_TRUE;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
Dependency paths:
{"lines": [7], "path": [{"kind": "REACHING_DEF", "variable": "grid->rows", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [11], "path": [{"kind": "REACHING_DEF", "variable": "grid->rows", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "REACHING_DEF", "variable": "&grid->rows", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown", "external_call_summary_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [12], "path": [{"kind": "REACHING_DEF", "variable": "grid->rows", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "REACHING_DEF", "variable": "grid->columns", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown", "external_call_summary_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [13], "path": [{"kind": "REACHING_DEF", "variable": "grid->rows", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [14], "path": [{"kind": "REACHING_DEF", "variable": "grid->columns", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [17], "path": [{"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [21], "path": [{"kind": "REACHING_DEF", "variable": "grid->outputWidth", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [22], "path": [{"kind": "REACHING_DEF", "variable": "grid->outputHeight", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [24], "path": [{"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [28], "path": [{"kind": "REACHING_DEF", "variable": "&grid->outputWidth", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown", "external_call_summary_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [29], "path": [{"kind": "REACHING_DEF", "variable": "&grid->outputHeight", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown", "external_call_summary_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [31], "path": [{"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}

## primevul_paired:valid:31
设置aAscii[0]=0，排除NUL token；状态初始化不是检查语句。
```diff
--- 
+++ 
@@ -9,4 +9,5 @@
     }
     iTbl++;
   }
+  aAscii[0] = 0;                  /* 0x00 is never a token character */
 }
```
Vulnerable source context:
```c
    1: void sqlite3Fts5UnicodeAscii(u8 *aArray, u8 *aAscii){
    2:   int i = 0;
    3:   int iTbl = 0;
    4:   while( i<128 ){
    5:     int bToken = aArray[ aFts5UnicodeData[iTbl] & 0x1F ];
    6:     int n = (aFts5UnicodeData[iTbl] >> 5) + i;
    7:     for(; i<128 && i<n; i++){
    8:       aAscii[i] = (u8)bToken;
    9:     }
   10:     iTbl++;
   11:   }
   12: }
```
{"review_id": "primevul_paired:valid:63:addition:1", "sample_key": "primevul_paired:valid:63", "pair_key": "primevul_paired:valid:31", "line": 1, "code": "  int i = 0;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:63:addition:2", "sample_key": "primevul_paired:valid:63", "pair_key": "primevul_paired:valid:31", "line": 2, "code": "  int iTbl = 0;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:63:addition:3", "sample_key": "primevul_paired:valid:63", "pair_key": "primevul_paired:valid:31", "line": 3, "code": "  while( i<128 ){", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:63:addition:4", "sample_key": "primevul_paired:valid:63", "pair_key": "primevul_paired:valid:31", "line": 4, "code": "    int bToken = aArray[ aFts5UnicodeData[iTbl] & 0x1F ];", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:63:addition:5", "sample_key": "primevul_paired:valid:63", "pair_key": "primevul_paired:valid:31", "line": 5, "code": "    int n = (aFts5UnicodeData[iTbl] >> 5) + i;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:63:addition:6", "sample_key": "primevul_paired:valid:63", "pair_key": "primevul_paired:valid:31", "line": 6, "code": "    for(; i<128 && i<n; i++){", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:63:addition:7", "sample_key": "primevul_paired:valid:63", "pair_key": "primevul_paired:valid:31", "line": 7, "code": "      aAscii[i] = (u8)bToken;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:63:addition:9", "sample_key": "primevul_paired:valid:63", "pair_key": "primevul_paired:valid:31", "line": 9, "code": "    iTbl++;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
Dependency paths:
{"lines": [2], "path": [{"kind": "REACHING_DEF", "variable": "aAscii[i]", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "REACHING_DEF", "variable": "bToken", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}, {"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "REACHING_DEF", "variable": "i", "traversal": "backward", "unknown": []}]}
{"lines": [3], "path": [{"kind": "REACHING_DEF", "variable": "aAscii[i]", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "REACHING_DEF", "variable": "bToken", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}, {"kind": "REACHING_DEF", "variable": "iTbl", "traversal": "backward", "unknown": []}]}
{"lines": [4], "path": [{"kind": "REACHING_DEF", "variable": "aAscii[i]", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}, {"kind": "REACHING_DEF", "variable": "i", "traversal": "forward", "unknown": []}]}
{"lines": [5], "path": [{"kind": "REACHING_DEF", "variable": "aAscii[i]", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "REACHING_DEF", "variable": "bToken", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [6], "path": [{"kind": "REACHING_DEF", "variable": "aAscii[i]", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "REACHING_DEF", "variable": "bToken", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}, {"kind": "REACHING_DEF", "variable": "aFts5UnicodeData[iTbl]", "traversal": "forward", "unknown": ["alias_or_memory_state_unknown"]}]}
{"lines": [7], "path": [{"kind": "REACHING_DEF", "variable": "aAscii[i]", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [8], "path": [{"kind": "REACHING_DEF", "variable": "aAscii[i]", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [10], "path": [{"kind": "REACHING_DEF", "variable": "aAscii[i]", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}, {"kind": "REACHING_DEF", "variable": "i", "traversal": "forward", "unknown": []}, {"kind": "CDG", "variable": "", "traversal": "forward", "unknown": []}]}

## primevul_paired:valid:54
释放后置NULL并补齐另一资源释放，属于生命周期补全。
```diff
--- 
+++ 
@@ -83,6 +83,7 @@
     *out_size = tmp_out_size;
     if (mp_buf != tmp_out_buf) {
         flb_free(mp_buf);
+        mp_buf = NULL;
     }
 
     /* Do time resolution ? */
@@ -113,6 +114,7 @@
         /* Ensure the pointer we are about to read is not NULL */
         if (k->via.str.ptr == NULL) {
             flb_free(mp_buf);
+            flb_free(tmp_out_buf);
             *out_buf = NULL;
             msgpack_unpacked_destroy(&result);
             return -1;
```
Vulnerable source context:
```c
   31: 
   32:     /* Convert incoming in_buf JSON message to message pack format */
   33:     ret = flb_pack_json_recs(in_buf, in_size, &mp_buf, &mp_size, &root_type,
   34:                              &records);
   35:     if (ret != 0) {
   36:         return -1;
   46:     if (msgpack_unpack_next(&result, mp_buf, mp_size, &off) == MSGPACK_UNPACK_SUCCESS) {
   47:         map = result.data;
   48:         if (map.type != MSGPACK_OBJECT_MAP) {
   49:             flb_free(mp_buf);
   50:             msgpack_unpacked_destroy(&result);
   61: 
   62:     /* Export results (might change later) */
   63:     tmp_out_buf = mp_buf;
   64:     tmp_out_size = mp_size;
   65: 
   66:     /* Do we have some decoders set ? */
   67:     if (parser->decoders) {
   68:         ret = flb_parser_decoder_do(parser->decoders,
   69:                                     mp_buf, mp_size,
   70:                                     &tmp_out_buf, &tmp_out_size);
   71:         if (ret == 0) {
   72:             /* re-process the unpack context */
   80: 
   81:     /* Set the possible outgoing buffer */
   82:     *out_buf = tmp_out_buf;
   83:     *out_size = tmp_out_size;
   84:     if (mp_buf != tmp_out_buf) {
   85:         flb_free(mp_buf);
   86:     }
  104:     skip = map_size;
  105:     for (i = 0; i < map_size; i++) {
  106:         k = &map.via.map.ptr[i].key;
  107:         v = &map.via.map.ptr[i].val;
  108: 
  109:         if (k->via.str.size != slen) {
  110:             continue;
  111:         }
  112: 
  113:         /* Ensure the pointer we are about to read is not NULL */
  114:         if (k->via.str.ptr == NULL) {
  115:             flb_free(mp_buf);
  116:             *out_buf = NULL;
  117:             msgpack_unpacked_destroy(&result);
```
{"review_id": "primevul_paired:valid:109:addition:32", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 32, "code": "    ret = flb_pack_json_recs(in_buf, in_size, &mp_buf, &mp_size, &root_type,", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:33", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 33, "code": "                             &records);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:47", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 47, "code": "        if (map.type != MSGPACK_OBJECT_MAP) {", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:62", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 62, "code": "    tmp_out_buf = mp_buf;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:63", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 63, "code": "    tmp_out_size = mp_size;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:66", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 66, "code": "    if (parser->decoders) {", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:67", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 67, "code": "        ret = flb_parser_decoder_do(parser->decoders,", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:68", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 68, "code": "                                    mp_buf, mp_size,", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:69", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 69, "code": "                                    &tmp_out_buf, &tmp_out_size);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:81", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 81, "code": "    *out_buf = tmp_out_buf;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:83", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 83, "code": "    if (mp_buf != tmp_out_buf) {", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:105", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 105, "code": "        k = &map.via.map.ptr[i].key;", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:108", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 108, "code": "        if (k->via.str.size != slen) {", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:113", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 113, "code": "        if (k->via.str.ptr == NULL) {", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
{"review_id": "primevul_paired:valid:109:addition:114", "sample_key": "primevul_paired:valid:109", "pair_key": "primevul_paired:valid:54", "line": 114, "code": "            flb_free(mp_buf);", "scope": "all candidate vulnerable-side lines in available original addition review cohort"}
Dependency paths:
{"lines": [33, 34], "path": [{"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "REACHING_DEF", "variable": "&mp_buf", "traversal": "backward", "unknown": ["external_call_summary_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [48], "path": [{"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [63], "path": [{"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "REACHING_DEF", "variable": "mp_buf", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [64], "path": [{"kind": "REACHING_DEF", "variable": "&tmp_out_buf", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown", "external_call_summary_unknown"]}, {"kind": "REACHING_DEF", "variable": "mp_size", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown", "external_call_summary_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [67], "path": [{"kind": "REACHING_DEF", "variable": "&tmp_out_buf", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown", "external_call_summary_unknown"]}, {"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [68, 69, 70], "path": [{"kind": "REACHING_DEF", "variable": "&tmp_out_buf", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown", "external_call_summary_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [82], "path": [{"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "REACHING_DEF", "variable": "tmp_out_buf", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [84], "path": [{"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [106], "path": [{"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "REACHING_DEF", "variable": "k", "traversal": "backward", "unknown": ["alias_or_memory_state_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [109], "path": [{"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [114], "path": [{"kind": "CDG", "variable": "", "traversal": "backward", "unknown": []}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}
{"lines": [115], "path": [{"kind": "REACHING_DEF", "variable": "mp_buf", "traversal": "forward", "unknown": ["external_call_summary_unknown"]}, {"kind": "VERSION_MAP", "source_side": "p", "target_side": "v"}]}