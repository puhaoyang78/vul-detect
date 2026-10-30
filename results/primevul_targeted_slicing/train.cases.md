# Fixed slicing source review

0-based stored / 1-based displayed physical rows. Assistant review, not independent vulnerability gold.

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
   70:       ((LocaleNCompare((char *) magick,"\200\052\137\327",4) != 0)))
   71:     ThrowReaderException(CorruptImageError,"ImproperImageHeader");
   72:   memset(&cin,0,sizeof(cin));
   73:   image->endian=(magick[0] == 0x80) && (magick[1] == 0x2a) &&
   74:     (magick[2] == 0x5f) && (magick[3] == 0xd7) ? MSBEndian : LSBEndian;
   75:   cin.file.image_offset=ReadBlobLong(image);
   76:   offset+=4;
   77:   cin.file.generic_length=ReadBlobLong(image);
  353:      int
  354:       c;
  355: 
  356:     c=ReadBlobByte(image);
  357:     if (c == EOF)
  358:       break;
  359:   }
  360:   if (offset < (MagickOffsetType) cin.file.image_offset)
  361:     ThrowReaderException(CorruptImageError,"ImproperImageHeader");
  362:   status=SetImageExtent(image,image->columns,image->rows,exception);
  363:   if (status == MagickFalse)
  364:     return(DestroyImageList(image));
  365:   (void) SetImageBackgroundColor(image,exception);
  366:   /*
  367:     Convert CIN raster image to pixel packets.
  368:   */
  369:   quantum_info=AcquireQuantumInfo(image_info,image);
  370:   if (quantum_info == (QuantumInfo *) NULL)
  371:     ThrowReaderException(ResourceLimitError,"MemoryAllocationFailed");
  372:   quantum_info->quantum=32;
  373:   quantum_info->pack=MagickFalse;
  374:   quantum_type=RGBQuantum;
  375:   length=GetQuantumExtent(image,quantum_info,quantum_type);
  376:   length=GetBytesPerRow(image->columns,3,image->depth,MagickTrue);
  377:   if (cin.image.number_channels == 1)
  378:     {
  379:       quantum_type=GrayQuantum;
  380:       length=GetBytesPerRow(image->columns,1,image->depth,MagickTrue);
  381:     }
  382:   for (y=0; y < (ssize_t) image->rows; y++)
  383:   {
  384:     q=QueueAuthenticPixels(image,0,y,image->columns,1,exception);
  385:     if (q == (Quantum *) NULL)
```
{"review_id": "primevul_paired:train:2381:line:72", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 72, "code": "  image->endian=(magick[0] == 0x80) && (magick[1] == 0x2a) &&", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:73", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 73, "code": "    (magick[2] == 0x5f) && (magick[3] == 0xd7) ? MSBEndian : LSBEndian;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:355", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 355, "code": "    c=ReadBlobByte(image);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:356", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 356, "code": "    if (c == EOF)", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:361", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 361, "code": "  status=SetImageExtent(image,image->columns,image->rows,exception);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:362", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 362, "code": "  if (status == MagickFalse)", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:363", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 363, "code": "    return(DestroyImageList(image));", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:364", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 364, "code": "  (void) SetImageBackgroundColor(image,exception);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:368", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 368, "code": "  quantum_info=AcquireQuantumInfo(image_info,image);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:374", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 374, "code": "  length=GetQuantumExtent(image,quantum_info,quantum_type);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:375", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 375, "code": "  length=GetBytesPerRow(image->columns,3,image->depth,MagickTrue);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:379", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 379, "code": "      length=GetBytesPerRow(image->columns,1,image->depth,MagickTrue);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2381:line:381", "pair_key": "primevul_paired:train:1190", "sample_key": "primevul_paired:train:2381", "line": 381, "code": "  for (y=0; y < (ssize_t) image->rows; y++)", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:2382
```c
```

## primevul_paired:train:1214 / addition_only / uncertain
```diff
--- 
+++ 
@@ -28,4 +28,9 @@
                      index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno *
                              pi->step_c + pi->precno * pi->step_p;
+                    /* Avoids index out of bounds access with include*/
+                    if (index >= pi->include_size) {
+                        opj_pi_emit_error(pi, "Invalid access to pi->include");
+                        return OPJ_FALSE;
+                    }
                      if (!pi->include[index]) {
                          pi->include[index] = 1;
```
### primevul_paired:train:2429
```c
   12:         pi->first = 0;
   13:     }
   14: 
   15:     for (pi->layno = pi->poc.layno0; pi->layno < pi->poc.layno1; pi->layno++) {
   16:         for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1;
   17:                 pi->resno++) {
   18:             for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {
   19:                 comp = &pi->comps[pi->compno];
   20:                 if (pi->resno >= comp->numresolutions) {
   21:                     continue;
   22:                 }
   23:                 res = &comp->resolutions[pi->resno];
   24:                 if (!pi->tp_on) {
   25:                     pi->poc.precno1 = res->pw * res->ph;
   26:                 }
   27:                  for (pi->precno = pi->poc.precno0; pi->precno < pi->poc.precno1; pi->precno++) {
   28:                      index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno *
   29:                              pi->step_c + pi->precno * pi->step_p;
   30:                      if (!pi->include[index]) {
   31:                          pi->include[index] = 1;
   32:                          return OPJ_TRUE;
   33:                     }
   34: LABEL_SKIP:
```
{"review_id": "primevul_paired:train:2429:line:14", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 14, "code": "    for (pi->layno = pi->poc.layno0; pi->layno < pi->poc.layno1; pi->layno++) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2429:line:15", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 15, "code": "        for (pi->resno = pi->poc.resno0; pi->resno < pi->poc.resno1;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2429:line:16", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 16, "code": "                pi->resno++) {", "cohort": "seed42_check_sample", "methods": ["mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2429:line:17", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 17, "code": "            for (pi->compno = pi->poc.compno0; pi->compno < pi->poc.compno1; pi->compno++) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2429:line:18", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 18, "code": "                comp = &pi->comps[pi->compno];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2429:line:19", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 19, "code": "                if (pi->resno >= comp->numresolutions) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2429:line:22", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 22, "code": "                res = &comp->resolutions[pi->resno];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2429:line:23", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 23, "code": "                if (!pi->tp_on) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:2429:line:24", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 24, "code": "                    pi->poc.precno1 = res->pw * res->ph;", "cohort": "seed42_check_sample", "methods": ["historical"]}
{"review_id": "primevul_paired:train:2429:line:26", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 26, "code": "                 for (pi->precno = pi->poc.precno0; pi->precno < pi->poc.precno1; pi->precno++) {", "cohort": "seed42_check_sample", "methods": ["historical", "targeted"]}
{"review_id": "primevul_paired:train:2429:line:27", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 27, "code": "                     index = pi->layno * pi->step_l + pi->resno * pi->step_r + pi->compno *", "cohort": "seed42_check_sample", "methods": ["historical", "targeted"]}
{"review_id": "primevul_paired:train:2429:line:28", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 28, "code": "                             pi->step_c + pi->precno * pi->step_p;", "cohort": "seed42_check_sample", "methods": ["historical", "targeted"]}
{"review_id": "primevul_paired:train:2429:line:29", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 29, "code": "                     if (!pi->include[index]) {", "cohort": "seed42_check_sample", "methods": ["historical", "targeted"]}
{"review_id": "primevul_paired:train:2429:line:30", "pair_key": "primevul_paired:train:1214", "sample_key": "primevul_paired:train:2429", "line": 30, "code": "                         pi->include[index] = 1;", "cohort": "seed42_check_sample", "methods": ["historical", "targeted"]}
### primevul_paired:train:2430
```c
```

## primevul_paired:train:1227 / addition_only / uncertain
```diff
--- 
+++ 
@@ -3,4 +3,5 @@
  	uint8 type;
  	uint16 len_src_descriptor, len_combined_caps;
+	struct stream packet = *s;
  
  	/* at this point we need to ensure that we have ui created */
@@ -10,4 +11,9 @@
  	in_uint16_le(s, len_src_descriptor);
  	in_uint16_le(s, len_combined_caps);
+
+	if (!s_check_rem(s, len_src_descriptor))
+	{
+		rdp_protocol_error("rdp_demand_active(), consume of source descriptor from stream would overrun", &packet);
+	}
  	in_uint8s(s, len_src_descriptor);
  
```
### primevul_paired:train:2455
```c
```
### primevul_paired:train:2456
```c
```

## primevul_paired:train:1229 / addition_only / uncertain
```diff
--- 
+++ 
@@ -5,4 +5,5 @@
  	uint16 component;
  	uint16 pakid;
+	struct stream packet = *s;
  
  	logger(Protocol, Debug, "rdpdr_process()");
@@ -24,6 +25,16 @@
  				in_uint8s(s, 2);	/* skip versionMajor */
  				in_uint16_le(s, vmin);	/* VersionMinor */
+
  				in_uint32_le(s, g_client_id);	/* ClientID */
  
+				/* g_client_id is sent back to server,
+				   so lets check that we actually got
+				   valid data from stream to prevent
+				   that we leak back data to server */
+				if (!s_check(s))
+				{
+					rdp_protocol_error("rdpdr_process(), consume of g_client_id from stream did overrun", &packet);
+				}
+
  				/* The RDP client is responsibility to provide a random client id
  				   if server version is < 12 */
```
### primevul_paired:train:2459
```c
```
### primevul_paired:train:2460
```c
```

## primevul_paired:train:1517 / addition_only / uncertain
```diff
--- 
+++ 
@@ -9,4 +9,6 @@
    Document& doc = element->GetDocument();
  
+  if (!doc.GetFrame())
+    return false;
  
   if (doc.IsSandboxed(WebSandboxFlags::kAutomaticFeatures)) {
```
### primevul_paired:train:3035
```c
    1: static bool ShouldAutofocus(const HTMLFormControlElement* element) {
    2:   if (!element->isConnected())
    3:     return false;
    4:   if (!element->IsAutofocusable())
    5:     return false;
    6: 
    7: 
    8: 
    9:    Document& doc = element->GetDocument();
   10:  
   11:  
   12:   if (doc.IsSandboxed(WebSandboxFlags::kAutomaticFeatures)) {
   13:     doc.AddConsoleMessage(ConsoleMessage::Create(
   14:         mojom::ConsoleMessageSource::kSecurity,
   15:         mojom::ConsoleMessageLevel::kError,
   16:         "Blocked autofocusing on a form control because the form's frame is "
   17:         "sandboxed and the 'allow-scripts' permission is not set."));
   18:     return false;
   19:   }
   20: 
   21:   if (!doc.IsInMainFrame() &&
   22:       !doc.TopFrameOrigin()->CanAccess(doc.GetSecurityOrigin())) {
   23:     doc.AddConsoleMessage(ConsoleMessage::Create(
   24:         mojom::ConsoleMessageSource::kSecurity,
   25:         mojom::ConsoleMessageLevel::kError,
   26:         "Blocked autofocusing on a form control in a cross-origin subframe."));
   27:     return false;
   28:   }
   29: 
   30:   return true;
   31: }
```
{"review_id": "primevul_paired:train:3035:line:1", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 1, "code": "  if (!element->isConnected())", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:2", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 2, "code": "    return false;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:3", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 3, "code": "  if (!element->IsAutofocusable())", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:4", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 4, "code": "    return false;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:8", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 8, "code": "   Document& doc = element->GetDocument();", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:11", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 11, "code": "  if (doc.IsSandboxed(WebSandboxFlags::kAutomaticFeatures)) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:12", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 12, "code": "    doc.AddConsoleMessage(ConsoleMessage::Create(", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:13", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 13, "code": "        mojom::ConsoleMessageSource::kSecurity,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:14", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 14, "code": "        mojom::ConsoleMessageLevel::kError,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:15", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 15, "code": "        \"Blocked autofocusing on a form control because the form's frame is \"", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:16", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 16, "code": "        \"sandboxed and the 'allow-scripts' permission is not set.\"));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:17", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 17, "code": "    return false;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:20", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 20, "code": "  if (!doc.IsInMainFrame() &&", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:21", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 21, "code": "      !doc.TopFrameOrigin()->CanAccess(doc.GetSecurityOrigin())) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:22", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 22, "code": "    doc.AddConsoleMessage(ConsoleMessage::Create(", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:23", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 23, "code": "        mojom::ConsoleMessageSource::kSecurity,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:24", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 24, "code": "        mojom::ConsoleMessageLevel::kError,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:25", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 25, "code": "        \"Blocked autofocusing on a form control in a cross-origin subframe.\"));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:26", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 26, "code": "    return false;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3035:line:29", "pair_key": "primevul_paired:train:1517", "sample_key": "primevul_paired:train:3035", "line": 29, "code": "  return true;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:3036
```c
```

## primevul_paired:train:1539 / addition_only / uncertain
```diff
--- 
+++ 
@@ -23,4 +23,17 @@
    }
  
+  // Defensive mode is a layer of defense in depth for applications that must
+  // run SQL queries from an untrusted source, such as WebDatabase. Refuse to
+  // proceed if this layer cannot be enabled.
+  open_error_ = sqlite3_db_config(db_, SQLITE_DBCONFIG_DEFENSIVE, 1, nullptr);
+  if (open_error_ != SQLITE_OK) {
+    open_error_message_ = sqlite3_errmsg(db_);
+    DLOG(ERROR) << "SQLite database error when enabling defensive mode - "
+                << open_error_message_.data();
+    sqlite3_close(db_);
+    db_ = nullptr;
+    return false;
+  }
+
    if (IsOpen())
      opening_thread_ = CurrentThread();
```
### primevul_paired:train:3079
```c
    2:   Close();
    3: 
    4:   open_error_ = SQLiteFileSystem::OpenDatabase(filename, &db_);
    5:   if (open_error_ != SQLITE_OK) {
    6:     open_error_message_ =
    7:         db_ ? sqlite3_errmsg(db_) : "sqlite_open returned null";
    8:     DLOG(ERROR) << "SQLite database failed to load from " << filename
    9:                 << "\nCause - " << open_error_message_.data();
   10:     sqlite3_close(db_);
   11:     db_ = nullptr;
   12:     return false;
   13:   }
   14: 
   15:   open_error_ = sqlite3_extended_result_codes(db_, 1);
   16:   if (open_error_ != SQLITE_OK) {
   17:     open_error_message_ = sqlite3_errmsg(db_);
   18:     DLOG(ERROR) << "SQLite database error when enabling extended errors - "
   19:                 << open_error_message_.data();
   22:      return false;
   23:    }
   24:  
   25:    if (IsOpen())
   26:      opening_thread_ = CurrentThread();
   27:    else
   28:     open_error_message_ = "sqlite_open returned null";
   29: 
   30:   if (!SQLiteStatement(*this, "PRAGMA temp_store = MEMORY;").ExecuteCommand())
   31:     DLOG(ERROR) << "SQLite database could not set temp_store to memory";
   32: 
   33:   if (!SQLiteStatement(*this, "PRAGMA foreign_keys = OFF;").ExecuteCommand())
   34:     DLOG(ERROR) << "SQLite database could not turn off foreign_keys";
   35: 
   36:   return IsOpen();
   37: }
```
{"review_id": "primevul_paired:train:3079:line:4", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 4, "code": "  if (open_error_ != SQLITE_OK) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:5", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 5, "code": "    open_error_message_ =", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:6", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 6, "code": "        db_ ? sqlite3_errmsg(db_) : \"sqlite_open returned null\";", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:7", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 7, "code": "    DLOG(ERROR) << \"SQLite database failed to load from \" << filename", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:8", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 8, "code": "                << \"\\nCause - \" << open_error_message_.data();", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:15", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 15, "code": "  if (open_error_ != SQLITE_OK) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:24", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 24, "code": "   if (IsOpen())", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:25", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 25, "code": "     opening_thread_ = CurrentThread();", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:27", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 27, "code": "    open_error_message_ = \"sqlite_open returned null\";", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:29", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 29, "code": "  if (!SQLiteStatement(*this, \"PRAGMA temp_store = MEMORY;\").ExecuteCommand())", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:30", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 30, "code": "    DLOG(ERROR) << \"SQLite database could not set temp_store to memory\";", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:32", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 32, "code": "  if (!SQLiteStatement(*this, \"PRAGMA foreign_keys = OFF;\").ExecuteCommand())", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:33", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 33, "code": "    DLOG(ERROR) << \"SQLite database could not turn off foreign_keys\";", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3079:line:35", "pair_key": "primevul_paired:train:1539", "sample_key": "primevul_paired:train:3079", "line": 35, "code": "  return IsOpen();", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:3080
```c
```

## primevul_paired:train:1589 / addition_only / uncertain
```diff
--- 
+++ 
@@ -37,4 +37,19 @@
      return;
  
+  // Check the source browsing context's CSP to fulfill the CSP check
+  // requirement of https://html.spec.whatwg.org/#navigate for javascript URLs.
+  // Although the spec states we should perform this check on task execution,
+  // we do this prior to dispatch since the parent frame's CSP may be
+  // inaccessible if the target frame is out of process.
+  Document* current_document = current_window->document();
+  if (current_document && completed_url.ProtocolIsJavaScript() &&
+      !ContentSecurityPolicy::ShouldBypassMainWorld(current_document)) {
+    String script_source = DecodeURLEscapeSequences(completed_url.GetString());
+    if (!current_document->GetContentSecurityPolicy()->AllowJavaScriptURLs(
+            nullptr, script_source, current_document->Url(), OrdinalNumber())) {
+      return;
+    }
+  }
+
    V8DOMActivityLogger* activity_logger =
        V8DOMActivityLogger::CurrentActivityLoggerIfIsolatedWorld();
```
### primevul_paired:train:3179
```c
   36:    if (dom_window_->IsInsecureScriptAccess(*current_window, completed_url))
   37:      return;
   38:  
   39:    V8DOMActivityLogger* activity_logger =
   40:        V8DOMActivityLogger::CurrentActivityLoggerIfIsolatedWorld();
   41:    if (activity_logger) {
   42:     Vector<String> argv;
   43:     argv.push_back("LocalDOMWindow");
   44:     argv.push_back("url");
   45:     argv.push_back(entered_document->Url());
   46:     argv.push_back(completed_url);
   47:     activity_logger->LogEvent("blinkSetAttribute", argv.size(), argv.data());
   48:   }
   49:   WebFrameLoadType frame_load_type = WebFrameLoadType::kStandard;
   50:   if (set_location_policy == SetLocationPolicy::kReplaceThisFrame)
   51:     frame_load_type = WebFrameLoadType::kReplaceCurrentItem;
   52:   dom_window_->GetFrame()->ScheduleNavigation(*current_window->document(),
   53:                                               completed_url, frame_load_type,
   54:                                               UserGestureStatus::kNone);
   55: }
```
{"review_id": "primevul_paired:train:3179:line:38", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 38, "code": "   V8DOMActivityLogger* activity_logger =", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:39", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 39, "code": "       V8DOMActivityLogger::CurrentActivityLoggerIfIsolatedWorld();", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:40", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 40, "code": "   if (activity_logger) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:41", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 41, "code": "    Vector<String> argv;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:42", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 42, "code": "    argv.push_back(\"LocalDOMWindow\");", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:43", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 43, "code": "    argv.push_back(\"url\");", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:44", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 44, "code": "    argv.push_back(entered_document->Url());", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:45", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 45, "code": "    argv.push_back(completed_url);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:46", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 46, "code": "    activity_logger->LogEvent(\"blinkSetAttribute\", argv.size(), argv.data());", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:48", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 48, "code": "  WebFrameLoadType frame_load_type = WebFrameLoadType::kStandard;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:49", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 49, "code": "  if (set_location_policy == SetLocationPolicy::kReplaceThisFrame)", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:50", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 50, "code": "    frame_load_type = WebFrameLoadType::kReplaceCurrentItem;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:51", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 51, "code": "  dom_window_->GetFrame()->ScheduleNavigation(*current_window->document(),", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:52", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 52, "code": "                                              completed_url, frame_load_type,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3179:line:53", "pair_key": "primevul_paired:train:1589", "sample_key": "primevul_paired:train:3179", "line": 53, "code": "                                              UserGestureStatus::kNone);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:3180
```c
```

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
```
### primevul_paired:train:3192
```c
```

## primevul_paired:train:1641 / addition_only / uncertain
```diff
--- 
+++ 
@@ -12,4 +12,7 @@
      if (NULL == mCodecCtx) {
          if (OK != initDecoder()) {
+            ALOGE("Failed to initialize decoder");
+            notify(OMX_EventError, OMX_ErrorUnsupportedSetting, 0, NULL);
+            mSignalledError = true;
              return;
          }
@@ -109,4 +112,23 @@
              status = ivdec_api_function(mCodecCtx, (void *)&s_dec_ip, (void *)&s_dec_op);
  
+            bool unsupportedResolution =
+                (IVD_STREAM_WIDTH_HEIGHT_NOT_SUPPORTED == (s_dec_op.u4_error_code & 0xFF));
+
+            /* Check for unsupported dimensions */
+            if (unsupportedResolution) {
+                ALOGE("Unsupported resolution : %dx%d", mWidth, mHeight);
+                notify(OMX_EventError, OMX_ErrorUnsupportedSetting, 0, NULL);
+                mSignalledError = true;
+                return;
+            }
+
+            bool allocationFailed = (IVD_MEM_ALLOC_FAILED == (s_dec_op.u4_error_code & 0xFF));
+            if (allocationFailed) {
+                ALOGE("Allocation failure in decoder");
+                notify(OMX_EventError, OMX_ErrorUnsupportedSetting, 0, NULL);
+                mSignalledError = true;
+                return;
+            }
+
              bool resChanged = (IVD_RES_CHANGED == (s_dec_op.u4_error_code & 0xFF));
  
```
### primevul_paired:train:3283
```c
   92:             WORD32 timeDelay, timeTaken;
   93:  size_t sizeY, sizeUV;
   94: 
   95:  if (!setDecodeArgs(&s_dec_ip, &s_dec_op, inHeader, outHeader, timeStampIx)) {
   96:                 ALOGE("Decoder arg setup failed");
   97:                 notify(OMX_EventError, OMX_ErrorUndefined, 0, NULL);
   98:                 mSignalledError = true;
  106: 
  107: 
  108:              IV_API_CALL_STATUS_T status;
  109:              status = ivdec_api_function(mCodecCtx, (void *)&s_dec_ip, (void *)&s_dec_op);
  110:  
  111:              bool resChanged = (IVD_RES_CHANGED == (s_dec_op.u4_error_code & 0xFF));
  112:  
  113:              GETTIME(&mTimeEnd, NULL);
  114:  /* Compute time taken for decode() */
  115:             TIME_DIFF(mTimeStart, mTimeEnd, timeTaken);
  116: 
  117:             ALOGV("timeTaken=%6d delay=%6d numBytes=%6d", timeTaken, timeDelay,
  118:                    s_dec_op.u4_num_bytes_consumed);
  119:  if (s_dec_op.u4_frame_decoded_flag && !mFlushNeeded) {
  120:                 mFlushNeeded = true;
  121:  }
  122: 
  123:  if ((inHeader != NULL) && (1 != s_dec_op.u4_frame_decoded_flag)) {
  124:  /* If the input did not contain picture data, then ignore
  125:                  * the associated timestamp */
  126:                 mTimeStampsValid[timeStampIx] = false;
  127:  }
  128: 
  129:  if (mChangingResolution && !s_dec_op.u4_output_present) {
  130:                 mChangingResolution = false;
  131:                 resetDecoder();
  132:                 resetPlugin();
  141:  continue;
  142:  }
  143: 
  144:  if ((0 < s_dec_op.u4_pic_wd) && (0 < s_dec_op.u4_pic_ht)) {
  145:  uint32_t width = s_dec_op.u4_pic_wd;
  146:  uint32_t height = s_dec_op.u4_pic_ht;
  147:  bool portWillReset = false;
  148:                 handlePortSettingsChange(&portWillReset, width, height);
  149: 
```
{"review_id": "primevul_paired:train:3283:line:94", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 94, "code": " if (!setDecodeArgs(&s_dec_ip, &s_dec_op, inHeader, outHeader, timeStampIx)) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:108", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 108, "code": "             status = ivdec_api_function(mCodecCtx, (void *)&s_dec_ip, (void *)&s_dec_op);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:110", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 110, "code": "             bool resChanged = (IVD_RES_CHANGED == (s_dec_op.u4_error_code & 0xFF));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:112", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 112, "code": "             GETTIME(&mTimeEnd, NULL);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:114", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 114, "code": "            TIME_DIFF(mTimeStart, mTimeEnd, timeTaken);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:116", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 116, "code": "            ALOGV(\"timeTaken=%6d delay=%6d numBytes=%6d\", timeTaken, timeDelay,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:117", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 117, "code": "                   s_dec_op.u4_num_bytes_consumed);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:118", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 118, "code": " if (s_dec_op.u4_frame_decoded_flag && !mFlushNeeded) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:122", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 122, "code": " if ((inHeader != NULL) && (1 != s_dec_op.u4_frame_decoded_flag)) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:128", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 128, "code": " if (mChangingResolution && !s_dec_op.u4_output_present) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:143", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 143, "code": " if ((0 < s_dec_op.u4_pic_wd) && (0 < s_dec_op.u4_pic_ht)) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:144", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 144, "code": " uint32_t width = s_dec_op.u4_pic_wd;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3283:line:145", "pair_key": "primevul_paired:train:1641", "sample_key": "primevul_paired:train:3283", "line": 145, "code": " uint32_t height = s_dec_op.u4_pic_ht;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:3284
```c
```

## primevul_paired:train:1651 / addition_only / uncertain
```diff
--- 
+++ 
@@ -13,4 +13,11 @@
      ALOGV("sectionLength %u, skip %u", sectionLength, mSkipBytes);
  
+
+    if(sectionLength < mSkipBytes) {
+        ALOGE("b/28333006");
+        android_errorWriteLog(0x534e4554, "28333006");
+        return false;
+    }
+
      sectionLength -= mSkipBytes;
  
```
### primevul_paired:train:3303
```c
    1: bool ATSParser::PSISection::isCRCOkay() const {
    2:  if (!isComplete()) {
    3:  return false;
    4:  }
    5:  uint8_t* data = mBuffer->data();
    6: 
    7:  if ((data[1] & 0x80) == 0) {
    8:  return true;
    9:  }
   10: 
   11: 
   12:      unsigned sectionLength = U16_AT(data + 1) & 0xfff;
   13:      ALOGV("sectionLength %u, skip %u", sectionLength, mSkipBytes);
   14:  
   15:      sectionLength -= mSkipBytes;
   16:  
   17:  uint32_t crc = 0xffffffff;
   18:  for(unsigned i = 0; i < sectionLength + 4 /* crc */; i++) {
   19:  uint8_t b = data[i];
   20:  int index = ((crc >> 24) ^ (b & 0xff)) & 0xff;
   21:         crc = CRC_TABLE[index] ^ (crc << 8);
   22:  }
   23:     ALOGV("crc: %08x\n", crc);
   24:  return (crc == 0);
   25: }
```
{"review_id": "primevul_paired:train:3303:line:1", "pair_key": "primevul_paired:train:1651", "sample_key": "primevul_paired:train:3303", "line": 1, "code": " if (!isComplete()) {", "cohort": "historical_discussed", "methods": ["historical"]}
{"review_id": "primevul_paired:train:3303:line:6", "pair_key": "primevul_paired:train:1651", "sample_key": "primevul_paired:train:3303", "line": 6, "code": " if ((data[1] & 0x80) == 0) {", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3303:line:12", "pair_key": "primevul_paired:train:1651", "sample_key": "primevul_paired:train:3303", "line": 12, "code": "     ALOGV(\"sectionLength %u, skip %u\", sectionLength, mSkipBytes);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3303:line:14", "pair_key": "primevul_paired:train:1651", "sample_key": "primevul_paired:train:3303", "line": 14, "code": "     sectionLength -= mSkipBytes;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3303:line:16", "pair_key": "primevul_paired:train:1651", "sample_key": "primevul_paired:train:3303", "line": 16, "code": " uint32_t crc = 0xffffffff;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3303:line:17", "pair_key": "primevul_paired:train:1651", "sample_key": "primevul_paired:train:3303", "line": 17, "code": " for(unsigned i = 0; i < sectionLength + 4 /* crc */; i++) {", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3303:line:18", "pair_key": "primevul_paired:train:1651", "sample_key": "primevul_paired:train:3303", "line": 18, "code": " uint8_t b = data[i];", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3303:line:19", "pair_key": "primevul_paired:train:1651", "sample_key": "primevul_paired:train:3303", "line": 19, "code": " int index = ((crc >> 24) ^ (b & 0xff)) & 0xff;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3303:line:20", "pair_key": "primevul_paired:train:1651", "sample_key": "primevul_paired:train:3303", "line": 20, "code": "        crc = CRC_TABLE[index] ^ (crc << 8);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3303:line:22", "pair_key": "primevul_paired:train:1651", "sample_key": "primevul_paired:train:3303", "line": 22, "code": "    ALOGV(\"crc: %08x\\n\", crc);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3303:line:23", "pair_key": "primevul_paired:train:1651", "sample_key": "primevul_paired:train:3303", "line": 23, "code": " return (crc == 0);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:3304
```c
```

## primevul_paired:train:1815 / addition_only / uncertain
```diff
--- 
+++ 
@@ -178,4 +178,6 @@
              break;
            Rd_WP_DWORD(image,&Rec.RecordLength);
+          if (Rec.RecordLength > GetBlobSize(image))
+            ThrowReaderException(CorruptImageError,"ImproperImageHeader");
            if(EOFBlob(image))
              break;
```
### primevul_paired:train:3631
```c
  167:   switch(Header.FileType)
  168:     {
  169:     case 1:     /* WPG level 1 */
  170:       while(!EOFBlob(image)) /* object parser loop */
  171:         {
  172:           (void) SeekBlob(image,Header.DataOffset,SEEK_SET);
  173:           if(EOFBlob(image))
  174:             break;
  175: 
  176:           Rec.RecType=(i=ReadBlobByte(image));
  177:            if(i==EOF)
  178:              break;
  179:            Rd_WP_DWORD(image,&Rec.RecordLength);
  180:            if(EOFBlob(image))
  181:              break;
  182:  
  183:           Header.DataOffset=TellBlob(image)+Rec.RecordLength;
  184: 
  185:           switch(Rec.RecType)
  186:             {
  187:             case 0x0B: /* bitmap type 1 */
  188:               BitmapHeader1.Width=ReadBlobLSBShort(image);
  189:               BitmapHeader1.Height=ReadBlobLSBShort(image);
  190:               if ((BitmapHeader1.Width == 0) || (BitmapHeader1.Height == 0))
  191:                 ThrowReaderException(CorruptImageError,"ImproperImageHeader");
  206:               goto UnpackRaster;
  207: 
  208:             case 0x0E:  /*Color palette */
  209:               WPG_Palette.StartIndex=ReadBlobLSBShort(image);
  210:               WPG_Palette.NumOfEntries=ReadBlobLSBShort(image);
  211:               if ((WPG_Palette.NumOfEntries-WPG_Palette.StartIndex) >
  212:                   (Rec2.RecordLength-2-2) / 3)
  227:               break;
  228: 
  229:             case 0x11:  /* Start PS l1 */
  230:               if(Rec.RecordLength > 8)
  231:                 image=ExtractPostscript(image,image_info,
  232:                   TellBlob(image)+8,   /* skip PS header in the wpg */
  233:                   (ssize_t) Rec.RecordLength-8,exception);
  234:               break;
  235: 
  236:             case 0x14:  /* bitmap type 2 */
  579:    }
  580: 
  581:  Finish:
  582:   (void) CloseBlob(image);
  583: 
  584:   {
  585:     Image
```
{"review_id": "primevul_paired:train:3631:line:169", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 169, "code": "      while(!EOFBlob(image)) /* object parser loop */", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:171", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 171, "code": "          (void) SeekBlob(image,Header.DataOffset,SEEK_SET);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:175", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 175, "code": "          Rec.RecType=(i=ReadBlobByte(image));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:178", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 178, "code": "           Rd_WP_DWORD(image,&Rec.RecordLength);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:179", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 179, "code": "           if(EOFBlob(image))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:182", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 182, "code": "          Header.DataOffset=TellBlob(image)+Rec.RecordLength;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:184", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 184, "code": "          switch(Rec.RecType)", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:187", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 187, "code": "              BitmapHeader1.Width=ReadBlobLSBShort(image);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:208", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 208, "code": "              WPG_Palette.StartIndex=ReadBlobLSBShort(image);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:229", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 229, "code": "              if(Rec.RecordLength > 8)", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:230", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 230, "code": "                image=ExtractPostscript(image,image_info,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:231", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 231, "code": "                  TellBlob(image)+8,   /* skip PS header in the wpg */", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:232", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 232, "code": "                  (ssize_t) Rec.RecordLength-8,exception);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:3631:line:581", "pair_key": "primevul_paired:train:1815", "sample_key": "primevul_paired:train:3631", "line": 581, "code": "  (void) CloseBlob(image);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:3632
```c
```

## primevul_paired:train:1830 / addition_only / legal
```diff
--- 
+++ 
@@ -79,4 +79,7 @@
  	}
  
+	if (p4 <= 0) {
+		return;
+	}
  	ut8 opcode = *(p4++) & 0xff;
  	while (keep) {
```
### primevul_paired:train:3661
```c
```
### primevul_paired:train:3662
```c
```

## primevul_paired:train:2349 / addition_only / uncertain
```diff
--- 
+++ 
@@ -118,6 +118,14 @@
 
 			case DNS_QUERY_PTR:
+			{
 				/* Reverse lookups just come back as char* */
 				resultstr = std::string((const char*)data.first);
+				if (resultstr.find_first_not_of("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-") != std::string::npos)
+				{
+					std::string ro = req->orig;
+					delete req;
+					return DNSResult(this_id | ERROR_MASK, "Invalid char(s) in reply", 0, ro);
+				}
+			}
 			break;
 
```
### primevul_paired:train:4699
```c
   37: 	}
   38: 
   39: 	/* Put the read header info into a header class */
   40: 	DNS::FillHeader(&header,buffer,length - 12);
   41: 
   42: 	/* Get the id of this request.
   43: 	 * Its a 16 bit value stored in two char's,
   44: 	 * so we use logic shifts to create the value.
   45: 	 */
   46: 	unsigned long this_id = header.id[1] + (header.id[0] << 8);
   47: 
   48: 	/* Do we have a pending request matching this id? */
   49: 	if (!requests[this_id])
   50: 	{
   51: 		/* Somehow we got a DNS response for a request we never made... */
   52: 		ServerInstance->Logs->Log("RESOLVER",DEBUG,"Hmm, got a result that we didn't ask for (id=%lx). Ignoring.", this_id);
   63: 	 * When its finished it will return a DNSInfo which is a pair of
   64: 	 * unsigned char* resource record data, and an error message.
   65: 	 */
   66: 	DNSInfo data = req->ResultIsReady(header, length);
   67: 	std::string resultstr;
   68: 
   69: 	/* Check if we got a result, if we didnt, its an error */
   70: 	if (data.first == NULL)
   71: 	{
   72: 		/* An error.
   73: 		 * Mask the ID with the value of ERROR_MASK, so that
   81: 	}
   82: 	else
   83: 	{
   84: 		unsigned long ttl = req->ttl;
   85: 		char formatted[128];
   86: 
   87: 		/* Forward lookups come back as binary data. We must format them into ascii */
   88: 		switch (req->type)
   89: 		{
   90: 			case DNS_QUERY_A:
   91: 				snprintf(formatted,16,"%u.%u.%u.%u",data.first[0],data.first[1],data.first[2],data.first[3]);
  118: 
  119: 			case DNS_QUERY_PTR:
  120: 				/* Reverse lookups just come back as char* */
  121: 				resultstr = std::string((const char*)data.first);
  122: 			break;
  123: 
  124: 			default:
  126: 		}
  127: 
  128: 		/* Build the reply with the id and hostname/ip in it */
  129: 		std::string ro = req->orig;
  130: 		DNSResult result = DNSResult(this_id,resultstr,ttl,ro,req->type);
  131: 		delete req;
  132: 		return result;
  133: 	}
  134: }
```
{"review_id": "primevul_paired:train:4699:line:39", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 39, "code": "\tDNS::FillHeader(&header,buffer,length - 12);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4699:line:45", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 45, "code": "\tunsigned long this_id = header.id[1] + (header.id[0] << 8);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4699:line:48", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 48, "code": "\tif (!requests[this_id])", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4699:line:65", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 65, "code": "\tDNSInfo data = req->ResultIsReady(header, length);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4699:line:69", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 69, "code": "\tif (data.first == NULL)", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4699:line:83", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 83, "code": "\t\tunsigned long ttl = req->ttl;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4699:line:87", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 87, "code": "\t\tswitch (req->type)", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4699:line:120", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 120, "code": "\t\t\t\tresultstr = std::string((const char*)data.first);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4699:line:128", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 128, "code": "\t\tstd::string ro = req->orig;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4699:line:129", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 129, "code": "\t\tDNSResult result = DNSResult(this_id,resultstr,ttl,ro,req->type);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4699:line:130", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 130, "code": "\t\tdelete req;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4699:line:131", "pair_key": "primevul_paired:train:2349", "sample_key": "primevul_paired:train:4699", "line": 131, "code": "\t\treturn result;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:4700
```c
```

## primevul_paired:train:2396 / addition_only / uncertain
```diff
--- 
+++ 
@@ -132,4 +132,9 @@
 
             auto master_iter = m_all_images.find(refs[0]);
+            if (master_iter == m_all_images.end()) {
+              return Error(heif_error_Invalid_input,
+                           heif_suberror_Nonexisting_item_referenced,
+                           "Non-existing alpha image referenced");
+            }
             if (image.get() == master_iter->second.get()) {
               return Error(heif_error_Invalid_input,
```
### primevul_paired:train:4793
```c
   68:           image->set_is_thumbnail_of(refs[0]);
   69: 
   70:           auto master_iter = m_all_images.find(refs[0]);
   71:           if (master_iter == m_all_images.end()) {
   72:             return Error(heif_error_Invalid_input,
   73:                          heif_suberror_Nonexisting_item_referenced,
   74:                          "Thumbnail references a non-existing image");
   75:           }
   76: 
   77:           if (master_iter->second->is_thumbnail()) {
   78:             return Error(heif_error_Invalid_input,
   79:                          heif_suberror_Nonexisting_item_referenced,
   80:                          "Thumbnail references another thumbnail");
   85:                          heif_suberror_Nonexisting_item_referenced,
   86:                          "Recursive thumbnail image detected");
   87:           }
   88:           master_iter->second->add_thumbnail(image);
   89: 
   90:           remove_top_level_image(image);
   91:         }
  126: 
  127:           // alpha channel
  128: 
  129:           if (auxC_property->get_aux_type() == "urn:mpeg:avc:2015:auxid:1" ||
  130:               auxC_property->get_aux_type() == "urn:mpeg:hevc:2015:auxid:1") {
  131:             image->set_is_alpha_channel_of(refs[0]);
  132: 
  133:             auto master_iter = m_all_images.find(refs[0]);
  134:             if (image.get() == master_iter->second.get()) {
  135:               return Error(heif_error_Invalid_input,
  136:                            heif_suberror_Nonexisting_item_referenced,
  137:                            "Recursive alpha image detected");
  138:             }
  139:             master_iter->second->set_alpha_channel(image);
  140:           }
  141: 
  142: 
  143:           // depth channel
  144: 
  145:           if (auxC_property->get_aux_type() == "urn:mpeg:hevc:2015:auxid:2") {
  146:             image->set_is_depth_channel_of(refs[0]);
  147: 
  148:             auto master_iter = m_all_images.find(refs[0]);
  149:             if (image.get() == master_iter->second.get()) {
  150:               return Error(heif_error_Invalid_input,
  151:                            heif_suberror_Nonexisting_item_referenced,
  152:                            "Recursive depth image detected");
  166:             }
  167:           }
  168: 
  169:           remove_top_level_image(image);
  170:         }
  171:         else {
  172:           // 'image' is a normal image, keep it as a top-level image
```
{"review_id": "primevul_paired:train:4793:line:70", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 70, "code": "          if (master_iter == m_all_images.end()) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:76", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 76, "code": "          if (master_iter->second->is_thumbnail()) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:87", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 87, "code": "          master_iter->second->add_thumbnail(image);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:128", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 128, "code": "          if (auxC_property->get_aux_type() == \"urn:mpeg:avc:2015:auxid:1\" ||", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:129", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 129, "code": "              auxC_property->get_aux_type() == \"urn:mpeg:hevc:2015:auxid:1\") {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:132", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 132, "code": "            auto master_iter = m_all_images.find(refs[0]);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:133", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 133, "code": "            if (image.get() == master_iter->second.get()) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:134", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 134, "code": "              return Error(heif_error_Invalid_input,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:135", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 135, "code": "                           heif_suberror_Nonexisting_item_referenced,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:136", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 136, "code": "                           \"Recursive alpha image detected\");", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:138", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 138, "code": "            master_iter->second->set_alpha_channel(image);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:144", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 144, "code": "          if (auxC_property->get_aux_type() == \"urn:mpeg:hevc:2015:auxid:2\") {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:145", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 145, "code": "            image->set_is_depth_channel_of(refs[0]);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:148", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 148, "code": "            if (image.get() == master_iter->second.get()) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4793:line:168", "pair_key": "primevul_paired:train:2396", "sample_key": "primevul_paired:train:4793", "line": 168, "code": "          remove_top_level_image(image);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:4794
```c
```

## primevul_paired:train:2426 / addition_only / legal
```diff
--- 
+++ 
@@ -20,4 +20,6 @@
             "of the input tensor: ",
             bias.shape().DebugString(), " vs. ", input.shape().DebugString()));
+    OP_REQUIRES(context, bias.NumElements() > 0,
+                errors::InvalidArgument("Must provide at least 1 bias"));
 
     Tensor* output = nullptr;
```
### primevul_paired:train:4853
```c
```
### primevul_paired:train:4854
```c
```

## primevul_paired:train:2455 / addition_only / uncertain
```diff
--- 
+++ 
@@ -4,4 +4,7 @@
     // make a copy of our configuration bytes
     unsigned int payload_size = size-AP4_ATOM_HEADER_SIZE;
+
+    // keep a raw copy
+    if (payload_size < 22) return;
     m_RawBytes.SetData(payload, payload_size);
 
```
### primevul_paired:train:4911
```c
    3: {
    4:     // make a copy of our configuration bytes
    5:     unsigned int payload_size = size-AP4_ATOM_HEADER_SIZE;
    6:     m_RawBytes.SetData(payload, payload_size);
    7: 
    8:     // parse the payload
    9:     m_ConfigurationVersion   = payload[0];
   10:     m_GeneralProfileSpace    = (payload[1]>>6) & 0x03;
   11:     m_GeneralTierFlag        = (payload[1]>>5) & 0x01;
   12:     m_GeneralProfile         = (payload[1]   ) & 0x1F;
   13:     m_GeneralProfileCompatibilityFlags = AP4_BytesToUInt32BE(&payload[2]);
   14:     m_GeneralConstraintIndicatorFlags  = (((AP4_UI64)AP4_BytesToUInt32BE(&payload[6]))<<16) | AP4_BytesToUInt16BE(&payload[10]);
   15:     m_GeneralLevel           = payload[12];
   16:     m_Reserved1              = (payload[13]>>4) & 0x0F;
   17:     m_MinSpatialSegmentation = AP4_BytesToUInt16BE(&payload[13]) & 0x0FFF;
   18:     m_Reserved2              = (payload[15]>>2) & 0x3F;
   19:     m_ParallelismType        = payload[15] & 0x03;
   20:     m_Reserved3              = (payload[16]>>2) & 0x3F;
   21:     m_ChromaFormat           = payload[16] & 0x03;
   22:     m_Reserved4              = (payload[17]>>3) & 0x1F;
```
{"review_id": "primevul_paired:train:4911:line:5", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 5, "code": "    m_RawBytes.SetData(payload, payload_size);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4911:line:8", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 8, "code": "    m_ConfigurationVersion   = payload[0];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4911:line:9", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 9, "code": "    m_GeneralProfileSpace    = (payload[1]>>6) & 0x03;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4911:line:10", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 10, "code": "    m_GeneralTierFlag        = (payload[1]>>5) & 0x01;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4911:line:11", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 11, "code": "    m_GeneralProfile         = (payload[1]   ) & 0x1F;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4911:line:12", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 12, "code": "    m_GeneralProfileCompatibilityFlags = AP4_BytesToUInt32BE(&payload[2]);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4911:line:13", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 13, "code": "    m_GeneralConstraintIndicatorFlags  = (((AP4_UI64)AP4_BytesToUInt32BE(&payload[6]))<<16) | AP4_BytesToUInt16BE(&payload[10]);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4911:line:14", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 14, "code": "    m_GeneralLevel           = payload[12];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4911:line:15", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 15, "code": "    m_Reserved1              = (payload[13]>>4) & 0x0F;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4911:line:16", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 16, "code": "    m_MinSpatialSegmentation = AP4_BytesToUInt16BE(&payload[13]) & 0x0FFF;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4911:line:17", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 17, "code": "    m_Reserved2              = (payload[15]>>2) & 0x3F;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:4911:line:18", "pair_key": "primevul_paired:train:2455", "sample_key": "primevul_paired:train:4911", "line": 18, "code": "    m_ParallelismType        = payload[15] & 0x03;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:4912
```c
```

## primevul_paired:train:2532 / addition_only / uncertain
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
### primevul_paired:train:5065
```c
   11:     if (s->buf[0] < 0x10)
   12:     {
   13:         av_log(s->avctx, AV_LOG_ERROR, "invalid header size (%d)\n", s->buf[0]);
   14:         return AVERROR_INVALIDDATA;
   15:     }
   16: 
   17:     /* unscramble the header bytes with a XOR operation */
   18:     for (i = 1; i < header.header_size; i++)
   19:         header_buffer[i - 1] = s->buf[i] ^ s->buf[i + 1];
   20: 
   21:     header.compression = header_buffer[0];
   22:     header.deltaset = header_buffer[1];
   23:     header.vectable = header_buffer[2];
   24:     header.ysize = AV_RL16(&header_buffer[3]);
   25:     header.xsize = AV_RL16(&header_buffer[5]);
   26:     header.checksum = AV_RL16(&header_buffer[7]);
   27:     header.version = header_buffer[9];
   28:     header.header_type = header_buffer[10];
   29:     header.flags = header_buffer[11];
   30:     header.control = header_buffer[12];
   31: 
   32:     /* Version 2 */
```
{"review_id": "primevul_paired:train:5065:line:13", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 13, "code": "        return AVERROR_INVALIDDATA;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5065:line:17", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 17, "code": "    for (i = 1; i < header.header_size; i++)", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5065:line:18", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 18, "code": "        header_buffer[i - 1] = s->buf[i] ^ s->buf[i + 1];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5065:line:20", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 20, "code": "    header.compression = header_buffer[0];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5065:line:21", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 21, "code": "    header.deltaset = header_buffer[1];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5065:line:22", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 22, "code": "    header.vectable = header_buffer[2];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5065:line:23", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 23, "code": "    header.ysize = AV_RL16(&header_buffer[3]);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5065:line:24", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 24, "code": "    header.xsize = AV_RL16(&header_buffer[5]);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5065:line:25", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 25, "code": "    header.checksum = AV_RL16(&header_buffer[7]);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5065:line:26", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 26, "code": "    header.version = header_buffer[9];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5065:line:27", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 27, "code": "    header.header_type = header_buffer[10];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5065:line:28", "pair_key": "primevul_paired:train:2532", "sample_key": "primevul_paired:train:5065", "line": 28, "code": "    header.flags = header_buffer[11];", "cohort": "seed42_check_sample", "methods": ["historical"]}
### primevul_paired:train:5066
```c
```

## primevul_paired:train:2548 / addition_only / legal
```diff
--- 
+++ 
@@ -134,4 +134,5 @@
   if (group4_image == (Image *) NULL)
     {
+      write_info=DestroyImageInfo(write_info);
       (void) CloseBlob(image);
       return(MagickFalse);
@@ -142,4 +143,5 @@
   if (group4 == (unsigned char *) NULL)
     {
+      write_info=DestroyImageInfo(write_info);
       (void) CloseBlob(image);
       return(MagickFalse);
```
### primevul_paired:train:5097
```c
   32:     Open output image file.
   33:   */
   34:   assert(image_info != (const ImageInfo *) NULL);
   35:   assert(image_info->signature == MagickCoreSignature);
   36:   assert(image != (Image *) NULL);
   37:   assert(image->signature == MagickCoreSignature);
   38:   if (image->debug != MagickFalse)
   39:     (void) LogMagickEvent(TraceEvent,GetMagickModule(),"%s",image->filename);
   40:   assert(exception != (ExceptionInfo *) NULL);
   41:   assert(exception->signature == MagickCoreSignature);
   42:   status=OpenBlob(image_info,image,WriteBinaryBlobMode,exception);
   43:   if (status == MagickFalse)
   44:     return(status);
   45:   /*
   46:     Create standard CALS header.
  109:     (unsigned long) image->columns,(unsigned long) image->rows);
  110:   count=WriteCALSRecord(image,header);  
  111:   density=200;
  112:   if (image_info->density != (char *) NULL)
  113:     {
  114:       GeometryInfo
  115:         geometry_info;
  116: 
  117:       (void) ParseGeometry(image_info->density,&geometry_info);
  118:       density=(size_t) floor(geometry_info.rho+0.5);
  119:     }
  120:   (void) FormatLocaleString(header,sizeof(header),"rdensty: %04lu",
  127:   /*
  128:     Write CALS pixels.
  129:   */
  130:   write_info=CloneImageInfo(image_info);
  131:   (void) CopyMagickString(write_info->filename,"GROUP4:",MagickPathExtent);
  132:   (void) CopyMagickString(write_info->magick,"GROUP4",MagickPathExtent);
  133:   group4_image=CloneImage(image,0,0,MagickTrue,exception);
  134:   if (group4_image == (Image *) NULL)
  135:     {
  136:       (void) CloseBlob(image);
  137:       return(MagickFalse);
  138:     }
  139:   group4=(unsigned char *) ImageToBlob(write_info,group4_image,&length,
  140:     exception);
  141:   group4_image=DestroyImage(group4_image);
  142:   if (group4 == (unsigned char *) NULL)
  143:     {
  144:       (void) CloseBlob(image);
  145:       return(MagickFalse);
```
{"review_id": "primevul_paired:train:5097:line:34", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 34, "code": "  assert(image_info->signature == MagickCoreSignature);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:41", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 41, "code": "  status=OpenBlob(image_info,image,WriteBinaryBlobMode,exception);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:42", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 42, "code": "  if (status == MagickFalse)", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:111", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 111, "code": "  if (image_info->density != (char *) NULL)", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:116", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 116, "code": "      (void) ParseGeometry(image_info->density,&geometry_info);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:129", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 129, "code": "  write_info=CloneImageInfo(image_info);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:130", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 130, "code": "  (void) CopyMagickString(write_info->filename,\"GROUP4:\",MagickPathExtent);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:131", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 131, "code": "  (void) CopyMagickString(write_info->magick,\"GROUP4\",MagickPathExtent);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:132", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 132, "code": "  group4_image=CloneImage(image,0,0,MagickTrue,exception);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:133", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 133, "code": "  if (group4_image == (Image *) NULL)", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:138", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 138, "code": "  group4=(unsigned char *) ImageToBlob(write_info,group4_image,&length,", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:139", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 139, "code": "    exception);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5097:line:141", "pair_key": "primevul_paired:train:2548", "sample_key": "primevul_paired:train:5097", "line": 141, "code": "  if (group4 == (unsigned char *) NULL)", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:5098
```c
```

## primevul_paired:train:2721 / addition_only / uncertain
```diff
--- 
+++ 
@@ -115,4 +115,12 @@
                 offset += 8;
 
+                if (s->sectorcounts[i] > DMG_SECTORCOUNTS_MAX) {
+                    error_report("sector count %" PRIu64 " for chunk %u is "
+                                 "larger than max (%u)",
+                                 s->sectorcounts[i], i, DMG_SECTORCOUNTS_MAX);
+                    ret = -EINVAL;
+                    goto fail;
+                }
+
                 ret = read_uint64(bs, offset, &s->offsets[i]);
                 if (ret < 0) {
@@ -127,4 +135,12 @@
                 }
                 offset += 8;
+
+                if (s->lengths[i] > DMG_LENGTHS_MAX) {
+                    error_report("length %" PRIu64 " for chunk %u is larger "
+                                 "than max (%u)",
+                                 s->lengths[i], i, DMG_LENGTHS_MAX);
+                    ret = -EINVAL;
+                    goto fail;
+                }
 
                 if (s->lengths[i] > max_compressed_size) {
```
### primevul_paired:train:5443
```c
   81:             s->sectors = g_realloc(s->sectors, new_size);
   82:             s->sectorcounts = g_realloc(s->sectorcounts, new_size);
   83: 
   84:             for (i = s->n_chunks; i < s->n_chunks + chunk_count; i++) {
   85:                 ret = read_uint32(bs, offset, &s->types[i]);
   86:                 if (ret < 0) {
   87:                     goto fail;
  114:                 }
  115:                 offset += 8;
  116: 
  117:                 ret = read_uint64(bs, offset, &s->offsets[i]);
  118:                 if (ret < 0) {
  119:                     goto fail;
  120:                 }
  121:                 s->offsets[i] += last_in_offset;
  130:                 if (s->lengths[i] > max_compressed_size) {
  131:                     max_compressed_size = s->lengths[i];
  132:                 }
  133:                 if (s->sectorcounts[i] > max_sectors_per_chunk) {
  134:                     max_sectors_per_chunk = s->sectorcounts[i];
  135:                 }
  136:             }
  152:     return 0;
  153: 
  154: fail:
  155:     g_free(s->types);
  156:     g_free(s->offsets);
  157:     g_free(s->lengths);
  158:     g_free(s->sectors);
  159:     g_free(s->sectorcounts);
  160:     g_free(s->compressed_chunk);
  161:     g_free(s->uncompressed_chunk);
  162:     return ret;
  163: }
```
{"review_id": "primevul_paired:train:5443:line:83", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 83, "code": "            for (i = s->n_chunks; i < s->n_chunks + chunk_count; i++) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5443:line:116", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 116, "code": "                ret = read_uint64(bs, offset, &s->offsets[i]);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5443:line:117", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 117, "code": "                if (ret < 0) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5443:line:132", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 132, "code": "                if (s->sectorcounts[i] > max_sectors_per_chunk) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5443:line:154", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 154, "code": "    g_free(s->types);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5443:line:155", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 155, "code": "    g_free(s->offsets);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5443:line:156", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 156, "code": "    g_free(s->lengths);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5443:line:157", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 157, "code": "    g_free(s->sectors);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5443:line:158", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 158, "code": "    g_free(s->sectorcounts);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5443:line:159", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 159, "code": "    g_free(s->compressed_chunk);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5443:line:160", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 160, "code": "    g_free(s->uncompressed_chunk);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:5443:line:161", "pair_key": "primevul_paired:train:2721", "sample_key": "primevul_paired:train:5443", "line": 161, "code": "    return ret;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:5444
```c
```

## primevul_paired:train:2752 / addition_only / legal
```diff
--- 
+++ 
@@ -20,4 +20,11 @@
 		memset(&pdu_header, 0, sizeof(pdu_header));
 		memset(&msg, 0, sizeof(msg));
+
+		if (urb->actual_length > 0 && !urb->transfer_buffer) {
+			dev_err(&sdev->udev->dev,
+				"urb: actual_length %d transfer_buffer null\n",
+				urb->actual_length);
+			return -1;
+		}
 
 		if (usb_pipetype(urb->pipe) == PIPE_ISOCHRONOUS)
```
### primevul_paired:train:5505
```c
```
### primevul_paired:train:5506
```c
```

## primevul_paired:train:3735 / addition_only / uncertain
```diff
--- 
+++ 
@@ -79,4 +79,9 @@
 	// Find out if we have a quote in the selection.
 	while (i <= col_end)
+	{
+	    // check for going over the end of the line, which can happen if
+	    // the line was changed after the Visual area was selected.
+	    if (line[i] == NUL)
+		break;
 	    if (line[i++] == quotechar)
 	    {
@@ -84,4 +89,5 @@
 		break;
 	    }
+	}
     }
 
```
### primevul_paired:train:7471
```c
```
### primevul_paired:train:7472
```c
```

## primevul_paired:train:641 / addition_only / legal
```diff
--- 
+++ 
@@ -5,4 +5,8 @@
  	void *buffer;
  
+	if (params->buffer.fragment_size == 0 ||
+	    params->buffer.fragments > SIZE_MAX / params->buffer.fragment_size)
+		return -EINVAL;
+
  	buffer_size = params->buffer.fragment_size * params->buffer.fragments;
  	if (stream->ops->copy) {
```
### primevul_paired:train:1283
```c
    4:  	unsigned int buffer_size;
    5:  	void *buffer;
    6:  
    7:  	buffer_size = params->buffer.fragment_size * params->buffer.fragments;
    8:  	if (stream->ops->copy) {
    9:  		buffer = NULL;
   10: 		/* if copy is defined the driver will be required to copy
   11: 		 * the data from core
   12: 		 */
   13: 	} else {
   14: 		buffer = kmalloc(buffer_size, GFP_KERNEL);
   15: 		if (!buffer)
   16: 			return -ENOMEM;
   17: 	}
   18: 	stream->runtime->fragment_size = params->buffer.fragment_size;
   19: 	stream->runtime->fragments = params->buffer.fragments;
   20: 	stream->runtime->buffer = buffer;
   21: 	stream->runtime->buffer_size = buffer_size;
   22: 	return 0;
   23: }
```
{"review_id": "primevul_paired:train:1283:line:6", "pair_key": "primevul_paired:train:641", "sample_key": "primevul_paired:train:1283", "line": 6, "code": " \tbuffer_size = params->buffer.fragment_size * params->buffer.fragments;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional", "targeted"]}
{"review_id": "primevul_paired:train:1283:line:7", "pair_key": "primevul_paired:train:641", "sample_key": "primevul_paired:train:1283", "line": 7, "code": " \tif (stream->ops->copy) {", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional", "targeted"]}
{"review_id": "primevul_paired:train:1283:line:8", "pair_key": "primevul_paired:train:641", "sample_key": "primevul_paired:train:1283", "line": 8, "code": " \t\tbuffer = NULL;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1283:line:13", "pair_key": "primevul_paired:train:641", "sample_key": "primevul_paired:train:1283", "line": 13, "code": "\t\tbuffer = kmalloc(buffer_size, GFP_KERNEL);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional", "targeted"]}
{"review_id": "primevul_paired:train:1283:line:14", "pair_key": "primevul_paired:train:641", "sample_key": "primevul_paired:train:1283", "line": 14, "code": "\t\tif (!buffer)", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1283:line:15", "pair_key": "primevul_paired:train:641", "sample_key": "primevul_paired:train:1283", "line": 15, "code": "\t\t\treturn -ENOMEM;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1283:line:17", "pair_key": "primevul_paired:train:641", "sample_key": "primevul_paired:train:1283", "line": 17, "code": "\tstream->runtime->fragment_size = params->buffer.fragment_size;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1283:line:18", "pair_key": "primevul_paired:train:641", "sample_key": "primevul_paired:train:1283", "line": 18, "code": "\tstream->runtime->fragments = params->buffer.fragments;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1283:line:19", "pair_key": "primevul_paired:train:641", "sample_key": "primevul_paired:train:1283", "line": 19, "code": "\tstream->runtime->buffer = buffer;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1283:line:20", "pair_key": "primevul_paired:train:641", "sample_key": "primevul_paired:train:1283", "line": 20, "code": "\tstream->runtime->buffer_size = buffer_size;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1283:line:21", "pair_key": "primevul_paired:train:641", "sample_key": "primevul_paired:train:1283", "line": 21, "code": "\treturn 0;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:1284
```c
```

## primevul_paired:train:664 / addition_only / uncertain
```diff
--- 
+++ 
@@ -25,4 +25,8 @@
  			goto next_desc;
  		}
+		if ((buflen < elength) || (elength < 3)) {
+			dev_err(&intf->dev, "invalid descriptor buffer length\n");
+			break;
+		}
  		if (buffer[1] != USB_DT_CS_INTERFACE) {
  			dev_err(&intf->dev, "skipping garbage\n");
```
### primevul_paired:train:1329
```c
   24:  			elength = 1;
   25:  			goto next_desc;
   26:  		}
   27:  		if (buffer[1] != USB_DT_CS_INTERFACE) {
   28:  			dev_err(&intf->dev, "skipping garbage\n");
   29:  			goto next_desc;
   30: 		}
   31: 
   32: 		switch (buffer[2]) {
   33: 		case USB_CDC_UNION_TYPE: /* we've found it */
   34: 			if (elength < sizeof(struct usb_cdc_union_desc))
   35: 				goto next_desc;
   36: 			if (union_header) {
   37: 				dev_err(&intf->dev, "More than one union descriptor, skipping ...\n");
   40: 			union_header = (struct usb_cdc_union_desc *)buffer;
   41: 			break;
   42: 		case USB_CDC_COUNTRY_TYPE:
   43: 			if (elength < sizeof(struct usb_cdc_country_functional_desc))
   44: 				goto next_desc;
   45: 			hdr->usb_cdc_country_functional_desc =
   46: 				(struct usb_cdc_country_functional_desc *)buffer;
   47: 			break;
   48: 		case USB_CDC_HEADER_TYPE:
   49: 			if (elength != sizeof(struct usb_cdc_header_desc))
   50: 				goto next_desc;
   51: 			if (header)
   52: 				return -EINVAL;
   53: 			header = (struct usb_cdc_header_desc *)buffer;
   54: 			break;
   55: 		case USB_CDC_ACM_TYPE:
   56: 			if (elength < sizeof(struct usb_cdc_acm_descriptor))
   57: 				goto next_desc;
   58: 			hdr->usb_cdc_acm_descriptor =
   59: 				(struct usb_cdc_acm_descriptor *)buffer;
   60: 			break;
   61: 		case USB_CDC_ETHERNET_TYPE:
   62: 			if (elength != sizeof(struct usb_cdc_ether_desc))
   63: 				goto next_desc;
   64: 			if (ether)
   65: 				return -EINVAL;
   66: 			ether = (struct usb_cdc_ether_desc *)buffer;
   67: 			break;
   68: 		case USB_CDC_CALL_MANAGEMENT_TYPE:
   69: 			if (elength < sizeof(struct usb_cdc_call_mgmt_descriptor))
   70: 				goto next_desc;
   71: 			hdr->usb_cdc_call_mgmt_descriptor =
   72: 				(struct usb_cdc_call_mgmt_descriptor *)buffer;
   73: 			break;
   74: 		case USB_CDC_DMM_TYPE:
   75: 			if (elength < sizeof(struct usb_cdc_dmm_desc))
   76: 				goto next_desc;
   77: 			hdr->usb_cdc_dmm_desc =
   78: 				(struct usb_cdc_dmm_desc *)buffer;
   79: 			break;
   80: 		case USB_CDC_MDLM_TYPE:
   81: 			if (elength < sizeof(struct usb_cdc_mdlm_desc *))
   82: 				goto next_desc;
   83: 			if (desc)
   84: 				return -EINVAL;
   85: 			desc = (struct usb_cdc_mdlm_desc *)buffer;
   86: 			break;
   87: 		case USB_CDC_MDLM_DETAIL_TYPE:
   88: 			if (elength < sizeof(struct usb_cdc_mdlm_detail_desc *))
   89: 				goto next_desc;
   90: 			if (detail)
   91: 				return -EINVAL;
   92: 			detail = (struct usb_cdc_mdlm_detail_desc *)buffer;
   93: 			break;
   94: 		case USB_CDC_NCM_TYPE:
   95: 			if (elength < sizeof(struct usb_cdc_ncm_desc))
   96: 				goto next_desc;
   97: 			hdr->usb_cdc_ncm_desc = (struct usb_cdc_ncm_desc *)buffer;
   98: 			break;
   99: 		case USB_CDC_MBIM_TYPE:
  100: 			if (elength < sizeof(struct usb_cdc_mbim_desc))
  101: 				goto next_desc;
  102: 
  103: 			hdr->usb_cdc_mbim_desc = (struct usb_cdc_mbim_desc *)buffer;
```
{"review_id": "primevul_paired:train:1329:line:26", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 26, "code": " \t\tif (buffer[1] != USB_DT_CS_INTERFACE) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1329:line:33", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 33, "code": "\t\t\tif (elength < sizeof(struct usb_cdc_union_desc))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1329:line:42", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 42, "code": "\t\t\tif (elength < sizeof(struct usb_cdc_country_functional_desc))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1329:line:48", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 48, "code": "\t\t\tif (elength != sizeof(struct usb_cdc_header_desc))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1329:line:55", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 55, "code": "\t\t\tif (elength < sizeof(struct usb_cdc_acm_descriptor))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1329:line:61", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 61, "code": "\t\t\tif (elength != sizeof(struct usb_cdc_ether_desc))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1329:line:68", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 68, "code": "\t\t\tif (elength < sizeof(struct usb_cdc_call_mgmt_descriptor))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1329:line:74", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 74, "code": "\t\t\tif (elength < sizeof(struct usb_cdc_dmm_desc))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1329:line:80", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 80, "code": "\t\t\tif (elength < sizeof(struct usb_cdc_mdlm_desc *))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1329:line:87", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 87, "code": "\t\t\tif (elength < sizeof(struct usb_cdc_mdlm_detail_desc *))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1329:line:94", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 94, "code": "\t\t\tif (elength < sizeof(struct usb_cdc_ncm_desc))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1329:line:99", "pair_key": "primevul_paired:train:664", "sample_key": "primevul_paired:train:1329", "line": 99, "code": "\t\t\tif (elength < sizeof(struct usb_cdc_mbim_desc))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:1330
```c
```

## primevul_paired:train:937 / addition_only / uncertain
```diff
--- 
+++ 
@@ -48,4 +48,6 @@
    if (acptr) {
      if (first) {
+      if (*parv[1] == ':' || strchr(parv[1], ' '))
+		return exit_client(cptr, sptr, sptr, "Malformed AUTHENTICATE");
        if (!EmptyString(cli_sslclifp(cptr)))
          sendcmdto_one(&me, CMD_SASL, acptr, "%C %C!%u.%u S %s :%s", acptr, &me,
@@ -65,4 +67,6 @@
    } else {
      if (first) {
+      if (*parv[1] == ':' || strchr(parv[1], ' '))
+        return exit_client(cptr, sptr, sptr, "Malformed AUTHENTICATE");
        if (!EmptyString(cli_sslclifp(cptr)))
          sendcmdto_serv_butone(&me, CMD_SASL, cptr, "* %C!%u.%u S %s :%s", &me,
```
### primevul_paired:train:1875
```c
   47:  
   48:    if (acptr) {
   49:      if (first) {
   50:        if (!EmptyString(cli_sslclifp(cptr)))
   51:          sendcmdto_one(&me, CMD_SASL, acptr, "%C %C!%u.%u S %s :%s", acptr, &me,
   52:                        cli_fd(cptr), cli_saslcookie(cptr),
   53:                       parv[1], cli_sslclifp(cptr));
   54:       else
   55:         sendcmdto_one(&me, CMD_SASL, acptr, "%C %C!%u.%u S :%s", acptr, &me,
   56:                       cli_fd(cptr), cli_saslcookie(cptr), parv[1]);
   57:       if (feature_bool(FEAT_SASL_SENDHOST))
   58:         sendcmdto_one(&me, CMD_SASL, acptr, "%C %C!%u.%u H :%s@%s:%s", acptr, &me,
   59:                       cli_fd(cptr), cli_saslcookie(cptr), cli_username(cptr),
   60:                       realhost, cli_sock_ip(cptr));
   61:     } else {
   62:       sendcmdto_one(&me, CMD_SASL, acptr, "%C %C!%u.%u C :%s", acptr, &me,
   63:                     cli_fd(cptr), cli_saslcookie(cptr), parv[1]);
   64:      }
   65:    } else {
   66:      if (first) {
   67:        if (!EmptyString(cli_sslclifp(cptr)))
   68:          sendcmdto_serv_butone(&me, CMD_SASL, cptr, "* %C!%u.%u S %s :%s", &me,
   69:                                cli_fd(cptr), cli_saslcookie(cptr),
   70:                               parv[1], cli_sslclifp(cptr));
   71:       else
   72:         sendcmdto_serv_butone(&me, CMD_SASL, cptr, "* %C!%u.%u S :%s", &me,
   73:                               cli_fd(cptr), cli_saslcookie(cptr), parv[1]);
   74:       if (feature_bool(FEAT_SASL_SENDHOST))
   75:         sendcmdto_serv_butone(&me, CMD_SASL, cptr, "* %C!%u.%u H :%s@%s:%s", &me,
   76:                               cli_fd(cptr), cli_saslcookie(cptr), cli_username(cptr),
   77:                               realhost, cli_sock_ip(cptr));
   81:     }
   82:   }
   83: 
   84:   if (!t_active(&cli_sasltimeout(cptr)))
   85:     timer_add(timer_init(&cli_sasltimeout(cptr)), sasl_timeout_callback, (void*) cptr,
   86:               TT_RELATIVE, feature_int(FEAT_SASL_TIMEOUT));
   87: 
   88:   return 0;
   89: }
```
{"review_id": "primevul_paired:train:1875:line:49", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 49, "code": "       if (!EmptyString(cli_sslclifp(cptr)))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:50", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 50, "code": "         sendcmdto_one(&me, CMD_SASL, acptr, \"%C %C!%u.%u S %s :%s\", acptr, &me,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:51", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 51, "code": "                       cli_fd(cptr), cli_saslcookie(cptr),", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:52", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 52, "code": "                      parv[1], cli_sslclifp(cptr));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:54", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 54, "code": "        sendcmdto_one(&me, CMD_SASL, acptr, \"%C %C!%u.%u S :%s\", acptr, &me,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:55", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 55, "code": "                      cli_fd(cptr), cli_saslcookie(cptr), parv[1]);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:56", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 56, "code": "      if (feature_bool(FEAT_SASL_SENDHOST))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:57", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 57, "code": "        sendcmdto_one(&me, CMD_SASL, acptr, \"%C %C!%u.%u H :%s@%s:%s\", acptr, &me,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:58", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 58, "code": "                      cli_fd(cptr), cli_saslcookie(cptr), cli_username(cptr),", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:59", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 59, "code": "                      realhost, cli_sock_ip(cptr));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:66", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 66, "code": "       if (!EmptyString(cli_sslclifp(cptr)))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:67", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 67, "code": "         sendcmdto_serv_butone(&me, CMD_SASL, cptr, \"* %C!%u.%u S %s :%s\", &me,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:68", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 68, "code": "                               cli_fd(cptr), cli_saslcookie(cptr),", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:69", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 69, "code": "                              parv[1], cli_sslclifp(cptr));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:71", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 71, "code": "        sendcmdto_serv_butone(&me, CMD_SASL, cptr, \"* %C!%u.%u S :%s\", &me,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:72", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 72, "code": "                              cli_fd(cptr), cli_saslcookie(cptr), parv[1]);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:73", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 73, "code": "      if (feature_bool(FEAT_SASL_SENDHOST))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:83", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 83, "code": "  if (!t_active(&cli_sasltimeout(cptr)))", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:84", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 84, "code": "    timer_add(timer_init(&cli_sasltimeout(cptr)), sasl_timeout_callback, (void*) cptr,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:85", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 85, "code": "              TT_RELATIVE, feature_int(FEAT_SASL_TIMEOUT));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:train:1875:line:87", "pair_key": "primevul_paired:train:937", "sample_key": "primevul_paired:train:1875", "line": 87, "code": "  return 0;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:train:1876
```c
```