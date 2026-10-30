# Fixed slicing source review

0-based stored / 1-based displayed physical rows. Assistant review, not independent vulnerability gold.

## primevul_paired:valid:104 / addition_only / uncertain
```diff
--- 
+++ 
@@ -22,4 +22,5 @@
 			stco = (GF_ChunkOffsetBox *)stbl->ChunkOffset;
 			if (!stco->offsets) return GF_ISOM_INVALID_FILE;
+			if (stco->nb_entries < sampleNumber) return GF_ISOM_INVALID_FILE;
 
 			(*offset) = (u64) stco->offsets[sampleNumber - 1];
@@ -27,4 +28,5 @@
 			co64 = (GF_ChunkLargeOffsetBox *)stbl->ChunkOffset;
 			if (!co64->offsets) return GF_ISOM_INVALID_FILE;
+			if (co64->nb_entries < sampleNumber) return GF_ISOM_INVALID_FILE;
 
 			(*offset) = co64->offsets[sampleNumber - 1];
```
### primevul_paired:valid:209
```c
    7: 	GF_StscEntry *ent;
    8: 
    9: 	(*offset) = 0;
   10: 	(*chunkNumber) = (*descIndex) = 0;
   11: 	if (out_ent) (*out_ent) = NULL;
   12: 	if (!stbl || !sampleNumber) return GF_BAD_PARAM;
   13: 	if (!stbl->ChunkOffset || !stbl->SampleToChunk || !stbl->SampleSize) return GF_ISOM_INVALID_FILE;
   14: 
   15: 	if (stbl->SampleSize && stbl->SampleToChunk->nb_entries == stbl->SampleSize->sampleCount) {
   16: 		ent = &stbl->SampleToChunk->entries[sampleNumber-1];
   17: 		if (!ent) return GF_BAD_PARAM;
   18: 		(*descIndex) = ent->sampleDescriptionIndex;
   19: 		(*chunkNumber) = sampleNumber;
   20: 		if (out_ent) *out_ent = ent;
   21: 		if ( stbl->ChunkOffset->type == GF_ISOM_BOX_TYPE_STCO) {
   22: 			stco = (GF_ChunkOffsetBox *)stbl->ChunkOffset;
   23: 			if (!stco->offsets) return GF_ISOM_INVALID_FILE;
   24: 
   25: 			(*offset) = (u64) stco->offsets[sampleNumber - 1];
   26: 		} else {
   27: 			co64 = (GF_ChunkLargeOffsetBox *)stbl->ChunkOffset;
   28: 			if (!co64->offsets) return GF_ISOM_INVALID_FILE;
   29: 
   30: 			(*offset) = co64->offsets[sampleNumber - 1];
   31: 		}
   32: 		return GF_OK;
   33: 	}
   34: 
   35: 	//check our cache: if desired sample is at or above current cache entry, start from here
```
{"review_id": "primevul_paired:valid:209:line:9", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 9, "code": "\t(*chunkNumber) = (*descIndex) = 0;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:209:line:11", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 11, "code": "\tif (!stbl || !sampleNumber) return GF_BAD_PARAM;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:209:line:15", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 15, "code": "\t\tent = &stbl->SampleToChunk->entries[sampleNumber-1];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:209:line:16", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 16, "code": "\t\tif (!ent) return GF_BAD_PARAM;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:209:line:18", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 18, "code": "\t\t(*chunkNumber) = sampleNumber;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:209:line:21", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 21, "code": "\t\t\tstco = (GF_ChunkOffsetBox *)stbl->ChunkOffset;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:209:line:22", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 22, "code": "\t\t\tif (!stco->offsets) return GF_ISOM_INVALID_FILE;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional", "targeted"]}
{"review_id": "primevul_paired:valid:209:line:24", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 24, "code": "\t\t\t(*offset) = (u64) stco->offsets[sampleNumber - 1];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional", "targeted"]}
{"review_id": "primevul_paired:valid:209:line:26", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 26, "code": "\t\t\tco64 = (GF_ChunkLargeOffsetBox *)stbl->ChunkOffset;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:209:line:27", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 27, "code": "\t\t\tif (!co64->offsets) return GF_ISOM_INVALID_FILE;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional", "targeted"]}
{"review_id": "primevul_paired:valid:209:line:29", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 29, "code": "\t\t\t(*offset) = co64->offsets[sampleNumber - 1];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional", "targeted"]}
{"review_id": "primevul_paired:valid:209:line:31", "pair_key": "primevul_paired:valid:104", "sample_key": "primevul_paired:valid:209", "line": 31, "code": "\t\treturn GF_OK;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:210
```c
```

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
```
### primevul_paired:valid:216
```c
```

## primevul_paired:valid:114 / addition_only / uncertain
```diff
--- 
+++ 
@@ -12,4 +12,7 @@
     if (inputs_shape.dims() != 3) {
       return errors::InvalidArgument("inputs is not a 3-Tensor");
+    }
+    if (inputs_shape.num_elements() == 0) {
+      return errors::InvalidArgument("inputs must not be empty");
     }
 
```
### primevul_paired:valid:229
```c
    5:     Status status = ctx->input("inputs", inputs);
    6:     if (!status.ok()) return status;
    7:     status = ctx->input("sequence_length", seq_len);
    8:     if (!status.ok()) return status;
    9: 
   10:     const TensorShape& inputs_shape = (*inputs)->shape();
   11: 
   12:     if (inputs_shape.dims() != 3) {
   13:       return errors::InvalidArgument("inputs is not a 3-Tensor");
   14:     }
   15: 
   16:     const int64 max_time = inputs_shape.dim_size(0);
   17:     const int64 batch_size = inputs_shape.dim_size(1);
   18: 
   19:     if (max_time == 0) {
   20:       return errors::InvalidArgument("max_time is 0");
   21:     }
   22:     if (!TensorShapeUtils::IsVector((*seq_len)->shape())) {
   23:       return errors::InvalidArgument("sequence_length is not a vector");
   24:     }
   25: 
   26:     if (!(batch_size == (*seq_len)->dim_size(0))) {
   27:       return errors::FailedPrecondition(
   28:           "len(sequence_length) != batch_size.  ",
   29:           "len(sequence_length):  ", (*seq_len)->dim_size(0),
   33:     auto seq_len_t = (*seq_len)->vec<int32>();
   34: 
   35:     for (int b = 0; b < batch_size; ++b) {
   36:       if (!(seq_len_t(b) <= max_time)) {
   37:         return errors::FailedPrecondition("sequence_length(", b,
   38:                                           ") <= ", max_time);
   39:       }
```
{"review_id": "primevul_paired:valid:229:line:7", "pair_key": "primevul_paired:valid:114", "sample_key": "primevul_paired:valid:229", "line": 7, "code": "    if (!status.ok()) return status;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:229:line:9", "pair_key": "primevul_paired:valid:114", "sample_key": "primevul_paired:valid:229", "line": 9, "code": "    const TensorShape& inputs_shape = (*inputs)->shape();", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:229:line:11", "pair_key": "primevul_paired:valid:114", "sample_key": "primevul_paired:valid:229", "line": 11, "code": "    if (inputs_shape.dims() != 3) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:229:line:12", "pair_key": "primevul_paired:valid:114", "sample_key": "primevul_paired:valid:229", "line": 12, "code": "      return errors::InvalidArgument(\"inputs is not a 3-Tensor\");", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:229:line:15", "pair_key": "primevul_paired:valid:114", "sample_key": "primevul_paired:valid:229", "line": 15, "code": "    const int64 max_time = inputs_shape.dim_size(0);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:229:line:16", "pair_key": "primevul_paired:valid:114", "sample_key": "primevul_paired:valid:229", "line": 16, "code": "    const int64 batch_size = inputs_shape.dim_size(1);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:229:line:18", "pair_key": "primevul_paired:valid:114", "sample_key": "primevul_paired:valid:229", "line": 18, "code": "    if (max_time == 0) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:229:line:19", "pair_key": "primevul_paired:valid:114", "sample_key": "primevul_paired:valid:229", "line": 19, "code": "      return errors::InvalidArgument(\"max_time is 0\");", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:229:line:21", "pair_key": "primevul_paired:valid:114", "sample_key": "primevul_paired:valid:229", "line": 21, "code": "    if (!TensorShapeUtils::IsVector((*seq_len)->shape())) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:229:line:25", "pair_key": "primevul_paired:valid:114", "sample_key": "primevul_paired:valid:229", "line": 25, "code": "    if (!(batch_size == (*seq_len)->dim_size(0))) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:229:line:35", "pair_key": "primevul_paired:valid:114", "sample_key": "primevul_paired:valid:229", "line": 35, "code": "      if (!(seq_len_t(b) <= max_time)) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:230
```c
```

## primevul_paired:valid:118 / addition_only / uncertain
```diff
--- 
+++ 
@@ -10,4 +10,8 @@
   int16_t klinux_family = input->klinux_sa_family;
   if (klinux_family == kLinux_AF_UNIX) {
+    if (input_len < sizeof(struct klinux_sockaddr_un)) {
+      return false;
+    }
+
     struct klinux_sockaddr_un *klinux_sockaddr_un_in =
         const_cast<struct klinux_sockaddr_un *>(
@@ -23,4 +27,7 @@
     CopySockaddr(&sockaddr_un_out, sizeof(sockaddr_un_out), output, output_len);
   } else if (klinux_family == kLinux_AF_INET) {
+    if (input_len < sizeof(struct klinux_sockaddr_in)) {
+      return false;
+    }
     struct klinux_sockaddr_in *klinux_sockaddr_in_in =
         const_cast<struct klinux_sockaddr_in *>(
@@ -38,4 +45,8 @@
     CopySockaddr(&sockaddr_in_out, sizeof(sockaddr_in_out), output, output_len);
   } else if (klinux_family == kLinux_AF_INET6) {
+    if (input_len < sizeof(struct klinux_sockaddr_in6)) {
+      return false;
+    }
+
     struct klinux_sockaddr_in6 *klinux_sockaddr_in6_in =
         const_cast<struct klinux_sockaddr_in6 *>(
```
### primevul_paired:valid:237
```c
    9: 
   10:   int16_t klinux_family = input->klinux_sa_family;
   11:   if (klinux_family == kLinux_AF_UNIX) {
   12:     struct klinux_sockaddr_un *klinux_sockaddr_un_in =
   13:         const_cast<struct klinux_sockaddr_un *>(
   14:             reinterpret_cast<const struct klinux_sockaddr_un *>(input));
   15: 
   16:     struct sockaddr_un sockaddr_un_out;
   17:     sockaddr_un_out.sun_family = AF_UNIX;
   18:     InitializeToZeroArray(sockaddr_un_out.sun_path);
   19:     ReinterpretCopyArray(
   20:         sockaddr_un_out.sun_path, klinux_sockaddr_un_in->klinux_sun_path,
   21:         std::min(sizeof(sockaddr_un_out.sun_path),
   22:                  sizeof(klinux_sockaddr_un_in->klinux_sun_path)));
   23:     CopySockaddr(&sockaddr_un_out, sizeof(sockaddr_un_out), output, output_len);
   24:   } else if (klinux_family == kLinux_AF_INET) {
   25:     struct klinux_sockaddr_in *klinux_sockaddr_in_in =
   26:         const_cast<struct klinux_sockaddr_in *>(
   27:             reinterpret_cast<const struct klinux_sockaddr_in *>(input));
   28: 
   29:     struct sockaddr_in sockaddr_in_out;
   30:     sockaddr_in_out.sin_family = AF_INET;
   31:     sockaddr_in_out.sin_port = klinux_sockaddr_in_in->klinux_sin_port;
   32:     InitializeToZeroSingle(&sockaddr_in_out.sin_addr);
   33:     ReinterpretCopySingle(&sockaddr_in_out.sin_addr,
   34:                           &klinux_sockaddr_in_in->klinux_sin_addr);
   35:     InitializeToZeroArray(sockaddr_in_out.sin_zero);
   36:     ReinterpretCopyArray(sockaddr_in_out.sin_zero,
   37:                          klinux_sockaddr_in_in->klinux_sin_zero);
   38:     CopySockaddr(&sockaddr_in_out, sizeof(sockaddr_in_out), output, output_len);
   65:       abort();
   66:     }
   67:   }
   68:   return true;
   69: }
```
{"review_id": "primevul_paired:valid:237:line:11", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 11, "code": "    struct klinux_sockaddr_un *klinux_sockaddr_un_in =", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:12", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 12, "code": "        const_cast<struct klinux_sockaddr_un *>(", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:13", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 13, "code": "            reinterpret_cast<const struct klinux_sockaddr_un *>(input));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:16", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 16, "code": "    sockaddr_un_out.sun_family = AF_UNIX;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:17", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 17, "code": "    InitializeToZeroArray(sockaddr_un_out.sun_path);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:18", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 18, "code": "    ReinterpretCopyArray(", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:19", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 19, "code": "        sockaddr_un_out.sun_path, klinux_sockaddr_un_in->klinux_sun_path,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:20", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 20, "code": "        std::min(sizeof(sockaddr_un_out.sun_path),", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:21", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 21, "code": "                 sizeof(klinux_sockaddr_un_in->klinux_sun_path)));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:22", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 22, "code": "    CopySockaddr(&sockaddr_un_out, sizeof(sockaddr_un_out), output, output_len);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:24", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 24, "code": "    struct klinux_sockaddr_in *klinux_sockaddr_in_in =", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:25", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 25, "code": "        const_cast<struct klinux_sockaddr_in *>(", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:26", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 26, "code": "            reinterpret_cast<const struct klinux_sockaddr_in *>(input));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:29", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 29, "code": "    sockaddr_in_out.sin_family = AF_INET;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:30", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 30, "code": "    sockaddr_in_out.sin_port = klinux_sockaddr_in_in->klinux_sin_port;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:31", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 31, "code": "    InitializeToZeroSingle(&sockaddr_in_out.sin_addr);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:32", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 32, "code": "    ReinterpretCopySingle(&sockaddr_in_out.sin_addr,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:33", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 33, "code": "                          &klinux_sockaddr_in_in->klinux_sin_addr);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:34", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 34, "code": "    InitializeToZeroArray(sockaddr_in_out.sin_zero);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:237:line:67", "pair_key": "primevul_paired:valid:118", "sample_key": "primevul_paired:valid:237", "line": 67, "code": "  return true;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:238
```c
```

## primevul_paired:valid:135 / addition_only / uncertain
```diff
--- 
+++ 
@@ -1,4 +1,15 @@
 Status GraphConstructor::MakeEdge(Node* src, int output_index, Node* dst,
                                   int input_index) {
+  if (output_index >= src->num_outputs()) {
+    return errors::InvalidArgument(
+        "Output ", output_index, " of node ", src->name(),
+        " does not exist. Node only has ", src->num_outputs(), " outputs.");
+  }
+  if (input_index >= dst->num_inputs()) {
+    return errors::InvalidArgument(
+        "Input ", input_index, " of node ", dst->name(),
+        " does not exist. Node only has ", dst->num_inputs(), " inputs.");
+  }
+
   DataType src_out = src->output_type(output_index);
   DataType dst_in = dst->input_type(input_index);
```
### primevul_paired:valid:271
```c
    1: Status GraphConstructor::MakeEdge(Node* src, int output_index, Node* dst,
    2:                                   int input_index) {
    3:   DataType src_out = src->output_type(output_index);
    4:   DataType dst_in = dst->input_type(input_index);
    5:   if (!TypesCompatible(dst_in, src_out)) {
    6:     return errors::InvalidArgument(
    7:         "Input ", input_index, " of node ", dst->name(), " was passed ",
    8:         DataTypeString(src_out), " from ", src->name(), ":", output_index,
    9:         " incompatible with expected ", DataTypeString(dst_in), ".");
   10:   }
   11:   g_->AddEdge(src, output_index, dst, input_index);
   12:   return Status::OK();
   13: }
```
{"review_id": "primevul_paired:valid:271:line:2", "pair_key": "primevul_paired:valid:135", "sample_key": "primevul_paired:valid:271", "line": 2, "code": "  DataType src_out = src->output_type(output_index);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:271:line:3", "pair_key": "primevul_paired:valid:135", "sample_key": "primevul_paired:valid:271", "line": 3, "code": "  DataType dst_in = dst->input_type(input_index);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:271:line:4", "pair_key": "primevul_paired:valid:135", "sample_key": "primevul_paired:valid:271", "line": 4, "code": "  if (!TypesCompatible(dst_in, src_out)) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:271:line:5", "pair_key": "primevul_paired:valid:135", "sample_key": "primevul_paired:valid:271", "line": 5, "code": "    return errors::InvalidArgument(", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:271:line:6", "pair_key": "primevul_paired:valid:135", "sample_key": "primevul_paired:valid:271", "line": 6, "code": "        \"Input \", input_index, \" of node \", dst->name(), \" was passed \",", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:271:line:7", "pair_key": "primevul_paired:valid:135", "sample_key": "primevul_paired:valid:271", "line": 7, "code": "        DataTypeString(src_out), \" from \", src->name(), \":\", output_index,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:271:line:8", "pair_key": "primevul_paired:valid:135", "sample_key": "primevul_paired:valid:271", "line": 8, "code": "        \" incompatible with expected \", DataTypeString(dst_in), \".\");", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:271:line:10", "pair_key": "primevul_paired:valid:135", "sample_key": "primevul_paired:valid:271", "line": 10, "code": "  g_->AddEdge(src, output_index, dst, input_index);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:271:line:11", "pair_key": "primevul_paired:valid:135", "sample_key": "primevul_paired:valid:271", "line": 11, "code": "  return Status::OK();", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:272
```c
```

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
```
### primevul_paired:valid:352
```c
```

## primevul_paired:valid:178 / addition_only / legal
```diff
--- 
+++ 
@@ -118,4 +118,6 @@
                         if (bb_data_len < DVB_S2_GSE_MINSIZE)
                             bb_data_len = 0;
+                    } else {
+                        bb_data_len = 0;
                     }
                 }
```
### primevul_paired:valid:357
```c
```
### primevul_paired:valid:358
```c
```

## primevul_paired:valid:203 / addition_only / uncertain
```diff
--- 
+++ 
@@ -12,4 +12,9 @@
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
### primevul_paired:valid:407
```c
   11: 	if (!alt) {
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
   36: 	reg_w(sd, R511_CAM_OPTS, 0x03);
```
{"review_id": "primevul_paired:valid:407:line:13", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 13, "code": "\t\treturn;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:407:line:16", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 16, "code": "\tpacket_size = le16_to_cpu(alt->endpoint[0].desc.wMaxPacketSize);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:407:line:17", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 17, "code": "\treg_w(sd, R51x_FIFO_PSIZE, packet_size >> 5);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:407:line:19", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 19, "code": "\treg_w(sd, R511_CAM_UV_EN, 0x01);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:407:line:20", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 20, "code": "\treg_w(sd, R511_SNAP_UV_EN, 0x01);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:407:line:21", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 21, "code": "\treg_w(sd, R511_SNAP_OPTS, 0x03);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:407:line:26", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 26, "code": "\thsegs = (sd->gspca_dev.pixfmt.width >> 3) - 1;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:407:line:27", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 27, "code": "\tvsegs = (sd->gspca_dev.pixfmt.height >> 3) - 1;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:407:line:29", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 29, "code": "\treg_w(sd, R511_CAM_PXCNT, hsegs);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:407:line:30", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 30, "code": "\treg_w(sd, R511_CAM_LNCNT, vsegs);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:407:line:31", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 31, "code": "\treg_w(sd, R511_CAM_PXDIV, 0x00);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:407:line:32", "pair_key": "primevul_paired:valid:203", "sample_key": "primevul_paired:valid:407", "line": 32, "code": "\treg_w(sd, R511_CAM_LNDIV, 0x00);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:408
```c
```

## primevul_paired:valid:211 / addition_only / uncertain
```diff
--- 
+++ 
@@ -73,4 +73,6 @@
 			if (di_size > XFS_DFORK_DSIZE(dip, mp))
 				return __this_address;
+			if (dip->di_nextents)
+				return __this_address;
 			/* fall through */
 		case XFS_DINODE_FMT_EXTENTS:
@@ -91,4 +93,7 @@
 		switch (dip->di_aformat) {
 		case XFS_DINODE_FMT_LOCAL:
+			if (dip->di_anextents)
+				return __this_address;
+		/* fall through */
 		case XFS_DINODE_FMT_EXTENTS:
 		case XFS_DINODE_FMT_BTREE:
@@ -97,4 +102,20 @@
 			return __this_address;
 		}
+	} else {
+		/*
+		 * If there is no fork offset, this may be a freshly-made inode
+		 * in a new disk cluster, in which case di_aformat is zeroed.
+		 * Otherwise, such an inode must be in EXTENTS format; this goes
+		 * for freed inodes as well.
+		 */
+		switch (dip->di_aformat) {
+		case 0:
+		case XFS_DINODE_FMT_EXTENTS:
+			break;
+		default:
+			return __this_address;
+		}
+		if (dip->di_anextents)
+			return __this_address;
 	}
 
```
### primevul_paired:valid:423
```c
```
### primevul_paired:valid:424
```c
```

## primevul_paired:valid:216 / addition_only / uncertain
```diff
--- 
+++ 
@@ -1,4 +1,6 @@
 static x3f_huffnode_t *new_node(x3f_hufftree_t *tree)
 {
+	if (tree->free_node_index >= tree->total_node_index)
+		throw LIBRAW_EXCEPTION_IO_CORRUPT;
   x3f_huffnode_t *t = &tree->nodes[tree->free_node_index];
 
```
### primevul_paired:valid:433
```c
    1: static x3f_huffnode_t *new_node(x3f_hufftree_t *tree)
    2: {
    3:   x3f_huffnode_t *t = &tree->nodes[tree->free_node_index];
    4: 
    5:   t->branch[0] = NULL;
    6:   t->branch[1] = NULL;
    7:   t->leaf = UNDEFINED_LEAF;
    8: 
    9:   tree->free_node_index++;
   10: 
   11:   return t;
   12: }
```
{"review_id": "primevul_paired:valid:433:line:2", "pair_key": "primevul_paired:valid:216", "sample_key": "primevul_paired:valid:433", "line": 2, "code": "  x3f_huffnode_t *t = &tree->nodes[tree->free_node_index];", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional", "targeted"]}
{"review_id": "primevul_paired:valid:433:line:4", "pair_key": "primevul_paired:valid:216", "sample_key": "primevul_paired:valid:433", "line": 4, "code": "  t->branch[0] = NULL;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:433:line:5", "pair_key": "primevul_paired:valid:216", "sample_key": "primevul_paired:valid:433", "line": 5, "code": "  t->branch[1] = NULL;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:433:line:6", "pair_key": "primevul_paired:valid:216", "sample_key": "primevul_paired:valid:433", "line": 6, "code": "  t->leaf = UNDEFINED_LEAF;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:433:line:8", "pair_key": "primevul_paired:valid:216", "sample_key": "primevul_paired:valid:433", "line": 8, "code": "  tree->free_node_index++;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:433:line:10", "pair_key": "primevul_paired:valid:216", "sample_key": "primevul_paired:valid:433", "line": 10, "code": "  return t;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:434
```c
```

## primevul_paired:valid:238 / addition_only / uncertain
```diff
--- 
+++ 
@@ -8,4 +8,8 @@
 
     trace_pci_nvme_map_addr(addr, len);
+
+    if (nvme_addr_is_iomem(n, addr)) {
+        return NVME_DATA_TRAS_ERROR;
+    }
 
     if (nvme_addr_is_cmb(n, addr)) {
```
### primevul_paired:valid:477
```c
    2: {
    3:     bool cmb = false, pmr = false;
    4: 
    5:     if (!len) {
    6:         return NVME_SUCCESS;
    7:     }
    8: 
    9:     trace_pci_nvme_map_addr(addr, len);
   10: 
   11:     if (nvme_addr_is_cmb(n, addr)) {
   12:         cmb = true;
   13:     } else if (nvme_addr_is_pmr(n, addr)) {
   14:         pmr = true;
   15:     }
   16: 
   17:     if (cmb || pmr) {
   18:         if (sg->flags & NVME_SG_DMA) {
   19:             return NVME_INVALID_USE_OF_CMB | NVME_DNR;
   20:         }
   21: 
   24:         }
   25: 
   26:         if (cmb) {
   27:             return nvme_map_addr_cmb(n, &sg->iov, addr, len);
   28:         } else {
   29:             return nvme_map_addr_pmr(n, &sg->iov, addr, len);
   30:         }
   31:     }
   32: 
   33:     if (!(sg->flags & NVME_SG_DMA)) {
   34:         return NVME_INVALID_USE_OF_CMB | NVME_DNR;
   35:     }
   36: 
   38:         goto max_mappings_exceeded;
   39:     }
   40: 
   41:     qemu_sglist_add(&sg->qsg, addr, len);
   42: 
   43:     return NVME_SUCCESS;
   44: 
```
{"review_id": "primevul_paired:valid:477:line:4", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 4, "code": "    if (!len) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:477:line:5", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 5, "code": "        return NVME_SUCCESS;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:477:line:8", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 8, "code": "    trace_pci_nvme_map_addr(addr, len);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:477:line:10", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 10, "code": "    if (nvme_addr_is_cmb(n, addr)) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:477:line:11", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 11, "code": "        cmb = true;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:477:line:12", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 12, "code": "    } else if (nvme_addr_is_pmr(n, addr)) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:477:line:16", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 16, "code": "    if (cmb || pmr) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:477:line:17", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 17, "code": "        if (sg->flags & NVME_SG_DMA) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:477:line:26", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 26, "code": "            return nvme_map_addr_cmb(n, &sg->iov, addr, len);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:477:line:28", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 28, "code": "            return nvme_map_addr_pmr(n, &sg->iov, addr, len);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:477:line:32", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 32, "code": "    if (!(sg->flags & NVME_SG_DMA)) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:477:line:40", "pair_key": "primevul_paired:valid:238", "sample_key": "primevul_paired:valid:477", "line": 40, "code": "    qemu_sglist_add(&sg->qsg, addr, len);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:478
```c
```

## primevul_paired:valid:298 / addition_only / legal
```diff
--- 
+++ 
@@ -29,4 +29,7 @@
         CHECK(avifROStreamReadU32(&s, &grid->outputHeight)); // unsigned int(FieldLength) output_height;
     }
+    if (grid->outputWidth > AVIF_MAX_IMAGE_SIZE / grid->outputHeight) {
+        return AVIF_FALSE;
+    }
     return AVIF_TRUE;
 }
```
### primevul_paired:valid:597
```c
    4: 
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
{"review_id": "primevul_paired:valid:597:line:6", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 6, "code": "    if (version != 0) {", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:597:line:10", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 10, "code": "    CHECK(avifROStreamRead(&s, &grid->rows, 1));    // unsigned int(8) rows_minus_one;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:597:line:11", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 11, "code": "    CHECK(avifROStreamRead(&s, &grid->columns, 1)); // unsigned int(8) columns_minus_one;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:597:line:12", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 12, "code": "    ++grid->rows;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:597:line:13", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 13, "code": "    ++grid->columns;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:597:line:16", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 16, "code": "    if (fieldLength == 16) {", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:597:line:20", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 20, "code": "        grid->outputWidth = outputWidth16;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:597:line:21", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 21, "code": "        grid->outputHeight = outputHeight16;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:597:line:23", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 23, "code": "        if (fieldLength != 32) {", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:597:line:27", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 27, "code": "        CHECK(avifROStreamReadU32(&s, &grid->outputWidth));  // unsigned int(FieldLength) output_width;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:597:line:28", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 28, "code": "        CHECK(avifROStreamReadU32(&s, &grid->outputHeight)); // unsigned int(FieldLength) output_height;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:597:line:30", "pair_key": "primevul_paired:valid:298", "sample_key": "primevul_paired:valid:597", "line": 30, "code": "    return AVIF_TRUE;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:598
```c
```

## primevul_paired:valid:307 / addition_only / uncertain
```diff
--- 
+++ 
@@ -4,4 +4,5 @@
     DriveInfo *di;
     int i;
+    IDEDevice *idedev;
 
     pci_ide = PCI_IDE(dev);
@@ -16,4 +17,10 @@
             }
             pci_ide->bus[di->bus].ifs[di->unit].blk = NULL;
+            if (!(i % 2)) {
+                idedev = pci_ide->bus[di->bus].master;
+            } else {
+                idedev = pci_ide->bus[di->bus].slave;
+            }
+            idedev->conf.blk = NULL;
             blk_unref(blk);
         }
```
### primevul_paired:valid:615
```c
    4:     DriveInfo *di;
    5:     int i;
    6: 
    7:     pci_ide = PCI_IDE(dev);
    8: 
    9:     for (i = 0; i < 4; i++) {
   10:         di = drive_get_by_index(IF_IDE, i);
   11:         if (di != NULL && !di->media_cd) {
   12:             BlockBackend *blk = blk_by_legacy_dinfo(di);
   13:             DeviceState *ds = blk_get_attached_dev(blk);
   14:             if (ds) {
   15:                 blk_detach_dev(blk, ds);
   16:             }
   17:             pci_ide->bus[di->bus].ifs[di->unit].blk = NULL;
   18:             blk_unref(blk);
   19:         }
   20:     }
   21:     qdev_reset_all(DEVICE(dev));
   22:     return 0;
   23: }
```
{"review_id": "primevul_paired:valid:615:line:6", "pair_key": "primevul_paired:valid:307", "sample_key": "primevul_paired:valid:615", "line": 6, "code": "    pci_ide = PCI_IDE(dev);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:615:line:8", "pair_key": "primevul_paired:valid:307", "sample_key": "primevul_paired:valid:615", "line": 8, "code": "    for (i = 0; i < 4; i++) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:615:line:9", "pair_key": "primevul_paired:valid:307", "sample_key": "primevul_paired:valid:615", "line": 9, "code": "        di = drive_get_by_index(IF_IDE, i);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:615:line:10", "pair_key": "primevul_paired:valid:307", "sample_key": "primevul_paired:valid:615", "line": 10, "code": "        if (di != NULL && !di->media_cd) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:615:line:11", "pair_key": "primevul_paired:valid:307", "sample_key": "primevul_paired:valid:615", "line": 11, "code": "            BlockBackend *blk = blk_by_legacy_dinfo(di);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:615:line:12", "pair_key": "primevul_paired:valid:307", "sample_key": "primevul_paired:valid:615", "line": 12, "code": "            DeviceState *ds = blk_get_attached_dev(blk);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:615:line:13", "pair_key": "primevul_paired:valid:307", "sample_key": "primevul_paired:valid:615", "line": 13, "code": "            if (ds) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:615:line:14", "pair_key": "primevul_paired:valid:307", "sample_key": "primevul_paired:valid:615", "line": 14, "code": "                blk_detach_dev(blk, ds);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:615:line:16", "pair_key": "primevul_paired:valid:307", "sample_key": "primevul_paired:valid:615", "line": 16, "code": "            pci_ide->bus[di->bus].ifs[di->unit].blk = NULL;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:615:line:17", "pair_key": "primevul_paired:valid:307", "sample_key": "primevul_paired:valid:615", "line": 17, "code": "            blk_unref(blk);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:615:line:20", "pair_key": "primevul_paired:valid:307", "sample_key": "primevul_paired:valid:615", "line": 20, "code": "    qdev_reset_all(DEVICE(dev));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:616
```c
```

## primevul_paired:valid:31 / addition_only / legal
```diff
--- 
+++ 
@@ -10,3 +10,4 @@
     iTbl++;
   }
+  aAscii[0] = 0;                  /* 0x00 is never a token character */
 }
```
### primevul_paired:valid:63
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
{"review_id": "primevul_paired:valid:63:line:1", "pair_key": "primevul_paired:valid:31", "sample_key": "primevul_paired:valid:63", "line": 1, "code": "  int i = 0;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:63:line:2", "pair_key": "primevul_paired:valid:31", "sample_key": "primevul_paired:valid:63", "line": 2, "code": "  int iTbl = 0;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:63:line:3", "pair_key": "primevul_paired:valid:31", "sample_key": "primevul_paired:valid:63", "line": 3, "code": "  while( i<128 ){", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:63:line:4", "pair_key": "primevul_paired:valid:31", "sample_key": "primevul_paired:valid:63", "line": 4, "code": "    int bToken = aArray[ aFts5UnicodeData[iTbl] & 0x1F ];", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:63:line:5", "pair_key": "primevul_paired:valid:31", "sample_key": "primevul_paired:valid:63", "line": 5, "code": "    int n = (aFts5UnicodeData[iTbl] >> 5) + i;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:63:line:6", "pair_key": "primevul_paired:valid:31", "sample_key": "primevul_paired:valid:63", "line": 6, "code": "    for(; i<128 && i<n; i++){", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:63:line:7", "pair_key": "primevul_paired:valid:31", "sample_key": "primevul_paired:valid:63", "line": 7, "code": "      aAscii[i] = (u8)bToken;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:63:line:9", "pair_key": "primevul_paired:valid:31", "sample_key": "primevul_paired:valid:63", "line": 9, "code": "    iTbl++;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:64
```c
```

## primevul_paired:valid:315 / addition_only / uncertain
```diff
--- 
+++ 
@@ -25,4 +25,6 @@
   width = LM_to_uint(hdrbuf, 0);
   height = LM_to_uint(hdrbuf, 2);
+  if (width == 0 || height == 0)
+    ERREXIT(cinfo, JERR_GIF_EMPTY);
   /* we ignore the color resolution, sort flag, and background color index */
   aspectRatio = UCH(hdrbuf[6]);
@@ -67,4 +69,6 @@
     width = LM_to_uint(hdrbuf, 4);
     height = LM_to_uint(hdrbuf, 6);
+    if (width == 0 || height == 0)
+      ERREXIT(cinfo, JERR_GIF_EMPTY);
     source->is_interlaced = (BitSet(hdrbuf[8], INTERLACE) != 0);
 
```
### primevul_paired:valid:631
```c
```
### primevul_paired:valid:632
```c
```

## primevul_paired:valid:333 / addition_only / legal
```diff
--- 
+++ 
@@ -507,4 +507,6 @@
   }
   if (logging) (void)LogMagickEvent(CoderEvent,GetMagickModule(),"return");
+  if ((image != image2) && (image2 != (Image *) NULL))
+    image2=DestroyImage(image2);
   if(image==NULL)
     ThrowReaderException(CorruptImageError,"ImproperImageHeader");
```
### primevul_paired:valid:667
```c
```
### primevul_paired:valid:668
```c
```

## primevul_paired:valid:340 / addition_only / uncertain
```diff
--- 
+++ 
@@ -12,4 +12,11 @@
 	size_t num_samples;
 	size_t num_samples_delta;
+
+	size_t total_samples;
+	if (!jas_safe_size_mul(siz->width, siz->height, &total_samples) ||
+	    (dec->max_samples > 0 && total_samples > dec->max_samples)) {
+		jas_eprintf("image too large\n");
+		return -1;
+	}
 
 	size_t tile_samples;
```
### primevul_paired:valid:681
```c
   13: 	size_t num_samples_delta;
   14: 
   15: 	size_t tile_samples;
   16: 	if (!jas_safe_size_mul(siz->tilewidth, siz->tileheight, &tile_samples) ||
   17: 	    (dec->max_samples > 0 && tile_samples > dec->max_samples)) {
   18: 		jas_eprintf("tile too large\n");
   19: 		return -1;
   20: 	}
   21: 
   22: 	dec->xstart = siz->xoff;
   23: 	dec->ystart = siz->yoff;
   24: 	dec->xend = siz->width;
   25: 	dec->yend = siz->height;
   26: 	dec->tilewidth = siz->tilewidth;
   27: 	dec->tileheight = siz->tileheight;
   28: 	dec->tilexoff = siz->tilexoff;
   29: 	dec->tileyoff = siz->tileyoff;
   30: 	dec->numcomps = siz->numcomps;
   31: 
   32: 	if (!(dec->cp = jpc_dec_cp_create(dec->numcomps))) {
   33: 		return -1;
   34: 	}
   35: 
   36: 	if (!(dec->cmpts = jas_alloc2(dec->numcomps, sizeof(jpc_dec_cmpt_t)))) {
   37: 		return -1;
   38: 	}
   39: 
```
{"review_id": "primevul_paired:valid:681:line:15", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 15, "code": "\tif (!jas_safe_size_mul(siz->tilewidth, siz->tileheight, &tile_samples) ||", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:16", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 16, "code": "\t    (dec->max_samples > 0 && tile_samples > dec->max_samples)) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:21", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 21, "code": "\tdec->xstart = siz->xoff;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:22", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 22, "code": "\tdec->ystart = siz->yoff;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:23", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 23, "code": "\tdec->xend = siz->width;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:24", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 24, "code": "\tdec->yend = siz->height;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:25", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 25, "code": "\tdec->tilewidth = siz->tilewidth;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:26", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 26, "code": "\tdec->tileheight = siz->tileheight;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:27", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 27, "code": "\tdec->tilexoff = siz->tilexoff;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:28", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 28, "code": "\tdec->tileyoff = siz->tileyoff;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:29", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 29, "code": "\tdec->numcomps = siz->numcomps;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:31", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 31, "code": "\tif (!(dec->cp = jpc_dec_cp_create(dec->numcomps))) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:681:line:35", "pair_key": "primevul_paired:valid:340", "sample_key": "primevul_paired:valid:681", "line": 35, "code": "\tif (!(dec->cmpts = jas_alloc2(dec->numcomps, sizeof(jpc_dec_cmpt_t)))) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:682
```c
```

## primevul_paired:valid:353 / addition_only / uncertain
```diff
--- 
+++ 
@@ -17,4 +17,6 @@
         check_dict_read(*op);
         if ( dict_find_string(op, ".jbig2globalctx", &sop) > 0) {
+            if (!r_is_struct(sop) || !r_has_stype(sop, imemory, st_jbig2_global_data_t))
+                return_error(gs_error_typecheck);
             gref = r_ptr(sop, s_jbig2_global_data_t);
             s_jbig2decode_set_global_data((stream_state*)&state, gref);
```
### primevul_paired:valid:707
```c
```
### primevul_paired:valid:708
```c
```

## primevul_paired:valid:371 / addition_only / uncertain
```diff
--- 
+++ 
@@ -5,4 +5,5 @@
 	ExifShort c;
 	size_t i, tcount, o, datao;
+	long failsafe_size = 0;
 
 	if (!n || !buf || !buf_size) {
@@ -96,4 +97,21 @@
 		}
 
+		/* Track the size of decoded tag data. A malicious file could
+		 * be crafted to cause extremely large values here without
+		 * tripping any buffer range checks.  This is especially bad
+		 * with the libexif representation of Canon MakerNotes because
+		 * some arrays are turned into individual tags that the
+		 * application must loop around. */
+		failsafe_size += mnote_canon_entry_count_values(&n->entries[tcount]);
+
+		if (failsafe_size > FAILSAFE_SIZE_MAX) {
+			/* Abort if the total size of the data in the tags extraordinarily large, */
+			exif_mem_free (ne->mem, n->entries[tcount].data);
+			exif_log (ne->log, EXIF_LOG_CODE_CORRUPT_DATA,
+					  "ExifMnoteCanon", "Failsafe tag size overflow (%lu > %ld)",
+					  failsafe_size, FAILSAFE_SIZE_MAX);
+			break;
+		}
+
 		/* Tag was successfully parsed */
 		++tcount;
```
### primevul_paired:valid:743
```c
```
### primevul_paired:valid:744
```c
```

## primevul_paired:valid:377 / addition_only / uncertain
```diff
--- 
+++ 
@@ -3,3 +3,5 @@
 	if (sk->sk_send_head == skb_unlinked)
 		sk->sk_send_head = NULL;
+	if (tcp_sk(sk)->highest_sack == skb_unlinked)
+		tcp_sk(sk)->highest_sack = NULL;
 }
```
### primevul_paired:valid:755
```c
    1: static inline void tcp_check_send_head(struct sock *sk, struct sk_buff *skb_unlinked)
    2: {
    3: 	if (sk->sk_send_head == skb_unlinked)
    4: 		sk->sk_send_head = NULL;
    5: }
```
{"review_id": "primevul_paired:valid:755:line:2", "pair_key": "primevul_paired:valid:377", "sample_key": "primevul_paired:valid:755", "line": 2, "code": "\tif (sk->sk_send_head == skb_unlinked)", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:755:line:3", "pair_key": "primevul_paired:valid:377", "sample_key": "primevul_paired:valid:755", "line": 3, "code": "\t\tsk->sk_send_head = NULL;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:756
```c
```

## primevul_paired:valid:40 / addition_only / uncertain
```diff
--- 
+++ 
@@ -4,4 +4,11 @@
   PyObject* value;
   Py_ssize_t pos = 0;
+
+  // Return an error if the fields attribute is null.
+  // Occurs with an improper conversion attempt to resource.
+  if (descr->fields == nullptr) {
+    return errors::Internal("Unexpected numpy data type");
+  }
+
   if (PyDict_Next(descr->fields, &pos, &key, &value)) {
     // In Python 3, the keys of numpy custom struct types are unicode, unlike
```
### primevul_paired:valid:81
```c
    3:   PyObject* key;
    4:   PyObject* value;
    5:   Py_ssize_t pos = 0;
    6:   if (PyDict_Next(descr->fields, &pos, &key, &value)) {
    7:     // In Python 3, the keys of numpy custom struct types are unicode, unlike
    8:     // Python 2, where the keys are bytes.
    9:     const char* key_string =
   10:         PyBytes_Check(key) ? PyBytes_AsString(key)
   11:                            : PyBytes_AsString(PyUnicode_AsASCIIString(key));
   12:     if (!key_string) {
   13:       return errors::Internal("Corrupt numpy type descriptor");
   14:     }
   15:     tensorflow::string key = key_string;
   16:     // The typenames here should match the field names in the custom struct
   17:     // types constructed in test_util.py.
   18:     // TODO(mrry,keveman): Investigate Numpy type registration to replace this
   19:     // hard-coding of names.
   20:     if (key == "quint8") {
   21:       *out_tf_datatype = TF_QUINT8;
   22:     } else if (key == "qint8") {
   23:       *out_tf_datatype = TF_QINT8;
   24:     } else if (key == "qint16") {
   25:       *out_tf_datatype = TF_QINT16;
   26:     } else if (key == "quint16") {
   27:       *out_tf_datatype = TF_QUINT16;
   28:     } else if (key == "qint32") {
   29:       *out_tf_datatype = TF_QINT32;
   30:     } else if (key == "resource") {
   31:       *out_tf_datatype = TF_RESOURCE;
   32:     } else {
   33:       return errors::Internal("Unsupported numpy data type");
   34:     }
   35:     return Status::OK();
   36:   }
   37:   return errors::Internal("Unsupported numpy data type");
   38: }
```
{"review_id": "primevul_paired:valid:81:line:5", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 5, "code": "  if (PyDict_Next(descr->fields, &pos, &key, &value)) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:8", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 8, "code": "    const char* key_string =", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:9", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 9, "code": "        PyBytes_Check(key) ? PyBytes_AsString(key)", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:10", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 10, "code": "                           : PyBytes_AsString(PyUnicode_AsASCIIString(key));", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:11", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 11, "code": "    if (!key_string) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:12", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 12, "code": "      return errors::Internal(\"Corrupt numpy type descriptor\");", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:14", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 14, "code": "    tensorflow::string key = key_string;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:19", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 19, "code": "    if (key == \"quint8\") {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:21", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 21, "code": "    } else if (key == \"qint8\") {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:23", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 23, "code": "    } else if (key == \"qint16\") {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:25", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 25, "code": "    } else if (key == \"quint16\") {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:27", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 27, "code": "    } else if (key == \"qint32\") {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:29", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 29, "code": "    } else if (key == \"resource\") {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:81:line:36", "pair_key": "primevul_paired:valid:40", "sample_key": "primevul_paired:valid:81", "line": 36, "code": "  return errors::Internal(\"Unsupported numpy data type\");", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:82
```c
```

## primevul_paired:valid:419 / addition_only / uncertain
```diff
--- 
+++ 
@@ -8,4 +8,6 @@
 
 	WLog_VRB(AUTODETECT_TAG, "received Bandwidth Measure Results PDU");
+	if (Stream_GetRemainingLength(s) < 8)
+		return -1;
 	Stream_Read_UINT32(s, rdp->autodetect->bandwidthMeasureTimeDelta); /* timeDelta (4 bytes) */
 	Stream_Read_UINT32(s, rdp->autodetect->bandwidthMeasureByteCount); /* byteCount (4 bytes) */
```
### primevul_paired:valid:839
```c
    1: static BOOL autodetect_recv_bandwidth_measure_results(rdpRdp* rdp, wStream* s,
    2:                                                       AUTODETECT_RSP_PDU* autodetectRspPdu)
    3: {
    4: 	BOOL success = TRUE;
    5: 
    6: 	if (autodetectRspPdu->headerLength != 0x0E)
    7: 		return FALSE;
    8: 
    9: 	WLog_VRB(AUTODETECT_TAG, "received Bandwidth Measure Results PDU");
   10: 	Stream_Read_UINT32(s, rdp->autodetect->bandwidthMeasureTimeDelta); /* timeDelta (4 bytes) */
   11: 	Stream_Read_UINT32(s, rdp->autodetect->bandwidthMeasureByteCount); /* byteCount (4 bytes) */
   12: 
   13: 	if (rdp->autodetect->bandwidthMeasureTimeDelta > 0)
   14: 		rdp->autodetect->netCharBandwidth = rdp->autodetect->bandwidthMeasureByteCount * 8 /
   15: 		                                    rdp->autodetect->bandwidthMeasureTimeDelta;
   16: 	else
   17: 		rdp->autodetect->netCharBandwidth = 0;
   18: 
   19: 	IFCALLRET(rdp->autodetect->BandwidthMeasureResults, success, rdp->context,
   20: 	          autodetectRspPdu->sequenceNumber);
   21: 	return success;
   22: }
```
{"review_id": "primevul_paired:valid:839:line:3", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 3, "code": "\tBOOL success = TRUE;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:5", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 5, "code": "\tif (autodetectRspPdu->headerLength != 0x0E)", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:6", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 6, "code": "\t\treturn FALSE;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:8", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 8, "code": "\tWLog_VRB(AUTODETECT_TAG, \"received Bandwidth Measure Results PDU\");", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:9", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 9, "code": "\tStream_Read_UINT32(s, rdp->autodetect->bandwidthMeasureTimeDelta); /* timeDelta (4 bytes) */", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:10", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 10, "code": "\tStream_Read_UINT32(s, rdp->autodetect->bandwidthMeasureByteCount); /* byteCount (4 bytes) */", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:12", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 12, "code": "\tif (rdp->autodetect->bandwidthMeasureTimeDelta > 0)", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:13", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 13, "code": "\t\trdp->autodetect->netCharBandwidth = rdp->autodetect->bandwidthMeasureByteCount * 8 /", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:14", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 14, "code": "\t\t                                    rdp->autodetect->bandwidthMeasureTimeDelta;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:16", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 16, "code": "\t\trdp->autodetect->netCharBandwidth = 0;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:18", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 18, "code": "\tIFCALLRET(rdp->autodetect->BandwidthMeasureResults, success, rdp->context,", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:19", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 19, "code": "\t          autodetectRspPdu->sequenceNumber);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:839:line:20", "pair_key": "primevul_paired:valid:419", "sample_key": "primevul_paired:valid:839", "line": 20, "code": "\treturn success;", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:840
```c
```

## primevul_paired:valid:53 / addition_only / uncertain
```diff
--- 
+++ 
@@ -15,4 +15,16 @@
     return errors::InvalidArgument("boxes must have 4 columns");
   }
+  for (int64 i = 0; i < *num_boxes; i++) {
+    for (int64 j = 0; j < 4; j++) {
+      if (!isfinite(boxes.tensor<float, 2>()(i, j))) {
+        return errors::InvalidArgument(
+            "boxes values must be finite, received boxes[", i, "]: ",
+            boxes.tensor<float, 2>()(i, 0), ", ",
+            boxes.tensor<float, 2>()(i, 1), ", ",
+            boxes.tensor<float, 2>()(i, 2), ", ",
+            boxes.tensor<float, 2>()(i, 3));
+      }
+    }
+  }
   // The shape of 'box_index' is [num_boxes].
   if (box_index.dims() != 1) {
```
### primevul_paired:valid:107
```c
    1: static inline Status ParseAndCheckBoxSizes(const Tensor& boxes,
    2:                                            const Tensor& box_index,
    3:                                            int* num_boxes) {
    4:   if (boxes.NumElements() == 0 && box_index.NumElements() == 0) {
    5:     *num_boxes = 0;
    6:     return Status::OK();
    7:   }
    8:   // The shape of 'boxes' is [num_boxes, 4].
    9:   if (boxes.dims() != 2) {
   10:     return errors::InvalidArgument("boxes must be 2-D",
   11:                                    boxes.shape().DebugString());
   12:   }
   13:   *num_boxes = boxes.dim_size(0);
   14:   if (boxes.dim_size(1) != 4) {
   15:     return errors::InvalidArgument("boxes must have 4 columns");
   16:   }
   17:   // The shape of 'box_index' is [num_boxes].
   18:   if (box_index.dims() != 1) {
   19:     return errors::InvalidArgument("box_index must be 1-D",
   20:                                    box_index.shape().DebugString());
   21:   }
   22:   if (box_index.dim_size(0) != *num_boxes) {
   23:     return errors::InvalidArgument("box_index has incompatible shape");
   24:   }
   25:   return Status::OK();
   26: }
```
{"review_id": "primevul_paired:valid:107:line:3", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 3, "code": "  if (boxes.NumElements() == 0 && box_index.NumElements() == 0) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:8", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 8, "code": "  if (boxes.dims() != 2) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:9", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 9, "code": "    return errors::InvalidArgument(\"boxes must be 2-D\",", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:10", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 10, "code": "                                   boxes.shape().DebugString());", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:12", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 12, "code": "  *num_boxes = boxes.dim_size(0);", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:13", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 13, "code": "  if (boxes.dim_size(1) != 4) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:14", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 14, "code": "    return errors::InvalidArgument(\"boxes must have 4 columns\");", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:17", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 17, "code": "  if (box_index.dims() != 1) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:18", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 18, "code": "    return errors::InvalidArgument(\"box_index must be 1-D\",", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:19", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 19, "code": "                                   box_index.shape().DebugString());", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:21", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 21, "code": "  if (box_index.dim_size(0) != *num_boxes) {", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:22", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 22, "code": "    return errors::InvalidArgument(\"box_index has incompatible shape\");", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:107:line:24", "pair_key": "primevul_paired:valid:53", "sample_key": "primevul_paired:valid:107", "line": 24, "code": "  return Status::OK();", "cohort": "seed42_check_sample", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:108
```c
```

## primevul_paired:valid:54 / addition_only / legal
```diff
--- 
+++ 
@@ -84,4 +84,5 @@
     if (mp_buf != tmp_out_buf) {
         flb_free(mp_buf);
+        mp_buf = NULL;
     }
 
@@ -114,4 +115,5 @@
         if (k->via.str.ptr == NULL) {
             flb_free(mp_buf);
+            flb_free(tmp_out_buf);
             *out_buf = NULL;
             msgpack_unpacked_destroy(&result);
```
### primevul_paired:valid:109
```c
   30:     struct flb_time *t;
   31: 
   32:     /* Convert incoming in_buf JSON message to message pack format */
   33:     ret = flb_pack_json_recs(in_buf, in_size, &mp_buf, &mp_size, &root_type,
   34:                              &records);
   35:     if (ret != 0) {
   36:         return -1;
   37:     }
   45:     msgpack_unpacked_init(&result);
   46:     if (msgpack_unpack_next(&result, mp_buf, mp_size, &off) == MSGPACK_UNPACK_SUCCESS) {
   47:         map = result.data;
   48:         if (map.type != MSGPACK_OBJECT_MAP) {
   49:             flb_free(mp_buf);
   50:             msgpack_unpacked_destroy(&result);
   51:             return -1;
   60:     }
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
   73:             off = 0;
   79:     }
   80: 
   81:     /* Set the possible outgoing buffer */
   82:     *out_buf = tmp_out_buf;
   83:     *out_size = tmp_out_size;
   84:     if (mp_buf != tmp_out_buf) {
   85:         flb_free(mp_buf);
   86:     }
   87: 
  103:     map_size = map.via.map.size;
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
  118:             return -1;
```
{"review_id": "primevul_paired:valid:109:line:32", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 32, "code": "    ret = flb_pack_json_recs(in_buf, in_size, &mp_buf, &mp_size, &root_type,", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:33", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 33, "code": "                             &records);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:47", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 47, "code": "        if (map.type != MSGPACK_OBJECT_MAP) {", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:62", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 62, "code": "    tmp_out_buf = mp_buf;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:63", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 63, "code": "    tmp_out_size = mp_size;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:66", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 66, "code": "    if (parser->decoders) {", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:67", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 67, "code": "        ret = flb_parser_decoder_do(parser->decoders,", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:68", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 68, "code": "                                    mp_buf, mp_size,", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:69", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 69, "code": "                                    &tmp_out_buf, &tmp_out_size);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:81", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 81, "code": "    *out_buf = tmp_out_buf;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:83", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 83, "code": "    if (mp_buf != tmp_out_buf) {", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:105", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 105, "code": "        k = &map.via.map.ptr[i].key;", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:108", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 108, "code": "        if (k->via.str.size != slen) {", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:113", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 113, "code": "        if (k->via.str.ptr == NULL) {", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
{"review_id": "primevul_paired:valid:109:line:114", "pair_key": "primevul_paired:valid:54", "sample_key": "primevul_paired:valid:109", "line": 114, "code": "            flb_free(mp_buf);", "cohort": "historical_discussed", "methods": ["historical", "mapped_bidirectional"]}
### primevul_paired:valid:110
```c
```