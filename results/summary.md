# Demo 结果摘要

- 函数级样本：60（VULNERABLE=35, BENIGN=25）
- LineVul Baseline：TP=1, FP=3, TN=22, FN=34, Precision=0.2500, Recall=0.0286, F1=0.0513, Accuracy=0.3833
- Proposed：TP=13, FP=8, TN=17, FN=22, Precision=0.6190, Recall=0.3714, F1=0.4643, Accuracy=0.5000
- 静态验证拒绝的语义摘要：676 条
- Z3 状态：POTENTIAL_VIOLATION=21, UNKNOWN=39

## 主要未解析原因

- 9 个：verification incomplete: no supported memory access was available for bounds analysis
- 3 个：verification incomplete: all currently modeled memory accesses satisfy their generated bounds conditions, but complete function-level memory-access coverage is not established
- 1 个：verification incomplete: 2 memory access(es) remain unresolved; first at line 98: capacity/valid extent is unknown for c->in_s->data
- 1 个：verification incomplete: 1 memory access(es) remain unresolved; first at line 356: capacity/valid extent is unknown for xcfdata
- 1 个：verification incomplete: 2 memory access(es) remain unresolved; first at line 244: capacity is unknown for indexed/pointer-offset access ie->ie_buffer+le16_to_cpu(ie->ie_length)
- 1 个：verification incomplete: 3 memory access(es) remain unresolved; first at line 92: capacity/valid extent is unknown for msg->stun_hdr.tran_id
- 1 个：verification incomplete: 3 memory access(es) remain unresolved; first at line 119: access extent is not represented as a bounded integer expression
- 1 个：verification incomplete: 2 memory access(es) remain unresolved; first at line 1223: access extent is not represented as a bounded integer expression
- 1 个：verification incomplete: 25 memory access(es) remain unresolved; first at line 315: capacity/valid extent is unknown for pass_salt
- 1 个：verification incomplete: 2 memory access(es) remain unresolved; first at line 290: capacity/valid extent is unknown for buf
- 1 个：verification incomplete: 3 memory access(es) remain unresolved; first at line 1593: access extent is not represented as a bounded integer expression
- 1 个：verification incomplete: 3 memory access(es) remain unresolved; first at line 444: access extent is not represented as a bounded integer expression
- 1 个：verification incomplete: 18 memory access(es) remain unresolved; first at line 265: capacity/valid extent is unknown for pkt_info
- 1 个：verification incomplete: 1 memory access(es) remain unresolved; first at line 306: capacity is unknown for indexed/pointer-offset access bucket->elem+(bucket->num++)
- 1 个：verification incomplete: 3 memory access(es) remain unresolved; first at line 2001: capacity/valid extent is unknown for blob->ei+(0)
- 1 个：verification incomplete: 8 memory access(es) remain unresolved; first at line 310: capacity is unknown for indexed/pointer-offset access g->codes+(code)
- 1 个：verification incomplete: 13 memory access(es) remain unresolved; first at line 840: capacity/valid extent is unknown for &sig8
- 1 个：verification incomplete: 7 memory access(es) remain unresolved; first at line 2016: capacity/valid extent is unknown for &cap
- 1 个：verification incomplete: 1 memory access(es) remain unresolved; first at line 101: capacity/valid extent is unknown for sb->buf+(0)
- 1 个：verification incomplete: 5 memory access(es) remain unresolved; first at line 306: capacity/valid extent is unknown for pkt->data
- 1 个：verification incomplete: 17 memory access(es) remain unresolved; first at line 2495: capacity is unknown for indexed/pointer-offset access &addr->ip6[dc+(14-n)]
- 1 个：verification incomplete: 12 memory access(es) remain unresolved; first at line 373: cannot encode access extent "--datadir"
- 1 个：verification incomplete: 16 memory access(es) remain unresolved; first at line 198: capacity/valid extent is unknown for &eld->b[eld->b_len]
- 1 个：verification incomplete: 2 memory access(es) remain unresolved; first at line 3254: capacity/valid extent is unknown for buf
- 1 个：verification incomplete: 5 memory access(es) remain unresolved; first at line 792: capacity/valid extent is unknown for data
- 1 个：verification incomplete: 5 memory access(es) remain unresolved; first at line 169: access extent is not represented as a bounded integer expression
- 1 个：verification incomplete: 2 memory access(es) remain unresolved; first at line 646: capacity/valid extent is unknown for map
- 1 个：verification incomplete: 4 memory access(es) remain unresolved; first at line 2619: capacity is unknown for indexed/pointer-offset access msg->attr+(i)
- 1 个：verification incomplete: 1 memory access(es) remain unresolved; first at line 365: capacity/valid extent is unknown for buf
