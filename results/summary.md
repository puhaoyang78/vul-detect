# Demo 结果摘要

- 真实 C/C++ 内存安全 CVE：60 个
- Baseline 检出：12/60
- Proposed 检出：21/60
- Proposed 纠正 Baseline 漏检：5 个
- 静态验证拒绝的语义摘要：676 条
- Z3 状态：POTENTIAL_VIOLATION=9, SAFE=3, UNKNOWN=36

结论：出现明确但有限的正向信号。自动恢复并验证的项目语义使检出数从 12 增加到 21，纠正了 5 个漏检。
该样本集全部为漏洞样本，并且优先选择了自定义内存函数，因此这里只能说明召回方向值得继续，不能据此判断误报率或泛化效果。

## 主要失败原因

- 18 个：Z3 could not decide: no supported memory access was available for bounds analysis
- 3 个：Z3 proved the generated bounds conditions under the available constraints
- 1 个：Z3 could not decide: 2 memory access(es) remain unresolved; first at line 98: capacity/valid extent is unknown for c->in_s->data
- 1 个：Z3 could not decide: 1 memory access(es) remain unresolved; first at line 356: capacity/valid extent is unknown for xcfdata
- 1 个：Z3 could not decide: 2 memory access(es) remain unresolved; first at line 92: capacity/valid extent is unknown for msg->stun_hdr.tran_id
- 1 个：Z3 could not decide: 3 memory access(es) remain unresolved; first at line 2390: cannot encode access extent sizeof(*(&(zip->si)))
- 1 个：Z3 could not decide: 24 memory access(es) remain unresolved; first at line 315: capacity/valid extent is unknown for pass_salt
- 1 个：Z3 could not decide: 1 memory access(es) remain unresolved; first at line 290: capacity/valid extent is unknown for buf
- 1 个：Z3 could not decide: 3 memory access(es) remain unresolved; first at line 1593: access extent is not represented as a bounded integer expression
- 1 个：Z3 could not decide: 12 memory access(es) remain unresolved; first at line 265: capacity/valid extent is unknown for pkt_info
- 1 个：Z3 could not decide: 1 memory access(es) remain unresolved; first at line 671: access extent is not represented as a bounded integer expression
- 1 个：Z3 could not decide: 3 memory access(es) remain unresolved; first at line 350: cannot encode access extent "corruptGIF(reason:noclearcode)."
- 1 个：Z3 could not decide: 12 memory access(es) remain unresolved; first at line 840: capacity/valid extent is unknown for &sig8
- 1 个：Z3 could not decide: 1 memory access(es) remain unresolved; first at line 2016: capacity/valid extent is unknown for &cap
- 1 个：Z3 could not decide: 2 memory access(es) remain unresolved; first at line 2467: capacity/valid extent is unknown for (addr).ip
- 1 个：Z3 could not decide: 12 memory access(es) remain unresolved; first at line 373: cannot encode access extent "--datadir"
- 1 个：Z3 could not decide: 1 memory access(es) remain unresolved; first at line 6637: access extent is not represented as a bounded integer expression
- 1 个：Z3 could not decide: 10 memory access(es) remain unresolved; first at line 198: capacity/valid extent is unknown for &eld->b[eld->b_len]
- 1 个：Z3 could not decide: 2 memory access(es) remain unresolved; first at line 3254: capacity/valid extent is unknown for buf
- 1 个：Z3 could not decide: 1 memory access(es) remain unresolved; first at line 365: capacity/valid extent is unknown for buf
- BSON 样本的模型摘要未严格归一化或未通过同一写入点的数据流验证。
- ImageMagick 样本需要关联分配大小与后续循环读取范围，新增 READ/VALUE 后需重新评估。
- Sofia SIP 样本依赖剩余输入长度与越界读取关系，新增 READ 语义后需重新评估。
- FreeType 样本依赖宏、回调和整数偏移，直接调用候选筛选未恢复到写入语义。
