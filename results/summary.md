# Demo 结果摘要

- 真实 C/C++ 内存安全 CVE：10 个
- Baseline 检出：3/10
- Proposed 检出：6/10
- Proposed 纠正 Baseline 漏检：3 个
- 静态验证拒绝的语义摘要：38 条

结论：出现明确但有限的正向信号。自动恢复并验证的项目语义使检出数从 3 增加到 6，纠正了 3 个漏检。
该样本集全部为漏洞样本，并且优先选择了自定义内存函数，因此这里只能说明召回方向值得继续，不能据此判断误报率或泛化效果。

## 主要失败原因

- 1 个：recovered 4 validated custom semantic operation(s), but the supported checks did not establish an unsafe memory relation
- 1 个：recovered 6 validated custom semantic operation(s), but the supported checks did not establish an unsafe memory relation
- 1 个：recovered 1 validated custom semantic operation(s), but the supported checks did not establish an unsafe memory relation
- 1 个：no validated custom memory semantic reached a supported safety check
- BSON 样本的模型摘要未严格归一化或未通过同一写入点的数据流验证。
- ImageMagick 样本需要关联分配大小与后续循环读取范围，新增 READ/VALUE 后需重新评估。
- Sofia SIP 样本依赖剩余输入长度与越界读取关系，新增 READ 语义后需重新评估。
- FreeType 样本依赖宏、回调和整数偏移，直接调用候选筛选未恢复到写入语义。
