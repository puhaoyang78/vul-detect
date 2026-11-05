# RGCL：ReposVul 数据质量检查

当前版本未通过单函数漏洞二分类的数据质量门槛。本轮完成真实数据检查、相关测试与冻结模型 GPU smoke，按任务第 7 条停止进入完整训练阶段。没有修改原始标签、生成训练子集或重建修复版本。CLS、Context-only、RGCL × Seed42/1234/123456 的三种子完整训练均未运行；Valid/Test 的 MCC、F1、AUC、PR-AUC、Recall、FPR 及均值/标准差均为未测量，不能据此判断 RGCL 的性能增益。

正式方法统一命名为 **RGCL（Repair-Guided Contextual Learning，即原 Context + Diff）**。模型、训练和评价实现保持不变：UniXcoder、512 Token、K=5、Token 加权区域损失、λ=0.1；原 PrimeVul、SecVulEval 结果和 checkpoint 保留。

## 数据来源与原始标签

本地 `/home/PublicData/PHY-data/vul_detect/data/reposvul` 只有 `ReposVul_c.jsonl`（1,498,622,806 字节），共 4,010 条 CVE/提交记录。筛选实际文件语言后得到 7,737 个 C/C++ 文件记录；嵌套函数记录只有 C 和 .h，C++ 文件没有函数数组。计数包含原生修复前/后记录与重复源码，不能解释为独立函数数量。

| 本地函数来源 | 原始标签 1 | 原始标签 0 | 总数 |
|---|---:|---:|---:|
| function_before | 1,270 | 89,423 | 90,693 |
| function_after | 0 | 91,001 | 91,001 |
| 合计 | 1,270 | 180,424 | 181,694 |

按函数记录的文件语言计：C 181,236、.h 458、C++ 0。标签严格来自函数的 `target`，不使用文件级 target 重建标签，也没有从修复源码额外合成负例。原始负正比为 142.07:1。能够对应原正例的 after/0 记录为 1,146，其中 262 个函数体完全未变；其他 0 标签记录没有被自动当作已确认修复函数。

作者在[官方仓库](https://github.com/Eshe0922/ReposVul)提供了[C/C++ 官方发布包](https://drive.google.com/file/d/1sQynG6Fe2h2zmZ7MFGtAHGIhL3PAZuXF/view)。已实际下载并检查 `ReposVul_c_cpp.tar.gz`（684,170,847 字节）；连接器的 256 MiB 限制触发后，改用原公开下载端点。归档的 `random/{train,valid,test}_c_cpp_repository2.jsonl` 分别复制为本地 `data/rgcl/reposvul_official/{train,valid,test}.jsonl`，仅用于检查。

| 官方划分 | 原始标签 1 | 原始标签 0 | 总数 | 可恢复唯一变化修复版本的正例 | 有效区域监督正例 |
|---|---:|---:|---:|---:|---:|
| train | 5,532 | 180,259 | 185,791 | 748 | 434 (7.85%) |
| valid | 721 | 22,503 | 23,224 | 98 | 63 (8.74%) |
| test | 670 | 22,554 | 23,224 | 92 | 59 (8.81%) |

官方函数记录共 232,239 条：C 211,762、C++（cpp/cc）19,776、.h 701；正例 6,923、负例 225,316。完整字段检查确认该发布包没有修复前后函数/固定函数 ID 字段。发布包的行区间包含 Patch 上下文，未被直接转换成真实 Diff 正标签。

## 决定停止训练的证据

本地有 296 组**完全相同源码**同时具有原始 0/1 标签，涉及 820 条记录，其中正例 338 条，占全部正例 26.61%。若仅归一化空白，则为 316 组、865 条记录、360 条正例；停止决策以严格源码相等的结果为依据。

官方发布包也有 1,173 组严格相同源码的标签冲突，涉及 3,969 条记录和 1,501 条正例，占正例 21.68%。本地和官方能精确匹配的源码中，另有 420 条官方正例在本地仅匹配到 0 标签。来源版本与标签处理并不一致，不能按函数 ID 或列表下标强行连接。

两个已实际复核的案例（完整来源键见 `cases.json`）：

- **CVE-2012-2121 / Linux / kvm_iommu_put_pages**：本地记录 37、文件 1、before/7 与 after/7 的源码严格相同，原始标签却为 1/0。真实 Patch 在其后插入包装函数并修改调用方；该函数本身没有修改行。
- **autotrace / concat_pixel_outline**：官方 Train 第 180 和 35,998 条记录具有相同项目、修复提交、文件、源码、物理行区间和 function_id；在 CVE-2019-19005 下为 1，在 CVE-2019-19004 下为 0。单函数输入无法同时满足两个标签。

Train 内部的同源码异标签案例已足以支持停止进入训练阶段；Test 没有用于方法设计、选模或调参。这些事实说明当前导出标签/上下文不足以支持本轮单函数任务；它们不等价于所有记录都是错误标签，也不否定 ReposVul 的仓库级用途。本轮没有通过删掉矛盾样本、重新定义标签或筛选负例来构造性能结果。

## 修复对应和 512 Token 监督覆盖率

对应规则为：同一 CVE/提交/文件内，函数声明文本唯一对应，且两侧函数源码确实存在于对应完整文件中。声明改变或重复等无法可靠对应的 124 个正例标为未知；不按列表下标、名称相似度或近邻代码强行配对。因此下面是可确认覆盖率的下界。

| 指标 | 实测 |
|---|---:|
| 原正例可对应到原 after/0 函数 | 1,146/1,270 |
| 对应两侧源码严格相同但标签相反 | 262 |
| 对应两侧 512 模型输入完全相同 | 452/1,146（39.44%） |
| 两侧有非空白直接修改 / 纯新增 / 纯删除 / 无非空白修改 | 700 / 160 / 13 / 273 |
| 漏洞侧存在直接 Diff 行 | 713/1,270 |
| 漏洞侧有效可见区域及有效背景同时存在 | 514/1,270（40.47%） |
| after/0 侧有效可见区域及有效背景同时存在 | 617 条记录 |

实际调用原 `_line_diff` 和 `paired_features`，有效区域要求可见有效代码行同时含正区域和背景，遵循原 Padding/截断规则。纯新增修复不会把旧侧邻近行改成 Diff 正标签。单对记录、两侧可见性、截断及输入相同性见 `repair_pairs.jsonl`。没有可靠 Diff 的成员没有被删除或改标签；本轮没有进入分类训练。

## 官方划分隔离与分组可行性

| 划分交集 | CVE | 修复提交 | 完全相同源码组 | 空白归一化源码组 | function_id |
|---|---:|---:|---:|---:|---:|
| train:valid | 2,888 | 2,962 | 10,029 | 10,077 | 2,869 |
| train:test | 2,886 | 2,974 | 10,096 | 10,144 | 2,949 |
| valid:test | 2,633 | 2,687 | 2,881 | 2,898 | 871 |

官方随机划分不满足本轮 CVE/修复提交/重复代码隔离要求，不能直接用于这次公平对照。

复用现有 `split_rgcl_groups` 做本地全体成员的 **Seed42 分组可行性诊断**：CVE、修复提交、修复对应及空白归一化源码做传递分组，共 892 组，最大组 12,752 条记录，组间交集为 0。没有移除成员。诊断得到：

| 诊断划分 | 正例 | 负例 | 总数 |
|---|---:|---:|---:|
| train | 1,016 | 144,339 | 145,355 |
| valid | 127 | 18,043 | 18,170 |
| test | 127 | 18,042 | 18,169 |

分组可以解决跨集合泄漏，无法解决同源码原标签矛盾。这些是诊断计数，未写出正式 Train/Valid/Test 成员或抽取 1:1 子集，不能当作已完成训练的划分。

## 实际验证与复现

71 项相关测试在宿主环境通过（15.775 秒），最终定向复验 4 项通过。覆盖原始标签保留、拒绝非法标签、非下标配对、未知对应、纯新增、截断、同输入异标签、官方划分泄漏以及不生成训练成员。首次完整回归因沙箱多进程 Unix socket 的 PermissionError 受阻；准确中断该测试进程后，在宿主环境原样重跑通过，保留 `tests.sandbox_failure.txt`。

真实 RTX4090 smoke 仅使用官方 Train 的前 8 个函数和冻结的三组历史模型：确认 CLS 使用标准 RobertaClassificationHead，单函数/批量推理最大差异 ≤1.79e-7，Diff 标注变化不改变推理，输出有限，原 checkpoint 文件未修改。没有创建优化器或更新参数，不是 ReposVul 的新训练结果；没有读取 Test 做模型推理。

```bash
python -m vulnmechanism.cli rgcl-study inspect --dataset reposvul \
  --raw-root /home/PublicData/PHY-data/vul_detect/data/reposvul \
  --official-dir data/rgcl/reposvul_official --output-dir results/rgcl_reposvul_512
python -m unittest tests.test_reposvul tests.test_rgcl tests.test_bounded_attention tests.test_losver
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli rgcl-study check --dataset reposvul \
  --official-dir data/rgcl/reposvul_official --output-dir results/rgcl_reposvul_512 --device cuda:0
```

环境、实际命令和未训练状态见 `commands.json`；完整字段、标签、配对和隔离统计见 `data_inspection.json`。原始数据、官方归档、函数源码和历史模型权重留在本地，提交范围仅包括检查代码、测试、说明和不含源码的统计/案例。
