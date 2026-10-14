# 历史研究归档索引

当前主线为动态困难源码增强，详见README。历史方法不再暴露训练入口；原代码、测试、模型、日志和预测仍在冷备份中。

## 恢复来源

- 原始Git提交：`c18e5e2aa295e96652d6392d291dd2fa5e2c90d8`。
- 恢复引用：`backup/pre-cleanup-20261014094242z`。
- 完整备份：`/home/PublicData/PHY-data/vul_detect/backups/vul-detect-20261014094242Z/repository/`，Git历史另见同级`pre-cleanup.bundle`。
- 隔离重复副本：2026-10-14经用户确认，删除清单内102个路径、1,071个文件（7.93 GiB）；对应完整备份保留。逐文件校验与删除记录见本地清理报告及`confirmed.deletion.json`。
- 正式11组实验与匹配BCE/LoRA/全参数基线仍保留在活动仓库。
- 原长篇README、CFG说明和完整早期总结在备份的相同相对路径；以下只作索引，不把不同cohort的成绩合并。

## 研究问题、方法及证据边界

| 历史方向 | 研究问题和方法 | 主要结果、失败原因或局限的证据 | 原始出处 |
|---|---|---|---|
| CPG/CFG、机制融合与程序关系 | 检验静态关系、属性和传播是否补充源码分类 | 原总结记录：静态关系可解释部分修复，但未形成稳定全面分类增益；CFG属性在旧741条Valid有信号，传播未增加收益。不能与当前1,014条清单直接比较 | 备份README、docs/EXPERIMENT_HISTORY.md、docs/CFG_ABLATION.md和cfg_*结果 |
| 局部风险、修复排序与区域监督 | 利用修复对增加安全差异敏感性 | 具体逐阶段结果以local_risk/change_learning产物为准，保留旧cohort、配对覆盖及截断限制，不声称已经确认因果根因 | 备份README、change_learning.py及相关结果/数据 |
| 浅层重加权、High–Low、长度去偏 | 检验规模/词法捷径及增强选择 | 不同方案结果有退化或取舍；保留OOF、分组口径及原阈值，不能把相关性当成原因 | 备份README、audit_statistics.py及codebert_shallow_*/conditional_*结果 |
| Surface/Binding VAT、旧随机增强 | 检验表面位置及绑定共享扰动 | 属于旧协议，不能混入当前动态搜索11组表；本轮只保留一套普通全局VAT | 备份README、原model.py/test_surface_vat.py和codebert_surface_vat_* |
| CGVL/OCA/EGCL、多原型、级联与跨数据集 | 检验构造监督、模型结构和迁移 | 每个方向保留自己的成员、阈值和结果。多数不支持全面提升；未记录的原因保留unknown，不用当前结果反推 | 备份完整源码、README及对应结果目录 |

## 本次移入冷备份的结果目录

下列每个目录的完整模型、配置、日志、预测和指标均可从`备份/repository/原路径`恢复。完整备份之外的隔离重复副本已逐文件校验并按用户确认删除。

| 原路径 | 文件数 | 大小MiB | 用途与处置依据 |
|---|---:|---:|---|
| results/cfg_abc_seed42 | 31 | 121.12 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_joint_seed42 | 71 | 168.35 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_pool_seed42 | 39 | 86.25 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_transport_seed42 | 39 | 84.81 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_project_relations_seed42 | 5 | 76.41 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_project_cfg_seed42 | 23 | 41.71 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_project_transport_seed42 | 23 | 42.59 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_origin_source_seed42 | 16 | 39.52 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/codebert_prototype_seed42 | 34 | 253.81 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cross_dataset_seed42 | 35 | 538.17 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/change_learning_smoke_seed42 | 24 | 86.27 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/local_risk_smoke_seed42 | 16 | 88.38 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/codebert_shallow_reweight_seed42 | 12 | 479.38 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/codebert_surface_vat_fp32_norm_seed42 | 20 | 964.49 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/codebert_surface_vat_mixed_token_seed42 | 4 | 0.03 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_dep_cfg_seed42 | 23 | 82.18 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_dep_cfg_windowfix_seed42 | 15 | 41.28 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_project_transport_shuffled_seed42 | 23 | 42.59 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_project_joint_seed42 | 23 | 42.59 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_origin_attributes_seed42 | 15 | 41.29 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/cfg_clm_source_seed42 | 16 | 39.52 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/codebert_source_seed42 | 19 | 10.72 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/change_learning_seed42 | 178 | 445.05 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/local_risk_seed42 | 158 | 474.77 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/codebert_shallow_augmentation_seed42 | 30 | 959.44 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/codebert_conditional_length_seed42 | 12 | 474.33 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |
| results/codebert_surface_vat_seed42 | 84 | 2391.69 | 保存指标结构与原README确认的历史实验；活动依赖闭包不引用；完整冷备 |

## 保留与未解决项

全部原始/准备数据、可能人工核查的repair/checked/binding/operation产物及无法找到充分出处的目录均保留。
`workspace/cpg.bin`的重建输入已不存在，标为UNRESOLVED；本次仅停止跟踪其二进制，保留本地文件和完整备份。
人工修复对位于相邻项目，1,279条，已单独备份；不根据补丁差异重写函数标签。

## 恢复方法

先确认没有活动任务正在写目标路径，再从完整备份复制指定原路径，核对`SHA256SUMS`后使用原Git恢复引用加载历史代码。
不要把不同版本的历史模块单独拷入当前主线；恢复历史代码应使用独立工作目录及完整bundle。
完整逐文件、模块调用/导入、进程、未知项和预测一致性证据见`cleanup_audit/20261014094242Z/report.md`。
