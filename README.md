# Dynamic Hard Augmentation for Function-Level Vulnerability Detection

输入单个C/C++函数源码，输出漏洞概率和0/1预测。当前主线使用CodeBERT全参数微调、原生滑窗、mean pooling＋linear，不输入修复端、diff、CPG或标签。

## 方法

- `hard_aug_bce`（默认核心方法）：合法源码变换→当前模型按BCE动态搜索→原版与最多一个困难变体联合BCE。
- `hard_aug_vat`（增强版本）：在上述训练上加入一套普通全局VAT。
- 对照：`bce`、`vat`、`random_aug_bce`、`random_aug_vat`。
- `ours`及其四项消融只保留模型、指标和只读评价入口，不能重新启动训练。

动态生成器和可见性检查沿用已验证算法：束宽2、深度3、每函数最多12个训练候选，最多24次节点提议，新增注释最多64 token。
随机组不查询模型；困难组以当前模型的BCE选择访问候选中的最困难变体。原版与有效变体的BCE先按函数平均，无法变换的函数仍参与训练。
普通VAT在全部有效源码token上扰动，alpha=0、epsilon=1、xi=.001、一次方向估计、系数1；排除prefix/特殊token/padding。
窗口副本共享同一源码token的扰动，按实际展开后的每函数L2范数归一化。辅助前向关闭dropout并保留梯度。当前训练不包含JS、表面软加权、排序或定位监督。

## 数据和训练协议

沿用 `data/benchmark/manifest_sven_commit_disjoint.jsonl`，使用补入修复端之前的PrimeVul函数分类成员。
官方target、现有划分、去重结果和函数顺序不变，不按CPG构建成功筛选。Train 7,820（正/负各3,910）；Valid/Test各1,014（正/负各507）。
这是当前平衡清单，不能当作官方全量PrimeVul Test。其他原始数据与人工标注修复对保留，不混入当前训练。

- Seed 42；同一未经本任务微调的CodeBERT基座、相同分类头初始化；完整编码器＋分类头更新。
- 2048逻辑总预算：prefix 21＋BOS/EOS 2＋源码最多2025；原生512窗口、源码容量489、步长244。
- 同一源码token重复窗口贡献去重，prefix/BOS/EOS跨窗口平均，整个函数只产生一个分类logit。
- 3 epochs、每轮978步、总2,934步；batch=4、累积2；AdamW lr=1e-5、weight_decay=.01、梯度裁剪1。
- 仅用Clean Valid MCC选择Epoch和阈值，并沿用F1/Accuracy/阈值接近.5的tie-break。Test使用保存的权重和阈值。
- checkpoint版本8、数据schema9不变，encoder/head严格加载；旧11组线性CodeBERT模型可继续恢复。其他历史架构使用归档源码。

## 统一入口

```bash
PY=/home/phy/miniconda3/envs/vul-detect/bin/python
# 准备与真实小样本检查，不是正式训练
$PY -m vulnmechanism.cli codebert-finetuning --stage prepare
$PY -m vulnmechanism.cli codebert-finetuning --robustness all --stage smoke --device cuda:0
# 核心方法和增强版本；匹配的已完成模型直接复用
$PY -m vulnmechanism.cli codebert-finetuning --robustness hard_aug_bce --stage all --device cuda:0
$PY -m vulnmechanism.cli codebert-finetuning --robustness hard_aug_vat --stage all --device cuda:0
# 六组活动方法的Valid汇总与固定权重Test评价
$PY -m vulnmechanism.cli codebert-finetuning --robustness all --stage report
$PY -m vulnmechanism.cli codebert-finetuning --robustness all --stage test
# 历史11组只读汇总；要求已有checkpoint/预测/评价缓存
$PY -m vulnmechanism.cli codebert-finetuning --robustness all --include-history --stage report
$PY -m vulnmechanism.cli codebert-finetuning --robustness all --include-history --stage test
```

新活动汇总写入`mainline.comparison.metrics.json/csv`，历史只读汇总写入`history.comparison.metrics.json/csv`；Test文件加`test.`前缀。
原三组、七组和11组汇总文件保留，不覆盖。`--include-incremental`仅作为已有11组只读命令的别名。

## 11组已完成实验

以下是清理前已完成的Seed 42实验，未重新训练、未重选阈值。每格依次为Clean Accuracy/MCC/AUC：

| 方法 | Epoch/阈值 | Valid | Test |
|---|---|---|---|
| bce | 3/0.29 | 0.7505/0.5082/0.8117 | 0.7485/0.5064/0.8325 |
| vat | 3/0.31 | 0.7416/0.4849/0.8052 | 0.7544/0.5089/0.8384 |
| ours | 3/0.45 | 0.7446/0.4901/0.8038 | 0.7525/0.5049/0.8373 |
| ours_no_search | 3/0.42 | 0.7465/0.4943/0.8120 | 0.7574/0.5149/0.8399 |
| ours_uniform | 3/0.41 | 0.7426/0.4852/0.8038 | 0.7594/0.5198/0.8392 |
| ours_no_consistency | 3/0.42 | 0.7485/0.4976/0.8088 | 0.7475/0.4952/0.8368 |
| ours_no_vat | 2/0.26 | 0.7446/0.4894/0.8090 | 0.7643/0.5287/0.8444 |
| random_aug_bce | 2/0.35 | 0.7475/0.4986/0.8110 | 0.7623/0.5255/0.8365 |
| hard_aug_bce | 3/0.41 | 0.7367/0.4784/0.8083 | 0.7485/0.5060/0.8280 |
| random_aug_vat | 3/0.33 | 0.7465/0.4944/0.8043 | 0.7515/0.5078/0.8391 |
| hard_aug_vat | 3/0.46 | 0.7456/0.4937/0.8074 | 0.7505/0.5012/0.8395 |

自适应表每格依次为最坏Accuracy/MCC/AUC；ASR和全部候选正确率按Valid/Test排列。
最坏检测包含原版和无候选函数，完整分母均为1,014；全部候选正确率要求原版及所有访问候选均正确，有效分母为Valid 655/Test 609。
ASR仅以其中原版预测正确者为分母，各模型不同；完整计数分母见JSON。主表包括函数名input-only stress，严格语义保持指标单列在JSON。

| 方法 | Valid自适应 | Test自适应 | ASR V/T | 全部正确 V/T | 训练分钟 |
|---|---|---|---|---|---:|
| bce | 0.4921/-0.0187/0.3455 | 0.5582/0.1324/0.4231 | 53.25%/44.68% | 35.11%/39.24% | 未记录 |
| vat | 0.5927/0.1881/0.6080 | 0.6341/0.2735/0.6461 | 31.01%/27.98% | 51.30%/51.56% | 52.21 |
| ours | 0.7160/0.4322/0.7691 | 0.7219/0.4441/0.8013 | 5.97%/7.09% | 69.77%/66.67% | 177.47 |
| ours_no_search | 0.6884/0.3768/0.7374 | 0.6992/0.3988/0.7755 | 11.94%/13.50% | 66.41%/62.07% | 146.41 |
| ours_uniform | 0.7101/0.4202/0.7670 | 0.7288/0.4597/0.8063 | 6.79%/7.06% | 69.16%/67.00% | 175.26 |
| ours_no_consistency | 0.7249/0.4498/0.7743 | 0.7209/0.4421/0.8016 | 4.91%/6.29% | 70.99%/66.01% | 157.34 |
| ours_no_vat | 0.7120/0.4243/0.7619 | 0.7160/0.4320/0.7926 | 6.78%/10.94% | 69.31%/65.52% | 137.78 |
| random_aug_bce | 0.6479/0.2986/0.6575 | 0.6588/0.3191/0.6902 | 20.61%/23.49% | 59.39%/56.16% | 84.60 |
| hard_aug_bce | 0.6874/0.3815/0.7508 | 0.7101/0.4316/0.7750 | 10.42%/8.99% | 65.65%/64.86% | 118.11 |
| random_aug_vat | 0.6755/0.3546/0.7237 | 0.6854/0.3790/0.7601 | 14.75%/15.44% | 63.51%/60.26% | 121.13 |
| hard_aug_vat | 0.7189/0.4396/0.7730 | 0.7199/0.4399/0.8070 | 5.52%/7.13% | 70.53%/66.34% | 155.55 |

固定/留出候选在所有模型间共享；自适应攻击各模型独立搜索，最多48个候选。固定/留出完整Accuracy、MCC、AUC、ASR、覆盖和计数继续保存在原JSON/CSV。
训练时间包括Clean Valid选模；历史BCE未记录耗时，不补估计值。

- 随机增强相对BCE的Test自适应Accuracy增加10.06个百分点。
- 无VAT时，困难搜索相对随机增强增加5.13个百分点；有VAT时仍增加3.45个百分点。
- 普通VAT在随机/困难增强上分别增加2.66/0.99个百分点。
- 完整旧`ours`比`hard_aug_vat`的Test自适应Accuracy仅高0.20个百分点（2个函数），Valid反而低0.30个百分点；训练多用21.92分钟。
- JS、软加权的额外收益较小或在Valid/Test间方向不一致，尚无证据证明必须保留。收益主要是抗扰动，Clean指标并非全面提高。

[Valid完整结果](results/codebert_robustness_seed42/incremental.comparison.metrics.json) · [Test完整结果](results/codebert_robustness_seed42/test.incremental.comparison.metrics.json) · [Test CSV](results/codebert_robustness_seed42/test.incremental.comparison.metrics.csv)

## 模型加载与单函数推理

```bash
$PY -m vulnmechanism.cli infer --checkpoint results/codebert_robustness_seed42/hard_aug_bce/best.pt --source-file function.c --language c --device cuda:0
```

只需提供一个原始函数；输出logit、概率、保存阈值和0/1预测，不要求标签或配对信息。
Python入口为`vulnmechanism.model.predict_checkpoint(checkpoint, [{"raw_source": code}], device="cuda:0")`，返回原始logit；使用checkpoint内保存的阈值判别。

## 历史归档

见[研究归档索引](docs/EXPERIMENT_HISTORY.md)。清理前完整源码、数据、模型、日志、预测、素材和候选已备份，历史结果不能与当前不同cohort直接比较。
保留整个当前11组实验目录、整个全参数/LoRA基线目录、全部data及未解决的workspace文件。历史资源通过仓库外备份/隔离区恢复。
本地详细清理报告、逐文件清单、依赖和恢复证据位于`cleanup_audit/20261014094242Z`。
当前cleanup分支等待用户审核；发布脚本拒绝从cleanup分支推送，只允许`--dry-run`。

## 下一阶段：高质量修复对监督（尚未验证）

已保留相邻项目的`high_quality_repair_pairs.json`（1,279条）及源身份/配对核查、原数据准备工具。
计划研究安全相关特征判别辅助任务，主任务仍是单函数0/1；不能把任意修复差异直接转成无条件函数标签。
此方向尚未实现或验证，本次没有新增训练。未来需要VAT时复用当前全局组件，不建立第二套VAT。

## 主线测试

```bash
$PY -m py_compile vulnmechanism/*.py
$PY -m unittest tests.test_mainline tests.test_codebert_source tests.test_surface_vat tests.test_prepare_benchmark tests.test_benchmark_view tests.test_training_progress tests.test_publish_code_metrics -v
bash scripts/publish_code_metrics.sh --dry-run
```
