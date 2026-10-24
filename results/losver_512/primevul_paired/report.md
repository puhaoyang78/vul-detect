# LOSVER：PrimeVul Paired，输入模式512

官方原始 Train/Valid/Test：4354/562/564 对；不重划分、不重采样。Seed 123456，函数Batch 8，每窗口最多512 Token。定位器40 Epoch按Valid修改行正类F1选模；三个分类器16 Epoch按Valid P-C选模。固定概率>0.5。

| 模型 | Epoch | Test P-C | P-V | P-B | P-R | Accuracy | Precision | Recall | F1 | MCC | AUC | AUPRC | 论文 P-C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unixcoder | 14 | 14.54% | 47.16% | 34.93% | 3.37% | 55.59% | 54.98% | 61.70% | 58.15% | 0.1125 | 0.5729 | 0.5664 | 10.11% |
| losver | 14 | 14.01% | 36.70% | 44.15% | 5.14% | 54.43% | 54.79% | 50.71% | 52.67% | 0.0889 | 0.5679 | 0.5669 | 14.01% |
| losver_oracle | 5 | 29.96% | 41.84% | 24.47% | 3.72% | 63.12% | 61.18% | 71.81% | 66.07% | 0.2665 | 0.7028 | 0.6765 | 33.16% |

论文P-C仅为历史参考；当前输入与分类头协议已修正，不作为公平对照。L1应与同一修正版512训练结果比较。

预测修改行LOSVER相对baseline：p_c -0.53个百分点、accuracy -1.15个百分点、f1 -5.48个百分点。这些是本轮固定协议的实测差值。

源Valid选模：unixcoder Epoch 14，P-C 16.19%；losver Epoch 14，P-C 14.23%；losver_oracle Epoch 5，P-C 33.81%。

定位器选定Epoch 8：Valid修改行正类F1 28.05%，Precision 24.96%、Recall 32.02%。

三个检测器共享UniXcoder、核心代码Token聚合及线性分类头。统一原始源码offset物理行映射；仅改变修改行5/1权重，Padding、特殊符号与纯空白Token不参与归一化。输入为510源码Token+BOS/EOS，截断末行保留可见片段。

Oracle单独读取真实修改行；正常单函数推理只使用源码与定位器预测。未使用Test调参。data_preparation仅保留原准备记录；当前输入覆盖审计见check-paired及训练配置中的input_coverage，不将历史截断统计当作L1覆盖率。

Seed123456、学习率2e-5、Batch8、旧Transformers AdamW、10% warmup和原Focal参数不变；定位器40 Epoch、三个检测器各16 Epoch。输入/聚合/分类头及定位选模已按本次要求修正，不能称为原始源码的数值一比一复现。
