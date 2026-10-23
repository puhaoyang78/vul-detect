# LOSVER：PrimeVul 原始 Paired 复现

官方原始 Train/Valid/Test：4354/562/564 对；不重划分、不重采样。Seed 123456，Batch 8，512 Token。定位器40 Epoch按Valid加权行F1选模；三个分类器16 Epoch按Valid P-C选模。固定概率>0.5。

| 模型 | Epoch | Test P-C | P-V | P-B | P-R | Accuracy | Precision | Recall | F1 | 论文 P-C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unixcoder | 16 | 10.64% | 39.01% | 45.92% | 4.43% | 53.10% | 53.33% | 49.65% | 51.42% | 10.11% |
| losver | 8 | 14.36% | 46.99% | 34.75% | 3.90% | 55.23% | 54.66% | 61.35% | 57.81% | 14.01% |
| losver_oracle | 11 | 31.91% | 50.53% | 15.25% | 2.30% | 64.80% | 60.94% | 82.45% | 70.08% | 33.16% |

预测修改行LOSVER相对baseline：p_c +3.72个百分点、accuracy +2.13个百分点、f1 +6.39个百分点。这些是本轮固定协议的实测差值。

源Valid选模：unixcoder Epoch 16，P-C 15.12%；losver Epoch 8，P-C 16.01%；losver_oracle Epoch 11，P-C 35.05%。

| Split | 原始对数 | 函数数 | baseline编码相同对 | 加权检测器Token ID相同对 | 可见修改行/全部修改行 |
|---|---:|---:|---:|---:|---:|
| train | 4354 | 8708 | 1254 | 1249 | 22320/49771 |
| valid | 562 | 1124 | 164 | 164 | 2665/5401 |
| test | 564 | 1128 | 163 | 163 | 2684/5162 |

编码相同指Token ID相同，Oracle的真实修改行权重仍可能不同；不删除这些官方成员。跨Split交集：train:valid 源码=0、baseline编码=18；train:test 源码=0、baseline编码=13；valid:test 源码=0、baseline编码=2。

定位器选定Epoch 32：Valid加权行F1 91.60%，修改行Precision 31.56%、Recall 19.38%。加权行F1与修改行正类F1不同。

官方换行匹配保留：Valid物理可见行43430，实际定位器输出行43266；可见正标签2665，实际消费正标签2663。Tokenizer会将空白和换行合并，官方精确换行ID匹配可能遗漏这些边界；以上数量差不代表已量化全部错位。没有据此修改规则或使用Test调参。

Oracle单独读取真实修改行；正常检测只读取单函数及定位器预测，不读测试Diff。

模型和预处理忠实对照官方源码 a158905cb45b4400f481f18eeb1cb4473ccfccdc；加权检测器保留原实现的第0维风险、Softmax前padding分数0及512截断细节。基线使用官方RobertaForSequenceClassification头。

环境差异：当前Torch 2.9.1/Transformers 4.57.3，官方Torch 2.4.1/Transformers 4.19.0；使用eager FP32注意力和Transformers4.19的AdamW更新。完整参数、逐Epoch日志、预测和权重分别保存在各方法目录。

数据与编码重复、元数据异常及不可区分配对见data/primevul_paired/statistics.json；原始成员全部保留，不通过修改Test口径追逐论文指标。

论文与源码差异：正式论文IV-E的通用检测设置为8 Epoch、按Valid Accuracy选模；官方PrimeVul专用脚本默认16 Epoch，Test加载最佳Valid P-C权重。本轮采用后者，未根据Test切换权重或选择协议。
