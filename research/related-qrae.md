# QRAE 相关工作与可辩护的研究贡献

调研日期：2026-10-26。只核查作者论文、会议论文页和官方文档。以下“建议”是本次研究设计判断，不是已经完成的实验结论。

## 最重要的判断

**冻结 embeddings 后，再用 relevance labels 学一个低维 projection，并不是尚未有人研究的新方向。** Proposal 可以继续做，但最稳妥的课程贡献是：在统一、无测试标签泄漏的设置下，系统验证 RAE 的几何保持目标与真实 retrieval utility 的差异，并研究低维、监督量和负样本的交互。不要把“RAE + triplet loss”本身宣称为首创。

## 建议阅读顺序：七篇

| 优先级 | 论文与一手来源 | 对项目的具体用途 |
|---|---|---|
| 1 | [Matryoshka-Adaptor, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.576/)；[PDF](https://aclanthology.org/2024.emnlp-main.576.pdf) | 最直接的 prior art：针对既有 embeddings 做有监督/无监督的多维压缩。§4 的 joint objective 已包含几何/重建类约束与 query–document ranking。必须用它界定 QRAE 的贡献。 |
| 2 | [Search-Adaptor, ACL 2024](https://aclanthology.org/2024.acl-long.661/)；[全文 §4–5](https://arxiv.org/html/2310.08750v3) | 冻结向量的共享 residual adapter，用 relevance ranking 与 recovery 等正则训练。其 train/validation/test 分离值得参考；其完整方法还有 query predictor，不能用极简 ranking + L1 就声称严格复现。 |
| 3 | [DIVE, arXiv 2605.20689v1](https://arxiv.org/html/2605.20689v1) | RAE 作者之一 Dongfang Zhao 的近邻方向：非线性多头 adapter，triplet + head-wise contrastive。与 QRAE 不同之处是没有 encoder–decoder reconstruction；更要注意其 §4.1 用同一批有标签 queries 训练并评估，结果不能当作 held-out query 指标。附录 F 说完整代码将在发表时发布，本次没有核实到作者完整 GitHub。 |
| 4 | [DPR, EMNLP 2020](https://aclanthology.org/2020.emnlp-main.550/)；[PDF §3.2、Table 3](https://aclanthology.org/2020.emnlp-main.550.pdf) | 学会 in-batch negatives、hard negatives 与 softmax ranking objective。项目可复用这种损失思想；无需下载 DPR 模型，也无需 fork 整个 QA 工程。 |
| 5 | [SMEC, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1332/) | 更新 related work：顺序多尺度训练、维度选择与跨 batch memory。先理解其与单维 QRAE 的区别；无需为了课程强行实现全部复杂组件。 |
| 6 | [Matryoshka Representation Learning, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html)；[作者仓库](https://github.com/RAIVNLab/MRL) | 理解为不同 prefix 维度安排训练目标的思想。原始 MRL 和事后 adaptor 不同；普通 E5/MPNet 直接截前 64 维不等于 MRL。 |
| 7 | [LeanVec, TMLR 2024](https://openreview.net/forum?id=wczqrpOrIc)；[原论文](https://arxiv.org/abs/2312.16335) | 已有 query-aware dimensionality reduction，考虑 query 与 corpus 分布差异。说明“query-aware”不是新概念；其原始目标主要是相似搜索近似，与你们的 relevance-label ranking 研究仍可区分。 |

补充：若需要讲清“压缩 embeddings”与“压缩 encoder 模型”的区别，再读 [HPD, Findings ACL 2022](https://aclanthology.org/2022.findings-acl.64/)。它包含 student Transformer 与 projection distillation，和固定 encoder 的后处理压缩不是完全相同的设置。

## Proposal 应如何定位

推荐表述：

> We study when retrieval supervision improves low-dimensional projections of frozen embeddings beyond geometry-preserving compression. We independently reproduce linear RAE and evaluate a retrieval-supervised extension under a shared, held-out-query retrieval protocol, with controlled ablations on dimension, label budget, and negative sampling.

这里“independently reproduce”需要由组员独立实现并理解，而不是复制作者工程。

当前 QRAE 若使用共享静态矩阵 W_e，它是 **retrieval-supervised / task-aware**，而不是“对每一个输入 query 动态生成降维映射”。可以保留 QRAE 暂名，但应解释 query-aware 指训练监督包含查询相关性。所有 documents 仍能预先投影并建立同一个 index。

## 可执行的 QRAE 最小版本（建议，不声称新理论）

1. 固定同一个 pretrained text encoder；预先保存规范化后的 query/document embeddings。
2. AE、RAE、QRAE 使用相同形状的 encoder/decoder。先做线性版本，避免同时改变结构和 loss。
3. reconstruction 使用未做 latent normalize 的 z；ranking 使用 L2 normalize 后的 z，与你们最终 cosine 检索一致。
4. 从训练 qrels 生成 (q, d+, d−)。先实现单 negative 的 hinge loss：`max(0, margin - cos(zq,zpos) + cos(zq,zneg))`。这只是标准 triplet ranking，不应当作原创损失。
5. 各项 loss 的 reduction 明确定义；MSE 的 mean 与 sum 差 D 倍，调参范围必须按实际数值尺度设定。
6. 首次 pilot 只做 128d，确认训练集 ranking loss 可降低、validation nDCG 有意义、没有 NaN 和 latent collapse，再扩大维度。
7. 之后将 triplet 替换为 batch softmax/InfoNCE 做一个次要对照即可。使用多正例 mask，不能把同一 query 的另一个正例当负例。

## 哪些控制实验最能增强结论（建议）

| 实验 | 解决的替代解释 |
|---|---|
| AE vs RAE vs QRAE，相同结构/训练样本/预算 | 提升来自 regularization，还是 relevance supervision？ |
| **Rank-only projection**（相同 W_e，去掉 reconstruction/reg） | 是否只要一个监督投影即可，RAE 的基础到底有无贡献？ |
| 768d supervised adapter / square projection | 压缩后的收益是否只是领域监督适配，而非压缩本身？ |
| QRAE 的 random / in-batch / frozen-full-space hard negatives | hard negative 是否真正有帮助？ |
| 固定负样本数与优化 steps 的采样比较 | 排除“更多 negatives/更多计算”造成的混淆。若 in-batch 使用 B−1 个，应明确报告。 |
| label budgets 如 10%/25%/100%（从训练 queries 抽样） | 小数据时监督收益是否稳定，何时过拟合？ |
| 相同维度内画 NN-overlap vs nDCG@10 | 区分维度本身导致的共同上升与 geometry utility 的关系。 |
| 3 个 seeds + paired query bootstrap CI | 小差异是否超出随机波动？ |

不要把所有 ablation 铺满所有数据集/维度。主表保留六法，额外控制集中在 SciFact、FiQA 的 64d/128d；NFCorpus 做外部验证即可。

## Hard negatives 的关键细节（建议）

- 只为训练 query mining，使用训练 query embeddings 搜索完整 corpus；不读取 held-out query 的 qrels 进行训练选择。
- 从 full-dimensional frozen retriever 的 top-K 取候选，排除该训练 query 的所有已知正例。保存 mining config、candidate IDs 与 seed。
- 未出现在 qrels 不等于已确认 irrelevant。BEIR relevance judgments 不完整时，某些高相似未标注文档可能是 false negatives；报告这一限制。
- 第一版使用静态 negatives，以便所有方法/seed 的比较可控。再考虑每隔若干 epochs mining，作为单独的 advanced ablation。
- 避免用 test nDCG 决定 margin、alpha、lambda、early stopping 或 hard-negative rank 范围。

## DIVE 为什么只能作为 related work，不能照搬其数值目标

直接核查的一手内容是 §4.1 的 Training Protocol：训练 triplets 来自全部可用 relevance judgments，并在相同 query set 评估。**这是 seen-query adaptation，不是 unseen-query generalization。** 同样设置下比较多个方法可以回答该设置的问题，但不能外推成你们严格 held-out test 的目标。

此外，我们自己的数学检查发现其附录 A.5 中 hinge loss 的文字条件与定义不一致：对 `max(0,m-Δ)`，提高 m 会让更多、不是更少的 triplets active；m=0 也仍然具有 hinge 的零梯度区。故不要转述“新 self-limiting 原理”或其理论保证为既定事实。这个检查是代数推导，不是对论文全部结果的否定。

建议 report 对其只陈述方法和实验 protocol 的差异，不作攻击性评价，也不用其 .99 级 SciFact 数字来评判你们代码是否正确。
