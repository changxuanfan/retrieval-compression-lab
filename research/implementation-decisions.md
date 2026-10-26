# 前四项实现的补充研究与决策依据

核查日期：2026-10-26。任务边界是 **Original、PCA、Gaussian Random Projection、Standard Linear Autoencoder**。RAE/QRAE 仅规划，不在这里实现。本文基于论文、官方文档和源码；“建议”不是已经测量的结果，实际实现与实验状态以 README 和输出 manifest 为准。

## 1. 独立实现 PCA/RP，标准库作数值校验

官方 PCA 会中心化，但不默认逐特征标准化；完整 SVD、随机化 SVD 和 covariance-eigh 是不同 solver。为了避免版本改变 `auto` 选择，独立 NumPy 实现可显式采用完整 SVD：`mean=X.mean(0)`，`_,s,Vt=svd(X-mean)`，`Z=(X-mean)@Vt[:d].T`。保存均值、投影矩阵、奇异值、fit 样本数和 solver；query 绝不能自己拟合均值。对照 sklearn 时使用 `svd_solver="full", whiten=False`。[sklearn PCA 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

建议测试比较投影矩阵 `W@W.T`、重建误差或 pairwise distances，而不是 loading 逐元素完全相等：SVD 符号可相反，重复奇异值对应的基底也不唯一。测试数据应避免精确重复奇异值。中心化本身改变 cosine，因此 Original 不能替换为全维 PCA；这是公式直接推导，不是运行结论。

Gaussian RP 的元素分布为 `N(0,1/d)`，也就是生成时 **标准差** `1/sqrt(d)`。`numpy.default_rng(seed)` 和 sklearn 使用的 RNG 算法不必生成相同矩阵；不能把“相同 seed 数字下逐元素不同”判为实现错误。需要保存真实矩阵；以固定矩阵验证 transform、同种子复现、不同种子不同，再验证矩阵尺度与 expected norm 的统计趋势。[sklearn GRP 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html)

## 2. Standard AE 的定义必须写清楚

建议第一版选择与后续线性 RAE 可匹配的轻量结构：一个 encoder `D→d`，一个 decoder `d→D`，无激活、无 dropout、无 relevance ranking loss。若 `bias=False`，必须显式传入：PyTorch `nn.Linear` 默认是 `bias=True`，默认参数初始化为由 input width 决定的均匀分布。[Linear 官方定义与初始化](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html)

“线性 AE 等价 PCA”只能在清楚的条件下使用。论文说明线性 AE 权重可以张成 principal subspace，但不必等于 principal component loading vectors，因而 latent cosine ranking 不自动相同。[Plaut 2018](https://arxiv.org/abs/1804.10253)

进一步的独立推导：如果 AE 使用无 bias 网络，直接训练**未中心化**的单位 embedding，它优化的是原始数据矩阵的低秩重建，不能无条件说它等于**中心化** PCA 的子空间。两种可辩护协议是：

- 保留 raw/no-bias AE，说明它是未来 RAE 去掉正则的控制，报告与 PCA 的中心化差异。
- 显式中心化 AE 输入，并将均值作为保存状态的一部分；也明确这与 raw RAE 的不同。

不要在看到 test 分数后再挑选协议。若需要归因分析，可以预先添加 centered-full-dimensional 或 centered-AE diagnostic，但不把 diagnostic 混成第五种主方法。

建议 AE 训练采用 `Adam(..., weight_decay=0)`，只最小化 reconstruction；不要使用默认 weight decay 非零的 AdamW 却称 vanilla AE。PyTorch Adam 文档区分 coupled L2 与 decoupled weight decay。[Adam 官方文档](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)

推荐 reconstruction 定义为每条向量平方误差之和，再对 batch 取平均。PyTorch `MSELoss(reduction="mean")` 是对所有元素取平均，两者差 D 倍。记录实际公式和 loss scale，后续 RAE/QRAE 权重才能解释。[MSELoss 官方定义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html)

Decoder 接收原始 latent；只有 retrieval 阶段再 L2 normalize。若先 normalize latent 再喂 decoder，已改变 AE 目标和可恢复尺度，不能悄悄当作同一个 baseline。

## 3. Query-disjoint 与 corpus-only 的边界

本次前四法都不使用 relevance labels 来拟合。主设置可统一使用全量目标 corpus 的无标签向量，报告为 **corpus-adaptive unsupervised compression**；这与未来使用 train queries/qrels 的 supervised QRAE 要明确区分。

AE 若保留 corpus reconstruction validation split，必须写明如何选择 checkpoint，以及最后是否重新拟合全部 corpus。若全部 corpus 用于训练，训练 loss 不能写成 unseen-document validation loss。两种设置均不意味着可用 test query relevance 调参。

建议用固定 update budget 和显式记录最终 checkpoint 作为默认，可避免为了初版引入 hidden query supervision。调学习率、epochs、checkpoint 的决定也属于 tuning；若依据 dev nDCG 选择，应在报告中诚实写明，test 始终保留。

数据读取应验证：每个 ID 唯一；qrels 的 query/document 存在；同 query 的所有标签留在一个 split；重复 qrels 行不能静默覆盖成另一标签。字符串 `query_id==doc_id` 不表示语义上的 self-match：BEIR evaluator 默认会删除同 ID 项，这是库行为而非所有数据集通用真理。[BEIR evaluation.py](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)

## 4. 分数实现：gain、ties、缺失 query

为了和 BEIR 的 `pytrec_eval ndcg_cut` 对齐，DCG gain 使用 qrels relevance 本身，而不是自动使用 `2^rel-1`。NFCorpus 的1、2级标注尤其会暴露差异。[NIST trec_eval ndcg_cut 源码](https://github.com/usnistgov/trec_eval/blob/master/m_ndcg_cut.c)

sklearn `ndcg_score` **同样以传入的真实分数作为 gain**，不能误称 sklearn 一定使用 exponential gain；它默认对同预测分数 ties 取平均。因此自写的固定 doc-ID tie-break 结果不能对 tied fixtures 要求和 sklearn 默认精确相等。它要求全候选的 `y_true/y_score`；如果只传已返回 top-k，会遗漏未召回相关文档，错误改变 ideal DCG。[sklearn ndcg_score 官方说明与 tie 例子](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)

所有具有评估 qrels 的 query 都要计入平均：无命中结果为0，不能只遍历 runfile 中“有分数”的 query。Recall 分母是该 query 全部已知相关文档，不是 min(k,相关文档数)，也不是 Hit@k。MRR 需写 cutoff。

精确搜索不保证 ties 的唯一排序，FAISS 官方 FAQ 也说明相同距离会产生任意顺序。仅把已经返回的 top-k 按 doc ID 重排，无法处理 cutoff 外还有同分文档的情况。建议：报告使用原生 FAISS tie 选择，校验时用无 ties 样例；或实现明确的全局 tie expansion/canonical ordering，但不要假称排序 top-k 已经完全解决。[FAISS FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)

## 5. Mac、seed 与保存状态

PyTorch 官方明确不保证不同 release、平台、CPU/GPU 的 bitwise 复现。`use_deterministic_algorithms(True)` 可限制已知非确定操作，但会影响速度或因不支持报错，不能把设 seed 宣称为跨设备严格复现。[PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

建议 seed 覆盖 Python、NumPy、PyTorch；DataLoader 若 shuffle，单独给固定 generator，初版 `num_workers=0`。记录 device、版本、seed、训练 update 数与 batch size。Mac MPS 的加速运行与 CPU numerical regression 分开：CPU 小测试负责稳定验证，真实 MPS 实验保存实际版本/输出。

如果实现 resume，仅保存网络权重不等于继续同一训练轨迹：至少还需 optimizer、scheduler（若有）、update/epoch、数据顺序/采样器状态和 RNG state。MPS 提供独立 RNG get/set API；如果未实现这些内容，文档应称“加载已训练压缩器作推理”，不要称“精确续训”。[MPS get_rng_state](https://docs.pytorch.org/docs/2.14/generated/torch.mps.get_rng_state.html)、[MPS set_rng_state](https://docs.pytorch.org/docs/2.14/generated/torch.mps.set_rng_state.html)

## 6. 下载与来源记录

数据和权重下载需要保存 URL/revision、获取时间、文件SHA256和来源，代码许可证与 dataset license 分开记录。三份 BeIR 官方 Hub cards 均标注 CC-BY-SA-4.0，但报告仍应引用原数据集与 BEIR；不要把第三方原始文本整体声明为自己代码许可证。[SciFact card](https://huggingface.co/datasets/BeIR/scifact)、[NFCorpus card](https://huggingface.co/datasets/BeIR/nfcorpus)、[FiQA card](https://huggingface.co/datasets/BeIR/fiqa)

独立 zip downloader 应检查目标路径不逃出数据目录、拒绝绝对路径/`..`、必要时拒绝 symlink；先下载 `.part`，完整通过校验后 rename，避免中断留下“看似完整”的文件。Python 官方要求解压前检查 archive 成员。[zipfile 官方警告](https://docs.python.org/3/library/zipfile.html)

建议只在本地缓存原始数据/大模型，Git 忽略这些大文件；提交 manifest、配置、运行脚本、结果表和出处即可。下载器应验证期望数据布局和条目数，不依赖一个目录存在就当作已成功下载。

## 7. 最有意义的验证清单

1. 手算 graded qrels 样例：线性 gain nDCG、Recall、多正例 MRR；有相关文档未召回和一个完全无命中 query。
2. exact search 与 NumPy brute-force cosine 在无 ties 数据上对齐；top-k 超过 corpus 时安全截断；doc→doc 排除自身使用 ID 过滤而不是假设第一项永远是自己。
3. PCA 与 sklearn 完整 SVD 的投影子空间/重建误差一致；保存和加载后 query transform 一致。
4. RP 固定 seed/矩阵能复现，query/doc 同矩阵；零向量/NaN/维度错配及时报错。
5. AE 在固定低秩合成数据上重建 loss 有实质下降；保存/加载后的 encoder 输出一致；训练不会修改原始 embedding 缓存。
6. end-to-end tiny fixture 包括数据解析→缓存→四法→全 corpus search→全部query评分→结果导出；结果标为 synthetic/smoke，不能充当BEIR正式分数。
7. 真实 SciFact pilot 最少完整 corpus、固定一组 held-out queries；若为省时缩减 corpus，必须标注为 reduced-corpus smoke，并验证其 qrels 已正确过滤，不能与全 corpus 分数混表。

这些验证用于检查方法和协议，不要求每个参数设置都有“实现镜像式”单元测试。真实实验结果、耗时及失败记录由运行后写入README，本文不捏造预期分数。
