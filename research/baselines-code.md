# 前三个 baseline 与统一评估：实现研究记录

调研日期：2026-10-26。本文是实施设计，代码仅为尚未在真实模型/数据上运行的接口骨架；不含实验结果。所有建议均需与组内 RAE 的原论文复现协议对齐。

## 1. 实施判断

前三项不需要寻找另一个完整科研项目来改。使用标准库生成冻结模型的 embedding、拟合两个压缩器，自行实现共享的数据/缓存/搜索/指标/实验编排，工作量集中在实验严谨性。AE、RAE、QRAE 必须与这三项读同一份原始向量缓存、相同 ID 顺序与相同评价 query，避免六个人各跑一套不一致系统。

建议依赖：`torch`、`sentence-transformers`、`numpy`、`scikit-learn`、`faiss-cpu`；数据下载可使用 BEIR；表格/画图可用 pandas、matplotlib。固定 Python 和所有包的版本、模型 revision SHA、数据校验值、配置文件及随机种子。不要混用量化 embedding 与 float32 embedding，否则维度与量化两种因素无法区分。SentenceTransformer 的 encode API 提供 numpy 输出、float32 precision 与 normalize_embeddings 参数。[SentenceTransformer 官方 API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)

## 2. 三个方法的精确定义

### Original

冻结检索模型产生完整 D 维 query/document 向量，按模型要求加前缀、pooling、截断与 L2 normalization，之后不降维。所有方法从这一缓存出发，避免重复编码。Original 不是“PCA 维度设成 768”。缓存除了 vectors.npy，还应保存有序 doc_ids.json/query_ids.json、模型 revision、文本拼接规则、最大 token 数、dtype 与 normalization 状态。

### PCA

在允许的训练文档向量 `X_fit` 上 fit，之后使用同一个 `mean_` 和 `components_` 变换 corpus 与 queries：

`z = (x - mean_train) @ components.T`。

scikit-learn PCA 会中心化但不逐维标准化；`whiten=False` 避免另加白化因素，`svd_solver="full"` 对本任务 768 维、几万文档规模是一个清晰确定的起点；若换 randomized solver 必须固定 random_state 并说明近似求解。[PCA 官方定义与参数](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

实施建议：目标 `[384, 256, 128, 64]` 均直接由原始 768 维压缩，绝不是先 768→384，再对结果 fit 384→256。可一次 fit 到最大 384 维，按 components 的前 d 行得到所有嵌套维度，记录该协议。最清楚的初版也可每个 d 使用 full solver 单独 fit。

研究设计陷阱（基于公式推导）：中心化改变向量夹角。即使 full-D PCA 的正交旋转本身保留点积，减均值仍使 cosine 改变。因此建议额外做 full-D centered 原始向量的 diagnostic，区分中心化和截断的影响；不要事后挑选最有利的 normalization 方式。主实验可约定所有方法压缩后 L2 normalize，另将原论文要求的不同流程作为单独复现轨道。

### Gaussian Random Projection

`GaussianRandomProjection(n_components=d, random_state=seed)` 使用元素服从 N(0, 1/d) 的矩阵。`fit` 只依赖输入形状生成投影，不学习语义；query 与 document 必须使用同一矩阵。设置指定 d，而非自动 Johnson-Lindenstrauss 下界选择维度；后者对小误差要求可能不给出想要的压缩规模。至少报告 3 个固定种子，条件允许 5 个，不能选最好种子。[GaussianRandomProjection 官方定义](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html)

## 3. 搜索与指标应当分开回答两个问题

主实验用 `faiss.IndexFlatIP(d)` 精确搜索，所有文档与 query 先 L2 normalize，因此最大 inner product 即 cosine。Faiss 官方也规定这个映射，并说明 L2 返回平方欧氏距离，不能将其直接当成 cosine 分数。[Faiss metric 说明](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)

为什么先 Flat：这样观察到的质量变化来自压缩；如果直接加 HNSW/IVF，则多了近似搜索误差。Flat 无训练参数，可作为其他索引的准确基准。[Faiss index 选择](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)

需要把三个不同概念分开命名：

1. `TaskRecall@k`：命中 qrels 中相关文档的比例，分母为该 query 相关文档数。
2. `NNOverlap@k`：压缩空间 top-k 与原始空间 top-k 的交集/k；它不使用 relevance 标签。
3. `ANNRecall@k`：近似 index 的 top-k 对同一压缩空间精确 top-k 的覆盖；仅扩展 ANN 实验使用。

BEIR 官方 evaluator 将 nDCG/Recall 等交给 pytrec_eval，MRR 则有独立实现。建议自行实现 harness 和所需指标，再用独立小样例/标准 evaluator 校验，符合老师要求，不直接复制整套评估流程。注意某些库默认删除 query/document 同 ID 的项；必须检查数据语义并固定行为，而不是无意删掉合法结果。[BEIR evaluation 源码](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)、[BEIR MRR 等源码](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/custom_metrics.py)

指标建议预注册：主指标 nDCG@10，次指标 Recall@10、Recall@100、MRR@10，几何指标 query→document NNOverlap@10（可加 document→document，并排除 self-match）。每个 query 保留结果与指标，才能做 paired bootstrap 的置信区间、排名改善/恶化案例、几何 overlap 与 relevance 改变散点分析。

## 4. 最小代码骨架（设计稿，未实测）

```python
import faiss
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


def unit_float32(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite embedding")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero-norm embedding")
    return np.ascontiguousarray(x / norms, dtype=np.float32)


def compress(method, fit_docs, corpus_docs, query_vecs, dim, seed):
    if method == "original":
        # dim 必须等于原始维度；这不是 PCA(full-D)
        assert dim == corpus_docs.shape[1]
        docs_z, queries_z, fitted = corpus_docs, query_vecs, None
    else:
        if method == "pca":
            fitted = PCA(n_components=dim, svd_solver="full", whiten=False)
        elif method == "grp":
            fitted = GaussianRandomProjection(n_components=dim, random_state=seed)
        else:
            raise ValueError(method)
        fitted.fit(fit_docs)
        docs_z = fitted.transform(corpus_docs)
        queries_z = fitted.transform(query_vecs)
    return unit_float32(docs_z), unit_float32(queries_z), fitted


def exact_search(docs_z, queries_z, top_k=100):
    index = faiss.IndexFlatIP(docs_z.shape[1])
    index.add(docs_z)
    scores, positions = index.search(queries_z, min(top_k, len(docs_z)))
    return index, scores, positions
```

这是标准 API 组合，不是从 RAE 仓库复制核心实现。返回的 positions 只是缓存行号，必须用 doc_ids 还原真实 ID；scores 越大越相关。整个 corpus 必须保留，不应只在已标注正负样本或 top-100 原始候选内搜索，否则测到的是 reranking，不能称为完整 dense retrieval。

## 5. 效率报告

Flat 的理论向量负载是每条 `4*d` bytes（float32），HNSW 额外存图，不保证减维后整个索引按 d 等比例缩小。[Faiss 索引存储表](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

因此，768→64 的向量负载是 12 倍缩减；这是算术，不是 12 倍端到端加速的实验结论。实际报告 `vectors.nbytes`、序列化 index 的文件字节、encoder/压缩器 checkpoint 大小，并单列暂存的 full-D 缓存，避免声称部署系统完全不占高维空间却将其保留在内存中。

计时建议：固定硬件与 faiss.omp_set_num_threads；先 warm-up；单 query 搜索记录 p50/p95，另报批处理 throughput；索引 add、压缩 fit、corpus transform、query transform、模型 query encoding 分开记录。端到端 query 时延至少包含文本编码+降维+检索，不能只报 search time 并声称整个 RAG 快了相同比例。三套数据很小，可验证质量和内存，但无法据此强推 million-scale production 速度收益。

可选 ANN 扩展固定 HNSW M、efConstruction、efSearch，先跑相同参数回答“同设置”问题，再做质量匹配的 Pareto 曲线回答“同准确率”问题；两者不能混为一个表。

## 6. 对后面三项的接口提醒

- 所有 neural compressor 的输入是冻结向量；不需要重复微调 backbone。
- Linear AE + MSE 在合适条件下学习 PCA 主子空间，但 latent basis 不必等于 PCA loading vectors，也不能推断两者 cosine ranking 一致。[Plaut 2018 原论文](https://arxiv.org/abs/1804.10253)
- 因此“standard AE”要明确 linear 或 nonlinear，架构及训练预算和 RAE/QRAE 匹配。若 RAE 论文使用线性结构，AE 可以作为去掉 regularizer 的消融，不能悄悄换成更弱架构。
- 额外读 [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)：它让训练后的表征支持多个前缀维度；与冻结已有模型之后学习 compressor 的范围不同。适合作 related work 或后续 stronger baseline，不应直接把任意 embedding 的前 64 维称为 MRL，也不宜为增加模型数量偏离主研究问题。
