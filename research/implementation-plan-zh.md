# COMP5331：从 proposal 到可复现实验的实施方案

调研日期：2026-10-26。依据本地 proposal、Raymond 评语、论文原文、公开参考代码、官方模型卡及库文档。用户负责前三种方法，主要使用 Mac，必要时可用云端 GPU。

这是研究与设计文件，不是实验报告：尚未下载模型、生成 embedding、训练或测出检索结果。下列代码是实施骨架，尚未做真实数据端到端验证。没有完整评分 rubric，关于“高分”的判断是对老师评语的响应，不能保证分数。

## 1. 结论与研究定位

六种方法均可实现。前三项不难在算法代码，而难在把数据、模型、query/document ID、归一化、索引和评估统一；它们应构成全组共用的实验基础，而不是三个孤立脚本。

建议主配置：**E5-base-v2 冻结 768 维向量；SciFact → NFCorpus → FiQA；压缩至 384/256/128/64；主质量实验使用完整 corpus 的精确 cosine 检索；三个随机种子。**

项目的研究贡献宜表述为：

> 在统一且测试 queries 独立的设置下，独立实现 RAE，研究低维压缩时相关性监督何时优于几何保持，并分析维度、负样本和监督量的影响。

不能声称“首个 query-aware / task-aware embedding compression”。[Matryoshka-Adaptor](https://aclanthology.org/2024.emnlp-main.576/) 已研究冻结向量的监督压缩；[LeanVec](https://arxiv.org/abs/2312.16335) 考虑 query/database 分布；[DIVE](https://arxiv.org/html/2605.20689v1) 也是直接相关工作。课程项目仍可通过可复现性、控制实验和分析提供有价值的贡献。

原 proposal 和 Raymond 评语是课程内部材料，保留在本地，不随公开仓库发布；本笔记保留与实现有关的研究要求。

## 2. 六种方法究竟做什么

| 方法 | 输入→输出 | 是否训练压缩器 | 需要 relevance labels | 实现重点 |
|---|---|---|---|---|
| Original | 冻结模型→768d | 否 | 否 | 正确生成并缓存向量，建立基准 |
| PCA | 768d→d | SVD 拟合 | 否 | 一个均值/投影同时作用于 corpus 和 query |
| Gaussian RP | 768d→d | 生成随机矩阵 | 否 | 同一矩阵，固定多个种子 |
| Standard AE | 768d→d→768d | reconstruction | 否 | 与 RAE 相同结构，去掉正则项 |
| RAE | 768d→d→768d | reconstruction + regularization | 否 | 先解决论文/代码细节差异 |
| QRAE | 768d→d→768d | reconstruction + regularization + ranking | 是，仅 train queries | 控制结构和监督数据，识别独立贡献 |

建议初版 AE/RAE/QRAE 都用相同的线性 encoder/decoder，不能只给 QRAE 更深的网络，再把差异解释为 loss 的效果。非线性 AE 可作为后续扩展。

共享静态 W_e 的 QRAE 在训练中使用 query supervision；它并不为每个新 query 动态生成压缩矩阵。documents 可以预先压缩并建立索引。这样符合你们的轻量后处理研究范围。

## 3. 数据集：下载什么，如何划分

| 数据集 | corpus 文档数 | train queries | dev queries | test queries |
|---|---:|---:|---:|---:|
| SciFact | 5,183 | 809 | 无，需从 train 内切 | 300 |
| NFCorpus | 3,633 | 2,590 | 324 | 323 |
| FiQA | 57,638 | 5,500 | 500 | 648 |

这些 query 数来自官方 qrels 去重统计；详细行数和固定 revision 见 [数据笔记](data-models.md)。官方来源：[BEIR](https://github.com/beir-cellar/beir)、[SciFact qrels](https://huggingface.co/datasets/BeIR/scifact-qrels/tree/main)、[NFCorpus qrels](https://huggingface.co/datasets/BeIR/nfcorpus-qrels/tree/main)、[FiQA qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels/tree/main)。

下载后需要三类东西：

- `corpus.jsonl`：doc ID、title、text。
- `queries.jsonl`：query ID、query text。
- `qrels/*.tsv`：query ID、doc ID、relevance score。

第一版使用官方已处理的 BEIR zip，不必自己抓网页、收集文本或标注相关性。文档不需要重新切 chunk；这是一个固定 corpus 的检索基准，随意切块会使原始 qrels 与文档单位失配。统一 title + 空格 + text，记录模型截断比例；chunking 若要研究，需另立协议。

最小下载示例，来源为 [BEIR 官方接口](https://github.com/beir-cellar/beir#beers-quick-example)：

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

name = "scifact"
url = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    f"thakur/BEIR/datasets/{name}.zip"
)
folder = util.download_and_unzip(url, "datasets")
loader = GenericDataLoader(data_folder=folder)
corpus, train_queries, train_qrels = loader.load(split="train")
_, test_queries, test_qrels = loader.load(split="test")
```

SciFact 从 809 个 train query IDs 中固定种子切约 80/20 的内部 train/dev，保存 ID 文件。同一 query 的所有正负样本必须在同一 split；不要随机切 q-d pairs。FiQA、NFCorpus 使用官方 dev。

### 建议主实验协议

全量目标 corpus 的**无标签向量**可供 PCA、AE、RAE 和 QRAE reconstruction 同池拟合；QRAE 的 ranking 只读取 train queries/qrels。dev 只用于选超参，test 只用于最终结果。这个设置应写作 **corpus-adaptive, in-domain supervised compression with held-out queries**。

目标 corpus 是已知检索库，使用它的无标签内容不自动构成测试标签泄漏；但这不同于从未见过目标 corpus 的严格 inductive 设置。使用目标域 train relevance labels 后，也不能称整个压缩实验是 zero-shot BEIR。冻结 backbone 与“压缩器无监督/零样本”是不同概念。

NFCorpus dev/test 含 1、2 级相关性，nDCG 保留原标签；Recall/MRR 可按 relevance > 0 二值化。未标注文档是 unknown，不是经过核实的 irrelevant。

## 4. 模型与 Mac 工作流

主模型建议 [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)：768 维，英文，最多 512 tokens，query 使用 `query: `，文档使用 `passage: `。SentenceTransformer 可按模型配置完成 pooling，输出后 L2 normalize。第二模型可选 BGE-base-en-v1.5，具体输入规则见数据笔记。

需要下载的是预训练 embedding 模型的权重、tokenizer 和配置，不是下载一个生成式聊天大模型。第一次 `SentenceTransformer(model_id)` 会从 Hugging Face 自动下载并缓存；后续重用本地缓存，不需要付费推理 API。[HF 下载文档](https://huggingface.co/docs/huggingface_hub/en/guides/download)、[SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)。

Mac 若支持 MPS 可用它生成 embedding，不支持则 CPU；FAISS/PCA 主要走 CPU。先用 batch=16 或32，实际测 200 篇文档后估算耗时，再决定 FiQA 是否迁云端。云端只需处理同配置编码并传回 `.npy`、IDs 和 manifest；压缩及 CPU 时延比较继续在同一台 Mac 上做。不要直接比较 Mac 与云端测出的 latency。

```python
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("intfloat/e5-base-v2", device=device)
model.eval()
model.max_seq_length = 512
for param in model.parameters():
    param.requires_grad_(False)

doc_ids = sorted(corpus)  # 保存同样顺序的 ID 文件
doc_texts = [
    "passage: " + (
        corpus[did].get("title", "") + " " + corpus[did]["text"]
    ).strip()
    for did in doc_ids
]
X = model.encode(
    doc_texts, batch_size=16, normalize_embeddings=True,
    convert_to_numpy=True, show_progress_bar=True,
).astype(np.float32)

qids = sorted(train_queries)
Q = model.encode(
    ["query: " + train_queries[qid] for qid in qids],
    batch_size=16, normalize_embeddings=True,
    convert_to_numpy=True,
).astype(np.float32)
```

正式代码固定 HF model revision，记录包版本、max length、前缀、pooling、dtype、硬件和文本拼接方式。每个 split 的 query 单独缓存。不是每个压缩维度重新调用一次 Transformer；全组六种方法应使用同一份原始缓存。

三套 corpus 共 66,454 条，768d float32 向量负载约 194.7 MiB（算术估计，不包含模型、token、索引副本和运行开销）。这说明缓存体积可控，不代表任意 Mac 的编码速度相同。

依赖建议：Python 3.11、torch、sentence-transformers、numpy、scikit-learn、faiss-cpu、beir、pandas、matplotlib。`pytrec_eval` 用于标准分数交叉验证；若 Mac 安装受阻，可在 Linux 单独验证评价结果。初期无需 LangChain、Milvus、Elasticsearch、vLLM 或 RAG 生成模块。环境跑通后固定实际版本，不在第一次试验就追逐所有最新包。

## 5. 你负责的前三项怎样写

### Original

使用缓存完整 768d 单位向量直接检索。这是 quality reference；监督式 QRAE 有可能超过它，因此不是理论性能上限。

### PCA

基本公式 `Z = (X - mean) @ V_d.T`。均值及 V_d 仅在约定的 corpus fit pool 上拟合，query 和 documents 共享。不要对 query 单独 fit，不要逐层累计降维。每个 d 都直接来自原始 768d。

可用 `sklearn.decomposition.PCA(n_components=d, svd_solver="full", whiten=False)`；其[官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)明确会中心化。考虑老师对独立实现的重视，也可独立用 NumPy SVD 实现十几行 PCA，再以 sklearn 校验投影子空间和检索分数。SVD 的符号选择可不同，不应直接逐元素比较 loading 矩阵。

### Gaussian Random Projection

生成 `R ∈ R^(768×d)`，各元素独立服从 `N(0,1/d)`，然后 `Z=X@R`。不需要 relevance labels，也不需要训练神经网络。query/doc 使用同一矩阵。至少跑3个预先固定的 seed，报告 mean/std，不挑最好 seed。[官方 GRP 定义](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html)。

独立实现骨架如下，**还未运行真实数据**：

```python
def fit_pca(X_fit, dim):
    mean = X_fit.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X_fit - mean, full_matrices=False)
    return mean, vt[:dim].T

mean, W = fit_pca(X, dim=128)
X_pca = (X - mean) @ W
Q_pca = (Q - mean) @ W

rng = np.random.default_rng(42)
R = rng.normal(0, 1 / np.sqrt(128), (768, 128)).astype(np.float32)
X_rp, Q_rp = X @ R, Q @ R
```

所有方法压缩后再统一 L2 normalize，保证主实验比较 cosine 检索。零范数、NaN、ID 错位应直接报错。

**有价值的诊断：PCA 中心化会改变 cosine。** 因此全维 PCA 不等于 Original。加一个 `centered-full-D` 对照，或加未中心化的 truncated-SVD 诊断，可区分“减均值”和“删维度”的效果。这是本次公式推导的设计建议，不是已证实解释了 RAE 的全部优势。

## 6. RAE：不能只看 README 就照搬

阅读来源：[arXiv 论文](https://arxiv.org/abs/2509.25839)、[公开参考仓库](https://github.com/explorerZH/RAE-KDD2026)。本次代码检查固定到 commit `138b00f19033121cec947be28a41c9a6bf0892e3`；它自称 reference implementation。本次未独立证实其维护者与论文作者身份关系，也未能获取 ACM 正式版全文。算法依据应以选定版本论文为主，代码只用于核实实现细节。

以下是对该 commit 源码的检查，并非运行结论：

| 位置 | 实际观察 | 实验需要怎样处理 |
|---|---|---|
| model.py | `nn.Linear` 未设 `bias=False`，默认带 bias | 别将它描述为严格无 bias 矩阵乘法；明确自己选择 |
| train.py | optimizer 接收全部 `model.parameters()` | 正则实际覆盖 encoder、decoder 和 bias，不只是注释声称的 encoder |
| train.py | 支持 Adam 和 AdamW | 两者的 weight decay 不可无条件当作相同 L2 objective |
| train.py | MSELoss 默认逐元素 mean | 与论文逐向量平方和的 scale 不同；数值 λ 不可直接搬 |
| config.py | weight_decay 默认0 | 默认命令实质没有该 regularization，需要明确设置非零值 |
| config.py / eval | 训练归一化默认关闭；cosine eval 使用归一化输入 | 主实验统一 train/eval 输入，原始默认复现另列 |
| train.py | steps 被换算成整数 epochs | 不一定严格执行指定步数；自己训练 loop 记录实际 update count |

对应源码：[model.py](https://github.com/explorerZH/RAE-KDD2026/blob/138b00f19033121cec947be28a41c9a6bf0892e3/code/model.py)、[train.py](https://github.com/explorerZH/RAE-KDD2026/blob/138b00f19033121cec947be28a41c9a6bf0892e3/code/train.py)、[config.py](https://github.com/explorerZH/RAE-KDD2026/blob/138b00f19033121cec947be28a41c9a6bf0892e3/code/config.py)。[PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) 与 [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) 官方算法明确区别 coupled L2 和 decoupled decay。

论文版本也需记录：本次 arXiv HTML/摘要、PDF header、仓库的 venue 标识不完全一致；Crossref DOI `10.1145/3770855.3817906` 登记为 KDD 2026。仓库 BibTeX 的 DOI 是占位符，不能复制进报告。以上只影响版本核实，不能据此断言实验不成立。

### 可复现方案

分开两个目标：

1. **RAE 算法复现**：独立写 encoder/decoder、loss、训练循环和近邻 overlap，记录和论文/参考代码的每个选择。
2. **原论文数字复现**：若选 IMDb 等原始数据，仍需相同 encoder checkpoint、预处理、抽样 ID、embedding。公开参考仓库要求自备预计算向量，未直接提供可完整重跑的同一批 embedding；缺少这些信息时只能诚实写“算法复现及趋势验证”，不能声称精确重现原表格。

BEIR qrels 的 nDCG 结果是你们新增的 relevance evaluation；不能在原论文未报告对应设置时声称复现了其 BEIR 检索数字。时间允许可加 IMDb-MPNet 一个原始域的几何验证；先把老师要求的复现差异记录交付出来，再扩大实验。

主控制实验建议先选择透明的 paper-style 线性、无 bias、显式正则实现，并说明与参考代码默认值不同；另用一个小 pilot 比较带 bias / 无 bias 和正则覆盖范围。如果最终决定以 reference-style 为主，也应统一 AE/RAE/QRAE 并写清实际设置。不能边看 test 结果边挑定义。

## 7. QRAE 最小实现与消融

建议明确自己的主公式，而不是留下泛化的 L_reg：

`z = W_e x`，`x_hat = W_d z`；decoder 接收未归一化的 z；ranking 使用 `u = normalize(z)`。

`L_rec = mean_over_examples(sum_over_dims((x_hat - x)^2))`

`L_reg = ||W_e||_F^2 + ||W_d||_F^2`

`L_rank = mean(max(0, margin - cosine(u_q,u_pos) + cosine(u_q,u_neg)))`

`L_total = L_rec + lambda * L_reg + alpha * L_rank`

这是建议采用的明确解释，并不声称所有 RAE 版本都使用完全相同定义。显式 L2 时 optimizer weight_decay 设0，避免重复正则；若使用隐式 decay，则写清等价系数和 optimizer。AE 设 lambda=alpha=0，RAE 设 alpha=0，QRAE 才加 train relevance。

```python
z = encoder(x_recon)                  # 同池无标签文档 reconstruction batch
recon = ((decoder(z) - x_recon) ** 2).sum(dim=1).mean()
reg = encoder.weight.square().sum() + decoder.weight.square().sum()
uq = torch.nn.functional.normalize(encoder(q), dim=1)
up = torch.nn.functional.normalize(encoder(pos), dim=1)
un = torch.nn.functional.normalize(encoder(neg), dim=1)
rank = torch.relu(margin - (uq * up).sum(1) + (uq * un).sum(1)).mean()
loss = recon + lam * reg + alpha * rank
```

rank batch 来自 train queries，reconstruction batch 从约定的统一 corpus pool 抽样。不要只用 QRAE triplets 的文档来 reconstruction，却让 RAE 看另一批文档。λ、α 的大小需要结合实际 reduction 与梯度规模在 dev 调，不应承诺某组数字必胜。

### Negatives 的公平比较

- random：完整 corpus 中均匀抽样，排除该训练 query 的已知 positives。
- in-batch：从同 batch 其他 query 的 positive documents 取候选，屏蔽重复文档和该 query 的已知 positives。
- hard：用冻结完整768d retriever 为 **train queries** 静态挖 top100/200，过滤已知 positives，再从预先约定的 rank 区间抽样。

初次比较每个 query 固定同样数量的 negatives、相同 loss、steps 和 batch size；否则难度、数量和计算量混淆。若改用 B−1 negatives 的 InfoNCE，应另标为 loss/negative-count 联合变化，或增加数量匹配控制。[DPR](https://aclanthology.org/2020.emnlp-main.550/) 是这部分的基础阅读。

hard 候选不一定真负：不要直接取最相似未标注文档便断言 irrelevant；记录 false-negative 风险，不用 test qrels 清洗训练样本。先做静态 mining，动态 mining 留作扩展。

### 最有解释力的额外 controls

| Control | 回答的问题 |
|---|---|
| Rank-only，只有相同 W_e 的监督 ranking | reconstruction/RAE 正则是否真的贡献？ |
| 768d supervised projection/adapter | 提升是否主要来自领域标签，而非降维设计？ |
| QRAE 10%/25%/100% train-query labels | 低数据时会不会过拟合？ |
| 同维度、同结构、同训练预算比较 | 能否排除维度或模型容量造成的差异？ |

不必给所有数据、维度铺满每个消融。优先 SciFact/FiQA 的64d、128d，NFCorpus 验证趋势；具体数量依 pilot 耗时决定。

## 8. 统一搜索和正确指标

所有 query/document 压缩后 L2 normalize，再使用 [FAISS IndexFlatIP](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) 对完整 corpus 精确搜索。FAISS 只负责索引和搜索，你们自行写 ID 映射、实验编排及评分流程。

```python
import faiss

X_z = np.ascontiguousarray(X_z, dtype=np.float32)
Q_z = np.ascontiguousarray(Q_z, dtype=np.float32)
faiss.normalize_L2(X_z)
faiss.normalize_L2(Q_z)
index = faiss.IndexFlatIP(X_z.shape[1])
index.add(X_z)
scores, rows = index.search(Q_z, min(100, len(X_z)))
# rows 是缓存行号；用保存的 doc_ids 还原，不能直接当 doc ID。
```

注意：仅重排 Original 的 top100 会把任务变成候选重排，无法评价压缩后从库中独立召回的能力。主表先不加 ANN；HNSW 另做效率扩展，避免引入额外近似误差。

必须区分：

- **TaskRecall@k**：找到的相关文档数 / 该 query 的相关文档总数。
- **NNOverlap@k**：低维 top-k 和原维 top-k 的交集/k，不读取相关性标签。
- **ANNRecall@k**：近似索引和同一维度精确索引的重叠，仅 ANN 实验使用。

主指标 nDCG@10；辅指标 Recall@10、Recall@100、MRR@10；几何指标 query→doc NNOverlap@10，再加 doc→doc overlap（排除自身）以连接 RAE 原目标。

若自行实现 nDCG 并与 BEIR/pytrec_eval 对齐，注意 **trec_eval ndcg_cut 使用 qrels relevance 本身作为 gain**，不是自动 `2^rel-1`。NFCorpus 有 graded labels 时两者会产生差别。[NIST trec_eval 源码](https://github.com/usnistgov/trec_eval/blob/master/m_ndcg_cut.c)、[BEIR evaluator](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)。同时固定 tie-breaking、同 ID 是否排除、空结果 query 如何计入，不能把没有命中的 queries 丢弃。

绘图建议：

1. 三个数据集的 dimension–nDCG 曲线，Original 为水平参考线。
2. 同维度内 NNOverlap–nDCG 散点/方法对比；不要只把不同维度混一起算相关性。
3. memory–quality Pareto 图，以及实际 CPU p50/p95 search latency。
4. 负样本/监督量消融；并展示3–5个改善和恶化案例，不只选成功例。

对 stochastic methods 至少3个固定 seeds；同一测试 query 上保存逐query分数，报告 mean/std 和 paired bootstrap 差异区间。Query bootstrap 与 seed variance 是不同不确定性，不能把所有 seed×query 当完全独立样本。

## 9. 效率与“实时”的准确含义

离线：下载数据 → 生成 corpus embedding → fit/train compressor → 压缩全部 documents → 建 index。

在线：新 query → 同一冻结模型 → 同一个 compressor → normalize → search → 返回文档。

在线推理只需 encoder，不需要 AE decoder。若问题中的“实时”指实际实现，以上也是落地流程；若指 real-time latency，则必须实测。

768→64 的 float32 向量 payload 缩小12倍，只是 `4×dimension` 的算术比例，不能声称整个系统快12倍。报告至少分开 query text encoding、projection、search；存储分开向量、index、压缩器参数，以及研究阶段保留的完整缓存。[FAISS 存储说明](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)。

固定同一硬件、线程、batch size、warmup 和计时方法；主表可用 CPU 单query p50/p95，另报批量 throughput。GPU/MPS 计时要同步。小 corpus 的速度曲线易受开销和缓存影响，只能支持当前规模结论，不能外推百万级速度。

如加 HNSW，先以相同 M/efConstruction/efSearch 比较，再额外画 matched-quality 曲线；明确两种比较回答的问题不同。不要给 QRAE 更高 efSearch 而让其他方法用默认参数。

## 10. Raymond 评语对应的交付物

| 老师要求 | 可审查的具体交付 |
|---|---|
| 固定数据、模型、指标 | 一份 experiment.yaml + 数据/model manifest |
| clean RAE reproduction | 独立实现、版本/公式对照、几何结果、无法精确复现的说明 |
| ranking loss 精确定义 | 数学公式、采样算法、固定 positive/negative IDs、loss reduction |
| pilot 后再全量 | 128d 的 loss/validation 曲线，训练稳定性检查 |
| 多压缩比、失败分析 | 主曲线、逐query结果、成功/失败案例 |
| 所有 baseline 相同 index/search | 共用的 search/evaluate 接口和配置 |
| 独立实现核心组件 | 独立 compressor、training loop、mining、evaluation harness；依赖来源表 |

老师允许用标准网络层、优化器、nearest-neighbor index；对 PCA/RP 也可选择自己用 NumPy 实现并用 sklearn 对照，避免把任务变成搬运别人完整工程。标准数据下载器和 metric checker 可作为资源，但 end-to-end pipeline 由你们编排和验证。公开 RAE 项目只用来澄清细节，不直接复制核心训练工程提交。

## 11. 实施顺序与阶段验收

### 阶段 A：先完成你的三个 baseline 和全组接口

1. 数据下载、ID/分割 manifest；只用小样本验证 loader。
2. E5 编码与缓存：先200文档检查形状、norm、时间，再完整 SciFact。
3. Original 完整检索与 qrels 指标；用手工小例子和标准 evaluator 交叉验证。
4. PCA / RP 128d pilot，确认同一变换同时作用 query/doc。
5. 扩展四个维度、三个数据集、RP 三种子，保存 runfiles/per-query metrics。

验收条件：原始与压缩结果可一键复跑；维度、归一化、ID 对齐、qrels 权限和完整 corpus 都可检查。可做 identity-transform 结果一致、cosine 与归一化 IP 排名一致等有意义验证。

### 阶段 B：组员接入后三项

1. AE 与 RAE 同结构训练，检查 reconstruction/regularizer 曲线。
2. 完成 RAE 几何验证和论文/代码差异记录。
3. QRAE128d + random negative，先验证 train ranking 能学到、dev 不崩。
4. dev 上小规模 λ/α/margin 搜索，定协议后才做 test。
5. 完整主表后补 hard negatives、Rank-only 和监督量控制。

### 阶段 C：解释研究问题

验证是否存在低 NNOverlap 但高 nDCG 的例子；也接受任务监督无益、只在部分维度有益的结论。分析压缩失败导致的 relevant-document rank 变化。图表/统计和复现材料的优先级高于额外制作漂亮的 RAG demo。

建议代码结构（尚未创建项目实现）：

```text
configs/experiment.yaml
src/data.py                  # 下载/读数据/query划分/ID manifest
src/embed.py                 # 冻结模型、缓存与revision
src/compressors/base.py      # fit / transform / save / load
src/compressors/pca.py
src/compressors/random_projection.py
src/compressors/autoencoder.py
src/training.py              # AE/RAE/QRAE共同训练逻辑
src/negatives.py             # 仅train query mining
src/search.py                # 所有方法共用FAISS入口
src/metrics.py               # 独立指标+标准实现校验
src/run_experiment.py
src/analyze.py
results/                     # 配置、runfile、per-query分数和时间
```

## 12. 阅读清单：优先级按本项目价值排序

| 次序 | 阅读内容 | 要带走的问题 |
|---|---|---|
| 1 | [RAE](https://arxiv.org/abs/2509.25839) §3/§4 + 上述固定commit代码 | 模型/正则/指标/训练协议到底是什么？ |
| 2 | [BEIR](https://arxiv.org/abs/2104.08663) + 官方数据/评估实现 | 什么是 query、corpus、qrels 和公平检索比较？ |
| 3 | [E5](https://arxiv.org/abs/2212.03533) + 模型卡 | 正确的检索向量怎样生成？ |
| 4 | [DPR](https://aclanthology.org/2020.emnlp-main.550/) §3.2 / 负样本实验 | positives、in-batch 和 hard negatives 如何进入训练？ |
| 5 | [Matryoshka-Adaptor](https://aclanthology.org/2024.emnlp-main.576/) §3/§4 | 已有监督压缩怎样做，QRAE的贡献边界在哪里？ |
| 6 | [Search-Adaptor](https://aclanthology.org/2024.acl-long.661/) | 冻结向量适配和 held-out queries 怎样处理？ |
| 7 | [DIVE](https://arxiv.org/html/2605.20689v1) §3/§4.1 | triplet + 多头与本项目有什么区别，评估protocol是否一致？ |
| 扩展 | [SMEC](https://aclanthology.org/2025.emnlp-main.1332/)、[MRL](https://arxiv.org/abs/2205.13147)、[LeanVec](https://arxiv.org/abs/2312.16335) | 多维训练、query分布、相关方法边界 |

PQ/HNSW 先读概念即可；初版不必复现它们，也不用下载 DPR/MRL/DIVE 的大模型。你负责前三项，先读 BEIR、E5、PCA/RP 官方文档，再理解 RAE/QRAE 接口，就可以开始编码。

DIVE §4.1 明确用全部可用 relevance judgments 训练并在同一 query set 评估；这是与本方案 held-out queries 不同的设置。其高分不能当作你们无泄漏测试结果的预期目标。相关工作比较要同时比较设置，而不是只抄数值。

## 13. 补充：理论表述应谨慎

这是本次独立数学检查：当 W 是 d×D、d<D 的降维矩阵时，存在非零 nullspace 向量。即使所有非零奇异值相同，也不能对整个原空间声称严格正的 norm 下界。例如 W=[1,0] 的非零奇异值 condition number 为1，却将(0,1)映成0。因此报告可把谱诊断作为经验分析，不应直接把全局近邻保持保证当成你们自己的已证明结论；需要额外子空间/数据分布假设。

另一个独立推导：只正则 encoder 而 decoder 不受约束，会有 encoder 缩小、decoder 放大的缩放自由度。因此应说清正则覆盖范围，监控两者范数，不盲从注释。这些检查不否定 RAE 的经验效果，目的是让实现和论证可核查。

## 14. 最终应交付什么

共享向量缓存与manifest；可复现六法运行入口；明确训练/验证/测试权限；统一检索/指标实现；三数据×四维度主结果；随机性与置信区间；关键消融；效率曲线；成功和失败案例；版本和依赖说明；诚实的相关工作与局限。

当前完成的是上述研究计划及三份专题笔记。尚无真实性能数字；后续先完成阶段A，再根据pilot确定阶段B/C的实验预算。
