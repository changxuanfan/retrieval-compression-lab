# BEIR 数据、冻结 embedding 模型与评估协议研究

核查日期：2026-10-26。以下区分“已核实事实”和“本项目建议”。未下载模型、未运行训练或检索实验；仅读取官方文档、源代码、Hub 元数据与小型 qrels TSV。

## 1. 已核实的数据规模和 split

| 数据集 | 全量检索 corpus | train queries / qrels 行数 | dev queries / qrels 行数 | test queries / qrels 行数 |
|---|---:|---:|---:|---:|
| SciFact | 5,183 | 809 / 919 | 无官方 dev | 300 / 339 |
| NFCorpus | 3,633 | 2,590 / 110,575 | 324 / 11,385 | 323 / 12,334 |
| FiQA | 57,638 | 5,500 / 14,166 | 500 / 1,238 | 648 / 1,706 |

文档数来自官方 BeIR Hub dataset_info；query 数是从官方各 split TSV 的 `query-id` 去重计算，qrels 行数为实际行数。表中 train/dev/test 分的是 query，不是把检索 corpus 分成三个子库。官方格式为 corpus.jsonl、queries.jsonl、qrels/{split}.tsv。不要误把 919 条 SciFact qrels 当成 919 个训练 query。

来源：[BEIR README 数据表与下载](https://github.com/beir-cellar/beir#beers-available-datasets)、[SciFact](https://huggingface.co/datasets/BeIR/scifact)、[NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus)、[FiQA](https://huggingface.co/datasets/BeIR/fiqa)；[SciFact qrels 文件](https://huggingface.co/datasets/BeIR/scifact-qrels/tree/main)、[NFCorpus qrels 文件](https://huggingface.co/datasets/BeIR/nfcorpus-qrels/tree/main)、[FiQA qrels 文件](https://huggingface.co/datasets/BeIR/fiqa-qrels/tree/main)。可交叉验证：[ir_datasets 官方 BEIR 文档](https://ir-datasets.com/beir.html)。

**标签细节**：实际读取的 NFCorpus train qrels 只有 score=1，dev/test 有 score=1、2；SciFact/FiQA 各 split 只有 score=1。nDCG 应保留官方分级标签；二元 Recall、MRR 使用 score>0。未标注文件不是已验证的负例，hard negatives 应称“候选负例”，至少排除该训练 query 已知正例，并报告 false-negative 风险。

可复现版本（本次 HF API 核查）：

| Hub repo | commit revision |
|---|---|
| BeIR/scifact | b3b5335604bf5ee3c4447671af975ea25143d4f5 |
| BeIR/nfcorpus | b5026a0e96e8a7ac4f95f482a596389289d46269 |
| BeIR/fiqa | 979c07a7cb5ccc6ca009792241fa1250b98055dd |
| BeIR/scifact-qrels | 2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029 |
| BeIR/nfcorpus-qrels | a451b3b26d3ae1358f259c1a3a4dd61fcea35a65 |
| BeIR/fiqa-qrels | 252958f2d646e22cab6d0c72dd3f0d5de6d0655a |

上述六个 repo 的完整 revision 来自 `https://huggingface.co/api/datasets/{repo}` 的 sha 字段。正式跑前保存数据 manifest 与文件 SHA256；不要依赖搜索引擎摘要中的表格拼接。

## 2. 最省心的数据下载路线

建议第一版用 BEIR 官方 zip 与 GenericDataLoader，避免不必要地处理多个 HF config；它不是把别人的项目当自己代码，而是标准公开数据的下载和解析。BEIR 官方代码示例：

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader

name = 'scifact'  # 后续改为 nfcorpus / fiqa
url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip'
data_path = util.download_and_unzip(url, 'datasets')
corpus, train_queries, train_qrels = GenericDataLoader(data_folder=data_path).load(split='train')
_, test_queries, test_qrels = GenericDataLoader(data_folder=data_path).load(split='test')
```

SciFact 不可直接 `.load(split='dev')`；要在 809 个 train query ID 内固定随机种子切 train/dev（建议约80/20，保存 ID）。FiQA/NFCorpus 用官方 dev。Loader 只返回该 split qrels 中的 query。来源：[官方 quick example](https://github.com/beir-cellar/beir#beers-quick-example)、[GenericDataLoader 源码](https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py)。

若想使用 Hub，corpus 与 queries 在 `BeIR/{name}`，qrels 在另一个 repo `BeIR/{name}-qrels`，不可只下载 corpus 就开始评估。公开模型/数据由 Python 库自动下载缓存，通常不需要付费推理 API；Hub 下载支持 `revision` 固定 commit。[HF 官方下载文档](https://huggingface.co/docs/huggingface_hub/en/guides/download)。

## 3. 冻结 embedding 模型选择

建议主实验先固定 `intfloat/e5-base-v2`，第二模型验证可用 `BAAI/bge-base-en-v1.5`；MPNet 可作兼容性备选，不要第一轮三个模型全跑。

| 模型 | 输出 | 正确输入及 pooling |
|---|---|---|
| intfloat/e5-base-v2 | 768 维；最多512 tokens；英文 | query 加 `query: `；document 加 `passage: `；mask-aware average pooling，L2 normalize |
| BAAI/bge-base-en-v1.5 | 768 维；512 tokens | 检索 query 加 `Represent this sentence for searching relevant passages: `，document 不加；原生 Transformers 用 CLS pooling，L2 normalize |
| sentence-transformers/all-mpnet-base-v2 | 768 维 | 不套 E5/BGE 前缀；官方 Transformers 示例是 mask-aware mean pooling + L2 normalize；使用模型自带最大序列长度设置 |

模型事实来源：[E5 官方模型卡](https://huggingface.co/intfloat/e5-base-v2)、[BGE 官方模型卡](https://huggingface.co/BAAI/bge-base-en-v1.5)、[MPNet 官方模型卡](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)。前缀只加一次，不可既手动加又通过 prompt 配置重复加。优先 SentenceTransformer 自动使用模型规定的 pooling，减少手写 pooling 错误。

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('intfloat/e5-base-v2')
model.eval()
model.max_seq_length = 512
for p in model.parameters():
    p.requires_grad_(False)

doc_ids = sorted(corpus)  # IDs 全部保留为 str
texts = [('passage: ' + (corpus[i].get('title', '') + ' ' + corpus[i]['text']).strip())
         for i in doc_ids]
D = model.encode(texts, batch_size=32, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
qids = sorted(train_queries)
Q = model.encode(['query: ' + train_queries[i] for i in qids], batch_size=32,
                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
np.save('doc_embeddings.npy', D)
# 同时必须保存 doc_ids、qids，以及模型revision、前缀、max_length、库版本。
```

以上是演示骨架，未在本次研究运行；正式代码加入目录、revision、分块缓存和完整性验证。SentenceTransformer 可接受 device='cuda'/'cpu'/'mps'、revision 与 cache_folder，按模型名自动下载。[官方类文档](https://sbert.net/docs/package_reference/sentence_transformer/model.html)。不需要 LangChain、向量数据库或 RAG 大框架。

E5 论文：[Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533)；[作者官方 e5 仓库](https://github.com/microsoft/unilm/tree/master/e5)。主项目是研究压缩方法；模型不微调，所有方法共享同一批缓存 embedding，避免比较模型本身。

## 4. 本项目建议的最小公平协议

以下属于依据任务推导的实验设计建议，不是官方 BEIR 强制规则。

1. 主设置明确命名 **in-domain supervised embedding compression**：QRAE 仅使用目标数据集 train query 的相关性监督，dev 选超参，test 一次最终评估。这不能再宣称严格 zero-shot BEIR 结果。
2. 压缩器可在全量未标注 corpus embedding 上训练：这是 collection-aware / transductive compression，要在报告里写明。PCA、AE、RAE 的无监督样本池一致，均不使用 dev/test query 或 qrels；QRAE 在同一 corpus 池上加 train-query 监督。若要严格 inductive 结果，单独设定 source corpus、source queries 训练，目标域完整不接触并另报，不能混表。
3. SciFact 内部 dev 仅从 train query IDs 切出；保证相同 query 的全部 positive/negative pairs 同属一个 split。不能把同一 query 的 q-d pair 随机切分，否则泄漏。
4. 正负采样只读取 train qrels。hard negative 用原始冻结 embedding 的 train query top100/200 中采样，过滤该 query 所有已知 train positive；不能利用 test qrels“清洗”训练负例。in-batch negatives 同时屏蔽同 query 多正例和重复正例，记录采样种子与具体 IDs。
5. 原维 768；压缩 384/256/128/64（必要加32作压力测试）。每个 target dimension 是直接768->d，不是768->384->256累计降维。
6. query 和 document 必须经过同一个已 fit 的压缩变换；PCA 均值只拟合一次，不能 query 单独 fit；RP 矩阵同样共享。压缩后所有方法统一 L2 normalize，再用 cosine 等价的内积搜全文档库。
7. 主质量表用 exact full-corpus search，防止压缩与 ANN 搜索误差混淆。以后单独加 HNSW 延迟实验，固定参数或展示 matched-recall 曲线。
8. 主指标 nDCG@10；辅指标 Recall@10/100、MRR@10（明确 cutoff），保存 per-query 指标。`Recall@10` 是相关文档召回率，不是 Hit@10。NFCorpus 多正例时差异明显。
9. 几何指标分清 query-document overlap@k 和 document-document kNN overlap@k；后者去掉自己。两者都对照原维 top-k，无须 qrels。原维邻居不是相关性真值；二者不能混叫 retrieval recall。
10. RP、AE、RAE、QRAE 至少3 seeds；报告均值/标准差；方法间 per-query paired bootstrap 给差异区间。PCA deterministic 不用伪造三次独立结果。
11. 原始 embedding 搜索质量是 reference，不是理论 upper bound；QRAE 用相关性监督，有可能超过原始向量。报告相对质量保持率外也显示绝对指标和提升。

## 5. 评估实现应参考、不能整套抄用

BEIR 官方 `EvaluateRetrieval.evaluate` 用 pytrec_eval 计算 nDCG、MAP、Recall、Precision；MRR 在 evaluate_custom 中。返回顺序需核实，不要把第二个 dict 误当 Recall。[evaluation.py](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)、[custom_metrics.py](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/custom_metrics.py)。

Raymond 要求自行实现 end-to-end evaluation harness。可独立写数据适配、embedding缓存、压缩、search、结果映射、指标汇总，把标准指标库用于评分和交叉验证，并注明来源。不要复制官方整段 pipeline 再声称独立实现。

实现时最容易出错的检查：embedding 行序与 doc_id 对齐、query/doc 前缀与池化、零向量归一化、所有 query 均被纳入平均（无命中不能静默丢弃）、top-k 不超过 corpus、相同 doc_id 重复、float32 连续矩阵、计时排除下载和模型加载并明确包括哪些步骤。报告 query encoding/projection/search 三段时间；压缩只加速 projection后搜索，不会自动缩短冻结 Transformer 的编码时间。
