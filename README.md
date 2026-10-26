# COMP5331 — Frozen Embedding Compression Baselines

本仓库实现 **Original、PCA、Gaussian Random Projection、Standard Autoencoder** 四种方法，研究冻结检索 embedding 降维后，几何近邻保持、相关文档检索质量和存储/搜索成本如何变化。

**RAE、QRAE 目前只有研究与接入规划，没有实现，也不出现在可运行方法中。** 传入这两个名字会报错，不会用普通 AE 冒充。

本 README 是当前实现的依据。`research/` 保存前期六方法调研，其中部分是历史建议；与当前范围冲突时以本文件和实际配置为准。代码骨架已落地，真实运行状态记录在 [实验日志](docs/experiment-log.md)。

协作入口：[CONTRIBUTING.md](CONTRIBUTING.md)；RAE/QRAE下一阶段任务见[路线图](docs/roadmap-rae-qrae.md)，GitHub issue模板已准备。原始代码/原创文档采用[MIT许可证](LICENSE)，数据、模型及数据摘录保留各自条款，见[第三方说明](THIRD_PARTY.md)。

## 1. 研究问题与边界

我们关心三个问题：

1. 压缩后保留多少原始近邻？
2. 原始近邻的保持是否对应真实 relevance retrieval 的保持？
3. 降维带来的存储减少和实际搜索成本有什么关系？

这个阶段没有微调 E5，没有训练新的文本 encoder，也没有用 relevance labels 训练压缩器。前三项加 AE 均是 **冻结 embedding 后的无监督后处理**。目标数据集完整 corpus 可供拟合，这是 **corpus-adaptive / transductive compression**；它不是“完全未见目标库”的严格 inductive 设置。

评估 queries 按 train/dev/test 分离，默认先跑 dev。后续 QRAE 若使用目标 train relevance labels，需要另标为 in-domain supervised compression，不能把两种设置混成 zero-shot 结论。Original 是参照，不是性能理论上限。

### 已完成的本机验证（2026-10-26）

23项测试通过（含新增3项审计检查）；真实SciFact pilot与完整dev配置共完成四法运行。完整dev sweep为29 runs：四个压缩维度、RP/AE各三个种子、AE3,000步。所有run的nDCG@10/100经独立sklearn、完整qrels交叉核对；数据/模型/source/环境信息均保留。

下表是 **SciFact内部dev、完整corpus、nDCG@10**；Original768d为0.75201。随机方法为3 seeds mean±sample std，不是test结果或置信区间。

| Method | 64d | 128d | 256d | 384d |
|---|---:|---:|---:|---:|
| PCA | 0.59230 | 0.67457 | 0.71950 | 0.72392 |
| Gaussian RP | 0.32941 ± 0.01675 | 0.51161 ± 0.01955 | 0.63982 ± 0.01900 | 0.68552 ± 0.01753 |
| Linear AE | 0.59166 ± 0.00370 | 0.68838 ± 0.00901 | 0.71648 ± 0.00312 | 0.71501 ± 0.00539 |

这个dev结果没有显示AE在所有维度优于PCA；不能从单个均值差推断显著性。完整表、图、逐query分数及provenance见 [SciFact dev结果](docs/results/scifact-dev/report.md)。NFCorpus/FiQA目前仅完成数据下载/解析，test尚未运行。

## 2. 快速开始

在本仓库根目录执行。推荐使用 [uv](https://docs.astral.sh/uv/) 管理 Python 与依赖；`uv.lock` 固定实际解析版本。

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run retrieval-lab demo
```

`demo` 不联网、不下载模型，生成低秩合成向量，实际运行四种方法、FAISS、指标与报告导出。**它只验证程序管线，不是 BEIR 实验，不得写成报告中的真实性能。** 路径打印在命令末尾。

首个真实实验：

```bash
uv run retrieval-lab pipeline --config configs/scifact-pilot.toml
```

该命令依次下载官方 SciFact、生成冻结 E5 缓存、运行四种方法、导出逐 query 分数和图表。它使用 **完整5,183篇文档、内部162个dev queries、128维、一个随机种子、AE1,000步**，因此称为 pilot，不是最终多种子主表。

首次编码需要联网下载模型；以后复用缓存。网络或模型下载失败时保留已下载的数据，并通过分步命令重试。编码中断会重算该未完成缓存，**没有声称支持按 token/batch 精确续算**。

## 3. 当前机器与计算安排

开发机器：Apple M5 Pro，24GB统一内存。Python使用独立 `.venv`，不修改系统 Python。

- embedding：`device="auto"` 按 CUDA → MPS → CPU 选择；默认 batch16、max512 tokens。
- PCA、RP、FAISS：CPU，默认4线程；限制 BLAS 与 Torch CPU 线程，减少不同实验之间的线程差异。
- AE训练：默认CPU。网络很小，便于复现和与MPS性能差异解耦；如改用MPS，修改配置并产生新的study。
- 推理：所有压缩器统一导出NumPy float32变换，在CPU变换 query/document。

已实测当前macOS wheels在同进程混用PyTorch/FAISS会触发不同OpenMP运行时冲突。`pipeline`因此按prepare/embedding/evaluation分进程，AE训练也通过独立Torch worker完成，再把数值权重交回FAISS进程；新query编码同样隔离。没有使用`KMP_DUPLICATE_LIB_OK`掩盖冲突。AE的fit wall time包含worker启动/数据传递，`training_worker_seconds`另记录worker内部训练耗时。

三套完整 corpus 共66,454文档，其768d float32向量负载约194.7MiB；这不包含模型、tokenization、临时矩阵与索引副本。先实测编码速度，再决定是否迁云端。**不要将不同机器的latency混成方法速度对比。**

## 4. 数据集、来源与划分

| 数据集 | Corpus | 官方train queries | 官方dev | 官方test |
|---|---:|---:|---:|---:|
| SciFact | 5,183 | 809 | 无 | 300 |
| NFCorpus | 3,633 | 2,590 | 324 | 323 |
| FiQA | 57,638 | 5,500 | 500 | 648 |

数据来自 [BEIR官方仓库](https://github.com/beir-cellar/beir) 给出的 UKP archive。下载器只处理支持的三套数据，独立读取 `corpus.jsonl`、`queries.jsonl`、`qrels/*.tsv`，并拒绝zip路径穿越和symlink。没有复制或调用BEIR完整检索工程。

SciFact按query ID固定种子2026、20%从官方train内切dev：**647 train / 162 dev / 300 test**。同一query的全部judgments保持在同一split，不按q-d pairs随机切分。NFCorpus/FiQA使用官方dev。加载器检查query跨split重复、缺失文档/查询和矛盾标注。

下载与原始格式：

```bash
uv run retrieval-lab prepare --config configs/scifact-pilot.toml
```

输出 `data/scifact/manifest.json` 保存来源URL、archive及各数据文件SHA256；`query_splits.json` 保存实际分割IDs。HF镜像可以交叉核查：[SciFact](https://huggingface.co/datasets/BeIR/scifact)、[NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus)、[FiQA](https://huggingface.co/datasets/BeIR/fiqa)。各数据许可按原始数据页与BEIR说明记录，不将它们作为本仓库自有数据再发布。

**不会只保留有标注文档。检索始终搜索完整corpus。** title与text按空格拼接，不另切chunk，以保持qrels的文档单位不变。NFCorpus dev/test的1、2级标签保留，未标注项在标准离线指标中按0计算，但不因此断言真实不相关。

## 5. 冻结 embedding 与缓存

主模型：[intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)，输出768维，英文，512-token上限。

- query文本：`query: ` + query。
- document文本：`passage: ` + `(title + " " + text).strip()`。
- pooling按模型的SentenceTransformer配置，不自行替换为CLS。
- 模型eval，参数requires_grad=False；输出L2 normalize，存float32。
- 输入超过512 tokens按模型tokenizer截断；本版尚未统计截断比例，这属于报告的限制。

```bash
uv run retrieval-lab embed --config configs/scifact-pilot.toml
```

首次用Hugging Face API将配置中的revision（默认main）解析到完整commit SHA，再按该SHA加载模型；`trust_remote_code=False`。同一请求后续复用已记录SHA，不因远端main改变而静默更新。要明确更新模型，填写新的revision并生成新缓存。

缓存位于 `cache/embeddings/<dataset>/<fingerprint>/`：

```text
documents.npy / doc_ids.json
queries.npy   / query_ids.json
manifest.json
```

数组行与ID文件严格对齐。缓存包含所有有qrels的query，以避免后续重复编码；**压缩器fit只收到documents矩阵**，不收到query文本、query vectors或qrels。预先编码test文本不等于用test labels训练，但也不应查看test表现调参。

manifest记录模型SHA、完整前缀和截断配置、实际device、包版本、数据manifest、数组/ID文件checksum及编码总耗时。未完成缓存使用 `.part.npy`，完成后才写manifest/pointer。读取缓存校验checksum，避免用错或损坏向量。

## 6. 四种方法的精确定义

令原始单位向量为x∈R^D，D=768；每个目标d都**直接从D压缩**，不是连续768→384→256累计压缩。所有方法压缩后再次L2 normalize，再进行cosine检索。

### 6.1 Original

Identity transform，保留完整768维；没有训练。独立运行一次，而不是对多个seed重复制造误差条。全维PCA不能代替Original，因为PCA减均值会改变cosine。

### 6.2 PCA：独立NumPy SVD

在完整未标注corpus拟合均值μ与中心化矩阵：

```text
X_c = X - μ
X_c = U S V^T
z = (x - μ) V_d
```

使用float64进行SVD拟合，部署均值/投影存float32；不whitening，不逐维标准化。query/document共用μ、V_d。实现没有调用sklearn PCA，测试以sklearn为独立数值对照；loading符号可能不同，所以比较投影后的Gram矩阵，而不要求每个特征向量符号一致。

PCA会改变cosine的原因同时包括中心化和截断，不能直接把所有损失归因于维度减少。未来可以加centered-full-D与uncentered-SVD诊断；它们尚未作为额外方法实现。

### 6.3 Gaussian Random Projection：独立随机矩阵

```text
R ∈ R^(d×D), R_ij independently ~ Normal(0, 1/d)
z = x R^T
```

使用NumPy独立RNG，不fit语义、不中心化。query/document共用一个R，种子固定为17/42/2026，报告全部种子mean/std，不挑最好的一次。JL理论不保证在任意指定低维下必然保持所有相关性排序；质量需测。

### 6.4 Standard Autoencoder：中心化线性AE

**当前标准AE是centered linear AE，不是非线性深层AE，也不是RAE。**

```text
x_c = x - μ
z = W_e x_c
x_hat = W_d z + μ
L_rec = (1/B) Σ_i ||x_hat_i - x_i||²
```

μ仅由同一corpus计算；encoder D→d、decoder d→D，无bias、无activation、untied weights。参数用PyTorch默认Linear初始化，固定seed。decoder接收原始latent，**不在reconstruction前normalize latent**。检索时才对z归一化。

默认训练：Adam，lr1e-3，weight_decay=0，batch128，3,000个实际optimizer updates；每步cosine退火到1e-5。每步用私有CPU RNG从完整corpus均匀有放回采样。无relevance supervision、无regularization、无contrastive loss、无query-based early stopping。

训练记录第一步、每100步、末步的batch loss与最多前1,024个corpus vectors上的probe reconstruction。probe不是held-out validation，不能将其写作泛化分数。保存固定最后一步，未实现精确中途续训；MPS/CPU/CUDA之间也不承诺逐bit一致。

中心化线性AE的最优重建子空间与PCA有联系，但latent基底未被强制正交，因此不能推断两者cosine检索完全相同；有限步数也未必达到最优。这个联系用于分析，不作为AE正确性或必胜的假设。[Plaut 2018](https://arxiv.org/abs/1804.10253)。

## 7. 共享检索、排序与指标

四法使用相同的完整corpus、query集合、float32和 `FAISS IndexFlatIP`。向量归一化后inner product即cosine；**没有ANN近似误差，也不是只重排Original的top100候选。** [FAISS metric文档](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)。

doc IDs按字符串排序固定行序。同分按较小行号排序；若top-k截止位置存在同分，会扩大召回候选直到覆盖截止分数的所有ties，再做统一排序。doc→doc几何评估按实际row ID排除自身，不假设自身永远是第一个结果。query与document是独立ID namespace，相同字符串ID不会无条件删掉。

本仓库独立实现（检索深度至少10，即使IR指标只配置@1/@3也保留真实overlap@10）：

- **nDCG@10/100**：gain=原qrels relevance，不自动变成2^rel−1；ideal排名基于完整qrels。
- **Recall@10/100**：相关文档命中数/该query全部相关文档数，不是Hit@k。
- **MRR@10/100**：cutoff内第一个relevance>0文档的倒数排名，没有则0。
- **query→doc overlap@10**：低维top10与原维top10交集/10。
- **doc→doc overlap@10**：最多256个固定seed2026抽取anchor，排除self，比较原维/低维近邻集合。

所有qrels queries都进入macro average，没有结果/没有命中的query按0计。重复检索文档会报错。nDCG定义对齐 [NIST trec_eval ndcg_cut](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c)，其gain就是relevance。测试中以sklearn无ties夹具交叉验证，并用手工例子验证Recall/MRR；**当前未声称已在全部BEIR结果上与pytrec_eval完成逐query交叉校验**。

每个随机seed保存逐query分数。每run计算相对Original的paired query bootstrap：2,000次、seed2026、95%区间。这个区间条件于当前拟合seed；多seed标准差单独报告，不能把query×seed当完全独立样本。自动报告里Original/PCA的std=0只表示未重复拟合，不代表没有统计不确定性。

## 8. 正式实验怎么跑

先完成pilot，再用默认四维度、多种子配置跑dev：

```bash
uv run retrieval-lab pipeline --config configs/scifact.toml
uv run retrieval-lab pipeline --config configs/nfcorpus.toml
uv run retrieval-lab pipeline --config configs/fiqa.toml
```

每个数据集是 **1 Original + 4 PCA + 12 RP + 12 AE = 29 runs**。配置中的dev是供方法设计/参数选择使用；不要把dev结果标成最终test。

冻结配置、记录选择理由后，显式测试：

```bash
uv run retrieval-lab run --config configs/scifact.toml --split test
uv run retrieval-lab run --config configs/nfcorpus.toml --split test
uv run retrieval-lab run --config configs/fiqa.toml --split test
```

test阶段不会自动调参；当前程序会在该study重新fit压缩器，因此固定CPU/seed/配置以对应同一方法。正式研究可进一步实现从dev已保存compressor直接评估test的接口；当前重新训练行为必须在报告说明，不宣称继承完全相同的dev权重。不得根据test结果更改超参再选择性报告。

分步运行适合定位故障：prepare→embed→run。`run`只使用已经存在且校验通过的embedding缓存，不会静默换模型。中断后同命令会跳过已完成且artifact checksum匹配的run；未完成的单run重新fit。更改源码/配置/embedding fingerprint会生成新的study目录，避免把不同协议覆盖混在一起。

## 9. 输出如何用于后续report

```text
runs/<experiment>/<split>/<study-id>/
  study.json                 # full config / model-data links / source hash / environment
  runs.csv                   # per-seed scores, sizes, timings
  summary.csv / summary.json # seed mean and sample std
  quality-memory.png         # quality-dimension and quality-payload plots
  report.md                  # automatically generated measured summary
  case-selection.json        # largest improvements AND regressions vs Original
  ae-128d-seed42/
    compressor.json          # training choices/history, actual steps/device
    compressor.npz           # deployable encoder + corpus mean; no decoder
    training_decoder.npz     # AE-only decoder, separate storage
    training.csv             # reconstruction curve
    index.faiss
    ranking.jsonl            # query ID, top-k document IDs, actual scores
    per_query.json           # all evaluated queries, individual metrics
    geometry.json            # geometry anchors and self-excluded overlap
    result.json              # summary / bootstrap / timing / artifact checksums
```

`study-id`取实际config、数据manifest、embedding文件hash、Python源码hash与稳定执行环境的fingerprint。Python/依赖版本、操作系统、芯片/内存变化会生成新study，不混用旧timing；复用时保留首次study metadata，不覆盖时间/环境。历史实验生成于首次Git提交之前，因此git_commit可以为null，但source_sha256仍标识运行的源码；与其匹配的核心源码归档也已保存。

`runs/`、data、cache、.venv被Git忽略，避免把模型和大文件提交。可用下列命令将已核实的轻量summary、figure、逐query指标、study与embedding manifest导出到 `docs/results/`；不复制模型/原始数据/大索引。快照保留原study ID、环境和source hash，不将导出时的源码当作实验源码。

```bash
uv run python scripts/export_snapshot.py \
  --study runs/<experiment>/<split>/<study-id> \
  --output docs/results/<experiment>
```

重新生成报告：

```bash
uv run retrieval-lab report --study runs/<experiment>/<split>/<study-id>
```

报告写作顺序见 [report-guide.md](docs/report-guide.md)：先交代协议，再展示绝对分数、相对保持率、多种子波动、几何/任务差异、效率与失败案例。**不预填“AE最好”或任何尚未测出的结论。**

## 10. 存储与时间：测到的是什么

- `vector_payload_bytes` = N×d×4：float32文档向量负载。
- `index_file_bytes`：实际序列化FAISS文件字节。
- `compressor_file_bytes`：部署变换文件，AE不含decoder。
- `training_decoder_file_bytes`：单独的研究阶段decoder成本。
- fit、corpus transform、query batch transform、index build分别记录。
- search p50/p95：固定线程、warmup后、默认最多100 queries×3次，CPU逐query调用整个search wrapper，含normalize/排序/tie expansion，**不含Transformer编码及压缩**。
- batch QPS：单次batched search的附加测量，不等于单query实时吞吐承诺。

query transform目前是批处理平均，不能冒充逐query p95；embedding manifest的编码时间包含tokenization和缓存写盘，排除模型加载/下载，也不是严格的在线单query延迟。小corpus上不稳定的时延不能外推million-scale性能。768→64只直接意味着向量负载缩小12倍，不是端到端快12倍。

本版没有HNSW、IVF、PQ，不报告ANNRecall或ANN加速。未来增加时需要把压缩误差和ANN误差分开。

## 11. 新query的实际检索

完成真实run后，可以载入同一模型SHA、同一个mean/encoder和FAISS index搜索新文本：

```bash
uv run retrieval-lab query \
  --run runs/scifact-pilot/dev/<study-id>/pca-128d-seed0 \
  --text "Does vitamin D reduce the risk of fractures?" \
  --top-k 5
```

这条命令展示检索到的doc ID、cosine、title和正文开头；没有生成式LLM或reranker。它用于功能演示，不参与标准数据集评分；目前在线展示使用FAISS原生同分顺序，正式evaluation使用第7节的完整tie规范。

## 12. 验证与已知限制

2026-10-26进一步完成了[结果复核与改进分析](docs/verification/scifact-dev/report-zh.md)：29个run的逐query指标独立重算、float64完整库穷举top10校验、选定PCA/RP/AE重新训练，以及AE重建floor、中心化与种子/查询统计诊断。复核没有改变原始四法分数，全部额外变体单独标为diagnostic。可复现命令、诊断图、截断比例和失败案例均在该文档。

测试覆盖：独立PCA对sklearn投影几何、RP随机性/方差/持久化、AE在低秩数据上学习reconstruction、保存后变换一致、FAISS对NumPy exact cosine、截止ties与self排除、graded nDCG、空结果queries、Recall≠Hit、query split隔离、zip路径安全，以及四方法完整离线pipeline与报告生成。

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest -q
```

真实pilot的所有四法nDCG@10/100也已对独立sklearn评分代码完成交叉校验，使用完整qrels计算IDCG，并按保存的实际ranking明确tie顺序：

```bash
uv run python scripts/verify_metrics.py --study runs/<experiment>/<split>/<study-id>
```

轻量实测快照见 [SciFact pilot](docs/results/scifact-pilot/report.md)，完整限制和解释见实验日志。它不替代正式四维度/三种子/test实验。

验证只说明相应行为通过检查；合成数据不证明真实retrieval效果。当前仍需在正式实验中关注：AE有限优化预算、长文本截断、多seed稳定性、数据/模型预训练重叠不可完全排除、跨硬件浮点差异、query bootstrap与seed variance区别。

未实现项目：RAE/QRAE、hard-negative训练、supervised训练、非线性AE、全维中心化诊断、ANN、精确训练续跑、逐query端到端latency、自动跨数据集综合统计。这些不会出现在结果表中冒充完成。

## 13. RAE/QRAE规划与Raymond评语

路线见 [后两项规划](docs/roadmap-rae-qrae.md)。当前接口已统一fit/transform/save/load；未来扩展仍要沿用缓存、ID、search和metrics。

老师要求独立实现核心方法和端到端评估。本仓库PCA/RP用NumPy独立实现，AE用标准PyTorch层/optimizer独立训练，数据解析/指标/编排由本仓库实现，FAISS仅承担标准索引。没有复制RAE项目的训练pipeline。RAE参考代码只用于调研细节；引用不等于将其代码提交为自己的贡献。

高质量交付的重点：固定协议；记录实现选择；检索完整corpus；正确划分；将geometry与task metrics分开；保存seed与逐query数据；报告负面结果。课程没有完整rubric，以上是对评语的响应，不是分数保证。

## 14. 论文、代码来源与延伸阅读

| 来源 | 在本仓库中的用途 |
|---|---|
| [BEIR (2021)](https://arxiv.org/abs/2104.08663) / [官方repo](https://github.com/beir-cellar/beir) | 数据定义、下载路径、IR benchmark背景 |
| [E5](https://arxiv.org/abs/2212.03533) / [模型卡](https://huggingface.co/intfloat/e5-base-v2) | 冻结模型、prefix、维度、pooling与归一化 |
| [sklearn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) | 定义与独立数值校验；不是主实现 |
| [Gaussian RP](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html) | 随机矩阵分布与理论边界 |
| [Linear AE/PCA, Plaut 2018](https://arxiv.org/abs/1804.10253) | 重建子空间与latent坐标的区别 |
| [FAISS](https://github.com/facebookresearch/faiss) | 标准exact index，不复制evaluation harness |
| [NIST trec_eval](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c) | graded nDCG的gain定义 |
| [RAE](https://arxiv.org/abs/2509.25839) | 后续方法规划和几何指标；尚未实现 |
| [Matryoshka-Adaptor](https://aclanthology.org/2024.emnlp-main.576/) | 相关工作，避免将监督压缩误称首创 |
| [DPR](https://aclanthology.org/2020.emnlp-main.550/) | 未来negative sampling/ranking训练背景 |

深入调研记录：[原实施研究](research/implementation-plan-zh.md)、[数据与模型](research/data-models.md)、[前三项细节](research/baselines-code.md)、[监督压缩相关工作](research/related-qrae.md)、[本轮实现决策](research/implementation-decisions.md)。

## 15. GitHub协作与公开材料

协作仓库：[changxuanfan/retrieval-compression-lab](https://github.com/changxuanfan/retrieval-compression-lab)，默认分支`main`。仓库包含许可证、贡献指南、PR/issue模板和GitHub Actions离线检查；远端CI状态以GitHub实际运行记录为准。

课程proposal/评语PDF含学生信息或内部反馈，保留本地并由.gitignore排除。data/cache/runs/private/.venv也不提交。公开结果快照只清理本机绝对路径并刷新导出checksum，数值指标保持不变；未清理副本在本地private/publish-backup，原始runs未修改。核心源码hash不因导出元数据清理而改变。

每次导出新的结果快照后运行：

```bash
uv run python scripts/sanitize_public_artifacts.py
uv run python scripts/check_publication.py
```

检查覆盖候选Git文件及ZIP内的个人路径、已知凭证模式、未审核PDF和大文件。特定课程标识可放本地忽略的private/publication-identifiers.json；该私有文件本身不发布。检查并非所有可能敏感信息或第三方权利的保证，新增材料仍需人工查看。

为后两项协作，先约定RAE设置和QRAE的train-only数据接口，再各自开PR。所有方法沿用同一模型revision、文档ID、检索库和评价定义，不覆盖已验证的前四项结果。
