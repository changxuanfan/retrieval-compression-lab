# 实验与验证记录

日期：2026-10-26（Asia/Hong_Kong）。开发机器Apple M5 Pro、24GB内存。

## 本轮范围

已建立四种方法的代码、共享完整corpus检索/指标/导出、配置与测试。RAE/QRAE仅规划。

真实数据准备已运行：SciFact官方archive可下载，5,183 documents；内部query split为647 train / 162 dev / 300 test。数据manifest保存在本机data/scifact/manifest.json。

## 已完成的真实pilot

E5-base-v2完整revision：`f52bf8ec8c7124536f0efb74aca902b2995e5bcd`。MPS、batch16、512 tokens；完整SciFact corpus编码约123.88秒，1,109个有标注query编码约3.82秒。这里排除模型下载/加载，包含tokenization、归一化和数组写盘，不是逐query在线延迟。

四法已完成完整corpus、162个内部dev queries上的pilot：Original768d，其余128d；RP/AE seed42；AE1,000步。nDCG@10为Original0.75201、PCA0.67457、RP0.52746、AE0.67907。轻量provenance、逐query指标和图表保存在 [results/scifact-pilot](results/scifact-pilot/report.md)。

AE−PCA的nDCG@10差为0.00450，2,000次paired query bootstrap 95%区间约[-0.01539,0.02398]，跨0；不能据此声称AE明显优于PCA。该区间也不包含不同训练seed的不确定性。128d文档向量负载约为768d的1/6；实际search timing只是本机pilot测量，不据此声称端到端加速。

## 环境问题与修复

全管线测试首先在AE probe索引处发生原生segmentation fault。最小化后，FAISS-first调用可重复崩溃，Torch-first调用明确报`OMP Error #15: libomp already initialized`。检查到Torch、FAISS各自携带不同libomp。不是通过吞掉异常或允许重复runtime处理。

修复为编码、Torch训练、FAISS评估的进程隔离，数值权重通过npz/JSON传递；限制BLAS线程时不再统一修改所有OpenMP运行时。完整合成pipeline现在覆盖同样的调用链，并检查AE不会在评估进程导入Torch。没有声称每一种macOS版本都存在此问题，也没有把“单次不崩溃”代替数值验证。

另外修复环境provenance和overlap截止：study fingerprint纳入稳定硬件/软件环境；复用不覆盖原metadata；IR cutoff小于10时仍召回足够10个邻居。新增对应回归检查。

## 验证范围

数值、指标、data split、安全解压、模型持久化和完整离线pipeline等20项测试全部通过，Ruff lint/format与uv lock检查通过。四个真实pilot的nDCG@10/100已通过独立sklearn、完整qrels的交叉校验。新文本query→E5→PCA→FAISS→文档展示也已实际执行成功。

NFCorpus/FiQA官方数据下载与解析已验证，规模和官方split与研究记录一致；尚未编码/训练这两套数据。原始日志留在本机，Git内只保存轻量结果快照，不提交模型/原始数据/大索引。

## 完整SciFact dev sweep

使用configs/scifact.toml，study `342e2396ee22`，已完成29 runs：Original一次；PCA四维度各一次；RP/AE四维度各三个固定seed17/42/2026。AE每run严格3,000步，所有fit wall time合计约150.70秒（含worker启动，排除之前的embedding生成与搜索/报告时间）。

Original nDCG@10=0.75201；PCA在64/128/256/384d为0.59230/0.67457/0.71950/0.72392；AE mean为0.59166/0.68838/0.71648/0.71501；RP mean为0.32941/0.51161/0.63982/0.68552。详见 [完整dev快照](results/scifact-dev/report.md)，含seed标准差与逐query指标。

这是内部dev上的实测，未使用test调参，也不称最终test结论。AE只在128d的点估计高于PCA，不能概括为AE全面更优。所有29个run的nDCG@10/100均以独立sklearn、完整qrels交叉验证。没有运行RAE/QRAE。

## 后续结果复核（2026-10-26）

用户要求verify并思考改进后，另外执行了29个run的文件/保存模型/FAISS向量检查、独立scalar指标重算、NumPy float64完整库穷举。29×162个query的top10均与原记录一致；最大rank score gap约1.235e-7，nDCG差为0。

PCA128、RP128 seed42、AE128 seed42重新fit的部署参数与保存参数最大差均为0。全corpus AE重建误差比PCA floor高约0.2%–1.0%，不支持简单将检索差异归为训练不充分。AE128−PCA的种子+query bootstrap区间约[-0.00115,0.03004]，仍跨0。

仅中心化768d nDCG=0.72762，证明减均值也是Original/PCA差异的重要环节。截断审计发现466/5,183 corpus documents超过512 tokens；唯一dev相关文档中36/171超过。相关数字都是当前缓存revision的复核，不是新模型结果。

详细建议、可复现脚本和诊断图见[复核报告](verification/scifact-dev/report-zh.md)。保留原始四法数据；test和其他数据集检索尚未在本轮运行。
