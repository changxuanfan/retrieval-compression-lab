# 从仓库材料写课程报告

当前实际范围是Original、PCA、Gaussian RP、centered linear AE。RAE/QRAE只能出现在future work/proposed extension，不能在abstract写成已完成六法。

## 方法与设置

报告说明冻结的E5完整revision、768维、英文、query/passsage prefix、512-token截断和L2 normalization。引用embedding manifest，不只写“用了BERT”。

数据表说明文档数、query数量、官方与内部split、SciFact split seed2026。描述完整corpus无标签拟合及query-disjoint评价；明确没有relevance supervision。将test作为最终冻结协议后的评估，dev表与test表分开。

四个方法写清公式。PCA减均值、不whiten；RP方差1/d、三seeds；AE中心化、无bias/activation、untied encoder–decoder、Adam无decay、loss为逐样本平方和均值。AE训练probe不叫validation集。

## 结果展示

主表使用summary.csv的nDCG@10、Recall@10/100、MRR@10，列出方法、维度、seed数量、mean±sample std。Original/PCA只运行一次，不能把std=0解释为显著优于随机方法。

用绝对指标为主；如果计算quality retention，保留其原始分母和绝对分数，不能掩盖低基准。原始向量不是理论上限。展示quality-dimension和quality-memory曲线，图下注明数据集、split、model和study ID。

geometry分析在相同维度内比较query→doc/doc→doc overlap与nDCG。不同维度混在一起的相关性可能只反映维度本身。寻找“较低overlap但较高nDCG”的真实例子，也如实写不存在这种现象。

效率只写测量范围支持的结论：vector payload与index实测大小分开，query encoding与search分开，不把压缩比直接称加速比。单queryp50/p95与批量QPS不是同一概念。

## Failure analysis

case-selection.json提供每run最大改善与恶化query IDs；去原始queries.jsonl查文本，再用ranking.jsonl查原始/压缩top10和qrels。比较相关文档rank变化、词义混淆、长文截断等可能因素。不能只根据一句query猜测原因；标为定性假设，并展示证据。

## Reproducibility与局限

附配置文件、依赖lock、data/model/source hashes、执行命令、实际训练步数、硬件、seed。说明本次选定哪些dev配置及理由，不能test后挑选最好seed。

局限包括：仅三套小型英文corpus；固定backbone；AE线性且有限优化；相同corpus参与无标签拟合；长文截断；未实现RAE/QRAE/ANN；单次query bootstrap不能覆盖所有seed不确定性；尚未做的检验应明确标明。

目前没有报告模板正文自动宣称某方法更好。最终结论必须从实测run导出。
