# 四法结果复核：统计解释、AE诊断与改进优先级

调研日期：2026-10-26。本文件为补充研究和建议；没有重跑模型或更改实现。读取了README、实际compressors.py、SciFact dev summary及逐seed配对区间。运行数值、完整数据评分、训练复现由主任务另行核验。

## 1. 目前结果可以支持什么

现有SciFact内部dev结果是完整5,183文档库、162个queries、固定E5、四维度、RP/AE三个种子的实测。它能支持这个具体设置下的比较；不能外推成跨域结论或最终test结果。

来自保存的summary与paired-comparisons：

- AE128d三个seed相对PCA128d的nDCG@10分别提高0.02419、0.00930、0.00795；只有seed17的单模型query bootstrap区间未跨0。这个区间条件于该次拟合，不能据此断言训练算法总体显著更好，也不能只挑seed17。
- AE128d mean nDCG@10为0.68838，PCA为0.67457；与此同时AE的query→doc overlap为0.60370，低于PCA的0.61790，doc→doc overlap也较低（0.68320 vs0.69336）。这是**同维度下geometry保持与task utility不完全一致的描述性例子**，不是两者永远负相关的证据。
- AE在64d与PCA近似，256/384d均值略低；当前结果不是“AE全面胜过PCA”。PCA应保留为强基线，RP低维表现较差也不等于实现错误。
- 64d AE的Recall@10均值高于PCA（0.74691 vs0.73364），但nDCG略低，说明“是否命中相关文档”和“是否排在很前面”不能互相替代。

上述数值来源：[SciFact summary](../docs/results/scifact-dev/summary.json)、[逐seed配对比较](../docs/results/scifact-dev/paired-comparisons.json)。它们是本仓库实测，不是论文提供的预期数值。

## 2. 为什么线性AE与PCA检索不必一样

文献区分principal subspace和principal component坐标：线性AE的平方重建目标能学习PCA主子空间，但权重不必就是正交loading。[Plaut 2018](https://arxiv.org/abs/1804.10253)，尤其§II-E与§III。

以下是对本实现的代数推导：设z=W_e(x−μ)，重建为W_d z+μ。对任意可逆A，将encoder换成A W_e、decoder换成W_d A⁻¹，重建完全相同。除非A只是全局缩放与正交旋转，一般会改变latent夹角。因此近似相同reconstruction不保证相同cosine排名。当前untied、无正交约束的AE存在这种自由度，AE与PCA排名不同本身并非bug。

这还意味着：如果延长训练只改善重建，却改变latent尺度，nDCG可能下降。不能把检索分数不升简单归因于“训练不够”。涉及正则线性AE的收敛研究也强调优化和维度可能影响收敛，但其正则设置与本仓库vanilla AE不同，不能直接搬其速度结论。[Bao et al., Regularized linear autoencoders recover the principal components, eventually](https://arxiv.org/abs/2007.06731)。

建议先诊断，再决定是否增加steps：

1. 用**整个拟合corpus**计算AE逐样本平方和平均，不只看前1,024个probe；probe是训练诊断，不是validation。
2. 相同中心化数据X_c、相同d下，PCA重建最优误差为 `L* = Σ_{j>d} σ_j² / N`。与AE实际loss使用同一reduction；逐元素MSE会差D倍。
3. 报告AE相对floor的gap `(L_AE−L*)/L*`，数值误差之外不应出现AE明显低于PCA理论floor；若发生优先检查center、loss定义或输入不一致。
4. 比较encoder行空间与PCA子空间的principal angles，以及encoder奇异值。接近cutoff的重复/近重复特征值会让个别方向不稳定，因此角度要与重建gap一起解读。
5. 如果gap已小，不必单纯堆训练步数；可以把“同子空间、不同latent几何”作为解释。正交化后的encoder可做诊断，但应另标diagnostic，不能悄悄替换原AE再覆盖旧结果。

PCA全维中心化、uncentered-SVD等也是低成本诊断，可分离中心化与截断影响；不应把所有PCA cosine损失都归因于降维。

## 3. 三个seed和162个queries：如何解释不确定性

固定模型的paired query bootstrap只覆盖query抽样变化，不覆盖训练随机性。报告seed分布和单模型区间是合理第一步，但要区分“这个模型实例”与“训练过程”的结论。[Reimers & Gurevych, EMNLP2017](https://aclanthology.org/D17-1035/)。

MultiBERTs提出同时考虑训练seed与evaluation examples的Multi-Bootstrap，并区分固定baseline、匹配seed和不匹配seed的设计。[Sellam et al., ICLR2022, §3](https://arxiv.org/html/2106.16163v2#S3)。本项目对PCA这个固定baseline已有逐query分数，可使用相同query抽样计算差值，再对AE训练seed抽样；不要把3×162行当486个独立query。

具体建议（本项目推导，不是文献保证）：构造每维度的 `delta[seed, query] = AE_score−PCA_score`；每次bootstrap独立抽seed行和query列，在所有seed中使用相同的query列抽样，取所抽矩阵均值。这个crossed-resampling区间比仅query抽样更符合当前比较目标，但只有3个seed时对训练分布仍然很粗，不能保证稳定覆盖率。

如果要坚持“128d AE有稳定增益”的小差异结论，考虑把这一个主要比较扩展到5–10个预先固定seed；这是计算成本折中建议，不是任何论文规定或显著性保证。不要为所有设置机械扩展大量seed。

还要预先指定主指标nDCG@10、主要比较和维度；多个方法/维度/指标都看过以后再挑一个未跨0的区间，会产生选择偏差。逐run的95%区间不是整张表同时95%覆盖。若正式做多重假设检验，需要说明检验族与校正方案。配对设计和检验选择的基础阅读见[Dror et al., ACL2018](https://aclanthology.org/P18-1128/)；该论文主要讨论单次比较，不应误引为已解决所有多数据集比较。

## 4. corpus适配不是query标签泄漏，但目标要写清

当前PCA/AE在完整目标corpus上无标签拟合，在“索引库已知、未来query未知”的检索部署假设下可成立，应持续标注corpus-adaptive/transductive。这里test单位是held-out queries，不是把corpus文档当监督分类中的测试样本。已有后处理压缩论文也明确以可用corpus embeddings作为无监督适配数据。[Matryoshka-Adaptor, §3](https://aclanthology.org/2024.emnlp-main.576.pdf)。

这不能外推为新corpus完全未见时的inductive泛化；如要声称后者，需要另外的source/target-corpus实验。query qrels、test分数不能用于选择steps、seed或center规则。通用模型选择原则要求把最终test与选择过程分开，[sklearn官方data leakage说明](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)；但它的监督分类例子不能不加区分地用来否定“已知检索库”的本项目假设。

当前CLI test会重新fit，尽管CPU+seed通常可复现，正式评估更清楚的实现是**固定已经选择的compressor artifact，仅对test queries transform/search**，并检查权重checksum。此建议减少协议歧义，不代表目前发生了test标签训练。

## 5. 最值得做的改进顺序

| 优先级 | 建议 | 要解决的问题 |
|---|---|---|
| P0 | 结果/模型/数据hash、指标和检索独立核验；CPU重训一个seed | 先确认数值可信，而不是追求更高分 |
| P1 | 完整corpus AE reconstruction相对PCA floor、训练曲线、子空间/谱诊断 | 判定优化是否充分，解释AE256→384不升 |
| P1 | matched-query统计与同维geometry/task图；保留成功和失败案例 | 让报告回答研究问题，而非只列leaderboard |
| P1 | 冻结协议后跑官方test；优先增加NFCorpus，再FiQA | 比在SciFact反复调到更高dev更能增加可信度 |
| P2 | 仅对128d关键比较增加seed或做crossed seed/query bootstrap | 检查约1.38个nDCG百分点的均值差是否稳健 |
| P2 | 统计512-token截断比例；两次独立latency采集、query encoding/projection/search分开 | 排查文本截断，避免夸大实际效率 |
| P3 | 非线性AE、正交/whitening等新变体另建配置/结果 | 可以探索，但不是让原baseline“赢”的必要条件 |

不建议为追求高分删RP、挑best seed、test调参，或把普通AE改成更强网络却继续声称只有loss不同。当前四法范围已经足以做严谨课程研究；RAE/QRAE应继续只保留规划。
