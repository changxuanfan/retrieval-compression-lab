# RAE / QRAE：仅规划，尚未实现

当前已完成前四项；RAE和QRAE是下一阶段的协作任务。以下不代表已有代码、实验或结果，接入方式与PR要求见仓库根目录CONTRIBUTING.md。

## RAE接入

先选择论文版本，并逐项核查公开参考实现的bias、归一化、正则覆盖范围、Adam/AdamW、MSE reduction和实际更新步数。当前AE使用corpus mean centering，不能悄悄把它的loss加上λ后称为所有原论文细节的严格复现。

可保留两条轨道：原论文设置的独立algorithm reproduction；本仓库与AE匹配的controlled RAE-style extension。每条轨道都记录差异。标准化明确的regularizer，避免显式L2和optimizer decay重复计算。

复用现有数据/缓存/搜索/metrics接口，但core训练由组员独立理解与实现，不复制公开RAE工程。缺少原论文embedding checkpoint/抽样IDs时，不能声称精确复现表格数字。

## QRAE接入

只有train queries/qrels允许生成triplets。dev选择margin、λ、α；test只做最终评价。ranking使用归一化latent，decoder仍使用原始latent。reconstruction应维持与AE/RAE相同corpus池和预算。

先做固定negative数的triplet pilot，再比较random/in-batch/static hard候选；排除已知train positives，不用test qrels清洗，不将unjudged自动认定为真negative。

必须增加Rank-only、full-dimension supervised projection等controls，区分监督适配与RAE结构贡献。共享静态encoder是训练时task-aware，不是每个query动态更换document embedding。

## 贡献表述

监督压缩不是空白领域，必须引用Matryoshka-Adaptor、Search-Adaptor等相关工作。项目可以贡献独立复现、统一query-disjoint实验、geometry/task关系和低维/负样本/标注量分析；不将普通triplet公式声称首创。

相关资料及核查链接见research/related-qrae.md与research/implementation-plan-zh.md。RAE/QRAE没有CLI实现、占位结果或默认运行配置。
