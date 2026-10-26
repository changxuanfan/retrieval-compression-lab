# 协作指南

当前可运行范围：Original、PCA、Gaussian RP、centered linear AE。RAE/QRAE是后续开发任务，不能把普通AE结果标成已实现后两项。

## 首次加入

```bash
git clone https://github.com/changxuanfan/retrieval-compression-lab.git
cd retrieval-compression-lab
uv sync --locked --extra dev
uv run pytest -q
uv run retrieval-lab demo
```

测试和demo不下载数据或模型。真实SciFact实验按README的prepare→embed→run执行；每人自行下载，使用记录的模型revision，避免通过Git传递大模型/向量文件。

## 分工与边界

| 工作 | 建议独立PR | 验收依据 |
|---|---|---|
| RAE论文/参考实现版本核对 | `RAE protocol` | 正则覆盖范围、bias、centering、loss reduction、optimizer、实际steps的对照表 |
| RAE独立实现 | `RAE implementation` | 论文设置与匹配AE结构的控制设置明确区分；几何与retrieval测试 |
| QRAE数据/negative sampling | `QRAE training data` | 只读train queries/qrels；同query不跨split；known positives mask；固定候选及seed |
| QRAE训练目标 | `QRAE implementation` | 显式loss公式、raw latent重建/normalized latent排名；rank-only等控制 |
| 统一评估和报告 | `Evaluation and analysis` | 共用缓存、完整corpus、相同search/metrics，dev/test明确 |

一名成员可承担多个任务，但不要让两个人同时大改同一训练文件。先在issue/PR中确定接口，再接入公共管线。无需重写embedding/search/metrics。

## 代码接入点

- `src/retrieval_lab/compressors.py`：目前的fit/transform/save/load；未来可提取独立方法模块，保持已有接口行为。
- `worker.py`：Torch训练在独立进程；保持与FAISS运行时隔离。
- `config.py`：新方法需要显式参数校验，不能静默忽略不支持的loss/采样选项。
- `experiment.py`：现有四法只向compressor提供corpus矩阵。QRAE需要设计独立的train-only数据入口，不能直接把整个dataset（含dev/test qrels）交给训练器。
- `embedding.py` / `search.py` / `metrics.py`：公共基础，修改应提供相同输入下的兼容验证。

未来RAE必须说明与当前centered AE的关系，不能不记录设置变化。QRAE不能只扩展方法名字而仍调用普通AE训练。具体研究约束见 [RAE/QRAE规划](docs/roadmap-rae-qrae.md)。

## PR要求

1. 在自己的分支开发，经PR合入默认分支；实验配置和方法代码放同一PR。
2. 写清最终行为、数学定义和验证，不把开发日志当PR描述。
3. 运行以下检查；新增数学/采样行为应有有意义的测试。

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest -q
```

4. 不覆盖既有结果。新配置/源码/环境生成新的study；分享轻量快照前去掉本机路径并重新校验导出hash。
5. 只用dev做选择，最终test读取冻结后的协议/模型；不得best-seed选择性报告。
6. 公共提交不包含课程PDF、学号、个人路径、凭证、原始数据、模型权重或大索引。使用本仓库的发布检查脚本验证候选文件。

## CI与结果声明

GitHub Actions配置运行锁定依赖下的lint/format和离线测试，不自动下载E5、不运行真实BEIR实验。首次推送后才能确认远端CI是否通过；本地通过不等于已经有GitHub绿色状态。

RAE/QRAE的几何复现、真实检索和消融仍需组员在本机/服务器运行，并记录model/data/source hashes。CI不能替代研究实验。
