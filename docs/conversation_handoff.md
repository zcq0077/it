# iTentformer DMA 项目对话交接摘要

更新时间：2026-07-18 15:50（Asia/Shanghai）

用途：在新 Codex 对话中先让助手阅读本文件，再继续监控实验或修改代码。本文只保留会影响后续工作的结论、路径、实验结果和待办，不复述逐轮问答。

## 1. 项目目标

- 项目目录：`D:\iTentformer-master`
- 基础模型：iTentformer，用于 AIS 船舶长期轨迹预测。
- 论文：*Trend-Enhanced Variate Transformer for Vessel Trajectory Prediction by Exploiting Short-Term Behavior Distribution Differences*。
- 当前任务：使用丹麦 DMA AIS 数据，以约 3 小时历史预测未来 3 小时，并重点改善主航路分叉和小样本子航路预测。
- 当前不是严格复刻论文的 132 条 Ti 轨迹实验，而是在同一海域思想上构建更大的六至十月数据集并扩展模型。

最重要的研究结论：如果不同子航路在当前历史窗口内几乎完全重合，且尚未出现转向、目的地等可区分信息，则任何模型都无法保证提前选中唯一正确分支。正确做法是识别可判别性、保留 Top-K 候选，并在意图显现后再收敛，而不是强迫早期硬分类。

## 2. 环境与入口

- Conda 环境：`lucky`
- Python：`D:\SoftWare\Environment_M\envs\lucky\python.exe`
- 本地 Qwen：`D:\Jason1982\wsl\Models\Qwen3-1.7B`
- 主入口：`iTentformer.py`
- 默认配置：`config_iTentformer.py`，直接执行主程序会自动读取，不必显式加 `--config`。

PowerShell 新会话建议先设置临时目录，规避中文用户名路径导致的 SciPy/临时文件问题：

```powershell
$env:TEMP='D:\iTentformer-master\.tmp'
$env:TMP='D:\iTentformer-master\.tmp'
python iTentformer.py
```

上述环境变量只对当前 PowerShell 会话有效。

## 3. 论文数据理解

- 论文 DMA 原始数据约有 187,048,476 个点，覆盖 2021 年 12 月至 2022 年 5 月。
- 论文在临界 Ti 区选择并平衡了 132 条典型轨迹，其中 58 条 O->A，其余 O->B；约 85/20/27 用作训练/验证/测试，并使用 K=5。
- 132 条是论文为受控实验筛选出的代表性轨迹，不是 iTentformer 的硬性数据量或类别限制。
- 航路 O/A/B1/B2 等分类主要属于论文的数据筛选与实验设计，模型本身可以扩展到 OC 或更多类别。
- 当前项目已改为传统固定 holdout，只运行一次，不再默认进行 5 折循环。

## 4. 数据预处理

原始 ZIP 位于：

```text
D:\AIS\2023_06_09\aisdk-2023-06.zip
D:\AIS\2023_06_09\aisdk-2023-07.zip
D:\AIS\2023_06_09\aisdk-2023-08.zip
D:\AIS\2023_06_09\aisdk-2023-09.zip
D:\AIS\2023_06_09\aisdk-2023-10.zip
```

`utils\preprocess_dma_zip.py` 可以流式读取 ZIP，不要求先完整解压几十 GB 数据。预处理大体包括：

1. 分块读取 AIS CSV。
2. 过滤空间范围、无效经纬度、异常 SOG/COG 和不连续时间点。
3. 按 MMSI 和时间连续性切分轨迹。
4. 重采样、平滑、长度筛选和异常跳点剔除。
5. 根据 O/A/B1/B2/C 区域、方向和中途分叉形状分类。
6. 转成 iTentformer 需要的轨迹对象和标签侧车。

模型数据为 15 列：

```text
[MMSI, Length, Course, Lon, Lat, SOG, vx, vy,
 delta_Course, delta_Lon, delta_Lat, delta_SOG,
 delta_vx, delta_vy, UnixTime]
```

当前代码不再使用早期截图中的固定经纬度 min-max 常数。均值和标准差只由固定训练集拟合，再应用到验证和测试集，避免数据泄漏。

主要工具：

- `utils\check_dma_quality.py`：数据质量报告。
- `utils\classify_itentformer_routes.py`：主航路分类。
- `utils\discover_subroutes.py`：子航路聚类和标签生成。
- `utils\plot_dma_route_heatmap.py`：航路热图。
- `utils\merge_itentformer_datasets.py`：月份数据合并。
- `utils\build_dma_voyage_context.py`：构建船型、吃水、Destination 等历史可见上下文。
- `utils\build_qwen_semantic_teacher.py`：生成 Qwen 语义向量侧车。

## 5. 当前数据集

默认数据：

```text
dataset\dma_raw_2023_06_07_08_plus_09_10_oa_s00\
  dma_2023_06_07_08_plus_09_10_oa_s00_target350.pkl
  dma_2023_06_07_08_plus_09_10_oa_s00_target350_route_labels.json
  dma_2023_06_07_08_plus_09_10_oa_s00_target350_subroute_labels.json
  dma_2023_06_07_08_plus_09_10_oa_s00_target350_fixed_split.json
  dma_2023_06_07_08_plus_09_10_oa_s00_target350_voyage_context.pkl
  dma_2023_06_07_08_plus_09_10_oa_s00_target350_qwen_semantic.pkl
```

总计 4,920 条轨迹，当前大类分布：

| 大类 | 轨迹数 |
|---|---:|
| OA | 1,564 |
| OB1 | 2,188 |
| OB2 | 643 |
| OC | 525 |

紧凑版六个子航路：

| 子航路 | 轨迹数 |
|---|---:|
| OA_S00 | 225 |
| OA_S01 | 713 |
| OA_S02 | 626 |
| OB1_S00 | 2,188 |
| OB2_S00 | 643 |
| OC_S00 | 525 |

固定 MMSI 分组划分：训练 3,465、验证 484、测试 971 条；同一 MMSI 不跨集合。窗口数为训练 124,126、验证 17,110、测试 34,329。

注意：九、十月目前只定向补入了 66 条合格 OA_S00，使其从 159 增至 225；文件名里的 `target350` 是目标值，不代表最终真的有 350 条。这个数据集适合做“小类补充”消融，但论文主结果最好再构建九、十月所有合格航路均纳入的全量版本，避免月份与类别来源耦合。

## 6. 航路标签与泄漏边界

- 大类：OA、OB1、OB2、OC。
- 子类：OA_S00、OA_S01、OA_S02、OB1_S00、OB2_S00、OC_S00。
- 子航路划分融合起点/终点导向和中途分叉形状，不只是按终点聚类。
- 完整真实轨迹可用于生成训练标签，但推理时不把真实航路标签或未来轨迹喂给模型。
- Destination、船型、吃水只有在预测时刻之前已经由 AIS 提供时才可使用；不能使用预测时刻之后更新的 Destination。
- 当前 Qwen 语义侧车标记为 `label_free=True`，不读取真实未来和航路答案。

## 7. 当前 3 小时设置

- `input_length = 13`：按当前重采样间隔表示约 3 小时历史。
- `target_length = 12`：表示约 3 小时未来。
- `window_stride = 1`。
- `target_mode = residual_linear`：先做线性运动基线，再预测残差，降低长时滚动漂移。
- 默认训练 50 轮，早停 `patience = 10`。
- 早停监控：`ADE + 0.2 * FDE`。

## 8. 当前模型流程

```text
历史 AIS 动态特征
    -> TCN/Transformer 共享轨迹编码
    -> 大航路概率 + 子航路概率 + 可判别性
    -> 航路/子航路 embedding 与训练集原型先验
    -> 共享轨迹解码器
    -> 通用子航路残差专家
    -> 全子航路候选轨迹池
    -> 学习式候选选择器
    -> 最终单条预测轨迹

历史可见船型/吃水/Destination
    -> 离线 Qwen3-1.7B 语义向量
    -> 温和融合到航路意图特征
```

### 已启用的重要模块

| 模块 | 作用 |
|---|---|
| 大类/子类意图头 | 辅助模型学习主航路和局部分支 |
| 分阶段可判别监督 | 分叉前降低硬标签权重，意图显现后再硬分类 |
| 置信度感知 Top-K | 不确定时保留候选，避免过早锁死错误分支 |
| 训练集航路原型 | 根据历史位置、方向与中心线匹配提供几何先验 |
| 层级约束 | 大类概率温和约束其所属子类 |
| Focal/class weight | 提高小类分类梯度，但限制最大权重避免过拟合 |
| 解耦双流训练 | 自然分布训练轨迹；均衡辅助流只强化意图分类和表示 |
| 按轨迹均衡采样 | 避免一条长轨迹产生大量窗口后支配辅助意图流 |
| Future-enhanced intent | 训练期用真实未来构造意图原型，推理期只用历史匹配 |
| 候选轨迹选择器 | 枚举全部六个子航路候选，最终从候选中选一条 |
| Qwen 语义教师 | 离线编码历史可见航次语义，不直接生成坐标 |
| 子航路残差专家 | 每个子航路用轻量专家修正共享解码器，减少类别间梯度干扰 |

轨迹损失包含位置回归、Haversine 地理距离、FDE、平滑约束和圆周 COG 误差；另有主/子航路分类、可判别性、对比学习、future-intent 和候选选择损失。

## 9. Qwen 相关结论

- 千问无法从完全相同且意图未显现的坐标历史中凭空确定未来分支。
- 千问最适合做 Destination 标准化、船型/吃水/港口语义编码和候选合理性先验，不适合直接回归高精度经纬度。
- 曾实现 Qwen candidate reranker，但验证集收益未达到阈值，`accepted=False`，最终自动回退基础选择器；该无效在线重排代码后来已清理。
- 当前只保留离线 Qwen semantic teacher。训练时读取已生成的 2,048 维侧车，不会每轮加载 1.7B 大模型，因此速度较稳定。

## 10. 关键实验结果

| 版本 | 数据 | 最佳轮 | ADE | FDE | 说明 |
|---|---|---:|---:|---:|---|
| v14 基线 | 2023-06/07/08，4,854 条 | 41 | 1.52298 nmi / 2,820.55 m | 2.95187 nmi / 5,466.87 m | 当前最好的总体结果之一 |
| v14 + 定向补 OA_S00 | 4,920 条 | 23 | 1.58430 nmi / 2,934.13 m | 3.02802 nmi / 5,607.89 m | OA_S00 改善，但总体和多条主航路退化 |

v14 增量前后部分子类 ADE：

```text
OA_S00: 2.069 -> 1.832 nmi，改善
OA_S02: 2.547 -> 2.304 nmi，改善
OA_S01: 1.359 -> 1.463 nmi，退化
OB1   : 1.102 -> 1.223 nmi，退化
OB2   : 1.981 -> 2.205 nmi，退化
OC    : 1.389 -> 1.447 nmi，退化
```

结论：定向补充小类并不保证总体 ADE/FDE 下降。共享解码器会受到类别分布和月份域偏移影响；小类分类准确率提高也不等于其生成轨迹一定更接近真实未来。

## 11. 最新 v15e 修改

用户认为仅按“新增数据”限制回归窗口不够通用，这一判断正确。现已完成：

1. 删除全部依赖 `supplement_index`、月份或新增来源的训练特判。
2. 自然轨迹流恢复使用全部 124,126 个训练窗口。
3. 新增通用 `SubrouteResidualExpertBank`，六个子航路自动各建一个小专家。
4. 每个专家只修正本子航路残差，规则对所有月份和类别一致。
5. 专家输出层零初始化，启用初始输出与旧模型逐点相同。
6. 模块增加 16,224 个参数；完整模型从约 587k 增至 603,322，增幅约 2.8%。
7. 旧完整模型 checkpoint 缺少专家属性时会自动按“专家关闭”兼容加载。

配置：

```python
use_subroute_residual_experts = True
subroute_residual_hidden_dim = 32
subroute_residual_scale = 0.25
subroute_residual_dropout = 0.10
model_prefix = "dma_v15e_3h"
```

已通过 `py_compile`、真实数据 `split_only`、GPU 前向/候选生成/反向梯度和旧 checkpoint 加载检查。单元测试确认零初始化最大输出差为 0，且只有命中的子航路专家收到梯度。

## 12. 当前正在运行的实验

- PID：`26088`
- 启动时间：2026-07-18 15:49
- 版本：`dma_v15e_3h`
- 日志：`results\dma_v15e_3h-20260718-154911\train.log`
- checkpoint：`save_models\dma_v15e_3h_fixed.pt`
- 第 1 轮已经完成，当前正在训练第 2 轮。

第 1 轮结果：验证 ADE/FDE 为 2.27732/4.14138 nmi，早停分数 3.10559；测试 ADE/FDE 为 2.24584/4.14159 nmi。相同增量数据的 v14 第 1 轮测试为 2.30353/4.03150 nmi，早停分数 3.16167。因此 v15e 的 ADE 领先约 0.05769 nmi（107 m），但 FDE 暂时落后约 0.11009 nmi（204 m），综合验证分数仍更好。六个子类中五个测试 ADE 优于同数据 v14，说明收益不是只来自 OA；当前主要风险是 OB1 等类别的三小时终点误差。

监控命令：

```powershell
Get-Content .\results\dma_v15e_3h-20260718-154911\train.log -Tail 80 -Wait
```

旧的 `dma_v15_oa225_3h` 仅跑到第 2 轮后被新实验替代，包含已经废弃的 `supplement_regression_windows` 逻辑，不应作为最终版本。

## 13. 常用命令

正常训练：

```powershell
python iTentformer.py
```

仅检查划分和配置：

```powershell
python iTentformer.py --split_only --plot_count 0
```

训练完成后直接测试 checkpoint：

```powershell
python iTentformer.py --eval_only --checkpoint_path save_models\dma_v15e_3h_fixed.pt
```

预测图保存在对应结果目录的 `plots` 中，包含历史轨迹、真实未来、最终预测、起点、预测起始点以及大类/子类判断和 ADE/FDE。

## 14. 日志指标解释

- ADE：所有未来点的平均位置误差，越低越好。
- FDE：最后预测点误差，越低越好。
- `hard_top1_acc`：模型认为足够确定并硬选的样本中，第一名正确率。
- `uncertain_top2_recall`：不确定样本中，真实类别是否位于前两名。
- `decidable_top1`：按几何历史判断已经可区分的窗口中的分类准确率。
- Candidate selector accuracy：部署时选择器选中真实代价最低候选的比例。
- Oracle ADE@K：假设知道真实未来、从 K 条候选中挑最好一条的理论上限，不能作为实际部署结果。
- 分类准确率提高但 ADE 变差并不矛盾：可能选对标签但候选轨迹几何质量下降，或为了小类扰动了共享主干。

## 15. 已知风险与下一步

1. 先等待 v15e 完成，必须与 v14 使用相同固定测试集比较总体 ADE/FDE、各子类 ADE/FDE、candidate selector 和 oracle ADE@K。
2. OA_S00 的历史可判别性接近 0，说明很多窗口在分叉前本来无法区分；不要只看其硬分类准确率。
3. 如果 v15e 总体仍退化，先检查候选 oracle 是否改善：oracle 变好而最终不变是选择器问题；oracle 也变差是生成器问题。
4. 正式论文建议增加“九、十月所有航路全量处理”数据集，而不是只补 OA_S00。
5. 论文至少需要固定测试集、多个随机种子或交叉验证、公开基线、模块消融和统计波动，单次最好结果不能直接证明可发表。
6. 推荐消融顺序：v14 基线、+Qwen semantic、+staged uncertainty、+candidate selector、+residual experts；每次只改变一个模块。
7. 多模态结果除 Top-1 ADE/FDE 外，建议同时报告 minADE@K、minFDE@K、Miss Rate、候选覆盖率和概率校准。

## 16. 代码与文件状态注意事项

- 工作树存在大量尚未提交的历史修改，不要使用 `git reset --hard` 或覆盖用户改动。
- `utils\qwen_candidate_reranker.py` 和 `utils\diagnose_reproduction.py` 已删除，是此前清理无效/重复模块的一部分。
- 冒烟测试输出应放在 `.tmp`，不要再污染正式 `results`。
- 现有说明文档：
  - `docs\fixed_train_valid_test_split.md`
  - `docs\qwen_semantic_teacher.md`
  - `docs\route_subroute_training_notes.md`
  - `docs\iTentformer_DMA数据处理与模型创新改动说明.docx`
  - `docs\iTentformer_新增模块与联动速记.docx`

## 17. 新对话开场提示词

在新对话中可直接发送：

```text
请先阅读 D:\iTentformer-master\docs\conversation_handoff.md，接手当前 iTentformer DMA 三小时轨迹预测项目。先检查正在运行的 dma_v15e_3h 日志和进程，不要重新处理数据或启动第二个训练；然后基于相同固定测试集与文档中的 v14 结果比较。
```
