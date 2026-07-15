# DMA 主航路、子航路划分与训练改进说明

本文档说明本项目在原始 iTentformer 复现基础上新增的 DMA 航路划分、子航路细化标签，以及训练阶段为提高分支航路预测效果而加入的辅助模块和损失函数。

当前默认配置位于 `config_iTentformer.py`，默认数据集为 2023 年 6 月、7 月、8 月 DMA 合并数据：

```text
dataset/dma_raw_2023_06_07_08/dma_itentformer_ti_4class_revnorm_lasthit.pkl
dataset/dma_raw_2023_06_07_08/dma_route_labels_ti_4class_revnorm_lasthit.json
dataset/dma_raw_2023_06_07_08/dma_subroutes_ti_4class_local_fused_v4_labels.json
```

## 1. 主航路划分

主航路用于把 DMA 船舶轨迹按典型目的方向分成粗粒度路线。当前共 4 类：

| 类别 | 含义 |
| --- | --- |
| `OA` | 从 O 区域驶向 A 区域的主航路 |
| `OB1` | 从 O 区域驶向 B1 区域的主航路 |
| `OB2` | 从 O 区域驶向 B2 区域的主航路 |
| `OC` | 从 O 区域驶向 C 区域的西侧竖向典型航路 |

主航路识别不是模型自己猜出来的第一步，而是数据预处理阶段根据轨迹经过的空间门限生成标签。当前门限来自论文图中区域的近似数字化，代码中保存在 `utils/preprocess_dma_zip.py` 的 `DEFAULT_ROUTE_GATES`。

| 区域 | 经度范围 | 纬度范围 |
| --- | --- | --- |
| `O` | 10.30 - 11.10 | 57.35 - 57.85 |
| `TI` | 11.65 - 12.10 | 56.45 - 56.85 |
| `A` | 12.10 - 12.75 | 55.90 - 56.35 |
| `B1` | 11.20 - 11.90 | 56.05 - 56.35 |
| `B2` | 10.95 - 11.65 | 56.30 - 56.65 |
| `C` | 10.35 - 11.20 | 55.65 - 56.15 |

当前主航路分类采用以下策略：

- `endpoint_policy=last_hit`：当一条轨迹可能经过多个目的区域时，以最后命中的目的区域作为主航路终点。
- `include_direct_c_route=True`：保留 O 到 C 的竖向典型航路，避免把西侧直航路线误删。
- `reverse_mode=normalize`：如果出现反向航行，例如 A 到 O，会归一化为对应的 O 到 A 方向，减少方向标签数量，让模型先学习统一的典型航路形态。

当前 6+7+8 月合并后的主航路数量为：

| 主航路 | 轨迹数 |
| --- | ---: |
| `OA` | 1498 |
| `OB1` | 2188 |
| `OB2` | 643 |
| `OC` | 525 |
| 合计 | 4854 |

## 2. 子航路划分

主航路只能表示大方向，但在同一主航路内部仍然可能存在明显分支。例如 OA 中途存在不同弯曲路线，OC 内部也有不同竖向通道。为了让模型学习这些细粒度分支，新增了子航路标签。

子航路由 `utils/discover_subroutes.py` 生成。它不是只看起点和终点，而是综合整条轨迹形态进行聚类，包含：

- 轨迹重采样到固定长度，当前为 `32` 个点。
- 使用经纬度形态特征，并融合相对形态，避免只按绝对位置机械分组。
- 可选使用运动特征，当前 `include_motion=True`。
- 对明显中途分叉的路线使用局部窗口加权，当前 OA 使用 `0.38:0.72` 的中段窗口，并设置 `route_window_feature_weight=3.0`。
- 对小类数量设置下限，当前 `min_subroute_size=40`，避免为了追求更多类别而切出极小噪声类。

当前参数中，OA 和 OC 被强制细分：

```text
force_route_k: OA=3, OC=4
```

OB1 和 OB2 则按轮廓系数自动选择较合理的类别数。当前结果为：

| 主航路 | 子航路数 | 子航路数量 |
| --- | ---: | --- |
| `OA` | 3 | `OA_S00=159`, `OA_S01=713`, `OA_S02=626` |
| `OB1` | 2 | `OB1_S00=2018`, `OB1_S01=170` |
| `OB2` | 2 | `OB2_S00=566`, `OB2_S01=77` |
| `OC` | 4 | `OC_S00=152`, `OC_S01=65`, `OC_S02=69`, `OC_S03=239` |

### Compact6 精简标签

当前默认实验使用 `compact6_v1`，原始 11 类标签仍保留用于消融对照。精简规则只保留
已经确认存在中途分叉的三条 OA 子航路；OB1、OB2 和 OC 中主要由轨迹覆盖长度、起终点
截断造成的细分类别分别合并为一个子航路：

| 主航路 | 精简后的子航路及轨迹数 |
|---|---|
| `OA` | `OA_S00=159`, `OA_S01=713`, `OA_S02=626` |
| `OB1` | `OB1_S00=2188` |
| `OB2` | `OB2_S00=643` |
| `OC` | `OC_S00=525` |

合并后的标签文件为
`dataset/dma_raw_2023_06_07_08/dma_subroutes_ti_4class_compact6_v1_labels.json`。
可提交的映射保存在 `utils/subroute_compact6_mapping.json`，统计报告与生成标签放在数据目录，
因此能够随时恢复到原始 11 类版本或重新生成精简标签。

可视化结果保存在：

```text
results/dma_subroutes_06_07_08/
```

其中：

- `*_subroute_overlay.png`：所有子航路叠加图。
- `*_subroute_panels.png`：按主航路分面展示。
- `*_branch_focus_diagnostics.png`：分支聚类诊断图。

## 3. 标签在模型中的使用方式

主航路和子航路标签不会作为真实输入特征直接喂给模型。模型推理时仍然只能看到历史 AIS 数值序列，包括经纬度、航向、速度及其变化量。

新增标签主要作为训练阶段的辅助监督信号：

1. 模型根据历史轨迹预测主航路概率。
2. 模型根据历史轨迹预测子航路概率。
3. 预测得到的主航路、子航路概率再转换为 embedding，作为软条件反馈给轨迹预测分支。
4. 轨迹预测分支最终仍输出未来 COG、Lon、Lat、SOG。

这种设计的目的是让模型在训练时学会“当前历史轨迹更像哪条航路”，而不是在推理时手工指定真实类别。因此如果分支判断正确，预测轨迹会更贴近对应的典型航路；如果分支判断不确定，模型会以概率形式融合多条可能分支。

## 4. 新增训练模块

### 4.1 主航路辅助头

配置：

```python
use_route_intent_head = True
route_intent_weight = 0.2
use_route_embedding = True
route_embedding_dim = 16
```

作用：

- 用交叉熵训练模型判断 `OA/OB1/OB2/OC`。
- 将预测出的主航路概率转成 embedding，融入轨迹预测分支。
- 帮助模型先区分大方向，减少多模态航路混在一起造成的平均化预测。

### 4.2 子航路辅助头

配置：

```python
use_subroute_intent_head = True
subroute_intent_weight = 0.35
use_subroute_embedding = True
subroute_embedding_dim = 16
```

作用：

- 在主航路内部继续判断具体小分支，例如 `OA_S00/OA_S01/OA_S02`。
- 将模型预测的子航路概率反馈给轨迹预测分支。
- 让同一主航路中的不同弯曲、分叉路线被模型区别对待。

### 4.3 层级意图约束

配置：

```python
use_hierarchical_intent = True
hierarchical_mask_strength = 1.5
```

作用：

- 主航路概率会约束子航路概率。
- 例如模型判断大类更像 `OA` 时，`OA_S00/OA_S01/OA_S02` 会更容易被选择，`OC_Sxx` 会被压低。
- 这样可以避免子航路头跨大类乱跳。

注意：约束强度不能过大。如果主航路判断错，过强的约束会把子航路也带错。当前 `1.5` 是偏温和设置。

## 5. 新增损失函数

### 5.1 地理距离损失

配置：

```python
use_geo_loss = True
geo_weight = 0.2
geo_loss_scale = 10.0
```

作用：

- 直接约束经纬度预测点之间的 Haversine 距离。
- 比单纯标准化 MSE 更贴近真实地理误差。

### 5.2 FDE 损失

配置：

```python
use_fde_loss = True
fde_weight = 0.5
```

作用：

- 加强最后一个预测点的误差约束。
- 对航路终点偏差较大的情况尤其有帮助。

### 5.3 平滑损失

配置：

```python
use_smooth_loss = True
smooth_weight = 0.2
```

作用：

- 约束预测轨迹的相邻点变化趋势，减少折返、锯齿、乱跳。
- 权重不能太大，否则可能把真实转弯过度拉直。

### 5.4 COG 圆周角损失

配置：

```python
use_circular_cog = True
cog_weight = 0.2
cog_loss_scale = 180.0
```

作用：

- 解决航向角 `359°` 和 `1°` 实际只差 `2°`，但普通 MSE 会认为差 `358°` 的问题。

### 5.5 子航路对比损失

配置：

```python
use_subroute_contrastive_loss = True
subroute_contrastive_weight = 0.05
subroute_contrastive_temperature = 0.2
```

作用：

- 同一子航路的隐藏特征被拉近。
- 不同子航路的隐藏特征被拉远。
- 让模型内部表征更清楚地区分不同小分支。

### 5.6 子航路 Focal Loss

配置：

```python
use_subroute_focal_loss = True
subroute_focal_gamma = 1.5
subroute_label_smoothing = 0.02
```

作用：

- 普通交叉熵容易被大类和简单样本主导。
- Focal Loss 会降低已经容易分对样本的梯度，让模型更关注难分样本和小分支样本。
- `label_smoothing=0.02` 用于降低过拟合和过度自信。

当前 `gamma=1.5` 是温和设置。如果小分支仍然分不好，可以尝试提高到 `2.0`；如果主航路效果下降，则应调回 `1.0` 或关闭 focal。

## 6. 小样本航路的训练改进

当前小样本问题主要出现在：

```text
OC_S01 = 65
OC_S02 = 69
OB2_S01 = 77
OA_S00 = 159
OB1_S01 = 170
```

这些类别如果按原始分布训练，出现频率远低于 `OB1_S00=2018` 这种大类，模型容易倾向预测大类，导致小分支被忽略。

为此加入了三层小样本增强策略。

### 6.1 子航路 class weight

配置：

```python
use_subroute_class_weight = True
subroute_class_weight_alpha = 0.5
subroute_class_weight_max_ratio = 5.0
```

作用：

- 根据子航路频次给小类更高权重。
- 大致形式为 `(max_count / class_count) ** alpha`。
- `max_ratio=5.0` 限制最高放大倍数，防止极小类噪声主导训练。

这只作用在子航路分类辅助头上，不直接改变轨迹回归 loss。

### 6.2 子航路均衡采样

配置：

```python
use_balanced_subroute_sampling = True
balanced_sampling_alpha = 0.4
balanced_sampling_max_ratio = 5.0
balanced_sampling_mix_ratio = 0.4
```

作用：

- 每轮训练仍保持相同窗口数量。
- 其中约 `60%` 按原始分布随机采样，保留真实数据分布。
- 约 `40%` 使用子航路加权采样，让小类更多进入 batch。

这样做比完全均衡采样更稳。完全均衡可能损害大类主航路表现，而当前混合采样能兼顾小类和整体分布。

### 6.3 子航路分层 K 折

配置：

```python
stratify_by_subroute = True
```

作用：

- K 折划分时按子航路分层。
- 尽量保证每一折训练集和测试集里都有各个子航路。
- 避免某一折刚好缺少小分支，导致验证或测试结果不稳定。

## 7. 日志中如何观察小分支效果

训练日志中需要重点看以下几类信息。

### 7.1 子航路采样比例

日志示例：

```text
subroute sampling expected class ratio: OA_S00:..., OB2_S01:..., OC_S01:...
```

用途：

- 看小类是否真的被采样策略提高了参与度。
- 如果小类比例仍然太低，可以提高 `balanced_sampling_mix_ratio`。

### 7.2 子航路分类准确率

日志示例：

```text
Subroute_ACC_by_class: OA_S00:..., OC_S01:..., OB2_S01:...
```

用途：

- 看模型是否能从历史轨迹判断出当前属于哪条小分支。
- 如果小类长期为 `0%`，说明模型还没有学会该分支，可能需要更多数据、提高采样比例，或重新检查子航路标签是否过细/不稳定。

### 7.3 子航路轨迹误差

日志示例：

```text
Subroute_ADE_FDE_by_class: OA_S00:ADE .../FDE ..., OC_S01:ADE .../FDE ...
```

用途：

- 直接看每条小分支的轨迹预测误差。
- 这比单看总 ADE/FDE 更重要，因为总指标容易被大类掩盖。

## 8. 常用调参建议

### 8.1 早停与最佳模型保存

加入主航路、子航路、对比损失、Focal Loss、地理距离损失以后，`Valid loss` 已经不是单纯的轨迹误差。它里面混合了多个辅助目标，所以可能出现：

- `Valid loss` 没有下降；
- 但验证集 `ADE/FDE` 还在变好；
- 旧逻辑因为只看 loss，提前早停，并保存了并非轨迹指标最好的 epoch。

现在默认改为：

```python
early_stop_metric = "ade_fde"
early_stop_fde_weight = 0.2
patience = 10
```

也就是早停和最佳模型保存都监控：

```text
ADE + 0.2 * FDE
```

训练日志里重点看这一行：

```text
Early-stop monitor, fold 1/5, epoch 013, ADE+0.200*FDE ...
```

如果这项连续 `patience` 轮没有变小，才会早停。这样更适合当前多任务训练，避免“辅助 loss 抖动导致轨迹指标还没收敛就停”。

如果要回到原始行为，可以临时运行：

```powershell
python iTentformer.py --early_stop_metric loss
```

### 8.2 分支效果调参

如果小分支预测仍然差，可以按顺序尝试：

1. 增加数据月份，优先补充小类轨迹。
2. 把 `balanced_sampling_mix_ratio` 从 `0.4` 提到 `0.5`。
3. 把 `subroute_focal_gamma` 从 `1.5` 提到 `2.0`。
4. 把 `subroute_intent_weight` 从 `0.35` 提到 `0.4`。
5. 如果大类主航路效果下降，优先把 `balanced_sampling_mix_ratio` 降回 `0.3`。
6. 如果子航路分类准确率低且 ADE/FDE 也差，检查子航路聚类图，确认是否把同一物理航路切得过细。

不建议一开始就把小类采样做成完全均衡，因为这会让训练分布明显偏离真实航路分布，可能导致大类预测变差。

### 8.3 分支选择增强

当前默认启用了互相配合的层级分支选择模块：

```python
intent_summary_mode = "mean_last_delta"
branch_routing_temperature = 0.7
hard_subroute_routing = True
confidence_aware_routing = True
routing_confidence_threshold = 0.8
routing_margin_threshold = 0.35
routing_top_k = 2
use_branch_teacher_forcing = True
use_route_prototype_prior = True
use_subroute_prototype_prior = True
```

- `mean_last_delta`：子航路分类同时使用历史均值、最后状态和首尾变化，增强分岔口附近的识别能力。
- `hard_subroute_routing`：允许高置信度样本明确使用概率最高的子航路。
- `confidence_aware_routing`：只有第一名概率和领先差值同时达标时才硬选择；否则保留 Top-2 概率，避免共享航段上的暂时误判锁死未来轨迹。
- `branch_teacher_forcing`：训练前期较多使用真实分支引导轨迹解码，随后逐步切换为模型预测分支。
- `route_prototype_prior`：每一折只从训练集构建 OA/OB1/OB2/OC 主航路原型，先校正大类判断。
- `subroute_prototype_prior`：每一折只从该折训练集构建平均航路原型，利用当前位置和行进方向给候选子航路加几何分数，不使用验证集和测试集。

训练日志会显示：

```text
Branch routing: summary=mean_last_delta, temperature=0.700, hard_subroute=True, ...
Branch teacher forcing, fold 1/5, epoch 001, ratio 0.700.
Fold 1/5 built route prototypes from ... training tracks only, shape (4, 32, 2).
Fold 1/5 built subroute prototypes from ... training tracks only, shape (11, 32, 2).
Final Test Route_Routing: hard ..., uncertain_top2_recall ...
```

做消融实验时，可以分别关闭模块：

```powershell
python iTentformer.py --no-use_subroute_prototype_prior
python iTentformer.py --no-use_route_prototype_prior
python iTentformer.py --no-confidence_aware_routing
python iTentformer.py --no-hard_subroute_routing
python iTentformer.py --no-use_branch_teacher_forcing
python iTentformer.py --intent_summary_mode mean
```

注意：这些增强改变了模型结构和 checkpoint 内容，增强前保存的旧模型不能作为新模型继续训练；需要重新训练后再测试。

### 8.4 显式候选轨迹筛选

低置信度 Top-2 不再只做 embedding 平均。模型按“Top-2 主航路 × 每类 Top-2 子航路”生成四条强制分支轨迹，并与原始预测组成五个候选：

```text
安全候选：原置信度路由预测
分支候选 1/2：Top-1 主航路下的 Top-2 子航路
分支候选 3/4：Top-2 主航路下的 Top-2 子航路
```

训练时根据真实 `ADE + candidate_fde_weight * FDE` 标记五个候选中的优胜者，只用这个标签训练候选筛选器。筛选器输入不包含真实未来，只包含历史特征、候选分类概率、原型贴合度和运动连续性。

候选监督使用 `min_hav=0` 的真实 Haversine 距离。旧实现使用 `min_hav=1e-7` 会人为制造约 2.17 海里的最小距离，使本来很接近真值的候选无法获得更低代价，现已修正。

为了避免随机初始化的筛选器拖累主模型：

```python
candidate_selector_warmup_epochs = 10
candidate_trajectory_weight = 0.05
candidate_switch_confidence_threshold = 0.45
candidate_switch_logit_margin = 0.15
```

- 前 10 轮只训练筛选器，正式指标继续使用安全候选。
- 筛选器不通过候选轨迹损失扰动原预测器。
- 预热后也只有当分支候选概率足够高、并明显胜过安全候选时才允许切换。
- 推理最终仍只输出一条轨迹。

日志中的 `Candidate_Selector` 会报告优胜者选择准确率、Top-2 主航路召回率、Top-4 子航路召回率、实际切换比例和理论 Oracle@5 指标。

### 8.5 Qwen3-1.7B 候选重排

本地千问不是用自然语言直接生成经纬度，而是作为第二阶段候选比较器：

1. 主 iTentformer 按正常流程训练并加载验证集最优 checkpoint。
2. 从训练集抽取候选筛选器选错、基础轨迹不是最优、分类置信度不足的困难窗口。
3. 将 10 个历史点映射为数值 soft token，将每条候选的类别概率、运动连续性、原型距离、相对轨迹形态映射为候选 token。
4. 冻结 Qwen3-1.7B 全部骨干参数，只训练数值适配器和打分头。
5. 推理时仅对不确定窗口融合 Qwen 分数，最后仍由保守门控输出唯一一条轨迹。

默认配置：

```python
use_qwen_reranker = True
qwen_model_path = r"D:\Jason1982\wsl\Models\Qwen3-1.7B"
qwen_reranker_epochs = 3
qwen_train_max_windows = 4096
qwen_valid_max_windows = 1024
qwen_hard_sample_ratio = 0.7
qwen_reranker_weight = 0.5
qwen_cost_temperature = 0.25
qwen_cost_regression_weight = 0.2
qwen_calibration_max_apply_ratio = 0.35
qwen_uncertain_only = True
qwen_require_validation_gain = True
qwen_min_validation_gain = 0.5
qwen_min_validation_cost_gain = 0.005
```

千问原始权重不会复制进 iTentformer checkpoint。每折只额外保存一个轻量适配器：

```text
save_models/<model_prefix>_K1_qwen_reranker.pt
```

当前默认适配器约有 41 万个可训练参数。训练完成后，融合选择准确率必须在验证集至少提升 `0.5` 个百分点，并且验证集真实 `ADE + candidate_fde_weight * FDE` 不能变差，才会在正式测试启用；否则自动退回原筛选器，避免负优化。日志会分别报告 `qwen_applied`、`qwen_winner_acc`、候选轨迹代价和融合后的 `selector_winner_acc`。关闭千问做消融实验：

```powershell
python iTentformer.py --no-use_qwen_reranker
```

## 9. 复现实验命令

默认训练：

```powershell
python iTentformer.py
```

只跑一折调参：

```powershell
python iTentformer.py --run_folds 1
```

冒烟测试：

```powershell
.\run_smoke_test.ps1
```

默认冒烟测试关闭千问以保持速度；需要连同千问二阶段一起检查时运行：

```powershell
.\run_smoke_test.ps1 -WithQwen -WindowStride 100
```

冒烟测试输出固定在：

```text
.tmp/smoke_results
.tmp/smoke_models
```

不会污染正式的 `results` 和 `save_models`。

## 10. 三小时预测与Qwen V2

DMA数据每15分钟一个点。当前默认设置为：

```python
input_length = 10   # 2.5小时历史
target_length = 12  # 3小时未来
```

模型、滑窗、候选轨迹和绘图都会读取这两个参数。改变任一长度都会改变模型参数形状，旧的10点预测checkpoint不能直接用于12点预测，必须重新训练。

Qwen V2不再分别重复编码5条候选，而是一次读取航海规则提示、完整历史和全部候选后联合评分。除原有概率与轨迹形态外，还加入11项本地航路知识，包括中心线横向距离、航路进度、前进方向、航向连续性、竞争分支间距和速度变化。

训练目标由硬候选编号改为基于 `ADE + candidate_fde_weight * FDE` 的软排序。训练结束后，在独立验证窗口上自动搜索融合权重和门控阈值；只有实际运行选择准确率提升，并且轨迹代价改善的95%下置信界达到要求，Qwen才会在最终测试中启用。

## 11. AIS航次语义上下文

原始丹麦AIS ZIP还包含船型、船宽、船长、吃水、Destination、ETA和航行状态。使用下面的命令直接流式读取三个ZIP并生成与当前轨迹逐点对齐的侧车文件，不需要解压：

```powershell
python utils\build_dma_voyage_context.py `
  --input_zip D:\AIS\2023_06_09\aisdk-2023-06.zip `
  --input_zip D:\AIS\2023_06_09\aisdk-2023-07.zip `
  --input_zip D:\AIS\2023_06_09\aisdk-2023-08.zip `
  --data_path dataset\dma_raw_2023_06_07_08\dma_itentformer_ti_4class_revnorm_lasthit.pkl `
  --labels_path dataset\dma_raw_2023_06_07_08\dma_route_labels_ti_4class_revnorm_lasthit.json `
  --output_path dataset\dma_raw_2023_06_07_08\dma_voyage_context_2023_06_07_08.pkl
```

脚本只会为当前轨迹涉及的MMSI和时间区间保留上下文。每个轨迹点只取该时刻之前最后已知的信息，并限制最大陈旧时间为24小时，避免读取未来更新的Destination。

生成完成后修改配置：

```python
voyage_context_path = "dataset/dma_raw_2023_06_07_08/dma_voyage_context_2023_06_07_08.pkl"
group_folds_by_mmsi = True
qwen_context_max_tokens = 64
qwen_context_dropout = 0.15
```

Qwen会读取航次语义文本、10点历史运动、11项航路几何特征和全部5条候选，再输出候选排序分数。训练时随机遮蔽15%的语义上下文，防止模型过度依赖错误或缺失的Destination。MMSI本身不会输入模型，只用于防泄漏分组。

建议使用同一套MMSI分组运行消融实验：

```powershell
# 纯运动/航路几何基线
python iTentformer.py --voyage_context_path none --model_prefix grouped_motion_baseline

# 加入AIS航次语义与Qwen联合重排
python iTentformer.py `
  --voyage_context_path dataset\dma_raw_2023_06_07_08\dma_voyage_context_2023_06_07_08.pkl `
  --model_prefix grouped_voyage_semantic_qwen
```

两组都保持 `group_folds_by_mmsi=True`，否则同一船舶跨训练/测试会使语义实验结果虚高。

## 12. 需要注意的限制

1. 主航路和子航路标签来自规则与聚类，不是人工逐条标注的绝对真值。
2. 分支发生前的历史窗口本身可能具有不确定性，此时模型难以准确判断最终子航路是正常现象。
3. 子航路数量越多，小类样本越少，分类和预测都会变难。
4. 当前策略是提高小分支参与度，而不是强行让模型一定预测小分支。
5. 正式比较实验效果时，应固定数据集、折数、随机种子，并比较同一折的 ADE/FDE、子航路准确率和可视化结果。

## 13. 分叉前未决、分叉后确定的分阶段监督

原来的主航路和子航路标签都属于整条轨迹。旧训练方式会把最终标签复制给轨迹产生的每个窗口，因此共享主干上的相似历史可能对应不同硬标签。当前版本增加主航路与子航路可判别性门控，只用当前折训练集建立的航路原型和窗口历史段计算连续权重，不读取该窗口的未来坐标。

1. 历史仍位于共享主干时，降低最终类别硬标签、对比损失和标签教师强制，并学习候选类别之间的软分布。
2. 历史接近或进入分叉后，逐步恢复最终类别硬监督，并允许真实分支进入候选训练。
3. 轨迹回归损失始终使用全部窗口，不删除主航路数据。
4. 候选选择器只有在大类和小类都达到可判别阈值时，才会在训练候选池中注入真实分支。
5. 验证与测试额外报告 `Route_Staged` 和 `Subroute_Staged`：可判别窗口看 Top-1，不可判别窗口看 Top-K recall。

默认配置如下：

```python
use_route_decidability = True
route_decidable_min_weight = 0.05
route_decidable_confidence_threshold = 0.60
route_decidable_margin_threshold = 0.10
route_decidable_threshold = 0.50
route_undecidable_soft_weight = 0.10

use_subroute_decidability = True
subroute_decidable_min_weight = 0.05
subroute_decidable_confidence_threshold = 0.60
subroute_decidable_margin_threshold = 0.10
subroute_decidable_threshold = 0.50
subroute_undecidable_soft_weight = 0.15
```

## 14. 抑制过度自信的学习型门控

几何可判别性用于生成训练监督，但部署时没有真实标签，因此模型还需要从历史轨迹自行判断“现在是否已经能选航路”。当前版本增加两个轻量可判别头，并把硬路由改成三重条件：类别最高概率、前两名间隔、可判别概率都达到阈值才硬选；任一条件不足就保留 Top-2。

主航路和子航路使用独立温度。主航路温度较高，用来缓和旧实验中高硬选率但低正确率的问题；子航路温度略低，在已经出现分叉趋势后仍保留足够区分度。层级约束也改为动态强度，大类不可靠时不会强行压低其他大类下的小分支。

```python
route_routing_temperature = 1.30
subroute_routing_temperature = 0.90

use_learned_decidability = True
route_decidability_gate_threshold = 0.65
subroute_decidability_gate_threshold = 0.60
route_decidability_loss_weight = 0.10
subroute_decidability_loss_weight = 0.10

confidence_gated_hierarchy = True
hierarchy_min_scale = 0.15
```

验证日志新增三类诊断：

1. `Route/Subroute_Routing`：真正送入生成器的硬选覆盖率与正确率。
2. `Route/Subroute_Calibration`：`ECE`、`Brier`、`NLL`，数值越低表示概率越可信。
3. `Route/Subroute_Decidability`：可判别头的门控比例、精确率、召回率和连续目标 MAE。

调参时优先观察 `hard_top1_acc` 是否上升和 `ECE` 是否下降，不以硬选覆盖率越高越好。门控过严会降低覆盖率但通常保护 ADE；门控过松会重新出现错误分支被高置信度注入生成器的问题。

## 15. Qwen 子航路反事实反推

当前版本不让 Qwen 直接生成经纬度，而是把它作为候选子航路的反事实验证器。对于每条候选，系统提出问题：“如果未来确实走这条子航路，那么沿该航路原型倒着追溯，能否解释已经观测到的完整历史？”

首先使用 `candidate_pool_strategy="all_subroutes"`。紧凑版只有 6 个子航路，因此全部进入候选池并自动绑定所属主航路，避免真实小分支因先验概率低而在 Qwen 评分前就被删掉。

每条候选新增 16 项反推证据：完整及最近历史到中心线的距离、横向接近趋势、沿航路进度及单调性、历史速度与局部切线的一致性、候选初始速度反向外推后的历史重构误差、速度与转向连续性，以及同一主航路下兄弟子路之间的历史匹配差。所有证据只使用预测时已经存在的历史、训练集航路原型和候选未来，不读取真实未来。

训练阶段仍用真实未来的候选 ADE/FDE 确定监督排序，同时增加成对排序损失。主选择器在同一主航路内选错子路的样本会被额外加权，使 Qwen 重点学习 `OA_S00/OA_S01/OA_S02` 这种难分支，而不是只学习容易区分的 OA 与 OC。

```python
candidate_pool_strategy = "all_subroutes"
candidate_max_subroutes = 8

qwen_pairwise_weight = 0.4
qwen_pairwise_margin = 0.3
qwen_pairwise_min_cost_gap = 0.01
qwen_same_route_hard_weight = 1.5
```

推理时真实未来不会输入 Qwen。Qwen 分数仍需通过独立验证集校准；只有融合后的候选胜者准确率和轨迹代价达到增益要求，最终测试才启用。若共享主干上的历史对所有子路反推结果都相同，Qwen 应保持不确定，而不能凭空确定未来分支。
