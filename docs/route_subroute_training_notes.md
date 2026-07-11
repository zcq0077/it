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

冒烟测试输出固定在：

```text
.tmp/smoke_results
.tmp/smoke_models
```

不会污染正式的 `results` 和 `save_models`。

## 10. 需要注意的限制

1. 主航路和子航路标签来自规则与聚类，不是人工逐条标注的绝对真值。
2. 分支发生前的历史窗口本身可能具有不确定性，此时模型难以准确判断最终子航路是正常现象。
3. 子航路数量越多，小类样本越少，分类和预测都会变难。
4. 当前策略是提高小分支参与度，而不是强行让模型一定预测小分支。
5. 正式比较实验效果时，应固定数据集、折数、随机种子，并比较同一折的 ADE/FDE、子航路准确率和可视化结果。
