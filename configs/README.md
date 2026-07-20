# 三组正式训练参数

三组实验使用同一数据集、固定训练/验证/测试划分和随机种子，只改变表中的训练参数。

| 组别 | 学习率调度耐心值 | 子航路残差强度 | 均衡意图损失权重 | 当前状态 |
| --- | ---: | ---: | ---: | --- |
| 第一组（默认） | 3 | 0.25 | 0.35 | 已完整训练并完成一次最终测试 |
| 第二组 | 5 | 0.20 | 0.35 | 正式训练中途停止，建议以后从头重跑 |
| 第三组 | 5 | 0.25 | 0.20 | 只完成过筛选训练，尚未进行完整训练 |

## 运行命令

第一组已经写入 `config_iTentformer.py`，直接运行：

```powershell
python iTentformer.py
```

第二组：

```powershell
python iTentformer.py --config configs/config_group2_expert020.py
```

第三组：

```powershell
python iTentformer.py --config configs/config_group3_balance020.py
```

每个预设都有独立的 `model_prefix`，不会覆盖其他组的模型文件。第二组中途保存的 checkpoint 仍保留在：

```text
tuning_results/dma_v15_auto_tune/checkpoints/final_02_screen_05_expert020_fixed.pt
```

该 checkpoint 不是完整实验结果。为了与第一组公平比较，应按上面的命令从头训练，并且只在验证集完成选模后执行一次最终测试。

第二组当时使用的完整参数快照：

```text
tuning_results/dma_v15_auto_tune/configs/final_02_screen_05_expert020.json
```

第三组尚未开始正式训练，但筛选阶段的参数和 checkpoint 均已保留：

```text
tuning_results/dma_v15_auto_tune/configs/screen_06_balance020.json
tuning_results/dma_v15_auto_tune/checkpoints/screen_06_balance020_fixed.pt
```

筛选阶段使用稀疏窗口，只适合辅助选参数，不应直接与第一组的完整训练结果比较。

第一组已经验证的 checkpoint 位于：

```text
tuning_results/dma_v15_auto_tune/checkpoints/final_01_screen_02_lr_pat3_fixed.pt
```
