# iTentformer 自动调参

## 目的

`utils/auto_tune_itentformer.py` 使用固定的训练、验证、测试划分自动比较参数。
搜索阶段只计算验证集，目标函数为：

```text
Validation ADE + 0.2 * Validation FDE
```

测试集不参与参数排名。所有候选完成后，只对验证集最优的完整模型测试一次。

## 搜索流程

1. 使用 `window_stride=4`、24轮训练筛选8组参数。
2. 比较学习率调度耐心值、残差专家强度和小类辅助损失权重。
3. 筛选阶段排名前3名使用 `window_stride=1`、最多50轮重新完整训练。
4. 按完整训练的验证集目标选择最终模型。
5. 加载胜出 checkpoint，对测试集评估一次并生成正式图片。

筛选结果只用于淘汰明显较差的参数。最终胜者一定经过完整窗口重新训练，不能把筛选阶段的稀疏窗口指标作为论文结果。

## 当前任务

当前研究目录：

```text
tuning_results/dma_v15_auto_tune/
```

重要文件：

```text
pipeline.log                 总体进度
leaderboard.csv              已完成实验排行榜
study.json                   可恢复的任务状态
configs/                     每个实验的完整配置
runs/<trial>/train.log       每个实验训练日志
checkpoints/                 每个实验最佳验证 checkpoint
summary.md                   完成后的最优参数和最终结果
```

查看进度：

```powershell
Get-Content tuning_results\dma_v15_auto_tune\pipeline.log -Tail 20
```

查看当前实验：

```powershell
Get-Content tuning_results\dma_v15_auto_tune\runs\screen_01_baseline\train.log -Tail 20
```

任务意外中断后，可以直接恢复：

```powershell
python utils\auto_tune_itentformer.py --study-name dma_v15_auto_tune
```

已完成的实验会自动跳过，失败或未完成的实验会重新运行。

## 注意

这里得到的是当前搜索范围和计算预算内的验证集最优解，不是数学意义上的全局最优解。论文正式报告时，建议再对胜出配置运行3个训练随机种子，报告均值和标准差。
