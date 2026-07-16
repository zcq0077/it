# 固定训练/验证/测试划分

默认配置使用传统固定划分，只训练一次：

```text
全部轨迹
├── 训练集 70%
├── 验证集 10%
└── 测试集 20%
```

划分以完整轨迹为单位，并按 MMSI 分组。同一 MMSI 不会同时进入训练、验证和测试集合。划分时还会尽量保持各子航路的比例。

关键配置：

```python
split_mode = "fixed"
test_ratio = 0.20
valid_ratio = 0.125  # 剩余80%中的12.5%，相当于总数据的10%
split_seed = 42
split_manifest_path = "dataset/dma_raw_2023_06_07_08/dma_fixed_split_70_10_20_seed42.json"
```

第一次运行会生成 `split_manifest_path` 指定的 JSON。里面记录三个集合的轨迹索引和 MMSI；以后训练其他模型时会直接读取该文件，不再重新随机划分。

当前清单的实际轨迹数为：训练 `3399` 条、验证 `484` 条、测试 `971` 条，对应 `70.02% / 9.97% / 20.00%`。三个集合之间的 MMSI 交集为零。标准化参数也只使用训练集计算。

直接训练：

```powershell
python iTentformer.py
```

固定模式的模型保存名以 `_fixed.pt` 结尾。日志内部仍会显示 `1/1`，它只表示唯一一次训练流程，不代表交叉验证折数。

需要恢复五折交叉验证时：

```python
split_mode = "kfold"
folds = 5
run_folds = 5
valid_ratio = 0.10
```

固定划分的单次结果可以作为标准 train/valid/test 实验，但不是五折平均结果。比较不同模型时必须共用同一个划分清单。
