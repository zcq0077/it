"""iTentformer experiment config.

这个文件就是 iTentformer 的默认参数总表。现在直接运行：

    python iTentformer.py

会自动读取本文件。命令行仍然可以临时覆盖这里的参数，例如：

    python iTentformer.py --epochs 1 --run_name smoke

当前默认数据集是 2023 年 6 月 + 7 月 + 8 月 DMA 四类航路数据：

    OA  : 从 O 区域到 A 区域的航路
    OB1 : 从 O 区域到 B1 区域的航路
    OB2 : 从 O 区域到 B2 区域的航路
    OC  : 从 O 区域到 C 区域的西侧竖向典型航路

注意：这些航路类别不会作为输入特征直接喂给模型；
但打开层级意图模块后，它们会作为辅助监督信号，帮助模型学习“大航路 -> 小分支 -> 未来轨迹”。
"""


class Config:
    # ======================================================================
    # 1. 数据集与航路标签
    # ======================================================================
    # 模型要读取的 pkl 数据。里面每条轨迹已经处理成 iTentformer 需要的 15 列格式：
    # [MMSI, Length, Course, Lon, Lat, SOG, vx, vy,
    #  delta_Course, delta_Lon, delta_Lat, delta_SOG, delta_vx, delta_vy, UnixTime]
    data_path = "dataset/dma_raw_2023_06_07_08/dma_itentformer_ti_4class_revnorm_lasthit.pkl"

    # 每条轨迹对应的航路类别标签。当前包含 OA / OB1 / OB2 / OC 四类。
    # 这个文件不是模型输入特征，主要用于：
    # 1. K 折划分时尽量让每折都有各类航路；
    # 2. 测试集可视化时按类别均衡抽样；
    # 3. 日志里统计每类轨迹数量，方便检查数据是否偏。
    route_labels_path = "dataset/dma_raw_2023_06_07_08/dma_route_labels_ti_4class_revnorm_lasthit.json"

    # True：按航路类别分层 K 折，推荐打开。
    # False：普通随机 K 折，可能出现某一折里某类航路很少。
    stratify_by_route = True

    # 大类航路分类辅助头。模型会先学习当前历史轨迹更像 OA/OB1/OB2/OC 哪一类。
    # 注意这里不是只看起点，而是看已观测历史段里的起点、当前位置、航向、速度和变化趋势。
    use_route_intent_head = True
    route_intent_weight = 0.2

    # 大类航路 embedding 融合。打开后，大类概率会反馈给轨迹预测分支。
    use_route_embedding = True
    route_embedding_dim = 16

    # 细分子航路标签，由 utils/discover_subroutes.py 生成。
    # 它会把 OA/OB1/OB2/OC 继续细化为 OA_S00、OB1_S01、OC_S02 等小分支。
    subroute_labels_path = "dataset/dma_raw_2023_06_07_08/dma_subroutes_ti_4class_local_fused_v4_labels.json"

    # True：K 折划分时按子航路分层，比只按 OA/OB1/OB2/OC 更细。
    # 如果子航路数量太多且某些类样本很少，可以改回 False。
    stratify_by_subroute = True

    # 子航路分类辅助头。打开后，模型训练时会同时学习“当前历史轨迹属于哪个小分支”。
    use_subroute_intent_head = True
    subroute_intent_weight = 0.35

    # 子航路 embedding 融合。模型会把自己预测的子航路概率转成 embedding，
    # 再融合回轨迹预测分支，让“像哪条小航路”真正影响未来轨迹预测。
    use_subroute_embedding = True
    subroute_embedding_dim = 16

    # 层级意图约束。大类概率会约束小类概率：
    # 例如大类更像 OA 时，OA_S00/OA_S01/OA_S02 会更容易被选中，OC_Sxx 会被压低。
    # strength 越大约束越强；太大时如果大类判断错，会拖累小类，所以默认用温和强度。
    use_hierarchical_intent = True
    hierarchical_mask_strength = 1.5

    # 子航路对比损失。同一子航路的隐藏特征会被拉近，不同子航路会被拉远。
    use_subroute_contrastive_loss = True
    subroute_contrastive_weight = 0.05
    subroute_contrastive_temperature = 0.2

    # Subroute focal loss: hard/rare branch windows get a little more gradient.
    # Keep gamma modest; too large can make the model chase noisy branch labels.
    use_subroute_focal_loss = True
    subroute_focal_gamma = 1.5
    subroute_label_smoothing = 0.02

    # 子航路分类 class weight。只加在子航路分类辅助头上，不直接改变轨迹回归 loss。
    # alpha 越大越照顾小类；0 表示不按频次加权，1 表示强逆频次加权。
    # max_ratio 限制小类最多被照顾到多少倍，避免小类噪声把模型带偏。
    use_subroute_class_weight = True
    subroute_class_weight_alpha = 0.5
    subroute_class_weight_max_ratio = 5.0

    # 温和子航路均衡采样。每轮训练仍保持同样窗口数，只把其中一部分替换成按小类加权抽样。
    # mix_ratio=0.3 表示约 70% 仍按原始分布训练，30% 用均衡抽样照顾小类。
    use_balanced_subroute_sampling = True
    balanced_sampling_alpha = 0.4
    balanced_sampling_max_ratio = 5.0
    balanced_sampling_mix_ratio = 0.4
    # Current default: 60% natural windows + 40% subroute-balanced replacement.

    # ======================================================================
    # 2. 实验输出位置
    # ======================================================================
    # 保存模型文件时用的前缀。最终通常类似：
    # save_models/dma_2023_06_07_08_ti_4class_K1.pt
    model_prefix = "dma_2023_06_07_08_ti_4class"

    # 模型 checkpoint 保存目录。
    model_dir = "save_models"

    # 日志、预测轨迹图、实验结果保存目录。
    results_dir = "results"

    # 单次运行的实验名。
    # None：自动生成，格式大概是 模型前缀-数据集名-时间戳。
    # 如果想固定名字方便找结果，可以改成 "my_exp_01"。
    run_name = None

    # 训练日志文件名，会保存在 results_dir/run_name/ 下面。
    log_file = "train.log"

    # False：每次运行重新写日志。
    # True：追加到已有日志后面。一般保持 False，避免日志混在一起。
    append_log = False

    # ======================================================================
    # 3. 训练 / 测试模式
    # ======================================================================
    # False：正常训练，然后在测试集上评估并画图。
    # True：不训练，只加载 checkpoint_path 或默认模型文件做测试。
    eval_only = False

    # eval_only=True 时加载的模型路径。
    # None：默认加载 model_dir/model_prefix_K当前折.pt。
    # 例如："save_models/dma_2023_06_07_08_ti_4class_K1.pt"
    checkpoint_path = None

    # ======================================================================
    # 4. K 折、训练轮数和数据窗口
    # ======================================================================
    # 总共划分多少折。论文常见是 5 折。
    # 注意：folds 必须 >= 2；如果只想跑一折，不要把这里改成 1，
    # 而是保持 folds = 5，然后把下面的 run_folds = 1。
    folds = 5

    # 实际跑几折。
    # 1：只跑第 1 折，省时间，适合调参。
    # 5：完整跑 5 折，更适合正式报告结果。
    # 0：只检查配置和数据能不能读取，不训练。
    run_folds = 1

    # 最大训练轮数。早停打开后，不一定会跑满。
    epochs = 50

    # 早停耐心值：验证集连续多少轮不提升就停。
    # 小数据/调参可以设 3-5；正式训练可设 8-15。
    patience = 10

    # 早停和最佳模型保存的监控指标。
    # loss：监控验证总 loss，适合单任务训练；
    # ade：监控验证 ADE；
    # ade_fde：监控 ADE + early_stop_fde_weight * FDE，更适合当前多任务轨迹预测。
    early_stop_metric = "ade_fde"
    early_stop_fde_weight = 0.2

    # 验证集比例。推荐用比例，不用再根据数据集大小手动改验证集数量。
    # 0.1 表示：每一折先分出测试集后，再从剩余训练候选轨迹里拿 10% 做验证集。
    # 以当前 4854 条轨迹、5 折为例，每折训练候选约 3883 条，验证集约 388 条。
    valid_ratio = 0.1

    # 固定验证集条数。只有当 valid_ratio = None 时才会使用这个参数。
    # 小样本复现实验想完全固定验证集数量时，可以设 valid_ratio = None，然后改这里。
    valid_count = None

    # 滑动窗口步长。
    # 1：窗口最密，训练样本最多，但训练更慢；
    # 20：窗口更稀，训练快，但样本少，容易学不充分。
    window_stride = 1

    # ======================================================================
    # 5. 测试集可视化
    # ======================================================================
    # 每折测试结束后画多少张 “历史轨迹 / 真实未来 / 预测未来” 对比图。
    # 0：不画图。
    plot_count = 16

    # 图片保存到 results_dir/run_name/plot_dir/。
    plot_dir = "plots"

    # first：直接画测试集前 plot_count 条；
    # route_balanced：按 OA/OB1/OB2/OC 尽量均衡抽图；
    # subroute_balanced：按 OA_S00/OC_S01 等子航路尽量均衡抽图，适合检查细分分支。
    plot_strategy = "subroute_balanced"

    # ======================================================================
    # 6. 预测目标形式
    # ======================================================================
    # absolute：直接预测未来点的 Course/Lon/Lat/SOG。
    # residual_linear：先按最后的运动趋势外推，再让模型学残差。
    # 对 AIS 轨迹来说 residual_linear 通常更稳，尤其能减少乱跳。
    target_mode = "residual_linear"

    # ======================================================================
    # 7. 可开关的优化损失
    # ======================================================================
    # 地理距离损失。让经纬度预测更贴近真实位置，推荐打开。
    use_geo_loss = True

    # 地理距离损失权重。太大可能压过 MSE，太小作用不明显。
    geo_weight = 0.2

    # 地理损失缩放项。数值越大，geo loss 对总 loss 的影响越小。
    geo_loss_scale = 10.0

    # FDE 损失。重点约束最后一个预测点，能改善终点偏差。
    use_fde_loss = True

    # FDE 损失权重。
    fde_weight = 0.5

    # 平滑损失。抑制预测轨迹突然折返、锯齿、乱跳。
    use_smooth_loss = True

    # 平滑损失权重。太大可能让转弯路线被过度拉直。
    smooth_weight = 0.2

    # 航向角循环损失。解决 359 度和 1 度其实只差 2 度的问题。
    use_circular_cog = True

    # 航向角损失权重。
    cog_weight = 0.2

    # 航向角损失缩放项。默认 180 表示按半圆角度尺度归一。
    cog_loss_scale = 180.0

    # ======================================================================
    # 8. 配置读取辅助函数
    # ======================================================================
    # 下面这个函数不用改。训练脚本会调用它，把本类里的参数转成字典。
    def to_dict(self):
        return {
            key: getattr(self, key)
            for key in dir(self)
            if not key.startswith("_") and not callable(getattr(self, key))
        }


config = Config()


def get_config():
    return config.to_dict()
