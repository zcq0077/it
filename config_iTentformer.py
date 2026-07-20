"""iTentformer experiment config.

这个文件就是 iTentformer 的默认参数总表。现在直接运行：

    python iTentformer.py

会自动读取本文件。命令行仍然可以临时覆盖这里的参数，例如：

    python iTentformer.py --epochs 1 --run_name smoke

当前默认数据集是 2023 年 6 月至 10 月 DMA 四类航路数据；
9 月和 10 月仅补入经过固定原型、距离阈值和 MMSI 隔离筛选的 OA_S00 轨迹：

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
    data_path = "dataset/dma_raw_2023_06_07_08_plus_09_10_oa_s00/dma_2023_06_07_08_plus_09_10_oa_s00_target350.pkl"

    # 由utils/build_dma_voyage_context.py生成，与当前增量轨迹数据逐点对齐。
    # 侧车只保存每个历史时刻之前最后已知的船型、吃水、Destination等语义信息。
    voyage_context_path = "dataset/dma_raw_2023_06_07_08_plus_09_10_oa_s00/dma_2023_06_07_08_plus_09_10_oa_s00_target350_voyage_context.pkl"

    # Qwen3-Embedding语义证据：侧车只包含预测时刻之前航次文本的冻结向量，
    # 不读取航路标签和真实未来，因此可以安全地跨不同固定划分复用。
    # Qwen不直接预测坐标；它只给层级航路意图提供可学习、可门控的语义先验。
    use_qwen_semantic_teacher = True
    qwen_semantic_path = "dataset/dma_raw_2023_06_07_08_plus_09_10_oa_s00/dma_2023_06_07_08_plus_09_10_oa_s00_target350_qwen3_embedding_0p6b.pkl"
    semantic_hidden_dim = 128
    # Qwen只与后续解码器共用的Route Embedding做余弦对齐。
    # 子航路不直接接收文本logits，而是通过主航路后验和层级约束间接受益。
    use_semantic_route_alignment = True
    use_semantic_subroute_alignment = False
    semantic_alignment_temperature = 0.20
    semantic_route_alignment_weight = 0.05
    semantic_subroute_alignment_weight = 0.0
    # 可靠性目标比较语义证据与非语义证据的逐样本分类损失。
    semantic_route_reliability_weight = 0.05
    semantic_reliability_temperature = 0.50
    # 类别均衡辅助流只训练运动意图，避免重采样改变Qwen语义先验的自然分布。
    use_semantic_in_balanced_intent_stream = False
    # 对齐后的语义logits只作为温和残差证据，错误Destination时可由门控关闭。
    semantic_fusion_weight = 0.10
    semantic_dropout = 0.20

    # 固定传统划分：按MMSI分组，约70%训练、10%验证、20%测试，只训练一次。
    # 同一艘船不会同时进入训练、验证和测试集，避免语义信息造成身份泄漏。
    # 第一次运行会生成划分清单，之后始终复用，保证不同模型公平比较。
    test_ratio = 0.20
    split_seed = 42
    # 固定训练随机种子，保证不同参数组合尽量只比较参数本身的影响。
    # 数据划分仍由 split_seed 和固定 manifest 单独控制。
    train_seed = 42
    split_manifest_path = "dataset/dma_raw_2023_06_07_08_plus_09_10_oa_s00/dma_2023_06_07_08_plus_09_10_oa_s00_target350_fixed_split.json"

    # 每条轨迹对应的航路类别标签。当前包含 OA / OB1 / OB2 / OC 四类。
    # 这个文件不是模型输入特征，主要用于：
    # 1. 固定划分时尽量让训练、验证、测试集都有各类航路；
    # 2. 测试集可视化时按类别均衡抽样；
    # 3. 日志里统计每类轨迹数量，方便检查数据是否偏。
    route_labels_path = "dataset/dma_raw_2023_06_07_08_plus_09_10_oa_s00/dma_2023_06_07_08_plus_09_10_oa_s00_target350_route_labels.json"

    # True：固定划分时兼顾大类航路比例，推荐打开。
    # False：只按MMSI随机分组，某些集合中的小类比例可能偏低。
    stratify_by_route = True

    # 大类航路分类辅助头。模型会先学习当前历史轨迹更像 OA/OB1/OB2/OC 哪一类。
    # 注意这里不是只看起点，而是看已观测历史段里的起点、当前位置、航向、速度和变化趋势。
    use_route_intent_head = True
    route_intent_weight = 0.2

    # 大类航路 embedding 融合。打开后，大类概率会反馈给轨迹预测分支。
    use_route_embedding = True
    route_embedding_dim = 16

    # 大类也使用分阶段监督。OA/OB1/OB2/OC 在 O 区域共享航段时不强迫提前硬选，
    # 当历史轨迹接近主分叉并与某一大类原型明显匹配后，再恢复硬分类和标签教师强制。
    use_route_decidability = True
    route_decidable_min_weight = 0.05
    route_decidable_confidence_threshold = 0.60
    route_decidable_margin_threshold = 0.10
    route_decidable_direction_points = 4
    route_decidable_threshold = 0.50
    route_undecidable_soft_weight = 0.10

    # 细分子航路标签，由 utils/discover_subroutes.py 生成。
    # 它会把 OA/OB1/OB2/OC 继续细化为 OA_S00、OB1_S01、OC_S02 等小分支。
    subroute_labels_path = "dataset/dma_raw_2023_06_07_08_plus_09_10_oa_s00/dma_2023_06_07_08_plus_09_10_oa_s00_target350_subroute_labels.json"

    # True：固定划分时优先兼顾子航路比例，比只看 OA/OB1/OB2/OC 更细。
    # 如果子航路数量太多且某些类样本很少，可以改回 False。
    stratify_by_subroute = True

    # 子航路分类辅助头。打开后，模型训练时会同时学习“当前历史轨迹属于哪个小分支”。
    use_subroute_intent_head = True
    subroute_intent_weight = 0.35

    # 子航路 embedding 融合。模型会把自己预测的子航路概率转成 embedding，
    # 再融合回轨迹预测分支，让“像哪条小航路”真正影响未来轨迹预测。
    use_subroute_embedding = True
    subroute_embedding_dim = 16

    # 通用子航路残差专家：共享解码器先学习所有船都遵循的运动规律，
    # 再由每个子航路的小专家修正该分支特有的转弯和横向偏移。
    # 专家由标签中的全部子航路自动建立，不区分六月/九月，也不识别“新增数据”。
    # 最后一层从0开始，scale保持温和，避免小类专家一开始把主航路预测拉乱。
    use_subroute_residual_experts = True
    subroute_residual_hidden_dim = 32
    subroute_residual_scale = 0.25
    subroute_residual_dropout = 0.10

    # 子航路判断使用“历史均值 + 最后位置 + 首尾变化”三部分特征。
    # 相比只做全历史平均，它更容易捕捉接近分岔口和刚开始转向的信号。
    intent_summary_mode = "mean_last_delta"

    # 旧版共用温度，保留作兼容回退；下面两个独立温度才是当前实验实际使用值。
    branch_routing_temperature = 0.7

    # 大类旧日志存在明显过度自信，因此把大类温度调高到 1.30，让概率更平缓。
    # 小类保持略低温度 0.90，在已经可判别时仍能形成明确分支，但不再像 0.70 那样过尖。
    route_routing_temperature = 1.30
    subroute_routing_temperature = 0.90
    hard_subroute_routing = True

    # 高置信度时明确选择第一名；置信度不足或前两名接近时，保留 Top-2 候选概率。
    # 这样 OA/OB1 等共享航段上的暂时犹豫不会立刻把整条未来轨迹锁到错误航路。
    confidence_aware_routing = True
    routing_confidence_threshold = 0.8
    routing_margin_threshold = 0.35
    routing_top_k = 2

    # 学习“当前历史是否已经足以判断航路”。该头只读取历史轨迹，不读取真实未来。
    # 只有类别置信度、前两名间隔和这个可判别概率同时过关，才允许硬选；否则保留 Top-2。
    use_learned_decidability = True
    decidability_hidden_dim = 64
    route_decidability_gate_threshold = 0.65
    subroute_decidability_gate_threshold = 0.60
    route_decidability_loss_weight = 0.10
    subroute_decidability_loss_weight = 0.10

    # 显式生成两条独立候选轨迹，并由学习型筛选器最终只选择一条输出。
    # 训练时用候选轨迹真实 ADE/FDE 的优胜者监督筛选器；推理时筛选器不读取真实未来。
    use_candidate_selector = True
    candidate_count = 2
    candidate_subroutes_per_route = 2

    # 紧凑版只有 6 个子航路，直接把全部子航路放进候选池并自动绑定所属大类。
    # 这样低先验的小分支也能被学习式候选选择器看到；正确子航路不在候选池时无法恢复。
    candidate_pool_strategy = "all_subroutes"
    candidate_max_subroutes = 8
    candidate_selector_hidden_dim = 64
    candidate_selector_weight = 0.5
    candidate_trajectory_weight = 0.08
    candidate_fde_weight = 0.2
    candidate_probability_prior_weight = 0.15
    candidate_base_prior_bias = 0.0
    candidate_cost_temperature = 0.35
    candidate_cost_regression_weight = 0.1
    candidate_selector_warmup_epochs = 10
    # 初始切换规则；训练完成后会用验证集自动校准，最终不一定沿用这组阈值。
    # 校准目标是降低候选轨迹代价，不是盲目提高 branch_switch。
    candidate_switch_confidence_threshold = 0.45
    candidate_switch_logit_margin = 0.15

    # 验证集校准候选切换阈值。max_switch_ratio 防止选择器为了追 oracle 过度乱切。
    use_candidate_selection_calibration = True
    candidate_calibration_max_switch_ratio = 0.50
    candidate_calibration_min_cost_gain = 0.0
    candidate_include_target_during_training = True

    # 训练前期用真实子航路帮助预测器学会“不同标签对应不同走向”，
    # 随训练逐步切换为模型自己的分类结果，减小训练和推理之间的差异。
    use_branch_teacher_forcing = True
    branch_teacher_forcing_start = 0.7
    branch_teacher_forcing_end = 0.1
    branch_teacher_forcing_decay_epochs = 30

    # 主航路同样只使用固定训练集原型，优先修正 OB1 被判断成 OA 一类的大类错误。
    use_route_prototype_prior = True
    route_prototype_points = 32
    route_prototype_weight = 0.6

    # 只使用固定训练集轨迹构建子航路平均原型，不读取验证集或测试集。
    # 当前轨迹越接近某条原型线、行进方向越一致，该子航路的分类分数越高。
    # 它主要在已经接近或进入分岔区域后提供帮助，共享主航道上不会凭空知道未来选择。
    use_subroute_prototype_prior = True
    subroute_prototype_points = 32
    subroute_prototype_weight = 0.8
    subroute_prototype_distance_scale = 0.25
    subroute_prototype_direction_weight = 0.5

    # 层级意图约束。大类概率会约束小类概率：
    # 例如大类更像 OA 时，OA_S00/OA_S01/OA_S02 会更容易被选中，OC_Sxx 会被压低。
    # strength 越大约束越强；太大时如果大类判断错，会拖累小类，所以默认用温和强度。
    use_hierarchical_intent = True
    hierarchical_mask_strength = 1.5

    # 大类判断不可靠时自动减弱“大类压小类”的层级约束，避免大类选错后封死正确小类。
    # min_scale 是最弱时仍保留的约束比例，不能为 0，否则小类会完全失去大类结构信息。
    confidence_gated_hierarchy = True
    hierarchy_min_scale = 0.15

    # 子航路对比损失。同一子航路的隐藏特征会被拉近，不同子航路会被拉远。
    use_subroute_contrastive_loss = True
    subroute_contrastive_weight = 0.05
    subroute_contrastive_temperature = 0.2

    # Subroute focal loss: hard/rare branch windows get a little more gradient.
    # Keep gamma modest; too large can make the model chase noisy branch labels.
    use_subroute_focal_loss = True
    subroute_focal_gamma = 1.5
    subroute_label_smoothing = 0.02

    # 分阶段子航路监督：只使用历史窗口判断“此刻是否已经能看出分支意图”。
    # 分叉前，各子航路历史几乎重合，最终标签只保留 5% 的硬分类权重，并学习同一大类内的软分布；
    # 接近或进入分叉后，历史与真实子航路原型逐渐匹配，硬标签、对比损失和教师强制才逐步恢复。
    # 原型只由固定训练集建立，计算可判别性时不读取该窗口的未来点，验证/测试不会泄漏。
    use_subroute_decidability = True
    subroute_decidable_min_weight = 0.05
    subroute_decidable_confidence_threshold = 0.60
    subroute_decidable_margin_threshold = 0.10
    subroute_decidable_direction_points = 4
    subroute_decidable_threshold = 0.50
    subroute_decidable_contrastive_threshold = 0.50
    subroute_undecidable_soft_weight = 0.15

    # 子航路分类 class weight。只加在子航路分类辅助头上，不直接改变轨迹回归 loss。
    # alpha 越大越照顾小类；0 表示不按频次加权，1 表示强逆频次加权。
    # max_ratio 限制小类最多被照顾到多少倍，避免小类噪声把模型带偏。
    use_subroute_class_weight = True
    subroute_class_weight_alpha = 0.5
    subroute_class_weight_max_ratio = 5.0

    # 温和子航路均衡采样。每轮训练仍保持同样窗口数，只把其中一部分替换成按小类加权抽样。
    # 下面保留旧版混合比例，只有关闭“解耦双流训练”时才会使用。
    use_balanced_subroute_sampling = True
    balanced_sampling_alpha = 0.4
    balanced_sampling_max_ratio = 5.0
    balanced_sampling_mix_ratio = 0.2

    # 解耦双流训练：主轨迹流始终使用完整自然分布，不再被重复的小类窗口替换；
    # 额外的小类均衡流只训练主/子航路分类、可判别性、对比表示和未来意图原型。
    # 这样能照顾 OA_S02/OB2 等小类，同时尽量保护总体 ADE/FDE 和大类轨迹质量。
    use_decoupled_balanced_intent_training = True

    # 辅助流额外处理的窗口数占自然训练流的比例。0.20 约增加 20% 的编码器计算，
    # 不增加主轨迹解码和候选生成计算；比例过大容易反复学习少量小类噪声。
    balanced_intent_ratio = 0.20
    balanced_intent_loss_weight = 0.35

    # 前3轮逐渐把辅助流从 1/3 强度升到完整强度，先让主模型学会基本运动规律。
    balanced_intent_warmup_epochs = 3

    # 辅助意图流先按轨迹归一化：一条轨迹无论产生5个还是50个重叠窗口，
    # 每轮被抽中的总概率都大致相同，再在此基础上照顾小类。
    use_track_balanced_intent_sampling = True

    # Future-enhanced 子航路意图原型。训练时用真实未来相对位移提炼“这条支路以后怎么走”，
    # 并把可判别历史拉向对应原型；验证和推理完全不读取未来，避免信息泄漏。
    # 对不可判别窗口，历史-未来对齐权重会按 decidability 自动接近0，不强迫凭空硬选。
    use_future_enhanced_intent = True
    future_intent_dim = 64
    future_intent_temperature = 0.20

    # 原型匹配只作为子航路 logits 的温和残差，避免它压过原始历史分类器。
    future_intent_logit_weight = 0.15
    future_intent_loss_weight = 0.08
    future_intent_alignment_weight = 0.50

    # ======================================================================
    # 2. 实验输出位置
    # ======================================================================
    # 简短实验代号，同时用于模型文件和自动生成的结果目录。
    # 当前含义：DMA、v17 Qwen主航路语义先验、历史/预测均为3小时。
    # 使用新前缀，保留已停止的dma_v16q_3h实验作为双层语义对齐对照。
    model_prefix = "dma_v17qr_3h"

    # 模型 checkpoint 保存目录。
    model_dir = "save_models"

    # 日志、预测轨迹图、实验结果保存目录。
    results_dir = "results"

    # 单次运行的实验名。
    # None：自动生成，格式为 模型前缀-时间戳。
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
    # None：默认加载 model_dir/model_prefix_fixed.pt。
    # 例如："save_models/dma_2023_06_07_08_ti_4class_fixed.pt"
    checkpoint_path = None

    # ======================================================================
    # 4. 数据划分、训练轮数和数据窗口
    # ======================================================================
    # 最大训练轮数。早停打开后，不一定会跑满。
    epochs = 50

    # 早停耐心值：验证集连续多少轮不提升就停。
    # 当前默认采用自动调参第一组（已完成最终测试）的正式训练设置。
    patience = 12

    # 早停和最佳模型保存的监控指标。
    # loss：监控验证总 loss，适合单任务训练；
    # ade：监控验证 ADE；
    # ade_fde：监控 ADE + early_stop_fde_weight * FDE，更适合当前多任务轨迹预测。
    early_stop_metric = "ade_fde"
    early_stop_fde_weight = 0.2

    # 优化器与学习率调度。plateau 会在验证指标连续若干轮不改善后再降低学习率，
    # 比旧版只比较最近3轮的触发规则更稳定。
    learning_rate = 2e-4
    lr_scheduler = "plateau"
    lr_reduce_factor = 0.5
    # 第一组最优候选：连续3轮无改善后降低学习率。
    lr_scheduler_patience = 3
    lr_min = 3.125e-6

    # 验证集占非测试数据的比例。固定测试集占20%后，剩余80%中的12.5%
    # 作为验证集，即总数据约10%。按MMSI分组后实际条数可能有轻微偏差。
    valid_ratio = 0.125

    # 滑动窗口步长。
    # 1：窗口最密，训练样本最多，但训练更慢；
    # 20：窗口更稀，训练快，但样本少，容易学不充分。
    window_stride = 1

    # DMA每15分钟一个点；13个历史点首尾跨度为12个间隔，即3小时历史轨迹。
    input_length = 13

    # 12个未来点对应3小时预测。修改任一长度后都需要重新训练模型。
    target_length = 12

    # ======================================================================
    # 5. 测试集可视化
    # ======================================================================
    # 是否在每个训练轮次后评估测试集。
    # 正式实验应保持 False：训练期间只看验证集，最佳模型确定后再测试一次。
    # True 仅用于临时诊断，不会参与反向传播、早停或最佳模型保存。
    evaluate_test_each_epoch = False

    # 普通正式训练保持 True，最佳验证模型训练完成后只测试一次。
    # 自动调参脚本会覆盖为 False，确保搜索阶段完全不读取测试集。
    evaluate_final_test = True

    # 最终测试结束后画多少张“历史轨迹 / 真实未来 / 预测未来”对比图。
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
