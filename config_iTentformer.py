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

    # 由utils/build_dma_voyage_context.py生成，与当前三个月轨迹数据逐点对齐。
    # 侧车只保存每个历史时刻之前最后已知的船型、吃水、Destination等语义信息。
    voyage_context_path = "dataset/dma_raw_2023_06_07_08/dma_voyage_context_2023_06_07_08.pkl"

    # 千问语义教师：该侧车只包含预测时刻之前航次文本的冻结向量，
    # 不读取航路标签和真实未来，因此可以安全地跨折复用。
    # 先运行 utils/build_qwen_semantic_teacher.py 生成该文件。
    use_qwen_semantic_teacher = True
    qwen_semantic_path = "dataset/dma_raw_2023_06_07_08/dma_qwen_semantic_teacher_v1.pkl"
    semantic_hidden_dim = 128
    # 语义仅作为温和软先验，避免错误Destination压过历史轨迹。
    semantic_fusion_weight = 0.25
    semantic_dropout = 0.20

    # 加入船舶静态/航次信息后必须按MMSI分组，避免同一艘船同时出现在训练和测试中。
    group_folds_by_mmsi = True

    # 固定传统划分：70%训练、10%验证、20%测试，只训练一次。
    # 第一次运行会生成划分清单，之后始终复用，保证不同模型公平比较。
    # 改回 "kfold" 可恢复原来的5折交叉验证。
    split_mode = "fixed"
    test_ratio = 0.20
    split_seed = 42
    split_manifest_path = "dataset/dma_raw_2023_06_07_08/dma_fixed_split_70_10_20_seed42.json"

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
    subroute_labels_path = "dataset/dma_raw_2023_06_07_08/dma_subroutes_ti_4class_compact6_v1_labels.json"

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
    # 这样低先验的小分支也有被千问反推纠正的机会；正确子航路不在候选池时任何重排器都无法恢复。
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

    # 千问候选重排器。主模型训练结束后，冻结本地 Qwen3-1.7B，使用数值软提示比较：
    # 基础预测 + Top-2 大类各自 Top-2 子航路，共 5 条候选，最终仍只输出一条。
    # 千问不直接生成经纬度，也不会把 4GB 权重复制进 iTentformer checkpoint；
    # 这里只训练并单独保存数值适配器和打分头，便于关闭后做消融实验。
    # 旧的训练后候选重排器在验证集未获得可靠增益，默认关闭并保留作消融实验。
    use_qwen_reranker = False
    qwen_model_path = r"D:\Jason1982\wsl\Models\Qwen3-1.7B"

    # None：每折自动保存为 model_dir/model_prefix_Kx_qwen_reranker.pt。
    # eval_only=True 时也会从这个位置自动加载；需要手工指定时再填写完整路径。
    qwen_adapter_path = None

    # 第二阶段只抽取一部分窗口，优先学习原筛选器选错、候选存在分歧的困难样本。
    # 这不会重新训练主模型，显存和时间都比把千问塞进 50 轮主训练稳定得多。
    qwen_reranker_epochs = 3
    qwen_reranker_batch_size = 4
    qwen_train_max_windows = 4096
    qwen_valid_max_windows = 1024
    qwen_hard_sample_ratio = 0.7

    # 数值特征先映射成 2048 维 Qwen soft token；仅下面的小适配器参与训练。
    qwen_adapter_dim = 64
    qwen_reranker_lr = 2e-4
    qwen_reranker_weight_decay = 1e-4
    qwen_reranker_clip = 1.0
    qwen_gradient_checkpointing = True

    # 千问分数与原候选筛选器分数的融合权重。过大可能让小样本重排器反客为主。
    qwen_reranker_weight = 0.5
    qwen_cost_temperature = 0.25
    qwen_cost_regression_weight = 0.2
    qwen_fused_loss_weight = 0.5

    # 反事实重排额外强化“真实优胜候选必须压过错误候选”，并重点照顾同一大类内选错子路的样本。
    qwen_pairwise_weight = 0.4
    qwen_pairwise_margin = 0.3
    qwen_pairwise_min_cost_gap = 0.01
    qwen_same_route_hard_weight = 1.5
    qwen_calibration_max_apply_ratio = 0.35
    qwen_context_max_tokens = 64
    qwen_context_dropout = 0.15

    # 只在大类置信度低、前两类接近，或候选筛选器前两名接近时调用千问。
    # 高置信度简单样本继续走原模型，减少推理耗时，也保护已经预测正确的大类。
    qwen_uncertain_only = True
    qwen_uncertainty_confidence_threshold = 0.85
    qwen_uncertainty_margin_threshold = 0.25

    # 千问只重点学习“原选择器会选错、但候选池里有明显更好轨迹”的窗口。
    # min_oracle_gain 越大，样本越纯但数量越少；winner_gap 过滤掉多个候选几乎一样好的模糊样本。
    # gain_weight 会给高收益纠错样本更高 loss 权重，避免被大量普通样本淹没。
    qwen_focus_high_gain = True
    qwen_min_oracle_gain_nmi = 0.03
    qwen_min_winner_gap_nmi = 0.02
    qwen_gain_weight = 2.0

    # 只有验证集候选优胜者准确率至少提升 0.5 个百分点，正式测试才启用千问。
    # 如果没有达到，适配器仍会保存供诊断，但自动回退到原候选筛选器，避免负优化。
    qwen_require_validation_gain = True
    qwen_min_validation_gain = 0.5
    qwen_min_validation_cost_gain = 0.005

    # 训练前期用真实子航路帮助预测器学会“不同标签对应不同走向”，
    # 随训练逐步切换为模型自己的分类结果，减小训练和推理之间的差异。
    use_branch_teacher_forcing = True
    branch_teacher_forcing_start = 0.7
    branch_teacher_forcing_end = 0.1
    branch_teacher_forcing_decay_epochs = 30

    # 主航路同样使用每折训练集原型，优先修正 OB1 被判断成 OA 一类的大类错误。
    use_route_prototype_prior = True
    route_prototype_points = 32
    route_prototype_weight = 0.6

    # 每一折只使用该折训练轨迹构建子航路平均原型，不读取验证集或测试集。
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
    # 原型只由当前折训练集建立，计算可判别性时不读取该窗口的未来点，验证/测试不会泄漏。
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
    # 保存模型文件时用的前缀。最终通常类似：
    # save_models/dma_2023_06_07_08_ti_4class_K1.pt
    model_prefix = "dma_2023_06_07_08_ti_4class_candidate_v14_tailintent_fixedsplit_qwen_semantic_compact6_hist3h_pred3h"

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
    # 4. 数据划分、训练轮数和数据窗口
    # ======================================================================
    # split_mode="fixed" 时下面两项不会控制数据划分，程序只训练一次。
    # split_mode="kfold" 时，folds是总折数，run_folds是实际运行折数。
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

    # 验证集占非测试数据的比例。固定测试集占20%后，剩余80%中的12.5%
    # 作为验证集，即总数据约10%。按MMSI分组后实际条数可能有轻微偏差。
    valid_ratio = 0.125

    # 固定验证集条数。只有当 valid_ratio = None 时才会使用这个参数。
    # 小样本复现实验想完全固定验证集数量时，可以设 valid_ratio = None，然后改这里。
    valid_count = None

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
