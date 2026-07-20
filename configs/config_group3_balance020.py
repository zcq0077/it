"""第三组保留预设：降低小类均衡意图损失权重。"""

from config_iTentformer import Config as DefaultConfig


class Config(DefaultConfig):
    # 第三组来自筛选阶段排名第三的 screen_06_balance020。
    lr_scheduler_patience = 5
    subroute_residual_scale = 0.25
    balanced_intent_loss_weight = 0.20

    # 使用独立名称，避免覆盖第一组模型和结果。
    model_prefix = "dma_group3_balance020_3h"
    run_name = None
