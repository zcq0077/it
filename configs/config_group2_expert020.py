"""第二组保留预设：降低子航路残差专家的修正强度。"""

from config_iTentformer import Config as DefaultConfig


class Config(DefaultConfig):
    # 第二组与默认第一组的主要差异。
    lr_scheduler_patience = 5
    subroute_residual_scale = 0.20
    balanced_intent_loss_weight = 0.35

    # 使用独立名称，避免覆盖第一组模型和结果。
    model_prefix = "dma_group2_expert020_3h"
    run_name = None
