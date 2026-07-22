"""Controlled v15e ablation using Qwen3-Embedding-0.6B features.

All trajectory, intent, routing, loss, split, seed, and fusion settings are
inherited unchanged from the validated dma_v15e_3h configuration. Only the
offline semantic sidecar and output prefix differ.
"""

from config_iTentformer import Config as V15eConfig


class Config(V15eConfig):
    qwen_semantic_path = (
        "dataset/dma_raw_2023_06_07_08_plus_09_10_oa_s00/"
        "dma_2023_06_07_08_plus_09_10_oa_s00_target350_"
        "qwen3_embedding_0p6b.pkl"
    )
    model_prefix = "dma_v15e_emb06_3h"
    run_name = None
