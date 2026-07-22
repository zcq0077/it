"""Controlled v15e experiment with the robust two-branch OC labels.

Only the subroute label sidecar and output prefix differ from the current
dma_v15e_3h configuration.  The dataset, fixed MMSI split, Qwen3-1.7B semantic
features, model architecture, losses, and optimization settings are unchanged.
"""

from config_iTentformer import Config as V15eConfig


class Config(V15eConfig):
    subroute_labels_path = (
        "dataset/dma_raw_2023_06_07_08_plus_09_10_oa_s00/"
        "dma_2023_06_07_08_plus_09_10_oa_s00_target350_"
        "subroute_labels_oc2_v1.json"
    )
    model_prefix = "dma_v15e_oc2_3h"
    run_name = None
