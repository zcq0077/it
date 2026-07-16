import torch
import torch.nn as nn
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser, BooleanOptionalAction
from collections import Counter
from dataclasses import asdict, is_dataclass
import importlib.util
from pathlib import Path
from types import ModuleType
import hashlib
import json
import logging
import sys
import time
import gc
from utils.EarlyStopping import EarlyStopping
from utils.Haversine_Loss import HaversineLoss
from utils.qwen_candidate_reranker import QwenCandidateReranker

sys.path.append("../../")
from model import *
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss
from utils.bohai_diff import window_slice
import numpy as np

try:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.umath", np.core.umath)
except AttributeError:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config_iTentformer.py"

"""
Limited by my code level and time factors, it is recommended to re-write the standardized training code according to 
your own needs, here can be for reference
"""
delta_cols = [8, 9, 10, 11]
intent_cols = [2]
src_cols = [2, 3, 4, 5]
tgt_cols = [2, 3, 4, 5] + [-10, -9]
in_cols = src_cols + delta_cols
local_intent_size = len(intent_cols)
intent_size = 8
input_size = 10
input_size_tcn = 8
output_size = 4
d_model = 128
num_channels = [32] * 2

kernel_size = 3
dropout = 0.2
clip = 0.1
batch_size = 16
input_length = 10
target_length = 10
concat_dim = input_size_tcn + num_channels[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion1 = nn.MSELoss().to(device)
criterion2 = nn.CrossEntropyLoss().to(device)
metric_haversine = HaversineLoss(min_hav=0.0).to(device)
stable_haversine = HaversineLoss(min_hav=1e-7).to(device)
subroute_class_weights = None
train_sampling_probabilities = None
candidate_selector_runtime_active = False
candidate_selection_calibration = None
qwen_reranker = None
qwen_reranker_runtime_active = False
voyage_context_payload = None
qwen_semantic_payload = None
qwen_semantic_text_to_id = None


def setup_logging(log_path, append=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)03d - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    file_mode = "a" if append else "w"
    file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def make_run_name(args):
    data_stem = Path(args.data_path).stem
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    raw_name = args.run_name or f"{args.model_prefix}-{data_stem}-{timestamp}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in raw_name)


def normalize_config_value(value):
    if isinstance(value, Path):
        return str(value)
    return value


def config_object_to_dict(config, config_path):
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    elif isinstance(config, type):
        config = {
            key: getattr(config, key)
            for key in dir(config)
            if not key.startswith("_")
        }
    elif hasattr(config, "to_dict") and callable(getattr(config, "to_dict")):
        config = config.to_dict()
    elif not isinstance(config, dict) and hasattr(config, "__dict__"):
        config = vars(config)

    if not isinstance(config, dict):
        raise ValueError(f"Config must provide a dict-like object: {config_path}")

    result = {}
    for key, value in config.items():
        if key.startswith("_") or callable(value) or isinstance(value, ModuleType):
            continue
        result[key] = normalize_config_value(value)
    return result


def load_python_config(config_path):
    path = Path(config_path)
    module_name = f"itentformer_config_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load Python config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "get_config") and callable(module.get_config):
        config = module.get_config()
    elif hasattr(module, "config"):
        config = module.config
    elif hasattr(module, "CONFIG"):
        config = module.CONFIG
    elif hasattr(module, "Config"):
        config = module.Config
    else:
        config = {
            key: value
            for key, value in vars(module).items()
            if not key.startswith("_")
        }
    return config_object_to_dict(config, config_path)


def load_config_defaults(config_path):
    if not config_path or str(config_path).lower() in {"none", "null", "false"}:
        return {}
    path = Path(config_path)
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        script_path = SCRIPT_DIR / path
        path = cwd_path if cwd_path.exists() else script_path
    if path.suffix.lower() == ".py":
        return load_python_config(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config_object_to_dict(config, config_path)


def get_default_config_path():
    return str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None


def apply_config_defaults(parser, config_path):
    defaults = load_config_defaults(config_path)
    if not defaults:
        return
    valid_keys = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown config option(s) in {config_path}: {unknown}")
    parser.set_defaults(**defaults)


def load_voyage_context_sidecar(path, tracks):
    if not path:
        return None
    payload = pd.read_pickle(path)
    if not isinstance(payload, dict) or int(payload.get("format_version", 0)) != 1:
        raise ValueError(f"Unsupported voyage context sidecar format: {path}")
    context_ids = payload.get("context_ids")
    text_pool = payload.get("text_pool")
    if context_ids is None or text_pool is None or len(context_ids) != len(tracks):
        raise ValueError("Voyage context sidecar does not match the trajectory dataset.")
    for index, (track, ids) in enumerate(zip(tracks, context_ids)):
        if len(track) != len(ids):
            raise ValueError(f"Voyage context point count mismatch at track {index}.")
    return payload


def voyage_text_pool_hash(text_pool):
    digest = hashlib.sha256()
    for text in text_pool:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_qwen_semantic_sidecar(path, context_payload):
    global qwen_semantic_text_to_id
    if not path:
        return None
    if context_payload is None:
        raise ValueError("Qwen semantic teacher requires --voyage_context_path.")
    payload = pd.read_pickle(path)
    if not isinstance(payload, dict) or int(payload.get("format_version", 0)) != 1:
        raise ValueError("Unsupported Qwen semantic sidecar format.")
    if not bool(payload.get("label_free", False)):
        raise ValueError("Qwen semantic sidecar must be label-free to avoid fold leakage.")
    embeddings = np.asarray(payload.get("embeddings"))
    text_pool = list(context_payload["text_pool"])
    if embeddings.ndim != 2 or embeddings.shape[0] != len(text_pool):
        raise ValueError("Qwen semantic embeddings do not match voyage-context text_pool.")
    expected_hash = voyage_text_pool_hash(text_pool)
    if payload.get("text_pool_hash") != expected_hash:
        raise ValueError("Qwen semantic sidecar was built from a different voyage context file.")
    if int(payload.get("embedding_dim", 0)) != embeddings.shape[1]:
        raise ValueError("Qwen semantic embedding dimension metadata is invalid.")
    payload["embeddings"] = embeddings
    qwen_semantic_text_to_id = {str(text): index for index, text in enumerate(text_pool)}
    return payload


def semantic_features_for_contexts(contexts):
    if qwen_semantic_payload is None:
        return None
    if contexts is None:
        raise ValueError("Semantic teacher is enabled but window voyage contexts are missing.")
    ids = np.fromiter(
        (qwen_semantic_text_to_id.get(str(text), 0) for text in contexts),
        dtype=np.int64,
        count=len(contexts),
    )
    rows = np.asarray(qwen_semantic_payload["embeddings"][ids], dtype=np.float32)
    return torch.from_numpy(rows).to(device)


def expand_track_contexts_to_windows(track_context_ids, window_slices, text_pool, stride):
    if track_context_ids is None:
        return None
    contexts = []
    for ids, windows in zip(track_context_ids, window_slices):
        ids = np.asarray(ids, dtype=np.int64)
        for window_index in range(len(windows)):
            history_end = window_index * stride + input_length - 1
            context_id = int(ids[history_end]) if history_end < len(ids) else 0
            if context_id < 0 or context_id >= len(text_pool):
                context_id = 0
            contexts.append(str(text_pool[context_id]))
    return np.asarray(contexts, dtype=object)


def resolve_valid_count(args, train_size):
    if train_size < 2:
        raise ValueError("At least 2 training tracks are required to split a validation set.")

    if args.valid_ratio is not None:
        if not 0 < args.valid_ratio < 1:
            raise ValueError("--valid_ratio must be between 0 and 1.")
        requested = int(round(train_size * args.valid_ratio))
        valid_count = min(max(requested, 1), train_size - 1)
        return valid_count, f"ratio={args.valid_ratio:.3f}"

    if args.valid_count is None:
        raise ValueError("Either --valid_ratio or --valid_count must be set.")
    if args.valid_count <= 0:
        raise ValueError("--valid_count must be positive.")

    valid_count = min(args.valid_count, train_size - 1)
    return valid_count, f"count={args.valid_count}"


def grouped_validation_split(labels, groups, valid_count, seed):
    indices = np.arange(len(groups))
    ratio = min(max(valid_count / max(len(groups), 1), 0.02), 0.5)
    if labels is not None:
        split_count = max(2, int(round(1.0 / ratio)))
        split_count = min(split_count, len(np.unique(groups)))
        try:
            splitter = StratifiedGroupKFold(
                n_splits=split_count,
                shuffle=True,
                random_state=seed,
            )
            train_indices, valid_indices = next(splitter.split(indices, labels, groups))
            return np.asarray(train_indices), np.asarray(valid_indices)
        except ValueError:
            pass
    splitter = GroupShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    train_indices, valid_indices = next(splitter.split(indices, groups=groups))
    return np.asarray(train_indices), np.asarray(valid_indices)


def optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "false"}:
        return None
    return float(value)


def optional_path(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "false", ""}:
        return None
    return str(value)


def count_parameters(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def log_metric_line(logger, stage, fold, total_folds, epoch, loss, ade, fde, rmse_cog, rmse_sog):
    ade = to_float(ade)
    fde = to_float(fde)
    logger.info(
        "%s, fold %d/%d, epoch %03d, loss %.5f, ADE %.5fnmi (%.2fm), "
        "FDE %.5fnmi (%.2fm), RMSE_COG %.5fdeg, RMSE_SOG %.5fkn.",
        stage,
        fold,
        total_folds,
        epoch,
        to_float(loss),
        ade,
        ade * 1852.0,
        fde,
        fde * 1852.0,
        to_float(rmse_cog),
        to_float(rmse_sog),
    )


def inverse_standardized(values, transform_matrix, mean_values):
    return values @ transform_matrix + mean_values[:4]


def standardized_to_real(values):
    return values @ transform_tensor + mean_tensor


def circular_angle_diff(pred_deg, true_deg):
    return torch.remainder(pred_deg - true_deg + 180.0, 360.0) - 180.0


def build_linear_baseline(src):
    history = src[:, :, :output_size]
    last = history[:, -1:, :]
    if history.size(1) < 2:
        return last.repeat(1, target_length, 1)

    last_delta = history[:, -1:, :] - history[:, -2:-1, :]
    steps = torch.arange(1, target_length + 1, device=src.device, dtype=src.dtype).view(1, -1, 1)
    return last + last_delta * steps


def compose_value_output(raw_output, src):
    if args.target_mode == "residual_linear":
        return build_linear_baseline(src) + raw_output
    return raw_output


def compose_candidate_outputs(raw_outputs, src):
    if args.target_mode == "residual_linear":
        return build_linear_baseline(src)[:, None, :, :] + raw_outputs
    return raw_outputs


def build_hierarchical_candidates(
        route_logits,
        subroute_logits,
        candidate_count,
        subroutes_per_route=1,
        pool_strategy="topk_routes",
        max_subroute_candidates=8,
        route_targets=None,
        subroute_targets=None,
        include_targets=False,
        target_include_mask=None,
):
    if route_logits is None or subroute_logits is None or route_to_subroute_mask is None:
        raise RuntimeError("Hierarchical candidates require route/subroute logits and their class mask.")

    if pool_strategy == "all_subroutes":
        parent_count = route_to_subroute_mask.sum(dim=0)
        if torch.any(parent_count != 1):
            raise RuntimeError("Every subroute must belong to exactly one main route.")
        subroute_parent_ids = torch.argmax(route_to_subroute_mask, dim=0)
        parent_route_logits = route_logits.detach()[:, subroute_parent_ids]
        joint_logits = subroute_logits.detach() + parent_route_logits
        branch_count = min(
            max(int(max_subroute_candidates), 1),
            subroute_logits.size(-1),
        )
        candidate_subroute_ids = torch.topk(
            joint_logits,
            k=branch_count,
            dim=-1,
        ).indices
        candidate_route_ids = subroute_parent_ids[candidate_subroute_ids]
    elif pool_strategy == "topk_routes":
        route_count = min(max(int(candidate_count), 1), route_logits.size(-1))
        subroutes_per_route = max(int(subroutes_per_route), 1)
        top_route_ids = torch.topk(route_logits.detach(), k=route_count, dim=-1).indices
        route_masks = route_to_subroute_mask[top_route_ids].bool()
        expanded_subroute_logits = subroute_logits.detach()[:, None, :].expand(-1, route_count, -1)
        masked_subroute_logits = expanded_subroute_logits.masked_fill(~route_masks, -1e9)
        top_subroute_ids = torch.topk(
            masked_subroute_logits,
            k=subroutes_per_route,
            dim=-1,
        ).indices
        valid_subroute_counts = route_masks.sum(dim=-1, keepdim=True)
        if torch.any(valid_subroute_counts < 1):
            raise RuntimeError("Every candidate route must have at least one valid subroute.")
        candidate_rank = torch.arange(
            subroutes_per_route,
            device=top_subroute_ids.device,
        ).view(1, 1, -1)
        # Compact label sets can have one child under a main route. Repeat that
        # child rather than allowing topk to return an unrelated masked class.
        top_subroute_ids = torch.where(
            candidate_rank < valid_subroute_counts,
            top_subroute_ids,
            top_subroute_ids[:, :, :1],
        )
        candidate_route_ids = top_route_ids[:, :, None].expand(-1, -1, subroutes_per_route).reshape(
            route_logits.size(0),
            route_count * subroutes_per_route,
        )
        candidate_subroute_ids = top_subroute_ids.reshape(
            route_logits.size(0),
            route_count * subroutes_per_route,
        )
    else:
        raise ValueError(f"Unsupported candidate pool strategy: {pool_strategy}")

    if include_targets and route_targets is not None and subroute_targets is not None:
        for batch_idx in range(candidate_route_ids.size(0)):
            if target_include_mask is not None and not bool(target_include_mask[batch_idx].item()):
                continue
            exact_matching = torch.nonzero(
                candidate_route_ids[batch_idx].eq(route_targets[batch_idx])
                & candidate_subroute_ids[batch_idx].eq(subroute_targets[batch_idx]),
                as_tuple=False,
            ).view(-1)
            if exact_matching.numel():
                continue
            route_matching = torch.nonzero(
                candidate_route_ids[batch_idx].eq(route_targets[batch_idx]),
                as_tuple=False,
            ).view(-1)
            slot = int(route_matching[-1].item()) if route_matching.numel() else candidate_route_ids.size(1) - 1
            candidate_route_ids[batch_idx, slot] = route_targets[batch_idx]
            candidate_subroute_ids[batch_idx, slot] = subroute_targets[batch_idx]

    return candidate_route_ids, candidate_subroute_ids


def candidate_trajectory_costs(candidate_value_outputs, value_target):
    batch_size, candidate_count, sequence_length, _ = candidate_value_outputs.shape
    real_candidates = standardized_to_real(candidate_value_outputs)
    real_target = standardized_to_real(value_target)
    expanded_target = real_target[:, None, :, :].expand(-1, candidate_count, -1, -1)
    distance = metric_haversine(
        real_candidates[:, :, :, 1:3].detach().reshape(batch_size * candidate_count, sequence_length, 2).float(),
        expanded_target[:, :, :, 1:3].reshape(batch_size * candidate_count, sequence_length, 2).float(),
    ).reshape(batch_size, candidate_count, sequence_length)
    ade = distance.mean(dim=-1)
    fde = distance[:, :, -1]
    return ade + args.candidate_fde_weight * fde, ade, fde


def cosine_similarity_2d(left, right):
    numerator = torch.sum(left * right, dim=-1)
    denominator = (
        torch.linalg.vector_norm(left, dim=-1)
        * torch.linalg.vector_norm(right, dim=-1)
    ).clamp_min(1e-6)
    return numerator / denominator


def build_qwen_maritime_features(
        src,
        candidate_route_ids,
        candidate_subroute_ids,
        candidate_value_outputs,
):
    """Describe candidate motion relative to learned local shipping lanes."""
    batch_size, candidate_count, sequence_length, _ = candidate_value_outputs.shape
    prototypes = getattr(model, "subroute_prototypes", None)
    if prototypes is None:
        return torch.zeros(
            batch_size,
            candidate_count,
            11,
            device=candidate_value_outputs.device,
            dtype=candidate_value_outputs.dtype,
        )

    prototypes = prototypes.to(dtype=candidate_value_outputs.dtype)
    own_prototypes = prototypes[candidate_subroute_ids]
    prototype_points = own_prototypes.size(2)
    current_position = src[:, None, -1, 1:3].expand(-1, candidate_count, -1)
    history_velocity = src[:, -1, 1:3] - src[:, -2, 1:3]
    candidate_positions = candidate_value_outputs[:, :, :, 1:3]

    current_distances = torch.linalg.vector_norm(
        own_prototypes - current_position[:, :, None, :],
        dim=-1,
    )
    current_cross_track, current_index = current_distances.min(dim=-1)
    previous_index = (current_index - 1).clamp_min(0)
    next_index = (current_index + 1).clamp_max(prototype_points - 1)
    gather_previous = previous_index[:, :, None, None].expand(-1, -1, 1, 2)
    gather_next = next_index[:, :, None, None].expand(-1, -1, 1, 2)
    local_tangent = (
        torch.gather(own_prototypes, 2, gather_next).squeeze(2)
        - torch.gather(own_prototypes, 2, gather_previous).squeeze(2)
    )

    endpoint = candidate_positions[:, :, -1, :]
    endpoint_distances = torch.linalg.vector_norm(
        own_prototypes - endpoint[:, :, None, :],
        dim=-1,
    )
    endpoint_cross_track, endpoint_index = endpoint_distances.min(dim=-1)
    current_progress = current_index.to(candidate_value_outputs.dtype) / max(prototype_points - 1, 1)
    endpoint_progress = endpoint_index.to(candidate_value_outputs.dtype) / max(prototype_points - 1, 1)
    progress_gain = endpoint_progress - current_progress

    own_point_distances = torch.linalg.vector_norm(
        candidate_positions[:, :, :, None, :] - own_prototypes[:, :, None, :, :],
        dim=-1,
    ).min(dim=-1).values
    own_mean_cross_track = own_point_distances.mean(dim=-1)
    own_max_cross_track = own_point_distances.max(dim=-1).values

    # Compare each candidate with other branches belonging to the same main route.
    all_distances = torch.linalg.vector_norm(
        candidate_positions[:, :, :, None, None, :]
        - prototypes[None, None, None, :, :, :],
        dim=-1,
    ).min(dim=-1).values.mean(dim=2)
    route_mask = model.route_to_subroute_mask[candidate_route_ids].bool()
    own_mask = F.one_hot(
        candidate_subroute_ids,
        num_classes=prototypes.size(0),
    ).bool()
    competing_distance = all_distances.masked_fill(~route_mask | own_mask, float("inf")).min(dim=-1).values
    competing_distance = torch.where(
        torch.isfinite(competing_distance),
        competing_distance,
        own_mean_cross_track,
    )
    branch_separation = competing_distance - own_mean_cross_track

    first_velocity = candidate_positions[:, :, 0, :] - current_position
    candidate_direction = endpoint - current_position
    history_alignment = cosine_similarity_2d(history_velocity[:, None, :], local_tangent)
    candidate_alignment = cosine_similarity_2d(candidate_direction, local_tangent)
    turn_alignment = cosine_similarity_2d(history_velocity[:, None, :], first_velocity)
    history_speed = torch.linalg.vector_norm(history_velocity, dim=-1)[:, None].clamp_min(1e-6)
    first_speed = torch.linalg.vector_norm(first_velocity, dim=-1).clamp_min(1e-6)
    speed_ratio_error = torch.abs(torch.log(first_speed / history_speed))

    return torch.stack(
        (
            current_cross_track,
            endpoint_cross_track,
            own_mean_cross_track,
            own_max_cross_track,
            current_progress,
            progress_gain,
            history_alignment,
            candidate_alignment,
            turn_alignment,
            branch_separation,
            speed_ratio_error,
        ),
        dim=-1,
    )


def build_qwen_reverse_features(
        src,
        candidate_route_ids,
        candidate_subroute_ids,
        candidate_value_outputs,
):
    """Measure whether each candidate can explain the observed history backwards."""
    batch_size, candidate_count, _, _ = candidate_value_outputs.shape
    prototypes = getattr(model, "subroute_prototypes", None)
    if prototypes is None:
        return torch.zeros(
            batch_size,
            candidate_count,
            16,
            device=candidate_value_outputs.device,
            dtype=candidate_value_outputs.dtype,
        )

    dtype = candidate_value_outputs.dtype
    prototypes = prototypes.to(device=src.device, dtype=dtype)
    own_prototypes = prototypes[candidate_subroute_ids]
    history_positions = src[:, :, 1:3].to(dtype=dtype)
    history_length = history_positions.size(1)
    prototype_points = prototypes.size(1)
    recent_count = min(4, history_length)

    history_to_own = torch.linalg.vector_norm(
        history_positions[:, None, :, None, :]
        - own_prototypes[:, :, None, :, :],
        dim=-1,
    )
    history_cross_track, nearest_indices = history_to_own.min(dim=-1)
    history_mean = history_cross_track.mean(dim=-1)
    history_max = history_cross_track.max(dim=-1).values
    history_recent = history_cross_track[:, :, -recent_count:].mean(dim=-1)
    history_last = history_cross_track[:, :, -1]
    long_approach = history_cross_track[:, :, 0] - history_last
    recent_approach = history_cross_track[:, :, -recent_count] - history_last

    progress = nearest_indices.to(dtype=dtype) / max(prototype_points - 1, 1)
    progress_gain = progress[:, :, -1] - progress[:, :, 0]
    if history_length > 1:
        progress_delta = progress[:, :, 1:] - progress[:, :, :-1]
        monotonic_progress = (progress_delta >= -1.0 / max(prototype_points - 1, 1)).to(dtype).mean(dim=-1)
    else:
        monotonic_progress = torch.ones_like(progress_gain)

    prototype_tangent = torch.empty_like(own_prototypes)
    prototype_tangent[:, :, 0] = own_prototypes[:, :, 1] - own_prototypes[:, :, 0]
    prototype_tangent[:, :, -1] = own_prototypes[:, :, -1] - own_prototypes[:, :, -2]
    prototype_tangent[:, :, 1:-1] = own_prototypes[:, :, 2:] - own_prototypes[:, :, :-2]
    gather_tangent_index = nearest_indices[..., None].expand(-1, -1, -1, 2)
    nearest_tangent = torch.gather(prototype_tangent, 2, gather_tangent_index)

    if history_length > 1:
        history_velocity = history_positions[:, 1:] - history_positions[:, :-1]
        expanded_velocity = history_velocity[:, None, :, :].expand(-1, candidate_count, -1, -1)
        tangent_alignment = cosine_similarity_2d(expanded_velocity, nearest_tangent[:, :, 1:])
        mean_tangent_alignment = tangent_alignment.mean(dim=-1)
        recent_alignment_count = min(3, tangent_alignment.size(-1))
        recent_tangent_alignment = tangent_alignment[:, :, -recent_alignment_count:].mean(dim=-1)
        last_history_velocity = history_velocity[:, -1]
        history_step_scale = torch.linalg.vector_norm(history_velocity, dim=-1).mean(dim=-1).clamp_min(1e-4)
    else:
        history_velocity = None
        mean_tangent_alignment = torch.zeros_like(history_mean)
        recent_tangent_alignment = torch.zeros_like(history_mean)
        last_history_velocity = torch.zeros(batch_size, 2, device=src.device, dtype=dtype)
        history_step_scale = torch.ones(batch_size, device=src.device, dtype=dtype)

    candidate_positions = candidate_value_outputs[:, :, :, 1:3]
    current_position = history_positions[:, None, -1, :]
    first_future_velocity = candidate_positions[:, :, 0] - current_position
    velocity_continuity_error = torch.linalg.vector_norm(
        first_future_velocity - last_history_velocity[:, None, :],
        dim=-1,
    ) / history_step_scale[:, None]

    reverse_count = min(4, max(history_length - 1, 0))
    if reverse_count > 0:
        reverse_steps = torch.arange(
            1,
            reverse_count + 1,
            device=src.device,
            dtype=dtype,
        ).view(1, 1, -1, 1)
        reverse_estimate = current_position[:, :, None, :] - first_future_velocity[:, :, None, :] * reverse_steps
        actual_reverse = torch.flip(
            history_positions[:, -(reverse_count + 1):-1],
            dims=[1],
        )[:, None, :, :]
        reverse_reconstruction_error = torch.linalg.vector_norm(
            reverse_estimate - actual_reverse,
            dim=-1,
        ).mean(dim=-1) / history_step_scale[:, None]
    else:
        reverse_reconstruction_error = torch.zeros_like(history_mean)

    if history_length > 2 and candidate_positions.size(2) > 1:
        previous_velocity = history_positions[:, -2] - history_positions[:, -3]
        history_turn = (
            previous_velocity[:, 0] * last_history_velocity[:, 1]
            - previous_velocity[:, 1] * last_history_velocity[:, 0]
        )
        second_future_velocity = candidate_positions[:, :, 1] - candidate_positions[:, :, 0]
        future_turn = (
            first_future_velocity[:, :, 0] * second_future_velocity[:, :, 1]
            - first_future_velocity[:, :, 1] * second_future_velocity[:, :, 0]
        )
        turn_sign_agreement = torch.sign(history_turn[:, None]) * torch.sign(future_turn)
    else:
        turn_sign_agreement = torch.zeros_like(history_mean)

    history_to_all = torch.linalg.vector_norm(
        history_positions[:, :, None, None, :]
        - prototypes[None, None, :, :, :],
        dim=-1,
    ).min(dim=-1).values
    all_history_mean = history_to_all.mean(dim=1)
    all_history_recent = history_to_all[:, -recent_count:].mean(dim=1)
    own_history_mean = torch.gather(all_history_mean, 1, candidate_subroute_ids)
    own_history_recent = torch.gather(all_history_recent, 1, candidate_subroute_ids)
    sibling_mask = model.route_to_subroute_mask[candidate_route_ids].bool()
    own_mask = F.one_hot(candidate_subroute_ids, num_classes=prototypes.size(0)).bool()
    valid_competitor = sibling_mask & ~own_mask
    competing_mean = all_history_mean[:, None, :].expand(-1, candidate_count, -1).masked_fill(
        ~valid_competitor,
        float("inf"),
    ).min(dim=-1).values
    competing_recent = all_history_recent[:, None, :].expand(-1, candidate_count, -1).masked_fill(
        ~valid_competitor,
        float("inf"),
    ).min(dim=-1).values
    competing_mean = torch.where(torch.isfinite(competing_mean), competing_mean, own_history_mean)
    competing_recent = torch.where(torch.isfinite(competing_recent), competing_recent, own_history_recent)
    sibling_mean_separation = competing_mean - own_history_mean
    sibling_recent_separation = competing_recent - own_history_recent
    normalized_branch_evidence = sibling_recent_separation / (
        competing_recent + own_history_recent
    ).clamp_min(1e-4)

    features = torch.stack(
        (
            history_mean,
            history_max,
            history_recent,
            history_last,
            long_approach,
            recent_approach,
            progress_gain,
            monotonic_progress,
            mean_tangent_alignment,
            recent_tangent_alignment,
            reverse_reconstruction_error,
            velocity_continuity_error,
            turn_sign_agreement,
            sibling_mean_separation,
            sibling_recent_separation,
            normalized_branch_evidence,
        ),
        dim=-1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)


def build_qwen_candidate_features(
        src,
        route_logits,
        subroute_logits,
        candidate_route_ids,
        candidate_subroute_ids,
        candidate_value_outputs,
        candidate_is_base,
):
    numeric_features = model.candidate_numeric_features(
        src,
        route_logits,
        subroute_logits,
        candidate_route_ids,
        candidate_subroute_ids,
        candidate_value_outputs,
        candidate_is_base=candidate_is_base,
    )
    route_one_hot = F.one_hot(
        candidate_route_ids,
        num_classes=route_logits.size(-1),
    ).to(dtype=numeric_features.dtype)
    subroute_one_hot = F.one_hot(
        candidate_subroute_ids,
        num_classes=subroute_logits.size(-1),
    ).to(dtype=numeric_features.dtype)
    relative_positions = (
        candidate_value_outputs[:, :, :, 1:3]
        - src[:, None, -1:, 1:3]
    ).reshape(candidate_value_outputs.size(0), candidate_value_outputs.size(1), -1)
    maritime_features = build_qwen_maritime_features(
        src,
        candidate_route_ids,
        candidate_subroute_ids,
        candidate_value_outputs,
    )
    reverse_features = build_qwen_reverse_features(
        src,
        candidate_route_ids,
        candidate_subroute_ids,
        candidate_value_outputs,
    )
    return torch.cat(
        (
            numeric_features,
            route_one_hot,
            subroute_one_hot,
            relative_positions,
            maritime_features,
            reverse_features,
        ),
        dim=-1,
    )


def candidate_score_margin(logits):
    probabilities = F.softmax(logits, dim=-1)
    top_values = torch.topk(probabilities, k=min(2, probabilities.size(-1)), dim=-1).values
    if top_values.size(1) == 1:
        return torch.ones_like(top_values[:, 0])
    return top_values[:, 0] - top_values[:, 1]


def normalize_candidate_scores(logits):
    return (
        logits - logits.mean(dim=-1, keepdim=True)
    ) / logits.std(dim=-1, keepdim=True).clamp_min(1e-4)


def qwen_soft_ranking_loss(
        logits,
        candidate_cost,
        sample_weight=None,
        selector_logits=None,
        fusion_weight=None,
):
    temperature = max(float(args.qwen_cost_temperature), 1e-4)
    target_probability = F.softmax(-candidate_cost / temperature, dim=-1)
    ranking_loss = -torch.sum(
        target_probability * F.log_softmax(logits, dim=-1),
        dim=-1,
    )
    target_utility = normalize_candidate_scores(-candidate_cost)
    normalized_logits = normalize_candidate_scores(logits)
    score_regression = F.smooth_l1_loss(
        normalized_logits,
        target_utility,
        reduction="none",
    ).mean(dim=-1)
    loss = ranking_loss + args.qwen_cost_regression_weight * score_regression

    if args.qwen_pairwise_weight > 0 and logits.size(1) > 1:
        winner = torch.argmin(candidate_cost, dim=-1)
        winner_score = logits.gather(1, winner.unsqueeze(1))
        winner_cost = candidate_cost.gather(1, winner.unsqueeze(1))
        score_advantage = winner_score - logits
        cost_gap = candidate_cost - winner_cost
        valid_negative = cost_gap >= float(args.qwen_pairwise_min_cost_gap)
        valid_negative.scatter_(1, winner.unsqueeze(1), False)
        pairwise_values = F.relu(float(args.qwen_pairwise_margin) - score_advantage)
        pairwise_weight = valid_negative.to(dtype=pairwise_values.dtype)
        pairwise_loss = torch.sum(pairwise_values * pairwise_weight, dim=-1) / pairwise_weight.sum(
            dim=-1
        ).clamp_min(1.0)
        loss = loss + args.qwen_pairwise_weight * pairwise_loss

    if (
            selector_logits is not None
            and getattr(args, "qwen_fused_loss_weight", 0.0) > 0
    ):
        fused_weight = args.qwen_reranker_weight if fusion_weight is None else fusion_weight
        fused_logits = selector_logits + float(fused_weight) * normalized_logits
        fused_ranking_loss = -torch.sum(
            target_probability * F.log_softmax(fused_logits, dim=-1),
            dim=-1,
        )
        fused_score_regression = F.smooth_l1_loss(
            normalize_candidate_scores(fused_logits),
            target_utility,
            reduction="none",
        ).mean(dim=-1)
        fused_loss = fused_ranking_loss + args.qwen_cost_regression_weight * fused_score_regression
        loss = loss + args.qwen_fused_loss_weight * fused_loss

    if sample_weight is None:
        return loss.mean()
    sample_weight = sample_weight.to(device=loss.device, dtype=loss.dtype).clamp_min(0.0)
    return torch.sum(loss * sample_weight) / sample_weight.sum().clamp_min(1e-6)


def qwen_example_stats(selector_scores, trajectory_costs, winners):
    base_index, _, _ = select_candidate_indices(selector_scores)
    base_cost = trajectory_costs.gather(1, base_index.unsqueeze(1)).squeeze(1)
    oracle_cost = trajectory_costs.gather(1, winners.unsqueeze(1)).squeeze(1)
    sorted_costs = torch.sort(trajectory_costs, dim=-1).values
    if sorted_costs.size(1) > 1:
        winner_gap = sorted_costs[:, 1] - sorted_costs[:, 0]
    else:
        winner_gap = torch.zeros_like(base_cost)
    oracle_gain = base_cost - oracle_cost
    high_gain = (
        base_index.ne(winners)
        & (oracle_gain >= float(args.qwen_min_oracle_gain_nmi))
        & (winner_gap >= float(args.qwen_min_winner_gap_nmi))
    )
    gain_scale = max(float(args.qwen_min_oracle_gain_nmi), 1e-4)
    sample_weight = 1.0 + float(args.qwen_gain_weight) * (
        oracle_gain.clamp_min(0.0) / gain_scale
    ).clamp(max=5.0)
    sample_weight = torch.where(high_gain, sample_weight, torch.ones_like(sample_weight))
    return base_index, base_cost, oracle_gain, winner_gap, high_gain, sample_weight


def fuse_qwen_candidate_scores(
        selector_logits,
        qwen_logits,
        fusion_weight,
        selector_margin_threshold,
        qwen_margin_threshold,
        uncertain_only=True,
):
    normalized_qwen = normalize_candidate_scores(qwen_logits)
    selector_gate = candidate_score_margin(selector_logits) <= selector_margin_threshold
    if not uncertain_only:
        selector_gate = torch.ones_like(selector_gate)
    qwen_confident = candidate_score_margin(normalized_qwen) >= qwen_margin_threshold
    applied = selector_gate & qwen_confident
    fused_logits = selector_logits + fusion_weight * normalized_qwen
    final_logits = torch.where(applied[:, None], fused_logits, selector_logits)
    return final_logits, applied


def select_candidate_indices_with_threshold(selector_logits, confidence_threshold, logit_margin):
    selector_probs = F.softmax(selector_logits, dim=-1)
    branch_score, branch_offset = torch.max(selector_logits[:, 1:], dim=-1)
    branch_index = branch_offset + 1
    branch_probability = selector_probs.gather(1, branch_index.unsqueeze(1)).squeeze(1)
    switch_to_branch = (
        (branch_probability >= confidence_threshold)
        & ((branch_score - selector_logits[:, 0]) >= logit_margin)
    )
    selected_index = torch.where(
        switch_to_branch,
        branch_index,
        torch.zeros_like(branch_index),
    )
    return selected_index, switch_to_branch, branch_probability


def select_candidate_indices(selector_logits):
    confidence_threshold = args.candidate_switch_confidence_threshold
    logit_margin = args.candidate_switch_logit_margin
    if candidate_selection_calibration is not None:
        confidence_threshold = candidate_selection_calibration["confidence_threshold"]
        logit_margin = candidate_selection_calibration["logit_margin"]
    return select_candidate_indices_with_threshold(
        selector_logits,
        confidence_threshold,
        logit_margin,
    )


def weighted_loss_mean(loss_values, sample_weights=None):
    if sample_weights is None:
        return loss_values.mean()
    sample_weights = sample_weights.to(device=loss_values.device, dtype=loss_values.dtype).reshape(-1)
    if sample_weights.numel() != loss_values.numel():
        raise ValueError("Sample weights must match the number of per-sample losses.")
    weight_sum = sample_weights.sum()
    if float(weight_sum.detach().item()) <= 0:
        return loss_values.sum() * 0.0
    return torch.sum(loss_values * sample_weights) / weight_sum


def candidate_soft_cost_loss(selector_logits, candidate_cost, sample_weights=None):
    temperature = max(float(args.candidate_cost_temperature), 1e-4)
    target_probability = F.softmax(-candidate_cost.detach() / temperature, dim=-1)
    ranking_loss = -torch.sum(
        target_probability * F.log_softmax(selector_logits, dim=-1),
        dim=-1,
    )
    ranking_loss = weighted_loss_mean(ranking_loss, sample_weights)
    if args.candidate_cost_regression_weight <= 0:
        return ranking_loss
    score_regression = F.smooth_l1_loss(
        normalize_candidate_scores(selector_logits),
        normalize_candidate_scores(-candidate_cost.detach()),
        reduction="none",
    ).mean(dim=-1)
    score_regression = weighted_loss_mean(score_regression, sample_weights)
    return ranking_loss + args.candidate_cost_regression_weight * score_regression


def prepare_candidate_prediction(
        delta,
        src,
        base_value_output,
        route_logits,
        subroute_logits,
        intent_feature,
        route_targets=None,
        subroute_targets=None,
        include_targets=False,
        target_include_mask=None,
        apply_qwen=True,
        voyage_contexts=None,
):
    branch_route_ids, branch_subroute_ids = build_hierarchical_candidates(
        route_logits,
        subroute_logits,
        args.candidate_count,
        subroutes_per_route=args.candidate_subroutes_per_route,
        pool_strategy=args.candidate_pool_strategy,
        max_subroute_candidates=args.candidate_max_subroutes,
        route_targets=route_targets,
        subroute_targets=subroute_targets,
        include_targets=include_targets,
        target_include_mask=target_include_mask,
    )
    candidate_raw_outputs = model.decode_candidates(
        delta,
        src,
        branch_route_ids,
        branch_subroute_ids,
    )
    branch_value_outputs = compose_candidate_outputs(candidate_raw_outputs, src)
    base_route_ids = torch.argmax(route_logits.detach(), dim=-1, keepdim=True)
    base_subroute_ids = torch.argmax(subroute_logits.detach(), dim=-1, keepdim=True)
    candidate_route_ids = torch.cat((base_route_ids, branch_route_ids), dim=1)
    candidate_subroute_ids = torch.cat((base_subroute_ids, branch_subroute_ids), dim=1)
    candidate_value_outputs = torch.cat(
        (base_value_output[:, None, :, :], branch_value_outputs),
        dim=1,
    )
    candidate_is_base = torch.zeros(
        candidate_route_ids.shape,
        device=candidate_route_ids.device,
        dtype=candidate_value_outputs.dtype,
    )
    candidate_is_base[:, 0] = 1.0
    selector_logits = model.score_candidates(
        src,
        intent_feature,
        route_logits,
        subroute_logits,
        candidate_route_ids,
        candidate_subroute_ids,
        candidate_value_outputs,
        candidate_is_base=candidate_is_base,
    )
    base_selector_logits = selector_logits
    qwen_logits = None
    qwen_applied = torch.zeros(
        selector_logits.size(0),
        device=selector_logits.device,
        dtype=torch.bool,
    )
    if (
            apply_qwen
            and args.use_qwen_reranker
            and qwen_reranker_runtime_active
            and qwen_reranker is not None
    ):
        selector_margin_threshold = float(
            getattr(qwen_reranker, "selector_margin_threshold", args.qwen_uncertainty_margin_threshold)
        )
        selector_gate = candidate_score_margin(selector_logits) <= selector_margin_threshold
        if not args.qwen_uncertain_only:
            selector_gate = torch.ones_like(selector_gate)
        if selector_gate.any():
            qwen_features = build_qwen_candidate_features(
                src,
                route_logits,
                subroute_logits,
                candidate_route_ids,
                candidate_subroute_ids,
                candidate_value_outputs,
                candidate_is_base,
            )
            qwen_logits = torch.zeros_like(selector_logits)
            context_kwargs = {}
            if voyage_contexts is not None:
                selected_contexts = np.asarray(voyage_contexts, dtype=object)[
                    selector_gate.detach().cpu().numpy()
                ]
                context_ids, context_mask = qwen_reranker.tokenize_contexts(
                    selected_contexts,
                    max_length=args.qwen_context_max_tokens,
                )
                context_kwargs = {
                    "context_input_ids": context_ids.to(device),
                    "context_attention_mask": context_mask.to(device),
                }
            reranked = qwen_reranker(
                src[selector_gate],
                qwen_features[selector_gate],
                **context_kwargs,
            ).to(dtype=selector_logits.dtype)
            qwen_logits[selector_gate] = reranked
            qwen_confident = candidate_score_margin(
                normalize_candidate_scores(reranked)
            ) >= float(getattr(qwen_reranker, "qwen_margin_threshold", 0.0))
            qwen_applied[selector_gate] = qwen_confident
            fusion_weight = float(
                getattr(qwen_reranker, "fusion_weight", args.qwen_reranker_weight)
            )
            fused = selector_logits + fusion_weight * normalize_candidate_scores(qwen_logits)
            selector_logits = torch.where(
                qwen_applied[:, None],
                fused,
                selector_logits,
            )

    selected_index, switch_to_branch, branch_probability = select_candidate_indices(
        selector_logits
    )
    gather_index = selected_index[:, None, None, None].expand(
        -1,
        1,
        candidate_value_outputs.size(2),
        candidate_value_outputs.size(3),
    )
    selected_output = torch.gather(candidate_value_outputs, 1, gather_index).squeeze(1)
    return {
        "route_ids": candidate_route_ids,
        "subroute_ids": candidate_subroute_ids,
        "branch_route_ids": branch_route_ids,
        "branch_subroute_ids": branch_subroute_ids,
        "is_base": candidate_is_base,
        "outputs": candidate_value_outputs,
        "base_selector_logits": base_selector_logits,
        "selector_logits": selector_logits,
        "qwen_logits": qwen_logits,
        "qwen_applied": qwen_applied,
        "selected_index": selected_index,
        "switch_to_branch": switch_to_branch,
        "branch_probability": branch_probability,
        "selected_output": selected_output,
    }


def calibrate_candidate_selection(
        X_data,
        route_targets=None,
        subroute_targets=None,
        voyage_contexts=None,
):
    global candidate_selection_calibration
    candidate_selection_calibration = None
    if not args.use_candidate_selection_calibration or not args.use_candidate_selector:
        return None
    if len(X_data) == 0:
        return None

    logger = logging.getLogger()
    selector_scores = []
    trajectory_costs = []
    model.eval()
    with torch.no_grad():
        eval_indices = np.arange(len(X_data), dtype="int32")
        for start in range(0, len(eval_indices), batch_size):
            batch_indices = eval_indices[start:start + batch_size]
            delta = torch.stack([X_data[i][:input_length, in_cols] for i in batch_indices]).to(device)
            src = torch.stack([X_data[i][:input_length, in_cols] for i in batch_indices]).to(device)
            value_target = torch.stack([
                X_data[i][input_length:input_length + target_length, src_cols]
                for i in batch_indices
            ]).to(device)
            route_target = None if route_targets is None else route_targets[batch_indices].to(device)
            subroute_target = None if subroute_targets is None else subroute_targets[batch_indices].to(device)
            semantic_feature = semantic_features_for_contexts(
                None if voyage_contexts is None else voyage_contexts[batch_indices]
            )
            _, raw_output, route_logits, subroute_logits, _, subroute_feature, _, _ = unpack_model_output(
                model(delta, src, semantic_feature=semantic_feature)
            )
            value_output = compose_value_output(raw_output, src)
            candidate_result = prepare_candidate_prediction(
                delta,
                src,
                value_output,
                route_logits,
                subroute_logits,
                subroute_feature,
                route_targets=route_target,
                subroute_targets=subroute_target,
                include_targets=False,
                apply_qwen=False,
            )
            candidate_cost, _, _ = candidate_trajectory_costs(
                candidate_result["outputs"],
                value_target,
            )
            selector_scores.append(candidate_result["base_selector_logits"].detach().cpu())
            trajectory_costs.append(candidate_cost.detach().cpu())

    selector_scores = torch.cat(selector_scores, dim=0)
    trajectory_costs = torch.cat(trajectory_costs, dim=0)
    winner = torch.argmin(trajectory_costs, dim=-1)
    default_index, default_switch, _ = select_candidate_indices_with_threshold(
        selector_scores,
        args.candidate_switch_confidence_threshold,
        args.candidate_switch_logit_margin,
    )
    default_cost = trajectory_costs.gather(1, default_index.unsqueeze(1)).squeeze(1)
    default_mean_cost = float(default_cost.mean().item())
    default_accuracy = 100.0 * float(default_index.eq(winner).float().mean().item())

    confidence_grid = sorted(set((
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        float(args.candidate_switch_confidence_threshold),
    )))
    margin_grid = sorted(set((
        -0.30,
        -0.20,
        -0.10,
        -0.05,
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        float(args.candidate_switch_logit_margin),
    )))
    max_switch_ratio = float(args.candidate_calibration_max_switch_ratio)
    best = {
        "confidence_threshold": float(args.candidate_switch_confidence_threshold),
        "logit_margin": float(args.candidate_switch_logit_margin),
        "trajectory_cost": default_mean_cost,
        "winner_accuracy": default_accuracy,
        "switch_ratio": float(default_switch.float().mean().item()),
    }
    for confidence_threshold in confidence_grid:
        for logit_margin in margin_grid:
            selected_index, switch_mask, _ = select_candidate_indices_with_threshold(
                selector_scores,
                confidence_threshold,
                logit_margin,
            )
            switch_ratio = float(switch_mask.float().mean().item())
            if switch_ratio > max_switch_ratio:
                continue
            selected_cost = trajectory_costs.gather(1, selected_index.unsqueeze(1)).squeeze(1)
            mean_cost = float(selected_cost.mean().item())
            accuracy = 100.0 * float(selected_index.eq(winner).float().mean().item())
            if (
                    mean_cost < best["trajectory_cost"] - 1e-9
                    or (
                        abs(mean_cost - best["trajectory_cost"]) <= 1e-9
                        and accuracy > best["winner_accuracy"]
                    )
            ):
                best = {
                    "confidence_threshold": float(confidence_threshold),
                    "logit_margin": float(logit_margin),
                    "trajectory_cost": mean_cost,
                    "winner_accuracy": accuracy,
                    "switch_ratio": switch_ratio,
                }

    cost_gain = default_mean_cost - best["trajectory_cost"]
    accepted = cost_gain >= float(args.candidate_calibration_min_cost_gain)
    if accepted:
        candidate_selection_calibration = best
    logger.info(
        "Candidate selection calibration: default cost %.4f nmi, acc %.1f%%, switch %.1f%%; "
        "best cost %.4f nmi, acc %.1f%%, switch %.1f%%, p>=%.2f, margin>=%.2f, "
        "gain %.4f nmi, accepted=%s.",
        default_mean_cost,
        default_accuracy,
        100.0 * float(default_switch.float().mean().item()),
        best["trajectory_cost"],
        best["winner_accuracy"],
        100.0 * best["switch_ratio"],
        best["confidence_threshold"],
        best["logit_margin"],
        cost_gain,
        accepted,
    )
    return best if accepted else None


def collect_qwen_reranker_examples(
        X_data,
        route_targets,
        subroute_targets,
        max_windows,
        seed,
        prioritize_hard,
        context_data=None,
):
    if max_windows <= 0 or len(X_data) == 0:
        raise ValueError("Qwen reranker example count must be positive.")
    if route_targets is None or subroute_targets is None:
        raise ValueError("Qwen reranker training requires route and subroute labels.")

    rng = np.random.default_rng(seed)
    pool_size = min(len(X_data), max(max_windows, max_windows * 3 if prioritize_hard else max_windows))
    pool_indices = rng.choice(len(X_data), size=pool_size, replace=False)
    histories = []
    features = []
    winners = []
    selector_scores = []
    trajectory_costs = []
    hard_masks = []
    same_route_hard_masks = []
    contexts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, pool_size, batch_size):
            batch_indices_np = pool_indices[start:start + batch_size]
            batch_indices = torch.as_tensor(batch_indices_np, dtype=torch.long)
            delta = torch.stack([X_data[int(i)][:input_length, in_cols] for i in batch_indices_np]).to(device)
            src = torch.stack([X_data[int(i)][:input_length, in_cols] for i in batch_indices_np]).to(device)
            value_target = torch.stack([
                X_data[int(i)][input_length:input_length + target_length, src_cols]
                for i in batch_indices_np
            ]).to(device)
            route_target = route_targets[batch_indices].to(device)
            subroute_target = subroute_targets[batch_indices].to(device)
            semantic_feature = semantic_features_for_contexts(
                None if context_data is None else context_data[batch_indices_np]
            )
            _, raw_output, route_logits, subroute_logits, _, subroute_feature, _, _ = unpack_model_output(
                model(delta, src, semantic_feature=semantic_feature)
            )
            value_output = compose_value_output(raw_output, src)
            candidate_result = prepare_candidate_prediction(
                delta,
                src,
                value_output,
                route_logits,
                subroute_logits,
                subroute_feature,
                route_targets=route_target,
                subroute_targets=subroute_target,
                include_targets=False,
                apply_qwen=False,
            )
            candidate_cost, _, _ = candidate_trajectory_costs(
                candidate_result["outputs"],
                value_target,
            )
            winner = torch.argmin(candidate_cost, dim=-1)
            qwen_features = build_qwen_candidate_features(
                src,
                route_logits,
                subroute_logits,
                candidate_result["route_ids"],
                candidate_result["subroute_ids"],
                candidate_result["outputs"],
                candidate_result["is_base"],
            )
            selector_winner, _, _ = select_candidate_indices(
                candidate_result["base_selector_logits"]
            )
            hard = selector_winner.ne(winner)
            selector_route = candidate_result["route_ids"].gather(
                1,
                selector_winner.unsqueeze(1),
            ).squeeze(1)
            selector_subroute = candidate_result["subroute_ids"].gather(
                1,
                selector_winner.unsqueeze(1),
            ).squeeze(1)
            winner_route = candidate_result["route_ids"].gather(
                1,
                winner.unsqueeze(1),
            ).squeeze(1)
            winner_subroute = candidate_result["subroute_ids"].gather(
                1,
                winner.unsqueeze(1),
            ).squeeze(1)
            same_route_hard = (
                hard
                & selector_route.eq(winner_route)
                & selector_subroute.ne(winner_subroute)
            )
            histories.append(src.detach().cpu())
            features.append(qwen_features.detach().cpu())
            winners.append(winner.detach().cpu())
            selector_scores.append(candidate_result["base_selector_logits"].detach().cpu())
            trajectory_costs.append(candidate_cost.detach().cpu())
            hard_masks.append(hard.detach().cpu())
            same_route_hard_masks.append(same_route_hard.detach().cpu())
            if context_data is None:
                contexts.extend([""] * len(batch_indices_np))
            else:
                contexts.extend(str(context_data[int(i)]) for i in batch_indices_np)

    histories = torch.cat(histories)
    features = torch.cat(features)
    winners = torch.cat(winners)
    selector_scores = torch.cat(selector_scores)
    trajectory_costs = torch.cat(trajectory_costs)
    hard_masks = torch.cat(hard_masks).bool()
    same_route_hard_masks = torch.cat(same_route_hard_masks).bool()
    contexts = np.asarray(contexts, dtype=object)
    (
        _,
        _,
        oracle_gain,
        winner_gap,
        high_gain_masks,
        sample_weights,
    ) = qwen_example_stats(selector_scores, trajectory_costs, winners)
    sample_weights = sample_weights * torch.where(
        same_route_hard_masks,
        torch.full_like(sample_weights, 1.0 + float(args.qwen_same_route_hard_weight)),
        torch.ones_like(sample_weights),
    )
    if prioritize_hard and len(histories) > max_windows:
        focus_enabled = bool(getattr(args, "qwen_focus_high_gain", True))
        focus_indices = torch.nonzero(
            (high_gain_masks | same_route_hard_masks)
            if focus_enabled
            else same_route_hard_masks,
            as_tuple=False,
        ).flatten().numpy()
        hard_pool = hard_masks & ~high_gain_masks & ~same_route_hard_masks
        hard_indices = torch.nonzero(hard_pool, as_tuple=False).flatten().numpy()
        easy_indices = torch.nonzero(~hard_masks, as_tuple=False).flatten().numpy()
        requested_hard = int(round(max_windows * args.qwen_hard_sample_ratio))
        chosen = []
        requested_focus = min(requested_hard, len(focus_indices))
        if requested_focus:
            chosen.extend(rng.choice(focus_indices, size=requested_focus, replace=False).tolist())
        requested_hard = min(requested_hard - requested_focus, len(hard_indices))
        if requested_hard:
            chosen.extend(rng.choice(hard_indices, size=requested_hard, replace=False).tolist())
        requested_easy = min(max_windows - len(chosen), len(easy_indices))
        if requested_easy:
            chosen.extend(rng.choice(easy_indices, size=requested_easy, replace=False).tolist())
        remaining = max_windows - len(chosen)
        if remaining:
            unused = np.setdiff1d(
                np.arange(len(histories)),
                np.asarray(chosen, dtype=np.int64),
                assume_unique=False,
            )
            chosen.extend(rng.choice(unused, size=min(remaining, len(unused)), replace=False).tolist())
        chosen = torch.as_tensor(chosen, dtype=torch.long)
        histories = histories[chosen]
        features = features[chosen]
        winners = winners[chosen]
        selector_scores = selector_scores[chosen]
        trajectory_costs = trajectory_costs[chosen]
        hard_masks = hard_masks[chosen]
        same_route_hard_masks = same_route_hard_masks[chosen]
        oracle_gain = oracle_gain[chosen]
        winner_gap = winner_gap[chosen]
        high_gain_masks = high_gain_masks[chosen]
        sample_weights = sample_weights[chosen]
        contexts = contexts[chosen.numpy()]
    elif len(histories) > max_windows:
        chosen = torch.as_tensor(rng.choice(len(histories), size=max_windows, replace=False), dtype=torch.long)
        histories = histories[chosen]
        features = features[chosen]
        winners = winners[chosen]
        selector_scores = selector_scores[chosen]
        trajectory_costs = trajectory_costs[chosen]
        hard_masks = hard_masks[chosen]
        same_route_hard_masks = same_route_hard_masks[chosen]
        oracle_gain = oracle_gain[chosen]
        winner_gap = winner_gap[chosen]
        high_gain_masks = high_gain_masks[chosen]
        sample_weights = sample_weights[chosen]
        contexts = contexts[chosen.numpy()]

    logging.getLogger().info(
        "Qwen examples collected: %d windows, hard %d (%.1f%%), same_route_hard %d (%.1f%%), "
        "high_gain %d (%.1f%%), "
        "mean_gain %.4f nmi, mean_gap %.4f nmi, candidates %d, feature_dim %d.",
        len(histories),
        int(hard_masks.sum().item()),
        100.0 * float(hard_masks.float().mean().item()),
        int(same_route_hard_masks.sum().item()),
        100.0 * float(same_route_hard_masks.float().mean().item()),
        int(high_gain_masks.sum().item()),
        100.0 * float(high_gain_masks.float().mean().item()),
        float(oracle_gain.clamp_min(0.0).mean().item()),
        float(winner_gap.mean().item()),
        features.size(1),
        features.size(2),
    )
    return histories, features, winners, selector_scores, trajectory_costs, contexts, sample_weights


def attach_qwen_context_tokens(reranker, examples):
    histories, features, winners, selector_scores, trajectory_costs, contexts, sample_weights = examples
    context_input_ids = None
    context_attention_mask = None
    if contexts is not None and any(bool(str(item).strip()) for item in contexts):
        context_input_ids, context_attention_mask = reranker.tokenize_contexts(
            contexts,
            max_length=args.qwen_context_max_tokens,
        )
    return (
        histories,
        features,
        winners,
        selector_scores,
        trajectory_costs,
        context_input_ids,
        context_attention_mask,
        sample_weights,
    )


def evaluate_qwen_reranker(reranker, examples, eval_batch_size):
    (
        histories,
        features,
        winners,
        selector_scores,
        trajectory_costs,
        context_input_ids,
        context_attention_mask,
        sample_weights,
    ) = examples
    reranker.eval()
    total_loss = 0.0
    total = 0
    qwen_correct = 0
    fused_correct = 0
    base_cost_sum = 0.0
    fused_cost_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(histories), eval_batch_size):
            end = start + eval_batch_size
            history = histories[start:end].to(device)
            candidate_features = features[start:end].to(device)
            winner = winners[start:end].to(device)
            selector = selector_scores[start:end].to(device)
            candidate_cost = trajectory_costs[start:end].to(device)
            sample_weight = sample_weights[start:end].to(device)
            context_kwargs = {}
            if context_input_ids is not None:
                context_kwargs = {
                    "context_input_ids": context_input_ids[start:end].to(device),
                    "context_attention_mask": context_attention_mask[start:end].to(device),
                }
            qwen_logits = reranker(history, candidate_features, **context_kwargs)
            loss = qwen_soft_ranking_loss(
                qwen_logits,
                candidate_cost,
                sample_weight=sample_weight,
                selector_logits=selector,
                fusion_weight=reranker.fusion_weight,
            )
            fused_logits, _ = fuse_qwen_candidate_scores(
                selector,
                qwen_logits,
                reranker.fusion_weight,
                reranker.selector_margin_threshold,
                reranker.qwen_margin_threshold,
                uncertain_only=args.qwen_uncertain_only,
            )
            base_index, _, _ = select_candidate_indices(selector)
            fused_index, _, _ = select_candidate_indices(fused_logits)
            count = winner.numel()
            total_loss += float(loss.item()) * count
            total += count
            qwen_correct += int(torch.argmax(qwen_logits, dim=-1).eq(winner).sum().item())
            fused_correct += int(fused_index.eq(winner).sum().item())
            base_cost_sum += float(candidate_cost.gather(1, base_index.unsqueeze(1)).sum().item())
            fused_cost_sum += float(candidate_cost.gather(1, fused_index.unsqueeze(1)).sum().item())
    return (
        total_loss / max(total, 1),
        100.0 * qwen_correct / max(total, 1),
        100.0 * fused_correct / max(total, 1),
        base_cost_sum / max(total, 1),
        fused_cost_sum / max(total, 1),
    )


def collect_qwen_validation_logits(reranker, examples, eval_batch_size):
    histories, features, _, _, _, context_input_ids, context_attention_mask, _ = examples
    outputs = []
    reranker.eval()
    with torch.no_grad():
        for start in range(0, len(histories), eval_batch_size):
            end = start + eval_batch_size
            kwargs = {}
            if context_input_ids is not None:
                kwargs = {
                    "context_input_ids": context_input_ids[start:end].to(device),
                    "context_attention_mask": context_attention_mask[start:end].to(device),
                }
            outputs.append(reranker(
                histories[start:end].to(device),
                features[start:end].to(device),
                **kwargs,
            ).float().cpu())
    return torch.cat(outputs, dim=0)


def calibrate_qwen_fusion(reranker, examples, eval_batch_size):
    _, _, winners, selector_scores, trajectory_costs, _, _, _ = examples
    qwen_logits = collect_qwen_validation_logits(reranker, examples, eval_batch_size)
    base_index, _, _ = select_candidate_indices(selector_scores)
    base_cost = trajectory_costs.gather(1, base_index.unsqueeze(1)).squeeze(1)
    base_accuracy = 100.0 * float(base_index.eq(winners).float().mean().item())

    selector_margin = candidate_score_margin(selector_scores)
    qwen_margin = candidate_score_margin(normalize_candidate_scores(qwen_logits))
    max_apply_ratio = min(max(float(args.qwen_calibration_max_apply_ratio), 0.05), 1.0)
    gate_fractions = sorted(set((0.10, max_apply_ratio / 2.0, max_apply_ratio)))
    selector_thresholds = [
        float(torch.quantile(selector_margin, min(max(fraction, 0.01), 1.0)).item())
        for fraction in gate_fractions
    ]
    qwen_thresholds = sorted(set((
        0.0,
        float(torch.quantile(qwen_margin, 0.25).item()),
        float(torch.quantile(qwen_margin, 0.50).item()),
    )))
    maximum_weight = max(float(args.qwen_reranker_weight), 0.1)
    fusion_weights = sorted(set((
        0.1,
        maximum_weight * 0.5,
        maximum_weight * 0.75,
        maximum_weight,
    )))

    best = {
        "fusion_weight": 0.0,
        "selector_margin_threshold": 0.0,
        "qwen_margin_threshold": 1.0,
        "winner_accuracy": base_accuracy,
        "trajectory_cost": float(base_cost.mean().item()),
        "applied_ratio": 0.0,
        "selected_index": base_index,
        "selected_cost": base_cost,
    }
    for fusion_weight in fusion_weights:
        for selector_threshold in selector_thresholds:
            for qwen_threshold in qwen_thresholds:
                fused_logits, applied = fuse_qwen_candidate_scores(
                    selector_scores,
                    qwen_logits,
                    fusion_weight,
                    selector_threshold,
                    qwen_threshold,
                    uncertain_only=args.qwen_uncertain_only,
                )
                selected_index, _, _ = select_candidate_indices(fused_logits)
                selected_cost = trajectory_costs.gather(
                    1,
                    selected_index.unsqueeze(1),
                ).squeeze(1)
                mean_cost = float(selected_cost.mean().item())
                accuracy = 100.0 * float(selected_index.eq(winners).float().mean().item())
                if (
                        mean_cost < best["trajectory_cost"] - 1e-9
                        or (
                            abs(mean_cost - best["trajectory_cost"]) <= 1e-9
                            and accuracy > best["winner_accuracy"]
                        )
                ):
                    best = {
                        "fusion_weight": float(fusion_weight),
                        "selector_margin_threshold": float(selector_threshold),
                        "qwen_margin_threshold": float(qwen_threshold),
                        "winner_accuracy": accuracy,
                        "trajectory_cost": mean_cost,
                        "applied_ratio": float(applied.float().mean().item()),
                        "selected_index": selected_index,
                        "selected_cost": selected_cost,
                    }

    improvement = base_cost - best["selected_cost"]
    standard_error = float(improvement.std(unbiased=False).item()) / max(len(improvement) ** 0.5, 1.0)
    best["base_winner_accuracy"] = base_accuracy
    best["base_trajectory_cost"] = float(base_cost.mean().item())
    best["cost_gain"] = float(improvement.mean().item())
    best["cost_gain_lower_95"] = best["cost_gain"] - 1.96 * standard_error
    return best


def train_qwen_candidate_reranker(
        X_train_data,
        train_route_targets,
        train_subroute_targets,
        X_valid_data,
        valid_route_targets,
        valid_subroute_targets,
        adapter_path,
        fold,
        train_context_data=None,
        valid_context_data=None,
):
    logger = logging.getLogger()
    train_examples = collect_qwen_reranker_examples(
        X_train_data,
        train_route_targets,
        train_subroute_targets,
        args.qwen_train_max_windows,
        seed=4200 + fold,
        prioritize_hard=True,
        context_data=train_context_data,
    )
    valid_examples = collect_qwen_reranker_examples(
        X_valid_data,
        valid_route_targets,
        valid_subroute_targets,
        args.qwen_valid_max_windows,
        seed=8400 + fold,
        prioritize_hard=False,
        context_data=valid_context_data,
    )
    candidate_dim = int(train_examples[1].size(-1))
    reranker = QwenCandidateReranker(
        args.qwen_model_path,
        history_dim=len(in_cols),
        candidate_dim=candidate_dim,
        adapter_dim=args.qwen_adapter_dim,
        gradient_checkpointing=args.qwen_gradient_checkpointing,
    ).to(device)
    reranker.fusion_weight = float(args.qwen_reranker_weight)
    reranker.selector_margin_threshold = float(args.qwen_uncertainty_margin_threshold)
    reranker.qwen_margin_threshold = 0.0
    train_examples = attach_qwen_context_tokens(reranker, train_examples)
    valid_examples = attach_qwen_context_tokens(reranker, valid_examples)
    logger.info(
        "Qwen reranker loaded from %s, trainable parameters %.6e, adapter path %s.",
        args.qwen_model_path,
        count_parameters(reranker),
        adapter_path,
    )
    qwen_optimizer = optim.AdamW(
        reranker.trainable_parameters(),
        lr=args.qwen_reranker_lr,
        weight_decay=args.qwen_reranker_weight_decay,
    )
    (
        histories,
        features,
        winners,
        selector_scores,
        trajectory_costs,
        train_context_input_ids,
        train_context_attention_mask,
        sample_weights,
    ) = train_examples
    valid_winners = valid_examples[2]
    valid_selector_scores = valid_examples[3]
    base_valid_index, _, _ = select_candidate_indices(valid_selector_scores)
    base_valid_acc = 100.0 * float(
        base_valid_index.eq(valid_winners).float().mean().item()
    )
    best_valid_loss = float("inf")
    best_valid_cost = float("inf")
    best_state = None
    for epoch in range(1, args.qwen_reranker_epochs + 1):
        reranker.train()
        order = torch.randperm(len(histories))
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        for start in range(0, len(order), args.qwen_reranker_batch_size):
            indices = order[start:start + args.qwen_reranker_batch_size]
            history = histories[indices].to(device)
            candidate_features = features[indices].to(device)
            selector = selector_scores[indices].to(device)
            candidate_cost = trajectory_costs[indices].to(device)
            sample_weight = sample_weights[indices].to(device)
            permutation = torch.argsort(
                torch.rand(candidate_cost.shape, device=device),
                dim=-1,
            )
            candidate_features = torch.gather(
                candidate_features,
                1,
                permutation[:, :, None].expand(-1, -1, candidate_features.size(-1)),
            )
            selector = torch.gather(selector, 1, permutation)
            candidate_cost = torch.gather(candidate_cost, 1, permutation)
            winner = torch.argmin(candidate_cost, dim=-1)
            qwen_optimizer.zero_grad(set_to_none=True)
            context_kwargs = {}
            if train_context_input_ids is not None:
                batch_context_mask = train_context_attention_mask[indices].to(device).clone()
                if args.qwen_context_dropout > 0:
                    drop_context = torch.rand(len(indices), device=device) < args.qwen_context_dropout
                    batch_context_mask[drop_context] = 0
                context_kwargs = {
                    "context_input_ids": train_context_input_ids[indices].to(device),
                    "context_attention_mask": batch_context_mask,
                }
            qwen_logits = reranker(history, candidate_features, **context_kwargs)
            loss = qwen_soft_ranking_loss(
                qwen_logits,
                candidate_cost,
                sample_weight=sample_weight,
                selector_logits=selector,
                fusion_weight=reranker.fusion_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.trainable_parameters(), args.qwen_reranker_clip)
            qwen_optimizer.step()
            count = winner.numel()
            train_loss += float(loss.item()) * count
            train_total += count
            train_correct += int(torch.argmax(qwen_logits, dim=-1).eq(winner).sum().item())

        valid_loss, valid_acc, fused_acc, base_valid_cost, fused_valid_cost = evaluate_qwen_reranker(
            reranker,
            valid_examples,
            max(args.qwen_reranker_batch_size, 1),
        )
        logger.info(
            "Qwen reranker, fold %d/%d, epoch %03d, train_loss %.5f, train_acc %.1f%%, "
            "valid_loss %.5f, valid_acc %.1f%%, fused_winner_acc %.1f%%, "
            "base/fused_traj_cost %.4f/%.4f nmi.",
            fold,
            args.folds,
            epoch,
            train_loss / max(train_total, 1),
            100.0 * train_correct / max(train_total, 1),
            valid_loss,
            valid_acc,
            fused_acc,
            base_valid_cost,
            fused_valid_cost,
        )
        if (
                fused_valid_cost < best_valid_cost - 1e-9
                or (
                    abs(fused_valid_cost - best_valid_cost) <= 1e-9
                    and valid_loss < best_valid_loss
                )
        ):
            best_valid_loss = valid_loss
            best_valid_cost = fused_valid_cost
            best_state = reranker.adapter_state_dict()

    if best_state is not None:
        missing, unexpected = reranker.load_state_dict(best_state, strict=False)
        missing = [name for name in missing if not name.startswith("backbone.")]
        if missing or unexpected:
            raise RuntimeError(f"Could not restore best Qwen adapter: {missing}, {unexpected}")
    calibration = calibrate_qwen_fusion(
        reranker,
        valid_examples,
        max(args.qwen_reranker_batch_size, 1),
    )
    reranker.fusion_weight = calibration["fusion_weight"]
    reranker.selector_margin_threshold = calibration["selector_margin_threshold"]
    reranker.qwen_margin_threshold = calibration["qwen_margin_threshold"]
    (
        final_valid_loss,
        final_qwen_acc,
        final_fused_acc,
        base_valid_cost,
        fused_valid_cost,
    ) = evaluate_qwen_reranker(
        reranker,
        valid_examples,
        max(args.qwen_reranker_batch_size, 1),
    )
    accepted = (
        not args.qwen_require_validation_gain
        or (
            final_fused_acc >= base_valid_acc + args.qwen_min_validation_gain
            and calibration["cost_gain_lower_95"] >= args.qwen_min_validation_cost_gain
        )
    )
    reranker.save_adapter(
        adapter_path,
        metadata={
            "fold": fold,
            "route_classes": len(route_classes),
            "subroute_classes": len(subroute_classes),
            "candidate_count": train_examples[1].size(1),
            "candidate_pool_strategy": args.candidate_pool_strategy,
            "candidate_max_subroutes": args.candidate_max_subroutes,
            "reverse_feature_dim": 16,
            "input_length": input_length,
            "target_length": target_length,
            "uses_voyage_context": train_context_data is not None,
            "qwen_context_max_tokens": args.qwen_context_max_tokens,
            "qwen_context_dropout": args.qwen_context_dropout,
            "qwen_fused_loss_weight": args.qwen_fused_loss_weight,
            "qwen_focus_high_gain": args.qwen_focus_high_gain,
            "qwen_min_oracle_gain_nmi": args.qwen_min_oracle_gain_nmi,
            "qwen_min_winner_gap_nmi": args.qwen_min_winner_gap_nmi,
            "qwen_gain_weight": args.qwen_gain_weight,
            "best_valid_loss": final_valid_loss,
            "best_valid_fused_trajectory_cost": best_valid_cost,
            "base_valid_winner_acc": base_valid_acc,
            "qwen_valid_winner_acc": final_qwen_acc,
            "fused_valid_winner_acc": final_fused_acc,
            "base_valid_trajectory_cost": base_valid_cost,
            "fused_valid_trajectory_cost": fused_valid_cost,
            "validation_cost_gain": calibration["cost_gain"],
            "validation_cost_gain_lower_95": calibration["cost_gain_lower_95"],
            "fusion_weight": reranker.fusion_weight,
            "selector_margin_threshold": reranker.selector_margin_threshold,
            "qwen_margin_threshold": reranker.qwen_margin_threshold,
            "validation_applied_ratio": calibration["applied_ratio"],
            "accepted": accepted,
        },
    )
    logger.info(
        "Saved best Qwen reranker adapter to %s: base/fused winner_acc %.1f%%/%.1f%%, "
        "traj_cost %.4f/%.4f nmi, calibrated weight %.3f, selector/qwen margin %.4f/%.4f, "
        "applied %.1f%%, cost_gain %.4f nmi (lower95 %.4f), "
        "required winner/cost gain %.1f points/%.4f nmi, accepted=%s.",
        adapter_path,
        base_valid_acc,
        final_fused_acc,
        base_valid_cost,
        fused_valid_cost,
        reranker.fusion_weight,
        reranker.selector_margin_threshold,
        reranker.qwen_margin_threshold,
        100.0 * calibration["applied_ratio"],
        calibration["cost_gain"],
        calibration["cost_gain_lower_95"],
        args.qwen_min_validation_gain,
        args.qwen_min_validation_cost_gain,
        accepted,
    )
    reranker.eval()
    return reranker, accepted


def load_qwen_candidate_reranker(adapter_path):
    reranker, metadata = QwenCandidateReranker.from_adapter(
        adapter_path,
        model_path=args.qwen_model_path,
    )
    expected_route_classes = len(route_classes)
    expected_subroute_classes = len(subroute_classes)
    if metadata.get("route_classes", expected_route_classes) != expected_route_classes:
        raise ValueError("Qwen adapter route class count does not match the current dataset.")
    if metadata.get("subroute_classes", expected_subroute_classes) != expected_subroute_classes:
        raise ValueError("Qwen adapter subroute class count does not match the current dataset.")
    if metadata.get("input_length", input_length) != input_length:
        raise ValueError("Qwen adapter history length does not match the current configuration.")
    if metadata.get("target_length", target_length) != target_length:
        raise ValueError("Qwen adapter prediction length does not match the current configuration.")
    if metadata.get("candidate_pool_strategy", args.candidate_pool_strategy) != args.candidate_pool_strategy:
        raise ValueError("Qwen adapter candidate-pool strategy does not match the current configuration.")
    if metadata.get("candidate_max_subroutes", args.candidate_max_subroutes) != args.candidate_max_subroutes:
        raise ValueError("Qwen adapter candidate-pool size does not match the current configuration.")
    if metadata.get("reverse_feature_dim", 0) != 16:
        raise ValueError("Qwen adapter was not trained with the current reverse-verification features.")
    adapter_uses_context = bool(metadata.get("uses_voyage_context", False))
    runtime_uses_context = voyage_context_payload is not None
    if adapter_uses_context != runtime_uses_context:
        raise ValueError(
            "Qwen adapter voyage-context setting does not match the current configuration."
        )
    if adapter_uses_context and metadata.get("qwen_context_max_tokens", args.qwen_context_max_tokens) != args.qwen_context_max_tokens:
        raise ValueError("Qwen adapter context token length does not match the current configuration.")
    reranker = reranker.to(device).eval()
    accepted = bool(metadata.get("accepted", True))
    logging.getLogger().info(
        "Loaded Qwen reranker adapter from %s, validation accepted=%s.",
        adapter_path,
        accepted,
    )
    return reranker, accepted


def supervised_contrastive_loss(features, labels, temperature):
    if features is None or labels is None or features.size(0) < 2:
        return torch.zeros((), device=device)
    labels = labels.view(-1)
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    self_mask = torch.eye(features.size(0), device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, -1e9)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    valid_anchor = positive_mask.any(dim=1)
    if not valid_anchor.any():
        return torch.zeros((), device=features.device)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = (log_prob * positive_mask.float()).sum(dim=1) / positive_mask.float().sum(dim=1).clamp_min(1.0)
    return -positive_log_prob[valid_anchor].mean()


def focal_cross_entropy_loss(logits, target, class_weights=None, gamma=1.5, label_smoothing=0.0):
    target = target.long()
    ce_loss = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(1, target.view(-1, 1)).squeeze(1)
    target_probs = torch.exp(target_log_probs).clamp(min=1e-6, max=1.0)
    focal_weight = torch.pow(1.0 - target_probs, gamma)
    return focal_weight * ce_loss


def compute_objective(
        intent,
        intent_y,
        value_output,
        value_target,
        route_logits=None,
        route_target=None,
        route_decidability=None,
        route_decidability_logits=None,
        subroute_logits=None,
        subroute_target=None,
        subroute_feature=None,
        subroute_decidability=None,
        subroute_decidability_logits=None,
):
    mse_loss = criterion1(value_output, value_target)
    intent = intent.reshape(-1, intent.size(-1))
    intent_y = intent_y.reshape(-1, intent_y.size(-1))
    loss_int = criterion1(intent, intent_y)
    loss = awl(loss_int, mse_loss)

    real_output = None
    real_target = None
    if args.use_geo_loss or args.use_circular_cog:
        real_output = standardized_to_real(value_output)
        real_target = standardized_to_real(value_target)

    if args.use_geo_loss:
        dist = stable_haversine(real_output[:, :, 1:3].float(), real_target[:, :, 1:3].float())
        geo_loss = torch.mean(dist) / args.geo_loss_scale
        loss = loss + args.geo_weight * geo_loss

    if args.use_fde_loss:
        fde_loss = criterion1(value_output[:, -1, 1:3], value_target[:, -1, 1:3])
        loss = loss + args.fde_weight * fde_loss

    if args.use_smooth_loss:
        pred_delta = value_output[:, 1:, 1:3] - value_output[:, :-1, 1:3]
        target_delta = value_target[:, 1:, 1:3] - value_target[:, :-1, 1:3]
        smooth_loss = criterion1(pred_delta, target_delta)
        loss = loss + args.smooth_weight * smooth_loss

    if args.use_circular_cog:
        cog_diff = circular_angle_diff(real_output[:, :, 0], real_target[:, :, 0])
        cog_loss = torch.mean((cog_diff / args.cog_loss_scale) ** 2)
        loss = loss + args.cog_weight * cog_loss

    if args.use_route_intent_head and route_logits is not None and route_target is not None:
        route_loss_values = F.cross_entropy(
            route_logits,
            route_target.long(),
            reduction="none",
        )
        route_hard_weights = None
        if route_decidability is not None:
            route_hard_weights = (
                args.route_decidable_min_weight
                + (1.0 - args.route_decidable_min_weight)
                * route_decidability.to(device=route_logits.device, dtype=route_logits.dtype).clamp(0.0, 1.0)
            )
        route_loss = weighted_loss_mean(route_loss_values, route_hard_weights)
        loss = loss + args.route_intent_weight * route_loss
        if route_decidability is not None and args.route_undecidable_soft_weight > 0:
            uniform_target = torch.full_like(route_logits, 1.0 / route_logits.size(-1))
            route_soft_loss_values = -torch.sum(
                uniform_target * F.log_softmax(route_logits, dim=-1),
                dim=-1,
            )
            route_ambiguous_weights = 1.0 - route_decidability.to(
                device=route_logits.device,
                dtype=route_logits.dtype,
            ).clamp(0.0, 1.0)
            route_soft_loss = weighted_loss_mean(
                route_soft_loss_values,
                route_ambiguous_weights,
            )
            loss = (
                loss
                + args.route_intent_weight
                * args.route_undecidable_soft_weight
                * route_soft_loss
            )

        if (
                args.use_learned_decidability
                and route_decidability_logits is not None
                and route_decidability is not None
        ):
            route_decidability_target = route_decidability.to(
                device=route_decidability_logits.device,
                dtype=route_decidability_logits.dtype,
            ).clamp(0.0, 1.0)
            route_decidability_loss = F.binary_cross_entropy_with_logits(
                route_decidability_logits.reshape(-1),
                route_decidability_target.reshape(-1),
            )
            loss = loss + args.route_decidability_loss_weight * route_decidability_loss

    if args.use_subroute_intent_head and subroute_logits is not None and subroute_target is not None:
        class_weights = subroute_class_weights if args.use_subroute_class_weight else None
        hard_sample_weights = None
        if subroute_decidability is not None:
            hard_sample_weights = (
                args.subroute_decidable_min_weight
                + (1.0 - args.subroute_decidable_min_weight)
                * subroute_decidability.to(device=subroute_logits.device, dtype=subroute_logits.dtype).clamp(0.0, 1.0)
            )
        if args.use_subroute_focal_loss:
            subroute_loss_values = focal_cross_entropy_loss(
                subroute_logits,
                subroute_target,
                class_weights=class_weights,
                gamma=args.subroute_focal_gamma,
                label_smoothing=args.subroute_label_smoothing,
            )
        else:
            subroute_loss_values = F.cross_entropy(
                subroute_logits,
                subroute_target.long(),
                weight=class_weights,
                label_smoothing=args.subroute_label_smoothing,
                reduction="none",
            )
        subroute_loss = weighted_loss_mean(subroute_loss_values, hard_sample_weights)
        loss = loss + args.subroute_intent_weight * subroute_loss

        if (
                subroute_decidability is not None
                and route_target is not None
                and route_to_subroute_mask is not None
                and args.subroute_undecidable_soft_weight > 0
        ):
            sibling_targets = route_to_subroute_mask[route_target.long()].to(
                device=subroute_logits.device,
                dtype=subroute_logits.dtype,
            )
            sibling_targets = sibling_targets / sibling_targets.sum(dim=-1, keepdim=True).clamp_min(1.0)
            soft_loss_values = -torch.sum(
                sibling_targets * F.log_softmax(subroute_logits, dim=-1),
                dim=-1,
            )
            ambiguous_weights = 1.0 - subroute_decidability.to(
                device=subroute_logits.device,
                dtype=subroute_logits.dtype,
            ).clamp(0.0, 1.0)
            soft_loss = weighted_loss_mean(soft_loss_values, ambiguous_weights)
            loss = (
                loss
                + args.subroute_intent_weight
                * args.subroute_undecidable_soft_weight
                * soft_loss
            )

        if args.use_subroute_contrastive_loss:
            contrastive_feature = subroute_feature
            contrastive_target = subroute_target
            if subroute_decidability is not None:
                contrastive_mask = subroute_decidability >= args.subroute_decidable_contrastive_threshold
                contrastive_feature = subroute_feature[contrastive_mask]
                contrastive_target = subroute_target[contrastive_mask]
            contrastive = supervised_contrastive_loss(
                contrastive_feature,
                contrastive_target,
                args.subroute_contrastive_temperature,
            )
            loss = loss + args.subroute_contrastive_weight * contrastive

        if (
                args.use_learned_decidability
                and subroute_decidability_logits is not None
                and subroute_decidability is not None
        ):
            subroute_decidability_target = subroute_decidability.to(
                device=subroute_decidability_logits.device,
                dtype=subroute_decidability_logits.dtype,
            ).clamp(0.0, 1.0)
            subroute_decidability_loss = F.binary_cross_entropy_with_logits(
                subroute_decidability_logits.reshape(-1),
                subroute_decidability_target.reshape(-1),
            )
            loss = loss + args.subroute_decidability_loss_weight * subroute_decidability_loss

    return loss


def metric_tensors(value_output, value_target):
    real_output = standardized_to_real(value_output)
    real_target = standardized_to_real(value_target)
    dist = metric_haversine(real_output[:, :, 1:3].float(), real_target[:, :, 1:3].float())
    dist_reshape = dist.reshape(-1, target_length)
    ade = torch.mean(dist)
    fde = torch.mean(dist_reshape[:, -1])
    rmse_sog = torch.sqrt(torch.mean((real_output[:, :, 3] - real_target[:, :, 3]) ** 2))
    cog_diff = circular_angle_diff(real_output[:, :, 0], real_target[:, :, 0])
    rmse_cog = torch.sqrt(torch.mean(cog_diff ** 2))
    return ade, fde, rmse_cog, rmse_sog, real_output, real_target


def metric_to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def early_stop_monitor_value(vloss, vade, vfde):
    if args.early_stop_metric == "loss":
        return metric_to_float(vloss), "loss"
    if args.early_stop_metric == "ade":
        return metric_to_float(vade), "ADE"
    if args.early_stop_metric == "ade_fde":
        score = metric_to_float(vade) + args.early_stop_fde_weight * metric_to_float(vfde)
        return score, f"ADE+{args.early_stop_fde_weight:.3f}*FDE"
    raise ValueError(f"Unsupported early_stop_metric: {args.early_stop_metric}")


def branch_teacher_forcing_ratio(epoch):
    if not args.use_branch_teacher_forcing:
        return 0.0
    if args.branch_teacher_forcing_decay_epochs <= 1:
        return args.branch_teacher_forcing_end

    progress = min(
        max((epoch - 1) / float(args.branch_teacher_forcing_decay_epochs - 1), 0.0),
        1.0,
    )
    return (
        args.branch_teacher_forcing_start
        + progress * (args.branch_teacher_forcing_end - args.branch_teacher_forcing_start)
    )


def load_route_labels(path, expected_count):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if len(labels) != expected_count:
        raise ValueError(f"Route labels count {len(labels)} does not match data count {expected_count}.")
    return [str(item["route"]) for item in labels]


def load_label_field(path, expected_count, field):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if len(labels) != expected_count:
        raise ValueError(f"{field} labels count {len(labels)} does not match data count {expected_count}.")
    result = []
    for idx, item in enumerate(labels):
        if isinstance(item, dict):
            if field not in item:
                raise ValueError(f"Missing field {field!r} in label item {idx}: {path}")
            result.append(str(item[field]))
        else:
            result.append(str(item))
    return result


def build_label_encoder(labels):
    if labels is None:
        return None, None, None
    classes = sorted(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(classes)}
    ids = np.array([label_to_id[label] for label in labels], dtype=np.int64)
    return classes, label_to_id, ids


def route_name_from_subroute(subroute_name):
    subroute_name = str(subroute_name)
    if "_S" in subroute_name:
        return subroute_name.split("_S", 1)[0]
    return subroute_name.split("_", 1)[0]


def build_route_to_subroute_mask(route_classes, subroute_classes):
    if route_classes is None or subroute_classes is None:
        return None
    route_to_id = {route: idx for idx, route in enumerate(route_classes)}
    mask = np.zeros((len(route_classes), len(subroute_classes)), dtype=np.float32)
    for subroute_idx, subroute_name in enumerate(subroute_classes):
        route_name = route_name_from_subroute(subroute_name)
        if route_name not in route_to_id:
            raise ValueError(f"Subroute {subroute_name!r} does not match any route class: {route_classes}")
        mask[route_to_id[route_name], subroute_idx] = 1.0
    return torch.tensor(mask, dtype=torch.float32, device=device)


def inverse_frequency_values(label_ids, class_count, alpha, max_ratio, sample_weights=None):
    label_ids = np.asarray(label_ids, dtype=np.int64)
    weights = np.zeros(class_count, dtype=np.float32)
    if len(label_ids) == 0:
        return weights

    counts = np.bincount(
        label_ids,
        weights=(None if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)),
        minlength=class_count,
    ).astype(np.float32)
    present = counts > 0
    if not np.any(present):
        return weights

    max_count = float(np.max(counts[present]))
    weights[present] = (max_count / counts[present]) ** alpha
    weights[present] = np.minimum(weights[present], max_ratio)

    sample_mean = float(np.mean(weights[label_ids]))
    if sample_mean > 0:
        weights[present] = weights[present] / sample_mean
    return weights


def make_subroute_class_weights(label_ids, class_count, alpha, max_ratio, sample_weights=None):
    weights = inverse_frequency_values(
        label_ids,
        class_count,
        alpha,
        max_ratio,
        sample_weights=sample_weights,
    )
    if not np.any(weights > 0):
        return None
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_balanced_sampling_probabilities(
        label_ids,
        class_count,
        alpha,
        max_ratio,
        supervision_weights=None,
):
    weights = inverse_frequency_values(
        label_ids,
        class_count,
        alpha,
        max_ratio,
        sample_weights=supervision_weights,
    )
    if not np.any(weights > 0):
        return None
    sample_weights = weights[np.asarray(label_ids, dtype=np.int64)]
    if supervision_weights is not None:
        sample_weights = sample_weights * np.asarray(supervision_weights, dtype=np.float64)
    total = float(np.sum(sample_weights))
    if total <= 0:
        return None
    return sample_weights / total


def resample_track_positions(track, point_count):
    positions = np.asarray(track[:, [3, 4]], dtype=np.float32)
    source_progress = np.linspace(0.0, 1.0, len(positions), dtype=np.float32)
    target_progress = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    return np.stack(
        [np.interp(target_progress, source_progress, positions[:, axis]) for axis in range(2)],
        axis=-1,
    ).astype(np.float32)


def build_class_prototypes(tracks, label_ids, class_count, point_count):
    label_ids = np.asarray(label_ids, dtype=np.int64)
    prototypes = []
    for class_id in range(class_count):
        class_tracks = [
            resample_track_positions(track, point_count)
            for track, label_id in zip(tracks, label_ids)
            if int(label_id) == class_id
        ]
        if not class_tracks:
            raise ValueError(f"Cannot build prototype for empty subroute class {class_id}.")
        prototypes.append(np.mean(np.stack(class_tracks, axis=0), axis=0))
    return torch.tensor(np.stack(prototypes, axis=0), dtype=torch.float32, device=device)


def compute_class_decidability(
        windows,
        target_ids,
        prototypes,
        class_names,
        history_length,
        distance_scale,
        direction_weight,
        direction_points,
        confidence_threshold,
        margin_threshold,
        group_names=None,
        compute_batch_size=1024,
):
    """Estimate how strongly the observed history supports its final class label."""
    if windows is None or target_ids is None or prototypes is None:
        return None
    if len(windows) != len(target_ids):
        raise ValueError("Subroute decidability inputs must have matching lengths.")

    prototype_tensor = prototypes.to(device)
    class_count, prototype_points, _ = prototype_tensor.shape
    sibling_mask = torch.zeros(class_count, class_count, dtype=torch.bool, device=device)
    if group_names is None:
        sibling_mask.fill_(True)
    else:
        if len(group_names) != class_count:
            raise ValueError("Class group names must match the class count.")
        for class_id, group_name in enumerate(group_names):
            sibling_mask[class_id] = torch.tensor(
                [item == group_name for item in group_names],
                dtype=torch.bool,
                device=device,
            )

    tangent = torch.empty_like(prototype_tensor)
    tangent[:, 0, :] = prototype_tensor[:, 1, :] - prototype_tensor[:, 0, :]
    tangent[:, -1, :] = prototype_tensor[:, -1, :] - prototype_tensor[:, -2, :]
    tangent[:, 1:-1, :] = prototype_tensor[:, 2:, :] - prototype_tensor[:, :-2, :]
    tangent = F.normalize(tangent, dim=-1, eps=1e-6)
    flat_prototypes = prototype_tensor.reshape(class_count * prototype_points, 2)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    direction_start = max(int(history_length) - max(int(direction_points), 2), 0)
    result_batches = []

    with torch.no_grad():
        for start in range(0, len(windows), compute_batch_size):
            end = min(start + compute_batch_size, len(windows))
            history = windows[start:end, :history_length, 3:5].to(
                device=device,
                dtype=torch.float32,
            )
            batch_targets = torch.as_tensor(target_ids[start:end], dtype=torch.long, device=device)
            distances = torch.cdist(history, flat_prototypes).reshape(
                history.size(0),
                history.size(1),
                class_count,
                prototype_points,
            )
            path_distance = distances.min(dim=-1).values.mean(dim=1)
            nearest_index = distances[:, -1].min(dim=-1).indices
            expanded_tangent = tangent.unsqueeze(0).expand(history.size(0), -1, -1, -1)
            gather_index = nearest_index[:, :, None, None].expand(-1, -1, 1, 2)
            nearest_tangent = torch.gather(expanded_tangent, 2, gather_index).squeeze(2)
            history_direction = F.normalize(
                history[:, -1, :] - history[:, direction_start, :],
                dim=-1,
                eps=1e-6,
            )
            direction_similarity = torch.sum(
                history_direction[:, None, :] * nearest_tangent,
                dim=-1,
            )
            scores = (
                -path_distance / max(float(distance_scale), 1e-6)
                + float(direction_weight) * direction_similarity
            )

            candidate_mask = sibling_mask[batch_targets]
            probabilities = F.softmax(scores.masked_fill(~candidate_mask, -1e9), dim=-1)
            target_probability = probabilities.gather(1, batch_targets[:, None]).squeeze(1)
            target_mask = F.one_hot(batch_targets, num_classes=class_count).bool()
            competitor_probability = probabilities.masked_fill(
                ~candidate_mask | target_mask,
                -1.0,
            ).max(dim=-1).values.clamp_min(0.0)
            sibling_count = candidate_mask.sum(dim=-1)
            chance_probability = 1.0 / sibling_count.clamp_min(1).to(probabilities.dtype)
            confidence_strength = (
                (target_probability - chance_probability)
                / (float(confidence_threshold) - chance_probability).clamp_min(1e-6)
            ).clamp(0.0, 1.0)
            margin_strength = (
                (target_probability - competitor_probability)
                / max(float(margin_threshold), 1e-6)
            ).clamp(0.0, 1.0)
            strength = confidence_strength * margin_strength
            strength = strength * strength * (3.0 - 2.0 * strength)
            strength = torch.where(sibling_count <= 1, torch.ones_like(strength), strength)
            result_batches.append(strength.cpu())

    return torch.cat(result_batches).numpy().astype(np.float32)


def log_class_decidability(logger, fold, label_name, split_name, values, target_ids, class_names, threshold):
    if values is None:
        return
    values = np.asarray(values, dtype=np.float32)
    logger.info(
        "Fold %d/%d %s %s decidability: mean %.3f, >=%.2f %d/%d (%.1f%%), near-zero %.1f%%.",
        fold,
        args.folds,
        split_name,
        label_name,
        float(np.mean(values)),
        threshold,
        int(np.sum(values >= threshold)),
        len(values),
        100.0 * float(np.mean(values >= threshold)),
        100.0 * float(np.mean(values <= 0.05)),
    )
    detail = []
    target_ids = np.asarray(target_ids, dtype=np.int64)
    for class_id, class_name in enumerate(class_names):
        class_values = values[target_ids == class_id]
        if len(class_values):
            detail.append(
                f"{class_name}:mean={float(np.mean(class_values)):.2f},"
                f"decidable={100.0 * float(np.mean(class_values >= threshold)):.1f}%"
            )
    logger.info(
        "Fold %d/%d %s decidability by %s: %s.",
        fold,
        args.folds,
        split_name,
        label_name,
        ", ".join(detail),
    )


def format_class_values(values, class_names, precision=3):
    if values is None:
        return "none"
    result = []
    for idx, value in enumerate(values):
        result.append(f"{label_id_to_name(idx, class_names)}:{float(value):.{precision}f}")
    return ", ".join(result)


def expand_track_labels_to_windows(track_labels, window_slices):
    if track_labels is None:
        return None
    window_labels = []
    for label, windows in zip(track_labels, window_slices):
        window_labels.extend([label] * len(windows))
    return np.array(window_labels)


def unpack_model_output(model_output):
    if len(model_output) == 2:
        return model_output[0], model_output[1], None, None, None, None, None, None
    if len(model_output) == 4:
        return model_output[0], model_output[1], None, model_output[2], None, model_output[3], None, None
    if len(model_output) == 6:
        return (*model_output, None, None)
    return tuple(model_output[:8])


def choose_plot_indices(window_labels, max_samples, strategy, seed):
    if max_samples <= 0:
        return []
    if window_labels is None or strategy == "first":
        return list(range(max_samples))

    rng = np.random.default_rng(seed)
    labels = np.asarray(window_labels)
    route_names = sorted(set(labels.tolist()))
    per_route = max(1, int(np.ceil(max_samples / max(len(route_names), 1))))
    selected = []
    for route in route_names:
        route_indices = np.flatnonzero(labels == route)
        if len(route_indices) == 0:
            continue
        take = min(per_route, len(route_indices))
        chosen = rng.choice(route_indices, size=take, replace=False)
        selected.extend(int(item) for item in chosen)
    if len(selected) > max_samples:
        selected = selected[:max_samples]
    return selected


def label_id_to_name(label_id, class_names):
    if label_id is None:
        return None
    label_id = int(label_id)
    if class_names is None or label_id < 0 or label_id >= len(class_names):
        return str(label_id)
    return str(class_names[label_id])


def sanitize_filename_part(value):
    if value is None:
        return ""
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value))


def format_top_probs(probs, class_names, top_k=3):
    if probs is None or probs.numel() == 0:
        return None
    top_k = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k=top_k)
    parts = []
    for prob, idx in zip(values.tolist(), indices.tolist()):
        parts.append(f"{label_id_to_name(idx, class_names)}:{prob:.3f}")
    return ";".join(parts)


def update_routing_stats(
        logits,
        targets,
        stats,
        temperature,
        decidability_logits=None,
        decidability_gate_threshold=0.5,
):
    if logits is None or targets is None or not args.confidence_aware_routing:
        return
    probs = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)
    top_k = min(max(args.routing_top_k, 1), probs.size(-1))
    top_values, top_indices = torch.topk(probs, k=top_k, dim=-1)
    top1 = top_indices[:, 0]
    if probs.size(-1) > 1:
        margin = top_values[:, 0] - top_values[:, 1]
    else:
        margin = torch.ones_like(top_values[:, 0])
    confident = (
        (top_values[:, 0] >= args.routing_confidence_threshold)
        & (margin >= args.routing_margin_threshold)
    )
    if args.use_learned_decidability and decidability_logits is not None:
        confident = confident & (
            torch.sigmoid(decidability_logits.reshape(-1))
            >= float(decidability_gate_threshold)
        )
    top1_correct = top1.eq(targets)
    topk_hit = top_indices.eq(targets.unsqueeze(1)).any(dim=1)

    stats["hard_total"] += int(confident.sum().item())
    stats["hard_correct"] += int((top1_correct & confident).sum().item())
    uncertain = ~confident
    stats["topk_total"] += int(uncertain.sum().item())
    stats["topk_top1_correct"] += int((top1_correct & uncertain).sum().item())
    stats["topk_hit"] += int((topk_hit & uncertain).sum().item())


def create_calibration_stats(bin_count=10):
    return {
        "count": 0,
        "nll_sum": 0.0,
        "brier_sum": 0.0,
        "bin_count": [0] * bin_count,
        "bin_confidence_sum": [0.0] * bin_count,
        "bin_correct_sum": [0.0] * bin_count,
    }


def update_calibration_stats(logits, targets, stats, temperature):
    if logits is None or targets is None:
        return
    probs = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)
    confidence, prediction = probs.max(dim=-1)
    correct = prediction.eq(targets).to(dtype=probs.dtype)
    one_hot_target = F.one_hot(targets.long(), num_classes=probs.size(-1)).to(dtype=probs.dtype)

    stats["count"] += int(targets.numel())
    stats["nll_sum"] += float(
        F.nll_loss(torch.log(probs.clamp_min(1e-8)), targets.long(), reduction="sum").item()
    )
    stats["brier_sum"] += float(torch.sum((probs - one_hot_target) ** 2).item())

    bin_count = len(stats["bin_count"])
    bin_ids = torch.clamp((confidence * bin_count).long(), max=bin_count - 1)
    for bin_id in range(bin_count):
        mask = bin_ids.eq(bin_id)
        count = int(mask.sum().item())
        if count == 0:
            continue
        stats["bin_count"][bin_id] += count
        stats["bin_confidence_sum"][bin_id] += float(confidence[mask].sum().item())
        stats["bin_correct_sum"][bin_id] += float(correct[mask].sum().item())


def format_calibration_stats(stats):
    total = max(int(stats["count"]), 1)
    ece = 0.0
    for count, confidence_sum, correct_sum in zip(
            stats["bin_count"],
            stats["bin_confidence_sum"],
            stats["bin_correct_sum"],
    ):
        if count == 0:
            continue
        average_confidence = confidence_sum / count
        average_accuracy = correct_sum / count
        ece += count / total * abs(average_confidence - average_accuracy)
    return (
        f"ECE {ece:.4f}, Brier {stats['brier_sum'] / total:.4f}, "
        f"NLL {stats['nll_sum'] / total:.4f}"
    )


def update_decidability_stats(
        logits,
        geometry_target,
        stats,
        gate_threshold,
        geometry_threshold,
):
    if logits is None or geometry_target is None:
        return
    probability = torch.sigmoid(logits.reshape(-1))
    geometry_target = geometry_target.reshape(-1).to(
        device=probability.device,
        dtype=probability.dtype,
    ).clamp(0.0, 1.0)
    predicted_decidable = probability >= float(gate_threshold)
    geometry_decidable = geometry_target >= float(geometry_threshold)

    stats["total"] += int(probability.numel())
    stats["probability_sum"] += float(probability.sum().item())
    stats["absolute_error_sum"] += float(torch.abs(probability - geometry_target).sum().item())
    stats["predicted_decidable"] += int(predicted_decidable.sum().item())
    stats["geometry_decidable"] += int(geometry_decidable.sum().item())
    stats["true_positive"] += int((predicted_decidable & geometry_decidable).sum().item())


def format_decidability_stats(stats):
    total = max(int(stats["total"]), 1)
    predicted = max(int(stats["predicted_decidable"]), 1)
    geometry = max(int(stats["geometry_decidable"]), 1)
    return (
        f"mean_p {stats['probability_sum'] / total:.3f}, "
        f"MAE {stats['absolute_error_sum'] / total:.3f}, "
        f"gate {stats['predicted_decidable']}/{total} "
        f"({100.0 * stats['predicted_decidable'] / total:.1f}%), "
        f"precision {100.0 * stats['true_positive'] / predicted:.1f}%, "
        f"recall {100.0 * stats['true_positive'] / geometry:.1f}%"
    )


def format_routing_stats(stats, top_k):
    hard_total = stats["hard_total"]
    topk_total = stats["topk_total"]
    total = hard_total + topk_total
    hard_acc = 100.0 * stats["hard_correct"] / max(hard_total, 1)
    topk_top1_acc = 100.0 * stats["topk_top1_correct"] / max(topk_total, 1)
    topk_recall = 100.0 * stats["topk_hit"] / max(topk_total, 1)
    return (
        f"hard {hard_total}/{total} ({100.0 * hard_total / max(total, 1):.1f}%), "
        f"hard_top1_acc {hard_acc:.1f}%, uncertain {topk_total}/{total}, "
        f"uncertain_top1_acc {topk_top1_acc:.1f}%, uncertain_top{top_k}_recall {topk_recall:.1f}%"
    )


def load_model_checkpoint(checkpoint_path, current_model):
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=device)
    if isinstance(loaded, nn.Module):
        return loaded.to(device)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        loaded = loaded["state_dict"]
    current_model.load_state_dict(loaded)
    return current_model.to(device)


def save_prediction_plots(
        X_data,
        fold,
        output_dir,
        max_samples,
        window_labels=None,
        plot_strategy="first",
        window_route_ids=None,
        route_classes=None,
        window_subroute_ids=None,
        subroute_classes=None,
        voyage_contexts=None,
):
    if max_samples <= 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    sample_count = min(max_samples, len(X_data))
    sampling_labels = window_labels
    if plot_strategy == "subroute_balanced" and window_subroute_ids is not None:
        sampling_labels = np.array([label_id_to_name(item, subroute_classes) for item in window_subroute_ids])
    plot_indices = choose_plot_indices(sampling_labels, sample_count, plot_strategy, seed=fold * 1009)
    diagnostics = []

    for plot_idx, sample_idx in enumerate(plot_indices):
        sample_data = X_data[sample_idx]
        route_label = None if window_labels is None else str(window_labels[sample_idx])
        true_route_id = None
        true_route = route_label
        if window_route_ids is not None:
            true_route_id = int(window_route_ids[sample_idx])
            true_route = label_id_to_name(true_route_id, route_classes)
        true_subroute_id = None
        true_subroute = None
        if window_subroute_ids is not None:
            true_subroute_id = int(window_subroute_ids[sample_idx])
            true_subroute = label_id_to_name(true_subroute_id, subroute_classes)

        delta = sample_data[:input_length, in_cols].unsqueeze(0).to(device)
        src = sample_data[:input_length, in_cols].unsqueeze(0).to(device)
        value_target = sample_data[input_length:input_length + target_length, src_cols].unsqueeze(0).to(device)
        semantic_feature = semantic_features_for_contexts(
            None if voyage_contexts is None else [voyage_contexts[sample_idx]]
        )

        with torch.no_grad():
            _, raw_output, route_logits, subroute_logits, _, subroute_feature, _, _ = unpack_model_output(
                model(delta, src, semantic_feature=semantic_feature)
            )
            output = compose_value_output(raw_output, src)
            selected_candidate_index = None
            selected_candidate_route_id = None
            selected_candidate_subroute_id = None
            candidate_summary = None
            if args.use_candidate_selector and candidate_selector_runtime_active:
                candidate_result = prepare_candidate_prediction(
                    delta,
                    src,
                    output,
                    route_logits,
                    subroute_logits,
                    subroute_feature,
                    voyage_contexts=(
                        None if voyage_contexts is None else [voyage_contexts[sample_idx]]
                    ),
                )
                output = candidate_result["selected_output"]
                selected_candidate_index = int(candidate_result["selected_index"].item())
                selected_candidate_route_id = int(
                    candidate_result["route_ids"][0, selected_candidate_index].item()
                )
                selected_candidate_subroute_id = int(
                    candidate_result["subroute_ids"][0, selected_candidate_index].item()
                )
                selector_probs = F.softmax(candidate_result["selector_logits"], dim=-1).squeeze(0)
                candidate_parts = []
                for candidate_idx in range(candidate_result["route_ids"].size(1)):
                    candidate_kind = (
                        "base" if float(candidate_result["is_base"][0, candidate_idx]) > 0.5
                        else f"branch{candidate_idx}"
                    )
                    route_name = label_id_to_name(
                        candidate_result["route_ids"][0, candidate_idx],
                        route_classes,
                    )
                    subroute_name = label_id_to_name(
                        candidate_result["subroute_ids"][0, candidate_idx],
                        subroute_classes,
                    )
                    candidate_parts.append(
                        f"{candidate_kind}-{route_name}/{subroute_name}:{float(selector_probs[candidate_idx]):.3f}"
                    )
                candidate_summary = ";".join(candidate_parts)
            sample_ade, sample_fde, sample_rmse_cog, sample_rmse_sog, real_output, real_target = metric_tensors(
                output,
                value_target,
            )

        pred_route_id = None
        pred_route = None
        pred_route_conf = None
        top_route_probs = None
        if route_logits is not None:
            route_probs = F.softmax(
                route_logits / args.route_routing_temperature,
                dim=-1,
            ).squeeze(0).detach().cpu()
            pred_route_id = int(torch.argmax(route_probs).item())
            pred_route = label_id_to_name(pred_route_id, route_classes)
            pred_route_conf = float(route_probs[pred_route_id].item())
            top_route_probs = format_top_probs(route_probs, route_classes)

        pred_subroute_id = None
        pred_subroute = None
        pred_subroute_conf = None
        top_subroute_probs = None
        if subroute_logits is not None:
            subroute_probs = F.softmax(
                subroute_logits / args.subroute_routing_temperature,
                dim=-1,
            ).squeeze(0).detach().cpu()
            pred_subroute_id = int(torch.argmax(subroute_probs).item())
            pred_subroute = label_id_to_name(pred_subroute_id, subroute_classes)
            pred_subroute_conf = float(subroute_probs[pred_subroute_id].item())
            top_subroute_probs = format_top_probs(subroute_probs, subroute_classes)

        pred = real_output.squeeze(0).detach().cpu().numpy()
        history = sample_data[:input_length, src_cols].detach().cpu().numpy()
        target = real_target.squeeze(0).detach().cpu().numpy()

        history = inverse_standardized(history, transform_matrix, mean_values)

        history_xy = history[:, [1, 2]]
        pred_xy = np.vstack([history_xy[-1:], pred[:, [1, 2]]])
        target_xy = np.vstack([history_xy[-1:], target[:, [1, 2]]])

        fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
        ax.plot(history_xy[:, 0], history_xy[:, 1], "-o", color="#2563eb", linewidth=2.0, markersize=3.5,
                label="History")
        ax.plot(target_xy[:, 0], target_xy[:, 1], "-o", color="#16a34a", linewidth=2.0, markersize=3.5,
                label="Ground truth")
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], "--o", color="#dc2626", linewidth=2.0, markersize=3.5,
                label="Prediction")

        ax.scatter(history_xy[0, 0], history_xy[0, 1], color="#1e40af", s=42, marker="s", label="Start")
        ax.scatter(history_xy[-1, 0], history_xy[-1, 1], color="#111827", s=46, marker="x", label="Predict from")
        route_text = true_route or "-"
        if pred_route is not None:
            route_text = f"true {route_text} | pred {pred_route} p={pred_route_conf:.2f}"
        subroute_text = ""
        if true_subroute is not None or pred_subroute is not None:
            confidence_text = "" if pred_subroute_conf is None else f" p={pred_subroute_conf:.2f}"
            subroute_text = f" | true {true_subroute or '-'} | pred {pred_subroute or '-'}{confidence_text}"
        if selected_candidate_route_id is not None:
            selected_route = label_id_to_name(selected_candidate_route_id, route_classes)
            selected_subroute = label_id_to_name(selected_candidate_subroute_id, subroute_classes)
            subroute_text += f" | selected {selected_route}/{selected_subroute}"
        sample_ade_value = to_float(sample_ade)
        sample_fde_value = to_float(sample_fde)
        sample_rmse_cog_value = to_float(sample_rmse_cog)
        sample_rmse_sog_value = to_float(sample_rmse_sog)
        metric_text = (
            f"ADE {sample_ade_value:.3f}nmi/{sample_ade_value * 1852.0:.0f}m, "
            f"FDE {sample_fde_value:.3f}nmi/{sample_fde_value * 1852.0:.0f}m"
        )
        ax.set_title(f"Fold {fold} Sample {sample_idx} [{route_text}{subroute_text}]\n{metric_text}", fontsize=9.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()

        route_part = "" if route_label is None else f"_{sanitize_filename_part(route_label)}"
        subroute_part = "" if true_subroute is None else f"_{sanitize_filename_part(true_subroute)}"
        pred_part = "" if pred_subroute is None else f"_pred-{sanitize_filename_part(pred_subroute)}"
        save_path = output_dir / f"fold_{fold:02d}_sample_{plot_idx:03d}_idx{sample_idx:05d}{route_part}.png"
        if subroute_part or pred_part:
            save_path = output_dir / (
                f"fold_{fold:02d}_sample_{plot_idx:03d}_idx{sample_idx:05d}"
                f"{route_part}{subroute_part}{pred_part}.png"
            )
        fig.savefig(save_path)
        plt.close(fig)

        diagnostics.append({
            "fold": fold,
            "plot_rank": plot_idx,
            "sample_index": int(sample_idx),
            "route": true_route,
            "true_route_id": true_route_id,
            "pred_route_id": pred_route_id,
            "pred_route": pred_route,
            "pred_route_conf": pred_route_conf,
            "route_match": (
                None if true_route_id is None or pred_route_id is None
                else true_route_id == pred_route_id
            ),
            "top_route_probs": top_route_probs,
            "true_subroute_id": true_subroute_id,
            "true_subroute": true_subroute,
            "pred_subroute_id": pred_subroute_id,
            "pred_subroute": pred_subroute,
            "pred_subroute_conf": pred_subroute_conf,
            "subroute_match": (
                None if true_subroute_id is None or pred_subroute_id is None
                else true_subroute_id == pred_subroute_id
            ),
            "top_subroute_probs": top_subroute_probs,
            "selected_candidate_index": selected_candidate_index,
            "selected_candidate_route": label_id_to_name(selected_candidate_route_id, route_classes),
            "selected_candidate_subroute": label_id_to_name(selected_candidate_subroute_id, subroute_classes),
            "candidate_selector_probs": candidate_summary,
            "ade_nmi": sample_ade_value,
            "ade_m": sample_ade_value * 1852.0,
            "fde_nmi": sample_fde_value,
            "fde_m": sample_fde_value * 1852.0,
            "rmse_cog_deg": sample_rmse_cog_value,
            "rmse_sog_kn": sample_rmse_sog_value,
            "history_end_lon": float(history_xy[-1, 0]),
            "history_end_lat": float(history_xy[-1, 1]),
            "pred_end_lon": float(pred[-1, 1]),
            "pred_end_lat": float(pred[-1, 2]),
            "true_end_lon": float(target[-1, 1]),
            "true_end_lat": float(target[-1, 2]),
            "plot_path": str(save_path),
        })

    if diagnostics:
        diagnostics_path = output_dir / f"fold_{fold:02d}_prediction_diagnostics.csv"
        pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
        logging.getLogger().info("Saved prediction diagnostics for fold %d to %s.", fold, diagnostics_path)


def evaluate(
        X_data,
        route_targets=None,
        route_decidability=None,
        subroute_targets=None,
        subroute_decidability=None,
        name='Eval',
        route_class_names=None,
        subroute_class_names=None,
        voyage_contexts=None,
):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    sample = 0
    ADE_list = []
    FDE_list = []
    rmse_cog_list = []
    rmse_sog_list = []
    route_correct = 0
    route_total = 0
    route_class_correct = Counter()
    route_class_total = Counter()
    route_routing_stats = Counter()
    route_calibration_stats = create_calibration_stats()
    route_decidability_stats = Counter()
    decidable_route_correct = 0
    decidable_route_total = 0
    ambiguous_route_topk_hit = 0
    ambiguous_route_total = 0
    subroute_correct = 0
    subroute_total = 0
    subroute_class_correct = Counter()
    subroute_class_total = Counter()
    subroute_ade_sum = Counter()
    subroute_fde_sum = Counter()
    subroute_metric_total = Counter()
    subroute_routing_stats = Counter()
    subroute_calibration_stats = create_calibration_stats()
    subroute_decidability_stats = Counter()
    decidable_subroute_correct = 0
    decidable_subroute_total = 0
    ambiguous_subroute_topk_hit = 0
    ambiguous_subroute_total = 0
    candidate_total = 0
    candidate_branch_slots = 0
    candidate_selector_correct = 0
    candidate_route_hit = 0
    candidate_subroute_hit = 0
    candidate_oracle_ade_sum = 0.0
    candidate_oracle_fde_sum = 0.0
    candidate_branch_switch_count = 0
    candidate_branch_switch_correct = 0
    qwen_applied_count = 0
    qwen_winner_correct = 0
    with torch.no_grad():
        for idx in range(0, len(eval_idx_list), batch_size):
            batch_indices = eval_idx_list[idx:idx + batch_size]
            delta = torch.stack([X_data[i][:input_length, in_cols] for i in batch_indices]).cuda()
            src = torch.stack([X_data[i][:input_length, in_cols] for i in batch_indices]).cuda()
            tgt_y = torch.stack(
                [X_data[i][input_length:input_length + target_length, src_cols] for i in batch_indices]).cuda()
            intent_y = torch.stack(
                [X_data[i][input_length:input_length + target_length, intent_cols] for i in batch_indices]).cuda()

            route_target = None
            if route_targets is not None:
                route_target = route_targets[batch_indices].to(device)
            batch_route_decidability = None
            if route_decidability is not None:
                batch_route_decidability = route_decidability[batch_indices].to(device)
            subroute_target = None
            if subroute_targets is not None:
                subroute_target = subroute_targets[batch_indices].to(device)
            batch_decidability = None
            if subroute_decidability is not None:
                batch_decidability = subroute_decidability[batch_indices].to(device)
            semantic_feature = semantic_features_for_contexts(
                None if voyage_contexts is None else voyage_contexts[batch_indices]
            )

            (
                intent,
                raw_output,
                route_logits,
                subroute_logits,
                route_feature,
                subroute_feature,
                route_decidability_logits,
                subroute_decidability_logits,
            ) = unpack_model_output(
                model(delta, src, semantic_feature=semantic_feature)
            )
            value_output = compose_value_output(raw_output, src)
            value_target = tgt_y

            selector_loss = None
            best_candidate_cost = None
            if args.use_candidate_selector and candidate_selector_runtime_active:
                candidate_result = prepare_candidate_prediction(
                    delta,
                    src,
                    value_output,
                    route_logits,
                    subroute_logits,
                    subroute_feature,
                    voyage_contexts=(
                        None if voyage_contexts is None else voyage_contexts[batch_indices]
                    ),
                )
                candidate_branch_slots = int(candidate_result["branch_subroute_ids"].size(1))
                value_output = candidate_result["selected_output"]
                candidate_cost, candidate_ade, candidate_fde = candidate_trajectory_costs(
                    candidate_result["outputs"],
                    value_target,
                )
                winner_target = torch.argmin(candidate_cost, dim=-1)
                selector_loss = candidate_soft_cost_loss(
                    candidate_result["selector_logits"],
                    candidate_cost,
                )
                best_candidate_cost = candidate_cost.gather(
                    1,
                    winner_target.unsqueeze(1),
                ).mean()
                candidate_total += int(winner_target.numel())
                candidate_selector_correct += int(
                    candidate_result["selected_index"].eq(winner_target).sum().item()
                )
                candidate_branch_switch_count += int(
                    candidate_result["switch_to_branch"].sum().item()
                )
                candidate_branch_switch_correct += int(
                    (
                        candidate_result["switch_to_branch"]
                        & candidate_result["selected_index"].eq(winner_target)
                    ).sum().item()
                )
                if candidate_result["qwen_logits"] is not None:
                    qwen_mask = candidate_result["qwen_applied"]
                    qwen_applied_count += int(qwen_mask.sum().item())
                    qwen_winner_correct += int(
                        (
                            torch.argmax(candidate_result["qwen_logits"], dim=-1).eq(winner_target)
                            & qwen_mask
                        ).sum().item()
                    )
                if route_target is not None:
                    candidate_route_hit += int(
                        candidate_result["branch_route_ids"].eq(route_target.unsqueeze(1)).any(dim=1).sum().item()
                    )
                if subroute_target is not None:
                    candidate_subroute_hit += int(
                        candidate_result["branch_subroute_ids"].eq(subroute_target.unsqueeze(1)).any(dim=1).sum().item()
                    )
                candidate_oracle_ade_sum += float(
                    candidate_ade.gather(1, winner_target.unsqueeze(1)).sum().item()
                )
                candidate_oracle_fde_sum += float(
                    candidate_fde.gather(1, winner_target.unsqueeze(1)).sum().item()
                )

            loss = compute_objective(
                intent,
                intent_y,
                value_output,
                value_target,
                route_logits=route_logits,
                route_target=route_target,
                route_decidability=batch_route_decidability,
                route_decidability_logits=route_decidability_logits,
                subroute_logits=subroute_logits,
                subroute_target=subroute_target,
                subroute_feature=subroute_feature,
                subroute_decidability=batch_decidability,
                subroute_decidability_logits=subroute_decidability_logits,
            )
            if selector_loss is not None:
                loss = (
                    loss
                    + args.candidate_selector_weight * selector_loss
                    + args.candidate_trajectory_weight * best_candidate_cost
                )
            if route_logits is not None and route_target is not None:
                route_pred = torch.argmax(route_logits, dim=-1)
                update_routing_stats(
                    route_logits,
                    route_target,
                    route_routing_stats,
                    args.route_routing_temperature,
                    decidability_logits=route_decidability_logits,
                    decidability_gate_threshold=args.route_decidability_gate_threshold,
                )
                update_calibration_stats(
                    route_logits,
                    route_target,
                    route_calibration_stats,
                    args.route_routing_temperature,
                )
                update_decidability_stats(
                    route_decidability_logits,
                    batch_route_decidability,
                    route_decidability_stats,
                    args.route_decidability_gate_threshold,
                    args.route_decidable_threshold,
                )
                route_correct += int((route_pred == route_target).sum().item())
                route_total += int(route_target.numel())
                for pred_item, target_item in zip(route_pred.detach().cpu().tolist(),
                                                  route_target.detach().cpu().tolist()):
                    class_name = label_id_to_name(target_item, route_class_names)
                    route_class_total[class_name] += 1
                    if int(pred_item) == int(target_item):
                        route_class_correct[class_name] += 1
                if batch_route_decidability is not None:
                    route_decidable_mask = batch_route_decidability >= args.route_decidable_threshold
                    route_ambiguous_mask = ~route_decidable_mask
                    decidable_route_total += int(route_decidable_mask.sum().item())
                    decidable_route_correct += int(
                        (route_pred.eq(route_target) & route_decidable_mask).sum().item()
                    )
                    route_top_k = min(max(args.routing_top_k, 1), route_logits.size(-1))
                    route_top_k_ids = torch.topk(route_logits, k=route_top_k, dim=-1).indices
                    route_top_k_hit = route_top_k_ids.eq(route_target.unsqueeze(1)).any(dim=1)
                    ambiguous_route_total += int(route_ambiguous_mask.sum().item())
                    ambiguous_route_topk_hit += int(
                        (route_top_k_hit & route_ambiguous_mask).sum().item()
                    )
            if subroute_logits is not None and subroute_target is not None:
                subroute_pred = torch.argmax(subroute_logits, dim=-1)
                update_routing_stats(
                    subroute_logits,
                    subroute_target,
                    subroute_routing_stats,
                    args.subroute_routing_temperature,
                    decidability_logits=subroute_decidability_logits,
                    decidability_gate_threshold=args.subroute_decidability_gate_threshold,
                )
                update_calibration_stats(
                    subroute_logits,
                    subroute_target,
                    subroute_calibration_stats,
                    args.subroute_routing_temperature,
                )
                update_decidability_stats(
                    subroute_decidability_logits,
                    batch_decidability,
                    subroute_decidability_stats,
                    args.subroute_decidability_gate_threshold,
                    args.subroute_decidable_threshold,
                )
                subroute_correct += int((subroute_pred == subroute_target).sum().item())
                subroute_total += int(subroute_target.numel())
                for pred_item, target_item in zip(subroute_pred.detach().cpu().tolist(),
                                                  subroute_target.detach().cpu().tolist()):
                    class_name = label_id_to_name(target_item, subroute_class_names)
                    subroute_class_total[class_name] += 1
                    if int(pred_item) == int(target_item):
                        subroute_class_correct[class_name] += 1
                if batch_decidability is not None:
                    decidable_mask = batch_decidability >= args.subroute_decidable_threshold
                    ambiguous_mask = ~decidable_mask
                    decidable_subroute_total += int(decidable_mask.sum().item())
                    decidable_subroute_correct += int(
                        (subroute_pred.eq(subroute_target) & decidable_mask).sum().item()
                    )
                    top_k = min(max(args.routing_top_k, 1), subroute_logits.size(-1))
                    top_k_ids = torch.topk(subroute_logits, k=top_k, dim=-1).indices
                    top_k_hit = top_k_ids.eq(subroute_target.unsqueeze(1)).any(dim=1)
                    ambiguous_subroute_total += int(ambiguous_mask.sum().item())
                    ambiguous_subroute_topk_hit += int((top_k_hit & ambiguous_mask).sum().item())
            ADE, FDE, rmse_cog, rmse_sog, real_output, real_target = metric_tensors(value_output, value_target)
            if subroute_target is not None:
                sample_dist = metric_haversine(
                    real_output[:, :, 1:3].float(),
                    real_target[:, :, 1:3].float(),
                )
                sample_dist = sample_dist.reshape(-1, target_length)
                sample_ade = sample_dist.mean(dim=1).detach().cpu().tolist()
                sample_fde = sample_dist[:, -1].detach().cpu().tolist()
                for target_item, ade_item, fde_item in zip(
                        subroute_target.detach().cpu().tolist(),
                        sample_ade,
                        sample_fde,
                ):
                    class_name = label_id_to_name(target_item, subroute_class_names)
                    subroute_ade_sum[class_name] += float(ade_item)
                    subroute_fde_sum[class_name] += float(fde_item)
                    subroute_metric_total[class_name] += 1

            ADE_list.append(ADE.detach().cpu())
            FDE_list.append(FDE.detach().cpu())
            rmse_cog_list.append(rmse_cog.detach().cpu())
            rmse_sog_list.append(rmse_sog.detach().cpu())

            pred_list.append(real_output.detach().cpu())
            Y_list.append(real_target.detach().cpu())

            total_loss += loss.item()
            sample += 1

        eval_loss = total_loss / sample
        print(name + " loss: {:.5f}".format(eval_loss))
        ADE = torch.stack(ADE_list).mean()
        print(" ADE: {:.5f}nmi-->{:.5f}m".format(ADE, ADE * 1852))
        FDE = torch.stack(FDE_list).mean()
        print(" FDE: {:.5f}nmi-->{:.5f}m".format(FDE, FDE * 1852))
        rmse_cog = torch.stack(rmse_cog_list).mean()
        print(" RMSE_COG: {:.5f}°".format(rmse_cog))
        rmse_sog = torch.stack(rmse_sog_list).mean()
        print(" RMSE_SOG: {:.5f}kn".format(rmse_sog))
        if args.use_candidate_selector and candidate_selector_runtime_active and candidate_total > 0:
            candidate_detail = (
                f"selector_winner_acc {100.0 * candidate_selector_correct / candidate_total:.1f}%, "
                f"route_recall@{candidate_branch_slots} {100.0 * candidate_route_hit / candidate_total:.1f}%, "
                f"subroute_recall@{candidate_branch_slots} "
                f"{100.0 * candidate_subroute_hit / candidate_total:.1f}%, "
                f"branch_switch {candidate_branch_switch_count}/{candidate_total} "
                f"({100.0 * candidate_branch_switch_count / candidate_total:.1f}%), "
                f"switch_winner_acc {100.0 * candidate_branch_switch_correct / max(candidate_branch_switch_count, 1):.1f}%, "
                f"oracle_ADE@{candidate_branch_slots + 1} "
                f"{candidate_oracle_ade_sum / candidate_total:.3f}nmi, "
                f"oracle_FDE@{candidate_branch_slots + 1} "
                f"{candidate_oracle_fde_sum / candidate_total:.3f}nmi, "
                f"qwen_applied {qwen_applied_count}/{candidate_total} "
                f"({100.0 * qwen_applied_count / candidate_total:.1f}%), "
                f"qwen_winner_acc {100.0 * qwen_winner_correct / max(qwen_applied_count, 1):.1f}%"
            )
            print(" Candidate_Selector: " + candidate_detail)
            logging.getLogger().info("%s Candidate_Selector: %s", name, candidate_detail)
        if route_total > 0:
            print(" Route_ACC: {:.2f}%".format(100.0 * route_correct / route_total))
            if route_decidability is not None:
                route_staged_detail = (
                    f"decidable_top1 {100.0 * decidable_route_correct / max(decidable_route_total, 1):.1f}% "
                    f"({decidable_route_correct}/{decidable_route_total}), "
                    f"ambiguous_top{args.routing_top_k}_recall "
                    f"{100.0 * ambiguous_route_topk_hit / max(ambiguous_route_total, 1):.1f}% "
                    f"({ambiguous_route_topk_hit}/{ambiguous_route_total})"
                )
                print(" Route_Staged: " + route_staged_detail)
                logging.getLogger().info("%s Route_Staged: %s", name, route_staged_detail)
            if route_class_total:
                detail = ", ".join(
                    "{}:{:.1f}%({}/{})".format(
                        class_name,
                        100.0 * route_class_correct[class_name] / route_class_total[class_name],
                        route_class_correct[class_name],
                        route_class_total[class_name],
                    )
                    for class_name in sorted(route_class_total)
                )
                print(" Route_ACC_by_class: " + detail)
                logging.getLogger().info("%s Route_ACC_by_class: %s", name, detail)
            if args.confidence_aware_routing:
                routing_detail = format_routing_stats(route_routing_stats, args.routing_top_k)
                print(" Route_Routing: " + routing_detail)
                logging.getLogger().info("%s Route_Routing: %s", name, routing_detail)
            calibration_detail = format_calibration_stats(route_calibration_stats)
            print(" Route_Calibration: " + calibration_detail)
            logging.getLogger().info("%s Route_Calibration: %s", name, calibration_detail)
            if route_decidability_stats["total"] > 0:
                decidability_detail = format_decidability_stats(route_decidability_stats)
                print(" Route_Decidability: " + decidability_detail)
                logging.getLogger().info("%s Route_Decidability: %s", name, decidability_detail)
        if subroute_total > 0:
            print(" Subroute_ACC: {:.2f}%".format(100.0 * subroute_correct / subroute_total))
            if subroute_decidability is not None:
                staged_detail = (
                    f"decidable_top1 {100.0 * decidable_subroute_correct / max(decidable_subroute_total, 1):.1f}% "
                    f"({decidable_subroute_correct}/{decidable_subroute_total}), "
                    f"ambiguous_top{args.routing_top_k}_recall "
                    f"{100.0 * ambiguous_subroute_topk_hit / max(ambiguous_subroute_total, 1):.1f}% "
                    f"({ambiguous_subroute_topk_hit}/{ambiguous_subroute_total})"
                )
                print(" Subroute_Staged: " + staged_detail)
                logging.getLogger().info("%s Subroute_Staged: %s", name, staged_detail)
            if subroute_class_total:
                detail = ", ".join(
                    "{}:{:.1f}%({}/{})".format(
                        class_name,
                        100.0 * subroute_class_correct[class_name] / subroute_class_total[class_name],
                        subroute_class_correct[class_name],
                        subroute_class_total[class_name],
                    )
                    for class_name in sorted(subroute_class_total)
                )
                print(" Subroute_ACC_by_class: " + detail)
                logging.getLogger().info("%s Subroute_ACC_by_class: %s", name, detail)
            if args.confidence_aware_routing:
                routing_detail = format_routing_stats(subroute_routing_stats, args.routing_top_k)
                print(" Subroute_Routing: " + routing_detail)
                logging.getLogger().info("%s Subroute_Routing: %s", name, routing_detail)
            calibration_detail = format_calibration_stats(subroute_calibration_stats)
            print(" Subroute_Calibration: " + calibration_detail)
            logging.getLogger().info("%s Subroute_Calibration: %s", name, calibration_detail)
            if subroute_decidability_stats["total"] > 0:
                decidability_detail = format_decidability_stats(subroute_decidability_stats)
                print(" Subroute_Decidability: " + decidability_detail)
                logging.getLogger().info("%s Subroute_Decidability: %s", name, decidability_detail)
            if subroute_metric_total:
                metric_detail = ", ".join(
                    "{}:ADE {:.3f}nmi/FDE {:.3f}nmi(n={})".format(
                        class_name,
                        subroute_ade_sum[class_name] / max(subroute_metric_total[class_name], 1),
                        subroute_fde_sum[class_name] / max(subroute_metric_total[class_name], 1),
                        subroute_metric_total[class_name],
                    )
                    for class_name in sorted(subroute_metric_total)
                )
                print(" Subroute_ADE_FDE_by_class: " + metric_detail)
                logging.getLogger().info("%s Subroute_ADE_FDE_by_class: %s", name, metric_detail)

        return eval_loss, ADE, FDE, rmse_cog, rmse_sog


def train(ep, parallel_train=False):
    model.train()
    teacher_forcing_ratio = branch_teacher_forcing_ratio(ep)
    total_loss = 0
    sample = 0
    epoch_total_loss = 0
    epoch_sample = 0
    train_idx_list = np.random.permutation(len(X_train)).astype("int32")
    if (
            args.use_balanced_subroute_sampling
            and train_sampling_probabilities is not None
            and args.balanced_sampling_mix_ratio > 0
    ):
        balanced_count = int(round(len(X_train) * args.balanced_sampling_mix_ratio))
        normal_count = max(len(X_train) - balanced_count, 0)
        normal_indices = train_idx_list[:normal_count]
        balanced_indices = np.random.choice(
            len(X_train),
            size=balanced_count,
            replace=True,
            p=train_sampling_probabilities,
        ).astype("int32")
        train_idx_list = np.concatenate([normal_indices, balanced_indices]).astype("int32")
        np.random.shuffle(train_idx_list)
    ADE_list = []
    FDE_list = []
    rmse_cog_list = []
    rmse_sog_list = []
    for idx in range(0, len(train_idx_list), batch_size):
        batch_indices = train_idx_list[idx:idx + batch_size]
        delta = torch.stack([X_train[i][:input_length, in_cols] for i in batch_indices]).cuda()
        src = torch.stack([X_train[i][:input_length, in_cols] for i in batch_indices]).cuda()
        tgt_y = torch.stack(
            [X_train[i][input_length:input_length + target_length, src_cols] for i in batch_indices]).cuda()
        intent_y = torch.stack(
            [X_train[i][input_length:input_length + target_length, intent_cols] for i in batch_indices]).cuda()

        optimizer.zero_grad()

        route_target = None
        if X_train_route_ids is not None:
            route_target = X_train_route_ids[batch_indices].to(device)
        batch_route_decidability = None
        if X_train_route_decidability is not None:
            batch_route_decidability = X_train_route_decidability[batch_indices].to(device)
        subroute_target = None
        if X_train_subroute_ids is not None:
            subroute_target = X_train_subroute_ids[batch_indices].to(device)
        batch_decidability = None
        if X_train_subroute_decidability is not None:
            batch_decidability = X_train_subroute_decidability[batch_indices].to(device)
        branch_decidability = batch_decidability
        if batch_route_decidability is not None:
            branch_decidability = (
                batch_route_decidability
                if branch_decidability is None
                else torch.minimum(batch_route_decidability, branch_decidability)
            )
        semantic_feature = semantic_features_for_contexts(
            None if X_train_contexts is None else X_train_contexts[batch_indices]
        )

        (
            intent,
            raw_output,
            route_logits,
            subroute_logits,
            route_feature,
            subroute_feature,
            route_decidability_logits,
            subroute_decidability_logits,
        ) = unpack_model_output(
            model(
                delta,
                src,
                semantic_feature=semantic_feature,
                route_target=route_target,
                subroute_target=subroute_target,
                teacher_forcing_ratio=teacher_forcing_ratio,
                route_supervision_weight=batch_route_decidability,
                subroute_supervision_weight=branch_decidability,
            )
        )

        value_output = compose_value_output(raw_output, src)
        value_target = tgt_y

        loss = compute_objective(
            intent,
            intent_y,
            value_output,
            value_target,
            route_logits=route_logits,
            route_target=route_target,
            route_decidability=batch_route_decidability,
            route_decidability_logits=route_decidability_logits,
            subroute_logits=subroute_logits,
            subroute_target=subroute_target,
            subroute_feature=subroute_feature,
            subroute_decidability=batch_decidability,
            subroute_decidability_logits=subroute_decidability_logits,
        )
        if args.use_candidate_selector:
            candidate_result = prepare_candidate_prediction(
                delta,
                src,
                value_output,
                route_logits,
                subroute_logits,
                subroute_feature,
                route_targets=route_target,
                subroute_targets=subroute_target,
                include_targets=args.candidate_include_target_during_training,
                target_include_mask=(
                    None
                    if branch_decidability is None
                    else branch_decidability >= max(
                        args.route_decidable_threshold,
                        args.subroute_decidable_threshold,
                    )
                ),
            )
            candidate_cost, _, _ = candidate_trajectory_costs(
                candidate_result["outputs"],
                value_target,
            )
            winner_target = torch.argmin(candidate_cost.detach(), dim=-1)
            selector_loss = candidate_soft_cost_loss(
                candidate_result["selector_logits"],
                candidate_cost,
                sample_weights=branch_decidability,
            )
            loss = loss + args.candidate_selector_weight * selector_loss
            if args.candidate_trajectory_weight > 0:
                winner_gather = winner_target[:, None, None, None].expand(
                    -1,
                    1,
                    candidate_result["outputs"].size(2),
                    candidate_result["outputs"].size(3),
                )
                winner_output = torch.gather(
                    candidate_result["outputs"],
                    1,
                    winner_gather,
                ).squeeze(1)
                candidate_regression_values = torch.mean(
                    (winner_output - value_target) ** 2,
                    dim=(1, 2),
                )
                candidate_regression_loss = weighted_loss_mean(
                    candidate_regression_values,
                    branch_decidability,
                )
                loss = loss + args.candidate_trajectory_weight * candidate_regression_loss
        with torch.no_grad():
            ADE, FDE, rmse_cog, rmse_sog, _, _ = metric_tensors(value_output, value_target)

        ADE_list.append(ADE.detach().cpu())
        FDE_list.append(FDE.detach().cpu())
        rmse_cog_list.append(rmse_cog.detach().cpu())
        rmse_sog_list.append(rmse_sog.detach().cpu())

        total_loss += loss.item()
        sample += 1
        epoch_total_loss += loss.item()
        epoch_sample += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if idx > 0 and idx % (10 * batch_size) == 0:
            cur_loss = total_loss / sample
            print("Epoch {:4d} | lr {:.9f} | loss {:.5f}".format(ep, lr, cur_loss))
            total_loss = 0.0
            sample = 0

    return epoch_total_loss / max(epoch_sample, 1)


def data_prepare(data, train_scale, valid_scale, lay_data=True):
    """
    先将时间窗拼接为3维->标准化->还原回原始形式->打乱每一个窗口->以窗口划分训练集和验证集->重新拼接
    """
    data_2lay = np.concatenate(data, axis=0)
    length = [len(l) for i, l in enumerate(data)]
    scaler_data = data_2lay[:, 2:-1]
    scaler = StandardScaler()
    scaler_data = scaler.fit_transform(scaler_data)
    mean_values = scaler.mean_
    std_values = scaler.scale_
    mean_values.astype(np.float32)
    std_values.astype(np.float32)

    data_2lay = np.concatenate((data_2lay[:, :2], scaler_data, data_2lay[:, -1:]), axis=-1)
    # 根据长度列表切割数组
    result_list = []
    start_idx = 0
    for leng in length:
        end_idx = start_idx + leng
        result_list.append(data_2lay[start_idx:end_idx])
        start_idx = end_idx
    return result_list, mean_values, std_values


if __name__ == '__main__':
    default_config_path = get_default_config_path()
    parser = ArgumentParser(description="Train and evaluate iTentformer.")
    parser.add_argument(
        "--config",
        default=default_config_path,
        help="JSON or Python config file. Defaults to config_iTentformer.py when it exists. Use --config none to disable.",
    )
    parser.add_argument("--data_path", default="dataset/example_bohai.pkl")
    parser.add_argument("--voyage_context_path", type=optional_path, default=None)
    parser.add_argument("--use_qwen_semantic_teacher", action=BooleanOptionalAction, default=False)
    parser.add_argument("--qwen_semantic_path", type=optional_path, default=None)
    parser.add_argument("--semantic_hidden_dim", type=int, default=128)
    parser.add_argument("--semantic_fusion_weight", type=float, default=0.25)
    parser.add_argument("--semantic_dropout", type=float, default=0.15)
    parser.add_argument("--group_folds_by_mmsi", action=BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run_folds", type=int, default=1)
    parser.add_argument("--input_length", type=int, default=10)
    parser.add_argument("--target_length", type=int, default=10)
    parser.add_argument("--valid_count", type=int, default=5)
    parser.add_argument("--valid_ratio", type=optional_float, default=None)
    parser.add_argument("--model_dir", default="save_models")
    parser.add_argument("--model_prefix", default="bohai")
    parser.add_argument("--eval_only", action=BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--log_file", default="train.log")
    parser.add_argument("--plot_count", type=int, default=0)
    parser.add_argument("--plot_dir", default="plots")
    parser.add_argument("--plot_strategy", choices=["first", "route_balanced", "subroute_balanced"], default="first")
    parser.add_argument("--route_labels_path", default=None)
    parser.add_argument("--stratify_by_route", action=BooleanOptionalAction, default=False)
    parser.add_argument("--use_route_intent_head", action=BooleanOptionalAction, default=False)
    parser.add_argument("--route_intent_weight", type=float, default=0.2)
    parser.add_argument("--use_route_embedding", action=BooleanOptionalAction, default=False)
    parser.add_argument("--route_embedding_dim", type=int, default=16)
    parser.add_argument("--use_route_decidability", action=BooleanOptionalAction, default=False)
    parser.add_argument("--route_decidable_min_weight", type=float, default=0.05)
    parser.add_argument("--route_decidable_confidence_threshold", type=float, default=0.60)
    parser.add_argument("--route_decidable_margin_threshold", type=float, default=0.10)
    parser.add_argument("--route_decidable_direction_points", type=int, default=4)
    parser.add_argument("--route_decidable_threshold", type=float, default=0.50)
    parser.add_argument("--route_undecidable_soft_weight", type=float, default=0.10)
    parser.add_argument("--subroute_labels_path", default=None)
    parser.add_argument("--stratify_by_subroute", action=BooleanOptionalAction, default=False)
    parser.add_argument("--use_subroute_intent_head", action=BooleanOptionalAction, default=False)
    parser.add_argument("--subroute_intent_weight", type=float, default=0.3)
    parser.add_argument("--use_subroute_embedding", action=BooleanOptionalAction, default=False)
    parser.add_argument("--subroute_embedding_dim", type=int, default=16)
    parser.add_argument("--intent_summary_mode", choices=["mean", "mean_last_delta"], default="mean")
    parser.add_argument("--branch_routing_temperature", type=float, default=1.0)
    parser.add_argument("--route_routing_temperature", type=optional_float, default=None)
    parser.add_argument("--subroute_routing_temperature", type=optional_float, default=None)
    parser.add_argument("--hard_subroute_routing", action=BooleanOptionalAction, default=False)
    parser.add_argument("--use_branch_teacher_forcing", action=BooleanOptionalAction, default=False)
    parser.add_argument("--branch_teacher_forcing_start", type=float, default=0.7)
    parser.add_argument("--branch_teacher_forcing_end", type=float, default=0.1)
    parser.add_argument("--branch_teacher_forcing_decay_epochs", type=int, default=30)
    parser.add_argument("--confidence_aware_routing", action=BooleanOptionalAction, default=False)
    parser.add_argument("--routing_confidence_threshold", type=float, default=0.8)
    parser.add_argument("--routing_margin_threshold", type=float, default=0.35)
    parser.add_argument("--routing_top_k", type=int, default=2)
    parser.add_argument("--use_learned_decidability", action=BooleanOptionalAction, default=False)
    parser.add_argument("--decidability_hidden_dim", type=int, default=64)
    parser.add_argument("--route_decidability_gate_threshold", type=float, default=0.65)
    parser.add_argument("--subroute_decidability_gate_threshold", type=float, default=0.60)
    parser.add_argument("--route_decidability_loss_weight", type=float, default=0.10)
    parser.add_argument("--subroute_decidability_loss_weight", type=float, default=0.10)
    parser.add_argument("--use_candidate_selector", action=BooleanOptionalAction, default=False)
    parser.add_argument("--candidate_count", type=int, default=2)
    parser.add_argument("--candidate_subroutes_per_route", type=int, default=2)
    parser.add_argument(
        "--candidate_pool_strategy",
        choices=["topk_routes", "all_subroutes"],
        default="topk_routes",
    )
    parser.add_argument("--candidate_max_subroutes", type=int, default=8)
    parser.add_argument("--candidate_selector_hidden_dim", type=int, default=64)
    parser.add_argument("--candidate_selector_weight", type=float, default=0.2)
    parser.add_argument("--candidate_trajectory_weight", type=float, default=0.1)
    parser.add_argument("--candidate_fde_weight", type=float, default=0.2)
    parser.add_argument("--candidate_probability_prior_weight", type=float, default=0.3)
    parser.add_argument("--candidate_base_prior_bias", type=float, default=0.5)
    parser.add_argument("--candidate_cost_temperature", type=float, default=0.35)
    parser.add_argument("--candidate_cost_regression_weight", type=float, default=0.1)
    parser.add_argument("--candidate_selector_warmup_epochs", type=int, default=10)
    parser.add_argument("--candidate_switch_confidence_threshold", type=float, default=0.45)
    parser.add_argument("--candidate_switch_logit_margin", type=float, default=0.15)
    parser.add_argument("--use_candidate_selection_calibration", action=BooleanOptionalAction, default=True)
    parser.add_argument("--candidate_calibration_max_switch_ratio", type=float, default=0.50)
    parser.add_argument("--candidate_calibration_min_cost_gain", type=float, default=0.0)
    parser.add_argument("--candidate_include_target_during_training", action=BooleanOptionalAction, default=True)
    parser.add_argument("--use_qwen_reranker", action=BooleanOptionalAction, default=False)
    parser.add_argument("--qwen_model_path", default=None)
    parser.add_argument("--qwen_adapter_path", default=None)
    parser.add_argument("--qwen_reranker_epochs", type=int, default=2)
    parser.add_argument("--qwen_reranker_batch_size", type=int, default=4)
    parser.add_argument("--qwen_train_max_windows", type=int, default=1024)
    parser.add_argument("--qwen_valid_max_windows", type=int, default=256)
    parser.add_argument("--qwen_adapter_dim", type=int, default=64)
    parser.add_argument("--qwen_reranker_lr", type=float, default=2e-4)
    parser.add_argument("--qwen_reranker_weight_decay", type=float, default=1e-4)
    parser.add_argument("--qwen_reranker_clip", type=float, default=1.0)
    parser.add_argument("--qwen_reranker_weight", type=float, default=0.5)
    parser.add_argument("--qwen_cost_temperature", type=float, default=0.25)
    parser.add_argument("--qwen_cost_regression_weight", type=float, default=0.2)
    parser.add_argument("--qwen_fused_loss_weight", type=float, default=0.5)
    parser.add_argument("--qwen_pairwise_weight", type=float, default=0.4)
    parser.add_argument("--qwen_pairwise_margin", type=float, default=0.3)
    parser.add_argument("--qwen_pairwise_min_cost_gap", type=float, default=0.01)
    parser.add_argument("--qwen_same_route_hard_weight", type=float, default=1.5)
    parser.add_argument("--qwen_calibration_max_apply_ratio", type=float, default=0.35)
    parser.add_argument("--qwen_context_max_tokens", type=int, default=64)
    parser.add_argument("--qwen_context_dropout", type=float, default=0.15)
    parser.add_argument("--qwen_gradient_checkpointing", action=BooleanOptionalAction, default=True)
    parser.add_argument("--qwen_uncertain_only", action=BooleanOptionalAction, default=True)
    parser.add_argument("--qwen_uncertainty_confidence_threshold", type=float, default=0.85)
    parser.add_argument("--qwen_uncertainty_margin_threshold", type=float, default=0.25)
    parser.add_argument("--qwen_hard_sample_ratio", type=float, default=0.8)
    parser.add_argument("--qwen_focus_high_gain", action=BooleanOptionalAction, default=True)
    parser.add_argument("--qwen_min_oracle_gain_nmi", type=float, default=0.03)
    parser.add_argument("--qwen_min_winner_gap_nmi", type=float, default=0.02)
    parser.add_argument("--qwen_gain_weight", type=float, default=2.0)
    parser.add_argument("--qwen_require_validation_gain", action=BooleanOptionalAction, default=True)
    parser.add_argument("--qwen_min_validation_gain", type=float, default=0.5)
    parser.add_argument("--qwen_min_validation_cost_gain", type=float, default=0.0)
    parser.add_argument("--use_route_prototype_prior", action=BooleanOptionalAction, default=False)
    parser.add_argument("--route_prototype_points", type=int, default=32)
    parser.add_argument("--route_prototype_weight", type=float, default=0.6)
    parser.add_argument("--use_subroute_prototype_prior", action=BooleanOptionalAction, default=False)
    parser.add_argument("--subroute_prototype_points", type=int, default=32)
    parser.add_argument("--subroute_prototype_weight", type=float, default=0.8)
    parser.add_argument("--subroute_prototype_distance_scale", type=float, default=0.25)
    parser.add_argument("--subroute_prototype_direction_weight", type=float, default=0.5)
    parser.add_argument("--use_hierarchical_intent", action=BooleanOptionalAction, default=False)
    parser.add_argument("--hierarchical_mask_strength", type=float, default=1.5)
    parser.add_argument("--confidence_gated_hierarchy", action=BooleanOptionalAction, default=False)
    parser.add_argument("--hierarchy_min_scale", type=float, default=0.15)
    parser.add_argument("--use_subroute_contrastive_loss", action=BooleanOptionalAction, default=False)
    parser.add_argument("--subroute_contrastive_weight", type=float, default=0.05)
    parser.add_argument("--subroute_contrastive_temperature", type=float, default=0.2)
    parser.add_argument("--use_subroute_focal_loss", action=BooleanOptionalAction, default=False)
    parser.add_argument("--subroute_focal_gamma", type=float, default=1.5)
    parser.add_argument("--subroute_label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_subroute_decidability", action=BooleanOptionalAction, default=False)
    parser.add_argument("--subroute_decidable_min_weight", type=float, default=0.05)
    parser.add_argument("--subroute_decidable_confidence_threshold", type=float, default=0.60)
    parser.add_argument("--subroute_decidable_margin_threshold", type=float, default=0.10)
    parser.add_argument("--subroute_decidable_direction_points", type=int, default=4)
    parser.add_argument("--subroute_decidable_threshold", type=float, default=0.50)
    parser.add_argument("--subroute_decidable_contrastive_threshold", type=float, default=0.50)
    parser.add_argument("--subroute_undecidable_soft_weight", type=float, default=0.15)
    parser.add_argument("--use_subroute_class_weight", action=BooleanOptionalAction, default=False)
    parser.add_argument("--subroute_class_weight_alpha", type=float, default=0.5)
    parser.add_argument("--subroute_class_weight_max_ratio", type=float, default=5.0)
    parser.add_argument("--use_balanced_subroute_sampling", action=BooleanOptionalAction, default=False)
    parser.add_argument("--balanced_sampling_alpha", type=float, default=0.3)
    parser.add_argument("--balanced_sampling_max_ratio", type=float, default=5.0)
    parser.add_argument("--balanced_sampling_mix_ratio", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--early_stop_metric", choices=["loss", "ade", "ade_fde"], default="loss")
    parser.add_argument("--early_stop_fde_weight", type=float, default=0.2)
    parser.add_argument("--window_stride", type=int, default=20)
    parser.add_argument("--target_mode", choices=["absolute", "residual_linear"], default="absolute")
    parser.add_argument("--use_geo_loss", action=BooleanOptionalAction, default=False)
    parser.add_argument("--geo_weight", type=float, default=0.2)
    parser.add_argument("--geo_loss_scale", type=float, default=10.0)
    parser.add_argument("--use_fde_loss", action=BooleanOptionalAction, default=False)
    parser.add_argument("--fde_weight", type=float, default=0.5)
    parser.add_argument("--use_smooth_loss", action=BooleanOptionalAction, default=False)
    parser.add_argument("--smooth_weight", type=float, default=0.2)
    parser.add_argument("--use_circular_cog", action=BooleanOptionalAction, default=False)
    parser.add_argument("--cog_weight", type=float, default=0.2)
    parser.add_argument("--cog_loss_scale", type=float, default=180.0)
    parser.add_argument("--append_log", action=BooleanOptionalAction, default=False)
    config_parser = ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=default_config_path)
    config_args, _ = config_parser.parse_known_args()
    apply_config_defaults(parser, config_args.config)
    args = parser.parse_args()
    if args.route_routing_temperature is None:
        args.route_routing_temperature = args.branch_routing_temperature
    if args.subroute_routing_temperature is None:
        args.subroute_routing_temperature = args.branch_routing_temperature
    if args.input_length < 2:
        raise ValueError("--input_length must be at least 2.")
    if args.target_length < 1:
        raise ValueError("--target_length must be positive.")
    if args.voyage_context_path and not Path(args.voyage_context_path).exists():
        raise FileNotFoundError(f"Voyage context sidecar not found: {args.voyage_context_path}")
    if args.use_qwen_semantic_teacher:
        if not args.voyage_context_path:
            raise ValueError("--use_qwen_semantic_teacher requires --voyage_context_path.")
        if not args.qwen_semantic_path:
            raise ValueError("--use_qwen_semantic_teacher requires --qwen_semantic_path.")
        if not Path(args.qwen_semantic_path).exists():
            raise FileNotFoundError(
                f"Qwen semantic sidecar not found: {args.qwen_semantic_path}. "
                "Run utils/build_qwen_semantic_teacher.py first."
            )
    if args.semantic_hidden_dim < 1:
        raise ValueError("--semantic_hidden_dim must be positive.")
    if args.semantic_fusion_weight < 0:
        raise ValueError("--semantic_fusion_weight must be non-negative.")
    if not 0 <= args.semantic_dropout < 1:
        raise ValueError("--semantic_dropout must be in [0, 1).")
    input_length = int(args.input_length)
    target_length = int(args.target_length)
    if args.geo_loss_scale <= 0:
        raise ValueError("--geo_loss_scale must be positive.")
    if args.cog_loss_scale <= 0:
        raise ValueError("--cog_loss_scale must be positive.")
    if args.use_route_embedding and not args.use_route_intent_head:
        raise ValueError("--use_route_embedding requires --use_route_intent_head.")
    if args.use_hierarchical_intent and not (args.use_route_intent_head and args.use_subroute_intent_head):
        raise ValueError("--use_hierarchical_intent requires route and subroute intent heads.")
    if args.route_intent_weight < 0:
        raise ValueError("--route_intent_weight must be non-negative.")
    if args.route_embedding_dim <= 0:
        raise ValueError("--route_embedding_dim must be positive.")
    if args.use_route_decidability and not args.use_route_intent_head:
        raise ValueError("--use_route_decidability requires --use_route_intent_head.")
    if not 0 <= args.route_decidable_min_weight <= 1:
        raise ValueError("--route_decidable_min_weight must be between 0 and 1.")
    if not 0.5 < args.route_decidable_confidence_threshold <= 1:
        raise ValueError("--route_decidable_confidence_threshold must be in (0.5, 1].")
    if not 0 < args.route_decidable_margin_threshold <= 1:
        raise ValueError("--route_decidable_margin_threshold must be in (0, 1].")
    if args.route_decidable_direction_points < 2:
        raise ValueError("--route_decidable_direction_points must be at least 2.")
    if not 0 <= args.route_decidable_threshold <= 1:
        raise ValueError("--route_decidable_threshold must be between 0 and 1.")
    if args.route_undecidable_soft_weight < 0:
        raise ValueError("--route_undecidable_soft_weight must be non-negative.")
    if args.hierarchical_mask_strength < 0:
        raise ValueError("--hierarchical_mask_strength must be non-negative.")
    if args.use_subroute_embedding and not args.use_subroute_intent_head:
        raise ValueError("--use_subroute_embedding requires --use_subroute_intent_head.")
    if args.use_subroute_contrastive_loss and not args.use_subroute_intent_head:
        raise ValueError("--use_subroute_contrastive_loss requires --use_subroute_intent_head.")
    if args.use_subroute_focal_loss and not args.use_subroute_intent_head:
        raise ValueError("--use_subroute_focal_loss requires --use_subroute_intent_head.")
    if args.subroute_intent_weight < 0:
        raise ValueError("--subroute_intent_weight must be non-negative.")
    if args.subroute_embedding_dim <= 0:
        raise ValueError("--subroute_embedding_dim must be positive.")
    if args.branch_routing_temperature <= 0:
        raise ValueError("--branch_routing_temperature must be positive.")
    if args.route_routing_temperature <= 0:
        raise ValueError("--route_routing_temperature must be positive.")
    if args.subroute_routing_temperature <= 0:
        raise ValueError("--subroute_routing_temperature must be positive.")
    if not 0 <= args.branch_teacher_forcing_start <= 1:
        raise ValueError("--branch_teacher_forcing_start must be between 0 and 1.")
    if not 0 <= args.branch_teacher_forcing_end <= 1:
        raise ValueError("--branch_teacher_forcing_end must be between 0 and 1.")
    if args.branch_teacher_forcing_decay_epochs < 1:
        raise ValueError("--branch_teacher_forcing_decay_epochs must be at least 1.")
    if not 0 <= args.routing_confidence_threshold <= 1:
        raise ValueError("--routing_confidence_threshold must be between 0 and 1.")
    if not 0 <= args.routing_margin_threshold <= 1:
        raise ValueError("--routing_margin_threshold must be between 0 and 1.")
    if args.routing_top_k < 1:
        raise ValueError("--routing_top_k must be at least 1.")
    if args.use_learned_decidability and not (
            args.use_route_decidability and args.use_subroute_decidability
    ):
        raise ValueError(
            "--use_learned_decidability requires route and subroute decidability supervision."
        )
    if args.decidability_hidden_dim < 1:
        raise ValueError("--decidability_hidden_dim must be positive.")
    if not 0 <= args.route_decidability_gate_threshold <= 1:
        raise ValueError("--route_decidability_gate_threshold must be between 0 and 1.")
    if not 0 <= args.subroute_decidability_gate_threshold <= 1:
        raise ValueError("--subroute_decidability_gate_threshold must be between 0 and 1.")
    if args.route_decidability_loss_weight < 0:
        raise ValueError("--route_decidability_loss_weight must be non-negative.")
    if args.subroute_decidability_loss_weight < 0:
        raise ValueError("--subroute_decidability_loss_weight must be non-negative.")
    if not 0 <= args.hierarchy_min_scale <= 1:
        raise ValueError("--hierarchy_min_scale must be between 0 and 1.")
    if args.use_candidate_selector and not (
            args.use_route_embedding and args.use_subroute_embedding and args.use_hierarchical_intent
    ):
        raise ValueError(
            "--use_candidate_selector requires route/subroute embeddings and hierarchical intent."
        )
    if args.candidate_count < 2:
        raise ValueError("--candidate_count must be at least 2.")
    if args.candidate_subroutes_per_route < 1:
        raise ValueError("--candidate_subroutes_per_route must be at least 1.")
    if args.candidate_max_subroutes < 1:
        raise ValueError("--candidate_max_subroutes must be at least 1.")
    if args.candidate_selector_hidden_dim < 1:
        raise ValueError("--candidate_selector_hidden_dim must be positive.")
    if args.candidate_selector_warmup_epochs < 0:
        raise ValueError("--candidate_selector_warmup_epochs must be non-negative.")
    if not 0 <= args.candidate_switch_confidence_threshold <= 1:
        raise ValueError("--candidate_switch_confidence_threshold must be between 0 and 1.")
    if args.candidate_switch_logit_margin < 0:
        raise ValueError("--candidate_switch_logit_margin must be non-negative.")
    if args.candidate_cost_temperature <= 0:
        raise ValueError("--candidate_cost_temperature must be positive.")
    if args.candidate_cost_regression_weight < 0:
        raise ValueError("--candidate_cost_regression_weight must be non-negative.")
    if not 0 < args.candidate_calibration_max_switch_ratio <= 1:
        raise ValueError("--candidate_calibration_max_switch_ratio must be in (0, 1].")
    if args.candidate_calibration_min_cost_gain < 0:
        raise ValueError("--candidate_calibration_min_cost_gain must be non-negative.")
    if args.candidate_selector_weight < 0 or args.candidate_trajectory_weight < 0:
        raise ValueError("Candidate loss weights must be non-negative.")
    if (
            args.candidate_fde_weight < 0
            or args.candidate_probability_prior_weight < 0
            or args.candidate_base_prior_bias < 0
    ):
        raise ValueError("Candidate FDE/prior weights must be non-negative.")
    if args.use_qwen_reranker and not args.use_candidate_selector:
        raise ValueError("--use_qwen_reranker requires --use_candidate_selector.")
    if args.use_qwen_reranker and not args.qwen_model_path:
        raise ValueError("--qwen_model_path is required when the Qwen reranker is enabled.")
    if args.use_qwen_reranker and not Path(args.qwen_model_path).exists():
        raise FileNotFoundError(f"Qwen model path not found: {args.qwen_model_path}")
    if args.qwen_reranker_epochs < 1 or args.qwen_reranker_batch_size < 1:
        raise ValueError("Qwen reranker epochs and batch size must be positive.")
    if args.qwen_train_max_windows < 1 or args.qwen_valid_max_windows < 1:
        raise ValueError("Qwen train/valid window limits must be positive.")
    if args.qwen_adapter_dim < 1 or args.qwen_reranker_lr <= 0:
        raise ValueError("Qwen adapter dimension and learning rate must be positive.")
    if args.qwen_reranker_weight_decay < 0 or args.qwen_reranker_clip <= 0:
        raise ValueError("Qwen weight decay must be non-negative and clip must be positive.")
    if args.qwen_reranker_weight < 0:
        raise ValueError("Qwen reranker weight must be non-negative.")
    if args.qwen_cost_temperature <= 0:
        raise ValueError("Qwen cost temperature must be positive.")
    if args.qwen_cost_regression_weight < 0:
        raise ValueError("Qwen cost regression weight must be non-negative.")
    if args.qwen_pairwise_weight < 0:
        raise ValueError("Qwen pairwise weight must be non-negative.")
    if args.qwen_pairwise_margin < 0:
        raise ValueError("Qwen pairwise margin must be non-negative.")
    if args.qwen_pairwise_min_cost_gap < 0:
        raise ValueError("Qwen pairwise minimum cost gap must be non-negative.")
    if args.qwen_same_route_hard_weight < 0:
        raise ValueError("Qwen same-route hard weight must be non-negative.")
    if args.qwen_fused_loss_weight < 0:
        raise ValueError("Qwen fused loss weight must be non-negative.")
    if not 0 < args.qwen_calibration_max_apply_ratio <= 1:
        raise ValueError("Qwen calibration max apply ratio must be in (0, 1].")
    if args.qwen_context_max_tokens < 8:
        raise ValueError("Qwen context max tokens must be at least 8.")
    if not 0 <= args.qwen_context_dropout < 1:
        raise ValueError("Qwen context dropout must be in [0, 1).")
    if not 0 <= args.qwen_uncertainty_confidence_threshold <= 1:
        raise ValueError("Qwen uncertainty confidence threshold must be between 0 and 1.")
    if not 0 <= args.qwen_uncertainty_margin_threshold <= 1:
        raise ValueError("Qwen uncertainty margin threshold must be between 0 and 1.")
    if not 0 <= args.qwen_hard_sample_ratio <= 1:
        raise ValueError("Qwen hard sample ratio must be between 0 and 1.")
    if args.qwen_min_oracle_gain_nmi < 0 or args.qwen_min_winner_gap_nmi < 0:
        raise ValueError("Qwen high-gain thresholds must be non-negative.")
    if args.qwen_gain_weight < 0:
        raise ValueError("Qwen gain weight must be non-negative.")
    if args.qwen_min_validation_gain < 0:
        raise ValueError("Qwen minimum validation gain must be non-negative.")
    if args.qwen_min_validation_cost_gain < 0:
        raise ValueError("Qwen minimum validation trajectory-cost gain must be non-negative.")
    if args.use_route_prototype_prior and not args.use_route_intent_head:
        raise ValueError("--use_route_prototype_prior requires --use_route_intent_head.")
    if args.route_prototype_points < 2:
        raise ValueError("--route_prototype_points must be at least 2.")
    if args.route_prototype_weight < 0:
        raise ValueError("--route_prototype_weight must be non-negative.")
    if args.use_subroute_prototype_prior and not args.use_subroute_intent_head:
        raise ValueError("--use_subroute_prototype_prior requires --use_subroute_intent_head.")
    if args.subroute_prototype_points < 2:
        raise ValueError("--subroute_prototype_points must be at least 2.")
    if args.subroute_prototype_weight < 0:
        raise ValueError("--subroute_prototype_weight must be non-negative.")
    if args.subroute_prototype_distance_scale <= 0:
        raise ValueError("--subroute_prototype_distance_scale must be positive.")
    if args.subroute_prototype_direction_weight < 0:
        raise ValueError("--subroute_prototype_direction_weight must be non-negative.")
    if args.subroute_contrastive_weight < 0:
        raise ValueError("--subroute_contrastive_weight must be non-negative.")
    if args.subroute_contrastive_temperature <= 0:
        raise ValueError("--subroute_contrastive_temperature must be positive.")
    if args.subroute_focal_gamma < 0:
        raise ValueError("--subroute_focal_gamma must be non-negative.")
    if not 0 <= args.subroute_label_smoothing < 1:
        raise ValueError("--subroute_label_smoothing must be in [0, 1).")
    if args.use_subroute_decidability and not args.use_subroute_intent_head:
        raise ValueError("--use_subroute_decidability requires --use_subroute_intent_head.")
    if not 0 <= args.subroute_decidable_min_weight <= 1:
        raise ValueError("--subroute_decidable_min_weight must be between 0 and 1.")
    if not 0.5 < args.subroute_decidable_confidence_threshold <= 1:
        raise ValueError("--subroute_decidable_confidence_threshold must be in (0.5, 1].")
    if not 0 < args.subroute_decidable_margin_threshold <= 1:
        raise ValueError("--subroute_decidable_margin_threshold must be in (0, 1].")
    if args.subroute_decidable_direction_points < 2:
        raise ValueError("--subroute_decidable_direction_points must be at least 2.")
    if not 0 <= args.subroute_decidable_threshold <= 1:
        raise ValueError("--subroute_decidable_threshold must be between 0 and 1.")
    if not 0 <= args.subroute_decidable_contrastive_threshold <= 1:
        raise ValueError("--subroute_decidable_contrastive_threshold must be between 0 and 1.")
    if args.subroute_undecidable_soft_weight < 0:
        raise ValueError("--subroute_undecidable_soft_weight must be non-negative.")
    if args.use_subroute_class_weight and not args.use_subroute_intent_head:
        raise ValueError("--use_subroute_class_weight requires --use_subroute_intent_head.")
    if args.use_balanced_subroute_sampling and not args.use_subroute_intent_head:
        raise ValueError("--use_balanced_subroute_sampling requires --use_subroute_intent_head.")
    if args.subroute_class_weight_alpha < 0:
        raise ValueError("--subroute_class_weight_alpha must be non-negative.")
    if args.subroute_class_weight_max_ratio < 1:
        raise ValueError("--subroute_class_weight_max_ratio must be at least 1.")
    if args.balanced_sampling_alpha < 0:
        raise ValueError("--balanced_sampling_alpha must be non-negative.")
    if args.balanced_sampling_max_ratio < 1:
        raise ValueError("--balanced_sampling_max_ratio must be at least 1.")
    if not 0 <= args.balanced_sampling_mix_ratio <= 1:
        raise ValueError("--balanced_sampling_mix_ratio must be between 0 and 1.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if args.early_stop_fde_weight < 0:
        raise ValueError("--early_stop_fde_weight must be non-negative.")

    run_name = make_run_name(args)
    run_dir = Path(args.results_dir) / run_name
    setup_logging(run_dir / args.log_file, append=args.append_log)
    logger = logging.getLogger()
    model_logger = logging.getLogger("models")
    logger.info("Run directory: %s", run_dir)
    logger.info("Arguments: %s", vars(args))
    logger.info(
        "Sequence lengths: history=%d points, prediction=%d points, total_window=%d points.",
        input_length,
        target_length,
        input_length + target_length,
    )
    logger.info(
        "Optimization switches: target_mode=%s, geo=%s(w=%.3f,scale=%.3f), "
        "fde=%s(w=%.3f), smooth=%s(w=%.3f), circular_cog=%s(w=%.3f,scale=%.3f).",
        args.target_mode,
        args.use_geo_loss,
        args.geo_weight,
        args.geo_loss_scale,
        args.use_fde_loss,
        args.fde_weight,
        args.use_smooth_loss,
        args.smooth_weight,
        args.use_circular_cog,
        args.cog_weight,
        args.cog_loss_scale,
    )
    logger.info(
        "Route switches: head=%s(w=%.3f), embedding=%s(dim=%d), hierarchical=%s(mask_strength=%.3f).",
        args.use_route_intent_head,
        args.route_intent_weight,
        args.use_route_embedding,
        args.route_embedding_dim,
        args.use_hierarchical_intent,
        args.hierarchical_mask_strength,
    )
    logger.info(
        "Qwen semantic teacher: enabled=%s, sidecar=%s, hidden=%d, "
        "fusion_weight=%.3f, dropout=%.2f; embeddings are label-free and trained "
        "with the main selector inside each fold.",
        args.use_qwen_semantic_teacher,
        args.qwen_semantic_path,
        args.semantic_hidden_dim,
        args.semantic_fusion_weight,
        args.semantic_dropout,
    )
    logger.info(
        "Subroute switches: labels=%s, head=%s(w=%.3f), embedding=%s(dim=%d), "
        "contrastive=%s(w=%.3f,temp=%.3f), focal=%s(gamma=%.3f,smooth=%.3f), "
        "stratify_by_subroute=%s.",
        args.subroute_labels_path,
        args.use_subroute_intent_head,
        args.subroute_intent_weight,
        args.use_subroute_embedding,
        args.subroute_embedding_dim,
        args.use_subroute_contrastive_loss,
        args.subroute_contrastive_weight,
        args.subroute_contrastive_temperature,
        args.use_subroute_focal_loss,
        args.subroute_focal_gamma,
        args.subroute_label_smoothing,
        args.stratify_by_subroute,
    )
    logger.info(
        "Branch routing: summary=%s, temperature=%.3f, hard_subroute=%s, "
        "teacher_forcing=%s(start=%.3f,end=%.3f,decay_epochs=%d), "
        "confidence_aware=%s(threshold=%.3f,margin=%.3f,top_k=%d), "
        "route_prototype=%s(points=%d,weight=%.3f), "
        "subroute_prototype=%s(points=%d,weight=%.3f,distance_scale=%.3f,direction_weight=%.3f), "
        "candidate_selector=%s(routes=%d,subroutes_per_route=%d,hidden=%d,selector_w=%.3f,"
        "traj_w=%.3f,fde_w=%.3f,prior_w=%.3f,base_bias=%.3f,soft_cost_temp=%.3f,"
        "cost_reg=%.3f,warmup=%d,switch_p=%.3f,switch_margin=%.3f,"
        "calibration=%s(max_switch=%.2f,min_gain=%.4f),include_target=%s).",
        args.intent_summary_mode,
        args.branch_routing_temperature,
        args.hard_subroute_routing,
        args.use_branch_teacher_forcing,
        args.branch_teacher_forcing_start,
        args.branch_teacher_forcing_end,
        args.branch_teacher_forcing_decay_epochs,
        args.confidence_aware_routing,
        args.routing_confidence_threshold,
        args.routing_margin_threshold,
        args.routing_top_k,
        args.use_route_prototype_prior,
        args.route_prototype_points,
        args.route_prototype_weight,
        args.use_subroute_prototype_prior,
        args.subroute_prototype_points,
        args.subroute_prototype_weight,
        args.subroute_prototype_distance_scale,
        args.subroute_prototype_direction_weight,
        args.use_candidate_selector,
        args.candidate_count,
        args.candidate_subroutes_per_route,
        args.candidate_selector_hidden_dim,
        args.candidate_selector_weight,
        args.candidate_trajectory_weight,
        args.candidate_fde_weight,
        args.candidate_probability_prior_weight,
        args.candidate_base_prior_bias,
        args.candidate_cost_temperature,
        args.candidate_cost_regression_weight,
        args.candidate_selector_warmup_epochs,
        args.candidate_switch_confidence_threshold,
        args.candidate_switch_logit_margin,
        args.use_candidate_selection_calibration,
        args.candidate_calibration_max_switch_ratio,
        args.candidate_calibration_min_cost_gain,
        args.candidate_include_target_during_training,
    )
    logger.info(
        "Candidate pool: strategy=%s, max_subroutes=%d; compact all-subroute mode "
        "keeps every child branch available to the selector and Qwen reranker.",
        args.candidate_pool_strategy,
        args.candidate_max_subroutes,
    )
    logger.info(
        "Routing calibration: route_temp=%.3f, subroute_temp=%.3f, "
        "learned_decidability=%s(hidden=%d,route_gate=%.3f,subroute_gate=%.3f,"
        "route_loss_w=%.3f,subroute_loss_w=%.3f), "
        "confidence_gated_hierarchy=%s(min_scale=%.3f).",
        args.route_routing_temperature,
        args.subroute_routing_temperature,
        args.use_learned_decidability,
        args.decidability_hidden_dim,
        args.route_decidability_gate_threshold,
        args.subroute_decidability_gate_threshold,
        args.route_decidability_loss_weight,
        args.subroute_decidability_loss_weight,
        args.confidence_gated_hierarchy,
        args.hierarchy_min_scale,
    )
    logger.info(
        "Subroute balance: class_weight=%s(alpha=%.3f,max_ratio=%.3f), "
        "balanced_sampling=%s(alpha=%.3f,max_ratio=%.3f,mix=%.3f).",
        args.use_subroute_class_weight,
        args.subroute_class_weight_alpha,
        args.subroute_class_weight_max_ratio,
        args.use_balanced_subroute_sampling,
        args.balanced_sampling_alpha,
        args.balanced_sampling_max_ratio,
        args.balanced_sampling_mix_ratio,
    )
    logger.info(
        "Subroute staged supervision: enabled=%s, min_hard_weight=%.3f, "
        "geometry_confidence=%.3f, geometry_margin=%.3f, direction_points=%d, "
        "decidable_threshold=%.3f, contrastive_threshold=%.3f, ambiguous_soft_weight=%.3f.",
        args.use_subroute_decidability,
        args.subroute_decidable_min_weight,
        args.subroute_decidable_confidence_threshold,
        args.subroute_decidable_margin_threshold,
        args.subroute_decidable_direction_points,
        args.subroute_decidable_threshold,
        args.subroute_decidable_contrastive_threshold,
        args.subroute_undecidable_soft_weight,
    )
    logger.info(
        "Route staged supervision: enabled=%s, min_hard_weight=%.3f, "
        "geometry_confidence=%.3f, geometry_margin=%.3f, direction_points=%d, "
        "decidable_threshold=%.3f, ambiguous_soft_weight=%.3f.",
        args.use_route_decidability,
        args.route_decidable_min_weight,
        args.route_decidable_confidence_threshold,
        args.route_decidable_margin_threshold,
        args.route_decidable_direction_points,
        args.route_decidable_threshold,
        args.route_undecidable_soft_weight,
    )
    logger.info(
        "Qwen candidate reranker: enabled=%s, model=%s, epochs=%d, batch=%d, "
        "train/valid_windows=%d/%d, adapter_dim=%d, lr=%.3e, fusion_weight=%.3f, "
        "joint_candidates=True, maritime_features=11, reverse_features=16, "
        "soft_cost(temp=%.3f,reg=%.3f,fused=%.3f), "
        "pairwise(w=%.3f,margin=%.3f,min_gap=%.3f), same_route_hard_w=%.3f, "
        "voyage_context=%s(max_tokens=%d,dropout=%.2f), "
        "uncertain_only=%s(initial_margin=%.3f,max_apply=%.3f), hard_sample_ratio=%.3f, "
        "high_gain_focus=%s(min_gain=%.4f,min_gap=%.4f,gain_w=%.2f), "
        "require_valid_gain=%s(min_winner=%.1f points,min_traj_cost=%.4f nmi).",
        args.use_qwen_reranker,
        args.qwen_model_path,
        args.qwen_reranker_epochs,
        args.qwen_reranker_batch_size,
        args.qwen_train_max_windows,
        args.qwen_valid_max_windows,
        args.qwen_adapter_dim,
        args.qwen_reranker_lr,
        args.qwen_reranker_weight,
        args.qwen_cost_temperature,
        args.qwen_cost_regression_weight,
        args.qwen_fused_loss_weight,
        args.qwen_pairwise_weight,
        args.qwen_pairwise_margin,
        args.qwen_pairwise_min_cost_gap,
        args.qwen_same_route_hard_weight,
        bool(args.voyage_context_path),
        args.qwen_context_max_tokens,
        args.qwen_context_dropout,
        args.qwen_uncertain_only,
        args.qwen_uncertainty_margin_threshold,
        args.qwen_calibration_max_apply_ratio,
        args.qwen_hard_sample_ratio,
        args.qwen_focus_high_gain,
        args.qwen_min_oracle_gain_nmi,
        args.qwen_min_winner_gap_nmi,
        args.qwen_gain_weight,
        args.qwen_require_validation_gain,
        args.qwen_min_validation_gain,
        args.qwen_min_validation_cost_gain,
    )
    logger.info(
        "Early stopping: metric=%s, fde_weight=%.3f, patience=%d.",
        args.early_stop_metric,
        args.early_stop_fde_weight,
        args.patience,
    )
    logger.info("Metrics: RMSE_COG is computed with circular 0/360 degree difference.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, precision=5, sci_mode=False)

    sample = 5
    np.random.seed(42)
    torch.manual_seed(42)
    # 'MMSI','Length','Course','Lon_d','Lat_d','SOG','vx','vy', delta 'Course','Lon_d','Lat_d','SOG','vx','vy', 'UnixTime'
    data = pd.read_pickle(args.data_path)
    voyage_context_payload = load_voyage_context_sidecar(args.voyage_context_path, data)
    qwen_semantic_payload = load_qwen_semantic_sidecar(
        args.qwen_semantic_path if args.use_qwen_semantic_teacher else None,
        voyage_context_payload,
    )
    semantic_feature_dim = (
        0 if qwen_semantic_payload is None
        else int(qwen_semantic_payload["embedding_dim"])
    )
    all_voyage_context_ids = (
        None if voyage_context_payload is None else voyage_context_payload["context_ids"]
    )
    voyage_context_text_pool = (
        None if voyage_context_payload is None else voyage_context_payload["text_pool"]
    )
    route_labels = load_route_labels(args.route_labels_path, len(data))
    route_classes, route_label_to_id, route_track_ids = build_label_encoder(route_labels)
    subroute_labels = load_label_field(args.subroute_labels_path, len(data), "subroute")
    subroute_classes, subroute_label_to_id, subroute_track_ids = build_label_encoder(subroute_labels)
    route_to_subroute_mask = build_route_to_subroute_mask(route_classes, subroute_classes)
    data_lengths = np.array([len(item) for item in data])
    logger.info(
        "Dataset loaded from %s, tracks %d, length min/mean/max %d/%.2f/%d.",
        args.data_path,
        len(data),
        data_lengths.min(),
        data_lengths.mean(),
        data_lengths.max(),
    )
    if voyage_context_payload is not None:
        alignment = voyage_context_payload.get("alignment_counters", {})
        available = int(alignment.get("available_points", 0))
        total_points = max(int(alignment.get("total_points", 0)), 1)
        logger.info(
            "Voyage context loaded from %s, unique texts %d, point coverage %.1f%%.",
            args.voyage_context_path,
            len(voyage_context_text_pool),
            100.0 * available / total_points,
        )
    if qwen_semantic_payload is not None:
        logger.info(
            "Qwen semantic teacher loaded from %s, model=%s, contexts=%d, "
            "embedding_dim=%d, label_free=%s, fusion_weight=%.3f.",
            args.qwen_semantic_path,
            qwen_semantic_payload.get("model_path"),
            qwen_semantic_payload.get("text_count"),
            semantic_feature_dim,
            qwen_semantic_payload.get("label_free"),
            args.semantic_fusion_weight,
        )
    if route_labels is not None:
        logger.info(
            "Route labels loaded from %s, classes %d, counts %s.",
            args.route_labels_path,
            len(route_classes),
            dict(Counter(route_labels)),
        )
    if subroute_labels is not None:
        logger.info(
            "Subroute labels loaded from %s, classes %d, counts %s.",
            args.subroute_labels_path,
            len(subroute_classes),
            dict(Counter(subroute_labels)),
        )
    if args.stratify_by_route and route_labels is None:
        raise ValueError("--stratify_by_route requires --route_labels_path.")
    if args.use_route_intent_head and route_labels is None:
        raise ValueError("--use_route_intent_head requires --route_labels_path.")
    if args.stratify_by_subroute and subroute_labels is None:
        raise ValueError("--stratify_by_subroute requires --subroute_labels_path.")
    if args.use_subroute_intent_head and subroute_labels is None:
        raise ValueError("--use_subroute_intent_head requires --subroute_labels_path.")

    evaluation_scores = []
    pred_list = []
    Y_list = []
    start_time = time.time()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    track_mmsi = np.asarray([int(track[0, 0]) for track in data], dtype=np.int64)
    split_labels = None
    split_label_name = "none"
    if args.stratify_by_subroute:
        split_labels = subroute_labels
        split_label_name = "subroute"
    elif args.stratify_by_route:
        split_labels = route_labels
        split_label_name = "route"

    if args.group_folds_by_mmsi and split_labels is not None:
        k_fold = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
        fold_iter = k_fold.split(np.arange(len(data)), split_labels, groups=track_mmsi)
        split_label_name = f"{split_label_name}+MMSI-grouped"
    elif args.group_folds_by_mmsi:
        k_fold = GroupKFold(n_splits=args.folds)
        fold_iter = k_fold.split(np.arange(len(data)), groups=track_mmsi)
        split_label_name = "MMSI-grouped"
    elif split_labels is not None:
        k_fold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        fold_iter = k_fold.split(np.arange(len(data)), split_labels)
    else:
        k_fold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        fold_iter = k_fold.split(data)
    logger.info("KFold split mode: %s.", split_label_name)
    fold = 1
    for i, (train_indices, test_indices) in enumerate(fold_iter):
        if fold > args.run_folds:
            break
        candidate_selection_calibration = None
        qwen_reranker = None
        qwen_reranker_runtime_active = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fold_train_indices = np.array(train_indices)
        fold_test_indices = np.array(test_indices)
        fold_train_context_tracks = (
            None if all_voyage_context_ids is None
            else [all_voyage_context_ids[int(idx)] for idx in fold_train_indices]
        )
        fold_test_context_tracks = (
            None if all_voyage_context_ids is None
            else [all_voyage_context_ids[int(idx)] for idx in fold_test_indices]
        )
        fold_train_labels = None
        fold_test_labels = None
        fold_train_route_ids = None
        fold_test_route_ids = None
        fold_train_subroute_ids = None
        fold_test_subroute_ids = None
        if route_labels is not None:
            fold_train_labels = np.array([route_labels[idx] for idx in fold_train_indices])
            fold_test_labels = np.array([route_labels[idx] for idx in fold_test_indices])
            fold_train_route_ids = route_track_ids[fold_train_indices]
            fold_test_route_ids = route_track_ids[fold_test_indices]
            logger.info(
                "Fold %d/%d route counts, train %s, test %s.",
                fold,
                args.folds,
                dict(Counter(fold_train_labels.tolist())),
                dict(Counter(fold_test_labels.tolist())),
            )
        if subroute_track_ids is not None:
            fold_train_subroute_ids = subroute_track_ids[fold_train_indices]
            fold_test_subroute_ids = subroute_track_ids[fold_test_indices]
            train_subroute_names = [subroute_classes[int(item)] for item in fold_train_subroute_ids]
            test_subroute_names = [subroute_classes[int(item)] for item in fold_test_subroute_ids]
            logger.info(
                "Fold %d/%d subroute counts, train %s, test %s.",
                fold,
                args.folds,
                dict(Counter(train_subroute_names)),
                dict(Counter(test_subroute_names)),
            )

        train_data, mean_values, std_values = data_prepare([data[i] for i in fold_train_indices], 0.6, 0.2)
        transform_matrix = np.diag(std_values[:4])
        transform_tensor = torch.from_numpy(transform_matrix).float().to(device)
        mean_tensor = torch.from_numpy(mean_values[:4]).float().to(device)

        test_data = [data[i] for i in fold_test_indices]
        test_2lay = np.concatenate(test_data, axis=0)
        length = [len(l) for i, l in enumerate(test_data)]
        scaler_data = test_2lay[:, 2:-1]
        scaler_data = (scaler_data - mean_values) / std_values

        test_2lay = np.concatenate((test_2lay[:, :2], scaler_data, test_2lay[:, -1:]), axis=-1)
        # 根据长度列表切割数组
        test_list = []
        start_idx = 0
        for leng in length:
            end_idx = start_idx + leng
            test_list.append(test_2lay[start_idx:end_idx])
            start_idx = end_idx

        valid_count, valid_split_desc = resolve_valid_count(args, len(train_data))
        if args.group_folds_by_mmsi:
            validation_labels = None
            if fold_train_subroute_ids is not None and args.stratify_by_subroute:
                validation_labels = fold_train_subroute_ids
            elif fold_train_route_ids is not None and args.stratify_by_route:
                validation_labels = fold_train_route_ids
            fold_train_mmsi = track_mmsi[fold_train_indices]
            train_indices, valid_indices = grouped_validation_split(
                validation_labels,
                fold_train_mmsi,
                valid_count,
                seed=420 + fold,
            )
            valid_split_desc += ", MMSI-grouped"
        else:
            valid_indices = np.random.choice(
                [i for i in range(len(train_data))],
                size=valid_count,
                replace=False,
            )
            train_indices_set = set([i for i in range(len(train_data))])
            valid_indices_set = set(valid_indices)
            train_indices = np.array(list(train_indices_set - valid_indices_set))
        logger.info(
            "Fold %d/%d validation split: train candidates %d, valid tracks %d (%s), final train tracks %d.",
            fold,
            args.folds,
            len(train_data),
            valid_count,
            valid_split_desc,
            len(train_indices),
        )
        train_track_labels = None if fold_train_labels is None else fold_train_labels[train_indices]
        valid_track_labels = None if fold_train_labels is None else fold_train_labels[valid_indices]
        test_track_labels = fold_test_labels
        train_route_track_ids = None if fold_train_route_ids is None else fold_train_route_ids[train_indices]
        valid_route_track_ids = None if fold_train_route_ids is None else fold_train_route_ids[valid_indices]
        test_route_track_ids = fold_test_route_ids
        train_subroute_track_ids = None if fold_train_subroute_ids is None else fold_train_subroute_ids[train_indices]
        valid_subroute_track_ids = None if fold_train_subroute_ids is None else fold_train_subroute_ids[valid_indices]
        test_subroute_track_ids = fold_test_subroute_ids
        train_context_tracks = (
            None if fold_train_context_tracks is None
            else [fold_train_context_tracks[int(idx)] for idx in train_indices]
        )
        valid_context_tracks = (
            None if fold_train_context_tracks is None
            else [fold_train_context_tracks[int(idx)] for idx in valid_indices]
        )
        test_context_tracks = fold_test_context_tracks


        def create_window_slices(data):
            window_size = input_length + target_length
            return [window_slice(trj, win_size=window_size, step=args.window_stride) for trj in data]


        X_train_slices = create_window_slices([train_data[i] for i in train_indices])
        X_valid_slices = create_window_slices([train_data[i] for i in valid_indices])
        X_test_slices = create_window_slices(test_list)
        X_train_window_labels = expand_track_labels_to_windows(train_track_labels, X_train_slices)
        X_valid_window_labels = expand_track_labels_to_windows(valid_track_labels, X_valid_slices)
        X_test_window_labels = expand_track_labels_to_windows(test_track_labels, X_test_slices)
        X_train_window_route_ids = expand_track_labels_to_windows(train_route_track_ids, X_train_slices)
        X_valid_window_route_ids = expand_track_labels_to_windows(valid_route_track_ids, X_valid_slices)
        X_test_window_route_ids = expand_track_labels_to_windows(test_route_track_ids, X_test_slices)
        X_train_window_subroute_ids = expand_track_labels_to_windows(train_subroute_track_ids, X_train_slices)
        X_valid_window_subroute_ids = expand_track_labels_to_windows(valid_subroute_track_ids, X_valid_slices)
        X_test_window_subroute_ids = expand_track_labels_to_windows(test_subroute_track_ids, X_test_slices)
        X_train_contexts = expand_track_contexts_to_windows(
            train_context_tracks,
            X_train_slices,
            voyage_context_text_pool,
            args.window_stride,
        )
        X_valid_contexts = expand_track_contexts_to_windows(
            valid_context_tracks,
            X_valid_slices,
            voyage_context_text_pool,
            args.window_stride,
        )
        X_test_contexts = expand_track_contexts_to_windows(
            test_context_tracks,
            X_test_slices,
            voyage_context_text_pool,
            args.window_stride,
        )

        X_train_list = X_train_slices
        X_valid_list = X_valid_slices
        X_test_list = X_test_slices

        X_train_list = np.concatenate(X_train_list, axis=0)
        X_valid_list = np.concatenate(X_valid_list, axis=0)
        X_test_list = np.concatenate(X_test_list, axis=0)

        X_train, X_valid, X_test = torch.tensor(X_train_list).float(), torch.tensor(
            X_valid_list).float(), torch.tensor(
            X_test_list).float()
        route_class_count = len(route_classes) if args.use_route_intent_head else 0
        subroute_class_count = len(subroute_classes) if args.use_subroute_intent_head else 0
        prototype_tracks = [train_data[i] for i in train_indices]
        route_prototypes = None
        subroute_prototypes = None
        if args.use_route_prototype_prior or args.use_route_decidability:
            route_prototypes = build_class_prototypes(
                prototype_tracks,
                train_route_track_ids,
                route_class_count,
                args.route_prototype_points,
            )
            logger.info(
                "Fold %d/%d built route prototypes from %d training tracks only, shape %s.",
                fold,
                args.folds,
                len(prototype_tracks),
                tuple(route_prototypes.shape),
            )
        if args.use_subroute_prototype_prior or args.use_subroute_decidability:
            subroute_prototypes = build_class_prototypes(
                prototype_tracks,
                train_subroute_track_ids,
                subroute_class_count,
                args.subroute_prototype_points,
            )
            logger.info(
                "Fold %d/%d built subroute prototypes from %d training tracks only, shape %s.",
                fold,
                args.folds,
                len(prototype_tracks),
                tuple(subroute_prototypes.shape),
            )
        X_train_route_ids = None
        X_valid_route_ids = None
        X_test_route_ids = None
        X_train_route_decidability = None
        X_valid_route_decidability = None
        X_test_route_decidability = None
        if X_train_window_route_ids is not None:
            X_train_route_ids = torch.tensor(X_train_window_route_ids, dtype=torch.long)
            X_valid_route_ids = torch.tensor(X_valid_window_route_ids, dtype=torch.long)
            X_test_route_ids = torch.tensor(X_test_window_route_ids, dtype=torch.long)
            if args.use_route_decidability:
                logger.info(
                    "Fold %d/%d computing history-only route decidability for train/valid/test windows.",
                    fold,
                    args.folds,
                )
                train_route_decidability = compute_class_decidability(
                    X_train,
                    X_train_window_route_ids,
                    route_prototypes,
                    route_classes,
                    input_length,
                    args.subroute_prototype_distance_scale,
                    args.subroute_prototype_direction_weight,
                    args.route_decidable_direction_points,
                    args.route_decidable_confidence_threshold,
                    args.route_decidable_margin_threshold,
                )
                valid_route_decidability = compute_class_decidability(
                    X_valid,
                    X_valid_window_route_ids,
                    route_prototypes,
                    route_classes,
                    input_length,
                    args.subroute_prototype_distance_scale,
                    args.subroute_prototype_direction_weight,
                    args.route_decidable_direction_points,
                    args.route_decidable_confidence_threshold,
                    args.route_decidable_margin_threshold,
                )
                test_route_decidability = compute_class_decidability(
                    X_test,
                    X_test_window_route_ids,
                    route_prototypes,
                    route_classes,
                    input_length,
                    args.subroute_prototype_distance_scale,
                    args.subroute_prototype_direction_weight,
                    args.route_decidable_direction_points,
                    args.route_decidable_confidence_threshold,
                    args.route_decidable_margin_threshold,
                )
                X_train_route_decidability = torch.tensor(train_route_decidability, dtype=torch.float32)
                X_valid_route_decidability = torch.tensor(valid_route_decidability, dtype=torch.float32)
                X_test_route_decidability = torch.tensor(test_route_decidability, dtype=torch.float32)
                log_class_decidability(
                    logger, fold, "route", "train", train_route_decidability,
                    X_train_window_route_ids, route_classes,
                    args.route_decidable_threshold,
                )
                log_class_decidability(
                    logger, fold, "route", "valid", valid_route_decidability,
                    X_valid_window_route_ids, route_classes,
                    args.route_decidable_threshold,
                )
                log_class_decidability(
                    logger, fold, "route", "test", test_route_decidability,
                    X_test_window_route_ids, route_classes,
                    args.route_decidable_threshold,
                )
        X_train_subroute_ids = None
        X_valid_subroute_ids = None
        X_test_subroute_ids = None
        X_train_subroute_decidability = None
        X_valid_subroute_decidability = None
        X_test_subroute_decidability = None
        subroute_class_weights = None
        train_sampling_probabilities = None
        if args.use_subroute_intent_head:
            X_train_subroute_ids = torch.tensor(X_train_window_subroute_ids, dtype=torch.long)
            X_valid_subroute_ids = torch.tensor(X_valid_window_subroute_ids, dtype=torch.long)
            X_test_subroute_ids = torch.tensor(X_test_window_subroute_ids, dtype=torch.long)
            subroute_class_count = len(subroute_classes)
            supervision_weights = None

            if args.use_subroute_decidability:
                logger.info(
                    "Fold %d/%d computing history-only subroute decidability for train/valid/test windows.",
                    fold,
                    args.folds,
                )
                subroute_group_names = [route_name_from_subroute(item) for item in subroute_classes]
                train_decidability = compute_class_decidability(
                    X_train,
                    X_train_window_subroute_ids,
                    subroute_prototypes,
                    subroute_classes,
                    input_length,
                    args.subroute_prototype_distance_scale,
                    args.subroute_prototype_direction_weight,
                    args.subroute_decidable_direction_points,
                    args.subroute_decidable_confidence_threshold,
                    args.subroute_decidable_margin_threshold,
                    group_names=subroute_group_names,
                )
                valid_decidability = compute_class_decidability(
                    X_valid,
                    X_valid_window_subroute_ids,
                    subroute_prototypes,
                    subroute_classes,
                    input_length,
                    args.subroute_prototype_distance_scale,
                    args.subroute_prototype_direction_weight,
                    args.subroute_decidable_direction_points,
                    args.subroute_decidable_confidence_threshold,
                    args.subroute_decidable_margin_threshold,
                    group_names=subroute_group_names,
                )
                test_decidability = compute_class_decidability(
                    X_test,
                    X_test_window_subroute_ids,
                    subroute_prototypes,
                    subroute_classes,
                    input_length,
                    args.subroute_prototype_distance_scale,
                    args.subroute_prototype_direction_weight,
                    args.subroute_decidable_direction_points,
                    args.subroute_decidable_confidence_threshold,
                    args.subroute_decidable_margin_threshold,
                    group_names=subroute_group_names,
                )
                X_train_subroute_decidability = torch.tensor(train_decidability, dtype=torch.float32)
                X_valid_subroute_decidability = torch.tensor(valid_decidability, dtype=torch.float32)
                X_test_subroute_decidability = torch.tensor(test_decidability, dtype=torch.float32)
                supervision_weights = (
                    args.subroute_decidable_min_weight
                    + (1.0 - args.subroute_decidable_min_weight) * train_decidability
                )
                log_class_decidability(
                    logger, fold, "subroute", "train", train_decidability,
                    X_train_window_subroute_ids, subroute_classes,
                    args.subroute_decidable_threshold,
                )
                log_class_decidability(
                    logger, fold, "subroute", "valid", valid_decidability,
                    X_valid_window_subroute_ids, subroute_classes,
                    args.subroute_decidable_threshold,
                )
                log_class_decidability(
                    logger, fold, "subroute", "test", test_decidability,
                    X_test_window_subroute_ids, subroute_classes,
                    args.subroute_decidable_threshold,
                )

            if args.use_subroute_class_weight:
                subroute_class_weights = make_subroute_class_weights(
                    X_train_window_subroute_ids,
                    subroute_class_count,
                    args.subroute_class_weight_alpha,
                    args.subroute_class_weight_max_ratio,
                    sample_weights=supervision_weights,
                )
                class_weight_values = None if subroute_class_weights is None else subroute_class_weights.detach().cpu().numpy()
                logger.info(
                    "Fold %d/%d subroute class weights: %s.",
                    fold,
                    args.folds,
                    format_class_values(class_weight_values, subroute_classes),
                )

            if args.use_balanced_subroute_sampling:
                train_sampling_probabilities = make_balanced_sampling_probabilities(
                    X_train_window_subroute_ids,
                    subroute_class_count,
                    args.balanced_sampling_alpha,
                    args.balanced_sampling_max_ratio,
                    supervision_weights=supervision_weights,
                )
                if train_sampling_probabilities is not None:
                    natural_mass = np.bincount(
                        X_train_window_subroute_ids,
                        minlength=subroute_class_count,
                    ).astype(np.float32)
                    natural_mass = natural_mass / max(float(np.sum(natural_mass)), 1.0)
                    balanced_mass = np.bincount(
                        X_train_window_subroute_ids,
                        weights=train_sampling_probabilities,
                        minlength=subroute_class_count,
                    )
                    mixed_mass = (
                        (1.0 - args.balanced_sampling_mix_ratio) * natural_mass
                        + args.balanced_sampling_mix_ratio * balanced_mass
                    )
                    logger.info(
                        "Fold %d/%d subroute sampling expected class ratio: %s.",
                        fold,
                        args.folds,
                        format_class_values(mixed_mass, subroute_classes),
                    )
        logger.info(
            "Fold %d/%d prepared, tracks train/valid/test %d/%d/%d, windows train/valid/test %d/%d/%d.",
            fold,
            args.folds,
            len(train_indices),
            len(valid_indices),
            len(test_list),
            len(X_train),
            len(X_valid),
            len(X_test),
        )
        if X_test_window_labels is not None:
            logger.info("Fold %d/%d test window route counts %s.", fold, args.folds, dict(Counter(X_test_window_labels.tolist())))
        if X_test_window_subroute_ids is not None:
            test_window_subroutes = [subroute_classes[int(item)] for item in X_test_window_subroute_ids.tolist()]
            logger.info(
                "Fold %d/%d test window subroute counts %s.",
                fold,
                args.folds,
                dict(Counter(test_window_subroutes)),
        )
        """---------------------------"""
        lr = 2e-4
        model = iTentformer(input_size_tcn, input_size, local_intent_size, output_size, concat_dim, input_length,
                              target_length,
                              num_channels, kernel_size, d_model, dropout,
                              subroute_classes=subroute_class_count,
                              use_subroute_intent_head=args.use_subroute_intent_head,
                              use_subroute_embedding=args.use_subroute_embedding,
                              subroute_embedding_dim=args.subroute_embedding_dim,
                              route_classes=route_class_count,
                              use_route_intent_head=args.use_route_intent_head,
                              use_route_embedding=args.use_route_embedding,
                              route_embedding_dim=args.route_embedding_dim,
                              use_hierarchical_intent=args.use_hierarchical_intent,
                              route_to_subroute_mask=route_to_subroute_mask,
                              hierarchical_mask_strength=args.hierarchical_mask_strength,
                              intent_summary_mode=args.intent_summary_mode,
                              branch_routing_temperature=args.branch_routing_temperature,
                              route_routing_temperature=args.route_routing_temperature,
                              subroute_routing_temperature=args.subroute_routing_temperature,
                              hard_subroute_routing=args.hard_subroute_routing,
                              route_prototypes=route_prototypes,
                              route_prototype_prior_weight=(
                                  args.route_prototype_weight
                                  if args.use_route_prototype_prior else 0.0
                              ),
                              subroute_prototypes=subroute_prototypes,
                              prototype_prior_weight=(
                                  args.subroute_prototype_weight
                                  if args.use_subroute_prototype_prior else 0.0
                              ),
                              prototype_distance_scale=args.subroute_prototype_distance_scale,
                              prototype_direction_weight=args.subroute_prototype_direction_weight,
                              confidence_aware_routing=args.confidence_aware_routing,
                              routing_confidence_threshold=args.routing_confidence_threshold,
                              routing_margin_threshold=args.routing_margin_threshold,
                              use_learned_decidability=args.use_learned_decidability,
                              decidability_hidden_dim=args.decidability_hidden_dim,
                              route_decidability_gate_threshold=args.route_decidability_gate_threshold,
                              subroute_decidability_gate_threshold=args.subroute_decidability_gate_threshold,
                              confidence_gated_hierarchy=args.confidence_gated_hierarchy,
                              hierarchy_min_scale=args.hierarchy_min_scale,
                              routing_top_k=args.routing_top_k,
                              use_candidate_selector=args.use_candidate_selector,
                              candidate_selector_hidden_dim=args.candidate_selector_hidden_dim,
                              candidate_probability_prior_weight=args.candidate_probability_prior_weight,
                              candidate_base_prior_bias=args.candidate_base_prior_bias,
                              use_semantic_teacher=args.use_qwen_semantic_teacher,
                              semantic_feature_dim=semantic_feature_dim,
                              semantic_hidden_dim=args.semantic_hidden_dim,
                              semantic_fusion_weight=args.semantic_fusion_weight,
                              semantic_dropout=args.semantic_dropout).to(device)
        awl = AutomaticWeightedLoss(2).cuda()
        if fold == 1:
            model_logger.info("number of parameters: %.6e", count_parameters(model))
            model_logger.info("number of AWL parameters: %.6e", count_parameters(awl))
            if args.use_route_intent_head:
                model_logger.info("route classes: %d, labels: %s", route_class_count, route_classes)
            if args.use_subroute_intent_head:
                model_logger.info("subroute classes: %d, labels: %s", subroute_class_count, subroute_classes)
        optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': awl.parameters()}], lr=lr, weight_decay=0)

        best_monitor_score = 1e8
        lr_lower_bound = 1e-10
        monitor_score_list = []
        model_name = str(Path(args.model_dir) / f"{args.model_prefix}_K{fold}.pt")
        qwen_adapter_name = str(
            Path(args.qwen_adapter_path)
            if args.qwen_adapter_path
            else Path(args.model_dir) / f"{args.model_prefix}_K{fold}_qwen_reranker.pt"
        )
        best_epoch = 0
        early_stopping = EarlyStopping(patience=args.patience, verbose=False)

        if args.eval_only:
            candidate_selector_runtime_active = args.use_candidate_selector
            checkpoint_path = args.checkpoint_path or model_name
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            logger.info("Eval-only mode, fold %d/%d, loading model from %s.", fold, args.folds, checkpoint_path)
            del optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = load_model_checkpoint(checkpoint_path, model)
            candidate_selector_runtime_active = args.use_candidate_selector
            calibrate_candidate_selection(
                X_valid,
                route_targets=X_valid_route_ids,
                subroute_targets=X_valid_subroute_ids,
                voyage_contexts=X_valid_contexts,
            )
            if args.use_qwen_reranker:
                if Path(qwen_adapter_name).exists():
                    qwen_reranker, adapter_accepted = load_qwen_candidate_reranker(qwen_adapter_name)
                    qwen_reranker_runtime_active = (
                        adapter_accepted or not args.qwen_require_validation_gain
                    )
                    if not qwen_reranker_runtime_active:
                        logger.warning(
                            "Qwen reranker did not improve validation winner accuracy; using the base selector."
                        )
                else:
                    logger.warning(
                        "Qwen reranker adapter not found at %s; evaluation will use the base selector.",
                        qwen_adapter_name,
                    )
            tloss, ADE, FDE, rmse_cog, rmse_sog = evaluate(
                X_test,
                route_targets=X_test_route_ids,
                route_decidability=X_test_route_decidability,
                subroute_targets=X_test_subroute_ids,
                subroute_decidability=X_test_subroute_decidability,
                name='Final Test',
                route_class_names=route_classes,
                subroute_class_names=subroute_classes,
                voyage_contexts=X_test_contexts,
            )
            log_metric_line(logger, "Final Test", fold, args.folds, 0, tloss, ADE, FDE, rmse_cog, rmse_sog)
            if args.plot_count > 0:
                fold_plot_dir = run_dir / args.plot_dir
                save_prediction_plots(
                    X_test,
                    fold,
                    fold_plot_dir,
                    args.plot_count,
                    window_labels=X_test_window_labels,
                    plot_strategy=args.plot_strategy,
                    window_route_ids=X_test_window_route_ids,
                    route_classes=route_classes,
                    window_subroute_ids=X_test_window_subroute_ids,
                    subroute_classes=subroute_classes,
                    voyage_contexts=X_test_contexts,
                )
                logger.info(
                    "Saved %d prediction plot(s) for fold %d/%d to %s with strategy %s.",
                    min(args.plot_count, len(X_test)),
                    fold,
                    args.folds,
                    fold_plot_dir,
                    args.plot_strategy,
                )
            print('-' * 89)
            print("K={}: ADE: {:.5f}nmi, FDE: {:.5f}nmi, COG_RMSE: {:.5f}, SOG_RMSE: {:.5f}kn".format(
                fold, ADE, FDE, rmse_cog, rmse_sog
            ))
            print('-' * 89)
            evaluation_score = np.array([ADE.cpu(), FDE.cpu(), rmse_cog.cpu(), rmse_sog.cpu()])
            evaluation_scores.append(evaluation_score)
            fold += 1
            continue

        # trainning
        for ep in range(1, args.epochs + 1):
            candidate_selector_runtime_active = (
                args.use_candidate_selector
                and ep > args.candidate_selector_warmup_epochs
            )
            logger.info(
                "Branch teacher forcing, fold %d/%d, epoch %03d, ratio %.3f.",
                fold,
                args.folds,
                ep,
                branch_teacher_forcing_ratio(ep),
            )
            logger.info(
                "Candidate selector runtime, fold %d/%d, epoch %03d, active=%s (warmup=%d).",
                fold,
                args.folds,
                ep,
                candidate_selector_runtime_active,
                args.candidate_selector_warmup_epochs,
            )
            train_loss = train(ep, parallel_train=False)
            logger.info("Training, fold %d/%d, epoch %03d, loss %.5f, lr %.6e.", fold, args.folds, ep, train_loss, lr)

            vloss, vade, vfde, vrmse_cog, vrmse_sog = evaluate(
                X_valid,
                route_targets=X_valid_route_ids,
                route_decidability=X_valid_route_decidability,
                subroute_targets=X_valid_subroute_ids,
                subroute_decidability=X_valid_subroute_decidability,
                name='Validation',
                route_class_names=route_classes,
                subroute_class_names=subroute_classes,
                voyage_contexts=X_valid_contexts,
            )
            log_metric_line(logger, "Valid", fold, args.folds, ep, vloss, vade, vfde, vrmse_cog, vrmse_sog)

            tloss, tade, tfde, trmse_cog, trmse_sog = evaluate(
                X_test,
                route_targets=X_test_route_ids,
                route_decidability=X_test_route_decidability,
                subroute_targets=X_test_subroute_ids,
                subroute_decidability=X_test_subroute_decidability,
                name='Test',
                route_class_names=route_classes,
                subroute_class_names=subroute_classes,
                voyage_contexts=X_test_contexts,
            )
            log_metric_line(logger, "Test", fold, args.folds, ep, tloss, tade, tfde, trmse_cog, trmse_sog)
            monitor_score, monitor_name = early_stop_monitor_value(vloss, vade, vfde)
            logger.info(
                "Early-stop monitor, fold %d/%d, epoch %03d, %s %.5f.",
                fold,
                args.folds,
                ep,
                monitor_name,
                monitor_score,
            )

            improved = monitor_score < best_monitor_score
            if improved:
                with open(model_name, "wb") as f:
                    torch.save(model, f)
                    print("Saved model!\n")
                best_epoch = ep
                logger.info(
                    "Best epoch: %03d, fold %d/%d, %s %.5f, saving model to %s",
                    ep,
                    fold,
                    args.folds,
                    monitor_name,
                    monitor_score,
                    model_name,
                )
                best_monitor_score = monitor_score

            early_stopping(monitor_score, model)
            if not improved:
                logger.info(
                    "No improvement, fold %d/%d, epoch %03d, %s %.5f, best %.5f, early-stop counter %d/%d.",
                    fold,
                    args.folds,
                    ep,
                    monitor_name,
                    monitor_score,
                    best_monitor_score,
                    early_stopping.counter,
                    early_stopping.patience,
                )
            if early_stopping.early_stop:
                logger.info(
                    "Early stopping, fold %d/%d, epoch %03d, best epoch %03d, best %s %.5f.",
                    fold,
                    args.folds,
                    ep,
                    best_epoch,
                    monitor_name,
                    best_monitor_score,
                )
                print("Early stopping")
                break

            if ep > 5 and len(monitor_score_list) >= 3 and monitor_score > max(monitor_score_list[-3:]) and lr > lr_lower_bound:
                lr /= 2.0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                logger.info(
                    "Learning rate reduced, fold %d/%d, epoch %03d, %s %.5f, lr %.6e.",
                    fold,
                    args.folds,
                    ep,
                    monitor_name,
                    monitor_score,
                    lr,
                )

            monitor_score_list.append(monitor_score)

        del optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = load_model_checkpoint(model_name, model)
        candidate_selector_runtime_active = args.use_candidate_selector
        calibrate_candidate_selection(
            X_valid,
            route_targets=X_valid_route_ids,
            subroute_targets=X_valid_subroute_ids,
            voyage_contexts=X_valid_contexts,
        )
        if args.use_qwen_reranker:
            logger.info("Starting second-stage Qwen candidate reranker training for fold %d/%d.", fold, args.folds)
            qwen_reranker, adapter_accepted = train_qwen_candidate_reranker(
                X_train,
                X_train_route_ids,
                X_train_subroute_ids,
                X_valid,
                X_valid_route_ids,
                X_valid_subroute_ids,
                qwen_adapter_name,
                fold,
                train_context_data=X_train_contexts,
                valid_context_data=X_valid_contexts,
            )
            qwen_reranker_runtime_active = (
                adapter_accepted or not args.qwen_require_validation_gain
            )
            if not qwen_reranker_runtime_active:
                logger.warning(
                    "Qwen reranker did not meet the validation gain threshold; final test uses the base selector."
                )
        """
        You can test the model by applying the training process annotation to the following code
        """
        tloss, ADE, FDE, rmse_cog, rmse_sog = evaluate(
            X_test,
            route_targets=X_test_route_ids,
            route_decidability=X_test_route_decidability,
            subroute_targets=X_test_subroute_ids,
            subroute_decidability=X_test_subroute_decidability,
            name='Final Test',
            route_class_names=route_classes,
            subroute_class_names=subroute_classes,
            voyage_contexts=X_test_contexts,
        )
        log_metric_line(logger, "Final Test", fold, args.folds, best_epoch, tloss, ADE, FDE, rmse_cog, rmse_sog)
        if args.plot_count > 0:
            fold_plot_dir = run_dir / args.plot_dir
            save_prediction_plots(
                X_test,
                fold,
                fold_plot_dir,
                args.plot_count,
                window_labels=X_test_window_labels,
                plot_strategy=args.plot_strategy,
                window_route_ids=X_test_window_route_ids,
                route_classes=route_classes,
                window_subroute_ids=X_test_window_subroute_ids,
                subroute_classes=subroute_classes,
                voyage_contexts=X_test_contexts,
            )
            logger.info(
                "Saved %d prediction plot(s) for fold %d/%d to %s with strategy %s.",
                min(args.plot_count, len(X_test)),
                fold,
                args.folds,
                fold_plot_dir,
                args.plot_strategy,
            )
        print('-' * 89)
        print("K={}: ADE: {:.5f}nmi, FDE: {:.5f}nmi, COG_RMSE: {:.5f}, SOG_RMSE: {:.5f}kn".format(fold, ADE, FDE,
                                                                                                  rmse_cog, rmse_sog))
        print('-' * 89)
        evaluation_score = np.array([ADE.cpu(), FDE.cpu(), rmse_cog.cpu(), rmse_sog.cpu()])
        evaluation_scores.append(evaluation_score)
        # tloss, ADE, FDE, rmse_cog, rmse_sog = evaluate(X_test, plot=False)
        # print('-' * 89)
        # print("K={}: ADE: {:.5f}nmi, FDE: {:.5f}nmi, COG_RMSE: {:.5f}°, SOG_RMSE: {:.5f}kn".format(fold, ADE, FDE,
        #                                                                                            rmse_cog, rmse_sog))
        # print('-' * 89)
        # evaluation_score = np.array([ADE, FDE, rmse_cog, rmse_sog])
        # evaluation_scores.append(evaluation_score)

        fold += 1

    end_time = time.time()
    print('-' * 89)
    print("Training time: {:.3f} s".format((end_time - start_time)))
    if not evaluation_scores:
        logger.info("No folds were evaluated. Check --run_folds if this was not intentional.")
        logger.info("Log saved to %s", run_dir / args.log_file)
        sys.exit(0)
    mean_evaluation_score = np.mean(evaluation_scores, axis=0)
    print("Mean evaluation score across all folds:", mean_evaluation_score)
    logger.info("Training time: %.3f s.", end_time - start_time)
    logger.info(
        "Mean evaluation score across all folds: ADE %.5fnmi (%.2fm), FDE %.5fnmi (%.2fm), "
        "RMSE_COG %.5fdeg, RMSE_SOG %.5fkn.",
        mean_evaluation_score[0],
        mean_evaluation_score[0] * 1852.0,
        mean_evaluation_score[1],
        mean_evaluation_score[1] * 1852.0,
        mean_evaluation_score[2],
        mean_evaluation_score[3],
    )
    logger.info("Log saved to %s", run_dir / args.log_file)
