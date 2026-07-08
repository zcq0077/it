import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold
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
import json
import logging
import sys
import time
from utils.EarlyStopping import EarlyStopping
from utils.Haversine_Loss import HaversineLoss

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


def optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "false"}:
        return None
    return float(value)


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


def compute_objective(intent, intent_y, value_output, value_target):
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


def load_route_labels(path, expected_count):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if len(labels) != expected_count:
        raise ValueError(f"Route labels count {len(labels)} does not match data count {expected_count}.")
    return [str(item["route"]) for item in labels]


def expand_track_labels_to_windows(track_labels, window_slices):
    if track_labels is None:
        return None
    window_labels = []
    for label, windows in zip(track_labels, window_slices):
        window_labels.extend([label] * len(windows))
    return np.array(window_labels)


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


def save_prediction_plots(X_data, fold, output_dir, max_samples, window_labels=None, plot_strategy="first"):
    if max_samples <= 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    plot_indices = choose_plot_indices(window_labels, min(max_samples, len(X_data)), plot_strategy, seed=fold * 1009)

    for plot_idx, sample_idx in enumerate(plot_indices):
        sample_data = X_data[sample_idx]
        route_label = None if window_labels is None else str(window_labels[sample_idx])
        delta = sample_data[:input_length, in_cols].unsqueeze(0).to(device)
        src = sample_data[:input_length, in_cols].unsqueeze(0).to(device)

        with torch.no_grad():
            _, raw_output = model(delta, src)
            output = compose_value_output(raw_output, src)

        pred = output.squeeze(0).detach().cpu().numpy()
        history = sample_data[:input_length, src_cols].detach().cpu().numpy()
        target = sample_data[input_length:input_length + target_length, src_cols].detach().cpu().numpy()

        pred = inverse_standardized(pred, transform_matrix, mean_values)
        history = inverse_standardized(history, transform_matrix, mean_values)
        target = inverse_standardized(target, transform_matrix, mean_values)

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
        title_route = "" if route_label is None else f" [{route_label}]"
        ax.set_title(f"Fold {fold} Sample {sample_idx}{title_route}: history vs prediction vs ground truth")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="best", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()

        route_part = "" if route_label is None else f"_{route_label}"
        save_path = output_dir / f"fold_{fold:02d}_sample_{plot_idx:03d}_idx{sample_idx:05d}{route_part}.png"
        fig.savefig(save_path)
        plt.close(fig)


def evaluate(X_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    sample = 0
    ADE_list = []
    FDE_list = []
    rmse_cog_list = []
    rmse_sog_list = []
    with torch.no_grad():
        for idx in range(0, len(eval_idx_list), batch_size):
            batch_indices = eval_idx_list[idx:idx + batch_size]
            delta = torch.stack([X_data[i][:input_length, in_cols] for i in batch_indices]).cuda()
            src = torch.stack([X_data[i][:input_length, in_cols] for i in batch_indices]).cuda()
            tgt_y = torch.stack(
                [X_data[i][input_length:input_length + target_length, src_cols] for i in batch_indices]).cuda()
            intent_y = torch.stack(
                [X_data[i][input_length:input_length + target_length, intent_cols] for i in batch_indices]).cuda()

            intent, raw_output = model(delta, src)
            value_output = compose_value_output(raw_output, src)
            value_target = tgt_y

            loss = compute_objective(intent, intent_y, value_output, value_target)
            ADE, FDE, rmse_cog, rmse_sog, real_output, real_target = metric_tensors(value_output, value_target)

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

        return eval_loss, ADE, FDE, rmse_cog, rmse_sog


def train(ep, parallel_train=False):
    model.train()
    total_loss = 0
    sample = 0
    epoch_total_loss = 0
    epoch_sample = 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
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

        intent, raw_output = model(delta, src)

        value_output = compose_value_output(raw_output, src)
        value_target = tgt_y

        loss = compute_objective(intent, intent_y, value_output, value_target)
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
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run_folds", type=int, default=1)
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
    parser.add_argument("--plot_strategy", choices=["first", "route_balanced"], default="first")
    parser.add_argument("--route_labels_path", default=None)
    parser.add_argument("--stratify_by_route", action=BooleanOptionalAction, default=False)
    parser.add_argument("--patience", type=int, default=3)
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
    if args.geo_loss_scale <= 0:
        raise ValueError("--geo_loss_scale must be positive.")
    if args.cog_loss_scale <= 0:
        raise ValueError("--cog_loss_scale must be positive.")

    run_name = make_run_name(args)
    run_dir = Path(args.results_dir) / run_name
    setup_logging(run_dir / args.log_file, append=args.append_log)
    logger = logging.getLogger()
    model_logger = logging.getLogger("models")
    logger.info("Run directory: %s", run_dir)
    logger.info("Arguments: %s", vars(args))
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
    route_labels = load_route_labels(args.route_labels_path, len(data))
    data_lengths = np.array([len(item) for item in data])
    logger.info(
        "Dataset loaded from %s, tracks %d, length min/mean/max %d/%.2f/%d.",
        args.data_path,
        len(data),
        data_lengths.min(),
        data_lengths.mean(),
        data_lengths.max(),
    )
    if route_labels is not None:
        logger.info("Route labels loaded from %s, counts %s.", args.route_labels_path, dict(Counter(route_labels)))
    if args.stratify_by_route and route_labels is None:
        raise ValueError("--stratify_by_route requires --route_labels_path.")

    evaluation_scores = []
    pred_list = []
    Y_list = []
    start_time = time.time()

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    if args.stratify_by_route:
        k_fold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        fold_iter = k_fold.split(np.arange(len(data)), route_labels)
    else:
        k_fold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        fold_iter = k_fold.split(data)
    fold = 1
    for i, (train_indices, test_indices) in enumerate(fold_iter):
        if fold > args.run_folds:
            break
        fold_train_indices = np.array(train_indices)
        fold_test_indices = np.array(test_indices)
        fold_train_labels = None
        fold_test_labels = None
        if route_labels is not None:
            fold_train_labels = np.array([route_labels[idx] for idx in fold_train_indices])
            fold_test_labels = np.array([route_labels[idx] for idx in fold_test_indices])
            logger.info(
                "Fold %d/%d route counts, train %s, test %s.",
                fold,
                args.folds,
                dict(Counter(fold_train_labels.tolist())),
                dict(Counter(fold_test_labels.tolist())),
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
        valid_indices = np.random.choice([i for i in range(len(train_data))], size=valid_count, replace=False)
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


        def create_window_slices(data):
            return [window_slice(trj, win_size=20, step=args.window_stride) for trj in data]


        X_train_slices = create_window_slices([train_data[i] for i in train_indices])
        X_valid_slices = create_window_slices([train_data[i] for i in valid_indices])
        X_test_slices = create_window_slices(test_list)
        X_train_window_labels = expand_track_labels_to_windows(train_track_labels, X_train_slices)
        X_valid_window_labels = expand_track_labels_to_windows(valid_track_labels, X_valid_slices)
        X_test_window_labels = expand_track_labels_to_windows(test_track_labels, X_test_slices)

        X_train_list = X_train_slices
        X_valid_list = X_valid_slices
        X_test_list = X_test_slices

        X_train_list = np.concatenate(X_train_list, axis=0)
        X_valid_list = np.concatenate(X_valid_list, axis=0)
        X_test_list = np.concatenate(X_test_list, axis=0)

        X_train, X_valid, X_test = torch.tensor(X_train_list).float(), torch.tensor(
            X_valid_list).float(), torch.tensor(
            X_test_list).float()
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
        """---------------------------"""
        lr = 2e-4
        model = iTentformer(input_size_tcn, input_size, local_intent_size, output_size, concat_dim, input_length,
                              num_channels, kernel_size, d_model, dropout).to(device)
        awl = AutomaticWeightedLoss(2).cuda()
        if fold == 1:
            model_logger.info("number of parameters: %.6e", count_parameters(model))
            model_logger.info("number of AWL parameters: %.6e", count_parameters(awl))
        optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': awl.parameters()}], lr=lr, weight_decay=0)

        best_vloss = 1e8
        lr_lower_bound = 1e-10
        vloss_list = []
        model_name = str(Path(args.model_dir) / f"{args.model_prefix}_K{fold}.pt")
        best_epoch = 0
        early_stopping = EarlyStopping(patience=args.patience, verbose=False)

        if args.eval_only:
            checkpoint_path = args.checkpoint_path or model_name
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            logger.info("Eval-only mode, fold %d/%d, loading model from %s.", fold, args.folds, checkpoint_path)
            model = load_model_checkpoint(checkpoint_path, model)
            tloss, ADE, FDE, rmse_cog, rmse_sog = evaluate(X_test, name='Final Test')
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
            train_loss = train(ep, parallel_train=False)
            logger.info("Training, fold %d/%d, epoch %03d, loss %.5f, lr %.6e.", fold, args.folds, ep, train_loss, lr)

            vloss, vade, vfde, vrmse_cog, vrmse_sog = evaluate(X_valid, name='Validation')
            log_metric_line(logger, "Valid", fold, args.folds, ep, vloss, vade, vfde, vrmse_cog, vrmse_sog)

            tloss, tade, tfde, trmse_cog, trmse_sog = evaluate(X_test, name='Test')
            log_metric_line(logger, "Test", fold, args.folds, ep, tloss, tade, tfde, trmse_cog, trmse_sog)
            # 设置 EarlyStopping
            early_stopping(vloss, model)

            if early_stopping.early_stop:
                logger.info("Early stopping, fold %d/%d, epoch %03d, best epoch %03d.", fold, args.folds, ep, best_epoch)
                print("Early stopping")
                break

            if vloss < best_vloss:
                with open(model_name, "wb") as f:
                    torch.save(model, f)
                    print("Saved model!\n")
                best_epoch = ep
                logger.info("Best epoch: %03d, fold %d/%d, saving model to %s", ep, fold, args.folds, model_name)
                best_vloss = vloss
            else:
                logger.info(
                    "No improvement, fold %d/%d, epoch %03d, early-stop counter %d/%d.",
                    fold,
                    args.folds,
                    ep,
                    early_stopping.counter,
                    early_stopping.patience,
                )

            if ep > 5 and vloss > max(vloss_list[-3:]) and lr > lr_lower_bound:
                lr /= 2.0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                logger.info("Learning rate reduced, fold %d/%d, epoch %03d, lr %.6e.", fold, args.folds, ep, lr)

            vloss_list.append(vloss)

        model = load_model_checkpoint(model_name, model)
        """
        You can test the model by applying the training process annotation to the following code
        """
        tloss, ADE, FDE, rmse_cog, rmse_sog = evaluate(X_test, name='Final Test')
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
