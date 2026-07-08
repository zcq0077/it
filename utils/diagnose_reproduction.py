from pathlib import Path
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.bohai_diff import window_slice


SRC_COLS = [2, 3, 4, 5]
INPUT_LENGTH = 10
TARGET_LENGTH = 10


def haversine_distance_2arrays(lat_a, lon_a, lat_b, lon_b):
    radius_km = 6371.0
    lat_a = np.radians(lat_a)
    lon_a = np.radians(lon_a)
    lat_b = np.radians(lat_b)
    lon_b = np.radians(lon_b)
    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    distance_km = 2 * radius_km * np.arcsin(
        np.sqrt(np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2)
    )
    return distance_km / 1.852


def standardize_like_training(train_tracks, test_tracks):
    train_2lay = np.concatenate(train_tracks, axis=0)
    scaler = StandardScaler()
    scaler.fit(train_2lay[:, 2:-1])

    def transform_tracks(tracks):
        flat = np.concatenate(tracks, axis=0)
        lengths = [len(track) for track in tracks]
        scaled = (flat[:, 2:-1] - scaler.mean_) / scaler.scale_
        flat = np.concatenate((flat[:, :2], scaled, flat[:, -1:]), axis=-1)
        out = []
        start = 0
        for length in lengths:
            out.append(flat[start:start + length])
            start += length
        return out

    return transform_tracks(train_tracks), transform_tracks(test_tracks), scaler.mean_, scaler.scale_


def make_windows(tracks, window_stride):
    windows = [window_slice(track, win_size=20, step=window_stride) for track in tracks]
    windows = [item for item in windows if item is not None and len(item) > 0]
    return np.concatenate(windows, axis=0)


def inverse_src(values, mean_values, std_values):
    transform_matrix = np.diag(std_values[:4])
    return values @ transform_matrix + mean_values[:4]


def calculate_metrics(pred, target):
    pred_lon_lat = pred[:, :, 1:3].reshape(-1, 2)
    target_lon_lat = target[:, :, 1:3].reshape(-1, 2)
    dist = haversine_distance_2arrays(
        pred_lon_lat[:, 1],
        pred_lon_lat[:, 0],
        target_lon_lat[:, 1],
        target_lon_lat[:, 0],
    ).reshape(-1, TARGET_LENGTH)

    ade = dist.mean()
    fde = dist[:, -1].mean()
    cog_diff = (pred[:, :, 0] - target[:, :, 0] + 180.0) % 360.0 - 180.0
    rmse_cog = np.sqrt((cog_diff ** 2).mean())
    rmse_sog = np.sqrt(((pred[:, :, 3] - target[:, :, 3]) ** 2).mean())
    return ade, fde, rmse_cog, rmse_sog


def eval_constant_baseline(windows, mean_values, std_values):
    history = windows[:, :INPUT_LENGTH, SRC_COLS]
    target = windows[:, INPUT_LENGTH:INPUT_LENGTH + TARGET_LENGTH, SRC_COLS]
    pred = np.repeat(history[:, -1:, :], TARGET_LENGTH, axis=1)

    pred = inverse_src(pred, mean_values, std_values)
    target = inverse_src(target, mean_values, std_values)

    return calculate_metrics(pred, target)


def eval_linear_baseline(windows, mean_values, std_values):
    history = windows[:, :INPUT_LENGTH, SRC_COLS]
    target = windows[:, INPUT_LENGTH:INPUT_LENGTH + TARGET_LENGTH, SRC_COLS]

    history = inverse_src(history, mean_values, std_values)
    target = inverse_src(target, mean_values, std_values)

    last = history[:, -1:, :]
    delta = history[:, -1:, :] - history[:, -2:-1, :]
    steps = np.arange(1, TARGET_LENGTH + 1, dtype=np.float64).reshape(1, TARGET_LENGTH, 1)
    pred = last + steps * delta
    pred[:, :, 0] = pred[:, :, 0] % 360.0
    pred[:, :, 3] = np.maximum(pred[:, :, 3], 0.0)

    return calculate_metrics(pred, target)


def main():
    parser = ArgumentParser(description="Compare simple baselines on an iTentformer-format dataset.")
    parser.add_argument("--data_path", default="dataset/ct_dma/ct_dma_itentformer_all.pkl")
    parser.add_argument("--window_stride", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    data = pd.read_pickle(data_path)
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    train_idx, test_idx = next(kfold.split(data))
    train_tracks = [data[i] for i in train_idx]
    test_tracks = [data[i] for i in test_idx]
    _, test_tracks, mean_values, std_values = standardize_like_training(train_tracks, test_tracks)
    windows = make_windows(test_tracks, args.window_stride)
    ade, fde, rmse_cog, rmse_sog = eval_constant_baseline(windows, mean_values, std_values)
    print(f"windows: {len(windows)}")
    print(f"constant baseline ADE: {ade:.5f} nmi ({ade * 1852:.2f} m)")
    print(f"constant baseline FDE: {fde:.5f} nmi ({fde * 1852:.2f} m)")
    print(f"constant baseline RMSE_COG: {rmse_cog:.5f} deg")
    print(f"constant baseline RMSE_SOG: {rmse_sog:.5f} kn")
    ade, fde, rmse_cog, rmse_sog = eval_linear_baseline(windows, mean_values, std_values)
    print(f"linear baseline ADE: {ade:.5f} nmi ({ade * 1852:.2f} m)")
    print(f"linear baseline FDE: {fde:.5f} nmi ({fde * 1852:.2f} m)")
    print(f"linear baseline RMSE_COG: {rmse_cog:.5f} deg")
    print(f"linear baseline RMSE_SOG: {rmse_sog:.5f} kn")


if __name__ == "__main__":
    main()
