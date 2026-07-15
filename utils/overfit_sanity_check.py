from argparse import ArgumentParser
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import iTentformer
from utils.Haversine_Loss import HaversineLoss
from utils.bohai_diff import window_slice


SRC_COLS = [2, 3, 4, 5]
DELTA_COLS = [8, 9, 10, 11]
IN_COLS = SRC_COLS + DELTA_COLS
INTENT_COLS = [2]
INPUT_LENGTH = 10
TARGET_LENGTH = 10


def standardize_tracks(data):
    flat = np.concatenate(data, axis=0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(flat[:, 2:-1])
    flat_scaled = np.concatenate((flat[:, :2], scaled, flat[:, -1:]), axis=-1)

    tracks = []
    start = 0
    for track in data:
        end = start + len(track)
        tracks.append(flat_scaled[start:end])
        start = end
    return tracks, scaler.mean_, scaler.scale_


def inverse_standardized(values, transform, mean):
    return values @ transform + mean[:4]


def get_window(data_path, sample_index):
    data = pd.read_pickle(data_path)
    tracks, mean, std = standardize_tracks(data)
    windows = []
    for track in tracks:
        sliced = window_slice(track, win_size=20, step=20)
        if sliced is not None:
            windows.extend(list(sliced))
    if not windows:
        raise ValueError("No 20-step windows found.")
    sample_index = sample_index % len(windows)
    return torch.tensor(windows[sample_index]).float(), mean, std, len(windows), sample_index


def compute_metrics(pred_real, target_real, haversine):
    dist = haversine(pred_real[:, :, 1:3].float(), target_real[:, :, 1:3].float()).reshape(-1, TARGET_LENGTH)
    ade = dist.mean()
    fde = dist[:, -1].mean()
    return ade, fde


def save_plot(history, target, pred, save_path, title):
    history_xy = history[:, [1, 2]]
    target_xy = np.vstack([history_xy[-1:], target[:, [1, 2]]])
    pred_xy = np.vstack([history_xy[-1:], pred[:, [1, 2]]])

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    ax.plot(history_xy[:, 0], history_xy[:, 1], "-o", color="#2563eb", linewidth=2.0, markersize=3.5, label="History")
    ax.plot(target_xy[:, 0], target_xy[:, 1], "-o", color="#16a34a", linewidth=2.0, markersize=3.5, label="Ground truth")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], "--o", color="#dc2626", linewidth=2.0, markersize=3.5, label="Prediction")
    ax.scatter(history_xy[0, 0], history_xy[0, 1], color="#1e40af", s=42, marker="s", label="Start")
    ax.scatter(history_xy[-1, 0], history_xy[-1, 1], color="#111827", s=46, marker="x", label="Predict from")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Overfit one trajectory window to sanity-check model capacity.")
    parser.add_argument("--data_path", default="dataset/example_bohai.pkl")
    parser.add_argument("--sample_index", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="results/overfit_sanity")
    parser.add_argument("--dropout", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    sample, mean, std, window_count, sample_index = get_window(args.data_path, args.sample_index)
    sample = sample.unsqueeze(0).to(device)
    transform = torch.tensor(np.diag(std[:4]), dtype=torch.float32, device=device)
    mean_tensor = torch.tensor(mean[:4], dtype=torch.float32, device=device)

    src = sample[:, :INPUT_LENGTH, IN_COLS]
    tgt = sample[:, INPUT_LENGTH:INPUT_LENGTH + TARGET_LENGTH, SRC_COLS]
    intent_tgt = sample[:, INPUT_LENGTH:INPUT_LENGTH + TARGET_LENGTH, INTENT_COLS]

    model = iTentformer(
        input_size_tcn=8,
        input_size=10,
        local_intent_size=1,
        output_size=4,
        concat_dim=40,
        input_length=INPUT_LENGTH,
        target_length=TARGET_LENGTH,
        num_channels=[32] * 2,
        kernel_size=3,
        d_model=128,
        dropout=args.dropout,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    haversine = HaversineLoss(min_hav=0.0).to(device)

    best_loss = float("inf")
    best_state = None
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        intent, pred = model(src, src)
        loss_traj = mse(pred, tgt)
        loss_intent = mse(intent, intent_tgt)
        loss = loss_traj + 0.1 * loss_intent
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if step == 1 or step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                _, pred_eval = model(src, src)
                pred_real = pred_eval @ transform + mean_tensor
                tgt_real = tgt @ transform + mean_tensor
                ade, fde = compute_metrics(pred_real, tgt_real, haversine)
            print(
                f"step {step:04d}, loss {loss.item():.8f}, "
                f"ADE {float(ade):.6f}nmi ({float(ade) * 1852:.2f}m), "
                f"FDE {float(fde):.6f}nmi ({float(fde) * 1852:.2f}m)"
            )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        _, pred = model(src, src)
        pred_real = (pred @ transform + mean_tensor).squeeze(0).detach().cpu().numpy()
        tgt_real = (tgt @ transform + mean_tensor).squeeze(0).detach().cpu().numpy()
        hist_real = (sample[:, :INPUT_LENGTH, SRC_COLS] @ transform + mean_tensor).squeeze(0).detach().cpu().numpy()
        ade, fde = compute_metrics(
            torch.tensor(pred_real).unsqueeze(0).to(device),
            torch.tensor(tgt_real).unsqueeze(0).to(device),
            haversine,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"overfit_sample_{sample_index:03d}.png"
    save_plot(
        hist_real,
        tgt_real,
        pred_real,
        plot_path,
        f"Overfit sanity sample {sample_index}: prediction vs ground truth",
    )

    print(
        f"Final best loss {best_loss:.8f}, sample {sample_index}/{window_count}, "
        f"ADE {float(ade):.6f}nmi ({float(ade) * 1852:.2f}m), "
        f"FDE {float(fde):.6f}nmi ({float(fde) * 1852:.2f}m)"
    )
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
