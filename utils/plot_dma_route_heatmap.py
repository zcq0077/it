from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


DEFAULT_ROUTE_GATES = {
    "O": "10.30,11.10,57.35,57.85",
    "TI": "11.65,12.10,56.45,56.85",
    "A": "12.10,12.75,55.90,56.35",
    "B1": "11.20,11.90,56.05,56.35",
    "B2": "10.95,11.65,56.30,56.65",
    "C": "10.35,11.20,55.65,56.15",
}

ROUTE_COLORS = {
    "OA": "#16a34a",
    "OB1": "#f97316",
    "OB2": "#2563eb",
    "OB": "#9333ea",
    "AO": "#65a30d",
    "B1O": "#ea580c",
    "B2O": "#1d4ed8",
    "BO": "#7e22ce",
    "OC": "#a21caf",
    "CO": "#86198f",
}


def parse_gate(value):
    lon_min, lon_max, lat_min, lat_max = [float(item) for item in value.split(",")]
    return lon_min, lon_max, lat_min, lat_max


def load_routes(labels_path, expected_count):
    with labels_path.open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if len(labels) != expected_count:
        raise ValueError(f"Label count {len(labels)} does not match track count {expected_count}.")
    return [item["route"] for item in labels]


def sampled_indices(route_indices, max_count, seed):
    if max_count <= 0 or len(route_indices) <= max_count:
        return route_indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(route_indices), size=max_count, replace=False)
    return sorted(int(item) for item in chosen)


def all_lon_lat(tracks, indices=None):
    if indices is None:
        indices = range(len(tracks))
    selected = [tracks[idx] for idx in indices]
    if not selected:
        return np.array([]), np.array([])
    arr = np.concatenate(selected, axis=0)
    return arr[:, 3], arr[:, 4]


def draw_density(ax, lon, lat, bins, cmap="magma"):
    if len(lon) == 0:
        return None
    return ax.hist2d(lon, lat, bins=bins, cmap=cmap, norm=LogNorm(), cmin=1)


def draw_gates(ax, gates):
    for name, gate in gates.items():
        lon_min, lon_max, lat_min, lat_max = gate
        rect = Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            fill=False,
            linewidth=1.2,
            edgecolor="#111827",
            linestyle="--",
            alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(
            lon_min,
            lat_max,
            name,
            fontsize=9,
            fontweight="bold",
            color="#111827",
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
        )


def finish_geo_axes(ax, title):
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")


def plot_overlay(tracks, routes, gates, args, output_path):
    route_counts = Counter(routes)
    lon, lat = all_lon_lat(tracks)

    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    image = draw_density(ax, lon, lat, bins=args.bins, cmap="Greys")

    route_to_indices = {}
    for idx, route in enumerate(routes):
        route_to_indices.setdefault(route, []).append(idx)

    for route in sorted(route_to_indices):
        indices = sampled_indices(route_to_indices[route], args.max_overlay_per_route, args.seed)
        color = ROUTE_COLORS.get(route, "#0f172a")
        for idx in indices:
            track = tracks[idx]
            ax.plot(track[:, 3], track[:, 4], color=color, linewidth=0.65, alpha=args.line_alpha)
        ax.plot([], [], color=color, linewidth=2.2, label=f"{route} ({route_counts[route]})")

    if args.draw_gates:
        draw_gates(ax, gates)

    finish_geo_axes(ax, f"{args.title}: density heatmap with route classes")
    ax.legend(loc="upper right", frameon=True)
    if image is not None:
        fig.colorbar(image[3], ax=ax, label="Point density")
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


def plot_panels(tracks, routes, gates, args, output_path):
    route_order = [
        route
        for route in ("OA", "OB1", "OB2", "OC", "OB", "AO", "B1O", "B2O", "CO", "BO")
        if route in routes
    ]
    route_order += [route for route in sorted(set(routes)) if route not in route_order]
    if not route_order:
        return

    cols = min(3, len(route_order))
    rows = int(np.ceil(len(route_order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), squeeze=False, constrained_layout=True)
    route_counts = Counter(routes)
    all_lon, all_lat = all_lon_lat(tracks)
    xlim = (float(np.min(all_lon)), float(np.max(all_lon)))
    ylim = (float(np.min(all_lat)), float(np.max(all_lat)))

    route_to_indices = {}
    for idx, route in enumerate(routes):
        route_to_indices.setdefault(route, []).append(idx)

    for panel_idx, route in enumerate(route_order):
        ax = axes[panel_idx // cols][panel_idx % cols]
        indices = route_to_indices[route]
        lon, lat = all_lon_lat(tracks, indices)
        draw_density(ax, lon, lat, bins=args.bins, cmap="viridis")
        color = ROUTE_COLORS.get(route, "#0f172a")
        for idx in sampled_indices(indices, args.max_panel_overlay_per_route, args.seed):
            track = tracks[idx]
            ax.plot(track[:, 3], track[:, 4], color=color, linewidth=0.75, alpha=0.38)
        if args.draw_gates:
            draw_gates(ax, gates)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        finish_geo_axes(ax, f"{route} ({route_counts[route]})")

    for panel_idx in range(len(route_order), rows * cols):
        axes[panel_idx // cols][panel_idx % cols].axis("off")

    fig.suptitle(f"{args.title}: route-specific heatmaps", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Plot DMA route heatmaps from iTentformer pkl and route labels.")
    parser.add_argument("--data_path", default="dataset/dma_raw_2023_07/dma_itentformer_ti_3class_revnorm.pkl")
    parser.add_argument("--labels_path", default="dataset/dma_raw_2023_07/dma_route_labels_ti_3class_revnorm.json")
    parser.add_argument("--output_dir", default="results/dma_route_heatmaps")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--title", default="DMA 2023-07 Ti routes")
    parser.add_argument("--bins", type=int, default=240)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_overlay_per_route", type=int, default=450)
    parser.add_argument("--max_panel_overlay_per_route", type=int, default=500)
    parser.add_argument("--line_alpha", type=float, default=0.18)
    parser.add_argument("--draw_gates", action="store_true")
    parser.add_argument("--gate_o", default=DEFAULT_ROUTE_GATES["O"])
    parser.add_argument("--gate_ti", default=DEFAULT_ROUTE_GATES["TI"])
    parser.add_argument("--gate_a", default=DEFAULT_ROUTE_GATES["A"])
    parser.add_argument("--gate_b1", default=DEFAULT_ROUTE_GATES["B1"])
    parser.add_argument("--gate_b2", default=DEFAULT_ROUTE_GATES["B2"])
    parser.add_argument("--gate_c", default=DEFAULT_ROUTE_GATES["C"])
    args = parser.parse_args()

    data_path = Path(args.data_path)
    labels_path = Path(args.labels_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or data_path.stem

    tracks = pd.read_pickle(data_path)
    routes = load_routes(labels_path, len(tracks))
    gates = {
        "O": parse_gate(args.gate_o),
        "TI": parse_gate(args.gate_ti),
        "A": parse_gate(args.gate_a),
        "B1": parse_gate(args.gate_b1),
        "B2": parse_gate(args.gate_b2),
        "C": parse_gate(args.gate_c),
    }

    overlay_path = output_dir / f"{prefix}_overlay_heatmap.png"
    panels_path = output_dir / f"{prefix}_class_heatmaps.png"
    plot_overlay(tracks, routes, gates, args, overlay_path)
    plot_panels(tracks, routes, gates, args, panels_path)

    print(json.dumps({"overlay": str(overlay_path), "panels": str(panels_path), "counts": Counter(routes)}, default=int))


if __name__ == "__main__":
    main()
