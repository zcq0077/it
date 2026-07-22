"""Build a robust two-branch OC subroute label set.

The generic subroute discovery utility resamples complete tracks by relative
progress.  For OC that makes start/end coverage dominate the clusters.  This
utility instead compares every OC track inside a geographic corridor traversed
by the complete class, so the labels represent the west/east branch geometry.
"""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json
import sys

import matplotlib
import numpy as np
import pandas as pd

try:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.umath", np.core.umath)
except AttributeError:
    pass

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    davies_bouldin_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COG_COL = 2
LON_COL = 3
LAT_COL = 4
SOG_COL = 5
MMSI_COL = 0
WEST_COLOR = "#0072B2"
EAST_COLOR = "#D55E00"
OUTLIER_COLOR = "#6B7280"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def load_labels(path, expected_count):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != expected_count:
        raise ValueError(
            f"Label count {len(rows)} does not match track count {expected_count}."
        )
    return [dict(row) for row in rows]


def is_oc(record):
    return str(record.get("parent_route", record.get("route", ""))) == "OC"


def interpolate_lon_at_lat(track, anchor_latitudes):
    track = np.asarray(track, dtype=float)
    order = np.argsort(track[:, LAT_COL])
    lat = track[order, LAT_COL]
    lon = track[order, LON_COL]

    unique_lat, inverse = np.unique(lat, return_inverse=True)
    unique_lon = np.zeros(len(unique_lat), dtype=float)
    counts = np.zeros(len(unique_lat), dtype=float)
    np.add.at(unique_lon, inverse, lon)
    np.add.at(counts, inverse, 1.0)
    unique_lon /= np.maximum(counts, 1.0)
    return np.interp(anchor_latitudes, unique_lat, unique_lon)


def build_corridor_features(tracks, indices, anchor_latitudes):
    return np.stack(
        [interpolate_lon_at_lat(tracks[index], anchor_latitudes) for index in indices]
    )


def lateral_deviation_km(features, anchor_latitudes):
    median = np.median(features, axis=0)
    km_per_degree_lon = 111.32 * np.cos(np.deg2rad(anchor_latitudes))
    return np.max(np.abs(features - median) * km_per_degree_lon, axis=1)


def ordered_labels(raw_labels, features, branch_anchor_count):
    cluster_order = sorted(
        np.unique(raw_labels),
        key=lambda cluster: float(
            np.mean(features[raw_labels == cluster, :branch_anchor_count])
        ),
    )
    mapping = {int(cluster): rank for rank, cluster in enumerate(cluster_order)}
    return np.asarray([mapping[int(cluster)] for cluster in raw_labels]), mapping


def distance_margin(features, centers, assigned):
    distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
    sorted_distances = np.sort(distances, axis=1)
    margin = (sorted_distances[:, 1] - sorted_distances[:, 0]) / (
        sorted_distances[:, 1] + 1e-8
    )
    return np.clip(margin, 0.0, 1.0), distances[np.arange(len(features)), assigned]


def safe_auc(target, score):
    if len(np.unique(target)) < 2:
        return None
    return float(roc_auc_score(target, score))


def history_feature(history):
    history = np.asarray(history, dtype=float)
    lon = history[:, LON_COL]
    lat = history[:, LAT_COL]
    sog = history[:, SOG_COL]
    cog = np.deg2rad(history[:, COG_COL])
    return np.concatenate(
        [
            lon,
            lat,
            lon - lon[0],
            lat - lat[0],
            sog,
            np.sin(cog),
            np.cos(cog),
            np.diff(lon),
            np.diff(lat),
        ]
    )


def observability_report(
    tracks,
    records,
    indices,
    labels,
    history_points,
    future_points,
    windows_per_track,
):
    features = []
    targets = []
    groups = []
    last_latitudes = []
    sources = []

    for index, label in zip(indices, labels):
        track = np.asarray(tracks[int(index)], dtype=float)
        max_start = len(track) - history_points - future_points
        if max_start < 0:
            continue
        starts = np.unique(
            np.linspace(
                0,
                max_start,
                min(windows_per_track, max_start + 1),
            )
            .round()
            .astype(int)
        )
        for start in starts:
            history = track[start : start + history_points]
            features.append(history_feature(history))
            targets.append(int(label))
            groups.append(str(track[0, MMSI_COL]))
            last_latitudes.append(float(history[-1, LAT_COL]))
            sources.append(str(records[int(index)].get("source", "unknown")))

    features = np.asarray(features)
    targets = np.asarray(targets)
    groups = np.asarray(groups)
    last_latitudes = np.asarray(last_latitudes)
    sources = np.asarray(sources)
    probabilities = np.zeros(len(targets), dtype=float)

    n_splits = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    for train_indices, test_indices in splitter.split(features, targets, groups):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=0.2,
            ),
        )
        classifier.fit(features[train_indices], targets[train_indices])
        probabilities[test_indices] = classifier.predict_proba(features[test_indices])[
            :, 1
        ]

    predictions = (probabilities >= 0.5).astype(int)

    def metrics(mask):
        if int(np.sum(mask)) == 0:
            return None
        target = targets[mask]
        prediction = predictions[mask]
        score = probabilities[mask]
        return {
            "count": int(np.sum(mask)),
            "accuracy": float(accuracy_score(target, prediction)),
            "balanced_accuracy": float(
                balanced_accuracy_score(target, prediction)
            ),
            "macro_f1": float(f1_score(target, prediction, average="macro")),
            "roc_auc": safe_auc(target, score),
        }

    latitude_bins = {
        "north_early_lat_ge_57p0": last_latitudes >= 57.0,
        "approach_lat_56p6_to_57p0": (last_latitudes >= 56.6)
        & (last_latitudes < 57.0),
        "branch_lat_56p3_to_56p6": (last_latitudes >= 56.3)
        & (last_latitudes < 56.6),
        "south_lat_lt_56p3": last_latitudes < 56.3,
    }
    source_metrics = {
        source: metrics(sources == source) for source in sorted(np.unique(sources))
    }
    return {
        "method": "MMSI-grouped 5-fold logistic diagnostic using history only",
        "history_points": int(history_points),
        "future_points": int(future_points),
        "windows_per_track": int(windows_per_track),
        "overall": metrics(np.ones(len(targets), dtype=bool)),
        "by_last_observed_latitude": {
            name: metrics(mask) for name, mask in latitude_bins.items()
        },
        "by_source": source_metrics,
    }


def cluster_stability(features, reference_labels, k, seed, rounds):
    rng = np.random.default_rng(seed + 101)
    scores = []
    for round_index in range(rounds):
        sample = rng.choice(len(features), len(features), replace=True)
        model = KMeans(
            n_clusters=k,
            random_state=seed + round_index,
            n_init=20,
        ).fit(features[sample])
        scores.append(adjusted_rand_score(reference_labels, model.predict(features)))
    return {
        "rounds": int(rounds),
        "ari_median": float(np.median(scores)),
        "ari_p10": float(np.percentile(scores, 10)),
        "ari_min": float(np.min(scores)),
    }


def plot_result(
    path,
    tracks,
    indices,
    labels,
    outlier_mask,
    anchor_latitudes,
    prototypes,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = [WEST_COLOR, EAST_COLOR]
    fig, ax = plt.subplots(figsize=(8.8, 8.0), constrained_layout=True)
    ax.axhspan(
        anchor_latitudes[0],
        anchor_latitudes[-1],
        color="#E5E7EB",
        alpha=0.35,
        zorder=0,
        label="Common alignment corridor",
    )

    for local_index, global_index in enumerate(indices):
        track = np.asarray(tracks[int(global_index)], dtype=float)
        if outlier_mask[local_index]:
            color = OUTLIER_COLOR
            alpha = 0.8
            linewidth = 0.9
            linestyle = "--"
        else:
            color = colors[int(labels[local_index])]
            alpha = 0.14
            linewidth = 0.65
            linestyle = "-"
        ax.plot(
            track[:, LON_COL],
            track[:, LAT_COL],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=1,
        )

    for cluster, name in enumerate(("OC_S00 west branch", "OC_S01 east branch")):
        ax.plot(
            prototypes[cluster],
            anchor_latitudes,
            color=colors[cluster],
            linewidth=4.0,
            solid_capstyle="round",
            label=f"{name} (n={int(np.sum(labels == cluster))})",
            zorder=4,
        )

    ax.set_title("OC subroutes aligned on the common geographic corridor")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    parser = ArgumentParser(
        description="Refine OC into stable west/east branches on a common corridor."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--output_labels_path", required=True)
    parser.add_argument("--report_path", default="")
    parser.add_argument("--prototypes_path", default="")
    parser.add_argument("--plot_path", default="")
    parser.add_argument("--corridor_lat_min", type=float, default=56.2)
    parser.add_argument("--corridor_lat_max", type=float, default=57.3)
    parser.add_argument("--anchor_count", type=int, default=12)
    parser.add_argument("--branch_anchor_count", type=int, default=3)
    parser.add_argument("--outlier_threshold_km", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap_rounds", type=int, default=100)
    parser.add_argument("--history_points", type=int, default=13)
    parser.add_argument("--future_points", type=int, default=12)
    parser.add_argument("--windows_per_track", type=int, default=6)
    parser.add_argument("--skip_observability", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    labels_path = Path(args.labels_path)
    output_labels_path = Path(args.output_labels_path)
    base = output_labels_path.with_suffix("")
    report_path = Path(args.report_path) if args.report_path else Path(f"{base}_report.json")
    prototypes_path = (
        Path(args.prototypes_path)
        if args.prototypes_path
        else Path(f"{base}_prototypes.json")
    )
    plot_path = Path(args.plot_path) if args.plot_path else Path(f"{base}_diagnostic.png")

    tracks = pd.read_pickle(data_path)
    records = load_labels(labels_path, len(tracks))
    oc_indices = np.asarray(
        [index for index, record in enumerate(records) if is_oc(record)], dtype=int
    )
    if len(oc_indices) < 4:
        raise ValueError(f"Expected at least 4 OC tracks, found {len(oc_indices)}.")

    anchor_latitudes = np.linspace(
        args.corridor_lat_min,
        args.corridor_lat_max,
        args.anchor_count,
    )
    for index in oc_indices:
        track = np.asarray(tracks[int(index)], dtype=float)
        if np.min(track[:, LAT_COL]) > args.corridor_lat_min or np.max(
            track[:, LAT_COL]
        ) < args.corridor_lat_max:
            raise ValueError(
                f"OC track {index} does not span the requested common corridor."
            )

    features = build_corridor_features(tracks, oc_indices, anchor_latitudes)
    deviations = lateral_deviation_km(features, anchor_latitudes)
    inlier_mask = deviations < args.outlier_threshold_km
    if int(np.sum(inlier_mask)) < 4:
        raise ValueError("Too few OC inliers after corridor quality filtering.")

    model = KMeans(n_clusters=2, random_state=args.seed, n_init=100).fit(
        features[inlier_mask]
    )
    inlier_ordered, mapping = ordered_labels(
        model.labels_,
        features[inlier_mask],
        args.branch_anchor_count,
    )
    ordered_centers = np.stack(
        [model.cluster_centers_[cluster] for cluster, rank in sorted(mapping.items(), key=lambda item: item[1])]
    )
    raw_all_labels = model.predict(features)
    all_labels = np.asarray([mapping[int(label)] for label in raw_all_labels])
    all_labels[inlier_mask] = inlier_ordered
    confidence, assigned_distance = distance_margin(
        features,
        ordered_centers,
        all_labels,
    )
    confidence[~inlier_mask] = 0.0

    for local_index, global_index in enumerate(oc_indices):
        record = records[int(global_index)]
        record["source_subroute"] = record.get("subroute")
        record["parent_route"] = "OC"
        record["subroute"] = f"OC_S{int(all_labels[local_index]):02d}"
        record["subroute_assignment"] = "oc_common_corridor_k2_v1"
        record["subroute_confidence"] = float(confidence[local_index])
        record["oc_corridor_outlier"] = bool(not inlier_mask[local_index])
        record["oc_corridor_distance"] = float(assigned_distance[local_index])

    write_json(output_labels_path, records)

    inlier_features = features[inlier_mask]
    inlier_labels = all_labels[inlier_mask]
    source_counts = {}
    mmsi_counts = {}
    subroute_counts = {}
    for cluster in range(2):
        member_indices = oc_indices[all_labels == cluster]
        name = f"OC_S{cluster:02d}"
        subroute_counts[name] = int(len(member_indices))
        source_counts[name] = dict(
            sorted(Counter(str(records[int(index)].get("source", "unknown")) for index in member_indices).items())
        )
        mmsi_counts[name] = len(
            {
                str(np.asarray(tracks[int(index)])[0, MMSI_COL])
                for index in member_indices
            }
        )

    prototypes = np.stack(
        [np.median(features[all_labels == cluster], axis=0) for cluster in range(2)]
    )
    separation_km = np.abs(prototypes[1] - prototypes[0]) * 111.32 * np.cos(
        np.deg2rad(anchor_latitudes)
    )
    stability = cluster_stability(
        inlier_features,
        inlier_labels,
        2,
        args.seed,
        args.bootstrap_rounds,
    )
    observability = None
    if not args.skip_observability:
        observability = observability_report(
            tracks,
            records,
            oc_indices[inlier_mask],
            inlier_labels,
            args.history_points,
            args.future_points,
            args.windows_per_track,
        )

    report = {
        "name": "oc_common_corridor_k2_v1",
        "description": (
            "Two OC branches discovered after latitude alignment on a corridor "
            "shared by every OC track; complete-track start/end coverage is excluded "
            "from the clustering features."
        ),
        "data_path": str(data_path),
        "input_labels_path": str(labels_path),
        "output_labels_path": str(output_labels_path),
        "oc_track_count": int(len(oc_indices)),
        "oc_inlier_count": int(np.sum(inlier_mask)),
        "oc_outlier_count": int(np.sum(~inlier_mask)),
        "outlier_indices": [int(index) for index in oc_indices[~inlier_mask]],
        "subroute_counts": subroute_counts,
        "mmsi_counts": mmsi_counts,
        "source_counts": source_counts,
        "parameters": {
            "corridor_lat_min": float(args.corridor_lat_min),
            "corridor_lat_max": float(args.corridor_lat_max),
            "anchor_count": int(args.anchor_count),
            "branch_anchor_count": int(args.branch_anchor_count),
            "outlier_threshold_km": float(args.outlier_threshold_km),
            "seed": int(args.seed),
        },
        "quality": {
            "silhouette_inliers": float(
                silhouette_score(inlier_features, inlier_labels)
            ),
            "davies_bouldin_inliers": float(
                davies_bouldin_score(inlier_features, inlier_labels)
            ),
            "bootstrap_stability": stability,
            "prototype_separation_km_by_anchor": separation_km.tolist(),
        },
        "history_observability": observability,
        "all_subroute_counts": dict(
            sorted(Counter(str(record.get("subroute")) for record in records).items())
        ),
    }
    write_json(report_path, report)
    write_json(
        prototypes_path,
        {
            "name": "oc_common_corridor_k2_v1",
            "anchor_latitudes": anchor_latitudes,
            "subroutes": [
                {
                    "subroute": f"OC_S{cluster:02d}",
                    "branch": "west" if cluster == 0 else "east",
                    "count": subroute_counts[f"OC_S{cluster:02d}"],
                    "prototype_lon": prototypes[cluster],
                }
                for cluster in range(2)
            ],
        },
    )
    plot_result(
        plot_path,
        tracks,
        oc_indices,
        all_labels,
        ~inlier_mask,
        anchor_latitudes,
        prototypes,
    )

    print(
        json.dumps(
            {
                "labels": str(output_labels_path),
                "report": str(report_path),
                "prototypes": str(prototypes_path),
                "plot": str(plot_path),
                "subroute_counts": subroute_counts,
                "silhouette": report["quality"]["silhouette_inliers"],
                "bootstrap_ari_median": stability["ari_median"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
