from argparse import ArgumentParser, BooleanOptionalAction
from collections import Counter, defaultdict
from pathlib import Path
import json
import sys

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.umath", np.core.umath)
except AttributeError:
    pass

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


LON_COL = 3
LAT_COL = 4
SOG_COL = 5
COG_COL = 2
HIGH_CONTRAST_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
    "#8B5CF6",  # violet
    "#A16207",  # brown
    "#DC2626",  # red
    "#0F766E",  # teal
]


def as_jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=as_jsonable), encoding="utf-8")


def counter_to_int_dict(counter):
    return {int(key): int(value) for key, value in counter.items()}


def parse_force_route_k(value):
    result = {}
    if not value:
        return result
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --force_route_k item {item!r}. Expected ROUTE=K, e.g. OA=3,OC=4.")
        route, raw_k = item.split("=", 1)
        route = route.strip()
        if not route:
            raise ValueError(f"Invalid --force_route_k item {item!r}: empty route.")
        k = int(raw_k.strip())
        if k < 1:
            raise ValueError(f"Invalid --force_route_k item {item!r}: K must be >= 1.")
        result[route] = k
    return result


def parse_route_feature_windows(value):
    result = {}
    if not value:
        return result
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise ValueError(
                f"Invalid --route_feature_windows item {item!r}. "
                "Expected ROUTE=START:END, e.g. OA=0.38:0.72."
            )
        route, raw_window = item.split("=", 1)
        start_text, end_text = raw_window.split(":", 1)
        route = route.strip()
        start = float(start_text.strip())
        end = float(end_text.strip())
        if not route:
            raise ValueError(f"Invalid --route_feature_windows item {item!r}: empty route.")
        if not 0.0 <= start < end <= 1.0:
            raise ValueError(
                f"Invalid --route_feature_windows item {item!r}: "
                "START and END must satisfy 0 <= START < END <= 1."
            )
        result[route] = (start, end)
    return result


def load_route_records(labels_path, expected_count):
    with labels_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if len(raw) != expected_count:
        raise ValueError(f"Label count {len(raw)} does not match track count {expected_count}.")

    records = []
    routes = []
    for idx, item in enumerate(raw):
        if isinstance(item, dict):
            record = dict(item)
            route = record.get("route")
        else:
            route = str(item)
            record = {"index": idx, "route": route}
        if route is None:
            raise ValueError(f"Missing route in label item {idx}.")
        record.setdefault("index", idx)
        records.append(record)
        routes.append(str(route))
    return records, routes


def select_track_segment(track, feature_segment):
    if len(track) < 2:
        return track
    mid = max(2, len(track) // 2)
    if feature_segment == "history":
        return track[:mid]
    if feature_segment == "future":
        return track[mid - 1 :]
    return track


def interp_angle_deg(values, positions):
    radians = np.unwrap(np.deg2rad(values.astype(float)))
    interpolated = np.interp(positions, np.arange(len(values)), radians)
    return np.rad2deg(interpolated) % 360.0


def resample_track(track, resample_points, feature_segment):
    segment = select_track_segment(track, feature_segment)
    if len(segment) < 2:
        raise ValueError("Track segment must contain at least 2 points.")

    positions = np.linspace(0, len(segment) - 1, resample_points)
    base = np.arange(len(segment))
    lon = np.interp(positions, base, segment[:, LON_COL].astype(float))
    lat = np.interp(positions, base, segment[:, LAT_COL].astype(float))
    sog = np.interp(positions, base, segment[:, SOG_COL].astype(float))
    cog = interp_angle_deg(segment[:, COG_COL].astype(float), positions)
    return np.stack([lon, lat, sog, cog], axis=1)


def lonlat_to_local_m(lon, lat, lon_ref, lat_ref):
    x = (lon - lon_ref) * 111320.0 * np.cos(np.deg2rad(lat_ref))
    y = (lat - lat_ref) * 110540.0
    return x, y


def build_feature(path, lon_ref, lat_ref, feature_mode, include_motion):
    lon = path[:, 0]
    lat = path[:, 1]
    sog = path[:, 2]
    cog = path[:, 3]
    x, y = lonlat_to_local_m(lon, lat, lon_ref, lat_ref)

    chunks = []
    if feature_mode in {"absolute", "combined"}:
        chunks.append(np.stack([x, y], axis=1).reshape(-1) / 1000.0)
    if feature_mode in {"relative", "combined"}:
        rel = np.stack([x - x[0], y - y[0]], axis=1).reshape(-1) / 1000.0
        chunks.append(rel)
    if include_motion:
        chunks.append((sog / 30.0).reshape(-1))
        chunks.append(np.sin(np.deg2rad(cog)).reshape(-1))
        chunks.append(np.cos(np.deg2rad(cog)).reshape(-1))
    return np.concatenate(chunks, axis=0)


def select_branch_focus_indices(paths, lon_ref, lat_ref, args):
    if not args.use_branch_features:
        return [], [], []

    lon = paths[:, :, 0]
    lat = paths[:, :, 1]
    x, y = lonlat_to_local_m(lon, lat, lon_ref, lat_ref)
    xy_km = np.stack([x, y], axis=2) / 1000.0
    spread = np.sqrt(np.var(xy_km[:, :, 0], axis=0) + np.var(xy_km[:, :, 1], axis=0))

    n_steps = paths.shape[1]
    start = max(0, int(round((args.branch_exclude_ends_ratio or 0.0) * n_steps)))
    end = min(n_steps, int(round((1.0 - (args.branch_exclude_ends_ratio or 0.0)) * n_steps)))
    if end <= start:
        start, end = 0, n_steps

    candidates = np.argsort(spread[start:end])[::-1] + start
    selected = []
    min_gap = max(1, int(args.branch_focus_min_gap))
    for idx in candidates:
        idx = int(idx)
        if all(abs(idx - prev) >= min_gap for prev in selected):
            selected.append(idx)
        if len(selected) >= args.branch_focus_points:
            break
    selected.sort()
    progress = [idx / max(n_steps - 1, 1) for idx in selected]
    selected_spread = [float(spread[idx]) for idx in selected]
    return selected, progress, selected_spread


def feature_window_to_indices(window, n_steps):
    if window is None:
        return None
    start_ratio, end_ratio = window
    start = int(np.floor(start_ratio * (n_steps - 1)))
    end = int(np.ceil(end_ratio * (n_steps - 1))) + 1
    start = max(0, min(start, n_steps - 2))
    end = max(start + 2, min(end, n_steps))
    return start, end


def build_route_feature_matrix(paths, lon_ref, lat_ref, args, route_feature_window=None):
    base_features = [
        build_feature(path, lon_ref, lat_ref, args.feature_mode, args.include_motion)
        for path in paths
    ]
    n_steps = paths.shape[1]
    window_indices = feature_window_to_indices(route_feature_window, n_steps)

    focus_indices, focus_progress, focus_spread = select_branch_focus_indices(paths, lon_ref, lat_ref, args)
    if window_indices is None and (not args.use_branch_features or not focus_indices):
        return np.stack(base_features), {
            "use_branch_features": False,
            "branch_focus_indices": [],
            "branch_focus_progress": [],
            "branch_focus_spread_km": [],
            "route_feature_window": None,
            "route_feature_window_indices": None,
            "route_window_feature_weight": None,
        }

    lon = paths[:, :, 0]
    lat = paths[:, :, 1]
    x, y = lonlat_to_local_m(lon, lat, lon_ref, lat_ref)
    xy_km = np.stack([x, y], axis=2) / 1000.0

    rows = []
    for track_idx, base in enumerate(base_features):
        extra_chunks = []

        if window_indices is not None:
            start, end = window_indices
            window_path = paths[track_idx, start:end, :]
            window_feature = build_feature(
                window_path,
                lon_ref,
                lat_ref,
                args.feature_mode,
                args.include_motion,
            )
            window_xy = xy_km[track_idx, start:end, :]
            route_mean_xy = np.mean(xy_km[:, start:end, :], axis=0)
            window_centered = (window_xy - route_mean_xy).reshape(-1)
            extra_chunks.append(np.concatenate([window_feature, window_centered], axis=0) * args.route_window_feature_weight)

        if args.use_branch_features and focus_indices:
            branch_chunks = []
            for focus_idx in focus_indices:
                start = max(0, focus_idx - args.branch_focus_window)
                end = min(n_steps, focus_idx + args.branch_focus_window + 1)
                local_xy = xy_km[track_idx, start:end, :]
                route_mean_xy = np.mean(xy_km[:, start:end, :], axis=0)
                branch_chunks.append(local_xy.reshape(-1))
                branch_chunks.append((local_xy - route_mean_xy).reshape(-1))

                left = max(0, focus_idx - 1)
                right = min(n_steps - 1, focus_idx + 1)
                direction = xy_km[track_idx, right, :] - xy_km[track_idx, left, :]
                branch_chunks.append(direction)
            extra_chunks.append(np.concatenate(branch_chunks, axis=0) * args.branch_feature_weight)

        rows.append(np.concatenate([base] + extra_chunks, axis=0))

    return np.stack(rows), {
        "use_branch_features": bool(args.use_branch_features and focus_indices),
        "branch_focus_indices": [int(item) for item in focus_indices],
        "branch_focus_progress": [float(item) for item in focus_progress],
        "branch_focus_spread_km": [float(item) for item in focus_spread],
        "branch_feature_weight": float(args.branch_feature_weight),
        "branch_focus_window": int(args.branch_focus_window),
        "route_feature_window": None if route_feature_window is None else [float(route_feature_window[0]), float(route_feature_window[1])],
        "route_feature_window_indices": None if window_indices is None else [int(window_indices[0]), int(window_indices[1])],
        "route_window_feature_weight": None if window_indices is None else float(args.route_window_feature_weight),
    }


def fit_route_clusters(features, args, forced_k=None):
    n_tracks = len(features)
    if forced_k == 1:
        labels = np.zeros(n_tracks, dtype=int)
        centers = np.mean(features, axis=0, keepdims=True)
        return labels, centers, {"selected_k": 1, "reason": "forced_single_cluster", "candidates": []}

    if n_tracks < max(2, args.min_subroute_size * 2):
        labels = np.zeros(n_tracks, dtype=int)
        centers = np.mean(features, axis=0, keepdims=True)
        return labels, centers, {"selected_k": 1, "reason": "too_few_tracks", "candidates": []}

    max_k = min(args.max_subroutes_per_route, n_tracks // args.min_subroute_size)
    if max_k < 2:
        labels = np.zeros(n_tracks, dtype=int)
        centers = np.mean(features, axis=0, keepdims=True)
        return labels, centers, {"selected_k": 1, "reason": "min_subroute_size", "candidates": []}

    best = None
    accepted = []
    candidates = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
        labels = model.fit_predict(features)
        counts = Counter(labels)
        if min(counts.values()) < args.min_subroute_size:
            candidates.append({"k": k, "accepted": False, "reason": "small_cluster", "counts": counter_to_int_dict(counts)})
            continue
        sil = float(silhouette_score(features, labels))
        adjusted = sil - args.complexity_penalty * k
        candidate = {
            "k": k,
            "accepted": True,
            "silhouette": sil,
            "adjusted_score": adjusted,
            "counts": counter_to_int_dict(counts),
        }
        candidates.append(candidate)
        accepted.append({**candidate, "labels": labels, "centers": model.cluster_centers_})
        if best is None or adjusted > best["adjusted_score"]:
            best = {**candidate, "labels": labels, "centers": model.cluster_centers_}

    if forced_k is not None:
        for item in accepted:
            if int(item["k"]) == int(forced_k):
                return item["labels"], item["centers"], {
                    "selected_k": int(item["k"]),
                    "reason": "selected_by_forced_route_k",
                    "silhouette": float(item["silhouette"]),
                    "adjusted_score": float(item["adjusted_score"]),
                    "candidates": candidates,
                }
        matching = [item for item in candidates if int(item.get("k", -1)) == int(forced_k)]
        detail = matching[0] if matching else {"reason": "not_evaluated"}
        raise ValueError(f"Forced k={forced_k} is not valid for this route: {detail}")

    if best is None or best["silhouette"] < args.min_silhouette:
        labels = np.zeros(n_tracks, dtype=int)
        centers = np.mean(features, axis=0, keepdims=True)
        reason = "low_silhouette" if best is not None else "no_valid_candidate"
        return labels, centers, {"selected_k": 1, "reason": reason, "candidates": candidates}

    selected = best
    selection_reason = "selected_by_silhouette"
    if args.prefer_local_branches and accepted:
        branch_candidates = [
            item
            for item in accepted
            if item["k"] > selected["k"]
            and item["silhouette"] >= args.min_silhouette
            and item["silhouette"] >= selected["silhouette"] - args.branch_silhouette_tolerance
        ]
        if branch_candidates:
            selected = max(branch_candidates, key=lambda item: (item["k"], item["adjusted_score"]))
            selection_reason = "selected_by_local_branch_preference"

    return selected["labels"], selected["centers"], {
        "selected_k": int(selected["k"]),
        "reason": selection_reason,
        "silhouette": float(selected["silhouette"]),
        "adjusted_score": float(selected["adjusted_score"]),
        "candidates": candidates,
    }


def assignment_confidence(features, centers, labels):
    if len(centers) <= 1:
        return np.ones(len(features), dtype=float)
    distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
    sorted_distances = np.sort(distances, axis=1)
    margin = (sorted_distances[:, 1] - sorted_distances[:, 0]) / (sorted_distances[:, 1] + 1e-8)
    return np.clip(margin, 0.0, 1.0)


def circular_mean_deg(values):
    radians = np.deg2rad(values)
    mean_angle = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    return float(np.rad2deg(mean_angle) % 360.0)


def cluster_prototype(paths, global_indices, local_indices, features, center):
    cluster_paths = paths[local_indices]
    mean_lon = np.mean(cluster_paths[:, :, 0], axis=0)
    mean_lat = np.mean(cluster_paths[:, :, 1], axis=0)
    mean_sog = np.mean(cluster_paths[:, :, 2], axis=0)
    mean_cog = np.array([circular_mean_deg(cluster_paths[:, step, 3]) for step in range(cluster_paths.shape[1])])
    prototype = np.stack([mean_lon, mean_lat, mean_sog, mean_cog], axis=1)

    distances = np.linalg.norm(features[local_indices] - center[None, :], axis=1)
    medoid_local = int(local_indices[int(np.argmin(distances))])
    medoid_global = int(global_indices[medoid_local])
    return prototype, medoid_global


def subroute_sort_key(prototype):
    mid = len(prototype) // 2
    end = prototype[-1]
    middle = prototype[mid]
    return (float(end[0]), float(end[1]), float(middle[0]), float(middle[1]))


def discover_subroutes(tracks, label_records, routes, args):
    route_to_indices = defaultdict(list)
    for idx, route in enumerate(routes):
        route_to_indices[route].append(idx)
    forced_route_k = parse_force_route_k(args.force_route_k)
    route_feature_windows = parse_route_feature_windows(args.route_feature_windows)

    output_records = [dict(item) for item in label_records]
    prototypes = {
        "data_path": args.data_path,
        "labels_path": args.labels_path,
        "feature_segment": args.feature_segment,
        "feature_mode": args.feature_mode,
        "include_motion": args.include_motion,
        "use_branch_features": args.use_branch_features,
        "resample_points": args.resample_points,
        "routes": {},
    }
    report = {
        "data_path": args.data_path,
        "labels_path": args.labels_path,
        "route_counts": dict(Counter(routes)),
        "subroute_counts": {},
        "routes": {},
        "parameters": {
            "feature_segment": args.feature_segment,
            "feature_mode": args.feature_mode,
            "include_motion": args.include_motion,
            "use_branch_features": args.use_branch_features,
            "branch_feature_weight": args.branch_feature_weight,
            "branch_focus_points": args.branch_focus_points,
            "branch_focus_window": args.branch_focus_window,
            "branch_focus_min_gap": args.branch_focus_min_gap,
            "branch_exclude_ends_ratio": args.branch_exclude_ends_ratio,
            "prefer_local_branches": args.prefer_local_branches,
            "branch_silhouette_tolerance": args.branch_silhouette_tolerance,
            "force_route_k": forced_route_k,
            "route_feature_windows": {
                key: [float(value[0]), float(value[1])]
                for key, value in route_feature_windows.items()
            },
            "route_window_feature_weight": args.route_window_feature_weight,
            "resample_points": args.resample_points,
            "min_subroute_size": args.min_subroute_size,
            "max_subroutes_per_route": args.max_subroutes_per_route,
            "min_silhouette": args.min_silhouette,
            "complexity_penalty": args.complexity_penalty,
            "seed": args.seed,
        },
    }

    all_subroute_labels = []
    for route in sorted(route_to_indices):
        global_indices = np.asarray(route_to_indices[route], dtype=int)
        route_tracks = [tracks[idx] for idx in global_indices]
        paths = np.stack([resample_track(track, args.resample_points, args.feature_segment) for track in route_tracks])

        lon_ref = float(np.mean(paths[:, :, 0]))
        lat_ref = float(np.mean(paths[:, :, 1]))
        raw_features, feature_info = build_route_feature_matrix(
            paths,
            lon_ref,
            lat_ref,
            args,
            route_feature_window=route_feature_windows.get(route),
        )
        scaled_features = StandardScaler().fit_transform(raw_features)
        labels, centers, selection = fit_route_clusters(scaled_features, args, forced_k=forced_route_k.get(route))
        confidence = assignment_confidence(scaled_features, centers, labels)

        cluster_ids = sorted(set(int(item) for item in labels))
        cluster_infos = []
        for cluster_id in cluster_ids:
            local_indices = np.flatnonzero(labels == cluster_id)
            prototype, medoid_global = cluster_prototype(
                paths,
                global_indices,
                local_indices,
                scaled_features,
                centers[cluster_id],
            )
            cluster_infos.append(
                {
                    "cluster_id": int(cluster_id),
                    "local_indices": local_indices,
                    "prototype": prototype,
                    "medoid_global_index": medoid_global,
                    "count": int(len(local_indices)),
                    "sort_key": subroute_sort_key(prototype),
                }
            )

        cluster_infos.sort(key=lambda item: item["sort_key"])
        cluster_to_name = {}
        for rank, info in enumerate(cluster_infos):
            cluster_to_name[info["cluster_id"]] = f"{route}_S{rank:02d}"

        route_subroutes = []
        for info in cluster_infos:
            subroute = cluster_to_name[info["cluster_id"]]
            proto = info["prototype"]
            route_subroutes.append(
                {
                    "subroute": subroute,
                    "parent_route": route,
                    "count": info["count"],
                    "medoid_index": info["medoid_global_index"],
                    "prototype_lon_lat": proto[:, :2].round(7).tolist(),
                    "prototype_sog": proto[:, 2].round(4).tolist(),
                    "prototype_cog": proto[:, 3].round(4).tolist(),
                }
            )

        for local_idx, global_idx in enumerate(global_indices):
            cluster_id = int(labels[local_idx])
            subroute = cluster_to_name[cluster_id]
            record = output_records[int(global_idx)]
            record["parent_route"] = route
            record["subroute"] = subroute
            record["subroute_cluster"] = cluster_id
            record["subroute_confidence"] = float(confidence[local_idx])
            all_subroute_labels.append(subroute)

        prototypes["routes"][route] = {
            "track_count": int(len(global_indices)),
            "selected_k": int(selection["selected_k"]),
            "feature_info": feature_info,
            "selection": selection,
            "subroutes": route_subroutes,
        }
        report["routes"][route] = {
            "track_count": int(len(global_indices)),
            "selected_k": int(selection["selected_k"]),
            "feature_info": feature_info,
            "selection": selection,
            "subroute_counts": {item["subroute"]: item["count"] for item in route_subroutes},
        }

    report["subroute_counts"] = dict(Counter(all_subroute_labels))
    return output_records, prototypes, report


def color_for(index):
    return HIGH_CONTRAST_COLORS[index % len(HIGH_CONTRAST_COLORS)]


def sampled(items, max_count, seed):
    if max_count <= 0 or len(items) <= max_count:
        return list(items)
    rng = np.random.default_rng(seed)
    return sorted(int(item) for item in rng.choice(np.asarray(items), size=max_count, replace=False))


def finish_axes(ax, title):
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")


def padded_bounds(points, pad_ratio=0.06):
    x_min = float(np.min(points[:, LON_COL]))
    x_max = float(np.max(points[:, LON_COL]))
    y_min = float(np.min(points[:, LAT_COL]))
    y_max = float(np.max(points[:, LAT_COL]))
    x_pad = max((x_max - x_min) * pad_ratio, 0.02)
    y_pad = max((y_max - y_min) * pad_ratio, 0.02)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_prototype(ax, arr, color, label=None, linewidth=3.0):
    line = ax.plot(
        arr[:, 0],
        arr[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=0.98,
        label=label,
        solid_capstyle="round",
        zorder=5,
    )[0]
    line.set_path_effects([pe.Stroke(linewidth=linewidth + 2.2, foreground="white"), pe.Normal()])
    ax.scatter(arr[0, 0], arr[0, 1], s=22, marker="o", color=color, edgecolor="white", linewidth=0.8, zorder=6)
    ax.scatter(arr[-1, 0], arr[-1, 1], s=28, marker="X", color=color, edgecolor="white", linewidth=0.8, zorder=6)
    return line


def plot_focus_markers(ax, arr, focus_indices, color):
    for focus_idx in focus_indices:
        if 0 <= focus_idx < len(arr):
            ax.scatter(
                arr[focus_idx, 0],
                arr[focus_idx, 1],
                s=34,
                marker="D",
                color=color,
                edgecolor="black",
                linewidth=0.45,
                zorder=7,
            )


def draw_route_background(ax, tracks, indices, max_count, seed):
    for idx in sampled(indices, max_count, seed):
        track = tracks[idx]
        ax.plot(track[:, LON_COL], track[:, LAT_COL], color="#9ca3af", linewidth=0.42, alpha=0.16, zorder=1)


def plot_subroutes(tracks, records, prototypes, output_dir, prefix, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    subroute_to_indices = defaultdict(list)
    route_to_subroutes = defaultdict(set)
    for idx, record in enumerate(records):
        subroute = record["subroute"]
        subroute_to_indices[subroute].append(idx)
        route_to_subroutes[record["parent_route"]].add(subroute)

    subroute_order = sorted(subroute_to_indices)
    subroute_color = {subroute: color_for(i) for i, subroute in enumerate(subroute_order)}

    fig, ax = plt.subplots(figsize=(12.6, 9.2), constrained_layout=True)
    for route in sorted(route_to_subroutes):
        route_indices = []
        for subroute in sorted(route_to_subroutes[route]):
            route_indices.extend(subroute_to_indices[subroute])
        draw_route_background(ax, tracks, route_indices, args.max_plot_tracks_per_subroute, args.seed)

    for subroute in subroute_order:
        color = subroute_color[subroute]
        for idx in sampled(subroute_to_indices[subroute], args.max_plot_tracks_per_subroute, args.seed):
            track = tracks[idx]
            ax.plot(track[:, LON_COL], track[:, LAT_COL], color=color, linewidth=0.58, alpha=0.20, zorder=2)
        ax.plot([], [], color=color, linewidth=3.0, label=f"{subroute} ({len(subroute_to_indices[subroute])})")

    for route in sorted(route_to_subroutes):
        focus_indices = prototypes["routes"][route].get("feature_info", {}).get("branch_focus_indices", [])
        for proto in prototypes["routes"][route]["subroutes"]:
            color = subroute_color[proto["subroute"]]
            arr = np.asarray(proto["prototype_lon_lat"], dtype=float)
            plot_prototype(ax, arr, color, linewidth=2.6)
            plot_focus_markers(ax, arr, focus_indices, color)

    finish_axes(ax, "Auto-discovered subroutes - high contrast overview")
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)
    overlay_path = output_dir / f"{prefix}_subroute_overlay.png"
    fig.savefig(overlay_path, dpi=args.dpi)
    plt.close(fig)

    route_order = sorted(route_to_subroutes)
    cols = min(2, len(route_order))
    rows = int(np.ceil(len(route_order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.2 * rows), squeeze=False, constrained_layout=True)

    for panel_idx, route in enumerate(route_order):
        ax = axes[panel_idx // cols][panel_idx % cols]
        route_indices = []
        for subroute in sorted(route_to_subroutes[route]):
            route_indices.extend(subroute_to_indices[subroute])
        route_points = np.concatenate([tracks[idx] for idx in route_indices], axis=0)
        xlim, ylim = padded_bounds(route_points)
        draw_route_background(ax, tracks, route_indices, args.max_panel_tracks_per_subroute, args.seed)
        focus_indices = prototypes["routes"][route].get("feature_info", {}).get("branch_focus_indices", [])
        focus_progress = prototypes["routes"][route].get("feature_info", {}).get("branch_focus_progress", [])

        for subroute in sorted(route_to_subroutes[route]):
            color = subroute_color[subroute]
            for idx in sampled(subroute_to_indices[subroute], args.max_panel_tracks_per_subroute, args.seed):
                track = tracks[idx]
                ax.plot(track[:, LON_COL], track[:, LAT_COL], color=color, linewidth=0.62, alpha=0.24, zorder=2)
            for proto in prototypes["routes"][route]["subroutes"]:
                if proto["subroute"] != subroute:
                    continue
                arr = np.asarray(proto["prototype_lon_lat"], dtype=float)
                plot_prototype(ax, arr, color, label=f"{subroute} ({len(subroute_to_indices[subroute])})")
                plot_focus_markers(ax, arr, focus_indices, color)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        focus_text = ""
        if focus_progress:
            focus_text = " | focus " + ",".join(f"{item:.2f}" for item in focus_progress)
        finish_axes(ax, f"{route}{focus_text}")
        ax.legend(loc="best", fontsize=8, frameon=True)

    for panel_idx in range(len(route_order), rows * cols):
        axes[panel_idx // cols][panel_idx % cols].axis("off")

    panels_path = output_dir / f"{prefix}_subroute_panels.png"
    fig.savefig(panels_path, dpi=args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.4 * rows), squeeze=False, constrained_layout=True)
    for panel_idx, route in enumerate(route_order):
        ax = axes[panel_idx // cols][panel_idx % cols]
        feature_info = prototypes["routes"][route].get("feature_info", {})
        focus_indices = feature_info.get("branch_focus_indices", [])
        focus_spread = feature_info.get("branch_focus_spread_km", [])
        if focus_indices and focus_spread:
            ax.bar([str(item) for item in focus_indices], focus_spread, color="#2563eb", alpha=0.82)
            ax.set_ylabel("Spread (km)")
            ax.set_xlabel("Resampled point index")
            ax.set_title(f"{route} branch-focus spread")
            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
        else:
            ax.text(0.5, 0.5, "No branch focus", ha="center", va="center")
            ax.set_title(f"{route} branch-focus spread")
            ax.axis("off")

    for panel_idx in range(len(route_order), rows * cols):
        axes[panel_idx // cols][panel_idx % cols].axis("off")

    diagnostics_path = output_dir / f"{prefix}_branch_focus_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=args.dpi)
    plt.close(fig)
    return overlay_path, panels_path, diagnostics_path


def main():
    parser = ArgumentParser(description="Discover fine-grained subroutes inside coarse DMA route classes.")
    parser.add_argument("--data_path", default="dataset/dma_raw_2023_06_07/dma_itentformer_ti_4class_revnorm_lasthit.pkl")
    parser.add_argument(
        "--labels_path",
        default="dataset/dma_raw_2023_06_07/dma_route_labels_ti_4class_revnorm_lasthit.json",
    )
    parser.add_argument("--output_dir", default="dataset/dma_raw_2023_06_07")
    parser.add_argument("--plot_dir", default="results/dma_subroutes")
    parser.add_argument("--prefix", default="")

    parser.add_argument("--feature_segment", choices=["full", "history", "future"], default="full")
    parser.add_argument("--feature_mode", choices=["absolute", "relative", "combined"], default="combined")
    parser.add_argument("--include_motion", action=BooleanOptionalAction, default=True)
    parser.add_argument("--use_branch_features", action=BooleanOptionalAction, default=False)
    parser.add_argument("--branch_feature_weight", type=float, default=2.5)
    parser.add_argument("--branch_focus_points", type=int, default=3)
    parser.add_argument("--branch_focus_window", type=int, default=2)
    parser.add_argument("--branch_focus_min_gap", type=int, default=4)
    parser.add_argument("--branch_exclude_ends_ratio", type=float, default=0.08)
    parser.add_argument("--prefer_local_branches", action=BooleanOptionalAction, default=True)
    parser.add_argument("--branch_silhouette_tolerance", type=float, default=0.03)
    parser.add_argument(
        "--force_route_k",
        default="",
        help="Comma-separated route-specific cluster counts, e.g. OA=3,OC=4.",
    )
    parser.add_argument(
        "--route_feature_windows",
        default="",
        help="Comma-separated route-specific progress windows added to full-route features, e.g. OA=0.38:0.72.",
    )
    parser.add_argument("--route_window_feature_weight", type=float, default=2.5)
    parser.add_argument("--resample_points", type=int, default=32)
    parser.add_argument("--min_subroute_size", type=int, default=40)
    parser.add_argument("--max_subroutes_per_route", type=int, default=6)
    parser.add_argument("--min_silhouette", type=float, default=0.10)
    parser.add_argument("--complexity_penalty", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--plot", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max_plot_tracks_per_subroute", type=int, default=260)
    parser.add_argument("--max_panel_tracks_per_subroute", type=int, default=180)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    labels_path = Path(args.labels_path)
    output_dir = Path(args.output_dir)
    prefix = args.prefix or data_path.stem.replace("dma_itentformer", "dma_subroutes")

    tracks = pd.read_pickle(data_path)
    label_records, routes = load_route_records(labels_path, len(tracks))

    output_records, prototypes, report = discover_subroutes(tracks, label_records, routes, args)

    labels_output = output_dir / f"{prefix}_labels.json"
    prototypes_output = output_dir / f"{prefix}_prototypes.json"
    report_output = output_dir / f"{prefix}_report.json"
    write_json(labels_output, output_records)
    write_json(prototypes_output, prototypes)
    write_json(report_output, report)

    plot_outputs = {}
    if args.plot:
        overlay_path, panels_path, diagnostics_path = plot_subroutes(
            tracks,
            output_records,
            prototypes,
            Path(args.plot_dir),
            prefix,
            args,
        )
        plot_outputs = {
            "overlay": str(overlay_path),
            "panels": str(panels_path),
            "branch_focus_diagnostics": str(diagnostics_path),
        }

    summary = {
        "labels": str(labels_output),
        "prototypes": str(prototypes_output),
        "report": str(report_output),
        "plots": plot_outputs,
        "route_counts": report["route_counts"],
        "subroute_counts": report["subroute_counts"],
        "selected_k": {route: item["selected_k"] for route, item in report["routes"].items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=as_jsonable))


if __name__ == "__main__":
    main()
