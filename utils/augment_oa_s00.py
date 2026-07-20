"""Add high-confidence OA_S00 tracks without redefining existing subroutes.

The classifier is reconstructed from the fixed 2023-06/07/08 OA labels using
the same trajectory features that produced the three OA clusters. New months
are assigned to those fixed centroids. Only conservative OA_S00 assignments
are appended, and an existing MMSI-grouped validation/test split is preserved.
"""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.umath", np.core.umath)
except AttributeError:
    pass

from discover_subroutes import (
    build_feature,
    feature_window_to_indices,
    lonlat_to_local_m,
    resample_track,
)


TARGET_ROUTE = "OA"
TARGET_SUBROUTE = "OA_S00"
SUBROUTE_ORDER = ("OA_S00", "OA_S01", "OA_S02")


def comma_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_alignment(tracks, *label_sets):
    for labels in label_sets:
        if len(labels) != len(tracks):
            raise ValueError(
                f"Track/label length mismatch: tracks={len(tracks)}, labels={len(labels)}."
            )


def build_feature_state(paths, window, window_weight):
    lon_ref = float(np.mean(paths[:, :, 0]))
    lat_ref = float(np.mean(paths[:, :, 1]))
    start, end = feature_window_to_indices(window, paths.shape[1])
    x, y = lonlat_to_local_m(
        paths[:, :, 0],
        paths[:, :, 1],
        lon_ref,
        lat_ref,
    )
    xy_km = np.stack([x, y], axis=2) / 1000.0
    return {
        "lon_ref": lon_ref,
        "lat_ref": lat_ref,
        "window": [float(window[0]), float(window[1])],
        "window_indices": [int(start), int(end)],
        "window_weight": float(window_weight),
        "route_mean_xy": np.mean(xy_km[:, start:end, :], axis=0),
    }


def fixed_feature_matrix(paths, state):
    lon_ref = state["lon_ref"]
    lat_ref = state["lat_ref"]
    start, end = state["window_indices"]
    route_mean_xy = state["route_mean_xy"]
    rows = []
    for path in paths:
        base = build_feature(path, lon_ref, lat_ref, "combined", True)
        window_path = path[start:end]
        window_feature = build_feature(
            window_path,
            lon_ref,
            lat_ref,
            "combined",
            True,
        )
        x, y = lonlat_to_local_m(
            window_path[:, 0],
            window_path[:, 1],
            lon_ref,
            lat_ref,
        )
        window_xy = np.stack([x, y], axis=1) / 1000.0
        window_centered = (window_xy - route_mean_xy).reshape(-1)
        local = np.concatenate([window_feature, window_centered], axis=0)
        rows.append(np.concatenate([base, local * state["window_weight"]], axis=0))
    return np.stack(rows)


def class_centroids(features, labels, class_names):
    centers = []
    for class_name in class_names:
        members = features[np.asarray(labels) == class_name]
        if len(members) == 0:
            raise ValueError(f"No samples available for {class_name}.")
        centers.append(np.mean(members, axis=0))
    return np.stack(centers)


def distance_predictions(features, centers, class_names):
    distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
    order = np.argsort(distances, axis=1)
    best = order[:, 0]
    best_distance = distances[np.arange(len(features)), best]
    second_distance = distances[np.arange(len(features)), order[:, 1]]
    margin = (second_distance - best_distance) / np.maximum(second_distance, 1e-8)
    predictions = np.asarray([class_names[index] for index in best], dtype=object)
    return predictions, best_distance, np.clip(margin, 0.0, 1.0), distances


def accuracy_summary(targets, predictions, class_names):
    targets = np.asarray(targets, dtype=object)
    predictions = np.asarray(predictions, dtype=object)
    result = {
        "accuracy": float(np.mean(targets == predictions)) if len(targets) else 0.0,
        "per_class_recall": {},
        "confusion": {},
    }
    for class_name in class_names:
        mask = targets == class_name
        result["per_class_recall"][class_name] = (
            float(np.mean(predictions[mask] == class_name)) if np.any(mask) else None
        )
        result["confusion"][class_name] = dict(
            Counter(str(item) for item in predictions[mask])
        )
    return result


def grouped_cross_validation(raw_features, labels, groups, class_names, folds):
    unique_groups = np.unique(groups)
    n_splits = min(int(folds), len(unique_groups))
    if n_splits < 2:
        return {"folds": 0, "reason": "too_few_mmsi"}

    predictions = np.full(len(labels), "", dtype=object)
    splitter = GroupKFold(n_splits=n_splits)
    for train_indices, valid_indices in splitter.split(raw_features, labels, groups):
        train_labels = np.asarray(labels, dtype=object)[train_indices]
        if any(not np.any(train_labels == name) for name in class_names):
            return {"folds": 0, "reason": "class_missing_in_training_fold"}
        scaler = StandardScaler().fit(raw_features[train_indices])
        train_scaled = scaler.transform(raw_features[train_indices])
        valid_scaled = scaler.transform(raw_features[valid_indices])
        centers = class_centroids(train_scaled, train_labels, class_names)
        predictions[valid_indices] = distance_predictions(
            valid_scaled,
            centers,
            class_names,
        )[0]
    return {
        "folds": int(n_splits),
        **accuracy_summary(labels, predictions, class_names),
    }


def load_candidate_sets(data_paths, label_paths):
    if len(data_paths) != len(label_paths):
        raise ValueError("Candidate data and route-label path counts must match.")
    candidates = []
    source_summaries = []
    for data_path, label_path in zip(data_paths, label_paths):
        tracks = list(pd.read_pickle(data_path))
        labels = read_json(label_path)
        validate_alignment(tracks, labels)
        oa_count = 0
        for local_index, (track, label) in enumerate(zip(tracks, labels)):
            if str(label.get("route")) != TARGET_ROUTE:
                continue
            oa_count += 1
            candidates.append(
                {
                    "track": track,
                    "route_label": dict(label),
                    "candidate_data_path": str(data_path),
                    "candidate_local_index": int(local_index),
                }
            )
        source_summaries.append(
            {
                "data_path": str(data_path),
                "labels_path": str(label_path),
                "tracks": int(len(tracks)),
                "oa_tracks": int(oa_count),
            }
        )
    return candidates, source_summaries


def preserve_split(base_manifest, base_tracks, selected, output_data_path):
    manifest = read_json(base_manifest)
    valid_indices = [int(item) for item in manifest["valid_indices"]]
    test_indices = [int(item) for item in manifest["test_indices"]]
    holdout_indices = valid_indices + test_indices
    holdout_mmsi = {int(base_tracks[index][0, 0]) for index in holdout_indices}

    eligible = []
    excluded = []
    for item in selected:
        mmsi = int(item["track"][0, 0])
        if mmsi in holdout_mmsi:
            excluded.append(item)
        else:
            eligible.append(item)

    combined_mmsi = np.asarray(
        [int(track[0, 0]) for track in base_tracks]
        + [int(item["track"][0, 0]) for item in eligible],
        dtype=np.int64,
    )
    appended_indices = list(range(len(base_tracks), len(base_tracks) + len(eligible)))
    train_indices = [int(item) for item in manifest["train_indices"]] + appended_indices
    split_indices = {
        "train": train_indices,
        "valid": valid_indices,
        "test": test_indices,
    }
    split_mmsi = {
        name: sorted(set(combined_mmsi[indices].astype(int).tolist()))
        for name, indices in split_indices.items()
    }
    payload = {
        "format_version": 1,
        "data_path": str(output_data_path),
        "track_count": int(len(combined_mmsi)),
        "mmsi_hash": hashlib.sha256(combined_mmsi.tobytes()).hexdigest(),
        "split_seed": int(manifest["split_seed"]),
        "test_ratio": float(manifest["test_ratio"]),
        "valid_ratio_within_non_test": float(manifest["valid_ratio_within_non_test"]),
        "group_by_mmsi": True,
        "stratify_label": f"{manifest.get('stratify_label', 'fixed subroute')} + train-only OA_S00 supplement",
        "train_indices": train_indices,
        "valid_indices": valid_indices,
        "test_indices": test_indices,
        "train_mmsi": split_mmsi["train"],
        "valid_mmsi": split_mmsi["valid"],
        "test_mmsi": split_mmsi["test"],
    }
    return eligible, excluded, payload


def select_diverse(items, count, max_per_mmsi):
    selected = []
    per_mmsi = Counter()
    for item in sorted(items, key=lambda row: row["selection_score"], reverse=True):
        mmsi = int(item["track"][0, 0])
        if max_per_mmsi > 0 and per_mmsi[mmsi] >= max_per_mmsi:
            continue
        selected.append(item)
        per_mmsi[mmsi] += 1
        if len(selected) >= count:
            break
    return selected


def main():
    parser = ArgumentParser(
        description="Append conservatively matched OA_S00 tracks from new DMA months."
    )
    parser.add_argument("--base_data_path", required=True)
    parser.add_argument("--base_route_labels_path", required=True)
    parser.add_argument("--base_subroute_labels_path", required=True)
    parser.add_argument("--base_split_manifest", required=True)
    parser.add_argument("--candidate_data_paths", required=True)
    parser.add_argument("--candidate_route_labels_paths", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="dma_2023_06_07_08_plus_09_10_oa_s00")
    parser.add_argument("--target_total", type=int, default=350)
    parser.add_argument("--max_per_mmsi", type=int, default=2)
    parser.add_argument("--resample_points", type=int, default=32)
    parser.add_argument("--feature_window", default="0.38:0.72")
    parser.add_argument("--window_weight", type=float, default=3.0)
    parser.add_argument("--distance_quantile", type=float, default=0.95)
    parser.add_argument("--margin_quantile", type=float, default=0.10)
    parser.add_argument("--min_margin", type=float, default=0.05)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--min_cv_target_recall", type=float, default=0.70)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    start_text, end_text = args.feature_window.split(":", 1)
    feature_window = (float(start_text), float(end_text))
    if not 0.0 <= feature_window[0] < feature_window[1] <= 1.0:
        raise ValueError("--feature_window must satisfy 0 <= START < END <= 1.")
    if not 0.0 < args.distance_quantile <= 1.0:
        raise ValueError("--distance_quantile must be in (0, 1].")
    if not 0.0 <= args.margin_quantile <= 1.0:
        raise ValueError("--margin_quantile must be in [0, 1].")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_data_path = output_dir / f"{args.prefix}.pkl"
    output_route_labels_path = output_dir / f"{args.prefix}_route_labels.json"
    output_subroute_labels_path = output_dir / f"{args.prefix}_subroute_labels.json"
    output_split_path = output_dir / f"{args.prefix}_fixed_split.json"
    supplement_data_path = output_dir / f"{args.prefix}_supplement.pkl"
    supplement_route_labels_path = output_dir / f"{args.prefix}_supplement_route_labels.json"
    supplement_subroute_labels_path = output_dir / f"{args.prefix}_supplement_subroute_labels.json"
    report_path = output_dir / f"{args.prefix}_report.json"
    outputs = [
        output_data_path,
        output_route_labels_path,
        output_subroute_labels_path,
        output_split_path,
        supplement_data_path,
        supplement_route_labels_path,
        supplement_subroute_labels_path,
        report_path,
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Output files already exist; pass --force to replace them: " + ", ".join(existing)
        )

    base_tracks = list(pd.read_pickle(args.base_data_path))
    base_route_labels = read_json(args.base_route_labels_path)
    base_subroute_labels = read_json(args.base_subroute_labels_path)
    validate_alignment(base_tracks, base_route_labels, base_subroute_labels)

    oa_indices = np.asarray(
        [index for index, row in enumerate(base_route_labels) if str(row.get("route")) == TARGET_ROUTE],
        dtype=int,
    )
    oa_targets = np.asarray(
        [str(base_subroute_labels[index].get("subroute")) for index in oa_indices],
        dtype=object,
    )
    unknown_targets = sorted(set(oa_targets) - set(SUBROUTE_ORDER))
    if unknown_targets:
        raise ValueError(f"Unexpected OA subroute labels: {unknown_targets}")
    current_target_count = int(np.sum(oa_targets == TARGET_SUBROUTE))
    additions_needed = max(0, int(args.target_total) - current_target_count)

    oa_paths = np.stack(
        [resample_track(base_tracks[index], args.resample_points, "full") for index in oa_indices]
    )
    state = build_feature_state(oa_paths, feature_window, args.window_weight)
    raw_features = fixed_feature_matrix(oa_paths, state)
    scaler = StandardScaler().fit(raw_features)
    scaled_features = scaler.transform(raw_features)
    centers = class_centroids(scaled_features, oa_targets, SUBROUTE_ORDER)
    base_predictions, base_distance, base_margin, base_distances = distance_predictions(
        scaled_features,
        centers,
        SUBROUTE_ORDER,
    )
    reconstruction = accuracy_summary(oa_targets, base_predictions, SUBROUTE_ORDER)
    groups = np.asarray([int(base_tracks[index][0, 0]) for index in oa_indices], dtype=np.int64)
    cross_validation = grouped_cross_validation(
        raw_features,
        oa_targets,
        groups,
        SUBROUTE_ORDER,
        args.cv_folds,
    )
    target_cv_recall = cross_validation.get("per_class_recall", {}).get(TARGET_SUBROUTE)
    if target_cv_recall is not None and target_cv_recall < args.min_cv_target_recall:
        raise ValueError(
            f"OA_S00 grouped-CV recall {target_cv_recall:.3f} is below "
            f"the required {args.min_cv_target_recall:.3f}; refusing automatic augmentation."
        )

    target_class_index = SUBROUTE_ORDER.index(TARGET_SUBROUTE)
    target_mask = oa_targets == TARGET_SUBROUTE
    target_own_distance = base_distances[target_mask, target_class_index]
    target_correct_margin = base_margin[target_mask & (base_predictions == TARGET_SUBROUTE)]
    distance_threshold = float(np.quantile(target_own_distance, args.distance_quantile))
    margin_threshold = float(args.min_margin)
    if len(target_correct_margin):
        margin_threshold = max(
            margin_threshold,
            float(np.quantile(target_correct_margin, args.margin_quantile)),
        )

    candidate_data_paths = comma_list(args.candidate_data_paths)
    candidate_label_paths = comma_list(args.candidate_route_labels_paths)
    candidates, source_summaries = load_candidate_sets(
        candidate_data_paths,
        candidate_label_paths,
    )
    if not candidates:
        raise ValueError("No OA candidate tracks were found.")
    candidate_paths = np.stack(
        [resample_track(item["track"], args.resample_points, "full") for item in candidates]
    )
    candidate_raw = fixed_feature_matrix(candidate_paths, state)
    candidate_scaled = scaler.transform(candidate_raw)
    predictions, best_distance, margins, all_distances = distance_predictions(
        candidate_scaled,
        centers,
        SUBROUTE_ORDER,
    )

    accepted = []
    rejection_counts = Counter()
    prediction_counts = Counter(str(item) for item in predictions)
    for index, item in enumerate(candidates):
        predicted = str(predictions[index])
        distance_to_target = float(all_distances[index, target_class_index])
        if predicted != TARGET_SUBROUTE:
            rejection_counts[f"predicted_{predicted}"] += 1
            continue
        if distance_to_target > distance_threshold:
            rejection_counts["target_distance"] += 1
            continue
        if float(margins[index]) < margin_threshold:
            rejection_counts["low_margin"] += 1
            continue
        selected = dict(item)
        selected.update(
            {
                "prediction": predicted,
                "assignment_distance": distance_to_target,
                "assignment_margin": float(margins[index]),
                "selection_score": float(margins[index] - 0.25 * distance_to_target / max(distance_threshold, 1e-8)),
            }
        )
        accepted.append(selected)

    # Rank first, then remove holdout-MMSI leakage, then enforce vessel diversity.
    accepted.sort(key=lambda row: row["selection_score"], reverse=True)
    manifest = read_json(args.base_split_manifest)
    holdout_indices = [int(item) for item in manifest["valid_indices"] + manifest["test_indices"]]
    holdout_mmsi = {int(base_tracks[index][0, 0]) for index in holdout_indices}
    leakage_safe = []
    excluded_holdout = []
    for item in accepted:
        if int(item["track"][0, 0]) in holdout_mmsi:
            excluded_holdout.append(item)
        else:
            leakage_safe.append(item)
    selected = select_diverse(leakage_safe, additions_needed, args.max_per_mmsi)
    selected, unexpectedly_excluded, output_manifest = preserve_split(
        args.base_split_manifest,
        base_tracks,
        selected,
        output_data_path,
    )
    if unexpectedly_excluded:
        raise RuntimeError("Holdout-MMSI filtering changed after candidate selection.")

    supplement_tracks = [item["track"] for item in selected]
    supplement_route_labels = []
    supplement_subroute_labels = []
    for local_index, item in enumerate(selected):
        route_record = dict(item["route_label"])
        route_record["index"] = int(local_index)
        route_record.setdefault("mmsi", int(item["track"][0, 0]))
        route_record["oa_s00_assignment_margin"] = float(item["assignment_margin"])
        route_record["oa_s00_assignment_distance"] = float(item["assignment_distance"])
        supplement_route_labels.append(route_record)

        subroute_record = dict(route_record)
        subroute_record.update(
            {
                "parent_route": TARGET_ROUTE,
                "subroute": TARGET_SUBROUTE,
                "subroute_confidence": float(item["assignment_margin"]),
                "subroute_assignment": "fixed_2023_06_07_08_centroid",
            }
        )
        supplement_subroute_labels.append(subroute_record)

    output_tracks = list(base_tracks) + supplement_tracks
    output_route_labels = [dict(item) for item in base_route_labels]
    output_subroute_labels = [dict(item) for item in base_subroute_labels]
    for index, record in enumerate(output_route_labels):
        record["index"] = int(index)
    for index, record in enumerate(output_subroute_labels):
        record["index"] = int(index)
    for local_index, (route_record, subroute_record) in enumerate(
        zip(supplement_route_labels, supplement_subroute_labels)
    ):
        global_index = len(base_tracks) + local_index
        merged_route = dict(route_record)
        merged_route["index"] = int(global_index)
        merged_route["supplement_index"] = int(local_index)
        merged_subroute = dict(subroute_record)
        merged_subroute["index"] = int(global_index)
        merged_subroute["supplement_index"] = int(local_index)
        output_route_labels.append(merged_route)
        output_subroute_labels.append(merged_subroute)

    pd.to_pickle(output_tracks, output_data_path)
    pd.to_pickle(supplement_tracks, supplement_data_path)
    write_json(output_route_labels_path, output_route_labels)
    write_json(output_subroute_labels_path, output_subroute_labels)
    write_json(supplement_route_labels_path, supplement_route_labels)
    write_json(supplement_subroute_labels_path, supplement_subroute_labels)
    write_json(output_split_path, output_manifest)

    selected_sources = Counter(
        str(item["route_label"].get("source", Path(item["candidate_data_path"]).parent.name))
        for item in selected
    )
    final_subroute_counts = Counter(str(item.get("subroute")) for item in output_subroute_labels)
    report = {
        "base_data_path": str(args.base_data_path),
        "candidate_sources": source_summaries,
        "output_data_path": str(output_data_path),
        "output_route_labels_path": str(output_route_labels_path),
        "output_subroute_labels_path": str(output_subroute_labels_path),
        "output_split_path": str(output_split_path),
        "supplement_data_path": str(supplement_data_path),
        "base_tracks": int(len(base_tracks)),
        "output_tracks": int(len(output_tracks)),
        "current_oa_s00": current_target_count,
        "target_oa_s00": int(args.target_total),
        "selected_oa_s00": int(len(selected)),
        "final_oa_s00": int(final_subroute_counts[TARGET_SUBROUTE]),
        "selected_unique_mmsi": int(len({int(item["track"][0, 0]) for item in selected})),
        "selected_source_counts": dict(selected_sources),
        "final_subroute_counts": dict(final_subroute_counts),
        "candidate_prediction_counts": dict(prediction_counts),
        "accepted_before_holdout_and_diversity": int(len(accepted)),
        "excluded_holdout_mmsi": int(len(excluded_holdout)),
        "rejection_counts": dict(rejection_counts),
        "thresholds": {
            "distance_quantile": float(args.distance_quantile),
            "distance": distance_threshold,
            "margin_quantile": float(args.margin_quantile),
            "margin": margin_threshold,
            "max_per_mmsi": int(args.max_per_mmsi),
        },
        "feature_state": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in state.items()
            if key != "route_mean_xy"
        },
        "base_reconstruction": reconstruction,
        "grouped_cross_validation": cross_validation,
    }
    write_json(report_path, report)
    print(json.dumps({
        "base_reconstruction": reconstruction,
        "grouped_cross_validation": cross_validation,
        "candidate_prediction_counts": dict(prediction_counts),
        "selected_oa_s00": len(selected),
        "final_oa_s00": final_subroute_counts[TARGET_SUBROUTE],
        "selected_source_counts": dict(selected_sources),
        "output_tracks": len(output_tracks),
    }, ensure_ascii=False, indent=2))
    print(f"dataset saved to {output_data_path}")
    print(f"report saved to {report_path}")


if __name__ == "__main__":
    main()
