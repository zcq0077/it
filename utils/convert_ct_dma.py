from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd


LAT_MIN = 55.5
LAT_RANGE = 2.5
LON_MIN = 10.3
LON_RANGE = 2.7
SOG_SCALE = 30.0
COG_SCALE = 360.0

RAW_COLUMNS = [
    "lat_norm",
    "lon_norm",
    "sog_norm",
    "cog_norm",
    "unix_timestamp",
    "mmsi",
]

ITENTFORMER_COLUMNS = [
    "MMSI",
    "Length",
    "Course",
    "Lon",
    "Lat",
    "SOG",
    "vx",
    "vy",
    "delta_Course",
    "delta_Lon",
    "delta_Lat",
    "delta_SOG",
    "delta_vx",
    "delta_vy",
    "UnixTime",
]

COLUMN_PROVENANCE = {
    "MMSI": "copied from item['mmsi']; falls back to traj[:, 5]",
    "Length": "not present in CT-DMA; filled with --default_length",
    "Course": "denormalized from cog_norm * 360",
    "Lon": "denormalized from 10.3 + lon_norm * 2.7",
    "Lat": "denormalized from 55.5 + lat_norm * 2.5",
    "SOG": "denormalized from sog_norm * 30",
    "vx": "derived as SOG * sin(Course)",
    "vy": "derived as SOG * cos(Course)",
    "delta_Course": "circular first-order difference in degrees",
    "delta_Lon": "first-order difference of Lon",
    "delta_Lat": "first-order difference of Lat",
    "delta_SOG": "first-order difference of SOG",
    "delta_vx": "first-order difference of vx",
    "delta_vy": "first-order difference of vy",
    "UnixTime": "copied from unix_timestamp",
}


def circular_diff_deg(values):
    diff = np.diff(values)
    diff = (diff + 180.0) % 360.0 - 180.0
    return np.concatenate(([0.0], diff))


def prepend_zero_diff(values):
    return np.concatenate(([0.0], np.diff(values)))


def as_python_number(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def summarize_numeric(values):
    values = np.asarray(values)
    return {
        "min": as_python_number(np.min(values)),
        "p01": as_python_number(np.percentile(values, 1)),
        "median": as_python_number(np.median(values)),
        "p99": as_python_number(np.percentile(values, 99)),
        "max": as_python_number(np.max(values)),
    }


def get_mmsi(item, traj):
    if "mmsi" in item and item["mmsi"] is not None:
        return float(item["mmsi"]), "item"
    return float(traj[0, 5]), "traj_col"


def get_length(item, default_length):
    for key in ("length", "Length", "ship_length", "vessel_length"):
        if key in item and item[key] is not None:
            return float(item[key]), f"item[{key!r}]"
    return float(default_length), "filled_default"


def check_normalized_ranges(traj, eps=1e-6):
    checks = {
        "lat_norm": traj[:, 0],
        "lon_norm": traj[:, 1],
        "sog_norm": traj[:, 2],
        "cog_norm": traj[:, 3],
    }
    out = {}
    for name, values in checks.items():
        out[name] = int(np.sum((values < -eps) | (values > 1.0 + eps)))
    return out


def convert_track(item, default_length=0.0, sort_by_time=True, clip_normalized=True):
    traj = np.asarray(item["traj"], dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 6:
        raise ValueError(f"Expected traj shape (n, 6), got {traj.shape}.")

    stats = {
        "points": int(len(traj)),
        "unsorted": False,
        "mmsi_source": None,
        "length_source": None,
        "mmsi_mismatch_points": 0,
        "normalized_out_of_range": check_normalized_ranges(traj),
    }

    if sort_by_time and len(traj) > 1 and np.any(np.diff(traj[:, 4]) < 0):
        traj = traj[np.argsort(traj[:, 4])]
        stats["unsorted"] = True

    norm = traj[:, :4].copy()
    if clip_normalized:
        norm[:, 0] = np.clip(norm[:, 0], 0.0, 1.0)
        norm[:, 1] = np.clip(norm[:, 1], 0.0, 1.0)
        norm[:, 2] = np.clip(norm[:, 2], 0.0, None)
        norm[:, 3] = np.mod(norm[:, 3], 1.0)

    lat = LAT_MIN + norm[:, 0] * LAT_RANGE
    lon = LON_MIN + norm[:, 1] * LON_RANGE
    sog = norm[:, 2] * SOG_SCALE
    cog = np.mod(norm[:, 3] * COG_SCALE, 360.0)
    unix_time = traj[:, 4]

    mmsi_value, mmsi_source = get_mmsi(item, traj)
    stats["mmsi_source"] = mmsi_source
    mmsi = np.full(len(traj), mmsi_value, dtype=np.float64)
    stats["mmsi_mismatch_points"] = int(np.sum(traj[:, 5] != mmsi_value))

    length_value, length_source = get_length(item, default_length)
    stats["length_source"] = length_source
    length = np.full(len(traj), length_value, dtype=np.float64)

    cog_rad = np.deg2rad(cog)
    vx = sog * np.sin(cog_rad)
    vy = sog * np.cos(cog_rad)

    delta_cog = circular_diff_deg(cog)
    delta_lon = prepend_zero_diff(lon)
    delta_lat = prepend_zero_diff(lat)
    delta_sog = prepend_zero_diff(sog)
    delta_vx = prepend_zero_diff(vx)
    delta_vy = prepend_zero_diff(vy)

    converted = np.stack(
        [
            mmsi,
            length,
            cog,
            lon,
            lat,
            sog,
            vx,
            vy,
            delta_cog,
            delta_lon,
            delta_lat,
            delta_sog,
            delta_vx,
            delta_vy,
            unix_time,
        ],
        axis=1,
    ).astype(np.float64)

    stats["zero_sog_points"] = int(np.sum(sog == 0.0))
    stats["low_sog_points_lt_0p5kn"] = int(np.sum(sog < 0.5))
    if len(unix_time) > 1:
        stats["time_step_values"] = [as_python_number(v) for v in np.unique(np.diff(unix_time))]
    else:
        stats["time_step_values"] = []

    return converted, stats


def build_split_report(raw_data, converted, track_stats):
    raw_arr = np.concatenate([np.asarray(item["traj"], dtype=np.float64) for item in raw_data], axis=0)
    converted_arr = np.concatenate(converted, axis=0)
    lengths = np.array([len(track) for track in converted])
    dts = []
    for item in raw_data:
        t = np.asarray(item["traj"])[:, 4]
        if len(t) > 1:
            dts.extend(np.diff(t))
    dt_counter = Counter(float(v) for v in dts)

    return {
        "tracks": int(len(converted)),
        "points": int(sum(len(track) for track in converted)),
        "track_length": {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        },
        "unique_mmsi": int(len({int(track[0, 0]) for track in converted})),
        "time_step_top": [[as_python_number(k), int(v)] for k, v in dt_counter.most_common(10)],
        "raw_ranges": {
            name: summarize_numeric(raw_arr[:, idx]) for idx, name in enumerate(RAW_COLUMNS)
        },
        "converted_ranges": {
            name: summarize_numeric(converted_arr[:, idx])
            for idx, name in enumerate(ITENTFORMER_COLUMNS)
        },
        "length_source_counts": dict(Counter(s["length_source"] for s in track_stats)),
        "mmsi_source_counts": dict(Counter(s["mmsi_source"] for s in track_stats)),
        "unsorted_tracks": int(sum(s["unsorted"] for s in track_stats)),
        "mmsi_mismatch_points": int(sum(s["mmsi_mismatch_points"] for s in track_stats)),
        "zero_sog_points": int(sum(s["zero_sog_points"] for s in track_stats)),
        "low_sog_points_lt_0p5kn": int(sum(s["low_sog_points_lt_0p5kn"] for s in track_stats)),
        "normalized_out_of_range": {
            key: int(sum(s["normalized_out_of_range"][key] for s in track_stats))
            for key in ("lat_norm", "lon_norm", "sog_norm", "cog_norm")
        },
    }


def convert_split(input_path, output_path, default_length, sort_by_time, clip_normalized):
    raw_data = pd.read_pickle(input_path)
    converted = []
    track_stats = []
    for item in raw_data:
        track, stats = convert_track(
            item,
            default_length=default_length,
            sort_by_time=sort_by_time,
            clip_normalized=clip_normalized,
        )
        converted.append(track)
        track_stats.append(stats)

    pd.to_pickle(converted, output_path)
    report = build_split_report(raw_data, converted, track_stats)
    print(
        f"{input_path.name} -> {output_path.name}: "
        f"{report['tracks']} tracks, length min/mean/max "
        f"{report['track_length']['min']}/{report['track_length']['mean']:.2f}/{report['track_length']['max']}"
    )
    return converted, report


def main():
    parser = ArgumentParser(description="Convert CT-DMA normalized tracks to iTentformer AIS format.")
    parser.add_argument("--input_dir", default="dataset/ct_dma")
    parser.add_argument("--output_dir", default="dataset/ct_dma")
    parser.add_argument("--output_prefix", default="ct_dma_itentformer_smart")
    parser.add_argument("--default_length", type=float, default=0.0)
    parser.add_argument("--no_sort_by_time", action="store_true")
    parser.add_argument("--no_clip_normalized", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_tracks = []
    report = {
        "raw_columns": RAW_COLUMNS,
        "itentformer_columns": ITENTFORMER_COLUMNS,
        "column_provenance": COLUMN_PROVENANCE,
        "normalization": {
            "lat": "55.5 + lat_norm * 2.5",
            "lon": "10.3 + lon_norm * 2.7",
            "sog": "sog_norm * 30",
            "cog": "(cog_norm * 360) mod 360",
        },
        "notes": [
            "CT-DMA files contain only mmsi and traj; vessel length is not recoverable from the local data.",
            "The current iTentformer.py uses Course/Lon/Lat/SOG and their deltas as inputs; Length/vx/vy are format-completion columns.",
        ],
        "splits": {},
    }

    for split in ("train", "valid", "test"):
        converted, split_report = convert_split(
            input_dir / f"ct_dma_{split}.pkl",
            output_dir / f"{args.output_prefix}_{split}.pkl",
            default_length=args.default_length,
            sort_by_time=not args.no_sort_by_time,
            clip_normalized=not args.no_clip_normalized,
        )
        all_tracks.extend(converted)
        report["splits"][split] = split_report

    all_path = output_dir / f"{args.output_prefix}_all.pkl"
    pd.to_pickle(all_tracks, all_path)
    report["merged"] = {
        "path": str(all_path),
        "tracks": int(len(all_tracks)),
        "points": int(sum(len(track) for track in all_tracks)),
    }

    report_path = output_dir / f"{args.output_prefix}_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"merged -> {all_path.name}: {len(all_tracks)} tracks")
    print(f"report -> {report_path.name}")


if __name__ == "__main__":
    main()
