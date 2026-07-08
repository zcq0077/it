from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd


COLUMNS = [
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


def percentiles(values, qs=(0, 1, 5, 50, 95, 99, 99.9, 100)):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {str(q): None for q in qs}
    return {str(q): float(np.percentile(values, q)) for q in qs}


def haversine_nm(lon1, lat1, lon2, lat2):
    radius_km = 6371.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distance_km = 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    return distance_km / 1.852


def inspect(data_path):
    data = pd.read_pickle(data_path)
    lengths = np.array([len(track) for track in data], dtype=int)
    arr = np.concatenate(data, axis=0) if data else np.empty((0, len(COLUMNS)))

    report = {
        "path": str(data_path),
        "tracks": int(len(data)),
        "points": int(len(arr)),
        "unique_mmsi": int(len(set(int(track[0, 0]) for track in data))) if data else 0,
        "track_length_percentiles": percentiles(lengths),
        "non_finite_values": int(np.size(arr) - np.isfinite(arr).sum()),
        "column_percentiles": {
            column: percentiles(arr[:, idx]) for idx, column in enumerate(COLUMNS)
        }
        if len(arr)
        else {},
    }

    dt_all = []
    geo_speed = []
    tracks_bad_dt = 0
    tracks_nonmonotonic = 0
    tracks_sog_gt30 = 0
    tracks_sog_gt50 = 0
    tracks_geo_gt30 = 0
    tracks_geo_gt50 = 0
    suspicious = []

    for idx, track in enumerate(data):
        times = track[:, 14]
        dts = np.diff(times)
        max_geo_speed = None
        if len(dts):
            dt_all.extend(dts.tolist())
            if np.any(dts <= 0):
                tracks_nonmonotonic += 1
            if np.any(dts != 900):
                tracks_bad_dt += 1
            dist = haversine_nm(track[:-1, 3], track[:-1, 4], track[1:, 3], track[1:, 4])
            speed = dist / (dts / 3600.0)
            geo_speed.extend(speed.tolist())
            max_geo_speed = float(np.nanmax(speed))
            if np.any(speed > 30):
                tracks_geo_gt30 += 1
            if np.any(speed > 50):
                tracks_geo_gt50 += 1

        if np.any(track[:, 5] > 30):
            tracks_sog_gt30 += 1
        if np.any(track[:, 5] > 50):
            tracks_sog_gt50 += 1

        suspicious.append(
            {
                "index": int(idx),
                "mmsi": int(track[0, 0]),
                "points": int(len(track)),
                "max_sog": float(np.nanmax(track[:, 5])),
                "max_abs_delta_sog": float(np.nanmax(np.abs(track[:, 11]))),
                "max_geo_speed_kn": max_geo_speed,
            }
        )

    sog = arr[:, 5] if len(arr) else np.array([])
    delta_sog = arr[:, 11] if len(arr) else np.array([])
    geo_speed = np.asarray(geo_speed)
    report.update(
        {
            "time_step_percentiles": percentiles(dt_all),
            "tracks_with_nonmonotonic_time": int(tracks_nonmonotonic),
            "tracks_with_dt_not_900": int(tracks_bad_dt),
            "reported_sog_gt30_points": int(np.sum(sog > 30)),
            "reported_sog_gt50_points": int(np.sum(sog > 50)),
            "tracks_with_reported_sog_gt30": int(tracks_sog_gt30),
            "tracks_with_reported_sog_gt50": int(tracks_sog_gt50),
            "abs_delta_sog_gt20_points": int(np.sum(np.abs(delta_sog) > 20)),
            "abs_delta_sog_gt50_points": int(np.sum(np.abs(delta_sog) > 50)),
            "geo_speed_kn_percentiles": percentiles(geo_speed),
            "geo_speed_gt30_segments": int(np.sum(geo_speed > 30)) if len(geo_speed) else 0,
            "geo_speed_gt50_segments": int(np.sum(geo_speed > 50)) if len(geo_speed) else 0,
            "tracks_with_geo_speed_gt30": int(tracks_geo_gt30),
            "tracks_with_geo_speed_gt50": int(tracks_geo_gt50),
            "top_max_sog_tracks": sorted(suspicious, key=lambda item: item["max_sog"], reverse=True)[:15],
            "top_geo_speed_tracks": sorted(
                suspicious,
                key=lambda item: -1.0 if item["max_geo_speed_kn"] is None else item["max_geo_speed_kn"],
                reverse=True,
            )[:15],
        }
    )
    return report


def main():
    parser = ArgumentParser(description="Inspect iTentformer-format DMA pkl quality.")
    parser.add_argument("--data_path", default="dataset/dma_raw_2023_06/dma_itentformer_all.pkl")
    parser.add_argument("--report_path", default=None)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    report_path = Path(args.report_path) if args.report_path else data_path.with_name("dma_quality_report.json")
    report = inspect(data_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_keys = [
        "tracks",
        "points",
        "unique_mmsi",
        "non_finite_values",
        "tracks_with_dt_not_900",
        "reported_sog_gt30_points",
        "reported_sog_gt50_points",
        "tracks_with_reported_sog_gt30",
        "tracks_with_reported_sog_gt50",
        "geo_speed_gt30_segments",
        "geo_speed_gt50_segments",
        "tracks_with_geo_speed_gt30",
        "tracks_with_geo_speed_gt50",
    ]
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2))
    print(f"report saved to {report_path}")


if __name__ == "__main__":
    main()
