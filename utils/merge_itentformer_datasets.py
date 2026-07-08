from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd


def comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_labels(path, count):
    with Path(path).open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if len(labels) != count:
        raise ValueError(f"{path} labels={len(labels)} does not match tracks={count}.")
    return labels


def summarize_tracks(tracks, labels):
    lengths = np.array([len(track) for track in tracks], dtype=int)
    points = int(lengths.sum()) if len(lengths) else 0
    unique_mmsi = len({int(track[0, 0]) for track in tracks}) if tracks else 0
    return {
        "tracks": int(len(tracks)),
        "points": points,
        "unique_mmsi": int(unique_mmsi),
        "route_counts": dict(Counter(item["route"] for item in labels)),
        "track_length": {
            "min": int(lengths.min()) if len(lengths) else 0,
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
        },
    }


def main():
    parser = ArgumentParser(description="Merge iTentformer-format pkl datasets and route label json files.")
    parser.add_argument("--data_paths", required=True, help="Comma-separated pkl paths.")
    parser.add_argument("--labels_paths", required=True, help="Comma-separated route-label json paths.")
    parser.add_argument("--source_names", default="", help="Comma-separated names, e.g. 2023-06,2023-07.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_labels_path", required=True)
    parser.add_argument("--report_path", required=True)
    args = parser.parse_args()

    data_paths = comma_list(args.data_paths)
    labels_paths = comma_list(args.labels_paths)
    source_names = comma_list(args.source_names)
    if len(data_paths) != len(labels_paths):
        raise ValueError("--data_paths and --labels_paths must have the same length.")
    if source_names and len(source_names) != len(data_paths):
        raise ValueError("--source_names must be empty or match --data_paths length.")
    if not source_names:
        source_names = [Path(path).parent.name for path in data_paths]

    merged_tracks = []
    merged_labels = []
    source_reports = []
    for source_idx, (data_path, labels_path, source_name) in enumerate(zip(data_paths, labels_paths, source_names)):
        tracks = pd.read_pickle(data_path)
        labels = load_labels(labels_path, len(tracks))
        source_reports.append(
            {
                "source": source_name,
                "data_path": data_path,
                "labels_path": labels_path,
                **summarize_tracks(tracks, labels),
            }
        )
        for local_idx, (track, label) in enumerate(zip(tracks, labels)):
            merged_tracks.append(track)
            merged_label = dict(label)
            merged_label["index"] = len(merged_labels)
            merged_label["source"] = source_name
            merged_label["source_index"] = int(local_idx)
            merged_label["source_order"] = int(source_idx)
            merged_labels.append(merged_label)

    output_path = Path(args.output_path)
    output_labels_path = Path(args.output_labels_path)
    report_path = Path(args.report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_labels_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pd.to_pickle(merged_tracks, output_path)
    output_labels_path.write_text(json.dumps(merged_labels, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "output_path": str(output_path),
        "output_labels_path": str(output_labels_path),
        "sources": source_reports,
        "merged": summarize_tracks(merged_tracks, merged_labels),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["merged"], indent=2, ensure_ascii=False))
    print(f"pkl saved to {output_path}")
    print(f"labels saved to {output_labels_path}")
    print(f"report saved to {report_path}")


if __name__ == "__main__":
    main()
