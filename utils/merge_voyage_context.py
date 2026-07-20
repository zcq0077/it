"""Merge aligned voyage-context sidecars while deduplicating text entries."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def merge_payloads(paths):
    text_pool = ["AIS voyage context unavailable at forecast time."]
    text_to_id = {text_pool[0]: 0}
    context_ids = []
    source_reports = []
    total_points = 0
    available_points = 0

    for path in paths:
        payload = pd.read_pickle(path)
        source_pool = list(payload.get("text_pool", []))
        source_ids = list(payload.get("context_ids", []))
        if not source_pool:
            raise ValueError(f"Context sidecar has no text_pool: {path}")

        remap = np.zeros(len(source_pool), dtype=np.int32)
        for source_id, text in enumerate(source_pool):
            text = str(text)
            target_id = text_to_id.get(text)
            if target_id is None:
                target_id = len(text_pool)
                text_to_id[text] = target_id
                text_pool.append(text)
            remap[source_id] = target_id

        for track_ids in source_ids:
            track_ids = np.asarray(track_ids, dtype=np.int64)
            if np.any(track_ids < 0) or np.any(track_ids >= len(source_pool)):
                raise ValueError(f"Invalid context id in {path}.")
            merged_ids = remap[track_ids].astype(np.int32)
            context_ids.append(merged_ids)
            total_points += int(merged_ids.size)
            available_points += int(np.count_nonzero(merged_ids))

        source_reports.append(
            {
                "path": str(path),
                "track_count": int(len(source_ids)),
                "text_count": int(len(source_pool)),
            }
        )
    alignment_counters = {
        "total_points": total_points,
        "available_points": available_points,
        "unavailable_points": total_points - available_points,
    }
    return text_pool, context_ids, source_reports, alignment_counters


def main():
    parser = ArgumentParser(description="Merge DMA voyage-context sidecars.")
    parser.add_argument("--input_paths", nargs="+", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists; pass --force to replace it.")

    text_pool, context_ids, source_reports, alignment_counters = merge_payloads(args.input_paths)
    tracks = pd.read_pickle(args.data_path)
    if len(tracks) != len(context_ids):
        raise ValueError(
            f"Merged context tracks={len(context_ids)} do not match dataset tracks={len(tracks)}."
        )
    for index, (track, ids) in enumerate(zip(tracks, context_ids)):
        if len(track) != len(ids):
            raise ValueError(
                f"Track {index} points={len(track)} do not match context ids={len(ids)}."
            )

    payload = {
        "format_version": 1,
        "data_path": str(args.data_path),
        "labels_path": str(args.labels_path),
        "track_count": int(len(tracks)),
        "text_pool": text_pool,
        "context_ids": context_ids,
        "alignment_counters": alignment_counters,
        "sources": source_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, output_path)
    print(f"saved {output_path}")
    print(f"tracks={len(tracks)}, unique_contexts={len(text_pool)}")
    coverage = 100.0 * alignment_counters["available_points"] / max(
        alignment_counters["total_points"], 1
    )
    print(f"point_coverage={coverage:.2f}%")


if __name__ == "__main__":
    main()
