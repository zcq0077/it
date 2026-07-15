"""Build point-aligned AIS voyage context without extracting the source ZIP files."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import zipfile

import numpy as np
import pandas as pd


CONTEXT_COLUMNS = [
    "# Timestamp",
    "MMSI",
    "Navigational status",
    "Ship type",
    "Width",
    "Length",
    "Draught",
    "Destination",
    "ETA",
]


def infer_source_name(path):
    match = re.search(r"(20\d{2})[-_](\d{2})", Path(path).stem)
    if not match:
        raise ValueError(f"Cannot infer YYYY-MM source from ZIP name: {path}")
    return f"{match.group(1)}-{match.group(2)}"


def clean_text(value, fallback="unknown"):
    if value is None or pd.isna(value):
        return fallback
    text = re.sub(r"\s+", " ", str(value).strip().upper())
    text = re.sub(r"[^A-Z0-9 /+>._-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or fallback


def format_number(value, unit):
    if value is None or not np.isfinite(value):
        return f"unknown {unit}"
    return f"{float(value):.1f} {unit}"


def build_context_text(row, fallback_length):
    length = row.get("length", np.nan)
    if not np.isfinite(length) and np.isfinite(fallback_length):
        length = fallback_length
    return (
        "Observed AIS voyage context at forecast time: "
        f"ship type {clean_text(row.get('ship_type'))}; "
        f"length {format_number(length, 'm')}; "
        f"width {format_number(row.get('width', np.nan), 'm')}; "
        f"draught {format_number(row.get('draught', np.nan), 'm')}; "
        f"destination {clean_text(row.get('destination'))}; "
        f"ETA {clean_text(row.get('eta'))}; "
        f"navigational status {clean_text(row.get('nav_status'))}."
    )


def parse_chunk(chunk):
    renamed = {name: name.lstrip("# ").strip() for name in chunk.columns}
    chunk = chunk.rename(columns=renamed)
    timestamps = pd.to_datetime(
        chunk["Timestamp"],
        utc=True,
        errors="coerce",
        dayfirst=True,
    )
    result = pd.DataFrame(
        {
            "ts": timestamps.astype("int64") // 1_000_000_000,
            "mmsi": pd.to_numeric(chunk["MMSI"], errors="coerce"),
            "nav_status": chunk["Navigational status"],
            "ship_type": chunk["Ship type"],
            "width": pd.to_numeric(chunk["Width"], errors="coerce"),
            "length": pd.to_numeric(chunk["Length"], errors="coerce"),
            "draught": pd.to_numeric(chunk["Draught"], errors="coerce"),
            "destination": chunk["Destination"],
            "eta": chunk["ETA"],
        }
    )
    result = result[timestamps.notna() & result["mmsi"].notna()].copy()
    result["mmsi"] = result["mmsi"].astype("int64")
    return result


def scan_zip(
        path,
        source,
        intervals_by_mmsi,
        tracks,
        aligned_texts,
        best_timestamps,
        max_staleness_seconds,
        chunksize,
        max_files,
        counters,
):
    target_mmsi = set(intervals_by_mmsi)
    with zipfile.ZipFile(path) as archive:
        members = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
        if max_files > 0:
            members = members[:max_files]
        for member_index, member in enumerate(members, start=1):
            print(f"[{source}] {member_index}/{len(members)} {member}", flush=True)
            with archive.open(member) as handle:
                reader = pd.read_csv(
                    handle,
                    usecols=CONTEXT_COLUMNS,
                    chunksize=chunksize,
                    low_memory=False,
                )
                for chunk in reader:
                    counters["raw_rows"] += len(chunk)
                    parsed = parse_chunk(chunk)
                    parsed = parsed[parsed["mmsi"].isin(target_mmsi)]
                    counters["target_mmsi_rows"] += len(parsed)
                    if parsed.empty:
                        continue
                    for mmsi, group in parsed.groupby("mmsi", sort=False):
                        for track_index, start_time, end_time in intervals_by_mmsi[int(mmsi)]:
                            selected = group[(group["ts"] >= start_time) & (group["ts"] <= end_time)]
                            if selected.empty:
                                continue
                            selected = selected.sort_values("ts").drop_duplicates("ts", keep="last")
                            record_times = selected["ts"].to_numpy(dtype=np.int64)
                            grid_times = np.asarray(tracks[track_index][:, 14], dtype=np.int64)
                            positions = np.searchsorted(record_times, grid_times, side="right") - 1
                            valid = positions >= 0
                            candidate_times = np.full(len(grid_times), -1, dtype=np.int64)
                            candidate_times[valid] = record_times[positions[valid]]
                            ages = grid_times - candidate_times
                            update = (
                                valid
                                & (ages >= 0)
                                & (ages <= max_staleness_seconds)
                                & (candidate_times > best_timestamps[track_index])
                            )
                            fallback_length = float(tracks[track_index][0, 1])
                            for point_index in np.flatnonzero(update):
                                row = selected.iloc[int(positions[point_index])]
                                aligned_texts[track_index][point_index] = build_context_text(
                                    row,
                                    fallback_length,
                                )
                            best_timestamps[track_index][update] = candidate_times[update]
                            counters["matched_rows"] += len(selected)


def pool_aligned_contexts(aligned_texts):
    text_pool = ["AIS voyage context unavailable at forecast time."]
    text_to_id = {text_pool[0]: 0}
    context_ids = []
    counters = Counter()

    for texts in aligned_texts:
        ids = np.zeros(len(texts), dtype=np.int32)
        if any(bool(text) for text in texts):
            for point_index, text in enumerate(texts):
                if not text:
                    continue
                context_id = text_to_id.get(text)
                if context_id is None:
                    context_id = len(text_pool)
                    text_to_id[text] = context_id
                    text_pool.append(text)
                ids[point_index] = context_id
                counters["available_points"] += 1
        else:
            counters["tracks_without_records"] += 1
        counters["total_points"] += len(texts)
        context_ids.append(ids)

    return text_pool, context_ids, counters


def main():
    parser = ArgumentParser(description="Build time-safe DMA voyage context sidecar data.")
    parser.add_argument("--input_zip", action="append", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_staleness_hours", type=float, default=24.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists; pass --force to replace it.")

    tracks = pd.read_pickle(args.data_path)
    with Path(args.labels_path).open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if len(tracks) != len(labels):
        raise ValueError("Track and label counts do not match.")

    staleness_seconds = int(round(args.max_staleness_hours * 3600.0))
    intervals_by_source = defaultdict(lambda: defaultdict(list))
    for index, (track, label) in enumerate(zip(tracks, labels)):
        source = str(label.get("source", ""))
        mmsi = int(label.get("mmsi", track[0, 0]))
        start_time = int(track[0, 14]) - staleness_seconds
        end_time = int(track[-1, 14])
        intervals_by_source[source][mmsi].append((index, start_time, end_time))

    aligned_texts = [np.full(len(track), "", dtype=object) for track in tracks]
    best_timestamps = [np.full(len(track), -1, dtype=np.int64) for track in tracks]
    scan_counters = Counter()
    for zip_path in args.input_zip:
        source = infer_source_name(zip_path)
        intervals = intervals_by_source.get(source)
        if not intervals:
            print(f"[{source}] no matching tracks; skipped {zip_path}")
            continue
        scan_zip(
            zip_path,
            source,
            intervals,
            tracks,
            aligned_texts,
            best_timestamps,
            staleness_seconds,
            args.chunksize,
            args.max_files,
            scan_counters,
        )

    text_pool, context_ids, alignment_counters = pool_aligned_contexts(aligned_texts)
    payload = {
        "format_version": 1,
        "data_path": str(args.data_path),
        "labels_path": str(args.labels_path),
        "track_count": len(tracks),
        "max_staleness_hours": args.max_staleness_hours,
        "text_pool": text_pool,
        "context_ids": context_ids,
        "scan_counters": dict(scan_counters),
        "alignment_counters": dict(alignment_counters),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, output_path)
    available = alignment_counters["available_points"]
    total = max(alignment_counters["total_points"], 1)
    print(f"saved {output_path}")
    print(f"tracks={len(tracks)}, unique_contexts={len(text_pool)}, coverage={100.0 * available / total:.2f}%")


if __name__ == "__main__":
    main()
