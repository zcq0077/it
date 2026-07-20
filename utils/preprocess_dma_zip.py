from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json
import math
import sqlite3
import zipfile

import numpy as np
import pandas as pd


RAW_USECOLS = [
    "# Timestamp",
    "Type of mobile",
    "MMSI",
    "Latitude",
    "Longitude",
    "SOG",
    "COG",
    "Heading",
    "Ship type",
    "Width",
    "Length",
    "Draught",
]

DB_COLUMNS = [
    "mmsi",
    "ts",
    "lat",
    "lon",
    "sog",
    "cog",
    "heading",
    "length",
    "width",
    "draught",
    "ship_type",
    "source",
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

DEFAULT_ROUTE_GATES = {
    # Approximate gates digitized from Fig. 10. They are intentionally configurable
    # because the paper does not publish exact Ti/O/A/B1/B2 polygons.
    "O": "10.30,11.10,57.35,57.85",
    "TI": "11.65,12.10,56.45,56.85",
    "A": "12.10,12.75,55.90,56.35",
    "B1": "11.20,11.90,56.05,56.35",
    "B2": "10.95,11.65,56.30,56.65",
    "C": "10.35,11.20,55.65,56.15",
}

DEFAULT_ROUTE_BALANCE = ""

REVERSE_ROUTE_LABELS = {
    "OA": "AO",
    "OB1": "B1O",
    "OB2": "B2O",
    "OC": "CO",
}


def comma_list(value):
    if value is None or value.strip().lower() in {"", "all", "*"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_gate(value, name):
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{name} must be lon_min,lon_max,lat_min,lat_max.")
    lon_min, lon_max, lat_min, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError(f"{name} has invalid bounds: {value}")
    return {
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
    }


def get_route_gates(args):
    return {
        "O": parse_gate(args.gate_o, "gate_o"),
        "TI": parse_gate(args.gate_ti, "gate_ti"),
        "A": parse_gate(args.gate_a, "gate_a"),
        "B1": parse_gate(args.gate_b1, "gate_b1"),
        "B2": parse_gate(args.gate_b2, "gate_b2"),
        "C": parse_gate(args.gate_c, "gate_c"),
    }


def parse_balance_counts(value):
    if not value:
        return {}
    result = {}
    for item in value.split(","):
        if not item.strip():
            continue
        label, count = item.split(":")
        result[label.strip().upper()] = int(count)
    return result


def as_number(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def numeric_summary(values):
    values = np.asarray(values)
    return {
        "min": as_number(np.min(values)),
        "mean": as_number(np.mean(values)),
        "max": as_number(np.max(values)),
    }


def create_connection(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    return conn


def init_db(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ais (
            mmsi INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            sog REAL NOT NULL,
            cog REAL NOT NULL,
            heading REAL,
            length REAL,
            width REAL,
            draught REAL,
            ship_type TEXT,
            source TEXT
        )
        """
    )
    conn.commit()


def make_index(conn):
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ais_mmsi_ts ON ais (mmsi, ts)")
    conn.commit()


def normalize_columns(chunk):
    return chunk.rename(columns={"# Timestamp": "Timestamp"})


def valid_mmsi_mask(mmsi_series, mid_min, mid_max):
    mmsi = mmsi_series.astype("string").str.strip()
    mmsi = mmsi.str.replace(r"\.0$", "", regex=True)
    base = mmsi.str.fullmatch(r"\d{9}", na=False)
    mid = pd.to_numeric(mmsi.str.slice(0, 3), errors="coerce")
    return base & mid.between(mid_min, mid_max), pd.to_numeric(mmsi, errors="coerce")


def contains_any(series, allowed):
    if not allowed:
        return pd.Series(True, index=series.index)
    text = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for item in allowed:
        mask |= text.str.contains(item.lower(), regex=False)
    return mask


def to_epoch_seconds(timestamp_series):
    parsed = pd.to_datetime(
        timestamp_series,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
        utc=True,
    )
    seconds = parsed.astype("int64") // 1_000_000_000
    seconds[parsed.isna()] = np.nan
    return seconds


def apply_filter(chunk, args, file_name, counters):
    chunk = normalize_columns(chunk)
    counters["raw_rows"] += int(len(chunk))

    mobile_types = comma_list(args.mobile_types)
    ship_types = comma_list(args.ship_types)

    mmsi_ok, mmsi_numeric = valid_mmsi_mask(chunk["MMSI"], args.mid_min, args.mid_max)
    ts = to_epoch_seconds(chunk["Timestamp"])
    lat = pd.to_numeric(chunk["Latitude"], errors="coerce")
    lon = pd.to_numeric(chunk["Longitude"], errors="coerce")
    sog = pd.to_numeric(chunk["SOG"], errors="coerce")
    cog = pd.to_numeric(chunk["COG"], errors="coerce")
    heading = pd.to_numeric(chunk["Heading"], errors="coerce")
    length = pd.to_numeric(chunk["Length"], errors="coerce")
    width = pd.to_numeric(chunk["Width"], errors="coerce")
    draught = pd.to_numeric(chunk["Draught"], errors="coerce")

    checks = [
        ("bad_mmsi", mmsi_ok),
        ("bad_timestamp", ts.notna()),
        ("bad_lat_lon", lat.notna() & lon.notna()),
        ("outside_roi", lat.between(args.lat_min, args.lat_max) & lon.between(args.lon_min, args.lon_max)),
        ("bad_sog", sog.notna() & (sog >= args.vth)),
        ("bad_cog", cog.notna() & (cog >= 0.0) & (cog < 360.0)),
        ("bad_mobile_type", contains_any(chunk["Type of mobile"], mobile_types)),
        ("bad_ship_type", contains_any(chunk["Ship type"], ship_types)),
    ]

    mask = pd.Series(True, index=chunk.index)
    for name, ok in checks:
        failed = mask & ~ok
        counters[name] += int(failed.sum())
        mask &= ok

    if not bool(mask.any()):
        return []

    heading = heading.where((heading >= 0.0) & (heading < 360.0))
    clean = pd.DataFrame(
        {
            "mmsi": mmsi_numeric[mask].astype("int64"),
            "ts": ts[mask].astype("int64"),
            "lat": lat[mask].astype("float64"),
            "lon": lon[mask].astype("float64"),
            "sog": sog[mask].astype("float64"),
            "cog": cog[mask].astype("float64"),
            "heading": heading[mask].astype("float64"),
            "length": length[mask].astype("float64"),
            "width": width[mask].astype("float64"),
            "draught": draught[mask].astype("float64"),
            "ship_type": chunk.loc[mask, "Ship type"].fillna("").astype(str),
            "source": file_name,
        }
    )
    counters["kept_rows"] += int(len(clean))
    return list(clean.itertuples(index=False, name=None))


def scan_zip_to_sqlite(args, db_path):
    if db_path.exists():
        if args.resume:
            print(f"stage database exists, resume scan skipped: {db_path}")
            return {"resume": True}
        if not args.force:
            raise FileExistsError(f"{db_path} exists. Use --force to rebuild it or --resume to reuse it.")
        db_path.unlink()

    conn = create_connection(db_path)
    init_db(conn)

    counters = Counter()
    insert_sql = "INSERT INTO ais VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    csv_count = 0
    chunk_count = 0

    with zipfile.ZipFile(args.input_zip) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        names.sort()
        for name in names:
            if args.max_files and csv_count >= args.max_files:
                break
            csv_count += 1
            print(f"[scan] {csv_count}/{len(names)} {name}")
            with archive.open(name) as handle:
                reader = pd.read_csv(
                    handle,
                    usecols=RAW_USECOLS,
                    dtype="string",
                    chunksize=args.chunksize,
                    on_bad_lines="skip",
                )
                for chunk in reader:
                    chunk_count += 1
                    rows = apply_filter(chunk, args, name, counters)
                    if rows:
                        conn.executemany(insert_sql, rows)
                    if chunk_count % args.commit_every_chunks == 0:
                        conn.commit()
                        print(
                            f"  chunks={chunk_count}, raw={counters['raw_rows']}, "
                            f"kept={counters['kept_rows']}"
                        )
                    if args.max_chunks and chunk_count >= args.max_chunks:
                        conn.commit()
                        make_index(conn)
                        conn.close()
                        counters["csv_files_seen"] = csv_count
                        counters["chunks_seen"] = chunk_count
                        return dict(counters)

    conn.commit()
    make_index(conn)
    conn.close()
    counters["csv_files_seen"] = csv_count
    counters["chunks_seen"] = chunk_count
    return dict(counters)


def haversine_nm(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distance_km = 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    return distance_km / 1.852


def circular_diff_deg(values):
    diff = np.diff(values)
    diff = (diff + 180.0) % 360.0 - 180.0
    return np.concatenate(([0.0], diff))


def prepend_zero_diff(values):
    return np.concatenate(([0.0], np.diff(values)))


def median_or_default(values, default):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) == 0:
        return float(default)
    return float(np.median(arr))


def remove_drift_points(frame, speed_limit_kn):
    if len(frame) <= 1:
        return frame, 0
    keep = [0]
    last = 0
    removed = 0
    lat = frame["lat"].to_numpy()
    lon = frame["lon"].to_numpy()
    ts = frame["ts"].to_numpy()
    for idx in range(1, len(frame)):
        dt = ts[idx] - ts[last]
        if dt <= 0:
            removed += 1
            continue
        dist_nm = haversine_nm(lat[last], lon[last], lat[idx], lon[idx])
        speed_kn = dist_nm / (dt / 3600.0)
        if speed_kn <= speed_limit_kn:
            keep.append(idx)
            last = idx
        else:
            removed += 1
    return frame.iloc[keep].reset_index(drop=True), removed


def interpolate_segment(frame, interval_s, min_points):
    if len(frame) < 2:
        return None

    ts = frame["ts"].to_numpy(dtype=np.float64)
    start = int(math.ceil(ts[0] / interval_s) * interval_s)
    end = int(math.floor(ts[-1] / interval_s) * interval_s)
    if end < start:
        return None
    grid = np.arange(start, end + 1, interval_s, dtype=np.float64)
    if len(grid) < min_points:
        return None

    lat = np.interp(grid, ts, frame["lat"].to_numpy(dtype=np.float64))
    lon = np.interp(grid, ts, frame["lon"].to_numpy(dtype=np.float64))
    sog = np.interp(grid, ts, frame["sog"].to_numpy(dtype=np.float64))

    cog_rad = np.unwrap(np.deg2rad(frame["cog"].to_numpy(dtype=np.float64)))
    cog = np.rad2deg(np.interp(grid, ts, cog_rad)) % 360.0
    return grid, lat, lon, sog, cog


def build_itentformer_track(mmsi, length_value, grid, lat, lon, sog, cog):
    cog_rad = np.deg2rad(cog)
    vx = sog * np.sin(cog_rad)
    vy = sog * np.cos(cog_rad)
    data = np.stack(
        [
            np.full(len(grid), float(mmsi), dtype=np.float64),
            np.full(len(grid), float(length_value), dtype=np.float64),
            cog,
            lon,
            lat,
            sog,
            vx,
            vy,
            circular_diff_deg(cog),
            prepend_zero_diff(lon),
            prepend_zero_diff(lat),
            prepend_zero_diff(sog),
            prepend_zero_diff(vx),
            prepend_zero_diff(vy),
            grid,
        ],
        axis=1,
    )
    return data.astype(np.float64)


def gate_hit_indices(track, gate):
    lon = track[:, 3]
    lat = track[:, 4]
    mask = (
        (lon >= gate["lon_min"])
        & (lon <= gate["lon_max"])
        & (lat >= gate["lat_min"])
        & (lat <= gate["lat_max"])
    )
    return np.flatnonzero(mask)


def recompute_track_features(track, start_ts=None, dt=None):
    track = track.copy()
    if len(track) == 0:
        return track

    track[:, 2] = np.mod(track[:, 2], 360.0)
    cog_rad = np.deg2rad(track[:, 2])
    sog = track[:, 5]
    track[:, 6] = sog * np.sin(cog_rad)
    track[:, 7] = sog * np.cos(cog_rad)
    track[:, 8] = circular_diff_deg(track[:, 2])
    track[:, 9] = prepend_zero_diff(track[:, 3])
    track[:, 10] = prepend_zero_diff(track[:, 4])
    track[:, 11] = prepend_zero_diff(track[:, 5])
    track[:, 12] = prepend_zero_diff(track[:, 6])
    track[:, 13] = prepend_zero_diff(track[:, 7])
    if start_ts is not None and dt is not None:
        track[:, 14] = float(start_ts) + np.arange(len(track), dtype=np.float64) * float(dt)
    return track


def reverse_track_to_forward(track, dt):
    reversed_track = track[::-1].copy()
    reversed_track[:, 2] = np.mod(reversed_track[:, 2] + 180.0, 360.0)
    return recompute_track_features(reversed_track, start_ts=track[0, 14], dt=dt)


def classify_forward_route(track, gates, args):
    n_points = len(track)
    if n_points < args.min_output_points:
        return None

    start_limit = int((n_points - 1) * args.route_start_max_fraction)
    end_limit = int((n_points - 1) * args.route_end_min_fraction)

    o_hits = gate_hit_indices(track, gates["O"])
    o_hits = o_hits[o_hits <= start_limit]
    if len(o_hits) < args.min_gate_hits:
        return None
    o_idx = int(o_hits[0])

    ti_hits = gate_hit_indices(track, gates["TI"])
    ti_hits = ti_hits[ti_hits > o_idx]
    if len(ti_hits) < args.min_gate_hits:
        return None
    ti_idx = int(ti_hits[0])

    candidates = []
    for endpoint, route_label in (("A", "OA"), ("B1", "OB1"), ("B2", "OB2")):
        endpoint_hits = gate_hit_indices(track, gates[endpoint])
        endpoint_hits = endpoint_hits[(endpoint_hits > ti_idx) & (endpoint_hits >= end_limit)]
        if len(endpoint_hits) >= args.min_gate_hits:
            candidates.append(
                {
                    "first": int(endpoint_hits[0]),
                    "last": int(endpoint_hits[-1]),
                    "hits": int(len(endpoint_hits)),
                    "route": route_label,
                }
            )

    if not candidates:
        return None
    if args.endpoint_policy == "first_hit":
        selected = min(candidates, key=lambda item: item["first"])
    elif args.endpoint_policy == "last_hit":
        selected = max(candidates, key=lambda item: item["last"])
    elif args.endpoint_policy == "most_hits":
        selected = max(candidates, key=lambda item: (item["hits"], item["last"]))
    else:
        raise ValueError(f"Unsupported endpoint_policy: {args.endpoint_policy}")
    return selected["route"]


def classify_direct_c_route(track, gates, args):
    if not args.include_direct_c_route:
        return None

    n_points = len(track)
    if n_points < args.min_output_points:
        return None

    start_limit = int((n_points - 1) * args.route_start_max_fraction)
    end_limit = int((n_points - 1) * args.route_end_min_fraction)

    o_hits = gate_hit_indices(track, gates["O"])
    o_hits = o_hits[o_hits <= start_limit]
    if len(o_hits) < args.min_gate_hits:
        return None
    o_idx = int(o_hits[0])

    c_hits = gate_hit_indices(track, gates["C"])
    c_hits = c_hits[(c_hits > o_idx) & (c_hits >= end_limit)]
    if len(c_hits) < args.min_gate_hits:
        return None
    return args.direct_c_label.upper()


def merge_route_label(route, merge_ob_routes):
    if not merge_ob_routes:
        return route
    if route in {"OB1", "OB2"}:
        return "OB"
    if route in {"B1O", "B2O"}:
        return "BO"
    return route


def classify_route(track, gates, args):
    route = classify_forward_route(track, gates, args)
    if route is not None:
        return merge_route_label(route, args.merge_ob_routes), track, "forward"

    route = classify_direct_c_route(track, gates, args)
    if route is not None:
        return merge_route_label(route, args.merge_ob_routes), track, "direct_c_forward"

    if args.reverse_mode == "none":
        return None, track, "rejected"

    route = classify_forward_route(track[::-1], gates, args)
    if route is None:
        route = classify_direct_c_route(track[::-1], gates, args)
        if route is None:
            return None, track, "rejected"

    if args.reverse_mode == "separate":
        reverse_label = REVERSE_ROUTE_LABELS.get(route, route)
        return merge_route_label(reverse_label, args.merge_ob_routes), track, "reverse_separate"

    if args.reverse_mode == "normalize":
        normalized_track = reverse_track_to_forward(track, args.dt)
        return merge_route_label(route, args.merge_ob_routes), normalized_track, "reverse_normalized"

    raise ValueError(f"Unsupported reverse_mode: {args.reverse_mode}")


def apply_route_balance(tracks, route_labels, args):
    groups = {}
    for idx, label in enumerate(route_labels):
        groups.setdefault(label, []).append(idx)

    target_counts = parse_balance_counts(args.route_balance_counts)
    if not target_counts:
        labels = sorted(groups)
        if not labels:
            return [], [], {"target_counts": {}, "available_counts": {}, "selected_counts": {}}
        if args.route_balance_strategy == "min_class":
            per_label = min(len(groups[label]) for label in labels)
            target_counts = {label: per_label for label in labels}
        elif args.route_balance_strategy == "total_even":
            per_label = args.target_total_tracks // len(labels)
            target_counts = {label: per_label for label in labels}
        else:
            target_counts = {label: len(groups[label]) for label in labels}

    rng = np.random.default_rng(args.route_random_seed)
    selected = []
    selected_counts = {}
    shortage = {}
    for label, target_count in target_counts.items():
        available = np.array(groups.get(label, []), dtype=np.int64)
        if len(available) == 0:
            selected_counts[label] = 0
            shortage[label] = int(target_count)
            continue
        if len(available) <= target_count:
            chosen = available
            if len(available) < target_count:
                shortage[label] = int(target_count - len(available))
        else:
            chosen = rng.choice(available, size=target_count, replace=False)
        selected.extend(int(item) for item in chosen)
        selected_counts[label] = int(len(chosen))

    selected = sorted(selected)
    balanced_tracks = [tracks[idx] for idx in selected]
    balanced_labels = [route_labels[idx] for idx in selected]
    report = {
        "strategy": args.route_balance_strategy,
        "target_counts": {key: int(value) for key, value in target_counts.items()},
        "available_counts": {key: int(len(value)) for key, value in groups.items()},
        "selected_counts": selected_counts,
        "shortage": shortage,
    }
    return balanced_tracks, balanced_labels, report


def split_by_time_gap(frame, gap_s):
    ts = frame["ts"].to_numpy()
    cuts = np.where(np.diff(ts) > gap_s)[0] + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(frame)]))
    for start, end in zip(starts, ends):
        yield frame.iloc[start:end].reset_index(drop=True)


def process_one_mmsi(mmsi, rows, args, counters, route_gates):
    frame = pd.DataFrame(rows, columns=DB_COLUMNS)
    frame = frame.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    counters["mmsi_groups"] += 1
    counters["deduplicated_points"] += int(len(rows) - len(frame))
    length_value = median_or_default(frame["length"], args.default_length)

    tracks = []
    for segment in split_by_time_gap(frame, args.tg):
        counters["segments_seen"] += 1
        if len(segment) < args.min_raw_points:
            counters["segments_short_raw"] += 1
            continue
        segment, removed = remove_drift_points(segment, args.ve)
        counters["drift_points_removed"] += int(removed)
        if len(segment) < args.min_raw_points:
            counters["segments_short_after_drift"] += 1
            continue
        interpolated = interpolate_segment(segment, args.dt, args.min_output_points)
        if interpolated is None:
            counters["segments_short_after_interp"] += 1
            continue
        grid, lat, lon, sog, cog = interpolated
        track = build_itentformer_track(mmsi, length_value, grid, lat, lon, sog, cog)
        route_label = "ALL"
        if args.route_filter == "ti":
            route_label, track, route_direction = classify_route(track, route_gates, args)
            if route_label is None:
                counters["segments_rejected_by_route"] += 1
                continue
            counters[f"route_direction_{route_direction}"] += 1
        tracks.append((track, route_label))
        counters["segments_kept"] += 1
        counters[f"candidate_route_{route_label}"] += 1
    return tracks


def build_tracks_from_sqlite(args, db_path, output_path):
    conn = create_connection(db_path)
    route_gates = get_route_gates(args)
    cursor = conn.execute(
        """
        SELECT mmsi, ts, lat, lon, sog, cog, heading, length, width, draught, ship_type, source
        FROM ais
        ORDER BY mmsi, ts
        """
    )

    tracks = []
    route_labels = []
    counters = Counter()
    current_mmsi = None
    current_rows = []

    def flush_group():
        nonlocal tracks, current_mmsi, current_rows
        if current_mmsi is None:
            return
        new_tracks = process_one_mmsi(current_mmsi, current_rows, args, counters, route_gates)
        if args.max_tracks_per_mmsi:
            new_tracks = new_tracks[: args.max_tracks_per_mmsi]
        for track, route_label in new_tracks:
            tracks.append(track)
            route_labels.append(route_label)
            counters["candidate_tracks"] += 1
            counters["candidate_points"] += int(len(track))
            if args.max_tracks and not args.balance_routes and len(tracks) >= args.max_tracks:
                break
        current_rows = []
        if args.progress_every_mmsi and counters["mmsi_groups"] % args.progress_every_mmsi == 0:
            route_summary = {
                key.replace("candidate_route_", ""): int(value)
                for key, value in counters.items()
                if key.startswith("candidate_route_")
            }
            print(
                f"[build] mmsi={counters['mmsi_groups']}, segments={counters['segments_seen']}, "
                f"candidates={counters['candidate_tracks']}, rejected_by_route={counters['segments_rejected_by_route']}, "
                f"routes={route_summary}",
                flush=True,
            )

    for row in cursor:
        mmsi = row[0]
        if current_mmsi is None:
            current_mmsi = mmsi
        if mmsi != current_mmsi:
            flush_group()
            if args.max_tracks and not args.balance_routes and len(tracks) >= args.max_tracks:
                break
            current_mmsi = mmsi
        current_rows.append(row)

    if args.balance_routes or not args.max_tracks or len(tracks) < args.max_tracks:
        flush_group()

    conn.close()

    if args.balance_routes:
        tracks, route_labels, balance_report = apply_route_balance(tracks, route_labels, args)
        counters["balance"] = balance_report
    elif args.max_tracks and len(tracks) > args.max_tracks:
        tracks = tracks[: args.max_tracks]
        route_labels = route_labels[: args.max_tracks]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(tracks, output_path)

    labels_path = output_path.parent / args.route_labels_name
    if args.route_filter != "none" or args.balance_routes:
        labels = [
            {
                "index": int(idx),
                "route": route_labels[idx],
                "mmsi": int(track[0, 0]),
                "points": int(len(track)),
                "start_time": int(track[0, 14]),
                "end_time": int(track[-1, 14]),
            }
            for idx, track in enumerate(tracks)
        ]
        with labels_path.open("w", encoding="utf-8") as handle:
            json.dump(labels, handle, indent=2, ensure_ascii=False)
        counters["route_labels_path"] = str(labels_path)

    counters["tracks_kept"] = int(len(tracks))
    counters["points_kept"] = int(sum(len(track) for track in tracks))
    counters["route_counts"] = dict(Counter(route_labels))

    if tracks:
        lengths = np.array([len(track) for track in tracks])
        all_data = np.concatenate(tracks, axis=0)
        counters["track_length"] = {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        }
        counters["ranges"] = {
            name: numeric_summary(all_data[:, idx]) for idx, name in enumerate(ITENTFORMER_COLUMNS)
        }
    else:
        counters["track_length"] = {"min": 0, "mean": 0.0, "max": 0}
        counters["ranges"] = {}
    counters["output_path"] = str(output_path)
    return dict(counters)


def write_report(args, report_path, scan_report, build_report):
    report = {
        "input_zip": str(args.input_zip),
        "output_dir": str(args.output_dir),
        "stage_db": str(Path(args.output_dir) / args.stage_db),
        "output_pickle": str(Path(args.output_dir) / args.output_name),
        "model_columns": ITENTFORMER_COLUMNS,
        "paper_aligned_parameters": {
            "Vth_kn": args.vth,
            "TG_s": args.tg,
            "VE_kn": args.ve,
            "delta_T_s": args.dt,
            "K": 5,
            "wD": 20,
            "sD": args.recommended_window_stride,
        },
        "filters": {
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
            "mobile_types": args.mobile_types,
            "ship_types": args.ship_types,
            "mid_min": args.mid_min,
            "mid_max": args.mid_max,
        },
        "route_filter": {
            "mode": args.route_filter,
            "gates": get_route_gates(args),
            "min_gate_hits": args.min_gate_hits,
            "route_start_max_fraction": args.route_start_max_fraction,
            "route_end_min_fraction": args.route_end_min_fraction,
            "endpoint_policy": args.endpoint_policy,
            "include_direct_c_route": args.include_direct_c_route,
            "direct_c_label": args.direct_c_label,
            "reverse_mode": args.reverse_mode,
            "allow_reverse_routes": args.allow_reverse_routes,
            "merge_ob_routes": args.merge_ob_routes,
            "balance_routes": args.balance_routes,
            "route_balance_strategy": args.route_balance_strategy,
            "target_total_tracks": args.target_total_tracks,
            "route_balance_counts": parse_balance_counts(args.route_balance_counts),
        },
        "notes": [
            "The script reads CSV members directly from the ZIP; no full extraction is required.",
            "The output is a list of arrays with iTentformer columns: MMSI, Length, Course, Lon, Lat, SOG, vx, vy, deltas, UnixTime.",
            "Training should use --window_stride 1 for DMA to match Table I sD=1.",
            "The paper does not release exact Ti/O/A/B1/B2 polygon coordinates; this script uses configurable route gates digitized approximately from Fig. 10.",
        ],
        "scan": scan_report,
        "build": build_report,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def main():
    parser = ArgumentParser(description="Stream-preprocess raw Danish DMA AIS ZIP into iTentformer pkl format.")
    parser.add_argument("--input_zip", default=r"D:\AIS\2023_06_09\aisdk-2023-06.zip")
    parser.add_argument("--output_dir", default="dataset/dma_raw_2023_06")
    parser.add_argument("--stage_db", default="dma_filtered.sqlite")
    parser.add_argument("--output_name", default="dma_itentformer_all.pkl")
    parser.add_argument("--report_name", default="dma_preprocess_report.json")
    parser.add_argument("--stage", choices=["all", "scan", "build"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--lat_min", type=float, default=55.5)
    parser.add_argument("--lat_max", type=float, default=58.0)
    parser.add_argument("--lon_min", type=float, default=10.3)
    parser.add_argument("--lon_max", type=float, default=13.0)
    parser.add_argument("--mobile_types", default="Class A")
    parser.add_argument("--ship_types", default="Cargo,Tanker,Container")
    parser.add_argument("--mid_min", type=int, default=200)
    parser.add_argument("--mid_max", type=int, default=799)

    parser.add_argument("--vth", type=float, default=1.0, help="Speed threshold in knots from the paper.")
    parser.add_argument("--tg", type=int, default=300, help="Trajectory split time gap in seconds from the paper.")
    parser.add_argument("--ve", type=float, default=30.0, help="Drift-point empirical speed threshold in knots.")
    parser.add_argument("--dt", type=int, default=900, help="Interpolation interval in seconds for DMA.")
    parser.add_argument("--min_raw_points", type=int, default=100)
    parser.add_argument("--min_output_points", type=int, default=20)
    parser.add_argument("--default_length", type=float, default=0.0)

    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--commit_every_chunks", type=int, default=4)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_chunks", type=int, default=0)
    parser.add_argument("--max_tracks", type=int, default=0)
    parser.add_argument("--max_tracks_per_mmsi", type=int, default=0)
    parser.add_argument("--recommended_window_stride", type=int, default=1)

    parser.add_argument("--route_filter", choices=["none", "ti"], default="none")
    parser.add_argument("--gate_o", default=DEFAULT_ROUTE_GATES["O"])
    parser.add_argument("--gate_ti", default=DEFAULT_ROUTE_GATES["TI"])
    parser.add_argument("--gate_a", default=DEFAULT_ROUTE_GATES["A"])
    parser.add_argument("--gate_b1", default=DEFAULT_ROUTE_GATES["B1"])
    parser.add_argument("--gate_b2", default=DEFAULT_ROUTE_GATES["B2"])
    parser.add_argument("--gate_c", default=DEFAULT_ROUTE_GATES["C"])
    parser.add_argument("--min_gate_hits", type=int, default=1)
    parser.add_argument("--route_start_max_fraction", type=float, default=0.45)
    parser.add_argument("--route_end_min_fraction", type=float, default=0.55)
    parser.add_argument(
        "--endpoint_policy",
        choices=["first_hit", "last_hit", "most_hits"],
        default="first_hit",
        help=(
            "How to choose among A/B1/B2 when a trajectory enters multiple endpoint gates. "
            "first_hit matches the earlier behavior; most_hits is usually better when upstream gates overlap."
        ),
    )
    parser.add_argument(
        "--include_direct_c_route",
        action="store_true",
        help="Also keep direct west-side O->C trajectories that do not pass through Ti.",
    )
    parser.add_argument("--direct_c_label", default="OC")
    parser.add_argument(
        "--reverse_mode",
        choices=["none", "separate", "normalize"],
        default="none",
        help=(
            "How to handle reverse routes such as A->Ti->O. "
            "none rejects them, separate keeps AO/B1O/B2O labels, "
            "normalize flips them to O->Ti->A/B and recomputes motion features."
        ),
    )
    parser.add_argument("--allow_reverse_routes", action="store_true")
    parser.add_argument("--merge_ob_routes", action="store_true")
    parser.add_argument("--balance_routes", action="store_true")
    parser.add_argument("--route_balance_strategy", choices=["min_class", "total_even", "none"], default="min_class")
    parser.add_argument("--target_total_tracks", type=int, default=132)
    parser.add_argument("--route_balance_counts", default=DEFAULT_ROUTE_BALANCE)
    parser.add_argument("--route_random_seed", type=int, default=42)
    parser.add_argument("--route_labels_name", default="dma_route_labels.json")
    parser.add_argument("--progress_every_mmsi", type=int, default=100)
    args = parser.parse_args()

    if not 0.0 <= args.route_start_max_fraction <= 1.0:
        raise ValueError("--route_start_max_fraction must be between 0 and 1.")
    if not 0.0 <= args.route_end_min_fraction <= 1.0:
        raise ValueError("--route_end_min_fraction must be between 0 and 1.")
    if args.balance_routes and args.route_filter == "none":
        raise ValueError("--balance_routes requires --route_filter ti.")
    if args.allow_reverse_routes and args.reverse_mode == "none":
        args.reverse_mode = "separate"
    get_route_gates(args)

    args.input_zip = Path(args.input_zip)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / args.stage_db
    output_path = args.output_dir / args.output_name
    report_path = args.output_dir / args.report_name

    scan_report = {}
    build_report = {}
    if args.stage in {"all", "scan"}:
        scan_report = scan_zip_to_sqlite(args, db_path)
    if args.stage in {"all", "build"}:
        if not db_path.exists():
            raise FileNotFoundError(f"Stage database not found: {db_path}")
        build_report = build_tracks_from_sqlite(args, db_path, output_path)

    write_report(args, report_path, scan_report, build_report)
    print(f"report saved to {report_path}")
    if build_report:
        print(f"tracks={build_report.get('tracks_kept', 0)}, points={build_report.get('points_kept', 0)}")
        print(f"pkl saved to {output_path}")


if __name__ == "__main__":
    main()
