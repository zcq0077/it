from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd

from preprocess_dma_zip import DEFAULT_ROUTE_GATES, classify_route, get_route_gates


def summarize(tracks, route_labels):
    lengths = np.array([len(track) for track in tracks], dtype=int)
    return {
        "tracks": int(len(tracks)),
        "points": int(lengths.sum()) if len(lengths) else 0,
        "route_counts": dict(Counter(route_labels)),
        "track_length": {
            "min": int(lengths.min()) if len(lengths) else 0,
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
        },
    }


def main():
    parser = ArgumentParser(description="Classify existing iTentformer all-track pkl into route-filtered classes.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_labels_path", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--source_name", default="")

    parser.add_argument("--gate_o", default=DEFAULT_ROUTE_GATES["O"])
    parser.add_argument("--gate_ti", default=DEFAULT_ROUTE_GATES["TI"])
    parser.add_argument("--gate_a", default=DEFAULT_ROUTE_GATES["A"])
    parser.add_argument("--gate_b1", default=DEFAULT_ROUTE_GATES["B1"])
    parser.add_argument("--gate_b2", default=DEFAULT_ROUTE_GATES["B2"])
    parser.add_argument("--gate_c", default=DEFAULT_ROUTE_GATES["C"])
    parser.add_argument("--min_gate_hits", type=int, default=1)
    parser.add_argument("--route_start_max_fraction", type=float, default=0.45)
    parser.add_argument("--route_end_min_fraction", type=float, default=0.55)
    parser.add_argument("--endpoint_policy", choices=["first_hit", "last_hit", "most_hits"], default="last_hit")
    parser.add_argument("--include_direct_c_route", action="store_true")
    parser.add_argument("--direct_c_label", default="OC")
    parser.add_argument("--reverse_mode", choices=["none", "separate", "normalize"], default="normalize")
    parser.add_argument("--merge_ob_routes", action="store_true")
    parser.add_argument("--dt", type=int, default=900)
    parser.add_argument("--min_output_points", type=int, default=20)
    args = parser.parse_args()

    route_args = Namespace(
        gate_o=args.gate_o,
        gate_ti=args.gate_ti,
        gate_a=args.gate_a,
        gate_b1=args.gate_b1,
        gate_b2=args.gate_b2,
        gate_c=args.gate_c,
        min_gate_hits=args.min_gate_hits,
        route_start_max_fraction=args.route_start_max_fraction,
        route_end_min_fraction=args.route_end_min_fraction,
        endpoint_policy=args.endpoint_policy,
        include_direct_c_route=args.include_direct_c_route,
        direct_c_label=args.direct_c_label,
        reverse_mode=args.reverse_mode,
        merge_ob_routes=args.merge_ob_routes,
        dt=args.dt,
        min_output_points=args.min_output_points,
    )
    gates = get_route_gates(route_args)

    source_tracks = pd.read_pickle(args.data_path)
    output_tracks = []
    output_labels = []
    counters = Counter()
    source_name = args.source_name or Path(args.data_path).parent.name

    for source_index, track in enumerate(source_tracks):
        route_label, classified_track, route_direction = classify_route(track, gates, route_args)
        if route_label is None:
            counters["rejected"] += 1
            continue
        output_tracks.append(classified_track)
        output_labels.append(
            {
                "index": len(output_labels),
                "route": route_label,
                "mmsi": int(classified_track[0, 0]),
                "points": int(len(classified_track)),
                "start_time": int(classified_track[0, 14]),
                "end_time": int(classified_track[-1, 14]),
                "source": source_name,
                "source_index": int(source_index),
                "route_direction": route_direction,
            }
        )
        counters[f"route_direction_{route_direction}"] += 1
        counters[f"route_{route_label}"] += 1

    output_path = Path(args.output_path)
    labels_path = Path(args.output_labels_path)
    report_path = Path(args.report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(output_tracks, output_path)
    labels_path.write_text(json.dumps(output_labels, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "data_path": args.data_path,
        "output_path": str(output_path),
        "output_labels_path": str(labels_path),
        "route_filter": {
            "gates": gates,
            "endpoint_policy": args.endpoint_policy,
            "include_direct_c_route": args.include_direct_c_route,
            "direct_c_label": args.direct_c_label,
            "reverse_mode": args.reverse_mode,
        },
        "source_tracks": int(len(source_tracks)),
        "counters": dict(counters),
        "summary": summarize(output_tracks, [item["route"] for item in output_labels]),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"pkl saved to {output_path}")
    print(f"labels saved to {labels_path}")
    print(f"report saved to {report_path}")


if __name__ == "__main__":
    main()
