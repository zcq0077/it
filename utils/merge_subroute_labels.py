import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge over-segmented subroute labels without modifying the source file."
    )
    parser.add_argument("--input", required=True, help="Source subroute label JSON.")
    parser.add_argument("--mapping", required=True, help="JSON file containing a 'mapping' object.")
    parser.add_argument("--output", required=True, help="Output label JSON.")
    parser.add_argument("--report", required=True, help="Output merge report JSON.")
    return parser.parse_args()


def route_from_subroute(label):
    label = str(label)
    if "_S" in label:
        return label.split("_S", 1)[0]
    return label.split("_", 1)[0]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    rows = load_json(args.input)
    mapping_config = load_json(args.mapping)
    mapping = mapping_config.get("mapping", mapping_config)

    if not isinstance(rows, list) or not all(isinstance(item, dict) for item in rows):
        raise ValueError("Input labels must be a JSON array of objects.")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Mapping must be a non-empty JSON object.")

    source_counts = Counter(str(item["subroute"]) for item in rows)
    missing = sorted(set(source_counts) - set(mapping))
    unused = sorted(set(mapping) - set(source_counts))
    if missing:
        raise ValueError(f"Mapping is missing source labels: {missing}")

    merged_rows = []
    changed = 0
    parent_mismatches = []
    version = str(mapping_config.get("name", Path(args.mapping).stem))
    for index, item in enumerate(rows):
        source = str(item["subroute"])
        target = str(mapping[source])
        parent = str(item.get("parent_route", item.get("route", "")))
        if route_from_subroute(target) != parent:
            parent_mismatches.append(
                {"index": index, "parent_route": parent, "source": source, "target": target}
            )
            continue

        merged = dict(item)
        merged["source_subroute"] = source
        merged["subroute"] = target
        merged["subroute_merge_version"] = version
        merged_rows.append(merged)
        changed += int(source != target)

    if parent_mismatches:
        raise ValueError(f"Target labels cross parent routes: {parent_mismatches[:5]}")

    target_counts = Counter(item["subroute"] for item in merged_rows)
    report = {
        "name": version,
        "description": mapping_config.get("description"),
        "input": str(Path(args.input)),
        "mapping_file": str(Path(args.mapping)),
        "output": str(Path(args.output)),
        "track_count": len(merged_rows),
        "changed_track_count": changed,
        "source_class_count": len(source_counts),
        "target_class_count": len(target_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "mapping": dict(sorted((str(key), str(value)) for key, value in mapping.items())),
        "unused_mapping_labels": unused,
        "parent_route_consistent": True,
    }

    write_json(args.output, merged_rows)
    write_json(args.report, report)
    print(
        f"Merged {len(merged_rows)} tracks from {len(source_counts)} to "
        f"{len(target_counts)} subroute classes: {dict(sorted(target_counts.items()))}"
    )


if __name__ == "__main__":
    main()
