"""Recompute all quality leaderboard means from the packaged immutable score rows."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "quality")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    selection = json.loads((args.data / "selection.json").read_text())["pairs"]
    expected_images = {item["image"] for item in selection}
    summary = json.loads((args.data / "summary.json").read_text())
    rows, total, greatest_error = [], 0, 0.0
    for score_file in sorted((args.data / "results").glob("*/per_image.jsonl.gz")):
        method = score_file.parent.name
        with gzip.open(score_file, "rt", encoding="utf-8") as stream:
            scores = [json.loads(line) for line in stream if line.strip()]
        total += len(scores)
        identities = {(item["condition"], item["image"]) for item in scores}
        if len(scores) != 3000 or len(identities) != 3000:
            raise ValueError(f"Incomplete or duplicated rows: {method}")
        for condition in summary["by_condition"]:
            chosen = [item for item in scores if item["condition"] == condition]
            if {item["image"] for item in chosen} != expected_images:
                raise ValueError(f"Selection differs for {condition}/{method}")
            for group in summary["by_condition"][condition]:
                subset = [item for item in chosen if group == "all" or item["group"] == group]
                record = {"condition": condition, "group": group, "method": method, "images": len(subset)}
                reference = summary["by_condition"][condition][group][method]["scene_balanced"]
                for metric in reference:
                    scenes = defaultdict(list)
                    for item in subset:
                        scenes[item["scene"]].append(item["metrics"][metric])
                    value = float(np.mean([np.mean(values) for values in scenes.values()]))
                    error = abs(value - reference[metric])
                    greatest_error = max(greatest_error, error)
                    if error > 1e-10:
                        raise ValueError(f"Aggregate mismatch: {condition}/{group}/{method}/{metric} ({error})")
                    record[metric] = value
                rows.append(record)
    if total != summary["expected"] or total != 39000:
        raise ValueError(f"Wrong total: {total}")
    args.output.mkdir(parents=True)
    with (args.output / "leaderboards.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"score_rows": total, "aggregate_rows": len(rows), "selection_pairs": len(selection),
              "methods": len({item["method"] for item in rows}), "max_abs_aggregate_error": greatest_error,
              "aggregation": "Mean per-image metric within each environment, then equal environment weights",
              "scope": "CPU score aggregation; no inference, target images, GPU or new selection"}
    (args.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
