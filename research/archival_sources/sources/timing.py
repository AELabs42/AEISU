"""Measure the frozen native runtime without modifying the deployed application."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

import cv2
import numpy as np

from common import (RESULTS, ROOT, clear_native_scratch, native, provenance, read_rgb,
                    sha256, write_json)


def gpu_status():
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,utilization.gpu,"
        "clocks.current.graphics,clocks.current.memory,power.draw,power.limit",
        "--format=csv"], capture_output=True, text=True, check=True, timeout=15)
    return {"time_unix": time.time(), "nvidia_smi": result.stdout.strip()}


def parse(log):
    total = re.search(r"runtime PASS \d+x\d+ FP8=([\d.]+)ms FP16=([\d.]+)ms Mixed=([\d.]+)ms", log)
    p95 = re.search(r"CUDA p95: FP8=([\d.]+)ms FP16=([\d.]+)ms Mixed=([\d.]+)ms", log)
    if total is None or p95 is None:
        raise ValueError("Missing native measurements")
    result = {name: {"median_ms": float(total[i + 1]), "p95_ms": float(p95[i + 1])}
              for i, name in enumerate(("fp8", "fp16", "mixed"))}
    for name, label in (("fp8", "FP8"), ("fp16", "FP16"), ("mixed", "Mixed")):
        match = re.search(label + r" stages: (.+)", log)
        if match is None:
            continue
        values = {key: float(value) for key, value in re.findall(r"(\w+)=([\d.]+)ms", match[1])}
        groups = [float(value) for value in re.search(r"groups=\[([^\]]+)\]", match[1])[1].split(",")]
        result[name]["stages_ms"] = {
            "analysis_and_project": values["feature_pack"] + groups[0],
            "compress_and_four_branch_pairs": groups[1],
            "expand_shuffle_skip": values["expand"],
            "refine_difference_of_medians": values["refine"],
            "coefficient_head": values["tail"],
        }
    reconstruction = re.search(r"Reconstruction: FP8=([\d.]+)ms FP16=([\d.]+)ms Mixed=([\d.]+)ms", log)
    if reconstruction:
        for i, name in enumerate(("fp8", "fp16", "mixed")):
            result[name]["stages_ms"]["reconstruction"] = float(reconstruction[i + 1])
    return result


def summarize(rows):
    summary = {}
    for resolution in sorted({row["resolution"] for row in rows}):
        summary[resolution] = {}
        for mode in ("minimal", "detailed"):
            trials = [row for row in rows if row["resolution"] == resolution and row["mode"] == mode]
            if not trials:
                continue
            summary[resolution][mode] = {}
            for precision in ("fp16", "fp8", "mixed"):
                values = [row["results"][precision] for row in trials]
                medians = [value["median_ms"] for value in values]
                record = {"trials": len(values), "median_of_medians_ms": statistics.median(medians),
                          "median_range_ms": [min(medians), max(medians)],
                          "median_of_p95_ms": statistics.median(value["p95_ms"] for value in values)}
                if mode == "detailed":
                    record["stages_ms"] = {stage: statistics.median(value["stages_ms"][stage] for value in values)
                                           for stage in values[0]["stages_ms"]}
                summary[resolution][mode][precision] = record
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--output", type=Path, default=RESULTS / "timing_stock")
    parser.add_argument("--heights", type=int, nargs="+", default=[720, 1080, 1440])
    args = parser.parse_args()
    output = args.output.resolve()
    shapes_requested = [(height * 16 // 9, height) for height in args.heights]
    if any(height not in (540, 720, 1080, 1440) for height in args.heights):
        raise ValueError("Supported timing heights: 540, 720, 1080, 1440")
    if (output / "trials.json").exists():
        raise FileExistsError("Timing run already exists; preserve it and select a new run directory for a repeat")
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/Benchmark Data/Urban100/HR/img001.png"
    image = read_rgb(source).astype(np.float32) / 255
    write_json(output / "protocol.json", provenance() | {
        "gpu_user_description": "ASUS TUF RTX 5090 OC; current settings recorded below, not modified by this script",
        "clock_or_power_changes": False, "hardware_start": gpu_status(),
        "input": str(source), "input_sha256": sha256(source),
        "input_preparation": "RGB/255 resized with OpenCV INTER_LINEAR to each LR timing shape; quality evaluated separately",
        "warmup_per_precision_per_trial": args.warmup, "iterations_per_precision_per_trial": args.iterations,
        "precision_order": ["fp8", "fp16", "mixed"], "order_caveat": "Existing native harness has fixed precision order",
        "metric": "CUDA event duration; no D3D readback, allocation, file I/O, capture, or presentation in interval",
        "timing_boundary": "native event0 to event16; AFTER external input fence wait, BEFORE output fence signal; preprocessing+controller+reconstruction",
        "aggregation": "median of trial medians; median of trial p95; run ranges retained",
        "stage_caveat": "Detailed events add overhead. Refine is difference of native group/expand medians and INCLUDES the head when fused. "
                        "The small coefficient_head field is only the remaining event gap for mixed/FP16, not a head compute measurement.",
        "resolutions": shapes_requested,
        "harness_sha256": sha256(Path(__file__)),
    })
    rows = []
    for trial in range(args.trials):
        shapes = shapes_requested.copy()
        if trial % 2:
            shapes.reverse()
        for width, height in shapes:
            rgb = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            for mode in ("minimal", "detailed"):
                directory = output / f"{width}x{height}_{mode}_{trial + 1}"
                before = gpu_status()
                _, log = native(rgb, directory, warmup=args.warmup, iterations=args.iterations,
                                detailed=mode == "detailed", output=False)
                record = {"resolution": f"{width}x{height}", "mode": mode, "trial": trial + 1,
                          "before": before, "after": gpu_status(), "results": parse(log)}
                rows.append(record)
                clear_native_scratch(directory)
                write_json(output / "trials.json", rows)
                write_json(output / "summary.json", summarize(rows))
                print(json.dumps({key: record[key] for key in ("resolution", "mode", "trial", "results")}), flush=True)


if __name__ == "__main__":
    main()
