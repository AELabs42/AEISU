"""Independent completeness, archive samples and saved-metric reproduction."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
import cv2
import numpy as np
import torch
sys.path.insert(0, str(HERE))
from common import read_rgb, sha256, write_json
from final_benchmark_metrics import PerceptualMetric, distortion


def verify_arrays(row, target_path):
    outputs = row["outputs"]
    for kind in ("npz", "png"):
        if sha256(Path(outputs[kind])) != outputs[kind + "_sha256"]:
            raise ValueError("Changed sampled output")
    with np.load(outputs["npz"], allow_pickle=False) as archive:
        prediction = archive["rgb"]
    target = read_rgb(target_path).astype(np.float32) / 255
    if prediction.dtype != np.float32 or prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid sampled array")
    expected = np.floor(np.clip(prediction, 0, 1) * 255 + .5).astype(np.uint8)
    if not np.array_equal(read_rgb(Path(outputs["png"])), expected):
        raise ValueError("Preview differs from recorded rounding")
    scores = distortion(prediction, target)
    differences = {key: abs(value - row["metrics"][key]) for key, value in scores.items()}
    if max(differences.values()) > 2e-10:
        raise ValueError("Distortion metric mismatch")
    return prediction, target, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("G:/Output Images/AEISU Core Ablations 2026-10-05/evaluation_v2"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    protocol = json.loads((args.output / "protocol.json").read_text())
    selection_path = Path(protocol["selection"])
    if sha256(selection_path) != protocol["selection_sha256"]:
        raise ValueError("Source selection changed")
    for model in protocol["models"].values():
        if sha256(Path(model["checkpoint"])) != model["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
    rows = [json.loads(line) for line in (args.output / "per_image.jsonl").read_text().splitlines() if line.strip()]
    identities = {(r["method"], r["precision"], r["condition"], r["image"]) for r in rows}
    selection = json.loads(selection_path.read_text())
    slots = protocol.get("slots") or [(m, "fp16", c) for m in protocol["models"] for c in protocol["conditions"]]
    if "slots" not in protocol:
        slots += [(m, "fp8_stress", "ssaa_area") for m in protocol["models"]]
    expected = {(*slot, pair["image"]) for slot in slots for pair in selection["pairs"]}
    if len(rows) != len(expected) or identities != expected:
        raise ValueError("Incomplete/duplicate identities")
    groups = defaultdict(list)
    for row in rows:
        for kind in ("npz", "png"):
            path = Path(row["outputs"][kind])
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Missing output file")
        groups[row["method"], row["precision"], row["condition"]].append(row)
    if len(groups) != len(slots) or any(len(v) != len(selection["pairs"]) for v in groups.values()):
        raise ValueError("Unexpected condition coverage")
    samples = [row for values in groups.values() for row in (values[0], values[-1])]
    archive_root = selection_path.parent
    metric = PerceptualMetric()
    maximum = {}
    lpips_error = 0.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for first in range(0, len(samples), 4):
            batch = samples[first:first + 4]
            futures = [pool.submit(verify_arrays, r, archive_root / "targets" / r["image"]) for r in batch]
            for row, future in zip(batch, futures):
                prediction, target, errors = future.result()
                for key, value in errors.items():
                    maximum[key] = max(maximum.get(key, 0.), value)
                error = abs(metric.score(prediction, target) - row["metrics"]["lpips_alex_v01"])
                lpips_error = max(lpips_error, error)
                if error > 1e-5:
                    raise ValueError(f"LPIPS reproduction failed: {error}")
            print(f"Verified archive samples {min(first + 4, len(samples))}/{len(samples)}", flush=True)
    record = {"passed": True, "records": len(rows), "all_output_files_present": True, "output_files": len(rows) * 2,
              "hash_decode_preview_and_metric_samples": len(samples), "sample_policy": f"First and last identity in all {len(slots)} model/precision/condition slots",
              "max_distortion_absolute_differences": maximum, "max_lpips_absolute_difference": lpips_error,
              "lpips_tolerance": 1e-5, "checkpoint_and_selection_hashes_match": True,
              "metric_source_sha256": sha256(HERE / "final_benchmark_metrics.py"), "verifier_sha256": sha256(Path(__file__))}
    write_json(args.output / "verification.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
