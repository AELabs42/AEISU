"""CPU-only Area smoothing and paired Native-minus-Area controls on frozen outputs."""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np

from common import read_rgb, sha256, write_json
from final_benchmark_analysis import point_weights, sample_weights
from luma_metrics import moments, y_channel
from pair_consistency import gaussian

ARCHIVE = Path("G:/Output Images/QRISP Final Benchmark 2026-10-04")
OUTPUT = Path(__file__).resolve().parents[2] / "output/peer_review_controls/2026-10-06"
METHODS = ("aif_mixed", "fsr1_easu")
SIGMAS = (0.5, 1.0, 2.0)


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def y_scores(prediction, target, target_mean, target_variance):
    value = y_channel(prediction)[2:-2, 2:-2]
    mse = float(np.mean((value - target) ** 2))
    mean, variance = moments(value)
    covariance = cv2.GaussianBlur(value * target, (11, 11), 1.5)[5:-5, 5:-5] - mean * target_mean
    score = (2 * mean * target_mean + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
    score /= (mean * mean + target_mean * target_mean + 0.01 ** 2) * (variance + target_variance + 0.03 ** 2)
    return {"y_psnr_shave2_db": float(-10 * np.log10(mse)), "y_ssim_shave2": float(score.mean())}


def area_blur(task):
    archive, row = task
    path = Path(row["outputs"]["npz"])
    if sha256(path) != row["outputs"]["npz_sha256"]:
        raise ValueError(f"Changed prediction: {path}")
    target_path = archive / "targets" / row["image"]
    if sha256(target_path) != row["target_sha256"]:
        raise ValueError(f"Changed target: {target_path}")
    with np.load(path, allow_pickle=False) as stored:
        prediction = stored["rgb"]
    target = y_channel(read_rgb(target_path).astype(np.float32) / 255)[2:-2, 2:-2]
    mean, variance = moments(target)
    baseline = y_scores(prediction, target, mean, variance)
    for key, value in baseline.items():
        if not np.isclose(value, row["metrics"][key], atol=1e-10, rtol=0):
            raise ValueError(f"Baseline mismatch: {row['image']}/{key}")
    results = []
    for sigma in SIGMAS:
        metrics = y_scores(gaussian(prediction, sigma), target, mean, variance)
        results.append({key: row[key] for key in ("image", "scene", "segment", "group", "method")} |
                       {"sigma": sigma, "metrics": metrics, "baseline_metrics": baseline,
                        "output_sha256": row["outputs"]["npz_sha256"], "target_sha256": row["target_sha256"]})
    return results


def balanced(rows, key, group="all"):
    rows = [r for r in rows if group == "all" or r["group"] == group]
    scenes = sorted({r["scene"] for r in rows})
    return float(np.mean([np.mean([key(r) for r in rows if r["scene"] == scene]) for scene in scenes]))


def summarize(archive, output, pairs, source, area_rows):
    blur = []
    native = read_lines(archive / "blur_per_image.jsonl")
    for condition, rows in (("ssaa_area", area_rows), ("native", native)):
        for method in METHODS:
            for sigma in SIGMAS:
                chosen = [r for r in rows if r["method"] == method and r["sigma"] == sigma
                          and (condition == "ssaa_area" or r["condition"] == condition)]
                if len(chosen) != len(pairs):
                    raise ValueError("Incomplete blur control")
                blur.append({"condition": condition, "method": method, "sigma": sigma, "images": len(chosen),
                             **{f"delta_{key}": balanced(chosen, lambda r, key=key: r["metrics"][key] - r["baseline_metrics"][key])
                                for key in ("y_psnr_shave2_db", "y_ssim_shave2")}})
    index = {(r["method"], r["condition"], r["image"]): r for r in source}
    differences = []
    for pair in pairs:
        def gap(condition):
            return (index[METHODS[0], condition, pair["image"]]["metrics"]["y_psnr_shave2_db"] -
                    index[METHODS[1], condition, pair["image"]]["metrics"]["y_psnr_shave2_db"])
        differences.append({key: pair[key] for key in ("image", "scene", "segment", "group", "pair_y_mse", "lr_gradient_energy")} |
                           {"native_gap_db": gap("native"), "area_gap_db": gap("ssaa_area"),
                            "native_minus_area_gap_db": gap("native") - gap("ssaa_area")})
    _, boot = sample_weights(pairs, 10000, 20261006)
    effects = []
    for group in ("all", "most_similar", "middle", "least_similar"):
        weight = point_weights(pairs, group)
        values = np.array([r["native_minus_area_gap_db"] for r in differences])
        if group == "all":
            samples = boot["all"] @ values
        else:
            indices = [i for i, p in enumerate(pairs) if p["group"] == group]
            _, group_boot = sample_weights([pairs[i] for i in indices], 10000, 20261006)
            samples = group_boot["all"] @ values[indices]
        effects.append({"group": group, "native_gap_db": balanced(differences, lambda r: r["native_gap_db"], group),
                        "area_gap_db": balanced(differences, lambda r: r["area_gap_db"], group),
                        "native_minus_area_gap_db": float(weight @ values),
                        "nominal_95_interval": np.quantile(samples, [.025, .975]).tolist()})
    result = {"blur": blur, "paired_gap": effects, "per_image_gap": differences,
              "protocol": {"cpu_only": True, "pairs": len(pairs), "new_inference": False,
                           "blur": "RGB float32 Gaussian sigma .5/1/2, radius ceil(3 sigma), replicate border; unchanged target",
                           "metrics": "Encoded-Y PSNR/SSIM, float64 metrics, two-HR-pixel crop; no new LPIPS computation",
                           "aggregation": "Equal environment means; paired 10000 environment/sequence bootstrap draws",
                           "interpretation": "Descriptive controls, not attribution of rendering mismatch or training effects",
                           "source_sha256": sha256(Path(__file__)), "selection_sha256": sha256(archive / "selection.json")}}
    write_json(output / "summary.json", result)
    lines = ["# Saved-output peer-review controls", "", "CPU-only reuse of frozen reconstructions; original benchmark archive unchanged.",
             "The Area extension measures PSNR and SSIM. It does not add LPIPS or a new perceptual-quality claim.", "",
             "| Condition | Model | Sigma | Delta PSNR | Delta SSIM |", "|---|---|---:|---:|---:|"]
    for r in blur:
        lines.append(f"| {r['condition']} | {r['method']} | {r['sigma']} | {r['delta_y_psnr_shave2_db']:+.6f} | "
                     f"{r['delta_y_ssim_shave2']:+.6f} |")
    lines += ["", "Paired per-frame difference: (AEISU - EASU on Native) - (AEISU - EASU on Area).",
              "This removes each pair's Area comparator gap, not every confound associated with texture and aliasing.", "",
              "| Stratum | Native gap | Area gap | Difference | Nominal 95% interval |", "|---|---:|---:|---:|---|"]
    for r in effects:
        lo, hi = r["nominal_95_interval"]
        lines.append(f"| {r['group']} | {r['native_gap_db']:+.6f} | {r['area_gap_db']:+.6f} | "
                     f"{r['native_minus_area_gap_db']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    pairs = json.loads((args.archive / "selection.json").read_text())["pairs"]
    source = [r for method in METHODS for r in read_lines(args.archive / "results" / method / "per_image.jsonl")]
    chosen = [r for r in source if r["condition"] == "ssaa_area"]
    if len(chosen) != len(pairs) * len(METHODS):
        raise ValueError("Incomplete source archive")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "area_blur_per_image.jsonl"
    rows = read_lines(path) if path.exists() else []
    done = {(r["method"], r["image"]) for r in rows if r["sigma"] == SIGMAS[-1]}
    tasks = [(args.archive, r) for r in chosen if (r["method"], r["image"]) not in done]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, path.open("a", encoding="utf-8") as stream:
        for index, batch in enumerate(pool.map(area_blur, tasks), 1):
            for row in batch:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.extend(batch)
            if index % 25 == 0:
                print(f"Area blur: {len(done) + index}/{len(chosen)} saved outputs scored", flush=True)
    summarize(args.archive, args.output, pairs, source, rows)
    print(f"Completed controls: {args.output}", flush=True)


if __name__ == "__main__":
    main()
