"""Benchmark final DIV2K L1 weights on the unchanged classic 219-image protocol."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["TORCH_HOME"] = str(ROOT / "data/benchmark_assets/torch")

from evaluate_classic import (
    ControlAdapter, METHODS, PerceptualMetric, SETS, distortion, freeze, np,
    pairs, read_rgb, save_rgb, sha256, torch, write_json,
)

CAMPAIGN = ROOT / "output/training_campaigns/div2k_l1_3000_2026-10-17"
BASELINE = ROOT / "output/training_campaigns/fixed_stride_2026-10-16/evaluations/div2k_classic"
OUTPUT = CAMPAIGN / "evaluations/classic_219"
NAME = "aeisu_div2k_l1_3000_fp16"
KEYS = ("y_psnr_shave2_db", "y_ssim_shave2", "lpips_alex_v01")


def baseline_means(records, metric_protocol):
    protocol = json.loads((BASELINE / "protocol.json").read_text())
    verification = json.loads((BASELINE / "verification.json").read_text())
    summary = json.loads((BASELINE / "summary.json").read_text())
    if not verification["passed"] or not summary["complete"]:
        raise ValueError("Comparator archive is not verified and complete")
    if protocol["images"] != records or protocol["metrics"] != metric_protocol:
        raise ValueError("Comparator images or metric protocol differ")
    rows = [json.loads(line) for line in (BASELINE / "per_image.jsonl").read_text().splitlines() if line.strip()]
    expected = {(method, r["dataset"], r["hr_filename"]) for method in METHODS for r in records}
    if len(rows) != len(expected) or {(r["method"], r["dataset"], r["image"]) for r in rows} != expected:
        raise ValueError("Comparator archive identity coverage differs")
    images = {(r["dataset"], r["hr_filename"]): r for r in records}
    for row in rows:
        source = images[row["dataset"], row["image"]]
        if (row["input_sha256"], row["target_sha256"]) != (source["lr_sha256"], source["hr_sha256"]):
            raise ValueError("Comparator source hashes differ")
    for dataset in SETS:
        for method in METHODS:
            group = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            for key, value in summary["means"][dataset][method].items():
                if abs(float(np.mean([row["metrics"][key] for row in group])) - value) > 1e-12:
                    raise ValueError("Archived comparator aggregate differs from its per-image scores")
    return summary["means"]


def verify_outputs(rows, records, metric):
    images = {(r["dataset"], r["hr_filename"]): r for r in records}
    for row in rows:
        for kind in ("npz", "png"):
            if sha256(Path(row["outputs"][kind])) != row["outputs"][kind + "_sha256"]:
                raise ValueError("Saved output hash mismatch")
    maximum_distortion = maximum_lpips = 0.
    samples = []
    for dataset in SETS:
        group = [r for r in rows if r["dataset"] == dataset]
        samples += [group[0], group[-1]]
    for row in samples:
        record = images[row["dataset"], row["image"]]
        with np.load(row["outputs"]["npz"], allow_pickle=False) as archive:
            prediction = archive["rgb"]
        high = read_rgb(Path(record["hr_path"])).astype(np.float32) / 255
        high = high[:prediction.shape[0], :prediction.shape[1]]
        expected_preview = np.floor(np.clip(prediction, 0, 1) * 255 + .5).astype(np.uint8)
        if not np.array_equal(read_rgb(Path(row["outputs"]["png"])), expected_preview):
            raise ValueError("Saved PNG preview rounding mismatch")
        scores = distortion(prediction, high)
        maximum_distortion = max(maximum_distortion, *(abs(value - row["metrics"][key]) for key, value in scores.items()))
        maximum_lpips = max(maximum_lpips, abs(metric.score(prediction, high) - row["metrics"]["lpips_alex_v01"]))
    if maximum_distortion > 2e-10 or maximum_lpips > 1e-5:
        raise ValueError("Saved-output metric reproduction failed")
    return {"passed": True, "records": len(rows), "all_output_hashes_verified": True,
            "saved_metric_samples": len(samples), "max_distortion_difference": maximum_distortion,
            "max_lpips_difference": maximum_lpips}


def main():
    if torch.version.cuda != "13.0":
        raise RuntimeError("Torch CUDA 13.0 required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    state = json.loads((CAMPAIGN / "state.json").read_text())
    checkpoint = Path(state["checkpoint"])
    if state["phase"] != "complete" or sha256(checkpoint) != state["checkpoint_sha256"]:
        raise ValueError("Final L1 training checkpoint is not verified complete")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg, experiment = saved["config"], saved["experiment"]
    if (cfg["mse_loss_weight"], cfg["l1_loss_weight"], cfg["lr_schedule_total_steps"]) != (0., 1., 915000):
        raise ValueError("Wrong final training objective or schedule")
    if experiment["initial_checkpoint"] or experiment["backend"] != "fused_cuda_16x5":
        raise ValueError("Expected fresh fused training")
    records = pairs()
    metric = PerceptualMetric()
    old_means = baseline_means(records, metric.protocol)
    model = ControlAdapter(NAME, checkpoint, 3000, "fixed", expected_settings={}, expected_updates=915000)
    model_check = model.validate(precisions=("fp16",))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {"training_checkpoint": str(checkpoint), "training_checkpoint_sha256": sha256(checkpoint),
                "training": experiment, "sets": SETS, "images": records, "metrics": metric.protocol,
                "method": NAME, "precision": "FP16 controller, FP32 reconstruction; same adapter as original MSE study",
                "distortion": "BT.601 encoded Y and RGB, float64; primary metrics shave 2 HR pixels",
                "input": "Existing hashed MATLAB-style antialiased bicubic X2 LR",
                "selection": "All 219 images; final scheduled epoch, no benchmark checkpoint selection",
                "baseline": str(BASELINE), "baseline_summary_sha256": sha256(BASELINE / "summary.json"),
                "baseline_per_image_sha256": sha256(BASELINE / "per_image.jsonl"),
                "baseline_verification_sha256": sha256(BASELINE / "verification.json"),
                "baseline_scope": "Previously verified comparator scores reused, not rerun",
                "source_sha256": sha256(Path(__file__)), "model_adapter_sha256": sha256(ROOT / "tools/paper_bench/control_ablation_models.py")}
    freeze(OUTPUT / "protocol.json", protocol)
    freeze(OUTPUT / "model.json", model.metadata)
    path = OUTPUT / "per_image.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    expected = {(r["dataset"], r["hr_filename"]) for r in records}
    done = {(r["dataset"], r["image"]) for r in rows}
    if len(done) != len(rows) or not done <= expected or any(r["method"] != NAME for r in rows):
        raise ValueError("Invalid existing evaluation identities")
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        for record in records:
            if (record["dataset"], record["hr_filename"]) in done:
                continue
            low = read_rgb(Path(record["lr_path"])).astype(np.float32) / 255
            high = read_rgb(Path(record["hr_path"])).astype(np.float32) / 255
            h, w = low.shape[:2]
            if high.shape[0] not in (2 * h, 2 * h + 1) or high.shape[1] not in (2 * w, 2 * w + 1):
                raise ValueError("Unexpected scale-modcrop geometry")
            high = high[:2 * h, :2 * w]
            prediction = model.predict(low).astype(np.float32)
            values = distortion(prediction, high) | {"lpips_alex_v01": metric.score(prediction, high)}
            outputs = save_rgb(prediction, OUTPUT / "outputs" / record["dataset"] / record["hr_filename"], verify=True)
            row = {"method": NAME, "dataset": record["dataset"], "image": record["hr_filename"],
                   "metrics": values, "outputs": outputs, "input_sha256": record["lr_sha256"], "target_sha256": record["hr_sha256"]}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            done.add((row["dataset"], row["image"]))
            write_json(OUTPUT / "status.json", {"phase": "evaluating", "completed": len(rows), "total": 219})
            print(f"{len(rows)}/219 {row['dataset']} {row['image']} PSNR={values[KEYS[0]]:.4f}", flush=True)
    if done != expected:
        raise ValueError("Incomplete evaluation")
    verification = verify_outputs(rows, records, metric) | {"model_check": model_check}
    means = {dataset: {NAME: {key: float(np.mean([r["metrics"][key] for r in rows if r["dataset"] == dataset]))
                              for key in rows[0]["metrics"]}} | old_means[dataset] for dataset in SETS}
    deltas = {dataset: {key: means[dataset][NAME][key] - means[dataset]["aeisu_div2k_fp16"][key] for key in KEYS} for dataset in SETS}
    write_json(OUTPUT / "summary.json", {"complete": True, "images": 219, "new_reconstructions": len(rows),
                                        "means": means, "l1_minus_mse": deltas})
    write_json(OUTPUT / "verification.json", verification)
    lines = ["# DIV2K: 3,000-epoch L1 benchmark", "", "Final fresh L1 checkpoint: 915,000 updates. Previous MSE: 305,000 updates.",
             "Both loss and training budget changed; this comparison does not separate their contributions.",
             "Same 219 hashed pairs, adapters, FP16 controller/FP32 reconstruction, Y PSNR/SSIM (shave 2), and RGB LPIPS-Alex v0.1.",
             "All new outputs saved as FP32 NPZ and PNG previews. Comparator scores reused from the verified earlier archive.",
             "Dataset means below are descriptive. No new latency, mixed-precision-runtime or manuscript claim is inferred."]
    for dataset in SETS:
        lines += ["", "## " + dataset, "", "| Method | Y PSNR ↑ | Y SSIM ↑ | LPIPS ↓ |", "|---|---:|---:|---:|"]
        for method in (NAME, *METHODS):
            v = means[dataset][method]
            lines.append(f"| {method} | {v[KEYS[0]]:.5f} | {v[KEYS[1]]:.7f} | {v[KEYS[2]]:.7f} |")
        d = deltas[dataset]
        lines += ["", f"L1 minus previous MSE: {d[KEYS[0]]:+.5f} dB, {d[KEYS[1]]:+.7f} SSIM, {d[KEYS[2]]:+.7f} LPIPS."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUTPUT / "status.json", {"phase": "complete", "completed": len(rows), "total": 219})
    print(json.dumps({"l1_minus_mse": deltas, "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
