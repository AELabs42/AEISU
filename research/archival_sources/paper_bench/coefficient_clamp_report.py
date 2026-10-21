"""Summarize paired clamp quality, precision sensitivity and coefficient ranges."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
sys.path.insert(0, str(HERE))

from common import sha256, write_json
from coefficient_clamp_evaluate import OUTPUT
from final_benchmark_analysis import CONDITIONS, METRICS, add_contrast, point_weights, sample_weights, summarize_family

METHODS = ("clamped", "unclamped")
LABELS = {"clamped": "Soft coefficient clamp", "unclamped": "No coefficient clamp"}


def audit_training(output, protocol):
    records = {}
    configs = {}
    fields = ("batch", "patch", "source_pad", "seed", "lr", "min_lr", "optimizer", "weight_decay",
              "adam_beta1", "adam_beta2", "adam_eps", "grad_clip", "basis_grad_stride", "amp_dtype",
              "geometric_augmentation", "train_source_dir", "train_target_dir")
    for method in METHODS:
        path = Path(protocol["models"][method]["checkpoint"])
        state = torch.load(path, map_location="cpu", weights_only=True)
        if sha256(path) != protocol["models"][method]["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash changed")
        cfg = state["config"]
        configs[method] = {key: cfg.get(key) for key in fields}
        events_path = path.parent.parent / "TensorBoard"
        events = EventAccumulator(str(events_path), size_guidance={"scalars": 0}).Reload()
        mse, total, lr = [events.Scalars(k) for k in ("train/mse", "train/total", "train/lr")]
        if any(len(v) != 100 or [p.step for p in v] != list(range(1, 101)) for v in (mse, total, lr)):
            raise ValueError("Incomplete training scalar trace")
        if any(a.value != b.value for a, b in zip(mse, total)):
            raise ValueError("Training objective differs from MSE")
        steps = sorted({int(v["step"].item()) for v in state["optimizer"]["state"].values()})
        if steps != [87500] or abs(mse[-1].value - state["metrics"]["train_mse"]) > 1e-10:
            raise ValueError("Optimizer/TensorBoard/checkpoint disagreement")
        error = max(abs(p.value - (cfg["min_lr"] + .5 * (cfg["lr"] - cfg["min_lr"])
                    * (1 + math.cos(math.pi * (p.step * 875 - 1) / 87499)))) for p in lr)
        if error > 2e-10:
            raise ValueError("Unexpected learning-rate schedule")
        with (output / f"{method}_training_scalars.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("epoch", "updates", "mse", "learning_rate"))
            writer.writerows((m.step, m.step * 875, m.value, rate.value) for m, rate in zip(mse, lr))
        records[method] = {"final_training_mse": state["metrics"]["train_mse"], "optimizer_steps": steps,
                           "tensorboard_epochs": 100, "cosine_max_absolute_error": error,
                           "checkpoint_sha256": sha256(path),
                           "tensorboard_sha256": {p.name: sha256(p) for p in events_path.glob("events.out.tfevents.*")}}
    if configs["clamped"] != configs["unclamped"]:
        raise ValueError("Unexpected common training recipe or recorded dataset-route mismatch")
    audit = {"passed": True, "common_recorded_config": configs["clamped"], "models": records,
             "caveat": protocol["training_caveat"], "scope": "Recorded paths and training traces; not recovered historical scene membership"}
    write_json(output / "training_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--training-audit-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((args.output / "protocol.json").read_text())
    audit = audit_training(args.output, protocol)
    if args.training_audit_only:
        print(json.dumps(audit, indent=2))
        return
    selection_path = Path(protocol["selection"])
    if sha256(selection_path) != protocol["selection_sha256"]:
        raise ValueError("Selection changed")
    pairs = json.loads(selection_path.read_text())["pairs"]
    rows = [json.loads(line) for line in (args.output / "per_image.jsonl").read_text().splitlines() if line.strip()]
    index = {(r["method"], r["precision"], r["condition"], r["image"]): r for r in rows}
    expected = {(*slot, p["image"]) for slot in protocol["slots"] for p in pairs}
    if len(rows) != len(index) or set(index) != expected:
        raise ValueError(f"Incomplete or duplicate scores: {len(rows)} of {len(expected)}")
    keys = tuple(rows[0]["metrics"])
    coefficients = tuple(rows[0]["coefficients"])
    arrays, means, statistics = {}, {}, {}
    for method, precision, condition in protocol["slots"]:
        slot_rows = [index[method, precision, condition, p["image"]] for p in pairs]
        values = np.array([[r["metrics"][k] for k in keys] for r in slot_rows])
        arrays[method, precision, condition] = values[:, [keys.index(k) for k in METRICS]]
        means.setdefault(precision, {}).setdefault(condition, {})[method] = {
            group: dict(zip(keys, map(float, point_weights(pairs, group) @ values)))
            for group in ("all", "most_similar", "middle", "least_similar")}
        stats = np.array([[r["coefficients"][k] for k in coefficients] for r in slot_rows])
        statistics.setdefault(precision, {}).setdefault(condition, {})[method] = {
            "mean_per_image": dict(zip(coefficients, map(float, point_weights(pairs) @ stats))),
            "largest_per_image": dict(zip(coefficients, map(float, stats.max(0))))}
    _, weights = sample_weights(pairs, 3000, 20261010)
    contrasts = []
    for condition in CONDITIONS:
        add_contrast(contrasts, pairs, condition, arrays["unclamped", "fp16", condition],
                     other_values=arrays["clamped", "fp16", condition])
    quality = summarize_family(contrasts, weights)
    changes = {}
    contrasts = []
    for method in METHODS:
        changes[method] = arrays[method, "fp8_stress", "ssaa_area"] - arrays[method, "fp16", "ssaa_area"]
        add_contrast(contrasts, pairs, method, changes[method])
    quantization = summarize_family(contrasts, weights)
    contrasts = []
    add_contrast(contrasts, pairs, "unclamped minus clamped precision change", changes["unclamped"],
                 other_values=changes["clamped"])
    difference = summarize_family(contrasts, weights)
    result = {
        "complete": True, "records": len(rows), "means": means, "coefficient_statistics": statistics,
        "quality_uncertainty": quality, "quantization_uncertainty": quantization,
        "quantization_change_comparison": difference,
        "bootstrap": {"draws": 3000, "seed": 20261010, "families": [15, 6, 3],
                      "clusters": "environment then source-sequence; paired", "scope": "Evaluation-sample uncertainty, not training-seed variation"},
        "training_caveat": protocol["training_caveat"], "training_audit": audit,
    }
    write_json(args.output / "summary.json", result)
    lines = ["# Coefficient-clamp checkpoint comparison", "", "## Protocol", "",
             protocol["training_caveat"], "", protocol["evaluation"], "",
             "All 600 frozen QRISP pairs and five established input conditions are used. Targets, selection, "
             "padding, folding, output clipping and metric definitions are shared. Old archive files are read-only. "
             "Both checkpoints use the same tiled FP32 reconstruction so the no-clamp model never encounters "
             "the fused runtime's hard-coded coefficient transfer.", "",
             "Primary metrics are encoded BT.601 Y PSNR/SSIM after a two-HR-pixel crop and full-resolution "
             "cropped RGB LPIPS-Alex v0.1. RGB and full-frame results are retained in summary.json.", "",
             "## Training loss", "", "| Model | Final training MSE |", "|---|---:|"]
    for method in METHODS:
        lines.append(f"| {LABELS[method]} | {protocol['models'][method]['training_mse']:.9g} |")
    lines += ["", "## Reconstruction quality", "", "| Input | Model | Y PSNR | Y SSIM | LPIPS |", "|---|---|---:|---:|---:|"]
    for condition in CONDITIONS:
        for method in METHODS:
            v = means["fp16"][condition][method]["all"]
            lines.append(f"| {condition} | {LABELS[method]} | {v[METRICS[0]]:.5f} | {v[METRICS[1]]:.7f} | {v[METRICS[2]]:.7f} |")
    lines += ["", "## Paired changes", "", "No-clamp minus clamped. Positive PSNR/SSIM and negative LPIPS favor no-clamp. "
              "Exploratory hierarchical bootstrap bands share a 15-contrast family; only five environments "
              "are sampled; the bands do not measure training-seed variation.", "",
              "| Input | Metric | Difference | Simultaneous 95% band |", "|---|---|---:|---|"]
    for row in quality["rows"]:
        lo, hi = row["simultaneous_95_percent_band"]
        lines.append(f"| {row['id']} | {row['metric']} | {row['point']:+.7g} | [{lo:+.7g}, {hi:+.7g}] |")
    lines += ["", "## Uniform-controller FP8 rounding stress on Area", "", protocol["quantization"], "",
              "Calibration uses the same 24 training crops, not evaluation images. This software stress test "
              "is not the production mixed FP8/FP16 model and is not a native runtime timing measurement.", "",
              "| Model | FP8 PSNR | Delta PSNR | Delta SSIM | Delta LPIPS |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        low = means["fp8_stress"]["ssaa_area"][method]["all"]
        high = means["fp16"]["ssaa_area"][method]["all"]
        lines.append(f"| {LABELS[method]} | {low[METRICS[0]]:.5f} | {low[METRICS[0]]-high[METRICS[0]]:+.5f} | "
                     f"{low[METRICS[1]]-high[METRICS[1]]:+.7f} | {low[METRICS[2]]-high[METRICS[2]]:+.7f} |")
    lines += ["", "No-clamp minus clamped within-model precision changes:", "",
              "| Metric | Difference | Simultaneous 95% band |", "|---|---:|---|"]
    for row in difference["rows"]:
        lo, hi = row["simultaneous_95_percent_band"]
        lines.append(f"| {row['metric']} | {row['point']:+.7g} | [{lo:+.7g}, {hi:+.7g}] |")
    lines += ["", "## Coefficient behavior", "", protocol["coefficient_statistics"], "",
              "The raw range and applied range are different quantities. The soft transfer is identity to "
              "absolute value 2, has slope .05 to raw magnitude 12, and caps applied magnitude at 2.5. "
              "The p99 statistic is the mean of per-image sampled p99s, not a pooled whole-dataset quantile.", "",
              "| Input | Model | Raw RMS | Mean raw p99 | Maximum raw magnitude | Fraction >2 | Applied RMS |", "|---|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for method in METHODS:
            s = statistics["fp16"][condition][method]
            v = s["mean_per_image"]
            lines.append(f"| {condition} | {LABELS[method]} | {v['raw_rms']:.5f} | {v['raw_abs_p99_sample4096']:.5f} | "
                         f"{s['largest_per_image']['raw_abs_max']:.5f} | {v['raw_fraction_above2']:.7f} | {v['applied_rms']:.5f} |")
    lines += ["", "## Interpretation limits", "",
              "This observes the two trained checkpoints, not optimization difficulty directly. A smaller "
              "raw coefficient range does not by itself establish that coefficients must be more precise, "
              "and a quantization response also depends on trained controller/bank weights and calibration. "
              "Do not infer a general precision mechanism or distillation failure from these results alone.", "",
              "Pre-output-clip RGB excursion fractions describe global [0,1] gamut violations, not local-edge "
              "haloing; legitimate transitions and ringing are not separated by this statistic. "
              "All 7200 float32 reconstructions and PNG previews are stored in this new results folder. "
              "No performance superiority claim is tested."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "report": str(args.output / "report.md"),
                      "training_mse": {m: protocol["models"][m]["training_mse"] for m in METHODS},
                      "area": means["fp16"]["ssaa_area"], "quantization_difference": difference}, indent=2))


if __name__ == "__main__":
    main()
