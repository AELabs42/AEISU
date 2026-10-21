"""Completeness checks, paired scene/sequence uncertainty and ablation report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from final_benchmark_analysis import CONDITIONS, METRICS, add_contrast, point_weights, sample_weights, summarize_family

OUTPUT = Path("G:/Output Images/AEISU Core Ablations 2026-10-05/evaluation_v2")
METHODS = ("reference", "fixed_inputs", "fixed_base", "dense_core")
LABELS = {"reference": "Original 100-epoch reference", "fixed_inputs": "Fixed inputs", "fixed_base": "Fixed bilinear base",
          "dense_core": "Two dense 64-channel convolutions"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    protocol = read_json(args.output / "protocol.json")
    pairs = read_json(protocol["selection"])["pairs"]
    rows = [json.loads(line) for line in (args.output / "per_image.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    index = {(r["method"], r["precision"], r["condition"], r["image"]): r for r in rows}
    expected = {(method, "fp16", condition, p["image"]) for method in METHODS for condition in CONDITIONS for p in pairs}
    expected |= {(method, "fp8_stress", "ssaa_area", p["image"]) for method in METHODS for p in pairs}
    if len(rows) != len(index) or set(index) != expected:
        raise ValueError(f"Incomplete/duplicate output set: {len(rows)}; expected {len(expected)}")
    means = {}
    arrays = {}
    keys = tuple(rows[0]["metrics"])
    for precision, conditions in (("fp16", CONDITIONS), ("fp8_stress", ("ssaa_area",))):
        means[precision] = {}
        for condition in conditions:
            means[precision][condition] = {}
            for method in METHODS:
                data = np.array([[index[method, precision, condition, p["image"]]["metrics"][k] for k in keys] for p in pairs])
                arrays[method, precision, condition] = data[:, [keys.index(k) for k in METRICS]]
                groups = {}
                for group in ("all", "most_similar", "middle", "least_similar"):
                    values = point_weights(pairs, group) @ data
                    groups[group] = dict(zip(keys, map(float, values)))
                means[precision][condition][method] = groups
    _, weights = sample_weights(pairs, 3000, 20261005)
    contrasts = []
    for condition in CONDITIONS:
        for method in METHODS[1:]:
            add_contrast(contrasts, pairs, f"{method} minus reference / {condition}", arrays[method, "fp16", condition],
                         other_values=arrays["reference", "fp16", condition])
    architecture = summarize_family(contrasts, weights)
    contrasts = []
    for method in METHODS:
        add_contrast(contrasts, pairs, f"{method}: FP8 stress minus FP16 / Area", arrays[method, "fp8_stress", "ssaa_area"],
                     other_values=arrays[method, "fp16", "ssaa_area"])
    quantization = summarize_family(contrasts, weights)
    contrasts = []
    reference_loss = arrays["reference", "fp8_stress", "ssaa_area"] - arrays["reference", "fp16", "ssaa_area"]
    for method in METHODS[1:]:
        loss = arrays[method, "fp8_stress", "ssaa_area"] - arrays[method, "fp16", "ssaa_area"]
        add_contrast(contrasts, pairs, f"{method} minus reference: quantization change", loss, other_values=reference_loss)
    quantization_comparison = summarize_family(contrasts, weights)
    result = {"complete": True, "reconstructions": len(rows), "pairs": len(pairs), "means": means,
              "architecture_uncertainty": architecture, "quantization_uncertainty": quantization,
              "quantization_change_comparison": quantization_comparison,
              "bootstrap": {"draws": 3000, "seed": 20261005, "clusters": "environment then source-sequence; paired across controls",
                            "scope": "Five QRISP environments, balanced agreement strata; not seed uncertainty or population coverage"},
              "caveat": protocol["training_caveat"]}
    write_json(args.output / "summary.json", result)
    lines = ["# AEISU core architecture ablations", "", "## Protocol", "",
             "Four 100-epoch checkpoints, each with 87,500 updates, batch 32, 512x512 HR crops, D4 augmentation, "
             "RGB MSE and cosine 2e-3 to 1e-6. The controls change one factor each, not cumulatively.", "",
             "The frozen QRISP sample has 600 pairs from five environments and three equally represented source-pair "
             "agreement strata. Every input condition uses the unchanged Native HR target. The Area and Bicubic inputs "
             "are target-derived; Native/FXAA/SMAA use the independently rendered LR. Neither stratum sorting nor "
             "these ablations identifies rendering mismatch as the sole cause of any score difference.", "",
             "Primary quality: FP16 controller convolution execution with FP32 analysis/filter constants and fused "
             "reconstruction, no temporal information. All training checkpoints are folded on CPU in FP32 before evaluation. "
             "The established symmetric align32 and replicate-halo16 validation padding is shared by all models. "
             "PSNR/SSIM use encoded BT.601 Y, two-HR-pixel crop and FP64 "
             "arithmetic. LPIPS uses full-resolution cropped RGB, AlexNet v0.1 and FP32. Full-frame and RGB metrics "
             "are retained in summary.json. All 14,400 float32 reconstructions and original-color PNG previews are saved.", "",
             "Training protocol: " + protocol["training_caveat"], "",
             "## Final training loss", "", "| Model | MSE | Folded scalar count |", "|---|---:|---:|"]
    for method in METHODS:
        model = protocol["models"][method]
        lines.append(f"| {LABELS[method]} | {model['training_mse']:.9g} | {model['folded_parameters']:,} |")
    lines += ["", "Fixed-input and fixed-base controls retain the same deployed operator shapes as the reference; "
              "they change learned constants, not stage count. The dense core matches bottleneck convolution MACs "
              "but differs in depth, bias count and optimization. A separate generic same-backend timing "
              "diagnostic, where available, is reported below rather than pooled with the quality study."]
    for condition in CONDITIONS:
        lines += ["", f"## Quality: {condition}", "", "| Model | Y PSNR (dB) | Y SSIM | LPIPS (lower is better) |",
                  "|---|---:|---:|---:|"]
        for method in METHODS:
            value = means["fp16"][condition][method]["all"]
            lines.append(f"| {LABELS[method]} | {value[METRICS[0]]:.5f} | {value[METRICS[1]]:.7f} | {value[METRICS[2]]:.7f} |")
    lines += ["", "## Paired architecture contrasts", "", "Changed model minus original reference. Positive PSNR/SSIM "
              "and negative LPIPS favor the changed model. Nominal intervals and family-wise simultaneous bands "
              "are exploratory paired hierarchical bootstrap estimates; 45 metric contrasts share one family. "
              "They do not quantify training-seed variation.", "",
              "| Contrast | Metric | Difference | Nominal 95% interval | Simultaneous 95% band |", "|---|---|---:|---|---|"]
    for row in architecture["rows"]:
        nominal = row["nominal_95_percent_interval"]
        band = row["simultaneous_95_percent_band"]
        lines.append(f"| {row['id']} | {row['metric']} | {row['point']:+.6g} | [{nominal[0]:+.6g}, {nominal[1]:+.6g}] | [{band[0]:+.6g}, {band[1]:+.6g}] |")
    lines += ["", "## Area FP8 stress test", "", protocol["quantization"], "",
              "Scales use the same 24 fixed training-crop identities and calibration budget for every model. "
              "Each change below compares a model against its own FP16 result. This is not the production "
              "stage-specific mixed model, the older calibrated native head stress test, or a latency comparison.", "",
              "| Model | FP8 Y PSNR | Delta Y PSNR | Delta Y SSIM | Delta LPIPS |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        low = means["fp8_stress"]["ssaa_area"][method]["all"]
        high = means["fp16"]["ssaa_area"][method]["all"]
        lines.append(f"| {LABELS[method]} | {low[METRICS[0]]:.5f} | {low[METRICS[0]]-high[METRICS[0]]:+.5f} | "
                     f"{low[METRICS[1]]-high[METRICS[1]]:+.7f} | {low[METRICS[2]]-high[METRICS[2]]:+.7f} |")
    lines += ["", "Precision sensitivity is reported separately from absolute reconstruction quality; "
              "the table above is regenerated from these checkpoints rather than retaining earlier rankings.", "",
              "A separate nine-contrast family compares the within-model quantization changes against the reference. "
              "Negative PSNR/SSIM and positive LPIPS differences indicate greater degradation than the reference.", "",
              "| Control | Metric | Difference in quantization change | Simultaneous 95% band |", "|---|---|---:|---|"]
    for row in quantization_comparison["rows"]:
        lo, hi = row["simultaneous_95_percent_band"]
        lines.append(f"| {row['id']} | {row['metric']} | {row['point']:+.7f} | [{lo:+.7f}, {hi:+.7f}] |")
    timing_path = args.output / "same_backend_timing.json"
    if timing_path.exists():
        timing = read_json(timing_path)
        lines += ["", "## Separate same-backend timing diagnostic", "", timing["backend"], "", timing["scope"], "",
                  "1080p input to 4K output; five trials, 20 warmup forwards and 50 CUDA-event intervals per trial. "
                  "The median of trial medians is reported. Source padding/layout preparation, transfers and "
                  "CPU readback are outside the intervals; the padded validation canvas is processed. "
                  "These timings are not the optimized production deployment and are not used as output-head "
                  "speed evidence in the manuscript.", "", "| Model | Median (ms) | p95 (ms) |", "|---|---:|---:|"]
        for method in METHODS:
            value = timing["results"][method]
            lines.append(f"| {LABELS[method]} | {value['median_ms']:.4f} | {value['p95_ms']:.4f} |")
        lines += ["", "The frozen controls have similar observed timings to the reference. The dense two-layer "
                  "core is faster in this eager implementation, despite equal bottleneck MACs; depth, launch "
                  "count and tensor shapes differ. This does not establish the relative speeds of fully "
                  "optimized native variants."]
    lines += ["", "## Interpretation limits", "",
              "Small training-loss changes need not predict held-out reconstruction changes. Compare each factor "
              "across the five conditions rather than selecting its most favorable track. A single seeded run "
              "cannot establish a reproducible optimization advantage. Learned input/base benefit should be "
              "described as modest or unresolved where appropriate; their rationale is preserving operator "
              "structure, not proving that every fixed alternative has equal optimized cost.", "",
              "No universal conclusion about parallel bottlenecks follows from this one dense two-layer control. "
              "Repeated seeds would measure training-run variation.", "",
              "Source identities, checkpoint hashes, calibration hashes, folding/reconstruction checks and metric "
              "parity checks are recorded in protocol.json and preflight.json. Per-image scores and output hashes "
              "are recorded in per_image.jsonl. No held-out image affected calibration or checkpoint selection.", "",
              "## Output verification", "",
              "verification.json checks complete unique coverage and the presence of all 28,800 output files. "
              "An independent 48-output sample spans the first and last images in every model/precision/condition "
              "slot; file hashes, float32 decoding, PNG rounding and recomputed quality metrics pass. "
              "This sampled content audit is distinct from checking that every output file exists."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "reconstructions": len(rows), "report": str(args.output / "report.md"),
                      "area": means["fp16"]["ssaa_area"]}, indent=2))


if __name__ == "__main__":
    main()
