"""Audit training histories and summarize paired Area/Bicubic checkpoint scores."""
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

from common import ROOT, sha256, write_json
from final_benchmark_analysis import CONDITIONS, METRICS, add_contrast, point_weights, sample_weights, summarize_family
from training_degradation_evaluate import OUTPUT

METHODS = ("area_trained", "bicubic_trained")
LABELS = {"area_trained": "Area-trained", "bicubic_trained": "Bicubic-trained"}


def training_audit(output, protocol):
    records = {}
    targets = {}
    for name, model in protocol["models"].items():
        path = Path(model["checkpoint"])
        state = torch.load(path, map_location="cpu", weights_only=True)
        cfg = state["config"]
        run = path.parent.parent
        event_path = run / "TensorBoard"
        events = EventAccumulator(str(event_path), size_guidance={"scalars": 0}).Reload()
        series = {tag: events.Scalars(tag) for tag in ("train/mse", "train/total", "train/lr")}
        if any(len(values) != 1000 or [p.step for p in values] != list(range(1, 1001)) for values in series.values()):
            raise ValueError("Incomplete TensorBoard epoch trace")
        if any(a.value != b.value for a, b in zip(series["train/mse"], series["train/total"])):
            raise ValueError("Objective differs from MSE")
        steps = sorted({int(s["step"].item()) for s in state["optimizer"]["state"].values()})
        if steps != [875000]:
            raise ValueError("Unexpected optimizer update count")
        error = max(abs(p.value - (cfg["min_lr"] + .5*(cfg["lr"]-cfg["min_lr"])
                    * (1 + math.cos(math.pi*(p.step*875-1)/874999)))) for p in series["train/lr"])
        if error > 2e-10 or abs(series["train/mse"][-1].value-state["metrics"]["train_mse"]) > 1e-10:
            raise ValueError("Cosine or checkpoint/TensorBoard loss mismatch")
        inputs = sorted(Path(cfg["train_source_dir"]).glob("*.png"))
        target_root = Path(cfg["train_target_dir"])
        targets[name] = {p.name: sha256(target_root / p.name) for p in inputs}
        if len(targets[name]) != 700:
            raise ValueError("Expected 700 training identities")
        with (output / f"{name}_training_scalars.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("epoch", "updates", "mse", "learning_rate"))
            writer.writerows((m.step, m.step*875, m.value, lr.value)
                             for m, lr in zip(series["train/mse"], series["train/lr"]))
        records[name] = {"final_training_mse": state["metrics"]["train_mse"], "optimizer_steps": steps,
                         "tensorboard_epochs": 1000, "cosine_max_absolute_error": error,
                         "checkpoint_sha256": sha256(path), "gradient_phase": model["gradient_phase"],
                         "tensorboard_sha256": {p.name: sha256(p) for p in event_path.glob("events.out.tfevents.*")}}
    if targets[METHODS[0]] != targets[METHODS[1]]:
        raise ValueError("Training target identity/content mismatch")
    audit = {"passed": True, "shared_700_training_target_hashes": targets[METHODS[0]], "models": records,
             "budget": protocol["shared_training_config"], "scope": "CPU-only checkpoint/optimizer and complete TensorBoard audit"}
    write_json(output / "training_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((args.output / "protocol.json").read_text())
    audit = training_audit(args.output, protocol)
    if args.audit_only:
        print(json.dumps({"training_audit_passed": True, "models": audit["models"]}, indent=2))
        return
    pairs = json.loads(Path(protocol["selection"]).read_text())["pairs"]
    rows = [json.loads(line) for line in (args.output / "per_image.jsonl").read_text().splitlines() if line.strip()]
    index = {(r["method"], r["precision"], r["condition"], r["image"]): r for r in rows}
    expected = {(*slot, p["image"]) for slot in protocol["slots"] for p in pairs}
    if len(rows) != len(index) or set(index) != expected:
        raise ValueError("Incomplete/duplicate scores")
    keys = list(rows[0]["metrics"])
    arrays, means = {}, {}
    for condition in CONDITIONS:
        means[condition] = {}
        for method in METHODS:
            values = np.array([[index[method, "fp16", condition, p["image"]]["metrics"][k] for k in keys] for p in pairs])
            arrays[method, condition] = values[:, [keys.index(k) for k in METRICS]]
            means[condition][method] = {group: dict(zip(keys, map(float, point_weights(pairs, group) @ values)))
                                       for group in ("all", "most_similar", "middle", "least_similar")}
    _, weights = sample_weights(pairs, 3000, 20261006)
    contrasts = []
    for condition in CONDITIONS:
        add_contrast(contrasts, pairs, f"Bicubic-trained minus Area-trained / {condition}", arrays[METHODS[1], condition],
                     other_values=arrays[METHODS[0], condition])
    uncertainty = summarize_family(contrasts, weights)
    interaction = []
    add_contrast(interaction, pairs, "Training-by-input: Bicubic gain minus Area gain",
                 arrays[METHODS[1], "bicubic_aa"]-arrays[METHODS[0], "bicubic_aa"],
                 other_values=arrays[METHODS[1], "ssaa_area"]-arrays[METHODS[0], "ssaa_area"])
    result = {"complete": True, "records": len(rows), "means": means, "paired_uncertainty": uncertainty,
              "training_input_interaction": summarize_family(interaction, weights),
              "bootstrap": {"draws": 3000, "seed": 20261006, "clusters": "environment then source-sequence",
                            "families": [15, 3], "scope": "Five QRISP environments; not training-seed uncertainty"}}
    write_json(args.output / "summary.json", result)
    lines = ["# AEISU training-degradation comparison", "",
             "Two main-architecture checkpoints, 1000 epochs and 875000 updates each. Shared 700 training targets, "
             "seed 1234, batch 32, 512 HR crops, D4, RGB MSE only and cosine 2e-3 to 1e-6. "
             "Complete optimizer/TensorBoard traces and target hashes are checked in training_audit.json.", "",
             "Area uses the historical training inputs. Bicubic uses MATLAB-style antialiased 8K-to-1080p inputs "
             "with the same existing 4K targets. The QRISP Bicubic evaluation input is Pillow Bicubic, not identical "
             "to the training filter. Both checkpoints are CPU-FP32 folded and evaluated through common "
             "FP16 controller/FP32 constants and reconstruction with align32/halo16 padding. This study does not "
             "replace the published mixed-precision production model scores.", "",
             "Evaluation reuses all 600 frozen QRISP pairs across five conditions and unchanged Native targets. "
             "PSNR/SSIM use encoded Y with two-HR-pixel cropping; LPIPS AlexNet v0.1 uses full-resolution cropped RGB. "
             "Full-frame and RGB metrics are retained. All 6000 float32 outputs and PNG previews are saved.", "",
             "## Training losses", "", "| Training input | Final MSE |", "|---|---:|"]
    for method in METHODS:
        lines.append(f"| {LABELS[method]} | {audit['models'][method]['final_training_mse']:.9g} |")
    lines += ["", "Training losses refer to different input distributions and are not held-out score comparisons.", "",
              "## Held-out reconstruction", "", "| Input | Training | Y PSNR | Y SSIM | LPIPS |", "|---|---|---:|---:|---:|"]
    for condition in CONDITIONS:
        for method in METHODS:
            mean = means[condition][method]["all"]
            lines.append(f"| {condition} | {LABELS[method]} | {mean[METRICS[0]]:.5f} | {mean[METRICS[1]]:.7f} | {mean[METRICS[2]]:.7f} |")
    lines += ["", "## Paired changes", "", "Bicubic-trained minus Area-trained. Positive PSNR/SSIM and negative "
              "LPIPS favor Bicubic training. The 15 contrasts share simultaneous paired hierarchical-bootstrap bands.", "",
              "| Input | Metric | Change | Simultaneous 95% band |", "|---|---|---:|---|"]
    for row in uncertainty["rows"]:
        lo, hi = row["simultaneous_95_percent_band"]
        lines.append(f"| {row['id']} | {row['metric']} | {row['point']:+.6g} | [{lo:+.6g}, {hi:+.6g}] |")
    lines += ["", "## Interpretation and limits", "",
              "Direction, magnitude and paired support are recomputed in the tables above and summary.json. "
              "Earlier checkpoints' numerical rankings are not carried forward into this report. "
              "Distortion and perceptual metrics assess different properties; neither establishes universal rendering fidelity.", "",
              protocol["training_caveat"], "",
              "One seed per training distribution; scene/sequence bands do not quantify retraining variation. "
              "All sampled environments/strata are retained, not selected according to resulting scores.", "",
              "Independent verification: verification.json checks complete slot coverage and output presence, "
              "with first/last hash, decoded-array, PNG-rounding and metric checks in all ten slots."]
    (args.output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"complete": True, "records": len(rows), "means": {c: {m: means[c][m]['all'] for m in METHODS}
                      for c in CONDITIONS}}, indent=2))


if __name__ == "__main__":
    main()
