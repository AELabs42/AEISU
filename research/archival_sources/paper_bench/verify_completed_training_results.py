"""CPU-only readback checks for the completed Bicubic and DIV2K paper results."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np

from luma_metrics import scores, target_statistics


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "output/training_campaigns/fixed_stride_2026-10-16"
L1 = ROOT / "output/training_campaigns/div2k_l1_3000_2026-10-17/evaluations/classic_219"
MIXED = L1.with_name("classic_219_mixed")
REVIEW = ROOT / "output/editorial_review/div2k_mixed_2026-10-17"
METRICS = ("y_psnr_shave2_db", "y_ssim_shave2", "lpips_alex_v01")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_row(text, label, expected, digits):
    line = next(line for line in text.splitlines() if line.startswith(label + " &"))
    actual = [float(value) for value in re.findall(r"[+-]?(?:\d*\.\d+|\d+)", line.split("&", 1)[1])]
    assert len(actual) == len(expected)
    for left, right, places in zip(actual, expected, digits):
        assert abs(left - right) <= .500001 * 10 ** -places, (label, left, right)
    return len(actual)


def main():
    cv2.setNumThreads(2)
    degradation = CAMPAIGN / "evaluations/degradation"
    classic = CAMPAIGN / "evaluations/div2k_classic"
    d, c = load(degradation / "summary.json"), load(MIXED / "summary.json")
    assert d["complete"] and c["complete"]
    assert load(degradation / "verification.json")["passed"] and load(degradation / "training_audit.json")["passed"]
    assert load(classic / "verification.json")["passed"]
    assert load(L1 / "verification.json")["passed"]
    assert load(MIXED / "verification.json")["passed"]
    text = (ROOT / "docs/manuscript/inverse_supersampling_supplement.tex").read_text(encoding="utf-8")
    cells = 0
    conditions = ("ssaa_area", "bicubic_aa", "fxaa", "smaa", "native")
    for name, label in (("area_trained", "Area"), ("bicubic_trained", "Bicubic")):
        expected = [d["means"][condition][name]["all"][metric] for condition in conditions for metric in METRICS]
        cells += check_row(text, label, expected, [3, 5, 5] * 5)
    labels = {"bilinear": "Bilinear", "fsr1_easu": "FSR1 EASU", "fsrcnn_x2": "FSRCNN",
              "repnet_x2": "RepNet", "rt4ksr_x2": "RT4KSR", "quicksrnet_small_x2": "QuickSRNet S",
              "quicksrnet_medium_x2": "QuickSRNet M", "span_x2": "SPAN", "swinir_light_x2": "SwinIR-Light",
              "aeisu_div2k_l1_3000_fp16": r"\model{} FP16",
              "aeisu_div2k_l1_3000_mixed": r"\model{} mixed"}
    text = (ROOT / "docs/manuscript/figures/classic_bicubic.tex").read_text(encoding="utf-8")
    sets = ("Set5", "Set14", "BSD100", "Urban100")
    for name, label in labels.items():
        expected = [c["means"][dataset][name][metric] for dataset in sets for metric in METRICS]
        cells += check_row(text, label, expected, [3, 5, 5] * 4)
    text = (ROOT / "docs/manuscript/figures/classic_training_comparison.tex").read_text(encoding="utf-8")
    for dataset in sets:
        for method, loss in (("aeisu_div2k_fp16", "MSE"), ("aeisu_div2k_l1_3000_fp16", "L1")):
            label = ("B100" if dataset == "BSD100" else dataset) + " " + loss
            cells += check_row(text, label, [c["means"][dataset][method][key] for key in METRICS], [3, 5, 5])
    rows = [json.loads(line) for folder in (classic, L1, MIXED)
            for line in (folder / "per_image.jsonl").read_text().splitlines() if line.strip()]
    protocol = load(L1 / "protocol.json")
    assert protocol["images"] == load(classic / "protocol.json")["images"]
    assert protocol["metrics"] == load(classic / "protocol.json")["metrics"]
    mixed_protocol = load(MIXED / "protocol.json")
    assert mixed_protocol["images"] == protocol["images"] and mixed_protocol["metrics"] == protocol["metrics"]
    assert hashlib.sha256(Path(mixed_protocol["artifact"]).read_bytes()).hexdigest() == mixed_protocol["artifact_sha256"]
    assert mixed_protocol["training_checkpoint_sha256"] == protocol["training_checkpoint_sha256"]
    splits = mixed_protocol["calibration_preflight"]["splits"]
    assert len(splits["calibration"]) == 24 and len(splits["calibration_check"]) == 6
    calibration_hashes = {item["sha256"] for item in splits["calibration"]}
    check_hashes = {item["sha256"] for item in splits["calibration_check"]}
    assert not calibration_hashes & check_hashes
    training = load(CAMPAIGN / "div2k/pairs.json")
    train_hashes = {row["target_sha256"] for row in training["pairs"]}
    train_input_hashes = {row["input_sha256"] for row in training["pairs"]}
    test_hashes = {row["hr_sha256"] for row in protocol["images"]}
    assert len(training["pairs"]) == 800 and not train_hashes & test_hashes
    assert (calibration_hashes | check_hashes) <= train_input_hashes
    benchmark_hashes = test_hashes | {row["lr_sha256"] for row in protocol["images"]}
    assert not (calibration_hashes | check_hashes) & benchmark_hashes
    model = load(L1 / "model.json")
    assert model["epoch"] == 3000 and model["updates"] == 915000 and model["gradient_phase"] == "fixed"
    assert model["config"]["mse_loss_weight"] == 0 and model["config"]["l1_loss_weight"] == 1
    assert model["training_loss"] == "RGB L1 only" and model["config"]["lr_schedule_total_steps"] == 915000
    assert model["folded_parameters"] == 115996
    assert hashlib.sha256(Path(model["checkpoint"]).read_bytes()).hexdigest() == protocol["training_checkpoint_sha256"]
    identities = {(r["dataset"], r["hr_filename"]): r for r in protocol["images"]}
    methods = [*labels, "aeisu_div2k_fp16"]
    expected = {(method, dataset, image) for method in methods for dataset, image in identities}
    assert Counter((r["method"], r["dataset"], r["image"]) for r in rows) == Counter(expected)
    maximum_aggregation_error = 0.
    for dataset in sets:
        for method in methods:
            selected = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            for metric, value in c["means"][dataset][method].items():
                mean = math.fsum(r["metrics"][metric] for r in selected) / len(selected)
                maximum_aggregation_error = max(maximum_aggregation_error, abs(mean - value))
            for row in selected:
                item = identities[row["dataset"], row["image"]]
                assert row["input_sha256"] == item["lr_sha256"] and row["target_sha256"] == item["hr_sha256"]
                assert all(math.isfinite(value) for value in row["metrics"].values())
    assert maximum_aggregation_error < 1e-12
    maximum_y_error = 0.
    samples = []
    for dataset in sets:
        for method in ("aeisu_div2k_fp16", "aeisu_div2k_l1_3000_fp16", "aeisu_div2k_l1_3000_mixed", "fsrcnn_x2"):
            row = next(r for r in rows if r["dataset"] == dataset and r["method"] == method)
            item = identities[row["dataset"], row["image"]]
            path = Path(row["outputs"]["npz"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["outputs"]["npz_sha256"]
            with np.load(path, allow_pickle=False) as source:
                rgb = source["rgb"]
            target = cv2.cvtColor(cv2.imread(item["hr_path"]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            target = target[:rgb.shape[0], :rgb.shape[1]]
            values = scores(rgb, target_statistics({"target": target}), expected_shape=target.shape)["target"]
            maximum_y_error = max(maximum_y_error, *(abs(value - row["metrics"][key]) for key, value in values.items()))
            samples.append([method, dataset, row["image"]])
    assert maximum_y_error < 1e-10
    result = {"passed": True, "numeric_table_cells": cells, "classic_records": len(rows),
              "div2k_training_pairs": 800, "exact_train_test_hr_hash_overlap": 0,
              "aggregate_max_absolute_error": maximum_aggregation_error, "sampled_y_metric_max_error": maximum_y_error,
              "cpu_y_metric_samples": samples, "lpips": "All means recomputed from saved per-image values; not rerun",
              "gpu_work": False}
    REVIEW.mkdir(parents=True, exist_ok=True)
    (REVIEW / "numerical_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
