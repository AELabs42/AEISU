"""Reproduce two frozen QRISP frames and make matched, unmodified RGB crops."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/paper_bench"))
from common import clear_native_scratch, native, provenance, sha256, write_json
from comparison_models import load_neural, neural_forward, spatial, tensor_input, SPATIAL_EXE
from qrisp_benchmark import pair_arrays

RECORDS = ROOT / "output/paper_measurements/2026-09-29/qrisp_area_v2"
OUTPUT = ROOT / "output/figures/external_comparisons"
METHODS = (("aif_mixed", "AEISU mixed"), ("fsr1_easu_fp32", "FSR1 EASU"),
           ("repnet_x2_fp16", "RepNet FP16"), ("rt4ksr_x2_fp16", "RT4KSR FP16"),
           ("span_x2_fp16", "SPAN FP16"), ("swinir_light_x2_fp16", "SwinIR FP16"))
# Chosen after inspecting only HR targets, before reconstructing any method.
CROPS = (("0000", "0145", "Gate edge", (160, 360, 96, 96)),
         ("0000", "0145", "Bark", (1380, 370, 96, 96)),
         ("0001", "0145", "Brick edge", (1550, 405, 96, 96)),
         ("0001", "0145", "Floor tiles", (1100, 830, 96, 96)))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity_check(identity, current):
    stored = read(RECORDS / identity / "protocol.json")
    keys = ("checkpoint_sha256", "artifact_sha256", "native_executable_sha256",
            "architecture_sha256", "exe_sha256", "shader_hashes", "prepared_operations", "loader_hashes")
    for key in keys:
        if key in stored and stored[key] != current.get(key):
            if identity == "aif" and key == "native_executable_sha256":
                continue  # The rebuilt binary is recorded; saved-score equivalence is checked per frame.
            raise ValueError(f"Frozen implementation mismatch: {identity}/{key}")
    return {"frozen": stored, "regenerated": current}


def save_crops(pair, method, value):
    value = np.clip(value, 0, 1)
    if method != "target":
        identity = "aif" if method == "aif_mixed" else method
        rows = [json.loads(line) for line in (RECORDS / identity / "per_image.jsonl").read_text().splitlines()]
        row = next(item for item in rows if (item["segment"], item["frame"]) == (pair["segment"], pair["frame"]))
        _, target = pair_arrays(pair)
        mse = float(np.mean((value.astype(np.float64) - target.astype(np.float64)) ** 2))
        old = row["methods"][method]["rgb_mse"]
        if not np.isclose(mse, old, rtol=1e-6, atol=2e-10):
            raise ValueError(f"Reconstruction changed: {method}, {mse} versus {old}")
    else:
        mse, old = None, None
    records = []
    for index, (segment, frame, label, crop) in enumerate(CROPS):
        if (segment, frame) != (pair["segment"], pair["frame"]):
            continue
        x, y, width, height = crop
        rgb = value[y:y + height, x:x + width].copy()
        assert rgb.shape == (height, width, 3) and np.isfinite(rgb).all()
        directory = OUTPUT / "crops" / str(index + 1)
        directory.mkdir(parents=True, exist_ok=True)
        raw, png = directory / f"{method}.npy", directory / f"{method}.png"
        np.save(raw, rgb, allow_pickle=False)
        Image.fromarray(np.floor(rgb * 255 + 0.5).astype(np.uint8)).save(png)
        records.append({"index": index + 1, "label": label, "segment": segment, "frame": frame,
                        "xywh": crop, "method": method, "raw": str(raw), "raw_sha256": sha256(raw),
                        "png_sha256": sha256(png), "full_frame_rgb_mse": mse, "frozen_rgb_mse": old})
    return records


def render(indices, name):
    labels = (("target", "Target"),) + METHODS
    fig, axes = plt.subplots(len(indices), len(labels), figsize=(6.9, len(indices) * 1.18), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.015, top=0.92, wspace=0.07, hspace=0.34)
    for row, index in enumerate(indices):
        for col, (method, label) in enumerate(labels):
            ax = axes[row, col]
            with Image.open(OUTPUT / "crops" / str(index) / f"{method}.png") as bitmap:
                ax.imshow(np.asarray(bitmap), interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            if row == 0:
                ax.set_title(label, fontsize=7)
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
        axes[row, 0].text(0, -0.08, f"{CROPS[index - 1][2]} / {CROPS[index - 1][0]}",
                          transform=axes[row, 0].transAxes, fontsize=6.5, va="top")
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT / f"{name}.{extension}", bbox_inches="tight", pad_inches=0.025, dpi=240)
    if name == "external_comparison_compact":
        fig.savefig(ROOT / "docs/manuscript/figures/external_comparison_compact.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


@torch.inference_mode()
def main():
    if torch.version.cuda != "13.0":
        raise RuntimeError("Torch CUDA 13.0 required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if torch.cuda.mem_get_info()[0] < 6 * 1024 ** 3:
        raise RuntimeError("Insufficient free VRAM for safe full-frame regeneration")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    selection = read(RECORDS / "selection.json")
    pairs = [pair for pair in selection["pairs"] if pair["frame"] == "0145"]
    assert len(pairs) == 2
    manifest = {"selection": "Two midpoint benchmark frames; four regions selected from HR targets before inference",
                "track": selection["track"], "selection_sha256": sha256(RECORDS / "selection.json"),
                "display": "Identical coordinates; clipped RGB rounded half-up to uint8; nearest enlargement; no enhancement",
                "protocols": {}, "crops": [], "torch": torch.__version__, "gpu_inference_run": True}
    for pair in pairs:
        _, target = pair_arrays(pair)
        manifest["crops"] += save_crops(pair, "target", target)
    for method, _ in METHODS:
        model = None
        if method == "aif_mixed":
            metadata = provenance()
            identity = "aif"
        elif method == "fsr1_easu_fp32":
            identity = method
            metadata = {"exe_sha256": sha256(SPATIAL_EXE),
                        "shader_hashes": {p.name: sha256(p) for p in sorted(SPATIAL_EXE.parent.glob("*.cso"))}}
        else:
            identity = method
            model, metadata = load_neural(method.removesuffix("_fp16"), "fp16", prepared=True)
        manifest["protocols"][method] = identity_check(identity, metadata)
        for pair in pairs:
            low, _ = pair_arrays(pair)
            scratch = OUTPUT / "scratch"
            if method == "aif_mixed":
                values, _ = native(low, scratch, warmup=1, iterations=1)
                value = values["mixed"]
                clear_native_scratch(scratch)
            elif method == "fsr1_easu_fp32":
                value, _ = spatial(low, scratch, "fsr1_easu")
            else:
                value = neural_forward(model, tensor_input(low, "fp16"), method.removesuffix("_fp16"))
                value = value[0].permute(1, 2, 0).float().cpu().numpy()
            manifest["crops"] += save_crops(pair, method, value)
            print(f"Verified frozen scores and saved crops: {method}/{pair['segment']}", flush=True)
        del model, value
        gc.collect()
        torch.cuda.empty_cache()
    write_json(OUTPUT / "manifest.json", manifest)
    render((1, 4), "external_comparison_compact")
    render((1, 2, 3, 4), "external_comparison_candidates")


if __name__ == "__main__":
    main()
