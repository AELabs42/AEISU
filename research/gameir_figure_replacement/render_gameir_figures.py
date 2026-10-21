"""Render publication figures from previously verified GameIR crop archives, CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "target": "Target", "aif_mixed": "AEISU mixed", "fsr1_easu_fp32": "FSR1 EASU",
    "repnet_x2_fp16": "RepNet", "rt4ksr_x2_fp16": "RT4KSR",
    "span_x2_fp16": "SPAN", "swinir_light_x2_fp16": "SwinIR-Light",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(archive, rows, stem):
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(len(rows), 7, figsize=(6.9, 1.2 * len(rows)), squeeze=False)
    fig.subplots_adjust(left=.01, right=.99, bottom=.13 / len(rows), top=.86 if len(rows) == 1 else .92,
                        wspace=.07, hspace=.34)
    records = []
    for output_row, archive_row in enumerate(rows):
        record = manifest["crops"][archive_row - 1]
        panels = []
        for column, panel in enumerate(record["panels"]):
            method = panel["method"]
            raw = archive / "crops" / str(archive_row) / (method + ".npy")
            png = raw.with_suffix(".png")
            if digest(raw) != panel["raw_sha256"] or digest(png) != panel["png_sha256"]:
                raise ValueError(f"Saved crop hash mismatch: {raw}")
            array = np.load(raw, allow_pickle=False)
            expected = np.floor(np.clip(array, 0, 1) * 255 + .5).astype(np.uint8)
            if not np.array_equal(expected, np.asarray(Image.open(png).convert("RGB"))):
                raise ValueError(f"Saved crop rounding mismatch: {png}")
            ax = axes[output_row, column]
            ax.imshow(expected, interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            if output_row == 0:
                ax.set_title(LABELS[method], fontsize=7)
            for spine in ax.spines.values():
                spine.set_linewidth(.35)
            panels.append({key: panel[key] for key in ("method", "source_sha256", "raw_sha256", "png_sha256")})
        axes[output_row, 0].text(0, -.08, record["label"] + " / " + record["sequence"],
                                 transform=axes[output_row, 0].transAxes, fontsize=6, va="top")
        records.append({key: record[key] for key in ("label", "screening_number", "sequence", "frame", "xywh")})
        records[-1]["panels"] = panels
    destination = ROOT / "paper/figures"
    for extension in ("pdf", "png"):
        fig.savefig(destination / (stem + "." + extension), bbox_inches="tight", pad_inches=.025, dpi=300)
    plt.close(fig)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-v3", type=Path, required=True)
    parser.add_argument("--archive-v2", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42})
    report = {
        "date": "2026-10-21", "dataset": "GameIR Town05", "track": "Target-derived 2x area / SSAA",
        "new_inference": False, "numerical_results_changed": False,
        "display": "Original RGB, common half-up uint8 rounding, nearest-neighbor enlargement; no contrast enhancement",
        "selection": "Previously author-selected candidates 51 and 56, plus previously requested candidate 41",
        "model_artifact_sha256": digest(ROOT / "optimized/native/weights/adaptive_fp8.bin"),
        "attribution": "GameIR (Zhou et al., 2024), dataset published under MIT; CARLA Simulator assets under CC-BY",
        "main": render(args.archive_v3, [1, 2], "final_gameir_comparison_compact"),
        "supplement": render(args.archive_v2, [2], "final_gameir_window_panel"),
    }
    audit = ROOT / "release/audits/gameir_figures_2026-10-21"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verified and rendered three GameIR crop groups; no model inference or benchmark changes.")


if __name__ == "__main__":
    main()
