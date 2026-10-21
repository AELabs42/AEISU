"""Render selected final-QRISP comparisons from verified archives, without inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from screen_qrisp_edges import METHODS, OUT, ROOT, read_json, sha256, write_json

DESTINATION = ROOT / "docs/manuscript/figures"
PANELS = (("target", "Target"), *METHODS)
TITLES = {"target": "Target", "aif_mixed": "AEISU mixed", "fsr1_easu": "FSR1 EASU",
          "quicksrnet_small_x2": "QuickSRNet Small", "rt4ksr_x2": "RT4KSR", "span_x2": "SPAN",
          "swinir_light_x2": "SwinIR-Light"}
REGIONS = {3: "(a) Curved bracket", 4: "(b) Thin branches", 6: "(c) Lamp panels"}


def verified_crops():
    cases = {c["number"]: c for c in read_json(OUT / "candidates.json")["candidates"] if c["number"] in REGIONS}
    manifest = read_json(OUT / "crop_manifest.json")
    archives = {(r["number"], r["method"]): r for r in manifest}
    crops, records = {}, []
    for number, case in cases.items():
        assert case["condition"] == "ssaa_area"
        x, y, w, h = case["xywh"]
        for method, _ in PANELS:
            source = archives[number, method]
            path = Path(source["source"])
            assert sha256(path) == source["source_sha256"]
            if method == "target":
                with Image.open(path) as bitmap:
                    expected = np.array(bitmap.convert("RGB"))[y:y + h, x:x + w]
            else:
                with np.load(path, allow_pickle=False) as saved:
                    value = saved["rgb"][y:y + h, x:x + w]
                    expected = np.floor(np.clip(value, 0, 1) * 255 + .5).astype(np.uint8)
            crop_path = Path(source["crop"])
            with Image.open(crop_path) as bitmap:
                pixels = np.array(bitmap.convert("RGB"))
            assert np.array_equal(pixels, expected), (number, method)
            crops[number, method] = pixels
            records.append({**source, "crop_sha256": sha256(crop_path)})
    return cases, crops, records


def render(numbers, crops, name):
    rows = len(numbers)
    fig, axes = plt.subplots(rows, len(PANELS), figsize=(6.9, rows * 1.06 + .18), squeeze=False)
    fig.subplots_adjust(left=.005, right=.995, top=1 - .17 / (rows * 1.06 + .18),
                        bottom=.03, wspace=.055, hspace=.24)
    for row, number in enumerate(numbers):
        for column, (method, _) in enumerate(PANELS):
            ax = axes[row, column]
            ax.imshow(crops[number, method], interpolation="nearest", resample=False)
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_linewidth(.35)
                spine.set_color("#777")
            if row == 0:
                ax.set_title(TITLES[method], fontsize=7.2, pad=4)
        label = REGIONS[number] if rows > 1 else REGIONS[number].split(") ", 1)[-1]
        axes[row, 0].text(0, -.055, label, transform=axes[row, 0].transAxes,
                          fontsize=6.7, ha="left", va="top")
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{extension}", bbox_inches="tight", pad_inches=.02, dpi=320)
    fig.savefig(DESTINATION / f"{name}.pdf", bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lamp-only", action="store_true", help="Render only the additional supplementary crop.")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    cases, crops, records = verified_crops()
    if args.lamp_only:
        name = "final_qrisp_lamp_panel"
        render((6,), crops, name)
        write_json(OUT / "lamp_panel_manifest.json", {
            "date": "2026-10-05", "candidate": 6, "case": cases[6],
            "display": "Original 96x96 RGB8 crops, nearest enlargement; unchanged from the extended comparison",
            "archives": [record for record in records if record["number"] == 6],
            "figure_sha256": sha256(DESTINATION / f"{name}.pdf"), "gpu_inference_run": False,
        })
        print("Verified archived crops; rendered the supplementary lamp-panel row without inference.")
        return
    render((3, 4), crops, "final_qrisp_comparison_compact")
    render((3, 4, 6), crops, "final_qrisp_comparison_extended")
    write_json(OUT / "publication_manifest.json", {
        "date": "2026-10-04", "main_candidates": [3, 4], "supplement_candidates": [3, 4, 6],
        "condition": "ssaa_area", "source_frame": "SeaPort/0000/0067.png", "cases": list(cases.values()),
        "selection": "Source-geometry candidate screening followed by human selection after viewing all method outputs",
        "scope": "Selected qualitative examples; not an artifact prevalence or overshoot-magnitude measurement",
        "display": "Identical 96x96 HR crops; unmodified colors; common half-up RGB8 rounding; nearest display",
        "gpu_inference_run": False, "archives": records,
        "figure_sha256": {name: sha256(DESTINATION / name) for name in
                           ("final_qrisp_comparison_compact.pdf", "final_qrisp_comparison_extended.pdf")}})
    print("Verified 21 matched crops; generated compact and extended final-QRISP figures without inference.")


if __name__ == "__main__":
    main()
