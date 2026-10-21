"""Render compact publication figures from frozen measurements and saved crops."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from render_parameter_figures import ROOT, legend, matrix


ASSETS = ROOT / "docs/manuscript/figures"
PREVIEWS = ROOT / "output/figures/compact_manuscript"
PARAMETERS = ROOT / "output/figures/learned_parameters/parameters.json"
LUMA = ROOT / "output/paper_measurements/2026-09-30/qrisp_all_luma_manuscript.json"
ARTIFACTS = ROOT / "output/figures/validation_head_candidates/validation_artifact_comparison_v4"
METHODS = (
    ("bilinear_fp32", "Bilinear"), ("fsr1_easu_fp32", "FSR1 EASU"),
    ("fsr1_rcas_02_fp32", "FSR1 + RCAS"), ("nis_0_fp32", "NIS (0)"),
    ("nis_05_fp32", "NIS (0.5)"), ("sgsr1_fp32", "SGSR1"),
    ("span_x2_fp32", "SPAN FP32"), ("span_x2_fp16", "SPAN FP16"),
    ("swinir_light_x2_fp32", "SwinIR-light FP32"), ("swinir_light_x2_fp16", "SwinIR-light FP16"),
    ("repnet_x2_fp32", "RepNet FP32"), ("repnet_x2_fp16", "RepNet FP16"),
    ("rt4ksr_x2_fp32", "RT4KSR FP32"), ("rt4ksr_x2_fp16", "RT4KSR FP16"),
    ("aif_fp32_reference", "AEISU FP32 ref."), ("aif_fp16", "AEISU FP16"),
    ("aif_fp8", "AEISU FP8"), ("aif_mixed", "AEISU mixed"),
)


def save(fig, name):
    fig.savefig(ASSETS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PREVIEWS / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PREVIEWS / f"{name}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def bank_figure(data):
    bank = np.asarray(data["arrays"]["basis_effective_k_phase_y_x"])
    assert bank.shape == (16, 4, 5, 5) and np.abs(bank).max() <= 0.18
    fig = plt.figure(figsize=(6.9, 2.65))
    width = 0.042
    height = width * 6.9 / 2.65
    for group in range(4):
        left = 0.055 + 0.245 * group
        for phase, label in enumerate(("TL", "TR", "BL", "BR")):
            fig.text(left + phase * 0.05 + width / 2, 0.95, label, ha="center", fontsize=8)
        for row in range(4):
            k = 4 * group + row
            bottom = 0.79 - row * 0.19
            fig.text(left - 0.01, bottom + height / 2, rf"$K_{{{k}}}$", ha="right", va="center", fontsize=8)
            for phase in range(4):
                matrix(fig, bank[k, phase], [left + phase * 0.05, bottom, width, height], 0.18)
    legend(fig, [0.32, 0.11, 0.36, 0.015], 0.18, "Shared signed weight scale", fontsize=7)
    save(fig, "adaptive_bank_compact")


def analysis_figure(data):
    arrays = {key: np.asarray(value) for key, value in data["arrays"].items()}
    fig = plt.figure(figsize=(6.9, 3.05))
    fig.text(0.02, 0.97, "(a) Four-phase base", fontsize=9, weight="bold")
    fig.text(0.55, 0.97, "(b) Learned input analysis", fontsize=9, weight="bold")
    original = arrays["base_bilinear_phase_y_x"]
    learned = arrays["base_learned_phase_y_x"]
    for phase, label in enumerate(("TL", "TR", "BL", "BR")):
        fig.text(0.135 + phase * 0.085 + 0.037, 0.91, label, ha="center", fontsize=8)
    for label, values, y in (("Original", original, 0.70), ("Learned", learned, 0.47),
                             ("Difference", learned - original, 0.24)):
        fig.text(0.125, y + 0.085, label, ha="right", va="center", fontsize=7)
        for phase in range(4):
            matrix(fig, values[phase], [0.135 + phase * 0.085, y, 0.074, 0.074 * 6.9 / 3.05], 1.05)
    legend(fig, [0.18, 0.16, 0.26, 0.012], 1.05, "Base / difference", fontsize=7)
    for x, kind, label in ((0.625, "original", "Original"), (0.825, "learned", "Learned")):
        fig.text(x + 0.05, 0.91, label, ha="center", fontsize=8)
        matrix(fig, arrays[f"input_rgb_{kind}"], [x, 0.66, 0.10, 0.10 * 6.9 / 3.05], 1.0)
        matrix(fig, arrays[f"input_luma_{kind}"][None], [x, 0.60, 0.10, 0.10 * 6.9 / 3.05 / 3], 1.0)
    fig.text(0.61, 0.80, "RGB", ha="right", fontsize=7)
    fig.text(0.61, 0.63, "Luma", ha="right", fontsize=7)
    legend(fig, [0.66, 0.56, 0.24, 0.012], 1.0, "RGB / luma", fontsize=7)
    for direction, name in enumerate(("H", "V", "D45", "D135")):
        x = 0.625 + direction * 0.083
        fig.text(x + 0.03, 0.425, name, ha="center", fontsize=7)
        for kind, y in (("original", 0.26), ("learned", 0.08)):
            matrix(fig, arrays[f"input_stencils_{kind}_direction_y_x"][direction],
                   [x, y, 0.06, 0.06 * 6.9 / 3.05], 4.0)
    fig.text(0.61, 0.32, "Original", ha="right", fontsize=7)
    fig.text(0.61, 0.14, "Learned", ha="right", fontsize=7)
    legend(fig, [0.66, 0.035, 0.24, 0.012], 4.0, "Directional stencils", fontsize=7)
    save(fig, "learned_analysis_compact")


def base_inset(data):
    """Show the actual four base kernels as a small architectural inset."""
    learned = np.asarray(data["arrays"]["base_learned_phase_y_x"])
    assert learned.shape == (4, 5, 5)
    fig = plt.figure(figsize=(1.15, 1.25))
    for phase, label in enumerate(("TL", "TR", "BL", "BR")):
        row, column = divmod(phase, 2)
        left, bottom = 0.03 + column * 0.50, 0.61 - row * 0.52
        matrix(fig, learned[phase], [left, bottom, 0.40, 0.40 * 1.15 / 1.25], 1.05)
        fig.text(left + 0.20, bottom - 0.032, label, ha="center", va="top", fontsize=6.5)
    save(fig, "learned_base_inset")


def prepare_artifact_selection():
    """Use the approved October 10 candidates 4/5/8 and retain the fourth region."""
    parent = ARTIFACTS.parent
    original_folder = parent / "validation_artifact_comparison_v3"
    original = json.loads((original_folder / "manifest.json").read_text())
    batch_folder = parent / "more_edges_2026-10-10"
    batch = json.loads((batch_folder / "selected_manifest.json").read_text())
    selected = ((4, "magenta-strip", "Magenta strip"),
                (5, "bright-edge", "Bright edge"), (8, "fine-line", "Fine line"))
    labels = ("Target", "Adaptive", "RGB residual", "Depth-to-space")
    keys = ("target", "adaptive", "rgb_residual", "depth_to_space")
    for head, model in batch["models"].items():
        if model["checkpoint_sha256"] != original["sources"][head]["checkpoint_sha256"]:
            raise ValueError(f"Output-head checkpoint changed: {head}")
    source_hashes = {Path(record["path"]): record["sha256"] for record in batch["sources"]}
    source_roots = {record["label"]: Path(record["path"]).parent for record in original["crops"][3]["files"]}
    crops = []
    for number, kind, label in selected:
        candidate = batch["candidates"][number - 1]
        if candidate["number"] != number:
            raise ValueError("Candidate numbering changed")
        record = {"image": candidate["image"], "kind": kind, "crop": candidate["crop"],
                  "figure_label": label, "selected_candidate_number": number,
                  "candidate_batch": "more_edges_2026-10-10", "files": [], "rgb_crops": []}
        x, y, width, height = record["crop"]
        stem = f"{Path(record['image']).stem}_{kind}_{x}_{y}_{width}_{height}"
        destination = ARTIFACTS / "crops" / stem
        destination.mkdir(parents=True, exist_ok=True)
        for panel_label, key in zip(labels, keys):
            source = source_roots[panel_label] / record["image"]
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            if source_hash != source_hashes[source]:
                raise ValueError(f"Validation output changed: {source}")
            candidate_crop = batch_folder / "crops" / f"{number:02d}" / f"{key}.png"
            with Image.open(source) as image, Image.open(candidate_crop) as saved:
                patch = image.convert("RGB").crop((x, y, x + width, y + height))
                if not np.array_equal(np.asarray(patch), np.asarray(saved.convert("RGB"))):
                    raise ValueError("Candidate crop does not match the original saved output")
            target = destination / f"{panel_label}_rgb.png"
            shutil.copy2(candidate_crop, target)
            record["files"].append({"label": panel_label, "path": str(source), "sha256": source_hash})
            record["rgb_crops"].append({"label": panel_label, "path": str(target),
                                       "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        crops.append(record)
    fourth = dict(original["crops"][3])
    fourth["figure_label"] = "Thin structure"
    x, y, width, height = fourth["crop"]
    stem = f"{Path(fourth['image']).stem}_{fourth['kind']}_{x}_{y}_{width}_{height}"
    destination = ARTIFACTS / "crops" / stem
    destination.mkdir(parents=True, exist_ok=True)
    for label in labels:
        shutil.copy2(original_folder / "crops" / stem / f"{label}_rgb.png", destination / f"{label}_rgb.png")
    crops.append(fourth)
    manifest = {"date": "2026-10-11", "inference_run": False, "source": original["source"],
                "sources": original["sources"], "crops": crops,
                "selection": "Author-selected candidates 4, 5, and 8 from the October 10 batch; original fourth crop retained",
                "candidate_manifest": str(batch_folder / "selected_manifest.json"),
                "candidate_manifest_sha256": hashlib.sha256((batch_folder / "selected_manifest.json").read_bytes()).hexdigest(),
                "previous_manifest": str(original_folder / "manifest.json"),
                "previous_manifest_sha256": hashlib.sha256((original_folder / "manifest.json").read_bytes()).hexdigest(),
                "display": "Original RGB; no contrast adjustment or sharpening", "benchmark_values_changed": False}
    (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def artifact_figure(contrast_gain=1.0, name="output_head_artifacts_compact"):
    """Apply one shared luminance contrast transform per target-defined region."""
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float64)
    display_record = {"gain": contrast_gain, "weights": weights.tolist(),
                      "formula": "display=clip(rgb+(gain-1)*(dot(rgb,weights)-mean(dot(target,weights))),0,1)",
                      "pivot": "Mean target-crop encoded-RGB luminance, shared by all four panels in that region",
                      "rounding": "floor(255*display+0.5)", "metrics_modified": False,
                      "source_manifest_sha256": hashlib.sha256((ARTIFACTS / "manifest.json").read_bytes()).hexdigest(),
                      "no_sharpening_or_spatial_filter": True, "crops": []}
    labels = ("Target", "Adaptive", "RGB residual", "Depth-to-space")
    fig = plt.figure(figsize=(6.9, 2.65))
    names = ("Curve", "Flower", "Curved trim", "Thin structure")
    for index, record in enumerate(manifest["crops"]):
        x, y, width, height = record["crop"]
        stem = f"{Path(record['image']).stem}_{record['kind']}_{x}_{y}_{width}_{height}"
        source_arrays = []
        source_paths = []
        for label in labels:
            path = ARTIFACTS / "crops" / stem / f"{label}_rgb.png"
            source_paths.append(path)
            with Image.open(path) as bitmap:
                source_arrays.append(np.asarray(bitmap.convert("RGB")).astype(np.float64) / 255)
        pivot = float(np.mean(source_arrays[0] @ weights))
        crop_record = {"stem": stem, "xywh": [x, y, width, height], "pivot": pivot, "panels": []}
        row, column = divmod(index, 2)
        left, top = 0.01 + column * 0.50, 0.91 - row * 0.49
        region_name = record.get("figure_label", names[index])
        fig.text(left, top + 0.045, f"({chr(97 + index)}) {region_name} / {Path(record['image']).stem}",
                 fontsize=8, weight="bold")
        for panel, label in enumerate(labels):
            ax = fig.add_axes([left + panel * 0.12, top - 0.36, 0.113, 0.30])
            rgb = source_arrays[panel]
            luminance = rgb @ weights
            adjusted = rgb + (contrast_gain - 1) * (luminance - pivot)[..., None]
            assert np.allclose(adjusted @ weights - pivot, contrast_gain * (luminance - pivot), atol=1e-12)
            clipped_pixels = np.any((adjusted < 0) | (adjusted > 1), axis=-1)
            displayed = np.floor(np.clip(adjusted, 0, 1) * 255 + 0.5).astype(np.uint8)
            if contrast_gain == 1:
                assert np.array_equal(displayed, np.floor(rgb * 255 + 0.5).astype(np.uint8))
            display_directory = PREVIEWS / "artifact_contrast" / name / stem
            display_directory.mkdir(parents=True, exist_ok=True)
            display_path = display_directory / f"{label}_rgb.png"
            Image.fromarray(displayed).save(display_path)
            crop_record["panels"].append({"label": label, "source": str(source_paths[panel]),
                "source_sha256": hashlib.sha256(source_paths[panel].read_bytes()).hexdigest(),
                "display": str(display_path), "display_sha256": hashlib.sha256(display_path.read_bytes()).hexdigest(),
                "clipped_pixel_fraction": float(clipped_pixels.mean())})
            ax.imshow(displayed, interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            ax.set_anchor("N")
            fig.text(left + panel * 0.12 + 0.0565, top - 0.03, label, ha="center", fontsize=6.8)
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
        display_record["crops"].append(crop_record)
    save(fig, name)
    (PREVIEWS / f"{name}_display.json").write_text(json.dumps(display_record, indent=2), encoding="utf-8")
    print(json.dumps({"gain": contrast_gain, "clipped_pixels": [
        {"crop": r["stem"], "panels": {p["label"]: p["clipped_pixel_fraction"] for p in r["panels"]}}
        for r in display_record["crops"]]}, indent=2))


def luma_table(data):
    titles = ("SSAA / Enhanced", "SMAA / Native", "FXAA / Native", "Native / Native")
    order = ("ssaa_to_enhanced", "smaa_to_none", "fxaa_to_none", "none_to_none")
    assert set(order).issubset(data["order"]) and len(METHODS) == 18
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Encoded-Y results for the primary area/SSAA track and three Native-target diagnostics. "
             r"All conditions use the same 60 QRISP frame identities. P: PSNR (dB, higher is better); "
             r"S: SSIM (higher); E: MSE $\times10^3$ (lower). Means use a two-HR-pixel border crop. "
             r"SSAA uses Enhanced targets; SMAA, FXAA, and unfiltered Native inputs are all compared to unchanged Native targets. "
             r"The AEISU FP32 row is an algebraic reference.}",
             r"\label{tab:luma-all}", r"\begingroup\fontsize{7.5}{9}\selectfont\setlength{\tabcolsep}{2pt}",
             r"\renewcommand{\arraystretch}{1.15}", r"\begin{tabular}{l*{12}{r}}", r"\toprule",
             "Method & " + " & ".join(r"\multicolumn{3}{c}{" + title + "}" for title in titles) + r"\\",
             "".join(r"\cmidrule(lr){" + f"{2 + 3 * i}-{4 + 3 * i}" + "}" for i in range(4)),
             " & " + " & ".join(["P", "S", "E"] * 4) + r"\\", r"\midrule"]
    for index, (key, label) in enumerate(METHODS):
        if index in (6, 14):
            lines.append(r"\midrule")
        values = [label]
        for condition in order:
            scores = data["quality"][condition][key]["all"]
            assert scores["images"] == 60
            values += [f"{scores['y_psnr_shave2_db']:.3f}", f"{scores['y_ssim_shave2']:.5f}",
                       f"{1000 * scores['y_mse_shave2']:.4f}"]
        lines.append(" & ".join(values) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", r"\end{table*}"]
    return "\n".join(lines)


def main():
    ASSETS.mkdir(exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42})
    data = json.loads(PARAMETERS.read_text())
    bank_figure(data)
    analysis_figure(data)
    base_inset(data)
    prepare_artifact_selection()
    artifact_figure()
    # Printed source is applied to the manuscript by apply_patch, not written here.
    print(luma_table(json.loads(LUMA.read_text())))


if __name__ == "__main__":
    main()
