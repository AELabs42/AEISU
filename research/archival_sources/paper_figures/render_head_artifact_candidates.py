"""CPU-only, reproducible visual candidates from saved output-head ablations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "output/paper_measurements/2026-09-30/output_head_ablation_v1"
OUTPUT = ROOT / "output/figures/output_head_visual_candidates"
HEADS = ("adaptive", "rgb_residual", "depth_to_space")
LABELS = ("Target", "Adaptive", "RGB residual", "Depth-to-space")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def luma(rgb):
    # BT.709 display diagnostic, not the encoded-Y benchmark metric.
    return rgb @ np.array([.2126, .7152, .0722], np.float32)


def select_flat(target, size):
    """Select a smooth mid-range crop using ONLY the target, not method errors."""
    gray = luma(target)
    mean = cv2.boxFilter(gray, -1, (size, size), normalize=True)
    variance = np.maximum(0, cv2.boxFilter(gray * gray, -1, (size, size), normalize=True) - mean * mean)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    gradient = cv2.boxFilter(np.abs(dx) + np.abs(dy), -1, (size, size), normalize=True)
    score = variance + gradient * .1
    margin = size // 2 + 8
    score[:margin] = score[-margin:] = np.inf
    score[:, :margin] = score[:, -margin:] = np.inf
    score[(mean < .1) | (mean > .9)] = np.inf
    cy, cx = np.unravel_index(np.argmin(score), score.shape)
    return int(cx - size // 2), int(cy - size // 2), size


def select_edge(target, size):
    """Prefer target-defined isolated strong edges over dense texture patches."""
    gray = luma(target)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    magnitude = np.hypot(dx, dy)
    strong = cv2.boxFilter((magnitude > .06).astype(np.float32), -1, (size, size), normalize=True)
    weak = cv2.boxFilter(magnitude, -1, (size, size), normalize=True)
    score = strong / (weak + .004)
    margin = size // 2 + 8
    score[:margin] = score[-margin:] = -np.inf
    score[:, :margin] = score[:, -margin:] = -np.inf
    score[(strong < .005) | (strong > .25)] = -np.inf
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    return int(cx - size // 2), int(cy - size // 2), size


def phase_summary(prediction, target, crop):
    x, y, size = crop
    error = luma(prediction[y:y+size, x:x+size]) - luma(target[y:y+size, x:x+size])
    # Index phases by absolute HR coordinates, not by the arbitrary crop origin.
    phases = [[float(error[(iy-y) % 4::4, (ix-x) % 4::4].mean()) for ix in range(4)] for iy in range(4)]
    phase_array = np.asarray(phases)
    return {"error_mean": float(error.mean()), "error_std": float(error.std()),
            "phase_means_4x4": phases, "phase_mean_rms_after_dc_removal": float(np.std(phase_array)),
            "target_luma_std": float(luma(target[y:y+size, x:x+size]).std())}


def draw_candidate(item, target, outputs, crops):
    arrays = [target, *outputs]
    fig, axes = plt.subplots(6, 4, figsize=(10, 13), constrained_layout=True)
    descriptions = ("Flat / RGB", "Flat / grayscale, contrast x6", "Flat / signed error", "Edge / RGB",
                    "Edge / grayscale, contrast x2", "Edge / signed error")
    for group, crop in enumerate(crops):
        x, y, size = crop
        reference = luma(target[y:y+size, x:x+size])
        center = float(reference.mean())
        gain = 6 if group == 0 else 2
        for column, array in enumerate(arrays):
            rgb = array[y:y+size, x:x+size]
            gray = luma(rgb)
            axes[group*3, column].imshow(rgb, interpolation="nearest")
            axes[group*3+1, column].imshow(np.clip(.5 + gain * (gray-center), 0, 1),
                                         cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[group*3+2, column].imshow(gray-reference, cmap="RdBu_r", vmin=-.03, vmax=.03,
                                         interpolation="nearest")
            for row in range(group*3, group*3+3):
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
            axes[group*3, column].set_title(LABELS[column], fontsize=11)
    for row, text in enumerate(descriptions):
        axes[row, 0].set_ylabel(text, fontsize=9)
    fig.suptitle(f"{item['dataset']} / {item['image']} - native FP16 candidates", fontsize=14)
    fig.supxlabel(f"HR crops: flat {crops[0]}, edge {crops[1]} (x, y, size). Nearest-neighbor display enlargement.\n"
                  "Shared target-centered contrast; signed BT.709 luma error scale +/-0.03. No per-method normalization.\n"
                  "Selected from target only. Examples are exploratory, not a population-level artifact conclusion.", fontsize=9)
    return fig


def main():
    cv2.setNumThreads(2)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((RECORDS / "protocol.json").read_text())
    manifest = {"selection": "All seven predetermined archived examples; target-only spatial crop selection",
                "precision": "Native FP16; FP8 quantization is not conflated with reconstruction artifacts",
                "luma": "BT.709 grayscale for visualization only; not the encoded-Y benchmark definition",
                "display": "Unmodified RGB plus shared target-centered grayscale gain (flat x6, edge x2); nearest display sampling",
                "error_scale": [-.03, .03], "inference_run": False, "examples": []}
    destination = ROOT / "output/pdf/output_head_visual_candidates.pdf"
    with PdfPages(destination) as pdf:
        for item in protocol["selection"]:
            directories = [RECORDS / "examples" / head / item["dataset"] / Path(item["image"]).stem for head in HEADS]
            if not all((path / "fp16.npy").exists() for path in directories):
                continue
            source = Path(item["hr"])
            if checksum(source) != item["hr_sha256"]:
                raise ValueError("Target identity changed")
            target = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            target = target[:target.shape[0] // 2 * 2, :target.shape[1] // 2 * 2]
            outputs = [np.load(path / "fp16.npy") for path in directories]
            if any(array.shape != target.shape for array in outputs):
                raise ValueError("Saved output/target geometry mismatch")
            size = 64 if min(target.shape[:2]) >= 500 else 32
            crops = (select_flat(target, size), select_edge(target, size))
            record = {"dataset": item["dataset"], "image": item["image"], "target": str(source),
                      "target_sha256": item["hr_sha256"], "flat_crop": crops[0], "edge_crop": crops[1],
                      "outputs": {head: {"file": str(path / "fp16.npy"), "sha256": checksum(path / "fp16.npy"),
                                          "flat_diagnostics": phase_summary(array, target, crops[0])}
                                  for head, path, array in zip(HEADS, directories, outputs)}}
            manifest["examples"].append(record)
            fig = draw_candidate(item, target, outputs, crops)
            name = f"{item['dataset']}_{Path(item['image']).stem}"
            fig.savefig(OUTPUT / f"{name}.png", dpi=180)
            fig.savefig(OUTPUT / f"{name}.pdf")
            pdf.savefig(fig)
            plt.close(fig)
            print(name, crops, {head: round(record['outputs'][head]['flat_diagnostics']['phase_mean_rms_after_dc_removal'], 6)
                               for head in HEADS}, flush=True)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
