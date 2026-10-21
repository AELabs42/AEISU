"""Inspect frozen GameIR SSAA outputs for illustrative edge artifacts, CPU only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/GameIR/subset_town05"
RUN = ROOT / "output/paper_measurements/2026-10-01/gameir_town05_v3"
OUT = ROOT / "output/figures/external_artifact_screening"
METHODS = (("aif", "aif_mixed", "AEISU mixed"),
           ("fsr1_easu_fp32", "fsr1_easu_fp32", "FSR1 EASU"),
           ("repnet_x2_fp16", "repnet_x2_fp16", "RepNet FP16"),
           ("rt4ksr_x2_fp16", "rt4ksr_x2_fp16", "RT4KSR FP16"),
           ("span_x2_fp16", "span_x2_fp16", "SPAN FP16"),
           ("swinir_light_x2_fp16", "swinir_light_x2_fp16", "SwinIR FP16"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs():
    return [p for p in read(DATA / "selection.json")["pairs"] if p["frame"] == "00000030"]


def overview():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = pairs()
    for group in range(2):
        canvas = Image.new("RGB", (1600, 5 * 324), "white")
        draw = ImageDraw.Draw(canvas)
        for index, pair in enumerate(selected[group * 20:(group + 1) * 20]):
            with Image.open(DATA / pair["hr"]) as rgb:
                thumbnail = rgb.resize((384, 288), Image.Resampling.LANCZOS)
            x, y = (index % 4) * 400, (index // 4) * 324
            canvas.paste(thumbnail, (x, y + 25))
            draw.text((x + 4, y + 6), pair["sequence"], fill="black")
        canvas.save(OUT / f"target_overview_{group + 1}.png")


def archives(identity, method):
    rows = [json.loads(line) for line in (RUN / identity / "per_image.jsonl").read_text().splitlines()]
    return {(r["sequence"], r["frame"]): r["outputs"][method] for r in rows if r["condition"] == "ssaa"}


def luminance(rgb):
    return rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)


def search():
    """Rank edge-adjacent flat-background deviations, not a validated halo metric."""
    OUT.mkdir(parents=True, exist_ok=True)
    lookup = {method: archives(identity, method) for identity, method, _ in METHODS}
    candidates = []
    for pair in pairs():
        with Image.open(DATA / pair["hr"]) as image:
            target = np.asarray(image.convert("RGB")).astype(np.float32) / 255
        y = luminance(target)
        lo = cv2.erode(y, np.ones((3, 3), np.uint8))
        hi = cv2.dilate(y, np.ones((3, 3), np.uint8))
        wide = cv2.dilate(y, np.ones((21, 21), np.uint8)) - cv2.erode(y, np.ones((21, 21), np.uint8))
        mask = ((hi - lo) < 0.025) & (wide > 0.15)
        values = {}
        for identity, method, _ in METHODS:
            entry = lookup[method][pair["sequence"], pair["frame"]]
            with Image.open(RUN / identity / entry["png"]) as image:
                prediction = luminance(np.asarray(image.convert("RGB")).astype(np.float32) / 255)
            excess = np.maximum(prediction - hi, 0) + np.maximum(lo - prediction, 0)
            values[method] = excess * mask
        for top in range(80, y.shape[0] - 80, 48):
            for left in range(80, y.shape[1] - 80, 48):
                region = np.s_[top:top + 64, left:left + 64]
                fraction = float(mask[region].mean())
                if fraction < 0.05 or fraction > 0.7:
                    continue
                scores = {m: float(v[region].sum() / max(mask[region].sum(), 1)) for m, v in values.items()}
                aif = scores["aif_mixed"]
                neural = [scores[m] for _, m, _ in METHODS[2:]]
                gain = min(neural) - aif
                if gain > 0.001:
                    candidates.append({"sequence": pair["sequence"], "frame": pair["frame"], "hr": pair["hr"],
                                       "xywh": [left, top, 64, 64], "screen_score": gain,
                                       "flat_edge_fraction": fraction, "screen_deviation": scores})
        print(f"Screened {pair['sequence']}: {len(candidates)} accumulated patches", flush=True)
    candidates.sort(key=lambda r: r["screen_score"], reverse=True)
    selected = []
    counts = {}
    for candidate in candidates:
        sequence = candidate["sequence"]
        if counts.get(sequence, 0) >= 3:
            continue
        if any(r["sequence"] == sequence and abs(r["xywh"][0] - candidate["xywh"][0]) < 96
               and abs(r["xywh"][1] - candidate["xywh"][1]) < 96 for r in selected):
            continue
        selected.append(candidate)
        counts[sequence] = counts.get(sequence, 0) + 1
        if len(selected) == 48:
            break
    for index, candidate in enumerate(selected):
        candidate["number"] = index + 1
    (OUT / "candidates.json").write_text(json.dumps({"selection": "Output-informed heuristic screening of 40 midpoint SSAA benchmark frames",
        "caveat": "Flat edge-adjacent local-envelope error is a candidate-ranking heuristic, not a validated artifact statistic or population result.",
        "candidates": selected}, indent=2), encoding="utf-8")
    render_candidates(selected, lookup)


def render_candidates(selected, lookup, prefix="screening"):
    labels = ["Target"] + [label for _, _, label in METHODS]
    for batch in range((len(selected) + 7) // 8):
        canvas = Image.new("RGB", (7 * 160, 8 * 194 + 30), "white")
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(labels):
            draw.text((column * 160 + 8, 8), label, fill="black")
        for row, record in enumerate(selected[batch * 8:(batch + 1) * 8]):
            x, y, w, h = record["xywh"]
            with Image.open(DATA / record["hr"]) as image:
                panels = [image.crop((x, y, x + w, y + h))]
            for identity, method, _ in METHODS:
                entry = lookup[method][record["sequence"], record["frame"]]
                with Image.open(RUN / identity / entry["png"]) as image:
                    panels.append(image.crop((x, y, x + w, y + h)))
            for column, panel in enumerate(panels):
                canvas.paste(panel.resize((152, 152), Image.Resampling.NEAREST), (column * 160, row * 194 + 50))
            draw.text((4, row * 194 + 30), f"#{record['number']} {record['sequence']} ({x},{y})", fill="black")
        canvas.save(OUT / f"{prefix}_{batch + 1}.png")


def curved_candidates():
    cases = (("static_town05/07", (1740, 468, 80, 80)),
             ("static_town05/07", (1680, 468, 80, 80)),
             ("static_town05/07", (1590, 464, 80, 80)),
             ("dynamic_town05/00", (1400, 264, 80, 80)),
             ("dynamic_town05/00", (1330, 238, 80, 80)),
             ("static_town05/01", (1400, 128, 80, 80)),
             ("static_town05/13", (1770, 604, 80, 80)),
             ("dynamic_town05/02", (360, 954, 80, 80)))
    selected = []
    for index, (sequence, crop) in enumerate(cases):
        pair = next(p for p in pairs() if p["sequence"] == sequence)
        selected.append({"number": index + 49, "sequence": sequence, "frame": pair["frame"],
                         "hr": pair["hr"], "xywh": list(crop)})
    lookup = {method: archives(identity, method) for identity, method, _ in METHODS}
    render_candidates(selected, lookup, "curved_screening")
    (OUT / "curved_candidates.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


def publish():
    """Make the final figure from verified lossless archives, preserving raw crops."""
    candidates = read(OUT / "curved_candidates.json")
    chosen = [(next(c for c in candidates if c["number"] == number), label)
              for number, label in ((51, "Curved lamp arms"), (56, "Hood trim"))]
    destination = ROOT / "output/figures/external_artifact_comparison_v3"
    destination.mkdir(parents=True, exist_ok=True)
    lookup = {method: archives(identity, method) for identity, method, _ in METHODS}
    manifest = {"selection": "Author selected curved-edge candidates51 and56 after reviewing all56 output-informed examples from40 midpoint GameIR SSAA frames",
                "track": "GameIR Town05 SSAA: aligned 2x area-derived inputs, unchanged supplied1440p targets",
                "screening": str(OUT / "curved_candidates.json"), "inference_run": False,
                "display": "Original RGB; identical80x80 HR coordinates; common half-up uint8 rounding; nearest enlargement; no contrast, sharpening, or per-panel normalization",
                "scope": "Author-selected examples of curved-edge fringing and local reconstruction differences; not an artifact-rate estimate, proof of zero overshoot, or a universal visual ranking",
                "crops": []}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 7, figsize=(6.9, 2.36), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.06, top=0.90, wspace=0.07, hspace=0.34)
    for row, (candidate, label) in enumerate(chosen):
        sequence, crop = candidate["sequence"], candidate["xywh"]
        pair = next(p for p in pairs() if p["sequence"] == sequence)
        target_path = DATA / pair["hr"]
        if digest(target_path) != pair["hr_sha256"]:
            raise ValueError(f"Target changed: {target_path}")
        with Image.open(target_path) as image:
            target = np.asarray(image.convert("RGB")).astype(np.float32) / 255
        x, y, width, height = crop
        record = {"label": label, "screening_number": candidate["number"], "sequence": sequence,
                  "frame": pair["frame"], "xywh": list(crop),
                  "target": str(target_path), "target_sha256": digest(target_path), "panels": []}
        panels = [("target", "Target", target, None)]
        for identity, method, method_label in METHODS:
            entry = lookup[method][sequence, pair["frame"]]
            source = RUN / identity / entry["npz"]
            if digest(source) != entry["npz_sha256"]:
                raise ValueError(f"Archive changed: {source}")
            with np.load(source) as saved:
                value = saved["rgb"].copy()
            if value.shape != target.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid saved output: {source}")
            panels.append((method, method_label, value, source))
        crop_directory = destination / "crops" / str(row + 1)
        crop_directory.mkdir(parents=True, exist_ok=True)
        for column, (method, method_label, value, source) in enumerate(panels):
            rgb = value[y:y + height, x:x + width].copy()
            raw_path = crop_directory / f"{method}.npy"
            png_path = crop_directory / f"{method}.png"
            np.save(raw_path, rgb, allow_pickle=False)
            bitmap = np.floor(np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
            Image.fromarray(bitmap).save(png_path)
            axes[row, column].imshow(bitmap, interpolation="nearest")
            axes[row, column].set(xticks=[], yticks=[])
            if row == 0:
                axes[row, column].set_title(method_label, fontsize=7)
            for spine in axes[row, column].spines.values():
                spine.set_linewidth(0.35)
            record["panels"].append({"method": method, "source": str(source or target_path),
                                     "source_sha256": digest(source or target_path), "raw_sha256": digest(raw_path),
                                     "png_sha256": digest(png_path)})
        axes[row, 0].text(0, -0.08, f"{label} / {sequence}", transform=axes[row, 0].transAxes,
                          fontsize=6, va="top")
        manifest["crops"].append(record)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(destination / f"external_comparison_compact.{extension}", bbox_inches="tight", pad_inches=0.025, dpi=300)
    fig.savefig(ROOT / "docs/manuscript/figures/external_comparison_compact.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Published two matched comparisons with twelve verified lossless output archives: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("overview", "search", "curved", "publish"))
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.command == "overview":
        overview()
    elif args.command == "search":
        search()
    elif args.command == "curved":
        curved_candidates()
    else:
        publish()


if __name__ == "__main__":
    main()
