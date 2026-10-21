"""Inspect saved validation images and render reproducible output-head crops.

CPU-only: no model loading, inference, or modifications to source images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from render_head_artifact_candidates import checksum, luma, select_edge, select_flat


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/figures/validation_head_candidates"
RUNS = {
    "adaptive": "Training Run 2026-09-30 17-53-21 - aif_reparam_aggressive64_parallel32x4",
    "rgb_residual": "Training Run 2026-09-30 17-13-28 - aif_rgb_residual_parallel32x4",
    "depth_to_space": "Training Run 2026-09-30 18-32-00 - aif_depth_to_space_parallel32x4",
}
LABELS = ("Target", "Adaptive", "RGB residual", "Depth-to-space")
TARGETS = ROOT / "data/Validation/4k"


def sources():
    records = {}
    for head, run in RUNS.items():
        manifests = list((ROOT / "Models" / run / "Inference").glob("*/eval_manifest.json"))
        if len(manifests) != 1:
            raise ValueError(f"Expected one evaluation manifest for {head}")
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        passes = [p for p in manifest["passes"] if p["name"] == "validation_area_1080p_to_4k"]
        if len(passes) != 1 or manifest["amp_dtype"] != "fp16":
            raise ValueError("Unexpected validation pass or precision")
        records[head] = {
            "directory": ROOT / passes[0]["output_root"],
            "manifest": manifests[0],
            "checkpoint": ROOT / manifest["checkpoint"],
            "policy": {key: manifest.get(key) for key in
                       ("amp_dtype", "tile", "overlap", "tile_batch", "source_pad", "coefficient_transfer", "clamp")},
        }
        print(head, passes[0]["images"], records[head]["policy"], flush=True)
    return records


def overview(records):
    available = sorted(path.name for path in TARGETS.glob("*.png")
                       if all((r["directory"] / path.name).exists() for r in records.values()))
    indices = sorted(set(range(min(6, len(available)))) |
                     set(np.linspace(0, len(available) - 1, 48, dtype=int).tolist()))
    cell_w, cell_h = 320, 205
    columns = 6
    sheet = Image.new("RGB", (columns * cell_w, ((len(indices) + columns - 1) // columns) * cell_h), "white")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    draw = ImageDraw.Draw(sheet)
    for slot, index in enumerate(indices):
        name = available[index]
        with Image.open(TARGETS / name) as image:
            image.thumbnail((cell_w - 8, cell_h - 30))
            x, y = (slot % columns) * cell_w, (slot // columns) * cell_h
            sheet.paste(image, (x + 4, y + 4))
            draw.text((x + 4, y + cell_h - 25), name, font=font, fill="black")
    sheet.save(OUTPUT / "target_overview.jpg", quality=95)
    (OUTPUT / "overview_selection.json").write_text(json.dumps({
        "common_images": len(available), "selection": "First six plus 48 equally spaced target frames",
        "images": [available[i] for i in indices],
    }, indent=2), encoding="utf-8")
    print(OUTPUT / "target_overview.jpg", flush=True)


def read_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def locate(reference, count, scene=None):
    """Locate a screenshot crop in target frames; this is not an artifact score."""
    template = cv2.imread(str(reference))
    if template is None:
        raise ValueError(f"Unreadable reference: {reference}")
    matches = []
    paths = [TARGETS / f"{scene}.png"] if scene else sorted(TARGETS.glob("*.png"))[:count]
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
        best = (-1., None)
        for scale in np.linspace(.04, .5, 47):
            small = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if small.shape[0] >= image.shape[0] or small.shape[1] >= image.shape[1]:
                continue
            scores = cv2.matchTemplate(image, small, cv2.TM_CCOEFF_NORMED)
            _, maximum, _, point = cv2.minMaxLoc(scores)
            if maximum > best[0]:
                best = (maximum, {"image": path.name, "x": point[0] * 4, "y": point[1] * 4,
                                  "width": small.shape[1] * 4, "height": small.shape[0] * 4, "score": maximum})
        matches.append(best[1])
    matches.sort(key=lambda item: item["score"], reverse=True)
    print(json.dumps(matches[:10], indent=2), flush=True)


def search_candidates(records, count):
    """Screen for illustrative edge-adjacent differences, not population metrics."""
    available = sorted(path.name for path in TARGETS.glob("*.png")
                       if all((r["directory"] / path.name).exists() for r in records.values()))
    indices = np.linspace(0, len(available) - 1, min(count, len(available)), dtype=int)
    candidates = []
    size = 48
    for index in indices:
        name = available[index]
        paths = [TARGETS / name, *(r["directory"] / name for r in records.values())]
        grays = []
        for path in paths:
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            grays.append(cv2.transform(rgb.astype(np.float32) / 255,
                                       np.array([[.2126, .7152, .0722]], np.float32)))
        target, adaptive, residual, depth = grays
        dx = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=3) / 8
        dy = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=3) / 8
        edge = np.hypot(dx, dy) > .035
        distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
        local_range = cv2.dilate(target, np.ones((3, 3), np.uint8)) - cv2.erode(target, np.ones((3, 3), np.uint8))
        mask = ((distance >= 2.5) & (distance <= 7) & (local_range < .025) & (target > .05) & (target < .8))
        # Favor positive exterior fringes that are less prominent in adaptive.
        # This output-informed ranking is deliberately exploratory/cherry-picked.
        advantage = np.maximum(np.maximum(residual - target, depth - target), 0) - np.maximum(adaptive - target, 0)
        score = cv2.boxFilter(advantage * mask, -1, (size, size), normalize=True)
        coverage = cv2.boxFilter(mask.astype(np.float32), -1, (size, size), normalize=True)
        density = cv2.boxFilter(edge.astype(np.float32), -1, (size, size), normalize=True)
        score[(coverage < .03) | (density < .015) | (density > .15)] = -np.inf
        margin = size // 2 + 16
        score[:margin] = score[-margin:] = -np.inf
        score[:, :margin] = score[:, -margin:] = -np.inf
        for _ in range(3):
            cy, cx = np.unravel_index(np.argmax(score), score.shape)
            value = float(score[cy, cx])
            if not np.isfinite(value):
                break
            candidates.append({"image": name, "crop": [int(cx - size // 2), int(cy - size // 2), size, size],
                               "screening_score": value})
            score[max(0, cy - size * 2):cy + size * 2, max(0, cx - size * 2):cx + size * 2] = -np.inf
        print("Screened", name, flush=True)
    candidates.sort(key=lambda item: item["screening_score"], reverse=True)
    candidates = candidates[:24]
    (OUTPUT / "search_candidates.json").write_text(json.dumps({
        "selection": "Uniformly spaced frames; output-informed exterior positive-error ranking; illustrative screening only",
        "frames": [available[i] for i in indices], "candidates": candidates,
    }, indent=2), encoding="utf-8")
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    for start in range(0, len(candidates), 8):
        subset = candidates[start:start + 8]
        sheet = Image.new("RGB", (800, len(subset) * 220 + 28), "white")
        draw = ImageDraw.Draw(sheet)
        for column, label in enumerate(LABELS):
            draw.text((column * 200 + 4, 4), label, font=font, fill="black")
        for row, record in enumerate(subset):
            name = record["image"]
            x, y, width, height = record["crop"]
            paths = [TARGETS / name, *(r["directory"] / name for r in records.values())]
            for column, path in enumerate(paths):
                with Image.open(path) as image:
                    crop = image.crop((x, y, x + width, y + height)).resize((192, 192), Image.Resampling.NEAREST)
                    sheet.paste(crop, (column * 200 + 4, row * 220 + 28))
            draw.text((4, row * 220 + 222), f"{start + row + 1}: {name} ({x}, {y})", font=font, fill="black")
        sheet.save(OUTPUT / f"screening_{start // 8 + 1}.png")
    print(OUTPUT / "search_candidates.json", flush=True)


def render_crop(name, arrays, crop, kind, gain):
    x, y, width, height = crop
    cropped = [image[y:y + height, x:x + width] for image in arrays]
    if any(image.shape != (height, width, 3) for image in cropped):
        raise ValueError("Crop outside image")
    reference = luma(cropped[0])
    center = float(reference.mean())
    fig, axes = plt.subplots(3, 4, figsize=(11, 9), constrained_layout=True)
    for column, rgb in enumerate(cropped):
        gray = luma(rgb)
        axes[0, column].imshow(rgb, interpolation="nearest")
        axes[1, column].imshow(np.clip(.5 + gain * (gray - center), 0, 1),
                              cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[2, column].imshow(gray - reference, cmap="RdBu_r", vmin=-.03, vmax=.03, interpolation="nearest")
        axes[0, column].set_title(LABELS[column], fontsize=13)
        for row in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Original RGB", fontsize=11)
    axes[1, 0].set_ylabel(f"Shared grayscale contrast x{gain:g}", fontsize=11)
    # The target error is identically zero; use this panel for spatial context.
    full_height, full_width = arrays[0].shape[:2]
    thumbnail = cv2.resize(arrays[0], (768, round(full_height * 768 / full_width)), interpolation=cv2.INTER_AREA)
    axes[2, 0].clear()
    axes[2, 0].imshow(thumbnail)
    scale = 768 / full_width
    axes[2, 0].add_patch(Rectangle((x * scale, y * scale), width * scale, height * scale,
                                  fill=False, edgecolor="#e00000", linewidth=1.5))
    axes[2, 0].set_title("Target context / marked crop", fontsize=10)
    axes[2, 0].set_xticks([])
    axes[2, 0].set_yticks([])
    for column in range(1, 4):
        axes[2, column].set_xlabel("Signed luma error", fontsize=10)
    fig.suptitle(f"{name} / {kind} - saved FP16 validation output", fontsize=16)
    fig.supxlabel(f"1080p area input to 4K; HR crop (x, y, width, height) = {crop}. Nearest-sampled enlargement.\n"
                  "Identical target-centered grayscale transformation for every method; error range +/-0.03.\n"
                  "Saved 8-bit PNG outputs, not the native-runtime quantization benchmark. Exploratory selection.", fontsize=10)
    return fig


def cover(output_name):
    fig = plt.figure(figsize=(11, 9))
    fig.text(.09, .90, "Output-head visual candidates", fontsize=24, fontweight="bold")
    fig.text(.09, .85, "Saved validation images / 1080p area input to 4K", fontsize=15)
    sections = [
        ("Source and controls", "All three checkpoints were trained for 100 epochs. These are existing FP16 AMP\n"
         "Eval.py outputs, saved as 8-bit PNGs. The same target, crop coordinates, grayscale\n"
         "mapping and display enlargement are used for all methods. No inference was rerun."),
        ("Regions of interest", "Inspect curved silhouettes and boundaries for bright fringes, and dense foliage for\n"
         "overlapping edge errors or patterned texture. Original RGB crops are preserved; the\n"
         "grayscale row applies exactly the same contrast mapping to target and predictions."),
        ("Interpretation of differences", "Check the target before identifying an added halo: highlights, blur, displaced edges\n"
         "and fine texture can also change a boundary. A signed-error stripe alone is not proof\n"
         "of ringing, and a spatially irregular error need not be random noise."),
        ("Interpretation limits", "Examples were selected for visual inspection, not as a representative statistical sample.\n"
         "There is no native-LR-processing control or training-duration sweep here. Consequently,\n"
         "these images cannot establish why noise occurs or whether further training removes it."),
        ("Reproducibility", "Companion files contain lossless original RGB crops, identically transformed grayscale\n"
         "crops, per-page PNG/PDF figures and a JSON manifest with coordinates, source hashes,\n"
         "checkpoint hashes and evaluation policies. The selected RGB panel is also exported\n"
         "without a report title or footer for inclusion in the research manuscript."),
    ]
    if output_name == "validation_artifact_comparison_v2":
        sections[1] = ("Selected comparisons", "The approved curve and flower from scene0045 are retained. Two new, tightly cropped\n"
                       "curved outlines replace the broad character and foliage examples: a shoulder in\n"
                       "scene0027 and a leg contour in scene0242, against relatively flat backgrounds.")
        sections[2] = ("Qualitative observation", "Haloing is significantly reduced, not eliminated, in the adaptive-head examples.\n"
                       "Faint edge fringing remains. The residual and depth-to-space examples show more\n"
                       "prominent edge-associated fringes; this is a visual comparison, not a significance test.")
        sections[3] = ("Selection and scope", "The two new crops were manually checked after output-informed screening of 24\n"
                       "uniformly spaced validation frames. They are illustrative examples, not a random\n"
                       "sample. Native-LR processing and longer training are not controlled by these figures.")
    elif output_name == "validation_artifact_comparison_v3":
        sections[1] = ("Selected comparisons", "The approved curve and flower from scene0045 are retained. The two additional\n"
                       "examples are your selections from the numbered screening sheets: candidate 10\n"
                       "from scene0135 and candidate 13 from scene0310. The shoulder and leg are omitted.")
        sections[2] = ("Qualitative observation", "Haloing is significantly reduced, not eliminated, in the adaptive-head examples.\n"
                       "Faint edge fringing remains. The residual and depth-to-space examples show more\n"
                       "prominent edge-associated fringes; this is a visual comparison, not a significance test.")
        sections[3] = ("Selection and scope", "Candidates 10 and 13 were selected by the author after output-informed screening\n"
                       "of 24 uniformly spaced validation frames. These are illustrative examples, not a\n"
                       "random sample. Native-LR processing and longer training are not controlled here.")
    y = .76
    for title, body in sections:
        fig.text(.09, y, title, fontsize=14, fontweight="bold")
        fig.text(.09, y - .035, body, fontsize=12, va="top", linespacing=1.5)
        y -= .14
    return fig


def summary_figure(crops, paper=False):
    ratios = [record["crop"][3] / record["crop"][2] for record in crops]
    fig, axes = plt.subplots(len(crops), 4, figsize=(11, sum(ratios) * 2.5 + (0.5 if paper else 1.5)),
                             gridspec_kw={"height_ratios": ratios}, constrained_layout=True, squeeze=False)
    for row, record in enumerate(crops):
        x, y, width, height = record["crop"]
        stem = f"{Path(record['image']).stem}_{record['kind']}_{x}_{y}_{width}_{height}"
        for column, label in enumerate(LABELS):
            with Image.open(OUTPUT / "crops" / stem / f"{label}_rgb.png") as image:
                axes[row, column].imshow(np.asarray(image), interpolation="nearest")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(label, fontsize=13)
        names = {"candidate-10": "Curved trim", "candidate-13": "Thin structure"}
        name = names.get(record["kind"], record["kind"].replace("-", " ").capitalize())
        axes[row, 0].set_ylabel(f"{name}\n{Path(record['image']).stem}\n{width} x {height} HR pixels",
                                fontsize=10)
    if paper:
        fig.savefig(OUTPUT / "rgb_comparison_paper.pdf", bbox_inches="tight", pad_inches=0.03)
        fig.savefig(OUTPUT / "rgb_comparison_paper.png", dpi=180, bbox_inches="tight", pad_inches=0.03)
        return fig
    fig.suptitle("Output-head comparison / matched validation crops", fontsize=17)
    fig.supxlabel("Original RGB crops from saved FP16 AMP validation images. Identical coordinates and nearest-sampled enlargement.\n"
                  "Adaptive is the original architecture; no contrast adjustment or sharpening is applied here.", fontsize=10)
    fig.savefig(OUTPUT / "rgb_comparison.png", dpi=180)
    fig.savefig(OUTPUT / "rgb_comparison.pdf")
    return fig


def candidates(records, specs, output_name):
    manifest = {"inference_run": False, "source": "Existing Eval.py FP16 AMP validation PNGs",
                "selection": "Exploratory; manually inspected targets and target-only automatic crops",
                "native_LR_control": "Absent; these examples do not establish a native-LR processing effect",
                "sources": {}, "crops": []}
    for head, record in records.items():
        manifest["sources"][head] = {"manifest": str(record["manifest"]),
                                     "manifest_sha256": checksum(record["manifest"]),
                                     "checkpoint": str(record["checkpoint"]),
                                     "checkpoint_sha256": checksum(record["checkpoint"]), "policy": record["policy"]}
    destination = ROOT / "output/pdf" / f"{output_name}.pdf"
    cache_name, arrays = None, None
    with PdfPages(destination) as pdf:
        fig = cover(output_name)
        pdf.savefig(fig)
        fig.savefig(OUTPUT / "cover.png", dpi=160)
        plt.close(fig)
        for spec in specs:
            fields = spec.split(":")
            name = fields[0]
            if not name.endswith(".png"):
                name += ".png"
            paths = [TARGETS / name, *(r["directory"] / name for r in records.values())]
            if name != cache_name:
                arrays = [read_rgb(path) for path in paths]
                cache_name = name
                if any(image.shape != arrays[0].shape for image in arrays):
                    raise ValueError("Target/output geometry mismatch")
            if len(fields) == 2:
                kind = fields[1]
                x, y, size = (select_flat if kind == "flat" else select_edge)(arrays[0], 96)
                crop = (x, y, size, size)
            elif len(fields) == 5:
                x, y, size = (int(value) for value in fields[1:4])
                crop = (x, y, size, size)
                kind = fields[4]
            elif len(fields) == 6:
                crop = tuple(int(value) for value in fields[1:5])
                kind = fields[5]
            else:
                raise ValueError("Use scene:flat|edge, scene:x:y:size:kind, or scene:x:y:width:height:kind")
            gain = 6 if kind == "flat" else 2
            fig = render_crop(name, arrays, crop, kind, gain)
            stem = f"{Path(name).stem}_{kind}_{crop[0]}_{crop[1]}_{crop[2]}_{crop[3]}"
            crop_directory = OUTPUT / "crops" / stem
            crop_directory.mkdir(parents=True, exist_ok=True)
            x, y, width, height = crop
            center = float(luma(arrays[0][y:y + height, x:x + width]).mean())
            display_records = []
            for label, image in zip(LABELS, arrays):
                rgb = image[y:y + height, x:x + width]
                gray_display = .5 + gain * (luma(rgb) - center)
                Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(crop_directory / f"{label}_rgb.png")
                Image.fromarray(np.rint(np.clip(gray_display, 0, 1) * 255).astype(np.uint8)).save(
                    crop_directory / f"{label}_gray.png")
                display_records.append({"label": label, "grayscale_display_clipped_fraction":
                                        float(np.mean((gray_display < 0) | (gray_display > 1)))})
            fig.savefig(OUTPUT / f"{stem}.png", dpi=160)
            fig.savefig(OUTPUT / f"{stem}.pdf")
            pdf.savefig(fig)
            plt.close(fig)
            manifest["crops"].append({"image": name, "kind": kind, "crop": crop, "grayscale_gain": gain,
                                       "grayscale_center": center, "display_diagnostics": display_records,
                                       "files": [{"label": label, "path": str(path), "sha256": checksum(path)}
                                                 for label, path in zip(LABELS, paths)]})
            print(stem, flush=True)
        if output_name == "validation_curved_edges_and_texture_candidates":
            manifest["selection"] = "User-identified scene0045 regions, located from the provided screenshot crops"
        elif output_name == "validation_artifact_comparison_v2":
            screening = OUTPUT.parent / "artifact_candidate_screening/search_candidates.json"
            manifest["selection"] = "User-approved scene0045 curve/flower plus manually reviewed scene0027 shoulder and scene0242 leg"
            manifest["qualitative_wording"] = "Significantly reduced haloing, not eliminated; no statistical significance claim"
            manifest["screening"] = {"file": str(screening), "sha256": checksum(screening),
                                      "method": "Output-informed positive edge-adjacent error ranking; 24 uniformly spaced frames"}
        elif output_name == "validation_artifact_comparison_v3":
            screening = OUTPUT.parent / "artifact_candidate_screening/search_candidates.json"
            candidate_records = json.loads(screening.read_text(encoding="utf-8"))["candidates"]
            selected = [candidate_records[number - 1] for number in (10, 13)]
            actual = manifest["crops"][2:]
            if any(a["image"] != b["image"] or tuple(a["crop"]) != tuple(b["crop"])
                   for a, b in zip(actual, selected)) or len(actual) != 2:
                raise ValueError("Crops do not match the author's selected candidates 10 and 13")
            manifest["selection"] = "User-approved scene0045 curve/flower plus author-selected screening candidates 10 and 13"
            manifest["selected_screening_numbers"] = [10, 13]
            manifest["qualitative_wording"] = "Significantly reduced haloing, not eliminated; no statistical significance claim"
            manifest["screening"] = {"file": str(screening), "sha256": checksum(screening),
                                      "method": "Output-informed positive edge-adjacent error ranking; 24 uniformly spaced frames"}
        if manifest["crops"]:
            fig = summary_figure(manifest["crops"])
            pdf.savefig(fig)
            plt.close(fig)
            fig = summary_figure(manifest["crops"], paper=True)
            plt.close(fig)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(destination, flush=True)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overview", action="store_true")
    parser.add_argument("--crop", action="append", default=[])
    parser.add_argument("--locate", type=Path)
    parser.add_argument("--locate-count", type=int, default=50)
    parser.add_argument("--scene", help="Restrict reference matching to this scene stem")
    parser.add_argument("--output-name", default="validation_output_head_visual_candidates")
    parser.add_argument("--search-count", type=int, default=0)
    args = parser.parse_args()
    if not args.output_name.replace("_", "").replace("-", "").isalnum():
        parser.error("Output name must contain only letters, digits, underscores or hyphens")
    if args.output_name != "validation_output_head_visual_candidates":
        OUTPUT = OUTPUT / args.output_name
    cv2.setNumThreads(2)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = sources()
    if args.overview:
        overview(records)
    if args.crop:
        candidates(records, args.crop, args.output_name)
    if args.locate:
        locate(args.locate, args.locate_count, args.scene)
    if args.search_count:
        search_candidates(records, args.search_count)


if __name__ == "__main__":
    main()
