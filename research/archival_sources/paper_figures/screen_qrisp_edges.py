"""CPU-only candidate galleries from frozen QRISP reconstructions."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/paper_bench"))
from common import read_rgb, sha256, write_json
from final_benchmark_data import DEFAULT_OUTPUT, input_record, read_json

OUT = ROOT / "output/figures/final_qrisp_edge_candidates"
METHODS = (("aif_mixed", "Ours (FP8/FP16)"), ("fsr1_easu", "FSR1 EASU"),
           ("quicksrnet_small_x2", "QuickSRNet Small"), ("rt4ksr_x2", "RT4KSR"),
           ("span_x2", "SPAN"), ("swinir_light_x2", "SwinIR-Light"))
CONDITIONS = ("ssaa_area", "smaa", "fxaa")
LABELS = {"ssaa_area": "SSAA / area input", "smaa": "SMAA input", "fxaa": "FXAA input"}


def targets_to_screen(selection):
    buckets = defaultdict(list)
    for pair in selection["pairs"]:
        buckets[(pair["scene"], pair["segment"])].append(pair)
    chosen = []
    for pairs in buckets.values():
        pairs.sort(key=lambda p: p["frame"])
        count = min(3, len(pairs))
        chosen.extend(pairs[int((i + .5) * len(pairs) / count)] for i in range(count))
    return chosen


def scan_pair(pair):
    record = input_record(DEFAULT_OUTPUT, pair)
    rgb = read_rgb(Path(record["target"]))
    y = rgb.astype(np.float32) @ np.array([.299, .587, .114], np.float32) / 255
    local = cv2.dilate(y, np.ones((5, 5), np.uint8)) - cv2.erode(y, np.ones((5, 5), np.uint8))
    wide = cv2.dilate(y, np.ones((25, 25), np.uint8)) - cv2.erode(y, np.ones((25, 25), np.uint8))
    flat = local < .025
    adjacent = flat & (wide > .18)
    smooth = cv2.GaussianBlur(y, (5, 5), .8)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    edges = magnitude > .22
    diagonal = edges & (np.minimum(np.abs(gx), np.abs(gy)) > .18 * np.maximum(np.abs(gx), np.abs(gy)))
    maps = [cv2.integral(a.astype(np.float32)) for a in (flat, adjacent, edges, diagonal)]
    candidates = []
    size = 96
    for top in range(32, y.shape[0] - size - 32, 32):
        for left in range(32, y.shape[1] - size - 32, 32):
            sums = [(a[top + size, left + size] - a[top, left + size] - a[top + size, left] + a[top, left])
                    / size ** 2 for a in maps]
            flat_fraction, near_fraction, edge_fraction, diag_fraction = sums
            if flat_fraction < .65 or not .08 < near_fraction < .55 or not .012 < edge_fraction < .12:
                continue
            if diag_fraction < .004:
                continue
            patch = y[top:top + size, left:left + size]
            contrast = float(np.percentile(patch, 95) - np.percentile(patch, 5))
            if contrast < .20:
                continue
            score = float(near_fraction * contrast * flat_fraction * (1 - edge_fraction / .15) *
                          (.5 + min(diag_fraction / max(edge_fraction, 1e-6), 1)))
            candidates.append({"image": pair["image"], "scene": pair["scene"], "segment": pair["segment"],
                               "group": pair["group"], "xywh": [left, top, size, size], "screen_score": score,
                               "target": record["target"], "target_sha256": record["target_sha256"],
                               "flat_fraction": float(flat_fraction), "edge_fraction": float(edge_fraction)})
    return sorted(candidates, key=lambda c: c["screen_score"], reverse=True)[:12]


def search():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = targets_to_screen(read_json(DEFAULT_OUTPUT / "selection.json"))
    candidates = []
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(scan_pair, pairs), 1):
            candidates.extend(result)
            if index % 20 == 0:
                print(f"Source-only edge screening {index}/{len(pairs)}", flush=True)
    candidates.sort(key=lambda c: c["screen_score"], reverse=True)
    selected, scene_counts, segment_counts, image_counts = [], defaultdict(int), defaultdict(int), defaultdict(int)
    for candidate in candidates:
        scene, segment, identity = candidate["scene"], (candidate["scene"], candidate["segment"]), candidate["image"]
        if scene_counts[scene] >= 6 or segment_counts[segment] >= 2 or image_counts[identity] >= 2:
            continue
        x, y, _, _ = candidate["xywh"]
        if any(c["image"] == identity and abs(c["xywh"][0] - x) < 128 and abs(c["xywh"][1] - y) < 128 for c in selected):
            continue
        selected.append(candidate)
        scene_counts[scene] += 1
        segment_counts[segment] += 1
        image_counts[identity] += 1
        if len(selected) == 24:
            break
    for index, candidate in enumerate(selected):
        candidate.update({"number": index + 1, "condition": CONDITIONS[index % 3]})
    record = {"selection": "Target-only edge/flatness screening; up to three uniform frames per sequence in the frozen 600-pair study",
              "frames_screened": len(pairs), "candidate_count": len(selected), "candidates": selected,
              "conditions": CONDITIONS, "benchmark": str(DEFAULT_OUTPUT), "gpu_work": False,
              "display": "Matched 96x96 HR coordinates; common RGB8 half-up rounding; 2x nearest enlargement; unchanged colors",
              "scope": "Illustrative candidates, not an artifact-frequency estimate; no method scores enter screening"}
    write_json(OUT / "candidates.json", record)
    write_json(OUT / "screened_candidates.json", record)
    overview(selected)


def fonts(size, bold=False):
    return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", size)


def overview(selected):
    canvas = Image.new("RGB", (6 * 210, 4 * 254), "white")
    draw = ImageDraw.Draw(canvas)
    for i, case in enumerate(selected):
        x, y, w, h = case["xywh"]
        rgb = read_rgb(Path(case["target"]))[y:y + h, x:x + w]
        col, row = i % 6, i // 6
        canvas.paste(Image.fromarray(rgb).resize((192, 192), Image.Resampling.NEAREST), (col * 210 + 8, row * 254 + 28))
        draw.text((col * 210 + 8, row * 254 + 5), f"#{case['number']}", fill="black", font=fonts(18, True))
        draw.text((col * 210 + 8, row * 254 + 222), case["scene"][:22], fill="black", font=fonts(12))
    canvas.save(OUT / "target_overview.png")


def context():
    """Render source contexts for manual geometric crop selection."""
    pairs = targets_to_screen(read_json(DEFAULT_OUTPUT / "selection.json"))
    pairs = [p for p in pairs if p["scene"] in ("SunTempleBush", "SeaPort")]
    records = []
    for batch in range((len(pairs) + 23) // 24):
        canvas = Image.new("RGB", (4 * 328, 6 * 208), "white")
        draw = ImageDraw.Draw(canvas)
        for index, pair in enumerate(pairs[batch * 24:(batch + 1) * 24]):
            number = batch * 24 + index + 1
            record = input_record(DEFAULT_OUTPUT, pair)
            bitmap = Image.fromarray(read_rgb(Path(record["target"])))
            col, row = index % 4, index // 4
            canvas.paste(bitmap.resize((320, 180), Image.Resampling.LANCZOS), (col * 328, row * 208 + 24))
            draw.text((col * 328, row * 208 + 2), f"{number}: {pair['image']}", fill="black", font=fonts(12))
            records.append({"number": number, **pair, "target": record["target"], "target_sha256": record["target_sha256"]})
        canvas.save(OUT / f"context_{batch + 1}.png")
    write_json(OUT / "context_records.json", records)


def curate():
    """Mix manually inspected curves with nonredundant source-screened edges."""
    existing = read_json(OUT / "screened_candidates.json")["candidates"]
    pairs = {p["image"]: p for p in read_json(DEFAULT_OUTPUT / "selection.json")["pairs"]}
    specs = [
        ("SunTempleBush/0008/0002.png", [1680, 600, 96, 96], "Scroll-shaped stone: upper curve"),
        ("SunTempleBush/0008/0002.png", [1496, 668, 96, 96], "Scroll-shaped stone: rounded end"),
        ("SeaPort/0000/0067.png", [1516, 280, 96, 96], "Curved lamp bracket against sky"),
        ("SeaPort/0000/0067.png", [1000, 144, 96, 96], "Branch against overcast sky"),
        ("SunTempleBush/0008/0002.png", [1576, 32, 96, 96], "Stone finial against blue sky"),
        ("SeaPort/0000/0067.png", [1456, 336, 96, 96], "Lamp silhouette against sky"),
    ]
    selected = []
    for identity, xywh, description in specs:
        pair = pairs[identity]
        record = input_record(DEFAULT_OUTPUT, pair)
        selected.append({"image": identity, "scene": pair["scene"], "segment": pair["segment"],
                         "group": pair["group"], "xywh": xywh, "description": description,
                         "target": record["target"], "target_sha256": record["target_sha256"]})
    for old_number, description in [(13, "Roof edge against grey sky"), (1, "High-contrast diagonal trim"),
                                    (10, "Diagonal panel against flat background"), (19, "Thin branch against blue sky"),
                                    (12, "Building silhouette against bright background"), (24, "Wall and railing edges")]:
        case = dict(next(c for c in existing if c["number"] == old_number))
        case["description"] = description
        selected.append(case)
    for index, case in enumerate(selected):
        case.update(number=index + 1, condition="ssaa_area")
    write_json(OUT / "candidates.json", {
        "selection": "Source-only edge screening followed by manual source-context selection of curved and diagonal edges",
        "frames_screened": 164, "candidate_count": len(selected), "candidates": selected,
        "benchmark": str(DEFAULT_OUTPUT), "gpu_work": False, "conditions": ["ssaa_area"],
        "display": "Matched 96x96 HR coordinates; common RGB8 half-up rounding; 2x nearest enlargement; unchanged colors",
        "scope": "Illustrative candidates, not a representative artifact-frequency estimate; no scores used for screening"})
    overview(selected)


def render():
    selected = read_json(OUT / "candidates.json")["candidates"]
    lookup = {}
    for method, _ in METHODS:
        with (DEFAULT_OUTPUT / "results" / method / "per_image.jsonl").open(encoding="utf-8") as stream:
            lookup[method] = {(row["image"], row["condition"]): row for line in stream if (row := json.loads(line))}
    manifest = []
    for batch in range((len(selected) + 5) // 6):
        sheet = Image.new("RGB", (7 * 206, 6 * 250 + 40), "white")
        draw = ImageDraw.Draw(sheet)
        for column, title in enumerate(["Native HR target", *[label for _, label in METHODS]]):
            draw.text((column * 206 + 5, 8), title, fill="black", font=fonts(18, True))
        for row_index, case in enumerate(selected[batch * 6:(batch + 1) * 6]):
            x, y, width, height = case["xywh"]
            if sha256(Path(case["target"])) != case["target_sha256"]:
                raise ValueError("Target hash mismatch")
            target = read_rgb(Path(case["target"]))
            panels = [("target", target[y:y + height, x:x + width], case["target"], case["target_sha256"])]
            input_hashes = set()
            for method, _ in METHODS:
                source = lookup[method][case["image"], case["condition"]]
                input_hashes.add(source["input_sha256"])
                path = Path(source["outputs"]["npz"])
                if source["target_sha256"] != case["target_sha256"] or sha256(path) != source["outputs"]["npz_sha256"]:
                    raise ValueError("Archive mismatch")
                with np.load(path, allow_pickle=False) as saved:
                    crop = saved["rgb"][y:y + height, x:x + width].copy()
                rgb = np.floor(np.clip(crop, 0, 1) * 255 + .5).astype(np.uint8)
                panels.append((method, rgb, str(path), source["outputs"]["npz_sha256"]))
            if len(input_hashes) != 1:
                raise ValueError("Compared methods used different source inputs")
            crop_folder = OUT / "crops" / str(case["number"])
            crop_folder.mkdir(parents=True, exist_ok=True)
            for col, (method, bitmap, source, digest) in enumerate(panels):
                image = Image.fromarray(bitmap)
                image.save(crop_folder / f"{method}.png")
                sheet.paste(image.resize((192, 192), Image.Resampling.NEAREST), (col * 206 + 5, row_index * 250 + 75))
                manifest.append({"number": case["number"], "method": method, "source": source, "source_sha256": digest,
                                 "crop": str(crop_folder / f"{method}.png"), "xywh": case["xywh"], "condition": case["condition"]})
            draw.text((5, row_index * 250 + 44),
                      f"#{case['number']}  {case['image']}  |  {LABELS[case['condition']]}  |  HR crop ({x}, {y})",
                      fill="black", font=fonts(19, True))
        sheet.save(OUT / f"candidates_{batch + 1}.png")
        print(f"Gallery {batch + 1}: candidates {batch * 6 + 1}-{min((batch + 1) * 6, len(selected))}", flush=True)
    write_json(OUT / "crop_manifest.json", manifest)


def native_check():
    """Inspect native inputs and saved outputs at preselected edge coordinates."""
    selected = read_json(OUT / "candidates.json")["candidates"]
    selected = [c for c in selected if c["number"] in (3, 4, 8, 9, 10, 11)]
    methods = (("bilinear", "Bilinear"), ("fsr1_easu", "FSR1 EASU"), ("aif_mixed", "Ours (FP8/FP16)"))
    lookup = {}
    for method, _ in methods:
        with (DEFAULT_OUTPUT / "results" / method / "per_image.jsonl").open(encoding="utf-8") as stream:
            lookup[method] = {(row["image"], row["condition"]): row for line in stream if (row := json.loads(line))}
    canvas = Image.new("RGB", (5 * 216, len(selected) * 280 + 40), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(["Native LR (nearest x2)", "Native HR target", *[label for _, label in methods]]):
        draw.text((column * 216 + 5, 8), label, fill="black", font=fonts(17, True))
    records = []
    for row_index, case in enumerate(selected):
        x, y, w, h = case["xywh"]
        record = input_record(DEFAULT_OUTPUT, {"image": case["image"]})
        native = record["inputs"]["native"]
        if sha256(Path(native["npz"])) != native["npz_sha256"]:
            raise ValueError("Native source hash mismatch")
        if sha256(Path(record["target"])) != case["target_sha256"]:
            raise ValueError("Target hash mismatch")
        with np.load(native["npz"], allow_pickle=False) as saved:
            source = saved["rgb"]
        source = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)[y:y + h, x:x + w]
        target = read_rgb(Path(record["target"]))[y:y + h, x:x + w].astype(np.float32) / 255
        panels = [source, target]
        scores = {}
        archive_records = []
        for method, _ in methods:
            row = lookup[method][case["image"], "native"]
            path = Path(row["outputs"]["npz"])
            assert row["input_sha256"] == native["npz_sha256"]
            assert row["target_sha256"] == record["target_sha256"]
            assert sha256(path) == row["outputs"]["npz_sha256"]
            with np.load(path, allow_pickle=False) as saved:
                prediction = saved["rgb"][y:y + h, x:x + w].copy()
            panels.append(prediction)
            delta = (prediction - target) @ np.array([.299, .587, .114], np.float32)
            mse = float(np.mean(delta.astype(np.float64) ** 2))
            scores[method] = {"crop_y_mse": mse, "crop_y_psnr_db": float(-10 * np.log10(max(mse, 1e-30))),
                              "whole_frame_metrics": row["metrics"]}
            archive_records.append({"method": method, "source": str(path), "sha256": row["outputs"]["npz_sha256"]})
        draw.text((5, row_index * 280 + 43), f"Candidate #{case['number']} | {case['image']} | Native input | ({x}, {y})",
                  fill="black", font=fonts(17, True))
        for column, pixels in enumerate(panels):
            bitmap = np.floor(np.clip(pixels, 0, 1) * 255 + .5).astype(np.uint8)
            canvas.paste(Image.fromarray(bitmap).resize((192, 192), Image.Resampling.NEAREST),
                         (column * 216 + 5, row_index * 280 + 73))
            if column >= 2:
                method = methods[column - 2][0]
                draw.text((column * 216 + 5, row_index * 280 + 269),
                          f"Crop Y PSNR: {scores[method]['crop_y_psnr_db']:.2f} dB", fill="black", font=fonts(15))
        records.append({"number": case["number"], "image": case["image"], "xywh": case["xywh"], "scores": scores,
                        "input_sha256": native["npz_sha256"], "target_sha256": record["target_sha256"], "outputs": archive_records})
    canvas.save(OUT / "native_edge_check.png")
    write_json(OUT / "native_edge_check.json", {"selection": "Six previously source-selected edge regions; no score-based crop selection",
               "metrics": "FP32 crop luma PSNR, unshaved; diagnostic, not an aggregate benchmark", "records": records})
    print(json.dumps([{ "number": r["number"], "psnr": {m: v["crop_y_psnr_db"] for m, v in r["scores"].items()}}
                      for r in records], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("search", "render", "context", "curate", "native", "all"), default="all", nargs="?")
    args = parser.parse_args()
    if args.command in ("search", "all"):
        search()
    if args.command in ("render", "all"):
        render()
    if args.command == "context":
        context()
    if args.command == "curate":
        curate()
    if args.command == "native":
        native_check()


if __name__ == "__main__":
    main()
