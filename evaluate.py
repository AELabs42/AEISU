"""Evaluate portable AEISU on paired images; save float outputs and per-image scores."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from aeisu.data import paired_images
from aeisu.metrics import build_lpips, distortion, perceptual_score
from aeisu.runtime import load_model, read_rgb, resolve_device, save_rgb, sha256, upscale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--crop", type=int, default=2)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--allow-download", action="store_true", help="Allow pretrained AlexNet download for LPIPS")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty evaluation output folder")
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    network = build_lpips(device, args.allow_download) if args.lpips else None
    pairs = paired_images(args.input, args.target)
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for index, (source, target) in enumerate(pairs):
        prediction = upscale(model, read_rgb(source), args.tile)
        reference = read_rgb(target)
        scores = distortion(prediction, reference, args.crop)
        if network is not None:
            scores["lpips_alex_v01"] = perceptual_score(network, prediction, reference, args.crop)
        identity = source.relative_to(args.input).with_suffix("")
        destination = args.output / "images" / identity
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_rgb(destination.parent / (destination.name + ".png"), prediction)
        np.savez_compressed(destination.parent / (destination.name + ".npz"), rgb=prediction.permute(1, 2, 0).numpy())
        rows.append({"image": identity.as_posix(), **scores})
        print(f"{index + 1}/{len(pairs)} {identity}: Y PSNR={scores['y_psnr']:.4f}", flush=True)
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "image"}
    # Encode exact-equality infinity explicitly, producing standards-compliant JSON.
    means = {key: value if math.isfinite(value) else "Infinity" for key, value in means.items()}
    summary = {"pairs": len(rows), "mean_per_image": means, "aggregation": "equal per-image weighting",
               "checkpoint_sha256": sha256(args.checkpoint), "precision": "FP32", "border_crop_hr": args.crop,
               "outputs": "clipped float32 before half-up PNG rounding", "device": device, "tile_lr": args.tile}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
