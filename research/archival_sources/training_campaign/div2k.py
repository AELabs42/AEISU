"""Download only authorized official DIV2K train HR/x2 data and prepare verified training pairs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_ROOT = ROOT / "data/train/div2k/official"
DOWNLOAD_ROOT = Path("G:/AEISU Research Data/DIV2K")
DATASET_PAGE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
URLS = {name: DATASET_PAGE + name + ".zip" for name in ("DIV2K_train_HR", "DIV2K_train_LR_bicubic_X2")}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_one(item):
    name, url = item
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    archive = DOWNLOAD_ROOT / (name + ".zip")
    partial = archive.with_suffix(".zip.part")
    if not archive.exists():
        first = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={first}-"} if first else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            append = first and response.status == 206
            if not append:
                first = 0
            total = first + int(response.headers.get("Content-Length", 0))
            last = 0.
            with partial.open("ab" if append else "wb") as output:
                while block := response.read(8 * 1024**2):
                    output.write(block)
                    first += len(block)
                    if time.monotonic() - last > 10:
                        print(f"{name}: {first / 1024**3:.2f}/{total / 1024**3:.2f} GiB", flush=True)
                        last = time.monotonic()
            if total and first != total:
                raise ValueError("Download length mismatch")
        os.replace(partial, archive)
    with zipfile.ZipFile(archive) as source:
        members = [m for m in source.infolist() if not m.is_dir() and m.filename.lower().endswith(".png")]
        if len(members) != 800:
            raise ValueError(f"Expected exactly 800 train images in {archive}")
        for member in members:
            target = (DATA_ROOT / member.filename).resolve()
            target.relative_to(DATA_ROOT.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                temporary = target.with_suffix(".png.tmp")
                with source.open(member) as stream, temporary.open("wb") as destination:
                    shutil.copyfileobj(stream, destination)
                os.replace(temporary, target)
            if target.stat().st_size != member.file_size:
                raise ValueError(f"Extracted size mismatch: {target}")
    return {"url": url, "archive": str(archive), "sha256": digest(archive), "training_images": 800}


def prepare(campaign):
    import cv2
    import numpy as np
    config_path = campaign / "data_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {"div2k_root": str(DATA_ROOT)}
    root = Path(config["div2k_root"])
    hr = root / "DIV2K_train_HR"
    lr_options = (root / "DIV2K_train_LR_bicubic/X2", root / "DIV2K_train_LR_bicubic_X2")
    lr = next((p for p in lr_options if p.is_dir()), lr_options[0])
    if not hr.is_dir() or not lr.is_dir():
        raise FileNotFoundError(f"Need official DIV2K train HR and bicubic X2 under {root}")
    expected = {f"{i:04d}.png" for i in range(1, 801)}
    if {p.name for p in hr.glob("*.png")} != expected:
        raise ValueError("Training split must be exactly DIV2K 0001..0800; no validation/test images")
    cv2.setNumThreads(1)
    data = campaign / "div2k"
    for folder in ("Input", "Target"):
        (data / folder).mkdir(parents=True, exist_ok=True)

    def pair(filename):
        high_path, low_path = hr / filename, lr / (Path(filename).stem + "x2.png")
        high, low = cv2.imread(str(high_path)), cv2.imread(str(low_path))
        if high is None or low is None or high.dtype != np.uint8 or low.dtype != np.uint8:
            raise ValueError(f"Invalid official DIV2K pair: {filename}")
        h, w = low.shape[:2]
        if high.shape[:2] != (h * 2, w * 2):
            raise ValueError(f"Unexpected pair geometry: {filename}: {high.shape}, {low.shape}")
        for source, destination in ((high_path, data / "Target" / filename), (low_path, data / "Input" / filename)):
            if not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            if digest(source) != digest(destination):
                raise ValueError(f"Prepared image differs: {destination}")
        return {"image": filename, "input_sha256": digest(low_path), "target_sha256": digest(high_path),
                "source_input": str(low_path), "source_target": str(high_path), "lr_hw": [h, w]}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(pair, sorted(expected)))
    manifest = {"dataset": "DIV2K train 0001..0800", "pairs": records, "scale": 2,
                "input": "Official MATLAB bicubic X2 LR images, not regenerated Pillow inputs",
                "dataset_page": DATASET_PAGE, "dataset_notice": "Academic research purpose only; original image copyrights retained",
                "checkpoint_selection": "Final scheduled epoch; no Set5/Set14/BSD100/Urban100 image enters training",
                "verified": True}
    (data / "pairs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Verified 800 official DIV2K train pairs; validation and test images excluded.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "prepare", "evaluate"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "download":
        with ThreadPoolExecutor(max_workers=2) as pool:
            records = dict(zip(URLS, pool.map(download_one, URLS.items())))
        (DOWNLOAD_ROOT / "download_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    elif args.action == "prepare":
        if args.campaign is None:
            parser.error("prepare requires --campaign")
        prepare(args.campaign)
    else:
        if args.campaign is None or args.output is None:
            parser.error("evaluate requires --campaign and --output")
        from evaluate_classic import evaluate
        evaluate(args.campaign, args.output)


if __name__ == "__main__":
    main()
