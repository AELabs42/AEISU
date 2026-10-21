"""Reproducible area-pair and native-runtime helpers for paper measurements."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

ROOT = Path(r"{research_workspace}")
AESS = Path(r"{source_workspace}\AESS")
RUN = ROOT / "Models/Training Run 2026-09-26 23-06-26 - aif_reparam_aggressive64_parallel32x4"
CHECKPOINT = RUN / "Pytorch Models/aif_reparam_aggressive64_parallel32x4_ep1000_folded_production.pt"
ARTIFACT = AESS / "models/aif_reparam_aggressive64_parallel32x4/adaptive_fp8.bin"
EXE = AESS / "bin/Release/SrcPyrRuntimeTest.exe"
RESULTS = ROOT / "output/paper_measurements/2026-09-28"
DATASETS = ("Set5", "Set14", "B100", "Urban100")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def area_pair(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Aligned 2x2 integer box average, uint8 round-half-up; no existing LR use."""
    original = read_rgb(path)
    h, w = original.shape[:2]
    hr = original[:h - h % 2, :w - w % 2].copy()
    h2, w2 = hr.shape[:2]
    sums = hr.astype(np.uint16).reshape(h2 // 2, 2, w2 // 2, 2, 3).sum((1, 3), dtype=np.uint16)
    lr = ((sums + 2) // 4).astype(np.uint8)
    return lr, hr, {"original_hw": [h, w], "target_hw": [h2, w2], "crop_bottom_right": [h % 2, w % 2]}


def psnr(a: np.ndarray, b: np.ndarray) -> float | None:
    difference = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(difference * difference))
    return float(-10 * np.log10(mse)) if mse else None


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """Wang-style population moments, Gaussian 11x11/sigma1.5, valid RGB mean."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    blur = lambda value: cv2.GaussianBlur(value, (11, 11), 1.5)[5:-5, 5:-5]
    ma, mb = blur(a), blur(b)
    va, vb = blur(a * a) - ma * ma, blur(b * b) - mb * mb
    covariance = blur(a * b) - ma * mb
    value = ((2 * ma * mb + 0.01 ** 2) * (2 * covariance + 0.03 ** 2))
    value /= ((ma * ma + mb * mb + 0.01 ** 2) * (va + vb + 0.03 ** 2))
    return float(value.mean())


def quality(prediction: np.ndarray, target: np.ndarray) -> dict:
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite reconstruction")
    return {"rgb_mse": float(np.mean((prediction.astype(np.float64) - target.astype(np.float64)) ** 2)),
            "rgb_psnr_db": psnr(prediction, target), "rgb_ssim": ssim_rgb(prediction, target),
            "rgb_psnr_shave2_db": psnr(prediction[2:-2, 2:-2], target[2:-2, 2:-2])}


def native(rgb: np.ndarray, directory: Path, *, warmup: int = 1, iterations: int = 1,
           detailed: bool = False, output: bool = True, artifact: Path = ARTIFACT,
           environment_overrides: dict | None = None) -> tuple[dict, str]:
    directory.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    source = directory / "source_rgba_f32.bin"
    np.concatenate((rgb, np.ones((h, w, 1), np.float32)), axis=2).astype("<f4").tofile(source)
    command = [str(EXE), "--model", str(artifact), "--width", str(w), "--height", str(h),
               "--input", str(source), "--dump-dir", str(directory), "--warmup", str(warmup),
               "--iterations", str(iterations), "--timing-mode", "detailed" if detailed else "minimal",
               "--output-only" if output else "--no-dump", "--min-psnr", "0"]
    # The executable's synthetic PSNR threshold is not our evaluation gate.
    # Every measured error is retained, and FP32-reference correctness is checked separately.
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("AESS_"):
            del environment[name]
    environment.update(environment_overrides or {})
    result = subprocess.run(command, cwd=AESS, env=environment, capture_output=True, text=True, timeout=120)
    log = result.stdout + result.stderr
    (directory / "native.log").write_text(log, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Native failure {result.returncode}: {log[-3500:]}")
    outputs = {}
    if output:
        for precision in ("fp16", "fp8", "mixed"):
            path = directory / f"output_{precision}_rgba_f32.bin"
            rgba = np.fromfile(path, dtype="<f4").reshape(2 * h, 2 * w, 4)
            if not np.isfinite(rgba).all() or not np.all(rgba[..., 3] == 1):
                raise ValueError(f"Invalid {precision} output")
            outputs[precision] = rgba[..., :3].copy()
    return outputs, log


def clear_native_scratch(directory: Path) -> None:
    for name in ("source_rgba_f32.bin", "input_rgba_f32.bin", "output_fp16_rgba_f32.bin",
                 "output_fp8_rgba_f32.bin", "output_mixed_rgba_f32.bin"):
        (directory / name).unlink(missing_ok=True)


def provenance() -> dict:
    return {"checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
            "artifact": str(ARTIFACT), "artifact_sha256": sha256(ARTIFACT),
            "native_executable": str(EXE), "native_executable_sha256": sha256(EXE),
            "final_marker": str(RUN / "final.txt"), "native_artifact_calibration_policy": "mixed",
            "note": "No retraining, deployment modification, or reuse of bicubic LR images."}
