"""Checkpoint loading, RGB image I/O and phase-aligned tiled inference."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .model import AEISU, FoldedAEISU, fold_model

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def read_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255)


def save_rgb(path: Path, image: torch.Tensor) -> None:
    array = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    array = np.floor(array * 255 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_state(payload):
    if isinstance(payload, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(payload.get(key), dict):
                return payload[key]
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise ValueError("Checkpoint does not contain a tensor state dictionary")


def load_model(path: Path, device: str = "cpu") -> FoldedAEISU:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint_state(payload)
    scale = payload.get("basis_l1_scale", payload.get("config", {}).get("basis_l1_scale", 0.5))
    if any(".rep_low_u" in key for key in state):
        training_model = AEISU(basis_l1_scale=scale)
        training_model.load_state_dict(state, strict=True)
        model = fold_model(training_model)
    else:
        model = FoldedAEISU(basis_l1_scale=scale)
        model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)


def resolve_device(name: str) -> str:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu")
        if torch.version.cuda != "13.0":
            raise ValueError("This release targets the PyTorch CUDA 13.0 build")
    return name


@torch.inference_mode()
def upscale(model, source: torch.Tensor, tile: int = 128) -> torch.Tensor:
    """Match symmetric align32/halo16 padding; internal tiles use a 32-LR halo.

    Tiles are anchored on the globally padded grid, preserving unshuffle parity.
    tile=0 evaluates the whole padded image; clipping occurs after reconstruction.
    """
    if source.ndim != 3 or source.shape[0] != 3:
        raise ValueError("Expected CHW RGB image")
    if tile < 0 or tile % 32:
        raise ValueError("tile must be zero or a positive multiple of 32")
    h, w = source.shape[-2:]
    extra_y, extra_x = -h % 32, -w % 32
    py, px = 16 + extra_y // 2, 16 + extra_x // 2
    image = F.pad(source[None], (px, 32 + extra_x - px, py, 32 + extra_y - py), mode="replicate")
    image = image.to(next(model.parameters()).device)
    ph, pw = image.shape[-2:]
    if tile == 0:
        result = model(image)
    else:
        # CPU output bounds device memory independently of output resolution.
        result = torch.empty(1, 3, ph * 2, pw * 2, dtype=torch.float32)
        for y in range(0, ph, tile):
            for x in range(0, pw, tile):
                y1, x1 = min(y + tile, ph), min(x + tile, pw)
                sy, sx = max(0, y - 32), max(0, x - 32)
                ey, ex = min(ph, y1 + 32), min(pw, x1 + 32)
                prediction = model(image[:, :, sy:ey, sx:ex])
                core = prediction[:, :, 2 * (y - sy):2 * (y1 - sy), 2 * (x - sx):2 * (x1 - sx)]
                result[:, :, 2 * y:2 * y1, 2 * x:2 * x1] = core.float().cpu()
    output = result[0, :, 2 * py:2 * (py + h), 2 * px:2 * (px + w)].float().cpu()
    if not torch.isfinite(output).all():
        raise FloatingPointError("Non-finite model reconstruction")
    return output.clamp(0, 1)
