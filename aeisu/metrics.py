"""Encoded-Y PSNR/SSIM and optional full-resolution RGB LPIPS."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.nn import functional as F


def luma(rgb: torch.Tensor) -> torch.Tensor:
    weights = rgb.new_tensor([65.481, 128.553, 24.966]).reshape(3, 1, 1)
    return ((rgb * weights).sum(dim=0, keepdim=True) + 16) / 255


def distortion(prediction, target, crop=2):
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Prediction and target must be same-size CHW RGB")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Prediction and target must be finite")
    if crop < 0 or min(prediction.shape[-2:]) < 11 + crop * 2:
        raise ValueError("Images are too small for the border crop and 11x11 SSIM window")
    a, b = prediction.clamp(0, 1).double(), target.double()
    if crop:
        a, b = a[:, crop:-crop, crop:-crop], b[:, crop:-crop, crop:-crop]
    y1, y2 = luma(a)[None], luma(b)[None]
    mse = float((y1 - y2).square().mean())
    coordinate = torch.arange(11, dtype=torch.float64, device=y1.device) - 5
    gaussian = torch.exp(-coordinate.square() / (2 * 1.5 ** 2))
    gaussian = gaussian / gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :])[None, None]
    def smooth(image):
        return F.conv2d(image, window)
    mu1, mu2 = smooth(y1), smooth(y2)
    var1, var2 = smooth(y1.square()) - mu1.square(), smooth(y2.square()) - mu2.square()
    covariance = smooth(y1 * y2) - mu1 * mu2
    ssim = ((2 * mu1 * mu2 + .01 ** 2) * (2 * covariance + .03 ** 2) /
            ((mu1.square() + mu2.square() + .01 ** 2) * (var1 + var2 + .03 ** 2)))
    return {"y_mse": mse, "y_psnr": math.inf if mse == 0 else -10 * math.log10(mse), "y_ssim": float(ssim.mean())}


def build_lpips(device="cpu", allow_download=False):
    # torchvision's pretrained AlexNet may otherwise download on construction.
    cached = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    if not allow_download and not cached.is_file():
        raise FileNotFoundError("AlexNet weights are not cached; pass --allow-download to explicitly permit the download")
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("Optional LPIPS needs: uv add lpips") from exc
    return lpips.LPIPS(net="alex", version="0.1").to(device).eval().requires_grad_(False)


@torch.inference_mode()
def perceptual_score(network, prediction, target, crop=2):
    if crop:
        prediction = prediction[:, crop:-crop, crop:-crop]
        target = target[:, crop:-crop, crop:-crop]
    device = next(network.parameters()).device
    return float(network(prediction[None].to(device) * 2 - 1, target[None].to(device) * 2 - 1).item())
