"""Checkpoint adapters and common software-FP8 stress test for core controls."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "ablations/_shared"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(SHARED / "snapshot"))
from models import build_raw, exact_bilinear
import adaptive_cuda_kernel

from common import read_rgb, sha256

VARIANTS = ("reference", "fixed_inputs", "fixed_base", "dense_core")
FOLDERS = {"fixed_inputs": "02_fixed_analysis", "fixed_base": "03_fixed_bilinear_base", "dense_core": "01_dense64x2_270p"}
REFERENCE = ROOT / "Models/Training Run 2026-09-30 17-53-21 - aif_reparam_aggressive64_parallel32x4/Pytorch Models/aif_reparam_aggressive64_parallel32x4_ep0100.pt"


def checkpoint_for(name):
    if name == "reference":
        return REFERENCE
    folder = ROOT / "ablations" / FOLDERS[name]
    runs = sorted((folder / "Models").glob("Training Run *"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(folder)
    return next((runs[-1] / "Pytorch Models").glob("*_ep0100.pt"))


def quantize(value, scale):
    return (value / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


class FP8Conv(nn.Module):
    """FP8 rounding/dequantization followed by FP32 arithmetic, not native FP8 timing."""

    def __init__(self, layer, activation_scale):
        super().__init__()
        scale = layer.weight.detach().abs().amax((1, 2, 3), keepdim=True).clamp_min(1e-12) / 448
        self.register_buffer("weight", quantize(layer.weight.detach(), scale))
        self.register_buffer("bias", layer.bias.detach().clone() if layer.bias is not None else None)
        self.register_buffer("activation_scale", torch.tensor(activation_scale, device=layer.weight.device))
        self.stride, self.padding, self.dilation, self.groups = layer.stride, layer.padding, layer.dilation, layer.groups

    def forward(self, value):
        return F.conv2d(quantize(value.float(), self.activation_scale), self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


class ControlAdapter:
    def __init__(self, name, checkpoint=None, epochs=100, gradient_phase=None, expected_settings=None, expected_updates=None):
        self.name = name
        path = Path(checkpoint) if checkpoint is not None else checkpoint_for(name)
        state = torch.load(path, map_location="cpu", weights_only=True)
        updates = epochs * 875 if expected_updates is None else expected_updates
        if state["epoch"] != epochs or (updates and state["global_step"] != updates):
            raise ValueError(f"Expected completed {epochs}-epoch checkpoint")
        actual_phase = state.get("experiment", {}).get("gradient_policy")
        if actual_phase is not None and gradient_phase is not None and actual_phase != gradient_phase:
            raise ValueError(f"Recorded gradient policy {actual_phase} differs from requested {gradient_phase}")
        actual_phase = actual_phase or gradient_phase or ("fixed" if name == "reference" else "cycle")
        cfg = state["config"]
        if (cfg["lr"], cfg["min_lr"], cfg["batch"], cfg["patch"], cfg["seed"]) != (.002, .000001, 32, 512, 1234):
            raise ValueError("Unexpected training recipe")
        settings = state.get("experiment", {}).get("model", {})
        expected = {} if checkpoint is not None else {
            "reference": {}, "fixed_inputs": {"analysis": "fixed"}, "fixed_base": {"learned_base": False},
            "dense_core": {"backbone": "dense64x2"}}[name]
        if expected_settings is not None:
            expected = expected_settings
        if settings != expected:
            raise ValueError("Unexpected model variant")
        raw = build_raw(settings).eval().requires_grad_(False)
        raw.load_state_dict(state["model"], strict=True)
        frozen_error = None
        if name == "fixed_base":
            frozen_error = float((raw.effective_base() - exact_bilinear(5)).abs().max())
            torch.testing.assert_close(raw.effective_base(), exact_bilinear(5), atol=1e-8, rtol=0)
        if name == "fixed_inputs":
            original = build_raw({"analysis": "fixed"})
            for key, value in raw._analysis_values().items():
                # Identity-preserving low-rank initialization has FP32
                # cancellation roundoff even when all analysis weights freeze.
                torch.testing.assert_close(value, original._analysis_values()[key], atol=3e-7, rtol=0)
            frozen_error = max(float((v - original._analysis_values()[k]).abs().max()) for k, v in raw._analysis_values().items())
        torch.manual_seed(20261005)
        probe = torch.rand(1, 3, 32, 40)
        with torch.no_grad():
            expected_meta = raw(probe)
            raw._remove_reparameterization()
            raw.spec["reparameterization"] = False
            actual_meta = raw(probe)
        torch.testing.assert_close(actual_meta, expected_meta, atol=3e-5, rtol=3e-5)
        self.model = raw.cuda().to(memory_format=torch.channels_last)
        self.base = self.model.effective_base().float().contiguous()
        self.basis = self.model.effective_basis().float().contiguous()
        self.metadata = {"checkpoint": str(path), "checkpoint_sha256": sha256(path), "epoch": epochs, "updates": state["global_step"],
                         "config": cfg, "model": settings,
                         "gradient_phase": actual_phase,
                         "training_backend": state.get("experiment", {}).get("backend", "fused_cuda_16x5 (historical)"),
                         "frozen_initialization_max_roundoff": frozen_error,
                         "fold_max_absolute_error": float((actual_meta - expected_meta).abs().max()),
                         "folded_parameters": sum(v.numel() for v in self.model.folded_production_state_dict().values())}
        for loss in ("mse", "l1"):
            if "train_" + loss in state["metrics"]:
                self.metadata["training_" + loss] = state["metrics"]["train_" + loss]
        self.metadata["training_loss"] = state.get("experiment", {}).get("loss", "RGB MSE only")
        self.quantized = None

    @torch.inference_mode()
    def calibrate(self, paths):
        maxima = {}
        handles = []
        for name, layer in self.model.named_children():
            if isinstance(layer, nn.Conv2d):
                maxima[name] = torch.tensor(0., device="cuda")

                def collect(module, args, key=name):
                    maxima[key] = torch.maximum(maxima[key], args[0].detach().abs().amax())

                handles.append(layer.register_forward_pre_hook(collect))
        try:
            for path in paths:
                rgb = read_rgb(path).astype(np.float32) / 255
                h, w = rgb.shape[:2]
                y, x = (h // 2 - 64) // 4 * 4, (w // 2 - 64) // 4 * 4
                source = torch.from_numpy(rgb[y:y + 128, x:x + 128].copy()).permute(2, 0, 1)[None].cuda()
                self.model(source)
        finally:
            for handle in handles:
                handle.remove()
        scales = {name: max(float(value) / 448, 1e-12) for name, value in maxima.items()}
        self.quantized = copy.deepcopy(self.model)
        for name, scale in scales.items():
            setattr(self.quantized, name, FP8Conv(getattr(self.quantized, name), scale))
        self.metadata["fp8_activation_scales"] = scales
        return scales

    @torch.inference_mode()
    def predict(self, rgb, precision="fp16"):
        source = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].cuda().float()
        h, w = source.shape[-2:]
        # Match the established Eval.py symmetric align32 + replicate halo16
        # convention, rather than exposing different-depth cores to image edges.
        py, px = 16 + (-h % 32) // 2, 16 + (-w % 32) // 2
        source = F.pad(source, (px, 32 + (-w % 32) - px, py, 32 + (-h % 32) - py),
                       mode="replicate").contiguous(memory_format=torch.channels_last)
        model = self.model if precision == "fp16" else self.quantized
        if model is None:
            raise ValueError("Model has not been calibrated")
        with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
            meta = model(source)
        result = adaptive_cuda_kernel.adaptive_forward(source.contiguous(), meta, self.base, self.basis)
        result = result[:, :, py * 2:py * 2 + h * 2, px * 2:px * 2 + w * 2].clamp(0, 1).float()
        return result[0].permute(1, 2, 0).cpu().numpy()

    @torch.inference_mode()
    def validate(self, precisions=("fp16", "fp8_stress")):
        rng = np.random.default_rng(20261005)
        rgb = rng.random((32, 40, 3), dtype=np.float32)
        source = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].cuda()
        meta = self.model(source)
        reference = self.model.differentiable_predict_rgb(source, meta).clamp(0, 1)
        fused = adaptive_cuda_kernel.adaptive_forward(source.contiguous(), meta, self.base, self.basis).clamp(0, 1)
        maximum = float((reference - fused).abs().max())
        if maximum > 5e-5:
            raise ValueError(f"Reconstruction parity failed: {maximum}")
        outputs = {precision: self.predict(rgb, precision) for precision in precisions}
        if any(p.shape != (64, 80, 3) or not np.isfinite(p).all() for p in outputs.values()):
            raise ValueError("Invalid output shape or values")
        return {"fused_vs_tiled_max_abs": maximum, "finite_outputs": True, "output_shape": [64, 80, 3]}
