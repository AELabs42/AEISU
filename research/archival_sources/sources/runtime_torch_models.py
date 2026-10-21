"""Ordinary PyTorch inference adapters for the matched runtime experiment."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from common import CHECKPOINT, sha256
from comparison_models import neural_forward
from final_benchmark_models import Adapter, QUICK

RELEASE = Path("{source_workspace}/AEISU")
sys.path.insert(0, str(RELEASE))
from aeisu.runtime import load_model
from aeisu.reconstruction import reconstruct, unpack_coefficients

MODELS = ("aeisu_fp16", "aeisu_fp16_fp32_synthesis", "quicksrnet_small_x2", "quicksrnet_medium_x2",
          "repnet_x2", "rt4ksr_x2", "span_x2", "swinir_light_x2")


def analysis_kernel(model):
    """Fold the linear eight-channel analysis into a single fixed 3x3 convolution."""
    weight = torch.zeros(8, 3, 3, 3, device=model.base_kernel.device)
    weight[:3, :, 1, 1] = model.analysis_rgb_basis
    mix = model.analysis_rgb_mix
    weight[3, :, 1, 1] = mix
    dx, dy = model.analysis_dx_rows, model.analysis_dy_cols
    d45, d135 = model.analysis_diag45, model.analysis_diag135
    stencils = torch.zeros(4, 3, 3, device=weight.device)
    stencils[0, :, 0], stencils[0, :, 2] = -dx, dx
    stencils[1, 0, :], stencils[1, 2, :] = -dy, dy
    for row, values in ((2, (d45[0], d45[1], d45[2])), (3, (d135[0], d135[1], d135[2]))):
        a, b, c = values
        if row == 2:
            stencils[row, 0, 0], stencils[row, 2, 2] = -a, a
            stencils[row, 0, 1], stencils[row, 2, 1] = -b, b
        else:
            stencils[row, 0, 2], stencils[row, 2, 0] = a, -a
            stencils[row, 0, 1], stencils[row, 2, 1] = b, -b
        stencils[row, 1, 0], stencils[row, 1, 2] = -c, c
    weight[4:] = stencils[:, None] * mix[None, :, None, None]
    return weight


class TorchAEISU:
    """Folded checkpoint, grouped parallel convolutions, GEMM + batched-matmul synthesis.

    Static analysis/filter projection and paired-branch packing are prepared once.
    Padding, analysis, controller, coefficient clamp, synthesis and clipping are timed.
    No custom CUDA extension, torch.compile, fake FP8, or image tiling is used.
    """

    def __init__(self, fp32_synthesis=False):
        self.model = load_model(CHECKPOINT, "cuda")
        self.dtype = torch.float32 if fp32_synthesis else torch.float16
        self.base = self.model.effective_base().detach().to(self.dtype).contiguous()
        self.basis = self.model.effective_basis().detach().to(self.dtype).contiguous()
        self.bank_matrix = self.basis.permute(1, 0, 2).reshape(16, 100).contiguous()
        self.analysis = analysis_kernel(self.model).to(self.dtype).contiguous(memory_format=torch.channels_last)
        self.groups = []
        for i in range(1, 5):
            layer = nn.Conv2d(64, 64, 3, padding=1, groups=2).cuda().half().eval().requires_grad_(False)
            a, b = getattr(self.model, f"bottleneck_270_a{i}"), getattr(self.model, f"bottleneck_270_b{i}")
            layer.weight.copy_(torch.cat((a.weight, b.weight)).half())
            layer.bias.copy_(torch.cat((a.bias, b.bias)).half())
            self.groups.append(layer.to(memory_format=torch.channels_last))
        for name in ("project_540", "compress_270", "expand_270", "refine_540", "metadata_head"):
            getattr(self.model, name).half().to(memory_format=torch.channels_last)
        self.metadata = {"checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
                         "parameters": sum(p.numel() for p in self.model.parameters()),
                         "precision": "FP16 controller / FP32 analysis and synthesis" if fp32_synthesis else "FP16 throughout",
                         "implementation": "Ordinary PyTorch; static linear analysis folding, grouped branch packing, GEMM/bmm synthesis",
                         "static_basis_projection": "FP32 projection at model preparation, then cast to execution precision",
                         "sources": {str(p): sha256(p) for p in (Path(__file__), RELEASE / "aeisu/model.py",
                                                                RELEASE / "aeisu/reconstruction.py", RELEASE / "aeisu/runtime.py")}}

    def controller(self, source):
        m = self.model
        feature = F.conv2d(F.pad(source, (1, 1, 1, 1), mode="replicate"), self.analysis)
        skip = m._activate(m.project_540(F.pixel_unshuffle(feature.half(), 2)))
        value = m._activate(m.compress_270(F.pixel_unshuffle(skip, 2)))
        for layer in self.groups:
            value = m._activate(layer(value))
        value = F.pixel_shuffle(m._activate(m.expand_270(value)), 2)
        value = m._activate(m.refine_540(value + skip))
        return m.metadata_head(value)

    def synthesis(self, source, metadata):
        b, _, h, w = source.shape
        coeff = unpack_coefficients(metadata.to(self.dtype)).flatten(2).transpose(1, 2)
        weights = (coeff @ self.bank_matrix + self.base.reshape(1, 1, 100)).reshape(b, h * w, 25, 4)
        patches = F.unfold(F.pad(source, (2, 2, 2, 2), mode="replicate"), 5)
        patches = patches.reshape(b, 3, 25, h * w).permute(0, 3, 1, 2)
        rgb = torch.matmul(patches, weights).permute(0, 2, 3, 1).reshape(b, 12, h, w)
        return F.pixel_shuffle(rgb, 2)

    def __call__(self, source):
        h, w = source.shape[-2:]
        ey, ex = -h % 32, -w % 32
        py, px = 16 + ey // 2, 16 + ex // 2
        padded = F.pad(source.to(self.dtype), (px, 32 + ex - px, py, 32 + ey - py), mode="replicate")
        output = self.synthesis(padded, self.controller(padded))
        return output[:, :, 2 * py:2 * (py + h), 2 * px:2 * (px + w)].clamp(0, 1)

    def validate(self):
        """Check packed controller and bmm synthesis against the original portable graph."""
        generator = torch.Generator(device="cuda").manual_seed(20261006)
        source = torch.rand(1, 3, 64, 96, device="cuda", generator=generator).to(self.dtype)
        reference_model = load_model(CHECKPOINT, "cuda")
        ref_meta = reference_model.predict_coefficients(source.float())
        ref = reference_model(source.float())
        meta = self.controller(source)
        out = self.synthesis(source, meta)
        direct = reconstruct(source, meta, self.base, self.basis)
        analysis_diff = (F.conv2d(F.pad(source.float(), (1, 1, 1, 1), mode="replicate"),
                                analysis_kernel(reference_model)) - reference_model._rgb_to_feature8(source.float())).abs()
        record = {"analysis_fp32_max_abs": float(analysis_diff.max()),
                  "controller_vs_fp32": difference(ref_meta, meta),
                  "synthesis_vs_direct": difference(direct, out), "output_vs_fp32": difference(ref, out)}
        if record["analysis_fp32_max_abs"] > 2e-6 or record["output_vs_fp32"]["psnr_db"] < 45:
            raise RuntimeError(f"AEISU parity gate failed: {record}")
        return record


def difference(reference, actual):
    delta = actual.float() - reference.float()
    if not torch.isfinite(actual).all() or reference.shape != actual.shape:
        raise RuntimeError("Non-finite or incorrectly shaped inference output")
    mse = float(delta.square().mean())
    import math
    return {"mse": mse, "psnr_db": -10 * math.log10(mse) if mse else 200.0, "max_abs": float(delta.abs().max())}


def load(name, output):
    if name.startswith("aeisu_"):
        model = TorchAEISU(name.endswith("fp32_synthesis"))
        return model, model.metadata, model.validate()
    adapter = Adapter(name, output)
    if name in QUICK:
        fn = lambda value: adapter.model(value).clamp(0, 1)
    else:
        fn = lambda value: neural_forward(adapter.model, value, name)
    metadata = adapter.metadata | {"parameters": sum(p.numel() for p in adapter.model.parameters())}
    return fn, metadata, None
