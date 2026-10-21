"""Polyphase RGB synthesis and selective base/bank gradients in PyTorch."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def project_basis(raw: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
    """Project [component, phase, tap] to zero-DC/L1-bounded [tap, component, phase]."""
    centered = raw - raw.mean(dim=-1, keepdim=True)
    normalized = centered * (scale / centered.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6))
    return normalized.permute(2, 0, 1).contiguous()


def coefficient_clamp(value: torch.Tensor) -> torch.Tensor:
    magnitude = value.abs()
    return torch.where(magnitude <= 2, value, value.sign() * (2 + 0.05 * (magnitude - 2)).clamp(max=2.5))


def unpack_coefficients(metadata: torch.Tensor) -> torch.Tensor:
    """Head channels are LR-parity-major, not HR output-phase-major."""
    b, channels, h, w = metadata.shape
    if channels != 64:
        raise ValueError("Expected 64 packed coefficient channels")
    packed = metadata.reshape(b, 4, 16, h, w).permute(0, 2, 1, 3, 4).reshape(b, 64, h, w)
    return coefficient_clamp(F.pixel_shuffle(packed, 2))


def reconstruct(source, metadata, base, basis, *, trim=0, gradient_stride=1, phase=0):
    """Return unclipped 2x RGB, with optional sparse global-filter gradients.

    ``trim`` removes HR pixels from every side. Forward values and coefficient
    gradients remain dense; only base/bank gradients use stride^2-scaled samples.
    Offsets are relative to the cropped LR core, matching the training kernel.
    """
    if source.ndim != 4 or source.shape[1] != 3:
        raise ValueError("Expected NCHW RGB source")
    if trim < 0 or trim % 2 or (trim and trim < 4):
        raise ValueError("trim must be zero or an even HR border of at least 4 pixels")
    if gradient_stride < 0 or phase < 0:
        raise ValueError("gradient_stride and phase must be nonnegative")
    dtype = torch.promote_types(source.dtype, base.dtype)
    with torch.autocast(device_type=source.device.type, enabled=False):
        source, metadata = source.to(dtype), metadata.to(dtype)
        base, basis = base.to(dtype), basis.to(dtype)
        b, _, h, w = source.shape
        offset = trim // 2
        ch, cw = h - 2 * offset, w - 2 * offset
        if ch <= 0 or cw <= 0 or metadata.shape != (b, 64, h // 2, w // 2):
            raise ValueError("Source, metadata or trim geometry is invalid")
        coeff = unpack_coefficients(metadata)
        if offset:
            local = source[:, :, offset - 2:h - offset + 2, offset - 2:w - offset + 2]
            coeff = coeff[:, :, offset:h - offset, offset:w - offset]
        else:
            local = F.pad(source, (2, 2, 2, 2), mode="replicate")
        patches = F.unfold(local, 5).reshape(b, 3, 25, ch * cw)
        coeff = coeff.reshape(b, 16, ch * cw)
        mask = None
        if gradient_stride > 1:
            phase %= gradient_stride ** 2
            yy = torch.arange(ch, device=source.device)[:, None]
            xx = torch.arange(cw, device=source.device)[None, :]
            mask = ((yy % gradient_stride == phase // gradient_stride) &
                    (xx % gradient_stride == phase % gradient_stride)).to(dtype).reshape(1, 1, -1)
            mask = mask * gradient_stride ** 2
        phases = []
        for output_phase in range(4):
            bp = base[:, output_phase].reshape(1, 25, 1)
            kp = basis[:, :, output_phase]
            if gradient_stride == 1:
                weights = bp + torch.einsum("bkn,tk->btn", coeff, kp)
            else:
                # Detached constants retain dense gradients to coefficients.
                weights = bp.detach() + torch.einsum("bkn,tk->btn", coeff, kp.detach())
                if gradient_stride:
                    # Zero-valued terms route only the global-constant derivatives.
                    weights = weights + (bp - bp.detach()) * mask
                    delta = torch.einsum("bkn,tk->btn", coeff.detach(), kp - kp.detach())
                    weights = weights + delta * mask
            rgb = (patches * weights[:, None]).sum(dim=2).reshape(b, 3, ch, cw)
            phases.append(rgb)
        return F.pixel_shuffle(torch.stack(phases, dim=2).reshape(b, 12, ch, cw), 2)
