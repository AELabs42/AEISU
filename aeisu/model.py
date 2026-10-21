"""AEISU training and folded inference models with legacy state-dict compatibility."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from .reparameterization import (
    LEAKY_RELU_SLOPE, REPARAM_BASE_EXTRA_BRANCHES, REPARAM_BASIS_MIXED_BANK,
    REPARAM_BASIS_SEPARABLE_RANK, ReparamConv2d,
    _reparam_5x5_templates, _reparam_outer_ring_templates,
)
from .reconstruction import project_basis, reconstruct

ARCHITECTURE_ID = "aeisu_parallel32x4_training_v1"
BT709_KR, BT709_KG, BT709_KB = 0.2126, 0.7152, 0.0722

CONVOLUTIONS = (
    ("project_540", 32, 32, 1),
    ("compress_270", 128, 64, 1),
    *((f"bottleneck_270_{branch}{i}", 32, 32, 3) for branch in ("a", "b") for i in range(1, 5)),
    ("expand_270", 64, 128, 1),
    ("refine_540", 32, 64, 3),
    ("metadata_head", 64, 64, 1),
)
ANALYSIS = {
    "analysis_rgb_basis": torch.eye(3),
    "analysis_rgb_mix": torch.tensor([BT709_KR, BT709_KG, BT709_KB]),
    "analysis_dx_rows": torch.tensor([0.125, 0.25, 0.125]),
    "analysis_dy_cols": torch.tensor([0.125, 0.25, 0.125]),
    "analysis_diag45": torch.tensor([0.25, 0.125, 0.125]),
    "analysis_diag135": torch.tensor([0.25, 0.125, 0.125]),
}

class AEISU(nn.Module):
    """Portable training model with the original foldable parameterization."""

    def __init__(
        self,
        in_channels: int = 3,
        channels_540: int = 32,
        channels_270: int = 64,
        scale: int = 2,
        basis_l1_scale: float = 0.50,
        basis_grad_stride: int = 16,
    ) -> None:
        super().__init__()
        required = {
            "in_channels": (in_channels, 3),
            "channels_540": (channels_540, 32),
            "channels_270": (channels_270, 64),
            "scale": (scale, 2),
        }
        for name, (value, expected) in required.items():
            if int(value) != expected:
                raise ValueError(
                    f"AEISU requires {name}={expected}, got {value}"
                )

        self.in_channels = 3
        self.logical_input_channels = 8
        self.packed_input_channels = 32
        self.out_channels = 3
        self.scale = 2
        self.channels_540 = 32
        self.channels_270 = 64
        self.refine_channels_540 = 64
        self.metadata_channels = 64
        self.metadata_coefficients = 16
        self.basis_l1_scale = float(basis_l1_scale)
        self.basis_grad_stride = int(basis_grad_stride)
        self.basis_grad_phase = 0
        self.architecture = ARCHITECTURE_ID

        self.project_540 = ReparamConv2d(32, 32, 1)
        self.compress_270 = ReparamConv2d(32 * 4, 64, 1)
        # Parallel 270p bottleneck: the learned 128->64 projection preserves a
        # 64-channel representation, then splits it into two independent 32ch
        # streams. Each stream gets four fully Aggressive64-reparameterized
        # 32->32 spatial layers before concatenation back to 64 channels.
        for branch in ("a", "b"):
            for i in range(1, 5):
                setattr(self, f"bottleneck_270_{branch}{i}",
                        ReparamConv2d(32, 32, 3, padding=1))
        self.expand_270 = ReparamConv2d(64, 32 * 4, 1)
        self.refine_540 = ReparamConv2d(32, 64, 3, padding=1)
        self.metadata_head = ReparamConv2d(64, 64, 1)

        # Twenty-four learned constants: a 3x3 learned color basis, one learned
        # RGB->analysis mix, and four independently trainable directional stencils.
        # The color basis starts as identity, so this architecture is exactly the
        # four-direction baseline at initialization.
        self.analysis_rgb_basis = nn.Parameter(torch.empty(3, 3, dtype=torch.float32))
        self.analysis_rgb_mix = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.analysis_dx_rows = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.analysis_dy_cols = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.analysis_diag45 = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.analysis_diag135 = nn.Parameter(torch.empty(3, dtype=torch.float32))

        # Training-only alternate coordinates for the static input analysis.
        # They fold back into the exact same 3x3 RGB matrix / 3-tap stencils.
        self.analysis_rgb_basis_diag = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.analysis_rgb_basis_u = nn.Parameter(torch.empty(3, 2, dtype=torch.float32))
        self.analysis_rgb_basis_v = nn.Parameter(torch.empty(2, 3, dtype=torch.float32))
        self.analysis_rgb_mix_bt709_gain = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.analysis_rgb_mix_uniform_gain = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.analysis_dx_seed_gain = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.analysis_dx_curve_gain = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.analysis_dy_seed_gain = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.analysis_dy_curve_gain = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.analysis_diag45_seed_gain = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.analysis_diag45_curve_gain = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.analysis_diag135_seed_gain = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.analysis_diag135_curve_gain = nn.Parameter(torch.empty((), dtype=torch.float32))

        # Trainable runtime-layout [tap, phase] 5x5 base. It is initialized to
        # exact 2x bilinear coefficients (1/16, 3/16, 9/16 nonzero taps) so the
        # model starts from the same valid interpolator, but the base remains
        # fully trainable thereafter.
        self.base_kernel = nn.Parameter(self._make_exact_bilinear_base())
        self.base_rep_kernels = nn.Parameter(torch.empty(
            REPARAM_BASE_EXTRA_BRANCHES, 25, 4, dtype=torch.float32
        ))
        self.base_rep_slopes = nn.Parameter(torch.empty(REPARAM_BASE_EXTRA_BRANCHES, dtype=torch.float32))
        # Aggressive structured base coordinates, all embedded into the same 5x5 support.
        self.base_rep_3x3 = nn.Parameter(torch.empty(4, 3, 3, dtype=torch.float32))
        self.base_rep_1x5 = nn.Parameter(torch.empty(4, 1, 5, dtype=torch.float32))
        self.base_rep_5x1 = nn.Parameter(torch.empty(4, 5, 1, dtype=torch.float32))
        self.base_rep_1x3 = nn.Parameter(torch.empty(4, 1, 3, dtype=torch.float32))
        self.base_rep_3x1 = nn.Parameter(torch.empty(4, 3, 1, dtype=torch.float32))
        self.base_rep_1x1 = nn.Parameter(torch.empty(4, 1, 1, dtype=torch.float32))
        self.base_rep_diag = nn.Parameter(torch.empty(4, 5, dtype=torch.float32))
        self.base_rep_anti_diag = nn.Parameter(torch.empty(4, 5, dtype=torch.float32))
        self.base_rep_outer = nn.Parameter(torch.empty(4, 16, dtype=torch.float32))
        self.base_rep_sep_y = nn.Parameter(torch.empty(4, REPARAM_BASIS_SEPARABLE_RANK, 5, dtype=torch.float32))
        self.base_rep_sep_x = nn.Parameter(torch.empty(4, REPARAM_BASIS_SEPARABLE_RANK, 5, dtype=torch.float32))
        self.base_rep_template_gain = nn.Parameter(torch.empty(4, 8, dtype=torch.float32))

        # The deployed adaptive bank remains exactly sixteen [4 phase x 5x5] bases.
        self.basis_raw = nn.Parameter(torch.empty(16, 4, 25, dtype=torch.float32))

        # Training-only 64-way adaptive bank. Each temporary basis has many
        # structured coordinate systems that all collapse to one static 5x5 kernel.
        m = REPARAM_BASIS_MIXED_BANK
        self.basis_mix_main = nn.Parameter(torch.empty(m, 4, 5, 5, dtype=torch.float32))
        self.basis_mix_full = nn.Parameter(torch.empty(m, 4, 5, 5, dtype=torch.float32))
        self.basis_mix_3x3 = nn.Parameter(torch.empty(m, 4, 3, 3, dtype=torch.float32))
        self.basis_mix_1x5 = nn.Parameter(torch.empty(m, 4, 1, 5, dtype=torch.float32))
        self.basis_mix_5x1 = nn.Parameter(torch.empty(m, 4, 5, 1, dtype=torch.float32))
        self.basis_mix_1x3 = nn.Parameter(torch.empty(m, 4, 1, 3, dtype=torch.float32))
        self.basis_mix_3x1 = nn.Parameter(torch.empty(m, 4, 3, 1, dtype=torch.float32))
        self.basis_mix_1x1 = nn.Parameter(torch.empty(m, 4, 1, 1, dtype=torch.float32))
        self.basis_mix_diag = nn.Parameter(torch.empty(m, 4, 5, dtype=torch.float32))
        self.basis_mix_anti_diag = nn.Parameter(torch.empty(m, 4, 5, dtype=torch.float32))
        self.basis_mix_outer = nn.Parameter(torch.empty(m, 4, 16, dtype=torch.float32))
        self.basis_mix_sep_y = nn.Parameter(torch.empty(m, 4, REPARAM_BASIS_SEPARABLE_RANK, 5, dtype=torch.float32))
        self.basis_mix_sep_x = nn.Parameter(torch.empty(m, 4, REPARAM_BASIS_SEPARABLE_RANK, 5, dtype=torch.float32))
        self.basis_mix_template_gain = nn.Parameter(torch.empty(m, 4, 8, dtype=torch.float32))
        # beta = A z during training conceptually; A is folded into the 64 basis bank
        # before the existing zero-DC/L1 projection, leaving sixteen deployed bases.
        self.basis_mix_matrix = nn.Parameter(torch.empty(m, 16, dtype=torch.float32))

        self.register_buffer("_rep_5x5_templates", _reparam_5x5_templates(), persistent=False)
        self.register_buffer("_rep_outer_templates", _reparam_outer_ring_templates(), persistent=False)

        self.reset_parameters()

    @staticmethod
    def _make_exact_bilinear_base() -> torch.Tensor:
        # Phase order: TL, TR, BL, BR. Tap coordinates are centered at [2,2].
        # 2x align_corners=False sample offsets are +/- 1/4 source pixel, giving
        # separable weights 1/4 and 3/4 -> exact products 1/16, 3/16, 9/16.
        base = torch.zeros((25, 4), dtype=torch.float32)
        def put(y: int, x: int, phase: int, value: float) -> None:
            base[y * 5 + x, phase] = value
        q1, q3, q9 = 1.0 / 16.0, 3.0 / 16.0, 9.0 / 16.0
        # TL: source coordinate (-1/4, -1/4) relative to the source pixel.
        put(1, 1, 0, q1)
        put(1, 2, 0, q3)
        put(2, 1, 0, q3)
        put(2, 2, 0, q9)
        # TR: (+1/4, -1/4).
        put(1, 2, 1, q3)
        put(1, 3, 1, q1)
        put(2, 2, 1, q9)
        put(2, 3, 1, q3)
        # BL: (-1/4, +1/4).
        put(2, 1, 2, q3)
        put(2, 2, 2, q9)
        put(3, 1, 2, q1)
        put(3, 2, 2, q3)
        # BR: (+1/4, +1/4).
        put(2, 2, 3, q9)
        put(2, 3, 3, q3)
        put(3, 2, 3, q3)
        put(3, 3, 3, q1)
        return base

    @staticmethod
    def _activate(x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=LEAKY_RELU_SLOPE, inplace=False)

    def _effective_analysis_rgb_basis(self) -> torch.Tensor:
        diag = torch.diag(self.analysis_rgb_basis_diag)
        low_rank = torch.matmul(self.analysis_rgb_basis_u, self.analysis_rgb_basis_v)
        return self.analysis_rgb_basis + diag + low_rank

    def _effective_analysis_rgb_mix(self) -> torch.Tensor:
        bt709 = self.analysis_rgb_mix.new_tensor((BT709_KR, BT709_KG, BT709_KB))
        uniform = self.analysis_rgb_mix.new_full((3,), 1.0 / 3.0)
        return (self.analysis_rgb_mix +
                self.analysis_rgb_mix_bt709_gain * bt709 +
                self.analysis_rgb_mix_uniform_gain * uniform)

    @staticmethod
    def _effective_analysis_stencil(main: torch.Tensor, seed_gain: torch.Tensor,
                                    curve_gain: torch.Tensor, seed_values) -> torch.Tensor:
        seed = main.new_tensor(seed_values)
        curve = main.new_tensor((0.5, -1.0, 0.5))
        return main + seed_gain * seed + curve_gain * curve

    def _structured_base_addition_5x5(self) -> torch.Tensor:
        # Return [phase,5,5]. Smaller branches are centered in the 5x5 support.
        out = F.pad(self.base_rep_3x3, (1, 1, 1, 1))
        out = out + F.pad(self.base_rep_1x5, (0, 0, 2, 2))
        out = out + F.pad(self.base_rep_5x1, (2, 2, 0, 0))
        out = out + F.pad(self.base_rep_1x3, (1, 1, 2, 2))
        out = out + F.pad(self.base_rep_3x1, (2, 2, 1, 1))
        out = out + F.pad(self.base_rep_1x1, (2, 2, 2, 2))
        out = out + torch.diag_embed(self.base_rep_diag)
        out = out + torch.flip(torch.diag_embed(self.base_rep_anti_diag), dims=(-1,))
        outer = self._rep_outer_templates.to(device=out.device, dtype=out.dtype)
        out = out + torch.einsum("pt,thw->phw", self.base_rep_outer, outer)
        out = out + torch.einsum("pry,prx->pyx", self.base_rep_sep_y, self.base_rep_sep_x)
        templates = self._rep_5x5_templates.to(device=out.device, dtype=out.dtype)
        out = out + torch.einsum("pt,thw->phw", self.base_rep_template_gain, templates)
        return out

    def _combined_base_kernel(self) -> torch.Tensor:
        dense = torch.einsum("j,jtp->tp", self.base_rep_slopes, self.base_rep_kernels)
        structured = self._structured_base_addition_5x5().reshape(4, 25).transpose(0, 1)
        return self.base_kernel + dense + structured

    def _basis_training_bank_5x5(self) -> torch.Tensor:
        # [64,4,5,5]. Every branch below is static with respect to image position,
        # so the complete temporary bank can be folded exactly before reconstruction.
        out = self.basis_mix_main + self.basis_mix_full
        out = out + F.pad(self.basis_mix_3x3, (1, 1, 1, 1))
        out = out + F.pad(self.basis_mix_1x5, (0, 0, 2, 2))
        out = out + F.pad(self.basis_mix_5x1, (2, 2, 0, 0))
        out = out + F.pad(self.basis_mix_1x3, (1, 1, 2, 2))
        out = out + F.pad(self.basis_mix_3x1, (2, 2, 1, 1))
        out = out + F.pad(self.basis_mix_1x1, (2, 2, 2, 2))
        out = out + torch.diag_embed(self.basis_mix_diag)
        out = out + torch.flip(torch.diag_embed(self.basis_mix_anti_diag), dims=(-1,))
        outer = self._rep_outer_templates.to(device=out.device, dtype=out.dtype)
        out = out + torch.einsum("mpt,thw->mphw", self.basis_mix_outer, outer)
        out = out + torch.einsum("mpry,mprx->mpyx", self.basis_mix_sep_y, self.basis_mix_sep_x)
        templates = self._rep_5x5_templates.to(device=out.device, dtype=out.dtype)
        out = out + torch.einsum("mpt,thw->mphw", self.basis_mix_template_gain, templates)
        return out

    def _combined_basis_raw(self) -> torch.Tensor:
        # Conceptually beta=A z with A[64,16], followed by 64 temporary bases.
        # Since no nonlinearity follows A, fold A into those bases first:
        # D_k = basis_raw_k + sum_m A[m,k] * B_m.
        bank = self._basis_training_bank_5x5()
        mixed = torch.einsum("mk,mphw->kphw", self.basis_mix_matrix, bank)
        return self.basis_raw + mixed.reshape(16, 4, 25)

    def _rgb_to_feature8(self, source_rgb: torch.Tensor) -> torch.Tensor:
        r = source_rgb[:, 0:1]
        g = source_rgb[:, 1:2]
        b = source_rgb[:, 2:3]

        # Identity-initialized learned RGB basis. This only changes the first
        # three controller inputs; Y and all directional maps remain derived from
        # the original source RGB so this is a clean RGB-only ablation.
        m = self._effective_analysis_rgb_basis()
        r_learned = r * m[0, 0] + g * m[0, 1] + b * m[0, 2]
        g_learned = r * m[1, 0] + g * m[1, 1] + b * m[1, 2]
        b_learned = r * m[2, 0] + g * m[2, 1] + b * m[2, 2]
        learned_rgb = torch.cat((r_learned, g_learned, b_learned), dim=1)

        # Same 3-multiply/2-add RGB mix as BT.709, but its three constants learn.
        rgb_mix = self._effective_analysis_rgb_mix()
        y = (r * rgb_mix[0] +
             g * rgb_mix[1] +
             b * rgb_mix[2])
        ypad = F.pad(y, (1, 1, 1, 1), mode="replicate")
        ul = ypad[:, :, 0:-2, 0:-2]
        uc = ypad[:, :, 0:-2, 1:-1]
        ur = ypad[:, :, 0:-2, 2:]
        cl = ypad[:, :, 1:-1, 0:-2]
        cr = ypad[:, :, 1:-1, 2:]
        dl = ypad[:, :, 2:, 0:-2]
        dc = ypad[:, :, 2:, 1:-1]
        dr = ypad[:, :, 2:, 2:]

        # Four independently learned directional Sobel-family responses.  Shared
        # center-row/column differences are written once so a fused compiler/kernel
        # can reuse them rather than recomputing the same subtraction.
        hmid = cr - cl
        vmid = dc - uc
        dx_rows = self._effective_analysis_stencil(
            self.analysis_dx_rows, self.analysis_dx_seed_gain, self.analysis_dx_curve_gain,
            (0.125, 0.25, 0.125),
        )
        dy_cols = self._effective_analysis_stencil(
            self.analysis_dy_cols, self.analysis_dy_seed_gain, self.analysis_dy_curve_gain,
            (0.125, 0.25, 0.125),
        )
        dx = ((ur - ul) * dx_rows[0] +
              hmid * dx_rows[1] +
              (dr - dl) * dx_rows[2])
        dy = ((dl - ul) * dy_cols[0] +
              vmid * dy_cols[1] +
              (dr - ur) * dy_cols[2])

        # +45-degree Sobel family:
        # [-a -b  0; -c 0 +c; 0 +b +a]
        diag45_w = self._effective_analysis_stencil(
            self.analysis_diag45, self.analysis_diag45_seed_gain, self.analysis_diag45_curve_gain,
            (0.25, 0.125, 0.125),
        )
        diag135_w = self._effective_analysis_stencil(
            self.analysis_diag135, self.analysis_diag135_seed_gain, self.analysis_diag135_curve_gain,
            (0.25, 0.125, 0.125),
        )
        diag45 = ((dr - ul) * diag45_w[0] +
                  vmid * diag45_w[1] +
                  hmid * diag45_w[2])

        # -45-degree Sobel family:
        # [0 +b +a; -c 0 +c; -a -b 0]
        diag135 = ((ur - dl) * diag135_w[0] -
                   vmid * diag135_w[1] +
                   hmid * diag135_w[2])
        return torch.cat((learned_rgb, y, dx, dy, diag45, diag135), dim=1)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=LEAKY_RELU_SLOPE,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Start the 270p decoder as a no-op residual and the 540p refine as identity.
        nn.init.zeros_(self.expand_270.weight)
        if self.expand_270.bias is not None:
            nn.init.zeros_(self.expand_270.bias)
        nn.init.zeros_(self.refine_540.weight)
        if self.refine_540.bias is not None:
            nn.init.zeros_(self.refine_540.bias)
        with torch.no_grad():
            center = self.refine_540.weight.shape[-1] // 2
            for channel in range(32):
                self.refine_540.weight[channel, channel, center, center] = 1.0

        # Neutral coefficient output is a stabilization initialization only; it
        # does not inherit any coefficient policy. The first gradient step is free
        # to move the controller, and the adaptive bank receives gradients as soon
        # as coefficients become nonzero.
        nn.init.zeros_(self.metadata_head.weight)
        if self.metadata_head.bias is not None:
            nn.init.zeros_(self.metadata_head.bias)

        # Initialize the learned RGB basis to exact identity, then BT.709 plus
        # four conventional Sobel-family directions. All twenty-four constants
        # are then free to learn independently.
        with torch.no_grad():
            self.analysis_rgb_basis.copy_(torch.eye(3, dtype=torch.float32))
            self.analysis_rgb_mix.copy_(torch.tensor([BT709_KR, BT709_KG, BT709_KB]))
            self.analysis_dx_rows.copy_(torch.tensor([0.125, 0.25, 0.125]))
            self.analysis_dy_cols.copy_(torch.tensor([0.125, 0.25, 0.125]))
            self.analysis_diag45.copy_(torch.tensor([0.25, 0.125, 0.125]))
            self.analysis_diag135.copy_(torch.tensor([0.25, 0.125, 0.125]))

            # No hand-designed filter seeds and no inherited bank: pure random
            # basis_raw. Its absolute initial scale is intentionally irrelevant
            # because effective_basis() imposes zero DC + fixed L1 every pass.
            nn.init.normal_(self.basis_raw, mean=0.0, std=0.05)

            # Reparameterize every controller convolution only after its original
            # initialization is complete, then subtract the alternate branches
            # from the main tensor so the exact starting function is unchanged.
            for module in self.modules():
                if isinstance(module, ReparamConv2d):
                    module.initialize_reparameterization_preserving_effective()

            # Input-analysis alternate coordinates: initialize them active, then
            # compensate the main parameters so effective RGB/Y/directional
            # operators remain exactly at the signature initialization.
            nn.init.normal_(self.analysis_rgb_basis_diag, mean=0.0, std=0.01)
            nn.init.normal_(self.analysis_rgb_basis_u, mean=0.0, std=0.02)
            nn.init.normal_(self.analysis_rgb_basis_v, mean=0.0, std=0.02)
            self.analysis_rgb_basis.sub_(torch.diag(self.analysis_rgb_basis_diag) +
                                         self.analysis_rgb_basis_u @ self.analysis_rgb_basis_v)

            self.analysis_rgb_mix_bt709_gain.normal_(0.0, 0.02)
            self.analysis_rgb_mix_uniform_gain.normal_(0.0, 0.02)
            bt709 = self.analysis_rgb_mix.new_tensor((BT709_KR, BT709_KG, BT709_KB))
            uniform = self.analysis_rgb_mix.new_full((3,), 1.0 / 3.0)
            self.analysis_rgb_mix.sub_(self.analysis_rgb_mix_bt709_gain * bt709 +
                                       self.analysis_rgb_mix_uniform_gain * uniform)

            stencil_specs = (
                (self.analysis_dx_rows, self.analysis_dx_seed_gain, self.analysis_dx_curve_gain, (0.125, 0.25, 0.125)),
                (self.analysis_dy_cols, self.analysis_dy_seed_gain, self.analysis_dy_curve_gain, (0.125, 0.25, 0.125)),
                (self.analysis_diag45, self.analysis_diag45_seed_gain, self.analysis_diag45_curve_gain, (0.25, 0.125, 0.125)),
                (self.analysis_diag135, self.analysis_diag135_seed_gain, self.analysis_diag135_curve_gain, (0.25, 0.125, 0.125)),
            )
            for main, seed_gain, curve_gain, seed_values in stencil_specs:
                seed_gain.normal_(0.0, 0.02)
                curve_gain.normal_(0.0, 0.02)
                seed = main.new_tensor(seed_values)
                curve = main.new_tensor((0.5, -1.0, 0.5))
                main.sub_(seed_gain * seed + curve_gain * curve)

            # Aggressive base reparameterization. Initialize every coordinate
            # active, then subtract their exact combined contribution from the
            # production base so iteration zero remains the same bilinear base.
            base_target = self.base_kernel.detach().clone()
            nn.init.normal_(self.base_rep_kernels, mean=0.0, std=0.025)
            nn.init.normal_(self.base_rep_slopes, mean=0.0, std=0.02)
            nn.init.normal_(self.base_rep_3x3, mean=0.0, std=0.008)
            nn.init.normal_(self.base_rep_1x5, mean=0.0, std=0.006)
            nn.init.normal_(self.base_rep_5x1, mean=0.0, std=0.006)
            nn.init.normal_(self.base_rep_1x3, mean=0.0, std=0.006)
            nn.init.normal_(self.base_rep_3x1, mean=0.0, std=0.006)
            nn.init.normal_(self.base_rep_1x1, mean=0.0, std=0.004)
            nn.init.normal_(self.base_rep_diag, mean=0.0, std=0.006)
            nn.init.normal_(self.base_rep_anti_diag, mean=0.0, std=0.006)
            nn.init.normal_(self.base_rep_outer, mean=0.0, std=0.004)
            nn.init.normal_(self.base_rep_sep_y, mean=0.0, std=0.012)
            nn.init.normal_(self.base_rep_sep_x, mean=0.0, std=0.012)
            nn.init.normal_(self.base_rep_template_gain, mean=0.0, std=0.006)
            base_other = self._combined_base_kernel() - self.base_kernel
            self.base_kernel.copy_(base_target - base_other)

            # 64 temporary adaptive bases, each with a rich branch pack, plus a
            # learned 64x16 linear coefficient mixer. The first 16 mixer rows start
            # near identity and all 64 rows receive small cross-coupling. The exact
            # initial folded contribution is compensated out of basis_raw.
            basis_target = self.basis_raw.detach().clone()
            nn.init.normal_(self.basis_mix_main, mean=0.0, std=0.025)
            nn.init.normal_(self.basis_mix_full, mean=0.0, std=0.015)
            nn.init.normal_(self.basis_mix_3x3, mean=0.0, std=0.010)
            nn.init.normal_(self.basis_mix_1x5, mean=0.0, std=0.008)
            nn.init.normal_(self.basis_mix_5x1, mean=0.0, std=0.008)
            nn.init.normal_(self.basis_mix_1x3, mean=0.0, std=0.008)
            nn.init.normal_(self.basis_mix_3x1, mean=0.0, std=0.008)
            nn.init.normal_(self.basis_mix_1x1, mean=0.0, std=0.006)
            nn.init.normal_(self.basis_mix_diag, mean=0.0, std=0.008)
            nn.init.normal_(self.basis_mix_anti_diag, mean=0.0, std=0.008)
            nn.init.normal_(self.basis_mix_outer, mean=0.0, std=0.005)
            nn.init.normal_(self.basis_mix_sep_y, mean=0.0, std=0.012)
            nn.init.normal_(self.basis_mix_sep_x, mean=0.0, std=0.012)
            nn.init.normal_(self.basis_mix_template_gain, mean=0.0, std=0.006)
            self.basis_mix_matrix.zero_()
            self.basis_mix_matrix[:16, :].copy_(torch.eye(16, dtype=self.basis_mix_matrix.dtype, device=self.basis_mix_matrix.device))
            self.basis_mix_matrix.add_(torch.randn_like(self.basis_mix_matrix) * 0.005)
            basis_mixed = self._combined_basis_raw() - self.basis_raw
            self.basis_raw.copy_(basis_target - basis_mixed)

    def _controller(self, source_rgb: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if source_rgb.ndim != 4 or int(source_rgb.shape[1]) != 3:
            raise ValueError(f"Expected source RGB [B,3,H,W], got {tuple(source_rgb.shape)}")
        if int(source_rgb.shape[-2]) % 4 or int(source_rgb.shape[-1]) % 4:
            raise ValueError(
                f"Source H/W must be divisible by 4, got {tuple(source_rgb.shape[-2:])}"
            )
        feat8 = self._rgb_to_feature8(source_rgb)
        enc540 = self._activate(self.project_540(F.pixel_unshuffle(feat8, 2)))
        enc270 = self._activate(self.compress_270(F.pixel_unshuffle(enc540, 2)))
        branch_a, branch_b = torch.split(enc270, 32, dim=1)
        for i in range(1, 5):
            branch_a = self._activate(getattr(self, f"bottleneck_270_a{i}")(branch_a))
            branch_b = self._activate(getattr(self, f"bottleneck_270_b{i}")(branch_b))
        bottleneck270 = torch.cat((branch_a, branch_b), dim=1)
        up540 = F.pixel_shuffle(self._activate(self.expand_270(bottleneck270)), 2)
        dec540 = self._activate(self.refine_540(enc540 + up540))
        meta = self.metadata_head(dec540)
        meta = meta.contiguous()
        return meta, (enc540, enc270, bottleneck270, dec540)

    def predict_coefficients(self, source_rgb: torch.Tensor) -> torch.Tensor:
        meta, _ = self._controller(source_rgb)
        return meta

    def effective_base(self) -> torch.Tensor:
        # Training branches are summed before reconstruction. Export folds them
        # permanently into one [25,4] production tensor.
        return self._combined_base_kernel()

    def effective_basis(self) -> torch.Tensor:
        # Sum the five-branch family for each deployed basis first, then preserve
        # the signature zero-DC + fixed-L1 projection exactly.
        raw = self._combined_basis_raw().contiguous()
        return project_basis(raw, self.basis_l1_scale)

    @torch.no_grad()
    def folded_production_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the original production architecture state_dict with all training branches folded."""
        state: Dict[str, torch.Tensor] = {}
        for name in (
            "project_540", "compress_270",
            "bottleneck_270_a1", "bottleneck_270_a2", "bottleneck_270_a3", "bottleneck_270_a4",
            "bottleneck_270_b1", "bottleneck_270_b2", "bottleneck_270_b3", "bottleneck_270_b4",
            "expand_270", "refine_540", "metadata_head",
        ):
            module = getattr(self, name)
            state[f"{name}.weight"] = module.effective_weight().detach().clone()
            if module.bias is not None:
                state[f"{name}.bias"] = module.bias.detach().clone()

        state["analysis_rgb_basis"] = self._effective_analysis_rgb_basis().detach().clone()
        state["analysis_rgb_mix"] = self._effective_analysis_rgb_mix().detach().clone()
        state["analysis_dx_rows"] = self._effective_analysis_stencil(
            self.analysis_dx_rows, self.analysis_dx_seed_gain, self.analysis_dx_curve_gain,
            (0.125, 0.25, 0.125),
        ).detach().clone()
        state["analysis_dy_cols"] = self._effective_analysis_stencil(
            self.analysis_dy_cols, self.analysis_dy_seed_gain, self.analysis_dy_curve_gain,
            (0.125, 0.25, 0.125),
        ).detach().clone()
        state["analysis_diag45"] = self._effective_analysis_stencil(
            self.analysis_diag45, self.analysis_diag45_seed_gain, self.analysis_diag45_curve_gain,
            (0.25, 0.125, 0.125),
        ).detach().clone()
        state["analysis_diag135"] = self._effective_analysis_stencil(
            self.analysis_diag135, self.analysis_diag135_seed_gain, self.analysis_diag135_curve_gain,
            (0.25, 0.125, 0.125),
        ).detach().clone()
        state["base_kernel"] = self._combined_base_kernel().detach().clone()
        state["basis_raw"] = self._combined_basis_raw().detach().clone()
        return state

    def forward(self, source_rgb: torch.Tensor, trim: int = 0, gradient_stride: int = 1, phase: int = 0) -> torch.Tensor:
        """Reconstruct unclipped 2x RGB; training may sample only base/bank gradients."""
        return reconstruct(
            source_rgb, self.predict_coefficients(source_rgb), self.effective_base(), self.effective_basis(),
            trim=trim, gradient_stride=gradient_stride, phase=phase,
        )


class FoldedAEISU(nn.Module):
    """Only deployed parameters; accepts the original folded checkpoint keys."""

    _activate = staticmethod(AEISU._activate)
    _controller = AEISU._controller
    _rgb_to_feature8 = AEISU._rgb_to_feature8
    predict_coefficients = AEISU.predict_coefficients
    forward = AEISU.forward

    def __init__(self, basis_l1_scale: float = 0.5) -> None:
        super().__init__()
        self.basis_l1_scale = basis_l1_scale
        for name, inputs, outputs, kernel in CONVOLUTIONS:
            setattr(self, name, nn.Conv2d(inputs, outputs, kernel, padding=kernel // 2))
        for name, value in ANALYSIS.items():
            setattr(self, name, nn.Parameter(value.clone()))
        self.base_kernel = nn.Parameter(AEISU._make_exact_bilinear_base())
        self.basis_raw = nn.Parameter(torch.zeros(16, 4, 25))
        for direction in ("dx", "dy", "diag45", "diag135"):
            for suffix in ("seed_gain", "curve_gain"):
                self.register_buffer(f"analysis_{direction}_{suffix}", torch.tensor(0.0), persistent=False)
        nn.init.zeros_(self.metadata_head.weight)
        nn.init.zeros_(self.metadata_head.bias)

    def _effective_analysis_rgb_basis(self) -> torch.Tensor:
        return self.analysis_rgb_basis

    def _effective_analysis_rgb_mix(self) -> torch.Tensor:
        return self.analysis_rgb_mix

    @staticmethod
    def _effective_analysis_stencil(main, seed_gain, curve_gain, seed_values):
        return main

    def effective_base(self) -> torch.Tensor:
        return self.base_kernel

    def effective_basis(self) -> torch.Tensor:
        return project_basis(self.basis_raw, self.basis_l1_scale)

    @torch.no_grad()
    def folded_production_state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.state_dict().items()}


def fold_model(model: AEISU) -> FoldedAEISU:
    folded = FoldedAEISU(model.basis_l1_scale).to(next(model.parameters()).device)
    folded.load_state_dict(model.folded_production_state_dict(), strict=True)
    return folded.eval()
