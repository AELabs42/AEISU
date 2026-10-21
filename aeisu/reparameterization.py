"""Foldable training coordinates; standard PyTorch only."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

LEAKY_RELU_SLOPE = 0.1
REPARAM_LOW_RANK = 8
REPARAM_SECOND_LOW_RANK = 16
REPARAM_SPATIAL_LOW_RANK = 8
REPARAM_BASIS_MIXED_BANK = 64
REPARAM_BASIS_SEPARABLE_RANK = 2
REPARAM_BASE_EXTRA_BRANCHES = 4


def _reparam_directional_templates() -> torch.Tensor:
    """Six L1-normalized 3x3 analysis templates used only to parameterize training."""
    templates = torch.tensor([
        [[-1.0,  0.0,  1.0], [-2.0,  0.0,  2.0], [-1.0,  0.0,  1.0]],  # dx
        [[-1.0, -2.0, -1.0], [ 0.0,  0.0,  0.0], [ 1.0,  2.0,  1.0]],  # dy
        [[-2.0, -1.0,  0.0], [-1.0,  0.0,  1.0], [ 0.0,  1.0,  2.0]],  # d45
        [[ 0.0,  1.0,  2.0], [-1.0,  0.0,  1.0], [-2.0, -1.0,  0.0]],  # d135
        [[ 0.0,  1.0,  0.0], [ 1.0, -4.0,  1.0], [ 0.0,  1.0,  0.0]],  # lap4
        [[ 1.0,  1.0,  1.0], [ 1.0, -8.0,  1.0], [ 1.0,  1.0,  1.0]],  # lap8
    ], dtype=torch.float32)
    denom = templates.abs().sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return templates / denom

def _reparam_5x5_templates() -> torch.Tensor:
    """Eight normalized 5x5 structural coordinates for training-only filter reparameterization."""
    t = torch.zeros((8, 5, 5), dtype=torch.float32)
    # Centered 3-tap first derivatives.
    t[0, 2, 1] = -1.0
    t[0, 2, 3] = 1.0
    t[1, 1, 2] = -1.0
    t[1, 3, 2] = 1.0
    t[2, 1, 1] = -1.0
    t[2, 3, 3] = 1.0
    t[3, 1, 3] = 1.0
    t[3, 3, 1] = -1.0
    # 4-neighbor / 8-neighbor Laplacians.
    t[4, 2, 2] = -4.0
    t[4, 1, 2] = t[4, 3, 2] = t[4, 2, 1] = t[4, 2, 3] = 1.0
    t[5, 2, 2] = -8.0
    for yy in range(1, 4):
        for xx in range(1, 4):
            if yy != 2 or xx != 2:
                t[5, yy, xx] = 1.0
    # Center-surround / outer-ring high-pass coordinates.
    t[6, 2, 2] = 1.0
    for yy in range(1, 4):
        for xx in range(1, 4):
            if yy != 2 or xx != 2:
                t[6, yy, xx] = -1.0 / 8.0
    t[7, 2, 2] = 1.0
    ring = []
    for yy in range(5):
        for xx in range(5):
            if yy in (0, 4) or xx in (0, 4):
                ring.append((yy, xx))
    for yy, xx in ring:
        t[7, yy, xx] = -1.0 / 16.0
    denom = t.abs().sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return t / denom

def _reparam_outer_ring_templates() -> torch.Tensor:
    """One-hot templates for the sixteen perimeter taps of a 5x5 kernel."""
    coords = []
    for yy in range(5):
        for xx in range(5):
            if yy in (0, 4) or xx in (0, 4):
                coords.append((yy, xx))
    out = torch.zeros((len(coords), 5, 5), dtype=torch.float32)
    for i, (yy, xx) in enumerate(coords):
        out[i, yy, xx] = 1.0
    return out

class ReparamConv2d(nn.Conv2d):
    """Conv2d whose training parameterization folds exactly to one deployed Conv2d.

    The image-sized forward still performs exactly one convolution. Extra branches
    are assembled into the effective kernel before F.conv2d, so the additional
    training cost is kernel algebra rather than parallel image convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, padding: int = 0,
                 bias: bool = True, rank: int = REPARAM_LOW_RANK) -> None:
        if int(kernel_size) not in (1, 3):
            raise ValueError("ReparamConv2d currently supports 1x1 or 3x3 kernels")
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.rep_rank = max(1, min(int(rank), int(in_channels), int(out_channels)))

        # Every deployed conv gets two independent low-rank center paths.
        self.rep_low_u = nn.Parameter(torch.empty(out_channels, self.rep_rank, dtype=torch.float32))
        self.rep_low_v = nn.Parameter(torch.empty(self.rep_rank, in_channels, dtype=torch.float32))
        self.rep_rank2 = max(1, min(REPARAM_SECOND_LOW_RANK, int(in_channels), int(out_channels)))
        self.rep_low2_u = nn.Parameter(torch.empty(out_channels, self.rep_rank2, dtype=torch.float32))
        self.rep_low2_v = nn.Parameter(torch.empty(self.rep_rank2, in_channels, dtype=torch.float32))

        # Square transforms also get an explicit identity/diagonal path.
        if in_channels == out_channels:
            self.rep_identity = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
            self.register_buffer(
                "_rep_eye", torch.eye(out_channels, dtype=torch.float32).view(out_channels, in_channels, 1, 1),
                persistent=False,
            )
        else:
            self.register_parameter("rep_identity", None)
            self.register_buffer("_rep_eye", torch.empty(0), persistent=False)

        if int(kernel_size) == 3:
            # Classic 1x1-in-3x3 structural branch.
            self.rep_center = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1, dtype=torch.float32))
            # Dense horizontal/vertical coordinates embedded into the deployed 3x3 kernel.
            self.rep_row = nn.Parameter(torch.empty(out_channels, in_channels, 1, 3, dtype=torch.float32))
            self.rep_col = nn.Parameter(torch.empty(out_channels, in_channels, 3, 1, dtype=torch.float32))
            # A full-spatial low-rank channel factorization: U[out,r] @ V[r,in,3,3].
            self.rep_spatial_rank = max(1, min(REPARAM_SPATIAL_LOW_RANK, int(in_channels), int(out_channels)))
            self.rep_spatial_u = nn.Parameter(torch.empty(out_channels, self.rep_spatial_rank, dtype=torch.float32))
            self.rep_spatial_v = nn.Parameter(torch.empty(self.rep_spatial_rank, in_channels, 3, 3, dtype=torch.float32))
            # Direct edge/high-pass coordinates for every input/output channel pair.
            self.rep_directional_mix = nn.Parameter(
                torch.empty(6, out_channels, in_channels, dtype=torch.float32)
            )
            # Per-channel spatial route when channel counts match.
            if in_channels == out_channels:
                self.rep_depthwise = nn.Parameter(torch.empty(out_channels, 3, 3, dtype=torch.float32))
            else:
                self.register_parameter("rep_depthwise", None)
            self.register_buffer("_rep_templates", _reparam_directional_templates(), persistent=False)
        else:
            self.register_parameter("rep_center", None)
            self.register_parameter("rep_row", None)
            self.register_parameter("rep_col", None)
            self.register_parameter("rep_spatial_u", None)
            self.register_parameter("rep_spatial_v", None)
            self.rep_spatial_rank = 0
            self.register_parameter("rep_directional_mix", None)
            self.register_parameter("rep_depthwise", None)
            self.register_buffer("_rep_templates", torch.empty(0), persistent=False)

    def _center_pad(self, center: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == (1, 1):
            return center
        return F.pad(center, (1, 1, 1, 1))

    def reparam_addition(self) -> torch.Tensor:
        add = self._center_pad(torch.matmul(self.rep_low_u, self.rep_low_v).view(
            self.out_channels, self.in_channels, 1, 1
        ))
        add = add + self._center_pad(torch.matmul(self.rep_low2_u, self.rep_low2_v).view(
            self.out_channels, self.in_channels, 1, 1
        ))
        if self.rep_identity is not None:
            eye = self._rep_eye.to(device=add.device, dtype=add.dtype)
            add = add + self._center_pad(eye * self.rep_identity.view(-1, 1, 1, 1))
        if self.rep_center is not None:
            add = add + self._center_pad(self.rep_center)
        if self.rep_row is not None:
            add = add + F.pad(self.rep_row, (0, 0, 1, 1))
        if self.rep_col is not None:
            add = add + F.pad(self.rep_col, (1, 1, 0, 0))
        if self.rep_spatial_u is not None:
            add = add + torch.einsum("or,rihw->oihw", self.rep_spatial_u, self.rep_spatial_v)
        if self.rep_directional_mix is not None:
            templates = self._rep_templates.to(device=add.device, dtype=add.dtype)
            add = add + torch.einsum("toi,thw->oihw", self.rep_directional_mix, templates)
        if self.rep_depthwise is not None:
            eye = self._rep_eye.to(device=add.device, dtype=add.dtype)
            add = add + eye * self.rep_depthwise[:, None, :, :]
        return add

    def effective_weight(self) -> torch.Tensor:
        return self.weight + self.reparam_addition()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.effective_weight(), self.bias)

    @torch.no_grad()
    def initialize_reparameterization_preserving_effective(self) -> None:
        """Activate all alternate coordinates while preserving the current exact kernel."""
        target = self.weight.detach().clone()
        nn.init.normal_(self.rep_low_u, mean=0.0, std=0.015)
        nn.init.normal_(self.rep_low_v, mean=0.0, std=0.015)
        nn.init.normal_(self.rep_low2_u, mean=0.0, std=0.010)
        nn.init.normal_(self.rep_low2_v, mean=0.0, std=0.010)
        if self.rep_identity is not None:
            nn.init.normal_(self.rep_identity, mean=0.0, std=0.005)
        if self.rep_center is not None:
            nn.init.normal_(self.rep_center, mean=0.0, std=0.005)
        if self.rep_row is not None:
            nn.init.normal_(self.rep_row, mean=0.0, std=0.004)
        if self.rep_col is not None:
            nn.init.normal_(self.rep_col, mean=0.0, std=0.004)
        if self.rep_spatial_u is not None:
            nn.init.normal_(self.rep_spatial_u, mean=0.0, std=0.008)
            nn.init.normal_(self.rep_spatial_v, mean=0.0, std=0.008)
        if self.rep_directional_mix is not None:
            nn.init.normal_(self.rep_directional_mix, mean=0.0, std=0.005)
        if self.rep_depthwise is not None:
            nn.init.normal_(self.rep_depthwise, mean=0.0, std=0.005)
        self.weight.copy_(target - self.reparam_addition())

    @torch.no_grad()
    def fold_(self) -> None:
        self.weight.copy_(self.effective_weight())
        self.rep_low_u.zero_()
        self.rep_low_v.zero_()
        self.rep_low2_u.zero_()
        self.rep_low2_v.zero_()
        if self.rep_identity is not None:
            self.rep_identity.zero_()
        if self.rep_center is not None:
            self.rep_center.zero_()
        if self.rep_row is not None:
            self.rep_row.zero_()
        if self.rep_col is not None:
            self.rep_col.zero_()
        if self.rep_spatial_u is not None:
            self.rep_spatial_u.zero_()
            self.rep_spatial_v.zero_()
        if self.rep_directional_mix is not None:
            self.rep_directional_mix.zero_()
        if self.rep_depthwise is not None:
            self.rep_depthwise.zero_()
