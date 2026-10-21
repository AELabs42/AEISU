"""Build a vector kernel/input atlas, SVG originals, and exact numeric exports."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np

from export_parameters import ROOT, extract

OUT = ROOT / "output/figures/learned_parameters"
PDF = ROOT / "output/pdf/aif_learned_parameters.pdf"
CMAP = LinearSegmentedColormap.from_list("signed_weights", ["#b35806", "#ffffff", "#2166ac"])
PHASE_LABELS = ("TL (0, 0)", "TR (0, 1)", "BL (1, 0)", "BR (1, 1)")


def canvas(title: str, subtitle: str, data: dict):
    fig = plt.figure(figsize=(8.0, 10.5), facecolor="white")
    fig.text(0.07, 0.964, title, fontsize=17, weight="bold", va="top")
    fig.text(0.07, 0.926, subtitle, fontsize=10, va="top", linespacing=1.4)
    fig.text(0.07, 0.024, "Selected final Parallel32x4 checkpoint | epoch 1000 | folded effective parameters", fontsize=8)
    fig.text(0.07, 0.009, f"SHA-256: {data['checkpoint_sha256']}", fontsize=6.7)
    return fig


def matrix(fig, values, box, limit, *, numbers=False, decimals=3, grid=True):
    ax = fig.add_axes(box)
    h, w = values.shape
    norm = Normalize(-limit, limit)
    # Rectangles remain true vector cells in PDF/SVG; no interpolated bitmap.
    for y in range(h):
        for x in range(w):
            value = float(values[y, x])
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=CMAP(norm(value)),
                                   edgecolor="#d0d0d0" if grid else "none", linewidth=0.25))
            if numbers:
                label = f"{value:.{decimals}f}" if abs(value) >= 0.5 * 10 ** -decimals else "0"
                ax.text(x, y, label, ha="center", va="center", fontsize=6.4 if w == 5 else 8.0,
                        color="white" if abs(value) > 0.72 * limit else "#171717")
    ax.set(xlim=(-0.5, w - 0.5), ylim=(h - 0.5, -0.5), xticks=[], yticks=[])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        # Weight masks always need a complete frame, regardless of prior plot styles.
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("#555555")
        spine.set_zorder(3)
    return ax


def legend(fig, box, limit, label, fontsize=8):
    ax = fig.add_axes(box)
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(-limit, limit), cmap=CMAP),
                      cax=ax, orientation="horizontal", ticks=[-limit, 0, limit])
    cb.ax.tick_params(labelsize=fontsize, length=2, pad=2)
    cb.set_label(label, fontsize=fontsize, labelpad=3)
    cb.outline.set_linewidth(0.4)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")


def atlas(data):
    bank = np.asarray(data["arrays"]["basis_effective_k_phase_y_x"])
    limit = 0.18
    assert np.abs(bank).max() <= limit
    fig = canvas("Effective adaptive reconstruction bank", "16 learned components, each with four 5 x 5 output-phase kernels", data)
    fig.text(0.5, 0.883, r"$W_{p,\phi}=B_\phi+\sum_{k=0}^{15}a_{p,k}K_{k,\phi}$", ha="center", fontsize=15)
    width, height = 0.065, 0.065 * 8 / 10.5
    for half, start in enumerate((0.105, 0.575)):
        for phase, label in enumerate(PHASE_LABELS):
            fig.text(start + phase * 0.086 + width / 2, 0.839, label, ha="center", fontsize=8)
        for row in range(8):
            k = half * 8 + row
            bottom = 0.766 - row * 0.067
            fig.text(start - 0.018, bottom + height / 2, rf"$K_{{{k}}}$", ha="right", va="center", fontsize=10)
            for phase in range(4):
                matrix(fig, bank[k, phase], [start + phase * 0.086, bottom, width, height], limit)
    legend(fig, [0.285, 0.246, 0.43, 0.012], limit, "One shared signed scale for all 64 maps (no per-kernel normalization)")
    fig.text(0.075, 0.169,
             "Rows continue from K0-K7 on the left to K8-K15 on the right. Phase coordinates are\n"
             "(output row, output column) within the 2 x 2 HR block. Tap coordinates increase right/down;\n"
             "the central cell is the original LR sample. Each phase uses the same weights for R, G and B.\n\n"
             "These are normalized adaptive components, not standalone interpolation kernels. Each\n"
             "has zero DC and L1 norm 0.5; image-dependent coefficients combine them with the base.\n"
             "Color shows signed weight, not measured activation or component importance.",
             fontsize=9, va="top", linespacing=1.35)
    return fig


def base_comparison(data):
    learned = np.asarray(data["arrays"]["base_learned_phase_y_x"])
    original = np.asarray(data["arrays"]["base_bilinear_phase_y_x"])
    limit = 1.05
    assert np.abs(learned).max() <= limit
    fig = canvas("Learned base versus bilinear initialization", "Exact fixed interpolation and the final learned base, in the same 5 x 5 support", data)
    starts = [0.12 + i * 0.2 for i in range(4)]
    for x, label in zip(starts, PHASE_LABELS):
        fig.text(x + 0.075, 0.868, label, ha="center", fontsize=10)
    rows = (("(a) Original half-pixel bilinear base", original, 0.685),
            ("(b) Learned folded base", learned, 0.48),
            ("(c) Learned minus original", learned - original, 0.275))
    for label, values, bottom in rows:
        fig.text(0.075, bottom + 0.133, label, fontsize=11, weight="bold")
        for phase, x in enumerate(starts):
            matrix(fig, values[phase], [x, bottom, 0.15, 0.15 * 8 / 10.5], limit, numbers=True, decimals=2)
    legend(fig, [0.285, 0.228, 0.43, 0.012], limit, "Shared signed scale across original, learned and difference maps")
    dc = data["validation"]["base_phase_dc_sums"]
    fig.text(0.075, 0.158,
             "Bilinear DC sums: 1, 1, 1, 1. Learned DC sums (TL, TR, BL, BR):\n"
             + ", ".join(f"{v:.6f}" for v in dc) + ".\n\n"
             "Numbers are rounded for display; exact FP32 weights are exported separately. The base\n"
             "has a wider coefficient range than the adaptive bank, so this page uses its own declared\n"
             "scale. This is a parameter comparison, not a fixed-base quality ablation. The full output\n"
             "also includes the image-conditioned adaptive contribution.", fontsize=9, va="top", linespacing=1.35)
    return fig


def input_comparison(data):
    arrays = {k: np.asarray(v) for k, v in data["arrays"].items()}
    fig = canvas("Learned input analysis versus fixed inputs", "Folded color mixtures and directional stencils; original values are the training initialization", data)
    fig.text(0.075, 0.871, r"(a) Color analysis: $z=M\,[R,G,B]^T$,   $y=m^T[R,G,B]^T$", fontsize=11, weight="bold")
    for x, kind, label in ((0.15, "original", "Original"), (0.59, "learned", "Learned")):
        fig.text(x + 0.105, 0.839, label, ha="center", fontsize=10)
        matrix(fig, arrays[f"input_rgb_{kind}"], [x, 0.655, 0.21, 0.16], 1.0, numbers=True)
        for col, name in enumerate(("R", "G", "B")):
            fig.text(x + 0.035 + col * 0.07, 0.818, name, ha="center", fontsize=8)
        for row in range(3):
            fig.text(x - 0.015, 0.788 - row * 0.0533, rf"$z_{row+1}$", ha="right", va="center", fontsize=9)
        matrix(fig, arrays[f"input_luma_{kind}"][None], [x, 0.583, 0.21, 0.0533], 1.0, numbers=True)
        fig.text(x - 0.015, 0.61, "$y$", ha="right", va="center", fontsize=10)
    legend(fig, [0.29, 0.553, 0.42, 0.009], 1.0, "Shared RGB/luma coefficient scale; matrix rows are output channels", 7.5)
    fig.text(0.075, 0.474, "(b) Spatial stencils applied to the corresponding luma mixture y", fontsize=11, weight="bold")
    for phase, name in enumerate(("Horizontal", "Vertical", "Diagonal 45", "Diagonal 135")):
        x = 0.15 + phase * 0.205
        fig.text(x + 0.053, 0.441, name, ha="center", fontsize=8.5)
        for kind, y in (("original", 0.335), ("learned", 0.206)):
            matrix(fig, arrays[f"input_stencils_{kind}_direction_y_x"][phase],
                   [x, y, 0.106, 0.106 * 8 / 10.5], 4.0, numbers=True, decimals=2)
    fig.text(0.125, 0.375, "Original", ha="right", va="center", fontsize=8)
    fig.text(0.125, 0.246, "Learned", ha="right", va="center", fontsize=8)
    legend(fig, [0.29, 0.17, 0.42, 0.009], 4.0, "Shared scale for all eight directional stencils", 7.5)
    fig.text(0.075, 0.104,
             "Original: identity RGB, BT.709 luma coefficients, Sobel-family directional stencils.\n"
             "The learned y is an analysis channel, not calibrated luminance. Stencil magnitudes alone\n"
             "do not indicate input gain: each acts on its own luma mixture. Exact composed RGB\n"
             "stencils are included in the numeric export. No quality benefit is inferred from these maps.",
             fontsize=9, va="top", linespacing=1.3)
    return fig


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.unicode_minus": False})
    data = extract()
    OUT.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    (OUT / "parameters.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    np.savez(OUT / "parameters.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in data["arrays"].items()})
    builders = (("adaptive_bank", atlas), ("learned_base", base_comparison), ("learned_inputs", input_comparison))
    with PdfPages(PDF, metadata={"Title": "AEISU effective kernel bank, learned base and input analysis",
                               "Subject": "Final checkpoint parameter visualization; no quality-ablation claim"}) as pdf:
        for name, builder in builders:
            fig = builder(data)
            pdf.savefig(fig)
            fig.savefig(OUT / f"{name}.svg")
            fig.savefig(OUT / f"{name}.png", dpi=180)
            plt.close(fig)
    print(json.dumps({"pdf": str(PDF), "source_directory": str(OUT), "validation": data["validation"]}, indent=2))


if __name__ == "__main__":
    main()
