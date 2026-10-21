"""
Generates publication-quality Pareto frontier figures (PSNR vs. Latency)
for the QRISP AbandonedSchool 2x super-resolution benchmark on RTX 5090.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(r"{research_workspace}\output\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_FIG_DIR = Path(r"{research_workspace}\docs\manuscript\figures")
MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9,
    'figure.autolayout': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

SUMMARY_JSON = Path(r"{research_workspace}\output\paper_measurements\2026-09-29\qrisp_area_v2\comparison_summary.json")

def load_data():
    with open(SUMMARY_JSON, "r") as f:
        data = json.load(f)
    
    q = data["quality"]
    t = data["timing_540p_to_1080p"]

    # Structured entries: (key, display_label, category, marker, color, size)
    methods = [
        # Proposed AEISU
        ("aif_mixed", "AEISU (Mixed FP8/16, Ours)", "aif", "*", "#d62728", 220),
        ("aif_fp16", "AEISU (Uniform FP16)", "aif", "o", "#e377c2", 110),
        ("aif_fp8", "AEISU (Uniform FP8)", "aif", "s", "#ff7f0e", 110),
        
        # Real-time Neural
        ("repnet_x2_fp16", "RepNet (FP16)", "rt_neural", "D", "#2ca02c", 100),
        ("repnet_x2_fp32", "RepNet (FP32)", "rt_neural", "d", "#98df8a", 70),
        ("rt4ksr_x2_fp16", "RT4KSR (FP16)", "rt_neural", "^", "#17becf", 100),
        ("rt4ksr_x2_fp32", "RT4KSR (FP32)", "rt_neural", "v", "#9edae5", 70),

        # Offline / Heavy Neural
        ("span_x2_fp16", "SPAN (FP16)", "heavy_neural", "P", "#9467bd", 110),
        ("span_x2_fp32", "SPAN (FP32)", "heavy_neural", "p", "#c5b0d5", 70),
        ("swinir_light_x2_fp16", "SwinIR-Light (FP16)", "heavy_neural", "X", "#8c564b", 110),
        ("swinir_light_x2_fp32", "SwinIR-Light (FP32)", "heavy_neural", "x", "#c49c94", 70),

        # Spatial
        ("fsr1_easu_fp32", "FSR1 EASU", "spatial", "h", "#1f77b4", 85),
        ("fsr1_rcas_02_fp32", "FSR1 EASU+RCAS", "spatial", "H", "#396ab1", 85),
        ("nis_0_fp32", "NIS (Sharpness 0)", "spatial", "<", "#4b4b4b", 80),
        ("nis_05_fp32", "NIS (Sharpness 0.5)", "spatial", ">", "#7f7f7f", 80),
        ("sgsr1_fp32", "SGSR1", "spatial", "8", "#008080", 80),
        ("bilinear_fp32", "Bilinear", "spatial", "+", "#a0a0a0", 90),
    ]

    points = []
    for key, label, cat, marker, color, size in methods:
        if key in q and key in t:
            psnr = q[key]["all"]["rgb_psnr_db"]
            ssim = q[key]["all"]["rgb_ssim"]
            latency = t[key]["median_ms"]
            p95 = t[key]["median_trial_p95_ms"]
            points.append({
                "key": key, "label": label, "cat": cat,
                "marker": marker, "color": color, "size": size,
                "psnr": psnr, "ssim": ssim, "latency": latency, "p95": p95
            })
    return points

def plot_pareto(points):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2), dpi=300)
    fig.subplots_adjust(left=.065, right=.985, top=.91, bottom=.29, wspace=.20)

    # ------------------ Subplot 1: Full Log Scale (0.01 to 300 ms) ------------------
    for pt in points:
        ax1.scatter(pt["latency"], pt["psnr"], color=pt["color"], marker=pt["marker"],
                    s=pt["size"], edgecolors='black', linewidths=0.6, zorder=5, label=pt["label"])

    # Draw Pareto boundary
    pareto_pts = []
    best_psnr = -np.inf
    for point in sorted(points, key=lambda item: (item["latency"], -item["psnr"])):
        if point["psnr"] > best_psnr:
            pareto_pts.append(point)
            best_psnr = point["psnr"]
    px = [p["latency"] for p in pareto_pts]
    py = [p["psnr"] for p in pareto_pts]
    ax1.plot(px, py, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7, zorder=3, label="Pareto Frontier")

    # Annotate AEISU Mixed
    aif_m = next(p for p in points if p["key"] == "aif_mixed")
    ax1.annotate(f"{aif_m['label']}\n30.95 dB @ 0.133 ms",
                 xy=(aif_m["latency"], aif_m["psnr"]),
                 xytext=(aif_m["latency"] * 1.8, aif_m["psnr"] + 0.12),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.0),
                 fontweight='bold', fontsize=9.5, color='#a00')

    # Annotate RepNet FP16
    rep = next(p for p in points if p["key"] == "repnet_x2_fp16")
    ax1.annotate(f"RepNet FP16\n29.52 dB @ 0.19 ms",
                 xy=(rep["latency"], rep["psnr"]),
                 xytext=(rep["latency"] * 0.15, rep["psnr"] - 0.7),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
                 fontsize=8.5, color='#1b6e1b')

    # Annotate SPAN FP16
    span = next(p for p in points if p["key"] == "span_x2_fp16")
    ax1.annotate(f"SPAN FP16\n29.86 dB @ 5.96 ms",
                 xy=(span["latency"], span["psnr"]),
                 xytext=(span["latency"] * 1.5, span["psnr"] - 0.5),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
                 fontsize=8.5, color='#6b3c94')

    ax1.set_xscale("log")
    ax1.set_xlim(0.01, 350)
    ax1.set_ylim(27.0, 31.4)
    ax1.set_xlabel("Resident GPU latency (ms; log scale)", labelpad=8)
    ax1.set_ylabel("Reconstruction Quality: RGB PSNR (dB)", fontweight='bold')
    ax1.set_title("(a) Full latency range", fontweight='bold', pad=10)
    ax1.grid(True, which="both", ls=":", alpha=0.5)

    # ------------------ Subplot 2: Real-Time Regime (< 2.0 ms, Linear Scale) ------------------
    rt_points = [p for p in points if p["latency"] <= 2.0]
    for pt in rt_points:
        ax2.scatter(pt["latency"], pt["psnr"], color=pt["color"], marker=pt["marker"],
                    s=pt["size"], edgecolors='black', linewidths=0.6, zorder=5)

    # Draw Pareto boundary on RT plot
    rt_px = [p["latency"] for p in pareto_pts if p["latency"] <= 2.0]
    rt_py = [p["psnr"] for p in pareto_pts if p["latency"] <= 2.0]
    ax2.plot(rt_px, rt_py, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.7, zorder=3)

    # Annotate points on RT plot
    ax2.annotate(f"AEISU Mixed (Ours)\n0.133 ms | 30.95 dB",
                 xy=(aif_m["latency"], aif_m["psnr"]),
                 xytext=(.40, 31.30), va="top",
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.9),
                 fontweight='bold', fontsize=9.5, color='#a00')

    aif_16 = next(p for p in points if p["key"] == "aif_fp16")
    ax2.annotate(f"AEISU FP16 (0.28 ms)",
                 xy=(aif_16["latency"], aif_16["psnr"]),
                 xytext=(.60, 30.95),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.7),
                 fontsize=8.5, color='#444')

    aif_8 = next(p for p in points if p["key"] == "aif_fp8")
    ax2.annotate(f"AEISU FP8 (0.20 ms)",
                 xy=(aif_8["latency"], aif_8["psnr"]),
                 xytext=(aif_8["latency"] + 0.05, aif_8["psnr"] - 0.25),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.7),
                 fontsize=8.5, color='#444')

    ax2.annotate(f"RepNet FP16\n0.19 ms | 29.52 dB",
                 xy=(rep["latency"], rep["psnr"]),
                 xytext=(rep["latency"] + 0.08, rep["psnr"] - 0.40),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
                 fontsize=8.5, color='#1b6e1b')

    rt4 = next(p for p in points if p["key"] == "rt4ksr_x2_fp16")
    ax2.annotate(f"RT4KSR FP16\n1.15 ms | 29.73 dB",
                 xy=(rt4["latency"], rt4["psnr"]),
                 xytext=(rt4["latency"] - 0.38, rt4["psnr"] - 0.35),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
                 fontsize=8.5, color='#0e717d')

    fsr = next(p for p in points if p["key"] == "fsr1_easu_fp32")
    ax2.annotate(f"FSR1 EASU\n(0.03 ms)",
                 xy=(fsr["latency"], fsr["psnr"]),
                 xytext=(fsr["latency"] + 0.08, fsr["psnr"] + 0.10),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.8),
                 fontsize=8.5, color='#1f77b4')

    ax2.set_xlim(-0.02, 1.75)
    ax2.set_ylim(27.0, 31.4)
    ax2.set_xlabel("Resident GPU latency (ms; linear scale)", labelpad=8)
    ax2.set_ylabel("Reconstruction Quality: RGB PSNR (dB)", fontweight='bold')
    ax2.set_title("(b) Real-time region ($\\leq 2$ ms)", fontweight='bold', pad=10)
    ax2.grid(True, ls=":", alpha=0.5)

    # Frame budget lines
    ax2.axvline(x=0.5, color='#888', linestyle=':', alpha=0.7)
    ax2.text(0.51, 27.2, "0.5 ms budget", rotation=90, fontsize=8, color='#666')
    ax2.axvline(x=1.0, color='#888', linestyle=':', alpha=0.7)
    ax2.text(1.01, 27.2, "1.0 ms budget", rotation=90, fontsize=8, color='#666')

    # Legend on first subplot
    handles, labels_text = ax1.get_legend_handles_labels()
    # Filter unique handles
    by_label = dict(zip(labels_text, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center',
               bbox_to_anchor=(0.5, .012), ncol=4, frameon=False, fontsize=8.5,
               columnspacing=2.0, handletextpad=.7, labelspacing=.8)
    
    # Save to all target paths
    fig_paths = [
        OUTPUT_DIR / "qrisp_psnr_vs_latency.png",
        OUTPUT_DIR / "qrisp_psnr_vs_latency.pdf",
        MANUSCRIPT_FIG_DIR / "qrisp_psnr_vs_latency.pdf"
    ]
    for p in fig_paths:
        fig.savefig(p, bbox_inches='tight', dpi=300)
        print(f"Saved: {p}")
    plt.close()

if __name__ == "__main__":
    pts = load_data()
    plot_pareto(pts)
