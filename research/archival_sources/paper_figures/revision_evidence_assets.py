"""Compact revision tables and implementation-labeled quality/latency figures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path("G:/Output Images/QRISP Final Benchmark 2026-10-04")
OUTPUT = ROOT / "output/peer_review_controls/2026-10-06"
FIGURES = ROOT / "docs/manuscript/figures"
OLD = ROOT / "output/paper_measurements/2026-09-29/qrisp_area_v2"
EDGES = ROOT / "output/edge_artifacts/2026-10-04_v1"
LABELS = {"aif_mixed": "AEISU mixed", "bilinear": "Bilinear", "bicubic": "Bicubic", "fsr1_easu": "EASU",
          "fsr1_rcas_02": "EASU+RCAS", "nis_0": "NIS", "sgsr1": "SGSR1", "repnet_x2": "RepNet",
          "rt4ksr_x2": "RT4KSR", "quicksrnet_small_x2": "QuickSRNet S", "quicksrnet_medium_x2": "QuickSRNet M",
          "span_x2": "SPAN", "swinir_light_x2": "SwinIR-Light"}
NEURAL = ("repnet_x2", "rt4ksr_x2", "quicksrnet_small_x2", "quicksrnet_medium_x2", "span_x2", "swinir_light_x2")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complexity_table():
    lines = [r"\begin{table}[t]", r"\centering\small",
             r"\caption{Deployed parameters and counted MACs for $960\times540$ to $1920\times1080$. "
             r"Counts include convolution/linear products and SwinIR attention products; AEISU also includes direct reconstruction. "
             r"Analysis arithmetic, additions, nonlinearities, and memory operations are excluded. AEISU includes its $992\times576$ controller padding; "
             r"SwinIR includes its window padding.}", r"\label{tab:comparator-complexity}",
             r"\begin{tabularx}{\columnwidth}{Yrr}", r"\toprule", r"Method & Parameters (k) & GMACs\\", r"\midrule"]
    records = []
    for name in NEURAL:
        record = read(OUTPUT / f"complexity_{name}.json")
        records.append(record)
        lines.append(f"{LABELS[name]} & {record['parameters'] / 1000:.2f} & {record['total_counted_macs'] / 1e9:.2f} " + r"\\")
    lines += [r"\midrule", r"AEISU &115.996&7.57\\", r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    records.append({"model": "aif_mixed", "parameters": 115996, "controller_macs_per_processed_pixel": 11520,
                    "processed_lr_shape": [576, 992], "requested_lr_shape": [540, 960],
                    "reconstruction_macs_per_requested_pixel": 1900,
                    "total_counted_macs": 11520 * 576 * 992 + 1900 * 540 * 960,
                    "counting": "Padded deployed convolution graph plus direct assembly/application on logical output sites; "
                                "excludes analysis arithmetic and non-MAC operations"})
    (FIGURES / "comparator_complexity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def edge_table():
    summary = read(EDGES / "summary.json")
    lines = [r"\begin{table*}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Target-selected isolated-edge diagnostic on 570 frames. B: bright excursion; R: excess profile ringing "
             r"(both percentages of target edge contrast, lower is better); W: 10--90\% transition width in HR pixels. "
             r"Means weight frames within environments, then environments equally. These are custom profile measurements, not whole-image quality scores.}",
             r"\label{tab:edge-profiles}", r"\begin{tabular}{l*{9}{r}}", r"\toprule",
             r" & \multicolumn{3}{c}{Area} & \multicolumn{3}{c}{Bicubic} & \multicolumn{3}{c}{Native}\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Method & B & R & W & B & R & W & B & R & W\\", r"\midrule"]
    for name in LABELS:
        numbers = []
        for condition in ("ssaa_area", "bicubic_aa", "native"):
            metrics = summary["conditions"][condition][name]["metrics"]
            numbers.extend(metrics[key] for key in ("bright_excursion_pct", "ringing_excess_pct", "width_10_90_px"))
        lines.append(LABELS[name] + " & " + " & ".join(f"{v:.3f}" for v in numbers) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (FIGURES / "isolated_edge_profiles.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def latency_points():
    timing = read(OLD / "comparison_summary.json")["timing_540p_to_1080p"]
    preflight = read(ARCHIVE / "preflight.json")["methods"]
    points = []
    audit = {}
    for name in ("bilinear", "fsr1_easu", "fsr1_rcas_02", "nis_0", "sgsr1", "repnet_x2", "rt4ksr_x2", "span_x2", "swinir_light_x2"):
        neural = name in NEURAL
        key = name + ("_fp16" if neural else "_fp32")
        protocol_path = OLD / key / "protocol.json"
        old = read(protocol_path)
        current = preflight[name]["metadata"]
        fields = ("checkpoint_sha256", "architecture_sha256", "prepared") if neural else ("exe_sha256", "shader_hashes")
        for field in fields:
            if old[field] != current[field]:
                raise ValueError(f"Different timed/quality identities for {name}/{field}")
        points.append({"model": name, "backend": "FP16 eager PyTorch" if neural else "FP32 D3D11 shader",
                       "latency_ms": timing[key]["median_ms"], "p95_ms": timing[key]["median_trial_p95_ms"]})
        audit[name] = {"protocol_sha256": digest(protocol_path), "runtime_sha256": digest(OLD / key / "runtime_960x540.json")}
    deployment = ROOT / "output/side_runtime/2026-10-04_deployment_compare"
    value = next(r for r in read(deployment / "summary.json") if r["input"] == [960, 540] and r["route"] == "tensor_mixed")
    if read(deployment / "protocol.json")["model"]["checkpoint_sha256"] != preflight["aif_mixed"]["metadata"]["checkpoint_sha256"]:
        raise ValueError("Different native model identities")
    points.append({"model": "aif_mixed", "backend": "Mixed FP8/FP16 CUDA graph", "latency_ms": value["median_ms"], "p95_ms": value["p95_ms"]})
    audit["aif_mixed"] = {"protocol_sha256": digest(deployment / "protocol.json"), "summary_sha256": digest(deployment / "summary.json")}
    return points, audit


def latency_figure(points):
    quality = read(ARCHIVE / "summary.json")["by_condition"]
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15))
    fig.subplots_adjust(left=.085, right=.99, top=.90, bottom=.31, wspace=.24)
    palette = plt.get_cmap("tab10")
    labels = []
    for number, point in enumerate(points, 1):
        name = point["model"]
        marker = "*" if name == "aif_mixed" else ("s" if name in NEURAL else "o")
        labels.append(f"{number}: {LABELS[name]}")
        for axis, condition in zip(axes, ("ssaa_area", "native")):
            x, y = point["latency_ms"], quality[condition]["all"][name]["scene_balanced"]["lpips_alex_v01"]
            axis.scatter(x, y, marker=marker, s=80 if name == "aif_mixed" else 30,
                         color=palette(number - 1), linewidths=.4, edgecolors="black", zorder=3)
            offset = (4, -9) if name == "repnet_x2" and condition == "ssaa_area" else (4, 4)
            axis.annotate(str(number), (x, y), xytext=offset, textcoords="offset points", fontsize=8)
    for axis, title in zip(axes, ("(a) Area", "(b) Native")):
        axis.set_xscale("log")
        axis.set_xlim(.01, 260)
        axis.set_ylim(.06, .29)
        axis.set_xlabel("Resident latency (ms; log scale)")
        axis.set_title(title)
        axis.grid(True, alpha=.25, linewidth=.5)
    axes[0].set_ylabel("RGB LPIPS-Alex (lower is better)")
    fig.text(.5, .13, "  ".join(labels[:5]), ha="center", fontsize=7)
    fig.text(.5, .085, "  ".join(labels[5:]), ha="center", fontsize=7)
    handles = [Line2D([], [], color="black", linestyle="", marker=marker, markersize=size, label=label)
               for marker, size, label in (("o", 4, "FP32 D3D11 shader"), ("s", 4, "FP16 eager PyTorch"),
                                          ("*", 7, "Mixed FP8/FP16 CUDA graph"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005), ncol=3, frameon=False, fontsize=7)
    for path in (FIGURES / "quality_latency_matched_geometry.pdf", OUTPUT / "quality_latency_matched_geometry.png"):
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    complexity = complexity_table()
    edge_table()
    points, audit = latency_points()
    latency_figure(points)
    record = {"complexity": complexity, "timing_points": points, "identity_checks": audit,
              "quality_summary_sha256": digest(ARCHIVE / "summary.json"), "edge_summary_sha256": digest(EDGES / "summary.json"),
              "source_sha256": digest(Path(__file__)),
              "scope": "Existing implementation timings at matched geometry; separate sessions/backends, "
                       "not architecture-only latency or a matched-runtime frontier"}
    (OUTPUT / "figure_evidence.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
