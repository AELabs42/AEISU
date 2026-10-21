"""Archive spatial controls and build fresh performance tables, figures and report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from common import ROOT, read_rgb, sha256, write_json
from comparison_models import SPATIAL_EXE, spatial
from timing import gpu_status

ARCHIVE = Path("G:/Output Images/QRISP Final Benchmark 2026-10-04")
LABELS = {"aeisu_fp16": "AEISU", "aeisu_fp16_fp32_synthesis": "AEISU (FP32 synthesis)",
          "quicksrnet_small_x2": "QuickSRNet S", "quicksrnet_medium_x2": "QuickSRNet M",
          "repnet_x2": "RepNet", "rt4ksr_x2": "RT4KSR", "span_x2": "SPAN", "swinir_light_x2": "SwinIR-Light",
          "aif_mixed": "AEISU mixed", "bilinear": "Bilinear", "fsr1_easu": "EASU", "fsr1_rcas_02": "EASU+RCAS",
          "nis_0": "NIS", "sgsr1": "SGSR1"}
NEURAL = ("aeisu_fp16", "quicksrnet_small_x2", "quicksrnet_medium_x2", "repnet_x2", "rt4ksr_x2",
          "span_x2", "swinir_light_x2")
SPATIAL = ("bilinear", "fsr1_easu", "fsr1_rcas_02", "nis_0", "sgsr1")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measure_spatial(folder):
    target = folder / "spatial"
    if (target / "summary.json").exists():
        raise FileExistsError("Preserve the completed spatial sweep")
    path = ROOT / "data/Benchmark Data/Urban100/HR/img001.png"
    image = read_rgb(path).astype(np.float32) / 255
    write_json(target / "protocol.json", {"input": str(path), "input_sha256": sha256(path),
               "input_preparation": "RGB/255; OpenCV INTER_LINEAR", "trials": 5, "warmup": 100, "iterations": 200,
               "timing": "Resident D3D11 dispatch timestamp; loading/transfers excluded", "precision": "FP32",
               "hardware_start": gpu_status(), "exe_sha256": sha256(SPATIAL_EXE),
               "shader_hashes": {p.name: sha256(p) for p in SPATIAL_EXE.parent.glob("*.cso")},
               "source_sha256": sha256(Path(__file__))})
    rows = []
    for height in (540, 1080):
        width = height * 16 // 9
        rgb = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        for trial in range(5):
            methods = SPATIAL if trial % 2 == 0 else tuple(reversed(SPATIAL))
            for name in methods:
                before = gpu_status()
                _, result = spatial(rgb, target / f"{name}_{height}_{trial}", name, warmup=100, iterations=200, output=False)
                rows.append({"model": name, "input_shape_wh": [width, height], "trial": trial + 1,
                             "result": result, "before": before, "after": gpu_status()})
                write_json(target / "trials.json", rows)
            print(f"Spatial {width}x{height} trial {trial + 1}/5 complete", flush=True)
    summary = []
    for height in (540, 1080):
        for name in SPATIAL:
            values = [r["result"] for r in rows if r["model"] == name and r["input_shape_wh"][1] == height]
            summary.append({"model": name, "input_shape_wh": [height * 16 // 9, height],
                            "median_ms": statistics.median(r["median_ms"] for r in values),
                            "p95_ms": statistics.median(r["p95_ms"] for r in values),
                            "median_range_ms": [min(r["median_ms"] for r in values), max(r["median_ms"] for r in values)]})
    write_json(target / "summary.json", summary)


def table(lines, caption, label, header, specification="Yrrrr", wide=False):
    environment = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    return "\n".join([f"\\begin{{{environment}}}[t]", r"\centering\small", f"\\caption{{{caption}}}",
                      f"\\label{{{label}}}", f"\\begin{{tabularx}}{{{width}}}{{{specification}}}",
                      r"\toprule", header + r"\\", r"\midrule", *lines, r"\bottomrule", r"\end{tabularx}",
                      f"\\end{{{environment}}}"]) + "\n"


def report(folder):
    pytorch = {(r["model"], r["input_shape_wh"][1]): r for r in read(folder / "pytorch_summary.json")}
    if len(pytorch) != 16:
        raise ValueError("Expected eight ordinary-PyTorch variants at two resolutions")
    sampled = 0
    for (name, height), summary in pytorch.items():
        directory = folder / "pytorch" / f"{name}_{height}"
        rows = read(directory / "trials.json")
        assert len(rows) == 10
        assert len({(r["trial"], r["execution"]) for r in rows}) == 10
        for row in rows:
            assert len(row["result"]["samples_ms"]) == 50
            assert row["result"]["p95_ms"] >= row["result"]["median_ms"] > 0
            sampled += len(row["result"]["samples_ms"])
        assert read(directory / "graph_parity.json")["max_abs"] <= 0.002
        protocol = read(directory / "protocol.json")
        assert protocol["script_sha256"] == sha256(folder / "sources/runtime_comparison.py")
        assert protocol["adapter_sha256"] == sha256(folder / "sources/runtime_torch_models.py")
    native_rows = read(folder / "native/trials.json")
    deployment_rows = read(folder / "deployment/trials.json")
    spatial_rows = read(folder / "spatial/trials.json")
    assert len(native_rows) == 40 and len(deployment_rows) == 40 and len(spatial_rows) == 50
    assert len({(r["resolution"], r["mode"], r["trial"]) for r in native_rows}) == 40
    assert len({(tuple(r["input"]), r["route"], r["trial"]) for r in deployment_rows}) == 40
    assert len({(tuple(r["input_shape_wh"]), r["model"], r["trial"]) for r in spatial_rows}) == 50
    assert read(folder / "deployment/instruction_audit.json")["counts"]["tensor_ops"] == 0
    write_json(folder / "campaign_checks.json", {"pytorch_model_shape_jobs": 16, "pytorch_trial_sets": 160,
               "pytorch_timed_forwards": sampled, "native_resolution_mode_trials": 40, "deployment_route_trials": 40,
               "spatial_method_trials": 50, "graph_parity_passed": True, "source_snapshots_verified": True,
               "shader_tensor_instruction_count": 0, "all_expected_measurements_complete": True})
    native = read(folder / "native/summary.json")
    deployments = read(folder / "deployment/summary.json")
    shaders = read(folder / "spatial/summary.json")
    preflight = read(ARCHIVE / "preflight.json")["methods"]
    points, identities = [], {}
    for name in NEURAL[1:]:
        protocol_path = folder / "pytorch" / f"{name}_540/protocol.json"
        metadata = read(protocol_path)["metadata"]
        old = preflight[name]["metadata"]
        if name.startswith("quicksrnet"):
            assert metadata["checkpoint"]["sha256"] == old["checkpoint"]["sha256"]
            assert metadata["source_hashes"] == old["source_hashes"]
        else:
            for key in ("checkpoint_sha256", "architecture_sha256", "prepared"):
                assert metadata[key] == old[key], (name, key)
        result = pytorch[name, 540]["results"]["cuda_graph"]
        points.append({"model": name, "backend": "FP16 PyTorch CUDA graph", "latency_ms": result["median_ms"],
                       "p95_ms": result["p95_ms"]})
        identities[name] = {"protocol_sha256": sha256(protocol_path)}
    protocol_path = folder / "native/protocol.json"
    assert read(protocol_path)["checkpoint_sha256"] == preflight["aif_mixed"]["metadata"]["checkpoint_sha256"]
    result = native["960x540"]["minimal"]["mixed"]
    points.append({"model": "aif_mixed", "backend": "Mixed custom CUDA graph",
                   "latency_ms": result["median_of_medians_ms"], "p95_ms": result["median_of_p95_ms"]})
    identities["aif_mixed"] = {"protocol_sha256": sha256(protocol_path)}
    spatial_protocol = read(folder / "spatial/protocol.json")
    for name in SPATIAL:
        old = preflight[name]["metadata"]
        assert spatial_protocol["exe_sha256"] == old["exe_sha256"]
        assert spatial_protocol["shader_hashes"] == old["shader_hashes"]
        r = next(r for r in shaders if r["model"] == name and r["input_shape_wh"][1] == 540)
        points.append({"model": name, "backend": "FP32 D3D11 shader", "latency_ms": r["median_ms"], "p95_ms": r["p95_ms"]})
    assets = ROOT / "docs/manuscript/figures"
    complexity = ROOT / "output/peer_review_controls/2026-10-06"
    lines = []
    for name in NEURAL:
        if name == "aeisu_fp16":
            parameters, macs = 115996, 7.56739584
        else:
            counts = read(complexity / f"complexity_{name}.json")
            parameters, macs = counts["parameters"], counts["total_counted_macs"] / 1e9
        result = pytorch[name, 540]["results"]
        lines.append(f"{LABELS[name]} & {parameters / 1000:.2f} & {macs:.2f} & "
                     f"{result['eager']['median_ms']:.3f} & {result['cuda_graph']['median_ms']:.3f}" + r"\\")
    (assets / "comparator_complexity.tex").write_text(table(lines,
        r"Deployed complexity and FP16 PyTorch timing (ms), 540p-to-1080p. "
        r"MACs include attention, AEISU reconstruction and required padding, but exclude analysis and non-MAC operations. "
        r"E: eager; G: graph replay.",
        "tab:comparator-complexity", "Method & Param. (k) & GMACs & E & G"), encoding="utf-8")
    lines = []
    for name in (*NEURAL, "aeisu_fp16_fp32_synthesis"):
        values = []
        for height in (540, 1080):
            for mode in ("eager", "cuda_graph"):
                r = pytorch[name, height]["results"][mode]
                values.append(f"{r['median_ms']:.3f} ({r['p95_ms']:.3f})")
        lines.append(LABELS[name] + " & " + " & ".join(values) + r"\\")
    (assets / "common_runtime.tex").write_text(table(lines,
        r"Common ordinary-PyTorch resident timing in ms, median (p95). Five trials, 20 warmup and 50 measured forwards; "
        r"batch one, full frame, FP16, TF32 disabled, channels-last, cuDNN autotuning. Eager and CUDA graph (G) execution "
        r"use the same operators. The final auxiliary row retains FP32 analysis and synthesis, unlike the uniform-FP16 rows.",
        "tab:common-runtime", " & 540p$\\to$1080p E & 540p$\\to$1080p G & 1080p$\\to$4K E & 1080p$\\to$4K G", wide=True),
        encoding="utf-8")
    lines = []
    for name in SPATIAL:
        values = []
        for height in (540, 1080):
            r = next(r for r in shaders if r["model"] == name and r["input_shape_wh"][1] == height)
            values.append(f"{r['median_ms']:.3f} ({r['p95_ms']:.3f})")
        lines.append(LABELS[name] + " & " + " & ".join(values) + r"\\")
    (assets / "spatial_runtime.tex").write_text(table(lines,
        "Resident FP32 D3D11 shader timing in ms, median (p95); five trials of 100 warmup and 200 measured dispatches.",
        "tab:external-runtime", "Method & 540p$\\to$1080p & 1080p$\\to$4K", specification="Yrr"), encoding="utf-8")
    quality = read(ARCHIVE / "summary.json")["by_condition"]
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.1))
    fig.subplots_adjust(left=.085, right=.99, top=.92, bottom=.40, wspace=.24)
    markers = {"FP16 PyTorch CUDA graph": "s", "Mixed custom CUDA graph": "*", "FP32 D3D11 shader": "o"}
    palette = plt.get_cmap("tab20")
    for number, point in enumerate(points, 1):
        for axis, condition in zip(axes, ("ssaa_area", "native")):
            x = point["latency_ms"]
            y = quality[condition]["all"][point["model"]]["scene_balanced"]["lpips_alex_v01"]
            axis.scatter(x, y, marker=markers[point["backend"]], s=85 if point["model"] == "aif_mixed" else 32,
                         color=palette(number - 1), edgecolors="black", linewidths=.4, zorder=3)
            offset = (4, -9) if point["model"] == "repnet_x2" else (4, 4)
            if point["model"] == "sgsr1" and condition == "native":
                offset = (-11, -10)
            axis.annotate(str(number), (x, y), xytext=offset, textcoords="offset points", fontsize=8)
    for axis, title in zip(axes, ("(a) Area", "(b) Native")):
        axis.set_xscale("log")
        axis.set_xlim(.01, 260)
        axis.set_ylim(.06, .29)
        axis.set_xlabel("Resident latency (ms; log scale)")
        axis.set_title(title)
        axis.grid(True, alpha=.25, linewidth=.5)
    axes[0].set_ylabel("RGB LPIPS-Alex (lower is better)")
    for row in range(3):
        subset = list(enumerate(points, 1))[row * 4:(row + 1) * 4]
        fig.text(.5, .22 - row * .05, "   ".join(f"{n}: {LABELS[p['model']]}" for n, p in subset), ha="center", fontsize=7)
    handles = [Line2D([], [], color="black", linestyle="", marker=m, markersize=6 if m == "*" else 4, label=b)
               for b, m in markers.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .01), ncol=3, frameon=False, fontsize=7)
    fig.savefig(assets / "quality_latency_matched_geometry.pdf", bbox_inches="tight", pad_inches=.06)
    fig.savefig(folder / "quality_latency_matched_geometry.png", dpi=220, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    record = {"timing_points": points, "quality_identity_checks": identities, "quality_sha256": sha256(ARCHIVE / "summary.json"),
              "scope": "Measured implementations on one GPU, not equally optimized backends or architecture-only speedup",
              "pytorch_summary_sha256": sha256(folder / "pytorch_summary.json"), "native_sha256": sha256(folder / "native/summary.json"),
              "deployment_sha256": sha256(folder / "deployment/summary.json"), "spatial_sha256": sha256(folder / "spatial/summary.json"),
              "source_sha256": sha256(Path(__file__))}
    write_json(folder / "figure_evidence.json", record)
    text = ["# AEISU runtime comparison - 6 October 2026", "", "## Common PyTorch", "",
            "All primary rows use batch-one full-frame FP16, channels-last, cuDNN autotuning and TF32 disabled. "
            "Each shape/model runs in an isolated process; eager and CUDA graph order alternate across five trials. "
            "CUDA events measure resident input through clipped output, including dynamic padding, analysis and reconstruction. "
            "Uploads, setup and disk I/O are outside the interval. No custom CUDA extension or torch.compile is used.", "",
            "| Model | 540p eager | 540p graph | 1080p eager | 1080p graph |", "|---|---:|---:|---:|---:|"]
    for name in (*NEURAL, "aeisu_fp16_fp32_synthesis"):
        values = [pytorch[name, height]["results"][mode] for height in (540, 1080) for mode in ("eager", "cuda_graph")]
        text.append("| " + LABELS[name] + " | " + " | ".join(f"{r['median_ms']:.3f} ({r['p95_ms']:.3f})" for r in values) + " |")
    text += ["", "Values are milliseconds, median (p95), each aggregated across trials. "
             "AEISU's ordinary-Torch adapter folds fixed analysis into one convolution, packs branch pairs as grouped convolutions, "
             "and reconstructs through unfold/GEMM/batched-matmul. Its FP16 random-probe output agrees with the FP32 checkpoint "
             "at over 68 dB; this is numerical agreement, not a target-quality benchmark. "
             "The auxiliary row retains FP32 analysis/synthesis and is not a uniform-precision comparison.", "",
             "## Optimized deployments", "", "| Route | 540p to 1080p | 1080p to 4K |", "|---|---:|---:|"]
    for route in ("tensor_fp16", "tensor_mixed", "shader_fp16", "shader_int8_fp16"):
        values = [next(r for r in deployments if r["route"] == route and r["input"][1] == h) for h in (540, 1080)]
        text.append("| " + route + " | " + " | ".join(f"{r['median_ms']:.3f} ({r['p95_ms']:.3f})" for r in values) + " |")
    text += ["", "This compares actual deployments, including different precision, fusion, layouts and instruction paths. "
             "It does not isolate Tensor Core hardware alone. Shader SASS is audited for zero matrix/tensor instructions.", "",
             "## Native scaling repeat", "", "| LR height | FP16 | uniform FP8 | mixed | mixed p95 |", "|---|---:|---:|---:|---:|"]
    for h in (540, 720, 1080, 1440):
        r = native[f"{h * 16 // 9}x{h}"]["minimal"]
        text.append(f"| {h} | {r['fp16']['median_of_medians_ms']:.3f} | {r['fp8']['median_of_medians_ms']:.3f} | "
                    f"{r['mixed']['median_of_medians_ms']:.3f} | {r['mixed']['median_of_p95_ms']:.3f} |")
    ratio = native["2560x1440"]["minimal"]["fp8"]["median_of_medians_ms"] / native["1920x1080"]["minimal"]["fp8"]["median_of_medians_ms"]
    text += ["", f"Uniform FP8 scales {ratio:.3f}x from 1080p to 1440p versus 1.778x logical pixels. "
             "The repeat reproduces nonlinear scaling, with stage growth in projection, refinement and the unfused coefficient head. "
             "Detailed events perturb execution and are not substituted for minimal-event total latency. "
             "The old 1080p mixed p95 of 0.678 ms is not reproduced; the fresh repeat is approximately 0.440 ms. "
             "Historical runs remain archived, with no unsupported attribution of the discrepancy to a single cause.", "",
             "## Interpretation", "", "AEISU is slower than QuickSRNet and RepNet in ordinary PyTorch. "
             "Its optimized deployment changes that ranking relative to QuickSRNet, while RepNet remains competitive in resident latency. "
             "The paper should present the optimized systems result and common-runtime result together, not infer an intrinsic "
             "architecture-only speed advantage. Quality remains degradation-dependent, and the latency-quality plot labels each backend. "
             "No output-head ablation performance claims are introduced.", "", "PyTorch/spatial raw samples, native timing logs, "
             "graph/eager checks, checkpoint/source hashes, "
             "instruction audit and before/after hardware records are retained below this report directory."]
    (folder / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measure-spatial", action="store_true")
    args = parser.parse_args()
    if args.measure_spatial:
        measure_spatial(args.output.resolve())
    else:
        report(args.output.resolve())


if __name__ == "__main__":
    main()
