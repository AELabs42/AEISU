"""CPU-only post-hoc edge diagnostics on the immutable saved QRISP predictions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import time

# No Torch, CUDA, D3D, model-loader or LPIPS imports: all predictions already exist.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import cv2
import numpy as np

from profiles import METRICS, PRIMARY, Protocol, profile_metrics, sample, select_edges

ARCHIVE = Path("G:/Output Images/QRISP Final Benchmark 2026-10-04")
DESTINATION = Path("{research_workspace}/output/edge_artifacts/2026-10-04_v1")
CONDITIONS = ("ssaa_area", "bicubic_aa", "fxaa", "smaa", "native")
METHODS = ("aif_mixed", "bilinear", "bicubic", "fsr1_easu", "fsr1_rcas_02", "nis_0", "sgsr1",
           "repnet_x2", "rt4ksr_x2", "quicksrnet_small_x2", "quicksrnet_medium_x2", "span_x2", "swinir_light_x2")
LABELS = dict(zip(METHODS, ("AEISU mixed", "Bilinear", "Bicubic", "EASU", "EASU+RCAS", "NIS", "SGSR1",
                           "RepNet", "RT4KSR", "QuickSRNet Small", "QuickSRNet Medium", "SPAN", "SwinIR-Light")))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_array(record):
    path = Path(record["outputs"]["npz"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["outputs"]["npz_sha256"]:
        raise ValueError(f"Changed archived prediction: {path}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
        value = stored["rgb"]
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"Invalid archived floating prediction: {path}")
    return value


def collect(archive):
    selected = read_json(archive / "selection.json")
    verification = read_json(archive / "verification_v3.json")
    if not verification.get("passed"):
        raise ValueError("The independent archive verification did not pass")
    pairs = selected["pairs"]
    rows = {}
    identities = {(condition, pair["image"]) for condition in CONDITIONS for pair in pairs}
    hashes = {}
    for method in METHODS:
        path = archive / "results" / method / "per_image.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != len(identities) or {(row["condition"], row["image"]) for row in records} != identities:
            raise ValueError(f"Missing/duplicate predictions for {method}")
        rows[method] = {(row["condition"], row["image"]): row for row in records}
        hashes[str(path)] = digest(path)
    return pairs, rows, hashes


def score_pair(pair, archive, destination, indexed, protocol):
    target_path = archive / "targets" / pair["image"]
    if digest(target_path) != pair["hr_sha256"]:
        raise ValueError(f"Changed target: {target_path}")
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    edges = select_edges(target, pair["image"], protocol)
    cache_path = destination / "edge_regions" / Path(pair["image"]).with_suffix(".npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **{key: value for key, value in edges.items()})
    coverage = {"image": pair["image"], "scene": pair["scene"], "segment": pair["segment"], "group": pair["group"],
                "candidates": edges["candidates"], "eligible_before_cap": edges["eligible_before_cap"],
                "edges": len(edges["x"]), "target_sha256": pair["hr_sha256"], "edge_cache": str(cache_path)}
    if not coverage["edges"]:
        return coverage, []
    outputs = []
    target_summary, _ = profile_metrics(edges["target_profiles"], edges, protocol)
    coverage["target_width_10_90_px"] = target_summary["width_10_90_px"]
    for condition in CONDITIONS:
        for method in METHODS:
            record = indexed[method][condition, pair["image"]]
            if record["target_sha256"] != pair["hr_sha256"]:
                raise ValueError("Archive target identities do not match")
            value = load_array(record)
            if value.shape != target.shape:
                raise ValueError("Prediction/target geometry differs")
            profiles = sample(value, edges["x"], edges["y"], edges["nx"], edges["ny"], protocol)
            metrics, _ = profile_metrics(profiles, edges, protocol)
            outputs.append({key: pair[key] for key in ("image", "scene", "segment", "group")} |
                           {"condition": condition, "method": method, "edges": coverage["edges"], "metrics": metrics,
                            "quality": record["metrics"], "output_sha256": record["outputs"]["npz_sha256"]})
    return coverage, outputs


def balanced_mean(rows, keys, category):
    scenes = sorted({row["scene"] for row in rows})
    result = {}
    for key in keys:
        scene_values = []
        for scene in scenes:
            values = [row[category][key] for row in rows if row["scene"] == scene and row[category][key] is not None]
            if values:
                scene_values.append(float(np.mean(values)))
        result[key] = float(np.mean(scene_values)) if scene_values else None
    return result


def bootstrap_weights(pairs, draws):
    rng = np.random.default_rng(2026100407)
    scenes = sorted({pair["scene"] for pair in pairs})
    scene_counts = rng.multinomial(len(scenes), np.ones(len(scenes)) / len(scenes), draws)
    weights = np.zeros((draws, len(pairs)), dtype=np.float64)
    point = np.zeros(len(pairs), dtype=np.float64)
    for scene_index, scene in enumerate(scenes):
        indices = np.array([i for i, pair in enumerate(pairs) if pair["scene"] == scene])
        segments = sorted({pairs[i]["segment"] for i in indices})
        sequence_id = np.array([segments.index(pairs[i]["segment"]) for i in indices])
        counts = rng.multinomial(len(segments), np.ones(len(segments)) / len(segments), draws)
        raw = counts[:, sequence_id].astype(np.float64)
        weights[:, indices] = raw / raw.sum(axis=1, keepdims=True) * scene_counts[:, scene_index, None] / len(scenes)
        point[indices] = 1 / len(indices) / len(scenes)
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
    return point, weights


def summarize(rows, coverage, draws):
    good = [row for row in coverage if row["edges"]]
    point, weights = bootstrap_weights(good, draws)
    result = {"conditions": {}, "paired_contrasts": {}, "coverage": {
        "frames": len(coverage), "frames_with_edges": len(good), "edges": sum(row["edges"] for row in coverage),
        "by_scene": {scene: {"frames": sum(row["scene"] == scene for row in coverage),
                             "eligible_frames": sum(row["scene"] == scene and row["edges"] > 0 for row in coverage),
                             "edges": sum(row["edges"] for row in coverage if row["scene"] == scene)}
                     for scene in sorted({row["scene"] for row in coverage})}}}
    quality_keys = ("y_psnr_shave2_db", "y_ssim_shave2", "lpips_alex_v01")
    for condition in CONDITIONS:
        result["conditions"][condition] = {}
        matrices = {}
        for method in METHODS:
            selected = [row for row in rows if row["condition"] == condition and row["method"] == method]
            by_image = {row["image"]: row for row in selected}
            if set(by_image) != {row["image"] for row in good}:
                raise ValueError("Unpaired artifact sample")
            result["conditions"][condition][method] = {
                "images": len(selected), "metrics": balanced_mean(selected, METRICS, "metrics"),
                "quality_same_eligible_frames": balanced_mean(selected, quality_keys, "quality")}
            matrices[method] = np.array([[by_image[pair["image"]]["metrics"][key] for key in PRIMARY] for pair in good])
        deltas = np.concatenate([matrices["aif_mixed"] - matrices[method] for method in METHODS[1:]], axis=1)
        mean = point @ deltas
        boot = weights @ deltas
        sd = boot.std(axis=0, ddof=1)
        safe = np.maximum(sd, 1e-12)
        cutoff = np.quantile(np.max(np.abs((boot - mean) / safe), axis=1), .95)
        result["paired_contrasts"][condition] = []
        for index, method in enumerate(METHODS[1:]):
            for metric_index, key in enumerate(PRIMARY):
                column = index * len(PRIMARY) + metric_index
                result["paired_contrasts"][condition].append({"comparator": method, "metric": key,
                    "aeisu_minus_comparator": float(mean[column]),
                    "nominal_95_interval": np.quantile(boot[:, column], [.025, .975]).tolist(),
                    "simultaneous_95_interval": [float(mean[column] - cutoff * sd[column]), float(mean[column] + cutoff * sd[column])],
                    "paired_frame_lower_fraction": float(point @ (deltas[:, column] < 0))})
    return result


def render_report(destination, summary, protocol):
    coverage = summary["coverage"]
    lines = ["# Saved-output isolated-edge artifact diagnostic", "",
             "CPU-only, post-hoc analysis. The manuscript and original predictions are unchanged.", "",
             f"Target-defined sample: {coverage['frames_with_edges']}/{coverage['frames']} frames, {coverage['edges']} edge profiles.", "",
             "## Interpretation and scope", "",
             "This is a custom diagnostic validated on synthetic cases, not an established perceptual quality metric. "
             "Lower excursion/ringing is desirable on this isolated-edge sample, but blur and even a constant image can score zero. "
             "Read retained contrast, transition width, alignment and existing fidelity/perceptual scores alongside it.", "",
             "Eligibility uses only the unchanged Native HR targets: coherent, contrast-bearing edges with approximately flat "
             "flanks and limited target oscillation. A target-gradient spatial grid and fixed-seed cap choose regions. "
             "All methods and input conditions use identical positions and normals. No outputs are used to select edges.", "",
             "Profiles span ±12 HR pixels at half-pixel spacing, averaging three parallel tangent samples. "
             "Plateau medians use the outer 8–12 pixels. Excursions are measured beyond each prediction's own plateaus, "
             "subtract the target's own excursions, and normalize by target edge contrast. "
             "Ringing is excess non-monotonic total variation divided by twice target contrast. "
             "RGB excursions are a secondary channel-envelope measure normalized by target luma contrast.", "",
             "Predictions were already clipped to [0,1] in the archived benchmark; pre-clamp excursions cannot be recovered. "
             "The sample excludes textured/multiple-edge regions and does not estimate all-image halo prevalence or temporal artifacts. "
             "Area is the primary same-HR-derived comparison. Native/FXAA/SMAA can involve independently rendered LR/HR differences, "
             "so measured oscillations there cannot automatically be attributed to synthesis artifacts.", "",
             "Frame means are equally weighted within environment, then environments equally weighted. "
             "Paired uncertainty resamples environments and their source-sequence clusters. "
             "Simultaneous bands cover 12 comparators × 3 primary diagnostics separately for each input condition. "
             "These are post-hoc diagnostics on five environments, not training-seed uncertainty or population-wide significance.", "",
             "## Coverage", "", "| Environment | Eligible frames | Selected edges |", "| --- | ---: | ---: |"]
    for scene, values in coverage["by_scene"].items():
        lines.append(f"| {scene} | {values['eligible_frames']}/{values['frames']} | {values['edges']} |")
    for condition in CONDITIONS:
        lines += ["", f"## {condition}", "", "Excursion/ringing are percentages of target edge contrast, not percentages of image pixels.", "",
                  "| Method | Bright % | Dark % | Ringing % | RGB % | Width px | Contrast ratio | Y PSNR | LPIPS |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method in METHODS:
            values = summary["conditions"][condition][method]
            metrics, quality = values["metrics"], values["quality_same_eligible_frames"]
            keys = ("bright_excursion_pct", "dark_excursion_pct", "ringing_excess_pct", "rgb_excursion_pct", "width_10_90_px", "contrast_ratio")
            numbers = [metrics[key] for key in keys] + [quality["y_psnr_shave2_db"], quality["lpips_alex_v01"]]
            formatted = ["—" if value is None else f"{value:.4f}" for value in numbers]
            lines.append(f"| {LABELS[method]} | " + " | ".join(formatted) + " |")
        lines += ["", "### AEISU paired contrasts", "", "Negative differences mean lower measured artifacts for AEISU.", "",
                  "| Comparator | Diagnostic | Difference | Simultaneous 95% band |", "| --- | --- | ---: | --- |"]
        for row in summary["paired_contrasts"][condition]:
            low, high = row["simultaneous_95_interval"]
            lines.append(f"| {LABELS[row['comparator']]} | {row['metric']} | {row['aeisu_minus_comparator']:.4f} | [{low:.4f}, {high:.4f}] |")
    lines += ["", "## Reproduction", "", "Frozen settings and source identities: `protocol.json`. "
              "Frame-level records: `per_image.jsonl`. Region coordinates and target profiles: `edge_regions/`. "
              "Complete coverage: `coverage.json`. Aggregate values and paired intervals: `summary.json`.", ""]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    colors = plt.get_cmap("tab20")
    for axis, condition in zip(axes, ("ssaa_area", "native")):
        for index, method in enumerate(METHODS):
            metrics = summary["conditions"][condition][method]["metrics"]
            x, y = metrics["width_10_90_px"], metrics["bright_excursion_pct"] + metrics["dark_excursion_pct"]
            axis.scatter(x, y, color=colors(index), label=f"{index + 1}: {LABELS[method]}")
            axis.annotate(str(index + 1), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
        axis.set(title=condition, xlabel="Mean 10–90% edge width (HR pixels)", ylabel="Bright + dark excursion (% target contrast)")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7)
    fig.savefig(destination / "edge_tradeoff.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0, help="Uniform index-quantile pilot; use a separate output folder")
    args = parser.parse_args()
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
    if args.workers < 1 or args.draws < 100:
        raise ValueError("Invalid worker/bootstrap settings")
    pairs, indexed, score_hashes = collect(args.archive)
    if args.limit and args.limit < len(pairs):
        pairs = [pairs[index] for index in np.linspace(0, len(pairs) - 1, args.limit, dtype=int)]
    protocol = Protocol()
    record = {"name": "Target-defined isolated-edge diagnostic v1", "date": "2026-10-04", "device": "CPU only",
              "settings": protocol.record(), "frames": [pair["image"] for pair in pairs], "conditions": CONDITIONS,
              "methods": METHODS, "workers": args.workers, "bootstrap_draws": args.draws,
              "selection_sha256": digest(args.archive / "selection.json"), "score_manifest_hashes": score_hashes,
              "code_sha256": {path.name: digest(path) for path in Path(__file__).parent.glob("*.py")},
              "software": {"numpy": np.__version__, "opencv": cv2.__version__}}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    record = json.loads(json.dumps(record))
    if protocol_path.exists() and read_json(protocol_path) != record:
        raise ValueError("Protocol changed; use a new versioned output folder")
    write_json(protocol_path, record)
    coverage, rows = [], []
    started = time.perf_counter()
    score_path = args.output / "per_image.jsonl"
    with score_path.open("w", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = executor.map(lambda pair: score_pair(pair, args.archive, args.output, indexed, protocol), pairs)
        for index, (region, scores) in enumerate(jobs, 1):
            coverage.append(region)
            rows.extend(scores)
            for row in scores:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if index == 1 or index % 10 == 0 or index == len(pairs):
                print(f"Scored {index}/{len(pairs)} frames; {len(rows)} predictions; "
                      f"{sum(r['edges'] for r in coverage)} target edges; {time.perf_counter() - started:.1f}s", flush=True)
    write_json(args.output / "coverage.json", coverage)
    summary = summarize(rows, coverage, args.draws)
    summary["elapsed_seconds"] = time.perf_counter() - started
    summary["predictions_scored"] = len(rows)
    summary["protocol_sha256"] = digest(protocol_path)
    write_json(args.output / "summary.json", summary)
    render_report(args.output, summary, protocol)
    print(f"Completed CPU-only diagnostic: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
