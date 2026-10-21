"""Calibrate and evaluate the DIV2K L1 checkpoint in the existing native mixed runtime."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
AESS = Path("{source_workspace}/AESS")
EXPORT = AESS / "tools/srcpyr_fp8"
CAMPAIGN = ROOT / "output/training_campaigns/div2k_l1_3000_2026-10-17"
WORK = CAMPAIGN / "mixed"
DATA = ROOT / "output/training_campaigns/fixed_stride_2026-10-16/div2k"
FLOAT_RESULTS = CAMPAIGN / "evaluations/classic_219"
OUTPUT = CAMPAIGN / "evaluations/classic_219_mixed"
NAME = "aeisu_div2k_l1_3000_mixed"
os.environ["TORCH_HOME"] = str(ROOT / "data/benchmark_assets/torch")

from evaluate_classic import PerceptualMetric, SETS, distortion, freeze, np, pairs, read_rgb, save_rgb, sha256, torch, write_json
from evaluate_div2k_l1 import NAME as FLOAT_NAME, KEYS, verify_outputs
from common import EXE, clear_native_scratch, native


def prepare_export():
    sys.path.insert(0, str(EXPORT))
    from adaptive_reference import (
        effective_base, effective_basis, effective_stencil, forward_controller, load_adaptive,
        prepare, reconstruct, reconstruction_features,
    )
    from model_format import ADAPTIVE_PARALLEL32X4_ARCHITECTURE_ID
    from models import build_raw

    state = json.loads((CAMPAIGN / "state.json").read_text())
    checkpoint = Path(state["checkpoint"])
    if state["phase"] != "complete" or sha256(checkpoint) != state["checkpoint_sha256"]:
        raise ValueError("Expected the verified final DIV2K L1 checkpoint")
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if original["epoch"] != 3000 or original["global_step"] != 915000 or original["experiment"]["model"] != {}:
        raise ValueError("Wrong training checkpoint")
    raw = build_raw({}).eval().requires_grad_(False)
    raw.load_state_dict(original["model"], strict=True)
    independently_folded = raw.folded_production_state_dict()
    stored_fold = torch.load(checkpoint.with_name(checkpoint.stem + "_folded_production.pt"), map_location="cpu", weights_only=True)
    saved_gpu_fold = stored_fold["model"]
    if set(independently_folded) != set(saved_gpu_fold):
        raise ValueError("Independent checkpoint folding keys differ")
    fold_error = max(float((independently_folded[key] - value).abs().max()) for key, value in saved_gpu_fold.items())
    # Match ControlAdapter's CPU FP32 folding of the training checkpoint, rather
    # than its separately saved GPU fold. Runtime parity below checks the bridge
    # against the full reparameterized model with TF32 disabled.
    folded = independently_folded
    WORK.mkdir(parents=True, exist_ok=True)
    bridge = WORK / "div2k_l1_3000_folded_native.pt"
    payload = {"architecture": ADAPTIVE_PARALLEL32X4_ARCHITECTURE_ID,
               "source_architecture": "aif16x5_reparam_aggressive64_parallel32x4_rgb4dir8_s32_64split_learnedbase_normbasis_oc25_v1",
               "folded_reparameterization": True, "config": original["config"],
               "model": folded, "epoch": 3000, "global_step": 915000, "metrics": original["metrics"],
               "training_checkpoint_sha256": sha256(checkpoint), "purpose": "Native-format bridge; CPU FP32 folding matches benchmark adapter"}
    if bridge.exists():
        old = torch.load(bridge, map_location="cpu", weights_only=True)
        if old["training_checkpoint_sha256"] != payload["training_checkpoint_sha256"]:
            raise ValueError("Native bridge already belongs to another checkpoint")
        if set(old["model"]) != set(folded) or any(not torch.equal(old["model"][k], v) for k, v in folded.items()):
            raise ValueError("Native bridge tensors differ from the CPU FP32 fold")
    else:
        torch.save(payload, bridge)
    torch.manual_seed(20261017)
    raw = raw.cuda()
    source = torch.rand(1, 3, 32, 40, device="cuda")
    runtime_state, runtime_payload = load_adaptive(bridge, torch.device("cuda"))
    stencil = effective_stencil(runtime_payload, torch.device("cuda"))
    packed, crop, padded = prepare(source, native_half_input=False, architecture=runtime_payload["architecture"],
                                   stencil=stencil, return_source=True)
    with torch.inference_mode():
        expected_meta = raw(padded)
        actual_meta = forward_controller(runtime_state, packed, is_parallel32x4=True)["metadata"]
        features = reconstruction_features(packed, effective_basis(runtime_payload, torch.device("cuda")), crop,
                                           effective_base(runtime_payload, torch.device("cuda")), source=padded)
        actual_rgb = reconstruct(actual_meta, features, max_coeff=2., overclock=True)
        y, x, h, w = crop
        expected_rgb = raw.differentiable_predict_rgb(padded, expected_meta)[:, :, y:y + h, x:x + w].clamp(0, 1)
        visible = (slice(None), slice(None), slice(y // 4, (y + h) // 4), slice(x // 4, (x + w) // 4))
        torch.testing.assert_close(actual_meta[visible], expected_meta[visible], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(actual_rgb, expected_rgb, atol=3e-5, rtol=3e-5)
    record = {"passed": True, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "bridge_sha256": sha256(bridge), "bridge_uses_cpu_fp32_fold": True,
              "independent_cpu_vs_saved_gpu_fold_max_abs": fold_error,
              "controller_max_abs_difference": float((actual_meta - expected_meta).abs().max()),
              "visible_controller_max_abs_difference": float((actual_meta[visible] - expected_meta[visible]).abs().max()),
              "rgb_max_abs_difference": float((actual_rgb - expected_rgb).abs().max())}
    del raw, runtime_state, original, stored_fold
    gc.collect()
    torch.cuda.empty_cache()
    return bridge, record


def calibration_sets():
    manifest = json.loads((DATA / "pairs.json").read_text())
    selected = [manifest["pairs"][i] for i in np.linspace(0, 799, 30, dtype=int)]
    records = {"calibration": [], "calibration_check": []}
    for index, row in enumerate(selected):
        split = "calibration_check" if index % 5 == 4 else "calibration"
        source = DATA / "Input" / row["image"]
        if sha256(source) != row["input_sha256"]:
            raise ValueError("DIV2K calibration image hash changed")
        destination = WORK / split / row["image"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(source, destination)
        if sha256(destination) != row["input_sha256"]:
            raise ValueError("Calibration hardlink differs")
        records[split].append({"image": row["image"], "sha256": row["input_sha256"]})
    assert len(records["calibration"]) == 24 and len(records["calibration_check"]) == 6
    return records


def calibrate():
    if shutil.disk_usage(ROOT).free < 50 * 1024**3:
        raise RuntimeError("Insufficient active SSD headroom")
    bridge, bridge_check = prepare_export()
    splits = calibration_sets()
    sources = {p.name: sha256(p) for p in EXPORT.glob("*.py")}
    preflight = {"bridge_check": bridge_check, "splits": splits,
                 "selection": "30 evenly spaced official training identities; every fifth reserved for calibration checking",
                 "check_scope": "Six DIV2K training images excluded from calibration; not held out from model training",
                 "exporter_sources": sources, "native_executable_sha256": sha256(EXE)}
    freeze(WORK / "preflight.json", preflight)
    artifact = WORK / "artifact/adaptive_fp8.bin"
    report = artifact.parent / "calibration_report.json"
    if not artifact.exists() or not report.exists():
        args = [sys.executable, "-u", str(EXPORT / "export_adaptive.py"), "--checkpoint", str(bridge),
                "--output-dir", str(artifact.parent), "--calibration-dir", str(WORK / "calibration"),
                "--heldout-dir", str(WORK / "calibration_check"), "--images", "24", "--heldout-images", "6",
                "--crop-size", "128", "--search-passes", "2", "--precision", "mixed"]
        write_json(WORK / "calibration_command.json", {"command": args})
        with (WORK / "calibration.log").open("a", encoding="utf-8") as log:
            subprocess.run(args, cwd=AESS, env=os.environ | {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(report.read_text())
    if result["checkpoint_sha256"] != sha256(bridge) or result["precision_policy"] != "mixed":
        raise ValueError("Calibration report does not match DIV2K bridge and precision")
    if result["fp16_stages"] != ["project", "refine", "tail"]:
        raise ValueError("Unexpected mixed layer policy")
    return artifact, preflight


def evaluate(artifact, preflight):
    records = pairs()
    metric = PerceptualMetric()
    old_protocol = json.loads((FLOAT_RESULTS / "protocol.json").read_text())
    old_summary = json.loads((FLOAT_RESULTS / "summary.json").read_text())
    if records != old_protocol["images"] or metric.protocol != old_protocol["metrics"] or not old_summary["complete"]:
        raise ValueError("Floating and mixed benchmark protocols differ")
    float_rows = [json.loads(line) for line in (FLOAT_RESULTS / "per_image.jsonl").read_text().splitlines() if line.strip()]
    references = {(r["dataset"], r["image"]): r for r in float_rows}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = {"method": NAME, "images": records, "sets": SETS, "metrics": metric.protocol,
                "artifact": str(artifact), "artifact_sha256": sha256(artifact),
                "native_executable": str(EXE), "native_executable_sha256": sha256(EXE),
                "training_checkpoint_sha256": preflight["bridge_check"]["checkpoint_sha256"],
                "calibration_preflight": preflight, "calibration_report_sha256": sha256(artifact.parent / "calibration_report.json"),
                "precision": "Native stage-specific FP8/FP16; FP16 project/refine/head; source/output RGBA16F resources",
                "selection": "All 219 unchanged benchmark pairs; final L1/3000-epoch checkpoint",
                "float_reference": str(FLOAT_RESULTS), "float_summary_sha256": sha256(FLOAT_RESULTS / "summary.json"),
                "scope": "Quality study; native timing below is a smoke check, not a new performance benchmark",
                "source_sha256": sha256(Path(__file__))}
    freeze(OUTPUT / "protocol.json", protocol)
    path = OUTPUT / "per_image.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    expected = {(r["dataset"], r["hr_filename"]) for r in records}
    done = {(r["dataset"], r["image"]) for r in rows}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError("Invalid mixed benchmark identities")
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        for record in records:
            key = record["dataset"], record["hr_filename"]
            if key in done:
                continue
            low = read_rgb(Path(record["lr_path"])).astype(np.float32) / 255
            high = read_rgb(Path(record["hr_path"])).astype(np.float32) / 255
            high = high[:low.shape[0] * 2, :low.shape[1] * 2]
            scratch = OUTPUT / "scratch" / record["dataset"] / Path(record["hr_filename"]).stem
            outputs, _ = native(low, scratch, artifact=artifact, warmup=0, iterations=1)
            prediction = np.clip(outputs["mixed"], 0, 1)
            native_fp16 = np.clip(outputs["fp16"], 0, 1)
            with np.load(references[key]["outputs"]["npz"], allow_pickle=False) as archive:
                reference = archive["rgb"]
            difference = prediction.astype(np.float64) - reference
            native_difference = prediction.astype(np.float64) - native_fp16
            values = distortion(prediction, high) | {"lpips_alex_v01": metric.score(prediction, high)}
            saved = save_rgb(prediction, OUTPUT / "outputs" / record["dataset"] / record["hr_filename"], verify=True)
            row = {"method": NAME, "dataset": key[0], "image": key[1], "metrics": values, "outputs": saved,
                   "input_sha256": record["lr_sha256"], "target_sha256": record["hr_sha256"],
                   "mixed_vs_common_fp16_mse": float(np.mean(difference ** 2)),
                   "mixed_vs_native_fp16_mse": float(np.mean(native_difference ** 2)),
                   "native_fp16_vs_common_fp16_mse": float(np.mean((native_fp16.astype(np.float64) - reference) ** 2))}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            done.add(key)
            clear_native_scratch(scratch)
            write_json(OUTPUT / "status.json", {"phase": "evaluating", "completed": len(rows), "total": 219})
            print(f"Mixed {len(rows)}/219 {key[0]} {key[1]} PSNR={values[KEYS[0]]:.4f}", flush=True)
    if done != expected:
        raise ValueError("Incomplete native mixed benchmark")
    verification = verify_outputs(rows, records, metric)
    means = {dataset: old_summary["means"][dataset] | {NAME: {
        key: float(np.mean([r["metrics"][key] for r in rows if r["dataset"] == dataset]))
        for key in rows[0]["metrics"]}} for dataset in SETS}
    deltas = {dataset: {key: means[dataset][NAME][key] - means[dataset][FLOAT_NAME][key] for key in KEYS} for dataset in SETS}
    agreement = {}
    for field in ("mixed_vs_common_fp16_mse", "mixed_vs_native_fp16_mse", "native_fp16_vs_common_fp16_mse"):
        scores = [-10 * np.log10(r[field]) if r[field] else 200. for r in rows]
        agreement[field.replace("mse", "psnr_db")] = {"mean": float(np.mean(scores)), "minimum": float(min(scores))}
    write_json(OUTPUT / "summary.json", {"complete": True, "images": 219, "new_reconstructions": 219,
                                        "means": means, "mixed_minus_fp16": deltas, "output_agreement": agreement})
    write_json(OUTPUT / "verification.json", verification)
    lines = ["# DIV2K L1: native mixed-FP8/FP16 quality", "", "Same final 3000-epoch checkpoint and 219 bicubic benchmark images.",
             "24 DIV2K training inputs calibrate scales and bias correction; six other training inputs check calibration.",
             "No benchmark image enters calibration. Floating scores and competitor scores are reused from verified archives.",
             "Native mixed uses FP16 projection/refinement/head and FP8 internal stages, including native half-precision image resources."]
    for dataset in SETS:
        lines += ["", "## " + dataset, "", "| Deployment | Y PSNR | Y SSIM | LPIPS |", "|---|---:|---:|---:|"]
        for name in (FLOAT_NAME, NAME):
            v = means[dataset][name]
            lines.append(f"| {name} | {v[KEYS[0]]:.5f} | {v[KEYS[1]]:.7f} | {v[KEYS[2]]:.7f} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUTPUT / "status.json", {"phase": "complete", "completed": len(rows), "total": 219})
    print(json.dumps({"mixed_minus_fp16": deltas, "output_agreement": agreement}, indent=2))


def main():
    if torch.version.cuda != "13.0":
        raise RuntimeError("Torch CUDA 13.0 required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    artifact, preflight = calibrate()
    evaluate(artifact, preflight)


if __name__ == "__main__":
    main()
