"""Launch an isolated fresh DIV2K L1 run without modifying the completed MSE campaign."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from div2k import digest
from train_all import write_json

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output/training_campaigns/fixed_stride_2026-10-16/div2k"
CAMPAIGN = ROOT / "output/training_campaigns/div2k_l1_3000_2026-10-17"
PACKAGE = Path("ablations/33_div2k_l1_3000")


def command(campaign, snapshot):
    return [sys.executable, "-u", str(snapshot / PACKAGE / "train.py"),
            "--epochs", "3000", "--batch", "32", "--seed", "1234", "--gradient-phase", "fixed",
            "--dataset-label", "div2k_train800_matlab_x2",
            "--source-dir", str(DATA / "Input"), "--target-dir", str(DATA / "Target"),
            "--cache-dir", str(DATA / "_tensor_cache"), "--no-cache-update",
            "--models-root", str(campaign / "runs")]


def prepare(campaign):
    for path in (campaign, DATA):
        path.resolve().relative_to(ROOT.resolve())
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise RuntimeError("Need at least 100 GiB SSD headroom; no HDD spill is permitted")
    manifest_path = DATA / "pairs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {f"{i:04d}.png" for i in range(1, 801)}
    if {row["image"] for row in manifest["pairs"]} != expected or len(manifest["pairs"]) != 800:
        raise ValueError("DIV2K must contain exactly the 800 official training pairs")
    for folder in ("Input", "Target"):
        if {path.name for path in (DATA / folder).glob("*.png")} != expected:
            raise ValueError(f"Unexpected DIV2K files in {folder}")
    for row in manifest["pairs"]:
        for folder, key in (("Input", "input_sha256"), ("Target", "target_sha256")):
            if digest(DATA / folder / row["image"]) != row[key]:
                raise ValueError(f"DIV2K content changed: {folder}/{row['image']}")
    campaign.mkdir(parents=True, exist_ok=True)
    snapshot = campaign / "source_snapshot"
    sources = [*(ROOT / "ablations/_shared").rglob("*.py"), *(ROOT / PACKAGE).glob("*")]
    sources += [Path(__file__), Path(__file__).with_name("check_fused.py")]
    hashes = {}
    for source in sorted(sources):
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        hashes[relative.as_posix()] = digest(target)
    frozen_variant = json.loads((snapshot / PACKAGE / "variant.json").read_text())
    if frozen_variant["training"] != {"mse_loss_weight": 0.0, "l1_loss_weight": 1.0, "save_every_epochs": 10}:
        raise ValueError("Frozen experiment has the wrong loss or checkpoint policy")
    write_json(snapshot / "hashes.json", hashes)
    plan = {"dataset": "800 official DIV2K train MATLAB bicubic X2 pairs",
            "training_manifest": str(manifest_path), "training_manifest_sha256": digest(manifest_path),
            "epochs": 3000, "expected_updates_per_epoch": 305, "expected_total_updates": 915000,
            "loss": "RGB L1 only", "mse_loss_weight": 0.0, "l1_loss_weight": 1.0,
            "lr": 2e-3, "min_lr": 1e-6, "schedule": "cosine over all 3000 epochs",
            "gradient_policy": "fixed stride 16, offset (0,0)", "seed": 1234,
            "batch": 32, "hr_crop": 512, "augmentation": "D4", "precision": "BF16 AMP",
            "initialization": "fresh; not initialized or resumed from the previous MSE model",
            "checkpoint_selection": "final scheduled epoch; no benchmark-driven selection",
            "checkpoint_interval_epochs": 10, "command": command(campaign, snapshot),
            "sources": hashes}
    write_json(campaign / "plan.json", plan)
    return snapshot, plan


def run(campaign, resume):
    import msvcrt

    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / ".training.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        state = {"phase": "preflight", "pid": os.getpid(), "started": time.time()}
        write_json(campaign / "state.json", state)
        try:
            existing = list((campaign / "runs").glob("Training Run */Pytorch Models/*_ep*.pt"))
            existing = [p for p in existing if not p.stem.endswith("_folded_production")]
            if existing and not resume:
                raise RuntimeError("This run already has checkpoints; use --resume rather than starting a duplicate")
            snapshot, plan = prepare(campaign)
            env = os.environ | {"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
            check = snapshot / "tools/training_campaign/check_fused.py"
            with (campaign / "fused_l1_check.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-u", str(check), "--loss", "l1", "--output", str(campaign / "fused_l1_check.json")],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            args = plan["command"]
            if resume and existing:
                args += ["--resume", str(max(existing, key=lambda p: int(p.stem.rsplit("ep", 1)[1])))]
            with (campaign / "train.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                state.update({"phase": "training", "child_pid": child.pid, "command": args})
                write_json(campaign / "state.json", state)
                code = child.wait()
            if code:
                raise RuntimeError(f"Training exited with code {code}; see train.log")
            checkpoints = list((campaign / "runs").glob("Training Run */Pytorch Models/*_ep3000.pt"))
            if len(checkpoints) != 1:
                raise ValueError("Expected exactly one final 3000-epoch training checkpoint")
            state.update({"phase": "complete", "checkpoint": str(checkpoints[0]), "checkpoint_sha256": digest(checkpoints[0])})
        except BaseException as error:
            state.update({"phase": "stopped", "error": str(error)})
            raise
        finally:
            state["updated"] = time.time()
            write_json(campaign / "state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume only this same L1/3000-epoch experiment")
    args = parser.parse_args()
    if args.run:
        run(CAMPAIGN, args.resume)
    else:
        print(json.dumps({"campaign": str(CAMPAIGN), "command": command(CAMPAIGN, CAMPAIGN / "source_snapshot")}, indent=2))


if __name__ == "__main__":
    main()
