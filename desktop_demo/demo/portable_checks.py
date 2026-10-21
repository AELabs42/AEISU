"""Opt-in packaged runtime checks; normal launch never invokes these."""
from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
import tkinter as tk

from .session import BACKEND, MODEL, ROOT, backend_environment
from .ui import Launcher


def run(mode):
    if mode == "--smoke-test":
        for options in (("--stop-hotkey",), ("--bilinear",)):
            sys.argv = ["portable_capture_check", "--toggle-bypass", *options]
            runpy.run_path(str(ROOT / "_capture_check.py"), run_name="__main__")
        return
    root = tk.Tk()
    root.withdraw()
    launcher = Launcher(root)
    root.update()
    assert launcher.button.cget("text") == "Start scaling"
    assert launcher.comparison_filter.get() == "Nearest"
    assert tuple(launcher.comparison_box.cget("values")) == ("Nearest", "Bilinear")
    assert "Windows HDR" not in " ".join(str(child.cget("text")) for child in launcher.button.master.master.winfo_children()
                                          if "text" in child.keys())
    root.destroy()
    assert sys.flags.isolated and sys.flags.no_site == 1
    for path in sys.path:
        assert str(ROOT).lower() in str(path).lower(), f"External Python path: {path}"
    record = json.loads((ROOT / "clone_provenance.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == record["model_sha256"]
    result = subprocess.run([str(BACKEND), "--self-test-sdr"], env=backend_environment(),
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    stop_result = subprocess.run([str(BACKEND), "--self-test-stop"], env=backend_environment(),
                                 capture_output=True, text=True, timeout=15)
    assert stop_result.returncode == 0, stop_result.stdout + stop_result.stderr
    print("PORTABLE_SELF_TEST=PASS; isolated CPython/Tk UI, model identity, native SDR policy")
    print(result.stdout)
    print(stop_result.stdout)
