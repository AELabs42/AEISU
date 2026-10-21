"""Manage one native AEISU session without moving image data through Python."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

from .windows import StopEvent, USER32

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "native/bin/aeisu_demo_backend.exe"
MODEL = ROOT / "native/models/aeisu_mixed/adaptive_fp8.bin"


def backend_environment():
    environment = os.environ.copy()
    paths = json.loads((BACKEND.parent / "runtime_paths.json").read_text(encoding="utf-8"))
    if paths.get("portable"):
        # The portable backend has application-local runtime DLLs, not absolute SDK paths.
        runtime_paths = [str(BACKEND.parent)]
    else:
        runtime_paths = [paths["cuda_bin"], paths["cudnn_bin"]]
    environment["PATH"] = os.pathsep.join([*runtime_paths, environment.get("PATH", "")])
    return environment


class Session:
    def __init__(self):
        self.process = None
        self.stop_event = None
        self.log = None
        self.log_path = ROOT / "outputs/session.log"

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, handle, fps=False, comparison_filter="nearest"):
        if comparison_filter not in ("nearest", "bilinear"):
            raise ValueError("Comparison filter must be nearest or bilinear.")
        if self.running:
            raise RuntimeError("A scaling session is already running.")
        self.close_handles()
        if not BACKEND.is_file():
            raise FileNotFoundError("Build the native backend first; see the SDK paths and build command in README.md.")
        if not USER32.IsWindow(handle):
            raise ValueError("The selected window no longer exists. Refresh the window list.")
        expected = json.loads((ROOT / "clone_provenance.json").read_text(encoding="utf-8"))["model_sha256"]
        if hashlib.sha256(MODEL.read_bytes()).hexdigest() != expected:
            raise ValueError("The packaged model does not match the selected AEISU mixed artifact.")
        self.stop_event = StopEvent(f"Local\\AEISUDemoStop_{uuid.uuid4().hex}")
        self.log_path.parent.mkdir(exist_ok=True)
        self.log = self.log_path.open("wb")
        command = [str(BACKEND), "--hwnd", hex(handle), "--stop-event", self.stop_event.name,
                   "--comparison-filter", comparison_filter]
        if fps:
            command.append("--fps-counter")
        try:
            USER32.SetForegroundWindow(handle)
            environment = backend_environment()
            self.process = subprocess.Popen(command, cwd=BACKEND.parent, stdout=self.log, stderr=subprocess.STDOUT,
                                            creationflags=subprocess.CREATE_NO_WINDOW, env=environment)
            USER32.AllowSetForegroundWindow(self.process.pid)
        except Exception:
            self.close_handles()
            raise

    def stop(self):
        if self.running and self.stop_event:
            self.stop_event.signal()

    def close_handles(self):
        if self.running:
            return
        if self.stop_event:
            self.stop_event.close()
            self.stop_event = None
        if self.log:
            self.log.close()
            self.log = None

    def error_tail(self):
        if not self.log_path.is_file():
            return "No diagnostic log was produced."
        return self.log_path.read_bytes()[-5000:].decode("utf-8", errors="replace").strip()
