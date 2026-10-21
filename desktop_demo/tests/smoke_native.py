"""Run a few captured frames from our own generated window, then stop gracefully."""
from __future__ import annotations

import ctypes
import csv
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tkinter as tk

from demo.session import BACKEND, ROOT, backend_environment
from demo.windows import StopEvent, USER32

root = tk.Tk()
root.title("AEISU synthetic capture fixture")
root.geometry("640x360+180+180")
canvas = tk.Canvas(root, background="#223344", highlightthickness=0)
canvas.pack(fill="both", expand=True)
canvas.create_text(320, 70, text="AEISU CAPTURE TEST", fill="white", font=("Segoe UI", 24))
canvas.create_rectangle(50, 140, 220, 260, fill="#dda040", outline="white", width=3)
shape = canvas.create_oval(330, 135, 470, 275, fill="#50b0a0", outline="white", width=3)
root.update()
USER32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
USER32.GetAncestor.restype = wintypes.HWND
handle = USER32.GetAncestor(root.winfo_id(), 2)
USER32.SetForegroundWindow(handle)
stop = StopEvent(f"Local\\AEISUDemoSmoke_{os.getpid()}")
environment = backend_environment()
output = ROOT / "outputs"
output.mkdir(exist_ok=True)
log = (output / "native_smoke.log").open("wb")
comparison_filter = "bilinear" if "--bilinear" in sys.argv else "nearest"
process = subprocess.Popen([str(BACKEND), "--hwnd", hex(handle), "--capture-wgc", "--no-gsync-compat", "--fps-counter",
                            "--perf-log", str(output / "native_smoke.csv"), "--stop-event", stop.name,
                            "--comparison-filter", comparison_filter],
                           cwd=BACKEND.parent, stdout=log, stderr=subprocess.STDOUT, env=environment,
                           creationflags=subprocess.CREATE_NO_WINDOW)
started = time.monotonic()
requested = False
stop_started = None
stop_hidden_seconds = None
stop_output = None
hotkey_stop_test = "--stop-hotkey" in sys.argv
toggle_test = "--toggle-bypass" in sys.argv
toggle_count = 0
USER32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
USER32.FindWindowW.restype = wintypes.HWND
USER32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
USER32.PostMessageW.restype = wintypes.BOOL
USER32.IsWindowVisible.argtypes = [wintypes.HWND]
USER32.IsWindowVisible.restype = wintypes.BOOL


def tick():
    global requested, toggle_count, stop_started, stop_hidden_seconds, stop_output
    elapsed = time.monotonic() - started
    canvas.move(shape, 1 if int(elapsed * 2) % 2 else -1, 0)
    if toggle_test and toggle_count < 2 and elapsed > (6 if toggle_count == 0 else 12):
        output_handle = USER32.FindWindowW("AEISUDemo.OutputWindow", None)
        if output_handle and USER32.PostMessageW(output_handle, 0x0312, 101, (0x09 << 16) | 0x06):
            toggle_count += 1
    if requested and stop_hidden_seconds is None and not USER32.IsWindowVisible(stop_output):
        stop_hidden_seconds = time.monotonic() - stop_started
    if process.poll() is not None:
        log.close()
        stop.close()
        root.destroy()
        return
    if elapsed > 20 and not requested:
        stop_output = USER32.FindWindowW("AEISUDemo.OutputWindow", None)
        assert stop_output and USER32.IsWindowVisible(stop_output), "Output must be visible before stop"
        stop_started = time.monotonic()
        if hotkey_stop_test:
            assert USER32.PostMessageW(stop_output, 0x0312, 100, (0x09 << 16) | 0x02)
        else:
            stop.signal()
        requested = True
    if elapsed > 45:
        # Test-created process only, and only after graceful stop timed out.
        process.terminate()
        process.wait(timeout=10)
        log.close()
        stop.close()
        root.destroy()
        return
    root.after(10 if requested else 33, tick)


root.after(33, tick)
root.mainloop()
text = (output / "native_smoke.log").read_bytes().decode("utf-8", errors="replace")
print(text[-9000:])
print(f"EXIT_CODE={process.returncode}; GRACEFUL_STOP_REQUESTED={requested}")
if process.returncode:
    raise SystemExit(process.returncode)
assert requested and stop_hidden_seconds is not None and stop_hidden_seconds < 0.5, "Output was not hidden promptly"
stop_record = {"route": "hotkey handler" if hotkey_stop_test else "named stop event",
               "comparison_filter": comparison_filter,
               "hidden_seconds": stop_hidden_seconds, "process_exit_seconds": time.monotonic() - stop_started,
               "exit_code": process.returncode, "limit_seconds": 0.5,
               "scope": "Generated WGC fixture; not a physical-keyboard or real-game latency benchmark"}
(output / f"stop_visibility_{comparison_filter}.json").write_text(json.dumps(stop_record, indent=2), encoding="utf-8")
print("STOP_VISIBILITY=" + json.dumps(stop_record))
if toggle_test:
    assert toggle_count == 2, "Both toggle messages must be delivered"
    comparison_label = "bilinear" if comparison_filter == "bilinear" else "nearest-neighbor"
    assert f"{comparison_label} (AEISU off)" in text and "Upscaler toggle: AEISU on" in text
    with (output / "native_smoke.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    comparison_frames = [int(row["frame"]) for row in rows if row["upscaler"] == comparison_filter]
    neural_frames = [int(row["frame"]) for row in rows if row["upscaler"] == "aeisu"]
    assert comparison_frames, "Selected comparison must actually present frames"
    assert min(neural_frames) < min(comparison_frames) < max(neural_frames), "AEISU must resume after bypass"
    print(f"TOGGLE_CHECK=AEISU -> {comparison_label} -> AEISU; all routes presented")
records = list(output.glob("native_smoke*"))
print("SMOKE_RECORDS=" + ", ".join(path.name for path in records))
