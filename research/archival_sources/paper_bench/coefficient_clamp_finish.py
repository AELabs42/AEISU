"""Finish an already running clamp benchmark with independent verification."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def wait_for_process(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        # The evaluation may have finished before this companion starts.
        if ctypes.get_last_error() == 87:
            return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        while True:
            result = kernel.WaitForSingleObject(handle, 60000)
            if result == 0:
                return
            if result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    if args.wait_pid:
        print(f"Waiting for evaluation process {args.wait_pid}", flush=True)
        wait_for_process(args.wait_pid)
    status = json.loads((args.output / "status.json").read_text())
    if status["phase"] != "evaluation_complete" or status["completed"] != status["total"]:
        raise RuntimeError("Evaluation did not finish; keep partial results for resume")
    for name in ("control_ablation_verify.py", "coefficient_clamp_report.py"):
        print(f"Running {name}", flush=True)
        subprocess.run([sys.executable, str(HERE / name), "--output", str(args.output)],
                       cwd=HERE.parents[1], check=True)
    print("Verified report ready. Manuscript integration awaits review of these complete findings.", flush=True)


if __name__ == "__main__":
    main()
