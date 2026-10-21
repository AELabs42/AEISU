# AEISU desktop demo

A minimal Windows desktop demo derived from AESS. The full AESS application is not bundled.

The Python UI contains a source-window selector, Refresh, a Nearest/Bilinear
comparison-filter selector, Start/Stop, and an optional FPS counter. There is one model: the paper's folded epoch-1000 AEISU **mixed
FP8/FP16 Tensor Core** artifact. No frame generation, alternate upscaling models,
precision selector, sharpening control, or effect stack is exposed or packaged.

Python controls the session only. GPU-resident capture, CUDA/D3D interop, inference,
SDR preprocessing, frame pacing and presentation remain native; frames never pass
through Python. Windows Graphics Capture is the default, requesting compositor
BGRA8 SDR pixels; Desktop Duplication is its fallback. The inherited injected
capture implementation remains in the source but is not used by the launcher.
Raw 10-bit storage alone does not establish HDR transfer metadata. Injection may be prohibited by a
game's anti-cheat or protection system; do not bypass those protections.

## Source release

This public snapshot contains the demo source, not prebuilt executables or vendor runtime binaries.
Build the native components using the documented local SDKs, then run `python app.py`.
The model is the mixed FP8/FP16 artifact included under `native/models/aeisu_mixed/`.
The demo targets Windows SDR and RTX 50-series hardware; it is tested on RTX 5090.
Disable game HDR, Windows HDR and Auto HDR. Its captured-input scaling is exactly 2x.
`Ctrl+Shift+Tab` toggles between neural reconstruction and the selected Nearest/Bilinear comparison.
`Ctrl+Tab` stops capture; `Ctrl+F` toggles the FPS counter.

## Build locally

No Python packages are needed: the launcher uses `tkinter` and `ctypes` from the
standard library. Install a normal Windows Python distribution with Tcl/Tk enabled.
No `uv add` command is required.

```powershell
python build.py --cutlass CUTLASS_ROOT --frontend FRONTEND_ROOT --cudnn-lib CUDNN_LIB_DIRECTORY
python app.py
```

The optimized backend requires Windows, Visual Studio x64 C++ tools, Windows SDK
10.0.26100.0, CUDA 13.x, cuDNN 9.25, CUTLASS and cuDNN Frontend. It currently targets
Blackwell SM 120a: **RTX 50-series only**, validated locally on RTX 5090. RTX
40-series support is not included, even though Ada has FP8-capable hardware.
The app does not depend on Torch;
the separate reference training/inference package uses Torch with CUDA 13.0.

CUDA/cuDNN discovery uses local SDK installations. CUTLASS and cuDNN Frontend
are external dependencies: provide their paths explicitly. The cuDNN import
library must match the installed cuDNN DLLs. To override all SDK paths:

```powershell
python build.py --cuda-root CUDA_ROOT --cudnn-root CUDNN_ROOT --cudnn-lib CUDNN_LIB_DIRECTORY --cutlass CUTLASS_ROOT --frontend FRONTEND_ROOT
```

Nothing is downloaded automatically. Large SDK DLLs are not duplicated; a generated,
ignored `native/bin/runtime_paths.json` records local runtime paths for the launcher.
After a full build, repeat the build command with `--app-only` to skip unchanged CUDA compilation.

## Layout

- `demo/`: Python UI, window enumeration and session lifecycle.
- `native/app/`: stripped AESS capture, render and session source sections.
- `native/src/`: retained capture/pacing helpers and the optimized model runtime.
- `native/models/aeisu_mixed/`: the single deployment artifact.
- `clone_provenance.json`: original app and model identities.

This is a local demo candidate, not a claim of compatibility with every game or
GPU. Do not apply the repository-wide license to third-party dependencies without
retaining their notices; ImGui and MinHook retain their bundled licenses.

## Local verification

On 7 October 2026, the native backend and capture DLL compiled successfully, all
ten CPU/unit checks passed (including native SDR/HDR rejection), and the launcher was rendered and inspected without
clipped controls. The retained CUDA model source matches the released optimized
runtime byte-for-byte; the original AESS main-source hash remains unchanged.

An animated, generated WGC test window ran the parallel32x4 model with the mixed
FP8/FP16 CUDA graph path on RTX 5090 and exited cleanly through the named stop
event. That automated fixture used the G-SYNC-disabled diagnostic route because
it could not acquire normal foreground ownership. Injected game capture, input
ownership and normal G-SYNC operation still require a manual game-window check.
This was a functional smoke test of the initial clone, not a performance or quality
benchmark. After the SDR-only revision, the live smoke test correctly refused to
start because Windows HDR was enabled on the fixture's display. The new SDR color
policy tests and build passed. A subsequent revision made compositor BGRA8 WGC
capture the default, avoiding the injected hook's uncertain 10-bit/HDR inference.
With Windows HDR disabled, the SDR fixture presented 309 frames at 960x540 to
1920x1080 and stopped gracefully with exit code 0. All eleven built-backend checks
passed. This fixture disables G-SYNC compatibility; real-game input ownership and
normal G-SYNC still need manual checking. The application does not change Windows HDR settings.
The restored bypass passed a subsequent AEISU → nearest-neighbor → AEISU fixture
test, with frames presented in both modes and graceful shutdown. All twelve
built-backend checks passed. The fixture posts the registered hotkey message to
exercise the handler; physical keyboard delivery remains a manual game check.

```powershell
python -m unittest discover -s tests -v
python -m tests.smoke_ui
python -m tests.smoke_native
python -m tests.smoke_native --toggle-bypass
```

In a fresh source-only checkout, the sixteen-check suite runs without the native build; the
native SDR policy check is skipped until the backend has been built. The UI and
native smoke tests are separate, opt-in interactive checks.

The optional UI inspection test uses the locally available Pillow package for a
test-only screenshot; the application itself does not need Pillow. If installing
that optional test dependency in a new environment, use `uv add pillow`.

## Packaging

The launcher offers Nearest and Bilinear comparison filters for Ctrl+Shift+Tab.
Choose before starting; the filter is fixed during capture. Both comparison modes
resize the original SDR capture directly and bypass the neural model.
Version 0.1.2 passed sixteen built-backend/unit checks, the packaged self-test,
and AEISU/Nearest/AEISU and AEISU/Bilinear/AEISU GPU fixtures. Both stop routes hid
the output in approximately 16 ms in those generated, poll-observed fixtures.

Version 0.1.1 hides the output on stop before GPU/capture teardown, retaining the
hidden owner window until resources drain. Fourteen built-backend/unit checks and
the packaged self-test passed. Generated WGC fixtures verified both hotkey and
named-event stop paths, observing hidden output in approximately 16 ms; those
poll-based checks are not real-game latency measurements. AEISU/nearest/AEISU
presentation and clean backend exit also passed with the installed SDK/Python
paths removed from the packaged process environment. Model arithmetic is unchanged.

`tools/package_portable.py` embeds the existing standard-library UI in an isolated
CPython/Tk Windows host and copies only installed runtime dependencies. It does
not install or download anything. Build the native backend first, then provide
the matching local SDK/runtime roots:

```powershell
python tools/package_portable.py --cuda-root CUDA_ROOT --cudnn-root CUDNN_ROOT --msvc-redist MSVC_X64_CRT_DIRECTORY
```

The resulting `dist/` folder and ZIP are ignored by Git. Release assets include
upstream runtime license notices, a file-hash manifest and an archive SHA-256.
`tools/PORTABLE_README.md` documents the end-user instructions and diagnostics.
The earlier frozen private preview archives predate the explicit CUTLASS and cuDNN Frontend
notices. The separate `AEISU-Demo-0.1.2-Notices.zip` asset on the private release adds these and the updated
runtime index: extract into the existing demo folder. It changes no executables,
weights or DLLs. Do not silently overwrite the original versioned archives or checksums.
The current complete With-Notices ZIP already contains these license files and the updated runtime index.
