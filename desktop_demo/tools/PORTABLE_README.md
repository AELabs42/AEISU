# AEISU portable desktop demo

Extract the complete ZIP to a writable folder, then double-click **AEISU Demo.exe**.
Keep its DLLs and subfolders together. No installer, administrator privileges,
Python installation, Visual Studio or CUDA Toolkit installation is required.

Requirements: Windows 10/11 x64, a current compatible NVIDIA driver, and an
RTX 50-series GPU. Tested on RTX 5090 only; RTX 40-series is not supported.
The model is mixed FP8/FP16 with exact 2x reconstruction. Use an already
anti-aliased SDR game input, with Windows HDR, game HDR and Auto HDR disabled.
The app does not tone-map HDR or perform active anti-aliasing.

Select a windowed/borderless game, choose the comparison filter (Nearest or Bilinear),
optionally enable the FPS counter, and start. The comparison choice applies to the next session.

- Ctrl+Shift+Tab: toggle AEISU / selected comparison filter.
- Ctrl+Tab: hide the upscaled output immediately and stop scaling safely.
- Ctrl+F: toggle the FPS counter.

Capture uses Windows Graphics Capture with BGRA8 SDR output. The model always
reconstructs at 2x; a display-fitting pass is used when necessary. Frame generation
and alternate neural upscalers are not included.

Diagnostics are saved beside the app in `outputs/`. This unsigned preview may
trigger Windows SmartScreen. Check the release SHA-256 and source before deciding
whether to run it; do not disable security protection globally.

Optional diagnostics from a terminal:

    "AEISU Demo.exe" --self-test
    "AEISU Demo.exe" --smoke-test

The first checks the bundled Python/Tk, model identity and native SDR policy without
GPU inference. The second opens an animated synthetic test window, uses the GPU,
tests both toggle routes and exits. It uses the G-SYNC-disabled diagnostic route;
real-game input focus and normal G-SYNC still require manual testing.

Stopping hides the output before GPU/capture teardown finishes.
The launcher returns once cleanup is complete; the backend is not force-killed.

This folder contains only the application's runtime dependencies, not developer
SDKs. The NVIDIA display driver is not bundled. See `THIRD_PARTY_RUNTIME.md` and
`licenses/` for dependency terms; the author's MIT license does not replace them.

Source and releases: https://github.com/AELabs42/AEISU
