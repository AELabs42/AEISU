## Portable Windows preview

### Version 0.1.2

#### Download and upgrade

For a fresh installation, download **AEISU-Demo-0.1.2-Windows-x64-With-Notices.zip**
from this release, verify its accompanying SHA-256, and extract it to a writable folder.
Launch **AEISU Demo.exe** inside the extracted folder. All runtime files and compiled-header
license notices are included; no separate notice overlay or older package is needed.
The complete archive is approximately 895 MB, mostly NVIDIA runtime libraries.

Existing 0.1.0 users can instead apply the **1.3 MB update ZIP** and the separate notice overlay.
Close AEISU, then extract the update ZIP directly into that existing demo folder,
replacing matching files. Keep the existing runtime DLLs and `python/` folder.
Launch `AEISU Demo.exe` from the updated folder. Do not run the update files alone.

Applying this update to 0.1.0 reproduced every file in the tested 0.1.2 package
exactly, and the relocated updated folder passed its isolated self-test.
The complete package with integrated notices was uploaded and its GitHub SHA-256 verified on 13 October 2026.
Its executable, backend, model and runtime binary hashes match the tested 0.1.2 package;
Only the dependency notice index and three compiled-header license files changed,
and the package inventory was regenerated.
The update uses the unchanged runtime payload, with no dependency downloads or installs.

The separate `AEISU-Demo-0.1.2-Notices.zip` release asset supplies the explicit CUTLASS,
cuDNN Frontend and JSON dependency notices omitted from the frozen preview.
Existing installations upgraded from 0.1.0 should extract it into the same demo folder.
The complete With-Notices ZIP already includes these notices. The overlay contains only notices and instructions,
not replacement executables, weights or DLLs. Its SHA-256 is supplied alongside it.

Choose Nearest or Bilinear in the launcher before starting. Ctrl+Shift+Tab toggles
AEISU against that comparison filter; both comparison routes bypass neural
inference and resize the captured SDR source directly. Nearest remains the default.
The immediate-hide stop behavior from 0.1.1 is retained.

### Version 0.1.1

Ctrl+Tab now hides the output immediately instead of leaving a frozen frame
visible while GPU/capture cleanup finishes. The hidden window stays alive until
resources drain safely; stop does not force-terminate the backend. Mouse-button
forwarding is disabled while the output is hidden. The launcher checks session
completion every 50 ms instead of 250 ms. Model arithmetic and precision are unchanged.

Version 0.1.1 was a local intermediate build, not a separately downloadable release.
Use the 0.1.2 download/upgrade route above, then run **AEISU Demo.exe**.
The executable is accompanied by application-local Python/Tk, Microsoft C++ and
NVIDIA runtime files. No Python installation, Visual Studio, CUDA Toolkit or
administrator privileges are needed. Keep the folder contents together.

- Windows 10/11 x64; RTX 50-series only; tested on RTX 5090.
- A compatible current NVIDIA driver is required. RTX 40-series is unsupported.
- SDR only: disable game HDR, Windows HDR and Auto HDR. Use an already anti-aliased input.
- One optimized mixed FP8/FP16 model, exact 2x reconstruction, no frame generation.
- Ctrl+Shift+Tab switches between AEISU and the selected Nearest/Bilinear filter without ending capture.
- Ctrl+Tab stops; Ctrl+F toggles the FPS counter.

This unsigned preview may trigger SmartScreen. A SHA-256 file is provided alongside
the ZIP; review the source and checksum before choosing to run it. Do not disable
security protection globally.

The bundled-runtime self-test and GPU fixture pass with installed Python/CUDA
paths removed from the process environment, including the AEISU → nearest → AEISU
toggle. Required cuDNN, cuBLASLt and MSVC libraries were observed loading from the
package directory. This is not a clean-machine or other-GPU compatibility test.
The fixture uses the G-SYNC-disabled route. The author reported successful
real-game use of version 0.1.0; normal G-SYNC is not independently verified here.
Version 0.1.1's stop-visibility change has a separate diagnostic check.
Runtime license notices and a file-hash manifest are in
the ZIP. The NVIDIA driver is not bundled.

The large download is mostly NVIDIA runtime libraries, not model weights.
