# Runtime licenses

AEISU's own code and model are MIT-licensed; see LICENSE. Bundled third-party
binaries remain subject to their upstream terms and are supplied solely for use
as part of this application, not as a separate SDK. This application is not
sponsored or endorsed by NVIDIA, Microsoft, Python or other dependency authors.

- CPython 3.13 and its standard-library dependencies: `licenses/Python-LICENSE.txt`.
  This includes the supplied libffi, Tcl/Tk and CPython Windows-runtime notices.
- Tcl/Tk 8.6: notices within the Tcl scripts and `licenses/Tk-license.terms`, plus the Python distribution's notices.
- Dear ImGui: `licenses/imgui-LICENSE.txt` (MIT).
- MinHook/Hacker Disassembler Engine: `licenses/minhook-LICENSE.txt` (BSD-style terms).
- CUTLASS C++ headers (recorded build: 4.5.3): `licenses/CUTLASS-LICENSE.txt` (BSD-3-Clause).
  The CuTe DSL Python implementation is not bundled.
- cuDNN Frontend C++ headers (recorded build: 1.17.0): `licenses/cuDNN-Frontend-LICENSE.txt` (MIT).
  These two header dependencies contribute compiled code even though their SDK trees are not bundled.
- JSON for Modern C++ 3.11.3, included through cuDNN Frontend:
  `licenses/nlohmann-json-LICENSE.txt` (MIT). Its header records copyright
  2013-2023 Niels Lohmann; the supplied upstream license file is preserved unchanged.
- CUDA runtime, cuBLAS/cuBLASLt, NVRTC and nvJitLink: `licenses/CUDA-EULA.txt` and `licenses/CUDA-LICENSE.txt`.
- cuDNN runtime: the original wheel notice is retained in `licenses/cuDNN-LICENSE.txt`.
  Current cuDNN SDK distribution terms, including runtime DLLs, are at
  [NVIDIA's cuDNN license page](https://docs.nvidia.com/cudnn/reference/eula/).
  The supplied wheel notice contains older cuDNN 7-era wording; it is preserved
  as delivered, not presented as a complete current-version license record.
  Confirm the applicable cuDNN version's terms before public binary distribution.
- Microsoft Visual C++ x64 runtime: application-local DLLs from the matching
  Visual Studio redistribution directory, under the applicable Microsoft terms:
  [Microsoft's redistribution guidance](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files).
  These DLLs may be used only under the applicable upstream terms; their presence
  does not grant a Visual Studio development license. Application-local copies
  require servicing through future application updates.

NVIDIA SDKs are licensed for NVIDIA GPU applications and retain restrictions on
their use, redistribution and modification. Do not apply AEISU's MIT license to
these files. Windows OS and NVIDIA display-driver components are not redistributed.

This software contains source code provided by NVIDIA Corporation.
The bundled proprietary NVIDIA runtime files are supplied only with AEISU for
use by this application, not for extraction as a standalone SDK. The upstream
terms control those files, including their use and redistribution restrictions.
AEISU's MIT terms continue to apply independently to the author's own code and weights.

## Notice correction for the private preview

The frozen 0.1.0 package and 0.1.2 update predate the explicit CUTLASS and cuDNN
Frontend (including its JSON dependency) notice files. A separate `AEISU-Demo-0.1.2-Notices.zip` is prepared locally
to supply those notices and this index without changing executables or runtime DLLs.
Extract it into the existing demo folder, retaining the original upstream notices.
Future full packages include the additions automatically. Previously distributed
archives and their checksums have not been silently replaced.
