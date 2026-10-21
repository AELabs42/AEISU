# Dependency and asset inventory

The portable model scripts require PyTorch, NumPy and Pillow; LPIPS is optional.
Install these through their normal package distribution channels. Their licenses remain their own.

Native deployment compilation additionally requires the Windows SDK, MSVC, CUDA, cuBLASLt, cuDNN,
CUTLASS headers and cuDNN Frontend headers. Vendor libraries/headers are not bundled or downloaded automatically.
The recorded header versions are CUTLASS 4.5.3 and cuDNN Frontend 1.17.0.
Consult the vendor license notices in the separately obtained dependency distributions; no ownership of vendor code is claimed.

The desktop demo bundles the source subset of Dear ImGui under its MIT license
(`desktop_demo/native/third_party/imgui/LICENSE.txt`, copyright Omar Cornut),
and MinHook under its BSD-style license, including the Hacker Disassembler Engine notices
(`desktop_demo/native/third_party/minhook/LICENSE.txt`). These upstream notices remain intact;
the repository's MIT license does not replace them. CUDA, cuDNN, CUTLASS and cuDNN Frontend
remain external SDK dependencies in the source checkout. The portable demo
release ZIP separately includes the runtime DLLs needed by the application,
under their upstream terms, along with CPython/Tk and application-local MSVC
runtime DLLs. It does not include developer SDKs or the NVIDIA display driver.
See desktop_demo/tools/THIRD_PARTY_RUNTIME.md and the release ZIP's licenses/ folder.
The compiled CUTLASS and cuDNN Frontend header code also requires its upstream
notices. These are retained in desktop_demo/tools/licenses/ and copied into
future portable packages. The frozen private previews require the separate
notice correction; SDK source trees are still not bundled.

Full QRISP frames and the private training capture corpus are not included. Figure crops already present in the manuscript
remain attributed to their benchmark context. Game and benchmark image content, including crops and reconstructions,
is excluded from the MIT and CC BY 4.0 grants. Review the applicable redistribution rights before public release;
the exclusion does not establish permission to redistribute those images.

Comparator records identify released checkpoints and pinned source hashes, but do not redistribute the checkpoints.
Archived generator/analysis files retain attribution and source identity. They may require separately obtained
benchmark images, model assets or optional analysis dependencies to reproduce the complete experiment.
