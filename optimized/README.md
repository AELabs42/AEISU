# Isolated optimized deployments

These are inference/benchmark components, not the desktop capture application. The portable scripts at the repository root
remain the default CPU/PyTorch interface. Optimized timings are implementation-specific resident-image measurements;
they exclude compilation, setup, transfers and image encoding.

## Tensor Core deployment

native/ contains the quoted-include source closure of the measured Windows D3D11/CUDA harness and its mixed FP8/FP16 artifact.
The original code is retained byte-for-byte, including legacy internal names. The included harness also supports its existing
precision controls; the small Python launcher defaults to the principal mixed artifact.

Build in a Windows MSVC x64 environment with externally installed CUDA, cuDNN, CUTLASS and cuDNN Frontend:

    python optimized/native/build_native.py --cuda-root CUDA_ROOT --cuda-compiler-root COMPILER_ROOT --cudnn-include CUDNN_INCLUDE --cudnn-lib CUDNN_LIB --cudnn-bin CUDNN_BIN --cutlass CUTLASS_ROOT --frontend FRONTEND_ROOT
    python optimized/native/benchmark_native.py --input input.png --output outputs/native --save-output

The validated build targets SM 120a on RTX 5090. The local build check uses CUDA compiler 13.3, cuDNN 9.25,
CUTLASS 4.5.3 and Frontend 1.17.0. The reference Python package uses PyTorch's CUDA 13.0 build.
No SDK binaries, vendor repositories or automatic downloads are included.
build/runtime_paths.json is local dependency configuration and must not be committed.

## Conventional-CUDA mixed INT8/FP16 deployment

shader/ contains the exact fused conventional-CUDA kernel sources and the exported profile-selected precision artifact:
FP16 project/refine/head, INT8 compression/parallel bottleneck/expansion, FP32 reconstruction.
Static weight preparation is outside inference. Run with PyTorch 2.11.0+cu130, ninja, MSVC and a matching CUDA toolkit:

If ninja is not already installed in the environment, add it with uv add ninja before the first local compilation.

    python optimized/shader/benchmark_shader.py --input input.png --output outputs/shader

The first run locally compiles an extension; it does not download dependencies. CUDA graph replay is compared to eager custom-kernel
execution before timing. The archived SASS audit records zero Tensor Core instructions for the measured conventional-CUDA binary.
A new build should be audited separately; source identity alone is not an instruction audit.
The staged build was audited too: zero Tensor Core instructions were found, and its RGB output matched the original
conventional-CUDA implementation exactly on the synthetic smoke input. All three native precision outputs also matched
the original harness exactly on that input. These small correctness probes are not a new quality benchmark.

The new standalone runners are functional release interfaces. Paper measurements retain their original executable/source identities.
Support for other operating systems or GPUs is not established by the local RTX 5090 checks.
The author's optimized runtime sources and released deployment weights are MIT-licensed.
External dependencies and any third-party content retain their original terms; see LICENSE_SCOPE.md and THIRD_PARTY.md.
