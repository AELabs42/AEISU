"""Build only the isolated Windows resident-image benchmark; never the desktop app."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parent


def developer_environment():
    if os.name != "nt":
        raise RuntimeError("The native resident-resource harness requires Windows")
    environment = os.environ.copy()
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    installation = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    if not installation:
        raise RuntimeError("MSVC x64 C++ build tools were not found")
    batch = Path(installation) / "VC/Auxiliary/Build/vcvars64.bat"
    result = subprocess.run(f'cmd.exe /d /s /c "call "{batch}" >nul && set"',
                            check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            environment[key] = value
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--cuda-compiler-root", type=Path)
    parser.add_argument("--cudnn-include", type=Path, required=True)
    parser.add_argument("--cudnn-lib", type=Path, required=True)
    parser.add_argument("--cudnn-bin", type=Path, required=True)
    parser.add_argument("--cutlass", type=Path, required=True, help="CUTLASS repository root")
    parser.add_argument("--frontend", type=Path, required=True, help="cuDNN Frontend repository root")
    args = parser.parse_args()
    compiler_root = args.cuda_compiler_root or args.cuda_root
    nvcc = compiler_root / "bin/nvcc.exe"
    required = [nvcc, args.cuda_root / "include/cuda_runtime.h", args.cudnn_include / "cudnn.h",
                args.cudnn_lib / "cudnn.lib", args.cutlass / "include/cutlass/cutlass.h",
                args.frontend / "include/cudnn_frontend.h"]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Missing dependency files: {[str(p) for p in required if not p.is_file()]}")
    environment = developer_environment()
    compiler = shutil.which("cl.exe", path=environment.get("Path", environment.get("PATH")))
    if compiler is None:
        raise FileNotFoundError("MSVC environment did not expose cl.exe")
    folder = ROOT / "build"
    folder.mkdir(exist_ok=True)
    includes = [compiler_root / "include", args.cuda_root / "include", args.cutlass / "include",
                args.cutlass / "tools/util/include", ROOT / "src/neural", ROOT / "src/neural/srcpyr"]
    common = ["/nologo", "/EHsc", "/std:c++20", "/O2", "/MD", "/DNDEBUG", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN"]
    commands = [[str(nvcc), "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a", "-allow-unsupported-compiler",
                 "-diag-suppress", "177,20012,20013,20015,20054", "-O3", "-lineinfo", "-DNDEBUG",
                 "-Xcompiler=/EHsc /Zc:preprocessor /DNOMINMAX /DWIN32_LEAN_AND_MEAN", "-Xcompiler=/O2 /MD",
                 *[f"-I{p}" for p in includes], "-c", str(ROOT / "src/neural/srcpyr/SrcPyrCudaRuntime.cu"),
                 "-o", str(folder / "runtime.obj")]]
    for stem in ("SrcPyrModel", "SrcPyrCudnnConv"):
        commands.append([compiler, *common, "/c", *[f"/I{p}" for p in includes],
                         f"/external:I{args.frontend / 'include'}", f"/external:I{args.cudnn_include}", "/external:W0",
                         str(ROOT / f"src/neural/srcpyr/{stem}.cpp"), f"/Fo{folder / (stem + '.obj')}"])
    commands.append([compiler, *common, *[f"/I{p}" for p in includes],
                     str(ROOT / "src/neural/srcpyr/SrcPyrRuntimeTest.cpp"), str(folder / "runtime.obj"),
                     str(folder / "SrcPyrModel.obj"), str(folder / "SrcPyrCudnnConv.obj"),
                     f"/Fe:{folder / 'aeisu_native_bench.exe'}", "/link", f"/LIBPATH:{args.cuda_root / 'lib/x64'}",
                     f"/LIBPATH:{args.cudnn_lib}", "d3d11.lib", "d3d12.lib", "dxgi.lib", "cublasLt.lib",
                     "cudnn.lib", "cudart.lib", "bcrypt.lib"])
    for command in commands:
        subprocess.run(command, cwd=folder, env=environment, check=True)
    # Local-only configuration: do not ship machine-specific dependency locations.
    (folder / "runtime_paths.json").write_text(json.dumps({"cuda_bin": str(args.cuda_root / "bin"),
                                                        "cudnn_bin": str(args.cudnn_bin)}, indent=2), encoding="utf-8")
    print("Built isolated benchmark; no AESS application files were built or modified.")


if __name__ == "__main__":
    main()
