"""Build the isolated native backend using locally installed SDKs; never download."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / "native"


def developer_environment():
    environment = os.environ.copy()
    locator = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    installation = subprocess.check_output([str(locator), "-latest", "-products", "*", "-requires",
                                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    if not installation:
        raise RuntimeError("Install Visual Studio C++ build tools and a Windows SDK.")
    batch = Path(installation) / "VC/Auxiliary/Build/vcvars64.bat"
    result = subprocess.run(f'cmd.exe /d /s /c "call "{batch}" >nul && set"', check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            environment[key] = value
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-root", type=Path, default=Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3")))
    parser.add_argument("--cudnn-root", type=Path, default=Path(os.__file__).parents[1] / "Lib/site-packages/nvidia/cudnn")
    parser.add_argument("--cudnn-lib", type=Path, help="Directory containing the matching cudnn.lib import library.")
    parser.add_argument("--cutlass", type=Path, required=True, help="CUTLASS repository root.")
    parser.add_argument("--frontend", type=Path, required=True, help="cuDNN Frontend repository root.")
    parser.add_argument("--app-only", action="store_true", help="Reuse this clone's previously built CUDA object.")
    args = parser.parse_args()
    cuda = args.cuda_root.resolve()
    cudnn = args.cudnn_root.resolve()
    cudnn_lib = args.cudnn_lib or cudnn / "lib"
    environment = developer_environment()
    compiler = shutil.which("cl.exe", path=environment.get("Path", environment.get("PATH")))
    required = [cuda / "bin/nvcc.exe", cudnn / "include/cudnn.h", cudnn_lib / "cudnn.lib",
                args.cutlass / "include/cutlass/cutlass.h", args.frontend / "include/cudnn_frontend.h"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not compiler:
        raise FileNotFoundError(f"Missing local build dependencies: {missing}; MSVC={compiler}")
    folder = NATIVE / "obj/Release"
    binary = NATIVE / "bin"
    folder.mkdir(parents=True, exist_ok=True)
    binary.mkdir(exist_ok=True)
    shader = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
              str(NATIVE / "tools/compile-d3d11-shaders.ps1"), "-Root", str(NATIVE), "-WindowsSdkVersion", "10.0.26100.0"]
    subprocess.run(shader, env=environment, check=True)
    includes = [NATIVE / "src", NATIVE / "src/neural", NATIVE / "src/neural/srcpyr", folder / "generated",
                NATIVE / "third_party/imgui", NATIVE / "third_party/imgui/backends", NATIVE / "third_party/minhook/include",
                NATIVE / "third_party/minhook/src", cuda / "include"]
    sdk = Path(environment["WindowsSdkDir"]) / "Include" / environment["WindowsSDKVersion"].strip("\\/") / "cppwinrt"
    includes.append(sdk)
    common = ["/nologo", "/EHsc", "/std:c++20", "/O2", "/MD", "/DNDEBUG", "/DNOMINMAX", "/DWIN32_LEAN_AND_MEAN",
              "/DUNICODE", "/D_UNICODE", "/DWINVER=0x0A00", "/D_WIN32_WINNT=0x0A00", "/Gy", "/Zc:preprocessor",
              *[f"/I{path}" for path in includes], f"/external:I{args.frontend / 'include'}", f"/external:I{cudnn / 'include'}", "/external:W0"]
    cuda_object = folder / "runtime.obj"
    if args.app_only:
        if not cuda_object.is_file():
            raise FileNotFoundError("Run one full build before --app-only.")
        if any(path.stat().st_mtime > cuda_object.stat().st_mtime for path in (NATIVE / "src/neural").rglob("*")
               if path.is_file() and path.suffix in (".cu", ".cuh", ".h") and path.name != "NeuralCudaRuntime.h"):
            raise RuntimeError("Native model sources changed; run a full build.")
    else:
        command = [str(cuda / "bin/nvcc.exe"), "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a",
                   "-allow-unsupported-compiler", "-diag-suppress", "177,20012,20013,20015,20054", "-O3", "-lineinfo", "-DNDEBUG",
                   "-Xcompiler=/EHsc /Zc:preprocessor /DNOMINMAX /DWIN32_LEAN_AND_MEAN", "-Xcompiler=/O2 /MD",
                   *[f"-I{path}" for path in [cuda / "include", args.cutlass / "include", args.cutlass / "tools/util/include",
                                             NATIVE / "src/neural", NATIVE / "src/neural/srcpyr"]],
                   "-c", str(NATIVE / "src/neural/srcpyr/SrcPyrCudaRuntime.cu"), "-o", str(cuda_object)]
        subprocess.run(command, cwd=folder, env=environment, check=True)
    source_files = [NATIVE / "app/app.cpp", folder / "generated/AESSProductionShaders.cpp",
                    *[NATIVE / f"src/neural/srcpyr/{name}.cpp" for name in ("SrcPyrModel", "SrcPyrCudnnConv")],
                    *[NATIVE / f"src/presentation/{name}.cpp" for name in ("FramePacer", "FrameTelemetry", "GsyncForegroundSpoof")],
                    *[NATIVE / f"third_party/imgui/{name}.cpp" for name in ("imgui", "imgui_draw", "imgui_widgets", "imgui_tables")],
                    *[NATIVE / f"third_party/imgui/backends/{name}.cpp" for name in ("imgui_impl_win32", "imgui_impl_dx11")]]
    hook_sources = [NATIVE / "third_party/minhook/src" / path for path in
                    ("buffer.c", "hook.c", "trampoline.c", "hde/hde32.c", "hde/hde64.c")]
    objects = []
    hook_objects = []
    native_headers = list((NATIVE / "src").rglob("*.h"))
    for source in [*source_files, *hook_sources]:
        output = folder / f"{source.stem}.obj"
        inputs = [source, *native_headers, *source.parent.glob("*.h")]
        if source.name == "app.cpp":
            inputs.extend((NATIVE / "app").glob("*.inl"))
            inputs.append(folder / "generated/AESSProductionShaders.h")
        unchanged = output.is_file() and all(path.stat().st_mtime <= output.stat().st_mtime for path in inputs)
        if not (args.app_only and unchanged):
            subprocess.run([compiler, *common, "/c", str(source), f"/Fo{output}"], cwd=folder, env=environment, check=True)
        objects.append(output)
        if source in hook_sources:
            hook_objects.append(output)
    libs = ["d3d11.lib", "d3d12.lib", "dcomp.lib", "dwmapi.lib", "dxgi.lib", "d3dcompiler.lib", "windowsapp.lib",
            "runtimeobject.lib", "user32.lib", "gdi32.lib", "shcore.lib", "comctl32.lib", "shell32.lib", "winmm.lib", "bcrypt.lib"]
    subprocess.run([compiler, *common, *map(str, objects), str(cuda_object), f"/Fe:{binary / 'aeisu_demo_backend.exe'}", "/link",
                    "/SUBSYSTEM:CONSOLE", "/OPT:REF", "/OPT:ICF", f"/LIBPATH:{cuda / 'lib/x64'}", f"/LIBPATH:{cudnn_lib}",
                    "cublasLt.lib", "cudnn.lib", "cudart.lib", *libs], cwd=folder, env=environment, check=True)
    hook_binary = binary / "AESSCaptureHookV25.dll"
    hook_inputs = [NATIVE / "src/capture/AESSCaptureHook.cpp", *hook_objects,
                   *list((NATIVE / "src/capture").glob("*.h"))]
    hook_unchanged = hook_binary.is_file() and all(path.stat().st_mtime <= hook_binary.stat().st_mtime for path in hook_inputs)
    if not (args.app_only and hook_unchanged):
        subprocess.run([compiler, *common, "/LD", str(NATIVE / "src/capture/AESSCaptureHook.cpp"), *map(str, hook_objects),
                        f"/Fe:{hook_binary}", "/link", "d3d11.lib", "d3d12.lib", "dxgi.lib", "user32.lib"],
                       cwd=folder, env=environment, check=True)
    # The model stays a single immutable artifact; shared SDK binaries are located
    # through the launcher's process-only PATH rather than copied into the clone.
    import json
    (binary / "runtime_paths.json").write_text(json.dumps({"cuda_bin": str(cuda / "bin"), "cudnn_bin": str(cudnn / "bin")}, indent=2),
                                              encoding="utf-8")
    print(f"Built {binary / 'aeisu_demo_backend.exe'}")


if __name__ == "__main__":
    main()
