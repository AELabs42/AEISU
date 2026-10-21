"""Package already-built binaries and installed runtimes; no downloads or installs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build import developer_environment  # noqa: E402


def copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source, destination):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pdb"))


def copy_license_notices(python, cuda, cudnn, package):
    """Retain runtime and compiled-header notices without installing dependencies."""
    licenses = package / "licenses"
    copy_file(python / "LICENSE.txt", licenses / "Python-LICENSE.txt")
    copy_file(python / "tcl/tk8.6/license.terms", licenses / "Tk-license.terms")
    copy_file(cuda / "EULA.txt", licenses / "CUDA-EULA.txt")
    copy_file(cuda / "LICENSE", licenses / "CUDA-LICENSE.txt")
    cudnn_licenses = list(cudnn.parents[1].glob("nvidia_cudnn_*.dist-info/licenses/License.txt"))
    if len(cudnn_licenses) != 1:
        raise RuntimeError("Provide a cuDNN Python distribution with its original license notice.")
    copy_file(cudnn_licenses[0], licenses / "cuDNN-LICENSE.txt")
    for dependency in ("imgui", "minhook"):
        copy_file(ROOT / "native/third_party" / dependency / "LICENSE.txt", licenses / f"{dependency}-LICENSE.txt")
    for filename in ("CUTLASS-LICENSE.txt", "cuDNN-Frontend-LICENSE.txt", "nlohmann-json-LICENSE.txt"):
        copy_file(ROOT / "tools/licenses" / filename, licenses / filename)
    copy_file(ROOT / "tools/THIRD_PARTY_RUNTIME.md", package / "THIRD_PARTY_RUNTIME.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-root", type=Path, default=ROOT, help="Source tree containing built native/bin binaries.")
    parser.add_argument("--python-root", type=Path, default=Path(sys.base_prefix))
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--cudnn-root", type=Path, required=True)
    parser.add_argument("--msvc-redist", type=Path, required=True, help="Matching x64 Microsoft.VC145.CRT directory.")
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--version", default="0.1.2")
    parser.add_argument("--skip-archive", action="store_true", help="Build folder only; archive after verification.")
    parser.add_argument("--archive-only", action="store_true", help="Archive an already verified package folder.")
    parser.add_argument("--notices-only", action="store_true", help="Create a notice-only correction; no compiler, binaries or GPU work.")
    args = parser.parse_args()
    name = f"AEISU-Demo-{args.version}-Windows-x64"
    output = args.output.resolve()
    package = output / name
    if args.notices_only:
        if args.archive_only or args.skip_archive:
            parser.error("--notices-only cannot be combined with archive options.")
        build_notice_correction(args.python_root, args.cuda_root, args.cudnn_root, output, args.version)
        return
    if args.archive_only:
        create_archive(package, output / f"{name}.zip")
        return
    if package.exists():
        raise FileExistsError(f"Package folder already exists; use a fresh output directory: {package}")
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("This embedded host targets the current CPython 3.13 x64 ABI.")
    python = args.python_root.resolve()
    native_bin = package / "native/bin"
    package.mkdir(parents=True)
    environment = developer_environment()
    compiler = shutil.which("cl.exe", path=environment.get("Path", environment.get("PATH")))
    compile_folder = output / "launcher-build"
    compile_folder.mkdir(exist_ok=True)
    subprocess.run([compiler, "/nologo", "/std:c++17", "/EHsc", "/O2", "/MD", "/DNOMINMAX", "/DUNICODE", "/D_UNICODE",
                    f"/I{python / 'include'}", str(ROOT / "native/app/portable_launcher.cpp"),
                    f"/Fo:{compile_folder / 'launcher.obj'}", f"/Fe:{package / 'AEISU Demo.exe'}", "/link", "/SUBSYSTEM:WINDOWS",
                    f"/LIBPATH:{python / 'libs'}", "python313.lib", "user32.lib"], cwd=compile_folder, env=environment, check=True)
    copy_file(python / "python313.dll", package / "python313.dll")
    copy_tree(ROOT / "demo", package / "demo")
    copy_file(ROOT / "tests/smoke_native.py", package / "_capture_check.py")
    copy_file(ROOT / "clone_provenance.json", package / "clone_provenance.json")
    copy_file(ROOT / "LICENSE", package / "LICENSE")
    copy_file(ROOT / "tools/PORTABLE_README.md", package / "README.md")
    copy_file(ROOT / "tools/THIRD_PARTY_RUNTIME.md", package / "THIRD_PARTY_RUNTIME.md")
    verification = ROOT / "PORTABLE_VERIFICATION.json"
    if verification.is_file():
        copy_file(verification, package / "PORTABLE_VERIFICATION.json")
    for filename in ("aeisu_demo_backend.exe", "AESSCaptureHookV25.dll"):
        copy_file(args.backend_root / "native/bin" / filename, native_bin / filename)
    copy_file(ROOT / "native/models/aeisu_mixed/adaptive_fp8.bin", package / "native/models/aeisu_mixed/adaptive_fp8.bin")
    (native_bin / "runtime_paths.json").write_text(json.dumps({"portable": True}, indent=2), encoding="utf-8")
    # Never bundle installed site-packages, the user's paths, SDK tools or tests.
    stdlib = package / "python/python313.zip"
    stdlib.parent.mkdir()
    excluded = {"site-packages", "__pycache__", "test", "tests", "idlelib", "ensurepip", "venv", "turtledemo"}
    with zipfile.ZipFile(stdlib, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted((python / "Lib").rglob("*.py")):
            relative = path.relative_to(python / "Lib")
            if not any(part in excluded for part in relative.parts):
                archive.write(path, relative.as_posix())
    for name in ("_tkinter.pyd", "_ctypes.pyd", "_uuid.pyd", "tcl86t.dll", "tk86t.dll", "libffi-8.dll", "zlib1.dll"):
        copy_file(python / "DLLs" / name, package / "python/DLLs" / name)
    for name in ("tcl8.6", "tk8.6", "tcl8"):
        copy_tree(python / "tcl" / name, package / "python/tcl" / name)
    # Use the runtime folder from the matching MSVC redistribution installation.
    for path in args.msvc_redist.glob("*.dll"):
        copy_file(path, package / path.name)
        copy_file(path, native_bin / path.name)
    cuda_bin = args.cuda_root / "bin/x64"
    for pattern in ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll", "nvrtc64_*.dll", "nvrtc-builtins64_*.dll", "nvJitLink_*.dll"):
        candidates = list(cuda_bin.glob(pattern))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one CUDA runtime matching {pattern}: {candidates}")
        copy_file(candidates[0], native_bin / candidates[0].name)
    for path in (args.cudnn_root / "bin").glob("*.dll"):
        copy_file(path, native_bin / path.name)
    copy_license_notices(python, args.cuda_root, args.cudnn_root, package)
    manifest = {
        "version": args.version, "platform": "Windows x64", "gpu_support": "RTX 50-series only; tested RTX 5090",
        "sdr_only": True, "python": sys.version.split()[0], "model_sha256": file_hash(package / "native/models/aeisu_mixed/adaptive_fp8.bin"),
        "files": {path.relative_to(package).as_posix(): {"bytes": path.stat().st_size, "sha256": file_hash(path)}
                  for path in sorted(package.rglob("*")) if path.is_file()},
    }
    (package / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"PACKAGE_FOLDER={package}", flush=True)
    print(f"UNCOMPRESSED_BYTES={sum(entry['bytes'] for entry in manifest['files'].values())}", flush=True)
    if not args.skip_archive:
        create_archive(package, output / f"{name}.zip")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_notice_correction(python, cuda, cudnn, output, version):
    """Produce a small overlay without changing a frozen portable package."""
    folder = output / f"AEISU-Demo-{version}-Notices"
    destination = output / f"{folder.name}.zip"
    if folder.exists() or destination.exists():
        raise FileExistsError("Use a fresh output directory for the notice correction.")
    copy_license_notices(python, cuda, cudnn, folder)
    copy_file(ROOT / "tools/NOTICE_UPDATE.md", folder / "UPDATE_NOTICES.md")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(folder).as_posix())
    with zipfile.ZipFile(destination) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Notice archive CRC verification failed: {bad_member}")
    digest = file_hash(destination)
    destination.with_suffix(".sha256").write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    print(f"NOTICE_ARCHIVE={destination}\nARCHIVE_BYTES={destination.stat().st_size}\nSHA256={digest}", flush=True)


def create_archive(package, destination):
    if destination.exists():
        raise FileExistsError(destination)
    manifest_path = package / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = {
        path.relative_to(package).as_posix(): {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        for path in sorted(package.rglob("*"))
        if path.is_file() and path != manifest_path and "outputs" not in path.relative_to(package).parts
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file() and "outputs" not in path.relative_to(package).parts:
                archive.write(path, (Path(package.name) / path.relative_to(package)).as_posix())
    with zipfile.ZipFile(destination) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Portable archive CRC verification failed: {bad_member}")
    digest = file_hash(destination)
    checksum = destination.with_suffix(".sha256")
    checksum.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    print(f"ARCHIVE={destination}\nARCHIVE_BYTES={destination.stat().st_size}\nSHA256={digest}", flush=True)


if __name__ == "__main__":
    main()
