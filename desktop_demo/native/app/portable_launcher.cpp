// Small Windows host for the bundled, isolated CPython/Tk launcher.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <windows.h>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

static bool CheckStatus(PyStatus status)
{
    if (!PyStatus_Exception(status)) return true;
    const char* message = status.err_msg ? status.err_msg : "Could not initialize bundled Python.";
    MessageBoxA(nullptr, message, "AEISU launcher", MB_OK | MB_ICONERROR);
    return false;
}

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR arguments, int)
{
    wchar_t executable[32768]{};
    const DWORD length = GetModuleFileNameW(nullptr, executable, 32768);
    if (!length || length >= 32768) return 1;
    const fs::path root = fs::path(executable).parent_path();
    const fs::path python = root / L"python";
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    AddDllDirectory((python / L"DLLs").c_str());
    SetEnvironmentVariableW(L"TCL_LIBRARY", (python / L"tcl/tcl8.6").c_str());
    SetEnvironmentVariableW(L"TK_LIBRARY", (python / L"tcl/tk8.6").c_str());

    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);
    config.site_import = 0;
    config.write_bytecode = 0;
    config.module_search_paths_set = 1;
    config.parse_argv = 0;
    bool okay = CheckStatus(PyConfig_SetString(&config, &config.home, python.c_str())) &&
        CheckStatus(PyConfig_SetString(&config, &config.program_name, executable)) &&
        CheckStatus(PyConfig_SetString(&config, &config.executable, executable));
    for (const fs::path& path : {root, python / L"python313.zip", python / L"DLLs"})
        okay = okay && CheckStatus(PyWideStringList_Append(&config.module_search_paths, path.c_str()));
    okay = okay && CheckStatus(PyWideStringList_Append(&config.argv, executable));
    const std::wstring mode = arguments ? arguments : L"";
    if (mode == L"--self-test" || mode == L"--smoke-test")
        okay = okay && CheckStatus(PyWideStringList_Append(&config.argv, mode.c_str()));
    else if (!mode.empty())
    {
        MessageBoxW(nullptr, L"Supported diagnostic arguments: --self-test, --smoke-test", L"AEISU launcher", MB_OK | MB_ICONERROR);
        okay = false;
    }
    if (okay) okay = CheckStatus(Py_InitializeFromConfig(&config));
    PyConfig_Clear(&config);
    if (!okay) return 1;
    const int result = PyRun_SimpleString(
        "import sys\n"
        "import os\n"
        "from pathlib import Path\n"
        "root = Path(sys.executable).resolve().parent\n"
        "os.environ['TCL_LIBRARY'] = str(root / 'python/tcl/tcl8.6')\n"
        "os.environ['TK_LIBRARY'] = str(root / 'python/tcl/tk8.6')\n"
        "(root / 'outputs').mkdir(exist_ok=True)\n"
        "sys.stdout = sys.stderr = (root / 'outputs/launcher.log').open('w', encoding='utf-8', buffering=1)\n"
        "if len(sys.argv) > 1:\n"
        "    from demo.portable_checks import run\n"
        "    run(sys.argv[1])\n"
        "else:\n"
        "    from demo.ui import main\n"
        "    main()\n");
    if (result != 0 && mode.empty())
        MessageBoxW(nullptr, L"The launcher failed. See outputs\\launcher.log in the extracted demo folder.", L"AEISU launcher", MB_OK | MB_ICONERROR);
    const int finalized = Py_FinalizeEx();
    return result == 0 && finalized == 0 ? 0 : 1;
}
