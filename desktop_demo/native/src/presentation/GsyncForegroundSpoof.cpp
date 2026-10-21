#include "GsyncForegroundSpoof.h"

#include <MinHook.h>

#include <atomic>
#include <sstream>

namespace
{
    using GetForegroundWindowFn = HWND(WINAPI*)();
    std::atomic<HWND> g_spoofedForegroundWindow{ nullptr };
    GetForegroundWindowFn g_originalGetForegroundWindow = nullptr;

    HWND WINAPI DetouredGetForegroundWindow()
    {
        return g_spoofedForegroundWindow.load(std::memory_order_relaxed);
    }

    std::string MinHookFailure(const char* operation, MH_STATUS status)
    {
        std::ostringstream message;
        message << operation << " failed: " << MH_StatusToString(status)
                << " (" << static_cast<int>(status) << ")";
        return message.str();
    }
}

namespace aess::presentation
{
    GsyncForegroundSpoof::~GsyncForegroundSpoof()
    {
        Disable();
    }

    bool GsyncForegroundSpoof::Enable(HWND outputWindow, std::string& diagnostic) noexcept
    {
        diagnostic.clear();
        if (m_enabled)
        {
            diagnostic = "foreground compatibility hook is already enabled";
            return outputWindow == g_spoofedForegroundWindow.load(std::memory_order_relaxed);
        }

        DWORD ownerProcessId = 0;
        if (!IsWindow(outputWindow) ||
            GetWindowThreadProcessId(outputWindow, &ownerProcessId) == 0 ||
            ownerProcessId != GetCurrentProcessId())
        {
            diagnostic = "output HWND is not owned by the AESS process";
            return false;
        }

        MH_STATUS status = MH_Initialize();
        if (status == MH_OK)
        {
            m_initializedMinHook = true;
        }
        else if (status != MH_ERROR_ALREADY_INITIALIZED)
        {
            diagnostic = MinHookFailure("MH_Initialize", status);
            return false;
        }

        status = MH_CreateHookApiEx(
            L"user32.dll",
            "GetForegroundWindow",
            reinterpret_cast<void*>(&DetouredGetForegroundWindow),
            reinterpret_cast<void**>(&g_originalGetForegroundWindow),
            &m_target);
        if (status != MH_OK)
        {
            diagnostic = MinHookFailure("MH_CreateHookApiEx(GetForegroundWindow)", status);
            if (m_initializedMinHook) MH_Uninitialize();
            m_initializedMinHook = false;
            g_originalGetForegroundWindow = nullptr;
            m_target = nullptr;
            return false;
        }

        g_spoofedForegroundWindow.store(outputWindow, std::memory_order_release);
        status = MH_EnableHook(m_target);
        if (status != MH_OK)
        {
            MH_RemoveHook(m_target);
            diagnostic = MinHookFailure("MH_EnableHook(GetForegroundWindow)", status);
            if (m_initializedMinHook) MH_Uninitialize();
            m_initializedMinHook = false;
            g_originalGetForegroundWindow = nullptr;
            m_target = nullptr;
            return false;
        }

        m_enabled = true;
        diagnostic = "enabled";
        return true;
    }

    void GsyncForegroundSpoof::Disable() noexcept
    {
        if (m_target)
        {
            if (m_enabled) MH_DisableHook(m_target);
            MH_RemoveHook(m_target);
        }

        if (m_initializedMinHook) MH_Uninitialize();
        g_originalGetForegroundWindow = nullptr;
        m_target = nullptr;
        m_initializedMinHook = false;
        m_enabled = false;
    }

    HWND GsyncForegroundSpoof::RealForegroundWindow() noexcept
    {
        return g_originalGetForegroundWindow
            ? g_originalGetForegroundWindow()
            : ::GetForegroundWindow();
    }

    bool GsyncForegroundSpoof::SelfTest(std::string& diagnostic) noexcept
    {
        HWND testWindow = CreateWindowExW(
            WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            L"STATIC",
            L"AESS foreground compatibility self-test",
            WS_POPUP,
            0,
            0,
            1,
            1,
            nullptr,
            nullptr,
            GetModuleHandleW(nullptr),
            nullptr);
        if (!testWindow)
        {
            diagnostic = "self-test window creation failed";
            return false;
        }

        GsyncForegroundSpoof spoof;
        std::string enableDiagnostic;
        const bool enabled = spoof.Enable(testWindow, enableDiagnostic);
        const bool redirected = enabled && GetForegroundWindow() == testWindow;
        spoof.Disable();
        const bool restored = GetForegroundWindow() != testWindow;
        DestroyWindow(testWindow);

        if (!enabled || !redirected || !restored)
        {
            std::ostringstream message;
            message << "enable=" << (enabled ? "yes" : "no")
                    << " redirect=" << (redirected ? "yes" : "no")
                    << " restore=" << (restored ? "yes" : "no")
                    << " detail=" << enableDiagnostic;
            diagnostic = message.str();
            return false;
        }

        diagnostic = "hook redirected only process-local foreground queries and restored cleanly";
        return true;
    }
}
