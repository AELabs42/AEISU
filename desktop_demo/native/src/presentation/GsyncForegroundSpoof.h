#pragma once

#include <windows.h>

#include <string>

namespace aess::presentation
{
    class GsyncForegroundSpoof final
    {
    public:
        GsyncForegroundSpoof() noexcept = default;
        ~GsyncForegroundSpoof();

        GsyncForegroundSpoof(const GsyncForegroundSpoof&) = delete;
        GsyncForegroundSpoof& operator=(const GsyncForegroundSpoof&) = delete;

        bool Enable(HWND outputWindow, std::string& diagnostic) noexcept;
        void Disable() noexcept;
        bool Enabled() const noexcept { return m_enabled; }
        // Bypasses the process-local detour and returns the actual OS
        // foreground window while compatibility spoofing is enabled.
        static HWND RealForegroundWindow() noexcept;

        static bool SelfTest(std::string& diagnostic) noexcept;

    private:
        void* m_target = nullptr;
        bool m_initializedMinHook = false;
        bool m_enabled = false;
    };
}
