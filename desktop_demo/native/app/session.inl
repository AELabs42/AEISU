    struct WindowState
    {
        WindowState()
        {
            shutdownEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            commandEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!shutdownEvent || !commandEvent)
            {
                if (shutdownEvent) CloseHandle(shutdownEvent);
                if (commandEvent) CloseHandle(commandEvent);
                throw std::runtime_error("Create AESS render-worker control events failed.");
            }
        }

        ~WindowState()
        {
            if (outputCursor) DestroyCursor(outputCursor);
            if (commandEvent) CloseHandle(commandEvent);
            if (shutdownEvent) CloseHandle(shutdownEvent);
        }

        WindowState(const WindowState&) = delete;
        WindowState& operator=(const WindowState&) = delete;

        std::atomic<bool> stopRequested{ false };
        std::atomic<uint64_t> upscalerBypassToggleCount{ 0 };
        std::atomic<uint64_t> fpsCounterToggleCount{ 0 };
        std::atomic<uint64_t> focusTransitionCount{ 0 };
        std::atomic<uint64_t> displayChangeCount{ 0 };
        std::atomic<bool> sourceIsForeground{ true };
        std::atomic<bool> upscalerBypass{ false };
        HANDLE shutdownEvent = nullptr;
        HANDLE commandEvent = nullptr;
        HWND sourceWindow = nullptr;
        // Private copy of the source/game cursor. Keeping our own handle lets
        // the hardware cursor remain independent from the source process's
        // cursor-resource lifetime without changing the global ShowCursor
        // counter or drawing a frame-late software cursor.
        HCURSOR outputCursor = nullptr;
        bool outputCursorVisible = true;
        std::atomic<D3D11VideoPresenter*> presentationDispatcher{ nullptr };
    };

    std::atomic<WindowState*> g_pacingWindowState{ nullptr };
    HWINEVENTHOOK g_foregroundTransitionHook = nullptr;

    bool IsSourceForeground(HWND source, HWND foreground) noexcept
    {
        if (!IsWindow(source) || !IsWindow(foreground)) return false;
        const HWND sourceRoot = GetAncestor(source, GA_ROOTOWNER);
        const HWND foregroundRoot = GetAncestor(foreground, GA_ROOTOWNER);
        return foreground == source ||
            (sourceRoot && foregroundRoot && sourceRoot == foregroundRoot);
    }

    void CALLBACK ForegroundTransitionCallback(
        HWINEVENTHOOK,
        DWORD event,
        HWND foreground,
        LONG,
        LONG,
        DWORD,
        DWORD)
    {
        if (event != EVENT_SYSTEM_FOREGROUND) return;
        WindowState* state = g_pacingWindowState.load(std::memory_order_acquire);
        if (!state) return;
        const bool sourceIsForeground =
            IsSourceForeground(state->sourceWindow, foreground);
        const bool previous = state->sourceIsForeground.exchange(
            sourceIsForeground,
            std::memory_order_acq_rel);
        if (previous != sourceIsForeground)
        {
            state->focusTransitionCount.fetch_add(1, std::memory_order_release);
            SetEvent(state->commandEvent);
        }
    }

    bool StartForegroundTransitionTracking(WindowState& state) noexcept
    {
        const HWND realForeground =
            aess::presentation::GsyncForegroundSpoof::RealForegroundWindow();
        state.sourceIsForeground.store(
            IsSourceForeground(state.sourceWindow, realForeground),
            std::memory_order_release);
        g_pacingWindowState.store(&state, std::memory_order_release);
        g_foregroundTransitionHook = SetWinEventHook(
            EVENT_SYSTEM_FOREGROUND,
            EVENT_SYSTEM_FOREGROUND,
            nullptr,
            &ForegroundTransitionCallback,
            0,
            0,
            WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS);
        if (!g_foregroundTransitionHook)
        {
            g_pacingWindowState.store(nullptr, std::memory_order_release);
            return false;
        }
        return true;
    }

    void StopForegroundTransitionTracking() noexcept
    {
        g_pacingWindowState.store(nullptr, std::memory_order_release);
        if (g_foregroundTransitionHook)
        {
            UnhookWinEvent(g_foregroundTransitionHook);
            g_foregroundTransitionHook = nullptr;
        }
    }

    class ScopedForegroundTransitionTracking final
    {
    public:
        explicit ScopedForegroundTransitionTracking(WindowState& state)
        {
            if (!StartForegroundTransitionTracking(state))
            {
                throw std::runtime_error(
                    "Install foreground-transition pacing hook failed.");
            }
            m_active = true;
        }

        ~ScopedForegroundTransitionTracking()
        {
            Reset();
        }

        ScopedForegroundTransitionTracking(
            const ScopedForegroundTransitionTracking&) = delete;
        ScopedForegroundTransitionTracking& operator=(
            const ScopedForegroundTransitionTracking&) = delete;

        void Reset() noexcept
        {
            if (!m_active) return;
            StopForegroundTransitionTracking();
            m_active = false;
        }

    private:
        bool m_active = false;
    };

    constexpr DWORD kOutputWindowStyle = WS_POPUP;
    constexpr DWORD kOutputWindowExStyle =
        WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE;

    HCURSOR DefaultOutputCursor() noexcept
    {
        static HCURSOR cursor = LoadCursorW(nullptr, IDC_ARROW);
        return cursor;
    }

    HCURSOR VisibleOutputCursor(const WindowState* state) noexcept
    {
        if (state && !state->outputCursorVisible) return nullptr;
        return state && state->outputCursor
            ? state->outputCursor
            : DefaultOutputCursor();
    }

    // Lossless Scaling recovers the active HCURSOR, its hotspot, and its
    // bitmap, then composites that bitmap through the GPU. AESS can retain
    // the same shape fidelity at lower latency by leaving it on Windows' own
    // hardware-cursor plane. CopyImage gives AESS a stable private handle;
    // no cursor visibility, clipping, sensitivity, or focus state is changed.
    void AdoptCurrentHardwareCursor(WindowState& state) noexcept
    {
        CURSORINFO info{};
        info.cbSize = sizeof(info);
        if (!GetCursorInfo(&info)) return;

        const bool visible =
            (info.flags & CURSOR_SHOWING) != 0 && info.hCursor != nullptr;
        state.outputCursorVisible = visible;
        if (!visible || info.hCursor == state.outputCursor) return;

        HCURSOR copy = static_cast<HCURSOR>(
            CopyImage(info.hCursor, IMAGE_CURSOR, 0, 0, 0));
        if (!copy) return;

        if (state.outputCursor) DestroyCursor(state.outputCursor);
        state.outputCursor = copy;
    }

    void ForwardLegacyMouseButton(
        HWND output,
        HWND source,
        UINT message,
        WPARAM wParam,
        LPARAM lParam)
    {
        if (!IsWindow(source)) return;

        LPARAM mappedLParam = lParam;
        HWND target = source;
        if (message != WM_MOUSEWHEEL && message != WM_MOUSEHWHEEL)
        {
            RECT outputClient{};
            RECT sourceClient{};
            if (GetClientRect(output, &outputClient) &&
                GetClientRect(source, &sourceClient))
            {
                const LONG outputWidth = std::max<LONG>(1, outputClient.right);
                const LONG outputHeight = std::max<LONG>(1, outputClient.bottom);
                const LONG sourceWidth = std::max<LONG>(1, sourceClient.right);
                const LONG sourceHeight = std::max<LONG>(1, sourceClient.bottom);
                const LONG x = static_cast<short>(LOWORD(lParam));
                const LONG y = static_cast<short>(HIWORD(lParam));
                const LONG mappedX = std::clamp<LONG>(
                    MulDiv(x, sourceWidth, outputWidth), 0, sourceWidth - 1);
                const LONG mappedY = std::clamp<LONG>(
                    MulDiv(y, sourceHeight, outputHeight), 0, sourceHeight - 1);
                POINT sourcePoint{ mappedX, mappedY };
                HWND child = ChildWindowFromPointEx(
                    source,
                    sourcePoint,
                    CWP_SKIPINVISIBLE | CWP_SKIPDISABLED | CWP_SKIPTRANSPARENT);
                if (child && child != source)
                {
                    POINT targetPoint = sourcePoint;
                    ClientToScreen(source, &targetPoint);
                    ScreenToClient(child, &targetPoint);
                    target = child;
                    mappedLParam = MAKELPARAM(targetPoint.x, targetPoint.y);
                }
                else
                {
                    mappedLParam = MAKELPARAM(mappedX, mappedY);
                }
            }
        }

        // Cross-process PostMessage is non-blocking. Raw relative movement is
        // deliberately not replayed: the foreground game receives WM_INPUT
        // directly and remains the sole owner of cursor locking.
        PostMessageW(target, message, wParam, mappedLParam);
    }

    HHOOK g_outputMouseHook = nullptr;
    HWND g_outputMouseHookWindow = nullptr;
    HWND g_outputMouseHookSource = nullptr;
    HANDLE g_outputMouseHookThread = nullptr;
    HANDLE g_outputMouseHookReady = nullptr;
    DWORD g_outputMouseHookThreadId = 0;
    volatile LONG64 g_outputMouseHookEvents = 0;
    volatile LONG64 g_outputMouseHookForwarded = 0;

    LRESULT CALLBACK OutputMouseHookProc(int code, WPARAM wParam, LPARAM lParam)
    {
        if (code == HC_ACTION && g_outputMouseHookWindow && g_outputMouseHookSource &&
            IsWindowVisible(g_outputMouseHookWindow))
        {
            InterlockedIncrement64(&g_outputMouseHookEvents);
            UINT message = static_cast<UINT>(wParam);
            UINT buttonState = 0;
            switch (message)
            {
            case WM_LBUTTONDOWN: buttonState = MK_LBUTTON; break;
            case WM_LBUTTONUP: break;
            case WM_RBUTTONDOWN: buttonState = MK_RBUTTON; break;
            case WM_RBUTTONUP: break;
            case WM_MBUTTONDOWN: buttonState = MK_MBUTTON; break;
            case WM_MBUTTONUP: break;
            case WM_XBUTTONDOWN: buttonState = MK_XBUTTON1; break;
            case WM_XBUTTONUP: break;
            case WM_MOUSEWHEEL:
            case WM_MOUSEHWHEEL:
                break;
            default:
                return CallNextHookEx(g_outputMouseHook, code, wParam, lParam);
            }

            const auto* mouse = reinterpret_cast<const MSLLHOOKSTRUCT*>(lParam);
            RECT outputRect{};
            const HWND source = g_outputMouseHookSource;
            if (mouse && GetWindowRect(g_outputMouseHookWindow, &outputRect) &&
                PtInRect(&outputRect, mouse->pt) && IsWindow(source))
            {
                POINT clientPoint = mouse->pt;
                ScreenToClient(g_outputMouseHookWindow, &clientPoint);
                WPARAM forwardedWParam = buttonState;
                if (message == WM_XBUTTONDOWN || message == WM_XBUTTONUP)
                {
                    forwardedWParam = MAKEWPARAM(
                        buttonState,
                        HIWORD(mouse->mouseData));
                }
                else if (message == WM_MOUSEWHEEL || message == WM_MOUSEHWHEEL)
                {
                    forwardedWParam = MAKEWPARAM(0, HIWORD(mouse->mouseData));
                }
                const LPARAM forwardedLParam =
                    message == WM_MOUSEWHEEL || message == WM_MOUSEHWHEEL
                        ? MAKELPARAM(mouse->pt.x, mouse->pt.y)
                        : MAKELPARAM(clientPoint.x, clientPoint.y);
                ForwardLegacyMouseButton(
                    g_outputMouseHookWindow,
                    source,
                    message,
                    forwardedWParam,
                    forwardedLParam);
                InterlockedIncrement64(&g_outputMouseHookForwarded);
            }
        }
        return CallNextHookEx(g_outputMouseHook, code, wParam, lParam);
    }

    DWORD WINAPI OutputMouseHookThreadProc(void*)
    {
        // The hook callback is delivered to this thread. Keep it independent
        // from CUDA initialization and render waits so Windows never removes
        // the hook for exceeding LowLevelHooksTimeout.
        MSG message{};
        PeekMessageW(&message, nullptr, WM_USER, WM_USER, PM_NOREMOVE);
        g_outputMouseHook = SetWindowsHookExW(
            WH_MOUSE_LL,
            OutputMouseHookProc,
            GetModuleHandleW(nullptr),
            0);
        SetEvent(g_outputMouseHookReady);
        if (!g_outputMouseHook) return 1;
        while (GetMessageW(&message, nullptr, 0, 0) > 0)
        {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        UnhookWindowsHookEx(g_outputMouseHook);
        g_outputMouseHook = nullptr;
        return 0;
    }

    bool StartOutputMouseHook(HWND output, HWND source)
    {
        g_outputMouseHookWindow = output;
        g_outputMouseHookSource = source;
        InterlockedExchange64(&g_outputMouseHookEvents, 0);
        InterlockedExchange64(&g_outputMouseHookForwarded, 0);
        g_outputMouseHookReady = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!g_outputMouseHookReady) return false;
        g_outputMouseHookThread = CreateThread(
            nullptr,
            0,
            OutputMouseHookThreadProc,
            nullptr,
            0,
            &g_outputMouseHookThreadId);
        if (!g_outputMouseHookThread ||
            WaitForSingleObject(g_outputMouseHookReady, 2000) != WAIT_OBJECT_0 ||
            !g_outputMouseHook)
        {
            return false;
        }
        return true;
    }

    void StopOutputMouseHook() noexcept
    {
        if (g_outputMouseHookThreadId)
            PostThreadMessageW(g_outputMouseHookThreadId, WM_QUIT, 0, 0);
        if (g_outputMouseHookThread)
            WaitForSingleObject(g_outputMouseHookThread, 2000);
        if (g_outputMouseHookThread) CloseHandle(g_outputMouseHookThread);
        if (g_outputMouseHookReady) CloseHandle(g_outputMouseHookReady);
        g_outputMouseHookThread = nullptr;
        g_outputMouseHookReady = nullptr;
        g_outputMouseHookThreadId = 0;
        g_outputMouseHookWindow = nullptr;
        g_outputMouseHookSource = nullptr;
    }

    constexpr UINT kHideStoppedOutputMessage = WM_APP + 0x42;

    void RequestOutputStop(HWND hwnd, WindowState* state)
    {
        if (state)
        {
            state->stopRequested.store(true, std::memory_order_release);
            SetEvent(state->shutdownEvent);
            // Remove the stale displayed frame immediately, but keep the HWND
            // alive until the render worker has drained GPU/swapchain resources.
            // Worker callers must not synchronously enter the window thread.
            if (IsWindow(hwnd))
            {
                if (GetWindowThreadProcessId(hwnd, nullptr) == GetCurrentThreadId())
                    ShowWindow(hwnd, SW_HIDE);
                else
                    PostMessageW(hwnd, kHideStoppedOutputMessage, 0, 0);
            }
        }
    }

    void RequestUpscalerBypassToggle(WindowState* state)
    {
        if (state)
        {
            state->upscalerBypassToggleCount.fetch_add(1, std::memory_order_release);
            SetEvent(state->commandEvent);
        }
    }

    void RequestFpsCounterToggle(WindowState* state)
    {
        if (state)
        {
            state->fpsCounterToggleCount.fetch_add(1, std::memory_order_release);
            SetEvent(state->commandEvent);
        }
    }

    LRESULT CALLBACK OutputWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
    {
        auto* state = reinterpret_cast<WindowState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        switch (msg)
        {
        case WM_NCCREATE:
        {
            auto* create = reinterpret_cast<CREATESTRUCTW*>(lParam);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
            return TRUE;
        }
        case WM_NCHITTEST:
            return HTCLIENT;
        case WM_MOUSEACTIVATE:
            return MA_NOACTIVATE;
        case WM_SETCURSOR:
            // If the foreground game changed its cursor asynchronously (for
            // example, arrow to hand or resize), retain that shape. Usually
            // this sees our existing private handle and performs no work.
            if (state && state->sourceIsForeground.load(std::memory_order_acquire))
                AdoptCurrentHardwareCursor(*state);
            SetCursor(VisibleOutputCursor(state));
            return TRUE;
        case kSwapChainLifecycleMessage:
            if (state)
            {
                D3D11VideoPresenter* dispatcher =
                    state->presentationDispatcher.load(std::memory_order_acquire);
                if (dispatcher)
                {
                    return static_cast<LRESULT>(
                        dispatcher->ExecuteWindowThreadSwapChainLifecycle());
                }
            }
            return static_cast<LRESULT>(E_HANDLE);
        case kHideStoppedOutputMessage:
            if (state && state->stopRequested.load(std::memory_order_acquire))
                ShowWindow(hwnd, SW_HIDE);
            return 0;
        case WM_HOTKEY:
            if (wParam == kToggleHotkeyId)
            {
                RequestOutputStop(hwnd, state);
                return 0;
            }
            if (wParam == kUpscalerBypassHotkeyId)
            {
                RequestUpscalerBypassToggle(state);
                return 0;
            }
            if (wParam == kFpsCounterHotkeyId)
            {
                RequestFpsCounterToggle(state);
                return 0;
            }
            break;
        case WM_KEYDOWN:
            if (wParam == VK_TAB &&
                (GetKeyState(VK_CONTROL) & 0x8000) != 0 &&
                (GetKeyState(VK_SHIFT) & 0x8000) != 0)
            {
                RequestUpscalerBypassToggle(state);
                return 0;
            }
            if (wParam == kFpsCounterHotkeyId)
            {
                RequestFpsCounterToggle(state);
                return 0;
            }
            if (wParam == VK_TAB && (GetKeyState(VK_CONTROL) & 0x8000) != 0)
            {
                RequestOutputStop(hwnd, state);
                return 0;
            }
            if (wParam == VK_ESCAPE)
            {
                RequestOutputStop(hwnd, state);
                return 0;
            }
            break;
        case WM_CLOSE:
            RequestOutputStop(hwnd, state);
            return 0;
        case WM_DISPLAYCHANGE:
            if (state)
            {
                state->displayChangeCount.fetch_add(1, std::memory_order_release);
                SetEvent(state->commandEvent);
            }
            return 0;
        case WM_DESTROY:
            if (g_outputMouseHookWindow == hwnd) StopOutputMouseHook();
            UnregisterHotKey(hwnd, kToggleHotkeyId);
            UnregisterHotKey(hwnd, kUpscalerBypassHotkeyId);
            UnregisterHotKey(hwnd, kFpsCounterHotkeyId);
            if (state)
            {
                state->stopRequested.store(true, std::memory_order_release);
                SetEvent(state->shutdownEvent);
            }
            return 0;
        default:
            break;
        }
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }

    HWND CreateOutputWindow(
        HINSTANCE instance,
        HWND sourceWindow,
        const RECT& rect,
        WindowState& state,
        bool excludeFromCapture,
        bool foregroundPresentationTest = false)
    {
        const wchar_t* className = L"AEISUDemo.OutputWindow";
        state.sourceWindow = sourceWindow;
        AdoptCurrentHardwareCursor(state);

        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = OutputWndProc;
        wc.hInstance = instance;
        wc.hCursor = DefaultOutputCursor();
        wc.lpszClassName = className;
        RegisterClassExW(&wc);

        int width = rect.right - rect.left;
        int height = rect.bottom - rect.top;

        HWND hwnd = CreateWindowExW(
            foregroundPresentationTest
                ? (kOutputWindowExStyle & ~WS_EX_NOACTIVATE)
                : kOutputWindowExStyle,
            className,
            L"AEISU Demo Output",
            kOutputWindowStyle,
            rect.left,
            rect.top,
            width,
            height,
            nullptr,
            nullptr,
            instance,
            &state);

        if (!hwnd)
        {
            throw std::runtime_error("CreateWindowEx failed.");
        }

        // Direct-hook and WGC capture target the source window. Desktop
        // duplication alone needs affinity to prevent recursive capture; do
        // not force normal output through capture-protected DWM composition.
        if (excludeFromCapture && !SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
        {
            SetWindowDisplayAffinity(hwnd, WDA_MONITOR);
        }
        RegisterHotKey(hwnd, kToggleHotkeyId, MOD_CONTROL | MOD_NOREPEAT, VK_TAB);
        if (!RegisterHotKey(hwnd, kUpscalerBypassHotkeyId, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, VK_TAB))
        {
            DestroyWindow(hwnd);
            throw std::runtime_error("Could not register Ctrl+Shift+Tab. Close another app using that hotkey and retry.");
        }
        RegisterHotKey(hwnd, kFpsCounterHotkeyId, MOD_CONTROL | MOD_NOREPEAT, 'F');

        const UINT positioningFlags =
            SWP_FRAMECHANGED | SWP_SHOWWINDOW | SWP_NOSENDCHANGING |
            (foregroundPresentationTest ? 0u : SWP_NOACTIVATE);
        if (!SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            rect.left,
            rect.top,
            width,
            height,
            positioningFlags))
        {
            DestroyWindow(hwnd);
            throw std::runtime_error("SetWindowPos failed for borderless fullscreen output.");
        }
        ShowWindow(hwnd, foregroundPresentationTest ? SW_SHOW : SW_SHOWNOACTIVATE);
        UpdateWindow(hwnd);
        // The pointer may already be stationary over the newly created output,
        // in which case Windows has no reason to send WM_SETCURSOR. Install the
        // visible hardware cursor immediately without changing ShowCursor's
        // process-global visibility count or taking input ownership from the
        // source game.
        SetCursor(VisibleOutputCursor(&state));
        if (!StartOutputMouseHook(hwnd, sourceWindow))
        {
            StopOutputMouseHook();
            DestroyWindow(hwnd);
            throw std::runtime_error("Install output mouse-button compatibility hook failed.");
        }
        return hwnd;
    }

    class ScopedOutputWindow final
    {
    public:
        explicit ScopedOutputWindow(HWND hwnd) noexcept : m_hwnd(hwnd) {}
        ~ScopedOutputWindow() { Reset(); }

        ScopedOutputWindow(const ScopedOutputWindow&) = delete;
        ScopedOutputWindow& operator=(const ScopedOutputWindow&) = delete;

        void Reset() noexcept
        {
            if (IsWindow(m_hwnd)) DestroyWindow(m_hwnd);
            m_hwnd = nullptr;
        }

    private:
        HWND m_hwnd = nullptr;
    };

    void RunRenderWorker(
        HWND output,
        WindowState& windowState,
        ICaptureSource& capture,
        const Options& options,
        std::unique_ptr<D3D11VideoPresenter> presenter,
        HANDLE doneEvent,
        std::exception_ptr& workerError) noexcept
    {
        try
        {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
            if (!SetThreadPriority(
                    GetCurrentThread(),
                    THREAD_PRIORITY_HIGHEST))
            {
                ThrowIfFailed(
                    HRESULT_FROM_WIN32(GetLastError()),
                    "Set AESS render thread priority to highest");
            }
            SetThreadDescription(GetCurrentThread(), L"AESS Render/Present");

            FrameInfo currentFrame{};
            std::array<SourceTimingSample, kSourceTimingJournalCapacity> sourceTimingSamples;
            LARGE_INTEGER qpcFrequency{};
            if (!QueryPerformanceFrequency(&qpcFrequency) || qpcFrequency.QuadPart <= 0)
            {
                throw std::runtime_error("Capture timing QPC frequency is unavailable.");
            }
            const double qpcToMilliseconds = 1000.0 / static_cast<double>(qpcFrequency.QuadPart);
            uint64_t submittedFrames = 0;
            uint64_t lastSubmittedSourceVersion = 0;
            const std::filesystem::path activationDumpDir =
                options.debugDumpsEnabled
                    ? ActivationDebugDumpDirectory()
                    : std::filesystem::path{};

            while (!windowState.stopRequested.load(std::memory_order_acquire))
            {
                if (!IsWindow(output))
                {
                    RequestOutputStop(output, &windowState);
                    break;
                }

                const uint64_t bypassToggles =
                    windowState.upscalerBypassToggleCount.exchange(0, std::memory_order_acq_rel);
                for (uint64_t index = 0; index < bypassToggles; ++index)
                {
                    const bool bypass = presenter->ToggleUpscalerBypass();
                    windowState.upscalerBypass.store(bypass, std::memory_order_release);
                }
                const uint64_t fpsToggles =
                    windowState.fpsCounterToggleCount.exchange(0, std::memory_order_acq_rel);
                for (uint64_t index = 0; index < fpsToggles; ++index)
                {
                    presenter->ToggleFpsCounter();
                }
                const uint64_t displayChanges =
                    windowState.displayChangeCount.exchange(0, std::memory_order_acq_rel);
                const uint64_t focusTransitions =
                    windowState.focusTransitionCount.exchange(0, std::memory_order_acq_rel);
                if (displayChanges != 0 || focusTransitions != 0)
                {
                    // A display change subsumes the focus-only pacing reset.
                    presenter->HandlePacingEnvironmentChange(displayChanges != 0);
                }

                // Capture cadence remains observable while a completed frame
                // is deliberately held for adaptive/smooth presentation. The
                // sideband owns no texture and cannot overwrite capture or
                // presentation resources.
                uint64_t sourceTimingOverflowCount = 0;
                const size_t sourceTimingCount = capture.DrainSourceTimings(
                    sourceTimingSamples.data(),
                    sourceTimingSamples.size(),
                    sourceTimingOverflowCount);
                for (size_t index = 0; index < sourceTimingCount; ++index)
                {
                    presenter->ObserveSourceTiming(
                        sourceTimingSamples[index].sourcePresentQpc,
                        sourceTimingSamples[index].sourcePresentOrdinal);
                }
                presenter->RecordSourceTimingJournalOverflow(
                    sourceTimingOverflowCount);

                bool didFrameWork = false;
                D3D11RenderResult pumpedResult{};
                if (presenter->PumpAsyncUpscaler(pumpedResult))
                {
                    didFrameWork = true;
                }

                FrameInfo acquired{};
                bool hasNewFrame = false;
                double acquireMs = 0.0;
                const bool canAcquireFrame = presenter->CanAcceptFrame();
                if (canAcquireFrame)
                {
                    const int64_t acquireBeginQpc = FramePacer::QueryQpc();
                    hasNewFrame = capture.Acquire(acquired);
                    const int64_t acquireEndQpc = FramePacer::QueryQpc();
                    acquireMs = static_cast<double>(acquireEndQpc - acquireBeginQpc) * qpcToMilliseconds;
                    if (hasNewFrame)
                    {
                        acquired.captureAcquireBeginQpc = acquireBeginQpc;
                        acquired.captureAcquireEndQpc = acquireEndQpc;
                        if (acquired.captureReadyQpc <= 0)
                        {
                            acquired.captureReadyQpc = acquireEndQpc;
                        }
                    }
                }
                else
                {
                    presenter->NotePipelineAdmissionBlocked();
                }

                if (hasNewFrame)
                {
                    didFrameWork = true;
                    currentFrame = std::move(acquired);
                    if (currentFrame.sharedHandle || currentFrame.pixels || currentFrame.d3d11Texture)
                    {
                        const uint64_t frameNumber = submittedFrames + 1;
                        uint64_t skippedFrames = 0;
                        if (lastSubmittedSourceVersion != 0 &&
                            currentFrame.version > lastSubmittedSourceVersion + 1)
                        {
                            skippedFrames = currentFrame.version - lastSubmittedSourceVersion - 1;
                        }
                        const std::filesystem::path dumpDir =
                            options.debugDumpsEnabled && frameNumber == kActivationDebugDumpFrame
                                ? activationDumpDir
                                : std::filesystem::path{};
                        try
                        {
                            presenter->Render(
                                currentFrame,
                                dumpDir,
                                frameNumber,
                                acquireMs,
                                skippedFrames);
                        }
                        catch (...)
                        {
                            capture.Release(currentFrame);
                            throw;
                        }
                        capture.Release(currentFrame);
                        submittedFrames = frameNumber;
                        lastSubmittedSourceVersion = currentFrame.version;
                        if (options.benchmarkFrames > 0 &&
                            submittedFrames >= options.benchmarkFrames)
                        {
                            RequestOutputStop(output, &windowState);
                        }
                    }
                }

                if (windowState.stopRequested.load(std::memory_order_acquire)) break;
                if (didFrameWork) continue;

                // Capacity covers shutdown, commands, capture, two model
                // completions, D3D completion fences, DXGI latency eligibility,
                // WGC input-consumption fence and the pacing timer without
                // crowding out a deadline wake.
                std::array<HANDLE, 11> waitHandles{};
                size_t waitCount = 0;
                const size_t shutdownIndex = waitCount;
                waitHandles[waitCount++] = windowState.shutdownEvent;
                const size_t commandIndex = waitCount;
                waitHandles[waitCount++] = windowState.commandEvent;

                // Deadline and completion wakes take precedence over a noisy
                // capture source.  Do not arm capture readiness while both
                // physical pipeline slots are occupied: the bounded transport
                // retains the newest source frame and the next completion wake
                // re-enables admission without spinning this highest-priority
                // owner thread.
                const HANDLE pacingTimer = presenter->PacingTimerEvent();
                const size_t pacingTimerIndex = pacingTimer && waitCount < waitHandles.size()
                    ? waitCount
                    : static_cast<size_t>(-1);
                if (pacingTimerIndex != static_cast<size_t>(-1))
                {
                    waitHandles[waitCount++] = pacingTimer;
                }
                waitCount += presenter->AppendAsyncWaitHandles(
                    waitHandles.data() + waitCount,
                    waitHandles.size() - waitCount);

                const bool canAcceptFrame = presenter->CanAcceptFrame();
                HANDLE frameReady = canAcceptFrame ? capture.FrameReadyEvent() : nullptr;
                if (frameReady && waitCount < waitHandles.size())
                {
                    waitHandles[waitCount++] = frameReady;
                }

                const DWORD timeoutMs = canAcceptFrame && !frameReady ? 1u : INFINITE;
                const DWORD waitResult = WaitForMultipleObjectsEx(
                    static_cast<DWORD>(waitCount),
                    waitHandles.data(),
                    FALSE,
                    timeoutMs,
                    FALSE);
                if (waitResult >= WAIT_OBJECT_0 &&
                    waitResult < WAIT_OBJECT_0 + static_cast<DWORD>(waitCount))
                {
                    presenter->NotifyAsyncWaitHandleSignaled(
                        waitHandles[waitResult - WAIT_OBJECT_0]);
                }
                if (waitResult == WAIT_OBJECT_0 + static_cast<DWORD>(shutdownIndex)) break;
                if (waitResult == WAIT_OBJECT_0 + static_cast<DWORD>(commandIndex)) continue;
                if (pacingTimerIndex != static_cast<size_t>(-1) &&
                    waitResult == WAIT_OBJECT_0 + static_cast<DWORD>(pacingTimerIndex))
                {
                    presenter->NotifyPacingTimerWake();
                }
                else if (waitResult == WAIT_FAILED)
                {
                    ThrowIfFailed(
                        HRESULT_FROM_WIN32(GetLastError()),
                        "Wait for AESS render-worker event");
                }
            }

            // Presenter destruction drains D3D11/CUDA work
            // while the foreground compatibility spoof is still installed.
            presenter.reset();
        }
        catch (...)
        {
            workerError = std::current_exception();
            RequestOutputStop(output, &windowState);
            // Destroy while the HWND owner is still servicing sent lifecycle
            // messages. Signaling completion first could strand a synchronous
            // owner-thread swapchain release behind the subsequent join.
            presenter.reset();
        }
        SetEvent(doneEvent);
    }

    constexpr int kLauncherInitialWidth = 1180;
    constexpr int kLauncherInitialHeight = 850;
    constexpr int kLauncherMinWidth = 880;
    constexpr int kLauncherMinHeight = 650;

    int RunAess(HINSTANCE instance, Options options)
    {
        // Keep CPU scheduling at NORMAL so AESS cannot starve the game's
        // simulation and render-submission threads. GPU scheduling is an
        // independent WDDM control used to keep bounded inference work from
        // becoming a second-class client under a fully GPU-bound game.
        ConfigureLowLatencyGpuScheduling();

        HWND target = options.hwnd;
        if (!target && !options.titleContains.empty())
        {
            target = FindWindowByTitle(options.titleContains);
        }
        if (!target)
        {
            std::wcout << L"Switch focus to the target window. Capturing foreground window in 3 seconds...\n";
            std::this_thread::sleep_for(std::chrono::seconds(3));
            target = GetForegroundWindow();
        }
        if (!target || target == GetConsoleWindow())
        {
            throw std::runtime_error("No suitable target window selected.");
        }

        std::unique_ptr<ICaptureSource> capture;
        RequireSdrDisplay(target);
        bool usingInjectedCapture = false;
        bool usingDesktopDuplication = false;
        try
        {
            if (options.forceWgc) throw std::runtime_error("SDR demo uses compositor BGRA8 capture.");
            auto injected = std::make_unique<InjectedD3D11Capture>(target);
            capture = std::move(injected);
            usingInjectedCapture = true;
        }
        catch (const std::exception& error)
        {
            if (options.forceWgc)
                std::cout << "Capture policy: Windows Graphics Capture / BGRA8 SDR (no injection).\n";
            else
                std::cout << "Injected D3D11 capture unavailable: " << error.what() << "\n";
            try
            {
                // WGC captures the selected window while it is occluded by the
                // borderless output. Desktop Duplication cannot reliably do so:
                // once the AESS swap chain independently flips above the game,
                // the desktop can stop producing source updates.
                capture = std::make_unique<WgcWindowCapture>(target, true);
            }
            catch (const std::exception& wgcError)
            {
                std::cout << "WGC capture unavailable; falling back to DXGI SDR desktop capture: "
                    << wgcError.what() << "\n";
                capture = std::make_unique<DxgiDesktopCapture>(target);
                usingDesktopDuplication = true;
            }
        }

        std::wcout << L"Capture source: " << capture->Description() << L"\n";
        std::wcout << L"Presentation backend: D3D11 (Unified production swapchain)\n";
        if (options.rendererBackend == RendererBackend::D3D12Experimental)
        {
            std::wcout << L"Notice: --renderer d3d12-experimental exercises D3D12 shared allocation/queue interop; active swapchain presentation runs on production D3D11.\n";
        }
        std::wcout << L"Capture API: "
            << (usingInjectedCapture ? L"Injected D3D11 shared-texture ring" :
                (usingDesktopDuplication ? L"DXGI Desktop Duplication BGRA8 SDR" : L"Windows Graphics Capture"))
            << L"\n";
        std::wcout << L"Frame pacing: DXGI latency-one flip meter; interval=0; VRR/G-SYNC compatibility requested\n";
        std::wcout << L"Input resolution: detected from captured window\n";
        std::wcout << L"Upscaler: " << InfoForUpscaler(options.upscalerMode).displayName << L"\n";
        std::wcout << L"Required resampling: Magic Kernel Sharp\n";
        std::wcout << L"Color pipeline: SDR encoded Rec.709 input and BGRA8 SDR output\n";
        if (IsNeuralUpscaler(options.upscalerMode))
        {
            std::wcout << L"Pipeline: Tensor Core exact-2x neural stage\n";
            std::wcout << L"Internal output: 2x captured dimensions; direct when monitor-sized, otherwise Magic Kernel fit to presentation\n";
        }
        else
        {
            std::wcout << L"Pipeline: unified D3D11 shell with shader upscaler stage\n";
            std::wcout << L"Internal upscaler output: presentation size\n";
        }
        std::wcout << L"Debug dumps: " << (options.debugDumpsEnabled ? L"on" : L"off") << L"\n";
        std::wcout << L"Perf log: " << (options.perfLogEnabled ? L"on" : L"off");
        if (options.perfLogEnabled)
        {
            const std::filesystem::path perfPath = options.perfLogPath.empty() ? DefaultPerfLogPath() : options.perfLogPath;
            std::wcout << L" -> " << perfPath.wstring();
        }
        std::wcout << L"\n";
        if (options.benchmarkFrames > 0)
        {
            std::wcout << L"Benchmark frames: " << options.benchmarkFrames << L"\n";
        }
        std::wcout << L"Hotkey: Ctrl+Tab stops capture\n";
        std::wcout << L"Hotkey: Ctrl+Shift+Tab toggles AEISU / "
            << (options.comparisonFilter == ComparisonFilter::Bilinear ? L"bilinear" : L"nearest-neighbor") << L"\n";
        std::wcout << L"Hotkey: Ctrl+F toggles the live FPS counter\n";
        RECT outputRect = capture->OutputRect();

        WindowState windowState;
        if (IsWindow(target))
        {
            const DWORD sourceThread = GetWindowThreadProcessId(target, nullptr);
            const DWORD currentThread = GetCurrentThreadId();
            const bool attached = sourceThread && sourceThread != currentThread &&
                AttachThreadInput(currentThread, sourceThread, TRUE);
            BringWindowToTop(target);
            SetForegroundWindow(target);
            if (attached) AttachThreadInput(currentThread, sourceThread, FALSE);
        }

        DWORD targetProcessId = 0;
        DWORD foregroundProcessId = 0;
        GetWindowThreadProcessId(target, &targetProcessId);
        GetWindowThreadProcessId(GetForegroundWindow(), &foregroundProcessId);
        if (options.gsyncCompatibilityEnabled && !options.testForegroundOutput && (!targetProcessId || foregroundProcessId != targetProcessId))
        {
            throw std::runtime_error(
                "The source game could not become the foreground input owner before presentation startup.");
        }

        HWND output = CreateOutputWindow(
            instance,
            target,
            outputRect,
            windowState,
            usingDesktopDuplication,
            options.testForegroundOutput);
        ScopedOutputWindow outputLifetime(output);
        if (options.testForegroundOutput)
        {
            const DWORD sourceThread = GetWindowThreadProcessId(target, nullptr);
            const DWORD currentThread = GetCurrentThreadId();
            const bool attached = sourceThread && sourceThread != currentThread &&
                AttachThreadInput(currentThread, sourceThread, TRUE);
            BringWindowToTop(output);
            SetForegroundWindow(output);
            SetActiveWindow(output);
            if (attached) AttachThreadInput(currentThread, sourceThread, FALSE);

            DWORD outputProcessId = 0;
            GetWindowThreadProcessId(GetForegroundWindow(), &outputProcessId);
            if (outputProcessId != GetCurrentProcessId())
            {
                throw std::runtime_error(
                    "The test-only AESS output could not become the real foreground window.");
            }
            std::wcout << L"Presentation ownership: test-only real-foreground AESS output (source input intentionally inactive)\n";
        }
        else
        {
            std::wcout << L"Presentation ownership: AESS borderless no-activate HWND; source remains OS foreground\n";
        }

        foregroundProcessId = 0;
        GetWindowThreadProcessId(GetForegroundWindow(), &foregroundProcessId);
        if (options.gsyncCompatibilityEnabled && !options.testForegroundOutput && foregroundProcessId != targetProcessId)
        {
            throw std::runtime_error(
                "The no-activate AESS output unexpectedly changed the system foreground process.");
        }

        aess::presentation::GsyncForegroundSpoof foregroundSpoof;
        if (options.gsyncCompatibilityEnabled)
        {
            if (!FindNvidiaHardwareAdapter())
            {
                throw std::runtime_error(
                    "G-SYNC compatibility requires an NVIDIA hardware adapter.");
            }
            std::string hookDiagnostic;
            if (!foregroundSpoof.Enable(output, hookDiagnostic))
            {
                throw std::runtime_error(
                    "Install AESS foreground-query compatibility hook failed: " + hookDiagnostic);
            }

            if (GetForegroundWindow() != output)
            {
                throw std::runtime_error(
                    "Foreground compatibility invariant failed before DXGI initialization.");
            }
            std::wcout << L"G-SYNC compatibility: eligible (AESS queries see output; OS input remains with source)\n";
        }
        else
        {
            std::wcout << L"G-SYNC compatibility: disabled by --no-gsync-compat\n";
        }
        ScopedForegroundTransitionTracking foregroundTransitionTracking(windowState);
        FramePacingConfig framePacingConfig;
        framePacingConfig.pacingMode = options.framePacingMode;
        std::wcout << L"Frame pacing mode: "
            << aess::presentation::FramePacingModeName(options.framePacingMode)
            << L"; uncapped\n";

        auto presenter = std::make_unique<D3D11VideoPresenter>(
            output,
            options.upscalerMode,


            options.sourceColorMode,
            options.outputMode,
            options.sdrWhiteNits,
            options.perfOverlayEnabled,
            foregroundSpoof.Enabled(),
            framePacingConfig,
            options.debugViewMode,
            options.perfLogEnabled,
            options.perfLogPath,
            GetCurrentThreadId(),
            options.presentDispatchMode,
            options.comparisonFilter);
        presenter->PrimePresentation(capture->Device());
        D3D11VideoPresenter* const windowPresentDispatcher = presenter.get();
        windowState.presentationDispatcher.store(
            windowPresentDispatcher,
            std::memory_order_release);

        winrt::handle renderDone{
            CreateEventW(nullptr, TRUE, FALSE, nullptr) };
        if (!renderDone)
        {
            throw std::runtime_error("Create AESS render-worker completion event failed.");
        }

        std::exception_ptr renderError;
        std::thread renderThread(
            [&, presenter = std::move(presenter)]() mutable
            {
                RunRenderWorker(
                    output,
                    windowState,
                    *capture,
                    options,
                    std::move(presenter),
                    renderDone.get(),
                    renderError);
            });

        DWORD mainWaitError = ERROR_SUCCESS;
        std::array<HANDLE, 2> mainWaitHandles{
            renderDone.get(), options.stopEvent };
        for (;;)
        {
            const DWORD waitResult = MsgWaitForMultipleObjectsEx(
                static_cast<DWORD>(mainWaitHandles.size()),
                mainWaitHandles.data(),
                INFINITE,
                QS_ALLINPUT,
                MWMO_INPUTAVAILABLE);
            if (waitResult == WAIT_OBJECT_0)
            {
                break;
            }
            if (waitResult == WAIT_OBJECT_0 + 1)
            {
                ResetEvent(options.stopEvent);
                RequestOutputStop(output, &windowState);
                continue;
            }
            if (waitResult == WAIT_OBJECT_0 + static_cast<DWORD>(mainWaitHandles.size()))
            {
                MSG msg{};
                while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
                {
                    if (msg.message == WM_QUIT)
                    {
                        RequestOutputStop(output, &windowState);
                        continue;
                    }
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
                if (!IsWindow(output))
                {
                    RequestOutputStop(output, &windowState);
                }
                continue;
            }
            if (waitResult == WAIT_FAILED)
            {
                mainWaitError = GetLastError();
                RequestOutputStop(output, &windowState);
                // Keep dispatching sent lifecycle messages while the worker
                // drains. A plain WaitForSingleObject here can deadlock the
                // synchronous owner-thread swapchain release on an error path.
                for (;;)
                {
                    const DWORD shutdownWait = MsgWaitForMultipleObjectsEx(
                        1,
                        mainWaitHandles.data(),
                        INFINITE,
                        QS_ALLINPUT,
                        MWMO_INPUTAVAILABLE);
                    if (shutdownWait == WAIT_OBJECT_0)
                    {
                        break;
                    }
                    if (shutdownWait == WAIT_OBJECT_0 + 1)
                    {
                        MSG msg{};
                        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
                        {
                            TranslateMessage(&msg);
                            DispatchMessageW(&msg);
                        }
                        continue;
                    }
                    break;
                }
                break;
            }
        }

        if (renderThread.joinable())
        {
            renderThread.join();
        }
        windowState.presentationDispatcher.store(nullptr, std::memory_order_release);

        // The worker has destroyed the presenter and all swapchain/CUDA/VFX
        // state. Only now remove the process-local foreground spoof.
        foregroundTransitionTracking.Reset();
        foregroundSpoof.Disable();
        capture->Stop();
        outputLifetime.Reset();

        if (renderError)
        {
            std::rethrow_exception(renderError);
        }
        if (mainWaitError != ERROR_SUCCESS)
        {
            ThrowIfFailed(
                HRESULT_FROM_WIN32(mainWaitError),
                "Wait for AESS render worker");
        }
        return 0;
    }
