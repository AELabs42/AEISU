int wmain(int argc, wchar_t** argv)
{
    try
    {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        Options options;
        std::wstring stopName;
        for (int index = 1; index < argc; ++index)
        {
            const std::wstring argument = argv[index];
            if (argument == L"--self-test-stop")
            {
                const wchar_t* className = L"AEISUDemo.StopVisibilityTest";
                WNDCLASSW windowClass{};
                windowClass.lpfnWndProc = OutputWndProc;
                windowClass.hInstance = GetModuleHandleW(nullptr);
                windowClass.lpszClassName = className;
                if (!RegisterClassW(&windowClass)) throw std::runtime_error("Register stop test window failed.");
                WindowState state;
                HWND window = CreateWindowExW(WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW, className,
                    L"AEISU stop test", WS_POPUP, -30000, -30000, 16, 16, nullptr, nullptr,
                    windowClass.hInstance, &state);
                if (!window) throw std::runtime_error("Create stop test window failed.");
                ScopedOutputWindow lifetime(window);
                ShowWindow(window, SW_SHOWNOACTIVATE);
                SendMessageW(window, WM_HOTKEY, kToggleHotkeyId, 0);
                if (!state.stopRequested.load() || IsWindowVisible(window) || !IsWindow(window) ||
                    WaitForSingleObject(state.shutdownEvent, 0) != WAIT_OBJECT_0)
                    throw std::runtime_error("Hotkey must hide output without destroying its GPU owner window.");
                ShowWindow(window, SW_SHOWNOACTIVATE);
                std::thread worker([&] { RequestOutputStop(window, &state); });
                worker.join();
                MSG message{};
                while (PeekMessageW(&message, window, 0, 0, PM_REMOVE)) DispatchMessageW(&message);
                if (IsWindowVisible(window) || !IsWindow(window))
                    throw std::runtime_error("Worker stop must asynchronously hide the live output window.");
                std::cout << "Stop visibility checks passed: hotkey and worker hide output before teardown.\n";
                return 0;
            }
            if (argument == L"--self-test-sdr")
            {
                const auto gamma = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
                if (SelectSdrSourceTransfer(gamma, DXGI_FORMAT_B8G8R8A8_UNORM,
                    CaptureColorProvenance::CompositorSdr) != kModelTransferSdrGamma)
                    throw std::runtime_error("SDR selection failed.");
                bool pqRejected = false;
                bool fp16Rejected = false;
                try { SelectSdrSourceTransfer(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
                    DXGI_FORMAT_R10G10B10A2_UNORM, CaptureColorProvenance::RawSwapChain); }
                catch (const std::runtime_error&) { pqRejected = true; }
                try { SelectSdrSourceTransfer(gamma, DXGI_FORMAT_R16G16B16A16_FLOAT, CaptureColorProvenance::RawSwapChain); }
                catch (const std::runtime_error&) { fp16Rejected = true; }
                if (!pqRejected || !fp16Rejected) throw std::runtime_error("HDR rejection failed.");
                std::cout << "SDR color-policy checks passed: SDR accepted, PQ and raw FP16 rejected.\n";
                return 0;
            }
            if (argument == L"--help")
            {
                std::wcout << L"AEISU demo backend: --hwnd HANDLE | --title TEXT [--fps-counter] "
                    L"[--stop-event NAME] [--benchmark-frames N] [--comparison-filter nearest|bilinear]\n";
                return 0;
            }
            if (argument == L"--fps-counter") options.perfOverlayEnabled = true;
            else if (argument == L"--capture-wgc") options.forceWgc = true;
            else if (argument == L"--no-gsync-compat") options.gsyncCompatibilityEnabled = false;
            else if (argument == L"--comparison-filter" && index + 1 < argc)
            {
                const std::wstring filter = argv[++index];
                if (filter == L"nearest") options.comparisonFilter = ComparisonFilter::Nearest;
                else if (filter == L"bilinear") options.comparisonFilter = ComparisonFilter::Bilinear;
                else throw std::invalid_argument("Comparison filter must be nearest or bilinear.");
            }
            else if (argument == L"--hwnd" && index + 1 < argc)
                options.hwnd = reinterpret_cast<HWND>(std::stoull(argv[++index], nullptr, 0));
            else if (argument == L"--title" && index + 1 < argc)
                options.titleContains = argv[++index];
            else if (argument == L"--stop-event" && index + 1 < argc)
                stopName = argv[++index];
            else if (argument == L"--benchmark-frames" && index + 1 < argc)
                options.benchmarkFrames = std::stoull(argv[++index]);
            else if (argument == L"--perf-log" && index + 1 < argc)
            {
                options.perfLogEnabled = true;
                options.perfLogPath = argv[++index];
            }
            else throw std::invalid_argument("Unknown or incomplete argument; use --help.");
        }
        if (!options.hwnd && options.titleContains.empty())
            throw std::invalid_argument("Select a source window in the Python launcher.");
        winrt::handle stop{CreateEventW(nullptr, TRUE, FALSE, stopName.empty() ? nullptr : stopName.c_str())};
        if (!stop) throw std::runtime_error("Could not create the session stop event.");
        options.stopEvent = stop.get();
        return RunAess(GetModuleHandleW(nullptr), options);
    }
    catch (const std::exception& error)
    {
        std::cerr << "AEISU: " << error.what() << '\n';
        return 1;
    }
    catch (const winrt::hresult_error& error)
    {
        std::wcerr << L"AEISU: " << error.message().c_str() << L"\n";
        return 1;
    }
}
