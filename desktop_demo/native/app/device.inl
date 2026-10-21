    void RequireSdrDisplay(HWND window)
    {
        const HMONITOR monitor = MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST);
        ComPtr<IDXGIFactory1> factory;
        ThrowIfFailed(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "Create SDR display query factory");
        for (UINT adapterIndex = 0;; ++adapterIndex)
        {
            ComPtr<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(adapterIndex, &adapter) == DXGI_ERROR_NOT_FOUND) break;
            for (UINT outputIndex = 0;; ++outputIndex)
            {
                ComPtr<IDXGIOutput> output;
                if (adapter->EnumOutputs(outputIndex, &output) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_OUTPUT_DESC description{};
                if (FAILED(output->GetDesc(&description)) || description.Monitor != monitor) continue;
                ComPtr<IDXGIOutput6> output6;
                ThrowIfFailed(output.As(&output6), "Query SDR display color state");
                DXGI_OUTPUT_DESC1 color{};
                ThrowIfFailed(output6->GetDesc1(&color), "Read SDR display color state");
                if (color.ColorSpace != DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709)
                    throw std::runtime_error("AEISU demo is SDR-only. Turn off Windows HDR/Auto HDR on the source display before starting.");
                return;
            }
        }
        throw std::runtime_error("Could not verify an SDR source display.");
    }

    ComPtr<ID3D11Device> CreateD3D11Device(IDXGIAdapter* adapter = nullptr)
    {
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT;
#if defined(_DEBUG)
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

        D3D_FEATURE_LEVEL requested[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
        };

        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        D3D_FEATURE_LEVEL created{};
        D3D_DRIVER_TYPE type = adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE;
        HRESULT hr = D3D11CreateDevice(
            adapter,
            type,
            nullptr,
            flags,
            requested,
            static_cast<UINT>(std::size(requested)),
            D3D11_SDK_VERSION,
            &device,
            &created,
            &context);

        if (FAILED(hr) && (flags & D3D11_CREATE_DEVICE_DEBUG))
        {
            flags &= ~D3D11_CREATE_DEVICE_DEBUG;
            hr = D3D11CreateDevice(
                adapter,
                type,
                nullptr,
                flags,
                requested,
                static_cast<UINT>(std::size(requested)),
                D3D11_SDK_VERSION,
                &device,
                &created,
                &context);
        }
        if (FAILED(hr) && (flags & D3D11_CREATE_DEVICE_VIDEO_SUPPORT))
        {
            flags &= ~D3D11_CREATE_DEVICE_VIDEO_SUPPORT;
            hr = D3D11CreateDevice(
                adapter,
                type,
                nullptr,
                flags,
                requested,
                static_cast<UINT>(std::size(requested)),
                D3D11_SDK_VERSION,
                &device,
                &created,
                &context);
        }

        ThrowIfFailed(hr, "D3D11CreateDevice");

#ifdef _DEBUG
        // Enable D3D11 debug layer: break on severe warnings/errors.
        // This catches SRV/UAV hazards, format mismatches, resource leaks.
        ComPtr<ID3D11Debug> debug;
        if (SUCCEEDED(device.As(&debug)))
        {
            ComPtr<ID3D11InfoQueue> infoQueue;
            if (SUCCEEDED(debug->QueryInterface(__uuidof(ID3D11InfoQueue), &infoQueue)))
            {
                // Allow specific non-critical categories to avoid noise.
                infoQueue->SetBreakOnSeverity(D3D11_SEVERITY_CORRUPTION, TRUE);
                infoQueue->SetBreakOnSeverity(D3D11_SEVERITY_ERROR, TRUE);
                infoQueue->SetBreakOnSeverity(D3D11_SEVERITY_WARNING, FALSE);

                // Suppress common non-critical warnings.
                D3D11_INFO_QUEUE_FILTER filter{};
                filter.DenyList.NumCategories = 0;
                filter.DenyList.pDenyCategories = nullptr;
                filter.DenyList.NumDescriptions = 0;
                filter.DenyList.pDenyDescriptions = nullptr;
                infoQueue->PushStorageFilter(&filter);
            }
        }
#endif

        return device;
    }

    ComPtr<IDXGIAdapter1> FindNvidiaHardwareAdapter()
    {
        ComPtr<IDXGIFactory6> factory6;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory6))))
        {
            for (UINT i = 0; ; ++i)
            {
                ComPtr<IDXGIAdapter1> adapter;
                HRESULT hr = factory6->EnumAdapterByGpuPreference(
                    i,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                    IID_PPV_ARGS(&adapter));
                if (hr == DXGI_ERROR_NOT_FOUND)
                {
                    break;
                }
                DXGI_ADAPTER_DESC1 desc{};
                adapter->GetDesc1(&desc);
                if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 && desc.VendorId == kNvidiaVendorId)
                {
                    return adapter;
                }
            }
        }

        ComPtr<IDXGIFactory1> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
        {
            return nullptr;
        }

        for (UINT i = 0; ; ++i)
        {
            ComPtr<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND)
            {
                break;
            }
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 && desc.VendorId == kNvidiaVendorId)
            {
                return adapter;
            }
        }
        return nullptr;
    }
