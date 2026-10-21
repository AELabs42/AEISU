    class D3D11SharedFrame
    {
    public:
        explicit D3D11SharedFrame(ComPtr<ID3D11Device> device)
            : m_device(std::move(device))
        {
            m_device->GetImmediateContext(&m_context);
        }

        ~D3D11SharedFrame() = default;

        D3D11SharedFrame(const D3D11SharedFrame&) = delete;
        D3D11SharedFrame& operator=(const D3D11SharedFrame&) = delete;

        bool CopyFrom(ID3D11Texture2D* source)
        {
            D3D11_TEXTURE2D_DESC sourceDesc{};
            source->GetDesc(&sourceDesc);
            DumpCaptureFrameIfRequested(source, sourceDesc);
            RECT fullRect{
                0,
                0,
                static_cast<LONG>(sourceDesc.Width),
                static_cast<LONG>(sourceDesc.Height),
            };
            return CopyToCpu(source, sourceDesc, fullRect);
        }

        FrameInfo Info() const
        {
            return m_info;
        }

    private:
        void DumpCaptureFrameIfRequested(ID3D11Texture2D* source, const D3D11_TEXTURE2D_DESC& sourceDesc)
        {
            if (m_dumpedCapture)
            {
                return;
            }

            std::filesystem::path path;
            if (!EnvironmentPath(L"AESS_DUMP_CAPTURE", path))
            {
                return;
            }

            m_dumpedCapture = true;
            try
            {
                SaveTexture2DAsBmp(m_device.Get(), m_context.Get(), source, sourceDesc, path);
            }
            catch (...)
            {
            }
        }

        void EnsureStaging(UINT width, UINT height, DXGI_FORMAT format)
        {
            if (m_staging &&
                m_stagingWidth == width &&
                m_stagingHeight == height &&
                m_stagingFormat == format)
            {
                return;
            }

            D3D11_TEXTURE2D_DESC stagingDesc{};
            stagingDesc.Width = width;
            stagingDesc.Height = height;
            stagingDesc.MipLevels = 1;
            stagingDesc.ArraySize = 1;
            stagingDesc.Format = format;
            stagingDesc.SampleDesc.Count = 1;
            stagingDesc.BindFlags = 0;
            stagingDesc.MiscFlags = 0;
            stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            stagingDesc.Usage = D3D11_USAGE_STAGING;

            m_staging.Reset();
            ThrowIfFailed(m_device->CreateTexture2D(&stagingDesc, nullptr, &m_staging), "Create capture staging texture");
            m_stagingWidth = width;
            m_stagingHeight = height;
            m_stagingFormat = format;
        }

        bool CopyToCpu(ID3D11Texture2D* source, const D3D11_TEXTURE2D_DESC& sourceDesc, const RECT& region)
        {
            const UINT width = static_cast<UINT>(RectWidth(region));
            const UINT height = static_cast<UINT>(RectHeight(region));
            if (width == 0 || height == 0)
            {
                return false;
            }

            EnsureStaging(width, height, sourceDesc.Format);

            D3D11_BOX sourceBox{};
            sourceBox.left = static_cast<UINT>(region.left);
            sourceBox.top = static_cast<UINT>(region.top);
            sourceBox.front = 0;
            sourceBox.right = static_cast<UINT>(region.right);
            sourceBox.bottom = static_cast<UINT>(region.bottom);
            sourceBox.back = 1;
            m_context->CopySubresourceRegion(m_staging.Get(), 0, 0, 0, 0, source, 0, &sourceBox);

            D3D11_MAPPED_SUBRESOURCE mapped{};
            ThrowIfFailed(m_context->Map(m_staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map capture staging texture");

            const uint32_t bytesPerPixel = BytesPerPixel(sourceDesc.Format);
            const uint32_t rowPitch = width * bytesPerPixel;
            auto pixels = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(rowPitch) * height);
            for (UINT y = 0; y < height; ++y)
            {
                const uint8_t* src = static_cast<const uint8_t*>(mapped.pData) + static_cast<size_t>(mapped.RowPitch) * y;
                uint8_t* dst = pixels->data() + static_cast<size_t>(rowPitch) * y;
                std::memcpy(dst, src, rowPitch);
            }
            m_context->Unmap(m_staging.Get(), 0);

            m_info.sharedHandle = nullptr;
            m_info.fenceHandle = nullptr;
            m_info.fenceValue = 0;
            m_info.width = width;
            m_info.height = height;
            m_info.rowPitch = rowPitch;
            m_info.format = SharedCopyFormat(sourceDesc.Format);
            m_info.colorSpace = DefaultColorSpaceForFormat(m_info.format);
            m_info.colorProvenance = CaptureColorProvenance::CompositorSdr;
            m_info.pixels = std::move(pixels);
            m_info.version = ++m_version;
            return true;
        }

        ComPtr<ID3D11Device> m_device;
        ComPtr<ID3D11DeviceContext> m_context;
        ComPtr<ID3D11Texture2D> m_staging;
        UINT m_stagingWidth = 0;
        UINT m_stagingHeight = 0;
        DXGI_FORMAT m_stagingFormat = DXGI_FORMAT_UNKNOWN;
        FrameInfo m_info{};
        uint64_t m_version = 0;
        bool m_dumpedCapture = false;
    };

    class ICaptureSource
    {
    public:
        virtual ~ICaptureSource() = default;
        virtual bool Acquire(FrameInfo& frame) = 0;
        virtual size_t DrainSourceTimings(
            SourceTimingSample*,
            size_t,
            uint64_t& overflowCount) noexcept
        {
            overflowCount = 0;
            return 0;
        }
        virtual void Release(const FrameInfo&)
        {
        }
        virtual void Stop() noexcept = 0;
        virtual RECT OutputRect() const = 0;
        virtual std::wstring Description() const = 0;
        virtual IDXGIAdapter1* Adapter() const = 0;
        virtual ID3D11Device* Device() const = 0;
        virtual HANDLE FrameReadyEvent() const noexcept
        {
            return nullptr;
        }
        virtual void AppendRuntimeLog(std::wostream&) const
        {
        }
    };

    bool WaitForCaptureFrame(
        ICaptureSource& capture,
        FrameInfo& frame,
        std::chrono::milliseconds timeout)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (capture.Acquire(frame))
            {
                return true;
            }
            const HANDLE ready = capture.FrameReadyEvent();
            if (ready)
            {
                WaitForSingleObject(ready, 5);
            }
            else
            {
                Sleep(2);
            }
        }
        return false;
    }

    int DebugViewModeToIndex(DebugViewMode mode)
    {
        return static_cast<int>(mode);
    }

    DebugViewMode DebugViewModeFromIndex(int index)
    {
        return index == 2 ? DebugViewMode::AiModels : (index == 1 ? DebugViewMode::Upscaled : DebugViewMode::Original);
    }

    void ComputeWindowClientCrop(
        HWND hwnd,
        uint32_t captureWidth,
        uint32_t captureHeight,
        uint32_t& cropLeft,
        uint32_t& cropTop,
        uint32_t& cropWidth,
        uint32_t& cropHeight)
    {
        cropLeft = 0;
        cropTop = 0;
        cropWidth = captureWidth;
        cropHeight = captureHeight;
        if (!IsWindow(hwnd) || captureWidth == 0 || captureHeight == 0)
        {
            return;
        }

        RECT clientRect{};
        if (!GetClientRect(hwnd, &clientRect))
        {
            return;
        }
        POINT clientPoints[2] = {
            POINT{ clientRect.left, clientRect.top },
            POINT{ clientRect.right, clientRect.bottom },
        };
        SetLastError(ERROR_SUCCESS);
        if (MapWindowPoints(hwnd, nullptr, clientPoints, 2) == 0 && GetLastError() != ERROR_SUCCESS)
        {
            return;
        }

        RECT windowBounds{};
        if (!GetWindowRect(hwnd, &windowBounds))
        {
            return;
        }
        RECT extendedBounds{};
        const bool hasExtendedBounds = SUCCEEDED(DwmGetWindowAttribute(
            hwnd,
            DWMWA_EXTENDED_FRAME_BOUNDS,
            &extendedBounds,
            sizeof(extendedBounds)));

        auto mismatch = [captureWidth, captureHeight](const RECT& bounds)
        {
            const int64_t width = std::max<LONG>(0, bounds.right - bounds.left);
            const int64_t height = std::max<LONG>(0, bounds.bottom - bounds.top);
            return std::llabs(width - static_cast<int64_t>(captureWidth)) +
                std::llabs(height - static_cast<int64_t>(captureHeight));
        };
        const RECT& captureBounds =
            hasExtendedBounds && mismatch(extendedBounds) <= mismatch(windowBounds)
            ? extendedBounds
            : windowBounds;

        const int64_t boundsWidth = captureBounds.right - captureBounds.left;
        const int64_t boundsHeight = captureBounds.bottom - captureBounds.top;
        if (boundsWidth <= 0 || boundsHeight <= 0)
        {
            return;
        }

        const double scaleX = static_cast<double>(captureWidth) / static_cast<double>(boundsWidth);
        const double scaleY = static_cast<double>(captureHeight) / static_cast<double>(boundsHeight);
        const int64_t left = static_cast<int64_t>(std::llround(
            static_cast<double>(clientPoints[0].x - captureBounds.left) * scaleX));
        const int64_t top = static_cast<int64_t>(std::llround(
            static_cast<double>(clientPoints[0].y - captureBounds.top) * scaleY));
        const int64_t right = static_cast<int64_t>(std::llround(
            static_cast<double>(clientPoints[1].x - captureBounds.left) * scaleX));
        const int64_t bottom = static_cast<int64_t>(std::llround(
            static_cast<double>(clientPoints[1].y - captureBounds.top) * scaleY));

        const uint32_t clampedLeft = static_cast<uint32_t>(std::clamp<int64_t>(left, 0, captureWidth));
        const uint32_t clampedTop = static_cast<uint32_t>(std::clamp<int64_t>(top, 0, captureHeight));
        const uint32_t clampedRight = static_cast<uint32_t>(std::clamp<int64_t>(right, clampedLeft, captureWidth));
        const uint32_t clampedBottom = static_cast<uint32_t>(std::clamp<int64_t>(bottom, clampedTop, captureHeight));
        if (clampedRight <= clampedLeft || clampedBottom <= clampedTop)
        {
            return;
        }

        cropLeft = clampedLeft;
        cropTop = clampedTop;
        cropWidth = clampedRight - clampedLeft;
        cropHeight = clampedBottom - clampedTop;
    }

    class WgcWindowCapture final : public ICaptureSource
    {
    public:
        explicit WgcWindowCapture(HWND hwnd, bool gpuFrames = false)
            : m_hwnd(hwnd),
              m_gpuFrames(gpuFrames)
        {
            if (!IsWindow(hwnd))
            {
                throw std::runtime_error("Invalid HWND for WGC capture.");
            }

            ComPtr<IDXGIAdapter1> preferredAdapter = m_gpuFrames ? FindNvidiaHardwareAdapter() : nullptr;
            m_device11 = CreateD3D11Device(preferredAdapter.Get());
            if (!m_gpuFrames)
            {
                m_shared = std::make_unique<D3D11SharedFrame>(m_device11);
            }

            ComPtr<IDXGIDevice> dxgiDevice;
            ThrowIfFailed(m_device11.As(&dxgiDevice), "Query IDXGIDevice");
            ComPtr<IDXGIAdapter> adapter;
            ThrowIfFailed(dxgiDevice->GetAdapter(&adapter), "IDXGIDevice::GetAdapter");
            ThrowIfFailed(adapter.As(&m_adapter), "Query IDXGIAdapter1");

            winrt::com_ptr<IInspectable> inspectableDevice;
            ThrowIfFailed(CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.Get(), inspectableDevice.put()), "CreateDirect3D11DeviceFromDXGIDevice");
            m_winrtDevice = inspectableDevice.as<wgd11::IDirect3DDevice>();

            auto interop = winrt::get_activation_factory<wgc::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
            ThrowIfFailed(interop->CreateForWindow(
                hwnd,
                winrt::guid_of<wgc::GraphicsCaptureItem>(),
                winrt::put_abi(m_item)), "GraphicsCaptureItem::CreateForWindow");

            m_size = m_item.Size();
            if (m_size.Width <= 0 || m_size.Height <= 0)
            {
                throw std::runtime_error("WGC item has invalid size.");
            }

            m_pool = wgc::Direct3D11CaptureFramePool::CreateFreeThreaded(
                m_winrtDevice,
                kWgcCaptureFormat,
                g_wgcFramePoolSize,
                m_size);
            m_frameArrivedEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!m_frameArrivedEvent)
            {
                throw std::runtime_error("CreateEvent failed for WGC frame arrival.");
            }
            m_frameArrivedToken = m_pool.FrameArrived(
                [this](wgc::Direct3D11CaptureFramePool const& sender, auto const&) noexcept
                {
                    try
                    {
                        // CreateFreeThreaded may dispatch overlapping
                        // FrameArrived callbacks.  Arrival ordinals are atomic,
                        // but publishing callback-local batches out of lock
                        // acquisition order can otherwise append 136..138
                        // before 135 and replace the latest frame with the
                        // older one. Serialize only the producer-side drain;
                        // the render owner still touches the short latest-frame
                        // mutex solely during publication/acquisition.
                        std::scoped_lock callbackDrainLock(m_callbackDrainMutex);
                        // Drain only inside the capture callback and retain the
                        // newest frame. This keeps WGC acquisition independent
                        // from neural inference without building a latency queue.
                        wgc::Direct3D11CaptureFrame newest{ nullptr };
                        // Producer callbacks are serialized by callbackDrainLock;
                        // reuse their scratch journal rather than clearing it
                        // on every capture notification.
                        auto& callbackTimings = m_callbackTimings;
                        size_t callbackTimingWrite = 0;
                        size_t callbackTimingTotal = 0;
                        uint64_t arrival = 0;
                        for (;;)
                        {
                            auto next = sender.TryGetNextFrame();
                            if (!next)
                            {
                                break;
                            }

                            arrival = m_arrivalCount.fetch_add(
                                1,
                                std::memory_order_relaxed) + 1;
                            callbackTimings[callbackTimingWrite] = SourceTimingSample{
                                SystemRelativeTimeToQpc(next.SystemRelativeTime()),
                                arrival,
                            };
                            callbackTimingWrite =
                                (callbackTimingWrite + 1) % callbackTimings.size();
                            ++callbackTimingTotal;
                            if (newest)
                            {
                                newest.Close();
                            }
                            newest = std::move(next);
                        }
                        if (!newest)
                        {
                            return;
                        }

                        const int64_t captureReadyQpc = FramePacer::QueryQpc();
                        {
                            std::scoped_lock lock(m_latestFrameMutex);
                            if (m_latestFrame)
                            {
                                m_latestFrame.Close();
                            }
                            const size_t callbackTimingCount = std::min(
                                callbackTimingTotal,
                                callbackTimings.size());
                            const size_t callbackTimingStart =
                                callbackTimingTotal > callbackTimings.size()
                                ? callbackTimingWrite
                                : 0;
                            m_sourceTimingOverflowCount +=
                                callbackTimingTotal - callbackTimingCount;
                            for (size_t index = 0;
                                index < callbackTimingCount;
                                ++index)
                            {
                                if (m_sourceTimingCount == m_sourceTimingJournal.size())
                                {
                                    m_sourceTimingRead =
                                        (m_sourceTimingRead + 1) % m_sourceTimingJournal.size();
                                    --m_sourceTimingCount;
                                    ++m_sourceTimingOverflowCount;
                                }
                                const size_t writeIndex =
                                    (m_sourceTimingRead + m_sourceTimingCount) %
                                    m_sourceTimingJournal.size();
                                m_sourceTimingJournal[writeIndex] = callbackTimings[
                                    (callbackTimingStart + index) % callbackTimings.size()];
                                ++m_sourceTimingCount;
                            }
                            m_latestFrame = std::move(newest);
                            m_latestArrivalCount = arrival;
                            m_latestCaptureReadyQpc = captureReadyQpc;
                            if (m_frameArrivedEvent)
                            {
                                SetEvent(m_frameArrivedEvent);
                            }
                        }
                    }
                    catch (...)
                    {
                        // Capture shutdown and frame-pool recreation can race a
                        // final callback. Stop() owns error reporting/lifetime.
                    }
                });
            m_hasFrameArrivedToken = true;

            m_session = m_pool.CreateCaptureSession(m_item);
            m_session.IsCursorCaptureEnabled(false);
            if (auto session3 = m_session.try_as<wgc::IGraphicsCaptureSession3>())
            {
                session3.IsBorderRequired(false);
            }
            ComPtr<IGraphicsCaptureSession5Abi> session5;
            auto sessionInspectable = reinterpret_cast<IInspectable*>(winrt::get_abi(m_session));
            if (sessionInspectable && SUCCEEDED(sessionInspectable->QueryInterface(IID_PPV_ARGS(&session5))))
            {
                // TimeSpan is represented as signed 100-nanosecond ticks at
                // the ABI. Zero requests the maximum available update rate.
                const HRESULT intervalHr = session5->put_MinUpdateInterval(0);
                if (SUCCEEDED(intervalHr))
                {
                    std::wcout << L"WGC capture pacing: MinUpdateInterval=0 (unthrottled)\n";
                }
                else
                {
                    std::wcout << L"WGC capture pacing: OS rejected unthrottled interval, HRESULT=0x"
                        << std::hex << static_cast<unsigned long>(intervalHr) << std::dec << L"\n";
                }
            }
            else
            {
                std::wcout << L"WGC capture pacing: OS does not expose MinUpdateInterval; using maximum legacy cadence\n";
            }
            m_session.StartCapture();
        }

        ~WgcWindowCapture() override
        {
            Stop();
        }

        size_t DrainSourceTimings(
            SourceTimingSample* destination,
            size_t capacity,
            uint64_t& overflowCount) noexcept override
        {
            overflowCount = 0;
            if (!destination || capacity == 0)
            {
                return 0;
            }

            std::scoped_lock lock(m_latestFrameMutex);
            const size_t count = std::min(capacity, m_sourceTimingCount);
            for (size_t index = 0; index < count; ++index)
            {
                destination[index] = m_sourceTimingJournal[
                    (m_sourceTimingRead + index) % m_sourceTimingJournal.size()];
            }
            m_sourceTimingRead =
                (m_sourceTimingRead + count) % m_sourceTimingJournal.size();
            m_sourceTimingCount -= count;
            overflowCount = m_sourceTimingOverflowCount;
            m_sourceTimingOverflowCount = 0;
            return count;
        }

        bool Acquire(FrameInfo& frame) override
        {
            if (m_stopped || !m_pool)
            {
                return false;
            }

            wgc::Direct3D11CaptureFrame captureFrame{ nullptr };
            uint64_t arrivalCount = 0;
            int64_t captureReadyQpc = 0;
            {
                std::scoped_lock lock(m_latestFrameMutex);
                captureFrame = std::move(m_latestFrame);
                arrivalCount = m_latestArrivalCount;
                captureReadyQpc = m_latestCaptureReadyQpc;
                // A callback can publish additional ready frames after the
                // render owner drains the sideband but before this acquire.
                // This admitted newest frame becomes the cadence observation;
                // retire older journal entries now so the next loop cannot
                // report them after it and manufacture a regression.
                while (m_sourceTimingCount != 0)
                {
                    const SourceTimingSample& sample =
                        m_sourceTimingJournal[m_sourceTimingRead];
                    if (sample.sourcePresentOrdinal > arrivalCount)
                    {
                        break;
                    }
                    m_sourceTimingRead =
                        (m_sourceTimingRead + 1) % m_sourceTimingJournal.size();
                    --m_sourceTimingCount;
                    // The newest WGC frame replaces these older samples by
                    // design.  Its ordinal gap is categorized separately as
                    // capture supersession; journal overflow is reserved for
                    // timing data that was irretrievably overwritten.
                }
            }
            if (!captureFrame)
            {
                return false;
            }

            wg::SizeInt32 contentSize = captureFrame.ContentSize();
            if (contentSize.Width <= 0 || contentSize.Height <= 0)
            {
                return false;
            }

            auto surface = captureFrame.Surface();
            auto access = surface.as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();

            ComPtr<ID3D11Texture2D> texture;
            ThrowIfFailed(access->GetInterface(IID_PPV_ARGS(&texture)), "Get WGC frame texture");

            if (m_gpuFrames)
            {
                D3D11_TEXTURE2D_DESC desc{};
                texture->GetDesc(&desc);

                frame = {};
                frame.width = desc.Width;
                frame.height = desc.Height;
                frame.rowPitch = 0;
                frame.format = SharedCopyFormat(desc.Format);
                frame.colorSpace = DefaultColorSpaceForFormat(frame.format);
                frame.colorProvenance = CaptureColorProvenance::CompositorSdr;
                frame.d3d11Texture = texture;
                frame.captureLease = std::make_shared<CaptureLease>();
                frame.captureLease->wgcFrame = captureFrame;
                const uint32_t contentWidth = std::min<uint32_t>(
                    desc.Width,
                    static_cast<uint32_t>(contentSize.Width));
                const uint32_t contentHeight = std::min<uint32_t>(
                    desc.Height,
                    static_cast<uint32_t>(contentSize.Height));
                ApplyClientCrop(frame, contentWidth, contentHeight);
                frame.width = frame.cropWidth;
                frame.height = frame.cropHeight;
                frame.version = ++m_gpuVersion;
                frame.captureArrivalCount = arrivalCount;
            }
            else
            {
                m_shared->CopyFrom(texture.Get());
                frame = m_shared->Info();
                ApplyClientCrop(frame, frame.width, frame.height);
            }
            frame.sourcePresentQpc = SystemRelativeTimeToQpc(captureFrame.SystemRelativeTime());
            frame.sourcePresentOrdinal = arrivalCount;
            frame.captureReadyQpc = captureReadyQpc;

            if (contentSize.Width != m_size.Width || contentSize.Height != m_size.Height)
            {
                m_size = contentSize;
                m_pool.Recreate(
                    m_winrtDevice,
                    kWgcCaptureFormat,
                    g_wgcFramePoolSize,
                    m_size);
            }
            return true;
        }

        void Release(const FrameInfo& frame) override
        {
            if (frame.captureLease && (frame.captureLease->consumedFence || frame.captureLease->deferredOwnership))
            {
                // A submitted frame is owned by the presenter's fence-retirement
                // queue or deferred presentation path.
                return;
            }
            if (frame.captureLease)
            {
                frame.captureLease->Retire();
            }
        }

        void Stop() noexcept override
        {
            if (m_stopped)
            {
                return;
            }

            m_stopped = true;
            try
            {
                if (m_pool && m_hasFrameArrivedToken)
                {
                    m_pool.FrameArrived(m_frameArrivedToken);
                    m_hasFrameArrivedToken = false;
                }
                if (m_session)
                {
                    m_session.Close();
                }
                if (m_pool)
                {
                    m_pool.Close();
                }
            }
            catch (...)
            {
            }

            m_session = nullptr;
            m_pool = nullptr;
            m_item = nullptr;
            m_winrtDevice = nullptr;
            m_shared.reset();
            HANDLE frameArrivedEvent = nullptr;
            {
                std::scoped_lock lock(m_latestFrameMutex);
                m_latestFrame = nullptr;
                m_sourceTimingRead = 0;
                m_sourceTimingCount = 0;
                frameArrivedEvent = m_frameArrivedEvent;
                m_frameArrivedEvent = nullptr;
            }
            if (frameArrivedEvent)
            {
                CloseHandle(frameArrivedEvent);
            }
        }

        RECT OutputRect() const override
        {
            return MonitorRectFromWindow(m_hwnd);
        }

        std::wstring Description() const override
        {
            return L"WGC window: " + WindowTitle(m_hwnd);
        }

        IDXGIAdapter1* Adapter() const override
        {
            return m_adapter.Get();
        }

        ID3D11Device* Device() const override
        {
            return m_device11.Get();
        }

        HANDLE FrameReadyEvent() const noexcept override
        {
            return m_frameArrivedEvent;
        }

    private:
        void ApplyClientCrop(FrameInfo& frame, uint32_t captureWidth, uint32_t captureHeight)
        {
            const LONG_PTR style = GetWindowLongPtrW(m_hwnd, GWL_STYLE);
            const LONG_PTR exStyle = GetWindowLongPtrW(m_hwnd, GWL_EXSTYLE);
            if (captureWidth != m_cropCaptureWidth ||
                captureHeight != m_cropCaptureHeight ||
                style != m_cropWindowStyle ||
                exStyle != m_cropWindowExStyle)
            {
                ComputeWindowClientCrop(
                    m_hwnd,
                    captureWidth,
                    captureHeight,
                    m_cachedCropLeft,
                    m_cachedCropTop,
                    m_cachedCropWidth,
                    m_cachedCropHeight);
                m_cropCaptureWidth = captureWidth;
                m_cropCaptureHeight = captureHeight;
                m_cropWindowStyle = style;
                m_cropWindowExStyle = exStyle;
            }

            frame.cropLeft = m_cachedCropLeft;
            frame.cropTop = m_cachedCropTop;
            frame.cropWidth = m_cachedCropWidth;
            frame.cropHeight = m_cachedCropHeight;
        }

        HWND m_hwnd = nullptr;
        ComPtr<IDXGIAdapter1> m_adapter;
        HANDLE m_frameArrivedEvent = nullptr;
        winrt::event_token m_frameArrivedToken{};
        bool m_hasFrameArrivedToken = false;
        ComPtr<ID3D11Device> m_device11;
        std::unique_ptr<D3D11SharedFrame> m_shared;
        wgd11::IDirect3DDevice m_winrtDevice{ nullptr };
        wgc::GraphicsCaptureItem m_item{ nullptr };
        wgc::Direct3D11CaptureFramePool m_pool{ nullptr };
        wgc::GraphicsCaptureSession m_session{ nullptr };
        std::mutex m_callbackDrainMutex;
        std::array<SourceTimingSample, kSourceTimingJournalCapacity> m_callbackTimings;
        std::mutex m_latestFrameMutex;
        wgc::Direct3D11CaptureFrame m_latestFrame{ nullptr };
        std::array<SourceTimingSample, kSourceTimingJournalCapacity>
            m_sourceTimingJournal{};
        size_t m_sourceTimingRead = 0;
        size_t m_sourceTimingCount = 0;
        uint64_t m_sourceTimingOverflowCount = 0;
        uint64_t m_latestArrivalCount = 0;
        int64_t m_latestCaptureReadyQpc = 0;
        wg::SizeInt32 m_size{};
        uint64_t m_gpuVersion = 0;
        std::atomic<uint64_t> m_arrivalCount{ 0 };
        uint32_t m_cropCaptureWidth = 0;
        uint32_t m_cropCaptureHeight = 0;
        LONG_PTR m_cropWindowStyle = 0;
        LONG_PTR m_cropWindowExStyle = 0;
        uint32_t m_cachedCropLeft = 0;
        uint32_t m_cachedCropTop = 0;
        uint32_t m_cachedCropWidth = 0;
        uint32_t m_cachedCropHeight = 0;
        bool m_gpuFrames = false;
        bool m_stopped = false;
    };

    bool InjectCaptureHook(DWORD processId, const std::filesystem::path& dllPath)
    {
        HANDLE process = OpenProcess(
            PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION |
            PROCESS_VM_WRITE | PROCESS_VM_READ | PROCESS_DUP_HANDLE,
            FALSE,
            processId);
        if (!process) return false;

        const std::wstring path = std::filesystem::absolute(dllPath).wstring();
        const SIZE_T bytes = (path.size() + 1) * sizeof(wchar_t);
        void* remotePath = VirtualAllocEx(process, nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        bool injected = false;
        if (remotePath && WriteProcessMemory(process, remotePath, path.c_str(), bytes, nullptr))
        {
            HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
            const auto loadLibrary = reinterpret_cast<LPTHREAD_START_ROUTINE>(
                GetProcAddress(kernel32, "LoadLibraryW"));
            if (loadLibrary)
            {
                HANDLE thread = CreateRemoteThread(process, nullptr, 0, loadLibrary, remotePath, 0, nullptr);
                if (thread)
                {
                    injected = WaitForSingleObject(thread, 2000) == WAIT_OBJECT_0;
                    if (injected)
                    {
                        DWORD moduleResult = 0;
                        GetExitCodeThread(thread, &moduleResult);
                        injected = moduleResult != 0;
                    }
                    CloseHandle(thread);
                }
            }
        }
        if (remotePath) VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
        CloseHandle(process);
        return injected;
    }

    class InjectedD3D11Capture final : public ICaptureSource
    {
    public:
        explicit InjectedD3D11Capture(HWND hwnd)
            : m_hwnd(hwnd)
        {
            GetWindowThreadProcessId(hwnd, &m_processId);
            if (!m_processId) throw std::runtime_error("Target window has no process ID.");

            // A capture hook cannot be safely hot-replaced while its detours are
            // executing in the target. If an older AESS hook is already resident,
            // use the zero-copy WGC fallback for this activation; the versioned
            // hook becomes active after the game next starts.
            const std::array<std::wstring, 15> legacyControlNames = {
                L"Local\\AESS.Capture.V10.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V11.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V12.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V13.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V14.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V15.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V16.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V17.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V18.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V19.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V20.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V21.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V22.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V23.Control." + std::to_wstring(m_processId),
                L"Local\\AESS.Capture.V24.Control." + std::to_wstring(m_processId),
            };
            bool legacyHookResident = false;
            for (const auto& legacyControlName : legacyControlNames)
            {
                if (HANDLE legacy = OpenFileMappingW(FILE_MAP_READ, FALSE, legacyControlName.c_str()))
                {
                    CloseHandle(legacy);
                    legacyHookResident = true;
                    break;
                }
            }
            if (legacyHookResident)
            {
                throw std::runtime_error(
                    "A previous capture hook is resident; using WGC until the target next starts.");
            }

            // Create and initialize the control block before injection. This
            // gives the hook the exact selected HWND on its first Present and
            // prevents an overlay/helper swap chain from winning a startup race.
            m_mapping = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                nullptr,
                PAGE_READWRITE,
                0,
                static_cast<DWORD>(sizeof(aess::capture::SharedControl)),
                aess::capture::ControlName(m_processId).c_str());
            if (!m_mapping)
                throw std::runtime_error("Create injected capture control mapping failed.");
            const bool newMapping = GetLastError() != ERROR_ALREADY_EXISTS;
            m_control = static_cast<aess::capture::SharedControl*>(MapViewOfFile(
                m_mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(aess::capture::SharedControl)));
            if (!m_control)
                throw std::runtime_error("Map injected capture control mapping failed.");
            if (newMapping)
            {
                std::memset(m_control, 0, sizeof(*m_control));
                m_control->magic = aess::capture::kTransportMagic;
                m_control->version = aess::capture::kTransportVersion;
                m_control->producerPid = m_processId;
                m_control->slotCount = aess::capture::kTransportSlotCount;
            }
            if (m_control->magic != aess::capture::kTransportMagic ||
                m_control->version != aess::capture::kTransportVersion ||
                m_control->producerPid != m_processId)
            {
                throw std::runtime_error("Invalid swap-chain capture transport.");
            }
            InterlockedExchange64(
                &m_control->targetWindow,
                static_cast<LONG64>(reinterpret_cast<uintptr_t>(m_hwnd)));

            m_frameEvent = CreateEventW(
                nullptr, FALSE, FALSE, aess::capture::FrameEventName(m_processId).c_str());
            if (!m_frameEvent)
                throw std::runtime_error("Create injected capture frame event failed.");

            const std::filesystem::path hookPath = ExecutableDirectory() / L"AESSCaptureHookV25.dll";
            if (m_control->generation == 0 &&
                (!std::filesystem::exists(hookPath) || !InjectCaptureHook(m_processId, hookPath)))
            {
                throw std::runtime_error("The low-latency swap-chain capture hook could not be loaded.");
            }

            m_process = OpenProcess(PROCESS_DUP_HANDLE | SYNCHRONIZE, FALSE, m_processId);
            if (!m_process) throw std::runtime_error("Open target process for shared GPU fence failed.");
            const auto transportDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2250);
            while (m_control->generation == 0 && std::chrono::steady_clock::now() < transportDeadline)
            {
                if (m_frameEvent) WaitForSingleObject(m_frameEvent, 5);
                else Sleep(5);
            }
            if (m_control->generation == 0)
            {
                std::ostringstream diagnostic;
                diagnostic << "No D3D11 Present transport (stage=" << m_control->diagnostics[0]
                    << " init=" << m_control->diagnostics[1]
                    << " present=" << m_control->diagnostics[2]
                    << " present1=" << m_control->diagnostics[3]
                    << " enable=" << m_control->diagnostics[4]
                    << " execute=" << m_control->diagnostics[5]
                    << " executeEnable=" << m_control->diagnostics[6]
                    << " executeCalls=" << m_control->diagnostics[7]
                    << " presentCalls=" << m_control->diagnostics[8]
                    << " present1Calls=" << m_control->diagnostics[9]
                    << " candidateHwnd=0x" << std::hex << m_control->diagnostics[10] << std::dec
                    << " filter=" << m_control->diagnostics[11]
                    << " transport=" << m_control->diagnostics[12]
                    << " nonD3D11=" << m_control->droppedFrames << ").";
                throw std::runtime_error(diagnostic.str());
            }

            m_adapter = FindNvidiaHardwareAdapter();
            m_device = CreateD3D11Device(m_adapter.Get());
            m_device->GetImmediateContext(&m_context);
            ThrowIfFailed(m_device.As(&m_device5), "Query D3D11 device 5 for injected capture");
            ThrowIfFailed(m_context.As(&m_context4), "Query D3D11 context 4 for injected capture");
            if (!OpenGeneration(m_control->generation))
            {
                throw std::runtime_error(
                    "Open injected shared GPU texture/fence generation failed: " + m_openError);
            }
            std::wcout << L"Injected capture transport: DXGI Present -> shared GPU texture ring\n";
        }

        ~InjectedD3D11Capture() override { Stop(); }



        size_t DrainSourceTimings(
            SourceTimingSample* destination,
            size_t capacity,
            uint64_t& overflowCount) noexcept override
        {
            overflowCount = 0;
            if (m_stopped || !m_control || !destination || capacity == 0)
            {
                return 0;
            }

            const uint32_t generation = m_control->generation;
            if (!generation)
            {
                return 0;
            }
            if (generation != m_generation)
            {
                if (!TryRetirePendingReleases(false))
                {
                    return 0;
                }
                try
                {
                    if (!OpenGeneration(generation))
                    {
                        return 0;
                    }
                }
                catch (...)
                {
                    return 0;
                }
            }

            const uint64_t latest = static_cast<uint64_t>(InterlockedCompareExchange64(
                &m_control->latestSequence,
                0,
                0));
            MemoryBarrier();
            if (m_control->generation != generation || latest == 0 || !m_fence)
            {
                return 0;
            }

            const uint64_t completed = m_fence->GetCompletedValue();
            const uint64_t ready = std::min(latest, completed);
            if (ready < latest && m_frameEvent && latest > m_fenceWakeSequence &&
                SUCCEEDED(m_fence->SetEventOnCompletion(latest, m_frameEvent)))
            {
                m_fenceWakeSequence = latest;
            }
            if (ready <= m_lastTimingSequence)
            {
                return 0;
            }

            constexpr uint64_t slotCount = aess::capture::kTransportSlotCount;
            const uint64_t oldestAvailable = latest >= slotCount
                ? latest - slotCount + 1
                : 1;
            uint64_t sequence = m_lastTimingSequence + 1;
            if (sequence < oldestAvailable)
            {
                m_sourceTimingOverflowCount += oldestAvailable - sequence;
                sequence = oldestAvailable;
            }

            size_t count = 0;
            while (sequence <= ready && count < capacity)
            {
                const uint32_t slot = static_cast<uint32_t>(sequence % slotCount);
                const int64_t sourcePresentQpc = static_cast<int64_t>(
                    InterlockedCompareExchange64(
                        &m_control->sourcePresentQpc[slot],
                        0,
                        0));
                const uint64_t sourcePresentOrdinal = static_cast<uint64_t>(
                    InterlockedCompareExchange64(
                        &m_control->sourcePresentOrdinal[slot],
                        0,
                        0));
                if (sourcePresentQpc > 0)
                {
                    destination[count++] = SourceTimingSample{
                        sourcePresentQpc,
                        sourcePresentOrdinal,
                    };
                }
                m_lastTimingSequence = sequence;
                ++sequence;
            }
            overflowCount = m_sourceTimingOverflowCount;
            m_sourceTimingOverflowCount = 0;
            return count;
        }

        bool Acquire(FrameInfo& frame) override
        {
            if (m_stopped || !m_control) return false;

            const uint32_t generation = m_control->generation;
            if (!generation) return false;
            if (generation != m_generation)
            {
                if (!TryRetirePendingReleases(false) || !OpenGeneration(generation))
                {
                    return false;
                }
            }

            // Monotonically retire any completed GPU release fences to acknowledge
            // consumed slots to the producer.
            TryRetirePendingReleases(true);

            // If all transport ring slots are occupied by in-flight uncompleted
            // consumer reads, backpressure acquisition until at least one release completes.
            if (m_highestSignaledReleaseSequence >= m_lastAcknowledgedReleaseSequence + aess::capture::kTransportSlotCount)
            {
                if (m_releaseFence && m_frameEvent)
                {
                    const uint64_t neededWake = m_highestSignaledReleaseSequence - aess::capture::kTransportSlotCount + 1;
                    if (neededWake > m_releaseWakeSequence &&
                        SUCCEEDED(m_releaseFence->SetEventOnCompletion(neededWake, m_frameEvent)))
                    {
                        m_releaseWakeSequence = neededWake;
                    }
                }
                return false;
            }

            // On x64 this aligned volatile 64-bit load is atomic; a compiler barrier keeps
            // the subsequent generation/metadata reads after the sequence.
            const uint64_t latest = static_cast<uint64_t>(InterlockedCompareExchange64(
                &m_control->latestSequence,
                0,
                0));
            MemoryBarrier();
            if (!latest || latest <= m_lastSequence) return false;

            // The producer publishes latestSequence when its D3D11 copy is
            // submitted, not when that copy has reached the GPU.  Returning
            // that texture immediately pushes the cross-process fence wait
            // into cudaGraphicsMapResources, which blocks the render thread
            // for an entire game frame.  Only expose fence-complete slots;
            // this turns the dependency into a one-frame pipeline delay and
            // keeps CUDA submission non-blocking.
            const uint64_t completed = m_fence ? m_fence->GetCompletedValue() : 0;
            const uint64_t ready = std::min(latest, completed);
            if (ready <= m_lastSequence)
            {
                // The producer's event means its copy was submitted, not that
                // the shared fence has completed. Waiting on that already-
                // consumed producer event made the main loop fall through only
                // on its 100 ms timeout (or an unrelated mouse message), which
                // produced the observed 10 FPS / 200 FPS-while-moving split.
                // Arm the same auto-reset event from the GPU fence so the loop
                // wakes exactly when this texture becomes consumable.
                if (m_fence && m_frameEvent && latest > m_fenceWakeSequence &&
                    SUCCEEDED(m_fence->SetEventOnCompletion(latest, m_frameEvent)))
                {
                    m_fenceWakeSequence = latest;
                }
                return false;
            }

            // Always consume the newest fence-complete source present. When
            // neural processing falls behind, walking the ring from its oldest
            // surviving entry leaves that texture one producer frame away from
            // overwrite and can expose alternating future/stale contents. The
            // newest entry has the full ring lifetime, minimizes latency, and
            // preserves every frame whenever the consumer is keeping up.
            const uint64_t sequence = ready;
            if (sequence > m_lastTimingSequence)
            {
                // The shared fence can advance between the sideband snapshot
                // and this acquisition.  Once the newest completed texture is
                // admitted, publishing older timing slots on the next loop
                // would rewind the source cadence (for example 136 then 131).
                // Retire those normally superseded journal entries now.  The
                // admitted frame carries the exact replacement timestamp and
                // its ordinal gap is categorized as capture supersession;
                // only an actual transport-ring overwrite is journal overflow.
                m_lastTimingSequence = sequence;
            }
            const uint32_t slot = static_cast<uint32_t>(sequence % aess::capture::kTransportSlotCount);
            m_context4->Wait(m_fence.Get(), sequence);
            m_lastSequence = sequence;
            m_fenceWakeSequence = 0;
            frame = {};
            frame.width = m_width;
            frame.height = m_height;
            frame.format = SharedCopyFormat(m_format);
            frame.colorSpace = DefaultColorSpaceForFormat(frame.format);
            frame.colorProvenance = CaptureColorProvenance::RawSwapChain;
            DXGI_COLOR_SPACE_TYPE reportedColorSpace{};
            if (aess::capture::DecodeColorSpaceDiagnostic(
                m_control->diagnostics[0], reportedColorSpace))
            {
                frame.colorSpace = reportedColorSpace;
            }
            
            frame.d3d11Texture = m_textures[slot];
            frame.d3d11Srv = m_textureSrvs[slot];
            frame.cropWidth = m_width;
            frame.cropHeight = m_height;
            frame.version = sequence;
            frame.captureArrivalCount = static_cast<uint64_t>(InterlockedCompareExchange64(
                &m_control->sourcePresentFrames, 0, 0));
            frame.sourcePresentQpc = static_cast<int64_t>(InterlockedCompareExchange64(
                &m_control->sourcePresentQpc[slot], 0, 0));
            frame.sourcePresentOrdinal = static_cast<uint64_t>(InterlockedCompareExchange64(
                &m_control->sourcePresentOrdinal[slot], 0, 0));
            frame.captureReadyQpc = FramePacer::QueryQpc();
            auto lease = std::make_shared<CaptureLease>();
            const uint64_t releaseSequence = sequence;
            lease->onRetire = [this, releaseSequence]() {
                this->SignalRelease(releaseSequence);
            };
            frame.captureLease = std::move(lease);
            return true;
        }

        void SignalRelease(uint64_t sequence)
        {
            if (m_stopped || !m_control || sequence == 0)
            {
                return;
            }
            if (m_releaseFence && sequence > m_highestSignaledReleaseSequence)
            {
                ThrowIfFailed(
                    m_context4->Signal(m_releaseFence.Get(), sequence),
                    "Signal injected capture release fence");
                m_highestSignaledReleaseSequence = sequence;
            }
            TryRetirePendingReleases(true);
        }

        void Release(const FrameInfo& frame) override
        {
            if (frame.captureLease && (frame.captureLease->consumedFence || frame.captureLease->deferredOwnership))
            {
                // A submitted frame is owned by the presenter's fence-retirement
                // queue or deferred presentation path.
                return;
            }
            if (frame.captureLease)
            {
                frame.captureLease->Retire();
                return;
            }
            SignalRelease(frame.version);
        }

        void Stop() noexcept override
        {
            if (m_stopped) return;
            m_stopped = true;
            for (auto& srv : m_textureSrvs) srv.Reset();
            for (auto& texture : m_textures) texture.Reset();
            m_fence.Reset();
            m_releaseFence.Reset();
            if (m_control) UnmapViewOfFile(m_control);
            m_control = nullptr;
            if (m_mapping) CloseHandle(m_mapping);
            if (m_frameEvent) CloseHandle(m_frameEvent);
            if (m_process) CloseHandle(m_process);
            m_mapping = m_frameEvent = m_process = nullptr;
        }

        HANDLE FrameReadyEvent() const noexcept override { return m_frameEvent; }
        void AppendRuntimeLog(std::wostream& stream) const override
        {
            if (!m_control) return;
            stream << L" source_presents=" << m_control->sourcePresentFrames
                << L" hook_captured=" << m_control->capturedFrames
                << L" hook_backpressured=" << m_control->backpressuredFrames
                << L" hook_dropped=" << m_control->droppedFrames;
        }
        RECT OutputRect() const override
        {
            MONITORINFO info{ sizeof(info) };
            GetMonitorInfoW(MonitorFromWindow(m_hwnd, MONITOR_DEFAULTTONEAREST), &info);
            return info.rcMonitor;
        }
        std::wstring Description() const override
        {
            return L"Injected D3D11 swap-chain capture: " + WindowTitle(m_hwnd);
        }
        IDXGIAdapter1* Adapter() const override { return m_adapter.Get(); }
        ID3D11Device* Device() const override { return m_device.Get(); }

    private:
        bool TryRetirePendingReleases(bool acknowledgeProducer) noexcept
        {
            if (m_highestSignaledReleaseSequence <= m_lastAcknowledgedReleaseSequence)
            {
                return true;
            }
            if (!m_releaseFence)
            {
                return false;
            }
            const uint64_t completed = m_releaseFence->GetCompletedValue();
            const uint64_t retiring = std::min(m_highestSignaledReleaseSequence, completed);
            if (retiring > m_lastAcknowledgedReleaseSequence)
            {
                if (acknowledgeProducer && m_control)
                {
                    InterlockedExchange64(
                        &m_control->consumedSequence,
                        static_cast<LONG64>(retiring));
                }
                m_lastAcknowledgedReleaseSequence = retiring;
            }

            if (m_highestSignaledReleaseSequence > m_lastAcknowledgedReleaseSequence)
            {
                if (m_frameEvent && m_highestSignaledReleaseSequence > m_releaseWakeSequence &&
                    SUCCEEDED(m_releaseFence->SetEventOnCompletion(
                        m_highestSignaledReleaseSequence,
                        m_frameEvent)))
                {
                    m_releaseWakeSequence = m_highestSignaledReleaseSequence;
                }
                return false;
            }
            m_releaseWakeSequence = 0;
            return true;
        }

        bool OpenGeneration(uint32_t generation)
        {
            m_openError.clear();
            if (m_control->generation != generation || !m_control->fenceHandle)
            {
                m_openError = "incomplete generation metadata";
                return false;
            }
            HANDLE duplicatedFence = nullptr;
            if (!DuplicateHandle(
                m_process,
                reinterpret_cast<HANDLE>(m_control->fenceHandle),
                GetCurrentProcess(),
                &duplicatedFence,
                0,
                FALSE,
                DUPLICATE_SAME_ACCESS))
            {
                m_openError = "DuplicateHandle(fence) Win32=" + std::to_string(GetLastError());
                return false;
            }
            ComPtr<ID3D11Fence> fence;
            const HRESULT fenceHr = m_device5->OpenSharedFence(
                duplicatedFence, IID_PPV_ARGS(&fence));
            CloseHandle(duplicatedFence);
            if (FAILED(fenceHr))
            {
                m_openError = "OpenSharedFence HRESULT=" + std::to_string(static_cast<int64_t>(fenceHr));
                return false;
            }

            std::array<ComPtr<ID3D11Texture2D>, aess::capture::kTransportSlotCount> textures;
            std::array<ComPtr<ID3D11ShaderResourceView>, aess::capture::kTransportSlotCount> textureSrvs;
            for (uint32_t slot = 0; slot < aess::capture::kTransportSlotCount; ++slot)
            {
                if (!m_control->textureHandles[slot])
                {
                    m_openError = "missing shared texture handle " + std::to_string(slot);
                    return false;
                }
                HRESULT textureHr = E_FAIL;
                if (m_control->diagnostics[13] == 1)
                {
                    textureHr = m_device->OpenSharedResource(
                        reinterpret_cast<HANDLE>(m_control->textureHandles[slot]),
                        IID_PPV_ARGS(&textures[slot]));
                }
                else
                {
                    HANDLE duplicatedTexture = nullptr;
                    if (!DuplicateHandle(
                        m_process,
                        reinterpret_cast<HANDLE>(m_control->textureHandles[slot]),
                        GetCurrentProcess(),
                        &duplicatedTexture,
                        0,
                        FALSE,
                        DUPLICATE_SAME_ACCESS))
                    {
                        m_openError = "DuplicateHandle(texture " + std::to_string(slot) +
                            ") Win32=" + std::to_string(GetLastError());
                        return false;
                    }
                    textureHr = m_device5->OpenSharedResource1(
                        duplicatedTexture, IID_PPV_ARGS(&textures[slot]));
                    CloseHandle(duplicatedTexture);
                }
                if (FAILED(textureHr))
                {
                    m_openError = "OpenSharedResource1(texture " + std::to_string(slot) +
                        ") HRESULT=" + std::to_string(static_cast<int64_t>(textureHr));
                    return false;
                }
                // D3D11-produced transport textures are shader-resource
                // capable. Some D3D12-opened formats/resources are not; keep
                // that case non-fatal so the presenter's proven copy fallback
                // remains authoritative.
                (void)m_device->CreateShaderResourceView(
                    textures[slot].Get(), nullptr, &textureSrvs[slot]);
            }

            ComPtr<ID3D11Fence> releaseFence;
            const HRESULT releaseFenceHr = m_device5->CreateFence(
                0, D3D11_FENCE_FLAG_NONE, IID_PPV_ARGS(&releaseFence));
            if (FAILED(releaseFenceHr))
            {
                m_openError = "CreateFence(capture release) HRESULT=" +
                    std::to_string(static_cast<int64_t>(releaseFenceHr));
                return false;
            }
            m_fence = fence;
            m_textures = std::move(textures);
            m_textureSrvs = std::move(textureSrvs);
            m_releaseFence = std::move(releaseFence);
            m_generation = generation;
            m_width = m_control->width;
            m_height = m_control->height;
            m_format = static_cast<DXGI_FORMAT>(m_control->format);
            m_lastSequence = 0;
            m_lastTimingSequence = 0;
            m_fenceWakeSequence = 0;
            m_highestSignaledReleaseSequence = 0;
            m_lastAcknowledgedReleaseSequence = 0;
            m_releaseWakeSequence = 0;
            return m_width != 0 && m_height != 0;
        }

        HWND m_hwnd = nullptr;
        DWORD m_processId = 0;
        HANDLE m_process = nullptr;
        HANDLE m_mapping = nullptr;
        HANDLE m_frameEvent = nullptr;
        aess::capture::SharedControl* m_control = nullptr;
        ComPtr<IDXGIAdapter1> m_adapter;
        ComPtr<ID3D11Device> m_device;
        ComPtr<ID3D11Device5> m_device5;
        ComPtr<ID3D11DeviceContext> m_context;
        ComPtr<ID3D11DeviceContext4> m_context4;
        ComPtr<ID3D11Fence> m_fence;
        ComPtr<ID3D11Fence> m_releaseFence;
        std::array<ComPtr<ID3D11Texture2D>, aess::capture::kTransportSlotCount> m_textures;
        std::array<ComPtr<ID3D11ShaderResourceView>, aess::capture::kTransportSlotCount> m_textureSrvs;
        uint32_t m_generation = 0;
        UINT m_width = 0;
        UINT m_height = 0;
        DXGI_FORMAT m_format = DXGI_FORMAT_UNKNOWN;




        uint64_t m_lastSequence = 0;
        uint64_t m_lastTimingSequence = 0;
        uint64_t m_sourceTimingOverflowCount = 0;
        uint64_t m_fenceWakeSequence = 0;
        uint64_t m_highestSignaledReleaseSequence = 0;
        uint64_t m_lastAcknowledgedReleaseSequence = 0;
        uint64_t m_releaseWakeSequence = 0;
        std::string m_openError;
        bool m_stopped = false;
    };

    class DxgiDesktopCapture final : public ICaptureSource
    {
        static constexpr UINT kSignatureWidth = 64;
        static constexpr UINT kSignatureHeight = 36;
        static constexpr UINT kSignatureValueCount = kSignatureWidth * kSignatureHeight;
        static constexpr size_t kCaptureSlotCount = 4;
        // A fixed, generously sized dirty-rectangle scratch buffer keeps the
        // Desktop Duplication fallback allocation-free after construction.
        // If DXGI ever reports more rectangles, processing the frame
        // conservatively is cheaper and safer than allocating in Acquire.
        static constexpr size_t kDirtyRectCapacity = 16 * 1024;

        struct CaptureSlot
        {
            ComPtr<ID3D11Texture2D> texture;
            ComPtr<ID3D11ShaderResourceView> srv;
            ComPtr<ID3D11Buffer> signature;
            ComPtr<ID3D11UnorderedAccessView> signatureUav;
            ComPtr<ID3D11Buffer> signatureReadback;
            ComPtr<ID3D11Query> completion;
            uint64_t sequence = 0;
            uint64_t reusableAfterSequence = 0;
            int64_t sourcePresentQpc = 0;
            uint64_t sourcePresentOrdinal = 0;
            int64_t captureReadyQpc = 0;
            bool pending = false;
        };

    public:
        explicit DxgiDesktopCapture(HWND hwnd)
            : m_hwnd(hwnd)
        {
            if (!IsWindow(hwnd))
            {
                throw std::runtime_error("Invalid HWND for DXGI desktop capture.");
            }

            m_adapter = FindNvidiaHardwareAdapter();
            if (!m_adapter)
            {
                throw std::runtime_error("No NVIDIA adapter is available for desktop capture.");
            }
            m_device = CreateD3D11Device(m_adapter.Get());
            m_device->GetImmediateContext(&m_context);
            m_enableDiagnosticSignature = (GetEnvironmentVariableW(L"AESS_DXGI_CONTENT_SIGNATURE", nullptr, 0) > 0);
            if (m_enableDiagnosticSignature)
            {
                CreateContentSignatureResources();
            }

            const HMONITOR targetMonitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
            for (UINT index = 0;; ++index)
            {
                ComPtr<IDXGIOutput> candidate;
                const HRESULT enumHr = m_adapter->EnumOutputs(index, &candidate);
                if (enumHr == DXGI_ERROR_NOT_FOUND)
                {
                    break;
                }
                ThrowIfFailed(enumHr, "Enumerate DXGI outputs");
                DXGI_OUTPUT_DESC candidateDesc{};
                ThrowIfFailed(candidate->GetDesc(&candidateDesc), "Get DXGI output description");
                if (candidateDesc.Monitor == targetMonitor)
                {
                    ThrowIfFailed(candidate.As(&m_output5), "Query IDXGIOutput5");
                    m_outputDesc = candidateDesc;
                    break;
                }
            }
            if (!m_output5)
            {
                throw std::runtime_error("The target window is not on an NVIDIA DXGI output.");
            }
            CreateDuplication();
        }

        ~DxgiDesktopCapture() override
        {
            Stop();
        }

        bool Acquire(FrameInfo& frame) override
        {
            if (m_stopped || !m_duplication)
            {
                return false;
            }

            if (m_enableDiagnosticSignature)
            {
                // Retire signature work before asking duplication for another
                // desktop image.  This path never waits: a tiny query/readback
                // remains one or more capture frames behind the GPU producer.
                if (TryAcquireChangedSlot(frame))
                {
                    return true;
                }
            }

            CaptureSlot* slot = FreeCaptureSlot();
            if (!slot)
            {
                return false;
            }

            DXGI_OUTDUPL_FRAME_INFO info{};
            ComPtr<IDXGIResource> resource;
            HRESULT hr = m_duplication->AcquireNextFrame(0, &info, &resource);
            if (hr == DXGI_ERROR_WAIT_TIMEOUT)
            {
                return false;
            }
            if (hr == DXGI_ERROR_ACCESS_LOST)
            {
                CreateDuplication();
                return false;
            }
            ThrowIfFailed(hr, "Acquire DXGI desktop frame");

            struct ReleaseFrameGuard
            {
                IDXGIOutputDuplication* duplication = nullptr;
                ~ReleaseFrameGuard()
                {
                    if (duplication) duplication->ReleaseFrame();
                }
            } release{ m_duplication.Get() };

            ComPtr<ID3D11Texture2D> desktopTexture;
            ThrowIfFailed(resource.As(&desktopTexture), "Query duplicated desktop texture");
            D3D11_TEXTURE2D_DESC desktopDesc{};
            desktopTexture->GetDesc(&desktopDesc);
            if (desktopDesc.Format != kSdrCaptureFormat)
            {
                throw std::runtime_error("DXGI desktop duplication did not provide BGRA8 SDR.");
            }

            RECT client{};
            if (!GetClientRect(m_hwnd, &client))
            {
                throw std::runtime_error("GetClientRect failed for desktop capture.");
            }
            POINT clientPoints[2] = {
                POINT{ client.left, client.top },
                POINT{ client.right, client.bottom },
            };
            if (!ClientToScreen(m_hwnd, &clientPoints[0]) ||
                !ClientToScreen(m_hwnd, &clientPoints[1]))
            {
                throw std::runtime_error("ClientToScreen failed for desktop capture.");
            }

            const LONG captureLeft = std::clamp(
                clientPoints[0].x,
                m_outputDesc.DesktopCoordinates.left,
                m_outputDesc.DesktopCoordinates.right);
            const LONG captureTop = std::clamp(
                clientPoints[0].y,
                m_outputDesc.DesktopCoordinates.top,
                m_outputDesc.DesktopCoordinates.bottom);
            const LONG captureRight = std::clamp(
                clientPoints[1].x,
                captureLeft,
                m_outputDesc.DesktopCoordinates.right);
            const LONG captureBottom = std::clamp(
                clientPoints[1].y,
                captureTop,
                m_outputDesc.DesktopCoordinates.bottom);
            if (captureRight <= captureLeft || captureBottom <= captureTop)
            {
                return false;
            }

            const UINT width = static_cast<UINT>(captureRight - captureLeft);
            const UINT height = static_cast<UINT>(captureBottom - captureTop);

            // Output duplication advances at desktop refresh even when the
            // target game did not produce a new visible image.  Intersect its
            // dirty and move metadata with the target client before spending
            // any D3D or tensor-core work, so repeated monitor frames are not
            // counted as unique inference.
            const RECT targetRect{
                captureLeft - m_outputDesc.DesktopCoordinates.left,
                captureTop - m_outputDesc.DesktopCoordinates.top,
                captureRight - m_outputDesc.DesktopCoordinates.left,
                captureBottom - m_outputDesc.DesktopCoordinates.top,
            };

            UINT dirtyBytes = 0;
            HRESULT dirtyHr = m_duplication->GetFrameDirtyRects(
                static_cast<UINT>(sizeof(m_dirtyRects)),
                m_dirtyRects.data(),
                &dirtyBytes);
            bool targetChanged = dirtyHr != S_OK || dirtyBytes == 0;
            if (SUCCEEDED(dirtyHr) && dirtyBytes >= sizeof(RECT))
            {
                targetChanged = false;
                const size_t dirtyCount = std::min<size_t>(
                    dirtyBytes / sizeof(RECT),
                    m_dirtyRects.size());
                for (size_t index = 0; index < dirtyCount; ++index)
                {
                    RECT intersection{};
                    if (IntersectRect(
                            &intersection,
                            &targetRect,
                            &m_dirtyRects[index]))
                    {
                        targetChanged = true;
                        break;
                    }
                }
            }

            if (!targetChanged)
            {
                UINT moveBytes = 0;
                HRESULT moveHr = m_duplication->GetFrameMoveRects(
                    static_cast<UINT>(sizeof(m_moveRects)),
                    m_moveRects.data(),
                    &moveBytes);
                if (SUCCEEDED(moveHr) && moveBytes >= sizeof(DXGI_OUTDUPL_MOVE_RECT))
                {
                    const size_t moveCount = std::min<size_t>(
                        moveBytes / sizeof(DXGI_OUTDUPL_MOVE_RECT),
                        m_moveRects.size());
                    for (size_t index = 0; index < moveCount; ++index)
                    {
                        RECT intersection{};
                        if (IntersectRect(
                                &intersection,
                                &targetRect,
                                &m_moveRects[index].DestinationRect))
                        {
                            targetChanged = true;
                            break;
                        }
                    }
                }
            }

            if (!targetChanged)
            {
                return false;
            }

            EnsureCaptureRing(width, height);

            D3D11_BOX sourceBox{};
            sourceBox.left = static_cast<UINT>(captureLeft - m_outputDesc.DesktopCoordinates.left);
            sourceBox.top = static_cast<UINT>(captureTop - m_outputDesc.DesktopCoordinates.top);
            sourceBox.front = 0;
            sourceBox.right = sourceBox.left + width;
            sourceBox.bottom = sourceBox.top + height;
            sourceBox.back = 1;
            m_context->CopySubresourceRegion(slot->texture.Get(), 0, 0, 0, 0, desktopTexture.Get(), 0, &sourceBox);

            if (m_enableDiagnosticSignature)
            {
                SubmitContentSignature(*slot, width, height);
                slot->sequence = ++m_captureSequence;
                slot->sourcePresentQpc = info.LastPresentTime.QuadPart;
                if (slot->sourcePresentQpc <= 0)
                {
                    LARGE_INTEGER now{};
                    QueryPerformanceCounter(&now);
                    slot->sourcePresentQpc = now.QuadPart;
                }
                slot->sourcePresentOrdinal = ++m_captureSubmitOrdinal;
                slot->pending = true;
                if (m_pendingSlotCount >= m_pendingSlotIndices.size())
                {
                    throw std::runtime_error(
                        "Desktop capture pending-slot ring overflowed its physical slots.");
                }
                const size_t pendingTail =
                    (m_pendingSlotHead + m_pendingSlotCount) %
                    m_pendingSlotIndices.size();
                m_pendingSlotIndices[pendingTail] =
                    static_cast<size_t>(slot - m_captureSlots.data());
                ++m_pendingSlotCount;

                return TryAcquireChangedSlot(frame);
            }

            // Direct zero-latency interactive presentation path: immediately
            // emit the acquired frame without compute dispatch, staging copy,
            // or CPU readback stalls.
            slot->sequence = ++m_captureSequence;
            slot->sourcePresentQpc = info.LastPresentTime.QuadPart;
            if (slot->sourcePresentQpc <= 0)
            {
                LARGE_INTEGER now{};
                QueryPerformanceCounter(&now);
                slot->sourcePresentQpc = now.QuadPart;
            }
            slot->sourcePresentOrdinal = ++m_captureSubmitOrdinal;
            slot->captureReadyQpc = FramePacer::QueryQpc();
            slot->reusableAfterSequence = m_captureSequence + 3;
            slot->pending = false;
            ++m_arrivalCount;

            frame = {};
            frame.width = m_captureWidth;
            frame.height = m_captureHeight;
            frame.format = SharedCopyFormat(kSdrCaptureFormat);
            frame.colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
            frame.colorProvenance = CaptureColorProvenance::CompositorSdr;
            frame.d3d11Texture = slot->texture;
            frame.d3d11Srv = slot->srv;
            frame.cropWidth = m_captureWidth;
            frame.cropHeight = m_captureHeight;
            frame.version = ++m_version;
            frame.captureArrivalCount = m_arrivalCount;
            frame.sourcePresentQpc = slot->sourcePresentQpc;
            frame.sourcePresentOrdinal = slot->sourcePresentOrdinal;
            frame.captureReadyQpc = slot->captureReadyQpc;
            return true;
        }

        void Stop() noexcept override
        {
            m_stopped = true;
            m_duplication.Reset();
            m_pendingSlotHead = 0;
            m_pendingSlotCount = 0;
            for (auto& slot : m_captureSlots)
            {
                slot = {};
            }
        }

        RECT OutputRect() const override
        {
            return m_outputDesc.DesktopCoordinates;
        }

        std::wstring Description() const override
        {
            return L"DXGI FP16 desktop duplication: " + WindowTitle(m_hwnd);
        }

        IDXGIAdapter1* Adapter() const override
        {
            return m_adapter.Get();
        }

        ID3D11Device* Device() const override
        {
            return m_device.Get();
        }

    private:
        void CreateContentSignatureResources()
        {
            static constexpr char kSignatureShader[] = R"(
Texture2D<float4> Source : register(t0);
RWStructuredBuffer<uint> Signature : register(u0);
cbuffer SignatureConstants : register(b0)
{
    uint2 SourceSize;
    uint2 Padding;
};
[numthreads(64, 1, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID)
{
    uint index = dispatchId.x;
    if (index >= 64u * 36u) return;
    uint2 grid = uint2(index % 64u, index / 64u);
    uint2 pixel = min(
        (grid * SourceSize + SourceSize / 2u) / uint2(64u, 36u),
        SourceSize - 1u);
    uint4 bits = asuint(Source.Load(int3(pixel, 0)));
    Signature[index] = bits.x * 0x9E3779B1u ^ bits.y * 0x85EBCA77u ^
        bits.z * 0xC2B2AE3Du ^ bits.w * 0x27D4EB2Fu ^ index;
}
)";
            ComPtr<ID3DBlob> bytecode;
            ComPtr<ID3DBlob> errors;
            const HRESULT compileHr = D3DCompile(
                kSignatureShader, sizeof(kSignatureShader) - 1,
                "AESS desktop content signature", nullptr, nullptr,
                "main", "cs_5_0",
                D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
                0, &bytecode, &errors);
            if (FAILED(compileHr))
            {
                const std::string message = errors
                    ? static_cast<const char*>(errors->GetBufferPointer())
                    : "D3DCompile failed";
                throw std::runtime_error("Compile desktop content signature shader: " + message);
            }
            ThrowIfFailed(
                m_device->CreateComputeShader(
                    bytecode->GetBufferPointer(), bytecode->GetBufferSize(),
                    nullptr, &m_signatureShader),
                "Create desktop content signature shader");

            D3D11_BUFFER_DESC constantsDesc{};
            constantsDesc.ByteWidth = 16;
            constantsDesc.Usage = D3D11_USAGE_DEFAULT;
            constantsDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            ThrowIfFailed(
                m_device->CreateBuffer(&constantsDesc, nullptr, &m_signatureConstants),
                "Create desktop content signature constants");
        }

        CaptureSlot* FreeCaptureSlot()
        {
            CaptureSlot* oldest = nullptr;
            for (auto& slot : m_captureSlots)
            {
                if (!slot.pending && m_captureSequence >= slot.reusableAfterSequence)
                {
                    return &slot;
                }
                if (!slot.pending && (!oldest || slot.sequence < oldest->sequence))
                {
                    oldest = &slot;
                }
            }
            if (!m_enableDiagnosticSignature && oldest)
            {
                return oldest;
            }
            return nullptr;
        }

        void SubmitContentSignature(CaptureSlot& slot, UINT width, UINT height)
        {
            const UINT constants[4] = { width, height, 0, 0 };
            m_context->UpdateSubresource(m_signatureConstants.Get(), 0, nullptr, constants, 0, 0);
            ID3D11ShaderResourceView* srv = slot.srv.Get();
            ID3D11UnorderedAccessView* uav = slot.signatureUav.Get();
            ID3D11Buffer* constantsBuffer = m_signatureConstants.Get();
            m_context->CSSetShader(m_signatureShader.Get(), nullptr, 0);
            m_context->CSSetShaderResources(0, 1, &srv);
            m_context->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
            m_context->CSSetConstantBuffers(0, 1, &constantsBuffer);
            m_context->Dispatch((kSignatureValueCount + 63) / 64, 1, 1);
            ID3D11ShaderResourceView* nullSrv = nullptr;
            ID3D11UnorderedAccessView* nullUav = nullptr;
            m_context->CSSetShaderResources(0, 1, &nullSrv);
            m_context->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
            m_context->CopyResource(slot.signatureReadback.Get(), slot.signature.Get());
            m_context->End(slot.completion.Get());
        }

        bool TryAcquireChangedSlot(FrameInfo& frame)
        {
            while (m_pendingSlotCount != 0)
            {
                CaptureSlot& slot =
                    m_captureSlots[m_pendingSlotIndices[m_pendingSlotHead]];
                const HRESULT ready = m_context->GetData(
                    slot.completion.Get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
                if (ready == S_FALSE)
                {
                    return false;
                }
                ThrowIfFailed(ready, "Poll desktop content signature");
                slot.captureReadyQpc = FramePacer::QueryQpc();

                D3D11_MAPPED_SUBRESOURCE mapped{};
                ThrowIfFailed(
                    m_context->Map(slot.signatureReadback.Get(), 0, D3D11_MAP_READ, 0, &mapped),
                    "Map desktop content signature");
                std::array<uint32_t, kSignatureValueCount> signature{};
                std::memcpy(signature.data(), mapped.pData, sizeof(signature));
                m_context->Unmap(slot.signatureReadback.Get(), 0);

                m_pendingSlotHead =
                    (m_pendingSlotHead + 1) % m_pendingSlotIndices.size();
                --m_pendingSlotCount;
                slot.pending = false;
                const bool changed = !m_haveLastSignature || signature != m_lastSignature;
                if (!changed)
                {
                    ++m_duplicateFrameCount;
                    continue;
                }

                m_lastSignature = signature;
                m_haveLastSignature = true;
                // CUDA/D3D interop consumes this texture asynchronously after
                // Acquire returns.  Keep three later desktop submissions
                // between lease and reuse, matching the original safe capture
                // ring without adding a CPU/GPU fence to the hot path.
                slot.reusableAfterSequence = m_captureSequence + 3;
                ++m_arrivalCount;
                frame = {};
                frame.width = m_captureWidth;
                frame.height = m_captureHeight;
                frame.format = SharedCopyFormat(kSdrCaptureFormat);
                frame.colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
                frame.colorProvenance = CaptureColorProvenance::CompositorSdr;
                frame.d3d11Texture = slot.texture;
                frame.d3d11Srv = slot.srv;
                frame.cropWidth = m_captureWidth;
                frame.cropHeight = m_captureHeight;
                frame.version = ++m_version;
                frame.captureArrivalCount = m_arrivalCount;
                frame.sourcePresentQpc = slot.sourcePresentQpc;
                frame.sourcePresentOrdinal = slot.sourcePresentOrdinal;
                frame.captureReadyQpc = slot.captureReadyQpc;
                return true;
            }
            return false;
        }

        void CreateDuplication()
        {
            m_duplication.Reset();
            const DXGI_FORMAT formats[] = { kSdrCaptureFormat };
            ThrowIfFailed(
                m_output5->DuplicateOutput1(
                    m_device.Get(),
                    0,
                    static_cast<UINT>(std::size(formats)),
                    formats,
                    &m_duplication),
                "Create SDR DXGI desktop duplication");
            DXGI_OUTDUPL_DESC desc{};
            m_duplication->GetDesc(&desc);
            if (desc.ModeDesc.Format != kSdrCaptureFormat)
            {
                m_duplication.Reset();
                throw std::runtime_error("SDR DXGI desktop duplication format was not selected.");
            }
            std::wcout << L"DXGI capture pacing: output duplication at desktop refresh\n";
            if (m_enableDiagnosticSignature)
            {
                std::wcout << L"DXGI capture: diagnostic content signature enabled\n";
            }
            else
            {
                std::wcout << L"DXGI capture: direct zero-latency presentation path active\n";
            }
        }

        void EnsureCaptureRing(UINT width, UINT height)
        {
            if (m_captureWidth == width && m_captureHeight == height && m_captureSlots[0].texture)
            {
                return;
            }
            m_pendingSlotHead = 0;
            m_pendingSlotCount = 0;
            m_haveLastSignature = false;
            for (auto& slot : m_captureSlots) slot = {};
            D3D11_TEXTURE2D_DESC desc{};
            desc.Width = width;
            desc.Height = height;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = kSdrCaptureFormat;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            for (auto& slot : m_captureSlots)
            {
                ThrowIfFailed(
                    m_device->CreateTexture2D(&desc, nullptr, &slot.texture),
                    "Create DXGI cropped capture texture");
                ThrowIfFailed(
                    m_device->CreateShaderResourceView(slot.texture.Get(), nullptr, &slot.srv),
                    "Create DXGI cropped capture SRV");

                if (m_enableDiagnosticSignature)
                {
                    D3D11_BUFFER_DESC signatureDesc{};
                    signatureDesc.ByteWidth = kSignatureValueCount * sizeof(uint32_t);
                    signatureDesc.Usage = D3D11_USAGE_DEFAULT;
                    signatureDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
                    signatureDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
                    signatureDesc.StructureByteStride = sizeof(uint32_t);
                    ThrowIfFailed(
                        m_device->CreateBuffer(&signatureDesc, nullptr, &slot.signature),
                        "Create desktop content signature buffer");
                    ThrowIfFailed(
                        m_device->CreateUnorderedAccessView(slot.signature.Get(), nullptr, &slot.signatureUav),
                        "Create desktop content signature UAV");

                    D3D11_BUFFER_DESC readbackDesc = signatureDesc;
                    readbackDesc.Usage = D3D11_USAGE_STAGING;
                    readbackDesc.BindFlags = 0;
                    readbackDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
                    ThrowIfFailed(
                        m_device->CreateBuffer(&readbackDesc, nullptr, &slot.signatureReadback),
                        "Create desktop content signature readback");

                    D3D11_QUERY_DESC queryDesc{};
                    queryDesc.Query = D3D11_QUERY_EVENT;
                    ThrowIfFailed(
                        m_device->CreateQuery(&queryDesc, &slot.completion),
                        "Create desktop content signature query");
                }
            }
            m_captureWidth = width;
            m_captureHeight = height;
        }

        HWND m_hwnd = nullptr;
        ComPtr<IDXGIAdapter1> m_adapter;
        ComPtr<ID3D11Device> m_device;
        ComPtr<ID3D11DeviceContext> m_context;
        ComPtr<IDXGIOutput5> m_output5;
        ComPtr<IDXGIOutputDuplication> m_duplication;
        DXGI_OUTPUT_DESC m_outputDesc{};
        ComPtr<ID3D11ComputeShader> m_signatureShader;
        ComPtr<ID3D11Buffer> m_signatureConstants;
        std::array<CaptureSlot, kCaptureSlotCount> m_captureSlots;
        std::array<size_t, kCaptureSlotCount> m_pendingSlotIndices{};
        size_t m_pendingSlotHead = 0;
        size_t m_pendingSlotCount = 0;
        std::array<uint32_t, kSignatureValueCount> m_lastSignature{};
        std::array<RECT, kDirtyRectCapacity> m_dirtyRects{};
        std::array<DXGI_OUTDUPL_MOVE_RECT, 256> m_moveRects{};
        UINT m_captureWidth = 0;
        UINT m_captureHeight = 0;
        uint64_t m_captureSequence = 0;
        uint64_t m_captureSubmitOrdinal = 0;
        uint64_t m_version = 0;
        uint64_t m_arrivalCount = 0;
        uint64_t m_duplicateFrameCount = 0;
        bool m_haveLastSignature = false;
        bool m_enableDiagnosticSignature = false;
        bool m_stopped = false;
    };

