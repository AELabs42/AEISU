
    using aess::presentation::FramePacer;
    using aess::presentation::FramePacingConfig;
    using aess::presentation::FramePacingMode;
    using aess::presentation::FrameDropReason;
    using aess::presentation::FrameTimingRecord;

    constexpr UINT kFrameCount = 2;
    // WGC capture pool sizing: default to 3 buffers to prevent capture starvation
    // while explicit input-consumed lease retirement frees buffers on preprocessing completion.
    constexpr int kDefaultWgcFramePoolSize = 3;
    static int g_wgcFramePoolSize = kDefaultWgcFramePoolSize;
    constexpr int kToggleHotkeyId = 100;
    constexpr int kUpscalerBypassHotkeyId = 101;
    constexpr int kFpsCounterHotkeyId = 102;
    constexpr uint64_t kActivationDebugDumpFrame = 30;
    constexpr DXGI_FORMAT kSdrCaptureFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
    constexpr DXGI_FORMAT kSdrOutputFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
    constexpr DXGI_FORMAT kSdrSwapChainFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
    constexpr DXGI_FORMAT kModelTextureFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
    constexpr DXGI_FORMAT kHdr10OutputFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
    // DXGI 10.0.26100 does not name this newer enum value yet; the value is
    // stable in the DXGI color-space ABI and is present in newer SDKs.
    constexpr DXGI_COLOR_SPACE_TYPE kRgbFullG10NoneP2020 =
        static_cast<DXGI_COLOR_SPACE_TYPE>(25);
    constexpr wgd::DirectXPixelFormat kWgcCaptureFormat = wgd::DirectXPixelFormat::B8G8R8A8UIntNormalized;
    constexpr UINT kNvidiaVendorId = 0x10DE;
    constexpr uint32_t kSourceFlagHdrScRgb = 1u;
    constexpr uint32_t kOutputFlagHdrScRgb = 1u;
    constexpr uint32_t kModelTransferSdrBridge = 0u;
    constexpr uint32_t kModelTransferHdr10Pq = 1u;
    constexpr uint32_t kModelTransferSdrGamma = 2u;
    constexpr uint32_t kModelPrimariesRec709 = 0u;
    constexpr uint32_t kModelPrimariesRec2020 = 1u;

    const char* WorkingColorEncodingName(uint32_t sourceTransfer) noexcept
    {
        return sourceTransfer == kModelTransferSdrGamma
            ? "gamma-rec709"
            : "pq-rec2020";
    }

    void ConfigureLowLatencyGpuScheduling()
    {
        // CUDA stream priority only orders work within AESS's CUDA context.  A
        // game is a different WDDM process, so the neural context also needs a
        // process-wide GPU scheduling class or it can receive only the small
        // amount of GPU time the game leaves idle. Persistent external-memory
        // interop keeps each realtime scheduling quantum bounded to the actual
        // neural work. CPU priority deliberately remains normal so AESS cannot
        // starve the game's simulation or render-submission threads.
        const NTSTATUS gpuPriorityStatus = D3DKMTSetProcessSchedulingPriorityClass(
            GetCurrentProcess(),
            D3DKMT_SCHEDULINGPRIORITYCLASS_REALTIME);
        D3DKMT_SCHEDULINGPRIORITYCLASS effectiveGpuPriority =
            D3DKMT_SCHEDULINGPRIORITYCLASS_NORMAL;
        const NTSTATUS queryStatus = D3DKMTGetProcessSchedulingPriorityClass(
            GetCurrentProcess(),
            &effectiveGpuPriority);
        if (gpuPriorityStatus == 0 && queryStatus == 0)
        {
            std::wcout << L"WDDM process GPU scheduling priority: "
                << (effectiveGpuPriority == D3DKMT_SCHEDULINGPRIORITYCLASS_REALTIME
                    ? L"realtime"
                    : L"requested realtime, effective class=" + std::to_wstring(static_cast<int>(effectiveGpuPriority)))
                << L"\n";
        }
        else
        {
            std::wcout << L"WDDM process GPU scheduling priority request failed, NTSTATUS=0x"
                << std::hex << static_cast<unsigned long>(gpuPriorityStatus)
                << L" query=0x" << static_cast<unsigned long>(queryStatus)
                << std::dec << L"\n";
        }
    }

    // Windows SDK 10.0.28000 exposes this as
    // Windows.Graphics.Capture.IGraphicsCaptureSession5.  Keep the ABI query
    // local so AESS still builds against its 10.0.26100 baseline and simply
    // enables unthrottled capture on operating systems that implement it.
    MIDL_INTERFACE("67C0EA62-1F85-5061-925A-239BE0AC09CB")
    IGraphicsCaptureSession5Abi : public IInspectable
    {
        virtual HRESULT STDMETHODCALLTYPE get_MinUpdateInterval(INT64* value) = 0;
        virtual HRESULT STDMETHODCALLTYPE put_MinUpdateInterval(INT64 value) = 0;
    };

    enum class UpscalerMode { NoUpscaler, AdaptiveFilter32Mixed, BilinearComparison };
    enum class ComparisonFilter { Nearest, Bilinear };

    enum class PresentationResamplePath
    {
        Direct,
        MagicKernel,
    };

    // Controls what the existing Ctrl+Shift+Tab debug toggle exposes while a
    // capture is running.  The selected upscaler remains unchanged; this only
    // changes the temporary view shown while the toggle is active.
    enum class DebugViewMode
    {
        Original,
        Upscaled,
        AiModels,
    };

    struct UpscalerInfo
    {
        UpscalerMode mode;
        const wchar_t* displayName;
        const wchar_t* cliName;
        const wchar_t* shaderPath;
    };

    // The comparison baseline and the retained exact-2x SourcePyramid runtime
    // are exposed. Optional post-processing is configured independently.
    struct UpscalerDefinition
    {
          UpscalerMode mode;
          const char* name;
          const wchar_t* displayName;
          const wchar_t* cliName;
          const wchar_t* shaderPath;  // wchar_t for filesystem paths on Windows
      };

    constexpr std::array<UpscalerDefinition, 3> g_UpscalerRegistry = {
        UpscalerDefinition{ UpscalerMode::NoUpscaler, "nearest", L"Nearest-neighbor (AEISU bypass)", L"nearest", nullptr },
        UpscalerDefinition{ UpscalerMode::AdaptiveFilter32Mixed, "aeisu", L"AEISU Mixed FP8/FP16 (exact 2x)", L"aeisu", nullptr },
        UpscalerDefinition{ UpscalerMode::BilinearComparison, "bilinear", L"Bilinear (AEISU bypass)", L"bilinear", nullptr },
    };

    constexpr size_t g_UpscalerRegistryCount = g_UpscalerRegistry.size();

    static const UpscalerDefinition* FindUpscalerDefinition(UpscalerMode mode)
     {
         for (const UpscalerDefinition& definition : g_UpscalerRegistry)
         {
             if (definition.mode == mode)
             {
                 return &definition;
             }
         }
         return nullptr;
     }

     static const UpscalerDefinition* FindUpscalerDefinitionByCliName(const wchar_t* cliName)
     {
         for (const UpscalerDefinition& definition : g_UpscalerRegistry)
         {
             if (cliName && _wcsicmp(cliName, definition.cliName) == 0)
             {
                 return &definition;
             }
         }
         return nullptr;
     }

    constexpr size_t GetUpscalerRegistryCount()
    {
        return g_UpscalerRegistryCount;
    }

    // Helper: get the const wchar_t* shader path for an UpscalerMode.
    // Returns nullptr for upscalers without shader files.
    inline const wchar_t* GetUpscalerShaderPath(UpscalerMode mode)
    {
        const UpscalerDefinition* def = FindUpscalerDefinition(mode);
        return def ? def->shaderPath : nullptr;
    }

    // Helper: get the internal name (char*) for an UpscalerMode.
    inline const char* GetUpscalerName(UpscalerMode mode)
    {
        const UpscalerDefinition* def = FindUpscalerDefinition(mode);
        return def ? def->name : "unknown";
    }

    enum class OutputMode
    {
        SdrShaderBridge,
        DirectSdr,
    };

    enum class RendererBackend
    {
        D3D11,
        D3D12Experimental,
    };

    enum class SourceColorMode { SdrGamma };

    // Color-space metadata on a raw swap-chain capture describes the source
    // pixels. WGC and Desktop Duplication instead return pixels that have
    // already passed through the Windows compositor and are encoded BGRA8 SDR
    // in this demo. Keeping that
    // provenance explicit prevents a storage-format heuristic from changing
    // the interpretation when AESS falls back between capture APIs.
    enum class CaptureColorProvenance : uint32_t
    {
        Unknown,
        RawSwapChain,
        CompositorSdr,
    };

    const char* CaptureColorProvenanceName(
        CaptureColorProvenance provenance) noexcept
    {
        switch (provenance)
        {
        case CaptureColorProvenance::RawSwapChain:
            return "raw-swapchain";
        case CaptureColorProvenance::CompositorSdr:
            return "compositor-sdr";
        case CaptureColorProvenance::Unknown:
        default:
            return "unknown";
        }
    }

    bool IsHdr10ColorSpace(DXGI_COLOR_SPACE_TYPE colorSpace) noexcept
    {
        return colorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 ||
            colorSpace == DXGI_COLOR_SPACE_RGB_STUDIO_G2084_NONE_P2020;
    }



    bool IsGammaColorSpace(DXGI_COLOR_SPACE_TYPE colorSpace) noexcept
    {
        return colorSpace == DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709 ||
            colorSpace == DXGI_COLOR_SPACE_RGB_STUDIO_G22_NONE_P709 ||
            colorSpace == DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P2020 ||
            colorSpace == DXGI_COLOR_SPACE_RGB_STUDIO_G22_NONE_P2020;
    }



    uint32_t SelectSdrSourceTransfer(DXGI_COLOR_SPACE_TYPE colorSpace, DXGI_FORMAT format,
        CaptureColorProvenance provenance)
    {
        if (colorSpace != DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709 &&
            colorSpace != DXGI_COLOR_SPACE_RGB_STUDIO_G22_NONE_P709)
            throw std::runtime_error("AEISU demo rejected non-SDR capture metadata: color_space=" +
                std::to_string(static_cast<uint32_t>(colorSpace)) + " format=" +
                std::to_string(static_cast<uint32_t>(format)) + " provenance=" +
                CaptureColorProvenanceName(provenance) + ". Disable game HDR, Auto HDR and Windows HDR.");
        if (provenance == CaptureColorProvenance::RawSwapChain && format == DXGI_FORMAT_R16G16B16A16_FLOAT)
            throw std::runtime_error("Raw FP16/scRGB capture is not supported by the SDR-only demo. Use a standard SDR swapchain.");
        return kModelTransferSdrGamma;
    }




    struct OutputModeInfo
    {
        OutputMode mode;
        const wchar_t* displayName;
        const wchar_t* cliName;
        const char* logName;
    };

    constexpr std::array<OutputModeInfo, 1> kOutputModes = {
        OutputModeInfo{ OutputMode::SdrShaderBridge, L"SDR / Rec.709", L"sdr", "sdr" },
    };
    const OutputModeInfo& InfoForOutputMode(OutputMode) { return kOutputModes[0]; }


    struct CaptureLease
    {
        wgc::Direct3D11CaptureFrame wgcFrame{ nullptr };
        std::function<void()> onRetire;
        bool deferredOwnership = false;
        ComPtr<ID3D11Fence> consumedFence;
        uint64_t consumedFenceValue = 0;

        CaptureLease() = default;
        ~CaptureLease() { Retire(); }
        CaptureLease(const CaptureLease&) = delete;
        CaptureLease& operator=(const CaptureLease&) = delete;
        CaptureLease(CaptureLease&& other) noexcept
            : wgcFrame(std::move(other.wgcFrame))
            , onRetire(std::move(other.onRetire))
            , deferredOwnership(other.deferredOwnership)
            , consumedFence(std::move(other.consumedFence))
            , consumedFenceValue(other.consumedFenceValue)
        {
            other.deferredOwnership = false;
            other.consumedFenceValue = 0;
        }
        CaptureLease& operator=(CaptureLease&& other) noexcept
        {
            if (this != &other)
            {
                Retire();
                wgcFrame = std::move(other.wgcFrame);
                onRetire = std::move(other.onRetire);
                deferredOwnership = other.deferredOwnership;
                consumedFence = std::move(other.consumedFence);
                consumedFenceValue = other.consumedFenceValue;
                other.deferredOwnership = false;
                other.consumedFenceValue = 0;
            }
            return *this;
        }

        void Retire() noexcept
        {
            if (onRetire)
            {
                try
                {
                    onRetire();
                }
                catch (...)
                {
                }
                onRetire = nullptr;
            }
            if (wgcFrame)
            {
                try
                {
                    wgcFrame.Close();
                }
                catch (...)
                {
                }
                wgcFrame = nullptr;
            }
            deferredOwnership = false;
            consumedFence.Reset();
            consumedFenceValue = 0;
        }

        bool IsConsumed() const noexcept
        {
            if (onRetire)
            {
                if (!consumedFence) return false;
                return consumedFence->GetCompletedValue() >= consumedFenceValue;
            }
            if (!wgcFrame)
            {
                return true;
            }
            if (!consumedFence)
            {
                return false;
            }
            return consumedFence->GetCompletedValue() >= consumedFenceValue;
        }
    };

    struct FrameInfo
    {
        HANDLE sharedHandle = nullptr;
        HANDLE fenceHandle = nullptr;
        uint64_t fenceValue = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t rowPitch = 0;
        DXGI_FORMAT format = DXGI_FORMAT_B8G8R8A8_UNORM;
        DXGI_COLOR_SPACE_TYPE colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        CaptureColorProvenance colorProvenance =
            CaptureColorProvenance::Unknown;
        std::shared_ptr<std::vector<uint8_t>> pixels;
        ComPtr<ID3D11Texture2D> d3d11Texture;
        // Capture sources with stable texture rings may provide the matching
        // persistent SRV. WGC deliberately leaves this null because its
        // frame-pool surfaces are leased dynamically.
        ComPtr<ID3D11ShaderResourceView> d3d11Srv;
        // The render owner retains this lease before submitting source reads.
        // Only its completed GPU fence (or a shutdown drain) returns the buffer
        // to WGC; returning from Render is not evidence of GPU completion.
        std::shared_ptr<CaptureLease> captureLease;
        uint32_t cropLeft = 0;
        uint32_t cropTop = 0;
        uint32_t cropWidth = 0;
        uint32_t cropHeight = 0;
        uint64_t version = 0;
        uint64_t captureArrivalCount = 0;
        int64_t sourcePresentQpc = 0;
        uint64_t sourcePresentOrdinal = 0;
        int64_t captureReadyQpc = 0;
        int64_t captureAcquireBeginQpc = 0;
        int64_t captureAcquireEndQpc = 0;
    };

    struct SourceTimingSample
    {
        int64_t sourcePresentQpc = 0;
        uint64_t sourcePresentOrdinal = 0;
    };

    constexpr size_t kSourceTimingJournalCapacity = 64;

    int64_t SystemRelativeTimeToQpc(winrt::Windows::Foundation::TimeSpan value) noexcept
    {
        static const int64_t frequency = []
        {
            LARGE_INTEGER result{};
            return QueryPerformanceFrequency(&result) && result.QuadPart > 0
                ? result.QuadPart
                : 1ll;
        }();
        // WinRT TimeSpan is expressed in 100-nanosecond units on the same
        // system-relative timeline as QPC. Long double prevents overflow on
        // machines with long uptimes before converting to native QPC ticks.
        return static_cast<int64_t>(std::llround(
            static_cast<long double>(value.count()) *
            static_cast<long double>(frequency) / 10000000.0L));
    }

    bool ComputeAlignedIntervalResidualMs(
        int64_t previousSourcePresentQpc,
        uint64_t previousSourcePresentOrdinal,
        int64_t previousSyncQpc,
        int64_t sourcePresentQpc,
        uint64_t sourcePresentOrdinal,
        int64_t syncQpc,
        int64_t qpcFrequency,
        double& residualMs) noexcept
    {
        residualMs = -1.0;
        if (qpcFrequency <= 0 ||
            previousSourcePresentQpc <= 0 || sourcePresentQpc <= previousSourcePresentQpc ||
            previousSyncQpc <= 0 || syncQpc <= previousSyncQpc ||
            (previousSourcePresentOrdinal > 0 && sourcePresentOrdinal > 0 &&
                sourcePresentOrdinal <= previousSourcePresentOrdinal))
        {
            return false;
        }

        const double sourceIntervalMs = static_cast<double>(
            sourcePresentQpc - previousSourcePresentQpc) * 1000.0 /
            static_cast<double>(qpcFrequency);
        const double displayIntervalMs = static_cast<double>(
            syncQpc - previousSyncQpc) * 1000.0 /
            static_cast<double>(qpcFrequency);
        // A ten-second bound rejects suspend/debugger discontinuities while
        // retaining deliberate burst/gap cadence tests.
        if (!std::isfinite(sourceIntervalMs) || !std::isfinite(displayIntervalMs) ||
            sourceIntervalMs <= 0.0 || displayIntervalMs <= 0.0 ||
            sourceIntervalMs > 10000.0 || displayIntervalMs > 10000.0)
        {
            return false;
        }

        residualMs = std::abs(displayIntervalMs - sourceIntervalMs);
        return std::isfinite(residualMs);
    }

    enum class PresentDispatchMode
    {
        Auto,
        Compute,
        Raster
    };

    struct Options
    {
        HWND hwnd = nullptr;
        HANDLE stopEvent = nullptr;
        // Compositor SDR capture avoids guessing raw 10-bit swapchain transfer functions.
        bool forceWgc = true;
        std::wstring titleContains;
        UpscalerMode upscalerMode = UpscalerMode::AdaptiveFilter32Mixed;
        ComparisonFilter comparisonFilter = ComparisonFilter::Nearest;
        SourceColorMode sourceColorMode = SourceColorMode::SdrGamma;
        OutputMode outputMode = OutputMode::SdrShaderBridge;
        float sdrWhiteNits = 80.0f;
        bool debugDumpsEnabled = false;
        bool perfLogEnabled = false;
        bool perfOverlayEnabled = false;
        bool gsyncCompatibilityEnabled = true;
        // Deterministic presentation-fixture escape hatch. Production never
        // sets this: taking foreground would intentionally transfer input
        // away from the source game.
        bool testForegroundOutput = false;
        FramePacingMode framePacingMode = FramePacingMode::Adaptive;
        DebugViewMode debugViewMode = DebugViewMode::Upscaled;
        uint64_t benchmarkFrames = 0;
        std::filesystem::path perfLogPath;
        bool validateUpscalers = false;
        int wgcPoolSize = kDefaultWgcFramePoolSize;
        bool preferMaxPerf = false;
        RendererBackend rendererBackend = RendererBackend::D3D11;
        PresentDispatchMode presentDispatchMode = PresentDispatchMode::Auto;
    };

    std::string Narrow(const std::wstring& value)
    {
        if (value.empty())
        {
            return {};
        }
        int bytes = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
        std::string output(static_cast<size_t>(bytes), '\0');
        WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), output.data(), bytes, nullptr, nullptr);
        return output;
    }



    std::string HrToHex(HRESULT hr)
    {
        std::ostringstream out;
        out << "0x" << std::hex << std::uppercase << static_cast<uint32_t>(hr);
        return out.str();
    }

    const char* FormatName(DXGI_FORMAT format)
    {
        switch (format)
        {
        case DXGI_FORMAT_B8G8R8A8_UNORM:
            return "B8G8R8A8_UNORM";
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return "B8G8R8A8_UNORM_SRGB";
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            return "R8G8B8A8_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return "R8G8B8A8_UNORM_SRGB";
        case DXGI_FORMAT_R10G10B10A2_UNORM:
            return "R10G10B10A2_UNORM";
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            return "R16G16B16A16_FLOAT";
        default:
            return "unknown";
        }
    }

    DXGI_FORMAT SharedCopyFormat(DXGI_FORMAT sourceFormat)
    {
        switch (sourceFormat)
        {
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        default:
            return sourceFormat;
        }
    }

    uint32_t BytesPerPixel(DXGI_FORMAT format)
    {
        switch (format)
        {
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_R10G10B10A2_UNORM:
            return 4;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            return 8;
        default:
            throw std::runtime_error(std::string("Unsupported capture texture format: ") + FormatName(format));
        }
    }

    uint32_t SourceFlagsFor(DXGI_FORMAT format)
    {
        switch (format)
        {
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            return 0;
        default:
            return 0;
        }
    }

    const UpscalerInfo& InfoForUpscaler(UpscalerMode mode)
    {
        const UpscalerDefinition* def = FindUpscalerDefinition(mode);
        static UpscalerInfo fallback{ UpscalerMode::NoUpscaler, L"unknown", L"unknown", nullptr };
        if (!def)
        {
            return fallback;
        }
        // Build UpscalerInfo from registry on the fly.
        // We return a reference to a static thread_local to avoid allocation.
        thread_local UpscalerInfo cached;
        cached.mode = mode;
        cached.displayName = def->displayName;
        cached.cliName = def->cliName;
        cached.shaderPath = def->shaderPath;
        return cached;
    }


    DXGI_COLOR_SPACE_TYPE DefaultColorSpaceForFormat(DXGI_FORMAT format)
    {
        switch (format)
        {
        case kModelTextureFormat:
            return DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
        case DXGI_FORMAT_R10G10B10A2_UNORM:
            return DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
        default:
            return DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        }
    }

    bool IsSourcePyramidUpscaler(UpscalerMode)
    {
        return false;
    }

    bool IsNeuralUpscaler(UpscalerMode mode) { return mode == UpscalerMode::AdaptiveFilter32Mixed; }


    aess::neural::ModelKind NeuralModelKindForMode(UpscalerMode)
    { return aess::neural::ModelKind::AdaptiveFilter32; }


    aess::neural::RuntimePrecision NeuralPrecisionForMode(UpscalerMode)
    { return aess::neural::RuntimePrecision::Mixed; }


    constexpr UINT kS2D32Scale = 2;

    UpscalerMode DebugUpscalerMode(UpscalerMode selected, DebugViewMode, bool) { return selected; }


    bool TryComputeS2D32OutputSize(UINT inputWidth, UINT inputHeight, UINT& outputWidth, UINT& outputHeight)
    {
        if (inputWidth == 0 || inputHeight == 0 ||
            inputWidth > std::numeric_limits<UINT>::max() / kS2D32Scale ||
            inputHeight > std::numeric_limits<UINT>::max() / kS2D32Scale)
        {
            return false;
        }
        outputWidth = inputWidth * kS2D32Scale;
        outputHeight = inputHeight * kS2D32Scale;
        return true;
    }

    bool IsUpscalerAvailableForInputHeight(UpscalerMode, uint32_t)
    {
        return true;
    }

    bool IsShaderBridgeMode(OutputMode) { return true; }


    int UpscalerModeToIndex(UpscalerMode mode)
    {
        for (size_t i = 0; i < g_UpscalerRegistryCount; ++i)
        {
            if (g_UpscalerRegistry[i].mode == mode)
            {
                return static_cast<int>(i);
            }
        }
        return 0;
    }

    UpscalerMode UpscalerModeFromIndex(int index)
    {
        if (index < 0 || index >= static_cast<int>(g_UpscalerRegistryCount))
        {
            return UpscalerMode::NoUpscaler;
        }
        return g_UpscalerRegistry[static_cast<size_t>(index)].mode;
    }

    bool EqualsInsensitive(const std::wstring& left, const wchar_t* right)
    {
        return CompareStringOrdinal(left.c_str(), -1, right, -1, TRUE) == CSTR_EQUAL;
    }





    bool TryParseRendererBackend(
        const std::wstring& value,
        RendererBackend& backend)
    {
        if (EqualsInsensitive(value, L"d3d11") ||
            EqualsInsensitive(value, L"dx11"))
        {
            backend = RendererBackend::D3D11;
            return true;
        }
        if (EqualsInsensitive(value, L"d3d12") ||
            EqualsInsensitive(value, L"dx12") ||
            EqualsInsensitive(value, L"d3d12-experimental") ||
            EqualsInsensitive(value, L"dx12-experimental"))
        {
            backend = RendererBackend::D3D12Experimental;
            return true;
        }
        return false;
    }








    OutputMode OutputModeFromIndex(int index)
    {
        if (index < 0 || index >= static_cast<int>(kOutputModes.size()))
        {
            return OutputMode::SdrShaderBridge;
        }
        return kOutputModes[static_cast<size_t>(index)].mode;
    }







    int FramePacingModeToIndex(FramePacingMode mode)
    {
        return static_cast<int>(mode);
    }

    FramePacingMode FramePacingModeFromIndex(int index)
    {
        switch (index)
        {
        case 0: return FramePacingMode::LowLatency;
        case 2: return FramePacingMode::Smooth;
        default: return FramePacingMode::Adaptive;
        }
    }

    bool TryParseFramePacingMode(const std::wstring& value, FramePacingMode& mode)
    {
        if (EqualsInsensitive(value, L"low-latency") || EqualsInsensitive(value, L"low"))
        {
            mode = FramePacingMode::LowLatency;
            return true;
        }
        if (EqualsInsensitive(value, L"adaptive") || EqualsInsensitive(value, L"auto"))
        {
            mode = FramePacingMode::Adaptive;
            return true;
        }
        if (EqualsInsensitive(value, L"smooth"))
        {
            mode = FramePacingMode::Smooth;
            return true;
        }
        return false;
    }

    PresentationResamplePath SelectPresentationResamplePath(
        UINT sourceWidth,
        UINT sourceHeight,
        UINT outputWidth,
        UINT outputHeight)
    {
        if (sourceWidth == outputWidth && sourceHeight == outputHeight)
        {
            return PresentationResamplePath::Direct;
        }
        return PresentationResamplePath::MagicKernel;
    }

    uint16_t ReadUInt16(const uint8_t* source)
    {
        uint16_t value = 0;
        std::memcpy(&value, source, sizeof(value));
        return value;
    }

    uint32_t ReadUInt32(const uint8_t* source)
    {
        uint32_t value = 0;
        std::memcpy(&value, source, sizeof(value));
        return value;
    }

    float HalfToFloat(uint16_t value)
    {
        const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
        int32_t exponent = static_cast<int32_t>((value >> 10) & 0x1Fu);
        uint32_t mantissa = value & 0x03FFu;
        uint32_t bits = 0;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                bits = sign;
            }
            else
            {
                exponent = 1;
                while ((mantissa & 0x0400u) == 0)
                {
                    mantissa <<= 1;
                    --exponent;
                }
                mantissa &= 0x03FFu;
                bits = sign | (static_cast<uint32_t>(exponent + (127 - 15)) << 23) | (mantissa << 13);
            }
        }
        else if (exponent == 0x1Fu)
        {
            bits = sign | 0x7F800000u | (mantissa << 13);
        }
        else
        {
            bits = sign | (static_cast<uint32_t>(exponent + (127 - 15)) << 23) | (mantissa << 13);
        }

        float result = 0.0f;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }

    uint16_t FloatToHalf(float value)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));

        const uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFFu;

        if (exponent <= 0)
        {
            if (exponent < -10)
            {
                return static_cast<uint16_t>(sign);
            }
            mantissa = (mantissa | 0x800000u) >> static_cast<uint32_t>(1 - exponent);
            return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
        }

        if (exponent >= 31)
        {
            return static_cast<uint16_t>(sign | 0x7C00u);
        }

        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | ((mantissa + 0x1000u) >> 13));
    }

    uint8_t ToByte(float value)
    {
        value = std::clamp(value, 0.0f, 1.0f);
        return static_cast<uint8_t>(value * 255.0f + 0.5f);
    }

    void ClampEncodedSdr(float& r, float& g, float& b)
    {
        r = std::clamp(r, 0.0f, 1.0f);
        g = std::clamp(g, 0.0f, 1.0f);
        b = std::clamp(b, 0.0f, 1.0f);
    }

    bool EnvironmentPath(const wchar_t* name, std::filesystem::path& path)
    {
        DWORD needed = GetEnvironmentVariableW(name, nullptr, 0);
        if (needed == 0)
        {
            return false;
        }

        std::wstring value(static_cast<size_t>(needed), L'\0');
        DWORD copied = GetEnvironmentVariableW(name, value.data(), needed);
        if (copied == 0)
        {
            return false;
        }
        value.resize(copied);
        path = value;
        return !path.empty();
    }

    float EnvironmentFloat(const wchar_t* name, float fallback, float minimum, float maximum)
    {
        wchar_t buffer[64]{};
        DWORD copied = GetEnvironmentVariableW(name, buffer, static_cast<DWORD>(std::size(buffer)));
        if (copied == 0 || copied >= std::size(buffer))
        {
            return fallback;
        }

        wchar_t* end = nullptr;
        float value = std::wcstof(buffer, &end);
        if (end == buffer || !std::isfinite(value))
        {
            return fallback;
        }
        return std::clamp(value, minimum, maximum);
    }

    bool FilenameEqualsInsensitive(const std::filesystem::path& path, const wchar_t* expected)
    {
        std::wstring name = path.filename().wstring();
        return CompareStringOrdinal(name.c_str(), -1, expected, -1, TRUE) == CSTR_EQUAL;
    }

    std::filesystem::path ExecutableDirectory()
    {
        std::wstring buffer(MAX_PATH, L'\0');
        for (;;)
        {
            DWORD copied = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
            if (copied == 0)
            {
                break;
            }
            if (copied < buffer.size() - 1)
            {
                buffer.resize(copied);
                return std::filesystem::path(buffer).parent_path();
            }
            buffer.resize(buffer.size() * 2);
        }

        try
        {
            return std::filesystem::current_path();
        }
        catch (...)
        {
            return {};
        }
    }

    std::filesystem::path DefaultDebugDumpRoot()
    {
        std::filesystem::path overrideRoot;
        if (EnvironmentPath(L"AESS_DEBUG_DUMP_DIR", overrideRoot))
        {
            return overrideRoot;
        }

        std::filesystem::path exeDir = ExecutableDirectory();
        if (!exeDir.empty() &&
            FilenameEqualsInsensitive(exeDir.parent_path(), L"bin") &&
            (FilenameEqualsInsensitive(exeDir, L"Release") || FilenameEqualsInsensitive(exeDir, L"Debug")))
        {
            return exeDir.parent_path().parent_path() / L"debug-dumps";
        }

        if (!exeDir.empty())
        {
            return exeDir / L"debug-dumps";
        }
        return L"debug-dumps";
    }

    std::filesystem::path ActivationDebugDumpDirectory()
    {
        SYSTEMTIME time{};
        GetLocalTime(&time);

        wchar_t folder[128]{};
        swprintf_s(
            folder,
            L"activation-%04u%02u%02u-%02u%02u%02u-pid%lu",
            static_cast<unsigned>(time.wYear),
            static_cast<unsigned>(time.wMonth),
            static_cast<unsigned>(time.wDay),
            static_cast<unsigned>(time.wHour),
            static_cast<unsigned>(time.wMinute),
            static_cast<unsigned>(time.wSecond),
            GetCurrentProcessId());

        return DefaultDebugDumpRoot() / folder;
    }

    std::filesystem::path DefaultPerfLogPath()
    {
        SYSTEMTIME time{};
        GetLocalTime(&time);

        wchar_t filename[128]{};
        swprintf_s(
            filename,
            L"perf-%04u%02u%02u-%02u%02u%02u-pid%lu.csv",
            static_cast<unsigned>(time.wYear),
            static_cast<unsigned>(time.wMonth),
            static_cast<unsigned>(time.wDay),
            static_cast<unsigned>(time.wHour),
            static_cast<unsigned>(time.wMinute),
            static_cast<unsigned>(time.wSecond),
            GetCurrentProcessId());

        return DefaultDebugDumpRoot() / filename;
    }

    std::filesystem::path FindProductFile(const wchar_t* filename)
    {
        std::vector<std::filesystem::path> roots;

        try
        {
            roots.push_back(std::filesystem::current_path());
        }
        catch (...)
        {
        }

        std::filesystem::path exeDir = ExecutableDirectory();
        if (!exeDir.empty())
        {
            roots.push_back(exeDir);
            roots.push_back(exeDir.parent_path());
            roots.push_back(exeDir.parent_path().parent_path());
        }

        for (const std::filesystem::path& root : roots)
        {
            if (root.empty())
            {
                continue;
            }

            std::filesystem::path candidate = root / filename;
            if (std::filesystem::exists(candidate))
            {
                return candidate;
            }
        }

        throw std::runtime_error("Required shader file is missing.");
    }

    std::string ReadTextFile(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
        {
            throw std::runtime_error("Failed to open shader file: " + path.string());
        }

        std::ostringstream text;
        text << input.rdbuf();
        return text.str();
    }

    bool SaveTexture2DAsBmp(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        ID3D11Texture2D* texture,
        const D3D11_TEXTURE2D_DESC& sourceDesc,
        const std::filesystem::path& path)
    {
        if (!device || !context || !texture || sourceDesc.Width == 0 || sourceDesc.Height == 0)
        {
            return false;
        }

        D3D11_TEXTURE2D_DESC stagingDesc = sourceDesc;
        stagingDesc.BindFlags = 0;
        stagingDesc.MiscFlags = 0;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stagingDesc.Usage = D3D11_USAGE_STAGING;

        ComPtr<ID3D11Texture2D> staging;
        if (FAILED(device->CreateTexture2D(&stagingDesc, nullptr, &staging)))
        {
            return false;
        }

        context->CopyResource(staging.Get(), texture);

        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped)))
        {
            return false;
        }

        const uint32_t bytesPerPixel = BytesPerPixel(sourceDesc.Format);
        std::vector<uint8_t> pixels(static_cast<size_t>(sourceDesc.Width) * sourceDesc.Height * 4);
        for (UINT y = 0; y < sourceDesc.Height; ++y)
        {
            const uint8_t* src = static_cast<const uint8_t*>(mapped.pData) + static_cast<size_t>(mapped.RowPitch) * y;
            uint8_t* dst = pixels.data() + static_cast<size_t>(sourceDesc.Width) * y * 4;
            for (UINT x = 0; x < sourceDesc.Width; ++x)
            {
                const uint8_t* s = src + static_cast<size_t>(x) * bytesPerPixel;
                uint8_t* d = dst + static_cast<size_t>(x) * 4;
                switch (sourceDesc.Format)
                {
                case DXGI_FORMAT_R8G8B8A8_UNORM:
                case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
                    d[0] = s[2];
                    d[1] = s[1];
                    d[2] = s[0];
                    d[3] = s[3];
                    break;
                case DXGI_FORMAT_R10G10B10A2_UNORM:
                {
                    uint32_t packed = ReadUInt32(s);
                    d[0] = ToByte(static_cast<float>((packed >> 20) & 0x3FFu) / 1023.0f);
                    d[1] = ToByte(static_cast<float>((packed >> 10) & 0x3FFu) / 1023.0f);
                    d[2] = ToByte(static_cast<float>(packed & 0x3FFu) / 1023.0f);
                    d[3] = 255;
                    break;
                }
                case DXGI_FORMAT_R16G16B16A16_FLOAT:
                {
                    float r = HalfToFloat(ReadUInt16(s));
                    float g = HalfToFloat(ReadUInt16(s + 2));
                    float b = HalfToFloat(ReadUInt16(s + 4));
                    // Handle NaN, Inf, negative scRGB, and values above 1.0
                    if (r != r) r = 0.0f;       // NaN -> 0
                    if (g != g) g = 0.0f;
                    if (b != b) b = 0.0f;
                    if (!std::isfinite(r)) r = (r > 0.0f) ? 1.0f : 0.0f;
                    if (!std::isfinite(g)) g = (g > 0.0f) ? 1.0f : 0.0f;
                    if (!std::isfinite(b)) b = (b > 0.0f) ? 1.0f : 0.0f;
                    // Clamp negative scRGB values to 0 for SDR preview
                    r = std::max(r, 0.0f);
                    g = std::max(g, 0.0f);
                    b = std::max(b, 0.0f);
                    ClampEncodedSdr(r, g, b);
                    d[0] = ToByte(b);
                    d[1] = ToByte(g);
                    d[2] = ToByte(r);
                    d[3] = 255;
                    break;
                }
                default:
                    d[0] = s[0];
                    d[1] = s[1];
                    d[2] = s[2];
                    d[3] = s[3];
                    break;
                }
            }
        }
        context->Unmap(staging.Get(), 0);

        if (!path.parent_path().empty())
        {
            std::filesystem::create_directories(path.parent_path());
        }
        std::ofstream output(path, std::ios::binary);
        if (!output)
        {
            return false;
        }

        BITMAPFILEHEADER fileHeader{};
        BITMAPINFOHEADER infoHeader{};
        infoHeader.biSize = sizeof(infoHeader);
        infoHeader.biWidth = static_cast<LONG>(sourceDesc.Width);
        infoHeader.biHeight = -static_cast<LONG>(sourceDesc.Height);
        infoHeader.biPlanes = 1;
        infoHeader.biBitCount = 32;
        infoHeader.biCompression = BI_RGB;
        infoHeader.biSizeImage = static_cast<DWORD>(pixels.size());

        fileHeader.bfType = 0x4D42;
        fileHeader.bfOffBits = sizeof(fileHeader) + sizeof(infoHeader);
        fileHeader.bfSize = fileHeader.bfOffBits + infoHeader.biSizeImage;

        output.write(reinterpret_cast<const char*>(&fileHeader), sizeof(fileHeader));
        output.write(reinterpret_cast<const char*>(&infoHeader), sizeof(infoHeader));
        output.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
        return output.good();
    }

    void ThrowIfFailed(HRESULT hr, const char* label)
    {
        if (FAILED(hr))
        {
            std::ostringstream out;
            out << label << " failed with HRESULT " << HrToHex(hr);
            throw std::runtime_error(out.str());
        }
    }

    // Set a debug name on a D3D11 object (works in _DEBUG builds).
    template <typename T>
    void SetD3DName(T* object, const char* name)
    {
#ifdef _DEBUG
        if (!object || !name)
        {
            return;
        }
        object->SetPrivateData(
            WKPDID_D3DDebugObjectName,
            static_cast<UINT>(std::strlen(name)),
            name);
#endif
    }

    RECT MonitorRectFromWindow(HWND hwnd)
    {
        HMONITOR monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        MONITORINFO info{};
        info.cbSize = sizeof(info);
        if (!GetMonitorInfoW(monitor, &info))
        {
            return RECT{ 0, 0, 1280, 720 };
        }
        return info.rcMonitor;
    }

    LONG RectWidth(const RECT& rect)
    {
        return std::max<LONG>(0, rect.right - rect.left);
    }

    LONG RectHeight(const RECT& rect)
    {
        return std::max<LONG>(0, rect.bottom - rect.top);
    }

    std::wstring WindowTitle(HWND hwnd)
    {
        int length = GetWindowTextLengthW(hwnd);
        std::wstring title(static_cast<size_t>(length) + 1, L'\0');
        if (length > 0)
        {
            int copied = GetWindowTextW(hwnd, title.data(), length + 1);
            title.resize(static_cast<size_t>(std::max(copied, 0)));
        }
        else
        {
            title.clear();
        }
        return title;
    }

    BOOL CALLBACK FindWindowByTitleProc(HWND hwnd, LPARAM lParam)
    {
        auto* data = reinterpret_cast<std::pair<std::wstring, HWND>*>(lParam);
        if (!IsWindowVisible(hwnd) || GetWindow(hwnd, GW_OWNER) != nullptr)
        {
            return TRUE;
        }

        std::wstring title = WindowTitle(hwnd);
        std::wstring haystack = title;
        std::wstring needle = data->first;
        std::transform(haystack.begin(), haystack.end(), haystack.begin(), ::towlower);
        std::transform(needle.begin(), needle.end(), needle.begin(), ::towlower);

        if (!needle.empty() && haystack.find(needle) != std::wstring::npos)
        {
            data->second = hwnd;
            return FALSE;
        }
        return TRUE;
    }

    HWND FindWindowByTitle(const std::wstring& title)
    {
        std::pair<std::wstring, HWND> data{ title, nullptr };
        EnumWindows(FindWindowByTitleProc, reinterpret_cast<LPARAM>(&data));
        return data.second;
    }

    uint64_t ParseInteger(const std::wstring& text)
    {
        int base = 10;
        const wchar_t* start = text.c_str();
        if (text.rfind(L"0x", 0) == 0 || text.rfind(L"0X", 0) == 0)
        {
            base = 16;
            start += 2;
        }
        return _wcstoui64(start, nullptr, base);
    }

