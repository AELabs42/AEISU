    std::filesystem::path FindNeuralModelPath(
        const std::filesystem::path& relative,
        const char* modelName)
    {
        std::vector<std::filesystem::path> candidates;
        try
        {
            const std::filesystem::path current = std::filesystem::current_path();
            candidates.push_back(current / L"AESS" / relative);
            candidates.push_back(current / relative);
        }
        catch (...)
        {
        }

        const std::filesystem::path executable = ExecutableDirectory();
        if (!executable.empty())
        {
            candidates.push_back(executable / relative);
            candidates.push_back(executable.parent_path() / relative);
            candidates.push_back(executable.parent_path().parent_path() / relative);
        }
        for (const auto& candidate : candidates)
        {
            std::error_code error;
            if (std::filesystem::is_regular_file(candidate, error))
            {
                return std::filesystem::weakly_canonical(candidate, error);
            }
        }
        throw std::runtime_error(
            std::string(modelName) + " model artifact was not found (expected " +
            relative.generic_string() + ")");
    }

    std::filesystem::path NeuralModelPath(aess::neural::ModelKind)
    {
        return FindNeuralModelPath(L"models/aeisu_mixed/adaptive_fp8.bin", "AEISU mixed");
    }

    void GetClientSize(HWND hwnd, UINT& width, UINT& height)
    {
        RECT rc{};
        GetClientRect(hwnd, &rc);
        width = std::max<LONG>(1, rc.right - rc.left);
        height = std::max<LONG>(1, rc.bottom - rc.top);
    }

    struct ToneConstants
    {
        float sdrWhiteScale = 1.0f;
        float gamma = 2.2f;
        float saturation = 1.0f;
        float contrast = 1.0f;
        float blackPoint = 0.0f;
        float highlightCompression = 0.0f;
        float sdrWhiteNits = 80.0f;
        float outputWidth = 1.0f;
        float outputHeight = 1.0f;
        float reservedPadding = 0.0f;
        uint32_t sourceTransfer = kModelTransferSdrGamma;
        uint32_t sourcePrimaries = kModelPrimariesRec709;
        float sourceCropLeft = 0.0f;
        float sourceCropTop = 0.0f;
        float sourceCropWidth = 1.0f;
        float sourceCropHeight = 1.0f;
        float sourceWidth = 1.0f;
        float sourceHeight = 1.0f;
        float reservedPadding2 = 0.0f;
        float reservedPadding3 = 0.0f;
    };

    struct D3D11RenderResult
    {
        uint32_t workingWidth = 0;
        uint32_t workingHeight = 0;
        uint32_t videoProcessorOutputWidth = 0;
        uint32_t videoProcessorOutputHeight = 0;
        uint32_t internalOutputScale = 2;
        double preprocessMs = 0.0;
        double videoProcessorMs = 0.0;
        double presentMs = 0.0;
        double presentCallMs = 0.0;
        double gpuFrameMs = 0.0;
        uint64_t skippedFrames = 0;
        uint64_t presentQueueBusyCount = 0;
        bool shaderUpscaler = false;
        UpscalerMode upscalerMode = UpscalerMode::AdaptiveFilter32Mixed;
        UpscalerMode selectedUpscalerMode = UpscalerMode::NoUpscaler;
        bool upscalerBypass = false;
        OutputMode outputMode = OutputMode::SdrShaderBridge;
        DXGI_FORMAT inputFormat = DXGI_FORMAT_UNKNOWN;
        DXGI_COLOR_SPACE_TYPE inputColorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        CaptureColorProvenance inputColorProvenance =
            CaptureColorProvenance::Unknown;
        DXGI_FORMAT videoProcessorOutputFormat = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT outputFormat = DXGI_FORMAT_UNKNOWN;
        ToneConstants tone{};
    };

    struct D3D11PerfFrameSummary
    {
        uint64_t frameNumber = 0;
        uint64_t sourceVersion = 0;
        uint64_t skippedFrames = 0;
        double cpuAcquireMs = 0.0;
        double cpuRenderMs = 0.0;
        double cpuPreprocessMs = 0.0;
        double cpuUpscaleMs = 0.0;
        double cpuPresentSetupMs = 0.0;
        double cpuPresentCallMs = 0.0;
        bool dumpFrame = false;
        std::array<char, 32> upscaler{};
        std::array<char, 32> selectedUpscaler{};
        std::array<char, 24> inputSize{};
        std::array<char, 24> internalOutputSize{};
        std::array<char, 32> outputFormat{};
        std::array<char, 48> totalStageName{
            'g', 'p', 'u', '-', 't', 'o', 't', 'a', 'l', '\0' };
    };

    template <size_t Count>
    std::string_view PerfTextView(const std::array<char, Count>& text) noexcept
    {
        return std::string_view(text.data(), strnlen_s(text.data(), text.size()));
    }

    template <size_t Count>
    void FormatPerfSize(
        std::array<char, Count>& destination,
        UINT width,
        UINT height) noexcept
    {
        static_assert(Count > 0);
        _snprintf_s(
            destination.data(),
            destination.size(),
            _TRUNCATE,
            "%ux%u",
            width,
            height);
    }

    template <size_t Count>
    void CopyPerfWideText(
        std::array<char, Count>& destination,
        const wchar_t* source) noexcept
    {
        static_assert(Count > 0);
        destination.fill('\0');
        if (!source || WideCharToMultiByte(
                CP_UTF8,
                WC_ERR_INVALID_CHARS,
                source,
                -1,
                destination.data(),
                static_cast<int>(destination.size()),
                nullptr,
                nullptr) == 0)
        {
            aess::presentation::CopyTelemetryText(destination, "unknown");
        }
    }

    void SetPerfSummarySignature(
        D3D11PerfFrameSummary& summary,
        UpscalerMode upscaler,
        UpscalerMode selectedUpscaler,
        UINT inputWidth,
        UINT inputHeight,
        UINT outputWidth,
        UINT outputHeight,
        DXGI_FORMAT outputFormat) noexcept
    {
        CopyPerfWideText(
            summary.upscaler,
            InfoForUpscaler(upscaler).cliName);
        CopyPerfWideText(
            summary.selectedUpscaler,
            InfoForUpscaler(selectedUpscaler).cliName);
        FormatPerfSize(summary.inputSize, inputWidth, inputHeight);
        FormatPerfSize(summary.internalOutputSize, outputWidth, outputHeight);
        aess::presentation::CopyTelemetryText(
            summary.outputFormat,
            FormatName(outputFormat));
    }

    class D3D11GpuProfiler
    {
    public:
        class Scope
        {
        public:
            Scope() = default;
            Scope(D3D11GpuProfiler* owner, size_t stageIndex)
                : m_owner(owner),
                  m_stageIndex(stageIndex)
            {
            }

            Scope(const Scope&) = delete;
            Scope& operator=(const Scope&) = delete;

            Scope(Scope&& other) noexcept
                : m_owner(other.m_owner),
                  m_stageIndex(other.m_stageIndex)
            {
                other.m_owner = nullptr;
                other.m_stageIndex = npos;
            }

            Scope& operator=(Scope&& other) noexcept
            {
                if (this != &other)
                {
                    End();
                    m_owner = other.m_owner;
                    m_stageIndex = other.m_stageIndex;
                    other.m_owner = nullptr;
                    other.m_stageIndex = npos;
                }
                return *this;
            }

            ~Scope()
            {
                End();
            }

        private:
            static constexpr size_t npos = static_cast<size_t>(-1);

            void End()
            {
                if (m_owner && m_stageIndex != npos)
                {
                    m_owner->EndStage(m_stageIndex);
                    m_owner = nullptr;
                    m_stageIndex = npos;
                }
            }

            D3D11GpuProfiler* m_owner = nullptr;
            size_t m_stageIndex = npos;
        };

        void Configure(
            bool enabled,
            const std::filesystem::path& logPath,
            aess::presentation::FrameTelemetryWriter* telemetry)
        {
            ReleaseQueryPool();
            m_enabled = enabled;
            m_logPath = enabled
                ? (logPath.empty() ? DefaultPerfLogPath() : logPath)
                : std::filesystem::path{};
            m_telemetry = enabled ? telemetry : nullptr;
            m_lastFrameGpuMs = 0.0;
            InitializeQueryPool();
        }

        bool Enabled() const
        {
            return m_enabled;
        }

        const std::filesystem::path& LogPath() const
        {
            return m_logPath;
        }

        double LastFrameGpuMs() const
        {
            return m_lastFrameGpuMs;
        }

        void SetDevice(ID3D11Device* device, ID3D11DeviceContext* context)
        {
            if (m_device.Get() == device && m_context.Get() == context)
            {
                return;
            }

            ReleaseQueryPool();
            m_device = device;
            m_context = context;
            InitializeQueryPool();
        }

        void BeginFrame(uint64_t frameNumber)
        {
            if (!m_enabled || !m_queryPoolReady || !m_context)
            {
                return;
            }

            CollectReady();
            if (m_currentIndex != npos)
            {
                CancelFrame();
            }

            for (size_t offset = 0; offset < kQueryFrameCapacity; ++offset)
            {
                const size_t index =
                    (m_nextFrameSearch + offset) % kQueryFrameCapacity;
                QueryFrame& frame = m_queryFrames[index];
                if (frame.state != QueryFrameState::Free)
                {
                    continue;
                }
                frame.stageCount = 0;
                frame.summary = {};
                frame.summary.frameNumber = frameNumber;
                frame.summaryReady = true;
                frame.discard = false;
                frame.state = QueryFrameState::Recording;
                m_currentIndex = index;
                m_nextFrameSearch = (index + 1) % kQueryFrameCapacity;
                m_context->Begin(frame.disjoint.Get());
                return;
            }
        }

        Scope Stage(const char* name)
        {
            if (!m_enabled || m_currentIndex == npos || !m_context)
            {
                return {};
            }

            QueryFrame& frame = m_queryFrames[m_currentIndex];
            if (frame.state != QueryFrameState::Recording ||
                frame.stageCount >= kStageCapacity)
            {
                return {};
            }

            const size_t stageIndex = frame.stageCount++;
            StageRecord& record = frame.stages[stageIndex];
            aess::presentation::CopyTelemetryText(
                record.name,
                name ? std::string_view(name) : std::string_view("unnamed"));
            m_context->End(record.start.Get());
            return Scope(this, stageIndex);
        }

        void EndStage(size_t stageIndex)
        {
            if (!m_enabled || m_currentIndex == npos || !m_context)
            {
                return;
            }
            QueryFrame& frame = m_queryFrames[m_currentIndex];
            if (frame.state != QueryFrameState::Recording ||
                stageIndex >= frame.stageCount)
            {
                return;
            }
            m_context->End(frame.stages[stageIndex].end.Get());
        }

        void EndFrame(const D3D11PerfFrameSummary& summary)
        {
            EndFrameInternal(summary, true);
        }

        void EndFrameDeferred(const D3D11PerfFrameSummary& summary)
        {
            EndFrameInternal(summary, false);
        }

        void FinalizeDeferredFrame(
            uint64_t frameNumber,
            const D3D11PerfFrameSummary& summary)
        {
            if (!m_enabled) return;
            for (size_t offset = 0; offset < m_pendingCount; ++offset)
            {
                QueryFrame& frame = m_queryFrames[
                    m_pendingIndices[
                        (m_pendingHead + offset) % kQueryFrameCapacity]];
                if (frame.summary.frameNumber == frameNumber &&
                    !frame.summaryReady)
                {
                    frame.summary = summary;
                    frame.summaryReady = true;
                    CollectReady();
                    return;
                }
            }
        }

        void CancelDeferredFrame(uint64_t frameNumber) noexcept
        {
            if (!m_enabled) return;
            for (size_t offset = 0; offset < m_pendingCount; ++offset)
            {
                QueryFrame& frame = m_queryFrames[
                    m_pendingIndices[
                        (m_pendingHead + offset) % kQueryFrameCapacity]];
                if (frame.summary.frameNumber == frameNumber &&
                    !frame.summaryReady)
                {
                    frame.summaryReady = true;
                    frame.discard = true;
                    CollectReady();
                    return;
                }
            }
        }

    private:
        void EndFrameInternal(
            const D3D11PerfFrameSummary& summary,
            bool summaryReady)
        {
            if (!m_enabled || m_currentIndex == npos || !m_context)
            {
                return;
            }

            QueryFrame& frame = m_queryFrames[m_currentIndex];
            frame.summary = summary;
            frame.summaryReady = summaryReady;
            m_context->End(frame.disjoint.Get());
            frame.state = QueryFrameState::Pending;
            EnqueuePending(m_currentIndex);
            m_currentIndex = npos;
            CollectReady();
        }

    public:

        struct ExternalStageTiming
        {
            std::string_view name;
            double elapsedMs = 0.0;
        };

        void RecordExternalFrame(
            const D3D11PerfFrameSummary& summary,
            std::span<const ExternalStageTiming> stages,
            double totalGpuMs)
        {
            if (!m_enabled)
            {
                return;
            }
            for (const ExternalStageTiming& stage : stages)
            {
                WriteRow(summary, stage.name, stage.elapsedMs);
            }
            m_lastFrameGpuMs = totalGpuMs;
            WriteRow(summary, "gpu-total", totalGpuMs);
        }

        void CancelFrame()
        {
            if (m_currentIndex == npos)
            {
                return;
            }
            QueryFrame& frame = m_queryFrames[m_currentIndex];
            if (frame.state == QueryFrameState::Recording && m_context)
            {
                // Persistent query objects cannot be reused until their prior
                // Begin/End pair retires.  Queue a discard record instead of
                // destroying and recreating the query as the old profiler did.
                m_context->End(frame.disjoint.Get());
                frame.summaryReady = true;
                frame.discard = true;
                frame.state = QueryFrameState::Pending;
                EnqueuePending(m_currentIndex);
            }
            else
            {
                frame.state = QueryFrameState::Free;
            }
            m_currentIndex = npos;
        }

        void CollectReady()
        {
            if (!m_enabled || !m_context)
            {
                return;
            }

            while (m_pendingCount != 0)
            {
                QueryFrame& frame = m_queryFrames[
                    m_pendingIndices[m_pendingHead]];
                if (!frame.summaryReady)
                {
                    break;
                }
                D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjoint{};
                HRESULT hr = m_context->GetData(
                    frame.disjoint.Get(),
                    &disjoint,
                    sizeof(disjoint),
                    D3D11_ASYNC_GETDATA_DONOTFLUSH);
                if (hr != S_OK)
                {
                    break;
                }

                std::array<double, kStageCapacity> gpuMs{};
                bool ready = true;
                for (size_t stageIndex = 0;
                     stageIndex < frame.stageCount;
                     ++stageIndex)
                {
                    const StageRecord& stage = frame.stages[stageIndex];
                    UINT64 start = 0;
                    UINT64 end = 0;
                    if (m_context->GetData(stage.start.Get(), &start, sizeof(start), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK ||
                        m_context->GetData(stage.end.Get(), &end, sizeof(end), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK)
                    {
                        ready = false;
                        break;
                    }

                    double elapsed = 0.0;
                    if (!disjoint.Disjoint && disjoint.Frequency > 0 && end >= start)
                    {
                        elapsed = static_cast<double>(end - start) * 1000.0 / static_cast<double>(disjoint.Frequency);
                    }
                    gpuMs[stageIndex] = elapsed;
                }

                if (!ready)
                {
                    break;
                }

                if (!frame.discard)
                {
                    WriteFrame(frame, gpuMs);
                }
                frame.stageCount = 0;
                frame.summaryReady = true;
                frame.discard = false;
                frame.state = QueryFrameState::Free;
                m_pendingHead =
                    (m_pendingHead + 1) % kQueryFrameCapacity;
                --m_pendingCount;
            }
        }

        void Flush()
        {
            if (!m_enabled)
            {
                return;
            }

            for (int i = 0; i < 200 && m_pendingCount != 0; ++i)
            {
                CollectReady();
                if (m_pendingCount != 0)
                {
                    Sleep(1);
                }
            }
        }

    private:
        static constexpr size_t kQueryFrameCapacity = 32;
        static constexpr size_t kStageCapacity = 32;
        static constexpr size_t npos = static_cast<size_t>(-1);

        enum class QueryFrameState : uint8_t
        {
            Free,
            Recording,
            Pending,
        };

        struct StageRecord
        {
            std::array<char, 48> name{};
            ComPtr<ID3D11Query> start;
            ComPtr<ID3D11Query> end;
        };

        struct QueryFrame
        {
            ComPtr<ID3D11Query> disjoint;
            std::array<StageRecord, kStageCapacity> stages{};
            D3D11PerfFrameSummary summary;
            size_t stageCount = 0;
            bool summaryReady = true;
            bool discard = false;
            QueryFrameState state = QueryFrameState::Free;
        };

        void InitializeQueryPool()
        {
            if (!m_enabled || !m_device || !m_context || m_queryPoolReady)
            {
                return;
            }

            D3D11_QUERY_DESC disjointDescription{};
            disjointDescription.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
            D3D11_QUERY_DESC timestampDescription{};
            timestampDescription.Query = D3D11_QUERY_TIMESTAMP;

            for (QueryFrame& frame : m_queryFrames)
            {
                if (FAILED(m_device->CreateQuery(
                        &disjointDescription,
                        &frame.disjoint)))
                {
                    ReleaseQueryPool();
                    return;
                }
                for (StageRecord& stage : frame.stages)
                {
                    if (FAILED(m_device->CreateQuery(
                            &timestampDescription,
                            &stage.start)) ||
                        FAILED(m_device->CreateQuery(
                            &timestampDescription,
                            &stage.end)))
                    {
                        ReleaseQueryPool();
                        return;
                    }
                }
            }
            m_queryPoolReady = true;
        }

        void ReleaseQueryPool() noexcept
        {
            m_currentIndex = npos;
            m_nextFrameSearch = 0;
            m_pendingHead = 0;
            m_pendingCount = 0;
            m_queryPoolReady = false;
            for (QueryFrame& frame : m_queryFrames)
            {
                frame.disjoint.Reset();
                for (StageRecord& stage : frame.stages)
                {
                    stage.start.Reset();
                    stage.end.Reset();
                }
                frame.stageCount = 0;
                frame.summaryReady = true;
                frame.discard = false;
                frame.state = QueryFrameState::Free;
            }
        }

        void EnqueuePending(size_t frameIndex) noexcept
        {
            // A recording frame occupies one pool entry, so a full pending
            // ring cannot coexist with a valid current frame.
            if (m_pendingCount >= kQueryFrameCapacity)
            {
                return;
            }
            m_pendingIndices[
                (m_pendingHead + m_pendingCount) % kQueryFrameCapacity] =
                frameIndex;
            ++m_pendingCount;
        }

        void WriteRow(
            const D3D11PerfFrameSummary& summary,
            std::string_view stage,
            double gpuMs)
        {
            if (!m_telemetry) return;
            aess::presentation::GpuStageTimingRecord record;
            record.frameNumber = summary.frameNumber;
            record.sourceVersion = summary.sourceVersion;
            record.skippedFrames = summary.skippedFrames;
            record.gpuMs = gpuMs;
            record.cpuRenderMs = summary.cpuRenderMs;
            record.cpuAcquireMs = summary.cpuAcquireMs;
            record.cpuPreprocessMs = summary.cpuPreprocessMs;
            record.cpuUpscaleMs = summary.cpuUpscaleMs;
            record.cpuPresentSetupMs = summary.cpuPresentSetupMs;
            record.cpuPresentCallMs = summary.cpuPresentCallMs;
            aess::presentation::CopyTelemetryText(record.stage, stage);
            record.upscaler = summary.upscaler;
            record.selectedUpscaler = summary.selectedUpscaler;
            record.inputSize = summary.inputSize;
            record.internalOutputSize = summary.internalOutputSize;
            record.outputFormat = summary.outputFormat;
            record.dumpFrame = summary.dumpFrame ? 1u : 0u;
            m_telemetry->TryEnqueueGpuStage(record);
        }

        void WriteFrame(
            const QueryFrame& frame,
            const std::array<double, kStageCapacity>& gpuMs)
        {
            double total = 0.0;
            for (size_t index = 0; index < frame.stageCount; ++index)
            {
                total += gpuMs[index];
                WriteRow(
                    frame.summary,
                    PerfTextView(frame.stages[index].name),
                    gpuMs[index]);
            }

            m_lastFrameGpuMs = total;
            const std::string_view totalStageName =
                PerfTextView(frame.summary.totalStageName);
            WriteRow(
                frame.summary,
                totalStageName.empty()
                    ? std::string_view("gpu-total")
                    : totalStageName,
                total);
        }

        bool m_enabled = false;
        double m_lastFrameGpuMs = 0.0;
        std::filesystem::path m_logPath;
        aess::presentation::FrameTelemetryWriter* m_telemetry = nullptr;
        ComPtr<ID3D11Device> m_device;
        ComPtr<ID3D11DeviceContext> m_context;
        std::array<QueryFrame, kQueryFrameCapacity> m_queryFrames{};
        std::array<size_t, kQueryFrameCapacity> m_pendingIndices{};
        size_t m_currentIndex = npos;
        size_t m_nextFrameSearch = 0;
        size_t m_pendingHead = 0;
        size_t m_pendingCount = 0;
        bool m_queryPoolReady = false;
    };

    void AppendRuntimeLog(uint64_t rendered, const FrameInfo& frame, const D3D11RenderResult& result)
    {
        try
        {
            std::filesystem::path path = DefaultDebugDumpRoot() / L"aess-d3d11-runtime.log";
            std::filesystem::create_directories(path.parent_path());

            SYSTEMTIME time{};
            GetLocalTime(&time);

            std::ofstream out(path, std::ios::app);
            if (!out)
            {
                return;
            }

            out << time.wYear << "-"
                << (time.wMonth < 10 ? "0" : "") << time.wMonth << "-"
                << (time.wDay < 10 ? "0" : "") << time.wDay << " "
                << (time.wHour < 10 ? "0" : "") << time.wHour << ":"
                << (time.wMinute < 10 ? "0" : "") << time.wMinute << ":"
                << (time.wSecond < 10 ? "0" : "") << time.wSecond
                << " rendered=" << rendered
                << " source=" << frame.width << "x" << frame.height
                << " input=" << result.workingWidth << "x" << result.workingHeight
                << " internal_output=" << result.videoProcessorOutputWidth << "x" << result.videoProcessorOutputHeight
                << " internal_output_scale=" << result.internalOutputScale
                << " upscaler=" << Narrow(std::wstring(InfoForUpscaler(result.upscalerMode).cliName))
                << " selected_upscaler=" << Narrow(std::wstring(InfoForUpscaler(result.selectedUpscalerMode).cliName))
                << " upscaler_bypass=" << (result.upscalerBypass ? "yes" : "no")
                << " mode=" << InfoForOutputMode(result.outputMode).logName
                << " input_format=" << FormatName(result.inputFormat)
                << " input_color_space=" << static_cast<uint32_t>(result.inputColorSpace)
                << " input_color_provenance="
                << CaptureColorProvenanceName(result.inputColorProvenance)
                << " model_transfer=" << result.tone.sourceTransfer
                << " model_primaries=" << result.tone.sourcePrimaries
                << " working_encoding="
                << WorkingColorEncodingName(result.tone.sourceTransfer)
                << " vp_output=" << FormatName(result.videoProcessorOutputFormat)
                << " sdr_white_nits=" << result.tone.sdrWhiteNits
                << " sdr_white_scale=" << result.tone.sdrWhiteScale
                << " gamma=" << result.tone.gamma
                << " tone_saturation=" << result.tone.saturation
                << " tone_contrast=" << result.tone.contrast
                << " tone_black=" << result.tone.blackPoint
                << " pre_ms=" << result.preprocessMs
                << " vp_ms=" << result.videoProcessorMs
                << " present_ms=" << result.presentMs
                << " present_call_ms=" << result.presentCallMs
                << " gpu_ms=" << result.gpuFrameMs
                << " skipped_frames=" << result.skippedFrames
                << " present_queue_busy=" << result.presentQueueBusyCount
                << " shader_upscaler=" << (result.shaderUpscaler ? "yes" : "no")
                << " output=" << FormatName(result.outputFormat)
                << "\n";
        }
        catch (...)
        {
        }
    }

    const std::string& D3D11PresenterShaderSource()
    {
        static const std::string source = ReadTextFile(
            FindProductFile(L"src/d3d11_presenter.hlsl"));
        return source;
    }

    constexpr UINT kSwapChainLifecycleMessage = WM_APP + 0x41;

