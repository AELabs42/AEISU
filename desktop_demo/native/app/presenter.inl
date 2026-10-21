    class D3D11VideoPresenter
    {
    private:
        enum class PipelineSlotState : uint8_t
        {
            Free,
            Submitted,
            Ready,
        };

        enum class SwapChainLifecycleOperation : uint32_t
        {
            None,
            Ensure,
            Release,
        };

        struct FrameGeometry
        {
            D3D11_TEXTURE2D_DESC sourceDesc{};
            UINT cropLeft = 0;
            UINT cropTop = 0;
            UINT cropWidth = 0;
            UINT cropHeight = 0;
            UINT presentationWidth = 1;
            UINT presentationHeight = 1;
        };

        FrameGeometry ResolveFrameGeometry(const FrameInfo& frame) const
        {
            FrameGeometry geometry;
            frame.d3d11Texture->GetDesc(&geometry.sourceDesc);
            if (geometry.sourceDesc.Width == 0 || geometry.sourceDesc.Height == 0)
            {
                throw std::runtime_error(
                    "Unified D3D11 mode received an invalid source texture.");
            }

            geometry.cropLeft =
                std::min<UINT>(frame.cropLeft, geometry.sourceDesc.Width - 1);
            geometry.cropTop =
                std::min<UINT>(frame.cropTop, geometry.sourceDesc.Height - 1);
            geometry.cropWidth = std::max<UINT>(
                1,
                std::min<UINT>(
                    frame.cropWidth != 0 ? frame.cropWidth : geometry.sourceDesc.Width,
                    geometry.sourceDesc.Width - geometry.cropLeft));
            geometry.cropHeight = std::max<UINT>(
                1,
                std::min<UINT>(
                    frame.cropHeight != 0 ? frame.cropHeight : geometry.sourceDesc.Height,
                    geometry.sourceDesc.Height - geometry.cropTop));
            GetClientSize(
                m_hwnd,
                geometry.presentationWidth,
                geometry.presentationHeight);
            return geometry;
        }

        double QpcElapsedMilliseconds(int64_t beginQpc, int64_t endQpc) const noexcept
        {
            if (beginQpc <= 0 || endQpc <= beginQpc)
            {
                return 0.0;
            }
            return static_cast<double>(endQpc - beginQpc) * 1000.0 /
                static_cast<double>(m_framePacer.QpcFrequency());
        }

        void StampPacingTelemetry(FrameTimingRecord& timing) noexcept
        {
            (void)timing;
        }

        void StampCompletionPacingTelemetry(FrameTimingRecord& timing) noexcept
        {
            (void)timing;
        }

        FrameTimingRecord BeginFrameTiming(
            const FrameInfo& frame,
            uint64_t frameNumber,
            uint64_t submitOrdinal,
            int64_t submitQpc,
            uint32_t workingWidth,
            uint32_t workingHeight) noexcept
        {
            FrameTimingRecord timing;
            timing.recordId = ++m_timingRecordId;
            timing.frameNumber = frameNumber;
            timing.sourceVersion = frame.version;
            timing.sourcePresentOrdinal = frame.sourcePresentOrdinal;
            timing.captureArrivalCount = frame.captureArrivalCount;
            timing.effectSubmitOrdinal = submitOrdinal;
            timing.sourcePresentQpc = frame.sourcePresentQpc;
            timing.captureReadyQpc = frame.captureReadyQpc;
            timing.captureAcquireBeginQpc = frame.captureAcquireBeginQpc;
            timing.captureAcquireEndQpc = frame.captureAcquireEndQpc;
            timing.effectSubmitQpc = submitQpc;
            // Gate signatures describe the work entering the selected effect,
            // not the dimensions of the captured window before preprocessing.
            timing.inputWidth = workingWidth;
            timing.inputHeight = workingHeight;
            timing.upscalerMode = static_cast<uint32_t>(m_activeUpscalerMode);
            timing.pacingMode = static_cast<uint32_t>(m_framePacingConfig.pacingMode);
            timing.writerOverflowBefore = m_telemetry.DroppedFrameRecords();
            timing.flags = aess::presentation::FrameTimingEffectSubmitted;
            if (frame.sourcePresentQpc > 0)
            {
                timing.flags |= aess::presentation::FrameTimingValidSourcePresentQpc;
            }
            if (frame.captureReadyQpc > 0)
            {
                timing.flags |= aess::presentation::FrameTimingValidCaptureReadyQpc;
            }
            if (frame.sourcePresentOrdinal > 0)
            {
                if (m_lastTimingSourceOrdinal > 0 &&
                    frame.sourcePresentOrdinal > m_lastTimingSourceOrdinal + 1)
                {
                    const uint64_t missingSourcePresents =
                        frame.sourcePresentOrdinal - m_lastTimingSourceOrdinal - 1;
                    FrameTimingRecord gap = timing;
                    gap.frameNumber = 0;
                    gap.captureArrivalCount = 0;
                    gap.effectSubmitOrdinal = 0;
                    gap.effectSubmitQpc = 0;
                    gap.effectCompleteQpc = 0;
                    gap.readyQueueEntryQpc = 0;
                    gap.droppedSourcePresentsBefore = missingSourcePresents;
                    gap.flags = aess::presentation::FrameTimingSyntheticGapRecord;
                    if (gap.sourcePresentQpc > 0)
                    {
                        gap.flags |=
                            aess::presentation::FrameTimingValidSourcePresentQpc;
                    }
                    EnqueueDroppedTiming(
                        gap,
                        m_pipelineAdmissionBlockedSinceLastFrame
                            ? FrameDropReason::PipelineSaturated
                            : FrameDropReason::CaptureSuperseded,
                        missingSourcePresents);
                    // Keep record IDs in publication order: the synthetic gap
                    // owns the provisional ID and this admitted frame follows.
                    timing.recordId = ++m_timingRecordId;
                    timing.writerOverflowBefore = m_telemetry.DroppedFrameRecords();
                }
                m_lastTimingSourceOrdinal = frame.sourcePresentOrdinal;
            }
            if (frame.sourcePresentQpc > 0)
            {
                m_lastTimingSourceQpc = frame.sourcePresentQpc;
            }
            m_pipelineAdmissionBlockedSinceLastFrame = false;
            return timing;
        }

        void EnqueueDroppedTiming(
            FrameTimingRecord& timing,
            FrameDropReason reason,
            uint64_t occurrenceCount = 1) noexcept
        {
            timing.dropReason = reason;
            timing.queueTarget = m_framePacer.TargetQueueDepth();
            StampPacingTelemetry(timing);
            const size_t reasonIndex = static_cast<size_t>(reason);
            if (reasonIndex < m_frameDropCounts.size())
            {
                m_frameDropCounts[reasonIndex] += std::max<uint64_t>(1, occurrenceCount);
            }
            m_telemetry.TryEnqueueFrame(timing);
        }

        bool TimingOlderThanTwoPeriods(
            const FrameTimingRecord& timing,
            int64_t nowQpc) const noexcept
        {
            // Queue staleness is time spent waiting after GPU completion.
            // Source-to-ready age includes capture and inference latency and
            // is telemetry, not a reason to continuously reset cadence under
            // a high-rate source or a saturated effect.
            const int64_t ageAnchor = timing.readyQueueEntryQpc;
            const int64_t period = m_framePacer.TargetPeriodQpc(nowQpc);
            return ageAnchor > 0 && period > 0 && nowQpc > ageAnchor &&
                nowQpc - ageAnchor > period * 2;
        }

    public:
        void MarkContextDirty() noexcept
        {
            m_contextDirty = true;
        }

        void FlushContext()
        {
            if (m_context && m_contextDirty)
            {
                m_context->Flush();
                m_contextDirty = false;
            }
        }

        explicit D3D11VideoPresenter(
            HWND hwnd,
            UpscalerMode upscalerMode,


            SourceColorMode sourceColorMode,
            OutputMode outputMode,
            float sdrWhiteNits,
            bool fpsCounterEnabled,
            bool gsyncCompatibilityEligible,
            const FramePacingConfig& framePacingConfig,
            DebugViewMode debugViewMode,
            bool perfLogEnabled,
            const std::filesystem::path& perfLogPath,
            DWORD windowThreadId,
            PresentDispatchMode presentDispatchMode = PresentDispatchMode::Auto,
            ComparisonFilter comparisonFilter = ComparisonFilter::Nearest)
            : m_hwnd(hwnd),
              m_upscalerMode(upscalerMode),


              m_sourceColorMode(sourceColorMode),
              m_outputMode(outputMode),
              m_requestedSdrWhiteNits(std::clamp(sdrWhiteNits, 80.0f, 500.0f)),
              m_fpsCounterEnabled(fpsCounterEnabled),
              m_gsyncCompatibilityEligible(gsyncCompatibilityEligible),
              m_framePacingConfig(framePacingConfig),
              m_debugViewMode(debugViewMode),
              m_presentDispatchMode(presentDispatchMode),
              m_framePacer(framePacingConfig),

              m_windowThreadId(windowThreadId),
              m_comparisonFilter(comparisonFilter)
        {
            
            if (perfLogEnabled)
            {
                LARGE_INTEGER frequency{};
                QueryPerformanceFrequency(&frequency);
                std::string diagnostic;
                const std::filesystem::path path = perfLogPath.empty()
                    ? DefaultPerfLogPath()
                    : perfLogPath;
                if (!m_telemetry.Start(path, frequency.QuadPart, diagnostic))
                {
                    throw std::runtime_error("Start background frame telemetry failed: " + diagnostic);
                }
            }
            m_profiler.Configure(perfLogEnabled, perfLogPath, &m_telemetry);
        }

        D3D11VideoPresenter(const D3D11VideoPresenter&) = delete;
        D3D11VideoPresenter& operator=(const D3D11VideoPresenter&) = delete;

        ~D3D11VideoPresenter()
        {
            
            ResetSynchronousReadyFrame(
                FrameDropReason::ShutdownDiscard,
                true);
            ResetS2D32PipelineSlots(FrameDropReason::ShutdownDiscard);
            DrainCaptureReads();
            ShutdownFpsCounterOverlay();
            m_profiler.Flush();
            ResolveLatestPresentationStatisticsBeforeReset();
            FlushPendingPresentationTimings(FrameDropReason::ShutdownDiscard);
            m_telemetry.Stop();
            ResetSwapChainDependentResources();
            const HRESULT releaseHr = ReleaseSwapChainOnOwningThread();
            if (FAILED(releaseHr))
            {
                std::cerr << "Release swapchain on HWND owner during shutdown failed, HRESULT="
                    << HrToHex(releaseHr) << "\n";
            }
        }

        // Keep every DXGI swapchain creation on the normal-priority
        // HWND owner. The render worker takes exclusive immediate-context
        // ownership after this startup step; rare lifecycle changes are sent
        // synchronously back to this thread without touching the Present path.
        void PrimePresentation(ID3D11Device* sourceDevice)
        {
            if (GetCurrentThreadId() != m_windowThreadId)
            {
                throw std::runtime_error(
                    "Presentation startup must execute on the output HWND owner thread.");
            }
            if (!sourceDevice)
            {
                throw std::runtime_error(
                    "Capture source did not expose its D3D11 device for presentation startup.");
            }
            EnsureDevice(sourceDevice);
            RECT client{};
            if (!GetClientRect(m_hwnd, &client))
            {
                ThrowIfFailed(
                    HRESULT_FROM_WIN32(GetLastError()),
                    "Get AESS output client rectangle for presentation startup");
            }
            EnsureSwapChain(
                static_cast<UINT>(std::max<LONG>(1, client.right - client.left)),
                static_cast<UINT>(std::max<LONG>(1, client.bottom - client.top)));
        }

        HRESULT ExecuteWindowThreadSwapChainLifecycle() noexcept
        {
            if (GetCurrentThreadId() != m_windowThreadId)
            {
                return HRESULT_FROM_WIN32(ERROR_INVALID_THREAD_ID);
            }

            const SwapChainLifecycleOperation operation =
                m_swapChainLifecycleOperation.exchange(
                    SwapChainLifecycleOperation::None,
                    std::memory_order_acquire);
            HRESULT result = E_FAIL;
            try
            {
                switch (operation)
                {
                case SwapChainLifecycleOperation::Ensure:
                    EnsureSwapChainOnWindowThread(
                        m_requestedSwapChainWidth.load(std::memory_order_relaxed),
                        m_requestedSwapChainHeight.load(std::memory_order_relaxed));
                    result = S_OK;
                    break;
                case SwapChainLifecycleOperation::Release:
                    ReleaseSwapChainOnWindowThread();
                    result = S_OK;
                    break;
                default:
                    result = E_UNEXPECTED;
                    break;
                }
            }
            catch (...)
            {
                result = E_FAIL;
            }
            // This release/acquire pair makes all owner-thread swapchain
            // writes visible before the render worker resumes from SendMessage.
            m_swapChainLifecycleResult.store(result, std::memory_order_release);
            return result;
        }

        void NotePipelineAdmissionBlocked() noexcept
        {
            m_pipelineAdmissionBlockedSinceLastFrame = true;
        }

        void RecordSourceTimingJournalOverflow(uint64_t overflowCount) noexcept
        {
            if (overflowCount == 0) return;
            FrameTimingRecord timing;
            timing.recordId = ++m_timingRecordId;
            timing.sourceTimingJournalOverflowCount = overflowCount;
            timing.upscalerMode = static_cast<uint32_t>(m_activeUpscalerMode);
            timing.pacingMode = static_cast<uint32_t>(m_framePacingConfig.pacingMode);
            timing.flags = aess::presentation::FrameTimingSyntheticGapRecord;
            timing.writerOverflowBefore = m_telemetry.DroppedFrameRecords();
            EnqueueDroppedTiming(
                timing,
                FrameDropReason::SourceTimingJournalOverflow,
                overflowCount);
        }

        void ObserveSourceTiming(
            int64_t sourcePresentQpc,
            uint64_t sourcePresentOrdinal,
            bool admittedFrame = false) noexcept
        {
            if (sourcePresentQpc <= 0)
            {
                return;
            }
            if (sourcePresentQpc == m_lastObservedSourceQpc &&
                sourcePresentOrdinal == m_lastObservedSourceOrdinal)
            {
                return;
            }

            const bool qpcRegression = m_lastObservedSourceQpc > 0 &&
                sourcePresentQpc <= m_lastObservedSourceQpc;
            const bool ordinalRegression = sourcePresentOrdinal > 0 &&
                m_lastObservedSourceOrdinal > 0 &&
                sourcePresentOrdinal <= m_lastObservedSourceOrdinal;
            if (qpcRegression || ordinalRegression)
            {
                // The sideband cadence journal can advance between its drain
                // and acquisition of the newest fence-complete texture.  An
                // admitted frame that still advances the admitted-frame
                // timeline is therefore older than the sideband cursor by
                // design, not a source timestamp regression.  Keep its exact
                // timestamp on the frame record, but do not rewind the pacer.
                const bool advancesAdmittedTimeline = sourcePresentOrdinal > 0
                    ? (m_lastTimingSourceOrdinal == 0 ||
                        sourcePresentOrdinal > m_lastTimingSourceOrdinal)
                    : (m_lastTimingSourceQpc == 0 ||
                        sourcePresentQpc > m_lastTimingSourceQpc);
                if (admittedFrame && advancesAdmittedTimeline)
                {
                    return;
                }
                FrameTimingRecord timing;
                timing.recordId = ++m_timingRecordId;
                timing.sourcePresentQpc = sourcePresentQpc;
                timing.sourcePresentOrdinal = sourcePresentOrdinal;
                timing.upscalerMode = static_cast<uint32_t>(m_activeUpscalerMode);
                timing.pacingMode = static_cast<uint32_t>(m_framePacingConfig.pacingMode);
                timing.flags = aess::presentation::FrameTimingValidSourcePresentQpc |
                    aess::presentation::FrameTimingSyntheticGapRecord;
                timing.writerOverflowBefore = m_telemetry.DroppedFrameRecords();
                EnqueueDroppedTiming(timing, FrameDropReason::TimestampRegression);
                ResetAddedIntervalResidualSampling();
            }

            m_framePacer.ObserveSource(
                sourcePresentQpc,
                sourcePresentOrdinal);
            m_lastObservedSourceQpc = sourcePresentQpc;
            m_lastObservedSourceOrdinal = sourcePresentOrdinal;
        }

        bool ToggleUpscalerBypass()
        {
            ResetS2D32PipelineSlots(FrameDropReason::UpscalerReset, false);
            ResetSynchronousReadyFrame(FrameDropReason::UpscalerReset);
            m_framePacer.ResetTimeline();
            ResetAddedIntervalResidualSampling();
            m_upscalerBypass = !m_upscalerBypass;
            m_activeUpscalerMode = EffectiveUpscalerMode();
            ResetFpsSampling();
            std::cout << "Upscaler toggle: " << (m_upscalerBypass
                ? (m_comparisonFilter == ComparisonFilter::Bilinear ? "bilinear (AEISU off)" : "nearest-neighbor (AEISU off)")
                : "AEISU on") << "\n";
            return m_upscalerBypass;
        }


        bool UpscalerBypassEnabled() const
        {
            return m_upscalerBypass;
        }

        UpscalerMode SelectedUpscalerMode() const
        {
            return m_upscalerMode;
        }

        UpscalerMode EffectiveUpscalerMode() const
        {
            if (!m_upscalerBypass) return m_upscalerMode;
            return m_comparisonFilter == ComparisonFilter::Bilinear ? UpscalerMode::BilinearComparison : UpscalerMode::NoUpscaler;
        }

        bool DebugViewOriginalActive() const noexcept
        {
            return m_upscalerBypass && m_comparisonFilter == ComparisonFilter::Nearest;
        }


        bool DebugViewEasuOnlyActive() const noexcept { return false; }


        bool ToggleFpsCounter()
        {
            m_fpsCounterEnabled = !m_fpsCounterEnabled;
            if (!m_fpsCounterEnabled && !m_telemetry.Enabled())
            {
                FlushPendingPresentationTimings(FrameDropReason::None);
            }
            ResetFpsSampling();
            return m_fpsCounterEnabled;
        }

        bool FpsCounterEnabled() const noexcept
        {
            return m_fpsCounterEnabled;
        }

        void HandlePacingEnvironmentChange(bool displayChanged)
        {
            if (displayChanged) RequireSdrDisplay(m_hwnd);
            const FrameDropReason resetReason = displayChanged
                ? FrameDropReason::PresentationReset
                : FrameDropReason::FocusReset;
            
            ResetS2D32PipelineSlots(resetReason, false);
            ResetSynchronousReadyFrame(resetReason);
            m_framePacer.ResetTimeline();
            ResetAddedIntervalResidualSampling();
        }

        double LastFlipMeterWaitMs() const noexcept
        {
            return m_lastFlipMeterWaitMs;
        }

        double AverageFlipMeterWaitMs() const noexcept
        {
            return m_flipMeterWaitCount != 0
                ? m_flipMeterWaitTotalMs / static_cast<double>(m_flipMeterWaitCount)
                : 0.0;
        }

        double AverageS2D32BlockingWaitMs() const noexcept
        {
            return m_s2d32BlockingWaitCount != 0
                ? m_s2d32BlockingWaitTotalMs / static_cast<double>(m_s2d32BlockingWaitCount)
                : 0.0;
        }

        bool PumpAsyncUpscaler(D3D11RenderResult& result)
        {
            RetireConsumedCaptureLeases();
            const UpscalerMode mode = EffectiveUpscalerMode();
            
            
            if (IsNeuralUpscaler(mode))
            {
                return PumpReadyS2D32Slots(result);
            }
            return PumpReadySynchronousFrame(result);
        }

        bool CanAcceptFrame() const noexcept
        {
            if (m_activeCaptureLeases.Full()) return false;
            if (HasSynchronousFrameWork())
            {
                for (const auto& slot : m_synchronousPipelineSlots)
                {
                    if (!SynchronousSlotOccupied(slot)) return true;
                }
                return false;
            }
            const UpscalerMode mode = EffectiveUpscalerMode();
            if (IsNeuralUpscaler(mode))
            {
                for (const auto& slot : m_s2d32PipelineSlots)
                {
                    if (slot.state == PipelineSlotState::Free) return true;
                }
                return false;
            }
            return true;
        }

        size_t AppendAsyncWaitHandles(HANDLE* handles, size_t capacity) const noexcept
        {
            if (!handles || capacity == 0) return 0;
            size_t count = 0;
            if (!m_activeCaptureLeases.Empty() && m_inputConsumedEvent)
            {
                // Pool return must wake independently of another capture or
                // neural completion, including during a source gap.
                handles[count++] = m_inputConsumedEvent.get();
            }
            for (const auto& slot : m_synchronousPipelineSlots)
            {
                if (slot.completionPending &&
                    slot.completionEvent &&
                    count < capacity)
                {
                    handles[count++] = slot.completionEvent.get();
                }
            }
            const UpscalerMode mode = EffectiveUpscalerMode();
            if (IsNeuralUpscaler(mode))
            {
                for (const auto& slot : m_s2d32PipelineSlots)
                {
                    if (slot.state == PipelineSlotState::Submitted && slot.runtime && count < capacity)
                    {
                        HANDLE event = slot.runtime->CompletionEvent();
                        if (event) handles[count++] = event;
                    }
                }
            }
            if (m_waitingForPresentationSlot &&
                m_frameLatencyWaitable &&
                count < capacity)
            {
                handles[count++] = m_frameLatencyWaitable;
            }
            return count;
        }

        void NotifyAsyncWaitHandleSignaled(HANDLE handle) noexcept
        {
            if (handle &&
                handle == m_frameLatencyWaitable &&
                m_waitingForPresentationSlot)
            {
                // The DXGI latency object is consumable. WaitForMultipleObjects
                // already acquired this eligibility token, so the subsequent
                // presentation pump must not poll and consume it a second time.
                m_frameLatencyWakeGranted = true;
            }
        }

        HANDLE PacingTimerEvent() const noexcept
        {
            
            return m_framePacer.TimerArmed() ? m_framePacer.TimerEvent() : nullptr;
        }

        void NotifyPacingTimerWake() noexcept
        {
            
            m_framePacer.FinishTimerWait();
        }

        D3D11RenderResult Render(const FrameInfo& frame, const std::filesystem::path& = {},
            uint64_t frameNumber = 0, double acquireMs = 0.0, uint64_t skippedFrames = 0)
        {
            if (!frame.d3d11Texture) throw std::runtime_error("Capture did not provide a GPU texture.");
            EnsureDevice(frame.d3d11Texture.Get());
            const FrameGeometry geometry = ResolveFrameGeometry(frame);
            m_activeUpscalerMode = EffectiveUpscalerMode();
            if (m_upscalerBypass)
                return RenderComparisonBypass(frame, geometry, frameNumber, acquireMs, skippedFrames);
            LeaveSynchronousRoute(FrameDropReason::UpscalerReset);
            return RenderS2D32Pipelined(frame, geometry, frameNumber, acquireMs, skippedFrames);
        }


    private:
        D3D11RenderResult RenderComparisonBypass(const FrameInfo& frame, const FrameGeometry& geometry,
            uint64_t frameNumber, double acquireMs, uint64_t skippedFrames)
        {
            const int64_t beginQpc = FramePacer::QueryQpc();
            UpdateSourceColorSpace(frame.colorSpace, frame.format, frame.colorProvenance);
            EnsureSwapChain(geometry.presentationWidth, geometry.presentationHeight);
            SynchronousRouteSignature signature;
            signature.mode = EffectiveUpscalerMode();
            signature.sourceFormat = geometry.sourceDesc.Format;
            signature.sourceColorSpace = frame.colorSpace;
            signature.sourceColorProvenance = frame.colorProvenance;
            signature.sourceWidth = geometry.sourceDesc.Width;
            signature.sourceHeight = geometry.sourceDesc.Height;
            signature.cropLeft = geometry.cropLeft;
            signature.cropTop = geometry.cropTop;
            signature.cropWidth = signature.workingWidth = geometry.cropWidth;
            signature.cropHeight = signature.workingHeight = geometry.cropHeight;
            signature.presentationWidth = signature.internalOutputWidth = geometry.presentationWidth;
            signature.presentationHeight = signature.internalOutputHeight = geometry.presentationHeight;
            ApplySynchronousRouteSignature(signature);
            auto* slot = FreeSynchronousFrameSlot();
            if (!slot) throw std::runtime_error("Comparison submission reached a full two-slot pipeline.");
            ObserveSourceTiming(frame.sourcePresentQpc, frame.sourcePresentOrdinal, true);
            UpdateSourceDimensions(geometry.sourceDesc.Width, geometry.sourceDesc.Height);
            UpdateToneConstants(geometry.presentationWidth, geometry.presentationHeight);
            D3D11RenderResult result;
            result.workingWidth = geometry.cropWidth;
            result.workingHeight = geometry.cropHeight;
            result.videoProcessorOutputWidth = geometry.presentationWidth;
            result.videoProcessorOutputHeight = geometry.presentationHeight;
            result.internalOutputScale = 1;
            result.upscalerMode = EffectiveUpscalerMode();
            result.selectedUpscalerMode = m_upscalerMode;
            result.upscalerBypass = true;
            result.shaderUpscaler = true;
            result.skippedFrames = skippedFrames;
            result.inputFormat = result.videoProcessorOutputFormat = frame.format;
            result.inputColorSpace = frame.colorSpace;
            result.inputColorProvenance = frame.colorProvenance;
            result.outputMode = m_outputMode;
            result.outputFormat = m_swapFormat;
            result.tone = m_toneConstants;
            BeginInputRead(frame.captureLease);
            slot->isDirectNoUpscaler = true;
            slot->directSourceTexture = frame.d3d11Texture;
            slot->directSourceSrv = frame.d3d11Srv;
            if (frame.captureLease) frame.captureLease->deferredOwnership = true;
            slot->directCaptureLease = frame.captureLease;
            slot->directSourceDesc = geometry.sourceDesc;
            slot->directCropLeft = geometry.cropLeft;
            slot->directCropTop = geometry.cropTop;
            slot->directCropWidth = geometry.cropWidth;
            slot->directCropHeight = geometry.cropHeight;
            slot->upscaledOutput = frame.d3d11Texture;
            slot->presentationSrv = frame.d3d11Srv;
            slot->outputWidth = geometry.presentationWidth;
            slot->outputHeight = geometry.presentationHeight;
            slot->captureArrivalCount = frame.captureArrivalCount;
            slot->submitOrder = ++m_synchronousSubmitOrder;
            slot->result = result;
            slot->usesPresentationBridge = true;
            slot->timing = BeginFrameTiming(frame, frameNumber, slot->submitOrder, beginQpc,
                geometry.cropWidth, geometry.cropHeight);
            if (m_profiler.Enabled())
            {
                m_profiler.BeginFrame(frameNumber);
                slot->perfSummary.frameNumber = frameNumber;
                slot->perfSummary.sourceVersion = frame.version;
                slot->perfSummary.skippedFrames = skippedFrames;
                slot->perfSummary.cpuAcquireMs = acquireMs;
                slot->perfSummary.cpuRenderMs = QpcElapsedMilliseconds(beginQpc, FramePacer::QueryQpc());
                SetPerfSummarySignature(slot->perfSummary, result.upscalerMode, result.selectedUpscalerMode,
                    result.workingWidth, result.workingHeight, result.videoProcessorOutputWidth,
                    result.videoProcessorOutputHeight, result.outputFormat);
                m_profiler.EndFrameDeferred(slot->perfSummary);
            }
            ArmCompletionFence(slot->completionEvent, slot->completionFenceValue);
            slot->completionPending = true;
            D3D11RenderResult completed;
            return PumpReadySynchronousFrame(completed) ? completed : result;
        }

        void WaitForPresentationSlot(FrameTimingRecord* timing = nullptr)
        {
            FlushContext();
            const int64_t beginQpc = FramePacer::QueryQpc();
            if (timing) timing->latencyGateBeginQpc = beginQpc;
            if (!m_frameLatencyWaitable)
            {
                m_lastFlipMeterWaitMs = 0.0;
                if (timing) timing->latencyGateEndQpc = FramePacer::QueryQpc();
                return;
            }

            const DWORD waitResult = WaitForSingleObjectEx(
                m_frameLatencyWaitable,
                1000,
                FALSE);
            const int64_t endQpc = FramePacer::QueryQpc();
            m_lastFlipMeterWaitMs = QpcElapsedMilliseconds(beginQpc, endQpc);
            m_flipMeterWaitTotalMs += m_lastFlipMeterWaitMs;
            ++m_flipMeterWaitCount;
            if (timing) timing->latencyGateEndQpc = endQpc;
            if (waitResult == WAIT_TIMEOUT)
            {
                ++m_flipMeterTimeoutCount;
                throw std::runtime_error(
                    "Timed out waiting for the diagnostic DXGI presentation slot.");
            }
            if (waitResult != WAIT_OBJECT_0)
            {
                ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()), "Wait for DXGI flip-meter slot");
            }
        }

        bool TryAcquirePresentationSlot(FrameTimingRecord* timing = nullptr)
        {
            RetireConsumedCaptureLeases();
            const int64_t nowQpc = FramePacer::QueryQpc();
            if (timing && timing->latencyGateBeginQpc == 0)
            {
                timing->latencyGateBeginQpc = nowQpc;
            }
            if (!m_frameLatencyWaitable)
            {
                m_waitingForPresentationSlot = false;
                m_frameLatencyWakeGranted = false;
                m_lastFlipMeterWaitMs = 0.0;
                if (timing) timing->latencyGateEndQpc = nowQpc;
                return true;
            }
            if (!m_waitingForPresentationSlot)
            {
                FlushContext();
            }

            if (!m_frameLatencyWakeGranted)
            {
                const DWORD waitResult = WaitForSingleObjectEx(
                    m_frameLatencyWaitable,
                    0,
                    FALSE);
                if (waitResult == WAIT_TIMEOUT)
                {
                    m_waitingForPresentationSlot = true;
                    return false;
                }
                if (waitResult != WAIT_OBJECT_0)
                {
                    ThrowIfFailed(
                        HRESULT_FROM_WIN32(GetLastError()),
                        "Poll DXGI flip-meter presentation slot");
                }
            }

            const int64_t acquiredQpc = FramePacer::QueryQpc();
            m_frameLatencyWakeGranted = false;
            m_waitingForPresentationSlot = false;
            if (timing)
            {
                timing->latencyGateEndQpc = acquiredQpc;
                m_lastFlipMeterWaitMs = static_cast<double>(
                    acquiredQpc - timing->latencyGateBeginQpc) * 1000.0 /
                    static_cast<double>(m_framePacer.QpcFrequency());
            }
            else
            {
                m_lastFlipMeterWaitMs = 0.0;
            }
            m_flipMeterWaitTotalMs += m_lastFlipMeterWaitMs;
            ++m_flipMeterWaitCount;
            return true;
        }

        void ArmCompletionFence(
            winrt::handle& completionEvent,
            uint64_t& fenceValue)
        {
            if (!completionEvent)
            {
                completionEvent.attach(
                    CreateEventW(nullptr, TRUE, FALSE, nullptr));
                if (!completionEvent)
                {
                    ThrowIfFailed(
                        HRESULT_FROM_WIN32(GetLastError()),
                        "Create D3D11 post-effect completion event");
                }
            }
            if (!m_context4 || !m_completionFence)
            {
                throw std::runtime_error(
                    "D3D11 post-effect completion fence is unavailable.");
            }
            if (!ResetEvent(completionEvent.get()))
            {
                ThrowIfFailed(
                    HRESULT_FROM_WIN32(GetLastError()),
                    "Reset D3D11 post-effect completion event");
            }
            const uint64_t nextFenceValue = ++m_completionFenceValue;
            ThrowIfFailed(
                m_context4->Signal(m_completionFence.Get(), nextFenceValue),
                "Signal D3D11 post-effect completion fence");
            ThrowIfFailed(
                m_completionFence->SetEventOnCompletion(
                    nextFenceValue,
                    completionEvent.get()),
                "Set D3D11 post-effect completion event");
            // Publish the timeline value only after both submission and event
            // registration succeed.  Reset paths must never wait on a value
            // whose wake was not installed.
            fenceValue = nextFenceValue;
            // Final-ready is itself the wake that permits Present, so there
            // may be no later draw or Present to make the driver submit this
            // otherwise-small command batch. Explicitly flush the fence
            // packet; this prevents adaptive/smooth holds from waiting on a
            // signal that is still buffered in the immediate context.
            m_context->Flush();
            m_contextDirty = false;
        }

        void WaitForCompletionFence(
            HANDLE completionEvent,
            uint64_t fenceValue) noexcept
        {
            if (!completionEvent || !m_completionFence || fenceValue == 0)
            {
                return;
            }
            while (m_completionFence->GetCompletedValue() < fenceValue)
            {
                const DWORD waitResult = WaitForSingleObject(
                    completionEvent,
                    5000);
                if (waitResult == WAIT_OBJECT_0)
                {
                    // Manual-reset events are wake hints; the non-consuming
                    // fence timeline remains authoritative.
                    continue;
                }
                if (!m_device || FAILED(m_device->GetDeviceRemovedReason()))
                {
                    return;
                }
                FlushContext();
            }
        }

        void BeginInputRead(const std::shared_ptr<CaptureLease>& lease)
        {
            if (!lease || (!lease->wgcFrame && !lease->onRetire)) return;
            if (!m_context4 || !m_inputConsumedFence || !m_inputConsumedEvent || !m_inputDrainQuery)
            {
                throw std::runtime_error("Capture source reads require GPU lifetime tracking.");
            }
            RetireConsumedCaptureLeases();
            const uint64_t fenceValue = m_inputConsumedFenceValue + 1;
            if (lease->consumedFence || !m_activeCaptureLeases.Track(lease, fenceValue))
            {
                throw std::runtime_error("Capture input lease already submitted or retirement queue full.");
            }
            m_inputConsumedFenceValue = fenceValue;
            lease->consumedFence = m_inputConsumedFence;
            lease->consumedFenceValue = fenceValue;
        }

        void SignalInputConsumed(const std::shared_ptr<CaptureLease>& lease, bool flush = true)
        {
            if (!lease || (!lease->wgcFrame && !lease->onRetire)) return;
            lease->deferredOwnership = false;
            ThrowIfFailed(
                m_context4->Signal(m_inputConsumedFence.Get(), lease->consumedFenceValue),
                "Signal D3D11 input consumed fence");
            if (flush)
            {
                // No later source or Present is required to submit this wake.
                m_context->Flush();
                m_contextDirty = false;
            }
            else
            {
                m_contextDirty = true;
            }
            RetireConsumedCaptureLeases();
        }

        void RetireConsumedCaptureLeases()
        {
            if (m_activeCaptureLeases.Empty()) return;
            m_activeCaptureLeases.RetireCompleted(m_inputConsumedFence->GetCompletedValue());
            const uint64_t next = m_activeCaptureLeases.FirstFenceValue();
            if (next == m_inputConsumedEventValue) return;
            ThrowIfFailed(ResetEvent(m_inputConsumedEvent.get()) ? S_OK :
                HRESULT_FROM_WIN32(GetLastError()), "Reset D3D11 input consumed event");
            m_inputConsumedEventValue = 0;
            if (next != 0)
            {
                ThrowIfFailed(m_inputConsumedFence->SetEventOnCompletion(next, m_inputConsumedEvent.get()),
                    "Arm D3D11 input consumed event");
                m_inputConsumedEventValue = next;
            }
        }

        void DrainCaptureReads() noexcept
        {
            if (m_activeCaptureLeases.Empty()) return;
            // A reserved fence may never have been signaled if preprocessing
            // threw. An event query after all issued commands covers that
            // failure as well as normal shutdown, without a per-frame wait.
            m_context->End(m_inputDrainQuery.Get());
            m_context->Flush();
            m_contextDirty = false;
            while (m_context->GetData(m_inputDrainQuery.Get(), nullptr, 0,
                D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK)
            {
                if (FAILED(m_device->GetDeviceRemovedReason())) break;
                Sleep(1);
            }
            m_activeCaptureLeases.ResetAfterDrain();
            m_inputConsumedEventValue = 0;
            if (m_inputConsumedEvent) ResetEvent(m_inputConsumedEvent.get());
        }

        bool PresentFrame(const char* failureContext, FrameTimingRecord* timing = nullptr)
        {
            // Borderless VRR requires interval zero and the tearing opt-in.
            // A VRR display performs tear-free scanout while it is in range.
            const UINT presentFlags = m_tearingSupported
                ? DXGI_PRESENT_ALLOW_TEARING
                : 0u;
            if (timing)
            {
                timing->presentCallBeginQpc = FramePacer::QueryQpc();
                timing->flags |= aess::presentation::FrameTimingPresentAttempted;
            }
            // The render worker exclusively owns the immediate context and
            // hot Present path. Swapchain creation/destruction still executes
            // synchronously on the HWND owner, but no per-frame event hop or
            // normal-priority UI-thread rendezvous remains.
            const HRESULT presentHr =
                m_swapChain->Present(0, presentFlags);
            if (timing) timing->presentCallEndQpc = FramePacer::QueryQpc();
            if (presentHr == DXGI_ERROR_WAS_STILL_DRAWING)
            {
                ++m_presentQueueBusyCount;
                return false;
            }
            ThrowIfFailed(presentHr, failureContext);
            if (timing)
            {
                timing->flags |= aess::presentation::FrameTimingPresentAccepted;
            }
            return true;
        }

        void CollectPresentTelemetry(FrameTimingRecord& timing)
        {
            if ((timing.flags & aess::presentation::FrameTimingPresentAccepted) == 0 ||
                (!m_fpsCounterEnabled && !m_telemetry.Enabled()))
            {
                return;
            }
            UINT presentCount = 0;
            if (SUCCEEDED(m_swapChain->GetLastPresentCount(&presentCount)))
            {
                timing.dxgiPresentCount = presentCount;
            }
        }

        struct SynchronousReadyFrame
        {
            ComPtr<ID3D11Texture2D> finalOutput;
            ComPtr<ID3D11RenderTargetView> finalOutputRtv;
            ComPtr<ID3D11ShaderResourceView> finalOutputSrv;
            ComPtr<ID3D11UnorderedAccessView> finalOutputUav;
            ComPtr<ID3D11Texture2D> upscaledOutput;
            ComPtr<ID3D11ShaderResourceView> presentationSrv;
            ComPtr<ID3D11Texture2D> directSourceTexture;
            ComPtr<ID3D11ShaderResourceView> directSourceSrv;
            std::shared_ptr<CaptureLease> directCaptureLease;
            D3D11_TEXTURE2D_DESC directSourceDesc{};
            UINT directCropLeft = 0;
            UINT directCropTop = 0;
            UINT directCropWidth = 0;
            UINT directCropHeight = 0;
            bool isDirectNoUpscaler = false;
            D3D11RenderResult result{};
            D3D11PerfFrameSummary perfSummary{};
            FrameTimingRecord timing{};
            uint64_t captureArrivalCount = 0;
            UINT outputWidth = 0;
            UINT outputHeight = 0;
            uint64_t completionFenceValue = 0;
            uint64_t submitOrder = 0;
            winrt::handle completionEvent;
            bool usesPresentationBridge = false;
            bool completionPending = false;
            bool ready = false;
        };

        struct SynchronousRouteSignature
        {
            UpscalerMode mode = UpscalerMode::NoUpscaler;
            DXGI_FORMAT sourceFormat = DXGI_FORMAT_UNKNOWN;
            DXGI_COLOR_SPACE_TYPE sourceColorSpace =
                DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
            CaptureColorProvenance sourceColorProvenance =
                CaptureColorProvenance::Unknown;
            UINT sourceWidth = 0;
            UINT sourceHeight = 0;
            UINT cropLeft = 0;
            UINT cropTop = 0;
            UINT cropWidth = 0;
            UINT cropHeight = 0;
            UINT workingWidth = 0;
            UINT workingHeight = 0;
            UINT internalOutputWidth = 0;
            UINT internalOutputHeight = 0;
            UINT presentationWidth = 0;
            UINT presentationHeight = 0;

            bool Equals(const SynchronousRouteSignature& other) const noexcept
            {
                return mode == other.mode &&
                    sourceFormat == other.sourceFormat &&
                    sourceColorSpace == other.sourceColorSpace &&
                    sourceColorProvenance == other.sourceColorProvenance &&
                    sourceWidth == other.sourceWidth &&
                    sourceHeight == other.sourceHeight &&
                    cropLeft == other.cropLeft &&
                    cropTop == other.cropTop &&
                    cropWidth == other.cropWidth &&
                    cropHeight == other.cropHeight &&
                    workingWidth == other.workingWidth &&
                    workingHeight == other.workingHeight &&
                    internalOutputWidth == other.internalOutputWidth &&
                    internalOutputHeight == other.internalOutputHeight &&
                    presentationWidth == other.presentationWidth &&
                    presentationHeight == other.presentationHeight;
            }
        };

        void BindSynchronousFinalOutput(
            SynchronousReadyFrame& slot,
            UINT width,
            UINT height)
        {
            bool compatible = false;
            if (slot.finalOutput)
            {
                D3D11_TEXTURE2D_DESC desc{};
                slot.finalOutput->GetDesc(&desc);
                compatible = desc.Width == width &&
                    desc.Height == height &&
                    desc.Format == kModelTextureFormat &&
                    (desc.BindFlags & D3D11_BIND_RENDER_TARGET) != 0 &&
                    (desc.BindFlags & D3D11_BIND_SHADER_RESOURCE) != 0 &&
                    (desc.BindFlags & D3D11_BIND_UNORDERED_ACCESS) != 0;
            }
            if (!compatible)
            {
                slot.finalOutputUav.Reset();
                slot.finalOutputSrv.Reset();
                slot.finalOutputRtv.Reset();
                slot.finalOutput.Reset();

                D3D11_TEXTURE2D_DESC desc{};
                desc.Width = width;
                desc.Height = height;
                desc.MipLevels = 1;
                desc.ArraySize = 1;
                desc.Format = kModelTextureFormat;
                desc.SampleDesc.Count = 1;
                desc.Usage = D3D11_USAGE_DEFAULT;
                desc.BindFlags = D3D11_BIND_RENDER_TARGET |
                    D3D11_BIND_SHADER_RESOURCE |
                    D3D11_BIND_UNORDERED_ACCESS;
                ThrowIfFailed(
                    m_device->CreateTexture2D(
                        &desc,
                        nullptr,
                        &slot.finalOutput),
                    "Create synchronous final output texture");
                ThrowIfFailed(
                    m_device->CreateRenderTargetView(
                        slot.finalOutput.Get(),
                        nullptr,
                        &slot.finalOutputRtv),
                    "Create synchronous final output RTV");
                ThrowIfFailed(
                    m_device->CreateShaderResourceView(
                        slot.finalOutput.Get(),
                        nullptr,
                        &slot.finalOutputSrv),
                    "Create synchronous final output SRV");
                ThrowIfFailed(
                    m_device->CreateUnorderedAccessView(
                        slot.finalOutput.Get(),
                        nullptr,
                        &slot.finalOutputUav),
                    "Create synchronous final output UAV");
            }

            m_videoProcessorOutput = slot.finalOutput;
            m_videoProcessorOutputRtv = slot.finalOutputRtv;
            m_videoProcessorOutputSrv = slot.finalOutputSrv;
            m_videoProcessorOutputUav = slot.finalOutputUav;
            m_videoProcessorOutputWidth = width;
            m_videoProcessorOutputHeight = height;
            m_videoProcessorOutputFormat = kModelTextureFormat;
        }

        void ResolveSynchronousFinalOutput(
            SynchronousReadyFrame& slot,
            UINT,
            UINT,
            ComPtr<ID3D11Texture2D>& renderedOutput)
        {
            if (renderedOutput.Get() != slot.finalOutput.Get())
            {
                throw std::runtime_error(
                    "Synchronous route did not render into its bound final output.");
            }
        }

        static bool SynchronousSlotOccupied(
            const SynchronousReadyFrame& slot) noexcept
        {
            return slot.completionPending || slot.ready;
        }

        void ClearSynchronousFrame(
            SynchronousReadyFrame& slot,
            bool releaseResources = false) noexcept
        {
            if (slot.isDirectNoUpscaler && slot.directCaptureLease)
            {
                SignalInputConsumed(slot.directCaptureLease);
            }
            slot.directSourceTexture.Reset();
            slot.directSourceSrv.Reset();
            slot.directCaptureLease.reset();
            slot.directSourceDesc = {};
            slot.directCropLeft = 0;
            slot.directCropTop = 0;
            slot.directCropWidth = 0;
            slot.directCropHeight = 0;
            slot.isDirectNoUpscaler = false;
            slot.upscaledOutput.Reset();
            slot.presentationSrv.Reset();
            slot.result = {};
            slot.perfSummary = {};
            slot.timing = {};
            slot.captureArrivalCount = 0;
            slot.outputWidth = 0;
            slot.outputHeight = 0;
            slot.completionFenceValue = 0;
            slot.submitOrder = 0;
            slot.usesPresentationBridge = false;
            slot.completionPending = false;
            slot.ready = false;
            if (releaseResources)
            {
                slot.finalOutputUav.Reset();
                slot.finalOutputSrv.Reset();
                slot.finalOutputRtv.Reset();
                slot.finalOutput.Reset();
                slot.completionEvent.close();
            }
        }

        size_t ReadySynchronousFrameCount() const noexcept
        {
            size_t count = 0;
            for (const auto& slot : m_synchronousPipelineSlots)
            {
                if (slot.ready) ++count;
            }
            return count;
        }

        bool HasSynchronousFrameWork() const noexcept
        {
            for (const auto& slot : m_synchronousPipelineSlots)
            {
                if (SynchronousSlotOccupied(slot)) return true;
            }
            return false;
        }

        SynchronousReadyFrame* FreeSynchronousFrameSlot() noexcept
        {
            for (auto& slot : m_synchronousPipelineSlots)
            {
                if (!SynchronousSlotOccupied(slot)) return &slot;
            }
            return nullptr;
        }

        SynchronousReadyFrame* OldestReadySynchronousFrame() noexcept
        {
            SynchronousReadyFrame* oldest = nullptr;
            for (auto& slot : m_synchronousPipelineSlots)
            {
                if (slot.ready &&
                    (!oldest || slot.submitOrder < oldest->submitOrder))
                {
                    oldest = &slot;
                }
            }
            return oldest;
        }

        SynchronousReadyFrame* NewestReadySynchronousFrame() noexcept
        {
            SynchronousReadyFrame* newest = nullptr;
            for (auto& slot : m_synchronousPipelineSlots)
            {
                if (slot.ready &&
                    (!newest || slot.submitOrder > newest->submitOrder))
                {
                    newest = &slot;
                }
            }
            return newest;
        }

        void DiscardSynchronousFrame(
            SynchronousReadyFrame& slot,
            FrameDropReason reason) noexcept
        {
            if (slot.timing.recordId != 0)
            {
                EnqueueDroppedTiming(slot.timing, reason);
            }
            if (m_profiler.Enabled())
            {
                m_profiler.CancelDeferredFrame(slot.perfSummary.frameNumber);
            }
            ClearSynchronousFrame(slot);
        }

        void ResetSynchronousReadyFrame(
            FrameDropReason reason,
            bool releaseResources = false) noexcept
        {
            m_waitingForPresentationSlot = false;
            m_frameLatencyWakeGranted = false;
            for (auto& slot : m_synchronousPipelineSlots)
            {
                if (slot.completionPending && slot.completionEvent)
                {
                    // Reset/reconfiguration is not a hot-path wait. Finish the
                    // queued D3D work before reusing its final surface.
                    WaitForCompletionFence(
                        slot.completionEvent.get(),
                        slot.completionFenceValue);
                }
                if (SynchronousSlotOccupied(slot))
                {
                    DiscardSynchronousFrame(slot, reason);
                }
                if (releaseResources)
                {
                    ClearSynchronousFrame(slot, true);
                }
            }
            m_profiler.CancelFrame();
        }

        void ApplySynchronousRouteSignature(
            const SynchronousRouteSignature& signature)
        {
            if (m_synchronousRouteSignatureValid &&
                !m_synchronousRouteSignature.Equals(signature))
            {
                ResetSynchronousReadyFrame(
                    FrameDropReason::ResizeReset,
                    false);
                m_framePacer.ResetTimeline();
                ResetAddedIntervalResidualSampling();
            }
            m_synchronousRouteSignature = signature;
            m_synchronousRouteSignatureValid = true;
        }

        void LeaveSynchronousRoute(FrameDropReason reason)
        {
            if (!m_synchronousRouteSignatureValid &&
                !HasSynchronousFrameWork())
            {
                return;
            }
            ResetSynchronousReadyFrame(reason, false);
            m_synchronousRouteSignature = {};
            m_synchronousRouteSignatureValid = false;
            m_framePacer.ResetTimeline();
            ResetAddedIntervalResidualSampling();
        }

        void CollectSynchronousCompletion() noexcept
        {
            if (!m_completionFence) return;
            const uint64_t completedFenceValue =
                m_completionFence->GetCompletedValue();
            for (auto& pending : m_synchronousPipelineSlots)
            {
                if (!pending.completionPending ||
                    completedFenceValue < pending.completionFenceValue)
                {
                    continue;
                }

                const int64_t completionQpc = FramePacer::QueryQpc();
                pending.completionPending = false;
                pending.ready = true;
                pending.timing.effectCompleteQpc = completionQpc;
                pending.timing.readyQueueEntryQpc = completionQpc;
                pending.timing.readyDepthAtEntry =
                    static_cast<uint32_t>(ReadySynchronousFrameCount());
                pending.timing.flags |=
                    aess::presentation::FrameTimingEffectCompleted |
                    aess::presentation::FrameTimingEnteredReadyQueue;
                ++m_inferredFrameCount;
                m_framePacer.ObserveCompletion(
                    completionQpc,
                    pending.timing.effectSubmitQpc);
                StampCompletionPacingTelemetry(pending.timing);
            }
        }

        bool PumpReadySynchronousFrame(D3D11RenderResult& completedResult)
        {
            CollectSynchronousCompletion();
            // Keep capture and preprocessing running into the bounded
            // newest completed source is selected as soon as the sequence
            // interpolation pair in flight.
            
            size_t readyDepth = ReadySynchronousFrameCount();
            if (readyDepth == 0) return false;
            const int64_t nowQpc = FramePacer::QueryQpc();
            SynchronousReadyFrame* selected = nullptr;
            SynchronousReadyFrame* newest = NewestReadySynchronousFrame();
            bool staleCadence = false;
            for (const auto& slot : m_synchronousPipelineSlots)
            {
                staleCadence |= slot.ready &&
                    TimingOlderThanTwoPeriods(slot.timing, nowQpc);
            }
            if (staleCadence)
            {
                selected = newest;
                for (auto& slot : m_synchronousPipelineSlots)
                {
                    if (slot.ready && &slot != selected)
                    {
                        slot.timing.readyDepthBeforePresent =
                            static_cast<uint32_t>(readyDepth);
                        DiscardSynchronousFrame(
                            slot,
                            FrameDropReason::StaleOverTwoPeriods);
                    }
                }
                readyDepth = 1;
                m_framePacer.ResetTimeline();
            }
            else if (m_framePacer.TargetQueueDepth() == 0)
            {
                selected = newest;
                for (auto& slot : m_synchronousPipelineSlots)
                {
                    if (slot.ready && &slot != selected)
                    {
                        slot.timing.readyDepthBeforePresent =
                            static_cast<uint32_t>(readyDepth);
                        DiscardSynchronousFrame(
                            slot,
                            FrameDropReason::ReadySuperseded);
                    }
                }
                readyDepth = 1;
            }
            else
            {
                selected = OldestReadySynchronousFrame();
            }
            if (!selected) return false;

            const bool readyToPresent = m_framePacer.ReadyToPresent(
                    nowQpc,
                    readyDepth,
                    false,
                    true,
                    selected->timing.sourcePresentQpc);
            selected->timing.targetDeadlineQpc = m_framePacer.DeadlineQpc();
            selected->timing.readyDepthBeforePresent =
                static_cast<uint32_t>(readyDepth);
            selected->timing.queueTarget = m_framePacer.TargetQueueDepth();
            if (!readyToPresent) return false;
            const int64_t setupBeginQpc = FramePacer::QueryQpc();

            auto& pending = *selected;
            D3D11RenderResult result = pending.result;
            
            if (!TryAcquirePresentationSlot(&pending.timing)) return false;
            ApplyToneConstantsSnapshot(result.tone);
            const int64_t presentBeginQpc = FramePacer::QueryQpc();
            ComPtr<ID3D11Texture2D> backBuffer = pending.upscaledOutput;
            if (pending.usesPresentationBridge)
            {
                if (pending.isDirectNoUpscaler)
                {
                    PresentDirectNoUpscalerToSwapChain(pending, backBuffer);
                }
                else
                {
                    PresentVideoOutputToSwapChain(
                        pending.outputWidth,
                        pending.outputHeight,
                        pending.presentationSrv.Get(),
                        backBuffer);
                }
            }
            RenderFpsCounterOverlay(pending.outputWidth, pending.outputHeight);
            result.presentMs = QpcElapsedMilliseconds(
                presentBeginQpc,
                FramePacer::QueryQpc());

            m_context->Flush();
            m_framePacer.WaitForPresentDeadline(setupBeginQpc);
            const bool presentAccepted = PresentFrame(
                "Event-driven synchronous flip-metered swapchain Present",
                &pending.timing);
            result.presentCallMs = QpcElapsedMilliseconds(
                pending.timing.presentCallBeginQpc,
                pending.timing.presentCallEndQpc);
            CollectPresentTelemetry(pending.timing);
            result.presentQueueBusyCount = m_presentQueueBusyCount;
            result.gpuFrameMs = m_profiler.LastFrameGpuMs();

            if (m_profiler.Enabled())
            {
                pending.perfSummary.cpuRenderMs += result.presentMs + result.presentCallMs;
                pending.perfSummary.cpuPresentSetupMs = result.presentMs;
                pending.perfSummary.cpuPresentCallMs = result.presentCallMs;
                m_profiler.FinalizeDeferredFrame(
                    pending.perfSummary.frameNumber,
                    pending.perfSummary);
            }
            result.gpuFrameMs = m_profiler.LastFrameGpuMs();

            pending.ready = false;
            const size_t readyDepthAfterPresent = ReadySynchronousFrameCount();
            pending.timing.readyDepthAfterPresent =
                static_cast<uint32_t>(readyDepthAfterPresent);
            if (presentAccepted)
            {
                m_framePacer.OnPresented(
                    pending.timing.presentCallBeginQpc,
                    pending.timing.sourcePresentQpc,
                    pending.timing.effectCompleteQpc,
                    readyDepthAfterPresent,
                    true);
            }
            else
            {
                pending.timing.dropReason = FrameDropReason::PresentRejected;
                m_framePacer.OnUnderrun(FramePacer::QueryQpc());
            }
            UpdateFpsSampling(
                pending.captureArrivalCount,
                presentAccepted,
                &pending.timing);
            completedResult = result;
            m_synchronousLastPresentedResult = result;
            ClearSynchronousFrame(pending);
            return true;
        }

        struct S2D32PipelineSlot
        {
            ComPtr<ID3D11Texture2D> input;
            ComPtr<ID3D11RenderTargetView> inputRtv;
            ComPtr<ID3D11ShaderResourceView> inputSrv;
            ComPtr<ID3D11UnorderedAccessView> inputUav;
            ComPtr<ID3D11Texture2D> output;
            ComPtr<ID3D11ShaderResourceView> outputSrv;
            std::unique_ptr<aess::neural::CudaRuntime> runtime;
            D3D11RenderResult result{};
            uint64_t submitOrder = 0;
            uint64_t sourceVersion = 0;
            uint64_t captureArrivalCount = 0;
            uint64_t frameNumber = 0;
            uint64_t sourcePresentOrdinal = 0;
            int64_t sourcePresentQpc = 0;
            int64_t submitQpc = 0;
            int64_t completionQpc = 0;
            double acquireMs = 0.0;
            double cpuSubmitMs = 0.0;
            aess::neural::FrameTimings timings{};
            aess::neural::RuntimeOutputMode outputMode =
                aess::neural::RuntimeOutputMode::Neural;
            FrameTimingRecord timing{};
            ComPtr<ID3D11Texture2D> preparedOutput;
            ComPtr<ID3D11ShaderResourceView> preparedPresentationSrv;




            ToneConstants preparedTone{};
            UINT preparedOutputWidth = 0;
            UINT preparedOutputHeight = 0;


            bool prepared = false;
            bool finalReady = false;
            PipelineSlotState state = PipelineSlotState::Free;
        };

        static void ClearPreparedS2D32Slot(S2D32PipelineSlot& slot) noexcept
        {
            slot.preparedOutput.Reset();
            slot.preparedPresentationSrv.Reset();

            slot.preparedTone = {};
            slot.preparedOutputWidth = 0;
            slot.preparedOutputHeight = 0;
            slot.prepared = false;
            slot.finalReady = false;
        }

        void ResetS2D32PipelineSlots(
            FrameDropReason reason = FrameDropReason::PresentationReset,
            bool releaseResources = true) noexcept
        {
            m_waitingForPresentationSlot = false;
            m_frameLatencyWakeGranted = false;
            for (auto& slot : m_s2d32PipelineSlots)
            {
                if (slot.runtime && slot.runtime->IsPending())
                {
                    try
                    {
                        slot.runtime->Wait();
                    }
                    catch (...)
                    {
                    }
                }
                if (slot.state != PipelineSlotState::Free && slot.timing.recordId != 0)
                {
                    EnqueueDroppedTiming(slot.timing, reason);
                }
                if (releaseResources)
                {
                    // Tear down CUDA's imported external-memory objects while
                    // their backing D3D11 textures are still alive. The two
                    // runtimes share an engine, but each slot owns its imports.
                    slot.runtime.reset();
                    slot = {};
                }
                else
                {
                    ClearPreparedS2D32Slot(slot);
                    slot.result = {};
                    slot.submitOrder = 0;
                    slot.sourceVersion = 0;
                    slot.captureArrivalCount = 0;
                    slot.frameNumber = 0;
                    slot.sourcePresentOrdinal = 0;
                    slot.sourcePresentQpc = 0;
                    slot.submitQpc = 0;
                    slot.completionQpc = 0;
                    slot.acquireMs = 0.0;
                    slot.cpuSubmitMs = 0.0;
                    slot.timings = {};
                    slot.outputMode = aess::neural::RuntimeOutputMode::Neural;
                    slot.timing = {};
                    slot.state = PipelineSlotState::Free;
                }
            }
            if (releaseResources)
            {
                m_s2d32PipelineWidth = 0;
                m_s2d32PipelineHeight = 0;
            }
            m_s2d32SubmitOrder = 0;
            m_s2d32HasPresented = false;
            m_s2d32LastPresentedResult = {};
            m_s2d32LastTimings = {};
        }

        void EnsureS2D32PipelineSlots(
            UINT inputWidth,
            UINT inputHeight,
            aess::neural::RuntimePrecision precision,
            aess::neural::ModelKind modelKind)
        {
            const UINT outputWidth = inputWidth * 2;
            const UINT outputHeight = inputHeight * 2;
            if (m_s2d32PipelineWidth == inputWidth &&
                m_s2d32PipelineHeight == inputHeight &&
                m_s2d32PipelinePrecision == precision &&
                m_s2d32PipelineModelKind == modelKind &&
                m_s2d32PipelineSlots[0].runtime &&
                m_s2d32PipelineSlots[1].runtime)
            {
                return;
            }

            const bool modelChanged = m_s2d32PipelineModelKind != modelKind;
            ResetS2D32PipelineSlots();
            m_framePacer.ResetTimeline();
            m_s2d32Runtime.reset();
            if (modelChanged) m_s2d32StartupReported = false;

            D3D11_TEXTURE2D_DESC inputDesc{};
            inputDesc.Width = inputWidth;
            inputDesc.Height = inputHeight;
            inputDesc.MipLevels = 1;
            inputDesc.ArraySize = 1;
            inputDesc.Format = kModelTextureFormat;
            inputDesc.SampleDesc.Count = 1;
            inputDesc.Usage = D3D11_USAGE_DEFAULT;
            inputDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
            inputDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

            D3D11_TEXTURE2D_DESC outputDesc = inputDesc;
            outputDesc.Width = outputWidth;
            outputDesc.Height = outputHeight;

            for (size_t index = 0; index < m_s2d32PipelineSlots.size(); ++index)
            {
                auto& slot = m_s2d32PipelineSlots[index];
                ThrowIfFailed(
                    aess::neural::CreateInteropTexture(m_device.Get(), inputDesc, &slot.input),
                    "Create pipelined neural input texture");
                ThrowIfFailed(
                    m_device->CreateRenderTargetView(slot.input.Get(), nullptr, &slot.inputRtv),
                    "Create pipelined neural input RTV");
                ThrowIfFailed(
                    m_device->CreateShaderResourceView(slot.input.Get(), nullptr, &slot.inputSrv),
                    "Create pipelined neural input SRV");
                ThrowIfFailed(
                    m_device->CreateUnorderedAccessView(slot.input.Get(), nullptr, &slot.inputUav),
                    "Create pipelined neural input UAV");
                ThrowIfFailed(
                    aess::neural::CreateInteropTexture(m_device.Get(), outputDesc, &slot.output),
                    "Create pipelined neural output texture");
                ThrowIfFailed(
                    m_device->CreateShaderResourceView(slot.output.Get(), nullptr, &slot.outputSrv),
                    "Create pipelined neural output SRV");
                slot.runtime = index == 0
                    ? std::make_unique<aess::neural::CudaRuntime>(
                        modelKind,
                        NeuralModelPath(modelKind))
                    : m_s2d32PipelineSlots[0].runtime->CreateSibling();
                slot.runtime->SetTimingMode(
                    m_profiler.Enabled()
                        ? aess::neural::RuntimeTimingMode::Detailed
                        : aess::neural::RuntimeTimingMode::TotalOnly);
                slot.runtime->Configure(
                    m_device.Get(),
                    slot.input.Get(),
                    slot.output.Get(),
                    inputWidth,
                    inputHeight,
                    precision);
            }

            m_s2d32PipelineWidth = inputWidth;
            m_s2d32PipelineHeight = inputHeight;
            m_s2d32PipelinePrecision = precision;
            m_s2d32PipelineModelKind = modelKind;
            m_videoProcessorOutputWidth = outputWidth;
            m_videoProcessorOutputHeight = outputHeight;
            m_videoProcessorOutputFormat = kModelTextureFormat;
            if (!m_s2d32StartupReported)
            {
                std::cout << m_s2d32PipelineSlots[0].runtime->StartupReport()
                    << " pipeline_slots=2 shared_execution_engines=1 shared_streams=1"
                    << " scheduling=double-buffered-ordered-single-inference\n";
                m_s2d32StartupReported = true;
            }
        }

        S2D32PipelineSlot* OldestSubmittedS2D32Slot()
        {
            S2D32PipelineSlot* oldest = nullptr;
            for (auto& slot : m_s2d32PipelineSlots)
            {
                if (slot.state == PipelineSlotState::Submitted &&
                    (!oldest || slot.submitOrder < oldest->submitOrder))
                {
                    oldest = &slot;
                }
            }
            return oldest;
        }

        S2D32PipelineSlot* NewestSubmittedS2D32Slot()
        {
            S2D32PipelineSlot* newest = nullptr;
            for (auto& slot : m_s2d32PipelineSlots)
            {
                if (slot.state == PipelineSlotState::Submitted &&
                    (!newest || slot.submitOrder > newest->submitOrder))
                {
                    newest = &slot;
                }
            }
            return newest;
        }

        S2D32PipelineSlot* FreeS2D32Slot()
        {
            for (auto& slot : m_s2d32PipelineSlots)
            {
                if (slot.state == PipelineSlotState::Free)
                {
                    return &slot;
                }
            }
            return nullptr;
        }

        void SubmitS2D32Inference(S2D32PipelineSlot& slot)
        {
            const S2D32PipelineSlot* dependencySlot =
                NewestSubmittedS2D32Slot();
            slot.runtime->ProcessAsyncAfter(
                m_s2d32PipelinePrecision,
                dependencySlot ? dependencySlot->runtime.get() : nullptr,
                slot.outputMode);
            slot.state = PipelineSlotState::Submitted;
        }

        size_t ReadyS2D32SlotCount() const noexcept
        {
            size_t count = 0;
            for (const auto& slot : m_s2d32PipelineSlots)
            {
                // Each neural slot already owns a retained completed output.
                // Shared post-processing is deferred pacing setup and must
                // not hide that completion from the bounded ready queue.
                if (slot.state == PipelineSlotState::Ready) ++count;
            }
            return count;
        }

        void MarkS2D32SlotFinalReady(
            S2D32PipelineSlot& slot,
            int64_t completionQpc,
            double serviceDurationMs) noexcept
        {
            if ((slot.timing.flags &
                aess::presentation::FrameTimingEnteredReadyQueue) == 0)
            {
                slot.timing.readyQueueEntryQpc = completionQpc;
                slot.timing.readyDepthAtEntry =
                    static_cast<uint32_t>(ReadyS2D32SlotCount());
                slot.timing.flags |=
                    aess::presentation::FrameTimingEnteredReadyQueue;
                ++m_inferredFrameCount;
            }
            slot.completionQpc = completionQpc;
            slot.finalReady = true;
            slot.timing.effectCompleteQpc = completionQpc;
            slot.timing.flags |=
                aess::presentation::FrameTimingEffectCompleted;
            m_framePacer.ObserveCompletion(
                completionQpc,
                slot.submitQpc,
                serviceDurationMs);
            StampCompletionPacingTelemetry(slot.timing);
        }

        S2D32PipelineSlot* OldestReadyS2D32Slot()
        {
            S2D32PipelineSlot* oldest = nullptr;
            for (auto& slot : m_s2d32PipelineSlots)
            {
                if (slot.state == PipelineSlotState::Ready &&
                    (!oldest || slot.submitOrder < oldest->submitOrder))
                {
                    oldest = &slot;
                }
            }
            return oldest;
        }

        S2D32PipelineSlot* NewestReadyS2D32Slot()
        {
            S2D32PipelineSlot* newest = nullptr;
            for (auto& slot : m_s2d32PipelineSlots)
            {
                if (slot.state == PipelineSlotState::Ready &&
                    (!newest || slot.submitOrder > newest->submitOrder))
                {
                    newest = &slot;
                }
            }
            return newest;
        }

        void CollectReadyS2D32Slots()
        {
            for (;;)
            {
                S2D32PipelineSlot* oldest = OldestSubmittedS2D32Slot();
                if (!oldest) return;
                aess::neural::FrameTimings timings{};
                if (!oldest->runtime->TryCollect(timings)) return;
                oldest->timings = timings;
                oldest->completionQpc = FramePacer::QueryQpc();
                oldest->state = PipelineSlotState::Ready;
                MarkS2D32SlotFinalReady(
                    *oldest,
                    oldest->completionQpc,
                    timings.totalCudaMs);
            }
        }

        static void ComputeS2D32SwapSurfaceSize(
            UINT clientWidth,
            UINT clientHeight,
            UINT neuralWidth,
            UINT neuralHeight,
            UINT& swapWidth,
            UINT& swapHeight)
        {
            const double downscale = std::min({
                1.0,
                static_cast<double>(clientWidth) / static_cast<double>(std::max<UINT>(1, neuralWidth)),
                static_cast<double>(clientHeight) / static_cast<double>(std::max<UINT>(1, neuralHeight)) });
            swapWidth = std::max<UINT>(1, static_cast<UINT>(std::llround(neuralWidth * downscale)));
            swapHeight = std::max<UINT>(1, static_cast<UINT>(std::llround(neuralHeight * downscale)));
        }



        void PrepareS2D32Presentation(S2D32PipelineSlot& slot)
        {
            auto& result = slot.result;
            UINT clientWidth = 1;
            UINT clientHeight = 1;
            GetClientSize(m_hwnd, clientWidth, clientHeight);
            ComputeS2D32SwapSurfaceSize(
                clientWidth,
                clientHeight,
                result.videoProcessorOutputWidth,
                result.videoProcessorOutputHeight,
                slot.preparedOutputWidth,
                slot.preparedOutputHeight);
            EnsureSwapChain(slot.preparedOutputWidth, slot.preparedOutputHeight);
            m_videoProcessorOutputWidth = result.videoProcessorOutputWidth;
            m_videoProcessorOutputHeight = result.videoProcessorOutputHeight;
            m_videoProcessorOutputFormat = kModelTextureFormat;
            slot.preparedOutput = slot.output;
            slot.preparedPresentationSrv = slot.outputSrv;
            slot.preparedTone = result.tone;
            
            slot.prepared = true;
        }

        bool PrepareOrPresentCompletedS2D32Slot(
            S2D32PipelineSlot& slot,
            const aess::neural::FrameTimings& timings,
            bool presentAllowed,
            int64_t setupBeginQpc = 0)
        {
            D3D11RenderResult result = slot.result;
            m_s2d32LastTimings = timings;
            result.videoProcessorMs = timings.totalCudaMs;
            result.gpuFrameMs = timings.totalCudaMs;

            UINT outputWidth = slot.preparedOutputWidth;
            UINT outputHeight = slot.preparedOutputHeight;
            ComPtr<ID3D11Texture2D> upscaledOutput = slot.preparedOutput;
            ComPtr<ID3D11ShaderResourceView> presentationSrv =
                slot.preparedPresentationSrv;
            if (!slot.prepared)
            {
                PrepareS2D32Presentation(slot);
                result = slot.result;
                outputWidth = slot.preparedOutputWidth;
                outputHeight = slot.preparedOutputHeight;
                upscaledOutput = slot.preparedOutput;
                presentationSrv = slot.preparedPresentationSrv;
            }

            if (!presentAllowed) return false;
            if (!TryAcquirePresentationSlot(&slot.timing)) return false;

            // Later capture submissions can update global tone constants. The
            // prepared frame retains the exact constants associated with its
            // source before the backbuffer pass is emitted.
            ApplyToneConstantsSnapshot(slot.preparedTone);

            D3D11PerfFrameSummary presentPerfSummary;
            if (m_profiler.Enabled())
            {
                presentPerfSummary.frameNumber = slot.frameNumber;
                presentPerfSummary.sourceVersion = slot.sourceVersion;
                presentPerfSummary.skippedFrames = result.skippedFrames;
                SetPerfSummarySignature(
                    presentPerfSummary,
                    result.upscalerMode,
                    result.selectedUpscalerMode,
                    result.workingWidth,
                    result.workingHeight,
                    result.videoProcessorOutputWidth,
                    result.videoProcessorOutputHeight,
                    m_swapFormat);
                aess::presentation::CopyTelemetryText(
                    presentPerfSummary.totalStageName,
                    "d3d-present-total");
                m_profiler.BeginFrame(slot.frameNumber);
            }
            const int64_t presentBeginQpc = FramePacer::QueryQpc();
            ComPtr<ID3D11Texture2D> backBuffer = upscaledOutput;
            PresentVideoOutputToSwapChain(
                outputWidth,
                outputHeight,
                presentationSrv.Get(),
                backBuffer);
            RenderFpsCounterOverlay(outputWidth, outputHeight);
            result.presentMs = QpcElapsedMilliseconds(
                presentBeginQpc,
                FramePacer::QueryQpc());
            if (m_profiler.Enabled())
            {
                presentPerfSummary.cpuPresentSetupMs = result.presentMs;
                m_profiler.EndFrame(presentPerfSummary);
            }

            m_context->Flush();
            m_framePacer.WaitForPresentDeadline(setupBeginQpc);
            const int64_t pacingPresentQpc = FramePacer::QueryQpc();
            bool presentAccepted = false;
            presentAccepted = PresentFrame(
                "Pipelined neural flip-metered swapchain Present",
                &slot.timing);
            result.presentCallMs = QpcElapsedMilliseconds(
                slot.timing.presentCallBeginQpc,
                slot.timing.presentCallEndQpc);
            CollectPresentTelemetry(slot.timing);
            result.presentQueueBusyCount = m_presentQueueBusyCount;
            result.outputMode = m_outputMode;
            result.videoProcessorOutputFormat = kModelTextureFormat;
            result.outputFormat = m_swapFormat;
            result.tone = m_toneConstants;

            if (m_profiler.Enabled())
            {
                D3D11PerfFrameSummary perfSummary;
                perfSummary.frameNumber = slot.frameNumber;
                perfSummary.sourceVersion = slot.sourceVersion;
                perfSummary.skippedFrames = result.skippedFrames;
                perfSummary.cpuAcquireMs = slot.acquireMs;
                perfSummary.cpuRenderMs = slot.cpuSubmitMs + result.presentMs + result.presentCallMs;
                perfSummary.cpuPreprocessMs = result.preprocessMs;
                perfSummary.cpuUpscaleMs = result.videoProcessorMs;
                perfSummary.cpuPresentSetupMs = result.presentMs;
                perfSummary.cpuPresentCallMs = result.presentCallMs;
                SetPerfSummarySignature(
                    perfSummary,
                    result.upscalerMode,
                    result.selectedUpscalerMode,
                    result.workingWidth,
                    result.workingHeight,
                    result.videoProcessorOutputWidth,
                    result.videoProcessorOutputHeight,
                    result.outputFormat);
                const std::array<D3D11GpuProfiler::ExternalStageTiming, 9>
                    cudaStages{{
                        { "interop-map-cpu", timings.interopMapMs },
                        { "cuda-easu-pack", timings.easuPackMs },
                        { "cuda-body-layer-1", timings.bodyLayerMs[0] },
                        { "cuda-body-layer-2", timings.bodyLayerMs[1] },
                        { "cuda-body-layer-3", timings.bodyLayerMs[2] },
                        { "cuda-tail-conversion", timings.tailConversionMs },
                        { "cuda-tail-gemm", timings.tailGemmMs },
                        { "cuda-composition", timings.compositionMs },
                        { "interop-unmap-cpu", timings.interopUnmapMs },
                    }};
                m_profiler.RecordExternalFrame(perfSummary, cudaStages, timings.totalCudaMs);
            }
            m_s2d32LastPresentedResult = result;
            m_s2d32HasPresented = true;
            ClearPreparedS2D32Slot(slot);
            slot.state = PipelineSlotState::Free;
            slot.timing.readyDepthAfterPresent = static_cast<uint32_t>(ReadyS2D32SlotCount());
            if (presentAccepted)
            {
                m_framePacer.OnPresented(
                    slot.timing.presentCallBeginQpc > 0
                        ? slot.timing.presentCallBeginQpc
                        : pacingPresentQpc,
                    slot.sourcePresentQpc,
                    slot.completionQpc,
                    ReadyS2D32SlotCount());
            }
            else
            {
                slot.timing.dropReason = FrameDropReason::PresentRejected;
                m_framePacer.OnUnderrun(FramePacer::QueryQpc());
            }
            UpdateFpsSampling(
                slot.captureArrivalCount,
                presentAccepted,
                &slot.timing);
            return true;
        }

        bool PumpReadyS2D32Slots(D3D11RenderResult& result)
        {
            CollectReadyS2D32Slots();
            // the active pair has emitted its endpoint.
            
            size_t readyDepth = ReadyS2D32SlotCount();
            bool haveModelCompletion = false;
            for (const auto& slot : m_s2d32PipelineSlots)
            {
                haveModelCompletion |= slot.state == PipelineSlotState::Ready;
            }
            if (!haveModelCompletion) return false;

            const int64_t nowQpc = FramePacer::QueryQpc();
            S2D32PipelineSlot* selected = nullptr;
            S2D32PipelineSlot* newest = NewestReadyS2D32Slot();
            bool staleCadence = false;
            for (const auto& slot : m_s2d32PipelineSlots)
            {
                staleCadence |= slot.state == PipelineSlotState::Ready &&
                    TimingOlderThanTwoPeriods(slot.timing, nowQpc);
            }
            if (staleCadence)
            {
                // Phase correction never drains old work. Keep only the
                // newest completed frame and make it the immediate anchor.
                selected = newest;
                for (auto& slot : m_s2d32PipelineSlots)
                {
                    if (slot.state == PipelineSlotState::Ready && &slot != selected)
                    {
                        slot.timing.readyDepthBeforePresent = static_cast<uint32_t>(readyDepth);
                        EnqueueDroppedTiming(slot.timing, FrameDropReason::StaleOverTwoPeriods);
                        ClearPreparedS2D32Slot(slot);
                        slot.state = PipelineSlotState::Free;
                    }
                }
                readyDepth = 1;
                m_framePacer.ResetTimeline();
            }
            else if (m_framePacer.TargetQueueDepth() == 0)
            {
                selected = newest;
                // Low-latency scheduling never emits an older completion once
                // a newer one is ready.
                for (auto& slot : m_s2d32PipelineSlots)
                {
                    if (slot.state == PipelineSlotState::Ready && &slot != selected)
                    {
                        slot.timing.readyDepthBeforePresent = static_cast<uint32_t>(readyDepth);
                        EnqueueDroppedTiming(slot.timing, FrameDropReason::ReadySuperseded);
                        ClearPreparedS2D32Slot(slot);
                        slot.state = PipelineSlotState::Free;
                    }
                }
                readyDepth = 1;
            }
            else
            {
                // With a one-frame smoothing target, emit the oldest ready
                // slot and retain the newer completion as the bounded buffer.
                selected = OldestReadyS2D32Slot();
            }

            bool readyToPresent = m_framePacer.ReadyToPresent(
                    nowQpc,
                    readyDepth,
                    !selected->finalReady,
                    true,
                    selected->sourcePresentQpc);
            selected->timing.targetDeadlineQpc = m_framePacer.DeadlineQpc();
            selected->timing.readyDepthBeforePresent = static_cast<uint32_t>(readyDepth);
            selected->timing.queueTarget = m_framePacer.TargetQueueDepth();

            // Prepare early only when a second completion remains behind this
            // one as the one-frame smoothing target.
            const uint32_t queueTarget = m_framePacer.TargetQueueDepth();
            const bool mayPrepareEarly =
                queueTarget != 0 && readyDepth > queueTarget;
            if (!selected->prepared)
            {
                if (!readyToPresent && !mayPrepareEarly) return false;
                PrepareOrPresentCompletedS2D32Slot(
                    *selected,
                    selected->timings,
                    false);
            }
            if (!selected->finalReady) return false;
            readyDepth = ReadyS2D32SlotCount();
            if (!readyToPresent)
            {
                readyToPresent = m_framePacer.ReadyToPresent(
                    FramePacer::QueryQpc(),
                    readyDepth,
                    false,
                    true,
                    selected->sourcePresentQpc);
                selected->timing.targetDeadlineQpc = m_framePacer.DeadlineQpc();
            }
            if (!readyToPresent) return false;
            
            const int64_t setupBeginQpc = FramePacer::QueryQpc();
            if (!PrepareOrPresentCompletedS2D32Slot(
                *selected,
                selected->timings,
                true,
                setupBeginQpc))
            {
                return false;
            }
            result = m_s2d32LastPresentedResult;
            return true;
        }

        D3D11RenderResult RenderS2D32Pipelined(
            const FrameInfo& frame,
            const FrameGeometry& geometry,
            uint64_t frameNumber,
            double acquireMs,
            uint64_t skippedFrames)
        {
            const int64_t submitCpuBeginQpc = FramePacer::QueryQpc();
            D3D11RenderResult completedResult{};

            if (UpdateSourceColorSpace(
                    frame.colorSpace,
                    frame.format,
                    frame.colorProvenance))
            {
                // Do not let an SDR/HDR transition pair frames that use
                
                ResetS2D32PipelineSlots(
                    FrameDropReason::UpscalerReset,
                    false);
            }

            S2D32PipelineSlot* freeSlot = FreeS2D32Slot();
            if (!freeSlot)
            {
                throw std::runtime_error("Neural submission reached a full event-driven pipeline.");
            }

            const D3D11_TEXTURE2D_DESC& sourceDesc = geometry.sourceDesc;
            const UINT cropLeft = geometry.cropLeft;
            const UINT cropTop = geometry.cropTop;
            const UINT cropWidth = geometry.cropWidth;
            const UINT cropHeight = geometry.cropHeight;
            const UINT presentationWidth = geometry.presentationWidth;
            const UINT presentationHeight = geometry.presentationHeight;

            UINT workingWidth = cropWidth;
            UINT workingHeight = cropHeight;
            UINT neuralWidth = 0;
            UINT neuralHeight = 0;
            if (!TryComputeS2D32OutputSize(workingWidth, workingHeight, neuralWidth, neuralHeight))
            {
                throw std::runtime_error("Neural input dimensions cannot produce a valid 2x output");
            }
            UINT swapWidth = 1;
            UINT swapHeight = 1;
            ComputeS2D32SwapSurfaceSize(
                presentationWidth,
                presentationHeight,
                neuralWidth,
                neuralHeight,
                swapWidth,
                swapHeight);
            EnsureSwapChain(swapWidth, swapHeight);

            D3D11RenderResult submission{};
            submission.workingWidth = workingWidth;
            submission.workingHeight = workingHeight;
            submission.videoProcessorOutputWidth = neuralWidth;
            submission.videoProcessorOutputHeight = neuralHeight;
            submission.internalOutputScale = kS2D32Scale;
            submission.upscalerMode = m_activeUpscalerMode;
            submission.selectedUpscalerMode = m_upscalerMode;
            submission.upscalerBypass = m_upscalerBypass;
            submission.shaderUpscaler = true;
            submission.skippedFrames = skippedFrames;
            submission.inputFormat = kModelTextureFormat;
            submission.inputColorSpace = frame.colorSpace;
            submission.inputColorProvenance = frame.colorProvenance;

            EnsureShaderWorkingFormat();
            const auto precision = NeuralPrecisionForMode(m_activeUpscalerMode);
            const auto modelKind = NeuralModelKindForMode(m_activeUpscalerMode);
            EnsureS2D32PipelineSlots(
                workingWidth,
                workingHeight,
                precision,
                modelKind);

            // Render preprocessing directly into the free CUDA slot.  The
            // slot retains its own views and lifetime, eliminating the old
            // full-frame working-input CopyResource from every neural frame.
            m_workingInput = freeSlot->input;
            m_workingInputRtv = freeSlot->inputRtv;
            m_workingInputSrv = freeSlot->inputSrv;
            m_workingInputUav = freeSlot->inputUav;
            m_workingWidth = workingWidth;
            m_workingHeight = workingHeight;
            m_workingInputFormat = kModelTextureFormat;
            UpdateSourceDimensions(sourceDesc.Width, sourceDesc.Height);
            UpdateToneConstants(swapWidth, swapHeight);
            submission.tone = m_toneConstants;
            D3D11PerfFrameSummary preprocessPerfSummary;
            if (m_profiler.Enabled())
            {
                preprocessPerfSummary.frameNumber = frameNumber;
                preprocessPerfSummary.sourceVersion = frame.version;
                preprocessPerfSummary.skippedFrames = skippedFrames;
                preprocessPerfSummary.cpuAcquireMs = acquireMs;
                SetPerfSummarySignature(
                    preprocessPerfSummary,
                    submission.upscalerMode,
                    submission.selectedUpscalerMode,
                    workingWidth,
                    workingHeight,
                    neuralWidth,
                    neuralHeight,
                    kModelTextureFormat);
                aess::presentation::CopyTelemetryText(
                    preprocessPerfSummary.totalStageName,
                    "d3d-preprocess-total");
                m_profiler.BeginFrame(frameNumber);
            }
            const int64_t preprocessBeginQpc = FramePacer::QueryQpc();
            BeginInputRead(frame.captureLease);
            PrepareWorkingInput(
                frame.d3d11Texture.Get(),
                frame.d3d11Srv.Get(),
                sourceDesc,
                cropLeft,
                cropTop,
                cropWidth,
                cropHeight,
                workingWidth,
                workingHeight);
            submission.preprocessMs = QpcElapsedMilliseconds(
                preprocessBeginQpc,
                FramePacer::QueryQpc());
            if (m_profiler.Enabled())
            {
                preprocessPerfSummary.cpuPreprocessMs = submission.preprocessMs;
                m_profiler.EndFrame(preprocessPerfSummary);
            }
            // Defer D3D11 Flush: ProcessAsync immediately enqueues the d3dFence
            // input-ready signal and flushes both fences in a single WDDM packet.
            SignalInputConsumed(frame.captureLease, false);

            const int64_t sourcePresentQpc = frame.sourcePresentQpc > 0
                ? frame.sourcePresentQpc
                : FramePacer::QueryQpc();
            const int64_t submitQpc = FramePacer::QueryQpc();
            freeSlot->result = submission;
            freeSlot->sourceVersion = frame.version;
            freeSlot->captureArrivalCount = frame.captureArrivalCount;
            freeSlot->frameNumber = frameNumber;
            freeSlot->sourcePresentOrdinal = frame.sourcePresentOrdinal;
            freeSlot->sourcePresentQpc = sourcePresentQpc;
            freeSlot->submitQpc = submitQpc;
            freeSlot->completionQpc = 0;
            freeSlot->acquireMs = acquireMs;
            freeSlot->outputMode = DebugViewEasuOnlyActive()
                ? aess::neural::RuntimeOutputMode::EasuOnly
                : aess::neural::RuntimeOutputMode::Neural;
            freeSlot->submitOrder = ++m_s2d32SubmitOrder;
            SubmitS2D32Inference(*freeSlot);
            m_contextDirty = false;
            const int64_t submitCpuEndQpc = FramePacer::QueryQpc();

            // Source-estimator and telemetry-record assembly are scheduler /
            // instrumentation work, not D3D/CUDA render submission. Keep them
            // outside cpu_render_ms, matching the V24 telemetry-only baseline;
            // total render-thread cost is independently gated by GetThreadTimes.
            ObserveSourceTiming(
                sourcePresentQpc,
                frame.sourcePresentOrdinal,
                true);
            freeSlot->timing = BeginFrameTiming(
                frame,
                frameNumber,
                freeSlot->submitOrder,
                submitQpc,
                workingWidth,
                workingHeight);
            freeSlot->cpuSubmitMs = QpcElapsedMilliseconds(
                submitCpuBeginQpc,
                submitCpuEndQpc);

            if (completedResult.workingWidth == 0 && m_s2d32HasPresented)
            {
                completedResult = m_s2d32LastPresentedResult;
            }
            return completedResult;
        }

        void EnsureFpsCounterOverlay()
        {
            if (m_fpsCounterContext || !m_device || !m_context)
            {
                return;
            }

            IMGUI_CHECKVERSION();
            m_fpsCounterContext = ImGui::CreateContext();
            ImGui::SetCurrentContext(m_fpsCounterContext);
            ImGuiIO& io = ImGui::GetIO();
            io.IniFilename = nullptr;
            io.LogFilename = nullptr;
            io.FontGlobalScale = 1.2f;
            ImGui::StyleColorsDark();
            ImGuiStyle& style = ImGui::GetStyle();
            style.WindowRounding = 5.0f;
            style.WindowPadding = ImVec2(12.0f, 9.0f);
            if (!ImGui_ImplDX11_Init(m_device.Get(), m_context.Get()))
            {
                ImGui::DestroyContext(m_fpsCounterContext);
                m_fpsCounterContext = nullptr;
                throw std::runtime_error("Failed to initialize FPS counter renderer.");
            }
        }

        void ShutdownFpsCounterOverlay() noexcept
        {
            if (!m_fpsCounterContext)
            {
                return;
            }
            ImGui::SetCurrentContext(m_fpsCounterContext);
            ImGui_ImplDX11_Shutdown();
            ImGui::DestroyContext(m_fpsCounterContext);
            m_fpsCounterContext = nullptr;
        }

        const char* FlipPresentationModeName() const noexcept
        {
            if (!m_havePresentationMode)
            {
                return "eligible; OS mode telemetry unavailable";
            }
            switch (m_presentationMode)
            {
            case DXGI_FRAME_PRESENTATION_MODE_NONE:
                return "independent flip";
            case DXGI_FRAME_PRESENTATION_MODE_OVERLAY:
                return "hardware overlay";
            case DXGI_FRAME_PRESENTATION_MODE_COMPOSED:
                return "DWM composed";
            default:
                return "composition failure";
            }
        }

        void RenderFpsCounterOverlay(UINT outputWidth, UINT outputHeight)
        {
            if (!m_fpsCounterEnabled || !m_swapChain)
            {
                return;
            }
            EnsureFpsCounterOverlay();
            ImGui::SetCurrentContext(m_fpsCounterContext);
            ImGuiIO& io = ImGui::GetIO();
            io.DisplaySize = ImVec2(static_cast<float>(outputWidth), static_cast<float>(outputHeight));
            io.DeltaTime = 1.0f / 120.0f;

            ImGui_ImplDX11_NewFrame();
            ImGui::NewFrame();
            ImGui::SetNextWindowPos(ImVec2(18.0f, 18.0f), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.78f);
            const ImGuiWindowFlags flags =
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoFocusOnAppearing |
                ImGuiWindowFlags_NoInputs |
                ImGuiWindowFlags_NoNav;
            ImGui::Begin("AESS FPS Counter", nullptr, flags);
            ImGui::TextUnformatted("AESS LIVE OUTPUT");
            ImGui::Separator();
            const ImVec4 healthy(0.35f, 1.0f, 0.48f, 1.0f);
            const ImVec4 warning(1.0f, 0.72f, 0.2f, 1.0f);
            const ImVec4 outputColor = m_outputFps + 0.5 < m_acceptedPresentFps ? warning : healthy;
            const auto pacing = m_framePacer.Snapshot();
            size_t completedDepth = ReadySynchronousFrameCount();
            for (const auto& slot : m_s2d32PipelineSlots)
            {
                completedDepth += slot.state == PipelineSlotState::Ready ? 1u : 0u;
            }
            const uint32_t completedQueueTarget =
                m_framePacer.TargetQueueDepth();
            ImGui::TextColored(outputColor, "DISPLAY  %7.1f FPS  %6.2f ms", m_outputFps, m_outputFrameMs);
            ImGui::Text("PRESENT  %7.1f FPS", m_acceptedPresentFps);
            ImGui::Text("INFER    %7.1f FPS", m_inferenceFps);
            ImGui::Text("SOURCE   %7.1f FPS", m_captureFps);
            if (IsNeuralUpscaler(m_activeUpscalerMode))
            {
                ImGui::Text(
                    DebugViewEasuOnlyActive()
                        ? "CUDA EASU %5.2f ms"
                        : "CUDA     %7.2f ms",
                    m_s2d32LastTimings.totalCudaMs);
                ImGui::Text("MODEL    %s%s", GetUpscalerName(m_activeUpscalerMode),
                    DebugViewEasuOnlyActive() ? " / EASU only" : "");
            }
            else
            {
                ImGui::TextUnformatted("CUDA          bypassed");
            }
            ImGui::Text(
                "PACING   p95 %5.2f ms  jitter %4.2f ms  residual p95 %4.2f ms",
                m_outputP95Ms,
                m_outputJitterMs,
                m_addedIntervalResidualP95Ms);
            ImGui::Text(
                "SOURCE   median %5.2f ms  jitter %4.2f ms",
                pacing.sourceMedianMs,
                pacing.sourceJitterMs);
            ImGui::Text(
                "EFFECT   service p95 %5.2f ms  cadence p95 %5.2f ms  prep lead %4.0f us",
                pacing.completionP95Ms,
                pacing.interCompletionP95Ms,
                pacing.postPreparationLeadUs);
            ImGui::Text(
                "QUEUE    %zu ready  target %u  %s",
                completedDepth,
                completedQueueTarget,
                pacing.buffering
                    ? "one-frame completed buffer"
                    : "zero-buffered");
            ImGui::Text(
                "LATENCY  wake p95 %4.2f ms  display age p95 %5.2f ms",
                pacing.completionWakeP95Ms,
                m_actualDisplayAgeP95Ms);
            ImGui::Text(
                "DEADLINE misses %llu",
                static_cast<unsigned long long>(pacing.deadlineMisses));
            ImGui::Text(
                "SYNC     %s  flip-meter wait %4.2f ms",
                m_gsyncCompatibilityEligible ? "VRR eligible" : "interval 0",
                m_lastFlipMeterWaitMs);
            ImGui::Text("FLIP     %s", FlipPresentationModeName());
            if (IsNeuralUpscaler(m_activeUpscalerMode))
            {
                size_t inFlight = 0;
                for (const auto& slot : m_s2d32PipelineSlots)
                {
                    inFlight += slot.state != PipelineSlotState::Free ? 1u : 0u;
                }
                ImGui::Text("PIPELINE %zu/%zu frame(s) in flight", inFlight, m_s2d32PipelineSlots.size());
            }
            const uint64_t readyDrops =
                m_frameDropCounts[static_cast<size_t>(FrameDropReason::ReadySuperseded)];
            const uint64_t staleDrops =
                m_frameDropCounts[static_cast<size_t>(FrameDropReason::StaleOverTwoPeriods)];
            const uint64_t presentDrops =
                m_frameDropCounts[static_cast<size_t>(FrameDropReason::PresentRejected)];
            const uint64_t dxgiUnobserved =
                m_frameDropCounts[static_cast<size_t>(FrameDropReason::DxgiSuperseded)];
            const uint64_t captureDrops =
                m_frameDropCounts[static_cast<size_t>(FrameDropReason::CaptureSuperseded)];
            const uint64_t saturatedDrops =
                m_frameDropCounts[static_cast<size_t>(FrameDropReason::PipelineSaturated)];
            const uint64_t timestampResets =
                m_frameDropCounts[static_cast<size_t>(FrameDropReason::TimestampRegression)];
            const uint64_t resizeResets =
                m_frameDropCounts[static_cast<size_t>(FrameDropReason::ResizeReset)];
            const uint64_t sourceTimingOverflows =
                m_frameDropCounts[
                    static_cast<size_t>(FrameDropReason::SourceTimingJournalOverflow)];
            ImGui::Text(
                "DROPS    ready %llu  stale %llu  present %llu  DXGI-unobserved %llu",
                static_cast<unsigned long long>(readyDrops),
                static_cast<unsigned long long>(staleDrops),
                static_cast<unsigned long long>(presentDrops),
                static_cast<unsigned long long>(dxgiUnobserved));
            ImGui::Text(
                "SOURCE   capture %llu  saturated %llu  timing-overflow %llu",
                static_cast<unsigned long long>(captureDrops),
                static_cast<unsigned long long>(saturatedDrops),
                static_cast<unsigned long long>(sourceTimingOverflows));
            ImGui::Text(
                "RESETS   timestamp %llu  resize %llu",
                static_cast<unsigned long long>(timestampResets),
                static_cast<unsigned long long>(resizeResets));
            if (m_captureBacklog > 0 || m_presentQueueBusyCount > 0 || m_flipMeterTimeoutCount > 0)
            {
                ImGui::TextColored(
                    warning,
                    "SRC GAP  %llu not inferred  busy %llu  meter timeout %llu",
                    static_cast<unsigned long long>(m_captureBacklog),
                    static_cast<unsigned long long>(m_presentQueueBusyCount),
                    static_cast<unsigned long long>(m_flipMeterTimeoutCount));
            }
            ImGui::TextDisabled(m_actualPresentStats ? "DXGI scan-out QPC flip meter" : "DXGI accepted-present fallback");
            ImGui::End();
            ImGui::Render();

            ComPtr<ID3D11Texture2D> backBuffer;
            ID3D11RenderTargetView* rtv = CurrentSwapBackBufferRtv(backBuffer);
            m_context->OMSetRenderTargets(1, &rtv, nullptr);
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        }

        void ResetAddedIntervalResidualSampling() noexcept
        {
            m_haveResidualCorrelation = false;
            m_lastResidualSourceQpc = 0;
            m_lastResidualSourceOrdinal = 0;
            m_lastResidualSyncQpc = 0;
            m_addedIntervalResidualCount = 0;
            m_addedIntervalResidualIndex = 0;
            m_addedIntervalResidualP95Ms = 0.0;
        }

        void ResetFpsSampling() noexcept
        {
            m_fpsSampleInitialized = false;
            m_haveMediaPresentCount = false;
            m_haveOutputObservation = false;
            m_haveOutputQpc = false;
            m_actualPresentStats = false;
            m_havePresentationMode = false;
            m_inferredFrameCount = 0;
            m_acceptedPresentCount = 0;
            m_outputPresentCount = 0;
            m_captureArrivalBase = 0;
            m_captureBacklog = 0;
            m_outputIntervalCount = 0;
            m_outputIntervalIndex = 0;
            ResetAddedIntervalResidualSampling();
            m_actualDisplayAgeCount = 0;
            m_actualDisplayAgeIndex = 0;
            m_actualDisplayAgeP95Ms = 0.0;
            m_captureFps = 0.0;
            m_inferenceFps = 0.0;
            m_acceptedPresentFps = 0.0;
            m_outputFps = 0.0;
            m_outputFrameMs = 0.0;
            m_outputP95Ms = 0.0;
            m_outputJitterMs = 0.0;
        }

        void RecordOutputInterval(double intervalMs)
        {
            if (intervalMs <= 0.0 || intervalMs > 1000.0)
            {
                return;
            }
            m_outputIntervals[m_outputIntervalIndex] = intervalMs;
            m_outputIntervalIndex = (m_outputIntervalIndex + 1) % m_outputIntervals.size();
            m_outputIntervalCount = std::min(m_outputIntervalCount + 1, m_outputIntervals.size());
        }

        void RecordAddedIntervalResidual(
            FrameTimingRecord& timing,
            int64_t actualSyncQpc) noexcept
        {
            if (timing.sourcePresentQpc <= 0 || actualSyncQpc <= 0)
            {
                return;
            }

            double residualMs = -1.0;
            if (m_haveResidualCorrelation &&
                ComputeAlignedIntervalResidualMs(
                    m_lastResidualSourceQpc,
                    m_lastResidualSourceOrdinal,
                    m_lastResidualSyncQpc,
                    timing.sourcePresentQpc,
                    timing.sourcePresentOrdinal,
                    actualSyncQpc,
                    m_framePacer.QpcFrequency(),
                    residualMs))
            {
                timing.addedIntervalResidualMs = residualMs;
                m_addedIntervalResiduals[m_addedIntervalResidualIndex] = residualMs;
                m_addedIntervalResidualIndex =
                    (m_addedIntervalResidualIndex + 1) %
                    m_addedIntervalResiduals.size();
                m_addedIntervalResidualCount = std::min(
                    m_addedIntervalResidualCount + 1,
                    m_addedIntervalResiduals.size());
            }

            // Any invalid pair is a discontinuity. Re-anchor on the current
            // exact DXGI/source correlation instead of manufacturing a sample.
            m_lastResidualSourceQpc = timing.sourcePresentQpc;
            m_lastResidualSourceOrdinal = timing.sourcePresentOrdinal;
            m_lastResidualSyncQpc = actualSyncQpc;
            m_haveResidualCorrelation = true;
        }

        struct PendingPresentTiming
        {
            FrameTimingRecord timing{};
            bool occupied = false;
        };

        static bool PresentCountPrecedes(uint32_t candidate, uint32_t observed) noexcept
        {
            // DXGI present counts are 32-bit monotonic counters. A modular
            // forward distance is unambiguous for this bounded (64-entry)
            // queue and remains correct across counter wrap.
            const uint32_t forwardDistance = observed - candidate;
            return forwardDistance != 0 && forwardDistance < 0x80000000u;
        }

        void FlushPendingPresentationTimings(FrameDropReason reason) noexcept
        {
            if (!m_telemetry.Enabled())
            {
                for (auto& pending : m_pendingPresentTimings)
                {
                    pending = {};
                }
                return;
            }

            for (auto& pending : m_pendingPresentTimings)
            {
                if (!pending.occupied) continue;
                pending.timing.flags |=
                    aess::presentation::FrameTimingPresentationStatsUnavailable;
                if (reason != FrameDropReason::None)
                {
                    EnqueueDroppedTiming(pending.timing, reason);
                }
                else
                {
                    m_telemetry.TryEnqueueFrame(pending.timing);
                }
                pending = {};
            }
        }

        void QueuePendingPresentationTiming(const FrameTimingRecord& source) noexcept
        {
            if (!m_telemetry.Enabled() && !m_fpsCounterEnabled) return;

            FrameTimingRecord timing = source;
            StampPacingTelemetry(timing);
            if (timing.dxgiPresentCount == 0)
            {
                timing.flags |=
                    aess::presentation::FrameTimingPresentationStatsUnavailable;
                if (m_telemetry.Enabled()) m_telemetry.TryEnqueueFrame(timing);
                return;
            }

            PendingPresentTiming* freeEntry = nullptr;
            PendingPresentTiming* oldestEntry = nullptr;
            for (auto& pending : m_pendingPresentTimings)
            {
                if (!pending.occupied)
                {
                    if (!freeEntry) freeEntry = &pending;
                    continue;
                }
                if (pending.timing.dxgiPresentCount == timing.dxgiPresentCount)
                {
                    // A duplicate DXGI count cannot be correlated
                    // independently. Preserve both records explicitly rather
                    // than silently replacing the older accepted present.
                    pending.timing.flags |=
                        aess::presentation::FrameTimingPresentationStatsUnavailable;
                    if (m_telemetry.Enabled())
                    {
                        EnqueueDroppedTiming(
                            pending.timing,
                            FrameDropReason::DxgiSuperseded);
                    }
                    pending = {};
                    freeEntry = &pending;
                    break;
                }
                if (!oldestEntry ||
                    pending.timing.recordId < oldestEntry->timing.recordId)
                {
                    oldestEntry = &pending;
                }
            }

            if (!freeEntry)
            {
                freeEntry = oldestEntry;
                if (freeEntry)
                {
                    freeEntry->timing.flags |=
                        aess::presentation::FrameTimingPresentationStatsUnavailable;
                    if (m_telemetry.Enabled())
                    {
                        m_telemetry.TryEnqueueFrame(freeEntry->timing);
                    }
                }
            }
            if (!freeEntry) return;
            freeEntry->timing = timing;
            freeEntry->occupied = true;
        }

        void ResolvePendingPresentationTimings(
            uint32_t observedPresentCount,
            int64_t actualSyncQpc,
            DXGI_FRAME_PRESENTATION_MODE presentationMode,
            bool havePresentationMode) noexcept
        {
            if (!m_telemetry.Enabled() && !m_fpsCounterEnabled) return;

            for (auto& pending : m_pendingPresentTimings)
            {
                if (!pending.occupied) continue;
                if (pending.timing.dxgiPresentCount == observedPresentCount)
                {
                    pending.timing.actualSyncQpc = actualSyncQpc;
                    pending.timing.presentationMode = havePresentationMode
                        ? static_cast<uint32_t>(presentationMode)
                        : 0u;
                    pending.timing.flags |=
                        aess::presentation::FrameTimingActualPresentObserved;
                    RecordAddedIntervalResidual(pending.timing, actualSyncQpc);
                    if (pending.timing.sourcePresentQpc > 0 &&
                        actualSyncQpc >= pending.timing.sourcePresentQpc)
                    {
                        RecordActualDisplayAge(
                            static_cast<double>(
                                actualSyncQpc - pending.timing.sourcePresentQpc) *
                            1000.0 /
                            static_cast<double>(m_framePacer.QpcFrequency()));
                    }
                    if (m_telemetry.Enabled())
                    {
                        m_telemetry.TryEnqueueFrame(pending.timing);
                    }
                    pending = {};
                    continue;
                }
                if (PresentCountPrecedes(
                    pending.timing.dxgiPresentCount,
                    observedPresentCount))
                {
                    // Latest-only frame statistics did not expose this ID
                    // before a newer one. This is operationally superseded;
                    // it is not proof that the frame never reached scanout.
                    pending.timing.flags |=
                        aess::presentation::FrameTimingPresentationStatsUnavailable;
                    if (m_telemetry.Enabled())
                    {
                        EnqueueDroppedTiming(
                            pending.timing,
                            FrameDropReason::DxgiSuperseded);
                    }
                    pending = {};
                }
            }
        }

        struct PresentationStatisticsSample
        {
            uint32_t presentCount = 0;
            int64_t syncQpc = 0;
            DXGI_FRAME_PRESENTATION_MODE mode = DXGI_FRAME_PRESENTATION_MODE_COMPOSED;
            bool valid = false;
            bool media = false;
            bool disjoint = false;
        };

        PresentationStatisticsSample QueryPresentationStatistics() noexcept
        {
            PresentationStatisticsSample sample;
            // GetFrameStatistics is the flip-model API whose PresentCount is
            // the Present ID returned by GetLastPresentCount.  The media API's
            // count is a monitor-presentation running total and is not a safe
            // key for the accepted-Present queue.  Query it only for the
            // composition mode (or as a last-resort timing fallback).
            DXGI_FRAME_STATISTICS_MEDIA mediaStats{};
            HRESULT mediaStatsHr = DXGI_ERROR_UNSUPPORTED;
            if (m_swapChainMedia)
            {
                mediaStatsHr = m_swapChainMedia->GetFrameStatisticsMedia(&mediaStats);
            }
            DXGI_FRAME_STATISTICS flipStats{};
            HRESULT flipStatsHr = DXGI_ERROR_UNSUPPORTED;
            if (m_swapChain)
            {
                flipStatsHr = m_swapChain->GetFrameStatistics(&flipStats);
            }
            sample.disjoint =
                mediaStatsHr == DXGI_ERROR_FRAME_STATISTICS_DISJOINT ||
                flipStatsHr == DXGI_ERROR_FRAME_STATISTICS_DISJOINT;
            if (SUCCEEDED(flipStatsHr))
            {
                sample.presentCount = flipStats.PresentCount;
                sample.syncQpc = flipStats.SyncQPCTime.QuadPart;
                if (SUCCEEDED(mediaStatsHr))
                {
                    sample.mode = mediaStats.CompositionMode;
                    sample.media = true;
                }
                sample.valid = true;
                return sample;
            }
            if (SUCCEEDED(mediaStatsHr))
            {
                sample.presentCount = mediaStats.PresentCount;
                sample.syncQpc = mediaStats.SyncQPCTime.QuadPart;
                sample.mode = mediaStats.CompositionMode;
                sample.valid = true;
                sample.media = true;
            }
            return sample;
        }

        void ResolveLatestPresentationStatisticsBeforeReset() noexcept
        {
            if (!m_telemetry.Enabled()) return;
            const PresentationStatisticsSample sample = QueryPresentationStatistics();
            if (!sample.valid) return;
            ResolvePendingPresentationTimings(
                sample.presentCount,
                sample.syncQpc,
                sample.mode,
                sample.media);
        }

        void RecordActualDisplayAge(double ageMs) noexcept
        {
            if (ageMs < 0.0 || ageMs > 10000.0) return;
            m_actualDisplayAges[m_actualDisplayAgeIndex] = ageMs;
            m_actualDisplayAgeIndex =
                (m_actualDisplayAgeIndex + 1) % m_actualDisplayAges.size();
            m_actualDisplayAgeCount = std::min(
                m_actualDisplayAgeCount + 1,
                m_actualDisplayAges.size());
        }

        void UpdateOutputPacingStatistics()
        {
            if (m_addedIntervalResidualCount != 0)
            {
                std::array<double, 120> residuals{};
                std::copy_n(
                    m_addedIntervalResiduals.begin(),
                    m_addedIntervalResidualCount,
                    residuals.begin());
                std::sort(
                    residuals.begin(),
                    residuals.begin() +
                        static_cast<std::ptrdiff_t>(m_addedIntervalResidualCount));
                const size_t residualP95Index = std::min(
                    m_addedIntervalResidualCount - 1,
                    static_cast<size_t>(std::ceil(
                        static_cast<double>(m_addedIntervalResidualCount - 1) *
                        0.95)));
                m_addedIntervalResidualP95Ms = residuals[residualP95Index];
            }
            else
            {
                m_addedIntervalResidualP95Ms = 0.0;
            }

            if (m_outputIntervalCount == 0)
            {
                m_outputP95Ms = 0.0;
                m_outputJitterMs = 0.0;
                return;
            }
            std::array<double, 120> values{};
            std::copy_n(
                m_outputIntervals.begin(),
                m_outputIntervalCount,
                values.begin());
            double sum = 0.0;
            for (size_t index = 0; index < m_outputIntervalCount; ++index)
            {
                sum += values[index];
            }
            const double mean = sum / static_cast<double>(m_outputIntervalCount);
            double variance = 0.0;
            for (size_t index = 0; index < m_outputIntervalCount; ++index)
            {
                const double delta = values[index] - mean;
                variance += delta * delta;
            }
            m_outputJitterMs = std::sqrt(
                variance / static_cast<double>(m_outputIntervalCount));
            std::sort(
                values.begin(),
                values.begin() + static_cast<std::ptrdiff_t>(m_outputIntervalCount));
            const size_t p95Index = std::min(
                m_outputIntervalCount - 1,
                static_cast<size_t>(std::floor(
                    static_cast<double>(m_outputIntervalCount - 1) * 0.95)));
            m_outputP95Ms = values[p95Index];

            if (m_actualDisplayAgeCount != 0)
            {
                std::array<double, 120> ages{};
                std::copy_n(
                    m_actualDisplayAges.begin(),
                    m_actualDisplayAgeCount,
                    ages.begin());
                std::sort(
                    ages.begin(),
                    ages.begin() + static_cast<std::ptrdiff_t>(m_actualDisplayAgeCount));
                const size_t ageP95Index = std::min(
                    m_actualDisplayAgeCount - 1,
                    static_cast<size_t>(std::floor(
                        static_cast<double>(m_actualDisplayAgeCount - 1) * 0.95)));
                m_actualDisplayAgeP95Ms = ages[ageP95Index];
            }
        }

        void UpdateFpsSampling(
            uint64_t captureArrivalCount,
            bool presentAccepted,
            FrameTimingRecord* timing = nullptr)
        {
            if (timing)
            {
                if (presentAccepted)
                {
                    QueuePendingPresentationTiming(*timing);
                }
                else if (m_telemetry.Enabled())
                {
                    if (timing->dropReason != FrameDropReason::None)
                    {
                        EnqueueDroppedTiming(*timing, timing->dropReason);
                    }
                    else
                    {
                        StampPacingTelemetry(*timing);
                        m_telemetry.TryEnqueueFrame(*timing);
                    }
                }
            }
            if (!m_fpsCounterEnabled && !m_telemetry.Enabled())
            {
                return;
            }

            using Clock = std::chrono::steady_clock;
            const auto now = Clock::now();
            uint64_t displayedDelta = 0;
            const PresentationStatisticsSample statistics =
                QueryPresentationStatistics();
            const bool haveActualPresentCount = statistics.valid;
            const bool haveMediaStats = statistics.media;
            const uint32_t actualPresentCount = statistics.presentCount;
            const int64_t actualSyncQpc = statistics.syncQpc;
            if (haveMediaStats)
            {
                m_presentationMode = statistics.mode;
                m_havePresentationMode = true;
            }

            if (statistics.disjoint && !haveActualPresentCount)
            {
                m_haveMediaPresentCount = false;
                m_haveOutputQpc = false;
                m_haveOutputObservation = false;
                ResetAddedIntervalResidualSampling();
                if (m_haveSuccessfulFrameStatistics)
                {
                    FlushPendingPresentationTimings(
                        FrameDropReason::PresentationReset);
                }
                // The first disjoint after swapchain creation is expected.
                // Retain pending IDs until a successful sample establishes a
                // baseline; a later disjoint explicitly resets correlation.
                m_haveSuccessfulFrameStatistics = false;
            }

            if (haveActualPresentCount)
            {
                m_haveSuccessfulFrameStatistics = true;
                ResolvePendingPresentationTimings(
                    actualPresentCount,
                    actualSyncQpc,
                    statistics.mode,
                    haveMediaStats);
            }

            if (!m_fpsCounterEnabled)
            {
                return;
            }

            if (presentAccepted)
            {
                ++m_acceptedPresentCount;
            }
            if (m_captureArrivalBase == 0 && captureArrivalCount > 0)
            {
                m_captureArrivalBase = captureArrivalCount - 1;
            }
            const uint64_t logicalArrivals = captureArrivalCount >= m_captureArrivalBase
                ? captureArrivalCount - m_captureArrivalBase
                : m_inferredFrameCount;
            m_captureBacklog = logicalArrivals > m_inferredFrameCount
                ? logicalArrivals - m_inferredFrameCount
                : 0;

            if (haveActualPresentCount)
            {
                m_actualPresentStats = true;
                if (m_haveMediaPresentCount &&
                    actualPresentCount != m_lastMediaPresentCount)
                {
                    // Frame-statistics PresentCount is the ID of the last
                    // Present that reached display, not a count of every ID
                    // in between. A jump therefore represents one newly
                    // observed scanout and may skip superseded Present IDs.
                    displayedDelta = 1;
                    ++m_outputPresentCount;
                }
                m_lastMediaPresentCount = actualPresentCount;
                m_haveMediaPresentCount = true;
            }
            else if (presentAccepted)
            {
                displayedDelta = 1;
                ++m_outputPresentCount;
            }

            if (displayedDelta > 0)
            {
                LARGE_INTEGER qpcFrequency{};
                const bool haveQpcFrequency = QueryPerformanceFrequency(&qpcFrequency) != FALSE &&
                    qpcFrequency.QuadPart > 0;
                if (haveQpcFrequency && actualSyncQpc > 0)
                {
                    if (m_haveOutputQpc && actualSyncQpc > m_lastOutputQpc)
                    {
                        const double elapsedMs =
                            static_cast<double>(actualSyncQpc - m_lastOutputQpc) * 1000.0 /
                            static_cast<double>(qpcFrequency.QuadPart);
                        RecordOutputInterval(elapsedMs / static_cast<double>(displayedDelta));
                    }
                    m_lastOutputQpc = actualSyncQpc;
                    m_haveOutputQpc = true;
                }
                else if (m_haveOutputObservation)
                {
                    const double elapsedMs = std::chrono::duration<double, std::milli>(
                        now - m_lastOutputObservation).count();
                    RecordOutputInterval(elapsedMs / static_cast<double>(displayedDelta));
                }
                m_lastOutputObservation = now;
                m_haveOutputObservation = true;
            }

            if (!m_fpsSampleInitialized)
            {
                m_fpsSampleInitialized = true;
                m_fpsSampleTime = now;
                m_sampleCaptureCount = logicalArrivals;
                m_sampleInferenceCount = m_inferredFrameCount;
                m_sampleAcceptedPresentCount = m_acceptedPresentCount;
                m_sampleOutputCount = m_outputPresentCount;
                return;
            }

            const double seconds = std::chrono::duration<double>(now - m_fpsSampleTime).count();
            if (seconds < 0.5)
            {
                return;
            }
            m_captureFps = static_cast<double>(logicalArrivals - m_sampleCaptureCount) / seconds;
            m_inferenceFps = static_cast<double>(m_inferredFrameCount - m_sampleInferenceCount) / seconds;
            m_acceptedPresentFps = static_cast<double>(
                m_acceptedPresentCount - m_sampleAcceptedPresentCount) / seconds;
            m_outputFps = static_cast<double>(m_outputPresentCount - m_sampleOutputCount) / seconds;
            m_outputFrameMs = m_outputFps > 0.0 ? 1000.0 / m_outputFps : 0.0;
            UpdateOutputPacingStatistics();
            m_fpsSampleTime = now;
            m_sampleCaptureCount = logicalArrivals;
            m_sampleInferenceCount = m_inferredFrameCount;
            m_sampleAcceptedPresentCount = m_acceptedPresentCount;
            m_sampleOutputCount = m_outputPresentCount;
        }

        bool UsesVideoProcessor() const
        {
            return false;
        }

        bool UsesShaderUpscaler() const
        {
            return !UsesVideoProcessor();
        }

        bool UsesShaderBridge() const { return true; }


        bool UsesPresentationBridge() const
        {
            return UsesShaderBridge() || UsesShaderUpscaler();
        }

        static uint32_t EffectiveShaderInputHeight(uint32_t requestedHeight, uint32_t sourceHeight, uint32_t outputHeight)
        {
            const uint32_t limit = std::min(sourceHeight, outputHeight);
            if (requestedHeight == 0 || requestedHeight >= limit)
            {
                return 0;
            }
            return requestedHeight;
        }

        DXGI_FORMAT SwapChainFormatForMode() const { return kSdrSwapChainFormat; }


        DXGI_COLOR_SPACE_TYPE ColorSpaceForSwapFormat(DXGI_FORMAT) const
        { return DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709; }


        DXGI_COLOR_SPACE_TYPE VideoProcessorOutputColorSpace() const
        { return DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709; }


        ID3D11PixelShader* PresentPixelShaderForMode(bool direct) const
        { return direct ? m_presentDirectSdrPixelShader.Get() : m_presentSdrPixelShader.Get(); }


        ID3D11ComputeShader* PresentComputeShaderForDirect() const { return m_csPresentDirectSdr.Get(); }


        static UINT ConstantBufferByteWidth(size_t byteSize)
        {
            return static_cast<UINT>((byteSize + 15u) & ~static_cast<size_t>(15u));
        }


        void ReportHardwareCompositionSupport(IDXGIAdapter* adapter) const
        {
            const HMONITOR targetMonitor = MonitorFromWindow(
                m_hwnd,
                MONITOR_DEFAULTTONEAREST);
            for (UINT outputIndex = 0;; ++outputIndex)
            {
                ComPtr<IDXGIOutput> output;
                const HRESULT enumHr = adapter->EnumOutputs(outputIndex, &output);
                if (enumHr == DXGI_ERROR_NOT_FOUND) break;
                if (FAILED(enumHr)) break;

                DXGI_OUTPUT_DESC outputDesc{};
                if (FAILED(output->GetDesc(&outputDesc)) ||
                    outputDesc.Monitor != targetMonitor)
                {
                    continue;
                }

                ComPtr<IDXGIOutput6> output6;
                UINT supportFlags = 0;
                const HRESULT supportHr = output.As(&output6);
                if (SUCCEEDED(supportHr))
                {
                    const HRESULT checkHr =
                        output6->CheckHardwareCompositionSupport(&supportFlags);
                    if (FAILED(checkHr))
                    {
                        std::wcout << L"Hardware composition capability query failed, HRESULT=0x"
                            << std::hex << static_cast<unsigned long>(checkHr)
                            << std::dec << L"\n";
                        return;
                    }
                }
                else
                {
                    std::wcout << L"Hardware composition capability query unavailable\n";
                    return;
                }

                std::wcout << L"Hardware composition: output=" << outputDesc.DeviceName
                    << L" fullscreen="
                    << ((supportFlags & DXGI_HARDWARE_COMPOSITION_SUPPORT_FLAG_FULLSCREEN) ? L"yes" : L"no")
                    << L" windowed="
                    << ((supportFlags & DXGI_HARDWARE_COMPOSITION_SUPPORT_FLAG_WINDOWED) ? L"yes" : L"no")
                    << L" cursor_stretched="
                    << ((supportFlags & DXGI_HARDWARE_COMPOSITION_SUPPORT_FLAG_CURSOR_STRETCHED) ? L"yes" : L"no")
                    << L"\n";
                return;
            }
            std::wcout << L"Hardware composition capability query could not match the output monitor\n";
        }

        void EnsureDevice(ID3D11Texture2D* source)
        {
            ComPtr<ID3D11Device> sourceDevice;
            source->GetDevice(&sourceDevice);
            EnsureDevice(sourceDevice.Get());
        }

        void EnsureDevice(ID3D11Device* sourceDevice)
        {
            if (m_device.Get() == sourceDevice)
            {
                return;
            }

            ResetDeviceObjects();
            m_device = sourceDevice;
            m_device->GetImmediateContext(&m_context);
            ThrowIfFailed(m_device.As(&m_device5), "Query ID3D11Device5 for frame completion");
            ThrowIfFailed(m_context.As(&m_context4), "Query ID3D11DeviceContext4 for frame completion");
            ThrowIfFailed(
                m_device5->CreateFence(
                    0,
                    D3D11_FENCE_FLAG_NONE,
                    IID_PPV_ARGS(&m_completionFence)),
                "Create D3D11 frame completion fence");
            m_completionFenceValue = 0;
            ThrowIfFailed(
                m_device5->CreateFence(
                    0,
                    D3D11_FENCE_FLAG_NONE,
                    IID_PPV_ARGS(&m_inputConsumedFence)),
                "Create D3D11 input consumed fence");
            m_inputConsumedFenceValue = 0;
            m_inputConsumedEvent.attach(CreateEventW(nullptr, TRUE, FALSE, nullptr));
            if (!m_inputConsumedEvent)
            {
                ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()), "Create D3D11 input consumed event");
            }
            D3D11_QUERY_DESC inputDrainDescription{ D3D11_QUERY_EVENT, 0 };
            ThrowIfFailed(m_device->CreateQuery(&inputDrainDescription, &m_inputDrainQuery),
                "Create WGC input shutdown drain query");
            m_profiler.SetDevice(m_device.Get(), m_context.Get());
            ThrowIfFailed(m_device.As(&m_videoDevice), "Query ID3D11VideoDevice");
            ThrowIfFailed(m_context.As(&m_videoContext), "Query ID3D11VideoContext");
            m_context.As(&m_videoContext1);

            ComPtr<IDXGIDevice> dxgiDevice;
            ThrowIfFailed(m_device.As(&dxgiDevice), "Query IDXGIDevice for presenter");
            // and final composition share the latency-critical priority. The
            // CUDA-only upscaler path retains neutral D3D presentation work.
            const int gpuPriority = false ? 7 : 0;
            const HRESULT gpuPriorityHr =
                dxgiDevice->SetGPUThreadPriority(gpuPriority);
            if (SUCCEEDED(gpuPriorityHr))
            {
                std::wcout
                    << L"D3D11 presentation scheduling priority: "
                    << (gpuPriority != 0 ? L"latency-critical (7)" : L"neutral")
                    << L"\n";
            }
            else
            {
                std::wcout << L"D3D11 GPU scheduling priority request failed, HRESULT=0x"
                    << std::hex << static_cast<unsigned long>(gpuPriorityHr) << std::dec << L"\n";
            }
            ComPtr<IDXGIAdapter> adapter;
            ThrowIfFailed(dxgiDevice->GetAdapter(&adapter), "IDXGIDevice::GetAdapter for presenter");
            ThrowIfFailed(adapter->GetParent(IID_PPV_ARGS(&m_factory)), "IDXGIAdapter::GetParent");

            m_tearingSupported = false;
            ComPtr<IDXGIFactory5> factory5;
            if (SUCCEEDED(m_factory.As(&factory5)))
            {
                BOOL allowTearing = FALSE;
                if (SUCCEEDED(factory5->CheckFeatureSupport(
                    DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                    &allowTearing,
                    sizeof(allowTearing))))
                {
                    m_tearingSupported = allowTearing == TRUE;
                }
            }
            std::wcout << L"Presentation API: flip-discard, interval=0; tearing/VRR opt-in="
                << (m_tearingSupported ? L"enabled" : L"disabled") << L"\n";
            ReportHardwareCompositionSupport(adapter.Get());

            CreatePreprocessPipeline();
        }

        void ResetDeviceObjects()
        {
            
            ShutdownFpsCounterOverlay();
            ResolveLatestPresentationStatisticsBeforeReset();
            FlushPendingPresentationTimings(FrameDropReason::PresentationReset);
            ResetSynchronousReadyFrame(
                FrameDropReason::PresentationReset,
                true);
            m_synchronousRouteSignature = {};
            m_synchronousRouteSignatureValid = false;
            m_profiler.SetDevice(nullptr, nullptr);
            ResetS2D32PipelineSlots();
            DrainCaptureReads();
            m_framePacer.ResetTimeline();
            m_s2d32Runtime.reset();
            m_s2d32StartupReported = false;
            m_videoProcessor.Reset();
            m_vpEnum.Reset();
            m_cachedVpInputView.Reset();
            m_cachedVpOutputView.Reset();
            m_cachedVpOutputTexture.Reset();
            m_workingInput.Reset();
            m_workingInputRtv.Reset();
            m_workingInputSrv.Reset();
            m_workingInputUav.Reset();
            m_videoProcessorOutput.Reset();
            m_videoProcessorOutputRtv.Reset();
            m_videoProcessorOutputSrv.Reset();
            m_videoProcessorOutputUav.Reset();
            m_upscaleScratch.Reset();
            m_upscaleScratchRtv.Reset();
            m_upscaleScratchSrv.Reset();
            m_upscaleScratchUav.Reset();
            
            
            
            
            
            
            m_presentResample.Reset();
            m_presentResampleRtv.Reset();
            m_presentResampleSrv.Reset();
            m_shaderSource.Reset();
            m_shaderSourceSrv.Reset();
            m_sourceViewCache.Reset();
            m_vertexShader.Reset();
            m_pixelShader.Reset();
            m_workingInputMagicPixelShader.Reset();
            m_magicSharpEncodedPixelShader.Reset();
            m_copySameSizePixelShader.Reset();
            m_bilinearResamplePixelShader.Reset();
            m_magicResamplePixelShader.Reset();
            m_presentSdrPixelShader.Reset();
            m_presentDirectSdrPixelShader.Reset();
            m_csPresentDirectSdr.Reset();
            m_csPrepareWorkingInput.Reset();
            m_uploadStaging.Reset();
            m_uploadStagingWidth = 0;
            m_uploadStagingHeight = 0;
            m_uploadStagingFormat = DXGI_FORMAT_UNKNOWN;
            m_pointSampler.Reset();
            m_linearSampler.Reset();
            m_toneBuffer.Reset();
            m_toneBufferCache.Reset();
            m_toneConstantsDirty = false;
            for (auto& backBuffer : m_swapBackBuffers)
            {
                backBuffer.Reset();
            }
            for (auto& rtv : m_swapBackBufferRtvs)
            {
                rtv.Reset();
            }
            for (auto& uav : m_swapBackBufferUavs)
            {
                uav.Reset();
            }
            ThrowIfFailed(
                ReleaseSwapChainOnOwningThread(),
                "Release swapchain on HWND owner for device reset");
            m_compositionScaleTransform.Reset();
            m_compositionVisual.Reset();
            m_compositionTarget.Reset();
            m_compositionDevice.Reset();
            m_factory.Reset();
            m_videoContext1.Reset();
            m_videoContext.Reset();
            m_videoDevice.Reset();
            m_completionFence.Reset();
            m_inputConsumedFence.Reset();
            m_inputConsumedEvent.close();
            m_inputConsumedEventValue = 0;
            m_inputDrainQuery.Reset();
            m_context4.Reset();
            m_device5.Reset();
            m_completionFenceValue = 0;
            m_inputConsumedFenceValue = 0;
            m_context.Reset();
            m_device.Reset();
            m_tearingSupported = false;
            m_presentQueueBusyCount = 0;
            m_lastFlipMeterWaitMs = 0.0;
            m_flipMeterTimeoutCount = 0;
            m_flipMeterWaitTotalMs = 0.0;
            m_flipMeterWaitCount = 0;
            m_s2d32BlockingWaitTotalMs = 0.0;
            m_s2d32BlockingWaitCount = 0;
            m_sourceWidth = 0;
            m_sourceHeight = 0;
            m_sourceFormat = DXGI_FORMAT_UNKNOWN;
            m_workingWidth = 0;
            m_workingHeight = 0;
            m_workingInputFormat = DXGI_FORMAT_UNKNOWN;
            m_swapWidth = 0;
            m_swapHeight = 0;
            m_vpInputWidth = 0;
            m_vpInputHeight = 0;
            m_vpOutputWidth = 0;
            m_vpOutputHeight = 0;
            m_workingFormat = DXGI_FORMAT_UNKNOWN;
            m_vpWorkingFormat = DXGI_FORMAT_UNKNOWN;
            m_videoProcessorOutputFormat = DXGI_FORMAT_UNKNOWN;
            m_videoProcessorOutputWidth = 0;
            m_videoProcessorOutputHeight = 0;
            m_presentResampleWidth = 0;
            m_presentResampleHeight = 0;
            m_presentResampleFormat = DXGI_FORMAT_UNKNOWN;
            m_upscaleScratchWidth = 0;
            m_upscaleScratchHeight = 0;
            m_upscaleScratchFormat = DXGI_FORMAT_UNKNOWN;
            m_outputSeq = 0;
        }

        void CreateVertexShaderFromBytecode(
            const aess::generated::d3d11::ShaderBytecode& bytecode,
            ID3D11VertexShader** shader,
            const char* operation)
        {
            ThrowIfFailed(
                m_device->CreateVertexShader(
                    bytecode.data,
                    bytecode.size,
                    nullptr,
                    shader),
                operation);
        }

        void CreatePixelShaderFromBytecode(
            const aess::generated::d3d11::ShaderBytecode& bytecode,
            ID3D11PixelShader** shader,
            const char* operation)
        {
            ThrowIfFailed(
                m_device->CreatePixelShader(
                    bytecode.data,
                    bytecode.size,
                    nullptr,
                    shader),
                operation);
        }

        void CreateComputeShaderFromBytecode(
            const aess::generated::d3d11::ShaderBytecode& bytecode,
            ID3D11ComputeShader** shader,
            const char* operation)
        {
            ThrowIfFailed(
                m_device->CreateComputeShader(
                    bytecode.data,
                    bytecode.size,
                    nullptr,
                    shader),
                operation);
        }

        ComPtr<ID3DBlob> CompileShaderSource(
            const std::string& source,
            const char* shaderName,
            const char* entryPoint,
            const char* target)
        {
            UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
            flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
            flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif
            ComPtr<ID3DBlob> shader;
            ComPtr<ID3DBlob> error;
            HRESULT hr = D3DCompile(
                source.data(),
                source.size(),
                shaderName,
                nullptr,
                nullptr,
                entryPoint,
                target,
                flags,
                0,
                &shader,
                &error);
            if (FAILED(hr))
            {
                std::string message = error ? static_cast<const char*>(error->GetBufferPointer()) : "D3DCompile failed";
                throw std::runtime_error(message);
            }
            return shader;
        }

        void CreatePreprocessPipeline()
        {
            m_toneConstants.sourceTransfer = kModelTransferSdrGamma;
            m_toneConstants.sourcePrimaries = kModelPrimariesRec709;
            m_toneConstants.sdrWhiteScale = 1.0f;
            m_toneConstants.sdrWhiteNits = 80.0f;
            m_toneConstants.gamma = 2.2f;
            m_toneConstants.saturation = 1.0f;
            m_toneConstants.contrast = 1.0f;
            m_toneConstants.blackPoint = 0.0f;
            m_toneConstants.highlightCompression = 0.0f;

            using namespace aess::generated::d3d11;

            CreateVertexShaderFromBytecode(
                kVsMain,
                &m_vertexShader,
                "Create preprocess vertex shader");
            CreatePixelShaderFromBytecode(
                kPsMain,
                &m_pixelShader,
                "Create preprocess pixel shader");
            CreatePixelShaderFromBytecode(
                kPsCopySameSizeMain,
                &m_copySameSizePixelShader,
                "Create D3D11 neutral same-size copy pixel shader");
            CreatePixelShaderFromBytecode(
                kPsBilinearResampleMain,
                &m_bilinearResamplePixelShader,
                "Create D3D11 neutral bilinear resize pixel shader");

            CreatePixelShaderFromBytecode(
                kPsWorkingInputMagicMain,
                &m_workingInputMagicPixelShader,
                "Create D3D11 working-input Magic Kernel pixel shader");
            CreateComputeShaderFromBytecode(
                kCsPrepareWorkingInput,
                &m_csPrepareWorkingInput,
                "Create D3D11 working-input compute shader");
            CreatePixelShaderFromBytecode(
                kPsMagicSharpEncodedMain,
                &m_magicSharpEncodedPixelShader,
                "Create D3D11 Magic Kernel Sharp pixel shader");
            CreatePixelShaderFromBytecode(
                kPsMagicResampleMain,
                &m_magicResamplePixelShader,
                "Create D3D11 Magic Kernel resample pixel shader");

            // The final encoding is also fixed for a presenter lifetime. Both
            // direct and resampled variants are retained because monitor-fit
            // can change when the source window is resized.
            CreatePixelShaderFromBytecode(kPsPresentSdrMain, &m_presentSdrPixelShader, "Create SDR present shader");
            CreatePixelShaderFromBytecode(kPsPresentDirectSdrMain, &m_presentDirectSdrPixelShader, "Create direct SDR present shader");
            CreateComputeShaderFromBytecode(kCsPresentDirectSdrMain, &m_csPresentDirectSdr, "Create direct SDR compute shader");

            D3D11_SAMPLER_DESC samplerDesc{};
            samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
            samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
            samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
            ThrowIfFailed(m_device->CreateSamplerState(&samplerDesc, &m_linearSampler), "Create preprocess sampler");

            D3D11_SAMPLER_DESC pointSamplerDesc = samplerDesc;
            pointSamplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
            ThrowIfFailed(m_device->CreateSamplerState(&pointSamplerDesc, &m_pointSampler), "Create D3D11 point sampler");

            m_toneConstantsDirty = true;
            UploadToneConstantsIfDirty();


        }

        static bool ToneConstantsEqual(
            const ToneConstants& left,
            const ToneConstants& right) noexcept
        {
            static_assert(sizeof(ToneConstants) == 80);
            return std::memcmp(&left, &right, sizeof(left)) == 0;
        }

        void ApplyToneConstantsSnapshot(const ToneConstants& constants) noexcept
        {
            if (!ToneConstantsEqual(m_toneConstants, constants))
            {
                m_toneConstants = constants;
                m_toneConstantsDirty = true;
            }
        }

        void UploadToneConstantsIfDirty()
        {
            if (!m_toneConstantsDirty)
            {
                return;
            }
            // SDR constants and crop values are unchanged.
            // Repeated passes bind their existing exact constant snapshot
            // instead of rewriting a shared buffer several times per frame.
            ThrowIfFailed(
                m_toneBufferCache.Resolve(
                    m_device.Get(), m_context.Get(), m_toneConstants, m_toneBuffer),
                "Resolve tone constant buffer");
            m_toneConstantsDirty = false;
        }

        void UpdateToneConstants(UINT outputWidth, UINT outputHeight)
        {
            float newOutputWidth = static_cast<float>(std::max<UINT>(1, outputWidth));
            float newOutputHeight = static_cast<float>(std::max<UINT>(1, outputHeight));
            if (m_toneConstants.outputWidth != newOutputWidth ||
                m_toneConstants.outputHeight != newOutputHeight)
            {
                m_toneConstants.outputWidth = newOutputWidth;
                m_toneConstants.outputHeight = newOutputHeight;
                m_toneConstantsDirty = true;
            }
        }

        void UpdateSourceDimensions(UINT width, UINT height)
        {
            const float newWidth = static_cast<float>(std::max<UINT>(1, width));
            const float newHeight = static_cast<float>(std::max<UINT>(1, height));
            if (m_toneConstants.sourceWidth != newWidth ||
                m_toneConstants.sourceHeight != newHeight)
            {
                m_toneConstants.sourceWidth = newWidth;
                m_toneConstants.sourceHeight = newHeight;
                m_toneConstantsDirty = true;
            }
        }

        void BindToneConstants()
        {
            // A resample pass can switch the cached buffer before the final
            // presentation pass. Resolve the pointer at each bind, never keep
            // a pointer array captured before that pass changed the crop.
            ID3D11Buffer* buffer = m_toneBuffer.Get();
            m_context->PSSetConstantBuffers(0, 1, &buffer);
            m_context->CSSetConstantBuffers(0, 1, &buffer);
        }

        bool CreateSwapChainForFormat(UINT width, UINT height, DXGI_FORMAT format)
        {
            if (GetCurrentThreadId() != m_windowThreadId)
            {
                throw std::runtime_error(
                    "CreateSwapChainForHwnd attempted outside the output HWND owner thread.");
            }
            DXGI_SWAP_CHAIN_DESC1 desc{};
            desc.Width = width;
            desc.Height = height;
            desc.Format = format;
            desc.SampleDesc.Count = 1;
            desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_UNORDERED_ACCESS;
            desc.BufferCount = kFrameCount;
            desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
            desc.Scaling = DXGI_SCALING_STRETCH;
            desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
            desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT |
                (m_tearingSupported ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);

            ComPtr<IDXGISwapChain1> swapChain;
            HRESULT hr = m_factory->CreateSwapChainForHwnd(
                m_device.Get(),
                m_hwnd,
                &desc,
                nullptr,
                nullptr,
                &swapChain);
            if (FAILED(hr))
            {
                // Fallback to render-target-only if display driver rejects swapchain UAV
                desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
                hr = m_factory->CreateSwapChainForHwnd(
                    m_device.Get(),
                    m_hwnd,
                    &desc,
                    nullptr,
                    nullptr,
                    &swapChain);
                if (FAILED(hr))
                {
                    return false;
                }
            }

            if (FAILED(swapChain.As(&m_swapChain)))
            {
                return false;
            }
            ThrowIfFailed(
                m_swapChain->SetMaximumFrameLatency(1),
                "Set borderless output maximum frame latency");
            m_frameLatencyWaitable = m_swapChain->GetFrameLatencyWaitableObject();
            if (!m_frameLatencyWaitable)
            {
                throw std::runtime_error("Borderless output frame-latency waitable object is unavailable.");
            }
            m_waitingForPresentationSlot = false;
            m_frameLatencyWakeGranted = false;
            swapChain.As(&m_swapChainMedia);
            m_haveSuccessfulFrameStatistics = false;
            ResetFpsSampling();
            m_factory->MakeWindowAssociation(m_hwnd, DXGI_MWA_NO_ALT_ENTER);
            m_swapFormat = format;
            m_swapWidth = width;
            m_swapHeight = height;
            ConfigureSwapChainColorSpace();
            return true;
        }

        void ConfigureSwapChainColorSpace()
        {
            if (!m_swapChain)
            {
                return;
            }

            m_outputColorSpace = ColorSpaceForSwapFormat(m_swapFormat);

            UINT support = 0;
            ThrowIfFailed(
                m_swapChain->CheckColorSpaceSupport(
                    m_outputColorSpace,
                    &support),
                "Check output swap-chain color-space support");
            if ((support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT) == 0)
            {
                throw std::runtime_error(
                    "The selected presentation color space is not supported "
                    "for this output swap chain.");
            }
            ThrowIfFailed(
                m_swapChain->SetColorSpace1(m_outputColorSpace),
                "Set output swap-chain color space");
            std::cout
                << "Presentation color space applied: "
                << static_cast<uint32_t>(m_outputColorSpace)
                << " format=" << FormatName(m_swapFormat)
                << '\n';
        }

        void ResetSwapChainDependentResources() noexcept
        {
            m_videoProcessor.Reset();
            m_vpEnum.Reset();
            m_cachedVpInputView.Reset();
            m_cachedVpOutputView.Reset();
            m_cachedVpOutputTexture.Reset();
            for (auto& backBuffer : m_swapBackBuffers)
            {
                backBuffer.Reset();
            }
            for (auto& rtv : m_swapBackBufferRtvs)
            {
                rtv.Reset();
            }
            for (auto& uav : m_swapBackBufferUavs)
            {
                uav.Reset();
            }
        }

        void ReleaseSwapChainOnWindowThread()
        {
            if (GetCurrentThreadId() != m_windowThreadId)
            {
                throw std::runtime_error(
                    "Swapchain release attempted outside the output HWND owner thread.");
            }

            if (m_frameLatencyWaitable)
            {
                CloseHandle(m_frameLatencyWaitable);
                m_frameLatencyWaitable = nullptr;
            }
            m_waitingForPresentationSlot = false;
            m_frameLatencyWakeGranted = false;
            m_swapChainMedia.Reset();
            m_swapChain.Reset();
            m_swapWidth = 0;
            m_swapHeight = 0;
            m_swapFormat = DXGI_FORMAT_UNKNOWN;
        }

        HRESULT RequestWindowThreadSwapChainLifecycle(
            SwapChainLifecycleOperation operation,
            UINT width = 0,
            UINT height = 0) noexcept
        {
            if (GetCurrentThreadId() == m_windowThreadId)
            {
                try
                {
                    if (operation == SwapChainLifecycleOperation::Ensure)
                    {
                        EnsureSwapChainOnWindowThread(width, height);
                    }
                    else if (operation == SwapChainLifecycleOperation::Release)
                    {
                        ReleaseSwapChainOnWindowThread();
                    }
                    else
                    {
                        return E_INVALIDARG;
                    }
                    return S_OK;
                }
                catch (...)
                {
                    return E_FAIL;
                }
            }

            if (!IsWindow(m_hwnd))
            {
                return E_HANDLE;
            }

            m_requestedSwapChainWidth.store(width, std::memory_order_relaxed);
            m_requestedSwapChainHeight.store(height, std::memory_order_relaxed);
            m_swapChainLifecycleResult.store(E_PENDING, std::memory_order_relaxed);
            SwapChainLifecycleOperation expected = SwapChainLifecycleOperation::None;
            if (!m_swapChainLifecycleOperation.compare_exchange_strong(
                    expected,
                    operation,
                    std::memory_order_release,
                    std::memory_order_relaxed))
            {
                return E_PENDING;
            }

            DWORD_PTR messageResult = static_cast<DWORD_PTR>(E_FAIL);
            SetLastError(ERROR_SUCCESS);
            const LRESULT delivered = SendMessageTimeoutW(
                m_hwnd,
                kSwapChainLifecycleMessage,
                0,
                0,
                SMTO_ABORTIFHUNG | SMTO_BLOCK | SMTO_ERRORONEXIT,
                5000,
                &messageResult);
            if (delivered == 0)
            {
                expected = operation;
                m_swapChainLifecycleOperation.compare_exchange_strong(
                    expected,
                    SwapChainLifecycleOperation::None,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed);
                const DWORD error = GetLastError();
                return HRESULT_FROM_WIN32(error != ERROR_SUCCESS ? error : ERROR_TIMEOUT);
            }
            const HRESULT messageHr =
                static_cast<HRESULT>(static_cast<LONG>(messageResult));
            const HRESULT synchronizedHr =
                m_swapChainLifecycleResult.load(std::memory_order_acquire);
            return FAILED(messageHr) ? messageHr : synchronizedHr;
        }

        HRESULT ReleaseSwapChainOnOwningThread() noexcept
        {
            if (!m_swapChain && !m_frameLatencyWaitable)
            {
                return S_OK;
            }
            return RequestWindowThreadSwapChainLifecycle(
                SwapChainLifecycleOperation::Release);
        }

        void EnsureSwapChainOnWindowThread(UINT width, UINT height)
        {
            if (GetCurrentThreadId() != m_windowThreadId)
            {
                throw std::runtime_error(
                    "Swapchain recreation attempted outside the output HWND owner thread.");
            }

            const DXGI_FORMAT requestedFormat = SwapChainFormatForMode();
            if (m_swapChain &&
                m_swapWidth == width &&
                m_swapHeight == height &&
                m_swapFormat == requestedFormat)
            {
                return;
            }

            const bool replacingSwapChain = m_swapChain != nullptr;
            m_framePacer.ResetTimeline();
            ResolveLatestPresentationStatisticsBeforeReset();
            FlushPendingPresentationTimings(
                replacingSwapChain
                    ? FrameDropReason::ResizeReset
                    : FrameDropReason::PresentationReset);

            ResetSwapChainDependentResources();
            ReleaseSwapChainOnWindowThread();
            ThrowIfFailed(
                m_factory->MakeWindowAssociation(m_hwnd, DXGI_MWA_NO_ALT_ENTER),
                "MakeWindowAssociation");
            ThrowIfFailed(
                CreateSwapChainForFormat(width, height, requestedFormat) ? S_OK : E_FAIL,
                "Create NVIDIA D3D11 swapchain");
        }

        void EnsureSwapChain(UINT width, UINT height)
        {
            const DXGI_FORMAT requestedFormat = SwapChainFormatForMode();
            if (m_swapChain &&
                m_swapWidth == width &&
                m_swapHeight == height &&
                m_swapFormat == requestedFormat)
            {
                return;
            }

            ThrowIfFailed(
                RequestWindowThreadSwapChainLifecycle(
                    SwapChainLifecycleOperation::Ensure,
                    width,
                    height),
                "Recreate swapchain on output HWND owner");
        }

        ID3D11RenderTargetView* CurrentSwapBackBufferRtv(ComPtr<ID3D11Texture2D>& backBuffer)
        {
            if (!m_swapChain)
            {
                throw std::runtime_error("Swapchain is not initialized.");
            }

            UINT index = m_swapChain->GetCurrentBackBufferIndex() % kFrameCount;
            if (!m_swapBackBuffers[index] || !m_swapBackBufferRtvs[index])
            {
                m_swapBackBuffers[index].Reset();
                m_swapBackBufferRtvs[index].Reset();
                m_swapBackBufferUavs[index].Reset();
                ThrowIfFailed(
                    m_swapChain->GetBuffer(index, IID_PPV_ARGS(&m_swapBackBuffers[index])),
                    "Get NVIDIA swapchain back buffer");
                ThrowIfFailed(
                    m_device->CreateRenderTargetView(m_swapBackBuffers[index].Get(), nullptr, &m_swapBackBufferRtvs[index]),
                    "Create NVIDIA swapchain RTV");
                D3D11_TEXTURE2D_DESC bbDesc{};
                m_swapBackBuffers[index]->GetDesc(&bbDesc);
                if (bbDesc.BindFlags & D3D11_BIND_UNORDERED_ACCESS)
                {
                    m_device->CreateUnorderedAccessView(
                        m_swapBackBuffers[index].Get(),
                        nullptr,
                        &m_swapBackBufferUavs[index]);
                }
            }

            backBuffer = m_swapBackBuffers[index];
            return m_swapBackBufferRtvs[index].Get();
        }

        ID3D11UnorderedAccessView* CurrentSwapBackBufferUav(ComPtr<ID3D11Texture2D>& backBuffer)
        {
            CurrentSwapBackBufferRtv(backBuffer);
            UINT index = m_swapChain->GetCurrentBackBufferIndex() % kFrameCount;
            return m_swapBackBufferUavs[index].Get();
        }

        void EnsureShaderSource(const D3D11_TEXTURE2D_DESC& sourceDesc)
        {
            if (m_shaderSource &&
                m_sourceWidth == sourceDesc.Width &&
                m_sourceHeight == sourceDesc.Height &&
                m_sourceFormat == sourceDesc.Format)
            {
                return;
            }

            m_shaderSource.Reset();
            m_shaderSourceSrv.Reset();

            D3D11_TEXTURE2D_DESC desc{};
            desc.Width = sourceDesc.Width;
            desc.Height = sourceDesc.Height;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = sourceDesc.Format;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            ThrowIfFailed(m_device->CreateTexture2D(&desc, nullptr, &m_shaderSource), "Create NVIDIA shader source texture");
            ThrowIfFailed(m_device->CreateShaderResourceView(m_shaderSource.Get(), nullptr, &m_shaderSourceSrv), "Create NVIDIA shader source SRV");

            m_sourceWidth = sourceDesc.Width;
            m_sourceHeight = sourceDesc.Height;
            m_sourceFormat = sourceDesc.Format;
        }

        ID3D11ShaderResourceView* DirectSourceSrvOrNull(ID3D11Texture2D* source, const D3D11_TEXTURE2D_DESC& sourceDesc)
        {
            return m_sourceViewCache.FindOrCreate(m_device.Get(), source, sourceDesc);
        }

        DXGI_FORMAT SelectWorkingInputFormat()
        {
            const std::array<DXGI_FORMAT, 4> candidates = {
                DXGI_FORMAT_R10G10B10A2_UNORM,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                kModelTextureFormat,
            };

            for (DXGI_FORMAT format : candidates)
            {
                UINT vpFlags = 0;
                if (FAILED(m_vpEnum->CheckVideoProcessorFormat(format, &vpFlags)) ||
                    (vpFlags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT) == 0)
                {
                    continue;
                }

                UINT formatSupport = 0;
                if (FAILED(m_device->CheckFormatSupport(format, &formatSupport)) ||
                    (formatSupport & D3D11_FORMAT_SUPPORT_RENDER_TARGET) == 0)
                {
                    continue;
                }

                return format;
            }

            throw std::runtime_error("NVIDIA video processor did not report a supported RGB input format.");
        }

        DXGI_FORMAT SelectVideoProcessorOutputFormat()
        {
            const std::array<DXGI_FORMAT, 3> candidates = {
                DXGI_FORMAT_R10G10B10A2_UNORM,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM,
            };

            for (DXGI_FORMAT format : candidates)
            {
                UINT vpFlags = 0;
                if (FAILED(m_vpEnum->CheckVideoProcessorFormat(format, &vpFlags)) ||
                    (vpFlags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT) == 0)
                {
                    continue;
                }

                UINT formatSupport = 0;
                if (FAILED(m_device->CheckFormatSupport(format, &formatSupport)) ||
                    (formatSupport & D3D11_FORMAT_SUPPORT_SHADER_SAMPLE) == 0 ||
                    (formatSupport & D3D11_FORMAT_SUPPORT_RENDER_TARGET) == 0)
                {
                    continue;
                }

                return format;
            }

            throw std::runtime_error("NVIDIA video processor did not report a supported sampled RGB output format.");
        }

        void RequireVideoProcessorOutputSupport(DXGI_FORMAT format)
        {
            UINT vpFlags = 0;
            if (FAILED(m_vpEnum->CheckVideoProcessorFormat(format, &vpFlags)) ||
                (vpFlags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT) == 0)
            {
                throw std::runtime_error(std::string("NVIDIA video processor does not support output format ") + FormatName(format));
            }
        }

        void EnsureWorkingInput(UINT width, UINT height)
        {
            if (m_workingFormat == DXGI_FORMAT_UNKNOWN)
            {
                m_workingFormat = kSdrOutputFormat;
            }

            const bool requireExternalInterop = IsNeuralUpscaler(m_activeUpscalerMode);
            bool hasRequiredInteropFlags = !requireExternalInterop;
            if (m_workingInput && requireExternalInterop)
            {
                D3D11_TEXTURE2D_DESC currentDesc{};
                m_workingInput->GetDesc(&currentDesc);
                hasRequiredInteropFlags =
                    (currentDesc.MiscFlags & (D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE)) != 0;
            }
            if (m_workingInput &&
                m_workingWidth == width &&
                m_workingHeight == height &&
                m_workingInputFormat == m_workingFormat &&
                hasRequiredInteropFlags)
            {
                return;
            }

            m_videoProcessor.Reset();
            m_vpEnum.Reset();
            m_workingInput.Reset();
            m_workingInputRtv.Reset();
            m_workingInputSrv.Reset();

            D3D11_TEXTURE2D_DESC desc{};
            desc.Width = width;
            desc.Height = height;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = m_workingFormat;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            if (requireExternalInterop)
            {
                desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
            }
            ThrowIfFailed(requireExternalInterop
                ? aess::neural::CreateInteropTexture(m_device.Get(), desc, &m_workingInput)
                : m_device->CreateTexture2D(&desc, nullptr, &m_workingInput), "Create working input texture");
            ThrowIfFailed(m_device->CreateRenderTargetView(m_workingInput.Get(), nullptr, &m_workingInputRtv), "Create NVIDIA VP input RTV");
            ThrowIfFailed(m_device->CreateShaderResourceView(m_workingInput.Get(), nullptr, &m_workingInputSrv), "Create NVIDIA VP input SRV");

            m_workingWidth = width;
            m_workingHeight = height;
            m_workingInputFormat = m_workingFormat;
        }

        void EnsureShaderWorkingFormat()
        {
            if (m_workingFormat == kModelTextureFormat)
            {
                return;
            }

            m_workingFormat = kModelTextureFormat;
            m_videoProcessor.Reset();
            m_vpEnum.Reset();
            m_workingInput.Reset();
            m_workingInputRtv.Reset();
            m_workingInputSrv.Reset();
            m_workingInputFormat = DXGI_FORMAT_UNKNOWN;
        }

        void EnsureVideoProcessorOutput(UINT width, UINT height)
        {
            if (m_videoProcessorOutputFormat == DXGI_FORMAT_UNKNOWN)
            {
                m_videoProcessorOutputFormat = kSdrOutputFormat;
            }

            if (m_videoProcessorOutput &&
                m_videoProcessorOutputWidth == width &&
                m_videoProcessorOutputHeight == height)
            {
                return;
            }

            m_videoProcessorOutput.Reset();
            m_videoProcessorOutputRtv.Reset();
            m_videoProcessorOutputSrv.Reset();
            m_videoProcessorOutputUav.Reset();

            D3D11_TEXTURE2D_DESC desc{};
            desc.Width = width;
            desc.Height = height;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = m_videoProcessorOutputFormat;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            ThrowIfFailed(m_device->CreateTexture2D(&desc, nullptr, &m_videoProcessorOutput), "Create NVIDIA VP output texture");
            ThrowIfFailed(m_device->CreateRenderTargetView(m_videoProcessorOutput.Get(), nullptr, &m_videoProcessorOutputRtv), "Create NVIDIA VP output RTV");
            ThrowIfFailed(m_device->CreateShaderResourceView(m_videoProcessorOutput.Get(), nullptr, &m_videoProcessorOutputSrv), "Create NVIDIA VP output SRV");

            m_videoProcessorOutputWidth = width;
            m_videoProcessorOutputHeight = height;
        }

        void EnsureShaderOutput(UINT width, UINT height, DXGI_FORMAT format, bool unorderedAccess)
        {
            const bool requireExternalInterop = IsNeuralUpscaler(m_activeUpscalerMode);
            bool hasRequiredInteropFlags = !requireExternalInterop;
            if (m_videoProcessorOutput && requireExternalInterop)
            {
                D3D11_TEXTURE2D_DESC currentDesc{};
                m_videoProcessorOutput->GetDesc(&currentDesc);
                hasRequiredInteropFlags =
                    (currentDesc.MiscFlags & (D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE)) != 0;
            }
            if (m_videoProcessorOutput &&
                m_videoProcessorOutputWidth == width &&
                m_videoProcessorOutputHeight == height &&
                m_videoProcessorOutputFormat == format &&
                (!unorderedAccess || m_videoProcessorOutputUav) &&
                hasRequiredInteropFlags)
            {
                return;
            }

            m_videoProcessorOutput.Reset();
            m_videoProcessorOutputRtv.Reset();
            m_videoProcessorOutputSrv.Reset();
            m_videoProcessorOutputUav.Reset();

            D3D11_TEXTURE2D_DESC desc{};
            desc.Width = width;
            desc.Height = height;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = format;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            if (unorderedAccess)
            {
                desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
            }
            if (requireExternalInterop)
            {
                desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
            }
            ThrowIfFailed(requireExternalInterop
                ? aess::neural::CreateInteropTexture(m_device.Get(), desc, &m_videoProcessorOutput)
                : m_device->CreateTexture2D(&desc, nullptr, &m_videoProcessorOutput), "Create upscaler output texture");
            ThrowIfFailed(m_device->CreateRenderTargetView(m_videoProcessorOutput.Get(), nullptr, &m_videoProcessorOutputRtv), "Create D3D11 upscaler output RTV");
            ThrowIfFailed(m_device->CreateShaderResourceView(m_videoProcessorOutput.Get(), nullptr, &m_videoProcessorOutputSrv), "Create D3D11 upscaler output SRV");
            if (unorderedAccess)
            {
                ThrowIfFailed(m_device->CreateUnorderedAccessView(m_videoProcessorOutput.Get(), nullptr, &m_videoProcessorOutputUav), "Create D3D11 upscaler output UAV");
            }

            m_videoProcessorOutputWidth = width;
            m_videoProcessorOutputHeight = height;
            m_videoProcessorOutputFormat = format;
        }

        void EnsurePresentResample(UINT width, UINT height, DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN)
        {
            const DXGI_FORMAT targetFormat = (format != DXGI_FORMAT_UNKNOWN) ? format : m_videoProcessorOutputFormat;
            if (m_presentResample &&
                m_presentResampleWidth == width &&
                m_presentResampleHeight == height &&
                m_presentResampleFormat == targetFormat)
            {
                return;
            }

            m_presentResample.Reset();
            m_presentResampleRtv.Reset();
            m_presentResampleSrv.Reset();

            D3D11_TEXTURE2D_DESC desc{};
            desc.Width = width;
            desc.Height = height;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = targetFormat;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            ThrowIfFailed(m_device->CreateTexture2D(&desc, nullptr, &m_presentResample), "Create NVIDIA Magic Kernel resample texture");
            ThrowIfFailed(m_device->CreateRenderTargetView(m_presentResample.Get(), nullptr, &m_presentResampleRtv), "Create NVIDIA Magic Kernel resample RTV");
            ThrowIfFailed(m_device->CreateShaderResourceView(m_presentResample.Get(), nullptr, &m_presentResampleSrv), "Create NVIDIA Magic Kernel resample SRV");

            m_presentResampleWidth = width;
            m_presentResampleHeight = height;
            m_presentResampleFormat = targetFormat;
        }

        void EnsureUpscaleScratch(UINT width, UINT height, DXGI_FORMAT format, bool unorderedAccess)
        {
            if (m_upscaleScratch &&
                m_upscaleScratchWidth == width &&
                m_upscaleScratchHeight == height &&
                m_upscaleScratchFormat == format &&
                (!unorderedAccess || m_upscaleScratchUav))
            {
                return;
            }

            m_upscaleScratch.Reset();
            m_upscaleScratchRtv.Reset();
            m_upscaleScratchSrv.Reset();
            m_upscaleScratchUav.Reset();

            D3D11_TEXTURE2D_DESC desc{};
            desc.Width = width;
            desc.Height = height;
            desc.MipLevels = 1;
            desc.ArraySize = 1;
            desc.Format = format;
            desc.SampleDesc.Count = 1;
            desc.Usage = D3D11_USAGE_DEFAULT;
            desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            if (unorderedAccess)
            {
                desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
            }
            ThrowIfFailed(m_device->CreateTexture2D(&desc, nullptr, &m_upscaleScratch), "Create D3D11 upscaler scratch texture");
            ThrowIfFailed(m_device->CreateRenderTargetView(m_upscaleScratch.Get(), nullptr, &m_upscaleScratchRtv), "Create D3D11 upscaler scratch RTV");
            ThrowIfFailed(m_device->CreateShaderResourceView(m_upscaleScratch.Get(), nullptr, &m_upscaleScratchSrv), "Create D3D11 upscaler scratch SRV");
            if (unorderedAccess)
            {
                ThrowIfFailed(m_device->CreateUnorderedAccessView(m_upscaleScratch.Get(), nullptr, &m_upscaleScratchUav), "Create D3D11 upscaler scratch UAV");
            }

            m_upscaleScratchWidth = width;
            m_upscaleScratchHeight = height;
            m_upscaleScratchFormat = format;
        }

        void DrawFullscreenPass(
            ID3D11RenderTargetView* target,
            ID3D11ShaderResourceView* source,
            ID3D11PixelShader* pixelShader,
            UINT sourceWidth,
            UINT sourceHeight,
            UINT width,
            UINT height,
            ID3D11SamplerState* sampler = nullptr)
        {
            if (!target || !source || !pixelShader)
            {
                throw std::runtime_error("D3D11 fullscreen shader pass is missing a required view or shader.");
            }

            UpdateSourceCropConstants(0, 0, sourceWidth, sourceHeight);
            UploadToneConstantsIfDirty();

            D3D11_VIEWPORT viewport{};
            viewport.Width = static_cast<FLOAT>(width);
            viewport.Height = static_cast<FLOAT>(height);
            viewport.MaxDepth = 1.0f;

            ID3D11RenderTargetView* targets[] = { target };
            ID3D11ShaderResourceView* srvs[] = { source };
            ID3D11SamplerState* samplers[] = {
                sampler ? sampler : m_linearSampler.Get() };

            m_context->OMSetRenderTargets(1, targets, nullptr);
            m_context->RSSetViewports(1, &viewport);
            m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            m_context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
            m_context->PSSetShader(pixelShader, nullptr, 0);
            m_context->PSSetShaderResources(0, 1, srvs);
            m_context->PSSetSamplers(0, 1, samplers);
            BindToneConstants();
            m_context->Draw(3, 0);
            MarkContextDirty();

            ID3D11ShaderResourceView* nullSrvs[] = { nullptr };
            ID3D11RenderTargetView* nullTargets[] = { nullptr };
            ID3D11Buffer* nullBuffers[] = { nullptr };
            m_context->PSSetShaderResources(0, 1, nullSrvs);
            m_context->PSSetConstantBuffers(0, 1, nullBuffers);
            m_context->OMSetRenderTargets(1, nullTargets, nullptr);
        }

        void PrepareWorkingInput(
            ID3D11Texture2D* source,
            ID3D11ShaderResourceView* suppliedSourceSrv,
            const D3D11_TEXTURE2D_DESC& sourceDesc,
            UINT cropLeft,
            UINT cropTop,
            UINT cropWidth,
            UINT cropHeight,
            UINT width,
            UINT height)
        {
            EnsureWorkingInput(width, height);

            ID3D11ShaderResourceView* sourceSrv = suppliedSourceSrv;
            if (!sourceSrv)
            {
                sourceSrv = DirectSourceSrvOrNull(source, sourceDesc);
            }
            if (!sourceSrv)
            {
                EnsureShaderSource(sourceDesc);
                {
                    auto scope = m_profiler.Stage("source-copy");
                    m_context->CopyResource(m_shaderSource.Get(), source);
                    MarkContextDirty();
                }
                sourceSrv = m_shaderSourceSrv.Get();
            }

            D3D11_VIEWPORT viewport{};
            viewport.Width = static_cast<FLOAT>(width);
            viewport.Height = static_cast<FLOAT>(height);
            viewport.MaxDepth = 1.0f;

            UpdateSourceDimensions(sourceDesc.Width, sourceDesc.Height);
            UpdateToneConstants(width, height);
            UpdateSourceCropConstants(cropLeft, cropTop, cropWidth, cropHeight);
            UploadToneConstantsIfDirty();

            const bool resized = cropWidth != width || cropHeight != height;
            const bool magicSharp = resized;
            if (magicSharp)
            {
                EnsureUpscaleScratch(width, height, m_workingFormat, false);
            }

            if (!magicSharp && m_csPrepareWorkingInput && m_workingInputUav)
            {
                auto scope = m_profiler.Stage("working-input-cs");
                ID3D11ShaderResourceView* srvs[] = { sourceSrv };
                ID3D11UnorderedAccessView* uavs[] = { m_workingInputUav.Get() };
                ID3D11SamplerState* samplers[] = { m_linearSampler.Get() };
                m_context->CSSetShader(m_csPrepareWorkingInput.Get(), nullptr, 0);
                m_context->CSSetShaderResources(0, 1, srvs);
                m_context->CSSetSamplers(0, 1, samplers);
                m_context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
                BindToneConstants();
                m_context->Dispatch((width + 15) / 16, (height + 15) / 16, 1);
                ID3D11UnorderedAccessView* nullUavs[] = { nullptr };
                ID3D11ShaderResourceView* nullSrvs[] = { nullptr };
                ID3D11SamplerState* nullSamplers[] = { nullptr };
                ID3D11Buffer* nullBuffers[] = { nullptr };
                m_context->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);
                m_context->CSSetShaderResources(0, 1, nullSrvs);
                m_context->CSSetSamplers(0, 1, nullSamplers);
                m_context->CSSetConstantBuffers(0, 1, nullBuffers);
                m_context->CSSetShader(nullptr, nullptr, 0);
                MarkContextDirty();
            }
            else
            {
                ID3D11PixelShader* workingInputShader =
                    resized ? m_workingInputMagicPixelShader.Get() : m_pixelShader.Get();
                const char* workingInputStage =
                    resized ? "working-input-magic" : "working-input";
                ID3D11RenderTargetView* targets[] = {
                    magicSharp ? m_upscaleScratchRtv.Get() : m_workingInputRtv.Get()
                };
                ID3D11ShaderResourceView* srvs[] = { sourceSrv };
                ID3D11SamplerState* samplers[] = { m_linearSampler.Get() };

                {
                    auto scope = m_profiler.Stage(workingInputStage);
                    m_context->OMSetRenderTargets(1, targets, nullptr);
                    m_context->RSSetViewports(1, &viewport);
                    m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    m_context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
                    m_context->PSSetShader(workingInputShader, nullptr, 0);
                    m_context->PSSetShaderResources(0, 1, srvs);
                    m_context->PSSetSamplers(0, 1, samplers);
                    BindToneConstants();
                    m_context->Draw(3, 0);
                    MarkContextDirty();
                }

                ID3D11ShaderResourceView* nullSrvs[] = { nullptr };
                ID3D11RenderTargetView* nullTargets[] = { nullptr };
                ID3D11Buffer* nullBuffers[] = { nullptr };
                m_context->PSSetShaderResources(0, 1, nullSrvs);
                m_context->PSSetConstantBuffers(0, 1, nullBuffers);
                m_context->OMSetRenderTargets(1, nullTargets, nullptr);
            }
            UpdateSourceCropConstants(0, 0, width, height);

            if (magicSharp)
            {
                // Magic Kernel Sharp is a factored algorithm: the Sharp pass
                // must immediately follow the Magic Kernel resize. Running a
                // neural model between those two operations is not equivalent.
                {
                    auto scope = m_profiler.Stage("working-input-magic-sharp");
                    DrawFullscreenPass(
                        m_workingInputRtv.Get(),
                        m_upscaleScratchSrv.Get(),
                        m_magicSharpEncodedPixelShader.Get(),
                        width,
                        height,
                        width,
                        height);
                }
            }
        }

        bool UpdateSourceColorSpace(DXGI_COLOR_SPACE_TYPE colorSpace, DXGI_FORMAT format, CaptureColorProvenance provenance)
        {
            SelectSdrSourceTransfer(colorSpace, format, provenance);
            const bool changed = m_toneConstants.sourceTransfer != kModelTransferSdrGamma ||
                m_toneConstants.sourcePrimaries != kModelPrimariesRec709;
            m_toneConstants.sourceTransfer = kModelTransferSdrGamma;
            m_toneConstants.sourcePrimaries = kModelPrimariesRec709;
            m_toneConstantsDirty |= changed;
            return changed;
        }


        void UpdateCompositionScale(UINT contentWidth, UINT contentHeight)
        {
            if (!m_compositionVisual || !m_compositionScaleTransform ||
                contentWidth == 0 || contentHeight == 0)
            {
                return;
            }
            UINT clientWidth = 1;
            UINT clientHeight = 1;
            GetClientSize(m_hwnd, clientWidth, clientHeight);
            const float scale = std::min(
                static_cast<float>(clientWidth) / static_cast<float>(contentWidth),
                static_cast<float>(clientHeight) / static_cast<float>(contentHeight));
            const float scaledWidth = static_cast<float>(contentWidth) * scale;
            const float scaledHeight = static_cast<float>(contentHeight) * scale;
            ThrowIfFailed(m_compositionScaleTransform->SetCenterX(0.0f), "Set composition scale center X");
            ThrowIfFailed(m_compositionScaleTransform->SetCenterY(0.0f), "Set composition scale center Y");
            ThrowIfFailed(m_compositionScaleTransform->SetScaleX(scale), "Set composition scale X");
            ThrowIfFailed(m_compositionScaleTransform->SetScaleY(scale), "Set composition scale Y");
            ThrowIfFailed(
                m_compositionVisual->SetOffsetX((static_cast<float>(clientWidth) - scaledWidth) * 0.5f),
                "Center composition output X");
            ThrowIfFailed(
                m_compositionVisual->SetOffsetY((static_cast<float>(clientHeight) - scaledHeight) * 0.5f),
                "Center composition output Y");
        }

        void UpdateSourceCropConstants(UINT left, UINT top, UINT width, UINT height)
        {
            const float newLeft = static_cast<float>(left);
            const float newTop = static_cast<float>(top);
            const float newWidth = static_cast<float>(std::max<UINT>(1, width));
            const float newHeight = static_cast<float>(std::max<UINT>(1, height));
            if (m_toneConstants.sourceCropLeft == newLeft &&
                m_toneConstants.sourceCropTop == newTop &&
                m_toneConstants.sourceCropWidth == newWidth &&
                m_toneConstants.sourceCropHeight == newHeight)
            {
                return;
            }

            m_toneConstants.sourceCropLeft = newLeft;
            m_toneConstants.sourceCropTop = newTop;
            m_toneConstants.sourceCropWidth = newWidth;
            m_toneConstants.sourceCropHeight = newHeight;
            m_toneConstantsDirty = true;
        }

        void EnsureVideoProcessor(UINT inputWidth, UINT inputHeight, UINT outputWidth, UINT outputHeight)
        {
            if (m_videoProcessor &&
                m_vpInputWidth == inputWidth &&
                m_vpInputHeight == inputHeight &&
                m_vpOutputWidth == outputWidth &&
                m_vpOutputHeight == outputHeight &&
                m_vpOutputFormat == (UsesShaderBridge() ? m_videoProcessorOutputFormat : m_swapFormat) &&
                m_vpWorkingFormat == m_workingFormat)
            {
                return;
            }

            m_videoProcessor.Reset();
            m_vpEnum.Reset();
            m_cachedVpInputView.Reset();
            m_cachedVpOutputView.Reset();
            m_cachedVpOutputTexture.Reset();

            D3D11_VIDEO_PROCESSOR_CONTENT_DESC desc{};
            desc.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
            desc.InputWidth = inputWidth;
            desc.InputHeight = inputHeight;
            desc.OutputWidth = outputWidth;
            desc.OutputHeight = outputHeight;
            desc.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;
            ThrowIfFailed(m_videoDevice->CreateVideoProcessorEnumerator(&desc, &m_vpEnum), "Create D3D11 video processor enumerator");

            DXGI_FORMAT selectedFormat = SelectWorkingInputFormat();
            if (selectedFormat != m_workingFormat)
            {
                m_workingFormat = selectedFormat;
                m_workingInput.Reset();
                m_workingInputRtv.Reset();
                m_workingInputFormat = DXGI_FORMAT_UNKNOWN;
            }

            if (UsesShaderBridge())
            {
                DXGI_FORMAT selectedOutputFormat = SelectVideoProcessorOutputFormat();
                if (selectedOutputFormat != m_videoProcessorOutputFormat)
                {
                    m_videoProcessorOutputFormat = selectedOutputFormat;
                    m_videoProcessorOutput.Reset();
                    m_videoProcessorOutputRtv.Reset();
                    m_videoProcessorOutputSrv.Reset();
                    m_videoProcessorOutputUav.Reset();
                    m_videoProcessorOutputWidth = 0;
                    m_videoProcessorOutputHeight = 0;
                }
                EnsureVideoProcessorOutput(outputWidth, outputHeight);
            }
            else
            {
                RequireVideoProcessorOutputSupport(m_swapFormat);
                m_videoProcessorOutput.Reset();
                m_videoProcessorOutputRtv.Reset();
                m_videoProcessorOutputSrv.Reset();
                m_videoProcessorOutputUav.Reset();
                m_videoProcessorOutputWidth = 0;
                m_videoProcessorOutputHeight = 0;
                m_videoProcessorOutputFormat = m_swapFormat;
            }

            D3D11_VIDEO_PROCESSOR_CAPS caps{};
            ThrowIfFailed(m_vpEnum->GetVideoProcessorCaps(&caps), "Get D3D11 video processor caps");
            UINT processorIndex = 0;
            for (UINT i = 0; i < caps.RateConversionCapsCount; ++i)
            {
                D3D11_VIDEO_PROCESSOR_RATE_CONVERSION_CAPS rateCaps{};
                if (SUCCEEDED(m_vpEnum->GetVideoProcessorRateConversionCaps(i, &rateCaps)))
                {
                    processorIndex = i;
                    break;
                }
            }

            ThrowIfFailed(m_videoDevice->CreateVideoProcessor(m_vpEnum.Get(), processorIndex, &m_videoProcessor), "Create D3D11 video processor");

            RECT sourceRect{ 0, 0, static_cast<LONG>(inputWidth), static_cast<LONG>(inputHeight) };
            RECT outputRect{ 0, 0, static_cast<LONG>(outputWidth), static_cast<LONG>(outputHeight) };
            m_videoContext->VideoProcessorSetStreamFrameFormat(m_videoProcessor.Get(), 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
            m_videoContext->VideoProcessorSetStreamSourceRect(m_videoProcessor.Get(), 0, TRUE, &sourceRect);
            m_videoContext->VideoProcessorSetStreamDestRect(m_videoProcessor.Get(), 0, TRUE, &outputRect);
            m_videoContext->VideoProcessorSetOutputTargetRect(m_videoProcessor.Get(), TRUE, &outputRect);
            m_videoContext->VideoProcessorSetStreamAutoProcessingMode(m_videoProcessor.Get(), 0, FALSE);
            m_videoContext->VideoProcessorSetStreamOutputRate(
                m_videoProcessor.Get(),
                0,
                D3D11_VIDEO_PROCESSOR_OUTPUT_RATE_NORMAL,
                FALSE,
                nullptr);

            if (m_videoContext1)
            {
                DXGI_COLOR_SPACE_TYPE inputColorSpace = m_workingFormat == kModelTextureFormat
                    ? DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709
                    : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
                m_videoContext1->VideoProcessorSetStreamColorSpace1(
                    m_videoProcessor.Get(),
                    0,
                    inputColorSpace);
                m_videoContext1->VideoProcessorSetOutputColorSpace1(
                    m_videoProcessor.Get(),
                    VideoProcessorOutputColorSpace());
            }
            else
            {
                D3D11_VIDEO_PROCESSOR_COLOR_SPACE csp{};
                csp.YCbCr_Matrix = 1;
                csp.Nominal_Range = 2;
                m_videoContext->VideoProcessorSetStreamColorSpace(m_videoProcessor.Get(), 0, &csp);
                m_videoContext->VideoProcessorSetOutputColorSpace(m_videoProcessor.Get(), &csp);
            }

            m_vpInputWidth = inputWidth;
            m_vpInputHeight = inputHeight;
            m_vpOutputWidth = outputWidth;
            m_vpOutputHeight = outputHeight;
            m_vpOutputFormat = UsesShaderBridge() ? m_videoProcessorOutputFormat : m_swapFormat;
            m_vpWorkingFormat = m_workingFormat;
            m_outputSeq = 0;
        }

        void RenderVideoProcessor(
            UINT inputWidth,
            UINT inputHeight,
            UINT outputWidth,
            UINT outputHeight,
            ComPtr<ID3D11Texture2D>& vpOutput)
        {
            EnsureVideoProcessor(inputWidth, inputHeight, outputWidth, outputHeight);
            ComPtr<ID3D11Texture2D> outputTexture;
            if (UsesShaderBridge())
            {
                EnsureVideoProcessorOutput(outputWidth, outputHeight);
                outputTexture = m_videoProcessorOutput;
            }
            else
            {
                ThrowIfFailed(m_swapChain->GetBuffer(0, IID_PPV_ARGS(&outputTexture)), "Get NVIDIA direct VP swapchain back buffer");
            }

            D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC inDesc{};
            inDesc.FourCC = 0;
            inDesc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
            inDesc.Texture2D.MipSlice = 0;
            inDesc.Texture2D.ArraySlice = 0;

            if (!m_cachedVpInputView)
            {
                HRESULT inputViewHr = m_videoDevice->CreateVideoProcessorInputView(
                    m_workingInput.Get(),
                    m_vpEnum.Get(),
                    &inDesc,
                    &m_cachedVpInputView);
                if (FAILED(inputViewHr))
                {
                    throw std::runtime_error(
                        std::string("Create D3D11 video processor input view failed for ") +
                        FormatName(m_workingFormat) +
                        " with HRESULT " +
                        HrToHex(inputViewHr));
                }
            }

            D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC outDesc{};
            outDesc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
            outDesc.Texture2D.MipSlice = 0;

            if (!m_cachedVpOutputView || m_cachedVpOutputTexture.Get() != outputTexture.Get())
            {
                m_cachedVpOutputView.Reset();
                m_cachedVpOutputTexture = outputTexture;
                ThrowIfFailed(
                    m_videoDevice->CreateVideoProcessorOutputView(
                        outputTexture.Get(),
                        m_vpEnum.Get(),
                        &outDesc,
                        &m_cachedVpOutputView),
                    "Create D3D11 video processor output view");
            }

            D3D11_VIDEO_PROCESSOR_STREAM stream{};
            stream.Enable = TRUE;
            stream.OutputIndex = 0;
            stream.InputFrameOrField = m_outputSeq;
            stream.pInputSurface = m_cachedVpInputView.Get();

            {
                auto scope = m_profiler.Stage("d3d11-video-processor");
                ThrowIfFailed(
                    m_videoContext->VideoProcessorBlt(
                        m_videoProcessor.Get(),
                        m_cachedVpOutputView.Get(),
                        m_outputSeq,
                        1,
                        &stream),
                    "D3D11 VideoProcessorBlt");
            }
            ++m_outputSeq;
            vpOutput = outputTexture;
        }



        void RenderNoUpscalerPassThrough(
            ID3D11Texture2D* source,
            ID3D11ShaderResourceView* suppliedSourceSrv,
            const D3D11_TEXTURE2D_DESC& sourceDesc,
            UINT cropLeft,
            UINT cropTop,
            UINT cropWidth,
            UINT cropHeight,
            UINT outputWidth,
            UINT outputHeight,
            ComPtr<ID3D11Texture2D>& upscaledOutput)
        {
            EnsureShaderOutput(outputWidth, outputHeight, kModelTextureFormat, false);

            ID3D11ShaderResourceView* sourceSrv = suppliedSourceSrv;
            if (!sourceSrv)
            {
                sourceSrv = DirectSourceSrvOrNull(source, sourceDesc);
            }
            if (!sourceSrv)
            {
                EnsureShaderSource(sourceDesc);
                {
                    auto scope = m_profiler.Stage("source-copy");
                    m_context->CopyResource(m_shaderSource.Get(), source);
                    MarkContextDirty();
                }
                sourceSrv = m_shaderSourceSrv.Get();
            }

            D3D11_VIEWPORT viewport{};
            viewport.Width = static_cast<FLOAT>(outputWidth);
            viewport.Height = static_cast<FLOAT>(outputHeight);
            viewport.MaxDepth = 1.0f;

            UpdateSourceDimensions(sourceDesc.Width, sourceDesc.Height);
            UpdateToneConstants(outputWidth, outputHeight);
            UpdateSourceCropConstants(cropLeft, cropTop, cropWidth, cropHeight);
            UploadToneConstantsIfDirty();

            const bool sameSize =
                cropWidth == outputWidth &&
                cropHeight == outputHeight;
            const char* passthroughStage = sameSize
                ? "no-upscaler-passthrough"
                : "no-upscaler-bilinear-resize";

            ID3D11RenderTargetView* targets[] = { m_videoProcessorOutputRtv.Get() };
            ID3D11ShaderResourceView* srvs[] = { sourceSrv };
            ID3D11SamplerState* samplers[] = {
                DebugViewOriginalActive() ? m_pointSampler.Get() : m_linearSampler.Get()
            };

            {
                auto scope = m_profiler.Stage(passthroughStage);
                m_context->OMSetRenderTargets(1, targets, nullptr);
                m_context->RSSetViewports(1, &viewport);
                m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                m_context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
                m_context->PSSetShader(m_pixelShader.Get(), nullptr, 0);
                m_context->PSSetShaderResources(0, 1, srvs);
                m_context->PSSetSamplers(0, 1, samplers);
                BindToneConstants();
                m_context->Draw(3, 0);
                MarkContextDirty();
            }

            ID3D11ShaderResourceView* nullSrvs[] = { nullptr };
            ID3D11RenderTargetView* nullTargets[] = { nullptr };
            ID3D11Buffer* nullBuffers[] = { nullptr };
            m_context->PSSetShaderResources(0, 1, nullSrvs);
            m_context->PSSetConstantBuffers(0, 1, nullBuffers);
            m_context->OMSetRenderTargets(1, nullTargets, nullptr);
            UpdateSourceCropConstants(0, 0, outputWidth, outputHeight);

            m_workingWidth = cropWidth;
            m_workingHeight = cropHeight;
            upscaledOutput = m_videoProcessorOutput;
        }

        void RenderNoUpscaler(UINT outputWidth, UINT outputHeight, ComPtr<ID3D11Texture2D>& upscaledOutput)
        {
            if (m_videoProcessorOutput && m_workingInput &&
                m_workingInput.Get() == m_videoProcessorOutput.Get())
            {
                upscaledOutput = m_videoProcessorOutput;
                return;
            }
            EnsureShaderOutput(outputWidth, outputHeight, kModelTextureFormat, false);
            UpdateSourceDimensions(m_workingWidth, m_workingHeight);
            UpdateToneConstants(outputWidth, outputHeight);
            const bool sameSize =
                m_workingWidth == outputWidth &&
                m_workingHeight == outputHeight;
            {
                auto scope = m_profiler.Stage(
                    sameSize
                        ? "no-upscaler-direct-copy"
                        : "no-upscaler-bilinear-resize");
                if (sameSize && m_workingInput)
                {
                    m_context->CopyResource(
                        m_videoProcessorOutput.Get(),
                        m_workingInput.Get());
                }
                else
                {
                    DrawFullscreenPass(
                        m_videoProcessorOutputRtv.Get(),
                        m_workingInputSrv.Get(),
                        m_bilinearResamplePixelShader.Get(),
                        m_workingWidth,
                        m_workingHeight,
                        outputWidth,
                        outputHeight);
                }
            }
            upscaledOutput = m_videoProcessorOutput;
        }

        void RenderNearestNeighbor(
            UINT outputWidth,
            UINT outputHeight,
            ComPtr<ID3D11Texture2D>& upscaledOutput)
        {
            EnsureShaderOutput(outputWidth, outputHeight, kModelTextureFormat, false);
            UpdateSourceDimensions(m_workingWidth, m_workingHeight);
            UpdateToneConstants(outputWidth, outputHeight);
            const bool sameSize =
                m_workingWidth == outputWidth &&
                m_workingHeight == outputHeight;
            {
                auto scope = m_profiler.Stage(
                    sameSize
                        ? "debug-original-direct-copy"
                        : "debug-original-nearest-resize");
                DrawFullscreenPass(
                    m_videoProcessorOutputRtv.Get(),
                    m_workingInputSrv.Get(),
                    sameSize
                        ? m_copySameSizePixelShader.Get()
                        : m_bilinearResamplePixelShader.Get(),
                    m_workingWidth,
                    m_workingHeight,
                    outputWidth,
                    outputHeight,
                    sameSize ? nullptr : m_pointSampler.Get());
            }
            upscaledOutput = m_videoProcessorOutput;
        }





















        void PresentDirectNoUpscalerToSwapChain(
            SynchronousReadyFrame& pending,
            ComPtr<ID3D11Texture2D>& backBuffer)
        {
            ID3D11RenderTargetView* backBufferRtv = CurrentSwapBackBufferRtv(backBuffer);
            const UINT outputWidth = pending.outputWidth;
            const UINT outputHeight = pending.outputHeight;

            const bool sameSize = (pending.directCropWidth == outputWidth &&
                                   pending.directCropHeight == outputHeight &&
                                   pending.directCropLeft == 0 &&
                                   pending.directCropTop == 0);
            const bool sameFormat = (pending.directSourceDesc.Format == m_swapFormat);
            const bool identityColor = (pending.result.tone.sourceTransfer != kModelTransferHdr10Pq) &&
                                       (m_outputMode == OutputMode::DirectSdr || m_outputMode == OutputMode::SdrShaderBridge) &&
                                       pending.result.tone.saturation == 1.0f &&
                                       pending.result.tone.contrast == 1.0f &&
                                       pending.result.tone.blackPoint == 0.0f;

            if (sameSize && sameFormat && identityColor && pending.directSourceTexture)
            {
                auto scope = m_profiler.Stage("present-direct-copy-resource");
                m_context->CopyResource(backBuffer.Get(), pending.directSourceTexture.Get());
                MarkContextDirty();
            }
            else
            {
                UpdateSourceDimensions(pending.directSourceDesc.Width, pending.directSourceDesc.Height);
                UpdateToneConstants(outputWidth, outputHeight);
                UpdateSourceCropConstants(
                    pending.directCropLeft,
                    pending.directCropTop,
                    pending.directCropWidth,
                    pending.directCropHeight);
                UploadToneConstantsIfDirty();

                ID3D11ShaderResourceView* sourceSrv = pending.directSourceSrv.Get();
                if (!sourceSrv && pending.directSourceTexture)
                {
                    sourceSrv = DirectSourceSrvOrNull(
                        pending.directSourceTexture.Get(),
                        pending.directSourceDesc);
                }
                if (!sourceSrv && pending.directSourceTexture)
                {
                    EnsurePresentResample(pending.directSourceDesc.Width, pending.directSourceDesc.Height, pending.directSourceDesc.Format);
                    m_context->CopyResource(m_presentResample.Get(), pending.directSourceTexture.Get());
                    sourceSrv = m_presentResampleSrv.Get();
                }
                if (!sourceSrv)
                {
                    throw std::runtime_error("Direct No-Upscaler source SRV is unavailable.");
                }

                ID3D11ShaderResourceView* srvs[] = { sourceSrv };
                ID3D11SamplerState* samplers[] = {
                    DebugViewOriginalActive() ? m_pointSampler.Get() : m_linearSampler.Get()
                };
                ID3D11ShaderResourceView* nullSrvs[] = { nullptr };
                ID3D11Buffer* nullBuffers[] = { nullptr };
                ID3D11RenderTargetView* nullTargets[] = { nullptr };

                ID3D11UnorderedAccessView* backBufferUav = CurrentSwapBackBufferUav(backBuffer);
                ID3D11ComputeShader* directCs = PresentComputeShaderForDirect();
                const bool computeAllowed = m_presentDispatchMode != PresentDispatchMode::Raster;

                if (computeAllowed && directCs && backBufferUav)
                {
                    auto scope = m_profiler.Stage("present-direct-compute-uav");
                    m_context->OMSetRenderTargets(1, nullTargets, nullptr);
                    ID3D11UnorderedAccessView* uavs[] = { backBufferUav };
                    UINT initialCounts[] = { 0 };
                    m_context->CSSetUnorderedAccessViews(0, 1, uavs, initialCounts);
                    m_context->CSSetShaderResources(0, 1, srvs);
                    m_context->CSSetSamplers(0, 1, samplers);
                    m_context->CSSetShader(directCs, nullptr, 0);
                    BindToneConstants();
                    UINT dispatchX = (outputWidth + 15) / 16;
                    UINT dispatchY = (outputHeight + 15) / 16;
                    m_context->Dispatch(dispatchX, dispatchY, 1);

                    ID3D11UnorderedAccessView* nullUavs[] = { nullptr };
                    m_context->CSSetUnorderedAccessViews(0, 1, nullUavs, initialCounts);
                    m_context->CSSetShaderResources(0, 1, nullSrvs);
                    m_context->CSSetConstantBuffers(0, 1, nullBuffers);
                    m_context->CSSetShader(nullptr, nullptr, 0);
                    MarkContextDirty();
                }
                else
                {
                    auto scope = m_profiler.Stage("present-direct-raster-rtv");
                    D3D11_VIEWPORT viewport{};
                    viewport.Width = static_cast<FLOAT>(outputWidth);
                    viewport.Height = static_cast<FLOAT>(outputHeight);
                    viewport.MaxDepth = 1.0f;

                    ID3D11RenderTargetView* targets[] = { backBufferRtv };
                    m_context->OMSetRenderTargets(1, targets, nullptr);
                    m_context->RSSetViewports(1, &viewport);
                    m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    m_context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
                    m_context->PSSetShader(PresentPixelShaderForMode(true), nullptr, 0);
                    m_context->PSSetShaderResources(0, 1, srvs);
                    m_context->PSSetSamplers(0, 1, samplers);
                    BindToneConstants();
                    m_context->Draw(3, 0);

                    m_context->PSSetShaderResources(0, 1, nullSrvs);
                    m_context->PSSetConstantBuffers(0, 1, nullBuffers);
                    m_context->OMSetRenderTargets(1, nullTargets, nullptr);
                    MarkContextDirty();
                }
            }

            if (pending.directCaptureLease)
            {
                SignalInputConsumed(pending.directCaptureLease);
                pending.directCaptureLease.reset();
            }
        }

        void PresentVideoOutputToSwapChain(
            UINT outputWidth,
            UINT outputHeight,
            ID3D11ShaderResourceView* sourceSrv,
            ComPtr<ID3D11Texture2D>& backBuffer)
        {
            if (!sourceSrv)
            {
                throw std::runtime_error("Presentation source SRV is missing.");
            }

            UpdateSourceDimensions(m_videoProcessorOutputWidth, m_videoProcessorOutputHeight);
            UpdateToneConstants(outputWidth, outputHeight);
            UpdateSourceCropConstants(
                0,
                0,
                m_videoProcessorOutputWidth,
                m_videoProcessorOutputHeight);
            UploadToneConstantsIfDirty();

            D3D11_VIEWPORT viewport{};
            viewport.Width = static_cast<FLOAT>(outputWidth);
            viewport.Height = static_cast<FLOAT>(outputHeight);
            viewport.MaxDepth = 1.0f;

            ID3D11ShaderResourceView* vpSrvs[] = { sourceSrv };
            ID3D11SamplerState* samplers[] = {
                DebugViewOriginalActive()
                    ? m_pointSampler.Get()
                    : m_linearSampler.Get() };
            const PresentationResamplePath resamplePath = SelectPresentationResamplePath(
                m_videoProcessorOutputWidth,
                m_videoProcessorOutputHeight,
                outputWidth,
                outputHeight);
            const bool sameSize = resamplePath == PresentationResamplePath::Direct;

            ID3D11ShaderResourceView* nullSrvs[] = { nullptr };
            ID3D11RenderTargetView* nullTargets[] = { nullptr };
            ID3D11Buffer* nullBuffers[] = { nullptr };

            ID3D11ShaderResourceView* presentationSrv = sourceSrv;
            const bool neutralNoUpscaler =
                m_activeUpscalerMode == UpscalerMode::NoUpscaler || m_activeUpscalerMode == UpscalerMode::BilinearComparison;
            if (!sameSize && !neutralNoUpscaler)
            {
                EnsurePresentResample(outputWidth, outputHeight);

                ID3D11RenderTargetView* resampleTargets[] = { m_presentResampleRtv.Get() };
                {
                    auto scope = m_profiler.Stage(
                        neutralNoUpscaler
                            ? "present-bilinear-resize"
                            : "present-magic-resample");
                    m_context->OMSetRenderTargets(1, resampleTargets, nullptr);
                    m_context->RSSetViewports(1, &viewport);
                    m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    m_context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
                    m_context->PSSetShader(
                        neutralNoUpscaler
                            ? m_bilinearResamplePixelShader.Get()
                            : m_magicResamplePixelShader.Get(),
                        nullptr,
                        0);
                    m_context->PSSetShaderResources(0, 1, vpSrvs);
                    m_context->PSSetSamplers(0, 1, samplers);
                    BindToneConstants();
                    m_context->Draw(3, 0);
                    MarkContextDirty();
                }

                m_context->PSSetShaderResources(0, 1, nullSrvs);
                m_context->PSSetConstantBuffers(0, 1, nullBuffers);
                m_context->OMSetRenderTargets(1, nullTargets, nullptr);
                presentationSrv = m_presentResampleSrv.Get();
                UpdateSourceCropConstants(0, 0, outputWidth, outputHeight);
                UploadToneConstantsIfDirty();
            }

            ID3D11RenderTargetView* backBufferRtv = CurrentSwapBackBufferRtv(backBuffer);
            ID3D11RenderTargetView* targets[] = { backBufferRtv };
            ID3D11ShaderResourceView* srvs[] = { presentationSrv };
            const bool directPresentation =
                resamplePath == PresentationResamplePath::Direct ||
                neutralNoUpscaler;
            const char* presentationStage = sameSize
                ? "present-direct-same-size"
                : (neutralNoUpscaler
                    ? "present-final-bilinear"
                    : "present-final-magic-sharp");

            ID3D11UnorderedAccessView* backBufferUav = directPresentation ? CurrentSwapBackBufferUav(backBuffer) : nullptr;
            ID3D11ComputeShader* directCs = directPresentation ? PresentComputeShaderForDirect() : nullptr;

            const bool computeAllowed = m_presentDispatchMode != PresentDispatchMode::Raster;
            const bool computeRequested = m_presentDispatchMode == PresentDispatchMode::Compute;

            if (computeAllowed && directCs && backBufferUav)
            {
                auto scope = m_profiler.Stage("present-direct-compute-uav");
                m_context->OMSetRenderTargets(1, nullTargets, nullptr);
                ID3D11UnorderedAccessView* uavs[] = { backBufferUav };
                UINT initialCounts[] = { 0 };
                m_context->CSSetUnorderedAccessViews(0, 1, uavs, initialCounts);
                m_context->CSSetShaderResources(0, 1, srvs);
                m_context->CSSetSamplers(0, 1, samplers);
                m_context->CSSetShader(directCs, nullptr, 0);
                BindToneConstants();
                UINT dispatchX = (outputWidth + 15) / 16;
                UINT dispatchY = (outputHeight + 15) / 16;
                m_context->Dispatch(dispatchX, dispatchY, 1);

                ID3D11UnorderedAccessView* nullUavs[] = { nullptr };
                m_context->CSSetUnorderedAccessViews(0, 1, nullUavs, initialCounts);
                m_context->CSSetShaderResources(0, 1, nullSrvs);
                m_context->CSSetConstantBuffers(0, 1, nullBuffers);
                m_context->CSSetShader(nullptr, nullptr, 0);
                MarkContextDirty();
            }
            else
            {
                if (computeRequested && (!directCs || !backBufferUav))
                {
                    static bool warned = false;
                    if (!warned)
                    {
                        warned = true;
                        std::cerr << "Warning: --present-mode compute requested, but direct CS or UAV is unavailable (falling back to raster).\n";
                    }
                }
                auto scope = m_profiler.Stage(presentationStage);
                m_context->OMSetRenderTargets(1, targets, nullptr);
                m_context->RSSetViewports(1, &viewport);
                m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                m_context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
                m_context->PSSetShader(PresentPixelShaderForMode(directPresentation), nullptr, 0);
                m_context->PSSetShaderResources(0, 1, srvs);
                m_context->PSSetSamplers(0, 1, samplers);
                BindToneConstants();
                m_context->Draw(3, 0);

                m_context->PSSetShaderResources(0, 1, nullSrvs);
                m_context->PSSetConstantBuffers(0, 1, nullBuffers);
                m_context->OMSetRenderTargets(1, nullTargets, nullptr);
                MarkContextDirty();
            }
        }

        void SaveQualityDump(
            const std::filesystem::path& directory,
            const FrameInfo& frame,
            const D3D11_TEXTURE2D_DESC& sourceDesc,
            ID3D11Texture2D* vpOutput,
            ID3D11Texture2D* backBuffer,
            uint64_t frameNumber,
            const D3D11RenderResult& result)
        {
            std::filesystem::create_directories(directory);
            SaveTexture2DAsBmp(m_device.Get(), m_context.Get(), frame.d3d11Texture.Get(), sourceDesc, directory / L"01_input_frame.bmp");

            D3D11_TEXTURE2D_DESC workingDesc{};
            m_workingInput->GetDesc(&workingDesc);
            SaveTexture2DAsBmp(m_device.Get(), m_context.Get(), m_workingInput.Get(), workingDesc, directory / L"02_d3d11_working_input.bmp");

            if (vpOutput)
            {
                D3D11_TEXTURE2D_DESC vpDesc{};
                vpOutput->GetDesc(&vpDesc);
                SaveTexture2DAsBmp(m_device.Get(), m_context.Get(), vpOutput, vpDesc, directory / L"03_upscaled_2x.bmp");
            }

            D3D11_TEXTURE2D_DESC backDesc{};
            backBuffer->GetDesc(&backDesc);
            SaveTexture2DAsBmp(m_device.Get(), m_context.Get(), backBuffer, backDesc, directory / L"04_final.bmp");

            std::ofstream manifest(directory / L"manifest.txt");
            if (manifest)
            {
                manifest << "AESS quality dump\n";
                manifest << "frame=" << frameNumber << "\n";
                manifest << "pipeline=unified-d3d11\n";
                manifest << "upscaler=" << Narrow(std::wstring(InfoForUpscaler(result.upscalerMode).displayName)) << "\n";
                manifest << "selectedUpscaler=" << Narrow(std::wstring(InfoForUpscaler(result.selectedUpscalerMode).displayName)) << "\n";
                manifest << "upscalerBypass=" << (result.upscalerBypass ? "yes" : "no") << "\n";
                manifest << "upscalerStage=" << (result.shaderUpscaler ? "shader" : "d3d11-video-processor") << "\n";
                manifest << "resampler=magic-kernel-sharp\n";
                manifest << "outputMode=" << InfoForOutputMode(result.outputMode).logName << "\n";
                manifest << "inputSizing=captured-window\n";
                manifest << "inputColorSpace=" << static_cast<uint32_t>(result.inputColorSpace) << "\n";
                manifest << "capturedInputFormat=" << FormatName(sourceDesc.Format) << "\n";
                manifest << "inputColorProvenance="
                    << CaptureColorProvenanceName(result.inputColorProvenance)
                    << "\n";
                manifest << "modelTransfer=" << result.tone.sourceTransfer << "\n";
                manifest << "modelPrimaries=" << result.tone.sourcePrimaries << "\n";
                manifest << "workingEncoding="
                    << WorkingColorEncodingName(result.tone.sourceTransfer)
                    << "\n";
                manifest << "workingInput=" << result.workingWidth << "x" << result.workingHeight << " " << FormatName(workingDesc.Format) << "\n";
                manifest << "sdrWhiteNits=" << result.tone.sdrWhiteNits << "\n";
                manifest << "sdrWhiteScale=" << result.tone.sdrWhiteScale << "\n";
                manifest << "internalOutputScale=" << result.internalOutputScale << "\n";
                manifest << "internalOutput=" << result.videoProcessorOutputWidth << "x" << result.videoProcessorOutputHeight << "\n";
                manifest << "upscaled2xOutput=" << result.videoProcessorOutputWidth << "x" << result.videoProcessorOutputHeight << "\n";
                manifest << "upscaledOutputFormat=" << FormatName(result.videoProcessorOutputFormat) << "\n";
                manifest << "output=" << backDesc.Width << "x" << backDesc.Height << " " << FormatName(backDesc.Format) << "\n";
                manifest << "note=Unified D3D11 uses captured-window dimensions, GPU textures, direct same-size presentation, and Magic Kernel plus Sharp for required resampling.\n";
            }
            std::wcout << L"Quality dump written to " << directory.wstring() << L"\n";
        }

        HWND m_hwnd = nullptr;
        ComPtr<ID3D11Device> m_device;
        ComPtr<ID3D11DeviceContext> m_context;
        ComPtr<ID3D11Device5> m_device5;
        ComPtr<ID3D11DeviceContext4> m_context4;
        ComPtr<ID3D11Fence> m_completionFence;
        uint64_t m_completionFenceValue = 0;
        ComPtr<ID3D11Fence> m_inputConsumedFence;
        uint64_t m_inputConsumedFenceValue = 0;
        winrt::handle m_inputConsumedEvent;
        uint64_t m_inputConsumedEventValue = 0;
        ComPtr<ID3D11Query> m_inputDrainQuery;
        aess::capture::CaptureRetirementQueue<CaptureLease> m_activeCaptureLeases;
        ComPtr<ID3D11VideoDevice> m_videoDevice;
        ComPtr<ID3D11VideoContext> m_videoContext;
        ComPtr<ID3D11VideoContext1> m_videoContext1;
        ComPtr<IDXGIFactory2> m_factory;
        ComPtr<IDXGISwapChain3> m_swapChain;
        ComPtr<IDXGISwapChainMedia> m_swapChainMedia;
        HANDLE m_frameLatencyWaitable = nullptr;
        bool m_waitingForPresentationSlot = false;
        bool m_frameLatencyWakeGranted = false;
        ComPtr<IDCompositionDevice> m_compositionDevice;
        ComPtr<IDCompositionTarget> m_compositionTarget;
        ComPtr<IDCompositionVisual> m_compositionVisual;
        ComPtr<IDCompositionScaleTransform> m_compositionScaleTransform;
        aess::presentation::FrameTelemetryWriter m_telemetry;
        std::array<PendingPresentTiming, 64> m_pendingPresentTimings{};
        D3D11GpuProfiler m_profiler;
        ImGuiContext* m_fpsCounterContext = nullptr;
        bool m_fpsCounterEnabled = false;
        bool m_gsyncCompatibilityEligible = false;
        FramePacingConfig m_framePacingConfig{};
        aess::presentation::FramePacer m_framePacer;

















        DWORD m_windowThreadId = 0;
        std::atomic<SwapChainLifecycleOperation> m_swapChainLifecycleOperation{
            SwapChainLifecycleOperation::None };
        std::atomic<HRESULT> m_swapChainLifecycleResult{ E_PENDING };
        std::atomic<UINT> m_requestedSwapChainWidth{ 0 };
        std::atomic<UINT> m_requestedSwapChainHeight{ 0 };
        uint64_t m_timingRecordId = 0;
        uint64_t m_lastTimingSourceOrdinal = 0;
        int64_t m_lastTimingSourceQpc = 0;
        int64_t m_lastObservedSourceQpc = 0;
        uint64_t m_lastObservedSourceOrdinal = 0;
        bool m_pipelineAdmissionBlockedSinceLastFrame = false;
        std::array<
            uint64_t,
            static_cast<size_t>(FrameDropReason::SourceTimingJournalOverflow) + 1>
            m_frameDropCounts{};
        bool m_fpsSampleInitialized = false;
        bool m_actualPresentStats = false;
        bool m_haveSuccessfulFrameStatistics = false;
        bool m_havePresentationMode = false;
        DXGI_FRAME_PRESENTATION_MODE m_presentationMode = DXGI_FRAME_PRESENTATION_MODE_COMPOSED;
        bool m_haveMediaPresentCount = false;
        bool m_haveOutputObservation = false;
        bool m_haveOutputQpc = false;
        uint32_t m_lastMediaPresentCount = 0;
        int64_t m_lastOutputQpc = 0;
        uint64_t m_inferredFrameCount = 0;
        uint64_t m_acceptedPresentCount = 0;
        uint64_t m_outputPresentCount = 0;
        uint64_t m_captureArrivalBase = 0;
        uint64_t m_captureBacklog = 0;
        uint64_t m_sampleCaptureCount = 0;
        uint64_t m_sampleInferenceCount = 0;
        uint64_t m_sampleAcceptedPresentCount = 0;
        uint64_t m_sampleOutputCount = 0;
        std::chrono::steady_clock::time_point m_fpsSampleTime{};
        std::chrono::steady_clock::time_point m_lastOutputObservation{};
        std::array<double, 120> m_outputIntervals{};
        size_t m_outputIntervalCount = 0;
        size_t m_outputIntervalIndex = 0;
        bool m_haveResidualCorrelation = false;
        int64_t m_lastResidualSourceQpc = 0;
        uint64_t m_lastResidualSourceOrdinal = 0;
        int64_t m_lastResidualSyncQpc = 0;
        std::array<double, 120> m_addedIntervalResiduals{};
        size_t m_addedIntervalResidualCount = 0;
        size_t m_addedIntervalResidualIndex = 0;
        double m_addedIntervalResidualP95Ms = 0.0;
        std::array<double, 120> m_actualDisplayAges{};
        size_t m_actualDisplayAgeCount = 0;
        size_t m_actualDisplayAgeIndex = 0;
        double m_actualDisplayAgeP95Ms = 0.0;
        double m_captureFps = 0.0;
        double m_inferenceFps = 0.0;
        double m_acceptedPresentFps = 0.0;
        double m_outputFps = 0.0;
        double m_outputFrameMs = 0.0;
        double m_outputP95Ms = 0.0;
        double m_outputJitterMs = 0.0;
        std::array<ComPtr<ID3D11Texture2D>, kFrameCount> m_swapBackBuffers;
        std::array<ComPtr<ID3D11RenderTargetView>, kFrameCount> m_swapBackBufferRtvs;
        std::array<ComPtr<ID3D11UnorderedAccessView>, kFrameCount> m_swapBackBufferUavs;
        ComPtr<ID3D11Texture2D> m_shaderSource;
        ComPtr<ID3D11ShaderResourceView> m_shaderSourceSrv;
        aess::render::SourceViewCache m_sourceViewCache;
        ComPtr<ID3D11Texture2D> m_workingInput;
        ComPtr<ID3D11RenderTargetView> m_workingInputRtv;
        ComPtr<ID3D11ShaderResourceView> m_workingInputSrv;
        ComPtr<ID3D11UnorderedAccessView> m_workingInputUav;
        ComPtr<ID3D11Texture2D> m_videoProcessorOutput;
        ComPtr<ID3D11RenderTargetView> m_videoProcessorOutputRtv;
        ComPtr<ID3D11ShaderResourceView> m_videoProcessorOutputSrv;
        ComPtr<ID3D11UnorderedAccessView> m_videoProcessorOutputUav;
        std::array<SynchronousReadyFrame, 2> m_synchronousPipelineSlots;
        SynchronousRouteSignature m_synchronousRouteSignature{};
        bool m_synchronousRouteSignatureValid = false;
        D3D11RenderResult m_synchronousLastPresentedResult{};
        uint64_t m_synchronousSubmitOrder = 0;
        std::unique_ptr<aess::neural::CudaRuntime> m_s2d32Runtime;
        std::array<S2D32PipelineSlot, 2> m_s2d32PipelineSlots;
        UINT m_s2d32PipelineWidth = 0;
        UINT m_s2d32PipelineHeight = 0;
        aess::neural::RuntimePrecision m_s2d32PipelinePrecision = aess::neural::RuntimePrecision::Fp8;
        aess::neural::ModelKind m_s2d32PipelineModelKind =
            aess::neural::ModelKind::AdaptiveFilter32;
        uint64_t m_s2d32SubmitOrder = 0;
        bool m_s2d32HasPresented = false;
        D3D11RenderResult m_s2d32LastPresentedResult{};
        aess::neural::FrameTimings m_s2d32LastTimings{};
        bool m_s2d32StartupReported = false;
        ComPtr<ID3D11Texture2D> m_upscaleScratch;
        ComPtr<ID3D11RenderTargetView> m_upscaleScratchRtv;
        ComPtr<ID3D11ShaderResourceView> m_upscaleScratchSrv;
        ComPtr<ID3D11UnorderedAccessView> m_upscaleScratchUav;
        // Cached upload staging texture (CPU-write).  Reallocated only when
        // dimensions change to avoid per-frame allocation of a staging texture.
        ComPtr<ID3D11Texture2D> m_uploadStaging;
        UINT m_uploadStagingWidth = 0;
        UINT m_uploadStagingHeight = 0;
        DXGI_FORMAT m_uploadStagingFormat = DXGI_FORMAT_UNKNOWN;
        ComPtr<ID3D11Texture2D> m_presentResample;
        ComPtr<ID3D11RenderTargetView> m_presentResampleRtv;
        ComPtr<ID3D11ShaderResourceView> m_presentResampleSrv;
        ComPtr<ID3D11VertexShader> m_vertexShader;
        ComPtr<ID3D11PixelShader> m_pixelShader;
        ComPtr<ID3D11PixelShader> m_workingInputMagicPixelShader;
        ComPtr<ID3D11PixelShader> m_magicSharpEncodedPixelShader;
        ComPtr<ID3D11PixelShader> m_copySameSizePixelShader;
        ComPtr<ID3D11PixelShader> m_bilinearResamplePixelShader;
        ComPtr<ID3D11PixelShader> m_magicResamplePixelShader;
        ComPtr<ID3D11PixelShader> m_presentSdrPixelShader;
        ComPtr<ID3D11PixelShader> m_presentDirectSdrPixelShader;
        ComPtr<ID3D11ComputeShader> m_csPresentDirectSdr;
        ComPtr<ID3D11ComputeShader> m_csPrepareWorkingInput;
        ComPtr<ID3D11SamplerState> m_pointSampler;
        ComPtr<ID3D11SamplerState> m_linearSampler;
        ComPtr<ID3D11Buffer> m_toneBuffer;
        aess::render::ConstantBufferCache<ToneConstants> m_toneBufferCache;
        ComPtr<ID3D11VideoProcessorInputView> m_cachedVpInputView;
        ComPtr<ID3D11VideoProcessorOutputView> m_cachedVpOutputView;
        ComPtr<ID3D11Texture2D> m_cachedVpOutputTexture;
        ComPtr<ID3D11VideoProcessorEnumerator> m_vpEnum;
        ComPtr<ID3D11VideoProcessor> m_videoProcessor;
        ToneConstants m_toneConstants{};
        bool m_toneConstantsDirty = false;
        UINT m_sourceWidth = 0;
        UINT m_sourceHeight = 0;
        DXGI_FORMAT m_sourceFormat = DXGI_FORMAT_UNKNOWN;
        UINT m_workingWidth = 0;
        UINT m_workingHeight = 0;
        DXGI_FORMAT m_workingFormat = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT m_workingInputFormat = DXGI_FORMAT_UNKNOWN;
        UINT m_swapWidth = 0;
        UINT m_swapHeight = 0;
        DXGI_FORMAT m_swapFormat = DXGI_FORMAT_UNKNOWN;
        DXGI_COLOR_SPACE_TYPE m_outputColorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        UINT m_vpInputWidth = 0;
        UINT m_vpInputHeight = 0;
        UINT m_vpOutputWidth = 0;
        UINT m_vpOutputHeight = 0;
        DXGI_FORMAT m_vpOutputFormat = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT m_vpWorkingFormat = DXGI_FORMAT_UNKNOWN;
        UINT m_videoProcessorOutputWidth = 0;
        UINT m_videoProcessorOutputHeight = 0;
        DXGI_FORMAT m_videoProcessorOutputFormat = DXGI_FORMAT_UNKNOWN;
        UINT m_presentResampleWidth = 0;
        UINT m_presentResampleHeight = 0;
        DXGI_FORMAT m_presentResampleFormat = DXGI_FORMAT_UNKNOWN;
        UINT m_upscaleScratchWidth = 0;
        UINT m_upscaleScratchHeight = 0;
        DXGI_FORMAT m_upscaleScratchFormat = DXGI_FORMAT_UNKNOWN;
        UINT m_outputSeq = 0;
        UpscalerMode m_upscalerMode = UpscalerMode::NoUpscaler;
        UpscalerMode m_activeUpscalerMode = UpscalerMode::NoUpscaler;
        DebugViewMode m_debugViewMode = DebugViewMode::Upscaled;
        PresentDispatchMode m_presentDispatchMode = PresentDispatchMode::Auto;
        bool m_contextDirty = false;
        SourceColorMode m_sourceColorMode = SourceColorMode::SdrGamma;
        OutputMode m_outputMode = OutputMode::SdrShaderBridge;
        float m_requestedSdrWhiteNits = 500.0f;
        bool m_upscalerBypass = false;
        ComparisonFilter m_comparisonFilter = ComparisonFilter::Nearest;
        bool m_debugViewActive = false;
        bool m_presentStatsCapabilityReported = false;
        bool m_tearingSupported = false;
        double m_lastFlipMeterWaitMs = 0.0;
        uint64_t m_flipMeterTimeoutCount = 0;
        double m_flipMeterWaitTotalMs = 0.0;
        uint64_t m_flipMeterWaitCount = 0;
        double m_s2d32BlockingWaitTotalMs = 0.0;
        uint64_t m_s2d32BlockingWaitCount = 0;
        uint64_t m_presentQueueBusyCount = 0;
    };

