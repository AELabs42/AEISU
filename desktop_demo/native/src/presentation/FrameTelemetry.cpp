#include "FrameTelemetry.h"

#if defined(_MSC_VER)
// The producer and consumer cursors are intentionally cache-line separated.
// MSVC reports the resulting explicit alignment padding as C4324.
#pragma warning(disable:4324)
#pragma comment(lib, "Synchronization.lib")
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <limits>
#include <system_error>
#include <thread>

namespace aess::presentation
{
    namespace
    {
        constexpr size_t kFrameQueueCapacity = 4096;
        constexpr size_t kGpuStageQueueCapacity = 16384;
        constexpr size_t kFrameDrainBatch = 1024;
        constexpr size_t kGpuStageDrainBatch = 4096;
        constexpr DWORD kWriterFlushPeriodMs = 1000;

        template <typename Value, size_t Capacity>
        class BoundedSpscQueue
        {
            static_assert(Capacity != 0 && (Capacity & (Capacity - 1)) == 0);
            static_assert(std::is_trivially_copyable_v<Value>);

        public:
            bool TryPush(const Value& value, bool& wasEmpty) noexcept
            {
                const uint64_t write = m_write.load(std::memory_order_relaxed);
                const uint64_t read = m_read.load(std::memory_order_acquire);
                if (write - read >= Capacity)
                {
                    wasEmpty = false;
                    return false;
                }
                wasEmpty = write == read;
                m_values[static_cast<size_t>(write) & (Capacity - 1)] = value;
                m_write.store(write + 1, std::memory_order_release);
                return true;
            }

            bool TryPop(Value& value) noexcept
            {
                const uint64_t read = m_read.load(std::memory_order_relaxed);
                const uint64_t write = m_write.load(std::memory_order_acquire);
                if (read == write)
                {
                    return false;
                }
                value = m_values[static_cast<size_t>(read) & (Capacity - 1)];
                m_read.store(read + 1, std::memory_order_release);
                return true;
            }

            bool Empty() const noexcept
            {
                return m_read.load(std::memory_order_acquire) ==
                    m_write.load(std::memory_order_acquire);
            }

            void Reset() noexcept
            {
                m_read.store(0, std::memory_order_relaxed);
                m_write.store(0, std::memory_order_relaxed);
            }

        private:
            std::array<Value, Capacity> m_values{};
            alignas(64) std::atomic<uint64_t> m_write{ 0 };
            alignas(64) std::atomic<uint64_t> m_read{ 0 };
        };

        template <size_t Count>
        void WriteCsvText(FILE* file, const std::array<char, Count>& value) noexcept
        {
            std::fputc('"', file);
            for (size_t index = 0; index < Count && value[index] != '\0'; ++index)
            {
                if (value[index] == '"')
                {
                    std::fputc('"', file);
                }
                std::fputc(static_cast<unsigned char>(value[index]), file);
            }
            std::fputc('"', file);
        }

        std::string ReadWholeFile(const std::filesystem::path& path)
        {
            std::ifstream input(path, std::ios::binary);
            if (!input)
            {
                return {};
            }
            return std::string(
                std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>());
        }
    }

    const char* FrameDropReasonName(FrameDropReason reason) noexcept
    {
        switch (reason)
        {
        case FrameDropReason::None: return "none";
        case FrameDropReason::CaptureSuperseded: return "capture-superseded";
        case FrameDropReason::PipelineSaturated: return "pipeline-saturated";
        case FrameDropReason::ReadySuperseded: return "ready-superseded";
        case FrameDropReason::StaleOverTwoPeriods: return "stale-over-two-periods";
        case FrameDropReason::TimestampRegression: return "timestamp-regression";
        case FrameDropReason::ResizeReset: return "resize-reset";
        case FrameDropReason::FocusReset: return "focus-reset";
        case FrameDropReason::UpscalerReset: return "upscaler-reset";
        case FrameDropReason::PresentationReset: return "presentation-reset";
        case FrameDropReason::PresentRejected: return "present-rejected";
        case FrameDropReason::DxgiSuperseded: return "dxgi-superseded";
        case FrameDropReason::ShutdownDiscard: return "shutdown-discard";
        case FrameDropReason::SourceTimingJournalOverflow: return "source-timing-journal-overflow";
        default: return "unknown";
        }
    }

    class FrameTelemetryWriter::Impl
    {
    public:
        ~Impl()
        {
            Stop();
        }

        bool Start(
            const std::filesystem::path& stageCsvPath,
            int64_t qpcFrequency,
            std::string& diagnostic) noexcept
        {
            Stop();
            try
            {
                if (stageCsvPath.empty())
                {
                    diagnostic = "frame telemetry requires a stage CSV path";
                    return false;
                }
                if (qpcFrequency <= 0)
                {
                    diagnostic = "frame telemetry requires a positive QPC frequency";
                    return false;
                }

                m_stageCsvPath = stageCsvPath;
                m_pacingCsvPath = FrameTelemetryWriter::PacingPathForStagePath(stageCsvPath);
                m_qpcFrequency = qpcFrequency;
                m_frameQueue.Reset();
                m_gpuStageQueue.Reset();
                m_droppedFrames.store(0, std::memory_order_relaxed);
                m_droppedGpuStages.store(0, std::memory_order_relaxed);
                m_writtenFrames.store(0, std::memory_order_relaxed);
                m_writtenGpuStages.store(0, std::memory_order_relaxed);
                m_writerError.store(ERROR_SUCCESS, std::memory_order_relaxed);
                m_lowPriorityApplied.store(false, std::memory_order_relaxed);
                m_stopping.store(false, std::memory_order_relaxed);
                m_wakeSequence.store(0, std::memory_order_relaxed);

                std::error_code directoryError;
                if (!m_stageCsvPath.parent_path().empty())
                {
                    std::filesystem::create_directories(
                        m_stageCsvPath.parent_path(),
                        directoryError);
                }
                if (!directoryError && !m_pacingCsvPath.parent_path().empty())
                {
                    std::filesystem::create_directories(
                        m_pacingCsvPath.parent_path(),
                        directoryError);
                }
                if (directoryError)
                {
                    diagnostic = "create frame telemetry output directory failed: " +
                        directoryError.message();
                    return false;
                }

                errno_t openError = _wfopen_s(
                    &m_stageFile,
                    m_stageCsvPath.c_str(),
                    L"wb");
                if (openError != 0 || !m_stageFile)
                {
                    diagnostic = "open GPU stage CSV failed with errno " +
                        std::to_string(static_cast<int>(openError));
                    CloseFiles();
                    return false;
                }
                openError = _wfopen_s(
                    &m_pacingFile,
                    m_pacingCsvPath.c_str(),
                    L"wb");
                if (openError != 0 || !m_pacingFile)
                {
                    diagnostic = "open frame pacing CSV failed with errno " +
                        std::to_string(static_cast<int>(openError));
                    CloseFiles();
                    return false;
                }

                if (std::setvbuf(
                        m_stageFile,
                        m_stageFileBuffer.data(),
                        _IOFBF,
                        m_stageFileBuffer.size()) != 0 ||
                    std::setvbuf(
                        m_pacingFile,
                        m_pacingFileBuffer.data(),
                        _IOFBF,
                        m_pacingFileBuffer.size()) != 0)
                {
                    diagnostic = "configure buffered frame telemetry files failed";
                    CloseFiles();
                    return false;
                }

                WriteHeaders();
                if (FilesFailed())
                {
                    diagnostic = "write frame telemetry CSV headers failed";
                    CloseFiles();
                    return false;
                }

                m_enabled.store(true, std::memory_order_release);
                try
                {
                    m_thread = std::thread([this] { WriterMain(); });
                }
                catch (...)
                {
                    m_enabled.store(false, std::memory_order_release);
                    CloseFiles();
                    throw;
                }

                diagnostic = "frame telemetry writer active: stage=" +
                    m_stageCsvPath.string() + " pacing=" + m_pacingCsvPath.string();
                return true;
            }
            catch (const std::exception& error)
            {
                m_enabled.store(false, std::memory_order_release);
                CloseFiles();
                diagnostic = error.what();
                return false;
            }
            catch (...)
            {
                m_enabled.store(false, std::memory_order_release);
                CloseFiles();
                diagnostic = "unknown frame telemetry startup failure";
                return false;
            }
        }

        void Stop() noexcept
        {
            m_enabled.store(false, std::memory_order_release);
            m_stopping.store(true, std::memory_order_release);
            m_wakeSequence.fetch_add(1, std::memory_order_release);
            WakeByAddressSingle(&m_wakeSequence);
            if (m_thread.joinable())
            {
                m_thread.join();
            }
            CloseFiles();
        }

        bool TryEnqueueFrame(const FrameTimingRecord& source) noexcept
        {
            if (!m_enabled.load(std::memory_order_acquire))
            {
                return false;
            }
            FrameTimingRecord record = source;
            record.writerOverflowBefore = m_droppedFrames.load(std::memory_order_relaxed);
            bool wasEmpty = false;
            if (!m_frameQueue.TryPush(record, wasEmpty))
            {
                m_droppedFrames.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            NotifyWriter();
            return true;
        }

        bool TryEnqueueGpuStage(const GpuStageTimingRecord& record) noexcept
        {
            if (!m_enabled.load(std::memory_order_acquire))
            {
                return false;
            }
            bool wasEmpty = false;
            if (!m_gpuStageQueue.TryPush(record, wasEmpty))
            {
                m_droppedGpuStages.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            NotifyWriter();
            return true;
        }

        void NotifyWriter() noexcept
        {
            // The consumer can drain the last item after TryPush reads its
            // cursor but before publication. A stale "not empty" observation
            // must not suppress this wake and strand a record for a second.
            // This path exists only with explicit CSV logging enabled.
            m_wakeSequence.fetch_add(1, std::memory_order_release);
            WakeByAddressSingle(&m_wakeSequence);
        }

        void WriterMain() noexcept
        {
            const bool background = SetThreadPriority(
                GetCurrentThread(),
                THREAD_MODE_BACKGROUND_BEGIN) != FALSE;
            bool lowPriorityApplied = background;
            if (!lowPriorityApplied)
            {
                lowPriorityApplied = SetThreadPriority(
                    GetCurrentThread(),
                    THREAD_PRIORITY_LOWEST) != FALSE;
            }
            m_lowPriorityApplied.store(lowPriorityApplied, std::memory_order_release);

            ULONGLONG nextFlush = GetTickCount64() + kWriterFlushPeriodMs;
            for (;;)
            {
                DrainBatch();
                const ULONGLONG now = GetTickCount64();
                if (now >= nextFlush)
                {
                    // A continuous stream must not postpone flushing forever
                    // by restarting an idle-only timeout on every frame.
                    FlushFiles();
                    nextFlush = now + kWriterFlushPeriodMs;
                }
                if (m_stopping.load(std::memory_order_acquire))
                {
                    if (QueuesEmpty())
                    {
                        break;
                    }
                    continue;
                }

                if (!QueuesEmpty())
                {
                    continue;
                }

                // WaitOnAddress avoids a kernel event transition whenever the
                // writer has already drained a frame-sized burst.  Loading the
                // sequence before the final queue check closes the lost-wake
                // window: a producer that publishes between the check and the
                // wait changes the value, so WaitOnAddress returns immediately.
                uint32_t observedWake =
                    m_wakeSequence.load(std::memory_order_acquire);
                if (!QueuesEmpty() || m_stopping.load(std::memory_order_acquire))
                {
                    continue;
                }

                const ULONGLONG beforeWait = GetTickCount64();
                const DWORD waitMs = beforeWait < nextFlush
                    ? static_cast<DWORD>(nextFlush - beforeWait)
                    : 0;
                if (!WaitOnAddress(
                        &m_wakeSequence,
                        &observedWake,
                        sizeof(observedWake),
                        waitMs))
                {
                    const DWORD error = GetLastError();
                    if (error != ERROR_TIMEOUT)
                    {
                        RecordWriterError(error);
                        m_stopping.store(true, std::memory_order_release);
                    }
                }
            }

            DrainAll();
            FlushFiles();
            if (background)
            {
                SetThreadPriority(GetCurrentThread(), THREAD_MODE_BACKGROUND_END);
            }
        }

        void DrainBatch() noexcept
        {
            FrameTimingRecord frame{};
            for (size_t count = 0;
                 count < kFrameDrainBatch && m_frameQueue.TryPop(frame);
                 ++count)
            {
                WriteFrame(frame);
                m_writtenFrames.fetch_add(1, std::memory_order_relaxed);
            }

            GpuStageTimingRecord stage{};
            for (size_t count = 0;
                 count < kGpuStageDrainBatch && m_gpuStageQueue.TryPop(stage);
                 ++count)
            {
                WriteGpuStage(stage);
                m_writtenGpuStages.fetch_add(1, std::memory_order_relaxed);
            }
        }

        void DrainAll() noexcept
        {
            while (!QueuesEmpty())
            {
                DrainBatch();
            }
        }

        bool QueuesEmpty() const noexcept
        {
            return m_frameQueue.Empty() && m_gpuStageQueue.Empty();
        }

        void WriteHeaders() noexcept
        {
            std::fputs(
                "frame,source_version,stage,gpu_ms,cpu_render_ms,cpu_acquire_ms,"
                "cpu_preprocess_ms,cpu_upscale_ms,cpu_present_setup_ms,cpu_present_call_ms,"
                "skipped_frames,upscaler,selected_upscaler,input,internal_output,output_format,dump_frame\n",
                m_stageFile);
            std::fputs(
                "schema_version,qpc_frequency,record_id,frame,source_version,source_present_ordinal,"
                "capture_arrival_count,effect_submit_ordinal,source_present_qpc,capture_ready_qpc,"
                "capture_acquire_begin_qpc,capture_acquire_end_qpc,effect_submit_qpc,effect_complete_qpc,"
                "ready_queue_entry_qpc,target_deadline_qpc,latency_gate_begin_qpc,latency_gate_end_qpc,"
                "present_call_begin_qpc,present_call_end_qpc,actual_sync_qpc,dropped_source_presents_before,"
                "source_timing_journal_overflow_count,added_interval_residual_ms,"
                "dxgi_present_count,input_width,input_height,ready_depth_at_entry,ready_depth_before_present,"
                "ready_depth_after_present,queue_target,upscaler_mode,pacing_mode,presentation_mode,"
                "drop_reason_id,drop_reason,flags,writer_overflow_before\n",
                m_pacingFile);
        }

        void WriteFrame(const FrameTimingRecord& record) noexcept
        {
            const int result = std::fprintf(
                m_pacingFile,
                "%u,%lld,%llu,%llu,%llu,%llu,%llu,%llu,"
                "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,"
                "%llu,%llu,%.6f,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,\"%s\",%u,%llu\n",
                static_cast<unsigned>(kFrameTimingSchemaVersion),
                static_cast<long long>(m_qpcFrequency),
                static_cast<unsigned long long>(record.recordId),
                static_cast<unsigned long long>(record.frameNumber),
                static_cast<unsigned long long>(record.sourceVersion),
                static_cast<unsigned long long>(record.sourcePresentOrdinal),
                static_cast<unsigned long long>(record.captureArrivalCount),
                static_cast<unsigned long long>(record.effectSubmitOrdinal),
                static_cast<long long>(record.sourcePresentQpc),
                static_cast<long long>(record.captureReadyQpc),
                static_cast<long long>(record.captureAcquireBeginQpc),
                static_cast<long long>(record.captureAcquireEndQpc),
                static_cast<long long>(record.effectSubmitQpc),
                static_cast<long long>(record.effectCompleteQpc),
                static_cast<long long>(record.readyQueueEntryQpc),
                static_cast<long long>(record.targetDeadlineQpc),
                static_cast<long long>(record.latencyGateBeginQpc),
                static_cast<long long>(record.latencyGateEndQpc),
                static_cast<long long>(record.presentCallBeginQpc),
                static_cast<long long>(record.presentCallEndQpc),
                static_cast<long long>(record.actualSyncQpc),
                static_cast<unsigned long long>(record.droppedSourcePresentsBefore),
                static_cast<unsigned long long>(record.sourceTimingJournalOverflowCount),
                record.addedIntervalResidualMs,
                static_cast<unsigned>(record.dxgiPresentCount),
                static_cast<unsigned>(record.inputWidth),
                static_cast<unsigned>(record.inputHeight),
                static_cast<unsigned>(record.readyDepthAtEntry),
                static_cast<unsigned>(record.readyDepthBeforePresent),
                static_cast<unsigned>(record.readyDepthAfterPresent),
                static_cast<unsigned>(record.queueTarget),
                static_cast<unsigned>(record.upscalerMode),
                static_cast<unsigned>(record.pacingMode),
                static_cast<unsigned>(record.presentationMode),
                static_cast<unsigned>(record.dropReason),
                FrameDropReasonName(record.dropReason),
                static_cast<unsigned>(record.flags),
                static_cast<unsigned long long>(record.writerOverflowBefore));
            if (result < 0)
            {
                RecordWriterError(ERROR_WRITE_FAULT);
            }
        }

        void WriteGpuStage(const GpuStageTimingRecord& record) noexcept
        {
            if (std::fprintf(
                    m_stageFile,
                    "%llu,%llu,",
                    static_cast<unsigned long long>(record.frameNumber),
                    static_cast<unsigned long long>(record.sourceVersion)) < 0)
            {
                RecordWriterError(ERROR_WRITE_FAULT);
                return;
            }
            WriteCsvText(m_stageFile, record.stage);
            std::fprintf(
                m_stageFile,
                ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%llu,",
                record.gpuMs,
                record.cpuRenderMs,
                record.cpuAcquireMs,
                record.cpuPreprocessMs,
                record.cpuUpscaleMs,
                record.cpuPresentSetupMs,
                record.cpuPresentCallMs,
                static_cast<unsigned long long>(record.skippedFrames));
            WriteCsvText(m_stageFile, record.upscaler);
            std::fputc(',', m_stageFile);
            WriteCsvText(m_stageFile, record.selectedUpscaler);
            std::fputc(',', m_stageFile);
            WriteCsvText(m_stageFile, record.inputSize);
            std::fputc(',', m_stageFile);
            WriteCsvText(m_stageFile, record.internalOutputSize);
            std::fputc(',', m_stageFile);
            WriteCsvText(m_stageFile, record.outputFormat);
            std::fprintf(
                m_stageFile,
                ",%s\n",
                record.dumpFrame != 0 ? "yes" : "no");
            if (std::ferror(m_stageFile) != 0)
            {
                RecordWriterError(ERROR_WRITE_FAULT);
            }
        }

        bool FilesFailed() const noexcept
        {
            return !m_stageFile || !m_pacingFile ||
                std::ferror(m_stageFile) != 0 ||
                std::ferror(m_pacingFile) != 0;
        }

        void FlushFiles() noexcept
        {
            if (m_stageFile && std::fflush(m_stageFile) != 0)
            {
                RecordWriterError(ERROR_WRITE_FAULT);
            }
            if (m_pacingFile && std::fflush(m_pacingFile) != 0)
            {
                RecordWriterError(ERROR_WRITE_FAULT);
            }
        }

        void CloseFiles() noexcept
        {
            if (m_stageFile)
            {
                std::fflush(m_stageFile);
                std::fclose(m_stageFile);
                m_stageFile = nullptr;
            }
            if (m_pacingFile)
            {
                std::fflush(m_pacingFile);
                std::fclose(m_pacingFile);
                m_pacingFile = nullptr;
            }
        }

        void RecordWriterError(DWORD error) noexcept
        {
            if (error == ERROR_SUCCESS)
            {
                error = ERROR_WRITE_FAULT;
            }
            DWORD expected = ERROR_SUCCESS;
            m_writerError.compare_exchange_strong(
                expected,
                error,
                std::memory_order_relaxed,
                std::memory_order_relaxed);
        }

        BoundedSpscQueue<FrameTimingRecord, kFrameQueueCapacity> m_frameQueue;
        BoundedSpscQueue<GpuStageTimingRecord, kGpuStageQueueCapacity> m_gpuStageQueue;
        std::atomic<bool> m_enabled{ false };
        std::atomic<uint64_t> m_droppedFrames{ 0 };
        std::atomic<uint64_t> m_droppedGpuStages{ 0 };
        std::atomic<uint64_t> m_writtenFrames{ 0 };
        std::atomic<uint64_t> m_writtenGpuStages{ 0 };
        std::atomic<DWORD> m_writerError{ ERROR_SUCCESS };
        std::atomic<bool> m_lowPriorityApplied{ false };
        std::atomic<bool> m_stopping{ false };
        alignas(4) std::atomic<uint32_t> m_wakeSequence{ 0 };
        std::filesystem::path m_stageCsvPath;
        std::filesystem::path m_pacingCsvPath;
        int64_t m_qpcFrequency = 0;
        std::thread m_thread;
        FILE* m_stageFile = nullptr;
        FILE* m_pacingFile = nullptr;
        std::array<char, 64 * 1024> m_stageFileBuffer{};
        std::array<char, 64 * 1024> m_pacingFileBuffer{};
    };

    FrameTelemetryWriter::FrameTelemetryWriter() = default;

    FrameTelemetryWriter::~FrameTelemetryWriter() = default;

    bool FrameTelemetryWriter::Start(
        const std::filesystem::path& stageCsvPath,
        int64_t qpcFrequency,
        std::string& diagnostic) noexcept
    {
        if (!m_impl)
        {
            try
            {
                // The multi-megabyte fixed queues exist only when logging is
                // explicitly enabled. Disabled telemetry retains just the
                // pImpl pointer and has no startup allocation.
                m_impl = std::make_unique<Impl>();
            }
            catch (const std::exception& error)
            {
                diagnostic = error.what();
                return false;
            }
        }
        return m_impl->Start(stageCsvPath, qpcFrequency, diagnostic);
    }

    void FrameTelemetryWriter::Stop() noexcept
    {
        if (m_impl)
        {
            m_impl->Stop();
        }
    }

    bool FrameTelemetryWriter::Enabled() const noexcept
    {
        return m_impl && m_impl->m_enabled.load(std::memory_order_acquire);
    }

    bool FrameTelemetryWriter::TryEnqueueFrame(const FrameTimingRecord& record) noexcept
    {
        return m_impl && m_impl->TryEnqueueFrame(record);
    }

    bool FrameTelemetryWriter::TryEnqueueGpuStage(const GpuStageTimingRecord& record) noexcept
    {
        return m_impl && m_impl->TryEnqueueGpuStage(record);
    }

    uint64_t FrameTelemetryWriter::DroppedFrameRecords() const noexcept
    {
        return m_impl
            ? m_impl->m_droppedFrames.load(std::memory_order_relaxed)
            : 0;
    }

    uint64_t FrameTelemetryWriter::DroppedGpuStageRecords() const noexcept
    {
        return m_impl
            ? m_impl->m_droppedGpuStages.load(std::memory_order_relaxed)
            : 0;
    }

    uint64_t FrameTelemetryWriter::WrittenFrameRecords() const noexcept
    {
        return m_impl
            ? m_impl->m_writtenFrames.load(std::memory_order_relaxed)
            : 0;
    }

    uint64_t FrameTelemetryWriter::WrittenGpuStageRecords() const noexcept
    {
        return m_impl
            ? m_impl->m_writtenGpuStages.load(std::memory_order_relaxed)
            : 0;
    }

    DWORD FrameTelemetryWriter::WriterError() const noexcept
    {
        return m_impl
            ? m_impl->m_writerError.load(std::memory_order_relaxed)
            : ERROR_SUCCESS;
    }

    bool FrameTelemetryWriter::LowPriorityApplied() const noexcept
    {
        return m_impl &&
            m_impl->m_lowPriorityApplied.load(std::memory_order_acquire);
    }

    const std::filesystem::path& FrameTelemetryWriter::StageCsvPath() const noexcept
    {
        static const std::filesystem::path empty;
        return m_impl ? m_impl->m_stageCsvPath : empty;
    }

    const std::filesystem::path& FrameTelemetryWriter::PacingCsvPath() const noexcept
    {
        static const std::filesystem::path empty;
        return m_impl ? m_impl->m_pacingCsvPath : empty;
    }

    std::filesystem::path FrameTelemetryWriter::PacingPathForStagePath(
        const std::filesystem::path& stageCsvPath)
    {
        const std::filesystem::path extension = stageCsvPath.extension();
        std::filesystem::path filename = stageCsvPath.stem().wstring() + L"-pacing";
        filename += extension.empty() ? L".csv" : extension.wstring();
        return stageCsvPath.parent_path() / filename;
    }

    bool FrameTelemetryWriter::SelfTest(std::string& diagnostic)
    {
        try
        {
            if (std::strcmp(
                    FrameDropReasonName(FrameDropReason::SourceTimingJournalOverflow),
                    "source-timing-journal-overflow") != 0)
            {
                diagnostic = "source timing journal overflow reason-name test failed";
                return false;
            }

            BoundedSpscQueue<uint64_t, 4> queue;
            bool wasEmpty = false;
            for (uint64_t value = 1; value <= 4; ++value)
            {
                if (!queue.TryPush(value, wasEmpty) || (value == 1) != wasEmpty)
                {
                    diagnostic = "SPSC fill/empty-transition test failed";
                    return false;
                }
            }
            uint64_t value = 5;
            if (queue.TryPush(value, wasEmpty))
            {
                diagnostic = "SPSC full-bound test failed";
                return false;
            }
            for (uint64_t expected = 1; expected <= 2; ++expected)
            {
                value = 0;
                if (!queue.TryPop(value) || value != expected)
                {
                    diagnostic = "SPSC initial-order test failed";
                    return false;
                }
            }
            for (uint64_t pushed = 5; pushed <= 6; ++pushed)
            {
                if (!queue.TryPush(pushed, wasEmpty) || wasEmpty)
                {
                    diagnostic = "SPSC wrap publication test failed";
                    return false;
                }
            }
            for (uint64_t expected = 3; expected <= 6; ++expected)
            {
                value = 0;
                if (!queue.TryPop(value) || value != expected)
                {
                    diagnostic = "SPSC wrapped-order test failed";
                    return false;
                }
            }
            if (!queue.Empty() || queue.TryPop(value))
            {
                diagnostic = "SPSC empty-bound test failed";
                return false;
            }

            std::array<wchar_t, MAX_PATH + 1> tempPath{};
            const DWORD tempLength = GetTempPathW(
                static_cast<DWORD>(tempPath.size()),
                tempPath.data());
            if (tempLength == 0 || tempLength >= tempPath.size())
            {
                diagnostic = "GetTempPath failed for telemetry self-test";
                return false;
            }
            const std::filesystem::path directory =
                std::filesystem::path(tempPath.data()) /
                (L"AESS-FrameTelemetry-SelfTest-" +
                    std::to_wstring(GetCurrentProcessId()) + L"-" +
                    std::to_wstring(GetCurrentThreadId()));
            const std::filesystem::path stagePath = directory / L"records.csv";
            const std::filesystem::path pacingPath = directory / L"records-pacing.csv";
            std::error_code ignored;
            std::filesystem::remove(stagePath, ignored);
            ignored.clear();
            std::filesystem::remove(pacingPath, ignored);
            ignored.clear();
            std::filesystem::remove(directory, ignored);

            FrameTelemetryWriter writer;
            std::string startDiagnostic;
            if (!writer.Start(stagePath, 10'000'000, startDiagnostic))
            {
                diagnostic = "writer lifecycle test failed to start: " + startDiagnostic;
                return false;
            }

            FrameTimingRecord frame{};
            frame.recordId = 17;
            frame.frameNumber = 23;
            frame.sourceVersion = 29;
            frame.sourcePresentOrdinal = 31;
            frame.sourcePresentQpc = 37;
            frame.presentCallBeginQpc = 41;
            frame.actualSyncQpc = 43;
            frame.droppedSourcePresentsBefore = 3;
            frame.sourceTimingJournalOverflowCount = 5;
            frame.addedIntervalResidualMs = 0.125;
            frame.dxgiPresentCount = 47;
            frame.dropReason = FrameDropReason::ReadySuperseded;
            frame.flags = FrameTimingValidSourcePresentQpc |
                FrameTimingPresentAttempted |
                FrameTimingActualPresentObserved;
            if (!writer.TryEnqueueFrame(frame))
            {
                diagnostic = "writer lifecycle frame enqueue failed";
                writer.Stop();
                return false;
            }

            GpuStageTimingRecord stage{};
            stage.frameNumber = 23;
            stage.sourceVersion = 29;
            stage.gpuMs = 1.25;
            stage.cpuRenderMs = 2.5;
            stage.skippedFrames = 3;
            CopyTelemetryText(stage.stage, "cuda-body-layer-1");
            CopyTelemetryText(stage.upscaler, "srcpyr-fp8");
            CopyTelemetryText(stage.selectedUpscaler, "srcpyr-fp8");
            CopyTelemetryText(stage.inputSize, "1920x1080");
            CopyTelemetryText(stage.internalOutputSize, "3840x2160");
            CopyTelemetryText(stage.outputFormat, "R16G16B16A16_FLOAT");
            if (!writer.TryEnqueueGpuStage(stage))
            {
                diagnostic = "writer lifecycle GPU stage enqueue failed";
                writer.Stop();
                return false;
            }

            const auto waitForWrittenCounts = [&writer](
                uint64_t frames,
                uint64_t stages) noexcept
            {
                const ULONGLONG deadline = GetTickCount64() + 2000;
                while (writer.WrittenFrameRecords() < frames ||
                    writer.WrittenGpuStageRecords() < stages)
                {
                    if (GetTickCount64() >= deadline)
                    {
                        return false;
                    }
                    Sleep(1);
                }
                return true;
            };
            if (!waitForWrittenCounts(1, 1))
            {
                diagnostic = "writer first idle-transition wake test timed out";
                writer.Stop();
                return false;
            }

            // Let the consumer return to WaitOnAddress, then publish another
            // pair.  Stop() must not be the operation that wakes either batch.
            Sleep(10);
            frame.recordId = 18;
            frame.frameNumber = 24;
            stage.frameNumber = 24;
            if (!writer.TryEnqueueFrame(frame) ||
                !writer.TryEnqueueGpuStage(stage) ||
                !waitForWrittenCounts(2, 2))
            {
                diagnostic = "writer repeated idle-transition wake test failed";
                writer.Stop();
                return false;
            }
            writer.Stop();

            if (writer.DroppedFrameRecords() != 0 ||
                writer.DroppedGpuStageRecords() != 0 ||
                writer.WrittenFrameRecords() != 2 ||
                writer.WrittenGpuStageRecords() != 2 ||
                writer.WriterError() != ERROR_SUCCESS ||
                !writer.LowPriorityApplied())
            {
                diagnostic = "writer lifecycle counters/error test failed";
                return false;
            }

            const std::string pacingCsv = ReadWholeFile(pacingPath);
            const std::string stageCsv = ReadWholeFile(stagePath);
            if (pacingCsv.find("schema_version,qpc_frequency") != 0 ||
                pacingCsv.find("source_timing_journal_overflow_count,added_interval_residual_ms,") == std::string::npos ||
                pacingCsv.find("ready-superseded") == std::string::npos ||
                pacingCsv.find(",17,23,29,31,") == std::string::npos ||
                pacingCsv.find(",3,5,0.125000,47,") == std::string::npos)
            {
                diagnostic = "pacing CSV schema/content test failed";
                return false;
            }
            if (stageCsv.find("frame,source_version,stage") != 0 ||
                stageCsv.find("\"cuda-body-layer-1\",1.2500") == std::string::npos ||
                stageCsv.find("\"srcpyr-fp8\"") == std::string::npos)
            {
                diagnostic = "GPU stage CSV schema/content test failed";
                return false;
            }

            std::filesystem::remove(stagePath, ignored);
            ignored.clear();
            std::filesystem::remove(pacingPath, ignored);
            ignored.clear();
            std::filesystem::remove(directory, ignored);
            diagnostic = "bounded SPSC ordering, repeated address-wake transitions, and background CSV writer lifecycle passed";
            return true;
        }
        catch (const std::exception& error)
        {
            diagnostic = error.what();
            return false;
        }
    }
}

#if defined(AESS_FRAME_TELEMETRY_STANDALONE_TEST)
int main()
{
    std::string diagnostic;
    const bool passed = aess::presentation::FrameTelemetryWriter::SelfTest(diagnostic);
    std::fprintf(stdout, "%s: %s\n", passed ? "PASS" : "FAIL", diagnostic.c_str());
    return passed ? 0 : 1;
}
#endif
