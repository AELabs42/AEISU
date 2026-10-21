#pragma once

#include <windows.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

namespace aess::presentation
{
    inline constexpr uint32_t kFrameTimingSchemaVersion = 3;

    enum class FrameDropReason : uint32_t
    {
        None = 0,
        CaptureSuperseded = 1,
        PipelineSaturated = 2,
        ReadySuperseded = 3,
        StaleOverTwoPeriods = 4,
        TimestampRegression = 5,
        ResizeReset = 6,
        FocusReset = 7,
        UpscalerReset = 8,
        PresentationReset = 9,
        PresentRejected = 10,
        DxgiSuperseded = 11,
        ShutdownDiscard = 12,
        SourceTimingJournalOverflow = 13,
    };

    enum FrameTimingFlag : uint32_t
    {
        FrameTimingValidSourcePresentQpc = 1u << 0,
        FrameTimingValidCaptureReadyQpc = 1u << 1,
        FrameTimingEffectSubmitted = 1u << 2,
        FrameTimingEffectCompleted = 1u << 3,
        FrameTimingEnteredReadyQueue = 1u << 4,
        FrameTimingPresentAttempted = 1u << 5,
        FrameTimingPresentAccepted = 1u << 6,
        FrameTimingActualPresentObserved = 1u << 7,
        FrameTimingPresentationStatsUnavailable = 1u << 8,
        FrameTimingSyntheticGapRecord = 1u << 9,
    };

    // This record deliberately contains only fixed-width scalar data. It is
    // copied into the telemetry queue with no allocation, formatting, or lock.
    // Zero QPC fields are absent unless the corresponding validity flag is set.
    struct FrameTimingRecord
    {
        uint64_t recordId = 0;
        uint64_t frameNumber = 0;
        uint64_t sourceVersion = 0;
        uint64_t sourcePresentOrdinal = 0;
        uint64_t captureArrivalCount = 0;
        uint64_t effectSubmitOrdinal = 0;

        int64_t sourcePresentQpc = 0;
        int64_t captureReadyQpc = 0;
        int64_t captureAcquireBeginQpc = 0;
        int64_t captureAcquireEndQpc = 0;
        int64_t effectSubmitQpc = 0;
        int64_t effectCompleteQpc = 0;
        int64_t readyQueueEntryQpc = 0;
        int64_t targetDeadlineQpc = 0;
        int64_t latencyGateBeginQpc = 0;
        int64_t latencyGateEndQpc = 0;
        int64_t presentCallBeginQpc = 0;
        int64_t presentCallEndQpc = 0;
        int64_t actualSyncQpc = 0;

        uint64_t droppedSourcePresentsBefore = 0;
        uint64_t sourceTimingJournalOverflowCount = 0;
        // Absolute difference between consecutive, ordinal-correlated source
        // and monitor-present intervals. Negative means unavailable.
        double addedIntervalResidualMs = -1.0;
        uint32_t dxgiPresentCount = 0;
        uint32_t inputWidth = 0;
        uint32_t inputHeight = 0;
        uint32_t readyDepthAtEntry = 0;
        uint32_t readyDepthBeforePresent = 0;
        uint32_t readyDepthAfterPresent = 0;
        uint32_t queueTarget = 0;
        uint32_t upscalerMode = 0;
        uint32_t pacingMode = 0;
        uint32_t presentationMode = 0;
        FrameDropReason dropReason = FrameDropReason::None;
        uint32_t flags = 0;
        uint64_t writerOverflowBefore = 0;
    };

    // Fixed-size equivalent of the existing stage-oriented performance CSV
    // row. Text is copied into inline arrays before publication, so the writer
    // can retain the current CSV schema without render-thread allocations.
    struct GpuStageTimingRecord
    {
        uint64_t frameNumber = 0;
        uint64_t sourceVersion = 0;
        uint64_t skippedFrames = 0;
        double gpuMs = 0.0;
        double cpuRenderMs = 0.0;
        double cpuAcquireMs = 0.0;
        double cpuPreprocessMs = 0.0;
        double cpuUpscaleMs = 0.0;
        double cpuPresentSetupMs = 0.0;
        double cpuPresentCallMs = 0.0;
        std::array<char, 48> stage{};
        std::array<char, 32> upscaler{};
        std::array<char, 32> selectedUpscaler{};
        std::array<char, 24> inputSize{};
        std::array<char, 24> internalOutputSize{};
        std::array<char, 32> outputFormat{};
        uint32_t dumpFrame = 0;
    };

    static_assert(std::is_trivially_copyable_v<FrameTimingRecord>);
    static_assert(std::is_trivially_copyable_v<GpuStageTimingRecord>);

    template <size_t Count>
    void CopyTelemetryText(
        std::array<char, Count>& destination,
        std::string_view source) noexcept
    {
        static_assert(Count > 0);
        const size_t copied = source.size() < Count - 1 ? source.size() : Count - 1;
        if (copied != 0)
        {
            std::memcpy(destination.data(), source.data(), copied);
        }
        destination[copied] = '\0';
        if (copied + 1 < Count)
        {
            std::memset(destination.data() + copied + 1, 0, Count - copied - 1);
        }
    }

    const char* FrameDropReasonName(FrameDropReason reason) noexcept;

    // A single render/present thread is the producer. One background writer is
    // the consumer. Start and Stop must run outside concurrent producer calls;
    // TryEnqueue* never blocks, allocates, formats text, or touches a file.
    class FrameTelemetryWriter
    {
    public:
        FrameTelemetryWriter();
        ~FrameTelemetryWriter();

        FrameTelemetryWriter(const FrameTelemetryWriter&) = delete;
        FrameTelemetryWriter& operator=(const FrameTelemetryWriter&) = delete;

        bool Start(
            const std::filesystem::path& stageCsvPath,
            int64_t qpcFrequency,
            std::string& diagnostic) noexcept;
        void Stop() noexcept;

        bool Enabled() const noexcept;
        bool TryEnqueueFrame(const FrameTimingRecord& record) noexcept;
        bool TryEnqueueGpuStage(const GpuStageTimingRecord& record) noexcept;

        uint64_t DroppedFrameRecords() const noexcept;
        uint64_t DroppedGpuStageRecords() const noexcept;
        uint64_t WrittenFrameRecords() const noexcept;
        uint64_t WrittenGpuStageRecords() const noexcept;
        DWORD WriterError() const noexcept;
        bool LowPriorityApplied() const noexcept;

        const std::filesystem::path& StageCsvPath() const noexcept;
        const std::filesystem::path& PacingCsvPath() const noexcept;

        static std::filesystem::path PacingPathForStagePath(
            const std::filesystem::path& stageCsvPath);
        static bool SelfTest(std::string& diagnostic);

    private:
        class Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
