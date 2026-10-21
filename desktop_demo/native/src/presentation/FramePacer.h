#pragma once

#include <windows.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace aess::presentation
{
    enum class FramePacingMode : uint32_t
    {
        LowLatency = 0,
        Adaptive = 1,
        Smooth = 2,
    };

    struct FramePacingConfig
    {
        FramePacingMode pacingMode = FramePacingMode::Adaptive;
    };

    struct FramePacingSnapshot
    {
        double sourceMedianMs = 0.0;
        double sourceMadMs = 0.0;
        double sourceJitterMs = 0.0;
        double completionP95Ms = 0.0;
        double interCompletionP95Ms = 0.0;
        double targetPeriodMs = 0.0;
        double postPreparationLeadUs = 0.0;
        double setupLeadUs = 0.0;
        double timerSpinUs = 0.0;
        double lastDeadlineErrorMs = 0.0;
        double completionWakeP95Ms = 0.0;
        double sourceAgeP95Ms = 0.0;
        uint64_t deadlineMisses = 0;
        uint64_t underruns = 0;
        uint64_t discontinuities = 0;
        bool buffering = false;
        uint32_t targetQueueDepth = 0;
    };

    // Allocation-free cadence estimator and deadline scheduler. The render
    // thread remains the sole caller; no internal locking is required.
    class FramePacer
    {
    public:
        FramePacer();
        explicit FramePacer(const FramePacingConfig& config);
        ~FramePacer();

        FramePacer(const FramePacer&) = delete;
        FramePacer& operator=(const FramePacer&) = delete;

        void Configure(const FramePacingConfig& config);
        void Reset() noexcept;

        void ObserveSource(int64_t sourceQpc, uint64_t sourceOrdinal = 0) noexcept;
        void ObserveCompletion(
            int64_t completionQpc,
            int64_t submissionQpc,
            double serviceDurationMs = -1.0) noexcept;
        void ObservePostPreparation(
            int64_t submissionQpc,
            int64_t completionQpc) noexcept;

        // Call when at least one completed frame is ready. Returns true when
        // it should be presented now; otherwise TimerEvent() is armed.
        bool ReadyToPresent(
            int64_t nowQpc,
            size_t readyDepth,
            bool includePostPreparation = false,
            bool completedFrameRetention = true,
            int64_t sourceQpc = 0) noexcept;
        void FinishTimerWait() noexcept;
        void WaitForPresentDeadline(int64_t setupBeginQpc) noexcept;
        void OnPresented(
            int64_t presentQpc,
            int64_t sourceQpc,
            int64_t completionQpc,
            size_t readyDepthAfterPresent,
            bool completedFrameRetention = true) noexcept;
        void OnUnderrun(int64_t nowQpc) noexcept;
        void ResetTimeline() noexcept;

        HANDLE TimerEvent() const noexcept;
        bool TimerArmed() const noexcept;
        bool BufferingActive() const noexcept;
        uint32_t TargetQueueDepth(
            bool completedFrameRetention = true) const noexcept;
        // Low latency has no scheduling period, but stale-frame policy still
        // needs the measured source cadence.
        int64_t EstimatedSourcePeriodQpc() const noexcept;
        int64_t TargetPeriodQpc(int64_t nowQpc) const noexcept;
        int64_t DeadlineQpc() const noexcept;
        int64_t QpcFrequency() const noexcept;
        FramePacingSnapshot Snapshot() const noexcept;

        static int64_t QueryQpc() noexcept;
        static bool SelfTest(std::string& diagnostic);

    private:
        static constexpr size_t kCadenceWindow = 15;
        static constexpr size_t kMetricWindow = 64;

        template <size_t Count>
        struct SampleWindow
        {
            std::array<double, Count> values{};
            size_t count = 0;
            size_t cursor = 0;
            // Queries often outnumber admitted frames. Reuse exact statistics
            // until the ring changes; the render thread is the only reader.
            mutable std::array<double, Count> sorted{};
            mutable double cadenceCenter = 0.0;
            mutable double mad = 0.0;
            mutable double madCenter = 0.0;
            mutable bool sortedValid = false;
            mutable bool cadenceValid = false;
            mutable bool madValid = false;

            void Invalidate() noexcept
            {
                sortedValid = false;
                cadenceValid = false;
                madValid = false;
            }

            void Clear() noexcept
            {
                count = 0;
                cursor = 0;
                // Only [0, count) is read; clearing the entire storage is not
                // necessary, including after a discontinuity.
                Invalidate();
            }

            void Push(double value) noexcept
            {
                // Zero is a meaningful sample for MAD and wake-latency
                // windows. Interval callers reject non-positive deltas before
                // reaching this container.
                if (value < 0.0 || !std::isfinite(value)) return;
                values[cursor] = value;
                cursor = (cursor + 1) % Count;
                count = count < Count ? count + 1 : Count;
                Invalidate();
            }
        };

        template <size_t Count>
        static const std::array<double, Count>& SortedValues(const SampleWindow<Count>& samples) noexcept;
        template <size_t Count>
        static double Percentile(const SampleWindow<Count>& samples, double percentile) noexcept;
        template <size_t Count>
        static double Median(const SampleWindow<Count>& samples) noexcept;
        template <size_t Count>
        static double MedianAbsoluteDeviation(const SampleWindow<Count>& samples, double median) noexcept;
        template <size_t Count>
        static double CadenceCenter(const SampleWindow<Count>& samples) noexcept;

        void CreateTimer();
        void CancelTimer() noexcept;
        bool ArmTimer(
            int64_t deadlineQpc,
            bool includePostPreparation) noexcept;
        bool ArmWaitableTimerAt(int64_t wakeTargetQpc) noexcept;
        void RecordTimerWake(int64_t nowQpc) noexcept;
        void RecordDeadlineMiss(int64_t nowQpc, int64_t periodQpc) noexcept;
        void ObservePresentationSetup(double durationUs) noexcept;
        int64_t SetupWakeTargetQpc(
            int64_t deadlineQpc,
            bool includePostPreparation = false) const noexcept;
        int64_t FinalWakeTargetQpc(int64_t deadlineQpc) const noexcept;
        void UpdateAdaptiveState() noexcept;
        double DesiredRateHz(int64_t nowQpc) const noexcept;
        int64_t DesiredPeriodQpc(int64_t nowQpc) const noexcept;
        int64_t MicrosecondsToQpc(double microseconds) const noexcept;
        double QpcToMilliseconds(int64_t ticks) const noexcept;
        FramePacingConfig m_config{};
        HANDLE m_timer = nullptr;
        int64_t m_qpcFrequency = 1;
        int64_t m_lastSourceQpc = 0;
        uint64_t m_lastSourceOrdinal = 0;
        int64_t m_lastCompletionQpc = 0;
        int64_t m_lastPresentQpc = 0;
        int64_t m_deadlineQpc = 0;
        int64_t m_timerWakeTargetQpc = 0;
        double m_postPreparationLeadUs = 500.0;
        double m_setupLeadUs = 500.0;
        double m_spinBudgetUs = 50.0;
        double m_armedSpinBudgetUs = 50.0;
        double m_previousSourceIntervalMs = 0.0;
        double m_lastDeadlineErrorMs = 0.0;
        uint32_t m_badAdaptiveWindows = 0;
        uint32_t m_stableAdaptiveFrames = 0;
        uint64_t m_sourceWindowGeneration = 0;
        uint64_t m_lastAdaptiveEvaluationGeneration = 0;
        uint64_t m_deadlineMisses = 0;
        uint64_t m_underruns = 0;
        uint64_t m_discontinuities = 0;
        bool m_adaptiveBuffering = false;
        bool m_timerArmed = false;
        bool m_presentImmediately = false;
        bool m_readyBeforeDeadline = false;
        bool m_setupWindowOpen = false;
        bool m_reanchorOnPresent = false;
        bool m_deadlineMissRecorded = false;
        bool m_timerResolutionRequested = false;
        SampleWindow<kCadenceWindow> m_sourceIntervalsMs;
        SampleWindow<kCadenceWindow> m_sourceSuccessiveDeltaMs;
        // Queue-free effect service time drives utilization and sustainable
        // capacity. Inter-completion cadence is diagnostic only because an
        // under-capacity pipeline inherits the source cadence.
        SampleWindow<kCadenceWindow> m_completionPeriodsMs;
        SampleWindow<kCadenceWindow> m_interCompletionPeriodsMs;
        SampleWindow<kMetricWindow> m_postPreparationUs;
        SampleWindow<kMetricWindow> m_presentationSetupUs;
        SampleWindow<kMetricWindow> m_timerOvershootUs;
        SampleWindow<kMetricWindow> m_completionWakeMs;
        SampleWindow<kMetricWindow> m_sourceAgeMs;
    };

    const wchar_t* FramePacingModeName(FramePacingMode mode) noexcept;
}
