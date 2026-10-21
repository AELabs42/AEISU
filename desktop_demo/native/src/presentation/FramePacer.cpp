#include "FramePacer.h"

#include <algorithm>
#include <cmath>
#include <intrin.h>
#include <limits>
#include <mmsystem.h>
#include <stdexcept>

namespace aess::presentation
{
    namespace
    {
        constexpr double kMillisecondsPerSecond = 1000.0;
        constexpr double kMicrosecondsPerSecond = 1000000.0;
        // Adaptive's retained completion is the temporal guard. It does not
        // stack a second, independent startup delay on top; the period timer
        // establishes cadence after the first immediate anchor. Smooth is the
        // explicit mode that deliberately starts with a full-frame lead.
        constexpr double kAdaptiveLeadFraction = 0.0;

    }

    FramePacer::FramePacer()
    {
        LARGE_INTEGER frequency{};
        if (QueryPerformanceFrequency(&frequency) && frequency.QuadPart > 0)
        {
            m_qpcFrequency = frequency.QuadPart;
        }
        CreateTimer();
    }

    FramePacer::FramePacer(const FramePacingConfig& config)
        : FramePacer()
    {
        Configure(config);
    }

    FramePacer::~FramePacer()
    {
        CancelTimer();
        if (m_timer)
        {
            CloseHandle(m_timer);
            m_timer = nullptr;
        }
        if (m_timerResolutionRequested)
        {
            timeEndPeriod(1);
            m_timerResolutionRequested = false;
        }
    }

    void FramePacer::CreateTimer()
    {
        m_timer = CreateWaitableTimerExW(
            nullptr,
            nullptr,
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
            TIMER_ALL_ACCESS);
        if (!m_timer)
        {
            m_timer = CreateWaitableTimerW(nullptr, FALSE, nullptr);
            if (m_timer && timeBeginPeriod(1) == TIMERR_NOERROR)
            {
                // Only the legacy timer fallback needs a process-wide timer
                // resolution request. The normal high-resolution waitable
                // timer path leaves the system timer policy untouched.
                m_timerResolutionRequested = true;
            }
        }
        if (!m_timer)
        {
            throw std::runtime_error("Create frame-pacing waitable timer failed");
        }
    }

    void FramePacer::Configure(const FramePacingConfig& config)
    {
        m_config = config;
        Reset();
    }

    void FramePacer::Reset() noexcept
    {
        CancelTimer();
        m_lastSourceQpc = 0;
        m_lastSourceOrdinal = 0;
        m_lastCompletionQpc = 0;
        m_lastPresentQpc = 0;
        m_deadlineQpc = 0;
        m_timerWakeTargetQpc = 0;
        m_postPreparationLeadUs = 500.0;
        m_setupLeadUs = 500.0;
        m_spinBudgetUs = 50.0;
        m_armedSpinBudgetUs = 50.0;
        m_previousSourceIntervalMs = 0.0;
        m_lastDeadlineErrorMs = 0.0;
        m_badAdaptiveWindows = 0;
        m_stableAdaptiveFrames = 0;
        m_sourceWindowGeneration = 0;
        m_lastAdaptiveEvaluationGeneration = 0;
        m_deadlineMisses = 0;
        m_underruns = 0;
        m_discontinuities = 0;
        m_adaptiveBuffering = false;
        m_presentImmediately = false;
        m_readyBeforeDeadline = false;
        m_setupWindowOpen = false;
        m_reanchorOnPresent = false;
        m_deadlineMissRecorded = false;
        m_sourceIntervalsMs.Clear();
        m_sourceSuccessiveDeltaMs.Clear();
        m_completionPeriodsMs.Clear();
        m_interCompletionPeriodsMs.Clear();
        m_postPreparationUs.Clear();
        m_presentationSetupUs.Clear();
        m_timerOvershootUs.Clear();
        m_completionWakeMs.Clear();
        m_sourceAgeMs.Clear();
    }

    void FramePacer::ResetTimeline() noexcept
    {
        CancelTimer();
        m_lastSourceQpc = 0;
        m_lastSourceOrdinal = 0;
        m_lastCompletionQpc = 0;
        m_lastPresentQpc = 0;
        m_deadlineQpc = 0;
        m_timerWakeTargetQpc = 0;
        m_badAdaptiveWindows = 0;
        m_stableAdaptiveFrames = 0;
        m_previousSourceIntervalMs = 0.0;
        m_postPreparationLeadUs = 500.0;
        m_setupLeadUs = 500.0;
        m_spinBudgetUs = 50.0;
        m_armedSpinBudgetUs = 50.0;
        m_lastDeadlineErrorMs = 0.0;
        m_adaptiveBuffering = false;
        m_presentImmediately = true;
        m_readyBeforeDeadline = false;
        m_setupWindowOpen = false;
        m_reanchorOnPresent = true;
        m_deadlineMissRecorded = false;
        m_sourceWindowGeneration = 0;
        m_lastAdaptiveEvaluationGeneration = 0;
        m_sourceIntervalsMs.Clear();
        m_sourceSuccessiveDeltaMs.Clear();
        m_completionPeriodsMs.Clear();
        m_interCompletionPeriodsMs.Clear();
        m_postPreparationUs.Clear();
        m_presentationSetupUs.Clear();
        m_timerOvershootUs.Clear();
        m_completionWakeMs.Clear();
        m_sourceAgeMs.Clear();
        // Deadline/underrun totals are lifetime diagnostics. Only estimator
        // and phase-local history is isolated at a timeline discontinuity.
        ++m_discontinuities;
    }

    template <size_t Count>
    const std::array<double, Count>& FramePacer::SortedValues(const SampleWindow<Count>& samples) noexcept
    {
        if (!samples.sortedValid)
        {
            std::copy_n(samples.values.begin(), samples.count, samples.sorted.begin());
            std::sort(samples.sorted.begin(), samples.sorted.begin() + static_cast<std::ptrdiff_t>(samples.count));
            samples.sortedValid = true;
        }
        return samples.sorted;
    }

    template <size_t Count>
    double FramePacer::Percentile(const SampleWindow<Count>& samples, double percentile) noexcept
    {
        if (samples.count == 0) return 0.0;
        const auto& sorted = SortedValues(samples);
        const double position = std::clamp(percentile, 0.0, 1.0) *
            static_cast<double>(samples.count - 1);
        const size_t index = static_cast<size_t>(std::ceil(position));
        return sorted[std::min(index, samples.count - 1)];
    }

    template <size_t Count>
    double FramePacer::Median(const SampleWindow<Count>& samples) noexcept
    {
        if (samples.count == 0) return 0.0;
        const auto& sorted = SortedValues(samples);
        const size_t middle = samples.count / 2;
        return (samples.count & 1u) != 0
            ? sorted[middle]
            : 0.5 * (sorted[middle - 1] + sorted[middle]);
    }

    template <size_t Count>
    double FramePacer::MedianAbsoluteDeviation(
        const SampleWindow<Count>& samples,
        double median) noexcept
    {
        if (samples.count == 0) return 0.0;
        if (samples.madValid && samples.madCenter == median) return samples.mad;
        SampleWindow<Count> deviations;
        for (size_t index = 0; index < samples.count; ++index)
        {
            deviations.Push(std::abs(samples.values[index] - median));
        }
        samples.mad = Median(deviations);
        samples.madCenter = median;
        samples.madValid = true;
        return samples.mad;
    }

    template <size_t Count>
    double FramePacer::CadenceCenter(const SampleWindow<Count>& samples) noexcept
    {
        if (samples.count == 0) return 0.0;
        if (samples.count == 1) return samples.values[0];
        if (samples.cadenceValid) return samples.cadenceCenter;

        // A raw odd-sized median changes phase on a bimodal alternating
        // cadence (8 short/7 long, then 7 short/8 long).  Median-filter the
        // adjacent two-interval averages instead: each pair represents the
        // local long-term source rate, rejects isolated outliers, and remains
        // allocation-free.  Jitter qualification still uses the mandated
        // 15-interval raw median/MAD above.
        SampleWindow<Count> pairAverages;
        const size_t oldest = samples.count == Count ? samples.cursor : 0;
        double previous = samples.values[oldest];
        for (size_t offset = 1; offset < samples.count; ++offset)
        {
            const size_t index = (oldest + offset) % Count;
            const double current = samples.values[index];
            pairAverages.Push(0.5 * (previous + current));
            previous = current;
        }
        samples.cadenceCenter = Median(pairAverages);
        samples.cadenceValid = true;
        return samples.cadenceCenter;
    }

    void FramePacer::ObserveSource(int64_t sourceQpc, uint64_t sourceOrdinal) noexcept
    {
        if (sourceQpc <= 0) return;
        if (m_lastSourceQpc > 0)
        {
            if (sourceQpc <= m_lastSourceQpc ||
                (sourceOrdinal > 0 && m_lastSourceOrdinal > 0 &&
                    sourceOrdinal <= m_lastSourceOrdinal))
            {
                ResetTimeline();
                m_lastSourceQpc = sourceQpc;
                m_lastSourceOrdinal = sourceOrdinal;
                return;
            }

            uint64_t ordinalDelta = 1;
            if (sourceOrdinal > 0 && m_lastSourceOrdinal > 0)
            {
                ordinalDelta = sourceOrdinal - m_lastSourceOrdinal;
            }
            const double intervalMs = QpcToMilliseconds(sourceQpc - m_lastSourceQpc) /
                static_cast<double>(ordinalDelta);
            const double medianMs = Median(m_sourceIntervalsMs);
            if (medianMs > 0.0 && intervalMs > medianMs * 6.0)
            {
                ResetTimeline();
                m_lastSourceQpc = sourceQpc;
                m_lastSourceOrdinal = sourceOrdinal;
                return;
            }
            else if (intervalMs <= 1000.0)
            {
                if (m_previousSourceIntervalMs > 0.0)
                {
                    m_sourceSuccessiveDeltaMs.Push(
                        std::abs(intervalMs - m_previousSourceIntervalMs));
                }
                m_previousSourceIntervalMs = intervalMs;
                m_sourceIntervalsMs.Push(intervalMs);
                ++m_sourceWindowGeneration;
            }
        }
        m_lastSourceQpc = sourceQpc;
        m_lastSourceOrdinal = sourceOrdinal;
    }

    void FramePacer::ObserveCompletion(
        int64_t completionQpc,
        int64_t submissionQpc,
        double serviceDurationMs) noexcept
    {
        if (completionQpc <= 0) return;

        double effectServiceMs = -1.0;
        if (std::isfinite(serviceDurationMs) && serviceDurationMs >= 0.0)
        {
            effectServiceMs = serviceDurationMs;
        }
        else if (submissionQpc > 0 && completionQpc >= submissionQpc)
        {
            effectServiceMs = QpcToMilliseconds(completionQpc - submissionQpc);
        }

        if (m_lastCompletionQpc > 0 && completionQpc <= m_lastCompletionQpc)
        {
            ResetTimeline();
            m_lastCompletionQpc = completionQpc;
            if (effectServiceMs >= 0.0 && effectServiceMs <= 10000.0)
            {
                m_completionPeriodsMs.Push(effectServiceMs);
            }
            return;
        }

        const int64_t previousCompletionQpc = m_lastCompletionQpc;
        m_lastCompletionQpc = completionQpc;
        if (previousCompletionQpc > 0)
        {
            const double interCompletionMs = QpcToMilliseconds(
                completionQpc - previousCompletionQpc);
            const double interCompletionMedianMs = Median(m_interCompletionPeriodsMs);
            if (interCompletionMedianMs > 0.0 &&
                interCompletionMs > interCompletionMedianMs * 6.0)
            {
                // Source Present timestamps are the authoritative
                // discontinuity clock; a completion gap only invalidates this
                // diagnostic history.
                m_interCompletionPeriodsMs.Clear();
            }
            else if (interCompletionMs > 0.0 && interCompletionMs <= 10000.0)
            {
                m_interCompletionPeriodsMs.Push(interCompletionMs);
            }
        }

        // Effect service time is the queue-free utilization/capacity signal.
        // Under capacity, inter-completion cadence merely follows the source
        // and must not make Adaptive infer 100% utilization.
        if (!(effectServiceMs >= 0.0) || effectServiceMs > 10000.0) return;
        m_completionPeriodsMs.Push(effectServiceMs);
        UpdateAdaptiveState();
    }

    void FramePacer::ObservePostPreparation(
        int64_t submissionQpc,
        int64_t completionQpc) noexcept
    {
        if (submissionQpc <= 0 || completionQpc < submissionQpc) return;
        const double durationUs = static_cast<double>(
            completionQpc - submissionQpc) * kMicrosecondsPerSecond /
            static_cast<double>(m_qpcFrequency);
        if (!(durationUs >= 0.0) || durationUs > 20000.0) return;
        m_postPreparationUs.Push(durationUs);
        // This is a wake lead, not deliberate buffering.  A p95 plus a small
        // command/wake guard starts the deferred bridge/post pass early enough
        // to finish before the unchanged absolute Present deadline.
        const int64_t targetPeriodQpc = TargetPeriodQpc(completionQpc);
        const double maximumLeadUs = targetPeriodQpc > 0
            ? std::clamp(
                QpcToMilliseconds(targetPeriodQpc) * 1000.0,
                50.0,
                20000.0)
            : 20000.0;
        m_postPreparationLeadUs = std::clamp(
            Percentile(m_postPreparationUs, 0.95) + 50.0,
            50.0,
            maximumLeadUs);
    }

    void FramePacer::UpdateAdaptiveState() noexcept
    {
        if (m_config.pacingMode != FramePacingMode::Adaptive ||
            m_sourceIntervalsMs.count < kCadenceWindow)
        {
            if (m_config.pacingMode != FramePacingMode::Adaptive)
            {
                m_adaptiveBuffering = false;
            }
            return;
        }

        // Qualification is defined in source-cadence windows, not in calls to
        // ObserveCompletion. Two completions collected in one render-thread
        // turn must not count the same source window twice.
        if (m_sourceWindowGeneration == m_lastAdaptiveEvaluationGeneration)
        {
            return;
        }
        m_lastAdaptiveEvaluationGeneration = m_sourceWindowGeneration;

        const double sourceMedian = Median(m_sourceIntervalsMs);
        const double sourceMad = MedianAbsoluteDeviation(m_sourceIntervalsMs, sourceMedian);
        const double completionP95 = Percentile(m_completionPeriodsMs, 0.95);
        const double targetPeriodMs = CadenceCenter(m_sourceIntervalsMs);
        const double successiveDelta = Median(m_sourceSuccessiveDeltaMs) * 0.5;
        const double normalizedJitter = sourceMedian > 0.0
            ? std::max(sourceMad, successiveDelta) / sourceMedian
            : 0.0;
        const bool stressed = normalizedJitter > 0.05 ||
            (completionP95 > 0.0 && targetPeriodMs > 0.0 && completionP95 > targetPeriodMs * 0.90);
        const bool stable = normalizedJitter < 0.025 &&
            (completionP95 <= 0.0 || completionP95 < targetPeriodMs * 0.75);

        if (!m_adaptiveBuffering)
        {
            m_badAdaptiveWindows = stressed ? m_badAdaptiveWindows + 1 : 0;
            if (m_badAdaptiveWindows >= 2)
            {
                m_adaptiveBuffering = true;
                m_stableAdaptiveFrames = 0;
                CancelTimer();
                m_deadlineQpc = 0;
                m_readyBeforeDeadline = false;
                m_setupWindowOpen = false;
                m_deadlineMissRecorded = false;
            }
        }
        else
        {
            m_stableAdaptiveFrames = stable ? m_stableAdaptiveFrames + 1 : 0;
            if (m_stableAdaptiveFrames >= 120)
            {
                m_adaptiveBuffering = false;
                m_badAdaptiveWindows = 0;
                CancelTimer();
                m_deadlineQpc = 0;
                m_readyBeforeDeadline = false;
                m_setupWindowOpen = false;
                m_deadlineMissRecorded = false;
            }
        }
    }

    double FramePacer::DesiredRateHz(int64_t) const noexcept
    {
        if (!BufferingActive())
        {
            return 0.0;
        }

        const double sourceCadenceMs = CadenceCenter(m_sourceIntervalsMs);
        return sourceCadenceMs > 0.0 ? 1000.0 / sourceCadenceMs : 0.0;
    }

    int64_t FramePacer::DesiredPeriodQpc(int64_t nowQpc) const noexcept
    {
        const double rate = DesiredRateHz(nowQpc);
        if (!(rate > 0.0)) return 0;
        return std::max<int64_t>(1, static_cast<int64_t>(
            std::llround(static_cast<double>(m_qpcFrequency) / rate)));
    }

    bool FramePacer::ReadyToPresent(
        int64_t nowQpc,
        size_t readyDepth,
        bool includePostPreparation,
        bool completedFrameRetention,
        int64_t sourceQpc) noexcept
    {
        // The first timer wake intentionally opens a CPU/DXGI setup window.
        // Once open, the selected completed frame stays eligible while the
        // caller prepares the backbuffer and reaches the final Present gate.
        if (m_setupWindowOpen)
        {
            if (m_deadlineQpc > 0 &&
                nowQpc > m_deadlineQpc &&
                !m_reanchorOnPresent)
            {
                RecordDeadlineMiss(nowQpc, TargetPeriodQpc(nowQpc));
                ++m_underruns;
                m_reanchorOnPresent = true;
            }
            return true;
        }

        if (m_presentImmediately)
        {
            CancelTimer();
            m_readyBeforeDeadline = false;
            m_reanchorOnPresent = true;
            return true;
        }

        const uint32_t targetQueue = TargetQueueDepth(completedFrameRetention);
        const int64_t period = DesiredPeriodQpc(nowQpc);
        const bool smoothingActive = BufferingActive();
        if (!smoothingActive)
        {
            CancelTimer();
            m_readyBeforeDeadline = false;
            return true;
        }

        if (period <= 0)
        {
            CancelTimer();
            m_readyBeforeDeadline = false;
            return true;
        }

        bool initializedDeadline = false;
        if (m_deadlineQpc <= 0)
        {
            if (!smoothingActive ||
                (completedFrameRetention &&
                    m_lastPresentQpc == 0 &&
                    readyDepth > targetQueue))
            {
                m_deadlineQpc = nowQpc;
                m_reanchorOnPresent = true;
            }
            else
            {
                // Adaptive uses only the lead required to absorb ordinary
                // cadence error.  Starting with a full-period delay can spend
                // most of the one-frame latency budget before the bounded 1%
                // phase controller has observed its first Present.  Smooth is
                // the explicit continuous one-frame mode and retains the full
                // period; Adaptive anchors immediately and then follows the
                // authoritative source age with its retained completion.
                const int64_t initialLead =
                    m_config.pacingMode == FramePacingMode::Adaptive
                    ? static_cast<int64_t>(std::llround(
                        static_cast<double>(period) * kAdaptiveLeadFraction))
                    : period;
                m_deadlineQpc = nowQpc + std::max<int64_t>(1, initialLead);
                m_reanchorOnPresent = false;
            }
            m_deadlineMissRecorded = false;
            initializedDeadline = true;
        }

        // Normal queue/phase recovery is deliberately limited to 1% per
        // Present.  It must not, however, keep an already retained Adaptive
        // frame waiting beyond a complete source period merely to preserve a
        // deadline phase.  Crossing this bound is an actual smoothing
        // underrun: show the completed frame immediately and re-anchor, as the
        // pacing contract requires.  Smooth remains the explicit mode that
        // may continuously hold a full completed frame.
        if (m_config.pacingMode == FramePacingMode::Adaptive &&
            smoothingActive && sourceQpc > 0 && nowQpc >= sourceQpc)
        {
            const int64_t maximumAgeDeadline = sourceQpc + period;
            if (m_deadlineQpc > maximumAgeDeadline)
            {
                CancelTimer();
                m_deadlineQpc = maximumAgeDeadline;
                m_deadlineMissRecorded = false;
                if (nowQpc >= maximumAgeDeadline)
                {
                    ++m_underruns;
                    m_deadlineQpc = nowQpc;
                    m_reanchorOnPresent = true;
                    m_readyBeforeDeadline = false;
                    m_setupWindowOpen = true;
                    return true;
                }
            }
        }

        if (nowQpc >= m_deadlineQpc)
        {
            CancelTimer();
            const int64_t latenessQpc = nowQpc - m_deadlineQpc;
            RecordDeadlineMiss(nowQpc, period);
            // No completed frame reached the pacer before an already
            // established deadline. Treat this as a natural underrun and
            // anchor the next phase to this immediate Present; advancing from
            // the missed deadline would manufacture a catch-up short frame.
            if (!initializedDeadline && latenessQpc > 0 && !m_readyBeforeDeadline)
            {
                ++m_underruns;
                m_deadlineQpc = nowQpc;
                m_reanchorOnPresent = true;
                m_deadlineMissRecorded = false;
            }
            m_readyBeforeDeadline = false;
            m_setupWindowOpen = true;
            return true;
        }

        if (ArmTimer(m_deadlineQpc, includePostPreparation))
        {
            m_readyBeforeDeadline = true;
            return false;
        }

        // A timer failure falls back to immediate presentation, so it must
        // establish a fresh phase instead of leaving a future deadline that
        // would create one abnormally long interval.
        m_deadlineQpc = nowQpc;
        m_reanchorOnPresent = true;
        m_deadlineMissRecorded = false;
        m_readyBeforeDeadline = false;
        m_setupWindowOpen = true;
        return true;
    }

    bool FramePacer::ArmTimer(
        int64_t deadlineQpc,
        bool includePostPreparation) noexcept
    {
        if (!m_timer || deadlineQpc <= 0) return false;
        const double rate = DesiredRateHz(QueryQpc());
        const double cpuBoundedMaximumUs = rate > 0.0
            ? std::clamp(30000.0 / rate, 25.0, 200.0)
            : 200.0;
        m_armedSpinBudgetUs = std::clamp(m_spinBudgetUs, 25.0, cpuBoundedMaximumUs);
        const int64_t wakeTargetQpc = std::max<int64_t>(
            QueryQpc(),
            SetupWakeTargetQpc(deadlineQpc, includePostPreparation));
        return ArmWaitableTimerAt(wakeTargetQpc);
    }

    bool FramePacer::ArmWaitableTimerAt(int64_t wakeTargetQpc) noexcept
    {
        if (!m_timer || wakeTargetQpc <= 0) return false;
        if (m_timerArmed && wakeTargetQpc == m_timerWakeTargetQpc) return true;
        m_timerWakeTargetQpc = wakeTargetQpc;
        const int64_t now = QueryQpc();
        int64_t remainingTicks = std::max<int64_t>(1, m_timerWakeTargetQpc - now);
        LARGE_INTEGER due{};
        due.QuadPart = -std::max<int64_t>(1, static_cast<int64_t>(
            std::llround(static_cast<double>(remainingTicks) * 10000000.0 /
                static_cast<double>(m_qpcFrequency))));
        if (SetWaitableTimerEx(m_timer, &due, 0, nullptr, nullptr, nullptr, 0))
        {
            m_timerArmed = true;
            return true;
        }
        m_timerArmed = false;
        m_timerWakeTargetQpc = 0;
        return false;
    }

    int64_t FramePacer::SetupWakeTargetQpc(
        int64_t deadlineQpc,
        bool includePostPreparation) const noexcept
    {
        const int64_t targetPeriodQpc = TargetPeriodQpc(deadlineQpc);
        const double maximumPostPreparationLeadUs = targetPeriodQpc > 0
            ? std::clamp(
                QpcToMilliseconds(targetPeriodQpc) * 1000.0,
                50.0,
                20000.0)
            : 20000.0;
        return deadlineQpc - MicrosecondsToQpc(
            (includePostPreparation
                ? std::clamp(
                    m_postPreparationLeadUs,
                    50.0,
                    maximumPostPreparationLeadUs)
                : 0.0) +
            std::clamp(m_setupLeadUs, 50.0, 2000.0) +
            std::clamp(m_armedSpinBudgetUs, 25.0, 200.0));
    }

    int64_t FramePacer::FinalWakeTargetQpc(int64_t deadlineQpc) const noexcept
    {
        return deadlineQpc - MicrosecondsToQpc(
            std::clamp(m_armedSpinBudgetUs, 25.0, 200.0));
    }

    void FramePacer::RecordTimerWake(int64_t nowQpc) noexcept
    {
        if (m_timerWakeTargetQpc > 0)
        {
            m_timerOvershootUs.Push(
                nowQpc > m_timerWakeTargetQpc
                    ? static_cast<double>(nowQpc - m_timerWakeTargetQpc) * kMicrosecondsPerSecond /
                        static_cast<double>(m_qpcFrequency)
                    : 0.0);
            const double p95 = Percentile(m_timerOvershootUs, 0.95);
            const double rate = DesiredRateHz(nowQpc);
            const double cpuBoundedMaximumUs = rate > 0.0
                ? std::clamp(30000.0 / rate, 25.0, 200.0)
                : 200.0;
            m_spinBudgetUs = std::clamp(p95 + 25.0, 25.0, cpuBoundedMaximumUs);
        }
        m_timerArmed = false;
        m_timerWakeTargetQpc = 0;
    }

    void FramePacer::RecordDeadlineMiss(int64_t nowQpc, int64_t periodQpc) noexcept
    {
        if (m_deadlineMissRecorded || m_deadlineQpc <= 0 || periodQpc <= 0 ||
            nowQpc <= m_deadlineQpc)
        {
            return;
        }

        // Ignore the normal sub-spin QPC overshoot. The existing quarter-frame
        // threshold classifies scheduler/DXGI stalls rather than nanosecond
        // deadline crossing noise.
        if (nowQpc - m_deadlineQpc > periodQpc / 4)
        {
            ++m_deadlineMisses;
            m_deadlineMissRecorded = true;
        }
    }

    void FramePacer::FinishTimerWait() noexcept
    {
        if (!m_timerArmed) return;
        RecordTimerWake(QueryQpc());
        m_setupWindowOpen = true;
    }

    void FramePacer::ObservePresentationSetup(double durationUs) noexcept
    {
        if (!(durationUs >= 0.0) || durationUs > 10000.0) return;
        m_presentationSetupUs.Push(durationUs);
        // A small fixed guard absorbs the call boundary and unmeasured
        // scheduler hand-off without allowing the setup phase to turn into a
        // second frame buffer.
        m_setupLeadUs = std::clamp(
            Percentile(m_presentationSetupUs, 0.95) + 50.0,
            50.0,
            2000.0);
    }

    void FramePacer::WaitForPresentDeadline(int64_t setupBeginQpc) noexcept
    {
        if (!m_setupWindowOpen || m_deadlineQpc <= 0) return;

        int64_t now = QueryQpc();
        if (setupBeginQpc > 0 && now >= setupBeginQpc)
        {
            ObservePresentationSetup(
                static_cast<double>(now - setupBeginQpc) * kMicrosecondsPerSecond /
                static_cast<double>(m_qpcFrequency));
        }
        // A completion/latency wake or the measured setup itself can cross the
        // absolute deadline. That is an underrun, not phase error to repay with
        // a short next interval. Account it once and anchor the next cadence to
        // the actual Present that follows.
        if (now > m_deadlineQpc && !m_reanchorOnPresent)
        {
            RecordDeadlineMiss(now, TargetPeriodQpc(now));
            ++m_underruns;
            m_reanchorOnPresent = true;
        }

        const int64_t finalWakeTargetQpc = FinalWakeTargetQpc(m_deadlineQpc);
        if (now < finalWakeTargetQpc)
        {
            if (ArmWaitableTimerAt(finalWakeTargetQpc))
            {
                const double remainingMs = static_cast<double>(finalWakeTargetQpc - now) *
                    kMillisecondsPerSecond / static_cast<double>(m_qpcFrequency);
                const DWORD timeoutMs = static_cast<DWORD>(std::clamp(
                    std::ceil(remainingMs) + 5.0,
                    1.0,
                    100.0));
                const DWORD waitResult = WaitForSingleObject(m_timer, timeoutMs);
                now = QueryQpc();
                if (waitResult != WAIT_OBJECT_0 && m_timerArmed)
                {
                    CancelWaitableTimer(m_timer);
                }
                // A timer failure/timeout must not expand the final busy-spin
                // beyond its bounded budget. Yield until the intended final
                // wake target, then use the normal short spin.
                while (now < finalWakeTargetQpc)
                {
                    SwitchToThread();
                    now = QueryQpc();
                }
                RecordTimerWake(now);
            }
            else
            {
                while (now < finalWakeTargetQpc)
                {
                    SwitchToThread();
                    now = QueryQpc();
                }
            }
        }

        while (now < m_deadlineQpc)
        {
            _mm_pause();
            now = QueryQpc();
        }
        RecordDeadlineMiss(now, TargetPeriodQpc(now));
        m_lastDeadlineErrorMs = QpcToMilliseconds(now - m_deadlineQpc);
    }

    void FramePacer::OnPresented(
        int64_t presentQpc,
        int64_t sourceQpc,
        int64_t completionQpc,
        size_t readyDepthAfterPresent,
        bool completedFrameRetention) noexcept
    {
        if (completionQpc > 0 && presentQpc >= completionQpc)
        {
            m_completionWakeMs.Push(QpcToMilliseconds(presentQpc - completionQpc));
        }
        if (sourceQpc > 0 && presentQpc >= sourceQpc)
        {
            m_sourceAgeMs.Push(QpcToMilliseconds(presentQpc - sourceQpc));
        }

        const int64_t period = DesiredPeriodQpc(presentQpc);
        const int64_t previousDeadline = m_deadlineQpc;
        const bool reanchorOnPresent = m_reanchorOnPresent;
        m_lastPresentQpc = presentQpc;
        m_presentImmediately = false;
        m_readyBeforeDeadline = false;
        m_setupWindowOpen = false;
        m_reanchorOnPresent = false;
        if (period <= 0)
        {
            m_deadlineQpc = 0;
            m_deadlineMissRecorded = false;
            return;
        }

        const uint32_t targetQueueDepth = TargetQueueDepth(
            completedFrameRetention);
        const double maximumCorrectionQpc = static_cast<double>(period) * 0.01;
        const bool hasAuthoritativeSourceAge =
            targetQueueDepth > 0 && sourceQpc > 0 && presentQpc >= sourceQpc;
        double sourceAgeCorrectionQpc = 0.0;
        if (hasAuthoritativeSourceAge)
        {
            // An exact depth-one controller can keep lengthening phase until a
            // completed frame remains after every Present.  With a genuinely
            // irregular source that retained frame can already be older than
            // one estimated source period.  Source-Present QPC is the stronger
            // signal: hold only enough temporal lead to absorb local jitter,
            // while the physical ready queue remains bounded to one frame.
            // The retained completion supplies Adaptive's temporal guard, so
            // its source-age target has no extra timer lead. This avoids
            // stacking two buffers while preserving the bounded one-slot
            // smoothing reserve.
            const double desiredSourceAgeQpc =
                static_cast<double>(period) * kAdaptiveLeadFraction;
            const double sourceAgeQpc = static_cast<double>(presentQpc - sourceQpc);
            sourceAgeCorrectionQpc = std::clamp(
                desiredSourceAgeQpc - sourceAgeQpc,
                -maximumCorrectionQpc,
                maximumCorrectionQpc);
        }

        if (reanchorOnPresent)
        {
            // Re-anchor to the actual Present, but retain the same bounded age
            // correction used on the normal path.  This does not repay a late
            // wake; it prevents repeated natural underruns from disabling the
            // one-period latency bound indefinitely.
            m_deadlineQpc = presentQpc + period + static_cast<int64_t>(
                std::llround(sourceAgeCorrectionQpc));
            m_deadlineMissRecorded = false;
            return;
        }

        if (previousDeadline > 0 && presentQpc > previousDeadline + period * 2)
        {
            m_deadlineQpc = presentQpc + period;
            m_deadlineMissRecorded = false;
            ++m_discontinuities;
            return;
        }

        const int queueError = static_cast<int>(targetQueueDepth) -
            static_cast<int>(readyDepthAfterPresent);
        const double bufferCorrectionQpc = hasAuthoritativeSourceAge
            ? sourceAgeCorrectionQpc
            : targetQueueDepth == 0
            ? 0.0
            : static_cast<double>(period) *
                std::clamp(static_cast<double>(queueError) * 0.01, -0.01, 0.01);
        const double phaseCorrectionQpc = previousDeadline > 0
            ? static_cast<double>(previousDeadline - presentQpc)
            : 0.0;
        // Anchor from the actual Present start. Both absolute-phase recovery
        // and queue-depth recovery share one +/-1% correction budget, so a
        // late wake can never manufacture a catch-up short frame.
        const double totalCorrectionQpc = std::clamp(
            phaseCorrectionQpc + bufferCorrectionQpc,
            -maximumCorrectionQpc,
            maximumCorrectionQpc);
        m_deadlineQpc = presentQpc + period + static_cast<int64_t>(
            std::llround(totalCorrectionQpc));
        m_deadlineMissRecorded = false;

    }

    void FramePacer::OnUnderrun(int64_t nowQpc) noexcept
    {
        ++m_underruns;
        (void)nowQpc;
        m_deadlineQpc = 0;
        m_presentImmediately = true;
        m_readyBeforeDeadline = false;
        m_setupWindowOpen = false;
        m_reanchorOnPresent = true;
        m_deadlineMissRecorded = false;
        CancelTimer();
    }

    void FramePacer::CancelTimer() noexcept
    {
        if (m_timer && m_timerArmed) CancelWaitableTimer(m_timer);
        m_timerArmed = false;
        m_timerWakeTargetQpc = 0;
    }

    HANDLE FramePacer::TimerEvent() const noexcept { return m_timer; }
    bool FramePacer::TimerArmed() const noexcept { return m_timerArmed; }
    bool FramePacer::BufferingActive() const noexcept
    {
        return m_config.pacingMode == FramePacingMode::Smooth ||
            (m_config.pacingMode == FramePacingMode::Adaptive && m_adaptiveBuffering);
    }
    uint32_t FramePacer::TargetQueueDepth(
        bool completedFrameRetention) const noexcept
    {
        return BufferingActive() && completedFrameRetention ? 1u : 0u;
    }
    int64_t FramePacer::EstimatedSourcePeriodQpc() const noexcept
    {
        const double sourceCadenceMs = CadenceCenter(m_sourceIntervalsMs);
        if (!(sourceCadenceMs > 0.0)) return 0;
        return std::max<int64_t>(1, static_cast<int64_t>(std::llround(
            sourceCadenceMs * static_cast<double>(m_qpcFrequency) /
            kMillisecondsPerSecond)));
    }
    int64_t FramePacer::TargetPeriodQpc(int64_t nowQpc) const noexcept
    {
        const int64_t schedulingPeriod = DesiredPeriodQpc(nowQpc);
        return schedulingPeriod > 0 ? schedulingPeriod : EstimatedSourcePeriodQpc();
    }
    int64_t FramePacer::DeadlineQpc() const noexcept { return m_deadlineQpc; }
    int64_t FramePacer::QpcFrequency() const noexcept { return m_qpcFrequency; }
    FramePacingSnapshot FramePacer::Snapshot() const noexcept
    {
        FramePacingSnapshot snapshot;
        snapshot.sourceMedianMs = Median(m_sourceIntervalsMs);
        snapshot.sourceMadMs = MedianAbsoluteDeviation(m_sourceIntervalsMs, snapshot.sourceMedianMs);
        snapshot.sourceJitterMs = std::max(
            snapshot.sourceMadMs,
            Median(m_sourceSuccessiveDeltaMs) * 0.5);
        snapshot.completionP95Ms = Percentile(m_completionPeriodsMs, 0.95);
        snapshot.interCompletionP95Ms = Percentile(m_interCompletionPeriodsMs, 0.95);
        const double desiredRate = DesiredRateHz(0);
        snapshot.targetPeriodMs = desiredRate > 0.0 ? 1000.0 / desiredRate : 0.0;
        snapshot.postPreparationLeadUs = m_postPreparationLeadUs;
        snapshot.setupLeadUs = m_setupLeadUs;
        snapshot.timerSpinUs = m_armedSpinBudgetUs;
        snapshot.lastDeadlineErrorMs = m_lastDeadlineErrorMs;
        snapshot.completionWakeP95Ms = Percentile(m_completionWakeMs, 0.95);
        snapshot.sourceAgeP95Ms = Percentile(m_sourceAgeMs, 0.95);
        snapshot.deadlineMisses = m_deadlineMisses;
        snapshot.underruns = m_underruns;
        snapshot.discontinuities = m_discontinuities;
        snapshot.buffering = BufferingActive();
        snapshot.targetQueueDepth = TargetQueueDepth();
        return snapshot;
    }

    int64_t FramePacer::QueryQpc() noexcept
    {
        LARGE_INTEGER value{};
        QueryPerformanceCounter(&value);
        return value.QuadPart;
    }

    int64_t FramePacer::MicrosecondsToQpc(double microseconds) const noexcept
    {
        return static_cast<int64_t>(std::llround(
            microseconds * static_cast<double>(m_qpcFrequency) / kMicrosecondsPerSecond));
    }

    double FramePacer::QpcToMilliseconds(int64_t ticks) const noexcept
    {
        return static_cast<double>(ticks) * kMillisecondsPerSecond /
            static_cast<double>(m_qpcFrequency);
    }

    bool FramePacer::SelfTest(std::string& diagnostic)
    {
        auto fail = [&](const char* message)
        {
            diagnostic = message;
            return false;
        };

        const auto checkStatistics = []<size_t Count>()
        {
            SampleWindow<Count> samples;
            const auto referenceMedian = [](std::array<double, Count> values, size_t count)
            {
                if (count == 0) return 0.0;
                std::sort(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count));
                return (count & 1u) != 0 ? values[count / 2]
                    : 0.5 * (values[count / 2 - 1] + values[count / 2]);
            };
            uint32_t seed = 0x9e3779b9u;
            for (size_t step = 0; step < 512; ++step)
            {
                if (step % 97 == 0)
                {
                    samples.Clear();
                    if (Median(samples) != 0.0 || CadenceCenter(samples) != 0.0 ||
                        Percentile(samples, 0.95) != 0.0 ||
                        MedianAbsoluteDeviation(samples, 0.0) != 0.0) return false;
                }
                seed = seed * 1664525u + 1013904223u;
                samples.Push(static_cast<double>(seed % 200u) * 0.125);
                auto sorted = samples.values;
                std::sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(samples.count));
                const double median = referenceMedian(samples.values, samples.count);
                std::array<double, Count> deviations{};
                std::array<double, Count> pairs{};
                const size_t oldest = samples.count == Count ? samples.cursor : 0;
                for (size_t index = 0; index < samples.count; ++index)
                {
                    deviations[index] = std::abs(samples.values[index] - median);
                    if (index > 0)
                    {
                        pairs[index - 1] = 0.5 * (samples.values[(oldest + index - 1) % Count] +
                            samples.values[(oldest + index) % Count]);
                    }
                }
                const double cadence = samples.count == 1 ? samples.values[0]
                    : referenceMedian(pairs, samples.count - 1);
                for (int repeat = 0; repeat < 3; ++repeat)
                {
                    if (Median(samples) != median || CadenceCenter(samples) != cadence ||
                        MedianAbsoluteDeviation(samples, median) != referenceMedian(deviations, samples.count)) return false;
                    for (double percentile : {0.0, 0.5, 0.95, 1.0})
                    {
                        const size_t index = static_cast<size_t>(std::ceil(percentile * static_cast<double>(samples.count - 1)));
                        if (Percentile(samples, percentile) != sorted[index]) return false;
                    }
                }
                // Invalid data must neither enter a sort nor invalidate/alter
                // the finite history. Zero remains a valid timing sample.
                const size_t cursor = samples.cursor;
                samples.Push(-1.0);
                samples.Push(std::numeric_limits<double>::quiet_NaN());
                samples.Push(std::numeric_limits<double>::infinity());
                if (samples.cursor != cursor || Median(samples) != median) return false;
            }
            return true;
        };
        if (!checkStatistics.operator()<kCadenceWindow>() || !checkStatistics.operator()<kMetricWindow>())
        {
            return fail("Cached cadence/statistics differ from uncached reference across ring wrap or reset");
        }

        FramePacingConfig lowLatencyConfig;
        lowLatencyConfig.pacingMode = FramePacingMode::LowLatency;
        FramePacer lowLatency(lowLatencyConfig);
        const int64_t lowPeriod = std::max<int64_t>(
            1,
            lowLatency.m_qpcFrequency / 120);
        int64_t qpc = lowPeriod;
        for (uint64_t ordinal = 1; ordinal <= 20; ++ordinal)
        {
            lowLatency.ObserveSource(qpc, ordinal);
            lowLatency.ObserveCompletion(qpc + lowPeriod / 4, qpc, 2.0);
            qpc += lowPeriod;
        }
        if (lowLatency.TargetQueueDepth() != 0 ||
            !lowLatency.ReadyToPresent(qpc, 1) ||
            lowLatency.TimerArmed())
        {
            return fail("Low latency introduced buffering or a timer wait");
        }

        FramePacingConfig smoothConfig;
        smoothConfig.pacingMode = FramePacingMode::Smooth;
        FramePacer smooth(smoothConfig);
        qpc = lowPeriod;
        for (uint64_t ordinal = 1; ordinal <= 20; ++ordinal)
        {
            smooth.ObserveSource(qpc, ordinal);
            qpc += lowPeriod;
        }
        if (!smooth.BufferingActive() ||
            smooth.TargetQueueDepth() != 1 ||
            smooth.EstimatedSourcePeriodQpc() <= 0)
        {
            return fail("Smooth did not retain its one-frame source-cadence target");
        }
        if (smooth.ReadyToPresent(qpc, 1) || !smooth.TimerArmed())
        {
            return fail("Smooth did not arm its source-cadence deadline");
        }
        smooth.CancelTimer();

        FramePacingConfig adaptiveConfig;
        adaptiveConfig.pacingMode = FramePacingMode::Adaptive;
        FramePacer adaptive(adaptiveConfig);
        qpc = lowPeriod;
        for (uint64_t ordinal = 1; ordinal <= 24; ++ordinal)
        {
            adaptive.ObserveSource(qpc, ordinal);
            adaptive.ObserveCompletion(qpc + lowPeriod / 5, qpc, 1.0);
            qpc += ordinal % 2 == 0
                ? lowPeriod + lowPeriod / 5
                : lowPeriod - lowPeriod / 5;
        }
        if (!adaptive.BufferingActive() || adaptive.TargetQueueDepth() != 1)
        {
            return fail("Adaptive did not enter bounded smoothing under source jitter");
        }

        const uint64_t discontinuitiesBefore = adaptive.m_discontinuities;
        adaptive.ObserveSource(qpc + lowPeriod * 8, 25);
        if (adaptive.m_discontinuities <= discontinuitiesBefore ||
            adaptive.BufferingActive())
        {
            return fail("Source discontinuity did not reset adaptive state");
        }

        diagnostic =
            "frame pacer uncapped low-latency, bounded adaptive/smooth, source-cadence, and discontinuity gates passed";
        return true;
    }

    const wchar_t* FramePacingModeName(FramePacingMode mode) noexcept
    {
        switch (mode)
        {
        case FramePacingMode::LowLatency: return L"Low latency";
        case FramePacingMode::Adaptive: return L"Adaptive";
        case FramePacingMode::Smooth: return L"Smooth";
        default: return L"Unknown";
        }
    }

}
