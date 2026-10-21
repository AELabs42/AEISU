#pragma once

#include <windows.h>
#include <dxgiformat.h>
#include <dxgicommon.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace aess::capture
{
    constexpr uint32_t kTransportMagic = 0x53455341u; // AESS
    constexpr uint32_t kTransportVersion = 12;
    constexpr uint32_t kTransportSlotCount = 8;
    constexpr uint32_t kColorSpaceDiagnosticTag = 0x43535043u; // CSPC
    constexpr size_t kD3D12ConsumerReuseDeferralDiagnostic = 14;
    constexpr size_t kD3D12ProducerReuseDeferralDiagnostic = 15;
    static_assert(kD3D12ProducerReuseDeferralDiagnostic < 16);

    inline uint64_t EncodeColorSpaceDiagnostic(DXGI_COLOR_SPACE_TYPE colorSpace)
    {
        return (static_cast<uint64_t>(kColorSpaceDiagnosticTag) << 32) |
            static_cast<uint32_t>(colorSpace);
    }

    inline bool DecodeColorSpaceDiagnostic(uint64_t diagnostic, DXGI_COLOR_SPACE_TYPE& colorSpace)
    {
        if (static_cast<uint32_t>(diagnostic >> 32) != kColorSpaceDiagnosticTag)
        {
            return false;
        }
        colorSpace = static_cast<DXGI_COLOR_SPACE_TYPE>(static_cast<uint32_t>(diagnostic));
        return true;
    }

    struct SharedControl
    {
        uint32_t magic = kTransportMagic;
        uint32_t version = kTransportVersion;
        uint32_t producerPid = 0;
        uint32_t generation = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t format = static_cast<uint32_t>(DXGI_FORMAT_UNKNOWN);
        uint32_t slotCount = kTransportSlotCount;
        alignas(8) volatile LONG64 targetWindow = 0;
        uint64_t fenceHandle = 0;
        uint64_t textureHandles[kTransportSlotCount]{};
        // Metadata belongs to the sequence occupying the corresponding ring
        // slot. The producer publishes both values before signaling that
        // sequence's GPU fence and advancing latestSequence.
        alignas(8) volatile LONG64 sourcePresentQpc[kTransportSlotCount]{};
        alignas(8) volatile LONG64 sourcePresentOrdinal[kTransportSlotCount]{};
        alignas(8) volatile LONG64 latestSequence = 0;
        // Every Present from the selected source swap chain, including frames
        // deliberately rejected by the bounded capture-admission gate.  This
        // is the true application render rate; latestSequence/capturedFrames
        // remain the rate at which full-resolution copies were admitted.
        alignas(8) volatile LONG64 sourcePresentFrames = 0;
        // The consumer acknowledges through the newest sequence whose reads
        // have completed on its D3D11 context. The producer may fill the
        // bounded ring but may never overwrite a slot still in consumer use.
        alignas(8) volatile LONG64 consumedSequence = 0;
        alignas(8) volatile LONG64 capturedFrames = 0;
        alignas(8) volatile LONG64 droppedFrames = 0;
        alignas(8) volatile LONG64 backpressuredFrames = 0;
        uint64_t diagnostics[16]{};
    };

    // Transport V25 / ABI v12 is consumed from a separately built injected
    // DLL and the AESS process. Keep every cross-process offset explicit so a
    // seemingly harmless field insertion or alignment change fails at build
    // time instead of corrupting live capture metadata.
    static_assert(kTransportVersion == 12);
    static_assert(kTransportSlotCount == 8);
    static_assert(std::is_standard_layout_v<SharedControl>);
    static_assert(alignof(SharedControl) == 8);
    static_assert(sizeof(SharedControl) == 416);
    static_assert(offsetof(SharedControl, magic) == 0);
    static_assert(offsetof(SharedControl, version) == 4);
    static_assert(offsetof(SharedControl, producerPid) == 8);
    static_assert(offsetof(SharedControl, generation) == 12);
    static_assert(offsetof(SharedControl, width) == 16);
    static_assert(offsetof(SharedControl, height) == 20);
    static_assert(offsetof(SharedControl, format) == 24);
    static_assert(offsetof(SharedControl, slotCount) == 28);
    static_assert(offsetof(SharedControl, targetWindow) == 32);
    static_assert(offsetof(SharedControl, fenceHandle) == 40);
    static_assert(offsetof(SharedControl, textureHandles) == 48);
    static_assert(offsetof(SharedControl, sourcePresentQpc) == 112);
    static_assert(offsetof(SharedControl, sourcePresentOrdinal) == 176);
    static_assert(offsetof(SharedControl, latestSequence) == 240);
    static_assert(offsetof(SharedControl, sourcePresentFrames) == 248);
    static_assert(offsetof(SharedControl, consumedSequence) == 256);
    static_assert(offsetof(SharedControl, capturedFrames) == 264);
    static_assert(offsetof(SharedControl, droppedFrames) == 272);
    static_assert(offsetof(SharedControl, backpressuredFrames) == 280);
    static_assert(offsetof(SharedControl, diagnostics) == 288);

    inline uint32_t AdvanceTransportGeneration(uint32_t& generation) noexcept
    {
        ++generation;
        // Zero is the transport's explicit unavailable state. Avoid exposing
        // it even after the theoretical 32-bit generation wrap.
        if (generation == 0)
        {
            generation = 1;
        }
        return generation;
    }

    inline void PublishTransportGeneration(
        SharedControl& control,
        uint32_t generation) noexcept
    {
        InterlockedExchange(
            reinterpret_cast<volatile LONG*>(&control.generation),
            static_cast<LONG>(generation));
    }

    inline LONG64 AdvanceSourcePresentOrdinal(SharedControl& control) noexcept
    {
        return InterlockedIncrement64(&control.sourcePresentFrames);
    }

    constexpr bool CaptureRingIsBackpressured(
        uint64_t latestSequence,
        uint64_t consumedSequence) noexcept
    {
        return latestSequence >= consumedSequence &&
            latestSequence - consumedSequence >= kTransportSlotCount;
    }

    constexpr uint64_t CaptureRingReusableSequence(
        uint64_t nextSequence) noexcept
    {
        return nextSequence > kTransportSlotCount
            ? nextSequence - kTransportSlotCount
            : 0;
    }

    // A D3D12 capture slot owns both a command allocator/list and the shared
    // texture bearing the same ring index. Reuse is legal only after the
    // producer copy has completed and the consumer has acknowledged that all
    // of its GPU reads have completed. The initial ring fill has no prior
    // owner. Keeping this rule independent of D3D12 makes the nonblocking
    // source-Present admission path exhaustively testable.
    constexpr bool CaptureRingSlotCanBeReused(
        uint64_t nextSequence,
        uint64_t consumedSequence,
        uint64_t producerCompletedSequence) noexcept
    {
        if (nextSequence == 0)
        {
            return false;
        }
        const uint64_t reusableSequence =
            CaptureRingReusableSequence(nextSequence);
        return reusableSequence == 0 ||
            (consumedSequence >= reusableSequence &&
             producerCompletedSequence >= reusableSequence);
    }

    struct SourceTimingReadWindow
    {
        uint64_t firstSequence = 0;
        uint64_t lastReadySequence = 0;
        uint64_t overflowCount = 0;
    };

    // Defines the bounded per-slot timing journal semantics independently of
    // GPU texture acquisition. A lagging timing reader resumes at the oldest
    // sequence that can still occupy the ring and accounts for every entry
    // overwritten since its prior cursor.
    constexpr SourceTimingReadWindow ComputeSourceTimingReadWindow(
        uint64_t latestSequence,
        uint64_t completedSequence,
        uint64_t lastTimingSequence) noexcept
    {
        if (latestSequence == 0)
        {
            return {};
        }
        const uint64_t oldestAvailable = latestSequence >= kTransportSlotCount
            ? latestSequence - kTransportSlotCount + 1
            : 1;
        const uint64_t requested = lastTimingSequence == UINT64_MAX
            ? UINT64_MAX
            : lastTimingSequence + 1;
        const uint64_t first = requested < oldestAvailable
            ? oldestAvailable
            : requested;
        return SourceTimingReadWindow{
            first,
            completedSequence < latestSequence
                ? completedSequence
                : latestSequence,
            requested < oldestAvailable
                ? oldestAvailable - requested
                : 0,
        };
    }

    // Publish one admitted capture sequence. The callback is the actual D3D
    // fence signal, giving tests an observation point that proves timestamp
    // and ordinal stores happen before the fence and that latestSequence is
    // not visible until the signal succeeds.
    template <typename SignalFence>
    HRESULT PublishCapturedSequence(
        SharedControl& control,
        uint64_t sequence,
        LONG64 sourcePresentQpc,
        LONG64 sourcePresentOrdinal,
        SignalFence&& signalFence)
    {
        if (sequence == 0 || sourcePresentQpc <= 0 || sourcePresentOrdinal <= 0)
        {
            return E_INVALIDARG;
        }
        const uint32_t slot = static_cast<uint32_t>(
            sequence % kTransportSlotCount);
        InterlockedExchange64(
            &control.sourcePresentQpc[slot],
            sourcePresentQpc);
        InterlockedExchange64(
            &control.sourcePresentOrdinal[slot],
            sourcePresentOrdinal);
        const HRESULT signalResult = std::forward<SignalFence>(signalFence)(sequence);
        if (FAILED(signalResult))
        {
            return signalResult;
        }
        InterlockedExchange64(
            &control.capturedFrames,
            static_cast<LONG64>(sequence));
        InterlockedExchange64(
            &control.latestSequence,
            static_cast<LONG64>(sequence));
        return S_OK;
    }

    inline std::wstring ControlName(DWORD pid)
    {
        return L"Local\\AESS.Capture.V25.Control." + std::to_wstring(pid);
    }

    inline std::wstring FrameEventName(DWORD pid)
    {
        return L"Local\\AESS.Capture.V25.Frame." + std::to_wstring(pid);
    }

    inline std::wstring FenceName(DWORD pid, uint32_t generation)
    {
        return L"Local\\AESS.Capture.V25.Fence." + std::to_wstring(pid) + L"." + std::to_wstring(generation);
    }

    inline std::wstring TextureName(DWORD pid, uint32_t generation, uint32_t slot)
    {
        return L"Local\\AESS.Capture.V25.Texture." + std::to_wstring(pid) + L"." +
            std::to_wstring(generation) + L"." + std::to_wstring(slot);
    }
}
