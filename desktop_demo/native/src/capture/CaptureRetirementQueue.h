#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace aess::capture
{
    // Single render-owner FIFO. Track BEFORE issuing source reads, then signal
    // the reserved fence value AFTER the last read. Releasing a FrameInfo must
    // not release this ownership. No allocations occur when tracking/retiring.
    template <typename Lease, std::size_t Capacity = 8>
    class CaptureRetirementQueue
    {
        static_assert(Capacity > 0);

    public:
        bool Empty() const noexcept { return m_count == 0; }
        bool Full() const noexcept { return m_count == Capacity; }
        std::size_t Size() const noexcept { return m_count; }
        uint64_t FirstFenceValue() const noexcept
        {
            return Empty() ? 0 : m_entries[m_head].fenceValue;
        }

        bool Track(const std::shared_ptr<Lease>& lease, uint64_t fenceValue) noexcept
        {
            if (!lease || Full() || fenceValue == 0 || fenceValue <= m_lastFenceValue) return false;
            m_entries[(m_head + m_count) % Capacity] = { lease, fenceValue };
            m_lastFenceValue = fenceValue;
            ++m_count;
            return true;
        }

        void RetireCompleted(uint64_t completedValue) noexcept
        {
            while (!Empty() && FirstFenceValue() <= completedValue)
            {
                auto& entry = m_entries[m_head];
                entry.lease->Retire();
                entry = {};
                m_head = (m_head + 1) % Capacity;
                --m_count;
            }
        }

        // Only after GPU drain or confirmed device removal. This also handles
        // an exception after Track/read submission but before fence Signal.
        void ResetAfterDrain() noexcept
        {
            RetireCompleted(UINT64_MAX);
            m_head = 0;
            m_lastFenceValue = 0;
        }

    private:
        struct Entry
        {
            std::shared_ptr<Lease> lease;
            uint64_t fenceValue = 0;
        };
        std::array<Entry, Capacity> m_entries{};
        std::size_t m_head = 0;
        std::size_t m_count = 0;
        uint64_t m_lastFenceValue = 0;
    };
}
