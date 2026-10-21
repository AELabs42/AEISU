#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace aess::render
{
    // Render-owner-only caches. No locks, staging surfaces, or image copies.
    // Reset before changing the D3D device or retiring source resources.
    class SourceViewCache
    {
    public:
        ID3D11ShaderResourceView* FindOrCreate(
            ID3D11Device* device,
            ID3D11Texture2D* texture,
            const D3D11_TEXTURE2D_DESC& description) noexcept
        {
            if (!device || !texture) return nullptr;
            if (m_width != description.Width || m_height != description.Height ||
                m_format != description.Format)
            {
                Reset();
                m_width = description.Width;
                m_height = description.Height;
                m_format = description.Format;
            }
            // A non-sampleable texture must not poison other textures with
            // the same dimensions/format (e.g. after switching capture paths).
            if ((description.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0) return nullptr;
            for (const auto& entry : m_entries)
            {
                if (entry.texture.Get() == texture) return entry.view.Get();
            }
            auto& entry = m_entries[m_next];
            m_next = (m_next + 1) % m_entries.size();
            entry.view.Reset();
            entry.texture = texture;
            // A failure is cached for this resource only. Holding a reference
            // prevents pointer reuse from turning a failed probe into an ABA hit.
            if (FAILED(device->CreateShaderResourceView(texture, nullptr, &entry.view)))
            {
                entry.view.Reset();
            }
            return entry.view.Get();
        }

        void Reset() noexcept
        {
            for (auto& entry : m_entries)
            {
                entry.view.Reset();
                entry.texture.Reset();
            }
            m_next = 0;
            m_width = 0;
            m_height = 0;
            m_format = DXGI_FORMAT_UNKNOWN;
        }

    private:
        struct Entry
        {
            Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
            Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> view;
        };
        // Covers the injected/WGC capture rings without an unbounded resource
        // map. Resize/format changes release all previous resource references.
        std::array<Entry, 8> m_entries{};
        size_t m_next = 0;
        UINT m_width = 0;
        UINT m_height = 0;
        DXGI_FORMAT m_format = DXGI_FORMAT_UNKNOWN;
    };

    template <typename Constants, size_t Capacity = 8>
    class ConstantBufferCache
    {
        static_assert(Capacity > 0);
        static_assert(std::is_trivially_copyable_v<Constants>);
        static_assert(sizeof(Constants) % 16 == 0);
        static_assert(sizeof(Constants) <= D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16);

    public:
        HRESULT Resolve(
            ID3D11Device* device,
            ID3D11DeviceContext* context,
            const Constants& constants,
            Microsoft::WRL::ComPtr<ID3D11Buffer>& output) noexcept
        {
            if (!device || !context) return E_INVALIDARG;
            for (const auto& entry : m_entries)
            {
                if (entry.buffer && std::memcmp(&entry.constants, &constants, sizeof(Constants)) == 0)
                {
                    output = entry.buffer;
                    return S_OK;
                }
            }

            auto& entry = m_entries[m_next];
            if (!entry.buffer)
            {
                D3D11_BUFFER_DESC description{};
                description.ByteWidth = static_cast<UINT>(sizeof(Constants));
                description.Usage = D3D11_USAGE_DEFAULT;
                description.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
                D3D11_SUBRESOURCE_DATA data{};
                data.pSysMem = &constants;
                const HRESULT result = device->CreateBuffer(&description, &data, &entry.buffer);
                if (FAILED(result)) return result;
            }
            else
            {
                // The miss path stays bounded even for continuously changing
                // crops. D3D11 retains the existing queued-draw ordering; never
                // map/overwrite memory that the GPU may still be reading.
                context->UpdateSubresource(entry.buffer.Get(), 0, nullptr, &constants, 0, 0);
            }
            entry.constants = constants;
            output = entry.buffer;
            m_next = (m_next + 1) % Capacity;
            return S_OK;
        }

        void Reset() noexcept
        {
            for (auto& entry : m_entries) entry.buffer.Reset();
            m_next = 0;
        }

    private:
        struct Entry
        {
            Constants constants{};
            Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
        };
        std::array<Entry, Capacity> m_entries{};
        size_t m_next = 0;
    };
}
