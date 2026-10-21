#include "AESSCaptureTransport.h"

#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <MinHook.h>

#include <array>
#include <atomic>
#include <cstring>

using Microsoft::WRL::ComPtr;
using namespace aess::capture;

namespace
{
    using PresentFn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
    using Present1Fn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
    using ResizeBuffersFn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT, UINT, DXGI_FORMAT, UINT);
    using ResizeBuffers1Fn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT, UINT, DXGI_FORMAT, UINT, const UINT*, IUnknown* const*);
    using SetColorSpaceFn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain3*, DXGI_COLOR_SPACE_TYPE);

    PresentFn g_originalPresent = nullptr;
    Present1Fn g_originalPresent1 = nullptr;
    ResizeBuffersFn g_originalResizeBuffers = nullptr;
    ResizeBuffers1Fn g_originalResizeBuffers1 = nullptr;
    SetColorSpaceFn g_originalSetColorSpace = nullptr;
    using ExecuteFn = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);
    ExecuteFn g_originalExecute = nullptr;
    ComPtr<ID3D12CommandQueue> g_observedD3D12Queue;
    SRWLOCK g_captureLock = SRWLOCK_INIT;
    SRWLOCK g_queueLock = SRWLOCK_INIT;
    thread_local bool g_inCapture = false;
    HMODULE g_module = nullptr;
    SRWLOCK g_colorSpaceLock = SRWLOCK_INIT;
    struct SwapChainColorSpace
    {
        // COM interface pointers for the same object are not required to have
        // the same address.  Store the canonical IUnknown identity so the
        // IDXGISwapChain3 pointer seen by SetColorSpace1 matches the
        // IDXGISwapChain pointer seen by Present/Present1.
        IUnknown* identity = nullptr;
        DXGI_COLOR_SPACE_TYPE colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
    };
    std::array<SwapChainColorSpace, 8> g_colorSpaces{};
    size_t g_nextColorSpace = 0;

    struct Transport
    {
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        ComPtr<ID3D11DeviceContext4> context4;
        ComPtr<ID3D11Fence> fence;
        std::array<ComPtr<ID3D11Texture2D>, kTransportSlotCount> textures;
        std::array<HANDLE, kTransportSlotCount> textureHandles{};
        HANDLE fenceHandle = nullptr;
        HANDLE mapping = nullptr;
        HANDLE frameEvent = nullptr;
        SharedControl* control = nullptr;
        D3D11_TEXTURE2D_DESC sourceDesc{};
        IDXGISwapChain* sourceSwapChain = nullptr;
        ComPtr<IDXGISwapChain3> sourceSwapChain3;
        ComPtr<ID3D11Device> sourceD3D11Device;
        ComPtr<ID3D12Device> sourceD3D12Device;
        HWND sourceOutputWindow = nullptr;
        HWND sourceRequestedWindow = nullptr;
        std::atomic<IUnknown*> sourceIdentity{ nullptr };
        std::atomic<uint64_t> sourceColorSpaceDiagnostic{
            EncodeColorSpaceDiagnostic(
                DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709) };
        std::atomic<uint64_t> publishedSourceColorSpaceDiagnostic{ UINT64_MAX };
        std::atomic<uint32_t> publishedSourceColorSpaceGeneration{ 0 };
        uint32_t generation = 0;
        uint64_t sequence = 0;
        std::array<ComPtr<ID3D11Texture2D>, 8> sourceBackBuffers11{};
        std::array<ComPtr<ID3D12Resource>, 8> sourceBackBuffers12{};
        bool sourceDescValid = false;

        void InvalidateBackBuffers() noexcept
        {
            for (auto& b : sourceBackBuffers11) b.Reset();
            for (auto& b : sourceBackBuffers12) b.Reset();
            sourceDescValid = false;
        }

        void ResetGpu() noexcept
        {
            InvalidateBackBuffers();
            for (HANDLE& handle : textureHandles)
            {
                // GetSharedHandle returns a legacy KMT shared handle, not an NT
                // handle owned by this process. It must not be passed to CloseHandle.
                handle = nullptr;
            }
            if (fenceHandle) CloseHandle(fenceHandle);
            fenceHandle = nullptr;
            for (auto& texture : textures) texture.Reset();
            fence.Reset();
            context4.Reset();
            context.Reset();
            device.Reset();
            sourceDesc = {};
        }

        ~Transport()
        {
            ResetGpu();
            if (control) UnmapViewOfFile(control);
            if (mapping) CloseHandle(mapping);
            if (frameEvent) CloseHandle(frameEvent);
        }
    };

    Transport g_transport;

    struct D3D12Transport
    {
        ComPtr<ID3D12Device> device;
        ComPtr<ID3D12CommandQueue> queue;
        ComPtr<ID3D12Fence> fence;
        std::array<ComPtr<ID3D12Resource>, kTransportSlotCount> textures;
        std::array<ComPtr<ID3D12CommandAllocator>, kTransportSlotCount> allocators;
        std::array<ComPtr<ID3D12GraphicsCommandList>, kTransportSlotCount> lists;
        std::array<HANDLE, kTransportSlotCount> textureHandles{};
        HANDLE fenceHandle = nullptr;
        D3D12_RESOURCE_DESC sourceDesc{};
        uint64_t sequence = 0;
        uint32_t generation = 0;

        void Reset() noexcept
        {
            for (HANDLE& handle : textureHandles)
            {
                if (handle) CloseHandle(handle);
                handle = nullptr;
            }
            if (fenceHandle) CloseHandle(fenceHandle);
            fenceHandle = nullptr;
            for (auto& list : lists) list.Reset();
            for (auto& allocator : allocators) allocator.Reset();
            for (auto& texture : textures) texture.Reset();
            fence.Reset();
            queue.Reset();
            device.Reset();
            sourceDesc = {};
        }
        ~D3D12Transport() { Reset(); }
    };

    D3D12Transport g_d3d12;

    bool SameSource(const D3D11_TEXTURE2D_DESC& a, const D3D11_TEXTURE2D_DESC& b)
    {
        return a.Width == b.Width && a.Height == b.Height && a.Format == b.Format &&
            a.SampleDesc.Count == b.SampleDesc.Count;
    }

    IUnknown* CanonicalIdentity(IUnknown* object)
    {
        if (!object) return nullptr;
        IUnknown* identity = nullptr;
        if (FAILED(object->QueryInterface(IID_PPV_ARGS(&identity))))
        {
            return nullptr;
        }
        // Retain only the stable identity value.  The swap chain itself owns
        // the lifetime while either hooked method can be executing.
        identity->Release();
        return identity;
    }

    DXGI_COLOR_SPACE_TYPE ResolveSwapChainColorSpace(
        IDXGISwapChain* swapChain,
        IUnknown* identity)
    {
        DXGI_COLOR_SPACE_TYPE colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        bool known = false;
        AcquireSRWLockShared(&g_colorSpaceLock);
        for (const auto& entry : g_colorSpaces)
        {
            if (identity && entry.identity == identity)
            {
                colorSpace = entry.colorSpace;
                known = true;
                break;
            }
        }
        ReleaseSRWLockShared(&g_colorSpaceLock);

        if (!known)
        {
            DXGI_SWAP_CHAIN_DESC desc{};
            if (SUCCEEDED(swapChain->GetDesc(&desc)) &&
                desc.BufferDesc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT)
            {
                colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
            }
            else if (desc.BufferDesc.Format == DXGI_FORMAT_R10G10B10A2_UNORM)
            {
                // HDR10 games often establish SetColorSpace1 before AESS is
                // injected. A 10-bit swap chain is the reliable fallback for
                // that missed event; ordinary 8-bit games remain SDR.
                colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
            }
        }

        return colorSpace;
    }

    void SelectSourceColorSpace(IDXGISwapChain* swapChain)
    {
        if (!swapChain) return;
        IUnknown* identity = CanonicalIdentity(swapChain);
        const uint64_t encoded = EncodeColorSpaceDiagnostic(
            ResolveSwapChainColorSpace(swapChain, identity));
        g_transport.sourceColorSpaceDiagnostic.store(
            encoded, std::memory_order_release);
        g_transport.sourceIdentity.store(identity, std::memory_order_release);
    }

    void PublishSourceColorSpace() noexcept
    {
        for (;;)
        {
            SharedControl* const control = g_transport.control;
            if (!control) return;
            const uint32_t generation = control->generation;
            if (generation == 0) return;

            const uint64_t encoded = g_transport.sourceColorSpaceDiagnostic.load(
                std::memory_order_acquire);
            if (g_transport.publishedSourceColorSpaceGeneration.load(
                    std::memory_order_acquire) == generation &&
                g_transport.publishedSourceColorSpaceDiagnostic.load(
                    std::memory_order_relaxed) == encoded)
            {
                return;
            }

            InterlockedExchange64(
                reinterpret_cast<volatile LONG64*>(&control->diagnostics[0]),
                static_cast<LONG64>(encoded));

            // A concurrent SetColorSpace1 may have selected a newer value
            // after our initial load. Repair that race before marking the
            // publication cached, so the unchanged-Present fast path can
            // safely avoid its former atomic exchange.
            if (control != g_transport.control ||
                control->generation != generation ||
                g_transport.sourceColorSpaceDiagnostic.load(
                    std::memory_order_acquire) != encoded)
            {
                continue;
            }
            g_transport.publishedSourceColorSpaceDiagnostic.store(
                encoded, std::memory_order_relaxed);
            g_transport.publishedSourceColorSpaceGeneration.store(
                generation, std::memory_order_release);
            return;
        }
    }

    HRESULT STDMETHODCALLTYPE HookSetColorSpace(
        IDXGISwapChain3* swapChain,
        DXGI_COLOR_SPACE_TYPE colorSpace)
    {
        const HRESULT hr = g_originalSetColorSpace(swapChain, colorSpace);
        if (SUCCEEDED(hr))
        {
            IUnknown* identity = CanonicalIdentity(swapChain);
            if (!identity) return hr;
            AcquireSRWLockExclusive(&g_colorSpaceLock);
            SwapChainColorSpace* target = nullptr;
            for (auto& entry : g_colorSpaces)
            {
                if (entry.identity == identity)
                {
                    target = &entry;
                    break;
                }
            }
            if (!target)
            {
                target = &g_colorSpaces[g_nextColorSpace++ % g_colorSpaces.size()];
            }
            target->identity = identity;
            target->colorSpace = colorSpace;
            ReleaseSRWLockExclusive(&g_colorSpaceLock);

            if (g_transport.sourceIdentity.load(std::memory_order_acquire) == identity)
            {
                const uint64_t encoded = EncodeColorSpaceDiagnostic(colorSpace);
                g_transport.sourceColorSpaceDiagnostic.store(
                    encoded, std::memory_order_release);
                PublishSourceColorSpace();
            }
        }
        return hr;
    }

    bool SameSource(const D3D12_RESOURCE_DESC& a, const D3D12_RESOURCE_DESC& b)
    {
        return a.Width == b.Width && a.Height == b.Height && a.Format == b.Format &&
            a.SampleDesc.Count == b.SampleDesc.Count;
    }

    bool EnsureControl()
    {
        if (g_transport.control) return true;
        const DWORD pid = GetCurrentProcessId();
        const std::wstring mappingName = ControlName(pid);
        g_transport.mapping = CreateFileMappingW(
            INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
            static_cast<DWORD>(sizeof(SharedControl)), mappingName.c_str());
        if (!g_transport.mapping) return false;
        const bool existing = GetLastError() == ERROR_ALREADY_EXISTS;
        g_transport.control = static_cast<SharedControl*>(MapViewOfFile(
            g_transport.mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedControl)));
        if (!g_transport.control) return false;
        if (!existing)
        {
            std::memset(g_transport.control, 0, sizeof(SharedControl));
            g_transport.control->magic = kTransportMagic;
            g_transport.control->version = kTransportVersion;
            g_transport.control->producerPid = pid;
            g_transport.control->slotCount = kTransportSlotCount;
        }
        else if (g_transport.control->magic != kTransportMagic ||
                 g_transport.control->version != kTransportVersion ||
                 g_transport.control->producerPid != pid)
        {
            return false;
        }
        g_transport.frameEvent = CreateEventW(nullptr, FALSE, FALSE, FrameEventName(pid).c_str());
        return g_transport.frameEvent != nullptr;
    }

    bool CreateTransport(ID3D11Device* device, const D3D11_TEXTURE2D_DESC& sourceDesc)
    {
        if (!EnsureControl()) return false;
        PublishTransportGeneration(*g_transport.control, 0);
        g_transport.control->diagnostics[12] = 1;
        g_transport.ResetGpu();
        g_transport.device = device;
        device->GetImmediateContext(&g_transport.context);
        if (FAILED(g_transport.context.As(&g_transport.context4))) return false;

        ComPtr<ID3D11Device5> device5;
        g_transport.control->diagnostics[12] = 2;
        if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device5)))) return false;
        g_transport.control->diagnostics[12] = 3;
        if (FAILED(device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&g_transport.fence)))) return false;

        AdvanceTransportGeneration(g_transport.generation);
        const DWORD pid = GetCurrentProcessId();
        const std::wstring fenceName = FenceName(pid, g_transport.generation);
        g_transport.control->diagnostics[12] = 4;
        if (FAILED(g_transport.fence->CreateSharedHandle(
            nullptr, GENERIC_ALL, fenceName.c_str(), &g_transport.fenceHandle))) return false;

        D3D11_TEXTURE2D_DESC desc = sourceDesc;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        // Legacy KMT sharing does not impose keyed-mutex ownership. The shared
        // D3D11 fence below is the sole cross-process synchronization primitive.
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        for (uint32_t slot = 0; slot < kTransportSlotCount; ++slot)
        {
            g_transport.control->diagnostics[12] = 10 + slot * 3;
            if (FAILED(device->CreateTexture2D(&desc, nullptr, &g_transport.textures[slot]))) return false;
            ComPtr<IDXGIResource> resource;
            g_transport.control->diagnostics[12] = 11 + slot * 3;
            if (FAILED(g_transport.textures[slot].As(&resource))) return false;
            g_transport.control->diagnostics[12] = 12 + slot * 3;
            if (FAILED(resource->GetSharedHandle(&g_transport.textureHandles[slot]))) return false;
        }

        g_transport.sourceDesc = sourceDesc;
        g_transport.sequence = 0;
        g_transport.control->width = sourceDesc.Width;
        g_transport.control->height = sourceDesc.Height;
        g_transport.control->format = static_cast<uint32_t>(sourceDesc.Format);
        g_transport.control->fenceHandle = reinterpret_cast<uint64_t>(g_transport.fenceHandle);
        g_transport.control->diagnostics[13] = 1; // D3D11 legacy KMT handles
        for (uint32_t slot = 0; slot < kTransportSlotCount; ++slot)
            g_transport.control->textureHandles[slot] = reinterpret_cast<uint64_t>(g_transport.textureHandles[slot]);
        InterlockedExchange64(&g_transport.control->latestSequence, 0);
        InterlockedExchange64(&g_transport.control->consumedSequence, 0);
        PublishTransportGeneration(
            *g_transport.control,
            g_transport.generation);
        g_transport.control->diagnostics[12] = 100;
        SetEvent(g_transport.frameEvent);
        return true;
    }

    bool CreateD3D12Transport(
        ID3D12Device* device,
        ID3D12CommandQueue* queue,
        const D3D12_RESOURCE_DESC& sourceDesc)
    {
        if (!EnsureControl()) return false;
        PublishTransportGeneration(*g_transport.control, 0);
        g_d3d12.Reset();
        g_d3d12.device = device;
        g_d3d12.queue = queue;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&g_d3d12.fence)))) return false;
        g_d3d12.generation = AdvanceTransportGeneration(
            g_transport.generation);
        const DWORD pid = GetCurrentProcessId();
        const std::wstring fenceName = FenceName(pid, g_d3d12.generation);
        if (FAILED(device->CreateSharedHandle(
            g_d3d12.fence.Get(), nullptr, GENERIC_ALL, fenceName.c_str(), &g_d3d12.fenceHandle))) return false;

        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap.CreationNodeMask = 1;
        heap.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC desc = sourceDesc;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        // D3D12_HEAP_FLAG_SHARED Texture2D resources must be render-target or
        // depth-stencil resources to be opened by D3D11.  The source formats
        // used by DXGI swap chains are render-target capable.
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
            D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
        for (uint32_t slot = 0; slot < kTransportSlotCount; ++slot)
        {
            if (FAILED(device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON,
                nullptr, IID_PPV_ARGS(&g_d3d12.textures[slot])))) return false;
            const std::wstring name = TextureName(pid, g_d3d12.generation, slot);
            if (FAILED(device->CreateSharedHandle(
                g_d3d12.textures[slot].Get(), nullptr, GENERIC_ALL,
                name.c_str(), &g_d3d12.textureHandles[slot]))) return false;
            if (FAILED(device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g_d3d12.allocators[slot])))) return false;
            if (FAILED(device->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_d3d12.allocators[slot].Get(),
                nullptr, IID_PPV_ARGS(&g_d3d12.lists[slot])))) return false;
            g_d3d12.lists[slot]->Close();
        }
        g_d3d12.sourceDesc = sourceDesc;
        g_d3d12.sequence = 0;
        g_transport.control->diagnostics[12] = 200;
        g_transport.control->width = static_cast<uint32_t>(sourceDesc.Width);
        g_transport.control->height = sourceDesc.Height;
        g_transport.control->format = static_cast<uint32_t>(sourceDesc.Format);
        g_transport.control->fenceHandle = reinterpret_cast<uint64_t>(g_d3d12.fenceHandle);
        g_transport.control->diagnostics[13] = 2; // D3D12 NT handles
        for (uint32_t slot = 0; slot < kTransportSlotCount; ++slot)
            g_transport.control->textureHandles[slot] = reinterpret_cast<uint64_t>(g_d3d12.textureHandles[slot]);
        InterlockedExchange64(&g_transport.control->latestSequence, 0);
        InterlockedExchange64(&g_transport.control->consumedSequence, 0);
        PublishTransportGeneration(
            *g_transport.control,
            g_d3d12.generation);
        SetEvent(g_transport.frameEvent);
        return true;
    }

    HWND RequestedCaptureWindow() noexcept
    {
        if (!g_transport.control) return nullptr;
        return reinterpret_cast<HWND>(static_cast<uintptr_t>(
            InterlockedCompareExchange64(
                &g_transport.control->targetWindow, 0, 0)));
    }

    bool ValidateAndSelectSwapChain(
        IDXGISwapChain* swapChain,
        bool knownSource)
    {
        const HWND requestedWindow = RequestedCaptureWindow();
        if (knownSource &&
            g_transport.sourceOutputWindow &&
            g_transport.sourceRequestedWindow == requestedWindow)
        {
            // OutputWindow and process ownership are immutable swap-chain
            // selection properties. Keep only the visibility check on the
            // steady-state Present path; target changes fall through to the
            // complete source-selection validation below.
            if (!IsWindowVisible(g_transport.sourceOutputWindow))
            {
                g_transport.control->diagnostics[11] = 3;
                return false;
            }
            return true;
        }

        DXGI_SWAP_CHAIN_DESC swapDesc{};
        if (FAILED(swapChain->GetDesc(&swapDesc)))
        {
            g_transport.control->diagnostics[11] = 1;
            return false;
        }
        g_transport.control->diagnostics[10] =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(swapDesc.OutputWindow));
        if (!swapDesc.OutputWindow)
        {
            g_transport.control->diagnostics[11] = 2;
            return false;
        }
        if (!IsWindowVisible(swapDesc.OutputWindow))
        {
            g_transport.control->diagnostics[11] = 3;
            return false;
        }
        if (requestedWindow)
        {
            const HWND candidateRoot = GetAncestor(swapDesc.OutputWindow, GA_ROOT);
            const HWND requestedRoot = GetAncestor(requestedWindow, GA_ROOT);
            if (swapDesc.OutputWindow != requestedWindow &&
                candidateRoot != requestedRoot)
            {
                g_transport.control->diagnostics[11] = 4;
                return false;
            }
        }
        DWORD windowPid = 0;
        GetWindowThreadProcessId(swapDesc.OutputWindow, &windowPid);
        RECT windowRect{};
        GetClientRect(swapDesc.OutputWindow, &windowRect);
        if (windowPid != GetCurrentProcessId())
        {
            g_transport.control->diagnostics[11] = 5;
            return false;
        }
        if (windowRect.right - windowRect.left < 640 ||
            windowRect.bottom - windowRect.top < 360)
        {
            g_transport.control->diagnostics[11] = 6;
            return false;
        }

        const bool sourceChanged = g_transport.sourceSwapChain != swapChain;
        g_transport.sourceSwapChain = swapChain;
        g_transport.sourceOutputWindow = swapDesc.OutputWindow;
        g_transport.sourceRequestedWindow = requestedWindow;
        if (sourceChanged)
        {
            g_transport.InvalidateBackBuffers();
            g_transport.sourceSwapChain3.Reset();
            g_transport.sourceD3D11Device.Reset();
            g_transport.sourceD3D12Device.Reset();
            (void)swapChain->QueryInterface(
                IID_PPV_ARGS(&g_transport.sourceSwapChain3));
            if (FAILED(swapChain->GetDevice(
                    IID_PPV_ARGS(&g_transport.sourceD3D11Device))))
            {
                (void)swapChain->GetDevice(
                    IID_PPV_ARGS(&g_transport.sourceD3D12Device));
            }
            SelectSourceColorSpace(swapChain);
        }
        return true;
    }

    bool CaptureD3D12(
        IDXGISwapChain* swapChain,
        LONG64 sourcePresentQpc,
        LONG64 sourcePresentOrdinal)
    {
        ID3D12Device* device = g_transport.sourceD3D12Device.Get();
        ComPtr<ID3D12Device> fallbackDevice;
        if (!device &&
            FAILED(swapChain->GetDevice(IID_PPV_ARGS(&fallbackDevice))))
        {
            g_transport.control->diagnostics[12] = 101;
            return false;
        }
        if (!device) device = fallbackDevice.Get();
        UINT backBufferIndex = 0;
        if (g_transport.sourceSwapChain3)
            backBufferIndex = g_transport.sourceSwapChain3->GetCurrentBackBufferIndex();
        ID3D12Resource* backBuffer = nullptr;
        const bool hasCachedBackBuffer = (backBufferIndex < g_transport.sourceBackBuffers12.size()) &&
                                         (g_transport.sourceBackBuffers12[backBufferIndex] != nullptr) &&
                                         (g_d3d12.device && g_d3d12.device.Get() == device);
        if (hasCachedBackBuffer)
        {
            backBuffer = g_transport.sourceBackBuffers12[backBufferIndex].Get();
        }
        else
        {
            ComPtr<ID3D12Resource> newBackBuffer;
            if (FAILED(swapChain->GetBuffer(backBufferIndex, IID_PPV_ARGS(&newBackBuffer))))
            {
                g_transport.control->diagnostics[12] = 102;
                return false;
            }
            if (backBufferIndex < g_transport.sourceBackBuffers12.size())
            {
                g_transport.sourceBackBuffers12[backBufferIndex] = newBackBuffer;
            }
            backBuffer = newBackBuffer.Get();
        }

        ComPtr<ID3D12CommandQueue> observedQueue;
        ID3D12CommandQueue* queue = nullptr;
        if (g_d3d12.device.Get() == device && g_d3d12.queue)
        {
            queue = g_d3d12.queue.Get();
        }
        else
        {
            AcquireSRWLockShared(&g_queueLock);
            observedQueue = g_observedD3D12Queue;
            ReleaseSRWLockShared(&g_queueLock);
            if (!observedQueue)
            {
                g_transport.control->diagnostics[12] = 103;
                return false;
            }
            ComPtr<ID3D12Device> queueDevice;
            if (FAILED(observedQueue->GetDevice(IID_PPV_ARGS(&queueDevice))) ||
                queueDevice.Get() != device)
            {
                g_transport.control->diagnostics[12] = 104;
                return false;
            }
            queue = observedQueue.Get();
        }

        if (!hasCachedBackBuffer)
        {
            const D3D12_RESOURCE_DESC desc = backBuffer->GetDesc();
            if (desc.SampleDesc.Count != 1) return false;
            if (!g_d3d12.device || g_d3d12.device.Get() != device ||
                !SameSource(g_d3d12.sourceDesc, desc))
            {
                if (!CreateD3D12Transport(device, queue, desc))
                {
                    g_transport.control->diagnostics[12] = 105;
                    return false;
                }
            }
        }

        const uint64_t sequence = g_d3d12.sequence + 1;
        const uint32_t slot = static_cast<uint32_t>(sequence % kTransportSlotCount);
        const uint64_t reusableSequence =
            CaptureRingReusableSequence(sequence);
        const uint64_t consumedSequence = static_cast<uint64_t>(
            InterlockedCompareExchange64(
                &g_transport.control->consumedSequence, 0, 0));
        const uint64_t producerCompletedSequence =
            g_d3d12.fence->GetCompletedValue();
        if (!CaptureRingSlotCanBeReused(
                sequence,
                consumedSequence,
                producerCompletedSequence))
        {
            // Never wait inside the source application's Present. A consumer
            // acknowledgement is published only after its GPU reads complete,
            // which necessarily follows this producer fence. The second branch
            // is therefore an invariant diagnostic, not an expected queueing
            // path; either case safely drops this admission and retries the same
            // sequence on a later Present without resetting an in-flight list.
            if (reusableSequence != 0 &&
                consumedSequence < reusableSequence)
            {
                InterlockedIncrement64(&g_transport.control->backpressuredFrames);
                InterlockedIncrement64(reinterpret_cast<volatile LONG64*>(
                    &g_transport.control->diagnostics[
                        kD3D12ConsumerReuseDeferralDiagnostic]));
            }
            else
            {
                InterlockedIncrement64(&g_transport.control->droppedFrames);
                InterlockedIncrement64(reinterpret_cast<volatile LONG64*>(
                    &g_transport.control->diagnostics[
                        kD3D12ProducerReuseDeferralDiagnostic]));
            }
            return true;
        }
        if (FAILED(g_d3d12.allocators[slot]->Reset()) ||
            FAILED(g_d3d12.lists[slot]->Reset(g_d3d12.allocators[slot].Get(), nullptr))) return false;

        D3D12_RESOURCE_BARRIER barriers[4]{};
        barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[0].Transition.pResource = backBuffer;
        barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barriers[1] = barriers[0];
        barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        barriers[2] = barriers[0];
        barriers[2].Transition.pResource = g_d3d12.textures[slot].Get();
        barriers[2].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        barriers[2].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        barriers[3] = barriers[2];
        barriers[3].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barriers[3].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        D3D12_RESOURCE_BARRIER begin[] = { barriers[0], barriers[2] };
        g_d3d12.lists[slot]->ResourceBarrier(2, begin);
        g_d3d12.lists[slot]->CopyResource(g_d3d12.textures[slot].Get(), backBuffer);
        D3D12_RESOURCE_BARRIER end[] = { barriers[1], barriers[3] };
        g_d3d12.lists[slot]->ResourceBarrier(2, end);
        if (FAILED(g_d3d12.lists[slot]->Close())) return false;
        ID3D12CommandList* lists[] = { g_d3d12.lists[slot].Get() };
        queue->ExecuteCommandLists(1, lists);
        const HRESULT publishResult = PublishCapturedSequence(
            *g_transport.control,
            sequence,
            sourcePresentQpc,
            sourcePresentOrdinal,
            [&](uint64_t fenceValue)
            {
                return queue->Signal(g_d3d12.fence.Get(), fenceValue);
            });
        if (FAILED(publishResult))
        {
            g_transport.control->diagnostics[12] = 106;
            return false;
        }
        g_d3d12.sequence = sequence;
        g_transport.control->diagnostics[12] = 201;
        SetEvent(g_transport.frameEvent);
        return true;
    }

    void CaptureSwapChain(IDXGISwapChain* swapChain)
    {
        if (g_inCapture || !TryAcquireSRWLockExclusive(&g_captureLock)) return;
        g_inCapture = true;
        LARGE_INTEGER presentQpc{};
        QueryPerformanceCounter(&presentQpc);
        LONG64 sourcePresentOrdinal = 0;

        // Count every Present from the already-validated source swap chain,
        // including frames that cannot enter the bounded effect pipeline. This
        // is one atomic increment rather than a wasteful full-resolution copy.
        const bool knownSource = g_transport.sourceSwapChain == swapChain;
        if (knownSource && g_transport.control && g_transport.control->generation != 0)
        {
            sourcePresentOrdinal =
                AdvanceSourcePresentOrdinal(*g_transport.control);
            const uint64_t latest = static_cast<uint64_t>(InterlockedCompareExchange64(
                &g_transport.control->latestSequence, 0, 0));
            const uint64_t consumed = static_cast<uint64_t>(InterlockedCompareExchange64(
                &g_transport.control->consumedSequence, 0, 0));
            // Keep the full shared ring available. AESS acknowledges only
            // after a GPU fence proves all reads of the acquired slot have
            // completed, so producer overwrite remains impossible.
            if (CaptureRingIsBackpressured(latest, consumed))
            {
                InterlockedIncrement64(&g_transport.control->backpressuredFrames);
                g_inCapture = false;
                ReleaseSRWLockExclusive(&g_captureLock);
                return;
            }
        }
        do
        {
            if (!ValidateAndSelectSwapChain(swapChain, knownSource))
            {
                break;
            }
            if (!knownSource)
            {
                sourcePresentOrdinal =
                    AdvanceSourcePresentOrdinal(*g_transport.control);
                const uint64_t latest = static_cast<uint64_t>(
                    InterlockedCompareExchange64(
                        &g_transport.control->latestSequence, 0, 0));
                const uint64_t consumed = static_cast<uint64_t>(
                    InterlockedCompareExchange64(
                        &g_transport.control->consumedSequence, 0, 0));
                if (CaptureRingIsBackpressured(latest, consumed))
                {
                    InterlockedIncrement64(
                        &g_transport.control->backpressuredFrames);
                    break;
                }
            }
            ID3D11Device* device = g_transport.sourceD3D11Device.Get();
            ComPtr<ID3D11Device> fallbackDevice;
            if (!device && !g_transport.sourceD3D12Device)
            {
                (void)swapChain->GetDevice(IID_PPV_ARGS(&fallbackDevice));
                device = fallbackDevice.Get();
            }
            if (!device)
            {
                g_transport.control->diagnostics[11] = 7;
                if (sourcePresentOrdinal == 0)
                {
                    sourcePresentOrdinal =
                        AdvanceSourcePresentOrdinal(*g_transport.control);
                }
                if (!CaptureD3D12(
                        swapChain, presentQpc.QuadPart, sourcePresentOrdinal) &&
                    g_transport.control)
                    InterlockedIncrement64(&g_transport.control->droppedFrames);
                break;
            }
            UINT backBufferIndex = 0;
            if (g_transport.sourceSwapChain3)
                backBufferIndex = g_transport.sourceSwapChain3->GetCurrentBackBufferIndex();
            ID3D11Texture2D* backBuffer = nullptr;
            const bool hasCachedBackBuffer = (backBufferIndex < g_transport.sourceBackBuffers11.size()) &&
                                             (g_transport.sourceBackBuffers11[backBufferIndex] != nullptr) &&
                                             g_transport.sourceDescValid &&
                                             (g_transport.device && g_transport.device.Get() == device);
            if (hasCachedBackBuffer)
            {
                backBuffer = g_transport.sourceBackBuffers11[backBufferIndex].Get();
            }
            else
            {
                ComPtr<ID3D11Texture2D> newBackBuffer;
                if (FAILED(swapChain->GetBuffer(backBufferIndex, IID_PPV_ARGS(&newBackBuffer))))
                {
                    g_transport.control->diagnostics[11] = 8;
                    break;
                }
                D3D11_TEXTURE2D_DESC desc{};
                newBackBuffer->GetDesc(&desc);
                if (desc.SampleDesc.Count != 1)
                {
                    g_transport.control->diagnostics[11] = 9;
                    break;
                }
                if (!g_transport.device || g_transport.device.Get() != device ||
                    !g_transport.sourceDescValid || !SameSource(g_transport.sourceDesc, desc))
                {
                    if (!CreateTransport(device, desc))
                    {
                        g_transport.control->diagnostics[11] = 10;
                        break;
                    }
                    g_transport.sourceDescValid = true;
                }
                if (backBufferIndex < g_transport.sourceBackBuffers11.size())
                {
                    g_transport.sourceBackBuffers11[backBufferIndex] = newBackBuffer;
                }
                backBuffer = newBackBuffer.Get();
            }

            const uint64_t sequence = g_transport.sequence + 1;
            const uint32_t slot = static_cast<uint32_t>(sequence % kTransportSlotCount);
            if (sourcePresentOrdinal == 0)
            {
                sourcePresentOrdinal =
                    AdvanceSourcePresentOrdinal(*g_transport.control);
            }
            g_transport.context->CopyResource(g_transport.textures[slot].Get(), backBuffer);
            const HRESULT publishResult = PublishCapturedSequence(
                *g_transport.control,
                sequence,
                presentQpc.QuadPart,
                sourcePresentOrdinal,
                [&](uint64_t fenceValue)
                {
                    const HRESULT signalHr = g_transport.context4->Signal(
                        g_transport.fence.Get(),
                        fenceValue);
                    if (SUCCEEDED(signalHr))
                    {
                        g_transport.context->Flush();
                    }
                    return signalHr;
                });
            if (FAILED(publishResult))
            {
                g_transport.control->diagnostics[11] = 12;
                break;
            }
            g_transport.sequence = sequence;
            g_transport.control->diagnostics[11] = 100;
            SetEvent(g_transport.frameEvent);
        } while (false);
        PublishSourceColorSpace();
        g_inCapture = false;
        ReleaseSRWLockExclusive(&g_captureLock);
    }

    HRESULT STDMETHODCALLTYPE HookPresent(IDXGISwapChain* swapChain, UINT syncInterval, UINT flags)
    {
        if (g_transport.control && g_transport.control->generation == 0)
            InterlockedIncrement64(reinterpret_cast<volatile LONG64*>(&g_transport.control->diagnostics[8]));
        CaptureSwapChain(swapChain);
        return g_originalPresent(swapChain, syncInterval, flags);
    }

    HRESULT STDMETHODCALLTYPE HookPresent1(
        IDXGISwapChain1* swapChain, UINT syncInterval, UINT flags,
        const DXGI_PRESENT_PARAMETERS* parameters)
    {
        if (g_transport.control && g_transport.control->generation == 0)
            InterlockedIncrement64(reinterpret_cast<volatile LONG64*>(&g_transport.control->diagnostics[9]));
        CaptureSwapChain(swapChain);
        return g_originalPresent1(swapChain, syncInterval, flags, parameters);
    }

    HRESULT STDMETHODCALLTYPE HookResizeBuffers(
        IDXGISwapChain* swapChain,
        UINT bufferCount,
        UINT width,
        UINT height,
        DXGI_FORMAT newFormat,
        UINT swapChainFlags)
    {
        AcquireSRWLockExclusive(&g_captureLock);
        if (g_transport.sourceSwapChain == swapChain)
        {
            g_transport.InvalidateBackBuffers();
        }
        ReleaseSRWLockExclusive(&g_captureLock);
        return g_originalResizeBuffers(swapChain, bufferCount, width, height, newFormat, swapChainFlags);
    }

    HRESULT STDMETHODCALLTYPE HookResizeBuffers1(
        IDXGISwapChain1* swapChain,
        UINT bufferCount,
        UINT width,
        UINT height,
        DXGI_FORMAT newFormat,
        UINT swapChainFlags,
        const UINT* nodeMasks,
        IUnknown* const* presentQueues)
    {
        AcquireSRWLockExclusive(&g_captureLock);
        if (g_transport.sourceSwapChain == swapChain)
        {
            g_transport.InvalidateBackBuffers();
        }
        ReleaseSRWLockExclusive(&g_captureLock);
        return g_originalResizeBuffers1(swapChain, bufferCount, width, height, newFormat, swapChainFlags, nodeMasks, presentQueues);
    }

    void STDMETHODCALLTYPE HookExecute(
        ID3D12CommandQueue* queue,
        UINT count,
        ID3D12CommandList* const* lists)
    {
        if (!g_inCapture)
        {
            if (queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT)
            {
                g_originalExecute(queue, count, lists);
                return;
            }
            if (g_transport.control) InterlockedIncrement64(
                reinterpret_cast<volatile LONG64*>(&g_transport.control->diagnostics[7]));
            AcquireSRWLockExclusive(&g_queueLock);
            g_observedD3D12Queue = queue;
            ReleaseSRWLockExclusive(&g_queueLock);
        }
        g_originalExecute(queue, count, lists);
    }

    LRESULT CALLBACK DummyWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
    {
        return DefWindowProcW(hwnd, message, wParam, lParam);
    }

    DWORD WINAPI InstallHooks(void*)
    {
        if (!EnsureControl()) return 1;
        g_transport.control->diagnostics[0] = 1;
        WNDCLASSW wc{};
        wc.lpfnWndProc = DummyWndProc;
        wc.hInstance = g_module;
        wc.lpszClassName = L"AESS.CaptureHook.Dummy";
        RegisterClassW(&wc);
        HWND hwnd = CreateWindowW(wc.lpszClassName, L"", WS_OVERLAPPED,
            0, 0, 2, 2, nullptr, nullptr, g_module, nullptr);

        DXGI_SWAP_CHAIN_DESC desc{};
        desc.BufferDesc.Width = 2;
        desc.BufferDesc.Height = 2;
        desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        desc.BufferCount = 2;
        desc.OutputWindow = hwnd;
        desc.Windowed = TRUE;
        desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        ComPtr<IDXGISwapChain> swapChain;
        D3D_FEATURE_LEVEL featureLevel{};
        if (SUCCEEDED(D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
            D3D11_SDK_VERSION, &desc, &swapChain, &device, &featureLevel, &context)))
        {
            g_transport.control->diagnostics[0] = 2;
            void** vtable = *reinterpret_cast<void***>(swapChain.Get());
            const MH_STATUS init = MH_Initialize();
            g_transport.control->diagnostics[1] = static_cast<uint64_t>(init);
            if (init == MH_OK)
            {
                g_transport.control->diagnostics[2] = static_cast<uint64_t>(
                    MH_CreateHook(vtable[8], &HookPresent, reinterpret_cast<void**>(&g_originalPresent)));
                MH_CreateHook(vtable[13], &HookResizeBuffers, reinterpret_cast<void**>(&g_originalResizeBuffers));
                g_transport.control->diagnostics[3] = static_cast<uint64_t>(
                    MH_CreateHook(vtable[22], &HookPresent1, reinterpret_cast<void**>(&g_originalPresent1)));
                ComPtr<IDXGISwapChain3> swapChain3;
                if (SUCCEEDED(swapChain.As(&swapChain3)))
                {
                    void** swapChain3Vtable = *reinterpret_cast<void***>(swapChain3.Get());
                    MH_CreateHook(
                        swapChain3Vtable[38],
                        &HookSetColorSpace,
                        reinterpret_cast<void**>(&g_originalSetColorSpace));
                    MH_CreateHook(
                        swapChain3Vtable[39],
                        &HookResizeBuffers1,
                        reinterpret_cast<void**>(&g_originalResizeBuffers1));
                }
                g_transport.control->diagnostics[4] = static_cast<uint64_t>(MH_EnableHook(MH_ALL_HOOKS));
            }
        }

        ComPtr<ID3D12Device> d3d12Device;
        ComPtr<ID3D12CommandQueue> d3d12Queue;
        if (SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12Device))))
        {
            D3D12_COMMAND_QUEUE_DESC queueDesc{};
            queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            if (SUCCEEDED(d3d12Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&d3d12Queue))))
            {
                void** queueVtable = *reinterpret_cast<void***>(d3d12Queue.Get());
                g_transport.control->diagnostics[5] = static_cast<uint64_t>(
                    MH_CreateHook(queueVtable[10], &HookExecute, reinterpret_cast<void**>(&g_originalExecute)));
                g_transport.control->diagnostics[6] = static_cast<uint64_t>(MH_EnableHook(queueVtable[10]));
            }
        }
        if (hwnd) DestroyWindow(hwnd);
        UnregisterClassW(wc.lpszClassName, g_module);
        return 0;
    }
}

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, void*)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        g_module = module;
        DisableThreadLibraryCalls(module);
        if (HANDLE thread = CreateThread(nullptr, 0, InstallHooks, nullptr, 0, nullptr))
        {
            CloseHandle(thread);
        }
    }
    return TRUE;
}
