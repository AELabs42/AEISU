#pragma once

#include <d3d11_1.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

namespace aess::neural
{
    // Keep the resource's real allocation description for CUDA import. Legacy
    // KMT textures do not expose their allocation footprint/tiling contract.
    inline constexpr GUID kCudaSharedResource =
        {0x6ad2890a,0x5ff2,0x4b12,{0x8c,0x46,0x26,0x9f,0xd4,0x66,0xf8,0x32}};
    inline constexpr GUID kCudaSharedDevice =
        {0x4647c4d6,0xadce,0x4904,{0xa2,0xe8,0x82,0x2e,0xb8,0xc2,0x2a,0x8c}};

    // D3D12 allocates once; D3D11 and CUDA use the same bytes. No D3D12 command
    // queue, cross-device frame copy, keyed mutex, or per-frame map/unmap.
    inline HRESULT CreateInteropTexture(ID3D11Device* device,
        const D3D11_TEXTURE2D_DESC& description, ID3D11Texture2D** result)
    {
        using Microsoft::WRL::ComPtr;
        if (!device || !result || description.Format!=DXGI_FORMAT_R16G16B16A16_FLOAT ||
            description.MipLevels!=1 || description.ArraySize!=1 || description.SampleDesc.Count!=1)
            return E_INVALIDARG;
        *result=nullptr;
        ComPtr<ID3D12Device> sharedDevice;
        UINT size=sizeof(ID3D12Device*);
        HRESULT hr=device->GetPrivateData(kCudaSharedDevice,&size,sharedDevice.GetAddressOf());
        if (FAILED(hr))
        {
            ComPtr<IDXGIDevice> dxgi;
            ComPtr<IDXGIAdapter> adapter;
            hr=device->QueryInterface(IID_PPV_ARGS(&dxgi));
            if (FAILED(hr)) return hr;
            hr=dxgi->GetAdapter(&adapter);
            if (FAILED(hr)) return hr;
            hr=D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&sharedDevice));
            if (FAILED(hr)) return hr;
            hr=device->SetPrivateDataInterface(kCudaSharedDevice,sharedDevice.Get());
            if (FAILED(hr)) return hr;
        }
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        heap.CreationNodeMask=1;
        heap.VisibleNodeMask=1;
        D3D12_RESOURCE_DESC resource{};
        resource.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        resource.Width=description.Width;
        resource.Height=description.Height;
        resource.DepthOrArraySize=1;
        resource.MipLevels=1;
        resource.Format=description.Format;
        resource.SampleDesc.Count=1;
        resource.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
        if ((description.BindFlags&D3D11_BIND_UNORDERED_ACCESS)!=0)
            resource.Flags|=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        ComPtr<ID3D12Resource> allocation;
        hr=sharedDevice->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_SHARED,&resource,
            D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&allocation));
        if (FAILED(hr)) return hr;
        HANDLE handle=nullptr;
        hr=sharedDevice->CreateSharedHandle(allocation.Get(),nullptr,GENERIC_ALL,nullptr,&handle);
        if (FAILED(hr)) return hr;
        ComPtr<ID3D11Device1> device1;
        hr=device->QueryInterface(IID_PPV_ARGS(&device1));
        ComPtr<ID3D11Texture2D> texture;
        if (SUCCEEDED(hr)) hr=device1->OpenSharedResource1(handle,IID_PPV_ARGS(&texture));
        CloseHandle(handle);
        if (FAILED(hr)) return hr;
        hr=texture->SetPrivateDataInterface(kCudaSharedResource,allocation.Get());
        if (FAILED(hr)) return hr;
        *result=texture.Detach();
        return S_OK;
    }
}
