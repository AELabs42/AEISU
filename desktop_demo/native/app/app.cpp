#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dcomp.h>
#include <dwmapi.h>
#include <dxgi1_6.h>
#include <commctrl.h>
#include <mmsystem.h>
#include <shellapi.h>
#include <shcore.h>
#include <wrl/client.h>

#include <imgui.h>
#include <imgui_impl_dx11.h>
#include <imgui_impl_win32.h>

#include <winrt/base.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include "../src/neural/NeuralCudaRuntime.h"
#include "../src/neural/NeuralInterop.h"
#include "../src/capture/AESSCaptureTransport.h"
#include "../src/capture/CaptureRetirementQueue.h"
#include "../src/presentation/FramePacer.h"
#include "../src/presentation/FrameTelemetry.h"
#include "../src/presentation/GsyncForegroundSpoof.h"
#include "../src/render/D3D11ResourceCache.h"
#include "AESSProductionShaders.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using Microsoft::WRL::ComPtr;

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

// ============================================================
// Self-test mode (behavior-preserving diagnostic path)
// ============================================================

namespace wg = winrt::Windows::Graphics;
namespace wgc = winrt::Windows::Graphics::Capture;
namespace wgd = winrt::Windows::Graphics::DirectX;
namespace wgd11 = winrt::Windows::Graphics::DirectX::Direct3D11;

namespace
{
#include "common.inl"
#include "device.inl"
#include "capture.inl"
#include "render_support.inl"
#include "presenter.inl"
#include "session.inl"
}
#include "entry.inl"
