"""Open the launcher briefly for visual inspection; never starts inference."""
import tkinter as tk
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import tempfile

from demo.ui import Launcher

root = tk.Tk()
launcher = Launcher(root)
launcher.selector.configure(values=["Example game  —  1920 × 1080"])
launcher.selector.current(0)
root.geometry("600x390+120+120")


def inspect():
    # The normal screen-copy helper returned the occluding desktop rather than
    # the launcher. Render this test-created window itself with PrintWindow.
    from PIL import Image
    user = ctypes.WinDLL("user32", use_last_error=True)
    gdi = ctypes.WinDLL("gdi32", use_last_error=True)
    user.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user.GetAncestor.restype = wintypes.HWND
    user.GetWindowDC.argtypes = [wintypes.HWND]
    user.GetWindowDC.restype = wintypes.HDC
    user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    user.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    gdi.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi.CreateCompatibleDC.restype = wintypes.HDC
    gdi.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi.SelectObject.restype = wintypes.HGDIOBJ
    gdi.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, wintypes.LPVOID, wintypes.LPVOID, wintypes.UINT]
    gdi.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi.DeleteDC.argtypes = [wintypes.HDC]
    handle = user.GetAncestor(root.winfo_id(), 2)
    rect = wintypes.RECT()
    user.GetWindowRect(handle, ctypes.byref(rect))
    width, height = rect.right - rect.left, rect.bottom - rect.top
    device = user.GetWindowDC(handle)
    memory = gdi.CreateCompatibleDC(device)
    bitmap = gdi.CreateCompatibleBitmap(device, width, height)
    previous = gdi.SelectObject(memory, bitmap)
    if not user.PrintWindow(handle, memory, 2):
        raise ctypes.WinError(ctypes.get_last_error())
    gdi.SelectObject(memory, previous)
    header = (ctypes.c_ubyte * 40)()
    import struct
    header[:] = struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 32, 0, width * height * 4, 0, 0, 0, 0)
    pixels = (ctypes.c_ubyte * (width * height * 4))()
    if not gdi.GetDIBits(memory, bitmap, 0, height, pixels, header, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    path = Path(tempfile.gettempdir()) / "aeisu-demo-ui.png"
    Image.frombytes("RGB", (width, height), bytes(pixels), "raw", "BGRX").save(path)
    gdi.DeleteObject(bitmap)
    gdi.DeleteDC(memory)
    user.ReleaseDC(handle, device)
    print(f"UI_SCREENSHOT={path}", flush=True)
    # Coordinate checks complement visual inspection.
    bottom = launcher.button.winfo_rooty() + launcher.button.winfo_height()
    assert bottom <= root.winfo_rooty() + root.winfo_height(), "Start button clips outside client area"
    root.after(1000, root.destroy)


root.after(800, inspect)
root.mainloop()
