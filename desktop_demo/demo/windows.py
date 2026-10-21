"""Small, typed Win32 interface for source selection and graceful shutdown."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os

USER32 = ctypes.WinDLL("user32", use_last_error=True)
KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
CALLBACK = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
USER32.EnumWindows.argtypes = [CALLBACK, wintypes.LPARAM]
USER32.EnumWindows.restype = wintypes.BOOL
USER32.IsWindowVisible.argtypes = [wintypes.HWND]
USER32.IsWindowVisible.restype = wintypes.BOOL
USER32.IsWindow.argtypes = [wintypes.HWND]
USER32.IsWindow.restype = wintypes.BOOL
USER32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
USER32.GetWindowTextLengthW.restype = ctypes.c_int
USER32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
USER32.GetWindowTextW.restype = ctypes.c_int
USER32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
USER32.GetWindowThreadProcessId.restype = wintypes.DWORD
USER32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
USER32.GetClientRect.restype = wintypes.BOOL
USER32.SetForegroundWindow.argtypes = [wintypes.HWND]
USER32.SetForegroundWindow.restype = wintypes.BOOL
USER32.AllowSetForegroundWindow.argtypes = [wintypes.DWORD]
USER32.AllowSetForegroundWindow.restype = wintypes.BOOL
KERNEL32.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
KERNEL32.CreateEventW.restype = wintypes.HANDLE
KERNEL32.SetEvent.argtypes = [wintypes.HANDLE]
KERNEL32.SetEvent.restype = wintypes.BOOL
KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
KERNEL32.CloseHandle.restype = wintypes.BOOL


@dataclass(frozen=True)
class SourceWindow:
    handle: int
    title: str
    width: int
    height: int

    @property
    def label(self):
        return f"{self.title}  —  {self.width} × {self.height}"


def source_windows():
    windows = []

    @CALLBACK
    def visit(handle, _):
        if not USER32.IsWindowVisible(handle):
            return True
        process = wintypes.DWORD()
        USER32.GetWindowThreadProcessId(handle, ctypes.byref(process))
        if process.value == os.getpid():
            return True
        length = USER32.GetWindowTextLengthW(handle)
        if length <= 0:
            return True
        title = ctypes.create_unicode_buffer(length + 1)
        USER32.GetWindowTextW(handle, title, len(title))
        rect = wintypes.RECT()
        if USER32.GetClientRect(handle, ctypes.byref(rect)):
            width, height = rect.right - rect.left, rect.bottom - rect.top
            if width >= 64 and height >= 64 and not title.value.startswith("AEISU Demo"):
                windows.append(SourceWindow(int(handle), title.value, width, height))
        return True

    if not USER32.EnumWindows(visit, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return sorted(windows, key=lambda window: window.title.casefold())


class StopEvent:
    def __init__(self, name):
        self.name = name
        self.handle = KERNEL32.CreateEventW(None, True, False, name)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def signal(self):
        if self.handle and not KERNEL32.SetEvent(self.handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            KERNEL32.CloseHandle(self.handle)
            self.handle = None
