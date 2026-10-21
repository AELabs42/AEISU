"""Minimal standard-library UI: source window, start/stop, optional FPS counter."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .session import Session
from .windows import source_windows


class Launcher:
    def __init__(self, root):
        self.root = root
        self.session = Session()
        self.windows = []
        self.closing = False
        self.was_running = False
        root.title("AEISU Demo")
        root.geometry("600x390")
        root.minsize(560, 390)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        panel = ttk.Frame(root, padding=22)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="AEISU", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        ttk.Label(panel, text="SDR only • RTX 50-series • Mixed FP8 / FP16 • 2× upscaling").pack(anchor="w", pady=(0, 18))
        ttk.Label(panel, text="Source window").pack(anchor="w")
        row = ttk.Frame(panel)
        row.pack(fill="x", pady=(5, 12))
        self.selector = ttk.Combobox(row, state="readonly")
        self.selector.pack(side="left", fill="x", expand=True)
        self.refresh_button = ttk.Button(row, text="Refresh", command=self.refresh)
        self.refresh_button.pack(side="left", padx=(8, 0))
        comparison_row = ttk.Frame(panel)
        comparison_row.pack(fill="x", pady=(0, 12))
        ttk.Label(comparison_row, text="Comparison filter").pack(side="left")
        self.comparison_filter = tk.StringVar(value="Nearest")
        self.comparison_box = ttk.Combobox(comparison_row, state="readonly", width=16,
                                          textvariable=self.comparison_filter, values=("Nearest", "Bilinear"))
        self.comparison_box.pack(side="right")
        self.fps = tk.BooleanVar(value=False)
        self.fps_box = ttk.Checkbutton(panel, text="Show FPS counter", variable=self.fps)
        self.fps_box.pack(anchor="w")
        ttk.Label(panel, text="Ctrl+Shift+Tab: AEISU / comparison • Ctrl+Tab: stop • Ctrl+F: FPS",
                  foreground="#666666").pack(anchor="w", pady=(10, 12))
        bottom = ttk.Frame(panel)
        bottom.pack(fill="x")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status).pack(side="left")
        self.button = ttk.Button(bottom, text="Start scaling", command=self.toggle)
        self.button.pack(side="right")
        self.refresh()
        root.after(50, self.poll)

    def refresh(self):
        if self.session.running:
            return
        current = self.selector.current()
        old_handle = self.windows[current].handle if 0 <= current < len(self.windows) else None
        try:
            self.windows = source_windows()
            self.selector["values"] = [window.label for window in self.windows]
            selected = next((index for index, window in enumerate(self.windows) if window.handle == old_handle), 0)
            if self.windows:
                self.selector.current(selected)
            else:
                self.selector.set("")
                self.status.set("Open a game window, then refresh")
        except Exception as error:
            messagebox.showerror("Window selection", str(error), parent=self.root)

    def toggle(self):
        if self.session.running:
            self.session.stop()
            self.status.set("Stopping…")
            self.button.configure(state="disabled")
            return
        index = self.selector.current()
        if not 0 <= index < len(self.windows):
            messagebox.showinfo("Select a source", "Select an open game window first.", parent=self.root)
            return
        try:
            self.session.start(self.windows[index].handle, self.fps.get(), self.comparison_filter.get().lower())
            self.was_running = True
            self.status.set("Scaling • Ctrl+Tab to stop")
            self.button.configure(text="Stop scaling")
            self.selector.configure(state="disabled")
            self.refresh_button.configure(state="disabled")
            self.fps_box.configure(state="disabled")
            self.comparison_box.configure(state="disabled")
            self.root.iconify()
        except Exception as error:
            messagebox.showerror("Cannot start AEISU", str(error), parent=self.root)

    def poll(self):
        if self.was_running and not self.session.running:
            code = self.session.process.returncode
            self.session.close_handles()
            self.was_running = False
            if self.closing:
                self.root.destroy()
                return
            self.root.deiconify()
            self.status.set("Ready" if code == 0 else "Session failed — see outputs/session.log")
            self.button.configure(text="Start scaling", state="normal")
            self.selector.configure(state="readonly")
            self.refresh_button.configure(state="normal")
            self.fps_box.configure(state="normal")
            self.comparison_box.configure(state="readonly")
            if code:
                messagebox.showerror("AEISU session ended", self.session.error_tail(), parent=self.root)
        self.root.after(50, self.poll)

    def close(self):
        if self.session.running:
            self.closing = True
            self.session.stop()
            self.status.set("Stopping…")
        else:
            self.session.close_handles()
            self.root.destroy()


def main():
    root = tk.Tk()
    Launcher(root)
    root.mainloop()
