from __future__ import annotations

import os
import platform
import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from sdr2hdr.app import (
    CancelToken,
    ConversionCallbacks,
    ConversionRequest,
    PRESETS,
    X265_PROFILE_DEFAULTS,
    build_output_path,
    default_encoder_for_platform,
    run_conversion,
)

X265_MODE_OPTIONS = {
    "preview": "Preview (Fast)",
    "balanced": "Balanced",
    "final": "Final (Best Quality)",
}

def build_encoder_options(system_name: str | None = None) -> dict[str, str]:
    system_name = system_name or platform.system()
    options = {"libx265": "libx265 (Quality)"}
    if system_name == "Darwin":
        options["hevc_videotoolbox"] = "VideoToolbox (Fast on Mac)"
    elif system_name == "Windows":
        options["hevc_nvenc"] = "NVENC (Fast on NVIDIA)"
    return options


def build_backend_options(system_name: str | None = None) -> dict[str, str]:
    system_name = system_name or platform.system()
    options = {"auto": "Auto (Recommended)"}
    if system_name == "Darwin":
        options["mps"] = "MPS (Apple GPU)"
    if system_name == "Windows":
        options["cuda"] = "CUDA (NVIDIA GPU)"
    options["numpy"] = "CPU / NumPy"
    return options


def open_path(path: str) -> None:
    system_name = platform.system()
    if system_name == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    if system_name == "Darwin":
        subprocess.run(["open", path], check=False)
        return
    subprocess.run(["xdg-open", path], check=False)


class AppState:
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SDR2HDRGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("sdr2hdr")
        self.root.geometry("760x580")
        self.state = AppState.IDLE
        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_token: CancelToken | None = None
        self.last_output_path: str | None = None

        self.system_name = platform.system()
        self.encoder_options = build_encoder_options(self.system_name)
        self.backend_options = build_backend_options(self.system_name)
        default_encoder = default_encoder_for_platform(self.system_name)
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.preset_var = tk.StringVar(value="portrait")
        self.encoder_var = tk.StringVar(value=self.encoder_options[default_encoder])
        self.x265_mode_var = tk.StringVar(value=X265_MODE_OPTIONS["balanced"])
        self.backend_var = tk.StringVar(value=self.backend_options["auto"])
        self.status_var = tk.StringVar(value="Idle")
        self.progress_var = tk.StringVar(value="0 frames")
        self.mode_hint_var = tk.StringVar(value="")

        self._build()
        self._set_state(AppState.IDLE)
        self.root.after(100, self._drain_events)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(outer, text="SDR to HDR10 Converter", font=("Helvetica", 18, "bold"))
        title.pack(anchor="w")
        subtitle = ttk.Label(
            outer,
            text="Single-job desktop UI for real footage conversion.",
            font=("Helvetica", 11),
        )
        subtitle.pack(anchor="w", pady=(4, 16))

        form = ttk.Frame(outer)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        self.input_entry = self._add_path_row(form, 0, "Input", self.input_var, self._browse_input)
        self.output_entry = self._add_path_row(form, 1, "Output", self.output_var, self._browse_output)

        self.preset_combo = self._add_combo_row(form, 2, "Preset", self.preset_var, list(PRESETS))
        self.encoder_combo = self._add_combo_row(form, 3, "Encoder", self.encoder_var, list(self.encoder_options.values()))
        self.x265_combo = self._add_combo_row(form, 4, "Speed/Quality", self.x265_mode_var, list(X265_MODE_OPTIONS.values()))
        self.backend_combo = self._add_combo_row(form, 5, "Backend", self.backend_var, list(self.backend_options.values()))
        ttk.Label(form, textvariable=self.mode_hint_var).grid(row=6, column=1, sticky="w", pady=(4, 0))

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(16, 12))
        self.start_button = ttk.Button(controls, text="Start", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop)
        self.stop_button.pack(side="left", padx=(8, 0))
        self.open_output_button = ttk.Button(controls, text="Open Output", command=self._open_output)
        self.open_output_button.pack(side="left", padx=(8, 0))
        self.open_folder_button = ttk.Button(controls, text="Open Folder", command=self._open_folder)
        self.open_folder_button.pack(side="left", padx=(8, 0))

        status_frame = ttk.Frame(outer)
        status_frame.pack(fill="x")
        ttk.Label(status_frame, textvariable=self.status_var, font=("Helvetica", 11, "bold")).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.progress_var).pack(anchor="w", pady=(4, 8))
        self.progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x")

        ttk.Label(outer, text="Log").pack(anchor="w", pady=(16, 6))
        self.log = tk.Text(outer, height=16, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

        self.input_var.trace_add("write", self._sync_output_path)
        self.encoder_var.trace_add("write", self._sync_encoder_ui)
        self.x265_mode_var.trace_add("write", self._sync_mode_hint)
        self.backend_var.trace_add("write", self._sync_mode_hint)
        self._sync_encoder_ui()
        self._sync_mode_hint()

    def _add_path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        browse_command: object,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6, padx=(0, 12))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        ttk.Button(parent, text="Browse", command=browse_command).grid(row=row, column=2, padx=(8, 0))
        return entry

    def _add_combo_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        values: list[str],
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6, padx=(0, 12))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(row=row, column=1, sticky="w", pady=6)
        return combo

    def _sync_output_path(self, *_: object) -> None:
        if not self.input_var.get():
            return
        current = self.output_var.get().strip()
        if not current or current == self.last_output_path:
            self.last_output_path = build_output_path(self.input_var.get())
            self.output_var.set(self.last_output_path)

    def _sync_encoder_ui(self, *_: object) -> None:
        if self._selected_encoder() == "libx265":
            self.x265_combo.configure(state="readonly")
        else:
            self.x265_combo.configure(state="disabled")
        self._sync_mode_hint()

    def _sync_mode_hint(self, *_: object) -> None:
        encoder = self._selected_encoder()
        mode = self._selected_x265_mode()
        backend = self._selected_backend()
        if encoder == "hevc_videotoolbox":
            self.mode_hint_var.set("Fastest on supported Macs. Falls back to libx265 if VideoToolbox fails.")
        elif encoder == "hevc_nvenc":
            self.mode_hint_var.set("Fastest on supported NVIDIA GPUs. Falls back to libx265 if NVENC fails.")
        else:
            if mode == "preview":
                speed = "Fastest x265 mode, lower compression efficiency."
            elif mode == "final":
                speed = "Slowest x265 mode, best compression quality."
            else:
                speed = "Balanced x265 mode for daily use."
            if backend == "mps":
                backend_hint = " Uses Apple GPU."
            elif backend == "cuda":
                backend_hint = " Uses NVIDIA GPU for processing."
            else:
                backend_hint = ""
            self.mode_hint_var.set(speed + backend_hint)

    def _selected_encoder(self) -> str:
        for key, label in self.encoder_options.items():
            if self.encoder_var.get() == label:
                return key
        return "libx265"

    def _selected_x265_mode(self) -> str:
        for key, label in X265_MODE_OPTIONS.items():
            if self.x265_mode_var.get() == label:
                return key
        return "balanced"

    def _selected_backend(self) -> str:
        for key, label in self.backend_options.items():
            if self.backend_var.get() == label:
                return key
        return "auto"

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(title="Select input video")
        if path:
            self.input_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Select output video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
        )
        if path:
            self.output_var.set(path)
            self.last_output_path = path

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"[{timestamp}] {message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _build_request(self) -> ConversionRequest:
        return ConversionRequest(
            input_path=self.input_var.get().strip(),
            output_path=self.output_var.get().strip(),
            preset=self.preset_var.get(),
            encoder=self._selected_encoder(),
            x265_mode=self._selected_x265_mode(),
            backend=self._selected_backend(),
            fallback_to_x265_on_hardware_error=True,
            keep_partial_output_on_cancel=True,
        )

    def _start(self) -> None:
        try:
            request = self._build_request()
            if not request.input_path:
                raise ValueError("Input path is required.")
            if not request.output_path:
                raise ValueError("Output path is required.")
        except ValueError as exc:
            messagebox.showerror("Cannot start", str(exc))
            return

        self.cancel_token = CancelToken()
        self.last_output_path = request.output_path
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(10)
        self._log(f"Starting conversion: {Path(request.input_path).name}")
        self._set_state(AppState.RUNNING)

        callbacks = ConversionCallbacks(
            on_status=lambda message: self.event_queue.put(("status", message)),
            on_progress=lambda processed, total, fps: self.event_queue.put(("progress", (processed, total, fps))),
            on_complete=lambda result: self.event_queue.put(("complete", result)),
            on_error=lambda message: self.event_queue.put(("error", message)),
        )

        def worker() -> None:
            try:
                run_conversion(request, callbacks=callbacks, cancel_token=self.cancel_token)
            except Exception as exc:
                self.event_queue.put(("failed", str(exc)))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        if self.cancel_token is not None:
            self.cancel_token.cancel()
            self._log("Stop requested")

    def _open_output(self) -> None:
        if not self.last_output_path:
            return
        open_path(self.last_output_path)

    def _open_folder(self) -> None:
        if not self.last_output_path:
            return
        open_path(str(Path(self.last_output_path).parent))

    def _set_state(self, state: str) -> None:
        self.state = state
        running = state == AppState.RUNNING
        idle_like = state in {AppState.IDLE, AppState.COMPLETED, AppState.FAILED, AppState.CANCELLED}
        field_state = "disabled" if running else "normal"
        combo_state = "disabled" if running else "readonly"
        self.input_entry.configure(state=field_state)
        self.output_entry.configure(state=field_state)
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.open_output_button.configure(state="normal" if idle_like and self.last_output_path else "disabled")
        self.open_folder_button.configure(state="normal" if idle_like and self.last_output_path else "disabled")
        self.preset_combo.configure(state=combo_state)
        self.encoder_combo.configure(state=combo_state)
        self.backend_combo.configure(state=combo_state)
        self._sync_encoder_ui()

    def _drain_events(self) -> None:
        while True:
            try:
                kind, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status_var.set(str(payload))
                self._log(str(payload))
            elif kind == "progress":
                processed, total, fps = payload
                if total:
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=100, value=(processed / total) * 100)
                fps_text = f"{fps:.1f} fps" if fps else "n/a"
                self.progress_var.set(f"{processed}/{total or '?'} frames, {fps_text}")
            elif kind == "complete":
                result = payload
                self.progress.stop()
                self.progress.configure(value=100 if not result.cancelled else 0)
                if result.cancelled:
                    self.status_var.set("Cancelled")
                    self.progress_var.set(f"{result.processed_frames} frames processed; partial output saved")
                    self._log("Conversion cancelled; partial output saved")
                    self._set_state(AppState.CANCELLED)
                else:
                    self.status_var.set("Completed")
                    self.progress_var.set(f"{result.processed_frames} frames written")
                    self._log(f"Completed: {Path(result.output_path).name}")
                    self._set_state(AppState.COMPLETED)
            elif kind == "error":
                self._log(str(payload))
            elif kind == "failed":
                self.progress.stop()
                self.progress.configure(value=0)
                self.status_var.set("Failed")
                self.progress_var.set("Conversion failed")
                self._log(str(payload))
                messagebox.showerror("Conversion failed", str(payload))
                self._set_state(AppState.FAILED)
        self.root.after(100, self._drain_events)


def main() -> int:
    root = tk.Tk()
    app = SDR2HDRGUI(root)
    app._log("Ready")
    root.mainloop()
    return 0
