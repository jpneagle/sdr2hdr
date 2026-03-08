from __future__ import annotations

import os
import platform
import queue
import subprocess
import threading
import time
import tkinter as tk
from dataclasses import dataclass
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
    validate_request,
)

X265_MODE_OPTIONS = {
    "preview": "Preview (Fast)",
    "balanced": "Balanced",
    "final": "Final (Best Quality)",
}


def describe_mode_hint(encoder: str, mode: str, backend: str, preset: str, model_path: str) -> str:
    if encoder == "hevc_videotoolbox":
        return "Fastest on supported Macs. Falls back to libx265 if VideoToolbox fails."
    if encoder == "hevc_nvenc":
        return "Fastest on supported NVIDIA GPUs. Falls back to libx265 if NVENC fails."
    if mode == "preview":
        speed = "Fastest x265 mode, lower compression efficiency."
    elif mode == "final":
        speed = "Slowest x265 mode, best compression quality."
    else:
        speed = "Balanced x265 mode for daily use."
    if model_path.strip():
        if backend == "cuda":
            return speed + " Learned-map mode enabled on NVIDIA GPU."
        if backend == "mps":
            return speed + " Learned-map mode enabled on Apple GPU."
        return speed + " Learned-map mode enabled."
    if backend == "mps":
        return speed + " Uses Apple GPU."
    if backend == "cuda":
        return speed + " Uses NVIDIA GPU for processing."
    return speed


def format_ai_strength(value: float) -> str:
    return f"{value:.2f}"

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


@dataclass
class QueueJob:
    request: ConversionRequest
    status: str = "pending"


STATUS_LABELS = {
    "pending": "",
    "running": "RUNNING",
    "completed": "OK",
    "failed": "FAILED",
    "cancelled": "CANCELLED",
}


class SDR2HDRGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("sdr2hdr")
        self.root.geometry("1120x720")
        self.state = AppState.IDLE
        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_token: CancelToken | None = None
        self.last_output_path: str | None = None
        self.queue_jobs: list[QueueJob] = []
        self.current_job_index: int | None = None

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
        self.model_path_var = tk.StringVar()
        self.ai_strength_var = tk.DoubleVar(value=0.2)
        self.ai_strength_label_var = tk.StringVar(value=format_ai_strength(0.2))
        self.status_var = tk.StringVar(value="Idle")
        self.progress_var = tk.StringVar(value="0 frames")
        self.mode_hint_var = tk.StringVar(value="")

        self._build()
        self._set_state(AppState.IDLE)
        self.root.after(100, self._drain_events)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(2, weight=1)

        title = ttk.Label(outer, text="SDR to HDR10 Converter", font=("Helvetica", 18, "bold"))
        title.grid(row=0, column=0, columnspan=2, sticky="w")
        subtitle = ttk.Label(
            outer,
            text="Queued desktop UI for real footage conversion.",
            font=("Helvetica", 11),
        )
        subtitle.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 16))

        left = ttk.Frame(outer)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(4, weight=1)

        right = ttk.Frame(outer)
        right.grid(row=2, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        form = ttk.Frame(left)
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)

        self.input_entry = self._add_path_row(form, 0, "Input", self.input_var, self._browse_input)
        self.output_entry = self._add_path_row(form, 1, "Output", self.output_var, self._browse_output)

        self.preset_combo = self._add_combo_row(form, 2, "Preset", self.preset_var, list(PRESETS))
        self.encoder_combo = self._add_combo_row(form, 3, "Encoder", self.encoder_var, list(self.encoder_options.values()))
        self.x265_combo = self._add_combo_row(form, 4, "Speed/Quality", self.x265_mode_var, list(X265_MODE_OPTIONS.values()))
        self.backend_combo = self._add_combo_row(form, 5, "Backend", self.backend_var, list(self.backend_options.values()))
        self.model_entry = self._add_path_row(form, 6, "Model Path", self.model_path_var, self._browse_model)
        ttk.Label(form, text="AI Strength").grid(row=7, column=0, sticky="w", pady=6, padx=(0, 12))
        slider_row = ttk.Frame(form)
        slider_row.grid(row=7, column=1, sticky="ew", pady=6)
        slider_row.columnconfigure(0, weight=1)
        self.ai_strength_scale = ttk.Scale(
            slider_row,
            from_=0.0,
            to=0.8,
            orient="horizontal",
            variable=self.ai_strength_var,
            command=self._sync_ai_strength_label,
        )
        self.ai_strength_scale.grid(row=0, column=0, sticky="ew")
        ttk.Label(slider_row, textvariable=self.ai_strength_label_var, width=5).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(form, textvariable=self.mode_hint_var).grid(row=8, column=1, sticky="w", pady=(4, 0))

        controls = ttk.Frame(left)
        controls.grid(row=1, column=0, sticky="ew", pady=(16, 12))
        self.start_button = ttk.Button(controls, text="Start Queue", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Stop Current", command=self._stop)
        self.stop_button.pack(side="left", padx=(8, 0))
        self.open_output_button = ttk.Button(controls, text="Open Output", command=self._open_output)
        self.open_output_button.pack(side="left", padx=(8, 0))
        self.open_folder_button = ttk.Button(controls, text="Open Folder", command=self._open_folder)
        self.open_folder_button.pack(side="left", padx=(8, 0))

        status_frame = ttk.Frame(left)
        status_frame.grid(row=2, column=0, sticky="ew")
        ttk.Label(status_frame, textvariable=self.status_var, font=("Helvetica", 11, "bold")).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.progress_var).pack(anchor="w", pady=(4, 8))
        self.progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x")

        ttk.Label(left, text="Log").grid(row=3, column=0, sticky="w", pady=(16, 6))
        self.log = tk.Text(left, height=16, wrap="word", state="disabled")
        self.log.grid(row=4, column=0, sticky="nsew")

        queue_controls = ttk.Frame(right)
        queue_controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.add_queue_button = ttk.Button(queue_controls, text="Add To Queue", command=self._enqueue_current)
        self.add_queue_button.pack(side="left")
        self.add_files_button = ttk.Button(queue_controls, text="Add Files", command=self._enqueue_files)
        self.add_files_button.pack(side="left", padx=(8, 0))
        self.remove_queue_button = ttk.Button(queue_controls, text="Remove Selected", command=self._remove_selected_job)
        self.remove_queue_button.pack(side="left", padx=(8, 0))
        self.clear_queue_button = ttk.Button(queue_controls, text="Clear Queue", command=self._clear_queue)
        self.clear_queue_button.pack(side="left", padx=(8, 0))

        ttk.Label(right, text="Queue").grid(row=1, column=0, sticky="w", pady=(8, 6))
        self.queue_view = ttk.Treeview(right, columns=("status", "input", "output"), show="headings", height=14)
        self.queue_view.heading("status", text="Status")
        self.queue_view.heading("input", text="Input")
        self.queue_view.heading("output", text="Output")
        self.queue_view.column("status", width=90, anchor="w")
        self.queue_view.column("input", width=170, anchor="w")
        self.queue_view.column("output", width=220, anchor="w")
        self.queue_view.grid(row=2, column=0, sticky="nsew")

        self.input_var.trace_add("write", self._sync_output_path)
        self.encoder_var.trace_add("write", self._sync_encoder_ui)
        self.x265_mode_var.trace_add("write", self._sync_mode_hint)
        self.backend_var.trace_add("write", self._sync_mode_hint)
        self.preset_var.trace_add("write", self._sync_mode_hint)
        self.model_path_var.trace_add("write", self._sync_model_controls)
        self._sync_encoder_ui()
        self._sync_ai_strength_label()
        self._sync_model_controls()
        self._sync_mode_hint()
        self._refresh_job_list()

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
        self.mode_hint_var.set(
            describe_mode_hint(
                self._selected_encoder(),
                self._selected_x265_mode(),
                self._selected_backend(),
                self.preset_var.get(),
                self.model_path_var.get(),
            )
        )

    def _sync_ai_strength_label(self, *_: object) -> None:
        self.ai_strength_label_var.set(format_ai_strength(self.ai_strength_var.get()))

    def _sync_model_controls(self, *_: object) -> None:
        has_model = bool(self.model_path_var.get().strip())
        scale_state = "normal" if has_model and self.state != AppState.RUNNING else "disabled"
        self.ai_strength_scale.configure(state=scale_state)
        self._sync_ai_strength_label()
        self._sync_mode_hint()

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

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select learned model",
            filetypes=[("TorchScript model", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_path_var.set(path)

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
            model_path=self.model_path_var.get().strip() or None,
            ai_strength=self.ai_strength_var.get() if self.model_path_var.get().strip() else None,
            device="auto",
            fallback_to_x265_on_hardware_error=True,
            keep_partial_output_on_cancel=True,
        )

    def _validate_request(self, request: ConversionRequest) -> None:
        if not request.input_path:
            raise ValueError("Input path is required.")
        if not request.output_path:
            raise ValueError("Output path is required.")
        validate_request(request)

    def _make_job_label(self, request: ConversionRequest) -> tuple[str, str, str]:
        return ("pending", Path(request.input_path).name, Path(request.output_path).name)

    def _refresh_job_list(self) -> None:
        self.queue_view.delete(*self.queue_view.get_children())
        for index, job in enumerate(self.queue_jobs):
            status = job.status
            if self.current_job_index == index and self.state == AppState.RUNNING:
                status = "running"
            self.queue_view.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    STATUS_LABELS.get(status, status.upper()),
                    Path(job.request.input_path).name,
                    Path(job.request.output_path).name,
                ),
            )

    def _enqueue_request(self, request: ConversionRequest) -> None:
        self._validate_request(request)
        self.queue_jobs.append(QueueJob(request=request))
        self.last_output_path = request.output_path
        self._refresh_job_list()
        self._log(f"Queued: {Path(request.input_path).name}")

    def _enqueue_current(self) -> None:
        try:
            self._enqueue_request(self._build_request())
        except ValueError as exc:
            messagebox.showerror("Cannot queue", str(exc))

    def _enqueue_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Select input videos")
        if not paths:
            return
        for raw_path in paths:
            input_path = str(raw_path)
            output_path = build_output_path(input_path)
            request = ConversionRequest(
                input_path=input_path,
                output_path=output_path,
                preset=self.preset_var.get(),
                encoder=self._selected_encoder(),
                x265_mode=self._selected_x265_mode(),
                backend=self._selected_backend(),
                model_path=self.model_path_var.get().strip() or None,
                ai_strength=self.ai_strength_var.get() if self.model_path_var.get().strip() else None,
                device="auto",
                fallback_to_x265_on_hardware_error=True,
                keep_partial_output_on_cancel=True,
            )
            self._enqueue_request(request)

    def _selected_job_indices(self) -> list[int]:
        return sorted((int(item_id) for item_id in self.queue_view.selection()), reverse=True)

    def _remove_selected_job(self) -> None:
        if self.state == AppState.RUNNING:
            return
        removed = False
        for index in self._selected_job_indices():
            if 0 <= index < len(self.queue_jobs):
                del self.queue_jobs[index]
                removed = True
        if removed:
            self._refresh_job_list()
            self._log("Removed selected queue items")

    def _clear_queue(self) -> None:
        if self.state == AppState.RUNNING:
            return
        self.queue_jobs.clear()
        self.current_job_index = None
        self._refresh_job_list()
        self._log("Cleared queue")

    def _next_pending_job_index(self) -> int | None:
        for index, job in enumerate(self.queue_jobs):
            if job.status == "pending":
                return index
        return None

    def _start_job(self, index: int) -> None:
        request = self.queue_jobs[index].request
        self.current_job_index = index
        self.queue_jobs[index].status = "running"
        self.cancel_token = CancelToken()
        self.last_output_path = request.output_path
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(10)
        self._refresh_job_list()
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

    def _start(self) -> None:
        if self.state == AppState.RUNNING:
            return
        if not self.queue_jobs:
            try:
                self._enqueue_request(self._build_request())
            except ValueError as exc:
                messagebox.showerror("Cannot start", str(exc))
                return
        next_index = self._next_pending_job_index()
        if next_index is None:
            return
        self._start_job(next_index)

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
        self.model_entry.configure(state=field_state)
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.add_queue_button.configure(state="disabled" if running else "normal")
        self.add_files_button.configure(state="disabled" if running else "normal")
        self.remove_queue_button.configure(state="disabled" if running else "normal")
        self.clear_queue_button.configure(state="disabled" if running else "normal")
        self.open_output_button.configure(state="normal" if idle_like and self.last_output_path else "disabled")
        self.open_folder_button.configure(state="normal" if idle_like and self.last_output_path else "disabled")
        self.preset_combo.configure(state=combo_state)
        self.encoder_combo.configure(state=combo_state)
        self.backend_combo.configure(state=combo_state)
        self._sync_encoder_ui()
        self._sync_model_controls()

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
                if self.current_job_index is not None:
                    self.queue_jobs[self.current_job_index].status = "cancelled" if result.cancelled else "completed"
                    self.last_output_path = self.queue_jobs[self.current_job_index].request.output_path
                self._refresh_job_list()
                if result.cancelled:
                    self.status_var.set("Cancelled")
                    self.progress_var.set(f"{result.processed_frames} frames processed; partial output saved")
                    self._log("Conversion cancelled; partial output saved")
                    self._set_state(AppState.CANCELLED)
                else:
                    self.status_var.set("Completed")
                    self.progress_var.set(f"{result.processed_frames} frames written")
                    self._log(f"Completed: {Path(result.output_path).name}")
                    next_index = self._next_pending_job_index()
                    if next_index is not None:
                        self._set_state(AppState.IDLE)
                        self._start_job(next_index)
                    else:
                        self._set_state(AppState.COMPLETED)
            elif kind == "error":
                self._log(str(payload))
            elif kind == "failed":
                self.progress.stop()
                self.progress.configure(value=0)
                if self.current_job_index is not None:
                    self.queue_jobs[self.current_job_index].status = "failed"
                    self._refresh_job_list()
                self.status_var.set("Failed")
                self.progress_var.set("Conversion failed")
                self._log(str(payload))
                messagebox.showerror("Conversion failed", str(payload))
                next_index = self._next_pending_job_index()
                if next_index is not None:
                    self._set_state(AppState.IDLE)
                    self._start_job(next_index)
                else:
                    self._set_state(AppState.FAILED)
        self.root.after(100, self._drain_events)


def main() -> int:
    root = tk.Tk()
    app = SDR2HDRGUI(root)
    app._log("Ready")
    root.mainloop()
    return 0
