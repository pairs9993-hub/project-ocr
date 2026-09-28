from __future__ import annotations

import hashlib
import json
import sys
import threading
from time import perf_counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import mss
from PIL import Image, ImageTk

from .capture import grab_screen_rect_with, timed_capture
from .compare import (
    compare_text,
    has_ui_image_token,
    normalize_ocr_ui_text,
    normalize_ui_text,
)
from .ocr_engine import OCREngine
from .preset_workflow import ROIPresetWorkflow
from .automation_workflow import AutomationWorkflow
from .roi_preprocess import RoiPreprocessConfig, crop_roi, pad_long_roi
from .scroll_merge import AdaptiveFrameSampler, ScrollTextAccumulator, VerticalListAccumulator


Rect = tuple[int, int, int, int]


def _pad_long_roi(image: Image.Image) -> Image.Image:
    return pad_long_roi(image)


def _absolute_selection_rect(
    monitor: dict,
    start: tuple[int, int],
    end: tuple[int, int],
) -> Rect | None:
    x1, y1 = min(start[0], end[0]), min(start[1], end[1])
    x2, y2 = max(start[0], end[0]), max(start[1], end[1])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (
        monitor["left"] + x1,
        monitor["top"] + y1,
        monitor["left"] + x2,
        monitor["top"] + y2,
    )


@dataclass
class ROIItem:
    roi_id: int
    rect: Rect
    expected: str = ""
    actual: str = ""
    passed: bool = False


# Fidelity of the image stored as roi_ocr_input.png.
FIDELITY_EXACT = "exact_recorded_ocr_input"
FIDELITY_RECONSTRUCTED = "representative_or_reconstructed_not_exact"


@dataclass
class OCRInputRecord:
    """The pixels actually handed to ``OCREngine.run()``, captured at call time.

    Recomputing the crop after the fact can diverge from what the recognizer
    saw -- a control may have changed, or the result may have come from a
    context crop or several accumulated frames. Recording the image at the call
    site is the only way to guarantee the saved PNG is the real input.
    """

    image: Image.Image
    path_kind: str          # "direct" | "context" | "context_fallback"
    language: str
    exact: bool
    raw_ocr_text: str = ""
    spacing_evidence: dict | None = None


def _save_failed_roi_diagnostic(
    source_image: Image.Image,
    roi: ROIItem,
    output_root: Path = Path("captures") / "failures",
    preprocess: RoiPreprocessConfig | None = None,
    model_info: dict | None = None,
    ocr_input: OCRInputRecord | None = None,
) -> Path:
    """Save a failed ROI for later analysis.

    ``roi.png`` is the ROI rect alone, as it has always been.

    ``roi_ocr_input.png`` is written when the recognizer input is available.
    When ``ocr_input`` carries an image recorded at the OCR call site it is
    saved verbatim and marked ``exact_recorded_ocr_input``. Otherwise, if
    ``preprocess`` is given, the crop is recomputed from ``source_image`` and
    marked ``representative_or_reconstructed_not_exact`` -- a reconstruction is
    never presented as the real input.
    """
    run_dir = output_root / f"failure_{datetime.now():%Y%m%d_%H%M%S_%f}_roi{roi.roi_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_crop = source_image.crop(roi.rect)
    raw_crop.save(run_dir / "roi.png")

    metadata = {
        "roi_id": roi.roi_id,
        "rect": list(roi.rect),
        "expected": roi.expected,
        "actual": roi.actual,
        "roi_raw_size": list(raw_crop.size),
    }

    saved_input = None
    if ocr_input is not None:
        saved_input = ocr_input.image
        metadata["ocr_input_fidelity"] = (
            FIDELITY_EXACT if ocr_input.exact else FIDELITY_RECONSTRUCTED
        )
        metadata["language"] = ocr_input.language
        metadata["ocr_path"] = ocr_input.path_kind
        metadata["ocr_raw_output"] = ocr_input.raw_ocr_text
        if ocr_input.spacing_evidence:
            metadata["image_spacing"] = ocr_input.spacing_evidence
    elif preprocess is not None:
        saved_input = crop_roi(source_image, roi.rect, preprocess)
        metadata["ocr_input_fidelity"] = FIDELITY_RECONSTRUCTED

    if saved_input is not None:
        saved_input.save(run_dir / "roi_ocr_input.png")
        metadata["roi_ocr_input_size"] = list(saved_input.size)

    if preprocess is not None:
        metadata["preprocess"] = preprocess.as_dict()

    if model_info:
        metadata["models"] = model_info

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _preview_selection_rect(monitor, preview_size, start, end):
    """Map preview pixels to physical desktop pixels, including negative origins."""
    width, height = preview_size
    def source_point(point):
        return (
            round(max(0, min(width, point[0])) * monitor["width"] / width),
            round(max(0, min(height, point[1])) * monitor["height"] / height),
        )
    return _absolute_selection_rect(monitor, source_point(start), source_point(end))


class ScreenAreaSelector(tk.Toplevel):
    def __init__(self, parent: tk.Tk, capture_rect: Rect | None = None):
        super().__init__(parent)
        self.withdraw()
        self.result_rect: Rect | None = None
        self.selection = None
        self._drawing = False
        self._render_pending = None
        try:
            with mss.mss() as sct:
                if capture_rect is None:
                    self.monitor = dict(sct.monitors[0])
                else:
                    left, top, right, bottom = capture_rect
                    self.monitor = {"left": left, "top": top, "width": right-left, "height": bottom-top}
                shot = sct.grab(self.monitor)
            self.full_image = Image.frombytes("RGB", shot.size, shot.rgb)
            self.title("화면 기준 선택 — 확대 후 정밀 선택")
            self.resizable(True, True)
            bar = ttk.Frame(self, padding=6)
            bar.pack(fill=tk.X)
            for label, command in (("전체 보기", self._fit), ("100%", lambda: self._zoom(1.0)),
                                   ("200%", lambda: self._zoom(2.0)), ("400%", lambda: self._zoom(4.0))):
                ttk.Button(bar, text=label, command=command).pack(side=tk.LEFT, padx=2)
            self.zoom_label = tk.StringVar()
            ttk.Label(bar, textvariable=self.zoom_label).pack(side=tk.LEFT, padx=8)
            ttk.Button(bar, text="선택 확정", command=self._confirm).pack(side=tk.RIGHT, padx=2)
            ttk.Button(bar, text="취소", command=self.destroy).pack(side=tk.RIGHT, padx=2)
            ttk.Label(self, text="왼쪽 드래그: 영역 선택 | 휠: 확대/축소 | 오른쪽 드래그/스크롤바: 이동 | Enter: 확정 | Esc: 취소",
                      wraplength=900, padding=6).pack(fill=tk.X)
            coordinates = ttk.Frame(self, padding=6)
            coordinates.pack(fill=tk.X)
            self.coordinate_vars = []
            for label in ("화면 X", "화면 Y", "너비(px)", "높이(px)"):
                ttk.Label(coordinates, text=label).pack(side=tk.LEFT, padx=3)
                var = tk.StringVar()
                self.coordinate_vars.append(var)
                ttk.Entry(coordinates, textvariable=var, width=8).pack(side=tk.LEFT)
            ttk.Button(coordinates, text="좌표 적용", command=self._apply_coordinates).pack(side=tk.LEFT, padx=6)
            area = ttk.Frame(self)
            area.pack(fill=tk.BOTH, expand=True)
            area.rowconfigure(0, weight=1)
            area.columnconfigure(0, weight=1)
            self.canvas = tk.Canvas(area, highlightthickness=0, borderwidth=0, cursor="cross", bg="#222222")
            self.canvas.grid(row=0, column=0, sticky="nsew")
            self.hscroll = ttk.Scrollbar(area, orient=tk.HORIZONTAL, command=lambda *a: self._scroll('x', *a))
            self.vscroll = ttk.Scrollbar(area, orient=tk.VERTICAL, command=lambda *a: self._scroll('y', *a))
            self.hscroll.grid(row=1, column=0, sticky="ew")
            self.vscroll.grid(row=0, column=1, sticky="ns")
            self.canvas.configure(xscrollcommand=self.hscroll.set, yscrollcommand=self.vscroll.set)
            self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW)
            self.rect_id = self.canvas.create_rectangle(0, 0, 0, 0, outline="red", width=2, state="hidden")
            self.preview_size = self.full_image.size
            self.scale = 1.0
            self.canvas.bind("<Configure>", lambda _e: self._queue_render())
            self.canvas.bind("<ButtonPress-1>", self._on_press)
            self.canvas.bind("<B1-Motion>", self._on_drag)
            self.canvas.bind("<ButtonRelease-1>", self._on_release)
            self.canvas.bind("<ButtonPress-3>", lambda e: self.canvas.scan_mark(e.x, e.y))
            self.canvas.bind("<B3-Motion>", self._pan)
            self.canvas.bind("<MouseWheel>", self._wheel)
            self.bind("<Return>", lambda _e: self._confirm())
            self.bind("<Escape>", lambda _e: self.destroy())
            self.protocol("WM_DELETE_WINDOW", self.destroy)
            width = max(320, min(1400, self.winfo_screenwidth()-80))
            height = max(300, min(950, self.winfo_screenheight()-100))
            self.geometry(f"{width}x{height}+30+30")
            self.deiconify()
            self.update_idletasks()
            self._fit()
            self.lift()
            self.focus_force()
        except Exception:
            self.destroy()
            raise

    def _queue_render(self):
        if self._render_pending is None:
            self._render_pending = self.after_idle(self._render_view)

    def _render_view(self):
        self._render_pending = None
        # Render only the viewport, so 400% on several 4K monitors does not
        # allocate a bitmap four times the entire desktop in each dimension.
        x, y = self.canvas.canvasx(0), self.canvas.canvasy(0)
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        sx, sy = self.full_image.width/self.preview_size[0], self.full_image.height/self.preview_size[1]
        view = self.full_image.transform((width, height), Image.Transform.AFFINE,
                                        (sx, 0, x*sx, 0, sy, y*sy),
                                        resample=Image.Resampling.NEAREST if self.scale >= 1 else Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(view, master=self)
        self.canvas.itemconfigure(self.image_id, image=self.photo)
        self.canvas.coords(self.image_id, x, y)
        self.canvas.tag_lower(self.image_id)

    def _fit(self):
        self._zoom(min(1.0, max(1, self.canvas.winfo_width())/self.full_image.width,
                       max(1, self.canvas.winfo_height())/self.full_image.height))

    def _zoom(self, scale, anchor=None):
        if self._drawing:
            return
        if anchor is None:
            anchor = (self.canvas.winfo_width()/2, self.canvas.winfo_height()/2)
        old_width, old_height = self.preview_size
        relative_x = self.canvas.canvasx(anchor[0])/old_width
        relative_y = self.canvas.canvasy(anchor[1])/old_height
        self.scale = max(0.01, min(4.0, scale))
        self.preview_size = (max(1, round(self.full_image.width*self.scale)),
                             max(1, round(self.full_image.height*self.scale)))
        width, height = self.preview_size
        self.canvas.configure(scrollregion=(0, 0, width, height))
        self.canvas.xview_moveto(max(0, relative_x-anchor[0]/width))
        self.canvas.yview_moveto(max(0, relative_y-anchor[1]/height))
        self.zoom_label.set(f"{self.scale:.0%}")
        self._draw_selection()
        self._queue_render()

    def _wheel(self, event):
        if event.delta:
            self._zoom(self.scale * (1.25 if event.delta > 0 else 0.8), (event.x, event.y))
        return "break"

    def _scroll(self, axis, *args):
        getattr(self.canvas, axis+'view')(*args)
        self._queue_render()

    def _pan(self, event):
        if not self._drawing:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            self._queue_render()

    def _point(self, event):
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def _on_press(self, event):
        self._drawing = True
        self.drag_start = self._point(event)

    def _on_drag(self, event):
        if self._drawing:
            self.selection = _preview_selection_rect(self.monitor, self.preview_size, self.drag_start, self._point(event))
            self._draw_selection()
            self._sync_coordinates()

    def _on_release(self, event):
        self._on_drag(event)
        self._drawing = False

    def _draw_selection(self):
        if self.selection is None:
            self.canvas.itemconfigure(self.rect_id, state="hidden")
            return
        x1, y1, x2, y2 = self.selection
        sx, sy = self.preview_size[0]/self.monitor['width'], self.preview_size[1]/self.monitor['height']
        self.canvas.coords(self.rect_id, (x1-self.monitor['left'])*sx, (y1-self.monitor['top'])*sy,
                           (x2-self.monitor['left'])*sx, (y2-self.monitor['top'])*sy)
        self.canvas.itemconfigure(self.rect_id, state="normal")

    def _sync_coordinates(self):
        values = ('', '', '', '') if self.selection is None else (
            self.selection[0], self.selection[1], self.selection[2]-self.selection[0], self.selection[3]-self.selection[1])
        for var, value in zip(self.coordinate_vars, values):
            var.set(value)

    def _apply_coordinates(self):
        try:
            x, y, width, height = (int(var.get()) for var in self.coordinate_vars)
            m = self.monitor
            if not (width >= 2 and height >= 2 and m['left'] <= x and m['top'] <= y
                    and x+width <= m['left']+m['width'] and y+height <= m['top']+m['height']):
                raise ValueError()
        except ValueError:
            messagebox.showwarning("영역 선택", "캡처 화면 안의 좌표와 2px 이상의 너비/높이를 입력하세요.", parent=self)
            return False
        self.selection = (x, y, x+width, y+height)
        self._draw_selection()
        return True

    def _confirm(self):
        if self._apply_coordinates():
            self.result_rect = self.selection
            self.destroy()

    def destroy(self):
        if self._render_pending is not None:
            self.after_cancel(self._render_pending)
            self._render_pending = None
        super().destroy()


class OCRValidatorGUI(AutomationWorkflow, ROIPresetWorkflow):
    def __init__(self, root: tk.Tk, engine: OCREngine):
        self.root = root
        self.engine = engine

        self.root.title("ROI OCR Validator")
        self.root.geometry("1400x860")

        self.source_image: Image.Image | None = None
        self.screen_base_rect: Rect | None = None
        self.display_scale = 1.0
        self.display_photo = None

        self.rois: dict[int, ROIItem] = {}
        self.next_roi_id = 1
        self.selected_roi_id: int | None = None

        # Pixels handed to OCREngine.run(), captured at the call site so a
        # failure diagnostic never has to reconstruct them afterwards.
        self._last_ocr_input: OCRInputRecord | None = None
        self._ocr_inputs: dict[int, OCRInputRecord] = {}

        self.drawing = False
        self.drag_start = (0, 0)
        self.preview_rect_id = None

        self.language_var = tk.StringVar(value="en_es")
        self.compare_mode_var = tk.StringVar(value="exact")
        self.similarity_threshold_var = tk.StringVar(value="0.90")
        self.duration_var = tk.StringVar(value="5")
        self.fps_var = tk.StringVar(value="2")
        self.save_images_var = tk.BooleanVar(value=False)
        self.roi_margin_var = tk.StringVar(value="8")
        self.min_roi_side_var = tk.StringVar(value="160")
        self.auto_upscale_var = tk.BooleanVar(value=True)
        self.fast_long_roi_var = tk.BooleanVar(value=True)
        self.context_detect_var = tk.BooleanVar(value=False)
        self.context_margin_var = tk.StringVar(value="120")
        self.scroll_mode_var = tk.BooleanVar(value=False)
        self.vertical_list_mode_var = tk.BooleanVar(value=False)
        self.scroll_fps_var = tk.StringVar(value="8")
        self.live_verify_var = tk.BooleanVar(value=True)
        self.live_fps_var = tk.StringVar(value="2")

        self.live_stop_event = threading.Event()
        self.live_thread: threading.Thread | None = None
        self.live_monitor_running = False
        self.live_ocr_active = threading.Event()
        self.live_accumulators: dict[int, ScrollTextAccumulator] = {}
        self.live_samplers: dict[int, AdaptiveFrameSampler] = {}
        self.live_log_lines: list[str] = []
        self.max_live_log_lines = 300

        self.status_var = tk.StringVar(value="Ready")

        self._init_presets()
        self._build_layout()

    def _build_layout(self):
        toolbar_outer = ttk.Frame(self.root, padding=0)
        toolbar_outer.pack(fill=tk.X)

        top_canvas = tk.Canvas(toolbar_outer, height=86, highlightthickness=0)
        top_hscroll = ttk.Scrollbar(toolbar_outer, orient=tk.HORIZONTAL, command=top_canvas.xview)
        top_canvas.configure(xscrollcommand=top_hscroll.set)
        top_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)
        top_hscroll.pack(side=tk.TOP, fill=tk.X)

        top = ttk.Frame(top_canvas, padding=8)
        top_window = top_canvas.create_window((0, 0), window=top, anchor=tk.NW)

        def _update_toolbar_scrollregion(_event=None):
            top.update_idletasks()
            top_canvas.configure(scrollregion=top_canvas.bbox("all"))
            top_canvas.itemconfigure(top_window, height=top.winfo_reqheight())

        def _sync_toolbar_width(_event=None):
            top_canvas.itemconfigure(top_window, width=max(top.winfo_reqwidth(), top_canvas.winfo_width()))
            _update_toolbar_scrollregion()

        top.bind("<Configure>", _update_toolbar_scrollregion)
        top_canvas.bind("<Configure>", _sync_toolbar_width)

        row_a = ttk.Frame(top)
        row_b = ttk.Frame(top)
        row_a.pack(fill=tk.X, pady=(0, 4))
        row_b.pack(fill=tk.X)

        ttk.Button(row_a, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(row_a, text="Capture Screen Area", command=self.capture_screen_area).pack(side=tk.LEFT, padx=4)
        ttk.Button(row_a, text="Clear ROIs", command=self.clear_rois).pack(side=tk.LEFT, padx=4)

        ttk.Label(row_a, text="Language").pack(side=tk.LEFT, padx=(20, 4))
        lang_box = ttk.Combobox(row_a, textvariable=self.language_var, state="readonly", width=10)
        lang_box["values"] = ("en_es", "fr", "zh")
        lang_box.pack(side=tk.LEFT)

        ttk.Label(row_a, text="Compare").pack(side=tk.LEFT, padx=(20, 4))
        mode_box = ttk.Combobox(row_a, textvariable=self.compare_mode_var, state="readonly", width=14)
        mode_box["values"] = ("exact", "ignore_case", "ignore_space", "similarity", "regex")
        mode_box.pack(side=tk.LEFT)

        ttk.Label(row_a, text="Similarity").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Entry(row_a, textvariable=self.similarity_threshold_var, width=6).pack(side=tk.LEFT)

        ttk.Button(row_a, text="Run Once", command=self.run_once).pack(side=tk.LEFT, padx=(20, 4))
        ttk.Button(row_a, text="Run Timed Capture", command=self.run_timed_capture).pack(side=tk.LEFT, padx=4)

        ttk.Label(row_a, text="Duration(s)").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Entry(row_a, textvariable=self.duration_var, width=6).pack(side=tk.LEFT)

        ttk.Label(row_a, text="FPS").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Entry(row_a, textvariable=self.fps_var, width=4).pack(side=tk.LEFT)

        ttk.Checkbutton(row_a, text="Save Images", variable=self.save_images_var).pack(side=tk.LEFT, padx=(10, 4))

        ttk.Button(row_b, text="Start Live", command=self.start_live_monitor).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(row_b, text="Stop Live", command=self.stop_live_monitor).pack(side=tk.LEFT, padx=4)
        ttk.Button(row_b, text="Start OCR", command=self.start_live_ocr).pack(side=tk.LEFT, padx=(12, 4))
        ttk.Button(row_b, text="Stop OCR", command=self.stop_live_ocr).pack(side=tk.LEFT, padx=4)
        ttk.Button(row_b, text="Clear Log", command=self.clear_live_log).pack(side=tk.LEFT, padx=4)

        ttk.Label(row_b, text="ROI Margin(px)").pack(side=tk.LEFT, padx=(20, 4))
        ttk.Entry(row_b, textvariable=self.roi_margin_var, width=4).pack(side=tk.LEFT)

        ttk.Label(row_b, text="Min Side(px)").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Entry(row_b, textvariable=self.min_roi_side_var, width=4).pack(side=tk.LEFT)

        ttk.Checkbutton(row_b, text="Auto Upscale", variable=self.auto_upscale_var).pack(side=tk.LEFT, padx=(6, 4))
        ttk.Checkbutton(row_b, text="Fast Long ROI", variable=self.fast_long_roi_var).pack(side=tk.LEFT, padx=(6, 4))

        ttk.Checkbutton(row_b, text="Context Detect", variable=self.context_detect_var).pack(side=tk.LEFT, padx=(10, 4))
        ttk.Label(row_b, text="Context Margin(px)").pack(side=tk.LEFT, padx=(6, 4))
        ttk.Entry(row_b, textvariable=self.context_margin_var, width=4).pack(side=tk.LEFT)

        ttk.Checkbutton(
            row_b,
            text="Scrolling / Loop",
            variable=self.scroll_mode_var,
            command=self._on_scroll_mode_changed,
        ).pack(side=tk.LEFT, padx=(10, 4))
        ttk.Checkbutton(
            row_b,
            text="Vertical Rows",
            variable=self.vertical_list_mode_var,
            command=self._on_vertical_list_mode_changed,
        ).pack(side=tk.LEFT, padx=(6, 4))
        ttk.Label(row_b, text="Scroll FPS").pack(side=tk.LEFT, padx=(6, 4))
        ttk.Entry(row_b, textvariable=self.scroll_fps_var, width=4).pack(side=tk.LEFT)

        ttk.Checkbutton(row_b, text="Live Verify", variable=self.live_verify_var).pack(side=tk.LEFT, padx=(10, 4))
        ttk.Label(row_b, text="Live FPS").pack(side=tk.LEFT, padx=(6, 4))
        ttk.Entry(row_b, textvariable=self.live_fps_var, width=4).pack(side=tk.LEFT)

        self._build_preset_toolbar(self.root)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.canvas = tk.Canvas(left, bg="#1e1e1e", width=1000, height=720, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        ttk.Label(right, text="ROIs").pack(anchor=tk.W)
        self.roi_list = tk.Listbox(right, height=10)
        self.roi_list.pack(fill=tk.X)
        self.roi_list.bind("<<ListboxSelect>>", self._on_select_roi)

        ttk.Label(right, text="Expected Text (one expected row per line)").pack(anchor=tk.W, pady=(8, 0))
        self.expected_text = tk.Text(right, height=7, wrap=tk.NONE)
        self.expected_text.pack(fill=tk.X)
        ttk.Button(right, text="Apply Expected", command=self.apply_expected).pack(anchor=tk.E, pady=(4, 8))

        ttk.Label(right, text="Results").pack(anchor=tk.W)
        columns = ("roi", "expected", "actual", "score", "result")
        self.result_tree = ttk.Treeview(right, columns=columns, show="headings", height=14)
        for col in columns:
            self.result_tree.heading(col, text=col.upper())
        self.result_tree.column("roi", width=40, anchor=tk.CENTER)
        self.result_tree.column("expected", width=160, anchor=tk.W)
        self.result_tree.column("actual", width=200, anchor=tk.W)
        self.result_tree.column("score", width=70, anchor=tk.CENTER)
        self.result_tree.column("result", width=70, anchor=tk.CENTER)
        self.result_tree.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Live Log").pack(anchor=tk.W, pady=(8, 0))
        log_frame = ttk.Frame(right)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.live_log_text = tk.Text(log_frame, height=10, wrap=tk.NONE)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.live_log_text.yview)
        self.live_log_text.configure(yscrollcommand=log_scroll.set)
        self.live_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        status = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W, padding=6)
        status.pack(fill=tk.X)

    def load_image(self):
        if not self._roi_changes_allowed():
            return
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")],
        )
        if not path:
            return

        self.source_image = Image.open(path).convert("RGB")
        self.screen_base_rect = None
        self.clear_rois()
        self._refresh_canvas()
        self.status_var.set(f"Loaded image: {Path(path).name}")

    def capture_screen_area(self):
        if not self._roi_changes_allowed():
            return
        selector = ScreenAreaSelector(self.root)
        selector.grab_set()
        self.root.wait_window(selector)

        if not selector.result_rect:
            return

        self.screen_base_rect = selector.result_rect
        left, top, right, bottom = self.screen_base_rect

        with mss.mss() as sct:
            monitor = {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
            }
            shot = sct.grab(monitor)

        self.source_image = Image.frombytes("RGB", shot.size, shot.rgb)
        self.clear_rois()
        self._refresh_canvas()
        self.status_var.set(
            f"Captured screen area: ({left}, {top}) - ({right}, {bottom})"
        )

    def _refresh_canvas(self):
        self.canvas.delete("all")
        if self.source_image is None:
            return

        c_w = max(1, self.canvas.winfo_width())
        c_h = max(1, self.canvas.winfo_height())
        if c_w < 20 or c_h < 20:
            c_w, c_h = 1000, 720

        src_w, src_h = self.source_image.size
        self.display_scale = min(c_w / src_w, c_h / src_h, 1.0)
        disp_w = int(src_w * self.display_scale)
        disp_h = int(src_h * self.display_scale)

        display_img = self.source_image.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self.display_photo = ImageTk.PhotoImage(display_img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_photo)

        for roi in self.rois.values():
            self._draw_roi(roi)

    def _draw_roi(self, roi: ROIItem):
        x1, y1, x2, y2 = roi.rect
        sx1 = int(x1 * self.display_scale)
        sy1 = int(y1 * self.display_scale)
        sx2 = int(x2 * self.display_scale)
        sy2 = int(y2 * self.display_scale)

        color = "#2ecc71" if roi.passed else "#f1c40f"
        self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline=color, width=2)
        self.canvas.create_text(sx1 + 6, sy1 + 12, text=f"ROI {roi.roi_id}", fill=color, anchor=tk.W)

    def _canvas_to_src(self, x: int, y: int) -> tuple[int, int]:
        if self.source_image is None:
            return 0, 0
        src_w, src_h = self.source_image.size
        sx = int(x / self.display_scale)
        sy = int(y / self.display_scale)
        sx = max(0, min(src_w - 1, sx))
        sy = max(0, min(src_h - 1, sy))
        return sx, sy

    def _on_canvas_press(self, event):
        if not self._roi_changes_allowed():
            return
        if self.source_image is None:
            return
        self.drawing = True
        self.drag_start = (event.x, event.y)
        if self.preview_rect_id:
            self.canvas.delete(self.preview_rect_id)
        self.preview_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#00bcd4", width=2)

    def _on_canvas_drag(self, event):
        if not self.drawing or not self.preview_rect_id:
            return
        sx, sy = self.drag_start
        self.canvas.coords(self.preview_rect_id, sx, sy, event.x, event.y)

    def _on_canvas_release(self, event):
        if not self.drawing:
            return
        self.drawing = False

        sx, sy = self.drag_start
        ex, ey = event.x, event.y
        x1, y1 = min(sx, ex), min(sy, ey)
        x2, y2 = max(sx, ex), max(sy, ey)

        if x2 - x1 < 3 or y2 - y1 < 3:
            if self.preview_rect_id:
                self.canvas.delete(self.preview_rect_id)
            self.preview_rect_id = None
            return

        src_x1, src_y1 = self._canvas_to_src(x1, y1)
        src_x2, src_y2 = self._canvas_to_src(x2, y2)

        roi = ROIItem(roi_id=self.next_roi_id, rect=(src_x1, src_y1, src_x2, src_y2))
        self.rois[roi.roi_id] = roi
        self.next_roi_id += 1

        if self.preview_rect_id:
            self.canvas.delete(self.preview_rect_id)
        self.preview_rect_id = None

        self._sync_roi_list()
        self._refresh_canvas()
        self.status_var.set(f"Added ROI {roi.roi_id}")

    def _sync_roi_list(self):
        self.roi_list.delete(0, tk.END)
        for roi_id in sorted(self.rois):
            roi = self.rois[roi_id]
            x1, y1, x2, y2 = roi.rect
            self.roi_list.insert(tk.END, f"ROI {roi_id}: ({x1},{y1})-({x2},{y2})")

    def _on_select_roi(self, _event):
        sel = self.roi_list.curselection()
        if not sel:
            return
        idx = sel[0]
        roi_id = sorted(self.rois)[idx]
        self.selected_roi_id = roi_id
        self.expected_text.delete("1.0", tk.END)
        self.expected_text.insert("1.0", self.rois[roi_id].expected)

    def apply_expected(self):
        if not self._roi_changes_allowed():
            return
        if self.selected_roi_id is None:
            messagebox.showinfo("Info", "Select an ROI first.")
            return
        self.rois[self.selected_roi_id].expected = self.expected_text.get("1.0", "end-1c")
        self.status_var.set(f"Updated expected text for ROI {self.selected_roi_id}")

    def clear_rois(self):
        if not self._roi_changes_allowed():
            return
        self._preset_direction = None
        self.rois.clear()
        self.next_roi_id = 1
        self.selected_roi_id = None
        self.expected_text.delete("1.0", tk.END)
        self._sync_roi_list()
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)
        self._refresh_canvas()

    def clear_live_log(self):
        self.live_log_lines.clear()
        self.live_log_text.delete("1.0", tk.END)

    def _append_live_log(self, line: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.live_log_lines.append(f"[{timestamp}] {line}")
        if len(self.live_log_lines) > self.max_live_log_lines:
            self.live_log_lines = self.live_log_lines[-self.max_live_log_lines :]
        self.live_log_text.delete("1.0", tk.END)
        self.live_log_text.insert(tk.END, "\n".join(self.live_log_lines) + "\n")
        self.live_log_text.see(tk.END)

    def _build_result_map(
        self,
        text_map: dict[int, str],
        use_accumulator_results: bool = False,
    ) -> dict[int, tuple[bool, str, str]]:
        mode = self.compare_mode_var.get()
        try:
            threshold = float(self.similarity_threshold_var.get())
        except ValueError:
            threshold = 0.9

        result_map: dict[int, tuple[bool, str, str]] = {}
        for roi_id in sorted(self.rois):
            roi = self.rois[roi_id]
            roi.actual = text_map.get(roi_id, "")
            live_accumulator = self.live_accumulators.get(roi_id)
            if use_accumulator_results and isinstance(live_accumulator, VerticalListAccumulator):
                roi.passed = live_accumulator.passed
                all_rows_seen = live_accumulator.coverage == 1.0
                score = live_accumulator.flat_coverage if all_rows_seen else live_accumulator.coverage
                if roi.passed:
                    result = "PASS"
                elif all_rows_seen:
                    result = f"MISMATCH {live_accumulator.flat_coverage:.1%}"
                else:
                    result = "SCANNING"
                result_map[roi_id] = (
                    roi.passed,
                    f"{score:.2f}",
                    result,
                )
                continue
            if use_accumulator_results and getattr(live_accumulator, "evidence", None) is not None:
                roi.actual = live_accumulator.final_text
                roi.passed = live_accumulator.cycle_complete
                result_map[roi_id] = (roi.passed, f"{live_accumulator.coverage:.2f}",
                                      "PASS" if roi.passed else "SCANNING")
                continue
            if roi.expected.strip():
                cmp = compare_text(roi.expected, roi.actual, mode=mode, similarity_threshold=threshold)
                roi.passed = cmp.passed
                result_map[roi_id] = (cmp.passed, f"{cmp.score:.2f}", "PASS" if cmp.passed else "FAIL")
            else:
                roi.passed = True
                result_map[roi_id] = (True, "-", "N/A")
        return result_map

    def _render_result_map(self, text_map: dict[int, str], use_accumulator_results: bool = False):
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)

        result_map = self._build_result_map(text_map, use_accumulator_results)
        for roi_id in sorted(self.rois):
            roi = self.rois[roi_id]
            _, score, result = result_map[roi_id]
            self._maybe_save_failure(roi_id, roi, use_accumulator_results)
            self.result_tree.insert(
                "",
                tk.END,
                values=(roi_id, roi.expected.replace("\n", " | "), roi.actual.replace("\n", " | "), score, result),
            )

        self._refresh_canvas()

    def _maybe_save_failure(
        self, roi_id: int, roi: ROIItem, use_accumulator_results: bool = False
    ) -> None:
        """Persist a diagnostic for a failed ROI.

        Expected text is used here only to decide that this ROI failed and is
        worth saving. It has already played no part in choosing a recognizer or
        in the OCR result itself.
        """
        if roi.passed or not roi.expected.strip() or self.source_image is None:
            return
        # Accumulator results are merged across frames, so no single recorded
        # image is "the" OCR input.
        record = None if use_accumulator_results else self._ocr_inputs.get(roi_id)
        try:
            diagnostic_dir = _save_failed_roi_diagnostic(
                self.source_image,
                roi,
                preprocess=self._roi_preprocess_config(),
                model_info=self._resolved_model_info(),
                ocr_input=record,
            )
            self._append_live_log(f"ROI{roi_id} FAILURE_SAVED={diagnostic_dir}")
            # Say plainly whether this capture can be used as an exact target
            # input, so nobody has to read metadata.json to find out.
            if record is not None and record.exact:
                self._append_live_log(
                    f"ROI{roi_id} CAPTURE=USABLE (exact recognizer input recorded)"
                )
            else:
                if use_accumulator_results:
                    reason = "scroll/live result merged from multiple frames"
                    hint = "use Run Once with Context Detect OFF"
                elif record is None:
                    reason = "no OCR input recorded for this ROI"
                    hint = "use Run Once with Context Detect OFF"
                elif record.path_kind == "context_fallback":
                    reason = "detector found no text in the ROI, so context fallback ran"
                    hint = "enlarge the ROI around the text and retry Run Once"
                else:
                    reason = f"OCR ran via '{record.path_kind}', not the direct ROI path"
                    hint = "turn Context Detect OFF and use Run Once"
                self._append_live_log(
                    f"ROI{roi_id} CAPTURE=NOT USABLE ({reason}); {hint}"
                )
        except OSError as exc:
            self._append_live_log(f"ROI{roi_id} FAILURE_SAVE_ERROR={exc}")

    def _append_result_summary(self, text_map: dict[int, str]):
        parts = []
        for roi_id in sorted(self.rois):
            text = text_map.get(roi_id, "")
            if text:
                parts.append(f"ROI{roi_id}={text.replace(chr(10), ' | ')}")
        if parts:
            self._append_live_log("; ".join(parts))

    @staticmethod
    def _safe_int(value: str, default: int, min_value: int, max_value: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(min_value, min(max_value, parsed))

    @staticmethod
    def _safe_float(value: str, default: float, min_value: float, max_value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        return max(min_value, min(max_value, parsed))

    def _roi_preprocess_config(self) -> RoiPreprocessConfig:
        return RoiPreprocessConfig(
            margin=self._safe_int(
                self.roi_margin_var.get(), default=8, min_value=0, max_value=80
            ),
            min_side=self._safe_int(
                self.min_roi_side_var.get(), default=160, min_value=32, max_value=640
            ),
            pad_long_roi=bool(self.fast_long_roi_var.get()),
            auto_upscale=bool(self.auto_upscale_var.get()),
        )

    def _crop_roi(self, image: Image.Image, rect: Rect) -> Image.Image:
        return crop_roi(image, rect, self._roi_preprocess_config())

    def _resolved_model_info(self) -> dict:
        """Absolute paths and SHA-256 of the models actually in use.

        Recorded in failure diagnostics so a saved failure can be tied to the
        exact binaries that produced it. Best-effort: a hashing failure must
        never prevent the diagnostic itself from being written.
        """
        package = getattr(self.engine, "package", None)
        if package is None:
            return {}

        def describe(path) -> dict:
            try:
                resolved = Path(path).resolve()
                digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
                return {"path": str(resolved), "sha256": digest}
            except OSError as exc:
                return {"path": str(path), "error": str(exc)}

        info: dict = {"language": self.language_var.get()}
        try:
            info["detector"] = describe(package.detector_model)
            info["dictionary"] = describe(package.dictionary)
            info["recognizers"] = {
                name: describe(path) for name, path in package.recognizers.items()
            }
        except AttributeError:
            return info
        return info

    def _evaluate_rois(self, frame_image: Image.Image):
        text_map: dict[int, str] = {}
        self._ocr_inputs = {}
        for roi_id in sorted(self.rois):
            roi = self.rois[roi_id]
            ocr = self._run_roi_ocr(frame_image, roi.rect, roi.expected)
            text_map[roi_id] = ocr.text
            spacing = getattr(ocr, "spacing_evidence", None)
            if isinstance(spacing, dict):
                self._append_live_log(f"ROI{roi_id} IMAGE_SPACING={spacing['status']} "
                                      f"RAW={spacing['raw_text']!r} RESULT={ocr.text!r}")
            if self._last_ocr_input is not None:
                self._ocr_inputs[roi_id] = self._last_ocr_input

        self._render_result_map(text_map)

    def _run_engine(
        self,
        image: Image.Image,
        expected_text: str,
        record_as: str | None = None,
    ):
        """Run OCR on ``image``.

        When ``record_as`` is given, the exact pixels passed to the engine are
        copied and stashed on ``self._last_ocr_input`` before inference. The
        copy matters: the caller may reuse or mutate the image afterwards, and
        the recorded input has to stay byte-identical to what was recognized.
        """
        language = self.language_var.get()
        if record_as is not None:
            self._last_ocr_input = OCRInputRecord(
                image=image.copy(),
                path_kind=record_as,
                language=language,
                exact=(record_as == "direct"),
            )
        result = self.engine.run(
            image,
            language,
            preserve_small_left_noise=has_ui_image_token(expected_text),
        )
        if record_as is not None and self._last_ocr_input is not None:
            # Raw recognizer output, before the expected-aware UI normalization.
            self._last_ocr_input.raw_ocr_text = result.text
        from .image_spacing import apply_image_spacing
        apply_image_spacing(image, result, expected_text)
        if record_as is not None and self._last_ocr_input is not None:
            self._last_ocr_input.spacing_evidence = result.spacing_evidence
        result.text = normalize_ocr_ui_text(result.text, expected_text)
        return result

    def _run_roi_ocr(self, frame_image: Image.Image, roi_rect: Rect, expected_text: str = ""):
        self._last_ocr_input = None
        if not self.context_detect_var.get():
            crop = self._crop_roi(frame_image, roi_rect)
            direct_ocr = self._run_engine(crop, expected_text, record_as="direct")
            if direct_ocr.boxes:
                return direct_ocr
            return self._run_context_roi_ocr(
                frame_image,
                roi_rect,
                expected_text=expected_text,
                fallback_result=direct_ocr,
            )

        return self._run_context_roi_ocr(frame_image, roi_rect, expected_text=expected_text)

    def _run_context_roi_ocr(
        self,
        frame_image: Image.Image,
        roi_rect: Rect,
        expected_text: str = "",
        fallback_result=None,
    ):

        x1, y1, x2, y2 = roi_rect
        img_w, img_h = frame_image.size
        context_margin = self._safe_int(self.context_margin_var.get(), default=120, min_value=0, max_value=600)

        cx1 = max(0, x1 - context_margin)
        cy1 = max(0, y1 - context_margin)
        cx2 = min(img_w, x2 + context_margin)
        cy2 = min(img_h, y2 + context_margin)

        context_crop = frame_image.crop((cx1, cy1, cx2, cy2))
        context_ocr = self._run_engine(context_crop, expected_text, record_as="context")

        if not context_ocr.boxes:
            if fallback_result is not None:
                return fallback_result
            # Fallback to direct ROI mode when context detect finds nothing.
            direct_crop = self._crop_roi(frame_image, roi_rect)
            return self._run_engine(direct_crop, expected_text, record_as="context_fallback")

        filtered = []
        for box in context_ocr.boxes:
            gx = cx1 + box.center_x
            gy = cy1 + box.center_y
            if x1 <= gx <= x2 and y1 <= gy <= y2:
                filtered.append(box)

        if not filtered:
            if fallback_result is not None:
                return fallback_result
            direct_crop = self._crop_roi(frame_image, roi_rect)
            return self._run_engine(direct_crop, expected_text, record_as="context_fallback")

        text, mean_score = self.engine.text_from_boxes(filtered)
        context_ocr.text = text
        context_ocr.mean_score = mean_score
        context_ocr.n_boxes = len(filtered)
        context_ocr.boxes = filtered
        from .image_spacing import apply_image_spacing
        apply_image_spacing(context_crop, context_ocr, expected_text)
        if self._last_ocr_input is not None:
            self._last_ocr_input.spacing_evidence = context_ocr.spacing_evidence
        context_ocr.text = normalize_ocr_ui_text(context_ocr.text, expected_text)
        return context_ocr

    def _apply_live_frame(self, frame_image: Image.Image, text_map: dict[int, str], log_active: bool = False):
        self.source_image = frame_image
        self._render_result_map(text_map, use_accumulator_results=True)
        if log_active:
            self._append_result_summary(text_map)

    def _new_live_accumulator(self, roi: ROIItem):
        expected_rows = [row for row in roi.expected.splitlines() if row.strip()]
        if not expected_rows:
            return ScrollTextAccumulator(min_length=3, min_score=0.25)
        if self.vertical_list_mode_var.get() or (
            len(expected_rows) > 1 and getattr(self, "_preset_direction", None) != "horizontal"
        ):
            return VerticalListAccumulator(
                expected_rows,
                require_loop=self._scrolling_enabled(),
            )
        try:
            threshold = float(self.similarity_threshold_var.get())
        except ValueError:
            threshold = 0.9
        return ScrollTextAccumulator(
            min_length=max(1, min(3, len(normalize_ui_text(roi.expected)))),
            min_score=0.25,
            expected_text=roi.expected,
            track_observed_text=True,
            compare_mode=self.compare_mode_var.get(),
            similarity_threshold=threshold,
        )

    def _scrolling_enabled(self) -> bool:
        return self.scroll_mode_var.get() or self.vertical_list_mode_var.get()

    def _on_scroll_mode_changed(self) -> None:
        if not self.scroll_mode_var.get():
            self.vertical_list_mode_var.set(False)

    def _on_vertical_list_mode_changed(self) -> None:
        self._preset_direction = "vertical" if self.vertical_list_mode_var.get() else "horizontal"
        if self.vertical_list_mode_var.get():
            self.scroll_mode_var.set(True)

    def start_live_ocr(self):
        if self.live_ocr_active.is_set():
            self._append_live_log("OCR is already active; existing progress was kept")
            return
        self.live_accumulators = {
            roi_id: self._new_live_accumulator(self.rois[roi_id])
            for roi_id in sorted(self.rois)
        }
        self.live_samplers = {roi_id: AdaptiveFrameSampler() for roi_id in sorted(self.rois)}
        self.live_ocr_active.set()
        self.live_verify_var.set(True)
        modes = {
            (
                "vertical"
                if isinstance(accumulator, VerticalListAccumulator) and accumulator.require_loop
                else "static-multiline"
                if isinstance(accumulator, VerticalListAccumulator)
                else "horizontal"
            )
            for roi_id, accumulator in self.live_accumulators.items()
            if self.rois[roi_id].expected.strip()
        }
        self._append_live_log(f"OCR started by user; mode={'+'.join(sorted(modes)) or 'unverified'}")
        self.status_var.set("OCR active")

    def stop_live_ocr(self):
        self.live_ocr_active.clear()
        self.live_verify_var.set(False)
        self._append_live_log("OCR stopped by user")
        self._finalize_scroll_results(self.live_accumulators)
        self.status_var.set("OCR stopped")

    def start_live_monitor(self):
        if getattr(self, "_automation_running", False):
            messagebox.showwarning("Automation", "Stop automation before starting Live.")
            return
        if self._timed_capture_running:
            messagebox.showwarning("ROI", "Wait for timed capture to finish.")
            return
        if self.screen_base_rect is None:
            messagebox.showwarning("Warning", "Timed/live monitoring requires screen capture mode.")
            return
        if self.live_thread and self.live_thread.is_alive():
            messagebox.showinfo("Info", "Live monitor is already running.")
            return

        try:
            fps = float(self.live_fps_var.get())
        except ValueError:
            messagebox.showerror("Error", "Live FPS must be numeric.")
            return
        if fps <= 0:
            messagebox.showerror("Error", "Live FPS must be > 0.")
            return

        self.live_stop_event.clear()
        self.live_monitor_running = True
        self.live_accumulators = {
            roi_id: self._new_live_accumulator(self.rois[roi_id])
            for roi_id in sorted(self.rois)
        }
        self.live_samplers = {roi_id: AdaptiveFrameSampler() for roi_id in sorted(self.rois)}

        def worker():
            interval = 1.0 / fps
            capture_session = mss.mss()
            while not self.live_stop_event.is_set():
                start = perf_counter()
                verification_completed = False
                try:
                    frame = grab_screen_rect_with(capture_session, self.screen_base_rect)
                    text_map: dict[int, str] = {}
                    log_active = self.live_verify_var.get() and self.live_ocr_active.is_set()
                    if log_active:
                        sampled_any = False
                        for roi_id in sorted(self.rois):
                            roi = self.rois[roi_id]
                            sampler = self.live_samplers.setdefault(roi_id, AdaptiveFrameSampler())
                            if not sampler.should_sample(frame.crop(roi.rect), perf_counter()):
                                text_map[roi_id] = self.live_accumulators[roi_id].final_text
                                continue
                            ocr = self._run_roi_ocr(frame, roi.rect, roi.expected)
                            acc = self.live_accumulators[roi_id]
                            if isinstance(acc, VerticalListAccumulator):
                                acc.add(ocr.text, ocr.mean_score)
                            else:
                                acc.add(ocr.text, ocr.mean_score, observed_at=perf_counter())
                            if isinstance(acc, ScrollTextAccumulator) and acc.expected_text and acc.coverage == 0:
                                raw_text = ocr.text.replace("\n", " | ") if ocr.text else "<no text>"
                                text_map[roi_id] = f"RAW={raw_text} [score={ocr.mean_score:.2f}, boxes={ocr.n_boxes}]"
                            else:
                                text_map[roi_id] = acc.final_text
                            sampled_any = True
                        log_active = sampled_any
                        expected_accumulators = [
                            accumulator
                            for roi_id, accumulator in self.live_accumulators.items()
                            if self.rois[roi_id].expected.strip()
                        ]
                        if expected_accumulators and all(
                            (
                                accumulator.passed
                                if isinstance(accumulator, VerticalListAccumulator)
                                else accumulator.ready_to_stop(perf_counter())
                            )
                            for accumulator in expected_accumulators
                        ):
                            self.live_ocr_active.clear()
                            verification_completed = True
                    else:
                        text_map = {roi_id: self.live_accumulators.get(roi_id, ScrollTextAccumulator()).final_text for roi_id in self.rois}

                    self.root.after(0, self._apply_live_frame, frame, text_map, log_active)
                    self.root.after(0, self.status_var.set, "Live monitoring...")
                    if verification_completed:
                        self.root.after(0, self._complete_live_ocr_automatically)
                except Exception as exc:
                    self.root.after(0, messagebox.showerror, "Live Monitor Error", str(exc))
                    break

                elapsed = perf_counter() - start
                remaining = interval - elapsed
                if remaining > 0:
                    self.live_stop_event.wait(remaining)

            capture_session.close()
            self.root.after(0, self._finalize_live_monitor)

        self.live_thread = threading.Thread(target=worker, daemon=True)
        self.live_thread.start()
        self.status_var.set("Live monitor started")
        self._append_live_log("Live monitor started")

    def _complete_live_ocr_automatically(self) -> None:
        self.live_verify_var.set(False)
        self._append_live_log("Content verification complete; OCR stopped automatically")
        self._finalize_scroll_results(self.live_accumulators)
        self.status_var.set("OCR passed and stopped")

    def stop_live_monitor(self):
        self.live_stop_event.set()
        self.live_monitor_running = False
        self.live_ocr_active.clear()
        self.live_verify_var.set(False)
        self._append_live_log("Live monitor stopping")
        self.status_var.set("Stopping live monitor...")

    def _finalize_live_monitor(self):
        if not self.live_accumulators:
            self.status_var.set("Live monitor stopped")
            return
        self._finalize_scroll_results(self.live_accumulators)
        self.status_var.set("Live monitor stopped")
        self._append_live_log("Live monitor stopped")

    def run_once(self):
        if not self._roi_changes_allowed():
            return
        if self.source_image is None:
            messagebox.showwarning("Warning", "Load an image or capture a screen area first.")
            return
        if not self.rois:
            messagebox.showwarning("Warning", "Add at least one ROI.")
            return

        try:
            if self.screen_base_rect is not None:
                with mss.mss() as sct:
                    self.source_image = grab_screen_rect_with(sct, self.screen_base_rect)
            self._evaluate_rois(self.source_image)
            self.status_var.set("Run once completed")
        except Exception as exc:
            self.status_var.set("Run failed")
            messagebox.showerror("OCR Error", str(exc))

    def run_timed_capture(self):
        if not self._roi_changes_allowed():
            return
        if not self.rois:
            messagebox.showwarning("Warning", "Add at least one ROI.")
            return
        if self.screen_base_rect is None:
            messagebox.showwarning("Warning", "Timed capture requires screen area mode.")
            return

        try:
            duration = float(self.duration_var.get())
            fps = float(self.scroll_fps_var.get() if self._scrolling_enabled() else self.fps_var.get())
        except ValueError:
            messagebox.showerror("Error", "Duration/FPS must be numeric.")
            return

        if duration <= 0 or fps <= 0:
            messagebox.showerror("Error", "Duration/FPS must be > 0.")
            return

        save_dir = None
        if self.save_images_var.get():
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            save_dir = Path("captures") / run_name

        self._timed_capture_running = True
        self.status_var.set("Timed capture running...")

        def worker():
            try:
                if self._scrolling_enabled():
                    self._run_scroll_capture(duration=duration, fps=fps, save_dir=save_dir)
                    return

                for frame in timed_capture(
                    rect=self.screen_base_rect,
                    duration_sec=duration,
                    fps=fps,
                    save_dir=save_dir,
                    save_prefix="frame",
                ):
                    self.source_image = frame.image
                    self.root.after(0, self._evaluate_rois, frame.image)
                    self.root.after(0, self.status_var.set, f"Frame {frame.index + 1} processed")
            except Exception as exc:
                self.root.after(0, messagebox.showerror, "Timed Capture Error", str(exc))
            finally:
                self.root.after(0, setattr, self, "_timed_capture_running", False)
                self.root.after(0, self.status_var.set, "Timed capture completed")

        threading.Thread(target=worker, daemon=True).start()

    def _run_scroll_capture(self, duration: float, fps: float, save_dir: Path | None):
        accumulators = {
            roi_id: self._new_live_accumulator(self.rois[roi_id])
            for roi_id in sorted(self.rois)
        }
        samplers = {roi_id: AdaptiveFrameSampler() for roi_id in sorted(self.rois)}
        ocr_calls = 0

        total_frames = max(1, int(duration * fps))

        for frame in timed_capture(
            rect=self.screen_base_rect,
            duration_sec=duration,
            fps=fps,
            save_dir=save_dir,
            save_prefix="scroll",
        ):
            self.source_image = frame.image
            for roi_id in sorted(self.rois):
                roi = self.rois[roi_id]
                roi_image = frame.image.crop(roi.rect)
                if not samplers[roi_id].should_sample(roi_image, perf_counter()):
                    continue
                ocr = self._run_roi_ocr(frame.image, roi.rect, roi.expected)
                accumulator = accumulators[roi_id]
                if isinstance(accumulator, VerticalListAccumulator):
                    accumulator.add(ocr.text, ocr.mean_score)
                else:
                    accumulator.add(ocr.text, ocr.mean_score, observed_at=perf_counter())
                ocr_calls += 1
            self.root.after(
                0,
                self.status_var.set,
                f"Scroll frame {frame.index + 1}/{total_frames}; OCR calls {ocr_calls}",
            )

        self.root.after(0, self._finalize_scroll_results, accumulators)

    def _finalize_scroll_results(self, accumulators):
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)

        mode = self.compare_mode_var.get()
        try:
            threshold = float(self.similarity_threshold_var.get())
        except ValueError:
            threshold = 0.9

        for roi_id in sorted(self.rois):
            roi = self.rois[roi_id]
            accumulator = accumulators.get(roi_id)
            if isinstance(accumulator, VerticalListAccumulator):
                accumulator.finalize()
            final_text = accumulator.final_text if accumulator is not None else ""
            roi.actual = final_text

            if isinstance(accumulator, VerticalListAccumulator):
                roi.passed = accumulator.passed
                if accumulator.passed:
                    result = "PASS"
                elif accumulator.flat_passed:
                    result = "FLAT ONLY"
                else:
                    result = "FAIL"
                score = f"{accumulator.coverage:.2f}"
                self._append_live_log(
                    f"ROI{roi_id} ROWS={len(accumulator.unique_indices)}/{len(accumulator.expected_rows)} "
                    f"ORDER={'Y' if accumulator.order_valid else 'N'} "
                    f"CONTENT={'PASS' if accumulator.passed else 'FAIL'} "
                    f"LAYOUT={'PASS' if accumulator.layout_passed else 'FAIL'} "
                    f"FLAT={'PASS' if accumulator.flat_passed else 'FAIL'} "
                    f"FLAT_COVERAGE={accumulator.flat_coverage:.0%} "
                    f"LOOP={accumulator.loop_status} {result}"
                )
                if accumulator.missing_indices:
                    missing_rows = ",".join(str(index + 1) for index in accumulator.missing_indices)
                    self._append_live_log(f"ROI{roi_id} MISSING_ROWS={missing_rows}")
            elif roi.expected.strip() and accumulator is not None and accumulator.expected_text:
                roi.passed = accumulator.cycle_complete
                result = "PASS" if roi.passed else "FAIL"
                score = f"{accumulator.coverage:.2f}"
                self._append_live_log(
                    f"ROI{roi_id} START={'Y' if accumulator.start_seen else 'N'} "
                    f"END={'Y' if accumulator.end_seen else 'N'} "
                    f"ORDER={'Y' if accumulator.order_valid else 'N'} "
                    f"COVERAGE={accumulator.coverage:.0%} "
                    f"LOOPS={accumulator.completed_wraps} {result}"
                )
            elif roi.expected.strip():
                cmp = compare_text(roi.expected, roi.actual, mode=mode, similarity_threshold=threshold)
                roi.passed = cmp.passed
                result = "PASS" if cmp.passed else "FAIL"
                score = f"{cmp.score:.2f}"
            else:
                roi.passed = True
                result = "N/A"
                score = "-"

            if not roi.passed and roi.expected.strip() and self.source_image is not None:
                try:
                    # Scroll results are merged from many frames, so no single
                    # recorded image is the input: save a reconstruction only.
                    diagnostic_dir = _save_failed_roi_diagnostic(
                        self.source_image,
                        roi,
                        preprocess=self._roi_preprocess_config(),
                        model_info=self._resolved_model_info(),
                    )
                    self._append_live_log(f"ROI{roi_id} FAILURE_SAVED={diagnostic_dir}")
                except OSError as exc:
                    self._append_live_log(f"ROI{roi_id} FAILURE_SAVE_ERROR={exc}")

            self.result_tree.insert(
                "",
                tk.END,
                values=(roi_id, roi.expected.replace("\n", " | "), roi.actual.replace("\n", " | "), score, result),
            )

        self._refresh_canvas()
        self.status_var.set("Scroll capture completed")


def run_gui(engine: OCREngine):
    if sys.platform == "win32":
        with mss.mss():
            pass
    root = tk.Tk()
    gui = OCRValidatorGUI(root, engine)
    root.after(200, gui._refresh_canvas)
    root.mainloop()
