"""Tk controls for named ROI libraries and Excel-driven preset selection."""
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from .roi_presets import load_library, save_library, resolve_preset, read_excel, excel_references


class ROIPresetWorkflow:
    def _init_presets(self):
        self.preset_path = Path(__file__).resolve().parents[1] / "roi_presets.json"
        self.preset_library = {"version": 1, "presets": {}}
        self._timed_capture_running = False
        self._preset_direction = None
        if self.preset_path.exists():
            try:
                self.preset_library = load_library(self.preset_path)
            except (OSError, ValueError) as exc:
                messagebox.showerror("ROI library", str(exc))
        self.preset_name_var = tk.StringVar()

    def _build_preset_toolbar(self, parent):
        bar = ttk.Frame(parent, padding=4)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="ROI preset").pack(side=tk.LEFT, padx=4)
        self.preset_box = ttk.Combobox(bar, textvariable=self.preset_name_var, width=24)
        self.preset_box.pack(side=tk.LEFT)
        for label, command in (("Save Preset", self.save_roi_preset), ("Apply Preset", self.apply_roi_preset),
                               ("Import JSON", self.import_roi_presets), ("Export JSON", self.export_roi_presets),
                               ("Load Excel", self.load_roi_excel)):
            ttk.Button(bar, text=label, command=command).pack(side=tk.LEFT, padx=4)
        self._refresh_preset_names()

    def _refresh_preset_names(self):
        self.preset_box["values"] = sorted(self.preset_library["presets"])

    def _roi_changes_allowed(self):
        if self._timed_capture_running or (self.live_thread and self.live_thread.is_alive()):
            messagebox.showwarning("ROI", "Stop Live and wait for timed capture to finish before changing ROIs.")
            return False
        return True

    def save_roi_preset(self):
        if not self._roi_changes_allowed():
            return
        name = self.preset_name_var.get().strip()
        if not name or self.source_image is None or not self.rois:
            messagebox.showwarning("ROI preset", "Enter a preset name and draw at least one ROI first.")
            return
        if name in self.preset_library["presets"] and not messagebox.askyesno("ROI preset", f"Replace preset '{name}'?"):
            return
        # Include edits still present in the selected ROI's expected-text editor.
        if self.selected_roi_id in self.rois:
            self.rois[self.selected_roi_id].expected = self.expected_text.get("1.0", "end-1c")
        preset = {
            "image_size": list(self.source_image.size),
            "direction": "vertical" if self.vertical_list_mode_var.get() else "horizontal",
            "scrolling": self._scrolling_enabled(),
            "rois": [{"id": r.roi_id, "rect": list(r.rect), "expected": r.expected}
                     for r in sorted(self.rois.values(), key=lambda r: r.roi_id)],
        }
        updated = {"version": 1, "presets": {**self.preset_library["presets"], name: preset}}
        try:
            save_library(self.preset_path, updated)
        except (OSError, ValueError) as exc:
            messagebox.showerror("ROI preset", str(exc))
            return
        self.preset_library = updated
        self._refresh_preset_names()
        self.status_var.set(f"Saved ROI preset: {name}")

    def apply_roi_preset(self, name=None):
        if not self._roi_changes_allowed():
            return False
        if self.source_image is None:
            messagebox.showwarning("ROI preset", "Load an image or capture a screen area first.")
            return False
        name = (name if name is not None else self.preset_name_var.get()).strip()
        try:
            preset = resolve_preset(self.preset_library, name, self.source_image.size)
        except ValueError as exc:
            messagebox.showerror("ROI preset", str(exc))
            return False
        from .gui import ROIItem
        self.clear_rois()
        self.rois = {r["id"]: ROIItem(r["id"], tuple(r["rect"]), r.get("expected", "")) for r in preset["rois"]}
        self.next_roi_id = max(self.rois) + 1
        self._preset_direction = preset["direction"]
        self.vertical_list_mode_var.set(preset["direction"] == "vertical")
        self.scroll_mode_var.set(preset["scrolling"])
        self.live_accumulators.clear()
        self.live_samplers.clear()
        self._ocr_inputs.clear()
        self._last_ocr_input = None
        self._sync_roi_list()
        self._refresh_canvas()
        self.preset_name_var.set(name)
        self.roi_list.selection_set(0)
        self._on_select_roi(None)
        self.status_var.set(f"Applied {name}: {len(self.rois)} ROIs, {preset['direction']}")
        return True

    def import_roi_presets(self):
        path = filedialog.askopenfilename(title="Import ROI library", filetypes=[("ROI JSON", "*.json")])
        if not path:
            return
        try:
            incoming = load_library(Path(path))
            duplicates = sorted(self.preset_library["presets"].keys() & incoming["presets"].keys())
            if duplicates and not messagebox.askyesno("Import ROI library", "Replace existing presets?\n" + ", ".join(duplicates)):
                return
            updated = {"version": 1, "presets": {**self.preset_library["presets"], **incoming["presets"]}}
            save_library(self.preset_path, updated)
            self.preset_library = updated
            self._refresh_preset_names()
            self.status_var.set(f"Imported {len(incoming['presets'])} ROI presets")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Import ROI library", str(exc))

    def export_roi_presets(self):
        path = filedialog.asksaveasfilename(title="Export all ROI presets", defaultextension=".json", initialfile="roi_presets.json", filetypes=[("ROI JSON", "*.json")])
        if path:
            try:
                save_library(Path(path), self.preset_library)
                self.status_var.set(f"Exported ROI library: {path}")
            except (OSError, ValueError) as exc:
                messagebox.showerror("Export ROI library", str(exc))

    def load_roi_excel(self):
        path = filedialog.askopenfilename(title="Select Excel ROI references", filetypes=[("Excel", "*.xlsx *.xlsm")])
        if not path:
            return
        try:
            sheets = read_excel(Path(path))
        except ImportError:
            messagebox.showerror("Excel", "Install Excel support: python -m pip install openpyxl>=3.1")
            return
        except Exception as exc:
            messagebox.showerror("Excel", str(exc))
            return
        if not sheets:
            messagebox.showerror("Excel", "No worksheets found.")
            return
        window = tk.Toplevel(self.root)
        window.title(f"Excel ROI presets - {Path(path).name}")
        window.geometry("760x440")
        bar = ttk.Frame(window, padding=8)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="Sheet").pack(side=tk.LEFT)
        sheet_var = tk.StringVar(value=next(iter(sheets)))
        sheet_box = ttk.Combobox(bar, textvariable=sheet_var, values=list(sheets), state="readonly", width=24)
        sheet_box.pack(side=tk.LEFT, padx=6)
        ttk.Label(bar, text="ROI name column").pack(side=tk.LEFT)
        column_box = ttk.Combobox(bar, state="readonly", width=30)
        column_box.pack(side=tk.LEFT, padx=6)
        ttk.Label(window, text="Select a row to apply its preset. First Excel row must contain headers.").pack(anchor=tk.W, padx=8)
        frame = ttk.Frame(window)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        tree = ttk.Treeview(frame, columns=("row", "preset", "state"), show="headings", selectmode="browse")
        for col, label in (("row", "Excel row"), ("preset", "ROI preset"), ("state", "Status")):
            tree.heading(col, text=label)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        names = {}

        def populate(_event=None):
            tree.delete(*tree.get_children())
            names.clear()
            if column_box.current() < 0:
                return
            for row_number, name in excel_references(sheets[sheet_var.get()], column_box.current()):
                state = "Ready" if name in self.preset_library["presets"] else "Blank" if not name else "Unknown preset"
                item = tree.insert("", tk.END, values=(row_number, name, state))
                names[item] = name

        def choose_sheet(_event=None):
            rows = sheets[sheet_var.get()]
            headers = rows[0] if rows else ()
            column_box["values"] = [f"{i + 1}: {header or '(blank)'}" for i, header in enumerate(headers)]
            if headers:
                default = next((i for i, h in enumerate(headers) if str(h).strip().lower() == "roi"), 0)
                column_box.current(default)
            else:
                column_box.set("")
            populate()

        def apply_selected(_event=None, verify=False):
            selection = tree.selection()
            if not selection or selection[0] not in names:
                return
            self._apply_excel_reference(names[selection[0]], verify=verify)

        tree.bind("<<TreeviewSelect>>", apply_selected)
        sheet_box.bind("<<ComboboxSelected>>", choose_sheet)
        column_box.bind("<<ComboboxSelected>>", populate)
        ttk.Button(window, text="Apply + Verify Selected Row", command=lambda: apply_selected(verify=True)).pack(anchor=tk.E, padx=8, pady=8)
        choose_sheet()

    def _apply_excel_reference(self, name, verify=False):
        if not self.apply_roi_preset(name):
            return False
        if verify:
            if self._scrolling_enabled():
                self.run_timed_capture()
            else:
                self.run_once()
        return True
