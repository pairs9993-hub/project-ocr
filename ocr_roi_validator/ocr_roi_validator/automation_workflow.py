"""Modal Tk workflow; background work communicates solely via a polled queue."""
from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import zipfile

from .automation import Cancelled, VestaSession, check_cancel, load_cases, number, product_config_names
from .automation_ocr import prepare_rois, snapshot_ocr, validate_cases, verify_case
from .automation_report import initial_result, write_excel_report
from .isolated_ocr import IsolatedOCR
from .roi_presets import save_library
from .window_anchor import client_windows, foreground, make_anchor, mapped_rect, selection_window


class AutomationWorkflow:
    def open_automation(self):
        if not self._roi_changes_allowed():
            return
        if getattr(self, "_automation_dialog", None) is not None:
            self._automation_dialog.window.lift()
            return
        self._automation_dialog = AutomationDialog(self)


class AutomationDialog:
    def __init__(self, gui):
        self.gui = gui
        self.session = None
        self.worker = None
        self.stop = threading.Event()
        self.events = queue.Queue()
        self.closing = False
        self.close_root = False
        self.calibrating = False
        self.results = []
        self.output = None
        self.window = tk.Toplevel(gui.root)
        self.window.title("자동화 검증 — Vesta + Excel TC")
        self.window.geometry("1000x720")
        self.window.transient(gui.root)
        self.window.grab_set()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        gui.root.protocol("WM_DELETE_WINDOW", self._close_application)
        self.variables = {}
        # Reserve actions before expandable content so small screens keep them visible.
        buttons = ttk.Frame(self.window, padding=10)
        buttons.pack(fill=tk.X)
        self.launch_button = ttk.Button(buttons, text="① Vesta 실행", command=self.launch)
        self.launch_button.pack(side=tk.LEFT, padx=4)
        self.run_button = ttk.Button(buttons, text="③ 자동화 검증 시작", command=self.run)
        self.run_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="중지 / Vesta 종료", command=self.stop_run).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="닫기", command=self.close).pack(side=tk.RIGHT, padx=4)

        settings = ttk.Frame(self.window, padding=12)
        settings.pack(fill=tk.X)
        settings.columnconfigure(1, weight=1)
        fields = [("zip", "Vesta ZIP", ""), ("excel", "TC Excel", ""),
                  ("references", "정답 이미지 폴더 (선택)", ""),
                  ("product", "Product (sim_config 이름)", ""),
                  ("roi_column", "ROI 이름 열", "ROI"),
                  ("lite", "LITE 폴더", str(Path(__file__).resolve().parents[2] / "lite")),
                  ("duration", "최대 ROI 관찰 시간 (≤30초)", "30"),
                  ("fps", "스크롤 FPS", gui.scroll_fps_var.get())]
        for row, (key, label, default) in enumerate(fields):
            var = tk.StringVar(value=default)
            self.variables[key] = var
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
            entry = ttk.Combobox(settings, textvariable=var) if key == "product" else ttk.Entry(settings, textvariable=var)
            entry.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
            if key == "product":
                self.product_box = entry
            if key in ("zip", "excel", "references", "lite"):
                ttk.Button(settings, text="찾기…", command=lambda k=key: self.browse(k)).grid(row=row, column=2)
        self.settings = settings
        ttk.Label(self.window, text="기존 프리셋: ① Vesta 실행 → ② 프리셋 선택 / 화면 기준 등록 (최초 1회) → ③ 검증 시작\n"
                  "정답: Expected_Text 셀의 [ROI1] 블록 / Expected_1, Expected_2… 열 / Image. 알집 ZIP 지원 (.alz 제외).\n"
                  "다른 창으로 Vesta를 가리지 마세요. 창 이동/DPI 비례 확대는 지원, GUI 레이아웃 변경은 재등록이 필요합니다.",
                  wraplength=950).pack(anchor="w", padx=16, pady=4)
        calibration = ttk.Frame(self.window, padding=8)
        calibration.pack(fill=tk.X)
        self.preset_name = tk.StringVar(value=gui.preset_name_var.get())
        names = sorted(gui.preset_library["presets"])
        if self.preset_name.get() not in names and names:
            self.preset_name.set(names[0])
        ttk.Label(calibration, text="기준 등록 프리셋").pack(side=tk.LEFT, padx=4)
        self.preset_box = ttk.Combobox(calibration, textvariable=self.preset_name, values=names, state="readonly", width=24)
        self.preset_box.pack(side=tk.LEFT, padx=4)
        self.calibrate_button = ttk.Button(calibration, text="화면 기준 등록", command=self.calibrate)
        self.calibrate_button.pack(side=tk.LEFT, padx=4)
        self.state = tk.StringVar(value="Ready — ZIP/TC를 첨부하세요. 첨부한 Vesta 실행 파일은 신뢰할 수 있는 파일이어야 합니다.")
        ttk.Label(self.window, textvariable=self.state, wraplength=950).pack(fill=tk.X, padx=16, pady=8)
        columns = ("sheet", "row", "title", "preset", "status")
        self.tree = ttk.Treeview(self.window, columns=columns, show="headings", height=10)
        for col in columns:
            self.tree.heading(col, text=col.upper())
            self.tree.column(col, width=170 if col != "row" else 60)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=12)
        self.log = tk.Text(self.window, height=6, state="disabled", wrap="word")
        self.log.pack(fill=tk.X, padx=12, pady=8)
        self.window.after(100, self.poll)

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def browse(self, key):
        if self.busy():
            return
        if key in ("references", "lite"):
            selected = filedialog.askdirectory(parent=self.window)
        else:
            selected = filedialog.askopenfilename(parent=self.window, filetypes=[
                ("Vesta ZIP", "*.zip") if key == "zip" else ("Excel TC", "*.xlsx *.xlsm")])
        if selected:
            self.variables[key].set(selected)
            if key == "zip":
                try:
                    with zipfile.ZipFile(selected) as archive:
                        products = product_config_names(archive.namelist())
                    self.product_box["values"] = products
                    if len(products) == 1:
                        self.variables["product"].set(products[0])
                except (OSError, zipfile.BadZipFile) as exc:
                    messagebox.showerror("ZIP", str(exc), parent=self.window)

    def config(self):
        values = {k: v.get().strip() for k, v in self.variables.items()}
        if not Path(values["zip"]).is_file() or Path(values["zip"]).suffix.lower() != ".zip":
            raise ValueError("Vesta ZIP 파일을 선택하세요.")
        values["duration"] = number(values["duration"], 5, "Duration", positive=True)
        if values["duration"] > 30:
            raise ValueError("최대 ROI 관찰 시간은 30초 이하여야 합니다.")
        values["fps"] = number(values["fps"], 8, "FPS", positive=True)
        if not values["roi_column"]:
            raise ValueError("ROI 이름 열을 입력하세요.")
        return values

    def log_line(self, line):
        self.log.configure(state="normal")
        self.log.insert(tk.END, line+"\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def set_busy(self, busy):
        self.gui._automation_running = busy or self.session is not None
        for button in (self.launch_button, self.run_button, self.calibrate_button):
            button.configure(state="disabled" if busy else "normal")
        for child in self.settings.winfo_children():
            if isinstance(child, (ttk.Entry, ttk.Button)):
                child.configure(state="disabled" if busy or self.session is not None else "normal")

    def start_worker(self, task):
        if self.busy():
            return
        self.stop.clear()
        self.set_busy(True)

        def work():
            try:
                task()
            except Cancelled:
                self.events.put(("log", "중지되었습니다. 완료된 결과는 보존됩니다."))
            except Exception as exc:
                self.events.put(("error", str(exc)))
            finally:
                self.events.put(("idle", None))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def ensure_session(self, values):
        if self.session is not None:
            self.session.pid
            return
        self.output = Path(__file__).resolve().parents[1] / "captures" / "automation" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output.mkdir(parents=True, exist_ok=False)
        session = VestaSession(Path(values["lite"]), self.output)
        try:
            self.events.put(("log", "ZIP 압축 해제 및 Vesta 시작 중…"))
            session.launch(Path(values["zip"]), values["product"], self.stop)
        except BaseException:
            session.close()
            raise
        self.session = session
        self.events.put(("log", f"Vesta 준비됨. 결과 폴더: {self.output}"))

    def launch(self):
        if self.busy() or self.session is not None:
            return
        try:
            values = self.config()
        except Exception as exc:
            messagebox.showerror("자동화 검증", str(exc), parent=self.window)
            return
        self.start_worker(lambda: self.ensure_session(values))

    def calibrate(self):
        if self.busy():
            return
        self.calibrating = True
        try:
            if self.session is None:
                raise ValueError("먼저 ① Vesta 실행을 누르세요.")
            name = self.preset_name.get()
            preset = self.gui.preset_library["presets"][name]
            windows = client_windows(self.session.pid)
            if not windows:
                raise ValueError("실행한 Vesta 창을 찾을 수 없습니다.")
            foreground(windows[0])
            messagebox.showinfo("화면 기준 등록", "다음 미리보기 창에서 기존 프리셋의 GUI 화면 전체를 같은 경계로 드래그하세요.\n"
                                "ROI 자체가 아니라 ROI의 기준 화면입니다. 제목 표시줄/도구 모음은 제외하세요.", parent=self.window)
            self.window.withdraw()
            self.gui.root.withdraw()
            self.window.update_idletasks()
            from .gui import ScreenAreaSelector
            # Show only the launched Vesta's visible client bounds in a preview.
            windows = client_windows(self.session.pid)
            if not windows:
                raise ValueError("실행한 Vesta 창을 찾을 수 없습니다.")
            bounds = (min(w.rect[0] for w in windows), min(w.rect[1] for w in windows),
                      max(w.rect[2] for w in windows), max(w.rect[3] for w in windows))
            selector = ScreenAreaSelector(self.gui.root, capture_rect=bounds)
            selector.grab_set()
            self.gui.root.wait_window(selector)
            if selector.result_rect is None:
                return
            target = selection_window(client_windows(self.session.pid), selector.result_rect)
            anchor = make_anchor(target, selector.result_rect)
            mapped_rect(anchor, target.rect, preset["image_size"])
            # Preview the exact mapped ROI geometry before committing calibration.
            preview_preset = {**preset, "capture_anchor": anchor}
            import mss
            with mss.mss() as capture:
                preview = self.session.frame(preview_preset, capture)
            self.show_preview(name, preview, preset)
            self.window.deiconify()
            if not messagebox.askyesno("화면 기준 저장", "뒤의 메인 창에서 ROI 위치를 확인하세요. 이 기준을 프리셋에 저장할까요?", parent=self.window):
                return
            updated = copy.deepcopy(self.gui.preset_library)
            updated["presets"][name]["capture_anchor"] = anchor
            save_library(self.gui.preset_path, updated)
            self.gui.preset_library = updated
            self.log_line(f"{name}: 창 내부 상대 좌표 저장 완료. JSON 내보내기로 다른 PC에서도 재사용할 수 있습니다.")
        except Exception as exc:
            messagebox.showerror("화면 기준 등록", str(exc), parent=self.window)
        finally:
            self.calibrating = False
            self.gui.root.deiconify()
            self.window.deiconify()
            self.window.grab_set()

    def show_preview(self, name, image, preset, results=None):
        from .gui import ROIItem
        self.gui.root.deiconify()
        self.gui.source_image = image
        self.gui.screen_base_rect = None
        self.gui._preset_direction = preset["direction"]
        self.gui.vertical_list_mode_var.set(preset["direction"] == "vertical")
        self.gui.scroll_mode_var.set(preset["scrolling"])
        self.gui.rois = {r["id"]: ROIItem(r["id"], tuple(r["rect"]), r.get("expected", "")) for r in preset["rois"]}
        self.gui.next_roi_id = max(self.gui.rois)+1
        self.gui.selected_roi_id = None
        self.gui.expected_text.delete("1.0", tk.END)
        self.gui.live_accumulators.clear()
        self.gui.live_samplers.clear()
        self.gui._ocr_inputs.clear()
        self.gui._last_ocr_input = None
        self.gui.result_tree.delete(*self.gui.result_tree.get_children())
        if results:
            for row in results:
                roi = self.gui.rois[row["roi_id"]]
                roi.expected, roi.actual, roi.passed = row["expected"], row["actual"], row["passed"]
                self.gui.result_tree.insert("", tk.END, values=(roi.roi_id, roi.expected, roi.actual,
                    f"{row['score']:.2f}", row.get("status", "PASS" if roi.passed else "FAIL")))
        self.gui.preset_name_var.set(name)
        self.gui._sync_roi_list()
        self.gui._refresh_canvas()

    def run(self):
        if self.busy():
            return
        try:
            values = self.config()
            cases, tables = load_cases(Path(values["excel"]), values["roi_column"])
            references = Path(values["references"]) if values["references"] else Path(values["excel"]).resolve().parent
            library = copy.deepcopy(self.gui.preset_library)
            validate_cases(cases, library, references)
            processor = snapshot_ocr(self.gui)
            threshold = number(processor.similarity_threshold_var.get(), 0.9, "Similarity")
            if threshold > 1:
                raise ValueError("Similarity must be between 0 and 1.")
        except Exception as exc:
            messagebox.showerror("TC 사전 검사", str(exc), parent=self.window)
            return
        self.results = []
        self.tree.delete(*self.tree.get_children())
        for i, case in enumerate(cases):
            self.tree.insert("", tk.END, iid=str(i), values=(case.sheet, case.row, case.title, case.preset, "PENDING"))

        def execute():
            report = {"tc": values["excel"], "zip": values["zip"], "state": "RUNNING",
                      "started_at": datetime.now().isoformat(), "presets": library,
                      "results": [initial_result(case, library["presets"][case.preset]) for case in cases], "settings": {
                          "duration": values["duration"], "fps": values["fps"],
                          "language": processor.language_var.get(), "compare": processor.compare_mode_var.get()}}
            report_dir = self.output or (Path(__file__).resolve().parents[1] / "captures" / "automation" /
                                         datetime.now().strftime("report_%Y%m%d_%H%M%S_%f"))
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"results_{datetime.now():%H%M%S_%f}.json"
            last_preview = None
            processor._isolated_ocr = IsolatedOCR(processor)

            def save_report():
                if report_path is not None:
                    temporary = report_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    temporary.replace(report_path)

            try:
                save_report()
                self.ensure_session(values)
                resolved_cases = copy.deepcopy(cases)
                for i, case in enumerate(resolved_cases):
                    check_cancel(self.stop)
                    self.events.put(("row", (i, "RUNNING")))
                    preset = library["presets"][case.preset]
                    entry = report["results"][i]
                    entry["status"] = "RUNNING"
                    try:
                        prepared = prepare_rois(processor, case, preset, references, stop=self.stop)
                        case.expected = {roi_id: roi.expected for roi_id, roi in prepared.items()}
                        for roi in entry["rois"]:
                            roi["expected"] = case.expected[roi["roi_id"]]
                        self.session.focus(preset)
                        self.session.execute(case, tables, self.stop)
                        row_output = report_dir / report_path.stem / f"row_{i+1:04d}"
                        result, frame = verify_case(processor, self.session, case, preset, references,
                            values["duration"], values["fps"], self.stop, row_output)
                        # Do not raise/redraw the main window over Vesta mid-run.
                        if frame is not None:
                            last_preview = (case.preset, frame, preset, result["rois"])
                    except Cancelled as exc:
                        result = {"status": "CANCELLED", "reason": str(exc) or "USER_STOP"}
                    except Exception as exc:
                        result = {"status": "ERROR", "error": str(exc), "reason": str(exc)}
                    if "rois" not in result:
                        for roi in entry["rois"]:
                            roi.update(status=result["status"], reason=result.get("reason", ""))
                    entry.update(result)
                    save_report()
                    self.events.put(("row", (i, result["status"])))
                    if result["status"] == "CANCELLED":
                        raise Cancelled("시험 중단: 결과를 Excel로 저장합니다.")
                    # TC commands independently establish their own starting state.
                    # Timeouts and execution errors are recorded; the next TC still runs.
                report["state"] = "COMPLETED"
                self.events.put(("log", f"검증 완료: {report_path}"))
            except Cancelled:
                report["state"] = "CANCELLED"
                raise
            except Exception as exc:
                report["state"] = "ERROR"
                report["error"] = str(exc)
                raise
            finally:
                try:
                    report["ended_at"] = datetime.now().isoformat()
                    save_report()
                    try:
                        excel_path = report_path.with_suffix(".xlsx")
                        write_excel_report(excel_path, report)
                    except PermissionError:
                        # A previous checkpoint may be open in Excel; never discard final results.
                        excel_path = report_path.with_name(report_path.stem+f"_final_{datetime.now():%H%M%S_%f}.xlsx")
                        write_excel_report(excel_path, report)
                    self.events.put(("log", f"Excel 결과 자동 저장: {excel_path}"))
                finally:
                    processor._isolated_ocr.close()
                    if self.session:
                        self.session.close()
                        self.session = None
                    if last_preview is not None:
                        self.events.put(("preview", last_preview))
        self.start_worker(execute)

    def stop_run(self):
        self.stop.set()
        self.state.set("중지 요청됨 — OCR 프로세스 종료 및 Excel 결과 저장 중…")
        if not self.busy() and self.session:
            session = self.session
            self.session = None
            self.start_worker(session.close)

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind in ("log", "error"):
                    self.state.set(value)
                    self.log_line(value)
                elif kind == "row":
                    index, status = value
                    self.tree.set(str(index), "status", status)
                    self.state.set(f"TC {index+1}: {status}")
                elif kind == "preview":
                    self.show_preview(*value)
                elif kind == "idle":
                    # The producer may still be returning from its finally block.
                    pass
        except queue.Empty:
            pass
        if not self.busy() and not self.calibrating:
            self.set_busy(False)
            if self.closing:
                if self.session:
                    self.stop_run()
                else:
                    self.window.grab_release()
                    self.window.destroy()
                    self.gui._automation_dialog = None
                    self.gui._automation_running = False
                    self.gui.root.protocol("WM_DELETE_WINDOW", self.gui.root.destroy)
                    if self.close_root:
                        self.gui.root.destroy()
                    return
        self.window.after(100, self.poll)

    def _close_application(self):
        self.close_root = True
        self.close()

    def close(self):
        self.closing = True
        self.stop_run()