from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
from openpyxl import Workbook
from ocr_roi_validator.roi_presets import validate_library, save_library, load_library, resolve_preset, read_excel, excel_references
from ocr_roi_validator.gui import OCRValidatorGUI, ROIItem
from ocr_roi_validator.scroll_merge import ScrollTextAccumulator, VerticalListAccumulator


def library(direction="horizontal", scrolling=False):
    return {"version": 1, "presets": {"roi_courseop": {
        "image_size": [800, 600], "direction": direction, "scrolling": scrolling,
        "rois": [{"id": i, "rect": [10, i * 50, 300, i * 50 + 40], "expected": f"문구 {i}\n다음 줄"} for i in (1, 2, 3)],
    }}}


def gui_stub():
    gui = OCRValidatorGUI.__new__(OCRValidatorGUI)
    gui.source_image = Image.new("RGB", (800, 600))
    gui.preset_library = library()
    gui.live_thread = None
    gui._timed_capture_running = False
    gui.rois = {9: ROIItem(9, (0, 0, 1, 1))}
    gui.selected_roi_id = None
    gui.live_accumulators = {9: object()}
    gui.live_samplers = {9: object()}
    gui._ocr_inputs = {9: object()}
    gui._last_ocr_input = object()
    for attr in ("preset_name_var", "vertical_list_mode_var", "scroll_mode_var", "status_var"):
        var = MagicMock()
        state = [False]
        var.set.side_effect = lambda value, state=state: state.__setitem__(0, value)
        var.get.side_effect = lambda state=state: state[0]
        setattr(gui, attr, var)
    gui.compare_mode_var = MagicMock()
    gui.compare_mode_var.get.return_value = "exact"
    gui.similarity_threshold_var = MagicMock()
    gui.similarity_threshold_var.get.return_value = "0.9"
    for attr in ("expected_text", "roi_list", "result_tree", "_refresh_canvas", "_on_select_roi"):
        setattr(gui, attr, MagicMock())
    return gui


class LibraryTests(unittest.TestCase):
    def test_round_trip_multiple_rois_unicode_and_direction(self):
        for direction, scrolling in (("horizontal", False), ("horizontal", True), ("vertical", True)):
            with self.subTest(direction=direction, scrolling=scrolling), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "library.json"
                data = library(direction, scrolling)
                save_library(path, data)
                self.assertEqual(load_library(path), data)
                self.assertFalse(path.with_suffix('.json.tmp').exists())

    def test_invalid_schema_and_coordinates(self):
        for field, value in (("image_size", [0, 600]), ("direction", "diagonal"), ("scrolling", "yes"), ("rois", [])):
            with self.subTest(field=field):
                data = library()
                data['presets']['roi_courseop'][field] = value
                with self.assertRaises(ValueError):
                    validate_library(data)
        for rect in ([-1, 0, 4, 4], [0, 0, 900, 4], [4, 0, 4, 4], [0.1, 0, 4, 4]):
            data = library()
            data['presets']['roi_courseop']['rois'][0]['rect'] = rect
            with self.assertRaises(ValueError):
                validate_library(data)

    def test_duplicate_ids_and_unsupported_version(self):
        data = library()
        data['presets']['roi_courseop']['rois'][1]['id'] = 1
        with self.assertRaises(ValueError):
            validate_library(data)
        with self.assertRaises(ValueError):
            validate_library({'version': 2, 'presets': {}})

    def test_unknown_blank_and_size_mismatch(self):
        for name, size in (("", (800, 600)), ("missing", (800, 600)), ("roi_courseop", (400, 300))):
            with self.subTest(name=name, size=size), self.assertRaises(ValueError):
                resolve_preset(library(), name, size)
        self.assertEqual(len(resolve_preset(library(), " roi_courseop ", (800, 600))['rois']), 3)

    def test_excel_multiple_sheets_arbitrary_column_and_blank_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cases.xlsx'
            book = Workbook()
            sheet = book.active
            sheet.title = 'Cases'
            sheet.append(['ID', 'Custom layout', 'roi'])
            sheet.append([1, ' roi_courseop ', 'other'])
            sheet.append([2, None, 'other'])
            sheet.append([3, 'missing', 'other'])
            book.create_sheet('Empty')
            book.save(path)
            before = path.read_bytes()
            sheets = read_excel(path)
            self.assertEqual(excel_references(sheets['Cases'], 1), [(2, 'roi_courseop'), (3, ''), (4, 'missing')])
            self.assertEqual(sheets['Empty'], [])
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_write_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'library.json'
            save_library(path, library())
            with self.assertRaises(ValueError):
                save_library(path, {'version': 9})
            self.assertEqual(load_library(path), library())


class WorkflowTests(unittest.TestCase):
    def test_apply_replaces_rois_resets_results_and_restores_direction(self):
        gui = gui_stub()
        gui.preset_library = library('vertical', True)
        self.assertTrue(gui.apply_roi_preset('roi_courseop'))
        self.assertEqual(list(gui.rois), [1, 2, 3])
        self.assertEqual(gui.next_roi_id, 4)
        self.assertTrue(gui.vertical_list_mode_var.get())
        self.assertTrue(gui.scroll_mode_var.get())
        self.assertEqual(gui.live_accumulators, {})
        self.assertEqual(gui._ocr_inputs, {})
        self.assertIsNone(gui._last_ocr_input)
        self.assertIsInstance(gui._new_live_accumulator(gui.rois[1]), VerticalListAccumulator)

    def test_explicit_horizontal_does_not_become_vertical_for_multiline_text(self):
        gui = gui_stub()
        gui.apply_roi_preset('roi_courseop')
        gui.scroll_mode_var.set(True)
        self.assertIsInstance(gui._new_live_accumulator(gui.rois[1]), ScrollTextAccumulator)

    @patch('ocr_roi_validator.preset_workflow.messagebox.showerror')
    def test_invalid_reference_keeps_current_rois_and_never_runs(self, error):
        gui = gui_stub()
        gui.run_once = MagicMock()
        gui.run_timed_capture = MagicMock()
        self.assertFalse(gui._apply_excel_reference('missing', verify=True))
        self.assertEqual(list(gui.rois), [9])
        gui.run_once.assert_not_called()
        gui.run_timed_capture.assert_not_called()
        error.assert_called_once()

    @patch('ocr_roi_validator.preset_workflow.messagebox.showwarning')
    def test_cannot_apply_during_capture(self, warning):
        gui = gui_stub()
        gui._timed_capture_running = True
        self.assertFalse(gui.apply_roi_preset('roi_courseop'))
        self.assertEqual(list(gui.rois), [9])
        gui._timed_capture_running = False
        gui.live_thread = MagicMock()
        gui.live_thread.is_alive.return_value = True
        self.assertFalse(gui.apply_roi_preset('roi_courseop'))

    def test_excel_verify_dispatches_to_matching_mode(self):
        for direction, scrolling in (('horizontal', False), ('horizontal', True), ('vertical', True)):
            with self.subTest(direction=direction, scrolling=scrolling):
                gui = gui_stub()
                gui.preset_library = library(direction, scrolling)
                gui.run_once = MagicMock()
                gui.run_timed_capture = MagicMock()
                self.assertTrue(gui._apply_excel_reference('roi_courseop', verify=True))
                self.assertEqual(gui.run_timed_capture.call_count, int(scrolling))
                self.assertEqual(gui.run_once.call_count, int(not scrolling))

    def test_save_includes_pending_expected_edits_and_persists(self):
        gui = gui_stub()
        gui.preset_name_var.set('new_preset')
        gui.selected_roi_id = 9
        gui.expected_text.get.return_value = 'updated expected'
        gui._refresh_preset_names = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            gui.preset_path = Path(directory) / 'presets.json'
            gui.save_roi_preset()
            saved = load_library(gui.preset_path)
            self.assertEqual(saved['presets']['new_preset']['rois'][0]['expected'], 'updated expected')
            self.assertIn('roi_courseop', saved['presets'])

    @patch('ocr_roi_validator.preset_workflow.messagebox.showerror')
    @patch('ocr_roi_validator.preset_workflow.filedialog.askopenfilename')
    def test_bad_import_leaves_library_and_disk_intact(self, choose, error):
        gui = gui_stub()
        original = gui.preset_library
        with tempfile.TemporaryDirectory() as directory:
            gui.preset_path = Path(directory) / 'presets.json'
            save_library(gui.preset_path, original)
            bad = Path(directory) / 'bad.json'
            bad.write_text('{"version": 99}', encoding='utf-8')
            choose.return_value = str(bad)
            gui.import_roi_presets()
            self.assertIs(gui.preset_library, original)
            self.assertEqual(load_library(gui.preset_path), original)
        error.assert_called_once()

    @patch('ocr_roi_validator.gui.grab_screen_rect_with')
    @patch('ocr_roi_validator.gui.mss.mss')
    def test_run_once_recaptures_current_screen(self, capture_session, grab):
        gui = gui_stub()
        gui.screen_base_rect = (50, 60, 850, 660)
        gui._evaluate_rois = MagicMock()
        latest = Image.new('RGB', (800, 600), 'red')
        grab.return_value = latest
        gui.run_once()
        self.assertIs(gui.source_image, latest)
        gui._evaluate_rois.assert_called_once_with(latest)

    def test_selection_only_does_not_start_ocr(self):
        gui = gui_stub()
        gui.run_once = MagicMock()
        gui.run_timed_capture = MagicMock()
        self.assertTrue(gui._apply_excel_reference('roi_courseop'))
        gui.run_once.assert_not_called()
        gui.run_timed_capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
