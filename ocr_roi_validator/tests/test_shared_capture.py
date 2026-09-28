import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
from test_roi_presets import library, gui_stub
from ocr_roi_validator.roi_presets import automation_preset, with_shared_capture, save_library, load_library, validate_library
from ocr_roi_validator.automation_ocr import validate_cases
from ocr_roi_validator.automation import TestCase, VestaSession
from ocr_roi_validator.automation_workflow import AutomationDialog
from ocr_roi_validator.window_anchor import make_anchor, ClientWindow


def anchor():
    return make_anchor(ClientWindow(1, 'SDL', 'Vesta', (0, 0, 800, 600)), (0, 0, 800, 600))


class SharedCaptureTests(unittest.TestCase):
    def test_shared_reference_applies_to_existing_and_future_presets_without_mutation(self):
        original = library()
        shared = with_shared_capture(original, 'roi_courseop', anchor())
        shared['presets']['second'] = copy.deepcopy(shared['presets']['roi_courseop'])
        for name in shared['presets']:
            resolved = automation_preset(shared, name)
            self.assertEqual(resolved['capture_anchor'], anchor())
            self.assertNotIn('capture_anchor', shared['presets'][name])
        self.assertNotIn('shared_capture', original)
        validate_cases([TestCase('tc', 2, 'test', 'second', [])], shared, Path('.'))

    def test_different_sizes_share_one_reference_without_changing_roi_geometry(self):
        shared = with_shared_capture(library(), 'roi_courseop', anchor())
        original = copy.deepcopy(shared['presets']['roi_courseop'])
        for size in ([319, 240], [315, 242], [900, 600]):
            shared['presets']['roi_courseop']['image_size'] = size
            resolved = automation_preset(shared, 'roi_courseop')
            self.assertEqual(resolved['capture_anchor'], anchor())
            self.assertEqual(resolved['capture_size'], original['image_size'])
            self.assertEqual(resolved['rois'], original['rois'])
            validate_cases([TestCase('tc', 2, 'test', 'roi_courseop', [])], shared, Path('.'))
        self.assertNotIn('capture_size', shared['presets']['roi_courseop'])

    def test_actual_selected_size_is_recorded_and_frame_resized_per_preset(self):
        window = ClientWindow(1, 'SDL', 'Vesta', (0, 0, 800, 600))
        selection = (40, 50, 355, 292)  # 315 x 242, not the preset's 320 x 240
        selected_anchor = make_anchor(window, selection)
        data = with_shared_capture(library(), 'roi_courseop', selected_anchor, [315, 242])
        session = VestaSession(Path('lite'), Path('output'))
        session.process = MagicMock(pid=123)
        session.process.poll.return_value = None
        with patch('ocr_roi_validator.automation.client_windows', return_value=[window]), \
             patch('ocr_roi_validator.automation.require_unobstructed'), \
             patch('ocr_roi_validator.capture.grab_screen_rect_with', return_value=Image.new('RGB', (315, 242))) as grab:
            for size in ([320, 240], [319, 240], [321, 241], [340, 260]):
                data['presets']['roi_courseop']['image_size'] = size
                frame = session.frame(automation_preset(data, 'roi_courseop'), object())
                self.assertEqual(frame.size, tuple(size))
                self.assertEqual(grab.call_args.args[1], selection)
        self.assertEqual(data['shared_capture']['image_size'], [315, 242])

    def test_shared_capture_still_rejects_actual_window_layout_change(self):
        window = ClientWindow(1, 'SDL', 'Vesta', (0, 0, 800, 600))
        data = with_shared_capture(library(), 'roi_courseop', make_anchor(window, (40, 50, 355, 292)), [315, 242])
        session = VestaSession(Path('lite'), Path('output'))
        session.process = MagicMock(pid=123)
        session.process.poll.return_value = None
        stretched = ClientWindow(1, 'SDL', 'Vesta', (0, 0, 1600, 600))
        with patch('ocr_roi_validator.automation.client_windows', return_value=[stretched]):
            with self.assertRaisesRegex(ValueError, 'aspect ratio'):
                session.frame(automation_preset(data, 'roi_courseop'), object())

    def test_gui_calibration_saves_actual_selection_once_for_all_sizes(self):
        data = library()
        data['presets']['second'] = {
            'image_size': [321, 241], 'direction': 'horizontal', 'scrolling': True,
            'rois': [{'id': 1, 'rect': [5, 6, 300, 50], 'expected': 'text'}],
        }
        window = ClientWindow(1, 'SDL', 'Vesta', (0, 0, 800, 600))
        selection = (40, 50, 355, 292)
        dialog = AutomationDialog.__new__(AutomationDialog)
        dialog.session = MagicMock(pid=123)
        dialog.session.frame.return_value = Image.new('RGB', (800, 600))
        dialog.gui = MagicMock(preset_library=data)
        dialog.preset_name = MagicMock()
        dialog.preset_name.get.return_value = 'roi_courseop'
        dialog.window = MagicMock()
        dialog.show_preview = MagicMock()
        dialog.log_line = MagicMock()
        dialog.busy = MagicMock(return_value=False)
        with tempfile.TemporaryDirectory() as folder:
            dialog.gui.preset_path = Path(folder)/'presets.json'
            with patch('ocr_roi_validator.automation_workflow.client_windows', return_value=[window]), \
                 patch('ocr_roi_validator.automation_workflow.foreground'), \
                 patch('ocr_roi_validator.automation_workflow.messagebox.showinfo'), \
                 patch('ocr_roi_validator.automation_workflow.messagebox.askyesno', return_value=True), \
                 patch('mss.mss'), \
                 patch('ocr_roi_validator.gui.ScreenAreaSelector') as picker:
                picker.return_value.result_rect = selection
                dialog.calibrate()
            saved = load_library(dialog.gui.preset_path)
        self.assertEqual(saved['shared_capture']['image_size'], [315, 242])
        self.assertEqual(saved['shared_capture']['capture_anchor'], make_anchor(window, selection))
        self.assertEqual(automation_preset(saved, 'second')['capture_size'], [315, 242])
        self.assertEqual(automation_preset(saved, 'second')['image_size'], [321, 241])
        self.assertNotIn('shared_capture', data)
        dialog.session.frame.assert_called_once()
        self.assertEqual(dialog.session.frame.call_args.args[0]['capture_size'], [315, 242])
        dialog.show_preview.assert_called_once()

    def test_shared_priority_preserves_old_registration(self):
        data = library()
        old = {**anchor(), 'title': 'old'}
        data['presets']['roi_courseop']['capture_anchor'] = old
        shared = with_shared_capture(data, 'roi_courseop', anchor())
        self.assertEqual(automation_preset(shared, 'roi_courseop')['capture_anchor'], anchor())
        self.assertEqual(shared['presets']['roi_courseop']['capture_anchor'], old)

    def test_save_new_preset_and_json_round_trip_keep_shared_reference(self):
        gui = gui_stub()
        gui.preset_library = with_shared_capture(library(), 'roi_courseop', anchor())
        gui.preset_name_var.set('new')
        gui.preset_box = MagicMock()
        with tempfile.TemporaryDirectory() as folder:
            gui.preset_path = Path(folder)/'presets.json'
            gui.save_roi_preset()
            restored = load_library(gui.preset_path)
        self.assertEqual(restored['shared_capture'], gui.preset_library['shared_capture'])
        self.assertEqual(automation_preset(restored, 'new')['capture_anchor'], anchor())

    def test_import_old_library_keeps_shared_reference(self):
        gui = gui_stub()
        gui.preset_library = with_shared_capture(library(), 'roi_courseop', anchor())
        gui.preset_box = MagicMock()
        with tempfile.TemporaryDirectory() as folder:
            gui.preset_path = Path(folder)/'current.json'
            incoming = Path(folder)/'incoming.json'
            save_library(incoming, library())
            with patch('ocr_roi_validator.preset_workflow.filedialog.askopenfilename', return_value=str(incoming)), patch('ocr_roi_validator.preset_workflow.messagebox.askyesno', return_value=True):
                gui.import_roi_presets()
            self.assertIn('shared_capture', load_library(gui.preset_path))

    def test_invalid_shared_reference_rejected(self):
        for shared in (None, {}, {'image_size': [0, 600], 'capture_anchor': anchor()},
                       {'image_size': [800, 600], 'capture_anchor': {}}):
            data = library()
            data['shared_capture'] = shared
            with self.assertRaises(ValueError):
                validate_library(data)
