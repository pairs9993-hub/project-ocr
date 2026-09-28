import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from test_roi_presets import library, gui_stub
from ocr_roi_validator.roi_presets import automation_preset, with_shared_capture, save_library, load_library, validate_library
from ocr_roi_validator.automation_ocr import validate_cases
from ocr_roi_validator.automation import TestCase
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

    def test_different_sizes_require_individual_reference(self):
        shared = with_shared_capture(library(), 'roi_courseop', anchor())
        shared['presets']['roi_courseop']['image_size'] = [900, 600]
        self.assertNotIn('capture_anchor', automation_preset(shared, 'roi_courseop'))
        with self.assertRaises(ValueError):
            validate_cases([TestCase('tc', 2, 'test', 'roi_courseop', [])], shared, Path('.'))
        shared['presets']['roi_courseop']['capture_anchor'] = anchor()
        self.assertEqual(automation_preset(shared, 'roi_courseop')['capture_anchor'], anchor())

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
