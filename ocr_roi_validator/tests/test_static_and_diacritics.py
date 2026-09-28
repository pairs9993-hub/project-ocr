import unittest
from unittest.mock import MagicMock
from PIL import Image
from ocr_roi_validator.static_text import StaticTextAccumulator
from ocr_roi_validator.image_spacing import apply_image_spacing
from ocr_roi_validator.ocr_engine import OCRRunResult, OCRBox
from ocr_roi_validator.gui import OCRValidatorGUI, ROIItem

RAW = "Temp. Agua fria"
EXPECTED = "Temp. Agua fr\u00eda"

def result(text, score=.95):
    return OCRRunResult(text, score, 1, [OCRBox(0, 0, 160, 30, 80, 15, text, score)])

class StaticAndDiacriticsTests(unittest.TestCase):
    def test_static_mismatch_visible_and_match_requires_repeated_observation(self):
        acc = StaticTextAccumulator(EXPECTED)
        acc.add(RAW, .95, 0)
        self.assertEqual(acc.final_text, RAW)
        self.assertFalse(acc.passed)
        acc.add(EXPECTED, .95, 1)
        self.assertFalse(acc.ready_to_stop(2))
        acc.add(EXPECTED, .95, 2)
        self.assertTrue(acc.ready_to_stop(2))
        acc.add('', 0, 3)
        self.assertFalse(acc.ready_to_stop(4))
        self.assertEqual(acc.final_text, '')

    def test_live_static_routing_and_rendering(self):
        gui = OCRValidatorGUI.__new__(OCRValidatorGUI)
        gui.scroll_mode_var = MagicMock(get=lambda: False)
        gui.vertical_list_mode_var = MagicMock(get=lambda: False)
        gui.compare_mode_var = MagicMock(get=lambda: 'exact')
        gui.similarity_threshold_var = MagicMock(get=lambda: '.9')
        roi = ROIItem(1, (0, 0, 160, 30), EXPECTED)
        acc = gui._new_live_accumulator(roi)
        self.assertIsInstance(acc, StaticTextAccumulator)
        acc.add(RAW, .95, 0)
        gui.rois, gui.live_accumulators = {1: roi}, {1: acc}
        mapped = gui._build_result_map({1: RAW}, use_accumulator_results=True)
        self.assertEqual(roi.actual, RAW)
        self.assertEqual(mapped[1][2], 'FAIL')

    def test_agreeing_scaled_ocr_adopted_and_raw_retained(self):
        original = result(RAW)
        recognize = MagicMock(side_effect=[result(EXPECTED), result(EXPECTED)])
        apply_image_spacing(Image.new('RGB', (160, 30)), original, EXPECTED, recognize)
        self.assertEqual(original.text, EXPECTED)
        self.assertEqual(original.raw_text, RAW)
        self.assertEqual([call.args[0].size for call in recognize.call_args_list], [(344, 84), (504, 114)])

    def test_no_truth_substitution_on_failed_or_disagreeing_retries(self):
        for texts in [(RAW, RAW), (EXPECTED, RAW), ('Wrong text', EXPECTED)]:
            original = result(RAW)
            recognize = MagicMock(side_effect=[result(t) for t in texts])
            apply_image_spacing(Image.new('RGB', (160, 30)), original, EXPECTED, recognize)
            self.assertEqual(original.text, RAW)

    def test_low_confidence_accent_not_adopted(self):
        original = result(RAW)
        apply_image_spacing(Image.new('RGB', (160, 30)), original, EXPECTED,
                            MagicMock(return_value=result(EXPECTED, .2)))
        self.assertEqual(original.text, RAW)

    def test_agreeing_wrong_accent_remains_wrong(self):
        original = result(RAW)
        wrong = 'Temp. Agua fr\u00eca'
        apply_image_spacing(Image.new('RGB', (160, 30)), original, EXPECTED,
                            MagicMock(return_value=result(wrong)))
        self.assertEqual(original.text, wrong)
        self.assertNotEqual(original.text, EXPECTED)

    def test_retry_diagnostic_marks_multiple_inputs(self):
        gui = OCRValidatorGUI.__new__(OCRValidatorGUI)
        gui.language_var = MagicMock(get=lambda: 'en_es')
        gui.engine = MagicMock()
        gui.engine.run.side_effect = [result(RAW), result(EXPECTED), result(EXPECTED)]
        value = gui._run_engine(Image.new('RGB', (160, 30)), EXPECTED, record_as='direct')
        self.assertEqual(value.text, EXPECTED)
        self.assertFalse(gui._last_ocr_input.exact)
        self.assertEqual(gui._last_ocr_input.raw_ocr_text, RAW)
        self.assertEqual(gui._last_ocr_input.path_kind, 'diacritic_scaled_retry')
