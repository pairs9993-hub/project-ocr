import json
import unittest
from unittest.mock import MagicMock

from PIL import Image
from ocr_roi_validator.ocr_engine import OCREngine, OCRRunResult
from ocr_roi_validator.overlap_ocr import apply_overlap_retry
from ocr_roi_validator.gui import OCRValidatorGUI
from ocr_roi_validator.automation_ocr import FrozenValue
from test_superscript import box, run_result


class OverlapRetryTests(unittest.TestCase):
    def value(self, right_x=131):
        return run_result([box('OK para', (39,11,146,44)),
                           box('a introducir.', (right_x,13,272,37))])

    def test_overlap_requires_independent_ocr_and_keeps_original_boxes(self):
        result = self.value()
        boxes = list(result.boxes)
        recognize = MagicMock(return_value=OCRRunResult('OK para introducir.', .98, 0, []))
        apply_overlap_retry(Image.new('RGB', (319,160)), result, recognize)
        self.assertEqual(result.text, 'OK para introducir.')
        self.assertEqual(result.raw_text, 'OK para a introducir.')
        self.assertEqual(result.boxes, boxes)
        recognize.assert_called_once()
        self.assertEqual(result.overlap_evidence['status'], 'OVERLAP_DUPLICATION_CONFIRMED')
        json.dumps(result.overlap_evidence)

    def test_real_repeated_letter_unrelated_text_and_low_confidence_are_preserved(self):
        for text, score in [('OK para a introducir.', .99), ('OK para entrar.', .99),
                            ('OK para introducir.', .8), ('OK para introducir.', float('nan'))]:
            with self.subTest(text=text, score=score):
                result = self.value()
                apply_overlap_retry(Image.new('RGB', (319,160)), result,
                                    lambda _: OCRRunResult(text, score, 0, []))
                self.assertEqual(result.text, 'OK para a introducir.')

    def test_no_overlap_contained_boxes_and_different_baselines_do_not_retry(self):
        for right in [box('a introducir.', (150,13,272,37)),
                      box('a introducir.', (80,13,130,37)),
                      box('a introducir.', (131,50,272,74))]:
            result = run_result([box('OK para', (39,11,146,44)), right])
            recognize = MagicMock()
            apply_overlap_retry(Image.new('RGB', (319,160)), result, recognize)
            recognize.assert_not_called()

    def test_matching_boundary_without_image_overlap_does_not_delete(self):
        result = self.value(right_x=148)
        recognize = MagicMock(return_value=OCRRunResult('OK para introducir.', .99, 0, []))
        apply_overlap_retry(Image.new('RGB', (319,160)), result, recognize)
        self.assertEqual(result.text, 'OK para a introducir.')
        recognize.assert_not_called()

    def test_timeout_propagates_to_existing_watchdog(self):
        recognize = MagicMock(side_effect=TimeoutError('deadline'))
        with self.assertRaises(TimeoutError):
            apply_overlap_retry(Image.new('RGB', (319,160)), self.value(), recognize)

    def test_recognition_only_uses_no_detection(self):
        engine = OCREngine(use_rapid_default=True)
        backend = MagicMock(return_value=([['actual image text', .98]], [.1]))
        engine._engine_for_language = MagicMock(return_value=backend)
        result = engine.recognize_line(Image.new('RGB', (100,30)), 'en_es')
        self.assertEqual(result.text, 'actual image text')
        self.assertEqual(backend.call_args.kwargs, {'use_det': False, 'use_cls': False})

    def test_gui_direct_and_context_keep_raw_with_one_retry_per_request(self):
        for context in (False, True):
            with self.subTest(context=context):
                gui = OCRValidatorGUI.__new__(OCRValidatorGUI)
                gui.language_var = FrozenValue('en_es')
                gui.context_margin_var = FrozenValue('0')
                gui.engine = MagicMock()
                gui.engine.text_from_boxes = OCREngine.text_from_boxes
                gui.engine.run.return_value = self.value()
                gui.engine.recognize_line.return_value = OCRRunResult('OK para introducir.', .98, 0, [])
                image = Image.new('RGB',(319,160))
                result = (gui._run_context_roi_ocr(image, (0,0,319,160)) if context else
                          gui._run_engine(image, '', record_as='direct'))
                self.assertEqual(result.raw_text, 'OK para a introducir.')
                self.assertEqual(result.text, 'OK para introducir.')
                self.assertEqual(gui._last_ocr_input.raw_ocr_text, 'OK para a introducir.')
                self.assertEqual(gui._last_ocr_input.overlap_evidence['status'], 'OVERLAP_DUPLICATION_CONFIRMED')
                gui.engine.recognize_line.assert_called_once()

    def test_paddle_does_not_invoke_rapid_recognition_api(self):
        engine = OCREngine(use_rapid_default=True)
        engine.backend = 'paddle'
        engine._engine_for_language = MagicMock()
        self.assertIsNone(engine.recognize_line(Image.new('RGB',(100,30)), 'en_es'))
        engine._engine_for_language.assert_not_called()


if __name__ == '__main__':
    unittest.main()
