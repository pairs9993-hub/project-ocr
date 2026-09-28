import unittest
from unittest.mock import MagicMock
from PIL import Image
from ocr_roi_validator.ocr_engine import OCRBox, OCRRunResult, OCREngine
from ocr_roi_validator.superscript import apply_superscript_tm
from ocr_roi_validator.gui import OCRValidatorGUI


def box(text, rect, score=.99):
    x1,y1,x2,y2 = rect
    return OCRBox(x1,y1,x2,y2,(x1+x2)/2,(y1+y2)/2,text,score)


def run_result(boxes):
    text, score = OCREngine.text_from_boxes(boxes)
    return OCRRunResult(text,score,len(boxes),boxes)


class SuperscriptTests(unittest.TestCase):
    def test_small_raised_adjacent_tm_is_attached_without_changing_raw_or_boxes(self):
        boxes = [box('Limpieza de boquillas ezDispense', (5,20,300,50)), box('TM',(303,15,321,30))]
        result = run_result(boxes)
        original = result.text
        apply_superscript_tm(result)
        self.assertEqual(result.text, 'Limpieza de boquillas ezDispenseTM')
        self.assertEqual(result.raw_text, original)
        self.assertEqual(result.boxes, boxes)
        self.assertEqual(result.superscript_evidence['status'], 'SUPERSCRIPT_TM_ATTACHED')

    def test_baseline_distant_low_confidence_and_ambiguous_marks_are_not_joined(self):
        body = box('ezDispense',(5,20,150,50))
        variants = [box('TM',(153,20,179,50)), box('TM',(200,15,218,30)),
                    box('TM',(153,15,171,30),.2)]
        for mark in variants:
            result = run_result([body,mark])
            raw = result.text
            apply_superscript_tm(result)
            self.assertEqual(result.text, raw)
        result = run_result([body, box('another',(20,20,150,50)), box('TM',(153,15,171,30))])
        raw = result.text
        apply_superscript_tm(result)
        self.assertEqual(result.text,raw)

    def test_one_box_tm_has_no_geometry_evidence(self):
        result = run_result([box('ezDispense TM',(5,20,180,50))])
        apply_superscript_tm(result)
        self.assertEqual(result.text,'ezDispense TM')
        self.assertEqual(result.superscript_evidence['status'],'NO_SEPARATE_TM_BOX')

    def test_gui_run_engine_uses_geometry_without_additional_ocr(self):
        gui = OCRValidatorGUI.__new__(OCRValidatorGUI)
        gui.language_var = MagicMock(get=lambda:'en_es')
        gui.engine = MagicMock()
        value = run_result([box('ezDispense',(5,20,150,50)),box('TM',(153,15,171,30))])
        raw = value.text
        gui.engine.run.return_value = value
        result = gui._run_engine(Image.new('RGB',(200,70)), 'ezDispenseTM', record_as='direct')
        self.assertEqual(result.text,'ezDispenseTM')
        gui.engine.run.assert_called_once()
        self.assertEqual(gui._last_ocr_input.raw_ocr_text,raw)
        self.assertTrue(gui._last_ocr_input.exact)
        self.assertEqual(gui._last_ocr_input.superscript_evidence['status'],'SUPERSCRIPT_TM_ATTACHED')
