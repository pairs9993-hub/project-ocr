import unittest
from unittest.mock import MagicMock
from PIL import Image
from ocr_roi_validator.ocr_engine import OCRBox, OCRRunResult, OCREngine
from ocr_roi_validator.superscript import apply_superscripts
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
        apply_superscripts(result)
        self.assertEqual(result.text, 'Limpieza de boquillas ezDispense\u2122')
        self.assertEqual(result.raw_text, original)
        self.assertEqual(result.boxes, boxes)
        self.assertEqual(result.superscript_evidence['status'], 'SUPERSCRIPT_ATTACHED')

    def test_baseline_distant_low_confidence_and_ambiguous_marks_are_not_joined(self):
        body = box('ezDispense',(5,20,150,50))
        variants = [box('TM',(153,20,179,50)), box('TM',(200,15,218,30)),
                    box('TM',(153,15,171,30),.2)]
        for mark in variants:
            result = run_result([body,mark])
            raw = result.text
            apply_superscripts(result)
            self.assertEqual(result.text, raw)
        result = run_result([body, box('another',(20,20,150,50)), box('TM',(153,15,171,30))])
        raw = result.text
        apply_superscripts(result)
        self.assertEqual(result.text,raw)

    def test_one_box_tm_has_no_geometry_evidence(self):
        result = run_result([box('ezDispense TM',(5,20,180,50))])
        apply_superscripts(result)
        self.assertEqual(result.text,'ezDispense TM')
        self.assertEqual(result.superscript_evidence['status'],'NO_SEPARATE_REGION')

    def test_gui_run_engine_uses_geometry_without_additional_ocr(self):
        gui = OCRValidatorGUI.__new__(OCRValidatorGUI)
        gui.language_var = MagicMock(get=lambda:'en_es')
        gui.engine = MagicMock()
        value = run_result([box('ezDispense',(5,20,150,50)),box('TM',(153,15,171,30))])
        raw = value.text
        gui.engine.run.return_value = value
        result = gui._run_engine(Image.new('RGB',(200,70)), 'ezDispenseTM', record_as='direct')
        self.assertEqual(result.text,'ezDispense\u2122')
        gui.engine.run.assert_called_once()
        self.assertEqual(gui._last_ocr_input.raw_ocr_text,raw)
        self.assertTrue(gui._last_ocr_input.exact)
        self.assertEqual(gui._last_ocr_input.superscript_evidence['status'],'SUPERSCRIPT_ATTACHED')

    def test_generic_short_scripts_and_short_body(self):
        for mark, rendered in [('MC','\u1d39\ua7f2'), ('2','\u00b2'), ('a','\u1d43'), ('\u03b2','\u1d5d'), ('\u00ae','\u00ae')]:
            value = run_result([box('x',(5,20,30,50)),box(mark,(33,15,51,30))])
            apply_superscripts(value)
            self.assertEqual(value.text, 'x'+rendered)
            evidence = value.superscript_evidence
            self.assertEqual(evidence['attachments'][0]['text'], mark)
            self.assertEqual(evidence['attachments'][0]['relation'], 'superscript')
            self.assertTrue(any(item['geometry_confirmed'] for item in evidence['relations']))

    def test_subscript_and_normal_small_word_are_not_attached(self):
        for rect in ((153,40,171,55), (153,35,171,50)):
            value = run_result([box('body',(5,20,150,50)),box('MC',rect)])
            raw = value.text
            apply_superscripts(value)
            self.assertEqual(value.text,raw)
            self.assertNotEqual(value.superscript_evidence['status'],'SUPERSCRIPT_ATTACHED')
            self.assertTrue(any('raised_baseline' in r['rejected_checks'] for r in value.superscript_evidence['relations']))

    def test_competing_marks_abstain_instead_of_dropping_one(self):
        value = run_result([box('body',(5,20,150,50)),box('M',(152,15,156,30)),box('C',(157,15,161,30))])
        raw = value.text
        apply_superscripts(value)
        self.assertEqual(value.text,raw)
        self.assertEqual(value.superscript_evidence['status'],'AMBIGUOUS_RELATION')

    def test_report_exports_geometry_and_reason_fields(self):
        import tempfile
        import json
        from pathlib import Path
        from openpyxl import load_workbook
        from ocr_roi_validator.automation_report import write_excel_report
        value = run_result([box('body',(5,20,150,50)),box('MC',(153,15,171,30))])
        apply_superscripts(value)
        timing = dict(frame=1,captured_sec=0,ocr_started_sec=.1,ocr_finished_sec=.2,
                      status='RETURNED',superscript=value.superscript_evidence)
        report = {'results':[dict(sheet='TC',row=2,title='script',preset='p',status='FAIL',
                  rois=[dict(roi_id=1,details={'frame_timings':[timing]})])]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'report.xlsx'
            write_excel_report(path,report)
            book = load_workbook(path)
            sheet = book['OCR Timing']
            fields = dict(zip([c.value for c in sheet[1]],[c.value for c in sheet[2]]))
            self.assertEqual(fields['Superscript status'],'SUPERSCRIPT_ATTACHED')
            self.assertEqual(json.loads(fields['Script boxes'])[1]['text'],'MC')
            self.assertTrue(any(r['geometry_confirmed'] for r in json.loads(fields['Script relations'])))
            book.close()

    def test_user_tm_coordinates_allow_three_pixel_box_overlap(self):
        value = run_result([box('ezDispense',(6,6,154,37)),box('TM',(151,6,175,20))])
        raw = value.text
        apply_superscripts(value)
        self.assertEqual(value.text,'ezDispense\u2122')
        self.assertEqual(value.raw_text,raw)
        evidence = value.superscript_evidence
        self.assertEqual(evidence['status'],'SUPERSCRIPT_ATTACHED')
        relation = next(r for r in evidence['relations'] if r['body_box'] == 0)
        self.assertEqual(relation['overlap_pixels'],3)
        self.assertAlmostEqual(relation['overlap_limit_pixels'],4.65)
        self.assertEqual(relation['rejected_checks'],[])

    def test_excessive_or_contained_overlap_and_low_score_still_rejected(self):
        for rect, score in [((145,6,169,20),.99), ((140,6,151,20),.99),
                            ((151,6,155,20),.99), ((151,6,175,20),.2)]:
            value = run_result([box('ezDispense',(6,6,154,37)),box('TM',rect,score)])
            raw = value.text
            apply_superscripts(value)
            self.assertEqual(value.text,raw)
            self.assertNotEqual(value.superscript_evidence['status'],'SUPERSCRIPT_ATTACHED')

    def test_confirmed_tm_matches_trademark_but_baseline_tm_does_not(self):
        from ocr_roi_validator.compare import compare_text
        value = run_result([box('ezDispense',(4,4,219,38)),box('TM',(215,6,240,20))])
        apply_superscripts(value)
        self.assertTrue(compare_text('ezDispense\u2122', value.text, 'exact', .9).passed)
        baseline = run_result([box('ezDispense',(4,4,219,38)),box('TM',(222,4,248,38))])
        apply_superscripts(baseline)
        self.assertFalse(compare_text('ezDispense\u2122', baseline.text, 'exact', .9).passed)

    def test_unsupported_unicode_script_gets_real_excel_superscript_format(self):
        from ocr_roi_validator.automation_report import superscript_rich_text
        from openpyxl.cell.rich_text import TextBlock
        value = run_result([box('body',(5,20,150,50)),box('Z',(153,15,171,30))])
        apply_superscripts(value)
        self.assertEqual(value.text,'body^{Z}')
        rich = superscript_rich_text(value.text, value.superscript_evidence['attachments'])
        blocks = [p for p in rich if isinstance(p,TextBlock)]
        self.assertEqual(blocks[0].font.vertAlign, 'superscript')
        self.assertEqual(blocks[0].text, 'Z')

    def test_excel_round_trip_keeps_superscript_style(self):
        import tempfile
        from pathlib import Path
        from openpyxl import load_workbook
        from openpyxl.cell.rich_text import TextBlock
        from ocr_roi_validator.automation_report import write_excel_report
        value = run_result([box('body',(5,20,150,50)),box('Z',(153,15,171,30))])
        apply_superscripts(value)
        timing = dict(frame=1,captured_sec=0,ocr_started_sec=.1,ocr_finished_sec=.2,
                      status='RETURNED',raw_text=value.raw_text,evaluated_text=value.text,
                      superscript=value.superscript_evidence)
        report = {'results':[dict(sheet='TC',row=2,title='script',preset='p',status='FAIL',
                  rois=[dict(roi_id=1,actual=value.text,details={'display_source':'evaluated_observations',
                                                            'frame_timings':[timing]})])]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'report.xlsx'
            write_excel_report(path,report)
            book = load_workbook(path,rich_text=True)
            for sheet, address in [('ROI Results','I2'),('OCR Timing','J2')]:
                value = book[sheet][address].value
                self.assertEqual(str(value),'bodyZ')
                self.assertTrue(any(isinstance(part,TextBlock) and part.font.vertAlign == 'superscript' for part in value))
            self.assertIsInstance(book['OCR Timing']['I2'].value,str)
            book.close()

    def test_mc_excel_uses_font_superscript_without_requiring_modifier_font(self):
        from ocr_roi_validator.automation_report import superscript_rich_text
        from openpyxl.cell.rich_text import TextBlock
        value = run_result([box('brand',(5,20,150,50)),box('MC',(153,15,171,30))])
        apply_superscripts(value)
        rich = superscript_rich_text(value.text,value.superscript_evidence['attachments'])
        self.assertEqual(str(rich),'brandMC')
        self.assertTrue(any(isinstance(part,TextBlock) and part.text == 'MC' and part.font.vertAlign == 'superscript' for part in rich))
