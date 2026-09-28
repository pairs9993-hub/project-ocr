import unittest
from unittest.mock import MagicMock
from PIL import Image, ImageDraw
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
    def test_recorded_clipped_o_and_m_are_not_promoted(self):
        examples = [
            [box('imorcer - Appuyez sur',(4,13,282,44),.965), box('O',(285,15,302,35),.762)],
            [box('de la buse ezDispense',(5,1,307,36),.980), box('M',(305,7,316,17),.976)],
        ]
        for boxes in examples:
            value = run_result(boxes)
            raw = value.text
            apply_superscripts(value, Image.new('RGB',(319,160)))
            self.assertEqual(value.text,raw)
            self.assertFalse(value.superscript_evidence['attachments'])
            relation = next(r for r in value.superscript_evidence['relations'] if r['body_box']==0)
            self.assertIn('single_letter_context',relation['rejected_checks'])

    def test_single_letter_with_room_and_confidence_remains_supported(self):
        value = run_result([box('x',(5,20,30,50)),box('a',(33,15,51,30))])
        apply_superscripts(value, Image.new('RGB',(150,70)))
        self.assertEqual(value.text,'x\u1d43')

    def test_complete_mc_near_right_edge_keeps_existing_geometry_policy(self):
        value = run_result([box('le la buse ezDispense',(5,1,298,36),.98),
                            box('MC',(294,4,319,19),.948)])
        apply_superscripts(value, Image.new('RGB',(319,160)))
        self.assertEqual(value.text,'le la buse ezDispense\U0001f16a')

    def pixel_example(self, light=False, bridge=False):
        image = Image.new('RGB', (190,60), 'white' if light else 'black')
        draw = ImageDraw.Draw(image)
        foreground = 'black' if light else 'white'
        for x in range(10, 141, 10):
            draw.rectangle((x,10,x+4,38), fill=foreground)
        draw.rectangle((149,10,157,21), fill=foreground)
        draw.rectangle((160,10,170,21), fill=foreground)
        if bridge:
            draw.rectangle((144,15,150,17), fill=foreground)
        boxes = [box('body',(5,5,151,43)), box('TM',(145,5,174,25))]
        return image, run_result(boxes)

    def test_excess_detector_overlap_needs_stable_pixel_gap(self):
        import json
        for light in (False, True):
            image, value = self.pixel_example(light)
            raw, original_boxes = value.text, list(value.boxes)
            apply_superscripts(value, image)
            self.assertEqual(value.text, 'body\u2122')
            self.assertEqual(value.raw_text, raw)
            self.assertEqual(value.boxes, original_boxes)
            relation = next(r for r in value.superscript_evidence['relations']
                            if r['body_box'] == 0 and r['script_box'] == 1)
            self.assertGreater(relation['overlap_pixels'], relation['overlap_limit_pixels'])
            self.assertEqual(relation['pixel_separation']['method'], 'stable_ink_separation')
            json.dumps(value.superscript_evidence)

    def test_touching_ink_blank_image_or_absent_pixels_do_not_relax_overlap(self):
        for image in (self.pixel_example(bridge=True)[0], Image.new('RGB',(190,60)), None):
            _, value = self.pixel_example()
            raw = value.text
            apply_superscripts(value, image)
            self.assertEqual(value.text, raw)
            self.assertEqual(value.superscript_evidence['status'], 'GEOMETRY_NOT_CONFIRMED')

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
        for mark, rendered in [('MC','\U0001f16a'), ('2','\u00b2'), ('a','\u1d43'), ('\u03b2','\u1d5d'), ('\u00ae','\u00ae')]:
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

    def test_confirmed_mc_matches_raised_mc_using_report_coordinates(self):
        from ocr_roi_validator.compare import compare_text
        boxes = [box('buse ezDispense', (9,16,271,53)), box('MC', (266,17,296,35))]
        value = run_result(boxes)
        raw = value.text
        apply_superscripts(value)
        self.assertEqual(value.text, 'buse ezDispense\U0001f16a')
        self.assertTrue(compare_text('buse ezDispense\U0001f16a', value.text, 'exact').passed)
        self.assertEqual(value.raw_text, raw)
        self.assertEqual(value.boxes, boxes)
        self.assertEqual(value.superscript_evidence['status'], 'SUPERSCRIPT_ATTACHED')
        self.assertEqual(value.superscript_evidence['attachments'][0]['text'], 'MC')

    def test_unconfirmed_mc_and_wrong_case_are_not_promoted_to_raised_mc(self):
        from ocr_roi_validator.compare import compare_text
        for boxes in (
            [box('ezDispense', (5,20,150,50)), box('MC', (153,20,180,50))],
            [box('ezDispense MC', (5,20,180,50))],
            [box('ezDispense', (5,20,150,50)), box('MC', (200,15,218,30))],
            [box('ezDispense', (5,20,150,50)), box('MC', (153,15,171,30), .2)],
            [box('ezDispense', (5,20,150,50)), box('Mc', (153,15,171,30))],
        ):
            with self.subTest(boxes=boxes):
                value = run_result(boxes)
                apply_superscripts(value)
                self.assertNotIn('\U0001f16a', value.text)
                self.assertFalse(compare_text('ezDispense\U0001f16a', value.text, 'exact').passed)

    def test_confirmed_mc_scroll_fragments_assemble_without_expected_completion(self):
        from ocr_roi_validator.horizontal_scroll import HorizontalScrollEvidence
        expected = 'Nettoyage de la buse ezDispense\U0001f16a'
        evidence = HorizontalScrollEvidence(expected)
        for _ in range(2):
            evidence.add('Nettoyage de la buse ezDis')
        self.assertFalse(evidence.assembly_complete)
        self.assertFalse(evidence.passed)
        for _ in range(2):
            value = run_result([box('buse ezDispense', (9,16,271,53)), box('MC', (266,17,296,35))])
            apply_superscripts(value)
            evidence.add(value.text)
            self.assertEqual(evidence.last_reason, 'ALIGNED')
        self.assertTrue(evidence.assembly_complete)
        self.assertEqual(evidence.assembled_text, expected)
        self.assertTrue(evidence.passed)

    def test_already_recognized_raised_mc_is_preserved(self):
        value = run_result([box('ezDispense', (5,20,150,50)), box('\U0001f16a', (153,15,171,30))])
        apply_superscripts(value)
        self.assertEqual(value.text, 'ezDispense\U0001f16a')
        self.assertEqual(value.superscript_evidence['status'], 'SUPERSCRIPT_ATTACHED')

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
