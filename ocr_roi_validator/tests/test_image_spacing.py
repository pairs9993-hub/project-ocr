import unittest
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image, ImageDraw, ImageFont
from ocr_roi_validator.image_spacing import infer_spacing, apply_image_spacing
from ocr_roi_validator.ocr_engine import OCRRunResult, OCRBox
from ocr_roi_validator.gui import OCRValidatorGUI

RAW = 'SuciedadPesado'
EXPECTED = 'Suciedad Pesado'


def glyph_line(space=True, light=False):
    image = Image.new('RGB', (150, 24), 'black' if light else 'white')
    draw = ImageDraw.Draw(image)
    x = 4
    for i in range(len(RAW)):
        if i == 8 and space:
            x += 9
        draw.rectangle((x, 4, x+5, 19), fill='white' if light else 'black')
        x += 8
    return image


def result(image):
    return OCRRunResult(RAW, .99, 1, [OCRBox(0, 0, image.width, image.height,
                                           image.width/2, image.height/2, RAW, .99)])


class ImageSpacingTests(unittest.TestCase):
    def test_image_supported_gap_with_both_polarities(self):
        for light in (False, True):
            text, evidence = infer_spacing(glyph_line(light=light), RAW)
            self.assertEqual(text, EXPECTED)
            self.assertEqual(evidence['boundaries'], [8])
            self.assertEqual(evidence['status'], 'IMAGE_GAP_CONFIRMED')

    def test_expected_space_is_not_inserted_when_image_has_none(self):
        image = glyph_line(space=False)
        value = result(image)
        apply_image_spacing(image, value, EXPECTED)
        self.assertEqual(value.text, RAW)
        self.assertEqual(value.spacing_evidence['status'], 'NO_WORD_GAP')

    def test_boundary_is_measured_even_if_expected_places_it_elsewhere(self):
        image = glyph_line()
        value = result(image)
        apply_image_spacing(image, value, 'Sucie dadPesado')
        self.assertEqual(value.text, EXPECTED)
        self.assertNotEqual(value.text, 'Sucie dadPesado')
        self.assertEqual(value.raw_text, RAW)

    def test_wrong_letters_are_not_fixed(self):
        image = glyph_line()
        value = result(image)
        apply_image_spacing(image, value, 'Suciedad Pesada')
        self.assertEqual(value.text, RAW)
        self.assertIsNone(value.spacing_evidence)

    def test_wrong_glyph_count_abstains(self):
        image = glyph_line()
        ImageDraw.Draw(image).rectangle((4,4,9,19), fill='white')
        text, evidence = infer_spacing(image, RAW)
        self.assertEqual(text, RAW)
        self.assertEqual(evidence['reason'], 'GLYPH_COUNT_MISMATCH')

    def test_unsupported_scripts_and_blank_input_abstain(self):
        for raw in ('한글테스트', 'line\nline', 'abc123'):
            self.assertEqual(infer_spacing(glyph_line(), raw)[0], raw)
        self.assertEqual(infer_spacing(Image.new('RGB',(150,24),'white'), RAW)[0], RAW)

    def test_existing_ocr_spaces_are_never_removed(self):
        raw = 'Suci edadPesado'
        text, evidence = infer_spacing(glyph_line(), raw)
        self.assertEqual(text, raw)
        self.assertEqual(evidence['reason'], 'OCR_IMAGE_DISAGREEMENT')

    @unittest.skipUnless(Path('C:/Windows/Fonts/arial.ttf').is_file(), 'Arial unavailable')
    def test_rendered_latin_letters_at_multiple_sizes(self):
        for size in (24,32,48):
            for light in (False, True):
                font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)
                image = Image.new('RGB', (500,90), 'black' if light else 'white')
                ImageDraw.Draw(image).text((8,8), EXPECTED, font=font, fill='white' if light else 'black')
                with self.subTest(size=size, light=light):
                    self.assertEqual(infer_spacing(image, RAW)[0], EXPECTED)

    def test_gui_engine_record_keeps_raw_ocr_and_exact_input(self):
        image = glyph_line()
        gui = OCRValidatorGUI.__new__(OCRValidatorGUI)
        gui.language_var = MagicMock()
        gui.language_var.get.return_value = 'en_es'
        gui.engine = MagicMock()
        gui.engine.run.return_value = result(image)
        value = gui._run_engine(image, EXPECTED, record_as='direct')
        self.assertEqual(value.text, EXPECTED)
        self.assertEqual(gui._last_ocr_input.raw_ocr_text, RAW)
        self.assertEqual(gui._last_ocr_input.image.tobytes(), image.tobytes())
        self.assertEqual(gui._last_ocr_input.spacing_evidence['status'], 'IMAGE_GAP_CONFIRMED')


    def test_automation_retains_raw_and_image_evidence_for_static_roi(self):
        import tempfile
        import threading
        from unittest.mock import patch
        from ocr_roi_validator.automation import TestCase
        from ocr_roi_validator.automation_ocr import verify_case
        from test_automation import processor, preset
        image = glyph_line()
        proc = processor()
        proc.engine.run.side_effect = lambda image, *args, **kwargs: result(image)
        p = preset()
        p['image_size'] = list(image.size)
        p['rois'] = [{'id': 1, 'rect': [0,0,image.width,image.height], 'expected': EXPECTED}]
        clock = [0.0]
        def capture(*args):
            clock[0] += .3
            return image
        session = MagicMock()
        session.frame.side_effect = capture
        with tempfile.TemporaryDirectory() as directory, \
             patch('ocr_roi_validator.automation_ocr.time.monotonic', side_effect=lambda: clock[0]), \
             patch('ocr_roi_validator.automation_ocr.mss.mss'):
            report, _ = verify_case(proc, session, TestCase('tc',2,'test','p',[]), p,
                Path(directory), 3, 1000, threading.Event(), Path(directory)/'output')
        self.assertEqual(report['status'], 'PASS')
        roi = report['rois'][0]
        self.assertEqual(roi['actual'], EXPECTED)
        self.assertEqual(roi['details']['raw_observations'], [RAW])
        self.assertEqual(roi['details']['image_spacing']['status'], 'IMAGE_GAP_CONFIRMED')


    def test_joined_letters_use_independent_word_ocr(self):
        image = glyph_line()
        # Two glyphs touch: there is no longer a one-column-run-per-letter map.
        ImageDraw.Draw(image).rectangle((9,4,12,19), fill='black')
        value = result(image)
        recognize = MagicMock(side_effect=[MagicMock(text='Suciedad',mean_score=.99),
                                           MagicMock(text='Pesado',mean_score=.99)])
        apply_image_spacing(image,value,EXPECTED,recognize)
        self.assertEqual(value.text, EXPECTED)
        self.assertEqual(value.raw_text, RAW)
        self.assertEqual(recognize.call_count, 2)
        self.assertEqual(value.spacing_evidence['method'], 'stable_gaps_independent_word_ocr')

    def test_word_retry_cannot_change_letters_to_expected(self):
        image = glyph_line()
        ImageDraw.Draw(image).rectangle((9,4,12,19), fill='black')
        value = result(image)
        recognize = MagicMock(side_effect=[MagicMock(text='Suciedad',mean_score=.99),
                                           MagicMock(text='Pesada',mean_score=.99)])
        apply_image_spacing(image,value,EXPECTED,recognize)
        self.assertEqual(value.text, RAW)
        self.assertEqual(value.spacing_evidence['status'], 'UNCERTAIN')

    def test_accent_mismatch_is_reported_without_correcting_from_expected(self):
        image = glyph_line()
        value = result(image)
        value.text = 'Temp. Agua fria'
        apply_image_spacing(image,value,'Temp. Agua fría')
        self.assertEqual(value.text, 'Temp. Agua fria')
        self.assertEqual(value.spacing_evidence['reason'], 'DIACRITIC_MISMATCH')

    def test_word_retry_diagnostic_does_not_claim_single_exact_input(self):
        image = glyph_line()
        ImageDraw.Draw(image).rectangle((9,4,12,19), fill='black')
        gui = OCRValidatorGUI.__new__(OCRValidatorGUI)
        gui.language_var = MagicMock()
        gui.language_var.get.return_value = 'en_es'
        gui.engine = MagicMock()
        gui.engine.run.side_effect = [result(image), MagicMock(text='Suciedad',mean_score=.99),
                                      MagicMock(text='Pesado',mean_score=.99)]
        value = gui._run_engine(image,EXPECTED,record_as='direct')
        self.assertEqual(value.text,EXPECTED)
        self.assertFalse(gui._last_ocr_input.exact)
        self.assertEqual(gui._last_ocr_input.raw_ocr_text,RAW)


if __name__ == '__main__':
    unittest.main()
