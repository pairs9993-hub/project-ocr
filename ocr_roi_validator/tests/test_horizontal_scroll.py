import unittest

from ocr_roi_validator.horizontal_scroll import HorizontalScrollEvidence
from ocr_roi_validator.scroll_merge import ScrollTextAccumulator


class HorizontalEvidenceTests(unittest.TestCase):
    def test_middle_start_and_wrap_preserve_sentence_start(self):
        evidence = HorizontalScrollEvidence('Centrifugado Extra Fuerte')
        frames = ['Extra Fuerte', 'Fuerte Centri', 'Centrifugado Ex', 'gado Extra Fue']
        for _ in range(2):
            for frame in frames:
                evidence.add(frame)
        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.assembled_text, 'Centrifugado Extra Fuerte')

    def test_missing_space_does_not_shift_character_placement_or_pass(self):
        evidence = HorizontalScrollEvidence('Suciedad Pesado')
        for _ in range(3):
            evidence.add('SuciedadPesado')
        self.assertEqual(evidence.coverage, 1)
        self.assertEqual(evidence.spacing_status, 'MISMATCH')
        self.assertEqual(evidence.assembled_text, 'SuciedadPesado')
        self.assertFalse(evidence.passed)

    def test_multiple_correct_space_observations_can_recover(self):
        evidence = HorizontalScrollEvidence('Suciedad Pesado')
        evidence.add('SuciedadPesado')
        for _ in range(2):
            evidence.add('ciedad Pesado')
        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.assembled_text, 'Suciedad Pesado')

    def test_unexpected_internal_space_is_a_mismatch(self):
        evidence = HorizontalScrollEvidence('Suciedad Pesado')
        for _ in range(2):
            evidence.add('Suci edad Pesado')
        self.assertFalse(evidence.passed)
        self.assertEqual(evidence.spacing_status, 'MISMATCH')

    def test_ignore_space_is_explicit_and_preserves_original_detection(self):
        evidence = HorizontalScrollEvidence('Suciedad Pesado', 'ignore_space')
        for _ in range(2):
            evidence.add('SuciedadPesado')
        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.assembled_text, 'SuciedadPesado')
        self.assertEqual(evidence.spacing_status, 'MISMATCH')

    def test_repeated_word_cannot_fill_two_positions(self):
        evidence = HorizontalScrollEvidence('ONE TWO ONE END')
        for _ in range(10):
            evidence.add('ONE')
        self.assertEqual(evidence.coverage, 0)
        self.assertFalse(evidence.passed)
        self.assertEqual(evidence.ambiguous_frames, 10)

    def test_disjoint_words_do_not_prove_word_order(self):
        evidence = HorizontalScrollEvidence('Centrifugado Extra Fuerte')
        for _ in range(3):
            for frame in ('Fuerte', 'Extra', 'Centrifugado'):
                evidence.add(frame)
        self.assertEqual(evidence.coverage, 1)
        self.assertFalse(evidence.order_valid)
        self.assertFalse(evidence.passed)
        self.assertIn('…', evidence.assembled_text)

    def test_noisy_frame_does_not_poison_later_valid_sequence(self):
        evidence = HorizontalScrollEvidence('Centrifugado Extra Fuerte')
        evidence.add('Centrifugado Extra Fuerte')
        evidence.add('xtrq fuert3')
        evidence.add('Centrifugado Extra Fuerte')
        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.unmatched_frames, 1)

    def test_unseen_letters_are_never_filled_from_expected(self):
        evidence = HorizontalScrollEvidence('Suciedad Pesado')
        evidence.add('Suciedad')
        self.assertNotIn('Pesado', evidence.assembled_text)
        self.assertFalse(evidence.passed)

    def test_casing_respects_compare_setting(self):
        for mode, passed in [('exact', False), ('ignore_case', True)]:
            evidence = HorizontalScrollEvidence('Suciedad Pesado', mode)
            evidence.add('suciedad pesado')
            evidence.add('suciedad pesado')
            self.assertEqual(evidence.passed, passed)
            self.assertEqual(evidence.assembled_text, 'suciedad pesado')

    def test_complete_repeated_sentence_can_pass(self):
        evidence = HorizontalScrollEvidence('ONE ONE')
        evidence.add('ONE ONE')
        evidence.add('ONE ONE')
        self.assertTrue(evidence.passed)

    def test_low_score_frame_does_not_confirm_cached_success(self):
        accumulator = ScrollTextAccumulator(expected_text='Suciedad Pesado', track_observed_text=True)
        accumulator.add('Suciedad Pesado', .99)
        accumulator.add('Suciedad Pesado', .99)
        self.assertTrue(accumulator.cycle_complete)
        accumulator.add('Suciedad Pesado', .1)
        self.assertFalse(accumulator.evidence.last_aligned)
        self.assertEqual(accumulator.final_text, 'Suciedad Pesado')

class AutomationScrollIntegrationTests(unittest.TestCase):
    def verify(self, frames, expected, compare_mode='exact'):
        import tempfile
        import threading
        from pathlib import Path
        from unittest.mock import MagicMock, patch
        from PIL import Image
        from ocr_roi_validator.automation import TestCase
        from ocr_roi_validator.automation_ocr import FrozenValue, verify_case
        from test_automation import processor, preset
        proc = processor()
        proc.compare_mode_var = FrozenValue(compare_mode)
        saved = preset(True, 'horizontal')
        saved['rois'][0]['expected'] = expected
        clock, index = [0.0], [0]
        session = MagicMock()
        def frame(*args):
            clock[0] += .3
            return Image.new('RGB', (320, 240))
        def ocr(*args):
            text = frames[index[0] % len(frames)]
            index[0] += 1
            return MagicMock(text=text, mean_score=.99)
        session.frame.side_effect = frame
        proc._run_roi_ocr = ocr
        with tempfile.TemporaryDirectory() as directory, \
             patch('ocr_roi_validator.automation_ocr.time.monotonic', side_effect=lambda: clock[0]), \
             patch('ocr_roi_validator.automation_ocr.mss.mss'):
            result, _ = verify_case(proc, session, TestCase('tc', 2, 'test', 'p', []), saved,
                Path(directory), 6, 1000, threading.Event(), Path(directory)/'output')
        return result

    def test_wrap_observation_confirms_and_raw_frames_remain_available(self):
        result = self.verify(['Extra Fuerte', 'Fuerte Centri', 'Centrifugado Ex', 'gado Extra Fue'],
                             'Centrifugado Extra Fuerte')
        self.assertEqual(result['status'], 'PASS')
        roi = result['rois'][0]
        self.assertEqual(roi['actual'], 'Centrifugado Extra Fuerte')
        self.assertEqual(roi['details']['raw_observations'][0], 'Extra Fuerte')
        self.assertEqual(roi['details']['spacing_status'], 'MATCH')

    def test_missing_space_only_passes_when_explicitly_ignored(self):
        for mode, status in [('exact', 'FAIL_TIMEOUT'), ('ignore_space', 'PASS')]:
            result = self.verify(['SuciedadPesado'], 'Suciedad Pesado', mode)
            self.assertEqual(result['status'], status)
            self.assertEqual(result['rois'][0]['actual'], 'SuciedadPesado')
            self.assertEqual(result['rois'][0]['details']['spacing_status'], 'MISMATCH')

    def test_report_includes_spacing_and_raw_evidence(self):
        import tempfile
        from pathlib import Path
        from openpyxl import load_workbook
        from ocr_roi_validator.automation_report import write_excel_report
        tc = self.verify(['SuciedadPesado'], 'Suciedad Pesado')
        tc.update(sheet='tc', row=2, title='test', preset='p')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.xlsx'
            write_excel_report(path, {'results': [tc]})
            book = load_workbook(path)
            try:
                row = list(book['ROI Results'].values)[1]
                self.assertEqual(row[9], 'SuciedadPesado')
                self.assertEqual(row[17], 'MISMATCH')
                self.assertEqual(row[18], 'SuciedadPesado')
            finally:
                book.close()


if __name__ == '__main__':
    unittest.main()
