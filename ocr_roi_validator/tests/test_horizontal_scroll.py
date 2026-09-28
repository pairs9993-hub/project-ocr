import unittest

from ocr_roi_validator.horizontal_scroll import HorizontalScrollEvidence
from ocr_roi_validator.scroll_merge import ScrollTextAccumulator


class HorizontalEvidenceTests(unittest.TestCase):
    def test_live_display_preserves_low_confidence_and_unaligned_text(self):
        from ocr_roi_validator.scroll_merge import ScrollTextAccumulator
        acc = ScrollTextAccumulator(expected_text='Temp. Agua fr\u00eda', track_observed_text=True)
        acc.add('Temp. Agua fria', .1)
        self.assertEqual(acc.display_text, 'Temp. Agua fria')
        self.assertFalse(acc.cycle_complete)
        acc.add('Temp. Agua frla', .99)
        self.assertIn('Temp. Agua frla', acc.display_text)
        self.assertNotIn('…', acc.display_text)

    def test_missing_accent_is_assembled_as_observed_and_fails(self):
        evidence = HorizontalScrollEvidence('Licencias de C\u00f3digo Abierto')
        for _ in range(2):
            for text in ['Licencias de Codigo', 'de Codigo Abierto']:
                evidence.add(text)
        self.assertEqual(evidence.assembled_text, 'Licencias de Codigo Abierto')
        self.assertEqual(evidence.coverage, 1)
        self.assertTrue(evidence.order_valid)
        self.assertFalse(evidence.passed)
        self.assertEqual(evidence.last_reason, 'ALIGNED_WITH_SUBSTITUTIONS')
        self.assertEqual(evidence.last_substitutions[0]['observed'], 'o')

    def test_wrong_tail_letter_is_not_replaced_with_expected(self):
        evidence = HorizontalScrollEvidence('Centrifugado Extra Fuerte')
        for _ in range(2):
            evidence.add('Centrifugado Extra Fuert3')
        self.assertEqual(evidence.assembled_text, 'Centrifugado Extra Fuert3')
        self.assertFalse(evidence.passed)

    def test_ambiguous_approximate_positions_are_rejected(self):
        evidence = HorizontalScrollEvidence('abcdefghXabcdefghY')
        evidence.add('abcdefghZ')
        self.assertEqual(evidence.coverage, 0)
        self.assertFalse(evidence.last_aligned)

    def test_tm_expansion_does_not_wrap_m_onto_first_letter(self):
        evidence = HorizontalScrollEvidence('Limpieza de boquillas ezDispense\u2122')
        for _ in range(3):
            evidence.add('Limpieza de boquillas ezDispense TM')
            evidence.add('boquillas ezDispense TM')
        self.assertEqual(evidence.assembled_text, 'Limpieza de boquillas ezDispense TM')
        self.assertEqual(evidence.letters[0], {'L': 3})
        self.assertEqual(evidence.last_substitutions, [])
        self.assertEqual(evidence.last_reason, 'ALIGNED')
        self.assertFalse(evidence.passed)

    def test_actual_trademark_glyph_still_passes(self):
        expected = 'Limpieza de boquillas ezDispense\u2122'
        evidence = HorizontalScrollEvidence(expected)
        evidence.add(expected)
        evidence.add(expected)
        self.assertEqual(evidence.assembled_text, expected)
        self.assertTrue(evidence.passed)

    def test_missing_m_is_not_filled_and_separated_t_m_not_grouped(self):
        for text in ('boquillas ezDispenseT', 'boquillas ezDispenseT M'):
            evidence = HorizontalScrollEvidence('Limpieza de boquillas ezDispense\u2122')
            evidence.add(text)
            self.assertFalse(evidence.passed)
            if text.endswith('T'):
                self.assertNotIn('TM', evidence.assembled_text)

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
        self.assertEqual(accumulator.evidence.last_reason, 'LOW_CONFIDENCE')
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

    def test_automation_assembles_accent_error_but_does_not_pass(self):
        result = self.verify(['Licencias de Codigo', 'de Codigo Abierto'], 'Licencias de C\u00f3digo Abierto')
        roi = result['rois'][0]
        self.assertEqual(result['status'], 'FAIL_TIMEOUT')
        self.assertEqual(roi['details']['assembled_text'], 'Licencias de Codigo Abierto')
        timing = roi['details']['frame_timings'][0]
        self.assertEqual(timing['assembly_reason'], 'ALIGNED_WITH_SUBSTITUTIONS')
        self.assertEqual(timing['substitutions'][0]['expected'], '\u00f3')
        self.assertEqual(timing['substitutions'][0]['observed'], 'o')

    def test_each_frame_reports_alignment_reason_and_location(self):
        result = self.verify(['Limpieza de boquillas e', 'garbled', 'ezDispenseTM'],
                             'Limpieza de boquillas ezDispenseTM')
        timings = result['rois'][0]['details']['frame_timings']
        aligned = [t for t in timings if t['evaluated_text'] == 'ezDispenseTM']
        self.assertTrue(aligned)
        self.assertEqual(aligned[0]['assembly_reason'], 'ALIGNED')
        self.assertTrue(aligned[0]['assembly_accepted'])
        self.assertGreater(aligned[0]['aligned_start'], 1)
        rejected = [t for t in timings if t['evaluated_text'] == 'garbled']
        self.assertEqual(rejected[0]['assembly_reason'], 'NO_EXACT_ALIGNMENT')
        self.assertFalse(rejected[0]['assembly_accepted'])

    def test_accent_mismatch_remains_visible_in_automation(self):
        result = self.verify(['Temp. Agua fria'], 'Temp. Agua fr\u00eda')
        roi = result['rois'][0]
        self.assertEqual(roi['actual'], 'Temp. Agua fria')
        self.assertFalse(roi['passed'])
        self.assertEqual(roi['details']['display_source'], 'raw_observations')

    def test_missing_superscript_keeps_read_body_visible(self):
        raw = 'Limpieza de boquillas ezdispense'
        result = self.verify([raw], raw+'TM')
        roi = result['rois'][0]
        self.assertEqual(roi['actual'], raw)
        self.assertFalse(roi['passed'])
        self.assertIn('…', roi['details']['assembled_text'])

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
