"""Image-defined spacing when touching letters defeat per-glyph mapping."""
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image, ImageDraw

from ocr_roi_validator.image_spacing import WordRetryCache, apply_image_spacing, infer_spacing, word_regions
from ocr_roi_validator.ocr_engine import OCRBox, OCRRunResult


WORDS = ['Nettoyage', '-', 'Appuyez', 'sur', 'OK.']
RAW = 'Nettoyage-Appuyez sur OK.'
EXPECTED = 'Nettoyage - Appuyez sur OK.'


def line_image(words=WORDS, light=False):
    background, foreground = ('black', 'white') if light else ('white', 'black')
    image = Image.new('RGB', (sum(map(len, words))*8 + (len(words)-1)*9 + 8, 28), background)
    draw = ImageDraw.Draw(image)
    x = 4
    for word_index, word in enumerate(words):
        if word_index:
            x += 9
        for char_index, char in enumerate(word):
            draw.rectangle((x, 12 if char == '-' else 4, x+5, 14 if char == '-' else 23), fill=foreground)
            # Join the two t crossbars without changing the text recognized by OCR.
            if word_index == 0 and char_index == 2:
                draw.rectangle((x+5, 10, x+8, 12), fill=foreground)
            x += 8
    return image


def run_result(image, text=RAW):
    return OCRRunResult(text, .99, 1, [OCRBox(0, 0, image.width, image.height,
                                            image.width/2, image.height/2, text, .99)])


def recognizer(words=WORDS):
    return MagicMock(side_effect=[MagicMock(text=text, mean_score=.99) for text in words])


class SpacingWordRetryTests(unittest.TestCase):
    def test_failed_identical_input_cools_down_but_changed_pixels_retry(self):
        image, cache = line_image(), WordRetryCache()
        failed = ['Nettoyage', '', 'Appuyez']
        recognize = recognizer(failed * 3)
        with patch('ocr_roi_validator.image_spacing.time.monotonic', return_value=0):
            first = run_result(image)
            apply_image_spacing(image, first, EXPECTED, recognize, cache)
            second = run_result(image)
            apply_image_spacing(image, second, EXPECTED, recognize, cache)
            self.assertEqual(recognize.call_count, 3)
            self.assertEqual(second.text, RAW)
            retry = second.spacing_evidence['word_retry']
            self.assertEqual(retry['retry_calls'], 0)
            self.assertEqual(retry['previous_attempt']['reason'], 'HYPHEN_CONTEXT_MISMATCH')
            changed = image.copy()
            changed.putpixel((0, 0), (254, 254, 254))
            apply_image_spacing(changed, run_result(changed), EXPECTED, recognize, cache)
            self.assertEqual(recognize.call_count, 6)
        with patch('ocr_roi_validator.image_spacing.time.monotonic', return_value=10):
            apply_image_spacing(image, run_result(image), EXPECTED, recognize, cache)
            self.assertEqual(recognize.call_count, 9)

    def test_success_is_not_reused_as_new_independent_ocr(self):
        image, cache = line_image(), WordRetryCache()
        recognize = recognizer(WORDS * 2)
        for _ in range(2):
            result = run_result(image)
            apply_image_spacing(image, result, EXPECTED, recognize, cache)
            self.assertEqual(result.text, EXPECTED)
        self.assertEqual(recognize.call_count, 10)

    def test_slow_retry_returns_raw_and_preserves_partial_evidence(self):
        image, clock = line_image(), [0.0]
        def slow(piece):
            clock[0] += 6.1
            return MagicMock(text='Nettoyage', mean_score=.99)
        recognize = MagicMock(side_effect=slow)
        result = run_result(image)
        with patch('ocr_roi_validator.image_spacing.time.monotonic', side_effect=lambda: clock[0]):
            apply_image_spacing(image, result, EXPECTED, recognize)
        self.assertEqual(result.text, RAW)
        self.assertEqual(result.raw_text, RAW)
        recognize.assert_called_once()
        retry = result.spacing_evidence['word_retry']
        self.assertEqual(retry['reason'], 'WORD_RETRY_BUDGET_REACHED')
        self.assertEqual(retry['retry_calls'], 1)
        self.assertEqual(retry['retry_outputs'][0]['text'], 'Nettoyage')

    def test_failure_cache_bounds_memory_and_does_not_share_mutable_evidence(self):
        cache = WordRetryCache()
        evidence = {'reason': 'failure', 'retry_outputs': []}
        for key in range(20):
            cache.remember(key, evidence)
        self.assertEqual(len(cache.failures), 16)
        self.assertIsNone(cache.get(0))
        cache.get(19)['retry_outputs'].append('changed')
        self.assertEqual(cache.get(19)['retry_outputs'], [])

    def test_touching_tt_and_four_gaps_restore_spaces_from_pixels(self):
        for light in (False, True):
            with self.subTest(light=light):
                image = line_image(light=light)
                self.assertEqual(infer_spacing(image, RAW)[1]['reason'], 'GLYPH_COUNT_MISMATCH')
                self.assertEqual(len(word_regions(image)), 5)
                value, recognize = run_result(image), recognizer()
                apply_image_spacing(image, value, EXPECTED, recognize)
                self.assertEqual(value.text, EXPECTED)
                self.assertEqual(value.raw_text, RAW)
                self.assertEqual(recognize.call_count, 5)
                self.assertEqual(value.spacing_evidence['word_ocr'], WORDS)
                self.assertEqual(value.spacing_evidence['retry_calls'], 5)

    def test_expected_boundaries_are_not_used(self):
        image = line_image()
        value = run_result(image)
        apply_image_spacing(image, value, 'Net toyage-AppuyezsurOK.', recognizer())
        self.assertEqual(value.text, EXPECTED)

    def test_two_words_joined_by_ocr_still_recover(self):
        words = ['Nettoyage', 'Appuyez']
        image = line_image(words)
        value = run_result(image, 'NettoyageAppuyez')
        apply_image_spacing(image, value, 'Nettoyage Appuyez', recognizer(words))
        self.assertEqual(value.text, 'Nettoyage Appuyez')

    def test_hyphen_without_image_spaces_is_not_split(self):
        words = ['Nettoyage-Appuyez', 'sur', 'OK.']
        image = line_image(words)
        value = run_result(image)
        apply_image_spacing(image, value, EXPECTED, recognizer(words))
        self.assertEqual(value.text, RAW)

    def test_no_clear_gaps_do_not_trigger_retry(self):
        image = line_image(['Nettoyage-AppuyezsurOK.'])
        value, recognize = run_result(image), recognizer()
        apply_image_spacing(image, value, EXPECTED, recognize)
        self.assertEqual(value.text, RAW)
        recognize.assert_not_called()

    def test_wrong_letters_low_confidence_or_missing_hyphen_abstain(self):
        for words, score in [(['Nettoyage', '-', 'Appuyes', 'sur', 'OK.'], .99),
                             (['Nettoyage', '', 'Appuyez', 'sur', 'OK.'], .99),
                             (WORDS, .3)]:
            with self.subTest(words=words, score=score):
                image = line_image()
                value = run_result(image)
                recognize = MagicMock(side_effect=[MagicMock(text=w, mean_score=score) for w in words])
                apply_image_spacing(image, value, EXPECTED, recognize)
                self.assertEqual(value.text, RAW)
                self.assertEqual(value.spacing_evidence['status'], 'UNCERTAIN')
                self.assertIn('reason', value.spacing_evidence['word_retry'])
                self.assertLessEqual(recognize.call_count, 5)

    def test_existing_ocr_space_is_not_removed(self):
        image = line_image()
        raw = 'Net toyage-Appuyez sur OK.'
        value = run_result(image, raw)
        apply_image_spacing(image, value, EXPECTED, recognizer())
        self.assertEqual(value.text, raw)
        self.assertEqual(value.spacing_evidence['word_retry']['reason'], 'OCR_IMAGE_DISAGREEMENT')

    def test_retry_budget_rejects_too_many_pieces_before_ocr(self):
        words = ['Nettoyage', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff']
        image = line_image(words)
        raw = ''.join(words)
        value, recognize = run_result(image, raw), recognizer(words)
        apply_image_spacing(image, value, ' '.join(words), recognize)
        self.assertEqual(value.text, raw)
        recognize.assert_not_called()

    def test_six_piece_limit_is_supported(self):
        words = WORDS + ['Suite']
        image = line_image(words)
        value, recognize = run_result(image, ''.join(words)), recognizer(words)
        apply_image_spacing(image, value, ' '.join(words), recognize)
        self.assertEqual(value.text, ' '.join(words))
        self.assertEqual(recognize.call_count, 6)

    def test_exact_match_or_wrong_letters_do_not_add_ocr(self):
        image = line_image()
        for raw, expected in [(EXPECTED, EXPECTED), (RAW, 'Nettoyage - Appuyes sur OK.')]:
            value, recognize = run_result(image, raw), recognizer()
            apply_image_spacing(image, value, expected, recognize)
            self.assertEqual(value.text, raw)
            recognize.assert_not_called()

    def test_retry_deadline_is_not_swallowed(self):
        image = line_image()
        value = run_result(image)
        recognize = MagicMock(side_effect=TimeoutError('deadline'))
        with self.assertRaises(TimeoutError):
            apply_image_spacing(image, value, EXPECTED, recognize)
        recognize.assert_called_once()

    def test_tiny_dash_misread_uses_right_context_without_more_calls(self):
        for tiny_dash in ('1', ''):
            with self.subTest(tiny_dash=tiny_dash):
                image, recognize = line_image(), recognizer(['Nettoyage', tiny_dash, '-Appuyez', 'sur', 'OK.'])
                value = run_result(image)
                apply_image_spacing(image, value, EXPECTED, recognize)
                self.assertEqual(value.text, EXPECTED)
                self.assertEqual(value.raw_text, RAW)
                self.assertEqual(recognize.call_count, 5)
                evidence = value.spacing_evidence
                self.assertTrue(evidence['hyphen_context_confirmed'])
                self.assertEqual(evidence['word_ocr'], WORDS)
                self.assertEqual(evidence['retry_outputs'][1]['text'], tiny_dash)
                self.assertEqual(evidence['retry_outputs'][2]['text'], '-Appuyez')

    def test_dash_context_cannot_invent_dash_or_change_letters(self):
        for context in ('Appuyez', '-Appuyes', '- Appuyez sur', '-'):
            with self.subTest(context=context):
                image = line_image()
                value = run_result(image)
                recognize = recognizer(['Nettoyage', '1', context])
                apply_image_spacing(image, value, EXPECTED, recognize)
                self.assertEqual(value.text, RAW)
                self.assertEqual(value.spacing_evidence['status'], 'UNCERTAIN')
                self.assertEqual(recognize.call_count, 3)

    def test_vertical_stroke_does_not_enable_dash_context(self):
        image = line_image()
        rect = word_regions(image)[1]
        draw = ImageDraw.Draw(image)
        draw.rectangle((rect[0], 0, rect[2]-1, image.height-1), fill='white')
        draw.rectangle((rect[0]+6, 4, rect[0]+8, 23), fill='black')
        value, recognize = run_result(image), recognizer(['Nettoyage', '1'])
        apply_image_spacing(image, value, EXPECTED, recognize)
        self.assertEqual(value.text, RAW)
        self.assertLessEqual(recognize.call_count, 2)

    def test_low_confidence_context_is_not_reported_as_confirmed(self):
        image = line_image()
        value = run_result(image)
        recognize = MagicMock(side_effect=[MagicMock(text='Nettoyage', mean_score=.99),
                                           MagicMock(text='1', mean_score=.65),
                                           MagicMock(text='-Appuyez', mean_score=.3)])
        apply_image_spacing(image, value, EXPECTED, recognize)
        self.assertEqual(value.text, RAW)
        retry = value.spacing_evidence['word_retry']
        self.assertEqual(retry['reason'], 'LOW_CONFIDENCE_WORD_OCR')
        self.assertNotIn('hyphen_context_confirmed', retry)

    def test_confirmed_spacing_feeds_scroll_assembly_and_keeps_raw(self):
        from ocr_roi_validator.horizontal_scroll import HorizontalScrollEvidence
        evidence = HorizontalScrollEvidence(EXPECTED)
        for _ in range(2):
            image = line_image()
            value = run_result(image)
            apply_image_spacing(image, value, EXPECTED, recognizer())
            self.assertEqual(value.raw_text, RAW)
            evidence.add(value.text)
        self.assertTrue(evidence.assembly_complete)
        self.assertEqual(evidence.spacing_status, 'MATCH')
        self.assertTrue(evidence.passed)


if __name__ == '__main__':
    unittest.main()
