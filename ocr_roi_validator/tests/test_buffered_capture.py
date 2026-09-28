import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
from ocr_roi_validator.buffered_capture import BufferedCapture
from ocr_roi_validator.automation import Cancelled, TestCase
from ocr_roi_validator.automation_ocr import verify_case
from ocr_roi_validator.image_spacing import apply_image_spacing
from test_automation import processor, preset
from test_static_and_diacritics import result


class BufferedCaptureTests(unittest.TestCase):
    def test_capture_keeps_latest_frame_instead_of_backlog(self):
        session = MagicMock()
        sequence = [0]
        def capture(*args):
            sequence[0] += 1
            return Image.new('RGB', (10, 10), (sequence[0], 0, 0))
        session.frame.side_effect = capture
        with patch('ocr_roi_validator.buffered_capture.mss.mss'):
            with BufferedCapture(session, {}, 200, .08, threading.Event(), max_bytes=600) as stream:
                stream.thread.join(timeout=1)
                stats = stream.stats()
                self.assertGreater(stats['captured_frames'], 2)
                self.assertEqual(stats['pending_frames'], 1)
                self.assertEqual(stats['superseded_frames'], stats['captured_frames']-1)
                self.assertEqual(stats['dropped_frames'], 0)
                first = stream.next_frame(time.monotonic()+1)
                self.assertIsNotNone(first)
                self.assertEqual(first[0].getpixel((0, 0))[0], stats["captured_frames"])
                self.assertIsNone(stream.next_frame(time.monotonic()+1))
            self.assertFalse(stream.thread.is_alive())

    def test_cancel_and_capture_error_propagate_and_stop_thread(self):
        stop = threading.Event()
        stop.set()
        with patch('ocr_roi_validator.buffered_capture.mss.mss'):
            with BufferedCapture(MagicMock(), {}, 100, 1, stop) as stream:
                with self.assertRaises(Cancelled):
                    stream.next_frame(time.monotonic()+1)
            session = MagicMock()
            session.frame.side_effect = RuntimeError('capture failed')
            with BufferedCapture(session, {}, 100, 1, threading.Event()) as stream:
                with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                    stream.next_frame(time.monotonic()+1)

    def test_automation_skips_stale_frames_and_stops_at_observation_deadline(self):
        proc = processor('Wrong')
        del proc._capture_factory
        def slow_ocr(*args):
            time.sleep(.02)
            return result('Wrong')
        proc._run_roi_ocr = slow_ocr
        session = MagicMock()
        session.frame.return_value = Image.new('RGB', (320, 240))
        with tempfile.TemporaryDirectory() as folder, patch('ocr_roi_validator.buffered_capture.mss.mss'):
            report, _ = verify_case(proc, session, TestCase('tc', 2, 'test', 'p', []),
                preset(True), Path(folder), .1, 100, threading.Event(), Path(folder)/'out')
        details = report['rois'][0]['details']
        self.assertGreater(details['capture']['captured_frames'], 2)
        self.assertGreater(details['capture']['captured_frames'], details['ocr_calls'])
        self.assertGreater(details['capture']['superseded_frames'], 0)
        self.assertLessEqual(details['capture']['pending_frames'], 1)
        self.assertLess(report['elapsed_sec'], .5)
        self.assertEqual(report['status'], 'FAIL_TIMEOUT')


class IndependentWordRetryTests(unittest.TestCase):
    def test_extra_letter_is_replaced_only_by_agreeing_word_ocr(self):
        raw, expected = 'Temp. Agua f fria', 'Temp. Agua fr\u00eda'
        value = result(raw)
        responses = [result(text) for text in ['Temp.', 'Temp.', 'Agua', 'Agua', 'fr\u00eda', 'fr\u00eda']]
        with patch('ocr_roi_validator.image_spacing.word_regions', return_value=[(0,0,40,30), (40,0,90,30), (90,0,160,30)]):
            apply_image_spacing(Image.new('RGB', (160,30)), value, expected, MagicMock(side_effect=responses), allow_word_retry=True)
        self.assertEqual(value.text, expected)
        self.assertEqual(value.raw_text, raw)
        self.assertEqual(value.spacing_evidence['method'], 'independent_word_retry')

    def test_disagreeing_word_ocr_does_not_fill_expected(self):
        value = result('Temp. Agua f fria')
        with patch('ocr_roi_validator.image_spacing.word_regions', return_value=[(0,0,40,30), (40,0,160,30)]):
            apply_image_spacing(Image.new('RGB', (160,30)), value, 'Temp. Agua fr\u00eda',
                                MagicMock(side_effect=[result('Temp.'), result('Wrong')]), allow_word_retry=True)
        self.assertEqual(value.text, 'Temp. Agua f fria')

    def test_expensive_word_retry_is_disabled_by_default(self):
        value = result('Temp. Agua f fria')
        recognize = MagicMock()
        apply_image_spacing(Image.new('RGB', (160,30)), value, 'Temp. Agua fr\u00eda', recognize)
        recognize.assert_not_called()
        self.assertEqual(value.text, 'Temp. Agua f fria')
