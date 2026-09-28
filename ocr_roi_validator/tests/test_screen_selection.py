"""Coordinate regression tests for scaled, multi-monitor selection previews."""
import unittest

from ocr_roi_validator.gui import _preview_selection_rect


class PreviewSelectionTests(unittest.TestCase):
    def test_negative_monitor_origin_and_half_size_preview(self):
        monitor = {"left": -1920, "top": -400, "width": 1600, "height": 1000}
        self.assertEqual(_preview_selection_rect(monitor, (800, 500), (50, 25), (750, 475)),
                         (-1820, -350, -420, 550))

    def test_reverse_drag_and_clamping_outside_preview(self):
        monitor = {"left": 1920, "top": 200, "width": 2560, "height": 1440}
        self.assertEqual(_preview_selection_rect(monitor, (1000, 563), (1050, 600), (-10, -20)),
                         (1920, 200, 4480, 1640))

    def test_rounded_preview_uses_independent_axis_ratios(self):
        monitor = {"left": -100, "top": 50, "width": 1001, "height": 701}
        self.assertEqual(_preview_selection_rect(monitor, (500, 350), (0, 0), (500, 350)),
                         (-100, 50, 901, 751))

    def test_click_without_area_is_cancelled(self):
        monitor = {"left": 0, "top": 0, "width": 1000, "height": 800}
        self.assertIsNone(_preview_selection_rect(monitor, (500, 400), (10, 20), (10, 20)))


if __name__ == '__main__':
    unittest.main()
