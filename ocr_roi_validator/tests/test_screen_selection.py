"""Coordinate regression tests for scaled, multi-monitor selection previews."""
import unittest

from unittest.mock import MagicMock
from ocr_roi_validator.gui import _preview_selection_rect, ScreenAreaSelector


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


class PrecisionSelectionTests(unittest.TestCase):
    def test_scrolled_zoomed_drag_uses_canvas_coordinates_and_waits_for_confirm(self):
        picker = ScreenAreaSelector.__new__(ScreenAreaSelector)
        picker.monitor = {"left": -1920, "top": -100, "width": 1920, "height": 1080}
        picker.preview_size = (7680, 4320)
        picker.canvas = MagicMock()
        picker.canvas.canvasx.side_effect = lambda x: x+2000
        picker.canvas.canvasy.side_effect = lambda y: y+400
        picker._draw_selection = MagicMock()
        picker._sync_coordinates = MagicMock()
        picker.destroy = MagicMock()
        picker.result_rect = None
        picker._on_press(MagicMock(x=40, y=40))
        picker._on_release(MagicMock(x=440, y=240))
        self.assertEqual(picker.selection, (-1410, 10, -1310, 60))
        self.assertIsNone(picker.result_rect)
        picker.destroy.assert_not_called()

    def test_confirm_uses_manually_adjusted_one_pixel_coordinates(self):
        picker = ScreenAreaSelector.__new__(ScreenAreaSelector)
        picker.monitor = {"left": -1920, "top": -100, "width": 1920, "height": 1080}
        picker.coordinate_vars = [MagicMock() for _ in range(4)]
        for var, value in zip(picker.coordinate_vars, ['-1409', '11', '99', '49']):
            var.get.return_value = value
        picker._draw_selection = MagicMock()
        picker.destroy = MagicMock()
        picker._confirm()
        self.assertEqual(picker.result_rect, (-1409, 11, -1310, 60))
        picker.destroy.assert_called_once()

    def test_zoom_redraw_preserves_source_selection(self):
        picker = ScreenAreaSelector.__new__(ScreenAreaSelector)
        picker.monitor = {"left": -1920, "top": -100, "width": 1920, "height": 1080}
        picker.selection = (-1410, 10, -1310, 60)
        picker.preview_size = (7680, 4320)
        picker.rect_id = 1
        picker.canvas = MagicMock()
        picker._draw_selection()
        picker.canvas.coords.assert_called_with(1, 2040, 440, 2440, 640)
        self.assertEqual(picker.selection, (-1410, 10, -1310, 60))


if __name__ == '__main__':
    unittest.main()
