"""Tests separating the runtime view from the causal-audit view.

An earlier revision cropped both members of a counterfactual pair using
geometry measured from the accented member, so a bare training image carried a
trace of its counterpart's accent -- information unavailable at runtime, where
only one image exists. These tests pin the separation that fixes it.
"""

import inspect
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_roi_validator.line_views import (
    CAUSAL_AUDIT,
    RUNTIME,
    LineGeometryConfig,
    assert_runtime_view,
    causal_audit_view,
    label_neutral_bounds,
    runtime_view,
)

FONT = "C:/Windows/Fonts/arial.ttf"


def render(text, size=20, origin=(10.0, 8.0), width=220, height=48):
    image = Image.new("RGB", (width, height), (14, 14, 14))
    font = ImageFont.truetype(FONT, size)
    ImageDraw.Draw(image).text(origin, text, font=font, fill=(240, 240, 240))
    return image


class LabelNeutralityTests(unittest.TestCase):
    """Audit geometry must not depend on which e-form was drawn."""

    def test_bounds_identical_across_the_pair(self) -> None:
        bare = label_neutral_bounds("Il bebe bd", FONT, 20, 8.0, 48)
        accented = label_neutral_bounds("Il bebé bd", FONT, 20, 8.0, 48)
        self.assertEqual(bare, accented)

    def test_bounds_unchanged_when_labels_are_swapped(self) -> None:
        first = label_neutral_bounds("mémé", FONT, 18, 6.0, 40)
        swapped = label_neutral_bounds("meme", FONT, 18, 6.0, 40)
        self.assertEqual(first, swapped)

    def test_bounds_independent_of_pair_order(self) -> None:
        forward = [label_neutral_bounds(t, FONT, 16, 5.0, 40)
                   for t in ("bebe", "bebé")]
        reverse = [label_neutral_bounds(t, FONT, 16, 5.0, 40)
                   for t in ("bebé", "bebe")]
        self.assertEqual(set(forward), set(reverse))
        self.assertEqual(forward[0], forward[1])

    def test_audit_view_gives_both_members_the_same_band(self) -> None:
        bare = causal_audit_view(render("Il bebe bd"), "Il bebe bd", FONT, 20, 8.0)
        accented = causal_audit_view(render("Il bebé bd"), "Il bebé bd", FONT,
                                     20, 8.0)
        self.assertIsNotNone(bare)
        self.assertIsNotNone(accented)
        self.assertEqual((bare.top, bare.bottom), (accented.top, accented.bottom))

    def test_audit_pair_differs_only_near_the_target(self) -> None:
        bare = causal_audit_view(render("Il bebe bd"), "Il bebe bd", FONT, 20, 8.0)
        accented = causal_audit_view(render("Il bebé bd"), "Il bebé bd", FONT,
                                     20, 8.0)
        difference = np.abs(bare.image.astype(int)
                            - accented.image.astype(int)).max(axis=(0, 2))
        columns = np.where(difference > 8)[0]
        self.assertTrue(columns.size > 0, "the accent must change something")
        span = int(columns.max() - columns.min() + 1)
        self.assertLess(span, bare.image.shape[1] // 3)


class RuntimeViewIsolationTests(unittest.TestCase):
    """The training view must be derivable from one image alone."""

    def test_signature_has_no_counterpart_parameter(self) -> None:
        parameters = inspect.signature(runtime_view).parameters
        self.assertEqual(list(parameters), ["page", "config"])
        for name in parameters:
            for banned in ("counterpart", "pair", "other", "reference", "bounds"):
                self.assertNotIn(banned, name.lower())

    def test_view_is_reproducible_without_the_counterpart(self) -> None:
        page = render("Il bebe bd")
        first = runtime_view(page)
        # Nothing about the counterpart is in scope; a second call on the same
        # page alone must reproduce it exactly.
        second = runtime_view(render("Il bebe bd"))
        self.assertEqual((first.top, first.bottom), (second.top, second.bottom))
        np.testing.assert_array_equal(first.image, second.image)

    def test_runtime_views_of_a_pair_are_independent(self) -> None:
        """Each member is measured on its own; they need not agree."""
        bare = runtime_view(render("Il bebe bd"))
        accented = runtime_view(render("Il bebé bd"))
        self.assertEqual(bare.view, RUNTIME)
        self.assertEqual(accented.view, RUNTIME)
        # The accent may raise the ink extent; that is expected and honest.
        self.assertGreaterEqual(accented.ink_height, bare.ink_height)

    def test_blank_page_returns_none(self) -> None:
        self.assertIsNone(runtime_view(Image.new("RGB", (200, 40), (14, 14, 14))))


class TrainingGuardTests(unittest.TestCase):
    def test_runtime_view_is_accepted(self) -> None:
        view = runtime_view(render("bebe"))
        self.assertIs(assert_runtime_view(view), view)

    def test_audit_view_is_rejected_for_training(self) -> None:
        view = causal_audit_view(render("bebe"), "bebe", FONT, 20, 8.0)
        with self.assertRaises(ValueError):
            assert_runtime_view(view)

    def test_views_are_tagged(self) -> None:
        self.assertEqual(runtime_view(render("bebe")).view, RUNTIME)
        self.assertEqual(
            causal_audit_view(render("bebe"), "bebe", FONT, 20, 8.0).view,
            CAUSAL_AUDIT,
        )


class ConfigTests(unittest.TestCase):
    def test_config_is_serializable_and_frozen(self) -> None:
        config = LineGeometryConfig()
        self.assertIn("margin_ratio", config.as_dict())
        with self.assertRaises(Exception):
            config.margin_ratio = 0.5  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
