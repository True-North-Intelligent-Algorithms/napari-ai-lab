"""The Segment tab offers a profile's segmenters in the profile's order."""

import pytest

from napari_ai_lab.apps.register_all import _order_as_listed, register_all
from napari_ai_lab.Segmenters.GlobalSegmenters import GlobalSegmenterBase


class Yolo:
    pass


class FastSam:
    pass


class Other:
    pass


def test_listed_ones_come_first_in_list_order():
    registry = {"YOLO": Yolo, "Other": Other, "FastSAM": FastSam}
    _order_as_listed(registry, ("fastsam", "yolo"))
    assert list(registry) == ["FastSAM", "YOLO", "Other"]


def test_no_list_keeps_the_registration_order():
    registry = {"YOLO": Yolo, "FastSAM": FastSam}
    _order_as_listed(registry, None)
    assert list(registry) == ["YOLO", "FastSAM"]


def test_2d_detect_starts_on_fastsam():
    pytest.importorskip("skop")
    register_all("2d-detect")
    first = next(iter(GlobalSegmenterBase.registry.values()))
    assert first.__name__ == "FastSamSkopDetector"
