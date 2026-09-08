"""Unit tests for Phase 2 channel statistics, BT.709 Luma, and Intensity (authentipix.pixel.statistics)."""

import pytest
import numpy as np

from authentipix.pixel.schemas import PixelProperties
from authentipix.pixel.statistics import (
    compute_channel_statistics,
    compute_bt709_luma,
    compute_rgb_intensity,
    extract_all_statistics,
)


def test_solid_image_statistics():
    # 10x10 array of solid 128
    data = np.full((10, 10), 128, dtype=np.uint8)
    stats = compute_channel_statistics(data, 0, "red")

    assert stats.min == 128.0
    assert stats.max == 128.0
    assert stats.mean == 128.0
    assert stats.median == 128.0
    assert stats.variance == 0.0
    assert stats.std_dev == 0.0
    assert stats.p5 == 128.0
    assert stats.p95 == 128.0
    assert stats.dynamic_range == 0.0


def test_known_gradient_statistics():
    # 1x5 array [0, 50, 100, 150, 200]
    data = np.array([[0, 50, 100, 150, 200]], dtype=np.uint8)
    stats = compute_channel_statistics(data, 0, "test")

    assert stats.min == 0.0
    assert stats.max == 200.0
    assert stats.mean == 100.0
    assert stats.median == 100.0
    assert stats.dynamic_range == 200.0


def test_bt709_luma_calculation():
    # Pure Red (255, 0, 0), Pure Green (0, 255, 0), Pure Blue (0, 0, 255)
    arr = np.zeros((1, 3, 3), dtype=np.uint8)
    arr[0, 0] = [255, 0, 0]
    arr[0, 1] = [0, 255, 0]
    arr[0, 2] = [0, 0, 255]

    luma = compute_bt709_luma(arr)
    assert luma is not None
    assert luma.shape == (1, 3)

    # Pure Red Luma = 0.2126 * 255 = 54.213
    assert pytest.approx(luma[0, 0], abs=1e-3) == 54.213
    # Pure Green Luma = 0.7152 * 255 = 182.376
    assert pytest.approx(luma[0, 1], abs=1e-3) == 182.376
    # Pure Blue Luma = 0.0722 * 255 = 18.411
    assert pytest.approx(luma[0, 2], abs=1e-3) == 18.411


def test_rgba_alpha_isolation():
    # 1x1 RGBA pixel: (255, 0, 0, 128) - Red with 50% alpha
    arr = np.array([[[255, 0, 0, 128]]], dtype=np.uint8)
    props = PixelProperties(
        width=1,
        height=1,
        pixel_count=1,
        aspect_ratio=1.0,
        native_mode="RGBA",
        channel_count=4,
        bit_depth=8,
        dtype="uint8",
        has_alpha=True,
    )

    channel_stats, luma_stats, intensity_stats = extract_all_statistics(arr, props)

    assert len(channel_stats) == 4
    assert channel_stats[0].channel_name == "red"
    assert channel_stats[0].mean == 255.0
    assert channel_stats[3].channel_name == "alpha"
    assert channel_stats[3].mean == 128.0

    # Luma must ONLY use R, G, B (Alpha is excluded)
    assert luma_stats is not None
    assert pytest.approx(luma_stats.mean, abs=1e-3) == 54.213

    # Intensity must ONLY use R, G, B: (255 + 0 + 0) / 3 = 85.0
    assert intensity_stats is not None
    assert intensity_stats.mean == 85.0


def test_grayscale_no_luma():
    arr = np.array([[[128]], [[200]]], dtype=np.uint8)
    props = PixelProperties(
        width=1,
        height=2,
        pixel_count=2,
        aspect_ratio=0.5,
        native_mode="L",
        channel_count=1,
        bit_depth=8,
        dtype="uint8",
    )

    channel_stats, luma_stats, intensity_stats = extract_all_statistics(arr, props)

    assert len(channel_stats) == 1
    assert channel_stats[0].channel_name == "gray"
    assert luma_stats is None
    assert intensity_stats is None
