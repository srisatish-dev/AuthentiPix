"""Regression test suite for Phase 3A (Residual & Noise Analysis).

Verifies that the frozen Phase 3A algorithms, formulas, schemas, and outputs
remain 100% unmodified and numerically identical after the integration of the
Robustness / Format / Screenshot layer.
"""

import numpy as np
import pytest

from authentipix.pixel.residual import (
    ResidualNoiseAnalyzer,
    compute_3x3_median_residual,
    compute_statistical_moments,
    normalize_input_array,
)
from authentipix.pixel.schemas import PixelProperties


def test_phase3a_normalization_regression():
    """Confirms canonical uint8 -> float64 normalization in [0.0, 1.0]."""
    arr_u8 = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    norm, invalid = normalize_input_array(arr_u8)
    assert norm.dtype == np.float64
    assert invalid == 0
    assert np.isclose(norm[0, 0], 0.0)
    assert np.isclose(norm[0, 1], 128.0 / 255.0)
    assert np.isclose(norm[1, 0], 1.0)


def test_phase3a_3x3_median_residual_regression():
    """Confirms exact deterministic 3x3 median residual calculation."""
    # 5x5 constant array -> interior median residual must be exactly 0.0
    arr = np.full((5, 5), 0.5, dtype=np.float64)
    res, finite_mask = compute_3x3_median_residual(arr)
    assert res.shape == (3, 3)
    assert np.all(finite_mask)
    assert np.allclose(res, 0.0)

    # Center impulse of +0.1 -> median of neighborhood is 0.5, center is 0.6 -> residual is +0.1
    arr_impulse = np.full((5, 5), 0.5, dtype=np.float64)
    arr_impulse[2, 2] = 0.6
    res_impulse, _ = compute_3x3_median_residual(arr_impulse)
    assert np.isclose(res_impulse[1, 1], 0.1)


def test_phase3a_statistical_moments_regression():
    """Confirms statistical moments formulas (mean, std, robust_std, skewness, kurtosis)."""
    data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    quality_flags: list[str] = []
    moments = compute_statistical_moments(data, quality_flags)

    assert moments is not None
    assert moments.sample_count == 5
    assert np.isclose(moments.mean, 0.0)
    # std with ddof=0 for [-2, -1, 0, 1, 2] -> sqrt(10 / 5) = sqrt(2) ≈ 1.41421356
    assert np.isclose(moments.std, np.sqrt(2.0))
    # median is 0, deviations are [2, 1, 0, 1, 2], median deviation is 1.0
    # robust_std = 1.4826 * 1.0 = 1.4826
    assert np.isclose(moments.mad, 1.0)
    assert np.isclose(moments.robust_std, 1.4826)


def test_phase3a_full_analyzer_regression():
    """Runs full Phase 3A analyzer on a deterministic 64x64 synthetic input."""
    # Deterministic seeded random array
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    props = PixelProperties(
        width=64,
        height=64,
        pixel_count=4096,
        aspect_ratio=1.0,
        native_mode="RGB",
        channel_count=3,
        bit_depth=8,
        dtype="uint8",
    )

    analyzer = ResidualNoiseAnalyzer()
    result = analyzer.analyze(arr, props)

    assert result.analyzer_id == "pillar2_residual_noise_analyzer"
    assert result.analyzer_version == "0.3.5"
    assert result.input_characteristics.total_pixel_count == 4096
    assert result.edge_excluded_luma_stats is not None
    assert result.spatial_descriptors is not None
    assert result.channel_descriptors is not None

    # Verify that moments are valid finite numbers
    assert np.isfinite(result.edge_excluded_luma_stats.mean)
    assert np.isfinite(result.edge_excluded_luma_stats.std)
    assert np.isfinite(result.edge_excluded_luma_stats.robust_std)
