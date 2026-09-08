"""Unit tests for Phase 2 histograms, Shannon entropy, clipping, and comb metrics (authentipix.pixel.histogram)."""

import pytest
import numpy as np

from authentipix.pixel.histogram import (
    quantize_to_256_bins,
    compute_shannon_entropy,
    compute_comb_metric,
    compute_histogram_features,
)


def test_solid_black_histogram():
    data = np.zeros((10, 10), dtype=np.uint8)
    feat = compute_histogram_features(data, "gray")

    assert feat.bins[0] == 100
    assert sum(feat.bins[1:]) == 0
    assert feat.entropy == 0.0
    assert feat.occupied_bin_count == 1
    assert feat.clipping_fraction_0 == 1.0
    assert feat.clipping_fraction_255 == 0.0
    assert feat.comb_metric == 0.0


def test_solid_white_histogram():
    data = np.full((10, 10), 255, dtype=np.uint8)
    feat = compute_histogram_features(data, "gray")

    assert feat.bins[255] == 100
    assert sum(feat.bins[:255]) == 0
    assert feat.entropy == 0.0
    assert feat.occupied_bin_count == 1
    assert feat.clipping_fraction_0 == 0.0
    assert feat.clipping_fraction_255 == 1.0
    assert feat.comb_metric == 0.0


def test_uniform_distribution_entropy():
    # Exactly one pixel for each intensity 0..255
    data = np.arange(256, dtype=np.uint8)
    feat = compute_histogram_features(data, "test")

    assert feat.occupied_bin_count == 256
    # Log2(256) = 8.0 bits per pixel
    assert pytest.approx(feat.entropy, abs=1e-5) == 8.0
    assert feat.comb_metric == 0.0


def test_float_quantization_bins():
    # Floats: 0.0, 0.5, 1.0
    data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    bins, total, nan_c, inf_c = quantize_to_256_bins(data)

    assert total == 3
    assert bins[0] == 1      # 0.0 -> bin 0
    assert bins[128] == 1    # 0.5 -> bin 128
    assert bins[255] == 1    # 1.0 -> bin 255


def test_float_nan_inf_handling():
    data = np.array([0.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    bins, total, nan_c, inf_c = quantize_to_256_bins(data)

    assert nan_c == 1
    assert inf_c == 2
    assert bins[0] == 2      # 0.0 + -Inf -> bin 0
    assert bins[255] == 1    # +Inf -> bin 255


def test_comb_metric_calculation():
    # Case 1: Adjacent bins only [10, 11] -> no interior bins -> 0.0
    bins1 = [0] * 256
    bins1[10] = 5
    bins1[11] = 5
    assert compute_comb_metric(bins1) == 0.0

    # Case 2: Bins [10, 12], bin 11 is zero -> i_min=10, i_max=12, L_int=1, Z_int=1 -> 1.0
    bins2 = [0] * 256
    bins2[10] = 5
    bins2[12] = 5
    assert compute_comb_metric(bins2) == 1.0

    # Case 3: Alternating bins [10, 0, 12, 0, 14] -> L_int=3, Z_int=2 (bins 11, 13) -> 2/3
    bins3 = [0] * 256
    bins3[10] = 5
    bins3[12] = 5
    bins3[14] = 5
    assert pytest.approx(compute_comb_metric(bins3), abs=1e-4) == 2.0 / 3.0

    # Guarantee range [0.0, 1.0]
    assert 0.0 <= compute_comb_metric(bins3) <= 1.0
