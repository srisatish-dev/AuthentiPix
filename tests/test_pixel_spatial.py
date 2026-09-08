"""Unit tests for Phase 2 spatial Sobel gradient magnitude and edge density (authentipix.pixel.spatial)."""

import pytest
import numpy as np

from authentipix.pixel.spatial import compute_sobel_gradients, compute_spatial_descriptors


def test_solid_array_zero_gradients():
    # 20x20 array of constant 100
    img = np.full((20, 20), 100, dtype=np.uint8)
    gx, gy = compute_sobel_gradients(img)

    assert np.all(gx == 0.0)
    assert np.all(gy == 0.0)

    desc = compute_spatial_descriptors(img)
    assert desc.gradient_mean == 0.0
    assert desc.gradient_std == 0.0
    assert desc.edge_density == 0.0


def test_horizontal_step_edge():
    # 20x20 image: top 10 rows 0, bottom 10 rows 255
    img = np.zeros((20, 20), dtype=np.uint8)
    img[10:, :] = 255

    gx, gy = compute_sobel_gradients(img)
    assert gx.shape == (20, 20)
    assert gy.shape == (20, 20)

    # Vertical derivative Gy should be strong at the edge (row 9/10 boundary)
    assert np.max(np.abs(gy)) > 0.0

    desc = compute_spatial_descriptors(img, edge_threshold=30.0)
    assert desc.gradient_mean > 0.0
    assert desc.edge_density > 0.0
    assert 0.0 <= desc.edge_density <= 1.0


def test_vertical_step_edge():
    # 20x20 image: left 10 cols 0, right 10 cols 255
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, 10:] = 255

    gx, gy = compute_sobel_gradients(img)

    # Horizontal derivative Gx should be strong at the edge
    assert np.max(np.abs(gx)) > 0.0

    desc = compute_spatial_descriptors(img, edge_threshold=30.0)
    assert desc.gradient_mean > 0.0
    assert desc.edge_density > 0.0


def test_block_checkerboard_high_edge_density():
    # 20x20 image with 2x2 blocks of 0 and 255
    img = np.zeros((20, 20), dtype=np.uint8)
    # Create 2x2 block grid
    for r in range(20):
        for c in range(20):
            if ((r // 2) + (c // 2)) % 2 == 0:
                img[r, c] = 255

    desc = compute_spatial_descriptors(img, edge_threshold=30.0)
    assert desc.gradient_mean > 20.0
    assert desc.edge_density > 0.3
