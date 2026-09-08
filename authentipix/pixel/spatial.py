"""Spatial gradient magnitude and standardized edge density extractor for Phase 2.

Implements 3x3 Sobel convolution with explicit reflect border padding,
normalized gradient magnitude scaling (max 1020*sqrt(2) mapped to 255.0),
and standardized edge density calculation (T_edge = 30.0).
"""

import math
from typing import Tuple
import numpy as np

from authentipix.pixel.schemas import SpatialDescriptors


def compute_sobel_gradients(img_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes horizontal (Gx) and vertical (Gy) Sobel derivatives with reflect padding.

    Kernels:
        Kx = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]
        Ky = [[-1, -2, -1], [ 0,  0,  0], [+1, +2, +1]]
    """
    # Ensure 2D float array in [0.0, 255.0] intensity scale
    if np.issubdtype(img_2d.dtype, np.floating):
        float_img = img_2d.astype(np.float64)
        if float_img.max() <= 1.0:
            float_img = float_img * 255.0
    else:
        float_img = img_2d.astype(np.float64)

    # Reflect padding by 1 pixel on all borders
    padded = np.pad(float_img, pad_width=1, mode="reflect")

    # Vectorized 3x3 Sobel filtering via slice shifts
    # Padded slices:
    # top_left = padded[:-2, :-2], top_mid = padded[:-2, 1:-1], top_right = padded[:-2, 2:]
    # mid_left = padded[1:-1, :-2], mid_mid = padded[1:-1, 1:-1], mid_right = padded[1:-1, 2:]
    # bot_left = padded[2:, :-2], bot_mid = padded[2:, 1:-1], bot_right = padded[2:, 2:]

    tl = padded[:-2, :-2]
    tm = padded[:-2, 1:-1]
    tr = padded[:-2, 2:]
    ml = padded[1:-1, :-2]
    mr = padded[1:-1, 2:]
    bl = padded[2:, :-2]
    bm = padded[2:, 1:-1]
    br = padded[2:, 2:]

    # Gx = (tr + 2*mr + br) - (tl + 2*ml + bl)
    gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)

    # Gy = (bl + 2*bm + br) - (tl + 2*tm + tr)
    gy = (bl + 2.0 * bm + br) - (tl + 2.0 * tm + tr)

    return gx, gy


def compute_normalized_sobel_map(img_2d: np.ndarray) -> np.ndarray:
    """Computes normalized Sobel gradient magnitude map g_norm in [0.0, 255.0].

    G_mag_norm = G_raw / (4 * sqrt(2)) mapped to [0.0, 255.0].
    """
    gx, gy = compute_sobel_gradients(img_2d)
    g_raw = np.sqrt(gx**2 + gy**2)
    norm_factor = 4.0 * math.sqrt(2.0)
    g_norm = g_raw / norm_factor
    return np.clip(g_norm, 0.0, 255.0)


def compute_spatial_descriptors(
    img_2d: np.ndarray, edge_threshold: float = 30.0
) -> SpatialDescriptors:
    """Calculates spatial gradient magnitude statistics and edge density.

    Normalized magnitude G_mag_norm = G_raw / (4 * sqrt(2)) mapped to [0.0, 255.0].
    Edge density D_edge = count(G_mag_norm > edge_threshold) / total_pixels.
    """
    g_norm = compute_normalized_sobel_map(img_2d)

    g_mean = float(np.mean(g_norm))
    g_std = float(np.std(g_norm))

    total_pixels = g_norm.size
    if total_pixels > 0:
        edge_pixels = int(np.sum(g_norm > edge_threshold))
        edge_density = float(edge_pixels) / float(total_pixels)
    else:
        edge_density = 0.0

    return SpatialDescriptors(
        gradient_mean=g_mean,
        gradient_std=g_std,
        edge_density=max(0.0, min(1.0, edge_density)),
        edge_threshold=edge_threshold,
    )

