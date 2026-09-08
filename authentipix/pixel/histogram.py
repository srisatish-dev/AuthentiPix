"""Histogram, Shannon entropy, clipping, and comb metric extractor for Phase 2.

Implements exact 256-bin bit-depth-normalized quantization (uint8, uint16, float32),
NaN/Inf handling, Shannon entropy calculation, clipping fractions, and the approved
Corrected Interior Histogram Comb Metric (C_comb = Z_int / L_interior).
"""

import math
from typing import List, Tuple
import numpy as np

from authentipix.pixel.schemas import HistogramFeatures, PixelProperties


def quantize_to_256_bins(data: np.ndarray) -> Tuple[List[int], int, int, int]:
    """Quantizes channel data into exactly 256 uniform bins (0-255).

    Returns:
        Tuple[List[int], int, int, int]: 
            - 256-bin count list
            - total valid pixel count
            - nan_pixel_count
            - inf_pixel_count
    """
    bins = [0] * 256
    nan_count = 0
    inf_count = 0

    flat_data = data.ravel()
    total_len = len(flat_data)

    if total_len == 0:
        return bins, 0, 0, 0

    if np.issubdtype(data.dtype, np.floating):
        # Handle Float data
        nan_mask = np.isnan(flat_data)
        inf_mask = np.isinf(flat_data)
        nan_count = int(nan_mask.sum())
        inf_count = int(inf_mask.sum())

        valid_mask = ~(nan_mask | inf_mask)
        valid_floats = flat_data[valid_mask]
        valid_count = len(valid_floats) + inf_count

        # Process finite floats
        if len(valid_floats) > 0:
            # If float values are in [0.0, 255.0] range (e.g. Luma Y'), scale by 255.0
            if float(valid_floats.max()) > 1.0:
                norm_floats = valid_floats / 255.0
            else:
                norm_floats = valid_floats
            clamped = np.clip(norm_floats, 0.0, 1.0)
            indices = np.floor(clamped * 256.0).astype(np.int64)
            indices = np.minimum(indices, 255)

            bin_counts = np.bincount(indices, minlength=256)
            for i in range(256):
                bins[i] += int(bin_counts[i])

        # Process Inf values: -Inf -> bin 0, +Inf -> bin 255
        if inf_count > 0:
            pos_inf_count = int((flat_data == np.inf).sum())
            neg_inf_count = int((flat_data == -np.inf).sum())
            bins[0] += neg_inf_count
            bins[255] += pos_inf_count

        return bins, valid_count, nan_count, inf_count

    elif np.issubdtype(data.dtype, np.unsignedinteger) and data.dtype.itemsize == 2:
        # uint16 data: bin = floor(val / 256) clamped to 255
        indices = (flat_data >> 8).astype(np.int64)
        indices = np.minimum(indices, 255)
        bin_counts = np.bincount(indices, minlength=256)
        bins = [int(x) for x in bin_counts[:256]]
        return bins, total_len, 0, 0

    elif np.issubdtype(data.dtype, np.integer) and data.dtype.itemsize > 1:
        # Integer data with size > 1 byte (int16, int32)
        # Scaled dynamically to [0, 255]
        min_val = float(flat_data.min())
        max_val = float(flat_data.max())
        if max_val == min_val:
            idx = 0 if min_val <= 0 else (255 if min_val >= 255 else int(min_val))
            bins[idx] = total_len
        else:
            norm = (flat_data.astype(np.float64) - min_val) / (max_val - min_val)
            indices = np.floor(norm * 256.0).astype(np.int64)
            indices = np.minimum(np.maximum(indices, 0), 255)
            bin_counts = np.bincount(indices, minlength=256)
            bins = [int(x) for x in bin_counts[:256]]
        return bins, total_len, 0, 0

    else:
        # uint8 / 8-bit data: direct bin indexing
        indices = np.clip(flat_data, 0, 255).astype(np.int64)
        bin_counts = np.bincount(indices, minlength=256)
        bins = [int(x) for x in bin_counts[:256]]
        return bins, total_len, 0, 0


def compute_shannon_entropy(bins: List[int], total_pixels: int) -> float:
    """Calculates Shannon entropy H = -sum p(i) log2(p(i)) in bits per pixel [0.0, 8.0]."""
    if total_pixels <= 0:
        return 0.0

    entropy = 0.0
    for count in bins:
        if count > 0:
            p = float(count) / float(total_pixels)
            entropy -= p * math.log2(p)

    return max(0.0, min(8.0, float(entropy)))


def compute_comb_metric(bins: List[int]) -> float:
    """Calculates Corrected Interior Histogram Comb Metric: C_comb = Z_int / L_interior.

    L_interior = i_max - i_min - 1 (for i_max - i_min > 1).
    Z_int = number of zero bins between i_min and i_max (exclusive).
    Returns 0.0 if i_max - i_min <= 1 or no active bins.
    Guarantees 0.0 <= C_comb <= 1.0.
    """
    active_indices = [i for i, count in enumerate(bins) if count > 0]
    if len(active_indices) <= 1:
        return 0.0

    i_min = active_indices[0]
    i_max = active_indices[-1]
    span = i_max - i_min

    if span <= 1:
        return 0.0

    l_interior = span - 1
    z_int = 0
    for i in range(i_min + 1, i_max):
        if bins[i] == 0:
            z_int += 1

    c_comb = float(z_int) / float(l_interior)
    return max(0.0, min(1.0, c_comb))


def compute_histogram_features(channel_data: np.ndarray, channel_name: str) -> HistogramFeatures:
    """Extracts 256-bin histogram, Shannon entropy, clipping fractions, and comb metric."""
    bins, total_valid, _, _ = quantize_to_256_bins(channel_data)

    entropy = compute_shannon_entropy(bins, total_valid)
    occupied_bins = sum(1 for b in bins if b > 0)

    clip_0 = float(bins[0]) / float(total_valid) if total_valid > 0 else 0.0
    clip_255 = float(bins[255]) / float(total_valid) if total_valid > 0 else 0.0
    comb_metric = compute_comb_metric(bins)

    return HistogramFeatures(
        channel_name=channel_name,
        bins=bins,
        entropy=entropy,
        occupied_bin_count=occupied_bins,
        clipping_fraction_0=clip_0,
        clipping_fraction_255=clip_255,
        comb_metric=comb_metric,
    )
