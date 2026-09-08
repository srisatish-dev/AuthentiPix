"""Parametric, non-parametric, and photometric Luma statistics extractor for Phase 2.

Calculates channel-wise min, max, mean, median, variance, std dev, percentiles,
sRGB BT.709 Luma (alpha excluded), and unweighted RGB intensity.
"""

from typing import List, Optional, Tuple
import numpy as np

from authentipix.pixel.schemas import ChannelStatistics, PixelProperties


def compute_channel_statistics(
    channel_data: np.ndarray, channel_idx: int, channel_name: str
) -> ChannelStatistics:
    """Calculates parametric and non-parametric statistics for a 1D or 2D intensity array."""
    # Filter non-finite values (NaN / Inf) if present
    if np.issubdtype(channel_data.dtype, np.floating):
        finite_mask = np.isfinite(channel_data)
        if not np.any(finite_mask):
            return ChannelStatistics(
                channel_index=channel_idx,
                channel_name=channel_name,
                min=0.0,
                max=0.0,
                mean=0.0,
                median=0.0,
                variance=0.0,
                std_dev=0.0,
                p5=0.0,
                p25=0.0,
                p75=0.0,
                p95=0.0,
                dynamic_range=0.0,
            )
        valid_data = channel_data[finite_mask]
    else:
        valid_data = channel_data.ravel()

    c_min = float(np.min(valid_data))
    c_max = float(np.max(valid_data))
    c_mean = float(np.mean(valid_data))
    c_median = float(np.median(valid_data))
    c_var = float(np.var(valid_data))
    c_std = float(np.std(valid_data))
    p5 = float(np.percentile(valid_data, 5))
    p25 = float(np.percentile(valid_data, 25))
    p75 = float(np.percentile(valid_data, 75))
    p95 = float(np.percentile(valid_data, 95))
    dyn_range = float(c_max - c_min)

    return ChannelStatistics(
        channel_index=channel_idx,
        channel_name=channel_name,
        min=c_min,
        max=c_max,
        mean=c_mean,
        median=c_median,
        variance=c_var,
        std_dev=c_std,
        p5=p5,
        p25=p25,
        p75=p75,
        p95=p95,
        dynamic_range=dyn_range,
    )


def compute_bt709_luma(arr: np.ndarray) -> Optional[np.ndarray]:
    """Computes non-linear sRGB BT.709 Luma: Y' = 0.2126 R' + 0.7152 G' + 0.0722 B'.

    Alpha channel (if present in RGBA) is strictly excluded.
    Returns None if arr is single-channel grayscale.
    """
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None

    r = arr[:, :, 0].astype(np.float64)
    g = arr[:, :, 1].astype(np.float64)
    b = arr[:, :, 2].astype(np.float64)

    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return luma


def compute_rgb_intensity(arr: np.ndarray) -> Optional[np.ndarray]:
    """Computes unweighted RGB intensity: I = (R' + G' + B') / 3.

    Alpha channel (if present in RGBA) is strictly excluded.
    Returns None if arr is single-channel grayscale.
    """
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None

    r = arr[:, :, 0].astype(np.float64)
    g = arr[:, :, 1].astype(np.float64)
    b = arr[:, :, 2].astype(np.float64)

    intensity = (r + g + b) / 3.0
    return intensity


def extract_all_statistics(
    arr: np.ndarray, props: PixelProperties
) -> Tuple[List[ChannelStatistics], Optional[ChannelStatistics], Optional[ChannelStatistics]]:
    """Extracts channel statistics, BT.709 Luma statistics, and unweighted Intensity statistics."""
    channel_stats: List[ChannelStatistics] = []
    C = props.channel_count

    # Define channel names based on native mode and channel count
    if C == 3:
        names = ["red", "green", "blue"]
    elif C == 4:
        names = ["red", "green", "blue", "alpha"]
    elif C == 1:
        names = ["gray"]
    else:
        names = [f"channel_{i}" for i in range(C)]

    for i in range(C):
        c_data = arr[:, :, i]
        c_name = names[i] if i < len(names) else f"channel_{i}"
        stats = compute_channel_statistics(c_data, i, c_name)
        channel_stats.append(stats)

    # Compute BT.709 Luma statistics if color image
    luma_stats: Optional[ChannelStatistics] = None
    luma_arr = compute_bt709_luma(arr)
    if luma_arr is not None:
        luma_stats = compute_channel_statistics(luma_arr, -1, "luma")

    # Compute Intensity statistics if color image
    intensity_stats: Optional[ChannelStatistics] = None
    intensity_arr = compute_rgb_intensity(arr)
    if intensity_arr is not None:
        intensity_stats = compute_channel_statistics(intensity_arr, -1, "intensity")

    return channel_stats, luma_stats, intensity_stats
