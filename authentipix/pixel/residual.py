"""Phase 3A: High-Frequency Residual & Noise-Characteristic Analyzer.

Extracts, characterizes, and evaluates high-frequency 3x3 median residuals,
Sobel edge-excluded residual properties, spatial block heterogeneity,
lag-1 autocorrelation, and multi-channel residual correlations.
"""

from typing import List, Optional, Tuple
import numpy as np

from authentipix.metadata.context import AnalysisContext
from authentipix.pixel.loader import PixelImageLoader
from authentipix.pixel.schemas import (
    ChannelResidualDescriptors,
    InputCharacteristics,
    PixelProperties,
    ResidualNoiseAnalysisResult,
    ResidualStatisticalMoments,
    SpatialResidualDescriptors,
)
from authentipix.pixel.spatial import compute_normalized_sobel_map


# ============================================================================
# Phase 3A Error Hierarchy
# ============================================================================

class ResidualAnalysisError(Exception):
    """Base exception for Phase 3A residual analysis errors."""
    pass


class UnsupportedDtypeError(ResidualAnalysisError):
    """Raised when an input array has an unsupported NumPy dtype."""
    pass


class FloatRangeViolationError(ResidualAnalysisError):
    """Raised when a floating-point input array contains finite values outside [0.0, 1.0]."""
    pass


class SmallImageError(ResidualAnalysisError):
    """Raised when image dimensions are smaller than 32x32."""
    pass


class SobelIntegrationError(ResidualAnalysisError):
    """Raised when Phase 2 Sobel map dimensions do not match the source image."""
    pass


# ============================================================================
# Authoritative Forensic Limitations
# ============================================================================

PHASE_3A_LIMITATIONS: List[str] = [
    "Residual analysis does not independently prove AI generation, camera capture, or malicious manipulation.",
    "The residual is a high-frequency residual proxy derived from the raster. It is not a direct measurement of physical sensor noise, photon noise, read noise, PRNU, or any other specific acquisition mechanism.",
    "Post-processing operations such as re-compression, downsampling, denoising, or social media re-encoding substantially modify high-frequency residual distributions.",
    "Edge-excluded residual characterization assumes valid homogeneous regions exist; high-frequency textures may suppress valid sample coverage.",
]


# ============================================================================
# Helper Numerical Functions
# ============================================================================

def normalize_input_array(arr: np.ndarray) -> Tuple[np.ndarray, int]:
    """Validates and normalizes input array to canonical float64 in [0.0, 1.0].

    Returns:
        Tuple[np.ndarray, int]: (arr_float64, input_invalid_float_count)
    """
    dtype = arr.dtype
    if dtype == np.uint8:
        arr_float = arr.astype(np.float64) / 255.0
    elif dtype == np.uint16:
        arr_float = arr.astype(np.float64) / 65535.0
    elif dtype == np.uint32:
        arr_float = arr.astype(np.float64) / 4294967295.0
    elif dtype in [np.float32, np.float64]:
        # Validate finite float range [0.0, 1.0]
        finite_mask = np.isfinite(arr)
        if np.any(finite_mask):
            min_val = float(np.min(arr[finite_mask]))
            max_val = float(np.max(arr[finite_mask]))
            if min_val < 0.0 or max_val > 1.0:
                raise FloatRangeViolationError(
                    f"Finite float values must lie in [0.0, 1.0], observed min={min_val}, max={max_val}"
                )
        arr_float = arr.astype(np.float64)
    else:
        raise UnsupportedDtypeError(f"Unsupported input array dtype: {dtype}")

    # Count non-finite scalar values after float64 conversion
    input_invalid_float_count = int(np.sum(~np.isfinite(arr_float)))
    return arr_float, input_invalid_float_count


def compute_3x3_median_residual(chan_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes (H-2, W-2) 3x3 median residual and 3x3 finite neighborhood validity mask.

    A position (r, c) is finite_3x3 iff all 9 pixels in its 3x3 neighborhood are finite.
    For non-finite positions, residual is NaN and finite_3x3 is False.
    """
    H, W = chan_2d.shape
    w0 = chan_2d[0 : H - 2, 0 : W - 2]
    w1 = chan_2d[0 : H - 2, 1 : W - 1]
    w2 = chan_2d[0 : H - 2, 2 : W]
    w3 = chan_2d[1 : H - 1, 0 : W - 2]
    w4 = chan_2d[1 : H - 1, 1 : W - 1]  # center
    w5 = chan_2d[1 : H - 1, 2 : W]
    w6 = chan_2d[2 : H, 0 : W - 2]
    w7 = chan_2d[2 : H, 1 : W - 1]
    w8 = chan_2d[2 : H, 2 : W]

    stacked = np.stack([w0, w1, w2, w3, w4, w5, w6, w7, w8], axis=-1)
    finite_3x3 = np.all(np.isfinite(stacked), axis=-1)

    residual = np.full((H - 2, W - 2), np.nan, dtype=np.float64)
    if np.any(finite_3x3):
        valid_stacked = stacked[finite_3x3]
        valid_medians = np.median(valid_stacked, axis=-1)
        residual[finite_3x3] = w4[finite_3x3] - valid_medians

    return residual, finite_3x3


def compute_statistical_moments(
    samples: np.ndarray,
    quality_flags: List[str],
) -> Optional[ResidualStatisticalMoments]:
    """Computes statistical moments over a 1D array of valid residual samples."""
    N = int(samples.size)
    if N == 0:
        return None

    mean_val = float(np.mean(samples))
    std_val = float(np.std(samples))  # ddof=0
    mad_val = float(np.median(np.abs(samples - np.median(samples))))
    robust_std_val = float(1.4826 * mad_val)
    rms_val = float(np.sqrt(np.mean(samples**2)))

    # Quality flag checks
    if 1 <= N < 32:
        if "LOW_SAMPLE_COUNT" not in quality_flags:
            quality_flags.append("LOW_SAMPLE_COUNT")

    if std_val <= 1e-12:
        if "ZERO_OR_NEGLIGIBLE_RESIDUAL_SCALE" not in quality_flags:
            quality_flags.append("ZERO_OR_NEGLIGIBLE_RESIDUAL_SCALE")

    # Higher moments
    if N < 8 or std_val <= 1e-12:
        skewness_val = None
        kurtosis_val = None
    else:
        diff = samples - mean_val
        m3 = float(np.mean(diff**3))
        m4 = float(np.mean(diff**4))
        skewness_val = float(m3 / (std_val**3))
        kurtosis_val = float(m4 / (std_val**4) - 3.0)

    return ResidualStatisticalMoments(
        mean=mean_val,
        std=std_val,
        robust_std=robust_std_val,
        mad=mad_val,
        rms=rms_val,
        skewness=skewness_val,
        kurtosis=kurtosis_val,
        sample_count=N,
    )


def compute_spatial_residual_descriptors(
    res_Y: np.ndarray,
    V_Y: np.ndarray,
    quality_flags: List[str],
) -> SpatialResidualDescriptors:
    """Computes 16x16 block spatial heterogeneity metrics and lag-1 autocorrelation."""
    H_int, W_int = res_Y.shape
    n_r = H_int // 16
    n_c = W_int // 16
    evaluated_block_count = n_r * n_c

    valid_block_variances: List[float] = []
    for r in range(n_r):
        for c in range(n_c):
            b_res = res_Y[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16]
            b_mask = V_Y[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16]
            n_valid = int(np.sum(b_mask))
            if n_valid >= 32:
                b_samples = b_res[b_mask]
                b_var = float(np.var(b_samples))  # ddof=0
                valid_block_variances.append(b_var)

    valid_block_count = len(valid_block_variances)

    if valid_block_count < 4:
        local_variance_mean = None
        local_variance_std = None
        spatial_heterogeneity_index = None
        if "INSUFFICIENT_VALID_BLOCKS" not in quality_flags:
            quality_flags.append("INSUFFICIENT_VALID_BLOCKS")
    else:
        var_arr = np.array(valid_block_variances, dtype=np.float64)
        local_variance_mean = float(np.mean(var_arr))
        local_variance_std = float(np.std(var_arr))  # ddof=0
        if local_variance_mean > 1e-12:
            spatial_heterogeneity_index = float(local_variance_std / local_variance_mean)
        else:
            spatial_heterogeneity_index = None

    # Autocorrelation (evaluated independently over V_Y)
    # Horizontal lag-1 pairs
    mask_h = V_Y[:, :-1] & V_Y[:, 1:]
    n_h = int(np.sum(mask_h))
    if n_h >= 32:
        x_h = res_Y[:, :-1][mask_h]
        y_h = res_Y[:, 1:][mask_h]
        mu_x = float(np.mean(x_h))
        mu_y = float(np.mean(y_h))
        std_x = float(np.std(x_h))
        std_y = float(np.std(y_h))
        if std_x > 1e-12 and std_y > 1e-12:
            cov_h = float(np.mean((x_h - mu_x) * (y_h - mu_y)))
            horizontal_lag1_autocorrelation = float(cov_h / (std_x * std_y))
        else:
            horizontal_lag1_autocorrelation = None
    else:
        horizontal_lag1_autocorrelation = None

    # Vertical lag-1 pairs
    mask_v = V_Y[:-1, :] & V_Y[1:, :]
    n_v = int(np.sum(mask_v))
    if n_v >= 32:
        x_v = res_Y[:-1, :][mask_v]
        y_v = res_Y[1:, :][mask_v]
        mu_x = float(np.mean(x_v))
        mu_y = float(np.mean(y_v))
        std_x = float(np.std(x_v))
        std_y = float(np.std(y_v))
        if std_x > 1e-12 and std_y > 1e-12:
            cov_v = float(np.mean((x_v - mu_x) * (y_v - mu_y)))
            vertical_lag1_autocorrelation = float(cov_v / (std_x * std_y))
        else:
            vertical_lag1_autocorrelation = None
    else:
        vertical_lag1_autocorrelation = None

    return SpatialResidualDescriptors(
        evaluated_block_count=evaluated_block_count,
        valid_block_count=valid_block_count,
        local_variance_mean=local_variance_mean,
        local_variance_std=local_variance_std,
        spatial_heterogeneity_index=spatial_heterogeneity_index,
        horizontal_lag1_autocorrelation=horizontal_lag1_autocorrelation,
        vertical_lag1_autocorrelation=vertical_lag1_autocorrelation,
    )


def compute_channel_residual_descriptors(
    res_R: np.ndarray,
    res_G: np.ndarray,
    res_B: np.ndarray,
    V_R: np.ndarray,
    V_G: np.ndarray,
    V_B: np.ndarray,
    quality_flags: List[str],
) -> ChannelResidualDescriptors:
    """Computes color channel moments, cross-channel Pearson correlations, and scale ratios."""
    red_stats = compute_statistical_moments(res_R[V_R], quality_flags)
    green_stats = compute_statistical_moments(res_G[V_G], quality_flags)
    blue_stats = compute_statistical_moments(res_B[V_B], quality_flags)

    def _calc_corr(
        r1: np.ndarray, r2: np.ndarray, v1: np.ndarray, v2: np.ndarray
    ) -> Optional[float]:
        v_pair = v1 & v2
        n_pair = int(np.sum(v_pair))
        if n_pair < 32:
            return None
        x = r1[v_pair]
        y = r2[v_pair]
        std_x = float(np.std(x))
        std_y = float(np.std(y))
        if std_x <= 1e-12 or std_y <= 1e-12:
            return None
        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        cov = float(np.mean((x - mu_x) * (y - mu_y)))
        return float(cov / (std_x * std_y))

    rg_corr = _calc_corr(res_R, res_G, V_R, V_G)
    rb_corr = _calc_corr(res_R, res_B, V_R, V_B)
    gb_corr = _calc_corr(res_G, res_B, V_G, V_B)

    # Scale Ratios (governed strictly by green_stats.robust_std)
    if green_stats is None or green_stats.robust_std <= 1e-12:
        blue_to_green_ratio = None
        red_to_green_ratio = None
        if "UNDEFINED_CHANNEL_RATIO" not in quality_flags:
            quality_flags.append("UNDEFINED_CHANNEL_RATIO")
    else:
        g_robust = green_stats.robust_std
        blue_to_green_ratio = (
            float(blue_stats.robust_std / g_robust) if blue_stats is not None else None
        )
        red_to_green_ratio = (
            float(red_stats.robust_std / g_robust) if red_stats is not None else None
        )

    return ChannelResidualDescriptors(
        red_stats=red_stats,
        green_stats=green_stats,
        blue_stats=blue_stats,
        rg_correlation=rg_corr,
        rb_correlation=rb_corr,
        gb_correlation=gb_corr,
        blue_to_green_ratio=blue_to_green_ratio,
        red_to_green_ratio=red_to_green_ratio,
    )


# ============================================================================
# Phase 3A Main Analyzer Class
# ============================================================================

class ResidualNoiseAnalyzer:
    """Deterministic high-frequency residual and noise-characteristic analyzer for Phase 3A."""

    def __init__(self, loader: Optional[PixelImageLoader] = None):
        self.loader = loader or PixelImageLoader()

    def analyze_context(self, ctx: AnalysisContext) -> ResidualNoiseAnalysisResult:
        """Loads context asset via PixelImageLoader and analyzes residual characteristics."""
        arr, props = self.loader.load_pixel_array(ctx)
        return self.analyze(arr, props)

    def analyze(
        self,
        arr: np.ndarray,
        props: Optional[PixelProperties] = None,
    ) -> ResidualNoiseAnalysisResult:
        """Executes Phase 3A Residual & Noise-Characteristic Analysis on a NumPy image array."""
        # 1. Dimensions check
        if arr.ndim == 2:
            H, W = arr.shape
            C = 1
        elif arr.ndim == 3:
            H, W, C = arr.shape
        else:
            raise UnsupportedDtypeError(f"Unsupported array dimensions: ndim={arr.ndim}")

        if H < 32 or W < 32:
            raise SmallImageError(f"Image dimensions ({W}x{H}) must be at least 32x32")

        total_pixel_count = W * H
        quality_flags: List[str] = []

        # 2. Dtype validation & canonical float64 [0.0, 1.0] conversion
        arr_float, input_invalid_float_count = normalize_input_array(arr)

        if input_invalid_float_count > 0:
            quality_flags.append("INVALID_FLOAT_VALUES")

        # 3. Acquire Phase 2 Sobel gradient map g_norm
        if C >= 3:
            # Canonical BT.709 Luma on [0, 255] scale for Phase 2 Sobel
            r = arr_float[:, :, 0]
            g = arr_float[:, :, 1]
            b = arr_float[:, :, 2]
            sobel_source = (0.2126 * r + 0.7152 * g + 0.0722 * b) * 255.0
        else:
            sobel_source = (arr_float[:, :, 0] if arr_float.ndim == 3 else arr_float) * 255.0

        g_norm = compute_normalized_sobel_map(sobel_source)
        if g_norm.shape != (H, W):
            raise SobelIntegrationError(
                f"Sobel map shape {g_norm.shape} does not match image shape {(H, W)}"
            )

        g_norm_interior = g_norm[1 : H - 1, 1 : W - 1]
        omega_size = (H - 2) * (W - 2)

        # 4. Construct Residuals & Validity Masks
        if C >= 3:
            r = arr_float[:, :, 0]
            g = arr_float[:, :, 1]
            b = arr_float[:, :, 2]
            luma_float = 0.2126 * r + 0.7152 * g + 0.0722 * b

            res_Y, _ = compute_3x3_median_residual(luma_float)
            res_R, finite_R = compute_3x3_median_residual(r)
            res_G, finite_G = compute_3x3_median_residual(g)
            res_B, finite_B = compute_3x3_median_residual(b)

            # Luma U_Y requires all contributing RGB channels to be finite
            U_Y = finite_R & finite_G & finite_B
            U_R = finite_R
            U_G = finite_G
            U_B = finite_B
        else:
            luma_float = arr_float[:, :, 0] if arr_float.ndim == 3 else arr_float
            res_Y, finite_Y = compute_3x3_median_residual(luma_float)
            U_Y = finite_Y
            res_R = res_G = res_B = None
            U_R = U_G = U_B = None

        # Edge exclusion
        edge_valid = g_norm_interior <= 30.0
        V_Y = U_Y & edge_valid

        if C >= 3:
            V_R = U_R & edge_valid
            V_G = U_G & edge_valid
            V_B = U_B & edge_valid
        else:
            V_R = V_G = V_B = None

        # 5. Input Characteristics & Flag Evaluation
        unmasked_finite_pixel_count = int(np.sum(U_Y))
        edge_excluded_finite_pixel_count = int(np.sum(V_Y))
        residual_invalid_neighborhood_count = int(omega_size - unmasked_finite_pixel_count)
        coverage_ratio = (
            float(edge_excluded_finite_pixel_count) / float(omega_size)
            if omega_size > 0
            else 0.0
        )

        if coverage_ratio < 0.05:
            quality_flags.append("LOW_HOMOGENEOUS_COVERAGE")

        edge_fraction = (
            float(np.sum(g_norm_interior > 30.0)) / float(omega_size)
            if omega_size > 0
            else 0.0
        )
        if edge_fraction > 0.80:
            quality_flags.append("HIGH_EDGE_DENSITY")

        input_characteristics = InputCharacteristics(
            total_pixel_count=total_pixel_count,
            input_invalid_float_count=input_invalid_float_count,
            residual_invalid_neighborhood_count=residual_invalid_neighborhood_count,
            unmasked_finite_pixel_count=unmasked_finite_pixel_count,
            edge_excluded_finite_pixel_count=edge_excluded_finite_pixel_count,
            coverage_ratio=coverage_ratio,
        )

        # 6. Global Luma Statistics
        unmasked_luma_stats = compute_statistical_moments(res_Y[U_Y], quality_flags)
        edge_excluded_luma_stats = compute_statistical_moments(res_Y[V_Y], quality_flags)

        # 7. Spatial Descriptors
        spatial_descriptors = compute_spatial_residual_descriptors(res_Y, V_Y, quality_flags)

        # 8. Channel Descriptors (RGB / RGBA)
        if C >= 3:
            channel_descriptors = compute_channel_residual_descriptors(
                res_R, res_G, res_B, V_R, V_G, V_B, quality_flags
            )
        else:
            channel_descriptors = None

        return ResidualNoiseAnalysisResult(
            analyzer_id="pillar2_residual_noise_analyzer",
            analyzer_version="0.3.5",
            input_characteristics=input_characteristics,
            unmasked_luma_stats=unmasked_luma_stats,
            edge_excluded_luma_stats=edge_excluded_luma_stats,
            spatial_descriptors=spatial_descriptors,
            channel_descriptors=channel_descriptors,
            quality_flags=quality_flags,
            limitations=list(PHASE_3A_LIMITATIONS),
        )
