"""Comprehensive test suite for Phase 3A: Residual & Noise-Characteristic Analyzer.

Tests all mathematical invariants, mask logic, sample count hierarchies,
degenerate edge cases, error conditions, and cross-dtype equivalence.
"""

import pytest
import numpy as np

from authentipix.pixel.residual import (
    ResidualNoiseAnalyzer,
    ResidualAnalysisError,
    UnsupportedDtypeError,
    FloatRangeViolationError,
    SmallImageError,
    SobelIntegrationError,
    compute_statistical_moments,
    compute_3x3_median_residual,
    PHASE_3A_LIMITATIONS,
)
from authentipix.pixel.schemas import (
    ResidualNoiseAnalysisResult,
    InputCharacteristics,
    ResidualStatisticalMoments,
    SpatialResidualDescriptors,
    ChannelResidualDescriptors,
)


@pytest.fixture
def analyzer():
    return ResidualNoiseAnalyzer()


# ============================================================================
# 1. Dimension Validation
# ============================================================================

def test_small_image_aborts(analyzer):
    """Images smaller than 32x32 must raise SmallImageError."""
    small_2d = np.zeros((31, 32), dtype=np.uint8)
    with pytest.raises(SmallImageError):
        analyzer.analyze(small_2d)

    small_3d = np.zeros((32, 31, 3), dtype=np.uint8)
    with pytest.raises(SmallImageError):
        analyzer.analyze(small_3d)

    small_exact = np.zeros((16, 16), dtype=np.float64)
    with pytest.raises(SmallImageError):
        analyzer.analyze(small_exact)


def test_minimum_valid_size_32x32(analyzer):
    """32x32 image is the minimum supported size and must complete analysis."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    result = analyzer.analyze(img)
    assert isinstance(result, ResidualNoiseAnalysisResult)
    assert result.input_characteristics.total_pixel_count == 1024
    # Interior lattice is (30, 30) -> evaluated blocks: floor(30/16)*floor(30/16) = 1*1 = 1
    assert result.spatial_descriptors.evaluated_block_count == 1


# ============================================================================
# 2. Dtype Normalization & Validation
# ============================================================================

def test_unsupported_dtype_aborts(analyzer):
    """Unsupported dtypes (e.g. int32, int64, complex) must raise UnsupportedDtypeError."""
    img_int32 = np.zeros((32, 32), dtype=np.int32)
    with pytest.raises(UnsupportedDtypeError):
        analyzer.analyze(img_int32)

    img_int64 = np.zeros((32, 32, 3), dtype=np.int64)
    with pytest.raises(UnsupportedDtypeError):
        analyzer.analyze(img_int64)

    img_int8 = np.zeros((32, 32, 3), dtype=np.int8)
    with pytest.raises(UnsupportedDtypeError):
        analyzer.analyze(img_int8)


def test_cross_dtype_equivalence(analyzer):
    """Equivalent normalized patterns in uint8, uint16, float64 must produce equivalent results within tolerances."""
    # Create a smooth gradient pattern with synthetic high frequency noise
    np.random.seed(42)
    base_pattern = np.random.uniform(0.1, 0.9, (64, 64, 3))

    # Represent as uint8, uint16, and float64
    u8_arr = np.clip(np.round(base_pattern * 255.0), 0, 255).astype(np.uint8)
    u16_arr = np.clip(np.round(base_pattern * 65535.0), 0, 65535).astype(np.uint16)
    f64_arr = u8_arr.astype(np.float64) / 255.0

    res_u8 = analyzer.analyze(u8_arr)
    res_f64 = analyzer.analyze(f64_arr)

    # u8 and normalized float64 from u8 should be identical
    assert np.isclose(
        res_u8.unmasked_luma_stats.mean,
        res_f64.unmasked_luma_stats.mean,
        atol=1e-10,
    )
    assert np.isclose(
        res_u8.unmasked_luma_stats.std,
        res_f64.unmasked_luma_stats.std,
        atol=1e-10,
    )

    # u16 quantization is finer, results should be close to u8 within quantization tolerance
    res_u16 = analyzer.analyze(u16_arr)
    assert np.isclose(
        res_u16.unmasked_luma_stats.std,
        res_u8.unmasked_luma_stats.std,
        atol=5e-3,
    )


# ============================================================================
# 3. Float Range Validation
# ============================================================================

def test_float_range_violation_negative(analyzer):
    """Float input with finite value < 0.0 must raise FloatRangeViolationError."""
    arr = np.zeros((32, 32), dtype=np.float64)
    arr[10, 10] = -0.001
    with pytest.raises(FloatRangeViolationError):
        analyzer.analyze(arr)


def test_float_range_violation_above_one(analyzer):
    """Float input with finite value > 1.0 must raise FloatRangeViolationError."""
    arr = np.ones((32, 32, 3), dtype=np.float32)
    arr[5, 5, 0] = 1.001
    with pytest.raises(FloatRangeViolationError):
        analyzer.analyze(arr)


# ============================================================================
# 4. Non-Finite Handling & Validity Masks
# ============================================================================

def test_isolated_nan_handling(analyzer):
    """Injected NaNs must raise INVALID_FLOAT_VALUES flag, exclude affected 3x3 neighborhoods, and not abort."""
    arr = np.full((34, 34, 3), 0.5, dtype=np.float64)
    # Inject isolated NaN at source pixel (10, 10) in Red channel
    arr[10, 10, 0] = np.nan

    res = analyzer.analyze(arr)
    assert "INVALID_FLOAT_VALUES" in res.quality_flags
    assert res.input_characteristics.input_invalid_float_count == 1

    # In 34x34 image, interior lattice is 32x32 = 1024 pixels.
    # An isolated NaN in 3x3 affects 9 interior positions in U_Y and U_R.
    assert res.input_characteristics.residual_invalid_neighborhood_count == 9
    assert res.input_characteristics.unmasked_finite_pixel_count == 1024 - 9


def test_mask_channel_independence(analyzer):
    """Non-finite in Red channel must not invalidate Green or Blue channel unmasked sets."""
    arr = np.full((34, 34, 3), 0.5, dtype=np.float64)
    # Inject isolated NaN in Red channel at (10, 10)
    arr[10, 10, 0] = np.nan

    # Test raw unmasked masks via residual computation
    res_R, finite_R = compute_3x3_median_residual(arr[:, :, 0])
    res_G, finite_G = compute_3x3_median_residual(arr[:, :, 1])
    res_B, finite_B = compute_3x3_median_residual(arr[:, :, 2])

    # Green and Blue channel 3x3 finite masks should have full 1024 valid samples
    assert int(np.sum(finite_G)) == 1024
    assert int(np.sum(finite_B)) == 1024
    # Red channel should have 1024 - 9 = 1015 valid samples
    assert int(np.sum(finite_R)) == 1015

    # Full analyzer execution
    res = analyzer.analyze(arr)
    assert res.channel_descriptors.green_stats is not None
    assert res.channel_descriptors.blue_stats is not None
    assert res.channel_descriptors.red_stats is not None


def test_multiple_nans_count_can_exceed_wh(analyzer):
    """input_invalid_float_count is scalar across channels and can exceed W*H."""
    arr = np.full((32, 32, 3), 0.5, dtype=np.float64)
    # Set all channels of all pixels to NaN
    arr[:, :, :] = np.nan

    res = analyzer.analyze(arr)
    assert res.input_characteristics.input_invalid_float_count == 32 * 32 * 3
    assert res.input_characteristics.unmasked_finite_pixel_count == 0
    assert res.input_characteristics.edge_excluded_finite_pixel_count == 0
    assert res.input_characteristics.coverage_ratio == 0.0
    assert res.unmasked_luma_stats is None
    assert res.edge_excluded_luma_stats is None


# ============================================================================
# 5. Statistical Moments & Sample Count Hierarchy
# ============================================================================

def test_sample_count_hierarchy_n_0():
    """N=0 produces None."""
    flags = []
    empty_samples = np.array([], dtype=np.float64)
    stats = compute_statistical_moments(empty_samples, flags)
    assert stats is None
    assert flags == []


def test_sample_count_hierarchy_1_to_7():
    """1 <= N < 8 computes basic moments, sets higher moments to None, raises LOW_SAMPLE_COUNT."""
    flags = []
    samples = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)  # N = 5
    stats = compute_statistical_moments(samples, flags)
    assert stats is not None
    assert stats.sample_count == 5
    assert stats.skewness is None
    assert stats.kurtosis is None
    assert "LOW_SAMPLE_COUNT" in flags


def test_sample_count_hierarchy_8_to_31():
    """8 <= N < 32 computes all moments (including skewness/kurtosis), raises LOW_SAMPLE_COUNT."""
    flags = []
    np.random.seed(123)
    samples = np.random.normal(0.0, 0.1, 20).astype(np.float64)  # N = 20
    stats = compute_statistical_moments(samples, flags)
    assert stats is not None
    assert stats.sample_count == 20
    assert stats.skewness is not None
    assert stats.kurtosis is not None
    assert "LOW_SAMPLE_COUNT" in flags


def test_sample_count_hierarchy_ge_32():
    """N >= 32 computes all moments and does NOT raise LOW_SAMPLE_COUNT."""
    flags = []
    np.random.seed(123)
    samples = np.random.normal(0.0, 0.1, 100).astype(np.float64)  # N = 100
    stats = compute_statistical_moments(samples, flags)
    assert stats is not None
    assert stats.sample_count == 100
    assert stats.skewness is not None
    assert stats.kurtosis is not None
    assert "LOW_SAMPLE_COUNT" not in flags


# ============================================================================
# 6. Zero Variance & Solid Color Handling
# ============================================================================

def test_zero_variance_solid_color(analyzer):
    """Solid color image has std=0.0 -> higher moments None, zero scale flag, SHI None."""
    solid = np.full((64, 64, 3), 128, dtype=np.uint8)
    res = analyzer.analyze(solid)

    assert "ZERO_OR_NEGLIGIBLE_RESIDUAL_SCALE" in res.quality_flags
    assert res.unmasked_luma_stats.std == 0.0
    assert res.unmasked_luma_stats.skewness is None
    assert res.unmasked_luma_stats.kurtosis is None

    # Spatial descriptors
    assert res.spatial_descriptors is not None
    # Block variances are all 0.0 -> local_variance_mean == 0.0 <= 1e-12 -> SHI = None
    assert res.spatial_descriptors.local_variance_mean == 0.0
    assert res.spatial_descriptors.spatial_heterogeneity_index is None


# ============================================================================
# 7. Spatial Block Heterogeneity & Autocorrelation Independence
# ============================================================================

def test_spatial_descriptors_block_preservation_under_4_blocks(analyzer):
    """When valid_block_count < 4, preserve evaluated/valid counts, set block stats to None, flag INSUFFICIENT_VALID_BLOCKS."""
    # 40x40 image: interior is 38x38. Evaluated 16x16 blocks: floor(38/16)*floor(38/16) = 2*2 = 4 blocks.
    # Put strong vertical stripes in top-left, top-right, bottom-left to exclude them via Sobel
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    for c in range(0, 32):
        if (c // 2) % 2 == 0:
            img[:32, c, :] = 255
    # Leave bottom-right (row 16..32, col 16..32) homogeneous
    img[16:32, 16:32, :] = 0

    res = analyzer.analyze(img)
    sp = res.spatial_descriptors
    assert sp.evaluated_block_count == 4
    assert sp.valid_block_count < 4
    assert sp.local_variance_mean is None
    assert sp.local_variance_std is None
    assert sp.spatial_heterogeneity_index is None
    assert "INSUFFICIENT_VALID_BLOCKS" in res.quality_flags


def test_autocorrelation_computed_independently_of_blocks(analyzer):
    """Autocorrelation is independent of block validity and must compute when adjacent pairs >= 32."""
    # 40x40 image with sufficient valid pairs in V_Y
    np.random.seed(99)
    noise = np.random.normal(128, 10, (40, 40, 3)).clip(0, 255).astype(np.uint8)
    res = analyzer.analyze(noise)

    sp = res.spatial_descriptors
    # Even if blocks were insufficient, autocorrelation would be computed
    assert sp.horizontal_lag1_autocorrelation is not None
    assert sp.vertical_lag1_autocorrelation is not None


# ============================================================================
# 8. Channel Analysis, Correlations, and Scale Ratios
# ============================================================================

def test_channel_ratio_guard_constant_green(analyzer):
    """When Green robust_std <= 1e-12, both ratios must be None and UNDEFINED_CHANNEL_RATIO raised."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    # Red and Blue have noise, Green is constant 100
    np.random.seed(42)
    img[:, :, 0] = np.random.normal(128, 15, (64, 64)).clip(0, 255).astype(np.uint8)
    img[:, :, 1] = 100  # constant green
    img[:, :, 2] = np.random.normal(128, 10, (64, 64)).clip(0, 255).astype(np.uint8)

    res = analyzer.analyze(img)
    ch = res.channel_descriptors
    assert ch.blue_to_green_ratio is None
    assert ch.red_to_green_ratio is None
    assert "UNDEFINED_CHANNEL_RATIO" in res.quality_flags


def test_channel_correlations_paired_samples(analyzer):
    """Inter-channel correlations evaluate strictly on pairwise valid sample intersections."""
    np.random.seed(42)
    # Generate correlated R and G residuals, uncorrelated B
    common_noise = np.random.normal(0, 15, (64, 64))
    r = np.clip(128 + common_noise + np.random.normal(0, 2, (64, 64)), 0, 255).astype(np.uint8)
    g = np.clip(128 + common_noise + np.random.normal(0, 2, (64, 64)), 0, 255).astype(np.uint8)
    b = np.clip(128 + np.random.normal(0, 15, (64, 64)), 0, 255).astype(np.uint8)

    img = np.stack([r, g, b], axis=-1)
    res = analyzer.analyze(img)
    ch = res.channel_descriptors

    assert ch.rg_correlation is not None
    assert ch.rg_correlation > 0.8  # Strong correlation between R and G
    assert ch.rb_correlation is not None
    assert abs(ch.rb_correlation) < 0.3  # Weak correlation with B


# ============================================================================
# 9. RGBA Alpha Exclusion & Grayscale
# ============================================================================

def test_rgba_alpha_excluded(analyzer):
    """Alpha channel in RGBA is strictly excluded and yields identical results to RGB."""
    np.random.seed(77)
    rgb = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    # Add random alpha channel
    alpha = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=-1)

    res_rgb = analyzer.analyze(rgb)
    res_rgba = analyzer.analyze(rgba)

    assert res_rgb.unmasked_luma_stats.mean == pytest.approx(res_rgba.unmasked_luma_stats.mean, abs=1e-10)
    assert res_rgb.unmasked_luma_stats.std == pytest.approx(res_rgba.unmasked_luma_stats.std, abs=1e-10)
    assert res_rgb.channel_descriptors.rg_correlation == pytest.approx(res_rgba.channel_descriptors.rg_correlation, abs=1e-10)


def test_grayscale_input_no_channel_descriptors(analyzer):
    """Grayscale input (2D or 1-channel) returns channel_descriptors=None."""
    gray_2d = np.full((48, 48), 100, dtype=np.uint8)
    res_2d = analyzer.analyze(gray_2d)
    assert res_2d.channel_descriptors is None

    gray_3d = np.full((48, 48, 1), 100, dtype=np.uint8)
    res_3d = analyzer.analyze(gray_3d)
    assert res_3d.channel_descriptors is None


# ============================================================================
# 10. Quality Flags & Limitations Payload
# ============================================================================

def test_high_edge_density_flag(analyzer):
    """HIGH_EDGE_DENSITY is raised when >80% of interior pixels exceed Sobel threshold."""
    # 2-pixel vertical stripes -> strong Sobel edges throughout
    img = np.zeros((64, 64), dtype=np.uint8)
    for c in range(64):
        if (c // 2) % 2 == 0:
            img[:, c] = 255

    res = analyzer.analyze(img)
    assert "HIGH_EDGE_DENSITY" in res.quality_flags


def test_low_homogeneous_coverage_flag(analyzer):
    """LOW_HOMOGENEOUS_COVERAGE is raised when |V_Y| / |Omega| < 0.05."""
    # 2-pixel vertical stripes -> almost all pixels exceed edge threshold
    img = np.zeros((64, 64), dtype=np.uint8)
    for c in range(64):
        if (c // 2) % 2 == 0:
            img[:, c] = 255

    res = analyzer.analyze(img)
    assert res.input_characteristics.coverage_ratio < 0.05
    assert "LOW_HOMOGENEOUS_COVERAGE" in res.quality_flags




def test_authoritative_limitations_preserved(analyzer):
    """The result must contain the exact 4 authoritative limitations from v0.3.5-final."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    res = analyzer.analyze(img)
    assert res.limitations == PHASE_3A_LIMITATIONS
    assert len(res.limitations) == 4
    for item in res.limitations:
        assert "authenticity" not in item.lower() or "not independently prove" in item.lower()


def test_uint32_normalization(analyzer):
    """uint32 inputs must normalize by 4294967295.0 correctly."""
    u32_arr = np.full((32, 32, 3), 4294967295, dtype=np.uint32)
    res = analyzer.analyze(u32_arr)
    assert res.input_characteristics.total_pixel_count == 1024
    assert res.unmasked_luma_stats.mean == pytest.approx(0.0, abs=1e-10)


def test_analyze_context_integration(analyzer):
    """analyze_context must successfully process real image assets via AnalysisContext."""
    from authentipix.metadata.context import AnalysisContext
    import os

    sample_path = "sample1.jpeg"
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            data = f.read()
        ctx = AnalysisContext.from_bytes(
            data,
            image_id="test_sample1",
            mime_type="image/jpeg",
        )
        res = analyzer.analyze_context(ctx)
        assert isinstance(res, ResidualNoiseAnalysisResult)
        assert res.analyzer_id == "pillar2_residual_noise_analyzer"
        assert res.analyzer_version == "0.3.5"
        assert res.input_characteristics.total_pixel_count > 0
        assert res.unmasked_luma_stats is not None
        assert res.edge_excluded_luma_stats is not None
        assert res.spatial_descriptors is not None
        assert res.channel_descriptors is not None


