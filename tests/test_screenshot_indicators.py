"""Unit tests for AuthentiPix candidate screenshot and recapture indicators."""

import pytest
from authentipix.pixel.schemas import (
    InputCharacteristics,
    PixelAnalysisResult,
    PixelProperties,
    ResidualNoiseAnalysisResult,
    ResidualStatisticalMoments,
    SpatialResidualDescriptors,
)
from authentipix.robustness.schemas import RecaptureCandidateStatus, SignalState
from authentipix.robustness.screenshot_indicators import analyze_screenshot_indicators, check_display_geometry


def make_dummy_residual_result(
    robust_std: float = 0.005,
    h_autocorr: float = 0.2,
    v_autocorr: float = 0.15,
    shi: float = 2.0,
) -> ResidualNoiseAnalysisResult:
    moments = ResidualStatisticalMoments(
        sample_count=10000,
        mean=0.0001,
        std=0.02,
        mad=0.005,
        robust_std=robust_std,
        rms=0.02,
        skewness=0.1,
        kurtosis=3.0,
    )
    spatial = SpatialResidualDescriptors(
        evaluated_block_count=100,
        valid_block_count=100,
        local_variance_mean=0.0004,
        local_variance_std=0.0001,
        spatial_heterogeneity_index=shi,
        horizontal_lag1_autocorrelation=h_autocorr,
        vertical_lag1_autocorrelation=v_autocorr,
    )
    inp = InputCharacteristics(
        total_pixel_count=10000,
        unmasked_finite_pixel_count=9800,
        edge_excluded_finite_pixel_count=9500,
        coverage_ratio=0.95,
    )
    return ResidualNoiseAnalysisResult(
        input_characteristics=inp,
        edge_excluded_luma_stats=moments,
        spatial_descriptors=spatial,
    )


def test_display_geometry_detection():
    """Validates geometry matching across common device viewports."""
    # 20:9 FHD+ display
    state, profile = check_display_geometry(1080, 2400)
    assert state == SignalState.PRESENT
    assert "1080x2400" in profile

    # 19.8:9 display (e.g. 1272x2800)
    state, profile = check_display_geometry(1272, 2800)
    assert state == SignalState.PRESENT
    assert "1272x2800" in profile

    # Standard camera 4:3 ratio should NOT match common mobile displays
    state, profile = check_display_geometry(4096, 3072)
    assert state == SignalState.NOT_DETECTED


def test_geometry_alone_is_not_conclusive():
    """Matching a display geometry alone without suppressed residual is inconclusive."""
    # 1080x2400 with normal camera sensor noise
    residual_res = make_dummy_residual_result(robust_std=0.008, h_autocorr=0.15, v_autocorr=0.10)
    res = analyze_screenshot_indicators(
        width=1080,
        height=2400,
        pixel_result=None,
        residual_result=residual_res,
    )

    assert res.display_geometry_match == SignalState.PRESENT
    # Must NOT jump to definitive candidate when noise is continuous and unsuppressed
    assert res.candidate_status == RecaptureCandidateStatus.INCONCLUSIVE


def test_corroborated_recapture_indicators():
    """Geometry match combined with suppressed residual noise produces candidate recapture status."""
    # 1272x2800 with robust_std == 0.0 and high spatial autocorrelation
    residual_res = make_dummy_residual_result(robust_std=0.000000, h_autocorr=0.75, v_autocorr=0.88, shi=4.5)
    res = analyze_screenshot_indicators(
        width=1272,
        height=2800,
        pixel_result=None,
        residual_result=residual_res,
    )

    assert res.display_geometry_match == SignalState.PRESENT
    assert res.candidate_status == RecaptureCandidateStatus.CANDIDATE_RECAPTURE_INDICATORS_OBSERVED
    assert len(res.supporting_observations) >= 2
    assert "robust_std = 0.000000" in " ".join(res.supporting_observations)


def test_clean_camera_photo_has_no_strong_indicators():
    """Standard photographic geometry with continuous noise shows no strong recapture indicators."""
    residual_res = make_dummy_residual_result(robust_std=0.006, h_autocorr=0.12, v_autocorr=0.09)
    res = analyze_screenshot_indicators(
        width=4096,
        height=3072,
        pixel_result=None,
        residual_result=residual_res,
    )

    assert res.display_geometry_match == SignalState.NOT_DETECTED
    assert res.candidate_status == RecaptureCandidateStatus.NO_STRONG_RECAPTURE_INDICATORS
