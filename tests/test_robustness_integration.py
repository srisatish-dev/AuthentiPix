"""Integration tests for AuthentiPix robustness, format, and screenshot handling.

Validates the forensic synthesis hierarchy and proves that no unsupported
deterministic thresholds or hardcoded filename rules act as classifiers.
"""

import os
import pytest
from PIL import Image
import numpy as np

from authentipix.metadata.schemas import AnalyzerOutput
from authentipix.pixel.schemas import (
    InputCharacteristics,
    PixelAnalysisResult,
    PixelProperties,
    ResidualNoiseAnalysisResult,
    ResidualStatisticalMoments,
    SpatialResidualDescriptors,
)
from authentipix.robustness.evidence import synthesize_robustness_evidence
from authentipix.robustness.screenshot_indicators import analyze_screenshot_indicators
from authentipix.robustness.schemas import (
    EvidenceTier,
    FormatDetectionResult,
    RecaptureCandidateStatus,
    ScreenshotIndicatorsResult,
    SignalState,
    TransformationAwarenessResult,
)


def make_dummy_residual_moments(robust_std: float = 0.005, std: float = 0.02) -> ResidualNoiseAnalysisResult:
    moments = ResidualStatisticalMoments(
        sample_count=50000,
        mean=0.0002,
        std=std,
        mad=robust_std / 1.4826 if robust_std > 0 else 0.0,
        robust_std=robust_std,
        rms=std,
        skewness=0.05,
        kurtosis=3.1,
    )
    spatial = SpatialResidualDescriptors(
        evaluated_block_count=100,
        valid_block_count=100,
        local_variance_mean=0.0004,
        local_variance_std=0.0001,
        spatial_heterogeneity_index=2.0,
        horizontal_lag1_autocorrelation=0.15,
        vertical_lag1_autocorrelation=0.12,
    )
    inp = InputCharacteristics(total_pixel_count=50000, coverage_ratio=0.96)
    return ResidualNoiseAnalysisResult(
        input_characteristics=inp,
        edge_excluded_luma_stats=moments,
        spatial_descriptors=spatial,
    )


def test_c2pa_ai_declaration_produces_confirmed_tier():
    """Cryptographic C2PA manifest with AI declaration yields CONFIRMED tier."""
    format_res = FormatDetectionResult(
        filename="ai_sample.png",
        filename_extension=".png",
        actual_format="PNG",
        mime_type="image/png",
        format_mismatch_detected=False,
    )
    raw_obs = {
        "c2pa_extractor": {
            "c2pa_present": True,
            "active_manifest": {"ai_generated_declared": True, "claim_generator": "Adobe Firefly"},
        }
    }
    meta_out = AnalyzerOutput(raw_observations=raw_obs, normalized_features={})
    screenshot_res = ScreenshotIndicatorsResult(
        candidate_status=RecaptureCandidateStatus.NO_STRONG_RECAPTURE_INDICATORS
    )
    trans_res = TransformationAwarenessResult()

    report = synthesize_robustness_evidence(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=None,
        residual_result=None,
        screenshot_result=screenshot_res,
        transformation_result=trans_res,
    )

    assert report.synthesized_assessment.evidence_tier == EvidenceTier.CONFIRMED
    assert "AI-GENERATED" in report.synthesized_assessment.assessment
    assert report.synthesized_assessment.original_source == "DECLARED_GENERATIVE_AI"
    assert report.synthesized_assessment.original_ai_provenance == "VERIFIED_C2PA_MANIFEST"


def test_camera_exif_produces_camera_consistent_tier_not_confirmed():
    """Camera EXIF produces CAMERA-CONSISTENT tier, strictly avoiding CONFIRMED."""
    format_res = FormatDetectionResult(
        filename="camera.jpg",
        filename_extension=".jpg",
        actual_format="JPEG",
        mime_type="image/jpeg",
        format_mismatch_detected=False,
    )
    norm = {"make": "Nikon", "model": "Z9", "datetime_original": "2026:05:01 10:00:00"}
    meta_out = AnalyzerOutput(raw_observations={}, normalized_features=norm)

    residual_res = make_dummy_residual_moments(robust_std=0.007, std=0.03)
    screenshot_res = ScreenshotIndicatorsResult(
        candidate_status=RecaptureCandidateStatus.NO_STRONG_RECAPTURE_INDICATORS
    )
    trans_res = TransformationAwarenessResult()

    report = synthesize_robustness_evidence(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=None,
        residual_result=residual_res,
        screenshot_result=screenshot_res,
        transformation_result=trans_res,
    )

    assert report.synthesized_assessment.evidence_tier == EvidenceTier.CAMERA_CONSISTENT
    assert "CAMERA-CONSISTENT" in report.synthesized_assessment.assessment
    assert report.synthesized_assessment.original_source == "CAMERA_CONSISTENT_PROBABLE"
    assert report.synthesized_assessment.original_ai_provenance == "NOT_CONFIRMED"


def test_screenshot_of_ai_image_remains_unconfirmed_and_unknown_source():
    """Screenshot of an AI image reports candidate recapture while original AI provenance is NOT CONFIRMED."""
    format_res = FormatDetectionResult(
        filename="ai_screenshot.png",
        filename_extension=".png",
        actual_format="PNG",
        mime_type="image/png",
        format_mismatch_detected=False,
    )
    # No EXIF, No C2PA (destroyed by screenshot)
    meta_out = AnalyzerOutput(raw_observations={}, normalized_features={})

    screenshot_res = ScreenshotIndicatorsResult(
        display_geometry_match=SignalState.PRESENT,
        candidate_status=RecaptureCandidateStatus.CANDIDATE_RECAPTURE_INDICATORS_OBSERVED,
        supporting_observations=["Matches 20:9 mobile display geometry", "Suppressed residual noise"],
    )
    trans_res = TransformationAwarenessResult(
        potential_recapture=SignalState.PRESENT,
        potential_metadata_loss=SignalState.POTENTIALLY_ALTERED_OR_DESTROYED,
    )

    report = synthesize_robustness_evidence(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=None,
        residual_result=None,
        screenshot_result=screenshot_res,
        transformation_result=trans_res,
    )

    sa = report.synthesized_assessment
    assert sa.evidence_tier == EvidenceTier.RECAPTURE_LIKE
    assert "SCREENSHOT / RECAPTURE-LIKE" in sa.assessment
    assert sa.original_source == "UNKNOWN"
    assert sa.original_ai_provenance == "NOT_CONFIRMED"
    assert "REAL" not in sa.assessment
    assert "AI-GENERATED" not in sa.assessment


def test_missing_provenance_remains_inconclusive():
    """No EXIF and no C2PA yields INCONCLUSIVE, never jumping to AI or REAL (Test D)."""
    format_res = FormatDetectionResult(
        filename="stripped.jpg",
        filename_extension=".jpg",
        actual_format="JPEG",
        mime_type="image/jpeg",
        format_mismatch_detected=False,
    )
    meta_out = AnalyzerOutput(raw_observations={}, normalized_features={})
    screenshot_res = ScreenshotIndicatorsResult(
        candidate_status=RecaptureCandidateStatus.NO_STRONG_RECAPTURE_INDICATORS
    )
    trans_res = TransformationAwarenessResult()

    report = synthesize_robustness_evidence(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=None,
        residual_result=None,
        screenshot_result=screenshot_res,
        transformation_result=trans_res,
    )

    sa = report.synthesized_assessment
    assert sa.evidence_tier == EvidenceTier.INCONCLUSIVE
    assert "INCONCLUSIVE / NO STRONG PROVENANCE SIGNAL" in sa.assessment
    assert sa.original_source == "UNKNOWN"
    assert sa.original_ai_provenance == "NOT_CONFIRMED"


# ----------------------------------------------------------------------------
# Specific Audit Tests (Tests A, B, C, E, F)
# ----------------------------------------------------------------------------

def test_audit_a_zero_residual_does_not_automatically_mean_screenshot():
    """Test A: An image with robust_std = 0 must NOT automatically become SCREENSHOT without contextual evidence."""
    format_res = FormatDetectionResult(
        filename="flat_graphic.png",
        filename_extension=".png",
        actual_format="PNG",
        mime_type="image/png",
        format_mismatch_detected=False,
    )
    meta_out = AnalyzerOutput(raw_observations={}, normalized_features={})
    # Zero residual on a standard 4:3 image (e.g. clean graphic or clean photograph)
    residual_res = make_dummy_residual_moments(robust_std=0.000000, std=0.005)
    
    # Analyze screenshot indicators on 4:3 non-screen dimensions
    screenshot_res = analyze_screenshot_indicators(
        width=1200,
        height=900,
        pixel_result=None,
        residual_result=residual_res,
    )

    # Geometry is NOT display profile; robust_std = 0 alone is INCONCLUSIVE, NOT a confirmed screenshot
    assert screenshot_res.candidate_status != RecaptureCandidateStatus.CANDIDATE_RECAPTURE_INDICATORS_OBSERVED

    report = synthesize_robustness_evidence(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=None,
        residual_result=residual_res,
        screenshot_result=screenshot_res,
        transformation_result=TransformationAwarenessResult(),
    )

    # Must NOT classify as SCREENSHOT
    assert report.synthesized_assessment.evidence_tier == EvidenceTier.INCONCLUSIVE
    assert "SCREENSHOT" not in report.synthesized_assessment.assessment


def test_audit_b_nonzero_residual_does_not_automatically_mean_camera():
    """Test B: A value such as robust_std = 0.0011 must NOT automatically become CAMERA without EXIF make/model."""
    format_res = FormatDetectionResult(
        filename="noisy_unknown.jpg",
        filename_extension=".jpg",
        actual_format="JPEG",
        mime_type="image/jpeg",
        format_mismatch_detected=False,
    )
    # NO EXIF make/model
    meta_out = AnalyzerOutput(raw_observations={}, normalized_features={})
    # Non-zero residual noise typical of sensor or high-frequency texture
    residual_res = make_dummy_residual_moments(robust_std=0.0011, std=0.02)
    screenshot_res = ScreenshotIndicatorsResult(
        candidate_status=RecaptureCandidateStatus.NO_STRONG_RECAPTURE_INDICATORS
    )

    report = synthesize_robustness_evidence(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=None,
        residual_result=residual_res,
        screenshot_result=screenshot_res,
        transformation_result=TransformationAwarenessResult(),
    )

    # Must NOT classify as CAMERA-CONSISTENT
    assert report.synthesized_assessment.evidence_tier == EvidenceTier.INCONCLUSIVE
    assert "CAMERA-CONSISTENT" not in report.synthesized_assessment.assessment


def test_audit_c_known_mobile_resolution_does_not_automatically_mean_screenshot():
    """Test C: Matching 1272x2800 must NOT automatically classify an image as screenshot."""
    format_res = FormatDetectionResult(
        filename="camera_fullscreen.jpg",
        filename_extension=".jpg",
        actual_format="JPEG",
        mime_type="image/jpeg",
        format_mismatch_detected=False,
    )
    # Camera EXIF present (e.g. phone camera used full-screen 20:9 capture mode)
    norm = {"make": "OnePlus", "model": "OnePlus Nord 5", "datetime_original": "2026:06:10 12:00:00"}
    meta_out = AnalyzerOutput(raw_observations={}, normalized_features=norm)

    # Continuous camera sensor noise (not suppressed) and low autocorrelation
    residual_res = make_dummy_residual_moments(robust_std=0.008, std=0.03)

    screenshot_res = analyze_screenshot_indicators(
        width=1272,
        height=2800,
        pixel_result=None,
        residual_result=residual_res,
    )

    # Geometry alone does NOT yield candidate recapture
    assert screenshot_res.candidate_status != RecaptureCandidateStatus.CANDIDATE_RECAPTURE_INDICATORS_OBSERVED

    report = synthesize_robustness_evidence(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=None,
        residual_result=residual_res,
        screenshot_result=screenshot_res,
        transformation_result=TransformationAwarenessResult(),
    )

    # Camera EXIF is respected, not overridden by geometry match alone
    assert report.synthesized_assessment.evidence_tier == EvidenceTier.CAMERA_CONSISTENT
    assert "CAMERA-CONSISTENT" in report.synthesized_assessment.assessment


def test_audit_e_screenshot_does_not_imply_ai():
    """Test E: A screenshot/recapture image must report Original Source = UNKNOWN and Original AI = NOT CONFIRMED."""
    format_res = FormatDetectionResult(
        filename="screen_grab.png",
        filename_extension=".png",
        actual_format="PNG",
        mime_type="image/png",
        format_mismatch_detected=False,
    )
    meta_out = AnalyzerOutput(raw_observations={}, normalized_features={})
    screenshot_res = ScreenshotIndicatorsResult(
        candidate_status=RecaptureCandidateStatus.CANDIDATE_RECAPTURE_INDICATORS_OBSERVED,
        supporting_observations=["Contextual display geometry match", "Suppressed residual noise", "Elevated autocorrelation"],
    )

    report = synthesize_robustness_evidence(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=None,
        residual_result=None,
        screenshot_result=screenshot_res,
        transformation_result=TransformationAwarenessResult(potential_recapture=SignalState.PRESENT),
    )

    sa = report.synthesized_assessment
    assert sa.evidence_tier == EvidenceTier.RECAPTURE_LIKE
    assert sa.original_source == "UNKNOWN"
    assert sa.original_ai_provenance == "NOT_CONFIRMED"
    assert "AI-GENERATED" not in sa.assessment


def test_audit_f_no_filename_hardcoding():
    """Test F: Changing the filename while keeping image evidence identical produces the same forensic inference."""
    norm = {"make": "Sony", "model": "A7IV", "datetime_original": "2026:07:01 14:00:00"}
    meta_out = AnalyzerOutput(raw_observations={}, normalized_features=norm)
    residual_res = make_dummy_residual_moments(robust_std=0.006)
    screenshot_res = ScreenshotIndicatorsResult(candidate_status=RecaptureCandidateStatus.NO_STRONG_RECAPTURE_INDICATORS)
    trans_res = TransformationAwarenessResult()

    # Run with standard filename
    fmt1 = FormatDetectionResult(
        filename="sample1.jpg",
        filename_extension=".jpg",
        actual_format="JPEG",
        mime_type="image/jpeg",
        format_mismatch_detected=False,
    )
    report1 = synthesize_robustness_evidence(fmt1, meta_out, None, residual_res, screenshot_res, trans_res)

    # Run with arbitrary custom filename
    fmt2 = FormatDetectionResult(
        filename="completely_arbitrary_name_9999.jpg",
        filename_extension=".jpg",
        actual_format="JPEG",
        mime_type="image/jpeg",
        format_mismatch_detected=False,
    )
    report2 = synthesize_robustness_evidence(fmt2, meta_out, None, residual_res, screenshot_res, trans_res)

    assert report1.synthesized_assessment.assessment == report2.synthesized_assessment.assessment
    assert report1.synthesized_assessment.evidence_tier == report2.synthesized_assessment.evidence_tier
    assert report1.synthesized_assessment.original_source == report2.synthesized_assessment.original_source
    assert report1.synthesized_assessment.primary_rationale == report2.synthesized_assessment.primary_rationale
