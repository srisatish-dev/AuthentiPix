"""Unit tests for AuthentiPix transformation awareness and evidence qualification."""

import pytest
from authentipix.metadata.schemas import AnalyzerOutput, NormalizedMetadata
from authentipix.pixel.schemas import PixelAnalysisResult, PixelProperties
from authentipix.robustness.format_detection import detect_actual_format_from_bytes
from authentipix.robustness.schemas import (
    FormatDetectionResult,
    RecaptureCandidateStatus,
    ScreenshotIndicatorsResult,
    SignalState,
)
from authentipix.robustness.transformation_awareness import analyze_transformations


def make_dummy_pixel_result(width: int = 1000, height: int = 1000) -> PixelAnalysisResult:
    props = PixelProperties(
        width=width,
        height=height,
        pixel_count=width * height,
        aspect_ratio=float(width) / float(height),
        native_mode="RGB",
        channel_count=3,
        bit_depth=8,
        dtype="uint8",
    )
    return PixelAnalysisResult(properties=props)


def test_clean_image_no_transformations():
    """Pristine camera image with matching format and metadata shows no destructive transformation."""
    format_res = FormatDetectionResult(
        filename="photo.jpg",
        filename_extension=".jpg",
        actual_format="JPEG",
        mime_type="image/jpeg",
        format_mismatch_detected=False,
    )
    norm = {"make": "Canon", "model": "EOS R5", "datetime_original": "2026:01:01 12:00:00"}
    meta_out = AnalyzerOutput(normalized_features=norm)
    pixel_res = make_dummy_pixel_result(4000, 3000)

    screenshot_res = ScreenshotIndicatorsResult(
        candidate_status=RecaptureCandidateStatus.NO_STRONG_RECAPTURE_INDICATORS
    )

    res = analyze_transformations(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=pixel_res,
        residual_result=None,
        screenshot_result=screenshot_res,
    )

    assert res.format_mismatch_observed is False
    assert res.potential_metadata_loss == SignalState.NOT_DETECTED
    assert res.potential_recapture == SignalState.NOT_DETECTED


def test_format_mismatch_triggers_transformation_signal():
    """Format mismatch triggers re-encoding signal and potential metadata loss qualification."""
    format_res = FormatDetectionResult(
        filename="sample7.png",
        filename_extension=".png",
        actual_format="JPEG",
        mime_type="image/jpeg",
        format_mismatch_detected=True,
    )
    meta_out = AnalyzerOutput(normalized_features={})
    pixel_res = make_dummy_pixel_result(314, 215)
    screenshot_res = ScreenshotIndicatorsResult(
        candidate_status=RecaptureCandidateStatus.NO_STRONG_RECAPTURE_INDICATORS
    )

    res = analyze_transformations(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=pixel_res,
        residual_result=None,
        screenshot_result=screenshot_res,
    )

    assert res.format_mismatch_observed is True
    assert res.potential_re_encoding == SignalState.PRESENT
    assert res.potential_metadata_loss == SignalState.POTENTIALLY_ALTERED_OR_DESTROYED


def test_screenshot_candidate_triggers_metadata_loss_qualification():
    """Observed screen recapture triggers 3-state qualification that metadata was potentially stripped."""
    format_res = FormatDetectionResult(
        filename="screenshot.png",
        filename_extension=".png",
        actual_format="PNG",
        mime_type="image/png",
        format_mismatch_detected=False,
    )
    meta_out = AnalyzerOutput(normalized_features={})
    pixel_res = make_dummy_pixel_result(1080, 2400)
    screenshot_res = ScreenshotIndicatorsResult(
        candidate_status=RecaptureCandidateStatus.CANDIDATE_RECAPTURE_INDICATORS_OBSERVED
    )

    res = analyze_transformations(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=pixel_res,
        residual_result=None,
        screenshot_result=screenshot_res,
    )

    assert res.potential_recapture == SignalState.PRESENT
    assert res.potential_metadata_loss == SignalState.POTENTIALLY_ALTERED_OR_DESTROYED


def test_dimension_mismatch_triggers_resizing():
    """EXIF dimensions differing from raster dimensions indicates resizing/cropping."""
    format_res = FormatDetectionResult(
        filename="resized.jpg",
        filename_extension=".jpg",
        actual_format="JPEG",
        mime_type="image/jpeg",
        format_mismatch_detected=False,
    )
    norm = {"make": "Nikon", "exif_width": 6000, "exif_height": 4000}
    meta_out = AnalyzerOutput(normalized_features=norm)
    pixel_res = make_dummy_pixel_result(3000, 2000)  # Resized raster
    screenshot_res = ScreenshotIndicatorsResult(
        candidate_status=RecaptureCandidateStatus.NO_STRONG_RECAPTURE_INDICATORS
    )

    res = analyze_transformations(
        format_result=format_res,
        metadata_output=meta_out,
        pixel_result=pixel_res,
        residual_result=None,
        screenshot_result=screenshot_res,
    )

    assert res.potential_resizing == SignalState.PRESENT
