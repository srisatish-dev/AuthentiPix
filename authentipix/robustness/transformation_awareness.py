"""Transformation awareness and evidence qualification.

Evaluates whether an image has undergone transformations (format conversions,
re-encoding, metadata stripping, resizing, or screen recapture) that could alter,
mask, or destroy original forensic signals.
"""

from typing import List, Optional
from authentipix.metadata.schemas import AnalyzerOutput
from authentipix.pixel.schemas import PixelAnalysisResult, ResidualNoiseAnalysisResult
from authentipix.robustness.schemas import (
    FormatDetectionResult,
    RecaptureCandidateStatus,
    ScreenshotIndicatorsResult,
    SignalState,
    TransformationAwarenessResult,
)


def analyze_transformations(
    format_result: FormatDetectionResult,
    metadata_output: Optional[AnalyzerOutput],
    pixel_result: Optional[PixelAnalysisResult],
    residual_result: Optional[ResidualNoiseAnalysisResult],
    screenshot_result: Optional[ScreenshotIndicatorsResult],
) -> TransformationAwarenessResult:
    """Audits potential transformations that could affect provenance and pixel evidence."""
    detected: List[str] = []
    notes: List[str] = []

    # 1. Format mismatch auditing
    has_format_mismatch = format_result.format_mismatch_detected
    if has_format_mismatch:
        detected.append(
            f"Format Mismatch: Ext '{format_result.filename_extension}' vs Actual '{format_result.actual_format}'"
        )
        notes.append(
            f"File extension '{format_result.filename_extension}' contradicts actual binary container "
            f"'{format_result.actual_format}'. Indicates cross-format saving, manual extension alteration, or conversion."
        )

    # 2. Metadata presence check
    has_exif = False
    has_c2pa = False
    if metadata_output:
        norm = metadata_output.normalized_features
        has_exif = bool(norm.get("make") or norm.get("model") or norm.get("datetime_original") or norm.get("exif_width"))
        raw_obs = metadata_output.raw_observations
        has_c2pa = raw_obs.get("c2pa_extractor", {}).get("c2pa_present", False)

    # 3. Recapture correlation
    is_recapture_candidate = (
        screenshot_result is not None
        and screenshot_result.candidate_status == RecaptureCandidateStatus.CANDIDATE_RECAPTURE_INDICATORS_OBSERVED
    )
    if is_recapture_candidate:
        detected.append("Candidate Screen Recapture")
        notes.append(
            "Screen capture / viewport characteristics observed. Screen capture processes "
            "inherently strip all pre-existing camera EXIF and C2PA provenance."
        )

    # 4. Evaluate 3-state Potential Metadata Loss
    if has_exif or has_c2pa:
        potential_metadata_loss = SignalState.NOT_DETECTED
        notes.append("Active provenance metadata is present in current file container.")
    elif is_recapture_candidate or has_format_mismatch:
        potential_metadata_loss = SignalState.POTENTIALLY_ALTERED_OR_DESTROYED
        detected.append("Potential Metadata Stripping via Transformation")
        notes.append(
            "Metadata is absent in current file, and observed transformations (recapture / format mismatch) "
            "are known to destroy original EXIF and C2PA manifests."
        )
    else:
        potential_metadata_loss = SignalState.NOT_DETECTED
        notes.append("No provenance metadata detected in current container; no specific stripping mechanism confirmed.")

    # 5. Potential Re-encoding
    if has_format_mismatch:
        potential_re_encoding = SignalState.PRESENT
        detected.append("Container Re-encoding / Extension Mismatch")
    else:
        # Check comb ratio from Phase 2 histogram if available
        comb_metric = 0.0
        if pixel_result and pixel_result.histograms:
            luma_hist = next((h for h in pixel_result.histograms if h.channel_name == "luma"), pixel_result.histograms[0])
            comb_metric = luma_hist.comb_metric
        if comb_metric > 0.05:
            potential_re_encoding = SignalState.PRESENT
            detected.append(f"Histogram Comb Periodic Gaps (C_comb = {comb_metric:.4f})")
            notes.append("Histogram periodicity indicates non-linear tone curves or secondary re-quantization.")
        else:
            potential_re_encoding = SignalState.NOT_DETECTED

    # 6. Potential Resizing
    potential_resizing = SignalState.NOT_DETECTED
    if pixel_result:
        props = pixel_result.properties
        # Compare raster with EXIF declared dimensions if EXIF exists
        if metadata_output:
            exif_w = metadata_output.normalized_features.get("exif_width")
            exif_h = metadata_output.normalized_features.get("exif_height")
            if exif_w and exif_h and (exif_w != props.width or exif_h != props.height):
                potential_resizing = SignalState.PRESENT
                detected.append(f"Dimension Mismatch: EXIF {exif_w}x{exif_h} vs Raster {props.width}x{props.height}")
                notes.append("Raster dimensions differ from EXIF headers, indicating spatial resampling or cropping.")

    # 7. Potential Recapture State
    if is_recapture_candidate:
        potential_recapture = SignalState.PRESENT
    elif screenshot_result and screenshot_result.candidate_status == RecaptureCandidateStatus.INCONCLUSIVE:
        potential_recapture = SignalState.POTENTIALLY_ALTERED_OR_DESTROYED
    else:
        potential_recapture = SignalState.NOT_DETECTED

    return TransformationAwarenessResult(
        potential_metadata_loss=potential_metadata_loss,
        potential_re_encoding=potential_re_encoding,
        potential_resizing=potential_resizing,
        potential_recapture=potential_recapture,
        format_mismatch_observed=has_format_mismatch,
        detected_transformations=detected,
        notes=notes,
    )
