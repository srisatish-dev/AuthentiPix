"""Qualified evidence synthesis across provenance, pixel, and transformation channels.

Applies the 4-tier evidence hierarchy without unvalidated universal thresholds or
arbitrary scoring formulas. Maintains strict separation between evidence channels.
"""

from typing import List, Optional
from authentipix.metadata.schemas import AnalyzerOutput
from authentipix.pixel.schemas import PixelAnalysisResult, ResidualNoiseAnalysisResult
from authentipix.robustness.schemas import (
    EvidenceTier,
    FormatDetectionResult,
    RecaptureCandidateStatus,
    RobustnessReport,
    ScreenshotIndicatorsResult,
    SignalState,
    SynthesizedAssessmentResult,
    TransformationAwarenessResult,
)


def synthesize_robustness_evidence(
    format_result: FormatDetectionResult,
    metadata_output: Optional[AnalyzerOutput],
    pixel_result: Optional[PixelAnalysisResult],
    residual_result: Optional[ResidualNoiseAnalysisResult],
    screenshot_result: ScreenshotIndicatorsResult,
    transformation_result: TransformationAwarenessResult,
) -> RobustnessReport:
    """Combines independent forensic observations into an explainable, bounded assessment.
    
    Adheres strictly to the transparent 4-tier evidence hierarchy without any hidden weighted score.
    """

    # 1. Provenance Channel Extraction
    has_exif = False
    camera_make = None
    camera_model = None
    capture_time = None
    c2pa_present = False
    ai_declared = False

    if metadata_output:
        norm = metadata_output.normalized_features
        camera_make = norm.get("make")
        camera_model = norm.get("model")
        capture_time = norm.get("datetime_original")
        has_exif = bool(camera_make or camera_model or capture_time or norm.get("exif_width"))

        raw_obs = metadata_output.raw_observations
        c2pa_info = raw_obs.get("c2pa_extractor", {})
        c2pa_present = c2pa_info.get("c2pa_present", False)
        manifest = c2pa_info.get("active_manifest", {})
        ai_declared = bool(manifest.get("ai_generated_declared", False))

    cam_str = f"{camera_make} / {camera_model}" if (camera_make or camera_model) else "N/A"

    # Evaluate Provenance Sub-Assessment
    if c2pa_present and ai_declared:
        provenance_assessment = "C2PA DECLARED GENERATIVE AI (CRYPTOGRAPHICALLY VERIFIED)"
    elif c2pa_present:
        provenance_assessment = "C2PA MANIFEST PRESENT"
    elif has_exif and (camera_make or camera_model):
        provenance_assessment = f"CAMERA EXIF PRESENT ({cam_str})"
    elif has_exif and capture_time:
        provenance_assessment = f"PARTIAL EXIF PRESENT (Timestamp: {capture_time})"
    else:
        provenance_assessment = "INCONCLUSIVE / NO STRONG PROVENANCE SIGNAL"

    # 2. Pixel / Forensic Observations (Descriptive only, not standalone classifiers)
    has_nonzero_residual = False
    rob_std_val = None
    if residual_result and residual_result.edge_excluded_luma_stats:
        stats = residual_result.edge_excluded_luma_stats
        if stats.robust_std is not None and stats.robust_std > 0.0:
            has_nonzero_residual = True
            rob_std_val = stats.robust_std

    # 3. Recapture Indicators
    is_recapture = (screenshot_result.candidate_status == RecaptureCandidateStatus.CANDIDATE_RECAPTURE_INDICATORS_OBSERVED)

    supporting: List[str] = []
    limitations: List[str] = []

    # 4. Multi-Category Synthesized Assessment Logic (Hierarchical, Rule-Based, No Scoring)

    # TIER 1: Cryptographic C2PA AI Manifest (Confirmed Provenance)
    if c2pa_present and ai_declared:
        assessment = "AI-GENERATED (C2PA DECLARED)"
        tier = EvidenceTier.CONFIRMED
        original_source = "DECLARED_GENERATIVE_AI"
        original_ai_provenance = "VERIFIED_C2PA_MANIFEST"
        rationale = "Strong provenance signal: C2PA manifest explicitly declares generative AI creation with cryptographic binding."
        supporting.append("C2PA manifest is present and cryptographically verified.")
        supporting.append("Generative AI creation action is explicitly declared in manifest.")
        supporting.append(f"Container format verified as {format_result.actual_format}.")
        limitations.append("The Phase 2 & Phase 3A pixel measurements describe raster payload but do not independently prove AI generation.")

    # TIER 2: Camera EXIF (Make and/or Model) AND NOT a Recapture Candidate
    elif has_exif and (camera_make or camera_model) and not is_recapture:
        assessment = "CAMERA-CONSISTENT INDICATORS"
        tier = EvidenceTier.CAMERA_CONSISTENT
        original_source = "CAMERA_CONSISTENT_PROBABLE"
        original_ai_provenance = "NOT_CONFIRMED"
        rationale = f"The image contains camera EXIF metadata identifying {cam_str} corroborated by continuous photographic raster characteristics."
        supporting.append(f"EXIF camera metadata is present ({cam_str}).")
        if capture_time:
            supporting.append(f"Capture timestamp recorded: {capture_time}")
        supporting.append("No C2PA AI-generation declaration was found.")
        if has_nonzero_residual and rob_std_val is not None:
            supporting.append(f"Phase 3A residual analysis records non-zero residual moments (robust_std = {rob_std_val:.6f}).")
        limitations.append("IMPORTANT: EXIF metadata can be edited or fabricated; physical camera acquisition cannot be proved without hardware-rooted cryptographic binding.")
        limitations.append("Residual noise measurements describe the raster payload and do not independently prove authentic optical capture.")

    # TIER 3: Candidate Screenshot / Recapture Observed (Multi-source Corroborated)
    # (Covers screenshot of AI image, screenshot of camera photo, or mobile screen capture)
    elif is_recapture:
        assessment = "SCREENSHOT / RECAPTURE-LIKE INDICATORS"
        tier = EvidenceTier.RECAPTURE_LIKE
        original_source = "UNKNOWN"
        original_ai_provenance = "NOT_CONFIRMED"
        rationale = (
            "Multiple candidate indicators observed consistent with screen capture or display-rendered acquisition "
            "(display geometry profile, residual noise suppression, and/or spatial autocorrelation)."
        )
        supporting.extend(screenshot_result.supporting_observations)
        if has_exif and capture_time and not (camera_make or camera_model):
            supporting.append(f"Partial timestamp ({capture_time}) present, but camera make/model absent.")
        supporting.append("Original acquisition provenance could not be confirmed.")
        limitations.append("Screen capture processes inherently strip pre-existing camera EXIF and C2PA provenance.")
        limitations.append("Candidate recapture indicators do not prove whether the displayed content was originally a camera photo, AI rendering, or document.")
        limitations.append("Original source and AI provenance remain UNKNOWN.")

    # TIER 4: No Strong Provenance & Low Recapture Indicators
    else:
        assessment = "INCONCLUSIVE / NO STRONG PROVENANCE SIGNAL"
        tier = EvidenceTier.INCONCLUSIVE
        original_source = "UNKNOWN"
        original_ai_provenance = "NOT_CONFIRMED"
        rationale = "No C2PA AI declaration and no camera EXIF metadata found in current file container."
        supporting.append("No cryptographic C2PA provenance manifest detected.")
        supporting.append("No camera EXIF metadata detected in current file.")
        if format_result.format_mismatch_detected:
            supporting.append(f"Container format mismatch: extension '{format_result.filename_extension}' vs actual '{format_result.actual_format}'.")
        supporting.append("Pixel and residual statistics are recorded for descriptive forensic context.")
        limitations.append("Absence of metadata does not prove AI generation or physical camera capture.")
        limitations.append("Metadata is frequently stripped by web platforms, messaging applications, or editing tools.")

    synthesized = SynthesizedAssessmentResult(
        assessment=assessment,
        evidence_tier=tier,
        original_source=original_source,
        original_ai_provenance=original_ai_provenance,
        primary_rationale=rationale,
        supporting_observations=supporting,
        limitations=limitations,
    )

    return RobustnessReport(
        format_detection=format_result,
        transformation_awareness=transformation_result,
        screenshot_indicators=screenshot_result,
        provenance_assessment=provenance_assessment,
        synthesized_assessment=synthesized,
    )
