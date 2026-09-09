"""Candidate screenshot and recapture indicator analysis.

Exposes measurable observations regarding display viewport geometry, Phase 3A
residual moments, and spatial characteristics without unvalidated universal thresholds.
"""

from typing import List, Optional, Tuple
from authentipix.pixel.schemas import PixelAnalysisResult, ResidualNoiseAnalysisResult
from authentipix.robustness.schemas import RecaptureCandidateStatus, ScreenshotIndicatorsResult, SignalState

# Contextual reference display aspect ratios (normalized as max_dim / min_dim)
# These represent common digital display form factors for contextual comparison only.
# A geometry match is NEVER definitive proof of screenshot capture.
COMMON_DISPLAY_ASPECT_RATIOS = [
    ("20:9 Display Profile", 20.0 / 9.0, 0.03),      # ~2.222 (e.g. 1080x2400, 1272x2800, 1440x3200)
    ("19.5:9 Display Profile", 19.5 / 9.0, 0.03),  # ~2.167 (e.g. 1170x2532, 1179x2556, 1284x2778)
    ("20.5:9 Display Profile", 20.5 / 9.0, 0.03),  # ~2.278
    ("18:9 / 2:1 Display Profile", 2.0, 0.02),       # 2.000 (e.g. 1080x2160)
    ("16:9 Display Profile", 16.0 / 9.0, 0.02),   # ~1.778 (e.g. 1920x1080, 2560x1440)
    ("16:10 Display Profile", 16.0 / 10.0, 0.02),    # 1.600 (e.g. 1920x1200, 2560x1600)
]

# Contextual reference viewport resolutions (checked unordered: min, max)
COMMON_VIEWPORT_RESOLUTIONS = {
    (1080, 2400): "1080x2400 (FHD+ 20:9 Display Reference)",
    (1080, 2340): "1080x2340 (FHD+ 19.5:9 Display Reference)",
    (1080, 2412): "1080x2412 (FHD+ 20:9 Display Reference)",
    (1080, 2460): "1080x2460 (FHD+ 20.5:9 Display Reference)",
    (1170, 2532): "1170x2532 (19.5:9 Display Reference)",
    (1179, 2556): "1179x2556 (19.5:9 Display Reference)",
    (1272, 2800): "1272x2800 (19.8:9 Display Reference)",
    (1284, 2778): "1284x2778 (19.5:9 Display Reference)",
    (1290, 2796): "1290x2796 (19.5:9 Display Reference)",
    (1440, 3120): "1440x3120 (QHD+ 19.5:9 Display Reference)",
    (1440, 3200): "1440x3200 (QHD+ 20:9 Display Reference)",
    (1080, 1920): "1080x1920 / 1920x1080 (FHD 16:9 Display Reference)",
    (1440, 2560): "1440x2560 / 2560x1440 (QHD 16:9 Display Reference)",
    (2160, 3840): "2160x3840 / 3840x2160 (4K UHD 16:9 Display Reference)",
}


def check_display_geometry(width: int, height: int) -> Tuple[SignalState, Optional[str]]:
    """Evaluates whether image dimensions match known display reference profiles.
    
    IMPORTANT: This is a contextual indicator only. Geometry alone does not establish
    that an image is a screenshot or recapture.
    """
    if width <= 0 or height <= 0:
        return SignalState.UNAVAILABLE, None

    min_dim = min(width, height)
    max_dim = max(width, height)
    ratio = float(max_dim) / float(min_dim)

    # 1. Check contextual reference resolution matches
    exact_match = COMMON_VIEWPORT_RESOLUTIONS.get((min_dim, max_dim))
    if exact_match:
        return SignalState.PRESENT, f"Contextual resolution match: {exact_match}"

    # 2. Check contextual reference aspect ratio matches
    for name, target_ratio, tol in COMMON_DISPLAY_ASPECT_RATIOS:
        if abs(ratio - target_ratio) <= tol:
            return SignalState.PRESENT, f"Contextual aspect ratio match: {name} (observed {ratio:.3f})"

    return SignalState.NOT_DETECTED, None


def analyze_screenshot_indicators(
    width: int,
    height: int,
    pixel_result: Optional[PixelAnalysisResult],
    residual_result: Optional[ResidualNoiseAnalysisResult],
) -> ScreenshotIndicatorsResult:
    """Extracts candidate screenshot / recapture indicators from available forensic measurements.

    Does NOT use unvalidated universal thresholds or arbitrary weighted scoring formulas.
    Evaluates individual indicators as descriptive observations requiring multi-source corroboration.
    """
    aspect_ratio = float(width) / float(height) if height > 0 else 1.0
    dims_str = f"{width} x {height}"

    # Contextual geometry evaluation
    geom_state, matched_profile = check_display_geometry(width, height)

    supporting: List[str] = []
    if geom_state == SignalState.PRESENT and matched_profile:
        supporting.append(f"Screen geometry match: {matched_profile} (contextual indicator only)")

    # Phase 3A Residual Moments Observation (Descriptive feature, not a deterministic threshold)
    residual_obs = "Phase 3A residual measurements unavailable"
    is_residual_suppressed = False
    if residual_result and residual_result.edge_excluded_luma_stats:
        stats = residual_result.edge_excluded_luma_stats
        rob_std = stats.robust_std if stats.robust_std is not None else -1.0
        std_val = stats.std if stats.std is not None else -1.0
        
        if rob_std == 0.0:
            is_residual_suppressed = True
            residual_obs = (
                f"Zero residual robust_std observed ({rob_std:.6f}, std={std_val:.6f}); "
                "consistent with display smoothing, flat regions, or synthetic rendering, "
                "but not deterministic proof of screenshot."
            )
            supporting.append("Residual noise suppression observed (robust_std = 0.000000)")
        elif rob_std > 0.0:
            residual_obs = f"Non-zero residual noise observed (robust_std={rob_std:.6f}, std={std_val:.6f})."

    # Spatial Autocorrelation Observation (Descriptive feature)
    autocorr_obs = "Spatial autocorrelation unavailable"
    is_high_autocorr = False
    if residual_result and residual_result.spatial_descriptors:
        sp = residual_result.spatial_descriptors
        h_ac = sp.horizontal_lag1_autocorrelation
        v_ac = sp.vertical_lag1_autocorrelation
        if h_ac is not None and v_ac is not None:
            if h_ac > 0.6 or v_ac > 0.6:
                is_high_autocorr = True
                autocorr_obs = (
                    f"Elevated lag-1 autocorrelation observed (H={h_ac:.4f}, V={v_ac:.4f}); "
                    "indicates directional spatial continuity or interpolation."
                )
                supporting.append(f"Elevated lag-1 spatial autocorrelation (H={h_ac:.3f}, V={v_ac:.3f})")
            else:
                autocorr_obs = f"Moderate/low spatial autocorrelation observed (H={h_ac:.4f}, V={v_ac:.4f})."

    # Spatial Heterogeneity Index (SHI) Observation (Descriptive feature)
    shi_obs = "SHI unavailable"
    is_elevated_shi = False
    if residual_result and residual_result.spatial_descriptors:
        shi_val = residual_result.spatial_descriptors.spatial_heterogeneity_index
        if shi_val is not None:
            shi_obs = f"Spatial Heterogeneity Index is {shi_val:.4f} (describes dispersion of local block variance)."
            if shi_val > 3.0:
                is_elevated_shi = True
                supporting.append(f"Elevated block variance dispersion (SHI = {shi_val:.2f})")

    # Dynamic Range Clipping Observation (Descriptive feature)
    dyn_obs = "Dynamic range clipping unavailable"
    if pixel_result and pixel_result.histograms:
        luma_hist = next((h for h in pixel_result.histograms if h.channel_name == "luma"), pixel_result.histograms[0])
        c0 = luma_hist.clipping_fraction_0
        c255 = luma_hist.clipping_fraction_255
        dyn_obs = f"Boundary clipping: Shadow(0)={c0 * 100:.2f}%, Highlight(255)={c255 * 100:.2f}%."
        if (c0 > 0.02 or c255 > 0.02):
            supporting.append(f"Boundary clipping observed (0: {c0*100:.1f}%, 255: {c255*100:.1f}%)")

    # Candidate Status Determination (Multi-source corroboration rule; no arbitrary weighted scoring)
    # A candidate screenshot assessment strictly requires MULTIPLE independent corroborating dimensions:
    # (e.g. contextual display geometry match AND residual noise suppression AND directional autocorrelation / SHI)
    if geom_state == SignalState.PRESENT and is_residual_suppressed and (is_high_autocorr or is_elevated_shi):
        candidate_status = RecaptureCandidateStatus.CANDIDATE_RECAPTURE_INDICATORS_OBSERVED
    elif is_residual_suppressed and is_high_autocorr and is_elevated_shi:
        # Screen capture of window or desktop region without full-screen geometry match
        candidate_status = RecaptureCandidateStatus.CANDIDATE_RECAPTURE_INDICATORS_OBSERVED
    elif geom_state == SignalState.PRESENT or is_residual_suppressed or is_high_autocorr:
        # Isolated single indicators are weak and inconclusive on their own
        candidate_status = RecaptureCandidateStatus.INCONCLUSIVE
    else:
        candidate_status = RecaptureCandidateStatus.NO_STRONG_RECAPTURE_INDICATORS

    limitations = [
        "Candidate indicators are descriptive features and do not prove screenshot/recapture with certainty.",
        "Matching a common display aspect ratio is a contextual feature; authentic camera photos can share aspect ratios or be cropped.",
        "Suppressed residual noise can result from strong digital filtering, flat compositions, or synthetic graphics.",
        "Screenshots of authentic photos, AI imagery, or documents can be subsequently resized or cropped, altering geometric and residual signatures.",
    ]

    return ScreenshotIndicatorsResult(
        display_geometry_match=geom_state,
        matched_display_profile=matched_profile,
        dimensions=dims_str,
        aspect_ratio=aspect_ratio,
        residual_observation=residual_obs,
        autocorrelation_observation=autocorr_obs,
        shi_observation=shi_obs,
        dynamic_range_observation=dyn_obs,
        candidate_status=candidate_status,
        supporting_observations=supporting,
        limitations=limitations,
    )
