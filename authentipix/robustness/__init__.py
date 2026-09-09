"""AuthentiPix Robustness, Format & Screenshot Handling Package.

Provides format detection, transformation awareness, candidate screenshot/recapture
indicators, and qualified evidence synthesis.
"""

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
from authentipix.robustness.format_detection import detect_actual_format, detect_actual_format_from_bytes
from authentipix.robustness.transformation_awareness import analyze_transformations
from authentipix.robustness.screenshot_indicators import analyze_screenshot_indicators
from authentipix.robustness.evidence import synthesize_robustness_evidence

__all__ = [
    "EvidenceTier",
    "FormatDetectionResult",
    "RecaptureCandidateStatus",
    "RobustnessReport",
    "ScreenshotIndicatorsResult",
    "SignalState",
    "SynthesizedAssessmentResult",
    "TransformationAwarenessResult",
    "detect_actual_format",
    "detect_actual_format_from_bytes",
    "analyze_transformations",
    "analyze_screenshot_indicators",
    "synthesize_robustness_evidence",
]
