"""Pydantic data schemas for AuthentiPix Robustness, Format & Screenshot Handling."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SignalState(str, Enum):
    """Four-state qualification model for forensic signals."""
    PRESENT = "PRESENT"
    NOT_DETECTED = "NOT_DETECTED"
    POTENTIALLY_ALTERED_OR_DESTROYED = "POTENTIALLY_ALTERED_OR_DESTROYED"
    UNAVAILABLE = "UNAVAILABLE"


class EvidenceTier(str, Enum):
    """Calibrated evidence hierarchy tiers."""
    CONFIRMED = "CONFIRMED"
    CAMERA_CONSISTENT = "CAMERA_CONSISTENT"
    RECAPTURE_LIKE = "RECAPTURE_LIKE"
    INCONCLUSIVE = "INCONCLUSIVE"


class RecaptureCandidateStatus(str, Enum):
    """Candidate status for screenshot / recapture characteristics."""
    NO_STRONG_RECAPTURE_INDICATORS = "NO_STRONG_RECAPTURE_INDICATORS"
    CANDIDATE_RECAPTURE_INDICATORS_OBSERVED = "CANDIDATE_RECAPTURE_INDICATORS_OBSERVED"
    INCONCLUSIVE = "INCONCLUSIVE"


class FormatDetectionResult(BaseModel):
    """Container and binary format inspection findings."""
    filename: str = Field(description="Name of the analyzed file")
    filename_extension: str = Field(description="Original extension on disk (e.g. .jpg, .png)")
    actual_format: str = Field(description="Format identified from binary magic bytes / structural header (e.g. JPEG, PNG)")
    mime_type: str = Field(description="Canonical MIME type corresponding to actual_format")
    format_mismatch_detected: bool = Field(description="True if filename_extension contradicts actual_format")
    is_supported_format: bool = Field(default=True, description="True if format is safely supported for decoding")
    magic_signature_bytes: str = Field(default="", description="Hex representation or descriptor of identified magic bytes")
    details: str = Field(default="", description="Detailed narrative of format validation")


class TransformationAwarenessResult(BaseModel):
    """Assessment of prior transformations that may have altered forensic signals."""
    potential_metadata_loss: SignalState = Field(
        default=SignalState.NOT_DETECTED,
        description="Whether metadata appears stripped or destroyed by prior processing"
    )
    potential_re_encoding: SignalState = Field(
        default=SignalState.NOT_DETECTED,
        description="Whether re-encoding, format conversion, or resaving was observed"
    )
    potential_resizing: SignalState = Field(
        default=SignalState.NOT_DETECTED,
        description="Whether dimension scaling or non-native sensor resampling was observed"
    )
    potential_recapture: SignalState = Field(
        default=SignalState.NOT_DETECTED,
        description="Whether screen capture or display recapture was observed"
    )
    format_mismatch_observed: bool = Field(
        default=False,
        description="True if container mismatch indicates conversion or renaming"
    )
    detected_transformations: List[str] = Field(
        default_factory=list,
        description="List of detected or suspected transformations"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Explanatory context regarding transformation implications"
    )


class ScreenshotIndicatorsResult(BaseModel):
    """Candidate screenshot / recapture observations.
    
    Exposes measurable features without unvalidated universal thresholds.
    """
    display_geometry_match: SignalState = Field(
        default=SignalState.NOT_DETECTED,
        description="Whether dimensions / aspect ratio match known display viewport profiles"
    )
    matched_display_profile: Optional[str] = Field(
        default=None,
        description="Name or descriptor of matched display geometry profile, if any"
    )
    dimensions: str = Field(default="", description="Raster dimensions formatted as W x H")
    aspect_ratio: float = Field(default=0.0, description="Aspect ratio (width / height)")
    
    residual_observation: str = Field(
        default="",
        description="Descriptive observation of Phase 3A residual moments (e.g. robust_std)"
    )
    autocorrelation_observation: str = Field(
        default="",
        description="Descriptive observation of directional lag-1 autocorrelation"
    )
    shi_observation: str = Field(
        default="",
        description="Descriptive observation of Spatial Heterogeneity Index (SHI)"
    )
    dynamic_range_observation: str = Field(
        default="",
        description="Descriptive observation of boundary clipping (0 and 255) from Phase 2"
    )
    
    candidate_status: RecaptureCandidateStatus = Field(
        default=RecaptureCandidateStatus.INCONCLUSIVE,
        description="Collective indicator status based on corroborated observations"
    )
    supporting_observations: List[str] = Field(
        default_factory=list,
        description="Individual supporting observations contributing to the candidate status"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Forensic limitations (e.g., candidate indicators are not proof of screenshot)"
    )


class SynthesizedAssessmentResult(BaseModel):
    """Unified, qualified forensic assessment combining all evidence channels."""
    assessment: str = Field(description="Top-level assessment summary string")
    evidence_tier: EvidenceTier = Field(description="Confidence / evidence tier")
    original_source: str = Field(
        default="UNKNOWN",
        description="Assessment of original acquisition source (e.g. UNKNOWN, CAMERA_CONSISTENT_PROBABLE, DECLARED_GENERATIVE_AI)"
    )
    original_ai_provenance: str = Field(
        default="NOT_CONFIRMED",
        description="Status of AI provenance (e.g. VERIFIED_C2PA_MANIFEST, NOT_CONFIRMED)"
    )
    primary_rationale: str = Field(description="Primary forensic reason supporting the assessment")
    supporting_observations: List[str] = Field(
        default_factory=list,
        description="List of factual observations corroborating the assessment"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Explicit forensic limitations and bounded conditions"
    )


class RobustnessReport(BaseModel):
    """Top-level report model for the Robustness layer."""
    format_detection: FormatDetectionResult
    transformation_awareness: TransformationAwarenessResult
    screenshot_indicators: ScreenshotIndicatorsResult
    provenance_assessment: str
    synthesized_assessment: SynthesizedAssessmentResult
