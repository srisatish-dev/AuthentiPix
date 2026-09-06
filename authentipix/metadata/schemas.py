"""Pydantic data schemas for AuthentiPix Phase 1 (Metadata & Provenance).

Implements the approved 3-dimensional evidence model:
- EvidenceCategory
- EvidenceStrength
- Validation Status Code (e.g. C2PA_VALID_TRUSTED, CONTENT_BINDING_FAILURE)
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvidenceCategory(str, Enum):
    """Categorizes the domain of an evidence finding."""
    METADATA_STRUCTURE = "METADATA_STRUCTURE"
    TEMPORAL_CHAIN = "TEMPORAL_CHAIN"
    SOFTWARE_WORKFLOW = "SOFTWARE_WORKFLOW"
    DIGITAL_PROVENANCE = "DIGITAL_PROVENANCE"
    CONSISTENCY_ANOMALY = "CONSISTENCY_ANOMALY"


class EvidenceStrength(str, Enum):
    """Evidentiary significance rating (NEUTRAL, LOW, MEDIUM, HIGH).

    Does NOT represent certainty of fraud or numeric confidence float.
    """
    NEUTRAL = "NEUTRAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class EvidenceItem(BaseModel):
    """Standardized representation of a single forensic evidence item.

    Strictly separates raw observation from interpretation and limitations.
    """
    evidence_id: str = Field(description="Unique UUID for this evidence finding")
    pillar: str = Field(default="metadata_provenance", description="Originating evidence pillar")
    analyzer_id: str = Field(default="pillar1_metadata_analyzer", description="Specific analyzer identifier")

    # 3 Orthogonal Dimensions
    category: EvidenceCategory
    evidence_strength: EvidenceStrength
    validation_status_code: Optional[str] = Field(
        default=None,
        description="Specific validation status (e.g. C2PA_VALID_TRUSTED, CONTENT_BINDING_FAILURE, DIMENSION_MISMATCH)"
    )

    rule_id: Optional[str] = Field(default=None, description="ID of consistency rule executed, if applicable")
    affected_fields: List[str] = Field(default_factory=list, description="Metadata fields involved in this finding")

    # Separation: Observation vs Interpretation
    raw_observation: Dict[str, Any] = Field(description="Exact extracted raw tag values")
    finding_summary: str = Field(description="Factual summary of the finding")
    interpretation: str = Field(description="Forensic-safe interpretation of potential implications")

    limitations: List[str] = Field(default_factory=list, description="Explicit forensic limitations of this finding")
    supporting_values: Dict[str, Any] = Field(default_factory=dict, description="Derived comparison values")
    analyzer_version: str = Field(default="1.0.0", description="Analyzer version that generated this item")
    reproducibility_info: Dict[str, Any] = Field(default_factory=dict, description="Execution context metadata")


class NormalizedMetadata(BaseModel):
    """Canonical representation of metadata across EXIF, XMP, IPTC, and container tags."""
    make: Optional[str] = None
    model: Optional[str] = None
    software: Optional[str] = None
    orientation: Optional[int] = None

    # Timestamps (ISO strings preserving timezone or uncalibrated offset)
    datetime_original: Optional[str] = None
    datetime_create: Optional[str] = None
    datetime_modify: Optional[str] = None
    offset_time_original: Optional[str] = None
    offset_time: Optional[str] = None
    subsec_time_original: Optional[str] = None

    # Camera capture parameters
    exposure_time: Optional[str] = None
    f_number: Optional[float] = None
    iso: Optional[int] = None
    focal_length: Optional[float] = None
    focal_length_35mm: Optional[float] = None
    flash: Optional[str] = None
    metering_mode: Optional[str] = None

    # Spatial GPS
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    gps_datetime: Optional[str] = None

    # Structural dimensions
    exif_width: Optional[int] = None
    exif_height: Optional[int] = None
    raster_width: Optional[int] = None
    raster_height: Optional[int] = None

    # Color & format
    color_space: Optional[str] = None
    icc_profile_name: Optional[str] = None
    mime_type: Optional[str] = None
    file_size_bytes: Optional[int] = None

    # Workflow & History
    creator_tool: Optional[str] = None
    document_id: Optional[str] = None
    instance_id: Optional[str] = None
    history_actions: List[Dict[str, Any]] = Field(default_factory=list)

    # IPTC
    byline: Optional[str] = None
    copyright_notice: Optional[str] = None
    credit: Optional[str] = None
    digital_creation_date: Optional[str] = None

    # Raw tag bag
    raw_tags: Dict[str, Any] = Field(default_factory=dict)


class AnalyzerOutput(BaseModel):
    """Top-level output emitted by MetadataAnalyzer for Evidence Fusion consumption."""
    analyzer_id: str = "pillar1_metadata_analyzer"
    analyzer_version: str = "1.0.0"
    execution_time_ms: float = 0.0
    raw_observations: Dict[str, Any] = Field(default_factory=dict)
    normalized_features: Dict[str, Any] = Field(default_factory=dict)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    reproducibility_metadata: Dict[str, Any] = Field(default_factory=dict)
