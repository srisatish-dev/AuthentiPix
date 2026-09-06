"""AuthentiPix Pillar 1 — Metadata & Provenance Module."""

from authentipix.metadata.analyzer import MetadataAnalyzer
from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.schemas import (
    AnalyzerOutput,
    EvidenceCategory,
    EvidenceItem,
    EvidenceStrength,
    NormalizedMetadata,
)
from authentipix.metadata.security import SecurityPolicyConfig, SecurityViolationError

__all__ = [
    "MetadataAnalyzer",
    "AnalysisContext",
    "AnalyzerOutput",
    "EvidenceItem",
    "EvidenceCategory",
    "EvidenceStrength",
    "NormalizedMetadata",
    "SecurityPolicyConfig",
    "SecurityViolationError",
]
