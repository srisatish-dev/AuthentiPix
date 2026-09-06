"""Unit tests for Phase 1 Pydantic schemas."""

from authentipix.metadata.schemas import (
    AnalyzerOutput,
    EvidenceCategory,
    EvidenceItem,
    EvidenceStrength,
    NormalizedMetadata,
)


def test_evidence_item_schema():
    item = EvidenceItem(
        evidence_id="ev-001",
        category=EvidenceCategory.CONSISTENCY_ANOMALY,
        evidence_strength=EvidenceStrength.HIGH,
        validation_status_code="DIMENSION_MISMATCH",
        rule_id="RULE_001",
        affected_fields=["exif_width", "raster_width"],
        raw_observation={"exif": 4000, "raster": 2000},
        finding_summary="Dimension mismatch detected",
        interpretation="Indicates possible cropping or resizing.",
        limitations=["Does not prove malicious editing."],
    )

    d = item.model_dump()
    assert d["evidence_id"] == "ev-001"
    assert d["category"] == "CONSISTENCY_ANOMALY"
    assert d["evidence_strength"] == "HIGH"
    assert d["validation_status_code"] == "DIMENSION_MISMATCH"
    assert "confidence_score" not in d  # Verify numeric confidence float is absent


def test_normalized_metadata_defaults():
    norm = NormalizedMetadata()
    assert norm.make is None
    assert norm.datetime_original is None
    assert norm.exif_width is None
    assert norm.raster_width is None
    assert norm.raw_tags == {}


def test_analyzer_output_serialization():
    out = AnalyzerOutput(analyzer_id="pillar1_metadata_analyzer")
    d = out.model_dump()
    assert d["analyzer_id"] == "pillar1_metadata_analyzer"
    assert isinstance(d["evidence_items"], list)
    assert isinstance(d["raw_observations"], dict)
