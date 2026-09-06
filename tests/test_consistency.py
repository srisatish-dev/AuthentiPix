"""Unit tests for ConsistencyEngine rules."""

from authentipix.metadata.consistency import ConsistencyEngine
from authentipix.metadata.schemas import NormalizedMetadata


def test_rule_001_dimension_mismatch():
    engine = ConsistencyEngine()
    norm = NormalizedMetadata(
        exif_width=4000,
        exif_height=3000,
        raster_width=2000,
        raster_height=1500,
    )

    item = engine.rule_001_exif_vs_raster_dimensions(norm)
    assert item is not None
    assert item.rule_id == "RULE_001"
    assert item.evidence_strength == "HIGH"
    assert item.validation_status_code == "DIMENSION_MISMATCH"
    assert "Does not prove malicious editing" in item.limitations[0]


def test_rule_001_dimension_match():
    engine = ConsistencyEngine()
    norm = NormalizedMetadata(
        exif_width=1920,
        exif_height=1080,
        raster_width=1920,
        raster_height=1080,
    )

    item = engine.rule_001_exif_vs_raster_dimensions(norm)
    assert item is None  # No anomaly triggered


def test_rule_002_inter_namespace_discrepancy():
    engine = ConsistencyEngine()
    norm = NormalizedMetadata(
        software="Canon Firmware 1.2",
        creator_tool="Adobe Photoshop 25.0",
    )

    item = engine.rule_002_inter_namespace_discrepancy(norm)
    assert item is not None
    assert item.rule_id == "RULE_002"
    assert item.evidence_strength == "MEDIUM"
    assert item.validation_status_code == "NAMESPACE_DISCREPANCY"


def test_rule_003_temporal_inversion():
    engine = ConsistencyEngine()
    norm = NormalizedMetadata(
        datetime_original="2026:05:14 12:00:00",
        datetime_modify="2026:05:14 10:00:00",  # Modify precedes Original
    )

    item = engine.rule_003_temporal_chronology(norm)
    assert item is not None
    assert item.rule_id == "RULE_003"
    assert item.validation_status_code == "TEMPORAL_INVERSION"


def test_rule_004_software_agent_presence():
    engine = ConsistencyEngine()
    norm = NormalizedMetadata(
        software="Adobe Lightroom 6.0",
    )

    item = engine.rule_004_software_agent_presence(norm)
    assert item is not None
    assert item.rule_id == "RULE_004"
    assert item.evidence_strength == "LOW"
    assert "Does not prove malicious or fraudulent manipulation" in item.limitations[1]


def test_rule_007_thumbnail_aspect_mismatch():
    engine = ConsistencyEngine()
    norm = NormalizedMetadata(
        raster_width=1920,
        raster_height=1080,  # Aspect 16:9 = 1.777
    )
    preview_data = {
        "thumbnail_present": True,
        "thumbnail_width": 160,
        "thumbnail_height": 160,  # Aspect 1:1 = 1.0 (mismatch > 0.15)
    }

    item = engine.rule_007_embedded_preview_alignment(norm, preview_data)
    assert item is not None
    assert item.rule_id == "RULE_007"
    assert item.evidence_strength == "HIGH"
    assert item.validation_status_code == "THUMBNAIL_ASPECT_MISMATCH"
