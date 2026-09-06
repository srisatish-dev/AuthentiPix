"""Ground-Truth Automated Test Suite for AuthentiPix Phase 1 (Metadata & Provenance).

Validates forensic behavior across 8 distinct ground-truth test categories:
- Category A: Pristine Camera Images
- Category B: Metadata-Only Manipulations
- Category C: Controlled Pixel Edits
- Category D: AI-Generated Images
- Category E: Metadata-Stripped Images
- Category F: Social/Media Transformed Images
- Category G: C2PA Validation Scenarios
- Category H: Malformed & Adversarial Inputs
"""

from authentipix.metadata.analyzer import MetadataAnalyzer
from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.security import SecurityPolicyConfig, SecurityViolationError
from tests.test_fixtures import (
    create_image_with_custom_exif,
    create_image_with_xmp,
    create_synthetic_image,
)


def test_category_a_pristine_camera_image():
    """Category A: Pristine camera image with intact EXIF and matching dimensions."""
    img_bytes = create_image_with_custom_exif(
        size=(1920, 1080),
        make="Canon",
        model="EOS R5",
        software="Canon Firmware 1.5.0",
    )
    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/jpeg")
    analyzer = MetadataAnalyzer()
    out = analyzer.analyze(ctx)

    assert out.analyzer_id == "pillar1_metadata_analyzer"
    assert out.normalized_features["make"] == "Canon"
    assert out.normalized_features["model"] == "EOS R5"

    # Verify NO binary "REAL" or "FAKE" verdict is produced
    assert not hasattr(out, "verdict")
    assert not hasattr(out, "is_real")


def test_category_b_metadata_only_manipulation():
    """Category B: Metadata-only edit (modify timestamp) with unchanged pixels.

    System MUST NOT claim pixel manipulation.
    """
    img_bytes = create_image_with_custom_exif(
        size=(800, 600),
        datetime_orig="2026:05:14 12:00:00",
        datetime_mod="2026:05:14 10:00:00",  # Inverted modify date
    )
    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/jpeg")
    analyzer = MetadataAnalyzer()
    out = analyzer.analyze(ctx)

    temporal_items = [i for i in out.evidence_items if i.validation_status_code == "TEMPORAL_INVERSION"]
    assert len(temporal_items) == 1
    item = temporal_items[0]

    # Verify forensic-safe interpretation
    assert item.evidence_strength == "MEDIUM"
    assert "precedes original capture timestamp" in item.finding_summary
    assert "Temporal inconsistency detected" in item.interpretation
    # CRITICAL FORENSIC CHECK: Must NOT claim pixel manipulation!
    assert "pixel" not in item.interpretation.lower() or "not prove" in item.interpretation.lower()


def test_category_c_controlled_pixel_edits():
    """Category C: Controlled pixel edit (EXIF dimensions conflict with raster dimensions)."""
    img_bytes = create_image_with_custom_exif(
        size=(1000, 500),          # Actual raster is 1000x500
        exif_dim=(4000, 3000),      # Stored EXIF says 4000x3000
    )
    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/jpeg")
    analyzer = MetadataAnalyzer()
    out = analyzer.analyze(ctx)

    dim_items = [i for i in out.evidence_items if i.validation_status_code == "DIMENSION_MISMATCH"]
    assert len(dim_items) == 1
    item = dim_items[0]

    assert item.evidence_strength == "HIGH"
    assert "Does not prove malicious editing" in item.limitations[0]


def test_category_d_ai_generated_provenance():
    """Category D: C2PA assertion containing explicit declaration of AI generation."""
    analyzer = MetadataAnalyzer()
    c2pa_data = {
        "validation_status_code": "C2PA_VALID_TRUSTED",
        "active_manifest": {
            "title": "DALL-E 3 Asset",
            "ai_generated_declared": True,
        },
    }

    items = analyzer._generate_c2pa_evidence(c2pa_data)
    status_codes = [i.validation_status_code for i in items]
    assert "C2PA_AI_DECLARED" in status_codes
    assert "C2PA_VALID_TRUSTED" in status_codes


def test_category_e_metadata_stripped_image():
    """Category E: Image stripped of all metadata headers.

    System MUST NOT claim fake; must return inconclusive neutral item.
    """
    img_bytes = create_synthetic_image(size=(400, 300), exif_data=None)
    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/jpeg")
    analyzer = MetadataAnalyzer()
    out = analyzer.analyze(ctx)

    no_meta_items = [i for i in out.evidence_items if i.validation_status_code == "NO_METADATA"]
    assert len(no_meta_items) == 1
    item = no_meta_items[0]

    assert item.evidence_strength == "NEUTRAL"
    assert "inconclusive" in item.interpretation.lower()
    assert "does not imply fabrication or manipulation" in item.interpretation.lower()


def test_category_f_social_media_recompressed():
    """Category F: Image processed with software tag (e.g. Canva or Lightroom)."""
    img_bytes = create_image_with_custom_exif(
        software="Adobe Lightroom 6.0 (Macintosh)",
    )
    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/jpeg")
    analyzer = MetadataAnalyzer()
    out = analyzer.analyze(ctx)

    soft_items = [i for i in out.evidence_items if i.validation_status_code == "SOFTWARE_METADATA_PRESENT"]
    assert len(soft_items) == 1
    item = soft_items[0]

    assert item.evidence_strength == "LOW"
    assert "Does not prove malicious or fraudulent manipulation" in item.limitations[1]


def test_category_g_c2pa_content_binding_failure():
    """Category G: C2PA content-binding validation failure scenario."""
    analyzer = MetadataAnalyzer()
    c2pa_data = {
        "validation_status_code": "CONTENT_BINDING_FAILURE",
        "validation_results_raw": {"status": "hash_mismatch"},
    }

    items = analyzer._generate_c2pa_evidence(c2pa_data)
    assert len(items) == 1
    assert items[0].validation_status_code == "CONTENT_BINDING_FAILURE"
    assert items[0].evidence_strength == "HIGH"
    assert "Does not automatically prove malicious fraud" in items[0].limitations[0]


def test_category_h_malformed_xxe_payload_defense():
    """Category H: Malformed XML with XXE payload handling. Safe exception without crashing."""
    xxe_xml = """<?xml version="1.0"?>
    <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
    <x:xmpmeta xmlns:x="adobe:ns:meta/"><foo>&xxe;</foo></x:xmpmeta>"""

    img_bytes = create_image_with_xmp(xmp_xml=xxe_xml)
    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/png")
    analyzer = MetadataAnalyzer()

    # Must execute safely without crashing server or reading host files
    out = analyzer.analyze(ctx)
    assert isinstance(out.errors, list)
    # Check that error is logged cleanly
    assert any("XXE" in err or "XML parsing failed" in err for err in out.errors)
