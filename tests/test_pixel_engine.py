"""Integration tests for PixelForensicsEngine pipeline (authentipix.pixel.engine)."""

import io
import os
import pytest
from PIL import Image

from authentipix.metadata.context import AnalysisContext
from authentipix.pixel.engine import PixelForensicsEngine


def _make_rgb_image_bytes(width=100, height=80) -> bytes:
    img = Image.new("RGB", (width, height), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_engine_analyze_rgb_image():
    img_bytes = _make_rgb_image_bytes(120, 90)
    ctx = AnalysisContext.from_bytes(img_bytes, image_id="test_rgb_001", mime_type="image/jpeg")

    engine = PixelForensicsEngine()
    result = engine.analyze(ctx)

    assert result.analyzer_id == "pillar2_basic_pixel_analyzer"
    assert result.properties.width == 120
    assert result.properties.height == 90
    assert result.properties.pixel_count == 10800
    assert result.properties.native_mode == "RGB"
    assert result.properties.channel_count == 3

    # Check statistics
    assert len(result.channel_statistics) == 3
    assert result.luma_statistics is not None
    assert result.intensity_statistics is not None

    # Check histograms
    assert len(result.histograms) == 4  # 3 channels (R, G, B) + 1 Luma
    for h in result.histograms:
        assert len(h.bins) == 256
        assert 0.0 <= h.entropy <= 8.0
        assert 0.0 <= h.comb_metric <= 1.0

    # Check spatial descriptors
    assert result.spatial_descriptors is not None
    assert result.spatial_descriptors.edge_threshold == 30.0

    # Check reproducibility metadata
    assert "input_sha256" in result.reproducibility_metadata
    assert result.reproducibility_metadata["luma_standard"] == "BT.709"


def test_engine_analyze_grayscale_image():
    img = Image.new("L", (50, 50), color=100)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    ctx = AnalysisContext.from_bytes(img_bytes, image_id="test_gray_001", mime_type="image/jpeg")
    engine = PixelForensicsEngine()
    result = engine.analyze(ctx)

    assert result.properties.native_mode == "L"
    assert result.properties.channel_count == 1
    assert len(result.channel_statistics) == 1
    assert result.luma_statistics is None
    assert result.intensity_statistics is None
    assert len(result.histograms) == 1


def test_engine_analyze_corrupt_bytes():
    bad_bytes = b"CORRUPT_NOT_AN_IMAGE"
    ctx = AnalysisContext.from_bytes(bad_bytes, image_id="corrupt_001", mime_type="image/jpeg")

    engine = PixelForensicsEngine()
    result = engine.analyze(ctx)

    assert len(result.errors) > 0
    assert len(result.evidence_items) == 1
    assert result.evidence_items[0].validation_status_code == "UNSUPPORTED_IMAGE_MODE"


def test_engine_analyze_real_samples():
    # If sample1.png or sample2.jpg exist in workspace root, run engine against them
    sample1_path = "sample1.png"
    if os.path.exists(sample1_path):
        ctx = AnalysisContext.from_file(sample1_path, mime_type="image/png")
        engine = PixelForensicsEngine()
        result = engine.analyze(ctx)

        assert result.properties.width > 0
        assert result.properties.height > 0
        assert result.spatial_descriptors is not None
