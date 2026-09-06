"""Unit tests for AnalysisContext."""

import os
import tempfile
import pytest

from authentipix.metadata.context import AnalysisContext


def test_analysis_context_from_bytes():
    raw_data = b"AuthentiPix test asset content"
    ctx = AnalysisContext.from_bytes(raw_data, image_id="asset_test_123", mime_type="image/jpeg")

    assert ctx.image_id == "asset_test_123"
    assert ctx.mime_type == "image/jpeg"
    assert ctx.file_size_bytes == len(raw_data)
    assert len(ctx.sha256_hash) == 64
    assert ctx.get_bytes() == raw_data


def test_analysis_context_from_file():
    raw_data = b"Sample file content for hash testing"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(raw_data)
        tmp_path = tmp.name

    try:
        ctx = AnalysisContext.from_file(tmp_path, mime_type="image/jpeg")
        assert ctx.file_size_bytes == len(raw_data)
        assert len(ctx.sha256_hash) == 64
        assert ctx.get_bytes() == raw_data
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_analysis_context_file_not_found():
    with pytest.raises(FileNotFoundError):
        AnalysisContext.from_file("non_existent_file_path_12345.jpg")
