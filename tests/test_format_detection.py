"""Unit tests for AuthentiPix actual format detection independent of extension."""

import io
import pytest
from PIL import Image
import numpy as np

from authentipix.robustness.format_detection import detect_actual_format, detect_actual_format_from_bytes


def create_minimal_image_bytes(fmt: str = "JPEG", size=(100, 100)) -> bytes:
    """Generates valid minimal in-memory image bytes for a specified format."""
    arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr)
    bio = io.BytesIO()
    img.save(bio, format=fmt)
    return bio.getvalue()


def test_jpeg_actual_format_detection():
    """Confirms valid JPEG byte stream is detected as JPEG."""
    jpeg_bytes = create_minimal_image_bytes("JPEG")
    res = detect_actual_format_from_bytes(jpeg_bytes, filename="photo.jpg")

    assert res.actual_format == "JPEG"
    assert res.mime_type == "image/jpeg"
    assert res.filename_extension == ".jpg"
    assert res.format_mismatch_detected is False
    assert res.is_supported_format is True


def test_png_actual_format_detection():
    """Confirms valid PNG byte stream is detected as PNG."""
    png_bytes = create_minimal_image_bytes("PNG")
    res = detect_actual_format_from_bytes(png_bytes, filename="graphic.png")

    assert res.actual_format == "PNG"
    assert res.mime_type == "image/png"
    assert res.filename_extension == ".png"
    assert res.format_mismatch_detected is False
    assert res.is_supported_format is True


def test_format_mismatch_jpeg_with_png_extension():
    """Detects JPEG payload disguised with a .png filename extension."""
    jpeg_bytes = create_minimal_image_bytes("JPEG")
    res = detect_actual_format_from_bytes(jpeg_bytes, filename="sneaky.png")

    assert res.actual_format == "JPEG"
    assert res.filename_extension == ".png"
    assert res.format_mismatch_detected is True
    assert "FORMAT MISMATCH DETECTED" in res.details


def test_format_mismatch_png_with_jpg_extension():
    """Detects PNG payload disguised with a .jpg filename extension."""
    png_bytes = create_minimal_image_bytes("PNG")
    res = detect_actual_format_from_bytes(png_bytes, filename="sneaky.jpg")

    assert res.actual_format == "PNG"
    assert res.filename_extension == ".jpg"
    assert res.format_mismatch_detected is True


def test_unknown_format_handling():
    """Non-image bytes are safely reported as UNKNOWN without crashing."""
    junk_bytes = b"NOT_AN_IMAGE_FILE_AT_ALL_12345678"
    res = detect_actual_format_from_bytes(junk_bytes, filename="corrupt.dat")

    assert res.actual_format == "UNKNOWN"
    assert res.format_mismatch_detected is False
    assert res.is_supported_format is False


def test_file_not_found_handling(tmp_path):
    """Missing file is safely handled with appropriate status."""
    non_existent = str(tmp_path / "does_not_exist.jpg")
    res = detect_actual_format(non_existent)

    assert res.actual_format == "NOT_FOUND"
    assert res.is_supported_format is False
