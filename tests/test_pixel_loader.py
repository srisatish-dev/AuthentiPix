"""Unit tests for safe image loading and Pillow mode policy (authentipix.pixel.loader)."""

import io
import pytest
import numpy as np
from PIL import Image

from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.security import SecurityViolationError
from authentipix.pixel.loader import PixelImageLoader, UnsupportedImageModeError


def _make_test_image_bytes(mode: str = "RGB", size: tuple = (50, 40), color="red") -> bytes:
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    fmt = "PNG" if mode in ["RGBA", "P", "1", "LA"] else "JPEG"
    if mode == "CMYK":
        fmt = "JPEG"
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_load_rgb_image():
    img_bytes = _make_test_image_bytes(mode="RGB", size=(100, 50))
    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/jpeg")
    loader = PixelImageLoader()
    arr, props = loader.load_pixel_array(ctx)

    assert arr.shape == (50, 100, 3)
    assert props.width == 100
    assert props.height == 50
    assert props.pixel_count == 5000
    assert props.aspect_ratio == 2.0
    assert props.native_mode == "RGB"
    assert props.channel_count == 3
    assert props.has_alpha is False
    assert props.was_palette_converted is False


def test_load_rgba_image():
    img_bytes = _make_test_image_bytes(mode="RGBA", size=(60, 40), color=(255, 0, 0, 128))
    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/png")
    loader = PixelImageLoader()
    arr, props = loader.load_pixel_array(ctx)

    assert arr.shape == (40, 60, 4)
    assert props.channel_count == 4
    assert props.has_alpha is True


def test_load_grayscale_l_image():
    img_bytes = _make_test_image_bytes(mode="L", size=(30, 20), color=128)
    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/jpeg")
    loader = PixelImageLoader()
    arr, props = loader.load_pixel_array(ctx)

    # Verify 2D -> 3D normalization (H, W, 1)
    assert arr.shape == (20, 30, 1)
    assert props.channel_count == 1
    assert props.native_mode == "L"


def test_load_palette_image():
    img = Image.new("P", (30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/png")
    loader = PixelImageLoader()
    arr, props = loader.load_pixel_array(ctx)

    assert props.native_mode == "P"
    assert props.was_palette_converted is True
    assert arr.shape == (30, 30, 3)


def test_load_bilevel_1_image():
    img = Image.new("1", (20, 20), color=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/png")
    loader = PixelImageLoader()
    arr, props = loader.load_pixel_array(ctx)

    assert props.native_mode == "1"
    assert props.was_bilevel_converted is True
    assert arr.shape == (20, 20, 1)


def test_load_invalid_bytes():
    bad_bytes = b"NOT_AN_IMAGE_PAYLOAD"
    ctx = AnalysisContext.from_bytes(bad_bytes, mime_type="image/jpeg")
    loader = PixelImageLoader()
    with pytest.raises(UnsupportedImageModeError):
        loader.load_pixel_array(ctx)


def test_load_file_size_security_violation():
    img_bytes = _make_test_image_bytes()
    ctx = AnalysisContext.from_bytes(img_bytes, mime_type="image/jpeg")
    ctx.file_size_bytes = 100_000_000  # Exceed 50 MB limit
    loader = PixelImageLoader()
    with pytest.raises(SecurityViolationError):
        loader.load_pixel_array(ctx)
