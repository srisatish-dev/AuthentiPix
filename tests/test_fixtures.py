"""Test fixtures and synthetic image generators for AuthentiPix ground-truth testing."""

import io
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageDraw


def create_synthetic_image(
    size: Tuple[int, int] = (800, 600),
    color: str = "blue",
    fmt: str = "JPEG",
    exif_data: Optional[Dict[int, Any]] = None,
) -> bytes:
    """Creates a basic synthetic in-memory image byte payload."""
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "AuthentiPix Test Asset", fill="white")

    buf = io.BytesIO()
    if exif_data is not None:
        exif = img.getexif()
        for k, v in exif_data.items():
            exif[k] = v
        img.save(buf, format=fmt, exif=exif)
    else:
        img.save(buf, format=fmt)

    return buf.getvalue()


def create_image_with_custom_exif(
    size: Tuple[int, int] = (1920, 1080),
    make: str = "Canon",
    model: str = "EOS R5",
    software: str = "Canon Firmware 1.5.0",
    datetime_orig: str = "2026:05:14 10:15:30",
    datetime_mod: str = "2026:05:14 10:15:30",
    exif_dim: Optional[Tuple[int, int]] = None,
) -> bytes:
    """Generates a synthetic JPEG containing specific camera EXIF tags."""
    img = Image.new("RGB", size, color="darkgreen")
    exif = img.getexif()

    # 271 = Make, 272 = Model, 305 = Software, 306 = ModifyDate
    exif[271] = make
    exif[272] = model
    exif[305] = software
    exif[306] = datetime_mod

    # ExifIFD sub-tags: 36867 = DateTimeOriginal, 40962 = PixelXDimension, 40963 = PixelYDimension
    exif_ifd = exif.get_ifd(0x8769)
    exif_ifd[36867] = datetime_orig

    if exif_dim:
        exif_ifd[40962] = exif_dim[0]
        exif_ifd[40963] = exif_dim[1]
    else:
        exif_ifd[40962] = size[0]
        exif_ifd[40963] = size[1]

    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def create_image_with_xmp(
    size: Tuple[int, int] = (600, 400),
    xmp_xml: str = "<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'><rdf:Description xmlns:xmp='http://ns.adobe.com/xap/1.0/' xmp:CreatorTool='Adobe Photoshop 25.0'/></rdf:RDF></x:xmpmeta>",
) -> bytes:
    """Generates a synthetic PNG image with embedded XMP metadata payload."""
    from PIL import PngImagePlugin
    img = Image.new("RGB", size, color="purple")
    info = PngImagePlugin.PngInfo()
    info.add_text("XML:com.adobe.xmp", xmp_xml)

    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()
