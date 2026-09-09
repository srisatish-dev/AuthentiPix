"""Actual binary image format detection independent of filename extension.

Inspects file magic/signature bytes and structural headers to determine the true
encoding format and detects container/extension mismatches.
"""

import io
import os
from typing import Dict, List, Optional, Tuple
from PIL import Image

from authentipix.robustness.schemas import FormatDetectionResult

# Canonical format definitions and magic byte signatures
MAGIC_SIGNATURES: List[Tuple[str, bytes, int]] = [
    ("PNG", b"\x89PNG\r\n\x1a\n", 0),
    ("JPEG", b"\xff\xd8\xff", 0),
    ("GIF", b"GIF87a", 0),
    ("GIF", b"GIF89a", 0),
    ("TIFF", b"II*\x00", 0),  # Little-endian TIFF
    ("TIFF", b"MM\x00*", 0),  # Big-endian TIFF
    ("BMP", b"BM", 0),
]

FORMAT_EXTENSIONS: Dict[str, List[str]] = {
    "JPEG": [".jpg", ".jpeg", ".jpe", ".jfif"],
    "PNG": [".png"],
    "WEBP": [".webp"],
    "TIFF": [".tif", ".tiff"],
    "BMP": [".bmp"],
    "GIF": [".gif"],
}

FORMAT_MIME_TYPES: Dict[str, str] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "TIFF": "image/tiff",
    "BMP": "image/bmp",
    "GIF": "image/gif",
}


def sniff_magic_bytes(header: bytes) -> Optional[str]:
    """Sniffs raw binary header bytes for known image file signatures."""
    if len(header) < 4:
        return None

    for fmt, sig, offset in MAGIC_SIGNATURES:
        if header[offset : offset + len(sig)] == sig:
            return fmt

    # WebP requires checking RIFF header and WEBP chunk tag
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "WEBP"

    return None


def detect_actual_format_from_bytes(
    file_bytes: bytes,
    filename: str = "asset",
) -> FormatDetectionResult:
    """Detects actual format from raw byte payload."""
    ext = os.path.splitext(filename)[1].lower()
    ext_str = ext if ext else "(none)"
    header = file_bytes[:32]
    sniffed_fmt = sniff_magic_bytes(header)
    hex_sig = header[:8].hex().upper()

    # Structural verification with Pillow
    pillow_fmt = None
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            pillow_fmt = img.format
    except Exception:
        pillow_fmt = None

    # Determine resolved actual format
    if sniffed_fmt and pillow_fmt:
        actual_fmt = sniffed_fmt if sniffed_fmt.upper() == pillow_fmt.upper() else pillow_fmt.upper()
    elif sniffed_fmt:
        actual_fmt = sniffed_fmt
    elif pillow_fmt:
        actual_fmt = pillow_fmt.upper()
    else:
        actual_fmt = "UNKNOWN"

    is_supported = actual_fmt in FORMAT_MIME_TYPES
    mime_type = FORMAT_MIME_TYPES.get(actual_fmt, f"image/{actual_fmt.lower()}")

    # Check for extension mismatch
    allowed_exts = FORMAT_EXTENSIONS.get(actual_fmt, [])
    if actual_fmt == "UNKNOWN":
        mismatch = False
        details = "Unrecognized binary image format; magic bytes did not match known signatures."
    elif not allowed_exts:
        mismatch = False
        details = f"Detected format {actual_fmt}; no standard extension mapping defined."
    else:
        mismatch = ext not in allowed_exts
        if mismatch:
            details = (
                f"FORMAT MISMATCH DETECTED: File extension '{ext_str}' contradicts actual "
                f"binary container format '{actual_fmt}' (Magic signature: {hex_sig})."
            )
        else:
            details = f"Verified binary format '{actual_fmt}' matches filename extension '{ext_str}'."

    return FormatDetectionResult(
        filename=os.path.basename(filename),
        filename_extension=ext_str,
        actual_format=actual_fmt,
        mime_type=mime_type,
        format_mismatch_detected=mismatch,
        is_supported_format=is_supported,
        magic_signature_bytes=hex_sig,
        details=details,
    )


def detect_actual_format(file_path: str) -> FormatDetectionResult:
    """Detects actual binary image format from file on disk."""
    filename = os.path.basename(file_path)
    if not os.path.exists(file_path):
        return FormatDetectionResult(
            filename=filename,
            filename_extension=os.path.splitext(filename)[1].lower() or "(none)",
            actual_format="NOT_FOUND",
            mime_type="application/octet-stream",
            format_mismatch_detected=False,
            is_supported_format=False,
            magic_signature_bytes="",
            details=f"File not found at path: {file_path}",
        )

    try:
        with open(file_path, "rb") as f:
            header_and_data = f.read(65536)  # Read up to 64KB for inspection
            f.seek(0)
            # Check full file if small, else read from file path with Pillow
            file_bytes = f.read() if os.path.getsize(file_path) < 10 * 1024 * 1024 else header_and_data
        return detect_actual_format_from_bytes(file_bytes, filename=filename)
    except Exception as e:
        return FormatDetectionResult(
            filename=filename,
            filename_extension=os.path.splitext(filename)[1].lower() or "(none)",
            actual_format="ERROR",
            mime_type="application/octet-stream",
            format_mismatch_detected=False,
            is_supported_format=False,
            magic_signature_bytes="",
            details=f"Error reading file for format inspection: {e}",
        )
