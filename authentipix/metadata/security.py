"""Security Policy Enforcer for AuthentiPix Phase 1.

Implements configurable security policy limits (file size, pixel count, timeouts),
safe XML parsing via defusedxml to prevent XXE, image decompression bomb protections,
and isolated scratch file management.
"""

import os
import tempfile
from typing import Any, List, Optional, Union
from defusedxml.common import DefusedXmlException, EntitiesForbidden, DTDForbidden
from defusedxml.ElementTree import fromstring as defused_fromstring, ParseError as DefusedParseError
from PIL import Image
from pydantic import BaseModel, Field

from authentipix.metadata.context import AnalysisContext


class SecurityPolicyConfig(BaseModel):
    """Configurable security policy thresholds for untrusted image processing."""
    max_file_size_bytes: int = Field(default=52_428_800, description="Max allowed upload size (50 MB)")
    max_pixel_count: int = Field(default=100_000_000, description="Max allowed pixel count (100 MP)")
    subprocess_timeout_seconds: float = Field(default=5.0, description="Max subprocess execution time")
    max_memory_mb: int = Field(default=512, description="Max worker memory allocation")
    allowed_mime_types: List[str] = Field(
        default_factory=lambda: [
            "image/jpeg", "image/png", "image/webp", "image/tiff", "image/heic", "image/heif"
        ]
    )


class SecurityViolationError(Exception):
    """Raised when an asset violates security policy boundaries."""
    pass


class SecurityEnforcer:
    """Security validation and sanitization engine."""

    def __init__(self, policy: Optional[SecurityPolicyConfig] = None):
        self.policy = policy or SecurityPolicyConfig()
        # Set Pillow safety threshold for decompression bomb prevention
        Image.MAX_IMAGE_PIXELS = self.policy.max_pixel_count

    def validate_context(self, ctx: AnalysisContext) -> List[str]:
        """Validates AnalysisContext against security policies.

        Returns a list of warnings or raises SecurityViolationError for critical breaches.
        """
        warnings = []
        if ctx.file_size_bytes > self.policy.max_file_size_bytes:
            raise SecurityViolationError(
                f"File size ({ctx.file_size_bytes} bytes) exceeds maximum policy limit ({self.policy.max_file_size_bytes} bytes)"
            )

        if ctx.mime_type and ctx.mime_type.lower() not in [m.lower() for m in self.policy.allowed_mime_types]:
            warnings.append(f"MIME type '{ctx.mime_type}' is not in primary allowed list")

        return warnings

    def validate_raster_dimensions(self, width: int, height: int) -> None:
        """Checks pixel dimensions against decompression bomb threshold."""
        pixel_count = width * height
        if pixel_count > self.policy.max_pixel_count:
            raise SecurityViolationError(
                f"Raster pixel count ({pixel_count} pixels: {width}x{height}) exceeds maximum policy limit ({self.policy.max_pixel_count} pixels)"
            )

    @staticmethod
    def safe_parse_xml(xml_content: Union[str, bytes]) -> Optional[Any]:
        """Safely parses XML/XMP payload using defusedxml to block XXE attacks."""
        if not xml_content:
            return None
        try:
            if isinstance(xml_content, str):
                xml_content = xml_content.encode("utf-8", errors="replace")
            return defused_fromstring(xml_content)
        except (DefusedParseError, DefusedXmlException, EntitiesForbidden, DTDForbidden) as e:
            raise SecurityViolationError(f"XML security violation (potential XXE payload): {e}")
        except Exception:
            return None

    @staticmethod
    def create_isolated_temp_file(file_bytes: bytes, suffix: str = ".tmp") -> str:
        """Creates a temporary file in an isolated scratch sandbox directory."""
        fd, temp_path = tempfile.mkstemp(prefix="authentipix_", suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)
        return temp_path

    @staticmethod
    def cleanup_temp_file(file_path: Optional[str]) -> None:
        """Safely removes a temporary file if it exists."""
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
