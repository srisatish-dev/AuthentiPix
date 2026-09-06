"""File & Container Property Extractor for AuthentiPix.

Inspects raw container properties, raster width/height, bit depth, color space,
ICC profile information, and enforces security pixel count thresholds.
"""

import io
from typing import Any, Dict, Optional
from PIL import Image, ImageCms

from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.extractors.base import BaseExtractor, ExtractionResult
from authentipix.metadata.security import SecurityEnforcer


class FileContainerExtractor(BaseExtractor):
    """Extractor for container properties and raster dimensions."""

    @property
    def extractor_name(self) -> str:
        return "file_container_extractor"

    def _extract_internal(self, ctx: AnalysisContext) -> ExtractionResult:
        result = ExtractionResult(extractor_name=self.extractor_name)
        image_bytes = ctx.get_bytes()
        security = SecurityEnforcer()

        data: Dict[str, Any] = {
            "mime_type": ctx.mime_type,
            "file_size_bytes": ctx.file_size_bytes,
            "sha256_hash": ctx.sha256_hash,
        }

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                w, h = img.size
                # Security check for decompression bomb
                security.validate_raster_dimensions(w, h)

                data["format"] = img.format
                data["raster_width"] = w
                data["raster_height"] = h
                data["color_mode"] = img.mode

                # Inspect ICC Profile
                icc_bytes = img.info.get("icc_profile")
                if icc_bytes:
                    data["icc_profile_present"] = True
                    icc_name = self._parse_icc_profile_name(icc_bytes)
                    if icc_name:
                        data["icc_profile_name"] = icc_name
                else:
                    data["icc_profile_present"] = False

        except Exception as e:
            result.errors.append(f"File container extraction failure: {e}")
            result.success = False

        result.data = data
        return result

    def _parse_icc_profile_name(self, icc_bytes: bytes) -> Optional[str]:
        """Extracts descriptive profile name from raw ICC profile bytes."""
        try:
            profile = ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
            name = ImageCms.getProfileName(profile)
            desc = ImageCms.getProfileDescription(profile)
            return desc or name
        except Exception:
            return None
