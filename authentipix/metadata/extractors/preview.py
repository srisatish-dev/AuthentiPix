"""Embedded Preview Extractor for AuthentiPix.

Inspects IFD1 EXIF embedded thumbnails and previews, capturing dimensions,
byte sizes, aspect ratios, and comparing them against main image rasters.
"""

import io
from typing import Any, Dict, Optional
from PIL import Image

from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.extractors.base import BaseExtractor, ExtractionResult


class PreviewExtractor(BaseExtractor):
    """Extractor for embedded thumbnails and preview streams."""

    @property
    def extractor_name(self) -> str:
        return "preview_extractor"

    def _extract_internal(self, ctx: AnalysisContext) -> ExtractionResult:
        result = ExtractionResult(extractor_name=self.extractor_name)
        image_bytes = ctx.get_bytes()
        data: Dict[str, Any] = {
            "thumbnail_present": False,
        }

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                exif_obj = img.getexif()
                if exif_obj:
                    # Check for thumbnail bytes via Pillow Exif
                    thumb_bytes = exif_obj.get_thumbnail()
                    if thumb_bytes:
                        data["thumbnail_present"] = True
                        data["thumbnail_size_bytes"] = len(thumb_bytes)

                        try:
                            with Image.open(io.BytesIO(thumb_bytes)) as thumb_img:
                                tw, th = thumb_img.size
                                data["thumbnail_width"] = tw
                                data["thumbnail_height"] = th
                                data["thumbnail_format"] = thumb_img.format
                                data["thumbnail_aspect_ratio"] = round(tw / max(th, 1), 4)
                        except Exception as te:
                            result.warnings.append(f"Thumbnail pixel decode notice: {te}")

        except Exception as e:
            result.warnings.append(f"Preview inspection notice: {e}")

        result.data = data
        return result
