"""EXIF Extractor for AuthentiPix.

Parses EXIF tags using PyExifTool when ExifTool binary is available on system PATH,
falling back gracefully to pure-Python PIL/ExifRead parsers.
"""

import io
import os
from typing import Any, Dict, List, Optional
from PIL import Image, ExifTags
import exifread

from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.extractors.base import BaseExtractor, ExtractionResult
from authentipix.metadata.security import SecurityEnforcer


class ExifExtractor(BaseExtractor):
    """Extractor for EXIF data and MakerNotes."""

    @property
    def extractor_name(self) -> str:
        return "exif_extractor"

    def _extract_internal(self, ctx: AnalysisContext) -> ExtractionResult:
        result = ExtractionResult(extractor_name=self.extractor_name)

        # Attempt PyExifTool extraction if available
        pyexif_data = self._try_pyexiftool(ctx)
        if pyexif_data:
            result.data = pyexif_data.get("parsed", {})
            result.raw_tags = pyexif_data.get("raw", {})
            result.warnings.append("Extracted via ExifTool binary")
            return result

        # Pure-Python fallback using PIL and ExifRead
        result.warnings.append("ExifTool binary unavailable; executed pure-Python PIL/ExifRead fallback")
        image_bytes = ctx.get_bytes()
        parsed_data: Dict[str, Any] = {}
        raw_tags: Dict[str, Any] = {}

        # 1. PIL Exif parsing
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                exif_obj = img.getexif()
                if exif_obj:
                    for tag_id, val in exif_obj.items():
                        tag_name = ExifTags.TAGS.get(tag_id, f"Tag_{tag_id}")
                        raw_tags[f"EXIF:{tag_name}"] = str(val)

                    # Extract ExifIFD sub-tags if present
                    try:
                        exif_ifd = exif_obj.get_ifd(ExifTags.IFD.Exif)
                        for tag_id, val in exif_ifd.items():
                            tag_name = ExifTags.TAGS.get(tag_id, f"SubTag_{tag_id}")
                            raw_tags[f"EXIF:{tag_name}"] = str(val)
                    except Exception:
                        pass

                    # Extract GPS IFD sub-tags if present
                    try:
                        gps_ifd = exif_obj.get_ifd(ExifTags.IFD.GPSInfo)
                        for tag_id, val in gps_ifd.items():
                            tag_name = ExifTags.GPSTAGS.get(tag_id, f"GPS_{tag_id}")
                            raw_tags[f"GPS:{tag_name}"] = str(val)
                    except Exception:
                        pass

        except Exception as e:
            result.warnings.append(f"PIL EXIF parsing notice: {e}")

        # 2. ExifRead parsing (complements PIL, extracts MakerNotes hints & timestamps)
        try:
            tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
            for tag_key, tag_val in tags.items():
                if not tag_key.startswith("JPEG"):
                    raw_tags[f"ExifRead:{tag_key}"] = str(tag_val)
        except Exception as e:
            result.warnings.append(f"ExifRead parsing notice: {e}")

        # Populate structured data fields from raw_tags
        parsed_data = self._structure_raw_exif(raw_tags)
        result.data = parsed_data
        result.raw_tags = raw_tags
        return result

    def _try_pyexiftool(self, ctx: AnalysisContext) -> Optional[Dict[str, Any]]:
        """Attempts process-isolated extraction via pyexiftool daemon if binary is present."""
        try:
            import exiftool
            with exiftool.ExifToolHelper() as et:
                if ctx.file_path and os.path.exists(ctx.file_path):
                    metadata_list = et.get_metadata(ctx.file_path)
                else:
                    temp_path = SecurityEnforcer.create_isolated_temp_file(ctx.get_bytes(), suffix=".tmp")
                    try:
                        metadata_list = et.get_metadata(temp_path)
                    finally:
                        SecurityEnforcer.cleanup_temp_file(temp_path)

                if metadata_list and len(metadata_list) > 0:
                    raw = metadata_list[0]
                    parsed = self._structure_raw_exif(raw)
                    return {"raw": raw, "parsed": parsed}
        except Exception:
            # ExifTool binary not installed on PATH or failed spawn
            return None
        return None

    def _structure_raw_exif(self, tags: Dict[str, Any]) -> Dict[str, Any]:
        """Maps heterogeneous tag dictionaries to standard structured keys."""
        def get_field(keys: List[str]) -> Optional[Any]:
            for k in keys:
                for tag_k, tag_v in tags.items():
                    if tag_k.lower().endswith(k.lower()) or k.lower() in tag_k.lower():
                        return tag_v
            return None

        structured: Dict[str, Any] = {
            "make": get_field(["Make", "EXIF:Make", "Image Make"]),
            "model": get_field(["Model", "EXIF:Model", "Image Model"]),
            "software": get_field(["Software", "EXIF:Software", "Image Software"]),
            "orientation": get_field(["Orientation", "EXIF:Orientation", "Image Orientation"]),
            "datetime_original": get_field(["DateTimeOriginal", "EXIF:DateTimeOriginal", "EXIF DateTimeOriginal"]),
            "datetime_create": get_field(["CreateDate", "EXIF:CreateDate", "EXIF DateTimeDigitized"]),
            "datetime_modify": get_field(["ModifyDate", "EXIF:ModifyDate", "Image DateTime"]),
            "offset_time_original": get_field(["OffsetTimeOriginal", "EXIF:OffsetTimeOriginal"]),
            "offset_time": get_field(["OffsetTime", "EXIF:OffsetTime"]),
            "subsec_time_original": get_field(["SubSecTimeOriginal", "EXIF:SubSecTimeOriginal"]),
            "exposure_time": get_field(["ExposureTime", "EXIF:ExposureTime"]),
            "f_number": get_field(["FNumber", "EXIF:FNumber"]),
            "iso": get_field(["ISO", "ISOSpeedRatings", "EXIF:ISO"]),
            "focal_length": get_field(["FocalLength", "EXIF:FocalLength"]),
            "focal_length_35mm": get_field(["FocalLengthIn35mmFormat", "EXIF:FocalLengthIn35mmFormat"]),
            "flash": get_field(["Flash", "EXIF:Flash"]),
            "metering_mode": get_field(["MeteringMode", "EXIF:MeteringMode"]),
            "gps_latitude": get_field(["GPSLatitude", "GPS:GPSLatitude"]),
            "gps_longitude": get_field(["GPSLongitude", "GPS:GPSLongitude"]),
            "gps_altitude": get_field(["GPSAltitude", "GPS:GPSAltitude"]),
            "gps_datetime": get_field(["GPSDateTime", "GPSDateStamp", "GPS:GPSDateStamp"]),
            "exif_width": get_field(["ExifImageWidth", "EXIF:ExifImageWidth", "Image ImageWidth"]),
            "exif_height": get_field(["ExifImageHeight", "EXIF:ExifImageHeight", "Image ImageLength"]),
            "color_space": get_field(["ColorSpace", "EXIF:ColorSpace"]),
        }
        # Filter out None values
        return {k: v for k, v in structured.items() if v is not None}
