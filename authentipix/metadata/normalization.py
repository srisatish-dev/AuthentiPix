"""Metadata Normalizer for AuthentiPix Phase 1.

Translates raw extraction outputs into a unified canonical NormalizedMetadata model.
Preserves original raw values, nulls, and timezone offsets without fabricating data.
"""

from typing import Any, Dict, List, Optional
from authentipix.metadata.extractors.base import ExtractionResult
from authentipix.metadata.schemas import NormalizedMetadata


class MetadataNormalizer:
    """Normalizes raw metadata payloads across disparate extractors."""

    def normalize(self, extraction_results: List[ExtractionResult]) -> NormalizedMetadata:
        """Compiles a unified NormalizedMetadata model from extractor results."""
        normalized = NormalizedMetadata()
        raw_bag: Dict[str, Any] = {}

        for res in extraction_results:
            raw_bag[res.extractor_name] = res.data
            if res.raw_tags:
                raw_bag[f"{res.extractor_name}_raw_tags"] = res.raw_tags

            # Extract EXIF fields
            if res.extractor_name == "exif_extractor" and res.data:
                d = res.data
                normalized.make = self._clean_str(d.get("make"))
                normalized.model = self._clean_str(d.get("model"))
                normalized.software = self._clean_str(d.get("software"))
                normalized.orientation = self._clean_int(d.get("orientation"))

                normalized.datetime_original = self._clean_str(d.get("datetime_original"))
                normalized.datetime_create = self._clean_str(d.get("datetime_create"))
                normalized.datetime_modify = self._clean_str(d.get("datetime_modify"))
                normalized.offset_time_original = self._clean_str(d.get("offset_time_original"))
                normalized.offset_time = self._clean_str(d.get("offset_time"))
                normalized.subsec_time_original = self._clean_str(d.get("subsec_time_original"))

                normalized.exposure_time = self._clean_str(d.get("exposure_time"))
                normalized.f_number = self._clean_float(d.get("f_number"))
                normalized.iso = self._clean_int(d.get("iso"))
                normalized.focal_length = self._clean_float(d.get("focal_length"))
                normalized.focal_length_35mm = self._clean_float(d.get("focal_length_35mm"))
                normalized.flash = self._clean_str(d.get("flash"))
                normalized.metering_mode = self._clean_str(d.get("metering_mode"))

                normalized.gps_latitude = self._clean_float(d.get("gps_latitude"))
                normalized.gps_longitude = self._clean_float(d.get("gps_longitude"))
                normalized.gps_altitude = self._clean_float(d.get("gps_altitude"))
                normalized.gps_datetime = self._clean_str(d.get("gps_datetime"))

                normalized.exif_width = self._clean_int(d.get("exif_width"))
                normalized.exif_height = self._clean_int(d.get("exif_height"))
                normalized.color_space = self._clean_str(d.get("color_space"))

            # Extract XMP fields
            elif res.extractor_name == "xmp_extractor" and res.data:
                d = res.data
                if not normalized.software and d.get("xmp_creatortool"):
                    normalized.software = self._clean_str(d.get("xmp_creatortool"))
                if not normalized.datetime_create and d.get("xmp_createdate"):
                    normalized.datetime_create = self._clean_str(d.get("xmp_createdate"))
                if not normalized.datetime_modify and d.get("xmp_modifydate"):
                    normalized.datetime_modify = self._clean_str(d.get("xmp_modifydate"))

                normalized.creator_tool = self._clean_str(d.get("xmp_creatortool"))
                normalized.document_id = self._clean_str(d.get("xmpmm_documentid"))
                normalized.instance_id = self._clean_str(d.get("xmpmm_instanceid"))
                if isinstance(d.get("history_events"), list):
                    normalized.history_actions = d["history_events"]

            # Extract IPTC fields
            elif res.extractor_name == "iptc_extractor" and res.data:
                d = res.data
                normalized.byline = self._clean_str(d.get("byline"))
                normalized.copyright_notice = self._clean_str(d.get("copyright_notice"))
                normalized.credit = self._clean_str(d.get("credit"))
                normalized.digital_creation_date = self._clean_str(d.get("digital_creation_date"))

            # Extract Container/Raster fields
            elif res.extractor_name == "file_container_extractor" and res.data:
                d = res.data
                normalized.mime_type = self._clean_str(d.get("mime_type"))
                normalized.file_size_bytes = self._clean_int(d.get("file_size_bytes"))
                normalized.raster_width = self._clean_int(d.get("raster_width"))
                normalized.raster_height = self._clean_int(d.get("raster_height"))
                normalized.icc_profile_name = self._clean_str(d.get("icc_profile_name"))

        normalized.raw_tags = raw_bag
        return normalized

    @staticmethod
    def _clean_str(val: Any) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip()
        return s if s else None

    @staticmethod
    def _clean_int(val: Any) -> Optional[int]:
        if val is None:
            return None
        try:
            return int(str(val).split(".")[0])
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _clean_float(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(str(val))
        except (ValueError, TypeError):
            return None
