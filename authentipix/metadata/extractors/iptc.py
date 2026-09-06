"""IPTC Extractor for AuthentiPix.

Extracts IPTC IIM dataset fields and IPTC Core rights/workflow tags.
"""

import io
from typing import Any, Dict, List, Optional
from PIL import Image, IptcImagePlugin

from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.extractors.base import BaseExtractor, ExtractionResult


class IptcExtractor(BaseExtractor):
    """Extractor for IPTC Information Interchange Model metadata."""

    @property
    def extractor_name(self) -> str:
        return "iptc_extractor"

    def _extract_internal(self, ctx: AnalysisContext) -> ExtractionResult:
        result = ExtractionResult(extractor_name=self.extractor_name)
        image_bytes = ctx.get_bytes()
        raw_iptc: Dict[str, Any] = {}

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                iptc_data = IptcImagePlugin.getiptcinfo(img)
                if iptc_data:
                    for key, val in iptc_data.items():
                        key_str = str(key)
                        if isinstance(val, bytes):
                            val_str = val.decode("utf-8", errors="replace")
                        elif isinstance(val, list):
                            val_str = [v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v) for v in val]
                        else:
                            val_str = str(val)
                        raw_iptc[key_str] = val_str
        except Exception as e:
            result.warnings.append(f"IPTC extraction notice: {e}")

        result.raw_tags = raw_iptc
        result.data = self._structure_iptc(raw_iptc)
        return result

    def _structure_iptc(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Maps IPTC key tuples to structured dictionary."""
        structured: Dict[str, Any] = {}
        # Key mapping for common IPTC IIM records (Record 2)
        mapping = {
            (2, 80): "byline",             # 2:80 = By-line
            (2, 116): "copyright_notice",  # 2:116 = Copyright Notice
            (2, 110): "credit",            # 2:110 = Credit
            (2, 115): "source",            # 2:115 = Source
            (2, 55): "date_created",       # 2:55 = Date Created
            (2, 60): "time_created",       # 2:60 = Time Created
            (2, 62): "digital_creation_date", # 2:62 = Digital Creation Date
            (2, 63): "digital_creation_time", # 2:63 = Digital Creation Time
        }

        for (record, dataset), field_name in mapping.items():
            if (record, dataset) in raw:
                structured[field_name] = raw[(record, dataset)]

        return structured
