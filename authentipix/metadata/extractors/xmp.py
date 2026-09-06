"""XMP Extractor for AuthentiPix.

Extracts XMP XML payloads and parses RDF namespaces (xmp, xmpMM, stEvt, crs)
safely using defusedxml to block XXE payloads.
"""

import io
from typing import Any, Dict, List, Optional
from PIL import Image

from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.extractors.base import BaseExtractor, ExtractionResult
from authentipix.metadata.security import SecurityEnforcer


class XmpExtractor(BaseExtractor):
    """Extractor for XMP metadata and edit history sequences."""

    @property
    def extractor_name(self) -> str:
        return "xmp_extractor"

    def _extract_internal(self, ctx: AnalysisContext) -> ExtractionResult:
        result = ExtractionResult(extractor_name=self.extractor_name)
        image_bytes = ctx.get_bytes()
        xmp_xml: Optional[str] = None

        # 1. Try PIL info dict to locate XMP payload
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                info = img.info or {}
                if "xmp" in info and isinstance(info["xmp"], (str, bytes)):
                    raw_val = info["xmp"]
                    xmp_xml = raw_val.decode("utf-8", errors="replace") if isinstance(raw_val, bytes) else raw_val
                elif "XML:com.adobe.xmp" in info:
                    raw_val = info["XML:com.adobe.xmp"]
                    xmp_xml = raw_val.decode("utf-8", errors="replace") if isinstance(raw_val, bytes) else raw_val
        except Exception as e:
            result.warnings.append(f"PIL XMP location notice: {e}")

        # 2. Binary marker scanning fallback for XMP packet in JPEG/PNG
        if not xmp_xml:
            xmp_xml = self._scan_xmp_packet(image_bytes)

        if not xmp_xml:
            result.data["xmp_present"] = False
            return result

        result.data["xmp_present"] = True
        result.raw_tags["xmp_raw_xml"] = xmp_xml[:2048] + ("..." if len(xmp_xml) > 2048 else "")

        # 3. Parse XML safely using defusedxml
        try:
            root = SecurityEnforcer.safe_parse_xml(xmp_xml)
            if root is not None:
                parsed_xmp = self._parse_xmp_tree(root)
                result.data.update(parsed_xmp)
        except Exception as e:
            result.errors.append(f"XMP XML parsing failed: {e}")

        return result

    def _scan_xmp_packet(self, data: bytes) -> Optional[str]:
        """Binary scan for XMP packet boundaries `<?xpacket begin` or `http://ns.adobe.com/xap/1.0/`."""
        start_tag = b"<x:xmpmeta"
        end_tag = b"</x:xmpmeta>"
        start_idx = data.find(start_tag)
        end_idx = data.find(end_tag)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return data[start_idx : end_idx + len(end_tag)].decode("utf-8", errors="replace")

        # Check alternative packet tag
        alt_start = b"<?xpacket begin"
        alt_end = b"<?xpacket end"
        s_idx = data.find(alt_start)
        e_idx = data.find(alt_end)

        if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
            return data[s_idx : e_idx + len(alt_end) + 6].decode("utf-8", errors="replace")

        return None

    def _parse_xmp_tree(self, root: Any) -> Dict[str, Any]:
        """Extracts structured values and history events from defusedxml Element tree."""
        structured: Dict[str, Any] = {
            "history_events": [],
        }

        # Common XMP XML Namespaces
        ns = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "xmp": "http://ns.adobe.com/xap/1.0/",
            "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
            "stEvt": "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#",
            "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
            "photoshop": "http://ns.adobe.com/photoshop/1.0/",
            "dc": "http://purl.org/dc/elements/1.1/",
        }

        # Recursive text extract helper
        for elem in root.iter():
            tag = elem.tag
            tag_name = tag.split("}")[-1] if "}" in tag else tag
            val = (elem.text or "").strip()

            if val and len(val) < 256:
                if tag_name in ("CreateDate", "ModifyDate", "MetadataDate", "CreatorTool"):
                    structured[f"xmp_{tag_name.lower()}"] = val
                elif tag_name in ("DocumentID", "InstanceID", "OriginalDocumentID"):
                    structured[f"xmpmm_{tag_name.lower()}"] = val

            # Check attributes on rdf:Description
            for attr_k, attr_v in elem.attrib.items():
                attr_name = attr_k.split("}")[-1] if "}" in attr_k else attr_k
                if attr_name in ("CreateDate", "ModifyDate", "CreatorTool"):
                    structured[f"xmp_{attr_name.lower()}"] = attr_v
                elif attr_name in ("DocumentID", "InstanceID", "OriginalDocumentID"):
                    structured[f"xmpmm_{attr_name.lower()}"] = attr_v

            # Extract stEvt:History items
            if tag_name == "History":
                for seq in elem.iter():
                    if seq.tag.endswith("ResourceEvent") or seq.tag.endswith("li"):
                        event: Dict[str, Any] = {}
                        for child in seq.iter():
                            c_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                            c_val = (child.text or "").strip()
                            if c_val:
                                event[c_tag] = c_val
                            for ak, av in child.attrib.items():
                                event[ak.split("}")[-1]] = av
                        if event:
                            structured["history_events"].append(event)

        return structured
