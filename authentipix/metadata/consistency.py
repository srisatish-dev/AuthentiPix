"""Metadata Consistency Analysis Engine for AuthentiPix Phase 1.

Implements the 7 approved deterministic consistency rules. Each rule produces
an EvidenceItem with forensic-safe interpretations, explicit strength ratings,
and limitations without prematurely claiming fraud or fabrication.
"""

import uuid
from typing import Any, Dict, List, Optional

from authentipix.metadata.extractors.base import ExtractionResult
from authentipix.metadata.schemas import (
    EvidenceCategory,
    EvidenceItem,
    EvidenceStrength,
    NormalizedMetadata,
)


class ConsistencyEngine:
    """Evaluates deterministic consistency rules against normalized metadata."""

    def evaluate_all(
        self,
        norm: NormalizedMetadata,
        extraction_results: List[ExtractionResult],
    ) -> List[EvidenceItem]:
        """Runs all 7 consistency rules and returns generated EvidenceItems."""
        items: List[EvidenceItem] = []

        # Find specific extraction results
        ext_map = {res.extractor_name: res for res in extraction_results}

        # Rule 1: EXIF vs Raster Dimensions
        r1 = self.rule_001_exif_vs_raster_dimensions(norm)
        if r1:
            items.append(r1)

        # Rule 2: Inter-Namespace Discrepancy
        r2 = self.rule_002_inter_namespace_discrepancy(norm)
        if r2:
            items.append(r2)

        # Rule 3: Temporal Chronology
        r3 = self.rule_003_temporal_chronology(norm)
        if r3:
            items.append(r3)

        # Rule 4: Software / Processing Agent Presence
        r4 = self.rule_004_software_agent_presence(norm)
        if r4:
            items.append(r4)

        # Rule 5: Color Profile Alignment
        r5 = self.rule_005_color_profile_alignment(norm)
        if r5:
            items.append(r5)

        # Rule 6: Container / Format Consistency
        r6 = self.rule_006_container_format_consistency(norm)
        if r6:
            items.append(r6)

        # Rule 7: Embedded Preview Alignment
        preview_res = ext_map.get("preview_extractor")
        if preview_res:
            r7 = self.rule_007_embedded_preview_alignment(norm, preview_res.data)
            if r7:
                items.append(r7)

        return items

    def rule_001_exif_vs_raster_dimensions(self, norm: NormalizedMetadata) -> Optional[EvidenceItem]:
        """RULE_001: Compares EXIF sensor dimensions against actual raster dimensions."""
        if not norm.exif_width or not norm.exif_height or not norm.raster_width or not norm.raster_height:
            return None  # Non-applicable state

        exif_dim = (norm.exif_width, norm.exif_height)
        raster_dim = (norm.raster_width, norm.raster_height)

        # Allow swapped orientation (portrait/landscape rotation)
        match_exact = (exif_dim == raster_dim) or (exif_dim == (raster_dim[1], raster_dim[0]))

        if not match_exact:
            return EvidenceItem(
                evidence_id=str(uuid.uuid4()),
                category=EvidenceCategory.CONSISTENCY_ANOMALY,
                evidence_strength=EvidenceStrength.HIGH,
                validation_status_code="DIMENSION_MISMATCH",
                rule_id="RULE_001",
                affected_fields=["exif_width", "exif_height", "raster_width", "raster_height"],
                raw_observation={"exif_dimensions": exif_dim, "raster_dimensions": raster_dim},
                finding_summary=f"EXIF dimensions ({exif_dim[0]}x{exif_dim[1]}) conflict with raster dimensions ({raster_dim[0]}x{raster_dim[1]})",
                interpretation=(
                    "Metadata dimensions differ from raster dimensions, which may be consistent with "
                    "resizing, cropping, export, or other post-capture processing."
                ),
                limitations=[
                    "Does not prove malicious editing.",
                    "Common in legitimate non-destructive cropping or export pipelines."
                ],
                supporting_values={"exif_aspect_ratio": round(exif_dim[0] / exif_dim[1], 4), "raster_aspect_ratio": round(raster_dim[0] / raster_dim[1], 4)},
            )
        return None

    def rule_002_inter_namespace_discrepancy(self, norm: NormalizedMetadata) -> Optional[EvidenceItem]:
        """RULE_002: Compares EXIF vs XMP software and date values."""
        exif_soft = (norm.software or "").strip().lower()
        xmp_tool = (norm.creator_tool or "").strip().lower()

        if exif_soft and xmp_tool and exif_soft != xmp_tool:
            return EvidenceItem(
                evidence_id=str(uuid.uuid4()),
                category=EvidenceCategory.CONSISTENCY_ANOMALY,
                evidence_strength=EvidenceStrength.MEDIUM,
                validation_status_code="NAMESPACE_DISCREPANCY",
                rule_id="RULE_002",
                affected_fields=["software", "creator_tool"],
                raw_observation={"EXIF:Software": norm.software, "XMP:CreatorTool": norm.creator_tool},
                finding_summary=f"EXIF Software ('{norm.software}') differs from XMP CreatorTool ('{norm.creator_tool}')",
                interpretation=(
                    "Metadata namespaces contain inconsistent values that may reflect multi-stage workflows, "
                    "metadata rewriting, or stale metadata."
                ),
                limitations=[
                    "Multi-stage image processing tools often add XMP blocks without updating legacy EXIF tags."
                ],
            )
        return None

    def rule_003_temporal_chronology(self, norm: NormalizedMetadata) -> Optional[EvidenceItem]:
        """RULE_003: Checks sequence order across capture and modification timestamps."""
        orig = norm.datetime_original
        mod = norm.datetime_modify

        # Basic string date comparison when formatted in standard YYYY:MM:DD HH:MM:SS
        if orig and mod and len(orig) >= 10 and len(mod) >= 10:
            if mod < orig:
                return EvidenceItem(
                    evidence_id=str(uuid.uuid4()),
                    category=EvidenceCategory.TEMPORAL_CHAIN,
                    evidence_strength=EvidenceStrength.MEDIUM,
                    validation_status_code="TEMPORAL_INVERSION",
                    rule_id="RULE_003",
                    affected_fields=["datetime_original", "datetime_modify"],
                    raw_observation={"datetime_original": orig, "datetime_modify": mod},
                    finding_summary=f"Modification timestamp ('{mod}') precedes original capture timestamp ('{orig}')",
                    interpretation=(
                        "Temporal inconsistency detected where modification timestamp precedes capture timestamp. "
                        "May be caused by system clock misconfiguration, timezone handling, export workflows, or metadata editing."
                    ),
                    limitations=[
                        "Camera clocks may not have been synchronized.",
                        "Timezone offsets without explicit EXIF:OffsetTime tags can cause apparent timestamp inversions."
                    ],
                )
        return None

    def rule_004_software_agent_presence(self, norm: NormalizedMetadata) -> Optional[EvidenceItem]:
        """RULE_004: Detects presence of image-editing software metadata."""
        software_found = norm.software or norm.creator_tool
        history = norm.history_actions

        if software_found or history:
            tools_list = []
            if software_found:
                tools_list.append(software_found)
            for act in history:
                agent = act.get("softwareAgent")
                if agent and agent not in tools_list:
                    tools_list.append(agent)

            return EvidenceItem(
                evidence_id=str(uuid.uuid4()),
                category=EvidenceCategory.SOFTWARE_WORKFLOW,
                evidence_strength=EvidenceStrength.LOW,
                validation_status_code="SOFTWARE_METADATA_PRESENT",
                rule_id="RULE_004",
                affected_fields=["software", "creator_tool", "history_actions"],
                raw_observation={"software_tools_detected": tools_list},
                finding_summary=f"Image software processing metadata detected: {', '.join(tools_list[:3])}",
                interpretation="Metadata indicates processing by image software.",
                limitations=[
                    "Image software processing is standard across professional digital photography.",
                    "Does not prove malicious or fraudulent manipulation."
                ],
                supporting_values={"history_event_count": len(history)},
            )
        return None

    def rule_005_color_profile_alignment(self, norm: NormalizedMetadata) -> Optional[EvidenceItem]:
        """RULE_005: Compares EXIF color space declaration against ICC profile."""
        cs = norm.color_space
        icc_name = norm.icc_profile_name

        if cs and icc_name:
            # Check for uncalibrated EXIF (65535 or 0xFFFF) with active sRGB profile, or explicit conflict
            if ("uncalibrated" in cs.lower() or "65535" in cs) and "srgb" in icc_name.lower():
                return EvidenceItem(
                    evidence_id=str(uuid.uuid4()),
                    category=EvidenceCategory.CONSISTENCY_ANOMALY,
                    evidence_strength=EvidenceStrength.LOW,
                    validation_status_code="COLOR_PROFILE_MISMATCH",
                    rule_id="RULE_005",
                    affected_fields=["color_space", "icc_profile_name"],
                    raw_observation={"EXIF:ColorSpace": cs, "ICC_Profile": icc_name},
                    finding_summary="EXIF ColorSpace is uncalibrated while embedded ICC Profile is sRGB",
                    interpretation=(
                        "Color space metadata declaration differs from embedded ICC profile information. "
                        "May indicate color conversion, export processing, or re-encoding."
                    ),
                    limitations=[
                        "Color space conversion is a routine non-destructive image processing operation."
                    ],
                )
        return None

    def rule_006_container_format_consistency(self, norm: NormalizedMetadata) -> Optional[EvidenceItem]:
        """RULE_006: Compares MIME type against raster container properties."""
        mime = (norm.mime_type or "").lower()
        exif_raw = str(norm.raw_tags).lower()

        # Simple container check: WebP container containing JPEG tags
        if "webp" in mime and "jpeg" in exif_raw and "exifread" in exif_raw:
            return EvidenceItem(
                evidence_id=str(uuid.uuid4()),
                category=EvidenceCategory.METADATA_STRUCTURE,
                evidence_strength=EvidenceStrength.MEDIUM,
                validation_status_code="CONTAINER_MISMATCH",
                rule_id="RULE_006",
                affected_fields=["mime_type", "raw_tags"],
                raw_observation={"mime_type": mime},
                finding_summary="WebP file container contains residual JPEG metadata tags",
                interpretation=(
                    "File container structure or MIME type conflicts with embedded metadata header signatures. "
                    "May indicate format conversion, re-encoding, or metadata copying."
                ),
                limitations=["Format conversion software routinely preserves original metadata headers."],
            )
        return None

    def rule_007_embedded_preview_alignment(
        self,
        norm: NormalizedMetadata,
        preview_data: Dict[str, Any],
    ) -> Optional[EvidenceItem]:
        """RULE_007: Compares IFD1 thumbnail aspect ratio against main image raster."""
        if not preview_data.get("thumbnail_present"):
            return None

        tw = preview_data.get("thumbnail_width")
        th = preview_data.get("thumbnail_height")
        rw = norm.raster_width
        rh = norm.raster_height

        if tw and th and rw and rh and th > 0 and rh > 0:
            thumb_aspect = tw / th
            main_aspect = rw / rh

            # Handle landscape vs portrait orientation differences
            diff = abs(thumb_aspect - main_aspect)
            diff_rot = abs(thumb_aspect - (rh / rw))

            if min(diff, diff_rot) > 0.15:  # Noticeable aspect ratio divergence threshold
                return EvidenceItem(
                    evidence_id=str(uuid.uuid4()),
                    category=EvidenceCategory.CONSISTENCY_ANOMALY,
                    evidence_strength=EvidenceStrength.HIGH,
                    validation_status_code="THUMBNAIL_ASPECT_MISMATCH",
                    rule_id="RULE_007",
                    affected_fields=["thumbnail_width", "thumbnail_height", "raster_width", "raster_height"],
                    raw_observation={"thumbnail_dimensions": (tw, th), "raster_dimensions": (rw, rh)},
                    finding_summary=f"Embedded thumbnail aspect ratio ({round(thumb_aspect, 2)}) differs from main raster aspect ratio ({round(main_aspect, 2)})",
                    interpretation=(
                        "Embedded preview differs from the current main image raster in aspect ratio, "
                        "which may reflect later canvas cropping or stale embedded preview content."
                    ),
                    limitations=[
                        "Editing applications frequently modify main canvas pixels without regenerating EXIF IFD1 thumbnails."
                    ],
                    supporting_values={"thumbnail_aspect": round(thumb_aspect, 4), "main_aspect": round(main_aspect, 4)},
                )
        return None
