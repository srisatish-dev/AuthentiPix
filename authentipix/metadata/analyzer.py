"""Metadata & Provenance Analyzer Engine for AuthentiPix Phase 1 (Pillar 1).

Main entrypoint executing security validation, metadata extractors, normalization,
consistency rules, C2PA evidence generation, and reproducibility logging.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from authentipix.metadata.consistency import ConsistencyEngine
from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.extractors import (
    BaseExtractor,
    C2paExtractor,
    ExifExtractor,
    FileContainerExtractor,
    IptcExtractor,
    PreviewExtractor,
    XmpExtractor,
)
from authentipix.metadata.normalization import MetadataNormalizer
from authentipix.metadata.reproducibility import ReproducibilityManager
from authentipix.metadata.schemas import (
    AnalyzerOutput,
    EvidenceCategory,
    EvidenceItem,
    EvidenceStrength,
)
from authentipix.metadata.security import SecurityEnforcer, SecurityPolicyConfig, SecurityViolationError


class MetadataAnalyzer:
    """Main analyzer engine for Pillar 1 — Metadata & Provenance."""

    def __init__(
        self,
        security_config: Optional[SecurityPolicyConfig] = None,
        extractors: Optional[List[BaseExtractor]] = None,
    ):
        self.security_enforcer = SecurityEnforcer(security_config)
        self.normalizer = MetadataNormalizer()
        self.consistency_engine = ConsistencyEngine()
        self.reproducibility_manager = ReproducibilityManager()

        # Default extractor chain
        self.extractors: List[BaseExtractor] = extractors or [
            ExifExtractor(),
            XmpExtractor(),
            IptcExtractor(),
            FileContainerExtractor(),
            PreviewExtractor(),
            C2paExtractor(),
        ]

    @property
    def analyzer_id(self) -> str:
        return "pillar1_metadata_analyzer"

    @property
    def analyzer_version(self) -> str:
        return "1.0.0"

    def analyze(self, ctx: AnalysisContext) -> AnalyzerOutput:
        """Executes full Phase 1 analysis pipeline on the provided AnalysisContext."""
        start_time = time.perf_counter()
        output = AnalyzerOutput(
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
        )

        # 1. Security Validation
        try:
            sec_warnings = self.security_enforcer.validate_context(ctx)
            output.warnings.extend(sec_warnings)
        except SecurityViolationError as e:
            output.errors.append(f"Security policy violation: {e}")
            output.execution_time_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
            return output

        # 2. Extract Metadata across all extractors
        extraction_results = []
        raw_obs: Dict[str, Any] = {}

        for extractor in self.extractors:
            res = extractor.extract(ctx)
            extraction_results.append(res)
            raw_obs[res.extractor_name] = res.data

            if res.warnings:
                output.warnings.extend([f"[{res.extractor_name}] {w}" for w in res.warnings])
            if res.errors:
                output.errors.extend([f"[{res.extractor_name}] {e}" for e in res.errors])

        output.raw_observations = raw_obs

        # 3. Normalization
        norm = self.normalizer.normalize(extraction_results)
        output.normalized_features = norm.model_dump()

        # 4. Evidence Generation: Consistency Rules
        consistency_items = self.consistency_engine.evaluate_all(norm, extraction_results)
        output.evidence_items.extend(consistency_items)

        # 5. Evidence Generation: C2PA / Provenance
        c2pa_res = next((r for r in extraction_results if r.extractor_name == "c2pa_extractor"), None)
        if c2pa_res and c2pa_res.data:
            c2pa_items = self._generate_c2pa_evidence(c2pa_res.data)
            output.evidence_items.extend(c2pa_items)

        # 6. Missing Metadata Evidence Handling (Neutral / Inconclusive)
        has_exif = bool(norm.make or norm.model or norm.datetime_original or norm.exif_width)
        has_xmp = bool(norm.creator_tool or norm.document_id or norm.history_actions)
        has_iptc = bool(norm.byline or norm.copyright_notice)
        has_c2pa = bool(c2pa_res and c2pa_res.data.get("c2pa_present"))

        if not (has_exif or has_xmp or has_iptc or has_c2pa):
            output.evidence_items.append(
                EvidenceItem(
                    evidence_id=str(uuid.uuid4()),
                    category=EvidenceCategory.METADATA_STRUCTURE,
                    evidence_strength=EvidenceStrength.NEUTRAL,
                    validation_status_code="NO_METADATA",
                    rule_id="NO_METADATA_CHECK",
                    affected_fields=["container"],
                    raw_observation={"metadata_found": False},
                    finding_summary="No embedded EXIF, XMP, IPTC, or C2PA provenance headers found",
                    interpretation=(
                        "The asset contains no header metadata or digital provenance credentials. "
                        "Metadata analysis is inconclusive. Missing metadata is common across social media, "
                        "messaging platforms, and web exports and does not imply fabrication or manipulation."
                    ),
                    limitations=[
                        "Over 90% of messaging platforms and social networks strip metadata automatically for user privacy.",
                        "Absence of metadata provides no evidentiary support for either authenticity or manipulation."
                    ],
                )
            )

        # 7. Reproducibility Metadata
        output.reproducibility_metadata = self.reproducibility_manager.generate_reproducibility_payload(ctx)
        output.execution_time_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

        return output

    def _generate_c2pa_evidence(self, c2pa_data: Dict[str, Any]) -> List[EvidenceItem]:
        """Maps C2PA extraction data into conservative EvidenceItems."""
        items: List[EvidenceItem] = []
        status_code = c2pa_data.get("validation_status_code", "NO_CREDENTIAL")
        active_m = c2pa_data.get("active_manifest", {})
        val_eval = c2pa_data.get("validation_eval", {})

        if status_code == "NO_CREDENTIAL":
            return items

        # Check AI declaration assertion
        ai_declared = active_m.get("ai_generated_declared", False)
        if ai_declared:
            agent = "N/A"
            for act in active_m.get("actions", []):
                sa = act.get("softwareAgent")
                if sa:
                    agent = sa.get("name") if isinstance(sa, dict) else str(sa)
                    break
            if agent == "N/A" and active_m.get("claim_generator"):
                agent = active_m.get("claim_generator")

            items.append(
                EvidenceItem(
                    evidence_id=str(uuid.uuid4()),
                    category=EvidenceCategory.DIGITAL_PROVENANCE,
                    evidence_strength=EvidenceStrength.HIGH,
                    validation_status_code="C2PA_AI_DECLARED",
                    rule_id="C2PA_RULE_001",
                    affected_fields=["c2pa:manifest", "c2pa:actions"],
                    raw_observation={
                        "claim_generator": active_m.get("claim_generator"),
                        "actions": active_m.get("actions"),
                        "ai_generated_declared": True,
                        "software_agent": agent,
                    },
                    finding_summary=f"Valid C2PA manifest contains explicit declaration of AI generation by {agent}",
                    interpretation=(
                        f"The asset contains a cryptographically verified C2PA manifest with an action assertion "
                        f"explicitly declaring generative AI creation (Software Agent: {agent})."
                    ),
                    limitations=[
                        "C2PA verifies declared provenance assertions signed by a key; it does not replace optical or physical analysis nor independently prove real-world scene authorship."
                    ],
                )
            )

        # Primary Status Evidence Item
        if status_code in ("C2PA_VALID_TRUSTED", "C2PA_VALID_UNTRUSTED"):
            is_trusted = (status_code == "C2PA_VALID_TRUSTED")
            strength = EvidenceStrength.HIGH if is_trusted else EvidenceStrength.MEDIUM
            summary = (
                "C2PA manifest signature and content binding are valid and trusted"
                if is_trusted
                else "C2PA manifest signature and content binding are valid, but signing certificate is untrusted under active trust list configuration"
            )
            interp = (
                "The image content-binding hash and claim signature are valid (image bytes were not modified post-signature). "
                + ("The signing entity certificate is cryptographically trusted." if is_trusted else "However, the signing entity certificate is untrusted under the active trust configuration (e.g. self-signed or missing from local trust list).")
            )
            limitations = [
                "Valid C2PA provenance confirms signature chain and asset hash integrity, but does not prevent analog hole tampering (re-photographing a screen)."
            ]
            if not is_trusted:
                limitations.append("Untrusted certificates can be generated by self-signed applications or signers not present on the active trust list.")

            items.append(
                EvidenceItem(
                    evidence_id=str(uuid.uuid4()),
                    category=EvidenceCategory.DIGITAL_PROVENANCE,
                    evidence_strength=strength,
                    validation_status_code=status_code,
                    rule_id="C2PA_RULE_002",
                    affected_fields=["c2pa:manifest", "c2pa:signature", "c2pa:hash_data"],
                    raw_observation={
                        "validation_state_raw": c2pa_data.get("validation_state_raw"),
                        "validation_eval": val_eval,
                    },
                    finding_summary=summary,
                    interpretation=interp,
                    limitations=limitations,
                )
            )

        elif status_code == "CONTENT_BINDING_FAILURE":
            items.append(
                EvidenceItem(
                    evidence_id=str(uuid.uuid4()),
                    category=EvidenceCategory.DIGITAL_PROVENANCE,
                    evidence_strength=EvidenceStrength.HIGH,
                    validation_status_code="CONTENT_BINDING_FAILURE",
                    rule_id="C2PA_RULE_003",
                    affected_fields=["c2pa:hash_data"],
                    raw_observation={"raw_validation_results": c2pa_data.get("validation_results_raw")},
                    finding_summary="C2PA content-binding validation failed (asset data hash mismatch)",
                    interpretation=(
                        "Content-binding validation failed. The image byte content was altered after signature generation."
                    ),
                    limitations=[
                        "Does not automatically prove malicious fraud; minor format re-encoding or re-saving breaks asset hash binding."
                    ],
                )
            )

        elif status_code == "SIGNATURE_FAILURE":
            items.append(
                EvidenceItem(
                    evidence_id=str(uuid.uuid4()),
                    category=EvidenceCategory.DIGITAL_PROVENANCE,
                    evidence_strength=EvidenceStrength.HIGH,
                    validation_status_code="SIGNATURE_FAILURE",
                    rule_id="C2PA_RULE_004",
                    affected_fields=["c2pa:claim_signature"],
                    raw_observation={"raw_validation_results": c2pa_data.get("validation_results_raw")},
                    finding_summary="C2PA claim signature validation failed",
                    interpretation="Signature validation failed. Manifest claim data or signature payload is invalid or corrupted.",
                    limitations=["Can result from file corruption or incomplete manifest copying."],
                )
            )

        # Auxiliary Evidence Item for Untrusted Timestamp
        if val_eval.get("timestamp_untrusted") and status_code not in ("CONTENT_BINDING_FAILURE", "SIGNATURE_FAILURE"):
            items.append(
                EvidenceItem(
                    evidence_id=str(uuid.uuid4()),
                    category=EvidenceCategory.DIGITAL_PROVENANCE,
                    evidence_strength=EvidenceStrength.LOW,
                    validation_status_code="TIMESTAMP_UNTRUSTED",
                    rule_id="C2PA_RULE_005",
                    affected_fields=["c2pa:timestamp"],
                    raw_observation={"timestamp_untrusted": True},
                    finding_summary="C2PA timestamp authority certificate is untrusted",
                    interpretation="The cryptographic timestamp certificate embedded in the signature is untrusted or self-signed under active trust configuration.",
                    limitations=["Timestamp authority certificates must be registered on the trust list for independent temporal verification."],
                )
            )

        return items
