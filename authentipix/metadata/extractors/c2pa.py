"""C2PA / Content Credentials Extractor for AuthentiPix.

Delegates JUMBF manifest parsing and cryptographic validation to the official c2pa-python SDK,
mapping validation outcomes into conservative, SDK-aligned validation status codes.
Preserves distinct 4-dimensional validation codes (content binding, claim signature,
signing credential trust, timestamp trust).
"""

import io
import json
from typing import Any, Dict, List, Optional

from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.extractors.base import BaseExtractor, ExtractionResult


class C2paExtractor(BaseExtractor):
    """Extractor for C2PA signed provenance manifests."""

    @property
    def extractor_name(self) -> str:
        return "c2pa_extractor"

    def _extract_internal(self, ctx: AnalysisContext) -> ExtractionResult:
        result = ExtractionResult(extractor_name=self.extractor_name)
        data: Dict[str, Any] = {
            "c2pa_present": False,
            "validation_status_code": "NO_CREDENTIAL",
        }

        try:
            import c2pa
        except ImportError:
            result.warnings.append("c2pa-python SDK is not installed in current python environment")
            result.data = data
            return result

        image_bytes = ctx.get_bytes()
        try:
            reader = c2pa.Reader.try_create(ctx.mime_type or "image/jpeg", stream=io.BytesIO(image_bytes))

            if reader is None:
                data["c2pa_present"] = False
                data["validation_status_code"] = "NO_CREDENTIAL"
                result.data = data
                return result

            # C2PA manifest found
            data["c2pa_present"] = True
            data["is_embedded"] = reader.is_embedded()

            # Retrieve validation state & results
            val_state = reader.get_validation_state()
            val_results = reader.get_validation_results()
            data["validation_state_raw"] = val_state
            data["validation_results_raw"] = val_results

            # Evaluate structured validation codes across the 4 dimensions
            val_eval = self._evaluate_validation_codes(val_results)
            data["validation_eval"] = val_eval

            # Retrieve active manifest
            active_manifest = reader.get_active_manifest()
            if active_manifest:
                data["active_manifest"] = self._sanitize_manifest_dict(active_manifest)

            # Map SDK validation status to AuthentiPix taxonomy
            data["validation_status_code"] = self._map_validation_status(val_state, val_eval)

        except Exception as e:
            result.warnings.append(f"C2PA reader error: {e}")
            data["validation_status_code"] = "C2PA_READ_ERROR"

        result.data = data
        return result

    def _evaluate_validation_codes(self, val_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Parses structured validation codes from C2PA SDK get_validation_results()."""
        status_eval = {
            "data_hash_valid": False,
            "data_hash_failed": False,
            "signature_valid": False,
            "signature_failed": False,
            "signing_credential_trusted": False,
            "signing_credential_untrusted": False,
            "timestamp_validated": False,
            "timestamp_untrusted": False,
            "raw_codes": {"success": [], "informational": [], "failure": []},
        }

        if not val_results:
            return status_eval

        # Extract activeManifest or main dict
        active_manifest = val_results.get("activeManifest") or val_results
        if not isinstance(active_manifest, dict):
            return status_eval

        success_list = active_manifest.get("success", [])
        info_list = active_manifest.get("informational", [])
        fail_list = active_manifest.get("failure", [])

        succ_codes = [s.get("code", "") for s in success_list if isinstance(s, dict)]
        info_codes = [i.get("code", "") for i in info_list if isinstance(i, dict)]
        fail_codes = [f.get("code", "") for f in fail_list if isinstance(f, dict)]

        status_eval["raw_codes"] = {
            "success": succ_codes,
            "informational": info_codes,
            "failure": fail_codes,
        }

        # 1. Content / Data Hash Binding Check
        if any("datahash.match" in c.lower() for c in succ_codes):
            status_eval["data_hash_valid"] = True
        if any("datahash.mismatch" in c.lower() or "datahash.fail" in c.lower() for c in fail_codes):
            status_eval["data_hash_failed"] = True

        # 2. Claim Signature Check
        if any("claimsignature.validated" in c.lower() or "claimsignature.insidevalidity" in c.lower() for c in succ_codes):
            status_eval["signature_valid"] = True
        if any("claimsignature.invalid" in c.lower() or "claimsignature.mismatch" in c.lower() for c in fail_codes):
            status_eval["signature_failed"] = True

        # 3. Signing Credential Trust Check
        if any("signingcredential.validated" in c.lower() or "signingcredential.trusted" in c.lower() for c in succ_codes):
            status_eval["signing_credential_trusted"] = True
        if any("signingcredential.untrusted" in c.lower() for c in fail_codes):
            status_eval["signing_credential_untrusted"] = True

        # 4. Timestamp Validation / Trust Check
        if any("timestamp.validated" in c.lower() for c in succ_codes):
            status_eval["timestamp_validated"] = True
        if any("timestamp.untrusted" in c.lower() for c in info_codes + fail_codes):
            status_eval["timestamp_untrusted"] = True

        return status_eval

    def _map_validation_status(self, val_state: Optional[str], val_eval: Dict[str, Any]) -> str:
        """Maps SDK validation results to AuthentiPix taxonomy based on exact code criteria."""
        if not val_state and not val_eval.get("raw_codes", {}).get("success"):
            return "NO_CREDENTIAL"

        # Priority 1: Data Hash / Content Binding Failure (Actual byte alteration post-signature)
        if val_eval.get("data_hash_failed"):
            return "CONTENT_BINDING_FAILURE"

        # Priority 2: Claim Signature Failure
        if val_eval.get("signature_failed"):
            return "SIGNATURE_FAILURE"

        # Priority 3: Data Hash Valid + Signature Valid
        state_str = str(val_state or "").lower()
        is_valid_state = state_str in ("valid", "pass") or val_eval.get("data_hash_valid") or val_eval.get("signature_valid")

        if is_valid_state:
            if val_eval.get("signing_credential_untrusted"):
                return "C2PA_VALID_UNTRUSTED"
            return "C2PA_VALID_TRUSTED"

        return "C2PA_UNVERIFIED"

    def _sanitize_manifest_dict(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Filters manifest data for key assertions (actions, ingredients, generator tags)."""
        sanitized: Dict[str, Any] = {
            "title": manifest.get("title"),
            "format": manifest.get("format"),
            "instance_id": manifest.get("instance_id"),
            "claim_generator": manifest.get("claim_generator"),
            "actions": [],
            "ingredients": [],
            "ai_generated_declared": False,
        }

        # Inspect assertions for c2pa.actions & AI indicators
        assertions = manifest.get("assertions", [])
        for assertion in assertions:
            label = assertion.get("label", "")
            data = assertion.get("data", {})

            if "c2pa.actions" in label or "actions" in label:
                actions_list = data.get("actions", [])
                for action in actions_list:
                    sanitized["actions"].append(action)
                    # Check for digitalSourceType indicating AI generation
                    digital_source = str(action.get("digitalSourceType", "")).lower()
                    if "trainedalgorithmicmedia" in digital_source or "compositingwithtrainedalgorithmicmedia" in digital_source:
                        sanitized["ai_generated_declared"] = True

        # Capture ingredients
        ingredients = manifest.get("ingredients", [])
        for ing in ingredients:
            sanitized["ingredients"].append({
                "title": ing.get("title"),
                "format": ing.get("format"),
                "instance_id": ing.get("instance_id"),
            })

        return sanitized
