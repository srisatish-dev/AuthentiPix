"""Regression test suite for C2PA validation auditing and evidence mapping.

Covers exact 4-dimensional SDK validation mappings:
a. Valid data binding + untrusted signing credential (C2PA_VALID_UNTRUSTED - sample1.png scenario)
b. Actual content/data binding failure (assertion.dataHash.mismatch -> CONTENT_BINDING_FAILURE)
c. Valid signature + untrusted timestamp (timeStamp.untrusted -> TIMESTAMP_UNTRUSTED)
d. AI-generation declaration (c2pa.created with trainedAlgorithmicMedia)
"""

from authentipix.metadata.analyzer import MetadataAnalyzer
from authentipix.metadata.extractors.c2pa import C2paExtractor


def test_regression_a_valid_data_binding_untrusted_signing_credential():
    """Scenario A: Data hash valid + claim signature valid + signingCredential.untrusted.

    Must map to C2PA_VALID_UNTRUSTED, NOT CONTENT_BINDING_FAILURE!
    Must NOT state that image bytes were modified after signature generation.
    """
    extractor = C2paExtractor()
    raw_results = {
        "activeManifest": {
            "success": [
                {"code": "timeStamp.validated", "explanation": "timestamp message digest matched: OpenAI TSA Leaf"},
                {"code": "claimSignature.insideValidity", "explanation": "claim signature valid"},
                {"code": "claimSignature.validated", "explanation": "claim signature valid"},
                {"code": "assertion.hashedURI.match", "explanation": "hashed uri matched"},
                {"code": "assertion.dataHash.match", "explanation": "data hash valid"},
            ],
            "informational": [
                {"code": "timeStamp.untrusted", "explanation": "timestamp cert untrusted: OpenAI TSA Leaf"}
            ],
            "failure": [
                {"code": "signingCredential.untrusted", "explanation": "signing certificate untrusted"}
            ]
        }
    }

    eval_res = extractor._evaluate_validation_codes(raw_results)
    assert eval_res["data_hash_valid"] is True
    assert eval_res["data_hash_failed"] is False
    assert eval_res["signature_valid"] is True
    assert eval_res["signing_credential_untrusted"] is True
    assert eval_res["timestamp_untrusted"] is True

    status_code = extractor._map_validation_status("Valid", eval_res)
    assert status_code == "C2PA_VALID_UNTRUSTED"

    # Evaluate generated evidence items via MetadataAnalyzer
    analyzer = MetadataAnalyzer()
    c2pa_data = {
        "validation_status_code": status_code,
        "validation_state_raw": "Valid",
        "validation_eval": eval_res,
        "active_manifest": {
            "claim_generator": "gpt-image 2.0",
            "ai_generated_declared": True,
            "actions": [
                {
                    "action": "c2pa.created",
                    "softwareAgent": {"name": "gpt-image", "version": "2.0"},
                    "digitalSourceType": "trainedAlgorithmicMedia",
                }
            ],
        },
    }

    items = analyzer._generate_c2pa_evidence(c2pa_data)
    status_codes = [i.validation_status_code for i in items]

    # Verify AI declaration and C2PA_VALID_UNTRUSTED items are present
    assert "C2PA_AI_DECLARED" in status_codes
    assert "C2PA_VALID_UNTRUSTED" in status_codes
    assert "CONTENT_BINDING_FAILURE" not in status_codes

    untrusted_item = next(i for i in items if i.validation_status_code == "C2PA_VALID_UNTRUSTED")
    # Verify accurate forensic interpretation: image bytes were NOT modified!
    assert "image bytes were not modified post-signature" in untrusted_item.interpretation.lower()
    assert "untrusted under active trust list configuration" in untrusted_item.finding_summary.lower()


def test_regression_b_actual_content_data_binding_failure():
    """Scenario B: Actual data hash mismatch (assertion.dataHash.mismatch).

    Must map to CONTENT_BINDING_FAILURE.
    """
    extractor = C2paExtractor()
    raw_results = {
        "activeManifest": {
            "success": [
                {"code": "claimSignature.validated", "explanation": "claim signature valid"},
            ],
            "failure": [
                {"code": "assertion.dataHash.mismatch", "explanation": "data hash mismatch"},
            ]
        }
    }

    eval_res = extractor._evaluate_validation_codes(raw_results)
    assert eval_res["data_hash_failed"] is True
    assert eval_res["data_hash_valid"] is False

    status_code = extractor._map_validation_status("Invalid", eval_res)
    assert status_code == "CONTENT_BINDING_FAILURE"

    analyzer = MetadataAnalyzer()
    c2pa_data = {
        "validation_status_code": status_code,
        "validation_eval": eval_res,
        "validation_results_raw": raw_results,
    }

    items = analyzer._generate_c2pa_evidence(c2pa_data)
    assert len(items) == 1
    assert items[0].validation_status_code == "CONTENT_BINDING_FAILURE"
    assert "content-binding validation failed" in items[0].finding_summary.lower()


def test_regression_c_valid_signature_untrusted_timestamp():
    """Scenario C: Valid signature + untrusted timestamp (timeStamp.untrusted).

    Must emit TIMESTAMP_UNTRUSTED auxiliary evidence item.
    """
    extractor = C2paExtractor()
    raw_results = {
        "activeManifest": {
            "success": [
                {"code": "claimSignature.validated", "explanation": "claim signature valid"},
                {"code": "assertion.dataHash.match", "explanation": "data hash valid"},
                {"code": "signingCredential.validated", "explanation": "cert trusted"},
            ],
            "informational": [
                {"code": "timeStamp.untrusted", "explanation": "timestamp cert untrusted"},
            ]
        }
    }

    eval_res = extractor._evaluate_validation_codes(raw_results)
    assert eval_res["timestamp_untrusted"] is True

    status_code = extractor._map_validation_status("Valid", eval_res)
    assert status_code == "C2PA_VALID_TRUSTED"

    analyzer = MetadataAnalyzer()
    c2pa_data = {
        "validation_status_code": status_code,
        "validation_eval": eval_res,
        "active_manifest": {},
    }

    items = analyzer._generate_c2pa_evidence(c2pa_data)
    codes = [i.validation_status_code for i in items]
    assert "C2PA_VALID_TRUSTED" in codes
    assert "TIMESTAMP_UNTRUSTED" in codes

    ts_item = next(i for i in items if i.validation_status_code == "TIMESTAMP_UNTRUSTED")
    assert "timestamp authority certificate is untrusted" in ts_item.finding_summary.lower()


def test_regression_d_ai_generation_declaration():
    """Scenario D: AI-generation declaration (trainedAlgorithmicMedia)."""
    analyzer = MetadataAnalyzer()
    c2pa_data = {
        "validation_status_code": "C2PA_VALID_UNTRUSTED",
        "active_manifest": {
            "claim_generator": "gpt-image 2.0",
            "ai_generated_declared": True,
            "actions": [
                {
                    "action": "c2pa.created",
                    "softwareAgent": {"name": "gpt-image", "version": "2.0"},
                    "digitalSourceType": "trainedAlgorithmicMedia",
                }
            ],
        },
    }

    items = analyzer._generate_c2pa_evidence(c2pa_data)
    ai_item = next(i for i in items if i.validation_status_code == "C2PA_AI_DECLARED")
    assert ai_item.raw_observation["software_agent"] == "gpt-image"
    assert "declaration of AI generation" in ai_item.finding_summary
    assert "does not replace optical or physical analysis nor independently prove real-world scene authorship" in ai_item.limitations[0]
