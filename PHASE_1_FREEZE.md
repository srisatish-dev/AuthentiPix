# AuthentiPix — Phase 1 Freeze 🔒

**Phase 1: Metadata & Provenance**
**Status: FROZEN**
**Freeze Date: 2026-09-07**
**Analyzer Version: 1.0.0**
**Package Version: 0.1.0**

---

## Scope

Phase 1 implements **Pillar 1 — Metadata & Provenance Analysis**, a self-contained forensic metadata extraction and consistency analysis engine. It produces structured **evidence items** with explicit limitations — never binary "real/fake" verdicts.

### Modules (14 source files)

| Module | File | Purpose |
|--------|------|---------|
| Package Root | `authentipix/__init__.py` | Version (`0.1.0`) |
| Metadata Init | `authentipix/metadata/__init__.py` | Public API exports |
| Analyzer Engine | `authentipix/metadata/analyzer.py` | Main pipeline orchestrator |
| Schemas | `authentipix/metadata/schemas.py` | `EvidenceItem`, `NormalizedMetadata`, `AnalyzerOutput` |
| Consistency Engine | `authentipix/metadata/consistency.py` | 7 deterministic consistency rules |
| Normalizer | `authentipix/metadata/normalization.py` | Cross-namespace metadata unification |
| Context | `authentipix/metadata/context.py` | `AnalysisContext` (bytes / file path) |
| Security Enforcer | `authentipix/metadata/security.py` | File size limits, XXE defense, decompression bomb guard |
| Reproducibility | `authentipix/metadata/reproducibility.py` | Runtime environment & rule manifest logging |
| Extractors Init | `authentipix/metadata/extractors/__init__.py` | Extractor registry |
| Base Extractor | `authentipix/metadata/extractors/base.py` | `BaseExtractor` / `ExtractionResult` ABC |
| EXIF Extractor | `authentipix/metadata/extractors/exif.py` | Pillow + ExifRead + optional ExifTool |
| XMP Extractor | `authentipix/metadata/extractors/xmp.py` | XMP/RDF parsing via defusedxml |
| IPTC Extractor | `authentipix/metadata/extractors/iptc.py` | IPTC-IIM header extraction |
| File Container | `authentipix/metadata/extractors/file_extractor.py` | Raster dimensions, ICC, format |
| Preview Extractor | `authentipix/metadata/extractors/preview.py` | IFD1 thumbnail extraction |
| C2PA Extractor | `authentipix/metadata/extractors/c2pa.py` | C2PA SDK integration, 4-dimensional validation |

### Key Design Principles

1. **No binary verdicts** — No `is_real`, `is_fake`, `verdict`, or `confidence_score` fields exist.
2. **Evidence-based output** — Each finding is an `EvidenceItem` with `category`, `evidence_strength`, `validation_status_code`, `raw_observation`, `interpretation`, and `limitations`.
3. **Forensic-safe language** — Interpretations never claim "fake", "fraud", or "manipulation" without explicit limitation caveats.
4. **Missing metadata ≠ manipulation** — Absence of metadata produces a `NEUTRAL` strength `NO_METADATA` item with explicit explanation.
5. **Editing ≠ malicious** — Software presence, dimension mismatches, and temporal inversions include limitations disclaiming malicious intent.

---

## C2PA Interpretation — Verified Correct

| Scenario | Expected Status | Verified |
|----------|----------------|----------|
| Valid content binding + valid signature + trusted cert | `C2PA_VALID_TRUSTED` | ✅ |
| Valid content binding + valid signature + **untrusted** cert | `C2PA_VALID_UNTRUSTED` (NOT `CONTENT_BINDING_FAILURE`) | ✅ |
| Actual data hash mismatch (bytes altered post-signature) | `CONTENT_BINDING_FAILURE` | ✅ |
| Claim signature invalid/corrupted | `SIGNATURE_FAILURE` | ✅ |
| Untrusted timestamp (auxiliary, separate from credential trust) | `TIMESTAMP_UNTRUSTED` (separate evidence item) | ✅ |
| AI-generation declaration via `trainedAlgorithmicMedia` | `C2PA_AI_DECLARED` (separate evidence item, preserved independently) | ✅ |
| No C2PA manifest found | `NO_CREDENTIAL` | ✅ |

### Critical Distinction Verified

- `signingCredential.untrusted` → `C2PA_VALID_UNTRUSTED`, NOT `CONTENT_BINDING_FAILURE`
- `assertion.dataHash.mismatch` → `CONTENT_BINDING_FAILURE` (actual byte alteration)
- These are **never conflated**. Tests `test_regression_a` and `test_regression_b` explicitly enforce this.

---

## Real-World Sample Validation

| Sample | Type | C2PA | Key Findings |
|--------|------|------|-------------|
| `sample1.png` | AI-generated (GPT Image 2.0) | ✅ Present, `C2PA_VALID_UNTRUSTED` | 3 evidence items: `C2PA_AI_DECLARED` (HIGH), `C2PA_VALID_UNTRUSTED` (MEDIUM), `TIMESTAMP_UNTRUSTED` (LOW) |
| `sample2.jpg` | Camera photo (OnePlus Nord 5) | ❌ Absent | No consistency anomalies; pristine EXIF with matching dimensions |

---

## Test Suite — 28/28 PASSED ✅

```
tests/test_c2pa_audit.py          — 4 tests  (C2PA 4-dimensional validation regression)
tests/test_consistency.py         — 6 tests  (Consistency rules 001–007)
tests/test_context.py             — 3 tests  (AnalysisContext factory methods)
tests/test_ground_truth_suite.py  — 8 tests  (Categories A–H ground truth)
tests/test_schemas.py             — 3 tests  (Pydantic schema validation)
tests/test_security.py            — 4 tests  (Security enforcer, XXE defense)
```

### Test Categories Covered

| Category | Description | Test |
|----------|-------------|------|
| A | Pristine camera image | `test_category_a_pristine_camera_image` |
| B | Metadata-only manipulation (temporal inversion) | `test_category_b_metadata_only_manipulation` |
| C | Controlled pixel edits (dimension mismatch) | `test_category_c_controlled_pixel_edits` |
| D | AI-generated provenance (C2PA declaration) | `test_category_d_ai_generated_provenance` |
| E | Metadata-stripped image (inconclusive/neutral) | `test_category_e_metadata_stripped_image` |
| F | Social/media transformed (software detection) | `test_category_f_social_media_recompressed` |
| G | C2PA content-binding failure | `test_category_g_c2pa_content_binding_failure` |
| H | Malformed XXE adversarial input | `test_category_h_malformed_xxe_payload_defense` |

---

## Independence Verification

Phase 1 has **zero dependencies** on future phases:

- ❌ No Pixel Forensics / ELA / noise analysis code
- ❌ No ML / neural network / torch / tensorflow imports
- ❌ No Evidence Fusion engine
- ❌ No FastAPI / REST / web framework
- ❌ No frontend / UI code
- ✅ Only references "Evidence Fusion" in a docstring describing downstream consumption intent

---

## Known Limitations

1. **EXIF-only GPS parsing** — GPS coordinates from `sample2.jpg` show raw null bytes (`\x00`) due to PIL ExifRead fallback; ExifTool path provides higher-fidelity GPS extraction when available.
2. **C2PA trust store** — Uses c2pa-python SDK default trust store. OpenAI-signed images report `signingCredential.untrusted` because OpenAI's CA is not in the default trust list. This is correct behavior.
3. **Timestamp comparison** — Rule 003 uses lexicographic string comparison for temporal inversion; timezone-aware datetime parsing across heterogeneous EXIF formats is deferred.
4. **Container format rule (006)** — Currently only detects JPEG-in-WebP residual metadata; other cross-format scenarios are not yet covered.
5. **No pixel-level analysis** — Phase 1 is metadata-only. ELA, noise analysis, and copy-move detection are Phase 2 scope.
6. **No ML models** — No trained classifiers are used. All rules are deterministic.
7. **No fusion scoring** — Individual evidence items are emitted without aggregation or composite scoring.

---

## Runtime Environment (Freeze Snapshot)

| Dependency | Version |
|------------|---------|
| Python | 3.13.7 |
| Pillow | 11.3.0 |
| c2pa-python | 0.37.10 |
| pyexiftool | 0.5.6 |
| pydantic | ≥2.x |
| defusedxml | ≥0.7.x |
| pytest | 9.1.1 |

---

## Files in Phase 1

```
authentipix/
├── __init__.py
└── metadata/
    ├── __init__.py
    ├── analyzer.py
    ├── consistency.py
    ├── context.py
    ├── normalization.py
    ├── reproducibility.py
    ├── schemas.py
    ├── security.py
    └── extractors/
        ├── __init__.py
        ├── base.py
        ├── c2pa.py
        ├── exif.py
        ├── file_extractor.py
        ├── iptc.py
        ├── preview.py
        └── xmp.py

tests/
├── __init__.py
├── test_fixtures.py
├── test_c2pa_audit.py
├── test_consistency.py
├── test_context.py
├── test_ground_truth_suite.py
├── test_schemas.py
└── test_security.py

run_analysis.py                    # Manual test runner
analysis_result_sample1.json       # Saved output (AI-generated PNG)
analysis_result_sample2.json       # Saved output (camera JPEG)
```
