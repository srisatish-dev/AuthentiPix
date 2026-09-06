"""Manual Test Runner for AuthentiPix Phase 1 (Metadata & Provenance Analyzer).

Prints a clean, concise, high-signal summary to the terminal while saving the complete
raw JSON payload to disk. Compatible with Windows CP1252 console encodings.

Usage:
    py run_analysis.py                       # Analyzes sample1.png and sample2.jpeg
    py run_analysis.py sample1.png           # Analyzes specific image
    py run_analysis.py sample1.png sample2.jpeg  # Analyzes multiple images
"""

import glob
import json
import os
import sys
from typing import Any, Dict, List
from authentipix.metadata import AnalysisContext, MetadataAnalyzer


def detect_mime_type(file_path: str) -> str:
    """Detects MIME type from file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".heic": "image/heic",
        ".heif": "image/heic",
    }
    return mapping.get(ext, "image/jpeg")


def analyze_single_image(image_path: str, analyzer: MetadataAnalyzer):
    """Executes analysis on a single image and prints concise summary to terminal."""
    if not os.path.exists(image_path):
        print(f"\n[ERROR] Target image file not found at: {os.path.abspath(image_path)}")
        return

    abs_path = os.path.abspath(image_path)
    file_name = os.path.basename(image_path)
    mime_type = detect_mime_type(image_path)

    # Ingest image into AnalysisContext
    context = AnalysisContext.from_file(abs_path, mime_type=mime_type)

    # Run MetadataAnalyzer
    output = analyzer.analyze(context)
    output_dict = output.model_dump()
    norm = output.normalized_features
    raw_obs = output_dict.get("raw_observations", {})

    print("\n" + "=" * 75)
    print(f"  AUTHENTIPIX ANALYSIS -- {file_name.upper()}")
    print("=" * 75)
    print(f"  File Path     : {abs_path}")
    print(f"  MIME / Size   : {mime_type} | {round(output.reproducibility_metadata.get('file_size_bytes', 0) / 1024, 1)} KB")
    print(f"  Execution Time: {output.execution_time_ms} ms")

    # 1. Extractor Run Overview
    print("\n[1] EXTRACTORS RUN OVERVIEW")
    extractors = [
        ("EXIF Extractor", bool(norm.get("make") or norm.get("datetime_original") or norm.get("exif_width"))),
        ("XMP Extractor", bool(norm.get("creator_tool") or norm.get("document_id") or norm.get("history_actions"))),
        ("IPTC Extractor", bool(norm.get("byline") or norm.get("copyright_notice"))),
        ("Container Extractor", True),
        ("Preview Extractor", raw_obs.get("preview_extractor", {}).get("thumbnail_present", False)),
        ("C2PA Provenance", raw_obs.get("c2pa_extractor", {}).get("c2pa_present", False)),
    ]

    for label, has_data in extractors:
        icon = "[+] DATA FOUND" if has_data else "[-] NO DATA"
        print(f"  * {label:<22} : {icon}")

    # 2. Key Normalized Metadata Highlights
    print("\n[2] KEY METADATA HIGHLIGHTS")
    make = norm.get("make") or "N/A"
    model = norm.get("model") or "N/A"
    soft = norm.get("software") or norm.get("creator_tool") or "N/A"
    dt_orig = norm.get("datetime_original") or "N/A"
    exif_w = norm.get("exif_width") or "N/A"
    exif_h = norm.get("exif_height") or "N/A"
    rast_w = norm.get("raster_width") or "N/A"
    rast_h = norm.get("raster_height") or "N/A"

    print(f"  * Device Make / Model : {make} / {model}")
    print(f"  * Software / Agent    : {soft}")
    print(f"  * Capture Timestamp   : {dt_orig}")
    print(f"  * EXIF Dimensions     : {exif_w} x {exif_h}")
    print(f"  * Raster Dimensions   : {rast_w} x {rast_h}")

    # C2PA Summary Highlight
    c2pa_info = raw_obs.get("c2pa_extractor", {})
    if c2pa_info.get("c2pa_present"):
        status_code = c2pa_info.get("validation_status_code", "N/A")
        manifest = c2pa_info.get("active_manifest", {})
        gen = manifest.get("claim_generator") or "N/A"
        ai_flag = " (Generative AI Declared)" if manifest.get("ai_generated_declared") else ""
        print(f"  * C2PA Credentials    : PRESENT | Status: {status_code}{ai_flag}")
        print(f"  * Claim Generator     : {gen}")
    else:
        print("  * C2PA Credentials    : ABSENT (No C2PA Manifest Found)")

    # 3. Evidence Items & Consistency Findings
    items = output.evidence_items
    print(f"\n[3] CONSISTENCY & EVIDENCE FINDINGS ({len(items)} items)")

    if not items:
        print("  [OK] No consistency anomalies or evidence items generated.")
    else:
        for idx, item in enumerate(items, start=1):
            print(f"\n  [{idx}] Category  : {item.category.value}")
            print(f"      Strength  : {item.evidence_strength.value}")
            print(f"      Status    : {item.validation_status_code}")
            print(f"      Summary   : {item.finding_summary}")
            print(f"      Details   : {item.interpretation}")

    # 4. Save JSON Export
    clean_name = os.path.splitext(file_name)[0]
    out_filename = f"analysis_result_{clean_name}.json"
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)

    print("\n" + "-" * 75)
    print(f"  [EXPORT] Complete raw JSON saved to: {out_filename}")
    print("-" * 75)


def main():
    analyzer = MetadataAnalyzer()

    # Determine target files
    if len(sys.argv) > 1:
        target_files = sys.argv[1:]
    else:
        # Default: auto-discover sample images in workspace
        target_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.tif", "*.tiff"):
            target_files.extend(glob.glob(ext))
        samples = [f for f in target_files if "sample" in os.path.basename(f).lower()]
        if samples:
            target_files = sorted(samples)

    if not target_files:
        print("[ERROR] No sample images found. Please specify image paths, e.g.:")
        print("  py run_analysis.py sample1.png sample2.jpeg")
        sys.exit(1)

    for img_path in target_files:
        analyze_single_image(img_path, analyzer)


if __name__ == "__main__":
    main()
