"""Manual Test Runner for AuthentiPix (Metadata & Pixel Forensics Engine).

Prints a clean, concise, high-signal summary of Phase 1 (Metadata) and Phase 2 (Pixel Analysis)
to the terminal while saving the complete raw JSON payload to disk.

Usage:
    py run_analysis.py                       # Analyzes sample images in workspace
    py run_analysis.py sample1.png           # Analyzes specific image
    py run_analysis.py sample1.png sample2.jpg  # Analyzes multiple images
"""

import glob
import json
import os
import sys
from typing import Any, Dict, List

from authentipix.metadata import AnalysisContext, MetadataAnalyzer
from authentipix.pixel.engine import PixelForensicsEngine


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


def analyze_single_image(
    image_path: str, metadata_analyzer: MetadataAnalyzer, pixel_engine: PixelForensicsEngine
):
    """Executes Phase 1 & Phase 2 analysis on a single image and prints summary to terminal."""
    if not os.path.exists(image_path):
        print(f"\n[ERROR] Target image file not found at: {os.path.abspath(image_path)}")
        return

    abs_path = os.path.abspath(image_path)
    file_name = os.path.basename(image_path)
    mime_type = detect_mime_type(image_path)

    # Ingest image into AnalysisContext
    context = AnalysisContext.from_file(abs_path, mime_type=mime_type)

    # Execute Phase 1: Metadata & Provenance Analyzer
    meta_output = metadata_analyzer.analyze(context)
    meta_dict = meta_output.model_dump()
    norm = meta_output.normalized_features
    raw_obs = meta_dict.get("raw_observations", {})

    # Execute Phase 2: Basic Pixel Forensics Engine
    pixel_result = pixel_engine.analyze(context)
    pixel_dict = pixel_result.model_dump()

    print("\n" + "=" * 75)
    print(f"  AUTHENTIPIX ANALYSIS -- {file_name.upper()}")
    print("=" * 75)
    print(f"  File Path     : {abs_path}")
    print(f"  MIME / Size   : {mime_type} | {round(meta_output.reproducibility_metadata.get('file_size_bytes', 0) / 1024, 1)} KB")
    print(f"  Execution Time: Metadata {meta_output.execution_time_ms} ms | Pixel {pixel_result.execution_time_ms} ms")

    # -------------------------------------------------------------------------
    # PILLAR 1: METADATA & PROVENANCE SUMMARY
    # -------------------------------------------------------------------------
    print("\n[1] PILLAR 1: METADATA & PROVENANCE OVERVIEW")
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

    print("\n[2] METADATA HIGHLIGHTS")
    make = norm.get("make") or "N/A"
    model = norm.get("model") or "N/A"
    soft = norm.get("software") or norm.get("creator_tool") or "N/A"
    dt_orig = norm.get("datetime_original") or "N/A"
    exif_w = norm.get("exif_width") or "N/A"
    exif_h = norm.get("exif_height") or "N/A"

    print(f"  * Device Make / Model : {make} / {model}")
    print(f"  * Software / Agent    : {soft}")
    print(f"  * Capture Timestamp   : {dt_orig}")
    print(f"  * EXIF Dimensions     : {exif_w} x {exif_h}")

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

    # -------------------------------------------------------------------------
    # PILLAR 2: BASIC PIXEL FORENSICS SUMMARY
    # -------------------------------------------------------------------------
    print("\n[3] PILLAR 2: BASIC PIXEL FORENSICS SUMMARY")
    props = pixel_result.properties
    print(f"  * Grid Geometry       : {props.width} x {props.height} ({props.pixel_count:,} pixels | Aspect {round(props.aspect_ratio, 3)})")
    print(f"  * Format & Channels   : Mode '{props.native_mode}' | {props.channel_count} Channels | {props.bit_depth}-bit ({props.dtype})")
    if props.has_alpha:
        print("  * Alpha Channel       : PRESENT (Isolated from Luma & RGB stats)")
    if props.was_palette_converted or props.was_cmyk_converted or props.was_bilevel_converted:
        conv_type = "Palette" if props.was_palette_converted else ("CMYK" if props.was_cmyk_converted else "Bilevel")
        print(f"  * Mode Conversion     : Converted from {conv_type} mode for analysis")

    # Photometric Luma & Intensity Summary
    luma = pixel_result.luma_statistics
    if luma:
        print(f"  * sRGB BT.709 Luma    : Mean {luma.mean:.2f} | StdDev {luma.std_dev:.2f} | Range [{luma.min:.0f}, {luma.max:.0f}]")
    
    # Histogram & Comb Features
    luma_hist = next((h for h in pixel_result.histograms if h.channel_name == "luma"), None) or (
        pixel_result.histograms[0] if pixel_result.histograms else None
    )
    if luma_hist:
        print(f"  * Shannon Entropy     : {luma_hist.entropy:.4f} bits/pixel (Max 8.0)")
        print(f"  * Occupied Bins       : {luma_hist.occupied_bin_count} / 256 bins")
        print(f"  * Boundary Clipping   : Shadow (0): {luma_hist.clipping_fraction_0 * 100:.2f}% | Highlight (255): {luma_hist.clipping_fraction_255 * 100:.2f}%")
        print(f"  * Interior Comb Ratio : {luma_hist.comb_metric:.4f} (C_comb)")

    # Spatial Edge Descriptors
    spatial = pixel_result.spatial_descriptors
    if spatial:
        print(f"  * Sobel Gradient Mean : {spatial.gradient_mean:.2f} / 255.0 (StdDev {spatial.gradient_std:.2f})")
        print(f"  * Edge Density (T=30) : {spatial.edge_density * 100:.2f}% of pixels")

    # Warnings & Evidence Findings
    print(f"\n[4] EVIDENCE & CONSISTENCY FINDINGS ({len(meta_output.evidence_items) + len(pixel_result.evidence_items)} items)")
    all_items = meta_output.evidence_items + pixel_result.evidence_items
    if not all_items:
        print("  [OK] No consistency anomalies or evidence items generated.")
    else:
        for idx, item in enumerate(all_items, start=1):
            print(f"\n  [{idx}] Pillar    : {item.pillar}")
            print(f"      Category  : {item.category.value}")
            print(f"      Strength  : {item.evidence_strength.value}")
            print(f"      Status    : {item.validation_status_code}")
            print(f"      Summary   : {item.finding_summary}")
            print(f"      Details   : {item.interpretation}")

    if pixel_result.warnings:
        print("\n[5] PIXEL ANALYSIS WARNINGS")
        for w in pixel_result.warnings:
            print(f"  [!] {w}")

    # Save Combined JSON Payload Export
    combined_export = {
        "metadata_analysis": meta_dict,
        "pixel_analysis": pixel_dict,
    }

    clean_name = os.path.splitext(file_name)[0]
    out_filename = f"analysis_result_{clean_name}.json"
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(combined_export, f, indent=2)

    print("\n" + "-" * 75)
    print(f"  [EXPORT] Complete raw JSON saved to: {out_filename}")
    print("-" * 75)


def main():
    metadata_analyzer = MetadataAnalyzer()
    pixel_engine = PixelForensicsEngine()

    # Determine target files
    if len(sys.argv) > 1:
        target_files = sys.argv[1:]
    else:
        target_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.tif", "*.tiff"):
            target_files.extend(glob.glob(ext))
        samples = [f for f in target_files if "sample" in os.path.basename(f).lower()]
        if samples:
            target_files = sorted(samples)

    if not target_files:
        print("[ERROR] No sample images found. Please specify image paths, e.g.:")
        print("  py run_analysis.py sample1.png sample2.jpg")
        sys.exit(1)

    for img_path in target_files:
        analyze_single_image(img_path, metadata_analyzer, pixel_engine)


if __name__ == "__main__":
    main()
