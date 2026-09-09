"""AuthentiPix Forensic Analysis Pipeline Runner.

Orchestrates the complete, integrated forensic pipeline:
1. Actual Binary Format Detection & Container Integrity
2. AnalysisContext Ingestion
3. Phase 1: Metadata & Provenance Analyzer (EXIF, C2PA, XMP)
4. Phase 2: Basic Pixel Forensics Engine (Histograms, Spatial Edges)
5. Phase 3A: High-Frequency Residual & Noise-Characteristic Analyzer
6. Candidate Screenshot & Recapture Indicators
7. Transformation Awareness & Signal Qualification
8. Evidence Synthesis & Bounded Forensic Assessment

Usage:
    py run_analysis.py                          # Analyzes all sample images in workspace
    py run_analysis.py sample1.jpg              # Analyzes a specific image
    py run_analysis.py sample1.jpg sample2.png  # Analyzes multiple images
"""

import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional
import numpy as np

from authentipix.metadata import AnalysisContext, MetadataAnalyzer
from authentipix.pixel.engine import PixelForensicsEngine
from authentipix.pixel.loader import PixelImageLoader
from authentipix.pixel.residual import ResidualNoiseAnalyzer
from authentipix.robustness import (
    analyze_screenshot_indicators,
    analyze_transformations,
    detect_actual_format,
    synthesize_robustness_evidence,
)
from authentipix.robustness.schemas import RobustnessReport


def natural_sort_key(s: str):
    """Sort strings containing numbers in natural human order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def format_num(val: Any, precision: int = 6) -> str:
    """Safely formats floats and numerical values, preserving None cleanly."""
    if val is None:
        return "None"
    if isinstance(val, (float, np.floating)):
        return f"{val:.{precision}f}"
    return str(val)


def analyze_single_image(
    image_path: str,
    metadata_analyzer: MetadataAnalyzer,
    pixel_engine: PixelForensicsEngine,
    residual_analyzer: ResidualNoiseAnalyzer,
    pixel_loader: PixelImageLoader,
) -> Optional[RobustnessReport]:
    """Executes the complete AuthentiPix pipeline on a single image asset."""
    abs_path = os.path.abspath(image_path)
    file_name = os.path.basename(image_path)

    if not os.path.exists(abs_path):
        print(f"\n[ERROR] Target image file not found at: {abs_path}")
        return None

    # 1. Actual Binary Format Detection (independent of filename extension)
    format_result = detect_actual_format(abs_path)
    mime_type = format_result.mime_type

    # 2. Ingest into AnalysisContext using the verified actual MIME type
    context = AnalysisContext.from_file(abs_path, mime_type=mime_type)

    # 3. Phase 1: Metadata & Provenance Analyzer
    meta_output = metadata_analyzer.analyze(context)
    meta_dict = meta_output.model_dump()
    norm = meta_output.normalized_features
    raw_obs = meta_dict.get("raw_observations", {})

    # 4. Phase 2: Basic Pixel Forensics Engine
    pixel_result = pixel_engine.analyze(context)
    pixel_dict = pixel_result.model_dump()
    props = pixel_result.properties

    # 5. Phase 3A: Residual & Noise-Characteristic Analyzer
    residual_result = None
    residual_dict = {}
    try:
        arr, residual_props = pixel_loader.load_pixel_array(context)
        residual_result = residual_analyzer.analyze(arr, residual_props)
        residual_dict = residual_result.model_dump()
    except Exception as e:
        print(f"  [!] Phase 3A analysis failed or skipped for {file_name}: {e}")

    # 6. Candidate Screenshot & Recapture Indicators
    screenshot_result = analyze_screenshot_indicators(
        width=props.width,
        height=props.height,
        pixel_result=pixel_result,
        residual_result=residual_result,
    )

    # 7. Transformation Awareness
    transformation_result = analyze_transformations(
        format_result=format_result,
        metadata_output=meta_output,
        pixel_result=pixel_result,
        residual_result=residual_result,
        screenshot_result=screenshot_result,
    )

    # 8. Evidence Synthesis & Final Bounded Assessment
    report = synthesize_robustness_evidence(
        format_result=format_result,
        metadata_output=meta_output,
        pixel_result=pixel_result,
        residual_result=residual_result,
        screenshot_result=screenshot_result,
        transformation_result=transformation_result,
    )

    # -------------------------------------------------------------------------
    # PRINT STANDARDIZED FORENSIC REPORT (Schema Section 12)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("AUTHENTIPIX FORENSIC REPORT")
    print("=" * 75)

    # FILE SECTION
    print("\nFILE")
    print("-" * 75)
    print(f"{'Filename':<28}: {format_result.filename}")
    print(f"{'Filename Extension':<28}: {format_result.filename_extension}")
    print(f"{'Actual Format':<28}: {format_result.actual_format}")
    mismatch_str = "TRUE" if format_result.format_mismatch_detected else "FALSE"
    print(f"{'Format Mismatch Detected':<28}: {mismatch_str}")
    print(f"{'Dimensions':<28}: {props.width} x {props.height}")
    print(f"{'Aspect Ratio':<28}: {props.aspect_ratio:.4f}")

    # PROVENANCE EVIDENCE SECTION
    print("\nPROVENANCE EVIDENCE")
    print("-" * 75)
    c2pa_info = raw_obs.get("c2pa_extractor", {})
    c2pa_present = c2pa_info.get("c2pa_present", False)
    manifest = c2pa_info.get("active_manifest", {})
    ai_declared = manifest.get("ai_generated_declared", False)
    c2pa_str = "PRESENT" if c2pa_present else "NOT DETECTED IN CURRENT FILE"
    ai_str = "YES" if ai_declared else "NO"

    make = norm.get("make")
    model = norm.get("model")
    dt_orig = norm.get("datetime_original")
    exif_present = bool(make or model or dt_orig or norm.get("exif_width"))
    exif_str = "PRESENT" if exif_present else "NOT DETECTED IN CURRENT FILE"
    cam_str = f"{make} / {model}" if (make or model) else "N/A"

    print(f"{'C2PA':<28}: {c2pa_str}")
    print(f"{'C2PA AI Declaration':<28}: {ai_str}")
    print(f"{'EXIF Camera Metadata':<28}: {exif_str}")
    print(f"{'Camera Make / Model':<28}: {cam_str}")
    print(f"{'Capture Timestamp':<28}: {dt_orig or 'N/A'}")
    print(f"{'Provenance Assessment':<28}: {report.provenance_assessment}")

    # PIXEL / RESIDUAL EVIDENCE SECTION
    print("\nPIXEL / RESIDUAL EVIDENCE")
    print("-" * 75)
    luma_hist = next((h for h in pixel_result.histograms if h.channel_name == "luma"), None) or (
        pixel_result.histograms[0] if pixel_result.histograms else None
    )
    entropy_str = f"{luma_hist.entropy:.4f} bits/pixel" if luma_hist else "N/A"
    occupied_str = f"{luma_hist.occupied_bin_count} / 256 bins" if luma_hist else "N/A"
    print(f"{'Histogram Statistics':<28}: Entropy {entropy_str} | Occupancy {occupied_str}")

    if residual_result and residual_result.edge_excluded_luma_stats:
        ls = residual_result.edge_excluded_luma_stats
        print(f"{'Residual Mean':<28}: {format_num(ls.mean)}")
        print(f"{'Residual Std':<28}: {format_num(ls.std)}")
        print(f"{'Residual Robust Std':<28}: {format_num(ls.robust_std)}")
        print(f"{'Residual Skewness':<28}: {format_num(ls.skewness)}")
        print(f"{'Residual Kurtosis':<28}: {format_num(ls.kurtosis)}")
    else:
        print(f"{'Residual Luma Stats':<28}: N/A")

    if residual_result and residual_result.spatial_descriptors:
        sp = residual_result.spatial_descriptors
        print(f"{'Horizontal Autocorrelation':<28}: {format_num(sp.horizontal_lag1_autocorrelation)}")
        print(f"{'Vertical Autocorrelation':<28}: {format_num(sp.vertical_lag1_autocorrelation)}")
        print(f"{'SHI':<28}: {format_num(sp.spatial_heterogeneity_index)}")
    else:
        print(f"{'Spatial Descriptors':<28}: N/A")

    if residual_result and residual_result.channel_descriptors:
        ch = residual_result.channel_descriptors
        rg = format_num(ch.rg_correlation)
        rb = format_num(ch.rb_correlation)
        gb = format_num(ch.gb_correlation)
        print(f"{'RGB Correlations':<28}: RG={rg} | RB={rb} | GB={gb}")
    else:
        print(f"{'RGB Correlations':<28}: N/A")

    # CANDIDATE TRANSFORMATION / RECAPTURE INDICATORS SECTION
    print("\nCANDIDATE TRANSFORMATION / RECAPTURE INDICATORS")
    print("-" * 75)
    geom_str = report.screenshot_indicators.display_geometry_match.value
    profile_str = f" ({report.screenshot_indicators.matched_display_profile})" if report.screenshot_indicators.matched_display_profile else ""
    print(f"{'Display Geometry Match':<28}: {geom_str}{profile_str}")
    print(f"{'Viewport-like Dimensions':<28}: {report.screenshot_indicators.dimensions}")
    print(f"{'Residual Observation':<28}: {report.screenshot_indicators.residual_observation}")
    print(f"{'Autocorrelation Obs':<28}: {report.screenshot_indicators.autocorrelation_observation}")
    print(f"{'SHI Observation':<28}: {report.screenshot_indicators.shi_observation}")
    print(f"{'Dynamic Range Observation':<28}: {report.screenshot_indicators.dynamic_range_observation}")
    print(f"{'Overall Recapture Status':<28}: {report.screenshot_indicators.candidate_status.value}")

    # TRANSFORMATION AWARENESS SECTION
    print("\nTRANSFORMATION AWARENESS")
    print("-" * 75)
    ta = report.transformation_awareness
    print(f"{'Potential Metadata Loss':<28}: {ta.potential_metadata_loss.value}")
    print(f"{'Potential Re-encoding':<28}: {ta.potential_re_encoding.value}")
    print(f"{'Potential Resizing':<28}: {ta.potential_resizing.value}")
    print(f"{'Potential Recapture':<28}: {ta.potential_recapture.value}")
    if ta.detected_transformations:
        print("Detected Transformations  :")
        for t in ta.detected_transformations:
            print(f"  - {t}")

    # SYNTHESIZED ASSESSMENT SECTION
    print("\nSYNTHESIZED ASSESSMENT")
    print("-" * 75)
    sa = report.synthesized_assessment
    print(f"{'Assessment':<28}: {sa.assessment}")
    print(f"{'Confidence / Evidence Tier':<28}: {sa.evidence_tier.value}")
    print(f"{'Original Source':<28}: {sa.original_source}")
    print(f"{'Original AI Provenance':<28}: {sa.original_ai_provenance}")
    print(f"{'Primary Rationale':<28}: {sa.primary_rationale}")
    print("Supporting Observations   :")
    for s in sa.supporting_observations:
        print(f"  - {s}")
    print("Limitations               :")
    for lim in sa.limitations:
        print(f"  - {lim}")
    print("=" * 75)

    # Export structured JSON payload
    clean_name = os.path.splitext(file_name)[0]
    out_filename = f"analysis_result_{clean_name}.json"
    combined_export = {
        "filename": file_name,
        "actual_format": format_result.actual_format,
        "format_mismatch_detected": "YES" if format_result.format_mismatch_detected else "NO",
        "exif": exif_present,
        "c2pa": c2pa_present,
        "provenance_assessment": report.provenance_assessment,
        "final_synthesized_assessment": sa.assessment,
        "evidence_tier": sa.evidence_tier.value,
        "original_source": sa.original_source,
        "candidate_recapture_indicators": report.screenshot_indicators.candidate_status.value,
        "robustness": {
            **report.model_dump(),
            "actual_format": format_result.actual_format,
            "format_mismatch_detected": "YES" if format_result.format_mismatch_detected else "NO",
            "provenance_assessment": report.provenance_assessment,
            "final_synthesized_assessment": sa.assessment,
            "evidence_tier": sa.evidence_tier.value,
            "original_source": sa.original_source,
            "candidate_recapture_indicators": report.screenshot_indicators.candidate_status.value,
        },
        "format_detection": format_result.model_dump(),
        "metadata_analysis": meta_dict,
        "pixel_analysis": pixel_dict,
        "residual_analysis": residual_dict,
        "screenshot_indicators": report.screenshot_indicators.model_dump(),
        "transformation_awareness": report.transformation_awareness.model_dump(),
        "synthesized_assessment": sa.model_dump(),
    }
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(combined_export, f, indent=2, default=str)
    print(f"\n  [EXPORT] Complete forensic JSON payload saved to: {out_filename}")

    return report


def main():
    metadata_analyzer = MetadataAnalyzer()
    pixel_engine = PixelForensicsEngine()
    residual_analyzer = ResidualNoiseAnalyzer()
    pixel_loader = PixelImageLoader()

    # Determine target files from argv or workspace discovery
    if len(sys.argv) > 1:
        target_files = sys.argv[1:]
    else:
        target_files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.tif", "*.tiff"):
            target_files.extend(glob.glob(ext))
        # Natural human order for sample images
        samples = [f for f in target_files if "sample" in os.path.basename(f).lower()]
        if samples:
            target_files = sorted(samples, key=natural_sort_key)
        else:
            target_files = sorted(target_files, key=natural_sort_key)

    if not target_files:
        print("[ERROR] No sample images found in current directory. Please specify image paths:")
        print("  py run_analysis.py sample1.jpg sample2.png")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("  AUTHENTIPIX FORENSIC ANALYSIS PIPELINE")
    print(f"  Target assets: {len(target_files)} image(s)")
    print("=" * 75)

    summary_records = []
    for img_path in target_files:
        report = analyze_single_image(
            img_path,
            metadata_analyzer,
            pixel_engine,
            residual_analyzer,
            pixel_loader,
        )
        if report:
            summary_records.append(report)

    # Print Batch Summary Table across all analyzed images
    if summary_records:
        print("\n" + "=" * 120)
        print("AUTHENTIPIX BATCH SUMMARY")
        print("=" * 120)
        headers = [
            "Image",
            "Extension",
            "Actual Format",
            "Mismatch",
            "EXIF",
            "C2PA",
            "Recapture Indicators",
            "Provenance",
            "Final Assessment",
        ]
        rows = []
        for r in summary_records:
            fd = r.format_detection
            si = r.screenshot_indicators
            sa = r.synthesized_assessment
            exif_val = "YES" if "CAMERA EXIF" in r.provenance_assessment or "PARTIAL EXIF" in r.provenance_assessment else "NO"
            c2pa_val = "YES" if "C2PA" in r.provenance_assessment else "NO"
            mismatch_val = "YES" if fd.format_mismatch_detected else "NO"
            recapture_val = si.candidate_status.value
            prov_short = "DECLARED AI" if "AI" in r.provenance_assessment else ("CAMERA EXIF" if exif_val == "YES" else "NONE")

            rows.append([
                fd.filename,
                fd.filename_extension,
                fd.actual_format,
                mismatch_val,
                exif_val,
                c2pa_val,
                recapture_val,
                prov_short,
                sa.assessment,
            ])

        col_w = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_w[i] = max(col_w[i], len(str(val)))

        h_line = "| " + " | ".join(f"{headers[i]:<{col_w[i]}}" for i in range(len(headers))) + " |"
        s_line = "|-" + "-|-".join("-" * col_w[i] for i in range(len(headers))) + "-|"
        print(h_line)
        print(s_line)
        for row in rows:
            print("| " + " | ".join(f"{str(row[i]):<{col_w[i]}}" for i in range(len(row))) + " |")
        print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
