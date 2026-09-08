"""Temporary Standalone Experiment Script: AI vs Real Image Comparison & Phase 3A Test.

IMPORTANT:
This is a temporary standalone testing script to experiment with image files in the project root
using the existing Phase 1 (Metadata & Provenance), Phase 2 (Basic Pixel Forensics), and
Phase 3A (Residual & Noise-Characteristic Analysis) pipelines.
It automatically discovers supported images regardless of file extension and detects actual image formats.
It is NOT a machine-learning detector and does NOT modify the AuthentiPix core architecture.
"""

import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from authentipix.metadata import AnalysisContext, MetadataAnalyzer
from authentipix.pixel.engine import PixelForensicsEngine
from authentipix.pixel.loader import PixelImageLoader
from authentipix.pixel.residual import ResidualNoiseAnalyzer
from authentipix.pixel.schemas import (
    ResidualNoiseAnalysisResult,
    ResidualStatisticalMoments,
)

# Optional explicit list of real image paths for testing.
# If empty, automatically discovers all supported image files in the project root.
REAL_IMAGE_PATHS: List[str] = [
    # "sample1.jpeg",
    # "sample2.png",
    # "sample3.jpeg",
    # "sample4.png",
]

# List of extensions explicitly ignored during discovery (non-image files)
IGNORED_EXTENSIONS = {
    ".py", ".json", ".txt", ".md", ".log", ".lock", ".gitignore",
    ".yml", ".yaml", ".git", ".pyc", ".exe", ".dll", ".so"
}


def discover_image_files(root_dir: str = ".") -> List[str]:
    """Discovers supported image files in the given directory by attempting Pillow decoding."""
    discovered = []
    for entry in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path):
            continue
        if entry.startswith("."):
            continue

        ext = os.path.splitext(entry)[1].lower()
        if ext in IGNORED_EXTENSIONS:
            continue

        # Validate with Pillow if it's a valid image
        try:
            with Image.open(full_path) as img:
                fmt = img.format
                if fmt:
                    discovered.append(full_path)
        except Exception:
            # Unsupported file or decoding error
            continue

    return discovered


def get_image_format_info(file_path: str) -> Dict[str, str]:
    """Inspects file with Pillow to determine actual encoded image format, MIME type, extension, dimensions, and mode."""
    ext = os.path.splitext(file_path)[1]
    ext_str = ext if ext else "(none)"
    with Image.open(file_path) as img:
        actual_format = img.format or "UNKNOWN"
        width, height = img.size
        mode = img.mode

    format_upper = actual_format.upper()
    mime_type = getattr(Image, "MIME", {}).get(format_upper)
    if not mime_type:
        mime_map = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "BMP": "image/bmp",
            "TIFF": "image/tiff",
            "GIF": "image/gif",
        }
        mime_type = mime_map.get(format_upper, f"image/{actual_format.lower()}")

    return {
        "actual_format": actual_format,
        "filename_ext": ext_str,
        "mime_type": mime_type,
        "dimensions": f"{width} x {height}",
        "mode": mode,
    }


def analyze_sample(
    file_path: str, metadata_analyzer: MetadataAnalyzer, pixel_engine: PixelForensicsEngine
) -> Optional[Dict[str, Any]]:
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        return None

    try:
        format_info = get_image_format_info(abs_path)
    except Exception as e:
        print(f"[SKIP] Unable to decode format for {file_path}: {e}")
        return None

    mime_type = format_info["mime_type"]
    ctx = AnalysisContext.from_file(abs_path, mime_type=mime_type)

    try:
        meta_out = metadata_analyzer.analyze(ctx)
        pixel_out = pixel_engine.analyze(ctx)
    except Exception as e:
        print(f"[SKIP] Forensic analysis failed for {file_path}: {e}")
        return None

    meta_dict = meta_out.model_dump()
    norm = meta_out.normalized_features
    raw_obs = meta_dict.get("raw_observations", {})

    # Provenance Extraction
    make = norm.get("make")
    model = norm.get("model")
    dt_orig = norm.get("datetime_original")
    has_exif = bool(make or model or dt_orig or norm.get("exif_width"))
    exif_str = "PRESENT" if has_exif else "ABSENT"
    cam_str = f"{make} / {model}" if (make or model) else "N/A"

    c2pa_info = raw_obs.get("c2pa_extractor", {})
    c2pa_present = c2pa_info.get("c2pa_present", False)
    c2pa_str = "PRESENT" if c2pa_present else "ABSENT"

    manifest = c2pa_info.get("active_manifest", {})
    ai_declared = bool(manifest.get("ai_generated_declared", False))
    ai_decl_str = "YES" if ai_declared else "NO"

    # Pixel Stats Extraction
    props = pixel_out.properties
    luma_hist = next((h for h in pixel_out.histograms if h.channel_name == "luma"), None) or (
        pixel_out.histograms[0] if pixel_out.histograms else None
    )
    entropy = luma_hist.entropy if luma_hist else 0.0
    occupied = luma_hist.occupied_bin_count if luma_hist else 0
    clip_0 = luma_hist.clipping_fraction_0 if luma_hist else 0.0
    clip_255 = luma_hist.clipping_fraction_255 if luma_hist else 0.0
    comb_ratio = luma_hist.comb_metric if luma_hist else 0.0

    spatial = pixel_out.spatial_descriptors
    sobel_mean = spatial.gradient_mean if spatial else 0.0
    edge_density = spatial.edge_density if spatial else 0.0

    # Assessment Logic
    if c2pa_present and ai_declared:
        assessment = "AI-GENERATED"
        primary_reason = "Strong provenance signal: C2PA manifest explicitly declares generative AI creation."
        supporting = [
            "C2PA manifest is present and cryptographically verified.",
            "Generative AI creation action is explicitly declared.",
            "Phase 2 pixel statistics are reported for forensic context only.",
        ]
        limitation = "The Phase 2 pixel measurements describe the raster payload but do not independently prove AI generation."
    elif has_exif and (make or model or dt_orig):
        assessment = "REAL / CAMERA-ORIGIN INDICATORS"
        primary_reason = f"The image contains camera-associated EXIF metadata identifying {cam_str} and capture timestamp {dt_orig or 'N/A'}."
        supporting = [
            f"EXIF camera metadata is present ({cam_str}).",
            "Image is a standard photographic raster payload.",
            "No C2PA AI-generation declaration was found.",
            "Phase 2 pixel measurements show ordinary continuous image statistics.",
        ]
        limitation = "IMPORTANT: EXIF presence and pixel statistics do NOT prove that an image is genuine. Metadata can be edited or fabricated, and Phase 2 is descriptive rather than an AI detector."
    else:
        assessment = "INCONCLUSIVE / NO STRONG PROVENANCE SIGNAL"
        primary_reason = "No C2PA AI declaration and no camera EXIF metadata found."
        supporting = ["Pixel statistics are recorded for context."]
        limitation = "Metadata absence does not establish manipulation or AI generation."

    return {
        "file_name": os.path.basename(file_path),
        "actual_format": format_info["actual_format"],
        "filename_ext": format_info["filename_ext"],
        "mime_type": format_info["mime_type"],
        "dims": format_info["dimensions"],
        "mode": format_info["mode"],
        "exif_str": exif_str,
        "cam_str": cam_str,
        "dt_orig": dt_orig or "N/A",
        "c2pa_str": c2pa_str,
        "ai_decl_str": ai_decl_str,
        "entropy": entropy,
        "occupied": occupied,
        "clip_0": clip_0,
        "clip_255": clip_255,
        "comb_ratio": comb_ratio,
        "sobel_mean": sobel_mean,
        "edge_density": edge_density,
        "assessment": assessment,
        "primary_reason": primary_reason,
        "supporting": supporting,
        "limitation": limitation,
    }


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    headers = ["Image", "Actual Format", "EXIF", "C2PA", "AI Declaration", "Assessment"]
    table_rows = []
    for r in results:
        exif_val = "YES" if r["exif_str"] == "PRESENT" else "NO"
        c2pa_val = "YES" if r["c2pa_str"] == "PRESENT" else "NO"
        table_rows.append([
            r["file_name"],
            r["actual_format"],
            exif_val,
            c2pa_val,
            r["ai_decl_str"],
            r["assessment"],
        ])

    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = "| " + " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))) + " |"
    sep_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"

    print("-" * 75)
    print("PHASE 1 & 2 SUMMARY TABLE")
    print("-" * 75)
    print(header_line)
    print(sep_line)
    for row in table_rows:
        row_line = "| " + " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row))) + " |"
        print(row_line)
    print("=" * 75)


# ============================================================================
# PHASE 3A — REAL IMAGE RESIDUAL / NOISE ANALYSIS SECTION
# ============================================================================

def format_val(val: Any, precision: int = 6) -> str:
    """Formats values, preserving None cleanly and formatting floats with given precision."""
    if val is None:
        return "None"
    if isinstance(val, (float, np.floating)):
        return f"{val:.{precision}f}"
    return str(val)


def print_moments_block(title: str, stats: Optional[ResidualStatisticalMoments]) -> None:
    """Prints a ResidualStatisticalMoments block."""
    print(f"\n{title}")
    print("-" * len(title))
    if stats is None:
        print("None")
        return
    print(f"sample_count : {stats.sample_count}")
    print(f"mean         : {format_val(stats.mean)}")
    print(f"std          : {format_val(stats.std)}")
    print(f"mad          : {format_val(stats.mad)}")
    print(f"robust_std   : {format_val(stats.robust_std)}")
    print(f"rms          : {format_val(stats.rms)}")
    print(f"skewness     : {format_val(stats.skewness)}")
    print(f"kurtosis     : {format_val(stats.kurtosis)}")


def run_phase_3a_analysis(image_files: List[str]) -> List[Dict[str, Any]]:
    """Runs the canonical Phase 3A ResidualNoiseAnalyzer against real image files."""
    print("\n" + "=" * 75)
    print("        PHASE 3A -- REAL IMAGE RESIDUAL / NOISE ANALYSIS")
    print("=" * 75)

    analyzer = ResidualNoiseAnalyzer()
    loader = PixelImageLoader()
    summary_records = []

    for file_path in image_files:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            print(f"\n[ERROR] Image not found: {file_path}")
            continue

        print("\n" + "=" * 60)
        print("PHASE 3A REAL IMAGE TEST")
        print("=" * 60)
        print(f"Image: {os.path.basename(file_path)}")
        print(f"Path : {abs_path}")
        print("=" * 60)

        try:
            format_info = get_image_format_info(abs_path)
            ctx = AnalysisContext.from_file(abs_path, mime_type=format_info["mime_type"])
            arr, props = loader.load_pixel_array(ctx)
            result = analyzer.analyze(arr, props)
        except Exception as e:
            print(f"\n[ERROR] Phase 3A analysis failed for {file_path}: {e}")
            continue

        # Basic input information
        print("\nInput Characteristics")
        print("---------------------")
        print(f"filename                            : {os.path.basename(file_path)}")
        print(f"image_dimensions                    : {props.width} x {props.height}")
        print(f"actual_format                       : {format_info['actual_format']}")
        print(f"dtype                               : {props.dtype}")
        print(f"channel_count                       : {props.channel_count}")
        print(f"total_pixel_count                   : {result.input_characteristics.total_pixel_count}")
        print(f"input_invalid_float_count           : {format_val(result.input_characteristics.input_invalid_float_count)}")
        print(f"residual_invalid_neighborhood_count : {format_val(result.input_characteristics.residual_invalid_neighborhood_count)}")
        print(f"unmasked_finite_pixel_count         : {format_val(result.input_characteristics.unmasked_finite_pixel_count)}")
        print(f"edge_excluded_finite_pixel_count    : {format_val(result.input_characteristics.edge_excluded_finite_pixel_count)}")
        print(f"coverage_ratio                      : {format_val(result.input_characteristics.coverage_ratio)}")

        # Luma Statistics
        print_moments_block("Unmasked Luma Residual Statistics", result.unmasked_luma_stats)
        print_moments_block("Edge-Excluded Luma Residual Statistics", result.edge_excluded_luma_stats)

        # Spatial Descriptors
        print("\nSpatial Residual Descriptors")
        print("----------------------------")
        sp = result.spatial_descriptors
        if sp is not None:
            print(f"evaluated_block_count           : {sp.evaluated_block_count}")
            print(f"valid_block_count               : {sp.valid_block_count}")
            print(f"local_variance_mean             : {format_val(sp.local_variance_mean)}")
            print(f"local_variance_std              : {format_val(sp.local_variance_std)}")
            print(f"SHI                             : {format_val(sp.spatial_heterogeneity_index)}")
            print(f"horizontal_lag1_autocorrelation : {format_val(sp.horizontal_lag1_autocorrelation)}")
            print(f"vertical_lag1_autocorrelation   : {format_val(sp.vertical_lag1_autocorrelation)}")
        else:
            print("None")

        # RGB Channel Descriptors
        print("\nRGB Residual Descriptors")
        print("------------------------")
        ch = result.channel_descriptors
        if ch is not None:
            print_moments_block("RED", ch.red_stats)
            print_moments_block("GREEN", ch.green_stats)
            print_moments_block("BLUE", ch.blue_stats)

            print("\nRGB Correlations")
            print("----------------")
            print(f"RG : {format_val(ch.rg_correlation)}")
            print(f"RB : {format_val(ch.rb_correlation)}")
            print(f"GB : {format_val(ch.gb_correlation)}")

            print("\nChannel Robust-Std Ratios")
            print("-------------------------")
            print(f"blue_to_green_ratio : {format_val(ch.blue_to_green_ratio)}")
            print(f"red_to_green_ratio  : {format_val(ch.red_to_green_ratio)}")
        else:
            print("Channel descriptors: None (grayscale input)")

        # Quality Flags
        print("\nQuality Flags")
        print("-------------")
        if result.quality_flags:
            for flag in result.quality_flags:
                print(f"- {flag}")
        else:
            print("None (clean analysis)")

        # Limitations
        print("\nLimitations")
        print("-----------")
        for lim in result.limitations:
            print(f"- {lim}")

        # Interpretation Boundary
        print("\nInterpretation Boundary")
        print("-----------------------")
        print("Phase 3A provides deterministic residual/noise measurements and")
        print("observations only.")
        print("")
        print("It does NOT independently prove:")
        print("- AI generation")
        print("- camera capture")
        print("- authenticity")
        print("- tampering")
        print("- malicious manipulation\n")

        # Record for summary table
        cov_str = f"{result.input_characteristics.coverage_ratio:.4f}" if result.input_characteristics.coverage_ratio is not None else "N/A"
        luma_rob = f"{result.edge_excluded_luma_stats.robust_std:.6f}" if (result.edge_excluded_luma_stats and result.edge_excluded_luma_stats.robust_std is not None) else "N/A"
        vb_str = f"{sp.valid_block_count}/{sp.evaluated_block_count}" if sp else "N/A"
        flags_str = ", ".join(result.quality_flags) if result.quality_flags else "None"

        summary_records.append({
            "image": os.path.basename(file_path),
            "size": f"{props.width}x{props.height}",
            "coverage": cov_str,
            "luma_robust_std": luma_rob,
            "valid_blocks": vb_str,
            "flags": flags_str,
        })

    return summary_records


def print_phase_3a_summary_table(summary_records: List[Dict[str, Any]]) -> None:
    """Prints a compact measurements-only summary table for Phase 3A."""
    if not summary_records:
        return

    headers = ["Image", "Size", "Coverage", "Luma Robust Std", "Valid Blocks", "Flags"]
    table_rows = []
    for r in summary_records:
        table_rows.append([
            r["image"],
            r["size"],
            r["coverage"],
            r["luma_robust_std"],
            r["valid_blocks"],
            r["flags"],
        ])

    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = "| " + " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))) + " |"
    sep_line = "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"

    print("=" * 80)
    print("PHASE 3A REAL IMAGE SUMMARY")
    print("=" * 80)
    print(header_line)
    print(sep_line)
    for row in table_rows:
        row_line = "| " + " | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row))) + " |"
        print(row_line)
    print("=" * 80 + "\n")


def main():
    metadata_analyzer = MetadataAnalyzer()
    pixel_engine = PixelForensicsEngine()

    root_dir = "."
    if REAL_IMAGE_PATHS:
        image_files = REAL_IMAGE_PATHS
    else:
        image_files = discover_image_files(root_dir)

    print("=" * 75)
    print("        AUTHENTIPIX -- AUTOMATIC IMAGE DISCOVERY & FORENSICS TEST")
    print("=" * 75)
    print("\nNOTE:")
    print("This is an experimental heuristic comparison only.")
    print("It is NOT a trained AI detector and pixel statistics alone cannot")
    print("prove whether an image is AI-generated or genuine.\n")

    if not image_files:
        print("[INFO] No supported image files discovered in project root.")
        return

    # 1. Run Phase 1 & Phase 2 Pipeline
    results = []
    for file_path in image_files:
        res = analyze_sample(file_path, metadata_analyzer, pixel_engine)
        if res:
            results.append(res)

    for res in results:
        print("-" * 75)
        print(f"IMAGE: {res['file_name']}")
        print("-" * 75)

        print("\nFILE FORMAT")
        print(f"Actual format : {res['actual_format']}")
        print(f"Filename ext  : {res['filename_ext']}")
        print(f"MIME type     : {res['mime_type']}")

        print("\nPILLAR 1 -- PROVENANCE")
        print(f"C2PA                  : {res['c2pa_str']}")
        print(f"C2PA AI DECLARATION   : {res['ai_decl_str']}")
        print(f"EXIF                  : {res['exif_str']}")
        print(f"Camera Make / Model   : {res['cam_str']}")
        if res["dt_orig"] != "N/A":
            print(f"Capture Timestamp     : {res['dt_orig']}")

        print("\nPILLAR 2 -- PIXEL OBSERVATIONS")
        print(f"Dimensions            : {res['dims']}")
        print(f"Mode                  : {res['mode']}")
        print(f"Entropy               : {res['entropy']:.4f} / 8.0")
        print(f"Histogram Occupancy   : {res['occupied']} / 256")
        print(f"Shadow Clipping       : {res['clip_0'] * 100:.2f}%")
        print(f"Highlight Clipping    : {res['clip_255'] * 100:.2f}%")
        print(f"Comb Ratio            : {res['comb_ratio']:.4f}")
        print(f"Sobel Mean            : {res['sobel_mean']:.2f}")
        print(f"Edge Density          : {res['edge_density'] * 100:.2f}%")

        print("\nASSESSMENT")
        print("----------------")
        print(f"{res['assessment']}")
        print(f"\nPRIMARY REASON:\n{res['primary_reason']}")
        print("\nSUPPORTING OBSERVATIONS:")
        for s in res["supporting"]:
            print(f"- {s}")
        print(f"\nLIMITATION:\n{res['limitation']}\n")

    if results:
        print_summary_table(results)

    # 2. Run Phase 3A Residual / Noise Analysis Pipeline
    phase_3a_records = run_phase_3a_analysis(image_files)
    if phase_3a_records:
        print_phase_3a_summary_table(phase_3a_records)


if __name__ == "__main__":
    main()


