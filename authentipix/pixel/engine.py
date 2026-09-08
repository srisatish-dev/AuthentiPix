"""Orchestration engine for Phase 2: Basic Pixel Analysis.

Pipeline:
AnalysisContext -> PixelImageLoader -> Statistics -> Histograms -> Spatial -> PixelAnalysisResult
"""

import time
from typing import List, Optional
import numpy as np

from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.reproducibility import ReproducibilityManager
from authentipix.metadata.schemas import EvidenceItem, EvidenceCategory, EvidenceStrength
from authentipix.pixel.loader import PixelImageLoader, UnsupportedImageModeError
from authentipix.pixel.statistics import extract_all_statistics, compute_bt709_luma
from authentipix.pixel.histogram import compute_histogram_features
from authentipix.pixel.spatial import compute_spatial_descriptors
from authentipix.pixel.schemas import (
    PixelAnalysisResult,
    PixelProperties,
    ChannelStatistics,
    HistogramFeatures,
    SpatialDescriptors,
)


class PixelForensicsEngine:
    """Orchestrates Phase 2 Basic Pixel Analysis over an AnalysisContext asset."""

    def __init__(self, loader: Optional[PixelImageLoader] = None):
        self.loader = loader or PixelImageLoader()
        self.reproducibility_manager = ReproducibilityManager()

    def analyze(self, ctx: AnalysisContext) -> PixelAnalysisResult:
        """Executes full Basic Pixel Analysis pipeline on the supplied context asset."""
        start_time = time.time()
        warnings: List[str] = []
        errors: List[str] = []
        evidence_items: List[EvidenceItem] = []

        # Step 1: Load image array & extract physical properties
        try:
            arr, props = self.loader.load_pixel_array(ctx)
        except UnsupportedImageModeError as e:
            err_msg = str(e)
            errors.append(err_msg)
            # Create structural warning evidence item
            evidence_items.append(
                EvidenceItem(
                    evidence_id=f"ev_px_unsupported_mode_{ctx.image_id}",
                    pillar="pixel_forensics",
                    analyzer_id="pillar2_basic_pixel_analyzer",
                    category=EvidenceCategory.METADATA_STRUCTURE,
                    evidence_strength=EvidenceStrength.LOW,
                    validation_status_code="UNSUPPORTED_IMAGE_MODE",
                    raw_observation={"mime_type": ctx.mime_type, "error": err_msg},
                    finding_summary="Unsupported image mode encountered during pixel analysis.",
                    interpretation="Image mode could not be loaded into standard pixel array.",
                    limitations=["Image decoding failed or mode is unconvertible."],
                    analyzer_version="0.1.0",
                )
            )
            dummy_props = PixelProperties(
                width=1,
                height=1,
                pixel_count=1,
                aspect_ratio=1.0,
                native_mode="UNKNOWN",
                channel_count=1,
                bit_depth=8,
                dtype="uint8",
            )
            return PixelAnalysisResult(
                properties=dummy_props,
                errors=errors,
                warnings=warnings,
                evidence_items=evidence_items,
                reproducibility_metadata=self.reproducibility_manager.generate_reproducibility_payload(ctx),
            )

        # Audit conversion warnings
        if props.was_palette_converted:
            warnings.append(f"Native palette mode '{props.native_mode}' was converted to RGB/RGBA for analysis.")
        if props.was_cmyk_converted:
            warnings.append("Native CMYK mode was converted to RGB for analysis.")
        if props.was_bilevel_converted:
            warnings.append("Native 1-bit bilevel mode was converted to 8-bit grayscale for analysis.")
        if props.nan_pixel_count > 0:
            warnings.append(f"Encountered {props.nan_pixel_count} NaN pixel values during floating-point analysis.")
        if props.inf_pixel_count > 0:
            warnings.append(f"Encountered {props.inf_pixel_count} +/-Inf pixel values during floating-point analysis.")

        # Step 2: Channel statistics, Luma, and Intensity
        channel_stats, luma_stats, intensity_stats = extract_all_statistics(arr, props)

        # Step 3: Histograms & Entropy
        histograms: List[HistogramFeatures] = []
        C = props.channel_count
        channel_names = ["red", "green", "blue", "alpha"] if C == 4 else (["red", "green", "blue"] if C == 3 else ["gray"])

        for i in range(C):
            c_name = channel_names[i] if i < len(channel_names) else f"channel_{i}"
            c_data = arr[:, :, i]
            hist_feat = compute_histogram_features(c_data, c_name)
            histograms.append(hist_feat)

        # Calculate Luma histogram if Luma stats exist
        luma_arr = compute_bt709_luma(arr)
        if luma_arr is not None:
            luma_hist = compute_histogram_features(luma_arr, "luma")
            histograms.append(luma_hist)

        # Step 4: Spatial gradient descriptors
        # Determine 2D array for Sobel gradient computation (use Luma if color image, else grayscale channel 0)
        if luma_arr is not None:
            spatial_data = luma_arr
        else:
            spatial_data = arr[:, :, 0]

        spatial_desc = compute_spatial_descriptors(spatial_data, edge_threshold=30.0)

        # Build reproducibility payload using Phase 1 ReproducibilityManager
        repro_metadata = self.reproducibility_manager.generate_reproducibility_payload(ctx)
        repro_metadata["pixel_analyzer_version"] = "0.1.0"
        repro_metadata["luma_standard"] = "BT.709"
        repro_metadata["sobel_edge_threshold"] = 30.0

        exec_time = (time.time() - start_time) * 1000.0

        return PixelAnalysisResult(
            analyzer_id="pillar2_basic_pixel_analyzer",
            analyzer_version="0.1.0",
            execution_time_ms=round(exec_time, 2),
            properties=props,
            channel_statistics=channel_stats,
            luma_statistics=luma_stats,
            intensity_statistics=intensity_stats,
            histograms=histograms,
            spatial_descriptors=spatial_desc,
            evidence_items=evidence_items,
            errors=errors,
            warnings=warnings,
            reproducibility_metadata=repro_metadata,
        )
