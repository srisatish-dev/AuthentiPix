"""Pydantic data schemas for AuthentiPix Phase 2 (Basic Pixel Analysis).

Defines structured data models for pixel properties, channel statistics,
quantized histograms, spatial descriptors, and top-level pixel analysis results.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from authentipix.metadata.schemas import EvidenceItem


class PixelProperties(BaseModel):
    """Physical dimensions, array geometry, native format, and conversion flags."""
    width: int = Field(ge=1, description="Raster width in pixels")
    height: int = Field(ge=1, description="Raster height in pixels")
    pixel_count: int = Field(ge=1, description="Total number of pixels (width * height)")
    aspect_ratio: float = Field(gt=0, description="Aspect ratio (width / height)")
    native_mode: str = Field(description="Original Pillow image mode (e.g. RGB, RGBA, L, P, CMYK)")
    channel_count: int = Field(ge=1, description="Number of active channels in pixel array")
    bit_depth: int = Field(ge=1, description="Native bit depth per channel (8, 16, or 32)")
    dtype: str = Field(description="NumPy data type string (e.g. uint8, uint16, float32)")
    has_alpha: bool = Field(default=False, description="True if an alpha channel is present")
    is_floating_point: bool = Field(default=False, description="True if input array is floating point")

    # Representation conversion auditing flags
    was_palette_converted: bool = Field(default=False, description="True if converted from palette mode P/PA")
    was_cmyk_converted: bool = Field(default=False, description="True if converted from CMYK mode")
    was_bilevel_converted: bool = Field(default=False, description="True if converted from 1-bit bilevel mode")

    # Invalid value auditing
    nan_pixel_count: int = Field(default=0, ge=0, description="Count of NaN pixel elements encountered")
    inf_pixel_count: int = Field(default=0, ge=0, description="Count of +/-Inf pixel elements encountered")


class ChannelStatistics(BaseModel):
    """Parametric and non-parametric statistical summary for a single channel or luma array."""
    channel_index: int = Field(description="Zero-based channel index (-1 for derived arrays like luma/intensity)")
    channel_name: str = Field(description="Descriptive name (e.g. red, green, blue, alpha, gray, luma, intensity)")
    min: float = Field(description="Minimum pixel intensity value")
    max: float = Field(description="Maximum pixel intensity value")
    mean: float = Field(description="Arithmetic mean pixel intensity")
    median: float = Field(description="50th percentile (median) intensity")
    variance: float = Field(ge=0, description="Variance of pixel intensity")
    std_dev: float = Field(ge=0, description="Standard deviation of pixel intensity")
    p5: float = Field(description="5th percentile intensity")
    p25: float = Field(description="25th percentile intensity")
    p75: float = Field(description="75th percentile intensity")
    p95: float = Field(description="95th percentile intensity")
    dynamic_range: float = Field(ge=0, description="Dynamic range (max - min)")


class HistogramFeatures(BaseModel):
    """256-bin discrete probability distribution features for a channel or luma profile."""
    channel_name: str = Field(description="Channel name corresponding to this histogram")
    bins: List[int] = Field(description="256-element integer bin counts")
    entropy: float = Field(ge=0.0, le=8.0, description="Shannon entropy in bits per pixel")
    occupied_bin_count: int = Field(ge=0, le=256, description="Count of non-zero histogram bins")
    clipping_fraction_0: float = Field(ge=0.0, le=1.0, description="Proportion of pixels clipped at bin 0")
    clipping_fraction_255: float = Field(ge=0.0, le=1.0, description="Proportion of pixels clipped at bin 255")
    comb_metric: float = Field(ge=0.0, le=1.0, description="Corrected interior histogram comb ratio C_comb")


class SpatialDescriptors(BaseModel):
    """Global spatial gradient magnitude and edge density summary."""
    gradient_mean: float = Field(ge=0.0, le=255.0, description="Mean of normalized Sobel gradient magnitudes")
    gradient_std: float = Field(ge=0.0, description="Standard deviation of normalized Sobel gradient magnitudes")
    edge_density: float = Field(ge=0.0, le=1.0, description="Proportion of pixels exceeding edge threshold")
    edge_threshold: float = Field(default=30.0, description="Normalized gradient magnitude threshold T_edge")


class PixelAnalysisResult(BaseModel):
    """Top-level envelope emitted by PixelForensicsEngine for Phase 2."""
    analyzer_id: str = Field(default="pillar2_basic_pixel_analyzer", description="Pillar identifier")
    analyzer_version: str = Field(default="0.1.0", description="Analyzer semantic version")
    execution_time_ms: float = Field(default=0.0, ge=0.0, description="Analysis runtime in milliseconds")
    properties: PixelProperties
    channel_statistics: List[ChannelStatistics] = Field(default_factory=list)
    luma_statistics: Optional[ChannelStatistics] = None
    intensity_statistics: Optional[ChannelStatistics] = None
    histograms: List[HistogramFeatures] = Field(default_factory=list)
    spatial_descriptors: Optional[SpatialDescriptors] = None
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    reproducibility_metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Phase 3A: Residual & Noise-Characteristic Analysis Schemas (v0.3.5-final)
# ============================================================================

class InputCharacteristics(BaseModel):
    """Input dimensions and validity counts.
    
    total_pixel_count is always known once dimensions are inspected.
    Residual-derived counts are Optional and remain None if analysis
    aborts before residual extraction.
    """
    total_pixel_count: int = Field(
        ..., description="Total raw image pixel count (W * H)"
    )
    input_invalid_float_count: Optional[int] = Field(
        default=None,
        description="Total non-finite scalar values (NaN, Inf) across all source channels evaluated after float64 conversion"
    )
    residual_invalid_neighborhood_count: Optional[int] = Field(
        default=None,
        description="Total spatial positions excluded from Luma residual calculation due to non-finite 3x3 source neighborhood"
    )
    unmasked_finite_pixel_count: Optional[int] = Field(
        default=None,
        description="Number of finite evaluated residual samples in set U_Y"
    )
    edge_excluded_finite_pixel_count: Optional[int] = Field(
        default=None,
        description="Number of finite residual samples in set V_Y (g_norm <= 30.0)"
    )
    coverage_ratio: Optional[float] = Field(
        default=None,
        description="Ratio of edge-excluded valid pixels |V_Y| to interior pixels |Omega|"
    )


class ResidualStatisticalMoments(BaseModel):
    """Statistical summary of a residual distribution."""
    mean: float = Field(..., description="Sample mean of residual (ddof=0)")
    std: float = Field(..., description="Sample standard deviation (ddof=0)")
    robust_std: float = Field(..., description="1.4826 * MAD (robust scale estimator)")
    mad: float = Field(..., description="Median Absolute Deviation")
    rms: float = Field(..., description="Root Mean Square residual energy")
    skewness: Optional[float] = Field(
        default=None,
        description="Standardized third central moment (None if N < 8 or std <= 1e-12)"
    )
    kurtosis: Optional[float] = Field(
        default=None,
        description="Fisher excess kurtosis (None if N < 8 or std <= 1e-12)"
    )
    sample_count: int = Field(..., description="Number of valid samples in set")


class SpatialResidualDescriptors(BaseModel):
    """Spatial block variance and autocorrelation metrics."""
    evaluated_block_count: int = Field(
        ..., description="Total non-overlapping 16x16 blocks in interior lattice"
    )
    valid_block_count: int = Field(
        ..., description="Number of 16x16 blocks containing >= 32 valid samples in V_Y"
    )
    local_variance_mean: Optional[float] = Field(
        default=None,
        description="Mean of local block variances across valid blocks (None if valid_block_count < 4)"
    )
    local_variance_std: Optional[float] = Field(
        default=None,
        description="Standard deviation of local block variances (None if valid_block_count < 4)"
    )
    spatial_heterogeneity_index: Optional[float] = Field(
        default=None,
        description="std(sigma^2) / mean(sigma^2) (None if valid_block_count < 4 or mean <= 1e-12)"
    )
    horizontal_lag1_autocorrelation: Optional[float] = Field(
        default=None,
        description="Image-wide lag-1 horizontal autocorrelation in V_Y (None if pairs < 32 or zero variance)"
    )
    vertical_lag1_autocorrelation: Optional[float] = Field(
        default=None,
        description="Image-wide lag-1 vertical autocorrelation in V_Y (None if pairs < 32 or zero variance)"
    )


class ChannelResidualDescriptors(BaseModel):
    """Color-channel specific residual statistics and inter-channel relationships."""
    red_stats: Optional[ResidualStatisticalMoments] = Field(
        default=None, description="Red channel residual moments over V_R"
    )
    green_stats: Optional[ResidualStatisticalMoments] = Field(
        default=None, description="Green channel residual moments over V_G"
    )
    blue_stats: Optional[ResidualStatisticalMoments] = Field(
        default=None, description="Blue channel residual moments over V_B"
    )
    rg_correlation: Optional[float] = Field(
        default=None, description="Pearson correlation between R and G residuals over V_RG"
    )
    rb_correlation: Optional[float] = Field(
        default=None, description="Pearson correlation between R and B residuals over V_RB"
    )
    gb_correlation: Optional[float] = Field(
        default=None, description="Pearson correlation between G and B residuals over V_GB"
    )
    blue_to_green_ratio: Optional[float] = Field(
        default=None, description="Ratio of Blue robust_std to Green robust_std"
    )
    red_to_green_ratio: Optional[float] = Field(
        default=None, description="Ratio of Red robust_std to Green robust_std"
    )


class ResidualNoiseAnalysisResult(BaseModel):
    """Top-level Phase 3A response payload."""
    analyzer_id: str = Field(
        default="pillar2_residual_noise_analyzer",
        description="Unique identifier of the analyzer"
    )
    analyzer_version: str = Field(
        default="0.3.5",
        description="Semantic version of the analyzer implementation"
    )
    input_characteristics: InputCharacteristics = Field(
        ..., description="Raster dimensions and sample validity counts"
    )
    unmasked_luma_stats: Optional[ResidualStatisticalMoments] = Field(
        default=None, description="Luma residual moments across all unmasked valid pixels U_Y"
    )
    edge_excluded_luma_stats: Optional[ResidualStatisticalMoments] = Field(
        default=None, description="Luma residual moments across edge-excluded pixels V_Y"
    )
    spatial_descriptors: Optional[SpatialResidualDescriptors] = Field(
        default=None, description="Spatial heterogeneity and lag-1 autocorrelation"
    )
    channel_descriptors: Optional[ChannelResidualDescriptors] = Field(
        default=None, description="Per-channel and inter-channel residual descriptors (None for grayscale)"
    )
    quality_flags: List[str] = Field(
        default_factory=list, description="List of quality/degradation flags raised"
    )
    limitations: List[str] = Field(
        default_factory=list, description="Forensic limitations and scope constraints"
    )

