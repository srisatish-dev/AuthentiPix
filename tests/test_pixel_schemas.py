"""Unit tests for Phase 2 Pydantic schemas (authentipix.pixel.schemas)."""

from authentipix.pixel.schemas import (
    PixelProperties,
    ChannelStatistics,
    HistogramFeatures,
    SpatialDescriptors,
    PixelAnalysisResult,
)


def test_pixel_properties_schema():
    props = PixelProperties(
        width=1920,
        height=1080,
        pixel_count=2073600,
        aspect_ratio=1.7777777777777777,
        native_mode="RGB",
        channel_count=3,
        bit_depth=8,
        dtype="uint8",
    )
    assert props.width == 1920
    assert props.height == 1080
    assert props.has_alpha is False
    assert props.was_palette_converted is False
    assert props.nan_pixel_count == 0


def test_channel_statistics_schema():
    stats = ChannelStatistics(
        channel_index=0,
        channel_name="red",
        min=10.0,
        max=240.0,
        mean=125.5,
        median=120.0,
        variance=2500.0,
        std_dev=50.0,
        p5=20.0,
        p25=80.0,
        p75=170.0,
        p95=230.0,
        dynamic_range=230.0,
    )
    assert stats.channel_name == "red"
    assert stats.dynamic_range == 230.0


def test_histogram_features_schema():
    hist = HistogramFeatures(
        channel_name="luma",
        bins=[10] * 256,
        entropy=7.5,
        occupied_bin_count=256,
        clipping_fraction_0=0.0,
        clipping_fraction_255=0.0,
        comb_metric=0.1,
    )
    assert len(hist.bins) == 256
    assert hist.entropy == 7.5
    assert hist.comb_metric == 0.1


def test_spatial_descriptors_schema():
    spatial = SpatialDescriptors(
        gradient_mean=15.2,
        gradient_std=12.1,
        edge_density=0.12,
        edge_threshold=30.0,
    )
    assert spatial.gradient_mean == 15.2
    assert spatial.edge_threshold == 30.0


def test_pixel_analysis_result_schema():
    props = PixelProperties(
        width=100,
        height=100,
        pixel_count=10000,
        aspect_ratio=1.0,
        native_mode="L",
        channel_count=1,
        bit_depth=8,
        dtype="uint8",
    )
    result = PixelAnalysisResult(properties=props)
    assert result.analyzer_id == "pillar2_basic_pixel_analyzer"
    assert result.properties.native_mode == "L"
    assert len(result.channel_statistics) == 0
