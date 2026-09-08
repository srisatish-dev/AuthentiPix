"""AuthentiPix Phase 2 & Phase 3A: Pixel Forensics Package."""

from authentipix.pixel.residual import (
    ResidualNoiseAnalyzer,
    ResidualAnalysisError,
    UnsupportedDtypeError,
    FloatRangeViolationError,
    SmallImageError,
    SobelIntegrationError,
)
from authentipix.pixel.schemas import (
    InputCharacteristics,
    ResidualStatisticalMoments,
    SpatialResidualDescriptors,
    ChannelResidualDescriptors,
    ResidualNoiseAnalysisResult,
)

__version__ = "0.3.5"

