"""Base extractor class and data structure for metadata extractors."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from authentipix.metadata.context import AnalysisContext


class ExtractionResult(BaseModel):
    """Container for data extracted by a single metadata extractor."""
    extractor_name: str
    success: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)
    raw_tags: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0


class BaseExtractor(ABC):
    """Abstract base class for all metadata and provenance extractors."""

    @property
    @abstractmethod
    def extractor_name(self) -> str:
        """Name identifying this extractor."""
        pass

    @abstractmethod
    def _extract_internal(self, ctx: AnalysisContext) -> ExtractionResult:
        """Internal implementation of extraction logic."""
        pass

    def extract(self, ctx: AnalysisContext) -> ExtractionResult:
        """Safely executes extraction with timing, error catching, and validation."""
        start_time = time.perf_counter()
        try:
            res = self._extract_internal(ctx)
        except Exception as e:
            res = ExtractionResult(
                extractor_name=self.extractor_name,
                success=False,
                errors=[f"Unhandled exception in {self.extractor_name}: {e}"],
            )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        res.execution_time_ms = round(elapsed_ms, 3)
        return res
