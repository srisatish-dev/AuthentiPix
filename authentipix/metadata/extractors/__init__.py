"""Metadata and provenance extraction sub-modules for AuthentiPix Phase 1."""

from authentipix.metadata.extractors.base import BaseExtractor, ExtractionResult
from authentipix.metadata.extractors.c2pa import C2paExtractor
from authentipix.metadata.extractors.exif import ExifExtractor
from authentipix.metadata.extractors.file_extractor import FileContainerExtractor
from authentipix.metadata.extractors.iptc import IptcExtractor
from authentipix.metadata.extractors.preview import PreviewExtractor
from authentipix.metadata.extractors.xmp import XmpExtractor

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "ExifExtractor",
    "XmpExtractor",
    "IptcExtractor",
    "FileContainerExtractor",
    "PreviewExtractor",
    "C2paExtractor",
]
