"""AnalysisContext implementation for flexible, decoupled analyzer inputs.

Maintains common asset metadata, file handles, SHA-256 hashes, and configuration
without forcing every analyzer to load full byte buffers into memory.
"""

import hashlib
import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, PrivateAttr


class AnalysisContext(BaseModel):
    """Common analysis context supplied to all forensic analyzers."""

    image_id: str = Field(description="Unique asset identifier")
    sha256_hash: str = Field(description="Hex-encoded SHA-256 hash of input asset bytes")
    mime_type: str = Field(default="image/jpeg", description="MIME type of the asset")
    file_size_bytes: int = Field(ge=0, description="Size of asset in bytes")

    file_path: Optional[str] = Field(default=None, description="Absolute or relative file path if stored on disk")

    # Optional in-memory byte buffer (for stream uploads or testing)
    _in_memory_bytes: Optional[bytes] = PrivateAttr(default=None)

    config: Dict[str, Any] = Field(default_factory=dict, description="Analyzer runtime configuration")
    reproducibility_context: Dict[str, Any] = Field(default_factory=dict, description="System/runtime metadata")

    def __init__(self, **data: Any):
        file_bytes = data.pop("file_bytes", None)
        super().__init__(**data)
        if file_bytes is not None:
            self._in_memory_bytes = file_bytes
            if not self.sha256_hash:
                self.sha256_hash = hashlib.sha256(file_bytes).hexdigest()
            if self.file_size_bytes == 0:
                self.file_size_bytes = len(file_bytes)

    @classmethod
    def from_bytes(
        cls,
        image_bytes: bytes,
        image_id: str = "asset_001",
        mime_type: str = "image/jpeg",
        config: Optional[Dict[str, Any]] = None,
    ) -> "AnalysisContext":
        """Factory creating AnalysisContext from in-memory bytes."""
        sha256 = hashlib.sha256(image_bytes).hexdigest()
        ctx = cls(
            image_id=image_id,
            sha256_hash=sha256,
            mime_type=mime_type,
            file_size_bytes=len(image_bytes),
            config=config or {},
        )
        ctx._in_memory_bytes = image_bytes
        return ctx

    @classmethod
    def from_file(
        cls,
        file_path: str,
        image_id: Optional[str] = None,
        mime_type: str = "image/jpeg",
        config: Optional[Dict[str, Any]] = None,
    ) -> "AnalysisContext":
        """Factory creating AnalysisContext from a filesystem path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        file_size = os.path.getsize(file_path)
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        sha256 = hasher.hexdigest()

        asset_id = image_id or os.path.basename(file_path)
        return cls(
            image_id=asset_id,
            sha256_hash=sha256,
            mime_type=mime_type,
            file_size_bytes=file_size,
            file_path=os.path.abspath(file_path),
            config=config or {},
        )

    def get_bytes(self) -> bytes:
        """Retrieve binary content of asset safely (from memory or reading file path)."""
        if self._in_memory_bytes is not None:
            return self._in_memory_bytes
        if self.file_path and os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                return f.read()
        raise ValueError("No file_bytes or valid file_path available in AnalysisContext")
