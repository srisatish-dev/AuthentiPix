"""Safe image loading and representation normalization engine for Phase 2.

Consumes Phase 1 AnalysisContext and SecurityEnforcer. Implements the strict
Pillow Image Mode Policy (L, RGB, RGBA, I, I;16, F, P, CMYK, 1, unsupported).
"""

import io
from typing import Tuple
import numpy as np
from PIL import Image

from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.security import SecurityEnforcer, SecurityViolationError
from authentipix.pixel.schemas import PixelProperties


class UnsupportedImageModeError(Exception):
    """Raised when an image contains an unsupported or unconvertible Pillow mode."""
    pass


class PixelImageLoader:
    """Safely decodes binary asset data into normalized NumPy array representations."""

    def __init__(self, enforcer: SecurityEnforcer = None):
        self.enforcer = enforcer or SecurityEnforcer()

    def load_pixel_array(self, ctx: AnalysisContext) -> Tuple[np.ndarray, PixelProperties]:
        """Loads asset bytes, enforces security policies, applies Pillow mode policy, and returns array + properties.

        Returns:
            Tuple[np.ndarray, PixelProperties]: Normalized 3D NumPy array (H, W, C) and physical properties metadata.
        """
        # Validate file size and mime limits using Phase 1 enforcer
        self.enforcer.validate_context(ctx)

        image_bytes = ctx.get_bytes()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise UnsupportedImageModeError(f"Failed to decode image bytes: {e}")

        # Validate raster dimensions against decompression bomb policy (100 MP limit)
        width, height = pil_image.size
        self.enforcer.validate_raster_dimensions(width, height)

        native_mode = pil_image.mode
        was_palette = False
        was_cmyk = False
        was_bilevel = False

        # Apply Pillow Image Mode Policy
        if native_mode == "P" or native_mode == "PA":
            was_palette = True
            if "A" in native_mode or ("transparency" in pil_image.info):
                pil_image = pil_image.convert("RGBA")
            else:
                pil_image = pil_image.convert("RGB")
        elif native_mode == "CMYK":
            was_cmyk = True
            pil_image = pil_image.convert("RGB")
        elif native_mode == "1":
            was_bilevel = True
            pil_image = pil_image.convert("L")
        elif native_mode in ["L", "RGB", "RGBA", "I", "F"] or native_mode.startswith("I;"):
            # Native supported mode - keep as is
            pass
        elif native_mode in ["LA"]:
            # Grayscale with alpha -> convert to RGBA for consistent 4-channel alpha handling
            pil_image = pil_image.convert("RGBA")
        else:
            raise UnsupportedImageModeError(f"Unsupported Pillow image mode: '{native_mode}'")

        # Convert PIL Image to NumPy array
        arr = np.array(pil_image)

        # Audit invalid NaN and Inf entries
        nan_count = 0
        inf_count = 0
        is_float = np.issubdtype(arr.dtype, np.floating)
        if is_float:
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())

        # Shape Normalization: Ensure arr is 3D (H, W, C)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=2)

        H, W, C = arr.shape
        pixel_count = W * H
        aspect_ratio = float(W) / float(H)

        # Determine bit depth and alpha presence
        has_alpha = (C == 4) or (native_mode in ["RGBA", "LA", "PA"])
        
        if arr.dtype == np.uint8:
            bit_depth = 8
        elif arr.dtype == np.uint16 or arr.dtype == np.int16 or native_mode.startswith("I;16"):
            bit_depth = 16
        elif arr.dtype == np.int32 or arr.dtype == np.float32:
            bit_depth = 32
        elif arr.dtype == np.float64:
            bit_depth = 64
        else:
            bit_depth = 8

        props = PixelProperties(
            width=W,
            height=H,
            pixel_count=pixel_count,
            aspect_ratio=aspect_ratio,
            native_mode=native_mode,
            channel_count=C,
            bit_depth=bit_depth,
            dtype=str(arr.dtype),
            has_alpha=has_alpha,
            is_floating_point=is_float,
            was_palette_converted=was_palette,
            was_cmyk_converted=was_cmyk,
            was_bilevel_converted=was_bilevel,
            nan_pixel_count=nan_count,
            inf_pixel_count=inf_count,
        )

        return arr, props
