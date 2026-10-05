"""
WebP Lossless Image Decompressor Module (algorithms/webp_lossless/decompressor.py)

Coordinates the WebP Lossless decompression pipeline supporting two engines:
1. Standard LibWebP Engine:
   - Decodes standard .webp files via Google libwebp (Pillow).
   - Compatible with any standard WebP Lossless image.
2. Pure Python Educational Engine:
   - Full reverse pipeline:
     * Step 1: Parse RIFF WebP container and VP8L chunk
     * Step 2: Read VP8L header and transform metadata
     * Step 3: WebP Canonical Huffman decode (huffman_webp)
     * Step 4: LZ77 dictionary decode (lz77)
     * Step 5: Reverse spatial prediction filters (spatial_filters)
     * Step 6: Reverse Subtract Green Transform (color decorrelation)
     * Step 7: Reconstruct original PIL Image (100% bit-for-bit lossless)

Written in simple A2/B1 English.
"""

import io
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Import common utilities
from ..common.image_utils import (
    raw_bytes_to_image,
    save_image,
    calculate_image_metrics,
)
from ..common.spatial_filters import unfilter_scanlines
from ..common.lz77 import lz77_decode
from ..common.huffman_webp import huffman_webp_decode

# Import container/header parsers and inverse transforms from compressor
from .compressor import (
    parse_riff_webp,
    parse_vp8l_header,
    parse_transform_metadata,
    inverse_subtract_green,
    TRANSFORM_META_SIZE,
    PY_ENGINE_MARKER,
)


# ==============================================================================
# WEBP LOSSLESS DECOMPRESSOR CLASS
# ==============================================================================

class WebPLosslessDecompressor:
    """
    Decompressor for WebP Lossless format (.webp).
    Supports auto-detecting between standard LibWebP and Pure Python streams.
    """

    def decompress(
        self, compressed_data: bytes, engine: str = "auto"
    ) -> Image.Image:
        """
        Decompress WebP Lossless binary data back into an exact PIL Image.

        Args:
            compressed_data: Compressed bytes (.webp file content).
            engine: "auto" (default, detects format), "libwebp", or "pure_python".
        Returns:
            Image.Image: Reconstructed PIL Image (100% bit-identical).
        """
        if engine == "pure_python":
            return self._decompress_pure_python(compressed_data)
        elif engine == "libwebp":
            return self._decompress_libwebp(compressed_data)
        elif engine == "auto":
            # Auto-detection: check if VP8L chunk starts with pure Python marker
            if self._is_pure_python_stream(compressed_data):
                return self._decompress_pure_python(compressed_data)
            else:
                try:
                    return self._decompress_libwebp(compressed_data)
                except Exception:
                    # Fallback to pure Python in case of unexpected header mismatch
                    return self._decompress_pure_python(compressed_data)
        else:
            raise ValueError(f"Unknown engine: {engine!r}. Use 'auto', 'libwebp', or 'pure_python'.")

    def _is_pure_python_stream(self, data: bytes) -> bool:
        """Check whether data has our pure Python transform marker."""
        try:
            vp8l_data = parse_riff_webp(data)
            # VP8L header is 5 bytes, transform metadata starts with PY_ENGINE_MARKER (4 bytes)
            if len(vp8l_data) >= 5 + 4:
                return vp8l_data[5:9] == PY_ENGINE_MARKER
        except Exception:
            pass
        return False

    def _decompress_libwebp(self, compressed_data: bytes) -> Image.Image:
        """Decompress standard WebP stream using Google libwebp via Pillow."""
        with Image.open(io.BytesIO(compressed_data)) as img:
            img.load()
            return img.copy()

    def _decompress_pure_python(self, compressed_data: bytes) -> Image.Image:
        """
        Decompress using pure Python modular pipeline:
        RIFF -> VP8L -> Huffman WebP Decode -> LZ77 Decode -> Unfilter -> Inverse SG
        """
        # Step 1: Parse RIFF WebP container -> extract VP8L chunk data
        vp8l_data = parse_riff_webp(compressed_data)

        # Step 2: Parse VP8L header (signature + width + height + alpha + version)
        width, height, has_alpha, version, header_size = parse_vp8l_header(vp8l_data)

        # Step 3: Parse custom transform metadata
        meta_offset = header_size
        has_subtract_green, filter_type_val, original_mode, channels = (
            parse_transform_metadata(
                vp8l_data[meta_offset : meta_offset + TRANSFORM_META_SIZE]
            )
        )

        # Step 4: Extract compressed payload
        payload_offset = meta_offset + TRANSFORM_META_SIZE
        compressed_payload = vp8l_data[payload_offset:]

        # Step 5: WebP Canonical Huffman decode -> get LZ77 token stream
        lz77_bytes = huffman_webp_decode(compressed_payload)

        # Step 6: LZ77 decompress -> get filtered pixel scanlines
        filtered_bytes = lz77_decode(lz77_bytes)

        # Step 7: Reverse spatial prediction filter (unfilter scanlines)
        pixel_data = unfilter_scanlines(filtered_bytes, width, height, channels)

        # Step 8: Reverse Subtract Green Transform (restore original R, B channels)
        if has_subtract_green:
            pixel_data = inverse_subtract_green(pixel_data, width, height)

        # Step 9: Reconstruct PIL Image from raw RGBA bytes
        reconstructed_img = raw_bytes_to_image(pixel_data, width, height, "RGBA")

        # Step 10: Convert back to original color mode if needed
        if original_mode != "RGBA":
            reconstructed_img = reconstructed_img.convert(original_mode)

        return reconstructed_img

    def decompress_file(
        self,
        input_webp_path: str,
        output_image_path: Optional[str] = None,
        engine: str = "auto",
    ) -> Tuple[Image.Image, float]:
        """
        Decompress a .webp file from disk and optionally save as image.

        Args:
            input_webp_path:  Path to compressed .webp file.
            output_image_path: Optional path to save the decompressed image.
            engine: "auto", "libwebp", or "pure_python".
        Returns:
            tuple: (PIL Image, decompression_time_seconds)
        """
        with open(input_webp_path, "rb") as f:
            data = f.read()

        start_time = time.perf_counter()
        img = self.decompress(data, engine=engine)
        decomp_time = time.perf_counter() - start_time

        if output_image_path:
            save_image(img, output_image_path)

        return img, decomp_time


# ==============================================================================
# CONVENIENCE HELPER FUNCTION
# ==============================================================================

def decompress_image_webp_lossless(
    input_webp_path: str,
    output_image_path: str,
    engine: str = "auto",
):
    """Convenience helper function to decompress a .webp file to an image file."""
    d = WebPLosslessDecompressor()
    return d.decompress_file(input_webp_path, output_image_path, engine=engine)
