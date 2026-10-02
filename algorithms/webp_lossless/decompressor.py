"""
WebP Lossless Image Decompressor Module  (algorithms/webp_lossless/decompressor.py)
Coordinates the whole WebP Lossless decompression pipeline:
1. Parses RIFF WebP container and extracts VP8L chunk
2. Decodes VP8L header and transform metadata
3. Reverses Huffman entropy coding   → LZ77 token stream
4. Reverses LZ77 dictionary coding   → filtered pixel scanlines
5. Reverses spatial prediction filter → transformed pixel bytes
6. Reverses Subtract Green Transform  → original pixel bytes
7. Reconstructs the original image (100% lossless, bit-for-bit identical)
Written in simple A2/B1 English.
"""

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
from ..common.huffman import huffman_decode

# Import container/header parsers and inverse transforms from compressor
from .compressor import (
    parse_riff_webp,
    parse_vp8l_header,
    parse_transform_metadata,
    inverse_subtract_green,
    TRANSFORM_META_SIZE,
)


# ==============================================================================
# WEBP LOSSLESS DECOMPRESSOR CLASS
# ==============================================================================

class WebPLosslessDecompressor:
    """
    Decompressor for WebP Lossless format (.webp with VP8L chunk).
    Pipeline: RIFF → VP8L Header → Huffman Decode → LZ77 Decode
              → Unfilter Scanlines → Inverse Subtract Green → Reconstruct Image
    """

    def decompress(self, compressed_data: bytes) -> Image.Image:
        """
        Decompress WebP Lossless binary data back into a 100% original PIL Image.

        Full pipeline (reverses the compressor):
        1. Parse RIFF WebP container → extract VP8L chunk
        2. Parse VP8L header (signature, width, height, alpha, version)
        3. Read custom transform metadata (subtract green flag, filter type, etc.)
        4. Huffman decode → recover LZ77 binary stream
        5. LZ77 decompress → recover filtered pixel scanlines
        6. Unfilter scanlines → recover transformed pixel bytes
        7. Inverse Subtract Green → recover original RGBA pixel bytes
        8. Reconstruct PIL Image, convert back to original color mode

        Args:
            compressed_data: Compressed bytes in RIFF/WebP/VP8L format (.webp file).
        Returns:
            Image.Image: Reconstructed 2D PIL Image (lossless, bit-for-bit identical).
        """
        # Step 1: Parse RIFF WebP container → extract VP8L chunk data
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

        # Step 4: Extract compressed payload (everything after header + metadata)
        payload_offset = meta_offset + TRANSFORM_META_SIZE
        compressed_payload = vp8l_data[payload_offset:]

        # Step 5: Huffman decode → get LZ77 binary stream
        lz77_bytes = huffman_decode(compressed_payload)

        # Step 6: LZ77 decompress → get filtered pixel scanlines
        filtered_bytes = lz77_decode(lz77_bytes)

        # Step 7: Reverse spatial prediction filter (unfilter scanlines)
        pixel_data = unfilter_scanlines(filtered_bytes, width, height, channels)

        # Step 8: Reverse Subtract Green Transform (restore original R, B channels)
        if has_subtract_green:
            pixel_data = inverse_subtract_green(pixel_data, width, height)

        # Step 9: Reconstruct PIL Image from raw RGBA bytes
        reconstructed_img = raw_bytes_to_image(pixel_data, width, height, "RGBA")

        # Step 10: Convert back to original color mode if needed
        #          (e.g., if original was "RGB" or "L", remove alpha channel)
        if original_mode != "RGBA":
            reconstructed_img = reconstructed_img.convert(original_mode)

        return reconstructed_img

    def decompress_file(
        self,
        input_webp_path: str,
        output_image_path: Optional[str] = None,
    ) -> Tuple[Image.Image, float]:
        """
        Decompress a .webp file from disk and optionally save as image.

        Args:
            input_webp_path:  Path to compressed .webp file.
            output_image_path: Optional path to save the decompressed image.
        Returns:
            tuple: (PIL Image, decompression_time_seconds)
        """
        with open(input_webp_path, "rb") as f:
            data = f.read()

        start_time = time.perf_counter()
        img = self.decompress(data)
        decomp_time = time.perf_counter() - start_time

        if output_image_path:
            save_image(img, output_image_path)

        return img, decomp_time


# ==============================================================================
# CONVENIENCE HELPER FUNCTION
# ==============================================================================

def decompress_image_webp_lossless(
    input_webp_path: str, output_image_path: str
):
    """Convenience helper function to decompress a .webp file to an image file."""
    d = WebPLosslessDecompressor()
    return d.decompress_file(input_webp_path, output_image_path)
