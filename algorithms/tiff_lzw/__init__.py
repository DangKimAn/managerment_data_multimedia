"""TIFF image compression using the lossless LZW codec."""

from .compressor import (
    TiffLzwCompressor,
    compress_image_tiff_lzw,
    decompress_image_tiff_lzw,
)

__all__ = [
    "TiffLzwCompressor",
    "compress_image_tiff_lzw",
    "decompress_image_tiff_lzw",
]
