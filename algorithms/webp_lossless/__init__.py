"""
algorithms/webp_lossless/__init__.py
Package thuật toán WebP Lossless Image Compression.
Tất cả module dùng chung (image_utils, spatial_filters, lz77, huffman)
nằm tại algorithms/common/.

Chứa:
  - compressor:    Pipeline nén WebP lossless hoàn chỉnh + RIFF/VP8L container.
  - decompressor:  Pipeline giải nén WebP lossless hoàn chỉnh.
"""

from ..common import (
    # Image utilities
    load_image,
    raw_bytes_to_image,
    save_image,
    calculate_image_metrics,
    # Spatial filters
    FilterType,
    paeth_predictor,
    filter_scanlines,
    unfilter_scanlines,
    # LZ77
    lz77_compress,
    lz77_decompress,
    lz77_encode,
    lz77_decode,
    # Huffman
    HuffmanNode,
    build_huffman_tree,
    build_codes,
    huffman_encode,
    huffman_decode,
)
from .compressor import (
    # WebP Lossless Compressor
    WebPLosslessCompressor,
    compress_image_webp_lossless,
    benchmark_webp_lossless,
    # Subtract Green Transform
    apply_subtract_green,
    inverse_subtract_green,
    # RIFF/VP8L container utilities
    pack_riff_webp,
    parse_riff_webp,
    pack_vp8l_header,
    parse_vp8l_header,
)
from .decompressor import (
    # WebP Lossless Decompressor
    WebPLosslessDecompressor,
    decompress_image_webp_lossless,
)

__all__ = [
    # Image utilities (re-exported from common)
    "load_image",
    "raw_bytes_to_image",
    "save_image",
    "calculate_image_metrics",
    # Spatial filters (re-exported from common)
    "FilterType",
    "paeth_predictor",
    "filter_scanlines",
    "unfilter_scanlines",
    # LZ77 (re-exported from common)
    "lz77_compress",
    "lz77_decompress",
    "lz77_encode",
    "lz77_decode",
    # Huffman (re-exported from common)
    "HuffmanNode",
    "build_huffman_tree",
    "build_codes",
    "huffman_encode",
    "huffman_decode",
    # WebP Lossless Compressor
    "WebPLosslessCompressor",
    "compress_image_webp_lossless",
    "benchmark_webp_lossless",
    # WebP Lossless Decompressor
    "WebPLosslessDecompressor",
    "decompress_image_webp_lossless",
    # Subtract Green Transform
    "apply_subtract_green",
    "inverse_subtract_green",
    # RIFF/VP8L container utilities
    "pack_riff_webp",
    "parse_riff_webp",
    "pack_vp8l_header",
    "parse_vp8l_header",
]
