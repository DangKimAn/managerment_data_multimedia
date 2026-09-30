"""
algorithms/deflate/__init__.py
Package thuật toán DEFLATE Image Compression.
Tất cả module dùng chung (image_utils, spatial_filters, lz77, huffman)
nằm tại algorithms/common/.

Chỉ chứa:
  - compressor: Pipeline nén/giải nén DEFLATE hoàn chỉnh.
  - demo:       Chương trình demo và benchmark.
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
    DeflateImageCompressor,
    compress_image_deflate,
    decompress_image_deflate,
    benchmark_image_compression
)

__all__ = [
    # Image utilities
    "load_image",
    "raw_bytes_to_image",
    "save_image",
    "calculate_image_metrics",
    # Spatial filters
    "FilterType",
    "paeth_predictor",
    "filter_scanlines",
    "unfilter_scanlines",
    # LZ77
    "lz77_compress",
    "lz77_decompress",
    "lz77_encode",
    "lz77_decode",
    # Huffman
    "HuffmanNode",
    "build_huffman_tree",
    "build_codes",
    "huffman_encode",
    "huffman_decode",
    # Deflate Compressor
    "DeflateImageCompressor",
    "compress_image_deflate",
    "decompress_image_deflate",
    "benchmark_image_compression",
]
