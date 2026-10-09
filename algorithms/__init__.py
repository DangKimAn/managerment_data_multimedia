"""
algorithms/__init__.py
Multimedia Image Compression Algorithms Package.

Cấu trúc thư mục:
  algorithms/
  ├── common/               ← Module dùng chung cho mọi codec
  │   ├── image_utils.py    ← Load, flatten, reconstruct, compare ảnh
  │   ├── spatial_filters.py← Bộ lọc dự đoán không gian PNG
  │   ├── lz77.py           ← LZ77 sliding window compression
  │   └── huffman.py        ← Huffman entropy coding
  ├── deflate/              ← Thuật toán DEFLATE Image Compression
  │   ├── compressor.py     ← Pipeline nén/giải nén hoàn chỉnh
  │   ├── demo.py
  │   └── README.md
  └── webp_lossless/        ← Thuật toán WebP Lossless Image Compression
      ├── compressor.py     ← Pipeline nén WebP lossless (LZ77 + Huffman + RIFF/VP8L)
      ├── decompressor.py   ← Pipeline giải nén WebP lossless
      └── __init__.py
"""

from .deflate import (
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
    # LZ77 (từ common, re-export qua deflate)
    lz77_compress,
    lz77_decompress,
    lz77_encode,
    lz77_decode,
    # Huffman (từ common, re-export qua deflate)
    HuffmanNode,
    build_huffman_tree,
    build_codes,
    huffman_encode,
    huffman_decode,
    # Deflate Compressor
    DeflateImageCompressor,
    compress_image_deflate,
    decompress_image_deflate,
    benchmark_image_compression,
)
from .tiff_lzw import (
    TiffLzwCompressor,
    compress_image_tiff_lzw,
    decompress_image_tiff_lzw,
)

from .webp_lossless import (
    # WebP Lossless Compressor
    WebPLosslessCompressor,
    compress_image_webp_lossless,
    benchmark_webp_lossless,
    # WebP Lossless Decompressor
    WebPLosslessDecompressor,
    decompress_image_webp_lossless,
    apply_subtract_green,
    inverse_subtract_green,
    # WebP Canonical Huffman
    huffman_webp_encode,
    huffman_webp_decode,
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
    # WebP Lossless Compressor
    "WebPLosslessCompressor",
    "compress_image_webp_lossless",
    "benchmark_webp_lossless",
    # WebP Lossless Decompressor
    "WebPLosslessDecompressor",
    "decompress_image_webp_lossless",
    # WebP Lossless Transforms
    "apply_subtract_green",
    "inverse_subtract_green",
    # WebP Canonical Huffman
    "huffman_webp_encode",
    "huffman_webp_decode",
    # TIFF-LZW Compressor
    "TiffLzwCompressor",
    "compress_image_tiff_lzw",
    "decompress_image_tiff_lzw",
]
