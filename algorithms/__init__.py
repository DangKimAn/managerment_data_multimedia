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
  └── deflate/              ← Thuật toán DEFLATE Image Compression
      ├── compressor.py     ← Pipeline nén/giải nén hoàn chỉnh
      ├── demo.py
      └── README.md
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
