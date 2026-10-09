"""
algorithms/common/__init__.py
Các module dùng chung — có thể tái sử dụng bởi nhiều codec khác nhau:
  - image_utils:     Load, flatten, reconstruct và so sánh ảnh.
  - spatial_filters: Bộ lọc dự đoán không gian PNG (Sub, Up, Average, Paeth, Adaptive).
  - lz77:            LZ77 sliding window dictionary compression & decompression.
  - huffman:         Huffman tree, bitstream encoding & decoding.
"""

from .image_utils import (
    load_image,
    raw_bytes_to_image,
    save_image,
    calculate_image_metrics
)
from .spatial_filters import (
    FilterType,
    paeth_predictor,
    filter_scanlines,
    unfilter_scanlines
)
from .lz77 import (
    lz77_compress,
    lz77_decompress,
    lz77_encode,
    lz77_decode
)
from .huffman import (
    HuffmanNode,
    build_huffman_tree,
    build_codes,
    huffman_encode,
    huffman_decode
)
from .huffman_webp import (
    huffman_webp_encode,
    huffman_webp_decode,
    build_canonical_codes,
    compute_code_lengths,
    LsbBitWriter,
    LsbBitReader,
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
    # WebP Canonical Huffman
    "huffman_webp_encode",
    "huffman_webp_decode",
    "build_canonical_codes",
    "compute_code_lengths",
    "LsbBitWriter",
    "LsbBitReader",
]
