"""
Multimedia Image Compression Algorithms Package.
Modular structure:
- image_utils: Image loading, flattening, reconstruction, and metric calculation.
- spatial_filters: PNG-style spatial prediction filters (Sub, Up, Average, Paeth, Adaptive).
- lz77: Standalone LZ77 sliding window dictionary compression and decompression.
- huffman: Standalone Huffman tree, bitstream encoding, and decoding.
- deflate_compressor: Combines all modules into the full DEFLATE image compression pipeline.
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
from .deflate_compressor import (
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
    "benchmark_image_compression"
]
