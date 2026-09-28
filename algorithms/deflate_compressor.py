"""
Deflate Image Compressor Module.
Coordinates the whole compression pipeline:
1. Loads and flattens image pixels (using image_utils)
2. Applies 2D spatial prediction filtering (using spatial_filters)
3. Compresses with DEFLATE (combining LZ77 and Huffman coding)
4. Saves to and loads from binary container format (.defimg)
Written in simple A2/B1 English.
"""

import os
import struct
import time
import zlib
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Import dedicated modular components
from .image_utils import (
    load_image,
    raw_bytes_to_image,
    save_image,
    calculate_image_metrics
)
from .spatial_filters import (
    FilterType,
    filter_scanlines,
    unfilter_scanlines
)
from .lz77 import lz77_encode, lz77_decode
from .huffman import huffman_encode, huffman_decode

# 20-byte fixed header format for .defimg file container
# Format: Magic (4s) + Version (B) + FilterMode (B) + ColorMode (4s) + Width (I) + Height (I) + Channels (B) + Engine (B)
HEADER_FORMAT = ">4sBB4sIIBB"
MAGIC_BYTES = b"DFLT"
CONTAINER_VERSION = 1


class DeflateImageCompressor:
    """
    Main image compressor using DEFLATE (LZ77 + Huffman Coding)
    and PNG-style spatial prediction filters.
    """

    def __init__(
        self,
        compression_level: int = 9,
        filter_type: FilterType = FilterType.ADAPTIVE,
        use_pure_python: bool = False
    ):
        """
        Initialize the compressor.
        
        Args:
            compression_level: Deflate level 1 (fastest) to 9 (best compression). Default is 9.
            filter_type: Spatial filter type (NONE, SUB, UP, AVERAGE, PAETH, ADAPTIVE).
            use_pure_python: If True, uses our standalone pure Python LZ77 + Huffman pipeline.
                             If False (default), uses fast C-accelerated DEFLATE (zlib RFC 1951).
        """
        if not (1 <= compression_level <= 9):
            raise ValueError("compression_level must be between 1 and 9.")
        self.compression_level = compression_level
        self.filter_type = filter_type
        self.use_pure_python = use_pure_python

    # --------------------------------------------------------------------------
    # CORE ENTROPY & DICTIONARY COMPRESSION (LZ77 + HUFFMAN)
    # --------------------------------------------------------------------------
    def _deflate_compress(self, data: bytes) -> bytes:
        """Compress byte stream using DEFLATE (LZ77 + Huffman Coding)"""
        if self.use_pure_python:
            # Step 1: Run LZ77 sliding window dictionary compression
            lz77_bytes = lz77_encode(data)
            # Step 2: Run Huffman entropy coding on top of LZ77 output
            return huffman_encode(lz77_bytes)
        else:
            # Standard RFC 1951 DEFLATE (C-optimized LZ77 + Huffman)
            return zlib.compress(data, level=self.compression_level)

    def _deflate_decompress(self, compressed_payload: bytes, is_pure_python: bool) -> bytes:
        """Decompress byte stream using DEFLATE (Huffman Decode + LZ77 Decode)"""
        if is_pure_python:
            # Step 1: Decode Huffman bitstream
            lz77_bytes = huffman_decode(compressed_payload)
            # Step 2: Decode LZ77 tokens back into raw bytes
            return lz77_decode(lz77_bytes)
        else:
            # Standard RFC 1951 DEFLATE
            return zlib.decompress(compressed_payload)

    # --------------------------------------------------------------------------
    # IMAGE COMPRESSION
    # --------------------------------------------------------------------------
    def compress(self, image_input: Union[str, Image.Image, np.ndarray]) -> bytes:
        """
        Compress an image:
        1. Read image and flatten into 1D byte array.
        2. Apply spatial prediction filter to remove 2D pixel correlation.
        3. Compress with DEFLATE (LZ77 + Huffman).
        4. Wrap into a 20-byte binary header.
        
        Args:
            image_input: File path, PIL Image, or numpy array.
        Returns:
            bytes: Compressed binary data in .defimg format.
        """
        # Step 1: Load image and flatten to 1D bytes using image_utils
        raw_bytes, width, height, channels, mode = load_image(image_input)
        bpp = channels

        # Step 2: Apply spatial prediction filter using spatial_filters
        filtered_bytes = filter_scanlines(raw_bytes, width, height, bpp, self.filter_type)

        # Step 3: Run DEFLATE compression (LZ77 + Huffman)
        compressed_payload = self._deflate_compress(filtered_bytes)

        # Step 4: Pack 20-byte binary header
        mode_bytes = mode.encode("ascii").ljust(4, b"\x00")
        engine_byte = 1 if self.use_pure_python else 0
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_BYTES,
            CONTAINER_VERSION,
            int(self.filter_type),
            mode_bytes,
            width,
            height,
            channels,
            engine_byte
        )

        return header + compressed_payload

    def compress_file(self, input_image_path: str, output_def_path: str) -> Dict[str, float]:
        """Compress an image file and save output as .defimg file on disk"""
        start_time = time.perf_counter()
        compressed_data = self.compress(input_image_path)
        compress_time = time.perf_counter() - start_time

        with open(output_def_path, "wb") as f:
            f.write(compressed_data)

        orig_size = os.path.getsize(input_image_path)
        comp_size = len(compressed_data)
        cr = orig_size / comp_size if comp_size > 0 else 0
        space_saving = (1.0 - comp_size / orig_size) * 100.0 if orig_size > 0 else 0

        return {
            "original_size": orig_size,
            "compressed_size": comp_size,
            "compression_ratio": cr,
            "space_saving_pct": space_saving,
            "compression_time_sec": compress_time
        }

    # --------------------------------------------------------------------------
    # IMAGE DECOMPRESSION
    # --------------------------------------------------------------------------
    def decompress(self, compressed_data: bytes) -> Image.Image:
        """
        Decompress .defimg binary data back into a 100% original PIL Image.
        
        Args:
            compressed_data: Compressed bytes containing header and payload.
        Returns:
            Image.Image: Reconstructed 2D PIL Image.
        """
        header_size = struct.calcsize(HEADER_FORMAT)
        if len(compressed_data) < header_size:
            raise ValueError("Invalid compressed data: Shorter than header size.")

        # Step 1: Read and parse 20-byte header
        magic, version, filter_mode_val, mode_bytes, width, height, channels, engine_byte = struct.unpack(
            HEADER_FORMAT, compressed_data[:header_size]
        )

        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid file signature: {magic}, expected: {MAGIC_BYTES}")

        mode = mode_bytes.decode("ascii").rstrip("\x00")
        is_pure_python = (engine_byte == 1)
        payload = compressed_data[header_size:]

        # Step 2: Decompress DEFLATE stream (Huffman + LZ77 decode)
        filtered_bytes = self._deflate_decompress(payload, is_pure_python=is_pure_python)

        # Step 3: Reverse spatial prediction filters using spatial_filters
        raw_bytes = unfilter_scanlines(filtered_bytes, width, height, channels)

        # Step 4: Reconstruct 2D PIL Image from 1D bytes using image_utils
        reconstructed_img = raw_bytes_to_image(raw_bytes, width, height, mode)
        return reconstructed_img

    def decompress_file(
        self,
        input_def_path: str,
        output_image_path: Optional[str] = None
    ) -> Tuple[Image.Image, float]:
        """Decompress a .defimg file from disk and optionally save as image"""
        with open(input_def_path, "rb") as f:
            data = f.read()

        start_time = time.perf_counter()
        img = self.decompress(data)
        decomp_time = time.perf_counter() - start_time

        if output_image_path:
            save_image(img, output_image_path)

        return img, decomp_time


# ------------------------------------------------------------------------------
# BENCHMARK & PERFORMANCE MEASUREMENT
# ------------------------------------------------------------------------------
def benchmark_image_compression(
    image_input: Union[str, Image.Image],
    compression_level: int = 9
) -> Dict[str, Union[float, int, bool, str]]:
    """
    Run full performance evaluation comparing:
    1. Direct DEFLATE (no spatial filters)
    2. Adaptive Filter + DEFLATE (PNG standard)
    Includes bit-for-bit lossless check, MSE, PSNR, CR, BPP, and speed.
    """
    # Load image using image_utils
    raw_bytes, width, height, channels, mode = load_image(image_input)
    raw_size = len(raw_bytes)
    img_orig = raw_bytes_to_image(raw_bytes, width, height, mode)

    # 1. Mode 1: Direct Deflate (No filter)
    comp_direct = DeflateImageCompressor(compression_level=compression_level, filter_type=FilterType.NONE)
    t0 = time.perf_counter()
    bytes_direct = comp_direct.compress(img_orig)
    t_comp_direct = time.perf_counter() - t0

    # 2. Mode 2: Adaptive Filter + Deflate
    comp_adaptive = DeflateImageCompressor(compression_level=compression_level, filter_type=FilterType.ADAPTIVE)
    t1 = time.perf_counter()
    bytes_adaptive = comp_adaptive.compress(img_orig)
    t_comp_adaptive = time.perf_counter() - t1

    # 3. Decompress and verify quality
    t2 = time.perf_counter()
    img_rec = comp_adaptive.decompress(bytes_adaptive)
    t_decomp_adaptive = time.perf_counter() - t2

    # Verify bit-for-bit match and calculate MSE/PSNR using image_utils
    is_bit_identical, mse, psnr_db = calculate_image_metrics(img_orig, img_rec)

    size_direct = len(bytes_direct)
    size_adaptive = len(bytes_adaptive)
    cr_direct = raw_size / size_direct if size_direct > 0 else 0
    cr_adaptive = raw_size / size_adaptive if size_adaptive > 0 else 0
    bpp_adaptive = (size_adaptive * 8) / (width * height)
    saving_adaptive = (1.0 - size_adaptive / raw_size) * 100.0

    return {
        "dimensions": f"{width}x{height}",
        "raw_pixel_bytes": raw_size,
        "direct_deflate_bytes": size_direct,
        "direct_cr": cr_direct,
        "adaptive_deflate_bytes": size_adaptive,
        "adaptive_cr": cr_adaptive,
        "space_saving_pct": saving_adaptive,
        "bits_per_pixel": bpp_adaptive,
        "compress_time_sec": t_comp_adaptive,
        "decompress_time_sec": t_decomp_adaptive,
        "is_lossless": is_bit_identical,
        "mse": mse,
        "psnr_db": psnr_db
    }


def compress_image_deflate(
    input_path: str,
    output_path: str,
    level: int = 9,
    filter_type: FilterType = FilterType.ADAPTIVE
):
    """Convenience helper function to compress an image file to .defimg"""
    c = DeflateImageCompressor(compression_level=level, filter_type=filter_type)
    return c.compress_file(input_path, output_path)


def decompress_image_deflate(input_def_path: str, output_image_path: str):
    """Convenience helper function to decompress a .defimg file to an image file"""
    c = DeflateImageCompressor()
    return c.decompress_file(input_def_path, output_image_path)
