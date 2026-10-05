"""
WebP Lossless Image Compressor Module  (algorithms/webp_lossless/compressor.py)
Coordinates the whole WebP Lossless compression pipeline:
1. Loads and flattens image pixels           (using image_utils)
2. Applies Subtract Green Transform          (WebP-specific color decorrelation)
3. Applies 2D spatial prediction filtering   (using spatial_filters)
4. Compresses with LZ77 sliding window       (using lz77)
5. Encodes with Huffman entropy coding        (using huffman)
6. Wraps in RIFF WebP container with VP8L chunk header
Written in simple A2/B1 English.
"""

import os
import struct
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Import dedicated modular components from common
from ..common.image_utils import (
    load_image,
    raw_bytes_to_image,
    save_image,
    calculate_image_metrics
)
from ..common.spatial_filters import (
    FilterType,
    filter_scanlines,
    unfilter_scanlines
)
from ..common.lz77 import lz77_encode, lz77_decode
from ..common.huffman import huffman_encode, huffman_decode


# ==============================================================================
# RIFF / WEBP / VP8L CONTAINER CONSTANTS
# ==============================================================================
RIFF_MAGIC = b"RIFF"         # RIFF container identifier
WEBP_MAGIC = b"WEBP"         # WebP format identifier
VP8L_CHUNK_ID = b"VP8L"      # VP8L lossless chunk identifier
VP8L_SIGNATURE = 0x2F        # VP8L bitstream signature byte
VP8L_VERSION = 0             # VP8L version (must be 0)

# Custom transform metadata format (appended after VP8L header)
# Layout: has_subtract_green (B) + filter_type (B) + original_mode (4s) + channels (B)
TRANSFORM_META_FORMAT = "<BB4sB"
TRANSFORM_META_SIZE = struct.calcsize(TRANSFORM_META_FORMAT)  # = 7 bytes


# ==============================================================================
# SUBTRACT GREEN TRANSFORM (WebP Lossless specific — color decorrelation)
# ==============================================================================

def apply_subtract_green(rgba_bytes: bytes, width: int, height: int) -> bytes:
    """
    Subtract Green Transform — reduces inter-channel color correlation.

    For each RGBA pixel:
        R' = (R - G) mod 256   ← Red minus Green
        B' = (B - G) mod 256   ← Blue minus Green
        G' = G                 ← Green unchanged
        A' = A                 ← Alpha unchanged

    This exploits the fact that natural images have strong correlation between
    RGB channels. By subtracting G from R and B, the residual values cluster
    around zero, making subsequent compression (LZ77 + Huffman) more efficient.

    Args:
        rgba_bytes: Flattened RGBA pixel bytes (length = width * height * 4).
        width:  Image width in pixels.
        height: Image height in pixels.
    Returns:
        bytes: Transformed RGBA bytes with same length.
    """
    arr = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape((height, width, 4)).copy()
    green = arr[:, :, 1].astype(np.int16)

    # R' = (R - G) & 0xFF
    arr[:, :, 0] = ((arr[:, :, 0].astype(np.int16) - green) & 0xFF).astype(np.uint8)
    # B' = (B - G) & 0xFF
    arr[:, :, 2] = ((arr[:, :, 2].astype(np.int16) - green) & 0xFF).astype(np.uint8)

    return arr.tobytes()


def inverse_subtract_green(transformed_bytes: bytes, width: int, height: int) -> bytes:
    """
    Reverse Subtract Green Transform — restore original R and B channels.

    For each RGBA pixel:
        R = (R' + G) mod 256
        B = (B' + G) mod 256

    Args:
        transformed_bytes: Subtract-Green-transformed RGBA bytes.
        width:  Image width in pixels.
        height: Image height in pixels.
    Returns:
        bytes: Original RGBA pixel bytes.
    """
    arr = np.frombuffer(transformed_bytes, dtype=np.uint8).reshape((height, width, 4)).copy()
    green = arr[:, :, 1].astype(np.int16)

    # R = (R' + G) & 0xFF
    arr[:, :, 0] = ((arr[:, :, 0].astype(np.int16) + green) & 0xFF).astype(np.uint8)
    # B = (B' + G) & 0xFF
    arr[:, :, 2] = ((arr[:, :, 2].astype(np.int16) + green) & 0xFF).astype(np.uint8)

    return arr.tobytes()


# ==============================================================================
# RIFF WEBP CONTAINER PACKING / PARSING
# ==============================================================================

def pack_riff_webp(vp8l_chunk_data: bytes) -> bytes:
    """
    Wrap VP8L bitstream data inside a RIFF WebP container.

    Container layout (little-endian):
    ┌─────────────────────────────────────────────┐
    │ 'RIFF'            (4 bytes)                 │
    │ file_size         (4 bytes LE, = total - 8) │
    │ 'WEBP'            (4 bytes)                 │
    │ 'VP8L'            (4 bytes)                 │
    │ chunk_data_size   (4 bytes LE)              │
    │ VP8L bitstream    (variable)                │
    │ [padding byte]    (1 byte if chunk is odd)  │
    └─────────────────────────────────────────────┘

    Args:
        vp8l_chunk_data: Complete VP8L bitstream bytes.
    Returns:
        bytes: Full RIFF WebP file bytes.
    """
    chunk_size = len(vp8l_chunk_data)
    # RIFF spec: add 1 padding byte if chunk data size is odd
    padding = b"\x00" if chunk_size % 2 != 0 else b""
    # file_size = everything after 'RIFF' + 4-byte size field
    #           = 4 (WEBP) + 4 (VP8L) + 4 (chunk_size) + data + padding
    file_size = 4 + 4 + 4 + chunk_size + len(padding)

    riff = bytearray()
    riff.extend(RIFF_MAGIC)                           # 'RIFF'
    riff.extend(struct.pack("<I", file_size))          # File size (LE)
    riff.extend(WEBP_MAGIC)                           # 'WEBP'
    riff.extend(VP8L_CHUNK_ID)                        # 'VP8L'
    riff.extend(struct.pack("<I", chunk_size))         # Chunk data size (LE)
    riff.extend(vp8l_chunk_data)                      # VP8L bitstream
    riff.extend(padding)                              # Padding (if odd)

    return bytes(riff)


def parse_riff_webp(webp_data: bytes) -> bytes:
    """
    Parse a RIFF WebP container and extract the VP8L chunk data.

    Args:
        webp_data: Full .webp file bytes.
    Returns:
        bytes: VP8L bitstream data (without RIFF/chunk headers).
    Raises:
        ValueError: If the file is not a valid RIFF WebP VP8L file.
    """
    if len(webp_data) < 20:
        raise ValueError("File too short to be a valid WebP file.")

    # Validate RIFF magic
    riff_magic = webp_data[0:4]
    if riff_magic != RIFF_MAGIC:
        raise ValueError(f"Invalid RIFF magic: {riff_magic!r}, expected {RIFF_MAGIC!r}")

    # Read file size (not strictly needed for parsing, but validate)
    file_size = struct.unpack("<I", webp_data[4:8])[0]

    # Validate WEBP magic
    webp_magic = webp_data[8:12]
    if webp_magic != WEBP_MAGIC:
        raise ValueError(f"Invalid WEBP magic: {webp_magic!r}, expected {WEBP_MAGIC!r}")

    # Validate VP8L chunk identifier
    chunk_id = webp_data[12:16]
    if chunk_id != VP8L_CHUNK_ID:
        raise ValueError(f"Expected VP8L chunk, got: {chunk_id!r}")

    # Read VP8L chunk data size
    chunk_size = struct.unpack("<I", webp_data[16:20])[0]
    vp8l_data = webp_data[20 : 20 + chunk_size]

    if len(vp8l_data) < chunk_size:
        raise ValueError("VP8L chunk data is truncated.")

    return vp8l_data


# ==============================================================================
# VP8L BITSTREAM HEADER PACKING / PARSING
# ==============================================================================

def pack_vp8l_header(width: int, height: int, has_alpha: bool) -> bytes:
    """
    Pack the VP8L bitstream header (5 bytes total, little-endian).

    Layout:
        Signature: 0x2F (1 byte)
        Packed bits (4 bytes, little-endian uint32):
            bits  [0..13]:  width  - 1   (14 bits)
            bits [14..27]:  height - 1   (14 bits)
            bit       [28]: alpha_is_used (1 bit)
            bits [29..31]:  version       (3 bits, must be 0)

    Args:
        width:     Image width (1 to 16384).
        height:    Image height (1 to 16384).
        has_alpha: True if image uses alpha channel.
    Returns:
        bytes: 5-byte VP8L header.
    """
    signature = bytes([VP8L_SIGNATURE])

    packed = (
        ((width - 1) & 0x3FFF)
        | (((height - 1) & 0x3FFF) << 14)
        | (int(has_alpha) << 28)
        | (VP8L_VERSION << 29)
    )
    header_bytes = struct.pack("<I", packed)

    return signature + header_bytes


def parse_vp8l_header(data: bytes) -> Tuple[int, int, bool, int, int]:
    """
    Parse the VP8L bitstream header.

    Args:
        data: VP8L bitstream data (at least 5 bytes).
    Returns:
        tuple: (width, height, has_alpha, version, header_byte_size)
    Raises:
        ValueError: If signature or version is invalid.
    """
    if len(data) < 5:
        raise ValueError("VP8L data too short for header.")

    signature = data[0]
    if signature != VP8L_SIGNATURE:
        raise ValueError(
            f"Invalid VP8L signature: 0x{signature:02X}, expected 0x{VP8L_SIGNATURE:02X}"
        )

    packed = struct.unpack("<I", data[1:5])[0]

    width = (packed & 0x3FFF) + 1
    height = ((packed >> 14) & 0x3FFF) + 1
    has_alpha = bool((packed >> 28) & 1)
    version = (packed >> 29) & 0x7

    if version != VP8L_VERSION:
        raise ValueError(f"Unsupported VP8L version: {version}, expected {VP8L_VERSION}")

    return width, height, has_alpha, version, 5


# ==============================================================================
# CUSTOM TRANSFORM METADATA (stored right after VP8L header)
# ==============================================================================

def pack_transform_metadata(
    has_subtract_green: bool,
    filter_type: int,
    original_mode: str,
    channels: int,
) -> bytes:
    """
    Pack custom transform metadata into binary (7 bytes).

    Stored so the decompressor knows which transforms to reverse.

    Args:
        has_subtract_green: Whether Subtract Green was applied.
        filter_type:        FilterType enum value used for spatial filtering.
        original_mode:      Original image color mode string ("RGB", "RGBA", "L").
        channels:           Number of color channels (e.g. 4 for RGBA).
    Returns:
        bytes: 7-byte packed metadata.
    """
    mode_bytes = original_mode.encode("ascii").ljust(4, b"\x00")[:4]
    return struct.pack(
        TRANSFORM_META_FORMAT,
        int(has_subtract_green),
        filter_type,
        mode_bytes,
        channels,
    )


def parse_transform_metadata(data: bytes) -> Tuple[bool, int, str, int]:
    """
    Parse custom transform metadata from binary data.

    Args:
        data: At least TRANSFORM_META_SIZE bytes.
    Returns:
        tuple: (has_subtract_green, filter_type_val, original_mode_str, channels)
    """
    has_sg, ftype, mode_bytes, channels = struct.unpack(
        TRANSFORM_META_FORMAT, data[:TRANSFORM_META_SIZE]
    )
    original_mode = mode_bytes.decode("ascii").rstrip("\x00")
    return bool(has_sg), ftype, original_mode, channels


# ==============================================================================
# WEBP LOSSLESS COMPRESSOR CLASS
# ==============================================================================

class WebPLosslessCompressor:
    """
    Main image compressor using the WebP Lossless algorithm.
    Pipeline: Subtract Green → Spatial Filter → LZ77 → Huffman → RIFF/VP8L container.
    """

    def __init__(
        self,
        filter_type: FilterType = FilterType.ADAPTIVE,
        use_subtract_green: bool = True,
    ):
        """
        Initialize the WebP Lossless compressor.

        Args:
            filter_type: Spatial prediction filter type
                         (NONE, SUB, UP, AVERAGE, PAETH, ADAPTIVE).
            use_subtract_green: If True, apply Subtract Green Transform before filtering.
                                This reduces color correlation and improves compression.
        """
        self.filter_type = filter_type
        self.use_subtract_green = use_subtract_green

    # --------------------------------------------------------------------------
    # CORE COMPRESSION PIPELINE
    # --------------------------------------------------------------------------
    def compress(self, image_input: Union[str, Image.Image, np.ndarray]) -> bytes:
        """
        Compress an image into WebP Lossless binary format (.webp).

        Full pipeline:
        1. Load image → convert to RGBA (4 channels)
        2. Subtract Green Transform (reduce inter-channel correlation)
        3. Spatial prediction filtering (reduce 2D spatial correlation)
        4. LZ77 sliding window dictionary compression
        5. Huffman entropy encoding
        6. Pack into VP8L bitstream
        7. Wrap in RIFF WebP container

        Args:
            image_input: File path (str), PIL Image, or numpy array.
        Returns:
            bytes: Compressed data in RIFF/WebP/VP8L format.
        """
        # Step 1: Load image and flatten pixels into 1D bytes
        raw_bytes, width, height, channels, mode = load_image(image_input)
        original_mode = mode

        # Convert to RGBA — WebP Lossless internally works on RGBA pixels
        if mode != "RGBA":
            img = raw_bytes_to_image(raw_bytes, width, height, mode)
            img = img.convert("RGBA")
            raw_bytes = img.tobytes()
            channels = 4
            mode = "RGBA"

        has_alpha = True  # RGBA always has alpha channel

        # Step 2: Subtract Green Transform (WebP-specific color decorrelation)
        if self.use_subtract_green:
            pixel_data = apply_subtract_green(raw_bytes, width, height)
        else:
            pixel_data = raw_bytes

        # Step 3: Spatial prediction filter (Predictor Transform)
        bpp = channels  # bytes per pixel = 4 for RGBA
        filtered_bytes = filter_scanlines(
            pixel_data, width, height, bpp, self.filter_type
        )

        # Step 4: LZ77 sliding window dictionary compression
        lz77_bytes = lz77_encode(filtered_bytes)

        # Step 5: Huffman entropy coding on top of LZ77 output
        compressed_payload = huffman_encode(lz77_bytes)

        # Step 6: Assemble VP8L bitstream (header + metadata + payload)
        vp8l_header = pack_vp8l_header(width, height, has_alpha)
        transform_meta = pack_transform_metadata(
            has_subtract_green=self.use_subtract_green,
            filter_type=int(self.filter_type),
            original_mode=original_mode,
            channels=channels,
        )
        vp8l_chunk_data = vp8l_header + transform_meta + compressed_payload

        # Step 7: Wrap in RIFF WebP container
        webp_bytes = pack_riff_webp(vp8l_chunk_data)

        return webp_bytes

    # --------------------------------------------------------------------------
    # FILE-LEVEL COMPRESSION
    # --------------------------------------------------------------------------
    def compress_file(
        self, input_image_path: str, output_webp_path: str
    ) -> Dict[str, float]:
        """
        Compress an image file and save output as .webp file on disk.

        Args:
            input_image_path: Path to source image file.
            output_webp_path: Path for output .webp file.
        Returns:
            dict: Compression statistics (sizes, ratio, time).
        """
        start_time = time.perf_counter()
        compressed_data = self.compress(input_image_path)
        compress_time = time.perf_counter() - start_time

        with open(output_webp_path, "wb") as f:
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
            "compression_time_sec": compress_time,
        }


# ==============================================================================
# BENCHMARK & PERFORMANCE MEASUREMENT
# ==============================================================================

def benchmark_webp_lossless(
    image_input: Union[str, Image.Image],
) -> Dict[str, Union[float, int, bool, str]]:
    """
    Run full performance evaluation comparing:
    1. WebP Lossless WITHOUT Subtract Green Transform
    2. WebP Lossless WITH Subtract Green Transform (full pipeline)
    Includes bit-for-bit lossless check, MSE, PSNR, CR, BPP, and speed.

    Args:
        image_input: File path or PIL Image.
    Returns:
        dict: Comprehensive benchmark metrics.
    """
    # Import decompressor here to avoid circular import
    from .decompressor import WebPLosslessDecompressor

    # Load image for benchmark
    raw_bytes, width, height, channels, mode = load_image(image_input)
    raw_size = len(raw_bytes)
    img_orig = raw_bytes_to_image(raw_bytes, width, height, mode)

    # 1. Mode 1: Without Subtract Green
    comp_no_sg = WebPLosslessCompressor(
        filter_type=FilterType.ADAPTIVE, use_subtract_green=False
    )
    t0 = time.perf_counter()
    bytes_no_sg = comp_no_sg.compress(img_orig)
    t_comp_no_sg = time.perf_counter() - t0

    # 2. Mode 2: With Subtract Green (full pipeline)
    comp_with_sg = WebPLosslessCompressor(
        filter_type=FilterType.ADAPTIVE, use_subtract_green=True
    )
    t1 = time.perf_counter()
    bytes_with_sg = comp_with_sg.compress(img_orig)
    t_comp_with_sg = time.perf_counter() - t1

    # 3. Decompress and verify quality
    decomp = WebPLosslessDecompressor()
    t2 = time.perf_counter()
    img_rec = decomp.decompress(bytes_with_sg)
    t_decomp = time.perf_counter() - t2

    # Verify bit-for-bit match and calculate MSE/PSNR
    is_bit_identical, mse, psnr_db = calculate_image_metrics(img_orig, img_rec)

    size_no_sg = len(bytes_no_sg)
    size_with_sg = len(bytes_with_sg)
    cr_no_sg = raw_size / size_no_sg if size_no_sg > 0 else 0
    cr_with_sg = raw_size / size_with_sg if size_with_sg > 0 else 0
    bpp = (size_with_sg * 8) / (width * height)
    saving = (1.0 - size_with_sg / raw_size) * 100.0

    return {
        "dimensions": f"{width}x{height}",
        "raw_pixel_bytes": raw_size,
        "no_subtract_green_bytes": size_no_sg,
        "no_subtract_green_cr": cr_no_sg,
        "with_subtract_green_bytes": size_with_sg,
        "with_subtract_green_cr": cr_with_sg,
        "space_saving_pct": saving,
        "bits_per_pixel": bpp,
        "compress_time_sec": t_comp_with_sg,
        "decompress_time_sec": t_decomp,
        "is_lossless": is_bit_identical,
        "mse": mse,
        "psnr_db": psnr_db,
    }


# ==============================================================================
# CONVENIENCE HELPER FUNCTIONS
# ==============================================================================

def compress_image_webp_lossless(
    input_path: str,
    output_path: str,
    filter_type: FilterType = FilterType.ADAPTIVE,
    use_subtract_green: bool = True,
):
    """Convenience helper function to compress an image file to .webp"""
    c = WebPLosslessCompressor(
        filter_type=filter_type, use_subtract_green=use_subtract_green
    )
    return c.compress_file(input_path, output_path)
