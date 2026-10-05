"""
WebP Lossless Image Compressor Module (algorithms/webp_lossless/compressor.py)

Coordinates the WebP Lossless compression pipeline supporting two engines:
1. Standard LibWebP Engine (default, use_pure_python=False):
   - Fast, C-accelerated WebP Lossless encoder via Google libwebp (Pillow).
   - Generates 100% compliant .webp files readable by any browser or image viewer.
2. Pure Python Educational Engine (use_pure_python=True):
   - Modular educational pipeline for studying WebP internal algorithms:
     * Step 1: Pixel loading and flattening (image_utils)
     * Step 2: Subtract Green Transform (color decorrelation)
     * Step 3: Spatial prediction filtering (spatial_filters)
     * Step 4: LZ77 sliding window dictionary coding (lz77)
     * Step 5: Canonical Huffman coding with LSB-first bitstream (huffman_webp)
     * Step 6: RIFF WebP container with VP8L chunk packing

Written in simple A2/B1 English.
"""

import io
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
    calculate_image_metrics,
)
from ..common.spatial_filters import (
    FilterType,
    filter_scanlines,
    unfilter_scanlines,
)
from ..common.lz77 import lz77_encode, lz77_decode
from ..common.huffman_webp import huffman_webp_encode, huffman_webp_decode


# ==============================================================================
# RIFF / WEBP / VP8L CONTAINER CONSTANTS
# ==============================================================================
RIFF_MAGIC = b"RIFF"          # RIFF container identifier
WEBP_MAGIC = b"WEBP"          # WebP format identifier
VP8L_CHUNK_ID = b"VP8L"       # VP8L lossless chunk identifier
VP8L_SIGNATURE = 0x2F         # VP8L bitstream signature byte
VP8L_VERSION = 0              # VP8L version (must be 0)

# Custom transform metadata format for Pure Python pipeline
# Marker: b"WPPY" (4 bytes) + has_sg (B) + filter_type (B) + original_mode (4s) + channels (B)
PY_ENGINE_MARKER = b"WPPY"
TRANSFORM_META_FORMAT = "<4sBB4sB"
TRANSFORM_META_SIZE = struct.calcsize(TRANSFORM_META_FORMAT)  # = 11 bytes


# ==============================================================================
# SUBTRACT GREEN TRANSFORM (WebP Lossless color decorrelation)
# ==============================================================================

def apply_subtract_green(rgba_bytes: bytes, width: int, height: int) -> bytes:
    """
    Subtract Green Transform - reduces inter-channel color correlation.

    For each RGBA pixel:
        R' = (R - G) mod 256   <- Red minus Green
        B' = (B - G) mod 256   <- Blue minus Green
        G' = G                 <- Green unchanged
        A' = A                 <- Alpha unchanged

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
    Reverse Subtract Green Transform - restore original R and B channels.

    For each RGBA pixel:
        R = (R' + G) mod 256
        B = (B' + G) mod 256

    Args:
        transformed_bytes: Subtract-Green-transformed RGBA bytes.
        width:  Image width in pixels.
        height: Image height in pixels.
    Returns:
        bytes: Restored RGBA pixel bytes.
    """
    arr = np.frombuffer(transformed_bytes, dtype=np.uint8).reshape((height, width, 4)).copy()
    green = arr[:, :, 1].astype(np.int16)

    # R = (R' + G) & 0xFF
    arr[:, :, 0] = ((arr[:, :, 0].astype(np.int16) + green) & 0xFF).astype(np.uint8)
    # B = (B' + G) & 0xFF
    arr[:, :, 2] = ((arr[:, :, 2].astype(np.int16) + green) & 0xFF).astype(np.uint8)

    return arr.tobytes()


# ==============================================================================
# RIFF CONTAINER PACKING / PARSING
# ==============================================================================

def pack_riff_webp(vp8l_chunk_data: bytes) -> bytes:
    """
    Wrap VP8L chunk data in a valid RIFF container.

    Structure:
        Offset  Size   Field
        0       4      "RIFF" magic
        4       4      File size - 8 (little-endian uint32)
        8       4      "WEBP" magic
        12      4      "VP8L" chunk identifier
        16      4      Chunk data size (little-endian uint32)
        20      N      VP8L bitstream
        20+N    0..1   Padding byte (0x00) if N is odd

    Args:
        vp8l_chunk_data: VP8L bitstream bytes.
    Returns:
        bytes: Complete .webp file byte stream.
    """
    chunk_size = len(vp8l_chunk_data)
    padding = b"\x00" if (chunk_size % 2 != 0) else b""

    # Total file size minus the 8-byte RIFF header
    file_size_field = 4 + 8 + chunk_size + len(padding)

    riff = bytearray()
    riff.extend(RIFF_MAGIC)                           # "RIFF"
    riff.extend(struct.pack("<I", file_size_field))   # File size - 8 (LE)
    riff.extend(WEBP_MAGIC)                           # "WEBP"
    riff.extend(VP8L_CHUNK_ID)                        # "VP8L"
    riff.extend(struct.pack("<I", chunk_size))        # Chunk data size (LE)
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

    riff_magic = webp_data[0:4]
    if riff_magic != RIFF_MAGIC:
        raise ValueError(f"Invalid RIFF magic: {riff_magic!r}, expected {RIFF_MAGIC!r}")

    webp_magic = webp_data[8:12]
    if webp_magic != WEBP_MAGIC:
        raise ValueError(f"Invalid WEBP magic: {webp_magic!r}, expected {WEBP_MAGIC!r}")

    chunk_id = webp_data[12:16]
    if chunk_id != VP8L_CHUNK_ID:
        raise ValueError(f"Expected VP8L chunk, got: {chunk_id!r}")

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
    version = (packed >> 29) & 7

    if version != VP8L_VERSION:
        raise ValueError(f"Unsupported VP8L version: {version}, expected 0")

    return width, height, has_alpha, version, 5


# ==============================================================================
# TRANSFORM METADATA PACKING / PARSING (Pure Python Engine)
# ==============================================================================

def pack_transform_metadata(
    has_subtract_green: bool,
    filter_type: int,
    original_mode: str,
    channels: int,
) -> bytes:
    """
    Pack transform settings for pure Python engine.

    Layout:
        marker:        "WPPY" (4 bytes)
        has_sg:        uint8 (1 byte, 1 if subtract green applied, 0 otherwise)
        filter_type:   uint8 (1 byte, FilterType enum int)
        original_mode: 4 bytes (padded with \x00, e.g. "RGB\x00", "RGBA")
        channels:      uint8 (1 byte, 1..4)
    """
    mode_bytes = original_mode.encode("ascii").ljust(4, b"\x00")[:4]
    return struct.pack(
        TRANSFORM_META_FORMAT,
        PY_ENGINE_MARKER,
        int(has_subtract_green),
        int(filter_type),
        mode_bytes,
        int(channels),
    )


def parse_transform_metadata(data: bytes) -> Tuple[bool, int, str, int]:
    """
    Parse transform metadata for pure Python engine.

    Returns:
        tuple: (has_subtract_green, filter_type_val, original_mode, channels)
    """
    if len(data) < TRANSFORM_META_SIZE:
        raise ValueError("Data too short for transform metadata.")

    marker, has_sg, filter_type_val, mode_bytes, channels = struct.unpack(
        TRANSFORM_META_FORMAT, data[:TRANSFORM_META_SIZE]
    )
    if marker != PY_ENGINE_MARKER:
        raise ValueError(f"Invalid pure Python engine marker: {marker!r}")

    original_mode = mode_bytes.rstrip(b"\x00").decode("ascii", errors="replace")
    return bool(has_sg), filter_type_val, original_mode, channels


# ==============================================================================
# WEBP LOSSLESS COMPRESSOR CLASS
# ==============================================================================

class WebPLosslessCompressor:
    """
    WebP Lossless Image Compressor.
    Supports both standard LibWebP (default) and Pure Python educational mode.
    """

    def __init__(
        self,
        filter_type: FilterType = FilterType.ADAPTIVE,
        use_subtract_green: bool = True,
        use_pure_python: bool = False,
        quality: int = 100,
    ):
        """
        Initialize the compressor.

        Args:
            filter_type: Spatial prediction filter (ADAPTIVE, NONE, SUB, UP, AVERAGE, PAETH).
                         Only used in pure Python mode.
            use_subtract_green: Apply Subtract Green Transform (color decorrelation).
                                Only used in pure Python mode.
            use_pure_python: If False (default), uses fast C-accelerated Google libwebp (Pillow).
                             If True, uses modular pure Python pipeline (LZ77 + Canonical Huffman).
            quality: Compression effort/quality level for libwebp (0 to 100). Default is 100.
        """
        self.filter_type = filter_type
        self.use_subtract_green = use_subtract_green
        self.use_pure_python = use_pure_python
        self.quality = quality

    # --------------------------------------------------------------------------
    # IMAGE COMPRESSION PIPELINE
    # --------------------------------------------------------------------------
    def compress(self, image_input: Union[str, Image.Image, np.ndarray]) -> bytes:
        """
        Compress an image into WebP Lossless (.webp) format.

        When use_pure_python is False (default):
            Uses Google's libwebp via Pillow (lossless=True).
            Generates standard, fully compatible .webp files.

        When use_pure_python is True:
            1. Load image and convert to RGBA pixels.
            2. Subtract Green Transform (decorrelate R/B from G).
            3. Spatial prediction filtering (scanlines).
            4. LZ77 sliding window dictionary coding.
            5. WebP Canonical Huffman coding (LSB-first bitstream).
            6. Pack into RIFF WebP container with VP8L chunk.

        Args:
            image_input: File path (str), PIL Image, or numpy array.
        Returns:
            bytes: Compressed data in .webp format.
        """
        if not self.use_pure_python:
            return self._compress_libwebp(image_input)
        else:
            return self._compress_pure_python(image_input)

    def _compress_libwebp(self, image_input: Union[str, Image.Image, np.ndarray]) -> bytes:
        """Compress image using Google libwebp via Pillow (standard compliant)."""
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            raw_bytes, width, height, channels, mode = load_image(image_input)
            img = raw_bytes_to_image(raw_bytes, width, height, mode)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        buffer = io.BytesIO()
        img.save(buffer, format="WEBP", lossless=True, exact=True, quality=self.quality)
        return buffer.getvalue()

    def _compress_pure_python(self, image_input: Union[str, Image.Image, np.ndarray]) -> bytes:
        """Compress image using pure Python modular pipeline."""
        # Step 1: Load image and flatten pixels into 1D bytes
        raw_bytes, width, height, channels, mode = load_image(image_input)
        original_mode = mode

        # Convert to RGBA - WebP Lossless internally works on RGBA pixels
        if mode != "RGBA":
            img = raw_bytes_to_image(raw_bytes, width, height, mode)
            img = img.convert("RGBA")
            raw_bytes = img.tobytes()
            channels = 4
            mode = "RGBA"

        has_alpha = True

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

        # Step 5: WebP Canonical Huffman entropy coding on top of LZ77 output
        compressed_payload = huffman_webp_encode(lz77_bytes)

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
        return pack_riff_webp(vp8l_chunk_data)

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
            "engine": "pure_python" if self.use_pure_python else "libwebp",
        }


# ==============================================================================
# BENCHMARK & PERFORMANCE MEASUREMENT
# ==============================================================================

def benchmark_webp_lossless(
    image_input: Union[str, Image.Image],
) -> Dict[str, Union[float, int, bool, str]]:
    """
    Run full performance evaluation comparing:
    1. Standard LibWebP (C-accelerated, standard .webp)
    2. Pure Python WebP WITH Subtract Green Transform
    3. Pure Python WebP WITHOUT Subtract Green Transform
    Includes bit-for-bit lossless check, MSE, PSNR, CR, BPP, and speed.

    Args:
        image_input: File path or PIL Image.
    Returns:
        dict: Comprehensive benchmark metrics.
    """
    from .decompressor import WebPLosslessDecompressor

    raw_bytes, width, height, channels, mode = load_image(image_input)
    raw_size = len(raw_bytes)
    img_orig = raw_bytes_to_image(raw_bytes, width, height, mode)

    # 1. Mode 1: Standard LibWebP
    comp_libwebp = WebPLosslessCompressor(use_pure_python=False)
    t0 = time.perf_counter()
    bytes_libwebp = comp_libwebp.compress(img_orig)
    t_comp_libwebp = time.perf_counter() - t0

    decomp = WebPLosslessDecompressor()
    t_d0 = time.perf_counter()
    img_rec_libwebp = decomp.decompress(bytes_libwebp)
    t_decomp_libwebp = time.perf_counter() - t_d0
    is_lossless_libwebp, mse_lib, psnr_lib = calculate_image_metrics(img_orig, img_rec_libwebp)

    # 2. Mode 2: Pure Python WITH Subtract Green
    comp_py_sg = WebPLosslessCompressor(
        use_pure_python=True, filter_type=FilterType.ADAPTIVE, use_subtract_green=True
    )
    t1 = time.perf_counter()
    bytes_py_sg = comp_py_sg.compress(img_orig)
    t_comp_py_sg = time.perf_counter() - t1

    t_d1 = time.perf_counter()
    img_rec_py_sg = decomp.decompress(bytes_py_sg)
    t_decomp_py_sg = time.perf_counter() - t_d1
    is_lossless_py, mse_py, psnr_py = calculate_image_metrics(img_orig, img_rec_py_sg)

    # 3. Mode 3: Pure Python WITHOUT Subtract Green
    comp_py_nosg = WebPLosslessCompressor(
        use_pure_python=True, filter_type=FilterType.ADAPTIVE, use_subtract_green=False
    )
    t2 = time.perf_counter()
    bytes_py_nosg = comp_py_nosg.compress(img_orig)
    t_comp_py_nosg = time.perf_counter() - t2

    size_lib = len(bytes_libwebp)
    size_py_sg = len(bytes_py_sg)
    size_py_nosg = len(bytes_py_nosg)

    return {
        "dimensions": f"{width}x{height}",
        "raw_pixel_bytes": raw_size,
        # LibWebP
        "libwebp_bytes": size_lib,
        "libwebp_cr": raw_size / size_lib if size_lib > 0 else 0,
        "libwebp_compress_time": t_comp_libwebp,
        "libwebp_decompress_time": t_decomp_libwebp,
        "libwebp_lossless": is_lossless_libwebp,
        # Pure Python with SG
        "py_with_sg_bytes": size_py_sg,
        "py_with_sg_cr": raw_size / size_py_sg if size_py_sg > 0 else 0,
        "py_with_sg_time": t_comp_py_sg,
        "py_with_sg_lossless": is_lossless_py,
        # Pure Python without SG
        "py_no_sg_bytes": size_py_nosg,
        "py_no_sg_cr": raw_size / size_py_nosg if size_py_nosg > 0 else 0,
        "py_no_sg_time": t_comp_py_nosg,
    }


# ==============================================================================
# CONVENIENCE HELPER FUNCTIONS
# ==============================================================================

def compress_image_webp_lossless(
    input_path: str,
    output_path: str,
    use_pure_python: bool = False,
    filter_type: FilterType = FilterType.ADAPTIVE,
    use_subtract_green: bool = True,
):
    """Convenience helper function to compress an image file to .webp"""
    c = WebPLosslessCompressor(
        use_pure_python=use_pure_python,
        filter_type=filter_type,
        use_subtract_green=use_subtract_green,
    )
    return c.compress_file(input_path, output_path)
