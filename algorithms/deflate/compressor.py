"""
Deflate Image Compressor Module  (algorithms/deflate/compressor.py)

Hai chế độ hoạt động:
  1. Pillow PNG (mặc định, use_pillow=True):
       - Nén : PIL.Image → PIL.Image.save(PNG, compress_level=9) → bytes
       - Giải: bytes → PIL.Image.open() → PIL Image
       - Toàn bộ dùng thư viện C (libpng + zlib), cùng chuẩn so sánh với
         TIFF-LZW (libtiff) và WebP (libwebp).
       - Đây là kết quả benchmark chính được báo cáo trong bảng so sánh.

  2. Custom pipeline (use_pillow=False, chỉ học thuật):
       - Nén : load pixel → spatial filter (NumPy) → zlib / LZ77+Huffman Python
       - Giải: unpack header → zlib / LZ77+Huffman decode → unfilter (Python loop)
       - Container: file .defimg (20-byte binary header tự định nghĩa)
       - Chỉ dùng để minh họa nguyên lý, KHÔNG dùng trong benchmark so sánh.

Written in simple A2/B1 English.
"""

import io
import os
import struct
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Import modular components (chỉ dùng trong custom pipeline)
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
from ..common.huffman import huffman_encode, huffman_decode

# 20-byte fixed header format for .defimg container (custom pipeline only)
# Format: Magic(4s) + Version(B) + FilterMode(B) + ColorMode(4s) + Width(I) + Height(I) + Channels(B) + Engine(B)
HEADER_FORMAT = ">4sBB4sIIBB"
MAGIC_BYTES = b"DFLT"
CONTAINER_VERSION = 1

# engine_byte values stored in .defimg header
ENGINE_ZLIB = 0        # zlib RFC-1951 (C)
ENGINE_PURE_PYTHON = 1 # custom LZ77 + Huffman (Python, educational)
ENGINE_PILLOW_PNG = 2  # Pillow PNG / libpng (C) — main benchmark pipeline


class DeflateImageCompressor:
    """
    Image compressor using DEFLATE/PNG.

    Chế độ mặc định (use_pillow=True):
      - Nén bằng Pillow PNG (libpng + zlib level 9) — chuẩn so sánh chính.
      - Output là PNG bytes thuần, KHÔNG dùng .defimg container.

    Chế độ học thuật (use_pillow=False):
      - Dùng pipeline tự cài: spatial filter → zlib hoặc LZ77+Huffman Python.
      - Output là .defimg binary container.
    """

    def __init__(
        self,
        compression_level: int = 9,
        filter_type: FilterType = FilterType.ADAPTIVE,
        use_pure_python: bool = False,
        use_pillow: bool = True,
    ):
        """
        Khởi tạo compressor.

        Args:
            compression_level: Mức nén zlib 1–9 (chỉ dùng khi use_pillow=False).
            filter_type: Loại spatial filter PNG (chỉ dùng khi use_pillow=False).
            use_pure_python: Nếu True dùng LZ77+Huffman Python (học thuật).
                             Chỉ có tác dụng khi use_pillow=False.
            use_pillow: Nếu True (mặc định) dùng Pillow PNG làm pipeline chính.
                        Nếu False dùng pipeline tự cài (học thuật).
        """
        if not (1 <= compression_level <= 9):
            raise ValueError("compression_level must be between 1 and 9.")
        self.compression_level = compression_level
        self.filter_type = filter_type
        self.use_pure_python = use_pure_python
        self.use_pillow = use_pillow

    # --------------------------------------------------------------------------
    # PILLOW PNG PIPELINE (benchmark chính)
    # --------------------------------------------------------------------------
    def _pillow_compress(self, img: Image.Image) -> bytes:
        """Nén ảnh thành PNG bytes bằng Pillow (libpng, zlib level 9)."""
        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=self.compression_level, optimize=False)
        return buf.getvalue()

    def _pillow_decompress(self, data: bytes) -> Image.Image:
        """Giải nén PNG bytes thành PIL Image bằng Pillow (libpng)."""
        buf = io.BytesIO(data)
        img = Image.open(buf)
        img.load()  # Đảm bảo dữ liệu pixel được giải mã ngay
        return img

    # --------------------------------------------------------------------------
    # CUSTOM PIPELINE (học thuật — LZ77 + Huffman / zlib + spatial filter)
    # --------------------------------------------------------------------------
    def _deflate_compress(self, data: bytes) -> bytes:
        """Nén byte stream bằng DEFLATE (LZ77 + Huffman Coding)."""
        if self.use_pure_python:
            lz77_bytes = lz77_encode(data)
            return huffman_encode(lz77_bytes)
        else:
            return zlib_compress(data, level=self.compression_level)

    def _deflate_decompress(self, compressed_payload: bytes, is_pure_python: bool) -> bytes:
        """Giải nén byte stream bằng DEFLATE (Huffman Decode + LZ77 Decode)."""
        if is_pure_python:
            lz77_bytes = huffman_decode(compressed_payload)
            return lz77_decode(lz77_bytes)
        else:
            return zlib_decompress(compressed_payload)

    # --------------------------------------------------------------------------
    # COMPRESS (router)
    # --------------------------------------------------------------------------
    def compress(self, image_input: Union[str, Image.Image, np.ndarray]) -> bytes:
        """
        Nén ảnh thành bytes.

        - use_pillow=True : trả về PNG bytes (chuẩn benchmark).
        - use_pillow=False: trả về .defimg bytes (học thuật).

        Args:
            image_input: Đường dẫn file, PIL Image, hoặc numpy array.
        Returns:
            bytes: Dữ liệu nén.
        """
        # Chuẩn hóa input thành PIL Image
        if isinstance(image_input, (str, np.ndarray)):
            raw_bytes, width, height, channels, mode = load_image(image_input)
            img = raw_bytes_to_image(raw_bytes, width, height, mode)
        else:
            img = image_input

        # ── Pillow PNG pipeline (benchmark chính) ──
        if self.use_pillow:
            return self._pillow_compress(img)

        # ── Custom pipeline (học thuật) ──
        raw_bytes, width, height, channels, mode = load_image(img)
        bpp = channels
        filtered_bytes = filter_scanlines(raw_bytes, width, height, bpp, self.filter_type)
        compressed_payload = self._deflate_compress(filtered_bytes)

        mode_bytes = mode.encode("ascii").ljust(4, b"\x00")
        engine_byte = ENGINE_PURE_PYTHON if self.use_pure_python else ENGINE_ZLIB
        header = struct.pack(
            HEADER_FORMAT,
            MAGIC_BYTES,
            CONTAINER_VERSION,
            int(self.filter_type),
            mode_bytes,
            width,
            height,
            channels,
            engine_byte,
        )
        return header + compressed_payload

    def decompress(self, compressed_data: bytes) -> Image.Image:
        """
        Giải nén bytes thành PIL Image.

        - use_pillow=True : đọc PNG bytes bằng Pillow.
        - use_pillow=False: đọc .defimg bytes bằng pipeline tự cài.

        Args:
            compressed_data: Dữ liệu nén từ compress().
        Returns:
            Image.Image: Ảnh gốc đã được khôi phục.
        """
        # ── Pillow PNG pipeline (benchmark chính) ──
        if self.use_pillow:
            return self._pillow_decompress(compressed_data)

        # ── Custom pipeline (học thuật) ──
        header_size = struct.calcsize(HEADER_FORMAT)
        if len(compressed_data) < header_size:
            raise ValueError("Invalid compressed data: shorter than header size.")

        magic, version, filter_mode_val, mode_bytes, width, height, channels, engine_byte = struct.unpack(
            HEADER_FORMAT, compressed_data[:header_size]
        )

        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid file signature: {magic}, expected: {MAGIC_BYTES}")

        mode = mode_bytes.decode("ascii").rstrip("\x00")
        is_pure_python = (engine_byte == ENGINE_PURE_PYTHON)
        payload = compressed_data[header_size:]

        filtered_bytes = self._deflate_decompress(payload, is_pure_python=is_pure_python)
        raw_bytes = unfilter_scanlines(filtered_bytes, width, height, channels)
        return raw_bytes_to_image(raw_bytes, width, height, mode)

    # --------------------------------------------------------------------------
    # FILE-LEVEL HELPERS
    # --------------------------------------------------------------------------
    def compress_file(self, input_image_path: str, output_path: str) -> Dict[str, float]:
        """Nén file ảnh và ghi ra đĩa. Trả về dict thống kê."""
        start = time.perf_counter()
        compressed_data = self.compress(input_image_path)
        compress_time = time.perf_counter() - start

        with open(output_path, "wb") as f:
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

    def decompress_file(
        self,
        input_path: str,
        output_image_path: Optional[str] = None,
    ) -> Tuple[Image.Image, float]:
        """Giải nén file từ đĩa, tùy chọn lưu ảnh ra file."""
        with open(input_path, "rb") as f:
            data = f.read()

        start = time.perf_counter()
        img = self.decompress(data)
        decomp_time = time.perf_counter() - start

        if output_image_path:
            save_image(img, output_image_path)

        return img, decomp_time


# ------------------------------------------------------------------------------
# PRIVATE HELPERS (tránh import zlib ở top-level để không nhầm với custom impl)
# ------------------------------------------------------------------------------
import zlib as _zlib


def zlib_compress(data: bytes, level: int = 9) -> bytes:
    return _zlib.compress(data, level=level)


def zlib_decompress(data: bytes) -> bytes:
    return _zlib.decompress(data)


# ------------------------------------------------------------------------------
# BENCHMARK & PERFORMANCE MEASUREMENT
# ------------------------------------------------------------------------------
def benchmark_image_compression(
    image_input: Union[str, Image.Image],
    compression_level: int = 9,
) -> Dict[str, Union[float, int, bool, str]]:
    """
    Đo hiệu năng Pillow PNG pipeline (benchmark chính).
    So sánh: No-filter Deflate vs Adaptive-filter Deflate (học thuật thêm vào).
    """
    raw_bytes, width, height, channels, mode = load_image(image_input)
    raw_size = len(raw_bytes)
    img_orig = raw_bytes_to_image(raw_bytes, width, height, mode)

    # ── Pillow PNG (benchmark chính) ──
    comp_pillow = DeflateImageCompressor(compression_level=compression_level, use_pillow=True)
    t0 = time.perf_counter()
    bytes_pillow = comp_pillow.compress(img_orig)
    t_comp = time.perf_counter() - t0

    t1 = time.perf_counter()
    img_rec = comp_pillow.decompress(bytes_pillow)
    t_decomp = time.perf_counter() - t1

    is_bit_identical, mse, psnr_db = calculate_image_metrics(img_orig, img_rec)

    size_pillow = len(bytes_pillow)
    cr = raw_size / size_pillow if size_pillow > 0 else 0
    bpp = (size_pillow * 8) / (width * height)
    saving = (1.0 - size_pillow / raw_size) * 100.0

    return {
        "dimensions": f"{width}x{height}",
        "raw_pixel_bytes": raw_size,
        "adaptive_deflate_bytes": size_pillow,
        "adaptive_cr": cr,
        "space_saving_pct": saving,
        "bits_per_pixel": bpp,
        "compress_time_sec": t_comp,
        "decompress_time_sec": t_decomp,
        "is_lossless": is_bit_identical,
        "mse": mse,
        "psnr_db": psnr_db,
    }


# ------------------------------------------------------------------------------
# CONVENIENCE HELPERS
# ------------------------------------------------------------------------------
def compress_image_deflate(
    input_path: str,
    output_path: str,
    level: int = 9,
    filter_type: FilterType = FilterType.ADAPTIVE,
    use_pillow: bool = True,
):
    """Nén file ảnh thành PNG (Pillow) hoặc .defimg (custom)."""
    c = DeflateImageCompressor(
        compression_level=level,
        filter_type=filter_type,
        use_pillow=use_pillow,
    )
    return c.compress_file(input_path, output_path)


def decompress_image_deflate(
    input_path: str,
    output_image_path: str,
    use_pillow: bool = True,
):
    """Giải nén PNG hoặc .defimg thành file ảnh."""
    c = DeflateImageCompressor(use_pillow=use_pillow)
    return c.decompress_file(input_path, output_image_path)
