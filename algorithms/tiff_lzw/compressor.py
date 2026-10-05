"""TIFF-LZW image compression and decompression.

The compressed byte stream is a standards-compliant TIFF file. Pillow/libtiff
handles the TIFF container and the LZW codec, while this module provides the
same in-memory and file-based interface used by the other project codecs.
"""

import os
import time
from io import BytesIO
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..common.image_utils import load_image, raw_bytes_to_image, save_image


ImageInput = Union[str, Image.Image, np.ndarray]
TIFF_COMPRESSION_TAG = 259
TIFF_LZW_CODE = 5


class TiffLzwCompressor:
    """Compress and decompress images with LZW inside a TIFF container."""

    format_name = "TIFF_LZW"
    file_extension = ".tiff"

    @staticmethod
    def _normalize_image(image_input: ImageInput) -> Image.Image:
        """Return a detached L, RGB, or RGBA image with only pixel data."""
        raw_bytes, width, height, _, mode = load_image(image_input)
        return raw_bytes_to_image(raw_bytes, width, height, mode)

    def compress(self, image_input: ImageInput) -> bytes:
        """Encode an image as a complete TIFF file using lossless LZW."""
        image = self._normalize_image(image_input)
        output = BytesIO()

        try:
            image.save(output, format="TIFF", compression="tiff_lzw")
            compressed_data = output.getvalue()
        except OSError as exc:
            raise RuntimeError(
                "TIFF-LZW compression is unavailable in this Pillow installation."
            ) from exc
        finally:
            output.close()
            image.close()

        if not compressed_data:
            raise RuntimeError("TIFF-LZW compression produced no data.")

        return compressed_data

    def decompress(self, compressed_data: bytes) -> Image.Image:
        """Decode TIFF-LZW bytes and return a detached PIL image."""
        if not isinstance(compressed_data, bytes) or not compressed_data:
            raise ValueError("compressed_data must be non-empty bytes.")

        try:
            with BytesIO(compressed_data) as stream:
                with Image.open(stream) as image:
                    if image.format != "TIFF":
                        raise ValueError("Compressed data is not a TIFF image.")

                    compression_code = image.tag_v2.get(TIFF_COMPRESSION_TAG)
                    if compression_code != TIFF_LZW_CODE:
                        raise ValueError(
                            "TIFF image does not use LZW compression "
                            f"(compression tag: {compression_code!r})."
                        )

                    image.load()
                    return image.copy()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Invalid or corrupted TIFF-LZW data.") from exc

    def compress_file(
        self,
        input_image_path: str,
        output_tiff_path: str,
    ) -> Dict[str, float]:
        """Compress an image file to TIFF-LZW and return basic statistics."""
        start_time = time.perf_counter()
        compressed_data = self.compress(input_image_path)
        compression_time = time.perf_counter() - start_time

        with open(output_tiff_path, "wb") as output_file:
            output_file.write(compressed_data)

        original_size = os.path.getsize(input_image_path)
        compressed_size = len(compressed_data)
        compression_ratio = (
            original_size / compressed_size if compressed_size > 0 else 0.0
        )
        space_saving = (
            (1.0 - compressed_size / original_size) * 100.0
            if original_size > 0
            else 0.0
        )

        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "space_saving_pct": space_saving,
            "compression_time_sec": compression_time,
        }

    def decompress_file(
        self,
        input_tiff_path: str,
        output_image_path: Optional[str] = None,
    ) -> Tuple[Image.Image, float]:
        """Decompress a TIFF-LZW file and optionally save the restored image."""
        with open(input_tiff_path, "rb") as input_file:
            compressed_data = input_file.read()

        start_time = time.perf_counter()
        image = self.decompress(compressed_data)
        decompression_time = time.perf_counter() - start_time

        if output_image_path:
            save_image(image, output_image_path)

        return image, decompression_time


def compress_image_tiff_lzw(
    input_path: str,
    output_path: str,
) -> Dict[str, float]:
    """Compress an image file to a TIFF file using LZW."""
    return TiffLzwCompressor().compress_file(input_path, output_path)


def decompress_image_tiff_lzw(
    input_path: str,
    output_path: str,
) -> Tuple[Image.Image, float]:
    """Decompress a TIFF-LZW file and save the restored image."""
    return TiffLzwCompressor().decompress_file(input_path, output_path)
