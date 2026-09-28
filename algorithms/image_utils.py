"""
Image utility module for loading, flattening, reconstructing, and comparing images.
Written in simple A2/B1 English.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image


def load_image(image_input: Union[str, Image.Image, np.ndarray]) -> Tuple[bytes, int, int, int, str]:
    """
    Load an image, normalize color mode, and flatten pixels into 1D bytes.
    
    Args:
        image_input: File path (str), PIL Image, or numpy array.
    Returns:
        tuple: (raw_bytes, width, height, channels, mode)
    """
    if isinstance(image_input, str):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        raise TypeError("image_input must be a file path, PIL Image, or numpy array.")

    # Normalize to RGB or Grayscale (L) or RGBA
    mode = img.mode
    if mode not in ("RGB", "L", "RGBA"):
        img = img.convert("RGB")
        mode = "RGB"

    width, height = img.size
    channels = len(img.getbands())
    raw_bytes = img.tobytes()

    return raw_bytes, width, height, channels, mode


def raw_bytes_to_image(raw_bytes: bytes, width: int, height: int, mode: str) -> Image.Image:
    """
    Reconstruct a 2D PIL Image from 1D raw bytes.
    
    Args:
        raw_bytes: 1D byte array of pixels.
        width: Image width in pixels.
        height: Image height in pixels.
        mode: Color mode (e.g. "RGB", "L", "RGBA").
    Returns:
        Image.Image: Reconstructed PIL Image.
    """
    return Image.frombytes(mode, (width, height), raw_bytes)


def save_image(img: Image.Image, output_path: str) -> None:
    """Save a PIL Image to a file path."""
    img.save(output_path)


def calculate_image_metrics(img1: Image.Image, img2: Image.Image) -> Tuple[bool, float, float]:
    """
    Compare two images and calculate error metrics.
    
    Returns:
        tuple: (is_lossless, mse, psnr_db)
            - is_lossless: True if 100% bit-for-bit identical.
            - mse: Mean Squared Error (0.0 means perfect match).
            - psnr_db: Peak Signal-to-Noise Ratio (infinity if identical).
    """
    arr1 = np.array(img1.convert("RGB"))
    arr2 = np.array(img2.convert("RGB"))

    is_lossless = bool(np.array_equal(arr1, arr2))
    mse = float(np.mean((arr1.astype(float) - arr2.astype(float)) ** 2))
    psnr_db = float("inf") if mse == 0.0 else float(20 * np.log10(255.0 / np.sqrt(mse)))

    return is_lossless, mse, psnr_db
