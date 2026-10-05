"""
Quick round-trip test for WebP Lossless Compressor & Decompressor.
Creates a small test image, compresses to .webp, decompresses back,
and verifies 100% bit-for-bit lossless match.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from PIL import Image

from algorithms.webp_lossless.compressor import (
    WebPLosslessCompressor,
    apply_subtract_green,
    inverse_subtract_green,
)
from algorithms.webp_lossless.decompressor import WebPLosslessDecompressor
from algorithms.common.image_utils import calculate_image_metrics


def test_subtract_green_roundtrip():
    """Test that Subtract Green Transform is perfectly reversible."""
    print("=" * 60)
    print("TEST 1: Subtract Green Transform Round-trip")
    print("=" * 60)

    width, height = 8, 8
    np.random.seed(42)
    rgba_data = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
    original_bytes = rgba_data.tobytes()

    # Apply forward transform
    transformed = apply_subtract_green(original_bytes, width, height)
    # Apply inverse transform
    restored = inverse_subtract_green(transformed, width, height)

    if original_bytes == restored:
        print("  [OK] PASS -- Subtract Green perfectly reversible!")
    else:
        print("  [X] FAIL -- Data mismatch after inverse Subtract Green!")
        return False

    return True


def test_compress_decompress_synthetic():
    """Test full round-trip with a synthetic gradient image."""
    print("\n" + "=" * 60)
    print("TEST 2: Full Compress -> Decompress Round-trip (Synthetic)")
    print("=" * 60)

    # Create a 64x64 gradient test image (RGB)
    width, height = 64, 64
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = (x * 4) % 256      # Red gradient
            arr[y, x, 1] = (y * 4) % 256      # Green gradient
            arr[y, x, 2] = ((x + y) * 2) % 256  # Blue gradient
    img_original = Image.fromarray(arr, mode="RGB")

    print(f"  Original image: {width}x{height}, mode={img_original.mode}")

    # Compress
    compressor = WebPLosslessCompressor(use_subtract_green=True)
    compressed_bytes = compressor.compress(img_original)
    print(f"  Compressed size: {len(compressed_bytes):,} bytes")

    # Verify RIFF header
    assert compressed_bytes[:4] == b"RIFF", "RIFF magic mismatch"
    assert compressed_bytes[8:12] == b"WEBP", "WEBP magic mismatch"
    assert compressed_bytes[12:16] == b"VP8L", "VP8L chunk mismatch"
    print("  [OK] RIFF/WEBP/VP8L container structure valid")

    # Decompress
    decompressor = WebPLosslessDecompressor()
    img_restored = decompressor.decompress(compressed_bytes)
    print(f"  Restored image: {img_restored.width}x{img_restored.height}, mode={img_restored.mode}")

    # Verify lossless
    is_lossless, mse, psnr = calculate_image_metrics(img_original, img_restored)
    print(f"  Lossless: {is_lossless}, MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

    if is_lossless:
        print("  [OK] PASS -- 100% bit-for-bit lossless!")
    else:
        print("  [X] FAIL -- Images differ!")
        return False

    # Compression stats
    raw_size = width * height * 3
    cr = raw_size / len(compressed_bytes)
    saving = (1.0 - len(compressed_bytes) / raw_size) * 100.0
    print(f"  Raw pixel size: {raw_size:,} bytes")
    print(f"  Compression ratio: {cr:.2f}x")
    print(f"  Space saving: {saving:.2f}%")

    return True


def test_compress_decompress_file():
    """Test file-level compress and decompress."""
    print("\n" + "=" * 60)
    print("TEST 3: File-level Compress -> Decompress Round-trip")
    print("=" * 60)

    # Create test image and save as BMP
    width, height = 32, 32
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [(x * 8) % 256, (y * 8) % 256, 128]
    img_original = Image.fromarray(arr, mode="RGB")

    test_dir = os.path.join(project_root, "output", "webp_lossless_test")
    os.makedirs(test_dir, exist_ok=True)

    bmp_path = os.path.join(test_dir, "test_input.bmp")
    webp_path = os.path.join(test_dir, "test_output.webp")
    restored_path = os.path.join(test_dir, "test_restored.bmp")

    img_original.save(bmp_path)
    print(f"  Saved test image: {bmp_path}")

    # Compress
    compressor = WebPLosslessCompressor()
    stats = compressor.compress_file(bmp_path, webp_path)
    print(f"  Compressed to: {webp_path}")
    print(f"    Original:   {stats['original_size']:,} bytes")
    print(f"    Compressed: {stats['compressed_size']:,} bytes")
    print(f"    Ratio:      {stats['compression_ratio']:.2f}x")
    print(f"    Saving:     {stats['space_saving_pct']:.2f}%")
    print(f"    Time:       {stats['compression_time_sec'] * 1000:.2f} ms")

    # Decompress
    decompressor = WebPLosslessDecompressor()
    img_restored, decomp_time = decompressor.decompress_file(webp_path, restored_path)
    print(f"  Decompressed to: {restored_path}")
    print(f"    Time: {decomp_time * 1000:.2f} ms")

    # Verify
    is_lossless, mse, psnr = calculate_image_metrics(img_original, img_restored)
    print(f"  Lossless: {is_lossless}, MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

    if is_lossless:
        print("  [OK] PASS -- File round-trip is 100% lossless!")
    else:
        print("  [X] FAIL -- File round-trip failed!")
        return False

    return True


def main():
    print("\n" + "+" + "=" * 58 + "+")
    print("|   WEBP LOSSLESS -- COMPRESSION & DECOMPRESSION TESTS     |")
    print("+" + "=" * 58 + "+\n")

    results = []
    results.append(("Subtract Green Round-trip", test_subtract_green_roundtrip()))
    results.append(("Full Compress/Decompress", test_compress_decompress_synthetic()))
    results.append(("File-level Round-trip", test_compress_decompress_file()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "[OK] PASS" if passed else "[X] FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  [SUCCESS] ALL {len(results)} TESTS PASSED!")
    else:
        print(f"\n  [FAILED] SOME TESTS FAILED!")

    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
