"""
Test suite for WebP Lossless Compressor & Decompressor.
Tests both LibWebP (standard engine) and Pure Python (educational engine),
verifying 100% bit-for-bit lossless accuracy, RIFF container compliance,
and Canonical Huffman coding.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from PIL import Image

from algorithms.webp_lossless.compressor import (
    WebPLosslessCompressor,
    apply_subtract_green,
    inverse_subtract_green,
    benchmark_webp_lossless,
)
from algorithms.webp_lossless.decompressor import WebPLosslessDecompressor
from algorithms.common.huffman_webp import (
    huffman_webp_encode,
    huffman_webp_decode,
    build_canonical_codes,
    compute_code_lengths,
)
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

    transformed = apply_subtract_green(original_bytes, width, height)
    restored = inverse_subtract_green(transformed, width, height)

    if original_bytes == restored:
        print("  [OK] PASS -- Subtract Green perfectly reversible!")
        return True
    else:
        print("  [X] FAIL -- Data mismatch after inverse Subtract Green!")
        return False


def test_huffman_webp_roundtrip():
    """Test Canonical Huffman with LSB bitstream round-trip."""
    print("\n" + "=" * 60)
    print("TEST 2: WebP Canonical Huffman (huffman_webp) Round-trip")
    print("=" * 60)

    sample_data = b"WebP Lossless Canonical Huffman Coding with LSB Bitstream!" * 20
    encoded = huffman_webp_encode(sample_data)
    decoded = huffman_webp_decode(encoded)

    print(f"  Original size: {len(sample_data)} bytes")
    print(f"  Encoded size:  {len(encoded)} bytes")
    print(f"  Ratio:         {len(sample_data) / len(encoded):.2f}x")

    if sample_data == decoded:
        print("  [OK] PASS -- Canonical Huffman round-trip is 100% identical!")
        return True
    else:
        print("  [X] FAIL -- Canonical Huffman decoded data mismatch!")
        return False


def test_pure_python_engine():
    """Test pure Python pipeline (Subtract Green + Filter + LZ77 + Canonical Huffman)."""
    print("\n" + "=" * 60)
    print("TEST 3: Pure Python Pipeline Round-trip (Educational Engine)")
    print("=" * 60)

    width, height = 48, 48
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = (x * 5) % 256
            arr[y, x, 1] = (y * 5) % 256
            arr[y, x, 2] = ((x + y) * 3) % 256
    img_orig = Image.fromarray(arr, mode="RGB")

    comp = WebPLosslessCompressor(use_pure_python=True, use_subtract_green=True)
    compressed_bytes = comp.compress(img_orig)
    print(f"  Pure Python compressed size: {len(compressed_bytes):,} bytes")

    # Verify RIFF structure
    assert compressed_bytes[:4] == b"RIFF", "RIFF magic mismatch"
    assert compressed_bytes[8:12] == b"WEBP", "WEBP magic mismatch"
    assert compressed_bytes[12:16] == b"VP8L", "VP8L chunk mismatch"

    decomp = WebPLosslessDecompressor()
    img_rec = decomp.decompress(compressed_bytes, engine="pure_python")

    is_lossless, mse, psnr = calculate_image_metrics(img_orig, img_rec)
    print(f"  Lossless: {is_lossless}, MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

    if is_lossless:
        print("  [OK] PASS -- Pure Python round-trip is 100% bit-for-bit lossless!")
        return True
    else:
        print("  [X] FAIL -- Pure Python round-trip mismatch!")
        return False


def test_libwebp_engine():
    """Test standard LibWebP pipeline (Google libwebp via Pillow)."""
    print("\n" + "=" * 60)
    print("TEST 4: LibWebP Standard Engine Round-trip")
    print("=" * 60)

    width, height = 64, 64
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x, 0] = (x * 4) % 256
            arr[y, x, 1] = (y * 4) % 256
            arr[y, x, 2] = ((x * 2 + y) * 2) % 256
    img_orig = Image.fromarray(arr, mode="RGB")

    # Compress with default libwebp
    comp = WebPLosslessCompressor(use_pure_python=False)
    compressed_bytes = comp.compress(img_orig)
    print(f"  LibWebP compressed size: {len(compressed_bytes):,} bytes")

    # Verify that standard Pillow Image.open can read it directly!
    import io
    pillow_img = Image.open(io.BytesIO(compressed_bytes))
    print(f"  Standard Pillow Image.open format: {pillow_img.format}")
    assert pillow_img.format == "WEBP", "Format must be WEBP"

    # Decompress using WebPLosslessDecompressor
    decomp = WebPLosslessDecompressor()
    img_rec = decomp.decompress(compressed_bytes)

    is_lossless, mse, psnr = calculate_image_metrics(img_orig, img_rec)
    print(f"  Lossless: {is_lossless}, MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

    if is_lossless:
        print("  [OK] PASS -- LibWebP standard round-trip is 100% lossless!")
        return True
    else:
        print("  [X] FAIL -- LibWebP round-trip mismatch!")
        return False


def test_file_level_both_engines():
    """Test saving and loading .webp files on disk for both engines."""
    print("\n" + "=" * 60)
    print("TEST 5: File-Level Round-trip (LibWebP + Pure Python)")
    print("=" * 60)

    test_dir = os.path.join(project_root, "output", "webp_lossless_test")
    os.makedirs(test_dir, exist_ok=True)

    # 1. Create source image
    width, height = 32, 32
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [(x * 8) % 256, (y * 8) % 256, 128]
    img_orig = Image.fromarray(arr, mode="RGB")

    input_bmp = os.path.join(test_dir, "test_input.bmp")
    img_orig.save(input_bmp)

    # Sub-test A: LibWebP engine (standard .webp)
    webp_standard = os.path.join(test_dir, "test_standard.webp")
    rec_standard = os.path.join(test_dir, "test_restored_standard.bmp")

    comp_lib = WebPLosslessCompressor(use_pure_python=False)
    stats_lib = comp_lib.compress_file(input_bmp, webp_standard)
    print(f"  [LibWebP] Compressed: {stats_lib['compressed_size']:,} bytes, CR: {stats_lib['compression_ratio']:.2f}x")

    decomp = WebPLosslessDecompressor()
    img_rec_lib, _ = decomp.decompress_file(webp_standard, rec_standard)
    is_lossless_lib, _, _ = calculate_image_metrics(img_orig, img_rec_lib)
    assert is_lossless_lib, "LibWebP file round-trip mismatch"
    print("  [LibWebP] File round-trip PASS (100% lossless)")

    # Sub-test B: Pure Python engine
    webp_py = os.path.join(test_dir, "test_pure_python.webp")
    rec_py = os.path.join(test_dir, "test_restored_py.bmp")

    comp_py = WebPLosslessCompressor(use_pure_python=True)
    stats_py = comp_py.compress_file(input_bmp, webp_py)
    print(f"  [PurePython] Compressed: {stats_py['compressed_size']:,} bytes, CR: {stats_py['compression_ratio']:.2f}x")

    img_rec_py, _ = decomp.decompress_file(webp_py, rec_py)
    is_lossless_py, _, _ = calculate_image_metrics(img_orig, img_rec_py)
    assert is_lossless_py, "Pure Python file round-trip mismatch"
    print("  [PurePython] File round-trip PASS (100% lossless)")

    print("  [OK] PASS -- Both file engines verified successfully!")
    return True


def test_benchmark_comparison():
    """Test benchmark_webp_lossless multi-engine evaluation."""
    print("\n" + "=" * 60)
    print("TEST 6: Comprehensive Benchmark Evaluation")
    print("=" * 60)

    width, height = 32, 32
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [x * 7 % 256, y * 7 % 256, (x + y) * 4 % 256]
    img = Image.fromarray(arr, mode="RGB")

    results = benchmark_webp_lossless(img)
    print(f"  Image dimensions:        {results['dimensions']}")
    print(f"  Raw pixel bytes:         {results['raw_pixel_bytes']:,} bytes")
    print(f"  LibWebP size:            {results['libwebp_bytes']:,} bytes (CR: {results['libwebp_cr']:.2f}x)")
    print(f"  PurePython (with SG):    {results['py_with_sg_bytes']:,} bytes (CR: {results['py_with_sg_cr']:.2f}x)")
    print(f"  PurePython (no SG):      {results['py_no_sg_bytes']:,} bytes (CR: {results['py_no_sg_cr']:.2f}x)")
    print(f"  LibWebP lossless:        {results['libwebp_lossless']}")
    print(f"  PurePython lossless:     {results['py_with_sg_lossless']}")

    if results['libwebp_lossless'] and results['py_with_sg_lossless']:
        print("  [OK] PASS -- Benchmark completed successfully!")
        return True
    return False


def main():
    print("\n" + "+" + "=" * 58 + "+")
    print("|   WEBP LOSSLESS -- COMPREHENSIVE TEST SUITE              |")
    print("|   Dual Engine: Google LibWebP + Pure Python (VP8L)       |")
    print("+" + "=" * 58 + "+\n")

    tests = [
        ("Subtract Green Transform Round-trip", test_subtract_green_roundtrip),
        ("WebP Canonical Huffman Round-trip", test_huffman_webp_roundtrip),
        ("Pure Python Pipeline Round-trip", test_pure_python_engine),
        ("LibWebP Standard Engine Round-trip", test_libwebp_engine),
        ("File-level Both Engines Round-trip", test_file_level_both_engines),
        ("Comprehensive Benchmark Evaluation", test_benchmark_comparison),
    ]

    all_pass = True
    for name, func in tests:
        passed = func()
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    if all_pass:
        print(f"  [SUCCESS] ALL {len(tests)} TESTS PASSED PERFECTLY!")
    else:
        print("  [FAILED] SOME TESTS FAILED!")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
