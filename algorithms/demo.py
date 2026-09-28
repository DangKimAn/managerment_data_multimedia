"""
Demo and Benchmark Program for DEFLATE Image Compression.
Compares compression ratio (CR) across 4 multimedia data types:
1. Natural Complex (Nature / Landscapes)
2. Flat Graphics (Flat Vector / UI / Icons)
3. Textures Noisy (Surface Textures / Grain)
4. Documents Text (Scanned Documents / Books / Text)
Written in simple A2/B1 English.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

try:
    from .deflate_compressor import (
        DeflateImageCompressor,
        FilterType,
        benchmark_image_compression
    )
except ImportError:
    from deflate_compressor import (
        DeflateImageCompressor,
        FilterType,
        benchmark_image_compression
    )


def generate_synthetic_samples() -> dict:
    """Generate 4 sample images representing the 4 dataset groups for testing"""
    w, h = 300, 300
    samples = {}

    # 1. Natural Complex: Smooth color gradients and organic curves
    arr_nat = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            r = int((x / w) * 200 + (y / h) * 55)
            g = int(np.sin(x / 30.0) * 40 + 150)
            b = int(255 - (y / h) * 180)
            arr_nat[y, x] = [r, g, b]
    samples["1_Natural_Complex"] = Image.fromarray(arr_nat)

    # 2. Flat Graphics: Solid color areas, clean geometric shapes, sharp borders
    img_flat = Image.new("RGB", (w, h), color=(240, 240, 245))
    draw_flat = ImageDraw.Draw(img_flat)
    draw_flat.rectangle([30, 30, 140, 140], fill=(230, 50, 50), outline=(20, 20, 20), width=3)
    draw_flat.ellipse([160, 50, 270, 160], fill=(50, 120, 220), outline=(20, 20, 20), width=3)
    draw_flat.polygon([(60, 270), (150, 170), (240, 270)], fill=(40, 180, 80), outline=(20, 20, 20))
    samples["2_Flat_Graphics"] = img_flat

    # 3. Textures Noisy: Random high-frequency noise and grainy texture
    noise = np.random.randint(60, 200, (h, w), dtype=np.uint8)
    arr_tex = np.stack([
        noise,
        noise + np.random.randint(-15, 15, (h, w), dtype=np.int16).clip(0, 255).astype(np.uint8),
        noise // 2
    ], axis=-1)
    samples["3_Textures_Noisy"] = Image.fromarray(arr_tex)

    # 4. Documents Text: White background, black lines, high contrast
    img_doc = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw_doc = ImageDraw.Draw(img_doc)
    for i in range(12):
        y_pos = 25 + i * 22
        # Simulate text lines
        draw_doc.line([(30, y_pos), (w - 30, y_pos)], fill=(20, 20, 20), width=4)
        draw_doc.rectangle([30, y_pos + 7, w - 80, y_pos + 11], fill=(80, 80, 80))
    samples["4_Documents_Text"] = img_doc

    return samples


def run_benchmark_suite(images_dict: dict):
    """Run benchmark tests and print a clean comparison table"""
    print("\n" + "=" * 95)
    print(f"{'DEFLATE IMAGE COMPRESSION BENCHMARK REPORT':^95}")
    print("=" * 95)
    print(f"{'Image Type':<20} | {'Raw (bytes)':<11} | {'Direct (CR)':<12} | {'Filtered (CR)':<14} | {'Space Saved':<13} | {'BPP':<6} | {'Lossless':<8}")
    print("-" * 95)

    results = {}
    for name, img in images_dict.items():
        metrics = benchmark_image_compression(img, compression_level=9)
        results[name] = metrics

        orig_b = metrics["raw_pixel_bytes"]
        d_cr = metrics["direct_cr"]
        a_cr = metrics["adaptive_cr"]
        sav = metrics["space_saving_pct"]
        bpp = metrics["bits_per_pixel"]
        lossless_str = "100% OK" if metrics["is_lossless"] else "FAIL"

        print(f"{name:<20} | {orig_b:<11,d} | {d_cr:>5.2f}x       | {a_cr:>6.2f}x        | {sav:>6.2f}%       | {bpp:>5.2f} | {lossless_str:<8}")

    print("=" * 95)
    print("\nCOMPRESSION CHARACTERISTICS ANALYSIS:")
    print("  1. Flat Graphics & Documents: Reach very high compression ratios (100x - 300x+)")
    print("     because of large solid color areas and high repetition for LZ77.")
    print("  2. Natural Complex: Spatial prediction filters turn smooth color gradients into zeros,")
    print("     dramatically boosting the compression ratio compared to direct DEFLATE.")
    print("  3. Textures Noisy: Has the lowest compression ratio because random noise has high entropy,")
    print("     leaving very few repeating patterns for LZ77 to find.\n")


def main():
    parser = argparse.ArgumentParser(description="Lossless DEFLATE image compression tool")
    parser.add_argument("--image", "-i", type=str, help="Input image file path to compress")
    parser.add_argument("--output", "-o", type=str, help="Output compressed file path (.defimg)")
    parser.add_argument("--decompress", "-d", type=str, help="Input .defimg file path to decompress")
    parser.add_argument("--restore", "-r", type=str, help="Output restored image path (e.g. restored.bmp)")
    parser.add_argument("--level", "-l", type=int, default=9, choices=range(1, 10), help="Compression level (1-9)")
    args = parser.parse_args()

    compressor = DeflateImageCompressor(compression_level=args.level)

    # 1. Decompression mode
    if args.decompress:
        restore_path = args.restore or "decompressed_output.bmp"
        print(f"[*] Decompressing: {args.decompress} -> {restore_path}")
        img, decomp_time = compressor.decompress_file(args.decompress, restore_path)
        print(f"[+] Decompression successful in {decomp_time * 1000:.2f} ms!")
        print(f"[+] Image dimensions: {img.width}x{img.height}, Color mode: {img.mode}")
        return

    # 2. Single image compression mode
    if args.image:
        out_path = args.output or f"{os.path.splitext(args.image)[0]}.defimg"
        print(f"[*] Compressing file: {args.image} -> {out_path}")
        stats = compressor.compress_file(args.image, out_path)
        print(f"[+] Compression successful in {stats['compression_time_sec'] * 1000:.2f} ms!")
        print(f"  - Original size:       {stats['original_size']:,} bytes")
        print(f"  - Compressed size:     {stats['compressed_size']:,} bytes")
        print(f"  - Compression Ratio:   {stats['compression_ratio']:.2f}x")
        print(f"  - Space Saved:         {stats['space_saving_pct']:.2f}%")
        return

    # 3. Default: Run benchmark suite on 4 sample images
    print("[*] No input file provided. Running benchmark test on 4 sample image types...")
    samples = generate_synthetic_samples()
    run_benchmark_suite(samples)


if __name__ == '__main__':
    main()
