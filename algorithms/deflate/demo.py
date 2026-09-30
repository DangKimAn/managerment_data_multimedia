"""
Demo and Benchmark Program for DEFLATE Image Compression.
Lấy 1 ảnh thật từ mỗi nhóm trên S3 và chạy benchmark so sánh:
  1. Natural Complex   (Nature / Landscapes)
  2. Flat Graphics     (Flat Vector / UI / Icons)
  3. Textures Noisy    (Surface Textures / Grain)
  4. Documents Text    (Scanned Documents / Books / Text)
"""

import argparse
import os
import sys

# Đọc .env
_env_file = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".env"
)
if os.path.exists(_env_file):
    with open(_env_file, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

S3_BUCKET_NAME     = os.getenv("S3_BUCKET_NAME", "YOUR_BUCKET_NAME")
S3_BASE_PREFIX     = os.getenv("S3_BASE_PREFIX", "Project_Data")
AWS_REGION         = os.getenv("AWS_REGION", "ap-southeast-1")
AWS_ACCESS_KEY_ID  = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
S3_ENDPOINT_URL    = os.getenv("S3_ENDPOINT_URL", None)

# Tên 4 nhóm trên S3
S3_GROUPS = [
    "1_Natural_Complex",
    "2_Flat_Graphics",
    "3_Textures_Noisy",
    "4_Documents_Text",
]

try:
    from .compressor import (
        DeflateImageCompressor,
        benchmark_image_compression
    )
except ImportError:
    from compressor import (
        DeflateImageCompressor,
        benchmark_image_compression
    )

# Import S3Storage từ root project
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)
from s3_storage import S3Storage


# ==============================================================================
# LẤY ẢNH THẬT TỪ S3 — 1 ẢNH/NHÓM
# ==============================================================================
def fetch_one_per_group() -> dict:
    """
    Kết nối S3, lấy 1 ảnh đầu tiên từ mỗi trong 4 nhóm.

    Returns:
        Dict[group_name, PIL.Image]  — ví dụ {"1_Natural_Complex": <Image>, ...}
    """
    samples = {}

    for group in S3_GROUPS:
        prefix = f"{S3_BASE_PREFIX}/{group}".strip("/")
        print(f"[S3] Đang lấy ảnh từ nhóm: {group} (s3://{S3_BUCKET_NAME}/{prefix}/)")

        try:
            storage = S3Storage(
                bucket_name=S3_BUCKET_NAME,
                prefix=prefix,
                endpoint_url=S3_ENDPOINT_URL,
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            )
            storage.sync_existing()

            # Lấy key đầu tiên trong nhóm (sort để ổn định)
            keys = sorted(storage._existing_keys)
            if not keys:
                print(f"  [CẢNH BÁO] Không tìm thấy ảnh nào trong nhóm '{group}', bỏ qua.")
                continue

            first_key = keys[0]
            img_obj, _, _ = storage.load_data(first_key)
            if img_obj is None:
                print(f"  [CẢNH BÁO] Không tải được ảnh '{first_key}', bỏ qua.")
                continue

            img = img_obj.convert("RGB")
            samples[group] = img
            print(f"  ✓ Tải thành công: {os.path.basename(first_key)} ({img.width}x{img.height})")

        except Exception as e:
            print(f"  [LỖI S3] Nhóm '{group}': {e}")

    return samples


# ==============================================================================
# BẢNG KẾT QUẢ BENCHMARK
# ==============================================================================
def run_benchmark_suite(images_dict: dict):
    """Chạy benchmark và in bảng so sánh kết quả nén"""
    if not images_dict:
        print("[LỖI] Không có ảnh nào để benchmark!")
        return

    print("\n" + "=" * 95)
    print(f"{'DEFLATE IMAGE COMPRESSION BENCHMARK REPORT':^95}")
    print("=" * 95)
    print(
        f"{'Image Type':<22} | {'Kích thước':<10} | {'Raw (bytes)':<11} | "
        f"{'Direct CR':<10} | {'Filtered CR':<12} | {'Tiết kiệm':<10} | "
        f"{'BPP':<6} | {'Lossless'}"
    )
    print("-" * 95)

    for name, img in images_dict.items():
        try:
            metrics = benchmark_image_compression(img, compression_level=9)

            orig_b  = metrics["raw_pixel_bytes"]
            d_cr    = metrics["direct_cr"]
            a_cr    = metrics["adaptive_cr"]
            sav     = metrics["space_saving_pct"]
            bpp     = metrics["bits_per_pixel"]
            lossless_str = "✓ OK" if metrics["is_lossless"] else "✗ FAIL"
            size_str = f"{img.width}x{img.height}"

            print(
                f"{name:<22} | {size_str:<10} | {orig_b:<11,d} | "
                f"{d_cr:>6.2f}x     | {a_cr:>7.2f}x      | "
                f"{sav:>7.2f}%    | {bpp:>5.2f} | {lossless_str}"
            )
        except Exception as e:
            print(f"{name:<22} | LỖI: {e}")

    print("=" * 95)
    print("\nPHÂN TÍCH ĐẶC ĐIỂM NÉN:")
    print("  1. Flat Graphics & Documents: CR rất cao (100x-300x+) nhờ vùng màu phẳng lặp lại.")
    print("  2. Natural Complex: Bộ lọc không gian biến gradient mượt → gần 0, tăng CR đáng kể.")
    print("  3. Textures Noisy: CR thấp nhất do nhiễu ngẫu nhiên có entropy cao.\n")


# ==============================================================================
# MAIN — CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="DEFLATE Image Compression — Demo & Benchmark với ảnh thật từ S3"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Đường dẫn file ảnh cục bộ để nén (bỏ qua S3 nếu truyền vào)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Đường dẫn file output nén (.defimg)"
    )
    parser.add_argument(
        "--decompress", "-d",
        type=str,
        help="Đường dẫn file .defimg để giải nén"
    )
    parser.add_argument(
        "--restore", "-r",
        type=str,
        help="Đường dẫn file ảnh đầu ra sau giải nén (vd: restored.bmp)"
    )
    parser.add_argument(
        "--level", "-l",
        type=int,
        default=9,
        choices=range(1, 10),
        help="Mức nén DEFLATE từ 1 (nhanh) đến 9 (tối đa), mặc định: 9"
    )
    args = parser.parse_args()

    compressor = DeflateImageCompressor(compression_level=args.level)

    # Chế độ 1: Giải nén file .defimg
    if args.decompress:
        restore_path = args.restore or "decompressed_output.bmp"
        print(f"[*] Giải nén: {args.decompress} → {restore_path}")
        img, decomp_time = compressor.decompress_file(args.decompress, restore_path)
        print(f"[+] Giải nén thành công trong {decomp_time * 1000:.2f} ms!")
        print(f"[+] Kích thước ảnh: {img.width}x{img.height}, Mode: {img.mode}")
        return

    # Chế độ 2: Nén 1 ảnh cục bộ
    if args.image:
        out_path = args.output or f"{os.path.splitext(args.image)[0]}.defimg"
        print(f"[*] Nén file: {args.image} → {out_path}")
        stats = compressor.compress_file(args.image, out_path)
        print(f"[+] Nén thành công trong {stats['compression_time_sec'] * 1000:.2f} ms!")
        print(f"  - Kích thước gốc:    {stats['original_size']:,} bytes")
        print(f"  - Kích thước nén:    {stats['compressed_size']:,} bytes")
        print(f"  - Compression Ratio: {stats['compression_ratio']:.2f}x")
        print(f"  - Tiết kiệm:         {stats['space_saving_pct']:.2f}%")
        return

    # Chế độ 3 (mặc định): Lấy ảnh thật từ S3 và chạy benchmark
    print("[*] Lấy 1 ảnh thật từ mỗi nhóm trên S3 để benchmark...")
    if S3_BUCKET_NAME == "YOUR_BUCKET_NAME":
        print("[LỖI] Chưa cấu hình S3. Vui lòng kiểm tra file .env.")
        sys.exit(1)

    samples = fetch_one_per_group()
    run_benchmark_suite(samples)


if __name__ == "__main__":
    main()
