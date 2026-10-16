#!/usr/bin/env python3
"""
Pipeline Tổng Hợp Tự Động (Master Pipeline):
  1. Crawl Dataset ảnh theo nhóm từ Bing và tải trực tiếp lên AWS S3 (crawl_dataset.py).
  2. Benchmark nén & giải nén các thuật toán (DEFLATE, TIFF-LZW, WebP) từ S3 (run.py).
  3. Trực quan hóa dữ liệu, vẽ biểu đồ phân phối & so sánh tổng hợp (plot_reports.py).
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import timedelta


def load_env() -> None:
    """Đọc file .env nếu có."""
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def run_command(cmd: list[str], stage_title: str) -> None:
    """Chạy command dòng lệnh bằng sys.executable và bắt lỗi."""
    print("\n" + "=" * 80)
    print(f"  >>> BẮT ĐẦU: {stage_title}")
    print(f"  >>> Lệnh: {' '.join(cmd)}")
    print("=" * 80 + "\n")
    start_time = time.time()
    
    result = subprocess.run(cmd)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n[LỖI] {stage_title} thất bại (mã thoát {result.returncode})!")
        sys.exit(result.returncode)
        
    print(f"\n[THÀNH CÔNG] {stage_title} hoàn thành trong {timedelta(seconds=int(duration))}.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline tổng hợp: Crawl S3 -> Benchmark Codecs -> Vẽ biểu đồ so sánh."
    )
    
    # Điều khiển các bước thực thi (Skip flags)
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Bỏ qua bước Crawl data lên S3 (nếu S3 đã có sẵn ảnh).",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Bỏ qua bước chạy Benchmark nén/giải nén (nếu đã có CSV kết quả).",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Bỏ qua bước vẽ biểu đồ.",
    )
    
    # Cấu hình bước Crawl
    parser.add_argument(
        "--generate-noise",
        action="store_true",
        help="Sinh 100 ảnh nhiễu toán học chuẩn bằng NumPy (U(0, 255)) đẩy lên S3 cho nhóm 3.",
    )
    parser.add_argument(
        "--crawl-group",
        type=str,
        default="all",
        choices=["all", "1_Natural_Complex", "2_Flat_Graphics", "3_Textures_Noisy", "4_Documents_Text"],
        help="Nhóm cần crawl (mặc định 'all' cho cả 4 nhóm).",
    )
    parser.add_argument(
        "--target-per-group",
        type=int,
        default=int(os.getenv("TARGET_IMAGES_PER_GROUP", 100)),
        help="Số lượng ảnh mục tiêu cho mỗi nhóm khi crawl (mặc định 100).",
    )

    # Cấu hình bước Benchmark
    parser.add_argument(
        "--algo", "-a",
        default="all",
        choices=["all", "deflate", "tiff_lzw", "webp_lossless"],
        help="Thuật toán cần benchmark ('all' hoặc chọn lẻ).",
    )
    parser.add_argument(
        "--repeat", "-n",
        type=int,
        default=int(os.getenv("TIMING_REPEATS", 5)),
        help="Số lần đo timing nén/giải nén mỗi ảnh (mặc định 5).",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=int(os.getenv("IMAGES_PER_GROUP", 100)),
        help="Số ảnh tối đa tải từ mỗi nhóm S3 để benchmark (mặc định 100).",
    )
    parser.add_argument(
        "--no-pillow",
        action="store_true",
        help="Benchmark DEFLATE bằng pipeline học thuật (spatial filter + zlib) thay vì Pillow.",
    )

    # Cấu hình bước Plot
    parser.add_argument(
        "--input-dir",
        type=str,
        default="output_not_commit",
        help="Thư mục chứa kết quả benchmark (default: output_not_commit).",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="output",
        help="Thư mục xuất biểu đồ (default: output).",
    )

    return parser.parse_args()


def stage_crawl(args: argparse.Namespace) -> None:
    """Giai đoạn 1: Crawl dữ liệu ảnh từ Bing đẩy trực tiếp lên S3 (hoặc sinh True Noise)."""
    if args.generate_noise or args.crawl_group == "3_Textures_Noisy":
        cmd = [sys.executable, "crawl_dataset.py", "--noise-only"]
    elif args.crawl_group == "all":
        cmd = [sys.executable, "crawl_dataset.py"]
    else:
        # Nếu muốn crawl riêng 1 nhóm thông thường
        env_cmd = [
            sys.executable,
            "-c",
            f"from crawl_dataset import crawl_images_for_group_s3, DATASET_PLAN; "
            f"crawl_images_for_group_s3('{args.crawl_group}', DATASET_PLAN['{args.crawl_group}'], {args.target_per_group})"
        ]
        cmd = env_cmd
    run_command(cmd, "GIAI ĐOẠN 1: CRAWL ẢNH / SINH NHIỄU TOÁN HỌC VÀ UPLOAD S3")


def stage_benchmark(args: argparse.Namespace) -> None:
    """Giai đoạn 2: Benchmark nén/giải nén từ S3 qua run.py."""
    cmd = [
        sys.executable,
        "run.py",
        "--algo", args.algo,
        "--repeat", str(args.repeat),
        "--limit", str(args.limit),
    ]
    if args.no_pillow:
        cmd.append("--no-pillow")
    
    run_command(cmd, f"GIAI ĐOẠN 2: BENCHMARK NÉN/GIẢI NÉN (Codec: {args.algo}, Repeat: {args.repeat})")


def stage_plot(args: argparse.Namespace) -> None:
    """Giai đoạn 3: Vẽ biểu đồ thống kê và so sánh qua plot_reports.py."""
    algos_to_plot = ["deflate", "tiff_lzw", "webp_lossless"] if args.algo == "all" else [args.algo]

    # 1. Vẽ biểu đồ chi tiết từng thuật toán
    for algo in algos_to_plot:
        cmd_single = [
            sys.executable,
            "plot_reports.py",
            "--algo", algo,
            "--input-dir", args.input_dir,
            "--plot-dir", args.plot_dir,
        ]
        run_command(cmd_single, f"GIAI ĐOẠN 3.1: VẼ BIỂU ĐỒ CHI TIẾT CODEC [{algo.upper()}]")

    # 2. Nếu benchmark cả 3 codec (hoặc nhiều hơn 1), vẽ biểu đồ so sánh tổng hợp
    if len(algos_to_plot) > 1:
        cmd_compare = [
            sys.executable,
            "plot_reports.py",
            "--algos", *algos_to_plot,
            "--input-dir", args.input_dir,
            "--plot-dir", args.plot_dir,
        ]
        run_command(cmd_compare, "GIAI ĐOẠN 3.2: VẼ BIỂU ĐỒ SO SÁNH TỔNG HỢP (COMPARISON CHARTS)")


def main() -> None:
    load_env()
    args = parse_args()

    total_start = time.time()
    print("=" * 80)
    print("           TỔNG HỢP TOÀN BỘ QUY TRÌNH (MASTER PIPELINE)")
    print("=" * 80)
    print(f"Python:       {sys.executable}")
    print(f"Crawl S3:     {'BỎ QUA (--skip-crawl)' if args.skip_crawl else f'BẬT (Nhóm: {args.crawl_group}, Target: {args.target_per_group})'}")
    print(f"Benchmark:    {'BỎ QUA (--skip-benchmark)' if args.skip_benchmark else f'BẬT (Algo: {args.algo}, Repeat: {args.repeat}, Limit: {args.limit})'}")
    print(f"Vẽ biểu đồ:   {'BỎ QUA (--skip-plot)' if args.skip_plot else f'BẬT (Lưu tại: {args.plot_dir}/)'}")
    print("=" * 80)

    # 1. Bước Crawl
    if not args.skip_crawl:
        stage_crawl(args)
    else:
        print("\n[BỎ QUA] Giai đoạn 1 (Crawl data) theo yêu cầu.")

    # 2. Bước Benchmark
    if not args.skip_benchmark:
        stage_benchmark(args)
    else:
        print("\n[BỎ QUA] Giai đoạn 2 (Benchmark nén/giải nén) theo yêu cầu.")

    # 3. Bước Vẽ biểu đồ
    if not args.skip_plot:
        stage_plot(args)
    else:
        print("\n[BỎ QUA] Giai đoạn 3 (Vẽ biểu đồ) theo yêu cầu.")

    total_duration = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"  HOÀN TẤT TOÀN BỘ PIPELINE TRONG {timedelta(seconds=int(total_duration))}!")
    print(f"  - Báo cáo dữ liệu:   {args.input_dir}/")
    print(f"  - Biểu đồ thống kê:  {args.plot_dir}/")
    print(f"  - Biểu đồ Entropy vs CR (RQ1): {args.plot_dir}/comparison/charts/6_entropy_vs_compression_ratio.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
