"""
Main Entry Point Script for Multimedia Image Compression & Decompression Benchmarking.
Fetches ALL files from S3, executes parallel benchmarks across N total runs, and generates reports.

Cấu hình mặc định đọc từ file .env:
    ENVIRONMENT         - Môi trường chạy (development / production)
    TOTAL_RUNS          - Tổng số lần chạy (mặc định 5)
    IMAGES_PER_GROUP    - Số ảnh mỗi nhóm S3 (mặc định 100)

Usage:
    # Dùng cấu hình từ .env (khuyến nghị):
    python run.py --algo deflate

    # Override từng tham số nếu cần:
    python run.py --algo deflate --total-runs 3 --limit 50
"""

import argparse
import os
import sys

# ==============================================================================
# ĐỌC .env TRƯỚC KHI PARSE ARGS
# ==============================================================================
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_file):
    with open(_env_file, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

# Đọc giá trị mặc định từ .env
_DEFAULT_TOTAL_RUNS = int(os.getenv("TOTAL_RUNS", 5))
_DEFAULT_IMAGES_PER_GROUP = int(os.getenv("IMAGES_PER_GROUP", 100))
_ENV = os.getenv("ENVIRONMENT", "development")

from s3_benchmark import run_s3_benchmark_pipeline
from algorithms import DeflateImageCompressor, FilterType


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run S3 Image Compression & Decompression Benchmark Pipeline"
    )
    parser.add_argument(
        "--algo", "-a",
        type=str,
        default="deflate",
        help="Algorithm identifier name (default: 'deflate')"
    )
    parser.add_argument(
        "--total-runs", "-r",
        type=int,
        default=_DEFAULT_TOTAL_RUNS,
        help=f"Tổng số lần chạy benchmark (default từ .env TOTAL_RUNS={_DEFAULT_TOTAL_RUNS})"
    )
    parser.add_argument(
        "--repeat", "-n",
        type=int,
        default=1,
        help="Number of times to repeat compress/decompress per file for timing measurement (default: 1)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=_DEFAULT_IMAGES_PER_GROUP,
        help=f"Số ảnh mỗi nhóm lấy từ S3 (default từ .env IMAGES_PER_GROUP={_DEFAULT_IMAGES_PER_GROUP})"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=10,
        help="Maximum parallel worker threads (default: 10)"
    )
    parser.add_argument(
        "--filter", "-f",
        type=str,
        default="ADAPTIVE",
        choices=["NONE", "SUB", "UP", "AVERAGE", "PAETH", "ADAPTIVE"],
        help="Spatial prediction filter type (default: 'ADAPTIVE')"
    )
    parser.add_argument(
        "--pure-python",
        action="store_true",
        help="Use pure Python LZ77 + Huffman pipeline instead of fast C-accelerated DEFLATE"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Map string filter name to FilterType enum
    filter_enum = FilterType[args.filter]

    print(f"================================================================================")
    print(f" KHỞI CHẠY BENCHMARK NÉN & GIẢI NÉN ẢNH (MULTIMEDIA COMPRESSION PIPELINE)")
    print(f"================================================================================")
    print(f" - Môi trường:            {_ENV.upper()}")
    print(f" - Tên thuật toán:        {args.algo}")
    print(f" - Tổng số lần chạy:      {args.total_runs}  (đọc từ .env TOTAL_RUNS)")
    print(f" - Số ảnh mỗi nhóm:       {args.limit}  (đọc từ .env IMAGES_PER_GROUP)")
    print(f" - Số luồng song song:    {args.workers}")
    print(f" - Lặp thời gian/file:    {args.repeat} lần")
    print(f" - Bộ lọc không gian:     {args.filter}")
    print(f" - Engine:                {'Pure Python (LZ77+Huffman)' if args.pure_python else 'C-Accelerated DEFLATE'}")
    print(f"================================================================================")

    # Initialize compressor once — reused across all runs
    compressor = DeflateImageCompressor(
        compression_level=9,
        filter_type=filter_enum,
        use_pure_python=args.pure_python
    )

    # Loop over total_runs — each run fetches ALL S3 files and saves to its own run_id folder
    for run_id in range(1, args.total_runs + 1):
        print(f"\n{'='*80}")
        print(f" ▶ BẮT ĐẦU LẦN CHẠY {run_id}/{args.total_runs}")
        print(f"{'='*80}")

        run_s3_benchmark_pipeline(
            compress_fn=compressor.compress,
            decompress_fn=compressor.decompress,
            algo_name=args.algo,
            run_id=run_id,
            num_runs=args.repeat,
            limit_files=args.limit,
            max_workers=args.workers
        )

        print(f" ✓ HOÀN THÀNH LẦN CHẠY {run_id}/{args.total_runs} — Kết quả: output_not_commit/{args.algo}/{run_id}/")

    print(f"\n{'='*80}")
    print(f" 🏁 HOÀN THÀNH TẤT CẢ {args.total_runs} LẦN CHẠY CHO THUẬT TOÁN [{args.algo.upper()}]")
    print(f" Để tạo báo cáo biểu đồ thống kê, chạy tiếp:")
    runs_str = " ".join(str(r) for r in range(1, args.total_runs + 1))
    print(f"   python plot_reports.py --algo {args.algo} --runs {runs_str}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
