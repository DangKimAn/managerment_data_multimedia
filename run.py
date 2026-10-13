"""Run image-compression benchmarks for one codec or all available codecs."""

import argparse
import os
import sys
from typing import Callable, Dict, Tuple


def _load_env() -> None:
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_file):
        return

    with open(env_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env()

from algorithms import (  # noqa: E402
    DeflateImageCompressor,
    FilterType,
    TiffLzwCompressor,
    WebPLosslessCompressor,
    WebPLosslessDecompressor,
)
from s3_benchmark import (  # noqa: E402
    run_multi_codec_benchmark_pipeline,
    run_s3_benchmark_pipeline,
)


DEFAULT_IMAGES_PER_GROUP = int(os.getenv("IMAGES_PER_GROUP", 100))
DEFAULT_TIMING_REPEATS   = int(os.getenv("TIMING_REPEATS", 5))   # Số lần đo timing mỗi ảnh
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
AVAILABLE_ALGORITHMS = (
    "deflate",
    "tiff_lzw",
    "webp_lossless",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DEFLATE, TIFF-LZW, và WebP Lossless trên cùng ảnh S3."
    )
    parser.add_argument(
        "--algo", "-a",
        choices=(*AVAILABLE_ALGORITHMS, "all"),
        default="all",
        help="Codec cần benchmark, hoặc 'all' để chạy tất cả.",
    )
    parser.add_argument(
        "--repeat", "-n",
        type=int,
        default=DEFAULT_TIMING_REPEATS,
        help=f"Số lần nén/giải nén mỗi ảnh để lấy thời gian trung bình (default: {DEFAULT_TIMING_REPEATS}).",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=DEFAULT_IMAGES_PER_GROUP,
        help=f"Số ảnh tối đa tải từ mỗi nhóm S3 (default: {DEFAULT_IMAGES_PER_GROUP}).",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Dự phòng cho chế độ song song; hiện tại benchmark chạy tuần tự.",
    )
    parser.add_argument(
        "--filter", "-f",
        choices=["NONE", "SUB", "UP", "AVERAGE", "PAETH", "ADAPTIVE"],
        default="ADAPTIVE",
        help="Spatial filter (chỉ dùng khi --no-pillow).",
    )
    parser.add_argument(
        "--no-pillow",
        action="store_true",
        help="Dùng pipeline học thuật (spatial filter + zlib/Python) thay vì Pillow PNG.",
    )
    parser.add_argument(
        "--pure-python",
        action="store_true",
        help="Dùng LZ77 + Huffman Python thuần (chỉ tác dụng khi --no-pillow).",
    )

    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


CodecFunctions = Tuple[Callable, Callable, str]


def build_codecs(args: argparse.Namespace) -> Dict[str, CodecFunctions]:
    """Khởi tạo các codec được chọn."""
    selected = AVAILABLE_ALGORITHMS if args.algo == "all" else (args.algo,)
    codecs: Dict[str, CodecFunctions] = {}

    for name in selected:
        if name == "deflate":
            use_pillow = not args.no_pillow
            file_ext = ".png" if use_pillow else ".defimg"
            codec = DeflateImageCompressor(
                compression_level=9,
                filter_type=FilterType[args.filter],
                use_pure_python=args.pure_python,
                use_pillow=use_pillow,
            )
            codecs[name] = (codec.compress, codec.decompress, file_ext)
        elif name == "tiff_lzw":
            codec = TiffLzwCompressor()
            codecs[name] = (codec.compress, codec.decompress, ".tiff")
        elif name == "webp_lossless":
            compressor = WebPLosslessCompressor()
            decompressor = WebPLosslessDecompressor()
            codecs[name] = (compressor.compress, decompressor.decompress, ".webp")

    return codecs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    codecs = build_codecs(args)

    print("=" * 80)
    print("MULTIMEDIA LOSSLESS IMAGE COMPRESSION BENCHMARK")
    print("=" * 80)
    print(f"Environment:       {ENVIRONMENT.upper()}")
    print(f"Algorithms:        {', '.join(codecs)}")
    print(f"Images per group:  {args.limit}")
    print(f"Timing repeats:    {args.repeat}  (mỗi ảnh nén/giải nén {args.repeat} lần → lấy trung bình)")
    if "deflate" in codecs:
        use_pillow = not args.no_pillow
        if use_pillow:
            print(f"DEFLATE engine:    Pillow PNG / libpng + zlib (benchmark chính)")
        else:
            engine = "LZ77+Huffman Python" if args.pure_python else "zlib"
            print(f"DEFLATE engine:    {engine} + spatial filter [{args.filter}] (học thuật)")
    print("=" * 80)

    if len(codecs) > 1:
        print("\nBenchmark: TẤT CẢ CODEC TRÊN TỪNG ẢNH")
        run_multi_codec_benchmark_pipeline(
            codecs=codecs,
            num_runs=args.repeat,
            limit_files=args.limit,
            max_workers=args.workers,
        )
    else:
        algo_name, (compress_fn, decompress_fn, file_extension) = next(iter(codecs.items()))
        print(f"\nBenchmark: {algo_name.upper()}")
        run_s3_benchmark_pipeline(
            compress_fn=compress_fn,
            decompress_fn=decompress_fn,
            algo_name=algo_name,
            file_extension=file_extension,
            num_runs=args.repeat,
            limit_files=args.limit,
            max_workers=args.workers,
        )

    algo_names = " ".join(codecs)
    print("\nBenchmark hoàn tất.")
    if len(codecs) > 1:
        print(f"Xem biểu đồ:\n  python plot_reports.py --algos {algo_names}")
    else:
        print(f"Xem biểu đồ:\n  python plot_reports.py --algo {algo_names}")


if __name__ == "__main__":
    main()
