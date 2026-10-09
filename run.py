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


DEFAULT_TOTAL_RUNS = int(os.getenv("TOTAL_RUNS", 5))
DEFAULT_IMAGES_PER_GROUP = int(os.getenv("IMAGES_PER_GROUP", 100))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
AVAILABLE_ALGORITHMS = (
    "deflate",
    "tiff_lzw",
    "webp_lossless",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DEFLATE, TIFF-LZW, and WebP Lossless on the same S3 images."
    )
    parser.add_argument(
        "--algo",
        "-a",
        choices=(*AVAILABLE_ALGORITHMS, "all"),
        default="all",
        help="Codec to benchmark, or 'all' to run every available codec.",
    )
    parser.add_argument(
        "--total-runs",
        "-r",
        type=int,
        default=DEFAULT_TOTAL_RUNS,
        help=f"Number of benchmark runs (default: {DEFAULT_TOTAL_RUNS}).",
    )
    parser.add_argument(
        "--repeat",
        "-n",
        type=int,
        default=1,
        help="Compression/decompression repetitions per image (default: 1).",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=DEFAULT_IMAGES_PER_GROUP,
        help=f"Maximum images loaded from each S3 group (default: {DEFAULT_IMAGES_PER_GROUP}).",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Reserved for future parallel mode; fair benchmark mode is sequential.",
    )
    parser.add_argument(
        "--filter",
        "-f",
        choices=["NONE", "SUB", "UP", "AVERAGE", "PAETH", "ADAPTIVE"],
        default="ADAPTIVE",
        help="Spatial prediction filter used only by DEFLATE.",
    )
    parser.add_argument(
        "--pure-python",
        action="store_true",
        help="Use the educational Python LZ77 + Huffman DEFLATE engine.",
    )
    args = parser.parse_args()

    if args.total_runs < 1:
        parser.error("--total-runs must be at least 1")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    return args


CodecFunctions = Tuple[Callable, Callable, str]


def build_codecs(args: argparse.Namespace) -> Dict[str, CodecFunctions]:
    """Build selected codecs and their compressed-file extensions."""
    selected = AVAILABLE_ALGORITHMS if args.algo == "all" else (args.algo,)
    codecs: Dict[str, CodecFunctions] = {}

    for name in selected:
        if name == "deflate":
            codec = DeflateImageCompressor(
                compression_level=9,
                filter_type=FilterType[args.filter],
                use_pure_python=args.pure_python,
            )
            codecs[name] = (codec.compress, codec.decompress, ".defimg")
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
    print(f"Total runs:        {args.total_runs}")
    print(f"Images per group:  {args.limit}")
    print(f"Timing repeats:    {args.repeat}")
    if "deflate" in codecs:
        print(f"DEFLATE filter:    {args.filter}")
        print(f"DEFLATE engine:    {'Pure Python' if args.pure_python else 'zlib'}")
    print("=" * 80)

    for run_id in range(1, args.total_runs + 1):
        if len(codecs) > 1:
            print(f"\nRun {run_id}/{args.total_runs}: ALL CODECS PER IMAGE")
            run_multi_codec_benchmark_pipeline(
                codecs=codecs,
                run_id=run_id,
                num_runs=args.repeat,
                limit_files=args.limit,
                max_workers=args.workers,
            )
            continue

        for algo_name, (compress_fn, decompress_fn, file_extension) in codecs.items():
            print(f"\nRun {run_id}/{args.total_runs}: {algo_name.upper()}")
            run_s3_benchmark_pipeline(
                compress_fn=compress_fn,
                decompress_fn=decompress_fn,
                algo_name=algo_name,
                file_extension=file_extension,
                run_id=run_id,
                num_runs=args.repeat,
                limit_files=args.limit,
                max_workers=args.workers,
            )

    runs = " ".join(str(run_id) for run_id in range(1, args.total_runs + 1))
    algo_names = " ".join(codecs)
    print("\nBenchmark complete.")
    if len(codecs) > 1:
        print(f"Generate comparison charts with:\n  python plot_reports.py --algos {algo_names} --runs {runs}")
    else:
        algo_name = next(iter(codecs))
        print(f"Generate charts with:\n  python plot_reports.py --algo {algo_name} --runs {runs}")


if __name__ == "__main__":
    main()
