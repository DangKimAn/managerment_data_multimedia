"""
S3 Image Fetching, Benchmark & Report Generation Module.
Xử lý từng ảnh một (streaming) để tránh tràn RAM khi số lượng file lớn.

Author: Antigravity AI
"""

import os
import csv
import json
import time
import gc
from io import BytesIO
from typing import Callable, Dict, List, Mapping, Optional, Any, Generator, Tuple

import numpy as np
from PIL import Image

# Import local modules
from s3_storage import S3Storage
from algorithms import (
    DeflateImageCompressor,
    FilterType,
    load_image,
    raw_bytes_to_image,
    save_image,
    calculate_image_metrics,
    calculate_entropy,
    calculate_residual_entropy,
)

# Load environment variables from .env if present
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_file):
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "YOUR_BUCKET_NAME")
S3_BASE_PREFIX = os.getenv("S3_BASE_PREFIX", "Project_Data")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_ACCESS_KEY")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", None)


# ==============================================================================
# 1. S3 FILE STREAMING (GENERATOR) — không load hết vào RAM
# ==============================================================================
def stream_s3_images(
    limit: int = 10,
    bucket_name: str = S3_BUCKET_NAME,
    prefix: str = S3_BASE_PREFIX
) -> Generator[Tuple[str, Image.Image], None, None]:
    """
    Generator: Yield từng cặp (filename, PIL Image) một lần, không giữ toàn bộ trong RAM.
    Nếu S3 chưa cấu hình, tạo ảnh synthetic thay thế.

    Yields:
        Tuple[str, Image.Image]: (tên file unique, ảnh PIL)

    Ghi chú về tên file:
        Ảnh từ 4 nhóm khác nhau có thể trùng basename (vd: 000001.bmp).
        Để tránh ghi đè kết quả, tên file được tạo thành:
            <group_name>__<basename>  (vd: 1_Natural_Complex__000001.bmp)
        Nếu key không có sub-folder (chỉ có basename), dùng basename trực tiếp.
    """
    count = 0

    # Thử kết nối S3
    if bucket_name and bucket_name != "YOUR_BUCKET_NAME":
        try:
            storage = S3Storage(
                bucket_name=bucket_name,
                prefix=prefix,
                endpoint_url=S3_ENDPOINT_URL,
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
            storage.sync_existing()

            # Phân loại key theo nhóm và lấy tối đa `limit` ảnh MỖI NHÓM
            from collections import defaultdict
            grouped_keys = defaultdict(list)
            for key in storage._existing_keys:
                key_stripped = key
                if prefix and key_stripped.startswith(prefix.rstrip("/") + "/"):
                    key_stripped = key_stripped[len(prefix.rstrip("/")) + 1:]
                parts = key_stripped.replace("\\", "/").split("/")
                group = parts[0] if len(parts) >= 2 else "unknown"
                grouped_keys[group].append(key)

            keys = []
            for g in sorted(grouped_keys.keys()):
                keys.extend(sorted(grouped_keys[g])[:limit])

            for key in keys:
                img_obj, _, _ = storage.load_data(key)
                if img_obj is not None:
                    # Tạo tên file unique: ghép tên sub-folder (nhóm) + basename
                    # key ví dụ: "Project_Data/1_Natural_Complex/000001.bmp"
                    # → relative_path = "1_Natural_Complex/000001.bmp"
                    # → fname = "1_Natural_Complex__000001.bmp"
                    key_stripped = key
                    if prefix and key_stripped.startswith(prefix.rstrip("/") + "/"):
                        key_stripped = key_stripped[len(prefix.rstrip("/")) + 1:]
                    parts = key_stripped.replace("\\", "/").split("/")
                    if len(parts) >= 2:
                        fname = "__".join(parts)  # "1_Natural_Complex__000001.bmp"
                    else:
                        fname = parts[-1]          # fallback: basename trực tiếp

                    yield fname, img_obj.convert("RGB")
                    count += 1
                    # Giải phóng ngay sau khi yield
                    del img_obj
                    gc.collect()
            return
        except Exception as e:
            print(f"[CẢNH BÁO S3] Lỗi khi kết nối S3: {e}. Chuyển sang tạo dữ liệu mẫu...")

    # Fallback: Sinh ảnh synthetic từng cái một
    print(f"[THÔNG BÁO] Tạo {limit} ảnh thử nghiệm synthetic cho bộ benchmark...")
    w, h = 400, 400
    for idx in range(limit):
        if count >= limit:
            return
        file_name = f"sample_{idx+1:02d}.bmp"
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                r = int(((x + idx * 20) % w) / w * 255)
                g = int(((y + idx * 30) % h) / h * 255)
                b = int((x * y) % 256)
                arr[y, x] = [r, g, b]
        img = Image.fromarray(arr)
        yield file_name, img
        count += 1
        del arr, img
        gc.collect()


# ==============================================================================
# 2. XỬ LÝ TỪNG ẢNH MỘT (STREAMING PIPELINE)
# ==============================================================================
def _process_single_image(
    fname: str,
    img: Image.Image,
    compress_fn: Callable[[Image.Image], bytes],
    decompress_fn: Callable[[bytes], Image.Image],
    algo_name: str,
    file_extension: str,
    num_runs: int,
    compress_dir: str,
    images_out_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Nén và giải nén 1 ảnh, trả về dict kết quả cho report.
    Không giữ dữ liệu lớn trong bộ nhớ sau khi xong.
    """
    raw_bytes = img.tobytes()
    orig_size = len(raw_bytes)
    del raw_bytes

    # Đo lường Shannon Entropy gốc (Zero-order) và Residual Entropy (sau bộ lọc Sub Predictor)
    arr_rgb_u8 = np.array(img, dtype=np.uint8)
    entropy_val = calculate_entropy(arr_rgb_u8)
    residual_entropy_val = calculate_residual_entropy(arr_rgb_u8)
    pixel_std = float(arr_rgb_u8.astype(np.float32).std())

    # Trích tên nhóm ảnh từ fname (vd: "1_Natural_Complex__000001.bmp" → "1_Natural_Complex")
    image_group = fname.split("__")[0] if "__" in fname else "unknown"

    # ---- COMPRESS ----
    comp_times = []
    comp_bytes = b""
    for _ in range(num_runs):
        t0 = time.perf_counter()
        comp_bytes = compress_fn(img)
        t1 = time.perf_counter()
        comp_times.append(t1 - t0)

    avg_comp = sum(comp_times) / len(comp_times)
    min_comp = min(comp_times)
    comp_size = len(comp_bytes)
    cr = orig_size / comp_size if comp_size > 0 else 0
    space_saving = (1.0 - comp_size / orig_size) * 100.0 if orig_size > 0 else 0
    bpp = (comp_size * 8) / (img.width * img.height)
    comp_speed = (orig_size / (1024 * 1024)) / avg_comp if avg_comp > 0 else 0

    # Ghi file nén xuống đĩa
    extension = file_extension if file_extension.startswith(".") else f".{file_extension}"
    out_fname = f"{os.path.splitext(fname)[0]}{extension}"
    out_path = os.path.join(compress_dir, out_fname)
    with open(out_path, "wb") as f:
        f.write(comp_bytes)

    comp_report = {
        "algorithm": algo_name,
        "filename": fname,
        "image_group": image_group,
        "compressed_filename": out_fname,
        "dimensions": f"{img.width}x{img.height}",
        "original_bytes": orig_size,
        "compressed_bytes": comp_size,
        "compression_ratio": round(cr, 2),
        "space_saving_pct": round(space_saving, 2),
        "bits_per_pixel": round(bpp, 2),
        "avg_compress_time_sec": round(avg_comp, 4),
        "min_compress_time_sec": round(min_comp, 4),
        "compress_speed_mb_s": round(comp_speed, 2),
        "entropy": round(entropy_val, 4),
        "residual_entropy": round(residual_entropy_val, 4),
        "pixel_std": round(pixel_std, 4),
    }

    # ---- DECOMPRESS ----
    decomp_times = []
    restored_img = None
    for _ in range(num_runs):
        t0 = time.perf_counter()
        restored_img = decompress_fn(comp_bytes)
        t1 = time.perf_counter()
        decomp_times.append(t1 - t0)

    # Xả comp_bytes ngay sau khi dùng xong
    del comp_bytes
    gc.collect()

    avg_decomp = sum(decomp_times) / len(decomp_times)
    min_decomp = min(decomp_times)
    decomp_speed = (orig_size / (1024 * 1024)) / avg_decomp if avg_decomp > 0 else 0

    # Kiểm tra chất lượng lossless
    is_lossless, mse, psnr_db = calculate_image_metrics(img, restored_img)

    # Ghi ảnh giải nén xuống đĩa
    out_image_path = os.path.join(images_out_dir, fname)
    save_image(restored_img, out_image_path)

    # Xả ảnh khôi phục
    del restored_img
    gc.collect()

    decomp_report = {
        "algorithm": algo_name,
        "filename": fname,
        "restored_path": out_image_path,
        "dimensions": f"{img.width}x{img.height}",
        "compressed_bytes": comp_size,
        "restored_bytes": orig_size,
        "avg_decompress_time_sec": round(avg_decomp, 4),
        "min_decompress_time_sec": round(min_decomp, 4),
        "decompress_speed_mb_s": round(decomp_speed, 2),
        "is_lossless": is_lossless,
        "mse": round(mse, 4),
        "psnr_db": "Infinity" if np.isinf(psnr_db) else round(psnr_db, 2),
    }

    return comp_report, decomp_report


def _write_report_csv(
    report_dir: str,
    filename: str,
    reports: List[Dict[str, Any]],
) -> None:
    """Atomically checkpoint report rows so interrupted runs keep their data."""
    if not reports:
        return

    output_path = os.path.join(report_dir, filename)
    temporary_path = f"{output_path}.tmp"
    rows = sorted(reports, key=lambda row: row["filename"])
    with open(temporary_path, "w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, output_path)


def _write_report_json(
    report_dir: str,
    filename: str,
    reports: List[Dict[str, Any]],
) -> None:
    """Write the completed JSON report for one operation."""
    if not reports:
        return

    output_path = os.path.join(report_dir, filename)
    temporary_path = f"{output_path}.tmp"
    rows = sorted(reports, key=lambda row: row["filename"])
    with open(temporary_path, "w", encoding="utf-8") as report_file:
        json.dump(rows, report_file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, output_path)


def run_multi_codec_benchmark_pipeline(
    codecs: Mapping[str, Tuple[Callable, Callable, str]],
    num_runs: int = 1,
    limit_files: int = 10,
    max_workers: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Fetch each image once, then benchmark every codec sequentially on it.

    Sequential codec execution keeps timing measurements comparable. CSV files
    are checkpointed after every successful codec/image pair, while JSON files
    are written when the complete run finishes.
    """
    if not codecs:
        raise ValueError("At least one codec is required.")
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1")

    base_output_dir = "output_not_commit"
    state: Dict[str, Dict[str, Any]] = {}

    for algo_name in codecs:
        compress_root = os.path.join(base_output_dir, algo_name, "compress")
        compress_dir = os.path.join(compress_root, "files")
        images_out_dir = os.path.join(compress_root, "images")
        report_dir = os.path.join(compress_root, "report")
        os.makedirs(compress_dir, exist_ok=True)
        os.makedirs(images_out_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        state[algo_name] = {
            "compress_dir": compress_dir,
            "images_out_dir": images_out_dir,
            "report_dir": report_dir,
            "compression": [],
            "decompression": [],
        }

    print(f"\n{'=' * 80}")
    print("MULTI-CODEC BENCHMARK: ONE S3 IMAGE -> ALL CODECS")
    print(f"Images per group: {limit_files} | Timing repeats per image: {num_runs}")
    print(f"Codec order: {' -> '.join(codecs)}")
    if max_workers != 1:
        print("Note: codecs run sequentially; --workers is ignored for fair timing.")
    print(f"{'=' * 80}")

    total_images = 0
    for fname, img in stream_s3_images(limit=limit_files):
        total_images += 1
        print(f"\n[{total_images:>4}] {fname} ({img.width}x{img.height})")

        try:
            for algo_name, (compress_fn, decompress_fn, file_extension) in codecs.items():
                algo_state = state[algo_name]
                print(f"       {algo_name:<16}", end="", flush=True)
                try:
                    comp_report, decomp_report = _process_single_image(
                        fname=fname,
                        img=img,
                        compress_fn=compress_fn,
                        decompress_fn=decompress_fn,
                        algo_name=algo_name,
                        file_extension=file_extension,
                        num_runs=num_runs,
                        compress_dir=algo_state["compress_dir"],
                        images_out_dir=algo_state["images_out_dir"],
                    )
                    algo_state["compression"].append(comp_report)
                    algo_state["decompression"].append(decomp_report)

                    _write_report_csv(
                        algo_state["report_dir"],
                        "compression_report.csv",
                        algo_state["compression"],
                    )
                    _write_report_csv(
                        algo_state["report_dir"],
                        "decompression_report.csv",
                        algo_state["decompression"],
                    )

                    lossless = "OK" if decomp_report["is_lossless"] else "FAIL"
                    print(
                        f" CR={comp_report['compression_ratio']:.2f}x"
                        f" | saving={comp_report['space_saving_pct']:.1f}%"
                        f" | lossless={lossless}"
                    )
                except Exception as error:
                    print(f" ERROR: {error}")
        finally:
            img.close()
            del img
            gc.collect()

    print(f"\n{'=' * 80}")
    print(f"MULTI-CODEC BENCHMARK COMPLETE: {total_images} S3 images fetched once each")
    for algo_name, algo_state in state.items():
        comp_reports = algo_state["compression"]
        decomp_reports = algo_state["decompression"]
        _write_report_json(
            algo_state["report_dir"],
            "compression_report.json",
            comp_reports,
        )
        _write_report_json(
            algo_state["report_dir"],
            "decompression_report.json",
            decomp_reports,
        )

        if comp_reports:
            avg_cr = sum(row["compression_ratio"] for row in comp_reports) / len(comp_reports)
            lossless_count = sum(row["is_lossless"] for row in decomp_reports)
            print(
                f"  {algo_name:<16} images={len(comp_reports):>4}"
                f" | avg CR={avg_cr:.2f}x"
                f" | lossless={lossless_count}/{len(decomp_reports)}"
            )
        else:
            print(f"  {algo_name:<16} no successful images")
    print(f"{'=' * 80}\n")

    return state


# ==============================================================================
# 3. PIPELINE CHÍNH — XỬ LÝ TỪNG ẢNH, TRÁNH TRÀN RAM
# ==============================================================================
def run_s3_benchmark_pipeline(
    compress_fn: Optional[Callable[[Image.Image], bytes]] = None,
    decompress_fn: Optional[Callable[[bytes], Image.Image]] = None,
    algo_name: str = "deflate",
    file_extension: str = ".defimg",
    num_runs: int = 1,
    limit_files: int = 10,
    max_workers: int = 1,
):
    """
    Pipeline benchmark end-to-end, xử lý TỪNG ẢNH MỘT để tiết kiệm RAM tối đa:
    1. Fetch 1 ảnh từ S3 / synthetic.
    2. Compress ảnh đó.
    3. Decompress ảnh đó.
    4. Ghi kết quả vào report.
    5. Xả RAM → sang ảnh tiếp theo.

    Lưu ý: max_workers giữ để tương thích với run.py nhưng pipeline
    hiện chạy tuần tự để đảm bảo RAM không bị tràn.
    """
    # Default compressor nếu không truyền vào
    if compress_fn is None or decompress_fn is None:
        compressor = DeflateImageCompressor(compression_level=9, filter_type=FilterType.ADAPTIVE)
        if compress_fn is None:
            compress_fn = compressor.compress
        if decompress_fn is None:
            decompress_fn = compressor.decompress

    if num_runs < 1:
        raise ValueError("num_runs must be at least 1")

    # Chuẩn bị thư mục output
    base_output_dir = "output_not_commit"
    compress_dir = os.path.join(base_output_dir, algo_name, "compress", "files")
    images_out_dir = os.path.join(base_output_dir, algo_name, "compress", "images")
    report_dir = os.path.join(base_output_dir, algo_name, "compress", "report")
    os.makedirs(compress_dir, exist_ok=True)
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"========== BẮT ĐẦU BENCHMARK (STREAMING - TỪNG ẢNH MỘT) ==========")
    print(f"Thuật toán: {algo_name} | Giới hạn: {limit_files} ảnh/nhóm | Lặp: {num_runs} lần/ảnh")
    print(f"Lưu file nén:     {compress_dir}")
    print(f"Lưu ảnh giải nén: {images_out_dir}")
    print(f"Lưu report:       {report_dir}")
    print(f"{'='*80}")

    comp_reports: List[Dict[str, Any]] = []
    decomp_reports: List[Dict[str, Any]] = []

    total_fetched = 0

    # *** VÒNG LẶP CHÍNH: fetch → compress → decompress → xả RAM → tiếp ***
    for fname, img in stream_s3_images(limit=limit_files):
        total_fetched += 1
        print(f"  [{total_fetched:>4}/{limit_files}] Đang xử lý: {fname} ({img.width}x{img.height})", end="", flush=True)

        try:
            comp_report, decomp_report = _process_single_image(
                fname=fname,
                img=img,
                compress_fn=compress_fn,
                decompress_fn=decompress_fn,
                algo_name=algo_name,
                file_extension=file_extension,
                num_runs=num_runs,
                compress_dir=compress_dir,
                images_out_dir=images_out_dir,
            )
            comp_reports.append(comp_report)
            decomp_reports.append(decomp_report)

            cr = comp_report["compression_ratio"]
            saving = comp_report["space_saving_pct"]
            lossless = "✓" if decomp_report["is_lossless"] else "✗"
            print(f" | CR={cr:.2f}x | Tiết kiệm={saving:.1f}% | Lossless={lossless}")

        except Exception as e:
            print(f" | ❌ LỖI: {e}")

        finally:
            # Xả ảnh gốc khỏi RAM ngay sau mỗi ảnh
            del img
            gc.collect()

    if not comp_reports:
        print("[CẢNH BÁO] Không có ảnh nào được xử lý!")
        return

    # Sắp xếp report theo tên file
    comp_reports.sort(key=lambda x: x["filename"])
    decomp_reports.sort(key=lambda x: x["filename"])

    # Ghi CSV + JSON báo cáo nén
    csv_comp = os.path.join(report_dir, "compression_report.csv")
    with open(csv_comp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comp_reports[0].keys()))
        writer.writeheader()
        writer.writerows(comp_reports)

    json_comp = os.path.join(report_dir, "compression_report.json")
    with open(json_comp, "w", encoding="utf-8") as f:
        json.dump(comp_reports, f, indent=2, ensure_ascii=False)

    # Ghi CSV + JSON báo cáo giải nén
    csv_decomp = os.path.join(report_dir, "decompression_report.csv")
    with open(csv_decomp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(decomp_reports[0].keys()))
        writer.writeheader()
        writer.writerows(decomp_reports)

    json_decomp = os.path.join(report_dir, "decompression_report.json")
    with open(json_decomp, "w", encoding="utf-8") as f:
        json.dump(decomp_reports, f, indent=2, ensure_ascii=False)

    # Tóm tắt kết quả
    avg_cr = sum(r["compression_ratio"] for r in comp_reports) / len(comp_reports)
    avg_saving = sum(r["space_saving_pct"] for r in comp_reports) / len(comp_reports)
    lossless_count = sum(1 for r in decomp_reports if r["is_lossless"])

    print(f"\n{'='*80}")
    print(f"HOÀN THÀNH TOÀN BỘ QUY TRÌNH BENCHMARK CHO THUẬT TOÁN [{algo_name.upper()}]")
    print(f"  Tổng ảnh đã xử lý:  {total_fetched}")
    print(f"  CR trung bình:       {avg_cr:.2f}x")
    print(f"  Tiết kiệm TB:        {avg_saving:.1f}%")
    print(f"  Lossless:            {lossless_count}/{total_fetched}")
    print(f"  Báo cáo nén:         {csv_comp}")
    print(f"  Báo cáo giải nén:    {csv_decomp}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    run_s3_benchmark_pipeline(
        algo_name="deflate",
        num_runs=1,
        limit_files=10,
    )
