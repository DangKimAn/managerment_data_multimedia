# Multimedia Data Management - Lossless Image Compression Comparison

This project benchmarks and compares three lossless image-compression methods on the same AWS S3 image dataset:

- DEFLATE
- TIFF-LZW
- WebP Lossless

## 1. Environment Setup

### Step 1: Create a virtual environment

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 3: Configure `.env`

Create a `.env` file in the project root with the following values:

```env
ENVIRONMENT=development

S3_BUCKET_NAME=your-bucket-name
S3_BASE_PREFIX=Project_Data
AWS_REGION=ap-southeast-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

IMAGES_PER_GROUP=100
TOTAL_RUNS=5
```

Configuration details:

- `IMAGES_PER_GROUP`: maximum number of images loaded from each S3 group.
- `TOTAL_RUNS`: number of times the complete benchmark is repeated.
- `S3_BASE_PREFIX`: root directory containing the dataset in S3.

Never commit `.env` or real AWS credentials. `.env.example` must contain placeholder values only.

## 2. Benchmark Flow

When `--algo all` is selected, each image is downloaded from S3 once and then processed sequentially by all three codecs:

```text
Download image 1 from S3
├── DEFLATE     → compress → decompress → measure → checkpoint CSV
├── TIFF-LZW    → compress → decompress → measure → checkpoint CSV
└── WebP        → compress → decompress → measure → checkpoint CSV
↓ release the image from memory
Download image 2 and repeat
```

The codecs run sequentially on the same pixel data so their timing results are not distorted by CPU contention. CSV reports are checkpointed after every codec/image result, so completed rows remain available if the process is interrupted.

## 3. Running the Project

### Quick test with a small dataset

The following command runs one benchmark round, loads up to two images from each S3 group, and measures each operation once:

```powershell
python run.py --algo all --total-runs 1 --limit 2 --repeat 1
```

After the benchmark finishes, generate the comparison tables and charts:

```powershell
python plot_reports.py --algos deflate tiff_lzw webp_lossless --runs 1
```

### Full benchmark using `.env`

```powershell
python run.py --algo all
```

Generate reports for the runs configured by `TOTAL_RUNS`:

```powershell
python plot_reports.py --algos deflate tiff_lzw webp_lossless
```

### Run one algorithm only

```powershell
python run.py --algo deflate
python run.py --algo tiff_lzw
python run.py --algo webp_lossless
```

Generate charts for one algorithm:

```powershell
python plot_reports.py --algo deflate
python plot_reports.py --algo tiff_lzw
python plot_reports.py --algo webp_lossless
```

### `run.py` options

```text
--algo, -a         Algorithm: deflate, tiff_lzw, webp_lossless, or all
--total-runs, -r   Number of complete benchmark runs
--limit, -l        Maximum images loaded from each S3 group
--repeat, -n       Compression/decompression repetitions per image for timing
--filter, -f       Spatial filter used only by DEFLATE
--pure-python      Use the educational pure-Python LZ77 + Huffman DEFLATE engine
--workers, -w      Reserved for a future parallel mode; current timing mode is sequential
```

Display all available options:

```powershell
python run.py --help
python plot_reports.py --help
```

## 4. Measured Metrics

For every image and algorithm, the pipeline records:

- Original raw pixel size.
- Compressed size.
- Compression ratio.
- Space-saving percentage.
- Bits per pixel (BPP).
- Compression time and speed.
- Decompression time and speed.
- Whether the restored image is 100% identical to the original.
- MSE and PSNR.
- Image group and pixel standard deviation for analyzing how image characteristics affect compression.

`original_bytes` represents the uncompressed pixel data size, not the original BMP file size in S3.

## 5. Output Files

Compressed files, restored images, and per-algorithm reports:

```text
output_not_commit/
├── deflate/<run_id>/compress/
│   ├── files/
│   ├── images/
│   └── report/
├── tiff_lzw/<run_id>/compress/
│   ├── files/
│   ├── images/
│   └── report/
└── webp_lossless/<run_id>/compress/
    ├── files/
    ├── images/
    └── report/
```

Each `report/` directory contains:

```text
compression_report.csv
compression_report.json
decompression_report.csv
decompression_report.json
```

Three-algorithm comparison tables and charts:

```text
output/comparison/charts/
├── comparison_summary.csv
├── comparison_details.csv
├── comparison_by_image_type.csv
├── 1_compression_efficiency.png
├── 2_compression_ratio_boxplot.png
├── 3_processing_time.png
├── 4_compression_ratio_by_image_type.png
└── 5_speed_vs_compression_ratio.png
```

`plot_reports.py` compares only `filename + run_id` pairs available for every selected algorithm. This ensures that all algorithms are evaluated on the same images.

## 6. Main Files

- `algorithms/`: compression and decompression implementations for each algorithm.
- `run.py`: selects codecs and coordinates benchmark runs.
- `s3_benchmark.py`: downloads S3 images, measures metrics, and writes reports.
- `plot_reports.py`: aggregates CSV data and generates charts.
- `s3_storage.py`: reads and writes S3 objects.
- `crawl_dataset.py`: collects the dataset and uploads images to S3.

## 7. Current WebP Limitation

The current WebP implementation restores pixels losslessly when used with the project's own compressor and decompressor. However, its VP8L payload uses a custom structure and cannot yet be decoded by standard WebP implementations such as Pillow/libwebp.

Before drawing final conclusions about standards-compliant WebP Lossless, replace this codec with libwebp/Pillow or implement the complete VP8L specification.
