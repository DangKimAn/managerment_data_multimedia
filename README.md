# Quản lý dữ liệu đa phương tiện - So sánh nén ảnh không tổn thất

Project dùng cùng một bộ ảnh trên AWS S3 để benchmark và so sánh ba phương pháp nén ảnh không tổn thất:

- DEFLATE
- TIFF-LZW
- WebP Lossless

## 1. Cài đặt môi trường

### Bước 1: Tạo môi trường ảo

```powershell
python -m venv .venv
```

Kích hoạt môi trường trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Bước 2: Cài thư viện

```powershell
python -m pip install -r requirements.txt
```

### Bước 3: Cấu hình `.env`

Tạo file `.env` tại thư mục gốc của project và khai báo:

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

Ý nghĩa cấu hình:

- `IMAGES_PER_GROUP`: số ảnh tối đa lấy từ mỗi nhóm trên S3.
- `TOTAL_RUNS`: số lần chạy lại toàn bộ benchmark.
- `S3_BASE_PREFIX`: thư mục gốc chứa dataset trên S3.

Không đưa file `.env` hoặc AWS credential thật lên Git. File `.env.example` chỉ được chứa giá trị minh họa.

## 2. Flow benchmark

Khi chạy `--algo all`, mỗi ảnh chỉ được tải một lần từ S3 rồi lần lượt đi qua ba codec:

```text
Tải ảnh 1 từ S3
├── DEFLATE     → nén → giải nén → đo → ghi CSV
├── TIFF-LZW    → nén → giải nén → đo → ghi CSV
└── WebP        → nén → giải nén → đo → ghi CSV
↓ giải phóng ảnh khỏi RAM
Tải ảnh 2 và lặp lại
```

Ba codec chạy tuần tự trên cùng dữ liệu pixel để kết quả thời gian không bị ảnh hưởng bởi việc tranh CPU. CSV được checkpoint sau từng codec/ảnh nên những dòng đã hoàn thành vẫn còn nếu chương trình bị gián đoạn.

## 3. Cách chạy

### Chạy thử với ít ảnh

Lệnh sau chạy một lần, lấy tối đa hai ảnh từ mỗi nhóm S3 và đo mỗi thao tác một lần:

```powershell
python run.py --algo all --total-runs 1 --limit 2 --repeat 1
```

Sau khi benchmark hoàn tất, tạo bảng và biểu đồ so sánh:

```powershell
python plot_reports.py --algos deflate tiff_lzw webp_lossless --runs 1
```

### Chạy đầy đủ theo `.env`

```powershell
python run.py --algo all
```

Sau đó tạo báo cáo cho các lần chạy được cấu hình bởi `TOTAL_RUNS`:

```powershell
python plot_reports.py --algos deflate tiff_lzw webp_lossless
```

### Chạy riêng từng thuật toán

```powershell
python run.py --algo deflate
python run.py --algo tiff_lzw
python run.py --algo webp_lossless
```

Tạo biểu đồ riêng cho một thuật toán:

```powershell
python plot_reports.py --algo deflate
python plot_reports.py --algo tiff_lzw
python plot_reports.py --algo webp_lossless
```

### Các tham số của `run.py`

```text
--algo, -a         Thuật toán: deflate, tiff_lzw, webp_lossless hoặc all
--total-runs, -r   Số lần chạy toàn bộ benchmark
--limit, -l        Số ảnh tối đa lấy từ mỗi nhóm S3
--repeat, -n       Số lần lặp nén/giải nén trên mỗi ảnh để đo thời gian
--filter, -f       Bộ lọc không gian dành riêng cho DEFLATE
--pure-python      Dùng DEFLATE LZ77 + Huffman Python thuần
--workers, -w      Dành cho chế độ song song trong tương lai; hiện chạy tuần tự
```

Xem toàn bộ tùy chọn:

```powershell
python run.py --help
python plot_reports.py --help
```

## 4. Các chỉ số được đo

Với mỗi ảnh và mỗi thuật toán, pipeline ghi nhận:

- Kích thước dữ liệu pixel gốc.
- Kích thước sau nén.
- Tỷ lệ nén.
- Phần trăm tiết kiệm dung lượng.
- Số bit trên mỗi pixel (BPP).
- Thời gian và tốc độ nén.
- Thời gian và tốc độ giải nén.
- Kiểm tra ảnh khôi phục có giống ảnh gốc 100% hay không.
- MSE và PSNR.
- Nhóm ảnh và độ lệch chuẩn pixel để phân tích ảnh hưởng của đặc điểm ảnh.

`original_bytes` là kích thước dữ liệu pixel thô, không phải dung lượng file BMP ban đầu trên S3.

## 5. Kết quả đầu ra

File nén, ảnh khôi phục và report riêng của từng thuật toán:

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

Mỗi thư mục `report/` chứa:

```text
compression_report.csv
compression_report.json
decompression_report.csv
decompression_report.json
```

Bảng và biểu đồ so sánh ba thuật toán:

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

`plot_reports.py` chỉ so sánh các cặp `filename + run_id` xuất hiện ở tất cả thuật toán được chọn. Điều này bảo đảm các thuật toán được đánh giá trên cùng bộ ảnh.

## 6. Các file chính

- `algorithms/`: code nén và giải nén của từng thuật toán.
- `run.py`: chọn codec và điều phối benchmark.
- `s3_benchmark.py`: tải ảnh S3, đo các chỉ số và ghi report.
- `plot_reports.py`: tổng hợp CSV và tạo biểu đồ.
- `s3_storage.py`: đọc và ghi dữ liệu trên S3.
- `crawl_dataset.py`: thu thập dataset và đưa ảnh lên S3.

## 7. Lưu ý về WebP hiện tại

WebP hiện tại khôi phục pixel không tổn thất khi dùng compressor và decompressor của project. Tuy nhiên payload VP8L đang dùng cấu trúc tùy biến nên chưa thể đọc bằng decoder WebP chuẩn như Pillow/libwebp.

Vì vậy, trước khi đưa ra kết luận cuối cùng về WebP Lossless chuẩn, cần chuyển codec này sang libwebp/Pillow hoặc triển khai đầy đủ đặc tả VP8L.
