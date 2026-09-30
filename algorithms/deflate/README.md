# Lossless Image Compression with DEFLATE

This project implements the **DEFLATE** image compression algorithm in Python. It combines **LZ77** (dictionary compression) and **Huffman Coding** (entropy coding). It also includes **PNG-style Spatial Prediction Filters** to get much better compression on images.

This project is designed for the **Multimedia Data Management** course.

---

## Project Structure

```
algorithms/
├── __init__.py            # Package entry point and exports
├── image_utils.py         # Reads images, flattens to 1D bytes, and reconstructs 2D images
├── spatial_filters.py     # Spatial prediction filters (Sub, Up, Average, Paeth, Adaptive)
├── lz77.py                # Standalone LZ77 sliding window compression & decompression
├── huffman.py             # Standalone Huffman tree, bitstream encoder, and decoder
├── deflate_compressor.py  # Coordinates the full pipeline (calls filters + LZ77 + Huffman)
├── demo.py                # Benchmark test suite and CLI tool
└── README.md              # Documentation
```

---

## 1. How DEFLATE Works on Images

The DEFLATE algorithm (RFC 1951) is the main compression engine behind popular image formats like **PNG** and **TIFF Zip**. The compression process has 3 main steps:

```
[Original Image (BMP / RGB)]
           │
           ▼
[Step 1: Spatial Prediction Filter]
   (Calculates differences between nearby pixels to create values close to 0)
           │
           ▼
[Step 2: LZ77 Algorithm]
   (Finds repeated byte patterns and replaces them with distance/length pairs)
           │
           ▼
[Step 3: Huffman Coding]
   (Assigns shorter binary codes to symbols that appear more often)
           │
           ▼
[Compressed Binary File (.defimg)]
```

### Why do we need filters before DEFLATE?

* Pure DEFLATE is a 1D byte-stream compressor. It works best when exact byte sequences repeat frequently.
* In natural photos, nearby pixels often have similar colors (for example: `150, 151, 150, 152...`). Even though the colors look similar to our eyes, the byte numbers are slightly different. Because of this, LZ77 cannot find long repeating patterns.
* **The Filter Solution:** We calculate the difference between the current pixel and its neighbor (left pixel, above pixel, or Paeth predictor). Then the numbers `150, 151, 150, 152` turn into `150, 1, -1, 2...`. Most values become small numbers near zero!
* Now, Huffman coding and LZ77 can compress these small numbers very easily. The compression ratio (CR) can improve from **1.5x up to 50x - 150x+**!

---

## 2. Supported Filters (`FilterType`)

| Code | Filter Name | Formula | Best Used For |
| :--- | :--- | :--- | :--- |
| `0` | `NONE` | No filter, keeps original bytes | Random noisy data, high entropy images |
| `1` | `SUB` | $F(x) = (x - a) \pmod{256}$ ($a$ is left pixel) | Images with horizontal gradients |
| `2` | `UP` | $F(x) = (x - b) \pmod{256}$ ($b$ is above pixel) | Images with vertical gradients |
| `3` | `AVERAGE` | $F(x) = (x - \lfloor\frac{a + b}{2}\rfloor) \pmod{256}$ | Smooth color transitions in both directions |
| `4` | `PAETH` | Paeth predictor from left, above, and top-left | Images with sharp edges and shapes |
| `5` | `ADAPTIVE` | Automatically picks the best filter for each row | Best file size (PNG standard) |

---

## 3. Binary File Format (`.defimg`)

The compressed file contains a **20-byte fixed header** followed by the DEFLATE compressed data:

| Byte Offset | Data Type | Field Name | Description |
| :--- | :--- | :--- | :--- |
| `0 - 3` | `char[4]` | `Magic` | File signature `b"DFLT"` |
| `4` | `uint8` | `Version` | Container version (currently `1`) |
| `5` | `uint8` | `FilterMode` | Filter type used (0: None, ..., 5: Adaptive) |
| `6 - 9` | `char[4]` | `ColorMode` | Image color mode (`"RGB "`, `"L   "`, `"RGBA"`) |
| `10 - 13` | `uint32` | `Width` | Image width in pixels |
| `14 - 17` | `uint32` | `Height` | Image height in pixels |
| `18` | `uint8` | `Channels` | Number of color channels (1 for Gray, 3 for RGB) |
| `19` | `uint8` | `Reserved` | Reserved byte (`0x00`) |
| `20+` | `bytes` | `Payload` | DEFLATE compressed byte stream |

---

## 4. How to Measure Compression Performance

1. **Compression Ratio (CR):**
   $$CR = \frac{\text{Original Size (bytes)}}{\text{Compressed Size (bytes)}}$$
   *(A higher number means better compression)*.

2. **Space Saving Percentage (SS%):**
   $$SS\% = \left(1 - \frac{\text{Compressed Size}}{\text{Original Size}}\right) \times 100\%$$
   *(Shows how much disk space was saved)*.

3. **Bits Per Pixel (BPP):**
   $$BPP = \frac{\text{Compressed Size (bytes)} \times 8}{\text{Width} \times \text{Height}}$$
   *(An uncompressed 24-bit RGB image has $BPP = 24$. After compression, lower BPP is better)*.

4. **Lossless Check:**
   * Mean Squared Error: $MSE = 0.0$.
   * Peak Signal-to-Noise Ratio: $PSNR = \infty\text{ dB}$.
   * The decompressed image is 100% bit-for-bit identical to the original image (`np.array_equal == True`).

---

## 5. How to Use

### 5.1. Run the Benchmark on 4 Image Types

Run the benchmark test on 4 sample image types (Natural, Flat Graphics, Textures, Documents):

```bash
python -m algorithms.demo
```

**Benchmark Results:**
```text
===============================================================================================
                       DEFLATE IMAGE COMPRESSION BENCHMARK REPORT                              
===============================================================================================
Image Type           | Raw (bytes) | Direct (CR)  | Filtered (CR)  | Space Saved   | BPP    | Lossless
-----------------------------------------------------------------------------------------------
1_Natural_Complex    | 270,000     |  1.79x       | 158.08x        |  99.37%       |  0.15 | 100% OK 
2_Flat_Graphics      | 270,000     | 185.31x       | 176.36x        |  99.43%       |  0.14 | 100% OK 
3_Textures_Noisy     | 270,000     |  1.39x       |   1.12x        |  11.00%       | 21.36 | 100% OK 
4_Documents_Text     | 270,000     | 317.65x       | 324.13x        |  99.69%       |  0.07 | 100% OK 
===============================================================================================
```

> **Key Observations:**
> * **Flat Graphics & Documents:** Have very high compression ratios (>150x – 300x) because they have large areas of solid color and repeated patterns.
> * **Natural Complex:** With spatial prediction filters, smooth color gradients turn into zeros, which makes the compression ratio much higher than direct DEFLATE.
> * **Textures Noisy:** Has the lowest compression ratio because random noise has high entropy, leaving very few repeating patterns for LZ77 to find.

### 5.2. Use via Command Line (CLI)

* **Compress any image file (BMP, PNG, JPG) to a `.defimg` file:**
  ```bash
  python -m algorithms.demo --image path/to/image.bmp --output image.defimg
  ```

* **Decompress a `.defimg` file back to an image:**
  ```bash
  python -m algorithms.demo --decompress image.defimg --restore restored_image.bmp
  ```

### 5.3. Use in Python Code

```python
from PIL import Image
from algorithms import DeflateImageCompressor, FilterType, benchmark_image_compression

# 1. Initialize compressor with Adaptive Filter
compressor = DeflateImageCompressor(compression_level=9, filter_type=FilterType.ADAPTIVE)

# 2. Compress an image from file or PIL Image
img = Image.open("photo.bmp")
compressed_bytes = compressor.compress(img)

# Save to a compressed file
with open("photo.defimg", "wb") as f:
    f.write(compressed_bytes)

# 3. Decompress back to original image (100% lossless)
decompressed_img = compressor.decompress(compressed_bytes)
decompressed_img.save("restored.bmp")

# 4. Measure compression metrics
metrics = benchmark_image_compression("photo.bmp")
print(f"Compression Ratio (CR): {metrics['adaptive_cr']:.2f}x")
print(f"Space Saved: {metrics['space_saving_pct']:.2f}%")
print(f"Lossless check: {metrics['is_lossless']}")
```

---

## 6. Biểu đồ phân tích (plot_reports.py)

Chạy sau khi có kết quả benchmark để sinh toàn bộ biểu đồ:

```bash
python plot_reports.py --algo deflate --runs 1 2 3 4 5
```

### Danh sách biểu đồ được sinh ra

| File | Mô tả | Phục vụ |
|------|--------|---------|
| `2_distribution_compression_ratio.png` | Histogram + KDE phân phối CR | Tổng quan |
| `3_distribution_processing_time.png` | KDE thời gian nén/giải nén | Tổng quan |
| `4_boxplot_cr_space_saving.png` | Box plot CR & Space Saving | Tổng quan |
| `5_scatter_size_vs_cr.png` | Scatter kích thước file vs CR | Tổng quan |
| `6_boxplot_speed.png` | Box plot tốc độ nén/giải nén | Tổng quan |
| `8_rq1_cr_by_image_type.png` | **[RQ1]** Grouped bar CR & Space Saving theo loại ảnh | **RQ1** |
| `9_rq1_boxplot_cr_by_type.png` | **[RQ1]** Box plot phân phối CR theo loại ảnh | **RQ1** |
| `10_rq1_scatter_entropy_vs_cr.png` | **[RQ1]** Scatter entropy (pixel std) vs CR | **RQ1** |
| `11_rq1_scatter_speed_vs_cr_by_type.png` | **[RQ1]** Trade-off tốc độ nén vs CR theo loại ảnh | **RQ1** |
| `12_rq1_grouped_bar_speed.png` | **[RQ1]** Grouped bar tốc độ nén vs giải nén theo loại ảnh | **RQ1** |
| `rq1_summary_by_image_type.csv` | **[RQ1]** Bảng số liệu tổng hợp cho Chương 5.1 | **RQ1** |

---

## ⚠️ TODO: RQ2 — Còn thiếu, chưa triển khai

> **RQ2:** *"So sánh trực tiếp hiệu quả nén giữa DEFLATE, LZW và WebP trên cùng bộ dữ liệu ảnh?"*

Hiện tại toàn bộ code và biểu đồ chỉ phân tích **DEFLATE** đơn lẻ.
Để trả lời RQ2 cần bổ sung:

- [ ] Triển khai thuật toán **LZW** (hoặc wrapper) tương tự `algorithms/deflate/`
- [ ] Triển khai thuật toán **WebP** (dùng `Pillow` hoặc `libwebp`)
- [ ] Chạy `run.py` cho cả 3 thuật toán, lưu kết quả vào `output_not_commit/lzw/` và `output_not_commit/webp/`
- [ ] Thêm vào `plot_reports.py`:
  - Grouped bar: CR trung bình của 3 thuật toán × 4 loại ảnh (biểu đồ tổng hợp chính)
  - Scatter trade-off: tốc độ vs CR, mỗi điểm = 1 thuật toán (centroid)
  - Bảng so sánh tổng hợp: CR, BPP, tốc độ theo (thuật toán × loại ảnh) cho Chương 5
