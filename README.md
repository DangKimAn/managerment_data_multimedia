# Multimedia Data Management - DEFLATE Compression Project

This project contains Python scripts to test and benchmark the **DEFLATE** image compression algorithm on images downloaded from S3 storage.

## 1. Installation

Please follow these simple steps to install and set up the project on your computer.

### Step 1: Open the terminal
Open your terminal or command prompt in this project folder.

### Step 2: Create a Virtual Environment (Recommended)
This keeps your project files clean.
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*(If you are on Windows, use `.venv\Scripts\activate` instead).*

### Step 3: Install Required Packages
Install all the necessary Python libraries like `pandas`, `matplotlib`, and `Pillow`.
```bash
pip install -r requirements.txt
```

### Step 4: Set up the Configuration File (`.env`)
You need a `.env` file to connect to S3 and configure your test settings.
1. Copy the example file:
```bash
cp .env.example .env
```
2. Open the `.env` file and enter your AWS S3 information (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
3. You can also change the number of test runs (`TOTAL_RUNS`) or images per group (`IMAGES_PER_GROUP`).

---

## 2. How to Run the Algorithm

### Step 1: Run the Benchmark Pipeline
To test the compression and decompression on your S3 images, run this simple command:
```bash
python run.py
```
*Note: This script will download images from S3, compress them using DEFLATE, measure the time, and save the data in the `output_not_commit/` folder.*

### Step 2: Generate the Charts
After the test finishes, you can draw beautiful charts to analyze the data. Just type:
```bash
python plot_reports.py
```
*Note: The script will read your `.env` file and automatically generate the graphs. You can find your charts in the `output/deflate/charts/` folder.*

---

## 3. Main Files
* **`algorithms/`**: Contains the core DEFLATE algorithm (LZ77, Huffman, Spatial Filters). Check `algorithms/deflate/README.md` for algorithm details.
* **`run.py`**: The main script to run the test.
* **`s3_benchmark.py`**: Connects to S3, loads images, and measures speed.
* **`plot_reports.py`**: Draws statistical charts based on the results.
