"""
Statistical Report Visualization Module using Matplotlib & Seaborn.
Designed for large-scale benchmarks: 400+ images x multiple runs.
Instead of per-file bar charts, generates statistical summary charts:
  - Distribution (KDE/Histogram) of compression metrics
  - Box plots comparing multiple runs
  - Scatter plots (file size vs CR)
  - Summary table (mean, median, std, min, max per run)
"""

import os
import argparse

# Keep Matplotlib's writable cache inside the ignored benchmark-output folder.
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_not_commit", ".matplotlib"),
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Đọc giá trị mặc định từ .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for _line in f:
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

_DEFAULT_TOTAL_RUNS = int(os.environ.get("TOTAL_RUNS", 5))
_DEFAULT_RUNS = list(range(1, _DEFAULT_TOTAL_RUNS + 1))

# Set publication-quality style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#aaaaaa",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
})

CHART_DPI = 200


def _load_reports(base_input_dir: str, algo_name: str, run_id: int):
    """Load compression & decompression CSV for a given algo/run_id."""
    report_dir = os.path.join(base_input_dir, algo_name, str(run_id), "compress", "report")
    comp_csv = os.path.join(report_dir, "compression_report.csv")
    decomp_csv = os.path.join(report_dir, "decompression_report.csv")
    if not os.path.exists(comp_csv) or not os.path.exists(decomp_csv):
        return None, None, report_dir
    return pd.read_csv(comp_csv), pd.read_csv(decomp_csv), report_dir


def _load_all_runs(base_input_dir: str, algo_name: str, run_ids: list[int]):
    """
    Load and merge reports from multiple runs into a single DataFrame,
    adding a 'run_id' column to distinguish them.
    """
    comp_frames = []
    decomp_frames = []
    for rid in run_ids:
        df_c, df_d, _ = _load_reports(base_input_dir, algo_name, rid)
        if df_c is not None:
            df_c = df_c.copy()
            df_c["run_id"] = str(rid)
            comp_frames.append(df_c)
        if df_d is not None:
            df_d = df_d.copy()
            df_d["run_id"] = str(rid)
            decomp_frames.append(df_d)
    df_comp_all = pd.concat(comp_frames, ignore_index=True) if comp_frames else None
    df_decomp_all = pd.concat(decomp_frames, ignore_index=True) if decomp_frames else None
    return df_comp_all, df_decomp_all


def generate_benchmark_charts(
    algo_name: str = "deflate",
    run_ids: list = None,
    base_input_dir: str = "output_not_commit",
    base_plot_dir: str = "output"
):
    """
    Generate statistical benchmark charts suitable for large datasets (400+ images x N runs).
    Bao gồm 7 chart thống kê tổng quát + 4 chart phân tích RQ1 (theo loại ảnh).

    Args:
        algo_name: Algorithm name (used for folder path and chart titles).
        run_ids: List of run IDs to include. Defaults to [1].
        base_input_dir: Root folder containing output_not_commit/ reports.
        base_plot_dir: Root folder where charts are saved.
    """
    if run_ids is None:
        run_ids = [1]

    charts_dir = os.path.join(base_plot_dir, algo_name, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Load all runs into combined DataFrames
    df_comp_all, df_decomp_all = _load_all_runs(base_input_dir, algo_name, run_ids)

    if df_comp_all is None:
        print(f"[ERROR] Cannot find report data in '{base_input_dir}/{algo_name}/'")
        return

    df_comp_all["orig_mb"] = df_comp_all["original_bytes"] / (1024 * 1024)
    df_comp_all["comp_mb"] = df_comp_all["compressed_bytes"] / (1024 * 1024)
    df_merged = pd.merge(df_comp_all, df_decomp_all, on=["filename", "run_id"], suffixes=("_c", "_d"))

    run_labels = [str(r) for r in run_ids]
    multi_run = len(run_ids) > 1

    generated_paths = []

    # --------------------------------------------------------------------------
    # CHART 1: Summary Statistics Table (per run)
    # Mean / Median / Std / Min / Max of CR, BPP, Speed
    # --------------------------------------------------------------------------
    metrics = {
        "Compression Ratio (x)": "compression_ratio",
        "Space Saving (%)": "space_saving_pct",
        "BPP": "bits_per_pixel",
        "Compress Speed (MB/s)": "compress_speed_mb_s",
        "Decompress Speed (MB/s)": "decompress_speed_mb_s",
    }

    summary_rows = []
    for run_id in run_ids:
        df_c = df_comp_all[df_comp_all["run_id"] == str(run_id)]
        df_d = df_decomp_all[df_decomp_all["run_id"] == str(run_id)] if df_decomp_all is not None else None
        row = {"Run": str(run_id), "N Images": len(df_c)}
        for label, col in metrics.items():
            if col in df_c.columns:
                src = df_c[col]
            elif df_d is not None and col in df_d.columns:
                src = df_d[col]
            else:
                continue
            row[f"{label} Mean"] = round(src.mean(), 3)
            row[f"{label} Median"] = round(src.median(), 3)
            row[f"{label} Std"] = round(src.std(), 3)
            row[f"{label} Min"] = round(src.min(), 3)
            row[f"{label} Max"] = round(src.max(), 3)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(charts_dir, "summary_statistics.csv"), index=False)

    # --------------------------------------------------------------------------
    # CHART 2: Distribution of Compression Ratio (KDE + Histogram)
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=CHART_DPI)
    for run_id in run_ids:
        data = df_comp_all[df_comp_all["run_id"] == str(run_id)]["compression_ratio"]
        sns.histplot(data, kde=True, bins=30, label=f"Run {run_id}", ax=ax, alpha=0.4, stat="density")
    ax.set_title(f"Distribution of Compression Ratio — {algo_name.upper()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Compression Ratio (x)")
    ax.set_ylabel("Density")
    ax.legend(title="Run")
    plt.tight_layout()
    p = os.path.join(charts_dir, "2_distribution_compression_ratio.png")
    fig.savefig(p); plt.close(fig)
    generated_paths.append(p)

    # --------------------------------------------------------------------------
    # CHART 3: Distribution of Compression Time (KDE)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=CHART_DPI)
    fig.suptitle(f"Processing Time Distributions — {algo_name.upper()}", fontsize=13, fontweight="bold")

    for run_id in run_ids:
        df_c = df_comp_all[df_comp_all["run_id"] == str(run_id)]
        df_d = df_decomp_all[df_decomp_all["run_id"] == str(run_id)] if df_decomp_all is not None else None
        sns.histplot(df_c["avg_compress_time_sec"], kde=True, bins=30, ax=axes[0],
                     label=f"Run {run_id}", alpha=0.4, stat="density")
        if df_d is not None:
            sns.histplot(df_d["avg_decompress_time_sec"], kde=True, bins=30, ax=axes[1],
                         label=f"Run {run_id}", alpha=0.4, stat="density")

    axes[0].set_title("Compression Time (s) Distribution", fontweight="bold")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Density")
    axes[0].legend(title="Run")
    axes[1].set_title("Decompression Time (s) Distribution", fontweight="bold")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Density")
    axes[1].legend(title="Run")
    plt.tight_layout()
    p = os.path.join(charts_dir, "3_distribution_processing_time.png")
    fig.savefig(p); plt.close(fig)
    generated_paths.append(p)

    # --------------------------------------------------------------------------
    # CHART 4: Box Plot — CR & Space Saving across runs
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=CHART_DPI)
    fig.suptitle(f"Compression Performance Box Plots — {algo_name.upper()}", fontsize=13, fontweight="bold")

    hue_col = "run_id" if multi_run else None
    sns.boxplot(data=df_comp_all, x="run_id", y="compression_ratio",
                ax=axes[0], palette="muted", linewidth=1.2)
    axes[0].set_title("Compression Ratio (CR) Distribution", fontweight="bold")
    axes[0].set_xlabel("Run ID")
    axes[0].set_ylabel("Compression Ratio (x)")

    sns.boxplot(data=df_comp_all, x="run_id", y="space_saving_pct",
                ax=axes[1], palette="muted", linewidth=1.2)
    axes[1].set_title("Space Saving (%) Distribution", fontweight="bold")
    axes[1].set_xlabel("Run ID")
    axes[1].set_ylabel("Space Saving (%)")

    plt.tight_layout()
    p = os.path.join(charts_dir, "4_boxplot_cr_space_saving.png")
    fig.savefig(p); plt.close(fig)
    generated_paths.append(p)

    # --------------------------------------------------------------------------
    # CHART 5: Scatter Plot — Original File Size (MB) vs Compression Ratio
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=CHART_DPI)
    scatter = sns.scatterplot(data=df_comp_all, x="orig_mb", y="compression_ratio",
                               hue="run_id", alpha=0.5, s=25, ax=ax, palette="tab10")
    ax.set_title(f"File Size vs Compression Ratio — {algo_name.upper()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Original File Size (MB)")
    ax.set_ylabel("Compression Ratio (x)")
    ax.legend(title="Run ID", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.tight_layout()
    p = os.path.join(charts_dir, "5_scatter_size_vs_cr.png")
    fig.savefig(p); plt.close(fig)
    generated_paths.append(p)

    # --------------------------------------------------------------------------
    # CHART 6: Box Plot — Processing Speed (compress vs decompress)
    # --------------------------------------------------------------------------
    if df_decomp_all is not None:
        df_speed = pd.merge(df_comp_all[["filename", "run_id", "compress_speed_mb_s"]],
                            df_decomp_all[["filename", "run_id", "decompress_speed_mb_s"]],
                            on=["filename", "run_id"])
        df_speed_melt = df_speed.melt(id_vars=["filename", "run_id"],
                                       value_vars=["compress_speed_mb_s", "decompress_speed_mb_s"],
                                       var_name="Operation", value_name="Speed (MB/s)")
        df_speed_melt["Operation"] = df_speed_melt["Operation"].str.replace("_speed_mb_s", "").str.replace("_", " ").str.title()

        fig, ax = plt.subplots(figsize=(12, 5), dpi=CHART_DPI)
        sns.boxplot(data=df_speed_melt, x="run_id", y="Speed (MB/s)", hue="Operation",
                    ax=ax, palette={"Compress": "#DD8452", "Decompress": "#4C72B0"}, linewidth=1.2)
        ax.set_title(f"Processing Speed Box Plot — {algo_name.upper()}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Run ID")
        ax.set_ylabel("Speed (MB / second)")
        ax.legend(title="Operation")
        plt.tight_layout()
        p = os.path.join(charts_dir, "6_boxplot_speed.png")
        fig.savefig(p); plt.close(fig)
        generated_paths.append(p)

    # ==========================================================================
    # RQ1: PHÂN TÍCH THEO LOẠI ẢNH (image_group)
    # Trả lời: "Đặc điểm ảnh ảnh hưởng thế nào đến tỷ lệ nén?"
    # ==========================================================================

    # Trích cột image_group từ filename (fallback nếu CSV cũ chưa có cột này)
    if "image_group" not in df_comp_all.columns:
        df_comp_all["image_group"] = df_comp_all["filename"].apply(
            lambda f: f.split("__")[0] if "__" in str(f) else "unknown"
        )

    # Chỉ giữ các nhóm hợp lệ (loại bỏ 'unknown' nếu có)
    df_rq1 = df_comp_all[df_comp_all["image_group"] != "unknown"].copy()
    has_rq1 = len(df_rq1) > 0

    GROUP_PALETTE = {
        "1_Natural_Complex":  "#4C72B0",
        "2_Flat_Graphics":    "#55A868",
        "3_Textures_Noisy":   "#C44E52",
        "4_Documents_Text":   "#DD8452",
    }
    GROUP_LABELS = {
        "1_Natural_Complex":  "Natural\nComplex",
        "2_Flat_Graphics":    "Flat\nGraphics",
        "3_Textures_Noisy":   "Textures\nNoisy",
        "4_Documents_Text":   "Documents\nText",
    }

    if has_rq1:
        group_order = [g for g in GROUP_PALETTE if g in df_rq1["image_group"].unique()]

        # ----------------------------------------------------------------------
        # CHART 8 (RQ1): Grouped Bar — Mean CR ± std theo loại ảnh
        # ----------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=CHART_DPI)
        fig.suptitle(
            f"[RQ1] Compression Ratio by Image Type — {algo_name.upper()}",
            fontsize=14, fontweight="bold"
        )

        means_by_group = [df_rq1[df_rq1["image_group"] == g]["compression_ratio"].mean() for g in group_order]
        stds_by_group  = [df_rq1[df_rq1["image_group"] == g]["compression_ratio"].std()  for g in group_order]
        colors = [GROUP_PALETTE.get(g, "#888888") for g in group_order]
        xlabels = [GROUP_LABELS.get(g, g) for g in group_order]

        bars = axes[0].bar(xlabels, means_by_group, yerr=stds_by_group,
                           capsize=6, color=colors, alpha=0.88, edgecolor="white", linewidth=1.2)
        axes[0].set_title("Mean Compression Ratio (CR ± std) by Image Type", fontweight="bold")
        axes[0].set_xlabel("Image Type")
        axes[0].set_ylabel("Compression Ratio (x)")
        for i, (m, s) in enumerate(zip(means_by_group, stds_by_group)):
            axes[0].text(i, m + s + max(means_by_group) * 0.02,
                         f"{m:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

        means_sav = [df_rq1[df_rq1["image_group"] == g]["space_saving_pct"].mean() for g in group_order]
        stds_sav  = [df_rq1[df_rq1["image_group"] == g]["space_saving_pct"].std()  for g in group_order]
        axes[1].bar(xlabels, means_sav, yerr=stds_sav,
                    capsize=6, color=colors, alpha=0.88, edgecolor="white", linewidth=1.2)
        axes[1].set_title("Mean Space Saving (% ± std) by Image Type", fontweight="bold")
        axes[1].set_xlabel("Image Type")
        axes[1].set_ylabel("Space Saving (%)")
        axes[1].set_ylim(0, 110)
        for i, (m, s) in enumerate(zip(means_sav, stds_sav)):
            axes[1].text(i, min(m + s + 2, 108), f"{m:.1f}%",
                         ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        p = os.path.join(charts_dir, "8_rq1_cr_by_image_type.png")
        fig.savefig(p); plt.close(fig)
        generated_paths.append(p)

        # ----------------------------------------------------------------------
        # CHART 9 (RQ1): Box Plot — Phân phối CR theo loại ảnh
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 6), dpi=CHART_DPI)
        sns.boxplot(
            data=df_rq1, x="image_group", y="compression_ratio",
            order=group_order, palette=GROUP_PALETTE,
            ax=ax, linewidth=1.2, width=0.5
        )
        ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in group_order])
        ax.set_title(
            f"[RQ1] Compression Ratio Distribution by Image Type — {algo_name.upper()}",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Image Type")
        ax.set_ylabel("Compression Ratio (x)")

        # Thêm annotation median trên mỗi box
        for i, g in enumerate(group_order):
            med = df_rq1[df_rq1["image_group"] == g]["compression_ratio"].median()
            ax.text(i, med, f" {med:.1f}x", ha="left", va="center",
                    fontsize=8, color="white", fontweight="bold")

        plt.tight_layout()
        p = os.path.join(charts_dir, "9_rq1_boxplot_cr_by_type.png")
        fig.savefig(p); plt.close(fig)
        generated_paths.append(p)

        # ----------------------------------------------------------------------
        # CHART 10 (RQ1): Scatter — Pixel Std Dev (proxy entropy) vs CR
        # Biến x: độ lệch chuẩn pixel (càng cao → ảnh càng nhiễu → entropy cao)
        # Biến y: Compression Ratio (càng thấp khi entropy cao)
        # ----------------------------------------------------------------------
        if "pixel_std" in df_rq1.columns:
            fig, ax = plt.subplots(figsize=(11, 7), dpi=CHART_DPI)
            for g in group_order:
                sub = df_rq1[df_rq1["image_group"] == g]
                ax.scatter(
                    sub["pixel_std"], sub["compression_ratio"],
                    label=GROUP_LABELS.get(g, g),
                    color=GROUP_PALETTE.get(g, "#888888"),
                    alpha=0.55, s=30, edgecolors="none"
                )

            # Đường xu hướng tổng thể (polynomial fit bậc 2)
            x_all = df_rq1["pixel_std"].values
            y_all = df_rq1["compression_ratio"].values
            if len(x_all) > 10:
                try:
                    z = np.polyfit(x_all, y_all, 2)
                    p_fit = np.poly1d(z)
                    x_fit = np.linspace(x_all.min(), x_all.max(), 200)
                    ax.plot(x_fit, p_fit(x_fit), "k--", linewidth=1.5,
                            alpha=0.6, label="Trend (poly-2)")
                except Exception:
                    pass

            ax.set_title(
                f"[RQ1] Image Entropy (Pixel Std Dev) vs Compression Ratio — {algo_name.upper()}\n"
                "(Higher Std Dev → More Noise → Higher Entropy → Lower CR)",
                fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Pixel Std Dev (Shannon Entropy Proxy)")
            ax.set_ylabel("Compression Ratio (x)")
            ax.legend(title="Image Type", bbox_to_anchor=(1.01, 1), borderaxespad=0)
            plt.tight_layout()
            p = os.path.join(charts_dir, "10_rq1_scatter_entropy_vs_cr.png")
            fig.savefig(p); plt.close(fig)
            generated_paths.append(p)

        # ----------------------------------------------------------------------
        # CHART 11 (RQ1): Scatter — Tốc độ nén vs CR (trade-off speed ↔ efficiency)
        # Mỗi điểm = 1 file, tô màu theo loại ảnh
        # → Thấy rõ loại ảnh nào vừa nén nhanh vừa đạt CR cao
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 7), dpi=CHART_DPI)
        for g in group_order:
            sub = df_rq1[df_rq1["image_group"] == g]
            ax.scatter(
                sub["compress_speed_mb_s"], sub["compression_ratio"],
                label=GROUP_LABELS.get(g, g),
                color=GROUP_PALETTE.get(g, "#888888"),
                alpha=0.55, s=30, edgecolors="none"
            )

        # Vẽ thêm centroid (tâm) của mỗi nhóm
        for g in group_order:
            sub = df_rq1[df_rq1["image_group"] == g]
            cx = sub["compress_speed_mb_s"].mean()
            cy = sub["compression_ratio"].mean()
            ax.scatter(cx, cy, color=GROUP_PALETTE.get(g, "#888"),
                       s=180, marker="*", edgecolors="black", linewidth=0.8, zorder=5)
            ax.annotate(GROUP_LABELS.get(g, g).replace("\n", " "),
                        (cx, cy), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, fontweight="bold",
                        color=GROUP_PALETTE.get(g, "#333"))

        ax.set_title(
            f"[RQ1] Trade-off: Compression Speed vs Compression Ratio — {algo_name.upper()}\n"
            "(★ = Average Centroid of Each Group)",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Compression Speed (MB/s)")
        ax.set_ylabel("Compression Ratio (x)")
        ax.legend(title="Image Type", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.tight_layout()
        p = os.path.join(charts_dir, "11_rq1_scatter_speed_vs_cr_by_type.png")
        fig.savefig(p); plt.close(fig)
        generated_paths.append(p)

        # ----------------------------------------------------------------------
        # CHART 12 (RQ1): Grouped Bar — Compress vs Decompress Speed
        # ----------------------------------------------------------------------
        if df_decomp_all is not None:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=CHART_DPI)
            
            comp_speeds = []
            decomp_speeds = []
            
            for g in group_order:
                dc = df_rq1[df_rq1["image_group"] == g]
                comp_speeds.append(dc["compress_speed_mb_s"].mean() if not dc.empty else 0)
                
                dd = df_decomp_all[df_decomp_all["filename"].str.startswith(g)]
                decomp_speeds.append(dd["decompress_speed_mb_s"].mean() if not dd.empty else 0)
            
            x = np.arange(len(group_order))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, comp_speeds, width, label='Compress Speed', color='#DD8452', edgecolor="white", linewidth=1.2)
            bars2 = ax.bar(x + width/2, decomp_speeds, width, label='Decompress Speed', color='#4C72B0', edgecolor="white", linewidth=1.2)
            
            ax.set_title(f"Compress vs Decompress Speed by Image Type — {algo_name.upper()}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Image Type")
            ax.set_ylabel("Processing Speed (MB/s)")
            ax.set_xticks(x)
            ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in group_order])
            ax.legend(title="Action")
            
            # Data labels
            for b in bars1:
                yval = b.get_height()
                ax.text(b.get_x() + b.get_width()/2.0, yval, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight="bold", color='#333')
            for b in bars2:
                yval = b.get_height()
                ax.text(b.get_x() + b.get_width()/2.0, yval, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight="bold", color='#333')
            
            plt.tight_layout()
            p = os.path.join(charts_dir, "12_rq1_grouped_bar_speed.png")
            fig.savefig(p); plt.close(fig)
            generated_paths.append(p)

        # ----------------------------------------------------------------------
        # BẢNG TỔNG HỢP RQ1: Mean CR, BPP, speed theo loại ảnh
        # Dùng cho mục 5.1 "Kết quả thực nghiệm tổng quan"
        # ----------------------------------------------------------------------
        summary_group_rows = []
        for g in group_order:
            dc = df_rq1[df_rq1["image_group"] == g]
            dd = df_decomp_all[df_decomp_all["filename"].str.startswith(g)] \
                 if df_decomp_all is not None else None
            row = {
                "image_group": g,
                "n_images": len(dc),
                "cr_mean": round(dc["compression_ratio"].mean(), 3),
                "cr_median": round(dc["compression_ratio"].median(), 3),
                "cr_std": round(dc["compression_ratio"].std(), 3),
                "space_saving_mean_pct": round(dc["space_saving_pct"].mean(), 2),
                "bpp_mean": round(dc["bits_per_pixel"].mean(), 3),
                "compress_speed_mean_mb_s": round(dc["compress_speed_mb_s"].mean(), 2),
                "avg_compress_time_sec": round(dc["avg_compress_time_sec"].mean(), 4),
            }
            if dd is not None and len(dd) > 0:
                row["decompress_speed_mean_mb_s"] = round(dd["decompress_speed_mb_s"].mean(), 2)
                row["avg_decompress_time_sec"] = round(dd["avg_decompress_time_sec"].mean(), 4)
            summary_group_rows.append(row)

        df_group_summary = pd.DataFrame(summary_group_rows)
        group_csv = os.path.join(charts_dir, "rq1_summary_by_image_type.csv")
        df_group_summary.to_csv(group_csv, index=False, encoding="utf-8")
        print(f"[+] Saved RQ1 summary table: {group_csv}")

    else:
        print("[WARNING] Cannot find image_group data. "
              "Run run.py again to create new CSV.")

    print(f"\n[+] Generated {len(generated_paths)} charts and saved to '{charts_dir}':")
    for i, path in enumerate(generated_paths, 1):
        print(f"  {i}. {path}")
    print(f"\n[+] Summary statistics table: {os.path.join(charts_dir, 'summary_statistics.csv')}\n")


def generate_comparison_charts(
    algo_names: list[str],
    run_ids: list[int],
    base_input_dir: str = "output_not_commit",
    base_plot_dir: str = "output",
):
    """Generate fair comparison tables and charts for two or more codecs."""
    if len(algo_names) < 2:
        raise ValueError("Comparison requires at least two algorithms.")

    merged_by_algorithm = {}
    for algo_name in algo_names:
        df_comp, df_decomp = _load_all_runs(base_input_dir, algo_name, run_ids)
        if df_comp is None or df_decomp is None:
            print(f"[WARNING] Missing reports for '{algo_name}', skipping it.")
            continue

        merged = pd.merge(
            df_comp,
            df_decomp,
            on=["filename", "run_id"],
            suffixes=("_compress", "_decompress"),
        )
        merged["algorithm"] = algo_name
        merged_by_algorithm[algo_name] = merged

    if len(merged_by_algorithm) < 2:
        print("[ERROR] At least two algorithms with complete reports are required.")
        return

    # Keep only image/run pairs present for every codec so the comparison is fair.
    common_keys = None
    for frame in merged_by_algorithm.values():
        keys = set(zip(frame["filename"], frame["run_id"]))
        common_keys = keys if common_keys is None else common_keys & keys

    if not common_keys:
        print("[ERROR] The algorithm reports do not contain common image/run pairs.")
        return

    comparison_frames = []
    for algo_name, frame in merged_by_algorithm.items():
        row_keys = list(zip(frame["filename"], frame["run_id"]))
        filtered = frame[[key in common_keys for key in row_keys]].copy()
        filtered["algorithm"] = algo_name
        comparison_frames.append(filtered)

    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison["algorithm_label"] = comparison["algorithm"].str.upper()
    comparison["is_lossless_numeric"] = (
        comparison["is_lossless"].astype(str).str.lower().eq("true").astype(float)
    )

    charts_dir = os.path.join(base_plot_dir, "comparison", "charts")
    os.makedirs(charts_dir, exist_ok=True)

    summary_rows = []
    for algo_name, frame in comparison.groupby("algorithm", sort=False):
        summary_rows.append({
            "algorithm": algo_name,
            "samples": len(frame),
            "unique_images": frame["filename"].nunique(),
            "original_bytes_mean": frame["original_bytes"].mean(),
            "compressed_bytes_mean": frame["compressed_bytes_compress"].mean(),
            "compression_ratio_mean": frame["compression_ratio"].mean(),
            "compression_ratio_median": frame["compression_ratio"].median(),
            "compression_ratio_std": frame["compression_ratio"].std(),
            "space_saving_mean_pct": frame["space_saving_pct"].mean(),
            "bits_per_pixel_mean": frame["bits_per_pixel"].mean(),
            "compress_time_mean_sec": frame["avg_compress_time_sec"].mean(),
            "decompress_time_mean_sec": frame["avg_decompress_time_sec"].mean(),
            "compress_speed_mean_mb_s": frame["compress_speed_mb_s"].mean(),
            "decompress_speed_mean_mb_s": frame["decompress_speed_mb_s"].mean(),
            "lossless_rate_pct": frame["is_lossless_numeric"].mean() * 100.0,
            "mse_mean": frame["mse"].mean(),
        })

    summary = pd.DataFrame(summary_rows).round(6)
    summary.to_csv(
        os.path.join(charts_dir, "comparison_summary.csv"),
        index=False,
        encoding="utf-8",
    )
    comparison.to_csv(
        os.path.join(charts_dir, "comparison_details.csv"),
        index=False,
        encoding="utf-8",
    )

    if "image_group" in comparison.columns:
        group_summary = comparison.groupby(
            ["algorithm", "image_group"], as_index=False
        ).agg(
            images=("filename", "nunique"),
            original_bytes_mean=("original_bytes", "mean"),
            compressed_bytes_mean=("compressed_bytes_compress", "mean"),
            compression_ratio_mean=("compression_ratio", "mean"),
            space_saving_mean_pct=("space_saving_pct", "mean"),
            bits_per_pixel_mean=("bits_per_pixel", "mean"),
            compress_time_mean_sec=("avg_compress_time_sec", "mean"),
            decompress_time_mean_sec=("avg_decompress_time_sec", "mean"),
            lossless_rate=("is_lossless_numeric", "mean"),
            mse_mean=("mse", "mean"),
            pixel_std_mean=("pixel_std", "mean"),
        )
        group_summary["lossless_rate_pct"] = group_summary.pop("lossless_rate") * 100.0
        group_summary.round(6).to_csv(
            os.path.join(charts_dir, "comparison_by_image_type.csv"),
            index=False,
            encoding="utf-8",
        )

    labels = summary["algorithm"].str.upper().tolist()
    x = np.arange(len(labels))
    colors = sns.color_palette("Set2", len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=CHART_DPI)
    axes[0].bar(x, summary["compression_ratio_mean"], color=colors)
    axes[0].set_title("Mean Compression Ratio")
    axes[0].set_ylabel("Compression Ratio (x)")
    axes[0].set_xticks(x, labels)
    axes[1].bar(x, summary["space_saving_mean_pct"], color=colors)
    axes[1].set_title("Mean Space Saving")
    axes[1].set_ylabel("Space Saving (%)")
    axes[1].set_xticks(x, labels)
    fig.suptitle("Lossless Compression Efficiency by Algorithm", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "1_compression_efficiency.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=CHART_DPI)
    sns.boxplot(
        data=comparison,
        x="algorithm_label",
        y="compression_ratio",
        hue="algorithm_label",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title("Compression Ratio Distribution by Algorithm", fontweight="bold")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Compression Ratio (x)")
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "2_compression_ratio_boxplot.png"))
    plt.close(fig)

    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6), dpi=CHART_DPI)
    ax.bar(
        x - width / 2,
        summary["compress_time_mean_sec"],
        width,
        label="Compress",
        color="#DD8452",
    )
    ax.bar(
        x + width / 2,
        summary["decompress_time_mean_sec"],
        width,
        label="Decompress",
        color="#4C72B0",
    )
    ax.set_title("Mean Processing Time by Algorithm", fontweight="bold")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x, labels)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "3_processing_time.png"))
    plt.close(fig)

    if "image_group" in comparison.columns:
        grouped = comparison.groupby(
            ["image_group", "algorithm_label"], as_index=False
        )["compression_ratio"].mean()
        fig, ax = plt.subplots(figsize=(13, 6), dpi=CHART_DPI)
        sns.barplot(
            data=grouped,
            x="image_group",
            y="compression_ratio",
            hue="algorithm_label",
            palette="Set2",
            ax=ax,
        )
        ax.set_title("Mean Compression Ratio by Image Type", fontweight="bold")
        ax.set_xlabel("Image Type")
        ax.set_ylabel("Compression Ratio (x)")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(title="Algorithm")
        plt.tight_layout()
        fig.savefig(os.path.join(charts_dir, "4_compression_ratio_by_image_type.png"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=CHART_DPI)
    sns.scatterplot(
        data=comparison,
        x="compress_speed_mb_s",
        y="compression_ratio",
        hue="algorithm_label",
        palette="Set2",
        alpha=0.65,
        ax=ax,
    )
    ax.set_title("Compression Speed vs Compression Ratio", fontweight="bold")
    ax.set_xlabel("Compression Speed (MB/s)")
    ax.set_ylabel("Compression Ratio (x)")
    ax.legend(title="Algorithm")
    plt.tight_layout()
    fig.savefig(os.path.join(charts_dir, "5_speed_vs_compression_ratio.png"))
    plt.close(fig)

    print(
        f"[+] Compared {len(merged_by_algorithm)} algorithms on "
        f"{len(common_keys)} common image/run pairs."
    )
    print(f"[+] Comparison reports and charts: {charts_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate statistical Matplotlib charts from large-scale benchmark reports"
    )
    algorithm_group = parser.add_mutually_exclusive_group()
    algorithm_group.add_argument("--algo", "-a", type=str,
                                 help="Generate the detailed report for one algorithm")
    algorithm_group.add_argument("--algos", type=str, nargs="+",
                                 help="Compare two or more algorithms")
    parser.add_argument("--runs", "-r", type=int, nargs="+", default=_DEFAULT_RUNS,
                        help=f"One or more run IDs to include (e.g. --runs 1 2 3 4 5). Default: {_DEFAULT_RUNS}")
    parser.add_argument("--input-dir", "-i", type=str, default="output_not_commit",
                        help="Input directory containing benchmark reports (default: 'output_not_commit')")
    parser.add_argument("--plot-dir", "-p", type=str, default="output",
                        help="Output directory for plot charts (default: 'output')")
    args = parser.parse_args()

    if args.algos:
        generate_comparison_charts(
            algo_names=args.algos,
            run_ids=args.runs,
            base_input_dir=args.input_dir,
            base_plot_dir=args.plot_dir,
        )
    else:
        generate_benchmark_charts(
            algo_name=args.algo or "deflate",
            run_ids=args.runs,
            base_input_dir=args.input_dir,
            base_plot_dir=args.plot_dir,
        )
