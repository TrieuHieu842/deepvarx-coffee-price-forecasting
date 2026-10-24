"""
Generate High-Quality Visual Charts for Report / Presentation Slides
Creates:
1. chart_before_vs_after_timeline.png: Compares original faulty decoupling with cleaned time series.
2. chart_weekend_spikes_fix.png: Zoom-in comparison on weekend spikes.
3. chart_correlation_matrix.png: Correlation across provinces before vs after.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Configure matplotlib style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#EEEEEE'
plt.rcParams['grid.linestyle'] = '--'

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    if len(df.columns) <= 1:
        df = pd.read_csv(file_path)
    df['parsed_date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    if df['parsed_date'].isna().sum() > 0:
        df['parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

def generate_all_charts():
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def find_file(rel_path):
        candidates = [
            os.path.join(pkg_root, "data", rel_path),
            os.path.join(pkg_root, rel_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path),
            os.path.join(r"d:\2026-2027\TLCN\crawl_data", rel_path)
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return rel_path

    old_file = find_file("coffee_price_cleaned_backup.csv")
    new_file = find_file("coffee_price_cleaned.csv")
    output_dir = os.path.join(pkg_root, "reports", "report_charts")
    os.makedirs(output_dir, exist_ok=True)


    print("[*] Loading datasets for charting...")
    df_old = load_data(old_file)
    df_new = load_data(new_file)

    piv_old = df_old.pivot(index='parsed_date', columns='province', values='coffee_price_vnd_kg')
    piv_new = df_new.pivot(index='parsed_date', columns='province', values='coffee_price_vnd_kg')

    # -------------------------------------------------------------
    # Chart 1: Before vs After Timeline (2020 - 2026)
    # -------------------------------------------------------------
    print("[1/3] Generating Timeline Comparison Chart...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, sharey=True, dpi=300)

    # Subplot 1: Before
    sub_old = piv_old.loc['2020-01-01':'2026-07-21']
    axes[0].plot(sub_old.index, sub_old['Dak Lak'], label='Dak Lak (Benchmark)', color='#D9534F', linewidth=1.8)
    axes[0].plot(sub_old.index, sub_old['Lam Dong'], label='Lam Dong', color='#F0AD4E', linewidth=1.2, alpha=0.8)
    axes[0].plot(sub_old.index, sub_old['Gia Lai'], label='Gia Lai', color='#0275D8', linewidth=1.2, alpha=0.8)
    axes[0].plot(sub_old.index, sub_old['Ba Ria - Vung Tau'], label='Ba Ria - Vung Tau (LỆCH PHA)', color='#5CB85C', linewidth=2.0, linestyle='--')
    axes[0].set_title('A. DỮ LIỆU GỐC TRƯỚC KHI VÁ (Lệch pha nghiêm trọng: Đông Nam Bộ bị bỏ lại ở 40,000 đ/kg)', fontsize=13, weight='bold', color='#B94A48')
    axes[0].set_ylabel('Giá cà phê (VNĐ/kg)', fontsize=11)
    axes[0].grid(True)
    axes[0].legend(loc='upper left', framealpha=0.9)

    # Subplot 2: After
    sub_new = piv_new.loc['2020-01-01':'2026-07-21']
    axes[1].plot(sub_new.index, sub_new['Dak Lak'], label='Dak Lak (Benchmark)', color='#D9534F', linewidth=1.8)
    axes[1].plot(sub_new.index, sub_new['Lam Dong'], label='Lam Dong', color='#F0AD4E', linewidth=1.2, alpha=0.8)
    axes[1].plot(sub_new.index, sub_new['Gia Lai'], label='Gia Lai', color='#0275D8', linewidth=1.2, alpha=0.8)
    axes[1].plot(sub_new.index, sub_new['Ba Ria - Vung Tau'], label='Ba Ria - Vung Tau (ĐÃ ĐỒNG BỘ)', color='#5CB85C', linewidth=1.8)
    axes[1].set_title('B. DỮ LIỆU SAU KHI VÁ (Đồng bộ theo Parity kinh tế và bám sát thực tế thị trường 120,000 đ/kg)', fontsize=13, weight='bold', color='#468847')
    axes[1].set_ylabel('Giá cà phê (VNĐ/kg)', fontsize=11)
    axes[1].set_xlabel('Thời gian (Năm)', fontsize=11)
    axes[1].grid(True)
    axes[1].legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "chart_before_vs_after_timeline.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"      -> Saved {chart1_path}")

    # -------------------------------------------------------------
    # Chart 2: Zoom on Weekend Spikes (March - May 2020)
    # -------------------------------------------------------------
    print("[2/3] Generating Weekend Spikes Zoom Chart...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, dpi=300)

    z_start, z_end = '2020-03-01', '2020-04-30'
    z_old = piv_old.loc[z_start:z_end, 'Dak Lak']
    z_new = piv_new.loc[z_start:z_end, 'Dak Lak']

    # Subplot 1: Old Spikes
    axes[0].plot(z_old.index, z_old.values, marker='o', markersize=4, color='#D9534F', linewidth=1.5, label='Dak Lak (Gốc - Răng cưa giả tạo)')
    axes[0].set_title('Trước khi vá: Gai nhọn thứ Bảy / Sụt giảm thứ Hai', fontsize=12, weight='bold', color='#B94A48')
    axes[0].set_ylabel('Giá (VNĐ/kg)', fontsize=11)
    axes[0].set_xlabel('Ngày giao dịch (03/2020 - 04/2020)', fontsize=11)
    axes[0].grid(True)
    axes[0].legend(loc='upper left')
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    # Subplot 2: Cleaned
    axes[1].plot(z_new.index, z_new.values, marker='s', markersize=4, color='#0275D8', linewidth=1.5, label='Dak Lak (Đã vá - Khử gai mượt mà)')
    axes[1].set_title('Sau khi vá: Khử sạch gai nhọn, bảo toàn xu thế thực tế', fontsize=12, weight='bold', color='#468847')
    axes[1].set_xlabel('Ngày giao dịch (03/2020 - 04/2020)', fontsize=11)
    axes[1].grid(True)
    axes[1].legend(loc='upper left')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "chart_weekend_spikes_fix.png")
    plt.savefig(chart2_path)
    plt.close()
    print(f"      -> Saved {chart2_path}")

    # -------------------------------------------------------------
    # Chart 3: Correlation Matrix Heatmap Before vs After
    # -------------------------------------------------------------
    print("[3/3] Generating Correlation Matrix Chart...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    corr_old = piv_old.corr()
    corr_new = piv_new.corr()

    cax0 = axes[0].imshow(corr_old, cmap='YlOrRd', vmin=0.7, vmax=1.0)
    axes[0].set_title('Tương quan các tỉnh TRƯỚC VÁ (Đông Nam Bộ bị lệch chỉ 0.73)', fontsize=11, weight='bold')
    axes[0].set_xticks(range(len(corr_old.columns)))
    axes[0].set_yticks(range(len(corr_old.columns)))
    axes[0].set_xticklabels(corr_old.columns, rotation=45, ha='right', fontsize=9)
    axes[0].set_yticklabels(corr_old.columns, fontsize=9)
    fig.colorbar(cax0, ax=axes[0], fraction=0.046, pad=0.04)

    cax1 = axes[1].imshow(corr_new, cmap='YlOrRd', vmin=0.7, vmax=1.0)
    axes[1].set_title('Tương quan các tỉnh SAU VÁ (Đồng bộ thị trường chuẩn > 0.99)', fontsize=11, weight='bold')
    axes[1].set_xticks(range(len(corr_new.columns)))
    axes[1].set_yticks(range(len(corr_new.columns)))
    axes[1].set_xticklabels(corr_new.columns, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels(corr_new.columns, fontsize=9)
    fig.colorbar(cax1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "chart_correlation_matrix.png")
    plt.savefig(chart3_path)
    plt.close()
    print(f"      -> Saved {chart3_path}")

    print("\n[+] Hoàn tất tạo biểu đồ báo cáo khoa học chất lượng cao!")

if __name__ == '__main__':
    generate_all_charts()
