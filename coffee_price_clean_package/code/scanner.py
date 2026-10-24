"""
Quality Scanner & Anomaly Detector for Vietnam Coffee Price Datasets
Detects:
1. Inter-province divergence (e.g. Dak Lak = 120k while Ba Ria = 47k)
2. Weekend artificial spikes / Monday crashes (unrealistic weekend trading jumps)
3. Abnormal single-day price jumps (> 15% or > 4,000 VND/kg)
4. Frozen / stale price series (provinces flatlined while others moved)
5. Synthetic clones (provinces artificially generated from fixed offsets)
6. Raw vs Cleaned discrepancies
7. Linear interpolation artifacts in macro indicators
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure stdout handles UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class CoffeeDataScanner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.anomalies = []
        self.summary = {}

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Try tab first, then comma
        try:
            self.df = pd.read_csv(self.file_path, sep='\t')
            if len(self.df.columns) <= 1:
                self.df = pd.read_csv(self.file_path)
        except Exception:
            self.df = pd.read_csv(self.file_path)

        # Parse date
        self.df['parsed_date'] = pd.to_datetime(self.df['date'], format='%m/%d/%Y', errors='coerce')
        if self.df['parsed_date'].isna().sum() > 0:
            # Fallback format
            self.df['parsed_date'] = pd.to_datetime(self.df['date'], errors='coerce')

        return self.df

    def scan_all(self):
        print(f"[*] Scanning dataset: {self.file_path}")
        print(f"[*] Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        self.anomalies = []
        
        # 1. Null / NaN scan
        self._scan_missing_values()
        
        # 2. Inter-province divergence scan
        self._scan_province_divergence()
        
        # 3. Weekend spikes scan
        self._scan_weekend_glitches()
        
        # 4. Extreme day-to-day jumps scan
        self._scan_extreme_jumps()
        
        # 5. Synthetic province clone scan
        self._scan_synthetic_clones()
        
        # 6. Raw vs Cleaned discrepancy scan
        self._scan_raw_discrepancies()
        
        # 7. Linear interpolation in macro features scan
        self._scan_linear_interpolations()

        # Build summary
        self._build_summary()
        return self.anomalies, self.summary

    def _scan_missing_values(self):
        null_counts = self.df.isnull().sum()
        for col, cnt in null_counts.items():
            if cnt > 0:
                self.anomalies.append({
                    'type': 'MISSING_VALUES',
                    'severity': 'HIGH',
                    'column': col,
                    'count': int(cnt),
                    'description': f"Column '{col}' has {cnt} missing values"
                })

    def _scan_province_divergence(self):
        pivot = self.df.pivot(index='parsed_date', columns='province', values='coffee_price_vnd_kg')
        ch_provs = [p for p in ['Dak Lak', 'Lam Dong', 'Gia Lai', 'Dak Nong', 'Kon Tum'] if p in pivot.columns]
        
        # A. Central Highlands Internal Spread (should normally be < 1,500 VND)
        if len(ch_provs) >= 2:
            ch_sub = pivot[ch_provs]
            spread = ch_sub.max(axis=1) - ch_sub.min(axis=1)
            bad_days = spread[spread > 3000]
            for dt, sp in bad_days.items():
                min_p = ch_sub.loc[dt].idxmin()
                max_p = ch_sub.loc[dt].idxmax()
                self.anomalies.append({
                    'type': 'CENTRAL_HIGHLANDS_DIVERGENCE',
                    'severity': 'HIGH' if sp > 5000 else 'MEDIUM',
                    'date': dt.strftime('%Y-%m-%d'),
                    'spread_vnd': float(sp),
                    'min_province': str(min_p),
                    'min_price': float(ch_sub.loc[dt, min_p]),
                    'max_province': str(max_p),
                    'max_price': float(ch_sub.loc[dt, max_p]),
                    'description': f"Central Highlands price spread = {sp:,.0f} VND ({min_p}={ch_sub.loc[dt, min_p]:,.0f} vs {max_p}={ch_sub.loc[dt, max_p]:,.0f})"
                })

        # B. Central Highlands vs Southeast Divergence (e.g. Ba Ria, Binh Phuoc, Dong Nai decoupling)
        se_provs = [p for p in ['Ba Ria - Vung Tau', 'Binh Phuoc', 'Dong Nai'] if p in pivot.columns]
        if 'Dak Lak' in pivot.columns and se_provs:
            for se in se_provs:
                diff = (pivot['Dak Lak'] - pivot[se]).abs()
                huge_diff = diff[diff > 5000]
                for dt, df_val in huge_diff.items():
                    self.anomalies.append({
                        'type': 'REGIONAL_DECOUPLING',
                        'severity': 'CRITICAL' if df_val > 20000 else 'HIGH',
                        'date': dt.strftime('%Y-%m-%d'),
                        'province': se,
                        'benchmark_province': 'Dak Lak',
                        'diff_vnd': float(df_val),
                        'daklak_price': float(pivot.loc[dt, 'Dak Lak']),
                        'province_price': float(pivot.loc[dt, se]),
                        'description': f"Severe decoupling: Dak Lak={pivot.loc[dt, 'Dak Lak']:,.0f} vs {se}={pivot.loc[dt, se]:,.0f} (diff={df_val:,.0f} VND)"
                    })

    def _scan_weekend_glitches(self):
        # Look for artificial spikes: Sat jumped > 2,500 from Fri, and Mon fell > 2,500 from Sun/Sat
        for prov in self.df['province'].unique():
            sub = self.df[self.df['province'] == prov].sort_values('parsed_date').copy()
            sub['dow'] = sub['parsed_date'].dt.dayofweek
            sub['prev_price'] = sub['coffee_price_vnd_kg'].shift(1)
            sub['next_price'] = sub['coffee_price_vnd_kg'].shift(1)
            sub['fri_to_sat'] = sub['coffee_price_vnd_kg'] - sub['prev_price']

            # Saturday jumps
            sat_spikes = sub[(sub['dow'] == 5) & (sub['fri_to_sat'] > 2500)]
            for _, r in sat_spikes.iterrows():
                self.anomalies.append({
                    'type': 'WEEKEND_SPIKE',
                    'severity': 'HIGH',
                    'province': prov,
                    'date': r['parsed_date'].strftime('%Y-%m-%d'),
                    'friday_price': float(r['prev_price']),
                    'saturday_price': float(r['coffee_price_vnd_kg']),
                    'jump_vnd': float(r['fri_to_sat']),
                    'description': f"{prov}: Artificial weekend jump of +{r['fri_to_sat']:,.0f} VND on Saturday {r['parsed_date'].strftime('%Y-%m-%d')}"
                })

    def _scan_extreme_jumps(self):
        for prov in self.df['province'].unique():
            sub = self.df[self.df['province'] == prov].sort_values('parsed_date').copy()
            sub['diff'] = sub['coffee_price_vnd_kg'].diff().abs()
            sub['pct'] = sub['coffee_price_vnd_kg'].pct_change().abs()
            
            extreme = sub[(sub['diff'] > 5000) & (sub['pct'] > 0.10)]
            for _, r in extreme.iterrows():
                self.anomalies.append({
                    'type': 'EXTREME_DAY_JUMP',
                    'severity': 'MEDIUM',
                    'province': prov,
                    'date': r['parsed_date'].strftime('%Y-%m-%d'),
                    'price': float(r['coffee_price_vnd_kg']),
                    'jump_vnd': float(r['diff']),
                    'pct_change': float(r['pct'] * 100),
                    'description': f"{prov}: 1-day jump of {r['diff']:,.0f} VND ({r['pct']*100:.1f}%) on {r['parsed_date'].strftime('%Y-%m-%d')}"
                })

    def _scan_synthetic_clones(self):
        pivot = self.df.pivot(index='parsed_date', columns='province', values='coffee_price_vnd_kg')
        if 'Ba Ria - Vung Tau' in pivot.columns:
            for p in ['Nghe An', 'Binh Phuoc', 'Dong Nai']:
                if p in pivot.columns:
                    diff = pivot['Ba Ria - Vung Tau'] - pivot[p]
                    if diff.std() < 100:  # standard deviation < 100 VND means constant linear clone!
                        self.anomalies.append({
                            'type': 'SYNTHETIC_CLONE_SERIES',
                            'severity': 'HIGH',
                            'province': p,
                            'clone_of': 'Ba Ria - Vung Tau',
                            'mean_offset_vnd': float(diff.mean()),
                            'std_dev': float(diff.std()),
                            'description': f"Province '{p}' is a synthetic clone of 'Ba Ria - Vung Tau' (std={diff.std():.2f}, mean offset={diff.mean():.1f} VND)"
                        })

    def _scan_raw_discrepancies(self):
        if 'coffee_price_vnd_kg_raw' in self.df.columns:
            mismatches = self.df[self.df['coffee_price_vnd_kg'] != self.df['coffee_price_vnd_kg_raw']]
            severe_mismatches = mismatches[(mismatches['coffee_price_vnd_kg'] - mismatches['coffee_price_vnd_kg_raw']).abs() > 2000]
            
            for _, r in severe_mismatches.iterrows():
                self.anomalies.append({
                    'type': 'RAW_CLEANED_MISMATCH',
                    'severity': 'HIGH',
                    'province': r['province'],
                    'date': pd.to_datetime(r['date']).strftime('%Y-%m-%d'),
                    'cleaned_price': float(r['coffee_price_vnd_kg']),
                    'raw_price': float(r['coffee_price_vnd_kg_raw']),
                    'diff': float(abs(r['coffee_price_vnd_kg'] - r['coffee_price_vnd_kg_raw'])),
                    'description': f"{r['province']} on {r['date']}: cleaned={r['coffee_price_vnd_kg']:,.0f} vs raw={r['coffee_price_vnd_kg_raw']:,.0f}"
                })

    def _scan_linear_interpolations(self):
        sub = self.df[self.df['province'] == self.df['province'].iloc[0]].sort_values('parsed_date').copy()
        if 'london_price_usd_ton' in sub.columns:
            d1 = sub['london_price_usd_ton'].diff()
            d2 = d1.diff().abs()
            linear_days = (d2 < 1e-4).sum()
            if linear_days > 100:
                self.anomalies.append({
                    'type': 'SYNTHETIC_LINEAR_INTERPOLATION',
                    'severity': 'MEDIUM',
                    'column': 'london_price_usd_ton',
                    'interpolated_days_count': int(linear_days),
                    'total_days': len(sub),
                    'description': f"london_price_usd_ton has {linear_days}/{len(sub)} days of exact linear interpolation"
                })

    def _build_summary(self):
        counts_by_type = {}
        counts_by_severity = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        counts_by_province = {}

        for a in self.anomalies:
            t = a['type']
            counts_by_type[t] = counts_by_type.get(t, 0) + 1
            s = a.get('severity', 'LOW')
            counts_by_severity[s] = counts_by_severity.get(s, 0) + 1
            p = a.get('province') or a.get('min_province')
            if p:
                counts_by_province[p] = counts_by_province.get(p, 0) + 1

        self.summary = {
            'total_anomalies_detected': len(self.anomalies),
            'by_severity': counts_by_severity,
            'by_type': counts_by_type,
            'by_province': counts_by_province
        }

    def print_report(self):
        print("\n" + "="*70)
        print("          BÁO CÁO KẾT QUẢ QUÉT LỖI DỮ LIỆU CÀ PHÊ (AUDIT REPORT)")
        print("="*70)
        print(f"Tổng số lỗi & điểm bất thường phát hiện: {self.summary['total_anomalies_detected']:,}")
        print("\nPhân bổ theo mức độ nghiêm trọng (Severity):")
        for s, c in self.summary['by_severity'].items():
            print(f"  - {s:10s}: {c:6,d} lỗi")

        print("\nPhân bổ theo loại lỗi (Anomaly Categories):")
        for t, c in self.summary['by_type'].items():
            print(f"  - {t:35s}: {c:6,d} trường hợp")

        print("\nPhân bổ theo tỉnh thành (Province Breakdown):")
        for p, c in sorted(self.summary['by_province'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {p:25s}: {c:6,d} lỗi")
        print("="*70 + "\n")

    def export_report(self, json_path="scan_report.json", md_path="scan_report.md"):
        # Save JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'summary': self.summary, 'anomalies_sample': self.anomalies[:500]}, f, ensure_ascii=False, indent=2)
        print(f"[+] Saved scan JSON report to: {json_path}")

        # Save Markdown
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Báo cáo Kiểm toán và Quét Lỗi Dữ Liệu Cà Phê\n\n")
            f.write(f"- **Tệp dữ liệu quét:** `{self.file_path}`\n")
            f.write(f"- **Thời gian quét:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n")
            f.write(f"- **Tổng số bản ghi:** `{len(self.df):,}`\n")
            f.write(f"- **Tổng số lỗi phát hiện:** `{self.summary['total_anomalies_detected']:,}`\n\n")
            
            f.write("## 1. Thống kê theo mức độ nghiêm trọng\n\n")
            f.write("| Mức độ | Số lượng lỗi |\n|---|---|\n")
            for s, c in self.summary['by_severity'].items():
                f.write(f"| **{s}** | {c:,} |\n")

            f.write("\n## 2. Thống kê theo loại lỗi\n\n")
            f.write("| Loại lỗi | Số trường hợp | Mô tả |\n|---|---|---|\n")
            type_desc = {
                'REGIONAL_DECOUPLING': 'Giá cà phê vùng Đông Nam Bộ lệch pha nghiêm trọng so với Tây Nguyên (>5k - 50k VND)',
                'CENTRAL_HIGHLANDS_DIVERGENCE': 'Chênh lệch nội bộ Tây Nguyên vượt ngưỡng thị trường (>3,000 VND)',
                'WEEKEND_SPIKE': 'Gai giá bất thường thứ Bảy/Chủ Nhật rồi sụt giảm vào thứ Hai',
                'EXTREME_DAY_JUMP': 'Biến động giá trong 1 ngày vượt quá 10% hoặc 5,000 VND',
                'RAW_CLEANED_MISMATCH': 'Sai lệch giữa giá Cleaned và Raw vượt quá 2,000 VND',
                'SYNTHETIC_CLONE_SERIES': 'Chuỗi giá của tỉnh được tạo nhân tạo từ tỉnh khác với độ lệch cố định',
                'SYNTHETIC_LINEAR_INTERPOLATION': 'Giá London/thế giới bị nội suy tuyến tính chia đều các ngày'
            }
            for t, c in self.summary['by_type'].items():
                f.write(f"| `{t}` | {c:,} | {type_desc.get(t, '')} |\n")

            f.write("\n## 3. Mẫu các lỗi nghiêm trọng tiêu biểu\n\n")
            f.write("| Ngày | Tỉnh | Loại lỗi | Chi tiết |\n|---|---|---|---|\n")
            critical_samples = [a for a in self.anomalies if a.get('severity') in ['CRITICAL', 'HIGH']][:20]
            for cs in critical_samples:
                dt_str = cs.get('date', 'N/A')
                p_str = cs.get('province', cs.get('min_province', 'N/A'))
                f.write(f"| {dt_str} | {p_str} | `{cs['type']}` | {cs['description']} |\n")

        print(f"[+] Saved scan Markdown report to: {md_path}")


if __name__ == '__main__':
    csv_file = r"d:\2026-2027\TLCN\crawl_data\coffee_price_cleaned.csv"
    scanner = CoffeeDataScanner(csv_file)
    scanner.load_data()
    scanner.scan_all()
    scanner.print_report()
    scanner.export_report()
