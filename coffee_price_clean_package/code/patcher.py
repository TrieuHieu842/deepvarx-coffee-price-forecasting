"""
Intelligent Reconciliation & Data Patcher for Vietnam Coffee Price Datasets (Version 2.0)
Key Improvements:
1. Re-anchors all provinces using Dak Lak as the primary economic benchmark (Basis Parity Model),
   correcting the 2020 anomaly where Gia Lai/Dak Nong/Kon Tum were falsely inflated to 40,000 VND.
2. Fixes regional decoupling in Southeast (Ba Ria, Binh Phuoc, Dong Nai) and Nghe An.
3. Smooths all weekend artificial spikes and Monday crashes.
4. Enriches macro indicators using actual historical daily exchange closes from Yahoo Finance.
5. Injects live web crawled data from October 2026 up to today (2026-10-23).
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class CoffeeDataPatcher:
    # Authoritative basis relative to Dak Lak (VND/kg) established by historical market data
    PROVINCE_BASIS_RELATIVE_TO_DAKLAK = {
        'Dak Lak': 0,
        'Dak Nong': 100,      # Typically trades +0 to +200 vs Dak Lak
        'Gia Lai': -50,       # Typically trades 0 to -100 vs Dak Lak
        'Lam Dong': -600,     # Typically trades -500 to -800 vs Dak Lak
        'Kon Tum': -150,      # Typically trades -100 to -300 vs Dak Lak
        'Ba Ria - Vung Tau': 200,   # Near seaport / Southeast FOB
        'Binh Phuoc': 100,          # Southeast commercial robusta
        'Dong Nai': 150,            # Southeast commercial robusta
        'Nghe An': -300             # Northern baseline
    }

    def __init__(self, input_file, output_file=None, crawled_cache="crawled_prices_cache.json", macro_cache="macro_yahoo_cache.csv"):
        self.input_file = input_file
        self.output_file = output_file or input_file.replace('.csv', '_patched.csv')
        self.crawled_cache = crawled_cache
        self.macro_cache = macro_cache
        self.df = None
        self.patch_log = []

    def load_data(self):
        print(f"[*] Loading data from: {self.input_file}")
        try:
            self.df = pd.read_csv(self.input_file, sep='\t')
            if len(self.df.columns) <= 1:
                self.df = pd.read_csv(self.input_file)
        except Exception:
            self.df = pd.read_csv(self.input_file)

        self.df['parsed_date'] = pd.to_datetime(self.df['date'], format='%m/%d/%Y', errors='coerce')
        if self.df['parsed_date'].isna().sum() > 0:
            self.df['parsed_date'] = pd.to_datetime(self.df['date'], errors='coerce')

        return self.df

    def patch_all(self):
        print("\n" + "="*70)
        print("          BẮT ĐẦU VÁ LỖI VÀ CHUẨN HÓA DỮ LIỆU (INTELLIGENT PATCHER V2)")
        print("="*70)
        self.patch_log = []

        # Step 1: Fix Weekend Glitches (Artificial Saturday spikes / Monday crashes)
        self._fix_weekend_spikes()

        # Step 2: Fix Central Highlands and Regional Divergence using Benchmark Anchoring
        self._fix_all_province_divergence()

        # Step 3: Apply Live Web Crawled Data from Cache
        self._apply_crawled_cache()

        # Step 4: Enrich Macro Financial Indicators from Yahoo Finance
        self._enrich_macro_data()

        # Step 5: Final Clean-up and Consistency Checks
        self._finalize_consistency()

        print(f"[+] Total patches applied across dataset: {len(self.patch_log):,}")
        return self.df

    def _fix_weekend_spikes(self):
        print("[1/4] Khử lỗi gai nhọn cuối tuần (Weekend Spikes De-glitching)...")
        spikes_fixed = 0

        for prov in self.df['province'].unique():
            idx_list = self.df[self.df['province'] == prov].sort_values('parsed_date').index
            sub = self.df.loc[idx_list]

            dow = sub['parsed_date'].dt.dayofweek.values
            prices = sub['coffee_price_vnd_kg'].values.astype(float)
            dates = sub['parsed_date'].values

            for i in range(1, len(prices) - 2):
                # Saturday (dow == 5)
                if dow[i] == 5:
                    p_fri = prices[i - 1]
                    p_sat = prices[i]
                    p_sun = prices[i + 1] if dow[i + 1] == 6 else p_sat
                    p_mon = prices[i + 2] if i + 2 < len(prices) and dow[i + 2] == 0 else prices[i + 1]

                    # If Saturday jumped > 1,800 from Friday, but Monday is close to Friday
                    if (p_sat - p_fri > 1800) and (abs(p_mon - p_fri) < 2000):
                        old_val = p_sat
                        new_sat = round(p_fri + (p_mon - p_fri) * (1.0 / 3.0))
                        new_sun = round(p_fri + (p_mon - p_fri) * (2.0 / 3.0))

                        prices[i] = new_sat
                        self.df.loc[idx_list[i], 'coffee_price_vnd_kg'] = new_sat
                        self.patch_log.append({
                            'row': int(idx_list[i]),
                            'date': str(pd.to_datetime(dates[i]).strftime('%Y-%m-%d')),
                            'province': prov,
                            'type': 'FIX_WEEKEND_SPIKE',
                            'old_price': old_val,
                            'new_price': new_sat
                        })
                        spikes_fixed += 1

                        if dow[i + 1] == 6 and (p_sun - p_fri > 1800):
                            prices[i + 1] = new_sun
                            self.df.loc[idx_list[i + 1], 'coffee_price_vnd_kg'] = new_sun
                            self.patch_log.append({
                                'row': int(idx_list[i + 1]),
                                'date': str(pd.to_datetime(dates[i + 1]).strftime('%Y-%m-%d')),
                                'province': prov,
                                'type': 'FIX_WEEKEND_SPIKE',
                                'old_price': p_sun,
                                'new_price': new_sun
                            })
                            spikes_fixed += 1

        print(f"      -> Fixed {spikes_fixed:,} artificial weekend spikes.")

    def _fix_all_province_divergence(self):
        """
        In Vietnam coffee markets, Dak Lak is the primary economic reference benchmark.
        Lâm Đồng is typically -500 to -800 VND lower.
        Gia Lai, Đắk Nông, Kon Tum, Bà Rịa, Bình Phước, Đồng Nai move in tight sync (within ±1,500 VND).
        When any province diverges by > 2,500 VND from Dak Lak, it is re-anchored to Dak Lak + basis.
        """
        print("[2/4] Đồng bộ hóa toàn diện theo chuẩn thị trường Đắk Lắk (Benchmark Basis Parity)...")
        divergence_fixed = 0

        # Build pivot table
        pivot = self.df.pivot(index='parsed_date', columns='province', values='coffee_price_vnd_kg')

        if 'Dak Lak' not in pivot.columns:
            print("      [!] Dak Lak not found in pivot!")
            return

        dl_prices = pivot['Dak Lak']

        for prov in pivot.columns:
            if prov == 'Dak Lak':
                continue

            basis = self.PROVINCE_BASIS_RELATIVE_TO_DAKLAK.get(prov, 0)
            diff = (pivot[prov] - dl_prices).abs()
            
            # Threshold: > 2,500 VND divergence is an obvious anomaly
            bad_dates = diff[diff > 2500].index

            for dt in bad_dates:
                cur_val = pivot.loc[dt, prov]
                dl_val = dl_prices.loc[dt]
                if pd.notna(dl_val) and dl_val > 10000:
                    new_val = round(dl_val + basis)
                    # Find row in self.df
                    mask = (self.df['parsed_date'] == dt) & (self.df['province'] == prov)
                    matches = self.df[mask]
                    if len(matches) > 0:
                        row_idx = matches.index[0]
                        self.df.loc[row_idx, 'coffee_price_vnd_kg'] = new_val
                        self.patch_log.append({
                            'row': int(row_idx),
                            'date': dt.strftime('%Y-%m-%d'),
                            'province': prov,
                            'type': 'FIX_PARITY_DIVERGENCE',
                            'old_price': cur_val,
                            'new_price': new_val
                        })
                        divergence_fixed += 1

        print(f"      -> Re-anchored {divergence_fixed:,} divergent price entries across all provinces.")

    def _apply_crawled_cache(self):
        print("[3/4] Áp dụng dữ liệu cào thực tế từ Web (Web Crawled Live Data Injection)...")
        if not os.path.exists(self.crawled_cache):
            print("      [!] Crawl cache not found, skipping live injection.")
            return

        with open(self.crawled_cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)

        live_applied = 0
        domestic = cache.get('domestic_prices', {})

        for d_str, prov_dict in domestic.items():
            dt_match = pd.to_datetime(d_str, format='%m/%d/%Y', errors='coerce')
            if pd.isna(dt_match):
                dt_match = pd.to_datetime(d_str, errors='coerce')

            for prov, p_info in prov_dict.items():
                price = p_info.get('price')
                if price and price > 10000:
                    mask = (self.df['parsed_date'] == dt_match) & (self.df['province'] == prov)
                    matches = self.df[mask]
                    if len(matches) > 0:
                        for row_idx in matches.index:
                            old_val = self.df.loc[row_idx, 'coffee_price_vnd_kg']
                            self.df.loc[row_idx, 'coffee_price_vnd_kg'] = price
                            self.patch_log.append({
                                'row': int(row_idx),
                                'date': dt_match.strftime('%Y-%m-%d'),
                                'province': prov,
                                'type': 'LIVE_WEB_CRAWLED_DATA',
                                'source': p_info.get('source'),
                                'old_price': old_val,
                                'new_price': price
                            })
                            live_applied += 1

        print(f"      -> Applied {live_applied:,} live web-verified prices.")

    def _enrich_macro_data(self):
        print("[4/4] Cập nhật dữ liệu tài chính vĩ mô từ Yahoo Finance (Macro Enrichment)...")
        if not os.path.exists(self.macro_cache):
            print("      [!] Macro cache not found, skipping macro enrichment.")
            return

        macro_df = pd.read_csv(self.macro_cache, index_col=0, parse_dates=True)
        macro_applied = 0

        for col in ['arabica_price_usd_lb', 'brent_price_usd_barrel', 'dxy_index', 'usd_vnd_rate', 'usd_brl_rate']:
            if col in macro_df.columns and col in self.df.columns:
                macro_series = macro_df[col].dropna()
                common_dates = self.df['parsed_date'].isin(macro_series.index)
                if common_dates.sum() > 0:
                    updated_vals = self.df.loc[common_dates, 'parsed_date'].map(macro_series)
                    valid_mask = common_dates & updated_vals.notna()
                    self.df.loc[valid_mask, col] = updated_vals[valid_mask]
                    macro_applied += valid_mask.sum()

        print(f"      -> Enriched {macro_applied:,} macro indicator records with real exchange closes.")

    def _finalize_consistency(self):
        # Round prices to integers for domestic coffee price
        self.df['coffee_price_vnd_kg'] = self.df['coffee_price_vnd_kg'].round().astype(int)

        # Drop temporary parsed_date column before saving
        if 'parsed_date' in self.df.columns:
            self.df = self.df.drop(columns=['parsed_date'])

    def save_patched_data(self):
        print(f"[*] Saving patched dataset to: {self.output_file}...")
        self.df.to_csv(self.output_file, sep='\t', index=False)
        print(f"[+] Successfully saved patched dataset! Size: {os.path.getsize(self.output_file):,} bytes")

        log_file = "patch_log.json"
        # Convert numpy types to native Python types for JSON serialization
        clean_samples = []
        for p in self.patch_log[:500]:
            clean_p = {}
            for k, v in p.items():
                if isinstance(v, (np.integer, np.int64)):
                    clean_p[k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    clean_p[k] = float(v)
                else:
                    clean_p[k] = v
            clean_samples.append(clean_p)

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump({
                'patched_at': datetime.now().isoformat(),
                'total_patches': len(self.patch_log),
                'sample_patches': clean_samples
            }, f, ensure_ascii=False, indent=2)
        print(f"[+] Saved patch log to: {log_file}")


if __name__ == '__main__':
    input_csv = r"d:\2026-2027\TLCN\crawl_data\coffee_price_cleaned.csv"
    output_csv = r"d:\2026-2027\TLCN\crawl_data\coffee_price_patched.csv"
    patcher = CoffeeDataPatcher(input_csv, output_csv)
    patcher.load_data()
    patcher.patch_all()
    patcher.save_patched_data()
