"""
Multi-Source Coffee Price Crawler Orchestrator
Coordinates crawling from GiaCaPhe, GiaCafe, VietnamBiz, and Yahoo Finance.
Normalizes province names and aggregates into a single high-confidence dataset.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Local imports
from .giacaphe_crawler import GiaCaPheCrawler
from .giacafe_crawler import GiaCafeCrawler
from .vietnambiz_crawler import VietnamBizCrawler
from .yahoo_crawler import YahooFinanceCrawler

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

PROVINCE_NORMALIZE = {
    'đắk lắk': 'Dak Lak',
    'dak lak': 'Dak Lak',
    'lâm đồng': 'Lam Dong',
    'lam dong': 'Lam Dong',
    'gia lai': 'Gia Lai',
    'đắk nông': 'Dak Nong',
    'dak nong': 'Dak Nong',
    'kon tum': 'Kon Tum',
    'bà rịa - vũng tàu': 'Ba Ria - Vung Tau',
    'ba ria - vung tau': 'Ba Ria - Vung Tau',
    'bà rịa vũng tàu': 'Ba Ria - Vung Tau',
    'bình phước': 'Binh Phuoc',
    'binh phuoc': 'Binh Phuoc',
    'đồng nai': 'Dong Nai',
    'dong nai': 'Dong Nai',
    'nghệ an': 'Nghe An',
    'nghe an': 'Nghe An'
}


class MultiSourceCrawler:
    def __init__(self, cache_file="crawled_prices_cache.json"):
        self.cache_file = cache_file
        self.giacaphe = GiaCaPheCrawler()
        self.giacafe = GiaCafeCrawler()
        self.vietnambiz = VietnamBizCrawler()
        self.yahoo = YahooFinanceCrawler()

    def normalize_province(self, p_name):
        return PROVINCE_NORMALIZE.get(p_name.strip().lower(), p_name.strip())

    def normalize_date(self, d_str):
        """Converts DD/MM/YYYY or YYYY-MM-DD into M/D/YYYY to match original dataset"""
        try:
            if '-' in d_str:
                dt = datetime.strptime(d_str, '%Y-%m-%d')
            else:
                parts = d_str.split('/')
                if len(parts) == 3:
                    dt = datetime(int(parts[2]), int(parts[1]), int(parts[0]))
                else:
                    return d_str
            return f"{dt.month}/{dt.day}/{dt.year}"
        except Exception:
            return d_str

    def crawl_all(self, max_threads=4):
        print("\n" + "="*70)
        print("          BẮT ĐẦU CÀO DỮ LIỆU ĐA NGUỒN (MULTI-SOURCE CRAWLER)")
        print("="*70)

        aggregated_prices = {}
        macro_df = pd.DataFrame()

        # Step 1: Crawl GiaCaPhe.com
        print("[1/4] Crawling GiaCaPhe.com (CSS de-obfuscation)...")
        try:
            curr_gcp = self.giacaphe.crawl_current_noi_dia()
            if curr_gcp and 'prices' in curr_gcp:
                d_norm = self.normalize_date(curr_gcp['date'])
                if d_norm not in aggregated_prices:
                    aggregated_prices[d_norm] = {}
                for prov, val in curr_gcp['prices'].items():
                    p_norm = self.normalize_province(prov)
                    if p_norm in PROVINCE_NORMALIZE.values():
                        aggregated_prices[d_norm][p_norm] = {'price': val, 'source': 'giacaphe.com'}

            # Crawl province histories
            for p_slug, p_name in [('dak-lak', 'Dak Lak'), ('lam-dong', 'Lam Dong'), ('gia-lai', 'Gia Lai'), ('dak-nong', 'Dak Nong')]:
                hist = self.giacaphe.crawl_province_history(p_slug)
                for d_str, val in hist.items():
                    d_norm = self.normalize_date(d_str)
                    if d_norm not in aggregated_prices:
                        aggregated_prices[d_norm] = {}
                    if p_name not in aggregated_prices[d_norm]:
                        aggregated_prices[d_norm][p_name] = {'price': val, 'source': 'giacaphe.com'}

            # Crawl available daily pages
            daily_arch = self.giacaphe.crawl_daily_archives(max_days_back=14)
            for d_str, p_dict in daily_arch.items():
                d_norm = self.normalize_date(d_str)
                if d_norm not in aggregated_prices:
                    aggregated_prices[d_norm] = {}
                for prov, val in p_dict.items():
                    p_norm = self.normalize_province(prov)
                    if p_norm in PROVINCE_NORMALIZE.values():
                        aggregated_prices[d_norm][p_norm] = {'price': val, 'source': 'giacaphe.com'}
            print(f"      -> GiaCaPhe: Collected dates count: {len(aggregated_prices)}")
        except Exception as e:
            print(f"[!] Error crawling GiaCaPhe: {e}")

        # Step 2: Crawl GiaCafe.vn
        print("[2/4] Crawling GiaCafe.vn...")
        try:
            gc_data = self.giacafe.crawl_all_provinces()
            for d_str, p_dict in gc_data.items():
                d_norm = self.normalize_date(d_str)
                if d_norm not in aggregated_prices:
                    aggregated_prices[d_norm] = {}
                for prov, val in p_dict.items():
                    p_norm = self.normalize_province(prov)
                    if p_norm in PROVINCE_NORMALIZE.values():
                        if p_norm not in aggregated_prices[d_norm]:
                            aggregated_prices[d_norm][p_norm] = {'price': val, 'source': 'giacafe.vn'}
            print(f"      -> GiaCafe: Total dates now: {len(aggregated_prices)}")
        except Exception as e:
            print(f"[!] Error crawling GiaCafe: {e}")

        # Step 3: Crawl VietnamBiz
        print("[3/4] Crawling VietnamBiz.vn...")
        try:
            vnb_data = self.vietnambiz.crawl_recent_articles()
            for d_str, p_dict in vnb_data.items():
                d_norm = self.normalize_date(d_str)
                if d_norm not in aggregated_prices:
                    aggregated_prices[d_norm] = {}
                for prov, val in p_dict.items():
                    p_norm = self.normalize_province(prov)
                    if p_norm in PROVINCE_NORMALIZE.values():
                        if p_norm not in aggregated_prices[d_norm]:
                            aggregated_prices[d_norm][p_norm] = {'price': val, 'source': 'vietnambiz.vn'}
            print(f"      -> VietnamBiz: Total dates now: {len(aggregated_prices)}")
        except Exception as e:
            print(f"[!] Error crawling VietnamBiz: {e}")

        # Step 4: Crawl Yahoo Finance (Macro series)
        print("[4/4] Crawling Yahoo Finance (Arabica, Brent, DXY, FX)...")
        try:
            macro_df = self.yahoo.fetch_all_macro_series(start_date="2010-01-01")
            print(f"      -> Yahoo Finance: Retrieved {len(macro_df)} macro daily rows")
        except Exception as e:
            print(f"[!] Error crawling Yahoo Finance: {e}")

        # Save to cache
        cache_data = {
            'crawled_at': datetime.now().isoformat(),
            'domestic_prices': aggregated_prices
        }
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        if not macro_df.empty:
            macro_cache = "macro_yahoo_cache.csv"
            macro_df.to_csv(macro_cache)
            print(f"[+] Saved macro cache to: {macro_cache}")

        print(f"[+] Multi-source crawl finished! Saved {len(aggregated_prices)} dates to {self.cache_file}\n")
        return aggregated_prices, macro_df


if __name__ == '__main__':
    crawler = MultiSourceCrawler()
    prices, macro = crawler.crawl_all()
    print("Sample crawled domestic prices:")
    for d in list(prices.keys())[:3]:
        print(f"  Date {d}: {prices[d]}")
