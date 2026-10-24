"""
GiaCafe.vn Crawler
Crawls clean historical tables from giacafe.vn without obfuscation.
"""

import sys
import re
import urllib.request
import ssl
from bs4 import BeautifulSoup
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class GiaCafeCrawler:
    BASE_URL = "https://giacafe.vn"

    def __init__(self, timeout=8):
        self.timeout = timeout
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def _fetch_html(self, url):
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req, context=self.ctx, timeout=self.timeout) as resp:
            return resp.read().decode('utf-8', errors='ignore')

    def _clean_price(self, val_str):
        if not val_str:
            return None
        cleaned = re.sub(r'[^\d]', '', val_str)
        return float(cleaned) if cleaned else None

    def crawl_all_provinces(self):
        """
        Scrapes tables from homepage.
        Returns: { 'date_str': { 'Dak Lak': price, 'Lam Dong': price, ... } }
        """
        results = {}
        try:
            html = self._fetch_html(self.BASE_URL)
        except Exception as e:
            print(f"[!] GiaCafeCrawler: Failed to fetch {self.BASE_URL}: {e}")
            return results

        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all('table')

        # Province mapping based on table header or nearby h2/h3 tags
        province_order = ['Đắk Lắk', 'Lâm Đồng', 'Gia Lai', 'Đắk Nông', 'Kon Tum', 'Bình Phước', 'Đồng Nai', 'Bà Rịa - Vũng Tàu']

        for idx, table in enumerate(tables):
            # Try finding title from preceding heading
            prov_name = None
            prev_tag = table.find_previous(['h2', 'h3', 'h4', 'strong', 'b'])
            if prev_tag:
                txt = prev_tag.get_text(strip=True)
                for p in province_order:
                    if p.lower() in txt.lower():
                        prov_name = p
                        break
            if not prov_name and idx < len(province_order):
                prov_name = province_order[idx]

            if not prov_name:
                continue

            for tr in table.find_all('tr'):
                cells = tr.find_all(['th', 'td'])
                if len(cells) >= 2:
                    d_txt = cells[0].get_text(strip=True)
                    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', d_txt):
                        price = self._clean_price(cells[1].get_text(strip=True))
                        if price and price > 10000:
                            if d_txt not in results:
                                results[d_txt] = {}
                            results[d_txt][prov_name] = price

        return results


if __name__ == '__main__':
    crawler = GiaCafeCrawler()
    res = crawler.crawl_all_provinces()
    print(f"Crawled {len(res)} dates from giacafe.vn:")
    for d, provs in list(res.items())[:5]:
        print(f"  {d}: {provs}")
