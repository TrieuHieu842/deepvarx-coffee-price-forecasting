"""
GiaCaPhe.com Crawler with CSS Pseudo-Element De-Obfuscation Engine
Extracts domestic coffee prices, London robusta, New York arabica, USD/VND rate.
"""

import sys
import re
import urllib.request
import ssl
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class GiaCaPheCrawler:
    BASE_URL = "https://giacaphe.com"

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

    def _extract_css_content_map(self, html):
        """
        GiaCaPhe.com embeds real numbers inside CSS pseudo-elements:
        .classname::after { content: '91,400'; }
        """
        css_map = {}
        pattern = r'\.([a-zA-Z0-9_-]+)::after\s*\{\s*(?:display:[^;]+;\s*)?content:\s*[\'"]([^\'"]*)[\'"]'
        for m in re.finditer(pattern, html):
            css_map[m.group(1)] = m.group(2)
        return css_map

    def _clean_price_str(self, val_str):
        if not val_str:
            return None
        # Remove non-digits (keep dot if float, but domestic is VND integer with comma)
        cleaned = re.sub(r'[^\d]', '', val_str)
        if cleaned:
            return float(cleaned)
        return None

    def crawl_current_noi_dia(self):
        """
        Crawls https://giacaphe.com/gia-ca-phe-noi-dia/
        Returns today's prices for Dak Lak, Lam Dong, Gia Lai, Dak Nong, etc.
        """
        url = f"{self.BASE_URL}/gia-ca-phe-noi-dia/"
        try:
            html = self._fetch_html(url)
        except Exception as e:
            print(f"[!] GiaCaPheCrawler: Failed to fetch {url}: {e}")
            return {}

        css_map = self._extract_css_content_map(html)
        soup = BeautifulSoup(html, 'html.parser')

        # Extract date from title or header e.g. "Giá cà phê hôm nay 23/10/2026"
        date_str = None
        title = soup.title.text if soup.title else ""
        m_date = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', title)
        if m_date:
            date_str = m_date.group(1)
        else:
            date_str = datetime.now().strftime('%d/%m/%Y')

        results = {}
        tables = soup.find_all('table')
        if not tables:
            return results

        # Table 0 is domestic prices
        t0 = tables[0]
        for tr in t0.find_all('tr'):
            cells = tr.find_all(['th', 'td'])
            if len(cells) >= 2:
                prov_name = cells[0].get_text(strip=True)
                # extract price
                spans = cells[1].find_all('span')
                if spans:
                    raw_val = "".join(css_map.get(cls, "") for s in spans for cls in s.get('class', []))
                    if not raw_val:
                        raw_val = cells[1].get_text(strip=True)
                else:
                    raw_val = cells[1].get_text(strip=True)

                price = self._clean_price_str(raw_val)
                if price and price > 10000:
                    results[prov_name] = price

        return {
            'date': date_str,
            'source': 'giacaphe.com',
            'prices': results
        }

    def crawl_province_history(self, province_slug="dak-lak"):
        """
        Crawls province page e.g. https://giacaphe.com/gia-ca-phe-dak-lak/
        Returns recent days historical table.
        """
        url = f"{self.BASE_URL}/gia-ca-phe-{province_slug}/"
        try:
            html = self._fetch_html(url)
        except Exception as e:
            print(f"[!] GiaCaPheCrawler: Failed to fetch {url}: {e}")
            return {}

        css_map = self._extract_css_content_map(html)
        soup = BeautifulSoup(html, 'html.parser')

        history = {}
        table = soup.find('table', class_='price-table')
        if table:
            for tr in table.find_all('tr'):
                cells = tr.find_all(['th', 'td'])
                if len(cells) >= 2:
                    d_txt = cells[0].get_text(strip=True)
                    # Check if d_txt is date dd/mm/yyyy
                    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', d_txt):
                        spans = cells[1].find_all('span')
                        if spans:
                            val_str = "".join(css_map.get(cls, "") for s in spans for cls in s.get('class', []))
                        else:
                            val_str = cells[1].get_text(strip=True)
                        price = self._clean_price_str(val_str)
                        if price:
                            history[d_txt] = price

        return history

    def crawl_daily_archives(self, max_days_back=14):
        """
        Crawls available historical days within the open window
        https://giacaphe.com/gia-ca-phe-noi-dia-ngay-{YYYY-MM-DD}/
        """
        data = {}
        today = datetime.now()
        for i in range(max_days_back):
            d = today - timedelta(days=i)
            d_slug = d.strftime('%Y-%m-%d')
            d_key = d.strftime('%d/%m/%Y')
            url = f"{self.BASE_URL}/gia-ca-phe-noi-dia-ngay-{d_slug}/"
            try:
                html = self._fetch_html(url)
                if "quá giới hạn cho phép" in html or "Vui lòng đăng nhập" in html:
                    continue
                css_map = self._extract_css_content_map(html)
                soup = BeautifulSoup(html, 'html.parser')
                tables = soup.find_all('table')
                if tables:
                    day_prices = {}
                    for tr in tables[0].find_all('tr'):
                        cells = tr.find_all(['th', 'td'])
                        if len(cells) >= 2:
                            prov = cells[0].get_text(strip=True)
                            spans = cells[1].find_all('span')
                            if spans:
                                val_str = "".join(css_map.get(cls, "") for s in spans for cls in s.get('class', []))
                            else:
                                val_str = cells[1].get_text(strip=True)
                            p = self._clean_price_str(val_str)
                            if p and p > 10000:
                                day_prices[prov] = p
                    if day_prices:
                        data[d_key] = day_prices
            except Exception:
                pass
        return data


if __name__ == '__main__':
    crawler = GiaCaPheCrawler()
    print("Testing current noi dia:")
    current = crawler.crawl_current_noi_dia()
    print(current)

    print("\nTesting Dak Lak history:")
    hist = crawler.crawl_province_history('dak-lak')
    print(hist)
