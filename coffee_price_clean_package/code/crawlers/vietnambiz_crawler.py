"""
VietnamBiz Crawler
Crawls news articles on VietnamBiz and parses domestic coffee prices by province using regex patterns.
"""

import sys
import re
import urllib.request
import ssl
from bs4 import BeautifulSoup
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class VietnamBizCrawler:
    SEARCH_URL = "https://vietnambiz.vn/tim-kiem.htm?q=gi%C3%A1+c%C3%A0+ph%C3%AA+h%C3%B4m+nay"

    def __init__(self, timeout=8):
        self.timeout = timeout
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def _fetch_html(self, url):
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req, context=self.ctx, timeout=self.timeout) as resp:
            return resp.read().decode('utf-8', errors='ignore')

    def _parse_article_prices(self, article_url):
        if not article_url.startswith('http'):
            article_url = f"https://vietnambiz.vn{article_url}"
        
        try:
            html = self._fetch_html(article_url)
        except Exception:
            return None, {}

        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()

        # Extract date from URL or title or text
        # URL pattern: ...-20261023... or title "Giá cà phê hôm nay 23/10..."
        date_str = None
        m_title = re.search(r'(\d{1,2})[/.-](\d{1,2})(?:[/.-](\d{4}))?', soup.title.text if soup.title else "")
        if m_title:
            day, month = m_title.group(1), m_title.group(2)
            year = m_title.group(3) or "2026"
            date_str = f"{int(day):02d}/{int(month):02d}/{year}"

        prices = {}
        patterns = {
            'Đắk Lắk': r'(?:Đắk Lắk|Dak Lak)[^\.\n\d]*?([0-9]{2,3}[\.,][0-9]{3})\s*(?:đồng|đ)',
            'Lâm Đồng': r'(?:Lâm Đồng|Lam Dong)[^\.\n\d]*?([0-9]{2,3}[\.,][0-9]{3})\s*(?:đồng|đ)',
            'Gia Lai': r'(?:Gia Lai)[^\.\n\d]*?([0-9]{2,3}[\.,][0-9]{3})\s*(?:đồng|đ)',
            'Đắk Nông': r'(?:Đắk Nông|Dak Nong)[^\.\n\d]*?([0-9]{2,3}[\.,][0-9]{3})\s*(?:đồng|đ)'
        }

        for prov, pat in patterns.items():
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                clean_num = float(re.sub(r'[^\d]', '', m.group(1)))
                if 20000 <= clean_num <= 150000:
                    prices[prov] = clean_num

        return date_str, prices

    def crawl_recent_articles(self, limit=10):
        results = {}
        try:
            html = self._fetch_html(self.SEARCH_URL)
        except Exception as e:
            print(f"[!] VietnamBizCrawler: Error fetching search: {e}")
            return results

        soup = BeautifulSoup(html, 'html.parser')
        article_links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'gia-ca-phe-hom-nay' in href and href not in article_links:
                article_links.append(href)

        for href in article_links[:limit]:
            d_str, p_dict = self._parse_article_prices(href)
            if d_str and p_dict:
                results[d_str] = p_dict

        return results


if __name__ == '__main__':
    crawler = VietnamBizCrawler()
    res = crawler.crawl_recent_articles()
    print("VietnamBiz Crawled Data:", res)
