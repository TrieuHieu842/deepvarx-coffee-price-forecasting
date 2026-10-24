"""
Yahoo Finance Fast Historical Crawler
Fetches high-accuracy historical daily market data for macro and financial indicators:
- KC=F: Arabica coffee futures (cents/lb -> USD/lb)
- BZ=F: Brent Crude Oil futures (USD/barrel)
- DX-Y.NYB: US Dollar Index (DXY)
- USDVND=X: USD to VND exchange rate
- USDBRL=X: USD to BRL exchange rate
"""

import sys
import json
import urllib.request
import ssl
import pandas as pd
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class YahooFinanceCrawler:
    TICKERS = {
        'arabica_price_usd_lb': 'KC=F',
        'brent_price_usd_barrel': 'BZ=F',
        'dxy_index': 'DX-Y.NYB',
        'usd_vnd_rate': 'USDVND=X',
        'usd_brl_rate': 'USDBRL=X'
    }

    def __init__(self, timeout=10):
        self.timeout = timeout
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def fetch_ticker_history(self, symbol, period1="2010-01-01", period2=None):
        """
        Fetches daily historical close prices from Yahoo Finance API.
        """
        p1 = int(datetime.strptime(period1, '%Y-%m-%d').timestamp())
        if period2:
            p2 = int(datetime.strptime(period2, '%Y-%m-%d').timestamp())
        else:
            p2 = int(datetime.now().timestamp())

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={p2}&interval=1d"
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, context=self.ctx, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                
                result = data['chart']['result'][0]
                timestamps = result.get('timestamp', [])
                quotes = result['indicators']['quote'][0]
                closes = quotes.get('close', [])
                
                series = {}
                for ts, close in zip(timestamps, closes):
                    if close is not None:
                        dt_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        # For Arabica KC=F, Yahoo quotes in US Cents/lb (e.g. 142.5 means 1.425 USD/lb)
                        if symbol == 'KC=F':
                            series[dt_str] = round(close / 100.0, 4)
                        else:
                            series[dt_str] = round(close, 2)
                return series
        except Exception as e:
            print(f"[!] YahooFinanceCrawler: Error fetching {symbol}: {e}")
            return {}

    def fetch_all_macro_series(self, start_date="2010-01-01"):
        """
        Fetches all configured tickers and returns a DataFrame indexed by date.
        """
        dfs = []
        for col_name, ticker in self.TICKERS.items():
            print(f"[*] Fetching Yahoo Finance data for {col_name} ({ticker})...")
            series_dict = self.fetch_ticker_history(ticker, period1=start_date)
            if series_dict:
                s = pd.Series(series_dict, name=col_name)
                dfs.append(s)
                print(f"    -> Received {len(series_dict)} daily observations for {ticker}")

        if dfs:
            macro_df = pd.concat(dfs, axis=1)
            macro_df.index = pd.to_datetime(macro_df.index)
            macro_df = macro_df.sort_index()
            return macro_df
        return pd.DataFrame()


if __name__ == '__main__':
    crawler = YahooFinanceCrawler()
    df_macro = crawler.fetch_all_macro_series(start_date="2026-01-01")
    print("\nRecent Macro Data Sample:")
    print(df_macro.tail(10))
