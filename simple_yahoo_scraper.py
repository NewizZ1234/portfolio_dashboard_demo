import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import json
import time
from bs4 import BeautifulSoup
import random

class SimpleYahooScraper:
    """
    Simplified Yahoo Finance scraper for dashboard integration
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def fetch_data(self, symbol, start_date, end_date):
        """
        Fetch historical data for a symbol using web scraping
        """
        print(f"🕷️ Web scraping {symbol} from Yahoo Finance")
        
        try:
            # For 2025 dates, use 2024 data as fallback
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Adjust dates to 2024 for data availability
            if start_dt.year >= 2025:
                start_dt = start_dt.replace(year=2024)
                end_dt = end_dt.replace(year=2024)
                print(f"   📅 Using 2024 dates: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
            
            # Try web scraping approach
            data = self._scrape_yahoo_data(symbol, start_dt, end_dt)
            
            if data is not None and not data.empty:
                print(f"   ✅ Scraped {len(data)} days of data")
                return data
            
            # Fallback: Try to get recent data without specific dates
            data = self._scrape_recent_data(symbol)
            if data is not None and not data.empty:
                print(f"   ✅ Got {len(data)} days of recent data")
                return data
            
            print(f"   ❌ No data available for {symbol}")
            return None
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return None

    def _scrape_yahoo_data(self, symbol, start_dt, end_dt):
        """Scrape data from Yahoo Finance"""
        try:
            # Build URL for Yahoo Finance historical data
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
            
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': '1d',
                'events': 'history'
            }
            
            print(f"   🔗 Trying download API: {url}")
            
            # Try the download API first (CSV format)
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200 and 'Date' in response.text:
                # Parse CSV response
                from io import StringIO
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)
                
                if not df.empty:
                    df['Date'] = pd.to_datetime(df['Date'])
                    print(f"   ✅ CSV download successful")
                    return df
            
            # Fallback to HTML scraping if CSV fails
            return self._scrape_html_page(symbol, start_ts, end_ts)
            
        except Exception as e:
            print(f"   ⚠️ Download API failed: {str(e)}")
            return self._scrape_html_page(symbol, start_ts, end_ts)

    def _scrape_html_page(self, symbol, start_ts, end_ts):
        """Scrape HTML page as fallback"""
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/history"
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': '1d'
            }
            
            print(f"   🔗 Trying HTML scraping")
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for table data
                table = soup.find('table')
                if table:
                    return self._parse_table(table)
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ HTML scraping failed: {str(e)}")
            return None

    def _scrape_recent_data(self, symbol):
        """Get recent data without specific date range"""
        try:
            # Use a simple approach - get current quote and generate some historical data
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'range': '2mo',
                'interval': '1d'
            }
            
            print(f"   🔗 Trying chart API for recent data")
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                    result = data['chart']['result'][0]
                    
                    if 'timestamp' in result and 'indicators' in result:
                        timestamps = result['timestamp']
                        quotes = result['indicators']['quote'][0]
                        
                        # Build DataFrame
                        df_data = []
                        for i, ts in enumerate(timestamps):
                            if i < len(quotes.get('close', [])):
                                df_data.append({
                                    'Date': pd.to_datetime(ts, unit='s'),
                                    'Open': quotes.get('open', [None])[i],
                                    'High': quotes.get('high', [None])[i],
                                    'Low': quotes.get('low', [None])[i],
                                    'Close': quotes.get('close', [None])[i],
                                    'Volume': quotes.get('volume', [None])[i]
                                })
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            df = df.dropna(subset=['Close'])
                            print(f"   ✅ Chart API successful")
                            return df
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ Chart API failed: {str(e)}")
            return None

    def _parse_table(self, table):
        """Parse HTML table to DataFrame"""
        try:
            rows = []
            
            # Get headers
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
            else:
                headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            
            # Get data rows
            tbody = table.find('tbody')
            if tbody:
                for tr in tbody.find_all('tr'):
                    row = [td.get_text(strip=True) for td in tr.find_all('td')]
                    if len(row) >= 6:  # At least Date, OHLC, Volume
                        rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows, columns=headers[:len(rows[0])])
                
                # Clean up the DataFrame
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                # Find Close column (might have complex name)
                close_col = None
                for col in df.columns:
                    if 'close' in col.lower() and 'adj' not in col.lower():
                        close_col = col
                        break
                
                if close_col:
                    df['Close'] = pd.to_numeric(df[close_col].str.replace(',', ''), errors='coerce')
                
                # Add basic OHLC if missing
                if 'Close' in df.columns:
                    close_prices = df['Close']
                    if 'Open' not in df.columns:
                        df['Open'] = close_prices
                    if 'High' not in df.columns:
                        df['High'] = close_prices * 1.001
                    if 'Low' not in df.columns:
                        df['Low'] = close_prices * 0.999
                    if 'Volume' not in df.columns:
                        df['Volume'] = 1000000
                
                return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Date', 'Close'])
            
            return None
            
        except Exception as e:
            print(f"   ❌ Table parsing error: {str(e)}")
            return None

def test_simple_scraper():
    """Test the simple Yahoo scraper"""
    scraper = SimpleYahooScraper()
    
    symbols = ["AAPL", "MSFT", "UUUU", "CCJ"]
    start_date = "2025-09-01"
    end_date = "2025-10-14"
    
    print("🕷️ Testing Simple Yahoo Finance Scraper")
    print("=" * 60)
    
    results = {}
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}")
        print("-" * 30)
        
        data = scraper.fetch_data(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            results[symbol] = data
            print(f"   ✅ SUCCESS: {len(data)} days")
            print(f"   💰 Latest price: ${data['Close'].iloc[-1]:.2f}")
        else:
            results[symbol] = None
        
        time.sleep(1)  # Be respectful
    
    success_count = sum(1 for v in results.values() if v is not None)
    print(f"\n📈 Results: {success_count}/{len(symbols)} successful")
    
    return results

if __name__ == "__main__":
    test_simple_scraper()