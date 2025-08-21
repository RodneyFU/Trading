import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from bs4 import BeautifulSoup
import random
import time
import yfinance as yf
import json
import sqlite3
from textblob import TextBlob
import urllib.request
import traceback

# 設置日誌
logging.basicConfig(
    filename=r'C:\Trading\logs\backtest.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def load_config():
    """中文註釋：載入所有配置文件，提供預設值防止 KeyError"""
    config = {'system_config': {}, 'api_keys': {}, 'trading_params': {}}
    try:
        with open(r'C:\Trading\config\api_keys.json', 'r') as f:
            config['api_keys'] = json.load(f)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 成功載入設定檔：C:\\Trading\\config\\api_keys.json")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 載入 api_keys.json 失敗：{str(e)}")
        logging.error(f"Failed to load api_keys.json: {str(e)}")
    
    try:
        with open(r'C:\Trading\config\system_config.json', 'r') as f:
            config['system_config'] = json.load(f)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 成功載入設定檔：C:\\Trading\\config\\system_config.json")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 載入 system_config.json 失敗：{str(e)}")
        logging.error(f"Failed to load system_config.json: {str(e)}")
        config['system_config'] = {
            'root_dir': r'C:\Trading',
            'db_path': r'C:\Trading\data\trading_data.db',  # 預設資料庫路徑
            'min_backtest_days': 180,
            'proxies': {},
            'indicators': {
                'RSI': True,
                'MACD': True,
                'ATR': True,
                'Stochastic': True,
                'Bollinger': True,
                'EMA': True
            }
        }
    
    try:
        with open(r'C:\Trading\config\trading_params.json', 'r') as f:
            config['trading_params'] = json.load(f)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 成功載入設定檔：C:\\Trading\\config\\trading_params.json")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 載入 trading_params.json 失敗：{str(e)}")
        logging.error(f"Failed to load trading_params.json: {str(e)}")
    
    # 確保 db_path 存在
    if 'db_path' not in config['system_config']:
        config['system_config']['db_path'] = r'C:\Trading\data\trading_data.db'
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 使用預設資料庫路徑：{config['system_config']['db_path']}")
        logging.info(f"Using default db_path: {config['system_config']['db_path']}")
    
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 設定檔載入成功！")
    logging.info("Configuration loaded successfully")
    return config

def initialize_database(config):
    """中文註釋：初始化 SQLite 資料庫，新增 sentiment_data 表格"""
    DB_PATH = Path(config['system_config']['db_path'])
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ohlc (
            date TEXT,
            timeframe TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (date, timeframe)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS indicators (
            date TEXT,
            timeframe TEXT,
            indicator TEXT,
            value REAL,
            PRIMARY KEY (date, timeframe, indicator)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS economic_data (
            date TEXT,
            event TEXT,
            impact TEXT,
            PRIMARY KEY (date, event)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data (
            date TEXT,
            sentiment REAL,
            PRIMARY KEY (date)
        )
    ''')
    conn.commit()
    conn.close()
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 資料庫初始化完成：{DB_PATH}")
    logging.info(f"Database initialized: {DB_PATH}")

def import_to_database(config):
    """中文註釋：將現有 economic_calendar.csv 匯入 SQLite 資料庫"""
    DB_PATH = Path(config['system_config']['db_path'])
    csv_path = Path(config['system_config']['root_dir']) / 'data' / 'economic_calendar.csv'
    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty and all(col in df.columns for col in ['date', 'event', 'impact']):
                df['date'] = pd.to_datetime(df['date'])
                conn = sqlite3.connect(DB_PATH)
                df.to_sql('economic_data', conn, if_exists='append', index=False)
                conn.commit()
                conn.close()
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已從 {csv_path} 匯入經濟日曆到 SQLite")
                logging.info(f"Imported economic calendar from {csv_path} to SQLite, shape={df.shape}")
            else:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CSV 檔案無效或缺少必要欄位")
                logging.warning(f"Invalid or missing columns in {csv_path}, columns={df.columns.tolist()}")
        else:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CSV 檔案不存在：{csv_path}")
            logging.warning(f"CSV file not found: {csv_path}")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CSV 匯入 SQLite 失敗：{str(e)}")
        logging.error(f"Failed to import CSV to SQLite: {str(e)}, traceback={traceback.format_exc()}")

def get_system_proxy(config):
    """中文註釋：獲取系統 Proxy 設定"""
    try:
        proxies = config['system_config'].get('proxies', {})
        if not proxies:
            proxies = urllib.request.getproxies()
        if proxies.get('http') or proxies.get('https'):
            fred_api_key = config['api_keys'].get('fred_api_key', 'TEST')
            test_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={fred_api_key}&file_type=json"
            response = requests.get(test_url, proxies=proxies, timeout=5)
            if response.status_code == 200:
                logging.debug(f"Proxy test successful: {proxies}")
            else:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Proxy 測試失敗，狀態碼：{response.status_code}")
                logging.warning(f"Proxy test failed: {proxies}, status_code={response.status_code}")
                proxies = {}
        logging.debug(f"Detected proxies: {proxies}")
        return proxies
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Proxy 測試失敗：{str(e)}")
        logging.warning(f"Proxy test failed: {proxies}, error={str(e)}")
        return {}

def filter_future_dates(df: pd.DataFrame) -> pd.DataFrame:
    """中文註釋：過濾未來日期"""
    current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if not df.empty and 'date' in df.columns:
        initial_rows = len(df)
        df = df[df['date'] <= current_date].copy()
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已過濾未來日期，初始行數={initial_rows}，剩餘行數={len(df)}")
        logging.info(f"Filtered future dates, initial_rows={initial_rows}, remaining_rows={len(df)}")
    return df

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_yahoo_finance_data(timeframe: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
    """中文註釋：從 Yahoo Finance 獲取歷史數據"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 Yahoo Finance 獲取 {timeframe} 歷史數據...")
    logging.info(f"Fetching Yahoo Finance data: timeframe={timeframe}, period={period}, start={start_date}, end={end_date}")
    try:
        symbol = 'USDJPY=X'
        interval_map = {'1 hour': '1h', '4 hours': '4h', '1 day': '1d'}
        interval = interval_map.get(timeframe, '1d')
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, start=start_date, end=end_date)
        if df.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Yahoo Finance 數據為空")
            logging.warning(f"Yahoo Finance data empty: symbol={symbol}, timeframe={timeframe}")
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'datetime': 'date'})
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df['date'] = pd.to_datetime(df['date'])
        df = filter_future_dates(df)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Yahoo Finance 歷史數據獲取成功：{timeframe}")
        logging.info(f"Successfully fetched Yahoo Finance data: shape={df.shape}, timeframe={timeframe}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Yahoo Finance 歷史數據獲取失敗：{str(e)}")
        logging.error(f"Failed to fetch Yahoo Finance data: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_fmp_data(timeframe: str, start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從 FMP API 獲取歷史數據"""
    FMP_API_KEY = config['api_keys'].get('fmp_api_key', '')
    if not FMP_API_KEY:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP API 鍵未配置")
        logging.error("FMP API key not configured")
        return pd.DataFrame()
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 FMP 獲取 {timeframe} 歷史數據...")
    logging.info(f"Fetching FMP data: timeframe={timeframe}, start={start_date}, end={end_date}")
    try:
        proxies = get_system_proxy(config)
        interval_map = {'1 hour': '1hour', '4 hours': '4hour', '1 day': '1day'}
        interval = interval_map.get(timeframe, '1day')
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/USDJPY?from={start_date}&to={end_date}&apikey={FMP_API_KEY}&interval={interval}"
        response = requests.get(url, proxies=proxies, timeout=10).json()
        if isinstance(response, dict) and 'Error Message' in response:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 歷史數據請求失敗：{response['Error Message']}")
            logging.error(f"FMP historical data request failed: {response['Error Message']}")
            return pd.DataFrame()
        df = pd.DataFrame(response.get('historical', []))
        if df.empty or 'date' not in df.columns:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 歷史數據為空或格式錯誤")
            logging.warning(f"FMP historical data empty or invalid: shape={df.shape}")
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df = filter_future_dates(df)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 歷史數據獲取成功：{timeframe}")
        logging.info(f"Successfully fetched FMP data: shape={df.shape}, timeframe={timeframe}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 歷史數據獲取失敗：{str(e)}")
        logging.error(f"Failed to fetch FMP data: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_fred_data(series_id: str, start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從 FRED API 獲取經濟數據"""
    FRED_API_KEY = config['api_keys'].get('fred_api_key', '')
    if not FRED_API_KEY:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FRED API 鍵未配置")
        logging.error("FRED API key not configured")
        return pd.DataFrame()
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 FRED 獲取 {series_id} 數據...")
    logging.info(f"Fetching FRED data: series_id={series_id}, start={start_date}, end={end_date}")
    try:
        proxies = get_system_proxy(config)
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': FRED_API_KEY,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        response = requests.get(url, params=params, proxies=proxies, timeout=10).json()
        if 'error_code' in response or not response.get('observations'):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FRED 數據為空或格式錯誤：{response.get('error_message', 'No error message')}")
            logging.warning(f"FRED data empty or invalid: {response}, series_id={series_id}")
            return pd.DataFrame()
        df = pd.DataFrame(response['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[['date', 'value']].rename(columns={'value': series_id})
        df = filter_future_dates(df)
        if not df.empty:
            df.set_index('date', inplace=True)
            df = df.resample('D').ffill().reset_index()
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FRED 數據獲取成功：{series_id}")
        logging.info(f"Successfully fetched FRED data: series_id={series_id}, shape={df.shape}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FRED 數據獲取失敗：{str(e)}")
        logging.error(f"Failed to fetch FRED data: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_forex_factory_calendar(start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從 Forex Factory 經濟日曆抓取數據"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 Forex Factory 獲取經濟事件日曆...")
    logging.info(f"Fetching Forex Factory calendar: start={start_date}, end={end_date}")
    try:
        proxies = get_system_proxy(config)
        headers = {
            'User-Agent': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            ])
        }
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        current_dt = start_dt
        events = []
        while current_dt <= end_dt:
            week_start = current_dt - timedelta(days=current_dt.weekday())
            week_str = week_start.strftime('week=%b%d.%Y').lower()
            url = f"https://www.forexfactory.com/calendar?{week_str}"
            response = requests.get(url, headers=headers, proxies=proxies, timeout=10)
            if response.status_code != 200:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Forex Factory 請求失敗，狀態碼：{response.status_code}")
                logging.warning(f"Forex Factory request failed, status_code={response.status_code}")
                current_dt += timedelta(days=7)
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', class_='calendar__table')
            if not table:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Forex Factory 未找到日曆表格")
                logging.warning("Could not find calendar table")
                current_dt += timedelta(days=7)
                continue
            for row in table.find_all('tr', class_='calendar_row'):
                date_cell = row.find('td', class_='calendar__date')
                time_cell = row.find('td', class_='calendar__time')
                currency_cell = row.find('td', class_='calendar__currency')
                impact_cell = row.find('td', class_='calendar__impact')
                event_cell = row.find('td', class_='calendar__event')
                if not all([date_cell, time_cell, currency_cell, impact_cell, event_cell]):
                    continue
                date_str = date_cell.text.strip()
                time_str = time_cell.text.strip()
                currency = currency_cell.text.strip()
                impact = impact_cell.find('span', class_='icon--ff-impact')['title'].lower() if impact_cell.find('span', class_='icon--ff-impact') else 'low'
                event = event_cell.text.strip()
                if currency not in ['USD', 'JPY']:
                    continue
                try:
                    event_datetime = pd.to_datetime(f"{date_str} {time_str} {week_start.year}", format='%a, %b %d %I:%M%p %Y')
                    if event_datetime < start_dt or event_datetime > end_dt:
                        continue
                    events.append({
                        'date': event_datetime,
                        'event': f"{currency} {event}",
                        'impact': impact.capitalize()
                    })
                except ValueError:
                    continue
            current_dt += timedelta(days=7)
            time.sleep(random.uniform(1, 3))
        df = pd.DataFrame(events)
        if df.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Forex Factory 經濟日曆為空")
            logging.warning(f"Forex Factory calendar empty: shape={df.shape}")
            return pd.DataFrame()
        df = filter_future_dates(df)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Forex Factory 經濟事件日曆獲取成功")
        logging.info(f"Successfully fetched Forex Factory calendar: shape={df.shape}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Forex Factory 經濟事件日曆獲取失敗：{str(e)}")
        logging.error(f"Failed to fetch Forex Factory calendar: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_x_data(query: str, start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從 X API 獲取推文情緒數據並儲存至 SQLite"""
    X_BEARER_TOKEN = config['api_keys'].get('x_bearer_token', '')
    if not X_BEARER_TOKEN:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} X Bearer Token 未配置")
        logging.error("X Bearer Token not configured")
        return pd.DataFrame()
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 X API 獲取推文數據：{query}")
    logging.info(f"Fetching X data: query={query}, start={start_date}, end={end_date}")
    try:
        proxies = get_system_proxy(config)
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
        params = {
            'query': f"{query} lang:en",
            'start_time': start_date,
            'end_time': end_date,
            'max_results': 100
        }
        response = requests.get(url, headers=headers, params=params, proxies=proxies, timeout=10).json()
        if 'data' not in response or not response['data']:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} X API 數據為空或格式錯誤")
            logging.warning(f"X API data empty or invalid: {response}, query={query}")
            return pd.DataFrame()
        df = pd.DataFrame(response['data'])
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df[['created_at', 'text']].rename(columns={'created_at': 'date'})
        df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df = filter_future_dates(df)
        # 儲存情緒數據到 SQLite
        if not df.empty:
            DB_PATH = Path(config['system_config']['db_path'])
            conn = sqlite3.connect(DB_PATH)
            df[['date', 'sentiment']].to_sql('sentiment_data', conn, if_exists='append', index=False)
            conn.commit()
            conn.close()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} X API 情緒數據已儲存到 SQLite")
            logging.info(f"X API sentiment data saved to SQLite: shape={df.shape}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} X API 數據獲取成功")
        logging.info(f"Successfully fetched X data: query={query}, shape={df.shape}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} X API 數據獲取失敗：{str(e)}")
        logging.error(f"Failed to fetch X data: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_fmp_economic_calendar(start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從 SQLite 或 FMP API 獲取經濟事件日曆，若失敗則使用 FRED 或 Forex Factory"""
    DB_PATH = Path(config['system_config']['db_path'])
    initialize_database(config)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 SQLite 載入經濟事件日曆...")
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM economic_data WHERE date BETWEEN ? AND ?"
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        if not df.empty and all(col in df.columns for col in ['date', 'event', 'impact']):
            df['date'] = pd.to_datetime(df['date'])
            df = filter_future_dates(df)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} SQLite 經濟事件日曆載入成功")
            logging.info(f"Successfully loaded economic calendar from SQLite: shape={df.shape}")
            return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} SQLite 經濟事件日曆載入失敗：{str(e)}")
        logging.error(f"Failed to load economic calendar from SQLite: {str(e)}, traceback={traceback.format_exc()}")

    FMP_API_KEY = config['api_keys'].get('fmp_api_key', '')
    if not FMP_API_KEY:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP API 鍵未配置")
        logging.error("FMP API key not configured")
    else:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 FMP 獲取經濟事件日曆...")
        logging.info(f"Fetching FMP economic calendar: start={start_date}, end={end_date}")
        try:
            proxies = get_system_proxy(config)
            url = f"https://financialmodelingprep.com/api/v3/economic-calendar?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
            response = requests.get(url, proxies=proxies, timeout=10).json()
            if isinstance(response, dict) and 'Error Message' in response:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 經濟日曆請求失敗：{response['Error Message']}")
                logging.error(f"FMP economic calendar request failed: {response['Error Message']}")
                raise Exception(f"FMP error: {response['Error Message']}")
            df = pd.DataFrame(response)
            if 'date' not in df.columns:
                raise Exception("Missing 'date' column")
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'event', 'impact']]
            df = filter_future_dates(df)
            if not df.empty:
                conn = sqlite3.connect(DB_PATH)
                df.to_sql('economic_data', conn, if_exists='append', index=False)
                conn.commit()
                conn.close()
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 經濟事件日曆獲取成功並儲存到 SQLite")
                logging.info(f"Successfully fetched and saved economic calendar from FMP: shape={df.shape}")
                return df
        except Exception as e:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 經濟事件日曆獲取失敗：{str(e)}")
            logging.error(f"Failed to fetch FMP economic calendar: {str(e)}, traceback={traceback.format_exc()}")

    # 嘗試 FRED 備用
    try:
        fred_df = fetch_fred_data('UNRATE', start_date, end_date, config)
        if not fred_df.empty:
            fred_df = fred_df.rename(columns={'UNRATE': 'event_value'})
            fred_df['event'] = 'US Unemployment Rate'
            fred_df['impact'] = 'High'
            fred_df = fred_df[['date', 'event', 'impact']]
            conn = sqlite3.connect(DB_PATH)
            fred_df.to_sql('economic_data', conn, if_exists='append', index=False)
            conn.commit()
            conn.close()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FRED 經濟日曆替代成功（UNRATE）並儲存到 SQLite")
            logging.info(f"Successfully used FRED UNRATE as economic calendar fallback: shape={fred_df.shape}")
            return fred_df
    except Exception as fred_e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FRED 經濟日曆替代失敗：{str(fred_e)}")
        logging.error(f"Failed to fetch FRED UNRATE as fallback: {str(fred_e)}, traceback={traceback.format_exc()}")

    # 嘗試 Forex Factory 備用
    try:
        ff_df = fetch_forex_factory_calendar(start_date, end_date, config)
        if not ff_df.empty:
            conn = sqlite3.connect(DB_PATH)
            ff_df.to_sql('economic_data', conn, if_exists='append', index=False)
            conn.commit()
            conn.close()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Forex Factory 經濟日曆備用成功並儲存到 SQLite")
            logging.info(f"Successfully used Forex Factory as economic calendar fallback: shape={ff_df.shape}")
            return ff_df
        return pd.DataFrame()
    except Exception as ff_e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Forex Factory 經濟日曆備用失敗：{str(ff_e)}")
        logging.error(f"Failed to fetch Forex Factory calendar as fallback: {str(ff_e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

def save_to_database(df: pd.DataFrame, timeframe: str, config):
    """將歷史數據增量儲存到 SQLite 資料庫"""
    DB_PATH = Path(config['system_config']['db_path'])
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM ohlc WHERE timeframe = ?", (timeframe,))
        last_date = cursor.fetchone()[0]
        df_to_save = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_to_save['timeframe'] = timeframe
        if last_date:
            df_to_save = df_to_save[df_to_save['date'] > pd.to_datetime(last_date)]
        if not df_to_save.empty:
            df_to_save.to_sql('ohlc', conn, if_exists='append', index=False)
            conn.commit()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 增量儲存 {len(df_to_save)} 行數據至 SQLite：{timeframe}")
            logging.info(f"Incrementally saved {len(df_to_save)} rows to SQLite: timeframe={timeframe}")
        conn.close()
        return True
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 歷史數據儲存失敗：{str(e)}")
        logging.error(f"Failed to save data to SQLite: {str(e)}, traceback={traceback.format_exc()}")
        return False

def load_from_database(timeframe: str, start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從 SQLite 資料庫載入歷史數據"""
    DB_PATH = Path(config['system_config']['db_path'])
    try:
        initialize_database(config)
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT * FROM ohlc WHERE timeframe = ? AND date BETWEEN ? AND ?"
        df = pd.read_sql_query(query, conn, params=(timeframe, start_date, end_date))
        conn.close()
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = filter_future_dates(df)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 SQLite 載入 {timeframe} 歷史數據成功")
            logging.info(f"Successfully loaded {timeframe} data from SQLite: shape={df.shape}")
            return df[['date', 'open', 'high', 'low', 'close', 'volume']]
        return pd.DataFrame()
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 SQLite 載入 {timeframe} 歷史數據失敗：{str(e)}")
        logging.error(f"Failed to load {timeframe} data from SQLite: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

def load_sentiment_from_database(start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從 SQLite 載入情緒數據"""
    DB_PATH = Path(config['system_config']['db_path'])
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM sentiment_data WHERE date BETWEEN ? AND ?"
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = filter_future_dates(df)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 SQLite 載入情緒數據成功")
            logging.info(f"Successfully loaded sentiment data from SQLite: shape={df.shape}")
            return df[['date', 'sentiment']]
        return pd.DataFrame()
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 SQLite 載入情緒數據失敗：{str(e)}")
        logging.error(f"Failed to load sentiment data from SQLite: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

def load_cached_data(timeframe: str, period: str, start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從快取或 SQLite 載入歷史數據，若無效則重新獲取"""
    DATA_DIR = Path(config['system_config']['root_dir']) / 'data'
    timeframe_dir = DATA_DIR / f'historical/{timeframe}'
    timeframe_dir.mkdir(parents=True, exist_ok=True)
    cache_file = timeframe_dir / f'USDJPY_{timeframe}.pkl'
    try:
        if cache_file.exists():
            df = pd.read_pickle(cache_file)
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns) or len(df) < 100:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 快取檔案 {cache_file} 無效或數據不足：{len(df)} 行")
                logging.error(f"Cache file {cache_file} invalid or insufficient: rows={len(df)}, columns={df.columns.tolist()}")
                cache_file.unlink()
                df = pd.DataFrame()
            else:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從快取載入 {timeframe} 數據")
                logging.info(f"Loaded {timeframe} data from cache: shape={df.shape}")
                return df
        df = load_from_database(timeframe, start_date, end_date, config)
        if not df.empty and len(df) >= 100:
            return df
        df = fetch_yahoo_finance_data(timeframe, period, start_date, end_date)
        if df.empty or len(df) < 100:
            df = fetch_fmp_data(timeframe, start_date, end_date, config)
        if not df.empty and len(df) >= 100:
            df.to_pickle(cache_file)
            save_to_database(df, timeframe, config)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 數據已快取並儲存到 SQLite：{cache_file}")
            logging.info(f"Data cached and saved to SQLite: {cache_file}, shape={df.shape}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 快取載入失敗：{str(e)}")
        logging.error(f"Failed to load cache: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame, config) -> pd.DataFrame:
    """中文註釋：計算技術指標（RSI, MACD, ATR 等）"""
    try:
        indicators = config['system_config'].get('indicators', {})
        if indicators.get('RSI', False):
            df['RSI'] = compute_rsi(df['close'], 14)
        if indicators.get('MACD', False):
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = compute_macd(df['close'])
        if indicators.get('ATR', False):
            df['ATR'] = compute_atr(df['high'], df['low'], df['close'], 14)
        if indicators.get('Stochastic', False):
            df['Stoch_K'], df['Stoch_D'] = compute_stochastic(df['high'], df['low'], df['close'], 14)
        if indicators.get('Bollinger', False):
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = compute_bollinger_bands(df['close'], 20)
        if indicators.get('EMA', False):
            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        logging.info(f"Calculated technical indicators: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 技術指標計算失敗：{str(e)}")
        logging.error(f"Failed to calculate technical indicators: {str(e)}, traceback={traceback.format_exc()}")
        return df

def compute_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
    """中文註釋：計算 RSI"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """中文註釋：計算 MACD"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, periods: int = 14) -> pd.Series:
    """中文註釋：計算 ATR"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=periods).mean()

def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, periods: int = 14) -> tuple:
    """中文註釋：計算隨機指標"""
    lowest_low = low.rolling(window=periods).min()
    highest_high = high.rolling(window=periods).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=3).mean()
    return k, d

def compute_bollinger_bands(data: pd.Series, periods: int = 20, std_dev: float = 2) -> tuple:
    """中文註釋：計算布林帶"""
    sma = data.rolling(window=periods).mean()
    std = data.rolling(window=periods).std()
    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std
    return upper_band, sma, lower_band

async def pre_collect_historical_data(timeframe: str, period: str, start_date: str = None, end_date: str = None, config=None) -> pd.DataFrame:
    """中文註釋：預收集歷史數據並計算技術指標與情緒數據"""
    if config is None:
        config = load_config()
    initialize_database(config)
    import_to_database(config)
    min_days = config['system_config'].get('min_backtest_days', 180)
    current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if not end_date:
        end_date = current_date.strftime('%Y-%m-%d')
    if not start_date:
        start_date = (pd.to_datetime(end_date) - timedelta(days=min_days)).strftime('%Y-%m-%d')
    try:
        end_date_dt = pd.to_datetime(end_date)
        start_date_dt = pd.to_datetime(start_date)
        if end_date_dt > current_date:
            end_date_dt = current_date
            end_date = end_date_dt.strftime('%Y-%m-%d')
        date_diff = (end_date_dt - start_date_dt).days
        if date_diff < min_days:
            start_date_dt = end_date_dt - timedelta(days=min_days)
            start_date = start_date_dt.strftime('%Y-%m-%d')
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 回測期間太短：{date_diff} 天，自動調整到 {min_days} 天")
            logging.info(f"Adjusted start_date to {start_date} to meet minimum {min_days} days")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 日期格式錯誤：{str(e)}")
        logging.error(f"Invalid date format: {str(e)}, using default {min_days} days")
        start_date = (current_date - timedelta(days=min_days)).strftime('%Y-%m-%d')
    df = load_cached_data(timeframe, period, start_date, end_date, config)
    if df.empty or len(df) < 100:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 數據量不足：{len(df)} 行，需至少 100 行")
        logging.error(f"Insufficient data: {len(df)} rows, required at least 100")
        return pd.DataFrame()
    df = calculate_technical_indicators(df, config)
    # 嘗試從 SQLite 載入情緒數據
    x_df = load_sentiment_from_database(start_date, end_date, config)
    if x_df.empty:
        x_query = "USDJPY OR USD/JPY OR 'Federal Reserve'"
        x_df = fetch_x_data(x_query, start_date, end_date, config)
    if not x_df.empty:
        df = df.merge(x_df[['date', 'sentiment']], on='date', how='left')
        df['sentiment'] = df['sentiment'].fillna(0)
    return df