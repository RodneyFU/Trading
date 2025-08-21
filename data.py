import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from textblob import TextBlob
import traceback

logging.basicConfig(
    filename=r'C:\Trading\logs\backtest.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def load_config():
    """載入配置檔案，確保 db_path 存在"""
    config_dir = r'C:\Trading\config'
    api_keys = json.load(open(f'{config_dir}\\api_keys.json'))
    system_config = json.load(open(f'{config_dir}\\system_config.json'))
    trading_params = json.load(open(f'{config_dir}\\trading_params.json'))
    config = {
        'api_keys': api_keys,
        'system_config': system_config,
        'trading_params': trading_params
    }
    # 設置預設 db_path
    if 'db_path' not in system_config:
        system_config['db_path'] = r'C:\Trading\data\trading_data.db'
    return config

def init_database(config):
    """初始化 SQLite 資料庫"""
    DB_PATH = Path(config['system_config']['db_path'])
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ohlc (
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            timeframe TEXT,
            PRIMARY KEY (date, timeframe)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS indicators (
            date TEXT,
            timeframe TEXT,
            RSI REAL,
            MACD REAL,
            MACD_signal REAL,
            ATR REAL,
            Stochastic REAL,
            Bollinger_upper REAL,
            Bollinger_lower REAL,
            EMA12 REAL,
            EMA26 REAL,
            PRIMARY KEY (date, timeframe)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS economic_data (
            date TEXT,
            event TEXT,
            impact TEXT,
            actual REAL,
            forecast REAL,
            previous REAL,
            PRIMARY KEY (date, event)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data (
            date TEXT,
            symbol TEXT,
            sentiment REAL,
            PRIMARY KEY (date, symbol)
        )
    ''')
    conn.commit()
    conn.close()

def save_to_database(df: pd.DataFrame, timeframe: str, config):
    """增量儲存歷史數據至 SQLite"""
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

def load_from_database(timeframe: str, start_date: str, end_date: str, config):
    """從 SQLite 載入最近 7 天數據"""
    DB_PATH = Path(config['system_config']['db_path'])
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        query = "SELECT * FROM ohlc WHERE timeframe = ? AND date >= ? AND date <= ?"
        df = pd.read_sql_query(query, conn, params=(timeframe, start_date, end_date))
        conn.close()
        if df.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} SQLite 中無 {timeframe} 數據")
            logging.warning(f"No {timeframe} data in SQLite")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從 SQLite 載入失敗：{str(e)}")
        logging.error(f"Failed to load from SQLite: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

def backup_database(config):
    """備份 SQLite 資料庫"""
    DB_PATH = Path(config['system_config']['db_path'])
    backup_path = DB_PATH.parent / 'backup' / f'trading_data_{datetime.now().strftime("%Y%m%d")}.db'
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import shutil
        shutil.copy(DB_PATH, backup_path)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 資料庫已備份至 {backup_path}")
        logging.info(f"Database backed up to {backup_path}")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 備份失敗：{str(e)}")
        logging.error(f"Backup failed: {str(e)}, traceback={traceback.format_exc()}")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_fmp_economic_calendar(start_date: str, end_date: str, config):
    """從 SQLite 或 FMP 獲取經濟日曆"""
    DB_PATH = Path(config['system_config']['db_path'])
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        query = "SELECT * FROM economic_data WHERE date >= ? AND date <= ?"
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        if not df.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 成功從 SQLite 載入經濟日曆")
            logging.info("Successfully loaded economic calendar from SQLite")
            return df

        url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={start_date}&to={end_date}&apikey={config['api_keys']['fmp_api_key']}"
        response = requests.get(url, proxies=config['system_config'].get('proxies', {}))
        response.raise_for_status()
        data = response.json()
        if not data:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 經濟日曆為空")
            logging.warning("FMP economic calendar is empty")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'event', 'impact', 'actual', 'forecast', 'previous']]
        df.to_sql('economic_data', sqlite3.connect(DB_PATH, timeout=10), if_exists='append', index=False)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 經濟日曆已儲存至 SQLite")
        logging.info("Economic calendar saved to SQLite")
        return df

    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 獲取經濟日曆失敗：{str(e)}")
        logging.error(f"Failed to fetch economic calendar: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_x_data(symbol: str, start_date: str, end_date: str, config):
    """從 SQLite 或 X API 獲取情緒數據"""
    DB_PATH = Path(config['system_config']['db_path'])
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        query = "SELECT * FROM sentiment_data WHERE symbol = ? AND date >= ? AND date <= ?"
        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
        conn.close()
        if not df.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 成功從 SQLite 載入 {symbol} 情緒數據")
            logging.info(f"Successfully loaded {symbol} sentiment data from SQLite")
            return df

        headers = {'Authorization': f"Bearer {config['api_keys']['x_bearer_token']}"}
        url = f"https://api.twitter.com/2/tweets/search/recent?query={symbol}&start_time={start_date}T00:00:00Z&end_time={end_date}T23:59:59Z"
        response = requests.get(url, headers=headers, proxies=config['system_config'].get('proxies', {}))
        response.raise_for_status()
        tweets = response.json().get('data', [])
        if not tweets:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} X API 無 {symbol} 數據")
            logging.warning(f"No {symbol} data from X API")
            return pd.DataFrame()

        sentiments = []
        for tweet in tweets:
            blob = TextBlob(tweet['text'])
            sentiments.append({
                'date': pd.to_datetime(tweet['created_at']).date(),
                'symbol': symbol,
                'sentiment': blob.sentiment.polarity
            })
        df = pd.DataFrame(sentiments)
        df = df.groupby(['date', 'symbol'])['sentiment'].mean().reset_index()
        df.to_sql('sentiment_data', sqlite3.connect(DB_PATH, timeout=10), if_exists='append', index=False)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 情緒數據已儲存至 SQLite")
        logging.info(f"Sentiment data saved to SQLite for {symbol}")
        return df

    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 獲取 X 數據失敗：{str(e)}")
        logging.error(f"Failed to fetch X data: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

async def pre_collect_historical_data(timeframe: str, period: str, start_date: str, end_date: str, config):
    """預收集歷史數據、指標和情緒數據"""
    try:
        # 嘗試從 SQLite 載入
        df = load_from_database(timeframe, start_date, end_date, config)
        if not df.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 成功從 SQLite 載入歷史數據")
            logging.info("Successfully loaded historical data from SQLite")
        else:
            # 從 Yahoo Finance 獲取
            ticker = 'USDJPY=X'
            df = yf.download(ticker, start=start_date, end=end_date, interval=timeframe)
            if df.empty:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Yahoo Finance 無數據")
                logging.warning("No data from Yahoo Finance")
                return pd.DataFrame()
            df.reset_index(inplace=True)
            df['date'] = pd.to_datetime(df['Date'])
            df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            save_to_database(df, timeframe, config)

        # 計算技術指標
        df['RSI'] = df['close'].rolling(window=14).apply(lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / abs(x.diff().clip(upper=0)).mean()))))
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['ATR'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift()), abs(x['low'] - x['close'].shift())), axis=1).rolling(window=14).mean()

        # 獲取情緒數據
        sentiment_df = fetch_x_data('USDJPY', start_date, end_date, config)
        if not sentiment_df.empty:
            df = df.merge(sentiment_df[['date', 'sentiment']], on='date', how='left')
            df['sentiment'] = df['sentiment'].fillna(0)
        else:
            df['sentiment'] = 0

        return df

    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 預收集歷史數據失敗：{str(e)}")
        logging.error(f"Failed to pre-collect historical data: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

def import_to_database(config):
    """從 CSV 匯入經濟日曆至 SQLite"""
    DB_PATH = Path(config['system_config']['db_path'])
    csv_path = Path(config['system_config']['root_dir']) / 'data' / 'economic_calendar.csv'
    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['date'] = pd.to_datetime(df['date'])
            conn = sqlite3.connect(DB_PATH, timeout=10)
            df.to_sql('economic_data', conn, if_exists='append', index=False)
            conn.commit()
            conn.close()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 成功從 CSV 匯入經濟日曆")
            logging.info("Successfully imported economic calendar from CSV")
        else:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CSV 檔案不存在：{csv_path}")
            logging.warning(f"CSV file not found: {csv_path}")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CSV 匯入失敗：{str(e)}")
        logging.error(f"Failed to import CSV: {str(e)}, traceback={traceback.format_exc()}")