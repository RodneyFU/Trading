import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3
import yfinance as yf
import pandas_ta as ta
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from config import load_config
import urllib.request
import traceback
import os
import pickle

def initialize_database(config):
    """中文註釋：初始化 SQLite 資料庫，創建 OHLC 和指標表格"""
    DATA_DIR = Path(config['system_config']['root_dir']) / 'data'
    DB_PATH = DATA_DIR / 'trading_data.db'
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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
    conn.commit()
    conn.close()
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 資料庫初始化完成：{DB_PATH}")
    logging.info(f"Database initialized: {DB_PATH}")

def clean_cache(config):
    """中文註釋：清理超過 30 天的快取檔案"""
    DATA_DIR = Path(config['system_config']['root_dir']) / 'data'
    try:
        now = datetime.now()
        for timeframe_dir in DATA_DIR.glob('historical/*/'):
            for file in timeframe_dir.glob('*.pkl'):
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if (now - mtime).days > 30:
                    file.unlink()
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已刪除過期快取：{file}")
                    logging.info(f"Deleted expired cache: {file}")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 快取清理失敗：{str(e)}")
        logging.error(f"Failed to clean cache: {str(e)}, traceback={traceback.format_exc()}")

def clean_invalid_models(config):
    """中文註釋：檢查並清除無效模型檔案"""
    MODEL_DIR = Path(config['system_config']['root_dir']) / 'models'
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for model_file in MODEL_DIR.glob('*.pkl'):
            try:
                with open(model_file, 'rb') as f:
                    pickle.load(f)
                logging.debug(f"Model file {model_file} is valid")
            except Exception as e:
                logging.warning(f"Invalid model file detected: {model_file}, error={str(e)}")
                model_file.unlink()
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已刪除無效模型檔案：{model_file}")
                logging.info(f"Deleted invalid model file: {model_file}")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 模型清理失敗：{str(e)}")
        logging.error(f"Failed to clean invalid models: {str(e)}, traceback={traceback.format_exc()}")

def get_system_proxy():
    """中文註釋：獲取系統 Proxy 設定並測試連線"""
    try:
        proxies = urllib.request.getproxies()
        if proxies.get('http'):
            try:
                response = requests.get('https://www.google.com', proxies=proxies, timeout=5)
                logging.debug(f"Proxy test successful: {proxies}")
            except Exception as e:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Proxy 測試失敗：{str(e)}")
                logging.warning(f"Proxy test failed: {proxies}, error={str(e)}")
                proxies = {}
        logging.debug(f"Detected system proxies: {proxies}")
        return proxies
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 獲取系統 Proxy 失敗：{str(e)}")
        logging.error(f"Failed to get system proxies: {str(e)}, traceback={traceback.format_exc()}")
        return {}

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_yahoo_finance_data(timeframe: str, period: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """中文註釋：從 Yahoo Finance 獲取 USD/JPY 歷史數據"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 Yahoo Finance 獲取 {timeframe} 數據...")
    try:
        proxies = get_system_proxy()
        if proxies.get('http') or proxies.get('https'):
            os.environ['HTTP_PROXY'] = proxies.get('http', '')
            os.environ['HTTPS_PROXY'] = proxies.get('https', '')
            logging.info(f"Using system proxies for yfinance: HTTP={proxies.get('http')}, HTTPS={proxies.get('https')}")
        else:
            logging.debug("No system proxies detected for yfinance")
        
        ticker = 'USDJPY=X'
        df = yf.download(ticker, period=period, interval=timeframe, start=start_date, end=end_date)
        if df.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Yahoo Finance 數據為空")
            logging.warning(f"Yahoo Finance returned empty data for timeframe={timeframe}, period={period}, start={start_date}, end={end_date}")
            return pd.DataFrame()
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Yahoo Finance 數據獲取成功")
        logging.info(f"Successfully fetched data from Yahoo Finance, shape={df.shape}, columns={df.columns.tolist()}")
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Yahoo Finance 數據獲取失敗：{str(e)}")
        logging.error(f"Failed to fetch Yahoo Finance data: {str(e)}, traceback={traceback.format_exc()}, timeframe={timeframe}, period={period}")
        return pd.DataFrame()
    finally:
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_fmp_data(timeframe: str, start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從 FMP API 獲取 USD/JPY 歷史數據"""
    FMP_API_KEY = config['api_keys']['fmp_api_key']
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 FMP 獲取 {timeframe} 數據...")
    logging.info(f"Fetching FMP data: timeframe={timeframe}, start={start_date}, end={end_date}, api_key_masked={FMP_API_KEY[:5]}...")
    try:
        proxies = get_system_proxy()
        tf_map = {'1 hour': '1hour', '4 hours': '4hour', '1 day': '1day'}
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/USDJPY?from={start_date}&to={end_date}&timeseries={tf_map[timeframe]}&apikey={FMP_API_KEY}"
        response = requests.get(url, proxies=proxies, timeout=10).json()
        logging.debug(f"FMP data response type={type(response)}, content_preview={str(response)[:200]}, proxies={proxies}, url={url}")
        data = response.get('historical', [])
        if not isinstance(data, list) or not data:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 數據為空或格式錯誤")
            logging.warning(f"FMP data non-list or empty: {response}, timeframe={timeframe}")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df.rename(columns={'date': 'date', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 數據獲取成功")
        logging.info(f"Successfully fetched FMP data: shape={df.shape}, columns={df.columns.tolist()}")
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 數據獲取失敗：{str(e)}")
        logging.error(f"Failed to fetch FMP data: {str(e)}, traceback={traceback.format_exc()}, proxies={proxies}, url={url}")
        return pd.DataFrame()

def load_from_database(timeframe: str, start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從 SQLite 資料庫載入歷史數據"""
    DB_PATH = Path(config['system_config']['root_dir']) / 'data' / 'trading_data.db'
    try:
        initialize_database(config)
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT * FROM ohlc WHERE timeframe = ? AND date BETWEEN ? AND ?"
        df = pd.read_sql_query(query, conn, params=(timeframe, start_date, end_date))
        conn.close()
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從資料庫載入 {timeframe} 數據成功")
            logging.info(f"Successfully loaded {timeframe} data from database, shape={df.shape}, columns={df.columns.tolist()}")
            return df[['date', 'open', 'high', 'low', 'close', 'volume']]
        return pd.DataFrame()
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從資料庫載入 {timeframe} 數據失敗：{str(e)}")
        logging.error(f"Failed to load {timeframe} data from database: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

def save_to_database(df: pd.DataFrame, timeframe: str, config):
    """中文註釋：將數據儲存到 SQLite 資料庫"""
    DB_PATH = Path(config['system_config']['root_dir']) / 'data' / 'trading_data.db'
    try:
        conn = sqlite3.connect(DB_PATH)
        df.to_sql('ohlc', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 數據已儲存到資料庫：{timeframe}")
        logging.info(f"Data saved to database: {timeframe}, shape={df.shape}, columns={df.columns.tolist()}")
        return True
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 資料庫儲存失敗：{str(e)}")
        logging.error(f"Failed to save data to database: {str(e)}, traceback={traceback.format_exc()}")
        return False

def load_cached_data(timeframe: str, period: str, start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從快取載入歷史數據，若無效或無快取則獲取新數據"""
    DATA_DIR = Path(config['system_config']['root_dir']) / 'data'
    timeframe_dir = DATA_DIR / f'historical/{timeframe}'
    timeframe_dir.mkdir(parents=True, exist_ok=True)
    cache_file = timeframe_dir / f'USDJPY_{timeframe}.pkl'
    
    try:
        if cache_file.exists():
            df = pd.read_pickle(cache_file)
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 快取檔案 {cache_file} 缺少必要欄位")
                logging.error(f"Cache file {cache_file} missing required columns, available_columns={df.columns.tolist()}")
                cache_file.unlink()
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已刪除無效快取檔案：{cache_file}")
                logging.info(f"Deleted invalid cache file: {cache_file}")
                df = fetch_yahoo_finance_data(timeframe, period, start_date, end_date)
            else:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                cache_duration = {'1 hour': 24, '4 hours': 48, '1 day': 168}
                if (datetime.now() - mtime).total_seconds() / 3600 < cache_duration.get(timeframe, 24):
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從快取載入 {timeframe} 數據")
                    logging.info(f"Loaded {timeframe} data from cache, shape={df.shape}, columns={df.columns.tolist()}")
                    return df
        else:
            df = load_from_database(timeframe, start_date, end_date, config)
            if not df.empty:
                return df
    
        df = fetch_yahoo_finance_data(timeframe, period, start_date, end_date)
        if df.empty:
            df = fetch_fmp_data(timeframe, start_date, end_date, config)
    
        if not df.empty:
            df.to_pickle(cache_file)
            save_to_database(df, timeframe, config)
            year = datetime.now().strftime('%Y')
            backup_dir = DATA_DIR / 'backtest' / year
            backup_dir.mkdir(parents=True, exist_ok=True)
            current_date = datetime.now().strftime('%Y%m%d')
            df.to_pickle(backup_dir / f'USDJPY_{timeframe}_{current_date}.pkl')
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 數據已快取並備份：{cache_file}")
            logging.info(f"Data cached and backed up: {cache_file}, shape={df.shape}, columns={df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 快取載入失敗：{str(e)}")
        logging.error(f"Failed to load cache: {str(e)}, traceback={traceback.format_exc()}")
        return pd.DataFrame()

def load_cached_indicators(timeframe: str, config) -> pd.DataFrame:
    """中文註釋：從快取載入技術指標，若無效則刪除"""
    DATA_DIR = Path(config['system_config']['root_dir']) / 'data'
    indicators_dir = DATA_DIR / f'indicators/{timeframe}'
    indicators_dir.mkdir(parents=True, exist_ok=True)
    cache_file = indicators_dir / f'USDJPY_{timeframe}_indicators.pkl'
    
    try:
        if cache_file.exists():
            df = pd.read_pickle(cache_file)
            required_columns = ['date', 'RSI', 'MACD', 'ATR', 'STOCH_k', 'EMA_20', 'EMA_50']
            if not all(col in df.columns for col in required_columns):
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 技術指標快取 {cache_file} 缺少必要欄位")
                logging.error(f"Indicators cache {cache_file} missing required columns, available_columns={df.columns.tolist()}")
                cache_file.unlink()
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已刪除無效指標快取：{cache_file}")
                logging.info(f"Deleted invalid indicators cache: {cache_file}")
                return pd.DataFrame()
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            cache_duration = {'1 hour': 24, '4 hours': 48, '1 day': 168}
            if (datetime.now() - mtime).total_seconds() / 3600 < cache_duration.get(timeframe, 24):
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 從快取載入 {timeframe} 技術指標")
                logging.info(f"Loaded {timeframe} indicators from cache, shape={df.shape}, columns={df.columns.tolist()}")
                return df
        return pd.DataFrame()
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 技術指標快取載入失敗：{str(e)}")
        logging.error(f"Failed to load indicators cache: {str(e)}, traceback={traceback.format_exc()}")
        if cache_file.exists():
            cache_file.unlink()
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已刪除無效指標快取：{cache_file}")
            logging.info(f"Deleted invalid indicators cache: {cache_file}")
        return pd.DataFrame()

def save_cached_indicators(df: pd.DataFrame, timeframe: str, config):
    """中文註釋：儲存技術指標到快取，確保包含必要欄位"""
    DATA_DIR = Path(config['system_config']['root_dir']) / 'data'
    indicators_dir = DATA_DIR / f'indicators/{timeframe}'
    indicators_dir.mkdir(parents=True, exist_ok=True)
    cache_file = indicators_dir / f'USDJPY_{timeframe}_indicators.pkl'
    try:
        required_columns = ['date', 'RSI', 'MACD', 'ATR', 'STOCH_k', 'EMA_20', 'EMA_50']
        if not all(col in df.columns for col in required_columns):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 無法儲存技術指標快取：缺少必要欄位")
            logging.error(f"Cannot cache indicators: missing required columns, available_columns={df.columns.tolist()}")
            return
        df.to_pickle(cache_file)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 技術指標已快取：{cache_file}")
        logging.info(f"Indicators cached: {cache_file}, shape={df.shape}, columns={df.columns.tolist()}")
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 技術指標快取儲存失敗：{str(e)}")
        logging.error(f"Failed to cache indicators: {str(e)}, traceback={traceback.format_exc()}")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_fmp_indicators(timeframe: str, start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從 FMP API 獲取技術指標"""
    FMP_API_KEY = config['api_keys']['fmp_api_key']
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 FMP 獲取 {timeframe} 技術指標...")
    logging.info(f"Fetching FMP indicators: timeframe={timeframe}, start={start_date}, end={end_date}, api_key_masked={FMP_API_KEY[:5]}...")
    try:
        proxies = get_system_proxy()
        tf_map = {'1 hour': '1hour', '4 hours': '4hour', '1 day': '1day'}
        url = f"https://financialmodelingprep.com/api/v3/technical_indicator/{tf_map[timeframe]}/USDJPY?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
        response = requests.get(url, proxies=proxies, timeout=10).json()
        logging.debug(f"FMP indicators response type={type(response)}, content_preview={str(response)[:200]}, proxies={proxies}, url={url}")
        df = pd.DataFrame(response)
        if df.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 技術指標為空")
            logging.warning(f"FMP returned empty indicators for timeframe={timeframe}")
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 技術指標獲取成功")
        logging.info(f"Successfully fetched indicators from FMP: shape={df.shape}, columns={df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 技術指標獲取失敗：{str(e)}")
        logging.error(f"Failed to fetch FMP indicators: {str(e)}, traceback={traceback.format_exc()}, proxies={proxies}, url={url}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame, config) -> pd.DataFrame:
    """中文註釋：計算技術指標的輔助函數"""
    INDICATORS = config['system_config']['indicators']
    try:
        if df.empty or not all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 輸入 DataFrame 缺少必要欄位")
            logging.error(f"Input DataFrame missing required columns, df_shape={df.shape}, available_columns={df.columns.tolist()}")
            return df
        
        # 驗證配置鍵
        required_indicators = ['RSI', 'MACD', 'ATR', 'Stochastic', 'Bollinger', 'EMA']
        for indicator in required_indicators:
            if indicator not in INDICATORS:
                logging.warning(f"Indicator '{indicator}' missing in config, setting to False")
                INDICATORS[indicator] = False
        
        df = df.copy()
        if INDICATORS['RSI']:
            df['RSI'] = ta.rsi(df['close'], length=14)
        if INDICATORS['MACD']:
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd is not None:
                df['MACD'] = macd['MACD_12_26_9']
                df['MACD_Signal'] = macd['MACDs_12_26_9']
        if INDICATORS['ATR']:
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        if INDICATORS['Bollinger']:
            bbands = ta.bbands(df['close'], length=20, std=2)
            if bbands is not None:
                df['BB_upper'] = bbands['BBU_20_2.0']
                df['BB_lower'] = bbands['BBL_20_2.0']
        if INDICATORS['Stochastic']:
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
            if stoch is not None:
                df['STOCH_k'] = stoch['STOCHk_14_3_3']
                df['STOCH_d'] = stoch['STOCHd_14_3_3']
        if INDICATORS['ADX']:
            df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        if INDICATORS['Fibonacci']:
            high = df['high'].max()
            low = df['low'].min()
            levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
            fib_levels = [low + level * (high - low) for level in levels]
            for i, level in enumerate(fib_levels):
                df[f'Fib_{levels[i]}'] = level
        if INDICATORS['Ichimoku']:
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=9, kijun=26, senkou=52)
            if ichimoku[0] is not None:
                df['ICHIMOKU_tenkan'] = ichimoku[0]['ITS_9']
                df['ICHIMOKU_kijun'] = ichimoku[0]['IKS_26']
                df['ICHIMOKU_senkou_a'] = ichimoku[0]['ISA_9']
                df['ICHIMOKU_senkou_b'] = ichimoku[0]['ISB_26']
        if INDICATORS['EMA']:
            df['EMA_20'] = ta.ema(df['close'], length=20)
            df['EMA_50'] = ta.ema(df['close'], length=50)
        if INDICATORS['OBV']:
            df['OBV'] = ta.obv(df['close'], df['volume'])
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 技術指標計算完成")
        logging.info(f"Computed indicators: columns_added={list(set(df.columns) - {'date', 'open', 'high', 'low', 'close', 'volume'})}, shape={df.shape}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 技術指標計算失敗：{str(e)}")
        logging.error(f"Failed to compute indicators: {str(e)}, traceback={traceback.format_exc()}, df_shape={df.shape}, available_columns={df.columns.tolist()}")
        return df

def calculate_technical_indicators(df: pd.DataFrame, config) -> pd.DataFrame:
    """中文註釋：計算技術指標並返回包含指標的 DataFrame"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在計算技術指標...")
    try:
        if df.empty or not all(col in df.columns for col in ['date', 'close', 'high', 'low', 'volume']):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 數據為空或缺少必要欄位")
            logging.error(f"Data is empty or missing required columns, df_shape={df.shape}, available_columns={df.columns.tolist()}")
            return df
        df_indicators = compute_indicators(df, config)
        required_columns = ['RSI', 'MACD', 'ATR', 'STOCH_k']
        if not all(col in df_indicators.columns for col in required_columns):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 技術指標計算失敗：缺少必要欄位")
            logging.error(f"Failed to compute required indicators, available_columns={df_indicators.columns.tolist()}, df_shape={df_indicators.shape}")
            return df_indicators
        df_indicators.dropna(inplace=True)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 技術指標計算完成")
        logging.info(f"Technical indicators calculated successfully, final_shape={df_indicators.shape}, columns={df_indicators.columns.tolist()}")
        return df_indicators
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 技術指標計算失敗：{str(e)}")
        logging.error(f"Failed to calculate technical indicators: {str(e)}, traceback={traceback.format_exc()}, df_shape={df.shape}, available_columns={df.columns.tolist()}")
        return df

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_fmp_economic_calendar(start_date: str, end_date: str, config) -> pd.DataFrame:
    """中文註釋：從 FMP API 獲取經濟事件日曆"""
    FMP_API_KEY = config['api_keys']['fmp_api_key']
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在從 FMP 獲取經濟事件日曆...")
    logging.info(f"Fetching FMP economic calendar: start={start_date}, end={end_date}, api_key_masked={FMP_API_KEY[:5]}...")
    try:
        proxies = get_system_proxy()
        url = f"https://financialmodelingprep.com/api/v3/economic-calendar?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
        response = requests.get(url, proxies=proxies, timeout=10).json()
        logging.debug(f"FMP response type={type(response)}, content_preview={str(response)[:200]}, proxies={proxies}, url={url}")
        if not isinstance(response, list) or not response:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 經濟事件日曆為空或格式錯誤")
            logging.warning(f"FMP economic calendar non-list or empty: {response}")
            return pd.DataFrame()
        df = pd.DataFrame(response)
        if 'date' not in df.columns:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 經濟事件日曆缺少 'date' 欄位")
            logging.error(f"FMP economic calendar missing 'date' column, available_columns={df.columns.tolist()}")
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'event', 'impact']]
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 經濟事件日曆獲取成功")
        logging.info(f"Successfully fetched economic calendar from FMP: shape={df.shape}, columns={df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} FMP 經濟事件日曆獲取失敗：{str(e)}")
        logging.error(f"Failed to fetch FMP economic calendar: {str(e)}, traceback={traceback.format_exc()}, proxies={proxies}, url={url}")
        try:
            calendar_path = Path(config['system_config']['economic_calendar_path'])
            if not calendar_path.exists():
                calendar_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(columns=['date', 'event', 'impact']).to_csv(calendar_path, index=False)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 已創建空經濟事件日曆：{calendar_path}")
                logging.info(f"Created empty economic calendar: {calendar_path}")
            df = pd.read_csv(calendar_path)
            df['date'] = pd.to_datetime(df['date'])
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 本地經濟事件日曆載入成功")
            logging.info(f"Successfully loaded local economic calendar: shape={df.shape}, columns={df.columns.tolist()}")
            return df
        except Exception as local_e:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 本地經濟事件日曆載入失敗：{str(local_e)}")
            logging.error(f"Failed to load local economic calendar: {str(local_e)}, traceback={traceback.format_exc()}")
            return pd.DataFrame()
    finally:
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)

async def pre_collect_historical_data(timeframe: str, period: str, start_date: str = None, end_date: str = None, config=None) -> pd.DataFrame:
    """中文註釋：預先收集歷史數據並計算技術指標，確保回測期間至少為配置的最小天數"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在預收集 {timeframe} 歷史數據...")
    try:
        # 檢查並調整回測天數
        min_days = config['system_config'].get('min_backtest_days', 180)
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (pd.to_datetime(end_date) - timedelta(days=min_days)).strftime('%Y-%m-%d')
        
        # 驗證並自動調整日期範圍
        try:
            end_date_dt = pd.to_datetime(end_date)
            start_date_dt = pd.to_datetime(start_date)
            date_diff = (end_date_dt - start_date_dt).days
            if date_diff < min_days:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 回測期間太短：{date_diff} 天，自動調整到 {min_days} 天")
                logging.warning(f"Backtest period too short: {date_diff} days, adjusting to {min_days} days")
                start_date_dt = end_date_dt - timedelta(days=min_days)
                start_date = start_date_dt.strftime('%Y-%m-%d')
                logging.info(f"Adjusted start_date to {start_date} to meet minimum {min_days} days")
        except Exception as e:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 日期格式錯誤：{str(e)}，使用預設 {min_days} 天")
            logging.error(f"Invalid date format: {str(e)}, using default {min_days} days")
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (pd.to_datetime(end_date) - timedelta(days=min_days)).strftime('%Y-%m-%d')
        
        # 清理無效模型
        clean_invalid_models(config)
        
        df = load_cached_data(timeframe, period, start_date, end_date, config)
        if df.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 無法獲取 {timeframe} 歷史數據")
            logging.warning(f"Failed to fetch {timeframe} historical data")
            return pd.DataFrame()
        if len(df) < 100:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 數據量不足：{len(df)} 行，需至少 100 行")
            logging.error(f"Insufficient data: {len(df)} rows, required at least 100")
            return pd.DataFrame()
        
        df_indicators = load_cached_indicators(timeframe, config)
        if df_indicators.empty:
            df_indicators = calculate_technical_indicators(df, config)
            if not df_indicators.empty and all(col in df_indicators.columns for col in ['RSI', 'MACD', 'ATR', 'STOCH_k']):
                save_cached_indicators(df_indicators, timeframe, config)
        else:
            df = df.merge(df_indicators, on='date', how='left', suffixes=('', '_ind'))
        
        if not all(col in df.columns for col in ['RSI', 'MACD', 'ATR', 'STOCH_k']):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 合併後缺少必要指標欄位")
            logging.error(f"Missing required indicator columns after merge, available_columns={df.columns.tolist()}, final_shape={df.shape}")
            return df
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {timeframe} 歷史數據和指標準備完成")
        logging.info(f"{timeframe} historical data and indicators prepared, final_shape={df.shape}, columns={df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 預收集 {timeframe} 歷史數據失敗：{str(e)}")
        logging.error(f"Failed to pre-collect {timeframe} historical data: {str(e)}, traceback={traceback.format_exc()}, df_shape={df.shape if 'df' in locals() else 'N/A'}")
        return pd.DataFrame()

if __name__ == "__main__":
    config = load_config()
    initialize_database(config)
    clean_cache(config)