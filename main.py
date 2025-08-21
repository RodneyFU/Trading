import pandas as pd
import numpy as np
import asyncio
import logging
import logging.handlers
from datetime import datetime, timedelta
import argparse
from pathlib import Path
from config import load_config
from data import pre_collect_historical_data, fetch_fmp_economic_calendar
from trading import train_short_term_model, train_medium_term_model, train_long_term_model, predict_price
from ib_insync import IB, Forex, util
import psutil
import matplotlib.pyplot as plt
from tabulate import tabulate

# 中文註釋：載入配置
config = load_config()
ROOT_DIR = Path(config['system_config']['root_dir'])
REPORTS_DIR = ROOT_DIR / config['system_config']['reports_dir']
MODEL_DIR = ROOT_DIR / config['system_config']['model_dir']
TRADING_PARAMS = config['trading_params']

# 中文註釋：確保日誌目錄存在並設置日誌
log_dir = ROOT_DIR / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

# 中文註釋：記錄路徑初始化
print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 路徑初始化完成：ROOT_DIR={ROOT_DIR}, REPORTS_DIR={REPORTS_DIR}, MODEL_DIR={MODEL_DIR}")
logging.info(f"Path initialization completed: ROOT_DIR={ROOT_DIR}, REPORTS_DIR={REPORTS_DIR}, MODEL_DIR={MODEL_DIR}")

async def monitor_positions(ib, trades, df_1h, current_price, session):
    """中文註釋：監控持倉並執行交易邏輯。"""
    try:
        if df_1h.empty or not all(col in df_1h.columns for col in ['close', 'RSI']):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 數據為空或缺少必要欄位")
            logging.error("Data is empty or missing required columns")
            return trades
        
        logging.debug(f"Monitoring positions: df_1h shape={df_1h.shape}, current_price={current_price}, session={session}")
        
        for _, row in df_1h.iterrows():
            if row['RSI'] > TRADING_PARAMS['rsi_overbought']:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} RSI 超買，賣出信號")
                logging.info("RSI overbought, sell signal")
                trades.append({'type': 'sell', 'price': current_price, 'time': datetime.now()})
            elif row['RSI'] < TRADING_PARAMS['rsi_oversold']:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} RSI 超賣，買入信號")
                logging.info("RSI oversold, buy signal")
                trades.append({'type': 'buy', 'price': current_price, 'time': datetime.now()})
        return trades
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 監控持倉失敗：{str(e)}")
        logging.error(f"Failed to monitor positions: {str(e)}, traceback={util.format_exc()}")
        return trades

async def run_backtest(start_date, end_date):
    """中文註釋：執行回測流程。"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 開始回測：{start_date} 至 {end_date}")
    logging.info(f"[BACKTEST] Starting backtest: start_date={start_date}, end_date={end_date}")
    
    try:
        logging.info("[BACKTEST] Fetching economic calendar...")
        economic_calendar = fetch_fmp_economic_calendar(start_date, end_date, config)
        logging.info(f"[BACKTEST] Economic calendar fetched: shape={economic_calendar.shape if not economic_calendar.empty else 'empty'}")
        
        logging.info("[BACKTEST] Collecting 1h historical data...")
        df_1h = await pre_collect_historical_data('1 hour', '30d', start_date, end_date, config)
        logging.info(f"[BACKTEST] 1h data collected: shape={df_1h.shape if not df_1h.empty else 'empty'}")
        
        logging.info("[BACKTEST] Collecting 4h historical data...")
        df_4h = await pre_collect_historical_data('4 hours', '30d', start_date, end_date, config)
        logging.info(f"[BACKTEST] 4h data collected: shape={df_4h.shape if not df_4h.empty else 'empty'}")
        
        logging.info("[BACKTEST] Collecting 1d historical data...")
        df_1d = await pre_collect_historical_data('1 day', '30d', start_date, end_date, config)
        logging.info(f"[BACKTEST] 1d data collected: shape={df_1d.shape if not df_1d.empty else 'empty'}")
        
        if df_1h.empty or df_4h.empty or df_1d.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 歷史數據為空，無法進行回測")
            logging.error("[BACKTEST] Historical data is empty, cannot proceed with backtest")
            return
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在訓練短期模型...")
        short_term_model = train_short_term_model(df_1h)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在訓練中期模型...")
        medium_term_model = train_medium_term_model(df_4h)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在訓練長期模型...")
        long_term_model = train_long_term_model(df_1d)
        
        trades = []
        session = None
        trades = await monitor_positions(None, trades, df_1h, df_1h['close'].iloc[-1], session)
        
        report = pd.DataFrame(trades)
        report.to_csv(REPORTS_DIR / 'backtest_report.csv', index=False)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 回測報告已生成：{REPORTS_DIR / 'backtest_report.csv'}")
        logging.info(f"[BACKTEST] Backtest report generated: {REPORTS_DIR / 'backtest_report.csv'}")
        
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 回測失敗：{str(e)}")
        logging.error(f"[BACKTEST] Backtest failed: {str(e)}, traceback={util.format_exc()}")
        
async def run_live_trading():
    """中文註釋：執行實盤交易流程。"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 開始實盤交易...")
    logging.info("[LIVE] Starting live trading...")
    ib = IB()
    try:
        ib.connect('127.0.0.1', config['system_config']['ib_params']['port'], clientId=1)
        while True:
            df_1h = await pre_collect_historical_data('1 hour', '1d')
            if not df_1h.empty:
                trades = []
                trades = await monitor_positions(ib, trades, df_1h, df_1h['close'].iloc[-1], ib)
            await asyncio.sleep(60)
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 實盤交易失敗：{str(e)}")
        logging.error(f"[LIVE] Live trading failed: {str(e)}, traceback={util.format_exc()}")
    finally:
        ib.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USD/JPY 交易系統")
    parser.add_argument('--backtest', action='store_true', help='執行回測')
    parser.add_argument('--start', type=str, help='回測開始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='回測結束日期 (YYYY-MM-DD)')
    parser.add_argument('--y', type=int, help='回測年數')
    args = parser.parse_args()
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 中文註釋：動態選擇日誌檔案並配置旋轉
    if args.backtest:
        log_file = log_dir / 'backtest.log'
    else:
        log_file = log_dir / 'live_trading.log'
    if not log_file.is_file():
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 日誌檔案初始化\n")
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    
    if args.backtest:
        start_date = args.start or (datetime.now() - timedelta(days=args.y * 365 if args.y else 100)).strftime('%Y-%m-%d')
        end_date = args.end or datetime.now().strftime('%Y-%m-%d')
        asyncio.run(run_backtest(start_date, end_date))
    else:
        asyncio.run(run_live_trading())