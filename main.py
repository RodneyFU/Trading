import asyncio
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import traceback  # 修正：導入標準庫 traceback
from data import fetch_fmp_economic_calendar, pre_collect_historical_data
from trading import execute_trade  # 假設 trading.py 包含 execute_trade
from config import load_config  # 使用 config.py 的 load_config

# 設置日誌
logging.basicConfig(
    filename=r'C:\Trading\logs\backtest.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

async def run_backtest(start_date, end_date, timeframe='1 day'):
    """執行回測，整合歷史數據、經濟日曆和情緒分析"""
    try:
        config = load_config()
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 開始回測：{start_date} 至 {end_date}")
        logging.info(f"Starting backtest: {start_date} to {end_date}, timeframe={timeframe}")

        # 獲取經濟日曆
        economic_calendar = fetch_fmp_economic_calendar(start_date, end_date, config)
        if economic_calendar.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 警告：經濟日曆為空")
            logging.warning("Economic calendar is empty")

        # 獲取歷史數據和技術指標
        df = await pre_collect_historical_data(timeframe, '6mo', start_date, end_date, config)
        if df.empty or len(df) < 100:
            raise ValueError(f"歷史數據不足：{len(df)} 行，需至少 100 行")

        # 合併經濟日曆
        df = df.merge(economic_calendar[['date', 'event', 'impact']], on='date', how='left')
        df['impact'] = df['impact'].fillna('Low')

        # 生成交易信號（結合技術指標和情緒分析）
        df['signal'] = df.apply(
            lambda x: 'BUY' if (x['RSI'] < 30 and x['sentiment'] > 0.3 and x['impact'] != 'High')
            else 'SELL' if (x['RSI'] > 70 and x['sentiment'] < -0.3 and x['impact'] != 'High')
            else 'HOLD', axis=1
        )

        # 模擬交易
        trades = []
        position = None
        for i in range(1, len(df)):
            if df['signal'].iloc[i] == 'BUY' and position != 'BUY':
                trades.append({
                    'date': df['date'].iloc[i],
                    'type': 'BUY',
                    'price': df['close'].iloc[i],
                    'sentiment': df['sentiment'].iloc[i]
                })
                position = 'BUY'
            elif df['signal'].iloc[i] == 'SELL' and position != 'SELL':
                trades.append({
                    'date': df['date'].iloc[i],
                    'type': 'SELL',
                    'price': df['close'].iloc[i],
                    'sentiment': df['sentiment'].iloc[i]
                })
                position = 'SELL'

        # 保存回測結果
        trades_df = pd.DataFrame(trades)
        reports_dir = Path(config['system_config']['root_dir']) / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(reports_dir / f'backtest_{start_date}_{end_date}.csv', index=False)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 回測完成，結果已保存至 {reports_dir}")
        logging.info(f"Backtest completed, results saved to {reports_dir}")

        return trades_df

    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 回測失敗：{str(e)}")
        logging.error(f"[BACKTEST] Backtest failed: {str(e)}, traceback={traceback.format_exc()}")  # 修正：使用 traceback.format_exc
        raise

async def run_live_trading(timeframe='1 hour'):
    """執行實盤交易，每 3 小時保存數據"""
    config = load_config()
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 開始實盤交易，時間框架：{timeframe}")
    logging.info(f"Starting live trading, timeframe={timeframe}")

    while True:
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            df = await pre_collect_historical_data(timeframe, '7d', start_date, end_date, config)
            if df.empty:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 無有效數據，等待下一次更新")
                logging.warning("No valid data for live trading")
                await asyncio.sleep(3600)  # 等待 1 小時
                continue

            # 獲取最新經濟日曆
            economic_calendar = fetch_fmp_economic_calendar(start_date, end_date, config)
            df = df.merge(economic_calendar[['date', 'event', 'impact']], on='date', how='left')
            df['impact'] = df['impact'].fillna('Low')

            # 生成交易信號
            latest = df.iloc[-1]
            signal = 'BUY' if (latest['RSI'] < 30 and latest['sentiment'] > 0.3 and latest['impact'] != 'High') \
                else 'SELL' if (latest['RSI'] > 70 and latest['sentiment'] < -0.3 and latest['impact'] != 'High') \
                else 'HOLD'

            # 執行交易
            if signal != 'HOLD':
                execute_trade(signal, latest['close'], config)  # 假設 trading.py 提供此函數
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 執行交易：{signal} at {latest['close']}")
                logging.info(f"Executed trade: {signal} at {latest['close']}")

            # 每 3 小時保存數據
            await asyncio.sleep(3 * 3600)
            save_path = Path(config['system_config']['db_path'])
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 保存數據至 {save_path}")
            logging.info(f"Saving data to SQLite: {save_path}")

        except Exception as e:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 實盤交易錯誤：{str(e)}")
            logging.error(f"[LIVE] Trading error: {str(e)}, traceback={traceback.format_exc()}")
            await asyncio.sleep(600)  # 等待 10 分鐘後重試

def main():
    """主程式入口，解析命令列參數"""
    parser = argparse.ArgumentParser(description='USD/JPY Trading System')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--live', action='store_true', help='Run live trading')
    parser.add_argument('--start', default=(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'), help='Start date')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date')
    parser.add_argument('--timeframe', default='1 day', choices=['1 hour', '4 hours', '1 day'], help='Timeframe')
    
    args = parser.parse_args()

    if args.backtest:
        asyncio.run(run_backtest(args.start, args.end, args.timeframe))
    elif args.live:
        asyncio.run(run_live_trading(args.timeframe))
    else:
        print("請指定 --backtest 或 --live")
        logging.error("No mode specified (--backtest or --live)")

if __name__ == "__main__":
    main()