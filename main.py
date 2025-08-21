import asyncio
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import traceback
import sqlite3
from data import fetch_fmp_economic_calendar, pre_collect_historical_data, save_to_database, load_from_database, backup_database
from trading import execute_trade
from config import load_config

# 設置日誌
logging.basicConfig(
    filename=r'C:\Trading\logs\backtest.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s'
)

async def save_data_periodically(df_buffer, timeframe, config):
    """動態頻率保存數據至 SQLite"""
    save_interval = 1800 if timeframe == '1 hour' else 3 * 3600  # 1 小時框架每 30 分鐘，否則 3 小時
    while True:
        try:
            if not df_buffer.empty:
                save_to_database(df_buffer, timeframe, config)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 數據已增量保存至 SQLite")
                logging.info(f"Data incrementally saved to SQLite: timeframe={timeframe}")
            # 每日備份（凌晨 00:00）
            if datetime.now().hour == 0 and datetime.now().minute < 5:
                backup_database(config)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 資料庫已備份")
                logging.info("Database backed up")
            await asyncio.sleep(save_interval)
        except Exception as e:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 定期保存失敗：{str(e)}")
            logging.error(f"Periodic save failed: {str(e)}, traceback={traceback.format_exc()}")

async def run_backtest(start_date, end_date, timeframe='1 day'):
    """執行回測"""
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

        # 生成交易信號（情緒權重提升至 30%）
        df['rsi_signal'] = df['RSI'].apply(lambda x: 1 if x < 30 else -1 if x > 70 else 0)
        df['macd_signal'] = df['MACD'].apply(lambda x: 1 if x > df['MACD_signal'] else -1 if x < df['MACD_signal'] else 0)
        df['signal'] = df.apply(
            lambda x: 'BUY' if (0.7 * (0.5 * x['rsi_signal'] + 0.5 * x['macd_signal']) + 0.3 * x['sentiment'] > 0.3 and x['impact'] not in ['High', 'Medium'])
            else 'SELL' if (0.7 * (0.5 * x['rsi_signal'] + 0.5 * x['macd_signal']) + 0.3 * x['sentiment'] < -0.3 and x['impact'] not in ['High', 'Medium'])
            else 'HOLD', axis=1
        )

        # 檢查情緒異常
        if (df['sentiment'][-3:] > 0.5).all() or (df['sentiment'][-3:] < -0.5).all():
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 警告：檢測到極端情緒，暫停交易")
            logging.warning("Extreme sentiment detected, pausing trading")
            df['signal'] = 'HOLD'

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
        logging.error(f"[BACKTEST] Backtest failed: {str(e)}, traceback={traceback.format_exc()}")
        raise

async def run_live_trading(timeframe='1 hour'):
    """執行實盤交易，動態保存數據"""
    config = load_config()
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 開始實盤交易，時間框架：{timeframe}")
    logging.info(f"Starting live trading, timeframe={timeframe}")

    # 啟動定期保存任務
    df_buffer = pd.DataFrame()
    asyncio.create_task(save_data_periodically(df_buffer, timeframe, config))

    while True:
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            df = await pre_collect_historical_data(timeframe, '7d', start_date, end_date, config)
            if df.empty:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 無有效數據，等待下一次更新")
                logging.warning("No valid data for live trading")
                await asyncio.sleep(3600)
                continue

            # 更新數據緩衝區
            df_buffer = pd.concat([df_buffer, df]).drop_duplicates(subset=['date', 'timeframe']).reset_index(drop=True)

            # 獲取最新經濟日曆
            economic_calendar = fetch_fmp_economic_calendar(start_date, end_date, config)
            df = df.merge(economic_calendar[['date', 'event', 'impact']], on='date', how='left')
            df['impact'] = df['impact'].fillna('Low')

            # 生成交易信號
            latest = df.iloc[-1]
            rsi_signal = 1 if latest['RSI'] < 30 else -1 if latest['RSI'] > 70 else 0
            macd_signal = 1 if latest['MACD'] > latest['MACD_signal'] else -1 if latest['MACD'] < latest['MACD_signal'] else 0
            signal_score = 0.7 * (0.5 * rsi_signal + 0.5 * macd_signal) + 0.3 * latest['sentiment']
            signal = 'BUY' if (signal_score > 0.3 and latest['impact'] not in ['High', 'Medium']) \
                else 'SELL' if (signal_score < -0.3 and latest['impact'] not in ['High', 'Medium']) \
                else 'HOLD'

            # 檢查情緒異常
            if (df['sentiment'][-3:] > 0.5).all() or (df['sentiment'][-3:] < -0.5).all():
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 警告：檢測到極端情緒，暫停交易")
                logging.warning("Extreme sentiment detected, pausing trading")
                signal = 'HOLD'

            # 動態倉位
            base_size = config['trading_params']['max_position_size'] * config['trading_params']['risk_per_trade']
            position_size = base_size * (1.2 if latest['sentiment'] > 0.4 else 0.8 if latest['sentiment'] < -0.4 else 1.0)

            # 執行交易
            if signal != 'HOLD':
                execute_trade(signal, latest['close'], config, position_size=position_size)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 執行交易：{signal} at {latest['close']}, 倉位大小：{position_size}")
                logging.info(f"Executed trade: {signal} at {latest['close']}, position_size={position_size}")

            await asyncio.sleep(3600)

        except Exception as e:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 實盤交易錯誤：{str(e)}")
            logging.error(f"[LIVE] Trading error: {str(e)}, traceback={traceback.format_exc()}")
            await asyncio.sleep(600)

def main():
    """主程式入口"""
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