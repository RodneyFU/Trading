# 回測與實盤交易運行步驟指導

## 概述
本指導說明如何運行 USD/JPY 交易系統。

## 回測步驟

### 1. 準備環境
- 完成《安裝與初始設置指導》。
- 驗證 `C:\Trading\data\trading_data.db`。
- 清除無效快取：
  ```bash
  del C:\Trading\data\historical\*\*.pkl
  ```

### 2. 運行回測
- 執行：
  ```bash
  C:\Trading\venv\Scripts\activate
  python C:\Trading\main.py --backtest --start 2025-05-13 --end 2025-08-21 --timeframe "1 day"
  ```

### 3. 檢查結果
- **日誌**：檢查 `C:\Trading\logs\backtest.log`。
- **資料庫**：
  ```python
  import sqlite3
  import pandas as pd
  conn = sqlite3.connect(r'C:\Trading\data\trading_data.db')
  print(pd.read_sql_query("SELECT * FROM ohlc LIMIT 5", conn))
  print(pd.read_sql_query("SELECT * FROM sentiment_data LIMIT 5", conn))
  conn.close()
  ```
- **報告**：檢查 `C:\Trading\reports\backtest_*.csv`。

### 4. 優化
- 調整 `trading_params.json`。
- 回測情緒閾值（0.2-0.5）：
  ```python
  for thresh in [0.2, 0.4, 0.5]:
      df['signal'] = df.apply(lambda x: 'BUY' if (0.7 * (0.5 * x['rsi_signal'] + 0.5 * x['macd_signal']) + 0.3 * x['sentiment'] > thresh) else 'SELL' if (0.7 * (0.5 * x['rsi_signal'] + 0.5 * x['macd_signal']) + 0.3 * x['sentiment'] < -thresh) else 'HOLD', axis=1)
  ```

## 實盤交易步驟

### 1. 準備環境
- 確保 API 鍵有效。
- 更新 SQLite：
  ```python
  from config import load_config
  from data import fetch_fmp_economic_calendar, fetch_x_data
  config = load_config()
  fetch_fmp_economic_calendar('2025-08-14', '2025-08-21', config)
  fetch_x_data('USDJPY', '2025-08-14', '2025-08-21', config)
  ```

### 2. 運行實盤
- 執行：
  ```bash
  C:\Trading\venv\Scripts\activate
  python C:\Trading\main.py --live --timeframe "1 hour"
  ```

### 3. 監控
- **日誌**：檢查 `C:\Trading\logs\backtest.log`。
- **資料庫**：驗證 `ohlc` 和 `sentiment_data` 更新。
- **備份**：檢查 `C:\Trading\data\backup`。

## 注意事項
- **保存頻率**：1 小時框架每 30 分鐘保存。
- **備份**：每日檢查備份。
- **網路**：確保穩定連線。