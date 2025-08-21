# 程式邏輯概述

## 概述
系統由 `main.py`, `config.py`, `data.py`, `trading.py` 組成，實現 USD/JPY 交易。數據儲存於 SQLite，1 小時框架每 30 分鐘保存，4 小時/1 天框架每 3 小時保存，重啟時載入最近 7 天數據，每日備份。

## 核心模組

### 1. main.py
- **功能**：
  - 解析命令列參數（`--backtest`, `--live`）。
  - `run_backtest`：模擬交易，保存至 `C:\Trading\reports`。
  - `run_live_trading`：每小時更新，每 30 分鐘/3 小時保存。
- **保存策略**：
  - `save_data_periodically`：異步任務，動態頻率保存。
  - 每日備份資料庫。
  - 重啟時通過 `load_from_database` 導入數據。

### 2. config.py
- **功能**：載入配置，確保 `db_path`。

### 3. data.py
- **資料庫管理**：
  - 初始化 `ohlc`, `indicators`, `economic_data`, `sentiment_data`。
  - 增量保存（僅新數據）。
  - 載入最近 7 天數據。
  - 每日備份。
- **數據獲取**：Yahoo Finance, FMP, FRED, Forex Factory, X API。
- **指標計算**：RSI, MACD, ATR 等。

### 4. trading.py
- **功能**：執行交易（`execute_trade`）。
- **保存**：交易記錄儲存於 `reports`。

## 資料流
1. **輸入**：時間框架、日期範圍、API 鍵。
2. **處理**：
   - 從 SQLite/快取/API 獲取數據。
   - 計算指標，合併情緒數據。
   - 動態頻率增量保存。
3. **輸出**：
   - 回測：CSV 報告。
   - 實盤：交易執行。

## 錯誤處理
- **重試**：API 重試 5 次。
- **日誌**：記錄至 `C:\Trading\logs\backtest.log`。
- **鎖定**：SQLite 連線 `timeout=10`。

## 注意事項
- **高頻交易**：每 30 分鐘保存。
- **備份**：每日檢查備份。
- **清理**：每月刪除舊數據。