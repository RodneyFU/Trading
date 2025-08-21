# 詳細程式邏輯概述

## 概述
USD/JPY 自動交易系統，模組化設計，包含數據獲取（Yahoo、FMP、SQLite）、指標計算（RSI、MACD 等）、模型預測（XGBoost、LSTM 等）、信號生成和訂單執行。支援回測和實盤交易，注重穩定性（錯誤重試、動態數據源、Proxy 支持）、高效性（並行計算、快取、資料庫）、可維護性（中文註釋、英文日誌）。日誌優化：區分回測 (`backtest_YYYY-MM-DD.log`) 和實盤 (`live_YYYY-MM-DD.log`)，記錄數據形狀、欄位、Proxy 細節，降低除錯難度。

## 模組結構
- **config.py**：載入 JSON 配置，生成 requirements.txt，支援快取。
- **data.py**：數據獲取（Yahoo、FMP、SQLite）、快取管理（USDJPY 命名+動態備份）、指標計算（含 FMP 指標）、經濟事件過濾，支援系統 Proxy。
- **trading.py**：訓練模型（XGBoost、隨機森林、LSTM），生成信號。
- **main.py**：整合模組，執行回測或實盤，生成報告，使用 traceback 處理異常。

## 程式邏輯詳情
### 1. 配置管理（config.py）
- 載入 `api_keys.json`, `trading_params.json`, `system_config.json`，合併為字典。
- 日誌：記錄載入細節、依賴生成、變數值。
- 依賴：生成 `requirements.txt`（pandas、yfinance 等）。
- 錯誤處理：檢查檔案存在和 JSON 格式。

### 2. 數據處理（data.py）
- **數據獲取**：
  - SQLite（`trading_data.db`）優先，支援快速查詢，新增儲存功能（`save_to_database`）。
  - Yahoo Finance（`yfinance`）下載 USDJPY=X OHLC，支援系統 Proxy（`os.environ`）。
  - FMP API 後備（歷史數據+技術指標），使用 `proxies=proxies`，處理非列表回應。
  - tenacity 重試 5 次，日誌記錄參數、回應預覽、形狀、欄位、Proxy 設置，含 `traceback.format_exc()`。
- **快取管理**：
  - 儲存到 `data/historical/timeframe/USDJPY_{timeframe}.pkl`，有效期 1h:24h、4h:48h、1d:7d。
  - 備份到 `data/backup/YYYY/USDJPY_{timeframe}_YYYYMMDD.pkl`，清理 >30 天。
- **資料庫**：
  - 初始化 `ohlc` 和 `indicators` 表格，支援儲存和載入。
- **技術指標**：
  - FMP API 或 `pandas_ta` 計算 RSI、MACD、斐波那契（7 水平）等，驗證數據完整性（移除 NaN）。
  - 日誌記錄新增欄位、形狀。
- **經濟事件**：
  - FMP API 或 CSV，支援 Proxy，日誌記錄形狀、欄位。

### 3. 交易邏輯（trading.py）
- **模型訓練**：
  - XGBoost（短期，100 棵樹）、隨機森林（中期，max_depth=5）、LSTM（長期，PyTorch）。
  - 儲存到 `models/period/`，每 7 天重訓，日誌記錄特徵、形狀、損失。
- **價格預測**：
  - 結合三模型，動態權重（高波動短期 50%），日誌記錄預測值、權重。
- **信號生成**：
  - 多框架（RSI、MACD、STOCH 等）+ OBV + ATR + 經濟事件。
  - 閾值：高波動 0.005，低波動 0.0004，日誌記錄信號細節。

### 4. 主程式（main.py）
- **回測**：
  - 獲取歷史數據，訓練模型，模擬交易。
  - 計算盈虧、勝率、回撤、Sortino。
  - 日誌：記錄數據形狀、欄位、異常（含 traceback）、交易數，儲存於 `backtest_YYYY-MM-DD.log`。
- **實盤**：
  - IB API（port 7497），每 60 秒檢查信號。
  - 括號訂單，動態止損/止盈。
  - 監控 CPU/記憶體（<80%）。
  - 日誌：記錄價格、信號、資源，含 traceback，儲存於 `live_YYYY-MM-DD.log`。
- **報告**：
  - CSV/TXT/HTML，Matplotlib 權益曲線，Chart.js 圖表。

### 5. 性能優化
- 快取/資料庫：減少 API 請求 50%。
- 錯誤重試：tenacity + backoff。
- 動態數據源：優先低延遲來源。
- Proxy 支持：自動檢測並應用系統 Proxy。

### 6. 風險管理邏輯
- 連續虧損：>3 次暫停 24h。
- Sortino：<1.5 暫停。
- 滑點：0.002。
- 訂單超時：900s。
- 持倉時間：ATR 動態調整。
- 經濟事件：高影響暫停 24h。

## 目標
- **穩定性**：錯誤處理、動態數據源、Proxy 支持、詳細日誌（含 shape/columns/traceback）。
- **高效性**：快取、資料庫。
- **可維護性**：中文註釋、英文日誌。
- **靈活性**：回測/實盤切換。