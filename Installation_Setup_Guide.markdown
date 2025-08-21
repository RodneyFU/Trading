# 安裝與初始設置指導

## 概述
本指導說明如何設置 USD/JPY 交易系統。

## 系統要求
- **作業系統**：Windows 10/11。
- **硬體**：4GB RAM，500MB 儲存。
- **網路**：穩定連線。

## 安裝步驟

### 1. 安裝 Python
- 下載 Python 3.10：https://www.python.org/downloads/
- 勾選「Add Python to PATH」。
- 驗證：
  ```bash
  python --version
  pip --version
  ```

### 2. 設置虛擬環境
- 創建並啟動：
  ```bash
  python -m venv C:\Trading\venv
  C:\Trading\venv\Scripts\activate
  ```

### 3. 安裝依賴
- 安裝套件：
  ```bash
  pip install pandas requests yfinance beautifulsoup4 tenacity textblob
  ```
- 驗證：
  ```bash
  pip list
  ```

### 4. 設置目錄結構
- 創建目錄：
  ```bash
  mkdir C:\Trading
  mkdir C:\Trading\config
  mkdir C:\Trading\data
  mkdir C:\Trading\data\historical
  mkdir C:\Trading\data\backup
  mkdir C:\Trading\logs
  mkdir C:\Trading\reports
  mkdir C:\Trading\models
  ```

### 5. 配置 API 鍵
- 創建 `C:\Trading\config\api_keys.json`：
  ```json
  {
      "fmp_api_key": "YOUR_FMP_API_KEY",
      "fred_api_key": "YOUR_FRED_API_KEY",
      "x_bearer_token": "YOUR_X_BEARER_TOKEN"
  }
  ```
- 創建 `C:\Trading\config\system_config.json`：
  ```json
  {
      "root_dir": "C:\\Trading",
      "db_path": "C:\\Trading\\data\\trading_data.db",
      "min_backtest_days": 180,
      "indicators": {
          "RSI": true,
          "MACD": true,
          "ATR": true,
          "Stochastic": true,
          "Bollinger": true,
          "EMA": true
      },
      "proxies": {}
  }
  ```
- 創建 `C:\Trading\config\trading_params.json`：
  ```json
  {
      "risk_per_trade": 0.02,
      "rr_ratio": 2,
      "max_position_size": 10000
  }
  ```

### 6. 保存程式碼
- 下載 `main.py`, `config.py`, `data.py`, `trading.py` 至 `C:\Trading`。
- 來源：https://github.com/RodneyFU/Trading

### 7. 匯入 CSV
- 若有 `C:\Trading\data\economic_calendar.csv`：
  ```python
  from config import load_config
  from data import import_to_database
  config = load_config()
  import_to_database(config)
  ```

### 8. 驗證設置
- 測試：
  ```python
  from config import load_config
  from data import fetch_fmp_economic_calendar
  config = load_config()
  df = fetch_fmp_economic_calendar('2025-05-13', '2025-08-21', config)
  print(df.head())
  ```

## 注意事項
- **API 配額**：檢查 FMP 和 X API 限制。
- **備份**：每日檢查 `C:\Trading\data\backup`。
- **日誌**：檢查 `C:\Trading\logs\backtest.log`。