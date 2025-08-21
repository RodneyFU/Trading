# USD/JPY 自動化交易系統 - 安裝與初始設置指導

本文件指導安裝和設置 USD/JPY 自動化交易系統，適合非專業人士，確保系統可運行回測或實盤交易。已更新以自動創建必要檔案和目錄，解決 `FileNotFoundError` 和 `ValueError: Invalid period format` 等問題。

## 環境要求
- **作業系統**：Windows 10/11（其他系統需調整路徑）。
- **硬體**：CPU 使用率 < 80%，記憶體 > 8GB（建議 16GB），硬碟空間 > 10GB。
- **Python**：3.8 或以上（已確認 3.10）。
- **Interactive Brokers Gateway**：用於實盤交易，端口 7497（模擬）或 7496（真實帳戶）。
- **網路**：穩定連線，支援 FMP API、Yahoo Finance 和 PyPI。

## 步驟 1：安裝 Python 和依賴
1. **下載並安裝 Python**：
   - 訪問 [python.org](https://www.python.org)，下載 Python 3.8 或以上。
   - 安裝時勾選「Add Python to PATH」。
   - 運行以下命令確認版本：
     ```bash
     python --version
     ```
     - 應顯示類似 `Python 3.10.x`。
2. **創建虛擬環境**：
   - 進入程式目錄：
     ```bash
     cd c:\Trading
     ```
   - 創建並啟動虛擬環境：
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
     - 確認提示符顯示 `(venv)`。
3. **安裝依賴**：
   - 將程式碼檔案（`config.py`、`data.py`、`trading.py`、`main.py`）儲存至 `C:\Trading`。
   - 運行 `config.py` 初始化結構並生成 `requirements.txt`：
     ```bash
     python config.py
     ```
     - 應顯示「創建目錄」、「已創建 api_keys.txt」、「已創建 economic_calendar.csv」、「設定檔載入成功！」。
     - 生成 `C:\Trading\config\requirements.txt`，內容：
       ```
       tenacity
       pandas>=1.0.0
       numpy==1.23.5
       yfinance
       requests
       pandas_ta==0.3.14b0
       torch
       xgboost
       scikit-learn
       joblib
       psutil
       ib_insync
       tabulate
       backoff
       ```
   - 安裝依賴：
     ```bash
     pip install -r C:\Trading\config\requirements.txt
     ```
   - 若失敗，逐一安裝：
     ```bash
     pip install --upgrade pip
     pip install tenacity pandas>=1.0.0 numpy==1.23.5 yfinance requests pandas_ta==0.3.14b0 torch xgboost scikit-learn joblib psutil ib_insync tabulate backoff
     ```
   - 驗證安裝：
     ```bash
     pip show numpy pandas_ta pandas
     ```
     - 應顯示：
       - `numpy: 1.23.5`
       - `pandas_ta: 0.3.14b0`
       - `pandas: >=1.0.0`

## 步驟 2：設置目錄結構和檔案
1. **自動創建目錄和檔案**：
   - 運行 `python config.py` 自動創建：
     - `C:\Trading`：根目錄。
     - `C:\Trading\config`：存放 `api_keys.txt` 和 `requirements.txt`。
     - `C:\Trading\historical_data`：歷史數據快取。
     - `C:\Trading\indicators`：指標快取。
     - `C:\Trading\models`：模型檔案。
     - `C:\Trading\backup`：數據備份。
     - `C:\Trading\reports`：交易報告。
     - `C:\Trading\config\api_keys.txt`：FMP API 密鑰範本。
     - `C:\Trading\economic_calendar.csv`：預設經濟事件。
   - 檢查目錄：
     ```bash
     dir C:\Trading
     dir C:\Trading\config
     ```
2. **編輯 api_keys.txt**：
   - 打開 `C:\Trading\config\api_keys.txt`，填寫 FMP API 密鑰：
     ```
     FMP_API_KEY=您的_FMP_API_密鑰
     ```
   - 從 [Financial Modeling Prep](https://site.financialmodelingprep.com/) 獲取密鑰。
3. **驗證 economic_calendar.csv**：
   - 打開 `C:\Trading\economic_calendar.csv`，預設內容：
     ```csv
     date,event
     2025-08-01 08:30:00,US Non-Farm Payroll
     2025-08-15 14:00:00,US CPI Release
     ```
   - 可從 [Investing.com](https://www.investing.com/economic-calendar/) 添加更多事件。
4. **檢查權限**：
   - 確保 `C:\Trading` 和子目錄可寫：
     ```bash
     echo test > C:\Trading\test.txt
     echo test > C:\Trading\config\test.txt
     ```
   - 若失敗，右鍵 `C:\Trading`，選擇「屬性」->「安全」，授予「完全控制」。

## 步驟 3：設置 Interactive Brokers（實盤交易）
1. **下載 IB Gateway**：
   - 從 [Interactive Brokers](https://www.interactivebrokers.com) 下載並安裝。
2. **配置帳戶**：
   - 開設模擬（端口 7497）或真實帳戶（端口 7496）。
   - 啟用 API 連線，允許本地連線（127.0.0.1）。
3. **驗證連線**：
   - 運行 `python main.py`（非回測模式），檢查 `C:\Trading\live_trading.log` 是否顯示「Connected to IB Gateway」。

## 步驟 4：驗證設置
1. **檢查檔案和目錄**：
   - 確認存在：
     ```bash
     dir C:\Trading\config\api_keys.txt
     dir C:\Trading\economic_calendar.csv
     dir C:\Trading\historical_data
     dir C:\Trading\indicators
     dir C:\Trading\models
     dir C:\Trading\backup
     dir C:\Trading\reports
     ```
   - 確認 `api_keys.txt` 包含有效 FMP_API_KEY。
2. **運行測試**：
   - 在虛擬環境中：
     ```bash
     python config.py
     ```
     - 應顯示「創建目錄」、「已創建 api_keys.txt」、「已創建 economic_calendar.csv」、「設定檔載入成功！」。
     - 若顯示「無權限創建」，檢查目錄權限。
3. **檢查依賴**：
   - 輸入 `pip list`，確認包含 `numpy==1.23.5`、`pandas_ta==0.3.14b0` 等。
4. **檢查日誌**：
   - 打開 `C:\Trading\live_trading.log`，確認無錯誤（如 `FileNotFoundError` 或 `ValueError`）。

## 除錯建議
- **檔案或目錄缺失**：
  - 運行 `python config.py` 自動創建。
  - 手動檢查：
    ```bash
    dir C:\Trading\config\api_keys.txt
    dir C:\Trading\economic_calendar.csv
    ```
- **權限問題**：
  - 以管理員身份運行：
    ```bash
    runas /user:Administrator cmd
    cd C:\Trading
    python config.py
    ```
- **FMP API 連線超時**：
  - 測試 API：
    ```bash
    curl "https://financialmodelingprep.com/api/v3/historical-chart/1hour/USDJPY?from=2025-08-19&to=2025-08-19&apikey=您的_FMP_API_KEY"
    ```
  - 若失敗，檢查網路或獲取新密鑰。
- **ValueError: Invalid period format**：
  - 確認 `main.py` 傳遞正確的時間框架（`1 hour`、`4 hours`、`1 day`）。
  - 檢查 `data.py` 中 `pre_collect_historical_data` 的參數處理。

## 注意事項
- **安全**：保護 `api_keys.txt`，勿公開 FMP API 密鑰。
- **備份**：定期檢查 `C:\Trading\backup`。
- **Python 版本**：使用 3.8+（已確認 3.10）。
- **環境變數**：若設置 `TRADING_CONFIG_PATH`，確保指向正確路徑。

完成後，參閱「回測與實盤交易運行步驟」。

(English Supplement: Updated to include automatic file/directory creation, parameter validation, and fixes for `ValueError: Invalid period format`. Ensures robust setup for backtesting and live trading.)