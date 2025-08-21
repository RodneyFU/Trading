# USD/JPY 自動化交易系統 - 回測與實盤交易運行步驟

本文件指導如何運行 USD/JPY 自動化交易系統的回測或實盤交易，步驟簡化，適合非專業人士，確保邏輯完整且易於操作。假設已完成「安裝與初始設置指導」中的環境配置。

## 前置條件
- **檔案**：config.py、data.py、trading.py、main.py 位於 C:\Trading。
- **設定檔**：C:\Trading\config\api_keys.txt 包含有效 FMP_API_KEY。
- **經濟日曆**：C:\Trading\economic_calendar.csv 包含高影響事件。
- **IB Gateway**（實盤）：已安裝並運行，端口 7497（模擬）或 7496（真實帳戶）。
- **依賴**：requirements.txt 中的模組已安裝（pandas、yfinance、ib_insync 等）。

## 運行回測
回測模擬歷史交易，根據指定日期範圍生成交易報告，無需連線 IB Gateway。

1. **打開命令提示字元**：
   - Windows：按 Win+R，輸入 `cmd`，按 Enter。
   - 進入程式目錄：輸入 `cd C:\Trading`。

2. **運行回測**：
   - **預設回測**（昨日前 6 個月）：
     ```bash
     python main.py --backtest
     ```
     - 開始日期：昨日前 6 個月。
     - 結束日期：昨日。
     - 控制台顯示：如「回測日期範圍：2025-02-19 到 2025-08-19」。
   - **指定月數**（例如 12 個月）：
     ```bash
     python main.py --backtest --months 12
     ```
     - 開始日期：昨日前 12 個月。
   - **指定年數**（例如 2 年）：
     ```bash
     python main.py --backtest --years 2
     ```
     - 開始日期：昨日前 2 年。
   - **指定日期範圍**（例如 2024-01-01 到 2024-12-31）：
     ```bash
     python main.py --backtest --start_date 2024-01-01 --end_date 2024-12-31
     ```

3. **檢查輸出**：
   - **控制台**：顯示進度，如「正在執行回測從 2024-01-01 到 2024-12-31...」「回測完成」。
   - **報告**：生成於 C:\Trading\reports，格式為：
     - daily_report_YYYYMMDD.txt：交易明細表格。
     - daily_report_YYYYMMDD.csv：CSV 格式數據。
     - report_YYYYMMDD.html：簡單盈虧圖表。
   - **日誌**：檢查 C:\Trading\live_trading.log，確認無錯誤（如 "Insufficient data"）。

4. **注意事項**：
   - 確保網路穩定，FMP/Yahoo Finance API 可正常訪問。
   - 若數據不足（< 60 筆），控制台顯示「資料筆數不足」，檢查日期範圍或 API 密鑰。
   - 回測假設每日以預測價平倉，盈虧考慮滑點（0.0003）。

## 運行實盤交易
實盤交易連線 IB Gateway，根據即時數據執行買/賣訂單。

1. **啟動 IB Gateway**：
   - 打開 IB Gateway，登錄模擬或真實帳戶。
   - 確認端口（模擬：7497；真實：7496，需修改 main.py 的 IB_PORT 若不同）。
   - 確保 API 連線啟用，允許本地連線（127.0.0.1）。

2. **運行實盤交易**：
   - 在命令提示字元中，進入 C:\Trading，輸入：
     ```bash
     python main.py
     ```
   - 控制台顯示：如「正在連線 IB Gateway...」「當前 USD/JPY 價格：145.1234」「交易執行：BUY @ 145.1234」。

3. **交易邏輯**：
   - **數據**：每 60 秒獲取即時價格，更新 1 小時（30 天）、4 小時（90 天）、1 天（1 年）數據。
   - **信號**：結合多時間框架指標、機器學習預測生成買/賣信號。
   - **訂單**：提交括號訂單（市場單 + 止損 + 止盈），訂單大小 25000 USD（按價格調整）。
   - **風險管理**：
     - 止損：ATR * 1.5；止盈：ATR * 2.0；追蹤止損：ATR * 1.0。
     - 最大持倉 24 小時，超時平倉。
     - 經濟事件前 24 小時暫停交易。
     - 連續虧損 > 5 次或 Sortino < 0.5 時重新訓練模型。
   - **報告**：每日生成於 C:\Trading\reports，格式同回測。

4. **監控與停止**：
   - **監控**：檢查控制台訊息和 live_trading.log，確認連線、價格更新、訂單執行正常。
   - **停止**：按 Ctrl+C 終止程式，系統自動斷開 IB Gateway 連線。
   - **資源檢查**：若 CPU/記憶體 > 80%，程式退出並記錄錯誤。

## 注意事項
- **回測**：
  - 日期格式必須為 YYYY-MM-DD，否則報錯。
  - 若指定日期範圍無效（例如未來日期），程式會失敗並記錄日誌。
- **實盤**：
  - 確保 IB Gateway 運行且帳戶資金充足（預設資本 50000 USD）。
  - 檢查網路穩定性，避免 API 或 IB 連線中斷。
  - 定期檢查 C:\Trading\backup，確保數據安全。
- **日誌與錯誤**：
  - 若出現「無法獲取價格」或「資料不足」，檢查 API 密鑰、網路或 IB Gateway 設置。
  - 日誌（live_trading.log）提供詳細除錯資訊（如 "Failed to fetch price"）。
- **收益最大化**：
  - 系統保留多指標確認、多模型預測、動態止損等邏輯，確保穩健交易。
  - 可透過調整 --months 或 --years 測試不同回測範圍，優化策略。

(English Supplement: This guide details steps to run backtesting or live trading for the USD/JPY system. Backtesting uses Args for flexible date ranges (default: 6 months ending yesterday), generating reports in C:\Trading\reports. Live trading connects to IB Gateway, executes bracket orders, and includes risk management. Logs and console messages aid monitoring.)