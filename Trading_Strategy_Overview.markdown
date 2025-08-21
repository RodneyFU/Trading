# USD/JPY 交易策略概述

## 概述
本交易策略針對 USD/JPY 貨幣對，整合技術指標、經濟事件和 X 平台情緒分析，支援回測和實盤交易。數據儲存於 SQLite（`ohlc`, `economic_data`, `sentiment_data`），1 小時框架每 30 分鐘增量保存，4 小時或 1 天框架每 3 小時保存，程式重啟時載入最近 7 天數據，每日備份資料庫。

## 核心組件

### 1. 技術指標
- **RSI**：超買（>70）/超賣（<30）。
- **MACD**：趨勢確認。
- **ATR**：動態止損/止盈。
- **Stochastic**：反轉確認。
- **Bollinger Bands**：突破交易。
- **EMA**：12天/26天交叉。

**信號邏輯**：
- **買入**：`0.7 * (0.5 * RSI_signal + 0.5 * MACD_signal) + 0.3 * sentiment > 0.3`, `impact not in ['High', 'Medium']`。
- **賣出**：`0.7 * (0.5 * RSI_signal + 0.5 * MACD_signal) + 0.3 * sentiment < -0.3`, `impact not in ['High', 'Medium']`。
- **止損**：2 * ATR。
- **止盈**：風險回報比 1:2。

### 2. 經濟事件
- 來源：SQLite（`economic_data`），備用 FMP/FRED/Forex Factory。
- **過濾**：排除高/中影響事件。
- **篩選**：僅 USD/JPY 事件。

### 3. X 平台情緒分析
- 數據：從 SQLite 或 X API 獲取，範圍 -1 至 1。
- **應用**：
  - 權重：技術指標 70%，情緒 30%。
  - 閾值：基準 0.3，測試 0.2/0.4/0.5。
  - 異常檢測：連續 3 天 `|sentiment| > 0.5` 暫停交易。
  - 倉位調整：`sentiment > 0.4` 增倉 20%，`< -0.4` 減倉 20%。
- **相關性**：記錄情緒與價格相關性。

### 4. 風險管理
- **倉位**：風險 ≤ 2%（`risk_per_trade`）。
- **止損**：基於 ATR。
- **事件風險**：高/中影響事件暫停。
- **情緒異常**：極端情緒暫停。

### 5. SQLite 數據管理
- **保存**：1 小時框架每 30 分鐘，4 小時/1 天框架每 3 小時，增量保存。
- **導入**：重啟時載入最近 7 天數據。
- **備份**：每日備份至 `C:\Trading\data\backup`。
- **清理**：每月刪除超過 1 年數據。

## 交易流程
1. **數據準備**：
   - 從 SQLite 載入數據，若無則抓取 API 並保存。
2. **信號生成**：
   ```python
   signal_score = 0.7 * (0.5 * rsi_signal + 0.5 * macd_signal) + 0.3 * sentiment
   signal = 'BUY' if (signal_score > 0.3 and impact not in ['High', 'Medium']) else 'SELL' if (signal_score < -0.3 and impact not in ['High', 'Medium']) else 'HOLD'
   ```
3. **交易執行**：
   - 回測：模擬交易，保存至 `C:\Trading\reports`。
   - 實盤：調用 `execute_trade`，動態倉位。
4. **數據保存**：
   - 動態頻率增量保存。
   - 重啟導入最近數據。

## 利益最大化
- **短期**：1 小時框架，每 30 分鐘保存，情緒 + ATR。
- **中期**：7 天情緒均值 + EMA。
- **優化**：回測情緒閾值（0.2-0.5）。
- **驗證**：比較 X 情緒與 Investing.com。

## 注意事項
- **API 配額**：監控 FMP 和 X API 限制。
- **備份**：每日檢查備份。
- **日誌**：檢查 `C:\Trading\logs\backtest.log`。