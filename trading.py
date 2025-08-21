import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from config import load_config

# 中文註釋：載入配置
config = load_config()
ROOT_DIR = Path(config['system_config']['root_dir'])
MODEL_DIR = ROOT_DIR / config['system_config']['model_dir']
MODEL_PERIODS = config['system_config']['model_periods']
INDICATORS = config['system_config']['indicators']
PRICE_DIFF_THRESHOLD = config['trading_params']['price_diff_threshold']
RSI_OVERBOUGHT = config['trading_params']['rsi_overbought']
RSI_OVERSOLD = config['trading_params']['rsi_oversold']
STOCH_OVERBOUGHT = config['trading_params']['stoch_overbought']
STOCH_OVERSOLD = config['trading_params']['stoch_oversold']
ADX_THRESHOLD = config['trading_params']['adx_threshold']
OBV_WINDOW = config['trading_params']['obv_window']

class LSTMModel(nn.Module):
    """LSTM 模型定義，用於長期價格預測。"""
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_short_term_model(df: pd.DataFrame) -> XGBRegressor:
    """訓練短期 XGBoost 模型。"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在訓練短期模型...")
    try:
        if df.empty or not all(col in df.columns for col in ['RSI', 'MACD', 'ATR', 'STOCH_k']):
            logging.error(f"Invalid input data for short-term model: shape={df.shape}, columns={df.columns.tolist()}")
            return None
        features = []
        for indicator in INDICATORS:
            if INDICATORS[indicator] and indicator in df.columns:
                features.append(indicator)
        X = df[features].dropna()
        y = df['close'].shift(-1).dropna()
        X = X.iloc[:len(y)]
        
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X, y)
        
        model_dir = MODEL_DIR / 'short_term'
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / 'xgboost_model.pkl')
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 短期模型訓練完成並儲存")
        logging.info("Short-term XGBoost model trained and saved")
        return model
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 短期模型訓練失敗：{str(e)}, traceback={traceback.format_exc()}")
        logging.error(f"Failed to train short-term model: {str(e)}, traceback={traceback.format_exc()}")
        return None

def train_medium_term_model(df: pd.DataFrame) -> RandomForestRegressor:
    """訓練中期隨機森林模型。"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在訓練中期模型...")
    try:
        if df.empty or not all(col in df.columns for col in ['RSI', 'MACD', 'ATR', 'STOCH_k']):
            logging.error(f"Invalid input data for short-term model: shape={df.shape}, columns={df.columns.tolist()}")
            return None
        features = []
        for indicator in INDICATORS:
            if INDICATORS[indicator] and indicator in df.columns:
                features.append(indicator)
        X = df[features].dropna()
        y = df['close'].shift(-1).dropna()
        X = X.iloc[:len(y)]
        
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        model_dir = MODEL_DIR / 'medium_term'
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / 'rf_model.pkl')
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 中期模型訓練完成並儲存")
        logging.info("Medium-term RandomForest model trained and saved")
        return model
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 中期模型訓練失敗：{str(e)}, traceback={traceback.format_exc()}")
        logging.error(f"Failed to train medium-term model: {str(e)}, traceback={traceback.format_exc()}")
        return None

def train_long_term_model(df: pd.DataFrame) -> LSTMModel:
    """訓練長期 LSTM 模型。"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在訓練長期模型...")
    try:
        if df.empty or not all(col in df.columns for col in ['RSI', 'MACD', 'ATR', 'STOCH_k']):
            logging.error(f"Invalid input data for short-term model: shape={df.shape}, columns={df.columns.tolist()}")
            return None
        features = []
        for indicator in INDICATORS:
            if INDICATORS[indicator] and indicator in df.columns:
                features.append(indicator)
        X = df[features].dropna().values
        y = df['close'].shift(-1).dropna().values
        X = X[:len(y)]
        
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        
        model = LSTMModel(input_size=len(features), hidden_size=50, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        model_dir = MODEL_DIR / 'long_term'
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / 'lstm_model.pth')
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 長期模型訓練完成並儲存")
        logging.info("Long-term LSTM model trained and saved")
        return model
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 長期模型訓練失敗：{str(e)}, traceback={traceback.format_exc()}")
        logging.error(f"Failed to train long-term model: {str(e)}, traceback={traceback.format_exc()}")
        return None

def load_models(df: pd.DataFrame) -> dict:
    """載入所有模型，若模型不存在則重新訓練。"""
    models = {}
    for period in MODEL_PERIODS:
        model_dir = MODEL_DIR / period
        if period == 'short_term':
            model_path = model_dir / 'xgboost_model.pkl'
            if model_path.exists():
                models[period] = joblib.load(model_path)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 載入 {period} 模型")
                logging.info(f"Loaded {period} model")
            else:
                models[period] = train_short_term_model(df)
        elif period == 'medium_term':
            model_path = model_dir / 'rf_model.pkl'
            if model_path.exists():
                models[period] = joblib.load(model_path)
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 載入 {period} 模型")
                logging.info(f"Loaded {period} model")
            else:
                models[period] = train_medium_term_model(df)
        elif period == 'long_term':
            model_path = model_dir / 'lstm_model.pth'
            if model_path.exists():
                model = LSTMModel(input_size=len([col for col in df.columns if col in INDICATORS and INDICATORS[col]]), hidden_size=50, num_layers=2)
                model.load_state_dict(torch.load(model_path))
                models[period] = model
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 載入 {period} 模型")
                logging.info(f"Loaded {period} model")
            else:
                models[period] = train_long_term_model(df)
    return models

def predict_price(models: dict, df: pd.DataFrame, session: str) -> tuple:
    """預測下一期價格，結合多模型結果。"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在預測價格...")
    try:
        features = []
        for indicator in INDICATORS:
            if INDICATORS[indicator] and indicator in df.columns:
                features.append(indicator)
        X = df[features].iloc[-1:].dropna()
        if X.empty:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 無有效特徵數據，無法預測")
            logging.warning("No valid feature data for prediction")
            return None, None, None, None
        
        weights = {'short_term': 0.5, 'medium_term': 0.3, 'long_term': 0.2} if session == 'high_volatility' else {'short_term': 0.2, 'medium_term': 0.3, 'long_term': 0.5}
        
        predictions = {}
        for period, model in models.items():
            if period == 'long_term':
                X_tensor = torch.tensor(X.values, dtype=torch.float32).unsqueeze(0)
                model.eval()
                with torch.no_grad():
                    pred = model(X_tensor).item()
            else:
                pred = model.predict(X)[0]
            predictions[period] = pred
        
        final_pred = sum(weights[period] * pred for period, pred in predictions.items())
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 價格預測完成：{final_pred:.4f}")
        logging.info(f"Price prediction completed: {final_pred:.4f}")
        return final_pred, predictions['short_term'], predictions['medium_term'], predictions['long_term']
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 價格預測失敗：{str(e)}")
        logging.error(f"Failed to predict price: {str(e)}")
        return None, None, None, None

def combine_timeframes(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_daily: pd.DataFrame, current_price: float, predicted_price: float, session: str) -> tuple:
    """結合多時間框架生成交易信號。"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 正在生成交易信號...")
    buy_signal = False
    sell_signal = False
    
    price_diff = (predicted_price - current_price) / current_price
    price_diff_threshold = PRICE_DIFF_THRESHOLD[session]
    
    obv_diff = df_1h['OBV'].diff(OBV_WINDOW).iloc[-1] if 'OBV' in df_1h.columns else 0
    obv_trend = obv_diff > 0
    
    try:
        if 'RSI' in df_1h.columns and 'RSI' in df_4h.columns and 'RSI' in df_daily.columns:
            rsi_1h = df_1h['RSI'].iloc[-1]
            rsi_4h = df_4h['RSI'].iloc[-1]
            rsi_daily = df_daily['RSI'].iloc[-1]
            rsi_buy = rsi_1h < RSI_OVERSOLD and rsi_4h < RSI_OVERSOLD and rsi_daily < RSI_OVERSOLD
            rsi_sell = rsi_1h > RSI_OVERBOUGHT and rsi_4h > RSI_OVERBOUGHT and rsi_daily > RSI_OVERBOUGHT
        else:
            rsi_buy = rsi_sell = False
        
        if 'MACD' in df_1h.columns and 'MACD_Signal' in df_1h.columns and 'MACD' in df_4h.columns and 'MACD_Signal' in df_4h.columns:
            macd_1h = df_1h['MACD'].iloc[-1] > df_1h['MACD_Signal'].iloc[-1] and df_1h['MACD'].iloc[-2] <= df_1h['MACD_Signal'].iloc[-2]
            macd_4h = df_4h['MACD'].iloc[-1] > df_4h['MACD_Signal'].iloc[-1] and df_4h['MACD'].iloc[-2] <= df_4h['MACD_Signal'].iloc[-2]
            macd_daily = df_daily['MACD'].iloc[-1] > df_daily['MACD_Signal'].iloc[-1] if 'MACD' in df_daily.columns else False
            macd_buy = macd_1h and macd_4h and macd_daily
            macd_sell = not macd_1h and not macd_4h and not macd_daily
        else:
            macd_buy = macd_sell = False
        
        if 'STOCH_k' in df_1h.columns and 'STOCH_d' in df_1h.columns:
            stoch_1h = df_1h['STOCH_k'].iloc[-1] < STOCH_OVERSOLD and df_1h['STOCH_d'].iloc[-1] < STOCH_OVERSOLD
            stoch_4h = df_4h['STOCH_k'].iloc[-1] < STOCH_OVERSOLD if 'STOCH_k' in df_4h.columns else False
            stoch_buy = stoch_1h and stoch_4h
            stoch_sell = df_1h['STOCH_k'].iloc[-1] > STOCH_OVERBOUGHT and df_1h['STOCH_d'].iloc[-1] > STOCH_OVERBOUGHT
        else:
            stoch_buy = stoch_sell = False
        
        if 'ADX' in df_daily.columns:
            adx_trend = df_daily['ADX'].iloc[-1] > ADX_THRESHOLD
        else:
            adx_trend = False
        
        if 'ICHIMOKU_tenkan' in df_daily.columns:
            ichimoku_buy = df_daily['close'].iloc[-1] > df_daily['ICHIMOKU_senkou_a'].iloc[-1] and df_daily['close'].iloc[-1] > df_daily['ICHIMOKU_senkou_b'].iloc[-1]
            ichimoku_sell = df_daily['close'].iloc[-1] < df_daily['ICHIMOKU_senkou_a'].iloc[-1] and df_daily['close'].iloc[-1] < df_daily['ICHIMOKU_senkou_b'].iloc[-1]
        else:
            ichimoku_buy = ichimoku_sell = False
        
        buy_signal = (price_diff > price_diff_threshold and rsi_buy and macd_buy and obv_trend and adx_trend and ichimoku_buy) or (stoch_buy and price_diff > price_diff_threshold * 0.5)
        sell_signal = (price_diff < -price_diff_threshold and rsi_sell and macd_sell and not obv_trend and adx_trend and ichimoku_sell) or (stoch_sell and price_diff < -price_diff_threshold * 0.5)
        
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 交易信號生成完成：買入 {buy_signal}, 賣出 {sell_signal}")
        logging.info(f"Trading signals generated: Buy {buy_signal}, Sell {sell_signal}")
        return buy_signal, sell_signal
    except Exception as e:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 交易信號生成失敗：{str(e)}")
        logging.error(f"Failed to generate trading signals: {str(e)}")