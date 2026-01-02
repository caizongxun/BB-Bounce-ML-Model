#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
實時監控系統
從 Binance 或 yfinance 抓取 K 棒數據
檢測 Bollinger Bands 觸及
自動調用模型進行預測
"""

import pandas as pd
import numpy as np
import requests
import pickle
import json
from datetime import datetime, timedelta
import ta
import time
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print(f"\n{'='*70}")
print(f"實時 BB 反彈預測監控系統")
print(f"{'='*70}\n")

# ============================================================================
# 配置
# ============================================================================

class Config:
    SYMBOLS = [
        'AAVESTDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
        'AVAXUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT',
        'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'LINKUSDT',
        'LTCUSDT', 'MATICUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT',
        'UNIUSDT', 'XRPUSDT'
    ]
    
    TIMEFRAMES = ['15m', '1h']
    BINANCE_BASE_URL = 'https://api.binance.us'
    MODELS_DIR = './models/specialized'
    FALLBACK_MODELS_DIR = './models'
    
    # Bollinger Bands 觸及閾值
    BB_LOWER_THRESHOLD = 1.005  # 下軌觸及：close <= bb_lower * 1.005
    BB_UPPER_THRESHOLD = 0.995  # 上軌觸及：close >= bb_upper * 0.995
    
    # K 棒回溯數
    LOOKBACK = 100
    
    # 監控間隔（秒）
    CHECK_INTERVAL = 60

# ============================================================================
# 模型加載
# ============================================================================

model_cache = {}

def load_model(symbol, timeframe):
    """加載專屬或通用模型"""
    
    model_key = f"{symbol}_{timeframe}"
    
    if model_key in model_cache:
        return model_cache[model_key]
    
    model_data = {}
    
    # 優先級1: 專屬模型
    specialized_path = f'{Config.MODELS_DIR}/model_{model_key}.pkl'
    if Path(specialized_path).exists():
        try:
            with open(specialized_path, 'rb') as f:
                model_data['model'] = pickle.load(f)
            with open(f'{Config.MODELS_DIR}/scaler_{model_key}.pkl', 'rb') as f:
                model_data['scaler'] = pickle.load(f)
            with open(f'{Config.MODELS_DIR}/features_{model_key}.json', 'r') as f:
                model_data['features'] = json.load(f)
            model_cache[model_key] = model_data
            return model_data
        except:
            pass
    
    # 優先級2: 通用優化模型
    try:
        with open(f'{Config.FALLBACK_MODELS_DIR}/best_model_optimized.pkl', 'rb') as f:
            model_data['model'] = pickle.load(f)
        with open(f'{Config.FALLBACK_MODELS_DIR}/scaler_optimized.pkl', 'rb') as f:
            model_data['scaler'] = pickle.load(f)
        with open(f'{Config.FALLBACK_MODELS_DIR}/feature_cols_optimized.json', 'r') as f:
            model_data['features'] = json.load(f)
        model_cache[model_key] = model_data
        return model_data
    except:
        return None

# ============================================================================
# 數據抓取
# ============================================================================

def fetch_klines_binance(symbol, timeframe, limit=100):
    """從 Binance US 抓取 K 棒"""
    
    timeframe_map = {
        '1m': '1m', '5m': '5m', '15m': '15m',
        '1h': '1h', '4h': '4h', '1d': '1d'
    }
    
    interval = timeframe_map.get(timeframe, '15m')
    
    url = f"{Config.BINANCE_BASE_URL}/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        klines = response.json()
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        print(f"[ERROR] 抓取 {symbol} {timeframe} 失敗: {str(e)[:50]}")
        return None

# ============================================================================
# 技術指標計算
# ============================================================================

def add_technical_indicators(df):
    """添加所有技術指標"""
    
    df = df.copy()
    
    try:
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_width_ma'] = df['bb_width'].rolling(20).mean()
        df['bb_width_ratio'] = df['bb_width'] / (df['bb_width_ma'] + 0.0001)
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # ATR
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr.average_true_range()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # 成交量
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 0.0001)
        
        # 動能
        df['roc'] = df['close'].pct_change(5)
        df['momentum'] = df['close'] - df['close'].shift(5)
        
        # 移動平均線
        df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['sma20'] = df['close'].rolling(20).mean()
        
        # K線形態
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_ratio'] = df['body_size'] / (df['high'] - df['low'] + 0.0001)
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['high'] - df['low'] + 0.0001)
        df['high_low_range'] = df['high'] - df['low']
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        
        # 微觀結構
        df['lower_touch_depth'] = (df['bb_lower'] - df['low']) / (df['bb_width'] + 0.0001)
        df['upper_touch_depth'] = (df['high'] - df['bb_upper']) / (df['bb_width'] + 0.0001)
        df['close_from_lower'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.0001)
        df['close_from_upper'] = (df['bb_upper'] - df['close']) / (df['bb_upper'] - df['bb_lower'] + 0.0001)
        df['volume_strength'] = (df['volume'] - df['vol_ma20']) / (df['vol_ma20'] + 0.0001)
        df['volatility_ratio'] = df['atr'] / (df['atr'].rolling(20).mean() + 0.0001)
        df['prev_5_trend'] = (df['close'].shift(5) - df['close'].shift(1)) / (df['close'].shift(1) + 0.0001)
        df['rsi_strength'] = np.abs(df['rsi'] - 50) / 50
        df['macd_strength'] = np.abs(df['macd_hist']) / (np.abs(df['macd_hist']).rolling(20).mean() + 0.0001)
        
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    except Exception as e:
        print(f"[ERROR] 指標計算失敗: {str(e)}")
        return None

# ============================================================================
# 特徵提取和預測
# ============================================================================

def extract_features_from_latest_candle(df):
    """從最新 K 棒提取特徵"""
    
    if len(df) < 50:
        return None
    
    row = df.iloc[-1]
    recent_close = df.iloc[-21:-1]['close']
    timestamp = df.index[-1]
    
    vol_spike_ratio = row['volume'] / (row['vol_ma20'] + 0.0001)
    price_trend = 1 if len(recent_close) >= 2 and recent_close.iloc[-1] > recent_close.iloc[0] else 0
    price_slope = (recent_close.iloc[-1] - recent_close.iloc[0]) / recent_close.iloc[0] if len(recent_close) >= 2 else 0
    bb_position = (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'] + 0.0001)
    
    features = {
        'body_ratio': row['body_ratio'],
        'wick_ratio': row['wick_ratio'],
        'high_low_range': row['high_low_range'],
        'vol_ratio': row['vol_ratio'],
        'vol_spike_ratio': vol_spike_ratio,
        'rsi': row['rsi'],
        'macd': row['macd'],
        'macd_hist': row['macd_hist'],
        'momentum': row['momentum'],
        'bb_width_ratio': row['bb_width_ratio'],
        'bb_position': bb_position,
        'price_trend': price_trend,
        'price_slope': price_slope,
        'hour': timestamp.hour,
        'is_high_volume_time': 1 if (timestamp.hour >= 20 or timestamp.hour < 4) else 0,
        'adx': row['adx'],
        'lower_touch_depth': row['lower_touch_depth'],
        'upper_touch_depth': row['upper_touch_depth'],
        'close_from_lower': row['close_from_lower'],
        'close_from_upper': row['close_from_upper'],
        'volume_strength': row['volume_strength'],
        'volatility_ratio': row['volatility_ratio'],
        'prev_5_trend': row['prev_5_trend'],
        'rsi_strength': row['rsi_strength'],
        'macd_strength': row['macd_strength']
    }
    
    return features

def predict_bounce(model_data, features):
    """調用模型進行預測"""
    
    if model_data is None or features is None:
        return None
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_cols = model_data['features']
    
    try:
        feature_vector = [features.get(col, 0) for col in feature_cols]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_scaled = scaler.transform(feature_vector)
        
        prob = model.predict_proba(feature_scaled)[0]
        success_prob = float(prob[1])
        
        return success_prob
    except:
        return None

def get_confidence_level(prob):
    """評估信心級別"""
    
    if prob > 0.75:
        return "EXCELLENT", 4
    elif prob > 0.65:
        return "GOOD", 3
    elif prob > 0.55:
        return "MODERATE", 2
    elif prob > 0.45:
        return "WEAK", 1
    else:
        return "POOR", 0

# ============================================================================
# 實時監控主程序
# ============================================================================

def monitor_symbol(symbol, timeframe):
    """監控單個幣種的觸及情況"""
    
    # 抓取 K 棒
    df = fetch_klines_binance(symbol, timeframe, limit=Config.LOOKBACK)
    
    if df is None or len(df) < 50:
        return None
    
    # 計算指標
    df = add_technical_indicators(df)
    if df is None:
        return None
    
    # 獲取最新 K 棒
    latest_row = df.iloc[-1]
    close_price = latest_row['close']
    bb_upper = latest_row['bb_upper']
    bb_lower = latest_row['bb_lower']
    timestamp = df.index[-1]
    
    # 檢測觸及
    touched = None
    touch_type = None
    
    if close_price <= bb_lower * Config.BB_LOWER_THRESHOLD:
        touched = True
        touch_type = "LOWER"
    elif close_price >= bb_upper / Config.BB_UPPER_THRESHOLD:
        touched = True
        touch_type = "UPPER"
    
    if not touched:
        return None
    
    # 提取特徵
    features = extract_features_from_latest_candle(df)
    if features is None:
        return None
    
    # 加載模型並預測
    model_data = load_model(symbol, timeframe)
    if model_data is None:
        return None
    
    success_prob = predict_bounce(model_data, features)
    if success_prob is None:
        return None
    
    confidence, confidence_level = get_confidence_level(success_prob)
    
    result = {
        'timestamp': timestamp,
        'symbol': symbol,
        'timeframe': timeframe,
        'close': close_price,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'touch_type': touch_type,
        'success_probability': success_prob,
        'confidence': confidence,
        'confidence_level': confidence_level,
        'rsi': latest_row['rsi'],
        'volume_ratio': latest_row['vol_ratio']
    }
    
    return result

def run_monitoring_loop():
    """運行持續監控循環"""
    
    print(f"[INFO] 開始監控 {len(Config.SYMBOLS)} 個幣種 x {len(Config.TIMEFRAMES)} 個時間框架")
    print(f"[INFO] 監控間隔: {Config.CHECK_INTERVAL} 秒\n")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n[SCAN] 第 {iteration} 次掃描 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 70)
            
            signals = []
            
            for symbol in Config.SYMBOLS:
                for timeframe in Config.TIMEFRAMES:
                    result = monitor_symbol(symbol, timeframe)
                    
                    if result is not None:
                        signals.append(result)
                        
                        confidence_color = "[HIGH]"
                        if result['confidence_level'] >= 3:
                            confidence_color = "[HIGH]"
                        elif result['confidence_level'] == 2:
                            confidence_color = "[MED]"
                        else:
                            confidence_color = "[LOW]"
                        
                        print(f"[SIGNAL] {result['symbol']} {result['timeframe']:>3} {confidence_color} | "
                              f"觸及: {result['touch_type']:>5} | "
                              f"成功率: {result['success_probability']:.2%} | "
                              f"信心: {result['confidence']}")
            
            if len(signals) == 0:
                print("[INFO] 暫無觸及信號")
            else:
                print(f"\n[SUMMARY] 本次掃描發現 {len(signals)} 個信號")
                
                # 按成功率排序
                signals_sorted = sorted(signals, key=lambda x: x['success_probability'], reverse=True)
                
                print("\n[TOP SIGNALS] 最佳信號 (按成功率排序):")
                for idx, sig in enumerate(signals_sorted[:5], 1):
                    print(f"  {idx}. {sig['symbol']} {sig['timeframe']} - {sig['success_probability']:.2%} "
                          f"({sig['confidence']})")
            
            print(f"\n[WAIT] 等待 {Config.CHECK_INTERVAL} 秒後進行下次掃描...")
            time.sleep(Config.CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n[INFO] 監控已停止")
    except Exception as e:
        print(f"\n[ERROR] 監控循環錯誤: {str(e)}")

# ============================================================================
# 執行
# ============================================================================

if __name__ == '__main__':
    print("[INFO] 開始初始化...")
    print(f"[INFO] 監控 {len(Config.SYMBOLS)} 個幣種")
    print(f"[INFO] 支持時間框架: {', '.join(Config.TIMEFRAMES)}")
    print(f"[INFO] 模型目錄: {Config.MODELS_DIR}")
    
    run_monitoring_loop()
