#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
實時 BB 反彈預測系統
從 Binance 抨客署抨作真實 K 棒並預測
"""

import pandas as pd
import numpy as np
import pickle
import json
import requests
import ta
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')

print(f"\n{'='*60}")
print(f"實時 BB 反彈預測系統")
print(f"{'='*60}\n")

# ============================================================================
# 配置
# ============================================================================

class Config:
    MODEL_DIR = './models'
    BINANCE_BASE_URL = 'https://api.binance.us'  # Binance US 公開 API
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # 策略測試的幣種
    TIMEFRAMES = ['15m', '1h', '4h']  # 时间框
    LOOKBACK = 100  # 抨戱回顾的根K棒數
    TOUCH_SENSITIVITY = 0.995  # BB 觸及牰先寶詠
    SUCCESS_THRESHOLD = 0.5  # 反彈成功的最低前上往支

# ============================================================================
# 加載模型
# ============================================================================

print("步驟1: 加載优化版本模型")
print("-" * 60)

try:
    with open(f'{Config.MODEL_DIR}/best_model_optimized.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{Config.MODEL_DIR}/scaler_optimized.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{Config.MODEL_DIR}/feature_cols_optimized.json', 'r') as f:
        feature_cols = json.load(f)
    print("✅ 优化模型加載成功")
    print(f"   特弶数: {len(feature_cols)}")
except:
    print("❌ 优化模型未找到，恢復使用原始模型")
    try:
        with open(f'{Config.MODEL_DIR}/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{Config.MODEL_DIR}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'{Config.MODEL_DIR}/feature_cols.json', 'r') as f:
            feature_cols = json.load(f)
        print("✅ 原始模型加載成功")
    except:
        print("❌ 模型不存在，請先運行訓練脚本")
        exit()

# ============================================================================
# Binance 數據推遭
# ============================================================================

def fetch_klines(symbol, timeframe, limit=100):
    """從 Binance API 推遭 K 棒數據"""
    
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
        
        # 轉換數據類型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        print(f"   ❌ 推遭 {symbol} {timeframe} 失敗: {str(e)}")
        return None

# ============================================================================
# 計算技術指標
# ============================================================================

def add_technical_indicators(df):
    """添加技術指標"""
    df = df.copy()
    
    # Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_width_ma'] = df['bb_width'].rolling(20).mean()
    df['bb_width_ratio'] = df['bb_width'] / (df['bb_width_ma'] + 0.0001)
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # MACD
    macd_indicator = ta.trend.MACD(df['close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()
    
    # ATR
    atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr_indicator.average_true_range()
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
    df['sma200'] = df['close'].rolling(200).mean()
    
    # K線形態
    df['body_size'] = abs(df['close'] - df['open'])
    df['body_ratio'] = df['body_size'] / (df['high'] - df['low'] + 0.0001)
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['high'] - df['low'] + 0.0001)
    df['high_low_range'] = df['high'] - df['low']
    
    # ADX
    adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_indicator.adx()
    
    # 新增特徵
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

# ============================================================================
# 提取觸及特徵
# ============================================================================

def extract_features_from_bar(df, index):
    """從提定的根K棒提取所有特徵"""
    
    if index < 50:
        return None
    
    row = df.iloc[index]
    recent_close = df.iloc[max(0, index-20):index]['close']
    timestamp = df.index[index]
    
    features = {
        'body_ratio': row['body_ratio'],
        'wick_ratio': row['wick_ratio'],
        'high_low_range': row['high_low_range'],
        'vol_ratio': row['vol_ratio'],
        'vol_spike_ratio': row['volume'] / (row['vol_ma20'] + 0.0001),
        'rsi': row['rsi'],
        'macd': row['macd'],
        'macd_hist': row['macd_hist'],
        'momentum': row['momentum'],
        'bb_width_ratio': row['bb_width_ratio'],
        'bb_position': (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'] + 0.0001),
        'price_trend': 1 if len(recent_close) >= 2 and recent_close.iloc[-1] > recent_close.iloc[0] else 0,
        'price_slope': (recent_close.iloc[-1] - recent_close.iloc[0]) / recent_close.iloc[0] if len(recent_close) >= 2 else 0,
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

# ============================================================================
# 預測函數
# ============================================================================

def predict_bounce(features):
    """預測 BB 反彈成功概率"""
    
    # 構建特徵向量
    feature_vector = []
    for col in feature_cols:
        feature_vector.append(features.get(col, 0))
    
    feature_vector = np.array(feature_vector).reshape(1, -1)
    feature_scaled = scaler.transform(feature_vector)
    
    prob = model.predict_proba(feature_scaled)[0]
    success_prob = float(prob[1])
    
    return success_prob

# ============================================================================
# 主程序
# ============================================================================

def run_real_time_predictions():
    """實時推鄭一次并顯示結果"""
    
    print("步驟2: 推遭實時 K 棒數據")
    print("-" * 60)
    
    all_results = []
    
    for symbol in Config.SYMBOLS:
        print(f"\n{symbol}:")
        
        for timeframe in Config.TIMEFRAMES:
            # 推遭 K 棒
            df = fetch_klines(symbol, timeframe, limit=Config.LOOKBACK)
            
            if df is None:
                continue
            
            # 計算技術指標
            df = add_technical_indicators(df)
            
            # 取最新根K棒
            latest_idx = len(df) - 1
            features = extract_features_from_bar(df, latest_idx)
            
            if features is None:
                continue
            
            # 預測
            success_prob = predict_bounce(features)
            
            # 判斷是否觸及了 BB
            latest_row = df.iloc[latest_idx]
            bb_touched_lower = latest_row['close'] <= latest_row['bb_lower'] * Config.TOUCH_SENSITIVITY
            bb_touched_upper = latest_row['close'] >= latest_row['bb_upper'] / Config.TOUCH_SENSITIVITY
            bb_touched = bb_touched_lower or bb_touched_upper
            touch_type = '下軌' if bb_touched_lower else '上軌' if bb_touched_upper else '沒觸及'
            
            # 顯示結果
            status = '✅' if success_prob > 0.5 else '❌'
            print(f"   {status} {timeframe:>3} | 成功率: {success_prob:.2%} | 觸及: {touch_type}")
            
            # 保存結果
            all_results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': df.index[latest_idx],
                'close': float(latest_row['close']),
                'bb_upper': float(latest_row['bb_upper']),
                'bb_lower': float(latest_row['bb_lower']),
                'success_probability': success_prob,
                'bb_touched': bb_touched,
                'touch_type': touch_type,
                'rsi': float(latest_row['rsi']),
                'volume_ratio': float(latest_row['vol_ratio'])
            })
    
    return all_results

# ============================================================================
# 執行
# ============================================================================

if __name__ == '__main__':
    try:
        results = run_real_time_predictions()
        
        print(f"\n步驟3: 結果汇總")
        print("-" * 60)
        
        # 轉換為 DataFrame 以便便查看
        results_df = pd.DataFrame(results)
        
        # 篩選高調度的信號
        high_confidence = results_df[results_df['success_probability'] > 0.6]
        
        if len(high_confidence) > 0:
            print(f"\n高調度信號 ({len(high_confidence)} 個):")
            for idx, row in high_confidence.iterrows():
                print(f"   {row['symbol']} {row['timeframe']:>3} | 成功率: {row['success_probability']:.2%}")
        else:
            print("\n很遗憾，目前沒有高調度的信號")
        
        # 保存每小時的結果
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df.to_csv(f'predictions_{timestamp_str}.csv', index=False)
        print(f"\n✅ 下次結果已保存至: predictions_{timestamp_str}.csv")
        
    except Exception as e:
        print(f"\n❌ 錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
