#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多幣种x多時間框模型訓練
為每個特分的幣种時間框組合訓練模型
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import json
from huggingface_hub import hf_hub_download
import ta
import warnings

warnings.filterwarnings('ignore')

print(f"\n{'='*60}")
print(f"多幣种 x 多時間框模型訓練")
print(f"{'='*60}\n")

# ============================================================================
# 配置
# ============================================================================

class Config:
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    TIMEFRAMES = ['15m', '1h', '4h']
    HF_REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    DATA_CACHE_DIR = './data_cache'
    MODELS_DIR = './models'

# ============================================================================
# 數據推遭函數
# ============================================================================

def download_from_hf(symbol, timeframe):
    """從 Hugging Face 推遭數據"""
    file_path = f"klines/{symbol}/{symbol.split('USDT')[0]}_{timeframe}.parquet"
    
    try:
        local_path = hf_hub_download(
            repo_id=Config.HF_REPO_ID,
            filename=file_path,
            repo_type="dataset",
            cache_dir=Config.DATA_CACHE_DIR
        )
        
        df = pd.read_parquet(local_path)
        
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                except:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
        
        df.columns = df.columns.str.lower()
        return df
    except:
        return None

def add_technical_indicators(df):
    """添加技術指標"""
    df = df.copy()
    
    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_middle'] = bb_indicator.bollinger_mavg()
    df['bb_lower'] = bb_indicator.bollinger_lband()
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_width_ma'] = df['bb_width'].rolling(20).mean()
    df['bb_width_ratio'] = df['bb_width'] / (df['bb_width_ma'] + 0.0001)
    
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    macd_indicator = ta.trend.MACD(df['close'])
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()
    
    atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr_indicator.average_true_range()
    df['atr_ratio'] = df['atr'] / df['close']
    
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 0.0001)
    
    df['roc'] = df['close'].pct_change(5)
    df['momentum'] = df['close'] - df['close'].shift(5)
    
    df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    
    df['body_size'] = abs(df['close'] - df['open'])
    df['body_ratio'] = df['body_size'] / (df['high'] - df['low'] + 0.0001)
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['high'] - df['low'] + 0.0001)
    df['high_low_range'] = df['high'] - df['low']
    
    adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_indicator.adx()
    
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

def create_labels(df, symbol, timeframe):
    """生成標籤"""
    labels_list = []
    
    for i in range(len(df) - 6):
        current_row = df.iloc[i]
        close_price = current_row['close']
        bb_upper = current_row['bb_upper']
        bb_lower = current_row['bb_lower']
        
        if close_price <= bb_lower * 1.005:
            future_prices = df.iloc[i:i+6]['high']
            max_price = future_prices.max()
            price_increase_pct = ((max_price - close_price) / close_price) * 100
            is_success = 1 if price_increase_pct > 0.5 else 0
            labels_list.append({'index': i, 'label': is_success, 'type': 'lower'})
        
        if close_price >= bb_upper * 0.995:
            future_prices = df.iloc[i:i+6]['low']
            min_price = future_prices.min()
            price_decrease_pct = ((close_price - min_price) / close_price) * 100
            is_success = 1 if price_decrease_pct > 0.5 else 0
            labels_list.append({'index': i, 'label': is_success, 'type': 'upper'})
    
    return pd.DataFrame(labels_list)

def extract_features(df, labels, timeframe):
    """提取特徵"""
    features_list = []
    
    feature_cols_names = [
        'body_ratio', 'wick_ratio', 'high_low_range', 'vol_ratio', 'vol_spike_ratio',
        'rsi', 'macd', 'macd_hist', 'momentum', 'bb_width_ratio', 'bb_position',
        'price_trend', 'price_slope', 'hour', 'is_high_volume_time', 'adx',
        'lower_touch_depth', 'upper_touch_depth', 'close_from_lower', 'close_from_upper',
        'volume_strength', 'volatility_ratio', 'prev_5_trend', 'rsi_strength', 'macd_strength'
    ]
    
    for _, label_row in labels.iterrows():
        idx = label_row['index']
        if idx < 50:
            continue
        
        current_row = df.iloc[idx]
        recent_close = df.iloc[max(0, idx-20):idx]['close']
        timestamp = df.index[idx]
        
        vol_spike_ratio = current_row['volume'] / (current_row['vol_ma20'] + 0.0001)
        price_trend = 1 if len(recent_close) >= 2 and recent_close.iloc[-1] > recent_close.iloc[0] else 0
        price_slope = (recent_close.iloc[-1] - recent_close.iloc[0]) / recent_close.iloc[0] if len(recent_close) >= 2 else 0
        bb_position = (current_row['close'] - current_row['bb_lower']) / (current_row['bb_upper'] - current_row['bb_lower'] + 0.0001)
        
        feature_dict = {
            'body_ratio': current_row['body_ratio'],
            'wick_ratio': current_row['wick_ratio'],
            'high_low_range': current_row['high_low_range'],
            'vol_ratio': current_row['vol_ratio'],
            'vol_spike_ratio': vol_spike_ratio,
            'rsi': current_row['rsi'],
            'macd': current_row['macd'],
            'macd_hist': current_row['macd_hist'],
            'momentum': current_row['momentum'],
            'bb_width_ratio': current_row['bb_width_ratio'],
            'bb_position': bb_position,
            'price_trend': price_trend,
            'price_slope': price_slope,
            'hour': timestamp.hour,
            'is_high_volume_time': 1 if (timestamp.hour >= 20 or timestamp.hour < 4) else 0,
            'adx': current_row['adx'],
            'lower_touch_depth': current_row['lower_touch_depth'],
            'upper_touch_depth': current_row['upper_touch_depth'],
            'close_from_lower': current_row['close_from_lower'],
            'close_from_upper': current_row['close_from_upper'],
            'volume_strength': current_row['volume_strength'],
            'volatility_ratio': current_row['volatility_ratio'],
            'prev_5_trend': current_row['prev_5_trend'],
            'rsi_strength': current_row['rsi_strength'],
            'macd_strength': current_row['macd_strength'],
            'label': label_row['label']
        }
        
        features_list.append(feature_dict)
    
    return pd.DataFrame(features_list)

# ============================================================================
# 訓練主程序
# ============================================================================

print("步驟1: 推遭數據並訓練")
print("-" * 60)

training_results = {}

for symbol in Config.SYMBOLS:
    for timeframe in Config.TIMEFRAMES:
        print(f"\n{symbol} {timeframe}...", end='', flush=True)
        
        # 推遭數據
        df = download_from_hf(symbol, timeframe)
        if df is None:
            print(" ❌ 推遭失敗")
            continue
        
        # 計算指標
        df = add_technical_indicators(df)
        
        # 生成標籤
        labels = create_labels(df, symbol, timeframe)
        
        # 提取特徵
        features_df = extract_features(df, labels, timeframe)
        
        if len(features_df) < 100:
            print(f" ❌ 样本太少 ({len(features_df)})")
            continue
        
        # 準備訓練数据
        X = features_df.drop('label', axis=1)
        y = features_df['label']
        X = X.fillna(X.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 訓練模型
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
        
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            max_depth=6,
            learning_rate=0.08,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # 評估
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # 保存模型
        model_id = f"{symbol}_{timeframe}"
        with open(f'{Config.MODELS_DIR}/model_{model_id}.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open(f'{Config.MODELS_DIR}/scaler_{model_id}.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(f'{Config.MODELS_DIR}/features_{model_id}.json', 'w') as f:
            json.dump(X.columns.tolist(), f)
        
        training_results[model_id] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'samples': len(features_df)
        }
        
        print(f" ✅ AUC={auc:.4f} Recall={recall:.2%}")

# ============================================================================
# 結果餐韩
# ============================================================================

print(f"\n\n步驟2: 結果汇總")
print("-" * 60)

results_df = pd.DataFrame(training_results).T
results_df = results_df.sort_values('auc', ascending=False)

print("\n模型性能排名:")
print("\n{:15} | {:10} | {:10} | {:10} | {:10}".format('ID', 'AUC', 'Recall', 'Precision', 'F1'))
print("-" * 60)
for model_id, row in results_df.iterrows():
    print("{:15} | {:10.4f} | {:10.2%} | {:10.2%} | {:10.4f}".format(
        model_id, row['auc'], row['recall'], row['precision'], row['f1']
    ))

print(f"\n✅ 已保存 {len(training_results)} 個模型")
print(f"\n下一步: 在實時預測中使用這些模型")
