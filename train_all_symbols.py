#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练所有22个币种的优化模型
每个币种训练 15m 和 1h 两个模型
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
from pathlib import Path

warnings.filterwarnings('ignore')

print(f"\n{'='*70}")
print(f"训练所有22个币种的优化模型系统")
print(f"{'='*70}\n")

# ============================================================================
# 配置
# ============================================================================

class Config:
    # 22个币种
    SYMBOLS = [
        'AAVESTDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
        'AVAXUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT',
        'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'LINKUSDT',
        'LTCUSDT', 'MATICUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT',
        'UNIUSDT', 'XRPUSDT'
    ]
    
    TIMEFRAMES = ['15m', '1h']  # 训练 15 分鐘和 1 小时
    HF_REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    DATA_CACHE_DIR = './data_cache'
    MODELS_DIR = './models/specialized'
    RESULTS_DIR = './training_results'

# 创建目录
Path(Config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(Config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# 数据下载函数
# ============================================================================

def download_from_hf(symbol, timeframe):
    """
    从 Hugging Face 下载数据
    文件命名规则: {SYMBOL_PREFIX}_{timeframe}.parquet
    """
    # 提取币种前缀（去掉USDT）
    symbol_prefix = symbol.replace('USDT', '')
    file_path = f"klines/{symbol}/{symbol_prefix}_{timeframe}.parquet"
    
    try:
        print(f"   下载 {symbol} {timeframe}...", end='', flush=True)
        
        local_path = hf_hub_download(
            repo_id=Config.HF_REPO_ID,
            filename=file_path,
            repo_type="dataset",
            cache_dir=Config.DATA_CACHE_DIR
        )
        
        df = pd.read_parquet(local_path)
        
        # 处理索引
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                except:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
        
        df.columns = df.columns.str.lower()
        print(f" 成功 ({len(df)} 根)")
        return df
    
    except Exception as e:
        print(f" 失败 ({str(e)[:30]}...)")
        return None

def add_technical_indicators(df):
    """
    添加所有技术指标
    """
    df = df.copy()
    
    try:
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
        
        # 动能
        df['roc'] = df['close'].pct_change(5)
        df['momentum'] = df['close'] - df['close'].shift(5)
        
        # 移动平均线
        df['ema9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        
        # K线形态
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_ratio'] = df['body_size'] / (df['high'] - df['low'] + 0.0001)
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['high'] - df['low'] + 0.0001)
        df['high_low_range'] = df['high'] - df['low']
        
        # ADX
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_indicator.adx()
        
        # 微觀结构特征
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
    except:
        return None

def create_labels(df):
    """
    生成交易标签
    """
    labels_list = []
    
    for i in range(len(df) - 6):
        current_row = df.iloc[i]
        close_price = current_row['close']
        bb_upper = current_row['bb_upper']
        bb_lower = current_row['bb_lower']
        
        # 下轨触及
        if close_price <= bb_lower * 1.005:
            future_prices = df.iloc[i:i+6]['high']
            max_price = future_prices.max()
            price_increase_pct = ((max_price - close_price) / close_price) * 100
            is_success = 1 if price_increase_pct > 0.5 else 0
            labels_list.append({'index': i, 'label': is_success, 'type': 'lower'})
        
        # 上轨触及
        if close_price >= bb_upper * 0.995:
            future_prices = df.iloc[i:i+6]['low']
            min_price = future_prices.min()
            price_decrease_pct = ((close_price - min_price) / close_price) * 100
            is_success = 1 if price_decrease_pct > 0.5 else 0
            labels_list.append({'index': i, 'label': is_success, 'type': 'upper'})
    
    return pd.DataFrame(labels_list)

def extract_features(df, labels):
    """
    提取特征
    """
    features_list = []
    
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
# 训练主程序
# ============================================================================

print(f"[INFO] 準备训练 {len(Config.SYMBOLS)} 个币种 x {len(Config.TIMEFRAMES)} 个时间框")
print(f"[INFO] 共 {len(Config.SYMBOLS) * len(Config.TIMEFRAMES)} 个模型\n")

training_results = {}
success_count = 0
fail_count = 0

for symbol_idx, symbol in enumerate(Config.SYMBOLS, 1):
    print(f"\n[{symbol_idx}/{len(Config.SYMBOLS)}] {symbol}")
    print("-" * 70)
    
    for timeframe in Config.TIMEFRAMES:
        # 下载数据
        df = download_from_hf(symbol, timeframe)
        if df is None or len(df) < 200:
            print(f"      跳过（数据不足）")
            fail_count += 1
            continue
        
        # 计算指标
        df = add_technical_indicators(df)
        if df is None:
            print(f"      跳过（指标计算失败）")
            fail_count += 1
            continue
        
        # 生成标签
        labels = create_labels(df)
        if len(labels) < 50:
            print(f"      跳过（样本太少 {len(labels)}）")
            fail_count += 1
            continue
        
        # 提取特征
        features_df = extract_features(df, labels)
        if len(features_df) < 50:
            print(f"      跳过（有效样本太少 {len(features_df)}）")
            fail_count += 1
            continue
        
        # 準备数据
        X = features_df.drop('label', axis=1)
        y = features_df['label']
        X = X.fillna(X.mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练模型
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        
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
        
        # 评估
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
            'symbol': symbol,
            'timeframe': timeframe,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'samples': len(features_df)
        }
        
        print(f"      训练完成 | AUC: {auc:.4f} | 召回率: {recall:.2%} | F1: {f1:.4f}")
        success_count += 1

# ============================================================================
# 结果汇总
# ============================================================================

print(f"\n\n{'='*70}")
print(f"训练完成")
print(f"{'='*70}\n")

print(f"成功: {success_count} | 失败: {fail_count}\n")

if training_results:
    results_df = pd.DataFrame(training_results).T
    results_df = results_df.sort_values('auc', ascending=False)
    
    print("前 10 个最佳模型:")
    print("\n{:20} | {:10} | {:10} | {:10}".format('MODEL ID', 'AUC', 'RECALL', 'F1'))
    print("-" * 70)
    for model_id, row in results_df.head(10).iterrows():
        print("{:20} | {:10.4f} | {:10.2%} | {:10.4f}".format(
            model_id, row['auc'], row['recall'], row['f1']
        ))
    
    # 保存结果
    results_df.to_csv(f'{Config.RESULTS_DIR}/training_results.csv')
    
    with open(f'{Config.RESULTS_DIR}/models_metadata.json', 'w') as f:
        json.dump({
            'total_models': len(training_results),
            'symbols': Config.SYMBOLS,
            'timeframes': Config.TIMEFRAMES,
            'results': training_results
        }, f, indent=2)
    
    print(f"\n[INFO] 已保存到 {Config.RESULTS_DIR}/")

print(f"[INFO] 模型已保存到 {Config.MODELS_DIR}/")
