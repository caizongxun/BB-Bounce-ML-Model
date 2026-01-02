#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BB反彈ML模型 - 從Hugging Face完整訓練流程
一個文件包含所有步驟，直接運行即可
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from huggingface_hub import hf_hub_download
import ta
import warnings
import pickle
import os
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

class Config:
    # 數據配置
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # 修改成您要的幣種
    TIMEFRAME = '15m'  # 15m 或 1h
    
    # HF 配置
    HF_REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    
    # 訓練配置
    LOOK_AHEAD = 5  # 觸及後往前看幾根K棒
    SUCCESS_THRESHOLD = 0.5  # 成功反彈的最小百分比
    
    # 路徑配置
    DATA_CACHE_DIR = './data_cache'
    PROCESSED_DATA_DIR = './processed_data'
    LABELS_DIR = './labels'
    FEATURES_DIR = './features'
    MODELS_DIR = './models'
    
    # 模型配置
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    @staticmethod
    def create_dirs():
        """創建所需的目錄"""
        for dir_path in [Config.DATA_CACHE_DIR, Config.PROCESSED_DATA_DIR, 
                         Config.LABELS_DIR, Config.FEATURES_DIR, Config.MODELS_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        print("✅ 目錄創建完成")

# ============================================================================
# 第一步：數據讀取
# ============================================================================

def download_from_hf(symbol, timeframe):
    """從Hugging Face下載數據"""
    print(f"\n📥 正在下載 {symbol} {timeframe} 數據...")
    
    file_path = f"klines/{symbol}/{symbol.split('USDT')[0]}_{timeframe}.parquet"
    
    try:
        local_path = hf_hub_download(
            repo_id=Config.HF_REPO_ID,
            filename=file_path,
            repo_type="dataset",
            cache_dir=Config.DATA_CACHE_DIR
        )
        
        df = pd.read_parquet(local_path)
        
        # 處理索引 - 關鍵修復！支持多種時間戳格式
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                try:
                    # 先嘗試毫秒時間戳
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                except:
                    try:
                        # 再嘗試秒時間戳
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    except:
                        # 最後嘗試字符串格式
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 'datetime' in df.columns:
                try:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                except:
                    # 如果是字符串格式
                    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
                df.set_index('datetime', inplace=True)
        
        # 統一列名
        df.columns = df.columns.str.lower()
        
        # 確保有必需的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"   ⚠️  缺少必需列，嘗試列名映射...")
            # 嘗試常見的列名映射
            rename_map = {
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }
            df.rename(columns=rename_map, inplace=True)
        
        print(f"   ✅ 下載成功: {len(df)} 行")
        print(f"   時間範圍: {df.index[0]} ~ {df.index[-1]}")
        print(f"   列: {list(df.columns[:10])}...")
        
        return df
    
    except Exception as e:
        print(f"   ❌ 下載失敗: {str(e)}")
        import traceback
        print(f"   詳細錯誤: {traceback.format_exc()}")
        return None

def download_all_data():
    """批量下載所有幣種數據"""
    print(f"\n{'='*60}")
    print(f"步驟1：從HF下載數據")
    print(f"{'='*60}")
    
    all_data = {}
    
    for symbol in Config.SYMBOLS:
        df = download_from_hf(symbol, Config.TIMEFRAME)
        if df is not None:
            all_data[symbol] = df
    
    return all_data

# ============================================================================
# 第二步：計算技術指標
# ============================================================================

def add_technical_indicators(df):
    """添加所有技術指標"""
    print(f"   計算指標中...", end='', flush=True)
    
    df = df.copy()
    
    # Bollinger Bands - 修復API調用
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
    
    # 處理 NaN
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    print(f" ✅ {len(df.columns)} 列")
    
    return df

def process_all_data(all_data):
    """處理所有數據，添加指標"""
    print(f"\n{'='*60}")
    print(f"步驟2：計算技術指標")
    print(f"{'='*60}\n")
    
    data_with_indicators = {}
    
    for symbol, df in all_data.items():
        print(f"{symbol:10s}", end='')
        processed_df = add_technical_indicators(df)
        data_with_indicators[symbol] = processed_df
    
    return data_with_indicators

# ============================================================================
# 第三步：生成標籤
# ============================================================================

def create_bounce_labels(df, symbol):
    """為BB觸及事件生成標籤"""
    labels_list = []
    
    for i in range(len(df) - Config.LOOK_AHEAD - 1):
        current_row = df.iloc[i]
        close_price = current_row['close']
        bb_upper = current_row['bb_upper']
        bb_lower = current_row['bb_lower']
        
        # 下軌觸及
        if close_price <= bb_lower * 1.005:
            future_prices = df.iloc[i:i+Config.LOOK_AHEAD+1]['high']
            max_price = future_prices.max()
            price_increase_pct = ((max_price - close_price) / close_price) * 100
            is_success = 1 if price_increase_pct > Config.SUCCESS_THRESHOLD else 0
            
            labels_list.append({
                'symbol': symbol,
                'index': i,
                'timestamp': df.index[i],
                'bounce_type': 'lower',
                'touch_price': close_price,
                'label': is_success,
                'success_pct': price_increase_pct
            })
        
        # 上軌觸及
        if close_price >= bb_upper * 0.995:
            future_prices = df.iloc[i:i+Config.LOOK_AHEAD+1]['low']
            min_price = future_prices.min()
            price_decrease_pct = ((close_price - min_price) / close_price) * 100
            is_success = 1 if price_decrease_pct > Config.SUCCESS_THRESHOLD else 0
            
            labels_list.append({
                'symbol': symbol,
                'index': i,
                'timestamp': df.index[i],
                'bounce_type': 'upper',
                'touch_price': close_price,
                'label': is_success,
                'success_pct': price_decrease_pct
            })
    
    return pd.DataFrame(labels_list)

def generate_all_labels(data_with_indicators):
    """為所有幣種生成標籤"""
    print(f"\n{'='*60}")
    print(f"步驟3：生成標籤")
    print(f"{'='*60}\n")
    
    all_labels = {}
    
    for symbol, df in data_with_indicators.items():
        labels = create_bounce_labels(df, symbol)
        all_labels[symbol] = labels
        success_rate = labels['label'].mean() if len(labels) > 0 else 0
        print(f"{symbol:10s} {len(labels):5d} 個觸及事件，成功率 {success_rate:.2%}")
    
    return all_labels

# ============================================================================
# 第四步：特徵提取
# ============================================================================

def extract_bounce_features(df, labels_df):
    """提取反彈特徵"""
    features_list = []
    
    for _, label_row in labels_df.iterrows():
        idx = label_row['index']
        
        if idx < 50:
            continue
        
        current_row = df.iloc[idx]
        bounce_type = label_row['bounce_type']
        
        # K線形態
        body_ratio = current_row['body_ratio']
        wick_ratio = current_row['wick_ratio']
        high_low_range = current_row['high_low_range']
        
        # 成交量
        vol_ratio = current_row['vol_ratio']
        vol_spike_ratio = current_row['volume'] / (current_row['vol_ma20'] + 0.0001)
        
        # 動能
        rsi = current_row['rsi']
        macd = current_row['macd']
        macd_hist = current_row['macd_hist']
        momentum = current_row['momentum']
        
        # BB
        bb_width_ratio = current_row['bb_width_ratio']
        bb_position = (current_row['close'] - current_row['bb_lower']) / (current_row['bb_upper'] - current_row['bb_lower'] + 0.0001)
        
        # 趨勢
        recent_close = df.iloc[max(0, idx-20):idx]['close']
        price_trend = 1 if len(recent_close) >= 2 and recent_close.iloc[-1] > recent_close.iloc[0] else 0
        price_slope = (recent_close.iloc[-1] - recent_close.iloc[0]) / recent_close.iloc[0] if len(recent_close) >= 2 else 0
        
        # 時間
        timestamp = df.index[idx]
        hour = timestamp.hour
        is_high_volume_time = 1 if (hour >= 20 or hour < 4) else 0
        
        # ADX
        adx = current_row['adx']
        
        feature_dict = {
            'body_ratio': body_ratio,
            'wick_ratio': wick_ratio,
            'high_low_range': high_low_range,
            'vol_ratio': vol_ratio,
            'vol_spike_ratio': vol_spike_ratio,
            'rsi': rsi,
            'macd': macd,
            'macd_hist': macd_hist,
            'momentum': momentum,
            'bb_width_ratio': bb_width_ratio,
            'bb_position': bb_position,
            'price_trend': price_trend,
            'price_slope': price_slope,
            'hour': hour,
            'is_high_volume_time': is_high_volume_time,
            'adx': adx,
            'label': label_row['label'],
            'bounce_type': bounce_type
        }
        
        features_list.append(feature_dict)
    
    return pd.DataFrame(features_list)

def extract_all_features(data_with_indicators, all_labels):
    """為所有幣種提取特徵"""
    print(f"\n{'='*60}")
    print(f"步驟4：提取特徵")
    print(f"{'='*60}\n")
    
    all_features = {}
    
    for symbol in data_with_indicators.keys():
        df = data_with_indicators[symbol]
        labels = all_labels[symbol]
        features = extract_bounce_features(df, labels)
        all_features[symbol] = features
        
        success_rate = features['label'].mean() if len(features) > 0 else 0
        print(f"{symbol:10s} {len(features):5d} 個樣本，成功率 {success_rate:.2%}")
    
    return all_features

# ============================================================================
# 第五步：模型訓練
# ============================================================================

def prepare_training_data(all_features):
    """準備訓練數據"""
    print(f"\n{'='*60}")
    print(f"步驟5：準備訓練數據")
    print(f"{'='*60}\n")
    
    # 合併所有特徵
    combined_features = pd.concat(all_features.values(), ignore_index=True)
    
    # 特徵列
    feature_cols = [col for col in combined_features.columns 
                    if col not in ['label', 'bounce_type']]
    
    X = combined_features[feature_cols]
    y = combined_features['label']
    
    # 處理缺失值
    X = X.fillna(X.mean())
    
    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, stratify=y
    )
    
    print(f"訓練集大小: {X_train.shape[0]}")
    print(f"測試集大小: {X_test.shape[0]}")
    print(f"特徵數: {len(feature_cols)}")
    print(f"正樣本: {(y==1).sum()} ({y.mean():.2%})")
    print(f"負樣本: {(y==0).sum()} ({(1-y.mean()):.2%})")
    
    # 保存 scaler
    with open(f'{Config.MODELS_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # 保存特徵列名
    with open(f'{Config.MODELS_DIR}/feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_models(X_train, X_test, y_train, y_test, feature_cols):
    """訓練多個模型"""
    print(f"\n{'='*60}")
    print(f"步驟6：訓練模型")
    print(f"{'='*60}\n")
    
    models = {}
    results = {}
    
    # Random Forest
    print("訓練 Random Forest...", end='', flush=True)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    results['RandomForest'] = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, zero_division=0),
        'recall': recall_score(y_test, rf_pred, zero_division=0),
        'f1': f1_score(y_test, rf_pred, zero_division=0),
        'auc': roc_auc_score(y_test, rf_proba)
    }
    print(f" ✅ AUC={results['RandomForest']['auc']:.4f}")
    
    # XGBoost
    print("訓練 XGBoost...", end='', flush=True)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    results['XGBoost'] = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, zero_division=0),
        'recall': recall_score(y_test, xgb_pred, zero_division=0),
        'f1': f1_score(y_test, xgb_pred, zero_division=0),
        'auc': roc_auc_score(y_test, xgb_proba)
    }
    print(f" ✅ AUC={results['XGBoost']['auc']:.4f}")
    
    # Gradient Boosting
    print("訓練 Gradient Boosting...", end='', flush=True)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    models['GradientBoosting'] = gb
    
    gb_pred = gb.predict(X_test)
    gb_proba = gb.predict_proba(X_test)[:, 1]
    results['GradientBoosting'] = {
        'accuracy': accuracy_score(y_test, gb_pred),
        'precision': precision_score(y_test, gb_pred, zero_division=0),
        'recall': recall_score(y_test, gb_pred, zero_division=0),
        'f1': f1_score(y_test, gb_pred, zero_division=0),
        'auc': roc_auc_score(y_test, gb_proba)
    }
    print(f" ✅ AUC={results['GradientBoosting']['auc']:.4f}")
    
    # 選擇最佳模型
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model = models[best_model_name]
    
    print(f"\n最佳模型: {best_model_name} (AUC={results[best_model_name]['auc']:.4f})")
    
    # 保存模型
    with open(f'{Config.MODELS_DIR}/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    # 顯示詳細結果
    print(f"\n{'='*60}")
    print(f"模型評估結果")
    print(f"{'='*60}")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  準確率: {metrics['accuracy']:.4f}")
        print(f"  精確率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分數: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
    
    # 特徵重要性
    print(f"\n{'='*60}")
    print(f"特徵重要性 (Top 10)")
    print(f"{'='*60}")
    
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print()
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    return best_model, results

# ============================================================================
# 主程序
# ============================================================================

def main():
    print(f"\n{'='*60}")
    print(f"BB反彈ML模型訓練 - 完整流程")
    print(f"{'='*60}")
    
    # 創建目錄
    Config.create_dirs()
    
    # 步驟1：下載數據
    all_data = download_all_data()
    if not all_data:
        print("❌ 無法下載數據，退出")
        return
    
    # 步驟2：計算指標
    data_with_indicators = process_all_data(all_data)
    
    # 步驟3：生成標籤
    all_labels = generate_all_labels(data_with_indicators)
    
    # 步驟4：提取特徵
    all_features = extract_all_features(data_with_indicators, all_labels)
    
    # 步驟5：準備數據
    X_train, X_test, y_train, y_test, feature_cols = prepare_training_data(all_features)
    
    # 步驟6：訓練模型
    best_model, results = train_models(X_train, X_test, y_train, y_test, feature_cols)
    
    print(f"\n{'='*60}")
    print(f"✅ 訓練完成！")
    print(f"{'='*60}")
    print(f"\n模型已保存到: {Config.MODELS_DIR}/best_model.pkl")
    print(f"準備進行部署或測試...\n")

if __name__ == '__main__':
    main()
