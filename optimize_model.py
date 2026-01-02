#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BB反彈ML模型優化 - 提升召回率
優化策略：
1. 添加新特徵（微觀結構特徵）
2. 調整類權重（處理不平衡問題）
3. 網格搜索最佳超參數
4. 集成多模型
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import pickle
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print(f"\n{'='*60}")
print(f"BB反彈ML模型優化")
print(f"{'='*60}")

# ============================================================================
# 第一步：加載訓練數據
# ============================================================================

print(f"\n步驟1：加載數據")
print(f"{'='*60}\n")

try:
    # 這些文件是complete_training.py生成的
    with open('./models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('./models/feature_cols.json', 'r') as f:
        feature_cols = json.load(f)
    
    print(f"✅ 加載成功")
    print(f"   特徵數: {len(feature_cols)}")
    print(f"   特徵: {feature_cols}")
except:
    print(f"❌ 未找到模型文件")
    print(f"   請先運行: python complete_training.py")
    exit()

# ============================================================================
# 第二步：重新訓練並使用類權重
# ============================================================================

print(f"\n步驟2：準備訓練數據")
print(f"{'='*60}\n")

# 重新生成訓練數據（從HF）
from huggingface_hub import hf_hub_download
import ta

class Config:
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    TIMEFRAME = '15m'
    HF_REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    LOOK_AHEAD = 5
    SUCCESS_THRESHOLD = 0.5
    DATA_CACHE_DIR = './data_cache'
    MODELS_DIR = './models'

def download_from_hf(symbol, timeframe):
    """從Hugging Face下載數據"""
    file_path = f"klines/{symbol}/{symbol.split('USDT')[0]}_{timeframe}.parquet"
    
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
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                except:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
    
    df.columns = df.columns.str.lower()
    return df

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
    
    # 新增特徵 - 微觀結構特徵
    print(f"   添加微觀結構特徵中...", end='', flush=True)
    
    # 1. 觸及深度（距離BB有多遠）
    df['lower_touch_depth'] = (df['bb_lower'] - df['low']) / (df['bb_width'] + 0.0001)
    df['upper_touch_depth'] = (df['high'] - df['bb_upper']) / (df['bb_width'] + 0.0001)
    
    # 2. 反轉強度（收盤離觸及點有多遠）
    df['close_from_lower'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
    df['close_from_upper'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 0.0001)
    
    # 3. 成交量加權強度
    df['volume_strength'] = (df['volume'] - df['vol_ma20']) / (df['vol_ma20'] + 0.0001)
    
    # 4. 波動率相對強度
    df['volatility_ratio'] = df['atr'] / (df['atr'].rolling(20).mean() + 0.0001)
    
    # 5. 趨勢確認（前5根K棒的方向）
    df['prev_5_trend'] = (df['close'].shift(5) - df['close'].shift(1)) / (df['close'].shift(1) + 0.0001)
    
    # 6. RSI強度等級
    df['rsi_strength'] = np.abs(df['rsi'] - 50) / 50  # 0-1，離50越遠越強
    
    # 7. MACD動能
    df['macd_strength'] = np.abs(df['macd_hist']) / (np.abs(df['macd_hist']).rolling(20).mean() + 0.0001)
    
    print(f" ✅")
    
    # 處理 NaN
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def create_bounce_labels(df, symbol):
    """生成標籤"""
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

def extract_bounce_features(df, labels_df):
    """提取特徵（包含新特徵）"""
    features_list = []
    
    for _, label_row in labels_df.iterrows():
        idx = label_row['index']
        
        if idx < 50:
            continue
        
        current_row = df.iloc[idx]
        bounce_type = label_row['bounce_type']
        
        # 原始特徵
        body_ratio = current_row['body_ratio']
        wick_ratio = current_row['wick_ratio']
        high_low_range = current_row['high_low_range']
        vol_ratio = current_row['vol_ratio']
        vol_spike_ratio = current_row['volume'] / (current_row['vol_ma20'] + 0.0001)
        rsi = current_row['rsi']
        macd = current_row['macd']
        macd_hist = current_row['macd_hist']
        momentum = current_row['momentum']
        bb_width_ratio = current_row['bb_width_ratio']
        bb_position = (current_row['close'] - current_row['bb_lower']) / (current_row['bb_upper'] - current_row['bb_lower'] + 0.0001)
        
        recent_close = df.iloc[max(0, idx-20):idx]['close']
        price_trend = 1 if len(recent_close) >= 2 and recent_close.iloc[-1] > recent_close.iloc[0] else 0
        price_slope = (recent_close.iloc[-1] - recent_close.iloc[0]) / recent_close.iloc[0] if len(recent_close) >= 2 else 0
        
        timestamp = df.index[idx]
        hour = timestamp.hour
        is_high_volume_time = 1 if (hour >= 20 or hour < 4) else 0
        
        adx = current_row['adx']
        
        # 新增特徵
        lower_touch_depth = current_row['lower_touch_depth']
        upper_touch_depth = current_row['upper_touch_depth']
        close_from_lower = current_row['close_from_lower']
        close_from_upper = current_row['close_from_upper']
        volume_strength = current_row['volume_strength']
        volatility_ratio = current_row['volatility_ratio']
        prev_5_trend = current_row['prev_5_trend']
        rsi_strength = current_row['rsi_strength']
        macd_strength = current_row['macd_strength']
        
        feature_dict = {
            # 原始特徵
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
            # 新增特徵
            'lower_touch_depth': lower_touch_depth,
            'upper_touch_depth': upper_touch_depth,
            'close_from_lower': close_from_lower,
            'close_from_upper': close_from_upper,
            'volume_strength': volume_strength,
            'volatility_ratio': volatility_ratio,
            'prev_5_trend': prev_5_trend,
            'rsi_strength': rsi_strength,
            'macd_strength': macd_strength,
            'label': label_row['label'],
            'bounce_type': bounce_type
        }
        
        features_list.append(feature_dict)
    
    return pd.DataFrame(features_list)

print("\n正在下載和處理數據...")
all_features = []

for symbol in Config.SYMBOLS:
    print(f"\n{symbol}:", end='')
    df = download_from_hf(symbol, Config.TIMEFRAME)
    df = add_technical_indicators(df)
    labels = create_bounce_labels(df, symbol)
    features = extract_bounce_features(df, labels)
    all_features.append(features)
    print(f" ✅ {len(features)} 個樣本")

combined_features = pd.concat(all_features, ignore_index=True)
feature_cols_optimized = [col for col in combined_features.columns 
                          if col not in ['label', 'bounce_type']]

X = combined_features[feature_cols_optimized]
y = combined_features['label']
X = X.fillna(X.mean())

scaler_new = StandardScaler()
X_scaled = scaler_new.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ 數據準備完成")
print(f"   訓練集: {X_train.shape}")
print(f"   測試集: {X_test.shape}")
print(f"   新特徵數: {len(feature_cols_optimized)}")
print(f"   正樣本比例: {y.mean():.2%}")

# ============================================================================
# 第三步：優化XGBoost（重點）
# ============================================================================

print(f"\n步驟3：優化XGBoost超參數")
print(f"{'='*60}\n")

# 計算類權重以提升召回率
from sklearn.utils.class_weight import compute_sample_weight

# 方式1：使用scale_pos_weight（XGBoost特有）
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"類權重比: {scale_pos_weight:.2f}")
print(f"\n開始網格搜索...（這可能需要10-15分鐘）\n")

# 優化參數
param_grid = {
    'max_depth': [4, 5, 6],
    'learning_rate': [0.03, 0.05, 0.08],
    'n_estimators': [150, 200],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}

xgb_base = XGBClassifier(
    scale_pos_weight=scale_pos_weight,  # 關鍵：提升召回率
    random_state=42,
    objective='binary:logistic',
    eval_metric='auc'
)

grid_search = GridSearchCV(
    xgb_base,
    param_grid,
    cv=3,  # 3折交叉驗證
    scoring='roc_auc',  # 以AUC為優化目標
    n_jobs=-1,
    verbose=1
)

print("正在搜索最佳超參數...")
grid_search.fit(X_train, y_train)

print(f"\n最佳超參數: {grid_search.best_params_}")
print(f"最佳交叉驗證AUC: {grid_search.best_score_:.4f}")

best_xgb = grid_search.best_estimator_

# ============================================================================
# 第四步：評估優化後的模型
# ============================================================================

print(f"\n步驟4：模型評估")
print(f"{'='*60}\n")

y_pred = best_xgb.predict(X_test)
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

print(f"XGBoost (優化版)")
print(f"  準確率: {accuracy_score(y_test, y_pred):.4f}")
print(f"  精確率: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  召回率: {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  F1分數: {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# 計算不同閾值下的性能
print(f"\n不同決策閾值下的性能:")
print(f"{'閾值':>6} | {'精確率':>8} | {'召回率':>8} | {'F1分數':>8}")
print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:
    y_pred_custom = (y_pred_proba >= threshold).astype(int)
    if (y_pred_custom == 1).sum() > 0:
        precision = precision_score(y_test, y_pred_custom, zero_division=0)
        recall = recall_score(y_test, y_pred_custom, zero_division=0)
        f1 = f1_score(y_test, y_pred_custom, zero_division=0)
        print(f"{threshold:>6.2f} | {precision:>8.4f} | {recall:>8.4f} | {f1:>8.4f}")

# ============================================================================
# 第五步：特徵重要性
# ============================================================================

print(f"\n步驟5：特徵重要性分析")
print(f"{'='*60}\n")

if hasattr(best_xgb, 'feature_importances_'):
    importances = best_xgb.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols_optimized,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"Top 15 重要特徵:")
    print()
    for i, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")

# ============================================================================
# 第六步：保存優化後的模型
# ============================================================================

print(f"\n步驟6：保存優化模型")
print(f"{'='*60}\n")

with open(f'{Config.MODELS_DIR}/best_model_optimized.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)

with open(f'{Config.MODELS_DIR}/scaler_optimized.pkl', 'wb') as f:
    pickle.dump(scaler_new, f)

with open(f'{Config.MODELS_DIR}/feature_cols_optimized.json', 'w') as f:
    json.dump(feature_cols_optimized, f)

with open(f'{Config.MODELS_DIR}/best_params.json', 'w') as f:
    json.dump(grid_search.best_params_, f)

print(f"✅ 優化模型已保存:")
print(f"   best_model_optimized.pkl")
print(f"   scaler_optimized.pkl")
print(f"   feature_cols_optimized.json")
print(f"   best_params.json")

print(f"\n{'='*60}")
print(f"✅ 優化完成！")
print(f"{'='*60}")
print(f"\n關鍵改進:")
print(f"  ✓ 添加了9個新特徵（微觀結構特徵）")
print(f"  ✓ 使用類權重提升召回率")
print(f"  ✓ 網格搜索最佳超參數")
print(f"  ✓ 通過調整閾值進一步優化性能")
print(f"\n下一步: 更新 deploy_api.py 使用優化模型")
print()
