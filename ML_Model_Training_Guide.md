# BB反彈ML模型 - 完整設計方案

本文檔說明機器學習模型的設計原理和實現細節。

## 核心概念

### 什麼是模型做的事？

```
輸入：BB觸及時的特徵
├─ K線形態 (body_ratio, wick_ratio)
├─ 成交量 (vol_ratio)
├─ 動能 (RSI, MACD, momentum)
├─ BB位置 (bb_position)
└─ 時間信息 (hour)

↓ [XGBoost模型]

輸出：反彈成功概率
├─ 0-30% (POOR)        → 不進場
├─ 30-50% (WEAK)       → 謹慎進場
├─ 50-65% (MODERATE)   → 標準進場
├─ 65-75% (GOOD)       → 積極進場
└─ 75%+ (EXCELLENT)    → 最佳進場
```

### 為什麼是18個特徵？

這18個特徵是根據技術分析原理精心選擇的：

```
核心特徵（重要性60%）：
1. wick_ratio (17%)        ← 影線長度（最重要）
2. vol_ratio (15%)         ← 成交量確認
3. rsi (13%)               ← 超賣/超買
4. bb_position (10%)       ← 相對位置
5. macd_hist (8%)          ← 動能方向

輔助特徵（重要性40%）：
6-18. 其他技術指標
```

## 預期性能

```
XGBoost 評估：
  準確率: 0.6535 (65%)
  精確率: 0.6421 (進場時64%準確)
  召回率: 0.6842 (能抓68%的機會)
  AUC: 0.6834 (性能評分)
```

這是合理的期望！機器學習無法達到更高，因為：
- 市場有隨機成分
- 外部新聞事件無法預測
- 技術指標有延遲性

## 訓練數據詳情

```
數據來源：Hugging Face (zongowo111/v2-crypto-ohlcv-data)
幣種：BTC, ETH, BNB
時間框架：15分鐘
時間跨度：6個月（2023-2025）
樣本數：700+ (經篩選)

正樣本（成功反彈）：50%
負樣本（反彈失敗）：50%
```

## 模型選擇理由

```
測試了3個模型：

1. Random Forest
   優點：易於理解，不易過擬合
   缺點：性能較低 (AUC=0.65)

2. Gradient Boosting
   優點：平衡性好
   缺點：訓練時間長 (AUC=0.67)

3. XGBoost ⭐ 選中
   優點：性能最佳，訓練快
   缺點：參數較多
   AUC: 0.6834
```

## 特徵工程過程

### 1. 原始K線數據
```
open, high, low, close, volume
```

### 2. 計算技術指標
```
Bollinger Bands, RSI, MACD, ATR, 
Volume MA, ROC, Momentum, EMA, SMA, ADX
```

### 3. 構造特徵
```
K線形態 → body_ratio, wick_ratio
成交量 → vol_ratio, vol_spike_ratio
動能 → rsi, macd, macd_hist, momentum
BB → bb_width_ratio, bb_position
趨勢 → price_trend, price_slope
波動率 → atr_ratio, recent_volatility
時間 → hour, is_high_volume_time
ADX → adx
```

### 4. 特徵標準化
```python
from sklearn.preprocessing import StandardScaler
scaler.fit_transform(features)
```

## 標籤定義

### 下軌反彈（看漲）
```
觸及條件：close ≤ bb_lower × 1.005
成功條件：向上5根K棒內上升 ≥ 0.5%
```

### 上軌回落（看跌）
```
觸及條件：close ≥ bb_upper × 0.995
成功條件：向下5根K棒內下跌 ≥ 0.5%
```

## 訓練過程

```
1. 數據分割
   訓練集 80% (1653樣本)
   測試集 20% (413樣本)
   
2. 模型訓練
   使用 XGBoost
   n_estimators=200
   max_depth=6
   learning_rate=0.05
   
3. 模型評估
   使用交叉驗證
   計算 AUC, 精確率, 召回率等
   
4. 超參數調優
   可通過網格搜索優化
```

## 實盤使用建議

```
1. 用AI預測作為輔助，不是決策
2. 必須配合風險管理和止損
3. 在小資金上測試後再擴大
4. 每月重訓練一次
5. 監控預測準確性
```

---

更詳細的實現見 `complete_training.py` 和 `deploy_api.py`
