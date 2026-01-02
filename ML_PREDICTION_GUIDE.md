# ML 預測功能完整指南

## 架構概述

現在系統包含三個主要服務：

```
┌─────────────────────────────────────────────────┐
│          Web 瀏覽器 - TradingView 圖表           │
│  (dashboard_with_ml_prediction.html)            │
└──────┬──────────────────────────────┬───────────┘
       │                              │
       │ HTTP API 調用                │ HTTP API 調用
       ↓                              ↓
┌──────────────────────┐      ┌──────────────────────┐
│  Chart Data Service  │      │ ML Prediction Service │
│  (Port 5001)         │      │ (Port 5002)          │
├──────────────────────┤      ├──────────────────────┤
│ - 推獲 K 線數據      │      │ - 加載 ML 模型       │
│ - 計算 BB 指標      │      │ - 生成預測信號      │
│ - 計算 RSI          │      │ - 計算信心度        │
└──────────────────────┘      └──────────────────────┘
```

---

## 快速開始（3 步驟）

### 步驟 1：啟動後端服務（Terminal 1）

```bash
# 啟動圖表數據服務
python chart_data_service.py

# 應該看到：
# [INFO] Chart Data Service Started
# [INFO] Address: http://localhost:5001
```

### 步驟 2：啟動 ML 預測服務（Terminal 2）

```bash
# 啟動 ML 預測服務
python ml_prediction_service.py

# 應該看到：
# [INFO] ML Prediction Service Started
# [INFO] Address: http://localhost:5002
# [INFO] Models loaded: X
# [INFO] Using fallback: False/True
```

### 步驟 3：打開儀表板

```
# 使用整合 ML 預測的版本
file:///C:.../dashboard_with_ml_prediction.html
```

---

## 使用 ML 預測功能

### 基本使用流程

```
1. 打開儀表板
   ↓
2. 選擇 Symbol 和 Timeframe
   ↓
3. 圖表加載並實時更新
   ↓
4. 點擊「Run Prediction」按鈕
   ↓
5. ML 模型生成預測信號（BUY / HOLD / SELL）
   ↓
6. 顯示信心度百分比
```

### 預測信號含義

| 信號 | 含義 | 顏色 | 建議 |
|------|------|------|------|
| **BUY** | 強烈買入信號 | 綠色 | 考慮買入 |
| **HOLD** | 保持持倉 | 橙色 | 觀望 |
| **SELL** | 強烈賣出信號 | 紅色 | 考慮賣出 |

### 信心度（Confidence）

- **80-100%**：極強信號，值得信賴
- **60-80%**：強信號，可以參考
- **40-60%**：中等信號，需要確認其他指標
- **<40%**：弱信號，不建議單獨依賴

---

## ML 模型詳解

### 模型加載過程

```python
# ml_prediction_service.py 的模型加載邏輯：

1. 掃描 models/ 目錄
   ├─ 查找 *.pkl 文件
   ├─ 查找 *.joblib 文件
   └─ 加載 models_metadata.json

2. 如果找到模型
   ├─ 使用已訓練的 ML 模型
   └─ 類型：ML (真實模型)

3. 如果沒找到模型
   ├─ 使用啟發式規則
   └─ 類型：FALLBACK (後備預測)
```

### 特徵工程

預測使用的特徵：

```python
features = [
    close,           # 當前收盤價
    bb_upper,        # BB 上軌
    bb_lower,        # BB 下軌
    bb_middle,       # BB 中線
    rsi,             # RSI 指標
    volatility       # 波動率
]
```

### 啟發式預測規則（后備方案）

當沒有 ML 模型時，系統使用簡單規則：

```python
# BB 位置分析
bb_position = (close - bb_lower) / (bb_upper - bb_lower)

# 評分機制
if bb_position > 0.7:     score += 2  # 接近上軌
elif bb_position < 0.3:   score -= 2  # 接近下軌

# RSI 分析
if 50 < rsi < 70:         score += 1  # 溫和上升
elif rsi >= 70:           score += 2  # 強烈上升
elif 30 < rsi < 50:       score -= 1  # 溫和下降
elif rsi <= 30:           score -= 2  # 強烈下降

# 轉換為信號
if score >= 2:   signal = BUY
elif score <= -2: signal = SELL
else:            signal = HOLD
```

---

## API 端點說明

### 1. 單次預測

**端點**：`POST /api/predict`

**請求**：
```json
{
  "close": 42500.50,
  "bb_upper": 43000.00,
  "bb_lower": 42000.00,
  "bb_middle": 42500.00,
  "rsi": 55.2,
  "volatility": 2.38
}
```

**回應**：
```json
{
  "signal": "BUY",
  "prediction": 1,
  "confidence": 0.82,
  "model_type": "ML",
  "model_name": "xgboost_v1.pkl",
  "timestamp": "2026-01-02T12:15:30"
}
```

### 2. 批量預測

**端點**：`POST /api/predict/batch`

**用途**：對多根 K 線進行預測

### 3. 服務狀態

**端點**：`GET /api/predict/status`

**回應**：
```json
{
  "status": "ok",
  "service": "ML Prediction Service",
  "models_loaded": 2,
  "model_names": ["xgboost_v1.pkl", "lightgbm_v1.pkl"],
  "using_fallback": false
}
```

---

## 添加自己的 ML 模型

### 步驟 1：準備模型文件

```bash
# 確保模型文件在 models/ 目錄中
mkdir models
# 將訓練好的模型（.pkl 或 .joblib）放入此目錄
cp my_model.pkl models/
```

### 步驟 2：更新元數據

```json
# models_metadata.json
{
  "my_model.pkl": {
    "type": "XGBoost",
    "accuracy": 0.85,
    "features": 6,
    "date_trained": "2026-01-01"
  }
}
```

### 步驟 3：重啟服務

```bash
# Ctrl+C 停止服務
# 重新啟動
python ml_prediction_service.py

# 應該顯示模型已加載
# [OK] Loaded model: my_model.pkl
```

---

## 故障排除

### 問題 1：找不到模型

**現象**：
```
[WARNING] No model files found. Using fallback predictor.
```

**解決**：
1. 檢查 models/ 目錄是否存在
2. 確認 .pkl 文件在目錄中
3. 檢查文件名是否正確

### 問題 2：模型加載失敗

**現象**：
```
[ERROR] Failed to load model.pkl: ...
```

**解決**：
1. 確認模型文件未損壞
2. 檢查依賴庫（joblib、sklearn、xgboost 等）是否安裝
3. 嘗試重新訓練模型

### 問題 3：預測端點不可用

**現象**：
```
Error: Cannot connect to localhost:5002
```

**解決**：
1. 確認 ml_prediction_service.py 正在運行
2. 檢查端口 5002 是否被佔用
3. 查看終端錯誤消息

---

## 模型性能監控

### 查看模型統計

```bash
# 調用狀態端點
curl http://localhost:5002/api/predict/status

# 回應示例：
{
  "status": "ok",
  "models_loaded": 2,
  "model_names": ["xgboost_v1.pkl", "ensemble_v2.pkl"],
  "using_fallback": false
}
```

### 實時監控預測準確度

1. 記錄預測結果
2. 等待 K 線完成
3. 比較預測 vs 實際
4. 計算準確度

---

## 高級用法

### 自動化預測（每秒）

```javascript
// 在儀表板中添加自動預測
setInterval(() => {
    runMLPrediction();
}, 1000);
```

### 組合多個模型

```python
# ml_prediction_service.py
# 修改 predict() 方法以支持投票機制
def predict_ensemble(features):
    predictions = []
    for model in self.models.values():
        pred = model.predict(X)[0]
        predictions.append(pred)
    
    # 多數投票
    final_signal = max(set(predictions), key=predictions.count)
    return final_signal
```

### 添加警報系統

```javascript
// 當信號強度超過閾值時發出警報
if (result.confidence > 0.85) {
    playAlert();  // 發出聲音提醒
    sendNotification(result.signal);  // 發送通知
}
```

---

## 文件結構

```
BB-Bounce-ML-Model/
├── models/                          # ML 模型目錄
│   ├── xgboost_v1.pkl
│   ├── lightgbm_v1.pkl
│   └── ...
├── models_metadata.json             # 模型元數據
├── chart_data_service.py            # 圖表數據服務
├── ml_prediction_service.py         # ML 預測服務 ✨ 新增
├── dashboard_with_ml_prediction.html # ML 集成儀表板 ✨ 新增
└── ...
```

---

## 總結

### 你現在有：

✅ **完整的 K 線圖表**（TradingView）
✅ **實時數據更新**（無 Loading）
✅ **Bollinger Bands 分析**
✅ **機器學習預測**（BUY/HOLD/SELL）
✅ **信心度指標**
✅ **多模型支持**
✅ **回退預測機制**

### 下一步可以做：

1. **訓練更好的模型**
   - 使用更多特徵
   - 試驗不同算法
   - 優化超參數

2. **添加更多指標**
   - MACD
   - Stochastic
   - VWAP

3. **實現交易系統**
   - 自動下單
   - 風險管理
   - 回測框架

4. **實時通知**
   - Email 警報
   - 手機推送
   - Telegram 通知

---

**祝交易順利！🚀**

任何問題都可以查看服務日誌或運行 `/api/predict/status` 端點診斷。
