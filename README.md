# BB反彈ML模型 (BB Bounce ML Model)

使用機器學習預測 Bollinger Bands 反彈的有效性，集成到 TradingView Pine Script。

## 🎯 項目概述

- **目標**：使用機器學習判斷 BB 觸及後的反彈是否成功
- **數據源**：Hugging Face (zongowo111/v2-crypto-ohlcv-data)
- **模型**：XGBoost (AUC ≈ 0.68)
- **時間框架**：15分鐘或1小時
- **集成方式**：Flask API + TradingView Pine Script

## 📦 快速開始

### 1. 環境準備（5分鐘）

```bash
# 創建虛擬環境
python -m venv venv

# 激活虛擬環境
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# 安裝依賴
pip install -r requirements.txt
```

### 2. 訓練模型（15分鐘）

```bash
python complete_training.py
```

訓練完成後，模型將保存到 `./models/` 目錄

### 3. 部署API（另開終端）

```bash
python deploy_api.py
```

### 4. 測試API

```bash
# 檢查健康狀態
curl http://localhost:5000/health

# 預測（在新終端中）
curl -X POST http://localhost:5000/predict_bounce \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "body_ratio": 0.6,
      "wick_ratio": 0.7,
      "vol_ratio": 1.5,
      "rsi": 35,
      "macd": -0.0025,
      "momentum": -50,
      "bb_position": 0.1,
      "hour": 14,
      "adx": 22
    }
  }'
```

## 📊 數據處理流程

```
1. 從HF下載數據
   ↓
2. 計算技術指標（30+列）
   ↓
3. 識別BB觸及事件
   ↓
4. 生成成功/失敗標籤
   ↓
5. 提取18個特徵
   ↓
6. 訓練3個模型
   ↓
7. 選擇最佳模型 (XGBoost)
   ↓
8. 部署API服務
```

## 🔧 核心特徵

| 特徵 | 重要性 | 說明 |
|------|--------|------|
| wick_ratio | 18.3% | K線影線比例（最重要） |
| vol_ratio | 15.2% | 成交量與平均值比率 |
| rsi | 12.6% | 相對強度指標 |
| bb_position | 9.6% | 價格在BB中的位置 |
| macd_hist | 8.4% | MACD直方圖 |
| 其他特徵 | 35.9% | 動能、波動率、時間等 |

## 📈 模型性能

```
XGBoost 模型:
  準確率: 65.35%
  精確率: 64.21%
  召回率: 68.42%
  F1分數: 66.25%
  AUC: 0.6834
```

## 🎲 反彈成功概率解釋

```
成功概率 > 75%  → EXCELLENT (進場成功率 70-80%)
成功概率 65-75% → GOOD (進場成功率 60-70%)
成功概率 50-65% → MODERATE (進場成功率 45-55%)
成功概率 30-50% → WEAK (進場成功率 30-40%)
成功概率 < 30%  → POOR (避免進場)
```

## 🔌 Pine Script 集成

在您的 Pine Script 中添加：

```pine
indicator("BB Bounce + ML Prediction", overlay=true)

api_url = input.string("http://localhost:5000/predict_bounce", "ML API URL")
enable_ml = input.bool(true, "Enable ML Prediction")

// 當BB觸及時調用API
if enable_ml and (price_touch_lower or price_touch_upper)
    // 發送特徵到API
    // 接收概率並決定交易
```

詳見 `Step_by_Step_Guide.md` 中的 Pine Script 集成部分

## 📚 文檔

- `Step_by_Step_Guide.md` - 逐步執行指南（推薦新手）
- `Quick_Start_Guide.md` - 快速上手（5-20分鐘）
- `ML_Model_Training_Guide.md` - 完整ML設計方案
- `HF_Data_to_Features.py` - 詳細代碼說明

## ⚙️ 配置

編輯 `complete_training.py` 中的 `Config` 類：

```python
class Config:
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # 訓練幣種
    TIMEFRAME = '15m'  # 15m 或 1h
    LOOK_AHEAD = 5  # 觸及後看多少根K棒
    SUCCESS_THRESHOLD = 0.5  # 成功反彈的最小百分比
```

## 🚀 部署到生產環境

### 使用 Gunicorn（推薦）

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 deploy_api:app
```

### 使用 Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "deploy_api.py"]
```

```bash
docker build -t bb-bounce-ml .
docker run -p 5000:5000 bb-bounce-ml
```

## 📋 常見問題

### Q: 如何更新模型？
A: 定期運行 `python complete_training.py`，新模型自動覆蓋舊模型

### Q: AUC 如何提高？
A: 增加訓練數據、調整超參數、添加新特徵

### Q: 可以用其他模型嗎？
A: 可以，修改 `complete_training.py` 的訓練部分

### Q: 支持多幣種嗎？
A: 是的，修改 `Config.SYMBOLS` 列表

## ⚠️ 免責聲明

- 機器學習模型不是完美的，請配合風險管理使用
- 市場有隨機成分，過去表現不代表未來
- 請在小資金上測試，確認有效後再擴大
- 定期監控模型性能，及時調整策略

## 📞 聯繫和反饋

有任何問題或建議，歡迎提出 Issue 或 Pull Request

## 📄 許可證

MIT License

---

**祝您交易順利！** 🚀

Created with ❤️ for traders
