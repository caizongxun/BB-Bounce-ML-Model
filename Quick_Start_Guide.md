# BB反彈ML模型 - 快速開始指南

## 最快方式（30分鐘）

### 步驟1：環境準備（5分鐘）

```bash
# 克隆倉庫
git clone https://github.com/caizongxun/BB-Bounce-ML-Model.git
cd BB-Bounce-ML-Model

# 創建虛擬環境
python -m venv venv

# 激活虛擬環境
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# 一鍵安裝所有依賴
pip install -r requirements.txt
```

### 步驟2：訓練模型（15分鐘）

```bash
python complete_training.py
```

預期輸出會顯示：
- ✅ 數據下載成功
- ✅ 指標計算完成  
- ✅ 標籤生成完成
- ✅ 特徵提取完成
- ✅ 模型訓練完成 (XGBoost AUC ≈ 0.68)

### 步驟3：部署API（另開終端）

```bash
# 激活虛擬環境（如果還沒激活）
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# 啟動 API 服務器
python deploy_api.py
```

### 步驟4：測試API（新終端）

```bash
# 檢查健康狀態
curl http://localhost:5000/health

# 應該返回
# {"status": "ok", "model_loaded": true}
```

✅ **完成！** 模型已訓練並可用

---

## 📊 後續步驟

1. 在 Pine Script 中集成 API URL (http://localhost:5000/predict_bounce)
2. 紙上交易驗證信號質量
3. 實盤小額測試
4. 定期重訓練（每月1-2次）

---

詳細步驟見 `Step_by_Step_Guide.md`
