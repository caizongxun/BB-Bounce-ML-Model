# 🚀 BB反彈ML模型 - 終極執行指南（逐步詳細版）

## 總耗時：約30-45分鐘

---

## 📍 第一部分：環境準備（5分鐘）

### 1.1 檢查Python版本

```bash
python --version
# 或
python3 --version

# 需要 Python 3.8+
# 如果版本太舊，請升級 Python
```

### 1.2 創建項目目錄

**Windows:**
```bash
md bb_ml_project
cd bb_ml_project
```

**Linux/Mac:**
```bash
mkdir bb_ml_project
cd bb_ml_project
```

### 1.3 創建虛擬環境（推薦）

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

成功後，命令行前面會顯示 `(venv)`

### 1.4 安裝所有依賴（2-3分鐘）

一次性複製以下命令：

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

✅ 驗證安裝成功：

```bash
python -c "import pandas, sklearn, xgboost, ta, huggingface_hub; print('✅ 所有依賴安裝成功')"
```

---

## 📍 第二部分：複製代碼文件（5分鐘）

### 2.1 獲取代碼

```bash
# 直接從 GitHub 克隆
git clone https://github.com/caizongxun/BB-Bounce-ML-Model.git
cd BB-Bounce-ML-Model

# 或下載 ZIP 後解壓
# https://github.com/caizongxun/BB-Bounce-ML-Model/archive/refs/heads/main.zip
```

### 2.2 驗證文件

```bash
# 檢查文件是否存在
ls -la  # Linux/Mac
dir     # Windows

# 應該看到：
# complete_training.py
# deploy_api.py
# requirements.txt
# README.md
# Step_by_Step_Guide.md
```

---

## 📍 第三部分：訓練模型（15-20分鐘）

### 3.1 開始訓練

確保虛擬環境已激活（命令行前有 `(venv)`），然後：

```bash
python complete_training.py
```

### 3.2 監控進度

訓練過程會輸出如下內容並等待完成

### 3.3 檢查生成的文件

訓練完成後，檢查是否生成了以下文件：

```bash
ls ./models/

# 應該包含：
# best_model.pkl (訓練好的模型)
# scaler.pkl (數據標準化工具)
# feature_cols.json (特徵列表)
```

---

## 📍 第四部分：部署API（10分鐘）

### 4.1 啟動API服務器

在同一個終端（確保虛擬環境仍激活）：

```bash
python deploy_api.py
```

### 4.2 檢查輸出

應該看到：

```
✅ 模型加載成功
✅ Scaler 加載成功
✅ 特徵列表加載成功

🚀 啟動 API 服務器...
   地址: http://localhost:5000
```

✅ **API 已啟動！** 保持這個終端窗口打開

### 4.3 測試API（在新終端中）

打開另一個終端窗口（不要關閉之前運行 API 的終端）：

```bash
# 檢查健康狀態
curl http://localhost:5000/health
```

---

## 📍 第五部分：集成到Pine Script（10分鐘）

### 5.1 獲取API地址

```
本地運行：http://localhost:5000/predict_bounce
```

### 5.2 在Pine Script中集成

見 README.md 中的 Pine Script 部分

---

## ✅ 驗證清單

```
環境：
□ Python 3.8+ 已安裝
□ 虛擬環境已創建
□ 所有依賴已安裝

文件：
□ complete_training.py 存在
□ deploy_api.py 存在

訓練：
□ 訓練腳本執行完成
□ ./models/best_model.pkl 存在
□ ./models/scaler.pkl 存在
□ ./models/feature_cols.json 存在

API：
□ API 服務器成功啟動
□ http://localhost:5000/health 返回 ok
```

---

**現在可以開始了！** 🚀

祝您ML模型訓練成功！
