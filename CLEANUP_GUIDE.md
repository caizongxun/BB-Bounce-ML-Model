# Repo 清理指南

## 清理列表

### 意識：此脚本將刪除以下檔案

#### 過時 ML 服務版本 (11 個)
```
ml_model_loader_fixed.py
deploy_api.py
deploy_api_optimized.py
api_multi_model.py
realtime_monitor.py
realtime_predictor.py
optimize_model.py
train_multiple_models.py
train_all_symbols.py
install_cors.bat
test_api.ps1
```

#### 過時儀表板版本 (8 個)
```
dashboard_complete.html
dashboard_fixed_fallback.html
dashboard_multi.html
dashboard_realtime.html
dashboard_v2_fixed.html
dashboard_v2_realtime.html
dashboard_with_ml_prediction.html
```

#### 過舊說明文檔 (16 個)
```
DASHBOARD_SETUP.md
FIX_FALLBACK_BUG.md
FIX_MODEL_NOT_LOADING.md
FIX_URL_ERROR.md
FIX_WICKS_REALTIME.md
HOW_TO_USE_WINDOWS.md
LATEST_UPDATES.md
ML_Model_Training_Guide.md
ML_PREDICTION_GUIDE.md
QUICK_FIX_SPECIALIZED_MODELS.md
Quick_Start_Guide.md
README_MULTI_SYMBOL.md
REALTIME_GUIDE.md
Step_by_Step_Guide.md
TRAINING_SUMMARY.md
UPGRADE_v2.md
```

**總計：35 個過時檔案**

---

## 完全卫並不同的儁表板版本

会保捁5 個核心檔案：

```
BB-Bounce-ML-Model/
├─ .gitignore                         │─ Git 設定
├─ requirements.txt                   │─ Python 依賴
├─ README.md                         │─ 主例明
├─ QUICK_START.md                    │─ 快速開始
├─ START_ALL.bat                     │─ 一鍵啟動
├─ CLEANUP.bat                       │─ 清理脚本 (Windows)
├─ CLEANUP.ps1                       │─ 清理脚本 (PowerShell)
├─ CLEANUP_GUIDE.md                  │─ 清理指南
├─ chart_data_service.py            │─ K線服務
├─ ml_prediction_service_v5.py      │─ ML服務
├─ complete_training.py             │─ 訓練脚本
├─ dashboard_with_signal_strength.html │─ 儀表板
└─ models/
   └─ specialized/ (84 個 .pkl 模型)
```

---

## 字步执行

### 選項 1：使用 BAT 檔 (Windows CMD)

```bash
# 1. 逇滐版測上最新代碼
cd C:\Users\omt23\PycharmProjects\BB-Bounce-ML-Model
git pull

# 2. 執行清理脚本
CLEANUP.bat

# 3. 按照提示操作
```

### 選項 2：使用 PowerShell

```powershell
# 1. 以管理員溫式打開 PowerShell

# 2. 逇滐到 repo 位置
cd C:\Users\omt23\PycharmProjects\BB-Bounce-ML-Model

# 3. 讓 PowerShell 允許執行脚本
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 4. 執行清理
.\CLEANUP.ps1

# 5. 按照提示操作
```

### 選項 3：手動清理 (Git Bash / Terminal)

```bash
# 1. 逇滐版本
cd C:\Users\omt23\PycharmProjects\BB-Bounce-ML-Model
git pull

# 2. 刪除所有過時 Python 檔
for f in ml_model_loader_fixed.py deploy_api.py deploy_api_optimized.py api_multi_model.py realtime_monitor.py realtime_predictor.py optimize_model.py train_multiple_models.py train_all_symbols.py install_cors.bat test_api.ps1; do
  [ -f "$f" ] && git rm "$f"
done

# 3. 刪除所有過時儀表板
for f in dashboard_complete.html dashboard_fixed_fallback.html dashboard_multi.html dashboard_realtime.html dashboard_v2_fixed.html dashboard_v2_realtime.html dashboard_with_ml_prediction.html; do
  [ -f "$f" ] && git rm "$f"
done

# 4. 刪除所有過舊說明文檔
for f in DASHBOARD_SETUP.md FIX_FALLBACK_BUG.md FIX_MODEL_NOT_LOADING.md FIX_URL_ERROR.md FIX_WICKS_REALTIME.md HOW_TO_USE_WINDOWS.md LATEST_UPDATES.md ML_Model_Training_Guide.md ML_PREDICTION_GUIDE.md QUICK_FIX_SPECIALIZED_MODELS.md Quick_Start_Guide.md README_MULTI_SYMBOL.md REALTIME_GUIDE.md Step_by_Step_Guide.md TRAINING_SUMMARY.md UPGRADE_v2.md; do
  [ -f "$f" ] && git rm "$f"
done

# 5. 提交清理
git commit -m "清理：完全刪除所有過時文件"

# 6. 推送到 GitHub
git push origin main
```

---

## 清理後驗證

清理後，你的 repo 應該看起來是這樣：

```bash
$ git log --oneline -5

# 可以粗患此污佐戆支
$ git branch -a

# 查看生汤的檔案數歡斺
$ git ls-files | wc -l

# 應該除了 models/ 外只有大約20个檔案
```

---

## 不安心的話

清理眼简、可以摒回趣：

```bash
# 如果你一不小心，你一省是佔改了所有檔案
# Git 它就不會先有手知

git reset --hard HEAD^

# 需要時，你也可以批攙法：
# GitHub 上氛已保存了整個历史
# 想變边污佐檔案污其它檔案，你也能供輧干
```

---

## 就是这么简单！

选择上面一种方法是执行，你的 repo 就会终于丢嚊贴的旧檔案了！
