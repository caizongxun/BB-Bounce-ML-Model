@echo off
REM ============================================================================
REM BB Bounce Pro - 一鍵啟動所有服務
REM 版本：v4 (最新修復版本)
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo BB Bounce Pro - 一鍵啟動系統
echo 版本: v4 (已修復所有 Bug)
echo ============================================================================
echo.

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 找不到 Python
    echo 請確保 Python 已安裝並添加到 PATH
    echo 下載：https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python 已安裝
echo.

REM 檢查依賴
echo [INFO] 檢查依賴...
pip list | find "flask" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] 正在安裝依賴...
    call :install_dependencies
)

echo.
echo ============================================================================
echo 準備啟動服務...
echo ============================================================================
echo.
echo 這個批次檔會開啟 2 個 Terminal：
echo   1. Chart Data Service (Port 5001) - K 線和 BB 軌道數據
echo   2. ML Prediction Service v4 (Port 5002) - ML 模型預測 (已修復)
echo.
echo 然後手勘打開瀏覽器訪問儀表板
echo.
pause

echo.
echo [Step 1] 啟動圖表數據服務...
echo.
start /d "%cd%" "Chart Data Service" cmd /k "python chart_data_service.py"
echo [OK] Chart Data Service 已啟動 (Port 5001)

REM 等待第一個服務啟動
timeout /t 3 /nobreak >nul 2>&1

echo.
echo [Step 2] 啟動 ML 預測服務 v4...
echo.
start /d "%cd%" "ML Prediction Service v4" cmd /k "python ml_prediction_service_v4.py"
echo [OK] ML Prediction Service v4 已啟動 (Port 5002)

REM 等待第二個服務啟動
timeout /t 3 /nobreak >nul 2>&1

echo.
echo ============================================================================
echo 所有服務已啟動！
echo ============================================================================
echo.
echo [Next Step] 打開瀏覽器並訪問以下地址：
echo.
echo 儀表板 (已修復版本) ：
echo   file:///C:/Users/omt23/PycharmProjects/BB-Bounce-ML-Model/dashboard_fixed_fallback.html
echo.
echo 或使用相對路徑：
echo   file:///./dashboard_fixed_fallback.html
echo.
echo 確保修改路徑為你的實際位置
echo.
echo ============================================================================
echo.
echo [INFO] 兩個 Terminal 窗口應該已打開
echo [INFO] 檢查它們是否正確運行（看是否有 [OK] 或 [ERROR] 信息）
echo.
echo [Tip] 如果需要停止：
echo   1. 在對應的 Terminal 中按 Ctrl+C
echo   2. 或直接關閉 Terminal 窗口
echo.
echo [重要] v4 版本已修復：
echo   - 修復了檔案名稱匹配問題
echo   - 應該看到 "Model: model_BTCUSDT_1h" (不是 fallback)
echo   - Confidence 應該在 70-90% 之間
echo.
echo ============================================================================
echo.
pause
goto :eof

:install_dependencies
echo.
echo [INFO] 正在安裝必要的 Python 依賴...
echo.
pip install --upgrade flask flask-cors joblib scikit-learn numpy pandas xgboost lightgbm
echo.
echo [OK] 依賴安裝完成
echo.
goto :eof
