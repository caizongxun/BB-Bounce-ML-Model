@echo off
REM ============================================================================
REM BB Bounce Pro - 一鍵啟動所有服務
REM ============================================================================
REM 此批次檔會啟動所有必要的服務

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo BB Bounce Pro - 一鍵啟動系統
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
echo   2. ML Prediction Service (Port 5002) - ML 模型預測
echo.
echo 然後手動打開瀏覽器訪問儀表板
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
echo [Step 2] 啟動 ML 預測服務...
echo.
start /d "%cd%" "ML Prediction Service" cmd /k "python ml_prediction_service_v3.py"
echo [OK] ML Prediction Service 已啟動 (Port 5002)

REM 等待第二個服務啟動
timeout /t 3 /nobreak >nul 2>&1

echo.
echo ============================================================================
echo 所有服務已啟動！
echo ============================================================================
echo.
echo [Next Step] 打開瀏覽器並訪問以下地址：
echo.
echo 儀表板：
echo   file:///C:/Users/omt23/PycharmProjects/BB-Bounce-ML-Model/dashboard_with_ml_prediction.html
echo.
echo 或者使用相對路徑：
echo   file:///./dashboard_with_ml_prediction.html
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
echo ============================================================================
echo.
pause
goto :eof

:install_dependencies
echo.
echo [INFO] 正在安裝必要的 Python 依賴...
pip install --upgrade flask flask-cors joblib scikit-learn numpy pandas
echo.
echo [OK] 依賴安裝完成
echo.
goto :eof
