@echo off
REM Windows 安裝脚本 - 安裝 flask-cors

echo .
echo ============================================
echo BB Bounce ML Model - 安裝 CORS 套件
echo ============================================
echo .

REM 檢查虫世一字檔案 .venv
if not exist ".venv" (
    echo 需要先易休易乙景臂境。
    echo 姓妷懂 venv 沒易囊來。詳後再試。
echo 或詵您手動死亦励何：
python -m venv .venv
    echo.
    pause
    exit /b 1
)

echo 啟動虫世一字檔...
call .venv\Scripts\activate.bat

echo.
echo 安裝 flask-cors 敖冶...
pip install flask-cors

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 安裝成功！
    echo.
    echo 下一步：你現在可以運行下傳組敗阪筱。
    echo.
    echo python deploy_api_optimized.py
    echo.
) else (
    echo.
    echo ❌ 安裝失敗。請檢查網路連接或嘗試手動安裝。
    echo.
)

pause
