@echo off
REM ============================================================================
REM BB Bounce ML Model - Final Cleanup
REM 删除所有.md（仅保留README.md）
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo BB Bounce ML Model - 最简洗清理
echo ============================================================================
echo.
echo 此脚本將删除所有.md檔案（除了README.md）
echo .

pause

echo.
echo 正在取得最新版本...
git pull

echo.
echo ============================================================================
echo 删除所有過旧的.md檔案
echo ============================================================================
echo.

if exist "QUICK_START.md" (
    echo 删除 QUICK_START.md
    git rm QUICK_START.md
)

if exist "cleanup_plan.md" (
    echo 删除 cleanup_plan.md
    git rm cleanup_plan.md
)

if exist "cleanup_status.md" (
    echo 删除 cleanup_status.md
    git rm cleanup_status.md
)

if exist "CLEANUP_GUIDE.md" (
    echo 删除 CLEANUP_GUIDE.md
    git rm CLEANUP_GUIDE.md
)

if exist "DASHBOARD_SETUP.md" (
    echo 删除 DASHBOARD_SETUP.md
    git rm DASHBOARD_SETUP.md
)

if exist "FIX_FALLBACK_BUG.md" (
    echo 删除 FIX_FALLBACK_BUG.md
    git rm FIX_FALLBACK_BUG.md
)

if exist "FIX_MODEL_NOT_LOADING.md" (
    echo 删除 FIX_MODEL_NOT_LOADING.md
    git rm FIX_MODEL_NOT_LOADING.md
)

if exist "FIX_URL_ERROR.md" (
    echo 删除 FIX_URL_ERROR.md
    git rm FIX_URL_ERROR.md
)

if exist "FIX_WICKS_REALTIME.md" (
    echo 删除 FIX_WICKS_REALTIME.md
    git rm FIX_WICKS_REALTIME.md
)

if exist "HOW_TO_USE_WINDOWS.md" (
    echo 删除 HOW_TO_USE_WINDOWS.md
    git rm HOW_TO_USE_WINDOWS.md
)

if exist "LATEST_UPDATES.md" (
    echo 删除 LATEST_UPDATES.md
    git rm LATEST_UPDATES.md
)

if exist "ML_Model_Training_Guide.md" (
    echo 删除 ML_Model_Training_Guide.md
    git rm ML_Model_Training_Guide.md
)

if exist "ML_PREDICTION_GUIDE.md" (
    echo 删除 ML_PREDICTION_GUIDE.md
    git rm ML_PREDICTION_GUIDE.md
)

if exist "QUICK_FIX_SPECIALIZED_MODELS.md" (
    echo 删除 QUICK_FIX_SPECIALIZED_MODELS.md
    git rm QUICK_FIX_SPECIALIZED_MODELS.md
)

if exist "Quick_Start_Guide.md" (
    echo 删除 Quick_Start_Guide.md
    git rm Quick_Start_Guide.md
)

if exist "README_MULTI_SYMBOL.md" (
    echo 删除 README_MULTI_SYMBOL.md
    git rm README_MULTI_SYMBOL.md
)

if exist "REALTIME_GUIDE.md" (
    echo 删除 REALTIME_GUIDE.md
    git rm REALTIME_GUIDE.md
)

if exist "Step_by_Step_Guide.md" (
    echo 删除 Step_by_Step_Guide.md
    git rm Step_by_Step_Guide.md
)

if exist "TRAINING_SUMMARY.md" (
    echo 删除 TRAINING_SUMMARY.md
    git rm TRAINING_SUMMARY.md
)

if exist "UPGRADE_v2.md" (
    echo 删除 UPGRADE_v2.md
    git rm UPGRADE_v2.md
)

echo.
echo ============================================================================
echo 提交清理
echo ============================================================================
echo.

git commit -m "清理：删除所有.md（仅保留README.md）"

echo.
echo ============================================================================
echo 推送到GitHub
echo ============================================================================
echo.

git push origin main

echo.
echo ============================================================================
echo 清理完成！
echo ============================================================================
echo.
echo [成功] 所有.md檔案已刪除，仅保留README.md

echo.
echo 最終保留的核心檔案：

echo.
echo   配置：
    echo     - .gitignore
    echo     - requirements.txt
    echo     - README.md

echo.
echo   脚本：
    echo     - START_ALL.bat
    echo     - CLEANUP.bat
    echo     - CLEANUP.ps1
    echo     - FINAL_CLEANUP.bat

echo.
echo   服務：
    echo     - chart_data_service.py
    echo     - ml_prediction_service_v5.py
    echo     - complete_training.py

echo.
echo   儀表板：
    echo     - dashboard_with_signal_strength.html

echo.
echo ============================================================================
echo.
pause
