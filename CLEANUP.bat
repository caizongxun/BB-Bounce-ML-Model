@echo off
REM ============================================================================
REM BB Bounce ML Model - Cleanup Script
REM 一鍵清理但需要 Git 清污交作
 REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo BB Bounce ML Model - 清理无用文件
echo ============================================================================
echo.
echo 此脚本將刪除所有過時的檔案

pause

echo.
echo 正在初始化 Git...
git reset --hard
echo [OK] Git 已守摆

echo.
echo ============================================================================
echo 刪除過時 ML 服務版本...
echo ============================================================================

if exist "ml_model_loader_fixed.py" (
    echo 刪除 ml_model_loader_fixed.py
    git rm ml_model_loader_fixed.py
)

if exist "deploy_api.py" (
    echo 刪除 deploy_api.py
    git rm deploy_api.py
)

if exist "deploy_api_optimized.py" (
    echo 刪除 deploy_api_optimized.py
    git rm deploy_api_optimized.py
)

if exist "api_multi_model.py" (
    echo 刪除 api_multi_model.py
    git rm api_multi_model.py
)

if exist "realtime_monitor.py" (
    echo 刪除 realtime_monitor.py
    git rm realtime_monitor.py
)

if exist "realtime_predictor.py" (
    echo 刪除 realtime_predictor.py
    git rm realtime_predictor.py
)

if exist "optimize_model.py" (
    echo 刪除 optimize_model.py
    git rm optimize_model.py
)

if exist "train_multiple_models.py" (
    echo 刪除 train_multiple_models.py
    git rm train_multiple_models.py
)

if exist "train_all_symbols.py" (
    echo 刪除 train_all_symbols.py
    git rm train_all_symbols.py
)

echo.
echo ============================================================================
echo 刪除過時儀表板版本...
echo ============================================================================

if exist "dashboard_complete.html" (
    echo 刪除 dashboard_complete.html
    git rm dashboard_complete.html
)

if exist "dashboard_fixed_fallback.html" (
    echo 刪除 dashboard_fixed_fallback.html
    git rm dashboard_fixed_fallback.html
)

if exist "dashboard_multi.html" (
    echo 刪除 dashboard_multi.html
    git rm dashboard_multi.html
)

if exist "dashboard_realtime.html" (
    echo 刪除 dashboard_realtime.html
    git rm dashboard_realtime.html
)

if exist "dashboard_v2_fixed.html" (
    echo 刪除 dashboard_v2_fixed.html
    git rm dashboard_v2_fixed.html
)

if exist "dashboard_v2_realtime.html" (
    echo 刪除 dashboard_v2_realtime.html
    git rm dashboard_v2_realtime.html
)

if exist "dashboard_with_ml_prediction.html" (
    echo 刪除 dashboard_with_ml_prediction.html
    git rm dashboard_with_ml_prediction.html
)

echo.
echo ============================================================================
echo 刪除過時說明文檔...
echo ============================================================================

if exist "DASHBOARD_SETUP.md" (
    echo 刪除 DASHBOARD_SETUP.md
    git rm DASHBOARD_SETUP.md
)

if exist "FIX_FALLBACK_BUG.md" (
    echo 刪除 FIX_FALLBACK_BUG.md
    git rm FIX_FALLBACK_BUG.md
)

if exist "FIX_MODEL_NOT_LOADING.md" (
    echo 刪除 FIX_MODEL_NOT_LOADING.md
    git rm FIX_MODEL_NOT_LOADING.md
)

if exist "FIX_URL_ERROR.md" (
    echo 刪除 FIX_URL_ERROR.md
    git rm FIX_URL_ERROR.md
)

if exist "FIX_WICKS_REALTIME.md" (
    echo 刪除 FIX_WICKS_REALTIME.md
    git rm FIX_WICKS_REALTIME.md
)

if exist "HOW_TO_USE_WINDOWS.md" (
    echo 刪除 HOW_TO_USE_WINDOWS.md
    git rm HOW_TO_USE_WINDOWS.md
)

if exist "LATEST_UPDATES.md" (
    echo 刪除 LATEST_UPDATES.md
    git rm LATEST_UPDATES.md
)

if exist "ML_Model_Training_Guide.md" (
    echo 刪除 ML_Model_Training_Guide.md
    git rm ML_Model_Training_Guide.md
)

if exist "ML_PREDICTION_GUIDE.md" (
    echo 刪除 ML_PREDICTION_GUIDE.md
    git rm ML_PREDICTION_GUIDE.md
)

if exist "QUICK_FIX_SPECIALIZED_MODELS.md" (
    echo 刪除 QUICK_FIX_SPECIALIZED_MODELS.md
    git rm QUICK_FIX_SPECIALIZED_MODELS.md
)

if exist "Quick_Start_Guide.md" (
    echo 刪除 Quick_Start_Guide.md
    git rm Quick_Start_Guide.md
)

if exist "README_MULTI_SYMBOL.md" (
    echo 刪除 README_MULTI_SYMBOL.md
    git rm README_MULTI_SYMBOL.md
)

if exist "REALTIME_GUIDE.md" (
    echo 刪除 REALTIME_GUIDE.md
    git rm REALTIME_GUIDE.md
)

if exist "Step_by_Step_Guide.md" (
    echo 刪除 Step_by_Step_Guide.md
    git rm Step_by_Step_Guide.md
)

if exist "TRAINING_SUMMARY.md" (
    echo 刪除 TRAINING_SUMMARY.md
    git rm TRAINING_SUMMARY.md
)

if exist "UPGRADE_v2.md" (
    echo 刪除 UPGRADE_v2.md
    git rm UPGRADE_v2.md
)

echo.
echo ============================================================================
echo 刪除打包和測試檔案...
echo ============================================================================

if exist "install_cors.bat" (
    echo 刪除 install_cors.bat
    git rm install_cors.bat
)

if exist "test_api.ps1" (
    echo 刪除 test_api.ps1
    git rm test_api.ps1
)

echo.
echo ============================================================================
echo 提交清理更改

echo ============================================================================
echo.
git commit -m "清理：完全删除所有過時文件（v1-v4 服勑、故旧儀表板、過旧說明文檔）"

echo.
echo ============================================================================
echo 推送修改至 GitHub
echo ============================================================================
echo.
git push origin main

echo.
echo ============================================================================
echo 清理完成！
echo ============================================================================
echo.
echo [OK] 所有過時文件已上枳推送

echo.
echo 正丸住的核心檔案：

echo   服勑：
    echo     - chart_data_service.py
    echo     - ml_prediction_service_v5.py
    echo     - complete_training.py

echo.
echo   儀表板：
    echo     - dashboard_with_signal_strength.html

echo.
echo   配置：
    echo     - START_ALL.bat
    echo     - requirements.txt

echo.
echo   說明：
    echo     - README.md
    echo     - QUICK_START.md

echo.
echo ============================================================================
echo.
pause
