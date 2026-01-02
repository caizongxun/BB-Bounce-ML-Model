# ============================================================================
# BB Bounce ML Model - Cleanup Script (PowerShell)
# 一鍵清理所有過時文件
# ============================================================================

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "BB Bounce ML Model - 清理無用文件" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "此腳本將刪除所有過時的檔案" -ForegroundColor Yellow
Write-Host ""

$continue = Read-Host "按 Enter 繼續，或按 Ctrl+C 取消"

Write-Host ""
Write-Host "正在初始化 Git..." -ForegroundColor Green
git reset --hard
Write-Host "[OK] Git 已復原" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "刪除過時 ML 服務版本..." -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

$filesToDelete = @(
    "ml_model_loader_fixed.py",
    "deploy_api.py",
    "deploy_api_optimized.py",
    "api_multi_model.py",
    "realtime_monitor.py",
    "realtime_predictor.py",
    "optimize_model.py",
    "train_multiple_models.py",
    "train_all_symbols.py",
    "install_cors.bat",
    "test_api.ps1",
    "dashboard_complete.html",
    "dashboard_fixed_fallback.html",
    "dashboard_multi.html",
    "dashboard_realtime.html",
    "dashboard_v2_fixed.html",
    "dashboard_v2_realtime.html",
    "dashboard_with_ml_prediction.html",
    "DASHBOARD_SETUP.md",
    "FIX_FALLBACK_BUG.md",
    "FIX_MODEL_NOT_LOADING.md",
    "FIX_URL_ERROR.md",
    "FIX_WICKS_REALTIME.md",
    "HOW_TO_USE_WINDOWS.md",
    "LATEST_UPDATES.md",
    "ML_Model_Training_Guide.md",
    "ML_PREDICTION_GUIDE.md",
    "QUICK_FIX_SPECIALIZED_MODELS.md",
    "Quick_Start_Guide.md",
    "README_MULTI_SYMBOL.md",
    "REALTIME_GUIDE.md",
    "Step_by_Step_Guide.md",
    "TRAINING_SUMMARY.md",
    "UPGRADE_v2.md"
)

foreach ($file in $filesToDelete) {
    if (Test-Path $file) {
        Write-Host "刪除 $file" -ForegroundColor Yellow
        git rm $file --quiet
    }
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "提交清理更改" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

git commit -m "清理：完全刪除所有過時文件（v1-v4 服務、故舊儀表板、過舊說明文檔）"

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "推送修改至 GitHub" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

git push origin main

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "清理完成！" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "[OK] 所有過時文件已上傳推送" -ForegroundColor Green
Write-Host ""
Write-Host "保留住的核心文件：" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "  服務：" -ForegroundColor Cyan
Write-Host "    - chart_data_service.py" -ForegroundColor White
Write-Host "    - ml_prediction_service_v5.py" -ForegroundColor White
Write-Host "    - complete_training.py" -ForegroundColor White
Write-Host "" -ForegroundColor Cyan
Write-Host "  儀表板：" -ForegroundColor Cyan
Write-Host "    - dashboard_with_signal_strength.html" -ForegroundColor White
Write-Host "" -ForegroundColor Cyan
Write-Host "  配置：" -ForegroundColor Cyan
Write-Host "    - START_ALL.bat" -ForegroundColor White
Write-Host "    - requirements.txt" -ForegroundColor White
Write-Host "" -ForegroundColor Cyan
Write-Host "  說明：" -ForegroundColor Cyan
Write-Host "    - README.md" -ForegroundColor White
Write-Host "    - QUICK_START.md" -ForegroundColor White
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""

Read-Host "按 Enter 結束"
