# PowerShell 測試脚本，用于測試优化模型API
# 使用方法：在 PowerShell 中執行
# . .\test_api.ps1

$API_URL = "http://localhost:5000"

Write-Host "\n========================================" -ForegroundColor Green
Write-Host "BB反彈ML模型 - API測試" -ForegroundColor Green
Write-Host "========================================\n" -ForegroundColor Green

# 測試1：檢查健康狀段
Write-Host "步驟1: 檢查健康狀段" -ForegroundColor Cyan
Write-Host "-" * 40

try {
    $response = Invoke-RestMethod -Uri "$API_URL/health" -Method Get
    Write-Host "✅ API 是正常的" -ForegroundColor Green
    Write-Host "模型版本: $($response.model_version)" -ForegroundColor Yellow
    Write-Host "特徵數: $($response.features_count)" -ForegroundColor Yellow
    Write-Host "狀段: $($response.status)" -ForegroundColor Yellow
} catch {
    Write-Host "❌ API 未平常 - 請確保已啟動 API 服務器" -ForegroundColor Red
    Write-Host "需要先運行: python deploy_api_optimized.py" -ForegroundColor Red
    exit
}

Write-Host "\n"

# 測試2：預測一個正常信号（下軌觸及日）
Write-Host "步驟2: 預測一個平常上洋觸及信号" -ForegroundColor Cyan
Write-Host "-" * 40

$testData = @{
    features = @{
        # 三個原始特徵
        body_ratio = 0.5
        wick_ratio = 0.8
        high_low_range = 180
        vol_ratio = 1.8
        vol_spike_ratio = 2.1
        rsi = 28
        macd = -0.005
        macd_hist = -0.001
        momentum = -120
        bb_width_ratio = 1.2
        bb_position = 0.1
        price_trend = 0
        price_slope = -0.015
        hour = 14
        is_high_volume_time = 1
        adx = 25
        
        # 九個新特徵
        lower_touch_depth = -0.05
        upper_touch_depth = 0.02
        close_from_lower = 0.3
        close_from_upper = 0.7
        volume_strength = 1.5
        volatility_ratio = 1.3
        prev_5_trend = -0.02
        rsi_strength = 0.55
        macd_strength = 0.8
    }
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$API_URL/predict_bounce" `
        -Method Post `
        -ContentType "application/json" `
        -Body $testData
    
    Write-Host "✅ 預測成功" -ForegroundColor Green
    Write-Host "反彈成功概率: $($response.success_probability.ToString('P2'))" -ForegroundColor Yellow
    Write-Host "預測結果: $($response.predicted_class)" -ForegroundColor Yellow
    Write-Host "信心級別: $($response.confidence) ($($response.confidence_level))" -ForegroundColor Yellow
    Write-Host "建議動作: $($response.action)" -ForegroundColor Yellow
    Write-Host "模型版本: $($response.model_version)" -ForegroundColor Yellow
    Write-Host "決策閾值: $($response.threshold)" -ForegroundColor Yellow
} catch {
    Write-Host "❌ 預測失敗" -ForegroundColor Red
    Write-Host "$_" -ForegroundColor Red
}

Write-Host "\n"

# 測試3：預測一個弱信号（上軌觸及日，但被低估）
Write-Host "步驟3: 預測一個弱信号" -ForegroundColor Cyan
Write-Host "-" * 40

$weakSignal = @{
    features = @{
        body_ratio = 0.2
        wick_ratio = 0.3
        high_low_range = 100
        vol_ratio = 0.9
        vol_spike_ratio = 1.1
        rsi = 45
        macd = 0.0001
        macd_hist = 0.00001
        momentum = -20
        bb_width_ratio = 0.9
        bb_position = 0.15
        price_trend = 1
        price_slope = 0.005
        hour = 10
        is_high_volume_time = 0
        adx = 18
        
        lower_touch_depth = -0.02
        upper_touch_depth = 0.01
        close_from_lower = 0.2
        close_from_upper = 0.8
        volume_strength = 0.3
        volatility_ratio = 0.8
        prev_5_trend = 0.001
        rsi_strength = 0.1
        macd_strength = 0.2
    }
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$API_URL/predict_bounce" `
        -Method Post `
        -ContentType "application/json" `
        -Body $weakSignal
    
    Write-Host "✅ 預測成功" -ForegroundColor Green
    Write-Host "反彈成功概率: $($response.success_probability.ToString('P2'))" -ForegroundColor Yellow
    Write-Host "預測結果: $($response.predicted_class)" -ForegroundColor Yellow
    Write-Host "信心級別: $($response.confidence) ($($response.confidence_level))" -ForegroundColor Yellow
    Write-Host "建議動作: $($response.action)" -ForegroundColor Yellow
} catch {
    Write-Host "❌ 預測失敗" -ForegroundColor Red
    Write-Host "$_" -ForegroundColor Red
}

Write-Host "\n"

# 測試4：預測一個強信号（上軌觸及，大鳳軌）
Write-Host "步驟4: 預測一個強信号" -ForegroundColor Cyan
Write-Host "-" * 40

$strongSignal = @{
    features = @{
        body_ratio = 0.7
        wick_ratio = 0.9
        high_low_range = 250
        vol_ratio = 2.5
        vol_spike_ratio = 3.5
        rsi = 22
        macd = -0.01
        macd_hist = -0.008
        momentum = -250
        bb_width_ratio = 1.4
        bb_position = 0.05
        price_trend = 0
        price_slope = -0.025
        hour = 2
        is_high_volume_time = 1
        adx = 32
        
        lower_touch_depth = -0.08
        upper_touch_depth = 0.03
        close_from_lower = 0.15
        close_from_upper = 0.85
        volume_strength = 2.2
        volatility_ratio = 1.6
        prev_5_trend = -0.03
        rsi_strength = 0.78
        macd_strength = 0.95
    }
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$API_URL/predict_bounce" `
        -Method Post `
        -ContentType "application/json" `
        -Body $strongSignal
    
    Write-Host "✅ 預測成功" -ForegroundColor Green
    Write-Host "反彈成功概率: $($response.success_probability.ToString('P2'))" -ForegroundColor Yellow
    Write-Host "預測結果: $($response.predicted_class)" -ForegroundColor Yellow
    Write-Host "信心級別: $($response.confidence) ($($response.confidence_level))" -ForegroundColor Yellow
    Write-Host "建議動作: $($response.action)" -ForegroundColor Yellow
} catch {
    Write-Host "❌ 預測失敗" -ForegroundColor Red
    Write-Host "$_" -ForegroundColor Red
}

Write-Host "\n"
Write-Host "========================================" -ForegroundColor Green
Write-Host "✅ 測試完成！" -ForegroundColor Green
Write-Host "========================================\n" -ForegroundColor Green
