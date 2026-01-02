#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BB反彈ML模型 - Flask API 部署 (優化版)
使用於 TradingView Pine Script 調用
優化: 使用優化模型和動態閾值
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 初始化
# ============================================================================

app = Flask(__name__)

# 啟用 CORS - 允許所有來源的跨域請求
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

MODEL_DIR = './models'
USE_OPTIMIZED = True  # 是否使用優化模型

# 嘗試加載優化模型
if USE_OPTIMIZED:
    try:
        with open(f'{MODEL_DIR}/best_model_optimized.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{MODEL_DIR}/scaler_optimized.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'{MODEL_DIR}/feature_cols_optimized.json', 'r') as f:
            feature_cols = json.load(f)
        print("✅ 優化模型加載成功")
        model_version = "optimized"
    except:
        print("⚠️  優化模型未找到，恢復使用原始模型")
        USE_OPTIMIZED = False

# 如果沒有優化模型，使用原始模型
if not USE_OPTIMIZED:
    try:
        with open(f'{MODEL_DIR}/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(f'{MODEL_DIR}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'{MODEL_DIR}/feature_cols.json', 'r') as f:
            feature_cols = json.load(f)
        print("✅ 原始模型加載成功")
        model_version = "original"
    except:
        print("❌ 模型未找到，請先運行一下面的一個腳本:")
        print("  python complete_training.py")
        print("  或")
        print("  python optimize_model.py")
        model = None
        scaler = None
        feature_cols = None
        model_version = "none"

# ============================================================================
# 輔助函數
# ============================================================================

def get_confidence_level(prob):
    """根據概率評估信心等級"""
    if prob > 0.75:
        return "EXCELLENT", 4
    elif prob > 0.65:
        return "GOOD", 3
    elif prob > 0.55:
        return "MODERATE", 2
    elif prob > 0.45:
        return "WEAK", 1
    else:
        return "POOR", 0

def get_action_recommendation(prob, confidence):
    """根據概率和信心前給建議"""
    if confidence >= 3:
        return "STRONG_BUY"
    elif confidence == 2:
        return "BUY"
    elif confidence == 1:
        return "CAUTIOUS"
    else:
        return "AVOID"

# ============================================================================
# API 端點
# ============================================================================

@app.route('/predict_bounce', methods=['POST', 'OPTIONS'])
def predict_bounce():
    """
    預測 BB 反彈成功概率
    
    POST 數據格式:
    {
        "features": {
            "body_ratio": 0.6,
            "wick_ratio": 0.7,
            "vol_ratio": 1.5,
            ...
        }
    }
    
    返回:
    {
        "success_probability": 0.75,
        "predicted_class": 1,
        "confidence": "GOOD",
        "confidence_level": 3,
        "action": "STRONG_BUY",
        "model_version": "optimized",
        "status": "success"
    }
    """
    
    if request.method == 'OPTIONS':
        return '', 204
    
    if model is None or scaler is None or feature_cols is None:
        return jsonify({
            "error": "模型未加載",
            "status": "error"
        }), 500
    
    try:
        data = request.json
        features_dict = data.get('features')
        
        if not features_dict:
            return jsonify({
                "error": "缺少特徵數據",
                "status": "error"
            }), 400
        
        # 構建特徵向量
        feature_vector = []
        for col in feature_cols:
            if col in features_dict:
                feature_vector.append(features_dict[col])
            else:
                feature_vector.append(0)
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # 標準化
        feature_scaled = scaler.transform(feature_vector)
        
        # 預測
        prob = model.predict_proba(feature_scaled)[0]
        success_prob = float(prob[1])
        
        # 動態閾值（優化模型下降低閾值以提升召回率）
        threshold = 0.45 if model_version == "optimized" else 0.5
        predicted_class = int(success_prob >= threshold)
        
        # 信心評級
        confidence, confidence_level = get_confidence_level(success_prob)
        action = get_action_recommendation(success_prob, confidence_level)
        
        response = {
            "success_probability": success_prob,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "action": action,
            "model_version": model_version,
            "threshold": threshold,
            "status": "success"
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

@app.route('/predict_bounce_batch', methods=['POST', 'OPTIONS'])
def predict_bounce_batch():
    """批量預測（用於多個幣種）
    
    POST 數據格式:
    {
        "predictions": [
            {"symbol": "BTCUSDT", "features": {...}},
            {"symbol": "ETHUSDT", "features": {...}}
        ]
    }
    """
    
    if request.method == 'OPTIONS':
        return '', 204
    
    if model is None:
        return jsonify({
            "error": "模型未加載",
            "status": "error"
        }), 500
    
    try:
        data = request.json
        predictions = data.get('predictions', [])
        
        results = []
        for pred in predictions:
            symbol = pred.get('symbol')
            features_dict = pred.get('features')
            
            # 構建特徵向量
            feature_vector = []
            for col in feature_cols:
                if col in features_dict:
                    feature_vector.append(features_dict[col])
                else:
                    feature_vector.append(0)
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_scaled = scaler.transform(feature_vector)
            
            prob = model.predict_proba(feature_scaled)[0]
            success_prob = float(prob[1])
            
            threshold = 0.45 if model_version == "optimized" else 0.5
            predicted_class = int(success_prob >= threshold)
            
            confidence, confidence_level = get_confidence_level(success_prob)
            action = get_action_recommendation(success_prob, confidence_level)
            
            results.append({
                "symbol": symbol,
                "success_probability": success_prob,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "action": action
            })
        
        return jsonify({
            "results": results,
            "count": len(results),
            "model_version": model_version,
            "status": "success"
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """檢查 API 健康狀態"""
    if request.method == 'OPTIONS':
        return '', 204
    
    status = "ok" if model is not None else "not_ready"
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "model_version": model_version,
        "scaler_loaded": scaler is not None,
        "features_loaded": feature_cols is not None,
        "features_count": len(feature_cols) if feature_cols else 0
    }), 200

@app.route('/', methods=['GET', 'OPTIONS'])
def index():
    """API 信息"""
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "name": "BB Bounce ML Predictor API",
        "version": "2.0 (Optimized)",
        "model_version": model_version,
        "endpoints": {
            "/predict_bounce": "POST - 預測反彈成功概率",
            "/predict_bounce_batch": "POST - 批量預測",
            "/health": "GET - 檢查健康狀態",
            "/": "GET - API 信息"
        },
        "improvements": [
            "已添加9個新特徵",
            "已使用類權重提升召回率",
            "已進行網格搜索優化超參數",
            "已實現動態閾值調整",
            "已支持 CORS 跨域請求"
        ]
    }), 200

# ============================================================================
# 錯誤處理
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "端點未找到", "status": "error"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "服務器內部錯誤", "status": "error"}), 500

# ============================================================================
# 啟動
# ============================================================================

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"BB反彈 ML 預測 API - 優化版")
    print(f"{'='*60}")
    
    if model is None:
        print("\n❌ 錯誤：模型未加載")
        print("請先運行以下命令之一:")
        print("  python complete_training.py        # 訓練原始模型")
        print("  python optimize_model.py           # 訓練優化模型")
    else:
        print(f"\n✅ 所有組件已準備就緒")
        print(f"   模型: {type(model).__name__}")
        print(f"   版本: {model_version}")
        print(f"   特徵數: {len(feature_cols)}")
        print(f"\n啟動 API 服務器...")
        print(f"   地址: http://localhost:5000")
        print(f"   CORS: 已啟用")
        print(f"   HTML 児表板: file:///C:/Users/omt23/PycharmProjects/BB-Bounce-ML-Model/dashboard.html")
        print(f"\n   按 CTRL+C 停止\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
