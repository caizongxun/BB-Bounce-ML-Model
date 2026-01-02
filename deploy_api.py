#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BB反彈ML模型 - Flask API 部署
用於 TradingView Pine Script 調用
"""

from flask import Flask, request, jsonify
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

# 加載模型和配置
MODEL_DIR = './models'

try:
    with open(f'{MODEL_DIR}/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ 模型加載成功")
except:
    print("❌ 模型未找到，請先運行 complete_training.py")
    model = None

try:
    with open(f'{MODEL_DIR}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler 加載成功")
except:
    scaler = None

try:
    with open(f'{MODEL_DIR}/feature_cols.json', 'r') as f:
        feature_cols = json.load(f)
    print("✅ 特徵列表加載成功")
except:
    feature_cols = None

# ============================================================================
# API 端點
# ============================================================================

@app.route('/predict_bounce', methods=['POST'])
def predict_bounce():
    """
    預測 BB 反彈成功概率
    
    POST 數據格式：
    {
        "features": {
            "body_ratio": 0.6,
            "wick_ratio": 0.7,
            "vol_ratio": 1.5,
            ...
        }
    }
    
    返回：
    {
        "success_probability": 0.75,
        "predicted_class": 1,
        "confidence": "HIGH",
        "status": "success"
    }
    """
    
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
        predicted_class = int(model.predict(feature_scaled)[0])
        
        # 信心評級
        if success_prob > 0.75:
            confidence = "VERY_HIGH"
            confidence_level = 4
        elif success_prob > 0.65:
            confidence = "HIGH"
            confidence_level = 3
        elif success_prob > 0.55:
            confidence = "MODERATE"
            confidence_level = 2
        else:
            confidence = "LOW"
            confidence_level = 1
        
        response = {
            "success_probability": success_prob,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "status": "success"
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

@app.route('/health', methods=['GET'])
def health():
    """檢查 API 健康狀態"""
    status = "ok" if model is not None else "not_ready"
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features_loaded": feature_cols is not None
    }), 200

@app.route('/', methods=['GET'])
def index():
    """API 信息"""
    return jsonify({
        "name": "BB Bounce ML Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "/predict_bounce": "POST - 預測反彈成功概率",
            "/health": "GET - 檢查健康狀態",
            "/": "GET - API 信息"
        }
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
    print(f"BB反彈 ML 預測 API")
    print(f"{'='*60}")
    
    if model is None:
        print("\n❌ 錯誤：模型未加載")
        print("請先運行以下命令：")
        print("  python complete_training.py")
    else:
        print("\n✅ 所有組件已準備就緒")
        print(f"   模型: {type(model).__name__}")
        print(f"   特徵數: {len(feature_cols)}")
        print(f"\n🚀 啟動 API 服務器...")
        print(f"   地址: http://localhost:5000")
        print(f"   檢查健康: http://localhost:5000/health")
        print(f"\n   按 CTRL+C 停止服務器\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
