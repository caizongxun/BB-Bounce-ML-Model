#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BB反弹 ML 预测 API - 支持多个币种和时间框架
动态加载不同的模型
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
CORS(app)

MODELS_DIR = './models/specialized'
FALLBACK_MODELS_DIR = './models'

# 全所有22个币种
AVAILABLE_SYMBOLS = [
    'AAVESTDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
    'AVAXUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT',
    'DOTUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT', 'LINKUSDT',
    'LTCUSDT', 'MATICUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT',
    'UNIUSDT', 'XRPUSDT'
]

AVAILABLE_TIMEFRAMES = ['15m', '1h']

# 模型快取
model_cache = {}

def load_model(symbol, timeframe):
    """
    加载特定币种时间框的模型
    优先级: 专所有模型 > 通用优化模型 > 原始模型
    """
    
    model_key = f"{symbol}_{timeframe}"
    
    # 检查快取
    if model_key in model_cache:
        return model_cache[model_key]
    
    model_data = {}
    model_source = None
    
    # 优先级1: 专所有模型目录
    specialized_model_path = f'{MODELS_DIR}/model_{model_key}.pkl'
    if Path(specialized_model_path).exists():
        try:
            with open(specialized_model_path, 'rb') as f:
                model_data['model'] = pickle.load(f)
            with open(f'{MODELS_DIR}/scaler_{model_key}.pkl', 'rb') as f:
                model_data['scaler'] = pickle.load(f)
            with open(f'{MODELS_DIR}/features_{model_key}.json', 'r') as f:
                model_data['features'] = json.load(f)
            model_source = 'specialized'
            model_cache[model_key] = model_data
            return model_data
        except:
            pass
    
    # 优先级2: 通用优化模型
    opt_model_path = f'{FALLBACK_MODELS_DIR}/best_model_optimized.pkl'
    if Path(opt_model_path).exists():
        try:
            with open(opt_model_path, 'rb') as f:
                model_data['model'] = pickle.load(f)
            with open(f'{FALLBACK_MODELS_DIR}/scaler_optimized.pkl', 'rb') as f:
                model_data['scaler'] = pickle.load(f)
            with open(f'{FALLBACK_MODELS_DIR}/feature_cols_optimized.json', 'r') as f:
                model_data['features'] = json.load(f)
            model_source = 'optimized'
            model_cache[model_key] = model_data
            return model_data
        except:
            pass
    
    # 优先级3: 原始模型
    try:
        with open(f'{FALLBACK_MODELS_DIR}/best_model.pkl', 'rb') as f:
            model_data['model'] = pickle.load(f)
        with open(f'{FALLBACK_MODELS_DIR}/scaler.pkl', 'rb') as f:
            model_data['scaler'] = pickle.load(f)
        with open(f'{FALLBACK_MODELS_DIR}/feature_cols.json', 'r') as f:
            model_data['features'] = json.load(f)
        model_source = 'original'
        model_cache[model_key] = model_data
        return model_data
    except:
        return None

def get_confidence_level(prob):
    """评估信心级别"""
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
    """给出建议"""
    if confidence >= 3:
        return "STRONG_BUY"
    elif confidence == 2:
        return "BUY"
    elif confidence == 1:
        return "CAUTIOUS"
    else:
        return "AVOID"

# ============================================================================
# API 端点
# ============================================================================

@app.route('/predict_bounce', methods=['POST', 'OPTIONS'])
def predict_bounce():
    """
    预测反弹成功概率
    
    POST 请求格式:
    {
        "symbol": "BTCUSDT",
        "timeframe": "15m",
        "features": {
            "body_ratio": 0.6,
            "wick_ratio": 0.7,
            ...
        }
    }
    """
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        symbol = data.get('symbol', 'BTCUSDT')
        timeframe = data.get('timeframe', '15m')
        features_dict = data.get('features', {})
        
        # 验证币种和时间框
        if symbol not in AVAILABLE_SYMBOLS:
            return jsonify({
                "error": f"不支持的币种: {symbol}",
                "available_symbols": AVAILABLE_SYMBOLS,
                "status": "error"
            }), 400
        
        if timeframe not in AVAILABLE_TIMEFRAMES:
            return jsonify({
                "error": f"不支持的时间框: {timeframe}",
                "available_timeframes": AVAILABLE_TIMEFRAMES,
                "status": "error"
            }), 400
        
        # 加载模型
        model_data = load_model(symbol, timeframe)
        if model_data is None:
            return jsonify({
                "error": "模型未找到",
                "status": "error"
            }), 500
        
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['features']
        
        if not features_dict:
            return jsonify({
                "error": "缺少特征数据",
                "status": "error"
            }), 400
        
        # 构建特征向量
        feature_vector = []
        for col in feature_cols:
            feature_vector.append(features_dict.get(col, 0))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_scaled = scaler.transform(feature_vector)
        
        # 预测
        prob = model.predict_proba(feature_scaled)[0]
        success_prob = float(prob[1])
        
        # 动态阈值
        threshold = 0.45
        predicted_class = int(success_prob >= threshold)
        
        # 信心评级
        confidence, confidence_level = get_confidence_level(success_prob)
        action = get_action_recommendation(success_prob, confidence_level)
        
        response = {
            "symbol": symbol,
            "timeframe": timeframe,
            "success_probability": success_prob,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "action": action,
            "threshold": threshold,
            "status": "success"
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

@app.route('/available_models', methods=['GET', 'OPTIONS'])
def available_models():
    """获取所有可用的币种和时间框"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "symbols": AVAILABLE_SYMBOLS,
        "timeframes": AVAILABLE_TIMEFRAMES,
        "total_combinations": len(AVAILABLE_SYMBOLS) * len(AVAILABLE_TIMEFRAMES),
        "status": "success"
    }), 200

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """检查 API 健康状态"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    status = "ok"
    return jsonify({
        "status": status,
        "api_version": "2.1 (Multi-Symbol, Multi-Timeframe)",
        "available_symbols": len(AVAILABLE_SYMBOLS),
        "available_timeframes": len(AVAILABLE_TIMEFRAMES),
        "features": [
            "Multi-symbol support",
            "Multi-timeframe support",
            "Dynamic model loading",
            "Fallback model strategy",
            "CORS enabled"
        ]
    }), 200

@app.route('/', methods=['GET', 'OPTIONS'])
def index():
    """API 信息"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        "name": "BB Bounce ML Predictor API",
        "version": "2.1 (Multi-Symbol, Multi-Timeframe)",
        "endpoints": {
            "/predict_bounce": "POST - 预测反弹成功概率",
            "/available_models": "GET - 获取可用的币种和时间框",
            "/health": "GET - 检查 API 状态",
            "/": "GET - API 信息"
        },
        "available_symbols": AVAILABLE_SYMBOLS,
        "available_timeframes": AVAILABLE_TIMEFRAMES
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "端点未找到", "status": "error"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "服务器内部错误", "status": "error"}), 500

# ============================================================================
# 启动
# ============================================================================

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"BB反弹 ML 预测 API - 上位版")
    print(f"{'='*60}\n")
    
    print("[INFO] 上位模型系统")
    print(f"[INFO] 支持币种: {len(AVAILABLE_SYMBOLS)} 个")
    print(f"[INFO] 支持时间框: {len(AVAILABLE_TIMEFRAMES)} 个")
    print(f"[INFO] 模型组合: {len(AVAILABLE_SYMBOLS) * len(AVAILABLE_TIMEFRAMES)} 个\n")
    
    print(f"[INFO] API 地址: http://localhost:5000")
    print(f"[INFO] CORS: 已启用")
    print(f"[INFO] 按 CTRL+C 停止\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
