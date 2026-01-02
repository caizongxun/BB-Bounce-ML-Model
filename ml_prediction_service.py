#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 預測服务
使用訓練的機晨学習模型預測 BB Bounce
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

print("\n" + "="*70)
print("ML Prediction Service - BB Bounce Model Prediction")
print("="*70 + "\n")

# ============================================================================
# 模型加载
# ============================================================================

class ModelLoader:
    def __init__(self):
        self.models = {}
        self.metadata = {}
        self.load_models()
    
    def load_models(self):
        """加载训练好的模型"""
        
        # 查找模型文件
        model_dirs = [
            'models/',
            './models/',
            '../models/',
        ]
        
        model_files = {}
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                print(f"[INFO] Found model directory: {model_dir}")
                for file in os.listdir(model_dir):
                    if file.endswith('.pkl') or file.endswith('.joblib'):
                        model_files[file] = os.path.join(model_dir, file)
        
        if not model_files:
            print("[WARNING] No model files found. Using fallback predictor.")
            self.use_fallback = True
            return
        
        self.use_fallback = False
        
        # 加载找到的模型
        for model_name, model_path in model_files.items():
            try:
                self.models[model_name] = joblib.load(model_path)
                print(f"[OK] Loaded model: {model_name}")
            except Exception as e:
                print(f"[ERROR] Failed to load {model_name}: {str(e)[:50]}")
        
        # 加载元数据
        try:
            if os.path.exists('models_metadata.json'):
                with open('models_metadata.json', 'r') as f:
                    self.metadata = json.load(f)
                print(f"[OK] Loaded metadata: {len(self.metadata)} models")
        except Exception as e:
            print(f"[WARNING] Failed to load metadata: {str(e)[:50]}")
    
    def predict(self, features, model_name='default'):
        """使用模型进行预测"""
        
        if self.use_fallback or not self.models:
            # 回退到基本预测
            return self.fallback_predict(features)
        
        try:
            # 获取模型
            if model_name not in self.models:
                model_name = list(self.models.keys())[0]  # 使用第一个可用模型
            
            model = self.models[model_name]
            
            # 准备特征
            X = np.array(features).reshape(1, -1)
            
            # 预测
            prediction = model.predict(X)[0]
            confidence = float(model.predict_proba(X).max()) if hasattr(model, 'predict_proba') else 0.5
            
            return {
                'prediction': int(prediction) if isinstance(prediction, np.integer) else prediction,
                'confidence': float(confidence),
                'model': model_name,
                'type': 'ML'
            }
        
        except Exception as e:
            print(f"[ERROR] Prediction failed: {str(e)[:50]}")
            return self.fallback_predict(features)
    
    def fallback_predict(self, features):
        """回退预测（如果模型不可用）"""
        
        # 简单的启发式规则
        # features: [close, bb_upper, bb_lower, bb_middle, rsi, volatility, ...]
        
        if len(features) >= 5:
            close, bb_upper, bb_lower, bb_middle, rsi = features[:5]
            
            # 基于 BB 和 RSI 的简单规则
            score = 0
            
            # BB 位置
            bb_position = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            # 上升趋势得分
            if bb_position > 0.7:
                score += 2  # 接近上轨
            elif bb_position < 0.3:
                score -= 2  # 接近下轨
            elif bb_position > 0.6:
                score += 1  # 上方
            elif bb_position < 0.4:
                score -= 1  # 下方
            
            # RSI 指标
            if 50 < rsi < 70:
                score += 1  # 温和上升
            elif rsi >= 70:
                score += 2  # 强烈上升
            elif 30 < rsi < 50:
                score -= 1  # 温和下降
            elif rsi <= 30:
                score -= 2  # 强烈下降
            
            # 转换为预测标签
            # 1 = 买入信号，0 = 保持，-1 = 卖出信号
            if score >= 2:
                prediction = 1
                confidence = 0.7 + min(score / 10, 0.2)
            elif score <= -2:
                prediction = -1
                confidence = 0.7 + min(abs(score) / 10, 0.2)
            else:
                prediction = 0
                confidence = 0.5
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'model': 'heuristic',
                'type': 'FALLBACK'
            }
        
        return {
            'prediction': 0,
            'confidence': 0.5,
            'model': 'unknown',
            'type': 'FALLBACK'
        }

# 全局模型加载器
model_loader = ModelLoader()

# ============================================================================
# API 端点
# ============================================================================

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """进行 ML 预测"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        # 提取特征
        features = [
            float(data.get('close', 0)),
            float(data.get('bb_upper', 0)),
            float(data.get('bb_lower', 0)),
            float(data.get('bb_middle', 0)),
            float(data.get('rsi', 50)),
            float(data.get('volatility', 0)),
        ]
        
        # 进行预测
        result = model_loader.predict(features)
        
        # 转换预测结果为可读的信号
        prediction = result['prediction']
        
        signal_map = {
            1: 'BUY',
            0: 'HOLD',
            -1: 'SELL'
        }
        
        return jsonify({
            'signal': signal_map.get(prediction, 'UNKNOWN'),
            'prediction': prediction,
            'confidence': result['confidence'],
            'model_type': result['type'],
            'model_name': result['model'],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/batch', methods=['POST', 'OPTIONS'])
def predict_batch():
    """批量预测"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        candlesticks = data.get('candles', [])
        bb_data = data.get('bb_data', {})
        rsi_data = data.get('rsi_data', {})
        
        results = []
        
        for i, candle in enumerate(candlesticks[-20:]):  # 最后 20 根 K 线
            features = [
                float(candle.get('close', 0)),
                float(bb_data.get('upper', [0])[i]),
                float(bb_data.get('lower', [0])[i]),
                float(bb_data.get('middle', [0])[i]),
                float(rsi_data.get('value', [50])[i]),
                0,  # volatility
            ]
            
            result = model_loader.predict(features)
            signal_map = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}
            
            results.append({
                'time': candle.get('time'),
                'signal': signal_map.get(result['prediction'], 'UNKNOWN'),
                'confidence': result['confidence'],
                'model': result['type']
            })
        
        return jsonify({
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/status', methods=['GET', 'OPTIONS'])
def predict_status():
    """获取预测服务状态"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        'status': 'ok',
        'service': 'ML Prediction Service',
        'models_loaded': len(model_loader.models),
        'model_names': list(model_loader.models.keys()),
        'using_fallback': model_loader.use_fallback,
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# 启动
# ============================================================================

if __name__ == '__main__':
    print("[INFO] ML Prediction Service Started")
    print("[INFO] Address: http://localhost:5002")
    print("[INFO] Endpoints:")
    print("   POST /api/predict - Single prediction")
    print("   POST /api/predict/batch - Batch predictions")
    print("   GET  /api/predict/status - Service status")
    print(f"[INFO] Models loaded: {len(model_loader.models)}")
    print(f"[INFO] Using fallback: {model_loader.use_fallback}")
    print("[INFO] Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5002, debug=False)
