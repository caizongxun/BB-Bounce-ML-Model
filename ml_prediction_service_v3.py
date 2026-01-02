#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 預測服務 v3
完整支持 models/specialized 子目錄中的 42 個訓練好的模型
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib
import os
import json
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

print("\n" + "="*80)
print("ML Prediction Service v3 - BB Bounce Model Prediction")
print("Support: models/specialized/ subdirectory")
print("="*80 + "\n")

# ============================================================================
# 模型加載器
# ============================================================================

class SpecializedModelLoader:
    def __init__(self):
        self.models = {}  # {symbol_timeframe: model}
        self.metadata = {}  # 檔案元數據
        self.model_configs = {}  # {symbol_timeframe: config}
        self.use_fallback = False
        self.models_dir = None
        
        print("[INFO] 正在加載模型...\n")
        self.load_all_models()
    
    def load_all_models(self):
        """加載所有訓練好的模型"""
        
        # 掃描檔案結構
        models_dir = self._find_models_directory()
        
        if not models_dir:
            print("[WARNING] 找不到 models 目錄")
            print("[WARNING] 將使用回退預測\n")
            self.use_fallback = True
            return
        
        print(f"[OK] 檔案目錄: {models_dir}\n")
        self.models_dir = models_dir
        
        # 加載元數據
        self.metadata = self._load_metadata()
        
        # 首先嘗試加載 models/specialized/ 中的模型
        specialized_dir = os.path.join(models_dir, 'specialized')
        
        if os.path.exists(specialized_dir):
            print(f"[OK] 找到 specialized 子目錄: {specialized_dir}\n")
            self._load_models_from_directory(specialized_dir)
        else:
            print(f"[WARNING] specialized 子目錄不存在: {specialized_dir}")
            print(f"[INFO] 嘗試從主 models 目錄加載...\n")
            self._load_models_from_directory(models_dir)
        
        if len(self.models) == 0:
            print("[WARNING] 找不到有效的模型")
            print("[WARNING] 將使用回退預測\n")
            self.use_fallback = True
        else:
            print(f"[OK] 可使用 {len(self.models)} 個模型\n")
    
    def _find_models_directory(self):
        """查找 models 目錄"""
        possible_paths = [
            'models',
            './models',
            '../models',
            os.path.join(os.path.expanduser('~'), 'models'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isdir(path):
                return os.path.abspath(path)
        
        return None
    
    def _load_metadata(self):
        """加載元數據檔案"""
        for path in ['models_metadata.json', './models_metadata.json', '../models_metadata.json']:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        metadata = json.load(f)
                    print(f"[OK] 加載元數據: {path}")
                    print(f"     統計模型數: {metadata.get('total_models', 'N/A')}")
                    print(f"     支持的幣種: {len(metadata.get('symbols', []))}")
                    print(f"     支持的時幀: {metadata.get('timeframes', [])}\n")
                    return metadata
                except Exception as e:
                    print(f"[WARNING] 加載元數據失敗: {e}\n")
        
        return {}
    
    def _load_models_from_directory(self, directory):
        """從指定目錄加載所有模型"""
        
        model_files = []
        
        # 查找所有 .pkl 和 .joblib 文件
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    if item.endswith('.pkl') or item.endswith('.joblib'):
                        model_files.append(item_path)
        except Exception as e:
            print(f"[ERROR] 掃描目錄失敗: {e}\n")
            return
        
        print(f"[INFO] 找到 {len(model_files)} 個模型檔案\n")
        
        success_count = 0
        fail_count = 0
        
        for model_path in model_files:
            try:
                model = joblib.load(model_path)
                filename = os.path.basename(model_path)
                
                # 解析模型章節（例如: BTCUSDT_1h.pkl)
                name_parts = filename.replace('.pkl', '').replace('.joblib', '').split('_')
                
                if len(name_parts) >= 2:
                    symbol = name_parts[0]
                    timeframe = '_'.join(name_parts[1:])
                    key = f"{symbol}_{timeframe}"
                    
                    self.models[key] = model
                    success_count += 1
                    
                    if success_count <= 5:  # 只顯示前 5 個
                        print(f"[OK] 加載 {key} ({filename})")
                    elif success_count == 6:
                        print(f"[OK] ...")
            
            except Exception as e:
                fail_count += 1
                if fail_count <= 3:
                    print(f"[ERROR] 加載失敗 {os.path.basename(model_path)}: {str(e)[:40]}")
        
        if fail_count > 3:
            print(f"[WARNING] ...共 {fail_count} 個加載失敗")
        
        print(f"\n[INFO] 成功加載: {success_count} 個模型")
        if fail_count > 0:
            print(f"[WARNING] 加載失敗: {fail_count} 個\n")
        else:
            print()
    
    def predict(self, symbol, timeframe, features):
        """根據 symbol 和 timeframe 使用精細模型進行預測"""
        
        key = f"{symbol}_{timeframe}"
        
        # 查找特定的模型
        if key in self.models:
            return self._use_model(key, features)
        
        # 嘗試找到 symbol 相關的其他時幀模型
        for available_key in self.models.keys():
            if available_key.startswith(f"{symbol}_"):
                print(f"[INFO] 找不到 {key}，使用 {available_key}")
                return self._use_model(available_key, features)
        
        # 使用回退預測
        return self.fallback_predict(features)
    
    def _use_model(self, key, features):
        """使用模型進行預測"""
        try:
            model = self.models[key]
            X = np.array(features).reshape(1, -1)
            
            prediction = model.predict(X)[0]
            
            # 計算信心度
            confidence = 0.5
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))
            elif hasattr(model, 'decision_function'):
                decision = model.decision_function(X)[0]
                confidence = float(1.0 / (1.0 + np.exp(-decision)))
            
            return {
                'prediction': int(prediction) if isinstance(prediction, np.integer) else prediction,
                'confidence': confidence,
                'model_key': key,
                'type': 'ML'
            }
        
        except Exception as e:
            print(f"[ERROR] 預測失敗: {e}")
            return self.fallback_predict(features)
    
    def fallback_predict(self, features):
        """回退預測"""
        if len(features) >= 5:
            close, bb_upper, bb_lower, bb_middle, rsi = features[:5]
            
            bb_position = (close - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            score = 0
            if bb_position > 0.7:
                score += 2
            elif bb_position < 0.3:
                score -= 2
            elif bb_position > 0.6:
                score += 1
            elif bb_position < 0.4:
                score -= 1
            
            if 50 < rsi < 70:
                score += 1
            elif rsi >= 70:
                score += 2
            elif 30 < rsi < 50:
                score -= 1
            elif rsi <= 30:
                score -= 2
            
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
                'model_key': 'fallback',
                'type': 'FALLBACK'
            }
        
        return {
            'prediction': 0,
            'confidence': 0.5,
            'model_key': 'unknown',
            'type': 'FALLBACK'
        }

# 全局模型加載器
model_loader = SpecializedModelLoader()

# ============================================================================
# API 端點
# ============================================================================

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """ML 預測"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        symbol = data.get('symbol', 'BTCUSDT')
        timeframe = data.get('timeframe', '1h')
        
        features = [
            float(data.get('close', 0)),
            float(data.get('bb_upper', 0)),
            float(data.get('bb_lower', 0)),
            float(data.get('bb_middle', 0)),
            float(data.get('rsi', 50)),
            float(data.get('volatility', 0)),
        ]
        
        result = model_loader.predict(symbol, timeframe, features)
        
        prediction = result['prediction']
        signal_map = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}
        
        return jsonify({
            'signal': signal_map.get(prediction, 'UNKNOWN'),
            'prediction': prediction,
            'confidence': result['confidence'],
            'model_type': result['type'],
            'model': result['model_key'],
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/status', methods=['GET', 'OPTIONS'])
def predict_status():
    """取得服務狀態"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        'status': 'ok',
        'service': 'ML Prediction Service v3',
        'models_loaded': len(model_loader.models),
        'model_list': sorted(list(model_loader.models.keys())),
        'using_fallback': model_loader.use_fallback,
        'models_directory': model_loader.models_dir,
        'metadata': {
            'total_models': model_loader.metadata.get('total_models', 0),
            'symbols': model_loader.metadata.get('symbols', []),
            'timeframes': model_loader.metadata.get('timeframes', [])
        },
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# 啟動
# ============================================================================

if __name__ == '__main__':
    print("[INFO] 啟動 ML 預測服務")
    print("[INFO] 位址: http://localhost:5002")
    print("[INFO] 端點:")
    print("   POST /api/predict - 單次預測")
    print("   GET  /api/predict/status - 服務狀態")
    print(f"[INFO] 加載的模型: {len(model_loader.models)}")
    print(f"[INFO] 使用回退預測: {model_loader.use_fallback}")
    print("[INFO] 按 Ctrl+C 停止\n")
    
    app.run(host='0.0.0.0', port=5002, debug=False)
