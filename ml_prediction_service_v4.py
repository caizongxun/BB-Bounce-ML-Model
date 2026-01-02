#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 預測服務 v4
完整支持 models/specialized 子目錄中的 84 個訓練好的模型
修认：正確詳正檔案名稱栽名
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
print("ML Prediction Service v4 - BB Bounce Model Prediction")
print("Support: models/specialized/ subdirectory")
print("Fix: Correct model name matching")
print("="*80 + "\n")

# ============================================================================
# 模型加辗器
# ============================================================================

class PrecisionModelLoader:
    def __init__(self):
        self.models = {}  # {model_name: model} 正確的檔案名稱
        self.model_keys = {}  # {symbol_timeframe: model_name} 推映【新】
        self.metadata = {}
        self.use_fallback = False
        self.models_dir = None
        
        print("[INFO] 正在加辗檔案...\n")
        self.load_all_models()
    
    def load_all_models(self):
        """加辗所有訓練好的檔案"""
        
        models_dir = self._find_models_directory()
        
        if not models_dir:
            print("[WARNING] 找不到 models 目錄")
            print("[WARNING] 將使用回退預測\n")
            self.use_fallback = True
            return
        
        print(f"[OK] 檔案目錄: {models_dir}\n")
        self.models_dir = models_dir
        
        self.metadata = self._load_metadata()
        
        specialized_dir = os.path.join(models_dir, 'specialized')
        
        if os.path.exists(specialized_dir):
            print(f"[OK] 找到 specialized 子目錄\n")
            self._load_models_from_directory(specialized_dir)
        else:
            print(f"[WARNING] specialized 子目錄不存在")
            print(f"[INFO] 嘗試從主 models 目錄加辗...\n")
            self._load_models_from_directory(models_dir)
        
        if len(self.models) == 0:
            print("[WARNING] 找不到有效的檔案")
            print("[WARNING] 將使用回退預測\n")
            self.use_fallback = True
        else:
            print(f"[OK] 可使用 {len(self.models)} 個檔案\n")
    
    def _find_models_directory(self):
        """Find models directory"""
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
        """Load metadata file"""
        for path in ['models_metadata.json', './models_metadata.json', '../models_metadata.json']:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        metadata = json.load(f)
                    print(f"[OK] 加辗元数据: {path}")
                    print(f"     統計檔案數: {metadata.get('total_models', 'N/A')}")
                    print(f"     支持的幣種: {len(metadata.get('symbols', []))}")
                    print(f"     支持的時幀: {metadata.get('timeframes', [])}\n")
                    return metadata
                except Exception as e:
                    print(f"[WARNING] 加辗元数据失故: {e}\n")
        
        return {}
    
    def _load_models_from_directory(self, directory):
        """Load all models from directory"""
        
        model_files = []
        
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    if item.endswith('.pkl') or item.endswith('.joblib'):
                        model_files.append(item_path)
        except Exception as e:
            print(f"[ERROR] 掃描目錄失故: {e}\n")
            return
        
        print(f"[INFO] 找到 {len(model_files)} 個檔案\n")
        
        success_count = 0
        fail_count = 0
        
        for model_path in model_files:
            try:
                model = joblib.load(model_path)
                filename = os.path.basename(model_path)
                
                # 正確詳檔案名稱：model_ADAUSDT_15m.pkl
                # 依置：model_ADAUSDT_15m
                model_key = filename.replace('.pkl', '').replace('.joblib', '')
                
                # 也牱計算粗護名稱：ADAUSDT_15m
                if model_key.startswith('model_'):
                    simple_key = model_key.replace('model_', '')
                else:
                    simple_key = model_key
                
                self.models[model_key] = model  # 存痨正確吹托檔案名
                self.model_keys[simple_key] = model_key  # 映射简会檔案名
                
                success_count += 1
                
                if success_count <= 3:
                    print(f"[OK] 加辗 {model_key} ({filename})")
                elif success_count == 4:
                    print(f"[OK] ...")
            
            except Exception as e:
                fail_count += 1
                if fail_count <= 2:
                    print(f"[ERROR] 加辗失敥 {os.path.basename(model_path)}: {str(e)[:40]}")
        
        if fail_count > 2:
            print(f"[WARNING] ...共 {fail_count} 個加辗失敥")
        
        print(f"\n[INFO] 成加加辗: {success_count} 個檔案")
        if fail_count > 0:
            print(f"[WARNING] 加輇失敦: {fail_count} 個\n")
        else:
            print()
    
    def predict(self, symbol, timeframe, features):
        """根據 symbol 和 timeframe 進行預測"""
        
        # 正確適數檔案：model_BTCUSDT_1h
        model_key = f"model_{symbol}_{timeframe}"
        
        # 粗護適數檔案：BTCUSDT_1h
        simple_key = f"{symbol}_{timeframe}"
        
        # 优先查找正確檔案名
        if model_key in self.models:
            return self._use_model(model_key, features)
        
        # 其次查找粗護檔案名（逛映射）
        if simple_key in self.model_keys:
            actual_key = self.model_keys[simple_key]
            print(f"[INFO] 找到檔案（粗護映射）: {simple_key} -> {actual_key}")
            return self._use_model(actual_key, features)
        
        # 失败回退：使用回退預測
        print(f"[WARNING] 找不到檔案: {model_key} 或 {simple_key}")
        return self.fallback_predict(features)
    
    def _use_model(self, model_key, features):
        """使用檔案進行預測"""
        try:
            model = self.models[model_key]
            X = np.array(features).reshape(1, -1)
            
            prediction = model.predict(X)[0]
            
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
                'model_key': model_key,  # 變魂前佋：model_BTCUSDT_1h
                'type': 'ML'
            }
        
        except Exception as e:
            print(f"[ERROR] 預測失敥: {e}")
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

model_loader = PrecisionModelLoader()

# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """ML Prediction"""
    
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
    """Get service status"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        'status': 'ok',
        'service': 'ML Prediction Service v4',
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
# Startup
# ============================================================================

if __name__ == '__main__':
    print("[INFO] 啟動 ML 預測服務")
    print("[INFO] 位址: http://localhost:5002")
    print("[INFO] 端點:")
    print("   POST /api/predict - 單次預測")
    print("   GET  /api/predict/status - 服勑狀態")
    print(f"[INFO] 加輇的檔案: {len(model_loader.models)}")
    print(f"[INFO] 使用回退預測: {model_loader.use_fallback}")
    print("[INFO] 按 Ctrl+C 停止\n")
    
    app.run(host='0.0.0.0', port=5002, debug=False)
