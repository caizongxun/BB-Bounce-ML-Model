#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML 預測服務 v5
自動適應不同的檔案特徵需求（正型呀，不是依賴）
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
print("ML Prediction Service v5 - BB Bounce Model Prediction")
print("Auto-adapt: Support both 6 and 25 feature models")
print("="*80 + "\n")

# ============================================================================
# Model Loader
# ============================================================================

class FlexibleModelLoader:
    def __init__(self):
        self.models = {}
        self.model_keys = {}
        self.model_feature_counts = {}  # Track feature count for each model
        self.metadata = {}
        self.use_fallback = False
        self.models_dir = None
        
        print("[INFO] Loading models...\n")
        self.load_all_models()
    
    def load_all_models(self):
        models_dir = self._find_models_directory()
        
        if not models_dir:
            print("[WARNING] Cannot find models directory")
            print("[WARNING] Using fallback prediction\n")
            self.use_fallback = True
            return
        
        print(f"[OK] Models directory: {models_dir}\n")
        self.models_dir = models_dir
        
        self.metadata = self._load_metadata()
        
        specialized_dir = os.path.join(models_dir, 'specialized')
        
        if os.path.exists(specialized_dir):
            print(f"[OK] Found specialized subdirectory\n")
            self._load_models_from_directory(specialized_dir)
        else:
            print(f"[WARNING] specialized subdirectory not found")
            print(f"[INFO] Trying to load from main models directory...\n")
            self._load_models_from_directory(models_dir)
        
        if len(self.models) == 0:
            print("[WARNING] No valid models found")
            print("[WARNING] Using fallback prediction\n")
            self.use_fallback = True
        else:
            print(f"[OK] Ready to use {len(self.models)} models\n")
    
    def _find_models_directory(self):
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
        for path in ['models_metadata.json', './models_metadata.json', '../models_metadata.json']:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        metadata = json.load(f)
                    print(f"[OK] Loaded metadata: {path}")
                    print(f"     Total models: {metadata.get('total_models', 'N/A')}")
                    print(f"     Supported symbols: {len(metadata.get('symbols', []))}")
                    print(f"     Supported timeframes: {metadata.get('timeframes', [])}\n")
                    return metadata
                except Exception as e:
                    print(f"[WARNING] Failed to load metadata: {e}\n")
        
        return {}
    
    def _load_models_from_directory(self, directory):
        model_files = []
        
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    if item.endswith('.pkl') or item.endswith('.joblib'):
                        model_files.append(item_path)
        except Exception as e:
            print(f"[ERROR] Failed to scan directory: {e}\n")
            return
        
        print(f"[INFO] Found {len(model_files)} model files\n")
        
        success_count = 0
        fail_count = 0
        
        for model_path in model_files:
            try:
                model = joblib.load(model_path)
                filename = os.path.basename(model_path)
                
                model_key = filename.replace('.pkl', '').replace('.joblib', '')
                
                if model_key.startswith('model_'):
                    simple_key = model_key.replace('model_', '')
                else:
                    simple_key = model_key
                
                # Try to detect feature count
                feature_count = self._detect_feature_count(model)
                
                self.models[model_key] = model
                self.model_keys[simple_key] = model_key
                self.model_feature_counts[model_key] = feature_count
                
                success_count += 1
                
                if success_count <= 3:
                    print(f"[OK] Loaded {model_key} (features: {feature_count})")
                elif success_count == 4:
                    print(f"[OK] ...")
            
            except Exception as e:
                fail_count += 1
                if fail_count <= 2:
                    print(f"[ERROR] Failed to load {os.path.basename(model_path)}: {str(e)[:40]}")
        
        if fail_count > 2:
            print(f"[WARNING] ... {fail_count} total load failures")
        
        print(f"\n[INFO] Successfully loaded: {success_count} models")
        if fail_count > 0:
            print(f"[WARNING] Load failures: {fail_count}\n")
        else:
            print()
    
    def _detect_feature_count(self, model):
        """Detect the number of features the model expects"""
        try:
            # For sklearn models
            if hasattr(model, 'n_features_in_'):
                return model.n_features_in_
            # For XGBoost models
            elif hasattr(model, 'n_features_'):
                return model.n_features_
            # For other sklearn models
            elif hasattr(model, 'coef_'):
                return model.coef_.shape[1]
        except:
            pass
        
        return 6  # Default fallback
    
    def predict(self, symbol, timeframe, features):
        """Make prediction"""
        
        model_key = f"model_{symbol}_{timeframe}"
        simple_key = f"{symbol}_{timeframe}"
        
        # Try to find the model
        if model_key in self.models:
            return self._use_model(model_key, features)
        
        # Try simple key mapping
        if simple_key in self.model_keys:
            actual_key = self.model_keys[simple_key]
            print(f"[INFO] Found model (via mapping): {simple_key} -> {actual_key}")
            return self._use_model(actual_key, features)
        
        # Fallback
        print(f"[WARNING] Model not found: {model_key} or {simple_key}")
        return self.fallback_predict(features)
    
    def _use_model(self, model_key, features):
        """Use model for prediction"""
        try:
            model = self.models[model_key]
            feature_count = self.model_feature_counts.get(model_key, 6)
            
            # **CRITICAL FIX**: Extend features to match model's expected input
            X = self._prepare_features(features, feature_count)
            
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
                'model_key': model_key,
                'type': 'ML',
                'features_used': len(X[0])
            }
        
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return self.fallback_predict(features)
    
    def _prepare_features(self, features, required_count):
        """Prepare features to match model's expected input"""
        features = np.array(features, dtype=float)
        
        # If we have fewer features than needed, pad with zeros or repeat
        if len(features) < required_count:
            print(f"[INFO] Extending features from {len(features)} to {required_count}")
            # Pad with the last value or zeros
            padding = np.zeros(required_count - len(features))
            features = np.concatenate([features, padding])
        
        # If we have more features than needed, truncate
        elif len(features) > required_count:
            print(f"[INFO] Truncating features from {len(features)} to {required_count}")
            features = features[:required_count]
        
        return features.reshape(1, -1)
    
    def fallback_predict(self, features):
        """Fallback prediction"""
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

model_loader = FlexibleModelLoader()

# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
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
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        'status': 'ok',
        'service': 'ML Prediction Service v5',
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

if __name__ == '__main__':
    print("[INFO] Starting ML Prediction Service")
    print("[INFO] Address: http://localhost:5002")
    print("[INFO] Endpoints:")
    print("   POST /api/predict - Single prediction")
    print("   GET  /api/predict/status - Service status")
    print(f"[INFO] Models loaded: {len(model_loader.models)}")
    print(f"[INFO] Using fallback: {model_loader.use_fallback}")
    print("[INFO] Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5002, debug=False)
