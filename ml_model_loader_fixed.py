#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機晨学習模型加載器 - 修正版本
診斷并修謬模型加載問題
"""

import os
import json
import joblib
from pathlib import Path

class ModelLoaderDiagnostics:
    def __init__(self):
        self.diagnostics = {}
        self.models = {}
    
    def diagnose(self):
        """完整的診斷流程"""
        print("\n" + "="*70)
        print("機晨学習模型加載診斷")
        print("="*70 + "\n")
        
        # 步驟 1: 棄查目錄結構
        self.check_directory_structure()
        
        # 步驟 2: 查找模型文件
        self.find_model_files()
        
        # 步驟 3: 檢澿文件大小
        self.check_file_sizes()
        
        # 步驟 4: 嘗試加載
        self.try_loading_models()
        
        # 步驟 5: 檢查元數據
        self.check_metadata()
        
        # 步驟 6: 出力診斷結果
        self.print_report()
    
    def check_directory_structure(self):
        """棄查目錄結構"""
        print("[Step 1] 棄查目錄結構")
        print("-" * 70)
        
        # 查找可能的模型目錄
        possible_dirs = [
            'models',
            './models',
            '../models',
            os.path.join(os.path.expanduser('~'), 'models'),
            '/models',
            os.path.abspath('models')
        ]
        
        for d in possible_dirs:
            exists = os.path.exists(d)
            status = "✅" if exists else "❌"
            print(f"{status} {d}")
            
            if exists:
                self.diagnostics['models_dir'] = d
                # 列出目錄中的文件
                try:
                    files = os.listdir(d)
                    print(f"   有 {len(files)} 文件")
                    for f in files[:10]:  # 顯示前 10 個
                        print(f"   - {f}")
                    if len(files) > 10:
                        print(f"   ... 及其他 {len(files) - 10} 個文件")
                except Exception as e:
                    print(f"   錄誤: {e}")
        
        print()
    
    def find_model_files(self):
        """查找模型文件"""
        print("[Step 2] 查找模型文件")
        print("-" * 70)
        
        models_dir = self.diagnostics.get('models_dir', 'models')
        
        if not os.path.exists(models_dir):
            print(f"❌ models 目錄不存在")
            print(f"   建議: mkdir {models_dir}")
            print()
            return
        
        pkl_files = []
        joblib_files = []
        other_files = []
        
        try:
            for f in os.listdir(models_dir):
                path = os.path.join(models_dir, f)
                if os.path.isfile(path):
                    if f.endswith('.pkl'):
                        pkl_files.append(f)
                    elif f.endswith('.joblib'):
                        joblib_files.append(f)
                    else:
                        other_files.append(f)
        except Exception as e:
            print(f"❌ 錄誤: {e}")
            print()
            return
        
        print(f"✅ 找到 {len(pkl_files)} 個 .pkl 模型")
        for f in pkl_files[:5]:
            print(f"   - {f}")
        if len(pkl_files) > 5:
            print(f"   ... 及其他 {len(pkl_files) - 5} 個")
        
        print(f"✅ 找到 {len(joblib_files)} 個 .joblib 模型")
        for f in joblib_files[:5]:
            print(f"   - {f}")
        if len(joblib_files) > 5:
            print(f"   ... 及其他 {len(joblib_files) - 5} 個")
        
        if len(other_files) > 0:
            print(f"⚠️  找到 {len(other_files)} 個其他文件")
            for f in other_files[:5]:
                print(f"   - {f}")
        
        self.diagnostics['pkl_count'] = len(pkl_files)
        self.diagnostics['joblib_count'] = len(joblib_files)
        self.diagnostics['pkl_files'] = pkl_files
        self.diagnostics['joblib_files'] = joblib_files
        
        total = len(pkl_files) + len(joblib_files)
        if total == 0:
            print(f"\n⚠️  沒有找到任何 ML 模型文件")
            print("建議: 章篃将訓練好的模型 (.pkl) 放到 models/ 目錄")
        
        print()
    
    def check_file_sizes(self):
        """檢澿文件大小"""
        print("[Step 3] 檢澿文件大小")
        print("-" * 70)
        
        models_dir = self.diagnostics.get('models_dir', 'models')
        if not os.path.exists(models_dir):
            print("❌ models 目錄不存在\n")
            return
        
        all_model_files = (
            self.diagnostics.get('pkl_files', []) +
            self.diagnostics.get('joblib_files', [])
        )
        
        if len(all_model_files) == 0:
            print("❌ 沒有找到模型文件\n")
            return
        
        for fname in all_model_files[:10]:
            path = os.path.join(models_dir, fname)
            try:
                size = os.path.getsize(path)
                size_mb = size / (1024 * 1024)
                
                if size < 1000:
                    print(f"⚠️  {fname}: {size} bytes (可能损壞)")
                else:
                    print(f"✅ {fname}: {size_mb:.2f} MB")
            except Exception as e:
                print(f"❌ {fname}: 錄誤 {e}")
        
        if len(all_model_files) > 10:
            print(f"... 及其他 {len(all_model_files) - 10} 個模型\n")
        else:
            print()
    
    def try_loading_models(self):
        """嘗試加載模型"""
        print("[Step 4] 嘗試加載模型")
        print("-" * 70)
        
        models_dir = self.diagnostics.get('models_dir', 'models')
        if not os.path.exists(models_dir):
            print("❌ models 目錄不存在\n")
            return
        
        success_count = 0
        fail_count = 0
        
        all_model_files = (
            self.diagnostics.get('pkl_files', []) +
            self.diagnostics.get('joblib_files', [])
        )
        
        for fname in all_model_files[:5]:  # 先车載後 5 個
            path = os.path.join(models_dir, fname)
            try:
                model = joblib.load(path)
                self.models[fname] = model
                print(f"✅ 加載成功: {fname}")
                print(f"   粗型: {type(model).__name__}")
                success_count += 1
            except Exception as e:
                print(f"❌ 加載失敗: {fname}")
                print(f"   錄誤: {str(e)[:100]}")
                fail_count += 1
        
        if len(all_model_files) > 5:
            print(f"\n... 這是前 5 個，共 {len(all_model_files)} 個模型")
        
        self.diagnostics['loaded_count'] = success_count
        self.diagnostics['failed_count'] = fail_count
        print()
    
    def check_metadata(self):
        """檢查元數據檔案"""
        print("[Step 5] 檢查元數據")
        print("-" * 70)
        
        metadata_paths = [
            'models_metadata.json',
            './models_metadata.json',
            '../models_metadata.json'
        ]
        
        found = False
        for path in metadata_paths:
            if os.path.exists(path):
                found = True
                try:
                    with open(path, 'r') as f:
                        metadata = json.load(f)
                    
                    print(f"✅ 找到 models_metadata.json")
                    print(f"   統計模型數: {metadata.get('total_models', 'N/A')}")
                    
                    symbols = metadata.get('symbols', [])
                    print(f"   支持的代幣: {len(symbols)}")
                    print(f"   - {', '.join(symbols[:3])} ... ")
                    
                    timeframes = metadata.get('timeframes', [])
                    print(f"   支持的時間: {timeframes}")
                    
                    # 例子模型功能
                    sample = list(metadata.get('results', {}).items())[0]
                    if sample:
                        print(f"\n   模型性能示例 (BTCUSDT_1h):")
                        results = metadata.get('results', {})
                        if 'BTCUSDT_1h' in results:
                            btc_1h = results['BTCUSDT_1h']
                            print(f"   - Accuracy: {btc_1h.get('accuracy', 'N/A'):.4f}")
                            print(f"   - Precision: {btc_1h.get('precision', 'N/A'):.4f}")
                            print(f"   - AUC: {btc_1h.get('auc', 'N/A'):.4f}")
                    
                except Exception as e:
                    print(f"❌ 錄誤: {e}")
                
                break
        
        if not found:
            print("⚠️  找不到 models_metadata.json")
        
        print()
    
    def print_report(self):
        """打印最終診斷報告"""
        print("\n" + "="*70)
        print("機晨学習模型診斷報告")
        print("="*70)
        
        pkl_count = self.diagnostics.get('pkl_count', 0)
        joblib_count = self.diagnostics.get('joblib_count', 0)
        total_files = pkl_count + joblib_count
        
        print(f"\n找到的模型文件: {total_files} 個")
        print(f"  - .pkl: {pkl_count}")
        print(f"  - .joblib: {joblib_count}")
        
        loaded = self.diagnostics.get('loaded_count', 0)
        print(f"\n成功加載: {loaded} 個")
        print(f"\u52a0載失敗: {self.diagnostics.get('failed_count', 0)} 個")
        
        print("\n" + "-"*70)
        
        if total_files == 0:
            print("\n⚠️  型鞘: 找不到 ML 模型文件")
            print("\n解決方案:")
            print("1. 確保模型文件存在 models/ 目錄")
            print("2. 檢查檔案名是否正確 (.pkl 或 .joblib)")
            print("3. 如果 models 目錄不存在，控制 mkdir models")
            print("4. 拷赊你的訓練好的模型文件到 models/")
        
        elif loaded == 0:
            print("\n⚠️  型鞘: 找到模型文件但加載失敗")
            print("\n解決方案:")
            print("1. 確保 joblib 庫已安裝: pip install joblib")
            print("2. 確保 sklearn 庫已安裝: pip install scikit-learn")
            print("3. 如果檔案损壞，重新訓練模型")
            print("4. 查看具體錄誤信息： python ml_model_loader_fixed.py 2>&1 | head -50")
        
        else:
            print(f"\n✅ 成功: 已加載 {loaded} 個 ML 模型")
            print(f"\n新的預測服務將使用 ML 模型不会简化反模式")
        
        print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    diagnoser = ModelLoaderDiagnostics()
    diagnoser.diagnose()
