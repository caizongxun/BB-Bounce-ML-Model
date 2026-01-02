#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图表数据源服务
提供真实 K 棒数据和 Bollinger Bands 计算
✅ 修复：排除不完整的最后一根 K 棒
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import ta
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

print("\n" + "="*70)
print("Chart Data Service - TradingView Real-time Data Provider")
print("="*70 + "\n")

# ============================================================================
# 配置
# ============================================================================

class Config:
    BINANCE_BASE_URL = 'https://api.binance.us'
    
    TIMEFRAME_MAP = {
        '1m': '1m', '5m': '5m', '15m': '15m',
        '1h': '1h', '4h': '4h', '1d': '1d'
    }
    
    # 根据时间框架提供更多K棒
    LOOKBACK_MAP = {
        '15m': 200,  # 增加到 200
        '1h': 200,   # 增加到 200
        '4h': 150,   # 增加到 150
        '1d': 100    # 增加到 100
    }

# ============================================================================
# 数据推獲
# ============================================================================

def fetch_klines(symbol, timeframe, limit=200):
    """从 Binance US 抓取 K 棒
    
    重要：排除最后一根未完整 K 棒以馁自宇模式正理中的 K 棒
    """
    
    interval = Config.TIMEFRAME_MAP.get(timeframe, '15m')
    url = f"{Config.BINANCE_BASE_URL}/api/v3/klines"
    # 触及最后一根正形成的 K 棒，所以额外请租 1 根
    params = {'symbol': symbol, 'interval': interval, 'limit': min(limit + 1, 1000)}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        klines = response.json()
        
        # 排除最后一根（正在形成中）
        if len(klines) > limit:
            klines = klines[:-1]
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    except Exception as e:
        print(f"[ERROR] Failed to fetch {symbol} {timeframe}: {str(e)[:50]}")
        return None

def calculate_bollinger_bands(df, period=20, stddev=2):
    """计算 Bollinger Bands"""
    
    try:
        bb = ta.volatility.BollingerBands(
            df['close'], 
            window=period, 
            window_dev=stddev
        )
        
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        return df
    except:
        return None

def calculate_rsi(df, period=14):
    """计算 RSI"""
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
        return df
    except:
        return None

# ============================================================================
# API 端点
# ============================================================================

@app.route('/api/klines', methods=['GET', 'OPTIONS'])
def get_klines():
    """获取 K 棒数据"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        timeframe = request.args.get('timeframe', '1h')
        period = int(request.args.get('period', 20))
        stddev = float(request.args.get('stddev', 2))
        limit = int(request.args.get('limit', 200))
        
        # 限制最大数量
        limit = min(limit, 1000)
        
        # 推遭数据（排除最后一根）
        df = fetch_klines(symbol, timeframe, limit=limit)
        if df is None:
            return jsonify({'error': 'Failed to fetch data'}), 500
        
        # 计算指標
        df = calculate_bollinger_bands(df, period=period, stddev=stddev)
        df = calculate_rsi(df)
        
        # 转换数据为 JSON
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'period': period,
            'stddev': stddev,
            'total_candles': len(df),
            'candles': [],
            'bb_upper': [],
            'bb_middle': [],
            'bb_lower': [],
            'rsi': [],
            'stats': {}
        }
        
        df = df.dropna()
        
        for idx, row in df.iterrows():
            timestamp = int(idx.timestamp())
            
            result['candles'].append({
                'time': timestamp,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })
            
            result['bb_upper'].append({
                'time': timestamp,
                'value': float(row['bb_upper'])
            })
            
            result['bb_middle'].append({
                'time': timestamp,
                'value': float(row['bb_middle'])
            })
            
            result['bb_lower'].append({
                'time': timestamp,
                'value': float(row['bb_lower'])
            })
            
            result['rsi'].append({
                'time': timestamp,
                'value': float(row['rsi'])
            })
        
        # 统计信息
        if len(df) > 0:
            last_row = df.iloc[-1]
            result['stats'] = {
                'current_price': float(last_row['close']),
                'bb_upper': float(last_row['bb_upper']),
                'bb_middle': float(last_row['bb_middle']),
                'bb_lower': float(last_row['bb_lower']),
                'bb_width': float(last_row['bb_upper'] - last_row['bb_lower']),
                'rsi': float(last_row['rsi']),
                'volatility': float((last_row['bb_upper'] - last_row['bb_lower']) / last_row['bb_middle'] * 100),
                'highest': float(df['high'].max()),
                'lowest': float(df['low'].min())
            }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """分析币种是否触及 BB"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        symbol = data.get('symbol', 'BTCUSDT')
        timeframe = data.get('timeframe', '1h')
        period = int(data.get('period', 20))
        stddev = float(data.get('stddev', 2))
        
        limit = Config.LOOKBACK_MAP.get(timeframe, 100)
        
        # 推遭数据搆除最后一根）
        df = fetch_klines(symbol, timeframe, limit=limit)
        if df is None:
            return jsonify({'error': 'Failed to fetch data'}), 500
        
        # 计算指標
        df = calculate_bollinger_bands(df, period=period, stddev=stddev)
        df = calculate_rsi(df)
        
        last_row = df.iloc[-1]
        close_price = last_row['close']
        bb_upper = last_row['bb_upper']
        bb_lower = last_row['bb_lower']
        rsi = last_row['rsi']
        
        # 判断是否触及
        touched = None
        touch_type = None
        distance = None
        
        if close_price <= bb_lower * 1.005:
            touched = True
            touch_type = 'LOWER'
            distance = ((close_price - bb_lower) / bb_lower * 100)
        elif close_price >= bb_upper / 1.005:
            touched = True
            touch_type = 'UPPER'
            distance = ((bb_upper - close_price) / bb_upper * 100)
        else:
            # 位于中间
            mid_to_lower = (close_price - bb_lower) / (bb_upper - bb_lower) * 100
            distance = min(mid_to_lower, 100 - mid_to_lower)
            touched = False
        
        return jsonify({
            'symbol': symbol,
            'current_price': float(close_price),
            'bb_upper': float(bb_upper),
            'bb_lower': float(bb_lower),
            'bb_middle': float(last_row['bb_middle']),
            'rsi': float(rsi),
            'touched': touched,
            'touch_type': touch_type,
            'distance_to_band': float(distance),
            'volatility': float((bb_upper - bb_lower) / last_row['bb_middle'] * 100)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    """健康状态"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    return jsonify({
        'status': 'ok',
        'service': 'Chart Data Service',
        'timestamp': datetime.now().isoformat(),
        'max_candles': 1000
    })

# ============================================================================
# 启动
# ============================================================================

if __name__ == '__main__':
    print("[INFO] Chart Data Service Started")
    print("[INFO] Address: http://localhost:5001")
    print("[INFO] Endpoints:")
    print("   GET  /api/klines - Fetch K-bars (completed candles only)")
    print("   POST /api/analyze - Analyze BB touch points")
    print("   GET  /api/health - Service health check")
    print("[INFO] Note: Last incomplete candle is excluded from data")
    print("[INFO] Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False)
