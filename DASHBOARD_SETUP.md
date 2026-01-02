# BB Bounce Pro - Advanced Trading Dashboard Setup Guide

## Overview

Your professional trading dashboard is ready. It features:

- **Real-time TradingView-like charts** (using Lightweight Charts library)
- **Bollinger Bands visualization** (Upper, Middle, Lower bands)
- **Live data from Binance** (22 supported cryptocurrencies)
- **Trading signals sidebar** (shows active buy/sell opportunities)
- **Statistics panel** (Current price, BB values, volatility, RSI)
- **Professional dark theme** (optimized for long trading sessions)

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard (Advanced)                      │
│                   dashboard_advanced.html                    │
│                                                              │
│  [Chart Area - Lightweight Charts]      [Sidebar Control]   │
│  - K-line candlesticks                  - Symbol selector   │
│  - BB Upper/Middle/Lower bands          - Timeframe picker  │
│  - RSI indicator                        - BB settings       │
│  - Crosshair interaction                - Live signals      │
│  - Zoom & Pan controls                  - Statistics        │
└─────────────────────────────────────────────────────────────┘
                        ↓ (fetch data)
┌─────────────────────────────────────────────────────────────┐
│          Chart Data Service (Backend)                       │
│               chart_data_service.py                         │
│                   (Port 5001)                               │
│                                                              │
│  - GET /api/klines → Real-time K-bar data from Binance    │
│  - POST /api/analyze → BB touch detection & analysis      │
│  - GET /api/health → Service status                        │
└─────────────────────────────────────────────────────────────┘
                        ↓ (get crypto data)
┌─────────────────────────────────────────────────────────────┐
│                Binance US Public API                        │
│            (No authentication required)                     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Step 1: Start Chart Data Service

```bash
# Terminal 1: Start the chart data backend service
python chart_data_service.py

# Output:
# [INFO] Chart Data Service Started
# [INFO] Address: http://localhost:5001
# [INFO] Endpoints:
#    GET  /api/klines - Fetch K-bars
#    POST /api/analyze - Analyze BB touches
#    GET  /api/health - Service status
```

### Step 2: Open Dashboard

```bash
# In your browser, open:
file:///C:/Users/omt23/PycharmProjects/BB-Bounce-ML-Model/dashboard_advanced.html

# Or use VS Code Live Server:
# Right-click dashboard_advanced.html → "Open with Live Server"
```

### Step 3: Use the Dashboard

1. **Select Symbol** - Choose from 12 major cryptocurrencies
2. **Select Timeframe** - 15m, 1h, 4h, 1d available
3. **Configure BB** - Adjust period (10-50) and standard deviation (1-5)
4. **Click "Update Chart"** - Fetches real data and displays chart
5. **View Signals** - Trading signals appear in the sidebar

## Features Explained

### Chart Area

**Candles:**
- Green = Bullish (close > open)
- Red = Bearish (close < open)
- Width = Time period you selected

**Bollinger Bands:**
- Red dashed line = BB Upper band (potential resistance)
- Blue dashed line = BB Middle band (20-period SMA)
- Green dashed line = BB Lower band (potential support)

**When price touches BB:**
- Touches lower band = potential BUY signal
- Touches upper band = potential SELL signal

### Sidebar Controls

**Symbol Selector:**
- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- BNBUSDT (Binance Coin)
- SOLUSDT (Solana)
- UNIUSDT (Uniswap)
- AVAXUSDT (Avalanche)
- ADAUSDT (Cardano)
- DOGEUSDT (Dogecoin)
- DOTUSDT (Polkadot)
- LINKUSDT (Chainlink)
- LTCUSDT (Litecoin)
- XRPUSDT (XRP)

**Timeframe Options:**
- 15m (volatile, many signals)
- 1h (balanced, recommended)
- 4h (stable, fewer false signals)
- 1d (long-term trend)

**BB Settings:**
- Period: 10-50 (default 20) - Controls BB width
- Standard Dev: 1-5 (default 2) - Controls band distance

### Statistics Panel

- **Current Price** - Latest closing price
- **BB Upper** - Resistance level
- **BB Lower** - Support level
- **Volatility** - (Upper - Lower) / Middle × 100%

### Trading Signals

Shows active buy/sell opportunities:
- Symbol + Timeframe
- Touch type (LOWER/UPPER)
- Confidence level
- RSI value
- Visual confidence bar

## Advanced Configuration

### Customize BB Parameters

Default: Period 20, StdDev 2 (standard Bollinger Bands)

**For more sensitivity (more signals):**
- Period: 15-18
- StdDev: 1.5-1.8

**For less noise (fewer false signals):**
- Period: 22-25
- StdDev: 2.2-2.5

**For swing trading (longer timeframe):**
- Period: 20
- StdDev: 3
- Timeframe: 4h or 1d

**For scalping (shorter timeframe):**
- Period: 18
- StdDev: 1.8
- Timeframe: 15m

### Add More Symbols

Edit `dashboard_advanced.html` around line 150:

```html
<select id="symbol" onchange="updateChart()">
    <option value="BTCUSDT">BTCUSDT - Bitcoin</option>
    <option value="ETHUSDT">ETHUSDT - Ethereum</option>
    <!-- Add new symbol here -->
    <option value="NEUSDT">NEUSDT - Neo</option>
</select>
```

## Troubleshooting

### Chart not loading?

1. Check if `chart_data_service.py` is running on port 5001
2. Verify Binance API is accessible (ping api.binance.us)
3. Check browser console for errors (F12 → Console)

### No data displayed?

1. Click "Update Chart" button
2. Verify symbol and timeframe are selected
3. Check network tab (F12 → Network) for API response

### Chart too crowded?

1. Increase timeframe (1h instead of 15m)
2. Adjust BB period to 25-30
3. Zoom out on chart (scroll wheel or double-click)

## Performance Tips

1. **Use 1-hour charts** for best performance
2. **Limit chart history** to 100 candles (auto-managed)
3. **Close unused tabs** - Charts consume memory
4. **Refresh periodically** - Every 5 minutes for live trading
5. **Use modern browser** - Chrome 90+, Firefox 88+, Safari 15+

## Next Steps

### Connect to ML Model

When you're ready to integrate signals with your XGBoost models:

1. Backend: Modify `chart_data_service.py` to call ML API
2. Display: Add confidence scores to signals
3. Integrate: Use ML predictions to filter signals

### Real-time Monitoring

Run monitoring in background:

```bash
# Terminal 2: Real-time monitoring
python realtime_monitor.py

# Terminal 3: ML prediction API
python api_multi_model.py
```

### Advanced Indicators

You can add more indicators by modifying the chart:

```javascript
// Add MACD
macdSeries = chart.addLineSeries({ color: '#FF6B00' });

// Add Volume
volumeSeries = chart.addHistogramSeries({ color: '#26a69a' });
```

## System Requirements

- **Browser:** Chrome, Firefox, Safari, Edge (modern versions)
- **Internet:** Stable connection for Binance API
- **Python:** 3.8+ for backend services
- **Memory:** ~200MB for chart + signals
- **CPU:** Minimal (mostly frontend rendering)

## API Endpoints

### GET /api/klines

Fetch K-bar data

```bash
curl "http://localhost:5001/api/klines?symbol=BTCUSDT&timeframe=1h&period=20&stddev=2"
```

Response:
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "candles": [...],
  "bb_upper": [...],
  "bb_middle": [...],
  "bb_lower": [...],
  "stats": {
    "current_price": 42500.25,
    "bb_upper": 42800.15,
    "bb_lower": 41900.45,
    "volatility": 2.15
  }
}
```

### POST /api/analyze

Analyze BB touch

```bash
curl -X POST http://localhost:5001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "timeframe": "1h"}'
```

Response:
```json
{
  "symbol": "BTCUSDT",
  "current_price": 42500.25,
  "bb_upper": 42800.15,
  "bb_lower": 41900.45,
  "touched": true,
  "touch_type": "LOWER",
  "distance_to_band": -0.5,
  "volatility": 2.15
}
```

## Support

For issues or questions:

1. Check browser console (F12)
2. Verify all services are running
3. Test Binance API connectivity
4. Review log files from each service

---

**Your professional trading dashboard is ready to use!**

Enjoy real-time trading signals and analysis.
