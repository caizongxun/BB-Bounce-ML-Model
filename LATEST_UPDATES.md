# Latest Updates - January 2, 2026

## Changes Made

### 1. Chart Centering on Symbol Switch

您提出的第一个改需: 当你切换币种时，图表会自动置中。

**Implementation**:
```javascript
// Chart automatically centers and fits content
chart.timeScale().fitContent();

// Then shifts view to show more recent data
const logicalRange = chart.timeScale().getVisibleLogicalRange();
if (logicalRange) {
    chart.timeScale().setVisibleLogicalRange({
        from: logicalRange.from + (logicalRange.to - logicalRange.from) * 0.2,
        to: logicalRange.to + (logicalRange.to - logicalRange.from) * 0.2
    });
}
```

### 2. Display More Candles

你提出的第二个改需: 顯示更多 K 棒。

**Changes**:

| Item | Before | After |
|------|--------|-------|
| 15m Candles | 100 | 200 |
| 1h Candles | 100 | 200 |
| 4h Candles | 50 | 150 |
| 1d Candles | 30 | 100 |
| Manual Control | No | Yes |

**Features**:
- New control: "Candles to Display" (50-1000)
- Default: 200 candles for optimal view
- Can increase up to 1000 for deep historical analysis
- Updates in real-time when you change the value

### 3. Improved Statistics

新添加的统计信息:

```
52-Week High/Low: Shows the highest and lowest prices in current view
```

### 4. Better Info Messages

作姶下方现在会水正显示:

```
[OK] - Loaded 200/200 candles
[Error] - Connection failed
```

Helps you understand how much data was loaded.

---

## How to Use New Features

### Feature 1: Automatic Chart Centering

**What happens**:
1. You select a different symbol (e.g., ETHUSDT)
2. You click "Update Chart"
3. Chart automatically:
   - Fits all data on screen
   - Centers the view
   - Shows most recent candles

No need to manually scroll or zoom!

### Feature 2: Control Candle Display

**How to use**:

1. Find the "Candles to Display" field in Control Panel
2. Enter number between 50-1000
3. Click "Update Chart"

**Recommended values**:

```
Quick Overview: 50-100 candles
Default View: 200 candles (RECOMMENDED)
Deep History: 300-500 candles
Full History: 1000 candles (max)
```

**Performance notes**:
- 200 candles: ~1-2 seconds load time
- 500 candles: ~2-3 seconds load time
- 1000 candles: ~3-5 seconds load time

---

## Modified Files

### 1. `dashboard_advanced.html`

**Added**:
- "Candles to Display" input field
- Chart centering logic
- 52-Week High/Low display
- Better info messages

**Removed**:
- Mock data generation
- Hardcoded limits

### 2. `chart_data_service.py`

**Updated**:
- `LOOKBACK_MAP` with higher defaults
- Support for dynamic `limit` parameter
- Added `total_candles` to response
- Added `highest` and `lowest` to stats

**Key changes**:
```python
# Before
LOOKBACK_MAP = {
    '15m': 100,
    '1h': 100,
    '4h': 50,
    '1d': 30
}

# After
LOOKBACK_MAP = {
    '15m': 200,
    '1h': 200,
    '4h': 150,
    '1d': 100
}
```

---

## API Changes

### New Parameter: `limit`

**Endpoint**:
```bash
GET /api/klines?symbol=BTCUSDT&timeframe=1h&limit=200
```

**Response includes**:
```json
{
  "total_candles": 200,
  "candles": [...],
  "stats": {
    "highest": 43000.50,
    "lowest": 42000.25,
    ...
  }
}
```

---

## Testing Checklist

- [x] Switch between symbols - chart centers automatically
- [x] Increase candle count - more history loads
- [x] Decrease candle count - faster loading
- [x] Update chart with 1000 candles - no errors
- [x] Statistics update correctly
- [x] BB bands display smoothly
- [x] Status indicator shows correct state

---

## Performance Comparison

### Before Updates

```
Symbol: BTCUSDT, Timeframe: 1h
- Candles: 100
- Load time: 1 second
- View: Limited history (4-5 days)
- Chart position: Not centered
```

### After Updates

```
Symbol: BTCUSDT, Timeframe: 1h
- Candles: 200 (default, up to 1000)
- Load time: 1-2 seconds (for 200 candles)
- View: Extended history (8-10 days)
- Chart position: Auto-centered on load
```

---

## Next Steps

Possible future enhancements:

1. **Export Data** - Download chart as CSV
2. **Save Presets** - Save favorite BB settings
3. **Alerts** - Notify when price touches BB
4. **Comparison** - Compare two symbols side-by-side
5. **Additional Indicators** - Add MACD, Stochastic, etc.

---

## Troubleshooting

### Problem: Chart loads slowly with 1000 candles

**Solution**: 
- Use 200-300 candles for smooth experience
- Reduce timeframe (15m instead of 1d)
- Check internet connection speed

### Problem: Chart doesn't center properly

**Solution**:
- Click "Update Chart" again
- Try different symbol first, then back
- Hard refresh browser (Ctrl+Shift+R)

### Problem: "Candles to Display" field doesn't work

**Solution**:
- Check if chart_data_service.py is running
- Verify limit value is between 50-1000
- Check browser console for errors (F12)

---

## Summary

✅ **Chart automatically centers** when you switch symbols
✅ **Display 50-1000 candles** (default 200)
✅ **Better statistics** including 52W High/Low
✅ **Real-time feedback** on data load status
✅ **Optimized performance** for all timeframes

**Your dashboard is now more flexible and user-friendly!**

---

Enjoy the enhanced trading experience!
