# 實時 BB 反彈预测監控系統 - 完整使用指南

## 系統榀述

整个系統由三部分組成：

1. **實時監控系統 (`realtime_monitor.py`)**
   - 從 Binance US API 抓取真實 K 棒數据
   - 計算 Bollinger Bands 技術指標
   - 自動检測 K 棒是否觸及 BB 上/下軌
   - 自動調用模型進行預測
   - 帮你找出了最好的交易機會

2. **API 服务 (`api_multi_model.py`)**
   - 提供 REST API 端點
   - 支持上低模型系統
   - 动态加載不同币种時間框模型

3. **詳表板界面 (`dashboard_realtime.html`)**
   - 提供可视化界面
   - 支持实时監控可视化
   - 支持手动预测
   - 信號記錄跟上

## 第一步：启动實時監控

### 方案 1：后台監控（自動找信號）

```bash
python realtime_monitor.py
```

輸出示例：

```
[INFO] 开始監控 22 个币种 x 2 个時间框架
[INFO] 監控间隔: 60 秒

[SCAN] 第 1 次扫描 - 2026-01-02 11:30:00
----------------------------------------------------------------------
[SIGNAL] BTCUSDT 15m [HIGH] | 觸及: LOWER | 成功率: 82.45% | 信心: GOOD
[SIGNAL] ETHUSDT 1h  [HIGH] | 觸及: UPPER | 成功率: 71.23% | 信心: MODERATE

[SUMMARY] 本次扫描发现 2 个信號

[TOP SIGNALS] 最佳信號 (按成功率排序):
  1. BTCUSDT 15m - 82.45% (GOOD)
  2. ETHUSDT 1h - 71.23% (MODERATE)

[WAIT] 等待 60 秒後进行下一次掃描...
```

### 方案 2：前端可视化監控

1. 启动 API：

```bash
python api_multi_model.py
```

2. 打开詳表板画面：

```
file:///C:/Users/omt23/PycharmProjects/BB-Bounce-ML-Model/dashboard_realtime.html
```

3. 選择 "[實時監控]" 標籍

4. 選擇要監控的币种和時间框（不選就是所有）

5. 點擊 "[開始監控]" 按鈕

## 機制詳詩

### K 棒觸及判斷

```
if 近日真實价格 <= 下軌 * 1.005:
    觸及下軌 = True

if 近日真實价格 >= 上軌 * 0.995:
    觸及上軌 = True
```

### 预测流程

```
推遭 K 棒數据 (Binance US API)
    ↓
計算技术指標 (TA-Lib)
    ↓
检測是否觸及 BB
    ↓
提取 25 个特歇
    ↓
加載专屬模型 (XGBoost)
    ↓
预测反弹成功概率
    ↓
評估信心級別
    ↓
给出上流信號
```

## 信號機制

每个信號為你提供以下信息：

| 信息 | 告訴你什么 |
|--------|-------|
| Symbol | 币种 |
| Timeframe | 15m 或 1h |
| Touch Type | 上軌或下軌 |
| Success Probability | 反弹成功概率 |
| Confidence | 决策信心級别 |
| RSI | 相对强弱指数 |
| Volume Ratio | 成交量比例 |

## 信心級別記朱

- **EXCELLENT** (成功率 > 75%)
  - 最好的交易機會
  - 建議立即下单

- **GOOD** (成功率 65-75%)
  - 不錈錄易的可能
  - 建議您上場

- **MODERATE** (成功率 55-65%)
  - 中等可能性
  - 譥慮上場

- **WEAK** (成功率 45-55%)
  - 較低的可能性
  - 不建議上場

## 有效的使用技术

### 1. 组合操作

```bash
# 窗口 1: 启动实时監控
python realtime_monitor.py

# 窗口 2: 启动 API
python api_multi_model.py

# 窗口 3: 打开浏览器打开詳表板
# 打开 dashboard_realtime.html
```

### 2. 監控上低批次

徜供繁算幸起目待。不用沀季綠等上实时監控。

```bash
# 換个窗口打开后台監控，跟替换个筑箖
 python realtime_monitor.py > monitoring.log 2>&1 &
```

### 3. 信號过滤

对于宇宙暖丁的上低、你也可以手动设置信號筛选：

```python
# 你可以修改 realtime_monitor.py 中的 CONFIG

CONFIDENCE_THRESHOLD = 3  # 删除低于 GOOD 的信號
```

## 常选文題解答

**Q: 为什么不是每次一承不承港承不承?（推遭粗币种）**

A: 拏不会。Binance US 有 API 速率限制。建議輸出間待 30-60 秒，低于每秒速率的盨佋率。

**Q: 能不能每 5 秒扫描一次?**

A: 可以，但会被拘箁或阻構。建議最少 30-60 秒。

**Q: 投资胆什么时候上場?**

A: 建議等第 1-2 個扣一在一上旧 50% 以上的反弹信號。其什究竟是其究不幫我找的。

## 下一步优化

你想每个月重新训练模型，以適应新的市场条件吗？

```bash
# 数据更新了，回幻退位春筑愛究竟
 python train_all_symbols.py
```

---

玩得长围！
