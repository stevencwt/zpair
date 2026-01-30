# 📊 Co-integration Testing Tool - Phase 1 (CLI)

Command-line tool for testing pair co-integration using statistical methods (Engle-Granger, ADF test).

---

## 🚀 QUICK START

### 1. Install Dependencies

```bash
pip install statsmodels numpy --break-system-packages
```

### 2. Configure Pairs to Test

Edit `config_cointegrate.json`:

```json
{
  "timeframe": "5m",
  "lookback_bars": 1000,
  "pairs_to_test": [
    {"asset1": "SOL", "asset2": "MELANIA"},
    {"asset1": "SOL", "asset2": "JUP"}
  ]
}
```

### 3. Run Tests

```bash
python run_cointegration_test.py
```

---

## 📋 FILES CREATED

| File | Purpose | Lines |
|------|---------|-------|
| **`cointegration_tester.py`** | Core testing logic (ADF, half-life) | ~350 |
| **`pair_validator.py`** | Data loading & orchestration | ~280 |
| **`run_cointegration_test.py`** | CLI entry point | ~250 |
| **`config_cointegrate.json`** | Configuration file | ~30 |

---

## 🎯 USAGE EXAMPLES

### Basic Usage
```bash
python run_cointegration_test.py
```

### Custom Config File
```bash
python run_cointegration_test.py --config my_pairs.json
```

### Override Parameters
```bash
# Use 1m timeframe with 500 bars
python run_cointegration_test.py --timeframe 1m --bars 500

# Use 10% significance level
python run_cointegration_test.py --significance 0.10
```

---

## 📊 EXPECTED OUTPUT

```
══════════════════════════════════════════════════════════════════
║          CO-INTEGRATION TEST RESULTS                           ║
╠════════════════════════════════════════════════════════════════╣
║  Testing 4 pairs using 5m timeframe (1000 bars)               ║
╚════════════════════════════════════════════════════════════════╝

[1/4] Testing SOL/MELANIA...
✅ COINTEGRATED - GOOD for trading
   P-value:      0.0234 (significant at 5% level)
   ADF Stat:     -3.245
   Half-life:    18.5 bars (92.5 minutes)
   Hedge Ratio:  175.34
   Samples:      1000 bars

[2/4] Testing SOL/JUP...
✅ COINTEGRATED - EXCELLENT for trading
   P-value:      0.0089 (significant at 1% level)
   ADF Stat:     -4.123
   Half-life:    12.3 bars (61.5 minutes)
   Hedge Ratio:  0.0234
   Samples:      1000 bars

[3/4] Testing ETH/BTC...
❌ NOT COINTEGRATED
   P-value:      0.3456 (not significant)
   ADF Stat:     -1.234
   Half-life:    ∞ (no mean reversion detected)
   Hedge Ratio:  15.23
   Samples:      1000 bars

[4/4] Testing SOL/BONK...
✅ COINTEGRATED - MODERATE for trading
   P-value:      0.0456 (significant at 5% level)
   ADF Stat:     -2.845
   Half-life:    45.2 bars (226.0 minutes)
   Hedge Ratio:  0.0012
   Samples:      1000 bars

══════════════════════════════════════════════════════════════════
SUMMARY: 3/4 pairs suitable for trading
  • 1 EXCELLENT pairs (fast mean reversion)
  • 1 GOOD pairs (moderate mean reversion)
══════════════════════════════════════════════════════════════════

✅ Results exported to: ./cointegration_results.json
✅ Testing completed at 2026-01-29 12:34:56
```

---

## 📈 RESULT INTERPRETATION

### Recommendation Levels:

| Rating | P-value | Half-life | Meaning |
|--------|---------|-----------|---------|
| **EXCELLENT** | < 0.01 | < 30 min | Very fast mean reversion, ideal |
| **GOOD** | < 0.05 | < 2 hours | Suitable for intraday trading |
| **MODERATE** | < 0.05 | < 6 hours | Usable but slower reversion |
| **WEAK** | < 0.05 | > 6 hours | Marginal, requires caution |
| **NOT_SUITABLE** | ≥ 0.05 | Any | Not cointegrated |

### Key Metrics:

- **P-value**: Lower = stronger cointegration (target: < 0.05)
- **ADF Statistic**: More negative = more stationary (target: < -2.86)
- **Half-life**: Time for spread to revert halfway (shorter = better)
- **Hedge Ratio**: Optimal position ratio (beta from OLS regression)

---

## 🔧 CONFIGURATION PARAMETERS

### `config_cointegrate.json`:

```json
{
  "timeframe": "5m",              // Trading timeframe
  "lookback_bars": 1000,          // Number of bars to analyze
  "significance_level": 0.05,     // P-value threshold (5%)
  "min_samples": 200,             // Minimum bars required
  "pairs_to_test": [...],         // List of pairs
  "output": {
    "terminal": true,             // Display in terminal
    "save_json": true,            // Export to JSON
    "json_filename": "..."        // Output filename
  }
}
```

### Timeframe Selection:

- **5m** (Recommended): Matches V14s HTF bias strategy
- **1m**: More data points, but noisier
- **15m**: Longer-term view, fewer data points

### Lookback Bars:

- **500 bars**: ~1.7 days (minimum for quick test)
- **1000 bars**: ~3.5 days (recommended)
- **2000 bars**: ~7 days (robust test)

---

## 🗂️ OUTPUT FILES

### JSON Export (`cointegration_results.json`):

```json
{
  "timestamp": "2026-01-29T12:34:56",
  "timeframe": "5m",
  "lookback_bars": 1000,
  "pairs_tested": 4,
  "results": [
    {
      "asset1": "SOL",
      "asset2": "MELANIA",
      "pair": "SOL/MELANIA",
      "is_cointegrated": true,
      "p_value": 0.0234,
      "adf_statistic": -3.245,
      "half_life_bars": 18.5,
      "half_life_minutes": 92.5,
      "hedge_ratio": 175.34,
      "recommendation": "GOOD",
      "samples_used": 1000
    }
  ]
}
```

---

## 🔍 DATA LOADING

The validator attempts to load data from:

1. **Aggregated data directories** (e.g., `aggregated_SOL_MELANIA_5m1m/`)
2. **Separate asset files** (e.g., `SOL_5m.json`, `MELANIA_5m.json`)
3. **Synthetic data** (fallback for testing when no real data available)

---

## 🚨 TROUBLESHOOTING

### "No module named 'statsmodels'"
```bash
pip install statsmodels --break-system-packages
```

### "Config file not found"
```bash
# Make sure config_cointegrate.json exists in current directory
ls -la config_cointegrate.json
```

### "Could not load data"
- Verify data files exist in expected locations
- Check asset symbols match your data files
- Tool will use synthetic data as fallback for testing

### "Insufficient data"
- Increase `lookback_bars` in config
- Ensure data source has enough historical data
- Minimum 200 bars required

---

## ✅ NEXT STEPS: PHASE 2

Phase 2 will add:
- **HTML Dashboard** for visual results
- **Interactive charts** (spread plot, residuals)
- **Historical tracking** of cointegration over time
- **Alert system** when cointegration weakens

---

## 📝 INTEGRATION WITH V14S

To integrate with your V14s trading system:

1. **Pre-trading validation**: Run before starting trading session
2. **Weekly re-validation**: Check if pairs still cointegrated
3. **Pair selection**: Use results to choose which pairs to trade
4. **Risk management**: Avoid pairs with weak cointegration

---

## 🔒 ZERO DELETION COMPLIANCE

✅ All files are NEW additions to your codebase
✅ No existing code modified or deleted
✅ Standalone tool - does not affect V14s operation

---

**Created:** January 29, 2026  
**Version:** Phase 1 (CLI)  
**Status:** Ready for Testing  
