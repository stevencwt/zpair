#!/usr/bin/env python3
import numpy as np
from single_asset_telemetry import SingleAssetTelemetry

# Test configuration
config = {
    "single_asset_config": {
        "regime_detection": {
            "hurst_window": 100,
            "hurst_trending_threshold": 0.55,
            "hurst_ranging_threshold": 0.45
        },
        "trend_analysis": {
            "trend_window": 50,
            "min_trend_strength": 0.3
        },
        "indicators": {
            "rsi_period": 14,
            "volatility_window": 20,
            "atr_period": 14
        }
    }
}

# Create telemetry instance
telemetry = SingleAssetTelemetry(config)

# Generate sample price data (trending up)
prices = np.cumsum(np.random.randn(200) * 2 + 0.5) + 100

# Test comprehensive analysis
analysis = telemetry.get_comprehensive_analysis(prices)

print("="*70)
print("SINGLE ASSET TELEMETRY TEST")
print("="*70)
print(f"\nCurrent Price: ${analysis['current_price']:.2f}")
print(f"\nHurst Exponent: {analysis['hurst']['value']:.4f}")
print(f"Regime: {analysis['hurst']['regime']}")
print(f"Interpretation: {analysis['hurst']['interpretation']}")
print(f"\nRSI: {analysis['rsi']['value']:.2f}")
print(f"Interpretation: {analysis['rsi']['interpretation']}")
print(f"\nTrend Direction: {analysis['trend']['direction']}")
print(f"Trend Strength: {analysis['trend']['strength']:.3f}")
print(f"\nVolatility: {analysis['volatility']['value']:.2f}%")
print(f"Interpretation: {analysis['volatility']['interpretation']}")
print(f"\nTrading Regime: {analysis['trading_regime']['regime_type']}")
print(f"Strategy: {analysis['trading_regime']['strategy_suggestion']}")
print(f"Confidence: {analysis['trading_regime']['confidence']:.2f}")
print(f"Reasoning:")
for reason in analysis['trading_regime']['reasoning']:
    print(f"  - {reason}")
print("="*70)