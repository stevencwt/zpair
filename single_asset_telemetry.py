#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Asset Telemetry - Hurst Exponent, RSI, and Trend Analysis
================================================================
Provides market data calculations for single-asset directional trading.
Replaces pair regression logic with Hurst regime detection and RSI signals.

Part of Phase 2: Multi-Mode Trading System Enhancement - Data Layer
Zero Deletion Compliance: New module, no existing code modified

FEATURES:
- Hurst Exponent: R/S analysis for regime detection (ranging vs trending)
- RSI: Relative Strength Index for overbought/oversold conditions
- Trend Analysis: Linear regression-based trend detection
- Volatility: Standard deviation of returns
- ATR: Average True Range for volatility measurement
- Comprehensive Analysis: Combined regime and signal detection

REGIME CLASSIFICATION:
- Hurst < 0.45: RANGING (mean-reverting, anti-persistent)
- Hurst > 0.55: TRENDING (persistent, momentum-driven)
- 0.45 <= Hurst <= 0.55: NEUTRAL (random walk)

USAGE:
    from single_asset_telemetry import SingleAssetTelemetry
    
    telemetry = SingleAssetTelemetry(config)
    analysis = telemetry.get_comprehensive_analysis(price_array)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone


class SingleAssetTelemetry:
    """
    Calculate Hurst exponent, RSI, and trend metrics for single asset trading.
    Used for regime detection (trending vs ranging) and directional signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize single asset telemetry calculator
        
        Args:
            config: Configuration dictionary with single_asset_config section
        """
        self.config = config
        sa_config = config.get('single_asset_config', {})
        
        # Regime detection parameters
        regime_config = sa_config.get('regime_detection', {})
        self.hurst_window = regime_config.get('hurst_window', 100)
        self.hurst_trending_threshold = regime_config.get('hurst_trending_threshold', 0.55)
        self.hurst_ranging_threshold = regime_config.get('hurst_ranging_threshold', 0.45)
        
        # Trend analysis parameters
        trend_config = sa_config.get('trend_analysis', {})
        self.trend_window = trend_config.get('trend_window', 50)
        self.trend_method = trend_config.get('trend_method', 'linear_regression')
        self.min_trend_strength = trend_config.get('min_trend_strength', 0.3)
        
        # Indicator parameters
        indicator_config = sa_config.get('indicators', {})
        self.rsi_period = indicator_config.get('rsi_period', 14)
        self.volatility_window = indicator_config.get('volatility_window', 20)
        self.atr_period = indicator_config.get('atr_period', 14)
    
    def calculate_hurst_exponent(self, prices: np.ndarray) -> Tuple[float, str]:
        """
        Calculate Hurst exponent using R/S (Rescaled Range) analysis.
        
        Hurst Interpretation:
        - H < 0.45: Mean-reverting (ranging market, anti-persistent)
        - H = 0.50: Random walk (Brownian motion, efficient market)
        - H > 0.55: Trending (persistent market, momentum-driven)
        
        The R/S method:
        1. Calculate log returns from prices
        2. For different lag sizes, compute rescaled range (R/S)
        3. R/S ratio grows with lag as: E[R/S] ~ lag^H
        4. H is the slope of log(R/S) vs log(lag)
        
        Args:
            prices: Array of price data (close prices)
            
        Returns:
            (hurst_value, regime_classification)
            
        Examples:
            >>> prices = np.array([100, 101, 99, 102, 98, ...])
            >>> hurst, regime = calculate_hurst_exponent(prices)
            >>> print(f"Hurst: {hurst:.3f}, Regime: {regime}")
            Hurst: 0.420, Regime: ranging
        """
        try:
            if len(prices) < self.hurst_window:
                return 0.5, "insufficient_data"
            
            # Use last N prices
            series = prices[-self.hurst_window:]
            
            # Calculate log returns
            log_returns = np.diff(np.log(series))
            
            if len(log_returns) < 10:
                return 0.5, "insufficient_data"
            
            # R/S Analysis: Calculate rescaled range for different lag sizes
            lags = self._get_hurst_lags(len(log_returns))
            rs_values = []
            
            for lag in lags:
                if lag < 2:
                    continue
                    
                # Split into chunks of size 'lag'
                n_chunks = len(log_returns) // lag
                if n_chunks < 1:
                    continue
                
                rs_chunk = []
                for i in range(n_chunks):
                    chunk = log_returns[i*lag:(i+1)*lag]
                    if len(chunk) < 2:
                        continue
                    
                    # Mean-adjusted series
                    mean_adjusted = chunk - np.mean(chunk)
                    
                    # Cumulative deviation from mean
                    cumdev = np.cumsum(mean_adjusted)
                    
                    # Range: max - min of cumulative deviation
                    R = np.max(cumdev) - np.min(cumdev)
                    
                    # Standard deviation
                    S = np.std(chunk, ddof=1)
                    
                    # Rescaled range (R/S ratio)
                    if S > 0:
                        rs_chunk.append(R / S)
                
                # Average R/S for this lag size
                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))
            
            if len(rs_values) < 3:
                return 0.5, "calculation_failed"
            
            # Linear regression on log-log plot: log(R/S) vs log(lag)
            # The slope is the Hurst exponent
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Remove any inf/nan values
            valid_idx = np.isfinite(log_lags) & np.isfinite(log_rs)
            if np.sum(valid_idx) < 3:
                return 0.5, "calculation_failed"
            
            log_lags = log_lags[valid_idx]
            log_rs = log_rs[valid_idx]
            
            # Calculate Hurst exponent (slope of log-log regression)
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            
            # Classify regime based on Hurst value
            if hurst < self.hurst_ranging_threshold:
                regime = "ranging"
            elif hurst > self.hurst_trending_threshold:
                regime = "trending"
            else:
                regime = "neutral"
            
            return float(hurst), regime
            
        except Exception as e:
            print(f"[HURST] Calculation error: {e}")
            return 0.5, "error"
    
    def _get_hurst_lags(self, n: int) -> List[int]:
        """
        Generate appropriate lag values for R/S analysis.
        
        Uses logarithmically spaced lags from 2 to n/4 to capture
        different time scales of market behavior.
        
        Args:
            n: Length of the time series
            
        Returns:
            List of lag values to test
        """
        # Use lags from 2 to n/4, logarithmically spaced
        max_lag = max(2, n // 4)
        if max_lag < 3:
            return [2]
        
        # Generate logarithmic spacing
        lags = np.logspace(np.log10(2), np.log10(max_lag), num=min(20, max_lag-1))
        return np.unique(np.round(lags).astype(int)).tolist()
    
    def calculate_rsi(self, prices: np.ndarray) -> float:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI measures the speed and magnitude of price changes:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Interpretation:
        - RSI > 70: Overbought (potential reversal down)
        - RSI < 30: Oversold (potential reversal up)
        - RSI = 50: Neutral
        
        Args:
            prices: Array of price data
            
        Returns:
            RSI value (0-100)
            
        Examples:
            >>> prices = np.array([100, 102, 101, 105, 103, 107, ...])
            >>> rsi = calculate_rsi(prices)
            >>> print(f"RSI: {rsi:.2f}")
            RSI: 65.23
        """
        try:
            if len(prices) < self.rsi_period + 1:
                return 50.0
            
            # Calculate price changes
            deltas = np.diff(prices[-self.rsi_period-1:])
            
            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calculate average gains and losses
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                return 100.0
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            return float(rsi)
            
        except Exception as e:
            print(f"[RSI] Calculation error: {e}")
            return 50.0
    
    def analyze_trend(self, prices: np.ndarray) -> Dict[str, Any]:
        """
        Analyze price trend using linear regression.
        
        Fits a linear regression line to recent prices and determines:
        - Direction: up, down, or sideways
        - Strength: R-squared value (0-1)
        - Slope: Rate of change
        - Angle: Trend steepness in degrees
        
        Args:
            prices: Array of price data
            
        Returns:
            Dictionary with trend analysis:
            {
                "direction": "up" | "down" | "sideways",
                "strength": 0.0 to 1.0 (R-squared),
                "slope": actual slope value,
                "angle_degrees": trend angle,
                "is_uptrend": boolean,
                "is_downtrend": boolean
            }
            
        Examples:
            >>> prices = np.array([100, 102, 105, 108, 112, ...])
            >>> trend = analyze_trend(prices)
            >>> print(f"Direction: {trend['direction']}, Strength: {trend['strength']:.2f}")
            Direction: up, Strength: 0.95
        """
        try:
            if len(prices) < self.trend_window:
                return {
                    "direction": "unknown",
                    "strength": 0.0,
                    "slope": 0.0,
                    "angle_degrees": 0.0,
                    "is_uptrend": False,
                    "is_downtrend": False
                }
            
            # Use last N prices
            y = prices[-self.trend_window:]
            x = np.arange(len(y))
            
            # Linear regression: y = slope * x + intercept
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            
            # Calculate R-squared (goodness of fit)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
            ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Calculate angle in degrees
            angle_rad = np.arctan(slope)
            angle_deg = np.degrees(angle_rad)
            
            # Determine direction based on R-squared and slope
            if r_squared < self.min_trend_strength:
                # Weak trend = sideways
                direction = "sideways"
                is_uptrend = False
                is_downtrend = False
            elif slope > 0:
                direction = "up"
                is_uptrend = True
                is_downtrend = False
            else:
                direction = "down"
                is_uptrend = False
                is_downtrend = True
            
            return {
                "direction": direction,
                "strength": float(r_squared),
                "slope": float(slope),
                "angle_degrees": float(angle_deg),
                "is_uptrend": is_uptrend,
                "is_downtrend": is_downtrend
            }
            
        except Exception as e:
            print(f"[TREND] Analysis error: {e}")
            return {
                "direction": "error",
                "strength": 0.0,
                "slope": 0.0,
                "angle_degrees": 0.0,
                "is_uptrend": False,
                "is_downtrend": False
            }
    
    def calculate_volatility(self, prices: np.ndarray) -> float:
        """
        Calculate price volatility (annualized standard deviation of returns).
        
        Volatility measures the dispersion of returns:
        - Higher volatility = more risky, larger price swings
        - Lower volatility = more stable, smaller price swings
        
        Args:
            prices: Array of price data
            
        Returns:
            Volatility value (annualized percentage)
            
        Examples:
            >>> prices = np.array([100, 102, 98, 105, 95, ...])
            >>> vol = calculate_volatility(prices)
            >>> print(f"Volatility: {vol:.2f}%")
            Volatility: 45.23%
        """
        try:
            if len(prices) < self.volatility_window + 1:
                return 0.0
            
            # Calculate log returns
            log_returns = np.diff(np.log(prices[-self.volatility_window-1:]))
            
            # Standard deviation of returns
            volatility = np.std(log_returns) * np.sqrt(252)  # Annualize (252 trading days)
            
            return float(volatility * 100)  # Convert to percentage
            
        except Exception as e:
            print(f"[VOLATILITY] Calculation error: {e}")
            return 0.0
    
    def calculate_atr(self, high_prices: np.ndarray, low_prices: np.ndarray, 
                     close_prices: np.ndarray) -> float:
        """
        Calculate Average True Range (ATR).
        
        ATR measures market volatility using high, low, and close prices:
        True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = average of True Range over N periods
        
        Args:
            high_prices: Array of high prices
            low_prices: Array of low prices
            close_prices: Array of close prices
            
        Returns:
            ATR value
            
        Examples:
            >>> atr = calculate_atr(highs, lows, closes)
            >>> print(f"ATR: {atr:.2f}")
            ATR: 12.34
        """
        try:
            if len(close_prices) < self.atr_period + 1:
                return 0.0
            
            # Calculate True Range for each period
            high_low = high_prices[-self.atr_period:] - low_prices[-self.atr_period:]
            high_close = np.abs(high_prices[-self.atr_period:] - close_prices[-self.atr_period-1:-1])
            low_close = np.abs(low_prices[-self.atr_period:] - close_prices[-self.atr_period-1:-1])
            
            # True Range is the maximum of the three
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # Average True Range
            atr = np.mean(true_range)
            
            return float(atr)
            
        except Exception as e:
            print(f"[ATR] Calculation error: {e}")
            return 0.0
    
    def get_comprehensive_analysis(self, prices: np.ndarray,
                                   high_prices: Optional[np.ndarray] = None,
                                   low_prices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get comprehensive single-asset market analysis.
        
        Combines all indicators to provide a complete picture:
        - Hurst exponent for regime detection
        - RSI for overbought/oversold conditions
        - Trend analysis for direction and strength
        - Volatility for risk assessment
        - ATR if high/low prices available
        - Trading regime recommendation
        
        Args:
            prices: Close price array (required)
            high_prices: Optional high price array for ATR
            low_prices: Optional low price array for ATR
            
        Returns:
            Complete analysis dictionary with all metrics
            
        Examples:
            >>> analysis = telemetry.get_comprehensive_analysis(close_prices)
            >>> print(f"Regime: {analysis['hurst']['regime']}")
            >>> print(f"RSI: {analysis['rsi']['value']:.2f}")
            >>> print(f"Trend: {analysis['trend']['direction']}")
            Regime: ranging
            RSI: 32.45
            Trend: down
        """
        # Calculate all metrics
        hurst, regime = self.calculate_hurst_exponent(prices)
        rsi = self.calculate_rsi(prices)
        trend = self.analyze_trend(prices)
        volatility = self.calculate_volatility(prices)
        
        # Calculate ATR if high/low available
        atr = 0.0
        if high_prices is not None and low_prices is not None:
            atr = self.calculate_atr(high_prices, low_prices, prices)
        
        # Current price
        current_price = float(prices[-1]) if len(prices) > 0 else 0.0
        
        # Determine trading regime and strategy suggestion
        trading_regime = self._determine_trading_regime(hurst, regime, trend, rsi)
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "current_price": current_price,
            "hurst": {
                "value": hurst,
                "regime": regime,
                "window": self.hurst_window,
                "interpretation": self._get_hurst_interpretation(hurst)
            },
            "rsi": {
                "value": rsi,
                "period": self.rsi_period,
                "oversold": rsi < 30,
                "overbought": rsi > 70,
                "interpretation": self._get_rsi_interpretation(rsi)
            },
            "trend": trend,
            "volatility": {
                "value": volatility,
                "window": self.volatility_window,
                "interpretation": self._get_volatility_interpretation(volatility)
            },
            "atr": {
                "value": atr,
                "period": self.atr_period
            },
            "trading_regime": trading_regime
        }
    
    def _determine_trading_regime(self, hurst: float, hurst_regime: str,
                                  trend: Dict[str, Any], rsi: float) -> Dict[str, Any]:
        """
        Determine overall trading regime and suggested strategy.
        
        Combines Hurst, trend, and RSI to provide actionable recommendations.
        
        Returns:
            Trading regime classification with strategy suggestion
        """
        regime_type = "unknown"
        strategy_suggestion = "wait"
        confidence = 0.0
        reasoning = []
        
        # Ranging market (mean reversion strategy)
        if hurst_regime == "ranging":
            regime_type = "mean_reversion"
            reasoning.append(f"Hurst={hurst:.3f} indicates mean-reverting behavior")
            
            if rsi < 30:
                strategy_suggestion = "buy_oversold"
                confidence = 0.8
                reasoning.append("RSI oversold - potential bounce")
            elif rsi > 70:
                strategy_suggestion = "sell_overbought"
                confidence = 0.8
                reasoning.append("RSI overbought - potential pullback")
            else:
                strategy_suggestion = "wait_for_extreme"
                confidence = 0.3
                reasoning.append("RSI neutral - wait for clearer signal")
        
        # Trending market (momentum strategy)
        elif hurst_regime == "trending":
            regime_type = "trending"
            reasoning.append(f"Hurst={hurst:.3f} indicates trending behavior")
            
            if trend["is_uptrend"] and trend["strength"] > self.min_trend_strength:
                if rsi > 50:
                    strategy_suggestion = "buy_uptrend"
                    confidence = 0.7
                    reasoning.append(f"Uptrend (strength={trend['strength']:.2f}) with RSI confirmation")
                else:
                    strategy_suggestion = "wait_for_confirmation"
                    confidence = 0.4
                    reasoning.append("Uptrend but RSI not confirming")
                    
            elif trend["is_downtrend"] and trend["strength"] > self.min_trend_strength:
                if rsi < 50:
                    strategy_suggestion = "sell_downtrend"
                    confidence = 0.7
                    reasoning.append(f"Downtrend (strength={trend['strength']:.2f}) with RSI confirmation")
                else:
                    strategy_suggestion = "wait_for_confirmation"
                    confidence = 0.4
                    reasoning.append("Downtrend but RSI not confirming")
            else:
                strategy_suggestion = "wait_for_clarity"
                confidence = 0.2
                reasoning.append("Trending regime but unclear direction")
        
        # Neutral market (no clear pattern)
        else:
            regime_type = "neutral"
            strategy_suggestion = "no_clear_signal"
            confidence = 0.1
            reasoning.append(f"Hurst={hurst:.3f} indicates random walk")
        
        return {
            "regime_type": regime_type,
            "strategy_suggestion": strategy_suggestion,
            "confidence": confidence,
            "hurst_classification": hurst_regime,
            "trend_direction": trend.get("direction", "unknown"),
            "trend_strength": trend.get("strength", 0.0),
            "reasoning": reasoning
        }
    
    def _get_hurst_interpretation(self, hurst: float) -> str:
        """Get human-readable interpretation of Hurst value"""
        if hurst < 0.45:
            return "Mean-reverting (ranging market)"
        elif hurst > 0.55:
            return "Trending (persistent market)"
        else:
            return "Random walk (neutral)"
    
    def _get_rsi_interpretation(self, rsi: float) -> str:
        """Get human-readable interpretation of RSI value"""
        if rsi < 30:
            return "Oversold (potential bounce)"
        elif rsi > 70:
            return "Overbought (potential pullback)"
        elif 45 <= rsi <= 55:
            return "Neutral"
        else:
            return "Normal range"
    
    def _get_volatility_interpretation(self, volatility: float) -> str:
        """Get human-readable interpretation of volatility"""
        if volatility < 20:
            return "Low volatility"
        elif volatility < 40:
            return "Moderate volatility"
        elif volatility < 60:
            return "High volatility"
        else:
            return "Extreme volatility"