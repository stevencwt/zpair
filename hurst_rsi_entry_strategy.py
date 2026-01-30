#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hurst-RSI Entry Strategy - Single Asset Directional Trading
===========================================================
Entry strategy for single-asset mode using Hurst regime detection and RSI signals.

Part of Phase 3B: Multi-Mode Trading System - Strategy Layer
Zero Deletion Compliance: New module, no existing code modified

STRATEGY LOGIC:
- Hurst < 0.45 (Ranging): Use RSI extremes for mean-reversion
  * RSI < 30 → LONG (oversold, expect bounce)
  * RSI > 70 → SHORT (overbought, expect pullback)

- Hurst > 0.55 (Trending): Use trend direction with RSI confirmation
  * Uptrend + RSI > 50 → LONG (momentum trade)
  * Downtrend + RSI < 50 → SHORT (momentum trade)

- Hurst 0.45-0.55 (Neutral): No clear signal, wait

CONFIGURATION:
{
  "single_asset_config": {
    "direction_control": {
      "allowed_directions": "both"  // "long_only", "short_only", "both"
    },
    "entry_thresholds": {
      "ranging_rsi_oversold": 30,
      "ranging_rsi_overbought": 70,
      "trending_rsi_neutral": 50,
      "min_trend_strength": 0.3
    }
  }
}
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone


class HurstRSIEntry:
    """
    Single asset entry strategy using Hurst exponent and RSI.
    Detects market regime (ranging vs trending) and generates appropriate signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with configuration
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        sa_config = config.get('single_asset_config', {})
        
        # Direction control
        direction_config = sa_config.get('direction_control', {})
        self.allowed_directions = direction_config.get('allowed_directions', 'both')
        
        # Entry thresholds
        threshold_config = sa_config.get('entry_thresholds', {})
        self.ranging_rsi_oversold = threshold_config.get('ranging_rsi_oversold', 30)
        self.ranging_rsi_overbought = threshold_config.get('ranging_rsi_overbought', 70)
        self.trending_rsi_neutral = threshold_config.get('trending_rsi_neutral', 50)
        self.min_trend_strength = threshold_config.get('min_trend_strength', 0.3)
        
        # Hurst thresholds
        regime_config = sa_config.get('regime_detection', {})
        self.hurst_ranging_threshold = regime_config.get('hurst_ranging_threshold', 0.45)
        self.hurst_trending_threshold = regime_config.get('hurst_trending_threshold', 0.55)
    
    def analyze_entry(self, marketstate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market conditions and generate entry signal
        
        Args:
            marketstate: Market data including single_asset_analysis
            
        Returns:
            Dictionary with signal decision:
            {
                "signal": "ENTER" | "WAIT",
                "direction": "LONG" | "SHORT" | None,
                "confidence": 0.0-1.0,
                "regime": "ranging" | "trending" | "neutral",
                "reasoning": "explanation"
            }
        """
        # Get single asset analysis from marketstate
        analysis = marketstate.get('single_asset_analysis', {})
        
        if not analysis:
            return {
                "signal": "WAIT",
                "direction": None,
                "confidence": 0.0,
                "regime": "unknown",
                "reasoning": "No single asset analysis available"
            }
        
        # Extract key metrics
        hurst_data = analysis.get('hurst', {})
        hurst_value = hurst_data.get('value', 0.5)
        regime = hurst_data.get('regime', 'neutral')
        
        rsi_data = analysis.get('rsi', {})
        rsi_value = rsi_data.get('value', 50)
        
        trend_data = analysis.get('trend', {})
        trend_direction = trend_data.get('direction', 'sideways')
        trend_strength = trend_data.get('strength', 0.0)
        
        # Determine regime and generate signal
        if regime == "ranging":
            return self._analyze_ranging_entry(hurst_value, rsi_value)
        elif regime == "trending":
            return self._analyze_trending_entry(hurst_value, rsi_value, trend_data)
        else:
            return {
                "signal": "WAIT",
                "direction": None,
                "confidence": 0.2,
                "regime": "neutral",
                "reasoning": f"Neutral regime (H={hurst_value:.3f}), no clear pattern"
            }
    
    def _analyze_ranging_entry(self, hurst: float, rsi: float) -> Dict[str, Any]:
        """
        Analyze entry in ranging market (mean-reversion strategy)
        
        Args:
            hurst: Hurst exponent value
            rsi: RSI value
            
        Returns:
            Entry signal dictionary
        """
        # Check for oversold (potential LONG)
        if rsi < self.ranging_rsi_oversold:
            if self._is_direction_allowed("LONG"):
                confidence = self._calculate_ranging_confidence(rsi, self.ranging_rsi_oversold, "oversold")
                return {
                    "signal": "ENTER",
                    "direction": "LONG",
                    "confidence": confidence,
                    "regime": "ranging",
                    "reasoning": f"Ranging market (H={hurst:.3f}), RSI oversold ({rsi:.1f} < {self.ranging_rsi_oversold}), expect bounce"
                }
            else:
                return {
                    "signal": "WAIT",
                    "direction": None,
                    "confidence": 0.0,
                    "regime": "ranging",
                    "reasoning": f"LONG signal detected but not allowed (mode: {self.allowed_directions})"
                }
        
        # Check for overbought (potential SHORT)
        elif rsi > self.ranging_rsi_overbought:
            if self._is_direction_allowed("SHORT"):
                confidence = self._calculate_ranging_confidence(rsi, self.ranging_rsi_overbought, "overbought")
                return {
                    "signal": "ENTER",
                    "direction": "SHORT",
                    "confidence": confidence,
                    "regime": "ranging",
                    "reasoning": f"Ranging market (H={hurst:.3f}), RSI overbought ({rsi:.1f} > {self.ranging_rsi_overbought}), expect pullback"
                }
            else:
                return {
                    "signal": "WAIT",
                    "direction": None,
                    "confidence": 0.0,
                    "regime": "ranging",
                    "reasoning": f"SHORT signal detected but not allowed (mode: {self.allowed_directions})"
                }
        
        # RSI in neutral zone
        else:
            return {
                "signal": "WAIT",
                "direction": None,
                "confidence": 0.3,
                "regime": "ranging",
                "reasoning": f"Ranging market but RSI neutral ({rsi:.1f}), waiting for extreme"
            }
    
    def _analyze_trending_entry(self, hurst: float, rsi: float, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze entry in trending market (momentum strategy)
        
        Args:
            hurst: Hurst exponent value
            rsi: RSI value
            trend_data: Trend analysis data
            
        Returns:
            Entry signal dictionary
        """
        trend_direction = trend_data.get('direction', 'sideways')
        trend_strength = trend_data.get('strength', 0.0)
        is_uptrend = trend_data.get('is_uptrend', False)
        is_downtrend = trend_data.get('is_downtrend', False)
        
        # Check trend strength
        if trend_strength < self.min_trend_strength:
            return {
                "signal": "WAIT",
                "direction": None,
                "confidence": 0.2,
                "regime": "trending",
                "reasoning": f"Trending regime (H={hurst:.3f}) but weak trend (strength={trend_strength:.2f} < {self.min_trend_strength})"
            }
        
        # Uptrend with RSI confirmation
        if is_uptrend and rsi > self.trending_rsi_neutral:
            if self._is_direction_allowed("LONG"):
                confidence = self._calculate_trending_confidence(trend_strength, rsi, "up")
                return {
                    "signal": "ENTER",
                    "direction": "LONG",
                    "confidence": confidence,
                    "regime": "trending",
                    "reasoning": f"Strong uptrend (H={hurst:.3f}, strength={trend_strength:.2f}), RSI confirming ({rsi:.1f} > {self.trending_rsi_neutral})"
                }
            else:
                return {
                    "signal": "WAIT",
                    "direction": None,
                    "confidence": 0.0,
                    "regime": "trending",
                    "reasoning": f"LONG signal detected but not allowed (mode: {self.allowed_directions})"
                }
        
        # Downtrend with RSI confirmation
        elif is_downtrend and rsi < self.trending_rsi_neutral:
            if self._is_direction_allowed("SHORT"):
                confidence = self._calculate_trending_confidence(trend_strength, rsi, "down")
                return {
                    "signal": "ENTER",
                    "direction": "SHORT",
                    "confidence": confidence,
                    "regime": "trending",
                    "reasoning": f"Strong downtrend (H={hurst:.3f}, strength={trend_strength:.2f}), RSI confirming ({rsi:.1f} < {self.trending_rsi_neutral})"
                }
            else:
                return {
                    "signal": "WAIT",
                    "direction": None,
                    "confidence": 0.0,
                    "regime": "trending",
                    "reasoning": f"SHORT signal detected but not allowed (mode: {self.allowed_directions})"
                }
        
        # Trend exists but no RSI confirmation
        else:
            return {
                "signal": "WAIT",
                "direction": None,
                "confidence": 0.4,
                "regime": "trending",
                "reasoning": f"Trend detected ({trend_direction}) but no RSI confirmation"
            }
    
    def _calculate_ranging_confidence(self, rsi: float, threshold: float, extreme_type: str) -> float:
        """
        Calculate confidence for ranging market entry
        
        Args:
            rsi: Current RSI value
            threshold: RSI threshold (30 for oversold, 70 for overbought)
            extreme_type: "oversold" or "overbought"
            
        Returns:
            Confidence value 0.0-1.0
        """
        if extreme_type == "oversold":
            # More oversold = higher confidence
            # RSI 30 → 0.6, RSI 20 → 0.8, RSI 10 → 0.9
            distance = threshold - rsi
            confidence = 0.6 + (distance / 30) * 0.3
        else:  # overbought
            # More overbought = higher confidence
            # RSI 70 → 0.6, RSI 80 → 0.8, RSI 90 → 0.9
            distance = rsi - threshold
            confidence = 0.6 + (distance / 30) * 0.3
        
        return min(0.9, max(0.5, confidence))
    
    def _calculate_trending_confidence(self, trend_strength: float, rsi: float, direction: str) -> float:
        """
        Calculate confidence for trending market entry
        
        Args:
            trend_strength: Trend strength (0-1, typically R-squared)
            rsi: Current RSI value
            direction: "up" or "down"
            
        Returns:
            Confidence value 0.0-1.0
        """
        # Base confidence from trend strength
        base_confidence = 0.5 + (trend_strength * 0.3)
        
        # Boost from RSI alignment
        if direction == "up":
            # Higher RSI in uptrend = stronger signal
            rsi_boost = ((rsi - 50) / 50) * 0.2  # Max +0.2 at RSI=100
        else:  # down
            # Lower RSI in downtrend = stronger signal
            rsi_boost = ((50 - rsi) / 50) * 0.2  # Max +0.2 at RSI=0
        
        confidence = base_confidence + max(0, rsi_boost)
        return min(0.9, max(0.5, confidence))
    
    def _is_direction_allowed(self, direction: str) -> bool:
        """
        Check if trading direction is allowed by configuration
        
        Args:
            direction: "LONG" or "SHORT"
            
        Returns:
            True if direction is allowed
        """
        if self.allowed_directions == "both":
            return True
        elif self.allowed_directions == "long_only":
            return direction == "LONG"
        elif self.allowed_directions == "short_only":
            return direction == "SHORT"
        else:
            return False