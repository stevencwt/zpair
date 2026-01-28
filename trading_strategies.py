#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Strategies Module - REFACTORED for Fully Modular Architecture
UPDATED VERSION: Removed hardcoded HTFBiasLTFTimingWithScaling class
STAGE 4 UPDATE: Strategies use marketstate P&L instead of recalculating
ENHANCED VERSION: Added configurable direction control for entry strategies
ARCHITECTURE: Core base classes + essential strategies only

MODULAR ARCHITECTURE:
- Scaling strategies moved to separate modules (simple_scaling_strategy.py, extreme_scaling_strategy.py)
- Take-profit strategies moved to separate modules (hit_and_run_profit.py, etc.)
- This module now contains only base classes and core non-scaling strategies
- Cleaner separation of concerns and better maintainability

NEW FEATURES:
- Configurable direction control (both, long_only, short_only)
- Enhanced validation and logging for direction restrictions
- Backward compatibility with existing configurations
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy_config = self._load_strategy_config()
    
    @abstractmethod
    def _load_strategy_config(self) -> Dict[str, Any]:
        """Load strategy-specific configuration"""
        pass


class EntryStrategy(BaseStrategy):
    """Base class for entry strategies with position validation"""
    
    @abstractmethod
    def analyze_entry(self, marketstate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market conditions for entry opportunities"""
        pass
    
    def _check_position_limits(self, marketstate: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if new positions are allowed based on configuration limits"""
        try:
            position_mgmt = self.strategy_config.get('position_management', {})
            max_positions = position_mgmt.get('max_concurrent_positions', 1)
            
            current_position = marketstate.get('_current_position')
            
            if current_position is not None:
                if max_positions == 1:
                    return False, f"Single position strategy - position already exists"
                
                component_count = current_position.get('aggregated_metrics', {}).get('position_count', 1)
                
                if component_count >= max_positions:
                    return False, f"Maximum positions reached: {component_count}/{max_positions}"
                
                return True, "Additional position allowed for scaling strategy"
            else:
                return True, "No existing position - entry allowed"
                
        except Exception as e:
            print(f"[STRATEGY ERROR] Position limit check failed: {e}")
            return False, f"Position validation error: {str(e)}"


class HTFBiasLTFTimingEntry(EntryStrategy):
    """HTF Bias + LTF Timing Entry Strategy with configurable direction control and position count validation"""
    
    def _load_strategy_config(self) -> Dict[str, Any]:
        """Load HTF bias LTF timing configuration"""
        active_entry = self.config.get("active_strategies", {}).get("entry", "")
        return self.config.get("entry_strategies", {}).get(active_entry, {})
    
    def _validate_direction_allowed(self, signal_direction: str) -> Tuple[bool, str]:
        """
        Check if the signal direction is allowed based on configuration.
        
        Args:
            signal_direction: "LONG" or "SHORT"
            
        Returns:
            (is_allowed, reason)
        """
        try:
            direction_config = self.strategy_config.get("direction_control", {})
            allowed_directions = direction_config.get("allowed_directions", "both").lower()
            
            if allowed_directions == "both":
                return True, f"Both directions allowed"
            elif allowed_directions == "long_only":
                if signal_direction == "LONG":
                    return True, f"Long-only mode: LONG signal allowed"
                else:
                    return False, f"Long-only mode: SHORT signals blocked"
            elif allowed_directions == "short_only":
                if signal_direction == "SHORT":
                    return True, f"Short-only mode: SHORT signal allowed"
                else:
                    return False, f"Short-only mode: LONG signals blocked"
            else:
                print(f"[STRATEGY WARNING] Invalid allowed_directions config: {allowed_directions}. Defaulting to 'both'")
                return True, f"Invalid config - defaulting to both directions"
                
        except Exception as e:
            print(f"[STRATEGY ERROR] Direction validation failed: {e}")
            return True, f"Direction validation error - allowing signal"
    
    def analyze_entry(self, marketstate: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze entry with position count validation and direction control"""
        try:
            # NOTE: config parameter added for interface consistency but we use self.config from __init__
            
            # STEP 1: Position count validation
            position_allowed, validation_message = self._check_position_limits(marketstate)
            
            if not position_allowed:
                print(f"[STRATEGY] Entry blocked: {validation_message}")
                return None
            
            print(f"[STRATEGY] Position validation passed: {validation_message}")
            
            # STEP 2: Market condition analysis
            potential_signal = self._analyze_market_conditions(marketstate)
            
            if not potential_signal:
                return None
            
            # STEP 3: Direction control validation
            signal_direction = potential_signal.get("direction")
            direction_allowed, direction_reason = self._validate_direction_allowed(signal_direction)
            
            if not direction_allowed:
                print(f"[STRATEGY] Entry blocked by direction control: {direction_reason}")
                return None
            
            print(f"[STRATEGY] Direction validation passed: {direction_reason}")
            
            # Update signal with direction validation info
            potential_signal["direction_control"] = {
                "validation_passed": True,
                "reason": direction_reason,
                "configured_mode": self.strategy_config.get("direction_control", {}).get("allowed_directions", "both")
            }
            
            return potential_signal
            
        except Exception as e:
            print(f"[STRATEGY ERROR] Entry analysis failed: {e}")
            return None
    
    def _evaluate_regime_safety(self, regime: Dict[str, Any]) -> bool:
        """
        Evaluate if market regime is safe for THIS strategy using configurable thresholds.
        This replaces the telemetry's hardcoded decision-making.
        """
        # Get this strategy's specific requirements
        regime_config = self.strategy_config.get("market_regime_requirements", {})
        
        # Extract raw metrics from regime data
        correlation = abs(regime.get("trend_strength", {}).get("correlation", 0.0))
        volatility_ratio = regime.get("volatility_sync", {}).get("ratio", 0.0)
        divergent_trends = regime.get("divergent_trends", False)
        both_trending_up = regime.get("both_trending_up", False)
        both_trending_down = regime.get("both_trending_down", False)
        both_trending = both_trending_up or both_trending_down
        synchronized_volatility = regime.get("synchronized_volatility", False)
        
        # Get configurable thresholds with defaults matching original hardcoded values
        min_correlation = regime_config.get("min_correlation", 0.15)
        min_correlation_when_trending = regime_config.get("min_correlation_when_trending", 0.30)
        min_volatility_sync = regime_config.get("min_volatility_sync", 0.4)
        allow_divergent_trends = regime_config.get("allow_divergent_trends", False)
        
        # Apply this strategy's rules (matching original telemetry logic but configurable)
        if divergent_trends and not allow_divergent_trends:
            print(f"[STRATEGY] Regime unsafe: divergent trends (allow_divergent={allow_divergent_trends})")
            return False
            
        if correlation < min_correlation:
            print(f"[STRATEGY] Regime unsafe: poor correlation {correlation:.3f} < {min_correlation}")
            return False
            
        if both_trending and correlation < min_correlation_when_trending:
            print(f"[STRATEGY] Regime unsafe: trending with weak correlation {correlation:.3f} < {min_correlation_when_trending}")
            return False
            
        if volatility_ratio < min_volatility_sync:
            print(f"[STRATEGY] Regime unsafe: poor volatility sync {volatility_ratio:.3f} < {min_volatility_sync}")
            return False
        
        print(f"[STRATEGY] Regime assessed as SAFE (corr={correlation:.3f}, vol_sync={volatility_ratio:.3f})")
        return True
    
    def _analyze_market_conditions(self, marketstate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze market conditions for entry opportunities.
        
        NOTE: This method generates potential signals without direction filtering.
        Direction filtering is applied in analyze_entry() for better separation of concerns.
        """
        try:
            # Extract market data
            regression = marketstate.get("regression", {})
            regression_htf = marketstate.get("regression_htf", {})
            regime = marketstate.get("regime_assessment", {})
            
            htf_zscore = regression_htf.get("zscore", 0.0)
            ltf_zscore = regression.get("zscore", 0.0)
            ltf_rsquare = regression.get("r_squared", 0.0)
            htf_rsquare = regression_htf.get("r_squared", 0.0)
            
            # Use the new regime evaluation method instead of pre-made decision
            safe_for_mr = self._evaluate_regime_safety(regime)
            
            # Load strategy configuration
            htf_config = self.strategy_config.get("htf_bias_config", {})
            ltf_config = self.strategy_config.get("ltf_timing_config", {})
            blocking_config = self.strategy_config.get("trade_blocking", {})
            direction_config = self.strategy_config.get("direction_control", {})
            
            # HTF bias thresholds
            strong_positive = htf_config.get("strong_positive_threshold", 1.0)
            strong_negative = htf_config.get("strong_negative_threshold", -1.0)
            neutral_band = htf_config.get("neutral_band", [-0.5, 0.5])
            
            # LTF timing thresholds
            entry_thresholds = ltf_config.get("entry_thresholds", {"long": -0.4, "short": 0.4})
            
            # Trade blocking gates
            rsquare_gate = blocking_config.get("htf_rsquare_gate", {})
            rsquare_enabled = rsquare_gate.get("enabled", False)
            min_rsquare = rsquare_gate.get("min_threshold", 0.2)
            
            # Direction control info for logging
            allowed_directions = direction_config.get("allowed_directions", "both")
            
            print(f"[STRATEGY] Market Analysis - HTF: {htf_zscore:.3f}, LTF: {ltf_zscore:.3f}, "
                  f"R²(LTF/HTF): {ltf_rsquare:.3f}/{htf_rsquare:.3f}, Safe: {safe_for_mr}, "
                  f"Direction Mode: {allowed_directions}")
            
            # Check trade blocking conditions
            if rsquare_enabled and htf_rsquare < min_rsquare:
                print(f"[STRATEGY] Entry blocked - HTF R² too low: {htf_rsquare:.3f} < {min_rsquare}")
                return None
            
            if not safe_for_mr:
                print(f"[STRATEGY] Entry blocked - regime not safe for mean reversion")
                return None
            
            # Determine HTF bias
            htf_bias_type = "neutral"
            if htf_zscore >= strong_positive:
                htf_bias_type = "strong_positive"
            elif htf_zscore <= strong_negative:
                htf_bias_type = "strong_negative"
            elif neutral_band[0] <= htf_zscore <= neutral_band[1]:
                htf_bias_type = "neutral"
            
            # Strategy logic based on HTF bias (direction filtering happens later)
            entry_signal = None
            
            if htf_bias_type == "strong_positive":
                if htf_config.get("favor_shorts_when_htf_positive", True):
                    if ltf_zscore >= entry_thresholds["short"]:
                        entry_signal = {
                            "direction": "SHORT",
                            "confidence": min(0.9, abs(ltf_zscore) * 0.3),
                            "htf_bias": htf_zscore,
                            "ltf_signal": ltf_zscore,
                            "reasoning": f"Strong HTF positive bias ({htf_zscore:.3f}) + LTF short signal ({ltf_zscore:.3f})",
                            "strategy": "htf_bias_ltf_timing",
                            "htf_bias_type": htf_bias_type
                        }
            
            elif htf_bias_type == "strong_negative":
                if htf_config.get("favor_longs_when_htf_negative", True):
                    if ltf_zscore <= entry_thresholds["long"]:
                        entry_signal = {
                            "direction": "LONG",
                            "confidence": min(0.9, abs(ltf_zscore) * 0.3),
                            "htf_bias": htf_zscore,
                            "ltf_signal": ltf_zscore,
                            "reasoning": f"Strong HTF negative bias ({htf_zscore:.3f}) + LTF long signal ({ltf_zscore:.3f})",
                            "strategy": "htf_bias_ltf_timing",
                            "htf_bias_type": htf_bias_type
                        }
            
            elif htf_bias_type == "neutral":
                if htf_config.get("both_directions_when_neutral", True):
                    if ltf_zscore >= entry_thresholds["short"]:
                        entry_signal = {
                            "direction": "SHORT",
                            "confidence": min(0.7, abs(ltf_zscore) * 0.25),
                            "htf_bias": htf_zscore,
                            "ltf_signal": ltf_zscore,
                            "reasoning": f"Neutral HTF bias ({htf_zscore:.3f}) + LTF short signal ({ltf_zscore:.3f})",
                            "strategy": "htf_bias_ltf_timing",
                            "htf_bias_type": htf_bias_type
                        }
                    elif ltf_zscore <= entry_thresholds["long"]:
                        entry_signal = {
                            "direction": "LONG",
                            "confidence": min(0.7, abs(ltf_zscore) * 0.25),
                            "htf_bias": htf_zscore,
                            "ltf_signal": ltf_zscore,
                            "reasoning": f"Neutral HTF bias ({htf_zscore:.3f}) + LTF long signal ({ltf_zscore:.3f})",
                            "strategy": "htf_bias_ltf_timing",
                            "htf_bias_type": htf_bias_type
                        }
            
            if entry_signal:
                print(f"[STRATEGY] Potential entry signal generated: {entry_signal['direction']} "
                      f"(confidence: {entry_signal['confidence']:.2f}) - awaiting direction validation")
            
            return entry_signal
            
        except Exception as e:
            print(f"[STRATEGY ERROR] Market analysis failed: {e}")
            return None


class ExitStrategy(BaseStrategy):
    """Base class for exit strategies"""
    
    @abstractmethod
    def analyze_exit_profit(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                           config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze profit-taking exit conditions"""
        pass
    
    @abstractmethod
    def analyze_exit_loss(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                         config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze loss-cutting exit conditions"""
        pass


class PositionScalingStrategy(ExitStrategy):
    """
    Base class for position scaling strategies.
    NOTE: Concrete implementations are now in separate modules:
    - simple_scaling_strategy.py
    - extreme_scaling_strategy.py
    """
    
    @abstractmethod 
    def analyze_scaling_opportunity(self, position: Dict[str, Any], 
                                  marketstate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze opportunities to scale into positions"""
        pass


class MultiCriteriaStopLoss(ExitStrategy):
    """
    Multi-criteria stop loss strategy
    STAGE 4 UPDATE: Uses marketstate P&L instead of recalculating
    """
    
    def _load_strategy_config(self) -> Dict[str, Any]:
        active_loss = self.config.get("active_strategies", {}).get("cut_loss", "")
        return self.config.get("cut_loss_strategies", {}).get(active_loss, {})
    
    def analyze_exit_profit(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                           config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Not implemented for loss-cutting strategy"""
        return None
    
    def analyze_exit_loss(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                         config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze loss-cutting conditions - UNIFIED ENGINE COMPATIBLE
        """
        try:
            # =================================================================
            # 1. UNIFIED P&L EXTRACTION (CLEAN)
            # =================================================================
            current_pnl = 0.0
            
            # Check for flat 'pnl' (Unified Engine Standard)
            if 'pnl' in position:
                current_pnl = float(position['pnl'])
            # Fallback for legacy data structures (just in case)
            elif 'pnl_tracking' in position: 
                current_pnl = float(position['pnl_tracking'].get('floating_pnl_usd', 0.0))
            else:
                # If neither exists, we cannot calculate loss, so we must abort safely
                print(f"[STRATEGY] ❌ CRITICAL: No P&L data available for Stop Loss analysis")
                return None
            
            # =================================================================
            # 2. STOP LOSS LOGIC (PRESERVED)
            # =================================================================
            stop_config = self.strategy_config.get("stop_loss_exit", {})
            max_loss_dollar = stop_config.get("max_loss_dollar", 30.0)
            
            # Check Max Dollar Loss
            if current_pnl <= -max_loss_dollar:
                return {
                    # Unified engine uses 'id', legacy used 'aggregate_id'. 
                    # We use .get('id') as the safe default.
                    "aggregate_id": position.get("id"), 
                    "exit_type": "stop_loss",
                    "reason": f"Maximum loss exceeded: ${current_pnl:.2f} <= -${max_loss_dollar}",
                    "priority": "high",
                    "emergency": True
                }
            
            # =================================================================
            # 3. EMERGENCY DATA AGE CHECK (PRESERVED)
            # =================================================================
            emergency_config = self.strategy_config.get("embedded_risk_management", {}).get("emergency_exits", {})
            max_age_minutes = emergency_config.get("max_data_age_minutes", 10)
            
            try:
                asof_str = marketstate.get("asof", "")
                if asof_str:
                    # Robust timestamp parsing
                    if asof_str.endswith('Z'): asof_str = asof_str[:-1]
                    asof_dt = datetime.fromisoformat(asof_str).replace(tzinfo=timezone.utc)
                    age_minutes = (datetime.now(timezone.utc) - asof_dt).total_seconds() / 60
                    
                    if age_minutes > max_age_minutes:
                        return {
                            "aggregate_id": position.get("id"),
                            "exit_type": "emergency_exit",
                            "reason": f"Data too old: {age_minutes:.1f} minutes > {max_age_minutes}",
                            "priority": "emergency",
                            "emergency": True
                        }
            except Exception:
                pass
            
            return None
            
        except Exception as e:
            print(f"[STRATEGY ERROR] Loss exit analysis failed: {e}")
            return None


# REMOVED: HTFBiasLTFTimingWithScaling class
# This class has been refactored into the modular extreme_scaling_strategy.py
# The functionality is preserved but now follows the modular architecture pattern
# Benefits:
# - Cleaner separation of concerns
# - Better maintainability
# - Consistent with take-profit strategy modularization
# - Easier testing and debugging
# - Reduced coupling between components

# REMOVED: ConfigurableProfitExit class  
# Take-profit strategies are now in separate modules:
# - hit_and_run_profit.py
# - let_profit_run_profit.py  
# - trailing_stop_profit.py

# This file now contains only:
# - Base strategy classes (BaseStrategy, EntryStrategy, ExitStrategy, PositionScalingStrategy)
# - Core entry strategy (HTFBiasLTFTimingEntry) with enhanced direction control
# - Basic stop-loss strategy (MultiCriteriaStopLoss)
# 
# All complex scaling and profit-taking logic has been moved to dedicated modules
# for better maintainability and cleaner architecture.
#
# NEW FEATURES in this version:
# - Configurable direction control for HTFBiasLTFTimingEntry
# - Enhanced validation and logging for direction restrictions  
# - Clean separation of market analysis and direction filtering
# - Backward compatibility with existing configurations