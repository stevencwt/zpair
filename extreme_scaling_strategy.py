#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extreme Scaling Strategy - Advanced Position Scaling with Dynamic Extreme Tracking
EXTRACTED VERSION: Modular implementation of HTFBiasLTFTimingWithScaling
STAGE 4 UPDATE: Uses marketstate P&L instead of recalculating

FEATURES:
- Multi-timeframe extreme tracking for dynamic threshold determination
- Primary timeframe-based scaling decisions
- Emergency exit conditions
- Quick profit exit capability
- Stabilization period enforcement
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone


class ExtremeScalingStrategy:
    """
    Advanced position scaling strategy using dynamic extreme tracking
    Extracted from HTFBiasLTFTimingWithScaling for modular use
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy_config = self._load_strategy_config()
        
        # Initialize extreme tracker if scaling is enabled
        scaling_config = self.strategy_config.get("position_scaling_config", {})
        if scaling_config.get("enabled", False):
            try:
                from extreme_tracker import ExtremeTracker
                
                # Get snapshot_dir and primary_timeframe from config
                snapshot_dir = config.get("snapshot_dir", "marketstate")
                
                # Get primary timeframe from multi_timeframe_buffers config
                buffer_config = config.get("multi_timeframe_buffers", {})
                primary_timeframe = buffer_config.get("primary_timeframe", "1m")
                
                # Initialize ExtremeTracker with multi-timeframe support
                self.extreme_tracker = ExtremeTracker(
                    self.strategy_config, 
                    snapshot_dir, 
                    primary_timeframe
                )
                
                self.primary_timeframe = primary_timeframe
                
                print(f"[EXTREME SCALING] Multi-timeframe extreme tracker initialized")
                print(f"  Storage location: {snapshot_dir}/extremes_history.json")
                print(f"  Primary timeframe for scaling: {primary_timeframe}")
                print(f"  Collects data from: {buffer_config.get('tactical_timeframe', '5s')}, "
                      f"{primary_timeframe}, {buffer_config.get('bias_timeframe', '1h')}")
                
            except ImportError as e:
                print(f"[EXTREME SCALING ERROR] Could not import ExtremeTracker: {e}")
                self.extreme_tracker = None
                self.primary_timeframe = "1m"
        else:
            self.extreme_tracker = None
            self.primary_timeframe = "1m"
    
    def _load_strategy_config(self) -> Dict[str, Any]:
        """Load strategy configuration"""
        active_loss = self.config.get("active_strategies", {}).get("cut_loss", "")
        return self.config.get("cut_loss_strategies", {}).get(active_loss, {})
    
    def analyze_exit_profit(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                           config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check quick profit exit conditions
        STAGE 4 CHANGE: Uses P&L from marketstate (fresh calculation)
        """
        try:
            scaling_config = self.strategy_config.get("position_scaling_config", {})
            quick_profit = scaling_config.get("quick_profit_exit", {})
            
            if not quick_profit.get("enabled", False):
                return None
            
            # Get current P&L from marketstate (calculated fresh by main loop)
            pnl_tracking = position.get("pnl_tracking", {})
            if not pnl_tracking:
                print(f"[EXTREME SCALING] No P&L data available for quick profit analysis")
                return None
            
            current_pnl = pnl_tracking.get("floating_pnl_usd", 0.0)
            calculation_source = pnl_tracking.get("calculation_source", "unknown")
            
            if calculation_source == "fresh_from_static_data":
                print(f"[EXTREME SCALING] Using fresh P&L from marketstate: ${current_pnl:.2f}")
            else:
                print(f"[EXTREME SCALING] Warning: P&L source unknown: {calculation_source}")
            
            profit_target = quick_profit.get("profit_target_usd", 2.0)
            
            if current_pnl >= profit_target:
                return {
                    "aggregate_id": position["aggregate_id"],
                    "exit_type": "quick_profit",
                    "reason": f"Quick profit target reached: ${current_pnl:.2f} >= ${profit_target}",
                    "priority": "high",
                    "pnl_source": "fresh_marketstate_calculation"
                }
            
            return None
            
        except Exception as e:
            print(f"[EXTREME SCALING ERROR] Profit exit analysis failed: {e}")
            return None
    
    def analyze_exit_loss(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                         config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze loss-cutting and scaling opportunities"""
        try:
            # First check for emergency exits and stop losses
            exit_signal = self._check_emergency_exits(position, marketstate)
            if exit_signal:
                return exit_signal
            
            # Then check for scaling opportunities
            scaling_signal = self._analyze_extreme_scaling(position, marketstate)
            if scaling_signal:
                return scaling_signal
            
            return None
            
        except Exception as e:
            print(f"[EXTREME SCALING ERROR] Loss exit analysis failed: {e}")
            return None
    
    def _check_emergency_exits(self, position: Dict[str, Any], 
                              marketstate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for emergency exit conditions"""
        try:
            risk_mgmt = self.strategy_config.get("embedded_risk_management", {})
            emergency_config = risk_mgmt.get("emergency_exits", {})
            time_config = risk_mgmt.get("time_limits", {})
            
            # Time limit check
            max_hold_hours = time_config.get("max_hold_hours", 8.5)
            
            try:
                timestamps = position.get("timestamps", {})
                initial_entry_str = timestamps.get("initial_entry", "")
                
                if initial_entry_str:
                    initial_entry = datetime.fromisoformat(initial_entry_str.replace('Z', '+00:00'))
                    current_time = datetime.now(timezone.utc)
                    hold_hours = (current_time - initial_entry).total_seconds() / 3600
                    
                    if hold_hours > max_hold_hours:
                        return {
                            "aggregate_id": position["aggregate_id"],
                            "exit_type": "time_limit",
                            "reason": f"Maximum hold time exceeded: {hold_hours:.1f}h > {max_hold_hours}h",
                            "priority": "high",
                            "emergency": True,
                            "pnl_source": "not_applicable_time_based_exit"
                        }
            except Exception:
                pass
            
            # Data age check
            max_age_minutes = emergency_config.get("max_data_age_minutes", 10)
            try:
                asof_str = marketstate.get("asof", "")
                if asof_str:
                    asof_dt = datetime.fromisoformat(asof_str.replace('Z', '+00:00'))
                    age_minutes = (datetime.now(timezone.utc) - asof_dt).total_seconds() / 60
                    
                    if age_minutes > max_age_minutes:
                        return {
                            "aggregate_id": position["aggregate_id"],
                            "exit_type": "emergency_exit",
                            "reason": f"Data too old: {age_minutes:.1f} minutes > {max_age_minutes}",
                            "priority": "emergency",
                            "emergency": True,
                            "pnl_source": "not_applicable_data_age_exit"
                        }
            except Exception:
                pass
            
            return None
            
        except Exception as e:
            print(f"[EXTREME SCALING ERROR] Emergency exit check failed: {e}")
            return None
    
    def _analyze_extreme_scaling(self, position: Dict[str, Any], 
                                marketstate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze opportunities to add to existing position using dynamic extreme tracking
        STAGE 4 CHANGE: Uses P&L from marketstate (fresh calculation)
        """
        try:
            scaling_config = self.strategy_config.get("position_scaling_config", {})
            
            if not scaling_config.get("enabled", False):
                return None
            
            if not self.extreme_tracker:
                print(f"[EXTREME SCALING] ExtremeTracker not available - scaling disabled")
                return None
            
            # Check current position metrics
            aggregated_metrics = position.get("aggregated_metrics", {})
            actual_component_count = aggregated_metrics.get("actual_component_count", 0)
            max_add_positions = scaling_config.get("max_add_positions", 5)
            
            # Check maximum additional components allowed
            max_add_positions = scaling_config.get("max_add_positions", 5)
            
            if actual_component_count >= max_add_positions:
                print(f"[EXTREME SCALING] Max additional components reached: {actual_component_count}/{max_add_positions}")
                print(f"[EXTREME SCALING]   (Initial position + {max_add_positions} additional components max)")
                return None
            
            # Get current P&L from marketstate (calculated fresh by main loop)
            pnl_tracking = position.get("pnl_tracking", {})
            if not pnl_tracking:
                print(f"[EXTREME SCALING] No P&L data available for scaling analysis")
                return None
            
            current_pnl = pnl_tracking.get("floating_pnl_usd", 0.0)
            calculation_source = pnl_tracking.get("calculation_source", "unknown")
            
            if calculation_source == "fresh_from_static_data":
                print(f"[EXTREME SCALING] Using fresh P&L from marketstate for scaling: ${current_pnl:.2f}")
            else:
                print(f"[EXTREME SCALING] Warning: P&L source unknown: {calculation_source}")
            
            # Check loss threshold
            add_conditions = scaling_config.get("add_conditions", {})
            loss_threshold = add_conditions.get("floating_loss_threshold_usd", 5.0)
            
            if current_pnl >= -loss_threshold:
                print(f"[EXTREME SCALING] Loss threshold not met: ${current_pnl:.2f} >= -${loss_threshold}")
                return None
            
            # Check stabilization period
            stabilization_minutes = add_conditions.get("stabilization_period_minutes", 15)
            most_recent_add = position.get("timestamps", {}).get("most_recent_add")
            
            if not self.extreme_tracker.is_stabilized(stabilization_minutes, most_recent_add):
                print(f"[EXTREME SCALING] Stabilization period not met: {stabilization_minutes} minutes required")
                return None
            
            # Update extreme tracker with PRIMARY TIMEFRAME data only
            primary_zscore = marketstate.get("regression", {}).get("zscore", 0.0)
            timestamp = marketstate.get("asof", datetime.now(timezone.utc).isoformat())
            
            self.extreme_tracker.update(self.primary_timeframe, primary_zscore, timestamp)
            
            # Get position direction and check for extreme opportunities
            position_direction = aggregated_metrics.get("direction", "")
            
            # Check if current conditions represent an extreme opportunity
            extreme_config = scaling_config.get("extreme_tracking", {})
            max_age_hours = extreme_config.get("max_age_hours", 24)
            min_extremes = extreme_config.get("min_extremes_required", 2)
            
            # Get direction-specific thresholds (backward compatibility with single threshold)
            min_long_threshold = extreme_config.get("min_long_threshold", 
                                                  -abs(extreme_config.get("min_threshold", 1.0)))
            min_short_threshold = extreme_config.get("min_short_threshold", 
                                                   abs(extreme_config.get("min_threshold", 1.0)))
            
            # Check if we have sufficient fresh extreme data for PRIMARY TIMEFRAME
            has_sufficient, reason = self.extreme_tracker.has_sufficient_fresh_data(
                min_extremes, max_age_hours, position_direction, self.primary_timeframe
            )
            
            if not has_sufficient:
                print(f"[EXTREME SCALING] Insufficient extreme data for {self.primary_timeframe}: {reason}")
                return None
            
            # Get extreme threshold for PRIMARY TIMEFRAME
            extreme_threshold = self.extreme_tracker.get_nth_least_extreme(
                1, position_direction, self.primary_timeframe
            )
            
            if extreme_threshold is None:
                print(f"[EXTREME SCALING] No extreme threshold available for {position_direction} in {self.primary_timeframe}")
                return None
            
            # Check if current zscore meets extreme threshold
            is_extreme = False
            threshold_type = ""
            
            if position_direction == "LONG":
                is_extreme = primary_zscore <= extreme_threshold
                threshold_type = "extreme_low"
            elif position_direction == "SHORT":
                is_extreme = primary_zscore >= extreme_threshold
                threshold_type = "extreme_high"
            
            if is_extreme:
                new_position_number = actual_component_count + 1
                
                print(f"[EXTREME SCALING] Adding position: {threshold_type} threshold met")
                print(f"[EXTREME SCALING] Primary TF zscore: {primary_zscore:.3f} vs threshold {extreme_threshold:.3f}")
                print(f"[EXTREME SCALING] Current loss: ${current_pnl:.2f}")
                
                return {
                    "action": "ADD_POSITION",
                    "aggregate_id": position["aggregate_id"],
                    "direction": position_direction,
                    "confidence": 0.8,
                    "reasoning": f"Extreme {threshold_type} reached in {self.primary_timeframe}: zscore {primary_zscore:.3f} vs threshold {extreme_threshold:.3f} with ${current_pnl:.2f} floating loss",
                    "strategy": "extreme_scaling_strategy",
                    "threshold_met": primary_zscore,
                    "ltf_signal": primary_zscore,
                    "htf_bias": marketstate.get("regression_htf", {}).get("zscore", 0.0),
                    "pnl_source": "fresh_marketstate_calculation",
                    "add_position_metadata": {
                        "add_reason": f"{threshold_type}_opportunity",
                        "threshold_value": extreme_threshold,
                        "current_zscore": primary_zscore,
                        "timeframe_used": self.primary_timeframe,
                        "floating_loss_usd": current_pnl,
                        "new_position_number": new_position_number,
                        "extreme_tracker_stats": self.extreme_tracker.get_statistics(self.primary_timeframe),
                        "pnl_calculation_source": calculation_source
                    }
                }
            else:
                print(f"[EXTREME SCALING] Extreme threshold not met: {primary_zscore:.3f} vs {extreme_threshold:.3f} ({threshold_type})")
            
            return None
            
        except Exception as e:
            print(f"[EXTREME SCALING ERROR] Extreme scaling analysis failed: {e}")
            return None
    
    def persist_extreme_state(self):
        """Persist extreme tracking state to disk"""
        if self.extreme_tracker:
            try:
                self.extreme_tracker.persist()
                print(f"[EXTREME SCALING] Multi-timeframe extreme tracking state persisted")
            except Exception as e:
                print(f"[EXTREME SCALING ERROR] Failed to persist extreme state: {e}")
