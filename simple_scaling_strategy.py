#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Scaling Strategy - Enhanced with Progressive Threshold Increments
ENHANCED VERSION: Supports both fixed and progressive threshold scaling
FEATURES:
- Fixed thresholds (original behavior)
- Progressive thresholds with configurable increments
- Automatic threshold calculation based on position count
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone


class SimpleScalingStrategy:
    """
    Simple position scaling strategy with enhanced threshold options:
    1. Fixed thresholds: Always use same thresholds (-1.0/+1.0)
    2. Progressive thresholds: Increase magnitude by fixed increment for each add-on
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy_config = self._load_strategy_config()
        
        # Enhanced threshold configuration
        self._validate_threshold_config()
        
        #print(f"[SIMPLE SCALING] Enhanced simple scaling strategy initialized")
        #print(f"[SIMPLE SCALING] Threshold mode: {self._get_threshold_mode()}")
        if self._is_progressive_mode():
            increment = self._get_threshold_increment()
            #print(f"[SIMPLE SCALING] Progressive increment: {increment}")
    
    def _load_strategy_config(self) -> Dict[str, Any]:
        """Load strategy configuration"""
        active_loss = self.config.get("active_strategies", {}).get("cut_loss", "")
        return self.config.get("cut_loss_strategies", {}).get(active_loss, {})
    
    def _validate_threshold_config(self):
        """Validate threshold configuration and set defaults"""
        scaling_config = self.strategy_config.get("position_scaling_config", {})
        threshold_config = scaling_config.get("simple_zscore_thresholds", {})
        
        # Note: long_add_threshold and short_add_threshold are no longer used
        # Strategy calculates thresholds dynamically based on entry z-scores
        
        # Set default for progressive mode
        if "use_progressive_thresholds" not in threshold_config:
            threshold_config["use_progressive_thresholds"] = False
        
        # Set default increment
        if "threshold_increment" not in threshold_config:
            threshold_config["threshold_increment"] = 0.2
        
        # Set default scaling mode
        if "scaling_mode" not in threshold_config:
            threshold_config["scaling_mode"] = "fixed_increment"
        
        # Set default percentage threshold
        if "percentage_threshold" not in threshold_config:
            threshold_config["percentage_threshold"] = 20.0
        
        #print(f"[SIMPLE SCALING] Threshold config validated:")
        #print(f"  Strategy mode: Progressive entry-based thresholds")
        #print(f"  No fixed base thresholds - calculated from position context")
        #print(f"  Progressive mode: {threshold_config['use_progressive_thresholds']}")
        #print(f"  Scaling mode: {threshold_config['scaling_mode']}")
        #print(f"  Increment: {threshold_config['threshold_increment']}")
        #print(f"  Percentage threshold: {threshold_config['percentage_threshold']}")
    
    def _get_threshold_mode(self) -> str:
        """Get threshold mode description"""
        scaling_config = self.strategy_config.get("position_scaling_config", {})
        threshold_config = scaling_config.get("simple_zscore_thresholds", {})
        scaling_mode = threshold_config.get("scaling_mode", "fixed_increment")
        
        if self._is_progressive_mode():
            return f"Progressive ({scaling_mode})"
        else:
            return "Fixed (same thresholds)"
    
    def _is_progressive_mode(self) -> bool:
        """Check if progressive thresholds are enabled"""
        scaling_config = self.strategy_config.get("position_scaling_config", {})
        threshold_config = scaling_config.get("simple_zscore_thresholds", {})
        return threshold_config.get("use_progressive_thresholds", False)
    
    def _get_threshold_increment(self) -> float:
        """Get threshold increment value"""
        scaling_config = self.strategy_config.get("position_scaling_config", {})
        threshold_config = scaling_config.get("simple_zscore_thresholds", {})
        return threshold_config.get("threshold_increment", 0.2)
    
    def _get_scaling_mode(self) -> str:
        """Get scaling mode (fixed_increment or percentage_based)"""
        scaling_config = self.strategy_config.get("position_scaling_config", {})
        threshold_config = scaling_config.get("simple_zscore_thresholds", {})
        return threshold_config.get("scaling_mode", "fixed_increment")
    
    def _get_percentage_threshold(self) -> float:
        """Get percentage threshold for percentage-based scaling"""
        scaling_config = self.strategy_config.get("position_scaling_config", {})
        threshold_config = scaling_config.get("simple_zscore_thresholds", {})
        return threshold_config.get("percentage_threshold", 20.0)
    
    def _extract_entry_zscore_direct(self, position: Dict[str, Any]) -> Optional[float]:
        """
        Extract entry z-score directly from position (for non-aggregated positions).
        Falls back to component_trades if available.
        
        Args:
            position: Position data
            
        Returns:
            Entry z-score if found, None otherwise
        """
        try:
            # Try direct field first (unified_trade_engine format)
            if 'entry_zscore' in position:
                return float(position['entry_zscore'])
            
            # Fall back to component_trades format (aggregated positions)
            return self._extract_entry_zscore_from_components(position, 0)
            
        except Exception as e:
            print(f"[SIMPLE SCALING] Failed to extract entry zscore: {e}")
            return None


    def _extract_entry_zscore_from_components(self, position: Dict[str, Any], component_index: int = 0) -> Optional[float]:
        """
        Extract entry z-score from position component_trades.
        
        Args:
            position: Position data
            component_index: Which component to extract from (0 = first, -1 = last)
            
        Returns:
            Entry z-score if found, None otherwise
        """
        try:
            # CHANGED: Look for 'component_trades' instead of 'components'
            component_trades = position.get("component_trades", [])
            if not component_trades:
                return None
            
            # Handle negative indexing
            if component_index < 0:
                component_index = len(component_trades) + component_index
            
            if component_index >= len(component_trades) or component_index < 0:
                return None
            
            # CHANGED: Get z-score directly from trade, not from nested metadata
            trade = component_trades[component_index]
            zscore = trade.get("entry_zscore")
            if zscore is not None:
                return float(zscore)
            
            # Fallback field names (keeping for compatibility)
            for field_name in ["ltf_signal", "signal_zscore"]:
                if field_name in trade:
                    zscore = trade[field_name]
                    if zscore is not None:
                        return float(zscore)
            
            return None
            
        except Exception as e:
            print(f"[SIMPLE SCALING] Failed to extract entry zscore: {e}")
            return None
    
    def _calculate_percentage_based_threshold(self, position: Dict[str, Any], direction: str) -> Optional[float]:
        """
        Calculate threshold based on percentage worse than PREVIOUS component's z-score.
        
        Args:
            position: Position data
            direction: "LONG" or "SHORT"
            
        Returns:
            Calculated threshold or None if previous z-score not available
        """
        try:
            # Get the actual component count
            component_count = position.get("aggregated_metrics", {}).get("actual_component_count", 1)
            
            if component_count == 1:
                # First add-on: compare to entry (direct from position)
                previous_zscore = self._extract_entry_zscore_direct(position)
            else:
                # Subsequent add-ons: compare to previous component
                previous_zscore = self._extract_entry_zscore_from_components(position, component_count - 1)
            
            if previous_zscore is None:
                return None
            
            percentage = self._get_percentage_threshold()
            
            # Calculate threshold as percentage worse than previous
            if direction == "LONG":
                # For LONG: make more negative (multiply by 1 + percentage)
                # e.g., -2.0 * 1.2 = -2.4 (20% more extreme)
                threshold = previous_zscore * (1 + percentage / 100.0)
            elif direction == "SHORT":
                # For SHORT: make more positive (multiply by 1 + percentage)
                # e.g., 2.0 * 1.2 = 2.4 (20% more extreme)
                threshold = previous_zscore * (1 + percentage / 100.0)
            else:
                return None
            
            #print(f"[SIMPLE SCALING] Percentage calculation: {previous_zscore:.3f} * "
            #      f"(1 + {percentage}%) = {threshold:.3f}")
            return threshold
            
           
        except Exception as e:
            print(f"[SIMPLE SCALING] Failed to calculate percentage-based threshold: {e}")
            return None
    
    def _calculate_fixed_increment_threshold(self, position: Dict[str, Any], direction: str, component_count: int) -> Optional[float]:
        """
        Calculate threshold using fixed increment from PREVIOUS component z-score.
        For first add-on, use entry z-score + increment.
        For subsequent add-ons, use previous component z-score + increment.
        
        Args:
            position: Position data
            direction: "LONG" or "SHORT" 
            component_count: Current number of components (1 means we're adding the 2nd)
            
        Returns:
            Calculated threshold or None if z-score not available
        """
        try:
            increment = self._get_threshold_increment()
            
            if component_count == 1:
                # First add-on: compare to entry (component 0)
                base_zscore = self._extract_entry_zscore_direct(position)
                if base_zscore is None:
                    return None
            else:
                # Subsequent add-ons: compare to previous component
                base_zscore = self._extract_entry_zscore_from_components(position, component_count - 1)
                if base_zscore is None:
                    return None
            
            # Calculate threshold as previous z-score ± increment
            if direction == "LONG":
                # For LONG: require more negative (subtract increment)
                threshold = base_zscore - increment
            elif direction == "SHORT":
                # For SHORT: require more positive (add increment)
                threshold = base_zscore + increment
            else:
                return None
            
            #print(f"[SIMPLE SCALING] Fixed increment: Component {component_count + 1} requires "
            #      f"z-score beyond {base_zscore:.3f} by {increment} = {threshold:.3f}")
            return threshold
            
        except Exception as e:
            print(f"[SIMPLE SCALING] Failed to calculate fixed increment threshold: {e}")
            return None
    
    def _calculate_threshold_for_position(self, base_threshold: float, current_position_count: int) -> float:
        """
        Calculate threshold based on current position count.
        
        Args:
            base_threshold: Base threshold value (e.g., -1.0 or 1.0)
            current_position_count: Current number of components in position
            
        Returns:
            Calculated threshold for next add-on
        """
        if not self._is_progressive_mode():
            # Fixed mode: always return base threshold
            return base_threshold
        
        # Progressive mode: increase magnitude by increment for each existing component beyond the first
        increment = self._get_threshold_increment()
        
        # Calculate how many add-ons have already been made
        # current_position_count includes initial position, so add-ons = count - 1
        add_on_count = max(0, current_position_count - 1)
        
        # Calculate progressive threshold
        if base_threshold < 0:
            # For negative thresholds (LONG positions), make more negative
            progressive_threshold = base_threshold - (add_on_count * increment)
        else:
            # For positive thresholds (SHORT positions), make more positive
            progressive_threshold = base_threshold + (add_on_count * increment)
        
        #print(f"[SIMPLE SCALING] Threshold calculation:")
        #print(f"  Base threshold: {base_threshold}")
        #print(f"  Current position count: {current_position_count}")
        #print(f"  Add-on count: {add_on_count}")
        #print(f"  Increment: {increment}")
        #print(f"  Progressive threshold: {progressive_threshold}")
        
        return progressive_threshold
    
    def analyze_exit_profit(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                           config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check quick profit exit conditions - UNIFIED COMPATIBLE
        """
        try:
            scaling_config = self.strategy_config.get("position_scaling_config", {})
            quick_profit = scaling_config.get("quick_profit_exit", {})
            
            if not quick_profit.get("enabled", False):
                return None
            
            # =================================================================
            # UNIFIED ENGINE P&L EXTRACTION (CLEAN)
            # =================================================================
            current_pnl = 0.0
            if 'pnl' in position:
                current_pnl = float(position['pnl'])
            elif 'pnl_tracking' in position:
                current_pnl = float(position['pnl_tracking'].get('floating_pnl_usd', 0.0))
            else:
                print(f"[SIMPLE SCALING] ❌ CRITICAL: No P&L data available for quick profit analysis")
                return None
            # =================================================================
            
            # Check Profit Target
            profit_target = quick_profit.get("profit_target_usd", 2.0)
            
            if current_pnl >= profit_target:
                return {
                    # Use .get('id') for safety with Unified Engine
                    "aggregate_id": position.get("id", position.get("aggregate_id")),
                    "exit_type": "quick_profit",
                    "reason": f"Quick profit target reached: ${current_pnl:.2f} >= ${profit_target}",
                    "priority": "high",
                    "pnl_source": "fresh_marketstate_calculation"
                }
            
            return None
            
        except Exception as e:
            print(f"[SIMPLE SCALING ERROR] Profit exit analysis failed: {e}")
            return None
    
    def analyze_exit_loss(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                         config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze loss-cutting and scaling opportunities with enhanced thresholds"""
        try:
            # First check for emergency exits and stop losses
            exit_signal = self._check_emergency_exits(position, marketstate)
            if exit_signal:
                return exit_signal
            
            # Then check for scaling opportunities with enhanced thresholds
            scaling_signal = self._analyze_enhanced_scaling(position, marketstate)
            if scaling_signal:
                return scaling_signal
            
            return None
            
        except Exception as e:
            print(f"[SIMPLE SCALING ERROR] Loss exit analysis failed: {e}")
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
                            # FIX: Use .get() with fallback to prevent crash
                            "aggregate_id": position.get("id", position.get("aggregate_id")),
                            "exit_type": "time_limit",
                            "reason": f"Maximum hold time exceeded: {hold_hours:.1f}h > {max_hold_hours}h",
                            "priority": "high",
                            "emergency": True
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
                            # FIX: Use .get() with fallback to prevent crash
                            "aggregate_id": position.get("id", position.get("aggregate_id")),
                            "exit_type": "emergency_exit",
                            "reason": f"Data too old: {age_minutes:.1f} minutes > {max_age_minutes}",
                            "priority": "emergency",
                            "emergency": True
                        }
            except Exception:
                pass
            
            return None
            
        except Exception as e:
            print(f"[SIMPLE SCALING ERROR] Emergency exit check failed: {e}")
            return None
    
    def _analyze_enhanced_scaling(self, position: Dict[str, Any], 
                                 marketstate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        ENHANCED: Analyze scaling opportunities using progressive or fixed thresholds
        """
        try:
            scaling_config = self.strategy_config.get("position_scaling_config", {})
            
            if not scaling_config.get("enabled", False):
                return None
            
            # Check current position metrics
            aggregated_metrics = position.get("aggregated_metrics", {})
            actual_component_count = aggregated_metrics.get("actual_component_count", 0)
            max_add_positions = scaling_config.get("max_add_positions", 3)
            
            # Calculate maximum total components: 1 initial + max_add_positions
            max_total_components = 1 + max_add_positions
                        
            if actual_component_count >= max_total_components:
                print(f"[SIMPLE SCALING] Max components reached: {actual_component_count}/{max_total_components}")
                return None
            
            # =================================================================
            # UNIFIED ENGINE P&L EXTRACTION (CLEAN)
            # =================================================================
            current_pnl = 0.0
            if 'pnl' in position:
                current_pnl = float(position['pnl'])
            elif 'pnl_tracking' in position:
                current_pnl = float(position['pnl_tracking'].get('floating_pnl_usd', 0.0))
            else:
                print(f"[SIMPLE SCALING] ❌ CRITICAL: No P&L data available for scaling analysis")
                return None
            # =================================================================
            
            # Check loss threshold
            add_conditions = scaling_config.get("add_conditions", {})
            loss_threshold = add_conditions.get("floating_loss_threshold_usd", 5.0)
            
            if current_pnl >= -loss_threshold:
                # print(f"[SIMPLE SCALING] Loss threshold not met: ${current_pnl:.2f} >= -${loss_threshold}")
                return None
            
            # Check stabilization period
            stabilization_minutes = add_conditions.get("stabilization_period_minutes", 15.0)
            
            if not self._is_stabilization_period_met(position, stabilization_minutes):
                print(f"[SIMPLE SCALING] Stabilization period not met: {stabilization_minutes} minutes required")
                return None
            
            # ENHANCED: Get current primary timeframe z-score and position direction
            primary_zscore = marketstate.get("regression", {}).get("zscore", 0.0)
            position_direction = aggregated_metrics.get("direction", "")
            
            # ENHANCED: Get base thresholds and scaling mode
            threshold_config = scaling_config.get("simple_zscore_thresholds", {})
            base_long_threshold = threshold_config.get("long_add_threshold", -1.5)
            base_short_threshold = threshold_config.get("short_add_threshold", 1.5)
            scaling_mode = self._get_scaling_mode()
            
            # Calculate thresholds based on scaling mode
            calculated_threshold = None
            threshold_description = ""
            
            if scaling_mode == "percentage_based":
                calculated_threshold = self._calculate_percentage_based_threshold(position, position_direction)
                if calculated_threshold is not None:
                    threshold_description = f"{scaling_mode} ({self._get_percentage_threshold()}% worse)"
                
            elif scaling_mode == "fixed_increment":
                calculated_threshold = self._calculate_fixed_increment_threshold(position, position_direction, actual_component_count)
                if calculated_threshold is not None:
                    threshold_description = f"{scaling_mode} (entry-based progressive)"
            
            # Fallback
            if calculated_threshold is None:
                if position_direction == "LONG":
                    calculated_threshold = self._calculate_threshold_for_position(base_long_threshold, actual_component_count)
                elif position_direction == "SHORT":
                    calculated_threshold = self._calculate_threshold_for_position(base_short_threshold, actual_component_count)
                threshold_description = "fallback_progressive"
            
            # Check threshold
            should_add_position = False
            threshold_check_description = ""
            
            if position_direction == "LONG" and calculated_threshold is not None:
                should_add_position = (primary_zscore <= calculated_threshold)
                threshold_check_description = f"LONG: zscore {primary_zscore:.3f} <= {calculated_threshold:.3f}"
                
            elif position_direction == "SHORT" and calculated_threshold is not None:
                should_add_position = (primary_zscore >= calculated_threshold)
                threshold_check_description = f"SHORT: zscore {primary_zscore:.3f} >= {calculated_threshold:.3f}"
            
            if should_add_position:
                new_position_number = actual_component_count + 1
                
                print(f"[SIMPLE SCALING] Adding position: {threshold_check_description}")
                print(f"[SIMPLE SCALING] Current loss: ${current_pnl:.2f}")
                
                add_metadata = {
                    "add_reason": f"simple_{scaling_mode}_scaling",
                    "threshold_type": f"{position_direction.lower()}_threshold",
                    "base_threshold": base_long_threshold if position_direction == "LONG" else base_short_threshold,
                    "calculated_threshold": calculated_threshold,
                    "scaling_mode": scaling_mode,
                    "threshold_description": threshold_description,
                    "current_zscore": primary_zscore,
                    "floating_loss_usd": current_pnl,
                    "new_position_number": new_position_number,
                    "scaling_type": f"simple_{scaling_mode}_scaling"
                }
                
                if scaling_mode == "percentage_based":
                    add_metadata.update({
                        "percentage_threshold": self._get_percentage_threshold(),
                        "previous_entry_zscore": self._extract_entry_zscore_from_components(position, -1),
                        "scaling_info": f"{self._get_percentage_threshold()}% worse than previous entry"
                    })
                elif scaling_mode == "fixed_increment":
                    entry_zscore = self._extract_entry_zscore_direct(position)
                    add_metadata.update({
                        "threshold_increment": self._get_threshold_increment(),
                        "entry_zscore": entry_zscore,
                        "add_on_count": actual_component_count - 1,
                        "scaling_info": f"Entry {entry_zscore:.3f} ± {actual_component_count - 1} increments = {calculated_threshold:.3f}"
                    })
                else:
                    add_metadata.update({
                        "threshold_increment": self._get_threshold_increment(),
                        "add_on_count": actual_component_count - 1,
                        "scaling_info": f"Fallback progressive: Base {add_metadata['base_threshold']} ± {actual_component_count - 1} increments = {calculated_threshold}"
                    })
                
                return {
                    "action": "ADD_POSITION",
                    # FIX: Use .get('id') with fallback for safety
                    "aggregate_id": position.get("id", position.get("aggregate_id")),
                    "direction": position_direction,
                    "confidence": 0.7,
                    "reasoning": f"Enhanced dual-mode scaling triggered: {threshold_check_description} with ${current_pnl:.2f} floating loss ({scaling_mode})",
                    "strategy": "simple_scaling_strategy",
                    "threshold_met": primary_zscore,
                    "ltf_signal": primary_zscore,
                    "htf_bias": marketstate.get("regression_htf", {}).get("zscore", 0.0),
                    "pnl_source": "fresh_marketstate_calculation",
                    "add_position_metadata": add_metadata
                }
            else:
                # Logging (Unchanged)
                if scaling_mode == "percentage_based":
                   pass
                elif scaling_mode == "fixed_increment":
                   pass
            
            return None
            
        except Exception as e:
            print(f"[SIMPLE SCALING ERROR] Enhanced scaling analysis failed: {e}")
            return None
    
    def _is_stabilization_period_met(self, position: Dict[str, Any], 
                                   stabilization_minutes: float) -> bool:
        """
        Check if enough time has passed since the last position add
        Supports floating point minutes (e.g., 0.5 for 30 seconds)
        """
        try:
            timestamps = position.get("timestamps", {})
            most_recent_add = timestamps.get("most_recent_add")
            
            if not most_recent_add:
                # No previous add, stabilization period is considered met
                return True
            
            # Parse the timestamp
            last_add_time = datetime.fromisoformat(most_recent_add.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            
            # Calculate time difference in minutes
            time_diff_seconds = (current_time - last_add_time).total_seconds()
            time_diff_minutes = time_diff_seconds / 60.0
            
            is_stabilized = time_diff_minutes >= stabilization_minutes
            
            if not is_stabilized:
                remaining_minutes = stabilization_minutes - time_diff_minutes
                print(f"[SIMPLE SCALING] Stabilization: {time_diff_minutes:.1f}/{stabilization_minutes} min "
                      f"(need {remaining_minutes:.1f} more)")
            
            return is_stabilized
            
        except Exception as e:
            print(f"[SIMPLE SCALING ERROR] Stabilization check failed: {e}")
            # On error, assume stabilization period is met (safer for trading)
            return True

    def get_threshold_info(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get threshold information for debugging/analysis purposes
        """
        try:
            aggregated_metrics = position.get("aggregated_metrics", {})
            actual_component_count = aggregated_metrics.get("actual_component_count", 1)
            position_direction = aggregated_metrics.get("direction", "UNKNOWN")
            
            scaling_config = self.strategy_config.get("position_scaling_config", {})
            threshold_config = scaling_config.get("simple_zscore_thresholds", {})
            
            base_long_threshold = threshold_config.get("long_add_threshold", -1.0)
            base_short_threshold = threshold_config.get("short_add_threshold", 1.0)
            scaling_mode = self._get_scaling_mode()
            
            # Calculate current and next thresholds based on scaling mode
            current_threshold = None
            next_threshold = None
            
            if scaling_mode == "percentage_based":
                current_threshold = self._calculate_percentage_based_threshold(position, position_direction)
                # Next threshold would be based on current entry (if added)
                next_threshold = "Depends on next entry z-score"
                
            elif scaling_mode == "fixed_increment":
                current_threshold = self._calculate_fixed_increment_threshold(position, position_direction, actual_component_count)
                next_threshold = self._calculate_fixed_increment_threshold(position, position_direction, actual_component_count + 1)
            
            # Fallback calculations
            if current_threshold is None:
                if position_direction == "LONG":
                    current_threshold = self._calculate_threshold_for_position(base_long_threshold, actual_component_count)
                    next_threshold = self._calculate_threshold_for_position(base_long_threshold, actual_component_count + 1)
                else:
                    current_threshold = self._calculate_threshold_for_position(base_short_threshold, actual_component_count)
                    next_threshold = self._calculate_threshold_for_position(base_short_threshold, actual_component_count + 1)
            
            # Extract entry information
            entry_zscore = self._extract_entry_zscore_direct(position)
            print(f"[DEBUG THRESHOLD] Entry z-score extracted: {entry_zscore}")
            print(f"[DEBUG THRESHOLD] Component count: {actual_component_count}")
            print(f"[DEBUG THRESHOLD] Expected next threshold: {entry_zscore - (actual_component_count * 0.5) if entry_zscore else 'UNKNOWN'}")            
            previous_zscore = self._extract_entry_zscore_from_components(position, -1)
            
            return {
                "threshold_mode": self._get_threshold_mode(),
                "scaling_mode": scaling_mode,
                "is_progressive": self._is_progressive_mode(),
                "position_direction": position_direction,
                "actual_component_count": actual_component_count,
                "base_threshold": base_long_threshold if position_direction == "LONG" else base_short_threshold,
                "current_threshold": current_threshold,
                "next_threshold": next_threshold,
                "threshold_increment": self._get_threshold_increment(),
                "percentage_threshold": self._get_percentage_threshold(),
                "entry_zscore": entry_zscore,
                "previous_entry_zscore": previous_zscore,
                "entry_zscore_available": entry_zscore is not None
            }
            
        except Exception as e:
            print(f"[SIMPLE SCALING ERROR] Failed to get threshold info: {e}")
            return {"error": str(e)}
