#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trailing Stop Profit Strategy - Dynamic Trailing Stop Exit Strategy with Full History Tracking
========================================================================

Dynamic trailing stop strategy that adjusts stop level as profits increase.
Uses configurable activation triggers and trailing distances with comprehensive history tracking.

Key Features:
- Activates trailing after initial profit target reached
- Tracks profit high-water mark and trails behind it
- Only moves stop level UP (never down) as profits increase
- Comprehensive trail adjustment history tracking
- Configurable fixed dollar or percentage trailing distance
- Position metadata persistence for system restart continuity

Configuration Requirements:
- Must specify activation_dollar_target OR activation_percent_target OR both
- Must specify trailing_distance_usd OR trailing_distance_percent
- Must specify use_percentage_trail to choose trailing method
- Must specify max_wait_hours and emergency_profit_threshold for safety exits

History Tracking:
- Records all trail activations and adjustments
- Maintains complete audit trail of trail level changes
- Stores configuration snapshot for analysis
- Automatic cleanup to prevent excessive history growth
"""

from trading_strategies import ExitStrategy
from typing import Dict, Any, Optional
from datetime import datetime, timezone


class TrailingStopProfit(ExitStrategy):
    """
    Trailing Stop profit-taking strategy with comprehensive history tracking.
    
    Activation Logic:
    - Config must specify activation_dollar_target OR activation_percent_target OR both
    - If both specified, use the lower value (activate sooner)
    - Strategy only activates after initial profit target is hit
    
    Trailing Logic:
    - Track profit_high_water_mark (highest profit achieved)
    - Set trail_stop_level = profit_high_water_mark - trailing_distance
    - Only move trail stop UP (never down) as profits increase
    - Exit when current profit falls below trail stop level
    
    Trail Distance Options:
    - Use trailing_distance_usd (fixed dollar) OR trailing_distance_percent
    - Config flag use_percentage_trail determines which method
    
    History Tracking:
    - Records all trail events (activation, adjustments, config changes)
    - Maintains audit trail for debugging and analysis
    - Automatic history size management
    
    Safety Features:
    - negative_trail_protection: Prevent trail from going below breakeven
    - min_trail_adjustment: Minimum movement to update trail level
    - Time-based exit: max_wait_hours with emergency_profit_threshold
    """
    
    def _load_strategy_config(self) -> Dict[str, Any]:
        """Load trailing stop strategy configuration"""
        active_profit = self.config.get("active_strategies", {}).get("take_profit", "")
        return self.config.get("take_profit_strategies", {}).get(active_profit, {})
    
    def analyze_exit_profit(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                           config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze trailing stop conditions with comprehensive history tracking.
        Uses P&L from marketstate and maintains trail state in position metadata.
        """
        try:
            # Get current P&L from marketstate (calculated fresh by main loop)
            pnl_tracking = position.get("pnl_tracking", {})
            if not pnl_tracking:
                print(f"[TRAILING_STOP] No P&L data available in position")
                return None
            
            current_pnl_usd = pnl_tracking.get("floating_pnl_usd", 0.0)
            current_pnl_percent = pnl_tracking.get("pnl_percentage", 0.0)
            calculation_source = pnl_tracking.get("calculation_source", "unknown")
            
            print(f"[TRAILING_STOP] Analyzing position P&L: ${current_pnl_usd:.2f} ({current_pnl_percent:.2f}%)")
            print(f"[TRAILING_STOP] P&L calculation source: {calculation_source}")
            
            # Get strategy configuration
            strategy_config = self.strategy_config
            if not strategy_config:
                print(f"[TRAILING_STOP] No strategy configuration found")
                return None
            
            # Extract activation triggers
            activation_dollar = strategy_config.get("activation_dollar_target")
            activation_percent = strategy_config.get("activation_percent_target")
            
            # Validate that at least one activation trigger is specified
            if activation_dollar is None and activation_percent is None:
                print(f"[TRAILING_STOP] Error: Must specify activation_dollar_target OR activation_percent_target")
                return None
            
            # Validate positive values
            if activation_dollar is not None and activation_dollar <= 0:
                print(f"[TRAILING_STOP] Error: activation_dollar_target must be positive, got {activation_dollar}")
                return None
            
            if activation_percent is not None and activation_percent <= 0:
                print(f"[TRAILING_STOP] Error: activation_percent_target must be positive, got {activation_percent}")
                return None
            
            print(f"[TRAILING_STOP] Activation triggers - Dollar: ${activation_dollar}, Percent: {activation_percent}%")
            
            # Get or initialize trail tracking metadata
            trail_tracking = position.get("trail_tracking", {})
            is_active = trail_tracking.get("is_active", False)
            profit_high_water_mark = trail_tracking.get("profit_high_water_mark", 0.0)
            current_trail_stop_level = trail_tracking.get("current_trail_stop_level", 0.0)
            
            print(f"[TRAILING_STOP] Trail state - Active: {is_active}, High: ${profit_high_water_mark:.2f}, Stop: ${current_trail_stop_level:.2f}")
            
            # Check if trailing should be activated
            if not is_active:
                activation_met = False
                activation_reason = ""
                
                # Check activation triggers (use lower value - activate sooner)
                if activation_dollar is not None and current_pnl_usd >= activation_dollar:
                    activation_met = True
                    activation_reason = f"Dollar activation: ${current_pnl_usd:.2f} >= ${activation_dollar}"
                elif activation_percent is not None and current_pnl_percent >= activation_percent:
                    activation_met = True
                    activation_reason = f"Percent activation: {current_pnl_percent:.2f}% >= {activation_percent}%"
                
                if not activation_met:
                    print(f"[TRAILING_STOP] Trailing not activated yet")
                    
                    # Check time-based exit even without trailing active
                    time_exit = self._check_time_based_exit(position, current_pnl_usd, strategy_config)
                    if time_exit:
                        return time_exit
                    
                    # Log progress toward activation
                    remaining_info = []
                    if activation_dollar is not None:
                        remaining_dollar = activation_dollar - current_pnl_usd
                        remaining_info.append(f"${remaining_dollar:.2f} to dollar activation")
                    if activation_percent is not None:
                        remaining_percent = activation_percent - current_pnl_percent
                        remaining_info.append(f"{remaining_percent:.2f}% to percent activation")
                    
                    print(f"[TRAILING_STOP] {' | '.join(remaining_info)}")
                    return None
                
                # Activate trailing
                print(f"[TRAILING_STOP] Activating trailing: {activation_reason}")
                is_active = True
                profit_high_water_mark = current_pnl_usd
                
                # Calculate initial trail stop level
                initial_trail_level, trailing_distance_used = self._calculate_trail_stop_level(
                    profit_high_water_mark, strategy_config
                )
                current_trail_stop_level = initial_trail_level
                
                # Initialize trail tracking with complete metadata
                position["trail_tracking"] = {
                    "is_active": True,
                    "profit_high_water_mark": profit_high_water_mark,
                    "current_trail_stop_level": current_trail_stop_level,
                    "activation_time": datetime.now(timezone.utc).isoformat(),
                    "last_trail_update": datetime.now(timezone.utc).isoformat(),
                    "trail_history": [],
                    "strategy_config_snapshot": {
                        "activation_dollar_target": activation_dollar,
                        "activation_percent_target": activation_percent,
                        "trailing_distance_usd": strategy_config.get("trailing_distance_usd"),
                        "trailing_distance_percent": strategy_config.get("trailing_distance_percent"),
                        "use_percentage_trail": strategy_config.get("use_percentage_trail", False),
                        "min_trail_adjustment": strategy_config.get("min_trail_adjustment", 0.25),
                        "negative_trail_protection": strategy_config.get("negative_trail_protection", True)
                    }
                }
                
                # Record activation in history
                self._update_trail_history(
                    position, "activation",
                    current_pnl_usd,
                    trail_stop_level=current_trail_stop_level,
                    trailing_distance=trailing_distance_used,
                    trailing_method="percentage" if strategy_config.get("use_percentage_trail", False) else "fixed_dollar",
                    reason=activation_reason
                )
                
                print(f"[TRAILING_STOP] Initial trail stop set at ${current_trail_stop_level:.2f}")
            
            # Update high-water mark and trail level if profit increased
            if current_pnl_usd > profit_high_water_mark:
                print(f"[TRAILING_STOP] New profit high: ${current_pnl_usd:.2f} (was ${profit_high_water_mark:.2f})")
                
                # Get minimum trail adjustment to avoid excessive updates
                min_trail_adjustment = strategy_config.get("min_trail_adjustment", 0.25)
                profit_increase = current_pnl_usd - profit_high_water_mark
                
                if profit_increase >= min_trail_adjustment:
                    # Calculate new trail stop level
                    new_trail_level, trailing_distance_used = self._calculate_trail_stop_level(
                        current_pnl_usd, strategy_config
                    )
                    
                    # Only move trail stop UP (never down)
                    if new_trail_level > current_trail_stop_level:
                        # Store previous values for history
                        previous_high_water_mark = profit_high_water_mark
                        previous_trail_level = current_trail_stop_level
                        
                        # Update high-water mark and trail level
                        profit_high_water_mark = current_pnl_usd
                        current_trail_stop_level = new_trail_level
                        
                        # Update position metadata
                        position["trail_tracking"].update({
                            "profit_high_water_mark": profit_high_water_mark,
                            "current_trail_stop_level": current_trail_stop_level,
                            "last_trail_update": datetime.now(timezone.utc).isoformat()
                        })
                        
                        # Record trail adjustment in history
                        self._update_trail_history(
                            position, "trail_adjustment",
                            current_pnl_usd,
                            previous_profit_high=previous_high_water_mark,
                            new_profit_high=profit_high_water_mark,
                            previous_trail_level=previous_trail_level,
                            new_trail_stop_level=current_trail_stop_level,
                            trailing_distance=trailing_distance_used,
                            profit_increase=profit_increase,
                            trailing_method="percentage" if strategy_config.get("use_percentage_trail", False) else "fixed_dollar",
                            reason=f"Profit increased from ${previous_high_water_mark:.2f} to ${current_pnl_usd:.2f}"
                        )
                        
                        print(f"[TRAILING_STOP] Trail stop updated to ${current_trail_stop_level:.2f}")
                    else:
                        print(f"[TRAILING_STOP] Trail stop unchanged (would move down: ${new_trail_level:.2f} <= ${current_trail_stop_level:.2f})")
                        
                        # Record rejected adjustment in history
                        self._update_trail_history(
                            position, "adjustment_rejected",
                            current_pnl_usd,
                            calculated_trail_level=new_trail_level,
                            current_trail_level=current_trail_stop_level,
                            reason="Calculated trail level would move down - rejected"
                        )
                else:
                    print(f"[TRAILING_STOP] Profit increase ${profit_increase:.2f} below min adjustment ${min_trail_adjustment}")
                    
                    # Record minimal adjustment in history
                    self._update_trail_history(
                        position, "minimal_adjustment",
                        current_pnl_usd,
                        profit_increase=profit_increase,
                        min_adjustment_required=min_trail_adjustment,
                        reason=f"Profit increase below minimum adjustment threshold"
                    )
            
            # Check if current profit has fallen below trail stop
            if current_pnl_usd <= current_trail_stop_level:
                print(f"[TRAILING_STOP] Trail stop triggered - generating exit signal")
                
                # Record trail stop trigger in history
                self._update_trail_history(
                    position, "trail_stop_triggered",
                    current_pnl_usd,
                    trail_stop_level=current_trail_stop_level,
                    profit_high_water_mark=profit_high_water_mark,
                    drawdown_from_peak=profit_high_water_mark - current_pnl_usd,
                    reason=f"Current profit ${current_pnl_usd:.2f} fell below trail stop ${current_trail_stop_level:.2f}"
                )
                
                return {
                    "aggregate_id": position["aggregate_id"],
                    "exit_type": "trailing_stop",
                    "reason": f"Profit fell below trail: ${current_pnl_usd:.2f} <= ${current_trail_stop_level:.2f} (peak was ${profit_high_water_mark:.2f})",
                    "priority": "high",
                    "pnl_source": "fresh_marketstate_calculation",
                    "trailing_stop_data": {
                        "activation_dollar": activation_dollar,
                        "activation_percent": activation_percent,
                        "profit_high_water_mark": profit_high_water_mark,
                        "trail_stop_level": current_trail_stop_level,
                        "final_pnl": current_pnl_usd,
                        "drawdown_from_peak": profit_high_water_mark - current_pnl_usd,
                        "total_trail_adjustments": len(position.get("trail_tracking", {}).get("trail_history", [])),
                        "strategy": "trailing_stop_profit"
                    }
                }
            
            # Check time-based exit conditions
            time_exit = self._check_time_based_exit(position, current_pnl_usd, strategy_config)
            if time_exit:
                # Record time-based exit in history
                self._update_trail_history(
                    position, "time_based_exit",
                    current_pnl_usd,
                    reason="Time-based emergency exit triggered"
                )
                return time_exit
            
            # No exit conditions met
            trail_buffer = current_pnl_usd - current_trail_stop_level
            print(f"[TRAILING_STOP] Trail active - buffer: ${trail_buffer:.2f} above stop level")
            
            # Record status check in history (every 10th call to avoid spam)
            if not hasattr(self, '_status_check_counter'):
                self._status_check_counter = 0
            self._status_check_counter += 1
            
            if self._status_check_counter % 10 == 0:
                self._update_trail_history(
                    position, "status_check",
                    current_pnl_usd,
                    trail_buffer=trail_buffer,
                    trail_stop_level=current_trail_stop_level,
                    reason=f"Periodic status check - trail buffer ${trail_buffer:.2f}"
                )
            
            return None
            
        except Exception as e:
            print(f"[TRAILING_STOP ERROR] Profit exit analysis failed: {e}")
            return None
    
    def _update_trail_history(self, position: Dict[str, Any], event_type: str, 
                             current_pnl: float, **kwargs) -> None:
        """
        Add entry to trail adjustment history with comprehensive tracking.
        Automatically manages history size to prevent excessive growth.
        """
        try:
            trail_tracking = position.get("trail_tracking", {})
            history = trail_tracking.get("trail_history", [])
            
            # Create comprehensive history entry
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "profit_level": current_pnl,
                "sequence_number": len(history) + 1,
                **kwargs  # Additional event-specific data
            }
            
            # Add to history
            history.append(entry)
            
            # Limit history size (keep last 100 entries for comprehensive tracking)
            # This provides sufficient history for analysis while preventing excessive growth
            max_history_entries = 100
            if len(history) > max_history_entries:
                # Keep recent entries and a few early entries for reference
                early_entries = history[:5]  # Keep first 5 entries
                recent_entries = history[-(max_history_entries-5):]  # Keep last 95 entries
                history = early_entries + recent_entries
                
                # Update sequence numbers
                for i, entry in enumerate(history):
                    entry["sequence_number"] = i + 1
                
                print(f"[TRAILING_STOP] Trail history trimmed to {len(history)} entries")
            
            # Update position metadata
            trail_tracking["trail_history"] = history
            position["trail_tracking"] = trail_tracking
            
            print(f"[TRAILING_STOP] Trail history updated: {event_type} (entry #{entry['sequence_number']})")
            
        except Exception as e:
            print(f"[TRAILING_STOP ERROR] Failed to update trail history: {e}")
    
    def _calculate_trail_stop_level(self, profit_high_water_mark: float, 
                                   strategy_config: Dict[str, Any]) -> tuple[float, float]:
        """
        Calculate trail stop level based on configuration.
        Returns (trail_stop_level, trailing_distance_used) as dollar amounts.
        """
        try:
            # Get trailing distance configuration
            trailing_distance_usd = strategy_config.get("trailing_distance_usd")
            trailing_distance_percent = strategy_config.get("trailing_distance_percent")
            use_percentage_trail = strategy_config.get("use_percentage_trail", False)
            negative_trail_protection = strategy_config.get("negative_trail_protection", True)
            
            # Validate trailing distance configuration
            if trailing_distance_usd is None and trailing_distance_percent is None:
                print(f"[TRAILING_STOP] Error: Must specify trailing_distance_usd OR trailing_distance_percent")
                return 0.0, 0.0
            
            # Calculate trail stop level
            if use_percentage_trail and trailing_distance_percent is not None:
                # Percentage-based trailing
                if trailing_distance_percent <= 0 or trailing_distance_percent >= 100:
                    print(f"[TRAILING_STOP] Error: trailing_distance_percent must be between 0 and 100, got {trailing_distance_percent}")
                    return 0.0, 0.0
                
                trail_distance = profit_high_water_mark * (trailing_distance_percent / 100.0)
                trail_stop_level = profit_high_water_mark - trail_distance
                
                print(f"[TRAILING_STOP] Percentage trail: {trailing_distance_percent}% of ${profit_high_water_mark:.2f} = ${trail_distance:.2f}")
                return trail_stop_level, trail_distance
                
            else:
                # Fixed dollar trailing
                if trailing_distance_usd is None or trailing_distance_usd <= 0:
                    print(f"[TRAILING_STOP] Error: trailing_distance_usd must be positive, got {trailing_distance_usd}")
                    return 0.0, 0.0
                
                trail_stop_level = profit_high_water_mark - trailing_distance_usd
                
                print(f"[TRAILING_STOP] Fixed dollar trail: ${profit_high_water_mark:.2f} - ${trailing_distance_usd:.2f} = ${trail_stop_level:.2f}")
                
                # Apply negative trail protection
                if negative_trail_protection and trail_stop_level < 0.0:
                    print(f"[TRAILING_STOP] Negative trail protection: adjusting ${trail_stop_level:.2f} to $0.00")
                    trail_stop_level = 0.0
                
                return trail_stop_level, trailing_distance_usd
            
        except Exception as e:
            print(f"[TRAILING_STOP ERROR] Trail calculation failed: {e}")
            return 0.0, 0.0
    
    def _check_time_based_exit(self, position: Dict[str, Any], current_pnl: float, 
                              strategy_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check time-based emergency exit conditions.
        If time limit reached AND profit >= emergency threshold → immediate exit
        """
        try:
            max_wait_hours = strategy_config.get("max_wait_hours")
            emergency_profit_threshold = strategy_config.get("emergency_profit_threshold")
            
            if max_wait_hours is None or emergency_profit_threshold is None:
                print(f"[TRAILING_STOP] Time-based exit disabled: missing max_wait_hours or emergency_profit_threshold")
                return None
            
            # Validate positive values
            if max_wait_hours <= 0:
                print(f"[TRAILING_STOP] Error: max_wait_hours must be positive, got {max_wait_hours}")
                return None
            
            if emergency_profit_threshold < 0:
                print(f"[TRAILING_STOP] Error: emergency_profit_threshold must be non-negative, got {emergency_profit_threshold}")
                return None
            
            # Calculate hold time
            timestamps = position.get("timestamps", {})
            initial_entry_str = timestamps.get("initial_entry", "")
            
            if not initial_entry_str:
                print(f"[TRAILING_STOP] No initial entry timestamp available")
                return None
            
            try:
                initial_entry = datetime.fromisoformat(initial_entry_str.replace('Z', '+00:00'))
                current_time = datetime.now(timezone.utc)
                hold_hours = (current_time - initial_entry).total_seconds() / 3600
                
                print(f"[TRAILING_STOP] Time analysis - Hold: {hold_hours:.1f}h, Max: {max_wait_hours}h, P&L: ${current_pnl:.2f}, Emergency: ${emergency_profit_threshold}")
                
                if hold_hours >= max_wait_hours:
                    if current_pnl >= emergency_profit_threshold:
                        print(f"[TRAILING_STOP] Time limit reached with emergency profit - generating exit signal")
                        return {
                            "aggregate_id": position["aggregate_id"],
                            "exit_type": "time_based_emergency_exit",
                            "reason": f"Time limit reached ({hold_hours:.1f}h >= {max_wait_hours}h) with emergency profit ${current_pnl:.2f} >= ${emergency_profit_threshold}",
                            "priority": "medium",
                            "pnl_source": "fresh_marketstate_calculation",
                            "trailing_stop_data": {
                                "max_wait_hours": max_wait_hours,
                                "actual_hold_hours": hold_hours,
                                "emergency_profit_threshold": emergency_profit_threshold,
                                "achieved_pnl": current_pnl,
                                "strategy": "trailing_stop_profit_time_exit"
                            }
                        }
                    else:
                        print(f"[TRAILING_STOP] Time limit reached but profit ${current_pnl:.2f} below emergency threshold ${emergency_profit_threshold}")
                        return None
                else:
                    remaining_hours = max_wait_hours - hold_hours
                    print(f"[TRAILING_STOP] Time limit not reached - {remaining_hours:.1f}h remaining")
                    return None
                
            except Exception as e:
                print(f"[TRAILING_STOP] Time calculation failed: {e}")
                return None
                
        except Exception as e:
            print(f"[TRAILING_STOP ERROR] Time-based exit check failed: {e}")
            return None
    
    def get_trail_summary(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of trailing stop performance for analysis.
        Useful for external tools and debugging.
        """
        try:
            trail_tracking = position.get("trail_tracking", {})
            history = trail_tracking.get("trail_history", [])
            
            if not history:
                return {"status": "no_trail_history"}
            
            # Count different event types
            event_counts = {}
            for entry in history:
                event_type = entry.get("event_type", "unknown")
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Find key events
            activation_event = next((e for e in history if e.get("event_type") == "activation"), None)
            last_adjustment = next((e for e in reversed(history) if e.get("event_type") == "trail_adjustment"), None)
            trigger_event = next((e for e in history if e.get("event_type") == "trail_stop_triggered"), None)
            
            summary = {
                "status": "active" if trail_tracking.get("is_active") else "inactive",
                "total_events": len(history),
                "event_counts": event_counts,
                "profit_high_water_mark": trail_tracking.get("profit_high_water_mark", 0.0),
                "current_trail_stop_level": trail_tracking.get("current_trail_stop_level", 0.0),
                "activation_time": activation_event.get("timestamp") if activation_event else None,
                "last_adjustment_time": last_adjustment.get("timestamp") if last_adjustment else None,
                "trail_adjustments": event_counts.get("trail_adjustment", 0),
                "triggered": trigger_event is not None
            }
            
            return summary
            
        except Exception as e:
            print(f"[TRAILING_STOP ERROR] Failed to generate trail summary: {e}")
            return {"status": "error", "error": str(e)}
    
    def analyze_exit_loss(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                         config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Not implemented for profit-only strategy"""
        return None
