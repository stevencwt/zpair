#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Let Profit Run Strategy - Mean-Reversion Profit-Taking Exit Strategy
========================================================================

Mean-reversion profit-taking strategy that waits for price to cross back to zero zscore.
Uses configurable minimum profit thresholds and time-based safety exits.

Key Features:
- Waits for zscore to cross zero for mean-reversion exit
- LONG positions: Exit when zscore crosses from negative to above zero
- SHORT positions: Exit when zscore crosses from positive to below zero
- Requires minimum profit threshold before considering zscore exit
- Time-based safety exit with satisfactory profit threshold

Configuration Requirements:
- Must specify min_dollar_target OR min_percent_target OR both
- Must specify max_wait_hours and satisfactory_profit_threshold_usd
"""

from trading_strategies import ExitStrategy
from typing import Dict, Any, Optional
from datetime import datetime, timezone


class LetProfitRun(ExitStrategy):
    """
    Let Profit Run strategy for mean-reversion exits.
    
    Profit Target Logic:
    - Config must specify min_dollar_target OR min_percent_target OR both
    - If both specified, use the larger value (more aggressive)
    - Must meet minimum profit before considering zscore exit
    
    Zscore-Based Exit Logic:
    - For LONG positions: Exit when zscore crosses from negative to above zero
    - For SHORT positions: Exit when zscore crosses from positive to below zero
    - Both conditions required: Profit target met AND zscore crossed zero
    
    Time-Based Safety Exit:
    - Config must specify max_wait_hours and satisfactory_profit_threshold_usd
    - If time limit reached AND profit >= satisfactory threshold → immediate exit
    """
    
    def _load_strategy_config(self) -> Dict[str, Any]:
        """Load let profit run strategy configuration"""
        active_profit = self.config.get("active_strategies", {}).get("take_profit", "")
        return self.config.get("take_profit_strategies", {}).get(active_profit, {})
    
    def analyze_exit_profit(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                           config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze let-profit-run conditions.
        Uses P&L from marketstate and zscore analysis for mean-reversion exits.
        """
        try:
            # Get current P&L from marketstate (calculated fresh by main loop)
            pnl_tracking = position.get("pnl_tracking", {})
            if not pnl_tracking:
                print(f"[LET_PROFIT_RUN] No P&L data available in position")
                return None
            
            current_pnl_usd = pnl_tracking.get("floating_pnl_usd", 0.0)
            current_pnl_percent = pnl_tracking.get("pnl_percentage", 0.0)
            calculation_source = pnl_tracking.get("calculation_source", "unknown")
            
            print(f"[LET_PROFIT_RUN] Analyzing position P&L: ${current_pnl_usd:.2f} ({current_pnl_percent:.2f}%)")
            print(f"[LET_PROFIT_RUN] P&L calculation source: {calculation_source}")
            
            # Get position direction
            aggregated_metrics = position.get("aggregated_metrics", {})
            position_direction = aggregated_metrics.get("direction", "")
            
            if not position_direction:
                print(f"[LET_PROFIT_RUN] No position direction available")
                return None
            
            # Get current zscore from primary timeframe
            regression = marketstate.get("regression", {})
            current_zscore = regression.get("zscore", 0.0)
            
            print(f"[LET_PROFIT_RUN] Position: {position_direction}, Current zscore: {current_zscore:.3f}")
            
            # Get strategy configuration
            strategy_config = self.strategy_config
            if not strategy_config:
                print(f"[LET_PROFIT_RUN] No strategy configuration found")
                return None
            
            # Extract and validate profit targets
            min_dollar_target = strategy_config.get("min_dollar_target")
            min_percent_target = strategy_config.get("min_percent_target")
            
            # Validate that at least one target is specified
            if min_dollar_target is None and min_percent_target is None:
                print(f"[LET_PROFIT_RUN] Error: Must specify min_dollar_target OR min_percent_target")
                return None
            
            # Validate positive values
            if min_dollar_target is not None and min_dollar_target <= 0:
                print(f"[LET_PROFIT_RUN] Error: min_dollar_target must be positive, got {min_dollar_target}")
                return None
            
            if min_percent_target is not None and min_percent_target <= 0:
                print(f"[LET_PROFIT_RUN] Error: min_percent_target must be positive, got {min_percent_target}")
                return None
            
            print(f"[LET_PROFIT_RUN] Min targets - Dollar: ${min_dollar_target}, Percent: {min_percent_target}%")
            
            # Check if minimum profit threshold is met (use larger value if both specified)
            profit_threshold_met = False
            threshold_reason = ""
            
            if min_dollar_target is not None and min_percent_target is not None:
                # Both specified - use larger value (more aggressive)
                dollar_met = current_pnl_usd >= min_dollar_target
                percent_met = current_pnl_percent >= min_percent_target
                
                if dollar_met and percent_met:
                    profit_threshold_met = True
                    threshold_reason = f"Both thresholds met: ${current_pnl_usd:.2f} >= ${min_dollar_target} AND {current_pnl_percent:.2f}% >= {min_percent_target}%"
                elif dollar_met:
                    # Only dollar met - check if it's the higher threshold
                    estimated_dollar_at_percent = (min_percent_target / 100.0) * 1000.0  # Assume $1000 base for comparison
                    if min_dollar_target >= estimated_dollar_at_percent:
                        profit_threshold_met = True
                        threshold_reason = f"Higher dollar threshold met: ${current_pnl_usd:.2f} >= ${min_dollar_target}"
                elif percent_met:
                    # Only percent met - check if it's the higher threshold
                    estimated_percent_at_dollar = (min_dollar_target / 1000.0) * 100.0  # Assume $1000 base for comparison
                    if min_percent_target >= estimated_percent_at_dollar:
                        profit_threshold_met = True
                        threshold_reason = f"Higher percent threshold met: {current_pnl_percent:.2f}% >= {min_percent_target}%"
            
            elif min_dollar_target is not None:
                # Only dollar target specified
                if current_pnl_usd >= min_dollar_target:
                    profit_threshold_met = True
                    threshold_reason = f"Dollar threshold met: ${current_pnl_usd:.2f} >= ${min_dollar_target}"
            
            elif min_percent_target is not None:
                # Only percent target specified
                if current_pnl_percent >= min_percent_target:
                    profit_threshold_met = True
                    threshold_reason = f"Percent threshold met: {current_pnl_percent:.2f}% >= {min_percent_target}%"
            
            if not profit_threshold_met:
                print(f"[LET_PROFIT_RUN] Minimum profit threshold not met")
                
                # Check time-based exit even if profit threshold not met
                time_exit = self._check_time_based_exit(position, current_pnl_usd, strategy_config)
                if time_exit:
                    return time_exit
                
                # Log remaining profit needed
                remaining_info = []
                if min_dollar_target is not None:
                    remaining_dollar = min_dollar_target - current_pnl_usd
                    remaining_info.append(f"${remaining_dollar:.2f} to min dollar target")
                if min_percent_target is not None:
                    remaining_percent = min_percent_target - current_pnl_percent
                    remaining_info.append(f"{remaining_percent:.2f}% to min percent target")
                
                print(f"[LET_PROFIT_RUN] {' | '.join(remaining_info)}")
                return None
            
            print(f"[LET_PROFIT_RUN] Profit threshold met: {threshold_reason}")
            
            # Check zscore mean-reversion conditions
            zscore_exit_met = False
            zscore_reason = ""
            
            if position_direction == "LONG":
                # LONG position: Exit when zscore crosses from negative to above zero
                if current_zscore > 0.0:
                    zscore_exit_met = True
                    zscore_reason = f"LONG mean reversion: zscore {current_zscore:.3f} crossed above zero"
                else:
                    zscore_reason = f"LONG waiting for reversion: zscore {current_zscore:.3f} still negative"
            
            elif position_direction == "SHORT":
                # SHORT position: Exit when zscore crosses from positive to below zero
                if current_zscore < 0.0:
                    zscore_exit_met = True
                    zscore_reason = f"SHORT mean reversion: zscore {current_zscore:.3f} crossed below zero"
                else:
                    zscore_reason = f"SHORT waiting for reversion: zscore {current_zscore:.3f} still positive"
            
            print(f"[LET_PROFIT_RUN] Zscore analysis: {zscore_reason}")
            
            # Both conditions must be met for exit
            if profit_threshold_met and zscore_exit_met:
                print(f"[LET_PROFIT_RUN] Both conditions met - generating exit signal")
                return {
                    "aggregate_id": position["aggregate_id"],
                    "exit_type": "mean_reversion_profit",
                    "reason": f"Mean reversion exit: {threshold_reason} AND {zscore_reason}",
                    "priority": "high",
                    "pnl_source": "fresh_marketstate_calculation",
                    "let_profit_run_data": {
                        "min_dollar_target": min_dollar_target,
                        "min_percent_target": min_percent_target,
                        "achieved_dollar": current_pnl_usd,
                        "achieved_percent": current_pnl_percent,
                        "position_direction": position_direction,
                        "exit_zscore": current_zscore,
                        "strategy": "let_profit_run"
                    }
                }
            
            # Check time-based exit conditions
            time_exit = self._check_time_based_exit(position, current_pnl_usd, strategy_config)
            if time_exit:
                return time_exit
            
            # Conditions not met - log status
            if profit_threshold_met:
                print(f"[LET_PROFIT_RUN] Profit threshold met but waiting for zscore reversion")
            
            return None
            
        except Exception as e:
            print(f"[LET_PROFIT_RUN ERROR] Profit exit analysis failed: {e}")
            return None
    
    def _check_time_based_exit(self, position: Dict[str, Any], current_pnl: float, 
                              strategy_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check time-based safety exit conditions.
        If time limit reached AND profit >= satisfactory threshold → immediate exit
        """
        try:
            max_wait_hours = strategy_config.get("max_wait_hours")
            satisfactory_profit_threshold = strategy_config.get("satisfactory_profit_threshold_usd")
            
            if max_wait_hours is None or satisfactory_profit_threshold is None:
                print(f"[LET_PROFIT_RUN] Time-based exit disabled: missing max_wait_hours or satisfactory_profit_threshold_usd")
                return None
            
            # Validate positive values
            if max_wait_hours <= 0:
                print(f"[LET_PROFIT_RUN] Error: max_wait_hours must be positive, got {max_wait_hours}")
                return None
            
            if satisfactory_profit_threshold < 0:
                print(f"[LET_PROFIT_RUN] Error: satisfactory_profit_threshold_usd must be non-negative, got {satisfactory_profit_threshold}")
                return None
            
            # Calculate hold time
            timestamps = position.get("timestamps", {})
            initial_entry_str = timestamps.get("initial_entry", "")
            
            if not initial_entry_str:
                print(f"[LET_PROFIT_RUN] No initial entry timestamp available")
                return None
            
            try:
                initial_entry = datetime.fromisoformat(initial_entry_str.replace('Z', '+00:00'))
                current_time = datetime.now(timezone.utc)
                hold_hours = (current_time - initial_entry).total_seconds() / 3600
                
                print(f"[LET_PROFIT_RUN] Time analysis - Hold: {hold_hours:.1f}h, Max: {max_wait_hours}h, P&L: ${current_pnl:.2f}, Satisfactory: ${satisfactory_profit_threshold}")
                
                if hold_hours >= max_wait_hours:
                    if current_pnl >= satisfactory_profit_threshold:
                        print(f"[LET_PROFIT_RUN] Time limit reached with satisfactory profit - generating exit signal")
                        return {
                            "aggregate_id": position["aggregate_id"],
                            "exit_type": "time_based_satisfactory_exit",
                            "reason": f"Time limit reached ({hold_hours:.1f}h >= {max_wait_hours}h) with satisfactory profit ${current_pnl:.2f} >= ${satisfactory_profit_threshold}",
                            "priority": "medium",
                            "pnl_source": "fresh_marketstate_calculation",
                            "let_profit_run_data": {
                                "max_wait_hours": max_wait_hours,
                                "actual_hold_hours": hold_hours,
                                "satisfactory_profit_threshold": satisfactory_profit_threshold,
                                "achieved_pnl": current_pnl,
                                "strategy": "let_profit_run_time_exit"
                            }
                        }
                    else:
                        print(f"[LET_PROFIT_RUN] Time limit reached but profit ${current_pnl:.2f} below satisfactory threshold ${satisfactory_profit_threshold}")
                        return None
                else:
                    remaining_hours = max_wait_hours - hold_hours
                    print(f"[LET_PROFIT_RUN] Time limit not reached - {remaining_hours:.1f}h remaining")
                    return None
                
            except Exception as e:
                print(f"[LET_PROFIT_RUN] Time calculation failed: {e}")
                return None
                
        except Exception as e:
            print(f"[LET_PROFIT_RUN ERROR] Time-based exit check failed: {e}")
            return None
    
    def analyze_exit_loss(self, position: Dict[str, Any], marketstate: Dict[str, Any], 
                         config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Not implemented for profit-only strategy"""
        return None