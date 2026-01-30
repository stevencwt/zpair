#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Target Exit Strategy - Simple Profit Taking
=================================================
Take-profit strategy for single-asset mode with fixed dollar or percentage targets.

Part of Phase 3B: Multi-Mode Trading System - Strategy Layer
Zero Deletion Compliance: New module, no existing code modified

STRATEGY LOGIC:
- Exit when P&L reaches fixed target (dollar or percentage)
- Optional max hold time
- Simple, predictable exits

CONFIGURATION:
{
  "single_asset_config": {
    "fixed_target_exit": {
      "target_type": "usd",  // "usd" or "percent"
      "target_value": 10.0,  // $10 or 1.0% (1.0 = 100%)
      "max_hold_hours": 24.0  // Optional max hold time
    }
  }
}
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta


class FixedTargetExit:
    """
    Simple fixed-target profit taking strategy.
    Exits when position reaches dollar or percentage target.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with configuration
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        sa_config = config.get('single_asset_config', {})
        exit_config = sa_config.get('fixed_target_exit', {})
        
        # Target configuration
        self.target_type = exit_config.get('target_type', 'usd')
        self.target_value = exit_config.get('target_value', 10.0)
        self.max_hold_hours = exit_config.get('max_hold_hours', None)
    
    def check_exit(self, position: Dict[str, Any], marketstate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if position should be exited based on profit target
        
        Args:
            position: Current position data with P&L
            marketstate: Current market state
            
        Returns:
            Dictionary with exit decision:
            {
                "action": "EXIT" | "HOLD",
                "reason": "explanation",
                "target_reached": True/False,
                "current_pnl": float
            }
        """
        # Get position P&L
        pnl = position.get('pnl', 0.0)
        entry_value = position.get('entry_value', 0.0)
        
        # Calculate percentage P&L if needed
        if entry_value > 0:
            pnl_percent = (pnl / entry_value) * 100
        else:
            pnl_percent = 0.0
        
        # Check dollar target
        if self.target_type == 'usd':
            if pnl >= self.target_value:
                return {
                    "action": "EXIT",
                    "reason": f"Profit target reached: ${pnl:.2f} >= ${self.target_value:.2f}",
                    "target_reached": True,
                    "current_pnl": pnl
                }
        
        # Check percentage target
        elif self.target_type == 'percent':
            target_percent = self.target_value * 100  # Convert to percentage
            if pnl_percent >= target_percent:
                return {
                    "action": "EXIT",
                    "reason": f"Profit target reached: {pnl_percent:.2f}% >= {target_percent:.2f}%",
                    "target_reached": True,
                    "current_pnl": pnl
                }
        
        # Check max hold time
        if self.max_hold_hours:
            if self._check_max_hold_time(position):
                return {
                    "action": "EXIT",
                    "reason": f"Max hold time exceeded ({self.max_hold_hours}h), P&L: ${pnl:.2f}",
                    "target_reached": False,
                    "current_pnl": pnl
                }
        
        # Continue holding
        return {
            "action": "HOLD",
            "reason": f"Target not reached, P&L: ${pnl:.2f} (target: ${self.target_value:.2f})",
            "target_reached": False,
            "current_pnl": pnl
        }
    
    def _check_max_hold_time(self, position: Dict[str, Any]) -> bool:
        """
        Check if position has exceeded max hold time
        
        Args:
            position: Position data with entry_time
            
        Returns:
            True if max hold time exceeded
        """
        if not self.max_hold_hours:
            return False
        
        entry_time = position.get('entry_time')
        if not entry_time:
            return False
        
        # Parse entry time
        if isinstance(entry_time, str):
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            except:
                return False
        elif isinstance(entry_time, datetime):
            entry_dt = entry_time
        else:
            return False
        
        # Calculate time held
        now = datetime.now(timezone.utc)
        time_held = now - entry_dt
        max_duration = timedelta(hours=self.max_hold_hours)
        
        return time_held > max_duration