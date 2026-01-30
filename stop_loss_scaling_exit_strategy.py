#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stop Loss Scaling Exit Strategy - Same Direction DCA
====================================================
Cut-loss strategy for single-asset mode with position scaling (averaging).

Part of Phase 3B: Multi-Mode Trading System - Strategy Layer
Zero Deletion Compliance: New module, no existing code modified

STRATEGY LOGIC:
- Stop loss triggered at loss threshold
- Add positions in SAME direction (DCA/averaging down or up)
- Max number of add positions
- Final stop loss after max adds reached

CONFIGURATION:
{
  "single_asset_config": {
    "stop_loss_scaling": {
      "initial_stop_loss_usd": 20.0,
      "max_add_positions": 3,
      "add_position_increment_usd": 10.0,  // Add every $10 loss
      "position_sizing_multiplier": 1.0,   // 1.0 = same size, 2.0 = double
      "final_stop_loss_usd": 100.0         // Hard stop after max adds
    }
  }
}
"""

from typing import Dict, Any, Optional


class StopLossScalingExit:
    """
    Stop-loss strategy with same-direction position scaling (DCA).
    Averages down/up in same direction to lower breakeven point.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with configuration
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        sa_config = config.get('single_asset_config', {})
        scaling_config = sa_config.get('stop_loss_scaling', {})
        
        # Stop loss configuration
        self.initial_stop_loss_usd = scaling_config.get('initial_stop_loss_usd', 20.0)
        self.max_add_positions = scaling_config.get('max_add_positions', 3)
        self.add_position_increment_usd = scaling_config.get('add_position_increment_usd', 10.0)
        self.position_sizing_multiplier = scaling_config.get('position_sizing_multiplier', 1.0)
        self.final_stop_loss_usd = scaling_config.get('final_stop_loss_usd', 100.0)
    
    def check_exit(self, position: Dict[str, Any], marketstate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if position should be exited or scaled
        
        Args:
            position: Current position data
            marketstate: Current market state
            
        Returns:
            Dictionary with exit decision:
            {
                "action": "EXIT" | "SCALE" | "HOLD",
                "reason": "explanation",
                "add_direction": "LONG" | "SHORT" | None,
                "add_size_multiplier": float
            }
        """
        pnl = position.get('pnl', 0.0)
        add_count = position.get('add_count', 0)  # Number of positions already added
        original_direction = position.get('direction', 'LONG')
        
        # Check final stop loss (hard exit)
        if pnl <= -self.final_stop_loss_usd:
            return {
                "action": "EXIT",
                "reason": f"Final stop loss hit: ${pnl:.2f} <= -${self.final_stop_loss_usd:.2f}",
                "add_direction": None,
                "add_size_multiplier": 0.0
            }
        
        # Check if max adds reached
        if add_count >= self.max_add_positions:
            # No more scaling allowed, just hold or exit
            if pnl <= -self.final_stop_loss_usd * 0.8:  # 80% of final stop
                return {
                    "action": "EXIT",
                    "reason": f"Max adds reached ({add_count}/{self.max_add_positions}), approaching final stop",
                    "add_direction": None,
                    "add_size_multiplier": 0.0
                }
            else:
                return {
                    "action": "HOLD",
                    "reason": f"Max adds reached ({add_count}/{self.max_add_positions}), holding for recovery",
                    "add_direction": None,
                    "add_size_multiplier": 0.0
                }
        
        # Calculate next add threshold
        next_add_threshold = -(self.initial_stop_loss_usd + (add_count * self.add_position_increment_usd))
        
        # Check if should add position (same direction)
        if pnl <= next_add_threshold:
            return {
                "action": "SCALE",
                "reason": f"Add position #{add_count + 1}: ${pnl:.2f} <= ${next_add_threshold:.2f}",
                "add_direction": original_direction,  # SAME direction (DCA)
                "add_size_multiplier": self.position_sizing_multiplier
            }
        
        # Continue holding
        return {
            "action": "HOLD",
            "reason": f"P&L ${pnl:.2f}, next add at ${next_add_threshold:.2f}",
            "add_direction": None,
            "add_size_multiplier": 0.0
        }