#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position Aggregator - Multi-Component Position Management
Aggregates multiple component trades into unified position view
UNIFIED VERSION: Supports both single and multi-component positions
STAGE 1 UPDATED: P&L storage removed, separate calculation method added
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import date
from datetime import datetime, timezone
from debug_utils import get_local_time  # Required for SGT synchronization

class PositionAggregator:
    """
    Aggregates multiple component trades into unified position view.
    Supports both single-position (force_single=True) and multi-position modes.
    Stateless utility class for position calculations.
    
    STAGE 1 CHANGES:
    - Removed P&L storage from aggregate_position()
    - Added separate calculate_current_pnl_for_marketstate() method
    - P&L calculations now only for marketstate display, never stored
    """
    
    def __init__(self):
        """Initialize position aggregator."""
        pass
    
    def aggregate_position(
        self,
        component_trades: List[Dict[str, Any]],
        current_prices: Dict[str, float],
        force_single: bool = False
    ) -> Dict[str, Any]:
        """
        Combines component trades into aggregated position.
        FIXED: force_single only affects position counting for limits, NOT component storage
        
        Args:
            component_trades: List of trade dictionaries
            current_prices: {'primary': float, 'reference': float} (for validation only)
            force_single: If True, counts as 1 position for concurrent limits (but stores all trades)
            
        Returns:
            Aggregated position dictionary with schema version 2.0.0 (NO P&L tracking)
            
        Raises:
            ValueError: If component_trades is empty, has mixed directions, or zero notional
        """
        if not component_trades:
            raise ValueError("Cannot aggregate empty trade list")
        
        # REMOVED: The buggy force_single filtering logic
        # force_single should NOT filter out component trades
        # It should only affect how we count positions for concurrent limits
        
        # Validate all trades are same direction
        directions = set(self._get_direction(trade) for trade in component_trades)
        if len(directions) > 1:
            raise ValueError(f"Cannot aggregate trades with mixed directions: {directions}")
        
        direction = self._get_direction(component_trades[0])
        
        # FIXED: Always aggregate ALL component trades regardless of force_single
        total_qty_a = sum(trade.get('quantity_a', 0.0) for trade in component_trades)
        total_qty_b = sum(trade.get('quantity_b', 0.0) for trade in component_trades)
        total_notional = sum(trade.get('total_position_size_usd', 0.0) for trade in component_trades)
        
        if total_notional == 0:
            raise ValueError("Cannot aggregate position with zero notional value")
        
        # Calculate position-weighted average entry prices
        weighted_price_a = sum(
            trade.get('entry_price_a', 0.0) * trade.get('total_position_size_usd', 0.0)
            for trade in component_trades
        ) / total_notional
        
        weighted_price_b = sum(
            trade.get('entry_price_b', 0.0) * trade.get('total_position_size_usd', 0.0)
            for trade in component_trades
        ) / total_notional
        
        # Extract timestamps with fallback to current time
        current_time = get_local_time().isoformat()
        timestamps = []
        for trade in component_trades:
            ts = trade.get('timestamp_open', '')
            if ts:
                try:
                    timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                except:
                    pass
        
        timestamps.sort()
        initial_timestamp = timestamps[0].isoformat() if timestamps else current_time
        recent_timestamp = timestamps[-1].isoformat() if timestamps else current_time
        
        # Generate aggregate ID (always generate, even for single trades)
        aggregate_id = self._generate_aggregate_id(component_trades)
        
        # Extract extreme tracking from most recent trade
        extreme_tracking = component_trades[-1].get('extreme_tracking_state', {})
        
        # FIXED: Separate actual component count from position limit counting
        actual_component_count = len(component_trades)
        position_count_for_limits = 1 if force_single else actual_component_count
        
        # STAGE 1 CHANGE: Return position WITHOUT P&L tracking section
        return {
            'schema_version': '2.0.0',  # Schema version for unified storage
            'aggregate_id': aggregate_id,
            'status': 'OPEN',
            'created_at': initial_timestamp,
            'last_updated': current_time,
            
            'component_trades': component_trades,  # FIXED: Store ALL trades
            
            'aggregated_metrics': {
                'total_quantity_a': total_qty_a,          # FIXED: Sum of ALL trades
                'total_quantity_b': total_qty_b,          # FIXED: Sum of ALL trades
                'avg_entry_price_a': weighted_price_a,    # FIXED: Weighted avg of ALL trades
                'avg_entry_price_b': weighted_price_b,    # FIXED: Weighted avg of ALL trades
                'total_notional_usd': total_notional,     # FIXED: Sum of ALL trades
                'position_count': position_count_for_limits,  # For concurrent limit checking only
                'actual_component_count': actual_component_count,  # NEW: Actual trade count
                'initial_entry_zscore': component_trades[0].get('entry_zscore', 0.0),
                'most_recent_entry_zscore': component_trades[-1].get('entry_zscore', 0.0),
                'direction': direction,
                'mode': 'single_position' if force_single else 'multi_position'
            },
            
            'timestamps': {
                'initial_entry': initial_timestamp,
                'most_recent_add': recent_timestamp
            },
            
            # STAGE 1 REMOVED: pnl_tracking section completely eliminated
            # P&L calculations moved to separate method for marketstate only
            
            'extreme_tracking': extreme_tracking,
            
            # Metadata for unified system
            'unified_metadata': {
                'is_single_position_mode': force_single,
                'original_trade_count': len(component_trades),  # FIXED: Actual count
                'aggregation_timestamp': current_time,
                'storage_format': 'unified_v2_no_pnl_fixed_scaling'  # Updated to indicate fix
            }
        }
    
    def calculate_current_pnl_for_marketstate(
        self, 
        static_position: Dict[str, Any], 
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        STAGE 1 NEW METHOD: Calculate fresh P&L for marketstate display only.
        This P&L data is NEVER stored to persistent files.
        
        Args:
            static_position: Position data from persistent storage (without P&L)
            current_prices: {'primary': float, 'reference': float}
            
        Returns:
            P&L tracking dictionary for marketstate inclusion only
        """
        try:
            # Extract static position data
            metrics = static_position.get('aggregated_metrics', {})
            total_qty_a = metrics.get('total_quantity_a', 0.0)
            total_qty_b = metrics.get('total_quantity_b', 0.0)
            avg_entry_price_a = metrics.get('avg_entry_price_a', 0.0)
            avg_entry_price_b = metrics.get('avg_entry_price_b', 0.0)
            total_notional = metrics.get('total_notional_usd', 0.0)
            
            # Validate current prices
            if current_prices.get('primary', 0) <= 0 or current_prices.get('reference', 0) <= 0:
                raise ValueError(f"Invalid current prices for P&L calculation: {current_prices}")
            
            # Calculate fresh P&L from static data + current prices
            current_price_a = current_prices['primary']
            current_price_b = current_prices['reference']
            
            pnl_a = total_qty_a * (current_price_a - avg_entry_price_a)
            pnl_b = total_qty_b * (current_price_b - avg_entry_price_b)
            total_pnl = pnl_a + pnl_b
            pnl_percent = (total_pnl / total_notional) * 100 if total_notional > 0 else 0.0
            
            current_time = get_local_time().isoformat()
            
            # Return P&L data for marketstate only (never stored)
            return {
                'floating_pnl_usd': total_pnl,
                'floating_pnl_percent': pnl_percent,
                'current_price_a': current_price_a,
                'current_price_b': current_price_b,
                'last_calculated': current_time,
                'calculation_source': 'fresh_from_static_data',
                'storage_policy': 'never_stored_to_disk'
            }
            
        except Exception as e:
            # Return zero P&L on calculation errors
            return {
                'floating_pnl_usd': 0.0,
                'floating_pnl_percent': 0.0,
                'current_price_a': current_prices.get('primary', 0.0),
                'current_price_b': current_prices.get('reference', 0.0),
                'last_calculated': get_local_time().isoformat(),
                'calculation_source': 'error_fallback',
                'calculation_error': str(e),
                'storage_policy': 'never_stored_to_disk'
            }
    
    def can_add_position(
        self,
        aggregated_position: Dict[str, Any],
        max_positions: int
    ) -> Tuple[bool, str]:
        """
        Check if another position can be added.
        Enhanced to handle single-position mode properly.
        
        Args:
            aggregated_position: Current aggregated position
            max_positions: Maximum allowed component positions
            
        Returns:
            (can_add, reason)
        """
        # Check if this is single-position mode
        is_single_mode = aggregated_position.get('unified_metadata', {}).get('is_single_position_mode', False)
        
        if is_single_mode:
            return False, "Cannot add positions in single-position mode"
        
        current_count = aggregated_position.get('aggregated_metrics', {}).get('position_count', 0)
        
        if current_count >= max_positions:
            return False, f"Maximum positions reached ({current_count}/{max_positions})"
        
        return True, f"Can add position ({current_count}/{max_positions})"
    
    def _generate_aggregate_id(self, component_trades: List[Dict[str, Any]]) -> str:
        """
        Generate stable aggregate ID from first trade ID.
        Enhanced to always generate ID even for single trades.
        
        Args:
            component_trades: List of component trade dictionaries
            
        Returns:
            Aggregate ID string (e.g., "AGG-SIM-12345678" or "AGG-SINGLE-SIM-12345678")
        """
        if not component_trades:
            raise ValueError("Cannot generate aggregate ID from empty trade list")
        
        first_trade_id = component_trades[0].get('trade_id', 'UNKNOWN')
        
        # Generate appropriate prefix based on context
        if len(component_trades) == 1:
            return f"AGG-SINGLE-{first_trade_id}"
        else:
            return f"AGG-{first_trade_id}"
    
    def _get_direction(self, trade: Dict[str, Any]) -> str:
        """
        Extract direction from trade dict.
        
        Args:
            trade: Trade dictionary
            
        Returns:
            Direction string ('LONG' or 'SHORT')
        """
        direction = trade.get('direction', 'UNKNOWN')
        
        # Handle if it's already a string
        if isinstance(direction, str):
            # Clean up any enum class name prefixes
            return direction.replace('TradeDirection.', '')
        
        # Handle enum objects
        if hasattr(direction, 'value'):
            return direction.value
        
        # Fallback
        return str(direction)
    
    def convert_single_trade_to_aggregated(
        self,
        trade: Dict[str, Any],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Convenience method to convert a single trade to aggregated format.
        Useful for migration from old single-trade storage.
        STAGE 1 UPDATE: No P&L data included in result
        
        Args:
            trade: Single trade dictionary
            current_prices: Current market prices (for validation only)
            
        Returns:
            Aggregated position dictionary with position_count = 1 (no P&L data)
        """
        return self.aggregate_position([trade], current_prices, force_single=True)
