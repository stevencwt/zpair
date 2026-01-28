#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperliquid Utils Adapter - Reusable interface for Hyperliquid API operations
Extracted from trading bot for modular use across different trading strategies
Updated to support Market Barometer module interface
"""

import time
from typing import Dict, List, Optional,Any
from dataclasses import dataclass

# Hyperliquid SDK
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants


@dataclass
class PositionInfo:
    """Position information for a single asset"""
    coin: str
    size: float  # Position size (positive for long, negative for short)
    unrealized_pnl: float
    entry_price: float
    mark_price: float


@dataclass
class PairPnLInfo:
    """P&L information for a trading pair"""
    total_pnl: float
    asset_a_pnl: float
    asset_b_pnl: float


@dataclass
class PairQuantities:
    """Quantity information for a trading pair"""
    asset_a_qty: float
    asset_b_qty: float


class HyperliquidAdapter:
    """
    Reusable adapter for Hyperliquid API operations
    Provides clean interface for common trading operations
    Compatible with Market Barometer module interface
    """
    
    def __init__(self, private_key_hex: str, wallet_address: str, base_url: str = None):
        """
        Initialize Hyperliquid adapter
        
        Args:
            private_key_hex: Private key in hex format
            wallet_address: Wallet address
            base_url: API base URL (defaults to mainnet)
        """
        self.private_key_hex = private_key_hex
        self.wallet_address = wallet_address
        self.base_url = base_url or constants.MAINNET_API_URL
        
        # Initialize SDK components
        self.account = Account.from_key(private_key_hex)
        self.exchange = Exchange(self.account, base_url=self.base_url, account_address=wallet_address)
        self.info = Info(self.base_url)
        
        # Cache for reducing API calls
        self._price_cache = {}
        self._cache_timestamp = 0.0
        self._cache_duration = 5.0  # 5 seconds
        
    def get_all_mids(self, use_cache: bool = True) -> Dict[str, float]:
        """
        Get all mid prices from Hyperliquid
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping coin symbols to mid prices
        """
        current_time = time.time()
        
        # Use cache if enabled and valid
        if use_cache and self._price_cache and (current_time - self._cache_timestamp) < self._cache_duration:
            return self._price_cache
            
        try:
            mids = self.info.all_mids()
            if isinstance(mids, dict):
                price_dict = {k: float(v) for k, v in mids.items()}
                
                # Update cache
                self._price_cache = price_dict
                self._cache_timestamp = current_time
                
                return price_dict
            else:
                return {}
        except Exception as e:
            print(f"[hyperliquid] Error getting all_mids: {e}")
            return {}
    
    def get_price(self, coin: str, use_cache: bool = True) -> Optional[float]:
        """
        Get current price for a specific coin
        
        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            use_cache: Whether to use cached data if available
            
        Returns:
            Current price or None if not available
        """
        mids = self.get_all_mids(use_cache=use_cache)
        price = mids.get(coin.upper())
        return float(price) if price is not None else None
    
    # Market Barometer Interface Methods
    def last_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for Market Barometer compatibility
        
        Args:
            symbol: Coin symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Current price or None if not available
        """
        return self.get_price(symbol, use_cache=True)

    def get_valuation_price(self, side: str, book_data: Dict[str, Any], mode: str = 'MID') -> float:
        """
        Determines the correct valuation price based on position side and user config.
        
        Args:
            side: 'LONG' or 'SHORT'
            book_data: Standardized dict {'bid': float, 'ask': float, 'mid': float}
            mode: 'MID' (default) or 'BID_ASK' (conservative)
            
        Returns:
            float: The price to use for P&L calculation
        """
        try:
            # Normalize inputs
            side = side.upper()
            mode = mode.upper()
            
            # Default to Mid-Price
            price = float(book_data.get('mid', 0.0))
            
            if mode == 'BID_ASK':
                # Conservative Valuation Logic:
                # If we are LONG, we exit by SELLING -> We hit the BID
                if side == 'LONG':
                    price = float(book_data.get('bid', price))
                
                # If we are SHORT, we exit by BUYING -> We hit the ASK
                elif side == 'SHORT':
                    price = float(book_data.get('ask', price))
                    
            return price
            
        except (ValueError, TypeError, AttributeError) as e:
            # In case of malformed data, return 0.0 or a safe fallback
            print(f"[Valuation Error] Invalid inputs: {e}")
            return 0.0  
    
    def get_candles_snapshot(self, symbol: str, timeframe: str, count: int) -> Optional[List[dict]]:
        """
        Get historical candles for Market Barometer compatibility
        
        Args:
            symbol: Coin symbol
            timeframe: Time interval ('1m', '5m', '1h', '1d') 
            count: Number of candles to retrieve
            
        Returns:
            List of candle dictionaries or None if failed
        """
        return self.get_candles(symbol, timeframe, count)
    
    def get_candles(self, coin: str, interval: str, bars: int, end_time_ms: Optional[int] = None) -> Optional[List[dict]]:
        """
        Get historical candlestick data
        
        Args:
            coin: Coin symbol
            interval: Time interval ('1m', '5m', '1h', '1d')
            bars: Number of bars to retrieve
            end_time_ms: End time in milliseconds (defaults to current time)
            
        Returns:
            List of candle dictionaries or None if failed
        """
        try:
            if end_time_ms is None:
                end_time_ms = int(time.time() * 1000)
                
            # Calculate start time based on interval and bars
            interval_ms = {
                "1m": 60 * 1000,
                "5m": 5 * 60 * 1000, 
                "1h": 60 * 60 * 1000,
                "1d": 24 * 60 * 60 * 1000
            }.get(interval, 60 * 1000)
            
            start_time_ms = end_time_ms - (bars * interval_ms)
            
            candles = self.info.candles_snapshot(coin.upper(), interval, start_time_ms, end_time_ms)
            
            if candles and isinstance(candles, list):
                return candles
            return None
            
        except Exception as e:
            print(f"[hyperliquid] Error getting candles for {coin}: {e}")
            return None
    
    @staticmethod
    def extract_closes(candles: List[dict]) -> List[float]:
        """
        Extract closing prices from candle data
        
        Args:
            candles: List of candle dictionaries
            
        Returns:
            List of closing prices
        """
        closes = []
        for candle in candles or []:
            try:
                close_price = float(candle.get("c", 0.0))
                closes.append(close_price)
            except (ValueError, TypeError):
                closes.append(0.0)
        return closes
    
    def get_user_state(self, wallet_address: Optional[str] = None) -> Optional[dict]:
        """
        Get user state including positions and balances
        
        Args:
            wallet_address: Wallet to query (defaults to configured wallet)
            
        Returns:
            User state dictionary or None if failed
        """
        try:
            address = wallet_address or self.wallet_address
            user_state = self.info.user_state(address)
            return user_state
        except Exception as e:
            print(f"[hyperliquid] Error getting user state: {e}")
            return None
    
    def get_positions(self, wallet_address: Optional[str] = None) -> List[PositionInfo]:
        """
        Get all active positions
        
        Args:
            wallet_address: Wallet to query (defaults to configured wallet)
            
        Returns:
            List of PositionInfo objects for active positions
        """
        try:
            user_state = self.get_user_state(wallet_address)
            if not user_state:
                return []
                
            positions = []
            asset_positions = user_state.get('assetPositions', [])
            
            for pos_data in asset_positions:
                position = pos_data.get('position', {})
                size = float(position.get('szi', 0))
                
                # Only include active positions
                if size != 0:
                    pos_info = PositionInfo(
                        coin=position.get('coin', '').upper(),
                        size=size,
                        unrealized_pnl=float(position.get('unrealizedPnl', 0)),
                        entry_price=float(position.get('entryPx', 0)),
                        mark_price=float(position.get('positionValue', 0)) / abs(size) if size != 0 else 0
                    )
                    positions.append(pos_info)
                    
            return positions
            
        except Exception as e:
            print(f"[hyperliquid] Error getting positions: {e}")
            return []
    
    def has_positions(self, asset_a: Optional[str] = None, asset_b: Optional[str] = None, wallet_address: Optional[str] = None) -> bool:
        """
        Check if positions exist, optionally for specific assets
        
        Args:
            asset_a: First asset to check (optional)
            asset_b: Second asset to check (optional) 
            wallet_address: Wallet to query (defaults to configured wallet)
            
        Returns:
            True if positions exist (matching criteria if assets specified)
        """
        try:
            positions = self.get_positions(wallet_address)
            
            if not positions:
                return False
                
            # If no specific assets specified, return True if any positions exist
            if asset_a is None and asset_b is None:
                return True
                
            # Check for specific assets
            if asset_a and asset_b:
                target_assets = {asset_a.upper(), asset_b.upper()}
                position_coins = {pos.coin for pos in positions}
                return target_assets.issubset(position_coins)
            elif asset_a:
                return any(pos.coin == asset_a.upper() for pos in positions)
            elif asset_b:
                return any(pos.coin == asset_b.upper() for pos in positions)
                
            return False
            
        except Exception as e:
            print(f"[hyperliquid] Error checking positions: {e}")
            return False
    
    def get_pair_pnl(self, asset_a: str, asset_b: str, wallet_address: Optional[str] = None) -> PairPnLInfo:
        """
        Get P&L information for a trading pair
        
        Args:
            asset_a: First asset symbol
            asset_b: Second asset symbol
            wallet_address: Wallet to query (defaults to configured wallet)
            
        Returns:
            PairPnLInfo object with P&L data
        """
        try:
            positions = self.get_positions(wallet_address)
            
            asset_a_pnl = 0.0
            asset_b_pnl = 0.0
            
            for pos in positions:
                if pos.coin == asset_a.upper():
                    asset_a_pnl = pos.unrealized_pnl
                elif pos.coin == asset_b.upper():
                    asset_b_pnl = pos.unrealized_pnl
                    
            total_pnl = asset_a_pnl + asset_b_pnl
            
            return PairPnLInfo(
                total_pnl=total_pnl,
                asset_a_pnl=asset_a_pnl,
                asset_b_pnl=asset_b_pnl
            )
            
        except Exception as e:
            print(f"[hyperliquid] Error getting pair P&L: {e}")
            return PairPnLInfo(total_pnl=0.0, asset_a_pnl=0.0, asset_b_pnl=0.0)
    
    def get_pair_quantities(self, asset_a: str, asset_b: str, wallet_address: Optional[str] = None) -> PairQuantities:
        """
        Get position quantities for a trading pair
        
        Args:
            asset_a: First asset symbol
            asset_b: Second asset symbol  
            wallet_address: Wallet to query (defaults to configured wallet)
            
        Returns:
            PairQuantities object with quantity data
        """
        try:
            positions = self.get_positions(wallet_address)
            
            asset_a_qty = 0.0
            asset_b_qty = 0.0
            
            for pos in positions:
                if pos.coin == asset_a.upper():
                    asset_a_qty = abs(pos.size)  # Use absolute value for quantity
                elif pos.coin == asset_b.upper():
                    asset_b_qty = abs(pos.size)
                    
            return PairQuantities(
                asset_a_qty=asset_a_qty,
                asset_b_qty=asset_b_qty
            )
            
        except Exception as e:
            print(f"[hyperliquid] Error getting pair quantities: {e}")
            return PairQuantities(asset_a_qty=0.0, asset_b_qty=0.0)
    
    def check_position_directions(self, asset_a: str, asset_b: str, wallet_address: Optional[str] = None) -> Dict:
        """
        Check position directions for a trading pair
        
        Args:
            asset_a: First asset symbol
            asset_b: Second asset symbol
            wallet_address: Wallet to query (defaults to configured wallet)
            
        Returns:
            Dictionary with position direction information
        """
        try:
            positions = self.get_positions(wallet_address)
            
            asset_a_pos = None
            asset_b_pos = None
            
            for pos in positions:
                if pos.coin == asset_a.upper():
                    asset_a_pos = pos.size
                elif pos.coin == asset_b.upper():
                    asset_b_pos = pos.size
                    
            if asset_a_pos is None or asset_b_pos is None:
                return {
                    'has_positions': False,
                    'asset_a_direction': None,
                    'asset_b_direction': None,
                    'description': 'Incomplete pair positions'
                }
                
            asset_a_direction = 'LONG' if asset_a_pos > 0 else 'SHORT'
            asset_b_direction = 'LONG' if asset_b_pos > 0 else 'SHORT'
            
            # Determine overall direction pattern
            if asset_a_pos > 0 and asset_b_pos < 0:
                pattern = f"LONG {asset_a} / SHORT {asset_b}"
            elif asset_a_pos < 0 and asset_b_pos > 0:
                pattern = f"SHORT {asset_a} / LONG {asset_b}"
            else:
                pattern = f"UNUSUAL: {asset_a}={asset_a_direction}, {asset_b}={asset_b_direction}"
                
            return {
                'has_positions': True,
                'asset_a_direction': asset_a_direction,
                'asset_b_direction': asset_b_direction,
                'asset_a_size': asset_a_pos,
                'asset_b_size': asset_b_pos,
                'pattern': pattern,
                'description': f"Active pair positions: {pattern}"
            }
            
        except Exception as e:
            print(f"[hyperliquid] Error checking position directions: {e}")
            return {
                'has_positions': False,
                'error': str(e),
                'description': f'Error checking positions: {e}'
            }
    
    def get_account_summary(self, wallet_address: Optional[str] = None) -> Dict:
        """
        Get account summary with key metrics
        
        Args:
            wallet_address: Wallet to query (defaults to configured wallet)
            
        Returns:
            Dictionary with account summary
        """
        try:
            user_state = self.get_user_state(wallet_address)
            positions = self.get_positions(wallet_address)
            
            if not user_state:
                return {'error': 'Unable to fetch user state'}
                
            # Calculate totals
            total_pnl = sum(pos.unrealized_pnl for pos in positions)
            total_positions = len(positions)
            
            # Get margin summary if available
            margin_summary = user_state.get('marginSummary', {})
            account_value = float(margin_summary.get('accountValue', 0))
            total_margin_used = float(margin_summary.get('totalMarginUsed', 0))
            
            return {
                'account_value': account_value,
                'total_margin_used': total_margin_used,
                'total_unrealized_pnl': total_pnl,
                'total_positions': total_positions,
                'positions': [
                    {
                        'coin': pos.coin,
                        'size': pos.size,
                        'pnl': pos.unrealized_pnl,
                        'direction': 'LONG' if pos.size > 0 else 'SHORT'
                    } for pos in positions
                ],
                'wallet_address': wallet_address or self.wallet_address
            }
            
        except Exception as e:
            print(f"[hyperliquid] Error getting account summary: {e}")
            return {'error': str(e)}

    def get_bbo(self, coin: str) -> Dict[str, float]:
        """
        Get Best Bid and Offer (BBO) for a specific coin.
        Returns dict with 'bid', 'ask', 'mid'.
        """
        try:
            # Fetch L2 Order Book Snapshot
            snapshot = self.info.l2_snapshot(coin.upper())
            
            # Hyperliquid structure: {'levels': [ [bids...], [asks...] ]}
            # Each level is [px, sz, n]
            levels = snapshot.get('levels', [])
            
            if len(levels) >= 2:
                bids = levels[0]
                asks = levels[1]
                
                best_bid = float(bids[0]['px']) if bids else 0.0
                best_ask = float(asks[0]['px']) if asks else 0.0
                
                # Fallback to mid if book is empty (unlikely)
                if best_bid == 0 or best_ask == 0:
                    mid = self.get_price(coin)
                    return {'bid': mid, 'ask': mid, 'mid': mid}
                    
                mid = (best_bid + best_ask) / 2.0
                
                return {
                    'bid': best_bid,
                    'ask': best_ask,
                    'mid': mid
                }
            
            # Fallback if structure is unexpected
            mid = self.get_price(coin)
            return {'bid': mid, 'ask': mid, 'mid': mid}
            
        except Exception as e:
            # print(f"[hyperliquid] Error getting BBO for {coin}: {e}")
            mid = self.get_price(coin)
            return {'bid': mid, 'ask': mid, 'mid': mid}
    
    def clear_cache(self):
        """Clear the price cache"""
        self._price_cache = {}
        self._cache_timestamp = 0.0
    
    def set_cache_duration(self, seconds: float):
        """Set cache duration in seconds"""
        self._cache_duration = max(0.0, seconds)