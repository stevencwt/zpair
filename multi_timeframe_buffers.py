#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Timeframe Buffer System - SOURCE-LEVEL DUPLICATE PREVENTION
-----------------------------------------------------------------------
Production Version - Cleaned Logs
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import threading
import copy
import math

# Import regression core for analysis
try:
    from regression_core import RegressionCore
    REGRESSION_AVAILABLE = True
except ImportError:
    REGRESSION_AVAILABLE = False
    # Quietly handle missing dependency in production

def debug_print(msg):
    """Debug print - DISABLED for production"""
    pass

class TimeSeriesBuffer:
    """
    Efficient time series buffer with source-level duplicate prevention
    """
    
    def __init__(self, symbol: str, timeframe: str, window_size: int, realtime_only: bool = False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.window_size = window_size
        self.realtime_only = realtime_only

        # Live prices cache
        #self.current_prices = {
        #    self.base_symbol: 0.0,
        #
        #    self.quote_symbol: 0.0
        #}
        # [NEW] Add BBO Storage
        #self.current_bbos = {}
        
        # Core OHLCV data storage (FIFO deques) - for CLOSED candles only
        self.timestamps = deque(maxlen=int(window_size))
        self.opens = deque(maxlen=window_size)
        self.highs = deque(maxlen=window_size)
        self.lows = deque(maxlen=window_size)
        self.closes = deque(maxlen=window_size)
        self.volumes = deque(maxlen=window_size)
        
        # Developing candle state
        self.developing_candle = None
        self.developing_candle_start_time = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Candle management state
        self.last_finalized_time = None
        self.last_candle_close = None
        
        # Timeframe parsing
        self.tf_seconds = self._parse_timeframe(timeframe)

    def _parse_timeframe(self, tf: str) -> int:
        """Parse timeframe string to seconds"""
        if tf.endswith('s'):
            return int(tf[:-1])
        elif tf.endswith('m'):
            return int(tf[:-1]) * 60
        elif tf.endswith('h'):
            return int(tf[:-1]) * 3600
        elif tf.endswith('d'):
            return int(tf[:-1]) * 86400
        return 60  # Default to 1m
    
    def _get_candle_start_time(self, timestamp: float) -> float:
        """Calculate precise candle start time based on timeframe epochs"""
        return (int(timestamp) // self.tf_seconds) * self.tf_seconds
        
    def add_tick(self, price: float, volume: float = 0.0, timestamp: float = None) -> bool:
        """
        Process a new tick and manage candle formation
        Returns True if a new candle was finalized
        """
        if timestamp is None:
            timestamp = time.time()
            
        with self.lock:
            candle_finalized = False
            
            # Calculate which candle this tick belongs to
            tick_candle_start = self._get_candle_start_time(timestamp)
            
            # Initialize developing candle if needed
            if self.developing_candle is None:
                self.developing_candle = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'start_time': tick_candle_start
                }
                self.developing_candle_start_time = tick_candle_start
                
            # Check if we've moved to a new candle
            elif tick_candle_start > self.developing_candle_start_time:
                # Finalize the previous candle
                self._finalize_candle(self.developing_candle, self.developing_candle_start_time)
                candle_finalized = True
                
                # Start new candle
                self.developing_candle = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'start_time': tick_candle_start
                }
                self.developing_candle_start_time = tick_candle_start
                
            else:
                # Update current developing candle
                self.developing_candle['high'] = max(self.developing_candle['high'], price)
                self.developing_candle['low'] = min(self.developing_candle['low'], price)
                self.developing_candle['close'] = price
                self.developing_candle['volume'] += volume
            
            return candle_finalized
            
    def _finalize_candle(self, candle_data: Dict[str, Any], start_time: float) -> None:
        """Add a finalized candle to storage"""
        # Convert timestamp to datetime for storage
        dt_timestamp = datetime.fromtimestamp(start_time, timezone.utc)
        
        # Duplicate prevention check
        if self.timestamps and dt_timestamp <= self.timestamps[-1]:
            return
            
        self.timestamps.append(dt_timestamp)
        self.opens.append(candle_data['open'])
        self.highs.append(candle_data['high'])
        self.lows.append(candle_data['low'])
        self.closes.append(candle_data['close'])
        self.volumes.append(candle_data['volume'])
        
        self.last_finalized_time = dt_timestamp
        self.last_candle_close = candle_data['close']

    def add_historical_candles(self, candles: List[Dict[str, Any]]) -> int:
        """
        Add multiple historical candles efficiently
        Expects dicts with keys: t (ms timestamp), o, h, l, c, v
        """
        added_count = 0
        
        with self.lock:
            # Sort by time just in case
            sorted_candles = sorted(candles, key=lambda x: x.get('t', 0))
            
            for c in sorted_candles:
                # Convert ms timestamp to datetime
                ts_ms = c.get('t', 0)
                dt = datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc)
                
                # Skip if older than what we have (unless empty)
                if self.timestamps and dt <= self.timestamps[-1]:
                    continue
                
                self.timestamps.append(dt)
                self.opens.append(float(c.get('o', 0)))
                self.highs.append(float(c.get('h', 0)))
                self.lows.append(float(c.get('l', 0)))
                self.closes.append(float(c.get('c', 0)))
                self.volumes.append(float(c.get('v', 0)))
                
                added_count += 1
            
            if self.timestamps:
                self.last_finalized_time = self.timestamps[-1]
                self.last_candle_close = self.closes[-1]
                
        return added_count
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get current buffer as DataFrame"""
        with self.lock:
            if not self.timestamps:
                return pd.DataFrame()
                
            data = {
                'timestamp': list(self.timestamps),
                'open': list(self.opens),
                'high': list(self.highs),
                'low': list(self.lows),
                'close': list(self.closes),
                'volume': list(self.volumes)
            }
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
    def get_status(self) -> Dict[str, Any]:
        """Get buffer status"""
        with self.lock:
            return {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'count': len(self.timestamps),
                'last_update': self.last_finalized_time.isoformat() if self.last_finalized_time else None,
                'current_price': self.developing_candle['close'] if self.developing_candle else (self.closes[-1] if self.closes else 0.0)
            }


class MultiTimeframeBuffers:
    """
    Manager for multiple timeframes and assets
    """
    
    def __init__(self, base_symbol: str, quote_symbol: str, config: Dict[str, Any], provider=None):
        self.base_symbol = base_symbol
        self.quote_symbol = quote_symbol
        self.config = config
        self.provider = provider
        
        self.buffers: Dict[str, TimeSeriesBuffer] = {}
        self.is_initialized = False
        self.initialization_time = None
        self.last_price_fetch = None
        self.last_historical_refresh = None
        

        # FIX: Initialize the BBO cache here where it belongs
        self.current_bbos = {}

        # Live prices cache
        self.current_prices = {
            self.base_symbol: 0.0,
            self.quote_symbol: 0.0
        }
        
        # Lock for management operations
        self.lock = threading.RLock()
        
        # Initialize buffers based on config
        self._setup_buffers()
        
        # Initialize regression core
        if REGRESSION_AVAILABLE:
            self.regression_core = RegressionCore()
        else:
            self.regression_core = None
    
    def _setup_buffers(self):
        """Create buffer instances for configured timeframes"""
        timeframes = self.config.get('timeframes', {'1m': 300})
        
        for tf, config_or_size in timeframes.items():
            # FIX: Handle both dictionary config (new) and integer (legacy)
            if isinstance(config_or_size, dict):
                # Extract 'buffer_size' from the config dictionary
                window_size = int(config_or_size.get('buffer_size', 300))
            else:
                # It's already a number (legacy format)
                window_size = int(config_or_size)
            
            # Determine if this timeframe is realtime-only (like 5s)
            is_realtime = False
            if tf in ['1s', '5s']:
                is_realtime = True
            
            # Create buffers for both assets
            base_key = f"{self.base_symbol}_{tf}"
            quote_key = f"{self.quote_symbol}_{tf}"
            
            self.buffers[base_key] = TimeSeriesBuffer(self.base_symbol, tf, window_size, is_realtime)
            self.buffers[quote_key] = TimeSeriesBuffer(self.quote_symbol, tf, window_size, is_realtime)
            
    def initialize_historical_data(self) -> bool:
        """Fetch and populate historical data for all buffers"""
        if not self.provider:
            return False
            
        success_count = 0
        
        for key, buffer in self.buffers.items():
            if buffer.realtime_only:
                # Skip historical fetch for realtime-only buffers
                success_count += 1
                continue
                
            try:
                # Calculate needed candles
                limit = buffer.window_size + 50 # Add buffer
                
                # Fetch candles
                candles = self.provider.get_candles(buffer.symbol, buffer.timeframe, limit)
                
                if candles:
                    added = buffer.add_historical_candles(candles)
                    success_count += 1
                    
            except Exception as e:
                pass
        
        self.is_initialized = (success_count > 0)
        if self.is_initialized:
            self.initialization_time = datetime.now(timezone.utc)
            self.last_historical_refresh = datetime.now(timezone.utc)
       

        return self.is_initialized
        
    def update_current_prices(self) -> bool:
        """
        RESOLVED FIX: Matches test_bbo.py logic.
        Provides definitive evidence of retrieval via ✅ [BBO PROOF SUCCESS].
        """
        if not self.provider: 
            return False
            
        try:
            # 1. Fetch RAW BBO objects (The test_bbo method)
            base_bbo = self.provider.get_bbo(self.base_symbol)
            quote_bbo = self.provider.get_bbo(self.quote_symbol)
            
            # 2. THE HIGHEST PRIORITY EVIDENCE: Match test_bbo extraction
            # Using 'or 0' handles potential None values from the dictionary
            b_bid = float(base_bbo.get('bid') or 0.0)
            b_ask = float(base_bbo.get('ask') or 0.0)
            q_bid = float(quote_bbo.get('bid') or 0.0)
            q_ask = float(quote_bbo.get('ask') or 0.0)

            if b_bid > 0 and b_ask > 0:
                # THIS IS THE PROOF: It will appear in your logs if successful
                print(f"✅ [BBO PROOF SUCCESS] {self.base_symbol}: Bid={b_bid}, Ask={b_ask}")
                with self.lock:
                    # Store as dicts for simulator fallback logic
                    self.current_bbos[self.base_symbol] = base_bbo
                    self.current_bbos[self.quote_symbol] = quote_bbo
            else:
                # PROOF OF API FAILURE: Tells us if Hyperliquid is returning 0s
                print(f"❌ [BBO PROOF FAILURE] API returned zeros for {self.base_symbol}")

            # 3. Extract prices for candles (FIXED: quote uses quote_bbo)
            base_price = float(base_bbo.get('mid') or b_bid or 0.0)
            quote_price = float(quote_bbo.get('mid') or q_bid or 0.0)
            
            if base_price > 0 and quote_price > 0:
                with self.lock:
                    self.current_prices[self.base_symbol] = base_price
                    self.current_prices[self.quote_symbol] = quote_price
                    self.last_price_fetch = datetime.now(timezone.utc)
                self._distribute_ticks(self.base_symbol, base_price)
                self._distribute_ticks(self.quote_symbol, quote_price)
                return True
            return False
        except Exception as e:
            print(f"🛑 [CRITICAL EVIDENCE ERROR] Method failed: {e}")
            return False

    def get_current_live_bbos(self):
        """Get currently cached BBOs"""
        with self.lock:
            b_bbo = self.current_bbos.get(self.base_symbol, {'bid':0,'ask':0,'mid':0})
            q_bbo = self.current_bbos.get(self.quote_symbol, {'bid':0,'ask':0,'mid':0})
            return b_bbo, q_bbo

    def get_current_live_prices(self) -> Tuple[float, float]:
        """
        Get currently cached live prices for base and quote assets.
        Required by telemetry_provider.py for real-time analysis.
        """
        with self.lock:
            base_price = self.current_prices.get(self.base_symbol, 0.0)
            quote_price = self.current_prices.get(self.quote_symbol, 0.0)
            return base_price, quote_price

    def _distribute_ticks(self, symbol: str, price: float):
        """Push new price to all buffers matching the symbol"""
        timestamp = time.time()
        
        for key, buffer in self.buffers.items():
            if buffer.symbol == symbol:
                buffer.add_tick(price, 0.0, timestamp)

    def get_analysis_data(self, timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get aligned DataFrames for analysis"""
        base_key = f"{self.base_symbol}_{timeframe}"
        quote_key = f"{self.quote_symbol}_{timeframe}"
        
        if base_key not in self.buffers or quote_key not in self.buffers:
            return None, None
            
        df1 = self.buffers[base_key].get_dataframe()
        df2 = self.buffers[quote_key].get_dataframe()
        
        return df1, df2

    def get_aligned_analysis_data(self, timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get ALIGNED DataFrames for analysis (intersection of timestamps).
        Required by telemetry_provider.py to ensure regression accuracy.
        """
        # Get raw data using existing method
        df1, df2 = self.get_analysis_data(timeframe)
        
        if df1 is None or df2 is None or df1.empty or df2.empty:
            return None, None
            
        # Align by index (timestamp intersection)
        common_index = df1.index.intersection(df2.index)
        
        if len(common_index) < 10:  # Minimum required points
            return None, None
            
        # Return only the matching timestamps
        return df1.loc[common_index], df2.loc[common_index]

    def analyze_timeframe(self, timeframe: str) -> Dict[str, Any]:
        """
        Perform analysis on a specific timeframe
        Returns formatted regression metrics
        """
        try:
            # Get data
            df1, df2 = self.get_analysis_data(timeframe)
            
            if df1 is None or df2 is None or df1.empty or df2.empty:
                return self._get_fallback_metrics(timeframe)
                
            # Align data by index (intersection)
            common_index = df1.index.intersection(df2.index)
            
            if len(common_index) < 20:
                return self._get_fallback_metrics(timeframe)
                
            s1 = df1.loc[common_index]['close']
            s2 = df2.loc[common_index]['close']
            
            # Get live prices
            p1 = self.current_prices.get(self.base_symbol, s1.iloc[-1])
            p2 = self.current_prices.get(self.quote_symbol, s2.iloc[-1])
            
            window = min(len(s1), self.config.get('regression_window', 200))
            if window < 10:
                return self._get_fallback_metrics(timeframe)
                
            # Perform calculation using the latest 'window' points
            y = s1.iloc[-window:]
            x = s2.iloc[-window:]
            
            try:
                # Calculate beta (slope)
                covariance = np.cov(x, y)[0, 1]
                variance = np.var(x)
                beta = covariance / variance if variance != 0 else 1.0
                alpha = np.mean(y) - beta * np.mean(x)
                
                # Calculate spread
                spread = y - (beta * x + alpha)
                mean_spread = np.mean(spread)
                std_spread = np.std(spread)
                
                # Calculate current live Z-Score
                current_spread = p1 - (beta * p2 + alpha)
                zscore = (current_spread - mean_spread) / std_spread if std_spread != 0 else 0.0
                
                r_squared = (covariance**2) / (np.var(x) * np.var(y)) if np.var(x)*np.var(y) != 0 else 0.0
                
                return {
                    'zscore': float(zscore),
                    'beta': float(beta),
                    'alpha': float(alpha),
                    'spread': float(current_spread),
                    'r_squared': float(r_squared),
                    'correlation': float(np.corrcoef(x, y)[0, 1]),
                    'window_size': int(window),
                    'timeframe': timeframe,
                    'health_status': 'ok'
                }
                
            except Exception as e:
                return self._get_fallback_metrics(timeframe)
            
        except Exception as e:
            return self._get_fallback_metrics(timeframe)
            
    def _get_fallback_metrics(self, timeframe: str) -> Dict[str, Any]:
        """Return safe default metrics"""
        return {
            'zscore': 0.0,
            'beta': 1.0,
            'alpha': 0.0,
            'spread': 0.0,
            'r_squared': 0.0,
            'correlation': 0.0,
            'window_size': 0,
            'timeframe': timeframe,
            'health_status': 'unknown'
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all buffers"""
        buffer_statuses = {}
        
        for buffer_key, buffer in self.buffers.items():
            buffer_statuses[buffer_key] = buffer.get_status()
        
        return {
            'is_initialized': self.is_initialized,
            'initialization_time': self.initialization_time.isoformat() if self.initialization_time else None,
            'last_price_fetch': self.last_price_fetch.isoformat() if self.last_price_fetch else None,
            'last_historical_refresh': self.last_historical_refresh.isoformat() if self.last_historical_refresh else None,
            'base_symbol': self.base_symbol,
            'quote_symbol': self.quote_symbol,
            'timeframes': list(self.config.get('timeframes', {}).keys()),
            'buffers': buffer_statuses
        }
    
    def should_refresh_historical_data(self) -> bool:
        """Check if historical data should be refreshed"""
        if not self.last_historical_refresh:
            return True
        
        refresh_interval = self.config.get('api_optimization', {}).get('historical_refresh_interval_hours', 6)
        hours_since_refresh = (datetime.now(timezone.utc) - self.last_historical_refresh).total_seconds() / 3600
        
        return hours_since_refresh >= refresh_interval