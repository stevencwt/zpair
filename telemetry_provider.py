#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Telemetry Provider Module - Price Consistency & Streaming
------------------------------------------------------------
UPDATED: Fixed SyntaxError and macOS SSL Certificate Verify Failed error.
Implemented "Fastmain" pattern with SSL bypass for robust streaming.
"""

import time
import math
import json
import threading
import ssl  # Required for SSL context manipulation
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from collections import deque

# --- IMPORT WEBSOCKET FOR STREAMING ---
WEBSOCKET_AVAILABLE = False
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    print("[WARN] 'websocket-client' library not found. Streaming disabled. Install via: pip install websocket-client")

# Import existing modules with debug logging
HYPERLIQUID_AVAILABLE = False
REGRESSION_AVAILABLE = False

try:
    from hyperliquid_utils_adapter import HyperliquidAdapter
    HYPERLIQUID_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG] HyperliquidAdapter import failed: {e}")

# Import regression modules
try:
    from regression_core import RegressionCore
    from zscore_engine import ZScoreEngine
    from fairvalue_builder import FairValueBuilder
    REGRESSION_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG] Regression modules import failed: {e}")

def calculate_hurst(ts, max_lag: int = 20) -> float:
    """
    Hurst Exponent (H) via RMS of Lagged Differences.
    Input: Log-Prices (Cumulative). 
    Output: 0.5 = Random, >0.5 = Trend, <0.5 = Mean Reverting
    """
    try:
        # [PHASE 5 FIX] Ensure input is Numpy Array
        if not isinstance(ts, np.ndarray):
            ts = np.array(ts)

        # [CLEANUP] Remove NaNs or Infs that crash the math
        ts = ts[np.isfinite(ts)]
        
        # [DEBUG DIAGNOSTIC] Check Length
        if len(ts) < max_lag * 2:
            # [FIX] Startup Mode: Dynamically reduce lag to fit available data
            # This ensures we get a 'good enough' reading immediately instead of flatlining at 0.5
            adjusted_lag = (len(ts) // 2) - 1
            
            if adjusted_lag < 4: # Absolute minimum needed for math
                print(f"[DEBUG HURST] ⚠️ Critical Insufficient Data: {len(ts)} points. Returning 0.5.")
                return 0.5
                
            # Use the safe, smaller lag temporarily
            max_lag = adjusted_lag
            # Optional: print(f"[DEBUG HURST] Startup: Adjusted lag to {max_lag} for {len(ts)} points")

        lags = range(2, max_lag)
        
        # Calculate RMS of differences (Standard Hurst Logic)
        tau = [np.sqrt(np.mean(np.square(np.subtract(ts[lag:], ts[:-lag])))) for lag in lags]
        
        # Handle case where tau has zeros (prevents log(0) error)
        if any(t <= 0 for t in tau):
             print(f"[DEBUG HURST] ⚠️ Math Instability (Tau=0). Returning 0.5.")
             return 0.5

        # Log-log regression
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # Return H = Slope
        return reg[0] 

    except Exception as e:
        print(f"[DEBUG HURST] ❌ Calculation Error: {e} -> Returning 0.5")
        return 0.5


def calculate_individual_trend_beta(prices, window=20):
    """Calculate individual asset trend beta (slope against time)"""
    try:
        if len(prices) < window:
            return 0.0
            
        recent_prices = prices[-window:]
        time_index = np.arange(len(recent_prices))
        
        # Linear regression: price = alpha + beta * time
        beta, alpha = np.polyfit(time_index, recent_prices, 1)
        
        return float(beta)  # Directional trend slope
    except Exception as e:
        print(f"[DEBUG TREND_BETA] Calculation error: {e}")
        return 0.0

#def calculate_pair_hurst(asset_a_prices, asset_b_prices, beta, window=100):
#    """Calculate Hurst exponent of the pair spread"""
#    try:
#        if len(asset_a_prices) < window or len(asset_b_prices) < window:
#            return 0.5
#        
#        # Calculate the spread (residuals from regression)
#        recent_a = asset_a_prices[-window:]
#        recent_b = asset_b_prices[-window:]
#       
#        # Create the spread: SOL - beta * MELANIA  
#        spread = recent_a - (beta * recent_b)
#        
#        # Calculate Hurst of the spread
#        spread_log = np.log(np.abs(spread) + 1e-8)  # Avoid log(0)
#        
#        return calculate_hurst(spread_log)
#    except Exception as e:
#        print(f"[DEBUG PAIR_HURST] Calculation error: {e}")
#        return 0.5

def calculate_pair_hurst(asset_a_prices, asset_b_prices, beta, window=100):
    """Calculate Hurst exponent of the pair spread"""
    try:
        # [FIX] Ensure valid data length
        if len(asset_a_prices) < 20 or len(asset_b_prices) < 20:
            return 0.5
        
        # Slice to window size
        recent_a = asset_a_prices[-window:]
        recent_b = asset_b_prices[-window:]
        
        # 1. Calculate the raw spread (Residuals)
        # This acts as the "Price" of our synthetic asset
        spread = recent_a - (beta * recent_b)
        
        # 2. [CORRECTION] Use Raw Spread
        # Do NOT use cumsum (integrates bias)
        # Do NOT use log (destroys negative values)
        # Passing the spread directly allows the estimator to detect
        # if the series is bounded (H < 0.5) or trending (H > 0.5).
        return calculate_hurst(spread)
        
    except Exception as e:
        print(f"[DEBUG PAIR_HURST] Calculation error: {e}")
        return 0.5


def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    """
    Calculate RSI using Wilder's Smoothing Method.
    """
    try:
        # [DEBUG TRACER] Check Input Data Size
        print(f"[DEBUG RSI] Calculating on {len(series)} points (Period: {period})")

        if len(series) < period + 1:
            print(f"[DEBUG RSI] Not enough data: {len(series)} < {period + 1} -> Returning 50.0")
            return 50.0
            
        delta = series.diff()
        
        # Wilder's Smoothing (alpha = 1/period)
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        val = float(rsi.iloc[-1])
        
        # [DEBUG TRACER] Check Calculated Result
        print(f"[DEBUG RSI] Calculated Value: {val}")
        return val
        
    except Exception as e:
        print(f"[DEBUG RSI] Calculation Error: {e}")
        return 50.0



# === INSERT NEW CLASS ===
class OrderbookStatsTracker:
    """
    Tracks statistics for BOTH Imbalance (Liquidity Ratio) and Spread (Bid-Ask Cost).
    Uses rolling windows to calculate Z-Scores for dynamic anomaly detection.
    """
    def __init__(self, window_size=60, min_history=10):
        self.window_size = window_size
        self.min_history = min_history
        
        # History Deques
        self.imbalance_history = deque(maxlen=window_size)
        self.spread_history = deque(maxlen=window_size)
        
        # Real-time metrics
        self.metrics = {
            "imbalance_ratio": 0.0,
            "imbalance_zscore": 0.0,
            "imbalance_mean": 0.0,
            "spread_bps": 0.0,
            "spread_zscore": 0.0,
            "spread_mean": 0.0,
            "spread_std": 0.0
        }

    def update(self, bid_px: float, ask_px: float, bid_sz: float, ask_sz: float) -> None:
        if bid_px <= 0 or ask_px <= 0: return

        # 1. Calculate Raw Metrics
        # Imbalance (Ratio)
        ratio = (bid_px * bid_sz) / (ask_px * ask_sz) if (ask_px * ask_sz) > 0 else 0.0
        
        # Spread (Basis Points)
        mid = (bid_px + ask_px) / 2.0
        spread_bps = ((ask_px - bid_px) / mid) * 10000.0

        # 2. Update Histories
        self.imbalance_history.append(ratio)
        self.spread_history.append(spread_bps)
        
        # 3. Calculate Statistics (Baseline)
        self.metrics["imbalance_ratio"] = ratio
        self.metrics["spread_bps"] = spread_bps
        
        if len(self.spread_history) >= self.min_history:
            # Spread Stats
            s_mean = np.mean(self.spread_history)
            s_std = np.std(self.spread_history)
            self.metrics["spread_mean"] = s_mean
            self.metrics["spread_std"] = s_std
            
            if s_std > 0:
                self.metrics["spread_zscore"] = (spread_bps - s_mean) / s_std
            else:
                self.metrics["spread_zscore"] = 0.0
                
            # Imbalance Stats (Simplified)
            i_mean = np.mean(self.imbalance_history)
            i_std = np.std(self.imbalance_history)
            self.metrics["imbalance_mean"] = i_mean
            if i_std > 0:
                self.metrics["imbalance_zscore"] = (ratio - i_mean) / i_std
            else:
                self.metrics["imbalance_zscore"] = 0.0
        else:
            self.metrics["spread_zscore"] = 0.0
            self.metrics["imbalance_zscore"] = 0.0

    def get_metrics(self) -> Dict[str, float]:
        return self.metrics.copy()

    def get_history_snapshot(self) -> Dict[str, List[float]]:
        return {
            "imbalance": list(self.imbalance_history),
            "spread": list(self.spread_history)
        }

    def restore_history(self, snapshot: Dict[str, List[float]]) -> None:
        if not snapshot: return
        if "imbalance" in snapshot:
            self.imbalance_history = deque(snapshot["imbalance"], maxlen=self.window_size)
        if "spread" in snapshot:
            self.spread_history = deque(snapshot["spread"], maxlen=self.window_size)
        print(f"[OB TRACKER] Restored history (Spread: {len(self.spread_history)}, Imb: {len(self.imbalance_history)})")
# -------------------- Streaming Client (Fast Brain) --------------------

class HyperliquidStreamingClient:
    """
    Background WebSocket client that maintains a real-time local cache of prices.
    Implements the 'Fastmain' pattern: Push updates to memory -> Poll memory instantly.
    """
    def __init__(self):
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self.ws = None
        self.wst = None
        self.is_running = False
        
        # The Hot Cache: { "SOL": 143.50, "MELANIA": 0.17 }
        self.price_cache = {} 
        self.bbo_cache = {} # { "SOL": {"bid": 143.49, "ask": 143.51} }
        
        self.imbalance_trackers = {}

        # Track subscriptions to avoid duplicates
        self.subscribed_coins = set()
        self.lock = threading.Lock()

    def start(self):
        """Start the background WebSocket thread"""
        if not WEBSOCKET_AVAILABLE:
            return
            
        self.is_running = True
        # websocket.enableTrace(False) # Uncomment for verbose WS debug
        self.wst = threading.Thread(target=self._run_forever)
        self.wst.daemon = True
        self.wst.start()
        print("[STREAM] WebSocket Client Thread Started")

    def _run_forever(self):
        """Main WebSocket loop with automatic reconnection and SSL fix"""
        while self.is_running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                
                # --- FASTMAIN SSL FIX ---
                # Bypass SSL verification to prevent "CERTIFICATE_VERIFY_FAILED" on macOS
                # This fixes the connection loop issue
                self.ws.run_forever(
                    ping_interval=30, 
                    sslopt={"cert_reqs": ssl.CERT_NONE}
                )
                
            except Exception as e:
                print(f"[STREAM ERROR] Connection failed: {e}. Retrying in 2s...")
                time.sleep(2)

    def subscribe(self, coin: str):
        """Lazy subscription: Only subscribe if we haven't already"""
        with self.lock:
            if coin in self.subscribed_coins:
                return # Already subscribed
            
            self.subscribed_coins.add(coin)
            if coin not in self.imbalance_trackers:
                self.imbalance_trackers[coin] = OrderbookStatsTracker()

            
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self._send_subscribe(coin)
        else:
            # If not connected yet, _on_open will handle it using subscribed_coins set
            pass

    def _send_subscribe(self, coin: str):
        msg = {
            "method": "subscribe",
            "subscription": {"type": "l2Book", "coin": coin}
        }
        try:
            self.ws.send(json.dumps(msg))
            print(f"[STREAM] Subscribed to {coin}")
        except Exception as e:
            print(f"[STREAM ERROR] Failed to subscribe to {coin}: {e}")

    def _on_open(self, ws):
        print("[STREAM] Connected to Hyperliquid Feed")
        # Resubscribe to everything we track
        with self.lock:
            for coin in self.subscribed_coins:
                self._send_subscribe(coin)

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            channel = data.get("channel")
            
            if channel == "l2Book":
                payload = data.get("data", {})
                coin = payload.get("coin")
                levels = payload.get("levels", [])
                
                if coin and len(levels) >= 2:
                    # Extract Best Bid and Ask
                    bids = levels[0]
                    asks = levels[1]
                    
                    if bids and asks:
                        bid_px = float(bids[0]['px'])
                        ask_px = float(asks[0]['px'])
                        # === START CHANGE ===
                        # 1. Capture Sizes
                        bid_sz = float(bids[0]['sz'])
                        ask_sz = float(asks[0]['sz'])
                        
                        mid_px = (bid_px + ask_px) / 2.0
                        
                        self.price_cache[coin] = mid_px
                        
                        # 2. Store in Cache (so Strategy can read it)
                        self.bbo_cache[coin] = {
                            'bid': bid_px, 
                            'ask': ask_px, 
                            'mid': mid_px,
                            'bid_sz': bid_sz,
                            'ask_sz': ask_sz
                        }
                        
                        
                        # 3. Update Tracker
                        if coin in self.imbalance_trackers:
                            # Pass raw Price and Size separately so the tracker 
                            # can calculate both Spread (bps) and Imbalance (ratio)
                            self.imbalance_trackers[coin].update(bid_px, ask_px, bid_sz, ask_sz)
                        
                        # Update Hot Cache
                        self.price_cache[coin] = mid_px
                        self.bbo_cache[coin] = {'bid': bid_px, 'ask': ask_px, 'mid': mid_px}
                        
        except Exception:
            pass # Fail silently on malformed packets for speed

    def _on_error(self, ws, error):
        # Filter out common noise errors
        err_str = str(error)
        if "CERTIFICATE_VERIFY_FAILED" in err_str:
            print(f"[STREAM ERROR] SSL Cert Error (Retrying with fix): {err_str}")
        else:
            print(f"[STREAM ERROR] {err_str}")

    def _on_close(self, ws, close_status_code, close_msg):
        print("[STREAM] Connection Closed. Reconnecting...")

    def get_latest_price(self, coin: str) -> Optional[float]:
        return self.price_cache.get(coin)
    
    def get_latest_bbo(self, coin: str) -> Optional[Dict[str, float]]:
        # 1. Get the raw data from cache
        bbo = self.bbo_cache.get(coin)
        if not bbo:
            return None
            
        # 2. Make a copy so we don't mess up the original cache
        bbo_with_metrics = bbo.copy()
        
        # 3. Pull metrics from the tracker (Z-Scores, means, etc.)
        if coin in self.imbalance_trackers:
            metrics = self.imbalance_trackers[coin].get_metrics()
            # This merges the metrics into our return dictionary
            bbo_with_metrics.update(metrics)
            
        return bbo_with_metrics


# -------------------- Base Provider Interface --------------------

class TelemetryProvider(ABC):
    """Abstract base class for telemetry providers"""
    
    @abstractmethod
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        pass
    
    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, count: int) -> Optional[List[Dict[str, Any]]]:
        """Get historical candles for indicator calculation"""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return provider identifier"""
        pass


# -------------------- Enhanced Hyperliquid Provider --------------------

class HyperliquidTelemetryProvider(TelemetryProvider):
    """
    Telemetry provider using Hybrid Architecture:
    1. Streaming (WebSocket) for real-time prices (Fast Brain)
    2. HTTP Adapter for historical candles and fallback (Slow Brain)
    """
    
    def __init__(self, private_key_hex: str, wallet_address: str, base_url: Optional[str] = None):
        if not HYPERLIQUID_AVAILABLE:
            raise RuntimeError("Hyperliquid adapter not available")
        
        try:
            # 1. Initialize REST Adapter (Slow Brain / Historical)
            self.adapter = HyperliquidAdapter(private_key_hex, wallet_address, base_url)
            
            # Test connection immediately
            test_price = self.adapter.get_price("BTC")
            if test_price is None or test_price <= 0:
                raise RuntimeError(f"Connection test failed - invalid BTC price: {test_price}")
            
            # 2. Initialize Streaming Client (Fast Brain / Real-time)
            self.stream_client = HyperliquidStreamingClient()
            if WEBSOCKET_AVAILABLE:
                self.stream_client.start()
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize HyperliquidTelemetryProvider: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_bbo(self, coin: str) -> Dict[str, float]:
        """
        Get Best Bid/Offer.
        Priority: Stream Cache -> REST API
        """
        # 1. Try Fast Stream
        if self.stream_client.is_running:
            # Ensure we are subscribed
            self.stream_client.subscribe(coin)
            
            bbo = self.stream_client.get_latest_bbo(coin)
            if bbo:
                return bbo
                
        # 2. Fallback to Slow REST
        if hasattr(self.adapter, 'get_bbo'):
            return self.adapter.get_bbo(coin)
            
        price = self.get_price(coin)
        return {'bid': price, 'ask': price, 'mid': price}

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current price.
        Priority: Stream Cache -> REST API
        """
        try:
            # 1. Try Fast Stream (Microsecond latency)
            if self.stream_client.is_running:
                # Lazy Subscription: Ensure we are listening to this symbol
                self.stream_client.subscribe(symbol)
                
                cached_price = self.stream_client.get_latest_price(symbol)
                if cached_price is not None:
                    # Success: Return cached price instantly
                    return cached_price
            
            # 2. Fallback to Slow REST (Millisecond latency)
            # This handles startup warmup or stream disconnection
            price = self.adapter.get_price(symbol, use_cache=True)
            
            # Strict validation
            if price is None or price <= 0:
                print(f"[CRITICAL] Invalid price for {symbol}: {price}")
                return None
            
            return price
            
        except Exception as e:
            print(f"[ERROR] HyperliquidTelemetryProvider.get_price({symbol}) failed: {e}")
            return None
    
    def get_candles(self, symbol: str, timeframe: str, count: int) -> Optional[List[Dict[str, Any]]]:
        """Get candles (Always uses REST API as streams don't provide deep history)"""
        try:
            # Convert timeframe format if needed
            interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
            interval = interval_map.get(timeframe, timeframe)
            
            candles = self.adapter.get_candles(symbol, interval, count)
            
            if not candles:
                print(f"[CRITICAL] No candles returned for {symbol}")
                return None
            
            # Strict validation of candle data with type conversion
            validated_candles = []
            for candle in candles:
                timestamp = candle.get("t")
                open_price = candle.get("o")
                high_price = candle.get("h")
                low_price = candle.get("l")
                close_price = candle.get("c")
                volume = candle.get("v")
                
                # Skip invalid candles
                if None in [timestamp, open_price, high_price, low_price, close_price, volume]:
                    continue
                
                # Convert to float BEFORE validation
                try:
                    open_price = float(open_price)
                    high_price = float(high_price)
                    low_price = float(low_price)
                    close_price = float(close_price)
                    volume = float(volume)
                except (ValueError, TypeError) as e:
                    continue
                
                # Validate price sanity
                prices = [open_price, high_price, low_price, close_price]
                if any(p <= 0 for p in prices):
                    continue
                
                # Validate OHLC logic
                if not (low_price <= open_price <= high_price and 
                        low_price <= close_price <= high_price):
                    continue
                
                validated_candles.append({
                    "t": int(timestamp),
                    "o": open_price,
                    "h": high_price,
                    "l": low_price,
                    "c": close_price,
                    "v": volume
                })
            
            if len(validated_candles) == 0:
                print(f"[CRITICAL] No valid candles after validation for {symbol}")
                return None
            
            return validated_candles
                
        except Exception as e:
            print(f"[ERROR] HyperliquidTelemetryProvider.get_candles({symbol}) failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_source_name(self) -> str:
        return "hyperliquid:hybrid_stream_v2"

    # === INSERT NEW METHODS ===
    def load_history(self, coin: str, history_data: Any):
        """Inject stored history into Fast Brain"""
        if self.stream_client:
            # Create tracker if missing using the NEW class name
            if coin not in self.stream_client.imbalance_trackers:
                self.stream_client.imbalance_trackers[coin] = OrderbookStatsTracker()
            
            # Call the new restore method
            self.stream_client.imbalance_trackers[coin].restore_history(history_data)

    def save_history(self, coin: str) -> Any:
        """Extract history for storage"""
        if self.stream_client and coin in self.stream_client.imbalance_trackers:
            # The new tracker returns a DICT of both spread and imbalance
            return self.stream_client.imbalance_trackers[coin].get_history_snapshot()
        return {}
    # ==========================


# -------------------- Data Validation Functions --------------------

def validate_series_alignment(series_a, series_b, min_points: int = 50) -> Tuple[Any, Any, bool]:
    """
    Validates and aligns two time series for analysis.
    Supports both Pandas Series and Numpy Arrays (Phase 3+ Compatibility).
    """
    try:
        # Check for empty inputs
        if len(series_a) < min_points or len(series_b) < min_points:
            return series_a, series_b, False
        
        # ---------------------------------------------------------
        # [PHASE 3 FIX] Handle Numpy Arrays (Buffer Mode)
        # ---------------------------------------------------------
        is_numpy = isinstance(series_a, np.ndarray) or isinstance(series_b, np.ndarray)

        if is_numpy:
            # Ensure both are numpy arrays
            arr_a = np.array(series_a) if not isinstance(series_a, np.ndarray) else series_a
            arr_b = np.array(series_b) if not isinstance(series_b, np.ndarray) else series_b
            
            # Align lengths (simple truncation from end)
            if len(arr_a) != len(arr_b):
                min_len = min(len(arr_a), len(arr_b))
                aligned_a = arr_a[-min_len:]
                aligned_b = arr_b[-min_len:]
            else:
                aligned_a = arr_a
                aligned_b = arr_b
                min_len = len(aligned_a)

            # Check for NaNs
            if np.isnan(aligned_a).sum() > min_len * 0.1 or np.isnan(aligned_b).sum() > min_len * 0.1:
                return aligned_a, aligned_b, False
                
            # Check for constant values (std dev == 0)
            if np.std(aligned_a) == 0 or np.std(aligned_b) == 0:
                return aligned_a, aligned_b, False

            return aligned_a, aligned_b, True

        # ---------------------------------------------------------
        # [LEGACY] Handle Pandas Series (Original Logic)
        # ---------------------------------------------------------
        else:
            if len(series_a) != len(series_b):
                min_len = min(len(series_a), len(series_b))
                aligned_a = series_a.tail(min_len).reset_index(drop=True)
                aligned_b = series_b.tail(min_len).reset_index(drop=True)
            else:
                aligned_a = series_a.reset_index(drop=True)
                aligned_b = series_b.reset_index(drop=True)
                min_len = len(aligned_a)
            
            if aligned_a.isna().sum() > min_len * 0.1 or aligned_b.isna().sum() > min_len * 0.1:
                return aligned_a, aligned_b, False
            
            if aligned_a.std() == 0 or aligned_b.std() == 0:
                return aligned_a, aligned_b, False
            
            return aligned_a, aligned_b, True
        
    except Exception as e:
        # Only print error if it's NOT the expected attribute error we just fixed
        if "reset_index" not in str(e): 
            print(f"[ERROR] Series validation failed: {e}")
        return series_a, series_b, False

def calculate_regression_metrics_with_live_prices(
    asset_a_historical: pd.Series, 
    asset_b_historical: pd.Series,
    current_live_price_a: float,
    current_live_price_b: float,
    regression_window: int = 250, 
    zscore_window: int = 60
) -> Dict[str, Any]:
    """
    Calculate regression using historical data, apply to current live prices
    """
    # ------------------------------------------------------------------
    # [PHASE 5 FIX] Handle Numpy Arrays from Buffer System
    # This prevents 'TypeError: cannot concatenate object of type numpy.ndarray'
    # ------------------------------------------------------------------
    if isinstance(asset_a_historical, np.ndarray):
        asset_a_historical = pd.Series(asset_a_historical)
    if isinstance(asset_b_historical, np.ndarray):
        asset_b_historical = pd.Series(asset_b_historical)
    # ------------------------------------------------------------------

    # Validate input data first
    aligned_a, aligned_b, is_valid = validate_series_alignment(
        asset_a_historical, asset_b_historical, min_points=max(regression_window//2, 50)
    )
    if not is_valid:
        return _fallback_regression_metrics(current_live_price_a, current_live_price_b)
    
    if not REGRESSION_AVAILABLE:
        return _fallback_regression_metrics(current_live_price_a, current_live_price_b)
    
    try:
        # Adjust windows based on available data
        effective_regression_window = min(regression_window, len(aligned_a) - 20)
        effective_zscore_window = min(zscore_window, len(aligned_a) // 3)
        
        # Initialize regression components
        regression_core = RegressionCore(
            window_size=effective_regression_window,
            min_periods=min(30, effective_regression_window // 4),
            rolling=True
        )
        
        zscore_engine = ZScoreEngine(
            window_size=effective_zscore_window,
            min_periods=min(10, effective_zscore_window // 3),
            calculation_method='rolling'
        )
        
        # Run regression analysis on historical data only
        regression_results = regression_core.fit_regression(aligned_a, aligned_b)
        
        # Get regression model parameters
        latest_params = regression_core.get_latest_relationship()
        alpha = latest_params.get('intercept', 0)
        beta = latest_params.get('slope', 1)
        r_squared = latest_params.get('r_squared', 0)
        
        # Apply regression model to current live prices
        current_fair_value = alpha + beta * current_live_price_b
        current_spread = current_live_price_a - current_fair_value
        
        # Get historical spreads for z-score calculation
        spread_series = regression_core.get_spread_series()
        
        if len(spread_series) > effective_zscore_window:
            # Calculate z-score statistics from historical spreads
            recent_spreads = spread_series.tail(effective_zscore_window)
            spread_mean = recent_spreads.mean()
            spread_std = recent_spreads.std()
            
            # Apply to current spread
            if spread_std > 0:
                current_zscore = (current_spread - spread_mean) / spread_std
            else:
                current_zscore = 0.0
        else:
            current_zscore = 0.0
        
        # Calculate health metrics
        correlation = calculate_correlation(aligned_a, aligned_b)
        hedge_ratio_series = regression_core.get_hedge_ratio_series()
        beta_drift = calculate_beta_drift(hedge_ratio_series)
        health_regime = classify_health_regime(correlation, beta_drift, r_squared)
        
        # Validate consistency
        validation = validate_regression_consistency(
            current_zscore, current_spread, alpha, beta,
            current_live_price_a, current_live_price_b
        )
        
        if not validation['is_consistent']:
            # Log warning but continue
            pass
        
        # Map z-score to trading regime
        trading_regime = map_zscore_to_regime(current_zscore)
        
        result = {
            'current_prices': {
                'A': current_live_price_a,
                'B': current_live_price_b
            },
            'fair_value': current_fair_value,
            'residual': current_spread,
            'z_score': current_zscore,
            'regression_params': {
                'alpha': float(alpha),
                'beta': float(beta),
                'r_squared': float(r_squared)
            },
            'health_metrics': {
                'correlation': correlation,
                'beta_drift_pct': beta_drift,
                'regime': health_regime
            },
            'trading_signals': {
                'regime': trading_regime,
                'confidence': calculate_signal_confidence(current_zscore, r_squared),
                'signal': get_trading_signal(current_zscore),
                'signal_strength': abs(current_zscore)
            },
            'data_quality': {
                'sufficient_for_trading': assess_data_quality(len(spread_series), r_squared, correlation),
                'total_periods': len(aligned_a),
                'historical_periods': len(spread_series),
                'validation_passed': validation['is_consistent'],
                'validation_warnings': validation['warnings'],
                'live_prices_used': True,
                'price_source': 'live_execution_context'
            }
        }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Regression calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return _fallback_regression_metrics(current_live_price_a, current_live_price_b)


def _fallback_regression_metrics(current_price_a: float, current_price_b: float) -> Dict[str, Any]:
    """Fallback uses current live prices, not synthetic data"""
    return {
        'current_prices': {'A': current_price_a, 'B': current_price_b},
        'fair_value': current_price_a,
        'residual': 0.0,
        'z_score': 0.0,
        'regression_params': {'alpha': 0.0, 'beta': 1.0, 'r_squared': 0.0},
        'health_metrics': {'correlation': 0.0, 'beta_drift_pct': 0.0, 'regime': 'S3_shift_risk'},
        'trading_signals': {'regime': 'RANGING', 'confidence': 0.0, 'signal': 'neutral', 'signal_strength': 0.0},
        'data_quality': {
            'sufficient_for_trading': False, 
            'total_periods': 0, 
            'historical_periods': 0, 
            'validation_passed': False, 
            'validation_warnings': ['Fallback mode - using live prices'],
            'live_prices_used': True,
            'price_source': 'live_fallback'
        }
    }


# -------------------- Utility Functions --------------------

def extract_volatility_from_regression(asset_a_data, asset_b_data, regression_window: int = 200) -> Tuple[float, float]:
    """Extract volatility metrics using same window as regression analysis"""
    try:
        # ---------------------------------------------------------
        # [PHASE 3 FIX] Handle Numpy Arrays
        # ---------------------------------------------------------
        if isinstance(asset_a_data, np.ndarray) or isinstance(asset_b_data, np.ndarray):
            # Convert to numpy if mixed
            arr_a = np.array(asset_a_data)
            arr_b = np.array(asset_b_data)
            
            # Slice recent window
            recent_a = arr_a[-regression_window:] if len(arr_a) > regression_window else arr_a
            recent_b = arr_b[-regression_window:] if len(arr_b) > regression_window else arr_b
            
            # Calculate pct_change equivalent: (price[t] - price[t-1]) / price[t-1]
            # using diff / slice
            if len(recent_a) < 2 or len(recent_b) < 2:
                return 0.0, 0.0
                
            returns_a = np.diff(recent_a) / recent_a[:-1]
            returns_b = np.diff(recent_b) / recent_b[:-1]
            
            # Std dev of returns
            vol_a = float(np.std(returns_a))
            vol_b = float(np.std(returns_b))
            
            return vol_a, vol_b

        # ---------------------------------------------------------
        # [LEGACY] Handle Pandas Series
        # ---------------------------------------------------------
        else:
            recent_a = asset_a_data.tail(regression_window)
            recent_b = asset_b_data.tail(regression_window)
            
            returns_a = recent_a.pct_change().dropna()
            returns_b = recent_b.pct_change().dropna()
            
            volatility_a = returns_a.std() if len(returns_a) > 1 else 0.0
            volatility_b = returns_b.std() if len(returns_b) > 1 else 0.0
            
            return float(volatility_a), float(volatility_b)
        
    except Exception as e:
        if "tail" not in str(e):
            print(f"[ERROR] Volatility extraction failed: {e}")
        return 0.0, 0.0


def calculate_correlation(asset_a: pd.Series, asset_b: pd.Series, window: int = 100) -> float:
    """Calculate robust correlation between two asset series"""
    try:
        aligned_a, aligned_b, is_valid = validate_series_alignment(asset_a, asset_b, min_points=20)
        if not is_valid:
            return 0.0
        
        returns_a = aligned_a.pct_change().dropna()
        returns_b = aligned_b.pct_change().dropna()
        
        if len(returns_a) < 10 or len(returns_b) < 10:
            return 0.0
        
        effective_window = min(window, len(returns_a), len(returns_b))
        
        if effective_window < 10:
            return 0.0
        
        recent_returns_a = returns_a.tail(effective_window)
        recent_returns_b = returns_b.tail(effective_window)
        
        correlation = recent_returns_a.corr(recent_returns_b)
        
        if pd.isna(correlation):
            return 0.0
        
        correlation = max(-1.0, min(1.0, float(correlation)))
        return correlation
        
    except Exception as e:
        print(f"[ERROR] Correlation calculation failed: {e}")
        return 0.0


def calculate_beta_drift(hedge_ratio_series: pd.Series, window: int = 30) -> float:
    """Calculate beta drift with robust error handling"""
    try:
        min_required = window * 2 + 10
        if len(hedge_ratio_series) < min_required:
            return 0.0
        
        clean_series = hedge_ratio_series.dropna()
        if len(clean_series) < min_required:
            return 0.0
        
        effective_window = min(window, len(clean_series) // 3)
        if effective_window < 10:
            return 0.0
        
        recent_beta = clean_series.tail(effective_window).mean()
        baseline_beta = clean_series.iloc[-effective_window*2:-effective_window].mean()
        
        if abs(baseline_beta) < 1e-4:
            return 0.0
        
        if pd.isna(recent_beta) or pd.isna(baseline_beta):
            return 0.0
        
        drift_pct = abs((recent_beta - baseline_beta) / baseline_beta) * 100
        drift_pct = min(drift_pct, 999.0)
        
        return float(drift_pct)
        
    except Exception as e:
        print(f"[ERROR] Beta drift calculation failed: {e}")
        return 0.0


def classify_health_regime(correlation: float, beta_drift_pct: float, r_squared: float) -> str:
    """Classify regression health with appropriate thresholds"""
    try:
        if correlation >= 0.7 and beta_drift_pct <= 10.0 and r_squared >= 0.4:
            result = "S1_stable"
        elif correlation < 0.4 or beta_drift_pct > 50.0 or r_squared < 0.2:
            result = "S3_shift_risk"
        else:
            result = "S2_mild_drift"
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Health classification failed: {e}")
        return "S3_shift_risk"


def validate_regression_consistency(zscore: float, spread: float, alpha: float, beta: float, 
                                  price_a: float, price_b: float) -> Dict[str, Any]:
    """Validate that regression parameters are internally consistent"""
    validation_results = {
        'is_consistent': True,
        'warnings': [],
        'errors': []
    }
    
    try:
        predicted_a = alpha + beta * price_b
        actual_spread = price_a - predicted_a
        
        spread_diff = abs(spread - actual_spread)
        if spread_diff > abs(spread) * 0.1:
            validation_results['warnings'].append(f"Spread inconsistency: calculated={actual_spread:.4f}, reported={spread:.4f}")
        
        if abs(beta) > 10000:
            validation_results['errors'].append(f"Extreme beta value: {beta:.2f}")
            validation_results['is_consistent'] = False
        
        if abs(zscore) > 10:
            validation_results['warnings'].append(f"Extreme z-score: {zscore:.2f}")
        
        if predicted_a <= 0:
            validation_results['errors'].append(f"Regression predicts negative price: {predicted_a:.4f}")
            validation_results['is_consistent'] = False
        
    except Exception as e:
        validation_results['errors'].append(f"Validation calculation failed: {str(e)}")
        validation_results['is_consistent'] = False
    
    return validation_results


def map_zscore_to_regime(zscore: float) -> str:
    """Map z-score to trading regime"""
    if zscore > 2.5:
        return "STRONG_DOWNTREND"
    elif zscore > 1.5:
        return "DOWNTRENDING"
    elif zscore > 1.0:
        return "DOWNBREAKOUT"
    elif zscore < -2.5:
        return "STRONG_UPTREND"
    elif zscore < -1.5:
        return "UPTRENDING"
    elif zscore < -1.0:
        return "UPBREAKOUT"
    else:
        return "RANGING"


def get_trading_signal(zscore: float, entry_threshold: float = 2.0, exit_threshold: float = 0.5) -> str:
    """Get trading signal based on z-score"""
    if zscore > entry_threshold:
        return "short_entry"
    elif zscore < -entry_threshold:
        return "long_entry"
    elif abs(zscore) < exit_threshold:
        return "exit"
    else:
        return "neutral"


def calculate_signal_confidence(zscore: float, r_squared: float) -> float:
    """Calculate signal confidence"""
    zscore_confidence = min(abs(zscore) / 3.0, 1.0)
    regression_confidence = max(0.0, min(1.0, r_squared))
    return (0.6 * zscore_confidence + 0.4 * regression_confidence)


def assess_data_quality(data_points: int, r_squared: float, correlation: float) -> bool:
    """Enhanced data quality assessment"""
    min_points = 50
    min_r_squared = 0.01
    min_correlation = 0.15
    
    quality_ok = (data_points >= min_points and 
                  r_squared >= min_r_squared and 
                  abs(correlation) >= min_correlation)
    
    return quality_ok


def _analyze_timeframe_from_buffers(data_buffers: 'MultiTimeframeBuffers', 
                                   timeframe: str, config: Dict[str, Any],
                                   current_base_price: float,
                                   current_quote_price: float) -> Dict[str, Any]:
    """
    [PHASE 1 UPDATED] Analyze a specific timeframe with GLOBAL HURST INTEGRATION
    """
    
    try:
        # Get aligned historical data only
        base_df, quote_df = data_buffers.get_aligned_analysis_data(timeframe)
        
        if base_df is None or quote_df is None:
            return _get_fallback_buffer_analysis(timeframe, current_base_price, current_quote_price)
        
        # Get timeframe-specific configuration
        buffer_config = config.get("multi_timeframe_buffers", {})
        tf_config = buffer_config.get("timeframes", {}).get(timeframe, {})
        
        regression_window = tf_config.get("regression_window", 200)
        zscore_window = tf_config.get("zscore_window", 40)
        
        # --- HURST CONFIGURATION & SAFETY CHECK ---
        hurst_window = tf_config.get("hurst_window", 100)
        hurst_max_lag = tf_config.get("hurst_max_lag", 20)
        
        # [SAFETY] Ensure window is large enough for the calculation (Need 2x Lag + buffer)
        min_hurst_data = hurst_max_lag * 2 + 5
        
        if hurst_window < min_hurst_data:
            # Auto-expand window if config is too small to prevent "Insufficient Data" error
            hurst_window = min_hurst_data

        # Extract historical price series
        base_prices = base_df['close'].values
        quote_prices = quote_df['close'].values
        
        # Calculate regression metrics using historical data + current live prices
        regression_metrics = calculate_regression_metrics_with_live_prices(
            base_prices, quote_prices, current_base_price, current_quote_price,
            regression_window, zscore_window
        )

        # [NEW] Calculate RSI (14-period standard)
        # We use the base asset (SOL) closing prices for RSI
        rsi_val = calculate_rsi(base_df['close'], period=14)
        
        # --- GLOBAL HURST CALCULATION (ROBUST) ---
        hurst_primary = 0.5   # Default to random walk
        hurst_reference = 0.5
        
        # Check if we have enough total data in the buffer
        if len(base_prices) >= min_hurst_data:
            try:
                # Use log-prices to assess rate of diffusion
                # We slice carefully to ensure we don't grab an empty array or wrong window
                slice_idx = -int(hurst_window) 
                
                base_log_prices = np.log(base_prices[slice_idx:])
                quote_log_prices = np.log(quote_prices[slice_idx:])
                
                # Pass Log-Prices to the Enhanced Hurst Calculator
                hurst_primary = calculate_hurst(base_log_prices, max_lag=hurst_max_lag)
                hurst_reference = calculate_hurst(quote_log_prices, max_lag=hurst_max_lag)
                
                # Debug Proof: Confirm valid calculation
                print(f"[DEBUG HURST RESULT] Primary: {hurst_primary:.4f} | Ref: {hurst_reference:.4f}")
            except Exception as e:
                print(f"[HURST EXEC ERROR] {e}")
        else:
            # Only print this if buffers are actually empty (startup)
            if len(base_prices) > 0:
                print(f"[DEBUG HURST] Buffer filling... {len(base_prices)}/{min_hurst_data}")
        
        # Inject Hurst values into the health metrics for dashboard/strategy use
        regression_metrics['health_metrics']['hurst_primary'] = float(hurst_primary)
        regression_metrics['health_metrics']['hurst_reference'] = float(hurst_reference)
        # --------------------------------------

        # Extract volatility using same window as regression
        volatility_a, volatility_b = extract_volatility_from_regression(
            base_prices, quote_prices, regression_window
        )
        
        # Calculate regime assessment using pre-calculated correlation and volatility
        actual_beta = regression_metrics.get('regression_params', {}).get('beta', 1.0)
        print(f"[DEBUG BETA_EXTRACT] First call - Timeframe: {timeframe}, Beta: {actual_beta}")
        regime_assessment = calculate_regime_assessment(
            base_prices, quote_prices, timeframe, 
            regression_metrics['health_metrics']['correlation'],
            volatility_a, volatility_b, actual_beta
        )
        
        # Extract volume data from historical data
        volume_data = {
            'base_volume': float(base_df['volume'].iloc[-1]) if len(base_df) > 0 else 0.0,
            'quote_volume': float(quote_df['volume'].iloc[-1]) if len(quote_df) > 0 else 0.0
        }
        
        result = {
            'timeframe': timeframe,
            'regression_metrics': regression_metrics,
            'regime_assessment': regime_assessment,
            'volume_data': volume_data,
            'data_points_analyzed': len(base_prices),
            'rsi': rsi_val,
            'analysis_success': True,
            'zscore_window': zscore_window,
            'live_prices_integrated': True
        }
        
        return result
        
    except Exception as e:
        print(f"[BUFFER ANALYSIS ERROR] {timeframe} analysis failed: {e}")
        return _get_fallback_buffer_analysis(timeframe, current_base_price, current_quote_price)


def _get_fallback_buffer_analysis(timeframe: str, current_base_price: float, current_quote_price: float) -> Dict[str, Any]:
    """Fallback analysis using current live prices"""
    return {
        'timeframe': timeframe,
        'regression_metrics': _fallback_regression_metrics(current_base_price, current_quote_price),
        'regime_assessment': _fallback_regime_assessment(timeframe, 20),
        'volume_data': {'base_volume': 0.0, 'quote_volume': 0.0},
        'data_points_analyzed': 0,
        'analysis_success': False,
        'zscore_window': 40,
        'live_prices_integrated': True
    }


def _combine_triple_regime_assessment(tactical_regime: Dict[str, Any], primary_regime: Dict[str, Any], 
                                    htf_regime: Dict[str, Any], tactical_tf: str, primary_tf: str, 
                                    htf_tf: str) -> Dict[str, Any]:
    """Combine regime assessment with triple-timeframe context"""
    
    # Use primary timeframe as base
    combined = primary_regime.copy()
    
    # Add tactical context (5s)
    combined["tactical_context"] = {
        "timeframe": tactical_tf,
        "both_trending_up": tactical_regime.get('both_trending_up', False),
        "both_trending_down": tactical_regime.get('both_trending_down', False),
        "divergent_trends": tactical_regime.get('divergent_trends', False),
        # REMOVED: "safe_for_mean_reversion" - decision logic moved to strategy layer
        "trend_strength": tactical_regime.get('trend_strength', {})
    }
    
    # Add HTF context (1h) 
    combined["htf_context"] = {
        "timeframe": htf_tf,
        "both_trending_up": htf_regime.get('both_trending_up', False),
        "both_trending_down": htf_regime.get('both_trending_down', False),
        "divergent_trends": htf_regime.get('divergent_trends', False),
        # REMOVED: "safe_for_mean_reversion" - decision logic moved to strategy layer
        "trend_strength": htf_regime.get('trend_strength', {})
    }
    
    return combined


def _create_triple_buffer_error_response(symbol: str, error_msg: str, asof: str) -> Dict[str, Any]:
    """Create error response when triple buffer analysis fails"""
    return {
        "asof": asof,
        "source": "multi_timeframe_buffers:error",
        "is_stale": True,
        "stale_seconds": 0,
        "error": error_msg,
        "price": {"value": 0.0, "type": "error", "quote_ccy": "UNKNOWN", "actual_price_a": 0.0, "actual_price_b": 0.0},
        "regression_5s": _get_fallback_regression_formatted("5s"),
        "regression": _get_fallback_regression_formatted("1m"),
        "regression_htf": _get_fallback_regression_formatted("1h"),
        "regime_assessment": _fallback_regime_assessment("1m", 20),
        "volume_data": {"primary": 0.0, "reference": 0.0},
        "data_quality_summary": {"tactical_sufficient": False, "primary_sufficient": False, "htf_sufficient": False},
        "indicators": {"zscore": {"value": 0.0, "method": "error"}},
        "asset_pair_info": {"base_asset": "UNKNOWN", "quote_asset": "UNKNOWN", "asset_type": "error"}
    }


def _format_regression_data(regression_metrics: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
    """Format regression data - ALWAYS capture calculated values regardless of quality assessment"""
    
    try:
        # FIXED: Check if we have calculated data, ignore quality assessment
        has_calculated_data = (regression_metrics and 
                              'z_score' in regression_metrics and 
                              'regression_params' in regression_metrics)
        
        # ALWAYS use calculated data if it exists (regardless of quality flags)
        if has_calculated_data:
            regression_params = regression_metrics.get('regression_params', {})
            health_metrics = regression_metrics.get('health_metrics', {})
            data_quality = regression_metrics.get('data_quality', {})
            
            formatted = {
                "zscore": float(regression_metrics.get('z_score', 0.0)),
                "spread": float(regression_metrics.get('residual', 0.0)),
                "alpha": float(regression_params.get('alpha', 0.0)),
                "beta": float(regression_params.get('beta', 1.0)),
                "r_squared": float(regression_params.get('r_squared', 0.0)),
                "correlation": float(health_metrics.get('correlation', 0.0)),
                "window_size": data_quality.get('historical_periods', 0),
                "health_status": health_metrics.get('regime', 'unknown'),
                "beta_drift_pct": float(health_metrics.get('beta_drift_pct', 0.0)),
                "timeframe": timeframe,
                # NEW: Preserve quality assessment for signal generation (don't use for data capture)
                "quality_sufficient_for_trading": bool(data_quality.get('sufficient_for_trading', False))
            }
            
            return formatted
        else:
            return _get_fallback_regression_formatted(timeframe)
        
    except Exception as e:
        print(f"[FORMAT ERROR] Failed to format {timeframe} regression data: {e}")
        import traceback
        traceback.print_exc()
        return _get_fallback_regression_formatted(timeframe)


def _get_fallback_regression_formatted(timeframe: str) -> Dict[str, Any]:
    """Fallback regression data in schema format"""
    return {
        "zscore": 0.0,
        "spread": 0.0,
        "alpha": 0.0,
        "beta": 1.0,
        "r_squared": 0.0,
        "correlation": 0.0,
        "window_size": 0,
        "health_status": "S3_shift_risk",
        "beta_drift_pct": 0.0,
        "timeframe": timeframe
    }

def calculate_regime_assessment(asset_a_data, asset_b_data, 
                              timeframe: str = "1m", correlation: float = 0.0,
                              volatility_a: float = 0.0, volatility_b: float = 0.0,
                              regression_beta: float = 1.0) -> Dict[str, Any]:
    """Calculate regime assessment using pre-calculated correlation from regression analysis"""
    
    # ------------------------------------------------------------------
    # [PHASE 5 FIX] Convert to Pandas Series if input is Numpy Array
    # This prevents 'AttributeError: numpy.ndarray object has no attribute tail'
    # ------------------------------------------------------------------
    if isinstance(asset_a_data, np.ndarray):
        asset_a_data = pd.Series(asset_a_data)
    if isinstance(asset_b_data, np.ndarray):
        asset_b_data = pd.Series(asset_b_data)
    # ------------------------------------------------------------------

    lookback_periods = 20  # Default lookback for windowing operations
    
    # Validate input data
    aligned_a, aligned_b, is_valid = validate_series_alignment(asset_a_data, asset_b_data, min_points=lookback_periods)
    if not is_valid:
        return _fallback_regime_assessment(timeframe, lookback_periods)
    
    try:
        # Get recent data for trend analysis
        # [NOTE] aligned_a/b are now guaranteed to be Pandas Series because of the conversion above
        recent_a = aligned_a.tail(lookback_periods)
        recent_b = aligned_b.tail(lookback_periods)
        
        # Calculate price changes and trends
        returns_a = recent_a.pct_change().dropna()
        returns_b = recent_b.pct_change().dropna()
        
        if len(returns_a) < 5 or len(returns_b) < 5:
            return _fallback_regime_assessment(timeframe, lookback_periods)
        
        # Calculate trend strength using robust method
        x = np.arange(len(recent_a))
        trend_a = calculate_trend_strength(recent_a.values, x)
        trend_b = calculate_trend_strength(recent_b.values, x)
        
        # Trend direction analysis with adaptive thresholds
        vol_a = returns_a.std() * 100
        vol_b = returns_b.std() * 100
        trend_threshold = max(0.2, min(vol_a, vol_b) * 0.5)
        
        trending_up_a = trend_a > trend_threshold
        trending_down_a = trend_a < -trend_threshold
        trending_up_b = trend_b > trend_threshold  
        trending_down_b = trend_b < -trend_threshold
        
        both_trending_up = trending_up_a and trending_up_b
        both_trending_down = trending_down_a and trending_down_b
        divergent_trends = (trending_up_a and trending_down_b) or (trending_down_a and trending_up_b)
        
        # Use pre-calculated correlation from regression analysis
        returns_correlation = correlation
        
        # Volatility ratio calculation
        if volatility_a > 0 and volatility_b > 0:
            volatility_ratio = min(volatility_a, volatility_b) / max(volatility_a, volatility_b)
        else:
            volatility_ratio = 0.0
        
        synchronized_volatility = volatility_ratio > 0.4
        
        # Enhanced volatility sync assessment
        if volatility_ratio > 0.8:
            volatility_sync_assessment = "highly_synchronized"
        elif volatility_ratio > 0.6:
            volatility_sync_assessment = "synchronized"
        elif volatility_ratio > 0.4:
            volatility_sync_assessment = "moderately_synchronized"
        else:
            volatility_sync_assessment = "divergent"
            
        # ------------------------------------------------------------------
        # [ENHANCED] Calculate All Hurst Values and Individual Trend Betas
        # ------------------------------------------------------------------
        # Individual asset Hurst values
        hurst_val = calculate_hurst(recent_a.values)
        hurst_reference = calculate_hurst(recent_b.values)  # NEW: Secondary asset Hurst
        
        # Individual asset trend betas (directional slopes)
        sol_trend_beta = calculate_individual_trend_beta(recent_a.values)
        melania_trend_beta = calculate_individual_trend_beta(recent_b.values)
        
        # Pair spread Hurst (using actual regression beta)
        actual_beta = regression_beta if abs(regression_beta) > 0.01 else 1.0
        hurst_spread = calculate_pair_hurst(recent_a.values, recent_b.values, actual_beta)
        print(f"[DEBUG PAIR_HURST] Beta: {actual_beta:.2f}, Spread Hurst: {hurst_spread:.4f}")
        # [NEW] Calculate trend direction of the SPREAD itself
        # The spread is: SOL - (beta * MELANIA)
        spread_series = recent_a.values - (actual_beta * recent_b.values)
        spread_trend_beta = calculate_individual_trend_beta(spread_series)
        print(f"[DEBUG SPREAD_TREND] Spread Trend Beta: {spread_trend_beta:.6f}")
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        
        result = {
            "timeframe": timeframe,
            "lookback_periods": lookback_periods,
            "both_trending_up": both_trending_up,
            "both_trending_down": both_trending_down,
            "divergent_trends": divergent_trends,
            "synchronized_volatility": synchronized_volatility,
            
            # Enhanced Hurst Analysis
            "hurst": float(hurst_val),                    # Primary asset (SOL)
            "hurst_reference": float(hurst_reference),    # Secondary asset (MELANIA)
            "hurst_spread": float(hurst_spread),          # Pair relationship
            
            # Individual Trend Directions
            "sol_trend_beta": float(sol_trend_beta),      # SOL directional slope
            "melania_trend_beta": float(melania_trend_beta),  # MELANIA directional slope
            "spread_trend_beta": float(spread_trend_beta),  # [NEW] Pair spread directional slope
            
            "trend_strength": {
                "primary": float(trend_a),
                "reference": float(trend_b),
                "correlation": float(returns_correlation)
            },
            "volatility_sync": {
                "ratio": float(volatility_ratio),
                "assessment": volatility_sync_assessment
            },
            "adaptive_threshold": float(trend_threshold)
        }
        
        # [DEBUG TRACER] Check what we are actually returning
        if "hurst" in result:
             print(f"[DEBUG] ✅ Regime Output contains Hurst: {result['hurst']}")
        else:
             print(f"[DEBUG] ❌ Regime Output MISSING Hurst. Keys: {list(result.keys())}")

        
        return result
        
    except Exception as e:
        print(f"[ERROR] Regime assessment calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return _fallback_regime_assessment(timeframe, lookback_periods)


def _fallback_regime_assessment(timeframe: str, lookback_periods: int) -> Dict[str, Any]:
    """Fallback regime assessment when calculation fails"""
    # [ENHANCED] Updated to include all new Hurst and trend beta fields
    return {
        "timeframe": timeframe,
        "lookback_periods": lookback_periods,
        "both_trending_up": False,
        "both_trending_down": False,
        "divergent_trends": False,
        "synchronized_volatility": True,
        
        # Enhanced Hurst Analysis (all default to neutral)
        "hurst": 0.5,                    # Primary asset neutral
        "hurst_reference": 0.5,          # Secondary asset neutral
        "hurst_spread": 0.5,             # Pair relationship neutral
        
        # Individual Trend Directions (all default to no trend)
        "sol_trend_beta": 0.0,           # SOL no directional trend
        "melania_trend_beta": 0.0,       # MELANIA no directional trend
        "spread_trend_beta": 0.0,        # [NEW] Pair spread no directional trend
        
        "trend_strength": {
            "primary": 0.0,
            "reference": 0.0,
            "correlation": 0.0
        },
        "volatility_sync": {
            "ratio": 1.0,
            "assessment": "unknown"
        },
        "adaptive_threshold": 0.0
    }


def calculate_trend_strength(prices: np.ndarray, x_values: np.ndarray) -> float:
    """Calculate trend strength with outlier handling"""
    try:
        if len(prices) < 5:
            return 0.0
        
        # Remove outliers using IQR method
        q1, q3 = np.percentile(prices, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter outliers
        mask = (prices >= lower_bound) & (prices <= upper_bound)
        if np.sum(mask) < 5:
            filtered_prices = prices
            filtered_x = x_values
        else:
            filtered_prices = prices[mask]
            filtered_x = x_values[mask]
        
        # Calculate slope using robust linear regression
        slope, _ = np.polyfit(filtered_x, filtered_prices, 1)
        avg_price = np.mean(filtered_prices)
        
        if avg_price > 0:
            normalized_slope = (slope / avg_price) * 100
            return float(normalized_slope)
        else:
            return 0.0
            
    except Exception as e:
        print(f"[ERROR] Trend strength calculation failed: {e}")
        return 0.0


# REMOVED: assess_mean_reversion_safety function
# Decision-making logic has been moved to trading_strategies.py
# The telemetry layer now only provides raw data metrics


# -------------------- LEGACY TELEMETRY FUNCTIONS (for backward compatibility) --------------------

def get_telemetry(symbol: str, symbol_info: Dict[str, Any], provider: TelemetryProvider,
                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced telemetry function with live prices and strict validation
    """
    
    cfg = config or {}
    
    # Get configuration parameters
    pair_cfg = cfg.get("pair_trading", {})
    regression_window = pair_cfg.get("regression_window", 250)
    zscore_window = pair_cfg.get("zscore_window", 60)
    
    # Get regime assessment parameters
    regime_cfg = cfg.get("regime_assessment", {})
    regime_lookback = regime_cfg.get("lookback_periods", 20)
    
    # Determine asset type
    asset_type = symbol_info.get("type", "single")
    base = symbol_info.get("base", symbol)
    quote = symbol_info.get("quote", "USDT")
    
    timestamp_start = time.time()
    asof = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    
    if asset_type == "pair":
        
        # Get current live prices first
        base_price = provider.get_price(base)
        quote_price = provider.get_price(quote)
        
        if base_price is None or quote_price is None:
            print(f"[CRITICAL] Cannot get live prices: base={base_price}, quote={quote_price}")
            raise RuntimeError(f"CRITICAL: Live prices unavailable for {base}/{quote}")
        
        # Get historical data for regression analysis
        required_bars = max(regression_window, zscore_window, regime_lookback) + 100
        
        base_candles = provider.get_candles(base, "1m", required_bars)
        quote_candles = provider.get_candles(quote, "1m", required_bars)
        
        if not base_candles or not quote_candles:
            print(f"[CRITICAL] Cannot get historical data for {base}/{quote}")
            raise RuntimeError(f"CRITICAL: Historical data unavailable for {base}/{quote}")
        
        # Extract volume data from candles
        try:
            base_volumes = [float(c['v']) for c in base_candles] if base_candles else []
            quote_volumes = [float(c['v']) for c in quote_candles] if quote_candles else []
            
            current_base_volume = base_volumes[-1] if base_volumes else 0.0
            current_quote_volume = quote_volumes[-1] if quote_volumes else 0.0
            
        except Exception as e:
            print(f"[ERROR] Volume extraction failed: {e}")
            current_base_volume = 0.0
            current_quote_volume = 0.0
        
        # Convert to pandas series with validation
        try:
            base_data = pd.Series([float(c['c']) for c in base_candles])
            quote_data = pd.Series([float(c['c']) for c in quote_candles])
            
            # Calculate regression metrics using historical data + current live prices
            regression_metrics = calculate_regression_metrics_with_live_prices(
                base_data, quote_data, base_price, quote_price,
                regression_window, zscore_window
            )
            
            # Extract volatility using same window as regression  
            volatility_a, volatility_b = extract_volatility_from_regression(
                base_data, quote_data, regression_window
            )
            
            # Calculate regime assessment using pre-calculated correlation and volatility
            # Extract primary timeframe from config (don't hardcode)
            buffer_config = cfg.get("multi_timeframe_buffers", {})
            primary_timeframe = buffer_config.get("primary_timeframe", "1m")
            actual_beta = regression_metrics.get('regression_params', {}).get('beta', 1.0)
            print(f"[DEBUG BETA_EXTRACT] Second call - Timeframe: {primary_timeframe}, Beta: {actual_beta}")
            regime_assessment = calculate_regime_assessment(
                base_data, quote_data, primary_timeframe,  # ✅ CORRECT VARIABLES
                regression_metrics['health_metrics']['correlation'],
                volatility_a, volatility_b, actual_beta
            )
            
            # Ensure prices in result are live prices
            regression_metrics['current_prices']['A'] = base_price
            regression_metrics['current_prices']['B'] = quote_price
            
            # Add data quality summary
            data_quality_summary = {
                'data_points_used': len(base_data),
                'data_validation_passed': True,
                'regression_consistency': regression_metrics['data_quality'].get('validation_passed', False),
                'sufficient_for_trading': regression_metrics['data_quality'].get('sufficient_for_trading', False),
                'live_prices_integrated': True
            }
            
            result = {
                "asof": asof,
                "source": provider.get_source_name(),
                "is_stale": False,
                "stale_seconds": 0,
                
                # Price information uses live prices
                "price": {
                    "value": regression_metrics.get('fair_value', base_price),
                    "type": "fair_value_with_live_prices",
                    "quote_ccy": quote,
                    "actual_price_a": base_price,
                    "actual_price_b": quote_price
                },
                
                # Core regression data with live price integration
                # FIX: Use _format_regression_data to flatten structure for vmain
                "regression": _format_regression_data(regression_metrics, "1m"),
                
                # Regime assessment data
                "regime_assessment": regime_assessment,
                
                # Volume data from candles
                "volume_data": {
                    'primary': current_base_volume,
                    'reference': current_quote_volume
                },
                
                # Enhanced data quality information
                "data_quality_summary": data_quality_summary,
                

                # [DEBUG TRACER] Trace the final RSI value before packaging
                #final_rsi_val = primary_analysis.get('rsi', 50.0)
                #print(f"[DEBUG FINAL PAYLOAD] RSI extraction: {final_rsi_val}")
                # Simplified indicators focused on regression
                "indicators": {
                    "zscore": {
                        "value": regression_metrics.get('z_score', 0.0),
                        "method": "regression_residual_live_prices",
                        "window": zscore_window
                    },
                    # FIX: Added placeholders for consistency with Buffer Mode
                    # [MODIFIED] Use dynamic value instead of placeholder
                    # "rsi": {"value": 50.0, "method": "placeholder"},
                    "rsi": {"value": primary_analysis.get('rsi', 50.0), "method": "calculated_buffer_ewm"},
                    "atr": float(primary_analysis.get('regression_metrics', {}).get('atr', 0.0)),
                    "regime": regression_metrics.get('trading_signals', {}).get('regime', 'RANGING'),
                    "signal": regression_metrics.get('trading_signals', {}).get('signal', 'neutral'),
                    "confidence": regression_metrics.get('trading_signals', {}).get('confidence', 0.0)
                },
                
                # Asset pair information
                "asset_pair_info": {
                    "base_asset": base,
                    "quote_asset": quote,
                    "asset_type": "pair",
                    "symbol": symbol
                }
            }
            
            # [DEBUG TRACER] Final check before handing data to vmain
            regime = result.get("regime_assessment", {})
            
            print(f"\n[DEBUG TELEMETRY EDGE]")
            if "hurst" in regime:
                print(f"✅ FINAL PAYLOAD contains Hurst: {regime['hurst']}")
            else:
                print(f"❌ FINAL PAYLOAD MISSING Hurst! Keys: {list(regime.keys())}")
                
            if "safe_for_mean_reversion" in regime:
                print(f"⚠️ FINAL PAYLOAD contains OLD KEY 'safe_for_mean_reversion'")
                
            return result
            
        except Exception as e:
            print(f"[CRITICAL] Analysis computation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"CRITICAL: Analysis failed for {base}/{quote}: {e}")
    
    else:
        print(f"[CRITICAL] Single asset mode not supported in corrected version")
        raise RuntimeError("CRITICAL: Only pair trading supported in corrected version")


# -------------------- Provider Factory --------------------

def get_telemetry_provider(config: Dict[str, Any]) -> TelemetryProvider:
    """
    Provider factory with NO fallbacks - fails fast on errors
    """
    
    if not HYPERLIQUID_AVAILABLE:
        raise RuntimeError("CRITICAL: Hyperliquid adapter not available - cannot proceed with live trading")
    
    trading_cfg = config.get("trading", {})
    
    # Check for Hyperliquid credentials
    hl_cfg = trading_cfg.get("hyperliquid", {})
    
    private_key = hl_cfg.get("private_key", "")
    wallet_address = hl_cfg.get("wallet_address", "")
    
    # Strict validation - no placeholders allowed
    if not private_key or len(private_key) < 60:
        raise RuntimeError(f"CRITICAL: Invalid private key - length: {len(private_key)}")
    
    if not wallet_address or len(wallet_address) < 40:
        raise RuntimeError(f"CRITICAL: Invalid wallet address - length: {len(wallet_address)}")
    
    if private_key.startswith('0x1234567890abcdef'):
        raise RuntimeError("CRITICAL: Placeholder private key detected - real credentials required")
    
    if wallet_address.startswith('0x1234567890123456'):
        raise RuntimeError("CRITICAL: Placeholder wallet address detected - real credentials required")
    
    try:
        provider = HyperliquidTelemetryProvider(private_key, wallet_address)
        return provider
        
    except Exception as e:
        print(f"[CRITICAL] Failed to create HyperliquidTelemetryProvider: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"CRITICAL: Cannot initialize live data provider: {e}")


def get_telemetry_from_buffers_triple_timeframe(symbol: str, symbol_info: Dict[str, Any], 
                                               data_buffers: 'MultiTimeframeBuffers', 
                                               config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Enhanced telemetry using pre-populated buffers - Triple-timeframe regression analysis
    
    Returns telemetry data with 5s, 1m, and 1h regression analysis.
    Schema: regression_5s + regression (1m) + regression_htf (1h)
    """
    
    cfg = config or {}
    buffer_config = cfg.get("multi_timeframe_buffers", {})
    
    # Get timeframe configuration
    tactical_timeframe = buffer_config.get("tactical_timeframe", "5s")
    primary_timeframe = buffer_config.get("primary_timeframe", "1m")
    bias_timeframe = buffer_config.get("bias_timeframe", "1h")
    
    timestamp_start = time.time()
    asof = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    
    try:
        current_base_price, current_quote_price = data_buffers.get_current_live_prices()
        # [NEW] Get BBO data alongside prices
        base_bbo, quote_bbo = data_buffers.get_current_live_bbos()
        # print(f"[DEBUG BBO PIPE] Base BBO: {base_bbo} | Quote BBO: {quote_bbo}")

        
        if current_base_price is None or current_quote_price is None:
            print(f"[BUFFER TELEMETRY] Failed to get current prices from buffers")
            return _create_triple_buffer_error_response(symbol, "Failed to get current prices", asof)
        
        # Analyze all three timeframes
        tactical_analysis = _analyze_timeframe_from_buffers(data_buffers, tactical_timeframe, cfg, current_base_price, current_quote_price)
        primary_analysis = _analyze_timeframe_from_buffers(data_buffers, primary_timeframe, cfg, current_base_price, current_quote_price)
        htf_analysis = _analyze_timeframe_from_buffers(data_buffers, bias_timeframe, cfg, current_base_price, current_quote_price)
        
        # Extract volume data from primary timeframe
        volume_data = {
            'primary': primary_analysis.get('volume_data', {}).get('base_volume', 1000.0),
            'reference': primary_analysis.get('volume_data', {}).get('quote_volume', 50000.0)
        }
        
        # Create telemetry response with triple-timeframe schema
        result = {
            "asof": asof,
            "source": "multi_timeframe_buffers:triple_regression",
            "is_stale": False,
            "stale_seconds": 0,
            
            # Price information
            "price": {
                "value": primary_analysis.get('regression_metrics', {}).get('fair_value', current_base_price),
                "type": "fair_value_primary_timeframe",
                "quote_ccy": symbol_info.get("quote", "MELANIA"),
                "actual_price_a": current_base_price,
                "actual_price_b": current_quote_price,
                # [NEW] Inject BBO Data here for Simulator usage
                "primary_data": base_bbo,
                "reference_data": quote_bbo
            },
            
            # Triple regression data
            "regression_5s": _format_regression_data(tactical_analysis.get('regression_metrics', {}), tactical_timeframe),
            "regression": _format_regression_data(primary_analysis.get('regression_metrics', {}), primary_timeframe),
            "regression_htf": _format_regression_data(htf_analysis.get('regression_metrics', {}), bias_timeframe),
            
            # Combined regime assessment with all timeframes
            "regime_assessment": _combine_triple_regime_assessment(
                tactical_analysis.get('regime_assessment', {}),
                primary_analysis.get('regime_assessment', {}),
                htf_analysis.get('regime_assessment', {}),
                tactical_timeframe, primary_timeframe, bias_timeframe
            ),
            
            # Volume data
            "volume_data": volume_data,
            
            # Data quality assessment for all timeframes
            "data_quality_summary": {
                'tactical_sufficient': tactical_analysis.get('regression_metrics', {}).get('data_quality', {}).get('sufficient_for_trading', False),
                'primary_sufficient': primary_analysis.get('regression_metrics', {}).get('data_quality', {}).get('sufficient_for_trading', False),
                'htf_sufficient': htf_analysis.get('regression_metrics', {}).get('data_quality', {}).get('sufficient_for_trading', False),
                'triple_timeframe_analysis': True,
                'tactical_periods': tactical_analysis.get('data_points_analyzed', 0),
                'primary_periods': primary_analysis.get('data_points_analyzed', 0),
                'htf_periods': htf_analysis.get('data_points_analyzed', 0)
            },
            
            # Technical indicators (primary timeframe as main)
            "indicators": {
                "zscore": {
                    "value": primary_analysis.get('regression_metrics', {}).get('z_score', 0.0),
                    "method": "regression_residual_triple_timeframe",
                    "window": primary_analysis.get('zscore_window', 40)
                },
                # [FIXED] Linking the actual calculated RSI from primary_analysis
                "rsi": {"value": primary_analysis.get('rsi', 50.0), "method": "calculated_buffer_ewm"},
                "atr": primary_analysis.get('regression_metrics', {}).get('atr', 0.0)
            },
            
            # Asset pair information
            "asset_pair_info": {
                "base_asset": symbol_info.get("base", "SOL"),
                "quote_asset": symbol_info.get("quote", "MELANIA"),
                "asset_type": "pair",
                "symbol": symbol
            }
        }
        
        # [DEBUG] Final Inspection of the Package
        regime = result.get("regime_assessment", {})
        print(f"\n[DEBUG FINAL] Inspecting Telemetry Package...")
        
        # Check if Hurst exists in the final combined dictionary
        if "hurst" in regime:
            print(f"[DEBUG FINAL] ✅ Hurst IS present: {regime['hurst']}")
        else:
            print(f"[DEBUG FINAL] ❌ Hurst is MISSING!")
            print(f"[DEBUG FINAL] Keys present: {list(regime.keys())}")
            
        return result
        
    except Exception as e:
        print(f"[BUFFER TELEMETRY ERROR] Triple-timeframe analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return _create_triple_buffer_error_response(symbol, str(e), asof)