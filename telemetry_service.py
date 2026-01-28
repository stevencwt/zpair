#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
telemetry_service.py - Telemetry Abstraction Layer
==================================================
Centralized telemetry service that abstracts buffer-based vs legacy data fetching,
providing a unified interface for market data acquisition and analysis.
"""

import time
import traceback
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from debug_utils import DebugLogger, PerformanceTimer, utcnow_iso
from symbol_utils import SymbolClassifier
from config_manager import ConfigurationManager


class TelemetryError(Exception):
    """Raised when telemetry operations fail"""
    pass


class TelemetryService:
    """
    Centralized telemetry service that handles both buffer-based and legacy data acquisition
    """
    
    def __init__(self, provider, config_manager: ConfigurationManager, logger: Optional[DebugLogger] = None):
        """
        Initialize telemetry service
        
        Args:
            provider: Telemetry provider instance
            config_manager: Configuration manager instance
            logger: Debug logger instance
        """
        self.provider = provider
        self.config_manager = config_manager
        self.logger = logger or DebugLogger(True, "TELEMETRY")
        
        # Service state
        self.buffer_mode_enabled = False
        self.data_buffers = None
        self.initialization_attempts = 0
        self.last_buffer_status_log = 0
        self.performance_metrics = {}
        
        # Import buffer system if available
        self._import_buffer_dependencies()
        
        # Initialize buffers if configured
        self._initialize_service()
    
    def _import_buffer_dependencies(self) -> None:
        """Import buffer system dependencies"""
        try:
            from multi_timeframe_buffers import MultiTimeframeBuffers
            from telemetry_provider import get_telemetry, get_telemetry_from_buffers_triple_timeframe
            
            self.MultiTimeframeBuffers = MultiTimeframeBuffers
            self.get_telemetry = get_telemetry
            self.get_telemetry_from_buffers_triple_timeframe = get_telemetry_from_buffers_triple_timeframe
            
            self.buffer_system_available = True
            self.logger.debug_print("Buffer system dependencies imported successfully")
            
        except ImportError as e:
            self.buffer_system_available = False
            self.logger.debug_print(f"Buffer system not available: {e}")
            
            # Import legacy telemetry
            try:
                from telemetry_provider import get_telemetry
                self.get_telemetry = get_telemetry
                self.logger.debug_print("Legacy telemetry imported successfully")
            except ImportError as e2:
                raise TelemetryError(f"No telemetry system available: {e2}")
    
    def _initialize_service(self) -> None:
        """Initialize telemetry service based on configuration"""
        self.logger.debug_print("Initializing telemetry service...")
        
        buffer_config = self.config_manager.get_buffer_config()
        
        if (self.buffer_system_available and 
            buffer_config.get("enabled", False)):
            
            self.logger.debug_print("Attempting to initialize buffer mode...")
            
            try:
                self.data_buffers = self._initialize_data_buffers()
                if self.data_buffers:
                    self.buffer_mode_enabled = True
                    self.logger.debug_print("Buffer mode enabled successfully")
                else:
                    self._handle_buffer_initialization_failure(buffer_config)
                    
            except Exception as e:
                self.logger.debug_print(f"Buffer initialization error: {e}")
                self._handle_buffer_initialization_failure(buffer_config)
        else:
            self.logger.debug_print("Buffer mode disabled or unavailable, using legacy telemetry")
    
    def _handle_buffer_initialization_failure(self, buffer_config: Dict[str, Any]) -> None:
        """Handle buffer initialization failure"""
        fallback_enabled = buffer_config.get("initialization", {}).get("fallback_to_legacy_mode", True)
        
        if fallback_enabled:
            self.logger.debug_print("Falling back to legacy telemetry mode")
            self.buffer_mode_enabled = False
        else:
            raise TelemetryError("Buffer initialization required but failed")
    
    def _initialize_data_buffers(self) -> Optional[object]:
        """Initialize triple-timeframe data buffers with historical data"""
        try:
            self.logger.debug_print("Starting triple-timeframe buffer initialization...")
            
            # Get asset information
            asset_info = self.config_manager.get_asset_info()
            base = asset_info["base"]
            quote = asset_info["quote"]
            
            if not base or not quote:
                raise TelemetryError(f"Invalid asset configuration: base='{base}', quote='{quote}'")
            
            self.logger.debug_print(f"Initializing buffers for {base}/{quote}")
            
            # Get buffer configuration
            buffer_config = self.config_manager.get_buffer_config()
            
            # Create buffer manager
            buffers = self.MultiTimeframeBuffers(base, quote, buffer_config, self.provider)
            
            # Initialize with historical data
            with PerformanceTimer("buffer_initialization", self.logger):
                success = buffers.initialize_historical_data()
            
            if success:
                self.logger.debug_print("Triple-timeframe buffers initialized successfully")
                self._log_buffer_initialization_status(buffers)
                return buffers
            else:
                self.logger.debug_print("Failed to populate buffers with historical data")
                return None
                
        except Exception as e:
            self.logger.debug_print(f"Buffer initialization failed: {e}")
            traceback.print_exc()
            return None
    
    def _log_buffer_initialization_status(self, buffers) -> None:
        """Log buffer initialization status"""
        try:
            status = buffers.get_status()
            self.logger.debug_print(f"Buffer status: {len(status['buffers'])} buffers created")
            
            for buffer_key, buffer_status in status['buffers'].items():
                if "5s" in buffer_key:
                    self.logger.debug_print(
                        f"  {buffer_key}: Real-time only (no historical), "
                        f"current_price={buffer_status['current_price']}"
                    )
                else:
                    self.logger.debug_print(
                        f"  {buffer_key}: {buffer_status['historical_candles']} candles, "
                        f"current_price={buffer_status['current_price']}"
                    )
        except Exception as e:
            self.logger.debug_print(f"Failed to log buffer status: {e}")
    
    def get_market_data(self, asset: str) -> Dict[str, Any]:
        """
        Get comprehensive market data using optimal method (buffers vs legacy)
        
        Args:
            asset: Asset symbol (e.g., "SOL/MELANIA")
            
        Returns:
            Telemetry data dictionary with market information
            
        Raises:
            TelemetryError: If data acquisition fails
        """
        self.logger.debug_print(f"Getting market data for {asset}")
        
        # Parse asset symbol
        symbol_info = SymbolClassifier.classify_symbol(asset)
        
        if not symbol_info["is_valid"]:
            raise TelemetryError(f"Invalid asset symbol: {asset}")
        
        try:
            if self.buffer_mode_enabled and self.data_buffers:
                return self._get_buffer_based_data(asset, symbol_info)
            else:
                return self._get_legacy_data(asset, symbol_info)
                
        except Exception as e:
            self.logger.debug_print(f"Market data acquisition failed: {e}")
            
            # If buffer mode fails, try fallback to legacy
            if self.buffer_mode_enabled:
                self.logger.debug_print("Attempting fallback to legacy telemetry")
                try:
                    return self._get_legacy_data(asset, symbol_info)
                except Exception as e2:
                    raise TelemetryError(f"Both buffer and legacy telemetry failed: {e}, {e2}")
            else:
                raise TelemetryError(f"Legacy telemetry failed: {e}")
    
    def _get_buffer_based_data(self, asset: str, symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get data using optimized buffer system"""
        self.logger.debug_print("Using optimized triple-timeframe buffer-based telemetry")
        
        with PerformanceTimer("buffer_price_update", self.logger):
            # Update current prices (only 2 API calls)
            price_update_success = self.data_buffers.update_current_prices()
        
        if not price_update_success:
            self.logger.debug_print("Failed to update current prices")
            raise TelemetryError("Buffer price update failed")
        
        # Get telemetry from buffers with triple-timeframe analysis (no additional API calls)
        with PerformanceTimer("buffer_telemetry_analysis", self.logger):
            config = self.config_manager.get_config()
            tele_wrapped = self.get_telemetry_from_buffers_triple_timeframe(
                asset, symbol_info, self.data_buffers, config
            )
        
        # Track performance metrics
        self._update_performance_metrics("buffer", 2)  # 2 API calls for price updates
        
        # Periodic buffer status logging
        self._maybe_log_buffer_status()
        
        return tele_wrapped
    
    def _get_legacy_data(self, asset: str, symbol_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get data using legacy full historical fetch"""
        self.logger.debug_print("Using legacy telemetry (full historical fetch)")
        
        with PerformanceTimer("legacy_telemetry", self.logger):
            config = self.config_manager.get_config()
            tele_wrapped = self.get_telemetry(asset, symbol_info, self.provider, config)
        
        # Track performance metrics
        self._update_performance_metrics("legacy", 900)  # Estimated 900 API calls for full fetch
        
        return tele_wrapped
    
    def _update_performance_metrics(self, mode: str, api_calls: int) -> None:
        """Update performance tracking metrics"""
        if mode not in self.performance_metrics:
            self.performance_metrics[mode] = {
                "total_api_calls": 0,
                "total_cycles": 0,
                "last_updated": time.time()
            }
        
        self.performance_metrics[mode]["total_api_calls"] += api_calls
        self.performance_metrics[mode]["total_cycles"] += 1
        self.performance_metrics[mode]["last_updated"] = time.time()
    
    def _maybe_log_buffer_status(self) -> None:
        """Log buffer status periodically"""
        current_time = time.time()
        if current_time - self.last_buffer_status_log > 300:  # Every 5 minutes
            self.log_buffer_status()
            self.last_buffer_status_log = current_time
    
    def update_realtime_data(self) -> bool:
        """
        Update real-time price data (buffer mode only)
        
        Returns:
            True if update successful, False otherwise
        """
        if not self.buffer_mode_enabled or not self.data_buffers:
            self.logger.debug_print("Real-time update not available in legacy mode")
            return False
        
        try:
            with PerformanceTimer("realtime_price_update", self.logger):
                success = self.data_buffers.update_current_prices()
            
            if success:
                self.logger.debug_print("Real-time data updated successfully")
            else:
                self.logger.debug_print("Real-time data update failed")
            
            return success
            
        except Exception as e:
            self.logger.debug_print(f"Real-time update error: {e}")
            return False
    
    def get_triple_timeframe_analysis(self, asset: str) -> Dict[str, Any]:
        """
        Get analysis across all timeframes (buffer mode only)
        
        Args:
            asset: Asset symbol
            
        Returns:
            Triple-timeframe analysis data
            
        Raises:
            TelemetryError: If not in buffer mode or analysis fails
        """
        if not self.buffer_mode_enabled:
            raise TelemetryError("Triple-timeframe analysis requires buffer mode")
        
        return self.get_market_data(asset)
    
    def get_chart_data(self, timeframe: Optional[str] = None) -> Tuple[Optional[object], Optional[object], str]:
        """
        Get chart data from buffers
        
        Args:
            timeframe: Specific timeframe to get data for (defaults to primary_timeframe)
            
        Returns:
            Tuple of (df1, df2, actual_timeframe) or (None, None, "") if unavailable
        """
        if not self.buffer_mode_enabled or not self.data_buffers:
            self.logger.debug_print("Chart data not available - buffer mode disabled")
            return None, None, ""
        
        try:
            # Get timeframe configuration
            buffer_config = self.config_manager.get_buffer_config()
            target_timeframe = timeframe or buffer_config.get("primary_timeframe", "1m")
            
            # Get available timeframes
            available_timeframes = list(buffer_config.get("timeframes", {}).keys())
            
            if target_timeframe not in available_timeframes:
                self.logger.debug_print(
                    f"Timeframe '{target_timeframe}' not available. "
                    f"Available: {available_timeframes}"
                )
                return None, None, ""
            
            # Get dataframes from buffers
            self.logger.debug_print(f"Retrieving {target_timeframe} chart data from buffers")
            
            df1, df2 = self.data_buffers.get_analysis_data(target_timeframe)
            
            if df1 is None or df2 is None or len(df1) == 0 or len(df2) == 0:
                self.logger.debug_print("No chart data available from buffers")
                return None, None, ""
            
            self.logger.debug_print(
                f"Chart data retrieved: {len(df1)} and {len(df2)} rows for {target_timeframe}"
            )
            
            return df1, df2, target_timeframe
            
        except Exception as e:
            self.logger.debug_print(f"Error retrieving chart data: {e}")
            return None, None, ""
    
    def log_buffer_status(self) -> None:
        """Log comprehensive buffer status"""
        if not self.data_buffers:
            self.logger.debug_print("No buffers available for status logging")
            return
        
        try:
            status = self.data_buffers.get_status()
            self.logger.debug_print("=== BUFFER STATUS ===")
            self.logger.debug_print(f"Initialized: {status['is_initialized']}")
            self.logger.debug_print(f"Last price fetch: {status['last_price_fetch']}")
            
            for buffer_key, buffer_status in status['buffers'].items():
                self.logger.debug_print(
                    f"{buffer_key}: {buffer_status['historical_candles']} candles, "
                    f"current=${buffer_status['current_price']:.4f}"
                )
            
            if self.data_buffers.should_refresh_historical_data():
                self.logger.debug_print("Historical data refresh recommended")
                
        except Exception as e:
            self.logger.debug_print(f"Buffer status logging failed: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status
        
        Returns:
            Service status dictionary
        """
        status = {
            "mode": "buffer" if self.buffer_mode_enabled else "legacy",
            "buffer_system_available": self.buffer_system_available,
            "buffer_mode_enabled": self.buffer_mode_enabled,
            "initialization_attempts": self.initialization_attempts,
            "performance_metrics": self.performance_metrics.copy(),
            "provider_type": type(self.provider).__name__ if self.provider else "None"
        }
        
        # Add buffer-specific status
        if self.buffer_mode_enabled and self.data_buffers:
            try:
                buffer_status = self.data_buffers.get_status()
                status["buffer_status"] = {
                    "is_initialized": buffer_status["is_initialized"],
                    "buffer_count": len(buffer_status["buffers"]),
                    "last_price_fetch": buffer_status["last_price_fetch"],
                    "needs_refresh": self.data_buffers.should_refresh_historical_data()
                }
            except Exception as e:
                status["buffer_status"] = {"error": str(e)}
        
        return status
    
    def get_efficiency_metrics(self, tick_count: int) -> Dict[str, Any]:
        """
        Calculate efficiency metrics for the current session
        
        Args:
            tick_count: Number of ticks processed
            
        Returns:
            Efficiency metrics dictionary
        """
        if not self.buffer_mode_enabled:
            legacy_calls = tick_count * 900
            return {
                "mode": "legacy",
                "total_api_calls": legacy_calls,
                "api_calls_per_tick": 900,
                "efficiency_percentage": 0.0,
                "api_calls_saved": 0
            }
        
        # Buffer mode metrics
        legacy_calls = tick_count * 900
        buffer_calls = 6 + (tick_count * 2)  # 6 initial + 2 per tick
        api_calls_saved = legacy_calls - buffer_calls
        efficiency_pct = (api_calls_saved / legacy_calls) * 100 if legacy_calls > 0 else 0
        
        return {
            "mode": "buffer",
            "total_api_calls": buffer_calls,
            "api_calls_per_tick": 2,
            "legacy_would_have_made": legacy_calls,
            "api_calls_saved": api_calls_saved,
            "efficiency_percentage": efficiency_pct,
            "initialization_overhead": 6
        }
    
    def create_fallback_telemetry(self) -> Dict[str, Any]:
        """
        Create minimal fallback telemetry data when all systems fail
        
        Returns:
            Minimal telemetry dictionary
        """
        self.logger.debug_print("Creating fallback telemetry data")
        
        return {
            "asof": utcnow_iso(),
            "price": {"value": 200.0, "actual_price_a": 200.0, "actual_price_b": 0.17},
            "regression_5s": {
                "zscore": 0.0, "spread": 0.0, "alpha": 0.0, "beta": 1.0, 
                "r_squared": 0.0, "correlation": 0.0, "window_size": 0, 
                "health_status": "unknown", "beta_drift_pct": 0.0, "timeframe": "5s"
            },
            "regression": {
                "zscore": 0.0, "spread": 0.0, "alpha": 0.0, "beta": 1.0, 
                "r_squared": 0.0, "correlation": 0.0, "window_size": 0, 
                "health_status": "unknown", "beta_drift_pct": 0.0, "timeframe": "1m"
            },
            "regression_htf": {
                "zscore": 0.0, "spread": 0.0, "alpha": 0.0, "beta": 1.0, 
                "r_squared": 0.0, "correlation": 0.0, "window_size": 0, 
                "health_status": "unknown", "beta_drift_pct": 0.0, "timeframe": "1h"
            },
            "regime_assessment": {
                "safe_for_mean_reversion": True, "timeframe": "5s", 
                "tactical_context": {}, "htf_context": {}
            },
            "indicators": {"rsi": {"value": 50.0}, "atr": 0.0},
            "volume_data": {"primary": 1000.0, "reference": 50000.0},
            "asset_pair_info": {"base_asset": "SOL", "quote_asset": "MELANIA", "asset_type": "pair"},
            "data_quality_summary": {
                "tactical_sufficient": False, 
                "primary_sufficient": False, 
                "htf_sufficient": False
            },
            "_fallback": True,
            "_error": "All telemetry systems failed"
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown telemetry service"""
        self.logger.debug_print("Shutting down telemetry service...")
        
        # Log final performance metrics
        if self.performance_metrics:
            self.logger.debug_print("=== FINAL TELEMETRY PERFORMANCE ===")
            for mode, metrics in self.performance_metrics.items():
                total_calls = metrics["total_api_calls"]
                total_cycles = metrics["total_cycles"]
                avg_calls = total_calls / total_cycles if total_cycles > 0 else 0
                
                self.logger.debug_print(
                    f"{mode.upper()}: {total_calls} API calls over {total_cycles} cycles "
                    f"(avg {avg_calls:.1f} calls/cycle)"
                )
        
        # Clean up buffers if available
        if self.data_buffers:
            try:
                # Buffer system doesn't have explicit cleanup, but log final status
                self.log_buffer_status()
            except Exception as e:
                self.logger.debug_print(f"Error during buffer cleanup: {e}")
        
        self.logger.debug_print("Telemetry service shutdown complete")


# Factory function for backward compatibility
def create_telemetry_service(provider, config_manager: ConfigurationManager, 
                           logger: Optional[DebugLogger] = None) -> TelemetryService:
    """
    Factory function to create telemetry service instance
    
    Args:
        provider: Telemetry provider instance
        config_manager: Configuration manager instance
        logger: Debug logger instance
        
    Returns:
        Configured TelemetryService instance
    """
    return TelemetryService(provider, config_manager, logger)
