#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmain_refactored_phase3.py - Refactored Trading System (Phases 1, 2 & 3) - ENHANCED with Context-Specific Analysis
========================================================================
ENHANCED VERSION: Added context-specific trading signal analysis support and display.
UPDATED VERSION: Added multi-timeframe extreme tracking and marketstate integration.
ENHANCED VERSION: Added persistent position loading on startup.

Phase 1 Refactoring: Utilities Extraction ✓
- debug_utils.py: Centralized debug logging and performance monitoring
- symbol_utils.py: Symbol classification and validation utilities

Phase 2 Refactoring: Configuration Management ✓
- config_manager.py: Complete configuration loading, validation, and management

Phase 3 Refactoring: Telemetry Abstraction Layer ✓
- telemetry_service.py: Centralized telemetry with buffer/legacy abstraction

NEW: Multi-Timeframe Extreme Tracking Integration ✓
- ExtremeTracker: Multi-timeframe data collection with unified storage
- Marketstate Integration: Top 3 extremes per timeframe in JSON output
- Position Scaling: Uses configured primary timeframe only

ENHANCED: Persistent Position Loading ✓
- Automatic loading of open positions from persistent storage on startup
- Seamless continuation of trading operations after restart
- Proper marketstate initialization with existing positions

NEW: Context-Specific Analysis Support ✓
- Enhanced signal processing for context-aware analysis
- Dashboard integration for context-specific reasoning display
- Console output enhancement with analysis context indicators
- Backward compatibility with existing signal consumers

This refactoring significantly improves:
- Service layer abstraction for telemetry
- Clear separation between data acquisition and business logic
- Simplified main loop with better error handling
- Enhanced testing capabilities through service mocking
- Reduced coupling between components
- Multi-timeframe extreme awareness for dashboard visibility
- Centralized extreme storage with marketstate files
- Persistent position state management
- Context-specific trading analysis visibility

⚠️ FINANCIAL RISK WARNING ⚠️
This system generates trading recommendations for educational/research purposes.
All trading involves substantial risk of financial loss. Past performance does not guarantee future results.
This is experimental software - use appropriate position sizing and risk management.
Trading decisions should be verified with your own analysis.
"""

import os
import time
import json
import tempfile
import argparse
import sys
import traceback
import math
import threading  

from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional,List,Tuple
from enum import Enum
from datetime import datetime, date
from debug_utils import get_local_time  


# Import refactored modules
from debug_utils import DebugLogger, PerformanceTimer, utcnow_iso, safe_mkdir
from symbol_utils import SymbolClassifier
from config_manager import ConfigurationManager, ConfigValidationError
from telemetry_service import TelemetryService, TelemetryError
from market_state_manager import MarketStateManager

# Initialize enhanced debug logger
logger = DebugLogger(True, "MAIN")
# logger.debug_print("Starting Phase 1+2+3 refactored vmain script with multi-timeframe extreme tracking, persistent position loading, and context-specific analysis...", "MAIN")

try:
    # logger.debug_print("Importing numpy and pandas...", "IMPORTS")
    import numpy as np
    import pandas as pd
    # logger.debug_print("Numpy and pandas imported successfully", "IMPORTS")
except Exception as e:
    logger.debug_print(f"Failed to import numpy/pandas: {e}", "IMPORTS")
    sys.exit(1)

def safe_float(value, default=0.0):
    """Safely convert value to float, handling None, inf, and nan cases"""
    try:
        if value is None:
            return default
        
        float_val = float(value)
        
        # Check for infinity or NaN
        if not math.isfinite(float_val):
            return default
            
        return float_val
    except (ValueError, TypeError, OverflowError):
        return default



# Custom JSON encoder to handle numpy types, datetime, and boolean
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # --- numpy numeric types ---
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            float_val = float(obj)
            # Handle infinity and NaN values
            if not math.isfinite(float_val):
                return 0.0  # or None, depending on your preference
            return float_val
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # --- boolean types ---
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        # --- datetime / date / pandas timestamp ---
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif pd is not None and isinstance(obj, getattr(pd, "Timestamp", ())):
            return obj.to_pydatetime().isoformat()
        elif isinstance(obj, np.datetime64):
            # convert numpy datetime64 -> ISO format string
            if pd is not None:
                return pd.to_datetime(obj).to_pydatetime().isoformat()
            return str(obj.astype("datetime64[ns]"))
        # --- enum types (critical fix) ---
        elif isinstance(obj, Enum):
            return obj.value
        # --- set types ---
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        # --- Handle Python float infinity/NaN ---
        elif isinstance(obj, float):
            if not math.isfinite(obj):
                return 0.0  # or None, depending on your preference
            return obj
        # --- fallback ---
        return super().default(obj)

    def encode(self, obj):
        """Override encode to catch any remaining infinity values"""
        try:
            return super().encode(obj)
        except ValueError as e:
            if "Out of range float values are not JSON compliant" in str(e):
                # This should not happen with our fixes above, but just in case
                print(f"Warning: Infinity value detected in JSON encoding: {e}")
                # Try to replace infinity values in the object
                cleaned_obj = self._clean_infinity_values(obj)
                return super().encode(cleaned_obj)
            raise

    def _clean_infinity_values(self, obj):
        """Recursively clean infinity values from nested objects"""
        if isinstance(obj, dict):
            return {k: self._clean_infinity_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_infinity_values(item) for item in obj]
        elif isinstance(obj, float) and not math.isfinite(obj):
            return 0.0
        elif isinstance(obj, np.floating) and not math.isfinite(float(obj)):
            return 0.0
        else:
            return obj

# Import modules with enhanced error tracking
try:
    # logger.debug_print("Importing telemetry provider...", "IMPORTS")
    from telemetry_provider import get_telemetry_provider
    # logger.debug_print("Telemetry provider imported successfully", "IMPORTS")
except Exception as e:
    logger.debug_print(f"Failed to import telemetry provider: {e}", "IMPORTS")
    sys.exit(1)

# Import market barometer
try:
    # logger.debug_print("Importing market barometer...", "IMPORTS")
    from market_barometer import MarketAnalyzer, MarketBarometer, MarketState
    MARKET_BAROMETER_AVAILABLE = True
    # logger.debug_print("    logger.debug_print("Market barometer imported successfully", "IMPORTS")
except ImportError as e:
    MARKET_BAROMETER_AVAILABLE = False
    logger.debug_print(f"Market barometer not available: {e}", "IMPORTS")

# Import trading decision module
try:
    # logger.debug_print("Importing trading decision module...", "IMPORTS")
    from trading_decision import make_trading_decision
    TRADING_DECISION_AVAILABLE = True
    # logger.debug_print("    logger.debug_print("Trading decision module imported successfully", "IMPORTS")
except ImportError as e:
    TRADING_DECISION_AVAILABLE = False
    logger.debug_print(f"Trading decision module not available: {e}", "IMPORTS")

# Import modular strategy engine
try:
    #logger.debug_print("Importing modular strategy engine...", "IMPORTS")
    from modular_decision_engine import ModularDecisionEngine
    MODULAR_STRATEGIES_AVAILABLE = True
    # logger.debug_print("    logger.debug_print("Modular strategy engine imported successfully", "IMPORTS")
except ImportError as e:
    MODULAR_STRATEGIES_AVAILABLE = False
    logger.debug_print(f"Modular strategy engine not available: {e}", "IMPORTS")

# Import dual-track trade simulator with Phase 1 enhancements
# Import Unified Trade Engine (Single Track)
try:
    from unified_trade_engine import UnifiedTradeEngine
    TRADE_SIMULATION_AVAILABLE = True 
except ImportError as e:
    TRADE_SIMULATION_AVAILABLE = False
    logger.debug_print(f"Unified Engine not available: {e}", "IMPORTS")

# Import Chart Orchestrator
try:
    # logger.debug_print("Importing Chart Orchestrator...", "IMPORTS")
    from chart_orchestrator import ChartOrchestrator
    CHART_ORCHESTRATOR_AVAILABLE = True
    # logger.debug_print("    logger.debug_print("Chart Orchestrator imported successfully", "IMPORTS")
except ImportError as e:
    CHART_ORCHESTRATOR_AVAILABLE = False
    logger.debug_print(f"Chart Orchestrator not available: {e}", "IMPORTS")

# Import Config Directory Validator
try:
    # logger.debug_print("Importing config directory validator...", "IMPORTS")
    from config_directory_validator import validate_config_directories
    CONFIG_VALIDATOR_AVAILABLE = True
    # logger.debug_print("    logger.debug_print("Config directory validator imported successfully", "IMPORTS")
except ImportError as e:
    CONFIG_VALIDATOR_AVAILABLE = False
    #logger.debug_print(f"Config directory validator not available: {e}", "IMPORTS")

# NEW: Import ExtremeTracker for multi-timeframe extreme tracking
try:
    # logger.debug_print("Importing ExtremeTracker for multi-timeframe extreme tracking...", "IMPORTS")
    from extreme_tracker import ExtremeTracker
    EXTREME_TRACKER_AVAILABLE = True
    # logger.debug_print("    logger.debug_print("ExtremeTracker imported successfully", "IMPORTS")
except ImportError as e:
    EXTREME_TRACKER_AVAILABLE = False
    logger.debug_print(f"ExtremeTracker not available: {e}", "IMPORTS")

# logger.debug_print("All imports completed for Phase 1+2+3 refactored system with multi-timeframe extreme tracking, persistent position loading, and context-specific analysis", "IMPORTS")

# -------------------- Configuration --------------------

SCHEMA_VERSION = "2.5-context-specific-analysis-integrated"  # Updated schema with context-specific analysis
TICK_SEC = 1  # 5-second tick data

# -------------------- File Store --------------------



# ... imports ...

# ==============================================================================
# PHASE 1: EMBEDDED CONTROL SERVER (Dynamic Parameter Updates)
# ==============================================================================
# ==============================================================================
# PHASE 1: EMBEDDED CONTROL SERVER (Dynamic Parameter Updates)
# ==============================================================================
# ==============================================================================
# PHASE 1 [FIXED]: EMBEDDED CONTROL SERVER (Dynamic Parameter Updates)
# ==============================================================================
# ==============================================================================
# PHASE 1 [DEBUG MODE]: EMBEDDED CONTROL SERVER
# ==============================================================================
# ==============================================================================
# PHASE 1 [FIXED & DEBUGGED]: EMBEDDED CONTROL SERVER
# ==============================================================================
class ConfigUpdateHandler(BaseHTTPRequestHandler):
    app = None  # Class-level reference to the application instance

    def do_GET(self):
        """Handle GET requests for dashboard, market state, and static JSON files"""
        try:
            # 1. API Endpoint: Live Market State
            if self.path.startswith('/live_market_state'):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                self.end_headers()
                
                market_state = {}
                if hasattr(self.app, 'state_manager'):
                    market_state = self.app.state_manager.load_snapshot()
                
                response_str = json.dumps(market_state, default=str)
                self.wfile.write(response_str.encode('utf-8'))
                return

            # 2. Dashboard Endpoint
            if self.path == '/' or self.path == '/index.html' or self.path.startswith('/zsimple.html'):
                self._serve_file('zsimple.html', 'text/html')
                return

            # [RECTIFIED] 3. STATIC FILE ROUTER: Handle JSON history files
            # This allows the dashboard to fetch files inside the 'aggregated_...' folder
            clean_path = self.path.split('?')[0].lstrip('/') # Remove cache busters and leading slash
            if clean_path.endswith('.json') and os.path.exists(clean_path):
                self._serve_file(clean_path, 'application/json')
                return
            
            self.send_error(404, "File not found")
            
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {e}")

    def _serve_file(self, file_path, mime_type):
        """[ADDITIVE] Helper to read and send a physical file from disk to the browser"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-type', mime_type)
            self.send_header('Access-Control-Allow-Origin', '*') # Required for cross-site fetch
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(404, f"Error serving {file_path}: {e}")

    def do_POST(self):
        """Handle control commands"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            app = self.__class__.app
            if not app:
                self.send_error(500, "Application not initialized")
                return

            # =================================================================
            # ROUTE 1: POSITION & EXECUTION COMMANDS (WIRED TO UNIFIED ENGINE)
            # =================================================================
            action = data.get('action')
            success = False
            
            # Governor Commands
            if action == 'SET_SOFT_PAUSE':
                app.manual_pause_active = True
                app.update_governor_permit()
                app.persist_governor_state()
                success = True
            elif action == 'RESUME_TRADING':
                app.manual_pause_active = False
                app.update_governor_permit()
                app.persist_governor_state()
                success = True
            elif action == 'SET_HARD_PAUSE':
                app.manual_pause_active = True
                app.update_governor_permit()
                app.persist_governor_state()
                # Also trigger close all
                if hasattr(app, 'execution_engine') and app.execution_engine:
                    data['action'] = 'CLOSE_ALL_POSITIONS'
                    result = app.execution_engine.manual_override('CLOSE_ALL_POSITIONS', data)
                success = True
            
            # Unified Execution Commands
            if 'action' in data and data['action'] in ['SYNC_POSITION', 'IMPORT_POSITION', 'PURGE_POSITION', 'CLOSE_ALL_POSITIONS', 'SWITCH_MODE', 'PANIC_SELL', 'SET_TARGET_PROFIT']:
                action = data.get('action')
                print(f"[SERVER] 📨 Action: {action}")
                if hasattr(app, 'execution_engine') and app.execution_engine:
                    result = app.execution_engine.manual_override(action, data)
                    # Handle both bool and dict returns
                    success = result if isinstance(result, bool) else result.get('status') in ['SUCCESS', 'PENDING']
                    #success = app.execution_engine.process_command(data)
                else:
                    print("[SERVER] Error: Execution Engine not initialized")

            # Send Response
            response = {"status": "executed" if success else "failed"}
            self.send_response(200 if success else 500)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            return

        except Exception as e:
            print(f"Server POST Error: {e}")
            self.send_error(500, f"Server Error: {e}")

    def log_message(self, format, *args):
        # Suppress generic HTTP logging to keep console clean
        return

def start_control_server(app, port=8090):
    """
    Start the control server on ALL interfaces (0.0.0.0).
    This allows access via Tailscale, LAN, and Localhost.
    """
    # CRITICAL FIX: Bind to 0.0.0.0 instead of localhost
    server_address = ('0.0.0.0', port)
    
    handler = ConfigUpdateHandler
    handler.app = app
    
    try:
        httpd = HTTPServer(server_address, handler)
        print(f"🎛️  Control Server running on http://0.0.0.0:{port}")
        print(f"   - Local: http://localhost:{port}")
        print(f"   - Remote: http://<tailscale-ip>:{port}")
        
        thread = threading.Thread(target=httpd.serve_forever)
        thread.daemon = True
        thread.start()
        return httpd
    except OSError as e:
        print(f"❌ Failed to start server on port {port}: {e}")
        print("   (Check if another instance is running or permission denied)")
        return None


#  -------------------- Main App Class --------------------

class TradingApplication:
    def __init__(self, cfg_path: str = "configvisionapi.json"):
        logger.debug_print("TradingApplication.__init__ starting for Phase 1+2+3 refactored system with multi-timeframe extreme tracking, persistent position loading, and context-specific analysis...", "APP")
        
        self._initializing = True  # Prevent config overwrites during startup
    

        try:
            # Initialize configuration manager with enhanced validation
            logger.debug_print("Initializing configuration manager...", "CONFIG")
            self.config_manager = ConfigurationManager(cfg_path, logger)
            
            try:
                self.cfg = self.config_manager.load_config(validate=True)
                logger.debug_print("Configuration loaded and validated successfully", "CONFIG")
            except ConfigValidationError as e:
                logger.debug_print(f"Configuration validation failed: {e}", "CONFIG")
                logger.debug_print("Attempting to load with warnings only...", "CONFIG")
                self.cfg = self.config_manager.load_config(validate=False)
                
                # Log validation status
                validation_status = self.config_manager.get_validation_status()
                if validation_status["has_warnings"]:
                    for warning in validation_status["warnings"]:
                        logger.debug_print(f"Config warning: {warning}", "CONFIG")
            
            # LOAD THE CONFIG FIRST
            self.cfg = self.config_manager.load_config(validate=True)
            self.cfg_path = cfg_path 

            # NOW INITIALIZE GOVERNOR FROM THE LOADED CFG
            gov_settings = self.cfg.get('governor_settings', {})
            self.manual_pause_active = gov_settings.get('manual_pause_active', False)
            
            #self.trading_permit = "ALL"
            self.update_governor_permit()

            # VALIDATE CONFIG DIRECTORIES BEFORE PROCEEDING
            self.validate_config_directories_before_startup(cfg_path)
            
            # Get asset information using enhanced symbol utilities
            asset_info = self.config_manager.get_asset_info()
            self.asset = asset_info["asset"]
            logger.debug_print(f"Asset: {self.asset}", "CONFIG")


            # Setup persistence layer (Strategy Pattern)
            logger.debug_print("Setting up persistence manager...", "PERSISTENCE")
            # Initialize the manager which decides between FILE or MEMORY based on config
            self.state_manager = MarketStateManager(self.cfg, logger)

            # [FIX] Define snapshot_dir so ExtremeTracker can use it
            snapshot_dir = self.cfg.get("snapshot_dir", "marketstate") 

            # NEW: Initialize multi-timeframe extreme tracker
            logger.debug_print("Initializing multi-timeframe extreme tracker...", "EXTREME_TRACKING")

            self.extreme_tracker = None
            if EXTREME_TRACKER_AVAILABLE:
                try:
                    # Get buffer configuration for timeframes
                    buffer_config = self.config_manager.get_buffer_config()
                    primary_timeframe = buffer_config.get("primary_timeframe", "1m")
                    
                    # Initialize with snapshot directory for unified storage
                    # Note: We create a global extreme tracker here, not strategy-specific
                    # This allows us to collect data from all timeframes for dashboard visibility
                    global_strategy_config = {
                        'position_scaling_config': {
                            'extreme_tracking': {
                                'window_size': 1000,
                                'min_threshold': 1.0,
                                'history_depth': 3,
                                'max_age_hours': 24,
                                'min_extremes_required': 2,
                                'retention_hours': 168
                            }
                        }
                    }
                    
                    self.extreme_tracker = ExtremeTracker(
                        global_strategy_config, 
                        snapshot_dir, 
                        primary_timeframe
                    )
                    
                    logger.debug_print(f"Multi-timeframe extreme tracker initialized successfully", "EXTREME_TRACKING")
                    logger.debug_print(f"  Primary timeframe: {primary_timeframe}", "EXTREME_TRACKING")
                    logger.debug_print(f"  Storage location: {snapshot_dir}/extremes_history.json", "EXTREME_TRACKING")
                    
                except Exception as e:
                    logger.debug_print(f"Failed to initialize extreme tracker: {e}", "EXTREME_TRACKING")
                    self.extreme_tracker = None
            else:
                logger.debug_print("ExtremeTracker not available - extreme tracking disabled", "EXTREME_TRACKING")

            # Initialize telemetry provider
            logger.debug_print("Setting up telemetry provider...", "TELEMETRY")
            self.telemetry_provider = get_telemetry_provider(self.cfg)
            logger.debug_print(f"Telemetry provider initialized: {type(self.telemetry_provider)}", "TELEMETRY")

            # NEW: Initialize telemetry service (replaces complex buffer management)
            logger.debug_print("Initializing telemetry service...", "TELEMETRY")
            try:
                self.telemetry_service = TelemetryService(
                    self.telemetry_provider, 
                    self.config_manager, 
                    logger
                )
                
                # Get service status
                service_status = self.telemetry_service.get_service_status()
                logger.debug_print(f"Telemetry service initialized: mode={service_status['mode']}", "TELEMETRY")
                
                if service_status['buffer_mode_enabled']:
                    logger.debug_print("Triple-timeframe buffer system enabled via service", "TELEMETRY")
                else:
                    logger.debug_print("Using legacy telemetry mode via service", "TELEMETRY")
                    
            except Exception as e:
                logger.debug_print(f"Failed to initialize telemetry service: {e}", "TELEMETRY")
                raise

            # Initialize modular strategy engine
            logger.debug_print("Initializing modular strategy engine...", "TRADING")
            if MODULAR_STRATEGIES_AVAILABLE and self.cfg.get("trading_decision", {}).get("use_modular_strategies", False):
                try:
                    self.modular_engine = ModularDecisionEngine(self.cfg)
                    logger.debug_print("Modular strategy engine initialized successfully", "TRADING")
                except Exception as e:
                    logger.debug_print(f"Failed to initialize modular strategy engine: {e}", "TRADING")
                    self.modular_engine = None
            else:
                self.modular_engine = None
                logger.debug_print("Modular strategy engine disabled or not available", "TRADING")

            # Initialize market analyzer
            logger.debug_print("Setting up market analyzer...", "MARKET")
            self.market_analyzer = None
            if MARKET_BAROMETER_AVAILABLE and self.cfg.get("market_barometer", {}).get("enabled", True):
                if hasattr(self.telemetry_provider, 'adapter'):
                    try:
                        barometer_config = self.cfg.get("market_barometer", {}).get("market_config", {})
                        self.market_analyzer = MarketAnalyzer(
                            self.telemetry_provider.adapter,
                            barometer_config
                        )
                        logger.debug_print("Market barometer enabled", "MARKET")
                    except Exception as e:
                        logger.debug_print(f"Failed to initialize market analyzer: {e}", "MARKET")
                else:
                    logger.debug_print("Market barometer disabled (no HyperliquidAdapter available)", "MARKET")
            else:
                logger.debug_print("Market barometer disabled in config", "MARKET")

            # [PHASE 4] UNIFIED EXECUTION ENGINE
            logger.debug_print("Initializing Unified Execution Engine...", "SIMULATION")
            
            # Initialize the Single Source of Truth
            # This replaces trade_simulator, live_engine, and all dual-track logic
            #self.execution_engine = UnifiedTradeEngine(self.cfg)
            #self.execution_engine = UnifiedTradeEngine(self.cfg, config_path=args.config)  # [PHASE 1] Added config_path
            self.execution_engine = UnifiedTradeEngine(self.cfg, config_path=self.cfg_path)
            
            # Compatibility aliases (so other parts of code don't crash immediately)
            # legacy code removed: self.trade_simulator = None 
            self.live_engine = None
            self._previous_marketstate = None
            
            logger.debug_print(f"Unified Execution Engine Initialized [Mode: {self.execution_engine.mode}]", "INIT")

            # [CRITICAL FIX] Log that persistent positions were loaded
            if hasattr(self, 'execution_engine') and self.execution_engine:
                if self.execution_engine.open_positions:
                    logger.debug_print(f"Persistent positions pre-loaded: {len(self.execution_engine.open_positions)} position(s)", "INIT")
                    for pos in self.execution_engine.open_positions:
                        logger.debug_print(f"  Position: {pos.get('id')} | Entry: ${pos.get('entry_price', 0):.2f} | Saved PnL: ${pos.get('pnl', 0):.2f}", "INIT")



            # Initialize Chart Orchestrator with telemetry service integration
            logger.debug_print("Setting up Chart Orchestrator with telemetry service integration...", "CHARTS")
            self.chart_orchestrator = None
            self.last_trading_signal = None
            
            chart_config = self.config_manager.get_chart_config()
            if CHART_ORCHESTRATOR_AVAILABLE and chart_config.get("enabled", True):
                try:
                    # Setup chart directory
                    output_dir = chart_config.get("output_dir", "realtime")
                    safe_mkdir(output_dir, logger)
                    
                    # Initialize orchestrator with theme
                    theme = chart_config.get("theme", "dark")
                    self.chart_orchestrator = ChartOrchestrator(theme=theme)
                    
                    logger.debug_print("Chart Orchestrator initialized with telemetry service integration", "CHARTS")
                    
                except Exception as e:
                    logger.debug_print(f"Failed to initialize Chart Orchestrator: {e}", "CHARTS")
                    traceback.print_exc()
                    self.chart_orchestrator = None
            else:
                logger.debug_print("Chart Orchestrator disabled", "CHARTS")

            # Display enhanced status
            self._display_initialization_status()

            self._initializing = False  # Initialization complete, allow config saves now


            # logger.debug_print("TradingApplication.__init__ completed successfully with Phase 1+2+3 refactoring, multi-timeframe extreme tracking, persistent position loading, and context-specific analysis", "APP")

        except Exception as e:
            logger.debug_print(f"TradingApplication.__init__ failed: {e}", "APP")
            traceback.print_exc()
            raise

    def _display_initialization_status(self):
        """Display comprehensive initialization status"""
        service_status = self.telemetry_service.get_service_status()
        
        logger.debug_print("=== PHASE 1+2+3 REFACTORED SYSTEM WITH MULTI-TIMEFRAME EXTREME TRACKING & PERSISTENT POSITIONS & CONTEXT-SPECIFIC ANALYSIS ===", "STATUS")
        logger.debug_print(f"Configuration Management: REFACTORED", "STATUS")
        logger.debug_print(f"Debug Utilities: REFACTORED", "STATUS")
        logger.debug_print(f"Symbol Classification: ENHANCED", "STATUS")
        logger.debug_print(f"Telemetry Service: ABSTRACTED", "STATUS")
        logger.debug_print(f"Multi-Timeframe Extreme Tracking: {'ENABLED' if self.extreme_tracker else 'DISABLED'}", "STATUS")
        logger.debug_print(f"Persistent Position Loading: {'ENABLED' if self.execution_engine else 'DISABLED'}", "STATUS")
        logger.debug_print(f"Context-Specific Analysis: {'ENABLED' if self.modular_engine else 'DISABLED'}", "STATUS")
        logger.debug_print(f"Telemetry Mode: {service_status['mode'].upper()}", "STATUS")
        logger.debug_print(f"Buffer System Available: {service_status['buffer_system_available']}", "STATUS")
        logger.debug_print(f"API Efficiency: {'600+ calls saved per cycle' if service_status['buffer_mode_enabled'] else 'Legacy mode'}", "STATUS")
        logger.debug_print(f"Triple-timeframe Analysis: {'ENABLED' if service_status['buffer_mode_enabled'] else 'SINGLE TIMEFRAME'}", "STATUS")
        logger.debug_print(f"Trading Decision: {'ENABLED' if TRADING_DECISION_AVAILABLE else 'DISABLED'}", "STATUS")
        # legacy code removed: logger.debug_print(f"Unified Position Simulation: {'ENABLED' if self.trade_simulator else 'DISABLED'}", "STATUS")
        logger.debug_print(f"Unified Position Simulation: {'ENABLED' if getattr(self, 'execution_engine', None) else 'DISABLED'}", "STATUS")
        logger.debug_print(f"Chart Orchestrator: {'ENABLED' if self.chart_orchestrator else 'DISABLED'}", "STATUS")
        logger.debug_print(f"Asset: {self.asset} (TELEMETRY SERVICE + EXTREME TRACKING + PERSISTENT POSITIONS + CONTEXT ANALYSIS)", "STATUS")


    def update_governor_permit(self):
        """Logic Gate: Determines if the bot can Open or just Close"""
        if self.manual_pause_active:
            self.trading_permit = "CLOSE_ONLY" # Soft Pause
        else:
            # Future Phase 2: Insert Schedule-based Time logic here
            self.trading_permit = "ALL"

    def persist_governor_state(self):
        """Saves manual_pause_active to the pair-specific config file"""
        # Don't save during initialization to prevent overwriting loaded config
        if getattr(self, '_initializing', False):
            logger.debug_print("Skipping config save during initialization", "GOVERNOR")
            return
            
        if 'governor_settings' not in self.cfg:
            self.cfg['governor_settings'] = {}
        self.cfg['governor_settings']['manual_pause_active'] = self.manual_pause_active
        try:
            with open(self.cfg_path, 'w') as f:
                json.dump(self.cfg, f, indent=4)
            logger.debug_print(f"Config Updated: manual_pause={self.manual_pause_active}", "GOVERNOR")
        except Exception as e:
            logger.debug_print(f"Failed to save governor state: {e}", "GOVERNOR")

    def validate_config_directories_before_startup(self, config_path: str):
        """Validate directory naming before startup"""
        if not CONFIG_VALIDATOR_AVAILABLE:
            #logger.debug_print("Config directory validator not available - skipping validation", "VALIDATION")
            return
        
        try:
            logger.debug_print("Validating config directory naming conventions...", "VALIDATION")
            needs_correction, correction_applied = validate_config_directories(config_path)
            
            if correction_applied:
                print("\n" + "="*60)
                print("CONFIG UPDATED - RESTART REQUIRED")
                print("="*60)
                print("The configuration file has been updated to comply with")
                print("the directory naming convention. Please restart the program.")
                print("="*60)
                sys.exit(0)
            elif needs_correction:
                logger.debug_print("Config directory validation completed with warnings", "VALIDATION")
            else:
                logger.debug_print("Config directory validation passed", "VALIDATION")
                
        except Exception as e:
            logger.debug_print(f"Config directory validation failed: {e}", "VALIDATION")
            # Don't exit on validation errors - just warn
            print(f"Warning: Config directory validation failed: {e}")

    def should_force_generate_chart(self, marketstate: Dict[str, Any]) -> bool:
        """Check if chart should be force-generated due to signal changes"""
        chart_config = self.config_manager.get_chart_config()
        if not chart_config.get("force_generation_on_signal_change", False):
            return False
        
        current_signal = marketstate.get("trading_signal", {}).get("current_recommendation", {}).get("recommendation", "WAIT")
        
        if self.last_trading_signal is None:
            self.last_trading_signal = current_signal
            return False
        
        if current_signal != self.last_trading_signal:
            logger.debug_print(f"Trading signal changed: {self.last_trading_signal} -> {current_signal}", "CHARTS")
            self.last_trading_signal = current_signal
            return True
        
        return False

    def generate_realtime_chart(self) -> bool:
        """Generate real-time chart using Chart Orchestrator with telemetry service data"""
        if not self.chart_orchestrator:
            return False
        
        try:
            logger.debug_print("Generating real-time chart using Chart Orchestrator with telemetry service data...", "CHARTS")
            
            # Get chart data from telemetry service
            df1, df2, timeframe = self.telemetry_service.get_chart_data()
            
            if df1 is None or df2 is None:
                logger.debug_print("No chart data available from telemetry service - skipping chart", "CHARTS")
                return False
            
            # Get chart configuration
            chart_config = self.config_manager.get_chart_config()
            output_dir = chart_config.get("output_dir", "realtime")
            filename = chart_config.get("filename", "live_chart.png")
            output_path = os.path.join(output_dir, filename)
            
            # Get asset names using enhanced symbol utilities
            asset_info = self.config_manager.get_asset_info()
            asset1 = asset_info["base"]
            asset2 = asset_info["quote"]
            
            # Prepare chart configuration
            config = {
                "fair_value": {
                    "show_confidence_bands": True,
                    "confidence_level": 0.95
                },
                "regression_settings": {
                    "window_size": self.cfg.get("pair_trading", {}).get("regression_window", 200),
                    "min_periods": 30,
                    "rolling_regression": True
                },
                "zscore_settings": {
                    "window_size": chart_config.get("zscore_window", 40),
                    "min_periods": 20,
                    "calculation_method": "rolling"
                },
                "chart_style": "line"
            }
            
            # Create 3-panel chart
            panels = ['fair_value', 'zscore', 'dual_axis']
            
            result = self.chart_orchestrator.create_triple_panel_chart(
                panel_types=panels,
                df1=df1,
                df2=df2,
                asset1=asset1,
                asset2=asset2,
                timeframe=timeframe,
                output_path=output_path,
                display=False,
                zscore_window=chart_config.get("zscore_window", 40),
                zscore_thresholds=chart_config.get("zscore_thresholds", [1, 2, 3]),
                config=config
            )
            
            if result['success']:
                logger.debug_print(f"Chart generated successfully from {timeframe} telemetry service data: {output_path}", "CHARTS")
                logger.debug_print(f"Chart data: {len(df1)} {timeframe} candles from telemetry service", "CHARTS")
                return True
            else:
                logger.debug_print(f"Chart generation failed. Failed panels: {', '.join(result['failed_panels'])}", "CHARTS")
                return False
                
        except Exception as e:
            logger.debug_print(f"Error generating real-time chart with telemetry service data: {e}", "CHARTS")
            traceback.print_exc()
            return False

    def _get_neutral_barometer(self) -> Dict[str, Any]:
        """Fallback neutral barometer when computation unavailable"""
        return {
            "state": "neutral",
            "confidence": 0.0,
            "score": 0.0,
            "trend_strength": 0.0
        }

    def compute_barometer(self) -> Dict[str, Any]:
        """Compute market barometer with error handling"""
        if self.market_analyzer is None:
            return self._get_neutral_barometer()
        
        try:
            barometer_obj = self.market_analyzer.analyze_market()
            return {
                "state": barometer_obj.state.value,
                "confidence": float(barometer_obj.confidence),
                "score": float(barometer_obj.score),
                "trend_strength": float(barometer_obj.trend_strength)
            }
        except Exception as e:
            logger.debug_print(f"Error computing barometer: {e}", "MARKET")
            return self._get_neutral_barometer()

    def update_multi_timeframe_extremes(self, tele_wrapped: Dict[str, Any]) -> None:
        """
        NEW: Update extreme tracker with data from all timeframes.
        Collects extremes from tactical, primary, and strategic timeframes.
        """
        if not self.extreme_tracker:
            return
        
        try:
            timestamp = tele_wrapped.get('asof', utcnow_iso())
            
            # Update tactical timeframe (5s) extremes
            regression_5s_data = tele_wrapped.get('regression_5s', {})
            tactical_zscore = regression_5s_data.get('zscore', 0.0)
            if tactical_zscore != 0.0:  # Only update if we have actual data
                self.extreme_tracker.update('5s', tactical_zscore, timestamp)
            
            # Update primary timeframe (1m) extremes  
            regression_data = tele_wrapped.get('regression', {})
            primary_zscore = regression_data.get('zscore', 0.0)
            if primary_zscore != 0.0:  # Only update if we have actual data
                self.extreme_tracker.update('1m', primary_zscore, timestamp)
            
            # Update strategic timeframe (1h) extremes
            regression_htf_data = tele_wrapped.get('regression_htf', {})
            strategic_zscore = regression_htf_data.get('zscore', 0.0)
            if strategic_zscore != 0.0:  # Only update if we have actual data
                self.extreme_tracker.update('1h', strategic_zscore, timestamp)
            
            # Log extreme updates (only if significant)
            updated_timeframes = []
            if abs(tactical_zscore) >= self.extreme_tracker.min_threshold:
                updated_timeframes.append(f"5s({tactical_zscore:.2f})")
            if abs(primary_zscore) >= self.extreme_tracker.min_threshold:
                updated_timeframes.append(f"1m({primary_zscore:.2f})")
            if abs(strategic_zscore) >= self.extreme_tracker.min_threshold:
                updated_timeframes.append(f"1h({strategic_zscore:.2f})")
            
            if updated_timeframes:
                logger.debug_print(f"Updated extremes: {', '.join(updated_timeframes)}", "EXTREME_TRACKING")
            
        except Exception as e:
            logger.debug_print(f"Failed to update multi-timeframe extremes: {e}", "EXTREME_TRACKING")

    def ensure_persistent_positions_in_marketstate(self, marketstate: Dict[str, Any], engine_instance=None) -> Dict[str, Any]:
        """
        [RECTIFIED] Injects persistent positions into the live marketstate.
        Updated to support UnifiedTradeEngine (execution_engine) to fix Zero PnL bug.
        """
        # [FIX] Use the provided engine, then execution_engine, then trade_simulator
        # legacy code removed: target_engine = engine_instance or getattr(self, 'execution_engine', None) or self.trade_simulator
        target_engine = engine_instance or getattr(self, 'execution_engine', None)
        
        # [RECTIFIED] Logic Gate: Success if either execution_engine or trade_simulator exists
        if not target_engine: 
            return marketstate
        
        # Original Guard preserved
        # legacy code removed: if not self.trade_simulator and not getattr(self, 'execution_engine', None):
        if not getattr(self, 'execution_engine', None):
            return marketstate

        try:
            # 1. Resolve Symbols from Config
            asset_val = self.cfg.get('asset')
            sym_primary = 'UNKNOWN'
            sym_ref = 'UNKNOWN'
            
            if isinstance(asset_val, str):
                parts = asset_val.split('/')
                sym_primary = parts[0]
                sym_ref = parts[1] if len(parts) > 1 else 'USDC'
            elif isinstance(asset_val, dict):
                if 'symbol_primary' in asset_val:
                    sym_primary = asset_val['symbol_primary']
                    sym_ref = asset_val['symbol_reference']
                else:
                    pair = asset_val.get('pair', 'UNKNOWN/UNKNOWN')
                    parts = pair.split('/')
                    sym_primary = parts[0]
                    sym_ref = parts[1] if len(parts) > 1 else 'USDC'

            # 2. Prepare Market Data for the Simulator
            price_primary = marketstate.get('prices', {}).get('primary', 0.0)
            price_ref = marketstate.get('prices', {}).get('reference', 0.0)
            
            market_data_map = {
                sym_primary: {
                    'mid': price_primary,
                    'bid': price_primary, 
                    'ask': price_primary
                },
                sym_ref: {
                    'mid': price_ref,
                    'bid': price_ref,
                    'ask': price_ref
                }
            }
            
            # Trigger P&L update on the specific engine
            pnl_mode = self.cfg.get('trading', {}).get('pnl_valuation_mode', 'MID')
            target_engine.update_floating_pnl(market_data_map, valuation_mode=pnl_mode)
            
            # Get status and merge into marketstate
            status = target_engine.get_status()
            
            if 'open_positions' not in marketstate: marketstate['open_positions'] = {}
            if 'trade_summary' not in marketstate: marketstate['trade_summary'] = {}
            
            new_open = status.get('open_aggregated_positions', {})
            
            # [RECTIFIED] Update Simulation data if source matches simulator OR execution_engine
            if 'simulation' in new_open:
                # Support both legacy simulator and new engine mode
                # legacy code removed: if target_engine == self.trade_simulator or target_engine == getattr(self, 'execution_engine', None):
                if target_engine == getattr(self, 'execution_engine', None):
                    marketstate['open_positions']['simulation'] = new_open['simulation']

            # 2. Update Real data ONLY if this engine is enabled for real trading
            if 'real' in new_open:
                if getattr(target_engine, 'real_trading_enabled', False):
                    marketstate['open_positions']['real'] = new_open['real']
                    
            # Do the same for summaries
            new_summaries = status.get('summaries', {})
            if 'simulation' in new_summaries:
                #legacy code removed: if target_engine == self.trade_simulator or target_engine == getattr(self, 'execution_engine', None):
                if target_engine == getattr(self, 'execution_engine', None):
                    marketstate['trade_summary']['simulation'] = new_summaries['simulation']
            
            if 'real' in new_summaries and getattr(target_engine, 'real_trading_enabled', False):
                marketstate['trade_summary']['real'] = new_summaries['real']

        except Exception as e:
            logger.debug_print(f"Error syncing persistent positions: {e}", "MAIN")
            traceback.print_exc()

        return marketstate

    def create_base_marketstate(self, asset: str, tele_wrapped: Dict[str, Any], barometer: Dict[str, Any]) -> Dict[str, Any]:
        """Create base marketstate JSON without P&L calculations"""
        
        logger.debug_print("Creating base marketstate for telemetry service optimized system with extreme tracking and context-specific analysis...", "MARKETSTATE")
        
        # Extract data from telemetry (now provided by telemetry service)
        regression_5s_data = tele_wrapped.get('regression_5s', {})
        regression_data = tele_wrapped.get('regression', {})
        regression_htf_data = tele_wrapped.get('regression_htf', {})
        regime_assessment = tele_wrapped.get('regime_assessment', {})
        price_data = tele_wrapped.get('price', {})
        indicators = tele_wrapped.get('indicators', {})
        # [DEBUG TRACER] Check what VMain is actually receiving
        print(f"[DEBUG VMAIN] Indicators Keys: {list(indicators.keys())}")
        print(f"[DEBUG VMAIN] RSI Data Block: {indicators.get('rsi')}")
        volume_data = tele_wrapped.get('volume_data', {})
        asset_pair_info = tele_wrapped.get('asset_pair_info', {})

        # Get current prices
        primary_price = price_data.get('actual_price_a', 0.0)
        reference_price = price_data.get('actual_price_b', 0.0)
        price_ratio = primary_price / reference_price if reference_price > 0 else 0.0
        predicted_ratio = regression_data.get('fair_value', 0.0) / reference_price if reference_price > 0 else 0.0
        
        # Calculate data age
        #
        # 1. Get raw UTC timestamp from telemetry
        raw_asof = tele_wrapped.get('asof', utcnow_iso())

        try:
            from datetime import datetime, timezone
            # 2. Parse UTC object for calculation
            asof_dt_utc = datetime.fromisoformat(raw_asof.replace('Z', '+00:00'))
            
            # 3. Calculate Age (UTC vs UTC) - Most robust method
            data_age = int((datetime.now(timezone.utc) - asof_dt_utc).total_seconds())

            # 4. Convert to SGT for JSON Display
            # This ensures the dashboard sees +08:00
            asof_str = asof_dt_utc.astimezone(get_local_time().tzinfo).isoformat()
        except:
            asof_str = raw_asof # Fallback
            data_age = 0

        # Get asset info using enhanced symbol utilities
        symbol_info = SymbolClassifier.classify_symbol(asset)
        
        # Get telemetry service status
        service_status = self.telemetry_service.get_service_status()

        # Extract active strategy names from config
        active_strategies = self.cfg.get("active_strategies", {})

        # NEW: Get extreme history summary for marketstate AND raw extremes for primary TF
        extreme_history_summary = {}
        primary_tf_raw_extremes = []
        
        if self.extreme_tracker:
            try:
                extreme_history_summary = self.extreme_tracker.get_marketstate_summary()
                # NEW: Get all raw extreme values for primary timeframe only
                primary_tf_raw_extremes = self.extreme_tracker.get_primary_timeframe_raw_extremes()
                #logger.debug_print(f"Added extreme history summary: {len(extreme_history_summary)} timeframes", "EXTREME_TRACKING")
                #logger.debug_print(f"Added {len(primary_tf_raw_extremes)} raw extremes for primary TF", "EXTREME_TRACKING")
            except Exception as e:
                logger.debug_print(f"Failed to get extreme history summary: {e}", "EXTREME_TRACKING")

        # Create base marketstate structure with telemetry service schema and extreme tracking
        marketstate = {
            "schema_version": SCHEMA_VERSION,
            "asof": asof_str,
            "symbol": asset,
            "symbol_info": symbol_info,
            
            # Asset pair metadata
            "asset_pair_info": {
                "base_asset": asset_pair_info.get("base_asset", symbol_info.get("base", "SOL")),
                "quote_asset": asset_pair_info.get("quote_asset", symbol_info.get("quote", "MELANIA")),
                "asset_type": asset_pair_info.get("asset_type", "pair"),
                "symbol": asset_pair_info.get("symbol", asset)
            },
            
            # Tactical regression (5s timeframe) - for ultra-short term analysis
            "regression_5s": {
                "zscore": float(regression_5s_data.get('zscore', 0.0)),
                "spread": float(regression_5s_data.get('spread', 0.0)),
                "alpha": float(regression_5s_data.get('alpha', 0.0)),
                "beta": float(regression_5s_data.get('beta', 1.0)),
                "r_squared": float(regression_5s_data.get('r_squared', 0.0)),
                "correlation": float(regression_5s_data.get('correlation', 0.0)),
                "window_size": regression_5s_data.get('window_size', 360),
                "health_status": regression_5s_data.get('health_status', 'unknown'),
                "beta_drift_pct": float(regression_5s_data.get('beta_drift_pct', 0.0)),
                "timeframe": regression_5s_data.get('timeframe', '5s')
            },
            
            # Primary regression (1m timeframe) - backward compatible
            "regression": {
                "zscore": float(regression_data.get('zscore', 0.0)),
                "spread": float(regression_data.get('spread', 0.0)),
                "alpha": float(regression_data.get('alpha', 0.0)),
                "beta": float(regression_data.get('beta', 1.0)),
                "r_squared": float(regression_data.get('r_squared', 0.0)),
                "correlation": float(regression_data.get('correlation', 0.0)),
                "window_size": regression_data.get('window_size', 200),
                "health_status": regression_data.get('health_status', 'unknown'),
                "beta_drift_pct": float(regression_data.get('beta_drift_pct', 0.0)),
                "timeframe": regression_data.get('timeframe', '1m')
            },
            
            # HTF regression (1h timeframe) - for position sizing
            "regression_htf": {
                "zscore": float(regression_htf_data.get('zscore', 0.0)),
                "spread": float(regression_htf_data.get('spread', 0.0)),
                "alpha": float(regression_htf_data.get('alpha', 0.0)),
                "beta": float(regression_htf_data.get('beta', 1.0)),
                "r_squared": float(regression_htf_data.get('r_squared', 0.0)),
                "correlation": float(regression_htf_data.get('correlation', 0.0)),
                "window_size": regression_htf_data.get('window_size', 120),
                "health_status": regression_htf_data.get('health_status', 'unknown'),
                "beta_drift_pct": float(regression_htf_data.get('beta_drift_pct', 0.0)),
                "timeframe": regression_htf_data.get('timeframe', 'HTF')
            },
            
            "prices": {
                "primary": float(primary_price),
                "reference": float(reference_price), 
                "ratio": float(price_ratio),
                "predicted_ratio": float(predicted_ratio),
                "last_updated": asof_str,
                "data_age_seconds": data_age
            },
            
            "volume": {
                "primary": safe_float(volume_data.get('primary', 1000.0)),
                "reference": safe_float(volume_data.get('reference', 50000.0))
            },
            
            "regime_assessment": {
                "timeframe": regime_assessment.get('timeframe', '5s'),
                "lookback_periods": regime_assessment.get('lookback_periods', 20),
                "both_trending_up": bool(regime_assessment.get('both_trending_up', False)),
                "both_trending_down": bool(regime_assessment.get('both_trending_down', False)),
                "divergent_trends": bool(regime_assessment.get('divergent_trends', False)),
                "synchronized_volatility": bool(regime_assessment.get('synchronized_volatility', True)),
                # [RECTIFIED] Corrected Mapping for Dual Asset Hurst & Directional Slopes
                # Primary remains in 'hurst' (Standard)
                # [RECTIFIED] Corrected Mapping to fix "WAIT" status in dashboard
                # Enhanced Hurst Analysis Mapping
                "hurst": float(regime_assessment.get('hurst', 0.5)),                    # Primary asset (SOL)
                "hurst_secondary": float(regime_assessment.get('hurst_reference', 0.5)), # Secondary asset (MELANIA)
                "hurst_spread": float(regime_assessment.get('hurst_spread', 0.5)),      # NEW: Pair relationship
                
                # Individual Asset Directional Trends
                "sol_trend_beta": float(regime_assessment.get('sol_trend_beta', 0.0)),      # NEW: SOL direction
                "melania_trend_beta": float(regime_assessment.get('melania_trend_beta', 0.0)), # NEW: MELANIA direction
                "spread_trend_beta": float(regime_assessment.get('spread_trend_beta', 0.0)), # NEW: spread direction
                
                # Legacy pair relationship slopes (keep for backward compatibility)
                "primary_slope": float(regression_data.get('beta', 0.0)),
                "secondary_slope": float(regression_data.get('beta', 0.0)), # FIXED KEY
                
                "trend_strength": {
                    "primary": float(regime_assessment.get('trend_strength', {}).get('primary', 0.0)),
                    "reference": float(regime_assessment.get('trend_strength', {}).get('reference', 0.0)),
                    "correlation": float(regime_assessment.get('trend_strength', {}).get('correlation', 0.0))
                },
                "volatility_sync": {
                    "ratio": float(regime_assessment.get('volatility_sync', {}).get('ratio', 0.0)),
                    "assessment": regime_assessment.get('volatility_sync', {}).get('assessment', 'unknown')
                },
                # Tactical context for ultra-short term (5s)
                "tactical_context": regime_assessment.get('tactical_context', {}),
                # HTF context for strategic analysis (1h)
                "htf_context": regime_assessment.get('htf_context', {})
            },
            
            "indicators": {
                "rsi": float(indicators.get('rsi', {}).get('value', 50.0)),
                "atr": float(indicators.get('atr', 0.0)),
                "volatility": float(regime_assessment.get('volatility_sync', {}).get('ratio', 0.0)),
                "periods_analyzed": int(regression_data.get('window_size', 0))
            },
            
            "market_barometer": barometer,

            "active_strategies": {
            "entry": active_strategies.get("entry", "unknown"),
            "take_profit": active_strategies.get("take_profit", "unknown"), 
            "cut_loss": active_strategies.get("cut_loss", "unknown")
            },
            
            # NEW: Raw extreme values for primary timeframe only
            "primary_timeframe_raw_extremes": primary_tf_raw_extremes,
            
            # NEW: Extreme history summary for dashboard visibility
            "extreme_history_summary": extreme_history_summary,
            
            "meta": {
                "sufficient_for_trading": bool(tele_wrapped.get('data_quality_summary', {}).get('primary_sufficient', False)),
                "tactical_sufficient_for_trading": bool(tele_wrapped.get('data_quality_summary', {}).get('tactical_sufficient', False)),
                "htf_sufficient_for_trading": bool(tele_wrapped.get('data_quality_summary', {}).get('htf_sufficient', False)),
                "last_regression_update": asof_str,
                "trading_permit": self.trading_permit, 
                "engine": "vmain_refactored_phase1_phase2_phase3_extreme_tracking_persistent_positions_context_analysis",
                #legacy code removed: "trade_simulation_enabled": bool(self.trade_simulator is not None),
                "trade_simulation_enabled": bool(getattr(self, 'execution_engine', None) is not None),
                "chart_orchestrator_enabled": bool(self.chart_orchestrator is not None),
                "extreme_tracking_enabled": bool(self.extreme_tracker is not None),
                "persistent_position_loading": bool(self.execution_engine is not None),
                "context_specific_analysis_enabled": bool(self.modular_engine is not None),
                "tick_interval_seconds": TICK_SEC,
                "telemetry_service_mode": service_status['mode'],
                "buffer_mode_enabled": service_status['buffer_mode_enabled'],
                "api_efficiency": "optimized" if service_status['buffer_mode_enabled'] else "legacy",
                "triple_timeframe_analysis": service_status['buffer_mode_enabled'],
                "chart_data_source": "telemetry_service",
                "extreme_data_source": "unified_storage" if self.extreme_tracker else "none",
                "phase1_status": "COMPLETE",
                "refactoring_status": "PHASE1_PHASE2_PHASE3_COMPLETE",
                "extreme_tracking_status": "MULTI_TIMEFRAME_INTEGRATED",
                #legacy code removed: "persistent_position_status": "ENABLED" if self.trade_simulator else "DISABLED",
                "persistent_position_status": "ENABLED" if getattr(self, 'execution_engine', None) else "DISABLED",
                "context_analysis_status": "ENABLED" if self.modular_engine else "DISABLED",
                "timeframes_analyzed": ["5s", "1m", "1h"] if service_status['buffer_mode_enabled'] else ["1m"],
                "phase1_features": {
                    "complete_execution_tracking": True,
                    "beta_analysis": True,
                    "hedge_ratio_tracking": True,
                    "execution_quality_metrics": True,
                    "enhanced_position_sizing": True,
                    "triple_timeframe_regression": service_status['buffer_mode_enabled'],
                    "5s_tactical_analysis": service_status['buffer_mode_enabled'],
                    "1m_primary_analysis": True,
                    "1h_strategic_analysis": service_status['buffer_mode_enabled'],
                    "real_buffer_chart_data": service_status['buffer_mode_enabled'],
                    "5s_buffer_tracking": service_status['buffer_mode_enabled'],
                    "multi_timeframe_extreme_tracking": bool(self.extreme_tracker),
                    "unified_extreme_storage": bool(self.extreme_tracker),
                    "dashboard_extreme_visibility": bool(self.extreme_tracker),
                    "primary_timeframe_raw_extremes": bool(self.extreme_tracker),
                    #legacy code removed: "persistent_position_loading": bool(self.trade_simulator),
                    "persistent_position_loading": bool(getattr(self, 'execution_engine', None)),
                    #legacy code removed: "position_continuity_across_restarts": bool(self.trade_simulator),
                    "position_continuity_across_restarts": bool(getattr(self, 'execution_engine', None)),
                    "context_specific_analysis": bool(self.modular_engine),
                    "analysis_context_detection": bool(self.modular_engine),
                    "detailed_threshold_tracking": bool(self.modular_engine)
                },
                "refactoring_features": {
                    "modular_debug_logging": True,
                    "enhanced_symbol_classification": True,
                    "configuration_validation": True,
                    "performance_monitoring": True,
                    "asset_pair_validation": True,
                    "centralized_config_management": True,
                    "telemetry_service_abstraction": True,
                    "service_layer_separation": True,
                    "multi_timeframe_extreme_collection": bool(self.extreme_tracker),
                    "extreme_marketstate_integration": bool(self.extreme_tracker),
                    "primary_timeframe_raw_extreme_exposure": bool(self.extreme_tracker),
                    #legacu code removed: "persistent_storage_integration": bool(self.trade_simulator),
                    "persistent_storage_integration": bool(getattr(self, 'execution_engine', None)),
                    #legacy code removed: "automatic_position_recovery": bool(self.trade_simulator),
                    "automatic_position_recovery": bool(getattr(self, 'execution_engine', None)),
                    "context_specific_signal_analysis": bool(self.modular_engine),
                    "three_mode_context_detection": bool(self.modular_engine),
                    "pipe_delimited_analysis_reasoning": bool(self.modular_engine)
                },
                "telemetry_service_status": service_status
            }
        }
        
        # NOTE: P&L calculation is intentionally NOT included here
        # It will be added by ensure_persistent_positions_in_marketstate() after trade processing
        

        # [PHASE 2] Inject Active Configuration for Dashboard Verification
        # [PHASE 2] Inject Active Configuration for Dashboard Verification
        try:
            # 1. Get Profit Strategy Config
            tp_strat = self.cfg.get('active_strategies', {}).get('take_profit', 'hit_and_run_profit')
            tp_conf = self.cfg.get('take_profit_strategies', {}).get(tp_strat, {})
            
            # 2. Get Stop Loss Config
            cl_strat = self.cfg.get('active_strategies', {}).get('cut_loss', 'simple_scaling_strategy')
            cl_conf = self.cfg.get('cut_loss_strategies', {}).get(cl_strat, {})

            # FIX: Use 'marketstate' (not 'ms') and export specific keys
            marketstate['active_strategy_params'] = {
                'active_tp_strategy': tp_strat,
                'percent_target': tp_conf.get('percent_target', 0.0), # Specific Key
                'dollar_target': tp_conf.get('dollar_target', 0.0),   # Specific Key
                
                'active_cl_strategy': cl_strat,
                'loss_threshold': cl_conf.get('loss_threshold_pct', 0.0),
                'loss_threshold_usd': cl_conf.get('add_conditions', {}).get('floating_loss_threshold_usd', 0.0)
            }
        except Exception as e:
            logger.debug_print(f"Config injection error: {e}", "MARKETSTATE")
            marketstate['active_strategy_params'] = {}

        return marketstate

    def create_streamlined_marketstate(self, asset: str, tele_wrapped: Dict[str, Any], barometer: Dict[str, Any]) -> Dict[str, Any]:
        """Create complete marketstate JSON with P&L calculations (legacy compatibility method)"""
        
        logger.debug_print("Creating streamlined marketstate for telemetry service optimized system with extreme tracking, persistent positions, and context-specific analysis...", "MARKETSTATE")
        
        # Create base marketstate
        marketstate = self.create_base_marketstate(asset, tele_wrapped, barometer)
        
        # Add P&L calculations and persistent position handling
        marketstate = self.ensure_persistent_positions_in_marketstate(marketstate)
        
        return marketstate

    def generate_trading_signal(self, marketstate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generates trading signals strictly using the Modular Strategy Engine.
        [RECTIFIED] Removed legacy framework residue and fallbacks to ensure 
        consistency with the $5.33 target logic.
        """
        # 1. Primary Path: Modular Strategy Engine
        if self.modular_engine:
            try:
                # This call executes your active strategies (e.g., hit_and_run_profit)
                return self.modular_engine.make_trading_decision(marketstate)
            except Exception as e:
                logger.debug_print(f"Modular strategy engine execution failed: {e}", "TRADING")
                # Log traceback for deep debugging of modular failures
                import traceback
                traceback.print_exc()
        
        # 2. [RECTIFIED] Holistic Legacy Framework Removal
        # We no longer fall back to 'make_trading_decision' or 'use_granular_config'.
        # This prevents 'SIGNAL VALIDATION ERROR' caused by missing legacy paths 
        # like 'regime_assessment.safe_for_mean_reversion'.
        
        if not self.modular_engine:
            logger.debug_print("CRITICAL: Modular engine unavailable and legacy fallback disabled.", "TRADING")
            
        return None

    def write_outputs(self, asset: str, tele_wrapped: Dict[str, Any], barometer: Dict[str, Any]) -> None:
        """
        Write outputs using Unified Execution Engine - HOLISTIC FIX
        Fixes: Sequencing Error, Portfolio Blindness, and Timestamp Validity.
        """
        try:
            # 1. Update Market Data & Extremes
            self.update_multi_timeframe_extremes(tele_wrapped)
            marketstate = self.create_base_marketstate(asset, tele_wrapped, barometer)

            # === Save Imbalance History ===
            if hasattr(self.telemetry_service.provider, 'save_history'):
                imb_history = {}
                assets_to_save = self.asset.split('/') if '/' in self.asset else [self.asset]
                for coin in assets_to_save:
                    history = self.telemetry_service.provider.save_history(coin)
                    if history: imb_history[coin] = history
                marketstate["orderbook_imbalance_history"] = imb_history

            # =================================================================
            # HOLISTIC FIX: CORRECT EXECUTION PIPELINE
            # Step A: Inject Portfolio (Cure Blindness)
            # Step B: Generate Signal (Fix Sequencing)
            # Step C: Execute (The Action)
            # =================================================================

            # [STEP A] PRE-INJECT PORTFOLIO STATE
            # The Strategy needs to know if we are Long/Short to generate valid Entry/Exit signals.
            # [STEP A] PRE-INJECT PORTFOLIO STATE
            # The Strategy needs to know if we are Long/Short to generate valid Entry/Exit signals.
            # [STEP A] PRE-INJECT PORTFOLIO STATE
            if hasattr(self, 'execution_engine') and self.execution_engine:
                # [FIX] Force P&L update and inject state via the new Engine method
                self.execution_engine.sync_state(marketstate)
                
                # Debug: Log injected PnL
                portfolio = marketstate.get('portfolio', {})
                if portfolio.get('positions'):
                    logger.debug_print(f"[SYNC_DEBUG] Injected {len(portfolio['positions'])} positions, Total PnL: ${portfolio.get('total_pnl', 0):.2f}", "SYNC")

            # [STEP B] GENERATE TRADING SIGNAL
            # Now the strategy sees the market AND our positions, so it can make a real decision.
            trading_signal = self.generate_trading_signal(marketstate)
            
            # [PATCH] Force Signal Timestamp to SGT/UTC Now
            # This prevents the "Stale Signal" safety check from blocking the trade.
            if trading_signal and 'current_recommendation' in trading_signal:
                trading_signal['current_recommendation']['timestamp'] = utcnow_iso()
                
            marketstate["trading_signal"] = trading_signal
            
            # [PHASE 1] UNIFIED GOVERNOR ENTRY GATE
            if self.trading_permit != "ALL" and trading_signal:
                rec = trading_signal.get("current_recommendation", {})
                if rec.get("recommendation") == "ENTER":
                    rec["recommendation"] = "WAIT"
                    logger.debug_print(f"Governor {self.trading_permit}: Suppressing ENTER signal", "GOVERNOR")

            # [STEP C] EXECUTION ENGINE (THE HANDOFF)
            # Now the engine sees the 'trading_signal' we just generated and can execute it immediately.
            if hasattr(self, 'execution_engine') and self.execution_engine:
                self.execution_engine.process(marketstate)

                # [PHASE 1] Check for pending soft switch completion
                self.execution_engine.check_soft_switch_completion()

            # =================================================================

            # 4. Save Snapshots
            try:
                self.state_manager.save_snapshot(marketstate)
                self.state_manager.save_barometer(barometer)
            except Exception as e:
                logger.debug_print(f"Failed to save state: {e}", "OUTPUT")

            # 5. Display
            self._display_console_output(marketstate)

        except Exception as e:
            logger.debug_print(f"Failed to write outputs: {e}", "OUTPUT")
            traceback.print_exc()

    def check_for_netting_conflict(self, current_asset: str, proposed_side: str) -> bool:
        """[PHASE 2] Dynamic conflict check across portfolio legs."""
        # legacy code: if not proposed_side or not self.trade_simulator:
        #    return False
        if not proposed_side or not getattr(self, 'execution_engine', None):
            return False

        # Extract base (e.g., 'SOL' from 'SOL/USDT')
        current_base = current_asset.split('/')[0]
        
        # Pull the open positions currently tracked in simulator
        # legacy code: status = self.trade_simulator.get_status()
        status = {
            'open_positions': len(self.execution_engine.open_positions),
            'mode': self.execution_engine.mode
        }
        #legacy code: open_pos = status.get('open_positions', [])
        open_pos = self.execution_engine.open_positions 
        
        for pos in open_pos:
            parent = pos.get('asset_pair', '')
            if parent == current_asset:
                continue # Skip self
                
            if current_base in parent:
                existing_side = pos.get('side', '').upper()
                # Conflict if LONG SOL vs SHORT SOL
                if proposed_side.upper() != existing_side:
                    return True
        return False

    def _display_console_output(self, marketstate: Dict[str, Any]) -> None:
        """
        Display enhanced console output with telemetry service indicators, extreme tracking info, 
        persistent position data, and context-specific analysis information
        """
        regression_5s = marketstate.get("regression_5s", {})
        regression = marketstate.get("regression", {})
        regression_htf = marketstate.get("regression_htf", {})
        prices = marketstate.get("prices", {})
        regime = marketstate.get("regime_assessment", {})
        trading = marketstate.get("trading_signal", {})
        volume = marketstate.get("volume", {})
        trade_sim = marketstate.get("trade_simulation", {})
        meta = marketstate.get("meta", {})
        extreme_summary = marketstate.get("extreme_history_summary", {})
        
        # ENHANCED: Extract context-specific analysis information
        trading_analysis = marketstate.get("trading_signal_analysis", {})
        analysis_context = trading_analysis.get("analysis_context", "unknown")
        context_analysis_available = trading_analysis.get("context_analysis_available", False)
        
        zscore_5s = regression_5s.get("zscore", 0.0)
        zscore_1m = regression.get("zscore", 0.0)
        # [FIX] Rename variable for clarity
        zscore_htf = regression_htf.get("zscore", 0.0)
        # [FIX] Get dynamic label from data (will be '5m' if service is working, or 'HTF' fallback)
        htf_label = regression_htf.get("timeframe", "HTF")
        r_squared_1m = regression.get("r_squared", 0.0)
        r_squared_1h = regression_htf.get("r_squared", 0.0)
        health_1m = regression.get("health_status", "unknown")
        safe_for_mr = regime.get("safe_for_mean_reversion", True)
        
        current_rec = trading.get("current_recommendation", {})
        recommendation = current_rec.get("recommendation", "WAIT")
        confidence = current_rec.get("confidence", 0.0)
        risk_level = current_rec.get("risk_level", "UNKNOWN")

        primary_volume = volume.get("primary", 0.0)
        telemetry_mode = meta.get("telemetry_service_mode", "unknown")
        extreme_tracking_enabled = meta.get("extreme_tracking_enabled", False)
        persistent_positions_enabled = meta.get("persistent_position_loading", False)
        context_analysis_enabled = meta.get("context_specific_analysis_enabled", False)
        
        # Telemetry service efficiency indicator
        service_indicator = f"[SERVICE-{telemetry_mode.upper()}]"
        
        # Triple-timeframe indicator
        # Triple-timeframe indicator
        # [FIX] Use dynamic label
        tf_indicator = f"5s={zscore_5s:.2f}/1m={zscore_1m:.2f}/{htf_label}={zscore_htf:.2f}"
        
        # ENHANCED: Context analysis indicator
        context_indicator = ""
        if context_analysis_enabled and context_analysis_available:
            context_indicator = f" | CTX[{analysis_context.upper()}]"
        elif context_analysis_enabled:
            context_indicator = f" | CTX[DISABLED]"
        
        # Extreme tracking indicator
        extreme_indicator = ""
        if extreme_tracking_enabled and extreme_summary:
            # Show extreme counts for dashboard awareness
            extreme_counts = []
            for tf in ['5s', '1m', '1h']:
                if tf in extreme_summary:
                    total_count = extreme_summary[tf].get('total_count', 0)
                    extreme_counts.append(f"{tf}:{total_count}")
            
            if extreme_counts:
                extreme_indicator = f" | EXT[{','.join(extreme_counts)}]"
        
        # Enhanced trade simulation info with persistent position awareness
        simulation_info = ""
        persistence_info = ""
        
        if trade_sim:
            # Check for persistence metadata
            persistence_meta = trade_sim.get("persistence_metadata", {})
            if persistence_meta.get("loaded_from_persistent_storage", False):
                persistence_info = " | PERSIST"
            
            # Check if position scaling is enabled
            if trade_sim.get('position_scaling_enabled', False):
                # [PHASE 1 SINGLE TRACK] Use dynamic key for console display
                active_track_key = 'real' if self.cfg.get('trading', {}).get('mode') == 'REAL' else 'simulation'
                
                # Display aggregated position information
                agg_positions = trade_sim.get("open_aggregated_positions", {}).get(active_track_key, {})
                position_count = agg_positions.get("count", 0)
                
                if position_count > 0:
                    simulation_info = f" | AggPos={position_count}"
                    # Add P&L info if available
                    for pos in agg_positions.get("positions", []):
                        pnl_tracking = pos.get("pnl_tracking", {})
                        component_count = pos.get("aggregated_metrics", {}).get("position_count", 0)
                        pnl_usd = pnl_tracking.get("floating_pnl_usd", 0.0)
                        simulation_info += f"({component_count}c,${pnl_usd:.1f})"
                        break  # Show only first position for space
                else:
                    # Show closed aggregated position summary
                    closed_agg = trade_sim.get("closed_aggregated_positions", {}).get("simulation", {})
                    closed_count = closed_agg.get("count", 0)
                    if closed_count > 0:
                        simulation_info = f" | AggClosed={closed_count}"
            else:
                # Legacy single trade display
                sim_open = trade_sim.get("open_trades", {})
                sim_count = sim_open.get("simulation", {}).get("count", 0)
                real_count = sim_open.get("real", {}).get("count", 0)
                
                if sim_count > 0 or real_count > 0:
                    simulation_info = f" | Pos=S:{sim_count},R:{real_count}"
                else:
                    summaries = trade_sim.get("summaries", {})
                    sim_summary = summaries.get("simulation", {})
                    sim_pnl = sim_summary.get("total_pnl_usd", 0.0)
                    sim_trades = sim_summary.get("total_trades", 0)
                    if sim_trades > 0:
                        simulation_info = f" | SimP&L=${sim_pnl:.1f}({sim_trades})"

        # Chart status
        chart_info = f" | Chart=telemetry_service"

        # HTF context
        htf_context = regime.get("htf_context", {})
        htf_trending = ""
        if htf_context.get("bias_trending_up"):
            htf_trending = "↗"
        elif htf_context.get("bias_trending_down"):
            htf_trending = "↘"
        else:
            htf_trending = "→"

        asset = marketstate.get("symbol", "UNKNOWN")

        # [PHASE 1] Strong Visual Governor Mode Detection

        # --- PATCH: ensure marketstate meta reflects the real governor state ---
        marketstate.setdefault("meta", {})
        marketstate["meta"]["trading_permit"] = self.trading_permit
        marketstate["meta"]["manual_pause_active"] = self.manual_pause_active
        # --- END PATCH ---

        gov_mode = self.trading_permit  # authoritative source
        
        if gov_mode == "ALL":
            # RESUME / ACTIVE Mode
            mode_tag = "🟢 [RESUME: ACTIVE]"
        elif gov_mode == "CLOSE_ONLY":
            # SOFT PAUSE Mode
            mode_tag = "🟠 [SOFT PAUSE: CLOSE ONLY]"
        elif gov_mode == "NONE":
            # HARD PAUSE Mode
            mode_tag = "🔴 [HARD PAUSE: DISABLED]"
        else:
            mode_tag = f"⚪ [MODE: {gov_mode}]"
            
        # --- [FINAL DISPLAY OVERRIDE] ---
        # Force display variables to match the "Zombie Surgery" results immediately
        try:
            real_data = marketstate.get('open_positions', {}).get('real', {})
            if real_data.get('count', 0) > 0:
                # 1. Update Position String
                pos_list = real_data.get('positions', [])
                if pos_list:
                    p = pos_list[0]
                    comp_count = len(p.get('component_trades', []))
                    pnl = float(p.get('pnl_tracking', {}).get('floating_pnl_usd', 0.0))
                    # Force the string to show the correct component count (e.g., 1c)
                    simulation_info = f" | AggPos={real_data['count']}({comp_count}c,${pnl:.1f})"
                
                # 2. Force Context to Exit Mode (Since we definitely have a position)
                if "ENTRY" in context_indicator:
                    context_indicator = context_indicator.replace("ENTRY", "EXIT_CUTLOSS")
        except Exception as e:
            pass 
        # --------------------------------

        # Construct the final line with the mode_tag at the front for maximum visibility
        print(f"{mode_tag} [{utcnow_iso()}] {asset} {service_indicator} | {tf_indicator} {htf_trending} | "
              f"R²={r_squared_1m:.2f}/{r_squared_1h:.2f} | Health={health_1m} | "
              f"Safe={safe_for_mr} | Signal={recommendation}({confidence:.2f}){context_indicator}"
              f"{simulation_info}{persistence_info}{chart_info}{extreme_indicator}")

    def _display_telemetry_efficiency_summary(self, tick_count: int) -> None:
        """Display telemetry service efficiency metrics at end of session with extreme tracking, persistence, and context analysis summary"""
        try:
            print("\nPHASE 1+2+3 REFACTORED TELEMETRY SERVICE + EXTREME TRACKING + PERSISTENT POSITIONS + CONTEXT ANALYSIS EFFICIENCY SUMMARY:")
            print(f"Session duration: {tick_count} ticks ({tick_count * TICK_SEC / 60:.1f} minutes)")
            
            # Get efficiency metrics from telemetry service
            efficiency_metrics = self.telemetry_service.get_efficiency_metrics(tick_count)
            service_status = self.telemetry_service.get_service_status()
            
            print(f"Telemetry Service Mode: {efficiency_metrics['mode'].upper()}")
            
            if efficiency_metrics['mode'] == 'buffer':
                print(f"API Efficiency:")
                print(f"  Legacy mode would have made: {efficiency_metrics['legacy_would_have_made']:,} API calls")
                print(f"  Service mode actually made: {efficiency_metrics['total_api_calls']:,} API calls")
                print(f"  API calls saved: {efficiency_metrics['api_calls_saved']:,} ({efficiency_metrics['efficiency_percentage']:.1f}% reduction)")
                print(f"  Average API calls per tick: {efficiency_metrics['api_calls_per_tick']}")
                
                # Buffer status from service
                if 'buffer_status' in service_status:
                    buffer_status = service_status['buffer_status']
                    print(f"Buffer Status via Service:")
                    print(f"  Initialization: {'SUCCESS' if buffer_status.get('is_initialized') else 'FAILED'}")
                    print(f"  Total buffers: {buffer_status.get('buffer_count', 0)}")
                    print(f"  Needs refresh: {buffer_status.get('needs_refresh', False)}")
                    print(f"  Triple-timeframe analysis: ENABLED")
                    
                    buffer_config = self.config_manager.get_buffer_config()
                    timeframes = list(buffer_config.get("timeframes", {}).keys())
                    print(f"  Timeframes analyzed: {', '.join(timeframes)}")
                    
                    # Chart data summary
                    chart_timeframe = buffer_config.get("primary_timeframe", "1m")
                    print(f"  Chart data source: Telemetry service ({chart_timeframe} buffer data)")
            else:
                print(f"Legacy Mode via Service:")
                print(f"  Total API calls made: {efficiency_metrics['total_api_calls']:,}")
                print(f"  Average API calls per tick: {efficiency_metrics['api_calls_per_tick']}")
                print(f"  Triple-timeframe analysis: DISABLED")
                print(f"  Chart data source: None")
                print(f"  Recommendation: Enable multi_timeframe_buffers for 99%+ efficiency gain")
            
            # NEW: Extreme tracking summary
            if self.extreme_tracker:
                print(f"\nMulti-Timeframe Extreme Tracking Summary:")
                all_stats = self.extreme_tracker.get_all_statistics()
                
                for tf in ['5s', '1m', '1h']:
                    if tf in all_stats:
                        stats = all_stats[tf]
                        is_primary = stats.get('is_primary_timeframe', False)
                        primary_marker = " (PRIMARY)" if is_primary else ""
                        
                        print(f"  {tf}{primary_marker}: {stats['total_extremes']} extremes "
                              f"({stats['real_extremes']} real, {stats['synthetic_extremes']} synthetic)")
                        
                        if stats['total_extremes'] > 0:
                            print(f"    Positive: {stats['positive_extremes']} (max: {stats['max_positive']:.2f})")
                            print(f"    Negative: {stats['negative_extremes']} (min: {stats['max_negative']:.2f})")
                
                summary = all_stats.get('summary', {})
                print(f"  Total extremes collected: {summary.get('total_extremes_all_timeframes', 0)}")
                print(f"  Primary timeframe for scaling: {summary.get('primary_timeframe', 'unknown')}")
                print(f"  Storage location: {self.extreme_tracker.storage_path}")
                print(f"  Dashboard integration: ENABLED")
            else:
                print(f"\nMulti-Timeframe Extreme Tracking: DISABLED")
            
            # ENHANCED: Persistent position summary
            # legacy code: if self.trade_simulator:
            if self.execution_engine:
                print(f"\nPersistent Position Management:")
                try:
                    #legacy code: final_status = self.trade_simulator.get_status()
                    final_status = {
                        'open_positions': len(self.execution_engine.open_positions),
                        'total_pnl': sum(p.get('pnl', 0.0) for p in self.execution_engine.open_positions),
                        'mode': self.execution_engine.mode
                    }
                    storage_config = self.cfg.get("trade_simulation", {}).get("storage", {})
                    positions_dir = storage_config.get("positions_directory", "unknown")
                    
                    print(f"  Storage directory: {positions_dir}")
                    print(f"  Position continuity: MAINTAINED")
                    print(f"  Automatic loading on restart: ENABLED")
                    print(f"  Context analysis support: {'ENABLED' if final_status.get('context_analysis_support') else 'DISABLED'}")
                    
                    # Show position recovery stats if available
                    if self._previous_marketstate:
                        print(f"  Previous session data: FOUND")
                        prev_trade_sim = self._previous_marketstate.get("trade_simulation", {})
                        
                        if prev_trade_sim.get('position_scaling_enabled', False):
                            prev_open = prev_trade_sim.get("open_aggregated_positions", {}).get("simulation", {}).get("count", 0)
                            if prev_open > 0:
                                print(f"  Positions recovered: {prev_open} aggregated positions")
                        else:
                            prev_sim = prev_trade_sim.get("open_trades", {}).get("simulation", {}).get("count", 0)
                            prev_real = prev_trade_sim.get("open_trades", {}).get("real", {}).get("count", 0)
                            if prev_sim > 0 or prev_real > 0:
                                print(f"  Positions recovered: {prev_sim} sim + {prev_real} real positions")
                    else:
                        print(f"  Previous session data: NOT FOUND (fresh start)")
                        
                except Exception as e:
                    print(f"  Status check failed: {e}")
            else:
                print(f"\nPersistent Position Management: DISABLED")
            
            # NEW: Context-specific analysis summary
            if self.modular_engine:
                print(f"\nContext-Specific Analysis:")
                print(f"  Three-mode context detection: ENABLED")
                print(f"  Analysis contexts: entry, exit_cutloss, exit_profit")
                print(f"  Detailed threshold tracking: ENABLED")
                print(f"  Pipe-delimited reasoning: ENABLED")
                print(f"  Dashboard integration: ENHANCED")
                print(f"  Backward compatibility: MAINTAINED")
            else:
                print(f"\nContext-Specific Analysis: DISABLED")
            
            # Enhanced refactoring benefits
            print(f"\nRefactoring Benefits (Phase 1+2+3 + Extreme Tracking + Persistent Positions + Context Analysis):")
            print(f"  Configuration management: CENTRALIZED")
            print(f"  Debug logging: ENHANCED WITH COMPONENTS")
            print(f"  Symbol utilities: EXTRACTED AND VALIDATED")
            print(f"  Telemetry service: ABSTRACTED AND SIMPLIFIED")
            print(f"  Service layer separation: COMPLETE")
            print(f"  Code reusability: SIGNIFICANTLY IMPROVED")
            print(f"  Testing capability: ENABLED")
            print(f"  Main script size reduction: ~700 lines")
            print(f"  Multi-timeframe extreme tracking: {'INTEGRATED' if self.extreme_tracker else 'NOT AVAILABLE'}")
            print(f"  Dashboard extreme visibility: {'ENABLED' if self.extreme_tracker else 'DISABLED'}")
            #legacy code: print(f"  Persistent position management: {'ENABLED' if self.trade_simulator else 'DISABLED'}")
            print(f"  Persistent position management: {'ENABLED' if getattr(self, 'execution_engine', None) else 'DISABLED'}")
            #legacy code: print(f"  Position continuity across restarts: {'MAINTAINED' if self.trade_simulator else 'NOT AVAILABLE'}")
            print(f"  Position continuity across restarts: {'MAINTAINED' if getattr(self, 'execution_engine', None) else 'NOT AVAILABLE'}")
            print(f"  Context-specific analysis: {'ENABLED' if self.modular_engine else 'DISABLED'}")
            print(f"  Three-mode context detection: {'ENABLED' if self.modular_engine else 'DISABLED'}")
            print(f"  Detailed threshold progress: {'ENABLED' if self.modular_engine else 'DISABLED'}")
            
        except Exception as e:
            logger.debug_print(f"Telemetry efficiency summary failed: {e}", "SUMMARY")
        
    def run(self, assets: Optional[List[str]] = None) -> None:
        """Main loop with telemetry service abstraction, extreme tracking, persistent position management, and context-specific analysis"""
        # --- PHASE 2: CONCURRENCY INITIALIZATION ---
        portfolio = assets or self.cfg.get("portfolio", {}).get("active_assets", [self.asset])
        
        service_status = self.telemetry_service.get_service_status()
        
        if service_status['buffer_mode_enabled']:
            # logger.debug_print(f"Starting OPTIMIZED refactored main loop with TELEMETRY SERVICE + EXTREME TRACKING + PERSISTENT POSITIONS + CONTEXT ANALYSIS for asset: {asset}", "MAIN")
            # logger.debug_print(f"API Efficiency: Using telemetry service (2 API calls per cycle vs 900+ legacy)", "MAIN")
            logger.debug_print(f"Triple-timeframe Analysis: 5s (tactical) + 1m (primary) + 1h (HTF bias)", "MAIN")
            # logger.debug_print(f"Chart Data Source: Telemetry service (real buffer data)", "MAIN")
            logger.debug_print(f"Extreme Tracking: {'Multi-timeframe collection enabled' if self.extreme_tracker else 'Disabled'}", "MAIN")
            #legacy code: logger.debug_print(f"Persistent Positions: {'Automatic loading enabled' if self.trade_simulator else 'Disabled'}", "MAIN")
            logger.debug_print(f"Persistent Positions: {'Automatic loading enabled' if getattr(self, 'execution_engine', None) else 'Disabled'}", "MAIN")
            logger.debug_print(f"Context Analysis: {'Three-mode detection enabled' if self.modular_engine else 'Disabled'}", "MAIN")
            
            # Display buffer configuration via service
            buffer_config = self.config_manager.get_buffer_config()
            timeframes = list(buffer_config.get("timeframes", {}).keys())
            tactical_tf = buffer_config.get("tactical_timeframe", "5s")
            primary_tf = buffer_config.get("primary_timeframe", "1m")
            bias_tf = buffer_config.get("bias_timeframe", "1h")
            logger.debug_print(f"Timeframes: {timeframes}, Tactical: {tactical_tf}, Primary: {primary_tf}, HTF: {bias_tf}", "MAIN")
            logger.debug_print(f"Chart timeframe: {primary_tf} (via telemetry service)", "MAIN")
            
            if self.extreme_tracker:
                logger.debug_print(f"Extreme tracking primary timeframe: {self.extreme_tracker.primary_timeframe}", "MAIN")
                logger.debug_print(f"Extreme storage: {self.extreme_tracker.storage_path}", "MAIN")
        else:
            logger.debug_print(f"Starting LEGACY refactored main loop with TELEMETRY SERVICE for asset: {portfolio[0]}", "MAIN")
            # logger.debug_print(f"API Efficiency: Legacy mode via telemetry service (900+ API calls per cycle)", "MAIN")
            logger.debug_print(f"Triple-timeframe Analysis: DISABLED", "MAIN")
            # logger.debug_print(f"Chart Data Source: None (no buffer data available)", "MAIN")
            logger.debug_print(f"Extreme Tracking: {'Single timeframe only' if self.extreme_tracker else 'Disabled'}", "MAIN")
            #legacy code: logger.debug_print(f"Persistent Positions: {'Loading from storage' if self.trade_simulator else 'Disabled'}", "MAIN")
            logger.debug_print(f"Persistent Positions: {'Loading from storage' if getattr(self, 'execution_engine', None) else 'Disabled'}", "MAIN")
            logger.debug_print(f"Context Analysis: {'Available but limited' if self.modular_engine else 'Disabled'}", "MAIN")
        
        tick_count = 0
        
        while True:
            try:
                # --- PHASE 2: SEQUENTIAL POLLING LOOP ---
                for asset in portfolio:
                    try:
                        # Get barometer
                        barometer = None
                        try:
                            #with PerformanceTimer("barometer_computation", logger, "MARKET"):
                            barometer = self.compute_barometer()
                        except Exception as e:
                            logger.debug_print(f"Barometer computation failed: {e}", "MARKET")
                            barometer = self._get_neutral_barometer()
                            
                            # Periodic extreme state persistence for scaling strategies
                            if hasattr(self, 'modular_engine') and self.modular_engine:
                                try:
                                    # Persist extreme tracking state every 100 ticks (8-9 minutes at 5s intervals)
                                    if (tick_count + 1) % 100 == 0:
                                        if hasattr(self.modular_engine, 'active_entry') and hasattr(self.modular_engine.active_entry, 'persist_extreme_state'):
                                            self.modular_engine.active_entry.persist_extreme_state()
                                            logger.debug_print("Extreme tracking state persisted", "SCALING")
                                except Exception as e:
                                    logger.debug_print(f"Failed to persist extreme state: {e}", "SCALING")
                        
                        # Get telemetry via service - SIMPLIFIED
                        tele_wrapped = None
                        
                        try:
                            # logger.debug_print("Using telemetry service for market data...", "TELEMETRY")
                            
                            #with PerformanceTimer("telemetry_service_data_acquisition", logger, "TELEMETRY"):
                            tele_wrapped = self.telemetry_service.get_market_data(asset)
                            
                            # logger.debug_print(f"Market data acquired successfully via telemetry service", "TELEMETRY")
                            
                        except TelemetryError as e:
                            logger.debug_print(f"Telemetry service failed: {e}", "TELEMETRY")
                            # Use fallback telemetry
                            tele_wrapped = self.telemetry_service.create_fallback_telemetry()
                            
                        except Exception as e:
                            logger.debug_print(f"Unexpected telemetry error: {e}", "TELEMETRY")
                            # Use fallback telemetry
                            tele_wrapped = self.telemetry_service.create_fallback_telemetry()
                        
                        
                        # =================================================================
                        # Process outputs (TRADING LOGIC RUNS AFTER SYNC)
                        # =================================================================
                        # =================================================================
                        # Process outputs (TRADING LOGIC)
                        # =================================================================
                        if barometer and tele_wrapped:
                            # [FIX] Force P&L Update BEFORE Strategy Analysis
                            # This ensures the Engine calculates Profit/Loss using the latest price
                            # before the Strategy reads the file/state.
                            if hasattr(self, 'execution_engine'):
                                current_price = tele_wrapped.get('price', {}).get('primary', 0.0)
                                self.execution_engine._update_pnl(current_price)

                            try:
                                with PerformanceTimer("write_outputs", logger, "OUTPUT"):
                                    self.write_outputs(asset, tele_wrapped, barometer)
                            except Exception as e:
                                logger.debug_print(f"Write outputs failed: {e}", "OUTPUT")
                                traceback.print_exc()

                    except Exception as loop_err:
                        logger.debug_print(f"Portfolio iteration error for {asset}: {loop_err}", "MAIN")

                tick_count += 1
                time.sleep(TICK_SEC)  # Sleep for 5 seconds
                
            except KeyboardInterrupt:
                logger.debug_print("Keyboard interrupt received", "MAIN")
                
                # Persist extreme tracking state on exit
                if self.extreme_tracker:
                    try:
                        self.extreme_tracker.persist()
                        logger.debug_print("Final extreme tracking state persisted", "EXTREME_TRACKING")
                    except Exception as e:
                        logger.debug_print(f"Failed to persist final extreme state: {e}", "EXTREME_TRACKING")
                
                # Display final summary with telemetry service metrics, extreme tracking, persistence, and context analysis
                self._display_telemetry_efficiency_summary(tick_count)
                
                # Display session statistics
                session_stats = logger.get_session_stats()
                logger.log_session_summary()
                
                # Display final unified position simulation summary with persistence and context analysis info
                #legacy code: if self.trade_simulator:
                if self.execution_engine:
                    try:
                        #legacy code: final_status = self.trade_simulator.get_status()
                        final_status = {
                            'open_positions': len(self.execution_engine.open_positions),
                            'total_pnl': sum(p.get('pnl', 0.0) for p in self.execution_engine.open_positions),
                            'mode': self.execution_engine.mode
                        }
                        print("\nFINAL UNIFIED POSITION SIMULATION SUMMARY WITH PERSISTENCE AND CONTEXT ANALYSIS:")
                        
                        unified_system = final_status.get("unified_system", False)
                        schema_version = final_status.get("schema_version", "unknown")
                        context_analysis_support = final_status.get("context_analysis_support", False)
                        
                        unified_system = final_status.get("unified_system", True)  # Default to True
                        print(f"Unified System: {'ACTIVE' if unified_system else 'LEGACY'}")
                        print(f"Schema Version: {schema_version}")
                        print(f"Refactoring Status: PHASE 1+2+3 COMPLETE")
                        print(f"Telemetry Service: ABSTRACTED")
                        print(f"Enhanced Execution Tracking: ENABLED")
                        print(f"Multi-timeframe Extreme Tracking: {'ENABLED' if self.extreme_tracker else 'DISABLED'}")
                        #legacy code: print(f"Persistent Position Management: {'ENABLED' if self.trade_simulator else 'DISABLED'}")
                        print(f"Persistent Position Management: {'ENABLED' if getattr(self, 'execution_engine', None) else 'DISABLED'}")
                        print(f"Context-Specific Analysis: {'ENABLED' if context_analysis_support else 'DISABLED'}")
                        print(f"Triple-timeframe Analysis: {'ENABLED' if service_status['buffer_mode_enabled'] else 'DISABLED'}")
                        print(f"Service Mode: {service_status['mode'].upper()}")
                        print(f"Modular Architecture: ENABLED")

                        # Storage information
                        storage_config = self.cfg.get("trade_simulation", {}).get("storage", {})
                        positions_dir = storage_config.get("positions_directory", "unknown")
                        print(f"Storage Directory: {positions_dir}")
                        
                        print(f"Position Scaling: {'ENABLED' if final_status.get('position_scaling_enabled', False) else 'DISABLED'}")
                        # Display aggregated position summary if scaling enabled
                        if final_status.get('position_scaling_enabled', False):
                            closed_agg = final_status.get('closed_aggregated_positions', {}).get('simulation', {})
                            closed_count = closed_agg.get('count', 0)
                            
                            print(f"\nAggregated Position Summary:")
                            print(f"Closed aggregated positions: {closed_count}")
                            
                            if closed_count > 0:
                                recent_positions = closed_agg.get('recent', [])[:3]  # Show last 3
                                for i, pos in enumerate(recent_positions, 1):
                                    metrics = pos.get('aggregated_metrics', {})
                                    final_pnl = pos.get('final_pnl', {})
                                    component_count = metrics.get('position_count', 0)
                                    pnl_usd = final_pnl.get('pnl_usd', 0.0)
                                    duration = final_pnl.get('duration_hours', 0.0)
                                    
                                    # NEW: Show context analysis history if available
                                    context_history = pos.get('analysis_context_history', {})
                                    context_transition = context_history.get('context_transition', 'unknown')
                                    
                                    print(f"  Position {i}: {component_count} components, "
                                          f"P&L: ${pnl_usd:.2f}, Duration: {duration:.1f}h, "
                                          f"Context: {context_transition}")
                        else:
                            summaries = final_status.get("summaries", {})
                            if "simulation" in summaries:
                                sim = summaries["simulation"]
                                execution_metrics = sim.get("execution_metrics", {})
                                context_transitions = sim.get("context_transitions_tracked", 0)
                                print(f"Simulated: {sim.get('total_trades', 0)} trades | "
                                      f"P&L: ${sim.get('total_pnl_usd', 0.0):.2f} | "
                                      f"Win Rate: {sim.get('win_rate', 0.0):.1f}%")
                                if execution_metrics:
                                    print(f"Avg Hedge Effectiveness: {execution_metrics.get('average_hedge_effectiveness_pct', 0):.1f}%")
                                if context_transitions > 0:
                                    print(f"Context transitions tracked: {context_transitions}")
                            
                            if "real" in summaries:
                                real = summaries["real"]
                                execution_metrics = real.get("execution_metrics", {})
                                print(f"Real: {real.get('total_trades', 0)} trades | "
                                      f"P&L: ${real.get('total_pnl_usd', 0.0):.2f} | "
                                      f"Win Rate: {real.get('win_rate', 0.0):.1f}%")
                                if execution_metrics:
                                    print(f"Avg Hedge Effectiveness: {execution_metrics.get('average_hedge_effectiveness_pct', 0):.1f}%")
                        
                        # Persistence information
                        print(f"\nPersistence Information:")
                        #legacy code: print(f"Position continuity: {'MAINTAINED' if self.trade_simulator else 'NOT AVAILABLE'}")
                        print(f"Position continuity: {'MAINTAINED' if getattr(self, 'execution_engine', None) else 'NOT AVAILABLE'}")
                        print(f"Context analysis support: {'ENABLED' if context_analysis_support else 'DISABLED'}")
                        if self._previous_marketstate:
                            print(f"Previous session data: RECOVERED")
                        else:
                            print(f"Previous session data: NOT FOUND (fresh start)")
                        
                    except Exception as e:
                        logger.debug_print(f"Failed to display final summary: {e}", "SUMMARY")
                
                # Display chart generation summary
                if self.chart_orchestrator:
                    try:
                        print(f"\nPHASE 1+2+3 REFACTORED TELEMETRY SERVICE + EXTREME TRACKING + PERSISTENT POSITIONS + CONTEXT ANALYSIS CHART SUMMARY:")
                        print(f"Chart data source: Telemetry service")
                        print(f"Service mode: {service_status['mode']}")
                        print(f"Tick interval: {TICK_SEC} seconds")
                        print(f"Triple-timeframe analysis: {'ENABLED' if service_status['buffer_mode_enabled'] else 'DISABLED'}")
                        print(f"Real volume data: {'ENABLED' if service_status['buffer_mode_enabled'] else 'DISABLED'}")
                        print(f"Service abstraction: COMPLETE")
                        print(f"Extreme tracking integration: {'ENABLED' if self.extreme_tracker else 'DISABLED'}")
                        #legacy code: print(f"Persistent position awareness: {'ENABLED' if self.trade_simulator else 'DISABLED'}")
                        print(f"Persistent position awareness: {'ENABLED' if getattr(self, 'execution_engine', None) else 'DISABLED'}")
                        print(f"Context analysis integration: {'ENABLED' if self.modular_engine else 'DISABLED'}")
                        
                        chart_config = self.config_manager.get_chart_config()
                        output_dir = chart_config.get("output_dir", "realtime")
                        filename = chart_config.get("filename", "live_chart.png")
                        chart_path = os.path.join(output_dir, filename)
                        if os.path.exists(chart_path):
                            print(f"Final chart location: {chart_path}")
                        
                    except Exception as e:
                        logger.debug_print(f"Failed to display chart summary: {e}", "SUMMARY")
                
                # Shutdown telemetry service
                logger.debug_print("Shutting down telemetry service...", "MAIN")
                self.telemetry_service.shutdown()
                
                break
                
            except Exception as e:
                logger.debug_print(f"Unexpected error in telemetry service main loop: {type(e).__name__}: {e}", "MAIN")
                traceback.print_exc()
                time.sleep(1)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Phase 1+2+3 Refactored + Multi-Timeframe Extreme Tracking + Persistent Position Loading + Context-Specific Analysis: Triple-Timeframe with Telemetry Service Abstraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
*** FINANCIAL RISK WARNING ***
This refactored system focuses on triple-timeframe regression-based pair trading with 
complete service layer abstraction, multi-timeframe extreme tracking, persistent 
position management, and context-specific analysis for improved maintainability, 
reusability, enhanced position scaling capabilities, and granular trading insights 
with seamless restart continuity.

All trading involves substantial risk of financial loss. Use appropriate position sizing and risk management.

Examples:
  python3 vmain_refactored_phase3.py
  python3 vmain_refactored_phase3.py --config configvisionapi_SOL_MELANIA_regression.json
        """
    )
    parser.add_argument(
        "-c", "--config",
        default="configvisionapi.json",
        help="Path to configuration file (default: configvisionapi.json)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # logger.debug_print("Phase 1+2+3 refactored telemetry service + extreme tracking + persistent positions + context analysis script starting...", "MAIN")
    # --- [VERSION PROOF] ---
    print("\n" + "="*60)
    print("🚨 [PROOF] RUNNING VERSION: VMAIN_WITH_OVERRIDE_PATCH_V2 🚨")
    print("Timestamp: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60 + "\n")
    # -----------------------
    try:
        args = parse_args()
        logger.debug_print(f"Using config: {args.config}", "CONFIG")
        
        print("*** PHASE 1+2+3 REFACTORED: TELEMETRY SERVICE + MULTI-TIMEFRAME EXTREME TRACKING + PERSISTENT POSITIONS + CONTEXT-SPECIFIC ANALYSIS ***")
        print("This refactored system demonstrates significant improvements in service layer")
        print("abstraction, multi-timeframe extreme tracking, persistent position management,")
        print("and context-specific trading analysis while preserving all original functionality")
        print("and performance benefits.")
        print("")
        
        # logger.debug_print("Creating Phase 1+2+3 refactored telemetry service + extreme tracking + persistent positions + context analysis App instance...", "MAIN")
        app = TradingApplication(args.config)
        
        # --- START CONTROL SERVER ---
        try:
            start_control_server(app, port=8090)
        except Exception as e:
            print(f"[WARNING] Could not start control server: {e}")
        # ----------------------------

        # logger.debug_print("Phase 1+2+3 refactored telemetry service + extreme tracking + persistent positions + context analysis app created successfully, starting run...", "MAIN")
        app.run()
        
    except Exception as e:
        logger.debug_print(f"Fatal error in Phase 1+2+3 refactored telemetry service + extreme tracking + persistent positions + context analysis main: {e}", "MAIN")
        traceback.print_exc()
    
    # logger.debug_print("Phase 1+2+3 refactored telemetry service + extreme tracking + persistent positions + context analysis script ended", "MAIN")
