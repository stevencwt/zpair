#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_manager.py - Configuration Management System
===================================================
Centralized configuration loading, validation, and management for trading systems.
"""

import os
import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from debug_utils import DebugLogger


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass


class ConfigurationManager:
    """
    Centralized configuration management with validation and schema support
    """
    
    # Default configuration template
    DEFAULT_CONFIG = {
        "market_state_storage": {
        "mode": "FILE", 
        "file_settings": {
            "snapshot_dir": "marketstate"
            }
        },
               
        "provider": "hyperliquid",
        "trading": {
            "hyperliquid": {
                "private_key": "",
                "wallet_address": ""
            },
            "telemetry": {
                "enabled": True,
                "primary_asset": "SOL",
                "reference_asset": "MELANIA", 
                "zscore_window": 40,
                "bootstrap_bars": 50,
                "rsi_period": 14,
                "historical_lookback": 250
            }
        },
        "multi_timeframe_buffers": {
            "enabled": True,
            "initialization": {
                "retry_attempts": 3,
                "retry_delay_seconds": 5,
                "fallback_to_legacy_mode": True
            },
            "timeframes": {
                "5s": {
                    "buffer_size": 720,
                    "regression_window": 360,
                    "zscore_window": 120,
                    "correlation_window": 200,
                    "beta_smoothing_window": 60,
                    "min_periods_for_analysis": 50,
                    "update_frequency_seconds": 5
                },
                "1m": {
                    "buffer_size": 350,
                    "regression_window": 200,
                    "zscore_window": 40,
                    "correlation_window": 80,
                    "beta_smoothing_window": 30,
                    "min_periods_for_analysis": 50,
                    "update_frequency_seconds": 60
                },
                "1h": {
                    "buffer_size": 200,
                    "regression_window": 120,
                    "zscore_window": 36,
                    "correlation_window": 60,
                    "beta_smoothing_window": 24,
                    "min_periods_for_analysis": 30,
                    "update_frequency_seconds": 3600
                }
            },
            "primary_timeframe": "1m",
            "bias_timeframe": "1h",
            "tactical_timeframe": "5s",
            "coordination": {
                "hourly_bias_filtering": True,
                "conflicting_signal_resolution": "primary_timeframe_priority",
                "minimum_bias_agreement_threshold": 0.3
            },
            "api_optimization": {
                "current_price_fetch_interval_seconds": 5,
                "historical_refresh_interval_hours": 6,
                "batch_candle_requests": True
            }
        },
        "pair_trading": {
            "method": "regression",
            "regression_window": 200,
            "zscore_window": 40,
            "correlation_window": 80,
            "beta_smoothing_window": 30,
            "health_thresholds": {
                "min_correlation": 0.5,
                "max_beta_drift_pct": 20.0,
                "min_r_squared": 0.2
            },
            "bands": {
                "z_entry": 1.5,
                "z_stop": 2.5,
                "z_exit": 0.3
            }
        },
        "regime_assessment": {
            "enabled": True,
            "timeframe": "5s",
            "lookback_periods": 20,
            "trend_threshold": 0.3,
            "volatility_sync_threshold": 0.5,
            "correlation_threshold": 0.2
        },
        "trading_decision": {
            "enabled": True,
            "rules_only": True
        },
        "trade_simulation": {
            "enabled": True,
            "real_trading_enabled": False,
            "phase1_enhancements": {
                "complete_execution_tracking": True,
                "beta_analysis": True,
                "hedge_ratio_tracking": True,
                "execution_quality_metrics": True
            }
        },
        "market_barometer": {
            "enabled": True
        },
        "realtime_charts": {
            "enabled": True,
            "output_dir": "realtime",
            "filename": "live_chart.png",
            "buffer_size": 360,
            "generation_interval_seconds": 30,
            "zscore_window": 40,
            "zscore_thresholds": [1, 2, 3],
            "theme": "dark",
            "force_generation_on_signal_change": True
        },
        "snapshot_dir": "marketstate",
        "asset": "SOL/MELANIA"
    }
    
    # Configuration schema for validation
    SCHEMA_REQUIREMENTS = {            
        "required_fields": [
            "provider", "asset", "trading", "multi_timeframe_buffers", 
            "pair_trading", "realtime_charts"
        ],
        "numeric_ranges": {
            "multi_timeframe_buffers.timeframes.1m.zscore_window": (10, 200),
            "multi_timeframe_buffers.timeframes.1h.zscore_window": (10, 100),
            "pair_trading.bands.z_entry": (0.5, 5.0),
            "pair_trading.bands.z_exit": (0.1, 2.0),
            "realtime_charts.generation_interval_seconds": (5, 300)
        },
        "enum_values": {
            "market_state_storage.mode": ["FILE", "MEMORY"],  
            "provider": ["hyperliquid", "binance", "coinbase"],
            "multi_timeframe_buffers.primary_timeframe": ["5s", "1m", "1h"],
            "realtime_charts.theme": ["light", "dark"]
        }
    }
    
    def __init__(self, config_path: str = "configvisionapi.json", logger: Optional[DebugLogger] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file
            logger: Debug logger instance
        """
        self.config_path = Path(config_path)
        self.logger = logger or DebugLogger(True, "CONFIG")
        self._config = None
        self._config_loaded_at = None
        self._validation_errors = []
        self._validation_warnings = []
        
        self.logger.debug_print(f"ConfigurationManager initialized with path: {config_path}")
    
    def load_config(self, validate: bool = True) -> Dict[str, Any]:
        """
        Load and optionally validate configuration
        
        Args:
            validate: Whether to perform validation
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ConfigValidationError: If validation fails and validate=True
        """
        self.logger.debug_print(f"Loading configuration from: {self.config_path}")
        
        # Start with default configuration
        config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # Load user configuration if file exists
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)
                
                self.logger.debug_print(f"User config loaded with keys: {list(user_config.keys())}")
                
                # Deep merge configurations
                config = self._deep_merge_configs(config, user_config)
                
            except Exception as e:
                self.logger.debug_print(f"Error loading config file: {e}")
                raise ConfigValidationError(f"Failed to load config file: {e}")
        else:
            self.logger.debug_print(f"Config file not found: {self.config_path}, using defaults")
        
        # Validate configuration if requested
        if validate:
            self._validate_config(config)
            
            if self._validation_errors:
                error_msg = f"Configuration validation failed: {'; '.join(self._validation_errors)}"
                self.logger.debug_print(error_msg)
                raise ConfigValidationError(error_msg)
            
            if self._validation_warnings:
                for warning in self._validation_warnings:
                    self.logger.debug_print(f"Config warning: {warning}")
        
        # Store loaded configuration
        self._config = config
        self._config_loaded_at = datetime.now()
        
        self.logger.debug_print("Configuration loaded successfully")
        return config
    
    def _deep_merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if (
                key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)
            ):
                result[key] = self._deep_merge_configs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration against schema
        
        Args:
            config: Configuration to validate
        """
        self._validation_errors = []
        self._validation_warnings = []
        
        # Check required fields
        self._validate_required_fields(config)
        
        # Check numeric ranges
        self._validate_numeric_ranges(config)
        
        # Check enum values
        self._validate_enum_values(config)
        
        # Custom validation logic
        self._validate_custom_rules(config)
    
    def _validate_required_fields(self, config: Dict[str, Any]) -> None:
        """Validate required fields are present"""
        for field_path in self.SCHEMA_REQUIREMENTS["required_fields"]:
            if not self._get_nested_value(config, field_path):
                self._validation_errors.append(f"Required field missing: {field_path}")
    
    def _validate_numeric_ranges(self, config: Dict[str, Any]) -> None:
        """Validate numeric values are within acceptable ranges"""
        for field_path, (min_val, max_val) in self.SCHEMA_REQUIREMENTS["numeric_ranges"].items():
            value = self._get_nested_value(config, field_path)
            if value is not None:
                try:
                    num_value = float(value)
                    if not (min_val <= num_value <= max_val):
                        self._validation_errors.append(
                            f"Value {field_path}={num_value} not in range [{min_val}, {max_val}]"
                        )
                except (ValueError, TypeError):
                    self._validation_errors.append(f"Field {field_path} must be numeric")
    
    def _validate_enum_values(self, config: Dict[str, Any]) -> None:
        """Validate enum fields have acceptable values"""
        for field_path, valid_values in self.SCHEMA_REQUIREMENTS["enum_values"].items():
            value = self._get_nested_value(config, field_path)
            if value is not None and value not in valid_values:
                self._validation_errors.append(
                    f"Field {field_path}='{value}' not in valid values: {valid_values}"
                )
    
    def _validate_custom_rules(self, config: Dict[str, Any]) -> None:
        """Apply custom validation rules"""
        # Validate asset format
        asset = config.get("asset", "")
        if asset and "/" not in asset:
            self._validation_warnings.append(f"Asset '{asset}' should be in format 'BASE/QUOTE'")
        
        # Validate timeframe consistency
        buffer_config = config.get("multi_timeframe_buffers", {})
        if buffer_config.get("enabled", False):
            timeframes = buffer_config.get("timeframes", {})
            primary_tf = buffer_config.get("primary_timeframe", "")
            
            if primary_tf and primary_tf not in timeframes:
                self._validation_errors.append(
                    f"Primary timeframe '{primary_tf}' not configured in timeframes"
                )
        
        # Validate chart configuration
        chart_config = config.get("realtime_charts", {})
        if chart_config.get("enabled", False):
            output_dir = chart_config.get("output_dir", "")
            if not output_dir:
                self._validation_warnings.append("Chart output directory not specified")
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """
        Get nested value from configuration using dot notation
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., "multi_timeframe_buffers.enabled")
            
        Returns:
            Value at path or None if not found
        """
        keys = path.split(".")
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def get_config(self) -> Dict[str, Any]:
        """Get loaded configuration (load if not already loaded)"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def get_buffer_config(self) -> Dict[str, Any]:
        """Get buffer-specific configuration"""
        config = self.get_config()
        return config.get("multi_timeframe_buffers", {})
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-specific configuration"""
        config = self.get_config()
        return config.get("trading", {})
    
    def get_chart_config(self) -> Dict[str, Any]:
        """Get chart-specific configuration"""
        config = self.get_config()
        return config.get("realtime_charts", {})
    
    def get_asset_info(self) -> Dict[str, str]:
        """Get asset information from configuration"""
        config = self.get_config()
        asset = config.get("asset", "")
        
        # Parse asset into base and quote
        if "/" in asset:
            base, quote = asset.split("/", 1)
        else:
            base, quote = asset, ""
        
        return {
            "asset": asset,
            "base": base.strip(),
            "quote": quote.strip()
        }
    
    def update_config(self, updates: Dict[str, Any], save: bool = False) -> None:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of updates to apply
            save: Whether to save to file
        """
        config = self.get_config()
        updated_config = self._deep_merge_configs(config, updates)
        
        # Validate updated configuration
        self._validate_config(updated_config)
        
        if self._validation_errors:
            error_msg = f"Updated configuration invalid: {'; '.join(self._validation_errors)}"
            raise ConfigValidationError(error_msg)
        
        self._config = updated_config
        
        if save:
            self.save_config()
        
        self.logger.debug_print("Configuration updated successfully")
    
    def save_config(self, backup: bool = True) -> None:
        """
        Save current configuration to file
        
        Args:
            backup: Whether to create backup of existing file
        """
        if self._config is None:
            raise ValueError("No configuration loaded to save")
        
        # Create backup if requested
        if backup and self.config_path.exists():
            backup_path = self.config_path.with_suffix(f".backup.{int(datetime.now().timestamp())}")
            backup_path.write_text(self.config_path.read_text(), encoding="utf-8")
            self.logger.debug_print(f"Created config backup: {backup_path}")
        
        # Save configuration
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
        
        self.logger.debug_print(f"Configuration saved to: {self.config_path}")
    
    def get_validation_status(self) -> Dict[str, Any]:
        """Get validation status and messages"""
        return {
            "has_errors": bool(self._validation_errors),
            "has_warnings": bool(self._validation_warnings),
            "errors": self._validation_errors.copy(),
            "warnings": self._validation_warnings.copy(),
            "config_loaded_at": self._config_loaded_at.isoformat() if self._config_loaded_at else None
        }
    
    def create_config_template(self, path: Optional[str] = None) -> str:
        """
        Create configuration template file
        
        Args:
            path: Path for template file (default: config_template.json)
            
        Returns:
            Path to created template file
        """
        template_path = Path(path) if path else self.config_path.parent / "config_template.json"
        
        template = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # Add comments to template
        template["_comments"] = {
            "provider": "Exchange provider (hyperliquid, binance, coinbase)",
            "asset": "Trading pair in format BASE/QUOTE",
            "multi_timeframe_buffers": "Configuration for triple-timeframe analysis",
            "realtime_charts": "Real-time chart generation settings"
        }
        
        with open(template_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        self.logger.debug_print(f"Configuration template created: {template_path}")
        return str(template_path)


def load_config(config_path: str = "configvisionapi.json") -> Dict[str, Any]:
    """
    Legacy function for backward compatibility
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    manager = ConfigurationManager(config_path)
    return manager.load_config()
