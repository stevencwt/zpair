#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Mode Validator - Mode Detection and Validation
======================================================
Ensures trading mode is properly configured and supported.
Provides graceful exit for unimplemented modes with helpful error messages.

Part of Phase 1: Multi-Mode Trading System Enhancement
Zero Deletion Compliance: New module, no existing code modified

REVISION NOTES:
- Fixed pairs_trading asset count validation (expects 1 pair string, not 2 assets)
- Improved mode name normalization (strip + lowercase)
- Standardized on "PLANNED" status (removed "PLACEHOLDER" confusion)
"""

import sys
from typing import Dict, Any, Optional


class TradingModeValidator:
    """
    Validates trading mode configuration and enforces mode requirements.
    Provides graceful exit for unimplemented modes.
    """
    
    # Supported trading modes registry
    SUPPORTED_MODES = {
        "pairs_trading": {
            "status": "IMPLEMENTED",
            "required_assets": 1,  # FIXED: 1 pair string like "ETH/BTC", not 2 separate assets
            "description": "Regression-based pair trading with alpha/beta hedging",
            "data_requirements": ["regression", "zscore", "correlation", "beta"],
            "strategy_types": ["entry", "take_profit", "cut_loss"]
        },
        "single_asset": {
            "status": "PLANNED",
            "required_assets": 1,
            "description": "Hurst/RSI directional trading (trending + ranging detection)",
            "data_requirements": ["price", "hurst", "rsi", "trend", "volatility"],
            "strategy_types": ["entry", "take_profit", "cut_loss"],
            "exit_message": "Single asset mode is planned but not yet implemented. The system will support Hurst exponent regime detection and RSI-based directional trading. Please use 'pairs_trading' mode for now."
        },
        "dual_hybrid": {
            "status": "PLANNED",
            "required_assets": 2,  # Two separate assets: ["ETH", "BTC"]
            "description": "Sequential alternating direction strategy with pair intelligence",
            "data_requirements": ["regression", "zscore", "hurst", "rsi", "trend"],
            "strategy_types": ["entry", "take_profit", "cut_loss"],
            "exit_message": "Dual hybrid mode is planned but not yet implemented. This mode will combine pair regression signals with single-asset regime analysis for intelligent sequential position building. Please use 'pairs_trading' mode for now."
        },
        "dual_single": {
            "status": "PLANNED",
            "required_assets": 2,
            "description": "Two independent single-asset strategies running in parallel",
            "exit_message": "Dual single-asset mode not yet implemented. Please use 'single_asset' mode for individual assets or 'pairs_trading' for correlated pairs."
        },
        "mixed_portfolio": {
            "status": "PLANNED",
            "required_assets": "2+",
            "description": "Mixed portfolio with both pairs and single assets",
            "exit_message": "Mixed portfolio mode not yet implemented. Please use 'pairs_trading' or 'single_asset' mode."
        },
        "dynamic_duo": {
            "status": "PLANNED",
            "required_assets": 2,
            "description": "Lead-lag correlation trading without regression hedging",
            "exit_message": "Dynamic duo mode not yet implemented. Please use 'pairs_trading' mode."
        },
        "multi_agent": {
            "status": "PLANNED",
            "required_assets": "3+",
            "description": "Multi-asset coordination strategy (3+ assets)",
            "exit_message": "Multi-agent mode not yet implemented. Please use 'pairs_trading' or 'single_asset' mode."
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator with configuration
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.portfolio = config.get("portfolio", {})
        
        # IMPROVED: Normalize mode name (strip whitespace, lowercase, handle hyphens)
        raw_mode = self.portfolio.get("trading_mode", "")
        self.mode_name = raw_mode.strip().lower().replace("-", "_")
        
        self.active_assets = self.portfolio.get("active_assets", [])
    
    def validate_and_get_mode(self) -> Dict[str, Any]:
        """
        Validate trading mode and return mode metadata.
        Exits gracefully if mode is not implemented.
        
        Returns:
            Dict with mode metadata if valid
            
        Raises:
            SystemExit: If mode is invalid or not implemented
        """
        # Step 1: Check if trading_mode key exists
        if not self.mode_name:
            self._exit_with_error(
                "CONFIGURATION ERROR: 'portfolio.trading_mode' not specified in config",
                self._get_available_modes_message(),
                exit_code=10
            )
        
        # Step 2: Check if mode is recognized
        if self.mode_name not in self.SUPPORTED_MODES:
            self._exit_with_error(
                f"CONFIGURATION ERROR: Unknown trading mode '{self.mode_name}'",
                self._get_available_modes_message(),
                exit_code=11
            )
        
        mode_info = self.SUPPORTED_MODES[self.mode_name]
        
        # Step 3: Check if mode is implemented
        if mode_info["status"] != "IMPLEMENTED":
            self._exit_with_placeholder_message(self.mode_name, mode_info)
        
        # Step 4: Validate asset count
        self._validate_asset_count(mode_info)
        
        # Step 5: Validate mode-specific requirements
        self._validate_mode_requirements(mode_info)
        
        # Return mode metadata
        return {
            "name": self.mode_name,
            "status": mode_info["status"],
            "description": mode_info["description"],
            "required_assets": mode_info["required_assets"],
            "data_requirements": mode_info.get("data_requirements", []),
            "active_assets": self.active_assets,
            "asset_count": len(self.active_assets)
        }
    
    def _validate_asset_count(self, mode_info: Dict[str, Any]) -> None:
        """
        Validate that active_assets count matches mode requirements
        
        CRITICAL FIX: pairs_trading expects 1 pair string like "ETH/BTC",
        not 2 separate assets like ["ETH", "BTC"]
        
        Args:
            mode_info: Mode metadata from SUPPORTED_MODES
            
        Raises:
            SystemExit: If asset count doesn't match requirements
        """
        required = mode_info["required_assets"]
        actual = len(self.active_assets)
        
        # SPECIAL CASE: pairs_trading expects 1 pair string
        if self.mode_name == "pairs_trading":
            if actual != 1:
                self._exit_with_error(
                    f"ASSET COUNT ERROR: Pairs trading expects exactly 1 pair string",
                    f"Your config has {actual} item(s): {self.active_assets}\n\n"
                    f"Correct format:\n"
                    f'{{\n  "portfolio": {{\n    "active_assets": ["ETH/BTC"]\n  }}\n}}\n\n'
                    f"NOT this:\n"
                    f'{{\n  "portfolio": {{\n    "active_assets": ["ETH", "BTC"]  // WRONG - use "ETH/BTC"\n  }}\n}}',
                    exit_code=12
                )
            return  # Validation passed for pairs_trading
        
        # Handle numeric requirements (for other modes)
        if isinstance(required, int):
            if actual != required:
                self._exit_with_error(
                    f"ASSET COUNT MISMATCH: Mode '{self.mode_name}' requires exactly {required} asset(s)",
                    f"Your config specifies {actual} asset(s): {self.active_assets}\n"
                    f"Please adjust 'portfolio.active_assets' in your config file.",
                    exit_code=12
                )
        
        # Handle string requirements like "2+" or "3+"
        elif isinstance(required, str) and "+" in required:
            min_required = int(required.replace("+", ""))
            if actual < min_required:
                self._exit_with_error(
                    f"ASSET COUNT MISMATCH: Mode '{self.mode_name}' requires at least {min_required} assets",
                    f"Your config specifies {actual} asset(s): {self.active_assets}\n"
                    f"Please adjust 'portfolio.active_assets' in your config file.",
                    exit_code=12
                )
    
    def _validate_mode_requirements(self, mode_info: Dict[str, Any]) -> None:
        """
        Validate mode-specific configuration requirements
        
        Args:
            mode_info: Mode metadata from SUPPORTED_MODES
            
        Raises:
            SystemExit: If mode-specific validation fails
        """
        # For pairs_trading mode
        if self.mode_name == "pairs_trading":
            # Check that we have pair_trading config
            if "pair_trading" not in self.config:
                print("[WARNING] 'pair_trading' config section missing. Using defaults.")
            
            # Validate asset format (should be "ASSET1/ASSET2")
            if len(self.active_assets) > 0:
                asset = self.active_assets[0]
                if "/" not in asset:
                    self._exit_with_error(
                        f"CONFIGURATION ERROR: Pairs trading requires asset format 'PRIMARY/REFERENCE'",
                        f"Your asset '{asset}' is not in pair format.\n"
                        f"Example: 'ETH/BTC', 'SOL/MELANIA'\n\n"
                        f"Please update your config:\n"
                        f'{{\n  "portfolio": {{\n    "active_assets": ["ETH/BTC"]\n  }}\n}}',
                        exit_code=13
                    )
        
        # For single_asset mode (when implemented)
        elif self.mode_name == "single_asset":
            # Check that we have single_asset_config
            if "single_asset_config" not in self.config:
                print("[WARNING] 'single_asset_config' section missing. Using defaults.")
            
            # Validate asset format (should be single asset, no "/")
            if len(self.active_assets) > 0:
                asset = self.active_assets[0]
                if "/" in asset:
                    self._exit_with_error(
                        f"CONFIGURATION ERROR: Single asset mode requires single asset symbol",
                        f"Your asset '{asset}' appears to be a pair (contains '/').\n"
                        f"For single asset mode, use asset symbols like 'BTC', 'ETH', 'SOL'\n"
                        f"If you want to trade pairs, set trading_mode to 'pairs_trading'",
                        exit_code=13
                    )
        
        # For dual_hybrid mode (when implemented)
        elif self.mode_name == "dual_hybrid":
            # Check that we have dual_hybrid_config
            if "dual_hybrid_config" not in self.config:
                print("[WARNING] 'dual_hybrid_config' section missing. Using defaults.")
            
            # Validate we have exactly 2 assets (separate, not pair format)
            if len(self.active_assets) != 2:
                self._exit_with_error(
                    f"CONFIGURATION ERROR: Dual hybrid mode requires exactly 2 separate assets",
                    f"Your config specifies {len(self.active_assets)} asset(s): {self.active_assets}\n\n"
                    f"Correct format:\n"
                    f'{{\n  "portfolio": {{\n    "active_assets": ["ETH", "BTC"]\n  }}\n}}\n\n'
                    f"NOT pair format:\n"
                    f'{{\n  "portfolio": {{\n    "active_assets": ["ETH/BTC"]  // WRONG for dual_hybrid\n  }}\n}}',
                    exit_code=13
                )
            
            # FUTURE: Validate asset_roles when dual_hybrid is implemented
            # For now, just warn if missing
            if "asset_roles" not in self.portfolio:
                print("[WARNING] Recommended: Specify 'portfolio.asset_roles' for dual_hybrid mode")
                print("[WARNING] Example: {\"primary\": \"ETH\", \"reference\": \"BTC\"}")
    
    def _get_available_modes_message(self) -> str:
        """
        Generate helpful message showing available modes
        
        Returns:
            Formatted string with available modes
        """
        message = "\nAVAILABLE TRADING MODES:\n" + "="*70 + "\n"
        
        for mode_name, info in self.SUPPORTED_MODES.items():
            status_icon = "✅" if info["status"] == "IMPLEMENTED" else "🚧"
            message += f"\n{status_icon} '{mode_name}' ({info['status']})\n"
            message += f"   Description: {info['description']}\n"
            message += f"   Required assets: {info['required_assets']}\n"
        
        message += "\n" + "="*70 + "\n"
        message += "Configuration examples:\n\n"
        message += "Pairs Trading:\n"
        message += '{\n  "portfolio": {\n    "trading_mode": "pairs_trading",\n'
        message += '    "active_assets": ["ETH/BTC"]  // One pair string\n  }\n}\n\n'
        message += "Single Asset (when implemented):\n"
        message += '{\n  "portfolio": {\n    "trading_mode": "single_asset",\n'
        message += '    "active_assets": ["BTC"]  // One asset symbol\n  }\n}\n\n'
        message += "Dual Hybrid (when implemented):\n"
        message += '{\n  "portfolio": {\n    "trading_mode": "dual_hybrid",\n'
        message += '    "active_assets": ["ETH", "BTC"]  // Two separate assets\n  }\n}\n'
        
        return message
    
    def _exit_with_placeholder_message(self, mode_name: str, mode_info: Dict[str, Any]) -> None:
        """
        Exit gracefully for planned modes with helpful message
        
        Args:
            mode_name: Name of the trading mode
            mode_info: Mode metadata from SUPPORTED_MODES
            
        Raises:
            SystemExit: Always exits with code 20
        """
        print("\n" + "="*70)
        print("🚧 TRADING MODE NOT YET IMPLEMENTED 🚧")
        print("="*70)
        print(f"\nRequested Mode: '{mode_name}'")
        print(f"Status: {mode_info['status']}")
        print(f"Description: {mode_info['description']}")
        print("\n" + "-"*70)
        print(mode_info.get("exit_message", "This mode is not yet implemented."))
        print("-"*70)
        print(self._get_available_modes_message())
        sys.exit(20)
    
    def _exit_with_error(self, error_message: str, details: str = "", exit_code: int = 1) -> None:
        """
        Exit with error message
        
        Args:
            error_message: Main error message
            details: Additional details or suggestions
            exit_code: System exit code (10=missing mode, 11=unknown, 12=count, 13=format)
            
        Raises:
            SystemExit: Always exits with specified code
        """
        print("\n" + "="*70)
        print("❌ CONFIGURATION ERROR ❌")
        print("="*70)
        print(f"\n{error_message}\n")
        if details:
            print(details)
        print("\n" + "="*70 + "\n")
        sys.exit(exit_code)