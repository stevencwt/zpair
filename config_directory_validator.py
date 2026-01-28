#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config Directory Validator - Unified Storage Version
----------------------------------------------------
Validates and corrects output directory naming conventions in config files.
Updated for unified position storage structure (Phase 2 Complete).

Expected format:
- marketstate_ASSET1_ASSET2_HTFtimeLTFtime (e.g. marketstate_SOL_MELANIA_1h5s)
- positions_directory: aggregated_ASSET1_ASSET2_HTFtimeLTFtime (UNIFIED)
- realtime_ASSET1_ASSET2_HTFtimeLTFtime (e.g. realtime_SOL_MELANIA_1h5s)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

def debug_print(msg):
    """Debug print with timestamp"""
    from datetime import datetime
    print(f"[CONFIG VALIDATOR {datetime.now().strftime('%H:%M:%S')}] {msg}")

class ConfigDirectoryValidator:
    """Validates and corrects directory naming conventions for unified storage"""
    
    def __init__(self):
        self.corrections_made = False
        self.validation_errors = []
        
    def parse_asset_from_symbol(self, asset_symbol: str) -> Tuple[str, str]:
        """Parse asset symbol to get individual asset names"""
        if "/" in asset_symbol:
            base, quote = asset_symbol.split("/", 1)
            return base.strip(), quote.strip()
        else:
            # Fallback for single asset symbols
            return asset_symbol.strip(), "USD"
    
    def get_timeframe_suffix(self, config: Dict[str, Any]) -> str:
        """Generate timeframe suffix from config (e.g., '1h5s', '1h1m')"""
        try:
            buffer_config = config.get("multi_timeframe_buffers", {})
            
            # Get primary and bias timeframes
            primary_tf = buffer_config.get("primary_timeframe", "1m")
            bias_tf = buffer_config.get("bias_timeframe", "1h")
            
            # Normalize timeframe strings (remove any spaces, ensure lowercase)
            primary_tf = primary_tf.strip().lower()
            bias_tf = bias_tf.strip().lower()
            
            # Create suffix: HTF + LTF (e.g., "1h" + "5s" = "1h5s")
            return f"{bias_tf}{primary_tf}"
            
        except Exception as e:
            debug_print(f"Error generating timeframe suffix: {e}")
            return "1h1m"  # Default fallback
    
    def generate_expected_directories(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Generate expected directory names based on asset and timeframes"""
        try:
            # Get asset info
            asset_symbol = config.get("asset", "SOL/MELANIA")
            base_asset, quote_asset = self.parse_asset_from_symbol(asset_symbol)
            
            # Get timeframe suffix
            tf_suffix = self.get_timeframe_suffix(config)
            
            # Generate expected directory names (UNIFIED)
            expected_dirs = {
                "snapshot_dir": f"marketstate_{base_asset}_{quote_asset}_{tf_suffix}",
                "positions_dir": f"aggregated_{base_asset}_{quote_asset}_{tf_suffix}",  # UNIFIED
                "realtime_charts_dir": f"realtime_{base_asset}_{quote_asset}_{tf_suffix}"
            }
            
            debug_print(f"Expected directories for {asset_symbol} with {tf_suffix}:")
            for key, value in expected_dirs.items():
                debug_print(f"  {key}: {value}")
            
            return expected_dirs
            
        except Exception as e:
            debug_print(f"Error generating expected directories: {e}")
            # Return safe defaults
            return {
                "snapshot_dir": "marketstate_SOL_MELANIA_1h1m",
                "positions_dir": "aggregated_SOL_MELANIA_1h1m",  # UNIFIED
                "realtime_charts_dir": "realtime_SOL_MELANIA_1h1m"
            }
    
    def validate_directory_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate directory configuration and return corrections needed"""
        
        corrections = {}
        expected_dirs = self.generate_expected_directories(config)
        
        # Check snapshot_dir
        current_snapshot_dir = config.get("snapshot_dir", "")
        expected_snapshot_dir = expected_dirs["snapshot_dir"]
        
        if current_snapshot_dir != expected_snapshot_dir:
            corrections["snapshot_dir"] = {
                "current": current_snapshot_dir,
                "expected": expected_snapshot_dir,
                "path": "snapshot_dir"
            }
            self.validation_errors.append(f"snapshot_dir: '{current_snapshot_dir}' != '{expected_snapshot_dir}'")
        
        # Check trade_simulation.storage.positions_directory (UNIFIED)
        trade_simulation = config.get("trade_simulation", {})
        storage = trade_simulation.get("storage", {})
        current_positions_dir = storage.get("positions_directory", "")
        expected_positions_dir = expected_dirs["positions_dir"]
        
        if current_positions_dir != expected_positions_dir:
            corrections["positions_directory"] = {
                "current": current_positions_dir,
                "expected": expected_positions_dir,
                "path": "trade_simulation.storage.positions_directory"
            }
            self.validation_errors.append(f"positions_directory: '{current_positions_dir}' != '{expected_positions_dir}'")
        
        # Check for deprecated legacy keys
        if "trades_directory" in storage:
            corrections["remove_trades_directory"] = {
                "current": storage["trades_directory"],
                "expected": "REMOVE (deprecated)",
                "path": "trade_simulation.storage.trades_directory"
            }
            self.validation_errors.append(f"trades_directory is deprecated, use positions_directory: '{expected_positions_dir}'")
        
        if "aggregated_positions_directory" in storage:
            corrections["remove_aggregated_positions_directory"] = {
                "current": storage["aggregated_positions_directory"],
                "expected": "REMOVE (deprecated)",
                "path": "trade_simulation.storage.aggregated_positions_directory"
            }
            self.validation_errors.append(f"aggregated_positions_directory is deprecated, use positions_directory: '{expected_positions_dir}'")
        
        # Check realtime_charts.output_dir
        realtime_charts = config.get("realtime_charts", {})
        current_charts_dir = realtime_charts.get("output_dir", "")
        expected_charts_dir = expected_dirs["realtime_charts_dir"]
        
        if current_charts_dir != expected_charts_dir:
            corrections["realtime_charts_dir"] = {
                "current": current_charts_dir,
                "expected": expected_charts_dir,
                "path": "realtime_charts.output_dir"
            }
            self.validation_errors.append(f"realtime_charts.output_dir: '{current_charts_dir}' != '{expected_charts_dir}'")
        
        # Check chart filename consistency
        chart_filename = realtime_charts.get("filename", "")
        tf_suffix = self.get_timeframe_suffix(config)
        asset_symbol = config.get("asset", "SOL/MELANIA")
        base_asset, quote_asset = self.parse_asset_from_symbol(asset_symbol)
        
        # Extract primary timeframe for filename
        primary_tf = config.get("multi_timeframe_buffers", {}).get("primary_timeframe", "1m")
        expected_filename = f"{base_asset}_{quote_asset}_live_{primary_tf}.png"
        
        if chart_filename != expected_filename:
            corrections["chart_filename"] = {
                "current": chart_filename,
                "expected": expected_filename,
                "path": "realtime_charts.filename"
            }
            self.validation_errors.append(f"chart filename: '{chart_filename}' != '{expected_filename}'")
        
        return corrections
    
    def apply_corrections(self, config: Dict[str, Any], corrections: Dict[str, Any]) -> Dict[str, Any]:
        """Apply corrections to config dictionary"""
        
        corrected_config = config.copy()
        
        # Apply snapshot_dir correction
        if "snapshot_dir" in corrections:
            corrected_config["snapshot_dir"] = corrections["snapshot_dir"]["expected"]
            debug_print(f"Corrected snapshot_dir: {corrections['snapshot_dir']['expected']}")
        
        # Apply positions_directory correction (UNIFIED)
        if "positions_directory" in corrections:
            if "trade_simulation" not in corrected_config:
                corrected_config["trade_simulation"] = {}
            if "storage" not in corrected_config["trade_simulation"]:
                corrected_config["trade_simulation"]["storage"] = {}
            
            corrected_config["trade_simulation"]["storage"]["positions_directory"] = corrections["positions_directory"]["expected"]
            debug_print(f"Corrected positions_directory: {corrections['positions_directory']['expected']}")
        
        # Remove deprecated keys
        if "remove_trades_directory" in corrections:
            if "trade_simulation" in corrected_config and "storage" in corrected_config["trade_simulation"]:
                if "trades_directory" in corrected_config["trade_simulation"]["storage"]:
                    del corrected_config["trade_simulation"]["storage"]["trades_directory"]
                    debug_print("Removed deprecated trades_directory")
        
        if "remove_aggregated_positions_directory" in corrections:
            if "trade_simulation" in corrected_config and "storage" in corrected_config["trade_simulation"]:
                if "aggregated_positions_directory" in corrected_config["trade_simulation"]["storage"]:
                    del corrected_config["trade_simulation"]["storage"]["aggregated_positions_directory"]
                    debug_print("Removed deprecated aggregated_positions_directory")
        
        # Apply realtime_charts_dir correction
        if "realtime_charts_dir" in corrections:
            if "realtime_charts" not in corrected_config:
                corrected_config["realtime_charts"] = {}
            
            corrected_config["realtime_charts"]["output_dir"] = corrections["realtime_charts_dir"]["expected"]
            debug_print(f"Corrected realtime_charts.output_dir: {corrections['realtime_charts_dir']['expected']}")
        
        # Apply chart filename correction
        if "chart_filename" in corrections:
            if "realtime_charts" not in corrected_config:
                corrected_config["realtime_charts"] = {}
            
            corrected_config["realtime_charts"]["filename"] = corrections["chart_filename"]["expected"]
            debug_print(f"Corrected chart filename: {corrections['chart_filename']['expected']}")
        
        if corrections:
            self.corrections_made = True
        
        return corrected_config
    
    def save_corrected_config(self, config: Dict[str, Any], config_path: str) -> bool:
        """Save corrected configuration to file"""
        try:
            # Create backup of original config
            backup_path = f"{config_path}.backup"
            if os.path.exists(config_path) and not os.path.exists(backup_path):
                import shutil
                shutil.copy2(config_path, backup_path)
                debug_print(f"Created backup: {backup_path}")
            
            # Save corrected config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            debug_print(f"Corrected configuration saved to: {config_path}")
            return True
            
        except Exception as e:
            debug_print(f"Error saving corrected config: {e}")
            return False
    
    def validate_and_correct_config(self, config_path: str) -> Tuple[bool, bool]:
        """
        Main validation function for unified storage structure
        
        Returns:
            (needs_correction, correction_applied)
        """
        debug_print(f"Validating UNIFIED directory configuration in: {config_path}")
        
        try:
            # Load config
            if not os.path.exists(config_path):
                debug_print(f"Config file not found: {config_path}")
                return False, False
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get asset and timeframe info for display
            asset_symbol = config.get("asset", "UNKNOWN")
            buffer_config = config.get("multi_timeframe_buffers", {})
            primary_tf = buffer_config.get("primary_timeframe", "1m")
            bias_tf = buffer_config.get("bias_timeframe", "1h")
            
            debug_print(f"Asset: {asset_symbol}")
            debug_print(f"Timeframes: HTF={bias_tf}, LTF={primary_tf}")
            
            # Validate directories
            corrections = self.validate_directory_config(config)
            
            if not corrections:
                debug_print("✓ Unified directory configuration is correct")
                return False, False
            
            # Display validation errors
            debug_print("⚠ Directory configuration errors found:")
            for error in self.validation_errors:
                debug_print(f"  {error}")
            
            debug_print(f"\n📋 Required corrections ({len(corrections)} items):")
            for key, correction in corrections.items():
                debug_print(f"  {correction['path']}:")
                debug_print(f"    Current:  '{correction['current']}'")
                debug_print(f"    Expected: '{correction['expected']}'")
            
            # Ask user for confirmation
            print(f"\n{'='*60}")
            print("CONFIG DIRECTORY VALIDATION")
            print(f"{'='*60}")
            print(f"Asset: {asset_symbol}")
            print(f"Timeframes: HTF={bias_tf}, LTF={primary_tf}")
            print(f"Config file: {config_path}")
            print(f"\nDirectory naming errors found: {len(corrections)}")
            print("\nThe configuration file needs to be updated to comply with the")
            print("unified directory naming convention:")
            print("  - marketstate_ASSET1_ASSET2_HTFtimeLTFtime")
            print("  - positions_directory: aggregated_ASSET1_ASSET2_HTFtimeLTFtime  (UNIFIED)")
            print("  - realtime_ASSET1_ASSET2_HTFtimeLTFtime")
            
            # Show specific changes needed
            print(f"\nChanges needed:")
            for key, correction in corrections.items():
                if "remove_" in key:
                    print(f"  - Remove deprecated {correction['path'].split('.')[-1]}")
                else:
                    print(f"  - {correction['path'].split('.')[-1]} should be: {correction['expected']}")
            
            response = input(f"\nUpdate config file and exit? (y/n): ").strip().lower()
            
            if response in ['y', 'yes']:
                # Apply corrections
                corrected_config = self.apply_corrections(config, corrections)
                
                # Save corrected config
                if self.save_corrected_config(corrected_config, config_path):
                    print(f"\n✓ Configuration updated successfully!")
                    print(f"✓ Backup created: {config_path}.backup")
                    print(f"✓ Unified storage structure enforced")
                    print(f"\nPlease restart the program with the corrected configuration.")
                    return True, True
                else:
                    print(f"\n⚠ Failed to save corrected configuration")
                    return True, False
            else:
                print(f"\nConfiguration not updated. Continuing with current settings...")
                return True, False
                
        except Exception as e:
            debug_print(f"Validation error: {e}")
            import traceback
            traceback.print_exc()
            return False, False

def validate_config_directories(config_path: str) -> Tuple[bool, bool]:
    """
    Convenience function for validating config directories with unified storage
    
    Returns:
        (needs_correction, correction_applied)
    """
    validator = ConfigDirectoryValidator()
    return validator.validate_and_correct_config(config_path)

# Example usage for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate unified config directory naming")
    parser.add_argument("config_path", help="Path to configuration file")
    args = parser.parse_args()
    
    needs_correction, correction_applied = validate_config_directories(args.config_path)
    
    if correction_applied:
        print("Configuration updated for unified storage. Please restart the program.")
        sys.exit(0)
    elif needs_correction:
        print("Configuration needs correction but was not updated.")
        sys.exit(1)
    else:
        print("Unified configuration is valid.")
        sys.exit(0)