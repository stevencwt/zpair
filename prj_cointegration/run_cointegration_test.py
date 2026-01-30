#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_cointegration_test.py - CLI Entry Point for Co-integration Testing

Reads pairs from config_cointegrate.json and displays results in terminal.

Usage:
    python run_cointegration_test.py
    python run_cointegration_test.py --config custom_config.json
    python run_cointegration_test.py --timeframe 1m --bars 500
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from pair_validator import PairValidator
from cointegration_tester import CointegrationResult


def print_header(pairs_count: int, timeframe: str, bars: int):
    """Print formatted header"""
    print("\n" + "═" * 70)
    print("║" + " " * 20 + "CO-INTEGRATION TEST RESULTS" + " " * 22 + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Testing {pairs_count} pairs using {timeframe} timeframe ({bars} bars)" + " " * (70 - 45 - len(str(pairs_count)) - len(timeframe) - len(str(bars))) + "║")
    print("╚" + "═" * 68 + "╝")


def print_result(result: CointegrationResult, index: int, total: int):
    """Print formatted result for a single pair"""
    
    pair_name = f"{result.asset1}/{result.asset2}"
    
    print(f"\n[{index}/{total}] Testing {pair_name}...")
    
    if result.recommendation == "INSUFFICIENT_DATA":
        print(f"❌ INSUFFICIENT DATA")
        print(f"   Samples:      {result.samples_used} (need at least 200)")
        return
    
    if result.recommendation == "ERROR":
        print(f"❌ ERROR during testing")
        return
    
    # Cointegration status
    if result.is_cointegrated:
        if result.recommendation == "EXCELLENT":
            status = "✅ COINTEGRATED - EXCELLENT for trading"
        elif result.recommendation == "GOOD":
            status = "✅ COINTEGRATED - GOOD for trading"
        elif result.recommendation == "MODERATE":
            status = "✅ COINTEGRATED - MODERATE for trading"
        else:
            status = "⚠️  COINTEGRATED - WEAK"
    else:
        status = "❌ NOT COINTEGRATED"
    
    print(f"{status}")
    
    # P-value with interpretation
    if result.p_value < 0.01:
        sig_level = "(significant at 1% level)"
    elif result.p_value < 0.05:
        sig_level = "(significant at 5% level)"
    elif result.p_value < 0.10:
        sig_level = "(marginally significant at 10% level)"
    else:
        sig_level = "(not significant)"
    
    print(f"   P-value:      {result.p_value:.4f} {sig_level}")
    print(f"   ADF Stat:     {result.adf_statistic:.3f}")
    
    # Half-life
    if result.half_life < 1000:
        half_life_minutes = result.half_life * 5  # Assuming 5m bars
        if half_life_minutes < 60:
            time_str = f"{half_life_minutes:.0f} minutes"
        else:
            time_str = f"{half_life_minutes/60:.1f} hours"
        print(f"   Half-life:    {result.half_life:.1f} bars ({time_str})")
    else:
        print(f"   Half-life:    ∞ (no mean reversion detected)")
    
    print(f"   Hedge Ratio:  {result.hedge_ratio:.4f}")
    print(f"   Samples:      {result.samples_used} bars")


def print_summary(results: list):
    """Print summary statistics"""
    
    cointegrated_count = sum(1 for r in results if r.is_cointegrated)
    excellent_count = sum(1 for r in results if r.recommendation == "EXCELLENT")
    good_count = sum(1 for r in results if r.recommendation == "GOOD")
    
    print("\n" + "═" * 70)
    print(f"SUMMARY: {cointegrated_count}/{len(results)} pairs suitable for trading")
    
    if excellent_count > 0:
        print(f"  • {excellent_count} EXCELLENT pairs (fast mean reversion)")
    if good_count > 0:
        print(f"  • {good_count} GOOD pairs (moderate mean reversion)")
    
    print("═" * 70)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print(f"Please create {config_path} with the following structure:")
        print("""
{
  "timeframe": "5m",
  "lookback_bars": 1000,
  "significance_level": 0.05,
  "pairs_to_test": [
    {"asset1": "SOL", "asset2": "MELANIA"},
    {"asset1": "SOL", "asset2": "JUP"}
  ],
  "output": {
    "save_json": true,
    "json_filename": "cointegration_results.json"
  }
}
        """)
        sys.exit(1)
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Co-integration testing for pair trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cointegration_test.py
  python run_cointegration_test.py --config custom_pairs.json
  python run_cointegration_test.py --timeframe 1m --bars 500
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config_cointegrate.json',
        help='Path to config file (default: config_cointegrate.json)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        help='Override timeframe from config (e.g., 5m, 1m)'
    )
    
    parser.add_argument(
        '--bars',
        type=int,
        help='Override lookback bars from config'
    )
    
    parser.add_argument(
        '--significance',
        type=float,
        help='Override significance level (e.g., 0.05 for 5%%)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI arguments
    timeframe = args.timeframe or config.get('timeframe', '5m')
    lookback_bars = args.bars or config.get('lookback_bars', 1000)
    significance_level = args.significance or config.get('significance_level', 0.05)
    pairs_to_test = config.get('pairs_to_test', [])
    
    if not pairs_to_test:
        print("ERROR: No pairs specified in config file")
        sys.exit(1)
    
    # Print header
    print_header(len(pairs_to_test), timeframe, lookback_bars)
    
    # Initialize validator
    validator = PairValidator(
        timeframe=timeframe,
        lookback_bars=lookback_bars,
        significance_level=significance_level,
        data_directory="."
    )
    
    # Run tests
    results = validator.validate_multiple_pairs(pairs_to_test)
    
    # Print results
    for i, result in enumerate(results, 1):
        print_result(result, i, len(results))
    
    # Print summary
    print_summary(results)
    
    # Export to JSON if configured
    output_config = config.get('output', {})
    if output_config.get('save_json', False):
        filename = output_config.get('json_filename', 'cointegration_results.json')
        validator.export_results_to_json(results, filename)
    
    print(f"\n✅ Testing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
